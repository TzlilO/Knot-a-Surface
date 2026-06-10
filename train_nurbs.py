#
# Knot a Surface — Training Script
# Equivalent of PGSR train.py for the BSpline/NURBS hybrid surface model.
#
# Key differences from PGSR:
#   1. Model is MultiSurfaceSplineModel (K parametric surfaces → Gaussians)
#      instead of a flat GaussianModel.
#   2. Densification = knot insertion/removal on the spline grid,
#      NOT clone/split of individual Gaussians.
#   3. Additional surface-specific regularizers:
#      - Eikonal loss (||Su × Sv|| ≈ 1)
#      - Normal smoothness / deviation (cosine-similarity on normal grids)
#      - Scaling deviation (keep texture scaling consistent with geometry)
#      - Local planar deviation
#   4. Optional warmup → decomposition → multi-surface training via
#      WarmupDecompositionController.
#   5. Chhugani-style adaptive tessellation (view-dependent UV interval
#      optimization) runs periodically.
#
# Copyright (C) 2023, Inria (original 3DGS/PGSR code)
# Extensions for Knot a Surface: Tzlil Ovadia
#

import os
import sys
import threading
import time
import shutil
import random
import uuid
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor, Future
from math import inf
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from arguments.nurbs_params import NurbsOptimizationParams
from gaussian_renderer import render
from optimize_nurbs import run_evaluation
from scene.app_model import AppModel
from scene.cameras import Camera
from spline_scene import SplineScene
from utils.general_utils import safe_state
from utils.graphics_utils import patch_offsets, patch_warp
from utils.image_utils import psnr
from utils.loss_utils import (
    l1_loss, ssim, lncc, get_img_grad_weight,
)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]


# Thread-safe global for best Chamfer distance
_best_cd_lock = threading.Lock()
best_cd = float("inf")


def _get_best_cd() -> float:
    with _best_cd_lock:
        return best_cd


def _set_best_cd(val: float) -> None:
    global best_cd
    with _best_cd_lock:
        best_cd = val


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(22)

# Chhugani aggregation refresh interval (iterations)
AGGREGATION_INTERVAL = 500


# ===================================================================
# Virtual Camera Generation (identical to PGSR)
# ===================================================================
def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    """Generate a perturbed virtual camera for multi-view consistency."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)

    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz),  np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3]  = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)

    virtul_cam = Camera(
        100000, Rt[:3, :3].transpose(), Rt[:3, 3],
        cam.FoVx, cam.FoVy, cam.image_width, cam.image_height,
        cam.image_path, cam.image_name, 100000,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
        preload_img=False, data_device="cuda",
    )
    return virtul_cam



# ===================================================================
# Per-View Loss Computation
# ===================================================================
def process_view(
    scene, nurbs, viewpoint_cam, pipe, background, app_model,
    opt, iteration, dataset, debug_path, device='cuda',
):
    """
    Render one view and compute ALL loss terms.
    Returns (total_loss, log_dict, render_pkg).

    Loss terms (mirrors PGSR, plus spline-specific):
      1. Photometric: L1 + λ·DSSIM
      2. Scale loss: penalise smallest Gaussian axis
      3. Single-view normal consistency (rendered vs depth-derived)
      4. Multi-view geometry consistency (reprojection error)
      5. Multi-view photometric consistency (NCC on warped patches)
      ── Spline-specific ──
      6. Eikonal: ||Su × Sv|| ≈ 1
      7. Normal smoothness / deviation on the control grid
      8. Scaling deviation
      9. Local planar deviation
    """
    total_loss = torch.tensor(0.0, device=device)
    log_dict = {}

    gt_image, gt_image_gray = viewpoint_cam.get_image()

    # Exposure compensation (after 1k iterations)
    if iteration > 1000 and opt.exposure_compensation:
        nurbs.use_app = True

    # --- Render -----------------------------------------------------------
    bg = torch.rand(3, device=device) if opt.random_background else background
    need_geo = iteration > min(opt.single_view_weight_from_iter,
                               getattr(opt, 'discrepancy_from_iter', inf))
    render_pkg = render(
        viewpoint_cam, nurbs, pipe, bg,
        app_model=app_model,
        return_plane=need_geo,
        return_depth_normal=need_geo,
    )
    image = render_pkg["render"]

    # --- 1. Photometric loss ----------------------------------------------
    ssim_loss_val = 1.0 - ssim(image, gt_image)
    img_for_l1 = render_pkg.get('app_image', image)
    l1_val = l1_loss(img_for_l1, gt_image)
    image_loss = (1.0 - opt.lambda_dssim) * l1_val + opt.lambda_dssim * ssim_loss_val
    total_loss += image_loss
    log_dict.update({"L1": l1_val.item(), "SSIM": (1.0 - ssim_loss_val).item(),
                     "Photometric": image_loss.item()})

    # --- 6. Eikonal loss (spline-specific) --------------------------------
    # if iteration >= opt.eikonal_from_iter and opt.lambda_eikonal > 0:
    #     eik = nurbs.eikonal_losses(opt.lambda_eikonal)
    #     total_loss += eik
    #     log_dict['Eikonal'] = eik.item()

    # --- 7-9. Surface regularisers (spline-specific) ----------------------
    scaling_w  = opt.scale_deviation_weight  if iteration >= opt.scale_consistency_from else 0.0
    n_smooth_w = opt.normal_smoothness_weight if iteration >= opt.normal_smooth_from else 0.0
    n_glob_w   = opt.normal_global_smoothness_weight if iteration >= opt.normal_smooth_from else 0.0
    n_dev_w    = opt.normal_dev_weight        if iteration >= opt.normal_dev_from else 0.0
    lpd_w      = opt.local_planar_deviation_weight if iteration >= opt.local_planar_deviation_from else 0.0
    #
    # if n_dev_w > 0:
    #     nd = param_surf_deviation(
    #         nurbs.global_normal_grids, geo_vecs=nurbs.normal_grids,
    #         weight_maps=nurbs.weight_map_grids(), w=n_dev_w)
    #     total_loss += nd;  log_dict['NormalDev'] = nd.item()
    #
    # if n_smooth_w > 0:
    #     ns = cossim_loss_multisurf(
    #         nurbs.normal_grids, weight_maps=nurbs.weight_map_grids(), w=n_smooth_w)
    #     total_loss += ns;  log_dict['NormalSmooth'] = ns.item()
    #
    # if n_glob_w > 0:
    #     ng = cossim_loss_multisurf(
    #         nurbs.global_normal_grids, weight_maps=nurbs.weight_map_grids(), w=n_glob_w)
    #     total_loss += ng;  log_dict['NormalGlobalSmooth'] = ng.item()
    #
    # if scaling_w > 0:
    #     sd = param_surf_deviation(
    #         nurbs.scaling_grids, geo_vecs=nurbs.geo_scaling_grids,
    #         weight_maps=nurbs.weight_map_grids(), w=scaling_w)
    #     total_loss += sd;  log_dict['ScaleDev'] = sd.item()
    #
    # if lpd_w > 0:
    #     lpd = nurbs.local_planar_deviation_loss(weight=lpd_w)
    #     total_loss += lpd;  log_dict['PlanarDev'] = lpd.item()

    # --- 2. Scale loss (same as PGSR) ------------------------------------
    visibility_filter = render_pkg["visibility_filter"]
    if opt.refine_scales and visibility_filter.sum() > 0 and opt.scale_loss_weight > 0:
        scale = nurbs.get_scaling[visibility_filter]
        sorted_scale, _ = torch.sort(scale, dim=-1)
        min_scale_loss = opt.scale_loss_weight * sorted_scale[..., 0].mean()
        total_loss += min_scale_loss
        log_dict['MinScale'] = min_scale_loss.item()

    # --- 3. Single-view normal consistency --------------------------------
    normal_loss = None
    if iteration > opt.single_view_weight_from_iter:
        weight = opt.single_view_weight
        normal = render_pkg["rendered_normal"]
        depth_normal = render_pkg["depth_normal"]

        image_weight = (1.0 - get_img_grad_weight(gt_image)).clamp(0, 1).detach() ** 2
        if not opt.wo_image_weight:
            normal_loss = weight * (image_weight * (depth_normal - normal).abs().sum(0)).mean()
        else:
            normal_loss = weight * ((depth_normal - normal).abs().sum(0)).mean()
        total_loss += normal_loss
        log_dict['SingleViewNormal'] = normal_loss.item()

    # --- 4 & 5. Multi-view consistency ------------------------------------
    geo_loss, ncc_loss = None, None
    if iteration > opt.multi_view_weight_from_iter:
        nearest_cam = None
        if len(viewpoint_cam.nearest_id) > 0:
            nearest_cam = scene.getTrainCameras()[
                random.sample(viewpoint_cam.nearest_id, 1)[0]]

        use_virtul_cam = False
        if opt.use_virtul_cam and (
            np.random.random() < opt.virtul_cam_prob or nearest_cam is None
        ):
            nearest_cam = gen_virtul_cam(
                viewpoint_cam,
                trans_noise=dataset.multi_view_max_dis,
                deg_noise=dataset.multi_view_max_angle,
            )
            use_virtul_cam = True

        if nearest_cam is not None:
            patch_size = opt.multi_view_patch_size
            sample_num = opt.multi_view_sample_num
            pixel_noise_th = opt.multi_view_pixel_noise_th
            total_patch_size = (patch_size * 2 + 1) ** 2
            ncc_weight = opt.multi_view_ncc_weight
            geo_weight = opt.multi_view_geo_weight

            # Geometry consistency
            H, W = render_pkg['plane_depth'].squeeze().shape
            ix, iy = torch.meshgrid(
                torch.arange(W), torch.arange(H), indexing='xy')
            pixels = torch.stack([ix, iy], dim=-1).float().to(device)

            nearest_render_pkg = render(
                nearest_cam, nurbs, pipe, bg,
                app_model=app_model, return_plane=True, return_depth_normal=False)

            pts = nurbs.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
            pts_in_near = (pts @ nearest_cam.world_view_transform[:3, :3]
                           + nearest_cam.world_view_transform[3, :3])
            map_z, d_mask = nurbs.get_points_depth_in_depth_map(
                nearest_cam, nearest_render_pkg['plane_depth'], pts_in_near)

            pts_in_near = pts_in_near / pts_in_near[:, 2:3]
            pts_in_near = pts_in_near * map_z.squeeze()[..., None]
            R_ = torch.tensor(nearest_cam.R).float().cuda()
            T_ = torch.tensor(nearest_cam.T).float().cuda()
            pts_ = (pts_in_near - T_) @ R_.transpose(-1, -2)
            pts_in_view = (pts_ @ viewpoint_cam.world_view_transform[:3, :3]
                           + viewpoint_cam.world_view_transform[3, :3])
            pts_proj = torch.stack([
                pts_in_view[:, 0] * viewpoint_cam.Fx / pts_in_view[:, 2] + viewpoint_cam.Cx,
                pts_in_view[:, 1] * viewpoint_cam.Fy / pts_in_view[:, 2] + viewpoint_cam.Cy,
            ], -1).float()

            pixel_noise = torch.norm(
                pts_proj - pixels.reshape(*pts_proj.shape), dim=-1)

            if not opt.wo_use_geo_occ_aware:
                d_mask = d_mask & (pixel_noise < pixel_noise_th)
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
            else:
                weights = torch.ones_like(pixel_noise)
                weights[~d_mask] = 0

            # Debug visualisation (every 200 iters)
            if iteration % 200 == 0 and normal_loss is not None:
                _save_debug_images(
                    debug_path, iteration, viewpoint_cam, gt_image, image,
                    render_pkg, weights, H, W, image_weight,
                )

            if d_mask.sum() > 0:
                geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                total_loss += geo_loss
                log_dict['GeoLoss'] = geo_loss.item()

                # NCC (photometric multi-view) — only for real neighbours
                if not use_virtul_cam:
                    ncc_loss = _compute_ncc_loss(
                        viewpoint_cam, nearest_cam, render_pkg,
                        gt_image_gray, d_mask, weights, pixels,
                        patch_size, sample_num, total_patch_size, ncc_weight,
                    )
                    if ncc_loss is not None:
                        total_loss += ncc_loss
                        log_dict['NCC'] = ncc_loss.item()

    return total_loss, log_dict, render_pkg


# ===================================================================
# NCC Helper (extracted for readability)
# ===================================================================
def _compute_ncc_loss(
    viewpoint_cam, nearest_cam, render_pkg, gt_image_gray,
    d_mask, weights, pixels, patch_size, sample_num,
    total_patch_size, ncc_weight,
):
    """Homography-based NCC multi-view photometric consistency (same as PGSR)."""
    with torch.no_grad():
        d_mask_flat = d_mask.reshape(-1)
        valid_indices = torch.arange(d_mask_flat.shape[0], device=d_mask.device)[d_mask_flat]
        if d_mask_flat.sum() > sample_num:
            idx = np.random.choice(d_mask_flat.sum().cpu().numpy(), sample_num, replace=False)
            valid_indices = valid_indices[idx]

        w = weights.reshape(-1)[valid_indices]
        pix = pixels.reshape(-1, 2)[valid_indices]
        offsets = patch_offsets(patch_size, pix.device)
        ori_pixels_patch = pix.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

        H, W = gt_image_gray.squeeze().shape
        pixels_patch = ori_pixels_patch.clone()
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
        ref_gray = F.grid_sample(
            gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
            align_corners=True).reshape(-1, total_patch_size)

        ref_to_near_r = (nearest_cam.world_view_transform[:3, :3].transpose(-1, -2)
                         @ viewpoint_cam.world_view_transform[:3, :3])
        ref_to_near_t = (-ref_to_near_r @ viewpoint_cam.world_view_transform[3, :3]
                         + nearest_cam.world_view_transform[3, :3])

    # Homography from rendered normal + distance
    ref_n = render_pkg["rendered_normal"].permute(1, 2, 0).reshape(-1, 3)[valid_indices]
    ref_d = render_pkg['rendered_distance'].squeeze().reshape(-1)[valid_indices]

    H_mat = (ref_to_near_r[None]
             - torch.matmul(
                 ref_to_near_t[None, :, None].expand(ref_d.shape[0], 3, 1),
                 ref_n[:, :, None].expand(ref_d.shape[0], 3, 1).permute(0, 2, 1),
             ) / ref_d[..., None, None])
    H_mat = torch.matmul(
        nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_d.shape[0], 3, 3),
        H_mat)
    H_mat = H_mat @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

    grid = patch_warp(H_mat.reshape(-1, 3, 3), ori_pixels_patch)
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0

    _, nearest_gray = nearest_cam.get_image()
    sampled_gray = F.grid_sample(
        nearest_gray[None], grid.reshape(1, -1, 1, 2),
        align_corners=True).reshape(-1, total_patch_size)

    ncc, ncc_mask = lncc(ref_gray, sampled_gray)
    mask = ncc_mask.reshape(-1)
    ncc_vals = (ncc.reshape(-1) * w)[mask].squeeze()

    if mask.sum() > 0:
        return ncc_weight * ncc_vals.mean()
    return None


# ===================================================================
# Debug Image Saver
# ===================================================================
def _save_debug_images(
    debug_path, iteration, viewpoint_cam, gt_image, image,
    render_pkg, weights, H, W, image_weight,
):
    """Save a tiled debug image (same layout as PGSR)."""
    gt_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255
               ).detach().cpu().numpy().astype(np.uint8)
    if 'app_image' in render_pkg:
        img_show = ((render_pkg['app_image']).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255
                    ).detach().cpu().numpy().astype(np.uint8)
    else:
        img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255
                    ).detach().cpu().numpy().astype(np.uint8)

    normal = render_pkg["rendered_normal"]
    depth_normal = render_pkg["depth_normal"]
    normal_show = (((normal + 1) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255
                   ).detach().cpu().numpy().astype(np.uint8)
    dn_show = (((depth_normal + 1) * 0.5).permute(1, 2, 0).clamp(0, 1) * 255
               ).detach().cpu().numpy().astype(np.uint8)
    dm_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8).reshape(H, W)
    dm_color = cv2.applyColorMap(dm_show, cv2.COLORMAP_JET)

    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
    di = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-20) * 255).clip(0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(di, cv2.COLORMAP_JET)

    dist = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
    disti = ((dist - dist.min()) / (dist.max() - dist.min() + 1e-20) * 255).clip(0, 255).astype(np.uint8)
    dist_color = cv2.applyColorMap(disti, cv2.COLORMAP_JET)

    iw = (image_weight.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    iw_color = cv2.applyColorMap(iw, cv2.COLORMAP_JET)

    row0 = np.concatenate([gt_show, img_show, normal_show, dist_color], axis=1)
    row1 = np.concatenate([dm_color, depth_color, dn_show, iw_color], axis=1)
    tile = np.concatenate([row0, row1], axis=0)
    cv2.imwrite(os.path.join(debug_path, f"{iteration:05d}_{viewpoint_cam.image_name}.jpg"), tile)


# ===================================================================
# Output / Logger Setup
# ===================================================================
def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv('OAR_JOB_ID') or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# ===================================================================
# Evaluation / Reporting
# ===================================================================
@torch.no_grad()
def training_report(
    tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed,
    testing_iterations, scene, renderFunc, renderArgs, app_model, nurbs,
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [
                scene.getTrainCameras()[i % len(scene.getTrainCameras())]
                for i in range(5, 30, 5)]},
        )
        for config in validation_configs:
            if not config['cameras']:
                continue
            l1_test = psnr_test = 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                out = renderFunc(
                    viewpoint, nurbs, *renderArgs, app_model=app_model,
                    return_plane=False, return_depth_normal=False)
                img = out.get('app_image', out['render']).clamp(0, 1)
                gt, _ = viewpoint.get_image()
                gt = gt.clamp(0, 1).to("cuda")
                if tb_writer and idx < 5:
                    tb_writer.add_images(
                        f"{config['name']}_view_{viewpoint.image_name}/render",
                        img[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                            gt[None], global_step=iteration)
                l1_test += l1_loss_fn(img, gt).mean().double()
                psnr_test += psnr(img, gt).mean().double()
            psnr_test /= len(config['cameras'])
            l1_test   /= len(config['cameras'])
            print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                  f"L1 {l1_test:.6f}  PSNR {psnr_test:.4f}")
            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
                tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', nurbs.total_gaussians, iteration)
        torch.cuda.empty_cache()


# ===================================================================
# Main Training Loop
# ===================================================================
def training(dataset, opt, pipe, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    device = 'cuda'
    scene_name = os.path.basename(args.model_path)
    # device = "cuda"
    dirname = dataset.model_path.split('/')[-1]
    # scene_name = os.path.basename(dataset.model_path)  # e.g., scan118

    # dirname = dataset.model_path.split('/')[-1]
    scan_id = dirname.split('n')[-1].split('_')[0]  # e.g., 118

    def log_evaluation_results(future: Future) -> None:
        """Callback for async evaluation results."""
        try:
            result = future.result()
            _set_best_cd(result["best_score"])
            if result.get("best"):
                print(result["best"])
            if result.get("metric_dict") is not None:
                log_data = {
                    "Evaluation/Chamfer Distance": result["metric_dict"],
                }
                # if args.use_wandb and wandb_queue is not None:
                #     wandb_queue.put({
                #         "data": log_data,
                #         "iteration": result["iteration"],
                #     })
            # else:
            #     print(
            #         f"[Eval Callback] Evaluation failed: "
            #         f"{result.get('logs', 'no logs')[:200]}"
            #     )
        except Exception as e:
            print(f"[Eval Callback] Error: {e}")
    # --- Backup source code -----------------------------------------------
    for pkg in ('optimize_nurbs_clean.py', 'arguments', 'gaussian_renderer',
                'scene', 'spline_scene', 'modules', 'utils'):
        src = os.path.join('.', pkg)
        dst = os.path.join(dataset.model_path, pkg)
        if os.path.isfile(src):
            os.system(f'cp {src} {dst}')
        elif os.path.isdir(src):
            os.system(f'cp -rf {src} {dst}')

    # --- Scene & Model Initialisation -------------------------------------
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    app_model = AppModel().cuda()
    app_model.train()

    # SplineScene handles: data loading → NURBS fitting → MultiSurfaceSplineModel
    scene = SplineScene(
        dataset, opt, scan_id=scene_name,
        pipe=pipe, background=background, app_model=app_model,
    )
    nurbs = scene.get_splines()#.surfaces[0]   # MultiSurfaceSplineModel / SplineModel

    # --- Checkpoint resume ------------------------------------------------
    if args.start_checkpoint:
        try:
            print(f"Loading checkpoint from {args.start_checkpoint}")
            checkpoint, loaded_iter = torch.load(args.start_checkpoint)
            first_iter = loaded_iter
            print(f"Resuming from iteration {first_iter}")
        except Exception as e:
            print(f"Checkpoint load failed: {e}")


    # --- Timing -----------------------------------------------------------
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    # --- State ------------------------------------------------------------
    viewpoint_stack = None
    ema_loss = 0.0
    ema_single = 0.0
    ema_geo = 0.0
    ema_pho = 0.0
    ema_psnr = 0.0
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    first_iter += 1
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training")

    # =====================================================================
    # Training Loop
    # ====================================================================

    for iteration in range(first_iter, opt.iterations + 1):
        # torch.cuda.synchronize()
        iter_start.record()

        # --- LR update & SH degree ---------------------------------------
        nurbs.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            nurbs.oneupSHdegree()

        # --- Camera sampling ----------------------------------------------
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam = viewpoint_stack.pop(
            random.randint(0, len(viewpoint_stack) - 1))

        # --- Forward + Loss -----------------------------------------------
        total_loss, log_dict, render_pkg = process_view(
            scene, nurbs, viewpoint_cam, pipe, background,
            app_model, opt, iteration, dataset, debug_path, device,
        )

        # --- Backward -----------------------------------------------------
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # --- EMA progress bar -----------------------------------------
            ema_loss   = 0.4 * log_dict.get('Photometric', 0) + 0.6 * ema_loss
            ema_single = (0.4 * log_dict['SingleViewNormal'] + 0.6 * ema_single
                          if 'SingleViewNormal' in log_dict else ema_single)
            ema_geo    = (0.4 * log_dict['GeoLoss'] + 0.6 * ema_geo
                          if 'GeoLoss' in log_dict else ema_geo)
            ema_pho    = (0.4 * log_dict['NCC'] + 0.6 * ema_pho
                          if 'NCC' in log_dict else ema_pho)

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss:.5f}",
                    "Single": f"{ema_single:.5f}",
                    "Geo": f"{ema_geo:.5f}",
                    "Pho": f"{ema_pho:.5f}",
                    "Pts": nurbs.total_gaussians,
                    "Params": nurbs.parameters_count,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # --- Reporting ------------------------------------------------
            training_report(
                tb_writer, iteration,
                torch.tensor(log_dict.get('L1', 0.0)),
                total_loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                args.test_iterations, scene, render, (pipe, background),
                app_model, nurbs,
            )
            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving model")
                torch.save(
                    (nurbs.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
                )
            if iteration in args.eval_iterations:# and args.include_eval:
                print(f"\n[ITER {iteration}] Staging model for evaluation.")
                temp_chk_path = os.path.join(scene.model_path, f"temp_chkpnt_eval_{iteration}.pth")
                paths = {
                    "out_base_path": args.model_path,
                    "data_base_path": args.data_base_path,
                    "dtu_eval_path": args.dtu_eval_path,
                }
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                eval_executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0]

                future = eval_executor.submit(
                    run_evaluation, iteration, scene_name, -1,
                    scan_id, gpu_id, paths, temp_chk_path, args.use_depth_normal, args.use_depth_filter, include_eval=True)#args.include_eval)
                future.add_done_callback(log_evaluation_results)

            # --- Densification (knot insertion / pruning) -----------------
            if iteration < opt.densify_until_iter:
                mask = (render_pkg["out_observe"] > 0) & render_pkg["visibility_filter"]
                nurbs.add_subdivision_stats(
                    mask,
                    render_pkg["viewspace_points"],
                    render_pkg["viewspace_points_abs"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                did_change = False
                if (iteration > opt.densify_from_iter
                        and iteration % opt.densification_interval == 1):
                    size_threshold = (opt.abs_split_radii2D_threshold
                                      if iteration > opt.opacity_reset_interval
                                      else np.inf)
                    did_change = nurbs.subdivide_and_cull(
                        max_grad=opt.densify_grad_threshold,
                        grad_abs_threshold=opt.densify_abs_grad_threshold,
                        min_opacity=opt.opacity_cull_threshold,
                        extent=scene.cameras_extent,
                        max_screen_size=size_threshold,
                        top_k_rate_subd=opt.max_k_subdiv,
                        max_prune_rate=opt.max_k_prune,
                        verbose=False,
                    )

                # Multi-view trim (visibility-based row/col pruning)
                if (opt.use_multi_view_trim
                        and iteration % 1000 == 1):
                    did_change |= nurbs.multi_view_trim_all(
                        cameras=scene.getTrainCameras(),
                        render_fn=render,
                        pipe=pipe,
                        background=background,
                        app_model=app_model,
                        min_observations=1,
                        row_threshold=0.8,
                        col_threshold=0.8,
                        top_k_rate=opt.max_k_prune_vis,
                        verbose=False,
                    )


            # --- Opacity reset --------------------------------------------
            if iteration < opt.densify_until_iter:
                if (iteration % opt.opacity_reset_interval == 0
                        or (dataset.white_background
                            and iteration == opt.densify_from_iter)):
                    nurbs.reset_opacity()

            # --- Optimizer step -------------------------------------------
            if iteration < opt.iterations:
                nurbs.optimizer.step()
                app_model.optimizer.step()
                nurbs.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)

            # --- Checkpoint -----------------------------------------------
            if iteration in args.checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (nurbs.capture(), iteration),
                    os.path.join(scene.model_path, f"chkpnt{iteration}.pth"),
                )
                app_model.save_weights(scene.model_path, iteration)

            # --- Spline-specific parameter updates ------------------------
            # nurbs.update_parameters(iteration)
            nurbs.invalidate_all_caches()
    # --- End of training --------------------------------------------------
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()
    print("\nTraining complete.")


# ===================================================================
# Entry Point
# ===================================================================
def main():
    torch.set_num_threads(8)
    parser = ArgumentParser(description="Knot a Surface — Training")
    lp = ModelParams(parser)
    op = NurbsOptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    # Iteration lists
    evals = [7_000, 9_000, 13_000, 15_000, 21_000, 25_000, 30_000]
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * i for i in range(1, 31)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=evals)
    parser.add_argument("--eval_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=evals)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--out_base_path', type=str, default=f'/sci/labs/sagieb/zlilovadia/{PROJECT_DIR}/output',
                        help="Base output directory for evaluation.")
    parser.add_argument('--data_base_path', type=str, default='/sci/labs/sagieb/zlilovadia/KnotSurface/DTU',
                        help="Path to DTU dataset for evaluation masks.")
    parser.add_argument('--dtu_eval_path', type=str,
                        default='/sci/labs/sagieb/zlilovadia/KnotSurface/datasets/DTU/SampleSet/MVSDATA',
                        help="Path to DTU evaluation data.")
    parser.add_argument('--create_base_checkpoint_at', type=int, default=-1, help="If set, train to this iteration, save a checkpoint, and exit.")
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable WandB logging")
    parser.add_argument('--wandb_project', type=str, default="NURBS-Training", help="WandB project name")
    parser.add_argument('--wandb_group', type=str, default=None, help="WandB group name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="WandB run name")


    parser.add_argument('--include_eval', action='store_true', default=False, help="Run Chamfer Distance")
    # GPU
    # parser.add_argument('--train_gpu', type=int, default=0)
    # parser.add_argument('--eval_gpu', type=int, default=0)

    # WandB (optional)
    # parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument('--wandb_project', type=str, default="Knots-Training")
    parser.add_argument('--seed', type=int, default=22)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--use_cont", action="store_true")
    parser.add_argument("--use_depth_normal", action="store_true")
    parser.add_argument('--train_gpu', type=int, default=0, help="GPU ID to use for the main training process.")
    parser.add_argument('--eval_gpu', type=int, default=0, help="GPU ID to use for asynchronous evaluation.")
    setup_seed(args.seed)
    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # torch.autograd.set_detect_anomaly(True)
    args.include_eval = True
    scan_id = os.environ.get("SCAN_ID", "")
    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, scan_id)
    args.source_path = os.path.join(args.source_path, scan_id)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), args,
    )


if __name__ == "__main__":
    main()