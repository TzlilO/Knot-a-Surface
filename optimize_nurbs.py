#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

from __future__ import annotations

# =====================================================================
# §1  Imports & Constants
# =====================================================================

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from argparse import ArgumentParser, Namespace
from concurrent.futures import Future, ProcessPoolExecutor
from multiprocessing import Process, Queue

import numpy as np
import torch
from matplotlib import pyplot as plt


# Try importing wandb; gracefully degrade if unavailable
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False

# Project name (used for output paths)
# PROJECT_DIR = "Knots"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

# How often to re-aggregate Chhugani importance maps
AGGREGATION_INTERVAL = 500

# How often to refresh camera neighborhoods post-warmup
WARM_UP_ITERATIONS = 2000

# Sentinel for terminating the WandB logger process
WAND_LOGGER_TERMINATE = "TERMINATE"

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


REPO_ROOT = pathlib.Path(__file__).resolve().parent
RUN_DIR    = pathlib.Path(os.environ.get("PGSR_RUN_DIR",
                    REPO_ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")))
SNAPSHOT   = RUN_DIR / "frozen_src"
# Detect debugger (e.g. PyCharm). Skip snapshotting/autocommit when True.
_DEBUG_MODE = sys.gettrace() is not None
show_interval = 500
# if not _DEBUG_MODE:
#     show_interval = 300
#     if not SNAPSHOT.exists():  # first time we touch this run
#         for pkg in ("model", "arguments", "utils"):  # add others if needed
#             shutil.copytree(REPO_ROOT / pkg, SNAPSHOT / pkg, dirs_exist_ok=True)
#         # Also freeze the main training script itself
#         shutil.copy2(REPO_ROOT / "optimize_nurbs.py", SNAPSHOT / "optimize_nurbs.py")
#         shutil.copy2(REPO_ROOT / "render.py", SNAPSHOT / "render.py")
#         shutil.copy2(REPO_ROOT / "spline_render.py", SNAPSHOT / "spline_render.py")
#     sys.path.insert(0, str(SNAPSHOT))  # guarantees *our* frozen code wins all imports
#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
#     if os.environ.get("PGSR_DISABLE_AUTOCOMMIT", "0") != "1":
#         safe_git_commit(f"run:{ts} on {branch}")
# else:
#     # Skip snapshotting and git auto‑commit while debugging; use live source tree instead.
#     sys.path.insert(0, str(REPO_ROOT))

from torch.nn import functional as F
from argparse import ArgumentParser, Namespace
from multiprocessing import Process, Queue
from concurrent.futures import Future, ProcessPoolExecutor
import cv2
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from arguments.nurbs_params import NurbsOptimizationParams
from gaussian_renderer import render
from scene.app_model import AppModel
from scene.cameras import Camera
from spline_scene import SplineScene
from utils.general_utils import safe_state
from utils.graphics_utils import patch_offsets, patch_warp, get_pixel_grid, get_cam_RT_cuda
from utils.image_utils import psnr
from utils.loss_utils import get_img_grad_weight, l1_loss, lncc, ssim, param_surf_deviation, cossim_loss_multisurf
import random
from utils.sh_utils import SH2RGB

best_cd = float('inf')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

AGGREGATION_INTERVAL = 500     # Re-aggregate every N iterations
AGGREGATION_MAX_VIEWS = 16     # Sample at most K views for aggregation
GLOBAL_REFINE_INTERVAL = 2000  # Full global refinement interval
WARM_UP_ITERATIONS = 2000      # Skip adaptive tessellation during warmup

setup_seed(22)
# region: Evaluation Worker
def run_evaluation(iteration, scene_name, trial_index, scan_id, eval_gpu, paths,
                   temp_checkpoint_path,
                   use_depth_normal=False,
                   use_depth_filter=False,
                   include_eval=False
                  ):
    """
    Worker function to run evaluation scripts asynchronously. It now tracks the best
    performing model and cleans up artifacts from sub-optimal runs to save disk space.

    do_cleanup – when *True* the worker deletes all non‑best model
    artefacts once evaluation is done (also on error).
    """

    do_cleanup = False
    use_depth_normal = '--use_depth_normal' if use_depth_normal else ''
    depth_filter= '--use_depth_filter' if use_depth_filter else ''

    python_bin_dir = os.path.dirname(sys.executable)
    env = os.environ.copy()
    env["PATH"] = f"{python_bin_dir}:{env['PATH']}"

    def execute_command(cmd, log_messages):
        log_messages.append(f"\n[Eval Worker] Running command:\n{cmd}")

        python_executable = sys.executable
        python_bin_dir = os.path.dirname(python_executable)
        env = os.environ.copy()
        env["PATH"] = f"{python_bin_dir}:{env['PATH']}"
        env["PYTHONPATH"] = f"{SNAPSHOT}:{env.get('PYTHONPATH', '')}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding='utf-8', env=env)

        full_stdout = []
        log_messages.append(f"[Eval Worker] --- Subprocess Output ---")
        for line in iter(proc.stdout.readline, ''):
            log_messages.append(line.strip())
            full_stdout.append(line)
        log_messages.append(f"[Eval Worker] --- End Subprocess Output ---")

        return_code = proc.wait()
        log_messages.append(f"[Eval Worker] Process finished with exit code: {return_code}")

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output="".join(full_stdout))

        return "".join(full_stdout)
    log_messages = []
    # ------------------------------------------------------------------
    # Helper : remove checkpoints / point‑clouds that are **not** best
    # ------------------------------------------------------------------
    def cleanup_non_best_artifacts(exp_dir: str, best_iter: int):
        """
        Delete every `chkpnt<iter>.pth` and every
        `point_cloud/iteration_<iter>/` directory whose <iter> ≠ best_iter.
        Keeps the disk footprint of a long run low.
        """
        # Remove outdated checkpoints
        for fname in os.listdir(exp_dir):
            if fname.startswith("chkpnt") and fname.endswith(".pth"):
                try:
                    iter_num = int(fname[len("chkpnt"):-4])   # strip 'chkpnt' and '.pth'
                except ValueError:
                    continue
                if iter_num != best_iter:
                    try:
                        os.remove(os.path.join(exp_dir, fname))
                    except FileNotFoundError:
                        pass

        # Remove stale point‑cloud directories
        pc_root = os.path.join(exp_dir, "point_cloud")
        if os.path.isdir(pc_root):
            for sub in os.listdir(pc_root):
                if sub.startswith("iteration_"):
                    try:
                        iter_num = int(sub[len("iteration_"):])
                    except ValueError:
                        continue
                    if iter_num != best_iter:
                        shutil.rmtree(os.path.join(pc_root, sub), ignore_errors=True)
    out_base_path = paths['out_base_path']
    # sys_args = f"CUDA_VISIBLE_DEVICES={eval_gpu} {python_executable}# optimize_nurbs.py --source_path {paths['source_path']} --model_path {temp_checkpoint_path} --images {paths['images']} --resolution {paths['resolution']} --white_background {paths['white_background']} --data_device {paths['data_device']} --eval {paths['eval']} --preload_img {paths['preload_img']} --ncc_scale {paths['ncc_scale']} --multi_view_num {paths['multi_view_num']} --multi_view_max_angle {paths['multi_view_max_angle']} --multi_view_min_dis {paths['multi_view_min_dis']} --multi_view_max_dis {paths['multi_view_max_dis']}"

    # if trial_index == 0:
    model_exp_dir = f"{out_base_path}"
    mesh_output_dir = os.path.join(model_exp_dir, "mesh")
    # A dedicated directory to store the artifacts of the best run
    best_artifacts_dir = os.path.join(model_exp_dir, "best_run_artifacts")
    # A file to persist the best score and iteration across evaluation calls
    best_model_info_file = os.path.join(model_exp_dir, "best_model_info.json")
    # Default best‑model record so it is always defined
    best_model_info = {'score': float('inf'), 'iteration': -1}

    try:
        # torch.cuda.set_device(eval_gpu)

        python_executable = sys.executable
        sys_args = f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')[0]} {python_executable} "
        common_args = f'--num_cluster 1 {depth_filter} --voxel_size {0.002} --max_depth {5.0} {use_depth_normal} --iteration {iteration}'

        with_rendering = "spline_render"


        render_cmd = f'{sys_args} {with_rendering}.py -m {model_exp_dir} {common_args}'
        # print(render_cmd)
        execute_command(render_cmd, log_messages)
        if include_eval:
            # 2. Run evaluation script
            current_mesh_path = os.path.join(mesh_output_dir, "tsdf_fusion_post.ply")
            if not os.path.exists(current_mesh_path):
                log_messages.append(f"Error: Mesh file not created at {current_mesh_path}. Evaluation cannot proceed.")
                if os.path.exists(mesh_output_dir): shutil.rmtree(mesh_output_dir)
                return {"logs": "\n".join(log_messages), "metric_dict": None, "iteration": iteration}
            # 3. Run evaluation script now that mesh exists
            eval_cmd = (
                f"{python_executable} scripts/eval_dtu/evaluate_single_scene.py "
                f"--input_mesh {current_mesh_path} "
                f"--scan_id {scan_id} "
                f"--mask_dir {paths['data_base_path']} --DTU {paths['dtu_eval_path']} --output_dir {model_exp_dir}"
            )
            # print(eval_cmd)
            eval_stdout = execute_command(eval_cmd, log_messages)

            #
            # # 3. Parse score from output
            chamfer_match = re.search(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", eval_stdout, re.MULTILINE)
            if chamfer_match:
                chamfer_dict = {'mean_d2s': float(chamfer_match.group(1)), 'mean_s2d': float(chamfer_match.group(2)),
                                'over_all': float(chamfer_match.group(3))}
                current_score = chamfer_dict['over_all']
                log_messages.append(f"[Eval Worker] Parsed Chamfer Metrics for iter {iteration}: {chamfer_dict}")
            # else:
            #     log_messages.append("[Eval Worker] Warning: Could not parse Chamfer metrics. Cannot compare performance.")
                # shutil.rmtree(mesh_output_dir)
                # return {"logs": "\n".join(log_messages), "metric_dict": None, "iteration": iteration}

            # 4. Read best score so far using the new JSON format
            best_model_info = {'score': best_cd, 'iteration': -1}
            if os.path.exists(best_model_info_file):
                try:
                    with open(best_model_info_file, 'r') as f:
                        best_model_info = json.load(f)
                except (json.JSONDecodeError, IOError):
                    log_messages.append(f"Warning: Could not read {best_model_info_file}. Assuming no prior best score.")

            # 5. Compare and manage artifacts
            if current_score < best_model_info['score']:
                best_info = f"+++ New Best Score Found at Iteration {iteration}! Score: {current_score:.4f} (Old best: {best_model_info['score']:.4f} at iter {best_model_info['iteration']}) +++"
                print(best_info)
                log_messages.append(
                    best_info)

                # Update best model info
                best_model_info = {'score': current_score, 'iteration': iteration}
                with open(best_model_info_file, 'w') as f:
                    json.dump(best_model_info, f, indent=4)

                # Remove old best artifacts if they exist
                if os.path.exists(best_artifacts_dir):
                    shutil.rmtree(best_artifacts_dir)

                # Promote current artifacts to be the new best
                os.rename(mesh_output_dir, best_artifacts_dir)
                log_messages.append(f"Saved new best model artifacts to: {best_artifacts_dir}")

            else:
                log_messages.append(
                    f"Score {current_score:.4f} did not improve over best of {best_model_info['score']:.4f} (from iter {best_model_info['iteration']}). Cleaning up intermediate files.")
                # shutil.rmtree(mesh_output_dir)
            best_info = f"--- Current Best Model: Iteration {best_model_info['iteration']} | Score: {best_model_info['score']:.5f} | Path: {best_artifacts_dir} ---"
            log_messages.append(best_info)
            # Keep only artefacts from the current best iteration
            if do_cleanup:
                cleanup_non_best_artifacts(model_exp_dir, best_model_info['iteration'])

            return {"logs": "\n".join(log_messages),
                    "metric_dict": chamfer_dict,
                    "iteration": iteration,
                    "best": best_info,
                    "best_score": best_model_info['score']}

        # Rendering-only path (include_eval=False): no metric to report.
        return {"logs": "\n".join(log_messages), "metric_dict": None,
                "iteration": iteration, "best": None,
                "best_score": best_model_info['score']}

    except Exception as e:
        log_messages.append(f"[Eval Worker] An unexpected error occurred: {e}")
        if os.path.exists(mesh_output_dir):
            shutil.rmtree(mesh_output_dir)
        return {"logs": "\n".join(log_messages), "metric_dict": None, "iteration": iteration, "best": None, "best_score": np.inf}

WAND_LOGGER_TERMINATE = "__WAND_TERMINATE__"


def wandb_logger_worker(queue: Queue, config):
    try:
        wandb.init(
            project=config.get("project", "NURBS"),
            name=config.get("name", None),
            group=config.get("group", None),
            config=config.get("config", {}),
            entity="sw7gynvgmn-hebrew-university-of-jerusalem",

            # entity=config.get("entity", None),
            settings=wandb.Settings(start_method='thread') #, _disable_stats=True, _disable_meta=True)
        )
        wandb.save("modules/KnotSurface.py")
        wandb.save("modules/KnotSurface.py")
        wandb.save("arguments/__init__.py")
        wandb.save("arguments/nurbs_params.py")
        wandb.save("optimize_nurbs.py")
        wandb.save("spline_render.py")

        wandb.define_metric("iteration")

        # -------------------------------------------------------------
        # Training-time metrics (higher = better)
        # -------------------------------------------------------------
        wandb.define_metric("Training/PSNR",
                            step_metric="iteration",
                            goal="maximize",
                            summary="max")

        wandb.define_metric("Training/SSIM",
                            step_metric="iteration",
                            goal="maximize",
                            summary="max")

        # Define a custom x-axis for all evaluation metrics.
        # wandb.define_metric("Evaluation/*", step_metric="iteration")
        wandb.define_metric("Training/L1",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Training/Photometric Loss",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Training/Total Loss",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Evaluation/Chamfer Distance/mean_d2s",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Evaluation/Chamfer Distance/mean_s2d",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Evaluation/Chamfer Distance/over_all",
                            step_metric="iteration",
                            goal="minimize",
                            summary="min")

        wandb.define_metric("Eval/test/PSNR", step_metric="iteration",
                            goal="maximize", summary="max")
        wandb.define_metric("Eval/train/PSNR", step_metric="iteration",
                            goal="maximize", summary="max")
        wandb.define_metric("Eval/test/L1", step_metric="iteration",
                            goal="minimize", summary="min")
        wandb.define_metric("Eval/train/L1", step_metric="iteration",
                            goal="minimize", summary="min")
        wandb.define_metric("Stats/*", step_metric="iteration")
        while True:
            item = queue.get()
            if item == WAND_LOGGER_TERMINATE:
                break
            try:
                # Items are {"data": payload, "step": iteration}; log the
                # payload itself, not the wrapper (which would nest every
                # key under "data." and log "step" as a metric).
                payload = item.get("data", item) if isinstance(item, dict) else item
                step = item.get("step", None) if isinstance(item, dict) else None
                payload = {
                    k: (v.detach().cpu().item() if isinstance(v, torch.Tensor) else v)
                    for k, v in payload.items()
                    if k != "step"
                }
                wandb.log(payload, step=step)
            except Exception as e:
                print(f"[W&B Worker] Logging failed: {e}")
    except Exception as e:
        print(f"[W&B Worker] Init failed: {e}")
    finally:
        if wandb.run:
            wandb.finish()
try:
    import wandb
    WANDB_FOUND = True
except ImportError:
    WANDB_FOUND = False



def visualize_view_distributions(model, cameras, save_dir=None, show=True):
    """
    Utility function to create nice-looking figures visualizing the view-dependent distribution
    (partition_density_u and partition_density_v) alongside the ground-truth image for each viewpoint.
    Includes 1D bar plots for u/v densities and a 2D intensity heatmap representing the joint density.

    Args:
        model: The Spline model instance with view_cache containing partition densities per camera UID.
        cameras: List of Camera instances (viewpoints), each with uid and original_image (assumed to be a tensor of shape (3, H, W)).
        save_dir: Optional directory to save figures (e.g., 'views_vis/'). If None, doesn't save.
        show: If True, displays the figure using plt.show().

    Returns:
        List of matplotlib Figure objects, one per camera.
    """
    figures = []
    for cam in cameras:
        uid = cam.uid
        if uid not in model.view_cache:
            print(f"Skipping camera {uid}: No view_cache entry found.")
            continue

        cache = model.view_cache[uid]
        density_u = cache.get('partition_density_u', torch.zeros(model.coarse_res_u, device=model.device)).cpu().numpy()
        density_v = cache.get('partition_density_v', torch.zeros(model.coarse_res_v, device=model.device)).cpu().numpy()

        # Normalize densities for visualization
        density_u = density_u / (density_u.max() + 1e-6)
        density_v = density_v / (density_v.max() + 1e-6)

        # Compute 2D joint density as outer product (intensity map)
        density_2d = np.outer(density_u, density_v)  # (len_u, len_v)

        # Get GT image (assume (3, H, W) tensor, convert to (H, W, 3) numpy)
        gt_image = cam.original_image.permute(1, 2, 0).cpu().numpy() if hasattr(cam, 'original_image') else np.zeros((100, 100, 3))

        # Create figure with subplots: GT on left, u/v bars in middle, 2D heatmap on right
        fig, axs = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={'width_ratios': [2, 1, 1, 1.5]})
        fig.suptitle(f"View-Dependent Distribution for Camera UID: {uid}", fontsize=16)

        # GT Image
        axs[0].imshow(gt_image)
        axs[0].set_title("Ground-Truth Image")
        axs[0].axis('off')

        # U Density Bar Plot
        axs[1].bar(np.arange(len(density_u)), density_u, color='skyblue')
        axs[1].set_title("U Partition Density")
        axs[1].set_xlabel("U Bins")
        axs[1].set_ylabel("Normalized Density")
        axs[1].set_ylim(0, 1)
        axs[1].grid(True, linestyle='--', alpha=0.5)

        # V Density Bar Plot
        axs[2].bar(np.arange(len(density_v)), density_v, color='lightgreen')
        axs[2].set_title("V Partition Density")
        axs[2].set_xlabel("V Bins")
        axs[2].set_ylabel("Normalized Density")
        axs[2].set_ylim(0, 1)
        axs[2].grid(True, linestyle='--', alpha=0.5)

        # 2D Intensity Heatmap
        im = axs[3].imshow(density_2d, cmap='viridis', aspect='auto', origin='lower')
        axs[3].set_title("2D Joint Density Intensity")
        axs[3].set_xlabel("V Bins")
        axs[3].set_ylabel("U Bins")
        fig.colorbar(im, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        axs[3].grid(False)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/view_{uid}_distribution.png", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figures.append(fig)

    return figures


def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
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
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                        preload_img=False, data_device="cuda")
    return virtul_cam


def process_batch_views(
        scene,
        nurbs,
        cameras_batch: List,
        pipe,
        background,
        app_model,
        opt,
        iteration,
        dataset,
        debug_path,
        device='cuda'
) -> Tuple[torch.Tensor, Dict, Dict]:
    """
    Process a batch of views with gradient accumulation.

    Args:
        cameras_batch: List of Camera objects to process in this batch

    Returns:
        (total_loss, aggregated_log_dict)
    """
    batch_size = len(cameras_batch)
    total_loss = torch.tensor(0.0, device=device)
    log_dict = {}

    for view_idx, viewpoint_cam in enumerate(cameras_batch):
        is_last_view = (view_idx == batch_size - 1)

        # Process this view (your existing logic)
        view_loss, view_log_dict, render_pkg = process_view(
            scene, nurbs, viewpoint_cam, pipe, background,
            app_model, opt, iteration, dataset, debug_path
        )

        # Accumulate loss (average across batch)
        total_loss = total_loss + view_loss / batch_size

        # Aggregate metrics
        for k, v in view_log_dict.items():
            log_dict[k] = log_dict.get(k, 0) + v / batch_size

        # Backward with graph retention for all but last view.
        # Scale by 1/B so accumulated gradients are the batch MEAN: without
        # this the effective learning rate grows linearly with batch size.
        retain_graph = not is_last_view
        (view_loss / batch_size).backward(retain_graph=retain_graph)

        # Optional: Clear intermediate tensors to save memory
        if not is_last_view:
            del render_pkg, view_loss
            # torch.cuda.empty_cache()
        # else:
            # log_dict['render_pkg'] = render_pkg
    return total_loss, log_dict, render_pkg
def process_view(scene, nurbs, viewpoint_cam, pipe, background, app_model, opt, iteration, dataset, debug_path, device='cuda'):

    """
    Handles rendering and loss calculation for a single viewpoint.
    """
    skip = True
    total_loss = torch.tensor(0.0, device=device)
    log_dict = {}

    gt_image, gt_image_gray = viewpoint_cam.get_image()
    if iteration > 1000 and opt.exposure_compensation:
        nurbs.use_app = True

    bg = torch.rand((3), device=device) if opt.random_background else background

    render_pkg = render(viewpoint_cam, nurbs, pipe, bg, app_model=app_model, return_plane=iteration > min(opt.single_view_weight_from_iter, opt.discrepancy_from_iter) ,
                        return_depth_normal = iteration > min(opt.single_view_weight_from_iter, opt.discrepancy_from_iter))
    image = render_pkg["render"]
    ssim_loss_val = (1.0 - ssim(image, gt_image))
    img_for_l1 = render_pkg.get('app_image', image)
    l1_loss_val = l1_loss(img_for_l1, gt_image)
    image_loss = (1.0 - opt.lambda_dssim) * l1_loss_val + opt.lambda_dssim * ssim_loss_val
    total_loss += image_loss
    log_dict.update({"L1": l1_loss_val, "SSIM": 1.0 - ssim_loss_val, "Photometric Loss": image_loss})
    scale_loss_weight = opt.scale_loss_weight #if iteration >= opt.refine_scaling_from else 0.0

    if iteration >= opt.eikonal_from_iter and opt.lambda_eikonal > 0:# and False:  # e.g., add to config as 1000
        eik_loss = nurbs.eikonal_losses(opt.lambda_eikonal)  # Add lambda_eikonal to config, e.g., 0.5
        total_loss += eik_loss
        log_dict['Eikonal Loss'] = eik_loss.item()


    if not skip:
        scaling_weight = opt.scale_deviation_weight if iteration >= opt.scale_consistency_from else 0.0
        normal_global_smoothness_weight = opt.normal_global_smoothness_weight if iteration >= opt.normal_smooth_from else 0.0
        normal_smoothness_weight = opt.normal_smoothness_weight if iteration >= opt.normal_smooth_from else 0.0
        normal_deviation_weight = opt.normal_dev_weight if iteration >= opt.normal_dev_from else 0.0

        if normal_deviation_weight > 0.0:
            normal_deviation = param_surf_deviation(nurbs.global_normal_grids, geo_vecs=nurbs.normal_grids, weight_maps=nurbs.weight_map_grids(), w=normal_deviation_weight)
            total_loss = total_loss + normal_deviation
            log_dict['Normal Deviation Loss'] = normal_deviation.item()
        if normal_smoothness_weight > 0.0:
            normal_smoothness = cossim_loss_multisurf(nurbs.normal_grids, weight_maps=nurbs.weight_map_grids(), w=normal_smoothness_weight)
            total_loss = total_loss + normal_smoothness
            log_dict['Normal Smoothness Loss'] = normal_smoothness.item()

        if normal_global_smoothness_weight > 0.0:
            global_normal_smoothness = cossim_loss_multisurf(nurbs.global_normal_grids, weight_maps=nurbs.weight_map_grids(), w=normal_global_smoothness_weight)
            total_loss = total_loss + global_normal_smoothness
            log_dict['Global Normal Consistency Loss'] = global_normal_smoothness.item()

        if scaling_weight > 0.0:
            scale_deviation = param_surf_deviation(nurbs.scaling_grids, geo_vecs=nurbs.geo_scaling_grids, weight_maps=nurbs.weight_map_grids(), w=scaling_weight)
            total_loss = total_loss + scale_deviation
            log_dict['Scaling Deviation Loss'] = scale_deviation.item()

        local_planar_deviation_weight = opt.local_planar_deviation_weight if iteration >= opt.local_planar_deviation_from else 0.0
        if iteration >= opt.local_planar_deviation_from:
            surf_deviation = nurbs.local_planar_deviation_loss(weight=local_planar_deviation_weight)
            total_loss = total_loss + surf_deviation
            log_dict['Surface Deviation Loss'] = surf_deviation.item()


    visibility_filter = render_pkg["visibility_filter"]
    if opt.refine_scales and visibility_filter.sum() > 0 and scale_loss_weight > 0.:
        scale = nurbs.get_scaling[visibility_filter]

        sorted_scale, _ = torch.sort(scale, dim=-1)
        min_scale_loss = sorted_scale[..., 0]
        scale_loss = opt.scale_loss_weight * (min_scale_loss.mean())
        total_loss += scale_loss
        log_dict['Min. Scale Loss'] = scale_loss.item()
    if iteration > opt.single_view_weight_from_iter:
        weight = opt.single_view_weight
        normal = render_pkg["rendered_normal"]
        depth_normal = render_pkg["depth_normal"]

        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = (image_weight).clamp(0, 1).detach() ** 2
        if not opt.wo_image_weight:
            # image_weight = erode(image_weight[None,None]).squeeze()
            normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
        else:
            normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()

        total_loss += normal_loss
        log_dict["Single-View Normal Loss"] = normal_loss.item()

        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[
                random.sample(viewpoint_cam.nearest_id, 1)[0]]
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis,
                                             deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True

            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                pixels = get_pixel_grid(W, H, render_pkg['plane_depth'].device)
                nearest_render_pkg = render(nearest_cam, nurbs, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = nurbs.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[
                                                                                      3, :3]
                map_z, d_mask = nurbs.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'],
                                                                        pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                R, T = get_cam_RT_cuda(nearest_cam)
                pts_ = (pts_in_nearest_cam - T) @ R.transpose(-1, -2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,
                                         :3] + viewpoint_cam.world_view_transform[3, :3]
                pts_projections = torch.stack(
                    [pts_in_view_cam[:, 0] * viewpoint_cam.Fx / pts_in_view_cam[:, 2] + viewpoint_cam.Cx,
                     pts_in_view_cam[:, 1] * viewpoint_cam.Fy / pts_in_view_cam[:, 2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                if not opt.wo_use_geo_occ_aware:
                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                    gt_img_show = ((gt_image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                   [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1, 2, 0).clamp(0, 1)[:, :,
                                    [2, 1, 0]] * 255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                 1) * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    depth_normal_show = (((depth_normal + 1.0) * 0.5).permute(1, 2, 0).clamp(0,
                                                                                             1) * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    d_mask_show = (weights.float() * 255).detach().cpu().numpy().astype(np.uint8).reshape(H, W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color],
                                          axis=1)
                    image_to_show = np.concatenate([row0, row1], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"),
                                image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    total_loss += geo_loss
                    log_dict["Geometric Loss"] = geo_loss.item()
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace=False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1, 2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()

                            H, W = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2),
                                                         align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3, :3].transpose(-1,
                                                                                                   -2) @ viewpoint_cam.world_view_transform[
                                                                                                         :3, :3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,
                                                                     :3] + nearest_cam.world_view_transform[3, :3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1, 2, 0)
                        ref_local_n = ref_local_n.reshape(-1, 3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                                            torch.matmul(
                                                ref_to_neareast_t[None, :, None].expand(ref_local_d.shape[0], 3, 1),
                                                ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2,
                                                                                                                   1)) / \
                                            ref_local_d[..., None, None]
                        H_ref_to_neareast = torch.matmul(
                            nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3),
                            H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)

                        grid = patch_warp(H_ref_to_neareast.reshape(-1, 3, 3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2),
                                                         align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            total_loss += ncc_loss
                            log_dict['NCC Loss'] = ncc_loss.item()
    return total_loss, log_dict, render_pkg


def async_wandb_logger(queue, img_dict, run_dict, iteration):
    try:
        if img_dict is not None:
            safe_img_dict = {k: v.detach().cpu().numpy() if torch.is_tensor(v) else v for k, v in img_dict.items()}

        if run_dict is not None:
            # Note: no `v > 0` filter — it raised on multi-element tensors
            # and silently dropped legitimately zero-valued losses.
            safe_run_dict = {k: v.detach().item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v
                             for k, v in run_dict.items()}

        safe_run_dict['iteration'] = iteration
        queue.put_nowait({"data": {**safe_img_dict, **safe_run_dict}, "step": iteration})
    except Exception as e:
        print(f"[W&B Queue] Logging enqueue failed: {e}")



def log_metrics(iteration, metrics_dict):
    """Log scalar metrics to WandB."""
    if not WANDB_FOUND or wandb.run is None:
        return

    log_data = {"iteration": iteration}
    log_data.update(metrics_dict)
    wandb.log(log_data, step=iteration)


def depth_to_colormap(depth_tensor, colormap=cv2.COLORMAP_JET):
    """Convert depth tensor to colormap image."""
    depth = depth_tensor.squeeze().detach().cpu().numpy()
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, colormap)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return depth_color / 255.0


def normal_to_rgb(normal_tensor):
    """Convert normal tensor [-1,1] to RGB [0,1]."""
    normal = normal_tensor.detach().cpu()
    if normal.dim() == 3 and normal.shape[0] == 3:
        normal = normal.permute(1, 2, 0)
    return ((normal.numpy() + 1) / 2).clip(0, 1)


def prepare_img_log(iteration, images_dict):
    """Log images to WandB."""

    log_data = {"iteration": iteration}
    for key, img in images_dict.items():
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy, handle different formats
            if img.dim() == 3:
                if img.shape[0] in [1, 3, 4]:  # CHW format
                    img = img.permute(1, 2, 0)
                img = img.detach().cpu().numpy()
            elif img.dim() == 2:  # HW format (grayscale/depth)
                img = img.detach().cpu().numpy()

        # Ensure proper range [0, 1] or [0, 255]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = np.clip(img, 0, 1)

        log_data[key] = wandb.Image(img, caption=key)
    return log_data

@torch.no_grad()
def log_qualitative_results(
    iteration: int,
    nurbs,
    scene: SplineScene,
    pipe,
    background: torch.Tensor,
    app_model,
    wandb_queue: Optional[Queue],
    scene_name: Optional[str] = None,
) -> None:
    """Render fixed views and log to WandB."""
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    validation_configs = [
        {"name": "test", "cameras": test_cameras},
        {
            "name": "train",
            "cameras": [
                train_cameras[idx % len(train_cameras)]
                for idx in range(5, 30, 5)
            ],
        },
    ]

    for config in validation_configs:
        if not config["cameras"]:
            continue

        l1_test = 0.0
        psnr_test = 0.0
        log_images = {}
        for viewpoint in config["cameras"]:
            render_pkg = render(
                viewpoint, nurbs, pipe, background,
                app_model=app_model,
                return_plane=False,
                return_depth_normal=False,
            )
            gt_img = viewpoint.get_image()[0]
            psnr_val = psnr(render_pkg["render"].detach(), gt_img).mean()#.double()
            l1_test += l1_loss(render_pkg["render"].detach(), gt_img).mean()#.double()
            psnr_test += psnr_val

            if WANDB_FOUND and wandb_queue is not None:
                gt_np = gt_img.detach().cpu().numpy().transpose(1, 2, 0)
                render_np = (
                    render_pkg["render"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
                )
                key_base = (
                    f"Qualitative/{config['name']}/View_{viewpoint.image_name}"
                )
                log_images[f"{key_base}/Ground_Truth"] = wandb.Image(gt_np)
                log_images[f"{key_base}/Render"] = wandb.Image(render_np)
            # nurbs_prop_dict=nurbs.prepare_grid_for_vis(viewpoint)
            # plot_render_outputs(
            #     render_pkg, nurbs_prop_dict,
            #     viewpoint.get_image()[0],
            #     nurbs, uid=viewpoint.uid,
            #     title=f"Report for scan{scene_name} after {iteration} Iterations"
            # )
        n = len(config["cameras"])
        psnr_test /= n
        l1_test /= n
        print(
            f"\n[ITER {iteration}] Evaluating {config['name']}: "
            f"L1 {l1_test:.4f}  PSNR {psnr_test:.2f}"
        )

        if wandb_queue is not None:
            log_images[f"Eval/{config['name']}/PSNR"] = float(psnr_test)
            log_images[f"Eval/{config['name']}/L1"] = float(l1_test)
            wandb_queue.put({"data": log_images, "step": iteration})


@torch.no_grad()
def log_model_statistics(nurbs) -> dict:
    """Scalar statistics on parameters and gaussians for wandb."""
    stats = {
        "Stats/num_gaussians": int(nurbs.total_gaussians),
        "Stats/num_parameters": int(nurbs.parameters_count),
        "Stats/num_surfaces": int(nurbs.num_surfaces),
    }
    try:
        scaling = nurbs.get_scaling
        stats["Stats/scale_mean"] = scaling.mean().item()
        stats["Stats/scale_max"] = scaling.max().item()
        stats["Stats/scale_p95"] = torch.quantile(
            scaling.reshape(-1)[:1_000_000], 0.95
        ).item()

        opacity = nurbs.get_opacity
        stats["Stats/opacity_mean"] = opacity.mean().item()
        stats["Stats/opacity_frac_solid"] = (opacity > 0.5).float().mean().item()

        xyz = nurbs.get_xyz
        stats["Stats/xyz_extent"] = (
            (xyz.max(dim=0).values - xyz.min(dim=0).values).norm().item()
        )
    except Exception as e:
        print(f"[Stats] Gaussian stats skipped: {e}")

    for i, surface in enumerate(getattr(nurbs, "surfaces", [])):
        try:
            stats[f"Stats/surf{i}/ctrl_H"] = int(surface.state.H)
            stats[f"Stats/surf{i}/ctrl_W"] = int(surface.state.W)
            stats[f"Stats/surf{i}/samples_Us"] = int(surface.state.Us)
            stats[f"Stats/surf{i}/samples_Vs"] = int(surface.state.Vs)
            if surface.weights is not None:
                w = surface.weights.features
                stats[f"Stats/surf{i}/weight_mean"] = w.mean().item()
                stats[f"Stats/surf{i}/weight_std"] = w.std().item()
        except Exception:
            pass
    return stats





def training(dataset, opt, pipe, args, **kwargs) -> None:
    """Main training loop."""
    # ── Setup ─────────────────────────────────────────────────────────
    device = "cuda"
    dirname = dataset.model_path.split('/')[-1]
    scene_name = os.path.basename(dataset.model_path)  # e.g., scan118
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    app_model = AppModel()

    # dirname = dataset.model_path.split('/')[-1]
    scan_id = dirname.split('n')[-1].split('_')[0]  # e.g., 118
    scene_name = os.path.basename(dataset.model_path)  # e.g., scan118

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    app_model = AppModel().cuda()
    app_model.train()

    if (scene := kwargs.get("scene")) is None:
        if len(dirname.split("_")) <= 1:# and not args.include_eval:
            trial_index = prepare_output_and_logger(dataset)
        # Import and run the standard training loop
        scene = SplineScene(
            dataset, opt, scan_id=kwargs.get("scene_name", "0"),
            pipe=pipe, background=background, app_model=app_model,
        )
    else:
        trial_index = scene_name.split("_")[-1] if len(scene_name.split("_")) > 1 else "0"
    args.model_path = dataset.model_path
    args.source_path = dataset.source_path
    if (nurbs := kwargs.get("nurbs")) is None:
        nurbs = scene.get_splines()


    # ── Checkpoint resume ─────────────────────────────────────────────
    first_iter = 0
    if args.start_checkpoint:
        try:
            print(f"Loading checkpoint from {args.start_checkpoint}")
            _checkpoint, loaded_iter = torch.load(args.start_checkpoint)
            nurbs.restore(_checkpoint, train_model=True)
            first_iter = loaded_iter
            print(f"Resuming training from iteration {first_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    first_iter += 1

    # ── Logging and async executor ────────────────────────────────────
    terminal_width = shutil.get_terminal_size().columns * 3
    progress_bar = tqdm(
        range(first_iter, opt.iterations + 1),
        desc="Training progress",
        ncols=terminal_width,
    )

    wandb_queue: Optional[Queue] = None
    wandb_proc = None
    if args.use_wandb and WANDB_FOUND:
        wandb_queue = Queue(maxsize=100)
        wandb_config = {
            "project": args.wandb_project,
            "name": f"{scene_name}",
            "group": scene_name,
            "config": vars(opt),
            "entity": "Tzlil",
        }
        wandb_proc = Process(
            target=wandb_logger_worker,
            args=(wandb_queue, wandb_config),
            daemon=True,
        )
        wandb_proc.start()

    def log_evaluation_results(future: Future) -> None:
        """Callback for async evaluation results."""
        try:
            result = future.result()
            if result is None:
                print("[Eval Callback] Worker returned no result.")
                return
            if result.get("best_score") is not None and np.isfinite(result["best_score"]):
                _set_best_cd(result["best_score"])
            if result.get("best"):
                print(result["best"])
            if result.get("metric_dict") is not None:
                log_data = {
                    "Evaluation/Chamfer Distance": result["metric_dict"],
                }
                if args.use_wandb and wandb_queue is not None:
                    wandb_queue.put({
                        "data": log_data,
                        "iteration": result["iteration"],
                    })
            else:
                print(
                    f"[Eval Callback] Evaluation failed: "
                    f"{result.get('logs', 'no logs')[:200]}"
                )
        except Exception as e:
            print(f"[Eval Callback] Error: {e}")

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    eval_executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)

    # ── Training state ────────────────────────────────────────────────
    BATCH_SIZE = opt.batch_size
    USE_BATCHED_TRAINING = BATCH_SIZE > 1
    debug_path = os.path.join(scene.model_path, "debug")
    viewpoint_stack = None
    # batched_optimizer = setup_batched_optimizer(
    #     nurbs, scene.getTrainCameras().copy()
    # )

    next_camera = 0
    cycles_complete = 0
    ema_psnr_for_log = 0.0
    show_interval = 500
    render_components = None
    decomposed_final_img = None

    # ══════════════════════════════════════════════════════════════════
    # MAIN TRAINING LOOP
    # ══════════════════════════════════════════════════════════════════
    for iteration in range(first_iter, opt.iterations + 1):

        if iteration % 1000 == 0 and iteration > 0:
            nurbs.oneupSHdegree()

        # ── 2. Camera selection (shuffled per epoch) ──────────────────
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            random.shuffle(viewpoint_stack)
            cycles_complete += 1

        # ── 3. LR schedule + forward pass + loss ──────────────────────
        learning_rates = nurbs.update_learning_rate(iteration)

        if USE_BATCHED_TRAINING:
            batch_cameras = []
            for _ in range(min(BATCH_SIZE, len(viewpoint_stack))):
                batch_cameras.append(viewpoint_stack.pop())
            total_loss, log_dict, render_pkg = process_batch_views(
                scene, nurbs, batch_cameras, pipe, background,
                app_model, opt, iteration, dataset, debug_path,
            )
            viewpoint_cam = batch_cameras[-1]
        else:
            viewpoint_cam = viewpoint_stack.pop()
            total_loss, log_dict, render_pkg = process_view(
                scene, nurbs, viewpoint_cam, pipe, background,
                app_model, opt, iteration, dataset, debug_path,
            )
            total_loss.backward()

        # ── 4. Optimizer step ─────────────────────────────────────────
        if iteration <= opt.iterations:
            nurbs.optimizer.step()
            app_model.optimizer.step()
            nurbs.optimizer.zero_grad(set_to_none=True)
            app_model.optimizer.zero_grad(set_to_none=True)


        # ── 6. Logging & progress bar ─────────────────────────────────
        with torch.no_grad():
            log_dict["Total Loss"] = total_loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{log_dict['Total Loss']:.4f}",
                    "Points": nurbs.total_gaussians,
                    "CD": f"{_get_best_cd():.5f}",
                    "Params": nurbs.parameters_count,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # WandB scalar logging
            if iteration % 2000 == 1 and args.use_wandb and wandb_queue is not None:
                psnr_val = psnr(
                    render_pkg["render"], viewpoint_cam.get_image()[0]
                ).mean()
                ema_psnr_for_log = (
                    0.9 * ema_psnr_for_log + 0.1 * psnr_val.item()
                    if ema_psnr_for_log > 0 else psnr_val.item()
                )
                training_log = {
                    "iteration": iteration,
                    "Training/Total Loss": log_dict["Total Loss"],
                    "Training/PSNR": psnr_val.item(),
                }
                for k, v in log_dict.items():
                    training_log[f"Training/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
                wandb_queue.put({"data": training_log, "step": iteration})


            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving checkpoint")
                torch.save(
                    (nurbs.capture(), iteration),
                    os.path.join(
                        scene.model_path, f"chkpnt{iteration}.pth"
                    ),
                )

            if iteration in args.evaluation_iterations:
                print(f"\n[ITER {iteration}] Staging model for evaluation.")
                torch.cuda.synchronize()
                temp_chk_path = os.path.join(scene.model_path, f"temp_chkpnt_eval_{iteration}.pth")
                paths = {
                    "out_base_path": args.model_path,
                    "data_base_path": args.data_base_path,
                    "dtu_eval_path": args.dtu_eval_path,
                }
                future = eval_executor.submit(
                    run_evaluation, iteration, scene_name, trial_index,
                    scan_id, args.eval_gpu, paths, temp_chk_path, args.use_depth_normal, args.use_depth_filter, include_eval=args.include_eval)
                future.add_done_callback(log_evaluation_results)

            # ── 5. Held-out (test) evaluation — every 10% of the run ─────
            #      Renders test views (train views as fallback) and logs
            #      rendered images, L1/PSNR, and parameter/gaussian stats.
            eval_every = max(1, opt.iterations // 50)
            if iteration % eval_every == 0 or iteration == opt.iterations:
                log_qualitative_results(
                    iteration, nurbs, scene, pipe, background,
                    app_model, wandb_queue
                )
                if args.use_wandb and wandb_queue is not None:
                    wandb_queue.put({
                        "data": log_model_statistics(nurbs),
                        "step": iteration,
                    })
            # ── 11. Visualization (interactive only while debugging;
            #        otherwise dump to file so training never blocks) ────
            if getattr(args, 'show_plots', False) and iteration % show_interval == 1:
                try:
                    from utils.image_utils import plot_render_outputs
                    to_render = dict(render_pkg)
                    to_render["gt_image"] = viewpoint_cam.get_image()[0]
                    if render_components is not None:
                        to_render.update(render_components)
                    if decomposed_final_img is not None:
                        to_render["decomposed_final_img"] = decomposed_final_img
                    nurbs_prop_dict = nurbs.prepare_grid_for_vis(viewpoint_cam)
                    os.makedirs(debug_path, exist_ok=True)
                    plot_render_outputs(
                        to_render, nurbs_prop_dict,
                        viewpoint_cam.get_image()[0],
                        nurbs, uid=viewpoint_cam.uid,
                        title=f"{scene_name}_{trial_index}",
                        save_path=os.path.join(debug_path, f"diag_{iteration:06d}.png"),
                        show=_DEBUG_MODE,
                    )
                    del to_render, nurbs_prop_dict
                except Exception as e:
                    print(f"[Visualization] Skipped due to error: {e}")
                # ── 7. Densification & pruning ────────────────────────────────
            if iteration < opt.densify_until_iter:
                mask = (
                    (render_pkg["out_observe"] > 0)
                    & render_pkg["visibility_filter"]
                )
                nurbs.add_subdivision_stats(
                    mask,
                    render_pkg["viewspace_points"],
                    render_pkg["viewspace_points_abs"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )

                # Compute size_threshold *before* it's needed
                size_threshold = (
                    opt.abs_split_radii2D_threshold
                    if iteration > opt.opacity_reset_interval
                    else np.inf
                )

                did_change = False
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 1
                ):
                    did_change = nurbs.subdivide_and_cull(
                        max_grad=opt.densify_grad_threshold,
                        grad_abs_threshold=opt.densify_grad_threshold,
                        min_opacity=opt.opacity_cull_threshold,
                        extent=scene.cameras_extent,
                        max_screen_size=size_threshold,
                        top_k_rate_subd=opt.max_k_subdiv,
                        max_prune_rate=opt.max_k_prune,
                        verbose=False,
                    )

                if (
                    opt.use_multi_view_trim
                    and iteration % 1000 == 1
                ):
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

                if (
                    iteration % opt.opacity_reset_interval == 0
                    or (
                        dataset.white_background
                        and iteration == opt.densify_from_iter
                    )
                ):
                    nurbs.reset_opacity()

            # ── 8. Model state updates ────────────────────────────────────
            nurbs.update_parameters(iteration)
            nurbs._invalidate_cache()


    # ══════════════════════════════════════════════════════════════════
    # End of training — cleanup
    # ══════════════════════════════════════════════════════════════════
    print("Shutting down evaluation worker pool...")
    eval_executor.shutdown(wait=True)
    print("Shutting down WandB logger...")
    if args.use_wandb and wandb_queue is not None:
        wandb_queue.put(WAND_LOGGER_TERMINATE)
        if wandb_proc is not None:
            wandb_proc.join(timeout=30)
    return scene

# region: Setup and Main Execution
def prepare_output_and_logger(args):
    base_path = args.model_path
    trial_index = 0
    # Append trial index if the base path already exists
    final_path = f"{base_path}"
    while os.path.isdir(final_path):
        trial_index += 1
        final_path = f"{base_path}_{trial_index}"

    # Update args with the final, unique path
    args.model_path = final_path
    os.makedirs(args.model_path, exist_ok=True)
    print("Output folder:", args.model_path)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    return trial_index

def main():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = NurbsOptimizationParams(parser)
    pp = PipelineParams(parser)
    # Standard arguments
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # first_iter = 1000
    first_iter = op.multi_view_weight_from_iter
    cycle = 1
    first_iter = first_iter // cycle
    # first_iter = 1
    eval_interval = 4000 // cycle
    test_iterations = list(range(0, 30000, 1000))
    # test_iterations = [30000]
    save_iterations = list(range(first_iter, 30000//cycle, eval_interval))
    evals = [9_000, 13_000, 15_000, 21_000, 25_000, 30_000]
    # Independent copies: argparse stores the default object itself, so a
    # shared list would alias all three arguments (and a later append to one
    # would mutate the others).
    evaluation_iterations = list(evals)
    checkpoint_iterations = list(evals)
    save_iterations = list(evals)

    parser.add_argument("--evaluation_iterations", nargs="+", type=int, default=evaluation_iterations)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=save_iterations)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=checkpoint_iterations)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument('--train_gpu', type=int, default=0, help="GPU ID to use for the main training process.")
    parser.add_argument('--eval_gpu', type=int, default=0, help="GPU ID to use for asynchronous evaluation.")
    parser.add_argument('--out_base_path', type=str, default=f'/sci/labs/sagieb/zlilovadia/{PROJECT_DIR}/output',
                        help="Base output directory for evaluation.")
    parser.add_argument('--data_base_path', type=str, default='/sci/labs/sagieb/zlilovadia/KnotSurface/DTU',
                        help="Path to DTU dataset for evaluation masks.")
    parser.add_argument('--dtu_eval_path', type=str,
                        default='/sci/labs/sagieb/zlilovadia/KnotSurface/datasets/DTU/SampleSet/MVSDATA',
                        help="Path to DTU evaluation data.")
    parser.add_argument('--create_base_checkpoint_at', type=int, default=-1, help="If set, train to this iteration, save a checkpoint, and exit.")
    parser.add_argument('--use_wandb', action='store_true', default=False, help="Enable WandB logging")
    parser.add_argument('--wandb_project', type=str, default="NURBS", help="WandB project name")
    parser.add_argument('--wandb_group', type=str, default=None, help="WandB group name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="WandB run name")


    parser.add_argument('--include_eval', action='store_true', default=False, help="Run Chamfer Distance")
    parser.add_argument('--show_plots', action='store_true', default=False,
                        help="Dump periodic render/grid diagnostic figures (interactive in debug mode)")

    parser.add_argument(
        "--seed", type=int, default=22,
        help="Random seed for reproducible runs"
    )
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--use_cont", action="store_true")
    parser.add_argument("--use_depth_normal", action="store_true")
    scan_id = os.environ.get("SCAN_ID", "")
    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, scan_id)
    args.source_path = os.path.join(args.source_path, scan_id)
    setup_seed(args.seed)
    print(f"[INFO] Using seed={args.seed}")
    if args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={devices}")
    gpu_id = int(devices[0]) if devices and devices[0].isdigit() else 0
    args.eval_gpu = args.train_gpu = f"cuda:{gpu_id}"

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), args,

    )



if __name__ == "__main__":
    main()

