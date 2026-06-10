#!/usr/bin/env python
"""
Two-Stage Coarse-to-Fine Training for Knots.

Stage 1: Train from SfM point cloud for N_warmup iterations.
Bridge:  Render all views → TSDF → mesh → fit B-spline to mesh.
Stage 2: Re-initialize spline from mesh-fitted control grid, train fully.

Usage:
    python scripts/two_stage_train.py -s <source_path> -m <model_path> \
        --warmup_iterations 5000 --total_iterations 30000 \
        --voxel_size 0.004 --max_depth 5.0
"""

import os
import gc
import numpy as np
import torch
import open3d
import open3d as o3d
from argparse import ArgumentParser
from tqdm import tqdm
from arguments import ModelParams, PipelineParams
from arguments.nurbs_params import NurbsOptimizationParams
from modules.KnotSurface import SplineModel
from modules.control_feature.utils.rotation_control import bridge_mesh_normals_to_control_quaternions, \
    inject_quaternions_into_fit_result
from modules.multisurf import MultiSurfaceSplineModel
from optimize_nurbs import training, prepare_output_and_logger
from scene import Scene
from scene.app_model import AppModel
from scene.cameras import Camera
from spline_scene import SplineScene
# from modules.fitting.customized import MultiSurfaceSplineModel
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal


# ======================================================================
#  STAGE 1: Warmup Training
# ======================================================================

def stage1_warmup(dataset:ModelParams, opt: NurbsOptimizationParams, pipe: PipelineParams, args, warmup_iters: int,
                  scene):
    """
    Run standard training for `warmup_iters` iterations.
    Saves a checkpoint at the end.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Warmup training for {warmup_iters} iterations")
    print(f"{'='*60}\n")

    # Override iteration count for warmup
    original_iters = opt.iterations
    densify_from = opt.densify_from_iter
    mv_from = opt.multi_view_weight_from_iter
    s_from = opt.single_view_weight_from_iter
    opt.multi_view_weight_from_iter = int(warmup_iters * 0.8)
    opt.single_view_weight_from_iter = int(warmup_iters * 0.8)

    # Force checkpoint at the end of warmup
    args.checkpoint_iterations = [warmup_iters]
    args.save_iterations = [warmup_iters]
    args.start_checkpoint = -1
    opt.iterations = warmup_iters
    opt.batch_size = 1
    test_iterations = list(range(0, 30000, 1000))
    # test_iterations = [30000]
    evals = [15_000, 21_000, 25_000, 30_000]
    args.evaluation_iterations = [warmup_iters]
    args.test_iterations = test_iterations
    training(dataset, opt, pipe, args, scene=scene)
    set_args_for_stage2(args, opt)


def set_scene(args, dataset, opt, pipe):
    dirname = dataset.model_path.split('/')[-1]
    scene_name = os.path.basename(dataset.model_path)  # e.g., scan118
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    app_model = AppModel()
    if len(dirname.split("_")) <= 1 and not args.include_eval:
        trial_index = prepare_output_and_logger(dataset)
    else:
        trial_index = scene_name.split("_")[-1] if len(scene_name.split("_")) > 1 else "0"
    # Import and run the standard training loop
    scene = SplineScene(
        dataset, opt, scan_id=scene_name,
        pipe=pipe, background=background, app_model=app_model,
    )
    return scene


def set_args_for_stage2(args, opt, **kwargs):
    """
    Adjust args for Stage 2, which will load the checkpoint and continue training.
    """

    # Set scheduling params
    args.evaluation_iterations = args.checkpoint_iterations = args.save_iterations = [i * 2000 for i in range(4, 16)]
    args.test_iterations = [i*1000 for i in range(0, 31)]
    opt.iterations = 30_000
    opt.densify_from_iter = kwargs.get("densify_from_iter", 500)
    args.checkpoint_iterations = args.save_iterations = args.evaluation_iterations = [9000, 11_000, 15_000, 21_000, 25_000, 30_000]
    args.test_iterations = [i*1000 for i in range(0, 31)]
    # ---- STAGE 2 ----
    args.include_eval = True
    args.use_wandb = True
    # Set Spline optimization params
    stage2_params = {
        # Training params
        "densify_from_iter": kwargs.get("densify_from_iter", 500),
        "densify_until_iter": kwargs.get("densify_until_iter", 15_000),
        "densification_interval": kwargs.get("densification_interval", 100),
        "multi_view_weight_from_iter": kwargs.get("multi_view_weight_from_iter", 7000),
        "single_view_weight_from_iter": kwargs.get("single_view_weight_from_iter", 7000),
        "lambda_eikonal": 0.1,
        "eikonal_from_iter": 0,
        "quat_smoothness_weight": 0.0,
        "quat_smooth_from": 0,
        "normal_smoothness_weight": 0.0,
        "normal_smooth_from": 0,
        "normal_dev_weight": 0.0,
        "normal_dev_from": 0,
        "local_planar_deviation_weight": 0.0,
        "local_planar_deviation_from": 7000,
        # Spline architecture params
        "sampling_density": 1,
    }

    stage2_params.update(kwargs)
    for key, value in stage2_params.items():
        if hasattr(opt, key):
            setattr(opt, key, value)



# ======================================================================
#  BRIDGE: Render → TSDF → Mesh → Fit B-Spline
# ======================================================================

def bridge_extract_mesh(
    checkpoint_path: str,
    dataset: ModelParams,
    pipe: PipelineParams,
    voxel_size: float = 0.004,
    max_depth: float = 5.0,
    depth_filter: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Load warmup checkpoint, render all training views,
    fuse depths via TSDF, extract mesh.
    """
    print(f"\n{'='*60}")
    print(f"  BRIDGE: Extracting mesh from warmup model")
    print(f"{'='*60}\n")

    from scene.gaussian_model import GaussianModel
    #
    # with torch.no_grad():
    #     # Load the warmup model
    #     # state_dict, iteration = torch.load(checkpoint_path)
    #     nurbs = MultiSurfaceSplineModel()
    #     # for surf_state in state_dict['surfaces']:
    #         from modules.KnotSurface import SplineModel
    #
    #         surface = SplineModel(late_init=True)
    #         surface.restore(surf_state)
    #         # surface.restore(surf_state, train_model=train_model)
    #     #     surfaces.append(surface)

    # Load scene for cameras
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    views = scene.getTrainCameras()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    print(f"[Bridge] Rendering {len(views)} views and integrating TSDF...")

    for idx, view in enumerate(tqdm(views, desc="Render + TSDF")):
        # Forward pass
        nurbs._invalidate_cache()
        out = render(view, nurbs, pipe, background, app_model=None)

        rendering = out["render"].clamp(0.0, 1.0)
        depth = out["plane_depth"].squeeze()

        _, H, W = rendering.shape

        # Optional: grazing angle filter
        if depth_filter and "depth_normal" in out:
            view_dir = torch.nn.functional.normalize(
                view.get_rays(), p=2, dim=-1
            )
            depth_normal = out["depth_normal"].permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(
                depth_normal, p=2, dim=-1
            )
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot.clamp(-1, 1))
            mask = angle > (80.0 / 180 * 3.14159)
            depth[mask] = 0

        # Clamp depth
        depth[depth > max_depth] = 0

        if view.mask is not None:
            depth[view.mask.squeeze() < 0.5] = 0

        # Convert to Open3D
        depth_np = depth.detach().cpu().numpy()
        depth_o3d = o3d.geometry.Image(
            (depth_np * 1000).astype(np.uint16)
        )

        # Save rendered color as temp image, read back as Open3D Image
        color_np = (
            rendering.permute(1, 2, 0).clamp(0, 1)
            .mul(255).byte().cpu().numpy()
        )
        color_o3d = o3d.geometry.Image(color_np)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )

        # W2C pose: [R^T | T]
        pose = np.identity(4)
        pose[:3, :3] = view.R.transpose(-1, -2)
        pose[:3, 3] = view.T

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            W, H, view.Fx, view.Fy, view.Cx, view.Cy
        )

        volume.integrate(rgbd, intrinsic, pose)

        # Extract mesh
        print("[Bridge] Extracting triangle mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        # Save mesh
        mesh_dir = os.path.join(dataset.model_path, "mesh")
        os.makedirs(mesh_dir, exist_ok=True)
        mesh_path = os.path.join(mesh_dir, "stage1_tsdf.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"[Bridge] Mesh saved: {mesh_path} "
              f"({len(mesh.vertices)} verts, {len(mesh.triangles)} faces)")

        # Cleanup
        del nurbs, scene, gaussians
        gc.collect()
        torch.cuda.empty_cache()

    return mesh


def bridge_fit_spline_to_mesh(
    mesh: open3d.geometry.TriangleMesh,
    dataset: ModelParams,
    config: NurbsOptimizationParams,
    resolution: tuple = (128, 128),
    smoothing: float = 0.01,
    use_pca=True) -> dict:
    """
    Fit B-spline surface(s) to the TSDF mesh.

    Returns dict with control points, colors, knots — everything
    needed to re-initialize a MultiSurfaceSplineModel.
    """
    print(f"\n[Bridge] Fitting B-spline to mesh...")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    try:
        # colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    except Exception as e:
        print(f"[Bridge] Error accessing vertex colors: {e}")
        # colors = np.full_like(vertices, 0.5)
        print("[Bridge] Warning: mesh has no vertex colors, using grey fallback")

    if len(colors) == 0 or colors.shape[0] != vertices.shape[0]:
        colors = np.full_like(vertices, 0.5)

    print(f"[Bridge] Mesh: {len(vertices)} vertices, "
          f"{len(triangles)} triangles")

    mesh = mesh.filter_smooth_laplacian(number_of_iterations=10)
    # --- 1. Sample mesh uniformly to get dense point cloud ---
    # Poisson disk sampling gives uniform coverage
    pcd = mesh.sample_points_poisson_disk(
        number_of_points=min(102_400, len(vertices) * 2),
        use_triangle_normal=True,
    )


    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    # Interpolate colors from nearest mesh vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    _, nearest_idx = tree.query(points, k=1)
    sample_colors = colors[nearest_idx]

    print(f"[Bridge] Sampled {len(points)} points from mesh")

    # --- 2. UV parameterization ---
    try:
        if use_pca:
            raise ImportError("PCA fallback forced for testing")
        from modules.fitting.parametrization.conformal_uv import (
            conformal_parameterize,
        )
        uv = conformal_parameterize(points, mesh=mesh)
        print(f"[Bridge] UV: conformal parameterization")
    except Exception as e:
        print(f"[Bridge] Conformal UV failed ({e}), using PCA fallback")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(points - points.mean(axis=0))
        uv = proj - proj.min(axis=0)
        rng = uv.max(axis=0)
        rng[rng < 1e-10] = 1.0
        uv = uv / rng
        uv = np.clip(uv, 0.001, 0.999)

    # --- 3. Least-squares B-spline fitting ---
    from modules.fitting.bspline_fitting import BSplineSurfaceFitter

    res_u, res_v = resolution
    fitter = BSplineSurfaceFitter(
        n_ctrl_u=res_u,
        n_ctrl_v=res_v,
        degree_u=3,
        degree_v=3,
        smoothing=smoothing,
        data_dependent_knots=True,
    )

    result = fitter.fit(
        points=points,
        uv_params=uv,
        colors=sample_colors,
    )

    print(f"[Bridge] B-spline fit: {res_u}×{res_v} control grid, "
          f"RMS residual = {result.residual_rms:.6f}")

    return {
        "control_points": result.control_points,      # [H, W, 3]
        "control_colors": result.control_colors,       # [H, W, 3]
        "knots_u": result.knots_u,                     # [H + degree + 1]
        "knots_v": result.knots_v,                     # [W + degree + 1]
        "degree_u": result.degree_u,
        "degree_v": result.degree_v,
        "uv": uv,
        "points": points,
        "colors": sample_colors,
    }

def bridge_fit_spline_to_mesh2(
    mesh: o3d.geometry.TriangleMesh,
    dataset: ModelParams,
    config: NurbsOptimizationParams,
    resolution: tuple = (128, 128),
    smoothing: float = 0.01,
    grid_sample_factor: int = 4,
) -> dict:
    """
    Fit B-spline surface(s) to the TSDF mesh.

    Uses the FAST separable path (fit_from_grid) by first resampling
    the mesh onto a regular UV grid.

    Args:
        grid_sample_factor: the resampling grid is (resolution * factor)
            in each direction. Higher = more accurate, but the solve
            cost is independent of this (only basis eval scales).
    """
    import time
    t0 = time.time()
    print(f"\n[Bridge] Fitting B-spline to mesh...")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)

    try:
        colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    except Exception:
        colors = np.full_like(vertices, 0.5)

    if len(colors) == 0 or colors.shape[0] != vertices.shape[0]:
        colors = np.full_like(vertices, 0.5)

    print(f"[Bridge] Mesh: {len(vertices)} verts, {len(triangles)} faces")

    # --- 1. UV parameterization of mesh vertices ---
    t1 = time.time()
    try:
        from modules.fitting.parametrization.conformal_uv import (
            conformal_parameterize,
        )
        uv = conformal_parameterize(vertices, mesh=mesh)
        print(f"[Bridge] UV: conformal ({time.time()-t1:.2f}s)")
    except Exception as e:
        print(f"[Bridge] Conformal UV failed ({e}), using PCA")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        proj = pca.fit_transform(vertices - vertices.mean(axis=0))
        uv = proj - proj.min(axis=0)
        rng = uv.max(axis=0)
        rng[rng < 1e-10] = 1.0
        uv = uv / rng
        uv = np.clip(uv, 0.001, 0.999)
        print(f"[Bridge] UV: PCA fallback ({time.time()-t1:.2f}s)")

    # --- 2. Resample onto regular grid ---
    t2 = time.time()
    res_u, res_v = resolution
    Gu = res_u * grid_sample_factor   # e.g., 512 for 128 ctrl points
    Gv = res_v * grid_sample_factor

    from scipy.interpolate import griddata

    u_grid = np.linspace(0.001, 0.999, Gu)
    v_grid = np.linspace(0.001, 0.999, Gv)
    uu, vv = np.meshgrid(u_grid, v_grid, indexing='ij')
    grid_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)  # [Gu*Gv, 2]

    grid_xyz = np.zeros((Gu * Gv, 3), dtype=np.float32)
    grid_rgb = np.zeros((Gu * Gv, 3), dtype=np.float32)

    for dim in range(3):
        grid_xyz[:, dim] = griddata(
            uv, vertices[:, dim], grid_uv,
            method='linear', fill_value=vertices[:, dim].mean()
        )
        grid_rgb[:, dim] = griddata(
            uv, colors[:, dim], grid_uv,
            method='linear', fill_value=colors[:, dim].mean()
        )

    # Fill NaN with nearest-neighbor
    nan_mask = np.isnan(grid_xyz).any(axis=1)
    if nan_mask.any():
        for dim in range(3):
            grid_xyz[nan_mask, dim] = griddata(
                uv, vertices[:, dim], grid_uv[nan_mask],
                method='nearest'
            )
            grid_rgb[nan_mask, dim] = griddata(
                uv, colors[:, dim], grid_uv[nan_mask],
                method='nearest'
            )

    grid_xyz = grid_xyz.reshape(Gu, Gv, 3)
    grid_rgb = np.clip(grid_rgb.reshape(Gu, Gv, 3), 0, 1)

    print(f"[Bridge] Resampled to {Gu}×{Gv} grid ({time.time()-t2:.2f}s)")

    # --- 3. FAST separable B-spline fit ---
    t3 = time.time()
    from modules.fitting.bspline_fitting import BSplineSurfaceFitter

    fitter = BSplineSurfaceFitter(
        n_ctrl_u=res_u,
        n_ctrl_v=res_v,
        degree_u=3,
        degree_v=3,
        smoothing=smoothing,
        use_gpu=True,
    )

    result = fitter.fit_from_grid(
        grid_xyz=grid_xyz,
        grid_rgb=grid_rgb,
    )

    t4 = time.time()
    print(f"[Bridge] B-spline fit: {res_u}×{res_v}, "
          f"RMS={result.residual_rms:.6f} ({t4-t3:.2f}s)")
    print(f"[Bridge] Total bridge time: {t4-t0:.2f}s")

    return {
        "control_points": result.control_points,
        "control_colors": result.control_colors,
        "knots_u": result.knots_u,
        "knots_v": result.knots_v,
        "degree_u": result.degree_u,
        "degree_v": result.degree_v,
        "uv": uv,
        "points": vertices,
        "colors": colors,
    }
# ======================================================================
#  STAGE 2: Full Training from Mesh-Fitted Initialization
# ======================================================================

def stage2_full_training(
    dataset: ModelParams,
    opt: NurbsOptimizationParams,
    pipe: PipelineParams,
    args,
    fit_result: dict,
    scene: SplineScene

):
    """
    Re-initialize the spline model from mesh-fitted B-spline
    and run full training.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Full training from mesh-fitted initialization")
    print(f"{'='*60}\n")
    surfaces = []
    with torch.no_grad():
        surface = SplineModel(
            surf_data=fit_result,
            config=opt,
            args=args,
            spatial_lr_scale=scene.cameras_extent,
            surf_uid=0,
        )

        surfaces.append(surface)

    nurbs = MultiSurfaceSplineModel(surfaces, setup_training=True)
    print(f"[Stage 2] Model re-initialized. Starting full training...")
    args.start_checkpoint = None
    training(dataset, opt, pipe, args, scene=scene, nurbs=nurbs)


# ======================================================================
#  MAIN
# ======================================================================

def main():
    parser = ArgumentParser(description="Two-stage training for Knots")
    lp = ModelParams(parser)
    # op = OptimizationParams(parser)
    op = NurbsOptimizationParams(parser)
    pp = PipelineParams(parser)
    # Standard arguments
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
    print(f"Project directory: {PROJECT_DIR}")
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # first_iter = 1000
    first_iter = op.multi_view_weight_from_iter
    cycle = 1
    test_iterations = list(range(0, 30000, 1000))
    evals = [-1]
    evaluation_iterations = checkpoint_iterations = save_iterations = evals

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
    parser.add_argument('--wandb_project', type=str, default="NURBS-Training", help="WandB project name")
    parser.add_argument('--wandb_group', type=str, default=None, help="WandB group name")
    parser.add_argument('--wandb_run_name', type=str, default=None, help="WandB run name")

    parser.add_argument(
        "--seed", type=int, default=22,
        help="Random seed for reproducible runs"
    )
    # Two-stage specific arguments
    parser.add_argument("--warmup_iterations", type=int, default=1000,
                        help="Number of iterations for Stage 1 warmup")
    parser.add_argument("--total_iterations", type=int, default=30000,
                        help="Total iterations for Stage 2 full training")
    parser.add_argument("--voxel_size", type=float, default=0.004,
                        help="TSDF voxel size for mesh extraction")
    parser.add_argument("--max_depth", type=float, default=5.0,
                        help="Maximum depth for TSDF integration")
    parser.add_argument("--mesh_fit_resolution", type=int, default=128,
                        help="Control grid resolution for mesh fitting")
    parser.add_argument("--mesh_fit_smoothing", type=float, default=0.01,
                        help="Regularization weight for mesh fitting")
    parser.add_argument("--use_depth_filter", action="store_true",
                        help="Filter grazing-angle depths in TSDF")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip warmup, use existing checkpoint")
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to existing Stage 1 checkpoint")
    parser.add_argument("--use_cont", action="store_true")
    parser.add_argument("--use_depth_normal", action="store_true")
    parser.add_argument('--include_eval', action='store_true', default=False, help="Run Chamfer Distance")
    scan_id = os.environ.get("SCAN_ID", "")
    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, scan_id)
    args.source_path = os.path.join(args.source_path, scan_id)
    args.use_wandb = True
    print(f"[INFO] Using seed={args.seed} for full determinism")
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={devices}, parsed devices: {devices}")


    args.eval_gpu = args.train_gpu = f"cuda:{devices[0]}"

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    warmup_iters = args.warmup_iterations
    total_iters = args.total_iterations
    fit_res = (args.mesh_fit_resolution, args.mesh_fit_resolution)

    # Ensure output directory exists
    os.makedirs(dataset.model_path, exist_ok=True)
    scene = set_scene(args, dataset, opt, pipe)

    # ---- STAGE 1 ----
    if args.skip_stage1 and args.stage1_checkpoint:
        checkpoint_path = args.stage1_checkpoint
        print(f"[Main] Skipping Stage 1, using checkpoint: {checkpoint_path}")
    else:
        import multiprocessing as mp

        mp.set_start_method("spawn", force=True)
        stage1_warmup(
            dataset, opt, pipe, args, warmup_iters, scene=scene
        )
    mesh_path = os.path.join(dataset.model_path, "mesh")
    mesh_path = os.path.join(mesh_path, "tsdf_fusion.ply")
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.compute_vertex_normals()

    fit_result = bridge_fit_spline_to_mesh(
        mesh=mesh,
        dataset=dataset,
        config=opt,
        resolution=fit_res,
        smoothing=args.mesh_fit_smoothing,
    )

    control_quats = bridge_mesh_normals_to_control_quaternions(
        mesh=mesh,
        fit_result=fit_result,
        smoothing_iters=1,
        orientation_reference='z_positive',
    )
    fit_result = inject_quaternions_into_fit_result(fit_result, control_quats)


    stage2_full_training(dataset, opt, pipe, args, fit_result, scene=scene)
    print(f"\n{'='*60}")
    print(f"  Two-stage training complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()