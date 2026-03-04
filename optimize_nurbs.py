#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
import builtins
# ----  F R E E Z E   S O U R C E   T R E E  ----
import pathlib, datetime
import time
from typing import List, Dict, Tuple

import json
import shutil
import os
import re
import subprocess
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from modules.decompose import WarmupDecompositionController, ControllerConfig, DecompPhase, safe_decompose

# from model.modules.decompose import WarmupDecompositionController, ControllerConfig, DecompPhase, safe_decompose

VERBOSE = os.environ.get('VERBOSE', '0') == '1'
# Override built-in print to respect VERBOSE flag
VERBOSE = '1' # By Force!
original_print = builtins.print
def conditional_print(*args, **kwargs):
    if VERBOSE:
        original_print(*args, **kwargs)
builtins.print = conditional_print
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
print(f"Project directory: {PROJECT_DIR}")
def safe_git_commit(message: str, max_attempts: int = 10, sleep_base: float = 1.0) -> None:
    """
    Atomically `git add -A` + `git commit -m <message>` with retries.
    If another process holds .git/index.lock we back‑off, wait, and retry.

    Args:
        message: Commit message.
        max_attempts: Max number of retries before bailing.
        sleep_base: Base seconds to wait; actual wait is sleep_base * (attempt #) plus jitter.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            subprocess.check_call(["git", "add", "-A"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(["git", "commit", "-m", message],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except subprocess.CalledProcessError:
            lock_path = pathlib.Path(".git/index.lock")
            if lock_path.exists():
                # Another process is mid‑commit; wait then retry
                attempt += 1
                time.sleep(sleep_base * attempt + random.uniform(0, 0.5))
            else:
                # Commit failed for another reason – re‑raise
                raise
    raise RuntimeError("safe_git_commit: giving up after multiple attempts – .git/index.lock is still present.")

REPO_ROOT = pathlib.Path(__file__).resolve().parent
RUN_DIR    = pathlib.Path(os.environ.get("PGSR_RUN_DIR",
                    REPO_ROOT / "runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
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

from arguments import ModelParams, NurbsOptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.app_model import AppModel
from scene.cameras import Camera
from spline_scene import SplineScene, refresh_camera_neighbors_post_warmup
from utils.general_utils import safe_state
from utils.graphics_utils import patch_offsets, patch_warp
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
    if trial_index == 0:

        model_exp_dir =  f"{out_base_path}/surfels/{scene_name}"
    else:
        model_exp_dir = f"{out_base_path}/surfels/{scene_name}_{trial_index}"

    # model_dir = f"{out_base_path}/surfels/{scene_name}_{trial_index}"
    # mesh_path = f"{model_dir}/mesh/tsdf_fusion_post.ply"
    # common_args = f'--num_cluster 1 --use_depth_filter --voxel_size {0.002} --max_depth {5.0} --iteration {iteration} --quiet'
    mesh_output_dir = os.path.join(model_exp_dir, "mesh")
    # A dedicated directory to store the artifacts of the best run
    best_artifacts_dir = os.path.join(model_exp_dir, "best_run_artifacts")
    # A file to persist the best score and iteration across evaluation calls
    best_model_info_file = os.path.join(model_exp_dir, "best_model_info.json")
    # Default best‑model record so it is always defined
    best_model_info = {'score': float('inf'), 'iteration': -1}



    try:
        # Set the CUDA device for this worker process
        # torch.cuda.set_device(eval_gpu)

        python_executable = sys.executable
        common_args = f'--num_cluster 1 {depth_filter} --voxel_size {0.002} --max_depth {5.0} {use_depth_normal} --iteration {iteration} --quiet'

        with_rendering = "spline_render"


        render_cmd = f'{python_executable} {with_rendering}.py -m {model_exp_dir} {common_args}'
        # print(render_cmd)
        execute_command(render_cmd, log_messages)
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


        # 3. Parse score from output
        chamfer_match = re.search(r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", eval_stdout, re.MULTILINE)
        if chamfer_match:
            chamfer_dict = {'mean_d2s': float(chamfer_match.group(1)), 'mean_s2d': float(chamfer_match.group(2)),
                            'over_all': float(chamfer_match.group(3))}
            current_score = chamfer_dict['over_all']
            log_messages.append(f"[Eval Worker] Parsed Chamfer Metrics for iter {iteration}: {chamfer_dict}")
        else:
            log_messages.append("[Eval Worker] Warning: Could not parse Chamfer metrics. Cannot compare performance.")
            shutil.rmtree(mesh_output_dir)
            return {"logs": "\n".join(log_messages), "metric_dict": None, "iteration": iteration}

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
            shutil.rmtree(mesh_output_dir)
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


    except Exception as e:
        log_messages.append(f"[Eval Worker] An unexpected error occurred: {e}")
        if os.path.exists(mesh_output_dir):
            shutil.rmtree(mesh_output_dir)
        return {"logs": "\n".join(log_messages), "metric_dict": None, "iteration": iteration, "best": None, "best_score": np.inf}
    finally:
        if os.path.exists(temp_checkpoint_path):
            os.remove(temp_checkpoint_path)
        # On error or normal exit: cleanup if requested
        if do_cleanup:
            try:
                cleanup_non_best_artifacts(model_exp_dir, best_model_info.get('iteration', -1))
            except Exception as e:
                log_messages.append(f"[Eval Worker] Cleanup failed: {e}")

WAND_LOGGER_TERMINATE = "__WAND_TERMINATE__"


def wandb_logger_worker(queue: Queue, config):
    try:
        wandb.init(
            project=config.get("project", "default"),
            name=config.get("name", None),
            group=config.get("group", None),
            config=config.get("config", {}),
            entity=config.get("entity", None),
            settings=wandb.Settings(start_method='thread') #, _disable_stats=True, _disable_meta=True)
        )
        wandb.save("model/__init__.py")
        wandb.save("model/modules/KnotSurface.py")
        wandb.save("arguments/__init__.py")
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
        while True:
            item = queue.get()
            if item == WAND_LOGGER_TERMINATE:
                break
            try:
                # For metrics with a defined step_metric, wandb will use the value
                # from the data dictionary (e.g., item['data']['iteration']).
                # The 'step' argument is used for all other standard metrics.
                wandb.log(item["data"], step=item.get("step", None))
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
        axs[1].sampling_grid(True, linestyle='--', alpha=0.5)

        # V Density Bar Plot
        axs[2].bar(np.arange(len(density_v)), density_v, color='lightgreen')
        axs[2].set_title("V Partition Density")
        axs[2].set_xlabel("V Bins")
        axs[2].set_ylabel("Normalized Density")
        axs[2].set_ylim(0, 1)
        axs[2].sampling_grid(True, linestyle='--', alpha=0.5)

        # 2D Intensity Heatmap
        im = axs[3].imshow(density_2d, cmap='viridis', aspect='auto', origin='lower')
        axs[3].set_title("2D Joint Density Intensity")
        axs[3].set_xlabel("V Bins")
        axs[3].set_ylabel("U Bins")
        fig.colorbar(im, ax=axs[3], orientation='vertical', fraction=0.046, pad=0.04)
        axs[3].sampling_grid(False)

        plt.tight_layout()

        if save_dir:
            plt.savefig(f"{save_dir}/view_{uid}_distribution.png", dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

        figures.append(fig)

    return figures


def plot_render_outputs(render_dict, nurbs_prop_dict, gt_image, nurbs: 'MultiSurfaceSplineModel', uid=None, title=None):
    """
    Side-by-side visualization:
    - Left (50%): High-res renders (GT, RGB, normals, depth) → stacked vertically
    - Right (50%): Grid quantities → arranged in a compact, near-square grid
    """
    import matplotlib.gridspec as gridspec
    render_dict['gt_image'] = gt_image
    # ------------------------------------------------------------------ #
    # Visualization helpers
    # ------------------------------------------------------------------ #
    vis_normal = lambda x: np.uint8((x[..., [1, 2, 0]] + 1) / 2 * 255)
    # vis_normal = lambda x: x
    vis_sh     = lambda x: SH2RGB(x[..., :3]).detach().cpu().numpy()
    vis_gray   = lambda x: (x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else x)
    vis_grad   = lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-6)).cpu().numpy() if isinstance(x, torch.Tensor) else x

    # ------------------------------------------------------------------ #
    # Plot definitions
    # ------------------------------------------------------------------ #
    image_plots = [
        {'key': 'gt_image',                     'title': 'Ground Truth',      'vis': None},
        {'key': 'render',                     'title': 'Rendered RGB',      'vis': None},
        {'key': 'rendered_normal',            'title': 'Rendered Normal',   'vis': lambda x: vis_normal(x.permute(1,2,0).cpu().numpy())},
        {'key': 'depth_normal',               'title': 'Depth Normal',      'vis': lambda x: vis_normal(x.permute(1,2,0).cpu().numpy())},
        {'key': 'plane_depth',                'title': 'Plane Depth',       'vis': None},
        {'key': 'decomposed_final_img',                'title': 'Decomposed Images',       'vis': None},
        {'key': 'object',                'title': 'Decomposed Images',       'vis': None},
        {'key': 'background',                'title': 'Decomposed Images',       'vis': None},
    ]


    grid_plots_all = []
    for i, surf in enumerate(nurbs.surfaces):
        grid_shape = surf.state.sampling_layout
        cp_grid_shape = surf.state.control_layout
        grid_plots_all.extend([
            {'key': f'norm_grid_{i}',               'title': f'Grid Normals {i}',      'vis': lambda x, gs=grid_shape: vis_normal(x.cpu().numpy().reshape(gs))},
            {'key': f'weights_map_per_view_{i}',               'title': f'Surface {surf.label}: Grid Weights {i}',      'vis': lambda x, gs=grid_shape: vis_gray(x.cpu().numpy().reshape(gs))},
            {'key': f'depth_map_per_view_{i}',               'title': f'Grid Depth {i}',      'vis': lambda x, gs=grid_shape: vis_grad(x.cpu().numpy().reshape(gs))},
            {'key': f'sh_grid_{i}',                 'title': f'Surface {surf.label}: Grid SH {i}',           'vis': lambda x, gs=grid_shape: vis_sh(x).reshape(gs)},
            {'key': f'sh_cpt_{i}',                 'title': f'Surface {surf.label}: Control - Grid SH {i}',           'vis': lambda x, gs=cp_grid_shape: vis_sh(x).reshape(gs)},
            {'key': f'visibility_cp_{i}',                 'title': f'Grid SH {i}',           'vis': lambda x, gs=cp_grid_shape: vis_sh(x).reshape(gs)},
            ])



    # (0, grid_shape_0, cp_grid_shape_0),
    # (1, grid_shape_1, cp_grid_shape_1),

    #grid_plots_all = [
    #     {'key': 'Grads XYZ',                  'title': 'Grads XYZ',         'vis': lambda x: vis_grad(x.reshape(nurbs.state.control_layout))},
    #     {'key': 'Grid Normals',               'title': 'Grid Normals',      'vis': lambda x: vis_normal(x.cpu().numpy().reshape(grid_shape))},
    #     {'key': f'sh_grid_{i}',   'title': 'Grid SH',           'vis': lambda x: vis_sh(x).reshape(grid_shape)},
    #     # {'key': 'Grid Opacity',               'title': 'Grid Opacity',      'vis': lambda x: vis_gray(x).reshape(grid_shape)},
    #     # {'key': 'Grid Scale',                 'title': 'Scaling Norm',      'vis': lambda x: vis_gray(x).reshape(grid_shape)},
    #     {'key': 'out_observe',                'title': 'Out Observe',       'vis': lambda x: vis_grad(x.float()).reshape(grid_shape)},
    #     {'key': 'radii',                      'title': 'Radii',             'vis': lambda x: vis_grad(x.float()).reshape(grid_shape)},
    #     {'key': 'Normals',          'title': 'Normals From Depths', 'vis': lambda x: vis_normal(x.float()).reshape(grid_shape)},
    #     {'key': 'Depths',          'title': 'Depths', 'vis': lambda x: vis_grad(x.float()).reshape(cp_grid_shape)},
    #     {'key': 'CP Spherical Harmonics',          'title': 'CP Spherical Harmonics', 'vis': lambda x: vis_sh(x.float()).reshape(cp_grid_shape)},
    #     {'key': 'Backfaces',          'title': 'Backfaces Filter', 'vis': lambda x: vis_gray(x.float()).reshape(grid_shape)},
    # ]
    to_include = []
    for i in range(len(nurbs.surfaces)):
        to_include.extend([
            f'norm_grid_{i}',
            f'sh_grid_{i}',
            f'weights_map_per_view_{i}',
            f'sh_cpt_{i}',
                           # f'sh_cpt_{i}',
                           # f'visibility_cp_{i}'
                           ])


    grid_plots = []
    for p in grid_plots_all:
        if p['key'] in to_include:
            grid_plots.append(p)

    # Optional density intensity
    if uid is not None:
        try:
            density_u = getattr(nurbs, 'partition_density_u', None)
            density_v = getattr(nurbs, 'partition_density_v', None)
            if density_u is not None and density_v is not None:
                du = density_u.cpu().numpy()[uid] / (density_u[uid].max() + 1e-6)
                dv = density_v.cpu().numpy()[uid] / (density_v[uid].max() + 1e-6)
                density_2d = np.outer(du, dv)
                grid_plots.append({'img': density_2d, 'title': 'Density Intensity', 'vis': None})
        except:
            pass

    # ------------------------------------------------------------------ #
    # Collect images
    # ------------------------------------------------------------------ #
    def collect(plist, rdict):
        imgs, titles = [], []
        for p in plist:
            if 'key' in p and p['key'] in rdict:
                data = rdict[p['key']]
            elif 'img' in p:
                data = p['img']
            else:
                continue
            img = p['vis'](data) if p['vis'] else data
            imgs.append(img)
            titles.append(p['title'])
        return imgs, titles

    img_images, img_titles   = collect(image_plots, render_dict)
    grid_images, grid_titles = collect(grid_plots,   nurbs_prop_dict)

    if not img_images and not grid_images:
        return

    # ------------------------------------------------------------------ #
    # Preprocessing + resizing
    # ------------------------------------------------------------------ #
    gt_h, gt_w = (gt_image.shape[-2:] if isinstance(gt_image, torch.Tensor) else gt_image.shape[:2])

    def preprocess(img, target_size=None):
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = img.permute(1, 2, 0)
            img = img.detach().cpu().numpy()

        img = np.clip(img, 0, 1) if img.dtype != np.uint8 else img.astype(np.float32)/255.0

        if target_size is not None:
            h, w = img.shape[:2]
            if (h, w) != target_size:
                c = 1 if img.ndim == 2 else img.shape[-1]
                t = torch.from_numpy(img)
                t = t.permute(2, 0, 1) if c == 3 else t.unsqueeze(0)
                t = F.interpolate(t.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)[0]
                img = t.permute(1, 2, 0).numpy() if c > 1 else t[0].numpy()
        return img

    # Preprocess all
    img_processed   = [preprocess(im, (gt_h, gt_w)) for im in img_images]
    grid_processed  = [preprocess(im ) for im in grid_images]

    # Find best near-square layout for grid plots
    def best_grid_layout(n):
        if n <= 1: return 1, 1
        best = (1, n)
        min_diff = float('inf')
        for rows in range(1, int(n**0.5) + 2):
            cols = (n + rows - 1) // rows
            diff = abs(cols - rows)
            waste = rows * cols - n
            if diff < min_diff or (diff == min_diff and waste < (best[0] * best[1] - n)):
                min_diff = diff
                best = (rows, cols)
        return best[0], best[1]

    grid_rows, grid_cols = best_grid_layout(len(grid_processed))
    im_rows, im_cols = best_grid_layout(len(img_images))

    # Target uniform size for grid plots (use largest dimension)
    if grid_processed:
        max_h = max(im.shape[0] for im in grid_processed)
        max_w = max(im.shape[1] for im in grid_processed)
        target_grid_size = (max_h, max_w)
        grid_processed = [preprocess(im.squeeze(), target_grid_size) for im in grid_images]  # re-process with final size

    # ------------------------------------------------------------------ #
    # Final figure with GridSpec
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(24, 10))  # Wide, balanced
    outer_gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, width_ratios=[1, 1])

    # Left: Image renders (vertical stack)
    left_gs = gridspec.GridSpecFromSubplotSpec(im_rows, im_cols, subplot_spec=outer_gs[0],
                                               wspace=0.01, hspace=0.1)
    try:
        for i, (img, ttl) in enumerate(zip(img_processed, img_titles)):

            ax = fig.add_subplot(left_gs[i // im_cols, i % im_cols])
            cmap = 'jet' if 'Depth' in ttl else ('gray' if img.ndim == 2 or img.shape[2] == 1 else None)
            ax.imshow(img.squeeze() if img.ndim == 3 and img.shape[2] == 1 else img, cmap=cmap)
            ax.set_title(ttl, fontsize=13, pad=10)
            ax.axis('off')
    except Exception as e:
        print(f"Error plotting image renders: {e}")
        pass


    # Right: Grid plots in compact grid
    if grid_processed:
        right_gs = gridspec.GridSpecFromSubplotSpec(grid_rows, grid_cols,
                                                    subplot_spec=outer_gs[1],
                                                    wspace=0.08, hspace=0.3)
        for i, (img, ttl) in enumerate(zip(grid_processed, grid_titles)):
            ax = fig.add_subplot(right_gs[i // grid_cols, i % grid_cols])
            cmap = 'gray' if img.ndim == 2 or img.shape[2] == 1 else None
            ax.imshow(img.squeeze() if img.ndim == 3 and img.shape[2] == 1 else img, cmap=cmap)
            ax.set_title(ttl, fontsize=11)
            ax.axis('off')

        # Turn off unused subplots
        total_cells = grid_rows * grid_cols
        for j in range(len(grid_processed), total_cells):
            ax = fig.add_subplot(right_gs[j // grid_cols, j % grid_cols])
            ax.axis('off')

    # ------------------------------------------------------------------ #
    # Final polish
    # ------------------------------------------------------------------ #
    suptitle = title + " — Renders (left) | Grid Fields (right)" if title else "Render vs Grid Diagnostics"
    fig.suptitle(suptitle, fontsize=18, y=0.98)
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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

        # Backward with graph retention for all but last view
        # CRITICAL: retain_graph=True for intermediate views
        retain_graph = not is_last_view
        view_loss.backward(retain_graph=retain_graph)

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
    skip = False
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

    if iteration >= opt.eikonal_from_iter and opt.lambda_eikonal >= 0:# and False:  # e.g., add to config as 1000
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
            normal_smoothness = cossim_loss_multisurf(nurbs.normal_grids, weight_maps=nurbs.weight_map_grids(), w=normal_global_smoothness_weight)
            total_loss = total_loss + normal_smoothness
            log_dict['Normal Smoothness Loss'] = normal_smoothness.item()

        if normal_global_smoothness_weight > 0.0:
            global_normal_smoothness = cossim_loss_multisurf(nurbs.global_normal_grids, weight_maps=nurbs.weight_map_grids(),w= normal_smoothness_weight)
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
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)
                nearest_render_pkg = render(nearest_cam, nurbs, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                pts = nurbs.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3, :3] + nearest_cam.world_view_transform[
                                                                                      3, :3]
                map_z, d_mask = nurbs.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'],
                                                                        pts_in_nearest_cam)

                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:, 2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[..., None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
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
            safe_run_dict = {k: v.detach().item() if isinstance(v, torch.Tensor) and v.numel() == 1 else v for k, v in run_dict.items() if v > 0}

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

def training(dataset, opt, pipe, args):
    global best_cd
    debug_mode = False
    test_iteration = args.test_iterations
    save_iteration = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations,
    start_iteration = args.start_checkpoint
    # torch.cuda.set_device(str(args.train_gpu))
    first_iter = 0
    scan_id = dataset.model_path.split('/')[-1].split('n')[-1]
    scene_name = os.path.basename(dataset.model_path)  # e.g., scan118
    trial_index = prepare_output_and_logger(dataset)
    device = 'cuda'
    render_components = None
    decomposed_final_img = None
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    app_model = AppModel().cuda()
    app_model.train()
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    scene = SplineScene(dataset, opt, scan_id=scene_name, pipe=pipe, background=background, app_model=app_model)
    nurbs = scene.get_splines()

    if args.start_checkpoint:
        try:
            print(f"Loading checkpoint from {args.start_checkpoint}")
            checkpoint, loaded_iter = torch.load(args.start_checkpoint)
            # model = SplineModel().restore(checkpoint)
            # model = MultiSurfaceSplineModel().restore()
            print(f"Checkpoint has being loaded.Continue optimization from iteration {loaded_iter}")
            first_iter = loaded_iter
            print(f"Resuming training from iteration {first_iter}")
            # nurbs = model
        except Exception as e:
            print(f"\n\n\nFailed with {e}")

            pass
    terminal_width = shutil.get_terminal_size().columns * 3
    first_iter += 1

    # region: Logging and Async Executor Setup
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training progress", ncols=terminal_width)
    wandb_queue,wandb_config, wandb_proc = None, None, None

    if args.use_wandb:
        wandb_queue = Queue(maxsize=100)
        wandb_config = {
            "project": args.wandb_project,
            "name": f"{scene_name}_{trial_index}",
            "group": scene_name,
            "config": vars(opt),
            "entity": "Tzlil"  # Replace with your entity or set to None
        }
        wandb_proc = Process(target=wandb_logger_worker, args=(wandb_queue, wandb_config), daemon=True)
        wandb_proc.start()


    def save_config_args(args):
        """Saves the config arguments to a file in the model's output directory."""
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

    def log_evaluation_results(future: Future):
        """Callback to log results from the evaluation worker."""
        global best_cd
        try:
            result = future.result()
            best_cd = result["best_score"]

            print(result["best"])  # Print all logs from the worker
            if result.get("metric_dict") is not None:
                log_data = {"Evaluation/Chamfer Distance": result["metric_dict"]}
                if args.use_wandb and wandb_queue is not None:
                    wandb_queue.put({"data": log_data, "iteration": result["iteration"]})
                best_cd = result["best_score"]
            else:
                print(f"[Eval Callback] Failure in processing evaluation result: {result['logs']}")
        except Exception as e:
            print(f"[Eval Callback] Error processing evaluation result: {e}")

    images_to_log = {}
    # Executor for running evaluation tasks on a separate process
    ctx = mp.get_context("spawn")
    eval_executor = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    viewpoint_stack = scene.getTrainCameras().copy()
    # Configure batch size
    BATCH_SIZE = opt.batch_size  # Start with 4, adjust based on GPU memory
    USE_BATCHED_TRAINING = BATCH_SIZE > 1
    debug_path = os.path.join(scene.model_path, "debug")
    batched_optimizer = setup_batched_optimizer(nurbs, scene.getTrainCameras().copy())
    # Initialize gradient accumulation
    if USE_BATCHED_TRAINING:
        nurbs.enable_gradient_accumulation(BATCH_SIZE)
    next_camera=0
    cycles_complete=0
    ema_psnr_for_log=0.0
    controller = build_controller(
                                  decomp_depth_views=4,
                                  warmup_iters=opt.densify_from_iter,
                                  min_psnr_to_decompose=15.0,
                                  n_components=2
                                  )

    triggered = False
    for iteration in range(first_iter, opt.iterations + 1):
        # --- Step 1: Periodic aggregation refresh ---
        if (iteration % AGGREGATION_INTERVAL == 1
                and iteration > WARM_UP_ITERATIONS and opt.sampling_strategy == 'adaptive'):
            nurbs.refresh_chhugani_aggregation(
                scene.getTrainCameras().copy(),
                max_views=2,
                aggregation_mode='max',  # Conservative: covers all views
            )
        if iteration % 1000 == 0 and iteration > 0:
            nurbs.oneupSHdegree()

            BATCH_SIZE = max(1, BATCH_SIZE - 1)
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            next_camera = -1 if next_camera == 0 else 0
            cycles_complete += 1
            if cycles_complete % 2 == 0:
                nurbs.densify_sampling_density(quant=0.05)
                # nurbs.update_uv_distribution_chhugani(viewpoint_cam)

        if USE_BATCHED_TRAINING:

            # Take random batch
            batch_cameras = []
            for _ in range(min(BATCH_SIZE, len(viewpoint_stack))):
                # next_camera = random.randint(0, len(viewpoint_stack) - 1)
                batch_cameras.append(viewpoint_stack.pop(next_camera))

            # Process batch with accumulation
            total_loss, log_dict, render_pkg = process_batch_views(
                scene, nurbs, batch_cameras, pipe, background,
                app_model, opt, iteration, dataset, debug_path
            )
            viewpoint_cam = batch_cameras[-1]
        else:
            # next_camera = random.randint(0, len(viewpoint_stack) - 1)

            viewpoint_cam = viewpoint_stack.pop(next_camera)
            total_loss, log_dict, render_pkg = process_view(
                scene, nurbs, viewpoint_cam, pipe, background,
                app_model, opt, iteration, dataset, debug_path
            )
            total_loss.backward()


        # ── 2. Controller update (before forward pass) ───────────────────────
        # NOTE: call update() AFTER the previous iteration's forward() so
        # that surface.ray is populated (required by _record_depth → uv_depth)


        if iteration <= opt.iterations:
            nurbs.optimizer.step()
            app_model.optimizer.step()
            nurbs.optimizer.zero_grad(set_to_none = True)
            app_model.optimizer.zero_grad(set_to_none = True)

        # Periodically: global interval optimization
        if iteration % 2000 == 0 and iteration < 16000:# and opt.sampling_strategy == 'adaptive':
            print(f"\n[Iteration {iteration}] Running batched interval optimization...")

            # This processes ALL surfaces × batch of views in one call
            results = batched_optimizer.optimize(
                render_fn=render,
                pipe=pipe,
                background=background,
                app_model=app_model,
            )
            # Results are automatically stored as surface._global_intervals
            # Per-view Chhugani will blend with these on subsequent forward passes

            print(f"[Iteration {iteration}] Batched interval optimization complete. "
                  f"Optimized {len(results)} surfaces.\n")
            for r, surface in zip(results, nurbs.surfaces):
                surface.uv_sampler.update_intervals_global(*r)


        with ((torch.no_grad())):

            learning_rates = nurbs.update_learning_rate(iteration)

            log_dict['Total Loss'] = total_loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{log_dict['Total Loss']:.4f}", "Points":( nurbs.total_gaussians), "CD": f"{best_cd:.5f}", "Params": {nurbs.parameters_count}})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration % 2000 == 1 and args.use_wandb:
                images_to_log = {}

                # Ground truth and rendered image
                # images_to_log["Renders/GT_Image"] = viewpoint_cam.get_image()[0].detach()
                images_to_log["Renders/Rendered_Image"] = render_pkg['render']

                # Depth visualization
                if 'plane_depth' in render_pkg:
                    depth_vis = depth_to_colormap(render_pkg['plane_depth'])
                    images_to_log["Renders/Depth"] = depth_vis

                # Normal visualization
                if 'rendered_normal' in render_pkg:
                    normal_vis = normal_to_rgb(render_pkg['rendered_normal'])
                    images_to_log["Renders/Rendered_Normal"] = normal_vis

                if 'depth_normal' in render_pkg:
                    depth_normal_vis = normal_to_rgb(render_pkg['depth_normal'])
                    images_to_log["Renders/Depth_Normal"] = depth_normal_vis

            new_psnr_val = psnr(render_pkg['render'], viewpoint_cam.original_image.cuda()).mean().item()
            ema_psnr_for_log = 0.4 * new_psnr_val + 0.6 * ema_psnr_for_log
            log_dict['PSNR'] = ema_psnr_for_log

            if iteration % 1000 == 0 and args.use_wandb:
                for lr_name, lr_value in learning_rates.items():
                    if lr_name == 'background':
                        log_dict['Learning Rate (Background-Position)'] = lr_value
                    else:
                        log_dict['Learning Rate (Object-Position)'] = lr_value


                log_dict['Total Splat Samplings'] = nurbs.total_gaussians
                async_wandb_logger(wandb_queue, prepare_img_log(iteration, images_to_log), log_dict, iteration)
            # if cycles_completed:
            if (iteration in save_iteration):
                print("\n[ITER {}] Saving NURBS-based 3D-Gaussians".format(iteration))
                scene.save(iteration, scan_name=scene_name)

                nurbs_state_dict = nurbs.capture()
                torch.save((nurbs_state_dict, iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


            BASE_MODEL_PATH = f"/sci/labs/sagieb/zlilovadia/{PROJECT_DIR}/base_models/DTU/scan{scan_id}/{trial_index}" # TODO: EXPORT to env var
            if iteration in args.checkpoint_iterations:
                checkpoint_path = os.path.join(scene.model_path, f"chkpnt{iteration}.pth")
                print(f"\n[ITER {iteration}] Saving full training checkpoint to {checkpoint_path}...")
                nurbs_state_dict = nurbs.capture()
                torch.save((nurbs_state_dict, iteration), checkpoint_path)

            if iteration in args.evaluation_iterations:
                print(f"\n[ITER {iteration}] Staging model for evaluation.")
                temp_chk_path = os.path.join(scene.model_path, f"temp_chkpnt_eval_{iteration}.pth")
                paths = {
                    "out_base_path": args.out_base_path,
                    "data_base_path": args.data_base_path,
                    "dtu_eval_path": args.dtu_eval_path,
                }
                future = eval_executor.submit(
                    run_evaluation, iteration, scene_name, trial_index,
                    scan_id, args.eval_gpu, paths, temp_chk_path, args.use_depth_normal, args.use_depth_filter)
                future.add_done_callback(log_evaluation_results)

            if iteration % show_interval == 1:
                to_render = {}
                to_render.update(render_pkg)
                to_render.update(
                    )
                try:
                    to_render.update(render_components)
                    to_render.update({'decomposed_final_img': decomposed_final_img})
                except:
                    pass
                nurbs_prop_dict = nurbs.prepare_grid_for_vis(viewpoint_cam)
                plot_render_outputs(to_render, nurbs_prop_dict, viewpoint_cam.get_image()[0], nurbs, uid=viewpoint_cam.uid, title=f"{scene_name}_{trial_index}")
                del to_render
                del nurbs_prop_dict

            if iteration < opt.densify_until_iter:# and controller.phase == DecompPhase.DECOMPOSED:
                mask = (render_pkg["out_observe"] > 0) & render_pkg['visibility_filter']
                nurbs.add_subdivision_stats(mask,
                                            render_pkg["viewspace_points"],
                                            render_pkg["viewspace_points_abs"],
                                            render_pkg['visibility_filter'],
                                            render_pkg['radii'],)
                if iteration in args.test_iterations:
                    log_qualitative_results(iteration, nurbs, scene, pipe, background, app_model, wandb_queue, dataset,
                                            scene_name=scan_id)

                if not (iteration % opt.densification_interval):
                    size_threshold = opt.abs_split_radii2D_threshold if iteration > opt.opacity_reset_interval else np.inf
                did_change = False
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 1:
                    did_change = nurbs.subdivide_and_cull(
                        max_grad=opt.densify_grad_threshold,
                        grad_abs_threshold=opt.densify_grad_threshold,
                        min_opacity=opt.opacity_cull_threshold,
                        extent=scene.cameras_extent,
                        max_screen_size=size_threshold,
                        top_k_rate_subd=opt.max_k_subdiv,
                        max_prune_rate=opt.max_k_prune,
                        verbose=False
                    )
                    # nurbs.mark_all_aggregations_stale()

                if opt.use_multi_view_trim and iteration % 1000 == 1:
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
                        verbose=False
                    )
                if did_change:
                    controller.reset_depth_buffer()
                    # nurbs.mark_all_aggregations_stale()
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    nurbs.reset_opacity()

            nurbs.update_parameters(iteration)

            # if iteration % WARM_UP_ITERATIONS == 1:
            #     refresh_camera_neighbors_post_warmup(
            #         scene, nurbs, iteration,
            #         warmup_iterations=WARM_UP_ITERATIONS,
            #         num_neighbors=dataset.multi_view_num,
            #         verbose=True
            #     )
            nurbs._invalidate_cache()
            if controller.phase != DecompPhase.DECOMPOSED:
                for surf in nurbs.surfaces:
                    surf.ray_info()  # Ensure ray is populated before depth recording

                triggered = controller.update(iteration, nurbs, viewpoint_cam, ema_psnr_for_log)

                if triggered and controller.phase != DecompPhase.DECOMPOSED:
                    print(f"\n[Train] Decomposing model at iteration {iteration}")

                    # Use safe_decompose instead of controller.decompose directly
                    nurbs = safe_decompose(controller, nurbs, opt, args)

                    # Reset LR (training_setup() already rebuilt optimizer,
                    # but position scheduler needs iteration context)
                    for surface in nurbs.surfaces:
                        surface._last_subdivision_step = iteration
                        surface.iteration = iteration

                    # Update surface offsets (required before add_subdivision_stats)
                    nurbs._update_surface_offsets()
                    batched_optimizer.model = nurbs
                    print(f"[Train] Decomposition complete. New model: {nurbs}. Resuming training.")
            if iteration == 2001:
                refresh_camera_neighbors_post_warmup(
                    scene, nurbs, iteration,
                    warmup_iterations=WARM_UP_ITERATIONS,
                    num_neighbors=dataset.multi_view_num,
                    verbose=True
                )
    # Final cleanup
    print("Shutting down evaluation worker pool...")
    eval_executor.shutdown(wait=True)
    print("Shutting down WandB logger...")
    if args.use_wandb:
        wandb_queue.put(WAND_LOGGER_TERMINATE)
        wandb_proc.join(timeout=10)
    app_model.save_weights(scene.model_path, opt.iterations)
    print("\nTraining complete.")



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


@torch.no_grad()
def log_qualitative_results(iteration, nurbs, scene, pipe, background, app_model, wandb_queue, push_logs=False, args=None, scene_name=None):
    """
    Renders a few fixed views and logs the qualitative results (images, normals, etc.) to WandB.
    This function is analogous to the original training_report_splines.
    """
    # Define which views to log
    test_cameras = scene.getTestCameras()
    train_cameras = scene.getTrainCameras()
    validation_configs = ({'name': 'test', 'cameras': test_cameras},
                          {'name': 'train',
                           'cameras': [train_cameras[idx % len(scene.getTrainCameras())] for idx in
                                       range(5, 30, 5)]})
    # save_training_collage, save_comparison_collage
    output_path = os.path.join(scene.model_path, scene_name)
    #
    # saved_files = save_training_collage(
    #     model=nurbs,
    #     cameras=train_cameras,
    #     render_fn=render,  # Your render function
    #     pipe=pipe,
    #     background=torch.tensor([0, 0, 0], device='cuda'),
    #     output_path=os.path.join(output_path, f'training_collage_{iteration}'),
    #     num_samples=16,
    #     selection_mode='uniform',
    #     add_labels=True,
    #     render_depth=True,
    #     render_normal=False
    # )
    #
    # # For GT vs Render comparison
    # save_comparison_collage(
    #     model=nurbs,
    #     cameras=train_cameras,
    #     render_fn=render,
    #     pipe=pipe,
    #     background=torch.tensor([0, 0, 0], device='cuda'),
    #     output_path=output_path + f'/comparison_collage_iter_{iteration}',
    #     num_samples=8
    # )
    # saved_paths = save_nurbs_surface_maps(
    #     model=nurbs,
    #     output_path=os.path.join(scene.model_path, scene_name),
    #     view_camera=scene.getTrainCameras()[0],  # For view-dependent normals
    #     save_color=True,  # SH DC color map
    #     save_normal=True,  # Surface normals
    #     save_opacity=True,  # Opacity values
    #     save_scaling=True  # Scaling in U, V, N directions
    # )
    for config in validation_configs:
        l1_test = 0.0
        psnr_test = 0.0
        if not config['cameras']:
            continue

        log_images = {}
        for idx, viewpoint in enumerate(config['cameras']):
            render_pkg = render(viewpoint, nurbs, pipe, background, app_model=app_model, return_plane=False,
                                return_depth_normal=False)

            gt_image = viewpoint.get_image()[0].cpu().numpy().transpose(1, 2, 0)

            render_image = render_pkg["render"].clamp(0.0, 1.0).cpu().numpy().transpose(1, 2, 0)

            log_key_base = f"Qualitative/{config['name']}/View_{viewpoint.image_name}"
            log_images[f"{log_key_base}/Ground_Truth"] = wandb.Image(gt_image, caption=f"GT_{viewpoint.image_name}")
            log_images[f"{log_key_base}/Render"] = wandb.Image(render_image, caption=f"Render_{viewpoint.image_name}")

            if "depth_normal" in render_pkg:
                normals = render_pkg["depth_normal"].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
                log_images[f"{log_key_base}/Normals"] = wandb.Image(normals, caption=f"Normals_{viewpoint.image_name}")

            if "plane_depth" in render_pkg:
                depth = render_pkg["plane_depth"].squeeze().cpu().numpy()
                if depth.max() > 0:
                    depth_vis = cv2.applyColorMap((depth / depth.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    log_images[f"{log_key_base}/Depth"] = wandb.Image(depth_vis,
                                                                      caption=f"Depth_{viewpoint.image_name}")
            psnr_val = psnr(render_pkg['render'], viewpoint.get_image()[0]).mean().double()
            l1_test += l1_loss(render_pkg['render'], viewpoint.get_image()[0]).mean().double()
            psnr_test += psnr_val

        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

        if log_images and push_logs and wandb_queue is not None:
            wandb_queue.put({"data": log_images, "step": iteration})


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
    save_iterations = list(range(first_iter, 30000//cycle, eval_interval))
    evals = [15_000, 21_000, 25_000, 30_000]
    evals.append(9_000)
    evals.append(13_000)
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
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--use_cont", action="store_true")
    parser.add_argument("--use_depth_normal", action="store_true")
    scan_id = os.environ.get("SCAN_ID", "")
    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, scan_id)
    args.source_path = os.path.join(args.source_path, scan_id)
    args.use_wandb = True
    print(f"[INFO] Using seed={args.seed} for full determinism")
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    # args.train_gpu = f"cuda:{args.train_gpu}"
    # args.eval_gpu = f"cuda:{args.eval_gpu}"
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={devices}, parsed devices: {devices}")
    if devices and devices[0].isdigit():
        gpu_id = int(devices[0])

    args.eval_gpu = args.train_gpu = f"cuda:{gpu_id}"
    # args.eval_gpu = f"cuda:{0}"
    # torch.cuda.set_device(args.train_gpu)
    # torch.set_num_threads(8)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # torch.autograd.set_detect_anomaly(True)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), args,

    )
"""
Updated training loop integration for batched interval optimization.

Replace the old single-view `optimize_global_intervals` call with the
batched version.
"""


def setup_batched_optimizer(model, training_cameras):
    """Call once at training start or after model structure changes."""
    from modules.IntervalsRefiner import BatchedIntervalOptimizer, BatchConfig


    config = BatchConfig(
        num_steps=100,
        batch_size=2,           # Views per gradient step
        lr=0.05,
        chhugani_weight=0.1,    # Don't stray too far from geometry
        reconstruction_weight=0.1,
        smoothness_weight=0.01,
        grad_clip=1.0,
        warmup_steps=len(training_cameras)//2,
    )

    return BatchedIntervalOptimizer(model, training_cameras, config)
"""
Minimal integration of WarmupDecompositionController into your train.py.

Assumes your train.py looks roughly like:
    model = MultiSurfaceSplineModel.from_pointcloud(...)
    for iteration in range(1, opt.iterations + 1):
        camera = get_camera(...)
        model.forward(camera)
        render_pkg = render(camera, model, ...)
        loss = compute_loss(...)
        loss.backward()
        model.step()
        model.update_learning_rate(iteration)
"""



def build_controller(**kwargs) -> WarmupDecompositionController:
    """Build controller from your existing config/args."""
    cfg = ControllerConfig(
        warmup_iters=getattr(kwargs, 'warmup_iters', 2000),
        n_depth_views=getattr(kwargs, 'decomp_depth_views', 32),
        min_psnr_to_decompose=getattr(kwargs, 'decomp_min_psnr', 20.0),
        n_components=getattr(kwargs, 'n_surface_components', 2),
        # segmentation_mode=
    )
    return WarmupDecompositionController(cfg)




if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()

