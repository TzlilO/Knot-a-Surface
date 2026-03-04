import os
import sys

python_executable = sys.executable
SCRIPT_DIR = '/sci/labs/sagieb/zlilovadia/nurbs'
TRAIN_PYSCRIPT='train_nurbs.py'
# TRAIN_PYSCRIPT='optimize_nurbs.py'
DBDIR = '/sci/labs/sagieb/zlilovadia/splines/datasets'
scenes = os.environ.get('DTU_SCENES', '24,37,40,55,63,65,69,83,97,105,106,110,114,118,122')
scenes = [int(s) for s in scenes.split(',')]
data_base_path = os.path.join(DBDIR, 'DTU')
out_base_path = os.path.join(SCRIPT_DIR, 'output_dtu')
eval_path = os.path.join(SCRIPT_DIR, 'dtu_eval')
out_name = 'test'
gpu_id = os.environ.get('GPU_ID', '0')
res = os.environ.get('RESOLUTION', '2')
sh_deg = os.environ.get('SH', '1')

# Checkpoint iterations to evaluate
# CHECKPOINTS_LIST = [7000, 15000, 20000, 30000]
CHECKPOINTS_LIST = [7000, 15000, 20000, 30000]

# Total training iterations (should be >= max checkpoint)
TOTAL_ITERATIONS = max(CHECKPOINTS_LIST)


def run_command(cmd, description=None):
    """Run a command and print it."""
    if description:
        print(f"\n{'=' * 60}")
        print(f"[STEP] {description}")
        print(f"{'=' * 60}")
    print(f"[CMD] {cmd}")
    return os.system(cmd)


def evaluate_checkpoint(scene, model_dir, iteration, gpu_id):
    """Run rendering and evaluation for a specific checkpoint."""
    print(f"\n{'#' * 60}")
    print(f"# Evaluating checkpoint at iteration {iteration}")
    print(f"{'#' * 60}")

    # Create iteration-specific output directory for mesh
    mesh_output_dir = os.path.join(model_dir, f"mesh_iter_{iteration}")
    os.makedirs(mesh_output_dir, exist_ok=True)
    print(f"[INFO] Mesh output directory: {mesh_output_dir}")
    # Render command with specific iteration
    render_args = f"--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0 --iteration {iteration}"
    render_cmd = (
        f'CUDA_VISIBLE_DEVICES={gpu_id} {python_executable} {SCRIPT_DIR}/spline_render.py '
        f'-m {model_dir} {render_args}'
    )
    run_command(render_cmd, f"Rendering mesh for iteration {iteration}")

    # The mesh is created in model_dir/mesh/tsdf_fusion_post.ply
    # Move it to iteration-specific directory
    default_mesh_path = os.path.join(model_dir, "mesh", "tsdf_fusion_post.ply")
    iter_mesh_path = os.path.join(mesh_output_dir, "tsdf_fusion_post.ply")

    if os.path.exists(default_mesh_path):
        # Copy mesh to iteration-specific directory
        import shutil
        shutil.copy2(default_mesh_path, iter_mesh_path)

        # Evaluate
        eval_cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} {python_executable} {SCRIPT_DIR}/scripts/eval_dtu/evaluate_single_scene.py "
            f"--input_mesh {iter_mesh_path} "
            f"--scan_id {scene} "
            f"--output_dir {mesh_output_dir} "
            f"--mask_dir {data_base_path} "
            f"--DTU /sci/labs/sagieb/zlilovadia/KnotSurface/datasets/DTU/SampleSet/MVSDATA"
        )
        run_command(eval_cmd, f"Evaluating mesh for iteration {iteration}")
    else:
        print(f"[WARNING] Mesh not found at {default_mesh_path} for iteration {iteration}")

DENSITY = os.environ.get('DENSITY', '1')
def main():
    # Build checkpoint iterations string for training
    checkpoint_iters_str = " ".join(map(str, CHECKPOINTS_LIST))
    save_iters_str = checkpoint_iters_str

    for scene in scenes:
        print(f"\n{'*' * 60}")
        print(f"* Processing Scene: scan{scene}")
        print(f"{'*' * 60}")

        model_dir = f"{out_base_path}/dtu_scan{scene}/{out_name}"

        # Clean previous output
        cmd = f'rm -rf {model_dir}/*'
        run_command(cmd, "Cleaning previous output")

        # Copy sparse data
        cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
        run_command(cmd, "Copying sparse data")

        # Training with checkpoint saving
        common_args = f"--ncc_scale .5 -r{res} --sampling_density {DENSITY}"
        train_cmd = (
            f'{python_executable} {SCRIPT_DIR}/{TRAIN_PYSCRIPT} '
            f'-s {data_base_path}/scan{scene} '
            f'-m {model_dir} '
            f'{common_args} '
            f'--use_wandb '
            f'--train_gpu {gpu_id} '
            f'--sh_degree {sh_deg} '
            f'--iterations {TOTAL_ITERATIONS} '
            f'--checkpoint_iterations {checkpoint_iters_str} '
            f'--save_iterations {save_iters_str} '
            f'--test_iterations {checkpoint_iters_str}'
        )

        run_command(train_cmd, f"Training scan{scene}")

        # Evaluate each checkpoint
        results = {}
        for iteration in CHECKPOINTS_LIST:
            checkpoint_path = os.path.join(model_dir, f"chkpnt{iteration}.pth")

            if os.path.exists(checkpoint_path):
                evaluate_checkpoint(scene, model_dir, iteration, gpu_id)

                # Try to read evaluation results
                result_file = os.path.join(model_dir, f"mesh_iter_{iteration}", "results.txt")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        results[iteration] = f.read().strip()
            else:
                print(f"[WARNING] Checkpoint not found:  {checkpoint_path}")

        # Print summary for this scene
        print(f"\n{'=' * 60}")
        print(f"Results Summary for scan{scene}")
        print(f"{'=' * 60}")
        for iteration, result in results.items():
            print(f"Iteration {iteration}: {result}")

        # Save summary to file
        summary_path = os.path.join(model_dir, "evaluation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Evaluation Summary for scan{scene}\n")
            f.write("=" * 50 + "\n")
            for iteration in CHECKPOINTS_LIST:
                mesh_dir = os.path.join(model_dir, f"mesh_iter_{iteration}")
                result_file = os.path.join(mesh_dir, "results.txt")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as rf:
                        f.write(f"\nIteration {iteration}:\n{rf.read()}\n")
                else:
                    f.write(f"\nIteration {iteration}: No results\n")

        print(f"Summary saved to:  {summary_path}")


if __name__ == "__main__":
    main()