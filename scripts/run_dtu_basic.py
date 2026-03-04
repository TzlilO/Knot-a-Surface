import os
import sys

python_executable = sys.executable
# SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
SCRIPT_DIR = '/sci/labs/sagieb/zlilovadia/nurbs'

DBDIR = '/sci/labs/sagieb/zlilovadia/splines/datasets'#/sci/labs/sagieb/zlilovadia/KnotSurface/datasets
scenes = os.environ.get('DTU_SCENES', '24,37,40,55,63,65,69,83,97,105,106,110,114,118,122')
scenes = [int(s) for s in scenes.split(',')]
data_base_path = os.path.join(DBDIR, 'DTU')
out_base_path = os.path.join(SCRIPT_DIR, 'output_dtu')
eval_path     = os.path.join(SCRIPT_DIR, 'dtu_eval')
out_name='test'
gpu_id=os.environ.get('GPU_ID', '0')
res=os.environ.get('RESOLUTION', '2')

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--ncc_scale 0.5 -r{}".format(res)
    cmd = f'{python_executable} {SCRIPT_DIR}/train_nurbs.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args} --use_wandb --train_gpu {gpu_id}'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python_executable} {SCRIPT_DIR}/spline_render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} {python_executable} {SCRIPT_DIR}/scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU /sci/labs/sagieb/zlilovadia/KnotSurface/datasets/DTU/SampleSet/MVSDATA"
    print(cmd)
    os.system(cmd)