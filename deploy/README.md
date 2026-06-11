# Deploying Knot-a-Surface on a fresh GPU server

One-command path from a bare Ubuntu 22.04 VM (TensorDock or similar) to a
running training setup: OS packages → NVIDIA driver → Docker + NVIDIA
container toolkit → training image (torch 2.5.1 + CUDA 12.1 + all three CUDA
extensions) → DTU dataset → smoke test.

## TL;DR — new server in 3 commands

```bash
# [local Mac] the repo is private and new servers have no GitHub auth →
# ship a git bundle (same trick as deploy_remote.md):
git bundle create /tmp/knots.bundle main
scp /tmp/knots.bundle user@NEW_SERVER:~/

# [server]
git clone -b main ~/knots.bundle ~/Knots && cd ~/Knots
SYNC_FROM=user@40.142.110.216 bash deploy/bootstrap_server.sh
```

That's it. The script is idempotent — if the driver install forces a reboot,
just run it again afterwards. At the end it prints the exact train command.

If the server *does* have GitHub auth, `git clone git@github.com:TzlilO/Knots`
replaces the bundle dance.

## What bootstrap_server.sh does

| Step | Detail | Skipped when |
|---|---|---|
| 1. apt basics | git, wget, rsync, tmux, build-essential | always runs (quiet) |
| 2. NVIDIA driver | `ubuntu-drivers autoinstall` + asks for reboot | `nvidia-smi` already works |
| 3. Docker + nvidia-container-toolkit | official install scripts, `nvidia-ctk runtime configure` | already installed |
| 4. image build | `knots:latest`, arch auto-detected from `nvidia-smi compute_cap` | — |
| 5. DTU dataset | `deploy/get_dtu.sh` → `~/datasets/DTU` + `~/datasets/dtu_eval/MVSDATA` | data already present |
| 6. smoke test | distCUDA2 + both rasterizer/eval extensions on the real GPU | — |

Env-var knobs:

- `SYNC_FROM=user@host` — rsync the DTU data from a machine that already has
  it (the current GPU server). **Fastest and the layout is guaranteed right.**
- `DTU_GDRIVE_URL=...` — public download instead: the 2DGS-preprocessed DTU
  link from the [2DGS README](https://github.com/hbb1/2d-gaussian-splatting)
  ("DTU dataset"). Eval GT comes from DTU's official servers automatically.
- `NO_DOCKER=1` — skip Docker entirely and build a conda env on the host via
  `setup_conda_env.sh` (the path used on the current server). Use this when
  the provider gives you a VM where nested virtualization/Docker is awkward.
- `DATA_DIR` / `OUT_DIR` — default `~/datasets` / `~/output_dtu`.

## The Docker image

`deploy/Dockerfile`, built FROM `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`.

Why this base: torch and nvcc come from the **same** image, so the toolchain
is coherent by construction — the entire class of failures documented in
`deploy_remote.md` (mixed-version conda CUDA packages → `cub/cub.cuh` /
`nv/target` not found) cannot happen. The remaining lessons still apply and
are baked in: extensions are built with `--no-build-isolation`, `CUDA_HOME`
is set, and `TORCH_CUDA_ARCH_LIST` is pinned via build arg (no GPU is visible
during `docker build`, so auto-detection is impossible — the bootstrap script
passes the right arch for you).

Layer order is cache-friendly: python deps → submodules → extension builds →
project code. Editing training code only re-runs the final cheap layers.

```bash
# manual build (from repo root)
docker build -t knots:latest -f deploy/Dockerfile .                       # 30xx+40xx
docker build -t knots:latest --build-arg TORCH_CUDA_ARCH_LIST="9.0" \
    -f deploy/Dockerfile .                                                # H100
```

## Running

```bash
# interactive shell (entrypoint runs the GPU smoke test first)
docker run --gpus all -it \
    -v ~/datasets:/datasets -v ~/output_dtu:/output \
    knots:latest

# train scan24 directly
docker run --gpus all -d --name knots_scan24 \
    -v ~/datasets:/datasets -v ~/output_dtu:/output \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    knots:latest bash -c \
    'SCAN_ID=scan24 python optimize_nurbs.py -s /datasets/DTU -m /output/scan24 \
     -r 2 --ncc_scale 0.5 --use_wandb'
docker logs -f knots_scan24

# live-code development: mount the repo over the baked-in copy
docker run --gpus all -it \
    -v ~/Knots:/workspace/Knots -v ~/datasets:/datasets -v ~/output_dtu:/output \
    knots:latest
# (extensions stay installed in the image's site-packages, so mounting the
#  repo does NOT require rebuilding them unless you edit submodules/*)
```

Outputs land in `~/output_dtu` on the host; eval snapshots (`render|GT`
PNGs every 500 iters) in `<out>/eval_vis/`.

## DTU dataset only

```bash
# from an existing machine (recommended)
SYNC_FROM=user@40.142.110.216 bash deploy/get_dtu.sh ~/datasets

# public download
DTU_GDRIVE_URL='<2DGS DTU drive link>' bash deploy/get_dtu.sh ~/datasets
```

Final layout (what the code expects):

```
~/datasets/DTU/scan24/...            # -s path; SCAN_ID env appended
~/datasets/dtu_eval/MVSDATA/Points   # scripts/eval_dtu.py --DTU
~/datasets/dtu_eval/MVSDATA/ObsMask
```

## Troubleshooting

- **`docker: unknown flag --gpus`** → nvidia-container-toolkit missing; rerun
  bootstrap (step 3) or `sudo nvidia-ctk runtime configure --runtime=docker
  && sudo systemctl restart docker`.
- **smoke test: "torch sees NO CUDA device"** → you forgot `--gpus all`, or
  driver/toolkit mismatch (`nvidia-smi` inside the container must work).
- **extension import error after editing `submodules/*`** → rebuild the image
  (only the extension layer reruns), or inside a live-mounted container:
  `pip install --no-build-isolation ./submodules/<name>`.
- **Google Drive quota on DTU download** → use `SYNC_FROM`, or download the
  zip in a browser and `unzip` to `~/datasets/DTU` manually.
- **conda path instead of Docker** → `NO_DOCKER=1 bash
  deploy/bootstrap_server.sh`; all the conda CUDA-pinning lessons live in
  `setup_conda_env.sh` (never the bare `nvidia` channel; `CUDA_HOME=$CONDA_PREFIX`;
  `--no-build-isolation`).
