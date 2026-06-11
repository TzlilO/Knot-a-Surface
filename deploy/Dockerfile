# ============================================================================
# Knot-a-Surface — training image
#
# Base: pytorch/pytorch *devel* image = torch 2.5.1 + nvcc 12.1 from the SAME
# build → a coherent CUDA toolchain by construction. This sidesteps the conda
# channel-mixing failure mode entirely (the cub/cccl header chaos that
# setup_conda_env.sh works around only matters when you assemble the
# toolchain yourself).
#
# Build (no GPU needed — arches are pinned via build arg):
#   docker build -t knots:latest -f deploy/Dockerfile .
#   docker build -t knots:latest --build-arg TORCH_CUDA_ARCH_LIST="8.9" -f deploy/Dockerfile .
#
# Run (see deploy/README.md):
#   docker run --gpus all -it -v ~/datasets:/datasets -v ~/output:/output knots:latest
# ============================================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Compute capabilities to compile for. No GPU is visible at build time, so we
# can't auto-detect — default covers RTX 30xx (8.6) and 40xx (8.9).
# 7.5=T4/RTX20xx, 8.0=A100, 8.6=RTX30xx/A10, 8.9=RTX40xx/L4, 9.0=H100.
ARG TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

# System deps: build tools for the extensions, GL/X libs for open3d/opencv,
# plus the basics for working inside the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl unzip rsync ninja-build build-essential \
        libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 libegl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (mirror setup_conda_env.sh §4) ──────────────────────
RUN pip install --no-cache-dir \
        numpy scipy scikit-image scikit-learn matplotlib cycler Pillow \
        opencv-python imageio imageio-ffmpeg \
        geomdl rhino3dm trimesh plyfile open3d \
        opt_einsum wandb joblib attrs pyyaml tqdm packaging ninja gdown

WORKDIR /workspace/Knots

# ── CUDA extensions (LESSONS: --no-build-isolation, CUDA_HOME set above) ────
# Copy only the submodules first so code-only edits don't bust this layer.
COPY submodules/ ./submodules/
RUN pip install --no-build-isolation --no-cache-dir \
        ./submodules/simple-knn \
        ./submodules/diff-plane-rasterization \
        ./submodules/bspline-eval

# pytorch3d (Chamfer eval only) — tolerated failure, same as setup_conda_env.sh
RUN FORCE_CUDA=1 pip install --no-build-isolation --no-cache-dir \
        "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || echo "WARNING: pytorch3d build failed — training works; only Chamfer eval needs it"

# ── Project code (changes most often → last layer) ──────────────────────────
COPY . .

# Import-only sanity check (CUDA calls need a GPU, deferred to entrypoint).
RUN python -c "import simple_knn, diff_plane_rasterization, bspline_eval; print('extensions import OK')"

# Conventional mount points
RUN mkdir -p /datasets /output
ENV DTU_DIR=/datasets/DTU
ENV OUTPUT_DIR=/output

ENTRYPOINT ["bash", "deploy/entrypoint.sh"]
CMD ["bash"]
