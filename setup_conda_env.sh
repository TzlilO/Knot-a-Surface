#!/usr/bin/env bash
# ============================================================================
# Knot-a-Surface — conda environment setup (Linux + NVIDIA GPU)
#
# Creates a complete training environment including the three CUDA
# extensions (simple-knn, diff-plane-rasterization, bspline-eval).
#
# Usage:
#   bash setup_conda_env.sh [ENV_NAME]        # default: knots
#
# Encodes the compatibility lessons learned the hard way:
#   1. NEVER install conda CUDA packages from the bare `nvidia` channel —
#      it resolves each package to its own latest version and you end up
#      with nvcc 12.1 + cccl 13.3 headers in CUDA-13 layout that nvcc
#      cannot find (fatal error: cub/cub.cuh, nv/target). ALWAYS pin via
#      `nvidia/label/cuda-X.Y.Z` matching torch.version.cuda exactly.
#   2. pip's PEP-517 build isolation hides the env's torch from extension
#      builds (ModuleNotFoundError: torch) — use --no-build-isolation.
#   3. nvcc lives INSIDE the conda env — export CUDA_HOME=$CONDA_PREFIX.
#   4. Pin TORCH_CUDA_ARCH_LIST to the actual GPU to avoid building all
#      architectures (8.6=RTX30xx, 8.9=RTX40xx, 9.0=H100).
# ============================================================================
set -euo pipefail

ENV_NAME="${1:-knots}"
PYTHON_VER="3.10"
TORCH_VER="2.5.1"
CUDA_LABEL="12.1.1"          # must match the torch build's CUDA version
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

GRN='\033[0;32m'; YEL='\033[1;33m'; RST='\033[0m'
info() { echo -e "${GRN}[setup]${RST} $*"; }
warn() { echo -e "${YEL}[setup]${RST} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. conda env ────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  info "env '${ENV_NAME}' exists — reusing"
else
  info "creating conda env '${ENV_NAME}' (python ${PYTHON_VER})"
  conda create -y -n "$ENV_NAME" "python=${PYTHON_VER}"
fi
conda activate "$ENV_NAME"

# ── 2. coherent CUDA toolkit (LESSON 1: version-pinned label) ──────────────
info "installing CUDA toolkit ${CUDA_LABEL} from pinned label channel"
# Remove any previously-installed mixed-version cuda packages first
MIXED=$(conda list 2>/dev/null | grep -E "^(cuda|libcub|libcuf|libcur|libcus|libnv|libnpp)" | awk '{print $1}' | tr '\n' ' ')
if [ -n "$MIXED" ]; then
  warn "removing previously installed cuda packages: $MIXED"
  conda remove -y --force $MIXED || true
fi
conda install -y -c "nvidia/label/cuda-${CUDA_LABEL}" cuda-toolkit

# ── 3. PyTorch matching the toolkit ─────────────────────────────────────────
info "installing torch ${TORCH_VER}+cu${CUDA_LABEL%%.*}${CUDA_LABEL#*.}"
pip install --quiet "torch==${TORCH_VER}" torchvision torchaudio \
  --index-url "$TORCH_INDEX"

python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no CUDA device"
print(f"  torch {torch.__version__} | CUDA build {torch.version.cuda} "
      f"| device {torch.cuda.get_device_name(0)}")
EOF

# ── 4. python dependencies ─────────────────────────────────────────────────
info "installing python dependencies"
pip install --quiet \
  numpy scipy scikit-image scikit-learn matplotlib cycler Pillow \
  opencv-python imageio imageio-ffmpeg \
  geomdl rhino3dm trimesh plyfile open3d \
  opt_einsum wandb joblib attrs pyyaml tqdm packaging ninja gdown

# ── 5. CUDA extensions (LESSONS 2-4) ────────────────────────────────────────
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  # Auto-detect compute capability of GPU 0
  TORCH_CUDA_ARCH_LIST=$(python -c \
    "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))")
  export TORCH_CUDA_ARCH_LIST
fi
info "building CUDA extensions for arch ${TORCH_CUDA_ARCH_LIST} (CUDA_HOME=$CUDA_HOME)"

for ext in simple-knn diff-plane-rasterization bspline-eval; do
  info "  building $ext ..."
  pip install --no-build-isolation --quiet "./submodules/$ext"
done

# ── 6. optional: pytorch3d (Chamfer eval only; slow source build) ───────────
if python -c "import pytorch3d" 2>/dev/null; then
  info "pytorch3d already present"
else
  warn "building pytorch3d from source (~10 min; needed only for Chamfer eval)"
  FORCE_CUDA=1 pip install --no-build-isolation --quiet \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    || warn "pytorch3d build failed — training works without it; only scripts/eval_dtu needs it"
fi

# ── 7. smoke test ───────────────────────────────────────────────────────────
info "running smoke test"
python - <<'EOF'
import torch
from simple_knn._C import distCUDA2
d = distCUDA2(torch.rand(512, 3, device="cuda"))
print(f"  simple_knn          OK (mean d2 {d.mean():.4f})")
import diff_plane_rasterization
print("  diff_plane_raster   OK")
import bspline_eval
out = bspline_eval._C  # noqa
print("  bspline_eval        OK")
for m in ("geomdl", "opt_einsum", "open3d", "wandb", "cv2", "trimesh"):
    __import__(m)
print("  python deps         OK")
EOF

info "running repo unit tests"
python tests/test_basis_math.py >/dev/null && echo "  basis math      PASS"
python tests/test_position_rational.py >/dev/null && echo "  position/NURBS  PASS"
python tests/test_rotation_frame.py >/dev/null && echo "  rotation frame  PASS"
python tests/test_fused_eval.py | tail -1

echo ""
info "done. activate with:  conda activate ${ENV_NAME}"
info "train:  SCAN_ID=scan24 python optimize_nurbs.py -s <DTU_DIR> -m <OUT> -r 2 --ncc_scale 0.5"
