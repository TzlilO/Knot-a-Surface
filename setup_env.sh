#!/usr/bin/env bash
# ============================================================
# Knot-a-Surface — environment setup script
#
# Usage:
#   ./setup_env.sh               # auto-detect CUDA / MPS / CPU
#   ./setup_env.sh --cuda        # force CUDA path (Linux cluster)
#   ./setup_env.sh --mps         # force MPS  path (Apple Silicon)
#   ./setup_env.sh --cpu         # force CPU-only  (CI / no GPU)
#   ./setup_env.sh --rebuild-ext # re-build CUDA extensions only
#
# After the script completes, activate with:
#   source .venv/bin/activate
# ============================================================
set -euo pipefail

# ── helpers ─────────────────────────────────────────────────
RED='\033[0;31m'; GRN='\033[0;32m'; YEL='\033[1;33m'; RST='\033[0m'
info()  { echo -e "${GRN}[setup]${RST} $*"; }
warn()  { echo -e "${YEL}[setup]${RST} $*"; }
die()   { echo -e "${RED}[setup] ERROR:${RST} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── argument parsing ─────────────────────────────────────────
TARGET=""
REBUILD_EXT=0
for arg in "$@"; do
  case "$arg" in
    --cuda)        TARGET=cuda  ;;
    --mps)         TARGET=mps   ;;
    --cpu)         TARGET=cpu   ;;
    --rebuild-ext) REBUILD_EXT=1 ;;
    *) die "Unknown argument: $arg" ;;
  esac
done

# ── auto-detect target ───────────────────────────────────────
if [[ -z "$TARGET" ]]; then
  if command -v nvcc &>/dev/null || (python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null | grep -q True) || \
     [[ "$(uname -s)" == "Linux" ]]; then
    # On Linux assume CUDA is intended; verify below
    if command -v nvcc &>/dev/null; then
      TARGET=cuda
    elif [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
      TARGET=mps
    else
      TARGET=cpu
    fi
  elif [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
    TARGET=mps
  else
    TARGET=cpu
  fi
fi

info "Target: ${TARGET}"

# ── Python interpreter ───────────────────────────────────────
# Prefer python3.10; fall back to whatever python3 is available.
PYTHON=""
for py in python3.10 python3.11 python3.9 python3; do
  if command -v "$py" &>/dev/null; then
    PYTHON="$py"
    break
  fi
done
[[ -z "$PYTHON" ]] && die "No python3 found in PATH"
PY_VER=$($PYTHON --version 2>&1)
info "Using: $PYTHON ($PY_VER)"

# ── virtual environment ──────────────────────────────────────
VENV="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
  info "Creating virtual environment at .venv …"
  $PYTHON -m venv "$VENV"
else
  info "Reusing existing .venv"
fi

PIP="$VENV/bin/pip"
PYTHON_VENV="$VENV/bin/python"

# setuptools<82 keeps torch 2.12 happy (it declares requires setuptools<82);
# install packaging early so _torch_meets_min can use it.
"$PIP" install --quiet --upgrade pip wheel packaging
"$PIP" install --quiet "setuptools<82"

# ── PyTorch ──────────────────────────────────────────────────────────────────
# Minimum acceptable version.  If the venv already has >= this, we skip.
# For fresh CUDA installs we target TORCH_CUDA_VER; MPS/CPU get the latest
# PyPI wheel (no pin — arm64/macOS wheel availability is narrow).
TORCH_MIN_VER="2.6.0"
TORCH_CUDA_VER="2.7.1"   # latest stable with cu118/cu121/cu124 wheels

_torch_meets_min() {
  "$PYTHON_VENV" -c "
from packaging.version import Version
import subprocess, sys
r = subprocess.run(['$PIP', 'show', 'torch'], capture_output=True, text=True)
for line in r.stdout.splitlines():
    if line.startswith('Version:'):
        v = line.split(':',1)[1].strip()
        sys.exit(0 if Version(v) >= Version('${TORCH_MIN_VER}') else 1)
sys.exit(1)
" 2>/dev/null
}

install_pytorch_cuda() {
  local cu_tag="${1:-cu118}"       # cu118 | cu121 | cu124
  info "Installing PyTorch ${TORCH_CUDA_VER} (${cu_tag}) …"
  "$PIP" install --quiet \
    "torch==${TORCH_CUDA_VER}" \
    torchvision \
    torchaudio \
    --index-url "https://download.pytorch.org/whl/${cu_tag}"
}

install_pytorch_mps() {
  info "Installing PyTorch (latest macOS/MPS wheel, Python 3.11) …"
  "$PIP" install --quiet torch torchvision torchaudio
}

install_pytorch_cpu() {
  info "Installing PyTorch (CPU-only, latest) …"
  "$PIP" install --quiet \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/cpu"
}

if _torch_meets_min; then
  INSTALLED_TORCH=$("$PYTHON_VENV" -c "import torch; print(torch.__version__)" 2>/dev/null)
  info "PyTorch ${INSTALLED_TORCH} already installed (>= ${TORCH_MIN_VER}) — skipping"
else
  case "$TARGET" in
    cuda)
      # Detect CUDA version from nvcc if available, default to 11.8
      if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if   (( CUDA_MAJOR == 12 && CUDA_MINOR >= 4 )); then CU_TAG="cu124"
        elif (( CUDA_MAJOR == 12 )); then CU_TAG="cu121"
        elif (( CUDA_MAJOR == 11 && CUDA_MINOR >= 8 )); then CU_TAG="cu118"
        elif (( CUDA_MAJOR == 11 )); then CU_TAG="cu118"  # best available for 11.x
        else warn "Unknown CUDA ${CUDA_VER}, defaulting to cu118"; CU_TAG="cu118"
        fi
      else
        warn "nvcc not found — assuming CUDA 11.8 wheels"
        CU_TAG="cu118"
      fi
      install_pytorch_cuda "$CU_TAG"
      ;;
    mps)  install_pytorch_mps  ;;
    cpu)  install_pytorch_cpu  ;;
  esac
fi

# ── core scientific stack ────────────────────────────────────
info "Installing core scientific packages …"
"$PIP" install --quiet \
  numpy \
  scipy \
  scikit-image \
  scikit-learn \
  matplotlib \
  cycler \
  Pillow \
  opencv-python \
  imageio \
  imageio-ffmpeg

# ── geometry / 3-D libs ──────────────────────────────────────
info "Installing geometry packages …"
"$PIP" install --quiet \
  geomdl \
  rhino3dm \
  trimesh \
  plyfile \
  open3d

# ── PyTorch3D — optional, used only for chamfer_distance in eval scripts ─────
# Live modules no longer import pytorch3d (replaced by local quaternion_utils),
# but eval/DTU scripts still call pytorch3d.loss.chamfer.
# Skip on CPU-only to avoid the painful source build.
install_pytorch3d() {
  info "Installing pytorch3d …"
  # Pre-built wheels for pytorch3d are hosted on the fair-internal conda channel
  # and on PyPI for specific torch/CUDA combos only.  We try PyPI first; if that
  # fails we build from source (requires CUDA toolkit + gcc).
  "$PIP" install --quiet pytorch3d 2>/dev/null && return 0

  warn "PyPI wheel unavailable — building pytorch3d from source (this takes ~10 min) …"
  "$PIP" install --quiet "git+https://github.com/facebookresearch/pytorch3d.git"
}

case "$TARGET" in
  cuda) install_pytorch3d ;;
  mps)
    warn "pytorch3d has no MPS support; skipping install."
    warn "  → chamfer_distance in eval scripts will fail locally."
    warn "  → Live training (modules/) does NOT need pytorch3d."
    ;;
  cpu)
    warn "pytorch3d skipped on CPU-only target."
    ;;
esac

# ── ML / training utilities ──────────────────────────────────
info "Installing ML/training utilities …"
"$PIP" install --quiet \
  opt_einsum \
  wandb \
  joblib \
  attrs \
  pyyaml \
  tqdm

# ── LPIPS (vendored as lpipsPyTorch/) — ensure torchvision is present ────────
# lpipsPyTorch is bundled; no pip install needed, but it imports torchvision.
"$PYTHON_VENV" -c "import torchvision" 2>/dev/null \
  || "$PIP" install --quiet torchvision

# ── CUDA extensions (simple-knn + diff-plane-rasterization) ──────────────────
build_cuda_extensions() {
  if [[ "$TARGET" != "cuda" ]]; then
    info "Skipping CUDA extension build (target=${TARGET})"
    return 0
  fi

  command -v nvcc &>/dev/null || { warn "nvcc not found — skipping CUDA extensions"; return 0; }

  # Verify torch+CUDA are consistent before compiling
  "$PYTHON_VENV" -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available in installed torch'
print('  torch CUDA:', torch.version.cuda)
"

  for ext_dir in \
    "$SCRIPT_DIR/submodules/simple-knn" \
    "$SCRIPT_DIR/submodules/diff-plane-rasterization"; do
    ext_name=$(basename "$ext_dir")
    info "Building CUDA extension: ${ext_name} …"
    (
      cd "$ext_dir"
      "$PYTHON_VENV" setup.py build_ext --inplace --quiet 2>&1 \
        | grep -E "error:|warning:|built" || true
      "$PIP" install --quiet -e . 2>&1 | tail -3
    )
    info "  ✓ ${ext_name} installed"
  done
}

if [[ "$REBUILD_EXT" -eq 1 ]]; then
  build_cuda_extensions
else
  # Only build if not already importable
  HAVE_EXTS=1
  "$PYTHON_VENV" -c "import simple_knn; import diff_plane_rasterization" 2>/dev/null \
    || HAVE_EXTS=0
  if [[ "$HAVE_EXTS" -eq 0 ]]; then
    build_cuda_extensions
  else
    info "CUDA extensions already importable — skipping build (use --rebuild-ext to force)"
  fi
fi

# ── MPS-specific: warn about ops not yet supported ───────────────────────────
if [[ "$TARGET" == "mps" ]]; then
  warn "MPS notes:"
  warn "  • simple_knn / diff_plane_rasterization are CUDA-only."
  warn "    conftest_stubs.py provides CPU fallbacks for the test suite."
  warn "  • To run the full training loop locally, pass --device cpu to avoid"
  warn "    MPS ops that are not yet stable (e.g. torch.cdist on MPS in <2.2)."
  warn "  • Use --include_eval only on the cluster (pytorch3d unavailable here)."
fi

# ── smoke test ───────────────────────────────────────────────
info "Running smoke test …"
"$PYTHON_VENV" - <<'PYEOF'
import sys, importlib

failures = []

def check(label, fn):
    try:
        fn()
        print(f"  ✓ {label}")
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        failures.append(label)

check("torch",             lambda: __import__("torch"))
check("torchvision",       lambda: __import__("torchvision"))
check("numpy",             lambda: __import__("numpy"))
check("scipy",             lambda: __import__("scipy"))
check("cv2",               lambda: __import__("cv2"))
check("PIL",               lambda: __import__("PIL"))
check("matplotlib",        lambda: __import__("matplotlib"))
check("geomdl",            lambda: __import__("geomdl"))
check("opt_einsum",        lambda: __import__("opt_einsum"))
check("plyfile",           lambda: __import__("plyfile"))
check("yaml",              lambda: __import__("yaml"))
check("wandb",             lambda: __import__("wandb"))
check("joblib",            lambda: __import__("joblib"))
check("trimesh",           lambda: __import__("trimesh"))
check("open3d",            lambda: __import__("open3d"))
check("rhino3dm",          lambda: __import__("rhino3dm"))
check("skimage",           lambda: __import__("skimage"))
check("sklearn",           lambda: __import__("sklearn"))
check("imageio",           lambda: __import__("imageio"))
check("tqdm",              lambda: __import__("tqdm"))
check("attrs",             lambda: __import__("attr"))

import torch
check("torch.cuda / mps",  lambda: print(
    f"CUDA={torch.cuda.is_available()}, "
    f"MPS={getattr(torch.backends, 'mps', type('',(),{'is_available':lambda:False})).is_available()}"
))

# CUDA extensions (best-effort)
try:
    import simple_knn
    print("  ✓ simple_knn (CUDA)")
except ImportError:
    print("  ~ simple_knn not built (CUDA target only)")
try:
    import diff_plane_rasterization
    print("  ✓ diff_plane_rasterization (CUDA)")
except ImportError:
    print("  ~ diff_plane_rasterization not built (CUDA target only)")

if failures:
    print(f"\nFailed: {failures}", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll checks passed.")
PYEOF

# ── unit tests ───────────────────────────────────────────────
info "Running CPU unit tests …"
"$PYTHON_VENV" -m pytest tests/ -q --tb=short 2>&1 \
  || warn "Some unit tests failed — check output above"

# ── done ─────────────────────────────────────────────────────
echo ""
info "Done.  Activate with:  source .venv/bin/activate"
info "Cluster training:      python optimize_nurbs.py --source_path <data> --use_wandb --include_eval"
info "Local unit tests:      .venv/bin/python -m pytest tests/ -q"
