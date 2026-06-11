#!/usr/bin/env bash
# ============================================================================
# Knot-a-Surface — fresh-server bootstrap (TensorDock-style Ubuntu 22.04 VM)
#
# Takes a brand-new GPU server from bare OS to ready-to-train:
#   1. system packages + NVIDIA driver (if missing)
#   2. Docker + NVIDIA container toolkit          (default path)
#      — or miniconda + setup_conda_env.sh        (NO_DOCKER=1)
#   3. project code (this repo — run FROM the repo root after cloning/bundling)
#   4. DTU dataset (training data + eval GT) via deploy/get_dtu.sh
#   5. build the training image / conda env, run the GPU smoke test
#
# Usage (on the new server, after getting the code there — see README):
#   bash deploy/bootstrap_server.sh
#   NO_DOCKER=1 bash deploy/bootstrap_server.sh        # conda directly on host
#   SYNC_FROM=user@old-server bash deploy/bootstrap_server.sh   # rsync DTU
#
# Idempotent: every step checks before it acts; safe to re-run after a
# failure or a driver-install reboot.
# ============================================================================
set -euo pipefail

GRN='\033[0;32m'; YEL='\033[1;33m'; RST='\033[0m'
info() { echo -e "${GRN}[bootstrap]${RST} $*"; }
warn() { echo -e "${YEL}[bootstrap]${RST} $*"; }

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/datasets}"
OUT_DIR="${OUT_DIR:-$HOME/output_dtu}"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ── 1. base system packages ─────────────────────────────────────────────────
info "installing base packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git wget curl unzip rsync htop tmux build-essential ca-certificates

# ── 2. NVIDIA driver (TensorDock images usually ship one — check first) ─────
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    info "NVIDIA driver OK: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
else
    warn "no working NVIDIA driver — installing recommended driver"
    $SUDO apt-get install -y -qq ubuntu-drivers-common
    $SUDO ubuntu-drivers autoinstall
    warn "driver installed — REBOOT REQUIRED. Run this script again afterwards:"
    warn "    sudo reboot   &&   bash deploy/bootstrap_server.sh"
    exit 0
fi

if [ "${NO_DOCKER:-0}" = "1" ]; then
    # ── conda-on-host path ──────────────────────────────────────────────────
    if ! command -v conda >/dev/null; then
        info "installing miniconda"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
        bash /tmp/mc.sh -b -p "$HOME/miniconda3"
        eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
        conda init bash
    fi
    info "creating env via setup_conda_env.sh (encodes all CUDA-pinning lessons)"
    bash "$REPO_DIR/setup_conda_env.sh" knots
    RUN_PREFIX="conda run -n knots"
else
    # ── 3. Docker + NVIDIA container toolkit ────────────────────────────────
    if ! command -v docker >/dev/null; then
        info "installing Docker"
        curl -fsSL https://get.docker.com | $SUDO sh
        $SUDO usermod -aG docker "$USER" || true
    fi
    if ! docker info 2>/dev/null | grep -qi nvidia && ! [ -f /etc/docker/daemon.json ] || \
       ! grep -q nvidia /etc/docker/daemon.json 2>/dev/null; then
        info "installing NVIDIA container toolkit"
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
            $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq nvidia-container-toolkit
        $SUDO nvidia-ctk runtime configure --runtime=docker
        $SUDO systemctl restart docker
    fi

    # ── 4. build the image for THIS machine's GPU ───────────────────────────
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    info "building knots:latest for compute capability ${ARCH}"
    $SUDO docker build -t knots:latest \
        --build-arg TORCH_CUDA_ARCH_LIST="$ARCH" \
        -f "$REPO_DIR/deploy/Dockerfile" "$REPO_DIR"
    RUN_PREFIX="$SUDO docker run --gpus all --rm -v $DATA_DIR:/datasets -v $OUT_DIR:/output -v $REPO_DIR:/workspace/Knots knots:latest"
fi

# ── 5. DTU dataset ───────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR" "$OUT_DIR"
if [ -d "$DATA_DIR/DTU/scan24" ]; then
    info "DTU already at $DATA_DIR/DTU"
else
    info "fetching DTU (SYNC_FROM='${SYNC_FROM:-}' DTU_GDRIVE_URL='${DTU_GDRIVE_URL:-}')"
    SYNC_FROM="${SYNC_FROM:-}" DTU_GDRIVE_URL="${DTU_GDRIVE_URL:-}" \
        bash "$REPO_DIR/deploy/get_dtu.sh" "$DATA_DIR"
fi

# ── 6. smoke test ────────────────────────────────────────────────────────────
info "running GPU smoke test"
if [ "${NO_DOCKER:-0}" = "1" ]; then
    conda run -n knots python -c "
import torch; from simple_knn._C import distCUDA2
distCUDA2(torch.rand(512,3,device='cuda'))
import diff_plane_rasterization, bspline_eval
print('smoke test OK on', torch.cuda.get_device_name(0))"
else
    $SUDO docker run --gpus all --rm knots:latest python -c "
import torch; from simple_knn._C import distCUDA2
distCUDA2(torch.rand(512,3,device='cuda'))
import diff_plane_rasterization, bspline_eval
print('smoke test OK on', torch.cuda.get_device_name(0))"
fi

info "DONE. Train with:"
if [ "${NO_DOCKER:-0}" = "1" ]; then
    echo "  conda activate knots"
    echo "  SCAN_ID=scan24 python optimize_nurbs.py -s $DATA_DIR/DTU -m $OUT_DIR/run1 -r 2 --ncc_scale 0.5"
else
    echo "  docker run --gpus all -it -v $DATA_DIR:/datasets -v $OUT_DIR:/output knots:latest \\"
    echo "    bash -c 'SCAN_ID=scan24 python optimize_nurbs.py -s /datasets/DTU -m /output/run1 -r 2 --ncc_scale 0.5'"
fi
