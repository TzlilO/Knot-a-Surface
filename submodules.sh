#!/usr/bin/env bash
# ============================================================
# PGSR Submodule Installation
# Target: TensorDock Ubuntu GPU Server (CUDA 12.1)
# ============================================================
set -euo pipefail

# 1. Activate your existing Conda environment
echo "Activating Conda environment 'ml_env'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_env

# 2. Clone the PGSR repository recursively to fetch the submodules
echo "Cloning the PGSR repository from zju3dv..."
# Using HTTPS instead of SSH to avoid key permission issues on a fresh server
git clone --recursive https://github.com/zju3dv/PGSR.git
cd PGSR

# 3. Build and install the diff-plane-rasterization submodule
echo "Building and installing diff-plane-rasterization..."
pip install ./submodules/diff-plane-rasterization

# 4. Build and install the simple-knn submodule (required by PGSR)
echo "Building and installing simple-knn..."
pip install ./submodules/simple-knn

echo -e "\033[0;32m[Success]\033[0m PGSR submodules installed successfully."