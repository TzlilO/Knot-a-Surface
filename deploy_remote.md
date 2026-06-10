# Remote GPU server deployment (conda)

Server: `ssh user@40.142.110.216` — RTX 3090 (sm_86), Ubuntu 22.04, driver 580.
Env: conda `ml_env` — Python 3.10, **torch 2.5.1+cu121**.

## Why the submodule builds failed (and the fixes)

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` during `pip install .` | PEP-517 build isolation hides the env's torch | `pip install --no-build-isolation` |
| `CUDA_HOME environment variable is not set` | nvcc lives inside the conda env, not `/usr/local/cuda` | `export CUDA_HOME=$CONDA_PREFIX` |
| `fatal error: cub/cub.cuh: No such file` / `nv/target: No such file` | conda env had **mixed CUDA packages** (12.1 nvcc + 12.6 compiler meta + 13.3 cccl/cudart-static). CUDA 13's cccl installs headers under `include/cccl/`, invisible to nvcc 12.1; 13.3 runtime headers shadowed the 12.1 ones | Remove all `cuda-*`/`libcu*`/`libnv*` packages, reinstall coherently from the version-pinned label: `conda install -c "nvidia/label/cuda-12.1.1" cuda-toolkit` |

Rule of thumb: **never** install conda CUDA packages from the bare `nvidia`
channel (it resolves each package to its own latest version); always pin via
`nvidia/label/cuda-X.Y.Z` matching `torch.version.cuda`.

## Full procedure (fresh server)

```bash
# 0. Repo is private and the server has no GitHub credentials → ship a bundle:
#    [local]
git bundle create /tmp/knots.bundle <branch>
scp /tmp/knots.bundle user@40.142.110.216:~/
#    [remote]
git clone -b <branch> ~/knots.bundle ~/Knots

# 1. Coherent CUDA toolchain matching torch's build (here 12.1):
conda activate ml_env
conda remove -y --force $(conda list | grep -E "^(cuda|libnv|libcub|libcuf|libcur|libcus|libnpp)" | awk '{print $1}')
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit

# 2. Build env vars (every build shell needs these):
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"        # RTX 3090; 8.9=4090, 9.0=H100

# 3. CUDA extensions (sources are tracked in the Knots repo, glm included):
cd ~/Knots
pip install ninja
pip install --no-build-isolation ./submodules/simple-knn
pip install --no-build-isolation ./submodules/diff-plane-rasterization

# 4. Python deps:
pip install numpy scipy scikit-image scikit-learn matplotlib cycler Pillow \
    opencv-python imageio imageio-ffmpeg geomdl rhino3dm trimesh plyfile \
    open3d opt_einsum wandb joblib attrs pyyaml tqdm packaging

# 5. pytorch3d (needed by scene/gaussian_model.py and Chamfer eval), from source:
FORCE_CUDA=1 pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 6. Verify:
python tests/test_basis_math.py
python tests/test_position_rational.py
python tests/test_rotation_frame.py
python -c "import optimize_nurbs"
```

PGSR reference clone lives at `~/PGSR` (upstream submodule sources, not used
for the install — Knots tracks its own copies).
