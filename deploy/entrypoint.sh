#!/usr/bin/env bash
# Container entrypoint: GPU smoke test, then exec whatever was asked.
set -euo pipefail

echo "── Knot-a-Surface container ──────────────────────────────────────────"

if python - <<'EOF'
import sys, torch
if not torch.cuda.is_available():
    print("  [!] torch sees NO CUDA device — did you pass --gpus all ?")
    sys.exit(1)
print(f"  torch {torch.__version__} | CUDA {torch.version.cuda} "
      f"| {torch.cuda.get_device_name(0)}")
from simple_knn._C import distCUDA2
distCUDA2(torch.rand(512, 3, device="cuda"))
import diff_plane_rasterization, bspline_eval
print("  simple_knn / diff_plane_rasterization / bspline_eval  OK")
EOF
then :; else
    echo "  Continuing anyway (CPU-only shells are fine for inspection)."
fi

if [ ! -d "${DTU_DIR:-/datasets/DTU}/scan24" ]; then
    echo "  [i] DTU not found at ${DTU_DIR:-/datasets/DTU} —"
    echo "      run: bash deploy/get_dtu.sh ${DTU_DIR:-/datasets/DTU}"
fi
echo "──────────────────────────────────────────────────────────────────────"

exec "$@"
