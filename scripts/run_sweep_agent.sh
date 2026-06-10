#!/usr/bin/env bash
# Run a W&B sweep agent for the NURBS hyperparameter sweep.
#
# Usage:  bash scripts/run_sweep_agent.sh <SWEEP_ID> [GPU_ID] [SCAN]
#         (SWEEP_ID as printed by `wandb sweep configs/sweep_nurbs.yaml`,
#          e.g. entity/project/abc123)
set -euo pipefail

SWEEP_ID="${1:?usage: run_sweep_agent.sh <SWEEP_ID> [GPU_ID] [SCAN]}"
GPU_ID="${2:-0}"
SCAN="${3:-scan24}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export SCAN_ID="$SCAN"
export DTU_DATA_PATH="${DTU_DATA_PATH:-$HOME/datasets/DTU}"
# Unique output dir per agent process; optimize_nurbs.py appends $SCAN_ID.
export SWEEP_OUT_PATH="${SWEEP_OUT_PATH:-$HOME/output_dtu/sweep_$$}"

echo "[sweep] agent on GPU $GPU_ID, scan $SCAN, out $SWEEP_OUT_PATH"
wandb agent "$SWEEP_ID"
