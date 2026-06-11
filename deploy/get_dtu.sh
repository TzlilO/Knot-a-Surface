#!/usr/bin/env bash
# ============================================================================
# Download + extract the DTU training data (2DGS-preprocessed, 15 scans) and
# the official DTU evaluation ground truth (SampleSet/MVSDATA + Points).
#
# Usage:
#   bash deploy/get_dtu.sh [TARGET_DIR]          # default: ~/datasets
#
# Two sources, tried in order:
#   1. SYNC_FROM — rsync from a machine that already has the data (fastest,
#      and guaranteed to be the exact layout the code expects). E.g.:
#        SYNC_FROM=user@40.142.110.216 bash deploy/get_dtu.sh
#   2. Public download — 2DGS-preprocessed DTU from Google Drive (gdown) and
#      eval GT from DTU's roboimagedata server. If Google Drive rate-limits
#      or the folder moves, grab the link from the 2DGS README
#      (https://github.com/hbb1/2d-gaussian-splatting → "DTU dataset") and
#      pass it via DTU_GDRIVE_URL.
#
# Final layout (what optimize_nurbs.py / scripts/eval_dtu.py expect):
#   <TARGET>/DTU/scan24/{images,sparse,...}      ← -s flag + SCAN_ID
#   <TARGET>/dtu_eval/MVSDATA/{Points,ObsMask}   ← --DTU flag of eval_dtu
# ============================================================================
set -euo pipefail

TARGET="${1:-$HOME/datasets}"
DTU_DIR="$TARGET/DTU"
EVAL_DIR="$TARGET/dtu_eval/MVSDATA"
SYNC_FROM="${SYNC_FROM:-}"
DTU_GDRIVE_URL="${DTU_GDRIVE_URL:-}"   # 2DGS README → DTU dataset link

GRN='\033[0;32m'; RST='\033[0m'
info() { echo -e "${GRN}[dtu]${RST} $*"; }

mkdir -p "$TARGET"

# ── Source 1: rsync from an existing machine ────────────────────────────────
if [ -n "$SYNC_FROM" ]; then
    info "syncing from $SYNC_FROM (training data + eval GT)"
    rsync -avz --progress "$SYNC_FROM:~/datasets/DTU/"  "$DTU_DIR/"
    rsync -avz --progress "$SYNC_FROM:~/datasets/dtu_eval/MVSDATA/" "$EVAL_DIR/"
    info "done."
    exit 0
fi

# ── Source 2a: 2DGS-preprocessed training data (Google Drive) ───────────────
if [ -d "$DTU_DIR/scan24" ]; then
    info "training data already present at $DTU_DIR — skipping"
else
    if [ -z "$DTU_GDRIVE_URL" ]; then
        echo "ERROR: set DTU_GDRIVE_URL to the 2DGS 'DTU dataset' Google-Drive"
        echo "       link (see https://github.com/hbb1/2d-gaussian-splatting#dataset)"
        echo "       or use SYNC_FROM=user@host to copy from an existing server."
        exit 1
    fi
    info "downloading 2DGS-preprocessed DTU via gdown"
    python -m pip install --quiet gdown
    mkdir -p "$DTU_DIR"
    if [[ "$DTU_GDRIVE_URL" == *"/folders/"* ]]; then
        gdown --folder --fuzzy "$DTU_GDRIVE_URL" -O "$DTU_DIR"
    else
        gdown --fuzzy "$DTU_GDRIVE_URL" -O "$TARGET/dtu.zip"
        unzip -q "$TARGET/dtu.zip" -d "$TARGET"
        rm "$TARGET/dtu.zip"
        # The zip may extract as DTU/ or dtu/ or with one extra level — normalize.
        [ -d "$TARGET/dtu" ] && mv "$TARGET/dtu" "$DTU_DIR"
    fi
    info "training data → $DTU_DIR"
fi

# ── Source 2b: official eval GT (stable DTU servers) ────────────────────────
if [ -d "$EVAL_DIR/Points" ] && [ -d "$EVAL_DIR/ObsMask" ]; then
    info "eval GT already present at $EVAL_DIR — skipping"
else
    info "downloading DTU eval GT (SampleSet ~6.3GB + Points ~6GB)"
    mkdir -p "$EVAL_DIR"
    wget -c -O "$TARGET/SampleSet.zip" \
        "http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip"
    wget -c -O "$TARGET/Points.zip" \
        "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
    unzip -q -o "$TARGET/SampleSet.zip" -d "$TARGET/dtu_eval_raw"
    unzip -q -o "$TARGET/Points.zip"    -d "$TARGET/dtu_eval_raw"
    # eval_dtu expects MVSDATA/{ObsMask,Points}
    rsync -a "$TARGET/dtu_eval_raw/SampleSet/MVS Data/ObsMask/" "$EVAL_DIR/ObsMask/" 2>/dev/null || \
        rsync -a "$TARGET/dtu_eval_raw/MVS Data/ObsMask/" "$EVAL_DIR/ObsMask/"
    rsync -a "$TARGET/dtu_eval_raw/Points/" "$EVAL_DIR/Points/"
    rm -rf "$TARGET/dtu_eval_raw" "$TARGET/SampleSet.zip" "$TARGET/Points.zip"
    info "eval GT → $EVAL_DIR"
fi

info "all done. Layout:"
info "  training: $DTU_DIR/scanXX"
info "  eval GT:  $EVAL_DIR/{Points,ObsMask}"
