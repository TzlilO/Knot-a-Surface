# tools/recon/gpu — masks and features on the owner's GPU machine (NOT RUN)

Neither script has been run: this repository has no GPU, no model weights and no photographs. They are short and
conventional, but the calls into the third-party packages are **unverified here** (listed below). Check them against
the installed versions before a long run; both scripts fail at once on an import or API error.

## Setup (pin what you install, and record it)

```bash
python3 -m venv ~/venv-recon && . ~/venv-recon/bin/activate
pip install torch torchvision          # a CUDA build matching your driver; the scripts record torch.__version__
pip install numpy scipy pillow
# Segment Anything (v1): the package from its GitHub repository. Pin the commit you install:
pip install "git+https://github.com/facebookresearch/segment-anything.git@<COMMIT>"
#   record <COMMIT> in your notes; run_sam.py records the package version it sees and the checkpoint's SHA-256
# checkpoint (Meta's release; ViT-H): sam_vit_h_4b8939.pth from dl.fbaipublicfiles.com/segment_anything/
# DINOv2: fetched by torch.hub from facebookresearch/dinov2 on first use (needs network once). To pin it, clone the
#   repository at a fixed commit and pass --repo /path/to/dinov2 (a directory is loaded with source='local');
#   the weights are still downloaded by the hub entry point.
```

No version numbers are given above on purpose: I could not check which versions are current or compatible from here.
Whatever you install, the run records (`run_sam.json`, `run_dino.json`) keep the versions and hashes.

## End to end: a folder of images + a COLMAP model -> per-object charts

```bash
# 0. a COLMAP TEXT model of the same images (colmap model_converter --output_type TXT if it is binary)
# 1. masks, one or several levels per image (resumable; --max-minutes 120)
python3 tools/recon/gpu/run_sam.py IMAGES/ WORK/masks --checkpoint sam_vit_h_4b8939.pth --levels 3
# 2. features (optional; resumable)
python3 tools/recon/gpu/run_dino.py IMAGES/ WORK/feats --model dinov2_vits14 --dim 16
# 3. CPU from here on (numpy + scipy)
python3 -I tools/recon/label_points.py MODEL/ -o WORK/labels --masks WORK/masks --features WORK/feats --ground-plane
python3 -I tools/recon/sfm_spline.py MODEL/ -o WORK/charts --labels WORK/labels/labels.npy --trim [--unit-mm U]
```

The label images are looked up by the image NAME of `images.txt` (`NAME.npy`), so keep IMAGES/ laid out as COLMAP
names them (sub-folders included). Images may be at any resolution: positions are scaled by the label image's size.

## Calls that were not verified here

| script | call | assumption |
| --- | --- | --- |
| run_sam.py | `segment_anything.sam_model_registry[type](checkpoint=...)` | registry keys `vit_h`, `vit_l`, `vit_b` |
| run_sam.py | `SamAutomaticMaskGenerator(sam, points_per_side, pred_iou_thresh, stability_score_thresh, min_mask_region_area)` | keyword names as in the package's README; `min_mask_region_area > 0` needs opencv-python |
| run_sam.py | `.generate(HxWx3 uint8 RGB)` | list of dicts with `segmentation` (bool HxW) and `area` |
| run_dino.py | `torch.hub.load(repo, 'dinov2_vits14', source='github' or 'local')` | patch 14, ImageNet normalisation |
| run_dino.py | `model.forward_features(x)['x_norm_patchtokens']` | shape (1, h*w, C), patches in row order |

## Expected run times

Unknown: not measured. Both scale with the number of images; SAM's automatic generator is the slow one (it runs the
mask decoder on a grid of point prompts per image). Use `--max-minutes` and rerun: finished images are skipped.

## Stop records

Both write `stop = {reason: done | wall_clock | signal | failures | nan, images_done, images_total, minutes}`. A
failing image is recorded under `failures` and skipped; more than `--max-failures` stops the run. A non-finite
feature map stops `run_dino.py` at once and is not written.
