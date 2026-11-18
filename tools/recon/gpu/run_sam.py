#!/usr/bin/env python3
"""NOT RUN in this repository (no GPU, no weights, no photographs here). Per-image automatic masks with Segment Anything,
written as the label images tools/recon/label_points.py reads.

    python3 run_sam.py IMAGES_DIR OUT_DIR --checkpoint sam_vit_h_4b8939.pth [--model-type vit_h] [--levels 3]
                       [--max-minutes 120] [--device cuda]

Assumes the package of facebookresearch/segment-anything (pip install from its GitHub repository; see README.md for
the pin) and one of its released checkpoints. Calls used: sam_model_registry[model_type](checkpoint=...),
SamAutomaticMaskGenerator(sam, ...).generate(rgb_uint8_HxWx3) -> list of dicts with 'segmentation' (H x W bool),
'area', 'predicted_iou', 'stability_score'. These are the package's documented entry points; check them against the
installed version before a long run (the script stops at once with the import error otherwise).

Output per image NAME (the name images.txt uses, sub-folders kept): OUT_DIR/NAME.npy, uint16 H x W x L.
Level k holds a set of masks that do not overlap one another (above --overlap of the smaller mask): masks are taken
largest first and each goes to the first level where it fits, so level 0 holds the coarse parts and later levels the
finer ones inside them (SAM's multi-granularity output as a hierarchy). 0 = no mask; ids are per image (1..n) and mean
nothing across images. At most --levels levels; a mask that fits none is dropped and counted.

Long-job rules: resumable (an image whose .npy exists is skipped; files are written to a temporary name and renamed);
--max-minutes wall clock; SIGINT / SIGTERM stop after the current image; a failing image is recorded and skipped (the
run stops after --max-failures); OUT_DIR/run_sam.json records every setting, the package version, the checkpoint's
SHA-256, per-image counts and stop {reason, images_done, minutes}."""
import argparse, hashlib, json, os, signal, sys, time
import numpy as np

STOP = {'sig': None}
def on_sig(s, f): STOP['sig'] = signal.Signals(s).name


def levels(masks, H, W, n_levels, overlap):
    out = np.zeros((H, W, n_levels), np.uint16); nxt = np.ones(n_levels, int); dropped = 0
    for m in sorted(masks, key=lambda d: -d['area']):
        seg = m['segmentation']; a = max(int(seg.sum()), 1)
        for k in range(n_levels):
            if (out[..., k][seg] > 0).sum() / a <= overlap:
                out[..., k][seg & (out[..., k] == 0)] = nxt[k]; nxt[k] += 1; break
        else: dropped += 1
    used = int((nxt > 1).sum()); return out[..., :max(used, 1)], dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('images'); ap.add_argument('out')
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    ap.add_argument('--levels', type=int, default=3); ap.add_argument('--overlap', type=float, default=0.05, help='a mask fits a level when at most this share of it is already taken there')
    ap.add_argument('--points-per-side', type=int, default=32); ap.add_argument('--pred-iou-thresh', type=float, default=0.88); ap.add_argument('--stability-score-thresh', type=float, default=0.95)
    ap.add_argument('--min-mask-region-area', type=int, default=100); ap.add_argument('--device', default='cuda')
    ap.add_argument('--max-minutes', type=float, default=120.0); ap.add_argument('--max-failures', type=int, default=5)
    a = ap.parse_args(); signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig); t0 = time.time()
    from PIL import Image
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    try:
        from importlib.metadata import version; pkg = version('segment_anything')
    except Exception: pkg = 'unknown'
    sha = hashlib.sha256(open(a.checkpoint, 'rb').read()).hexdigest()
    sam = sam_model_registry[a.model_type](checkpoint=a.checkpoint).to(a.device)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=a.points_per_side, pred_iou_thresh=a.pred_iou_thresh,
                                    stability_score_thresh=a.stability_score_thresh, min_mask_region_area=a.min_mask_region_area)
    names = sorted(os.path.relpath(os.path.join(r, f), a.images) for r, _, fs in os.walk(a.images) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')))
    rec = dict(tool='tools/recon/gpu/run_sam.py', package='segment_anything %s' % pkg, torch=torch.__version__, checkpoint=os.path.basename(a.checkpoint), checkpoint_sha256=sha,
               settings={k: v for k, v in vars(a).items()}, images={}, failures={}); stop = 'done'
    for name in names:
        dst = os.path.join(a.out, name + '.npy')
        if os.path.exists(dst): continue
        if STOP['sig']: stop = 'signal'; break
        if (time.time() - t0) / 60 > a.max_minutes: stop = 'wall_clock'; break
        try:
            rgb = np.array(Image.open(os.path.join(a.images, name)).convert('RGB'))
            with torch.no_grad(): masks = gen.generate(rgb)
            L, dropped = levels(masks, rgb.shape[0], rgb.shape[1], a.levels, a.overlap)
            os.makedirs(os.path.dirname(dst), exist_ok=True); tmp = dst + '.tmp.npy'; np.save(tmp, L); os.replace(tmp, dst)
            rec['images'][name] = dict(masks=len(masks), levels=int(L.shape[2]), dropped=dropped, covered=float((L[..., 0] > 0).mean()))
            print('%s: %d masks, %d levels, %d dropped' % (name, len(masks), L.shape[2], dropped), flush=True)
        except Exception as e:                                        # one bad image is recorded, not fatal
            rec['failures'][name] = repr(e); print('%s: FAILED %r' % (name, e), flush=True)
            if len(rec['failures']) > a.max_failures: stop = 'failures'; break
    done = sum(os.path.exists(os.path.join(a.out, n + '.npy')) for n in names)
    rec['stop'] = dict(reason=stop, images_done=done, images_total=len(names), minutes=round((time.time() - t0) / 60, 2))
    os.makedirs(a.out, exist_ok=True); old = os.path.join(a.out, 'run_sam.json')
    if os.path.exists(old):                                           # keep earlier runs' per-image records
        prev = json.load(open(old)); prev['images'].update(rec['images']); rec['images'] = prev['images']; rec['previous_stops'] = prev.get('previous_stops', []) + [prev.get('stop')]
    json.dump(rec, open(old, 'w'), indent=1); print('stop: %s' % rec['stop'])


if __name__ == '__main__':
    main()
