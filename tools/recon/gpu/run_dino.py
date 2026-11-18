#!/usr/bin/env python3
"""NOT RUN in this repository (no GPU, no weights, no photographs here). Per-image DINOv2 patch features, PCA-reduced,
written as the feature maps tools/recon/label_points.py reads.

    python3 run_dino.py IMAGES_DIR OUT_DIR [--model dinov2_vits14] [--dim 16] [--long-side 896] [--max-minutes 120]

Model: DINOv2 from torch.hub ('facebookresearch/dinov2', e.g. dinov2_vits14; patch 14). Calls used:
torch.hub.load(repo, name) and model.forward_features(x)['x_norm_patchtokens'] -> (1, h * w, C). These are the entry
points the DINOv2 repository documents; check them against the version torch.hub fetches (the script records the hub
commit it used when torch exposes it, else 'unknown'). DINOv3 is NOT wired in: its access terms and loader differ
(the repo's tools/dtu/features.py runs a DINOv3 ONNX export); add a backend only after checking its API.

Two passes, both resumable:
  1. PCA: patch features of up to --pca-images images (evenly spread), --pca-patches per image at random (seed), L2-
     normalised -> mean and the top --dim directions -> OUT_DIR/pca.npz (skipped when it exists).
  2. per image NAME: features of the image resized so its long side is --long-side (rounded to the patch), L2-normalised
     per patch, projected -> OUT_DIR/NAME.npy, float16, h x w x dim (one cell per patch; label_points.py reads it
     bilinearly between cell centres at any stride). Skipped when it exists; written to a temporary name and renamed.
Long-job rules: --max-minutes wall clock; SIGINT / SIGTERM stop after the current image; a non-finite feature map
stops the run at once and is not written; failures are recorded (stop after --max-failures); OUT_DIR/run_dino.json
holds every setting, the explained variance, per-image sizes and stop {reason, images_done, minutes}."""
import argparse, json, os, signal, time
import numpy as np

STOP = {'sig': None}
def on_sig(s, f): STOP['sig'] = signal.Signals(s).name
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def feats(model, path, a, torch, Image):
    im = Image.open(path).convert('RGB'); W, H = im.size; s = a.long_side / max(W, H)
    w, h = max(1, round(W * s / a.patch)) * a.patch, max(1, round(H * s / a.patch)) * a.patch
    x = (np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(x.transpose(2, 0, 1)[None]).to(a.device)
    with torch.no_grad(): t = model.forward_features(x)['x_norm_patchtokens'][0].float().cpu().numpy()
    f = t.reshape(h // a.patch, w // a.patch, -1); return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('images'); ap.add_argument('out')
    ap.add_argument('--repo', default='facebookresearch/dinov2'); ap.add_argument('--model', default='dinov2_vits14'); ap.add_argument('--patch', type=int, default=14)
    ap.add_argument('--dim', type=int, default=16); ap.add_argument('--long-side', type=int, default=896)
    ap.add_argument('--pca-images', type=int, default=16); ap.add_argument('--pca-patches', type=int, default=4000); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--max-minutes', type=float, default=120.0); ap.add_argument('--max-failures', type=int, default=5)
    a = ap.parse_args(); signal.signal(signal.SIGINT, on_sig); signal.signal(signal.SIGTERM, on_sig); t0 = time.time()
    import torch
    from PIL import Image
    model = torch.hub.load(a.repo, a.model, source='local' if os.path.isdir(a.repo) else 'github').to(a.device).eval()   # a local clone pins the commit
    names = sorted(os.path.relpath(os.path.join(r, f), a.images) for r, _, fs in os.walk(a.images) for f in fs if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')))
    os.makedirs(a.out, exist_ok=True); pf = os.path.join(a.out, 'pca.npz'); rng = np.random.default_rng(a.seed)
    rec = dict(tool='tools/recon/gpu/run_dino.py', torch=torch.__version__, hub=a.repo + ':' + a.model, settings=vars(a), images={}, failures={}); stop = 'done'
    if not os.path.exists(pf):
        pick = [names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(a.pca_images, len(names))).astype(int))]; S = []
        for n in pick:
            f = feats(model, os.path.join(a.images, n), a, torch, Image); f = f.reshape(-1, f.shape[-1]); S.append(f[rng.choice(len(f), min(a.pca_patches, len(f)), replace=False)])
        S = np.concatenate(S); mu = S.mean(0); _, sv, vt = np.linalg.svd(S - mu, full_matrices=False)
        if not np.all(np.isfinite(vt)): raise SystemExit('non-finite PCA: nothing written')
        np.savez(pf, mean=mu, basis=vt[:a.dim], explained=(sv ** 2 / (sv ** 2).sum())[:a.dim], images=np.array(pick))
    P = np.load(pf); mu, B = P['mean'], P['basis']; rec['pca'] = dict(explained_variance=P['explained'].tolist(), images=P['images'].tolist())
    for n in names:
        dst = os.path.join(a.out, n + '.npy')
        if os.path.exists(dst): continue
        if STOP['sig']: stop = 'signal'; break
        if (time.time() - t0) / 60 > a.max_minutes: stop = 'wall_clock'; break
        try:
            f = feats(model, os.path.join(a.images, n), a, torch, Image); r = ((f - mu) @ B.T).astype(np.float16)
            if not np.all(np.isfinite(r)): stop = 'nan'; rec['failures'][n] = 'non-finite features (not written)'; break
            os.makedirs(os.path.dirname(dst), exist_ok=True); tmp = dst + '.tmp.npy'; np.save(tmp, r); os.replace(tmp, dst)
            rec['images'][n] = list(r.shape); print('%s: %s' % (n, r.shape), flush=True)
        except Exception as e:
            rec['failures'][n] = repr(e); print('%s: FAILED %r' % (n, e), flush=True)
            if len(rec['failures']) > a.max_failures: stop = 'failures'; break
    rec['stop'] = dict(reason=stop, images_done=sum(os.path.exists(os.path.join(a.out, n + '.npy')) for n in names), images_total=len(names), minutes=round((time.time() - t0) / 60, 2))
    json.dump(rec, open(os.path.join(a.out, 'run_dino.json'), 'w'), indent=1, default=str); print('stop: %s' % rec['stop'])


if __name__ == '__main__':
    main()
