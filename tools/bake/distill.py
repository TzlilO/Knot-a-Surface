"""Step 2 of the feature bake: DINOv3 on the captured views -> one feature per control point.

  python3 tools/bake/distill.py <capture dir> <model.onnx> [scene ...]

For every view the image goes through DINOv3 (ViT-S/16); each visible pixel knows its chart and
(u, v), so its patch feature (bilinearly interpolated between patch centres, L2-normalised) is added
to the control points around that (u, v). Control points no view saw take their neighbours' value.
The scene's features are then reduced by PCA and written, 8 bits per number, to
lab/features/<scene>.js. Needs numpy, pillow and onnxruntime; the weights are not in this repo
(e.g. onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX on Hugging Face, DINOv3 licence)."""
import base64, json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

DIM, PATCH, MODEL = 16, 16, 'DINOv3 ViT-S/16'
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def index(t, g, n, periodic):
    """parameter value(s) -> fractional control index, from the Greville sites g of the n control points"""
    if periodic:
        return ((t - g[0]) * n) % n
    return np.interp(t, g, np.arange(n))


def fill(F, W, pu, pv):
    """control points no view saw: take the mean of seen neighbours, repeatedly"""
    seen = W > 1e-6
    F = np.where(seen[..., None], F / np.maximum(W, 1e-6)[..., None], 0)
    for _ in range(400):
        if seen.all() or not seen.any():
            break
        acc, cnt = np.zeros_like(F), np.zeros(seen.shape)
        for ax, per, sh in ((1, pu, 1), (1, pu, -1), (0, pv, 1), (0, pv, -1)):
            f, s = np.roll(F, sh, ax), np.roll(seen, sh, ax)
            if not per:                       # no wrap across an open edge
                e = [slice(None)] * 2; e[ax] = 0 if sh == 1 else -1; s[tuple(e)] = False
            acc += f * s[..., None]; cnt += s
        new = (~seen) & (cnt > 0)
        F[new] = acc[new] / cnt[new][:, None]; seen = seen | new
    return F, seen.any()


def main():
    cap, model, scenes = sys.argv[1], sys.argv[2], sys.argv[3:]
    sess = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lab', 'features')
    for scn in scenes or sorted(os.listdir(cap)):
        meta = json.load(open(os.path.join(cap, scn, 'meta.json')))
        res, w, h, st = meta['res'], meta['idw'], meta['idh'], meta['stride']
        g = res // PATCH
        acc = [np.zeros((c['nv'], c['nu'], 384)) for c in meta['charts']]
        wgt = [np.zeros((c['nv'], c['nu'])) for c in meta['charts']]
        for i in range(meta['views']):
            stem = os.path.join(cap, scn, 'v%02d' % i)
            img = np.asarray(Image.open(stem + '.png').convert('RGB'), np.float32) / 255
            x = ((img - MEAN) / STD).transpose(2, 0, 1)[None]
            tok = sess.run(None, {'pixel_values': x})[0][0]
            f = tok[-g * g:].reshape(g, g, -1)                 # patch tokens (after the class and register tokens)
            f = f / np.linalg.norm(f, axis=-1, keepdims=True)
            ids = np.frombuffer(open(stem + '.ids', 'rb').read(), np.float32).reshape(h, w, 3)[::-1]   # rows were bottom-up
            ys, xs = np.nonzero(ids[..., 2] > 0.5)
            px, py = (xs * st + 0.5) / PATCH - 0.5, (ys * st + 0.5) / PATCH - 0.5      # in patch-centre units
            x0, y0 = np.clip(np.floor(px).astype(int), 0, g - 2), np.clip(np.floor(py).astype(int), 0, g - 2)
            tx, ty = np.clip(px - x0, 0, 1)[:, None], np.clip(py - y0, 0, 1)[:, None]
            pf = (f[y0, x0] * (1 - tx) + f[y0, x0 + 1] * tx) * (1 - ty) + (f[y0 + 1, x0] * (1 - tx) + f[y0 + 1, x0 + 1] * tx) * ty
            chart = np.rint(ids[ys, xs, 2]).astype(int) - 1
            for k, c in enumerate(meta['charts']):
                m = chart == k
                if not m.any():
                    continue
                fu = index(ids[ys[m], xs[m], 0], np.array(c['gU']), c['nu'], c['periodicU'])
                fv = index(ids[ys[m], xs[m], 1], np.array(c['gV']), c['nv'], c['periodicV'])
                i0, j0 = np.floor(fu).astype(int), np.floor(fv).astype(int)
                a, b = fu - i0, fv - j0
                for di, wa in ((0, 1 - a), (1, a)):
                    for dj, wb in ((0, 1 - b), (1, b)):
                        ii = (i0 + di) % c['nu'] if c['periodicU'] else np.minimum(i0 + di, c['nu'] - 1)
                        jj = (j0 + dj) % c['nv'] if c['periodicV'] else np.minimum(j0 + dj, c['nv'] - 1)
                        np.add.at(acc[k], (jj, ii), pf[m] * (wa * wb)[:, None]); np.add.at(wgt[k], (jj, ii), wa * wb)
            print('%s view %d/%d' % (scn, i + 1, meta['views']), end='\r')
        feats, unseen = [], 0
        for k, c in enumerate(meta['charts']):
            unseen += int((wgt[k] <= 1e-6).sum())
            F, ok = fill(acc[k], wgt[k], c['periodicU'], c['periodicV'])
            feats.append(F if ok else None)
        fit = np.concatenate([F.reshape(-1, 384) for F, c in zip(feats, meta['charts']) if F is not None and not c['lock']])
        mean = fit.mean(0)
        feats = [np.broadcast_to(mean, (c['nv'], c['nu'], 384)) if F is None else F for F, c in zip(feats, meta['charts'])]
        _, s, vt = np.linalg.svd(fit - mean, full_matrices=False)
        co = [(F.reshape(-1, 384) - mean) @ vt[:DIM].T for F in feats]
        top = (fit - mean) @ vt[:3].T
        scale = max(np.abs(c).max() for c in co) / 127
        data = {'model': MODEL, 'dim': DIM, 'views': meta['views'], 'res': res,
                'explained': round(float((s[:DIM] ** 2).sum() / (s ** 2).sum()), 4), 'scale': float(scale),
                'lo': [float(v) for v in np.percentile(top, 2, axis=0)], 'hi': [float(v) for v in np.percentile(top, 98, axis=0)],
                'charts': {c['name']: {'nu': c['nu'], 'nv': c['nv'],
                                       'data': base64.b64encode(np.rint(x / scale).astype(np.int8).tobytes()).decode()} for c, x in zip(meta['charts'], co)}}
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, scn + '.js'), 'w') as fh:
            fh.write('/* Generated by tools/bake (capture.js + distill.py): %s features of this scene, one per control point,\n'
                     '   reduced to %d dimensions by PCA. Do not edit; re-run the bake when the scene\'s control nets change. */\n'
                     'window.KnotLab = window.KnotLab || {}; KnotLab.features = KnotLab.features || {};\n'
                     'KnotLab.features[%s] = %s;\n' % (MODEL, DIM, json.dumps(scn), json.dumps(data, separators=(',', ':'))))
        print('%s: %d control points, %d never seen (filled from neighbours), PCA-%d keeps %.1f %% of the variance'
              % (scn, sum(c['nu'] * c['nv'] for c in meta['charts']), unseen, DIM, 100 * data['explained']))


if __name__ == '__main__':
    main()
