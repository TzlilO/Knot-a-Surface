#!/usr/bin/env python3
"""Bake DINOv3 features onto the control points of a fitted DTU scan.

    python3 -I tools/dtu/features.py <data-dir> 122 <model.onnx>             -> talk/dtu/features122.js
    options: [--charts talk/dtu-data.js] [-o FILE] [--step 2] [--size 1024x768] [--cache DIR] [--check DIR --seed name:i,j ...]

The features come from a PRETRAINED model (DINOv3 ViT-S/16, Meta) run on the scan's own photographs. They are
lifted onto the control points of the charts that fit.py made, offline. Nothing here is trained, and nothing here
is a result of the thesis method.

What it does:
  1. reads the fitted charts from the scan's data file (talk/dtu-data.js for 122, talk/dtu/scan<N>.js otherwise) and
     finds the pose of that scene in the scanner's frame from the camera centres the file carries (rigid fit)
  2. runs the model on every photograph, resized to --size (a multiple of the 16 px patch; ImageNet mean / std)
  3. for every stored control point: its node on the surface (the spline at the point's Greville site) is projected
     into each photograph that sees it, by the rule fit.py uses for colour (front-on, in front of the fitted
     surface by a z-buffer of it, and in agreement with the scan's depth map); the patch features, L2-normalised,
     are read bilinearly between patch centres and averaged with weight cos^2 of the viewing angle
  4. nodes no view saw take their neighbours' value
  5. PCA over the stored nodes of all charts -> 16 numbers per node; the few nodes that stand out from their
     5 x 5 neighbourhood (one or two views, a patch of something else) take its median -> 8 bits each

Output: the format of lab/features/<scene>.js (tools/bake/distill.py), registered as KnotTalk.dtuFeatures[N], with one
addition: a chart stored at every `step`-th control point carries `step`, and its nu, nv are the stored grid's size
(stored node (a, b) = control point (a * step, b * step); read the points between bilinearly).

Needs numpy, scipy, Pillow, onnxruntime and fit.py next to this file. The weights are NOT in this repo: an ONNX
export of facebook/dinov3-vits16-pretrain-lvd1689m, e.g. onnx/model.onnx + model.onnx_data of the Hugging Face repo
onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX (DINOv3 License). About five minutes on four cores."""
import argparse, base64, json, os, re, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit                                                    # cameras, depth cleaning, spline evaluation, hole filling

DIM, PATCH, MODEL = 16, 16, 'DINOv3 ViT-S/16'
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
log = fit.log


def read_charts(path):
    """The data file fit.py wrote -> its object, with P decoded the way the file's own loader does it."""
    m = re.search(r'= (\{"scan".*\});\n', open(path).read())
    d = json.loads(m.group(1))
    for c in d['charts']:
        nu, nv = c['nu'], c['nv']
        if 'plane' in c:
            x0, x1, z0, z1 = c['plane']; X, Z = np.meshgrid(np.linspace(x0, x1, nu), np.linspace(z0, z1, nv))
            c['P'] = np.stack([X, np.zeros_like(X), Z], -1)
        else:
            q = np.frombuffer(base64.b64decode(c['P']), '<u2').reshape(nv, nu, 3).astype(float)
            c['P'] = np.array(c['lo']) + (np.array(c['hi']) - np.array(c['lo'])) * q / 65535
    return d


def pose(d, S):
    """Scene units -> scanner millimetres: X = R (unit_mm * p) + t, from the camera centres both sides know."""
    a = np.array(d['cams']) * d['unit_mm']; b = np.array([-R.T @ t for K, R, t in S['cams']])
    ca, cb = a.mean(0), b.mean(0); U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T; t = cb - R @ ca; res = np.linalg.norm(a @ R.T + t - b, axis=1)
    log('scene pose from %d camera centres: residual rms %.3f mm, max %.3f mm' % (len(a), np.sqrt((res ** 2).mean()), res.max()))
    return lambda p: (np.asarray(p) * d['unit_mm']) @ R.T + t, R


def encode(sess, path, size):
    """One photograph -> (rows, cols, 384) patch features, each of unit length."""
    w, h = size; im = Image.open(path).convert('RGB'); full = im.size
    x = ((np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    tok = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]; gw, gh = w // PATCH, h // PATCH
    f = tok[-gw * gh:].reshape(gh, gw, -1)                    # patch tokens come after the class and register tokens
    return f / np.linalg.norm(f, axis=-1, keepdims=True), full


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('scan', type=int); ap.add_argument('model')
    ap.add_argument('--charts', help='the scan\'s data file (default: talk/dtu-data.js for 122, else talk/dtu/scan<N>.js)')
    ap.add_argument('-o', '--out'); ap.add_argument('--step', type=int, default=2, help='store every step-th control point of every chart')
    ap.add_argument('--size', default='1024x768', help='what each photograph is resized to, WxH, multiples of 16')
    ap.add_argument('--cache', help='directory that keeps the patch features of every photograph between runs')
    ap.add_argument('--check', help='directory for check images'); ap.add_argument('--seed', action='append', default=[], help='name:i,j = a similarity map from control point (i, j) of the first chart')
    a = ap.parse_args(); talk = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk')
    a.charts = a.charts or (os.path.join(talk, 'dtu-data.js') if a.scan == 122 else os.path.join(talk, 'dtu', 'scan%d.js' % a.scan))
    a.out = a.out or os.path.join(talk, 'dtu', 'features%d.js' % a.scan); size = tuple(int(v) for v in a.size.split('x'))
    assert size[0] % PATCH == 0 and size[1] % PATCH == 0, 'size must be a multiple of %d' % PATCH
    import onnxruntime as ort
    d = read_charts(a.charts); assert d['scan'] == a.scan, 'the data file is scan %d' % d['scan']
    S = fit.load(os.path.join(a.data, 'scan%d' % a.scan)); log('scan %d: %d views; charts %s' % (a.scan, S['n'], ', '.join('%s %d x %d' % (c['name'], c['nu'], c['nv']) for c in d['charts'])))
    fit.clean_depth(S); world, R = pose(d, S)

    # ---- the nodes: where each stored control point sits on the surface, and which way the surface faces there
    st = a.step; nodes = []; occ = None
    for c in d['charts']:
        nu, nv = c['nu'], c['nv']; G = fit.evaluate(c['P'], fit.greville(nu), fit.greville(nv))
        if 'plane' in c: N = np.broadcast_to(np.array([0., 1, 0]), G.shape)
        else:
            N = fit.Sheet.normals(G)                          # S_u x S_v: out of the object
            occ = world(fit.evaluate(c['P'], np.linspace(0, 1, nu * 3), np.linspace(0, 1, nv * 3)).reshape(-1, 3))
        G, N = G[::st, ::st], N[::st, ::st]
        nodes.append(dict(name=c['name'], shape=G.shape[:2], W=world(G.reshape(-1, 3)), N=N.reshape(-1, 3) @ R.T, net=[nu, nv]))
    W = np.vstack([n['W'] for n in nodes]); Wn = np.vstack([n['N'] for n in nodes])
    log('%d stored nodes (every %s control point)' % (len(W), {1: '', 2: '2nd', 3: '3rd'}.get(st, '%dth' % st)))

    # ---- the model on every photograph; each node reads the views that see it
    opt = ort.SessionOptions(); opt.log_severity_level = 3
    sess = ort.InferenceSession(a.model, opt, providers=['CPUExecutionProvider'])
    acc = np.zeros((len(W), 384)); wgt = np.zeros(len(W)); nseen = np.zeros(len(W), int); gw, gh = size[0] // PATCH, size[1] // PATCH
    for i in range(S['n']):
        K, Rc, t = S['cams'][i]; view = -Rc.T @ t - W; cosv = np.einsum('pc,pc->p', view, Wn) / np.linalg.norm(view, axis=1)
        # z-buffer of the fitted object in the depth map's frame, at a quarter of its resolution (as fit.colours)
        h = occ @ (K @ Rc).T + K @ t; zo = h[:, 2]; uo = (h[:, 0] / zo / 4).astype(int); vo = (h[:, 1] / zo / 4).astype(int); ok = (uo >= 0) & (uo < 400) & (vo >= 0) & (vo < 300)
        zb = np.full((300, 400), np.inf); np.minimum.at(zb, (vo[ok], uo[ok]), zo[ok]); zb = ndimage.minimum_filter(zb, 3)
        h = W @ (K @ Rc).T + K @ t; z = h[:, 2]; u = h[:, 0] / z; v = h[:, 1] / z; vis = (u >= 1) & (u < 1598) & (v >= 1) & (v < 1198)
        ui = np.clip(u.round().astype(int), 0, 1599); vi = np.clip(v.round().astype(int), 0, 1199)
        vis &= (cosv > 0.2) & (z < zb[np.clip(vi // 4, 0, 299), np.clip(ui // 4, 0, 399)] + 2.5)
        dm = S['depth'][i][vi, ui]; vis &= (dm <= 0) | (np.abs(dm - z) < 2.0)
        cf = a.cache and os.path.join(a.cache, 'patches_%dx%d_%04d.npz' % (size[0], size[1], i))
        if cf and os.path.exists(cf): z_ = np.load(cf); f, full = z_['f'].astype(np.float32), tuple(z_['full'])
        else:
            f, full = encode(sess, os.path.join(S['dir'], 'images', '%04d.png' % i), size)
            if cf: os.makedirs(a.cache, exist_ok=True); np.savez(cf, f=f.astype(np.float16), full=np.array(full))
        pk, pr, pt = S['photo'][i]; h = W @ (pk @ pr).T + pk @ pt; x = h[:, 0] / h[:, 2]; y = h[:, 1] / h[:, 2]
        vis &= (x >= 0) & (x <= full[0]) & (y >= 0) & (y <= full[1])
        k = np.nonzero(vis)[0]; px = x[k] * size[0] / full[0] / PATCH - 0.5; py = y[k] * size[1] / full[1] / PATCH - 0.5      # in patch-centre units
        x0 = np.clip(np.floor(px).astype(int), 0, gw - 2); y0 = np.clip(np.floor(py).astype(int), 0, gh - 2)
        tx = np.clip(px - x0, 0, 1)[:, None]; ty = np.clip(py - y0, 0, 1)[:, None]
        pf = (f[y0, x0] * (1 - tx) + f[y0, x0 + 1] * tx) * (1 - ty) + (f[y0 + 1, x0] * (1 - tx) + f[y0 + 1, x0 + 1] * tx) * ty
        wt = cosv[k] ** 2; acc[k] += pf * wt[:, None]; wgt[k] += wt; nseen[k] += 1
        if i % 8 == 7 or i == S['n'] - 1: log('view %d / %d' % (i + 1, S['n']))

    # ---- per chart: mean over the views, fill what nobody saw, then one PCA for all
    feats, o, unseen, total = [], 0, 0, len(W)
    for n in nodes:
        nv, nu = n['shape']; m = nu * nv; seen = (wgt[o:o + m] > 1e-6).reshape(nv, nu); unseen += int((~seen).sum()); n['unseen'] = float((~seen).mean())
        F = np.where(seen[..., None], acc[o:o + m].reshape(nv, nu, -1) / np.maximum(wgt[o:o + m], 1e-6).reshape(nv, nu, 1), 0.0)
        feats.append(fit.diffuse(F, seen, iters=2000) if not seen.all() else F); n['views'] = float(nseen[o:o + m].mean()); o += m
    X = np.concatenate([F.reshape(-1, 384) for F in feats]); mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False); var = s ** 2 / (s ** 2).sum()
    co = [(F.reshape(-1, 384) - mean) @ vt[:DIM].T for F in feats]; top = (X - mean) @ vt[:3].T
    # a few nodes, seen by one or two views only, read a patch of something else: a node far from the median of its
    # 5 x 5 neighbourhood (by more than half the mean feature length) takes that median
    size_ = np.mean([np.linalg.norm(c, axis=1).mean() for c in co]); speck = 0
    for k, n in enumerate(nodes):
        g = co[k].reshape(n['shape'] + (DIM,)); med = np.stack([ndimage.median_filter(g[..., c], 5, mode='nearest') for c in range(DIM)], -1)
        bad = np.linalg.norm(g - med, axis=-1) > 0.5 * size_; g[bad] = med[bad]; speck += int(bad.sum()); co[k] = g.reshape(-1, DIM)
    scale = max(np.abs(c).max() for c in co) / 127; q = [np.rint(c / scale).astype(np.int8) for c in co]
    data = {'scan': a.scan, 'model': MODEL, 'weights': 'facebook/dinov3-vits16-pretrain-lvd1689m (pretrained by Meta; DINOv3 License), run from an ONNX export',
            'note': 'features of a pretrained model, baked onto the control points offline by tools/dtu/features.py; not trained here',
            'dim': DIM, 'views': S['n'], 'res': size[0], 'size': list(size),
            'explained': round(float(var[:DIM].sum()), 4), 'variance': [round(float(v), 4) for v in var[:DIM]], 'unseen': round(unseen / total, 4), 'despeckled': speck, 'scale': float(scale),
            'lo': [float(v) for v in np.percentile(top, 2, axis=0)], 'hi': [float(v) for v in np.percentile(top, 98, axis=0)],
            'charts': {n['name']: {'nu': n['shape'][1], 'nv': n['shape'][0], 'step': st, 'net': n['net'], 'data': base64.b64encode(x.tobytes()).decode()} for n, x in zip(nodes, q)}}
    js = HEAD % dict(scan=a.scan, model=MODEL, dim=DIM, step=st) + json.dumps(data, separators=(',', ':')) + ';\n})();\n'
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh: fh.write(js)
    log('%d stored nodes; %.2f %% no view saw (filled from neighbours): %s' % (total, 100 * unseen / total, ', '.join('%s %.2f %% unseen, %.1f views per node' % (n['name'], 100 * n['unseen'], n['views']) for n in nodes)))
    log('%d nodes (%.2f %%) stood out from their neighbourhood and took its median' % (speck, 100 * speck / total))
    log('PCA-%d keeps %.1f %% of the variance; per component, %%: %s' % (DIM, 100 * var[:DIM].sum(), ' '.join('%.1f' % (100 * v) for v in var[:DIM])))
    log('wrote %s, %.0f KB' % (os.path.normpath(a.out), len(js) / 1e3))
    if a.cache: np.savez(os.path.join(a.cache, 'debug.npz'), nseen=nseen, wgt=wgt, shapes=np.array([n['shape'] for n in nodes]), X=X.astype(np.float32), co=np.concatenate(co).astype(np.float32))

    if a.check:                   # pictures of the first chart's stored grid, row 0 (the table) at the bottom
        os.makedirs(a.check, exist_ok=True); nv, nu = nodes[0]['shape']; F = (q[0] * scale).reshape(nv, nu, DIM); up = lambda im: np.kron(im[::-1], np.ones((3, 3, 1)))
        rgb = np.clip((F[..., :3] - np.array(data['lo'])) / (np.array(data['hi']) - np.array(data['lo'])), 0, 1)
        Image.fromarray((up(rgb) * 255).astype(np.uint8)).save(os.path.join(a.check, 'features%d-pca.png' % a.scan))
        c = d['charts'][0]; C = np.frombuffer(base64.b64decode(c['C']), np.uint8).reshape(c['nv'], c['nu'], 3)[::st, ::st] / 255
        Image.fromarray((up(C) * 255).astype(np.uint8)).save(os.path.join(a.check, 'features%d-colour.png' % a.scan))
        for sd in a.seed:         # cosine similarity in the stored 16-D space, as the page computes it
            name, ij = sd.split(':'); i, j = (int(v) // st for v in ij.split(',')); f0 = F[j, i]
            sim = np.clip((F @ f0) / (np.linalg.norm(F, axis=-1) * np.linalg.norm(f0) + 1e-12), 0, 1)
            im = up(np.stack([sim, sim * 0.85, 0.25 + 0.5 * C.mean(-1) * (1 - sim)], -1)); im[(nv - 1 - j) * 3 - 3:(nv - 1 - j) * 3 + 6, i * 3 - 3:i * 3 + 6] = [0, 1, 1]
            Image.fromarray((im * 255).astype(np.uint8)).save(os.path.join(a.check, 'features%d-sim-%s.png' % (a.scan, name)))
            others = [((x.reshape(-1, DIM) * scale) @ f0 / (np.linalg.norm(x.reshape(-1, DIM) * scale, axis=1) * np.linalg.norm(f0) + 1e-12)) for x in q[1:]]
            log('similarity to %s (%d, %d): %.1f %% of the %s nodes above 0.5, %.1f %% above 0.8%s' % (name, i * st, j * st, 100 * (sim > 0.5).mean(), nodes[0]['name'], 100 * (sim > 0.8).mean(),
                ''.join('; %s: mean %.2f, max %.2f' % (n['name'], o_.mean(), o_.max()) for n, o_ in zip(nodes[1:], others))))


HEAD = """/* %(model)s features of DTU scan %(scan)d on the control points of its charts. GENERATED by tools/dtu/features.py: do not
   edit, re-run the script (and re-run it whenever the scan's control nets change).
   The features come from a PRETRAINED model (DINOv3, Meta; DINOv3 License) run on the scan's own photographs; they were
   lifted onto the control points offline and reduced to %(dim)d numbers by PCA. Nothing here was trained by the thesis method.
   KnotTalk.dtuFeatures[%(scan)d] = the format of lab/features/<scene>.js: { model, dim, views, res, explained, scale, lo, hi,
     charts: { <chart name>: { nu, nv, step, net, data } } }. data = base64 of Int8Array(nu * nv * dim); a feature = byte * scale.
     A chart is stored at every step-th control point: nu, nv are the STORED grid's size, net = [nu, nv] of the control net,
     stored node (a, b) = control point (a * step, b * step); row b, column a -> (b * nu + a) * dim. lo, hi = the range of
     the first three components, for showing them as a colour. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {};
  (KT.dtuFeatures = KT.dtuFeatures || {})[%(scan)d] = """

if __name__ == '__main__':
    main()
