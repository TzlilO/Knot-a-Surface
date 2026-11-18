"""--self-test of label_points.py: scene B of selftest.py (a tall open sheet, a low mound and the ground plane, 20
cameras on a 150 degree arc, 0.5 px observation noise, 4 % outliers) with RENDERED per-view masks (a z-buffer of
the true surfaces at a quarter of the image size), ids shuffled in every view, then degraded.

Variants (each through associate(); the first degraded one also through the real file entry, .npy + .png + a
two-level view):
  clean        every view, exact masks
  degraded     20 % of the views missing, 30 % over-segmented (an object cut in two by a random image line), 10 %
               under-segmented (the two objects in one mask), boundary jitter of 1 mask pixel (4 image px), 5 %
               salt noise (a pixel takes a random id of its view, or 0)
  hierarchy    degraded, plus a second level in every view with every object cut in two
  half-split   the tall object cut at a FIXED height in 60 % of the views (masks undecided there): masks alone,
               then masks + synthetic features (one feature per object + noise), which should merge it back
  features     synthetic features alone (no masks)
  two-views    degraded masks, but every track cut to its two lowest-index views (as talk/dtu/pipeline122.js keeps)
  sweep        degraded with salt noise 0, 20, 40, 50, 60 %, 2 seeds: adjusted Rand index and failure rate
Metrics on the non-outlier points, leaving out those within 1 % H of two true surfaces (the mound's apron is tangent
to the plane: about a fifth of the mound is there, its label is not defined; the count is printed): adjusted Rand index (ARI, on the labelled points), purity, completeness, the share
left unlabelled, and "objects back" = every true object has a cluster holding >= 90 % of it at >= 95 % purity.
Pass: clean, degraded, hierarchy, half-split with features and features alone get the objects back with ARI >= 0.9 and
<= 5 % unlabelled; the file entry agrees with the in-memory run; the whole test under 60 s."""
import math, os, tempfile, time, json
import numpy as np

import colmap_io
import label_points as lp
import selftest as st

RES = 0.25


def ari(a, b):
    ua, ia = np.unique(a, return_inverse=True); ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub))); np.add.at(M, (ia, ib), 1); c2 = lambda x: x * (x - 1) / 2
    sij, sa, sb, n = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum(), c2(len(a))
    e = sa * sb / n; mx = 0.5 * (sa + sb); return float((sij - e) / (mx - e)) if mx != e else 1.0


def metrics(pred, truth):
    m = truth >= 0; p, t = pred[m], truth[m]; lab = p >= 0          # truth: -1 = outlier, -2 = ambiguous (not scored)
    out = dict(unlabelled=float((~lab).mean()), ari=ari(p[lab], t[lab]) if lab.sum() > 1 else 0.0, clusters=int(len(np.unique(p[lab]))))
    pur = sum(np.bincount(t[lab][p[lab] == c]).max() for c in np.unique(p[lab])) / max(lab.sum(), 1)
    comp, back = [], True
    for k in np.unique(t):
        tk = p[t == k]; tk = tk[tk >= 0]
        if not len(tk): comp.append(0.0); back = False; continue
        c = np.bincount(tk).argmax(); share = (tk == c).mean() * len(tk) / (t == k).sum(); comp.append(float(share))
        purity_c = (t[lab][p[lab] == c] == k).mean(); back &= share >= 0.9 and purity_c >= 0.95
    out.update(purity=float(pur), completeness=float(np.mean(comp)), objects_back=bool(back))
    return out


def ambiguous(sc, ids, tol=0.01):
    """Points whose true position lies within tol x H of TWO true surfaces (the mound's apron meets the plane at a
    tangent, so a band around it belongs to both): they are left out of the scores, and counted."""
    from scipy.spatial import cKDTree
    P = sc['true'][ids - 1]; d = [cKDTree(tr.dense(320)).query(P)[0] for _, tr, _ in sc['truths']] + [np.abs(P[:, 2])]
    d = np.sort(np.stack(d, 1), 1); return (d[:, 1] < tol * sc['H']) & (sc['lab'][ids - 1] >= 0)


def render(sc, rng_plane=400):
    """Per camera: an integer image of the true object index + 1 (0 = background), z-buffered, at RES of the image."""
    K = sc['K']; W, H = int(K['W'] * RES), int(K['H'] * RES); objs = []
    for name, tr, lab in sc['truths']: objs.append((tr.dense(320), lab))
    g = np.linspace(-1.3, 1.3, rng_plane); G = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2); objs.append((np.c_[G, np.zeros(len(G))], 2))
    P = np.concatenate([o[0] for o in objs]); L = np.concatenate([np.full(len(o[0]), o[1] + 1) for o in objs])
    out = []
    for C, Rc in zip(sc['cams'], sc['Rs']):
        q = (P - C) @ Rc.T; z = q[:, 2]; x = (K['fx'] * q[:, 0] / z + K['cx']) * RES; y = (K['fy'] * q[:, 1] / z + K['cy']) * RES
        ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H); pix = y[ok].astype(int) * W + x[ok].astype(int)
        o = np.lexsort((z[ok], pix)); pix, ll = pix[o], L[ok][o]; first = np.r_[True, pix[1:] != pix[:-1]]
        img = np.zeros(W * H, np.int64); img[pix[first]] = ll[first]; out.append(img.reshape(H, W))
    return out


def degrade(imgs, rng, missing=0.0, over=0.0, under=0.0, jitter=0, salt=0.0, fixed_split=None, levels2=False):
    """-> dict name -> mask (H x W x K) with ids shuffled per view, or None for a missing view."""
    out = {}; n = len(imgs)
    gone = set(rng.choice(n, int(round(missing * n)), replace=False).tolist()) if missing else set()
    for k, base in enumerate(imgs):
        name = '%04d.png' % k
        if k in gone: out[name] = None; continue
        H, W = base.shape; a = base.copy()
        if jitter:
            dy, dx = rng.integers(-jitter, jitter + 1, (2, H, W)); yy, xx = np.mgrid[:H, :W]
            a = a[np.clip(yy + dy, 0, H - 1), np.clip(xx + dx, 0, W - 1)]
        if rng.random() < under: a = np.where(a == 2, 1, a)                         # the two objects in one mask
        nxt = 10
        if fixed_split is not None and rng.random() < fixed_split[1]:
            a = np.where((a == 1) & (fixed_split[0][k] > 0), nxt, a); nxt += 1
        if rng.random() < over:
            for c in np.unique(a[a > 0]):
                ys, xs = np.nonzero(a == c)
                if len(ys) < 20: continue
                th = rng.uniform(0, np.pi); cy, cx = ys.mean(), xs.mean(); yy, xx = np.mgrid[:H, :W]
                side = ((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)) > 0
                a = np.where((a == c) & side, nxt, a); nxt += 1
        if salt:
            ids = np.unique(a); m = rng.random(a.shape) < salt; a = np.where(m, rng.choice(np.r_[0, ids], a.shape), a)
        u = np.unique(a); new = rng.choice(np.arange(1, 60000), len(u), replace=False); new[u == 0] = 0     # ids shuffled per view
        lv = new[np.searchsorted(u, a)]
        if levels2:
            ys, xs = np.mgrid[:H, :W]; fine = lv * 2 + ((xs + ys) > (W + H) / 2); fine[lv == 0] = 0
            lv = np.stack([lv, fine], -1)
        out[name] = lv if lv.ndim == 3 else lv[:, :, None]
    return out


def height_split(sc, imgs):
    """Per view: an image that is 1 where the tall object's surface lies above 55 % of its height (a fixed 3-D cut)."""
    K = sc['K']; tr = sc['truths'][0][1]; S = tr.dense(320); up = S[:, 2] > 0.55 * sc['H']; out = []
    for (C, Rc), base in zip(zip(sc['cams'], sc['Rs']), imgs):
        H, W = base.shape; q = (S - C) @ Rc.T; x = (K['fx'] * q[:, 0] / q[:, 2] + K['cx']) * RES; y = (K['fy'] * q[:, 1] / q[:, 2] + K['cy']) * RES
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H); m = np.zeros((H, W), int); m[y[ok].astype(int), x[ok].astype(int)] = up[ok].astype(int) + 1
        # fill the cut image by nearest painted pixel so every pixel of the object has a side
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(m == 0, return_distances=False, return_indices=True); out.append((m[tuple(idx)] == 2).astype(int))
    return out


def features(sc, imgs, rng, sigma=0.35, D=16, stride=8):
    base = rng.normal(size=(4, D)); base /= np.linalg.norm(base, axis=1, keepdims=True); out = {}
    for k, im in enumerate(imgs):
        H, W = im.shape; s = stride // 4 if stride >= 4 else 1; cls = im[s // 2::s, s // 2::s]
        f = base[cls] + rng.normal(scale=sigma / math.sqrt(D), size=cls.shape + (D,)); out['%04d.png' % k] = f.astype(np.float16)
    return out


def args(**kw):
    a = lp.parser().parse_args([]); a.max_seconds = 60
    for k, v in kw.items(): setattr(a, k, v)
    return a


def two_views(M):
    """Every track cut to its two lowest-index views (images.txt and points3D.txt both)."""
    import copy
    M = copy.deepcopy(M); keep = {}
    for k, (ti, tp) in enumerate(zip(M['points']['track_image'], M['points']['track_point2d'])):
        o = np.argsort(ti)[:2]; M['points']['track_image'][k] = ti[o]; M['points']['track_point2d'][k] = tp[o]; keep[int(M['points']['id'][k])] = set(ti[o].tolist())
    for iid, im in M['images'].items():
        ok = np.array([p >= 0 and iid in keep.get(int(p), ()) for p in im['p3d']], bool); im['p3d'] = np.where(ok, im['p3d'], -1)
    return M


def main(a0):
    t0 = time.time(); base = a0.out or tempfile.mkdtemp(prefix='label_points_selftest_'); rng = np.random.default_rng(11); lp.QUIET['on'] = True
    st.tool.log('label_points self-test in %s' % base)
    sc = st.scene('B', rng); Rw = st.rot(rng); tw = rng.normal(size=3) * 2; scale = 3.7; d = os.path.join(base, 'model'); st.write(sc, d, rng, Rw, tw, scale)
    M = colmap_io.read_model(d, lambda *x: None); ids = M['points']['id']; truth = sc['lab'][ids - 1]
    amb = ambiguous(sc, ids); truth = np.where(amb, -2, truth)
    imgs = render(sc); st.tool.log('scene B: %d points (%d outliers, %d not scored: within 1 %% H of two true surfaces), %d views; masks rendered at %d x %d' % (len(ids), sc['n_out'], amb.sum(), len(imgs), imgs[0].shape[1], imgs[0].shape[0]))
    res = {}; ok = True
    def go(name, masks=None, feats=None, M_=M, **kw):
        lab, conf, rec = lp.associate(M_, (lambda nm: masks.get(nm)) if masks is not None else None, (lambda nm: feats.get(nm)) if feats is not None else None, args(**kw))
        m = metrics(lab, truth); m['median_confidence'] = float(np.median(conf[(conf >= 0) & (truth >= 0)])) if (conf >= 0).any() else None
        m['signal'] = rec['signal']; m['feature_merges'] = len(rec['feature_merges']); res[name] = m
        st.tool.log('   %-24s ARI %.3f  purity %.3f  completeness %.3f  unlabelled %.1f %%  clusters %d  objects back: %s%s'
                    % (name, m['ari'], m['purity'], m['completeness'], 100 * m['unlabelled'], m['clusters'], m['objects_back'], '  (feature merges %d)' % m['feature_merges'] if m['feature_merges'] else ''))
        return lab, m
    DEG = dict(missing=0.2, over=0.3, under=0.1, jitter=1, salt=0.05)
    go('clean', degrade(imgs, np.random.default_rng(1)))
    deg = degrade(imgs, np.random.default_rng(2), **DEG); lab_mem, _ = go('degraded', deg)
    go('hierarchy (2 levels)', degrade(imgs, np.random.default_rng(3), levels2=True, **DEG))
    hs = height_split(sc, imgs); half = degrade(imgs, np.random.default_rng(4), fixed_split=(hs, 0.6))
    go('half-split, masks', half)
    fe = features(sc, imgs, np.random.default_rng(5)); go('half-split, masks+features', half, fe)
    go('features alone', None, fe)
    go('two views per point', deg, M_=two_views(M), min_covis=1)
    # the real entry: files on disk (.npy, .png, one view with two levels)
    md = os.path.join(base, 'masks'); os.makedirs(md, exist_ok=True); k2 = 0
    for k, (nm, v) in enumerate(sorted(deg.items())):
        if v is None: continue
        if k % 3 == 0: lp.write_png16(os.path.join(md, nm), v[:, :, 0])                 # <name> itself is .png
        elif k2 == 0: np.save(os.path.join(md, nm[:-4] + '.npy'), np.concatenate([v, v], 2)); k2 = 1   # two identical levels
        else: np.save(os.path.join(md, nm + '.npy'), v[:, :, 0])
    _, _, rec = lp.run(lp.parser().parse_args([d, '-o', os.path.join(base, 'out'), '--masks', md, '--max-seconds', '60']))
    L = np.load(os.path.join(base, 'out', 'labels.npy'))[ids]; mf = metrics(L, truth); same = ari(L[truth >= 0], lab_mem[truth >= 0])
    st.tool.log('   %-24s ARI %.3f, agreement with the in-memory run (ARI) %.3f (a view with two identical levels counts its votes twice)' % ('file entry (degraded)', mf['ari'], same))
    res['file_entry'] = dict(mf, ari_vs_memory=same)
    # sweep
    sweep = []
    for salt in (0.0, 0.2, 0.4, 0.5, 0.6):
        rows = []
        for seed in range(2):
            dd = degrade(imgs, np.random.default_rng(100 + seed), **dict(DEG, salt=salt))
            lab, conf, rec = lp.associate(M, lambda nm: dd.get(nm), None, args()); rows.append(metrics(lab, truth))
        fail = np.mean([not (r['objects_back'] and r['ari'] >= 0.9) for r in rows])
        sweep.append(dict(salt=salt, ari_mean=float(np.mean([r['ari'] for r in rows])), ari_min=float(np.min([r['ari'] for r in rows])), unlabelled=float(np.mean([r['unlabelled'] for r in rows])), failure_rate=float(fail)))
        st.tool.log('   sweep salt %3.0f %% (+ 20 %% missing, 30 %% over-, 10 %% under-segmented, jitter 1): ARI mean %.3f min %.3f, unlabelled %.1f %%, failures %d / 2'
                    % (100 * salt, sweep[-1]['ari_mean'], sweep[-1]['ari_min'], 100 * sweep[-1]['unlabelled'], int(round(fail * 2))))
    res['sweep'] = sweep
    need = ['clean', 'degraded', 'hierarchy (2 levels)', 'half-split, masks+features', 'features alone']
    for k in need:
        good = res[k]['objects_back'] and res[k]['ari'] >= 0.9 and res[k]['unlabelled'] <= 0.05; ok &= good
        if not good: st.tool.log('   FAIL: %s' % k)
    ok &= same >= 0.95; dt = time.time() - t0; ok &= dt < 60
    st.tool.log('label_points self-test %s in %.1f s' % ('PASSED' if ok else 'FAILED', dt))
    json.dump(dict(passed=bool(ok), seconds=dt, results=res), open(os.path.join(base, 'selftest_labels.json'), 'w'), indent=1)
    return 0 if ok else 1
