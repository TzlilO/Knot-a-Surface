#!/usr/bin/env python3
"""STAND-IN per-view masks for DTU scan 122, for testing label_points.py end to end where no photographs (and so no
Segment Anything) are available. NOT SAM masks, and not evidence of what SAM would produce.

    python3 -I tools/recon/standin_dtu.py OUT [--model MODEL_DIR --synthetic-visibility MODEL_OUT] [--seed 0]

The masks are RENDERED from the repo's depth-fitted charts (the yardstick): talk/dtu-data.js (the object chart and
the table plane) and talk/dtu/owlmask122.js (owl vs books per control point, itself segmented offline). A z-buffer of
the three parts (1 owl, 2 books, 3 table, 0 nothing) per camera of talk/dtu/pipeline122.js, then per-view shuffled ids:
  OUT/masks_clean/NNNN.npy     every view, exact (a quarter of the image size: 389 x 291)
  OUT/masks_degraded/NNNN.npy  20 % of the views missing, 50 % over-segmented (each part cut by a random image line),
                               10 % under-segmented (owl and books in one mask), boundary jitter of 1 mask pixel
                               (4 image pixels), 2 % salt noise
  OUT/truth_render/NNNN.npy    the un-shuffled class image (for pictures and checks only)
Because the masks come from the same surfaces the yardstick measures, a run on them tests the plumbing and what
per-object charts buy, nothing about segmentation.

--synthetic-visibility: talk/dtu/pipeline122.js keeps two observations per point. This writes a copy of MODEL_DIR
whose tracks are extended by SYNTHETIC visibility: a point is added to a view when it projects inside the image and
lies no more than --vis-tol behind the stand-in surfaces' z-buffer there (half image size). The two stored
observations are always kept; with --cap-to-track (default) a track is cut to the point's TRUE track length (from
MODEL_DIR/track_length.npy), the extra views drawn at random among the visible ones. The 2-D positions are exact
projections. Use the result for labelling only; it is not a COLMAP output."""
import argparse, base64, json, os, re, sys, time
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bspline as bs
import colmap_io

TALK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk')
NAMES = {1: 'owl', 2: 'books', 3: 'table'}


def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)
T0 = time.time()


def charts():
    d = json.loads(re.search(r'= (\{"scan".*\});\n', open(os.path.join(TALK, 'dtu-data.js')).read()).group(1)); out = {}
    for c in d['charts']:
        if 'plane' in c: out['table'] = c['plane']
        else:
            q = np.frombuffer(base64.b64decode(c['P']), '<u2').reshape(c['nv'], c['nu'], 3).astype(float)
            out['object'] = np.array(c['lo']) + (np.array(c['hi']) - np.array(c['lo'])) * q / 65535
    m = json.loads(re.search(r'\[122\] = (\{.*\});\n', open(os.path.join(TALK, 'dtu', 'owlmask122.js')).read()).group(1))
    L = np.zeros(m['nu'] * m['nv']); k, on = 0, 0
    for r in m['runs']: L[k:k + r] = on; k += r; on ^= 1                   # runs: books first, then owl, alternating
    out['owl_cp'] = L.reshape(m['nv'], m['nu'])
    p = json.loads(re.search(r'\]\s*=\s*(\{"scan".*\});\n', open(os.path.join(TALK, 'dtu', 'pipeline122.js')).read()).group(1))
    out['cams'] = p['cameras']; out['unit_mm'] = d['unit_mm']
    return out


def samples(D, dens=3, table_step_mm=0.4):
    P = D['object']; nv, nu = P.shape[:2]; au, av = bs.Open.uniform(nu), bs.Open.uniform(nv)
    S, _, _ = bs.grid(au, av, np.concatenate([P, D['owl_cp'][..., None]], 2), dens * nu, dens * nv)
    S = S.reshape(-1, 4); lab = np.where(S[:, 3] > 0.5, 1, 2)
    x0, x1, z0, z1 = D['table']; st = table_step_mm / D['unit_mm']
    X, Z = np.meshgrid(np.arange(x0, x1, st), np.arange(z0, z1, st)); T = np.c_[X.ravel(), np.zeros(X.size), Z.ravel()]
    return np.r_[S[:, :3], T], np.r_[lab, np.full(len(T), 3)]


def render(P, L, cam, K, scale):
    R = np.array(cam['R']).reshape(3, 3); C = np.array(cam['c']); W, H = int(round(K['width'] * scale)), int(round(K['height'] * scale))
    q = (P - C) @ R.T; z = q[:, 2]; x = (K['fx'] * q[:, 0] / z + K['cx']) * scale; y = (K['fy'] * q[:, 1] / z + K['cy']) * scale
    ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H); pix = y[ok].astype(int) * W + x[ok].astype(int)
    o = np.lexsort((z[ok], pix)); pix, zz, ll = pix[o], z[ok][o], L[ok][o]; f = np.r_[True, pix[1:] != pix[:-1]]
    zb = np.full(W * H, np.inf); lb = np.zeros(W * H, np.int64); zb[pix[f]] = zz[f]; lb[pix[f]] = ll[f]
    zb, lb = zb.reshape(H, W), lb.reshape(H, W); hole = np.isinf(zb)
    if hole.any():                                                          # pinholes between samples: nearest filled pixel
        zm = ndimage.minimum_filter(np.where(hole, 1e9, zb), 3); fill = hole & (zm < 1e9)
        idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
        zb[fill] = zm[fill]; lb[fill] = lb[tuple(idx)][fill]
    return zb, lb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('out')
    ap.add_argument('--model', help='COLMAP model made by from_pipeline_js.py (for --synthetic-visibility)')
    ap.add_argument('--synthetic-visibility', metavar='MODEL_OUT', help='write a copy of --model with tracks extended by synthetic visibility')
    ap.add_argument('--vis-tol', type=float, default=2.0, help='mm a point may lie behind the z-buffer and still count as seen')
    ap.add_argument('--cap-to-track', type=int, default=1, help='1: cut each extended track to the point\'s true track length')
    ap.add_argument('--mask-scale', type=float, default=0.25); ap.add_argument('--depth-scale', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); D = charts(); P, L = samples(D); K = D['cams']['intrinsics']; cams = D['cams']['list']; um = D['unit_mm']
    log('stand-in surfaces: %d samples (owl %d, books %d, table %d); %d cameras' % (len(P), (L == 1).sum(), (L == 2).sum(), (L == 3).sum(), len(cams)))
    import selftest_labels as sl
    rd = os.path.join(a.out, 'truth_render'); os.makedirs(rd, exist_ok=True); base, zbufs = [], {}
    for c in cams:
        zb, lb = render(P, L, c, K, a.depth_scale); zbufs[c['id']] = zb
        f = int(round(a.depth_scale / a.mask_scale)); m = lb[f // 2::f, f // 2::f]; base.append(m); np.save(os.path.join(rd, '%04d.npy' % c['id']), m.astype(np.uint8))
    log('rendered: masks %d x %d, depth %d x %d; share of mask pixels: owl %.1f %%, books %.1f %%, table %.1f %%'
        % (base[0].shape[1], base[0].shape[0], zbufs[cams[0]['id']].shape[1], zbufs[cams[0]['id']].shape[0], *[100 * np.mean([(b == k).mean() for b in base]) for k in (1, 2, 3)]))
    sl.RES = 1.0
    for name, kw in (('masks_clean', {}), ('masks_degraded', dict(missing=0.2, over=0.5, under=0.1, jitter=1, salt=0.02))):
        out = sl.degrade(base, np.random.default_rng(a.seed + (1 if kw else 0)), **kw); dd = os.path.join(a.out, name); os.makedirs(dd, exist_ok=True); n = 0
        for k, c in enumerate(cams):
            v = out['%04d.png' % k]
            if v is None: continue
            np.save(os.path.join(dd, '%04d.npy' % c['id']), v[:, :, 0].astype(np.uint16)); n += 1
        log('%s: %d views written (%s)' % (name, n, kw or 'exact'))
    meta = dict(what='STAND-IN masks rendered from the depth-fitted charts (talk/dtu-data.js, talk/dtu/owlmask122.js): NOT SAM masks', classes=NAMES,
                mask_scale=a.mask_scale, depth_scale=a.depth_scale, degraded=dict(missing=0.2, over=0.5, under=0.1, jitter_px_mask=1, salt=0.02), seed=a.seed)
    if a.synthetic_visibility:
        M = colmap_io.read_model(a.model, lambda *x: None); Pp = M['points']; tl = np.load(os.path.join(a.model, 'track_length.npy'))[Pp['id']] if a.cap_to_track else None
        rng = np.random.default_rng(a.seed + 7); X = Pp['X']; byid = {im['name']: iid for iid, im in M['images'].items()}; cam = M['cameras'][1]
        vis = {}; ds = a.depth_scale; tol = a.vis_tol / um
        for c in cams:
            iid = byid['%04d.png' % c['id']]; im = M['images'][iid]; q = X @ im['R'].T + im['t']; z = q[:, 2]
            x = cam['fx'] * q[:, 0] / z + cam['cx']; y = cam['fy'] * q[:, 1] / z + cam['cy']; zb = zbufs[c['id']]; Hd, Wd = zb.shape
            ok = (z > 0) & (x >= 0) & (x < cam['width']) & (y >= 0) & (y < cam['height'])
            xi = np.clip((x * ds).astype(int), 0, Wd - 1); yi = np.clip((y * ds).astype(int), 0, Hd - 1); ok &= z <= zb[yi, xi] + tol
            vis[iid] = (ok, np.c_[x, y])
        ims = sorted(M['images']); V = np.stack([vis[i][0] for i in ims], 1)                           # points x views
        own = np.zeros_like(V)
        for k, ti in enumerate(Pp['track_image']): own[k, [ims.index(t) for t in np.unique(ti)]] = True
        keep = own.copy(); add = V & ~own
        for k in range(len(X)):
            cand = np.nonzero(add[k])[0]
            if not len(cand): continue
            room = len(cand) if tl is None else max(0, int(tl[k]) - int(own[k].sum()))
            if room <= 0: continue
            keep[k, rng.choice(cand, min(room, len(cand)), replace=False)] = True
        images = {}; ti = [[] for _ in range(len(X))]; tp = [[] for _ in range(len(X))]
        for v, iid in enumerate(ims):
            idx = np.nonzero(keep[:, v])[0]; xy = vis[iid][1][idx]; im = M['images'][iid]
            for m, j in enumerate(idx): ti[j].append(iid); tp[j].append(m)
            images[iid] = dict(name=im['name'], camera_id=im['camera_id'], R=im['R'], t=im['t'], xy=xy, p3d=Pp['id'][idx])
        pts = dict(id=Pp['id'], X=X, rgb=Pp['rgb'], error=Pp['error'], track_image=[np.array(t) for t in ti], track_point2d=[np.array(t) for t in tp])
        colmap_io.write_model(a.synthetic_visibility, {1: dict(model='PINHOLE', width=cam['width'], height=cam['height'], params=[cam['fx'], cam['fy'], cam['cx'], cam['cy']])}, images, pts)
        tlv = np.load(os.path.join(a.model, 'track_length.npy')); np.save(os.path.join(a.synthetic_visibility, 'track_length.npy'), tlv)
        n_ = keep.sum(1); vis_n = (V | own).sum(1)
        rec = dict(note='SYNTHETIC visibility from the stand-in surfaces (z-buffer, tol %.1f mm); 2-D positions are exact projections; not a COLMAP output' % a.vis_tol,
                   observations_before=int(own.sum()), observations_after=int(keep.sum()), per_point_after=dict(median=float(np.median(n_)), mean=float(n_.mean())),
                   visible_per_point=dict(median=float(np.median(vis_n)), mean=float(vis_n.mean())), true_track_length=dict(median=float(np.median(tl)) if tl is not None else None),
                   stored_observation_not_visible_by_the_z_buffer=float((own & ~V).sum() / own.sum()), cap_to_track=bool(a.cap_to_track))
        meta['synthetic_visibility'] = rec; json.dump(rec, open(os.path.join(a.synthetic_visibility, 'SYNTHETIC_VISIBILITY.json'), 'w'), indent=1)
        log('synthetic visibility -> %s: observations %d -> %d (median per point %.0f; visible by the z-buffer %.0f; true track length %s); %.1f %% of the stored observations lie behind the stand-in surface by more than %.1f mm'
            % (a.synthetic_visibility, own.sum(), keep.sum(), np.median(n_), np.median(vis_n), rec['true_track_length']['median'], 100 * rec['stored_observation_not_visible_by_the_z_buffer'], a.vis_tol))
    json.dump(meta, open(os.path.join(a.out, 'STANDIN.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
