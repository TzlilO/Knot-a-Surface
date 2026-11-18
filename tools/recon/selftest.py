"""--self-test of sfm_spline.py: synthetic scenes with known spline surfaces, written as COLMAP text models and fed
through the tool's real entry (the reader, every step, the writer), then measured against the truth.

  A  one closed body (a periodic cubic sheet, bumpy) on a tilted plane, 24 cameras all round, noise and 4 % outliers:
     expects a plane, a CLOSED sheet, and the surface back within the tolerances below.
  B  two objects with labels on the plane, cameras on a 150 degree arc in front: a tall open sheet (label 0) and a low
     wide mound that is a known height field (label 1); the plane points carry label 2, outliers -1. Expects one chart
     per label, a height field for the mound, and each surface back within the tolerances.

Tolerances, as shares of the object's height H: on the part of the fitted chart that stands on points, distance to
the true surface median <= 0.5 % H and 95th percentile <= 2 % H; from the true surface (where points were sampled
and seen) to the fitted chart, 95th percentile <= 2 % H. Point noise is 0.3 % H (sigma, per axis)."""
import math, os, sys, tempfile, time
import numpy as np
from scipy.spatial import cKDTree

import bspline as bs
import colmap_io
import sfm_spline as tool

TOL_MED, TOL_P95, TOL_BACK = 0.005, 0.02, 0.02


def rot(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); return colmap_io.qvec_to_R(q)


def look_at(C, target, up):
    z = target - C; z /= np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); y = np.cross(z, x); return np.stack([x, y, z])


class Truth:
    """A known chart in the scene's own frame (z up before the scene's random rotation)."""
    def __init__(self, au, av, net, sign=1.0): self.au, self.av, self.net, self.sign = au, av, net, sign
    def sample(self, n, rng):
        u, v = rng.random(n), rng.random(n); S, Su, Sv = bs.evaluate(self.au, self.av, self.net, u, v, True)
        a = np.linalg.norm(np.cross(Su, Sv), axis=1); keep = rng.random(n) < a / a.max()     # uniform by area
        return S[keep], self.sign * np.cross(Su, Sv)[keep]          # outward normals
    def dense(self, k=300): return bs.grid(self.au, self.av, self.net, k, k)[0].reshape(-1, 3)


def body(rng, H=1.0, R0=0.35, closed=True, span=2 * np.pi, centre=(0, 0)):
    nu, nv = (10 if closed else 9), 7
    au = bs.Periodic(nu) if closed else bs.Open.uniform(nu); av = bs.Open.uniform(nv)
    th = au.sites() * 2 * np.pi if closed else -span / 2 + au.sites() * span
    hv = av.sites() * H; net = np.zeros((nv, nu, 3))
    for j in range(nv):
        f = 1.0 - 0.75 * max(0, (hv[j] / H - 0.6) / 0.4) ** 1.5                  # narrowing to the top
        r = R0 * f * (1 + 0.15 * np.sin(3 * th + 1.3 * j) * (0.3 + hv[j] / H))
        net[j, :, 0] = centre[0] + r * np.sin(th); net[j, :, 1] = centre[1] + r * np.cos(th); net[j, :, 2] = hv[j]
    net[0, :, 2] = 0.0
    return Truth(au, av, net, -1.0)                                          # S_u x S_v points inwards here


def mound(centre, half, A):
    nu, nv = 8, 7; au, av = bs.Open.uniform(nu), bs.Open.uniform(nv); su, sv = au.sites(), av.sites(); net = np.zeros((nv, nu, 3))
    U, V = np.meshgrid(su, sv)
    net[..., 0] = centre[0] + (U - 0.5) * 2 * half[0]; net[..., 1] = centre[1] + (V - 0.5) * 2 * half[1]
    net[..., 2] = A * np.exp(-((U - 0.5) ** 2 + (V - 0.45) ** 2) / 0.05) + 0.3 * A * np.exp(-((U - 0.75) ** 2 + (V - 0.7) ** 2) / 0.01)
    net[0, :, 2] = net[-1, :, 2] = net[:, 0, 2] = net[:, -1, 2] = 0
    return Truth(au, av, net)


def occluded(X, C, solids, steps=24):
    """Is the segment C -> X blocked (before the last 2 %) by any solid? solids = callables p -> inside mask."""
    t = np.linspace(0.02, 0.97, steps); bad = np.zeros(len(X), bool)
    for tt in t:
        p = C + (X - C) * tt
        for f in solids: bad |= f(p)
    return bad


def inside_body(tr):
    """Inside test of a body chart: below its top and closer to its axis than the sheet at that azimuth and height."""
    S = tr.dense(180).reshape(180, 180, 3); c = S[..., :2].reshape(-1, 2).mean(0)
    th = np.arctan2(S[..., 0] - c[0], S[..., 1] - c[1]); r = np.hypot(S[..., 0] - c[0], S[..., 1] - c[1]); z = S[..., 2]
    tree = cKDTree(np.c_[np.cos(th).ravel(), np.sin(th).ravel(), z.ravel()]); zmax = z.max()
    def f(p):
        tp = np.arctan2(p[:, 0] - c[0], p[:, 1] - c[1]); rp = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
        _, j = tree.query(np.c_[np.cos(tp), np.sin(tp), np.clip(p[:, 2], 0, zmax)]); return (p[:, 2] > 0) & (p[:, 2] < zmax * 0.98) & (rp < r.ravel()[j] * 0.97)
    return f


def inside_mound(tr):
    S = tr.dense(120); tree = cKDTree(S[:, :2])
    def f(p): _, j = tree.query(p[:, :2]); return (p[:, 2] > 0) & (p[:, 2] < S[j, 2] * 0.97)
    return f


def scene(kind, rng, H=1.0):
    """Returns the model's pieces in the scene frame, the truths and their labels; the caller rotates everything."""
    noise = 0.003 * H; truths, solids = [], []
    if kind == 'A':
        b = body(rng, H); truths.append(('object', b, 0)); solids.append(inside_body(b))
        az = np.linspace(0, 2 * np.pi, 24, endpoint=False); cams = [np.array([3.2 * np.sin(t), 3.2 * np.cos(t), 1.2 + 0.6 * (k % 2)]) for k, t in enumerate(az)]
        npts = [7000]; plane_half = 1.2
    else:
        b = body(rng, H, R0=0.3, closed=False, span=np.radians(230), centre=(-0.45, 0.0)); m = mound((0.55, 0.1), (0.45, 0.4), 0.22 * H)
        truths += [('tall', b, 0), ('mound', m, 1)]; solids += [inside_body(b), inside_mound(m)]
        az = np.radians(np.linspace(-75, 75, 20)); cams = [np.array([3.4 * np.sin(t), 3.4 * np.cos(t), 1.0 + 0.8 * (k % 2)]) for k, t in enumerate(az)]
        npts = [5000, 4000]; plane_half = 1.3
    P, L, NRM = [], [], []
    for (name, tr, lab), n in zip(truths, npts):
        X, N = tr.sample(n, rng); P.append(X); NRM.append(N); L.append(np.full(len(X), lab))
    g = rng.uniform(-plane_half, plane_half, (2500, 2)); G = np.c_[g, np.zeros(len(g))]
    inside = np.zeros(len(G), bool)
    for f in solids: inside |= f(G + [0, 0, 0.01])
    G = G[~inside]; P.append(G); NRM.append(np.tile([0, 0, 1.0], (len(G), 1))); L.append(np.full(len(G), 2 if kind == 'B' else 1))
    X = np.concatenate(P); N = np.concatenate(NRM); lab = np.concatenate(L)
    K = dict(fx=900.0, fy=900.0, cx=480.0, cy=360.0, W=960, H=720); target = np.array([0, 0, 0.45 * H])
    Rs = [look_at(C, target, np.array([0, 0, 1.0])) for C in cams]
    # visibility: in the image, facing the camera, not blocked by a solid
    vis = np.zeros((len(cams), len(X)), bool)
    for k, (C, Rc) in enumerate(zip(cams, Rs)):
        q = (X - C) @ Rc.T; x = K['fx'] * q[:, 0] / q[:, 2] + K['cx']; y = K['fy'] * q[:, 1] / q[:, 2] + K['cy']
        ok = (q[:, 2] > 0) & (x >= 0) & (x < K['W']) & (y >= 0) & (y < K['H']) & (np.einsum('pc,pc->p', N, C - X) > 0)
        ok[ok] &= ~occluded(X[ok], C, solids); vis[k] = ok
    seen = vis.sum(0) >= 2; X, lab, vis = X[seen], lab[seen], vis[:, seen]
    Xn = X + rng.normal(scale=noise, size=X.shape)
    no = int(0.04 * len(X)); lo, hi = X.min(0) - 0.2, X.max(0) + 0.2; Xo = rng.uniform(lo, hi, (no, 3))
    vo = np.zeros((len(cams), no), bool)
    for j in range(no): vo[rng.choice(len(cams), 2, replace=False), j] = True
    Xall = np.r_[Xn, Xo]; labs = np.r_[lab, -np.ones(no, int)]; V = np.c_[vis, vo]
    return dict(X=Xall, true=np.r_[X, Xo], lab=labs, vis=V, cams=cams, Rs=Rs, K=K, truths=truths, noise=noise, H=H, n_out=no)


def write(sc, d, rng, Rw, tw, scale):
    """The scene through a random similarity (rotation Rw, translation tw, scale) into a COLMAP text model."""
    K = sc['K']; X = (sc['X'] @ Rw.T + tw) * scale; npnt = len(X)
    images, tracks_i, tracks_p = {}, [[] for _ in range(npnt)], [[] for _ in range(npnt)]; err = np.zeros(npnt); cnt = np.zeros(npnt)
    for k, (C, Rc) in enumerate(zip(sc['cams'], sc['Rs'])):
        Cw = (Rw @ C + tw) * scale; Rcw = Rc @ Rw.T; t = -Rcw @ Cw
        idx = np.nonzero(sc['vis'][k])[0]; q = (sc['true'][idx] - C) @ Rc.T
        xy = np.c_[K['fx'] * q[:, 0] / q[:, 2] + K['cx'], K['fy'] * q[:, 1] / q[:, 2] + K['cy']]
        nz = rng.normal(scale=0.5, size=xy.shape); xy = xy + nz
        for m, j in enumerate(idx): tracks_i[j].append(k + 1); tracks_p[j].append(m)
        err[idx] += np.linalg.norm(nz, axis=1); cnt[idx] += 1
        images[k + 1] = dict(name='%04d.png' % k, camera_id=1, R=Rcw, t=t, xy=xy, p3d=idx + 1)
    err /= np.maximum(cnt, 1); err[-sc['n_out']:] += rng.uniform(0.2, 3.0, sc['n_out'])
    pts = dict(id=np.arange(1, npnt + 1), X=X, rgb=np.full((npnt, 3), 128, np.uint8), error=err, track_image=[np.array(t) for t in tracks_i], track_point2d=[np.array(t) for t in tracks_p])
    colmap_io.write_model(d, {1: dict(model='PINHOLE', width=K['W'], height=K['H'], params=[K['fx'], K['fy'], K['cx'], K['cy']])}, images, pts)
    L = np.full(npnt + 1, -1, int); L[1:] = sc['lab']; np.save(os.path.join(d, 'labels.npy'), L)


def check(name, ch, tr, Rw, tw, scale, H, rep, s):
    """Fitted chart (back in the scene frame) vs the truth."""
    (S, _, _), U, V = bs.grid(ch.au, ch.av, ch.net, 160, 160, True)
    Ss = (S.reshape(-1, 3) / scale - tw) @ Rw                                # model -> scene frame
    T = tr.dense(400); dt = cKDTree(T)
    d_fit, _ = dt.query(Ss)
    # only where the fitted chart stands on points: lattice cells near an input point
    sup = cKDTree(rep['_pts']).query(S.reshape(-1, 3))[0] < 3 * s
    dF = d_fit[sup] / H
    seen_true = rep['_true_seen']; d_back, _ = cKDTree(Ss).query(seen_true); dB = d_back / H
    ok = np.median(dF) <= TOL_MED and np.percentile(dF, 95) <= TOL_P95 and np.percentile(dB, 95) <= TOL_BACK
    tool.log('   check %-7s chart->truth (supported %.0f %% of it): median %.3f %% H, p95 %.3f %% H | truth->chart p95 %.3f %% H  -> %s'
             % (name, 100 * sup.mean(), 100 * np.median(dF), 100 * np.percentile(dF, 95), 100 * np.percentile(dB, 95), 'ok' if ok else 'FAIL'))
    return ok, dict(chart=ch.name, supported_share=float(sup.mean()), chart_to_truth_median_H=float(np.median(dF)), chart_to_truth_p95_H=float(np.percentile(dF, 95)),
                    truth_to_chart_p95_H=float(np.percentile(dB, 95)), pass_=bool(ok))


def main(a):
    t0 = time.time(); base = a.out or tempfile.mkdtemp(prefix='sfm_spline_selftest_'); rng = np.random.default_rng(7); results = []; allok = True
    tool.log('self-test: tolerances median <= %.1f %% H, p95 <= %.1f %% H (chart on points -> truth), truth -> chart p95 <= %.1f %% H; noise 0.3 %% H; in %s' % (100 * TOL_MED, 100 * TOL_P95, 100 * TOL_BACK, base))
    for kind in ('A', 'B'):
        sc = scene(kind, rng); Rw = rot(rng); tw = rng.normal(size=3) * 2; scale = 3.7
        d = os.path.join(base, 'scene' + kind); write(sc, d, rng, Rw, tw, scale)
        argv = [d, '-o', os.path.join(d, 'out'), '--max-seconds', '60', '--unit-mm', str(100 / scale)] + (['--labels', os.path.join(d, 'labels.npy')] if kind == 'B' else [])
        tool.log('scene %s: %d points (%d outliers), %d cameras -> %s' % (kind, len(sc['X']), sc['n_out'], len(sc['cams']), ' '.join(argv)))
        R, charts = tool.run(tool.parser().parse_args(argv))
        s = R['clean']['spacing'] / scale
        exp = {'A': {'object': (0, 'closed')}, 'B': {'label0': (0, 'open'), 'label1': (1, 'height')}}[kind]
        ok_plane = R['plane']['found'] and math.degrees(math.acos(abs(np.array(R['plane']['normal']) @ (Rw @ [0, 0, 1.0])))) < 1.0
        tool.log('   check plane found, normal within 1 deg of the truth: %s' % ('ok' if ok_plane else 'FAIL')); allok &= ok_plane
        for ch in charts:
            if ch.name not in exp: continue
            want_lab, want_kind = exp[ch.name]; tr = [t for t in sc['truths'] if t[2] == want_lab][0][1]
            got = 'height' if ch.lay.kind == 'height' else ('closed' if ch.lay.closed else 'open')
            okk = got == want_kind or (want_kind == 'open' and got == 'open'); tool.log('   check %s layout: %s (expected %s) -> %s' % (ch.name, got, want_kind, 'ok' if okk else 'FAIL'))
            sel = (sc['lab'] == want_lab)[:len(sc['true']) - sc['n_out']]
            rep = dict(R, _pts=(sc['X'][:len(sel)][sel] @ Rw.T + tw) * scale, _true_seen=sc['true'][:len(sel)][sel])
            ok, res = check(ch.name, ch, tr, Rw, tw, scale, sc['H'], rep, R['clean']['spacing'])
            res['layout'] = got; res['layout_ok'] = okk; results.append(res); allok &= ok and okk
        allok &= all(k in [c.name for c in charts] for k in exp)
        # the free-space diagnostic itself: its floor on the TRUE surfaces, and what it says of a wrong one
        class C_: pass
        def as_chart(name, au, av, net):
            c = C_(); c.name, c.au, c.av, c.net = name, au, av, (net @ Rw.T + tw) * scale; return c
        D = tool.load(d); keep, sp, _ = tool.clean(D, tool.parser().parse_args(argv), {}); okp = keep[D['obs_pt']]
        a2 = tool.parser().parse_args(argv)
        truth = [as_chart(n, tr.au, tr.av, tr.net) for n, tr, _ in sc['truths']]
        pl = sc['X'][sc['lab'] == (1 if kind == 'A' else 2)]; ext = np.abs(pl[:, :2]).max() + 0.05
        ga, gb = bs.Open.uniform(4), bs.Open.uniform(4); g = np.zeros((4, 4, 3)); U, V = np.meshgrid(ga.sites(), gb.sites()); g[..., 0] = (U - 0.5) * 2 * ext; g[..., 1] = (V - 0.5) * 2 * ext
        truth.append(as_chart('plane', ga, gb, g))
        f_true = tool.freespace(truth, D['cams'], D['obs_cam'][okp], D['obs_pt'][okp], D['X'], sp, a2)
        b0 = sc['truths'][0][1]; cx = b0.net[..., :2].reshape(-1, 2).mean(0); big = b0.net.copy(); big[..., :2] = cx + (big[..., :2] - cx) * 1.15
        wrong = [as_chart('inflated', b0.au, b0.av, big)] + truth[1:]
        f_wrong = tool.freespace(wrong, D['cams'], D['obs_cam'][okp], D['obs_pt'][okp], D['X'], sp, a2)
        okf = f_true['share'] < 0.02 and f_wrong['share'] > 5 * max(f_true['share'], 0.002)
        tool.log('   check free space: on the TRUE surfaces %.2f %% of the pairs violate (the diagnostic\'s floor with this scene\'s noise and visibility), '
                 'on the fit %.2f %%, with the first object inflated by 15 %% sideways %.2f %% -> %s' % (100 * f_true['share'], 100 * R['freespace']['share'], 100 * f_wrong['share'], 'ok' if okf else 'FAIL'))
        results.append(dict(scene=kind, freespace_true=f_true['share'], freespace_fit=R['freespace']['share'], freespace_inflated=f_wrong['share'], pass_=bool(okf))); allok &= okf
    dt = time.time() - t0; tool.log('self-test %s in %.1f s' % ('PASSED' if allok and dt < 60 else 'FAILED', dt))
    with open(os.path.join(base, 'selftest.json'), 'w') as fh:
        import json; json.dump(dict(passed=bool(allok), seconds=dt, tolerances=dict(median_H=TOL_MED, p95_H=TOL_P95, back_p95_H=TOL_BACK), results=results), fh, indent=1)
    return 0 if allok and dt < 60 else 1
