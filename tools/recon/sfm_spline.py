#!/usr/bin/env python3
"""Cubic B-spline charts fitted to a structure-from-motion point cloud ALONE (no depth maps, no silhouettes).

    python3 -I tools/recon/sfm_spline.py MODEL_DIR -o OUT [--labels labels.npy] [--unit-mm U] [options]
    python3 -I tools/recon/sfm_spline.py --self-test [-o OUT]

MODEL_DIR = a COLMAP text model (cameras.txt, images.txt, points3D.txt). --labels = one integer per point3D id
(index = POINT3D_ID; -1 = ignore, every label >= 0 gets its own chart). OUT gets charts.json and report.json.

  1. clean     track length, mean reprojection error, statistical k-nearest-neighbour outliers
  2. plane     RANSAC on the points for a support plane (all cameras and nearly all other points on one side of it);
               it defines "up" and becomes a planar chart (a height field over the plane where points exist).
               No plane -> up = the cameras' mean up vector, said so.
  3. layout    per object (the largest connected component of the kNN graph above the plane, or one per label):
               candidates = a sheet around an axis (cylindrical or spherical rows) or a height field over the object's
               best plane, each measured on the points (share of points in cells with two layers + share of empty
               cells); the lowest score wins and every candidate's numbers are written down.
  4. fit       envelope / plane start, then robust least squares on the control net (Tukey weights, second
               differences as smoothness, parameter correction), coarse to fine by knot insertion where at least
               --min-support points per knot cell carry it and its residual is above the chart's median (--refine-ratio).
  5. free space a camera that observed a point saw it unoccluded: no fitted surface may lie in front of the point on
               that ray (beyond --fs-tol). Measured as a diagnostic; --freespace W adds it as a penalty in the fit.
  6. write     charts.json, report.json (thresholds, counts, residuals, unsupported share, violations, stop)

Long-job rules (as tools/train): --max-seconds and --max-iters; NaN / Inf stops at once and the bad state is not
kept; convergence on a held-out share of the points (--holdout); SIGINT / SIGTERM stop cleanly; the best state (by
the held-out median distance) is what is written; the reason it stopped is recorded.

This is a GEOMETRIC FIT to SfM points: nothing here is trained. numpy + scipy only."""
import argparse, json, math, os, signal, sys, time
import numpy as np
from scipy import sparse, ndimage
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bspline as bs
import colmap_io

T0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

STOP = {'signal': None}
def _on_signal(sig, frame): STOP['signal'] = signal.Signals(sig).name; log('received %s: stopping after the current step, the best state is kept' % STOP['signal'])


class Halt(Exception):
    def __init__(self, reason, detail=''): super().__init__(reason); self.reason = reason; self.detail = detail


class Budget:
    """Wall clock and iteration cap shared by every chart's fit; checked at every solve."""
    def __init__(self, seconds, iters): self.seconds, self.iters, self.it, self.t0 = seconds, iters, 0, time.time()
    def tick(self):
        self.it += 1
        if STOP['signal']: raise Halt('signal', STOP['signal'])
        if time.time() - self.t0 > self.seconds: raise Halt('wall_clock', '%.0f s budget' % self.seconds)
        if self.it > self.iters: raise Halt('iteration_cap', '%d solves' % self.iters)


def summary(d, tol, unit_mm=None):
    d = np.asarray(d, float)
    if not len(d): return dict(points=0)
    o = dict(points=int(len(d)), median=float(np.median(d)), p95=float(np.percentile(d, 95)), rms=float(np.sqrt((d ** 2).mean())),
             within_tol=float((d < tol).mean()), tol=float(tol))
    if unit_mm: o.update(median_mm=o['median'] * unit_mm, p95_mm=o['p95'] * unit_mm, rms_mm=o['rms'] * unit_mm, tol_mm=tol * unit_mm)
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in o.items()}


def fmt(s, unit_mm=None):
    if not s.get('points'): return 'no points'
    if unit_mm: return 'median %.3f  p95 %.3f  rms %.3f mm, within %.2f mm %.1f %% (%d points)' % (s['median_mm'], s['p95_mm'], s['rms_mm'], s['tol_mm'], 100 * s['within_tol'], s['points'])
    return 'median %.4g  p95 %.4g  rms %.4g, within %.3g %.1f %% (%d points)' % (s['median'], s['p95'], s['rms'], s['tol'], 100 * s['within_tol'], s['points'])


# ====================================================================== 0. model
def load(model_dir, track_len_file=None, labels_file=None):
    M = colmap_io.read_model(model_dir, log); P = M['points']
    ims = sorted(M['images']); iidx = {iid: k for k, iid in enumerate(ims)}
    cams = []
    for iid in ims:
        im = M['images'][iid]; c = M['cameras'][im['camera_id']]
        cams.append(dict(id=iid, name=im['name'], R=im['R'], t=im['t'], C=im['centre'], fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy'], W=c['width'], H=c['height']))
    tl = np.array([len(np.unique(t)) for t in P['track_image']]); obs_n = np.array([len(t) for t in P['track_image']])
    if track_len_file:
        T = np.load(track_len_file); tl = T[P['id']].astype(int); log('track lengths read from %s (the model\'s own tracks are incomplete)' % track_len_file)
    oc = np.concatenate([[iidx[i] for i in np.unique(t)] for t in P['track_image']]).astype(np.int64) if len(P['id']) else np.zeros(0, int)
    op = np.repeat(np.arange(len(P['id'])), [len(np.unique(t)) for t in P['track_image']])
    labels = None
    if labels_file:
        L = np.load(labels_file)
        if L.ndim != 1 or len(L) <= P['id'].max(): raise SystemExit('%s: expected one integer per point3D id (length > %d), got shape %s' % (labels_file, P['id'].max(), L.shape))
        labels = L[P['id']].astype(int)
    return dict(X=P['X'], rgb=P['rgb'], err=P['error'], tl=tl, obs_n=obs_n, ids=P['id'], cams=cams, obs_cam=oc, obs_pt=op, labels=labels)


# ====================================================================== 1. clean
def clean(D, a, R):
    X = D['X']; n0 = len(X); keep = np.ones(n0, bool)
    k1 = D['tl'] >= a.min_track; k2 = D['err'] <= a.max_reproj
    keep &= k1 & k2; pre = keep.copy()
    log('1. clean: %d points; track length >= %d drops %d, reprojection error <= %.2f px drops %d' % (n0, a.min_track, (~k1).sum(), a.max_reproj, (~k2 & k1).sum()))
    idx = np.nonzero(keep)[0]; tree = cKDTree(X[idx]); dk, _ = tree.query(X[idx], a.knn + 1, workers=-1); md = dk[:, 1:].mean(1)
    lm = np.log(md); med = np.median(lm); mad = 1.4826 * np.median(np.abs(lm - med)); lim = float(np.exp(med + a.knn_sigma * mad))
    out = md > lim; keep[idx[out]] = False          # robust z-score of log(mean neighbour distance): density varies by factors, not offsets
    idx = np.nonzero(keep)[0]; d1 = cKDTree(X[idx]).query(X[idx], 2, workers=-1)[0][:, 1]; s = float(np.median(d1))
    log('   k-NN outliers (mean distance to %d neighbours: log above median + %.1f MAD, i.e. > %.4g): %d dropped; %d kept; spacing (median nearest-neighbour distance) s = %.4g%s'
        % (a.knn, a.knn_sigma, lim, out.sum(), keep.sum(), s, ' = %.3f mm' % (s * a.unit_mm) if a.unit_mm else ''))
    R['clean'] = dict(points_in=int(n0), min_track=a.min_track, dropped_track=int((~k1).sum()), max_reproj_px=a.max_reproj, dropped_reproj=int((~k2 & k1).sum()),
                      knn=a.knn, knn_sigma=a.knn_sigma, knn_limit=float(lim), dropped_knn=int(out.sum()), kept=int(keep.sum()), spacing=s,
                      spacing_mm=s * a.unit_mm if a.unit_mm else None,
                      track_length=dict(median=float(np.median(D['tl'])), max=int(D['tl'].max())), reproj_px=dict(median=float(np.median(D['err'])), p95=float(np.percentile(D['err'], 95))))
    return keep, s, pre


# ====================================================================== 2. support plane
def ortho(n):
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0]); e1 = np.cross(n, t); e1 /= np.linalg.norm(e1); return e1, np.cross(n, e1)


def support_plane(X, cams, s, a, R, rng):
    C = np.array([c['C'] for c in cams]); thr = a.plane_tol * s; n = len(X); best = None
    nb = cKDTree(X).query(X, min(n, a.plane_local), workers=-1)[1]       # local sampling: three points from one neighbourhood
    for it in range(a.plane_iters):
        i0 = rng.integers(n); j = rng.choice(nb.shape[1] - 1, 2, replace=False) + 1
        p = X[[i0, nb[i0, j[0]], nb[i0, j[1]]]]; nv = np.cross(p[1] - p[0], p[2] - p[0]); L = np.linalg.norm(nv)
        if L < 1e-12: continue
        nv /= L; d = (X - p[0]) @ nv; inl = np.abs(d) < thr; ni = inl.sum()
        if best is not None and ni <= best[0]: continue
        dc = (C - p[0]) @ nv
        if not (np.all(dc > 0) or np.all(dc < 0)): continue                 # every camera on one side
        sg = np.sign(dc[0]); far = d[~inl]
        if not len(far) or (np.sign(far) == sg).mean() < a.plane_side: continue  # nearly every other point on the cameras' side
        best = (ni, nv * sg, p[0])
    up_cams = -np.mean([c['R'][1] for c in cams], 0); up_cams /= np.linalg.norm(up_cams)
    if best is None or best[0] < max(a.plane_min_points, a.plane_min_share * n):
        log('2. plane: none found (no plane with %d+ points that has every camera and %.0f %% of the other points on one side); up = the cameras\' mean up vector'
            % (max(a.plane_min_points, a.plane_min_share * n), 100 * a.plane_side))
        R['plane'] = dict(found=False, up_from='cameras', up=up_cams.tolist(), tol=thr, iters=a.plane_iters)
        return None, up_cams
    nv, p0 = best[1], best[2]; inl = np.abs((X - p0) @ nv) < thr
    for _ in range(3):                                                      # least squares refinement on the inliers
        c = X[inl].mean(0); w, V = np.linalg.eigh(np.cov((X[inl] - c).T)); nn = V[:, 0] * np.sign(V[:, 0] @ nv); nv, p0 = nn, c; inl = np.abs((X - p0) @ nv) < thr
    rms = float(np.sqrt((((X[inl] - p0) @ nv) ** 2).mean())); side = float((np.sign((X[~inl] - p0) @ nv) > 0).mean())
    ang = math.degrees(math.acos(np.clip(nv @ up_cams, -1, 1)))
    log('2. plane: %d points within %.3g of it (rms %.3g%s), %.1f %% of the others above it, all %d cameras above; its normal is %.1f deg from the cameras\' mean up vector'
        % (inl.sum(), thr, rms, ' = %.3f mm' % (rms * a.unit_mm) if a.unit_mm else '', 100 * side, len(cams), ang))
    R['plane'] = dict(found=True, up_from='plane', normal=nv.tolist(), point=p0.tolist(), inliers=int(inl.sum()), tol=thr, rms=rms, above_share=side,
                      iters=a.plane_iters, min_side_share=a.plane_side, deg_from_camera_up=ang, up_cameras=up_cams.tolist())
    return (p0, nv, inl), nv


# ====================================================================== 3. layout
class Layout:
    """How points get (u, v) on a chart, and the chart's start. kind: 'cyl' | 'sph' (a sheet around an axis) | 'height'."""

    def __init__(self, kind, frame, **k): self.kind = kind; self.frame = frame; self.__dict__.update(k)

    # frame = (origin o, e1, e2, n): local (a, b, h) = ((X - o).e1, (X - o).e2, (X - o).n)
    def local(self, X):
        o, e1, e2, n = self.frame; Y = X - o; return np.stack([Y @ e1, Y @ e2, Y @ n], 1)

    def world(self, L):
        o, e1, e2, n = self.frame; return o + L[:, :1] * e1 + L[:, 1:2] * e2 + L[:, 2:3] * n

    def polar(self, L):
        """(azimuth, row coordinate w, depth rho). cyl: w = height, rho = distance from the axis. sph: w = elevation seen
        from the axis point at mid height, rho = distance from it. hyb (as fit.py): height below the shoulder hs, above it
        the elevation seen from the axis point at shoulder height, as arc length at radius rs (so w is continuous)."""
        a, b, h = L.T; th = np.arctan2(b - self.b0, a - self.a0); r = np.hypot(a - self.a0, b - self.b0)
        if self.kind == 'cyl': return th, h, r
        if self.kind == 'sph': w = np.arctan2(h - self.hc, r); return th, w, np.hypot(r, h - self.hc)
        up = h > self.hs; w = np.where(up, self.hs + self.rs * np.arctan2(h - self.hs, r), h); return th, w, np.where(up, np.hypot(r, h - self.hs), r)

    def from_polar(self, th, w, rho):
        if self.kind == 'cyl': return np.stack([self.a0 + rho * np.cos(th), self.b0 + rho * np.sin(th), w], 1)
        if self.kind == 'sph': return np.stack([self.a0 + rho * np.cos(w) * np.cos(th), self.b0 + rho * np.cos(w) * np.sin(th), self.hc + rho * np.sin(w)], 1)
        ph = np.maximum(w - self.hs, 0) / self.rs; up = w > self.hs
        return np.stack([self.a0 + rho * np.cos(ph) * np.cos(th), self.b0 + rho * np.cos(ph) * np.sin(th), np.where(up, self.hs + rho * np.sin(ph), w)], 1)

    def u_of(self, th):
        if self.closed: return np.mod(th - self.thL, 2 * np.pi) / (2 * np.pi)
        return np.clip(np.mod(th - self.thL, 2 * np.pi) / self.span, 0, 1)

    def th_of(self, u): return self.thL + u * (2 * np.pi if self.closed else self.span)

    def lim(self, u):
        return np.interp(u, self.bu, self.foot, period=1.0 if self.closed else None), np.interp(u, self.bu, self.top, period=1.0 if self.closed else None)

    def uv(self, X):
        L = self.local(X)
        if self.kind == 'height': return np.clip((L[:, 0] - self.p0) / self.dp, 0, 1), np.clip((L[:, 1] - self.q0) / self.dq, 0, 1)
        th, w, _ = self.polar(L); u = self.u_of(th); f, t = self.lim(u)
        return u, np.clip((w - f) / np.maximum(t - f, 1e-9), 0, 1)

    def depth(self, X):       # the coordinate a single sheet must hold one value of per direction
        L = self.local(X); return L[:, 2] if self.kind == 'height' else self.polar(L)[2]


def widest_gap(th, w, bins, cover, rows):
    """The widest run of azimuth bins whose points cover fewer than `cover` of the `rows` row bins (foot to top):
    a bin with a thin scatter of strays is as empty as one with nothing."""
    b = ((np.mod(th, 2 * np.pi)) / (2 * np.pi) * bins).astype(int) % bins; lo, hi = np.percentile(w, [1, 99])
    r = np.clip(((w - lo) / max(hi - lo, 1e-12) * rows).astype(int), 0, rows - 1); occ = np.zeros((bins, rows), bool); occ[b, r] = True
    h = occ.mean(1); empty = h < cover
    if empty.all() or not empty.any(): return 0, 0, h
    best, start = 0, 0
    e2 = np.r_[empty, empty]; run = 0
    for i in range(2 * bins):
        run = run + 1 if e2[i] else 0
        if run > best and run <= bins: best, start = run, i - run + 1
    return best, start % bins, h


def make_sheet(kind, frame, X, s, a, centre=None):
    lay = Layout(kind, frame); L = lay.local(X)
    lay.a0, lay.b0 = (np.median(L[:, 0]), np.median(L[:, 1])) if centre is None else centre; lo, hi = np.percentile(L[:, 2], [1, 99]); lay.hc = 0.5 * (lo + hi)
    lay.hs = lo + a.shoulder * (hi - lo); band = np.abs(L[:, 2] - lay.hs) < 0.05 * (hi - lo)
    lay.rs = float(np.median(np.hypot(L[band, 0] - lay.a0, L[band, 1] - lay.b0))) if band.sum() > 10 else 0.25 * (hi - lo)
    th, w, rho = lay.polar(L); bins = a.az_bins
    g, start, hist = widest_gap(th, w, bins, a.gap_cover, a.cover_rows); gap_deg = 360.0 * g / bins
    lay.closed = gap_deg < a.min_gap_deg; lay.gap_deg = gap_deg
    if lay.closed: lay.thL = 0.0; lay.span = 2 * np.pi
    else:
        g0 = (start + g) % bins * 2 * np.pi / bins                        # first non-empty bin after the gap
        rel = np.mod(th - g0, 2 * np.pi); keep = rel < 2 * np.pi - g * 2 * np.pi / bins + 1e-9
        lo_, hi_ = np.percentile(rel[keep], [0.2, 99.8]); lay.thL = g0 + lo_; lay.span = hi_ - lo_
    u = lay.u_of(th); nb = a.row_bins; bu = (np.arange(nb) + 0.5) / nb; ib = np.minimum((u * nb).astype(int), nb - 1)
    foot = np.full(nb, np.nan); top = np.full(nb, np.nan); cnt = np.bincount(ib, minlength=nb)
    for i in range(nb):
        if cnt[i] >= a.min_support: foot[i], top[i] = np.percentile(w[ib == i], [a.edge_q, 100 - a.edge_q])
    ok = ~np.isnan(foot)
    if ok.sum() < 2: return None
    per = 1.0 if lay.closed else None
    foot = np.interp(bu, bu[ok], foot[ok], period=per); top = np.interp(bu, bu[ok], top[ok], period=per)
    mode = 'wrap' if lay.closed else 'nearest'
    lay.foot = ndimage.median_filter(foot, 3, mode=mode); lay.top = ndimage.median_filter(top, 3, mode=mode); lay.bu = bu
    lay.bins_without_rows = int((~ok).sum())
    return lay


def make_height(frame_hint, X, cams, a):
    c = np.median(X, 0); w, V = np.linalg.eigh(np.cov((X - c).T)); n = V[:, 0]
    if (np.mean([cm['C'] for cm in cams], 0) - c) @ n < 0: n = -n
    e1, e2 = ortho(n); L2 = np.c_[(X - c) @ e1, (X - c) @ e2]; best = None
    for ang in np.radians(np.arange(0, 90, 2.0)):          # the in-plane rotation with the smallest box (rotating calipers, coarse)
        ca, sa = math.cos(ang), math.sin(ang); Q = L2 @ np.array([[ca, -sa], [sa, ca]]); ar = np.prod(np.percentile(Q, 99.5, 0) - np.percentile(Q, 0.5, 0))
        if best is None or ar < best[0]: best = (ar, ang)
    ca, sa = math.cos(best[1]), math.sin(best[1]); e1, e2 = ca * e1 + sa * e2, -sa * e1 + ca * e2
    lay = Layout('height', (c, e1, e2, n)); L = lay.local(X)
    (p0, q0), (p1, q1) = np.percentile(L[:, :2], 0.5, axis=0), np.percentile(L[:, :2], 99.5, axis=0)
    lay.p0, lay.q0, lay.dp, lay.dq = p0, q0, max(p1 - p0, 1e-9), max(q1 - q0, 1e-9); lay.closed = False; lay.eig = w.tolist()
    return lay


def measure_layout(lay, X, s, a):
    """Measured on the points: the share of points in cells where the depth coordinate has two layers (a gap of more
    than --layer-gap x s, at least 3 points each side), and the share of the chart's cells with no point at all."""
    u, v = lay.uv(X); dep = lay.depth(X); n = len(X)
    nc = max(4, int(round(math.sqrt(n / a.layout_cell_points))))
    if lay.kind == 'height': ncu = max(2, int(round(nc * math.sqrt(lay.dp / lay.dq)))); ncv = max(2, int(round(nc * nc / ncu)))
    else: ncu = ncv = nc
    cu = np.minimum((u * ncu).astype(int), ncu - 1); cv = np.minimum((v * ncv).astype(int), ncv - 1); cell = cv * ncu + cu
    o = np.lexsort((dep, cell)); cs, ds = cell[o], dep[o]; multi = np.zeros(ncu * ncv, bool); gap = a.layer_gap * s
    bnd = np.r_[0, np.nonzero(np.diff(cs))[0] + 1, len(cs)]
    for i in range(len(bnd) - 1):
        d = ds[bnd[i]:bnd[i + 1]]
        if len(d) < 6: continue
        g = np.diff(d); g[:2] = 0; g[-2:] = 0           # at least 3 points on each side of the gap
        if g.max(initial=0) > gap: multi[cs[bnd[i]]] = True
    count = np.bincount(cell, minlength=ncu * ncv)
    m_share = float(count[multi].sum() / n); e_share = float((count == 0).mean())
    return dict(kind=lay.kind, closed=bool(lay.closed), cells=[int(ncu), int(ncv)], two_layer_point_share=m_share, empty_cell_share=e_share,
                score=m_share + e_share, **({'gap_deg': lay.gap_deg} if lay.kind != 'height' else {}))


def choose_layout(X, frame, s, cams, a, name):
    cands = {}; axis_rec = {}
    for k in a.layouts.split(','):
        if k == 'height': lay = make_height(frame, X, cams, a)
        else:
            lay = make_sheet(k, frame, X, s, a)                               # the axis through the robust centroid
            if lay is None: continue
            m0 = measure_layout(lay, X, s, a); axis_rec[k] = dict(centroid=[float(lay.a0), float(lay.b0)], centroid_score=m0['score'])
            if a.axis == 'search':
                # the centroid of a shell seen from one side lies near its seen surface; the axis is instead placed where
                # the same layout score is lowest, on a grid over the middle of the object's footprint
                L = Layout(k, frame).local(X); lo, hi = np.percentile(L[:, :2], [20, 80], axis=0); best = (m0['score'], lay)
                for ca in np.linspace(lo[0], hi[0], a.axis_grid):
                    for cb in np.linspace(lo[1], hi[1], a.axis_grid):
                        l2 = make_sheet(k, frame, X, s, a, (ca, cb))
                        if l2 is None: continue
                        sc = measure_layout(l2, X, s, a)['score']
                        if sc < best[0] - 1e-9: best = (sc, l2)
                lay = best[1]; axis_rec[k].update(searched=[float(lay.a0), float(lay.b0)], searched_score=best[0], grid=a.axis_grid, range='20th to 80th percentile of the footprint')
        cands[k] = (lay, measure_layout(lay, X, s, a))
    lowest = min(v[1]['score'] for v in cands.values())
    best = next(k for k in ('height', 'cyl', 'hyb', 'sph') if k in cands and cands[k][1]['score'] <= lowest + a.layout_tie)   # near-ties go to the simpler chart
    log('3. layout of %s (%d points), score = two-layer point share + empty cell share (lower is better):' % (name, len(X)))
    for k, (lay, m) in cands.items():
        log('     %-6s %s  two-layer %.1f %%  empty cells %.1f %%  score %.3f%s' % (k, '<- chosen' if k == best else '         ', 100 * m['two_layer_point_share'], 100 * m['empty_cell_share'], m['score'],
            '' if k == 'height' else ('  closed all round (widest azimuth gap %.0f deg)' % m['gap_deg'] if m['closed'] else '  open, azimuth gap %.0f deg' % m['gap_deg'])))
    for k, r in axis_rec.items():
        if 'searched' in r: log('     axis of %s: centroid score %.3f -> searched %s score %.3f (moved %.3g from the centroid)' % (k, r['centroid_score'], 'position' , r['searched_score'], math.hypot(r['searched'][0] - r['centroid'][0], r['searched'][1] - r['centroid'][1])))
    lay, m = cands[best]; warn = []
    if m['two_layer_point_share'] > a.warn_two_layer:
        warn.append('%.0f %% of the points lie in cells with two layers: one %s chart cannot hold this object; it will bridge or fold' % (100 * m['two_layer_point_share'], best))
    if m['empty_cell_share'] > a.warn_empty:
        warn.append('%.0f %% of the chart\'s layout cells hold no point: the chart is unsupported there' % (100 * m['empty_cell_share']))
    for w in warn: log('   WARNING (%s): %s' % (name, w))
    return lay, dict(axis=axis_rec, chosen=best, rule='lowest score = two_layer_point_share + empty_cell_share; scores within %g of the lowest count as a tie, ties go to height field, then cylindrical, then hybrid, then spherical' % a.layout_tie, candidates={k: v[1] for k, v in cands.items()}, warnings=warn)


# ====================================================================== 4. fit
class Chart:
    def __init__(self, name, lay, nu, nv, label=None):
        self.name, self.lay, self.label = name, lay, label
        self.au = bs.Periodic(nu) if lay.closed else bs.Open.uniform(nu); self.av = bs.Open.uniform(nv); self.net = None

    @property
    def height(self): return self.lay.kind == 'height'

    def eval(self, u, v, deriv=False): return bs.evaluate(self.au, self.av, self.net, u, v, deriv)

    def start(self, X, s, a):
        """Height field: the best plane (h = 0). Sheet: a smoothed radial envelope (outermost point per direction cell)."""
        lay = self.lay; su, sv = self.au.sites(), self.av.sites(); U, V = np.meshgrid(su, sv)
        if self.height:
            L = np.stack([lay.p0 + U.ravel() * lay.dp, lay.q0 + V.ravel() * lay.dq, np.zeros(U.size)], 1)
            self.net = lay.world(L).reshape(self.av.n, self.au.n, 3); return
        u, v = lay.uv(X); rho = lay.depth(X); g = a.envelope_cells
        cu = np.minimum((u * g).astype(int), g - 1); cv = np.minimum((v * g).astype(int), g - 1); env = np.full((g, g), np.nan)
        for c in np.unique(cv * g + cu):
            sel = (cv * g + cu) == c
            if sel.sum() >= 2: env[c // g, c % g] = np.percentile(rho[sel], 90)
        known = ~np.isnan(env)
        if not known.any(): env[:] = np.median(rho); known[:] = True
        ii = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        env = env[tuple(ii)]; env = ndimage.gaussian_filter(env, 1.0, mode=('wrap', 'nearest') if lay.closed else 'nearest')
        cc = (np.arange(g) + 0.5) / g
        r = np.array([np.interp(su, cc, env[j], period=1.0 if lay.closed else None) for j in range(g)])     # g x nu
        r = np.array([np.interp(sv, cc, r[:, i]) for i in range(self.au.n)]).T                              # nv x nu
        th = lay.th_of(U); f, t = lay.lim(U); w = f + V * (t - f)
        nodes = lay.world(lay.from_polar(th.ravel(), w.ravel(), r.ravel())).reshape(self.av.n, self.au.n, 3)
        self.net = nodes                       # control points at their nodes: a smooth start (the fit corrects it)

    def to_json(self, support, unit_mm, ndig=6):
        au, av = self.au, self.av
        o, e1, e2, n = self.lay.frame
        fr = dict(origin=o.tolist(), e1=e1.tolist(), e2=e2.tolist(), normal=n.tolist())
        if not self.height: fr.update(axis_point=self.lay.world(np.array([[self.lay.a0, self.lay.b0, 0.0]]))[0].tolist(), axis_dir=n.tolist(),
                                     centre_height=float(self.lay.hc), shoulder_height=float(self.lay.hs), shoulder_radius=float(self.lay.rs), theta_start=float(self.lay.thL), theta_span=float(self.lay.span))
        else: fr.update(extent=[float(self.lay.p0), float(self.lay.q0), float(self.lay.dp), float(self.lay.dq)])
        return dict(name=self.name, label=self.label, kind={'cyl': 'sheet-cylindrical', 'sph': 'sheet-spherical', 'hyb': 'sheet-hybrid', 'height': 'height-field'}[self.lay.kind],
                    frame=fr, degree=3, nu=au.n, nv=av.n,
                    u=dict(periodic=True, knots='uniform, control point i at u = i / nu') if au.periodic else dict(periodic=False, knots=np.round(au.U, 8).tolist()),
                    v=dict(periodic=False, knots=np.round(av.U, 8).tolist()),
                    net=np.round(self.net, ndig).tolist(), support=np.round(support, 3).tolist(),
                    layout='u along the azimuth %s, v from the foot to the top per azimuth (%s rows)' % ('closed all round' if self.lay.closed else 'across its observed range', {'cyl': 'height', 'sph': 'elevation', 'hyb': 'height below the shoulder, elevation above'}[self.lay.kind]) if not self.height
                    else 'u, v linear over the plane box (e1, e2), the net\'s offset along normal is the height', units_mm=unit_mm)


def reparam(ch, X, u, v, steps=1, maxstep=0.05):
    """Parameter correction: Gauss-Newton steps towards each point's closest point on the chart (sheets only)."""
    for _ in range(steps):
        S, Su, Sv = ch.eval(u, v, True); r = X - S
        a11 = (Su * Su).sum(1); a12 = (Su * Sv).sum(1); a22 = (Sv * Sv).sum(1); b1 = (Su * r).sum(1); b2 = (Sv * r).sum(1)
        det = a11 * a22 - a12 ** 2; det = np.where(np.abs(det) < 1e-18, 1e-18, det)
        du = np.clip((a22 * b1 - a12 * b2) / det, -maxstep, maxstep); dv = np.clip((a11 * b2 - a12 * b1) / det, -maxstep, maxstep)
        u = np.mod(u + du, 1.0) if ch.au.periodic else np.clip(u + du, 0, 1); v = np.clip(v + dv, 0, 1)
    return u, v


def residual(ch, X, u, v):
    if ch.height:
        n = ch.lay.frame[3]; S, Su, Sv = ch.eval(u, v, True); nn = np.cross(Su, Sv); nn /= np.linalg.norm(nn, axis=1, keepdims=True) + 1e-18
        # vertical offset, turned into the distance to the tangent plane
        return np.abs(((X - S) @ n) * (nn @ n))
    return np.linalg.norm(X - ch.eval(u, v), axis=1)


def solve(ch, X, u, v, w, lam, extra=None):
    A = bs.design(ch.au, ch.av, u, v); W = sparse.diags(w); Rm = bs.second_differences(ch.au, ch.av); RtR = (Rm.T @ Rm).tocsc()
    AtA = (A.T @ W @ A).tocsc(); b = A.T @ (w[:, None] * X)
    if extra is not None:
        uve, Xe, we = extra; Ae = bs.design(ch.au, ch.av, uve[:, 0], uve[:, 1]); extra = (Ae, Xe, we)       # rows for the current net
        AtA = AtA + (Ae.T @ sparse.diags(we) @ Ae).tocsc(); b = b + Ae.T @ (we[:, None] * Xe)
    scale = AtA.diagonal().sum() / max(RtR.diagonal().sum(), 1e-18)
    M = (AtA + lam * scale * RtR + 1e-10 * scale * sparse.identity(AtA.shape[0])).tocsc()
    if ch.height:
        o, e1, e2, n = ch.lay.frame; base = ch.net.reshape(-1, 3) - np.outer((ch.net.reshape(-1, 3) - o) @ n, n)
        # unknown = height of each control point; in-plane parts stay where they are (linear precision of the basis)
        bh = A.T @ (w * ((X - o) @ n)) if extra is None else A.T @ (w * ((X - o) @ n)) + extra[0].T @ (extra[2] * ((extra[1] - o) @ n))
        # in-plane coordinates of the data relative to the in-plane net do not enter: the point's (u, v) is its foot point
        h = spsolve(M, bh); return base + np.outer(h, n)
    return np.column_stack([spsolve(M, b[:, k]) for k in range(3)])


def tukey(r, cmin):
    s = 1.4826 * np.median(r) + 1e-18; c = max(4.685 * s, cmin)
    return np.where(r < c, (1 - (r / c) ** 2) ** 2, 0.0), c


def support_of(ch, u, v, w):
    A = bs.design(ch.au, ch.av, u, v); return np.asarray((A.T @ (w > 0).astype(float))).reshape(ch.av.n, ch.au.n)


def cell_counts(ch, u, v, inl):
    su, sv = ch.au.spans(), ch.av.spans()
    iu = np.clip(np.searchsorted(su[:, 0], u[inl], side='right') - 1, 0, len(su) - 1); iv = np.clip(np.searchsorted(sv[:, 0], v[inl], side='right') - 1, 0, len(sv) - 1)
    C = np.zeros((len(sv), len(su)), int); np.add.at(C, (iv, iu), 1); return C, iu, iv, su, sv


def refine(ch, u, v, r, inl, a, tol):
    """Knot insertion where the data asks for it and carries it: a span is split at its middle when at least half of
    its knot cells (span x span of the other direction) hold >= --min-support inlier points, AND the median residual of
    its inliers is above `tol` (= --refine-ratio x the chart's median inlier residual, at least --refine-floor x s): the
    worse spans of the level. Whether a level helped at all is decided on the held-out points, not here. A periodic
    direction is doubled when the rule holds for at least half its spans."""
    C, iu, iv, su, sv = cell_counts(ch, u, v, inl); ri = r[inl]; added = [0, 0]
    def med(idx, k): sel = idx == k; return np.median(ri[sel]) if sel.any() else 0.0
    cu = [k for k in range(len(su)) if np.median(C[:, k] >= a.min_support) >= 0.5 and med(iu, k) > tol]
    cv = [k for k in range(len(sv)) if np.median(C[k, :] >= a.min_support) >= 0.5 and med(iv, k) > tol]
    if ch.au.n + (len(cu) if not ch.au.periodic else ch.au.n) > a.max_net[0]: cu = []
    if ch.av.n + len(cv) > a.max_net[1]: cv = []
    if ch.au.periodic:
        if len(cu) >= 0.5 * len(su): ch.au, ch.net = bs.double_periodic(ch.au, ch.net, 1); added[0] = ch.au.n // 2
    else:
        for k in sorted(cu, reverse=True):
            t = 0.5 * (su[k, 0] + su[k, 1]); ch.au, ch.net = ch.au.insert(t, ch.net, 1); added[0] += 1
    for k in sorted(cv, reverse=True):
        t = 0.5 * (sv[k, 0] + sv[k, 1]); ch.av, ch.net = ch.av.insert(t, ch.net, 0); added[1] += 1
    blocked = dict(u=int(sum(1 for k in range(len(su)) if med(iu, k) > tol and np.median(C[:, k] >= a.min_support) < 0.5)),
                   v=int(sum(1 for k in range(len(sv)) if med(iv, k) > tol and np.median(C[k, :] >= a.min_support) < 0.5)))
    return added, blocked


def fit_chart(ch, X, s, a, budget, rng, cams=None, obs=None, unit_mm=None):
    """Coarse to fine. Returns the record of the fit; ch holds the best state (by the held-out median distance)."""
    n = len(X); hold = rng.random(n) < a.holdout; tr, ho = np.nonzero(~hold)[0], np.nonzero(hold)[0]
    Xt, Xh = X[tr], X[ho]; u, v = ch.lay.uv(Xt)
    tol = a.tol * s; cmin = a.tukey_min * s
    ch.start(Xt, s, a); w = np.ones(len(Xt)); levels = []; best = None; stop = None; still = 0; fs_pen = None
    for level in range(a.max_levels):
        try:
            for it in range(a.irls):
                budget.tick()
                net = solve(ch, Xt, u, v, w, a.smooth, fs_pen)
                if not np.all(np.isfinite(net)): raise Halt('nan', 'non-finite control points at level %d, solve %d: not kept' % (level, it))
                prev = ch.net; ch.net = net.reshape(ch.av.n, ch.au.n, 3)
                if not ch.height: u, v = reparam(ch, Xt, u, v, a.reparam_steps)
                r = residual(ch, Xt, u, v)
                if not np.all(np.isfinite(r)): ch.net = prev; raise Halt('nan', 'non-finite residuals at level %d: state before kept' % level)
                w, c = tukey(r, cmin)
        except Halt as h:
            stop = dict(reason=h.reason, detail=h.detail, level=level); break
        rh = distances(ch, Xh, s, cap=400)[0]; hs = summary(rh, tol, unit_mm); ts = summary(r[w > 0], tol, unit_mm)
        rec = dict(level=level, net=[ch.au.n, ch.av.n], control_points=ch.au.n * ch.av.n, tukey_cutoff=float(c), inliers=int((w > 0).sum()), train_inliers=ts, held_out=hs)
        if cams is not None and a.freespace > 0:
            fs = freespace([ch], cams, *obs, s, a, want_pairs=True); rec['freespace_violations'] = fs['share']
            fs_pen = penalty_rows(ch, fs, a) if fs['count'] else None
        levels.append(rec)
        log('   %s level %d: net %d x %d, inliers %d (Tukey cut %.3g), train %s | held-out %s' % (ch.name, level, ch.au.n, ch.av.n, (w > 0).sum(), c, fmt(ts, unit_mm), fmt(hs, unit_mm))
            + ('  | free-space violations %.2f %%' % (100 * rec['freespace_violations']) if 'freespace_violations' in rec else ''))
        if best is None or hs['median'] < best['score'] * (1 - a.min_gain):
            best = dict(score=hs['median'], level=level, au=ch.au, av=ch.av, net=ch.net.copy(), u=u.copy(), v=v.copy(), w=w.copy()); still = 0
        else:
            still += 1
            if still >= a.patience: stop = dict(reason='converged', detail='held-out median did not improve by %.0f %% for %d level(s)' % (100 * a.min_gain, a.patience), level=level); break
        rtol = max(a.refine_ratio * float(np.median(r[w > 0])), a.refine_floor * s); rec['refine_tol'] = rtol
        added, blocked = refine(ch, u, v, r, w > 0, a, rtol); rec['knots_added'] = added; rec['spans_wanting_refinement_without_support'] = blocked
        if not any(added):
            stop = dict(reason='data_cap', detail='no span both above the refinement tolerance and carried by --min-support points per knot cell (spans that wanted more but lacked points: u %d, v %d)' % (blocked['u'], blocked['v']), level=level); break
        # knot insertion keeps the surface; the parameters of the points stay valid
    else:
        stop = dict(reason='level_cap', detail='%d levels' % a.max_levels, level=a.max_levels - 1)
    if best is None: raise SystemExit('%s: no level finished (%s): nothing to write' % (ch.name, stop))
    ch.au, ch.av, ch.net = best['au'], best['av'], best['net']
    u, v, w = best['u'], best['v'], best['w']; stop['best_level'] = best['level']
    return dict(levels=levels, stop=stop, train_index=tr, hold_index=ho, u=u, v=v, w=w)


# ====================================================================== 5. free space
def chart_lattice(ch, s, cap=360):
    S0, _, _ = bs.grid(ch.au, ch.av, ch.net, 24, 24)
    lu = np.linalg.norm(np.diff(S0, axis=1), axis=-1).sum(1).max(); lv = np.linalg.norm(np.diff(S0, axis=0), axis=-1).sum(0).max()
    nu_s = int(np.clip(lu / (0.5 * s), 16, cap)); nv_s = int(np.clip(lv / (0.5 * s), 16, cap))
    (S, Su, Sv), U, V = bs.grid(ch.au, ch.av, ch.net, nu_s, nv_s, deriv=True)
    N = np.cross(Su, Sv); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-18
    step = np.maximum(np.linalg.norm(Su, axis=-1) / nu_s, np.linalg.norm(Sv, axis=-1) / nv_s)      # lattice spacing at each sample
    return S, N, step, U, V


def freespace(charts, cams, oc, op, X, s, a, want_pairs=False, use_trim=False):
    """For every (camera, point) observation: is there fitted surface in front of the point along that camera's ray,
    by more than --fs-tol x s? A z-buffer of the charts' lattices (index of the nearest sample per pixel) is drawn per
    camera, at a pixel size that keeps the lattice's projected edges below one pixel. The pair is then judged on the
    point's own ray: it meets the tangent plane of the sample that won its pixel, the hit must lie within that sample's
    footprint (else the pixel only grazed a silhouette), and must be nearer than the point by more than tol. Pairs whose
    ray meets that surface within --fs-min-cos of edge-on are not judged (counted as grazing)."""
    tol = a.fs_tol * s; lat = []
    alive = []
    for ci, ch in enumerate(charts):
        S, N, step, U, V = chart_lattice(ch, s)
        lat.append((S.reshape(-1, 3), N.reshape(-1, 3), step.ravel(), np.c_[U.ravel(), V.ravel()], np.full(S.shape[0] * S.shape[1], ci), S.shape[:2]))
        tm = getattr(ch, 'trim', None) if use_trim else None; g = tm.shape[0] if tm is not None else 1
        alive.append(np.ones(S.shape[0] * S.shape[1], bool) if tm is None else tm[np.minimum((V.ravel() * g).astype(int), g - 1), np.minimum((U.ravel() * g).astype(int), g - 1)] == 1)
    alive = np.concatenate(alive)
    P = np.concatenate([l[0] for l in lat]); NR = np.concatenate([l[1] for l in lat]); ST = np.concatenate([l[2] for l in lat])
    UV = np.concatenate([l[3] for l in lat]); CI = np.concatenate([l[4] for l in lat])
    edges = []; off = 0
    for l in lat:
        nv_s, nu_s = l[5]; idx = np.arange(nv_s * nu_s).reshape(nv_s, nu_s) + off; edges.append(np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()]); edges.append(np.c_[idx[:-1].ravel(), idx[1:].ravel()]); off += nv_s * nu_s
    edges = np.concatenate(edges)
    viol = np.zeros(len(oc), bool); occ = np.full(len(oc), -1); zx_all = np.zeros(len(oc)); zs_all = np.zeros(len(oc)); checked = np.zeros(len(oc), bool); graze = np.zeros(len(oc), bool); gap_all = np.full(len(oc), -np.inf)
    for k, c in enumerate(cams):
        sel = np.nonzero(oc == k)[0]
        if not len(sel): continue
        q = (P - c['C']) @ c['R'].T; front = (q[:, 2] > 1e-9) & alive                # a trimmed-away sample hides nothing
        x = c['fx'] * q[:, 0] / np.where(front, q[:, 2], 1) + c['cx']; y = c['fy'] * q[:, 1] / np.where(front, q[:, 2], 1) + c['cy']
        e = edges[front[edges[:, 0]] & front[edges[:, 1]]]
        el = np.hypot(x[e[:, 0]] - x[e[:, 1]], y[e[:, 0]] - y[e[:, 1]]); inimg = (x[e[:, 0]] >= 0) & (x[e[:, 0]] < c['W']) & (y[e[:, 0]] >= 0) & (y[e[:, 0]] < c['H'])
        px = max(1.0, 1.05 * float(np.percentile(el[inimg], a.fs_edge_q))) if inimg.any() else 1.0
        Wb, Hb = int(math.ceil(c['W'] / px)), int(math.ceil(c['H'] / px))
        bx, by = np.floor(x / px).astype(int), np.floor(y / px).astype(int); ok = front & (bx >= 0) & (bx < Wb) & (by >= 0) & (by < Hb)
        pix = by[ok] * Wb + bx[ok]; z = q[ok, 2]; ids = np.nonzero(ok)[0]
        if not len(pix): continue
        o = np.lexsort((z, pix)); pix, z, ids = pix[o], z[o], ids[o]; first = np.r_[True, pix[1:] != pix[:-1]]
        ib = np.full(Wb * Hb, -1); ib[pix[first]] = ids[first]
        Xp = X[op[sel]]; qp = (Xp - c['C']) @ c['R'].T; zx = qp[:, 2]
        xb = np.floor((c['fx'] * qp[:, 0] / zx + c['cx']) / px).astype(int); yb = np.floor((c['fy'] * qp[:, 1] / zx + c['cy']) / px).astype(int)
        ins = (zx > 0) & (xb >= 0) & (xb < Wb) & (yb >= 0) & (yb < Hb); j = sel[ins]; checked[j] = True
        hit_id = ib[yb[ins] * Wb + xb[ins]]; has = hit_id >= 0; j, hid = j[has], hit_id[has]
        Xj = X[op[j]]; dX = np.linalg.norm(Xj - c['C'], axis=1); wdir = (Xj - c['C']) / dX[:, None]
        Sj, Nj = P[hid], NR[hid]; cs = np.einsum('pc,pc->p', Nj, wdir); gr = np.abs(cs) < a.fs_min_cos
        tt = np.einsum('pc,pc->p', Nj, Sj - c['C']) / np.where(gr, 1.0, cs); hit = c['C'] + tt[:, None] * wdir
        rad = np.maximum(1.5 * ST[hid], 1.5 * px * np.abs(tt) / c['fx'])
        valid = ~gr & (tt > 0) & (np.linalg.norm(hit - Sj, axis=1) < rad); vio = valid & (tt < dX - tol)
        gap_all[j] = np.where(valid, dX - tt, -np.inf)
        graze[j[gr]] = True; viol[j] = vio; occ[j] = np.where(vio, hid, -1); zx_all[j] = dX; zs_all[j] = np.where(vio, tt, 0)
    nchk = int(checked.sum()); cnt = int(viol.sum())
    out = dict(pairs=int(len(oc)), checked=nchk, grazing_not_judged=int(graze.sum()), count=cnt, share=cnt / max(nchk, 1), tol=tol, points_with_violation=int(len(np.unique(op[viol]))),
               min_cos=a.fs_min_cos,
               by_tolerance=[dict(tol_s=k, tol=k * s, share=float((gap_all[checked] > k * s).mean()) if nchk else 0.0) for k in (1, 2, 4, 8, 16)])
    per = []
    for ci, ch in enumerate(charts):
        m = viol & (occ >= 0); m[m] = CI[occ[m]] == ci; uv = UV[occ[m]]
        H = np.histogram2d(uv[:, 1], uv[:, 0], bins=[8, 8], range=[[0, 1], [0, 1]])[0].astype(int) if len(uv) else np.zeros((8, 8), int)
        per.append(dict(chart=ch.name, count=int(m.sum()), where_v_rows_u_cols_8x8=H.tolist()))
    out['per_chart'] = per
    if want_pairs:
        m = viol & (occ >= 0); Cc = np.array([c['C'] for c in cams])
        # distances along the ray: zx = to the point, zs = to the hiding surface
        out['pairs_data'] = dict(cam=oc[m], pt=op[m], uv=UV[occ[m]], chart=CI[occ[m]], zx=zx_all[m], zs=zs_all[m], P=P[occ[m]], C=Cc[oc[m]])
    return out


def penalty_rows(ch, fs, a):
    """Free space as extra least-squares rows: the surface point that hides an observed point is pulled back along
    its own camera ray to the point's depth plus --fs-push x the tolerance. Weight --freespace per row (data rows weigh 1)."""
    d = fs['pairs_data']; m = d['chart'] == 0
    if not m.any(): return None
    uv, S, C = d['uv'][m], d['P'][m], d['C'][m]
    T = C + (S - C) * ((d['zx'][m] + a.fs_push * fs['tol']) / np.linalg.norm(S - C, axis=1))[:, None]
    return uv, T, np.full(int(m.sum()), a.freespace)


# ====================================================================== distances, bridges
def distances(ch, X, s, cap=700):
    """Distance from points to the chart: nearest of a dense lattice, refined to the tangent plane there when close."""
    (S, Su, Sv), U, V = bs.grid(ch.au, ch.av, ch.net, *lattice_size(ch, s, cap), deriv=True)
    nrm = np.cross(Su, Sv).reshape(-1, 3); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-18; S = S.reshape(-1, 3)
    d, j = cKDTree(S).query(X, workers=-1); pl = np.abs(np.einsum('pc,pc->p', X - S[j], nrm[j]))
    step = np.median(np.linalg.norm(np.diff(S.reshape(V.shape[0], V.shape[1], 3), axis=1), axis=-1))
    return np.where(d < 3 * step, np.minimum(d, pl), d), np.c_[U.ravel()[j], V.ravel()[j]]


def lattice_size(ch, s, cap):
    S0, _, _ = bs.grid(ch.au, ch.av, ch.net, 24, 24)
    lu = np.linalg.norm(np.diff(S0, axis=1), axis=-1).sum(1).max(); lv = np.linalg.norm(np.diff(S0, axis=0), axis=-1).sum(0).max()
    return int(np.clip(lu / (0.3 * s), 32, cap)), int(np.clip(lv / (0.3 * s), 32, cap))


def bridges(ch, X, inl_uv, s, a):
    """Where the sheet stands on no point: a 48 x 48 grid of the chart's (u, v); a cell is supported when an inlier's
    foot point falls in it or a point lies within --bridge-dist x s of its lattice. Area-weighted share + the largest
    unsupported regions, by where they sit on the chart."""
    g = 48; (S, Su, Sv), U, V = bs.grid(ch.au, ch.av, ch.net, 4 * g, 4 * g, deriv=True)
    area = np.linalg.norm(np.cross(Su, Sv), axis=-1)
    dist = cKDTree(X).query(S.reshape(-1, 3), workers=-1)[0].reshape(U.shape)
    sc = float(np.median(cKDTree(X).query(X, 2, workers=-1)[0][:, 1]))       # the chart's own point spacing
    near = (dist < a.bridge_dist * sc).reshape(g, 4, g, 4).any(axis=(1, 3))
    A = area.reshape(g, 4, g, 4).sum(axis=(1, 3))
    lab, nl = ndimage.label(~near); regs = []
    for k in range(1, nl + 1):
        m = lab == k; jj, ii = np.nonzero(m); sh = float(A[m].sum() / A.sum())
        if sh < 0.002: continue
        uc, vc = (ii.mean() + 0.5) / g, (jj.mean() + 0.5) / g
        where = []
        if jj.max() == g - 1: where.append('top edge (v = 1)')
        if jj.min() == 0: where.append('foot (v = 0)')
        if ii.min() == 0 and not ch.au.periodic: where.append('left edge (u = 0)')
        if ii.max() == g - 1 and not ch.au.periodic: where.append('right edge (u = 1)')
        if not where: where.append('interior')
        regs.append(dict(area_share=round(sh, 4), u=[round(float(ii.min()) / g, 3), round(float(ii.max() + 1) / g, 3)], v=[round(float(jj.min()) / g, 3), round(float(jj.max() + 1) / g, 3)],
                         centre=[round(float(uc), 3), round(float(vc), 3)], where=', '.join(where)))
    regs.sort(key=lambda r: -r['area_share'])
    return dict(unsupported_area_share=float(A[~near].sum() / A.sum()), grid=g, rule='no point within %.2f x the chart\'s own point spacing (%.4g) of the lattice in that cell' % (a.bridge_dist, sc), chart_spacing=sc, regions=regs[:8], mask=(~near).astype(int), area=A)


def trim_mask(br, periodic_u, a):
    """The part of the chart's domain its points support: the supported cells of the 48 x 48 bridge grid, closed by
    --trim-close cells, enclosed holes filled (interpolation between supported parts is kept), pieces smaller than
    --trim-min-piece of the area dropped. 1 = kept. The chart's net is not changed: the trim only says where it holds."""
    sup = br['mask'] == 0; A = br['area']; g = sup.shape[0]
    if periodic_u: sup3 = np.concatenate([sup, sup, sup], 1)                  # wrap round in u
    else: sup3 = sup
    k = ndimage.binary_closing(np.pad(sup3, a.trim_close, mode='edge'), iterations=a.trim_close)[a.trim_close:-a.trim_close, a.trim_close:-a.trim_close] if a.trim_close else sup3
    k = ndimage.binary_fill_holes(k) | sup3
    if periodic_u: k = k[:, g:2 * g]
    lab, nl = ndimage.label(k); keep = np.zeros_like(k)
    for i in range(1, nl + 1):
        m = lab == i
        if A[m].sum() >= a.trim_min_piece * A.sum(): keep |= m
    return keep.astype(int)


# ====================================================================== main
def run(a):
    R = dict(tool='tools/recon/sfm_spline.py', what='cubic B-spline charts fitted to SfM points only: a geometric fit, nothing trained',
             model=os.path.abspath(a.model), labels=a.labels, args={k: v for k, v in vars(a).items()}); rng = np.random.default_rng(a.seed)
    budget = Budget(a.max_seconds, a.max_iters); stop = dict(reason='done', detail='every chart reached its own stop')
    D = load(a.model, a.track_len, a.labels); log('model: %d images, %d points, %d observations' % (len(D['cams']), len(D['X']), len(D['obs_cam'])))
    R['model_counts'] = dict(images=len(D['cams']), points=int(len(D['X'])), observations=int(len(D['obs_cam'])))
    keep, s, pre = clean(D, a, R); X = D['X']; um = a.unit_mm
    # the plane is searched among the points that passed track and reprojection error, before the density test: a sparse,
    # texture-poor table is what that test removes first, and RANSAC is robust on its own. Its inliers are kept.
    pidx = np.nonzero(pre)[0]; plane, up = support_plane(X[pidx], D['cams'], s, a, R, rng)
    if plane is not None:
        back = pidx[plane[2]][~keep[pidx[plane[2]]]]; keep[back] = True
        log('   plane inliers that the density test had dropped, kept again: %d' % len(back)); R['plane']['readmitted_from_density_test'] = int(len(back))
        plane = (plane[0], plane[1], np.abs((X[keep] - plane[0]) @ plane[1]) < a.plane_tol * s)
        p0, nv, _ = plane; e1, e2 = ortho(nv); frame = (p0, e1, e2, nv); hgt = (X - p0) @ nv
    else:
        lo = np.percentile(X[keep] @ up, 1); p0 = X[keep][np.argmin(np.abs(X[keep] @ up - lo))]; e1, e2 = ortho(up); frame = (p0, e1, e2, up); hgt = (X - p0) @ up
    charts, fits, recs = [], [], []
    ptol = (a.plane_tol * s) if plane is not None else 0.0
    groups = []
    if D['labels'] is None:
        cand = np.nonzero(keep & (hgt > a.above * s))[0]
        dk, jk = cKDTree(X[cand]).query(X[cand], a.graph_k + 1, workers=-1); lim = a.link * float(np.median(dk[:, -1]))
        ok = dk[:, 1:] < lim; rows = np.repeat(np.arange(len(cand)), a.graph_k)[ok.ravel()]; cols = jk[:, 1:].ravel()[ok.ravel()]
        G = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cand), len(cand)))
        nc, lab = connected_components(G, directed=False); sizes = np.bincount(lab); order = np.argsort(-sizes)
        log('3. objects: %d points above the plane (> %.3g); kNN graph (%d neighbours, links < %.1f x the median %d-th neighbour distance = %.3g): %d components, largest %s'
            % (len(cand), a.above * s, a.graph_k, a.link, a.graph_k, lim, nc, sizes[order[:5]].tolist()))
        R['objects'] = dict(above=a.above * s, graph_k=a.graph_k, link=lim, candidates=int(len(cand)), components=int(nc), largest=sizes[order[:10]].tolist())
        groups.append(('object', None, cand[lab == order[0]]))
    else:
        L = D['labels']; labs = sorted(set(L[keep].tolist()) - {-1})
        log('3. objects from labels: %s (points per label: %s; -1 ignored: %d)' % (labs, [int((L[keep] == l).sum()) for l in labs], int((L[keep] == -1).sum())))
        R['objects'] = dict(labels={str(l): int((L[keep] == l).sum()) for l in labs}, ignored=int((L[keep] == -1).sum()))
        for l in labs: groups.append(('label%d' % l, int(l), np.nonzero(keep & (L == l))[0]))
    if plane is not None and (D['labels'] is None or a.plane_chart == 'always'):
        pin = np.nonzero(keep)[0][plane[2]]
        groups.insert(0, ('plane', None, pin))
    obs_keep = keep[D['obs_pt']]; oc, op = D['obs_cam'][obs_keep], D['obs_pt'][obs_keep]
    R['charts'] = []
    for name, label, idx in groups:
        Xg = X[idx]
        if len(Xg) < 4 * a.min_support: log('   %s: only %d points, no chart' % (name, len(Xg))); continue
        if name == 'plane':
            lay = Layout('height', frame); L = lay.local(Xg); (pp, qq), (p1, q1) = np.percentile(L[:, :2], 0.5, 0), np.percentile(L[:, :2], 99.5, 0)
            lay.p0, lay.q0, lay.dp, lay.dq, lay.closed = pp, qq, p1 - pp, q1 - qq, False
            lrec = dict(chosen='height', rule='the support plane: a height field over it, inside the box of its points (0.5 to 99.5 %)')
            log('   plane chart: %.4g x %.4g%s' % (lay.dp, lay.dq, ' (%.0f x %.0f mm)' % (lay.dp * um, lay.dq * um) if um else ''))
        else:
            lay, lrec = choose_layout(Xg, frame, s, D['cams'], a, name)
        if lay.kind == 'height': asp = math.sqrt(lay.dp / lay.dq)
        else: asp = 1.0
        n0 = a.start_net; nu = max(4, int(round(n0 * asp))); nv = max(4, int(round(n0 / asp)))
        ch = Chart(name, lay, nu, nv, label)
        log('4. fit %s: start %d x %d (%s)' % (name, ch.au.n, ch.av.n, 'periodic u' if lay.closed else 'open'))
        obs = None
        if a.freespace > 0 and name != 'plane':
            obs = (oc, op, X)
        f = fit_chart(ch, Xg, s, a, budget, rng, D['cams'] if obs else None, obs, um)
        charts.append(ch); fits.append((idx, f)); recs.append(lrec)
        if f['stop']['reason'] in ('signal', 'wall_clock', 'iteration_cap', 'nan'):
            stop = dict(reason=f['stop']['reason'], detail=f['stop']['detail'] + ' (in %s)' % name); log('   stopping: %s' % stop); break
    # ---- measure every chart (its best state) and free space over all of them
    out_charts = []
    for ch, (idx, f), lrec in zip(charts, fits, recs):
        Xg = X[idx]; tol = a.tol * s
        d_all, uv = distances(ch, Xg, s); tr, ho = f['train_index'], f['hold_index']
        sup = support_of(ch, f['u'], f['v'], f['w']); br = bridges(ch, Xg[tr][f['w'] > 0], uv, s, a)
        rep = dict(name=ch.name, label=ch.label, layout=lrec, net=[ch.au.n, ch.av.n], control_points=ch.au.n * ch.av.n, points=int(len(idx)),
                   train=int(len(tr)), held_out=int(len(ho)),
                   residuals=dict(train_all=summary(d_all[tr], tol, um), train_inliers=summary(d_all[tr][f['w'] > 0], tol, um), held_out=summary(d_all[ho], tol, um),
                                  note='distance from each point to the chart (dense lattice + tangent plane), on the best level\'s net'),
                   control_points_without_support=int((sup < a.unsupported_below).sum()), unsupported_rule='summed basis weight of the inlier points < %.2g' % a.unsupported_below,
                   bridges={k: v for k, v in br.items() if k not in ('mask', 'area')}, levels=f['levels'], stop=f['stop'])
        log('   %s: net %d x %d, held-out %s; control points without support %d of %d; unsupported area %.1f %%'
            % (ch.name, ch.au.n, ch.av.n, fmt(rep['residuals']['held_out'], um), rep['control_points_without_support'], ch.au.n * ch.av.n, 100 * br['unsupported_area_share']))
        for r in br['regions'][:4]: log('      unsupported region: %.1f %% of the area, u %s, v %s, %s' % (100 * r['area_share'], r['u'], r['v'], r['where']))
        cj = ch.to_json(sup, um); cj['bridge_mask_48x48_v_rows'] = br['mask'].tolist()
        if a.trim:
            tm = trim_mask(br, ch.au.periodic, a); ch.trim = tm; A = br['area']; kept = tm == 1; jj, ii = np.nonzero(kept); g = tm.shape[0]
            cj['trim_mask_48x48_v_rows'] = tm.tolist(); cj['trimmed_domain'] = dict(u=[float(ii.min()) / g, float(ii.max() + 1) / g], v=[float(jj.min()) / g, float(jj.max() + 1) / g], rule='1 = kept: the chart holds only there (cells of the 48 x 48 grid of (u, v), row j = v)') if kept.any() else None
            rep['trim'] = dict(kept_area_share=float(A[kept].sum() / A.sum()), unsupported_share_before=float(br['unsupported_area_share']),
                               unsupported_share_of_kept=float(A[kept & (br['mask'] == 1)].sum() / max(A[kept].sum(), 1e-18)), close_cells=a.trim_close, min_piece=a.trim_min_piece)
            log('      trim: keeps %.1f %% of the area; unsupported share %.1f %% -> %.1f %% of what is kept' % (100 * rep['trim']['kept_area_share'], 100 * br['unsupported_area_share'], 100 * rep['trim']['unsupported_share_of_kept']))
        R['charts'].append(rep); out_charts.append(cj)
    if charts:
        fs = freespace(charts, D['cams'], oc, op, X, s, a, want_pairs=True); pdx = fs.pop('pairs_data')
        for ci, (p, cj) in enumerate(zip(fs['per_chart'], out_charts)):
            m = pdx['chart'] == ci; bm = np.array(cj['bridge_mask_48x48_v_rows']); g = bm.shape[0]; uv = pdx['uv'][m]
            on_bridge = bm[np.minimum((uv[:, 1] * g).astype(int), g - 1), np.minimum((uv[:, 0] * g).astype(int), g - 1)] == 1 if len(uv) else np.zeros(0, bool)
            p['share_on_unsupported_part'] = float(on_bridge.mean()) if len(uv) else 0.0
            p['unsupported_area_share_of_chart'] = R['charts'][ci]['bridges']['unsupported_area_share']
        log('5. free space: %d (camera, point) pairs of the kept points, %d inside an image (%d grazing, not judged); %d (%.2f %%) have fitted surface in front of the point by more than %.3g%s (%d points)'
            % (fs['pairs'], fs['checked'], fs['grazing_not_judged'], fs['count'], 100 * fs['share'], fs['tol'], ' = %.2f mm' % (fs['tol'] * um) if um else '', fs['points_with_violation']))
        log('      share by tolerance: ' + ', '.join('%g s%s: %.2f %%' % (b['tol_s'], ' (%.1f mm)' % (b['tol'] * um) if um else '', 100 * b['share']) for b in fs['by_tolerance']))
        for p in fs['per_chart']:
            if p['count']: log('      hidden by %s: %d, of which %.0f %% by its unsupported part (%.0f %% of its area); by chart region (v rows top = 1, u cols):' % (p['chart'], p['count'], 100 * p['share_on_unsupported_part'], 100 * p['unsupported_area_share_of_chart'])); [log('        ' + ' '.join('%5d' % x for x in row)) for row in p['where_v_rows_u_cols_8x8'][::-1]]
        if a.trim:
            ft = freespace(charts, D['cams'], oc, op, X, s, a, use_trim=True)
            log('      with the charts trimmed: %d (%.2f %%) by more than %.3g; by tolerance: %s' % (ft['count'], 100 * ft['share'], ft['tol'], ', '.join('%g s: %.2f %%' % (b['tol_s'], 100 * b['share']) for b in ft['by_tolerance'])))
            fs['trimmed'] = dict(share=ft['share'], count=ft['count'], by_tolerance=ft['by_tolerance'], per_chart=[dict(chart=p['chart'], count=p['count']) for p in ft['per_chart']])
        R['freespace'] = dict(fs, rule='a camera that observed a point saw it unoccluded: z-buffer of every chart per camera picks the nearest lattice sample in the point\'s pixel; violation = the point\'s ray meets that sample\'s tangent plane inside its footprint and nearer than the point by more than tol',
                              penalty_weight=a.freespace, note='observations are those in the model\'s tracks' + (' (incomplete tracks: see --track-len)' if a.track_len else ''))
    R['stop'] = dict(stop, iterations=budget.it, seconds=round(time.time() - budget.t0, 2))
    R['thresholds'] = dict(spacing_s=s, unit_mm=um, **{k: getattr(a, k) for k in ('min_track', 'max_reproj', 'knn', 'knn_sigma', 'plane_tol', 'plane_iters', 'plane_side', 'plane_min_points', 'plane_min_share', 'above', 'graph_k', 'link', 'plane_local',
                           'layouts', 'shoulder', 'axis', 'axis_grid', 'layout_tie', 'warn_two_layer', 'warn_empty', 'az_bins', 'gap_cover', 'cover_rows', 'min_gap_deg', 'row_bins', 'edge_q', 'layer_gap', 'layout_cell_points', 'envelope_cells', 'start_net', 'smooth', 'irls', 'tukey_min', 'reparam_steps',
                           'min_support', 'refine_ratio', 'refine_floor', 'max_net', 'max_levels', 'holdout', 'min_gain', 'patience', 'tol', 'fs_tol', 'fs_edge_q', 'fs_min_cos', 'freespace', 'fs_push', 'bridge_dist', 'unsupported_below', 'trim', 'trim_close', 'trim_min_piece', 'max_seconds', 'max_iters', 'seed')},
                         note='lengths given as multiples of s are in model units = multiple x s')
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'charts.json'), 'w') as fh:
        json.dump(dict(what='cubic B-spline charts fitted to SfM points by tools/recon/sfm_spline.py: a geometric fit, not trained', units='model units' + (' (%g mm each)' % um if um else ''),
                       net_layout='net[j][i] = control point of row j (v) and column i (u)', charts=out_charts), fh, separators=(',', ':'))
    with open(os.path.join(a.out, 'report.json'), 'w') as fh: json.dump(R, fh, indent=1, default=_js)
    log('6. wrote %s and %s; stop: %s (%d solves, %.1f s)' % (os.path.join(a.out, 'charts.json'), os.path.join(a.out, 'report.json'), R['stop']['reason'], budget.it, R['stop']['seconds']))
    return R, charts


def _js(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    return str(o)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('model', nargs='?', help='COLMAP text model directory'); ap.add_argument('-o', '--out', help='output directory')
    ap.add_argument('--labels', help='labels.npy: one integer per point3D id, -1 = ignore'); ap.add_argument('--track-len', help='npy of true track lengths per point3D id, for models whose tracks are incomplete')
    ap.add_argument('--unit-mm', type=float, help='millimetres per model unit, when the scale is known'); ap.add_argument('--self-test', action='store_true')
    g = ap.add_argument_group('1. clean')
    g.add_argument('--min-track', type=int, default=2, help='minimum number of distinct images in a track')
    g.add_argument('--max-reproj', type=float, default=2.0, help='maximum mean reprojection error, px')
    g.add_argument('--knn', type=int, default=12, help='neighbours for the statistical outlier test')
    g.add_argument('--knn-sigma', type=float, default=3.5, help='drop points whose log mean neighbour distance exceeds its median + this x MAD (robust z-score)')
    g = ap.add_argument_group('2. plane (lengths in multiples of the spacing s)')
    g.add_argument('--plane-tol', type=float, default=1.5); g.add_argument('--plane-iters', type=int, default=3000)
    g.add_argument('--plane-local', type=int, default=60, help='RANSAC draws its three points among a random point\'s this many nearest neighbours')
    g.add_argument('--plane-side', type=float, default=0.95, help='share of the non-plane points that must lie on the cameras\' side')
    g.add_argument('--plane-min-points', type=int, default=100); g.add_argument('--plane-min-share', type=float, default=0.02, help='a plane needs this share of the points (or --plane-min-points, the larger)')
    g.add_argument('--plane-chart', choices=['auto', 'always'], default='auto', help='auto: a plane chart only without labels')
    g = ap.add_argument_group('3. objects and layout')
    g.add_argument('--above', type=float, default=3.0, help='object points lie this many s above the plane')
    g.add_argument('--graph-k', type=int, default=10, help='kNN graph: each point links to this many nearest neighbours ...')
    g.add_argument('--link', type=float, default=3.0, help='... when closer than this x the median distance to the k-th neighbour')
    g.add_argument('--layouts', default='cyl,sph,hyb,height', help='candidate layouts: cyl (height rows), sph (elevation rows), hyb (height below the shoulder, elevation above), height (height field)')
    g.add_argument('--shoulder', type=float, default=0.75, help='hyb: the shoulder at this share of the object\'s height (1st to 99th percentile)')
    g.add_argument('--axis', choices=['search', 'centroid'], default='search', help='sheet axis: through the robust centroid, or where the layout score is lowest')
    g.add_argument('--axis-grid', type=int, default=7, help='axis search: grid points per side')
    g.add_argument('--az-bins', type=int, default=72); g.add_argument('--gap-cover', type=float, default=0.3, help='an azimuth bin is empty when its points cover less than this share of the row bins')
    g.add_argument('--cover-rows', type=int, default=30, help='row bins (foot to top) for that coverage')
    g.add_argument('--min-gap-deg', type=float, default=20.0, help='a narrower widest gap means closed all round')
    g.add_argument('--row-bins', type=int, default=36, help='azimuth bins for the foot and top of the rows'); g.add_argument('--edge-q', type=float, default=1.0, help='percentile for foot and top per azimuth')
    g.add_argument('--layer-gap', type=float, default=6.0, help='two layers in a cell = a gap in depth of more than this many s')
    g.add_argument('--layout-cell-points', type=float, default=25.0, help='layout cells hold this many points on average')
    g.add_argument('--layout-tie', type=float, default=0.03, help='layout scores this close to the lowest are a tie (resolved: height, cyl, sph)')
    g.add_argument('--warn-two-layer', type=float, default=0.10, help='warn (and record) when the chosen layout has more two-layer points than this share')
    g.add_argument('--warn-empty', type=float, default=0.15, help='warn (and record) when the chosen layout has more empty cells than this share')
    g = ap.add_argument_group('4. fit')
    g.add_argument('--envelope-cells', type=int, default=16, help='direction cells per side of the starting envelope (outermost point per cell, smoothed)')
    g.add_argument('--start-net', type=int, default=6); g.add_argument('--smooth', type=float, default=1e-3, help='weight of the second differences (relative to the data term\'s trace)')
    g.add_argument('--irls', type=int, default=5, help='robust reweighting solves per level'); g.add_argument('--tukey-min', type=float, default=2.0, help='Tukey cut-off is at least this many s')
    g.add_argument('--reparam-steps', type=int, default=1, help='Gauss-Newton parameter corrections per solve')
    g.add_argument('--min-support', type=int, default=8, help='inlier points a knot cell needs for its span to be split')
    g.add_argument('--refine-ratio', type=float, default=1.0, help='split only spans whose median residual is above this x the chart\'s median residual')
    g.add_argument('--refine-floor', type=float, default=0.05, help='... and above this many s')
    g.add_argument('--max-net', type=lambda t: [int(x) for x in t.split('x')], default=[160, 160], help='largest net, nu x nv')
    g.add_argument('--max-levels', type=int, default=20)
    g.add_argument('--holdout', type=float, default=0.1); g.add_argument('--min-gain', type=float, default=0.01, help='a level must lower the held-out median by this share')
    g.add_argument('--patience', type=int, default=2); g.add_argument('--tol', type=float, default=1.0, help='"within tolerance" = within this many s')
    g = ap.add_argument_group('5. free space')
    g.add_argument('--fs-tol', type=float, default=2.0, help='surface may be this many s in front of an observed point'); g.add_argument('--fs-edge-q', type=float, default=95.0, help='z-buffer pixel = this percentile of projected lattice edges')
    g.add_argument('--fs-min-cos', type=float, default=0.1, help='a ray meeting the hiding surface closer to edge-on than this cosine is not judged')
    g.add_argument('--freespace', type=float, default=0.0, help='weight of the free-space penalty rows (0 = diagnostic only)')
    g.add_argument('--fs-push', type=float, default=0.0, help='penalty target: the point\'s depth along the ray plus this x --fs-tol')
    g = ap.add_argument_group('report'); g.add_argument('--bridge-dist', type=float, default=3.0); g.add_argument('--unsupported-below', type=float, default=1.0)
    g.add_argument('--trim', action='store_true', help='write a trim mask per chart: the part of its domain its points support (closed, holes filled); free space is also measured on the trimmed charts')
    g.add_argument('--trim-close', type=int, default=2, help='trim: morphological closing of the supported cells, in cells of the 48 x 48 grid'); g.add_argument('--trim-min-piece', type=float, default=0.01, help='trim: drop kept pieces smaller than this share of the chart\'s area')
    g = ap.add_argument_group('job'); g.add_argument('--max-seconds', type=float, default=900.0); g.add_argument('--max-iters', type=int, default=400, help='least-squares solves, all charts')
    g.add_argument('--seed', type=int, default=0)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)
    if a.self_test:
        import selftest; sys.exit(selftest.main(a))
    if not a.model or not a.out: parser().error('MODEL_DIR and -o OUT are required (or --self-test)')
    run(a)


if __name__ == '__main__':
    main()
