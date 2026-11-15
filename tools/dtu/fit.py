#!/usr/bin/env python3
"""Fit a DTU scan with two cubic B-spline charts and bake it for the talk (talk/dtu-data.js).

    python3 -I tools/dtu/fit.py <data-dir> <scan> [-o talk/dtu-data.js] [--cache DIR]

<data-dir>/scan<N>/ must hold, in the layout of the Hugging Face dataset repo Y1ue/DTU:
    cameras.npz                  IDR convention: world_mat_i = K [R|t] for the 1600x1200 frames, scale_mat_i
    depths/depth_map_%04d.pfm    1600x1200 depth maps in millimetres, one per view
    mask/%03d.png                object silhouettes, 1600x1200
    images/%04d.png              the photographs as undistorted by COLMAP (pinhole, principal point centred)
    sparse/0/cameras.txt, images.txt   the pinhole camera and the poses that go with images/

What it does (README.md next to this file has the reasons):
  1. frame     table plane by RANSAC on the off-object depth -> +y up, millimetres
  2. fuse      depth maps -> one truncated signed distance volume (0.5 mm voxels) -> surface points with normals.
               Only what two or more views measured is kept; nothing is closed or extrapolated.
  3. chart     ONE chart for the object: a rectangle of nodes over everything that was measured. Its four edges
               are the foot on the table, the two azimuths (around an axis through the object) where the
               measurements thin out, and the highest measured direction at each azimuth. The nodes start on a
               coarse envelope of the measurements, are pulled onto the measured surface, and their rows and
               columns are spread evenly by arc length. No pole. Small holes inside are inpainted and flagged.
  4. spline    control net of the clamped cubic tensor B-spline that INTERPOLATES the nodes (with a limiter where
               the sheet has to bridge a step)
  5. colour    every node is projected into the photographs that see it front-on and unoccluded (z-buffer of the
               fitted surface + the scan's own depth map); the four best views are blended
  6. table     a second, flat chart on the fitted table plane, coloured the same way, as far as it was seen
  7. measure   fit error against the fused surface points, observed / inpainted shares -> stdout and the .js

Needs numpy, scipy (>= 1.8), scikit-image, Pillow. No GPU. About three minutes on four cores, 3 GB of memory."""
import argparse, base64, json, os, sys, time
import numpy as np
from PIL import Image
from scipy import ndimage, sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from skimage import measure

HEIGHT_UNITS = 3.55       # every object stands about this tall in scene units (the owl: 284 mm at 80 mm per unit)
NODES = 288 * 392         # nodes of the object's chart when a scan sets no net size: what a 1.5 MB file holds
VOX, COARSE = 0.5, 2.0    # fusion voxel, and the coarse voxel that finds where a surface is (mm)
Y0 = 3.0                  # the sheet is laid out from this height up (mm above the table); its foot is lowered at the end
# Per scan: chart name and caption label, and whatever was fixed by hand. Scan 122 was fitted first, with these
# numbers; they are kept so that its file stays exactly as it was. Other scans derive them from their own data.
# by_mask=False: the two scans on the flat brown board were fitted before the silhouette filter existed (the
# height above the board separates object and table there); every later scan uses the filter.
SCANS = {
    122: dict(name='owl', label='Owl', by_mask=False, unit=80.0, nu=288, nv=392, lo=[-92., -4., -128.], hi=[92., 292., 44.], table=(104, 80, 170.0, 110.0, 150.0)),
    55: dict(name='rabbit', label='Rabbit', by_mask=False, top_mm=250.0),          # the photographs crop its ears: the chart ends below them
}
T0 = time.time()
def log(*a): print('[%5.0fs]' % (time.time() - T0), *a, flush=True)

# ------------------------------------------------------------------ input
def read_pfm(fn):
    with open(fn, 'rb') as f:
        f.readline(); w, h = map(int, f.readline().split()); sc = float(f.readline())
        return np.frombuffer(f.read(), '<f4' if sc < 0 else '>f4').reshape(h, w)[::-1].copy()

def decompose(P):         # P = K [R|t]  ->  K, R, t   (RQ decomposition)
    Q, R = np.linalg.qr(np.flipud(P[:3, :3]).T); K = np.flipud(R.T)[:, ::-1]; R = Q.T[::-1]
    S = np.diag(np.sign(np.diag(K))); K = K @ S; R = S @ R; t = np.linalg.solve(K, P[:3, 3])
    if np.linalg.det(R) < 0: R, t = -R, -t
    return K / K[2, 2], R, t

def load(d):
    z = np.load(os.path.join(d, 'cameras.npz'), allow_pickle=False)
    n = sum(1 for k in z.files if k.startswith('world_mat_') and 'inv' not in k)
    cams = [decompose(z['world_mat_%d' % i][:3].astype(float)) for i in range(n)]
    sm = z['scale_mat_0'].astype(float)
    depth = [read_pfm(os.path.join(d, 'depths', 'depth_map_%04d.pfm' % i)) for i in range(n)]
    mask = [np.asarray(Image.open(os.path.join(d, 'mask', '%03d.png' % i)).convert('L')) > 127 for i in range(n)]
    # the photographs' own pinhole model: poses live in the IDR-normalised frame  X_n = (X_world - c) / s
    cam = [l.split() for l in open(os.path.join(d, 'sparse', '0', 'cameras.txt')) if not l.startswith('#')][0]
    assert cam[1] == 'PINHOLE', cam
    pk = np.array([[float(cam[4]), 0, float(cam[6])], [0, float(cam[5]), float(cam[7])], [0, 0, 1]])
    poses = {}
    for l in [l for l in open(os.path.join(d, 'sparse', '0', 'images.txt')) if not l.startswith('#')][0::2]:
        t = l.split(); w, x, y, zq = map(float, t[1:5])
        R = np.array([[1 - 2 * y * y - 2 * zq * zq, 2 * x * y - 2 * zq * w, 2 * x * zq + 2 * y * w],
                      [2 * x * y + 2 * zq * w, 1 - 2 * x * x - 2 * zq * zq, 2 * y * zq - 2 * x * w],
                      [2 * x * zq - 2 * y * w, 2 * y * zq + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
        poses[int(os.path.splitext(t[9])[0])] = (R, np.array(list(map(float, t[5:8]))))
    photo = []
    for i in range(n):    # world (mm) -> pixel of photograph i
        R, T = poses[i]; s, c = sm[0, 0], sm[:3, 3]
        photo.append((pk, R / s, T - R @ c / s))
    return dict(n=n, cams=cams, depth=depth, mask=mask, photo=photo, dir=d)

# ------------------------------------------------------------------ 1. frame
def frame(S):
    """Table plane -> upright millimetre frame. L = (X - base) @ A, columns of A = right, up, front."""
    obj, tab = [], []
    for i in range(S['n']):
        K, R, t = S['cams'][i]; s = 4; v, u = np.mgrid[0:1200:s, 0:1600:s]; dd = S['depth'][i][::s, ::s]; ok = dd > 0
        X = (np.c_[(u[ok] - K[0, 2]) / K[0, 0] * dd[ok], (v[ok] - K[1, 2]) / K[1, 1] * dd[ok], dd[ok]] - t) @ R
        m = S['mask'][i][::s, ::s][ok]; obj.append(X[m]); tab.append(X[~m])
    obj, tab = np.vstack(obj), np.vstack(tab); C = np.array([-R.T @ t for K, R, t in S['cams']])
    rng = np.random.default_rng(0); best = None
    for it in range(300):
        p = tab[rng.choice(len(tab), 3, replace=False)]; n = np.cross(p[1] - p[0], p[2] - p[0]); l = np.linalg.norm(n)
        if l < 1e-6: continue
        c = (np.abs((tab - p[0]) @ (n / l)) < 0.5).sum()
        if best is None or c > best[0]: best = (c, n / l, p[0])
    inl = np.abs((tab - best[2]) @ best[1]) < 1.0; cen = tab[inl].mean(0); n = np.linalg.svd(tab[inl] - cen, full_matrices=False)[2][2]
    if (C.mean(0) - cen) @ n < 0: n = -n
    rms = np.sqrt((((tab[inl] - cen) @ n) ** 2).mean())
    above = obj[(obj - cen) @ n > 3]; base = above.mean(0); base = base - ((base - cen) @ n) * n
    f = C.mean(0) - base; f -= (f @ n) * n; f /= np.linalg.norm(f); r = np.cross(n, f)
    log('table plane: %.1f %% of the off-object depth, rms %.2f mm; object height %.0f mm' % (100 * inl.mean(), rms, ((obj - cen) @ n).max()))
    A = np.c_[r, n, f]; L = (above - base) @ A                     # the volume to fuse in: the object's own depth, with a margin
    lo = np.floor((np.percentile(L, 0.2, axis=0) - 14) / 4) * 4; hi = np.ceil((np.percentile(L, 99.8, axis=0) + 14) / 4) * 4; lo[1] = -4.0
    return dict(A=A, base=base, table_rms=rms, lo=lo, hi=hi)

# ------------------------------------------------------------------ 2. fuse

def clean_depth(S):
    """The depth maps were rendered from a surface with holes, so some rays pass through a hole and land on the
    INSIDE of the surface. Such a hit claims free space inside the object. Test: step 3 mm back along the ray;
    if the other views see that point just behind their own surface more often than in front of it, drop the pixel."""
    s = 4; bad = []
    for i in range(S['n']):
        K, R, t = S['cams'][i]; v, u = np.mgrid[0:1200:s, 0:1600:s]; dd = S['depth'][i][::s, ::s]; ok = dd > 0
        ray = np.c_[(u[ok] - K[0, 2]) / K[0, 0], (v[ok] - K[1, 2]) / K[1, 1], np.ones(ok.sum())]
        m = ((ray * dd[ok][:, None] - t) - 3.0 * ray / np.linalg.norm(ray, axis=1, keepdims=True)) @ R; front = np.zeros(len(m), int); behind = np.zeros(len(m), int)
        for j in range(S['n']):
            if j == i: continue
            Kj, Rj, tj = S['cams'][j]; h = m @ (Kj @ Rj).T + Kj @ tj; zc = h[:, 2]; uj = (h[:, 0] / zc).round().astype(int); vj = (h[:, 1] / zc).round().astype(int)
            inb = np.nonzero((uj >= 0) & (uj < 1600) & (vj >= 0) & (vj < 1200))[0]; d = S['depth'][j][vj[inb], uj[inb]]; sd = d - zc[inb]
            front[inb] += (d > 0) & (sd > 0.7); behind[inb] += (d > 0) & (sd < -0.7) & (sd > -12)
        b = np.zeros(ok.shape, bool); b[ok] = behind > front; bad.append(np.kron(ndimage.binary_dilation(b), np.ones((s, s), bool)))
    tot = sum((S['depth'][i] > 0).sum() for i in range(S['n'])); cut = 0
    for i in range(S['n']): cut += (bad[i] & (S['depth'][i] > 0)).sum(); S['depth'][i][bad[i]] = 0
    log('depth maps: %.2f %% of the valid pixels are see-through hits, dropped' % (100 * cut / tot))

def fuse_points(S, F, P, tau):
    acc = np.zeros(len(P), np.float32); w = np.zeros(len(P), np.float32); near = np.zeros(len(P), np.float32)
    for i in range(S['n']):
        K, R, t = S['cams'][i]; M = (K @ R @ F['A']).astype(np.float32); m0 = (K @ (R @ F['base'] + t)).astype(np.float32); Di = S['depth'][i]
        for a in range(0, len(P), 4000000):
            h = P[a:a + 4000000] @ M.T + m0; zc = h[:, 2]; u = h[:, 0] / zc; v = h[:, 1] / zc
            idx = np.nonzero((u >= 0) & (u < 1598.99) & (v >= 0) & (v < 1198.99))[0]
            u = u[idx]; v = v[idx]; zc = zc[idx]; u0 = u.astype(np.int32); v0 = v.astype(np.int32); fu = u - u0; fv = v - v0
            d00 = Di[v0, u0]; d01 = Di[v0, u0 + 1]; d10 = Di[v0 + 1, u0]; d11 = Di[v0 + 1, u0 + 1]
            mn = np.minimum(np.minimum(d00, d01), np.minimum(d10, d11)); mx = np.maximum(np.maximum(d00, d01), np.maximum(d10, d11))
            good = (mn > 0) & (mx - mn < 1.5)                       # all four depths valid and on one surface
            sdf = (d00 * (1 - fu) + d01 * fu) * (1 - fv) + (d10 * (1 - fu) + d11 * fu) * fv - zc
            use = good & (sdf > -tau); free = (~good) & (mn > 0) & (mn - zc > tau)      # a depth edge may only carve
            acc[a + idx[use]] += np.clip(sdf[use] / tau, -1, 1); w[a + idx[use]] += 1; near[a + idx[use & (sdf < tau)]] += 1
            acc[a + idx[free]] += 1; w[a + idx[free]] += 1
    return acc, w, near

def fuse(S, F, LO, HI, by_mask=True):
    """All depth maps -> one truncated signed distance volume -> its zero surface as points with outward normals.
    Only what at least two views measured is kept: nothing is closed, filled or extrapolated here."""
    n = np.ceil((HI - LO) / COARSE).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing='ij'), -1).reshape(-1, 3); Pc = (LO + (g + 0.5) * COARSE).astype(np.float32)
    band = ndimage.binary_dilation((fuse_points(S, F, Pc, 3 * COARSE)[2] > 0).reshape(n), iterations=1)        # coarse cells near a measured surface
    k = int(round(COARSE / VOX)); N = n * k; idx = np.nonzero(np.repeat(np.repeat(np.repeat(band, k, 0), k, 1), k, 2).ravel())[0]
    P = (LO + (np.stack(np.unravel_index(idx, N), -1) + 0.5) * VOX).astype(np.float32)
    log('fusing %d views into %d voxels of %.1f mm' % (S['n'], len(P), VOX))
    acc, w, near = fuse_points(S, F, P, 4 * VOX)
    phi = np.ones(N.prod(), np.float32); ok = np.zeros(N.prod(), bool); cnt = np.zeros(N.prod(), np.uint8)
    phi[idx] = np.where(w >= 1, acc / np.maximum(w, 1), 1.0); ok[idx] = w >= 1; cnt[idx] = np.minimum(w, 255)
    V, _, Nn, _ = measure.marching_cubes(phi.reshape(N), 0.0, mask=ok.reshape(N))
    vi = np.clip(np.round(V).astype(int), 0, N - 1); m = ndimage.minimum_filter(cnt.reshape(N), 3)[vi[:, 0], vi[:, 1], vi[:, 2]] >= 2      # 2+ views everywhere around the point
    V = (LO + (V[m] + 0.5) * VOX).astype(np.float32); Nn = -Nn[m].astype(np.float32)
    if by_mask:                   # the object only: a point of it lies inside the object's silhouette in EVERY view
        W = V.astype(float) @ F['A'].T + F['base']; out = np.zeros(len(V), int)
        for i in range(S['n']):
            K, R, t = S['cams'][i]; h = W @ (K @ R).T + K @ t; u = (h[:, 0] / h[:, 2]).round().astype(int); v = (h[:, 1] / h[:, 2]).round().astype(int)
            inb = (u >= 0) & (u < 1600) & (v >= 0) & (v < 1200); out[inb] += ~ndimage.binary_dilation(S['mask'][i], iterations=4)[v[inb], u[inb]]
        k = out <= max(1, S['n'] // 25); log('object by its silhouettes: %.1f %% of the fused points' % (100 * k.mean())); V, Nn = V[k], Nn[k]
    return dict(V=V, N=Nn)

# ------------------------------------------------------------------ 3. the object's chart
class Sheet:
    """The object's chart: a rectangle of nodes laid over everything that was measured.
    Directions are counted from an axis through the object: azimuth around it, and a `row` that is height below
    the shoulder (y1) and elevation above it, seen from the axis point at shoulder height. In these coordinates
    the chart is the rectangle [thL, thR] x [foot, top(azimuth)]: its sides are the azimuths where the
    measurements thin out, its top edge is the highest measured direction at each azimuth. No pole, and nothing
    beyond the measurements."""
    DT, DY = np.radians(1.5), 1.5         # one row / column of the envelope: 1.5 degrees, 1.5 mm

    def __init__(self, D, nu, nv, edge, pre=None):
        m = D['V'][:, 1] > 0.4; self.T = D['V'][m].astype(float); self.TN = D['N'][m].astype(float)
        self.nu, self.nv, self.edge, self.pre = nu, nv, edge, pre or {}
        self.axis_and_wedge(); self.envelope()
        k = self.keep(self.T, 0.5, 0.4); self.Tall, self.TNall = self.T[k], self.TN[k]
        k = self.Tall[:, 1] >= Y0 - 0.5; self.T, self.TN = self.Tall[k], self.TNall[k]; self.ttree = cKDTree(self.T)

    def axis_and_wedge(self):
        """The axis runs from the middle of the foot to a point towards the back of the head at shoulder height and
        is vertical above it. The chart's sides: walk out from the front until less than `edge` of the heights hold
        measurements."""
        T = self.T[self.T[:, 1] > 1.5]; h = T[:, 1].max(); self.y1 = 0.78 * h
        lo = T[T[:, 1] < min(40.0, 0.25 * h)]; hi = T[(T[:, 1] > 0.85 * h) & (T[:, 1] < 0.95 * h)]
        self.a0 = (lo[:, [0, 2]].min(0) + lo[:, [0, 2]].max(0)) / 2
        self.a1 = np.array([(hi[:, 0].min() + hi[:, 0].max()) / 2, hi[:, 2].min() + 0.3 * (hi[:, 2].max() - hi[:, 2].min())])
        th = np.degrees(self.azimuth(T))
        cov = np.zeros((int(h / 2) + 1, 180), bool); cov[(T[:, 1] / 2).astype(int), np.clip(((th + 180) / 2).astype(int), 0, 179)] = True
        share = ndimage.uniform_filter1d(cov.mean(0), 3, mode='wrap'); k = 90
        if share[90] <= self.edge: raise SystemExit('scan not fitted: only %.0f %% of the heights were measured straight in front (need more than %.0f %%); one sheet cannot stand for it' % (100 * share[90], 100 * self.edge))
        while k > 2 and share[k - 1] > self.edge: k -= 1
        self.thL = np.radians(k * 2 - 180 + 4); k = 90
        while k < 177 and share[k + 1] > self.edge: k += 1
        self.thR = np.radians(k * 2 - 180 - 2)
        log('axis from x %.0f z %.0f mm on the table to x %.0f z %.0f at %.0f mm; measured from %.0f to %.0f degrees of azimuth'
            % (self.a0[0], self.a0[1], self.a1[0], self.a1[1], self.y1, np.degrees(self.thL), np.degrees(self.thR)))

    def axis(self, y): t = np.clip(np.asarray(y, float) / self.y1, 0, 1)[..., None]; return self.a0 + (self.a1 - self.a0) * t
    def azimuth(self, p): a = self.axis(p[..., 1]); return np.arctan2(p[..., 0] - a[..., 0], p[..., 2] - a[..., 1])
    def radius(self, p): a = self.axis(p[..., 1]); return np.hypot(p[..., 0] - a[..., 0], p[..., 2] - a[..., 1])
    def half(self, th, r, y): a = self.axis(y); return np.stack([a[..., 0] + r * np.sin(th), y + 0 * r, a[..., 1] + r * np.cos(th)], -1)
    def row(self, p):         # bin centres sit at whole numbers
        return np.where(p[..., 1] < self.y1, p[..., 1] / self.DY - 0.5, self.ny + np.arctan2(p[..., 1] - self.y1, self.radius(p)) / self.DT - 0.5)
    def top(self, th): return np.interp((th - self.th0) / self.DT - 0.5, np.arange(len(self.rowtop)), self.rowtop)
    def keep(self, p, pad=0.0, ymin=Y0 - 0.5):
        """Is p inside the chart's reach: between the two side azimuths, above ymin, below the top edge?"""
        th = self.azimuth(p); r = self.radius(p) + 1e-6
        return ((th >= self.thL - pad / r) & (th <= self.thR + pad / r) | (r < 1.0)) & (p[..., 1] >= ymin) & (self.row(p) <= self.top(th) + 0.7)

    def envelope(self):
        """A coarse, complete stand-in for the object, used to lay the sheet out and to carry it across holes: the
        OUTERMOST measured point in every direction (azimuth x row), gaps filled from their neighbours, lightly
        smoothed. It is never written out; nodes that end on it are flagged as not observed."""
        T = self.T[self.T[:, 1] > 1.5]; self.th0 = self.thL - np.radians(9)
        nt = int(np.ceil((self.thR - self.thL + np.radians(18)) / self.DT)); self.ny = int(np.ceil(self.y1 / self.DY)); nphi = 60
        T = T[(self.azimuth(T) > self.th0) & (self.azimuth(T) < self.th0 + nt * self.DT)]
        r = self.radius(T); col = np.clip(((self.azimuth(T) - self.th0) / self.DT).astype(int), 0, nt - 1)
        row = np.clip((self.row(T) + 0.5).astype(int), 0, self.ny + nphi - 1); val = np.where(T[:, 1] < self.y1, r, np.hypot(r, T[:, 1] - self.y1))
        tab = np.zeros((self.ny + nphi, nt)); np.maximum.at(tab, (row, col), val); known = tab > 0
        tab = ndimage.gaussian_filter(diffuse(tab[..., None], known)[..., 0], 0.7, mode='nearest')
        for k in range(1, 9): w = (9 - k) / 8; tab[-k] = (1 - w) * tab[-k] + w * tab[-k].mean()      # every azimuth meets in one point straight up
        self.tab = tab
        # the top edge: at each azimuth the highest row up to which the measurements are dense, smoothed along azimuth
        # and set two rows (3 mm) inside, so that the edge does not run along a ragged rim
        dense = known & (ndimage.uniform_filter1d(known.astype(float), 7, axis=0, mode='constant') >= 0.5)
        top = np.where(dense.any(0), known.shape[0] - 1 - np.argmax(dense[::-1], 0), 0).astype(float)
        self.rowtop = np.minimum(ndimage.gaussian_filter1d(ndimage.median_filter(top, 5, mode='nearest'), 3.0, mode='nearest') - 2.0, known.shape[0] - 3.0)
        if self.pre.get('top_mm'):     # a scan whose photographs crop the top of the object: end the chart at one level
            rows, cols = np.meshgrid(np.arange(tab.shape[0], dtype=float), np.arange(nt, dtype=float), indexing='ij')
            Y = self.point(rows, self.th0 + (cols + 0.5) * self.DT)[..., 1]; over = Y > self.pre['top_mm']; k = np.clip(np.argmax(over, 0), 1, None); c = np.arange(nt)
            cut = np.where(over.any(0), k - 1 + (self.pre['top_mm'] - Y[k - 1, c]) / np.maximum(Y[k, c] - Y[k - 1, c], 1e-6), tab.shape[0])
            self.rowtop = np.minimum(self.rowtop, cut)
        # the envelope's surface inside the chart's reach, as points with normals
        rr, cc = np.meshgrid(np.arange(0, tab.shape[0] - 1, 0.5), np.arange(0, nt - 1, 0.5), indexing='ij'); Q = self.point(rr, self.th0 + (cc + 0.5) * self.DT)
        QN = np.cross(np.gradient(Q, axis=1), np.gradient(Q, axis=0)); QN /= np.linalg.norm(QN, axis=-1, keepdims=True) + 1e-12
        Q, QN = Q.reshape(-1, 3), QN.reshape(-1, 3); k = self.keep(Q, 0.5); self.Q, self.QN = Q[k], QN[k]; self.qtree = cKDTree(self.Q)
        # the chart's four edges on the envelope
        r0 = Y0 / self.DY - 0.5; th = np.linspace(self.thL, self.thR, 2400); t = np.linspace(0, 1, 3000)
        self.edges = dict(foot=self.point(np.full(th.shape, r0), th), top=self.point(self.top(th), th),
                          left=self.point(r0 + (self.top(self.thL) - r0) * t, np.full(t.shape, self.thL)), right=self.point(r0 + (self.top(self.thR) - r0) * t, np.full(t.shape, self.thR)))
        el = (self.top(th) + 0.5 - self.ny) * np.degrees(self.DT)
        log('envelope: %.0f %% of its directions hold measured points; the top edge runs at %.0f to %.0f degrees above the shoulder point'
            % (100 * known[:, 6:-6].mean(), el.min(), el.max()))
        if self.pre: log('top edge at %s mm (every 10th percentile of its length)' % ' '.join('%.0f' % v for v in np.percentile(self.edges['top'][:, 1], np.arange(0, 101, 10))))

    def point(self, row, th):
        """The envelope's point at (row, azimuth)."""
        e = ndimage.map_coordinates(self.tab, [np.clip(row, 0, self.tab.shape[0] - 1), (th - self.th0) / self.DT - 0.5], order=1, mode='nearest')
        ph = np.clip((row + 0.5 - self.ny) * self.DT, 0, np.pi / 2); cyl = row + 0.5 < self.ny
        return self.half(th, np.where(cyl, e, e * np.cos(ph)), np.where(cyl, (row + 0.5) * self.DY, self.y1 + e * np.sin(ph)))

    def onto_envelope(self, p): return self.point(self.row(p), self.azimuth(p))

    @staticmethod
    def along(c, t):          # points at fractions t of a polyline's length
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        return np.stack([np.interp(t * s[-1], s, c[:, k]) for k in range(3)], -1)

    def border(self, G):
        nv, nu = G.shape[:2]; e = self.edges
        G[0] = self.along(e['foot'], np.linspace(0, 1, nu)); G[-1] = self.along(e['top'], np.linspace(0, 1, nu))
        G[:, 0] = self.along(e['left'], np.linspace(0, 1, nv)); G[:, -1] = self.along(e['right'], np.linspace(0, 1, nv))

    def start(self, nu, nv):
        """The rectangle [thL, thR] x [foot, top(azimuth)] on the envelope: one point per direction, so no folds."""
        th, t = np.meshgrid(np.linspace(self.thL, self.thR, nu), np.linspace(0, 1, nv)); r0 = Y0 / self.DY - 0.5
        G = self.point(r0 + (self.top(th) - r0) * t, th); self.border(G)
        return G

    @staticmethod
    def normals(G):
        n = np.cross(np.gradient(G, axis=1), np.gradient(G, axis=0)); return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

    def project(self, G, delta, fine, stay=False):
        """Each node to the surface nearest to it: the measured points when one lies within delta and faces the same
        way; else the envelope, or (stay) nowhere. Returns the nodes and which of them sit on measured surface."""
        p = G.reshape(-1, 3); n = self.normals(G).reshape(-1, 3); res = p.copy(); seen = np.zeros(len(p), bool)
        def onto(p, q, qn, slip):
            pr = p - np.einsum('pc,pc->p', p - q, qn)[:, None] * qn; tg = pr - q; tl = np.linalg.norm(tg, axis=1, keepdims=True)
            return q + tg * np.minimum(1, slip / (tl + 1e-9))
        if fine:
            dd, ii = self.ttree.query(p, k=6, distance_upper_bound=delta, workers=-1); ok = np.isfinite(dd); ii = np.where(ok, ii, 0)
            ok &= np.einsum('pkc,pc->pk', self.TN[ii], n) > 0.1; seen = ok.any(1); j = ii[np.arange(len(p)), np.argmax(ok, 1)]
            res[seen] = onto(p, self.T[j], self.TN[j], 0.6)[seen]
        rest = ~seen
        if rest.any() and not stay:
            _, j = self.qtree.query(p[rest], workers=-1); res[rest] = onto(p[rest], self.Q[j], self.QN[j], 1.5)
        return res.reshape(G.shape), seen.reshape(G.shape[:2])

    def relax(self, G, iters, delta, fine):
        B = G.copy(); seen = None
        for it in range(iters):
            G[1:-1, 1:-1] += 0.6 * (0.25 * (G[1:-1, :-2] + G[1:-1, 2:] + G[:-2, 1:-1] + G[2:, 1:-1]) - G[1:-1, 1:-1])
            G, seen = self.project(G, delta, fine)
            G[0] = B[0]; G[-1] = B[-1]; G[:, 0] = B[:, 0]; G[:, -1] = B[:, -1]
        return G, seen

    @staticmethod
    def resize(G, nu, nv):
        jj, ii = np.meshgrid(np.linspace(0, G.shape[0] - 1, nv), np.linspace(0, G.shape[1] - 1, nu), indexing='ij')
        return np.stack([ndimage.map_coordinates(G[..., k], [jj, ii], order=1) for k in range(G.shape[2])], -1)

    @staticmethod
    def uniform(c, n):
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]; t = np.linspace(0, s[-1], n)
        return np.stack([np.interp(t, s, c[:, k]) for k in range(3)], -1)

    def respace(self, G, axis, fine, delta, stay=False, k=4):
        """Spread the nodes of every column (axis 0) or row (axis 1) evenly along its own path over the surface. The
        path is the line through a 4x denser, re-projected copy of the sheet, so it cannot cut corners."""
        nv, nu = G.shape[:2]; N = G.copy()
        D, _ = self.project(self.resize(G, (nu - 1) * k + 1 if axis else nu, nv if axis else (nv - 1) * k + 1), delta, fine, stay)
        if axis == 0:
            for i in range(1, nu - 1): N[:, i] = self.uniform(D[:, i], nv)
        else:
            for j in range(1, nv - 1): N[j] = self.uniform(D[j], nu)
        return N

    def spread(self, G, axis):
        H = G if axis == 0 else G.transpose(1, 0, 2); e = np.linalg.norm(np.diff(H, axis=0), axis=-1); med = np.median(e); H = H.copy(); n = len(H)
        for i in range(1, H.shape[1] - 1):
            done = 0
            for k in np.nonzero(e[:, i] > 2.5 * med)[0]:
                if k < done: continue
                w = int(np.clip(round(e[k, i] / med / 2), 2, 8)); a, b = max(k - w, 0), min(k + 1 + w, n - 1)
                H[a:b + 1, i] = self.uniform(H[a:b + 1, i], b - a + 1); done = b
        return H if axis == 0 else H.transpose(1, 0, 2)

    def equalise(self, G, fine, delta, passes):
        seen = None
        for it in range(passes):
            for axis in (0, 1):
                G = 0.3 * G + 0.7 * self.respace(G, axis, fine, delta); self.border(G); G, seen = self.relax(G, 12, delta, fine)
        return G, seen

    def size(self, nodes):
        """A net for this object: as many nodes as the measurements can feed (one per 0.3 mm2 of surface: the
        fusion is 0.5 mm) but no more than `nodes`, and as many columns per row as the sheet is wide per tall."""
        G = self.start(36, 49); G, _ = self.relax(G, 100, 6.0, False); G, _ = self.equalise(G, False, 6.0, 2)
        lr = np.linalg.norm(np.diff(G, axis=1), axis=-1).sum(1).mean(); lc = np.linalg.norm(np.diff(G, axis=0), axis=-1).sum(0).mean()
        area = np.linalg.norm(np.cross(G[1:, :-1] - G[:-1, :-1], G[:-1, 1:] - G[:-1, :-1]), axis=-1).sum()
        n = min(nodes, 1.15 * area / 0.3); self.nu = int(8 * round(np.sqrt(n * lr / lc) / 8)); self.nv = int(8 * (n / self.nu // 8))
        log('sheet about %.0f mm wide, %.0f mm tall, %.0f cm2 -> net %d x %d' % (lr, lc, area / 100, self.nu, self.nv))

    def fit(self):
        lv = 3; nu, nv = self.nu >> lv, self.nv >> lv
        G = self.start(nu, nv); G, seen = self.relax(G, 100, 6.0, False); G, seen = self.equalise(G, False, 6.0, 4)
        for l in range(lv - 1, -1, -1):
            nu, nv = (self.nu >> l, self.nv >> l) if l else (self.nu, self.nv); G = self.resize(G, nu, nv); self.border(G); fine = l < 2; delta = [2.5, 5.0, 6.0][l]
            G, seen = self.relax(G, [30, 50, 80][l], delta, fine); G, seen = self.equalise(G, fine, delta, [1, 2, 3][l])
            log('sheet %d x %d' % (nu, nv))
        # Where the surface steps (an overhang, a deep groove) the sheet has to bridge a gap nothing was measured in,
        # and so far it does that with ONE long cell. Hand the bridge a few nodes from either side, evenly spread.
        self.T, self.TN = self.Tall, self.TNall; self.ttree = cKDTree(self.T)
        for axis in (0, 1): G = self.spread(G, axis)
        # the foot: the lowest row goes down to the measured surface just above the table, the rows above make room
        low = self.Tall[(self.Tall[:, 1] > 0.6) & (self.Tall[:, 1] < 1.4)]; ok = np.zeros(nu, bool); G[0, :, 1] = 1.0
        if len(low): d, j = cKDTree(low[:, [0, 2]]).query(G[0][:, [0, 2]]); ok = d < 4.0; G[0][ok] = low[j[ok]]
        for i in range(nu): G[:9, i] = self.uniform(G[:9, i], 9)
        B = G.copy(); G, seen = self.project(G, 1.2, True, stay=True); G[0] = B[0]; seen[0] = ok
        # Nodes with no measured surface under them take the envelope's shape (the outermost measured radius around
        # the hole, interpolated across it), shifted so that it meets the measured surface at the hole's rim: the
        # shift is interpolated between the rim's own offsets from the envelope, along the edges first, then as a
        # membrane over the interior.
        E = self.onto_envelope(G); O = G - E
        for e in (np.s_[0], np.s_[-1], np.s_[:, 0], np.s_[:, -1]):
            k = seen[e]; x = np.arange(len(k))
            if k.any() and not k.all(): O[e] = np.stack([np.interp(x, x[k], O[e][k, c]) for c in range(3)], -1)
        hole = ~seen; hole[0] = hole[-1] = False; hole[:, 0] = hole[:, -1] = False
        if hole.any():
            nv_, nu_ = hole.shape; idn = np.arange(nv_ * nu_).reshape(nv_, nu_); jj, ii = np.nonzero(hole); r = idn[jj, ii]; O = O.reshape(-1, 3)
            L = sparse.csr_matrix((np.r_[4 * np.ones(len(r)), -np.ones(4 * len(r))], (np.tile(r, 5), np.r_[r, idn[jj, ii - 1], idn[jj, ii + 1], idn[jj - 1, ii], idn[jj + 1, ii]])), shape=(nv_ * nu_, nv_ * nu_))
            kn = np.nonzero(~hole.ravel())[0]; O[r] = spsolve(L[r][:, r].tocsc(), -(L[r][:, kn] @ O[kn]))
        return (E.reshape(-1, 3) + O.reshape(-1, 3)).reshape(G.shape), seen

# ------------------------------------------------------------------ 4. spline
def knots(n): return np.r_[np.zeros(4), np.arange(1, n - 3) / (n - 3), np.ones(4)]      # = clampedKnots(n, 3) in talk/nurbs.js
def greville(n): U = knots(n); return (U[1:n + 1] + U[2:n + 2] + U[3:n + 3]) / 3
def basis(n, t): return BSpline.design_matrix(np.clip(t, 0, 1), knots(n), 3).toarray()

def interpolate(G):
    """Control net of the clamped cubic tensor B-spline that passes through the nodes G at the Greville abscissae."""
    nv, nu = G.shape[:2]; Bu, Bv = basis(nu, greville(nu)), basis(nv, greville(nv))
    return np.linalg.solve(Bv, np.linalg.solve(Bu, G.transpose(1, 0, 2).reshape(nu, -1)).reshape(nu, nv, -1).transpose(1, 0, 2).reshape(nv, -1)).reshape(nv, nu, -1)

def evaluate(Cn, us, vs):
    nv, nu = Cn.shape[:2]; return np.einsum('bj,jic,ai->bac', basis(nv, vs), Cn, basis(nu, us), optimize=True)

# ------------------------------------------------------------------ 5. colour from the photographs
def colours(S, F, P, Nrm, valid, occluder, down=2, best=4):
    """P (n,3) upright mm, Nrm unit normals. Per point: the views that see it front-on, in front of the fitted
    surface (z-buffer of `occluder`, a dense sampling of it) and in agreement with the dataset's depth map."""
    n = len(P); score = np.zeros((S['n'], n), np.float32); rgb = np.zeros((S['n'], n, 3), np.float32)
    W = P @ F['A'].T + F['base']; Wn = Nrm @ F['A'].T; Wo = occluder @ F['A'].T + F['base']
    for i in range(S['n']):
        K, R, t = S['cams'][i]; c = -R.T @ t; view = c - W; dist = np.linalg.norm(view, axis=1); cosv = np.einsum('pc,pc->p', view, Wn) / dist
        # z-buffer of the fitted surface in the depth map's frame, at a quarter of its resolution
        h = Wo @ (K @ R).T + K @ t; zo = h[:, 2]; uo = (h[:, 0] / zo / 4).astype(int); vo = (h[:, 1] / zo / 4).astype(int); ok = (uo >= 0) & (uo < 400) & (vo >= 0) & (vo < 300)
        zb = np.full((300, 400), np.inf); np.minimum.at(zb, (vo[ok], uo[ok]), zo[ok]); zb = ndimage.minimum_filter(zb, 3)
        h = W @ (K @ R).T + K @ t; z = h[:, 2]; u = h[:, 0] / z; v = h[:, 1] / z; inb = (u >= 1) & (u < 1598) & (v >= 1) & (v < 1198)
        ui = np.clip(u.round().astype(int), 0, 1599); vi = np.clip(v.round().astype(int), 0, 1199)
        vis = inb & (cosv > 0.2) & (z < zb[np.clip(vi // 4, 0, 299), np.clip(ui // 4, 0, 399)] + 2.5)
        dm = S['depth'][i][vi, ui]; vis &= (dm <= 0) | (np.abs(dm - z) < 2.0)
        # the photograph (its own pinhole camera), box-filtered to the sheet's sampling
        im = Image.open(os.path.join(S['dir'], 'images', '%04d.png' % i)).convert('RGB'); w, hh = im.size
        im = np.asarray(im.resize((w // down, hh // down), Image.BOX), np.float32) / 255
        pk, pr, pt = S['photo'][i]; h = W @ (pk @ pr).T + pk @ pt; x = h[:, 0] / h[:, 2] / down - 0.5; y = h[:, 1] / h[:, 2] / down - 0.5
        vis &= (x >= 0) & (x < im.shape[1] - 1.01) & (y >= 0) & (y < im.shape[0] - 1.01)
        k = np.nonzero(vis & valid)[0]; x0 = x[k].astype(int); y0 = y[k].astype(int); fx = (x[k] - x0)[:, None]; fy = (y[k] - y0)[:, None]
        rgb[i, k] = (im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx) * (1 - fy) + (im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx) * fy
        score[i, k] = cosv[k] ** 2
    order = np.argsort(-score, axis=0)[:best]; sc = np.take_along_axis(score, order, 0) ** 3
    col = (np.take_along_axis(rgb, order[..., None], 0) * sc[..., None]).sum(0) / np.maximum(sc.sum(0), 1e-12)[:, None]
    return col, (score > 0).sum(0)

def diffuse(img, known, iters=400):
    """Fill the unknown cells of img (h,w,c) from their neighbours."""
    out = np.where(known[..., None], img, 0.0); w = known.astype(float)
    for it in range(iters):
        if w.min() > 0: break
        s = ndimage.uniform_filter(out * w[..., None], (3, 3, 1), mode='nearest'); ww = ndimage.uniform_filter(w, 3, mode='nearest')
        new = (ww > 0.05) & (w == 0); out[new] = s[new] / ww[new][:, None]; w[new] = 1
    hole = ~known
    for it in range(30): out[hole] = ndimage.uniform_filter(out, (3, 3, 1), mode='nearest')[hole]
    return out

# ------------------------------------------------------------------ 6. the table
def table(S, F, centre, nu, nv, half_x, back, front):
    xs = centre[0] + np.linspace(-half_x, half_x, nu); zs = centre[1] + np.linspace(-back, front, nv); X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], -1); W = P @ F['A'].T + F['base']; up = F['A'][:, 1]
    acc = np.zeros((len(P), 3)); wsum = np.zeros(len(P)); count = np.zeros(len(P), int)
    for i in range(S['n']):
        K, R, t = S['cams'][i]; c = -R.T @ t; view = c - W; cosv = (view @ up) / np.linalg.norm(view, axis=1)
        # hidden behind the object in this view? its silhouette, grown a little, says so
        h = W @ (K @ R).T + K @ t; u = h[:, 0] / h[:, 2]; v = h[:, 1] / h[:, 2]; inb = (u >= 0) & (u < 1599) & (v >= 0) & (v < 1199)
        m = ndimage.binary_dilation(S['mask'][i], iterations=6)[np.clip(v.astype(int), 0, 1199), np.clip(u.astype(int), 0, 1599)]
        im = Image.open(os.path.join(S['dir'], 'images', '%04d.png' % i)).convert('RGB'); w, hh = im.size; im = np.asarray(im.resize((w // 4, hh // 4), Image.BOX), np.float32) / 255
        pk, pr, pt = S['photo'][i]; h = W @ (pk @ pr).T + pk @ pt; x = h[:, 0] / h[:, 2] / 4 - 0.5; y = h[:, 1] / h[:, 2] / 4 - 0.5
        edge = np.minimum(np.minimum(x, im.shape[1] - 1 - x), np.minimum(y, im.shape[0] - 1 - y))
        k = np.nonzero(inb & ~m & (edge > 1.5) & (cosv > 0.15))[0]; x0 = x[k].astype(int); y0 = y[k].astype(int); fx = (x[k] - x0)[:, None]; fy = (y[k] - y0)[:, None]
        px = (im[y0, x0] * (1 - fx) + im[y0, x0 + 1] * fx) * (1 - fy) + (im[y0 + 1, x0] * (1 - fx) + im[y0 + 1, x0 + 1] * fx) * fy
        wt = cosv[k] * np.clip(edge[k] / 40, 0, 1) ** 2                      # fade each photograph out towards its frame
        acc[k] += px * wt[:, None]; wsum[k] += wt; count[k] += 1
    seen = (count >= 2).reshape(nv, nu); col = diffuse((acc / np.maximum(wsum, 1e-9)[:, None]).reshape(nv, nu, 3), seen)
    return dict(xs=xs, zs=zs, C=col, seen=seen)

# ------------------------------------------------------------------ 7. measure + write
def measure_fit(Cn, G, seen, D, sheet):
    nv, nu = Cn.shape[:2]; k = 3; us = np.linspace(0, 1, nu * k); vs = np.linspace(0, 1, nv * k); Sd = evaluate(Cn, us, vs)
    n = np.cross(np.gradient(Sd, axis=1), np.gradient(Sd, axis=0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    tree = cKDTree(Sd.reshape(-1, 3)); T = D['V'][D['V'][:, 1] > 1.0].astype(float); size = np.linalg.norm(T.max(0) - T.min(0))
    d, j = tree.query(T, workers=-1); q = Sd.reshape(-1, 3)[j]; pl = np.abs(np.einsum('pc,pc->p', T - q, n.reshape(-1, 3)[j]))
    step = np.median(np.linalg.norm(np.diff(Sd, axis=1), axis=-1)); dist = np.where(d < 3 * step, np.minimum(d, pl), d)
    inw = sheet.keep(T, 0.0, 1.0); e = dist[inw]
    node = np.linalg.norm(evaluate(Cn, greville(nu), greville(nv)) - G, axis=-1)
    return dict(size_mm=size, rms_mm=float(np.sqrt((e ** 2).mean())), p95_mm=float(np.percentile(e, 95)), median_mm=float(np.median(e)),
                within_1mm=float((e < 1).mean()), points=int(inw.sum()), points_all=int(len(T)), covered_all=float((dist < 1).mean()),
                node_residual_mm=[float(np.median(node)), float(np.percentile(node, 99)), float(node.max())], dense=Sd, dense_n=n, spacing_mm=[float(np.median(np.linalg.norm(np.diff(G, axis=1), axis=-1))), float(np.median(np.linalg.norm(np.diff(G, axis=0), axis=-1)))])

def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')

def views_for(lo, hi, target, cams):
    """Four cameras in the shape talk/scenes.js uses ({ th, ph, r, tx, ty, tz }), all inside the dome of the scan's
    own cameras: azimuths within 85 % of theirs, elevations between their lowest and highest ring."""
    c = np.asarray(cams) - target; az = np.arctan2(c[:, 0], c[:, 2]); el = np.arcsin(c[:, 1] / np.linalg.norm(c, axis=1)); rho = np.linalg.norm(hi - lo) / 2
    ph = lambda deg: np.pi / 2 - np.clip(np.radians(deg), el.min() + 0.03, el.max() - 0.1); th = lambda x: np.clip(x, 0.85 * az.min(), 0.85 * az.max()); tx, ty, tz = target
    v = dict(wide=dict(th=th(0.30), ph=ph(20), r=4.15 * rho, tx=tx, ty=ty, tz=tz), close=dict(th=th(-0.15), ph=ph(21), r=max(1.6 * rho, 1.6), tx=tx, ty=(ty + hi[1]) / 2, tz=tz),
             a=dict(th=th(-0.85), ph=ph(26), r=3.3 * rho, tx=tx, ty=ty, tz=tz), b=dict(th=th(0.80), ph=ph(33), r=3.3 * rho, tx=tx, ty=ty, tz=tz))
    return {k: {q: round(float(x), 3) for q, x in d.items()} for k, d in v.items()}

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('scan', type=int)
    ap.add_argument('-o', '--out', help='default: talk/dtu-data.js, or talk/dtu/scan<N>.js with --register scans')
    ap.add_argument('--register', choices=('dtu', 'scans'), default='dtu', help='dtu: the file sets KnotTalk.dtu, the talk\'s one subject. scans: it adds itself to KnotTalk.dtuScans[N] and carries its own views')
    ap.add_argument('--cache', help='directory for intermediate results (fusion takes most of the time)')
    ap.add_argument('--nu', type=int, help='columns and'); ap.add_argument('--nv', type=int, help='rows of the object\'s net (default: the scan\'s preset, else from its size)')
    ap.add_argument('--edge', type=float, help='the chart ends at the azimuth where less than this share of the heights was measured (0.6)')
    ap.add_argument('--name', help='name of the object\'s chart'); ap.add_argument('--label', help='what the object is, for captions')
    a = ap.parse_args(); d = os.path.join(a.data, 'scan%d' % a.scan); pre = SCANS.get(a.scan, {})
    a.name = a.name or pre.get('name', 'object'); a.label = a.label or pre.get('label', a.name.capitalize())
    talk = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk')
    a.out = a.out or (os.path.join(talk, 'dtu-data.js') if a.register == 'dtu' else os.path.join(talk, 'dtu', 'scan%d.js' % a.scan))
    def cached(name, fn):
        p = a.cache and os.path.join(a.cache, name + '.npz')
        if p and os.path.exists(p): z = np.load(p); return {k: z[k] for k in z.files}
        r = fn()
        if p: os.makedirs(a.cache, exist_ok=True); np.savez(p, **r)
        return r
    S = load(d); log('scan %d: %d views' % (a.scan, S['n']))
    F = cached('frame', lambda: frame(S)); clean_depth(S)
    lo, hi = (np.array(pre['lo']), np.array(pre['hi'])) if 'lo' in pre else (F['lo'], F['hi'])
    D = cached('fused', lambda: fuse(S, F, lo, hi, pre.get('by_mask', True)))
    log('fused surface: %d measured points' % len(D['V']))
    sheet = Sheet(D, a.nu or pre.get('nu'), a.nv or pre.get('nv'), a.edge or pre.get('edge', 0.6), pre)
    if not (sheet.nu and sheet.nv): sheet.size(pre.get('nodes', NODES))
    nu, nv = sheet.nu, sheet.nv; height = float(sheet.Tall[:, 1].max()); UNIT_MM = pre.get('unit') or float(round(height / HEIGHT_UNITS))
    R = cached('sheet_%d_%d' % (nu, nv), lambda: dict(zip(('G', 'seen'), sheet.fit()))); G, seen = R['G'], R['seen'].astype(bool)
    # the control net of the interpolating spline; where the sheet bridges a step, exact interpolation would ring,
    # so no control point may sit further than LIMIT from its node (there the spline only approximates the node)
    Cn = interpolate(G); dev = np.linalg.norm(Cn - G, axis=-1, keepdims=True); LIMIT = 0.6
    limited = float((dev > LIMIT).mean()); Cn = G + (Cn - G) * np.minimum(1, LIMIT / np.maximum(dev, 1e-9)); M = measure_fit(Cn, G, seen, D, sheet)
    # appearance: one colour per node from the photographs, then control colours whose spline passes through them
    nrm = Sheet.normals(G); col, nviews = colours(S, F, G.reshape(-1, 3), nrm.reshape(-1, 3), np.ones(seen.size, bool), M['dense'].reshape(-1, 3))
    lit = (nviews > 0).reshape(seen.shape); col = diffuse(col.reshape(G.shape), lit)
    Cc = np.clip(col + np.clip(interpolate(col) - col, -0.12, 0.12), 0, 1)
    if 'table' in pre: tn = pre['table']
    else:                         # around the object's footprint, wider in front; nodes about 3.3 mm apart, at most 9,000
        c = sheet.axis(0.0); T = sheet.Tall; hx = np.percentile(np.abs(T[:, 0] - c[0]), 99.5) + 0.33 * height
        back = c[1] - np.percentile(T[:, 2], 0.5) + 0.2 * height; front = np.percentile(T[:, 2], 99.5) - c[1] + 0.35 * height
        k = min(1.0, np.sqrt(9000 / (2 * hx / 3.27 * (back + front) / 3.27))); tn = (int(2 * hx / 3.27 * k), int((back + front) / 3.27 * k), float(hx), float(back), float(front))
    tb = table(S, F, sheet.axis(0.0), *tn)
    # ---- scene coordinates: origin on the table under the axis, +y up, +z towards the cameras
    o = np.array([sheet.axis(0.0)[0], 0.0, sheet.axis(0.0)[1]]); to_scene = lambda p: (p - o) / UNIT_MM
    P = to_scene(Cn); lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    q = np.round((P - lo) / (hi - lo) * 65535).astype('<u2')
    cams = to_scene((np.array([-Rm.T @ t for K, Rm, t in S['cams']]) - F['base']) @ F['A'])
    area = lambda g: np.linalg.norm(np.cross(g[1:, :-1] - g[:-1, :-1], g[:-1, 1:] - g[:-1, :-1]), axis=-1)
    ar = area(G); cs = seen[:-1, :-1] & seen[1:, :-1] & seen[:-1, 1:] & seen[1:, 1:]
    stats = dict(scan=a.scan, views=S['n'], unit_mm=UNIT_MM, charts=2, control_points=int(nu * nv + tb['C'].shape[0] * tb['C'].shape[1]),
                 object=dict(nu=nu, nv=nv, observed_nodes=float(seen.mean()), observed_area=float(ar[cs].sum() / ar.sum()), coloured_nodes=float(lit.mean()), coloured_observed=float(lit[seen].mean()), coloured_inpainted=float(lit[~seen].mean()), limited_control_points=limited,
                             area_mm2=float(ar.sum()), **{k: v for k, v in M.items() if k not in ('dense', 'dense_n')}),
                 table=dict(nu=tb['C'].shape[1], nv=tb['C'].shape[0], observed_nodes=float(tb['seen'].mean()), plane_rms_mm=float(F['table_rms'])))
    log('charts 2, control points %d (object %d x %d, table %d x %d)' % (stats['control_points'], nu, nv, tb['C'].shape[1], tb['C'].shape[0]))
    log('object: node spacing %.2f x %.2f mm; %.2f %% of the control points held back by the limiter; spline minus node: median %.1e, 99th pct %.3f, max %.2f mm'
        % (M['spacing_mm'][0], M['spacing_mm'][1], 100 * limited, *M['node_residual_mm']))
    log('fit error, %d measured points inside the chart: rms %.3f mm, 95th pct %.3f mm, median %.3f mm = %.5f / %.5f of the object (%.0f mm diagonal)'
        % (M['points'], M['rms_mm'], M['p95_mm'], M['median_mm'], M['rms_mm'] / M['size_mm'], M['p95_mm'] / M['size_mm'], M['size_mm']))
    log('measured points within 1 mm of the chart: %.1f %% inside its border, %.1f %% of everything measured' % (100 * M['within_1mm'], 100 * M['covered_all']))
    log('object chart: %.1f %% of the nodes (%.1f %% of the area) observed, the rest inpainted; coloured from photographs: %.1f %% of the observed nodes, %.1f %% of the inpainted ones (the rest: from their neighbours)'
        % (100 * seen.mean(), 100 * stats['object']['observed_area'], 100 * lit[seen].mean(), 100 * lit[~seen].mean()))
    log('table chart: %.1f %% of the nodes seen by two or more photographs, the rest (under and behind the object) filled from their neighbours' % (100 * tb['seen'].mean()))
    data = dict(scan=a.scan, label=a.label, source='DTU MVS dataset (Jensen et al., CVPR 2014), scan %d, from the Hugging Face dataset repo Y1ue/DTU' % a.scan,
                note='fitted offline to the scan\'s depth maps and photographs by tools/dtu/fit.py; not trained', unit_mm=UNIT_MM, stats=stats,
                cams=np.round(cams, 3).tolist(), target=[0, round(float((hi[1] + lo[1]) / 2), 3), round(float((hi[2] + lo[2]) / 2), 3)],
                charts=[dict(name=a.name, nu=nu, nv=nv, lo=np.round(lo, 5).tolist(), hi=np.round(hi, 5).tolist(), P=b64(q), C=b64(np.round(Cc * 255).astype(np.uint8)),
                             seen=b64(np.packbits(seen.ravel()))),
                        dict(name='table', nu=tb['C'].shape[1], nv=tb['C'].shape[0], plane=[round(float(v), 4) for v in ((tb['xs'][0] - o[0]) / UNIT_MM, (tb['xs'][-1] - o[0]) / UNIT_MM, (tb['zs'][0] - o[2]) / UNIT_MM, (tb['zs'][-1] - o[2]) / UNIT_MM)],
                             C=b64(np.round(np.clip(tb['C'], 0, 1) * 255).astype(np.uint8)), seen=b64(np.packbits(tb['seen'].ravel())))])
    if a.register == 'scans':     # one of several: it carries its own cameras for the scene, and registers under its number
        charts = data.pop('charts'); data['views'] = dict(views_for(lo, hi, np.array(data['target'], float), cams), **pre.get('views', {})); data['tour'] = pre.get('tour', ['wide', 'close', 'a', 'b']); data['charts'] = charts
        js = JS_HEAD_SCANS % dict(scan=a.scan) + json.dumps(data, separators=(',', ':')) + ';\n' + JS_TAIL.replace('KT.dtu.charts', 'D.charts')
    else: js = JS_HEAD % dict(scan=a.scan) + 'KT.dtu = ' + json.dumps(data, separators=(',', ':')) + ';\n' + JS_TAIL
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f: f.write(js)
    log('wrote %s, %.2f MB' % (os.path.normpath(a.out), len(js) / 1e6))
    if a.cache: np.savez(os.path.join(a.cache, 'result.npz'), G=G, seen=seen, col=col, Cn=Cn, Cc=Cc, tableC=tb['C'], tableSeen=tb['seen'], lit=lit)

JS_HEAD = """/* DTU scan %(scan)d as B-spline charts. GENERATED by tools/dtu/fit.py: do not edit, re-run the script.
   A real object from the DTU multi-view stereo dataset (Jensen et al., CVPR 2014; this copy of the scan comes from
   the Hugging Face dataset repo Y1ue/DTU). Geometry: the scan's depth maps, fused, then interpolated by clamped cubic
   tensor B-splines. Colour: the scan's photographs, projected onto the fitted surface. It was FITTED OFFLINE to that
   data; it was not trained or reconstructed by the thesis method.
   KnotTalk.dtu = { scan, source, note, unit_mm, stats, cams: [[x,y,z]...], target, charts: [{ name, nu, nv,
     P: Float32Array(nu*nv*3) control points (scene units, +y up, table at y = 0), C: Float32Array(nu*nv*3) control
     colours 0..1 (display space), seen: Uint8Array(nu*nv) 1 = measured, 0 = inpainted }] }.  Row j, column i -> j*nu+i. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {};
"""
JS_TAIL = """  function bytes(s) { var b = atob(s), n = b.length, a = new Uint8Array(n), i; for (i = 0; i < n; i++) a[i] = b.charCodeAt(i); return a; }
  KT.dtu.charts.forEach(function (c) {
    var n = c.nu * c.nv, P = new Float32Array(n * 3), C = new Float32Array(n * 3), b = bytes(c.C), s = bytes(c.seen), seen = new Uint8Array(n), i, k;
    if (c.plane) { for (i = 0; i < n; i++) { P[i * 3] = c.plane[0] + (c.plane[1] - c.plane[0]) * (i % c.nu) / (c.nu - 1); P[i * 3 + 2] = c.plane[2] + (c.plane[3] - c.plane[2]) * Math.floor(i / c.nu) / (c.nv - 1); } }
    else { var q = bytes(c.P); for (i = 0; i < n * 3; i++) { k = i % 3; P[i] = c.lo[k] + (c.hi[k] - c.lo[k]) * (q[i * 2] + 256 * q[i * 2 + 1]) / 65535; } }
    for (i = 0; i < n * 3; i++) C[i] = b[i] / 255;
    for (i = 0; i < n; i++) seen[i] = s[i >> 3] >> (7 - (i & 7)) & 1;
    c.P = P; c.C = C; c.seen = seen;
  });
})();
"""

JS_HEAD_SCANS = """/* DTU scan %(scan)d as B-spline charts. GENERATED by tools/dtu/fit.py --register scans: do not edit, re-run the script.
   A real object from the DTU multi-view stereo dataset (Jensen et al., CVPR 2014; this copy of the scan comes from
   the Hugging Face dataset repo Y1ue/DTU). Geometry: the scan's depth maps, fused, then interpolated by clamped cubic
   tensor B-splines. Colour: the scan's photographs, projected onto the fitted surface. It was FITTED OFFLINE to that
   data; it was not trained or reconstructed by the thesis method.
   KnotTalk.dtuScans[%(scan)d] = the same object as KnotTalk.dtu in talk/dtu-data.js: { scan, label, source, note, unit_mm,
     stats, cams: [[x,y,z]...], target, charts: [{ name, nu, nv, P, C, seen }] } (P: Float32Array(nu*nv*3) control
     points in scene units, +y up, table at y = 0; C: control colours 0..1 in display space; seen: Uint8Array(nu*nv),
     1 = measured, 0 = inpainted; row j, column i -> j*nu+i), plus views: { wide, close, a, b } as { th, ph, r, tx, ty, tz }
     and tour: [...], in the shape talk/scenes.js uses; every view lies inside the dome of the scan's real cameras. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {}, D = (KT.dtuScans = KT.dtuScans || {})[%(scan)d] = """

if __name__ == '__main__':
    main()
