#!/usr/bin/env python3
"""Data for the talk's pipeline chapter: how the owl (DTU scan 122) gets from photographs to a spline chart.

    python3 -I tools/dtu/pipeline.py <data-dir> 122 [-o talk/dtu/pipeline122.js] [--cache DIR] [--check DIR]

<data-dir>/scan122/ as for fit.py (cameras.npz, depths/, mask/, sparse/0/cameras.txt + images.txt) plus
sparse/0/points3D.txt, the dataset's own COLMAP points (all from the Hugging Face dataset repo Y1ue/DTU). The
photographs themselves are not read.

Everything is written in the scene frame and units of the scan's data file (talk/dtu-data.js for 122): the same
transform fit.py applies, re-derived here by running fit.py's own steps, and checked against that file.

  cameras  the 64 cameras: centre, rotation, and the pinhole intrinsics of the photographs
  sfm      the dataset's sparse point cloud (DTU/scan122/sparse/0/points3D.txt): position, colour, track length,
           mean reprojection error, the first and the second camera of each track. The repo's DTU/README.md says
           it was made "with COLMAP given GT camera poses": COLMAP matched and triangulated the points, the cameras
           are the dataset's calibrated ones; nothing estimated them.
  init     a coarse chart (12 x 16 control points) fitted to those sparse points ONLY, in fit.py's parameterisation
           of the owl (its axis, azimuth range and rows): a sketch of initialising a spline chart from SfM, made here
  levels   the coarse-to-fine levels fit.py goes through when it lays the owl's chart onto the fused depth maps:
           the nodes fit.py holds at the start and at the end of each level, each written as the control net of the
           cubic spline that passes through them, with its distance to the measured surface. An offline GEOMETRIC
           fit to the scan's depth maps; nothing in this file is trained, by the thesis method or otherwise.

Needs numpy, scipy, scikit-image and Pillow (through fit.py). About two minutes on four cores."""
import argparse, base64, json, os, re, sys, time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit                                                    # the scan's frame, fusion and chart: the same code, the same numbers

log = fit.log
SAMPLES = (864, 1176)     # every surface is read at this many (u, v) samples to measure distances: what fit.py uses for the owl
# what fit.Sheet.fit() does at each level (relax sweeps; the projection's reach in mm; onto what), in its order
STEPS = [dict(relax=100, equalise=4, delta=6.0, onto='envelope'), dict(relax=80, equalise=3, delta=6.0, onto='envelope'),
         dict(relax=50, equalise=2, delta=5.0, onto='measured'), dict(relax=30, equalise=1, delta=2.5, onto='measured')]


class Recorder(fit.Sheet):
    """fit.Sheet, unchanged, that keeps a copy of the nodes at the start and at the end of every level of fit()."""
    def start(self, nu, nv):
        G = super().start(nu, nv); self.levels = [dict(G=G.copy(), seen=np.zeros(G.shape[:2], bool), fine=False)]
        return G

    def equalise(self, G, fine, delta, passes):
        G, seen = super().equalise(G, fine, delta, passes)
        self.levels.append(dict(G=G.copy(), seen=seen.copy(), fine=fine, delta=delta, passes=passes))
        return G, seen


# ------------------------------------------------------------------ packing: 12 bits per coordinate, two per 3 bytes
def box(P, digits=4):
    P = np.asarray(P).reshape(-1, 3); k = 10 ** digits
    return np.floor(P.min(0) * k) / k, np.ceil(P.max(0) * k) / k

def pack12(P, lo, hi):
    q = np.rint((np.asarray(P).reshape(-1, 3) - lo) / (hi - lo) * 4095).astype(np.uint16).ravel()
    if len(q) % 2: q = np.r_[q, 0]
    a, b = q[0::2], q[1::2]; out = np.empty(len(a) * 3, np.uint8)
    out[0::3] = a & 255; out[1::3] = (a >> 8) | ((b & 15) << 4); out[2::3] = b >> 4
    return base64.b64encode(out.tobytes()).decode('ascii')

def unpack12(s, n, lo, hi):           # the inverse, as the file's own loader does it: n points
    by = np.frombuffer(base64.b64decode(s), np.uint8).astype(np.uint16)
    q = np.empty(len(by) // 3 * 2, np.uint16); q[0::2] = by[0::3] | ((by[1::3] & 15) << 8); q[1::2] = (by[1::3] >> 4) | (by[2::3] << 4)
    return lo + (hi - lo) * q[:n * 3].reshape(-1, 3) / 4095

def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode('ascii')


# ------------------------------------------------------------------ distances to a chart
def distances(Cn, pts, samples=SAMPLES):
    """Distance from each point to the spline surface with control net Cn (nv, nu, 3), as fit.measure_fit measures it:
    nearest of a dense (u, v) sampling, refined to the tangent plane there when the sample is close."""
    Sd = fit.evaluate(Cn, np.linspace(0, 1, samples[0]), np.linspace(0, 1, samples[1]))
    n = np.cross(np.gradient(Sd, axis=1), np.gradient(Sd, axis=0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    d, j = cKDTree(Sd.reshape(-1, 3)).query(pts, workers=-1); q = Sd.reshape(-1, 3)[j]
    pl = np.abs(np.einsum('pc,pc->p', pts - q, n.reshape(-1, 3)[j]))
    step = np.median(np.linalg.norm(np.diff(Sd, axis=1), axis=-1))
    return np.where(d < 3 * step, np.minimum(d, pl), d)

def summary(d):
    return dict(rms_mm=round(float(np.sqrt((d ** 2).mean())), 4), p95_mm=round(float(np.percentile(d, 95)), 4),
                median_mm=round(float(np.median(d)), 4), within_1mm=round(float((d < 1).mean()), 4), points=int(len(d)))

def control_net(G, limit=0.6):
    """The control net fit.py makes from a grid of nodes: the interpolating cubic spline, no control point further
    than `limit` mm from its node."""
    Cn = fit.interpolate(G); dev = np.linalg.norm(Cn - G, axis=-1, keepdims=True)
    return G + (Cn - G) * np.minimum(1, limit / np.maximum(dev, 1e-9)), float((dev > limit).mean())


# ------------------------------------------------------------------ the dataset's COLMAP model
def read_colmap(d):
    names, obs = {}, {}
    L = [l for l in open(os.path.join(d, 'sparse', '0', 'images.txt')) if not l.startswith('#')]
    for a, b in zip(L[0::2], L[1::2]):
        t = a.split(); iid = int(t[0]); names[iid] = int(os.path.splitext(t[9])[0])          # image id -> camera index (0000.png -> 0)
        v = b.split(); obs[iid] = np.array(v, float).reshape(-1, 3) if v else np.zeros((0, 3))
    X, C, E, TL, first, second, track = [], [], [], [], [], [], []
    for l in open(os.path.join(d, 'sparse', '0', 'points3D.txt')):
        if l.startswith('#'): continue
        t = l.split(); tr = np.array(t[8:], int).reshape(-1, 2); cams = np.unique([names[i] for i in tr[:, 0]])
        X.append([float(v) for v in t[1:4]]); C.append([int(v) for v in t[4:7]]); E.append(float(t[7])); TL.append(len(tr))
        first.append(cams[0]); second.append(cams[1] if len(cams) > 1 else cams[0]); track.append(tr)
    return dict(X=np.array(X), C=np.array(C, np.uint8), E=np.array(E), TL=np.array(TL), first=np.array(first), second=np.array(second),
                track=track, names=names, obs=obs)


def init_chart(sheet, Lp, nu, nv, lam, iters=8):
    """A cubic tensor B-spline chart with nu x nv control points fitted to points Lp (upright mm) ONLY. Each point gets
    the (u, v) of fit.py's layout: u across the azimuth range of the owl's chart, v from the foot row to the top edge
    at that azimuth. Least squares on the control points, with second differences of the net as a light smoothness
    term (the net must be defined where the cloud has no points), and robust weights (Tukey) against outliers."""
    th = sheet.azimuth(Lp); r0 = fit.Y0 / sheet.DY - 0.5
    u = np.clip((th - sheet.thL) / (sheet.thR - sheet.thL), 0, 1); v = np.clip((sheet.row(Lp) - r0) / (sheet.top(th) - r0), 0, 1)
    Bu, Bv = fit.basis(nu, u), fit.basis(nv, v); A = (Bv[:, :, None] * Bu[:, None, :]).reshape(len(u), nv * nu)
    d2 = lambda n: np.diff(np.eye(n), 2, axis=0)
    Rm = np.vstack([np.kron(np.eye(nv), d2(nu)), np.kron(d2(nv), np.eye(nu))]); RtR = Rm.T @ Rm
    w = np.ones(len(u))
    for it in range(iters):
        AtA = A.T @ (A * w[:, None]); M = AtA + lam * np.trace(AtA) / np.trace(RtR) * RtR
        Q = np.linalg.solve(M, A.T @ (Lp * w[:, None]))
        res = np.linalg.norm(A @ Q - Lp, axis=1); s = 1.4826 * np.median(res[w > 0]) + 1e-9; c = 4.685 * s
        w = np.where(res < c, (1 - (res / c) ** 2) ** 2, 0.0)
    cell = np.zeros((32, 24), int); np.add.at(cell, (np.minimum((v * 32).astype(int), 31), np.minimum((u * 24).astype(int), 23)), (w > 0).astype(int))
    support = np.rint(np.minimum(255, (A * (w > 0)[:, None]).sum(0))).astype(int)       # inlier weight under each control point's basis
    return Q.reshape(nv, nu, 3), w > 0, dict(lambda_=lam, robust_iterations=iters, cutoff_mm=round(float(c), 3), cells_with_points=round(float((cell > 0).mean()), 4)), support.reshape(nv, nu)


# ------------------------------------------------------------------ check pictures (plain numpy: points, z-sorted)
def picture(sets, path, az=0.0, el=8.0, px=3.0, size=(420, 520), centre=(0.0, 140.0, 0.0)):
    from PIL import Image
    a, e = np.radians(az), np.radians(el)
    R = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]]) @ np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    tiles = []
    for pts, col in sets:
        p = (np.asarray(pts).reshape(-1, 3) - centre) @ R.T; col = np.broadcast_to(np.asarray(col, float).reshape(-1, 3), p.shape)
        W, H = size; img = np.zeros((H, W, 3)); x = (W / 2 + p[:, 0] * px / 2).astype(int); y = (H / 2 - p[:, 1] * px / 2).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H); o = np.argsort(p[ok, 2]); img[y[ok][o], x[ok][o]] = col[ok][o]
        tiles.append(img)
    Image.fromarray((np.clip(np.hstack(tiles), 0, 1) * 255).astype(np.uint8)).save(path)

def shaded(Cn, samples=(300, 400)):
    Sd = fit.evaluate(Cn, np.linspace(0, 1, samples[0]), np.linspace(0, 1, samples[1]))
    n = np.cross(np.gradient(Sd, axis=1), np.gradient(Sd, axis=0)); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    L = np.array([0.35, 0.5, 0.8]); L /= np.linalg.norm(L); s = np.clip(n @ L, 0, 1) * 0.8 + 0.2
    return Sd.reshape(-1, 3), np.repeat(s.reshape(-1, 1), 3, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data'); ap.add_argument('scan', type=int)
    ap.add_argument('--charts', help='the scan\'s data file (default: talk/dtu-data.js for 122, else talk/dtu/scan<N>.js)')
    ap.add_argument('-o', '--out'); ap.add_argument('--cache', help='directory for intermediate results')
    ap.add_argument('--init', default='12x16', help='control points of the chart fitted to the sparse points, columns x rows')
    ap.add_argument('--smooth', type=float, default=1e-3, help='weight of the smoothness term of that fit')
    ap.add_argument('--check', help='directory for check pictures')
    a = ap.parse_args(); d = os.path.join(a.data, 'scan%d' % a.scan); pre = fit.SCANS.get(a.scan, {})
    talk = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk')
    a.charts = a.charts or (os.path.join(talk, 'dtu-data.js') if a.scan == 122 else os.path.join(talk, 'dtu', 'scan%d.js' % a.scan))
    a.out = a.out or os.path.join(talk, 'dtu', 'pipeline%d.js' % a.scan)
    def cached(name, fn):
        p = a.cache and os.path.join(a.cache, name + '.npz')
        if p and os.path.exists(p): z = np.load(p, allow_pickle=False); return {k: z[k] for k in z.files}
        r = fn()
        if p: os.makedirs(a.cache, exist_ok=True); np.savez(p, **r)
        return r

    # ---- 1. fit.py's own steps, in its order, with its presets: frame, fusion, the owl's chart (levels recorded)
    S = fit.load(d); log('scan %d: %d views' % (a.scan, S['n']))
    F = cached('frame', lambda: fit.frame(S)); fit.clean_depth(S)
    lo, hi = (np.array(pre['lo']), np.array(pre['hi'])) if 'lo' in pre else (F['lo'], F['hi'])
    D = cached('fused', lambda: fit.fuse(S, F, lo, hi, pre.get('by_mask', True)))
    sheet = Recorder(D, pre.get('nu'), pre.get('nv'), pre.get('edge', 0.6), pre)
    if not (sheet.nu and sheet.nv): sheet.size(pre.get('nodes', fit.NODES))
    def run():
        G, seen = sheet.fit(); out = dict(G=G, seen=seen)
        for k, l in enumerate(sheet.levels): out['G%d' % k] = l['G']; out['S%d' % k] = l['seen']; out['M%d' % k] = np.array([l['fine'], l.get('delta', 0), l.get('passes', 0)], float)
        return out
    R = cached('levels_%d_%d' % (sheet.nu, sheet.nv), run); G, seen = R['G'], R['seen'].astype(bool)
    levels = [dict(G=R['G%d' % k], seen=R['S%d' % k].astype(bool)) for k in range(5)]
    for k in range(1, 5):                     # the recorded calls are the ones STEPS describes
        st = STEPS[k - 1]; m = R['M%d' % k]; assert (bool(m[0]), m[1], int(m[2])) == (st['onto'] == 'measured', st['delta'], st['equalise']), (k, m, st)
    height = float(sheet.Tall[:, 1].max()); U = pre.get('unit') or float(round(height / fit.HEIGHT_UNITS))
    o = np.array([sheet.axis(0.0)[0], 0.0, sheet.axis(0.0)[1]]); scene = lambda p: (np.asarray(p) - o) / U; mm = lambda p: np.asarray(p) * U + o

    # ---- 2. the same chart as the data file? Its control net, quantised the way fit.py writes it, byte for byte
    txt = open(a.charts).read(); data = json.loads(re.search(r'= (\{"scan".*\});\n', txt).group(1)); chart = data['charts'][0]
    Cn, _ = control_net(G); P = scene(Cn); plo, phi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
    same = np.round((P - plo) / (phi - plo) * 65535).astype('<u2').tobytes() == base64.b64decode(chart['P'])
    cams_same = np.abs(np.round(scene((np.array([-Rc.T @ t for K, Rc, t in S['cams']]) - F['base']) @ F['A']), 3) - np.array(data['cams'])).max()
    log('the chart fitted here is %s the one in %s (%s); camera centres differ by at most %.4f scene units'
        % ('byte for byte' if same else 'NOT', os.path.basename(a.charts), chart['name'], cams_same))
    if not same: raise SystemExit('the levels would not belong to the shipped chart: stop')

    T = D['V'][D['V'][:, 1] > 1.0].astype(float); T = T[sheet.keep(T, 0.0, 1.0)]       # the measured points the shipped stats use
    shipped = unpack_shipped(chart); dsh = distances(mm(shipped), T); log('shipped %d x %d net, as stored: %s' % (chart['nu'], chart['nv'], summary(dsh)))

    # ---- 3. the levels: each level's nodes -> control net (as fit.py makes the final one) -> 12 bits -> measured
    out_levels, nets = [], []
    for k, l in enumerate(levels):
        Gl = l['G']; nv_, nu_ = Gl.shape[:2]; Cl = fit.interpolate(Gl); Pl = scene(Cl); llo, lhi = box(Pl)
        dec = unpack12(pack12(Pl, llo, lhi), nu_ * nv_, llo, lhi).reshape(nv_, nu_, 3); dl = distances(mm(dec), T)
        st = STEPS[max(0, k - 1)]
        e = dict(name='layout' if k == 0 else 'level %d' % k, nu=nu_, nv=nv_, **summary(dl),
                 on_measured=round(float(l['seen'].mean()), 4),
                 spacing_mm=[round(float(np.median(np.linalg.norm(np.diff(Gl, axis=1), axis=-1))), 3), round(float(np.median(np.linalg.norm(np.diff(Gl, axis=0), axis=-1))), 3)],
                 what=('the rectangle of directions (azimuth x row) laid on the envelope, before any relaxation' if k == 0 else
                       '%d relaxation sweeps and %d arc-length pass%s; after each sweep every node is projected onto %s'
                       % (st['relax'], st['equalise'], '' if st['equalise'] == 1 else 'es', 'the envelope (the outermost measured point in each direction from the axis, smoothed): not yet onto the measured surface' if st['onto'] == 'envelope'
                          else 'the nearest measured point within %.1f mm that faces the same way, else onto the envelope' % st['delta'])))
        log('%-9s %3d x %3d  rms %.3f  p95 %.3f  median %.3f mm  within 1 mm %.1f %%  nodes on measured %.1f %%' % (e['name'], nu_, nv_, e['rms_mm'], e['p95_mm'], e['median_mm'], 100 * e['within_1mm'], 100 * e['on_measured']))
        if k < 4: e.update(lo=llo.tolist(), hi=lhi.tolist(), P=pack12(Pl, llo, lhi)); nets.append(mm(dec))
        out_levels.append(e)
    final = dict(name='shipped', nu=chart['nu'], nv=chart['nv'], **summary(dsh), on_measured=round(float(seen.mean()), 4),
                 what='level 4 after fit.py\'s finishing steps (nodes spread over bridges, the foot lowered to the table, a last projection within 1.2 mm, '
                      'holes filled from the envelope) and its control net (interpolation with control points held within 0.6 mm of their nodes): the net of talk/dtu-data.js, not repeated here')

    # ---- 4. cameras: centres as in the data file, rotations and intrinsics of the photographs
    pk = S['photo'][0][0]; sm = np.load(os.path.join(d, 'cameras.npz'))['scale_mat_0']
    cam_out = []; worst = 0.0; off = 0.0
    for i in range(S['n']):
        K, Rc, t = S['cams'][i]; C = scene((-Rc.T @ t - F['base']) @ F['A'])
        _, Rp, Tp = S['photo'][i]; Rs = (Rp * sm[0, 0]) @ F['A']                         # world (scene axes) -> camera of the photograph
        worst = max(worst, np.abs(Rs - Rc @ F['A']).max()); off = max(off, np.linalg.norm(-np.linalg.solve(Rp, Tp) - (-Rc.T @ t)))
        cam_out.append(dict(id=i, c=[round(float(v), 5) for v in C], R=[round(float(v), 6) for v in Rs.ravel()]))
    log('the photographs\' poses vs. the depth maps\' cameras: rotations at most %.2e apart, centres at most %.3f mm' % (worst, off)); checks = dict(rotation_vs_depth_cameras=float('%.2e' % worst), centre_vs_depth_cameras_mm=round(off, 4))
    cameras = dict(count=S['n'], image='images/%04d.png of the scan, camera i = photograph i',
                   intrinsics=dict(fx=float(pk[0, 0]), fy=float(pk[1, 1]), cx=float(pk[0, 2]), cy=float(pk[1, 2]), width=1554, height=1162),
                   note='the same pinhole camera for all 64 photographs (one COLMAP PINHOLE camera; they are crops of the 1600 x 1200 frames that centre the principal point)',
                   R='world (scene axes) -> camera, row-major: the rows are the camera\'s x (right), y (down), z (forward, into the picture) in scene axes',
                   list=cam_out)

    # ---- 5. the dataset's SfM points, in the same frame
    M = read_colmap(d); Xw = M['X'] * sm[0, 0] + sm[:3, 3]; Lp = (Xw - F['base']) @ F['A']; Ps = scene(Lp)
    # check frame, rotations and intrinsics at once: reproject every observation with the cameras as written here
    errs = np.zeros(len(Ps)); ns = np.zeros(len(Ps))
    for k, tr in enumerate(M['track']):
        for iid, j in tr:
            ci = M['names'][iid]; Rs = np.array(cam_out[ci]['R']).reshape(3, 3); c = np.array(cam_out[ci]['c']); q = Rs @ (Ps[k] - c)
            uv = np.array([pk[0, 0] * q[0] / q[2] + pk[0, 2], pk[1, 1] * q[1] / q[2] + pk[1, 2]]); errs[k] += np.linalg.norm(uv - M['obs'][iid][j, :2]); ns[k] += 1
    errs /= ns; checks.update(reprojection_px_median=round(float(np.median(errs)), 4), colmap_px_median=round(float(np.median(M['E'])), 4), largest_difference_px=round(float(np.abs(errs - M['E']).max()), 4))
    cameras['checks'] = dict(checks, note='every observation of the SfM points reprojected through the cameras as written here, against COLMAP\'s own mean reprojection errors')
    log('reprojection through the cameras written here: median %.3f px (COLMAP\'s own: %.3f px), largest difference %.3f px'
                    % (np.median(errs), np.median(M['E']), np.abs(errs - M['E']).max()))
    near = np.all((Ps >= [-5, -0.5, -5]) & (Ps <= [5, 4.5, 5]), axis=1); order = np.r_[np.nonzero(near)[0], np.nonzero(~near)[0]]
    Pn = Ps[order]; nn = int(near.sum()); slo, shi = np.array([-5.0, -0.5, -5.0]), np.array([5.0, 4.5, 5.0])
    above = Lp[:, 1] > fit.Y0; reach = sheet.keep(Lp, 0.0, fit.Y0) & (sheet.radius(Lp) < 250)
    dsurf = distances(mm(shipped), Lp[reach])
    ERR = 0.02
    sfm = dict(source='Y1ue/DTU on Hugging Face, DTU/scan122/sparse/0/points3D.txt (+ images.txt for the image names)',
               how='COLMAP features, matches and triangulation with the dataset\'s calibrated camera poses ("We generate the SFM point cloud with COLMAP given GT camera poses", DTU/README.md of that repo); the cameras were not estimated',
               count=int(len(Ps)), near=nn, box=[slo.tolist(), shi.tolist()],
               note='points 0..near-1 lie in the box (12-bit positions over it); the rest, outside it (background or stray), are stored as float32',
               P=pack12(Pn[:nn], slo, shi), far=b64(Pn[nn:].astype('<f4')), C=b64(M['C'][order]), track=b64(np.minimum(M['TL'][order], 255).astype(np.uint8)),
               error=b64(np.minimum(np.rint(M['E'][order] / ERR), 255).astype(np.uint8)), error_scale=ERR,
               first=b64(M['first'][order].astype(np.uint8)), second=b64(M['second'][order].astype(np.uint8)),
               per_camera=dict(first=np.bincount(M['first'], minlength=S['n']).tolist(), second=np.bincount(M['second'], minlength=S['n']).tolist()),
               stats=dict(error_px_median=round(float(np.median(M['E'])), 3), error_px_p95=round(float(np.percentile(M['E'], 95)), 3), error_px_max=round(float(M['E'].max()), 3),
                          track_median=float(np.median(M['TL'])), track_max=int(M['TL'].max()), tracks_seen_twice_in_one_photo=int(sum(len(np.unique([M['names'][i] for i in tr[:, 0]])) < len(tr) for tr in M['track'])), above_table=int(above.sum()), on_table=int((np.abs(Lp[:, 1]) <= fit.Y0).sum()),
                          below_table=int((Lp[:, 1] < -fit.Y0).sum()), in_chart_reach=int(reach.sum()),
                          to_dense_surface=dict(**summary(dsurf), note='the points in the owl chart\'s reach, distance to the shipped chart')))
    log('SfM: %d points (%d in the box), %d above the table, %d in the owl chart\'s reach; those are %s from the shipped surface'
        % (len(Ps), nn, above.sum(), reach.sum(), summary(dsurf)))

    # ---- 6. a coarse chart fitted to the sparse points only
    inu, inv = (int(v) for v in a.init.split('x'))
    Q, inl, how, support = init_chart(sheet, Lp[reach], inu, inv, a.smooth)
    ds = distances(Q, Lp[reach]); di = ds[inl]; dd = distances(Q, T)
    sub = T[np.random.default_rng(0).choice(len(T), min(40000, len(T)), replace=False)]; Qr = init_chart(sheet, sub, inu, inv, a.smooth)[0]; dr = distances(Qr, T)
    init = dict(name='owl (from SfM)', nu=inu, nv=inv, P=np.round(scene(Q), 4).reshape(-1).tolist(), support=support.reshape(-1).tolist(),
                fit=dict(points=int(reach.sum()), inliers=int(inl.sum()), **how),
                to_sparse=dict(all=summary(ds), inliers=summary(di)), to_dense=summary(dd),
                reference=dict(**summary(dr), note='the same %d x %d fit made to 40,000 of the DENSE measured points instead: what a net this coarse can do at all' % (inu, inv)),
                unsupported=int((support < 1).sum()),
                note='least squares on the control points against the SfM points in the owl chart\'s reach (azimuth range of fit.py\'s chart, above 3 mm, '
                     'below its top edge, within 250 mm of its axis), each at the (u, v) of fit.py\'s layout; second differences of the net as smoothness; '
                     'Tukey weights against outliers. No depth map was used for its geometry (the axis, azimuth range and rows are fit.py\'s, which came from them). '
                     'support = the inlier points\' summed basis weight under each control point; below 1 the net there is set by the smoothness term alone.')
    log('init %d x %d from %d SfM points (%d inliers): to them rms %.3f mm (inliers %.3f), to the dense measured surface rms %.3f p95 %.3f median %.3f mm; '
        'the same fit to dense points: rms %.3f p95 %.3f median %.3f mm; control points without support: %d'
        % (inu, inv, reach.sum(), inl.sum(), init['to_sparse']['all']['rms_mm'], init['to_sparse']['inliers']['rms_mm'], init['to_dense']['rms_mm'], init['to_dense']['p95_mm'], init['to_dense']['median_mm'],
           init['reference']['rms_mm'], init['reference']['p95_mm'], init['reference']['median_mm'], init['unsupported']))

    # ---- 7. write
    obj = dict(scan=a.scan, label=data.get('label'), unit_mm=U,
               source='DTU MVS dataset (Jensen et al., CVPR 2014), scan %d, from the Hugging Face dataset repo Y1ue/DTU' % a.scan,
               note='the steps from photographs to the owl\'s chart, made offline by tools/dtu/pipeline.py: the dataset\'s own COLMAP points, a coarse chart '
                    'fitted to them, and the coarse-to-fine levels of fit.py\'s geometric fit to the fused depth maps. Nothing here is trained.',
               frame='the scene frame and units of %s: +y up, the table at y = 0, one unit = %g mm' % (os.path.relpath(a.charts, os.path.join(talk, '..')), U),
               cameras=cameras, sfm=sfm, init=init, levels=out_levels[:4], level_last=out_levels[4], shipped=final,
               photos=None, photos_note='left out: the terms of use on the dataset\'s official page (roboimagedata.compute.dtu.dk) could not be read from the session that made this file')
    js = HEAD % dict(scan=a.scan) + json.dumps(obj, separators=(',', ':')) + ';\n' + TAIL
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as fh: fh.write(js)
    parts = {k: len(json.dumps(v, separators=(',', ':'))) for k, v in obj.items() if k in ('cameras', 'sfm', 'init', 'levels')}
    log('wrote %s, %.0f KB (%s)' % (os.path.normpath(a.out), len(js) / 1e3, ', '.join('%s %.0f KB' % (k, v / 1e3) for k, v in parts.items())))

    if a.check:
        os.makedirs(a.check, exist_ok=True)
        cols = M['C'][reach] / 255.0
        sets = [(Lp[reach], np.clip(cols * 1.6, 0, 1))] + [shaded(Q)] + [shaded(n_) for n_ in nets] + [shaded(mm(shipped))]
        picture(sets, os.path.join(a.check, 'pipeline%d-front.png' % a.scan))
        picture(sets, os.path.join(a.check, 'pipeline%d-left.png' % a.scan), az=-55)
        log('check pictures: SfM points, init, layout, level 1..3, shipped -> %s' % a.check)


def unpack_shipped(chart):
    q = np.frombuffer(base64.b64decode(chart['P']), '<u2').reshape(chart['nv'], chart['nu'], 3).astype(float)
    return np.array(chart['lo']) + (np.array(chart['hi']) - np.array(chart['lo'])) * q / 65535


HEAD = """/* DTU scan %(scan)d, from photographs to charts: the data for the talk's pipeline chapter. GENERATED by tools/dtu/pipeline.py:
   do not edit, re-run the script. Everything is in the scene frame and units of talk/dtu-data.js (+y up, table at y = 0).
   It holds the scan's sparse COLMAP points (triangulated by the maker of that copy with DTU's calibrated cameras),
   a coarse chart fitted to those points only, and the coarse-to-fine levels of fit.py's GEOMETRIC fit of the owl's
   chart to the fused depth maps, each with its measured distance to the measured surface. Nothing here is trained,
   by the thesis method or otherwise.
   KnotTalk.dtuPipeline[%(scan)d] = {
     cameras: { count, intrinsics: { fx, fy, cx, cy, width, height }, list: [{ id, c: [x,y,z], R: [9] }] }
       R = world (scene axes) -> camera, row-major, OpenCV axes (x right, y down, z forward); a pixel (px, py) of the
       photograph looks along R^T [(px - cx) / fx, (py - cy) / fy, 1] from c. For a three.js camera: columns x, -y, -z.
       Camera i = images/%%04d.png of the scan; COLMAP's IMAGE_ID is i + 1.
     sfm: { count, near, P: Float32Array(count*3), C: Uint8Array(count*3) colour, track: Uint8Array (observations),
            error: Float32Array (mean reprojection error, px), first / second: Uint8Array (the smallest and second smallest
            camera index in the point's track: a point can be triangulated once its second camera is there),
            per_camera: { first, second } counts, stats }
     init: { nu, nv, P: Float32Array(nu*nv*3) control points, support, fit, to_sparse, to_dense }   (row j, column i -> j*nu+i)
     levels: [{ name, nu, nv, P: Float32Array(nu*nv*3) control points of the spline through that level's nodes, rms_mm,
                p95_mm, median_mm, within_1mm (distance from the measured points to it), on_measured, what }]
     level_last, shipped: the numbers of the last level and of the shipped net (talk/dtu-data.js), without the nets }
   Clamped uniform cubic knots in u and v, as KnotTalk.nurbs.makePatch makes them. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {}, D = (KT.dtuPipeline = KT.dtuPipeline || {})[%(scan)d] = """

TAIL = """  function bytes(s) { var b = atob(s), n = b.length, a = new Uint8Array(n), i; for (i = 0; i < n; i++) a[i] = b.charCodeAt(i); return a; }
  function u12(s, n, lo, hi, out, off) {         // n numbers, 12 bits each, two in three bytes, over lo..hi per axis
    var b = bytes(s), i, k, q;
    for (i = 0; i < n; i++) {
      k = (i >> 1) * 3; q = i & 1 ? (b[k + 1] >> 4) | (b[k + 2] << 4) : b[k] | ((b[k + 1] & 15) << 8);
      out[off + i] = lo[i % 3] + (hi[i % 3] - lo[i % 3]) * q / 4095;
    }
    return out;
  }
  var s = D.sfm, P = new Float32Array(s.count * 3), far = new Float32Array(bytes(s.far).buffer), e = bytes(s.error), E = new Float32Array(s.count), i;
  u12(s.P, s.near * 3, s.box[0], s.box[1], P, 0); P.set(far, s.near * 3);
  for (i = 0; i < s.count; i++) E[i] = e[i] * s.error_scale;
  s.P = P; s.C = bytes(s.C); s.track = bytes(s.track); s.error = E; s.first = bytes(s.first); s.second = bytes(s.second);
  D.init.P = new Float32Array(D.init.P);
  D.levels.forEach(function (l) { l.P = u12(l.P, l.nu * l.nv * 3, l.lo, l.hi, new Float32Array(l.nu * l.nv * 3), 0); });
})();
"""

if __name__ == '__main__':
    main()
