#!/usr/bin/env python3
"""The spline charts of a fitted DTU scan as triangle meshes in the DTU world frame, in millimetres.

    python3 -I tools/eval/chart_mesh.py talk/dtu-data.js    --cameras DATA/scan122/cameras.npz -o OUT
    python3 -I tools/eval/chart_mesh.py talk/dtu/scan55.js  --cameras DATA/scan55/cameras.npz  -o OUT --budget 160000 30000

Input: a file written by tools/dtu/fit.py (`KnotTalk.dtu` or `KnotTalk.dtuScans[N]`): control nets in scene units
(+y up, the table at y = 0, one unit = `unit_mm`), and `cams`, the scan's camera centres in that frame. These
charts were FITTED OFFLINE to the dataset's depth maps; they are not trained and not a result of the thesis method.

The frame. fit.py maps world millimetres X to scene coordinates p by  L = (X - base) @ A,  p = (L - o) / unit_mm
(A: a rotation whose columns are right, up, front; base, o: points on the table), so  X = R (unit_mm p) + t  with
R = A and t = base + A o. Neither A nor base nor o is in the file, but the camera centres are on both sides: in the
file in scene units, and in the dataset's `cameras.npz` (`world_mat_i` = K [R|t], the null vector of which is the
centre) in the DTU world frame, the frame of the reference clouds. R, t = the rigid fit (Kabsch, no scale, no
reflection) between the two sets. The script stops unless they agree to --tolerance (0.2 mm): the file stores the
centres with three decimals of a scene unit, which alone is up to 0.07 mm. With --calibration DIR (the dataset's
`Calibration/cal18` folder with `pos_NNN.txt`) the centres of `cameras.npz` are also checked against the official
projection matrices.

Meshes written (PLY, binary, float x y z + triangles), each as `<scan>_<lattice>_object.ply` (the object's chart)
and `<scan>_<lattice>_all.ply` (object + table chart):

  full            one vertex per control point: the spline at the Greville abscissae of its clamped cubic knots, the
                  parameters at which fit.py's nodes sit; the lattice reaches the chart's border. --refine K lays K
                  times as many vertices per axis between them.
  talk<N>         what the talk's `mesh` / `stitch` scenes draw and export at a budget of N Gaussians: one even
                  lattice per chart at the cell centres (u = (a + 0.5) / na), the budget shared by area x gain and the
                  cells square on the surface (`sizeLattices` and `coarse` in talk/sims/manifold.js, ported here),
                  stitched by the running 2 x 2 window (`lattice` in talk/meshlist.js). N = 160,000 is the talk's
                  default for a DTU scan on a desktop, 30,000 on a small screen; the slider runs from 6,000 to 300,000.

`<scan>_frame.json` records the transform, its residuals and every lattice written. The evaluation itself:
tools/eval/dtu_chamfer.py.
"""
import argparse, base64, json, os, re, sys

import numpy as np
from scipy.interpolate import BSpline

NOTE = 'spline charts fitted offline to the DTU depth maps by tools/dtu/fit.py; not trained, not a result of the thesis method'
# how talk/scenes.js tiles a scanned object's chart and its table (function scan: tu, tv, cu, cv -> gu = tu * cu, gv = tv * cv)
COARSE = dict(object=(9 * 4, 12 * 4), table=(5 * 5, 4 * 5))


def read_charts(path):
    """The JSON object inside a file written by fit.py, with P decoded as its own loader does (Float32)."""
    s = open(path, encoding='utf-8').read(); m = re.search(r'=\s*(\{"scan":)', s)
    if not m: sys.exit('%s: no `= {"scan": ...}` object; is this a file written by tools/dtu/fit.py?' % path)
    d = json.JSONDecoder().raw_decode(s, m.start(1))[0]
    for c in d['charts']:
        nu, nv = c['nu'], c['nv']
        if 'plane' in c:
            x0, x1, z0, z1 = c['plane']; P = np.zeros((nv, nu, 3), np.float32)
            P[..., 0] = (np.float32(x0) + (x1 - x0) * np.arange(nu) / (nu - 1))[None, :]; P[..., 2] = (np.float32(z0) + (z1 - z0) * np.arange(nv) / (nv - 1))[:, None]
        else:
            q = np.frombuffer(base64.b64decode(c['P']), '<u2').reshape(nv, nu, 3).astype(float); lo, hi = np.array(c['lo']), np.array(c['hi'])
            P = (lo + (hi - lo) * q / 65535).astype(np.float32)
        c['net'] = P.astype(float); c['kind'] = 'table' if 'plane' in c else 'object'
    return d


# ---------- clamped cubic B-splines, the knots of KnotTalk.nurbs.clampedKnots(n, 3) and of fit.py
def knots(n): return np.r_[np.zeros(4), np.arange(1, n - 3) / (n - 3), np.ones(4)]
def greville(n): U = knots(n); return (U[1:n + 1] + U[2:n + 2] + U[3:n + 3]) / 3
def basis(n, t): return BSpline.design_matrix(np.clip(t, 0, 1), knots(n), 3).toarray()
def dbasis(n, t): return BSpline(knots(n), np.eye(n), 3).derivative(1)(np.clip(t, 0, 1))


def surface(net, us, vs, deriv=False):
    """S(u, v) on the grid vs x us -> (len(vs), len(us), 3); with deriv also S_u and S_v."""
    nv, nu = net.shape[:2]; Bu, Bv = basis(nu, us), basis(nv, vs); S = np.einsum('bj,jic,ai->bac', Bv, net, Bu, optimize=True)
    if not deriv: return S
    return S, np.einsum('bj,jic,ai->bac', Bv, net, dbasis(nu, us), optimize=True), np.einsum('bj,jic,ai->bac', dbasis(nv, vs), net, Bu, optimize=True)


def stitch(na, nb, off=0):
    """The running 2 x 2 window of talk/meshlist.js: per stop (v00, v10, v11), (v00, v11, v01)."""
    a, b = np.meshgrid(np.arange(na - 1), np.arange(nb - 1)); v00 = (off + b * na + a).ravel(); v10 = v00 + 1; v01 = v00 + na; v11 = v01 + 1
    return np.stack([np.c_[v00, v10, v11], np.c_[v00, v11, v01]], 1).reshape(-1, 3)


def talk_lattices(charts, budget):
    """`coarse` + the gain rule + `sizeLattices` of talk/sims/manifold.js -> {chart name: (na, nb)}."""
    info = []
    for c in charts:
        gu, gv = COARSE[c['kind']]; _, su, sv = surface(c['net'], (np.arange(gu) + 0.5) / gu, (np.arange(gv) + 0.5) / gv, True)
        nl = np.linalg.norm(np.cross(su, sv), axis=-1); n = gu * gv; carea = nl / n; clu = np.linalg.norm(su, axis=-1); clv = nl / np.maximum(clu, 1e-9)
        info.append(dict(area=carea.sum(), lu=(clu * carea).sum(), lv=(clv * carea).sum()))
    rho = [c['nu'] * c['nv'] / i['area'] for c, i in zip(charts, info)]
    for i, r in zip(info, rho): i['gain'] = float(np.clip((r / max(rho)) ** 0.35, 0.3, 1))
    W = sum(i['area'] * i['gain'] for i in info); out = {}; js_round = lambda x: int(np.floor(x + 0.5))
    for c, i in zip(charts, info):
        n = budget * i['area'] * i['gain'] / W; nb = max(2, js_round(np.sqrt(n * i['lv'] / i['lu']))); na = max(2, js_round(n / nb)); out[c['name']] = (na, nb)
    return out


def centres_npz(path):
    z = np.load(path, allow_pickle=False); n = sum(1 for k in z.files if k.startswith('world_mat_') and 'inv' not in k); C = []
    for i in range(n):
        P = z['world_mat_%d' % i][:3].astype(float); C.append(-np.linalg.solve(P[:, :3], P[:, 3]))
    return np.array(C)


def rigid(a, b):
    """R, t with b ~ a @ R.T + t (Kabsch: a proper rotation, no scale), and the scale a similarity would have chosen."""
    ca, cb = a.mean(0), b.mean(0); U, s, Vt = np.linalg.svd((a - ca).T @ (b - cb)); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T; return R, cb - R @ ca, float((s * np.diag(D)).sum() / ((a - ca) ** 2).sum())


def write_ply(path, V, F, comments):
    head = 'ply\nformat binary_little_endian 1.0\n' + ''.join('comment %s\n' % c for c in comments) + 'element vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n' % (len(V), len(F))
    a = np.zeros(len(F), [('n', 'u1'), ('i', '<i4', (3,))]); a['n'] = 3; a['i'] = F
    with open(path, 'wb') as f: f.write(head.encode('ascii')); f.write(np.asarray(V, '<f4').tobytes()); f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='See tools/eval/README.md.')
    ap.add_argument('charts', help='talk/dtu-data.js or talk/dtu/scan<N>.js')
    ap.add_argument('--cameras', required=True, help='the scan\'s cameras.npz (IDR convention, as in the Hugging Face dataset repo Y1ue/DTU: DTU/scan<N>/cameras.npz)')
    ap.add_argument('--calibration', help='optional: the dataset\'s Calibration/cal18 folder (pos_NNN.txt), to check cameras.npz against the official projection matrices')
    ap.add_argument('-o', '--out', required=True, help='folder for the PLY files and <scan>_frame.json')
    ap.add_argument('--budget', type=int, nargs='*', default=[160000], help='talk lattices to write, as Gaussian budgets (default: 160000, the talk\'s default for a scan)')
    ap.add_argument('--refine', type=int, default=1, help='K: also write full lattices with K times the vertices per axis (default 1: only the plain one)')
    ap.add_argument('--tolerance', type=float, default=0.2, help='largest distance allowed between a camera centre mapped from the file and the dataset\'s, mm (default 0.2)')
    a = ap.parse_args()
    d = read_charts(a.charts); scan, unit = int(d['scan']), float(d['unit_mm']); os.makedirs(a.out, exist_ok=True)
    src = np.array(d['cams'], float) * unit; dst = centres_npz(a.cameras)
    if len(src) != len(dst): sys.exit('%d cameras in %s, %d in %s' % (len(src), a.charts, len(dst), a.cameras))
    R, t, scale = rigid(src, dst); res = np.linalg.norm(src @ R.T + t - dst, axis=1)
    frame = dict(scan=scan, unit_mm=unit, note=NOTE, charts_file=os.path.abspath(a.charts), cameras=os.path.abspath(a.cameras),
                 transform='X_world_mm = R @ (unit_mm * p_scene) + t', R=R.tolist(), t=t.tolist(),
                 check=dict(cameras=len(src), residual_rms_mm=float(np.sqrt((res ** 2).mean())), residual_max_mm=float(res.max()), similarity_scale=scale,
                            stored_precision_mm=unit * 0.0005 * np.sqrt(3), tolerance_mm=a.tolerance))
    print('scan %d: %d cameras, scene -> world: rms %.4f mm, max %.4f mm (the file rounds to +-%.3f mm per axis); a similarity would scale by %.6f'
          % (scan, len(src), frame['check']['residual_rms_mm'], res.max(), unit * 0.0005, scale))
    if res.max() > a.tolerance or abs(scale - 1) > 1e-3: sys.exit('the camera centres do not map onto each other to %.2f mm: wrong cameras.npz for this file?' % a.tolerance)
    if a.calibration:
        pos = []
        for i in range(len(dst)):
            P = np.loadtxt(os.path.join(a.calibration, 'pos_%03d.txt' % (i + 1))); pos.append(-np.linalg.solve(P[:, :3], P[:, 3]))
        dev = np.linalg.norm(np.array(pos) - dst, axis=1); frame['check']['official_calibration'] = dict(dir=os.path.abspath(a.calibration), max_mm=float(dev.max()), rms_mm=float(np.sqrt((dev ** 2).mean())))
        print('  cameras.npz against the official pos_NNN.txt: centres differ by at most %.5f mm' % dev.max())
        if dev.max() > a.tolerance: sys.exit('cameras.npz is not in the frame of the official calibration')
    to_world = lambda p: (p.reshape(-1, 3) * unit) @ R.T + t
    charts = d['charts']; frame['charts'] = [dict(name=c['name'], kind=c['kind'], nu=c['nu'], nv=c['nv']) for c in charts]; frame['meshes'] = []
    # the table of the charts in the world frame: a point on it and its normal (+y of the scene)
    frame['table_plane'] = dict(point=t.tolist(), normal=R[:, 1].tolist(), what='the plane y = 0 of the scene frame, fit.py\'s table')

    def emit(tag, lat, what):
        """lat: {chart name: (us, vs)} -> two files, the object alone and object + table."""
        parts = []
        for c in charts:
            us, vs = lat[c['name']]; S = surface(c['net'], us, vs); W = to_world(S).reshape(len(vs), len(us), 3)
            eu = np.linalg.norm(np.diff(W, axis=1), axis=-1); ev = np.linalg.norm(np.diff(W, axis=0), axis=-1)
            parts.append(dict(c=c, V=W.reshape(-1, 3), na=len(us), nb=len(vs), eu=float(np.median(eu)), ev=float(np.median(ev))))
        for name, sel in (('object', [p for p in parts if p['c']['kind'] == 'object']), ('all', parts)):
            V, F, off = [], [], 0
            for p in sel: V.append(p['V']); F.append(stitch(p['na'], p['nb'], off)); off += len(p['V'])
            V, F = np.vstack(V), np.vstack(F); fn = os.path.join(a.out, 'scan%d_%s_%s.ply' % (scan, tag, name))
            tri = V[F]; area = float(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2)
            write_ply(fn, V, F, ['Knot-a-Surface: DTU scan %d, %s' % (scan, what), NOTE, 'DTU world frame, millimetres', 'parts: ' + ', '.join('%s %d x %d' % (p['c']['name'], p['na'], p['nb']) for p in sel)])
            frame['meshes'].append(dict(file=os.path.basename(fn), lattice=tag, what=what, charts=[dict(name=p['c']['name'], na=p['na'], nb=p['nb'], median_edge_mm=[round(p['eu'], 4), round(p['ev'], 4)]) for p in sel],
                                        vertices=int(len(V)), triangles=int(len(F)), area_mm2=round(area, 1)))
            print('  %-34s %s: %d vertices, %d triangles, %.0f mm2' % (os.path.basename(fn), ' + '.join('%s %d x %d (%.2f x %.2f mm)' % (p['c']['name'], p['na'], p['nb'], p['eu'], p['ev']) for p in sel), len(V), len(F), area))

    for k in sorted(set([1, max(1, a.refine)])):
        def grid(n):
            g = greville(n); return g if k == 1 else np.interp(np.linspace(0, n - 1, k * (n - 1) + 1), np.arange(n), g)
        emit('full' if k == 1 else 'full_x%d' % k, {c['name']: (grid(c['nu']), grid(c['nv'])) for c in charts},
             'one vertex per control point (the spline at its Greville abscissae)' if k == 1 else '%d x %d vertices per control point' % (k, k))
    for N in a.budget:
        sz = talk_lattices(charts, N)
        emit('talk%d' % N, {n: ((np.arange(na) + 0.5) / na, (np.arange(nb) + 0.5) / nb) for n, (na, nb) in sz.items()},
             'the lattice the talk stitches at a budget of %d Gaussians' % N)
    fn = os.path.join(a.out, 'scan%d_frame.json' % scan)
    with open(fn, 'w') as f: json.dump(frame, f, indent=1)
    print('  wrote %s' % fn)


if __name__ == '__main__':
    main()
