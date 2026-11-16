#!/usr/bin/env python3
"""Snapshots of a plain 3DGS run on DTU scan 122 -> talk/dtu/train122.js (KnotTalk.dtuTrain[122]), for the `train`
scene of the talk's pipeline chapter (talk/sims/pipeline.js).

    python3 -I tools/train/export.py <run-dir> [-o talk/dtu/train122.js] [--budget 60000] [--seed 0] [--rule area]

<run-dir> is what tools/train/train3dgs.py wrote with graphdeco-inria/gaussian-splatting:
  point_cloud/iteration_<k>/point_cloud.ply   the Gaussians, graphdeco's vertex layout: x, y, z, nx, ny, nz,
                                              f_dc_0..2, f_rest_*, opacity, scale_0..2, rot_0..3 (log-scales, logit
                                              opacity, unnormalised quaternion w x y z, COLMAP frame)
  log.csv, run.json                           the wrapper's log and the run's facts (implementation, GPU, date)
  cameras.json, input.ply                     graphdeco's own copies of the cameras and of the SfM points it started from
A run of graphdeco's own train.py works too: no log then, and the note says what is unknown.

FRAME. The Gaussians live in the COLMAP frame of the dataset's sparse/0 model. tools/dtu/pipeline.py maps that frame
into the owl's (talk/dtu-data.js: +y up, the table at y = 0, 1 unit = 80 mm): X_mm = X * s + c (scale_mat_0 of
cameras.npz), L = (X_mm - base) @ A (fit.py's table frame), P = (L - o) / 80 (o: the owl's axis). That is a similarity
P = k R X + T. pipeline.py wrote the 64 cameras into talk/dtu/pipeline122.js in the owl's frame (centre, rotation);
the run holds the same cameras in the COLMAP frame (cameras.json; or images.txt with --data). R comes from the
rotations, k and T from the centres. Then a check that does not use the cameras: every one of the dataset's SfM points
(input.ply, or points3D.txt with --data), mapped, has to land on the copy of it that pipeline122.js holds in the owl's
frame, within that file's own rounding. If not, nothing is written.

PER GAUSSIAN: centre k R x + T; rotation q_R * q (q normalised; Hamilton product, w x y z); sigma = exp(scale) * k
per axis; colour = 0.5 + 0.28209479 * f_dc, clamped, 8 bits (the DC term: no view dependence); opacity = sigmoid,
8 bits. Rows with a non-finite number are left out (counted in ntotal, reported).

BUDGET. A snapshot with more than --budget Gaussians keeps --budget of them, drawn without replacement with
probability growing with opacity x (s1 s2 + s2 s3 + s3 s1), the Gaussian's mean footprint on screen up to a constant
(Efraimidis-Spirakis keys log(u) / w, u from numpy's PCG64 seeded with [seed, iteration]). --rule uniform draws
uniformly. Kept Gaussians stay in the checkpoint's order. Same input, seed and numpy: the same bytes.

STORAGE (the file decodes itself at load, like talk/dtu-data.js): centres inside a box of 10 x 5 x 10 units around
the owl (pipeline.py's box for the SfM points) as 16-bit integers (steps of 0.012 mm), the rest as float32; sigmas as
8-bit logarithms over the snapshot's range; quaternions as their three smallest components at 8 bits each plus 2 bits
naming the one left out; colour and opacity as bytes. About 16.3 bytes per Gaussian, 21.7 in base64.

Standard library and numpy only, no GPU. Nothing here trains anything; it converts what a run left behind."""
import argparse, base64, csv, json, math, os, re, sys
import numpy as np

C0 = 0.28209479177387814                                  # SH band 0: colour = 0.5 + C0 * f_dc
BOX = np.array([[-5.0, -0.5, -5.0], [5.0, 4.5, 5.0]])    # owl units: pipeline.py's box for the SfM points
FLOOR = math.log(1e-5)                                    # sigmas below 1e-5 units (0.0008 mm) are stored as that
NEED = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
TALK = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk'))


def say(*a): print(*a, flush=True)


# ------------------------------------------------------------------ input
PLY_TYPES = {'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1', 'short': 'i2', 'int16': 'i2', 'ushort': 'u2',
             'uint16': 'u2', 'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4', 'float': 'f4', 'float32': 'f4',
             'double': 'f8', 'float64': 'f8'}

def read_ply(path):
    """The vertex element of a binary PLY with scalar properties (what 3DGS writes) as a structured array."""
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply': raise SystemExit('%s: not a PLY file' % path)
        fmt, elems = None, []
        while True:
            line = f.readline()
            if not line: raise SystemExit('%s: no end_header' % path)
            t = line.decode('ascii', 'replace').split()
            if not t or t[0] in ('comment', 'obj_info'): continue
            if t[0] == 'end_header': break
            if t[0] == 'format': fmt = t[1]
            elif t[0] == 'element': elems.append((t[1], int(t[2]), []))
            elif t[0] == 'property':
                if t[1] == 'list': raise SystemExit('%s: list properties are not supported' % path)
                elems[-1][2].append((t[2], PLY_TYPES[t[1]]))
        if fmt not in ('binary_little_endian', 'binary_big_endian'): raise SystemExit('%s: %s PLY, only binary is supported' % (path, fmt))
        data = f.read()
    e, off = '<' if fmt == 'binary_little_endian' else '>', 0
    for name, n, props in elems:
        dt = np.dtype([(p, e + t) for p, t in props])
        if name == 'vertex':
            if len(data) < off + n * dt.itemsize: raise SystemExit('%s: file ends before its %d vertices' % (path, n))
            return np.frombuffer(data, dt, count=n, offset=off)
        off += n * dt.itemsize
    raise SystemExit('%s: no vertex element' % path)

def read_gaussians(path):
    v = read_ply(path); miss = [k for k in NEED if k not in v.dtype.names]
    if miss: raise SystemExit('%s: not the 3DGS vertex layout, no %s' % (path, ', '.join(miss)))
    col = lambda *k: np.stack([v[i].astype(np.float64) for i in k], 1)
    return dict(x=col('x', 'y', 'z'), dc=col('f_dc_0', 'f_dc_1', 'f_dc_2'), op=v['opacity'].astype(np.float64),
                sc=col('scale_0', 'scale_1', 'scale_2'), rot=col('rot_0', 'rot_1', 'rot_2', 'rot_3'))

def read_pipeline(path):
    m = re.search(r'\|\| \{\}\)\[(\d+)\] = (\{.*\});\n', open(path).read())
    if not m: raise SystemExit('%s: not a file written by tools/dtu/pipeline.py' % path)
    return int(m.group(1)), json.loads(m.group(2))

def photo_index(name): return int(os.path.splitext(os.path.basename(name))[0])     # 0000.png -> 0, as pipeline.py counts

def cameras_run(run):
    """graphdeco's cameras.json: per camera the centre ('position') and the camera-to-world rotation, COLMAP frame."""
    return {photo_index(c['img_name']): (np.array(c['rotation'], float).T, np.array(c['position'], float))
            for c in json.load(open(os.path.join(run, 'cameras.json')))}

def cameras_colmap(data):
    L = [l for l in open(os.path.join(data, 'sparse', '0', 'images.txt')) if not l.startswith('#')]
    out = {}
    for l in L[0::2]:
        t = l.split(); R = qmat(np.array([float(v) for v in t[1:5]])); tv = np.array([float(v) for v in t[5:8]])
        out[photo_index(t[9])] = (R, -R.T @ tv)                         # world -> camera rotation, centre
    return out

def sfm_points(path):
    if path.endswith('.ply'):
        v = read_ply(path); return np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    return np.array([[float(a) for a in l.split()[1:4]] for l in open(path) if l.strip() and not l.startswith('#')])


# ------------------------------------------------------------------ rotations (w, x, y, z), Hamilton
def qmat(q):
    """Unit quaternion(s) -> rotation matrix: COLMAP's qvec2rotmat, 3DGS's build_rotation."""
    w, x, y, z = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)

def qmul(a, b):
    """a * b: rotate by b, then by a (qmat(a * b) = qmat(a) @ qmat(b))."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0); bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)

def quat(R):
    """Rotation matrix -> unit quaternion with w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2 * math.sqrt(t + 1); q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R))); j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]); q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s; q[1 + i] = s / 4; q[1 + j] = (R[j, i] + R[i, j]) / s; q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q) / np.linalg.norm(q)
    q = -q if q[0] < 0 else q
    assert np.abs(qmat(q) - R).max() < 1e-9
    return q


# ------------------------------------------------------------------ the frame: pipeline.py's similarity, read back
def similarity(pipe, cams):
    L = pipe['cameras']['list']; ids = [c['id'] for c in L]; miss = [i for i in ids if i not in cams]
    if miss: raise SystemExit('cameras %s of pipeline122.js are not in the run: not the same photographs' % miss[:8])
    Rs = np.array([np.reshape(c['R'], (3, 3)) for c in L]); Cs = np.array([c['c'] for c in L], float)
    Rc = np.array([cams[i][0] for i in ids]); Cc = np.array([cams[i][1] for i in ids])
    each = np.einsum('kji,kjl->kil', Rs, Rc)              # R_owl^T R_colmap: the same rotation for every camera
    u, _, vt = np.linalg.svd(each.sum(0)); R = u @ vt
    if np.linalg.det(R) < 0: raise SystemExit('the two frames differ by a reflection: not the model pipeline.py read')
    deg = np.degrees(2 * np.arcsin(np.minimum(1, np.linalg.norm(each - R, axis=(1, 2)) / (2 * math.sqrt(2))))).max()   # |R1 - R2|_F = 2 sqrt2 sin(angle / 2)
    Y = Cc @ R.T; my, mc = Y.mean(0), Cs.mean(0)
    k = float(((Y - my) * (Cs - mc)).sum() / ((Y - my) ** 2).sum()); T = mc - k * my
    res = np.linalg.norm(k * Y + T - Cs, axis=1).max()
    return k, R, T, dict(cameras=len(ids), rotation_deg=float(deg), centre_units=float(res))

def unpack12(s, n):
    by = np.frombuffer(base64.b64decode(s), np.uint8).astype(np.int64)
    q = np.empty(len(by) // 3 * 2, np.int64); q[0::2] = by[0::3] | ((by[1::3] & 15) << 8); q[1::2] = (by[1::3] >> 4) | (by[2::3] << 4)
    return q[:n * 3].reshape(-1, 3)

def sfm_check(pipe, X, k, R, T):
    """Map the SfM points X (COLMAP frame, any order) and match them one to one with the copy in pipeline122.js:
    12-bit integers over a box for the near ones (the deviation is reported in 12-bit steps; rounding alone gives at
    most 0.5), float32 for the rest (deviation in scene units)."""
    s = pipe['sfm']; lo, hi = np.array(s['box'][0], float), np.array(s['box'][1], float); nn = s['near']; step = (hi - lo) / 4095
    P = k * X @ R.T + T; near = np.all((P >= lo) & (P <= hi), 1)
    out = dict(points=len(X), stored=s['count'], near=int(near.sum()), stored_near=nn)
    if len(X) != s['count'] or near.sum() != nn: return dict(out, ok=False)
    Q = unpack12(s['P'], nn); key = lambda a: (a[:, 0] * 4096 + a[:, 1]) * 4096 + a[:, 2]
    f = (P[near] - lo) / step; r = np.rint(f).astype(np.int64); stored = np.sort(key(Q)); match = key(r)
    for i in np.nonzero(~np.isin(match, stored))[0]:   # a point within a hair of a rounding boundary: try the neighbours
        nb = r[i] + np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T
        dev = np.abs(f[i] - nb).max(1); ok = np.isin(key(nb), stored)
        if ok.any(): j = np.argmin(np.where(ok, dev, np.inf)); match[i] = key(nb[j:j + 1])[0]
    same = bool(np.array_equal(np.sort(match), stored))
    mq = np.stack([match // 4096 ** 2, match // 4096 % 4096, match % 4096], 1)
    F = np.frombuffer(base64.b64decode(s['far']), '<f4').reshape(-1, 3).astype(np.float64); Pf = P[~near]
    d = np.abs(Pf[:, None, :] - F[None, :, :]).max(2) if len(F) else np.zeros((0, 0))
    far_dev = float(d.min(1).max()) if len(F) else 0.0; far_one = len(F) == 0 or len(np.unique(d.argmin(1))) == len(F)
    out.update(near_steps=float(np.abs(f - mq).max()), far_units=far_dev)
    out['ok'] = bool(same and far_one and out['near_steps'] < 0.51 and far_dev < 1e-4)
    return out


# ------------------------------------------------------------------ one snapshot
def owl_frame(G, k, R, T, qR):
    with np.errstate(over='ignore', invalid='ignore'):
        P = k * G['x'] @ R.T + T; n = np.linalg.norm(G['rot'], axis=1); S = np.exp(G['sc']) * k
        q = qmul(qR, G['rot'] / np.where(n > 0, n, 1)[:, None]); q /= np.linalg.norm(q, axis=1)[:, None]
    ok = (np.isfinite(P).all(1) & np.isfinite(S).all(1) & np.isfinite(q).all(1) & (n > 0)
          & np.isfinite(G['op']) & np.isfinite(G['dc']).all(1))
    a = 0.5 * (1 + np.tanh(0.5 * G['op']))                                        # sigmoid, without overflow
    C = np.rint(np.clip(np.nan_to_num(0.5 + C0 * G['dc']), 0, 1) * 255).astype(np.uint8)
    return ok, P, S, q, C, a

def choose(w, budget, seed, it, rule):
    if len(w) <= budget: return np.arange(len(w))
    u = np.random.default_rng([seed, it]).random(len(w))
    key = u if rule == 'uniform' else np.log(u) / np.maximum(w, 1e-300)            # larger key = kept
    return np.sort(np.argpartition(-key, budget - 1)[:budget])

def b64(a, dt=None): return base64.b64encode(np.ascontiguousarray(a if dt is None else np.asarray(a).astype(dt)).tobytes()).decode('ascii')

def encode(it, ntotal, P, S, q, C, a):
    lo, hi = BOX; near = np.all((P >= lo) & (P <= hi), 1); o = np.r_[np.nonzero(near)[0], np.nonzero(~near)[0]]
    P, S, q, C, a = P[o], S[o], q[o], C[o], a[o]; n, m = len(P), int(near.sum())
    pq = np.clip(np.rint((P[:m] - lo) / (hi - lo) * 65535), 0, 65535)
    ls = np.log(np.maximum(S, 1e-300)); slo = math.floor(max(ls.min(), FLOOR) * 1e4) / 1e4 if n else 0.0
    shi = math.ceil(max(ls.max() if n else 0.0, slo + 1e-3) * 1e4) / 1e4
    sq = np.rint((np.clip(ls, slo, shi) - slo) / (shi - slo) * 255)
    big = np.argmax(np.abs(q), 1); q = q * np.where(q[np.arange(n), big] < 0, -1.0, 1.0)[:, None]
    keep = np.ones((n, 4), bool); keep[np.arange(n), big] = False
    qq = np.clip(np.rint((q[keep].reshape(n, 3) * math.sqrt(2) + 1) * 127.5), 0, 255)
    ib = np.zeros(-(-n // 4) * 4, np.uint8); ib[:n] = big; ib = ib.reshape(-1, 4); ib = ib[:, 0] | ib[:, 1] << 2 | ib[:, 2] << 4 | ib[:, 3] << 6
    A = np.rint(a * 255)
    return dict(iter=int(it), ntotal=int(ntotal), n=n, near=m, box=BOX.tolist(), slog=[slo, shi],
                P=b64(pq, '<u2'), F=b64(P[m:], '<f4'), S=b64(sq, np.uint8), Q=b64(qq, np.uint8), I=b64(ib, np.uint8),
                C=b64(C, np.uint8), A=b64(A, np.uint8))


# ------------------------------------------------------------------ log and note
def read_log(run):
    p = os.path.join(run, 'log.csv')
    if not os.path.exists(p): return dict(iter=[], loss=[], psnr=[]), None
    rows = [r for r in csv.DictReader(open(p, newline='')) if r.get('iteration')]
    which = 'test' if any(r.get('psnr_test') for r in rows) else 'train'
    out = dict(iter=[], loss=[], psnr=[])
    for r in rows:
        if not r.get('psnr_' + which) or not r.get('loss'): continue
        out['iter'].append(int(r['iteration'])); out['loss'].append(float('%.6g' % float(r['loss']))); out['psnr'].append(round(float(r['psnr_' + which]), 3))
    return out, which

def note(run, which, a, nlog):
    if run:
        impl = '%s at %s%s%s' % (run.get('implementation', 'graphdeco-inria/gaussian-splatting'), (run.get('commit') or 'an unknown commit')[:7],
                                 '' if run.get('matches_pin') else ' (NOT the pinned 54c035f)', ' with local changes' if run.get('local_changes') else '')
        tv, te = run.get('train_views'), run.get('test_views') or []
        views = ('%s of the %s photographs, every 8th held out' % (tv, tv + len(te) if tv else '?') if te else 'all %s photographs' % (tv or '?'))
        s = ('plain 3D Gaussian Splatting (Kerbl et al. 2023): %s, its default settings%s, %s iterations on %s (%s), on the dataset\'s COLMAP model; '
             'trained on %s, %s' % (impl, (' plus ' + ' '.join(run['extra_args'])) if run.get('extra_args') else '', run.get('iterations', '?'), views,
                                     'alpha used as a loss mask' if run.get('keep_alpha') else 'RGB, alpha dropped', run.get('gpu', 'an unrecorded GPU'),
                                     (run.get('finished') or run.get('started') or 'an unrecorded date')[:10]))
        s += '' if run.get('finished') and not run.get('error') else ' (the run did NOT finish)'
    else:
        s = 'plain 3D Gaussian Splatting (Kerbl et al. 2023), presumably graphdeco-inria/gaussian-splatting: the run has no run.json, so implementation, version, settings, GPU and date are unknown'
    s += ('. Not the thesis method. Iteration 0 = the Gaussians made from the dataset\'s SfM points, before the first optimizer step; iteration k = '
          'graphdeco\'s save at k (after k - 1 steps). Each snapshot keeps at most %d Gaussians, drawn %s (seed %d)'
          % (a.budget, 'uniformly' if a.rule == 'uniform' else 'with probability growing with opacity x (s1 s2 + s2 s3 + s3 s1)', a.seed))
    if nlog: s += ('. log: loss = training loss (0.8 L1 + 0.2 D-SSIM) averaged over the iterations since the previous entry; psnr = mean PSNR of the '
                   + ('%d held-out photographs' % len(run['test_views']) if which == 'test' and run and run.get('test_views') else '%s photographs' % ('held-out' if which == 'test' else 'training')))
    else: s += '. No log was found'
    return s + '.'


HEAD = """/* DTU scan %(scan)d: snapshots of ONE run of PLAIN 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023), for the `train`
   scene of the talk's pipeline chapter. GENERATED by tools/train/export.py from a run of tools/train/train3dgs.py: do not
   edit, re-run them. NOT the thesis method: graphdeco-inria/gaussian-splatting's own training on the scan's photographs
   and the dataset's COLMAP model, mapped into the owl's frame of talk/dtu-data.js (+y up, the table at y = 0, 1 unit =
   80 mm) the way tools/dtu/pipeline.py maps that model.
   %(note)s
   KnotTalk.dtuTrain[%(scan)d] = { note, unit, log: { iter: [], loss: [], psnr: [] },
     snaps: [{ iter, ntotal (Gaussians in the checkpoint), n (kept here), p: Float32Array(3n) centres,
               s: Float32Array(3n) sigma per axis (owl units), q: Float32Array(4n) unit quaternions (w, x, y, z),
               c: Uint8Array(3n) colour of the DC term (sRGB), a: Uint8Array(n) opacity 0..255 }] }
   The covariance of Gaussian i is R(q_i) diag(s_i^2) R(q_i)^T. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {}, D = (KT.dtuTrain = KT.dtuTrain || {})[%(scan)d] = """

TAIL = """  function bytes(s) { var b = atob(s), n = b.length, a = new Uint8Array(n), i; for (i = 0; i < n; i++) a[i] = b.charCodeAt(i); return a; }
  D.snaps.forEach(function (z) {
    var n = z.n, m = z.near, lo = z.box[0], hi = z.box[1], s0 = z.slog[0], ds = (z.slog[1] - z.slog[0]) / 255, R = Math.SQRT1_2,
        b = bytes(z.P), f = bytes(z.F), fv = new DataView(f.buffer), e = bytes(z.S), r = bytes(z.Q), x = bytes(z.I),
        p = new Float32Array(3 * n), s = new Float32Array(3 * n), q = new Float32Array(4 * n), v = [0, 0, 0, 0], i, j, k, c, l, w;
    for (i = 0; i < 3 * m; i++) { k = i % 3; p[i] = lo[k] + (hi[k] - lo[k]) * (b[2 * i] | b[2 * i + 1] << 8) / 65535; }
    for (i = 3 * m; i < 3 * n; i++) p[i] = fv.getFloat32((i - 3 * m) * 4, true);
    for (i = 0; i < 3 * n; i++) s[i] = Math.exp(s0 + ds * e[i]);
    for (i = 0; i < n; i++) {                       // the three smallest components; the largest is the rest of the unit length
      k = x[i >> 2] >> ((i & 3) << 1) & 3; l = 0;
      for (j = 0, c = 0; j < 4; j++) if (j !== k) { w = (r[3 * i + c++] / 127.5 - 1) * R; v[j] = w; l += w * w; }
      v[k] = Math.sqrt(Math.max(0, 1 - l)); l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
      for (j = 0; j < 4; j++) q[4 * i + j] = v[j] / l;
    }
    z.p = p; z.s = s; z.q = q; z.c = bytes(z.C); z.a = bytes(z.A);
    ['near', 'box', 'slog', 'P', 'F', 'S', 'Q', 'I', 'C', 'A'].forEach(function (key) { delete z[key]; });
  });
})();
"""


def main():
    ap = argparse.ArgumentParser(description=' '.join(__doc__.split('\n\n')[0].split()), formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split('\n\n', 2)[2])
    ap.add_argument('run', help='the run folder (graphdeco\'s -m): point_cloud/iteration_*/point_cloud.ply, log.csv, run.json')
    ap.add_argument('-o', '--out', help='default: talk/dtu/train<scan>.js of this repo')
    ap.add_argument('--data', help='the scan folder (sparse/0/images.txt, points3D.txt): read the frame and the check from the dataset instead of the run')
    ap.add_argument('--pipeline', help='default: talk/dtu/pipeline122.js of this repo (the cameras and SfM points in the owl\'s frame)')
    ap.add_argument('--budget', type=int, default=60000, help='Gaussians kept per snapshot (default 60000)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rule', choices=['area', 'uniform'], default='area', help='how the kept ones are drawn (default: area)')
    ap.add_argument('--iters', help='only these iterations, comma separated (default: every snapshot in the run)')
    a = ap.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(a.run)); pj = os.path.join(run_dir, 'run.json')
    run = json.load(open(pj)) if os.path.exists(pj) else None
    if run is None: say('no run.json in %s: implementation, version, GPU and date will be unknown in the note' % run_dir)

    # ---- the frame
    scan, pipe = read_pipeline(a.pipeline or os.path.join(TALK, 'dtu', 'pipeline122.js')); U = pipe.get('unit_mm', 80)
    if a.data: cams, src = cameras_colmap(a.data), os.path.join(a.data, 'sparse', '0', 'images.txt')
    elif os.path.exists(os.path.join(run_dir, 'cameras.json')): cams, src = cameras_run(run_dir), os.path.join(run_dir, 'cameras.json')
    elif run and run.get('data'): cams, src = cameras_colmap(run['data']), os.path.join(run['data'], 'sparse', '0', 'images.txt')
    else: raise SystemExit('no cameras: the run has no cameras.json; pass --data <scan folder>')
    k, R, T, how = similarity(pipe, cams); qR = quat(R)
    say('frame from %d cameras (%s): k = %.9f, rotations agree within %.2e deg, centres within %.2e units (%.4f mm)'
        % (how['cameras'], src, k, how['rotation_deg'], how['centre_units'], how['centre_units'] * U))
    if how['rotation_deg'] > 0.01 or how['centre_units'] > 1e-3: raise SystemExit('the cameras do not agree with pipeline%d.js: not the same model, nothing written' % scan)
    pts = os.path.join(a.data, 'sparse', '0', 'points3D.txt') if a.data else os.path.join(run_dir, 'input.ply')
    if not os.path.exists(pts) and not a.data and run and run.get('data'): pts = os.path.join(run['data'], 'sparse', '0', 'points3D.txt')
    if os.path.exists(pts):
        X = sfm_points(pts); chk = sfm_check(pipe, X, k, R, T)
        say('check: %d SfM points (%s) mapped vs pipeline%d.js: %s; near %d / %d, worst %.4f of a 12-bit step (rounding alone: 0.5); far worst %.2e units'
            % (chk['points'], os.path.basename(pts), scan, 'one to one' if chk['ok'] else 'MISMATCH', chk['near'], chk['stored_near'],
               chk.get('near_steps', float('nan')), chk.get('far_units', float('nan'))))
        if not chk['ok']: raise SystemExit('the SfM points do not land where pipeline.py put them: nothing written')
    else:
        X = None; say('no %s: the frame is NOT checked against the SfM points' % pts)

    # ---- the snapshots
    pc = os.path.join(run_dir, 'point_cloud'); found = {}
    for d in (os.listdir(pc) if os.path.isdir(pc) else []):
        m = re.fullmatch(r'iteration_(\d+)', d)
        if m and os.path.exists(os.path.join(pc, d, 'point_cloud.ply')): found[int(m.group(1))] = os.path.join(pc, d, 'point_cloud.ply')
    its = sorted(found) if not a.iters else [int(v) for v in a.iters.split(',')]
    miss = [i for i in its if i not in found]
    if miss or not its: raise SystemExit('no snapshot for iteration %s in %s' % (miss or 'any', pc))
    snaps = []
    say('%8s %9s %7s %7s %6s %14s %8s %7s' % ('iter', 'ntotal', 'kept', 'far', 'bad', 'sigma mm', 'opacity', 'KB'))
    for it in its:
        G = read_gaussians(found[it]); N = len(G['x'])
        if it == 0 and X is not None:
            same = len(X) == N and np.array_equal(np.sort(X.astype(np.float32).view([('', np.float32)] * 3).ravel(), order=['f0', 'f1', 'f2']),
                                                  np.sort(G['x'].astype(np.float32).view([('', np.float32)] * 3).ravel(), order=['f0', 'f1', 'f2']))
            say('    iteration 0 holds %s' % ('exactly the %d SfM points' % N if same else 'NOT the SfM points of %s' % os.path.basename(pts)))
        ok, P, S, q, C, al = owl_frame(G, k, R, T, qR); good = np.nonzero(ok)[0]
        w = al[good] * (S[good, 0] * S[good, 1] + S[good, 1] * S[good, 2] + S[good, 2] * S[good, 0])
        idx = good[choose(w, a.budget, a.seed, it, a.rule)]
        z = encode(it, N, P[idx], S[idx], q[idx], C[idx], al[idx]); snaps.append(z)
        say('%8d %9d %7d %7d %6d %6.3f..%-7.2f %8.3f %7.0f' % (it, N, z['n'], z['n'] - z['near'], N - len(good), S[idx].min() * U, S[idx].max() * U,
                                                            al[idx].mean(), sum(len(z[c]) for c in 'PFSQICA') / 1e3))

    # ---- write
    lg, which = read_log(run_dir)
    if not lg['iter']: say('no log.csv rows in %s: the file carries an empty log' % run_dir)
    obj = dict(note=note(run, which, a, len(lg['iter'])), unit='owl frame, 1 unit = %g mm' % U, log=lg, snaps=snaps)
    wrap = lambda s, w=114: '\n   '.join(c.strip() for c in re.findall(r'.{1,%d}(?:\s|$)' % w, s.replace('*/', '* /')) if c.strip())
    js = HEAD % dict(scan=scan, note=wrap(obj['note'])) + json.dumps(obj, separators=(',', ':')) + ';\n' + TAIL
    out = os.path.abspath(a.out or os.path.join(TALK, 'dtu', 'train%d.js' % scan))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + '.part', 'w') as fh: fh.write(js)
    os.replace(out + '.part', out)
    say('wrote %s: %.2f MB, %d snapshots, %d Gaussians, log of %d rows (psnr: %s)'
        % (out, len(js) / 1e6, len(snaps), sum(z['n'] for z in snaps), len(lg['iter']), which or '-'))
    if len(js) > 8.5e6: say('note: more than 8 MB; --budget lowers it (about 21.7 bytes per kept Gaussian)')


if __name__ == '__main__':
    main()
