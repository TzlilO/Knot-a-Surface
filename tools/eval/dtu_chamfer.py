#!/usr/bin/env python3
"""DTU Chamfer distance of a mesh (or point cloud), by the standard DTU evaluation protocol.

    python3 -I tools/eval/dtu_chamfer.py MESH.ply --scan 122 --data REF_DIR [--out result.json]
    python3 -I tools/eval/dtu_chamfer.py --self-test                       # synthetic data, no download
    python3 -I tools/eval/dtu_chamfer.py --self-test --data REF_DIR --scan 1   # the same, plus checks on a real reference

MESH.ply must be in the DTU world frame, in millimetres (the frame of the dataset's calibration, `pos_NNN.txt`, and
of the IDR-style `cameras.npz`: `world_mat_i`). REF_DIR holds, for scan N (searched in the layouts listed in FIND):

    ObsMask/ObsMask{N}_10.mat       observation mask: ObsMask (bool volume), BB (2 x 3, mm), Res (mm per voxel)
    ObsMask/Plane{N}.mat            ground plane P (4 numbers): a reference point counts if P . [x y z 1] > 0
    Points/stl/stl{N:03d}_total.ply the structured-light reference cloud

The protocol is that of the DTU MVS benchmark: R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, H. Aanaes, "Large Scale
Multi-view Stereopsis Evaluation", CVPR 2014, as implemented by the MATLAB code distributed with the dataset
(SampleSet.zip: BaseEvalMain_web.m, PointCompareMain.m, reducePts_haa.m, MaxDistCP.m, ComputeStat_web.m) and by
its widely used Python port, DTUeval-python (github.com/jzhangbs/DTUeval-python, eval.py), which this file
follows step by step. It is written from those two sources in numpy + scipy only (no open3d, no scikit-learn):

  1. sample   the mesh to points: every vertex, plus a lattice on every triangle with spacing --density (0.2 mm)
  2. reduce   shuffle (seeded), then walk the points: one that is still kept removes every other point within
              --density of it, so no two kept points are closer than 0.2 mm
  3. mask     keep points inside the reference bounding box grown by --patch (60 mm below, 120 mm above, as the
              port does) = `data_in`; of those, the ones whose voxel of ObsMask is set = `data_in_obs`
  4. accuracy       for every point of data_in_obs, the distance to the nearest reference point;
                    mean of the distances below --max-dist (20 mm)                                  "d2s"
  5. completeness   for every reference point above the ground plane, the distance to the nearest point of
                    data_in; mean of the distances below --max-dist                                 "s2d"
  6. overall  (accuracy + completeness) / 2, in mm: the number papers print as "Chamfer distance" on DTU

Differences from the MATLAB code, none of which changes a distance below 20 mm: MATLAB searches block by block
with a 60 mm cap instead of cropping to the box; it rounds half away from zero where numpy rounds half to even
(a point exactly on a voxel face); its mesh sampler is a compiled mex without source, the port's lattice is used
here. The shuffle is random in both; here it is seeded (--seed) so that a run can be repeated exactly.

A long numerical job by this repository's rule: it prints its progress, has a wall-clock budget (--max-minutes)
and a memory guard (--max-points), refuses non-finite input, handles SIGINT / SIGTERM, and always writes the JSON
with the reason it stopped (`stop.reason`: completed, or what cut it short; then there is no result in it).

Exit status: 0 completed, 2 stopped early or bad input, 1 a self-test failed.
"""
import argparse, hashlib, json, os, platform, signal, struct, sys, tempfile, time

import numpy as np
import scipy
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree

VERSION = '1.0'
PROTOCOL = ('DTU MVS evaluation (Jensen et al., CVPR 2014; MATLAB code in the dataset\'s SampleSet.zip), '
            'as implemented by DTUeval-python (github.com/jzhangbs/DTUeval-python, eval.py)')
OFFICIAL = ('the DTU MVS data set, roboimagedata.compute.dtu.dk -> "MVS Data Set - 2014": SampleSet.zip (6.3 GB; '
            'ObsMask/ObsMask{N}_10.mat and ObsMask/Plane{N}.mat for every scan) and Points.zip (6.3 GB; '
            'stl/stl{N:03d}_total.ply for every scan)')
# where the three files of a scan may sit under --data: DTUeval-python's layout, the dataset's own, or flat
FIND = dict(obsmask=['ObsMask', 'SampleSet/MVS Data/ObsMask', 'SampleSet/MVS_Data/ObsMask', 'MVS Data/ObsMask', 'MVS_Data/ObsMask', 'SampleSet/mvs_data/ObsMask', ''],
            stl=['Points/stl', 'SampleSet/MVS Data/Points/stl', 'SampleSet/MVS_Data/Points/stl', 'MVS Data/Points/stl', 'MVS_Data/Points/stl', 'stl', ''])
T0 = time.time()


def log(*a):
    print('[%6.1fs]' % (time.time() - T0), *a, flush=True)


class Stop(Exception):
    """The job judged that it has to end here; `reason` goes into the JSON."""
    def __init__(self, reason, detail=''):
        super().__init__(reason + (': ' + detail if detail else '')); self.reason = reason; self.detail = detail


class Guard:
    """Wall clock and signals. check() is called between stages and between chunks of every long loop."""
    def __init__(self, max_minutes):
        self.deadline = T0 + max_minutes * 60 if max_minutes and max_minutes > 0 else None; self.max_minutes = max_minutes; self.signal = None
        for s in (signal.SIGINT, signal.SIGTERM):
            try: signal.signal(s, self._on)
            except (ValueError, OSError): pass                       # not the main thread
    def _on(self, num, frame): self.signal = signal.Signals(num).name
    def check(self, where=''):
        if self.signal: raise Stop('signal', '%s received during %s' % (self.signal, where))
        if self.deadline and time.time() > self.deadline: raise Stop('max_minutes', 'the %.1f minute budget ran out during %s' % (self.max_minutes, where))


# ------------------------------------------------------------------ PLY (plain reader and writer)
PLY_T = dict(char='i1', uchar='u1', short='i2', ushort='u2', int='i4', uint='u4', float='f4', double='f8',
             int8='i1', uint8='u1', int16='i2', uint16='u2', int32='i4', uint32='u4', float32='f4', float64='f8')


def read_ply(path, faces=True):
    """-> (vertices float64 (n, 3), triangles int64 (m, 3) or None). ASCII or binary, either byte order, CRLF or LF
    headers, any extra properties; polygons are fanned into triangles."""
    with open(path, 'rb') as f:
        if f.readline().strip() != b'ply': raise Stop('bad_input', '%s is not a PLY file' % path)
        fmt, elems, n = None, [], 0
        while True:
            line = f.readline(); n += 1
            if not line or n > 100000: raise Stop('bad_input', '%s: no end_header' % path)
            t = line.decode('latin1').split()
            if not t or t[0] in ('comment', 'obj_info'): continue
            if t[0] == 'format': fmt = t[1]
            elif t[0] == 'element': elems.append(dict(name=t[1], n=int(t[2]), props=[]))
            elif t[0] == 'property':
                if t[1] == 'list': elems[-1]['props'].append((t[4], PLY_T[t[2]], PLY_T[t[3]]))
                else: elems[-1]['props'].append((t[2], PLY_T[t[1]], None))
            elif t[0] == 'end_header': break
        if fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'): raise Stop('bad_input', '%s: PLY format %r' % (path, fmt))
        bo = '>' if fmt == 'binary_big_endian' else '<'; V = None; F = None
        for e in elems:
            has_list = any(p[2] for p in e['props']); want = e['name'] == 'vertex' or (e['name'] == 'face' and faces)
            if fmt == 'ascii':
                rows = [f.readline().split() for _ in range(e['n'])]
                if e['name'] == 'vertex':
                    k = [p[0] for p in e['props']]; ix = [k.index(c) for c in 'xyz']
                    V = np.array([[float(r[i]) for i in ix] for r in rows], float).reshape(-1, 3)
                elif e['name'] == 'face' and faces:
                    if [bool(p[2]) for p in e['props']].index(True) != 0: raise Stop('bad_input', '%s: ASCII faces must start with the index list' % path)
                    F = fan([[int(x) for x in r[1:1 + int(r[0])]] for r in rows])
                continue
            if not has_list:
                dt = np.dtype([(p[0], bo + p[1]) for p in e['props']]); a = np.fromfile(f, dt, e['n'])
                if len(a) != e['n']: raise Stop('bad_input', '%s: truncated (%s)' % (path, e['name']))
                if e['name'] == 'vertex': V = np.stack([a['x'], a['y'], a['z']], -1).astype(float)
                continue
            # an element with a list: fast when every list has the length of the first one, else row by row
            pos = f.tell(); cnt = np.dtype(bo + [p[1] for p in e['props'] if p[2]][0]); done = False
            if e['n'] and len([p for p in e['props'] if p[2]]) == 1:
                li = [bool(p[2]) for p in e['props']].index(True); f.seek(pos + sum(np.dtype(p[1]).itemsize for p in e['props'][:li]))
                k = int(np.frombuffer(f.read(cnt.itemsize), cnt)[0]); f.seek(pos)
                dt = np.dtype([(p[0], bo + p[1]) if not p[2] else (p[0], [('n', bo + p[1]), ('i', bo + p[2], (k,))]) for p in e['props']])
                a = np.fromfile(f, dt, e['n']); name = e['props'][li][0]
                if len(a) == e['n'] and (a[name]['n'] == k).all():
                    done = True
                    if want and e['name'] == 'face': F = fan(a[name]['i'].astype(np.int64))
                else: f.seek(pos)
            if not done:
                out = []
                for _ in range(e['n']):
                    for p in e['props']:
                        if p[2]:
                            k = int(np.frombuffer(f.read(np.dtype(p[1]).itemsize), bo + p[1])[0]); idx = np.frombuffer(f.read(k * np.dtype(p[2]).itemsize), bo + p[2])
                            if want and e['name'] == 'face': out.append(idx.astype(np.int64))
                        else: f.read(np.dtype(p[1]).itemsize)
                if want and e['name'] == 'face': F = fan(out)
    if V is None: raise Stop('bad_input', '%s has no vertex element' % path)
    return V, F


def fan(polys):
    """Polygons -> triangles (v0, vi, vi+1)."""
    if isinstance(polys, np.ndarray):
        k = polys.shape[1]
        return polys.reshape(-1, 3) if k == 3 else np.concatenate([polys[:, [0, i, i + 1]] for i in range(1, k - 1)]) if k > 3 else np.zeros((0, 3), np.int64)
    out = [[p[0], p[i], p[i + 1]] for p in polys for i in range(1, len(p) - 1)]
    return np.array(out, np.int64).reshape(-1, 3)


def write_ply(path, V, F=None, comments=()):
    """Binary little-endian PLY: float x y z, and triangles as uchar 3 + int indices."""
    V = np.asarray(V, '<f4'); n = 0 if F is None else len(F)
    head = 'ply\nformat binary_little_endian 1.0\n' + ''.join('comment %s\n' % c for c in comments) + 'element vertex %d\nproperty float x\nproperty float y\nproperty float z\n' % len(V)
    if F is not None: head += 'element face %d\nproperty list uchar int vertex_indices\n' % n
    with open(path, 'wb') as f:
        f.write((head + 'end_header\n').encode('ascii')); f.write(V.tobytes())
        if F is not None:
            a = np.zeros(n, [('n', 'u1'), ('i', '<i4', (3,))]); a['n'] = 3; a['i'] = F; f.write(a.tobytes())


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ 1. mesh -> points
def sample_mesh(V, F, density, guard, max_points):
    """DTUeval-python's sampler: all vertices, and on every triangle with edges v1, v2 from its first corner the points
    corner + a v1 + b v2 with a = (i + 0.5) / n1, b = (j + 0.5) / n2, a + b < 1, where n1 = floor(|v1| / thr),
    n2 = floor(|v2| / thr) and thr = density * sqrt(|v1| |v2| / |v1 x v2|). A triangle with n1 = 0 or n2 = 0 adds none."""
    tri = V[F]; v1 = tri[:, 1] - tri[:, 0]; v2 = tri[:, 2] - tri[:, 0]
    l1 = np.linalg.norm(v1, axis=1); l2 = np.linalg.norm(v2, axis=1); area2 = np.linalg.norm(np.cross(v1, v2), axis=1)
    ok = area2 > 0; tri0, v1, v2, l1, l2, area2 = tri[ok, 0], v1[ok], v2[ok], l1[ok], l2[ok], area2[ok]
    thr = density * np.sqrt(l1 * l2 / area2); n1 = np.floor(l1 / thr).astype(np.int64); n2 = np.floor(l2 / thr).astype(np.int64)
    keep = (n1 > 0) & (n2 > 0); tri0, v1, v2, n1, n2 = tri0[keep], v1[keep], v2[keep], n1[keep], n2[keep]
    big = int(n2.max()) + 1 if len(n2) else 1; key = n1 * big + n2; order = np.argsort(key, kind='stable'); uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    pats = []; total = len(V)
    for k in uk:
        a, b = int(k // big), int(k % big); c = np.mgrid[:a + 1, :b + 1].astype(float) + 0.5; c[0] /= a; c[1] /= b
        c = c.transpose(1, 2, 0); pats.append(c[c.sum(-1) < 1])
    total += int(sum(len(p) * int(n) for p, n in zip(pats, cnt)))
    if total > max_points: raise Stop('max_points', 'sampling this mesh at %.3g mm gives %d points, more than --max-points %d' % (density, total, max_points))
    out = np.empty((total, 3)); out[:len(V)] = V; o = len(V)
    for g, (p, s, n) in enumerate(zip(pats, start, cnt)):
        if not len(p): continue
        idx = order[s:s + n]
        for a in range(0, len(idx), max(1, 4000000 // len(p))):
            i = idx[a:a + max(1, 4000000 // len(p))]
            q = tri0[i][:, None, :] + v1[i][:, None, :] * p[None, :, :1] + v2[i][:, None, :] * p[None, :, 1:]
            out[o:o + q.shape[0] * q.shape[1]] = q.reshape(-1, 3); o += q.shape[0] * q.shape[1]
        if g % 64 == 0: guard.check('sampling the mesh')
    assert o == total
    return out, dict(triangles=int(len(F)), degenerate_triangles=int((~ok).sum()), area_mm2=float(area2.sum() / 2), sampled_points=int(total))


# ------------------------------------------------------------------ 2. reduce to one point per `density`
def reduce_points(P, density, seed, guard, workers, chunk=200000):
    """reducePts_haa.m / the port's loop: in a random order, a point that is still kept removes every other point
    within `density` of it (distance <= density, as scikit-learn's radius query). Points already removed need no
    query, which changes nothing in the outcome."""
    rng = np.random.default_rng(seed); P = P[rng.permutation(len(P))]; tree = cKDTree(P); keep = np.ones(len(P), bool); t = time.time()
    for a in range(0, len(P), chunk):
        guard.check('reducing the points')
        live = a + np.flatnonzero(keep[a:a + chunk])
        if not len(live): continue
        nb = tree.query_ball_point(P[live], density, workers=workers, return_sorted=False)
        for i, idx in zip(live.tolist(), nb):
            if keep[i]: keep[idx] = False; keep[i] = True
        if time.time() - t > 10: log('    %d of %d points walked, %d kept so far' % (min(a + chunk, len(P)), len(P), int(keep[:a + chunk].sum()))); t = time.time()
    return P[keep]


def nearest(tree, Q, guard, workers, what, chunk=500000):
    out = np.empty(len(Q))
    for a in range(0, len(Q), chunk):
        guard.check(what); out[a:a + chunk] = tree.query(Q[a:a + chunk], workers=workers)[0]
    return out


# ------------------------------------------------------------------ reference data
def find(data, kind, names):
    tried = []
    for d in FIND[kind]:
        for n in names:
            p = os.path.join(data, d, n); tried.append(p)
            if os.path.isfile(p): return p
    return tried


def reference(data, scan, stl=True):
    """-> paths of the three files (two with stl=False), or Stop('missing_reference') naming exactly what is not there."""
    want = dict(obsmask=find(data, 'obsmask', ['ObsMask%d_10.mat' % scan]), plane=find(data, 'obsmask', ['Plane%d.mat' % scan]),
                stl=find(data, 'stl', ['stl%03d_total.ply' % scan]))
    miss = {k: v for k, v in want.items() if not isinstance(v, str) and (stl or k != 'stl')}
    if miss:
        lines = ['%s: not found (looked for %s under %s in: %s)' % (k, os.path.basename(v[0]), data, ', '.join(repr(d) for d in FIND['stl' if k == 'stl' else 'obsmask'])) for k, v in miss.items()]
        raise Stop('missing_reference', '; '.join(lines) + '. They come from ' + OFFICIAL.replace('{N}', str(scan)).replace('{N:03d}', '%03d' % scan))
    return want


def stats(d, cap):
    """What the protocol reports (mean of the distances below the cap) and what helps to read it."""
    u = d[d < cap]
    if not len(u): return dict(mean_mm=None, used=0, of=int(len(d)))
    return dict(mean_mm=float(u.mean()), median_mm=float(np.median(u)), rms_mm=float(np.sqrt((u ** 2).mean())), p95_mm=float(np.percentile(u, 95)),
                max_mm=float(u.max()), used=int(len(u)), of=int(len(d)), at_or_over_cap=int(len(d) - len(u)),
                within_mm={('%g' % t): float((u < t).mean()) for t in (0.2, 0.5, 1, 2, 5)})


# ------------------------------------------------------------------ the evaluation
def evaluate(mesh, scan, data, mode='mesh', density=0.2, patch=60.0, max_dist=20.0, seed=0, max_minutes=30.0, max_points=120000000, workers=-1, note=None, quiet=False, mask_only=False):
    """Runs the protocol; returns the result as a dict (also when it stopped early: then `result` is absent)."""
    global log
    keep_log = log
    if quiet: log = lambda *a: None
    guard = Guard(max_minutes); times = {}; began = time.time(); R = dict(
        tool='tools/eval/dtu_chamfer.py', version=VERSION, protocol=PROTOCOL, note=note,
        parameters=dict(scan=scan, mode=mode, downsample_density_mm=density, patch_size_mm=patch, max_dist_mm=max_dist, obsmask_margin_mm=10, seed=seed, max_minutes=max_minutes, max_points=max_points,
                        box='points with BB[0] - patch <= p < BB[1] + 2 * patch are kept (DTUeval-python, eval.py)', voxel='round((p - BB[0]) / Res), half to even'),
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(), cpus=os.cpu_count()),
        started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    def stage(name, t): times[name] = round(time.time() - t, 2)
    try:
        ref = reference(data, scan, stl=not mask_only)
        t = time.time(); log('reading %s' % mesh)
        V, F = read_ply(mesh, faces=(mode == 'mesh')); R['input'] = dict(path=os.path.abspath(mesh), bytes=os.path.getsize(mesh), sha256=sha256(mesh), vertices=int(len(V)))
        if not len(V): raise Stop('bad_input', 'no vertices')
        if not np.isfinite(V).all(): raise Stop('non_finite_input', '%d vertices are NaN or Inf' % int((~np.isfinite(V).all(1)).sum()))
        R['input']['bbox_mm'] = [np.round(V.min(0), 3).tolist(), np.round(V.max(0), 3).tolist()]
        stage('read_mesh', t); guard.check('reading the mesh'); counts = R['counts'] = {}
        if mode == 'mesh':
            if F is None or not len(F): raise Stop('bad_input', 'the PLY has no faces: use --mode pcd for a point cloud')
            if F.min() < 0 or F.max() >= len(V): raise Stop('bad_input', 'face indices out of range')
            t = time.time(); log('1. sampling %d triangles at %.3g mm' % (len(F), density))
            P, info = sample_mesh(V, F, density, guard, max_points); R['input'].update(info); stage('sample', t)
            log('    %d points (%d vertices + %d on the triangles), surface area %.0f mm2' % (len(P), len(V), len(P) - len(V), info['area_mm2']))
        else: P = V
        counts['data_points'] = int(len(P))
        t = time.time(); log('2. reducing to one point per %.3g mm (seed %d)' % (density, seed))
        P = reduce_points(P, density, seed, guard, workers); counts['data_reduced'] = int(len(P)); stage('reduce', t); log('    %d points kept' % len(P))
        t = time.time(); log('3. observation mask %s' % ref['obsmask'])
        m = loadmat(ref['obsmask']); mask, BB, res = m['ObsMask'], m['BB'].astype(np.float32), float(np.asarray(m['Res']).ravel()[0])
        inb = ((P >= BB[:1] - patch) & (P < BB[1:] + patch * 2)).sum(-1) == 3; Pin = P[inb]
        g = np.around((Pin - BB[:1]) / res).astype(np.int32); gin = ((g >= 0) & (g < np.array(mask.shape)[None])).sum(-1) == 3
        obs = np.zeros(len(Pin), bool); gg = g[gin]; obs[np.flatnonzero(gin)] = mask[gg[:, 0], gg[:, 1], gg[:, 2]].astype(bool); Pobs = Pin[obs]
        counts.update(data_in_box=int(len(Pin)), data_in_mask=int(len(Pobs)))
        plane = loadmat(ref['plane'])['P'].astype(float).ravel()
        R['reference'] = dict(dir=os.path.abspath(data),
                              obsmask=dict(path=os.path.abspath(ref['obsmask']), bytes=os.path.getsize(ref['obsmask']), sha256=sha256(ref['obsmask']), shape=list(mask.shape), set=int(mask.astype(bool).sum()), BB=BB.tolist(), Res=res),
                              plane=dict(path=os.path.abspath(ref['plane']), bytes=os.path.getsize(ref['plane']), sha256=sha256(ref['plane']), P=plane.tolist()))
        stage('mask', t); log('    %d in the box, %d of them in the mask' % (len(Pin), len(Pobs)))
        if not len(Pobs): raise Stop('no_points_in_mask', 'no point of the input lies in the observation mask: is the mesh in the DTU world frame, in mm?')
        h = (np.c_[Pobs, np.ones(len(Pobs))] @ plane) / np.linalg.norm(plane[:3]); counts['data_in_mask_above_plane'] = int((h > 0).sum())      # not part of the protocol: for reading the result
        if mask_only: raise Stop('mask_only', 'steps 1 to 3 only, as asked: no distance was measured and there is no result')
        t = time.time(); log('4. reference cloud %s' % ref['stl']); guard.check('reading the reference')
        S, _ = read_ply(ref['stl'], faces=False)
        if not np.isfinite(S).all(): raise Stop('non_finite_input', 'the reference cloud holds NaN or Inf')
        above = (np.c_[S, np.ones(len(S))] @ plane) > 0
        R['reference']['stl'] = dict(path=os.path.abspath(ref['stl']), bytes=os.path.getsize(ref['stl']), sha256=sha256(ref['stl']), points=int(len(S)), above_plane=int(above.sum()))
        stage('read_reference', t); log('    %d points, %d above the ground plane' % (len(S), int(above.sum())))
        t = time.time(); log('5. accuracy: %d points of the input -> nearest of %d reference points' % (len(Pobs), len(S)))
        tree = cKDTree(S); d2s = nearest(tree, Pobs, guard, workers, 'the accuracy search'); del tree; stage('accuracy', t)
        t = time.time(); log('6. completeness: %d reference points above the plane -> nearest of %d input points' % (int(above.sum()), len(Pin)))
        tree = cKDTree(Pin); s2d = nearest(tree, S[above], guard, workers, 'the completeness search'); del tree; stage('completeness', t)
        if not (np.isfinite(d2s).all() and np.isfinite(s2d).all()): raise Stop('non_finite_result', 'a distance is NaN or Inf')
        acc, comp = stats(d2s, max_dist), stats(s2d, max_dist)
        if acc['mean_mm'] is None or comp['mean_mm'] is None: raise Stop('no_distance_below_cap', 'no distance below %g mm: is the mesh in the DTU world frame, in mm?' % max_dist)
        R['result'] = dict(accuracy_mm=acc['mean_mm'], completeness_mm=comp['mean_mm'], overall_mm=(acc['mean_mm'] + comp['mean_mm']) / 2,
                           accuracy=dict(acc, what='input -> reference, points of the input inside the observation mask'),
                           completeness=dict(comp, what='reference -> input, reference points above the ground plane'))
        R['stop'] = dict(reason='completed', detail='')
        log('accuracy %.4f mm (%d points)   completeness %.4f mm (%d points)   overall %.4f mm' % (acc['mean_mm'], acc['used'], comp['mean_mm'], comp['used'], R['result']['overall_mm']))
    except Stop as e:
        R['stop'] = dict(reason=e.reason, detail=e.detail); log('STOPPED: %s' % e)
    except MemoryError:
        R['stop'] = dict(reason='out_of_memory', detail='lower --max-points or evaluate a coarser mesh'); log('STOPPED: out of memory')
    finally:
        log = keep_log
    R['timing_s'] = times; R['seconds'] = round(time.time() - began, 2); R['stop']['minutes'] = round((time.time() - began) / 60, 3)
    return R


# ------------------------------------------------------------------ self-tests
def synthetic(d, scan=900, size=60.0, step=0.2, margin=10.0):
    """A reference made up for the tests, in the dataset's file formats: a 60 x 60 mm plate at z = 500 with a smooth
    bump, sampled every 0.2 mm, plus a strip of 'table' 5 mm lower that the ground plane cuts away.
    ObsMask = every voxel within `margin` of the plate (like the dataset's 10 mm margin); Plane = z > 497.5."""
    os.makedirs(os.path.join(d, 'ObsMask'), exist_ok=True); os.makedirs(os.path.join(d, 'Points', 'stl'), exist_ok=True)
    def surf(x, y): return 500.0 + 6.0 * np.exp(-((x - size / 2) ** 2 + (y - size / 2) ** 2) / 200.0)
    x, y = np.meshgrid(np.arange(0, size + 1e-9, step), np.arange(0, size + 1e-9, step)); plate = np.c_[x.ravel(), y.ravel(), surf(x, y).ravel()]
    tx, ty = np.meshgrid(np.arange(-20, 0, step), np.arange(0, size + 1e-9, step)); table = np.c_[tx.ravel(), ty.ravel(), np.full(tx.size, 495.0)]
    S = np.vstack([plate, table]).astype('<f4')
    with open(os.path.join(d, 'Points', 'stl', 'stl%03d_total.ply' % scan), 'wb') as f:      # CRLF header, normals and colours, as the dataset writes them
        f.write(('ply\r\nformat binary_little_endian 1.0\r\nelement vertex %d\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\nproperty float nx\r\nproperty float ny\r\nproperty float nz\r\n'
                 'property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\nend_header\r\n' % len(S)).encode())
        a = np.zeros(len(S), [('p', '<f4', 3), ('n', '<f4', 3), ('c', 'u1', 3)]); a['p'] = S; a['n'][:, 2] = 1; f.write(a.tobytes())
    BB = np.array([[-30, -10, 480], [70, 70, 520]], np.int16); shape = tuple((BB[1] - BB[0] + 1).tolist()); mask = np.zeros(shape, np.uint8)
    gx, gy = np.meshgrid(np.arange(shape[0]) + BB[0, 0], np.arange(shape[1]) + BB[0, 1], indexing='ij'); gz = np.arange(shape[2]) + BB[0, 2]
    inside = (gx >= -margin) & (gx <= size + margin) & (gy >= -margin) & (gy <= size + margin); h = surf(np.clip(gx, 0, size), np.clip(gy, 0, size))
    mask[inside[:, :, None] & (np.abs(gz[None, None, :] - h[:, :, None]) <= margin)] = 1
    savemat(os.path.join(d, 'ObsMask', 'ObsMask%d_10.mat' % scan), dict(ObsMask=mask, BB=BB, Res=np.uint8(1), Margin=np.uint8(10), cSet=np.uint16(scan)))
    savemat(os.path.join(d, 'ObsMask', 'Plane%d.mat' % scan), dict(P=np.array([[0.0], [0.0], [1.0], [-497.5]])))
    return surf, len(plate), len(table)


def grid_mesh(surf, x0, x1, y0, y1, h, dz=0.0, quads=False):
    xs = np.arange(x0, x1 + 1e-9, h); ys = np.arange(y0, y1 + 1e-9, h); x, y = np.meshgrid(xs, ys); V = np.c_[x.ravel(), y.ravel(), surf(x, y).ravel() + dz]
    i = (np.arange(len(ys) - 1)[:, None] * len(xs) + np.arange(len(xs) - 1)[None]).ravel(); nx = len(xs)
    if quads: return V, np.c_[i, i + 1, i + 1 + nx, i + nx]
    return V, np.vstack([np.c_[i, i + 1, i + 1 + nx], np.c_[i, i + 1 + nx, i + nx]])


def self_test(data=None, scan=None, max_minutes=30.0, workers=-1):
    fails = []; rows = []
    def check(name, got, want, tol, unit='mm'):
        ok = got is not None and abs(got - want) <= tol; rows.append(dict(test=name, got=got, expected=want, tolerance=tol, ok=bool(ok)))
        print('  %s  %-74s got %s, expected %s +- %s %s' % ('ok  ' if ok else 'FAIL', name, 'None' if got is None else '%.5f' % got, '%g' % want, '%g' % tol, unit), flush=True)
        if not ok: fails.append(name)
    def run(mesh, sc, dd, **kw):
        R = evaluate(mesh, sc, dd, max_minutes=max_minutes, workers=workers, quiet=True, **kw)
        return R.get('result') or dict(accuracy_mm=None, completeness_mm=None, overall_mm=None, stop=R['stop']), R
    with tempfile.TemporaryDirectory(prefix='dtu-chamfer-selftest-') as d:
        print('synthetic reference (made up for the tests: a 60 mm plate with a bump, in the dataset\'s file formats)', flush=True)
        surf, nplate, ntable = synthetic(d); flat = lambda x, y: np.full(np.shape(x), 500.0) + 0 * x
        # a. PLY reader: binary triangles, binary quads, ASCII, big-endian, all the same mesh
        V, F = grid_mesh(surf, 0, 60, 0, 60, 1.0); p = os.path.join(d, 'a.ply'); write_ply(p, V, F); V2, F2 = read_ply(p)
        check('PLY binary: vertices and faces read back', float(np.abs(V2 - V.astype('f4')).max() + np.abs(F2 - F).max()), 0, 1e-9, '')
        Vq, Fq = grid_mesh(surf, 0, 60, 0, 60, 1.0, quads=True); p = os.path.join(d, 'q.ply')
        with open(p, 'w') as f:
            f.write('ply\nformat ascii 1.0\ncomment a quad mesh\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nelement face %d\nproperty list uchar int vertex_index\nend_header\n' % (len(Vq), len(Fq)))
            f.write(''.join('%r %r %r\n' % tuple(v) for v in Vq.tolist())); f.write(''.join('4 %d %d %d %d\n' % tuple(q) for q in Fq.tolist()))
        V3, F3 = read_ply(p); check('PLY ASCII quads: fanned into twice as many triangles', float(len(F3)), 2 * len(Fq), 0, 'triangles')
        p = os.path.join(d, 'be.ply')
        with open(p, 'wb') as f:
            f.write(('ply\r\nformat binary_big_endian 1.0\r\nelement vertex %d\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\nproperty uchar q\r\nelement face %d\r\nproperty uchar flag\r\nproperty list uchar uint vertex_indices\r\nend_header\r\n' % (len(V), len(F))).encode())
            a = np.zeros(len(V), [('p', '>f4', 3), ('q', 'u1')]); a['p'] = V; f.write(a.tobytes())
            a = np.zeros(len(F), [('g', 'u1'), ('n', 'u1'), ('i', '>u4', 3)]); a['n'] = 3; a['i'] = F; f.write(a.tobytes())
        V4, F4 = read_ply(p); check('PLY big-endian, CRLF header, extra properties', float(np.abs(V4 - V.astype('f4')).max() + np.abs(F4 - F).max()), 0, 1e-9, '')
        # b. the sampler: area per point = density^2 on big triangles; nothing further than the lattice allows from a sample
        g = Guard(0); Vb = np.array([[0, 0, 0], [40, 0, 0], [0, 30, 0.0]]); Pb, info = sample_mesh(Vb, np.array([[0, 1, 2]]), 0.2, g, 10 ** 8)
        check('sampler: one 40 x 30 mm triangle gives area / 0.2^2 * (|v1||v2| / 2A) points', float(len(Pb) - 3), 600 / 0.04, 600 / 0.04 * 0.01, 'points')
        rng = np.random.default_rng(1); w = rng.random((20000, 2)); w[w.sum(1) > 1] = 1 - w[w.sum(1) > 1]; Q = w[:, :1] * Vb[1] + w[:, 1:] * Vb[2]
        check('sampler: largest gap from a random point of the triangle to a sample', float(cKDTree(Pb).query(Q)[0].max()), 0.15, 0.1)
        Pr = reduce_points(Pb, 0.2, 0, g, workers); dd = cKDTree(Pr).query(Pr, k=2)[0][:, 1]
        check('reduce: smallest distance between two kept points', float(dd.min()), 0.2, 0.2 * 0.5)
        check('reduce: every removed point has a kept one within 0.2 mm', float(cKDTree(Pr).query(Pb)[0].max()), 0.1, 0.1 + 1e-9)
        # c. the surface the reference was sampled from, as a mesh. "About zero" has a floor in this protocol: the reference
        #    is a 0.2 mm lattice and the input is reduced to points at least 0.2 mm apart, so neither side hits the other
        V, F = grid_mesh(surf, 0, 60, 0, 60, 0.5); p = os.path.join(d, 'same.ply'); write_ply(p, V, F); r, R0 = run(p, 900, d)
        check('same surface as a 0.5 mm mesh: accuracy (floor of the 0.2 mm sampling: below 0.15)', r['accuracy_mm'], 0.075, 0.075)
        check('same surface as a 0.5 mm mesh: completeness (floor of the 0.2 mm reduction: below 0.2)', r['completeness_mm'], 0.1, 0.1)
        check('ground plane: the table strip is left out of completeness (points used)', float(R0['result']['completeness']['of']) if 'result' in R0 else None, nplate, 0, 'points')
        # d. the reference cloud itself as a point cloud: exactly zero accuracy
        r, _ = run(os.path.join(d, 'Points', 'stl', 'stl900_total.ply'), 900, d, mode='pcd')
        check('the reference cloud as input (pcd): accuracy', r['accuracy_mm'], 0, 1e-9); check('the reference cloud as input (pcd): completeness (at most the 0.2 mm of the reduction)', r['completeness_mm'], 0.1, 0.1)
        # e. a known rigid offset d along the normal of an exactly flat reference: both ways between d and sqrt(d^2 + 0.2^2)
        d2 = os.path.join(d, 'flat'); synthetic_flat(d2)
        for off in (0.5, 2.0, 7.0):
            V, F = grid_mesh(flat, 0, 60, 0, 60, 1.0, dz=off); p = os.path.join(d, 'off.ply'); write_ply(p, V, F); r, _ = run(p, 901, d2); hi = float(np.hypot(off, 0.2))
            check('flat plate moved rigidly by %g mm along its normal: accuracy (%g .. %.4f)' % (off, off, hi), r['accuracy_mm'], (off + hi) / 2, (hi - off) / 2 + 1e-4)
            check('flat plate moved rigidly by %g mm along its normal: completeness (%g .. %.4f)' % (off, off, hi), r['completeness_mm'], (off + hi) / 2, (hi - off) / 2 + 1e-4)
        V, F = grid_mesh(flat, 0, 60, 0, 60, 1.0); p = os.path.join(d, 'base.ply'); write_ply(p, V, F); r0, _ = run(p, 901, d2); a0 = r0['accuracy_mm']
        check('flat plate in place: accuracy (the sampling floor again)', a0, 0.075, 0.075)
        p = os.path.join(d, 'slide.ply'); write_ply(p, V + [3.0, 0, 0], F); r, _ = run(p, 901, d2)
        check('flat plate slid by 3 mm in its own plane: accuracy = 57 mm at the floor, a 3 mm lip at 0 .. 3 mm', r['accuracy_mm'], (57 * (a0 or 0) + 3 * 1.5) / 60, 0.02)
        # f. the 20 mm cap and the mask
        p = os.path.join(d, 'far.ply'); write_ply(p, V + [0, 0, 25.0], F); _, R1 = run(p, 901, d2)
        check('plate 25 mm away: stops, nothing in the mask or below the cap (1 = yes)', float(R1['stop']['reason'] in ('no_points_in_mask', 'no_distance_below_cap')), 1, 0, '')
        p = os.path.join(d, 'out.ply'); write_ply(p, np.vstack([V, V + [0, 0, 15.0]]), np.vstack([F, F + len(V)])); r, _ = run(p, 901, d2)
        check('a second sheet 15 mm above, outside the observation mask: accuracy unchanged', r['accuracy_mm'], a0 or 0, 0.01)
        p = os.path.join(d, 'in.ply'); write_ply(p, np.vstack([V, V + [0, 0, 6.0]]), np.vstack([F, F + len(V)])); r, _ = run(p, 901, d2)
        check('a second sheet 6 mm above, inside the mask: accuracy = half the points at 6 mm', r['accuracy_mm'], 3.0 + (a0 or 0) / 2, 0.06)
        synthetic_flat(os.path.join(d, 'tall'), margin=30.0)
        r, R7 = run(os.path.join(d, 'in.ply'), 901, os.path.join(d, 'tall'), max_dist=5.0)
        check('the same with --max-dist 5: the sheet at 6 mm is an outlier, accuracy back at the floor', r['accuracy_mm'], a0 or 0, 0.01)
        # g. the guards
        Vn = V.copy(); Vn[5, 1] = np.nan; p = os.path.join(d, 'nan.ply'); write_ply(p, Vn, F); _, R2 = run(p, 901, d2)
        check('a NaN vertex: refused (1 = yes)', float(R2['stop']['reason'] == 'non_finite_input'), 1, 0, '')
        p = os.path.join(d, 'same.ply'); R3 = evaluate(p, 900, d, max_minutes=1e-9, quiet=True); check('a wall-clock budget of zero: stops, writes no result (1 = yes)', float(R3['stop']['reason'] == 'max_minutes' and 'result' not in R3), 1, 0, '')
        R4 = evaluate(p, 900, d, max_points=1000, quiet=True); check('--max-points too small: stops before allocating (1 = yes)', float(R4['stop']['reason'] == 'max_points'), 1, 0, '')
        R5 = evaluate(p, 902, d, quiet=True); check('a scan without reference files: says which are missing (1 = yes)', float(R5['stop']['reason'] == 'missing_reference' and 'stl902_total.ply' in R5['stop']['detail']), 1, 0, '')
        ra, _ = run(p, 900, d, seed=1); rb, _ = run(p, 900, d, seed=1); rc, _ = run(p, 900, d, seed=2)
        check('same seed, same numbers', abs(ra['overall_mm'] - rb['overall_mm']), 0, 0); check('another seed: overall changes by less than 1 %', abs(ra['overall_mm'] - rc['overall_mm']) / ra['overall_mm'], 0, 0.01, '')
        # h. a real reference, when one is given: the reference against itself, and moved rigidly
        if data and scan is not None:
            print('real reference: scan %d under %s' % (scan, data), flush=True)
            try: ref = reference(data, scan)
            except Stop as e: ref = None; print('  skipped: %s' % e)
            if ref:
                r, R6 = run(ref['stl'], scan, data, mode='pcd')
                check('scan %d: its reference cloud as input (pcd): accuracy' % scan, r['accuracy_mm'], 0, 1e-9); check('scan %d: its reference cloud as input (pcd): completeness (at most the 0.2 mm of the reduction)' % scan, r['completeness_mm'], 0.1, 0.1)
                # a rigid shift by d reads as d only where the surface faces the shift (see the flat plate above); on a real
                # scene the parts that lie along the shift slide on themselves, so the accuracy is above 0 and at most d
                S, _ = read_ply(ref['stl'], faces=False); P = loadmat(ref['plane'])['P'].astype(float).ravel(); n = P[:3] / np.linalg.norm(P[:3])
                for off in (1.0, 3.0):
                    p = os.path.join(d, 'moved.ply'); write_ply(p, S + off * n); r, _ = run(p, scan, data, mode='pcd'); got = r['accuracy_mm']
                    check('scan %d: reference moved rigidly by %g mm along the table normal: accuracy (above 0, at most %g)' % (scan, off, off), got, off / 2, off / 2 + 1e-4)
                    check('scan %d: the same, completeness (above 0, at most %g)' % (scan, off), r['completeness_mm'], off / 2, off / 2 + 0.2)
    print('%d checks, %d failed' % (len(rows), len(fails)), flush=True)
    return dict(tool='tools/eval/dtu_chamfer.py', version=VERSION, self_test=rows, failed=fails)


def synthetic_flat(d, scan=901, size=60.0, step=0.2, margin=10.0):
    """As synthetic(), but exactly flat (z = 500) and without the table strip: a rigid shift along z must read as itself."""
    os.makedirs(os.path.join(d, 'ObsMask'), exist_ok=True); os.makedirs(os.path.join(d, 'Points', 'stl'), exist_ok=True)
    x, y = np.meshgrid(np.arange(0, size + 1e-9, step), np.arange(0, size + 1e-9, step)); S = np.c_[x.ravel(), y.ravel(), np.full(x.size, 500.0)]
    write_ply(os.path.join(d, 'Points', 'stl', 'stl%03d_total.ply' % scan), S)
    BB = np.array([[-30, -10, 480], [70, 70, 520]], np.int16); shape = tuple((BB[1] - BB[0] + 1).tolist()); mask = np.zeros(shape, np.uint8)
    gx = np.arange(shape[0]) + BB[0, 0]; gy = np.arange(shape[1]) + BB[0, 1]; gz = np.arange(shape[2]) + BB[0, 2]
    mask[np.ix_((gx >= -margin) & (gx <= size + margin), (gy >= -margin) & (gy <= size + margin), np.abs(gz - 500) <= margin)] = 1
    savemat(os.path.join(d, 'ObsMask', 'ObsMask%d_10.mat' % scan), dict(ObsMask=mask, BB=BB, Res=np.uint8(1), Margin=np.uint8(10), cSet=np.uint16(scan)))
    savemat(os.path.join(d, 'ObsMask', 'Plane%d.mat' % scan), dict(P=np.array([[0.0], [0.0], [1.0], [-497.5]])))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], epilog='See tools/eval/README.md.')
    ap.add_argument('mesh', nargs='?', help='PLY in the DTU world frame, millimetres')
    ap.add_argument('--scan', type=int, help='DTU scan number')
    ap.add_argument('--data', help='folder with the reference: ObsMask/ObsMask{N}_10.mat, ObsMask/Plane{N}.mat, Points/stl/stl{N:03d}_total.ply')
    ap.add_argument('--mode', choices=('mesh', 'pcd'), default='mesh', help='mesh: sample the triangles first (default). pcd: the vertices are the points')
    ap.add_argument('--out', help='where to write the JSON (default: MESH with .dtu_chamfer.json)')
    ap.add_argument('--density', type=float, default=0.2, help='sampling spacing and smallest distance between kept points, mm (protocol: 0.2)')
    ap.add_argument('--patch', type=float, default=60.0, help='how far around the reference bounding box points are kept, mm (protocol: 60)')
    ap.add_argument('--max-dist', type=float, default=20.0, help='distances at or above this are outliers and left out of the means, mm (protocol: 20)')
    ap.add_argument('--seed', type=int, default=0, help='seed of the shuffle before the reduction (the reference implementations do not seed it); default 0')
    ap.add_argument('--max-minutes', type=float, default=30.0, help='wall-clock budget; the job stops with stop.reason = max_minutes (default 30; 0 = none)')
    ap.add_argument('--max-points', type=int, default=120000000, help='memory guard: refuse a mesh that samples to more points than this (default 120 million, about 3 GB)')
    ap.add_argument('--workers', type=int, default=-1, help='threads of the nearest-neighbour searches (default -1 = all)')
    ap.add_argument('--note', help='free text stored in the JSON, e.g. what the mesh is')
    ap.add_argument('--mask-only', action='store_true', help='stop after step 3 (needs only ObsMask and Plane): how many points of the input the protocol would count. No distance, no result')
    ap.add_argument('--self-test', action='store_true', help='run the checks on synthetic data (and, with --data and --scan, on that real reference)')
    a = ap.parse_args()
    if a.self_test:
        R = self_test(a.data, a.scan, a.max_minutes, a.workers)
        if a.out:
            with open(a.out, 'w') as f: json.dump(R, f, indent=1)
        sys.exit(1 if R['failed'] else 0)
    if not (a.mesh and a.scan is not None and a.data): ap.error('MESH, --scan and --data are needed (or --self-test)')
    R = evaluate(a.mesh, a.scan, a.data, a.mode, a.density, a.patch, a.max_dist, a.seed, a.max_minutes, a.max_points, a.workers, a.note, mask_only=a.mask_only)
    out = a.out or os.path.splitext(a.mesh)[0] + '.dtu_chamfer.json'
    with open(out, 'w') as f: json.dump(R, f, indent=1)
    log('wrote %s (stop: %s)' % (out, R['stop']['reason']))
    sys.exit(0 if R['stop']['reason'] == 'completed' else 2)


if __name__ == '__main__':
    main()
