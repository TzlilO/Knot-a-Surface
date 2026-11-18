#!/usr/bin/env python3
"""Label the points of a COLMAP model by per-view image labels (segmentation masks and / or feature maps), so that
sfm_spline.py --labels can fit one chart per object.

    python3 -I tools/recon/label_points.py MODEL_DIR -o OUT [--masks DIR] [--features DIR] [--ground-plane] [options]
    python3 -I tools/recon/label_points.py --self-test [-o DIR]

MODEL_DIR = a COLMAP text model WITH tracks (images.txt POINTS2D). OUT gets labels.npy (one integer per POINT3D_ID,
index = id, -1 = no label), labels_confidence.npy (same indexing, -1 = no evidence) and labels.json.

Per-view inputs, looked up by the image's NAME in images.txt (NAME.npy, STEM.npy, NAME.png, STEM.png):
  --masks DIR     integer label images, H x W (one level) or H x W x K (K levels, e.g. a mask hierarchy); 0 = no mask.
                  Ids mean nothing across views. Any resolution: pixel positions are scaled by its size / the camera's.
                  PNG: 8 / 16-bit grey or palette (the value is the id) or RGB(A) (id = R + 256 G + 65536 B).
  --features DIR  float feature maps H' x W' x D (any stride, read bilinearly between cell centres).
A view without a file is simply missing.

Method (every threshold is a flag):
  1. observations: each point is read ONLY in the views of its own track, at the 2-D position the track stores.
  2. candidate pairs: 3-D k nearest neighbours (links shorter than --link x the median k-th distance) and, per view,
     the --img-knn nearest observed points in the image within --img-radius (co-observed neighbours, which may be far
     apart in depth). Sparse: O(points x k).
  3. mask votes per candidate pair: in every (view, level) where both points are observed and both lie in a mask,
     the pair votes "same" (both in one mask) or "different". c = number of votes (co-visibility), s = "same" votes.
     A pair with c >= --min-covis gets the weight w = s - tau c (each same vote +(1 - tau), each different vote -tau).
  4. clustering: greedy agglomeration on that signed graph, the standard heuristic for correlation clustering /
     minimum-cost multicut (GAEC, Keuper et al. 2015): contract two touching clusters while the summed weight between
     them is positive. The ORDER is average linkage (--linkage mean, as in GASP, Bailoni et al. 2019): the pair with
     the highest weight per vote (agreement - tau) first; --linkage sum is plain GAEC; --linkage best (default) runs
     both and keeps the partition with the higher correlation-clustering objective (the summed weight of the pairs
     inside clusters): both orders are heuristics for that one objective. Deterministic (ties by index).
     Two objects stay apart when the votes between them sum negative, however many single pairs agree by noise.
  5. spatial sanity: each cluster is split into its connected pieces of the 3-D kNN graph.
  6. features (when given): per point the mean of its L2-normalised feature samples, normalised. They (a) merge two
     touching clusters whose mean features have cosine >= --feat-merge where the masks between them are undecided
     (agreement >= tau - --tie-band, or no votes), (b) decide where a small cluster goes, and (c) without masks are the
     only signal: agglomeration over the same candidate pairs with centroid linkage (merge the touching clusters whose
     mean features are most alike while their cosine is above --feat-tau).
  7. small clusters (< --min-size points) join the touching cluster with the best evidence (features, else mask
     weight, else number of links) unless the masks say they differ; otherwise they become -1.
  8. --ground-plane (prior): the support plane of sfm_spline.py (RANSAC with every camera on one side); points within
     --ground-tol x s of it form the ground cluster before anything else and take no part in steps 3-7.
Confidence of a point = the share of its votes (mask votes, or feature-edge votes without masks) that agree with the
final labels; -1 where it has none. labels.json records every threshold, the clusters (size, which signal formed
them, confidence), every merge and split with its reason, and the stop record.

Long-job rules: --max-seconds bounds the run (GAEC stops, the partition reached so far is valid and is written with
stop.reason = wall_clock); SIGINT / SIGTERM do the same with reason = signal. numpy + scipy only; nothing is trained."""
import argparse, heapq, json, math, os, signal, struct, sys, time, zlib
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io

T0 = time.time()
QUIET = {'on': False}
def log(*a):
    if not QUIET['on']: print('[%6.1fs]' % (time.time() - T0), *a, flush=True)

STOP = {'signal': None}
def _on_signal(sig, frame): STOP['signal'] = signal.Signals(sig).name; log('received %s: stopping, the partition so far is written' % STOP['signal'])


# ====================================================================== image files (numpy + zlib only)
PNG_SIG = b'\x89PNG\r\n\x1a\n'


def read_png(path):
    """Minimal PNG decoder: non-interlaced, bit depth 8 or 16, grey / grey+alpha / RGB / RGBA / palette (8 bit).
    Returns H x W (grey, palette index) or H x W x C integers."""
    data = open(path, 'rb').read()
    if data[:8] != PNG_SIG: raise ValueError('%s: not a PNG' % path)
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        n = struct.unpack('>I', data[pos:pos + 4])[0]; typ = data[pos + 4:pos + 8]; body = data[pos + 8:pos + 8 + n]; pos += 12 + n
        if typ == b'IHDR': hdr = struct.unpack('>IIBBBBB', body)
        elif typ == b'IDAT': idat.append(body)
        elif typ == b'IEND': break
    W, H, bd, ct, _, _, inter = hdr
    if inter: raise ValueError('%s: interlaced PNG is not supported (write it without interlacing, or as .npy)' % path)
    ch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct]
    if bd not in (8, 16) or (ct == 3 and bd != 8): raise ValueError('%s: bit depth %d (colour type %d) is not supported' % (path, bd, ct))
    bpp = ch * bd // 8; stride = W * bpp; raw = np.frombuffer(zlib.decompress(b''.join(idat)), np.uint8).reshape(H, stride + 1)
    out = np.zeros((H, stride), np.int32); prev = np.zeros(stride, np.int32)
    for y in range(H):
        f, line = raw[y, 0], raw[y, 1:].astype(np.int32)
        if f == 0: cur = line
        elif f == 1:
            cur = np.empty(stride, np.int32)
            for k in range(bpp): cur[k::bpp] = np.cumsum(line[k::bpp]) % 256
        elif f == 2: cur = (line + prev) % 256
        else:
            cur = line.copy()
            for i in range(stride):                                   # Average / Paeth are sequential
                a = cur[i - bpp] if i >= bpp else 0; b = prev[i]; c = prev[i - bpp] if i >= bpp else 0
                if f == 3: cur[i] = (cur[i] + (a + b) // 2) % 256
                else:
                    p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    cur[i] = (cur[i] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))) % 256
        out[y] = cur; prev = cur
    v = out.reshape(H, W, ch * (bd // 8))
    if bd == 16: v = v[..., 0::2] * 256 + v[..., 1::2]
    if ct in (0, 3): return v[..., 0]
    if ct == 4: return v[..., 0]
    return v[..., 0] + 256 * v[..., 1] + 65536 * v[..., 2]           # RGB(A): id = R + 256 G + 65536 B


def write_png16(path, a):
    """16-bit grey PNG (filter 0), for the self-test and for tools that want PNG label images."""
    a = np.asarray(a).astype('>u2'); H, W = a.shape
    raw = b''.join(b'\x00' + a[y].tobytes() for y in range(H))
    def chunk(t, b): return struct.pack('>I', len(b)) + t + b + struct.pack('>I', zlib.crc32(t + b) & 0xffffffff)
    with open(path, 'wb') as fh:
        fh.write(PNG_SIG + chunk(b'IHDR', struct.pack('>IIBBBBB', W, H, 16, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def find_file(d, name, exts):
    stem = os.path.splitext(name)[0]
    for e in exts:
        for c in (name + e, stem + e):
            p = os.path.join(d, c)
            if os.path.isfile(p): return p
    return None


class DirSource:
    """Per-view arrays from a directory: masks (integer, H x W [x K]) or features (H' x W' x D)."""
    def __init__(self, d, kind): self.d, self.kind = d, kind
    def __call__(self, name):
        p = find_file(self.d, name, ('.npy', '.png') if self.kind == 'masks' else ('.npy',))
        if p is None: return None
        a = np.load(p, allow_pickle=False) if p.endswith('.npy') else read_png(p)
        if self.kind == 'masks':
            if not np.issubdtype(a.dtype, np.integer): raise SystemExit('%s: mask images must hold integers, got %s' % (p, a.dtype))
            return a[:, :, None] if a.ndim == 2 else a
        if a.ndim != 3: raise SystemExit('%s: feature maps must be H x W x D, got shape %s' % (p, a.shape))
        return a


# ====================================================================== model -> observations
def observations(M):
    P = M['points']; ids = P['id']; mp = np.full(int(ids.max()) + 2, -1, np.int64); mp[ids] = np.arange(len(ids))
    names, cams, pt, im, xy = [], [], [], [], []
    for k, iid in enumerate(sorted(M['images'])):
        I = M['images'][iid]; c = M['cameras'][I['camera_id']]; names.append(I['name']); cams.append(dict(W=c['width'], H=c['height'], R=I['R'], C=I['centre'], fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy']))
        p = I['p3d']; ok = (p >= 0) & (p < len(mp)); ok[ok] &= mp[p[ok]] >= 0
        pt.append(mp[p[ok]]); im.append(np.full(int(ok.sum()), k)); xy.append(I['xy'][ok])
    pt, im, xy = np.concatenate(pt), np.concatenate(im), np.concatenate(xy)
    key = pt * len(names) + im; _, first = np.unique(key, return_index=True)      # one observation per (point, view)
    first = np.sort(first)
    return dict(ids=ids, X=P['X'], names=names, cams=cams, pt=pt[first], im=im[first], xy=xy[first])


def sample_masks(O, source, max_levels):
    """-> per observation and level: a global segment id (>= 1) or 0, and the per-view record."""
    n_obs = len(O['pt']); K = max_levels; seg = np.zeros((n_obs, K), np.int64); views = []
    for k, name in enumerate(O['names']):
        sel = np.nonzero(O['im'] == k)[0]; a = source(name)
        if a is None: views.append(dict(name=name, mask=False)); continue
        Hm, Wm, L = a.shape; c = O['cams'][k]
        if L > K: raise SystemExit('%s: %d mask levels, more than --max-levels %d' % (name, L, K))
        x = np.clip((O['xy'][sel, 0] * Wm / c['W']).astype(int), 0, Wm - 1); y = np.clip((O['xy'][sel, 1] * Hm / c['H']).astype(int), 0, Hm - 1)
        v = a[y, x].astype(np.int64)                                   # (n_k, L)
        for l in range(L):
            nz = v[:, l] != 0; seg[sel[nz], l] = ((k * K + l) << 34) + v[nz, l]
        views.append(dict(name=name, mask=True, levels=int(L), size=[int(Wm), int(Hm)], observations=int(len(sel)),
                          in_a_mask=float((v != 0).any(1).mean()) if len(sel) else 0.0, ids_seen=int(len(np.unique(v[v != 0])))))
    u, inv = np.unique(seg, return_inverse=True); inv = inv.reshape(seg.shape)
    if u[0] != 0: inv = inv + 1
    return inv, views                                                  # 0 = no mask, otherwise compact ids


def sample_features(O, source, n_pts):
    F, cnt, D, views = None, np.zeros(n_pts), None, []
    for k, name in enumerate(O['names']):
        a = source(name)
        if a is None: views.append(dict(name=name, features=False)); continue
        a = np.asarray(a, np.float32); Hf, Wf, d = a.shape
        if F is None: D = d; F = np.zeros((n_pts, D))
        if d != D: raise SystemExit('%s: feature dimension %d differs from the first view\'s %d' % (name, d, D))
        sel = np.nonzero(O['im'] == k)[0]; c = O['cams'][k]
        fx = np.clip(O['xy'][sel, 0] * Wf / c['W'] - 0.5, 0, Wf - 1); fy = np.clip(O['xy'][sel, 1] * Hf / c['H'] - 0.5, 0, Hf - 1)
        x0, y0 = np.minimum(fx.astype(int), Wf - 2 if Wf > 1 else 0), np.minimum(fy.astype(int), Hf - 2 if Hf > 1 else 0)
        x1, y1 = np.minimum(x0 + 1, Wf - 1), np.minimum(y0 + 1, Hf - 1); tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
        f = (a[y0, x0] * (1 - tx) * (1 - ty) + a[y0, x1] * tx * (1 - ty) + a[y1, x0] * (1 - tx) * ty + a[y1, x1] * tx * ty)
        if not np.all(np.isfinite(f)): raise SystemExit('%s: non-finite feature values' % name)
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-12
        np.add.at(F, O['pt'][sel], f); np.add.at(cnt, O['pt'][sel], 1)
        views.append(dict(name=name, features=True, size=[int(Wf), int(Hf)], dim=int(d)))
    if F is None: return None, cnt, views
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    F[cnt == 0] = 0
    return F, cnt, views


# ====================================================================== candidate pairs and votes
def pairs_3d(X, k, link):
    d, j = cKDTree(X).query(X, k + 1, workers=-1); lim = link * float(np.median(d[:, -1]))
    ok = (d[:, 1:] < lim) & np.isfinite(d[:, 1:]); I = np.repeat(np.arange(len(X)), k)[ok.ravel()]; J = j[:, 1:].ravel()[ok.ravel()]
    return np.minimum(I, J), np.maximum(I, J), lim


def pairs_image(O, k, radius_frac):
    I, J = [], []
    for v in range(len(O['names'])):
        sel = np.nonzero(O['im'] == v)[0]
        if len(sel) < 2: continue
        c = O['cams'][v]; r = radius_frac * math.hypot(c['W'], c['H']); kk = min(k + 1, len(sel))
        d, j = cKDTree(O['xy'][sel]).query(O['xy'][sel], kk, distance_upper_bound=r, workers=-1)
        ok = np.isfinite(d[:, 1:]); a = np.repeat(sel, kk - 1)[ok.ravel()]; b = sel[np.minimum(j[:, 1:], len(sel) - 1)].ravel()[ok.ravel()]
        I.append(O['pt'][a]); J.append(O['pt'][b])
    if not I: return np.zeros(0, int), np.zeros(0, int)
    I, J = np.concatenate(I), np.concatenate(J); keep = I != J
    return np.minimum(I, J)[keep], np.maximum(I, J)[keep]


def unique_pairs(n, *lists):
    I = np.concatenate([l[0] for l in lists]); J = np.concatenate([l[1] for l in lists]); key = np.unique(I.astype(np.int64) * n + J)
    return key // n, key % n


def rowdot(A, I, J, chunk=400000):
    out = np.empty(len(I))
    for a in range(0, len(I), chunk):
        b = min(len(I), a + chunk); out[a:b] = np.asarray(A[I[a:b]].multiply(A[J[a:b]]).sum(1)).ravel()
    return out


def votes(O, seg, n_pts, I, J):
    """c = co-visible (view, level) votes, s = 'same mask' votes, per candidate pair."""
    n_obs, K = seg.shape; rows, colv, cols = [], [], []
    for l in range(K):
        nz = seg[:, l] > 0; rows.append(O['pt'][nz]); colv.append(O['im'][nz] * K + l); cols.append(seg[nz, l])
    rows = np.concatenate(rows); colv = np.concatenate(colv); cols = np.concatenate(cols)
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, colv)), shape=(n_pts, len(O['names']) * K)); B.data[:] = 1
    S = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pts, int(seg.max()) + 1)); S.data[:] = 1
    return rowdot(B, I, J), rowdot(S, I, J)


# ====================================================================== clustering
class Clock:
    def __init__(self, seconds): self.seconds, self.t0, self.reason = seconds, time.time(), None
    def out(self):
        if STOP['signal']: self.reason = 'signal'; return True
        if time.time() - self.t0 > self.seconds: self.reason = 'wall_clock'; return True
        return False


def gaec(n, I, J, W, clock, Cv=None, linkage='sum'):
    """Greedy agglomeration on a signed graph (GAEC, Keuper et al. 2015; average linkage as in GASP, Bailoni et al.
    2019): contract the touching pair of clusters with the highest priority, sum parallel edges, repeat while the
    summed weight between them is positive. Priority = the summed weight ('sum') or the summed weight per vote
    ('mean', i.e. the agreement between the two clusters minus tau). Both stop on the same sign rule; they differ in
    the order of the contractions. Returns a label per node (the representative's index) and the contractions."""
    nb = [dict() for _ in range(n)]; Cv = np.ones(len(I)) if Cv is None else Cv
    for i, j, w, c in zip(I.tolist(), J.tolist(), W.tolist(), Cv.tolist()):
        if i == j: continue
        e = nb[i].get(j)
        if e is None: e = [0.0, 0.0]; nb[i][j] = e; nb[j][i] = e
        e[0] += w; e[1] += c
    pr = (lambda e: e[0]) if linkage == 'sum' else (lambda e: e[0] / max(e[1], 1e-12))
    heap = [(-pr(e), i, j) for i in range(n) for j, e in nb[i].items() if i < j and e[0] > 0]; heapq.heapify(heap)
    alive = np.ones(n, bool); parent = np.arange(n); merges = 0; pops = 0
    while heap:
        pops += 1
        if pops % 20000 == 0 and clock.out(): break
        nw, i, j = heapq.heappop(heap)
        if not (alive[i] and alive[j]): continue
        e = nb[i].get(j)
        if e is None or pr(e) != -nw or e[0] <= 0: continue
        if len(nb[i]) < len(nb[j]): i, j = j, i                       # i absorbs j
        del nb[i][j]; del nb[j][i]
        for x, ex in nb[j].items():
            del nb[x][j]; ei = nb[i].get(x)
            if ei is None: ei = [0.0, 0.0]; nb[i][x] = ei; nb[x][i] = ei
            ei[0] += ex[0]; ei[1] += ex[1]
            if ei[0] > 0: heapq.heappush(heap, (-pr(ei), min(i, x), max(i, x)))
        nb[j] = {}; alive[j] = False; parent[j] = i; merges += 1
    for _ in range(64):                                                # path compression
        p2 = parent[parent]
        if np.array_equal(p2, parent): break
        parent = p2
    return parent, merges


def centroid_features(n, I, J, F, tau, clock):
    """Features alone: agglomeration over the candidate pairs with CENTROID linkage: contract the touching pair of
    clusters whose mean features (normalised sums of the points' unit features) have the highest cosine, while it is
    above tau. Comparing cluster means, not single pairs, keeps a gradual blend at a boundary from chaining two objects."""
    nb = [set() for _ in range(n)]
    for i, j in zip(I.tolist(), J.tolist()):
        if i != j: nb[i].add(j); nb[j].add(i)
    Sm = F.astype(float).copy(); ver = np.zeros(n, int); alive = np.ones(n, bool); parent = np.arange(n); merges = 0
    unit = lambda i: Sm[i] / (np.linalg.norm(Sm[i]) + 1e-12)
    U0 = Sm / (np.linalg.norm(Sm, axis=1, keepdims=True) + 1e-12); c0 = np.einsum('ij,ij->i', U0[I], U0[J]); k = (c0 > tau) & (I != J)
    heap = list(zip((-c0[k]).tolist(), np.minimum(I, J)[k].tolist(), np.maximum(I, J)[k].tolist(), [0] * int(k.sum()), [0] * int(k.sum())))
    heapq.heapify(heap); pops = 0
    while heap:
        pops += 1
        if pops % 20000 == 0 and clock.out(): break
        nc, i, j, vi, vj = heapq.heappop(heap)
        if not (alive[i] and alive[j]) or ver[i] != vi or ver[j] != vj: continue
        if len(nb[i]) < len(nb[j]): i, j = j, i
        Sm[i] += Sm[j]; nb[i] |= nb[j]; nb[i].discard(i); nb[i].discard(j)
        for x in nb[j]:
            if x != i: nb[x].discard(j); nb[x].add(i)
        nb[j] = set(); alive[j] = False; parent[j] = i; ver[i] += 1; merges += 1; ui = unit(i)
        for x in nb[i]:
            c = float(ui @ unit(x))
            if c > tau: heapq.heappush(heap, (-c, min(i, x), max(i, x), ver[min(i, x)], ver[max(i, x)]))
    for _ in range(64):
        p2 = parent[parent]
        if np.array_equal(p2, parent): break
        parent = p2
    return parent, merges


def relabel(lab):
    """Compact labels >= 0 ordered by size (largest = 0); -1 stays."""
    out = np.full(len(lab), -1); m = lab >= 0
    if not m.any(): return out
    u, inv, cnt = np.unique(lab[m], return_inverse=True, return_counts=True); order = np.argsort(-cnt, kind='stable'); rank = np.empty(len(u), int); rank[order] = np.arange(len(u))
    out[m] = rank[inv]; return out


def split_spatial(lab, I3, J3):
    """Each cluster -> its connected pieces in the 3-D kNN graph restricted to the cluster."""
    n = len(lab); same = (lab[I3] == lab[J3]) & (lab[I3] >= 0)
    G = sparse.coo_matrix((np.ones(int(same.sum())), (I3[same], J3[same])), shape=(n, n))
    _, comp = connected_components(G, directed=False)
    new = np.where(lab >= 0, comp, -1); splits = []
    for l in np.unique(lab[lab >= 0]):
        pcs = np.unique(comp[lab == l])
        if len(pcs) > 1:
            sz = np.sort(np.bincount(comp[lab == l]))[::-1]; sz = sz[sz > 0]
            splits.append(dict(cluster_points=int((lab == l).sum()), pieces=int(len(pcs)), largest=sz[:5].tolist()))
    return new, splits


def cluster_pairs(lab, I, J, W, C):
    """Summed mask weight, votes and links between every two touching clusters."""
    m = (lab[I] >= 0) & (lab[J] >= 0) & (lab[I] != lab[J]); a, b = lab[I][m], lab[J][m]; lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo.astype(np.int64) * (int(lab.max()) + 1) + hi; u, inv = np.unique(key, return_inverse=True)
    Ws = np.bincount(inv, W[m], len(u)); Cs = np.bincount(inv, C[m], len(u)); Ls = np.bincount(inv, minlength=len(u))
    n1 = int(lab.max()) + 1
    return {(int(k // n1), int(k % n1)): (float(w), float(c), int(l)) for k, w, c, l in zip(u, Ws, Cs, Ls)}


def means(lab, F):
    out = {}
    for l in np.unique(lab[lab >= 0]):
        v = F[lab == l].sum(0); out[int(l)] = v / (np.linalg.norm(v) + 1e-12)
    return out


def merge_by_features(lab, F, I, J, W, C, a, min_size, rec):
    """Touching clusters (both >= min_size) whose masks are undecided and whose mean features agree: merge, the most
    similar pair first, until none qualifies."""
    while True:
        size = np.bincount(lab[lab >= 0]); big = set(np.nonzero(size >= min_size)[0].tolist())
        P = cluster_pairs(lab, I, J, W, C); mu = means(lab, F); best = None
        for (p, q), (w, c, l) in P.items():
            if p not in big or q not in big or l < a.min_links: continue
            agree = (w + a.tau * c) / c if c > 0 else None
            if agree is not None and agree < a.tau - a.tie_band: continue
            cs = float(mu[p] @ mu[q])
            if cs >= a.feat_merge and (best is None or cs > best[0]): best = (cs, p, q, agree, l)
        if best is None: return lab
        cs, p, q, agree, l = best
        rec.append(dict(merged=[int(size[p]), int(size[q])], reason='features: cosine of the mean features %.3f >= %.2f; masks %s' % (cs, a.feat_merge, 'gave no votes between them' if agree is None else 'undecided (agreement %.2f >= tau - tie band = %.2f)' % (agree, a.tau - a.tie_band)), links=l))
        lab = np.where(lab == q, p, lab)


def absorb_small(lab, F, I, J, W, C, a, min_size, rec):
    """Clusters smaller than min_size join the touching cluster with the best evidence, smallest first; when the masks
    say they differ from every neighbour (agreement < tau - tie band), or nothing touches them, they become -1."""
    for _ in range(8):
        size = np.bincount(lab[lab >= 0]) if (lab >= 0).any() else np.zeros(0, int)
        small = [l for l in np.argsort(size, kind='stable') if 0 < size[l] < min_size]
        if not small: break
        P = cluster_pairs(lab, I, J, W, C); nbr = {}
        for (p, q), v in P.items(): nbr.setdefault(p, []).append((q, v)); nbr.setdefault(q, []).append((p, v))
        mu = means(lab, F) if F is not None else None; changed = False; mp = np.arange(len(size) + 1); mp[-1] = -1    # mp[-1]: label -1 stays
        def root(q):
            while q >= 0 and mp[q] != q: q = mp[q]
            return q
        for l in small:
            l = int(l); cands = []
            for q, (w, c, k) in nbr.get(l, []):
                q = root(q)
                if q == l or q < 0: continue
                agree = (w + a.tau * c) / c if c > 0 else None
                if agree is not None and agree < a.tau - a.tie_band: continue
                score = (float(mu[l] @ mu[q]) if mu is not None and l in mu and q in mu else -2.0, w, k)
                cands.append((score, int(size[q]), q))
            if cands:
                cands.sort(key=lambda t: (t[0], t[1]), reverse=True); mp[l] = cands[0][2]; changed = True
                rec['joined'] += 1; rec['joined_points'] += int(size[l])
            else:
                mp[l] = -1; rec['dropped'] += 1; rec['dropped_points'] += int(size[l]); changed = True
        if not changed: break
        fin = np.array([root(q) for q in range(len(size))] + [-1]); lab = fin[lab]
    return lab


def confidence(lab, I, J, C, S, feat_edges=None):
    """Share of a point's votes that agree with the final labels; -1 without votes."""
    n = len(lab); same = (lab[I] == lab[J]) & (lab[I] >= 0)
    if feat_edges is None:
        agree = np.where(same, S, C - S); tot = C
    else:
        agree = np.where(same, feat_edges, 1 - feat_edges).astype(float); tot = np.ones(len(I))
    A = np.bincount(I, agree, n) + np.bincount(J, agree, n); T = np.bincount(I, tot, n) + np.bincount(J, tot, n)
    return np.where(T > 0, A / np.maximum(T, 1e-12), -1.0)


# ====================================================================== ground prior
def ground(M, O, a):
    import sfm_spline as tool
    D = dict(X=O['X'], cams=[]); cams = []
    for k in sorted(M['images']):
        I = M['images'][k]; cams.append(dict(C=I['centre'], R=I['R']))
    P = M['points']; X = P['X']; ok = (np.array([len(np.unique(t)) for t in P['track_image']]) >= 2) & (P['error'] <= a.ground_max_reproj)
    s = float(np.median(cKDTree(X[ok]).query(X[ok], 2, workers=-1)[0][:, 1]))
    ta = tool.parser().parse_args(['x', '-o', 'y']); ta.plane_tol = a.ground_tol; R = {}
    pl, up = tool.support_plane(X[ok], cams, s, ta, R, np.random.default_rng(a.seed))
    if pl is None: return None, dict(found=False, note='no support plane found: no ground cluster')
    p0, nv, _ = pl; d = (X - p0) @ nv; m = np.abs(d) < a.ground_tol * s
    return m, dict(found=True, normal=nv.tolist(), point=p0.tolist(), spacing=s, tol=a.ground_tol * s, points=int(m.sum()), plane=R.get('plane'))


# ====================================================================== main
def associate(M, masks=None, features=None, a=None):
    """The whole method on a model read by colmap_io.read_model; masks / features are callables name -> array or None.
    Returns (labels per point in the model's order, confidence, record)."""
    t0 = time.time(); clock = Clock(a.max_seconds); O = observations(M); n = len(O['ids']); rec = dict(points=int(n), observations=int(len(O['pt'])), views=len(O['names']))
    tl = np.bincount(O['pt'], minlength=n); rec['observations_per_point'] = dict(median=float(np.median(tl)), mean=float(tl.mean()), max=int(tl.max()) if n else 0)
    log('label_points: %d points, %d views, %d observations (median %.0f per point)' % (n, len(O['names']), len(O['pt']), np.median(tl)))
    X = O['X']; lab0 = np.full(n, -2)                                  # -2 = still to cluster
    if a.ground_plane:
        gm, grec = ground(M, O, a); rec['ground'] = grec
        if gm is not None: lab0[gm] = -3; log('   ground prior: %d points within %.3g of the support plane form the ground cluster' % (gm.sum(), grec['tol']))
    I3, J3, lim = pairs_3d(X, a.knn, a.link); Ii, Ji = pairs_image(O, a.img_knn, a.img_radius) if a.img_knn > 0 else (np.zeros(0, int), np.zeros(0, int))
    I, J = unique_pairs(n, (I3, J3), (Ii, Ji)); rec['pairs'] = dict(knn=a.knn, link=lim, spatial=int(len(I3)), image=int(len(Ii)), img_knn=a.img_knn, img_radius_of_diagonal=a.img_radius, union=int(len(I)))
    log('   candidate pairs: %d spatial (k = %d, links < %.4g), %d co-observed image neighbours (k = %d, within %.1f %% of the image diagonal); %d distinct'
        % (len(I3), a.knn, lim, len(Ii), a.img_knn, 100 * a.img_radius, len(I)))
    seg = None; F = None; signal_used = []
    if masks is not None:
        seg, mviews = sample_masks(O, masks, a.max_levels); rec['mask_views'] = dict(with_mask=int(sum(v['mask'] for v in mviews)), missing=int(sum(not v['mask'] for v in mviews)), per_view=mviews)
        obs_in = (seg > 0).any(1); rec['observations_in_a_mask'] = float(obs_in.mean()) if len(obs_in) else 0.0
        if rec['mask_views']['with_mask'] == 0: seg = None; log('   no mask file matched any view name: masks unused')
        else: log('   masks: %d views with a file, %d missing; %.1f %% of the observations fall in a mask' % (rec['mask_views']['with_mask'], rec['mask_views']['missing'], 100 * rec['observations_in_a_mask']))
    if features is not None:
        F, fcnt, fviews = sample_features(O, features, n)
        rec['feature_views'] = dict(with_features=int(sum(v['features'] for v in fviews)), missing=int(sum(not v['features'] for v in fviews)), points_with_features=int((fcnt > 0).sum()))
        if F is None: log('   no feature file matched any view name: features unused')
        else: log('   features: %d views, dimension %d; %d points have a feature' % (rec['feature_views']['with_features'], F.shape[1], (fcnt > 0).sum()))
    if seg is None and F is None: raise SystemExit('neither masks nor features could be read for any view: nothing to label by')
    free = (lab0[I] == -2) & (lab0[J] == -2)
    if seg is not None:
        C, Sv = votes(O, seg, n, I, J); W = np.where(C >= a.min_covis, Sv - a.tau * C, 0.0); use = free & (C >= a.min_covis)
        rec['votes'] = dict(min_covis=a.min_covis, tau=a.tau, pairs_with_votes=int((C > 0).sum()), pairs_used=int(use.sum()), median_votes_of_used=float(np.median(C[use])) if use.any() else 0.0,
                            agreement_of_used=dict(median=float(np.median(Sv[use] / C[use])) if use.any() else None, share_below_tau=float((Sv[use] < a.tau * C[use]).mean()) if use.any() else None))
        log('   mask votes: %d pairs co-visible in a mask, %d with >= %d votes used (median %.0f votes); %.1f %% of them vote "different" on balance'
            % ((C > 0).sum(), use.sum(), a.min_covis, rec['votes']['median_votes_of_used'], 100 * (rec['votes']['agreement_of_used']['share_below_tau'] or 0)))
        signal_used.append('masks'); fe = None
    else:
        C = np.zeros(len(I)); Sv = np.zeros(len(I))
        cs = np.einsum('ij,ij->i', F[I], F[J]); has = (np.linalg.norm(F[I], axis=1) > 0) & (np.linalg.norm(F[J], axis=1) > 0)
        W = np.where(has, cs - a.feat_tau, 0.0); use = free & has; fe = (cs > a.feat_tau) & has
        rec['feature_edges'] = dict(feat_tau=a.feat_tau, pairs_used=int(use.sum()), cosine=dict(median=float(np.median(cs[use])) if use.any() else None, share_above_tau=float((cs[use] > a.feat_tau).mean()) if use.any() else None))
        log('   features alone (no masks): w = cos - %.2f on %d pairs; %.1f %% above' % (a.feat_tau, use.sum(), 100 * (rec['feature_edges']['cosine']['share_above_tau'] or 0)))
        signal_used.append('features')
    if seg is not None:
        runs = {}
        for lk in (('sum', 'mean') if a.linkage == 'best' else (a.linkage,)):
            lb, mg = gaec(n, I[use], J[use], W[use], clock, C[use], lk); runs[lk] = (float(W[use][lb[I[use]] == lb[J[use]]].sum()), lb, mg)
        pick = max(runs, key=lambda k: runs[k][0]); lab, merges = runs[pick][1], runs[pick][2]
        rec['linkage'] = dict(rule=a.linkage, objective={k: v[0] for k, v in runs.items()}, picked=pick, note='objective = summed weight of the pairs inside clusters (correlation clustering); --linkage best keeps the higher')
        log('   linkage: %s (objective %s)' % (pick, ', '.join('%s %.1f' % (k, v[0]) for k, v in runs.items())))
    else: lab, merges = centroid_features(n, I[use], J[use], F, a.feat_tau, clock)
    rec['gaec'] = dict(contractions=int(merges), clusters=int(len(np.unique(lab[lab0 == -2]))))
    lab = np.where(lab0 == -2, lab, -1)
    log('   GAEC: %d contractions -> %d clusters (incl. singletons)%s' % (merges, rec['gaec']['clusters'], ' -- stopped early: %s' % clock.reason if clock.reason else ''))
    lab, splits = split_spatial(lab, I3, J3); rec['spatial_splits'] = splits
    if splits: log('   spatial sanity: %d cluster(s) were not connected in the kNN graph and were split: %s' % (len(splits), ['%d -> %s' % (s['cluster_points'], s['largest']) for s in splits[:6]]))
    nl = int((lab0 == -2).sum()); min_size = max(a.min_size, int(math.ceil(a.min_share * nl)))
    merges_f = []
    if F is not None and seg is not None:
        lab = merge_by_features(lab, F, I, J, W, C, a, min_size, merges_f)
        if merges_f: signal_used.append('features (merges)')
    rec['feature_merges'] = merges_f
    for m in merges_f: log('   merged %s: %s' % (m['merged'], m['reason']))
    sm = dict(min_size=min_size, joined=0, joined_points=0, dropped=0, dropped_points=0)
    lab = absorb_small(lab, F, I, J, W, C, a, min_size, sm); rec['small_clusters'] = sm
    log('   small clusters (< %d points): %d joined a neighbour (%d points), %d became -1 (%d points)' % (min_size, sm['joined'], sm['joined_points'], sm['dropped'], sm['dropped_points']))
    lab = relabel(lab)
    if a.ground_plane and (lab0 == -3).any():
        lab = np.where(lab >= 0, lab + 1, lab); lab[lab0 == -3] = 0
    conf = confidence(lab, I, J, C, Sv, fe) if seg is not None or fe is not None else np.full(n, -1.0)
    if a.ground_plane: conf[lab0 == -3] = -1.0
    clusters = []
    for l in range(int(lab.max()) + 1 if (lab >= 0).any() else 0):
        m = lab == l; cf = conf[m & (conf >= 0)]
        lo, hi = np.percentile(X[m], 1, 0), np.percentile(X[m], 99, 0)
        clusters.append(dict(label=l, points=int(m.sum()), ground=bool(a.ground_plane and (lab0[m] == -3).all()),
                             signal='ground prior' if a.ground_plane and (lab0[m] == -3).all() else ' + '.join(signal_used),
                             confidence=dict(median=float(np.median(cf)) if len(cf) else None, p10=float(np.percentile(cf, 10)) if len(cf) else None, without_votes=int((conf[m] < 0).sum())),
                             box_1_99=[lo.tolist(), hi.tolist()]))
    rec['clusters'] = clusters; rec['unlabelled'] = int((lab < 0).sum()); rec['signal'] = signal_used
    rec['stop'] = dict(reason=clock.reason or 'done', seconds=round(time.time() - t0, 2))
    log('   -> %d clusters %s; unlabelled %d; stop: %s (%.1f s)' % (len(clusters), [c['points'] for c in clusters[:12]], rec['unlabelled'], rec['stop']['reason'], rec['stop']['seconds']))
    return lab, conf, rec


def run(a):
    M = colmap_io.read_model(a.model, log)
    lab, conf, rec = associate(M, DirSource(a.masks, 'masks') if a.masks else None, DirSource(a.features, 'features') if a.features else None, a)
    ids = M['points']['id']; L = np.full(int(ids.max()) + 1, -1, np.int64); L[ids] = lab; Cf = np.full(len(L), -1.0); Cf[ids] = conf
    os.makedirs(a.out, exist_ok=True); np.save(os.path.join(a.out, 'labels.npy'), L); np.save(os.path.join(a.out, 'labels_confidence.npy'), Cf.astype(np.float32))
    out = dict(tool='tools/recon/label_points.py', what='point labels from per-view image labels (masks and / or features) by vote-weighted correlation clustering; nothing trained',
               model=os.path.abspath(a.model), masks=a.masks, features=a.features,
               files=dict(labels='labels.npy: int64, index = POINT3D_ID, -1 = no label', confidence='labels_confidence.npy: float32, same index, share of the point\'s votes that agree with its label, -1 = no votes'),
               thresholds={k: getattr(a, k) for k in ('knn', 'link', 'linkage', 'img_knn', 'img_radius', 'min_covis', 'tau', 'max_levels', 'feat_tau', 'feat_merge', 'tie_band', 'min_links', 'min_size', 'min_share', 'ground_plane', 'ground_tol', 'ground_max_reproj', 'max_seconds', 'seed')},
               **rec, confidence_per_point=dict(ids=ids.tolist(), value=np.round(conf, 3).tolist()))
    with open(os.path.join(a.out, 'labels.json'), 'w') as fh: json.dump(out, fh, separators=(',', ':'), default=_js)
    log('wrote %s/labels.npy, labels_confidence.npy, labels.json' % a.out)
    return lab, conf, rec


def _js(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    return str(o)


def parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('model', nargs='?', help='COLMAP text model directory (with tracks)'); ap.add_argument('-o', '--out')
    ap.add_argument('--masks', help='directory of per-view integer label images'); ap.add_argument('--features', help='directory of per-view feature maps')
    ap.add_argument('--self-test', action='store_true')
    g = ap.add_argument_group('candidate pairs')
    g.add_argument('--knn', type=int, default=12, help='3-D nearest neighbours per point'); g.add_argument('--link', type=float, default=3.0, help='... kept when closer than this x the median k-th neighbour distance')
    g.add_argument('--img-knn', type=int, default=8, help='co-observed image neighbours per observation (0 = none)'); g.add_argument('--img-radius', type=float, default=0.02, help='... within this share of the image diagonal')
    g = ap.add_argument_group('masks')
    g.add_argument('--min-covis', type=int, default=2, help='a pair needs this many (view, level) votes to carry mask weight')
    g.add_argument('--tau', type=float, default=0.5, help='vote weight: same = +(1 - tau), different = -tau')
    g.add_argument('--linkage', choices=['sum', 'mean', 'best'], default='best', help='contraction order: summed weight (GAEC), weight per vote (average linkage), or both and keep the partition with the higher objective')
    g.add_argument('--max-levels', type=int, default=8, help='largest number of mask levels per view')
    g = ap.add_argument_group('features')
    g.add_argument('--feat-tau', type=float, default=0.8, help='features alone: clusters merge while the cosine of their mean features is above this')
    g.add_argument('--feat-merge', type=float, default=0.9, help='merge touching clusters whose mean features have at least this cosine (masks undecided)')
    g.add_argument('--tie-band', type=float, default=0.15, help='masks are undecided between two clusters when their agreement >= tau - this')
    g.add_argument('--min-links', type=int, default=3, help='clusters touch when at least this many candidate pairs join them')
    g = ap.add_argument_group('clusters')
    g.add_argument('--min-size', type=int, default=30, help='smaller clusters join a neighbour or become -1 ...'); g.add_argument('--min-share', type=float, default=0.005, help='... or smaller than this share of the points (the larger)')
    g.add_argument('--ground-plane', action='store_true', help='prior: points on the support plane (sfm_spline.py\'s RANSAC) form the ground cluster, label 0')
    g.add_argument('--ground-tol', type=float, default=1.5, help='... within this many s (median nearest-neighbour distance)'); g.add_argument('--ground-max-reproj', type=float, default=2.0)
    g = ap.add_argument_group('job'); g.add_argument('--max-seconds', type=float, default=1800.0); g.add_argument('--seed', type=int, default=0)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)
    if a.self_test:
        import selftest_labels; sys.exit(selftest_labels.main(a))
    if not a.model or not a.out or not (a.masks or a.features): parser().error('MODEL_DIR, -o OUT and --masks and / or --features are required (or --self-test)')
    run(a)


if __name__ == '__main__':
    main()
