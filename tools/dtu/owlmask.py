#!/usr/bin/env python3
"""Tell the owl of DTU scan 122 apart from the books it stands on, per control point of its chart.

    python3 -I tools/dtu/owlmask.py                      -> talk/dtu/owlmask122.js
    options: [--charts talk/dtu-data.js] [--features talk/dtu/features122.js] [-o FILE] [--check DIR]

The scan's object chart (`owl`, made by fit.py) is ONE sheet over the owl and the books under it. The talk's materials
can be put on the owl alone; this script decides, offline, which control points are the owl. Nothing in it is trained
by the thesis method; the features it reads come from a PRETRAINED model (DINOv3, baked by features.py).

What it does:
  1. reads the object's chart from the scan's data file (control points, colours) and the DINOv3 features on its
     control points (talk/dtu/features122.js, stored at every second control point: read between them bilinearly)
  2. seeds by height above the table: everything above y = 2.4 (head and shoulders) is owl, everything below y = 0.9
     (the two closed books) is book; nothing else is given by hand
  3. a linear classifier on the 16 feature numbers, fitted to the seeds (logistic regression with a ridge, the two
     classes weighted equally) -> per control point the log-odds z of owl against book
  4. the geometry of the net: its normal curvature across every lattice edge, on the net smoothed over 1.5 control
     points. Where the owl rests on the books the surface folds inwards: a concave crease
  5. a minimum cut over the lattice (8 neighbours). A control point pays for its label by z, but only beyond a margin
     (|z| > 1.25): a feature patch is seven control points wide, so next to the boundary the features cannot tell, and
     there the geometry decides. An edge pays for being cut, the less the more concave it is. Inside the margin a
     control point leans a little to the owl (labelling it book costs 0.12), so that a small part of the owl lying on
     the book, the talons, goes with the owl where a crease outlines it. The cut is the cheapest line between the sure
     owl and the sure books, so it runs along the crease where the owl rests on the books
  6. one owl and one book region: a piece of either that holds none of its seeds is given to the other
  7. writes the labels as runs (a few KB); the file feathers the edge over about two control points when it loads

How it was checked (scan 122, defaults): the owl is 58,466 of the chart's 112,896 control points. The cut crosses
1,033 lattice edges, 98 % of them concave (median normal curvature -18 per scene unit, against +0.8 over the whole
lattice); every column of the chart crosses it once; no control point disagrees with the majority of its 3 x 3
neighbourhood. Varying the margin (1.0 .. 1.5), the lean (0.08 .. 0.16), the smoothing (1.0 .. 2.5), the crease scale
(6 .. 12) or the weight of the features (0.2 .. 0.8) changes at most 186 control points (0.3 % of the owl). A margin of
1.75 changes 828; at 2 the cut spills onto the top of the open book (5,719). And by eye, in the pictures of --check and
in the talk's renderer, against the photographs' colours: front, sides, back, the talons and the open page.

--check DIR writes the pictures the numbers come from: the chart's lattice (colour, classifier, crease, labels) and
the object seen from the front, both sides, the back and close to the feet and the open page, with the owl tinted.

Needs numpy and scipy (Pillow for --check). About a second."""
import argparse, base64, json, os, re, sys, time
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import maximum_flow, breadth_first_order

T0 = time.time()
SCAN = 122
OWL_Y, BOOK_Y = 2.4, 0.9           # seeds: above OWL_Y all owl, below BOOK_Y all book (scene units, table at y = 0)


def log(msg):
    print('[owlmask %5.1fs] %s' % (time.time() - T0, msg), file=sys.stderr, flush=True)


def read(path):
    """A data file the talk loads (a classic script) -> the JSON object it assigns."""
    m = re.search(r'= (\{"scan".*\});\n', open(path).read())
    if not m: sys.exit('no data object in ' + path)
    return json.loads(m.group(1))


def chart(d, name):
    """The object chart of a data file fit.py wrote, decoded the way the file's own loader does it."""
    c = next(c for c in d['charts'] if c['name'] == name)
    nu, nv = c['nu'], c['nv']
    q = np.frombuffer(base64.b64decode(c['P']), '<u2').reshape(nv, nu, 3).astype(float)
    P = np.array(c['lo']) + (np.array(c['hi']) - np.array(c['lo'])) * q / 65535
    C = np.frombuffer(base64.b64decode(c['C']), np.uint8).reshape(nv, nu, 3) / 255
    seen = np.unpackbits(np.frombuffer(base64.b64decode(c['seen']), np.uint8))[:nu * nv].reshape(nv, nu)
    return P, C, seen


def features(f, name, nu, nv):
    """The baked features of a chart on every control point: stored node (a, b) = control point (a * step, b * step)."""
    c = f['charts'][name]; st = c.get('step', 1)
    if c.get('net', [nu, nv]) != [nu, nv]: sys.exit('the features were baked for another net (%s); re-run features.py' % c.get('net'))
    F = np.frombuffer(base64.b64decode(c['data']), np.int8).reshape(c['nv'], c['nu'], f['dim']).astype(float) * f['scale']
    J, I = np.meshgrid(np.minimum(np.arange(nv) / st, c['nv'] - 1), np.minimum(np.arange(nu) / st, c['nu'] - 1), indexing='ij')
    return np.stack([ndimage.map_coordinates(F[..., k], [J, I], order=1, mode='nearest') for k in range(f['dim'])], -1)


def classify(F, owl, book, ridge):
    """Logistic regression (Newton steps) on the features of the seeds, both classes weighted equally -> log-odds."""
    X = np.concatenate([F.reshape(-1, F.shape[-1]), np.ones((owl.size, 1))], 1)
    idx = np.flatnonzero(owl.ravel() | book.ravel()); y = owl.ravel()[idx].astype(float); Xs = X[idx]
    wt = np.where(y > 0, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    w = np.zeros(X.shape[1])
    for _ in range(60):
        p = 1 / (1 + np.exp(-np.clip(Xs @ w, -60, 60)))
        g = Xs.T @ (wt * (p - y)) + ridge * w
        H = (Xs * (wt * p * (1 - p))[:, None]).T @ Xs + ridge * np.eye(len(w))
        s = np.linalg.solve(H, g); w -= s
        if np.abs(s).max() < 1e-9: break
    z = (X @ w).reshape(owl.shape)
    return z, w, float(((z.ravel()[idx] > 0) == (y > 0)).mean())


def smooth(P, s):
    return np.stack([ndimage.gaussian_filter(P[..., k], s, mode='nearest') for k in range(P.shape[-1])], -1)


def normals(P):
    """Unit normals of a net (nv, nu, 3): central differences along u and v, crossed (S_u x S_v points out)."""
    nv, nu = P.shape[:2]
    ip, im = np.minimum(np.arange(nu) + 1, nu - 1), np.maximum(np.arange(nu) - 1, 0)
    jp, jm = np.minimum(np.arange(nv) + 1, nv - 1), np.maximum(np.arange(nv) - 1, 0)
    N = np.cross(P[:, ip] - P[:, im], P[jp] - P[jm])
    return N / (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-12)


# the lattice's neighbours: (dj, di, weight of the edge per unit of boundary length); 8 neighbours, the diagonal
# edges weighted 1 / sqrt(2) so that a cut's cost is close to its length whatever its direction on the lattice
EDGES = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.5 ** 0.5), (1, -1, 0.5 ** 0.5)]


def pairs(nv, nu, dj, di):
    """Index pairs (p, q) of all lattice edges of one direction."""
    J, I = np.mgrid[0:nv, 0:nu]
    ok = (J + dj < nv) & (I + di >= 0) & (I + di < nu)
    p = (J * nu + I)[ok]
    return p, p + dj * nu + di


def creases(P, s):
    """Per edge direction: the normal curvature of the net smoothed over s control points across each edge
    (negative = the surface folds inwards there)."""
    Ps = smooth(P, s); N = normals(Ps); nv, nu = P.shape[:2]
    out = []
    for dj, di, _ in EDGES:
        p, q = pairs(nv, nu, dj, di)
        A, B = Ps.reshape(-1, 3), N.reshape(-1, 3)
        d = A[q] - A[p]
        out.append(((B[q] - B[p]) * d).sum(-1) / ((d * d).sum(-1) + 1e-12))
    return out


def cut(nv, nu, src, snk, edges):
    """Minimum s-t cut on the lattice. src[i] = cost of labelling i book, snk[i] = cost of labelling it owl (np.inf =
    never), edges = [(p, q, w)] (each both ways). -> boolean (nv, nu), True = owl (the source side)."""
    n = nv * nu; S, Tt = n, n + 1
    fin = np.concatenate([src[np.isfinite(src)], snk[np.isfinite(snk)]] + [w for _, _, w in edges])
    k = 2e8 / max(1.0, fin.sum())                       # integers for scipy; the sum of all finite capacities fits
    big = np.int64(2 ** 30)
    def ints(x): return np.where(np.isfinite(x), np.round(x * k), big).astype(np.int64)
    rows, cols, cap = [], [], []
    a = np.flatnonzero(src > 0); rows.append(np.full(len(a), S)); cols.append(a); cap.append(ints(src[a]))
    b = np.flatnonzero(snk > 0); rows.append(b); cols.append(np.full(len(b), Tt)); cap.append(ints(snk[b]))
    for p, q, w in edges:
        wi = ints(w); rows += [p, q]; cols += [q, p]; cap += [wi, wi]
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.minimum(np.concatenate(cap), big)
    G = sparse.csr_matrix((v.astype(np.int32), (r, c)), shape=(n + 2, n + 2))
    G.sum_duplicates()
    res = maximum_flow(G, S, Tt, method='dinic')
    R = (G - res.flow).tocsr(); R.data[R.data < 0] = 0; R.eliminate_zeros()
    seen = breadth_first_order(R, S, directed=True, return_predecessors=False)
    lab = np.zeros(n + 2, bool); lab[seen] = True
    return lab[:n].reshape(nv, nu), res.flow_value / k


def tidy(lab, owl_seed, book_seed):
    """One owl region and one book region: a piece of either that does not hold its seeds goes to the other."""
    four = ndimage.generate_binary_structure(2, 1)
    def keep(m, seed):
        L, k = ndimage.label(m, four)
        hold = np.unique(L[seed & m]); hold = hold[hold > 0]
        return np.isin(L, hold), k
    lab2, ko = keep(lab, owl_seed)
    book, kb = keep(~lab2, book_seed)
    return ~book, ko, kb


def majority(lab):
    """How many control points disagree with the majority of their 3 x 3 neighbourhood (a ragged edge, a lone node)."""
    k = ndimage.uniform_filter(lab.astype(float), 3, mode='nearest') * 9
    return ((lab & (k < 4.5)) | (~lab & (k > 4.5))).sum()


def runs(lab):
    """Lengths of the alternating runs of a boolean array, row by row (j * nu + i), starting with False."""
    x = lab.ravel().astype(np.int8)
    edges = np.flatnonzero(np.diff(x)) + 1
    b = np.concatenate([[0], edges, [len(x)]])
    r = np.diff(b).tolist()
    if x[0]: r = [0] + r
    return r


JS = '''/* The owl of DTU scan %(scan)d told apart from the books it stands on, per control point of the object's chart.
   GENERATED by tools/dtu/owlmask.py: do not edit, re-run the script (and re-run it whenever the scan's control nets change).
   Segmented OFFLINE from the DINOv3 features of the scan's photographs (talk/dtu/features%(scan)d.js; a pretrained model,
   Meta, DINOv3 License) and the geometry of the net (the concave crease where the owl rests on the book), by a minimum cut
   over the chart's lattice. Not trained, and not a result of the thesis method.
   KnotTalk.dtuMasks[%(scan)d] = { scan, chart, nu, nv, feather, runs, stats, mask }. runs = lengths of alternating runs of
   books (first) and owl over control point j * nu + i; on load they become mask = Float32Array(nu * nv), 0 = books ..
   1 = owl, its edge feathered over about `feather` control points on each side (a box blur of the labels, twice). */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {};
  var M = (KT.dtuMasks = KT.dtuMasks || {})[%(scan)d] = %(json)s;
  var nu = M.nu, nv = M.nv, n = nu * nv, L = new Float32Array(n), T = new Float32Array(n), r, k = 0, on = 0, i, j, a, b, s, c;
  for (r = 0; r < M.runs.length; r++, on ^= 1) for (i = 0; i < M.runs[r]; i++) L[k++] = on;
  for (var pass = 0; pass < M.feather; pass++) {                 // a 3 x 3 box blur, edges clamped
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) { s = 0; for (a = -1; a <= 1; a++) s += L[j * nu + Math.min(nu - 1, Math.max(0, i + a))]; T[j * nu + i] = s / 3; }
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) { s = 0; for (b = -1; b <= 1; b++) s += T[Math.min(nv - 1, Math.max(0, j + b)) * nu + i]; L[j * nu + i] = s / 3; }
  }
  M.mask = L;
})();
'''


def write(path, scan, nu, nv, lab, stats, feather):
    d = {'scan': scan, 'chart': 'owl', 'nu': nu, 'nv': nv, 'feather': feather, 'stats': stats, 'runs': runs(lab)}
    txt = JS % {'scan': scan, 'json': json.dumps(d, separators=(',', ':'))}
    open(path, 'w').write(txt)
    log('wrote %s (%.1f KB, %d runs)' % (path, len(txt) / 1024, len(d['runs'])))


# ---------- pictures for --check ----------
def render(P, col, N, th, ph, W, H, scale, centre):
    """Orthographic splats of the control points, back faces left out, seen from azimuth th and polar angle ph
    (the talk's orbit convention: the cameras are towards +z)."""
    f = -np.array([np.sin(ph) * np.sin(th), np.cos(ph), np.sin(ph) * np.cos(th)])
    r = np.cross(f, [0, 1, 0]); r /= np.linalg.norm(r); u = np.cross(r, f)
    q = P.reshape(-1, 3) - np.array(centre); col = col.reshape(-1, 3)
    X, Y, Z = q @ r, q @ u, q @ f
    vis = N.reshape(-1, 3) @ -f > -0.05
    px, py = (W / 2 + X * scale).astype(int), (H / 2 - Y * scale).astype(int)
    img = np.empty((H, W, 3)); img[:] = (0.03, 0.04, 0.06)
    o = np.argsort(-Z); o = o[vis[o]]
    for dx in (0, 1):
        for dy in (0, 1):
            x, y = px[o] + dx, py[o] + dy; ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            img[y[ok], x[ok]] = col[o[ok]]
    return img


def check(dirn, P, C, z, crs, lab, mask):
    from PIL import Image
    os.makedirs(dirn, exist_ok=True)
    def save(a, name, k=2):
        a = np.clip(a, 0, 1)
        if a.ndim == 2: a = np.repeat(a[..., None], 3, -1)
        im = Image.fromarray((a * 255).astype(np.uint8))
        if k != 1: im = im.resize((im.width * k, im.height * k), Image.NEAREST)
        im.save(os.path.join(dirn, name)); log('  ' + os.path.join(dirn, name))
    flip = lambda a: a[::-1]                                       # v runs up the picture
    photo = np.minimum(1, C * 1.8)
    save(flip(photo), 'lattice-colour.png')
    p = 1 / (1 + np.exp(-z)); save(flip(np.stack([p, 0.15 + 0 * p, 1 - p], -1)), 'lattice-classifier.png')
    nv, nu = lab.shape; k = np.zeros(nv * nu)                      # per control point: the most concave edge at it
    for (dj, di, _), c in zip(EDGES, crs):
        p, q = pairs(nv, nu, dj, di); np.minimum.at(k, p, c); np.minimum.at(k, q, c)
    k = k.reshape(nv, nu); k = np.clip(-k / np.percentile(-k, 97), 0, 1)
    save(flip(np.stack([k, 0.6 * k, 0.15 + 0 * k], -1)), 'lattice-crease.png')
    edge = (ndimage.binary_dilation(lab) & ~lab) | (ndimage.binary_dilation(~lab) & lab)
    tint = photo * (1 - 0.55 * mask[..., None]) + np.array([0.95, 0.35, 0.2]) * 0.55 * mask[..., None]
    save(flip(np.where(edge[..., None], 1.0, tint)), 'lattice-owl.png')
    N = normals(P)
    shots = [('front', 0.0, 1.25, 640, 760, 190, (0, 1.75, 0.2)), ('left', 1.25, 1.25, 640, 760, 190, (0, 1.75, 0.2)),
             ('right', -1.25, 1.25, 640, 760, 190, (0, 1.75, 0.2)), ('back-left', 2.2, 1.15, 640, 760, 190, (0, 1.75, 0.2)),
             ('back-right', -2.2, 1.15, 640, 760, 190, (0, 1.75, 0.2)), ('top', 0.3, 0.45, 640, 760, 190, (0, 1.75, 0.2)),
             ('feet', 0.35, 1.2, 760, 560, 520, (0.2, 1.55, 0.35)), ('page', -0.35, 1.05, 760, 560, 440, (-0.45, 1.45, 0.25)),
             ('tail', -1.9, 1.1, 760, 560, 420, (-0.2, 1.6, 0.0))]
    for name, th, ph, W, H, sc, ce in shots:
        a = render(P, tint, N, th, ph, W, H, sc, ce)
        save(a, 'view-%s.png' % name, 1)


def main():
    here = os.path.dirname(os.path.abspath(__file__)); talk = os.path.join(here, '..', '..', 'talk')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--charts', default=os.path.join(talk, 'dtu-data.js'))
    ap.add_argument('--features', default=os.path.join(talk, 'dtu', 'features%d.js' % SCAN))
    ap.add_argument('-o', '--out', default=os.path.join(talk, 'dtu', 'owlmask%d.js' % SCAN))
    ap.add_argument('--check', help='write pictures of the result into this directory')
    ap.add_argument('--smooth', type=float, default=1.5, help='net smoothed over this many control points for the crease')
    ap.add_argument('--margin', type=float, default=1.25, help='log-odds within which the features do not decide')
    ap.add_argument('--unary', type=float, default=0.4, help='weight of the features beyond the margin')
    ap.add_argument('--fold', type=float, default=8.0, help='concavity (1 / scene unit) at which an edge costs 1/e')
    ap.add_argument('--grow', type=float, default=0.12, help='where the features do not decide, what labelling a node book costs')
    ap.add_argument('--feather', type=int, default=2, help='box blurs of the labels when the page loads them')
    a = ap.parse_args()

    d = read(a.charts)
    if d['scan'] != SCAN: sys.exit('this script knows scan %d only' % SCAN)
    P, C, seen = chart(d, 'owl'); nv, nu = P.shape[:2]
    f = read(a.features); F = features(f, 'owl', nu, nv)
    log('chart owl %d x %d, features %s, %d numbers' % (nu, nv, f['model'], f['dim']))
    y = P[..., 1]
    owl_seed, book_seed = y > OWL_Y, y < BOOK_Y
    z, w, acc = classify(F, owl_seed, book_seed, 1e-3)
    log('classifier on %d owl and %d book seeds: %.2f %% of them on the right side' % (owl_seed.sum(), book_seed.sum(), 100 * acc))

    crs = creases(P, a.smooth)
    src = a.unary * np.maximum(0, z - a.margin) + a.grow * (np.abs(z) < a.margin)   # sure owl: labelling it book costs this much
    snk = a.unary * np.maximum(0, -z - a.margin)                   # sure book
    src = np.where(owl_seed, np.inf, src).ravel(); snk = np.where(book_seed, np.inf, snk).ravel()
    edges = []
    for (dj, di, wl), k in zip(EDGES, crs):
        p, q = pairs(nv, nu, dj, di)
        edges.append((p, q, wl * np.exp(-np.maximum(0, -k) / a.fold)))
    lab, flow = cut(nv, nu, src, snk, edges)
    log('cut: cost %.1f' % flow)
    lab, ko, kb = tidy(lab, owl_seed, book_seed)
    if ko > 1 or kb > 1: log('tidied: %d owl and %d book pieces before, one of each after' % (ko, kb))

    # what the cut runs along: the crease across the cut edges, against the same over the whole lattice
    on = []
    for (dj, di, _), k in zip(EDGES, crs):
        p, q = pairs(nv, nu, dj, di); x = lab.ravel()
        on.append(k[x[p] != x[q]])
    on = np.concatenate(on); allk = np.concatenate(crs)
    feather = a.feather
    stats = {'control_points': int(nu * nv), 'owl': int(lab.sum()), 'books': int((~lab).sum()),
             'owl_share': round(float(lab.mean()), 4),
             'seeds': {'owl_above_y': OWL_Y, 'books_below_y': BOOK_Y, 'owl': int(owl_seed.sum()), 'books': int(book_seed.sum())},
             'classifier_on_seeds': round(acc, 4),
             'classifier_agrees_off_the_seeds': round(float(((z > 0) == lab)[~(owl_seed | book_seed)].mean()), 4),
             'cut_edges': int(len(on)), 'cut_on_concave_edges': round(float((on < 0).mean()), 4),
             'median_curvature_cut': round(float(np.median(on)), 2), 'median_curvature_all': round(float(np.median(allk)), 2),
             'pieces_before_tidy': [int(ko), int(kb)],
             'columns_crossing_once': round(float(((lab[1:] != lab[:-1]).sum(0) == 1).mean()), 4),
             'against_3x3_majority': int(majority(lab)),
             'owl_lowest_y': round(float(np.percentile(y[lab], 0.5)), 3), 'books_highest_y': round(float(np.percentile(y[~lab], 99.5)), 3),
             'params': {'smooth': a.smooth, 'margin': a.margin, 'unary': a.unary, 'fold': a.fold, 'grow': a.grow}}
    log('owl %d of %d control points (%.1f %%); the cut crosses %d edges, %.0f %% of them concave (median curvature %.1f, '
        'whole lattice %.1f); off the seeds the classifier alone agrees on %.1f %%'
        % (stats['owl'], nu * nv, 100 * lab.mean(), len(on), 100 * stats['cut_on_concave_edges'], stats['median_curvature_cut'],
           stats['median_curvature_all'], 100 * stats['classifier_agrees_off_the_seeds']))
    write(a.out, SCAN, nu, nv, lab, stats, feather)
    if a.check:
        L = lab.astype(float)
        for _ in range(feather):
            L = ndimage.uniform_filter1d(L, 3, axis=1, mode='nearest'); L = ndimage.uniform_filter1d(L, 3, axis=0, mode='nearest')
        check(a.check, P, C, z, crs, lab, L)


if __name__ == '__main__':
    main()
