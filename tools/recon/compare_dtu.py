#!/usr/bin/env python3
"""YARDSTICK ONLY: compare a chart from sfm_spline.py with the shipped depth-fitted owl chart (talk/dtu-data.js) and
with tools/dtu/pipeline.py's 12 x 16 sketch (talk/dtu/pipeline122.js init). Nothing here feeds back into the fit.

    python3 -I tools/recon/compare_dtu.py OUT/charts.json TRUTH/transform.json [--chart object] [--pictures DIR] [-o compare.json]

The shipped chart is itself an OFFLINE FIT to the dataset's depth maps (rendered from DTU's reference surface): it is
a yardstick, not ground truth, and it is open at the back and on top of the head. Distances are one-sided and in mm:
  ours -> shipped    every sample of our chart (and only the part standing on SfM points, by its bridge mask)
  shipped -> ours    every sample of the shipped chart where it was measured (its seen mask)
Several charts (--chart a,b or all): sampled in proportion to their area and pooled; per-chart numbers too. A chart
with a trim mask (sfm_spline.py --trim) is also measured on its trimmed part.
"Where": our samples binned by height above the table and by azimuth around the shipped owl's vertical axis (front =
towards the cameras), ledges (normal within 30 degrees of vertical, above the table), plus the share of our chart that lies beyond the shipped chart's border (no counterpart)."""
import argparse, base64, json, math, os, re, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bspline as bs

TALK = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'talk')


def shipped():
    t = open(os.path.join(TALK, 'dtu-data.js')).read(); d = json.loads(re.search(r'= (\{"scan".*\});\n', t).group(1)); c = d['charts'][0]
    q = np.frombuffer(base64.b64decode(c['P']), '<u2').reshape(c['nv'], c['nu'], 3).astype(float)
    P = np.array(c['lo']) + (np.array(c['hi']) - np.array(c['lo'])) * q / 65535
    bits = np.unpackbits(np.frombuffer(base64.b64decode(c['seen']), np.uint8))[:c['nu'] * c['nv']].reshape(c['nv'], c['nu']).astype(bool)
    return P, bits, d['cams'], d['unit_mm']


def sketch():
    t = open(os.path.join(TALK, 'dtu', 'pipeline122.js')).read(); d = json.loads(re.search(r'\]\s*=\s*(\{"scan".*\});\n', t).group(1)); i = d['init']
    return np.array(i['P']).reshape(i['nv'], i['nu'], 3), np.array(i['support']).reshape(i['nv'], i['nu'])


def chart_axes(c):
    au = bs.Periodic(c['nu']) if c['u']['periodic'] else bs.Open(c['u']['knots']); av = bs.Open(c['v']['knots']); return au, av


def dense(au, av, net, n=(400, 400)):
    (S, Su, Sv), U, V = bs.grid(au, av, net, n[0], n[1], True); N = np.cross(Su, Sv); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-18
    return S.reshape(-1, 3), N.reshape(-1, 3), U.ravel(), V.ravel()


def dist(Q, S, N):
    """Distance from points Q to the surface sampled as S (normals N): nearest sample, then its tangent plane if close."""
    d, j = cKDTree(S).query(Q, workers=-1); pl = np.abs(np.einsum('pc,pc->p', Q - S[j], N[j]))
    step = np.median(cKDTree(S).query(S[::50], 2)[0][:, 1])
    return np.where(d < 3 * step, np.minimum(d, pl), d), d


def stats(d):
    d = np.asarray(d)
    return dict(n=int(len(d)), mean_mm=round(float(d.mean()), 3), median_mm=round(float(np.median(d)), 3), p95_mm=round(float(np.percentile(d, 95)), 3), max_mm=round(float(d.max()), 2)) if len(d) else dict(n=0)


def render(sets, path, az=0.0, el=10.0, px=2.4, size=(440, 560), centre=(0.0, 140.0, 0.0)):
    """Plain z-sorted point renders, one tile per set, side by side (mm, +y up)."""
    from PIL import Image
    a, e = np.radians(az), np.radians(el)
    R = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]]) @ np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
    tiles = []
    for pts, col in sets:
        p = (np.asarray(pts).reshape(-1, 3) - centre) @ R.T; col = np.broadcast_to(np.asarray(col, float).reshape(-1, 3), p.shape)
        W, H = size; img = np.full((H, W, 3), 0.08); x = (W / 2 + p[:, 0] * px).astype(int); y = (H / 2 - p[:, 1] * px).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H); o = np.argsort(p[ok, 2]); img[y[ok][o], x[ok][o]] = col[ok][o]
        tiles.append(img)
    Image.fromarray((np.clip(np.hstack(tiles), 0, 1) * 255).astype(np.uint8)).save(path)


def shade(N, base=(0.75, 0.78, 0.82)):
    L = np.array([0.35, 0.5, 0.8]); L /= np.linalg.norm(L); s = np.abs(N @ L) * 0.75 + 0.25; return s[:, None] * np.array(base)


def ramp(d, top):
    t = np.clip(d / top, 0, 1)[:, None]; return (1 - t) * np.array([0.2, 0.55, 1.0]) + t * np.array([1.0, 0.15, 0.1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('charts'); ap.add_argument('transform')
    ap.add_argument('--chart', default='object', help='chart name, a comma list (pooled), or all'); ap.add_argument('--pictures'); ap.add_argument('-o', '--out'); ap.add_argument('--model', help='the COLMAP model, for the points in the pictures')
    a = ap.parse_args(); J = json.load(open(a.charts)); T = json.load(open(a.transform)); Q, Tt, U = np.array(T['Q']), np.array(T['T']), T['unit_mm']
    tomm = lambda X: ((np.asarray(X).reshape(-1, 3) - Tt) @ Q) * U                  # model -> scene frame -> mm
    names = [x['name'] for x in J['charts']] if a.chart == 'all' else a.chart.split(',')
    sel = [next(x for x in J['charts'] if x['name'] == nm) for nm in names]
    # several charts: each sampled in proportion to its area (one chart: 400 x 400, as before), then pooled
    areas = []
    for c in sel:
        au, av = chart_axes(c); (S0, Su0, Sv0), _, _ = bs.grid(au, av, np.array(c['net']), 64, 64, True); areas.append(float(np.linalg.norm(np.cross(Su0, Sv0), axis=-1).mean()))
    S, N, Uu, Vv, on_pts, trim, cid, A = [], [], [], [], [], [], [], []
    for k, c in enumerate(sel):
        au, av = chart_axes(c); f = math.sqrt(areas[k] / max(areas)); n_ = (max(40, int(400 * f)), max(40, int(400 * f)))
        s_, n2, u_, v_ = dense(au, av, np.array(c['net']), n_); S.append(s_); N.append(n2); Uu.append(u_); Vv.append(v_); cid.append(np.full(len(s_), k))
        bm = np.array(c['bridge_mask_48x48_v_rows']); g = bm.shape[0]; on_pts.append(bm[np.minimum((v_ * g).astype(int), g - 1), np.minimum((u_ * g).astype(int), g - 1)] == 0)
        tm = np.array(c['trim_mask_48x48_v_rows']) if 'trim_mask_48x48_v_rows' in c else np.ones((g, g), int)
        trim.append(tm[np.minimum((v_ * g).astype(int), g - 1), np.minimum((u_ * g).astype(int), g - 1)] == 1)
        (_, Su1, Sv1), _, _ = bs.grid(au, av, np.array(c['net']), n_[0], n_[1], True); A.append(np.linalg.norm(np.cross(Su1, Sv1), axis=-1).ravel())
    S, N, Uu, Vv, on_pts, trim, cid, A = [np.concatenate(x) for x in (S, N, Uu, Vv, on_pts, trim, cid, A)]
    Smm = tomm(S); Nmm = N @ Q
    P, seen, cams, um = shipped(); sau, sav = bs.Open.uniform(P.shape[1]), bs.Open.uniform(P.shape[0])
    Sh, Nh, Us, Vs = dense(sau, sav, P * um, (576, 784)); seen_s = seen[np.minimum((Vs * P.shape[0]).astype(int), P.shape[0] - 1), np.minimum((Us * P.shape[1]).astype(int), P.shape[1] - 1)]
    d_ours, d_raw = dist(Smm, Sh, Nh); d_back, _ = dist(Sh[seen_s], Smm, Nmm)
    # beyond the shipped chart: our samples whose nearest shipped sample sits on its border and is not along the normal
    tree = cKDTree(Sh); _, j = tree.query(Smm, workers=-1); edge = (Us[j] < 0.003) | (Us[j] > 0.997) | (Vs[j] > 0.997)
    beyond = edge & (d_raw > 3.0)
    # where: height bands and azimuth sectors around the shipped owl's axis (front = towards the mean camera)
    ax = np.median(Sh[:, [0, 2]], 0); cm = np.mean(np.array(cams) * um, 0); front = np.arctan2(cm[0] - ax[0], cm[2] - ax[1])
    az = np.degrees(np.mod(np.arctan2(Smm[:, 0] - ax[0], Smm[:, 2] - ax[1]) - front + np.pi, 2 * np.pi) - np.pi)
    hy = Smm[:, 1]; where = []; ledge = (np.abs(Nmm[:, 1]) > math.cos(math.radians(30))) & (hy > 10)
    for nm, m in [('front |az| < 45', np.abs(az) < 45), ('sides 45 < |az| < 120', (np.abs(az) >= 45) & (np.abs(az) < 120)), ('back |az| >= 120', np.abs(az) >= 120),
                  ('foot y < 40 mm', hy < 40), ('books / body 40-200 mm', (hy >= 40) & (hy < 200)), ('head y >= 200 mm', hy >= 200),
                  ('ledges: normal within 30 deg of vertical, y > 10 mm', ledge),
                  ('our top edge v > 0.95', Vv > 0.95), ('our side edges u < 0.05 or > 0.95', (Uu < 0.05) | (Uu > 0.95))]:
        where.append(dict(region=nm, share_of_chart=round(float(m.mean()), 4), all=stats(d_ours[m]), on_points=stats(d_ours[m & on_pts]), trimmed=stats(d_ours[m & trim]), beyond_shipped_border=round(float(beyond[m].mean()), 4) if m.any() else 0))
    per = [dict(chart=c['name'], kind=c['kind'], net=[c['nu'], c['nv']], area_mm2=round(areas[k] * U * U, 1), unsupported_area_share=round(float(A[(cid == k) & ~on_pts].sum() / A[cid == k].sum()), 4),
                trimmed_area_share=round(float(A[(cid == k) & trim].sum() / A[cid == k].sum()), 4), to_shipped_all=stats(d_ours[cid == k]), to_shipped_on_points=stats(d_ours[(cid == k) & on_pts])) for k, c in enumerate(sel)]
    out = dict(note='yardstick only: distances to the shipped depth-fitted chart (itself an offline fit, open at the back and on top), in mm; several charts are sampled in proportion to their area and pooled',
               ours=dict(chart=','.join(names), net=[[c['nu'], c['nv']] for c in sel], to_shipped_all=stats(d_ours), to_shipped_on_points=stats(d_ours[on_pts]), to_shipped_trimmed=stats(d_ours[trim]),
                         to_shipped_not_beyond_border=stats(d_ours[~beyond]), beyond_shipped_border_share=round(float(beyond.mean()), 4), on_points_share=round(float(on_pts.mean()), 4),
                         unsupported_area_share=round(float(A[~on_pts].sum() / A.sum()), 4), trimmed_area_share=round(float(A[trim].sum() / A.sum()), 4),
                         unsupported_share_of_trimmed_area=round(float(A[trim & ~on_pts].sum() / max(A[trim].sum(), 1e-18)), 4),
                         shipped_seen_to_ours=stats(d_back), where=where, per_chart=per))
    Pi, sup = sketch(); iau, iav = bs.Open.uniform(Pi.shape[1]), bs.Open.uniform(Pi.shape[0]); Si, Ni, Ui, Vi = dense(iau, iav, Pi * um, (300, 400))
    di, _ = dist(Si, Sh, Nh); dib, _ = dist(Sh[seen_s], Si, Ni)
    out['pipeline_sketch_12x16'] = dict(to_shipped_all=stats(di), shipped_seen_to_it=stats(dib),
                                        note='tools/dtu/pipeline.py init: SfM points only, but laid out in fit.py\'s depth-derived axis, azimuth range and rows')
    print(json.dumps(out, indent=1))
    if a.out: json.dump(out, open(a.out, 'w'), indent=1)
    if a.pictures:
        os.makedirs(a.pictures, exist_ok=True); sets = []
        if a.model:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import colmap_io
            M = colmap_io.read_points(os.path.join(a.model, 'points3D.txt')); X = tomm(M['X']); ok = (np.abs(X - [0, 140, 0]) < [200, 200, 200]).all(1)
            sets.append((X[ok], np.clip(M['rgb'][ok] / 255 * 1.6, 0, 1)))
        sets += [(Smm, shade(Nmm)), (Sh, shade(Nh, (0.8, 0.75, 0.65))), (Smm, ramp(d_ours, 5.0))]
        if len(sel) > 1: pal = np.array([[0.95, 0.6, 0.2], [0.3, 0.6, 1.0], [0.5, 0.9, 0.4], [0.9, 0.4, 0.8], [0.9, 0.9, 0.3]]); sets.append((Smm, pal[cid % len(pal)] * shade(Nmm, (1, 1, 1))))
        for az in (0, -60, 60, 180):
            render(sets, os.path.join(a.pictures, 'compare-az%+04d.png' % az), az=az)
        print('pictures (SfM points within 200 mm | ours | shipped | ours coloured by distance to shipped, blue 0 -> red >= 5 mm | with several charts: ours by chart), turned about +y by 0, -60, 60, 180 deg in the scene frame -> %s' % a.pictures)


if __name__ == '__main__':
    main()
