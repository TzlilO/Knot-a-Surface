#!/usr/bin/env python3
"""Turn talk/dtu/pipeline<N>.js (KnotTalk.dtuPipeline[N]) back into a COLMAP text model, so that sfm_spline.py is
tested through its real entry on the scan's real SfM points.

    python3 -I tools/recon/from_pipeline_js.py talk/dtu/pipeline122.js MODEL_DIR [--truth TRUTH_DIR] [--seed 3] [--no-scramble]

What the file holds and what it does not (so what the model can and cannot be):
  - the 64 cameras (centre, world -> camera rotation, one pinhole) and the 21,760 COLMAP points (position, colour,
    track length, mean reprojection error, the smallest and second smallest camera index of each track);
  - NOT the full tracks and NOT the measured 2-D positions. The model's tracks therefore hold only those two images
    per point (one where both are the same), and each observation's (x, y) is the point's exact projection: the
    per-point ERROR column is the original's. The true track lengths go to MODEL_DIR/track_length.npy (index =
    POINT3D_ID) for sfm_spline.py --track-len. Free-space checks on this model use two rays per point, not all.
  - positions are 12-bit over a 10 x 5 x 10 scene-unit box (steps of 0.2 mm or finer), as stored.

By default the whole model goes through a random similarity (rotation, translation; scale 1 = the file's scene unit,
80 mm for 122), so that nothing downstream can lean on the file's +y-up, table-at-0 frame. The transform is written
to TRUTH_DIR/transform.json, which sfm_spline.py never reads: it is for the yardstick (compare_dtu.py) only."""
import argparse, base64, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colmap_io


def b(s): return np.frombuffer(base64.b64decode(s), np.uint8)


def u12(s, n, lo, hi):
    by = b(s).astype(np.uint16); q = np.empty(len(by) // 3 * 2, np.uint16)
    q[0::2] = by[0::3] | ((by[1::3] & 15) << 8); q[1::2] = (by[1::3] >> 4) | (by[2::3] << 4)
    return np.asarray(lo) + (np.asarray(hi) - np.asarray(lo)) * q[:n * 3].reshape(-1, 3) / 4095


def read_pipeline(path):
    t = open(path).read(); m = re.search(r'\]\s*=\s*(\{"scan".*\});\n', t)
    if not m: raise SystemExit('%s: no KnotTalk.dtuPipeline object found' % path)
    D = json.loads(m.group(1)); s = D['sfm']; n, near = s['count'], s['near']
    P = np.r_[u12(s['P'], near, *s['box']), np.frombuffer(base64.b64decode(s['far']), '<f4').reshape(-1, 3).astype(float)]
    assert len(P) == n, (len(P), n)
    return D, dict(P=P, C=b(s['C']).reshape(-1, 3), track=b(s['track']).astype(int), error=b(s['error']) * s['error_scale'],
                   first=b(s['first']).astype(int), second=b(s['second']).astype(int))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('js'); ap.add_argument('model')
    ap.add_argument('--truth', help='directory for transform.json (default: MODEL_DIR/../truth)'); ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--no-scramble', action='store_true', help='keep the file\'s own frame')
    a = ap.parse_args(); D, S = read_pipeline(a.js); cams = D['cameras']; K = cams['intrinsics']
    rng = np.random.default_rng(a.seed)
    if a.no_scramble: Q, T = np.eye(3), np.zeros(3)
    else:
        q = rng.normal(size=4); Q = colmap_io.qvec_to_R(q / np.linalg.norm(q)); T = rng.normal(size=3) * 5
    X = S['P'] @ Q.T + T
    images = {}; ti = [[] for _ in range(len(X))]; tp = [[] for _ in range(len(X))]
    for c in cams['list']:
        i = c['id']; R = np.array(c['R']).reshape(3, 3); C = np.array(c['c'])
        Rm = R @ Q.T; Cm = Q @ C + T; t = -Rm @ Cm
        idx = np.nonzero((S['first'] == i) | (S['second'] == i))[0]
        qv = (X[idx] - Cm) @ Rm.T; xy = np.c_[K['fx'] * qv[:, 0] / qv[:, 2] + K['cx'], K['fy'] * qv[:, 1] / qv[:, 2] + K['cy']]
        for m, j in enumerate(idx): ti[j].append(i + 1); tp[j].append(m)
        images[i + 1] = dict(name='%04d.png' % i, camera_id=1, R=Rm, t=t, xy=xy, p3d=idx + 1)
    pts = dict(id=np.arange(1, len(X) + 1), X=X, rgb=S['C'], error=S['error'], track_image=[np.array(v) for v in ti], track_point2d=[np.array(v) for v in tp])
    colmap_io.write_model(a.model, {1: dict(model='PINHOLE', width=K['width'], height=K['height'], params=[K['fx'], K['fy'], K['cx'], K['cy']])}, images, pts)
    tl = np.zeros(len(X) + 1, np.int32); tl[1:] = S['track']; np.save(os.path.join(a.model, 'track_length.npy'), tl)
    truth = a.truth or os.path.join(os.path.dirname(os.path.abspath(a.model)), 'truth'); os.makedirs(truth, exist_ok=True)
    with open(os.path.join(truth, 'transform.json'), 'w') as fh:
        json.dump(dict(note='model = Q @ scene + T (scene = the frame of %s: +y up, table at y = 0, %g mm per unit). For the yardstick only.' % (os.path.basename(a.js), D['unit_mm']),
                       Q=Q.tolist(), T=T.tolist(), unit_mm=D['unit_mm'], scan=D['scan']), fh, indent=1)
    print('wrote %s: %d images, %d points, %d observations (two per point at most: the file keeps no full tracks); true track lengths -> track_length.npy; transform -> %s'
          % (a.model, len(images), len(X), sum(len(v) for v in ti), os.path.join(truth, 'transform.json')))


if __name__ == '__main__':
    main()
