"""A small reader and writer for COLMAP text models (cameras.txt, images.txt, points3D.txt). numpy only.

Camera models accepted: PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV. Distortion parameters are read and
ignored (a note is printed once): everything downstream treats each camera as a pinhole. That is exact for models
that COLMAP undistorted (DTU's copy is PINHOLE) and an approximation of a few pixels near the border otherwise.

    M = read_model(dir)
    M['cameras'][cam_id]  = dict(model, width, height, fx, fy, cx, cy, params)
    M['images'][img_id]   = dict(name, camera_id, R (3x3, world -> camera), t (3,), centre (3,), xy (n, 2), p3d (n,) ints)
    M['points']           = dict(id (N,), X (N, 3), rgb (N, 3) uint8, error (N,), track_image (list of int arrays),
                                 track_point2d (list of int arrays))
World -> camera is x_cam = R X + t, a pixel is (fx x/z + cx, fy y/z + cy): COLMAP's convention."""
import os
import numpy as np

KNOWN = {'SIMPLE_PINHOLE': 3, 'PINHOLE': 4, 'SIMPLE_RADIAL': 4, 'RADIAL': 5, 'OPENCV': 8}


def qvec_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_qvec(R):
    K = np.array([[R[0, 0] - R[1, 1] - R[2, 2], 0, 0, 0],
                  [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], 0, 0],
                  [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], 0],
                  [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], R[0, 0] + R[1, 1] + R[2, 2]]]) / 3.0
    w, V = np.linalg.eigh(K); q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def _lines(path):
    with open(path) as fh:
        for l in fh:
            l = l.strip()
            if l and not l.startswith('#'): yield l


def read_cameras(path, log=print):
    cams, noted = {}, set()
    for l in _lines(path):
        t = l.split(); cid, model, W, H = int(t[0]), t[1], int(t[2]), int(t[3]); p = np.array(t[4:], float)
        if model not in KNOWN: raise ValueError('camera %d: model %s is not supported (accepted: %s)' % (cid, model, ', '.join(KNOWN)))
        if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'): fx = fy = p[0]; cx, cy = p[1], p[2]
        else: fx, fy, cx, cy = p[:4]
        if model not in ('PINHOLE', 'SIMPLE_PINHOLE') and model not in noted:
            dist = p[3:] if model in ('SIMPLE_RADIAL', 'RADIAL') else p[4:]
            noted.add(model); log('note: camera model %s: its distortion parameters (%s, first camera) are ignored: treated as a pinhole' % (model, ', '.join('%.3g' % v for v in dist)))
        cams[cid] = dict(model=model, width=W, height=H, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), params=p)
    return cams


def read_images(path):
    imgs = {}; L = list(_open_pairs(path))
    for a, b in L:
        t = a.split(); iid = int(t[0]); q = np.array(t[1:5], float); tv = np.array(t[5:8], float); cid = int(t[8]); name = ' '.join(t[9:])
        R = qvec_to_R(q); v = b.split()
        arr = np.array(v, float).reshape(-1, 3) if v else np.zeros((0, 3))
        imgs[iid] = dict(name=name, camera_id=cid, R=R, t=tv, centre=-R.T @ tv, xy=arr[:, :2], p3d=arr[:, 2].astype(np.int64))
    return imgs


def _open_pairs(path):
    """images.txt holds two lines per image; the second (POINTS2D) may be empty, so blank lines count there."""
    with open(path) as fh: raw = [l.rstrip('\n') for l in fh if not l.startswith('#')]
    i = 0
    while i < len(raw):
        if not raw[i].strip(): i += 1; continue
        yield raw[i], (raw[i + 1] if i + 1 < len(raw) else '')
        i += 2


def read_points(path):
    ids, X, rgb, err, ti, tp = [], [], [], [], [], []
    for l in _lines(path):
        t = l.split(); ids.append(int(t[0])); X.append([float(v) for v in t[1:4]]); rgb.append([int(v) for v in t[4:7]]); err.append(float(t[7]))
        tr = np.array(t[8:], np.int64).reshape(-1, 2); ti.append(tr[:, 0]); tp.append(tr[:, 1])
    return dict(id=np.array(ids, np.int64), X=np.array(X, float).reshape(-1, 3), rgb=np.array(rgb, np.uint8).reshape(-1, 3),
                error=np.array(err, float), track_image=ti, track_point2d=tp)


def read_model(d, log=print):
    for f in ('cameras.txt', 'images.txt', 'points3D.txt'):
        if not os.path.exists(os.path.join(d, f)): raise FileNotFoundError('%s: no %s (a COLMAP TEXT model is expected; convert a binary one with colmap model_converter --output_type TXT)' % (d, f))
    return dict(cameras=read_cameras(os.path.join(d, 'cameras.txt'), log), images=read_images(os.path.join(d, 'images.txt')),
                points=read_points(os.path.join(d, 'points3D.txt')))


def write_model(d, cameras, images, points):
    """cameras {id: dict(model, width, height, params)}, images {id: dict(name, camera_id, R, t, xy, p3d)},
    points dict(id, X, rgb, error, track_image, track_point2d), as read_model returns them."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'cameras.txt'), 'w') as fh:
        fh.write('# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for cid, c in sorted(cameras.items()):
            fh.write('%d %s %d %d %s\n' % (cid, c['model'], c['width'], c['height'], ' '.join('%.10g' % v for v in c['params'])))
    with open(os.path.join(d, 'images.txt'), 'w') as fh:
        fh.write('# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for iid, im in sorted(images.items()):
            q = R_to_qvec(im['R'])
            fh.write('%d %s %s %d %s\n' % (iid, ' '.join('%.12g' % v for v in q), ' '.join('%.12g' % v for v in im['t']), im['camera_id'], im['name']))
            fh.write(' '.join('%.4f %.4f %d' % (x, y, p) for (x, y), p in zip(im['xy'], im['p3d'])) + '\n')
    with open(os.path.join(d, 'points3D.txt'), 'w') as fh:
        fh.write('# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for k in range(len(points['id'])):
            tr = ' '.join('%d %d' % (a, b) for a, b in zip(points['track_image'][k], points['track_point2d'][k]))
            X = points['X'][k]; c = points['rgb'][k]
            fh.write('%d %.9g %.9g %.9g %d %d %d %.6g %s\n' % (points['id'][k], X[0], X[1], X[2], c[0], c[1], c[2], points['error'][k], tr))
