#!/usr/bin/env python3
"""Train PLAIN 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) on DTU scan 122 with its authors' implementation,
graphdeco-inria/gaussian-splatting, and keep what the talk's pipeline chapter replays: snapshots of the Gaussians and a
log of the loss and the PSNR. For a machine with an NVIDIA GPU and that implementation installed (tools/train/README.md).

    python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122 --dry-run
    python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122
    python3 -I tools/train/train3dgs.py ... -- --data_device cpu          # after --: options of graphdeco's own parser

It calls graphdeco's training() in this process, with the implementation's default settings. Three things around it:
  1. The dataset's photographs are RGBA with the object's mask in alpha, and graphdeco would multiply its render by that
     mask in the loss. Plain 3DGS sees the whole photograph, so it trains on RGB copies in <out>/images_rgb (PIL drops
     the channel; the RGB values are untouched). --keep-alpha trains on the files as they are.
  2. graphdeco's --eval: every 8th photograph by name (0000, 0008, ... 0056) is held out. --all-views trains on all.
  3. graphdeco's training_report() is wrapped: it saves the initialisation as iteration 0 (the Gaussians graphdeco made
     from the SfM points, before the first optimizer step) and writes log.csv. The saves at 100 ... 30000 are
     graphdeco's own (its iteration k: the state after k - 1 steps). Nothing in the training changes: the extra
     renders run inside graphdeco's no_grad block, touch no state it keeps, and draw no random numbers.
Writes into <out>: point_cloud/iteration_<k>/point_cloud.ply, log.csv, run.json and graphdeco's usual files.
Next: python3 -I tools/train/export.py <out>      (writes talk/dtu/train122.js)"""
import argparse, csv, datetime, importlib.util, inspect, json, math, os, shutil, subprocess, sys, time

PIN = '54c035f7834b564019656c3e3fcc3646292f727d'       # graphdeco-inria/gaussian-splatting, 2024-10-30
SAVE = [0, 100, 300, 1000, 3000, 7000, 15000, 30000]
HOLD = 8                                               # graphdeco's --eval: names sorted, every 8th is a test view
PACKAGES = ['torch', 'torchvision', 'plyfile', 'tqdm', 'cv2', 'PIL', 'diff_gaussian_rasterization', 'simple_knn', 'fused_ssim']
TRAINING = ['dataset', 'opt', 'pipe', 'testing_iterations', 'saving_iterations', 'checkpoint_iterations', 'checkpoint', 'debug_from']
REPORT = ['iteration', 'Ll1', 'loss', 'scene', 'renderFunc', 'renderArgs']
COLUMNS = ['iteration', 'loss', 'l1', 'psnr_test', 'psnr_train', 'gaussians', 'seconds']


def say(*a): print(*a, flush=True)
def utc(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def git(d, *a):
    try: r = subprocess.run(['git', '-C', d] + list(a), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError): return None
    return r.stdout.strip() if r.returncode == 0 else None

def log_points(n, save):
    """Rows of log.csv: 0 (the initialisation), every 10 to 100, every 50 to 1000, every 250 after, each save, the last."""
    s = {0, n} | set(range(10, 101, 10)) | set(range(150, 1001, 50)) | set(range(1250, n + 1, 250)) | set(save)
    return sorted(i for i in s if 0 <= i <= n)

def photographs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))) if os.path.isdir(d) else []

def rgb_copies(src, dst):
    """The photographs without alpha. PIL's RGBA -> RGB drops the channel: the RGB values are kept, nothing is blended."""
    from PIL import Image
    os.makedirs(dst, exist_ok=True); n = 0
    for f in photographs(src):
        a, b = os.path.join(src, f), os.path.join(dst, f)
        if os.path.exists(b) and os.path.getmtime(b) >= os.path.getmtime(a): continue
        with Image.open(a) as im:
            if im.mode == 'RGB': shutil.copy2(a, b)
            else: im.convert('RGB').save(b, **({'compress_level': 1} if b.lower().endswith('.png') else {}))
        n += 1
    return n

def check(gs, data):
    """What graphdeco will find: (photograph names in the COLMAP model, facts, warnings, problems)."""
    facts, warn, bad = [], [], []
    if not (os.path.exists(os.path.join(gs, 'train.py')) and os.path.exists(os.path.join(gs, 'arguments', '__init__.py'))):
        bad.append('%s is not a checkout of graphdeco-inria/gaussian-splatting (no train.py, arguments/)' % gs)
    names = photographs(os.path.join(data, 'images')); sp = os.path.join(data, 'sparse', '0')
    have = set(os.listdir(sp)) if os.path.isdir(sp) else set()
    if not names: bad.append('no photographs in %s' % os.path.join(data, 'images'))
    for f in ('cameras', 'images'):
        if f + '.bin' not in have and f + '.txt' not in have: bad.append('no %s.txt in %s' % (f, sp))
    pts = next((f for f in ('points3D.ply', 'points3D.bin', 'points3D.txt') if f in have), None)
    if not pts: bad.append('no points3D.txt in %s' % sp)
    reads = [f + ('.bin' if f + '.bin' in have else '.txt') for f in ('cameras', 'images')] + [pts or '-']
    facts.append('sparse/0: graphdeco will read %s' % ', '.join(reads))
    if any(r.endswith(('.bin', '.ply')) for r in reads):
        warn.append('graphdeco reads .bin files and points3D.ply before the .txt ones; tools/dtu/pipeline.py read the .txt. '
                    'Remove them to train on exactly that model (the exporter checks the frame either way)')
    model = []
    if os.path.exists(os.path.join(sp, 'cameras.txt')):
        kinds = sorted({l.split()[1] for l in open(os.path.join(sp, 'cameras.txt')) if l.strip() and not l.startswith('#')})
        facts.append('camera model: %s' % ', '.join(kinds))
        if not set(kinds) <= {'PINHOLE', 'SIMPLE_PINHOLE'}: bad.append('graphdeco needs PINHOLE or SIMPLE_PINHOLE cameras, not %s' % kinds)
    if os.path.exists(os.path.join(sp, 'images.txt')):
        model = sorted(t[9] for t in (l.split() for l in [l for l in open(os.path.join(sp, 'images.txt')) if not l.startswith('#')][0::2]) if len(t) >= 10)
        gone = [m for m in model if m not in names]
        if gone: bad.append('%d photographs of the model are not in images/: %s' % (len(gone), ' '.join(gone[:6])))
    facts.append('%d photographs in images/%s' % (len(names), ', %d in the model' % len(model) if model else ''))
    if names:
        try:
            from PIL import Image
            with Image.open(os.path.join(data, 'images', names[0])) as im: facts.append('%s: %d x %d, %s' % (names[0], im.size[0], im.size[1], im.mode))
        except ImportError: pass
    return model or names, facts, warn, bad


class Recorder:
    """Wraps graphdeco's training_report(): the iteration-0 save, and a row of log.csv at the logged iterations."""
    def __init__(self, out, save, at, run, masked, torch):
        self.out, self.save, self.at, self.run, self.masked, self.torch = out, save, at, run, masked, torch
        self.f = open(os.path.join(out, 'log.csv'), 'w', newline=''); self.csv = csv.writer(self.f); self.csv.writerow(COLUMNS); self.f.flush()
        self.acc, self.t_eval, self.rows, self.n, self.t0 = [0.0, 0.0, 0], 0.0, 0, None, time.time()
        self.dump()

    def dump(self):
        with open(os.path.join(self.out, 'run.json'), 'w') as f: json.dump(self.run, f, indent=1)

    def wrap(self, orig):
        sig = inspect.signature(orig)
        def report(*args, **kw):
            b = sig.bind(*args, **kw).arguments; it = b['iteration']
            self.acc[0] += float(b['loss'].item()); self.acc[1] += float(b['Ll1'].item()); self.acc[2] += 1
            if it == 1 or it in self.at:
                t = time.time(); scene = b['scene']; rf, ra, tte = b['renderFunc'], b['renderArgs'], bool(b.get('train_test_exp', False))
                if it == 1: self.begin(scene)
                self.row(0 if it == 1 else it, self.psnr(scene.getTestCameras(), scene, rf, ra, tte),
                         self.psnr(scene.getTrainCameras(), scene, rf, ra, tte), scene.gaussians.get_xyz.shape[0])
                self.t_eval += time.time() - t
            return orig(*args, **kw)
        return report

    def begin(self, scene):
        self.run.update(train_views=len(scene.getTrainCameras()), test_views=sorted(c.image_name for c in scene.getTestCameras()),
                        gaussians_initial=int(scene.gaussians.get_xyz.shape[0]))
        self.dump()
        if 0 in self.save:
            say('\n[ITER 0] Saving Gaussians: the initialisation from the SfM points, before the first optimizer step'); scene.save(0)

    def psnr(self, cams, scene, render, rargs, tte):
        """Mean PSNR over cams, rendered as graphdeco's training_report() renders them (clamped to 0..1); with
        --keep-alpha, over each photograph's mask (alpha-weighted)."""
        if not cams: return None
        T, total = self.torch, 0.0
        for cam in cams:
            img = T.clamp(render(cam, scene.gaussians, *rargs)['render'], 0.0, 1.0)
            gt = T.clamp(cam.original_image.to('cuda'), 0.0, 1.0)
            m = cam.alpha_mask.to('cuda') if self.masked else None
            if tte:
                h = img.shape[-1] // 2; img, gt = img[..., h:], gt[..., h:]; m = None if m is None else m[..., h:]
            d = (img - gt) ** 2
            mse = float(d.mean()) if m is None else float((d * m).sum()) / max(3.0 * float(m.sum()), 1e-12)
            total += -10.0 * math.log10(max(mse, 1e-20))
        return total / len(cams)

    def row(self, it, pt, pr, n):
        s, l1, c = self.acc; self.acc = [0.0, 0.0, 0]; self.n = int(n)
        self.csv.writerow([it, '%.6g' % (s / max(c, 1)), '%.6g' % (l1 / max(c, 1)), '' if pt is None else '%.4f' % pt,
                           '' if pr is None else '%.4f' % pr, int(n), '%.1f' % (time.time() - self.t0 - self.t_eval)])
        self.f.flush(); self.rows += 1

    def close(self, error=None):
        self.f.close()
        self.run.update(finished=utc(), minutes=round((time.time() - self.t0) / 60, 2), minutes_logging=round(self.t_eval / 60, 2),
                        log_rows=self.rows, gaussians_final=self.n)
        try: self.run['peak_gpu_memory_gb'] = round(self.torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        except Exception: pass
        if error: self.run['error'] = error
        self.dump()


def main():
    argv, extra = sys.argv[1:], []
    if '--' in argv: i = argv.index('--'); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=' '.join(__doc__.split('\n\n')[0].split()), formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 2)[2] + '\n\nAfter --: options of graphdeco\'s ModelParams / OptimizationParams / '
                                        'PipelineParams (e.g. --data_device cpu); not -s, -m, --eval, --images or --iterations.')
    ap.add_argument('--gs', required=True, help='a checkout of graphdeco-inria/gaussian-splatting with its submodules installed (pinned: %s)' % PIN[:7])
    ap.add_argument('--data', required=True, help='the scan folder: images/ and sparse/0/ (Y1ue/DTU on Hugging Face, DTU/scan122)')
    ap.add_argument('--out', required=True, help='the run folder (graphdeco\'s -m)')
    ap.add_argument('--iterations', type=int, default=30000, help='default 30000, graphdeco\'s own')
    ap.add_argument('--save', default=','.join(map(str, SAVE)), help='iterations to keep the Gaussians at; 0 = the initialisation (default %(default)s)')
    ap.add_argument('--all-views', action='store_true', help='train on every photograph (default: every 8th held out, graphdeco\'s --eval)')
    ap.add_argument('--keep-alpha', action='store_true', help='train on the RGBA photographs as they are (graphdeco then masks its render with alpha)')
    ap.add_argument('--dry-run', action='store_true', help='check the checkout, the packages and the data, print the plan, train nothing')
    a = ap.parse_args(argv)
    gs, data, out = (os.path.abspath(os.path.expanduser(p)) for p in (a.gs, a.data, a.out))
    try: save = sorted({int(v) for v in a.save.split(',') if v.strip()})
    except ValueError: ap.error('--save takes comma-separated iterations')
    late = [s for s in save if s > a.iterations]; save = [s for s in save if s >= 0 and s <= a.iterations]
    rgb = None if a.keep_alpha else os.path.join(out, 'images_rgb')
    gargs = ['-s', data, '-m', out, '--iterations', str(a.iterations)] + ([] if a.all_views else ['--eval']) + (['--images', rgb] if rgb else []) + extra
    names, facts, warn, bad = check(gs, data)
    test = [] if a.all_views else [n for i, n in enumerate(sorted(names)) if i % HOLD == 0]
    head, dirty, subs = git(gs, 'rev-parse', 'HEAD'), git(gs, 'status', '--porcelain', '--untracked-files=no'), git(gs, 'submodule', 'status')

    say('graphdeco    %s: %s' % (gs, ('HEAD %s, %s' % (head[:7], 'the pinned commit' if head == PIN else 'NOT the pinned %s' % PIN[:7])) if head else 'not a git checkout: version unknown'))
    if dirty: say('             local changes: %s' % '; '.join(l.strip() for l in dirty.splitlines()))
    for l in (subs or '').splitlines(): say('             submodule %s' % l.strip())
    say('data         %s' % data)
    for f in facts: say('             %s' % f)
    say('photographs  %s' % ('RGB copies in %s (alpha dropped)' % rgb if rgb else 'as they are: graphdeco masks its render with their alpha'))
    say('views        %s' % ('all %d for training' % len(names) if a.all_views else
                             'held out (every %dth by name): %s; %d for training' % (HOLD, ' '.join(test), len(names) - len(test))))
    say('iterations   %d; Gaussians kept at %s%s' % (a.iterations, ', '.join(map(str, save)), '; beyond the run, skipped: %s' % late if late else ''))
    say('log          log.csv, %d rows: %s' % (len(log_points(a.iterations, save)), ', '.join(COLUMNS)))
    say('graphdeco    training(...) with: %s' % ' '.join(gargs))
    for w in warn: say('NOTE         %s' % w)
    if a.dry_run:
        miss = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
        say('packages     %s' % ('all found: ' + ', '.join(PACKAGES) if not miss else 'MISSING: ' + ', '.join(miss)))
        for b in bad: say('PROBLEM      %s' % b)
        say('dry run: nothing trained%s' % ('; fix the above first' if bad or miss else ''))
        sys.exit(1 if bad or miss else 0)
    if bad: raise SystemExit('stopped: ' + '; '.join(bad))

    os.makedirs(out, exist_ok=True)
    if rgb: say('RGB copies of the photographs: %d written to %s' % (rgb_copies(os.path.join(data, 'images'), rgb), rgb))
    sys.path.insert(0, gs)                                 # graphdeco's own modules: scene, utils, arguments, gaussian_renderer
    spec = importlib.util.spec_from_file_location('gs_train', os.path.join(gs, 'train.py'))
    G = importlib.util.module_from_spec(spec); sys.modules['gs_train'] = G; spec.loader.exec_module(G)
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from utils.general_utils import safe_state
    have, rep = list(inspect.signature(G.training).parameters), inspect.signature(G.training_report).parameters
    if have != TRAINING or any(p not in rep for p in REPORT):
        raise SystemExit('graphdeco\'s training(%s) / training_report(%s) are not those of %s, which this wrapper was written for'
                         % (', '.join(have), ', '.join(rep), PIN[:7]))
    p = argparse.ArgumentParser(); lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p); ga = p.parse_args(gargs)
    run = dict(what='plain 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) on DTU scan 122 with graphdeco\'s implementation and its default '
                    'settings, for the talk\'s pipeline chapter. Not the thesis method.',
               implementation='graphdeco-inria/gaussian-splatting', url='https://github.com/graphdeco-inria/gaussian-splatting',
               commit=head, pinned=PIN, matches_pin=head == PIN, local_changes=bool(dirty), changes=(dirty or '').splitlines(),
               submodules=[l.strip() for l in (subs or '').splitlines()], python=sys.version.split()[0], torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), data=data, keep_alpha=a.keep_alpha,
               images='RGB copies of images/ (alpha dropped)' if rgb else 'images/ as they are (RGBA: graphdeco masks its render with alpha)',
               iterations=a.iterations, save=save, holdout=0 if a.all_views else HOLD, extra_args=extra, graphdeco_args=gargs, started=utc(),
               log=dict(file='log.csv', rows='0 = the initialisation, evaluated at the start of iteration 1 before the first optimizer step; '
                                             'k = inside graphdeco\'s iteration k, where it also saves (the state after k - 1 steps)',
                        columns=dict(loss='training loss as graphdeco computes it, 0.8 L1 + 0.2 (1 - SSIM), mean over the iterations since the previous row',
                                     l1='its L1 part, the same mean',
                                     psnr_test='mean PSNR (dB) of the held-out photographs, rendered by graphdeco\'s render() at that iteration and '
                                               'clamped to 0..1 as its training_report() does' + (', over the alpha mask' if a.keep_alpha else ''),
                                     psnr_train='the same over the training photographs', gaussians='Gaussians at that iteration',
                                     seconds='wall-clock seconds since training() was called, without the time this log and the iteration-0 save took')))
    R = Recorder(out, save, set(log_points(a.iterations, save)) - {0}, run, a.keep_alpha, torch)
    G.training_report = R.wrap(G.training_report)
    safe_state(False)                                      # graphdeco's seeds (random, numpy, torch: 0), cuda:0 and its timestamped console
    R.t0 = time.time()
    try:
        G.training(lp.extract(ga), op.extract(ga), pp.extract(ga), [a.iterations], [s for s in save if s > 0], [], None, -1)
    except BaseException as e:
        R.close('%s: %s' % (type(e).__name__, e)); raise
    R.close()
    say('\ndone: %d Gaussians after %d iterations, %.1f min (%.1f of them for the log); peak GPU memory %s GB\nnext: python3 -I tools/train/export.py %s'
        % (R.n or 0, a.iterations, run['minutes'], run['minutes_logging'], run.get('peak_gpu_memory_gb', '?'), out))


if __name__ == '__main__':
    main()
