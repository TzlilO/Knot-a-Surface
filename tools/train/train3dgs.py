#!/usr/bin/env python3
"""Train PLAIN 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) on DTU scan 122 with its authors' implementation,
graphdeco-inria/gaussian-splatting, and keep what the talk's pipeline chapter replays: snapshots of the Gaussians and a
log of the loss and the PSNR. For a machine with an NVIDIA GPU and that implementation installed (tools/train/README.md).

    python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122 --dry-run
    python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122
    python3 -I tools/train/train3dgs.py ... -- --data_device cpu          # after --: options of graphdeco's own parser

It calls graphdeco's training() in this process, with the implementation's default settings. Four things around it:
  1. The dataset's photographs are RGBA with the object's mask in alpha, and graphdeco would multiply its render by that
     mask in the loss. Plain 3DGS sees the whole photograph, so it trains on RGB copies in <out>/images_rgb (PIL drops
     the channel; the RGB values are untouched). --keep-alpha trains on the files as they are.
  2. graphdeco's --eval: every 8th photograph by name (0000, 0008, ... 0056) is held out. --all-views trains on all.
  3. graphdeco's training_report() is wrapped: it saves the initialisation as iteration 0 (the Gaussians graphdeco made
     from the SfM points, before the first optimizer step) and writes log.csv. The saves at 100 ... 30000 are
     graphdeco's own (its iteration k: the state after k - 1 steps). Nothing in the steps it takes changes: the extra
     renders run inside graphdeco's no_grad block, touch no state it keeps, and draw no random numbers.
  4. The run judges itself and stops (README, "When it stops"): at the iteration cap or after --max-minutes; at once on
     NaN or Inf, without saving that state; on a sustained blow-up of the loss, never on the spikes graphdeco's opacity
     resets cause on purpose; on a plateau of the held-out PSNR once densification has ended; at --learn-by if it is
     not learning; before the GPU's memory or --max-gaussians runs out; when no iteration completes for
     --stall-minutes; on SIGINT or SIGTERM. It keeps its best state, and run.json says why and where it stopped.
Writes into <out>: point_cloud/iteration_<k>/point_cloud.ply, log.csv, run.json and graphdeco's usual files.
Next: python3 -I tools/train/export.py <out>      (writes talk/dtu/train122.js)"""
import argparse, csv, datetime, faulthandler, importlib.util, inspect, json, math, os, re, shutil, signal, subprocess, sys, threading, time

PIN = '54c035f7834b564019656c3e3fcc3646292f727d'       # graphdeco-inria/gaussian-splatting, 2024-10-30
SAVE = [0, 100, 300, 1000, 3000, 7000, 15000, 30000]
HOLD = 8                                               # graphdeco's --eval: names sorted, every 8th is a test view
PACKAGES = ['torch', 'torchvision', 'numpy', 'plyfile', 'tqdm', 'cv2', 'PIL', 'diff_gaussian_rasterization', 'simple_knn', 'fused_ssim']
TRAINING = ['dataset', 'opt', 'pipe', 'testing_iterations', 'saving_iterations', 'checkpoint_iterations', 'checkpoint', 'debug_from']
REPORT = ['iteration', 'Ll1', 'loss', 'scene', 'renderFunc', 'renderArgs']
COLUMNS = ['iteration', 'loss', 'l1', 'psnr_test', 'psnr_train', 'gaussians', 'seconds']
PARAMS = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation']    # GaussianModel's tensors, as save_ply writes them
SCHEDULE = dict(densify_from_iter=500, densify_until_iter=15000, opacity_reset_interval=3000, white_background=False)   # graphdeco's defaults
STOPS = [   # the thresholds of the stop rules; tools/train/README.md, "When it stops", says why each default
    ('--max-minutes', 120.0, 'wall-clock budget: minutes from the call of training(), loading the photographs included'),
    ('--max-gaussians', 4000000, 'stop, with a snapshot, once there are more Gaussians than this'),
    ('--max-memory', 0.85, 'stop, with a snapshot, once the CUDA memory allocated peaks above this fraction of the GPU\'s total'),
    ('--stall-minutes', 10.0, 'end the process when no iteration completes for this long (a hung kernel or I/O)'),
    ('--check-every', 100, 'check the Gaussians\' parameters for NaN / Inf every N iterations, and at each row and save (the loss: every iteration)'),
    ('--eval-every', 250, 'a row of log.csv with the held-out PSNR every E iterations (before 1000: more often); the plateau rule counts these'),
    ('--patience', 16, 'plateau: evaluations in a row, after densification has ended, without a gain of --min-delta over the best'),
    ('--min-delta', 0.05, 'plateau: the gain in dB that counts as progress; also how close a snapshot must come to the best state'),
    ('--learn-by', 1000, 'not learning: the iteration by which the held-out PSNR must have risen --learn-margin over iteration 0'),
    ('--learn-margin', 3.0, 'not learning: that rise, in dB'),
    ('--blowup-factor', 2.0, 'blow-up: the smoothed training loss above this many times its best ...'),
    ('--blowup-window', 500, '... for this many iterations in a row'),
    ('--blowup-smooth', 100, 'blow-up: the smoothing, an exponential mean of the loss over about this many iterations'),
    ('--reset-grace', 1000, 'blow-up: iterations after each of graphdeco\'s opacity resets that the rule leaves alone'),
]


def say(*a): print(*a, flush=True)
def utc(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

def git(d, *a):
    try: r = subprocess.run(['git', '-C', d] + list(a), capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError): return None
    return r.stdout.strip() if r.returncode == 0 else None

def log_points(n, save, every=250, extra=()):
    """Rows of log.csv: 0 (the initialisation), every 10 to 100, every 50 to 1000, every `every` (the evaluations the
    stop rules read), each save, the last."""
    s = {0, n} | set(range(10, 101, 10)) | set(range(150, 1001, 50)) | set(range(every, n + 1, every)) | set(save) | set(extra)
    return sorted(i for i in s if 0 <= i <= n)

def opacity_resets(o, n):
    """Iterations at whose end graphdeco sets every opacity to at most 0.01 (train.py at 54c035f: below
    densify_until_iter, every opacity_reset_interval, and at densify_from_iter with a white background)."""
    end = min(o['densify_until_iter'], n + 1); k = o['opacity_reset_interval']
    r = set(range(k, end, k)) if k > 0 else set()
    if o['white_background'] and 0 < o['densify_from_iter'] < end: r.add(o['densify_from_iter'])
    return sorted(r)

def schedule(gs, gargs):
    """graphdeco's densification schedule under these arguments, read with its own parser (arguments/ imports only the
    standard library); its defaults when that cannot be read."""
    try:
        if gs not in sys.path: sys.path.insert(0, gs)
        from arguments import ModelParams, OptimizationParams
        p = argparse.ArgumentParser(); ModelParams(p); OptimizationParams(p); ns = p.parse_known_args(gargs)[0]
        return {k: getattr(ns, k) for k in SCHEDULE}, True
    except Exception:
        return dict(SCHEDULE), False

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
    if pts == 'points3D.ply': facts.append('points3D.ply: graphdeco writes it from points3D.txt on its first run; the copy on Y1ue/DTU holds the same points in another order')
    if any(r.endswith('.bin') for r in reads[:2]):
        warn.append('graphdeco reads cameras.bin / images.bin before the .txt files tools/dtu/pipeline.py read. Remove the .bin files '
                    'to train on exactly that model (the exporter checks the frame against pipeline122.js either way)')
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


# ------------------------------------------------------------------ snapshots from host memory
def host(g):
    """The tensors graphdeco's save_ply writes, copied to host memory: that takes no GPU memory, and the copy no longer
    changes when training goes on."""
    import numpy as np
    return {k: np.asarray(getattr(g, k).detach().cpu().numpy(), np.float32) for k in PARAMS}

def finite(A):
    import numpy as np
    return all(bool(np.isfinite(v).all()) for v in A.values())

def write_ply(path, A):
    """A snapshot in graphdeco's layout (its save_ply at 54c035f: x y z nx ny nz f_dc_* f_rest_* opacity scale_* rot_*,
    all float32, the features channel by channel), written whole into a .part file and then renamed into place."""
    import numpy as np
    n = len(A['_xyz']); dc, rest = (A[k].transpose(0, 2, 1).reshape(n, -1) for k in ('_features_dc', '_features_rest'))
    cols = [A['_xyz'], np.zeros((n, 3), np.float32), dc, rest, A['_opacity'].reshape(n, -1), A['_scaling'], A['_rotation']]
    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + ['f_dc_%d' % i for i in range(dc.shape[1])] + ['f_rest_%d' % i for i in range(rest.shape[1])]
             + ['opacity'] + ['scale_%d' % i for i in range(A['_scaling'].shape[1])] + ['rot_%d' % i for i in range(A['_rotation'].shape[1])])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.part', 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n' % (n, ''.join('property float %s\n' % c for c in names))).encode('ascii'))
        f.write(np.ascontiguousarray(np.concatenate(cols, 1), '<f4').tobytes())
    os.replace(path + '.part', path)

def exit_code(reason):
    """0 done or converged; 1 crashed (with the traceback); 2 stopped early by a limit or a signal; 3 stopped because the
    run went wrong: not learning, diverged, out of GPU memory, stalled. Every one of them leaves an exportable run."""
    if reason.startswith(('done', 'converged')): return 0
    if reason.startswith(('limit', 'signal')): return 2
    if reason.startswith('crashed') and 'memory' not in reason: return 1
    return 3


class Stop(Exception):
    """Raised inside the wrapped training_report() to end graphdeco's loop, once the reason is recorded."""


class Watchdog(threading.Thread):
    """Ends the process when no iteration has completed for --stall-minutes (a hung CUDA kernel or I/O): the stacks of
    all threads go to stall.txt, the record to run.json, the best state from host memory to disk, then the process exits.
    A Python thread runs only while the main thread lets go of Python's lock, which a call into a C extension that hangs
    may never do; for that case faulthandler's timer, which needs no lock, ends the process at twice the limit and
    writes the stacks to stall.txt (run.json then keeps its last progress, and export.py reads the stall from there)."""
    def __init__(self, rec, minutes):
        threading.Thread.__init__(self, name='stall watchdog', daemon=True)
        self.rec, self.limit, self.quit = rec, 60.0 * minutes, threading.Event()

    def run(self):
        while not self.quit.wait(min(30.0, max(0.05, self.limit / 20))):
            idle = time.perf_counter() - self.rec.last
            if idle > self.limit and not self.quit.is_set():
                self.rec.stalled(idle); return


class Recorder:
    """Wraps graphdeco's training_report(): the iteration-0 save, the rows of log.csv, and the rules that stop the run
    (README, "When it stops"). It keeps the best evaluated state in host memory and writes run.json."""
    def __init__(self, out, save, run, a, sched, torch):
        self.out, self.save, self.run, self.a, self.torch = out, set(save), run, a, torch
        self.at = set(log_points(a.iterations, save, a.eval_every, [a.learn_by])) - {0}
        self.evals = set(range(a.eval_every, a.iterations + 1, a.eval_every))
        self.until, self.resets = sched['densify_until_iter'], opacity_resets(sched, a.iterations)
        self.f = open(os.path.join(out, 'log.csv'), 'w', newline=''); self.csv = csv.writer(self.f); self.csv.writerow(COLUMNS); self.f.flush()
        self.acc, self.t_eval, self.rows, self.rows_at, self.n, self.t0 = [0.0, 0.0, 0], 0.0, 0, set(), None, time.monotonic()
        self.it, self.psnr_at, self.best, self.kept, self.disk = 0, {}, None, None, {}   # best: (psnr, iteration); kept: its Gaussians
        self.which = 'training' if a.all_views else 'held-out'
        self.ema = self.low = None; self.low_it = self.high = 0                       # blow-up: smoothed loss, its best, iterations above
        self.ref = self.ref_it = self.learnt = None; self.wait = 0                     # plateau, not learning
        self.stop = self.signal = self.total = self.stall = self.dog = None; self.peak = 0
        self.lock, self.last, self.armed = threading.Lock(), time.perf_counter(), 0.0
        self.dump()

    def dump(self):                                        # atomic: a kill never leaves half a record
        p = os.path.join(self.out, 'run.json'); tmp = '%s.%d.part' % (p, threading.get_ident()); got = self.lock.acquire(timeout=10)
        try:
            with open(tmp, 'w') as f: json.dump(self.run, f, indent=1)
            os.replace(tmp, p)
        finally:
            if got: self.lock.release()

    # ---- the clock, the watchdog, signals
    def start(self):
        """From the call of training(): the budget's clock, the stall watchdog (a thread, and faulthandler's timer as its
        fallback) and the SIGINT / SIGTERM handlers."""
        self.t0 = time.monotonic(); self.stall = open(os.path.join(self.out, 'stall.txt'), 'w'); self.beat()
        self.dog = Watchdog(self, self.a.stall_minutes); self.dog.start()
        for s in (signal.SIGINT, signal.SIGTERM): signal.signal(s, self.on_signal)

    def end(self):
        self.dog.quit.set(); faulthandler.cancel_dump_traceback_later()
        signal.signal(signal.SIGINT, signal.default_int_handler); signal.signal(signal.SIGTERM, signal.SIG_DFL)
        p = self.stall.name; self.stall.close()
        if os.path.exists(p) and os.path.getsize(p) == 0: os.remove(p)

    def beat(self):                                        # progress: for the watchdog, and the fallback timer re-armed now and then
        self.last = now = time.perf_counter()
        if self.stall and threading.current_thread() is threading.main_thread() and now - self.armed > min(5.0, 6 * self.a.stall_minutes):
            self.armed = now; faulthandler.dump_traceback_later(120 * self.a.stall_minutes, exit=True, file=self.stall)

    def on_signal(self, signum, frame):
        """The first SIGINT / SIGTERM asks for a stop at the next iteration, with a snapshot; a second one stops at once.
        Only a flag is set here (and a raw write to stderr), so the handler is safe wherever it lands."""
        name = signal.Signals(signum).name
        if self.signal: raise KeyboardInterrupt(name + ' twice')
        self.signal = name
        try: os.write(2, ('\n%s: the run stops at the next iteration, with a snapshot and its record (%s again: at once)\n' % (name, name)).encode())
        except OSError: pass

    def stalled(self, idle):
        """In the watchdog thread: the stacks, the record and the best state, then the end of the process."""
        try:
            self.stall.write('stall watchdog: no iteration completed for %.1f minutes, after iteration %d. The stacks of all threads:\n' % (idle / 60, self.it))
            self.stall.flush(); faulthandler.dump_traceback(file=self.stall, all_threads=True); self.stall.flush()
        except Exception: pass
        try:
            self.finish(dict(reason='stalled', iteration=self.it, detail='no iteration completed for %.1f minutes after iteration %d (--stall-minutes %g): '
                                                                          'a hung CUDA kernel or I/O; the stacks of its threads are in stall.txt' % (idle / 60, self.it, self.a.stall_minutes)), watchdog=True)
            self.tell('\nSTALLED: no iteration for %.1f minutes after iteration %d; the record is written, the process ends' % (idle / 60, self.it))
        finally:
            os._exit(3)

    # ---- the wrapped training_report()
    def wrap(self, orig):
        sig = inspect.signature(orig)
        def report(*args, **kw):
            b = sig.bind(*args, **kw).arguments; it = self.it = b['iteration']; self.beat()
            if it > 1 and it - 1 in self.save: self.disk.setdefault(it - 1, 'scheduled')    # graphdeco saved it at the end of iteration it - 1
            loss = float(b['loss'].item()); self.acc[0] += loss; self.acc[1] += float(b['Ll1'].item()); self.acc[2] += 1
            self.rules(it, loss, (b['scene'], b['renderFunc'], b['renderArgs'], bool(b.get('train_test_exp', False))))
            return orig(*args, **kw)
        return report

    def rules(self, it, loss, ctx):
        """At graphdeco's iteration it, before its save, densification and optimizer step: the safety rules (NaN / Inf,
        resources, time, signals), then the judgement (blow-up; at a row, not learning and plateau). halt() ends the run."""
        a, g = self.a, ctx[0].gaussians; n = self.n = int(g.get_xyz.shape[0]); row = it == 1 or it in self.at
        if not math.isfinite(loss):
            self.halt(it, 'diverged: NaN/Inf', 'the training loss was %s at iteration %d' % (loss, it), ctx, evaluate=False, save=False)
        if it == 1 or it % a.check_every == 0 or row or it in self.save:
            bad = self.nonfinite(g)
            if bad: self.halt(it, 'diverged: NaN/Inf', 'NaN or Inf in the Gaussians\' parameters at iteration %d: %s' % (it, bad), ctx, evaluate=False, save=False)
        if it == 1: self.begin(ctx)
        self.memory(it, ctx)
        if n > a.max_gaussians:
            self.halt(it, 'limit: Gaussians', '%d Gaussians at iteration %d, more than --max-gaussians %d' % (n, it, a.max_gaussians), ctx)
        m = (time.monotonic() - self.t0) / 60
        if m > a.max_minutes:
            self.halt(it, 'limit: time', '%.1f minutes at iteration %d, past --max-minutes %g' % (m, it, a.max_minutes), ctx)
        if self.signal: self.halt(it, 'signal: ' + self.signal, 'stopped by %s at iteration %d' % (self.signal, it), ctx)
        self.blowup(it, loss, ctx)
        if row and it > 1:
            p = self.evaluate(it, ctx); self.learning(it, ctx); self.plateau(it, p, ctx)

    def halt(self, it, reason, detail, ctx, evaluate=True, save=True):
        """Stop at graphdeco's iteration it (the state after it - 1 steps; at it = 1 that is iteration 0, already saved):
        the last row, the state as the final snapshot if it is sound and `save` is set, the reason; then end the loop."""
        k = it if it > 1 else 0
        if k and k not in self.rows_at:
            if evaluate: self.evaluate(k, ctx)
            else: self.row(k, None, None, self.n)
        if save and k and k not in self.disk:
            t = time.monotonic(); A = self.kept if self.best and self.best[1] == k else host(ctx[0].gaussians)
            if finite(A): self.snapshot(k, A, 'final')
            else: reason, detail = 'diverged: NaN/Inf', detail + '; the Gaussians hold NaN or Inf there, so that state is not saved'
            self.t_eval += time.monotonic() - t
        self.stop = dict(reason=reason, iteration=k, detail=detail)
        say('\n[ITER %d] STOP, %s: %s' % (it, reason, detail))
        raise Stop(reason)

    def begin(self, ctx):
        scene, t = ctx[0], time.monotonic(); test = scene.getTestCameras()
        self.which = 'held-out' if test else 'training'
        self.run.update(train_views=len(scene.getTrainCameras()), test_views=sorted(c.image_name for c in test),
                        gaussians_initial=int(scene.gaussians.get_xyz.shape[0]))
        try: self.total = int(self.torch.cuda.get_device_properties(0).total_memory)
        except Exception: self.total = None
        if 0 in self.save:
            say('\n[ITER 0] Saving Gaussians: the initialisation from the SfM points, before the first optimizer step'); scene.save(0); self.disk[0] = 'initialisation'
        self.t_eval += time.monotonic() - t
        self.evaluate(0, ctx)

    # ---- rows of the log, and the best state
    def evaluate(self, k, ctx):
        """The row at k with the PSNR of the held-out and of the training photographs; a new best is copied to host memory."""
        scene, rf, ra, tte = ctx; t = time.monotonic(); self.beat()
        pt, pr = self.psnr(scene.getTestCameras(), scene, rf, ra, tte), self.psnr(scene.getTrainCameras(), scene, rf, ra, tte)
        self.row(k, pt, pr, int(scene.gaussians.get_xyz.shape[0]))
        p = self.psnr_at[k] = pt if pt is not None else pr
        if p is not None and (self.best is None or p > self.best[0]):
            A = host(scene.gaussians)
            if finite(A): self.best, self.kept = (p, k), A
        self.t_eval += time.monotonic() - t; self.beat()
        self.run['progress'] = dict(iteration=k, minutes=round((time.monotonic() - self.t0) / 60, 2), gaussians=self.n,
                                    best_psnr=self.best and round(self.best[0], 4), best_iteration=self.best and self.best[1])
        self.dump()
        return p

    def psnr(self, cams, scene, render, rargs, tte):
        """Mean PSNR over cams, rendered as graphdeco's training_report() renders them (clamped to 0..1); with
        --keep-alpha, over each photograph's mask (alpha-weighted)."""
        if not cams: return None
        T, total = self.torch, 0.0
        for cam in cams:
            img = T.clamp(render(cam, scene.gaussians, *rargs)['render'], 0.0, 1.0)
            gt = T.clamp(cam.original_image.to('cuda'), 0.0, 1.0)
            m = cam.alpha_mask.to('cuda') if self.a.keep_alpha else None
            if tte:
                h = img.shape[-1] // 2; img, gt = img[..., h:], gt[..., h:]; m = None if m is None else m[..., h:]
            d = (img - gt) ** 2
            mse = float(d.mean()) if m is None else float((d * m).sum()) / max(3.0 * float(m.sum()), 1e-12)
            total += -10.0 * math.log10(max(mse, 1e-20))
        return total / len(cams)

    def row(self, k, pt, pr, n):
        s, l1, c = self.acc; self.acc = [0.0, 0.0, 0]
        self.csv.writerow([k, '%.6g' % (s / max(c, 1)), '%.6g' % (l1 / max(c, 1)), '' if pt is None else '%.4f' % pt,
                           '' if pr is None else '%.4f' % pr, int(n), '%.1f' % (time.monotonic() - self.t0 - self.t_eval)])
        self.f.flush(); self.rows += 1; self.rows_at.add(k)

    def snapshot(self, k, A, what):
        self.beat(); write_ply(os.path.join(self.out, 'point_cloud', 'iteration_%d' % k, 'point_cloud.ply'), A); self.beat()
        self.disk[k] = what
        self.tell('[ITER %d] Saving Gaussians: %s' % (k, 'the state where the run stopped' if what == 'final' else 'the best %s state, from host memory' % self.which))

    def tell(self, msg):                                   # the watchdog writes straight to stderr: the hung main thread may hold stdout's lock
        if threading.current_thread() is threading.main_thread(): return say(msg)
        try: os.write(2, (msg + '\n').encode())
        except OSError: pass

    def keep_best(self):
        """The best evaluated state goes to disk unless a snapshot there comes within --min-delta of its PSNR."""
        if not self.best or self.kept is None or self.best[1] in self.disk: return False
        p, b = self.best
        if any(self.psnr_at.get(k) is not None and self.psnr_at[k] >= p - self.a.min_delta for k in self.disk): return False
        self.snapshot(b, self.kept, 'best'); return True

    # ---- the rules
    def nonfinite(self, g):
        """'' if the Gaussians' parameters are all finite, else which hold NaN or Inf. A sum per tensor on the GPU (NaN or
        Inf anywhere in a tensor makes its sum NaN or Inf), and only when one is not finite, a count per Gaussian."""
        T = self.torch
        if bool(T.isfinite(T.stack([getattr(g, k).detach().sum() for k in PARAMS])).all()): return ''
        bad = []
        for k in PARAMS:
            m = ~T.isfinite(getattr(g, k).detach()); c = int(m.reshape(m.shape[0], -1).any(1).sum())
            if c: bad.append('%s in %d Gaussians' % (k.strip('_'), c))
        return ', '.join(bad)

    def memory(self, it, ctx):
        T = self.torch
        try: m = int(T.cuda.max_memory_allocated()); T.cuda.reset_peak_memory_stats()
        except Exception: return
        self.peak = max(self.peak, m)
        if self.total and m > self.a.max_memory * self.total:
            self.halt(it, 'limit: GPU memory', 'the CUDA memory allocated peaked at %.2f GB by iteration %d, above %g of the GPU\'s %.2f GB (--max-memory)'
                      % (m / 2 ** 30, it, self.a.max_memory, self.total / 2 ** 30), ctx, evaluate=False)

    def blowup(self, it, loss, ctx):
        """The training loss, smoothed, against its best: too high for too long is a blow-up, except right after
        graphdeco's opacity resets, whose spikes are on purpose (the reset makes nearly every Gaussian see-through)."""
        a = self.a; self.ema = loss if self.ema is None else self.ema + (loss - self.ema) / a.blowup_smooth
        if any(r < it <= r + a.reset_grace for r in self.resets): self.high = 0; return
        if self.low is None or self.ema < self.low: self.low, self.low_it = self.ema, it
        self.high = self.high + 1 if self.ema > a.blowup_factor * self.low else 0
        if self.high >= a.blowup_window and not a.no_early_stop:
            self.halt(it, 'diverged: loss blow-up', 'the training loss, smoothed over about %d iterations, stayed above %g times its best (%.4g at iteration %d) '
                      'for %d iterations; %.4g at iteration %d' % (a.blowup_smooth, a.blowup_factor, self.low, self.low_it, self.high, self.ema, it), ctx, save=False)

    def learning(self, it, ctx):
        """Not learning: once, at the first row at or after --learn-by, the best PSNR so far against iteration 0's."""
        a, p0 = self.a, self.psnr_at.get(0)
        if self.learnt is not None or it < a.learn_by or p0 is None: return
        best = max([v for k, v in self.psnr_at.items() if 0 < k <= it and v is not None] or [p0])
        self.learnt = gain = best - p0
        self.run['learning'] = dict(iteration=it, psnr_0=round(p0, 4), best=round(best, 4), gain=round(gain, 4), margin=a.learn_margin)
        if gain < a.learn_margin and not a.no_early_stop:
            self.halt(it, 'not learning', 'by iteration %d the %s PSNR rose %.2f dB above iteration 0\'s %.2f dB, less than --learn-margin %g'
                      % (it, self.which, gain, p0, a.learn_margin), ctx)

    def plateau(self, it, p, ctx):
        """Converged: at the evaluations after densification has ended, --patience of them in a row without a gain of
        --min-delta over the best (the reference moves only on such a gain, so a slow creep still counts once it adds up)."""
        a = self.a
        if it not in self.evals or it <= self.until or p is None: return
        if self.ref is None:
            prev = [(v, k) for k, v in self.psnr_at.items() if k < it and v is not None]
            self.ref, self.ref_it = max(prev) if prev else (p, it)
        if p > self.ref + a.min_delta: self.ref, self.ref_it, self.wait = p, it, 0
        else: self.wait += 1
        if self.wait >= a.patience and not a.no_early_stop:
            self.halt(it, 'converged: plateau', 'after densification ended at %d, %d evaluations in a row (one every %d iterations) gained less than %g dB over '
                      'the %s PSNR of %.3f dB at iteration %d' % (self.until, self.wait, a.eval_every, a.min_delta, self.which, self.ref, self.ref_it), ctx)

    # ---- the record
    def finish(self, stop, watchdog=False):
        """The stop record, then the best state if no snapshot comes close to it, then run.json again with the list."""
        self.stop = stop; m = (time.monotonic() - self.t0) / 60
        if not watchdog and self.acc[2] and self.it > 1 and self.it not in self.rows_at:
            try: self.row(self.it, None, None, self.n or 0)    # where a crash or a second signal ended it: the loss, no evaluation
            except Exception: pass
        self.run['stop'] = dict(reason=stop['reason'], iteration=stop['iteration'], minutes=round(m, 2), best_psnr=self.best and round(self.best[0], 4),
                                best_iteration=self.best and self.best[1], detail=stop['detail'])
        try: peak = max(self.peak, int(self.torch.cuda.max_memory_allocated()))
        except Exception: peak = self.peak
        self.run.update(finished=utc(), minutes=round(m, 2), minutes_logging=round(self.t_eval / 60, 2), log_rows=self.rows, gaussians_final=self.n,
                        peak_gpu_memory_gb=round(peak / 2 ** 30, 2))
        self.listing(); self.dump()                       # the record first: a slow disk must not cost it
        try:
            if self.keep_best(): self.listing(); self.dump()
        except Exception as e:
            self.run['stop']['detail'] += '; the best state could not be written (%s: %s)' % (type(e).__name__, e); self.dump()
        if not watchdog: self.f.close()

    def listing(self):
        self.run['snapshots'] = [dict(iteration=k, kind=self.disk[k], psnr=None if self.psnr_at.get(k) is None else round(self.psnr_at[k], 4)) for k in sorted(self.disk)]


def main():
    argv, extra = sys.argv[1:], []
    if '--' in argv: i = argv.index('--'); argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=' '.join(__doc__.split('\n\n')[0].split()), formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 2)[2] + '\n\nAfter --: options of graphdeco\'s ModelParams / OptimizationParams / '
                                        'PipelineParams (e.g. --data_device cpu); not -s, -m, --eval, --images or --iterations.')
    ap.add_argument('--gs', required=True, help='a checkout of graphdeco-inria/gaussian-splatting with its submodules installed (pinned: %s)' % PIN[:7])
    ap.add_argument('--data', required=True, help='the scan folder: images/ and sparse/0/ (Y1ue/DTU on Hugging Face, DTU/scan122)')
    ap.add_argument('--out', required=True, help='the run folder (graphdeco\'s -m); it must not hold an earlier run\'s snapshots')
    ap.add_argument('--iterations', type=int, default=30000, help='the iteration cap; default 30000, graphdeco\'s own')
    ap.add_argument('--save', default=','.join(map(str, SAVE)), help='iterations to keep the Gaussians at; 0 = the initialisation; the last is always kept (default %(default)s)')
    ap.add_argument('--all-views', action='store_true', help='train on every photograph (default: every 8th held out, graphdeco\'s --eval)')
    ap.add_argument('--keep-alpha', action='store_true', help='train on the RGBA photographs as they are (graphdeco then masks its render with alpha)')
    ap.add_argument('--dry-run', action='store_true', help='check the checkout, the packages and the data, print the plan, train nothing')
    st = ap.add_argument_group('when it stops', 'tools/train/README.md, "When it stops", says what each rule watches and why its default')
    for flag, default, text in STOPS: st.add_argument(flag, type=type(default), default=default, help='%s (default %s)' % (text.replace('%', '%%'), default))
    st.add_argument('--no-early-stop', action='store_true', help='turn off the judgement rules (plateau, not learning, blow-up); never the safety ones')
    a = ap.parse_args(argv)
    for flag, default, text in STOPS:
        if not getattr(a, flag[2:].replace('-', '_')) > 0: ap.error('%s must be above 0' % flag)
    if a.max_memory > 1: ap.error('--max-memory is a fraction of the GPU\'s memory, at most 1')
    if a.blowup_factor <= 1: ap.error('--blowup-factor must be above 1')
    gs, data, out = (os.path.abspath(os.path.expanduser(p)) for p in (a.gs, a.data, a.out))
    try: save = sorted({int(v) for v in a.save.split(',') if v.strip()})
    except ValueError: ap.error('--save takes comma-separated iterations')
    late = [s for s in save if s > a.iterations]; save = sorted({s for s in save if 0 <= s <= a.iterations} | {a.iterations})   # the last, as graphdeco's train.py
    rgb = None if a.keep_alpha else os.path.join(out, 'images_rgb')
    gargs = ['-s', data, '-m', out, '--iterations', str(a.iterations)] + ([] if a.all_views else ['--eval']) + (['--images', rgb] if rgb else []) + extra
    names, facts, warn, bad = check(gs, data)
    pc = os.path.join(out, 'point_cloud')
    old = sorted((d for d in os.listdir(pc) if re.fullmatch(r'iteration_\d+', d)), key=lambda d: int(d[10:])) if os.path.isdir(pc) else []
    if old: bad.append('%s already holds snapshots of an earlier run (%s): its export would mix two runs. Use another --out, or move point_cloud/ away'
                       % (pc, ' '.join(d[10:] for d in old[:8]) + (' ...' if len(old) > 8 else '')))
    test = [] if a.all_views else [n for i, n in enumerate(sorted(names)) if i % HOLD == 0]
    head, dirty, subs = git(gs, 'rev-parse', 'HEAD'), git(gs, 'status', '--porcelain', '--untracked-files=no'), git(gs, 'submodule', 'status')
    sched, known = schedule(gs, gargs); resets = opacity_resets(sched, a.iterations); which = 'training' if a.all_views else 'held-out'

    say('graphdeco    %s: %s' % (gs, ('HEAD %s, %s' % (head[:7], 'the pinned commit' if head == PIN else 'NOT the pinned %s' % PIN[:7])) if head else 'not a git checkout: version unknown'))
    if dirty: say('             local changes: %s' % '; '.join(l.strip() for l in dirty.splitlines()))
    for l in (subs or '').splitlines():
        say('             submodule %s' % l.strip())
        if l[:1] in ('-', '+') and 'SIBR' not in l:
            warn.append('submodule %s %s' % (l.split()[1], 'is not checked out: its package must come from elsewhere' if l[0] == '-'
                                             else 'is not at the commit the checkout pins (git submodule update)'))
    say('data         %s' % data)
    for f in facts: say('             %s' % f)
    say('photographs  %s' % ('RGB copies in %s (alpha dropped)' % rgb if rgb else 'as they are: graphdeco masks its render with their alpha'))
    say('views        %s' % ('all %d for training' % len(names) if a.all_views else
                             'held out (every %dth by name): %s; %d for training' % (HOLD, ' '.join(test), len(names) - len(test))))
    say('iterations   %d; Gaussians kept at %s%s' % (a.iterations, ', '.join(map(str, save)), '; beyond the run, skipped: %s' % late if late else ''))
    say('log          log.csv, %d rows: %s' % (len(log_points(a.iterations, save, a.eval_every, [a.learn_by])), ', '.join(COLUMNS)))
    say('graphdeco    training(...) with: %s' % ' '.join(gargs))
    say('schedule     densification from %d to %d; %s%s' % (sched['densify_from_iter'], sched['densify_until_iter'],
        'opacity resets at the end of iteration %s' % ' '.join(map(str, resets)) if resets else 'no opacity reset within the run',
        '' if known else ' (graphdeco\'s defaults: its arguments/ could not be read)'))
    say('stops        safety: %d iterations, %g min, NaN / Inf (the loss every iteration, the Gaussians every %d and at each row and save), %d Gaussians, '
        '%g of the GPU\'s memory, no iteration for %g min, SIGINT / SIGTERM'
        % (a.iterations, a.max_minutes, a.check_every, a.max_gaussians, a.max_memory, a.stall_minutes))
    if a.no_early_stop: say('             judgement: OFF (--no-early-stop): no plateau, not-learning or blow-up stop')
    else:
        say('             plateau: after %d, %d evaluations in a row (every %d) without +%g dB over the best %s PSNR' % (sched['densify_until_iter'], a.patience, a.eval_every, a.min_delta, which))
        say('             not learning: by %d, less than +%g dB over iteration 0' % (a.learn_by, a.learn_margin))
        say('             blow-up: the loss smoothed over ~%d iterations above %gx its best for %d iterations, not within %d after an opacity reset'
            % (a.blowup_smooth, a.blowup_factor, a.blowup_window, a.reset_grace))
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
    if gs not in sys.path: sys.path.insert(0, gs)          # graphdeco's own modules: scene, utils, arguments, gaussian_renderer
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
    sched = dict(densify_from_iter=ga.densify_from_iter, densify_until_iter=ga.densify_until_iter, opacity_reset_interval=ga.opacity_reset_interval,
                 white_background=ga.white_background)
    run = dict(what='plain 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) on DTU scan 122 with graphdeco\'s implementation and its default '
                    'settings, for the talk\'s pipeline chapter. Not the thesis method.',
               implementation='graphdeco-inria/gaussian-splatting', url='https://github.com/graphdeco-inria/gaussian-splatting',
               commit=head, pinned=PIN, matches_pin=head == PIN, local_changes=bool(dirty), changes=(dirty or '').splitlines(),
               submodules=[l.strip() for l in (subs or '').splitlines()], python=sys.version.split()[0], torch=torch.__version__,
               cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), data=data, keep_alpha=a.keep_alpha,
               images='RGB copies of images/ (alpha dropped)' if rgb else 'images/ as they are (RGBA: graphdeco masks its render with alpha)',
               iterations=a.iterations, save=save, holdout=0 if a.all_views else HOLD, extra_args=extra, graphdeco_args=gargs, started=utc(),
               stops=dict({k[2:].replace('-', '_'): getattr(a, k[2:].replace('-', '_')) for k, _, _ in STOPS}, early_stop=not a.no_early_stop,
                          psnr=which, densify_until_iter=sched['densify_until_iter'], opacity_resets=opacity_resets(sched, a.iterations)),
               log=dict(file='log.csv', rows='0 = the initialisation, evaluated at the start of iteration 1 before the first optimizer step; '
                                             'k = inside graphdeco\'s iteration k, where it also saves (the state after k - 1 steps); every %d from %d '
                                             'are the evaluations the stop rules read; the last row is where the run stopped' % (a.eval_every, a.eval_every),
                        columns=dict(loss='training loss as graphdeco computes it, 0.8 L1 + 0.2 (1 - SSIM), mean over the iterations since the previous row',
                                     l1='its L1 part, the same mean',
                                     psnr_test='mean PSNR (dB) of the held-out photographs, rendered by graphdeco\'s render() at that iteration and '
                                               'clamped to 0..1 as its training_report() does' + (', over the alpha mask' if a.keep_alpha else ''),
                                     psnr_train='the same over the training photographs', gaussians='Gaussians at that iteration',
                                     seconds='wall-clock seconds since training() was called, without the time this log, its host copies and the '
                                             'wrapper\'s own saves took')))
    R = Recorder(out, save, run, a, sched, torch)
    G.training_report = R.wrap(G.training_report)
    safe_state(False)                                      # graphdeco's seeds (random, numpy, torch: 0), cuda:0 and its timestamped console
    R.start(); err = None
    try:
        G.training(lp.extract(ga), op.extract(ga), pp.extract(ga), [a.iterations], [s for s in save if s > 0], [], None, -1)
        R.disk.setdefault(a.iterations, 'scheduled')
        stop = dict(reason='done: iteration cap', iteration=a.iterations, detail='all %d iterations ran; no rule stopped the run' % a.iterations)
    except Stop:
        stop = R.stop
    except KeyboardInterrupt as e:
        stop = dict(reason='signal: ' + (R.signal or 'SIGINT'), iteration=R.it, detail='%s: stopped at once, so the state at the stop is not saved' % (e or 'interrupted'))
    except BaseException as e:
        err, oom = e, isinstance(e, getattr(torch.cuda, 'OutOfMemoryError', ())) or 'out of memory' in str(e).lower()
        stop = dict(reason='crashed: out of GPU memory' if oom else 'crashed: %s' % type(e).__name__, iteration=R.it,
                    detail='%s at iteration %d: %s; the state at the crash is not saved (it may have come in the middle of a step)' % (type(e).__name__, R.it, str(e).strip()[:300]))
    R.finish(stop); R.end()
    s = R.run['stop']
    say('\nstopped      %s, at iteration %d of %d after %.1f min (%.1f of them for the log and the wrapper\'s saves)'
        % (s['reason'], s['iteration'], a.iterations, s['minutes'], R.run['minutes_logging']))
    say('             %s' % s['detail'])
    if R.best: say('best         %s PSNR %.2f dB at iteration %d' % (R.which, R.best[0], R.best[1]))
    say('snapshots    %s' % ' '.join('%d%s' % (k, {'final': ' (where it stopped)', 'best': ' (the best state)'}.get(w, '')) for k, w in sorted(R.disk.items())))
    say('memory       peak GPU memory allocated %s GB; %d Gaussians at the end' % (R.run.get('peak_gpu_memory_gb', '?'), R.n or 0))
    say('next: python3 -I tools/train/export.py %s' % out)
    if err is not None and not s['reason'].endswith('memory'): raise err
    sys.exit(exit_code(s['reason']))


if __name__ == '__main__':
    main()
