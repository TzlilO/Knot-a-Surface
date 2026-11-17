# tools/train: one real run of plain 3DGS on the owl (DTU scan 122)

The pipeline chapter's `train` scene (`talk/sims/pipeline.js`) replays snapshots of a real 3D Gaussian Splatting run
on the owl, with its loss and PSNR. Training needs an NVIDIA GPU, which the sessions that build the talk do not have, so
the run happens on your machine:

- `train3dgs.py` trains with graphdeco-inria/gaussian-splatting and keeps snapshots of the Gaussians and a log.
- `export.py` turns them into `talk/dtu/train122.js`.

Until that file exists, the scene says the run is pending.

**What the result is:** plain 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) with its authors' implementation and
default settings. It trains on 56 of the scan's 64 photographs (every 8th held out) from the dataset's own COLMAP model
and points.

**What it is not:** it is not the thesis method, and nothing of the spline charts is in it. It is not tuned for DTU, and
it is not the offline fit in `talk/dtu-data.js`.

## Short version

```bash
# 1. the implementation, pinned (with the submodules that commit pins)
git clone https://github.com/graphdeco-inria/gaussian-splatting ~/gaussian-splatting
cd ~/gaussian-splatting && git checkout 54c035f7834b564019656c3e3fcc3646292f727d
git submodule update --init --recursive submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim

# 2. an environment: Linux, an NVIDIA driver for CUDA 12.4 or newer (nvidia-smi shows it), conda
conda create -y -n gs3d python=3.10 && conda activate gs3d
conda install -y -c nvidia/label/cuda-12.4.1 cuda-toolkit && export CUDA_HOME=$CONDA_PREFIX   # nvcc, for the extensions
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install numpy==1.26.4 plyfile==1.0.3 tqdm==4.66.5 opencv-python==4.10.0.84 joblib==1.4.2 pillow==10.4.0
pip install --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim

# 3. the data: the 64 photographs and the COLMAP model as text, about 190 MB
B=https://huggingface.co/datasets/Y1ue/DTU/resolve/main/DTU/scan122; D=~/dtu/DTU/scan122
mkdir -p $D/images $D/sparse/0
for f in cameras.txt images.txt points3D.txt; do curl -sSL -o $D/sparse/0/$f $B/sparse/0/$f; done
for i in $(seq 0 63); do n=$(printf %04d $i); curl -sSL -o $D/images/$n.png $B/images/$n.png; done

# 4. train, from a checkout of this repo
cd ~/Knot-a-Surface
python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122 --dry-run
python3 -I tools/train/train3dgs.py --gs ~/gaussian-splatting --data ~/dtu/DTU/scan122 --out ~/runs/train122
#    it stops by itself: at 30,000 iterations, or earlier when a rule of "When it stops" fires; either way it exports

# 5. export (numpy only, no GPU), then commit talk/dtu/train122.js
python3 -I tools/train/export.py ~/runs/train122          # writes talk/dtu/train122.js of this checkout
```

The dry run checks the following, then prints the plan without training anything:

- the checkout and its commit, and the submodules;
- the Python packages (without importing them);
- the photographs and the COLMAP model;
- that the run folder holds no snapshots of an earlier run;
- graphdeco's densification schedule, as its own parser reads it, and the stop rules with their thresholds.

Fix what it reports first. To train on one machine and export on another, copy the run folder, or at least
`point_cloud/`, `log.csv`, `run.json`, `cameras.json` and `input.ply`.

The page has to load the file before `talk/sims/pipeline.js`. In `talk/sims/pipeline.html`, that script tag is commented
out until the file exists.

## Expected time and memory (estimates: nothing was trained where this was written)

- **Time:** 30,000 iterations on 56 photographs of 1554 x 1162 should take roughly 15 to 30 minutes on an RTX
  3090/4090-class GPU, and longer on smaller ones. The log adds 145 evaluations of 64 renders each, roughly 1 to 3
  minutes, and loading the photographs about a minute. `run.json` records the real minutes, the share the log took, and
  the peak GPU memory. Whatever happens, the run stops itself after 120 minutes (`--max-minutes`).
- **GPU memory:** about 4 to 8 GB at the peak, so an 8 GB card should do.
  - graphdeco keeps every photograph on the GPU: about 1.9 GB for 64 RGB images plus their masks, as float32.
  - Then come the Gaussians and their optimizer state, about 1 KB each with SH degree 3.
  - The rasterizer's buffers come on top.

  If memory runs out, add `-- --data_device cpu` to keep the photographs in RAM. The run stops itself, with a snapshot,
  once the memory allocated passes 85 % of the GPU's (`--max-memory`).
- **Gaussians:** they start from the dataset's 21,760 SfM points. A 30,000-iteration DTU run typically ends with a few
  hundred thousand; my guess for this one is 0.3 to 0.8 million (an estimate, not measured).

## Why graphdeco's implementation, and which version

graphdeco-inria/gaussian-splatting rather than gsplat's `simple_trainer.py`, for four reasons:

- It is the paper's reference implementation, so its default settings are what "plain 3DGS" means.
- Its PLY is the layout the exporter reads: `x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`, with
  log-scales, logit opacity and quaternion `w x y z`.
- It keeps the Gaussians in the COLMAP frame of the model it was given, so `export.py` can map them into the owl's frame
  exactly as `tools/dtu/pipeline.py` maps that model. gsplat's trainer normalises the world (it re-centres, rescales and
  rotates the scene) by default, which would be one more transform to undo, and it saves `.pt` checkpoints unless asked
  for PLY.
- It reads COLMAP text models directly and has `--eval` and per-iteration saves built in.

Nothing of it is copied or changed. The wrapper imports its `train.py`, wraps `training_report()` and calls
`training()`.

**Pinned versions:**

| What | Pin |
| --- | --- |
| graphdeco-inria/gaussian-splatting | commit `54c035f` (2024-10-30), the head of `main` when this was written |
| diff-gaussian-rasterization | `9c5c202` (branch `dr_aa`), as that commit pins it |
| simple-knn | `86710c2` |
| fused-ssim | `1272e21` |
| Python | 3.10 |
| PyTorch | 2.4.1 built for CUDA 12.4 |
| CUDA toolkit | 12.4 (nvcc), to compile the three extensions |

The major CUDA versions of nvcc and PyTorch must match. The implementation's own `environment.yml` pins Python 3.7,
PyTorch 1.12.1 and CUDA 11.6, which recent GPUs and compilers no longer build; its README reports that newer
environments work. If the run is not at the pinned commit, `run.json` and the exported note say so.

The implementation's licence is the Gaussian-Splatting License (Inria and MPII): non-commercial research and evaluation
use.

**If the build fails:**

- **GCC 13 or newer (Ubuntu 24.04).** The rasterizer at `9c5c202` uses `uint32_t` without including `<cstdint>` (checked
  in its source). Add it, then build again:
  `sed -i 's/#include <iostream>/#include <iostream>\n#include <cstdint>/' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
- **simple-knn stops at "FLT_MAX is undefined".** Add `#include <cfloat>` at the top of
  `submodules/simple-knn/simple_knn.cu`. Its source sits on gitlab.inria.fr, which could not be read from here, so this
  is the community's fix, not checked against that commit.
- **"No module named torch" while building.** The `--no-build-isolation` flag is missing.
- **nvcc not found, or the wrong version.** Set `CUDA_HOME` to the toolkit: `$CONDA_PREFIX` for the conda one, or
  `/usr/local/cuda-12.4`. If the conda toolkit's headers are not found, also run
  `export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH`.
- **RTX 50xx (Blackwell).** These cards need CUDA 12.8: the `nvidia/label/cuda-12.8.1` toolkit and
  `torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128`.
- **Windows.** graphdeco supports it with the Visual Studio 2019 build tools; type `python` for `python3`. This was not
  tried.

These fixes only make the code build. The dry run lists them as local changes, and so do `run.json` and the exported
note.

## The data

Get `DTU/scan122/images/0000.png .. 0063.png` and `DTU/scan122/sparse/0/{cameras,images,points3D}.txt` from the Hugging
Face dataset repo `Y1ue/DTU`. This is the same COLMAP model `tools/dtu/pipeline.py` reads. Its `DTU/README.md` says the
SfM points were made "with COLMAP given GT camera poses (cameras.npz)".

Leave out the `.bin` files and `points3D.ply` of that folder:

- graphdeco would read `cameras.bin` / `images.bin` before the `.txt` files. Those were not compared here, so the dry run
  warns if they are present.
- The Hub's `points3D.ply` holds the same 21,760 points and colours as `points3D.txt`, in another order (checked).
  graphdeco writes its own `points3D.ply` from the `.txt` on its first run.

**The photographs are RGBA.** The alpha channel is the object's mask: 71 % of `0000.png` is fully transparent. The RGB
values under the mask are the real background, not black. graphdeco multiplies its render by alpha in the loss. On these
files that would mean two things:

- The background is never learnt.
- The held-out PSNR compares against a background the model never saw.

Plain 3DGS sees the whole photograph, so the wrapper trains on RGB copies in `<run>/images_rgb`. PIL drops the channel,
and the RGB values are kept exactly (checked). `--keep-alpha` trains on the files as they are; the PSNR is then taken
inside the mask.

## What `train3dgs.py` does

It calls graphdeco's `training()` in-process with the implementation's defaults: 30,000 iterations, densification from
500 to 15,000, opacity resets every 3,000, SH degree 3, black background.

- **Held-out photographs.** graphdeco's `--eval` holds out every 8th photograph by name: `0000 0008 ... 0056` (8), which
  leaves 56 to train on. `--all-views` trains on all 64.
- **Snapshots.** It keeps the Gaussians at 0, 100, 300, 1000, 3000, 7000, 15000 and 30000 (`--save`); the last iteration
  is always kept.
  - Iteration 0 is the initialisation from the SfM points, before the first optimizer step. The wrapper saves it at
    the start of iteration 1.
  - Every other iteration k is graphdeco's own save at k: the state after k - 1 steps, saved before that iteration's
    densification. So 3000 is the state just before the opacity reset.
  - A run that stops early has the snapshots of the schedule it reached, plus a final one where it stopped and,
    if it is clearly better than all of those, its best state (see "When it stops"). Those two are written by the
    wrapper, in graphdeco's PLY layout, under the same convention: k is the state after k - 1 steps.
- **`log.csv`.** It has a row at 0, every 10 iterations to 100, every 50 to 1,000, then every 250 (`--eval-every`):
  145 rows. The rows every 250 are the evaluations the stop rules read, and the last row is where the run stopped. The
  columns are:
  - `loss`: the training loss as graphdeco computes it, 0.8 L1 + 0.2 (1 - SSIM), averaged over the iterations since
    the previous row;
  - `l1`: its L1 part;
  - `psnr_test`, `psnr_train`: the mean PSNR over the held-out and the training photographs, rendered by graphdeco's
    `render()` at that iteration and clamped as its `training_report()` does;
  - `gaussians`: the number at that iteration;
  - `seconds`: training time, without the log's own (its renders, its copies to host memory, the wrapper's saves).

  Each row is flushed as it is written, so a crash leaves a usable log.
- **`run.json`.** It records the commit, local changes, submodules, Python, PyTorch, CUDA, the GPU, the data path,
  graphdeco's arguments, the views, the start and finish times, the minutes, and the peak GPU memory. Since the stop
  rules, also why and where the run stopped and which snapshots it left (see "When it stops").
- **Nothing in the steps it takes changes.** The extra renders run inside graphdeco's `no_grad` block, touch no state it
  keeps, and draw no random numbers. The stop rules only read: the loss, the Gaussians' parameters, the memory counters
  and the clock. The camera order and the result are graphdeco's, seeded as its own `train.py` seeds
  them. The CUDA rasterizer's backward pass is not bit-for-bit deterministic, so two runs differ slightly.
- **Viewer.** graphdeco's network viewer is not started, the same as its `--disable_viewer`.
- **Options for graphdeco.** Anything after `--` goes to graphdeco's own parser, for example `-- --data_device cpu`.
  The exported note lists such options.
- **Where packages must live.** `python3 -I` ignores `PYTHONPATH` and packages installed with `pip --user`, so install
  into the conda environment.

## When it stops

A long run on a GPU should not need watching, so the run judges itself. At every iteration, inside graphdeco's own
`training_report()` (after the backward pass, before graphdeco saves, densifies or steps), the wrapper checks the
rules below and ends the run when one fires. The safety rules are always on. The three judgement rules (marked *) can
be turned off with `--no-early-stop`; the run then goes to its iteration cap or to a safety rule. Every threshold is a
flag, and `--dry-run` prints them all with the schedule they apply to.

| Rule | Stops when | Default (flag) | Recorded reason | State at the stop |
| --- | --- | --- | --- | --- |
| Iteration cap | graphdeco's loop ends | 30,000 (`--iterations`) | `done: iteration cap` | graphdeco's own save |
| Time | the wall clock passes the budget, counted from the call of `training()` | 120 min (`--max-minutes`) | `limit: time` | saved |
| NaN / Inf | the loss is not finite (checked every iteration), or a parameter is not (checked every 100 iterations, at each row of the log and before each save) | 100 (`--check-every`) | `diverged: NaN/Inf` | **not** saved |
| Gaussians | there are more of them than the cap | 4,000,000 (`--max-gaussians`) | `limit: Gaussians` | saved |
| GPU memory | the CUDA memory allocated peaks above a share of the GPU's total | 0.85 (`--max-memory`) | `limit: GPU memory` | saved, from host memory |
| Out of memory | a CUDA out-of-memory error comes anyway | | `crashed: out of GPU memory` | not saved |
| Stall | no iteration completes for this long | 10 min (`--stall-minutes`) | `stalled` | not saved |
| Signal | SIGINT (Ctrl-C) or SIGTERM: the run stops at the next iteration; a second one stops it at once | | `signal: SIGINT`, `signal: SIGTERM` | saved (after a second signal: not) |
| Blow-up * | the training loss, smoothed, stays above twice its best for 500 iterations in a row, outside the 1,000 iterations after each opacity reset | 2 (`--blowup-factor`), 500 (`--blowup-window`), 100 (`--blowup-smooth`), 1000 (`--reset-grace`) | `diverged: loss blow-up` | not saved |
| Plateau * | once densification has ended (15,000), 16 evaluations in a row (one every 250 iterations: 4,000 iterations) bring no gain of 0.05 dB over the best held-out PSNR | 16 (`--patience`), 0.05 (`--min-delta`), 250 (`--eval-every`) | `converged: plateau` | saved |
| Not learning * | at the first evaluation from 1,000 on, the best held-out PSNR so far is less than 3 dB above iteration 0's | 1000 (`--learn-by`), 3 (`--learn-margin`) | `not learning` | saved |

Any other error is recorded as `crashed: <its type>`, without saving the state, and its traceback is printed. "Saved"
means the state at the stop becomes the final snapshot, `point_cloud/iteration_<k>`, k being the stop iteration. A
state that holds NaN or Inf is never saved: before any save the wrapper checks it, and if it is not finite the reason
becomes `diverged: NaN/Inf`. With `--all-views` there is no held-out photograph, and the rules read the training PSNR.

**The best state.** Whenever the held-out PSNR reaches a new best at a row of the log, the wrapper copies the Gaussians
to host memory and checks the copy is finite. When the run ends, for whatever reason, that copy is written as one more
snapshot, `point_cloud/iteration_<b>`, unless a snapshot already on disk comes within `--min-delta` of its PSNR. So a run
that diverges at 25,000 leaves its best state of, say, 24,750, not only the 15,000 of the schedule. A run that is cut
short right after an opacity reset leaves both: the dip where it stopped, and the state just before the reset.

### Why these defaults

- **Time, 120 minutes.** This README expects 15 to 30 minutes for 30,000 iterations on an RTX 3090/4090-class card, and
  longer on smaller ones. 120 minutes is four times the upper estimate, which leaves room for an 8 GB card two or three
  times slower. It still ends a run whose iterations have slowed to a crawl (a runaway densification) within the
  evening.
- **Gaussians, 4,000,000.** A DTU object typically ends with a few hundred thousand Gaussians; the estimate for this one
  is 0.3 to 0.8 million. Several million is what Mip-NeRF 360's outdoor scenes reach, with three times as many
  photographs. 4 million on one object means densification has run away. At about 1 KB per Gaussian (the parameters,
  their gradients and two Adam moments), that is also about where an 8 GB card runs out.
- **GPU memory, 0.85.** The check reads PyTorch's peak allocation (`torch.cuda.max_memory_allocated()`, reset every
  iteration), so it sees every iteration's forward and backward pass and every densification. The CUDA context and the
  allocator's fragmentation come on top of that figure, so 85 % allocated is close to the end on an 8 GB card. Memory
  grows mostly at the densifications every 100 iterations, and one of them can still overshoot what is left before
  the next check. That is why an out-of-memory error is caught as well. The final snapshot is copied to host memory
  first, which takes no GPU memory, and no last evaluation is rendered.
- **NaN / Inf, every iteration and every 100.** The loss costs nothing to check, since graphdeco reads it anyway. The
  parameters cost a sum per tensor on the GPU and one synchronisation, a fraction of a millisecond. They need their own
  check because NaN parameters do not always reach the loss: a Gaussian can be culled before it touches a pixel, and
  graphdeco's pruning compares opacity with 0.005, which a NaN never fails. The check also runs before every save, so a
  bad state never reaches a snapshot, graphdeco's own included.
- **Stall, 10 minutes.** The longest pause a healthy run makes is graphdeco's own save: its `save_ply` builds a Python
  tuple per Gaussian, measured here at 6 microseconds each, so about 25 s at 4 million. Loading the photographs before
  iteration 1 takes about a minute. Ten minutes is many times either.
- **Plateau, only after densification, 16 evaluations of 250 iterations, 0.05 dB.** Until `densify_until_iter` (15,000),
  graphdeco resets every opacity every 3,000 iterations and keeps cloning, splitting and pruning. The held-out PSNR
  falls and recovers there by design, so a plateau means nothing yet. Afterwards the set of Gaussians is fixed. The
  PSNR climbs more and more slowly as the position learning rate decays, by 30,000 to 1/100 of its start. The paper's
  Table 1 shows 1.6 to 1.9 dB gained between 7,000 and 30,000 iterations on its three benchmarks, so the late gains are
  small but real. A window of 4,000 iterations keeps a run going as long as it gains 0.05 dB per 4,000 iterations,
  which is about the smallest difference worth a snapshot on 8 photographs. The earliest plateau stop is at 19,000. On a
  curve that flattens the way such curves do, the rule should fire late in the schedule, with a few hundredths of a dB
  left on the table (an estimate, not measured here). It ends early only a run that has truly flattened. The
  reference moves only on a gain of 0.05 dB, so a slow creep still counts once it adds up, and noise around a flat
  curve does not keep a run alive.
- **Not learning, +3 dB by 1,000.** Iteration 0 renders the SfM Gaussians at opacity 0.1 on black. Measured on two of
  the held-out photographs, black scores 12.3 and 17.1 dB (`0000`, `0040`), and the best constant colour 20.0 and 19.8
  dB. A run that learns at all should pass +3 dB within its first few hundred iterations. A run that does not learn
  (no gradient reaches the Gaussians, the wrong images or cameras) stays at iteration 0's level. Iteration 1,000 comes
  before the first opacity reset (3,000) and early enough to waste little. The rule does not catch a run that learns
  only the average colour, which gains up to about 6 dB here; the plateau rule and the time budget end that one.
- **Blow-up, twice the best for 500 iterations, smoothed over 100, not within 1,000 after a reset.** graphdeco's loss is
  measured on one random photograph per iteration, so it jumps from view to view. An exponential mean over about 100
  iterations, two passes over the 56 views, smooths that to a few per cent. A healthy run never climbs back to twice
  its best after its first few hundred iterations, and 500 iterations in a row is longer than any single bad view or
  densification step can hold it there. The opacity resets (at the end of 3,000, 6,000, 9,000 and 12,000, and of 500
  with `--white_background`) make nearly every Gaussian see-through at once. The loss then jumps towards that of an
  almost empty render and comes back as the opacities are relearned: Adam moves a logit by about the learning rate,
  0.025, per step, and the reset puts them at logit(0.01) = -4.6, so recovering takes on the order of a few hundred
  iterations (an estimate). The rule leaves the 1,000 iterations after each reset alone. The schedule comes from
  graphdeco's own parser, so options after `--` that change it (`--opacity_reset_interval`, `--densify_until_iter`,
  `--white_background`) move the grace windows and the start of the plateau rule with it.

### What the record says

- **`run.json`** gains:
  - `stop: {reason, iteration, minutes, best_psnr, best_iteration, detail}`: the reason as in the table, the iteration
    where it stopped, and `detail`, one sentence with the numbers that made the rule fire;
  - `snapshots: [{iteration, kind, psnr}]`, each with its kind: `initialisation`, `scheduled`, `final` (where it
    stopped) or `best` (from host memory);
  - `stops`: every threshold, whether the judgement rules were on, and the opacity resets and `densify_until_iter` they
    used;
  - `learning`: the numbers of the not-learning check;
  - `progress`: the last evaluation's iteration, minutes, Gaussians and best. It is updated at every row, so even a run
    killed from outside says how far it got.

  `run.json` is written atomically: a kill never leaves half of it.
- **`log.csv`** keeps every evaluation row; its last row is where the run stopped. After a NaN that row has the loss and
  no PSNR.
- **`stall.txt`** exists only after a stall: the stacks of all threads, showing where the main thread hung. A thread can
  only run while the main thread lets go of Python's lock, which a call into a C extension that hangs may never do. In
  that case Python's `faulthandler` timer, which needs no lock, ends the process at twice `--stall-minutes` and writes
  the stacks. `run.json` then has no stop, and `export.py` reads the stall from `stall.txt` and `progress`.
- **The exit code:**

  | Code | Meaning |
  | --- | --- |
  | 0 | done or converged |
  | 2 | stopped early by a limit or a signal |
  | 3 | stopped because the run went wrong: not learning, diverged, out of GPU memory, stalled |
  | 1 | crashed, with the traceback printed (also faulthandler's last resort) |

  A hung run handles SIGTERM only once it moves again, or when the watchdog ends it; SIGKILL always works.
- **The run folder must be fresh.** The wrapper refuses a `--out` that already holds snapshots of an earlier run, since
  its export would mix two runs; the dry run reports it.

**A run that stops early is still exportable**, whatever the reason. `export.py` takes the snapshots `run.json` lists:
those of the schedule the run reached, its final one and its best one. Snapshots of the schedule that the run never
reached are simply absent. It copies the stop into the file as `stop: {reason, iter, best_psnr, best_iter}` next to
`note`, says it in the note, and the `train` scene shows it as "the run stopped".

## What `export.py` does

**The frame.**

- The checkpoints are in the COLMAP frame. `tools/dtu/pipeline.py` maps that frame into the owl's by a similarity
  P = k R X + T, through `scale_mat_0`, fit.py's table frame and the owl's axis, with 80 mm per unit.
- `pipeline122.js` holds the 64 cameras in the owl's frame, and the run holds them in the COLMAP frame
  (`cameras.json`). From the two, `export.py` reads R off the rotations, and k and T off the centres.
- It then checks the result with something that does not involve the cameras: the 21,760 SfM points the run started
  from (`input.ply`), mapped, must pair one to one with the copy in `pipeline122.js` within that file's rounding.
- If either check fails, it writes nothing.
- `--data <scan folder>` reads `images.txt` / `points3D.txt` instead, and gives the same bytes.

On scan 122:

| Check | Result |
| --- | --- |
| scale k | 3.383896360 |
| cameras | agree within 2.2e-4 degrees and 0.002 mm |
| SfM points inside the box (21,565) | paired within 0.5008 of a 12-bit step (the rounding alone gives 0.5) |
| the 195 outside it | within 3.5e-6 units |
| iteration 0 | is exactly those points |

**Per Gaussian:**

- centre: k R x + T;
- rotation: q_R * q, with q normalised;
- sigma per axis: exp(scale) * k;
- colour: 0.5 + 0.28209479 * f_dc, clamped, in 8 bits (the DC term only, no view dependence);
- opacity: the sigmoid, in 8 bits.

Rows that are not finite, including a scale that overflows float32, are left out but counted in `ntotal`.

**Budget.** A snapshot with more than `--budget` Gaussians (default 60,000) keeps that many. They are drawn without
replacement with probability growing with opacity x (s1 s2 + s2 s3 + s3 s1), which is the Gaussian's average footprint
on screen up to a constant (Efraimidis-Spirakis; numpy's PCG64 seeded with `[seed, iteration]`). The kept ones stay in
the checkpoint's order. `--rule uniform` draws uniformly instead. The same input and seed give the same bytes.

**Storage.** The file decodes itself on load, like `talk/dtu-data.js`.

- Centres are 16-bit integers over a box per snapshot: pipeline.py's box for the SfM points (10 x 5 x 10 units, steps
  of 0.012 mm), stretched to all but the outermost 0.2 % on each side. The rest are float32.
- Sigmas are 8-bit logarithms over the snapshot's range.
- Quaternions are stored as their three smallest components at 8 bits each, plus 2 bits for the one left out.

That comes to 21.7 bytes per Gaussian in base64.

**The stop, and which snapshots.** `run.json`'s stop record goes into the file and into its note (see "When it stops").
When `run.json` lists the run's snapshots, only those are exported: one left on disk by an earlier run in the same
folder, or a save cut off, is left out and named. A `run.json` from before the stop rules (no stop, no list) exports as
it always did, without the field.

## The file

`KnotTalk.dtuTrain[122] = { note, stop, unit: 'owl frame, 1 unit = 80 mm', log: { iter, loss, psnr }, snaps: [{ iter,
ntotal, n, p, s, q, c, a }] }`:

- `stop` (optional): `{ reason, iter, best_psnr, best_iter }`, why and where the run stopped and its best held-out PSNR
  (dB, 3 decimals) and that iteration; a number not known is `null`. A file without it is valid, as before;

- `p`: Float32Array(3n), centres;
- `s`: Float32Array(3n), sigma per axis in owl units;
- `q`: Float32Array(4n), unit quaternions `w x y z`;
- `c`: Uint8Array(3n), sRGB of the DC term;
- `a`: Uint8Array(n), opacity 0..255.

The covariance is R(q) diag(s^2) R(q)^T. `log.psnr` is the held-out PSNR (the training PSNR with `--all-views`).
`note` names the implementation, commit, settings, iterations, views, GPU and date, and, from the stop record, where and
why the run stopped and which snapshots are its final and best ones.

**Size at the default budget:** every snapshot from iteration 1000 on probably holds more than 60,000 Gaussians, so the
file holds 3 x 21,760 + 5 x 60,000 = 365,280 of them: 7.95 MB.

## What was tested where this was written, and what was not

**Not run:** the training, the installation and the build. There is no GPU there, and none of it was installed.

**Tested,** with synthetic checkpoints built from the real COLMAP model of scan 122 (never in the repo):

- `export.py`. Its output loads under node in 0.31 s and keeps the contract: types, lengths, finite values and unit
  quaternions (|q| - 1 < 5e-8). Against a frame derived independently (from the camera centres alone), the decoded
  values agree as follows:

  | Quantity | Agreement |
  | --- | --- |
  | positions | within 0.52 of a 16-bit step |
  | sigmas | within half a log step |
  | rotations | within 0.93 degrees (0.66 at the 99th percentile) |
  | covariances | within 4.6 % |
  | colour and opacity | exact |

  Probes land where `pipeline122.js` puts them:
  - a Gaussian at camera 5's centre lands 0.0004 mm from it;
  - one at an SfM point lands within half a 12-bit step of its copy;
  - long axes point within 0.44 degrees of directions taken from the file's own cameras.

  A camera moved by 2.7 mm, or SfM points shifted by 0.27 mm, make it write nothing. Re-runs give identical bytes.
- `train3dgs.py`, in three ways:
  - `--help`;
  - `--dry-run` against the real checkout at the pinned commit;
  - a full run against a mock of graphdeco: its real argument parser, and stand-ins for `train.py`, `Scene` and `torch`
    with the same signatures, call order and files.

  The full run covered the RGB copies, the iteration-0 save, `log.csv`, `run.json`, the options after `--`, and then the
  export of that run.

**The stop rules, tested on the mock.** These tests use the mock's real argument parser, graphdeco's densification and
opacity-reset schedule, and a scripted loss whose opacity resets make the render nearly empty until the opacities come
back. Each run was exported, and the file was loaded under node and checked against the contract, with its `stop`. All
21 pass:

| Test | Ended | Snapshots left |
| --- | --- | --- |
| a normal run, rules on, none fires | `done: iteration cap` at 30,000 | the schedule |
| opacity-reset spikes at 3,000 ... 12,000 (the logged loss 8 to 17 times the row before) | `done: iteration cap` at 16,000 | the schedule |
| the same with `--reset-grace 1` | `diverged: loss blow-up` at 3,508 | up to 3,000 |
| the same spike off the schedule, at 4,500 | `diverged: loss blow-up` at 5,006 | up to 3,000; best 4,500 |
| plateau after densification | `converged: plateau` at 21,250 | the schedule to 15,000; final 21,250 |
| NaN loss at 7,000 (a scheduled save) | `diverged: NaN/Inf` at 7,000 | up to 3,000; best 6,000; no 7,000 |
| NaN in one Gaussian, the loss finite | `diverged: NaN/Inf` at 5,100 | up to 3,000; best 5,000 |
| a sustained blow-up from 17,000 | `diverged: loss blow-up` at 17,514 | the schedule to 15,000 (within 0.05 dB of the best) |
| not learning | `not learning` at 1,000 | final 1,000 |
| time limit, mocked clock | `limit: time` at 6,099, just after a reset | best 6,000; final 6,099 |
| Gaussian cap | `limit: Gaussians` at 3,301 | final 3,301 |
| memory cap | `limit: GPU memory` at 6,601 | best 6,000; final 6,601 |
| a CUDA out-of-memory error | `crashed: out of GPU memory` at 4,321 | best 4,250 |
| a hang (in `time.sleep`) | `stalled` at 2,599, 3.5 s with `--stall-minutes 0.05` | best 2,500; stacks in `stall.txt` |
| a hang holding Python's lock (a C call) | faulthandler ends it after 6 s; no stop in `run.json` | the export reads `stalled` from `stall.txt` |
| SIGTERM | `signal: SIGTERM` at 2,500 | final 2,500 |
| SIGINT from outside | `signal: SIGINT` | final where it stopped |
| `--no-early-stop`, not learning, NaN at 2,500 | `diverged: NaN/Inf` at 2,500, not stopped at 1,000 | up to 1,000 |
| the schedule changed after `--` | resets read as 500 2000 4000 6000 8000; plateau after 9,000 | final where it stopped |
| `--all-views` | `converged: plateau`, read on the training PSNR | final where it stopped |
| a used folder, an old run | refused; an old run exports as before, without `stop` | |

The tests also checked the following:

- every snapshot left is whole and finite, and no `.part` file remains;
- a NaN or blow-up state is never on disk;
- the last log row is where the run stopped;
- `run.json`'s list matches the disk;
- the export carries the same stop.

The wrapper's PLY writer gives byte-for-byte the file graphdeco's `save_ply` writes through `plyfile` 1.0.3, and
graphdeco's `load_ply` logic reads it back exactly.

Not tested: any of it on a GPU. The rules are tuned on the reasoning above, not on a measured run of this scene.
