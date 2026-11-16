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

# 5. export (numpy only, no GPU), then commit talk/dtu/train122.js
python3 -I tools/train/export.py ~/runs/train122          # writes talk/dtu/train122.js of this checkout
```

The dry run checks the following, then prints the plan without training anything:

- the checkout and its commit, and the submodules;
- the Python packages (without importing them);
- the photographs and the COLMAP model.

Fix what it reports first. To train on one machine and export on another, copy the run folder, or at least
`point_cloud/`, `log.csv`, `run.json`, `cameras.json` and `input.ply`.

The page has to load the file before `talk/sims/pipeline.js`. In `talk/sims/pipeline.html`, that script tag is commented
out until the file exists.

## Expected time and memory (estimates: nothing was trained where this was written)

- **Time:** 30,000 iterations on 56 photographs of 1554 x 1162 should take roughly 15 to 30 minutes on an RTX
  3090/4090-class GPU, and longer on smaller ones. The log adds 145 evaluations of 64 renders each, roughly 1 to 3
  minutes, and loading the photographs about a minute. `run.json` records the real minutes, the share the log took, and
  the peak GPU memory.
- **GPU memory:** about 4 to 8 GB at the peak, so an 8 GB card should do.
  - graphdeco keeps every photograph on the GPU: about 1.9 GB for 64 RGB images plus their masks, as float32.
  - Then come the Gaussians and their optimizer state, about 1 KB each with SH degree 3.
  - The rasterizer's buffers come on top.

  If memory runs out, add `-- --data_device cpu` to keep the photographs in RAM.
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
- **`log.csv`.** It has a row at 0, every 10 iterations to 100, every 50 to 1,000, then every 250: 145 rows. The
  columns are:
  - `loss`: the training loss as graphdeco computes it, 0.8 L1 + 0.2 (1 - SSIM), averaged over the iterations since
    the previous row;
  - `l1`: its L1 part;
  - `psnr_test`, `psnr_train`: the mean PSNR over the held-out and the training photographs, rendered by graphdeco's
    `render()` at that iteration and clamped as its `training_report()` does;
  - `gaussians`: the number at that iteration;
  - `seconds`: training time, without the log's own.

  Each row is flushed as it is written, so a crash leaves a usable log.
- **`run.json`.** It records the commit, local changes, submodules, Python, PyTorch, CUDA, the GPU, the data path,
  graphdeco's arguments, the views, the start and finish times, the minutes, and the peak GPU memory.
- **Nothing in the training changes.** The extra renders run inside graphdeco's `no_grad` block, touch no state it
  keeps, and draw no random numbers. The camera order and the result are graphdeco's, seeded as its own `train.py` seeds
  them. The CUDA rasterizer's backward pass is not bit-for-bit deterministic, so two runs differ slightly.
- **Viewer.** graphdeco's network viewer is not started, the same as its `--disable_viewer`.
- **Options for graphdeco.** Anything after `--` goes to graphdeco's own parser, for example `-- --data_device cpu`.
  The exported note lists such options.
- **Where packages must live.** `python3 -I` ignores `PYTHONPATH` and packages installed with `pip --user`, so install
  into the conda environment.

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

## The file

`KnotTalk.dtuTrain[122] = { note, unit: 'owl frame, 1 unit = 80 mm', log: { iter, loss, psnr }, snaps: [{ iter, ntotal,
n, p, s, q, c, a }] }`:

- `p`: Float32Array(3n), centres;
- `s`: Float32Array(3n), sigma per axis in owl units;
- `q`: Float32Array(4n), unit quaternions `w x y z`;
- `c`: Uint8Array(3n), sRGB of the DC term;
- `a`: Uint8Array(n), opacity 0..255.

The covariance is R(q) diag(s^2) R(q)^T. `log.psnr` is the held-out PSNR (the training PSNR with `--all-views`).
`note` names the implementation, commit, settings, iterations, views, GPU and date.

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
