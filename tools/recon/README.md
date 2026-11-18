# tools/recon — B-spline charts from structure-from-motion alone

`sfm_spline.py` fits cubic tensor B-spline charts to a COLMAP point cloud and nothing else: no depth maps, no
silhouettes, no frame given by hand. It is a **geometric fit to SfM points**. Nothing in it is trained, and its
output is not a result of the thesis method. `tools/dtu/fit.py`, which made the shipped owl, fuses the dataset's
depth maps instead. That chart is used here only as a yardstick, at the end, and never as an input.

numpy + scipy only. The pictures that `compare_dtu.py` draws also need Pillow. Always run with `python3 -I`.

```bash
python3 -I tools/recon/sfm_spline.py MODEL_DIR -o OUT [--labels labels.npy] [--unit-mm U]   # MODEL_DIR = COLMAP text model
python3 -I tools/recon/sfm_spline.py --self-test [-o DIR]                                   # synthetic scenes, ~30 s
# scan 122 from the repo's own data (talk/dtu/pipeline122.js), through the real entry:
python3 -I tools/recon/from_pipeline_js.py talk/dtu/pipeline122.js /tmp/owl/model --truth /tmp/owl/truth
python3 -I tools/recon/sfm_spline.py /tmp/owl/model -o /tmp/owl/out --unit-mm 80 --track-len /tmp/owl/model/track_length.npy
python3 -I tools/recon/compare_dtu.py /tmp/owl/out/charts.json /tmp/owl/truth/transform.json --model /tmp/owl/model --pictures /tmp/owl/pics
```

| file | what |
| --- | --- |
| `sfm_spline.py` | the tool (steps below) |
| `colmap_io.py` | COLMAP text reader / writer. Camera models: PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV. Distortion is ignored, with a printed note. |
| `bspline.py` | cubic bases with derivatives, Boehm insertion, periodic direction, sparse design matrix (the maths of `talk/nurbs.js`, vectorised) |
| `selftest.py` | `--self-test`: synthetic COLMAP models with known spline surfaces |
| `from_pipeline_js.py` | `talk/dtu/pipeline<N>.js` -> COLMAP text model (+ `track_length.npy`), in a random similarity frame |
| `compare_dtu.py` | YARDSTICK ONLY: distances to the shipped depth-fitted chart and to `pipeline.py`'s 12 x 16 sketch, plus pictures |

## Inputs

- A COLMAP **text** model: `cameras.txt`, `images.txt`, `points3D.txt`. Convert a binary model first with
  `colmap model_converter --output_type TXT`.
- `--labels labels.npy` (optional): one integer per POINT3D_ID (the array is indexed by the id). Each label >= 0
  gets its own chart; -1 means ignore. With labels, the support plane still defines "up", but it gets a chart of its
  own only with `--plane-chart always`, so label the table yourself if you want it.
- `--unit-mm`: millimetres per model unit, when the scale is known. Every length threshold is a multiple of the
  spacing `s`, the median nearest-neighbour distance of the cleaned cloud. The thresholds are therefore scale-free,
  and the report gives each length in model units and, with `--unit-mm`, in mm.
- `--track-len`: true track lengths per POINT3D_ID, for models whose tracks are incomplete (the converted
  pipeline file, see below).

## What it does

1. **Clean.** Drops points whose track has fewer than `--min-track` images (default 2) or whose mean reprojection
   error is above `--max-reproj` px (default 2.0). Then a k-NN outlier test: a robust z-score of log(mean distance to
   12 neighbours) above 3.5 MAD. Density varies by factors, so the test works on the log.
2. **Plane.** RANSAC over the points that passed step 1, before the density test (that test removes a sparse,
   texture-poor table first: on scan 122 it dropped 509 of 857 table points). The three points are drawn among a
   random point's 60 neighbours. A plane must have every camera on one side and at least 95 % of the other points
   on that side; the most inliers then wins. It is refined by least squares on its inliers, and its normal is "up".
   Its inliers become a height-field chart over the plane's box (0.5 to 99.5 % of them). If no plane is found, up is
   taken from the cameras' mean up vector, and the log says so.
3. **Objects and layout.**
   - **Which points.** Without labels: the largest connected component of the kNN graph (10 neighbours, links
     shorter than 3 x the median 10th-neighbour distance) among the points 3 s above the plane. With labels: one
     object per label.
   - **The candidates.** Each is measured on the points:
     - `cyl`: a sheet around a vertical axis, rows by height
     - `sph`: the same sheet, rows by elevation from the axis at mid height
     - `hyb`: rows by height below a shoulder (75 % of the height), elevation above it, as fit.py does
     - `height`: a height field over the object's own PCA plane, with its box rotated to the smallest area
   - **The axis.** It is vertical. It starts at the robust centroid and is then moved, on a 7 x 7 grid over the
     middle of the footprint, to where the layout score is lowest (`--axis search`; `--axis centroid` keeps the
     centroid, as originally specified). The centroid of a shell seen from one side lies near its seen surface; on
     scan 122 the search lowered the score from 0.298 to 0.213.
   - **Azimuth range.** The widest run of 5-degree azimuth bins whose points cover fewer than 30 % of 30 row bins.
     The sheet is open across that gap, or closed all round (a periodic direction) when the gap is under 20 degrees.
     Rows run from the foot to the top per azimuth (1st / 99th percentile, median-filtered).
   - **The score.** score = (share of points in layout cells whose depth coordinate has two layers, i.e. a gap of more
     than 6 s with 3 or more points on each side) + (share of layout cells holding no point). The lowest score wins.
     Within 0.03 of the lowest, the simpler chart wins: height field, then cyl, hyb, sph.
   - **Warnings.** Above 10 % two-layer points or 15 % empty cells the tool prints a WARNING and records it.
4. **Fit.**
   - **Start.** For a sheet: a smoothed radial envelope (90th percentile of the radius per direction cell, holes filled
     from the nearest cell, Gaussian-smoothed). For a height field: the plane.
   - **Each level.** 5 robust least-squares solves on the control net. Tukey weights, with the cut at 4.685 x 1.4826 x
     the median residual, at least 2 s. Second differences of the net are the smoothness term (`--smooth` 1e-3,
     relative to the data term's trace). After every solve, one Gauss-Newton parameter correction per point.
   - **Refinement (knot insertion).** A span is split at its middle only when (a) at least half of its knot cells hold
     8 or more inlier points (`--min-support`), and (b) its median residual is above the chart's median residual.
     A periodic direction is doubled instead (exact uniform subdivision).
   - **When it stops.** When the held-out median does not fall by 1 % for 2 levels, when no span qualifies (`data_cap`;
     the record says how many spans wanted refinement but lacked points), or at `--max-levels` (20).
5. **Free space.** A camera that observed a point saw it unoccluded.
   - **The test.** For every (camera, point) pair in the tracks, a z-buffer of all charts finds the nearest lattice
     sample in the point's pixel. The point's own ray is then intersected with that sample's tangent plane. A
     violation is a hit that lies inside the sample's footprint and is nearer than the point by more than `--fs-tol`
     (2 s). Hits closer to edge-on than cos 0.1 are not judged and are counted separately.
   - **Reported.** The share at 1, 2, 4, 8 and 16 s; an 8 x 8 map over each chart's (u, v); and the share that falls on
     the unsupported part of each chart.
   - **As a penalty.** `--freespace W` adds rows that pull the hiding surface point back along its ray to the point's
     depth. Off by default; see the results below.
6. **Output.**
   - `charts.json`, per chart: kind, frame (origin, e1, e2, normal; axis and row parameters), degree 3, knots (or
     periodic n), control net `net[j][i]` (row j = v), support per control point (the summed basis weight of the
     inliers), the 48 x 48 unsupported mask, and label.
   - `report.json`: every threshold, the counts of every step, the layout candidates with their numbers, the level
     history, residuals (train / inliers / held-out: median, p95, rms, share within s; model units and mm), control
     points without support (support < 1), unsupported area share and regions (top edge, foot, side edges, interior),
     free space, and `stop`.

**Long-job rules.** These are the repo's rules.

- `--max-seconds` (900) and `--max-iters` (400 solves over all charts) cap the fit.
- A NaN / Inf net or residual stops at once, and that state is not kept. If no level finished, nothing is written.
- Convergence is judged on a 10 % held-out share (`--holdout`). Its residuals are reported.
- SIGINT / SIGTERM stop after the current solve.
- The best level by held-out median is what is written. `stop` holds {reason, detail, iterations, seconds}, and each
  chart's own stop holds {reason, detail, level, best_level}.

All four stops were exercised on scan 122:
- `--max-iters 25` gives `iteration_cap`, with best level 0 written.
- `--max-seconds 6` gives `wall_clock`.
- `kill -INT` at 7 s gives `signal`; files are written.
- `--smooth nan` gives `nan`: "no level finished: nothing to write".

Measuring and the free-space pass run after a stop and take another ~6 s.

## Self-test (`--self-test`, ~30 s, passes)

Two synthetic scenes go through a random similarity into COLMAP text models, then through the real entry. Each
has 0.3 % H point noise, 0.5 px observation noise and 4 % outliers.

- **A:** a closed bumpy body (a periodic cubic sheet) on a plane, with 24 cameras all round.
- **B:** a tall open sheet (label 0) and a low mound that is a known height field (label 1), with 20 cameras on a
  150-degree arc.

**Tolerances**, as shares of the object height H:

| measure | limit |
| --- | --- |
| fitted chart -> truth, median (where the chart stands on points) | <= 0.5 % H |
| fitted chart -> truth, p95 | <= 2 % H |
| truth -> fitted chart, p95 | <= 2 % H |

Measured: medians 0.11 to 0.15 % H, p95 0.18 to 0.25 % H, back p95 0.34 to 0.67 % H. Plane normals were within 1
degree of the truth. Layouts came out closed (A), open (B label 0) and height field (B label 1), as expected.

**The free-space diagnostic, calibrated:**

| surface | scene A | scene B |
| --- | --- | --- |
| the TRUE surfaces | 0.85 % | 1.68 % |
| the fit | 0.86 % | 1.81 % |
| the first object inflated by 15 % | 62 % | 41 % |

The floor on the true surfaces comes from the synthetic visibility test, which is approximate.

## Scan 122 (the owl), real SfM points, no depth

**Input.** `from_pipeline_js.py` rebuilds a COLMAP model from `talk/dtu/pipeline122.js`: 64 cameras and 21,760
points, put through a random similarity, with scale 1 = 80 mm. That file keeps no full tracks and no measured 2-D
positions. The model therefore holds each point's two lowest-index images, with exact reprojections as their
pixels, the original per-point error, and the true track lengths in `track_length.npy`. Free space on it uses two
rays per point, not the full tracks.

Default flags, 24 s on 4 cores, 95 solves; every chart converged on its held-out set:

**Clean**

| | |
| --- | --- |
| dropped by reprojection error | 579 |
| dropped by the k-NN test | 958 (507 table points kept again as plane inliers) |
| spacing s | 0.648 mm |

**Plane**

| | |
| --- | --- |
| inliers | 851, rms 0.41 mm |
| other points above it | 98.9 % (all 64 cameras above it) |
| plane chart | 6 x 10, held-out median 0.29 mm |
| plane chart, unsupported area | 82 % (the table is sparse in SfM) |

**Object**

| | |
| --- | --- |
| largest kNN component | 19,630 points (owl + books; next 47) |
| layout scores | cyl 0.213 (two-layer 3.4 %, empty cells 17.9 %), hyb 0.232, sph 0.538, height 0.576 -> **cyl**, open, 75-degree gap |
| warning | 18 % empty cells |
| net | 6 x 6 -> **35 x 69** (2,415 control points) in 13 levels |
| why it stopped | at level 12, 11 u-spans and 30 v-spans wanted refinement but lacked 8 points per knot cell |
| held-out residuals (1,956 points) | median **0.21 mm**, p95 1.68 mm, rms 1.04 mm, within 0.65 mm 82.5 % |
| train inliers | median 0.17 mm, p95 0.76 mm |
| control points without support | 482 of 2,415 |
| unsupported area | **34 %**: u < 0.21 near the foot (14 %), the right edge (12 %) |

**Free space** (41,460 pairs, 267 grazing). The tool's own figure, over both charts, is 7.6 % at 1.3 mm. To compare
like with like (yardstick use), the same test was run on the owl chart alone, for ours and for the shipped
depth-fitted chart:

| tolerance | 0.6 mm | 1.3 mm | 2.6 mm | 5.2 mm | 10.4 mm |
| --- | --- | --- | --- | --- | --- |
| our object chart | 12.9 % | 7.3 % | 4.2 % | 2.3 % | 1.1 % |
| shipped depth-fitted chart | 19.0 % | 7.6 % | 3.3 % | 1.2 % | 0.33 % |

Up to 1.3 mm the diagnostic measures the SfM points' own depth scatter: the depth-fitted chart violates as often as
ours, or more. Only above 2.6 mm does ours show more violations, 1.3 to 3.4 times as many. 91 % of ours fall on the
supported part of the chart.

**Free-space penalty (`--freespace`): it did not help.** At weights 0.1, 0.3, 1 and 10 the held-out median got worse
(0.37 to 2.0 mm against 0.21). The sheet's unsupported parts were pushed far off (unsupported area 65 to 94 %,
samples up to 0.9 m away). The violation share at 1.3 mm rose (10.8 to 31 % against 7.6 %).
The quadratic pull along the ray has nothing to stop it where no point holds the sheet. The flag is kept for
experiments, and the default is off.

**Yardstick** (`compare_dtu.py`; the shipped chart is itself an offline fit to depth maps, and is open at the back
and on top). Distances in mm, as mean / median / p95:

| | mean | median | p95 |
| --- | --- | --- | --- |
| ours -> shipped, all of our chart | 5.2 | 0.58 | 29.2 |
| ours -> shipped, the part on points | 2.4 | 0.42 | 16.9 |
| ours -> shipped, without the 21 % that lies beyond the shipped border | 1.1 | 0.41 | 4.3 |
| shipped (measured nodes) -> ours | 1.05 | 0.41 | 4.2 |
| pipeline.py's 12 x 16 sketch -> shipped | 1.7 | 0.95 | 5.9 |
| shipped -> sketch | 1.8 | 1.10 | 6.1 |

Where ours is worst, by median (mm):

| region | share of our chart | median | median on points |
| --- | --- | --- | --- |
| side edges u < 0.05 or > 0.95 | 10 % | 27.8 | 18.0 |
| top edge v > 0.95 | 5 % | 12.7 | 12.8 |
| back (azimuth >= 120 degrees from the cameras) | 26 % | 10.2 | 1.5 |

The front, the sides, the foot, the body / books and the head measure 0.35 to 0.47 mm median on points. For scale: the SfM points
themselves lie 0.31 mm (median) and 1.63 mm (p95) from the shipped chart (`pipeline122.js` stats).

**Side experiment.** `--layouts hyb` (forced) gave no clear difference: on points, median 0.42 and p95 14.9 mm. The
default stays with what the stated criterion picks.

### What is weak

- **One sheet is the wrong shape for owl + books.**
  - The cloud is 3.4 % two-layer even at the best axis, and 18 % of the layout cells are empty.
  - The sheet wraps about 285 degrees around an axis that the search pushed towards the back. The DTU cameras see
    the front and sides only, so the back quarter of the sheet bridges sparse strays and lies beyond the shipped
    chart. Its edges (u near 0 or 1) are extrapolations, with medians of 18 to 28 mm.
  - A horizontal ledge (the book tops, the top of the head) is not single-valued in a height-row sheet: the top edge
    is 12.7 mm off. `hyb` rows did not fix it.
- **The sparse cloud is thin where texture is poor.** The table has 0.86 k points spread over 0.3 m², and the plane
  chart is 82 % unsupported by the 3 x spacing rule. The data cap stops refinement at 35 x 69 (shipped: 288 x 392).
- **Free space is a weak cue at SfM noise.** It separates the depth-fitted surface from ours only above about 4 s,
  and as a quadratic penalty it hurts (see above).
- **The converted model is not a full COLMAP run.** It has two observations per point and exact pixel positions, so
  the reprojection filter acts on the stored per-point error only.

### Development disclosure (what was looked at while building this)

- **The truth frame.** The tool first found no plane on scan 122. I used the truth transform to count table points
  per cleaning step, and found that the density test removed them. That led to "plane before the density test".
- **Pictures and the yardstick.** The axis search was motivated by the azimuth coverage table, fit.py's design and
  the two-layer score. The coverage-based gap rule was motivated by the yardstick pictures (the sheet wrapping the
  back), and confirmed by the per-bin coverage of the points.
- **Calibrating free space.** The 1 / 2 / 4 / 8 / 16 s curve was added after running the diagnostic on the shipped
  chart. Its default tolerance did not change.
- **Not decided by the yardstick.** `--max-levels 20` (was 10) was decided on held-out convergence. `hyb` was added
  as a candidate, and the stated criterion still picks `cyl`.
- **Never tuned.** The k-NN, Tukey, smoothness, min-support and layout-gap thresholds were not tuned to the yardstick.
- **Not run.** A real COLMAP model with full tracks, scan 55, labelled (SAM / DINO) inputs on real data, and any
  Chamfer evaluation against DTU's reference (`tools/eval`).
