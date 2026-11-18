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
# objects from per-view labels (section "One chart per object" below)
python3 -I tools/recon/label_points.py MODEL_DIR -o LAB --masks MASKS_DIR [--features FEAT_DIR] [--ground-plane]
python3 -I tools/recon/label_points.py --self-test                                       # ~45 s
```

| file | what |
| --- | --- |
| `sfm_spline.py` | the tool (steps below) |
| `colmap_io.py` | COLMAP text reader / writer. Camera models: PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV. Distortion is ignored, with a printed note. |
| `bspline.py` | cubic bases with derivatives, Boehm insertion, periodic direction, sparse design matrix (the maths of `talk/nurbs.js`, vectorised) |
| `selftest.py` | `--self-test`: synthetic COLMAP models with known spline surfaces |
| `from_pipeline_js.py` | `talk/dtu/pipeline<N>.js` -> COLMAP text model (+ `track_length.npy`), in a random similarity frame |
| `compare_dtu.py` | YARDSTICK ONLY: distances to the shipped depth-fitted chart and to `pipeline.py`'s 12 x 16 sketch, plus pictures (`--chart a,b` pools several charts) |
| `label_points.py` | point labels from per-view masks and / or feature maps -> `labels.npy` for `--labels` (below) |
| `selftest_labels.py` | `label_points.py --self-test` |
| `standin_dtu.py` | STAND-IN masks for scan 122 rendered from the repo's depth-fitted charts (not SAM), and optional synthetic visibility |
| `gpu/` | `run_sam.py`, `run_dino.py`: masks and features on a GPU machine. **NOT RUN** (see `gpu/README.md`) |

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
   - **Trimmed (`--trim`).** Measured a second time with each chart cut to its trim mask (step 6): trimmed-away
     surface hides nothing.
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
   - `--trim` (off by default): each chart gains `trim_mask_48x48_v_rows` (1 = kept) and `trimmed_domain` (its (u, v)
     box). The kept part = the supported cells of the 48 x 48 bridge grid, closed by `--trim-close` 2 cells, enclosed
     holes filled, pieces under `--trim-min-piece` 1 % of the area dropped. The net is not changed; the trim only says
     where the chart holds. `report.json` gives per chart the kept area share and the unsupported share of the kept part.

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
(for the single-sheet tool; the labelled pipeline has its own disclosure at the end of the next section)

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

## One chart per object, from per-view labels (`label_points.py`)

The owner's design: a point is labelled in the views that OBSERVED it (its track), the per-view labels are associated
across views, and every resulting cluster gets its own chart in `sfm_spline.py --labels`.

### Method (all thresholds are flags; `labels.json` records them)

1. **Observations.** Each point is read only in the views of its own track, at the 2-D position the track stores.
   Nothing is reprojected into views that did not observe it. Masks: integer images `NAME.npy` / `NAME.png`
   (H x W, or H x W x K for K levels of a hierarchy; 0 = no mask; ids mean nothing across views; any resolution; a
   view may be missing). Features: `NAME.npy`, H' x W' x D, read bilinearly between cell centres.
2. **Candidate pairs** (sparse, O(points x k)): 3-D kNN (`--knn` 12, links < `--link` 3 x the median 12th-neighbour
   distance) and, per view, the `--img-knn` 8 nearest co-observed points in the image within `--img-radius` 2 % of the
   diagonal (these can be far apart in depth, which is where "different" votes come from).
3. **Votes.** In every (view, level) where both points of a pair are observed and both lie in a mask, the pair votes
   same / different. With c votes and s "same", a pair with c >= `--min-covis` (2) gets w = s - tau c (`--tau` 0.5).
4. **Clustering** = correlation clustering (minimum-cost multicut) by greedy agglomeration: contract touching clusters
   while their summed weight is positive. Two orders: summed weight (GAEC, Keuper et al. 2015) and weight per vote
   (average linkage, as in GASP, Bailoni et al. 2019); `--linkage best` (default) runs both and keeps the partition with
   the higher objective (summed weight inside clusters). Deterministic.
5. **Spatial sanity.** A cluster that is not connected in the 3-D kNN graph is split into its pieces.
6. **Features** (optional): per point the normalised mean of its normalised samples. They merge touching clusters whose
   mean features have cosine >= `--feat-merge` 0.9 where the masks are undecided (agreement >= tau - `--tie-band` 0.15,
   or no votes); they decide where a small cluster goes; without masks they are the only signal (agglomeration with
   CENTROID linkage while the cosine of the cluster means is > `--feat-tau` 0.8: comparing means keeps a blend at a
   boundary from chaining two objects). `labels.json` says which signal formed each cluster and lists every merge.
7. **Small clusters** (< `--min-size` 30 or `--min-share` 0.5 % of the points) join the touching cluster with the best
   evidence unless the masks say they differ; otherwise -1.
8. **`--ground-plane`** (prior): `sfm_spline.py`'s support plane; points within `--ground-tol` 1.5 s of it are label 0
   before anything else.

Output: `labels.npy` (index = POINT3D_ID, -1 = none), `labels_confidence.npy` (share of the point's votes that agree
with its label, -1 = no votes), `labels.json` (thresholds, clusters with size / signal / confidence, merges, splits,
per-view mask statistics, stop). `--max-seconds` and SIGINT / SIGTERM stop the agglomeration; the partition reached is
valid and is written with its stop reason.

### Self-test (`--self-test`, 44 s, passes)

Scene B of `selftest.py` (tall open sheet, low mound, ground plane; 20 cameras on 150 degrees; 0.5 px noise, 4 %
outliers) with masks rendered by a z-buffer of the true surfaces at a quarter of the image size and ids shuffled per
view. Scored on the non-outlier points, leaving out the 1,551 that lie within 1 % H of TWO true surfaces (the mound's
apron is tangent to the plane, so their label is undefined; with them in, every ARI is about 0.04 lower).
Degraded = 20 % of views missing, 30 % over-segmented (each part cut by a random image line), 10 % under-segmented
(both objects in one mask), boundary jitter 1 mask px (4 image px), 5 % salt noise.

| variant | ARI | purity | completeness | unlabelled | objects back |
| --- | --- | --- | --- | --- | --- |
| clean masks | 0.999 | 1.000 | 0.999 | 0 % | yes |
| degraded | 0.996 | 0.999 | 0.999 | 0 % | yes |
| degraded + a second level cutting every part in two | 0.993 | 0.998 | 0.998 | 0 % | yes |
| tall object cut at a fixed height in 60 % of views, masks alone | 0.759 | 1.000 | 0.872 | 0 % | no (4 clusters) |
| same + synthetic features | 0.999 | 1.000 | 0.999 | 0 % | yes (1 feature merge) |
| synthetic features alone | 0.982 | 0.998 | 0.993 | 0 % | yes (+1 small boundary cluster) |
| degraded, tracks cut to their 2 lowest-index views | 0.111 | 0.563 | 0.858 | 1.5 % | no |
| the file entry (.npy, .png, a 2-level view) | 0.994 | | | | ARI 0.996 vs in memory |

Noise sweep on top of "degraded" (salt = share of mask pixels set to a random id of the view, or 0), 2 seeds each:

| salt | 0 | 20 % | 40 % | 50 % | 60 % |
| --- | --- | --- | --- | --- | --- |
| ARI mean | 0.996 | 0.992 | 0.942 | 0.845 | 0.356 |
| failure rate (ARI < 0.9 or objects not back) | 0 / 2 | 0 / 2 | 0 / 2 | 2 / 2 | 2 / 2 |

"Objects back" = every true object has a cluster holding >= 90 % of it at >= 95 % purity. Synthetic features are
one random 16-D vector per object plus noise; they say nothing about real DINO features.

### Scan 122 with STAND-IN masks (not SAM)

No photographs are in the repository, so SAM cannot run. `standin_dtu.py` RENDERS masks from the repo's own
depth-fitted charts (`talk/dtu-data.js`: object chart + table; `talk/dtu/owlmask122.js`: owl vs books) into the 64
cameras: classes owl / books / table, ids shuffled per view, 389 x 291. A degraded copy has 20 % of the views missing,
50 % over-segmented, 10 % under-segmented (owl and books in one mask), 1 px (4 image px) jitter and 2 % salt. These
masks come from the yardstick itself: the runs below test the plumbing and what per-object charts buy, and are no
evidence that SAM would produce such masks.

**Tracks.** `pipeline122.js` keeps two observations per point. `standin_dtu.py --synthetic-visibility` writes a copy of
the model whose tracks are extended by SYNTHETIC visibility (a view is added when the point projects inside it and lies
at most 2 mm behind the stand-in surfaces' z-buffer), cut to each point's TRUE track length (median 4; uncut, median
44 views). 6.9 % of the stored observations lie more than 2 mm behind that z-buffer (points off the surfaces). Only
the labelling uses these tracks; every fit below runs on the original two-observation model, so free space is
measured on the same pairs as the single sheet.

```bash
python3 -I tools/recon/standin_dtu.py OUT/standin --model OUT/model --synthetic-visibility OUT/model_synvis
python3 -I tools/recon/label_points.py OUT/model_synvis -o OUT/lab --masks OUT/standin/masks_clean --ground-plane
python3 -I tools/recon/sfm_spline.py OUT/model -o OUT/fit --unit-mm 80 --track-len OUT/model/track_length.npy --labels OUT/lab/labels.npy --trim
python3 -I tools/recon/compare_dtu.py OUT/fit/charts.json OUT/truth/transform.json --chart label1,label2
```

**Labels** against the stand-in classes (each SfM point takes the class of the nearest stand-in surface sample within
3 mm; 3,763 of 21,760 points are farther and not scored; owl 8,557, books 8,813, table 627):

| tracks, masks | ground prior | ARI | owl / books / table recovered |
| --- | --- | --- | --- |
| 2 per point, clean (`--min-covis 1`) | no | 0.878 | owl split off, 475 owl points in books; table absorbed |
| 2 per point, degraded | yes | 0.111 | owl and books MERGED |
| synthetic (cut to true length), clean | no | 0.977 | owl, books yes; table in pieces / absorbed into books |
| synthetic (cut), clean | yes | 0.991 | yes (objects back) |
| synthetic (cut), degraded | yes | 0.987 | yes |
| synthetic (uncut, 44 views), degraded | yes | 0.991 | yes |

Without the ground prior the sparse table (627 points) never forms its own cluster. Every run also returns two small
clusters (about 300 and 400 points) that lie mostly more than 3 mm from the stand-in surfaces (scene clutter / strays);
they get charts of their own and are left out of the comparison below.

**Charts** (`sfm_spline.py --labels ... --trim`, default flags otherwise; label 0 = table, 1 = owl, 2 = books in all
three labelled runs). Held-out residuals per chart (mm): owl median 0.24 / p95 1.48, books 0.21 / 2.98 (clean labels);
single sheet 0.21 / 1.68. Layouts: owl `cyl`, two-layer 1.0 %, empty cells 8.2 % (single sheet: 3.4 %, 17.9 %); books
`cyl`, 3.9 % / 18.0 % with an 18 % empty-cell warning, open over 190 degrees. No object's criterion picked a height
field (the table's did).

Yardstick (`compare_dtu.py`, owl + books charts pooled, sampled by area; mean / median / p95 in mm):

| | single sheet | per object, clean | per object, degraded | per object, 2-view labels |
| --- | --- | --- | --- | --- |
| ours -> shipped, all | 5.24 / 0.58 / 29.2 | 3.15 / 0.50 / 20.1 | 3.10 / 0.50 / 18.7 | 3.45 / 0.52 / 20.0 |
| ours -> shipped, on points | 2.45 / 0.42 / 16.9 | 1.70 / 0.37 / 10.1 | 1.63 / 0.37 / 9.5 | 1.86 / 0.37 / 12.4 |
| shipped (seen) -> ours | 1.05 / 0.41 / 4.2 | 1.22 / 0.44 / 4.7 | 1.34 / 0.43 / 5.7 | 1.30 / 0.42 / 5.3 |
| beyond the shipped border | 20.5 % | 10.6 % | 9.7 % | 11.7 % |
| unsupported area | 34.5 % | 39.8 % | 41.3 % | 45.0 % |
| top edge v > 0.95, median | 12.7 | 0.82 | 0.73 | 2.67 |
| side edges, median | 27.8 | 12.9 | 12.0 | 13.3 |
| back (az >= 120), share / median | 26 % / 10.2 | 14 % / 9.2 | 13 % / 9.5 | 14 % / 12.0 |
| ledges (normal within 30 deg of up), median / p95 | 0.35 / 32.3 | 0.31 / 28.8 | 0.28 / 28.6 | 0.26 / 30.8 |
| foot y < 40 mm, p95 | 33.3 | 6.2 | 5.4 | 9.0 |

Free space (share of observation pairs with fitted surface in front of the point; same pairs in every column):

| charts | 0.6 mm | 1.3 mm | 2.6 mm | 5.2 mm | 10.4 mm |
| --- | --- | --- | --- | --- | --- |
| single object sheet | 13.2 % | 7.4 % | 4.2 % | 2.3 % | 1.1 % |
| owl + books, clean labels | 15.5 % | 9.2 % | 5.3 % | 2.8 % | 1.3 % |
| owl + books, degraded labels | 17.2 % | 11.1 % | 6.7 % | 3.6 % | 1.3 % |
| single sheet, trimmed | 12.3 % | 6.9 % | 3.9 % | 1.9 % | 0.8 % |
| owl + books, clean, trimmed | 14.6 % | 8.6 % | 4.9 % | 2.4 % | 1.0 % |

**Trim.** Single sheet: keeps 75 % of the area, unsupported 34 % -> 12 % of what is kept. Owl: 80 %, 32 % -> 15 %.
Books: 64 %, 46 % -> 16 %. Table: 38-41 %, 82 % -> 56-60 % (the table is too sparse for the 3 x spacing rule). On
the yardstick the trimmed part is about as good as the "on points" part (pooled owl + books 1.95 / 0.42 / 11.1 against
1.70 / 0.37 / 10.1): filled holes and the closing keep some bridging, notably at the back (trimmed back median 2.7 mm).
It lowers free-space violations a little (above).

**Does one chart per object fix the ledges and the unsupported back? Partly, and not the back.**
- Fixed: the top edge (median 12.7 -> 0.8 mm: the owl's sheet ends at the top of the head instead of bridging the
  books' ledge), the foot (p95 33 -> 6 mm), the overall tail (p95 29 -> 20 mm all, 17 -> 10 mm on points), and half of
  the area beyond the shipped border. The owl's layout is much cleaner (two-layer 1 %, empty 8 %).
- Not fixed: the back (median still about 9 mm; the cameras do not see it, and each object sheet still bridges it); the
  ledges' tail (p95 about 29 mm; their median was already good); the side edges (12-13 mm median).
- Worse: unsupported area (34.5 -> 40-45 %: the books' open 190-degree sheet is 46-51 % unsupported), shipped -> ours
  (p95 4.2 -> 4.7 mm), and free space (7.4 -> 9.2 % at 1.3 mm with clean labels, 11.1 % degraded): two sheets meet in
  the crease where the owl stands on the books, and each one's border runs on in front of the other's points.
- Two observations per point are not enough for the association: clean masks still split owl from books (ARI 0.88),
  but with 10 % under-segmented views owl and books merge (ARI 0.11). The self-test shows the same (ARI 0.11).

### Development disclosure (label_points / stand-in runs)

- `--linkage best` and the self-test's "ambiguous band" were decided on the SELF-TEST: plain GAEC (sum) merged the mound
  with the plane in 2 of 3 seeds at 0 % salt; average linkage over-split in others; keeping the higher objective was
  right in all 6 cases I looked at. `--tau` stayed 0.5 (0.6 and 0.7 were tried on the self-test, with sum linkage, and were worse at high
  noise). The centroid linkage for features alone replaced a pairwise rule that chained mound and plane in the self-test.
- On the owl, the stand-in classes were used to SCORE labels, and the ground prior was turned on after the run
  without it left the table in pieces. No threshold of label_points.py or sfm_spline.py was changed after looking at
  the yardstick. `--trim-close` 2 and `--trim-min-piece` 1 % were set once and not tuned.
- Not run: SAM / DINO on real photographs (`gpu/`, NOT RUN), real full COLMAP tracks, scan 55, and a book-top height
  field (the layout criterion never picked one for an object, and I did not force it).
