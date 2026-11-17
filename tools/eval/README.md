# tools/eval — DTU Chamfer distance

Three scripts, numpy + scipy only, all run with `python3 -I` (the reference data is downloaded data):

| file | what it does |
| --- | --- |
| `dtu_chamfer.py` | the DTU evaluation of ONE mesh (or point cloud): accuracy, completeness, overall, in mm; writes a JSON |
| `chart_mesh.py` | the spline charts of a fitted scan (`talk/dtu-data.js`, `talk/dtu/scan55.js`) as PLY meshes in the DTU world frame, with a check of that frame |
| `eval_fits.py` | both, for the owl (scan 122) and the rabbit (scan 55), and a table |

```bash
python3 -I tools/eval/dtu_chamfer.py --self-test                                  # 31 checks on made-up data, 7 s
python3 -I tools/eval/dtu_chamfer.py --self-test --data REF --scan 1              # + 10 on a real reference, 2 min
python3 -I tools/eval/dtu_chamfer.py MESH.ply --scan 24 --data REF --out r.json   # any mesh in the DTU world frame, mm
python3 -I tools/eval/eval_fits.py --dtu DATA --ref REF -o OUT                    # the repository's two fits
```

## Read this first: what the numbers for the owl and the rabbit are

**They are offline fits, not trained results, and not results of the thesis method.** `tools/dtu/fit.py` fitted
those charts to the depth maps of the Hugging Face copy `Y1ue/DTU`. Those depth maps are not a reconstruction from
the photographs: they are MVSNet's `Depths_raw` (Yao et al., ECCV 2018), made, as that paper describes its depth
maps, by screened Poisson reconstruction of DTU's ground-truth point clouds and rendering the mesh into every view.
The evidence gathered here:

- `DTU/scan114/depths/depth_map_0000.pfm` and `DTU/scan118/depths/depth_map_0010.pfm` of `Y1ue/DTU` are byte for
  byte the files `dtu/depth_raw/Depths/scan114|118/…` of the public benchmark copy `depth-anything/DA3-BENCH`
  (`dtu.zip`), which ships MVSNet's evaluation set. Scans 122 and 55 are not in that archive, so for them the
  identity is inferred from the same naming, size and behaviour, not checked.
- Scan 114, view 0, back-projected with the copy's `cameras.npz`: every third valid pixel (114,319 points) lies
  0.135 mm (median) from the nearest point of `stl114_total.ply`, 99.8 % within 0.5 mm. A point cloud with 0.2 mm
  between points cannot be met much closer than that: the depth map lies ON the reference.

So the chain is: structured-light reference → Poisson surface → depth maps → spline chart → compared with the
structured-light reference. The DTU Chamfer distance of these charts measures **how well a spline chart of this size
holds the reference surface** (and what the single open sheet leaves out or bridges). It does not measure
reconstruction from photographs, and it is **not comparable with any row of a published DTU table**, where every
method sees only the photographs. Never print it next to such rows without saying so.

## Status: the two reference clouds are missing

The numbers for scans 122 and 55 have **not** been computed. From the container this was written in:

| file | needed for | status |
| --- | --- | --- |
| `ObsMask122_10.mat`, `Plane122.mat`, `ObsMask55_10.mat`, `Plane55.mat` | mask, ground plane | obtained (see "Reference data") |
| `stl122_total.ply`, `stl055_total.ply` | the reference clouds | **missing** |

They are in `Points.zip` (7.0 GB, all 124 scans) of the DTU MVS data set, "MVS Data Set - 2014" at
`roboimagedata.compute.dtu.dk` (files served from `roboimagedata2.compute.dtu.dk/data/MVS/`). Both hosts were
refused by the container's network policy. On Hugging Face the two files were found only in gated repositories
(`SRPLB1/benchmarks-dtu-mvs`: `parts/Points.zip`; `Bluefish2333/benchmark`: `MVRecon/dtu/Points.tar`; both need the
owner's approval). The public copies hold the clouds of other scans: `rhayat/DTU` (`SampleSet.zip`: the masks and
planes of every scan, the clouds of scans 1 and 6 only) and `depth-anything/DA3-BENCH` (`dtu.zip`: the 22 scans of
MVSNet's test set, among them 24, 110, 114, 118).
Nothing was put in their place: the depth maps the fits were made from are NOT a reference for this number.

With the two files in `REF/Points/stl/`, the whole evaluation is one command, under ten minutes on four cores:

```bash
python3 -I tools/eval/eval_fits.py --dtu DATA --ref REF -o OUT     # OUT/summary.md, OUT/summary.json, one JSON per mesh
```

Until then it runs the protocol up to the observation mask (`--mask-only`) and says which file is missing.

## The protocol, and every parameter

R. Jensen, A. Dahl, G. Vogiatzis, E. Tola, H. Aanæs, "Large Scale Multi-view Stereopsis Evaluation", CVPR 2014; the
MATLAB code that comes with the data set (`SampleSet.zip`, "Matlab evaluation code": `BaseEvalMain_web.m`,
`PointCompareMain.m`, `reducePts_haa.m`, `MaxDistCP.m`, `ComputeStat_web.m`); and its Python port
`github.com/jzhangbs/DTUeval-python` (`eval.py`), the one NeuS, 2DGS, PGSR and many others call. `dtu_chamfer.py`
was written from those two sources and follows the port step by step.

| step | what | parameter | value |
| --- | --- | --- | --- |
| 1 sample | every vertex + a lattice on every triangle: corner + a·v1 + b·v2, a = (i + ½)/n1, b = (j + ½)/n2, a + b < 1, n = ⌊edge / (density·√(│v1││v2│ / │v1 × v2│))⌋ | `--density` | 0.2 mm |
| 2 reduce | shuffle; walking the points, one that is still kept removes all others within the density | `--density`, `--seed` | 0.2 mm, seed 0 |
| 3 mask | points inside the reference box grown by the patch (BB₀ − 60 ≤ p < BB₁ + 120, as the port) = `data_in`; of those, the ones whose voxel `round((p − BB₀) / Res)` of `ObsMask{N}_10.mat` is set = `data_in_obs` | `--patch`; mask margin | 60 mm; 10 mm |
| 4 accuracy | nearest reference point for every point of `data_in_obs` (ALL reference points); mean of the distances below the cap | `--max-dist` | 20 mm |
| 5 completeness | nearest point of `data_in` for every reference point with P · [x y z 1] > 0 (`Plane{N}.mat`); mean of the distances below the cap | `--max-dist` | 20 mm |
| 6 overall | (accuracy + completeness) / 2, the "Chamfer distance" of DTU tables | | mm |

Where the MATLAB code differs, and why no distance below 20 mm changes: it searches block by block with a 60 mm cap
instead of cropping the input to the box; it rounds half away from zero (numpy: half to even; only a point exactly
on a voxel face); its mesh sampler `MeshSupSamp` is a compiled mex without source, so the port's lattice is used.
One thing differs from the port: the nearest-neighbour searches here stop at the cap (a distance of 20 mm or more is
an outlier whatever its value), which gives the same means to the last digit and keeps a search far from everything
from walking the whole tree.
Both reference implementations shuffle without a seed (the port reports a scatter of 0.01 %); here the shuffle is
seeded, so a run repeats exactly.

**A perfect mesh does not score zero.** The reference is a cloud with 0.2 mm between points and the input is
reduced to points at least 0.2 mm apart, so neither side lands on the other: the exact surface as a mesh scores
about 0.08 mm accuracy and 0.13 mm completeness in the self-test. Read numbers near 0.1 mm as "at the floor".

A long numerical job by this repository's rule: progress on stdout; `--max-minutes` (30) checked between stages
and between chunks of every search; `--max-points` (120 million) refuses a mesh before memory is allocated; NaN or
Inf in the input or in a distance stops it; SIGINT / SIGTERM stop it cleanly. The JSON is written in every case,
with `stop.reason` (`completed`, `max_minutes`, `max_points`, `signal`, `non_finite_input`, `missing_reference`,
`non_finite_result`, `no_points_in_mask`, `no_distance_below_cap`, `mask_only`, `bad_input`, `out_of_memory`) and no `result` unless it
completed. `eval_fits.py` carries the reason into `summary.json`.

The JSON: `parameters`, `input` (path, bytes, sha256, vertices, triangles, area), `reference` (the three files with
bytes and sha256, BB, Res, the plane), `counts` (points after each step), `result` (`accuracy_mm`,
`completeness_mm`, `overall_mm`, and per side: median, rms, 95th percentile, how many distances were used and how
many fell at or over the cap, the share within 0.2 / 0.5 / 1 / 2 / 5 mm), `timing_s`, `environment`, `stop`.

## Reference data

`--data REF` is searched for `ObsMask{N}_10.mat` and `Plane{N}.mat` in `ObsMask/`, `SampleSet/MVS Data/ObsMask/`,
`MVS Data/ObsMask/` or flat, and for `stl{N:03d}_total.ply` in `Points/stl/` (the port's layout), the data set's
own folders, or flat. What was downloaded for this work (kept outside the repository; nothing of it is committed):

| source | files | bytes |
| --- | --- | --- |
| Hugging Face `rhayat/DTU` @ 0e6f3b0, `SampleSet.zip` (6,924,030,144 bytes: the data set's own SampleSet, read by HTTP range requests, 305 MB of it) | `MVS Data/ObsMask/ObsMask{1,6,55,122}_10.mat`, `Plane{1,6,55,122}.mat` | 408,809 / 425,750 / 1,141,604 / 899,765; 206 each |
| | `MVS Data/Points/stl/stl001_total.ply`, `stl006_total.ply` | 77,783,981 / 88,559,951 |
| | `MVS Data/Surfaces/tola/tola001_l3_surf_11_trim_8.ply`, `…/furu/furu001_l3_surf_11_trim_8.ply`, `MVS Data/Points/tola/tola001_l3.ply` | 88,587,567 / 46,090,125 / 26,837,421 |
| | `MVS Data/Calibration/cal18/pos_001.txt … pos_064.txt`, "Matlab evaluation code" | small |
| Hugging Face `depth-anything/DA3-BENCH` @ cb98821, `dtu.zip` | `dtu/Points/stl/stl114_total.ply`, two files of `dtu/depth_raw/Depths/` | 83,446,718; 7,680,016 each |
| Hugging Face `Y1ue/DTU` | `DTU/scan{122,55,114}/cameras.npz`, two depth maps | 124,574 / 95,354 / 124,574 |

sha256 (first 16 hex digits): `ObsMask122_10.mat` 9459187f316711c5, `Plane122.mat` 3d32a015e38db441,
`ObsMask55_10.mat` 6b1a9dad344311db, `Plane55.mat` 73ecd180ea64563a, `stl001_total.ply` 91c07cde9e631391.
The data set is citeware (its ReadMe: cite the CVPR 2014 paper); the mirrors state their own licences.

## The meshes of the fits, and their frame

`chart_mesh.py` reads a file written by `fit.py` and writes PLY in the DTU world frame, in mm.

**Frame.** `fit.py` maps world mm `X` to scene coordinates `p` by `L = (X − base) @ A`, `p = (L − o) / unit_mm`
(`A`: rotation with columns right, up = the table's normal, front; `base`, `o`: points on the table), so
`X = R (unit_mm · p) + t` with `R = A`, `t = base + A·o`. None of `A`, `base`, `o` is stored, but the camera centres
are on both sides: `cams` in the file (scene units) and `cameras.npz` of the data set copy (`world_mat_i` = K [R|t]
in the DTU world frame). `R, t` = the rigid fit between them (Kabsch: no scale, no reflection); the script stops
unless every centre maps to within `--tolerance` (0.2 mm). Measured:

| | scan 122 (owl) | scan 55 (rabbit) |
| --- | --- | --- |
| cameras | 64 | 49 |
| file's centres → data set's centres, rms / max | 0.041 / 0.059 mm | 0.034 / 0.050 mm |
| what the file's rounding alone allows (3 decimals of a scene unit, per axis) | ±0.040 mm | ±0.036 mm |
| scale a similarity would have chosen | 0.999993 | 1.000006 |
| `cameras.npz` centres vs. the data set's own `Calibration/cal18/pos_NNN.txt`, max | 0.00006 mm | 0.00004 mm |
| the charts' table plane vs. the data set's `Plane{N}.mat`: angle; the table lies below the plane by | 0.035°; 3.1 … 3.4 mm | 0.042°; 3.0 … 3.3 mm |
| points of the object's full lattice (sampled, reduced) inside `ObsMask{N}_10.mat` | 752,829 of 762,207 (98.8 %) | 867,686 of 892,718 (97.2 %) |

The last three rows are independent of the fit's own data: the official projection matrices, the official ground
plane (which the data set places a few mm above the table: in scans 1 and 6 its table points lie 5.6 mm below it)
and the official observation mask all agree with where the charts land.

**Lattices.** Every mesh is a lattice of one chart stitched by the 2 × 2 window of `talk/meshlist.js`
(two triangles per stop):

| name | what | scan 122 | scan 55 |
| --- | --- | --- | --- |
| `full` | one vertex per control point: the spline at the Greville abscissae of its knots (where `fit.py`'s nodes sit); reaches the chart's border | owl 288 × 392 (0.78 × 0.89 mm), table 104 × 80 | rabbit 320 × 352 (0.89 × 0.89 mm), table 111 × 80 |
| `full_x2` | 2 × 2 vertices per control point: shows what the lattice's flat triangles cost against the spline itself | 575 × 783 | 639 × 703 |
| `talk160000` | what the talk's `mesh` / `stitch` scenes draw and export at their default budget for a scan, 160,000 Gaussians: even lattices at the cell centres, sized by `sizeLattices` | owl 282 × 382 (0.79 × 0.90 mm), table 263 × 199 | rabbit 348 × 309 (0.81 × 1.01 mm), table 271 × 194 |
| `talk30000` | the same at the default of a small screen | owl 122 × 165 (1.80 × 2.07 mm), table 114 × 86 | rabbit 150 × 134 (1.87 × 2.31 mm), table 117 × 84 |

The talk's slider runs from 6,000 to 300,000 (`--budget`). `chart_mesh.py` ports `coarse` and `sizeLattices` of
`talk/sims/manifold.js`; for scan 122 at 160,000 the port was compared with the lab page itself
(`talk/sims/manifold.html#mesh`, headless, `api.meshFile('ply')`): the same lattices, 160,061 vertices that differ
by at most 0.00004 mm, 317,874 triangles. The other three talk lattices are the port alone.

**The table.** Every lattice is written twice: `…_object.ply` (the object's chart) and `…_all.ply` (object +
table chart). The protocol does not remove the table from the input: the plane file only decides which REFERENCE
points count for completeness, and the observation mask contains the table around the object. Measured with the
official masks (full lattice, after the reduction): the owl's table chart adds 878,183 points, 493,975 of them
inside the mask; the rabbit's adds 1,076,734, 648,197 inside. All of those lie below the ground plane (the charts'
table is 3.2 mm under it), and they still count: with the table, 40 % and 43 % of the accuracy term is flat table
measured against the reference's table points. So the two differ, and both are reported: `…_object` is the fit of
the object, `…_all` is what the talk shows. A published mesh usually contains the table too.

## Verification

1. **Self-test on made-up data** (`--self-test`, 31 checks, all pass): the PLY reader (binary, ASCII, big-endian,
   CRLF headers, quads, mixed polygons); the sampler's density and largest gap; the reduction (no two kept points
   closer than 0.2 mm, every removed one within 0.2 mm of a kept one); the same surface as a mesh: 0.079 / 0.135 mm
   (the floor); the reference cloud itself: accuracy 0.000000; a flat plate moved rigidly by 0.5, 2 and 7 mm along
   its normal: 0.513 / 0.515, 2.003 / 2.004, 7.001 / 7.001 mm (between d and √(d² + 0.2²), as the sampling allows);
   slid by 3 mm in its own plane: only its 3 mm lip counts; the ground plane, the mask, the 20 mm cap; the guards;
   same seed, same numbers.
2. **On the official reference of scans 1 and 6** (`--self-test --data REF --scan 1`, 41 checks, all pass): the
   reference cloud as input scores accuracy 0.000000 mm (completeness 0.034 and 0.013 mm: the reduction removes
   points that are exactly 0.2 mm apart). A rigid shift reads as d where the surface faces it: the reference's
   table alone (its 109,141 and 148,433 points under the ground plane) moved by 1 and 3 mm along the plane's normal
   scores a median of 0.992 and 2.931 mm (scan 1), 0.977 and 2.899 mm (scan 6); the means are 0.933, 2.806, 0.916,
   2.723 mm, because the board is not flat to better than 0.5 mm. The whole cloud moved the same way scores 0.367
   and 1.126 mm (scan 1), 0.407 and 1.282 mm (scan 6): above zero and below d, as it must, since every surface
   that lies along the shift slides on itself.
3. **Against the reference implementation**, DTUeval-python's `eval.py` run unmodified (open3d 0.20.0,
   scikit-learn 1.9.1) on official inputs of scan 1; accuracy / completeness / overall, mm:

   | input | DTUeval-python, two runs | `dtu_chamfer.py`, seeds 0 and 1 |
   | --- | --- | --- |
   | `tola001_l3_surf_11_trim_8.ply` (mesh, 4.2 M triangles) | 0.245385 / 0.431018 / 0.338201 · 0.245437 / 0.431076 / 0.338256 | 0.245403 / 0.431028 / 0.338216 · 0.245431 / 0.431044 / 0.338237 |
   | `furu001_l3_surf_11_trim_8.ply` (mesh, 2.2 M triangles) | 0.431408 / 0.432997 / 0.432203 · 0.431458 / 0.432982 / 0.432220 | 0.431440 / 0.432963 / 0.432202 · 0.431077 / 0.432980 / 0.432029 |
   | `tola001_l3.ply` (point cloud) | 0.223438 / 0.479048 / 0.351243 · 0.223486 / 0.479067 / 0.351276 | 0.223480 / 0.478987 / 0.351234 · 0.223428 / 0.479018 / 0.351223 |
   | `stl001_total.ply` moved by 1 mm (point cloud) | 0.367443403070 / 0.407178101514 / 0.387310752292, both | the same 12 digits, both |
   | `stl001_total.ply` moved by 3 mm (point cloud) | 1.126178143964 / 1.184998391197 / 1.155588267581, both | seed 1: the same 12 digits; seed 0: 1.126178101790 / … |

   Where the shuffle decides nothing the two agree to the last digit; elsewhere they differ by what a shuffle
   changes (over eight seeds the Furukawa mesh's accuracy runs from 0.43108 to 0.43144 mm; the port's two runs lie
   inside). Run time 17 to 45 s here against 22 to 246 s for the port on the same four cores.
4. **A published number was not reproduced.** For scans 122 and 55 the reference is missing; for the scans whose
   reference is public, no published mesh with a published per-scan number could be obtained (DTUeval-python's own
   table is on COLMAP meshes that are not distributed). Item 3 is the substitute: agreement with the implementation
   the published numbers were computed with, on the data set's own files.

## Orientation only

`talk/deck.js` prints, from Table 1 of the paper, Chamfer distances in mm for the same two scans: scan 55: PGSR
0.28, ours 0.329; scan 122: PGSR 0.31, ours 0.341. Those are methods that reconstruct from photographs. An
offline fit to depth rendered from the reference is a different thing and may come out lower or higher for reasons
that have nothing to do with reconstruction quality (the floor of about 0.1 mm; the open back and top of the
single sheet, which completeness counts up to 20 mm; the 6 % and 4 % of nodes that are inpainted).
