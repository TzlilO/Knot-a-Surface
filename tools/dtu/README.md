# tools/dtu — a real DTU object as B-spline charts

`fit.py` turns one scan of the DTU multi-view stereo dataset into `talk/dtu-data.js`, the data behind the
talk's `scan` scene (`talk/scenes.js`). The scene is an **offline fit to the dataset's own depth maps and
photographs**. It was not trained and it is not a result of the thesis method; nothing may say that it is.

```bash
pip install numpy scipy scikit-image Pillow
python3 -I tools/dtu/fit.py <data-dir> 122          # writes talk/dtu-data.js (about two minutes, no GPU)
python3 -I tools/dtu/fit.py <data-dir> 122 --cache /tmp/dtu-cache -o /tmp/try.js   # keep the fusion between runs
python3 -I tools/dtu/fit.py <data-dir> 55 --register scans    # another scan: writes talk/dtu/scan55.js
```

`--register scans` writes `talk/dtu/scan<N>.js`, which adds itself to `KnotTalk.dtuScans[N]` instead of setting
`KnotTalk.dtu` (that stays the owl), and carries its own `views` and `tour`. See "More scans" below.

`-I` because the input is downloaded data: keep it in a directory of its own, away from any code.

## Data

`<data-dir>/scan122/` in the layout of the Hugging Face dataset repo `Y1ue/DTU` (its card states
`license: apache-2.0`; the scans themselves are the DTU MVS data set of Jensen et al., CVPR 2014). Files used:

| path under `DTU/scan122/` | what | used for |
| --- | --- | --- |
| `cameras.npz` | IDR-style `world_mat_i` (= K [R\|t], 1600 x 1200) and `scale_mat_i` | geometry, the capture rig |
| `depths/depth_map_00NN.pfm` | 64 depth maps, 1600 x 1200, millimetres | geometry |
| `mask/0NN.png` | object silhouettes | table plane, occlusion of the table |
| `images/00NN.png` | 64 photographs, 1554 x 1162 (COLMAP-undistorted) | colour |
| `sparse/0/cameras.txt`, `images.txt` | the pinhole camera and poses of `images/` | colour |

```bash
B=https://huggingface.co/datasets/Y1ue/DTU/resolve/main/DTU/scan122; D=<data-dir>/scan122
mkdir -p $D/depths $D/mask $D/images $D/sparse/0
curl -sSL -o $D/cameras.npz $B/cameras.npz
for f in cameras.txt images.txt; do curl -sSL -o $D/sparse/0/$f $B/sparse/0/$f; done
for i in $(seq 0 63); do n=$(printf %04d $i); m=$(printf %03d $i)
  curl -sSL -o $D/depths/depth_map_$n.pfm $B/depths/depth_map_$n.pfm
  curl -sSL -o $D/images/$n.png $B/images/$n.png; curl -sSL -o $D/mask/$m.png $B/mask/$m.png; done   # about 650 MB
```

Not used: `fused_pointcloud.ply` of that repo. It is in another frame, has no normals and its colours are a
height ramp, not the photographs. The depth maps are consistent with `cameras.npz` to a few tenths of a
millimetre and carry the relief, so the script fuses those instead. The repo does not say how they were made;
they behave like depth rendered from DTU's reference surface (as in MVSNet's `Depths_raw`), including rays that
pass through holes of that surface and land on its inside. `clean_depth` drops those (1.3 % of the pixels).

## What the script does

1. **Frame.** RANSAC plane through the depth that falls outside the masks = the table (rms 0.17 mm). +y is its
   normal, +z points to the middle of the cameras. One scene unit = 80 mm, so the 284 mm object stands as tall
   as the talk's other scenes.
2. **Fuse.** The 64 depth maps go into one truncated signed distance volume (0.5 mm voxels, narrow band); its
   zero surface, where two or more views support it, is the measured surface: points with outward normals.
   Nothing is closed, filled or extrapolated at this stage.
3. **Chart.** One chart for the whole object. Directions are counted around an axis through the object
   (azimuth; height below the shoulder, elevation above it). In these coordinates the chart is a rectangle:
   left and right it ends at the azimuths where fewer than `--edge` (0.6) of the heights were measured, at the
   bottom at the table, at the top just inside the highest measured direction of each azimuth. So the sheet covers the
   front, both sides and as much of the top as the scan holds, has no pole, and is open at the back and on top
   of the head, where DTU has no data. A coarse envelope of the measurements (outermost point per direction)
   lays the sheet out; from coarse to fine the nodes are pulled onto the measured surface (nearest point that
   faces the same way) and each row and column is spread evenly along its own path, so the sampling is even
   over the relief (median spacing 0.78 x 0.89 mm).
4. **Holes.** A node with no measured surface within 1.2 mm is not observed: eye sockets, the undersides of
   overhangs the sheet has to bridge, the junction with the table at the back corners. Those nodes take the
   envelope's shape, shifted to meet the measured surface around them, and are flagged (`seen = 0`).
5. **Spline.** The control net is that of the clamped cubic tensor B-spline that passes through the nodes at
   the Greville abscissae (the same knots as `KnotTalk.nurbs.clampedKnots`). Where the sheet bridges a step,
   exact interpolation would ring, so a control point may not sit further than 0.6 mm from its node; that
   holds back 4 % of them, and there the surface only approximates the node.
6. **Colour.** Every node is projected into each photograph. A view counts if it sees the node front-on, in
   front of the fitted surface (a z-buffer of the fitted spline, rendered per camera) and, where the scan's own
   depth map is valid, in agreement with it. The four best views are blended, weighted by cos^6 of the viewing
   angle. Control colours are again solved so that the spline passes through the node colours. No colour comes
   from a point cloud. Nodes nobody sees take their neighbours' colour.
7. **Table.** A second chart: the fitted plane, 340 x 260 mm, coloured from the photographs wherever the
   object's silhouette does not cover it. Under and behind the object it is filled from its surroundings.
8. **Rig.** The 64 real camera centres from `cameras.npz`, in the object's frame.

## What it measured (scan 122, defaults)

| | |
| --- | --- |
| charts | 2 (owl 288 x 392, table 104 x 80) |
| control points | 121,216 |
| fit error, 303,155 measured points within the chart | rms 0.21 mm, 95th percentile 0.07 mm, median 0.012 mm |
| the same as a fraction of the object (337 mm diagonal) | 0.0006, 0.0002 |
| measured points within 1 mm of the chart | 99.6 % inside its border, 87.2 % of everything measured |
| owl chart observed / inpainted | 93.6 % / 6.4 % of the nodes, 83.6 % / 16.4 % of the area |
| owl nodes coloured from photographs | 96.6 % of the observed, 46.3 % of the inpainted |
| table chart seen by two or more photographs | 81.6 % |
| `talk/dtu-data.js` | 1.41 MB |

The rms is carried by the 0.4 % of points the single sheet cannot reach (inside grooves, under overhangs); the
median and the 95th percentile describe the surface it does cover.

**Resolution.** The depth maps sample the object at about 0.24 mm per pixel and the fusion at 0.5 mm, which
would feed a net of roughly 500 x 700. 288 x 392 (0.8 to 0.9 mm) is what a 1.5 MB data file holds at
12 base64 characters per control point; `--nu` / `--nv` raise it.

## More scans

Same script, same standard, `--register scans`. Nothing is tuned per scan except what `SCANS` in `fit.py` lists;
a scan without an entry takes everything from its own data: the fusion volume from its depth, the scale so
that it stands 3.55 scene units tall, the net from the sheet's size (as many columns per row as the sheet is
wide per tall, at most 112,896 nodes = what 1.5 MB holds), the table chart around its footprint. From the
second scan on, the object is separated from what it stands on by its silhouettes (a fused point belongs to the
object if it lies inside the mask in every view), not by its height above the table.

| scan | object | net (object + table) | control points | fit error: rms / 95th pct / median, mm | observed: nodes / area | within 1 mm of the chart | file |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 122 | owl on books | 288 x 392 + 104 x 80 | 121,216 | 0.21 / 0.068 / 0.012 | 93.6 % / 83.6 % | 99.6 % | `talk/dtu-data.js`, 1.41 MB |
| 55 | rabbit | 320 x 352 + 111 x 80 | 121,520 | 1.54 / 0.070 / 0.015 | 96.2 % / 91.2 % | 99.4 % | `talk/dtu/scan55.js`, 1.41 MB |

Scan 55, by hand: `top_mm=250`. Its photographs crop the ears, so only stubs of them were measured; the chart
ends level, 250 mm above the table, below them. Its rms comes from 0.5 % of the measured points, 20 to 29 mm
from the sheet: the throat under the muzzle, which the sheet bridges. One scene unit is 71 mm; 49 views.

### Scans that were tried and are NOT shipped

One sheet around an axis stands for a compact object whose measured surface can be seen from that axis. These
scans are not that, and each failed in a way the numbers show. The bar was the owl's: at least about 80 % of
the chart's area observed.

| scan | object | what happened | observed area | fit error rms / 95th pct, mm |
| --- | --- | --- | --- | --- |
| 106 | two birds | the sheet bridges the gap between the two bodies and their legs | 54.0 % | 3.2 / 6.3 |
| 118 | child in a pair of hands | the sheet stretches from the fingertips across the hollow to the child | 47.9 % | 4.4 / 11.8 |
| 105 | plush toy | arms and ears stick out; membranes span from limb to limb | 7.1 % | 6.4 / 15.8 |
| 69 | three snowmen | three bodies and thin arms; the sheet is mostly bridge | 2.7 % | 4.4 / 9.7 |
| 97 | cans and bags | separate objects at different depths; long bridges between them | 44.5 % | 1.4 / 1.9 |
| 110 | seated figure | script stops: only 28 % of the heights measured straight in front of the axis | - | - |
| 40 | concrete blocks | script stops: 43 % (an arch with a hole through it) | - | - |
| 37 | tools on a block | script stops: 59 % (flat, thin parts) | - | - |
| 24 | model building | script stops: 27 % (a facade wider than the frame) | - | - |
| 63 | fruit | script stops: 49 % (separate fruits, an axis through the gaps between them) | - | - |

"Script stops" is the check in `axis_and_wedge`: fewer than `--edge` of the heights hold measurements in the
direction the cameras look from, so there is no front to wrap a sheet around. With these, every scan of the repo
has been tried except 65, 83 and 114, which were excluded on purpose.

What would bring them in is an atlas of several charts per object, or leaving the unobserved nodes of a chart
out when it is drawn (the `seen` mask is in every file for that); neither is done here.

## Features of a pretrained model on the control points (`features.py`)

`features.py` bakes DINOv3 ViT-S/16 features onto the control points of a fitted scan and writes
`talk/dtu/features<N>.js` (`KnotTalk.dtuFeatures[N]`). The model is PRETRAINED (by Meta); it is run on the
scan's own photographs and its features are lifted onto the charts offline. Nothing here is trained, and none
of it is a result of the thesis method.

```bash
pip install onnxruntime
python3 -I tools/dtu/features.py <data-dir> 122 <model.onnx>      # writes talk/dtu/features122.js, about 3 minutes
```

The weights are not in this repo. The file above was made with `onnx/model.onnx` + `onnx/model.onnx_data`
(fp32, 86 MB) of the Hugging Face repo `onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX`, revision
`48988dfe`, an ONNX export of `facebook/dinov3-vits16-pretrain-lvd1689m`; its card names the licence
`dinov3-license` and the repo carries the text as `LICENSE.md` ("DINOv3 License", last updated August 14, 2025).

1. The charts and the camera centres are read from the scan's data file; the pose of the scene in the
   scanner's frame is the rigid fit between those centres and the ones in `cameras.npz` (residual 0.04 mm).
2. Each of the 64 photographs is resized from 1554 x 1162 to 1024 x 768 (64 x 48 patches of 16 px, about 5.8 mm
   on the object; ImageNet mean / std) and goes through the model once.
3. Every stored control point has a node on the surface (the spline at its Greville site). The node is projected
   into every photograph that sees it, by the rule the colour step uses: front-on (cosine above 0.2), in front
   of the fitted surface (z-buffer), in agreement with the scan's depth map where that is valid. The patch
   features, each of unit length, are read bilinearly between patch centres and averaged over the views with
   weight cos^2 of the viewing angle. Nodes no view saw take their neighbours' value.
4. One PCA over the stored nodes of both charts keeps 16 numbers per node; 27 nodes that stand out from their
   5 x 5 neighbourhood (seen by one or two views, reading a patch of something else) take its median; then
   8 bits per number.

Both charts are stored at every second control point (`step: 2`): 144 x 196 for the owl, 52 x 40 for the table.
The patches are seven times coarser than the control net, so nothing is lost by that.

| scan 122 | |
| --- | --- |
| stored nodes | 30,304 (owl 28,224, table 2,080) |
| views per node | owl 23.4, table 29.2 (mean) |
| nodes no view saw (filled from neighbours) | 7.05 % (owl 6.41 %, table 15.62 %) |
| variance kept by 16 components | 89.4 % (39.0, 9.8, 7.8, 6.3, 5.5, 5.0, 3.9, 2.2, 1.8, 1.5, 1.4, 1.3, 1.1, 1.0, 0.8, 0.8) |
| `talk/dtu/features122.js` | 648 KB |

What the features can tell apart, measured as cosine similarity in the stored 16 dimensions (the way the page
computes it): parts. From a node on the chest feathers, 8.8 % of the owl's nodes are above 0.8, all in the
body's rows of the chart (240 to 350); the two closed books (rows 0 to 120) stay below -0.15 and the table
below -0.14. From a node at an eye, the whole face
lights up (5.8 % above 0.8): the other eye reads 0.99, but so does the forehead between them (0.98). The eyes
are NOT separable from the face. That is not the PCA or the resolution alone: with all 384 dimensions, and
with the photographs at full size (1552 x 1152), the other eye reads 0.89 and the forehead 0.82.

`--check DIR --seed name:i,j` writes the pictures these numbers come from (the chart's stored grid coloured by
the first three components, and one similarity map per seed). `--cache DIR` keeps the patch features between
runs as 16-bit floats; a run from the cache differs from a fresh one by one unit in under 0.1 % of the bytes.

The file has the format of `lab/features/<scene>.js` (`tools/bake/distill.py`): `model, dim, views, res,
explained, scale, lo, hi, charts: { name: { nu, nv, data } }`, with `step` and `net` added per chart: `nu, nv`
are the STORED grid's size, `net` the control net's, and stored node (a, b) is control point (a * step, b * step).
Re-run it whenever the scan's control nets change.

## The file it writes

`KnotTalk.dtu = { scan, label, source, note, unit_mm, stats, cams, target, charts }`, a classic script that
decodes itself on load. Per chart: `P` (Float32Array, control points in scene units; stored as 16-bit integers
over the chart's bounding box, or as four numbers for the flat table), `C` (Float32Array, control colours 0..1
in display space; stored as bytes), `seen` (Uint8Array, 1 = measured, 0 = inpainted; stored as bits). Row j,
column i sits at `j * nu + i`; `S_u x S_v` points out of the object.

A file written with `--register scans` holds the same object under `KnotTalk.dtuScans[N]`, plus
`views: { wide, close, a, b }` (each `{ th, ph, r, tx, ty, tz }`, the shape `talk/scenes.js` uses) and
`tour: ['wide', 'close', 'a', 'b']`. The views are computed from the scan's own cameras: azimuths within 85 %
of the widest camera azimuth, elevations between the lowest and the highest camera ring, distances from the
object's size. To show one in a page: load its file, then `KnotTalk.dtu = Object.assign({}, KnotTalk.dtuScans[N])`
before `talk/scenes.js`, and take `views` / `tour` from the data instead of the owl's.
