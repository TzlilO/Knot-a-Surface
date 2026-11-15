# tools/dtu — a real DTU object as B-spline charts

`fit.py` turns one scan of the DTU multi-view stereo dataset into `talk/dtu-data.js`, the data behind the
talk's `scan` scene (`talk/scenes.js`). The scene is an **offline fit to the dataset's own depth maps and
photographs**. It was not trained and it is not a result of the thesis method; nothing may say that it is.

```bash
pip install numpy scipy scikit-image Pillow
python3 -I tools/dtu/fit.py <data-dir> 122          # writes talk/dtu-data.js (about two minutes, no GPU)
python3 -I tools/dtu/fit.py <data-dir> 122 --cache /tmp/dtu-cache -o /tmp/try.js   # keep the fusion between runs
```

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

## The file it writes

`KnotTalk.dtu = { scan, label, source, note, unit_mm, stats, cams, target, charts }`, a classic script that
decodes itself on load. Per chart: `P` (Float32Array, control points in scene units; stored as 16-bit integers
over the chart's bounding box, or as four numbers for the flat table), `C` (Float32Array, control colours 0..1
in display space; stored as bytes), `seen` (Uint8Array, 1 = measured, 0 = inpainted; stored as bits). Row j,
column i sits at `j * nu + i`; `S_u x S_v` points out of the object.
