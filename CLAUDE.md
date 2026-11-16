# Knot-a-Surface — Swarm Simulator & Thesis Deck

Live: https://tzlilo.github.io/Knot-a-Surface/ (landing page) · /terrain.html (swarm simulator) · /deck.html (reveal.js interview deck)

## What this is
A single-file three.js simulator illustrating the Knot-a-Surface thesis (NURBS-grounded
Gaussian splatting): a drone swarm reconstructs battlefield scenes as a spline-coupled
splat surface under a fixed Gaussian budget. The deck embeds the same simulator as a slide.
The site root is a landing page with method + measured results; three more sims (UV atlas,
knot chart, surface-prior bench) ship as sibling pages.

## Build & deploy
- The site root `index.html` is the hand-written landing page (its own source of truth — NOT a
  flatten output): a chrome-free `manifold` hero (`KnotTalk.mount('manifold', el, {bare:true,
  scene:'teaser', onStats})`) plus links to the talk, its lab modules and the older pages. Scene `teaser`
  (`tease()` in `talk/sims/manifold.js`) is a 30.6 s loop on the owl that ends exactly where it begins: three
  readings in rolling bands (normals, Gaussians, the picture), a dissolve into the picture (`uFade`), the materials
  wiping in one over another with a knob sweep, and back. Everything in it is a function of the time in the loop;
  keep it that way (the length is nine breaths of the owl, so its tilt and blink are at rest at both ends).
- Other pages' source of truth: the `*.dc.html` design files. `python3 deploy/flatten.py` rewrites
  links and emits the deployed pages (`project-page.html` = the previous landing page with the
  measured-results table, `terrain.html`, `knot.html`,
  `terrain-atlas.html`, `surface-prior.html`, `teaser-*.html`, `review/index.html`) —
  never edit those outputs directly. They sibling-load `support.js`, `_ds/` tokens and
  the `src/` cores, so there is no bundle to keep fresh.
- `python3 build.py` injects `src/sim_core.js` + `src/sim_style.css` + tokens into
  `src/deck_a/b/c.html` → `deck.html`, and `src/arena_core.js` + `src/arena_style.css`
  into `src/arena_template.html` → `arena.html`. It no longer emits `index.html`
  (`src/swarm_simulator_template.html` is kept for reference only).
- Deploy = `bash deploy/deploy-site.sh` (flatten + build + commit + push `main`).
  GitHub Pages serves in ~1 min (cache-control 600 s — hard-refresh or add `?v=N`).
  Root `.nojekyll` is required or Pages skips `_ds/`.
- Deck parts: `src/deck_a.html` (head+theme+slides 1-11) · `deck_b.html` (slides 12-21) ·
  `deck_c.html` (all deck JS: widgets, charts, reveal init).

## talk.html — the thesis talk (scroll-driven deck)
Independent of deck.html and of the build: plain static files, classic scripts (works from `file://`).
- `talk.html` = all slide copy. A step is `<article class="step nav" data-scene="…">`; the step nearest
  the reading line is active and its `data-scene` is pushed to the chapter's simulation
  (`<section class="chapter" data-sim="…">`). `data-depth="deep"` steps are hidden in the short cut;
  `data-min40`/`data-min20` on each section are the pace-clock budgets: the talk after the opening must sum to at
  most 40 / 20, the opening comes on top, and the cut buttons and the clock show the computed totals.
  A section's `data-mount='{…}'` (JSON) is merged into the options its simulation is mounted with (`bare`,
  `clear`, `scn`, `budget`, `cap`), and the section receives a `kt-stats` event with the module's stats.
- The opening (`<div class="prelude">`, first in talk.html): the English version of talk-he.html, "How does a
  computer see in 3D?" (explicit, implicit, Gaussians, the 3D breakfast scene, "Which one?"), then the talk's
  cover, all in the Hebrew deck's light "liquid glass" look, scoped to `.prelude` by `talk/prelude.css` (it
  re-points the kit's and deck's tokens; a sticky "room" of slow light behind it, clipped to `.prelude`; under the
  cover the page fades into the talk's night). While a step inside it is active, deck.js sets `body.lightroom` and
  the fixed chrome (rail, HUD, keys) turns light. `talk/prelude.js` = the 3D step's Gaussian slider and count
  (fewer Gaussians on a phone). The cover (`section.cover`, rail "K", key 0) mounts `manifold` with scene
  `cover`: the owl and the rabbit (hidden scene `duo`, DTU scans 122 and 55 side by side, the rabbit at the
  owl's scale) on a 22 s loop: the picture, a dissolve into rolling bands (normals, Gaussians, visual) and back,
  with a slow camera sway. Home goes to the opening.
- `talk/kit.js` = `KnotTalk.register(name, {title, blurb, tool, scenes, create(env)})` registry,
  lifecycle (`env.loop` pauses offscreen), UI helpers, and the standalone lab shell
  (`KnotTalk.standalone(name)`, used by every `talk/sims/<name>.html`).
- `talk/nurbs.js` = spline math (Cox–de Boor, derivatives, Boehm insertion, tensor patches; also
  `require()`-able under node). `talk/gl.js` = three.js helpers incl. `writePlanoids` (the derived
  Gaussian: centre + tangent axes from S_u, S_v and the sampling interval).
- Modules: `flatland` (2D differentiable-splatting toy, free vs surface-bound; its last scene `handoff`, SCRIPTED,
  fades everything but the bound spline and eases it onto the toy's true curve written as a 14-point cubic B-spline,
  and `basis` uses those same 14 control points (a literal copy) in every single-curve scene, so chapter 01 ends on
  chapter 02's first picture; both share one layout rule and the bright cyan / amber look), `basis` (SVG curve
  playground; its `letters` / `script` scenes draw "Shalom Rafael" as one spline per letter and as
  three handwritten strokes, from the control polygons in `PRINT` / `HAND`; `morph` blends letters →
  handwriting → the `MISSILE` outline: `morphSets()` cuts and refines all three into twelve cubics with
  matching control-point counts by knot insertion, so a blend is a lerp of control points and knots;
  `fit` takes a drawing as ground truth, fits it by least squares with n control points, reads it at m
  samples and shows the gap in red plus a measured gap-vs-control-points plot; `lens` draws the morph's
  curves ONLY as a fixed number of Gaussians that sit on them, spread evenly over the track inside a
  zoomable view window (2D, with an outside-view inset and a freeze toggle): Gaussians never go on empty
  plane; the drawings and `morphSets` live in `talk/ink.js` = `KnotTalk.ink`), `planoid` (one patch → derived Gaussians, resolution knob; lit translucent
  sheet, solid depth-tested vectors via `gl.vector` / `gl.sheet` / `gl.hidden`, a face-on tangent-plane
  inset, and a scale stage that measures the size from the lattice cell's 4 or 8 neighbours against the
  derivative formula),
  `ledger` (parameter arithmetic), `refine` (knot insertion/removal on a height field; a top view plus a
  draggable side cut, a normals shading, up to 4,096 control points and 256² Gaussians, and scene
  `balance`: a MEASURED map of drawn error over control points × Gaussians with three regime presets),
  `manifold` (the full-power demo: per-tile view-adaptive sampling under a fixed Gaussian budget over
  the scenes in `talk/scenes.js` — barn, truck, bust, garden as procedural stand-ins modelled after
  TnT / DTU / Mip-NeRF 360 captures, plus the abstract `studio` still life. A scene is only B-spline
  charts: `chart(list, name, opts, fn(s,t,i,j) -> [x,y,z,r,g,b])`; each returns views, a tour, a
  capture rig and an `edit(k)` animation. The stand-ins are NOT offered on the manifold lab page any more: its picker and tabs show only the DTU scans; the barn remains the landing page's hero and the fallback without scan data). The deck's chapter 06 runs on ONE subject, scene `scan`: a REAL
  DTU object (scan 122, an owl) baked offline by `python3 -I tools/dtu/fit.py <data-dir> 122` into
  `talk/dtu-data.js` (`KnotTalk.dtu`, must load before `talk/scenes.js`; without it the barn stands in).
  It is an offline fit to the dataset's depth maps and photographs: never call it trained or a result of
  the method. Further scans baked with `--register scans` land in `talk/dtu/scan<N>.js` (`KnotTalk.dtuScans`) and
  become scenes `scan<N>` in the lab page's picker only; one sheet around an axis suits a compact single body, and
  `tools/dtu/README.md` lists the scans that failed. `photo: true` = colours already hold the light (`uFlat`). Nets longer than `DENSE` control
  points are drawn thinned (`c.ns`), and dragging one of their points is a soft brush. Scene `mesh` lays one even
  lattice per chart, puts the Gaussians on its points and lets a running 2 x 2 window stitch two triangles per stop
  (`buildMesh` / `stepMesh`; instances are written in reverse window order so `setCount` retires the stitched ones).
  The owl's `animate(t, paint)` (`owlLife` in `talk/scenes.js`) is SCRIPTED: breathing, a head tilt and a blink as
  control-point edits, and `paint` wipes a MATERIAL in (scene `paint`): one entry of `MATS`, a rule that writes control
  colours and optionally control opacities (`s.a`) and motion (`move`), with its properties in `p` and the ones the
  simulator may turn in `knobs` (one slider each; `tune()` repaints); `animate(t, paint, material, from)` wipes one
  material in over another; the picker fills itself from `MATS`. To add or
  tune one, use the `owl-material` skill (`.claude/skills/owl-material/`). A scene without `animate` falls back to
  `edit(k)`. Opacity: `gl.splats(n, {alpha: true})` gives a field an opacity per Gaussian; while its
  `ktSplat.thin` is set, `gl.draw` draws the solid Gaussians as before and the see-through ones as two more layers
  (far side, near side) blended over them. Other fields and modules are untouched. Scene `hands` (lab page only) hosts
  the tool layer `talk/hands.js` (`KnotTalk.hands.create(host)`, its own file so that any module with charts can host it):
  Hands = grab / pinch / pull / push / hold on the object with mouse or touch, as soft-brush edits of control points;
  an atlas of the object's chart, one pixel per control point, in a chosen channel (rgb, normals, opacity, height,
  baked features when `talk/dtu/features<N>.js` is loaded, masks) with wand / brush / erase selection and tint / hide /
  inflate / hold on the selection. The module gives it `pickSurface` (nearest Gaussian under the pointer -> chart and
  control point, from `instC` / `instK` filled in `emit`) and the orbit's `pick` / `hit` / `release` hooks; it adds no
  touch handlers of its own. A chart marked `ground` (the table) is never touched and can be put away (`c.off`).
  The features are REAL DINOv3 ViT-S/16 features of the scan's photographs, baked by `python3 -I tools/dtu/features.py
  <data-dir> 122 <model.onnx>` (weights not in the repo; see `tools/dtu/README.md`) in the format of `lab/features/`;
  they tell parts apart (head, body, books, table), not the eyes from the face. Only the lab page loads them. Its views are the splat shader's render modes (`uMode` in
  `talk/gl.js`: appearance, depth, normals, single Gaussians, chart identity); view `split` (scene
  `modes`) shows all five at once as rolling slanted bands — the mode is picked per pixel (`uSplit`),
  so it is still one draw call; the band labels and dividers are DOM, moved every frame). The `bust`
  head is shaped from MediaPipe's canonical face mesh (`talk/face-data.js`, Apache-2.0; `faceFit` casts a
  ray per control point; without that file it falls back to sculpted bumps).
  `structure` (talk/sims/structure.js, chapter 07) = the scanned owl (the truck only on a page without the scan data) stored two ways in two viewports: a
  shuffled list (EMULATED unstructured 3DGS: jittered samples of the same surfaces, never call it trained)
  and the chart lattices; scenes `bag` (colour by memory order), `neighbours` (hash-grid kNN vs index
  ring), `mesh` (kNN graph vs lattice edges), `segments` (k-means vs chart id); all counts are measured.
  `touch` (talk/sims/touch.js, chapter 07b "Can't touch this") = a ball against one bowl stored two ways, 12 m apart:
  an EMULATED list of Gaussians and one cubic spline chart. Scenes `ghost` (the list has no boundary: fall through, or
  loose particles), `solid` (contact on the spline: closest point by Gauss-Newton in (u,v), n = S_u x S_v, restitution),
  `brittle` (toughness in N·s; pieces of the (u,v) domain as rigid bodies), `elastic` (control points on springs, force
  shared by `patchSupport` weights), `flight` (the ball is thrown from the spline bowl into the list). Fixed sub-steps,
  seeded, every number on screen measured; only the launch direction of `flight` is chosen.
- Rendering: modules draw with `gl.draw(R, scene, camera)`, not `renderer.render`. It draws the `gl.splats` of a scene
  as a smooth surface (depth pre-pass, weighted sum into a half-float target, normalising resolve, depth again for the
  rest of the scene) with a per-pixel low-pass on every Gaussian; without float render targets it falls back to the
  old dithered look. A new 3D module must use it too.
- Step music: `<article data-audio="name">` loops the first `data-audio-len` seconds of `KnotTalk.tracks[name]`
  (`talk/audio/tracks.js`, empty by default: no recording is in the repo) while the step is active; M mutes; the
  `.credit` line of a step shows only when a track has loaded. `data-sfx` marks a step whose simulation makes
  its own synthesized sound effects (chapter 07b); the ♪ button shows there too and `KnotTalk.muted` carries
  the mute to the module.
- Back faces: `gl.writePlanoids(..., cull, kept)` leaves out samples of CLOSED charts whose normal faces
  away from the eye; the manifold module and the lab lay their lattices `boost` times larger so what is
  written still meets the budget (the far side of a closed chart gets zero weight).
- Colour code is semantic and fixed: cyan = surface, amber = control points/knots, violet =
  Gaussians, red = error (chart identity colours in the finale's `charts` view are a separate,
  non-semantic set in `talk/scenes.js`). Evidence data (DTU table, bench numbers) = constants at the
  top of `talk/deck.js`; the DTU exhibit notes on screen when a row's mean differs from the printed mean.
- Touch: ONE shared layer in `talk/kit.js` — `KnotTalk.touchGuard` (claims a touch that starts on a
  handle, or two fingers, so the page neither scrolls nor zooms) and `KnotTalk.gestures` (pointer set,
  pinch distance/centre, claim/release) — used by `gl.orbit` and the 2D modules; do not add per-module
  touch handlers. Roots stay `touch-action:pan-y` so a vertical swipe over an embedded stage scrolls the
  deck; embedded 3D = one finger yaw, two fingers pitch, pinch zoom; standalone lab pages own every
  gesture. `KnotTalk.HIT` = touch hit radius; only a first finger may grab a control point (a second
  finger turns it into a pinch). Coarse-pointer sizes live in the CSS under `(pointer:coarse)`.
  `KnotTalk.mounted` lists live simulators for tests (multi-touch via CDP `Input.dispatchTouchEvent`).
- Verify with node playwright (`/opt/node-tools/node_modules/playwright`) against `file://…/talk.html`;
  scene ids double as URL hashes on the standalone pages (`talk/sims/planoid.html#knob`).

## talk-he.html — short Hebrew deck (explicit vs implicit vs 3DGS)
A separate 5–10 minute RTL deck with its own light "liquid glass" look; not a translation of talk.html.
- `talk/he/he.css` loads after `talk/kit.css` and re-points the kit's tokens to a light palette, so kit
  panels/sliders/segmented controls become glass capsules. `talk/he/he.js` = its scroll engine
  (RTL: ← forward, → back; space/↓ advance). Copy lives in `talk-he.html`; keep it to one headline and
  one short sentence per step.
- `talk/he/sims.js` = three Canvas-2D modules registered with the kit: `he-shapes` (a doughnut as
  points / triangles / voxels), `he-field` (signed-distance probe + ray marching), `he-fit` (a real
  optimiser fitting 2D Gaussians to a picture; normalised blend, Adam). Their labels are bilingual: Hebrew when
  the page's `<html lang>` starts with `he`, English otherwise (talk.html's opening uses them). The 3D step mounts `manifold`
  bare with `clear:'transparent'` and the hidden `breakfast` scene from `talk/scenes.js`.

## lab.html — the interactive figure (studio renderer, DINOv3 features, editing)
A standalone page set like a journal page (serif text, sans captions, booktabs table, light paper);
plain static files and classic scripts like the talk. It reuses `talk/nurbs.js`, `talk/kit.js`
(gestures only) and `talk/gl.js` (orbit, `writePlanoids`), and nothing from it is used by the decks.
- `lab/objects.js` = `KnotLab.scenes` (`tea`: the Utah teapot rebuilt from Newell's Bezier patches in
  `lab/teapot-data.js`, `knot`, `still`). Same chart shape as `talk/scenes.js` plus `mat {rough, metal}`
  and `lock` on the ground. Colours are PURE ALBEDO: the renderer lights the scene, never bake shading.
- `lab/render.js` = `KnotLab.renderer`: shadow pass -> G-buffer (MRT) -> lighting pass (key light with
  shadow map, screen-space occlusion, procedural studio reflections, tone mapping, or a data view) ->
  accumulation. While nothing moves, frames are averaged with a new dither, a sub-pixel camera jitter and
  a jittered light, so Gaussian fringes, shadows and occlusion converge (64 frames, then it stops
  drawing). Anything that changes the picture must call `invalidate()` (or pass `soft` / use `live`).
- `lab/app.js` = sampling (a port of the manifold module's per-tile allocation; `st.sampling` = even /
  adaptive / locked, where locked puts the whole budget inside the view frustum by area on screen), per-control-
  point channels `c.A` (feature colour x3, similarity, selection) blended to the Gaussians with the same
  basis as colour, picking (nearest Gaussian on the ray), tools (sculpt = soft brush on control points;
  select by feature similarity or by chart, then inflate / lift / scale / tint / hide), the split screen.
  `KnotLab.app` is the handle for tests and for the bake.
- DINOv3 features are REAL and baked offline: `node tools/bake/capture.js <dir>` renders each scene from
  its 24-camera rig (plus a per-pixel chart/(u,v) map), `python3 tools/bake/distill.py <dir> <model.onnx>`
  runs DINOv3 ViT-S/16, splats patch features onto control points, PCA -> 16-D int8 ->
  `lab/features/<scene>.js`. Re-run both whenever a scene's control nets change (the page checks net
  sizes and falls back to per-chart selection on a mismatch). The weights are not in the repo.

## arena.html — Arena: live 4D match reconstruction
A second, independent single-file three.js simulator (`window.KnotArenaSim`, ~`src/arena_core.js`)
applying the same thesis to a sports court instead of terrain: the swarm is FIXED (perimeter +
roof broadcast rigs — court bounds are known, nothing needs to fly) and the point of the demo is
flying INSIDE the live reconstruction, not orbiting a drone around it.
- `COURTS.{basketball,football}` = dims + markings (`courtLineSegs`/`courtColor`, single source
  for both the static splat floor and the minimap — same field/props invariant as the terrain sim).
- Ground truth: each player wanders a closed `CatmullRomCurve3` loop; the ball hops between
  players with a parabolic height arc (`stepGroundTruth`).
- Sensing → reconstruction: `updateSensing` fuses per-camera noisy detections (FOV+range gated)
  into a per-entity ring buffer at 10 Hz; `reconAt` fits an OPEN Catmull-Rom through that buffer
  and evaluates it `LATENCY` (0.22s) in the past — this is the "spline-coupled 4D" position
  estimate that gets rendered, distinct from the ground truth (toggle "ghost truth" to compare).
- Splats reuse the terrain sim's soft-falloff billboard technique (`splatAlphaTex`,
  confidence-driven alpha via `onBeforeCompile`); static court floor is baked once (flat, no
  billboard needed), dynamic per-entity clusters billboard to the fly camera every frame and
  grow/dim with tracking confidence (`e.cov`, driven by camera coverage count).
- Free-fly camera (WASD + mouse via pointer lock) is the whole point — no commander/main split
  view like the terrain sim. Scoped out for now: mobile touch controls, occlusion-aware camera
  visibility (FOV+range only, no line-of-sight blocking).

## sim_core.js architecture (one IIFE, ~3.3k lines)
Module scope: `hash2/vnoise/fbm` noise · env definitions `ENVS.{urban,forest,ocean,dust}`
with `h(x,z,t)` occupancy + `col(x,z,h,t)` albedo · `bicubic()` = switchable basis
(CR default path / clamped B-spline p2-5 / NURBS rational, module var `BASIS_DEG`).
`create(container,{embed})` closure holds everything else:

- **fit model**: `fit{CN,cur,tgt,res,crop,x0,z0}` — control grid of heights.
  `sampleTarget` (4-tap max supersampling), `optimStep` (time-based rate, Laplacian
  regularizer REG=0.3, weighted by `covW` swarm-view coverage × `peW` photometric,
  plus bidirectional-Chamfer geometry supervision `GEO` when `st.geoSup`, residual
  `fit.gres`), `computeWeights` (strided ≤64/axis), `truncSubdivStep` (error-driven
  subdivision), warm-up levels `WARM_LEVELS` (64×64 floor), `reWarm()` on any MODEL
  change. Control-net UV resolution is `st.cnet` (default 96, user-adjustable).
- **sampling modes** (`st.adaptive/st.fluid`): `rebuildImportance`+`axisSample` =
  inverse-CDF warp (curvature+texture+residual); `FLU` = fluid particles in
  window-relative coords, potential = truncation error, settle/freeze logic in
  `fluidStep` (calm<thr or flowT>8 → frozen; wakes only when frozen).
- **recon**: `updateRecon` two-pass (heights `HH`, then NN-stretched splats + facade
  wall-scan on vertical jumps) · `reconFluidPass` for fluid mode · `allocRecon` splits
  budget into surface `S×S` + `wallCap` by `wallFrac(pitch)`.
- **views**: main renderer + commander renderer share one scene; layers: 0 world,
  1 recon, 2 overlays. `st.swapped=true` by default (commander = MAIN, world = inset).
  `frustumWindow()` = view-driven allocation window `win` (always on).
  `applyFit()` = frozen image-plane framing (binary search).
  Channels RGB/DEPTH/NORMAL/CONF via `fillChannels`+`pushChannel/popChannel` buffer swap.
- **observer drone** (`st.drone=true` by default → owns the MAIN POV): `droneStep(dt)`
  is a kinematic entity `drone{px,pz,hd,alt,lx,lz,ly,auto}` that drives `cmdrCam`
  pos/lookAt + `st.roi` + `cOrbit`. Inputs: joysticks `joy/joyL/joyR` + `elev` (wired
  by `makeStick`) and WASD/arrows/QE keys; autonomous patrol resumes after 5 s idle.
  Desktop = one stick; mobile = twin sticks (left rotate+alt, right move) + zen-on-move
  (`.kss-moving`). Guard flight with `st.drone && !st.warm.active`.
- **swarm**: `updateSwarm` — physical footprints `footF = alt*0.85`, overlap slider
  sets spacing; `swarmFoot.pts` feeds `computeWeights` (layout matters!).
- **commander tools**: ruler+LOS (`surfacePickFrom` ray-march), markers, TACREF
  (`tacQuery` — MGRS-style grid + confidence, clipboard), pitch slider, LINK/FREE.
- **segmentation** (`seg`): UV-map panel, click ±seeds, `segRun` = seeded mean-split +
  flood-fill (classical CV) on the active channel. Mask is WORLD-anchored in a fixed
  grid `seg.gmask` (`SEG_GW=384` over ±`SEG_BND=120` m) via `segMarkWorld`/`segLook`,
  so it stays put as the drone flies and strokes AGGREGATE until `revert`. `applySeg`
  zeroes per-splat `conf` outside the mask (revertible). Works on RGB/DEPTH/NORMAL.
- **gradient readout** (`grad`): `gradDraw` renders a slope map with downhill arrows.
- **mesh recon** (`st.recon='mesh'`, `st.meshKind` tri/quad/tetra): `meshIndexFor`/
  `setMeshIndex` stitch the S×S sample grid live via the 2×2 UV kernel into an indexed
  mesh (`meshObj`+`meshWire`, shared vertex buffer); no facade pool (grid connectivity
  spans jumps), whole budget → surface. Mutually exclusive with fluid sampling.
- **multi-spline** (`SUBS`): `subFromMask` detaches a seg-mask region into a dedicated
  finer world-anchored spline (warm-started from the composite), `subOptimStep` refines
  each independently, `subH` = composite height (global + feathered local overrides —
  patched into `updateRecon` HH, `reconH`, `buildPLY`). `autoDecompose` = bulk: threshold
  spline-vs-GT estimation error + steady-climb gradient term → flood-fill components →
  top-4 get their own splines. Seg-panel buttons: decompose+refine / auto-decomp /
  merge back. SPLINES dock menu (`renderSubMenu`) = per-surface hide/refine(subdivide)/
  remove + `rebuildBounds` draped mask-contour overlay (layer 2, color per surface,
  `st.show.bounds`). Cleared on env change.
- **export**: `startExport` 2×2 UV-kernel sweep animation → `buildPLY` binary PLY.
- **UI**: dock + popup menus in `bar`, 3 s auto-fold, zen mode, mobile media queries
  (in sim_style.css), pinch zoom, joystick/twin-stick + elevation controls.
- Debug: `window.__kss = {fit, swarmFoot, st, FLU, topo, seg, grad, drone}` +
  `__kss.segStats()` (visible-vs-masked splat counts).

## Invariants — do not break
1. Field/props single-source: anything visible in 3D must ALSO be in `h()`/`col()`
   via the SAME deterministic rule (see `forestTree`, `FINE.*`, `SITE_BOXES`,
   `DUST_TROOPS`, `patrolPos`, `boatPos`). Divergence = "invisible to the model" bugs.
2. Grid structure: wall-scan, NN-stretch, and PLY 2×2 tessellation assume the S×S
   sampling grid (adaptive warp keeps it; fluid mode bypasses walls deliberately).
3. All optimization/animation rates must be TIME-based (`1-exp(-K*dt)`), never
   per-frame — fps varies wildly (drone kinematics included).
4. Optimization weights come from SWARM views; splat allocation window from the
   COMMANDER frustum. Don't mix them.
5. Segmentation masks are WORLD-anchored (`seg.gmask` in fixed world coords), never
   frustum/screen space — otherwise the mask follows the drone and re-masks the scene.
6. Defaults: forest env, 64k budget, MANUAL mode, CR basis, ×2 params, adaptive
   sampling, commander-as-main, observer-drone POV on, mesh-supervision (Chamfer) on.

## Testing
Headless (no GPU here — 64k is slow, test at 4k):
```bash
sed 's/budget: 65536/budget: 4096/' index.html > /tmp/t.html
python3 - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b=p.chromium.launch(args=["--no-sandbox","--headless=new","--disable-gpu","--single-process"])
    pg=b.new_page(viewport={"width":1100,"height":650})
    pg.on("pageerror", lambda e: print("PE:",e))
    pg.goto("file:///tmp/t.html"); pg.wait_for_timeout(5000)
    print(pg.inner_text(".kss-hud-tl")); b.close()
PY
```
Always: syntax-check every inline script block with `new Function(...)` after building,
grep the HUD for expected state, screenshot before shipping. Push only after zero
console errors.
