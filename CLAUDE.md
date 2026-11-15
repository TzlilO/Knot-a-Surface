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
  scene:'hero', onStats})`) plus links to the talk, its lab modules and the older pages.
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
  (`<section class="chapter" data-sim="…">`). `data-depth="deep"` steps are hidden in the 20′ cut;
  `data-min40`/`data-min20` on each section are the pace-clock budgets (must sum to 40 / 20).
- `talk/kit.js` = `KnotTalk.register(name, {title, blurb, tool, scenes, create(env)})` registry,
  lifecycle (`env.loop` pauses offscreen), UI helpers, and the standalone lab shell
  (`KnotTalk.standalone(name)`, used by every `talk/sims/<name>.html`).
- `talk/nurbs.js` = spline math (Cox–de Boor, derivatives, Boehm insertion, tensor patches; also
  `require()`-able under node). `talk/gl.js` = three.js helpers incl. `writePlanoids` (the derived
  Gaussian: centre + tangent axes from S_u, S_v and the sampling interval).
- Modules: `flatland` (2D differentiable-splatting toy, free vs surface-bound), `basis` (SVG curve
  playground; its `letters` / `script` scenes draw "Shalom Rafael" as one spline per letter and as
  three handwritten strokes, from the control polygons in `PRINT` / `HAND`; `morph` blends letters →
  handwriting → the `MISSILE` outline: `morphSets()` cuts and refines all three into twelve cubics with
  matching control-point counts by knot insertion, so a blend is a lerp of control points and knots), `planoid` (one patch → derived Gaussians, resolution knob; also the cover's hero),
  `ledger` (parameter arithmetic), `refine` (knot insertion/removal on a height field),
  `manifold` (the full-power demo: per-tile view-adaptive sampling under a fixed Gaussian budget over
  the scenes in `talk/scenes.js` — barn, truck, bust, garden as procedural stand-ins modelled after
  TnT / DTU / Mip-NeRF 360 captures, plus the abstract `studio` still life. A scene is only B-spline
  charts: `chart(list, name, opts, fn(s,t,i,j) -> [x,y,z,r,g,b])`; each returns views, a tour, a
  capture rig and an `edit(k)` animation). Its views are the splat shader's render modes (`uMode` in
  `talk/gl.js`: appearance, depth, normals, single Gaussians, chart identity); view `split` (scene
  `modes`) shows all five at once as rolling slanted bands — the mode is picked per pixel (`uSplit`),
  so it is still one draw call; the band labels and dividers are DOM, moved every frame).
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
  optimiser fitting 2D Gaussians to a picture; normalised blend, Adam). The 3D step mounts `manifold`
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
- `lab/app.js` = sampling (a port of the manifold module's per-tile view-adaptive allocation), per-control-
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
