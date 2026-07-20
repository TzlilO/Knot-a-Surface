# Knot-a-Surface — Swarm Simulator & Thesis Deck

Live: https://tzlilo.github.io/Knot-a-Surface/ (simulator) · /deck.html (reveal.js interview deck)

## What this is
A single-file three.js simulator illustrating the Knot-a-Surface thesis (NURBS-grounded
Gaussian splatting): a drone swarm reconstructs battlefield scenes as a spline-coupled
splat surface under a fixed Gaussian budget. The deck embeds the same simulator as a slide.

## Build & deploy
- Source of truth: `src/` — never edit `index.html`/`deck.html`/`arena.html` directly (generated).
- `python3 build.py` injects `src/sim_core.js` + `src/sim_style.css` into the two
  templates (`/*__SIM_CORE__*/`, `/*__SIM_CSS__*/` markers) → `index.html`, `deck.html`,
  and `src/arena_core.js` + `src/arena_style.css` into `src/arena_template.html`
  (`/*__ARENA_CORE__*/`, `/*__ARENA_CSS__*/`) → `arena.html`.
- Deploy = commit + push to `main`; GitHub Pages serves in ~1 min (cache-control 600 s —
  hard-refresh or add `?v=N` to see changes).
- Deck parts: `src/deck_a.html` (head+theme+slides 1-11) · `deck_b.html` (slides 12-21) ·
  `deck_c.html` (all deck JS: widgets, charts, reveal init).

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
