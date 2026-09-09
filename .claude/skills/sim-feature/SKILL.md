---
name: sim-feature
description: Add or modify a feature in the Knot-a-Surface swarm simulator (sim_core.js). Use for any request touching the simulator — new scene objects, HUD elements, commander tools, sampling modes, optimization behavior, UI controls, environments. Triggers on "add to the simulator", "change the sim", "new scene", "new tool", "fix the gaussians", "make the swarm…".
---

# Adding a simulator feature

Read `CLAUDE.md` first — especially the architecture map and the five invariants.

## Workflow
1. Edit `src/sim_core.js` (and `src/sim_style.css` for styles). NEVER edit
   `index.html`/`deck.html` — they are generated.
2. Prefer anchored replacement via a small python script over hand-editing long
   files — assert the anchor exists so silent misses fail loudly:
   ```python
   c = open('src/sim_core.js').read()
   old = "exact unique snippet"
   assert old in c, 'anchor missing'
   open('src/sim_core.js','w').write(c.replace(old, new))
   ```
3. `python3 build.py`
4. Syntax gate: `node -e "new Function(require('fs').readFileSync('index.html','utf8').match(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/g).map(s=>s.replace(/<\/?script[^>]*>/g,'')).join(';'))"` — or per-block; must parse clean.
5. Run the /sim-verify skill before claiming done.

## Recipes
- **New scene object**: add it to BOTH the env's `h(x,z,t)`/`col(x,z,h,t)` AND the
  matching `build*Props()` using ONE shared deterministic rule (grid-hash like
  `FINE.car`, instance rule like `forestTree`, or a box list like `SITE_BOXES`).
  Moving objects: parametric position fn of `t` (see `patrolPos`, `boatPos`,
  `troopPos`) + set env `animated: true` + per-frame mesh update in the loop.
- **New env**: ENVS entry, `ENV_HOME`, terra branch in `buildEnv`, props builder,
  env button in the ENV menu, `facadeCol` branch.
- **New commander tool**: button in `.kss-cmdr-tools`, handler in `cUp` (inset) and/or
  `onUp` (main, check `st.swapped`), picking via `surfacePickFrom`, draw on layer 2
  via `ovl()`, readout via `cmdrRead`.
- **New dock menu**: menu div in `.kss-menus` + `data-open` button in `.kss-dock`;
  existing wiring handles open/close/fold automatically.
- **Optimization change**: keep rates time-based; weights flow through
  `computeWeights` (covW/peW/mvW — the last is multi-view photo-consistency, gated
  post-warm-up); geometry supervision is the bidirectional-Chamfer term in
  `optimStep` (`GEO`, gated by `st.geoSup`, residual `fit.gres`); anything that
  reshapes the domain must call `setFitDomain(..., keepCur=true)` and set
  `reconDirty` directly (domain reshapes are a discrete change, not camera-follow —
  don't route them through `camDirty`).
- **Camera-follow vs. discrete resample**: anything that moves/resizes the sampling
  window because the COMMANDER VIEW is panning/rotating/flying (frustum tracking)
  must set `camDirty = true` (+ `_lastCamMoveT = st.t`), not `reconDirty` — `loop()`'s
  gate promotes it on settle/timer/drift (tunables in `__kss.st.resample`). Only
  discrete state changes (env/mode/budget/basis switch, subdivision, explicit UI
  toggle) should set `reconDirty` directly for an immediate rebuild.
- **Per-splat data**: extend the arrays allocated in `allocRecon`
  (posArr/nrmArr/errArr pattern), fill in `updateRecon` AND `reconFluidPass`.
- **Drone / camera control**: `droneStep(dt)` owns the main POV. Add an input axis
  by wiring a UI element into `joy/joyL/joyR/elev` (see `makeStick`) or the `keys`
  map, then read it in `droneStep` — keep integration time-based and in the heading
  frame. Anything gating flight uses `st.drone && !st.warm.active`. Pitch/gimbal
  controls must branch on `st.drone`: when the observer drone owns the POV, write
  `drone.gp` (manual gimbal pitch; `null` = terrain-follow via `impliedGp()`), NOT
  `cOrbit.ph` — `droneStep` only reads `cOrbit` when the drone is off.
- **Onboarding gallery** (`onboard()`, only used by the standalone `index.html` entry
  point — the deck embed calls `create()` directly): new preview screens/cards must
  sample the REAL `ENVS.h()/col()` fields or the real technique (splat/mesh), never a
  decorative stand-in (same field/props invariant as the sim). Route the chosen value
  through `sel` and, on "▶ START", into `opts.initial` → `create()`'s onboarding
  hand-off block (merges into `st` before `applyDensity()`/fit construction, then
  syncs the dock's hard-coded `kss-on` buttons via `syncRow`). New RAF-driven preview
  canvases must be registered in `items` and disposed in `killPreviews()` (geometry +
  material + texture + renderer.dispose/forceContextLoss) — onboarding tears down
  every preview context before booting the real WebGL context.
- **Segmentation**: masks are WORLD-anchored in `seg.gmask` (`segMarkWorld`/`segLook`,
  `SEG_GW`×`SEG_GW` over ±`SEG_BND` m) — never store mask in screen/frustum space.
  New channel: extend `fillChannels` + the `seg.src` switch. Strokes AGGREGATE; only
  `revert` clears `seg.gmask`. `applySeg` zeroes per-splat `conf` outside the mask.
- **Mobile / zen-on-move**: gate with `st.mobile` (`(pointer:coarse)`); moving-drone
  zen toggles the `.kss-moving` class. Make anything view-sized swap-aware (`resize()`
  picks `cmdrCam` when `st.swapped`) so mobile portrait aspect stays correct.

## Perf budget
Per-frame O(S²) work is the ceiling (S≤240 at 64k default; 512k is a stress option).
Field sampling (`sampleTarget`, importance, weights) must be throttled/strided —
follow the existing cadence patterns. Test at 4k, sanity-check at 64k.
