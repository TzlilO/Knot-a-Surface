---
name: sim-reference
description: Ground-truth reference for exactly what is LIVE on GitHub Pages right now (https://tzlilo.github.io/Knot-a-Surface/) — pinned to commit 3fab12a. Use when the user asks what the simulator currently does/has, "what's live", "what features does it have", wants a walkthrough/demo script, or asks about behavior and you need to know the deployed truth rather than any local branch. NOT for making changes (see sim-feature) or testing (see sim-verify) — this is a description-only reference and gets stale the moment `main` moves, so re-verify the pinned commit before trusting it (see "Staleness check" below).
---

# What's actually deployed (commit `3fab12a`)

This document describes ONLY what is live at
https://tzlilo.github.io/Knot-a-Surface/ (and `/deck.html`, `/arena.html`) as of
GitHub Pages build `3fab12a3518f389ff3bde105ae35445fc5b94355` ("Fix pitch controls
under drone POV + rebuild with 2-col tool rail"). Everything below is verifiable
by reading `index.html`/`deck.html`/`arena.html`/`src/*` at that commit. Anything
in an open branch/PR that hasn't merged to `main` is deliberately excluded — don't
blend in half-landed work from other branches when using this doc.

## Staleness check (do this first if it matters)
`main` moves; this file doesn't move itself. Before relying on specifics here for
anything user-facing or decision-relevant:
```bash
gh api repos/TzlilO/Knot-a-Surface/pages/builds/latest --jq .commit
git log -1 --format='%H %s' origin/main
```
If the commit differs from `3fab12a3518f389ff3bde105ae35445fc5b94355`, this
document is stale — treat it as historical context only and re-derive from
`origin/main` + a fresh Pages-build check (see `sim-deploy` for how deploys work).

## The three deployed pages
- **`/` (`index.html`)** — the terrain swarm simulator. Boots into a **pre-run
  onboarding gallery** (`KnotSwarmSim.onboard`), not the sim itself — see below.
- **`/deck.html`** — a reveal.js 5.1.0 interview deck (21 slides across
  `src/deck_a/b/c.html`); the sim appears embedded as one slide via
  `KnotSwarmSim.create(slot, {embed:true})` — this path calls `create()` directly
  and **skips onboarding**.
- **`/arena.html`** — a second, independent single-file three.js simulator
  (`window.KnotArenaSim`, `src/arena_core.js`, unchanged since before this pin):
  live 4D sports-match reconstruction. See "Arena" section below.
- CDN deps (all three pages): three.js `0.158.0` (non-module build), plus for the
  deck: reveal.js `5.1.0`, KaTeX, Chart.js. All served as static HTML — no build
  step at request time, no server code; presenting the deck needs internet for CDNs.
- The in-page version badge (bottom corner) currently reads **"v18"** — this is a
  stale literal in `sim_core.js`/`index.html` (`SIM_VERSION`), not accurate; commit
  messages informally call this state "v21". Don't infer the feature set from the
  on-screen badge text.

## First thing a user sees: the onboarding gallery (`index.html` only)
Three screens, `KnotSwarmSim.onboard(container, opts)`, each with **live rotating
preview canvases sampled from the real simulator fields** (not decorative mockups
— same field/props invariant as the sim itself):
1. **MAP** — one orbiting-camera preview card per environment (urban/forest/ocean/
   dust), heightfield mesh built from that env's actual `ENVS.h()/col()`.
2. **REPRESENTATION** — side-by-side splat vs. mesh preview of the SAME patch,
   using the sim's real splat sprite (`mkSplatAlphaTex`) and the real mesh-stitch
   wire pattern (tri/quad/tetra sub-choice).
3. **RESOLUTION & SAMPLING** — sliders for params density (×1–×16) and control-net
   UV (48–192), a 3-way sampling toggle (adaptive/uniform/fluid — fluid is disabled
   whenever mesh recon is selected, since stitching needs the S×S grid), and a
   budget picker (1k–512k). A live canvas diagram animates the control-net grid
   size toward `round(cnet * sqrt(den))` and shows where samples concentrate under
   the chosen sampling mode (uniform grid / importance-warped dots / fluid jitter).

Default selections (pre-highlighted, matching the sim's own defaults): forest,
64k budget, splats, tri mesh-kind (if mesh chosen), ×2 density, 96×96 control net,
adaptive sampling. Clicking "▶ START" on screen 3 tears every preview WebGL
context down and boots the real sim with those choices via
`create(container, {...opts, initial: {...selections}})`.

## The simulator itself, once running
**Defaults on boot** (whether reached via onboarding picks or `deck.html`'s direct
`create()`): forest environment, 64k Gaussian budget, MANUAL mode, Catmull-Rom
basis, ×2 param density, 96×96 control net, adaptive sampling, 3DGS recon (tri
mesh-kind held in reserve), commander-view-as-main (`st.swapped=true`), observer
drone owns the main POV (`st.drone=true`), mesh-in-the-loop Chamfer supervision on
(`st.geoSup=true`).

### Fit model — what's actually being optimized
A control grid of heights `fit{CN,cur,tgt,res,crop,x0,z0,cov,pres,mvc,gres}`,
default control-net resolution `st.cnet=96` (user-adjustable 48–192, step 8),
warmed up through levels `[64,96,128]`. Per-step update (`optimStep`, always
time-based, never per-frame):
- Laplacian smoothness regularizer (base `REG=0.3`).
- `covW` — swarm-view coverage weight (unobserved cells ≈ frozen).
- `peW` — photometric residual weight.
- `mvW` — **multi-view photo-consistency** (new this pin): each drone that covers
  a cell photographs the surface THROUGH the current height hypothesis; parallax
  makes independent views' colors diverge when the height is wrong. Cross-view
  agreement across ≥2 views becomes a confidence signal (`fit.mvc`, HUD-exposed as
  "multi-view N%"); <2 views is "unverified" (mv=0.25, mildly untrusted). Fusion:
  consistent cells trust their own data more (`w *= 0.55+0.45·mv`), poorly-agreed
  cells lean harder on the smoothness prior (`REG *= 1+0.9·(1-mv)`). Gated to
  post-warm-up only (can't judge consistency before a coarse estimate exists).
  `swarmFoot.eye`/`.alt` carry the per-drone eye positions + altitude needed for
  the back-projection geometry that drives this.
- Optional bidirectional-Chamfer mesh-supervision term (`GEO`, on by default,
  toggle in the MODEL menu; residual shown as `fit.gres` in the HUD).
- **Visible side effect in forest**: tree-trunk stems are pinned to the
  reconstructed canopy height (`reconH`), so they can no longer visibly float
  detached from the canopy; a tree unobserved by any drone (swarm blind zone)
  renders no trunk splats at all — trunk opacity/support scales with how many
  drones currently confirm it.

### Sampling modes
`st.adaptive` (default) — inverse-CDF importance warp on curvature/texture/
residual. `st.fluid` — particle-based, settles/freezes when calm, wakes on
change. Mutually exclusive with mesh recon (mesh needs the plain S×S grid;
selecting mesh recon while fluid is active silently falls back to adaptive, both
in onboarding and in the live MODEL/RECON menus).

### Recon + event-driven resample (new this pin)
`updateRecon` still does the same two-pass build (heights, then NN-stretched
splats + facade wall-scan) or 2×2 UV-stitched mesh. What changed: while the
commander frustum is merely panning/rotating/flying (WASD/elev), the code no
longer forces a full S×S rebuild every frame (that used to show as visible sample
"jelly"). Camera-follow motion now sets `camDirty` instead of `reconDirty`
directly; the main loop promotes it to a real rebuild only when the view
**settles** (0.18s idle, `st.resample.settle`), a **periodic timer** elapses
(`st.resample.interval`, default 0.15s, clamped 0.07125–1.0s), or the sampling
window/crop has **drifted** past `st.resample.devThresh` (0.05, gate toggle
`st.resample.devGate`). Discrete changes — env switch, mode/budget/basis change,
subdivision, explicit UI toggles — are untouched and still rebuild immediately.
Debug: `__kss.resampleStats()` → `{camDirty, builds, lastSampleT}`.

### Observer drone / camera (pitch fix, this pin)
`st.drone=true` by default: a kinematic `drone{px,pz,hd,alt,lx,lz,ly,auto,gp}`
owns the main POV. `drone.gp` is the manual gimbal pitch in radians below horizon
(`null` = terrain-follow cinematic gaze). The pitch slider, `T`/`G` keys, and
Shift+↑/↓ now correctly drive `drone.gp` while the drone owns the view (this used
to silently write `cOrbit.ph`, which `droneStep` ignored — fixed). Plain ↑/↓ still
climbs; Shift+↑/↓ is pitch-only now (previously it also climbed). Patrol resume
(5s idle) reverts `gp` to `null`. Controls: joysticks + `elev` (desktop: one
stick; mobile: twin sticks, left=rotate+alt/right=move) or WASD/arrows/QE; guard
condition `st.drone && !st.warm.active`.

### UI chrome (reworked this pin)
- **Left tool rail** (`.kss-tools`) is a **2-column grid** (was a single column)
  so its 10 buttons — nav / ruler / line-of-sight / mark / topo-probe / residual
  heatmap / drone-POV toggle / segmentation / gradient / FIT-to-screen / channel
  cycle (RGB→DEPTH→NORMAL→CONF) — never overflow a short viewport.
- **Collapsible floating bubbles**: the top-left HUD ("RECON"), top-right HUD
  ("SWARM"), the segmentation panel, the gradient panel, and the commander
  text-readout each have a `.kss-fold` (▾/▸) toggle that hides the body via a
  shared `.kss-min` class — same pattern across all five.
- **Dock menus** (`mode`, `env`, `budget`, `swarm`, `model`, `export`, `show`,
  `subs`) now stack controls in labeled `.kss-mrow` rows instead of one long wrap,
  for legibility with the taller touch-friendly buttons.

### HUD field reference (top-left "RECON", top-right "SWARM")
Top-left: splat/face count + budget · surface vs. facade split (or mesh
tri/quad/tetra face count) · pitch · sampling-mode badge · crop size · footprint ·
GRID reference · **swarm-view coverage %** · **photometric residual** ·
**multi-view N%** (cyan, new) · Chamfer residual (or "mesh-loss off") · domain
resolution · fit residual · LOD multiplier · multi-spline count/local-res (if any
`SUBS` exist) · BOOST flag · view-driven window %. Top-right: mode (AUTO/MANUAL +
phase) · drone count (+ "(adaptive)" if `st.drAdapt`) · altitude · overlap % ·
drone footprint · observer-drone-or-commander GRID position · residual-view flag.

### Other subsystems (unchanged in behavior this pin, still live)
- **Commander tools**: ruler+LOS, tactical markers, TACREF (MGRS-style grid +
  confidence, clipboard copy), pitch slider, LINK/FREE view toggle.
- **Segmentation** (`seg`): world-anchored mask (`seg.gmask`, 384² over ±120m),
  seeded mean-split + flood-fill on RGB/DEPTH/NORMAL, strokes aggregate until
  `revert`.
- **Gradient readout**: slope map with downhill arrows.
- **Mesh recon** (`st.recon='mesh'`): live 2×2 UV-stitched tri/quad/tetra
  connectivity, whole budget on the surface, no facade pool.
- **Multi-spline decomposition** (`SUBS`): detach a seg-masked region into its own
  finer world-anchored spline, refine independently; `autoDecompose` bulk-detects
  high-error/steady-climb regions and gives the top 4 their own splines; SPLINES
  dock menu = per-surface hide/refine/remove + draped boundary-contour overlay.
- **Export**: 2×2 UV-kernel sweep animation → binary PLY download.
- **Mobile**: twin-stick controls, zen-on-move, aspect-correct portrait layout.
- Basis switch (MODEL menu): Catmull-Rom (default) / NURBS / uniform B-spline
  degree 2–5.

### Debug handle (browser console / headless tests)
```js
window.__kss = { fit, swarmFoot, st, FLU, topo, seg, grad, drone, SUBS,
                  resampleStats, segStats }
__kss.segStats()       // {visible, inMask} splat counts
__kss.resampleStats()  // {camDirty, builds, lastSampleT}
```

## Arena (`/arena.html`) — unchanged this pin, still exactly as documented in
`CLAUDE.md`'s "arena.html" section: fixed perimeter+roof swarm around a
basketball/football court (`COURTS.{basketball,football}`), free-fly WASD+mouse
camera INSIDE the live reconstruction (no commander split-view), open-Catmull-Rom
4D position estimate evaluated `LATENCY=0.22s` behind ground truth, confidence-
driven splat billboards reusing the terrain sim's falloff sprite. Scoped out:
mobile touch controls, occlusion-aware visibility.

## Deliberately NOT covered here
Anything that exists only on a feature branch, in an open PR, or in this
worktree's uncommitted state is out of scope for this document by design — it
isn't live. If you need the bleeding edge instead of the deployed truth, read
`git log`/the branch directly rather than trusting this file for it.
