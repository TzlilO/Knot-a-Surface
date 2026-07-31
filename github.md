repo: TzlilO/Knot-a-Surface
branch: main
path: src/

## Last sync
date: 2026-07-31T13:20:00Z

### Updated in this project
- `src/sim_v4_core.js` — fixed the splats popping instead of flowing when the camera zooms or
  orbits. Root cause: a splat's identity was its rank in a global CDF over a **Morton**-ordered
  triangle domain, so any camera change restretched the CDF and re-drew the whole cloud, and
  Morton's spatial discontinuities (half of all steps non-unit, up to 63 cells) turned small
  rank slides into cross-object jumps that `adEase` then hard-teleported. Four changes:
  Hilbert ordering (`adBuildDomain`), a time-eased weight field driving the CDF (`AD.ws`,
  `AD_FLOW`), local surplus→nearest-deficit transport replacing the per-refresh redraw
  (`adTransport`, `adPlace`, `adMark`), and a shrink-fade for relocations too long to walk
  (`AD.scl`/`AD.pend`, reusing `aScale` on the neighbour-spacing shader path).
  Measured on the real 60 Hz loop: teleporting slot-frames 25.2% → 3.3% on a fast zoom,
  27.2% → 2.5% on a fast orbit, and 100% of what remains happens while the splat is scaled to
  zero. Mean splat speed 2.24 → 0.44 R/s. Allocation is unchanged at rest (concentration 7.65
  vs 7.63, in-view 89.0% vs 88.7%). Refresh cost 4.6 → 6.9 ms at 12k budget, 7.0 → 12.0 ms at
  57k. Debug handle `controller._dbg` added (mirrors `window.__kss` in the swarm sim).
  No deploy bundle regeneration needed — v4 has no published bundle yet, and `build.py` does
  not consume `sim_v4_core.js`, so `index.html` / `deck.html` / `arena.html` are unaffected.

### Updated 2026-07-29
- New `Sim v4 - Surface Prior.dc.html` + `src/sim_v4_core.js` (local, not upstream): a side-by-side reconstruction bench that runs our surface-coupled model against PGSR, 2DGS, SuGaR, 3DGS, NeuS/VolSDF and MVS on the same capture, step counter and camera. Fixed-budget vs densified primitive counts, DoF ledgers, view-driven allocation and zoom re-resolution are live; convergence / chamfer / photometric numbers are placeholder curves awaiting measured DTU + TnT benchmarks.
- Capture proxies (TnT Barn, TnT Truck, DTU scan110 bust) are procedurally built stand-ins, sampled into a 110k-point surface pool that both models draw from — so every visual difference comes from the representation, not the scene.
- v1–v3 untouched; the word "arena" is gone from project-authored files (remaining hits are design-system source comments and pre-built bundles).

## Deploy
### Review deployment (alternative page)
url: https://tzlilo.github.io/Knot-a-Surface/review/
files: deploy/review/{index,teaser-v1,teaser-bold,teaser-modest,terrain,terrain-atlas,knot}.html + .nojekyll, deploy/deploy-review.sh, deploy/README.md
notes: repo Pages source is main / (root), so the alternative page ships as a repo-root /review/ folder — the existing root index.html is untouched. Push with deploy/deploy-review.sh from a local clone. Sources in build/review/; docs/ keeps a mirror copy.

### Main page
target: GitHub Pages — main / (root), pushed by deploy/deploy-site.sh
files: deploy/site/{index,teaser,terrain,terrain-atlas,knot}.html + .nojekyll
notes: every file is self-contained (three.js, sim_core.js, fonts and support runtime inlined); sources live in build/ and are regenerated from the .dc.html designs. Replacing the root index.html does NOT retire the simulator — the same sim_core.js build is published as terrain.html.

## Screen map
| File | Built from |
| --- | --- |
| Sim v1 - Swarm Core.dc.html | src/sim_core.js, src/sim_style.css, src/swarm_simulator_template.html (deployed core + local camera-detach fix) |
| Sim v2 - UV Atlas.dc.html | CLAUDE.md (fit model, seg, recon, splat billboards), src/sim_style.css, src/swarm_simulator_template.html |
| Sim v3 - Knot Chart.dc.html | CLAUDE.md (fly camera, seg, splats), src/sim_style.css |
| Sim v4 - Surface Prior.dc.html | src/sim_v4_core.js (new), representation ledgers for PGSR / 2DGS / SuGaR / 3DGS / NeuS / MVS |
| Project Page.dc.html | project.html (hero/teaser/demo/method/footer structure and design tokens) |

## Sync history
### 2026-07-29T13:20:00Z
- Camera fully detached from the swarm in the imported core; default gimbal pitch 0.64 rad; cured the "goes dark" patrol drift.
### 2026-07-29T12:40:00Z
- Imported the real deployed simulator (`src/sim_core.js` + `src/sim_style.css`, the build behind the live root `index.html`) and mounted it as a Design Component.
- The terrain slot in both deployments is now that core simulator; the lightweight UV-atlas sim moved to `terrain-atlas.html`.
### 2026-07-28T18:20:00Z
- Main project page republished at the site root (Teaser v2 Bold embedded, simulator cards rebuilt); review hub added at docs/review/; Sim v2 ROI frustum culling; decomposition, density field and direct 3D editing in both sims.
### 2026-07-28T00:00:00Z
- Decomposition, density field, direct 3D editing in both sims; review hub deployment.
### 2026-07-27T13:44:28Z
- Built the two simulators and the project page from src/ + CLAUDE.md.
