# Knot-a-Surface

NURBS-grounded Gaussian splatting: splat positions are never free in 3D — they are
derived from a single (u,v) parametrization of the underlying surface.

Live: **https://tzlilo.github.io/Knot-a-Surface/**

## The site

| URL | Page |
| --- | --- |
| `/` | Landing page — a live spline-chart scene (barn / truck / bust / garden), the idea in three moves, the talk and its lab modules |
| `/project-page.html` | Previous landing page — hero, embedded teaser, simulator cards, method, **measured results** |
| `/terrain.html` | Sim v1 — Swarm Resolution-Knob simulator (drone swarm, fixed splat budget, commander toolkit) |
| `/terrain-atlas.html` | Sim v2 — UV-atlas editing (segment, paint, sculpt in the chart) |
| `/knot.html` | Sim v3 — Knot Chart (8 parametric surfaces, control cage, geometry layers) |
| `/surface-prior.html` | Sim v4 — Surface Prior bench: ours vs PGSR / 2DGS / SuGaR / 3DGS / NeuS / MVS, all rates **measured in-page** against the procedural ground truth |
| `/v4.html` | Sampler lab for the uvUniform / uvAdaptive (u,v)-atlas samplers |
| `/teaser-{v1,bold,modest,cinematic}.html` | Four teaser cuts |
| `/review/` | Review hub for the cuts + sims |
| `/deck.html` | Interview deck (reveal.js), simulator embedded as slide 14 |
| `/talk.html` | Thesis talk — scroll-driven deck (20′ / 40′ cuts), from one spline to a manifold of spline charts |
| `/talk/sims/{flatland,basis,planoid,ledger,refine,manifold,structure,touch}.html` | The talk's eight simulations as standalone lab pages |
| `/lab.html` | Lab — an interactive figure: the Utah teapot and other objects as spline charts, studio-lit Gaussians, baked DINOv3 control features, feature-based selection and editing |
| `/talk-he.html` | Short Hebrew deck (5–10 min) — explicit vs implicit 3D representations and Gaussian Splatting; light "liquid glass" theme |
| `/arena.html` | Arena: live 4D match reconstruction (second application of the thesis) |
| `/project.html` | Retired predecessor landing page (kept live, unlinked) |

## Developing

- **The landing page** (`index.html`) is hand-written and is its own source of truth; it mounts the
  talk's `manifold` module as a chrome-free hero.
- **Source of truth for the other pages**: the `*.dc.html` design files. The deployed pages
  are flattened from them by `python3 deploy/flatten.py` (link rewrites only — no
  bundling; everything sibling-loads `support.js`, `_ds/` tokens and the `src/` cores).
  Never edit the flattened outputs (`project-page.html`, `terrain.html`, …) by hand.
- **Simulator cores**: `src/sim_core.js` (swarm sim), `src/sim_v4_core.js` (v4 bench),
  `src/arena_core.js`. `python3 build.py` injects sim_core/sim_style into `deck.html`
  and arena_core/arena_style into `arena.html` (the only remaining build step; the site
  pages themselves sibling-load the cores).
- **Thesis talk**: `talk.html` + `talk/` are plain static files with no build step
  (`kit.js` module registry, `nurbs.js` spline math, `gl.js` three.js helpers, `scenes.js` spline-chart
  scenes (barn / truck / bust / garden stand-ins, and `scan`: a real DTU object fitted offline by
  `tools/dtu/fit.py` into `talk/dtu-data.js`), `deck.js`/`deck.css`
  scroll engine, one module per simulation in `talk/sims/`). three.js and KaTeX are vendored in
  `talk/vendor/` so the talk runs offline. The evidence numbers live at the top of `talk/deck.js`.
- **Deploy**: `bash deploy/deploy-site.sh` = flatten + build + commit + push to `main`
  (GitHub Pages, root `.nojekyll` required for `_ds/`). See `deploy/README.md`.

## Conventions

Design tokens in `_ds/…/tokens/design_tokens.css` — instrument palette for HUD chrome,
never invent new colors. Working agreements for the cores live in `CLAUDE.md`; repo
state receipts in `github.md`; the original design-handoff doc in `docs/HANDOFF.md`.
