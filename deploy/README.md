# Publishing Knot-a-Surface

Pages for this repo is served from **main → / (root)**.

## 1. Main page (replaces the old root page and its simulator widget)
    https://tzlilo.github.io/Knot-a-Surface/

| File | Page |
| --- | --- |
| site/index.html | Project page — hero, embedded Teaser v2 (Bold), simulator cards, method |
| site/teaser.html | Teaser v2 — Bold (1:28 cut) |
| site/knot.html | Sim v3 — Knot Chart |
| site/terrain.html | Swarm Resolution-Knob Simulator — the src/sim_core.js build that used to be the root index.html |
| site/terrain-atlas.html | Sim v2 — UV Atlas (illustrative variant) |

```bash
cd /path/to/Knot-a-Surface        # local clone, main branch
bash deploy/deploy-site.sh        # copies deploy/site/* to the repo root, commits, pushes
```
This overwrites the previous root `index.html` (the old simulators widget page). The simulator
itself is not lost: the same build is published as `terrain.html` and linked from the project page.

## 2. Review hub (independent second page, unchanged)
    https://tzlilo.github.io/Knot-a-Surface/review/

| File | Page |
| --- | --- |
| review/index.html | Review hub |
| review/teaser-bold.html | Teaser v2 — Bold |
| review/teaser-modest.html | Teaser v2 — Modest |
| review/teaser-v1.html | Original teaser |
| review/terrain.html | Swarm Resolution-Knob Simulator (deployed core) |
| review/terrain-atlas.html | Sim v2 — UV Atlas (illustrative variant) |
| review/knot.html | Sim v3 — Knot Chart |

```bash
bash deploy/deploy-review.sh
```

All files are self-contained (three.js, fonts and runtime inlined). Sources live in the design
project under `build/` (main page) and `build/review/` (review hub); `docs/` keeps a mirror.
