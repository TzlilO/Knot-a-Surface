---
name: sim-verify
description: Verify the simulator or deck in a headless browser after changes — console errors, HUD state, interactions, screenshots. Use before every commit/deploy, and whenever asked "does it work", "test it", "check the sim".
---

# Headless verification

Requires playwright + chromium (`pip install playwright && playwright install chromium`).

`index.html` boots into the pre-run **onboarding gallery** (`KnotSwarmSim.onboard`),
not the sim directly — click `.kss-ob-next` three times (MAP → REPRESENTATION →
RESOLUTION & SAMPLING → START) before `.kss-hud-tl`/`.kss-cmdr-canvas` exist. Onboarding
errors (broken preview canvases, `mkSplatAlphaTex`/`patchGeo` failures) are just as
real as sim errors — don't skip past them by waiting the click-through out. `deck.html`'s
embedded slide calls `KnotSwarmSim.create` directly and skips onboarding — if you
extract/test the deck's embed script standalone, no click-through is needed.

## Standard pass
```bash
sed 's/budget: 65536/budget: 4096/' index.html > /tmp/t.html   # 4k: headless-friendly
python3 - <<'PY'
from playwright.sync_api import sync_playwright
errors=[]
with sync_playwright() as p:
    b=p.chromium.launch(args=["--no-sandbox","--headless=new","--disable-gpu","--single-process"])
    pg=b.new_page(viewport={"width":1100,"height":650})
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type=="error" else None)
    pg.goto("file:///tmp/t.html")
    pg.wait_for_timeout(600)
    for _ in range(3):                       # click through onboarding (MAP/REP/RES) to START
        pg.click(".kss-ob-next"); pg.wait_for_timeout(300)
    pg.wait_for_timeout(6000)
    print(pg.inner_text(".kss-hud-tl"))          # splat counts, domain, residual, coverage
    pg.screenshot(path="/tmp/shot.png")
    print("ERRORS:", errors or "none")
    b.close()
PY
```

## What to check per subsystem
- Any change: zero console/page errors; HUD lines sane (budget split, domain,
  residual falling, coverage %, photometric res).
- Scene/props: screenshot; toggle to the env via
  `pg.click(".kss-dbtn[data-open='env']")` then `.kss-btn.kss-env[data-env='X']`.
- Optimization: poll residual over time — it must decrease; after warm-up
  `.kss-phase` disappears.
- Fluid mode: `window.__kss.FLU.frozen` must become true (settle) and particle
  coords static while frozen.
- Geometry supervision: with `st.geoSup` on, `fit.gres` (Chamfer, in the HUD) must
  fall over time; A/B a change against `st.geoSup=false` before trusting it.
- Multi-view photo-consistency: `fit.mvc` ("multi-view N%" in the HUD, cyan) should
  rise as swarm redundancy increases (more drones / more overlap) and should NOT be
  computed during warm-up (`hasMV = ... && !st.warm.active`). In forest, confirm no
  scattered floating tree-trunk splats — stems must terminate at `reconH`, not at a
  fixed `hh*0.5`, and unobserved trees (swarm blind zones) should render nothing.
- Event-driven resample: `window.__kss.resampleStats()` → `{camDirty, builds,
  lastSampleT}`. While panning/rotating/flying the commander view continuously,
  `builds` must NOT increment every frame — only on settle (~0.18s idle), the
  `st.resample.interval` timer (default 0.15s), or window drift past `devThresh`.
  Discrete changes (env/mode/budget switch, subdivision) must still resample
  immediately regardless of this gate.
- Drone: `window.__kss.drone` moves under keys/joystick and returns to autonomous
  patrol after idle; the mask must NOT move with it — check `__kss.segStats()` counts
  are stable as the drone flies (world-anchored). Segmentation should aggregate across
  strokes and only reset on `revert`. Pitch control: with `st.drone` true, the pitch
  slider / T,G / Shift+↑↓ must move `drone.gp` (readable via `__kss.drone.gp`) and
  visibly tilt the main POV gimbal — NOT `cOrbit.ph`, which `droneStep` ignores while
  the drone owns the view. Patrol resume (5s idle) should revert `drone.gp` to `null`
  (terrain-follow gaze resumes).
- Onboarding (`index.html` only): each of the 3 screens renders live preview canvases
  with zero console errors; selecting a map/representation/resolution option updates
  the `kss-on` highlight; the control-net diagram's shown grid count animates toward
  `round(cnet * sqrt(den))` on slider input; clicking "▶ START" on screen 3 tears down
  the gallery and the real sim (HUD, dock, tool rail) appears with the chosen options
  already applied (check `__kss.st.env`/`.budget`/`.recon`/`.cnet` match the picks).
- Menus/zen/fold: class assertions (`kss-folded`, `kss-zenmode`, `kss-open`);
  moving-drone zen toggles `kss-moving`. Collapsible bubbles (HUD/seg/grad/commander
  readout): clicking `.kss-fold` toggles `kss-min` on the parent and flips the ▾/▸ glyph.
- Export: `pg.expect_download()` around the EXPORT PLY click; validate the PLY
  header math (bytes = header + nV*15 + nF*13).
- Mobile: rerun with `viewport 390×740, is_mobile=True, has_touch=True` — expect the
  `kss-mobile` class, twin sticks, and a non-distorted (aspect-correct) main view.
  A fresh browser per context: `--single-process` can't hold two.

## Caveats
- No GPU in most sandboxes: 64k+ budgets render at seconds/frame — always test
  logic at 4k; game-time (dt-capped) runs ~3× slower than wall-clock, so double
  every timing expectation.
- Debug handle: `window.__kss = {fit, swarmFoot, st, FLU, topo, seg, grad, drone,
  SUBS, resampleStats, segStats}` + `__kss.segStats()` + `__kss.resampleStats()`.
- Screenshots: read them — layout regressions don't throw errors.
