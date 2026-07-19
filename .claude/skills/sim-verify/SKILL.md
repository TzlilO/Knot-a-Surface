---
name: sim-verify
description: Verify the simulator or deck in a headless browser after changes — console errors, HUD state, interactions, screenshots. Use before every commit/deploy, and whenever asked "does it work", "test it", "check the sim".
---

# Headless verification

Requires playwright + chromium (`pip install playwright && playwright install chromium`).

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
    pg.goto("file:///tmp/t.html"); pg.wait_for_timeout(6000)
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
- Menus/zen/fold: class assertions (`kss-folded`, `kss-zenmode`, `kss-open`).
- Export: `pg.expect_download()` around the EXPORT PLY click; validate the PLY
  header math (bytes = header + nV*15 + nF*13).
- Mobile: rerun with `viewport 390×740, is_mobile=True, has_touch=True`.

## Caveats
- No GPU in most sandboxes: 64k+ budgets render at seconds/frame — always test
  logic at 4k; game-time (dt-capped) runs ~3× slower than wall-clock, so double
  every timing expectation.
- Debug handle: `window.__kss = {fit, swarmFoot, st, FLU}`.
- Screenshots: read them — layout regressions don't throw errors.
