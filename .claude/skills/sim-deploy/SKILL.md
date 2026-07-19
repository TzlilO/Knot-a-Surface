---
name: sim-deploy
description: Build and deploy the simulator + deck to GitHub Pages. Use for "deploy", "push", "publish", "update the site", "go live".
---

# Deploy to GitHub Pages

Repo: `TzlilO/Knot-a-Surface` (capital K — Pages URL path is case-sensitive:
https://tzlilo.github.io/Knot-a-Surface/).

## Steps
1. `python3 build.py`  (regenerates index.html + deck.html from src/)
2. Run /sim-verify — deploy only on zero errors.
3. `git add -A && git commit -m "<what changed>" && git push`
4. Pages rebuilds in ~1 min. Verify the live copy actually updated:
   ```bash
   curl -s https://tzlilo.github.io/Knot-a-Surface/ | grep -c "<marker from your change>"
   ```
   Cache is `max-age=600` — check with a `?v=N` query or curl, not a stale browser tab.

## Gotchas
- Never commit generated files without rebuilding first — src/ and index.html
  must not drift.
- deck.html embeds the same sim core; build.py keeps them in sync automatically.
- All CDN deps (three.js r158 non-module, reveal.js 5, KaTeX, Chart.js) — the host
  only serves static HTML; presenting requires internet.
