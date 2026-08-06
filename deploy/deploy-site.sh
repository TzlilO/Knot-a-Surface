#!/usr/bin/env bash
# Publish the site: flatten the .dc.html designs into the deployed sibling-file
# pages, then commit + push main (GitHub Pages serves main / root in ~1 min).
#
# There is NO bundle step anymore: deployed pages load support.js, the _ds
# tokens and the simulator cores as sibling files. Edit the .dc.html designs
# (or src/ cores), run this, done. Run from the repo root.
set -euo pipefail
python3 deploy/flatten.py
python3 build.py
git add -A
git commit -m "${1:-Publish site (flatten designs + rebuild deck/arena)}"
git push origin main
echo "Live in ~1 min at: https://tzlilo.github.io/Knot-a-Surface/"
