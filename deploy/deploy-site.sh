#!/usr/bin/env bash
# Publishes the MAIN project page (root of the site), REPLACING the old root index.html
# and the simulator widget it carried. Live at: https://tzlilo.github.io/Knot-a-Surface/
# NOTE: the swarm simulator that used to BE the root index.html now ships as terrain.html
# (same src/sim_core.js build), so replacing the root page does not retire it.
# Run from the root of a local clone of TzlilO/Knot-a-Surface.
set -euo pipefail
SRC="${1:-deploy/site}"
cp "$SRC"/index.html "$SRC"/teaser.html "$SRC"/knot.html "$SRC"/terrain.html "$SRC"/terrain-atlas.html .
cp "$SRC"/.nojekyll . 2>/dev/null || true
git add index.html teaser.html knot.html terrain.html terrain-atlas.html .nojekyll
git commit -m "Replace root project page: v2 Bold teaser + updated simulators"
git push origin main
echo "Live in ~1 min at: https://tzlilo.github.io/Knot-a-Surface/"
