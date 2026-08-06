#!/usr/bin/env bash
# Publishes the review hub as an ALTERNATIVE GitHub Page, alongside the existing site.
# Existing page:    https://tzlilo.github.io/Knot-a-Surface/        (root index.html — untouched)
# Alternative page: https://tzlilo.github.io/Knot-a-Surface/review/
# Run from the root of a local clone of TzlilO/Knot-a-Surface, with this deploy/ folder next to it.
set -euo pipefail
SRC="${1:-deploy/review}"
mkdir -p review
cp "$SRC"/*.html review/
cp "$SRC"/.nojekyll review/ 2>/dev/null || true
git add review
git commit -m "Add alternative review page at /review/"
git push origin main
echo "Live in ~1 min at: https://tzlilo.github.io/Knot-a-Surface/review/"
