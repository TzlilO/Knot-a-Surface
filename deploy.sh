#!/usr/bin/env bash
# One-time GitHub Pages deploy. Requires: gh CLI, logged in (gh auth login).
# Usage: ./deploy.sh <repo-name>   e.g. ./deploy.sh knot-a-surface
set -euo pipefail
REPO="${1:-knot-a-surface}"
gh repo create "$REPO" --public -y 2>/dev/null || true
git init -b main .
git add index.html deck.html README.md
git commit -m "Knot-a-Surface simulator + deck"
git remote add origin "https://github.com/$(gh api user -q .login)/$REPO.git" 2>/dev/null || true
git push -u origin main --force
gh api "repos/$(gh api user -q .login)/$REPO/pages" -X POST -f "source[branch]=main" -f "source[path]=/" 2>/dev/null || true
echo "Live in ~1 min at: https://$(gh api user -q .login).github.io/$REPO/"
