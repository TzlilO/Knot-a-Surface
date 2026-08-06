#!/usr/bin/env bash
# The review hub is now just another flattened page (review/index.html, built by
# deploy/flatten.py from "Review Hub.dc.html") and publishes with the rest of the
# site. This wrapper stays for muscle memory — it does the same as deploy-site.sh.
set -euo pipefail
exec "$(dirname "$0")/deploy-site.sh" "${1:-Publish site incl. review hub}"
