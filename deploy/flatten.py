#!/usr/bin/env python3
"""Flatten the .dc.html design files into the deployed sibling-file site.

The .dc.html files are the design source of truth. The deployed pages are the
same documents under flat filenames with internal links rewritten — they load
support.js, the _ds design tokens and the simulator cores as SIBLING files, so
there is no bundling step and deploying is a push to main. Re-run this after
editing any .dc.html file.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# design source -> deployed path
MAP = {
    "Project Page.dc.html": "index.html",
    "Sim v1 - Swarm Core.dc.html": "terrain.html",
    "Sim v2 - UV Atlas.dc.html": "terrain-atlas.html",
    "Sim v3 - Knot Chart.dc.html": "knot.html",
    "Sim v4 - Surface Prior.dc.html": "surface-prior.html",
    "Teaser.dc.html": "teaser-v1.html",
    "Teaser v2 - Bold.dc.html": "teaser-bold.html",
    "Teaser v2 - Modest.dc.html": "teaser-modest.html",
    "Teaser v4 - Cinematic.dc.html": "teaser-cinematic.html",
    "Review Hub.dc.html": "review/index.html",
}


def flatten(src, dst):
    txt = (ROOT / src).read_text()
    for s, d in MAP.items():
        txt = txt.replace(s, d)
    if "/" in dst:
        # review/index.html: assets and pages live one level up, at the repo root
        txt = (txt.replace('src="./support.js"', 'src="../support.js"')
                  .replace('href="_ds/', 'href="../_ds/')
                  .replace('src="_ds/', 'src="../_ds/')
                  .replace('src="src/', 'src="../src/')
                  .replace('href="src/', 'href="../src/'))
        for d in set(MAP.values()):
            if "/" in d:
                continue
            txt = txt.replace(f'"{d}"', f'"../{d}"').replace(f"'{d}'", f"'../{d}'")
        # the published hub always uses the deployed (flat) link map
        txt = txt.replace('&quot;default&quot;:&quot;local&quot;',
                          '&quot;default&quot;:&quot;deployed&quot;')
    out = ROOT / dst
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt)
    print(f"{src} -> {dst}")


if __name__ == "__main__":
    for src, dst in MAP.items():
        flatten(src, dst)
