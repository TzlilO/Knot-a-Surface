#!/usr/bin/env python3
"""Build deck.html + arena.html from src/ parts.
Run from the repo root:  python3 build.py

NOTE: the root index.html is NO LONGER built here — the site root is the
landing page flattened from "Project Page.dc.html" by deploy/flatten.py, and
the swarm simulator ships as terrain.html ("Sim v1 - Swarm Core.dc.html"),
sibling-loading src/sim_core.js + src/sim_style.css with no build step.
src/swarm_simulator_template.html is kept for reference but is no longer
consumed."""
from datetime import datetime, timezone
S = 'src/'
built = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
core = open(S + 'sim_core.js').read().replace('/*__BUILD_TIME__*/', built)
css  = open(S + 'sim_style.css').read()
tokens = open(S + 'design_tokens.css').read()
deck_theme = open(S + 'theme_v2.css').read()
deck = open(S + 'deck_a.html').read() + open(S + 'deck_b.html').read() + open(S + 'deck_c.html').read()
open('deck.html', 'w').write(deck.replace('/*__DESIGN_TOKENS__*/', tokens)
                                  .replace('/*__DECK_THEME__*/', deck_theme)
                                  .replace('/*__SIM_CSS__*/', css).replace('/*__SIM_CORE__*/', core))
arena_css = open(S + 'arena_style.css').read()
arena_core = open(S + 'arena_core.js').read()
arena = open(S + 'arena_template.html').read()
open('arena.html', 'w').write(arena.replace('/*__DESIGN_TOKENS__*/', tokens)
                                    .replace('/*__ARENA_CSS__*/', arena_css).replace('/*__ARENA_CORE__*/', arena_core))
print('built deck.html + arena.html (' + built + ')')
