#!/usr/bin/env python3
"""Build index.html (simulator) + deck.html from src/ parts.
Run from the repo root:  python3 build.py"""
from datetime import datetime, timezone
S = 'src/'
built = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
core = open(S + 'sim_core.js').read().replace('/*__BUILD_TIME__*/', built)
css  = open(S + 'sim_style.css').read()
deck = open(S + 'deck_a.html').read() + open(S + 'deck_b.html').read() + open(S + 'deck_c.html').read()
open('deck.html', 'w').write(deck.replace('/*__SIM_CSS__*/', css).replace('/*__SIM_CORE__*/', core))
sim = open(S + 'swarm_simulator_template.html').read()
open('index.html', 'w').write(sim.replace('/*__SIM_CSS__*/', css).replace('/*__SIM_CORE__*/', core))
arena_css = open(S + 'arena_style.css').read()
arena_core = open(S + 'arena_core.js').read()
arena = open(S + 'arena_template.html').read()
open('arena.html', 'w').write(arena.replace('/*__ARENA_CSS__*/', arena_css).replace('/*__ARENA_CORE__*/', arena_core))
print('built index.html + deck.html + arena.html (' + built + ')')
