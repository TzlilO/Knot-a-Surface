# Handoff: Knot-a-Surface — continuing in Claude Code

## Overview
Knot-a-Surface is a NURBS-grounded Gaussian-splatting project: splat positions are never free in 3D,
they are derived from a single (u,v) parametrization of the underlying surface. This bundle contains
the **project page, review hub, four interactive simulators and four teaser cuts** as they exist
today, plus the simulator cores they run on, so you can keep working on them from a terminal with
Claude Code.

The GitHub source of truth:

```
repo:   TzlilO/Knot-a-Surface
branch: main
pages:  main / (root)      → https://tzlilo.github.io/Knot-a-Surface/
                            https://tzlilo.github.io/Knot-a-Surface/review/
```

## About the design files
The `*.dc.html` files in this bundle are **design/prototype files, not the production build**. Each one
is a single HTML document: an inline-styled markup template plus one JavaScript class holding its logic.
They open in a browser over a static server and they expect `support.js` (included) to sit next to them.

Treat them as the reference for look and behaviour. The *shipping* artifacts are the self-contained
single-file HTML bundles in `deploy/site/` and `deploy/review/` (three.js, the simulator core, fonts
and the runtime are inlined into each file).

**Where the real work is:** the simulators are not written in the `.dc.html` files. They live in
plain JavaScript that Claude Code can edit directly:

| File | Lines | What it is |
| --- | --- | --- |
| `src/sim_core.js` | ~4.3k | The deployed swarm simulator. Exposes `window.KnotSwarmSim.create(el, opts)` and `.onboard(el, opts)` (onboarding walkthrough). Warm-up subdivision of the parameter domain, frustum-aware budget allocation, inverse-CDF sampling, coarse-to-fine refinement. |
| `src/sim_v4_core.js` | ~2.6k | The v4 "Surface Prior" reconstruction bench: our model vs PGSR, 2DGS, SuGaR, 3DGS, NeuS/VolSDF, MVS on one capture. **Local only — not yet upstream.** |
| `src/sim_style.css` | ~290 | The instrument/HUD stylesheet (`.kss-*`) shared by the simulator builds. |

The `.dc.html` wrappers just mount those cores and own the page chrome. Sim v2 and v3 are the
exception — their canvas rendering lives inside the `.dc.html` logic class itself.

## Fidelity
**High fidelity.** Colors, type, spacing and interaction states are final and come from the bound
design system. Keep the exact values below; don't re-derive them.

## Getting started in the terminal

```bash
git clone https://github.com/TzlilO/Knot-a-Surface.git
cd Knot-a-Surface

# drop this bundle's contents in (keeps repo files, adds the design files + v4 core)
unzip ~/Downloads/design_handoff_knot_a_surface.zip -d /tmp/kas
cp -R /tmp/kas/design_handoff_knot_a_surface/. .

python3 -m http.server 8000          # any static server
# → http://localhost:8000/Project%20Page.dc.html
# → http://localhost:8000/deploy/site/terrain.html   (self-contained build)

claude
```

Everything is static — no npm install, no build step, no bundler. three.js loads from a CDN pin
(`three@0.158.0`) in the design files and is inlined in the deploy bundles.

### First prompt to give Claude Code

> Read `github.md` and `CLAUDE.md` first. The simulators are `src/sim_core.js` (deployed swarm sim,
> exposes `window.KnotSwarmSim`) and `src/sim_v4_core.js` (v4 reconstruction bench). The `*.dc.html`
> files are design prototypes that mount those cores — inline styles only, one logic class each, no
> stylesheets or CSS classes. Design tokens are in
> `_ds/knot-a-surface-design-system-*/tokens/design_tokens.css`; use the instrument palette for HUD
> chrome and never invent new colors. When you change `src/`, tell me which deploy bundles need
> regenerating, and update `github.md`'s `## Last sync` and `## Screen map` as the receipt.

### Deploying

```bash
bash deploy/deploy-site.sh     # publishes deploy/site/*   → repo root (main project page)
bash deploy/deploy-review.sh   # publishes deploy/review/* → /review/ (review hub)
```

Both run from the root of a local clone and push to `main`. The bundles under `deploy/` were generated
from the design files; **they do not update themselves when you edit `src/`**. For terminal work the
simpler maintenance path is to have the published pages load `sim_core.js` and `sim_style.css` as
sibling files instead of inlining them — ask Claude Code to make that change once and the deploy step
becomes a plain file copy.

## Screens / views

All screens share the same dark instrument aesthetic. Page chrome is Inter with Space Grotesk
headings; every HUD, badge, meta line and readout is JetBrains Mono, uppercase, letter-spaced.

### 1. Project Page (`Project Page.dc.html`)
Public landing page. Max width 1100px (hero and method sections 900px), 24px side padding.
- **Sticky nav** — 12px vertical padding, `rgba(11,16,23,.86)` + 14px backdrop blur, 1px `#1b2a42`
  bottom border. Wordmark left ("Knot-a-" in `#4fc3f7`), then Teaser / Simulators / Method (13px,
  600, `#7f93ad`, hover `#dde5f0` on `rgba(79,195,247,.09)`), then a bordered "Review builds" mono chip.
- **Hero** — 76px top padding. Three pill badges (10.5px mono, 700, .13em tracking, 1.5px border,
  20px radius, 5×12px padding) in cyan / amber / purple. H1 `clamp(2.2rem,5.5vw,3.6rem)`, 700,
  -.02em, line-height 1.05. Lead paragraph Space Grotesk `clamp(1rem,2.2vw,1.28rem)`, 500, `#aeb9cc`,
  max 62ch. Three CTAs: solid `#4fc3f7` on `#04070d` text (hover `#8fd8fb`), and two 2px `#1b2a42`
  outline buttons that tint green / amber on hover. 8px radius, 11×22px padding.
- **Teaser frame** — `#06090e` card, 1.5px `#4fc3f7` border, 12px radius, two 28px corner brackets
  (2.5px, opacity .85) top-left and bottom-right, drop shadow `0 32px 80px rgba(0,0,0,.5)` plus a
  cyan bloom. Top overlay: left status chip with a 6px throbbing dot (`throb` 2s: opacity 1 → .22)
  reading `TEASER · V2 BOLD CUT`, right chip reading `1:44 · 8 chapters`. The player itself is an
  iframe at `aspect-ratio:16/8.6`. Footer strip `#070b12`, 10.5px mono, three fact columns.
  Below the frame, an "Other cuts" row of mono pill links (Modest · 1:13, Cinematic · 1:18,
  Original · 1:00) that tint cyan / amber / green on hover.
- **Simulators** — label + H2 "Four ways to explore the representation", then a
  `repeat(auto-fit,minmax(300px,1fr))` grid, 18px gap, of four cards. Card anatomy (identical
  everywhere in the project): `#0d1220` background, 1.5px border in the version accent at 42%
  alpha, 10px radius, 22px padding; row of 7px accent dot + mono version label; 16.5px Space Grotesk
  title; 13px `#7f93ad` body; a 9.5px mono spec line above a 1px `#1b2a42` divider; accent-colored
  "Open … →" CTA. Hover: border to 90% alpha, background `#101a2a`, `translateY(-2px)`, .16s.
- **Method** — three 15.5px paragraphs, `#8fa4c0`, max 70ch.
- **Footer** — `#070b12`, 1px top border, © 2026 left, five muted links right that tint to their
  section accent on hover.

Tweaks exposed: `featuredCut` (bold / modest / cinematic / original — swaps the embedded player,
its status chip, the duration chip and the caption) and `tone` (confident / measured — rewrites the
hero lead and the first method paragraph).

### 2. Review Hub (`Review Hub.dc.html`)
Staging index for the in-progress versions. Same nav/hero/footer system as the project page, with a
"REVIEW BUILD / JULY 2026 / 4 CUTS · 4 SIMS" badge row and H1 "Versions for review". Two sections —
**Teaser cuts** (4 cards: Cinematic, Bold, Modest, Original) and **Simulators** (4 cards: v4, v1, v2,
v3), newest first, using the card anatomy above. Each section label pairs a colored mono kicker with a
dim mono note. Tweak: `links` (`local` → sibling `.dc.html` files, `deployed` → the flat filenames
published under `/review/`).

### 3. Sim v1 — Swarm Resolution Knob (`Sim v1 - Swarm Core.dc.html`)
Thin wrapper: 14px-padded column, header row (19px Space Grotesk title, 13px `#5c6f8a` spec line,
`← hub` / `project page` mono links, a right-aligned "Knot-a-Surface" chip), then the simulator fills
the rest. Mounts `KnotSwarmSim.onboard()` (or `.create()` when the `onboarding` tweak is off) and shows
a red mono error line if three.js or the core fails to load. Tweaks: `onboarding`, `compact`.

### 4. Sim v2 — Terrain UV Atlas (`Sim v2 - UV Atlas.dc.html`)
Full-viewport canvas on `#07090f` with absolutely positioned chrome (`pointer-events:none` on the
layer, `auto` on controls): top-left FIT readout panel (now carrying the `← HUB` link), top-right READ
hint panel, a 2-column tool rail on the left (SEG± / PAINT / BLUR / RAISE / LOWER / dens± , 44px
columns, 8px radius), the UV atlas panel bottom-right (250px, amber 1.5px border, square canvas,
5 swatches, ISOLATE / CLEAR / RESET), and a bottom dock with the object chips, env and mode pill
groups and the CNET / BUDGET / BRUSH / SEG τ / COVER sliders. Tweaks: `env`, `mode`, `autoOrbit`.

### 5. Sim v3 — Knot Chart (`Sim v3 - Knot Chart.dc.html`)
Radial-gradient stage (`#0d1220 → #04070d → #020409`). Title block top-left ("KNOT CHART", .38em
tracking, with the `⟨p,q⟩` readout and the hub links beneath), a scrolling left column of mono
readouts, a control-point panel top-right, and the chart strip along the bottom. Eight parametric
surfaces, object decomposition, painted density, free-form control points, direct 3D editing,
differential-geometry layers.

### 6. Sim v4 — Surface Prior bench (`Sim v4 - Surface Prior.dc.html`)
Header band (`linear-gradient(90deg,#080d16,#0b1220 52%,#080d16)`, 1px `#16223a` rules): green mono
eyebrow + 27px "SURFACE PRIOR" title, a 12.5px explanatory paragraph, and an amber
"◇ placeholder rates" chip beside the hub links. Body is the side-by-side bench from
`src/sim_v4_core.js`. Accent is green (`#69f0ae`) rather than cyan. **Convergence / chamfer /
photometric curves are placeholders** awaiting measured DTU + Tanks-and-Temples numbers — that chip
must stay until they are real.

### 7. Teaser cuts
Four timeline players, each a canvas stage + caption layer + a bottom control bar (play / replay,
3px scrub track, mono time readout). Playback position persists in `localStorage`.

| File | Duration | Notes |
| --- | --- | --- |
| `Teaser v2 - Bold.dc.html` | 104s (1:44) | 9 acts, claim-forward copy. Embedded on the project page. |
| `Teaser v2 - Modest.dc.html` | 73s (1:13) | 8 chapters, hedged committee tone. |
| `Teaser v4 - Cinematic.dc.html` | 78s (1:18) | Single continuous camera, keyframe table, 16:9 / 1:1 formats. |
| `Teaser.dc.html` | 60s (1:00) | The original baseline cut. |

Every end card closes on three mono pill CTAs → UV Atlas (cyan), Knot 2D Map (purple), Surface Prior
(amber).

## Interactions & behaviour
- **Hover** on cards: border alpha 42% → 90%, background `#0d1220` → `#101a2a`, `translateY(-2px)`,
  transition `.16s`. Links and buttons transition color/border in `.15s`.
- **Throb** keyframe (`0%,100% opacity:1; 50% opacity:.22`, 2s ease-in-out) marks *live* items only:
  the embedded teaser chip, the deployed sim card, the newest cut and the newest sim.
- **Simulators**: drag to orbit, scroll to zoom, paint/segment in the atlas or chart panel; the
  surface follows the edit. Sim v1 opens with an onboarding walkthrough.
- **Teasers**: autoplay from the persisted position, click the track to scrub, replay resets to 0.
- **Responsive**: page grids are `auto-fit`/`minmax` and reflow to one column; the simulators are
  desktop-first full-viewport instruments.

## Design tokens
From `_ds/knot-a-surface-design-system-*/tokens/design_tokens.css`. Two groups — do not cross them.

**Page palette (dark values; the file also carries a light set)**
`--paper #0b1017` · `--card #0f1622` · `--psurface #0d131c` · `--pink #dde5f0` · `--pbody #8fa4c0` ·
`--pmuted #5c6f8a` · `--pborder-c #22314e` · `--pcobalt #4fc3f7` · `--pamber #ffa726` ·
`--pgreen #69f0ae` · `--ppurple #ce93d8` · `--pred #ef5350`

**Instrument palette (always dark, never themed)**
bg `#07090f` / `#04070d` · surfaces `#121b2e`, `#0d1220`, `#060a12`, `#0a0e18`, `#0e1626` ·
borders `#1b2a42`, `#22314e` · text `#dde5f0`, `#cfe0f0` · dim `#8fa4c0`, `#7f93ad`, `#5c6f8a`,
`#4a5b73` · accent `#4fc3f7` · warn `#ffa726` · ok `#69f0ae` · danger `#ef5350` · special `#ce93d8`

**Version accents** v1 green `#69f0ae` · v2 cyan `#4fc3f7` · v3 purple `#ce93d8` · v4 amber `#ffa726`
(the v4 *simulator* uses green for its own chrome).

**Radii** `--r-tool 8px` · `--r-pill 15px` · `--r-hint-pill 999px` · `--r-panel-sm 6px` ·
`--r-panel 10px` · `--r-panel-lg 12px`. Functional circles stay `50%`.

**Type** `--font-ui` Inter · `--font-display` Space Grotesk · `--font-mono` JetBrains Mono.
Page body 13–15.5px; HUD 8.5–11px mono with .08–.3em tracking; card titles 16–17px; H2 1.5–1.6rem;
H1 clamps as noted above.

**Component recipes** — `tokens/design_components.css` documents the canonical shapes as reference
classes (`.ds-btn-tool`, `.ds-btn-pill`, `.ds-panel`, `.ds-hud`, `.ds-badge-instr`, `.ds-btn-page`,
`.ds-badge-page`). Production code consumes the tokens directly rather than these classes.

## Assets
None external. All imagery is procedurally rendered to canvas/WebGL — the capture proxies in the v4
bench (Tanks-and-Temples Barn and Truck, DTU scan110 bust) are procedural stand-ins sampled into a
shared 110k-point surface pool, so any visual difference between models comes from the representation
and not the scene. Fonts come from Google Fonts; three.js from jsDelivr.

## Files in this bundle
```
Project Page.dc.html            Review Hub.dc.html
Sim v1 - Swarm Core.dc.html     Sim v2 - UV Atlas.dc.html
Sim v3 - Knot Chart.dc.html     Sim v4 - Surface Prior.dc.html
Teaser.dc.html                  Teaser v2 - Bold.dc.html
Teaser v2 - Modest.dc.html      Teaser v4 - Cinematic.dc.html
support.js                      runtime the .dc.html files load
src/sim_core.js                 deployed swarm simulator core
src/sim_v4_core.js              v4 reconstruction bench core (local only)
src/sim_style.css               instrument/HUD stylesheet
_ds/.../tokens/*.css            design tokens + component recipes
deploy/deploy-site.sh           publish the main page
deploy/deploy-review.sh         publish the /review/ hub
deploy/README.md                deployment notes
github.md                       repo association, last sync, screen map
```

## Open items
1. **v4 numbers are placeholders.** Convergence, chamfer and photometric curves need measured DTU +
   Tanks-and-Temples runs. Keep the "placeholder rates" chip until then.
2. **`src/sim_v4_core.js` is not upstream.** It only exists locally — commit it when you're ready.
3. **Deploy bundles are stale** with respect to this polish pass: `deploy/site/`, `deploy/review/`,
   `docs/` and `build/` still show three simulators and the old card treatment. Regenerate before the
   next push (or switch to sibling-file loading first, as noted above).
4. **Review-hub links** are local `.dc.html` paths; the deployed build needs the flat filenames
   (`terrain.html`, `teaser-bold.html`, …) — that's what the `links: deployed` tweak produces.
