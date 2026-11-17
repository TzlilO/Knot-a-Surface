---
name: owl-material
description: Add or change a material on the scanned owl of the thesis talk (talk.html, section 06, step "Same owl, new material" — snowy owl, alien jelly, …). Use this whenever the user wants the owl repainted, re-textured or turned into another material or creature (glass, fire, grass, gold, stone, a named character…), wants a look that moves, glows or is see-through, or wants an existing material's colours, opacity or motion tuned — even if they only say "make the owl look like X" or "add another texture edit".
---

# A new material for the owl

The owl of section 06 is a real scan stored as B-spline charts. Its materials are the talk's demonstration that
appearance, opacity and shape all live on the same control points, so a material here is never a texture file or
a shader: it is **a rule that writes control colours (and optionally control opacities and control-point motion)**.
Keep it that way; the slide's claim depends on it.

Read the `manifold` paragraph of `CLAUDE.md` first if you have not this session.

## Where things are

| What | Where |
| --- | --- |
| The materials table `MATS`, the targets `TARGETS` and `owlLife` (regions, wipe, blink, motion) | `talk/scenes.js` |
| The pickers, `st.mat` / `st.target`, `api.setMaterial(name)` / `api.setTarget(name)`, scene `paint` | `talk/sims/manifold.js` |
| The owl's mask (target `owl`), segmented offline; the script that makes it | `talk/dtu/owlmask122.js`, `tools/dtu/owlmask.py` |
| The selection (target `selection`): the Hands atlas in its select mode | `talk/hands.js` |
| Per-Gaussian opacity and the layered draw | `talk/gl.js` (`opts.alpha`, `ktSplat.thin`, `gl.draw`) |
| The slide | `talk.html`, `<article data-scene="paint">` |
| Check scripts | `.claude/skills/owl-material/scripts/` |

The picker fills itself from `MATS`, so a new entry appears in the deck and on the lab page without UI work.

Where a material goes is its TARGET, a second control next to the picker (`object · owl · selection`, default
`object`): every control point of the scanned object (the owl and the books, all that is stone), the owl alone (a
mask segmented offline from the DINOv3 features of the photographs and the net's geometry, with a feathered edge;
without its file, a line by hand), or what the presenter selects on the Hands atlas (wand, brush, eraser, on any
channel, on the atlas or on the object), feathered the same way. The table is a chart of its own and is never
painted. A rule does not know its target: write it for any control point of the object. The talk does not load the
scan's DINOv3 features (`talk/dtu/features122.js`, 648 KB); it fetches them the first time `selection` is picked, and
the atlas offers the DINOv3 channel from then on. The selection needs `talk/hands.js` on the page; without it the
control offers `object · owl` only.

## Writing the rule

Add one entry to `MATS`. It is evaluated once per control point of the object (the owl and the books; never the
table) and cached, so it can be as elaborate as you like; only `move` runs every frame. The target then decides how
much of it shows at each control point.

```js
name: { label: T('shown in the picker'), beak: [r, g, b], lid: [r, g, b], view: 'through', // view: optional
  p: { opacity: 0.34, gloss: 1 },                                                         // its properties, at their defaults
  knobs: [{ k: 'opacity', label: T('opacity'), min: 0.08, max: 1, step: 0.01 }],          // the ones the simulator may turn
  colour: function (s, p) { ...; s.r = ; s.g = ; s.b = ; /* optional */ s.a = ; },
  eye: function (d, lx, ly) { return [r, g, b]; },         // optional
  move: function (s, t, p) { s.x += ; s.y += ; s.z += ; }  // optional
}
```

Labels go through `T` (`KnotTalk.t`): the talk has a Hebrew version, and a label without a line in
`talk/i18n/he.js` under exactly its English shows in English there. Add that line with the material
(`"molten gold": "זהב מותך"`), then `node tools/i18n/check.js` (see `tools/i18n/README.md`).

Every number that is a property of the material (how opaque, how glossy, how stiff, how much of a pattern)
belongs in `p`, read through the last argument, and gets a `knobs` entry: the simulator builds one slider per
entry, and the landing page's teaser turns them too. Name knobs after physical properties, give them ranges in
which the material still reads, and make `move` respond the way the property would (the jelly's `stiffness`
divides the amplitude and multiplies the frequencies by its square root, like a mass on a spring). A knob that
`colour` reads repaints the owl on every change, which is fine; do not read `p` anywhere but in the rule.

What the rule can see in `s`:

- `s.x, s.y, s.z`: the point, in scene units (1 = 80 mm). The books stand on the table at y = 0, the owl from about
  y = 1.5 (feet; its tail hangs to 1.2 at the back) to 3.5 (ear tufts); the cameras are towards +z; the head is above
  y = 2.85 (`s.head`).
- `s.nx, s.ny, s.nz`: unit normal of the net, pointing out. `s.nz` near 1 faces the cameras, near 0 is the rim.
  This is your only handle on lighting: highlights are a power of `n · h` for a half-vector you choose, a rim
  glow is a power of `1 - |s.nz|`.
- `s.sh`: 0..1, how bright the photograph is at that point. It is the carving's relief (grooves between feathers
  are dark). Multiply by something like `0.5 + 0.5 * s.sh` to keep the feathers readable, or threshold it to
  turn the grooves into a pattern (the snowy owl's dark bars).

What it can write:

- `s.r, s.g, s.b`: colour, 0..1, already "lit" (the scene draws photograph colours unlit). The photograph is
  dark (median luminance about 0.12), so a full-white material glares; the snowy owl tops out near 0.9.
- `s.a` (optional): opacity, 1 = solid. Anything below 1 makes the renderer draw that Gaussian in the
  see-through layers: the solid scene shows through it, and so does the owl's own far side. Keep highlights
  solid (`s.a = min(1, base + highlight)`) or they wash out. The sheet is open at the back, so through the
  middle of the body you see the background, not a far wall.
  Opacity only reads when something is behind the owl. From the step's wide camera there is nothing but dark
  background, and a half-opaque material just looks like a darker solid one (the first jelly was sent back for
  exactly this). So give a see-through material `view: 'through'` (a close three-quarter camera with the books
  behind the body; cameras are the `views` of `scan()`), go as low as about 0.35 where you look straight into
  it, and judge it from that camera, not from a close-up you set up by hand.
- `eye(d, lx, ly)`: colour inside an eye socket; `d` = 0 centre .. 1 rim, `lx`, `ly` = -1..1 across it. Eyes
  and beak are always solid. Without `eye` the sockets take the body colour.
- `lid`: the colour that closes over the eyes when the owl blinks. Pick the body colour around the eyes.
- `move(s, t, p)`: displace the point for this frame (t in seconds). It runs for every control point of the target
  every frame, so keep it to a few sines. Tie the amplitude to `s.h`, the height of the point above where the painted
  part stands (the table for `object`, the feet for `owl`, the lowest selected point for `selection`; `s.H` is the
  painted part's height), so that what is painted stays standing. `s.h` and `s.H` exist in `move` only.

Things that go wrong:

- Sign of the normal at the rim: use `Math.abs(s.nz)` for rim terms once a material is see-through, because
  the far side (nz < 0) is then visible too.
- Colours are view-independent. A highlight is painted on and does not follow the camera; say so on the slide
  rather than implying real reflection.
- A material that should glow or flicker over time needs per-frame colour, which `MATS` does not have yet. Add
  it deliberately (an optional `shade(s, t)` next to `move`, recolouring only while the material is selected)
  rather than faking it inside `move`.

## The slide

The step lists the materials in one sentence each and says, in the aside, how each rule works. Add yours to
both. Every statement has to be true of the code: these are rules, not learned, not simulated, not generated,
and the wording must not suggest otherwise. If the material names a character or a product, name it once as a
reference; do not copy artwork.

Do not add a step per material. The user asked for the deck to stay short; the picker is how one step carries
all of them. The time budgets (`data-min40`, `data-min20`) stay as they are.

## Checking it

Software WebGL only draws what it must with these scripts; a full check takes a few minutes.

```bash
S=.claude/skills/owl-material/scripts
# the finished look (the wipe takes 3.5 s = 106 steps), then mid-wipe, then from the side
node $S/shot.js /path/out.png "talk/sims/manifold.html#paint" 130 1280 800 1 "KnotTalk.mounted[0].api.setMaterial('NAME')"
node $S/shot.js /path/mid.png "talk/sims/manifold.html#paint" 60  1280 800 1 "KnotTalk.mounted[0].api.setMaterial('NAME')"
node $S/shot.js /path/side.png "talk/sims/manifold.html#paint" 150 1280 800 1 \
  "var A=KnotTalk.mounted[0].api; A.setMaterial('NAME'); A.orbit.touched=true; A.orbit.goal.th=-0.6; A.orbit.goal.r=4.6; A.orbit.goal.ty=2.5"
# the owl alone (the default target is the whole object)
node $S/shot.js /path/owl.png "talk/sims/manifold.html#paint" 130 1280 800 1 "var A=KnotTalk.mounted[0].api; A.setTarget('owl'); A.setMaterial('NAME')"
# every scene, every material, the whole deck, desktop and phone; ends with ALL CLEAN
node $S/sweep.js
```

Look at the screenshots yourself before showing them: does it read as the material at the size it has on the
slide, on the books as well (target `object`), do the books stay stone with target `owl` and is the line between
owl and books clean from the side and at the feet, are the eyes right.
Write screenshots to the scratchpad, never into the repo.

Nothing here runs on a real GPU. Say so, and say what the material costs: a see-through material draws four
more passes, and any material with `move` re-samples all Gaussians every frame.

## Shipping

One material per pull request, on the session's work branch: commit, push, open a draft PR that says what the
rule does and what was and was not verified. Merging is the user's call each time; wait for their word.
Update the `manifold` paragraph of `CLAUDE.md` only when the mechanism changes (a new hook in `MATS`, a renderer
change), not for each material.
