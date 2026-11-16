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
| The materials table `MATS` and `owlLife` (regions, wipe, blink, motion) | `talk/scenes.js` |
| The picker, `st.mat`, `api.setMaterial(name)`, scene `paint` | `talk/sims/manifold.js` |
| Per-Gaussian opacity and the layered draw | `talk/gl.js` (`opts.alpha`, `ktSplat.thin`, `gl.draw`) |
| The slide | `talk.html`, `<article data-scene="paint">` |
| Check scripts | `.claude/skills/owl-material/scripts/` |

The picker fills itself from `MATS`, so a new entry appears in the deck and on the lab page without UI work.

## Writing the rule

Add one entry to `MATS`. It is evaluated once per control point of the owl (not the books, not the table) and
cached, so it can be as elaborate as you like; only `move` runs every frame.

```js
name: { label: 'shown in the picker', beak: [r, g, b], lid: [r, g, b],
  colour: function (s) { ...; s.r = ; s.g = ; s.b = ; /* optional */ s.a = ; },
  eye: function (d, lx, ly) { return [r, g, b]; },      // optional
  move: function (s, t) { s.x += ; s.y += ; s.z += ; }  // optional
}
```

What the rule can see in `s`:

- `s.x, s.y, s.z`: the point, in scene units (1 = 80 mm). The owl stands from about y = 1.5 (feet) to 3.5 (ear
  tufts); the cameras are towards +z; the head is above y = 2.85 (`s.head`).
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
- `eye(d, lx, ly)`: colour inside an eye socket; `d` = 0 centre .. 1 rim, `lx`, `ly` = -1..1 across it. Eyes
  and beak are always solid. Without `eye` the sockets take the body colour.
- `lid`: the colour that closes over the eyes when the owl blinks. Pick the body colour around the eyes.
- `move(s, t)`: displace the point for this frame (t in seconds). It runs for every control point of the owl
  every frame, so keep it to a few sines. Tie the amplitude to height above the feet (`s.y - 1.45`) so the feet
  stay on the book.

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
# every scene, every material, the whole deck, desktop and phone; ends with ALL CLEAN
node $S/sweep.js
```

Look at the screenshots yourself before showing them: does it read as the material at the size it has on the
slide, do the books stay stone, is the line between owl and books clean from the side, are the eyes right.
Write screenshots to the scratchpad, never into the repo.

Nothing here runs on a real GPU. Say so, and say what the material costs: a see-through material draws four
more passes, and any material with `move` re-samples all Gaussians every frame.

## Shipping

One material per pull request, on the session's work branch: commit, push, open a draft PR that says what the
rule does and what was and was not verified. Merging is the user's call each time; wait for their word.
Update the `manifold` paragraph of `CLAUDE.md` only when the mechanism changes (a new hook in `MATS`, a renderer
change), not for each material.
