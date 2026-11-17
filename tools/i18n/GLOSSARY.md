# Hebrew for the Knot-a-Surface talk: glossary and rules

For whoever maintains `talk/i18n/he.js`. The talk's author is a native Hebrew speaker presenting a thesis to a
technical audience, and wrote a short Hebrew deck, `talk-he.html` (with the Hebrew strings of `talk/he/sims.js`).
**That deck is the reference for voice and terminology.** Where an English sentence of the talk is the counterpart of
a sentence in that deck (the talk's opening is its English version), the dictionary uses the deck's own Hebrew, word
for word. `tools/i18n/README.md` describes how the translation works and the checks.

## Rule one: the field's terms are written in English
The author's instruction (2026-11): the vocabulary of our field is not translated. Inside a Hebrew sentence it is
written in English, in Latin letters, lower case (proper names keep their capitals), spelled exactly as the talk's
English copy spells it (optimiser, regulariser: the spelling of the entry's English key; "optimizer" where the key
says so). The sentence around it stays Hebrew: Hebrew word order, verbs and connectives; the English term is a noun
dropped into it, as a Hebrew-speaking graphics researcher says it aloud.

**Written in English:** spline(s), B-spline, cubic spline, knot(s), knot vector, knot span (short: span, u-span,
v-span), knot insertion, knot removal, control point(s), control net(s), control heights, basis function(s), basis,
local support, manifold, spline manifold, chart(s), spline chart(s), atlas, patch(es), spline patch, lattice, mesh(es),
seam, sheet, planoid, splat(s) / splatting, floater(s), densification / densify, pruning / prune, clone, split, coarse
to fine, refinement, level of detail, tangent frame, loss, optimiser, render(s) / rendering, pipeline(s), depth
map(s), point cloud, held-out view (label: held-out; "the held-out camera" = מצלמת ה‑held-out), regulariser(s),
ablation(s), features (image features, DINOv3 features, control features), pose(s), tessellation, watertight,
non-manifold edge(s), inliers, least squares, robust least squares, conjugate gradients, checkpoint, snapshot(s),
tensor product, convex combination(s), convex hull.

**Stays Hebrew:** the everyday mathematical and descriptive words: משטח, עקומה, נקודה, נגזרת, משיק, נורמל, מעלה,
משקל / ממוצע משוקלל, דגימה, רזולוציה, פרמטר, תקציב, מאזן, שגיאה, התאמה, אימון, אופטימיזציה (the process; the
optimiser, the thing, is English), מצלמה, מבט, קרן, עומק, תצלום, חומר, אטימות, עריכה, רשימה, כתם, גאוסיאן /
גאוסיאנים, ענן (a cloud of Gaussians; "point cloud" is English), משולשים, ווקסלים, גשר / חור (mesh defects), לתפור /
תפירה (to stitch), רשת (a net or a grid alone, e.g. "רשת ה‑5 × 5", "רשת הקפיצים", "רשת ווקסלים"; a control net is
English), מפה only for an ordinary map (the error map of chapter 05, "כמו מפה", "אטלס גאוגרפי"), סריקה, צומת, מעטפת,
רמה, and every wording of "What is never strengthened" below. Already Latin: 3DGS, NeRF, NURBS, COLMAP, SfM, DINOv3,
PSNR, RMS, SH, DTU, TSDF, Poisson, Chamfer, the file formats.

**Ambiguous Hebrew, decided per sentence by the English key:** "קשור" (bound) and "קשר" (connection: "אין קשר בין מקום
ברשימה למקום במרחב") are not "knot"; "רשת" was net, grid or mesh: only "mesh" became English; "מפה" was chart, map or
depth map; "גס / עדין" stay Hebrew as plain description ("רשת גסה", "לדגום עדין יותר"), only the method name is
"coarse to fine"; "split" of a knot span (refine) stays "פיצול"; "prune" of knots in chapter 05 is "prune", as in 3DGS.

**Gender.** A borrowed term takes the masculine (ה‑chart הזה, ה‑mesh רך, ה‑control net שלו, ה‑poses המכוילים,
ה‑snapshots שלה, זה אחר זה), except control point(s) and basis function(s), which keep the feminine of נקודה and
פונקציה (ה‑control points הכתומות, basis functions שונות מאפס). Prefer rephrasing over an agreement that jars.

**Writing the mix so that it renders** (the page is right to left; the bidi algorithm decides the order of every
run):
1. A Hebrew prefix (ה, ב, ל, מ, ש, ו, כ) before an English term takes U+2011, as before a digit: ה‑spline,
   ב‑lattice, מ‑<em>chart אחד</em> (the prefix goes before the tag). Never a Hebrew letter glued to a Latin word.
2. A single English word or a plain multi-word term needs no wrapper.
3. Wrap (`<span dir="ltr">` in HTML, U+2066 … U+2069 in plain strings) a run with digits, ×, a range, =, an arrow,
   brackets, a slash or a formula, as before. "12 × 16" is wrapped; "14 control points" is not (a number then a
   term reads in the right order by itself).
4. **Two English runs must never meet across punctuation alone.** "least squares · conjugate gradients", "charts,
   121,216 control points", "control points – control features", "knot {k} · {where} · +{n} control points": the
   algorithm joins them into ONE left-to-right block, so the parts swap sides, a Hebrew prefix lands at the far end of
   the block and the punctuation moves. Put a Hebrew word between them (שני charts ו‑121,216 control points;
   least squares · עם conjugate gradients; אלה ה‑control features), or, in a label of " · " fields, wrap each field
   that holds English in a right-to-left isolate U+2067 … U+2069 (`חסר control points · ⁧⁦{pts}⁩ control points ×
   ⁦{g}⁩ גאוסיאנים⁩`). The same holds for a number after an English term when the number belongs to the next Hebrew
   words ("{n} splines · ⁦{m}⁩ נקודות"): a digit after a Latin letter joins the Latin run.
5. A value may start with an English term ("mesh, <em>רק תבקשו</em>", "knots <b>לא אחידים</b> …"); a value wholly in
   English (a label such as "control points", "held-out", "→ control points") stays unwrapped and reads left to right.
6. Do not stack three or more English terms with only prefixes between them; split the sentence. Look at every
   changed headline at desktop and phone width.
7. The teleprompter script (`talk/script/he.js`) uses the same English terms where the presenter says them, with no
   wrappers.

## Voice
- Clear, direct, spoken-academic Hebrew: short sentences, no niqqud, no flourish, no slang.
- Translate the meaning, not the words: a sentence must read as if it had been written in Hebrew, and stay as short
  as the English. This is slide copy.
- The audience is addressed in the plural imperative: גררו, הזיזו, לחצו, בחרו, שנו, כבו, הפעילו. Narration of what
  happens on screen is in the plural present: מוסיפים knot, מזיזים control point, מאטים את הסנטימטרים האחרונים.
- Labels of controls are nouns, not commands: הוספה, הסרה, איפוס, השהיה, ייצוא, אחיזה. A button that the copy quotes
  keeps the same words in both places ("drop again" = הפילו שוב, quoted as “הפילו שוב”).
- "we / our / ours" = אנחנו / שלנו ("ours" as a label = שלנו; "our method" = השיטה שלנו).
- Every number, unit and symbol of the English appears in the Hebrew, unchanged.

## What is never strengthened
The talk says exactly what was and was not done. These wordings are fixed:

| English | Hebrew |
|---|---|
| fitted offline (the owl, its charts) | הותאם מראש / הותאמו מראש / התאמה שנעשתה מראש |
| not trained, was not trained | לא אומן, לא אומנה, לא אומנו |
| emulated (the list of Gaussians, the baselines) | מדומה, מדומות; "behavioural emulations" = הדמיות התנהגותיות |
| a geometric fit, not training | התאמה גאומטרית, לא אימון |
| the run is pending | ממתין להרצה; "has not been made yet" = עדיין לא בוצעה |
| untrained (the 3DGS initialisation) | לא מאומן |
| scripted | מתוסרט |
| measured | נמדד |
| stand-in (for the run that is pending) | ממלא מקום |
| procedural stand-in, proxy (a scene, the bench) | תחליף פרוצדורלי; proxy bench = מבחן תחליפי |
| pretrained model (DINOv3) | מודל שאומן מראש |
| not measured yet; measurements are pending | עדיין לא נמדדו; עדיין חסרות |

Never write that the owl, the list or anything else was trained (אומן) unless the English says so. "Offline" is
always מראש, never "לא מקוון".

## Terms
| English | Hebrew |
|---|---|
| Gaussian, Gaussians | גאוסיאן, גאוסיאנים |
| Gaussian Splatting, 3DGS, NeRF, NURBS, B-spline, T-splines, COLMAP, SfM, DINOv3, PSNR, RMS, RMSE, SH, DTU, TSDF, Poisson, Chamfer, PLY / GLB / OBJ / STL | stay Latin |
| blob (soft blob) | כתם (כתם רך) |
| splat (one drawn Gaussian) | splat, splats (where the English says splat; "Gaussian" stays גאוסיאן); "Gaussian Splatting" stays Latin |
| surface | משטח |
| curve | עקומה |
| spline, splines, cubic spline, B-spline | **English**: spline, splines, cubic spline, cubic B-spline |
| control point(s), control net | **English**: control point, control points (feminine), control net |
| control heights | **English**: control heights |
| knot, knots, knot vector | **English**: knot, knots, knot vector |
| knot span | **English**: knot span; "span" alone = span; u-span, v-span |
| knot insertion / removal; interior knots; redundant knot | **English**: knot insertion ("Boehm’s" = knot insertion של Boehm), knot removal; knots פנימיים; knot מיותר. The verb stays Hebrew: מוסיפים knot, הוספת knot |
| degree (of the polynomial) | מעלה |
| basis function(s), basis, basis weights | **English**: basis function, basis functions (feminine), basis; משקלי ה‑basis |
| weight(s), rational weights, weighted average | משקל, משקלים, משקלים רציונליים, ממוצע משוקלל |
| convex combination, tensor product, convex hull; conics | **English**: convex combinations, tensor product, convex hull; חתכי חרוט |
| local support | **English**: local support |
| smooth, continuous, continuity | חלק, רציף, רציפות |
| patch (surface patch) | **English**: patch, patches, spline patch |
| chart, atlas of charts | **English**: chart, charts, spline chart, atlas של charts; "the (u, v) chart" = ה‑chart של (u, v). An ordinary map stays מפה |
| manifold, spline manifold | **English**: manifold ("a manifold of splines" = manifold של splines), spline manifold |
| sheet (one chart wrapped around an object) | **English**: sheet |
| seam | **English**: seam |
| lattice | **English**: lattice ("even lattice" = lattice אחיד, "warped lattice" = lattice מעוות) |
| grid | רשת; the (u, v) sample lattice is "lattice" |
| cell, cell side | תא, צלע התא |
| mesh, triangles | **English**: mesh, meshes (masculine); משולשים |
| stitch (a mesh) | לתפור, תפירה |
| tessellation, triangulation, connectivity | **English**: tessellation; טריאנגולציה, קישוריות |
| watertight | **English**: watertight; watertight stitching = תפירה watertight |
| edge (of a mesh), non-manifold edge | צלע; **English**: non-manifold edge(s) |
| bridge, hole (mesh defects) | גשר, חור |
| sample, samples, sampling | דגימה, דגימות, דגימה |
| adaptive; adaptive sampling; view-adaptive sampling | אדפטיבי; דגימה אדפטיבית; דגימה אדפטיבית למבט |
| even, uniform (lattice, net, knots) | אחיד |
| warped lattice | lattice מעוות |
| resolution | רזולוציה |
| budget, fixed budget | תקציב, תקציב קבוע |
| cap (the most control points) | תקרה |
| derived (Gaussians derived from the surface) | נגזר, נגזרים |
| derivative, partial derivative, closed form, finite differences | נגזרת, נגזרת חלקית, ביטוי סגור, הפרשים סופיים |
| tangent, tangent plane, tangent vectors, tangent frame | משיק, מישור משיק, וקטורים משיקים; **English**: tangent frame |
| normal | נורמל |
| cross product, unit vector | מכפלה וקטורית, וקטור יחידה |
| position / rotation / scale (of a Gaussian) | מיקום / סיבוב / גודל |
| orientation, tilt | כיוון, נטייה |
| shear | עיוות הגזירה |
| chord | מיתר |
| outline (1σ), footprints | קו מתאר, טביעות רגל |
| uncovered; piled three deep | לא מכוסה; בשלוש שכבות |
| planoid | **English**: planoid |
| parameter(s), learnable parameters | פרמטר, פרמטרים, פרמטרים נלמדים |
| ledger | מאזן |
| by construction | מעצם הבנייה |
| lightweight | קל משקל |
| capacity | קיבולת |
| degrees of freedom | דרגות חופש |
| free / bound (Gaussians, model) | חופשי / קשור (the pair as in physics; where "קשור" stands alone, write "קשור למשטח") |
| optimise, optimisation, optimiser | אופטימיזציה (לבצע אופטימיזציה ל..., עובר אופטימיזציה); **English**: optimiser ("optimizer" where the key spells it so) |
| fit, to fit | התאמה, להתאים |
| least squares, robust least squares, conjugate gradients | **English**: least squares (ב‑least squares), robust least squares, conjugate gradients |
| inliers (points the fit kept) | **English**: inliers |
| error, gap, largest miss | שגיאה, פער, הפער הגדול ביותר |
| drawn error | שגיאת הציור |
| "of the relief" (an error as a share of the terrain's height range) | מטווח הגבהים |
| tolerance | סבילות |
| floaters | **English**: floater, floaters |
| training, trained, to train, retrain | אימון, מאומן, לאמן, אימון מחדש |
| training view; held-out view | מבט אימון; **English**: held-out view (label: held-out; the camera: מצלמת ה‑held-out) |
| converge, convergence | להתכנס, התכנסות |
| loss | **English**: loss (masculine); training loss = ה‑loss של האימון |
| iteration, run, snapshot, checkpoint | איטרציה, הרצה; **English**: snapshot(s), checkpoint |
| replay | הילוך חוזר |
| prune, densify, clone, split | **English**: pruning / prune, densification, clone, split ("עושים prune", "מבצע split ו‑pruning") |
| regularisers, heuristics, ablation | **English**: regulariser(s), ablations; היוריסטיקות |
| variance, bias | שונות, הטיה |
| draw / render | לצייר (ציור) for "draw"; **English**: render(s), rendering ("החלטה של rendering") |
| camera, view (camera view) | מצלמה, מבט |
| pose, pinhole, focal length, field of view | **English**: pose, poses; מודל מצלמת הנקב, אורך מוקד, שדה ראייה |
| frustum (drawn as a pyramid) | פירמידה |
| photograph(s) | תצלום, תצלומים |
| dataset | מערך הנתונים |
| feature (a matched image feature); features (DINOv3) | **English**: feature, features, DINOv3 features |
| control features | **English**: control features |
| triangulated; reprojection error; track length; observations | עבר טריאנגולציה; שגיאת הטלה חוזרת; אורך מסלול; תצפיות |
| ray | קרן; along / across the ray = לאורך הקרן / בניצב לקרן |
| depth, depth map | עומק; **English**: depth map(s) |
| point cloud, sparse, dense | **English**: point cloud (point cloud דליל); דליל, צפוף |
| initialisation, reference implementation | אתחול, מימוש הייחוס |
| pipeline | **English**: pipeline(s) |
| level (of the coarse-to-fine fit), layout, the shipped net | רמה, פריסה, הרשת הסופית שבהרצאה (label: סופית) |
| envelope, node, relaxation sweeps, arc-length passes | מעטפת, צומת, סבבי הרפיה, מעברי אורך קשת |
| level of detail | **English**: level of detail |
| coarse, fine, refine, refinement; coarse to fine | גס, עדין as plain description; **English**: coarse to fine (the method), refinement ("עבר refinement") |
| edit, editable | עריכה, ניתן לעריכה |
| material, opacity, see-through | חומר, אטימות, שקוף למחצה |
| gloss, refractive index, albedo, barring | ברק, מקדם שבירה, אלבדו, פסים |
| highlights (sharp highlights) | החזרי אור (חדים) |
| spherical harmonics, view-dependent colour | הרמוניות ספריות, צבע תלוי־מבט |
| segmentation, selection, mask | סגמנטציה, בחירה, מסכה |
| wand / brush / erase | שרביט / מברשת / מחק |
| grab / pinch / pull / push / hold | אחיזה / צביטה / משיכה / דחיפה / החזקה |
| tint / hide / inflate / deflate; select none; let go | צביעה / הסתרה / ניפוח / כיווץ; ביטול הבחירה; שחרור |
| reach, strength (of a brush) | טווח, עוצמה |
| list (unstructured list of Gaussians) | רשימה |
| unstructured; unordered | לא מובנה (רשימה לא מובנית); לא מסודרת, בלי סדר |
| structured | מובנה |
| neighbour(s), nearest neighbours, adjacency | שכן, שכנים, השכנים הקרובים, שכנות |
| search | חיפוש |
| address, index, spatial index | כתובת, אינדקס, אינדקס מרחבי |
| cluster (verb, noun), label | לקבץ לאשכולות, אשכול, תווית |
| lift 2D masks | להרים מסכות דו־ממדיות לתלת־ממד |
| TSDF fusion, Poisson reconstruction | מיזוג TSDF, שחזור Poisson |
| voxel (the mesh board) | ווקסל, ווקסלים (the opening keeps the author's קוביות for "voxels") |
| job (a computation stepped a few ms per frame) | משימה |
| to hash (blocks of voxels) | לגבב |
| level set, marching cubes, k-d tree | משטח רמה, marching cubes, עץ k-d |
| mean, median, 95th percentile | ממוצע, חציון, אחוזון 95 |
| contact, moment of contact, collision | מגע, רגע המגע, התנגשות |
| ball, bowl, spring, stiffness | כדור, קערה, קפיץ, קשיחות |
| restitution (coefficient of) | מקדם התקומה |
| impulse | מתקף |
| toughness | חוסן |
| damped (springs), damping | מרוסן, ריסון |
| bounce; point mass; rigid body | ניתור; מסה נקודתית; גוף קשיח |
| loose particle; knocked loose | חלקיק חופשי; הועפו ממקומם |
| shell (3σ shell); summed density; threshold | קליפה; צפיפות מסוכמת; סף |
| root (of an equation), residual | שורש, שארית |
| rigid, elastic, brittle | קשיח, אלסטי, שביר |
| slow motion, timeline | הילוך איטי, ציר הזמן |
| terrain, crater, ridge, height map, height field | תבליט, מכתש, רכס, מפת גבהים, שדה גבהים |
| frame, frame time, pixel, screen | פריים, זמן פריים, פיקסל, מסך |
| buffer | באפר |
| real time | זמן אמת |
| live (computed as you watch) | בזמן אמת; a live figure = איור חי |
| GPU / graphics card | כרטיס גרפי (GPU) |
| owl, rabbit, books, table, figurine | ינשוף, ארנב, ספרים, שולחן, פסלון |
| stone, clear glass, alien jelly, snowy owl | אבן, זכוכית שקופה, ג'לי חייזרי, ינשוף שלג |
| barn, truck, bust, garden, studio, still life, breakfast | אסם, משאית, פרוטומה, גינה, סטודיו, טבע דומם, ארוחת בוקר |
| scan (DTU scan 122) | סריקה (סריקה 122 של DTU) |
| evidence, benchmark, baseline, accuracy | ראיות, מבחן השוואה, שיטת בסיס, דיוק |
| leaderboard, scope, future work, takeaway | טבלת דירוג, היקף, עבודה עתידית, מסקנה |
| state of the art | מהמתקדמות ביותר |
| interoperable | משתלב בכלים קיימים |
| toy (the 2-D example), flatland | דוגמת צעצוע, עולם שטוח |
| strip (of pixels); band (rolling bands of readings); reading | רצועה; רצועה; קריאה |
| tile (of the atlas); to tile (the surface) | אריח; לרצף |
| inset; face-on; edge-on | חלונית; במבט חזיתי; נראה כקו |
| handwriting, stroke, morphing, track | כתב יד, משיכה, מורפינג, מסלול |
| zoom, pan, freeze | הגדלה, הזזת התצוגה, הקפאה |
| explicit / implicit representation | ייצוג מפורש / ייצוג סתום |
| Intuition / Mechanism / Math (level badges) | אינטואיציה / מנגנון / מתמטיקה |
| "Surfaces first. Gaussians second." | קודם משטח, ורק אחר כך גאוסיאנים. |
| Knot-a-Surface | stays Latin |

### The deck's own furniture
| English | Hebrew |
|---|---|
| chapter, step, part | פרק, שלב, חלק |
| cut (long / short cut) | גרסה (ארוכה / קצרה) |
| deep-dive step | שלב העמקה |
| the opening, the cover | הפתיחה, השער |
| deck | מצגת |
| slide master | מנהל השקפים |
| presenter view | תצוגת מציג |
| pace clock; plan (the planned minutes) | שעון הקצב; תכנון |
| controls (the kit's gear label) | בקרות |
| slider | מחוון ("take / move the … slider" = הזיזו את מחוון ה…) |
| knob, dial | כפתור |
| preset | מצב מוכן |
| on / off | a part of the talk: דולק / כבוי; view-adaptive sampling: פועלת / כבויה |
| focus (the simulation) | מיקוד |
| mute | השתקה |
| reset; pause / resume; replay; export; download | איפוס; השהיה / המשך; הילוך חוזר; ייצוא; להוריד |
| language | שפה |

## Decisions (keep them when adding sentences)
- **Opacity.** The opening follows the author's deck: שקיפות in the list of a Gaussian's attributes ("מרכז, גודל,
  כיוון, שקיפות וצבע"). From chapter 01 on the quantity is אטימות, and "see-through" is שקוף למחצה.
- **Bound / free** = קשור / חופשי in every part, the pair physics uses (אלקטרון קשור / חופשי). On screen the two
  always appear together. Where "bound" would stand alone and could be read as "related", the sentence says
  "קשור למשטח" ("גאוסיאן שקשור למשטח לא יכול לעזוב אותו").
- **×k** is written "פי k", as the author's deck does ("פי שניים פירוט — פי שמונה זיכרון"). A range the paper quotes as a
  figure stays a left-to-right run: `<span dir="ltr">10–50×</span>`.
- **Time labels**: `≈ 6 דקות` with no wrapper; with a fraction `≈ <span dir="ltr">2½</span> דקות`. A wrapper around
  "≈ 5" before a Hebrew word would put the sign on the wrong side.
  The same holds for any "≈ number" before a Hebrew noun: the sign stays outside the left-to-right run
  (`\u2067≈ \u2066{n}\u2069 גאוסיאנים\u2069`). "≈ 58 mm", where the unit is Latin too, is one run.
- **Keyboard help.** A key cap shows what is printed on the key: letters, `esc` and `Home` stay Latin; the space bar
  is `<kbd>רווח</kbd>`, as the author's deck says ("הקישו רווח"). The arrows are swapped in the values (← is "next"
  in Hebrew). `talk.html?master` is wrapped left to right and followed by " = ".
- **Presenter view** = תצוגת מציג everywhere (the window title, the button, the message that quotes the button).
- **Arrows.** Between Hebrew words an arrow points the way Hebrew reads: אותיות ← כתב יד ← טיל. Between two numbers
  it stays inside the left-to-right run: `6.3 → 6.0%`. A link or button arrow is mirrored (↗ becomes ↖, "next →"
  becomes "הבא ←"). An axis title of a figure is written arrow-first ("→ control points"), so the arrow is drawn at the
  right end, where the axis grows.
- **Pairs beside a left-to-right pair of values.** The panel labels "one · own" and "in · out" are written with
  their two words swapped ("לפי התא · אחד", "בחוץ · בפנים"): the label reads right to left and the two numbers
  beside it stay left to right, so each word ends up on the same side as its number. Keep the swap.
- **Left and right.** A stage is not mirrored, so "bottom left", "on the right" inside a figure stay as they are
  (למטה משמאל, מימין). Where the English points at the stage from the text ("the optimiser on the right"), the
  Hebrew does not name a side (ה‑optimiser שלפניכם): the page is mirrored and the stage is on the other side.
- **"Can't touch this"** (chapter 09) stays in English: it is the song's title.
- **Names of people** in method names stay Latin (Boehm, Cox–de Boor, Schlick, Fresnel, Gauss–Newton, Poisson).

## Typography and direction (the page is right to left)
- Western digits, decimal point, thousands comma: exactly the English numbers. Percent sign attached: `5.7%`.
- **After a Hebrew prefix before a digit, a Latin word, a `{placeholder}` or a wrapped run: U+2011, the non-breaking
  hyphen** (ל‑72, ב‑3DGS, מ‑{n}, ב‑`<span dir="ltr">…</span>`), as the author's deck writes it. Never a maqaf there.
- A maqaf stays inside Hebrew compounds: תלת־ממד, דו־ממדי, חד־ממדי, לא־יריעתית, תלוי־מבט.
- Latin names and abbreviations stay Latin. Units stay as symbols (mm, ms, dB, px, cm, m/s, N·s).
- **Left-to-right runs.** Anything that must be read left to right as one piece gets a wrapper, or the bidi algorithm
  reorders it: a number with a unit (`1.36 mm`), a product or size (`12 × 16`), an arrow between numbers
  (`6.3 → 6.0%`), a range (`4–8%`), `±`, a formula written as text (`n = S_u × S_v`, `(u, v)`), a file name.
  - In copy that is HTML, and in a script's string that the script sets as HTML: `<span dir="ltr">12 × 16</span>`.
  - In plain strings (canvas text, `textContent`, attributes) and in the call-to-action lines (`span.do` in
    `tools/i18n/extract.js`'s listing: the line is a flex row, and a span inside it becomes a separate item with a
    gap): U+2066 … U+2069, written `⁦` and `⁩`.
  - A lone number or a lone Latin word inside Hebrew needs no wrapper.
- `KnotTalk.t` wraps what it returns in an isolate of its own (U+2067 when the text has Hebrew letters), unless the
  value already starts with a direction mark. A value that begins with a left-to-right run and goes on in Hebrew
  therefore starts with `⁧` itself, and ends with `⁩`.
- A script's string may contain markup only where the script sets it as HTML (`innerHTML`, the kit's `ui.el`,
  labels of `ui.slider / seg / toggle / button / readout`, captions). A readout's value and a hint are plain text.
  `node tools/i18n/check.js` lists the candidates under "(b) markup added to a string of a script"; look at the
  call before adding a span.
- **Line breaks.** A formula placeholder followed by a comma, a full stop or a colon gets a word joiner between them
  (`<span data-tex="…"></span>\u2060,`): without it the browser may start the next line with the sign. In a
  headline, `&nbsp;` keeps the words of one phrase on one line ("ה‑control&nbsp;points", a sentence of two words); after
  changing a headline, look at how it breaks at desktop and at phone width.
- Punctuation: Hebrew uses the same `.` `,` `:` `;` `?`; the English quotation marks “ ” stay; an em dash becomes a
  comma, a colon or ` – ` as reads best (the one em dash in the dictionary is the author's own sentence).

## Markup (HTML keys)
- A key is the block's inner HTML and is never changed here: the page looks the sentence up by it.
- Keep every inline element of the English, with the same tag and the same attributes, around the corresponding
  Hebrew words: `<b>`, `<em class="c-surf">`, `<b class="c-err">`, `<kbd>`, `<sub>`, `<a href=…>`, `<i><b></b></i>`.
  Their order may change with the sentence.
- `<span data-tex="…"></span>` is a formula placeholder: copy it verbatim.
- Entities stay as they are in the key (`&amp;`, `&nbsp;`, `&gt;`).
- The only elements a value may add are the `<span dir="ltr">…</span>` wrappers above.
