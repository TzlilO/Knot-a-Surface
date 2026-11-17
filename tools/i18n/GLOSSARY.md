# Hebrew for the Knot-a-Surface talk: glossary and rules

For whoever maintains `talk/i18n/he.js`. The talk's author is a native Hebrew speaker presenting a thesis to a
technical audience, and wrote a short Hebrew deck, `talk-he.html` (with the Hebrew strings of `talk/he/sims.js`).
**That deck is the reference for voice and terminology.** Where an English sentence of the talk is the counterpart of
a sentence in that deck (the talk's opening is its English version), the dictionary uses the deck's own Hebrew, word
for word. `tools/i18n/README.md` describes how the translation works and the checks.

## Voice
- Clear, direct, spoken-academic Hebrew: short sentences, no niqqud, no flourish, no slang.
- Translate the meaning, not the words: a sentence must read as if it had been written in Hebrew, and stay as short
  as the English. This is slide copy.
- The audience is addressed in the plural imperative: גררו, הזיזו, לחצו, בחרו, שנו, כבו, הפעילו. Narration of what
  happens on screen is in the plural present: מוסיפים קשר, מזיזים נקודת בקרה, מאטים את הסנטימטרים האחרונים.
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
| splat (one drawn Gaussian) | גאוסיאן; "splatting" alone = Splatting |
| surface | משטח |
| curve | עקומה |
| spline, splines | ספליין, ספליינים |
| control point(s), control net | נקודת בקרה, נקודות בקרה, רשת בקרה |
| control heights | גובהי בקרה |
| knot, knots, knot vector | קשר, קשרים, וקטור הקשרים |
| knot span | מקטע (בין שני קשרים) |
| knot insertion / removal; interior knots; redundant knot | הוספת קשר / הסרת קשר; קשרים פנימיים; קשר מיותר |
| degree (of the polynomial) | מעלה |
| basis function(s) | פונקציית בסיס, פונקציות בסיס |
| weight(s), rational weights, weighted average | משקל, משקלים, משקלים רציונליים, ממוצע משוקלל |
| convex combination, tensor product, conics | צירוף קמור, מכפלה טנזורית, חתכי חרוט |
| local support | תמך מקומי |
| smooth, continuous, continuity | חלק, רציף, רציפות |
| patch (surface patch) | טלאי |
| chart, atlas of charts | מפה, אטלס של מפות |
| manifold, spline manifold | יריעה, יריעת ספליינים |
| sheet (one chart wrapped around an object) | גיליון |
| seam | תפר |
| lattice | סריג |
| grid | רשת (סריג when it is the (u, v) sample lattice) |
| cell, cell side | תא, צלע התא |
| mesh, triangles | רשת משולשים, משולשים |
| stitch (a mesh) | לתפור, תפירה |
| tessellation, triangulation, connectivity | טסלציה, טריאנגולציה, קישוריות |
| watertight | סגור הרמטית; watertight stitching = תפירה הרמטית |
| edge (of a mesh), non-manifold edge | צלע, צלע לא־יריעתית |
| bridge, hole (mesh defects) | גשר, חור |
| sample, samples, sampling | דגימה, דגימות, דגימה |
| adaptive; adaptive sampling; view-adaptive sampling | אדפטיבי; דגימה אדפטיבית; דגימה אדפטיבית למבט |
| even, uniform (lattice, net, knots) | אחיד |
| warped lattice | סריג מעוות |
| resolution | רזולוציה |
| budget, fixed budget | תקציב, תקציב קבוע |
| cap (the most control points) | תקרה |
| derived (Gaussians derived from the surface) | נגזר, נגזרים |
| derivative, partial derivative, closed form, finite differences | נגזרת, נגזרת חלקית, ביטוי סגור, הפרשים סופיים |
| tangent, tangent plane, tangent vectors, tangent frame | משיק, מישור משיק, וקטורים משיקים, מערכת הצירים המשיקה |
| normal | נורמל |
| cross product, unit vector | מכפלה וקטורית, וקטור יחידה |
| position / rotation / scale (of a Gaussian) | מיקום / סיבוב / גודל |
| orientation, tilt | כיוון, נטייה |
| shear | עיוות הגזירה |
| chord | מיתר |
| outline (1σ), footprints | קו מתאר, טביעות רגל |
| uncovered; piled three deep | לא מכוסה; בשלוש שכבות |
| planoid | פלנואיד |
| parameter(s), learnable parameters | פרמטר, פרמטרים, פרמטרים נלמדים |
| ledger | מאזן |
| by construction | מעצם הבנייה |
| lightweight | קל משקל |
| capacity | קיבולת |
| degrees of freedom | דרגות חופש |
| free / bound (Gaussians, model) | חופשי / קשור (the pair as in physics; where "קשור" stands alone, write "קשור למשטח") |
| optimise, optimisation, optimiser | אופטימיזציה (לבצע אופטימיזציה ל..., עובר אופטימיזציה), אופטימייזר |
| fit, to fit | התאמה, להתאים |
| least squares, robust least squares, conjugate gradients | ריבועים פחותים, ריבועים פחותים חסינים, גרדיאנטים צמודים |
| inliers (points the fit kept) | (נקודות ש)נשארו בהתאמה |
| error, gap, largest miss | שגיאה, פער, הפער הגדול ביותר |
| drawn error | שגיאת הציור |
| "of the relief" (an error as a share of the terrain's height range) | מטווח הגבהים |
| tolerance | סבילות |
| floaters | מרחפים (גאוסיאנים מרחפים) |
| training, trained, to train, retrain | אימון, מאומן, לאמן, אימון מחדש |
| training view; held-out view | מבט אימון; מבט שלא שימש באימון (short label: לא באימון) |
| converge, convergence | להתכנס, התכנסות |
| loss | הפסד (פונקציית ההפסד) |
| iteration, run, snapshot, checkpoint | איטרציה, הרצה, תמונת מצב, נקודת ביקורת |
| replay | הילוך חוזר |
| prune, densify, clone, split | גיזום (לגזום), ציפוף, שכפול, פיצול |
| regularisers, heuristics, ablation | איברי רגולריזציה, היוריסטיקות, ניסויי אבלציה |
| variance, bias | שונות, הטיה |
| draw / render | לצייר (ציור); "rendering" as a noun = רינדור |
| camera, view (camera view) | מצלמה, מבט |
| pose, pinhole, focal length, field of view | תנוחה, מודל מצלמת הנקב, אורך מוקד, שדה ראייה |
| frustum (drawn as a pyramid) | פירמידה |
| photograph(s) | תצלום, תצלומים |
| dataset | מערך הנתונים |
| feature (a matched image feature); features (DINOv3) | מאפיין; מאפיינים (מאפייני DINOv3) |
| control features | מאפייני בקרה |
| triangulated; reprojection error; track length; observations | עבר טריאנגולציה; שגיאת הטלה חוזרת; אורך מסלול; תצפיות |
| ray | קרן; along / across the ray = לאורך הקרן / בניצב לקרן |
| depth, depth map | עומק, מפת עומק |
| point cloud, sparse, dense | ענן נקודות, דליל, צפוף |
| initialisation, reference implementation | אתחול, מימוש הייחוס |
| pipeline | צינור עיבוד |
| level (of the coarse-to-fine fit), layout, the shipped net | רמה, פריסה, הרשת הסופית שבהרצאה (label: סופית) |
| envelope, node, relaxation sweeps, arc-length passes | מעטפת, צומת, סבבי הרפיה, מעברי אורך קשת |
| level of detail | רמת פירוט |
| coarse, fine, refine, refinement | גס, עדין, לעדן, עידון |
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
  becomes "הבא ←"). An axis title of a figure is written arrow-first ("→ נקודות בקרה"), so the arrow is drawn at the
  right end, where the axis grows.
- **Pairs beside a left-to-right pair of values.** The panel labels "one · own" and "in · out" are written with
  their two words swapped ("לפי התא · אחד", "בחוץ · בפנים"): the label reads right to left and the two numbers
  beside it stay left to right, so each word ends up on the same side as its number. Keep the swap.
- **Left and right.** A stage is not mirrored, so "bottom left", "on the right" inside a figure stay as they are
  (למטה משמאל, מימין). Where the English points at the stage from the text ("the optimiser on the right"), the
  Hebrew does not name a side (האופטימייזר שלפניכם): the page is mirrored and the stage is on the other side.
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
  headline, `&nbsp;` keeps the words of one phrase on one line ("נקודות&nbsp;הבקרה", a sentence of two words); after
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
