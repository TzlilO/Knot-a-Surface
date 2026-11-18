/* One take (talk-onetake.html) — Hebrew. Extends the talk's dictionary (talk/i18n/he.js, loaded first) with this
   page's own strings, keyed by their English exactly as talk/onetake/onetake.js writes them. Terms per
   tools/i18n/GLOSSARY.md: the field's words in English, runs with digits or symbols in <span dir="ltr">, a no-break
   space between a number and the English term it counts. Edited by hand. */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {}; KT.i18nData = KT.i18nData || {};
  var D = KT.i18nData.he = KT.i18nData.he || {};
  var L = function (s) { return '<span dir="ltr">' + s + '</span>'; };
  Object.assign(D, {
    "One take — Knot-a-Surface": "טייק אחד — Knot-a-Surface",
    "The exhibit": "המוצג",

    /* chrome */
    "No. {n} / {total}": "מס׳ " + L("{n} / {total}"),
    "{fwd} or space · next   {back} · back   T · clock   L · language": L("{fwd}") + " או רווח · הבא   " + L("{back}") + " · הקודם   T · שעון   L · שפה",
    "throw": "זריקה",
    "This simulation could not be loaded.": "הסימולציה הזאת לא נטענה.",
    "The scanned owl or its simulation could not be loaded on this page.": "הינשוף הסרוק או הסימולציה שלו לא נטענו בדף הזה.",
    "The finale needs the touch module and the rabbit’s scan, which could not be loaded.": "הסיום צריך את מודול המגע ואת הסריקה של הארנב, והם לא נטענו.",

    /* 1 */
    "This is not a photograph.": "זה לא תצלום.",
    "Nothing here is a pixel that a camera recorded: it is being drawn now, from a description of the owl.": "אין כאן אף פיקסל שמצלמה קלטה: הכול מצויר עכשיו, מתוך תיאור של הינשוף.",
    "DTU scan 122 · surface fitted offline to the dataset’s depth maps · not trained": "סריקה " + L("122") + " של DTU · המשטח הותאם מראש ל‑depth maps של מאגר הנתונים · לא אומן",
    /* 2 */
    "It is drawn from soft blobs.": "הוא מצויר מכתמים רכים.",
    "A Gaussian has a centre, a size, an orientation, an opacity and a colour; blend enough of them and you get the picture.": "לגאוסיאן יש מרכז, גודל, כיוון, אטימות וצבע; ממזגים מספיק מהם ומקבלים את התמונה.",
    "{n} Gaussians on screen · the count is measured": L("{n}") + " גאוסיאנים על המסך · הספירה נמדדה",
    "the count on screen is measured": "הספירה שעל המסך נמדדה",
    /* 3 */
    "A list of blobs does not know where the surface is.": "רשימה של כתמים לא יודעת איפה המשטח.",
    "Free Gaussians match the photographs and still float off the object: along a camera ray, depth is a free choice.": "גאוסיאנים חופשיים מתאימים לתצלומים ובכל זאת מרחפים מחוץ לעצם: לאורך קרן של מצלמה, העומק הוא בחירה חופשית.",
    "2D toy · optimiser running live in this page": "צעצוע 2D · ה‑optimiser רץ בזמן אמת בדף הזה",
    /* 4 */
    "Under these blobs there is a surface.": "מתחת לכתמים האלה יש משטח.",
    "A spline surface: a net of control points, smooth between them, and the only thing that is stored.": "משטח spline: רשת של control points, חלק ביניהן, והדבר היחיד שנשמר.",
    "{c} charts · {n} control points": L("{c}") + " charts ו‑" + L("{n}") + " control points",
    "two charts · 121,216 control points": "שני charts ו‑" + L("121,216") + " control points",
    /* 5 */
    "Ask the surface, get a Gaussian.": "שואלים את המשטח, מקבלים גאוסיאן.",
    "Position is a point of the surface; orientation is its two tangents and their normal; size is the distance to the next sample.": "המיקום הוא נקודה על המשטח; הכיוון הוא שני המשיקים שלו והנורמל שלהם; הגודל הוא המרחק לדגימה הבאה.",
    "computed from derivatives · nothing learned per Gaussian": "מחושב מנגזרות · שום דבר לא נלמד לכל גאוסיאן",
    /* 6 */
    "Move the surface and every blob follows.": "מזיזים את המשטח, וכל כתם הולך אחריו.",
    "The Gaussians are not stored and corrected: they are computed again from the control net, every frame.": "הגאוסיאנים לא נשמרים ומתוקנים: הם מחושבים מחדש מה‑control net, בכל פריים.",
    "the owl’s motion is scripted": "התנועה של הינשוף מתוסרטת",
    /* 7 */
    "How many blobs is a knob, not a property.": "כמה כתמים זה כפתור, לא תכונה.",
    "The same surface can be drawn with a thousand Gaussians or a million, and a fixed budget can be spent where the camera looks.": "את אותו משטח אפשר לצייר באלף גאוסיאנים או במיליון, ותקציב קבוע אפשר להוציא במקום שאליו המצלמה מביטה.",
    "same control points in both · occlusion is not tested": "אותן control points בשני המצבים · הסתרה לא נבדקת",
    /* 8 */
    "Normals are read, not estimated.": "את הנורמלים קוראים, לא מעריכים.",
    "Depth, normal and the chart a blob belongs to come straight from the surface.": "העומק, הנורמל וה‑chart שהכתם שייך אליו באים ישר מהמשטח.",
    "analytic · from S<sub>u</sub> × S<sub>v</sub>": "אנליטי · מתוך " + L("S<sub>u</sub> × S<sub>v</sub>"),
    /* 9 */
    "A scene is an atlas of charts.": "סצנה היא atlas של charts.",
    "Each chart is one spline patch with its own (u, v) square, like a page of a map.": "כל chart הוא spline patch אחד עם ריבוע " + L("(u, v)") + " משלו, כמו דף במפה.",
    "at the seam between charts nothing is guaranteed": "ב‑seam שבין charts שום דבר לא מובטח",
    /* 10 */
    "The mesh was already there.": "ה‑mesh כבר היה שם.",
    "Samples on a lattice know their neighbours, so a sliding 2 × 2 window stitches triangles with no search.": "דגימות על lattice מכירות את השכנות שלהן, ולכן חלון נע של " + L("2 × 2") + " תופר משולשים בלי שום חיפוש.",
    "stitched live at the current budget": "נתפר בזמן אמת בתקציב הנוכחי",
    /* 11 */
    "A material is an edit of the control points.": "חומר הוא עריכה של ה‑control points.",
    "Colour, opacity and shape live on the same points, so glass is a rule written on them.": "צבע, אטימות וצורה יושבים על אותן נקודות, ולכן זכוכית היא כלל שנכתב עליהן.",
    "a rule on the scan’s colours and normals · nothing painted by hand": "כלל על הצבעים והנורמלים של הסריקה · שום דבר לא צויר ביד",
    /* 12 */
    "And a surface can be hit.": "ובמשטח אפשר לפגוע.",
    "On a spline, contact is a closest point and a normal; on a list of blobs there is nothing to hit.": "על spline, מגע הוא נקודה קרובה ביותר ונורמל; ברשימה של כתמים אין במה לפגוע.",
    "contact computed · the headers are scripted · the lists are emulated · the smoke is an invented rule": "המגע מחושב · הנגיחות מתוסרטות · הרשימות מדומות · העשן הוא כלל מומצא",
    /* 13 */
    "Does it still reconstruct real objects?": "האם זה עדיין משחזר אובייקטים אמיתיים?",
    "{a} mm against {b} mm": L("{a} mm") + " מול " + L("{b} mm"),
    "mean Chamfer distance on {n} DTU scans, ours against PGSR (the paper, Table 1)": "מרחק Chamfer ממוצע על " + L("{n}") + " סריקות של DTU, שלנו מול PGSR (המאמר, טבלה " + L("1") + ")",
    "{c} of {n} scans": L("{c}") + " מתוך " + L("{n}") + " סריקות",
    "within 0.05 mm of PGSR; ours is lower on {w}": "בפער של עד " + L("0.05 mm") + " מ‑PGSR; שלנו נמוך יותר ב‑" + L("{w}"),
    "{p} mm from {k} k parameters": L("{p} mm") + " מ‑" + L("{k} k") + " פרמטרים",
    "proxy bench: the densifying baselines reach ≈ {b} mm, more accurate, with {lo}–{hi}× as many; baselines are behavioural emulations": "מבחן תחליפי: שיטות הבסיס שעושות densification מגיעות אל " + L("≈ {b} mm") + ", מדויק יותר, עם " + L("{lo}–{hi}×") + " יותר פרמטרים; שיטות הבסיס הן הדמיות התנהגותיות",
    "figures from the paper’s table and a proxy bench · the owl on this page is a fit, not a result of the method": "המספרים מטבלת המאמר וממבחן תחליפי · הינשוף בדף הזה הוא התאמה, לא תוצאה של השיטה",
    /* 14 */
    "What is open.": "מה עוד פתוח.",
    "Seams between charts, thin structures, foliage and glass are open; image quality, training and rendering time, and ablations are not measured yet.": "פתוחים: ה‑seams בין charts, מבנים דקים, צמחייה וזכוכית; איכות התמונה, זמני האימון וה‑rendering, וה‑ablations עדיין לא נמדדו.",
    "stated limits": "מגבלות מוצהרות",
    /* 15 */
    "The surface is not reconstructed at the end. It is the representation.": "המשטח לא משוחזר בסוף. הוא הייצוג עצמו.",
    "Knot-a-Surface · thesis talk": "Knot-a-Surface · הרצאת התזה"
  });
})();
