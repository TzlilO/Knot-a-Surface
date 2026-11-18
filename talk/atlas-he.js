/* Hebrew for design variant B (talk/atlas.js): the twelve claims and the knob's and atlas's words. Loaded by talk.html's head
   only with the design on and the page in Hebrew, after talk/i18n/he.js, whose dictionary it extends (keys = the English
   of talk/atlas.js; tools/i18n/GLOSSARY.md "Rule one": the field's terms in English, digits and symbols isolated). */
(function () {
  var KT = window.KnotTalk = window.KnotTalk || {};
  KT.i18nData = KT.i18nData || {};
  KT.i18nData.he = KT.i18nData.he || {};
  Object.assign(KT.i18nData.he, {
    /* the control net of the talk: its twelve claims */
    'A computer describes a 3D world as a list or as a question.': 'מחשב מתאר עולם בתלת־ממד כרשימה או כשאלה.',
    'Gaussians are a list of soft items: learnable from photographs, fast, with no defined surface.': 'גאוסיאנים הם רשימה של פריטים רכים: אפשר ללמוד אותם מתצלומים, הם מהירים, ואין להם משטח מוגדר.',
    'Free Gaussians fit the photographs and still float off the surface.': 'גאוסיאנים חופשיים מתאימים לתצלומים, ובכל זאת מרחפים מחוץ למשטח.',
    'Invert the question: optimise a smooth surface and derive every Gaussian from it.': 'הופכים את השאלה: עושים אופטימיזציה למשטח חלק, וגוזרים ממנו כל גאוסיאן.',
    'A spline: few control points, local edits, knots where detail is needed.': 'spline: מעט control points, עריכות מקומיות, ו‑knots היכן שצריך פירוט.',
    'Position, orientation and size of a Gaussian follow from the surface and its derivatives.': 'המיקום, הכיוון והגודל של גאוסיאן נובעים מהמשטח ומהנגזרות שלו.',
    'The number of Gaussians is a render-time knob, and a budget can follow the camera.': 'מספר הגאוסיאנים הוא כפתור שמכוונים בזמן ה‑rendering, ותקציב יכול לעקוב אחרי המצלמה.',
    'Parameters live on control points: 16 each, zero per Gaussian.': 'הפרמטרים יושבים על ה‑control points: לכל אחת ⁦16⁩, ולכל גאוסיאן אפס.',
    'Capacity comes from knots, resolution from samples: balance the two.': 'הקיבולת באה מה‑knots, הרזולוציה מהדגימות: צריך לאזן בין השתיים.',
    'A real object: an owl as charts, fitted offline, with mesh, normals, edits and materials read from the control net.': 'אובייקט אמיתי: ינשוף כ‑charts שהותאמו מראש, ועם mesh, נורמלים, עריכות וחומרים שנקראים מה‑control net.',
    'A surface can be touched: contact is a formula on a spline and undefined on a list.': 'במשטח אפשר לגעת: על spline מגע הוא נוסחה, וברשימה הוא אינו מוגדר.',
    'Evidence and limits: close to PGSR on DTU and behind on average; seams, thin structures, image quality and timing are open.': 'ראיות וגבולות: קרוב ל‑PGSR על DTU ומאחוריו בממוצע; ה‑seams, מבנים דקים, איכות התמונה והזמנים עדיין פתוחים.',

    /* the knob */
    're-sample the talk': 'דגימה מחדש של ההרצאה',
    '{min} min · {s} of {t} steps': '⁦{min}⁩ דק׳ · ⁦{s}⁩ מתוך ⁦{t}⁩ שלבים',
    'The floor: only the twelve claims, {m} min in all.': 'הרצפה: רק שתים־עשרה הטענות, ⁦{m}⁩ דק׳ בסך הכול.',
    'Length of the talk in minutes: turning it re-samples the talk': 'אורך ההרצאה בדקות: סיבוב הכפתור דוגם את ההרצאה מחדש',
    'A · the talk as a chart: control points, samples, budget': 'A · ההרצאה כ‑chart: control points, דגימות, תקציב',
    'the talk as a chart · the knob in the corner re-samples the talk': 'ההרצאה כ‑chart · הכפתור בפינה דוגם את ההרצאה מחדש',

    /* the atlas */
    'The talk as a chart': 'ההרצאה כ‑chart',
    'The talk, as a chart': 'ההרצאה, כ‑chart',
    'control points': 'control points',
    'samples': 'דגימות',
    'budget': 'תקציב',
    'control claims': 'טענות הבקרה',
    'deep': 'העמקה',
    'seam': 'seam',
    'More here: move minutes to {part}': 'יותר כאן: להעביר דקות אל {part}',
    'Less here: move minutes away from {part}': 'פחות כאן: להעביר דקות מ‑{part}',
    'reset': 'איפוס',
    'Spread the budget by weight again, and drop the raised steps': 'לפזר שוב את התקציב לפי המשקלות, ולבטל את השלבים שהועלו',
    'raised for this session': 'הועלה לסשן הזה',
    'not sampled at {m} min': 'לא נדגם ב‑⁦{m}⁩ דק׳',
    'Switched off in the slide master': 'כבוי במנהל השקפים',
    'Go to this step': 'מעבר לשלב הזה',
    'Not sampled: a click raises it for this session and goes there': 'לא נדגם: לחיצה מעלה אותו לסשן הזה ועוברת אליו',
    'Close the atlas (Esc)': 'סגירת ה‑atlas (Esc)',
    'Click a tile to go there · A or Esc closes': 'לחצו על אריח כדי לעבור אליו · A או Esc סוגרים',
    'The opening': 'הפתיחה',
    'minutes': 'דקות',
    '{on} of {all} min sampled': 'נדגמו \u2066{on}\u2069 מתוך \u2066{all}\u2069 דק׳'
  });
})();
