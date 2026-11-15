/* Knot-a-Surface talk — the drawings shared by two modules: "Shalom Rafael" as single-stroke letters
   and as joined-up handwriting, and a missile outline. Everything here is control polygons of cubic
   B-splines in a world box x in [0,2], y in [0,1] (y up). Used by sims/basis.js (letters, script,
   morph, fit) and by the `poster` scene in scenes.js. Depends on nurbs.js + kit.js.
   Exposed as KnotTalk.ink = { texts: { letters(), script() }, missile, morphSets() }. */
(function () {
  'use strict';
  var KT = window.KnotTalk;

  /* ---------- two words, twice: one spline per letter, and joined-up handwriting ----------
     Letter units: baseline 0, x-height 1, ascender 1.7. K = a corner: three control points on one spot. */
  function K(x, y) { return [[x, y], [x, y], [x, y]]; }
  var PRINT = {
    S: [1.0, [0.95, 1.42], [0.78, 1.8], [0.25, 1.82], [0.0, 1.45], [0.1, 1.02], [0.5, 0.86], [0.92, 0.68], [1.02, 0.25], [0.75, -0.12], [0.25, -0.1], [0.0, 0.28]],
    h: [0.95, [0.1, 1.7], [0.1, 0.9], K(0.1, 0), [0.1, 0.6], [0.14, 1.04], [0.5, 1.16], [0.86, 1.02], [0.9, 0.5], [0.9, 0]],
    a: [0.95, [0.88, 0.72], [0.72, 1.12], [0.2, 1.12], [-0.04, 0.5], [0.2, -0.12], [0.72, -0.12], [0.88, 0.35], K(0.88, 1.0), [0.88, 0.4], [0.88, 0]],
    l: [0.35, [0.1, 1.7], [0.1, 1.0], [0.1, 0.3], [0.16, 0.0], [0.36, 0.03]],
    o: [1.0, [0.5, 1.03], [0.1, 1.08], [-0.06, 0.5], [0.1, -0.08], [0.5, -0.08], [0.9, -0.08], [1.06, 0.5], [0.9, 1.08], [0.5, 1.03]],
    m: [1.3, [0.1, 1.0], K(0.1, 0), [0.1, 0.6], [0.15, 1.1], [0.5, 1.14], [0.63, 0.9], [0.65, 0.4], K(0.65, 0), [0.65, 0.6], [0.7, 1.1], [1.05, 1.14], [1.18, 0.9], [1.2, 0.4], [1.2, 0]],
    R: [1.05, [0.1, 0], [0.1, 0.9], K(0.1, 1.7), [0.65, 1.74], [1.08, 1.56], [1.08, 1.02], [0.65, 0.86], K(0.3, 0.86), [0.6, 0.5], [1.0, 0.0]],
    f: [0.8, [0.9, 1.46], [0.72, 1.8], [0.36, 1.78], [0.3, 1.3], [0.3, 0.6], K(0.3, 0), K(0.3, 1.0), K(0.02, 1.0), [0.4, 1.0], [0.74, 1.0]],
    e: [0.95, [0.06, 0.52], [0.5, 0.52], K(0.94, 0.54), [0.9, 0.96], [0.5, 1.14], [0.1, 1.02], [-0.04, 0.5], [0.12, -0.04], [0.5, -0.14], [0.82, -0.02], [0.95, 0.2]]
  };
  /* joined-up forms: each letter starts where the one before left the pen (about (0, 0.1)) */
  var HAND = {
    S: [1.15, [0.9, 1.45], [0.75, 1.8], [0.3, 1.85], [0.05, 1.5], [0.15, 1.05], [0.5, 0.85], [0.9, 0.6], [0.95, 0.2], [0.7, -0.05], [0.25, -0.02], [0.05, 0.25]],
    h: [1.0, [0.3, 0.8], [0.52, 1.6], [0.4, 1.95], [0.2, 1.75], [0.18, 1.0], [0.18, 0], [0.18, 0], [0.24, 0.7], [0.5, 1.08], [0.74, 0.9], [0.76, 0.35], [0.82, 0.02], [1.0, 0.1]],
    /* a and o: the lead-in runs up the left of the bowl and over its top, then the bowl comes back along the same path */
    a: [1.1, [0.04, 0.5], [0.28, 1.08], [0.74, 0.92], [0.74, 0.92], [0.28, 1.08], [0.04, 0.5], [0.2, 0.0], [0.55, 0.0], [0.8, 0.45], [0.82, 1.0], [0.82, 1.0], [0.82, 0.4], [0.9, 0.02], [1.1, 0.1]],
    l: [0.55, [0.25, 0.7], [0.5, 1.6], [0.38, 1.95], [0.2, 1.75], [0.17, 1.0], [0.17, 0.3], [0.3, 0.0], [0.55, 0.1]],
    o: [0.95, [0.02, 0.5], [0.22, 1.06], [0.6, 1.0], [0.6, 1.0], [0.22, 1.06], [0.02, 0.5], [0.15, 0.02], [0.5, -0.04], [0.78, 0.4], [0.7, 0.9], [0.58, 1.0], [0.75, 0.86], [0.95, 0.9]],
    m: [1.4, [0.12, 1.0], [0.17, 0.5], [0.17, 0], [0.17, 0], [0.22, 0.7], [0.42, 1.08], [0.62, 0.9], [0.64, 0.4], [0.64, 0], [0.64, 0], [0.69, 0.7], [0.89, 1.08], [1.09, 0.9], [1.11, 0.4], [1.17, 0.02], [1.4, 0.12]],
    R: [1.3, [0.2, 1.7], [0.2, 0.9], [0.2, 0], [0.2, 0], [0.2, 0.9], [0.22, 1.65], [0.6, 1.78], [1.0, 1.55], [0.95, 1.05], [0.55, 0.9], [0.35, 0.9], [0.6, 0.75], [0.85, 0.2], [1.05, 0.02], [1.25, 0.12]],
    f: [0.6, [0.28, 0.7], [0.5, 1.6], [0.38, 1.95], [0.2, 1.75], [0.17, 1.0], [0.17, 0.0], [0.17, -0.7], [0.3, -0.95], [0.42, -0.6], [0.25, 0.0], [0.2, 0.12], [0.6, 0.1]],
    e: [0.75, [0.3, 0.42], [0.62, 0.82], [0.5, 1.08], [0.2, 0.98], [0.08, 0.5], [0.25, 0.02], [0.55, 0.0], [0.75, 0.1]]
  };
  /* lines = [[stroke, stroke, ...], ...]; a stroke is a run of letters drawn without lifting the pen.
     Returns one control polygon per stroke, in the world box. o = { font, unit, slant, gap, base: [y per line] } */
  function setText(lines, o) {
    var out = [];
    lines.forEach(function (strokes, li) {
      var wd = -o.gap, x = 0;
      strokes.forEach(function (sk) { wd += o.gap; sk.split('').forEach(function (ch) { wd += o.font[ch][0]; }); });
      strokes.forEach(function (sk) {
        var P = [];
        sk.split('').forEach(function (ch, k) {
          var g = o.font[ch], pts = [].concat.apply([], g.slice(1).map(function (q) { return typeof q[0] === 'number' ? [q] : q; }));
          if (!k && o.font === HAND && ch === ch.toLowerCase()) pts = [[0, 0.12]].concat(pts);
          pts.forEach(function (q) { P.push([1 + (x + q[0] + o.slant * q[1] - wd / 2) * o.unit, o.base[li] + q[1] * o.unit]); });
          x += g[0];
        });
        x += o.gap; out.push(P);
      });
    });
    return out;
  }
  var TEXTS = {
    letters: function () { return setText([['S', 'h', 'a', 'l', 'o', 'm'], ['R', 'a', 'f', 'a', 'e', 'l']], { font: PRINT, unit: 0.23, slant: 0, gap: 0.3, base: [0.57, 0.08] }); },
    script:  function () { return setText([['S', 'halom'], ['Rafael']], { font: HAND, unit: 0.19, slant: 0.22, gap: 0.12, base: [0.6, 0.21] }); }
  };

  /* ---------- a third shape, and the morph between all three ----------
     A missile drawn as twelve strokes, in the world box, nose to the right; MISSILE[i] takes the place
     of letter i (S h a l o m / R a f a e l), chosen so the strokes travel short, uncrossed paths. */
  var MISSILE = (function () {
    var flame = [[0.44, 0.545], [0.33, 0.605], [0.22, 0.53], K(0.09, 0.5), [0.22, 0.47], [0.33, 0.395], [0.44, 0.455]];
    var parts = [
      flame,                                                                               // S  exhaust plume
      [[0.74, 0.57], K(0.58, 0.76), K(0.44, 0.76), [0.5, 0.57]],                           // h  tail fin, top
      [[0.5, 0.57], [0.76, 0.57], [1.02, 0.57], [1.28, 0.57]],                             // a  body, top edge
      [[1.08, 0.57], K(0.98, 0.675), K(0.88, 0.675), [0.84, 0.57]],                        // l  wing, top
      [[1.28, 0.57], [1.262, 0.535], [1.262, 0.465], [1.28, 0.43]],                        // o  seeker band
      [[1.28, 0.57], [1.42, 0.57], [1.55, 0.548], K(1.63, 0.5), [1.55, 0.452], [1.42, 0.43], [1.28, 0.43]],   // m  nose cone
      [[0.44, 0.52], [0.35, 0.538], [0.27, 0.5], [0.35, 0.462], [0.44, 0.48]],             // R  inner flame
      [[0.74, 0.43], K(0.58, 0.24), K(0.44, 0.24), [0.5, 0.43]],                           // a  tail fin, bottom
      [[0.5, 0.57], [0.44, 0.553], [0.44, 0.447], [0.5, 0.43]],                            // f  nozzle
      [[0.5, 0.43], [0.76, 0.43], [1.02, 0.43], [1.28, 0.43]],                             // a  body, bottom edge
      [[1.08, 0.43], K(0.98, 0.325), K(0.88, 0.325), [0.84, 0.43]],                        // e  wing, bottom
      [[0.8, 0.57], [0.782, 0.535], [0.782, 0.465], [0.8, 0.43]]                           // l  body band
    ];
    var c = Math.cos(0.2), sn = Math.sin(0.2), k = 1.12;                                   // a little larger, nose up
    return parts.map(function (P) {
      return [].concat.apply([], P.map(function (q) { return typeof q[0] === 'number' ? [q] : q; })).map(function (q) {
        var x = (q[0] - 0.87) * k, y = (q[1] - 0.5) * k; return [1 + x * c - y * sn, 0.5 + x * sn + y * c];
      });
    });
  })();

  /* ---------- making the three drawings compatible, so one can flow into another ---------- */
  var NB = KT.nurbs;
  function cubic(P) { return { P: P, w: P.map(function () { return 1; }), U: NB.clampedKnots(P.length, 3) }; }
  /* cut one cubic into pieces at the parameters `cuts` (a knot inserted three times splits the curve exactly) */
  function split(c, cuts) {
    var out = [], lo = 0;
    cuts.concat([1]).forEach(function (u) {
      if (u >= 1) { out.push(c); return; }
      var n = 3 - c.U.filter(function (t) { return Math.abs(t - u) < 1e-9; }).length, j;
      while (n-- > 0) c = NB.insertKnot(c.P, c.w, 3, c.U, u);
      for (j = 0; Math.abs(c.U[j] - u) > 1e-9; j++);
      out.push({ P: c.P.slice(0, j), w: c.w.slice(0, j), U: c.U.slice(0, j + 3).concat([u]) });
      c = { P: c.P.slice(j - 1), w: c.w.slice(j - 1), U: [u].concat(c.U.slice(j)) };
    });
    return out.map(function (q) {          // each piece on its own [0, 1]
      var a = q.U[0], b = q.U[q.U.length - 1]; q.U = q.U.map(function (t) { return (t - a) / (b - a); }); return q;
    });
  }
  /* the same curve with n control points: knots go into the widest spans until it has them */
  function refine(c, n) {
    while (c.P.length < n) {
      var best = 0, at = 0, i;
      for (i = 3; i < c.P.length; i++) if (c.U[i + 1] - c.U[i] > best) { best = c.U[i + 1] - c.U[i]; at = (c.U[i] + c.U[i + 1]) / 2; }
      c = NB.insertKnot(c.P, c.w, 3, c.U, at);
    }
    return c;
  }
  /* three shapes, twelve cubics each, slot by slot with the same number of control points: so a blend of
     two shapes is a blend of their control points (and knots), and both ends of it are exact */
  function morphSets() {
    var hand = TEXTS.script().map(cubic);
    function cuts(c, counts) {             // where each letter of a stroke ends, as a curve parameter
      var at = 0; return counts.slice(0, -1).map(function (k) { at += k; return (at - 2) / (c.P.length - 3); });
    }
    var sets = [TEXTS.letters().map(cubic),
                [hand[0]].concat(split(hand[1], cuts(hand[1], [14, 14, 8, 13, 16])), split(hand[2], cuts(hand[2], [15, 14, 12, 14, 8, 8]))),
                MISSILE.map(cubic)];
    for (var i = 0; i < 12; i++) {
      var n = Math.max(sets[0][i].P.length, sets[1][i].P.length, sets[2][i].P.length);
      sets.forEach(function (st) { st[i] = refine(st[i], n); });
    }
    return sets;
  }

  KT.ink = { texts: TEXTS, missile: MISSILE, morphSets: morphSets };
})();
