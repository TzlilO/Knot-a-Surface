/* Knot-a-Surface talk — the terrain, one definition for every module that uses it.
   Chapter 05 (talk/sims/refine.js) fits a spline to it and refines that spline where it misses; chapter 03
   (talk/sims/planoid.js) fits its 6 x 6 patch to the same terrain, so the audience meets it there first.
   terrain(x, y), (x, y) in [0, 1]^2: smooth rolling ground plus one crater field. Raw heights (roughly
   -0.05 ... 0.5): each module normalises them itself (refine: makeTarget, planoid: its fit).
   Pure, no dependencies. Exposed as window.KnotTalk.terrain (and module.exports under node, for tests). */
(function (root) {
  'use strict';

  function crater(x, y, cx, cy, R, w, rim, depth) {
    var dx = x - cx, dy = y - cy, r = Math.sqrt(dx * dx + dy * dy), q = (r - R) / w, s = Math.max(0.62 * R, w);
    return rim * Math.exp(-q * q) - depth * Math.exp(-(r * r) / (s * s));
  }
  function ridge(x, y, ax, ay, bx, by, w, amp) {
    var ux = bx - ax, uy = by - ay, L2 = ux * ux + uy * uy;
    var t = ((x - ax) * ux + (y - ay) * uy) / L2; t = t < 0 ? 0 : t > 1 ? 1 : t;
    var dx = x - (ax + t * ux), dy = y - (ay + t * uy);
    return amp * (1 - 0.55 * t) * Math.exp(-(dx * dx + dy * dy) / (w * w));
  }
  function terrain(x, y) {
    // rolling ground: about a third of a wavelength across the map — a 5 x 5 net can hold it
    var h = 0.20 * Math.sin(2.3 * x + 0.5) * Math.cos(1.7 * y - 0.3)
          + 0.13 * Math.cos(1.4 * x - 2.1 * y + 0.9)
          + 0.10 * (y - 0.5) * (x - 0.3);
    // crater field: one rimmed crater, a secondary bowl on its flank, two spur ridges
    h += crater(x, y, 0.655, 0.365, 0.104, 0.050, 0.21, 0.17);
    h += crater(x, y, 0.560, 0.455, 0.046, 0.050, 0.060, 0.080);
    h += ridge(x, y, 0.745, 0.290, 0.815, 0.225, 0.048, 0.095);
    h += ridge(x, y, 0.735, 0.450, 0.780, 0.505, 0.048, 0.070);
    return h;
  }

  if (typeof module !== 'undefined' && module.exports) module.exports = terrain;
  root.KnotTalk = root.KnotTalk || {};
  root.KnotTalk.terrain = terrain;
})(typeof window !== 'undefined' ? window : globalThis);
