/* Knot-a-Surface lab — object library.
   Three product-photography still lifes, each nothing but B-spline charts: geometry in the control
   points, PURE ALBEDO in the control features (one sRGB colour per control point; no baked light of
   any kind: the lab renderer adds shadows, ambient occlusion and reflections itself).
     tea   — the Utah teapot rebuilt from Newell's Bezier patches, with a cup and saucer on a cloth
     knot  — a (2,3) torus-knot tube with a twisting rounded-triangle section, on a stone plinth
     still — fluted vase, pear and two apples on a plinth half covered by a draped cloth
   A chart is what chart() in talk/scenes.js builds, plus `mat: {rough, metal}` and, on the ground
   only, `lock: true`. Two things differ from talk/scenes.js on purpose: fn is sampled at the Greville
   abscissae (so the surface at (u, v) sits where fn(u, v) is), and the control points are then pulled
   towards interpolation by a few passes of progressive-iterative approximation (P += target - S(P)),
   so a sampled shape such as the teapot is matched closely instead of being smoothed away.
   Exposed as KnotLab.scenes[key] = { label, credit, build() -> { charts, views, tour, rig, radius } }.
   Classic script. Depends on talk/nurbs.js, talk/kit.js and lab/teapot-data.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, TAU = 6.283185307, PI = TAU / 2;
  var HEX = ['#4db6ac', '#f06292', '#9ccc65', '#7986cb', '#bcaaa4', '#78909c', '#a5d6a7', '#b0bec5'];
  var PAPER = [0.93, 0.92, 0.89];

  /* ---------- small procedural toolbox (as in talk/scenes.js) ---------- */
  function fract(x) { return x - Math.floor(x); }
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function h2(x, y) { return fract(Math.sin(x * 127.1 + y * 311.7) * 43758.5453); }
  function vn(x, y) {
    var xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi, u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    return (h2(xi, yi) * (1 - u) + h2(xi + 1, yi) * u) * (1 - v) + (h2(xi, yi + 1) * (1 - u) + h2(xi + 1, yi + 1) * u) * v;
  }
  function fbm(x, y) { return 0.5 * vn(x, y) + 0.25 * vn(x * 2.1, y * 2.1) + 0.125 * vn(x * 4.3, y * 4.3) + 0.0625 * vn(x * 8.7, y * 8.7); }
  /* noise that wraps cleanly around a closed chart: s in [0,1) rides a circle of circumference k */
  function pfbm(s, k, t) { var R = k / TAU, a = s * TAU; return fbm(R * Math.cos(a) + 31.7 + (t || 0) * 0.83, R * Math.sin(a) + 17.3 - (t || 0) * 0.61); }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function mul(c, k) { return [c[0] * k, c[1] * k, c[2] * k]; }
  function ss(a, b, x) { var t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }
  function bell(x) { return x * x > 20 ? 0 : Math.exp(-0.5 * x * x); }
  /* box-filtered coverage of a 0/1 pattern over one control cell: painted edges stay clean */
  function aa(f, x, y, dx, dy) {
    var s = 0, i, j;
    for (j = -1; j <= 1; j++) for (i = -1; i <= 1; i++) s += f(x + i * dx * 0.25, y + j * dy * 0.25);
    return s / 9;
  }

  /* ---------- charts ---------- */
  /* Greville abscissae: the parameter each control point "owns" (cubic; clamped or periodic) */
  function grev(n, periodic) {
    var g = new Float32Array(n), i, U;
    if (periodic) { for (i = 0; i < n; i++) g[i] = ((i + n - 1) % n) / n; return g; }
    U = NB.clampedKnots(n, 3);
    for (i = 0; i < n; i++) g[i] = (U[i + 1] + U[i + 2] + U[i + 3]) / 3;
    return g;
  }
  /* One chart = one patch + how its (u,v) domain is tiled for sampling (same record as talk/scenes.js).
     fn(s, t, i, j) -> [x, y, z, r, g, b]: the surface point and its albedo at parameter (s, t).
     o.fit = passes of P += fn - S (default 2; 0 = use the samples as the control net itself); o.cfit = the
     same for the colours (default 0). */
  function chart(list, name, o, fn) {
    var p = NB.makePatch(o), nu = o.nu, nv = o.nv, us = grev(nu, o.periodicU), vs = grev(nv, o.periodicV), i, j, q, r, k;
    var F = new Float32Array(nu * nv * 3);
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
      r = fn(us[i], vs[j], i, j); q = (j * nu + i) * 3;
      F[q] = r[0]; F[q + 1] = r[1]; F[q + 2] = r[2];
      p.C[q] = clamp(r[3], 0, 1); p.C[q + 1] = clamp(r[4], 0, 1); p.C[q + 2] = clamp(r[5], 0, 1);
    }
    p.P.set(F);
    var g = { pos: new Float32Array(nu * nv * 3) }, fit = o.fit == null ? 2 : o.fit;
    for (k = 0; k < fit; k++) { NB.patchGrid(p, us, vs, g); for (q = 0; q < F.length; q++) p.P[q] += F[q] - g.pos[q]; }
    if (o.cfit) {                                   // the same for the albedo, where a painted pattern has to stay crisp
      var K = new Float32Array(p.C); g.col = new Float32Array(nu * nv * 3);
      for (k = 0; k < o.cfit; k++) { NB.patchGrid(p, us, vs, g); for (q = 0; q < K.length; q++) p.C[q] += K[q] - g.col[q]; }
      var lo = [1, 1, 1], hi = [0, 0, 0];           // never brighter or darker than what was painted
      for (q = 0; q < K.length; q++) { lo[q % 3] = Math.min(lo[q % 3], K[q]); hi[q % 3] = Math.max(hi[q % 3], K[q]); }
      for (q = 0; q < K.length; q++) p.C[q] = clamp(p.C[q], lo[q % 3], hi[q % 3]);
    }
    var tu = o.tu || 4, tv = o.tv || 4;
    var c = { name: name, hex: HEX[list.length % HEX.length], patch: p, closed: !!o.closed, flip: 1, ref: o.ref || null,
              tu: tu, tv: tv, gu: tu * (o.cu || 5), gv: tv * (o.cv || 5), minA: o.minA || 2, minB: o.minB || 2, tiles: [], count: 0,
              mat: o.mat || { rough: 0.6, metal: 0 } };
    if (o.lock) c.lock = true;
    for (j = 0; j < tv; j++) for (i = 0; i < tu; i++) c.tiles.push({ iu: i, iv: j, na: 0, nb: 0 });
    c.base = new Float32Array(p.P); list.push(c);
    return c;
  }
  /* the surface itself on a grid, twice as fine as the control net */
  function surface(c) {
    var p = c.patch, na = p.nu * 2, nb = p.nv * 2, us = new Float32Array(na), vs = new Float32Array(nb), i;
    for (i = 0; i < na; i++) us[i] = (i + 0.5) / na;
    for (i = 0; i < nb; i++) vs[i] = (i + 0.5) / nb;
    return NB.patchGrid(p, us, vs, { pos: new Float32Array(na * nb * 3) }).pos;
  }
  function lowest(c) { var s = surface(c), m = 1e9, i; for (i = 1; i < s.length; i += 3) if (s[i] < m) m = s[i]; return m; }
  /* put a chart down: translate it so the lowest point of its SURFACE (not of its control net) is at y */
  function rest(c, y) {
    var d = y - lowest(c), P = c.patch.P, i;
    for (i = 1; i < P.length; i += 3) P[i] += d;
    c.base.set(P);
    return d;
  }
  /* a cubic B-spline curve from a control polygon [[x,y],...], re-parametrised by arc length; or, with
     `natural`, in its own parameter, so rows crowd where the polygon's points do (rims, foot rings) */
  function path(pts, natural) {
    var U = NB.clampedKnots(pts.length, 3), M = 320, tab = [], len = [0], i;
    for (i = 0; i <= M; i++) {
      tab.push(NB.curvePoint(pts, null, 3, U, i / M));
      if (i) len.push(len[i - 1] + Math.hypot(tab[i][0] - tab[i - 1][0], tab[i][1] - tab[i - 1][1]));
    }
    var f = natural ? function (t) { return NB.curvePoint(pts, null, 3, U, clamp(t, 0, 1)); } : function (t) {
      var x = clamp(t, 0, 1) * len[M], lo = 0, hi = M, mid;
      while (hi - lo > 1) { mid = (lo + hi) >> 1; if (len[mid] <= x) lo = mid; else hi = mid; }
      var k = (x - len[lo]) / ((len[hi] - len[lo]) || 1);
      return [lerp(tab[lo][0], tab[hi][0], k), lerp(tab[lo][1], tab[hi][1], k)];
    };
    f.len = len[M];
    return f;
  }
  /* monotone piecewise-linear map of [0,1] from segments [[length, share of the control points], ...] */
  function warp(segs, x0) {
    var N = 0, L = 0, ks = [0], xs = [x0 || 0], i;
    segs.forEach(function (s) { L += s[0]; N += s[1]; ks.push(N); xs.push((x0 || 0) + L); });
    return function (t) {
      var x = clamp(t, 0, 1) * N;
      for (i = 1; i < ks.length - 1 && ks[i] < x; i++);
      return lerp(xs[i - 1], xs[i], (x - ks[i - 1]) / (ks[i] - ks[i - 1]));
    };
  }
  /* surface of revolution about a vertical axis through o.c. o.prof = control polygon [[radius, height], ...]
     of the profile curve; o.mod(a, t, r, y) -> [r', y', ox, oz] bends it out of the perfectly round;
     o.xf(p) rotates the local point (tilted fruit); o.inside(t, r, y) -> a profile point on the solid side
     of the wall, for thin shells such as a cup (default: the axis). colour(a, t, r, y, s) -> albedo. */
  function lathe(list, name, o, colour) {
    var pf = o.path || path(o.prof), c = o.c, tw = o.rows || function (t) { return t; };
    function world(a, r, y, ox, oz) {
      var p = [(ox || 0) + r * Math.cos(a), y, (oz || 0) + r * Math.sin(a)];
      if (o.xf) p = o.xf(p);
      return [c[0] + p[0], c[1] + p[1], c[2] + p[2]];
    }
    return chart(list, name, { nu: o.nu, nv: o.nv, periodicU: true, closed: true, tu: o.tu || 4, tv: o.tv || 2, cu: o.cu || 4, cv: o.cv || 5,
      minA: o.minA || 4, mat: o.mat, fit: o.fit, cfit: o.cfit,
      ref: function (u, v) { var q = pf(tw(v)), k = o.inside ? o.inside(tw(v), q[0], q[1]) : [0, q[1]]; return world(u * TAU, k[0], k[1]); } },
      function (s, t0) {
        var t = tw(t0), q = pf(t), a = s * TAU, m = o.mod ? o.mod(a, t, q[0], q[1]) : [q[0], q[1], 0, 0];
        var p = world(a, m[0], m[1], m[2], m[3]), k = colour(a, t, q[0], q[1], s);
        return [p[0], p[1], p[2], k[0], k[1], k[2]];
      });
  }
  function ground(list) {
    return chart(list, 'ground', { nu: 6, nv: 6, tu: 6, tv: 6, fit: 0, lock: true, mat: { rough: 0.9, metal: 0 } }, function (s, t) {
      var x = s * 24 - 12, z = t * 24 - 12, c = mul(PAPER, 0.985 + 0.03 * vn(x * 0.21 + 4, z * 0.21));
      return [x, 0, z, c[0], c[1], c[2]];
    });
  }
  /* the 24 capture cameras and the subject's bounding sphere (everything but the ground) */
  function frame(charts) {
    var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], R = 0, k, i, cams = [];
    var subj = charts.filter(function (c) { return !c.lock; });
    subj.forEach(function (c) { var P = c.patch.P; for (i = 0; i < P.length; i++) { k = i % 3; if (P[i] < lo[k]) lo[k] = P[i]; if (P[i] > hi[k]) hi[k] = P[i]; } });
    var tg = [0, 1, 2].map(function (j) { return Math.round((lo[j] + hi[j]) * 50) / 100; });
    subj.forEach(function (c) { var P = c.patch.P; for (i = 0; i < P.length; i += 3) R = Math.max(R, Math.hypot(P[i] - tg[0], P[i + 1] - tg[1], P[i + 2] - tg[2])); });
    R = Math.ceil(R * 100) / 100;
    var D = R / (0.7 * Math.tan(18 * PI / 180));         // the sphere spans 70 % of a 36-degree frame
    [[20, 0], [50, 0.5]].forEach(function (ring) {
      var el = ring[0] * PI / 180;
      for (i = 0; i < 12; i++) { var a = (i + ring[1]) / 12 * TAU; cams.push([tg[0] + D * Math.cos(el) * Math.sin(a), tg[1] + D * Math.sin(el), tg[2] + D * Math.cos(el) * Math.cos(a)]); }
    });
    return { rig: { cams: cams, target: tg }, radius: R };
  }
  function scene(C, views, tour) { var f = frame(C); return { charts: C, views: views, tour: tour, rig: f.rig, radius: f.radius }; }

  /* ================= the Utah teapot's Bezier patches ================= */
  /* 32 patches x 16 points, [row * 4 + col]: rows run along the profile / the tube, columns around it */
  function teapotPatches() {
    var T = window.KnotLab && window.KnotLab.teapot;
    if (!T) throw new Error('lab/objects.js: load lab/teapot-data.js first');
    var V = T.vertices, I = T.patches, pat = [], n, k, q;
    for (n = 0; n < 32; n++) {
      pat[n] = [];
      for (k = 0; k < 16; k++) { q = I[n * 16 + k] * 3; pat[n].push([V[q], V[q + 1], V[q + 2]]); }
    }
    return pat;
  }
  function bern(t) { var m = 1 - t; return [m * m * m, 3 * m * m * t, 3 * m * t * t, t * t * t]; }
  function bez(pt, u, v) {
    var bu = bern(u), bv = bern(v), x = 0, y = 0, z = 0, r, c, w, q;
    for (r = 0; r < 4; r++) for (c = 0; c < 4; c++) { w = bv[r] * bu[c]; q = pt[r * 4 + c]; x += w * q[0]; y += w * q[1]; z += w * q[2]; }
    return [x, y, z];
  }

  /* ================= TEA — the Utah teapot, a cup and saucer, a linen cloth ================= */
  function tea() {
    var C = [], pat = teapotPatches();
    ground(C);
    var WHITE = [0.94, 0.93, 0.90], COBALT = [0.07, 0.15, 0.50], COBALT2 = [0.13, 0.24, 0.62], TEA = [0.56, 0.27, 0.07];
    var PORCELAIN = { rough: 0.12, metal: 0 }, CF = 2;
    var CL = 0.012;                                           // the cloth lies this far above the ground
    var S = 0.4, YAW = -0.44, PX = -0.5, PY = CL + 0.003, PZ = -0.2, cy = Math.cos(YAW), sy = Math.sin(YAW);
    var CX = 1.22, CZ = 1.08, SAUCER_Y = CL + 0.002, WELL = 0.03;
    function potW(p) { var x = p[0] * S, y = p[2] * S, z = -p[1] * S; return [PX + x * cy + z * sy, PY + y, PZ - x * sy + z * cy]; }
    function glaze(cov, s, t) { return mix(mul(WHITE, 0.985 + 0.03 * vn(s * 9 + 2, t * 7)), mix(COBALT, COBALT2, fbm(s * 37, t * 23)), clamp(cov, 0, 1)); }

    /* --- the painted band: a vine between two lines, a leaf in every hollow. x across one motif of
       width P, y from the lower line (0) to the upper (Hb), both in the same length unit --- */
    function leaf(dx, dy, ang, L, W) {
      var u = dx * Math.cos(ang) + dy * Math.sin(ang), w = -dx * Math.sin(ang) + dy * Math.cos(ang), k = u / (L / 2);
      return k * k < 1 && Math.abs(w) < W * (1 - k * k) ? 1 : 0;
    }
    function vine(x, y, P, Hb, lw) {
      if (y < -lw * 1.6 || y > Hb + lw * 1.6) return 0;
      if (y < 0 || y > Hb) return (y < -lw * 0.5 || y > Hb + lw * 0.5) ? 1 : 0;      // the two border lines
      var ph = x / P * TAU, yc = Hb * (0.5 + 0.19 * Math.sin(ph)), sl = Hb * 0.19 * TAU / P * Math.cos(ph);
      if (Math.abs(y - yc) < lw * 0.55 * Math.sqrt(1 + sl * sl)) return 1;           // the stem
      return Math.max(leaf(x - 0.76 * P, y - 0.70 * Hb, 0.42, 0.40 * P, 0.17 * Hb),   // a leaf above, in the hollow
                      leaf(x - 0.26 * P, y - 0.30 * Hb, 0.42, 0.40 * P, 0.17 * Hb));  // and one below the crest
    }

    /* --- pot body: lid seat + rim + upper body + lower body + bottom, one chart around the axis --- */
    var T0 = 0.04, T1 = 0.19, T2 = 0.575, T3 = 0.875, BNU = 96, BNV = 48;
    function bodyAt(s, t) {
      var q = Math.min(3, Math.floor(s * 4)), u = s * 4 - q, p, k;
      if (t < T0) { k = t / T0; p = bez(pat[q], u, 0); return [p[0] * lerp(1.18, 1.4, k) / 1.4, p[1] * lerp(1.18, 1.4, k) / 1.4, lerp(2.385, 2.4, k)]; }
      if (t < T1) return bez(pat[q], u, (t - T0) / (T1 - T0));
      if (t < T2) return bez(pat[4 + q], u, (t - T1) / (T2 - T1));
      if (t < T3) return bez(pat[8 + q], u, (t - T2) / (T3 - T2));
      return bez(pat[28 + 3 - q], 1 - u, 1 - (t - T3) / (1 - T3));                    // the bottom runs the other way round
    }
    function bodyPaint(s, t) {
      if (t >= T0 && t < T1) return 1;                                                // the rim is solid cobalt
      if (t < T0 || t >= T3) return 0;
      var h = bodyAt(s, t)[2];
      if (h > 0.33 && h < 0.45) return 1;                                             // foot line
      return vine(fract(s * 7) * 1.72, h - 0.96, 1.72, 0.86, 0.11);
    }
    chart(C, 'pot body', { nu: BNU, nv: BNV, periodicU: true, closed: true, tu: 8, tv: 6, minA: 3, mat: PORCELAIN, cfit: CF,
      ref: function (u, v, pos) { return [PX, pos[1], PZ]; } }, function (s, t) {
      var p = potW(bodyAt(fract(s), t)), c = glaze(aa(function (a, b) { return bodyPaint(fract(a), clamp(b, 0, 1)); }, s, t, 1 / BNU, 1 / (BNV - 3)), s, t);
      return [p[0], p[1], p[2], c[0], c[1], c[2]];
    });

    /* --- lid: knob + dish --- */
    var LNU = 32, LNV = 18, LT = 0.46;
    function lidAt(s, t) {
      var q = Math.min(3, Math.floor(s * 4)), u = s * 4 - q;
      return t < LT ? bez(pat[20 + q], u, t / LT) : bez(pat[24 + q], u, (t - LT) / (1 - LT));
    }
    function lidPaint(s, t) {
      var p = lidAt(s, t), r = Math.hypot(p[0], p[1]);
      if (t < LT * 0.62) return 1;                                                    // the knob
      if (t < LT) return 0;
      return (r > 1.07 && r < 1.24) || (r > 0.52 && r < 0.66) ? 1 : 0;                // a line at the edge, a ring round the knob
    }
    chart(C, 'lid', { nu: LNU, nv: LNV, periodicU: true, closed: true, tu: 4, tv: 3, minA: 4, mat: PORCELAIN, cfit: CF,
      ref: function (u, v, pos) { return [PX, pos[1] - 0.1, PZ]; } }, function (s, t) {
      var p = potW(lidAt(fract(s), t)), c = glaze(aa(function (a, b) { return lidPaint(fract(a), clamp(b, 0, 1)); }, s, t, 1 / LNU, 1 / (LNV - 3)), s, t);
      return [p[0], p[1], p[2], c[0], c[1], c[2]];
    });

    /* --- spout and handle: four patches each, a tube closed around and open along --- */
    function tube(name, first, split, nu, nv, paint) {
      function at(s, t) {
        var half = Math.min(1, Math.floor(s * 2)), u = s * 2 - half;
        return t < split ? bez(pat[first + half], u, t / split) : bez(pat[first + 2 + half], u, (t - split) / (1 - split));
      }
      chart(C, name, { nu: nu, nv: nv, periodicU: true, closed: true, tu: 2, tv: 4, cu: 4, cv: 5, minA: 8, mat: PORCELAIN, cfit: CF,
        ref: function (u, v) { var a = at(0, v), b = at(0.5, v); return potW([(a[0] + b[0]) / 2, 0, (a[2] + b[2]) / 2]); } }, function (s, t) {
        var p = potW(at(fract(s), t)), c = glaze(aa(function (a, b) { return paint(fract(a), clamp(b, 0, 1)); }, s, t, 1 / nu, 1 / (nv - 3)), s, t);
        return [p[0], p[1], p[2], c[0], c[1], c[2]];
      });
    }
    tube('spout', 16, 0.72, 14, 18, function (s, t) { return t > 0.74 ? 1 : (t > 0.10 && t < 0.15) ? 1 : 0; });
    tube('handle', 12, 0.48, 12, 22, function (s, t) { return Math.abs(s - 0.5) < 0.13 && t > 0.10 && t < 0.88 ? 1 : 0; });   // one brush stroke down its back

    /* --- saucer and cup: thin shells of revolution, so the wall thickness shows at the rim --- */
    function peak(pf, k) { var best = 0, i, t; for (i = 0; i <= 400; i++) { t = i / 400; if (pf(t)[k] > pf(best)[k]) best = t; } return best; }
    var SPROF = [[0, 0.014], [0.14, 0.014], [0.25, 0.014], [0.275, 0], [0.31, 0], [0.33, 0.014], [0.42, 0.03], [0.55, 0.062], [0.63, 0.092], [0.652, 0.105],
                 [0.648, 0.116], [0.62, 0.108], [0.54, 0.079], [0.42, 0.048], [0.31, 0.034], [0.265, WELL], [0.2, WELL], [0.1, WELL], [0, WELL]];
    var saucerPath = path(SPROF, true), SAUCER_RIM = peak(saucerPath, 0);
    lathe(C, 'saucer', { c: [CX, SAUCER_Y, CZ], nu: 32, nv: 16, path: saucerPath, tu: 4, tv: 2, minA: 5, mat: PORCELAIN, cfit: CF,
      inside: function (t, r, y) { return [r, y + (t < SAUCER_RIM ? 0.012 : -0.012)]; } }, function (a, t, r, y, s) {
      return glaze(aa(function (s1, t1) {
        var rr = saucerPath(t1)[0];
        return (t1 > SAUCER_RIM - 0.03 && rr > 0.56) || (t1 > SAUCER_RIM && rr > 0.275 && rr < 0.35) ? 1 : 0;   // a cobalt edge, a ring round the well
      }, s, t, 1 / 32, 1 / 13), s, t);
    });
    var CH = 0.44, TEA_Y = 0.30, CUP_Y = SAUCER_Y + WELL + 0.002;
    var CPROF = [[0, 0.02], [0.1, 0.02], [0.17, 0.02], [0.195, 0], [0.225, 0], [0.238, 0.03], [0.30, 0.075], [0.36, 0.14], [0.395, 0.22], [0.418, 0.31], [0.43, 0.39], [0.436, CH - 0.004],
                 [0.426, CH + 0.006], [0.412, CH - 0.004], [0.406, 0.40], [0.399, 0.35], [0.391, TEA_Y + 0.012], [0.385, TEA_Y], [0.37, TEA_Y], [0.2, TEA_Y], [0, TEA_Y]];
    var cupPath = path(CPROF, true), CUP_RIM = peak(cupPath, 1);
    lathe(C, 'cup', { c: [CX, CUP_Y, CZ], nu: 40, nv: 24, path: cupPath, tu: 4, tv: 3, minA: 5, mat: PORCELAIN, cfit: CF,
      inside: function (t, r, y) { return t < CUP_RIM ? [0, y] : [r + 0.03, y - 0.03]; } }, function (a, t, r, y, s) {
      if (t > CUP_RIM && y < TEA_Y + 0.004) return mul(TEA, 0.96 + 0.08 * vn(r * 9 * Math.cos(a), r * 9 * Math.sin(a)));                       // the tea
      return glaze(aa(function (s1, t1) {
        var yy = cupPath(t1)[1];
        if (t1 > CUP_RIM) return 0;
        if (yy > CH - 0.06) return 1;                                                 // rim band
        if (yy > 0.085 && yy < 0.125) return 1;                                       // a line above the foot
        return leaf((fract(s1 * 5) - 0.5) * 0.49, yy - 0.265, 0.45, 0.31, 0.062);      // five leaves, the pot's own
      }, s, t, 1 / 40, 1 / 21), s, t);
    });
    /* cup handle: an ear in the plane through the cup's axis */
    var HA = 0.62, hx = Math.cos(HA), hz = Math.sin(HA), ear = path([[0.42, 0.365], [0.48, 0.385], [0.565, 0.375], [0.60, 0.30], [0.585, 0.21], [0.50, 0.135], [0.40, 0.115], [0.33, 0.125]]);
    chart(C, 'cup handle', { nu: 8, nv: 14, periodicU: true, closed: true, tu: 2, tv: 3, cu: 4, cv: 5, minA: 7, mat: PORCELAIN,
      ref: function (u, v) { var q = ear(v); return [CX + q[0] * hx, CUP_Y + q[1], CZ + q[0] * hz]; } }, function (s, t) {
      var q = ear(t), q1 = ear(Math.min(1, t + 0.01)), q0 = ear(Math.max(0, t - 0.01)), tx = q1[0] - q0[0], ty = q1[1] - q0[1], tl = Math.hypot(tx, ty) || 1;
      var nx = ty / tl, ny = -tx / tl, a = s * TAU, w = 0.026 + 0.012 * (1 - Math.sin(t * PI)), ca = Math.cos(a) * w, sa = Math.sin(a) * w * 1.35;
      var r = q[0] + nx * ca, c = glaze(0, s, t);
      return [CX + r * hx - sa * hz, CUP_Y + q[1] + ny * ca, CZ + r * hz + sa * hx, c[0], c[1], c[2]];
    });

    /* --- cloth: a square of terracotta linen, flat under the pot and the saucer, a few soft folds, one corner lifted --- */
    var LINEN = [0.72, 0.41, 0.29], LINEN2 = [0.86, 0.78, 0.66], HALF = 2.05, CA = -0.21, cca = Math.cos(CA), sca = Math.sin(CA), CLX = 0.2, CLZ = 0.3, CN = 40;
    var RIDGES = [[-0.15, 1.05, -1.75, 2.05, 0.095, 0.10], [0.75, -1.0, 2.05, -1.75, 0.08, 0.11], [-1.3, -0.75, -2.05, -1.85, 0.07, 0.10], [-0.75, 1.55, 0.2, 2.05, 0.05, 0.12],
                  [0.45, -0.15, 2.05, -0.4, 0.065, 0.10], [-1.25, 0.3, -2.05, 0.65, 0.06, 0.09]];     // [from a, b, to a, b, height, width]: each grows towards the hem
    function clothH(a, b, x, z) {
      var free = ss(0.62, 1.0, Math.hypot(x - PX, z - PZ)) * ss(0.67, 1.0, Math.hypot(x - CX, z - CZ)), h = 0, i, R, dx, dz, L, k, px, pz, d;
      for (i = 0; i < RIDGES.length; i++) {
        R = RIDGES[i]; dx = R[2] - R[0]; dz = R[3] - R[1]; L = Math.hypot(dx, dz);
        k = clamp(((a - R[0]) * dx + (b - R[1]) * dz) / (L * L), 0, 1);
        px = R[0] + dx * k; pz = R[1] + dz * k;
        d = Math.hypot(a - px, b - pz) + 0.05 * Math.sin(k * 5 + i * 2.1) * 0;
        h += R[4] * (0.25 + 0.75 * k) * ss(0, 0.25, k) * bell(d / (R[5] * (0.7 + 0.5 * k)));
      }
      h += 0.012 * fbm(a * 1.1 + 3, b * 1.1 + 8);
      var cd = Math.hypot(a - HALF, b - HALF), lift = ss(0.95, 0, cd);            // the front right corner, picked up
      return CL + h * free + 0.2 * lift * lift * (0.6 + 0.4 * ss(0.5, 0, cd));
    }
    chart(C, 'cloth', { nu: CN, nv: CN, tu: 6, tv: 6, mat: { rough: 0.95, metal: 0 } }, function (s, t, i, j) {
      var a = (s * 2 - 1) * HALF, b = (t * 2 - 1) * HALF, x = CLX + a * cca + b * sca, z = CLZ - a * sca + b * cca;
      var c = mul(LINEN, (0.955 + 0.09 * h2(i, 1.7)) * (0.955 + 0.09 * h2(3.3, j)) * (0.97 + 0.06 * fbm(a * 2.3, b * 2.3)));   // slub yarns, warp and weft
      var e = HALF - Math.abs(b);
      if (e > 0.2 && e < 0.29 || e > 0.36 && e < 0.40) c = mix(c, LINEN2, 0.85);                                        // woven stripes along two edges
      var cd = Math.hypot(a - HALF, b - HALF), lift = ss(0.95, 0, cd);
      return [x - 0.07 * lift * lift, clothH(a, b, x, z), z - 0.07 * lift * lift, c[0], c[1], c[2]];
    });
    return scene(C,
      { wide: { th: 0.55, ph: 1.12, r: 9.6, tx: 0.2, ty: 0.45, tz: 0.2 }, close: { th: 0.2, ph: 1.25, r: 3.3, tx: -0.35, ty: 0.62, tz: -0.1 },
        cup: { th: 1.0, ph: 1.0, r: 2.6, tx: CX, ty: 0.3, tz: CZ }, low: { th: -0.5, ph: 1.42, r: 6.0, tx: 0, ty: 0.55, tz: 0 },
        top: { th: 0.3, ph: 0.45, r: 8.0, tx: 0.2, ty: 0.3, tz: 0.3 } },
      ['wide', 'close', 'cup', 'low', 'top']);
  }

  /* ================= KNOT — a trefoil in brushed bronze on a travertine plinth ================= */
  function knot() {
    var C = [], KS = 0.62, RIN = 0.25, RMAX = 0.41, NU = 240, NV = 24, TWIST = 1, PH = 0.62, PR = 1.3;
    var BRONZE = [0.72, 0.45, 0.20], BRONZE2 = [0.60, 0.40, 0.21], PATINA = [0.50, 0.47, 0.30], STONE = [0.86, 0.81, 0.72];
    ground(C);
    lathe(C, 'plinth', { c: [0, 0, 0], nu: 32, nv: 14, tu: 4, tv: 3, minA: 5, mat: { rough: 0.8, metal: 0 }, fit: 0,
      prof: [[PR, 0], [PR, 0.02], [PR, 0.2], [PR, PH - 0.12], [PR, PH - 0.03], [PR - 0.006, PH - 0.006], [PR - 0.03, PH], [PR - 0.14, PH], [0.7, PH], [0.35, PH], [0, PH]] },
      function (a, t, r, y, s) {
        var top = y > PH - 0.004, x = r * Math.cos(a), z = r * Math.sin(a);
        var strata = top ? fbm(x * 1.3 + 5, z * 7) : pfbm(s, 5, y * 26);              // bedding: bands on the side, veins across the top
        var c = mul(STONE, 0.93 + 0.11 * strata);
        return mix(c, [0.74, 0.66, 0.54], 0.5 * ss(0.56, 0.7, top ? fbm(x * 4, z * 14 + 3) : pfbm(s, 22, y * 60)));   // filled pores
      });
    /* the knot's centre line on a torus of radii 2 and 1 about the z axis, and the torus's own frame:
       n1 = the torus surface normal (always square to the curve, never flips), n2 = tangent x n1 */
    function frameAt(u) {
      var a = u * 2 * TAU, f = 1.5 * a, cf = Math.cos(f), sf = Math.sin(f), ca = Math.cos(a), sa = Math.sin(a), R = 2 + cf;
      var tx = -1.5 * sf * ca - R * sa, ty = -1.5 * sf * sa + R * ca, tz = 1.5 * cf, tl = Math.hypot(tx, ty, tz);
      tx /= tl; ty /= tl; tz /= tl;
      var n = [cf * ca, cf * sa, sf];
      return { p: [R * ca, R * sa, sf], n1: n, n2: [ty * n[2] - tz * n[1], tz * n[0] - tx * n[2], tx * n[1] - ty * n[0]] };
    }
    /* stand it up with one lobe at the top: local (x, y) turned a quarter turn in its own plane */
    function up(v) { return [-v[1], v[0], v[2]]; }
    /* the section: a triangle with flat sides and soft corners (one corner at a = 0), as a polar radius */
    function section(a) {
      var d = Math.abs(a - Math.round(a / (TAU / 3)) * TAU / 3), flat = RIN / Math.cos(PI / 3 - d), h = 0.16, k = Math.max(0, 1 - Math.abs(flat - RMAX) / h);
      return Math.min(flat, RMAX) - h * k * k / 4;
    }
    var k = chart(C, 'knot', { nu: NU, nv: NV, periodicU: true, periodicV: true, closed: true, tu: 48, tv: 1, cu: 4, cv: 12, minB: 18, mat: { rough: 0.32, metal: 1 },
      ref: function (u) { var f = frameAt(u), p = up(f.p); return [p[0] * KS, Y0 + p[1] * KS, p[2] * KS]; } }, function (s, t, i, j) {
      var f = frameAt(s), p = up(f.p), n1 = up(f.n1), n2 = up(f.n2), a = t * TAU - 0.55 * Math.sin(3 * t * TAU) / 3, rho = section(a), al = a + TWIST * TAU * s;   // rows crowd at the corners
      var ca = Math.cos(al) * rho, sa = Math.sin(al) * rho;
      var c = mix(BRONZE, BRONZE2, pfbm(s, 14, t * 2 + Math.cos(a) * 0.6));
      c = mix(c, PATINA, 0.30 * ss(0.50, 0.72, pfbm(s, 9, 4 + Math.sin(a) * 0.5)));
      c = mul(c, 0.95 + 0.10 * pfbm(s, 60, (j % NV) * 5.3));                           // brushed along its length
      return [p[0] * KS + n1[0] * ca + n2[0] * sa, p[1] * KS + n1[1] * ca + n2[1] * sa, p[2] * KS + n1[2] * ca + n2[2] * sa, c[0], c[1], c[2]];
    });
    var Y0 = 0; Y0 = rest(k, PH - 0.03);                                              // both feet sunk a little into the stone
    return scene(C,
      { wide: { th: 0.5, ph: 1.28, r: 9.0, tx: 0, ty: 2.1, tz: 0 }, close: { th: 0.9, ph: 1.2, r: 3.4, tx: 0.3, ty: 3.0, tz: 0.2 },
        foot: { th: -0.4, ph: 1.4, r: 3.2, tx: 0, ty: 1.0, tz: 0 }, side: { th: 1.6, ph: 1.3, r: 7.5, tx: 0, ty: 2.2, tz: 0 },
        top: { th: 0.2, ph: 0.7, r: 7.5, tx: 0, ty: 2.0, tz: 0 } },
      ['wide', 'close', 'side', 'foot', 'top']);
  }

  /* ================= STILL — vase, pear and apples on a draped plinth ================= */
  function still() {
    var C = [], OX = -0.5, OZ = -0.4, BX = 1.4, BZ = 0.8, H = 0.62;                  // the plinth: a box 2.8 x 0.62 x 1.6 centred on (OX, OZ)
    var TH = 0.012, CLR = 0.014, TOP = H + CLR;                                       // cloth above the ground / clear of the stone
    var STONE = [0.80, 0.78, 0.74], FLAX = [0.76, 0.70, 0.59], INDIGO = [0.16, 0.20, 0.36], TEAL = [0.04, 0.31, 0.35], CLAY = [0.78, 0.65, 0.52];
    ground(C);

    /* --- plinth: a rounded-rectangle section lofted up the wall and in across the top --- */
    var RC = 0.045, per = warp([[BX - RC, 5], [RC * PI / 2, 4], [2 * (BZ - RC), 6], [RC * PI / 2, 4], [2 * (BX - RC), 10], [RC * PI / 2, 4], [2 * (BZ - RC), 6], [RC * PI / 2, 4], [BX - RC, 5]]);
    function rrect(s) {                              // point on the box's plan outline by arc length, from the middle of the front edge, anticlockwise from above
      var d = per(s), hx = BX - RC, hz = BZ - RC, q = RC * PI / 2, L = [hx, q, 2 * hz, q, 2 * hx, q, 2 * hz, q, hx], k = 0, a;
      while (k < 8 && d > L[k]) { d -= L[k]; k++; }
      switch (k) {
        case 0: return [d, BZ]; case 8: return [-hx + d, BZ];
        case 1: a = d / RC; return [hx + RC * Math.sin(a), hz + RC * Math.cos(a)];
        case 2: return [BX, hz - d];
        case 3: a = d / RC; return [hx + RC * Math.cos(a), -hz - RC * Math.sin(a)];
        case 4: return [hx - d, -BZ];
        case 5: a = d / RC; return [-hx - RC * Math.sin(a), -hz - RC * Math.cos(a)];
        case 6: return [-BX, -hz + d];
        default: a = d / RC; return [-hx - RC * Math.cos(a), hz + RC * Math.sin(a)];
      }
    }
    var wall = path([[1, 0], [1, 0.03], [1, 0.2], [1, H - 0.1], [1, H - 0.02], [0.997, H - 0.004], [0.985, H], [0.93, H], [0.6, H], [0.3, H], [0, H]]);
    chart(C, 'plinth', { nu: 48, nv: 12, periodicU: true, closed: true, tu: 8, tv: 2, minA: 3, fit: 0, mat: { rough: 0.85, metal: 0 },
      ref: function () { return [OX, H / 2, OZ]; } }, function (s, t) {
      var q = rrect(fract(s)), w = wall(t), k = w[0], x = q[0], z = q[1];
      x *= k; z *= k;
      var top = w[1] > H - 0.004, c = mul(STONE, 0.94 + 0.10 * (top ? fbm((OX + x) * 1.1, (OZ + z) * 6) : fbm(s * 40, w[1] * 22)));
      return [OX + x, w[1], OZ + z, c[0], c[1], c[2]];
    });

    /* --- drape: a linen cloth thrown over the right-hand part of the plinth. It is modelled in the flat:
       a point of the cloth at (x, z), laid out at the height of the plinth top, is at distance d outside
       the top's rounded outline; the cloth spends d first over the edge, then down the wall, then out
       across the ground (an arc-length profile, so it does not stretch), and folds push it OUTWARDS only
       along that profile's normal: away from the stone and up from the ground, never into either. --- */
    var DR = 0.06, E = 0.04, G = 0.07, PHI = 1.45;                                     // outline radius, edge roll, ground turn, fall angle
    var yG = -(TOP - TH), D1 = E * PHI, LS = (-E * (1 - Math.cos(PHI)) - (yG + G * (1 - Math.cos(PHI)))) / Math.sin(PHI), D2 = D1 + LS, D3 = D2 + G * PHI;
    function profile(d) {                           // -> [out, down, normal out, normal up]
      var a, x1 = E * Math.sin(PHI), y1 = -E * (1 - Math.cos(PHI)), cs = Math.cos(PHI), sn = Math.sin(PHI), x2, y2;
      if (d < D1) { a = d / E; return [E * Math.sin(a), -E * (1 - Math.cos(a)), Math.sin(a), Math.cos(a)]; }
      if (d < D2) return [x1 + (d - D1) * cs, y1 - (d - D1) * sn, sn, cs];
      x2 = x1 + LS * cs; y2 = y1 - LS * sn;
      if (d < D3) { a = PHI - (d - D2) / G; return [x2 + G * sn - G * Math.sin(a), y2 + G * cs - G * Math.cos(a), Math.sin(a), Math.cos(a)]; }
      return [x2 + G * sn + (d - D3), yG, 0, 1];
    }
    /* hanging folds: an irregular undulation along the edge, 0 in the hollows and 1 on the crests */
    function waves(w, d) {
      var a = Math.sin(TAU * (w + 0.11 * Math.sin(1.7 * w + 0.5) - 0.12 * d) / 0.47 + 0.6);
      var b = Math.sin(TAU * (w + 0.2 * d) / 0.29 + 2.1 + 0.9 * Math.sin(0.83 * w));
      var m = 0.5 + 0.5 * (0.64 * a + 0.36 * b);
      return m * m * (3 - 2 * m);
    }
    /* where it pools: creases that wander across the fall line instead of following it */
    function crumple(x, z) {
      var n = 2 * vn(x * 2.5 + 5.3 + 0.5 * Math.sin(z * 1.9), z * 2.5 + 1.7 + 0.5 * Math.sin(x * 1.6)) - 1, r = 1 - Math.sqrt(n * n + 0.03);
      return r * r;
    }
    var COR = 0.5;                                  // how much fold coordinate a corner's quarter fan is given
    function drapeAt(x, z, hem) {
      var hx = BX + CLR - DR, hz = BZ + CLR - DR, qx = clamp(x, -hx, hx), qz = clamp(z, -hz, hz), vx = x - qx, vz = z - qz, l = Math.hypot(vx, vz), d = l - DR;
      if (d <= 0) return [x, TOP, z];
      var nx = vx / l, nz = vz / l, w, q = COR * PI / 2;
      // fold coordinate: along the front edge, round the front right corner, back along the right side, ...
      if (vx === 0 && vz > 0) w = qx + hx;
      else if (vx > 0 && vz > 0) w = 2 * hx + COR * Math.atan2(nx, nz);
      else if (vx > 0 && vz === 0) w = 2 * hx + q + (hz - qz);
      else if (vx > 0) w = 2 * hx + q + 2 * hz + COR * Math.atan2(-nz, nx);
      else if (vx === 0) w = 2 * hx + 2 * q + 2 * hz + (hx - qx);
      else if (vz < 0) w = 4 * hx + 3 * q + 2 * hz - COR * Math.atan2(-nz, -nx);
      else if (vz === 0) w = 4 * hx + 3 * q + 2 * hz + (qz + hz);
      else w = -COR * Math.atan2(-nx, nz);
      var pr = profile(d), edge = 0.5 + 0.5 * ss(0, 0.45, hem), g = ss(D2, D3 + 0.12, d), far = d - D3;
      var cone = bell((w - 2 * hx - q / 2) / 0.34) + bell((w - 2 * hx - 1.5 * q - 2 * hz) / 0.34);          // the one big fold a corner throws
      var up = 0.17 * ss(0, 0.32, d) * edge * (waves(w, d) * (1 - 0.5 * g) * (1 - 0.75 * cone) + 1.25 * cone * (1 - 0.45 * g));
      up += 0.16 * g * crumple(x, z) * (0.45 + 0.55 * ss(1.0, 0.1, far)) * edge;
      var roll = 0.11 * (0.2 + 0.8 * vn(w * 1.9, 3.3)) * bell((far - 0.11) / 0.075) + 0.05 * vn(w * 1.4 + 9, 1.1) * bell((far - 0.5) / 0.13);
      up += roll * ss(0, 0.25, hem);                                                     // it buckles where it meets the ground
      var o = DR + pr[0] + pr[2] * up - 0.35 * roll * ss(0, 0.25, hem);                  // and what the buckle takes up is missing from its reach
      return [qx + nx * o, TOP + pr[1] + pr[3] * up, qz + nz * o];
    }
    var AL = -0.6, AR = BX + 1.55, BB = -BZ - 0.3, BF = BZ + 1.95, DNU = 64, DNV = 65;
    var bend = [[D1 + 0.12, 7], [D2 - D1 - 0.07, 5], [D3 - D2 + 0.09, 4], [0.3, 6]], pre = 0.08 - CLR, used = D3 + 0.44 - pre;   // roll, fall, turn, foot
    var wx = warp([[BX - AL - 0.08, 34]].concat(bend, [[AR - BX - used, 8]]), AL);
    var wz = warp([[0.2, 3], [0.18, 6], [2 * BZ - 0.16, 21]].concat(bend, [[BF - BZ - used, 13]]), BB);
    function stripes(q) { return (q > 0.26 && q < 0.44) || (q > 0.54 && q < 0.63) ? 1 : 0; }
    chart(C, 'drape', { nu: DNU, nv: DNV, tu: 8, tv: 8, cu: 4, cv: 4, cfit: 2, mat: { rough: 0.95, metal: 0 } }, function (s, t, i, j) {
      var x = wx(s), z = wz(t);
      x += 0.16 * ss(0.3, 0, s) * (z / BZ);                                             // the free edge on the plinth lies a little askew
      var hem = Math.min(AR - x, BF - z, z - BB + 0.3, x - AL + 0.25), p = drapeAt(x, z, hem);
      var c = mul(FLAX, (0.955 + 0.09 * h2(i, 2.9)) * (0.955 + 0.09 * h2(5.1, j)) * (0.97 + 0.06 * fbm(x * 2.1, z * 2.1)));   // slub yarns, warp and weft
      c = mix(c, INDIGO, 0.92 * aa(function (a) { return stripes(a); }, BF - z, 0, 0.08, 0));                               // two woven stripes along the front hem
      return [OX + p[0], p[1], OZ + p[2], c[0], c[1], c[2]];
    });

    /* --- vase: tall, fluted, with a flared lip; it stands on the bare stone --- */
    var VX = OX - 0.98, VZ = OZ - 0.12, NF = 12;
    var VPROF = [[0, 0.012], [0.16, 0.012], [0.22, 0.008], [0.245, 0], [0.262, 0.012], [0.25, 0.05], [0.235, 0.10], [0.28, 0.24], [0.38, 0.52], [0.43, 0.86], [0.41, 1.18], [0.32, 1.46],
                 [0.21, 1.68], [0.165, 1.84], [0.17, 1.95], [0.23, 2.04], [0.315, 2.085], [0.335, 2.10], [0.325, 2.112], [0.27, 2.08], [0.19, 2.0], [0.14, 1.9], [0.13, 1.74], [0.135, 1.55], [0.07, 1.5], [0, 1.5]];
    var vase = lathe(C, 'vase', { c: [VX, H, VZ], nu: NF * 6, nv: 28, prof: VPROF, tu: 6, tv: 5, minA: 5, mat: { rough: 0.18, metal: 0 },
      inside: function (t, r, y) { return t < 0.66 ? [0, y] : [r + 0.03, y - 0.02]; },
      mod: function (a, t, r, y) {
        var x = fract(a / TAU * NF), flute = 1 - Math.sqrt(1 - Math.pow(2 * x - 1, 2) * 0.96);    // hollow channels meeting in ridges
        var on = t < 0.66 ? ss(0.16, 0.34, y) * ss(1.72, 1.5, y) : 0;
        return [r * (1 - 0.085 * on * (1 - flute)), y, 0, 0];
      } }, function (a, t, r, y, s) {
        if (t < 0.075) return mul(CLAY, 0.95 + 0.1 * pfbm(s, 20, y * 30));               // the unglazed foot
        return mul(TEAL, 0.9 + 0.22 * pfbm(s, 7, y * 2.2));
      });

    rest(vase, H + 0.001);

    /* --- fruit: one lathe each, stalk included (the profile starts at the stalk's tip) --- */
    function fruit(name, o, skin) {
      var K = 0.78, prof = o.prof.map(function (q) { return [q[0] * K, q[1] * K]; });
      var cx = Math.cos(o.yaw || 0), sx = Math.sin(o.yaw || 0), ct = Math.cos(o.tilt || 0), st = Math.sin(o.tilt || 0), hgt = o.prof[0][1];
      var c = lathe(C, name, { c: [o.at[0], 0, o.at[1]], nu: 32, nv: o.nv, prof: prof, tu: 4, tv: 3, minA: 5, mat: { rough: 0.45, metal: 0 },
        xf: function (p) {                              // tilt about the local z axis, then turn about the vertical
          var x = p[0] * ct - p[1] * st, y = p[0] * st + p[1] * ct;
          return [x * cx + p[2] * sx, y, -x * sx + p[2] * cx];
        },
        inside: function () { return [0, hgt * K * 0.45]; },
        mod: function (a, t, r0, y0) {
          var r = r0 / K, y = y0 / K;                    // the recipe below is written for the unscaled profile
          var body = ss(o.stalk, o.stalk + 0.05, t), k = 1 + body * (o.lop * Math.cos(a - 0.8) + o.lobes * Math.cos(5 * a + 0.4) * ss(hgt * 0.5, 0, y));
          var bend = (y - hgt * 0.4) / hgt, sw = Math.max(0, y - o.neck);                // the body leans, the stalk curls
          return [r0 * k, y0, K * (o.lean * bend * Math.abs(bend) * hgt + o.curl * sw * sw / 0.02), K * o.lean * 0.4 * bend * hgt];
        } }, function (a, t, r, y, s) {
          if (t < o.stalk) return mul([0.30, 0.20, 0.11], 0.85 + 0.3 * vn(a * 2, y * 40));
          return skin(a, t, r / K, y / K, s);
        });
      rest(c, o.on);
      return c;
    }
    var PEAR = [[0, 1.12], [0.016, 1.118], [0.022, 1.09], [0.019, 1.0], [0.022, 0.935], [0.05, 0.925], [0.10, 0.93], [0.145, 0.89], [0.168, 0.80], [0.19, 0.68], [0.245, 0.55],
                [0.32, 0.42], [0.365, 0.29], [0.36, 0.16], [0.29, 0.05], [0.19, 0.0], [0.10, 0.012], [0.04, 0.04], [0, 0.045]];
    var APPLE = [[0, 0.70], [0.014, 0.698], [0.018, 0.66], [0.016, 0.58], [0.02, 0.535], [0.05, 0.535], [0.11, 0.575], [0.19, 0.625], [0.28, 0.615], [0.35, 0.53], [0.378, 0.40],
                 [0.37, 0.26], [0.31, 0.11], [0.22, 0.025], [0.14, 0.0], [0.08, 0.02], [0.035, 0.055], [0, 0.06]];
    function lenticels(c, s, t, n, to) { return h2(Math.floor(s * 32) + 0.5, Math.floor(t * n) + 0.5) > 0.90 ? mix(c, to, 0.6) : c; }
    fruit('pear', { at: [OX + 0.33, OZ - 0.26], on: TOP + 0.002, prof: PEAR, nv: 24, stalk: 0.085, neck: 0.94, lean: 0.10, curl: 0.02, lop: 0.03, lobes: 0.012, yaw: 0.7 },
      function (a, t, r, y, s) {
        var c = mix([0.80, 0.72, 0.22], [0.58, 0.66, 0.20], ss(0.3, 0.95, y) * 0.7 + 0.3 * pfbm(s, 5, y * 3));
        c = mix(c, [0.78, 0.42, 0.20], 0.45 * ss(0.3, 0.85, Math.cos(a - 1.9)) * ss(0.75, 0.3, y));          // a blush on one cheek
        c = mix(c, [0.55, 0.38, 0.18], 0.6 * Math.max(ss(0.86, 0.94, y), ss(0.07, 0.01, y)));                // russet at stalk and calyx
        return lenticels(c, s, t, 44, [0.50, 0.38, 0.16]);
      });
    fruit('apple a', { at: [OX + 0.82, OZ + 0.18], on: TOP + 0.002, prof: APPLE, nv: 22, stalk: 0.10, neck: 0.56, lean: 0.05, curl: 0.03, lop: 0.03, lobes: 0.02, yaw: 2.2 },
      function (a, t, r, y, s) {
        var streak = pfbm(s, 26, y * 1.6), c = mix([0.66, 0.08, 0.08], [0.46, 0.04, 0.06], pfbm(s, 6, y * 4));
        c = mix(c, [0.86, 0.66, 0.22], 0.75 * ss(0.50, 0.68, streak) * ss(0.1, 0.5, Math.cos(a - 4.0) * 0.5 + 0.5 + (0.3 - y) * 0.8));   // yellow breaking through in streaks
        c = mix(c, [0.78, 0.70, 0.26], ss(0.13, 0.03, r) * (y > 0.3 ? 1 : 0.8));                                  // the stalk cavity and the calyx stay green-gold
        return lenticels(c, s, t, 40, [0.92, 0.72, 0.46]);
      });
    fruit('apple b', { at: [OX - 0.9, OZ + 1.8], on: 0.001, prof: APPLE, nv: 22, stalk: 0.10, neck: 0.56, lean: -0.04, curl: -0.03, lop: 0.025, lobes: 0.028, yaw: -0.9, tilt: 1.25 },
      function (a, t, r, y, s) {
        var c = mix([0.50, 0.63, 0.16], [0.68, 0.72, 0.22], pfbm(s, 7, y * 5));
        c = mix(c, [0.80, 0.34, 0.16], 0.55 * ss(0.2, 0.9, Math.cos(a - 1.0)) * ss(0.05, 0.4, y) * (0.6 + 0.4 * pfbm(s, 22, y * 2)));     // a soft blush
        c = mix(c, [0.62, 0.58, 0.24], ss(0.12, 0.03, r) * 0.7);
        return lenticels(c, s, t, 40, [0.84, 0.86, 0.60]);
      });
    return scene(C,
      { wide: { th: 0.5, ph: 1.15, r: 9.8, tx: 0.2, ty: 0.9, tz: 0.3 }, close: { th: 0.35, ph: 1.2, r: 3.0, tx: 0.2, ty: 0.85, tz: -0.3 },
        folds: { th: 0.75, ph: 1.25, r: 4.2, tx: 0.7, ty: 0.25, tz: 0.9 }, vase: { th: -0.3, ph: 1.3, r: 5.0, tx: -1.4, ty: 1.5, tz: -0.5 },
        low: { th: 0.1, ph: 1.45, r: 7.0, tx: 0.2, ty: 0.8, tz: 0 }, top: { th: 0.4, ph: 0.5, r: 8.5, tx: 0.3, ty: 0.5, tz: 0.3 } },
      ['wide', 'folds', 'close', 'vase', 'low', 'top']);
  }

  window.KnotLab = window.KnotLab || {};
  window.KnotLab.scenes = {
    tea:   { label: 'Tea service', credit: 'Utah teapot · Martin Newell, 1975', build: tea },
    knot:  { label: 'Trefoil',     credit: '(2,3) torus knot · twisting rounded-triangle section', build: knot },
    still: { label: 'Still life',  credit: 'vase, pear and apples on a draped plinth · all B-spline charts', build: still }
  };
})();
