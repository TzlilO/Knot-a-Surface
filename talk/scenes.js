/* Knot-a-Surface talk — scene library for the `manifold` module.
   Every scene is nothing but B-spline charts: geometry in the control points, appearance in the
   control features (one colour per control point). The benchmark-style scenes are PROCEDURAL
   STAND-INS modelled after the kind of capture each dataset contains — they are not the datasets.
     barn, truck — Tanks and Temples style (outdoor object, handheld orbit)
     bust        — DTU style (scanned object on a table, camera dome)
     garden      — Mip-NeRF 360 style (object in the middle of an unbounded surround)
     studio      — the abstract still life (vase, cloth, torus knot)
     scan        — the one REAL object: a DTU scan fitted offline (only when talk/dtu-data.js is loaded first)
   Exposed as KnotTalk.scenes[key] = { label, suite, build() -> { charts, views, tour, rig, edit } }.
   Depends on nurbs.js + kit.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, TAU = 6.283185307;
  var SUN = (function () { var v = [0.35, 0.86, 0.36], l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; })();
  /* chart identity colours for the 'charts' view. Deliberately NOT the talk's semantic hues
     (cyan surface, amber control, violet Gaussians, red error) so the amber control net stays readable. */
  var HEX = ['#4db6ac', '#f06292', '#9ccc65', '#7986cb', '#bcaaa4', '#78909c', '#a5d6a7', '#b0bec5'];

  /* ---------- small procedural toolbox ---------- */
  function fract(x) { return x - Math.floor(x); }
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
  function gs(x, y, x0, y0, sx, sy) { return Math.exp(-0.5 * ((x - x0) * (x - x0) / (sx * sx) + (y - y0) * (y - y0) / (sy * sy))); }
  /* baked sun shadow: is the ray from p towards the sun blocked by any box [x0,y0,z0,x1,y1,z1]? */
  function lit(p, boxes) {
    for (var b = 0; b < boxes.length; b++) {
      var B = boxes[b], t0 = 0.02, t1 = 1e9, k, ok = true;
      for (k = 0; k < 3 && ok; k++) {
        var a = (B[k] - p[k]) / SUN[k], c = (B[k + 3] - p[k]) / SUN[k];
        if (a > c) { var s = a; a = c; c = s; }
        if (a > t0) t0 = a; if (c < t1) t1 = c; if (t0 > t1) ok = false;
      }
      if (ok) return 0.52;
    }
    return 1;
  }

  /* One chart = one patch + how its (u,v) domain is tiled for sampling.
     fn(s, t, i, j) -> [x, y, z, r, g, b] for control point (i, j); s, t in [0,1]. */
  function chart(list, name, o, fn) {
    var p = NB.makePatch(o), i, j, q, r;
    for (j = 0; j < o.nv; j++) for (i = 0; i < o.nu; i++) {
      r = fn(o.periodicU ? i / o.nu : i / (o.nu - 1), o.periodicV ? j / o.nv : j / (o.nv - 1), i, j); q = (j * o.nu + i) * 3;
      p.P[q] = r[0]; p.P[q + 1] = r[1]; p.P[q + 2] = r[2]; p.C[q] = r[3]; p.C[q + 1] = r[4]; p.C[q + 2] = r[5];
    }
    var tu = o.tu || 4, tv = o.tv || 4;
    var c = { name: name, hex: HEX[list.length % HEX.length], patch: p, closed: !!o.closed, flip: 1, ref: o.ref || null,
              tu: tu, tv: tv, gu: tu * (o.cu || 5), gv: tv * (o.cv || 5), minA: o.minA || 2, minB: o.minB || 2, tiles: [], count: 0 };
    for (j = 0; j < tv; j++) for (i = 0; i < tu; i++) c.tiles.push({ iu: i, iv: j, na: 0, nb: 0 });
    c.base = new Float32Array(p.P); list.push(c);
    return c;
  }
  /* surface of revolution around the y axis (or z, for wheels); prof = [[radius, height], ...] */
  function revolve(list, name, o, colour) {
    var prof = o.prof, c = o.c, zAxis = o.axis === 'z';
    return chart(list, name, { nu: o.nu, nv: prof.length, periodicU: true, closed: true, tu: o.tu || 4, tv: o.tv || 2,
      cu: o.cu || 4, cv: o.cv || 5, minA: o.minA || 4,
      ref: function (u, v, pos) { return zAxis ? [c[0], c[1], pos[2]] : [c[0], pos[1], c[2]]; } }, function (s, t, i, j) {
      var a = s * TAU, r = prof[j][0], h = prof[j][1], q = o.squash ? o.squash(j) : [1, 1], sx = q[0], sz = q[1];
      var p = zAxis ? [c[0] + r * Math.cos(a), c[1] + r * Math.sin(a), c[2] + h] : [c[0] + sx * r * Math.cos(a), c[1] + h, c[2] + sz * r * Math.sin(a)];
      var k = colour(r, h, a, p);
      return [p[0], p[1], p[2], k[0], k[1], k[2]];
    });
  }
  function ring(n, R, y0, y1, target) {          // a handheld orbit, heights weaving up and down
    var out = [];
    for (var i = 0; i < n; i++) { var a = i / n * TAU; out.push([target[0] + R * Math.sin(a), y0 + (y1 - y0) * (0.5 + 0.5 * Math.sin(a * 3)), target[2] + R * Math.cos(a)]); }
    return { cams: out, target: target };
  }

  /* ================= BARN — Tanks and Temples style ================= */
  function barn() {
    var C = [], W = 3.5, L = 5, boxes = [[-W, 0, -L, W, 5.4, L], [4.5, 0, -3.1, 6.9, 6.6, -0.7]];
    var RED = [0.56, 0.13, 0.10], RED2 = [0.44, 0.09, 0.08], WHITE = [0.88, 0.86, 0.80], DARK = [0.06, 0.07, 0.09];
    function roofY(x) { var a = Math.abs(x); return a > 2.3 ? 3.6 + (3.5 - a) / 1.2 * 2.0 : 5.6 + (2.3 - a) / 2.3 * 1.0; }   // gambrel
    function planks(u, y) {
      var c = mix(RED, RED2, h2(Math.floor(u / 0.44), 3.1));
      return mul(c, (0.80 + 0.34 * fbm(u * 0.9, y * 2.4)) * (0.78 + 0.22 * ss(0, 1.3, y)));
    }
    function win(c, du, y, y0, y1, hw) {         // a framed dark window, if (du, y) falls inside
      if (Math.abs(du) > hw || y < y0 || y > y1) return c;
      return (Math.abs(du) > hw - 0.14 || y < y0 + 0.14 || y > y1 - 0.14 || Math.abs(du) < 0.06) ? WHITE.slice() : DARK.slice();
    }
    var inside = function () { return [0, 2.4, 0]; };

    chart(C, 'ground', { nu: 46, nv: 46, tu: 6, tv: 6 }, function (s, t) {
      var x = s * 44 - 22, z = t * 44 - 22, d = Math.max(Math.abs(x) - W, Math.abs(z) - L);
      var y = d > 0.5 ? 0.5 * (fbm(x * 0.11 + 7, z * 0.11) - 0.45) * ss(0.5, 7, d) : 0;
      var g = mix([0.19, 0.31, 0.12], [0.38, 0.44, 0.17], fbm(x * 0.33, z * 0.33));
      var path = ss(1.9, 0.8, Math.abs(x + 0.9 * Math.sin(z * 0.22))) * ss(L - 0.6, L + 0.8, z);
      g = mix(g, [0.44, 0.36, 0.25], path * 0.92);
      g = mix(g, [0.31, 0.27, 0.19], ss(1.6, 0, d) * 0.75);
      g = mul(g, lit([x, y + 0.02, z], boxes));
      return [x, y, z, g[0], g[1], g[2]];
    });
    function gable(name, zc, front) {
      chart(C, name, { nu: 58, nv: 38, tu: 6, tv: 4, closed: true, ref: inside }, function (s, t) {
        var x = (s * 2 - 1) * W * (front ? 1 : -1), hy = roofY(x), y = t * hy, c = planks(x, y);
        if (Math.abs(x) > W - 0.24 || y > hy - 0.26) c = WHITE.slice();
        if (front) {
          if (Math.abs(x) < 1.6 && y < 3.15) {                                   // the big sliding door
            c = mul(planks(x * 1.6 + 9, y), 1.18);
            var fy = y / 3.15, hx = (Math.abs(x) - 0.08) / 1.34;
            if (Math.abs(x) > 1.4 || y > 2.95 || Math.abs(x) < 0.09) c = WHITE.slice();
            else if (Math.abs(hx - fy) < 0.085 || Math.abs(hx - (1 - fy)) < 0.085) c = WHITE.slice();   // X braces
          }
          c = win(c, x, y, 4.2, 5.45, 0.68);                                     // hayloft window
        } else { c = win(c, x - 1.7, y, 1.4, 2.6, 0.55); c = win(c, x + 1.7, y, 1.4, 2.6, 0.55); c = win(c, x, y, 4.3, 5.3, 0.55); }
        return [x, y, zc, c[0], c[1], c[2]];
      });
    }
    gable('front', L, true); gable('back', -L, false);
    function side(name, xc, dir) {
      chart(C, name, { nu: 72, nv: 20, tu: 8, tv: 3, closed: true, ref: inside }, function (s, t) {
        var z = (s * 2 - 1) * L * dir, y = t * 3.62, c = planks(z + 20 * dir, y);
        if (Math.abs(z) > L - 0.24 || y > 3.36) c = WHITE.slice();
        c = win(c, z - 3, y, 1.4, 2.6, 0.55); c = win(c, z, y, 1.4, 2.6, 0.55); c = win(c, z + 3, y, 1.4, 2.6, 0.55);
        return [xc, y, z, c[0], c[1], c[2]];
      });
    }
    side('side L', -W, 1); side('side R', W, -1);
    chart(C, 'roof', { nu: 34, nv: 26, tu: 4, tv: 5, closed: true, ref: inside }, function (s, t) {
      var x = (s * 2 - 1) * 3.9, z = (t * 2 - 1) * (L + 0.45), a = Math.abs(x);
      var y = (a > 3.5 ? 3.6 - (a - 3.5) * 1.67 : roofY(x)) + 0.07;
      var c = mul([0.40, 0.43, 0.46], 0.72 + 0.4 * h2(Math.floor(z / 0.62), 5.5));     // metal sheets
      c = mix(c, [0.46, 0.27, 0.16], ss(0.52, 0.72, fbm(x * 0.9 + 3, z * 0.5)) * 0.85); // rust
      return [x, y, z, c[0], c[1], c[2]];
    });
    revolve(C, 'silo', { c: [5.7, 0, -1.9], nu: 18, tu: 4, tv: 4, minA: 5,
      prof: [[1.3, 0], [1.3, 1.2], [1.3, 2.4], [1.3, 3.6], [1.3, 4.8], [1.3, 5.6], [1.24, 6.0], [0.95, 6.5], [0.4, 6.9], [0, 7.0]] },
      function (r, h, a) {
        if (h > 5.7) return mul([0.33, 0.35, 0.38], 0.8 + 0.3 * vn(a * 3, h));
        var c = mul([0.63, 0.62, 0.57], 0.78 + 0.3 * fbm(a * 2.2, h * 0.7));
        return mul(c, fract(h / 1.2) < 0.12 ? 0.7 : 1);
      });
    chart(C, 'treeline', { nu: 80, nv: 9, periodicU: true, tu: 16, tv: 1, cu: 3, cv: 5 }, function (s, t) {
      var a = s * TAU, top = 3.5 + 4.5 * pfbm(s, 22), r = 21 + 1.6 * Math.sin(t * 3.1416) * (0.4 + pfbm(s, 30, t * 3));
      var c = mul(mix([0.07, 0.15, 0.07], [0.19, 0.30, 0.11], pfbm(s, 40, t * 4)), 0.55 + 0.6 * t);
      return [r * Math.sin(a), t * top, r * Math.cos(a), c[0], c[1], c[2]];
    });
    return {
      charts: C,
      views: { wide: { th: 0.62, ph: 1.30, r: 23, tx: 1.0, ty: 2.6, tz: 0 }, close: { th: 0.16, ph: 1.36, r: 4.6, tx: -0.2, ty: 2.5, tz: 5 },
               a: { th: -0.75, ph: 1.22, r: 15, tx: 0, ty: 3.2, tz: 0 }, b: { th: 2.1, ph: 1.18, r: 13, tx: 3.2, ty: 3.4, tz: -1.5 } },
      tour: ['wide', 'close', 'a', 'b'],
      rig: ring(30, 15, 1.6, 6.5, [0.8, 2.6, 0]),
      edit: function (k) {                         // raise the roof
        var c = C[5], P = c.patch.P, B = c.base, i;
        for (i = 0; i < P.length; i += 3) { var w = ss(3.4, 6.6, B[i + 1]); P[i + 1] = B[i + 1] + 1.7 * k * w; P[i] = B[i] * (1 + 0.1 * k * w); }
        return [c];
      }
    };
  }

  /* ================= TRUCK — Tanks and Temples style ================= */
  function truck() {
    var C = [], boxes = [[-2.7, 0.3, -0.95, 2.7, 1.3, 0.95], [-0.45, 0.3, -0.9, 0.9, 1.9, 0.9]];
    var X0 = -2.75, X1 = 2.75, WX = 1.65, WR = 0.44;
    function topY(x) {
      if (x < -0.62) return 1.30; if (x < -0.42) return 1.30 + (x + 0.62) / 0.2 * 0.62;
      if (x < 0.72) return 1.92; if (x < 1.3) return 1.92 - (x - 0.72) / 0.58 * 0.62;
      if (x < 2.5) return 1.30 - (x - 1.3) / 1.2 * 0.08; return 1.22 - (x - 2.5) / 0.25 * 0.14;
    }
    var PAINT = [0.13, 0.42, 0.45], PAINT2 = [0.25, 0.55, 0.54], RUST = [0.42, 0.22, 0.11], GLASS = [0.07, 0.10, 0.13], CHROME = [0.64, 0.66, 0.68];
    chart(C, 'street', { nu: 50, nv: 50, tu: 6, tv: 6 }, function (s, t) {
      var x = s * 30 - 15, z = t * 30 - 15, walk = ss(-3.7, -4.0, z), y = 0.16 * walk;
      var c = mul([0.21, 0.21, 0.22], 0.8 + 0.45 * fbm(x * 0.7, z * 0.7));
      c = mix(c, [0.74, 0.60, 0.14], ss(0.34, 0.08, Math.abs(z - 3.6)) * 0.8);                               // lane line
      c = mix(c, mul([0.47, 0.46, 0.43], 0.85 + 0.3 * vn(x * 0.9, z * 2)), walk);
      c = mix(c, [0.08, 0.08, 0.09], 0.6 * gs(x, z, 0.6, 0.2, 0.5, 0.35));                                    // oil stain
      c = mul(c, lit([x, y + 0.02, z], boxes));
      return [x, y, z, c[0], c[1], c[2]];
    });
    // the body: closed cross-sections lofted from tail to nose
    var BN = 52, BM = 40;
    chart(C, 'body', { nu: BM, nv: BN, periodicU: true, closed: true, tu: 5, tv: 8, minA: 4,
      ref: function (u, v, pos) { return [pos[0], 1.0, 0]; } }, function (s, t, i, j) {
      // rounded end caps: the last three rows at each end close the section down to a pole
      var e = Math.min(j, BN - 1 - j), pole = e === 0, shrink = e === 0 ? 0.02 : e === 1 ? 0.62 : e === 2 ? 0.93 : 1;
      var inset = e === 0 ? 0 : e === 1 ? 0.015 : e === 2 ? 0.07 : 0, mid = Math.min(Math.max(j, 3), BN - 4);
      var x = e < 3 ? (j < 3 ? X0 + inset : X1 - inset) : X0 + 0.16 + (mid - 3) / (BN - 7) * (X1 - X0 - 0.32), yt = topY(x), yb = 0.42, a = s * TAU;
      var zs = Math.sign(Math.cos(a)) * Math.pow(Math.abs(Math.cos(a)), 0.3), ys = Math.sign(Math.sin(a)) * Math.pow(Math.abs(Math.sin(a)), 0.3);
      var y = yb + (ys + 1) / 2 * (yt - yb), w = 0.95 * (y > 1.36 ? 1 - 0.17 * (y - 1.36) / 0.56 : 1);
      w *= 1 + 0.035 * (gs(x, 0, WX, 0, 0.5, 9) + gs(x, 0, -WX, 0, 0.5, 9));                                   // fenders
      var z = w * zs;
      if (shrink < 1) { y = (yb + yt) / 2 + (y - (yb + yt) / 2) * shrink; z *= shrink; }
      var c = mix(PAINT, PAINT2, fbm(x * 1.4, y * 3 + z));
      c = mix(c, RUST, ss(0.9, 0.5, y) * ss(0.35, 0.7, fbm(x * 2.2 + 4, z * 2 + y * 3)) * 0.9);
      var sideF = Math.abs(zs) > 0.82, cab = x > -0.5 && x < 1.25;
      if (cab && y > 1.40 && y < 1.82) {
        if (sideF && x > -0.36 && x < 1.22 - (y - 1.4) * 1.0 && Math.abs(x - 0.2) > 0.07) c = GLASS.slice();    // side windows + B pillar
        if (!sideF && x > 0.72 && Math.abs(z) < 0.7) c = mix(GLASS, [0.16, 0.22, 0.28], (y - 1.4) / 0.42);     // windscreen
      }
      if (x < -0.62 && ys > 0.86) c = [0.09, 0.10, 0.11];                                                       // bed cover
      if (sideF && y < 1.05) { var dw = Math.min(Math.hypot(x - WX, y - WR), Math.hypot(x + WX, y - WR)); if (dw < 0.6) c = [0.04, 0.04, 0.05]; }
      if (x > 2.58) {                                                                        // the face
        c = mix(PAINT, PAINT2, 0.4);
        if (Math.abs(z) < 0.56 && y > 0.74 && y < 1.08) c = fract(z / 0.16) < 0.5 ? [0.05, 0.05, 0.06] : CHROME.slice();
        if (Math.abs(z) > 0.6 && Math.abs(z) < 0.88 && y > 0.80 && y < 1.07) c = [0.97, 0.93, 0.74];
      }
      if (x < -2.6) { if (Math.abs(z) > 0.66 && y > 0.86 && y < 1.12) c = [0.72, 0.08, 0.07]; }                 // tail lights
      if ((x > 2.42 || x < -2.5) && y < 0.68) c = CHROME.slice();                                               // bumpers
      return [x, y, z, c[0], c[1], c[2]];
    });
    var TIRE = [[0, -0.13], [0.2, -0.14], [0.26, -0.12], [0.30, -0.155], [0.41, -0.145], [WR, 0], [0.41, 0.145], [0.30, 0.155], [0.26, 0.12], [0.2, 0.14], [0, 0.13]];
    [[WX, 0.8, 'wheel FL'], [WX, -0.8, 'wheel FR'], [-WX, 0.8, 'wheel RL'], [-WX, -0.8, 'wheel RR']].forEach(function (wl) {
      revolve(C, wl[2], { c: [wl[0], WR, wl[1]], axis: 'z', nu: 16, prof: TIRE, tu: 4, tv: 2, cu: 3, cv: 5, minA: 4 },
        function (r, h, a) { return r < 0.275 ? mul([0.62, 0.62, 0.60], 0.7 + 0.4 * (fract(a / TAU * 5) < 0.5 ? 1 : 0.55)) : [0.055, 0.055, 0.06]; });
    });
    chart(C, 'wall', { nu: 84, nv: 24, tu: 8, tv: 3, closed: true, ref: function () { return [0, 2, -12]; } }, function (s, t) {
      var x = s * 26 - 13, y = t * 5.6, c = mul([0.50, 0.24, 0.17], 0.72 + 0.4 * h2(Math.floor(y / 0.42), Math.floor((x + (Math.floor(y / 0.42) % 2) * 0.6) / 1.2)));
      c = mul(c, 0.8 + 0.3 * fbm(x * 0.6, y * 1.2));
      var tag = fbm(x * 0.35 + 11, y * 0.5 + 3);                                 // paint on the wall, in the talk's own hues
      if (y > 0.5 && y < 3.6 && x > -9 && x < 4) {
        if (tag > 0.60) c = mix(c, [0.31, 0.76, 0.97], 0.5); else if (tag > 0.55) c = mix(c, [0.81, 0.58, 0.85], 0.5); else if (tag < 0.34) c = mix(c, [1.0, 0.65, 0.15], 0.45);
      }
      if (x > 6.2 && x < 10.4 && y < 3.7) c = mul([0.42, 0.44, 0.46], fract(y / 0.3) < 0.2 ? 0.6 : 0.95);      // roller door
      return [x, y, -6.6, c[0], c[1], c[2]];
    });
    return {
      charts: C,
      views: { wide: { th: 0.72, ph: 1.30, r: 11.5, tx: 0, ty: 0.9, tz: 0 }, close: { th: 0.25, ph: 1.38, r: 2.5, tx: 1.75, ty: 0.75, tz: 0.9 },
               a: { th: 1.45, ph: 1.25, r: 6.2, tx: 0.6, ty: 1.2, tz: 0 }, b: { th: -0.5, ph: 1.15, r: 8.2, tx: -0.4, ty: 1.0, tz: 0 } },
      tour: ['wide', 'close', 'a', 'b'],
      rig: ring(28, 7.6, 0.8, 3.4, [0, 1.0, 0]),
      edit: function (k) {                         // stretch the bed
        var c = C[1], P = c.patch.P, B = c.base, i;
        for (i = 0; i < P.length; i += 3) P[i] = B[i] - 1.3 * k * ss(-0.5, -2.6, B[i]);
        var out = [c];
        [4, 5].forEach(function (n) { var w = C[n], Q = w.patch.P, WB = w.base; for (i = 0; i < Q.length; i += 3) Q[i] = WB[i] - 0.95 * k; out.push(w); });
        return out;
      }
    };
  }

  /* A measured face for the bust (talk/face-data.js). The face mesh is set into the head's own coordinates
     (its width fills the skull's, its top meets the forehead), then every control point of the head asks,
     along a horizontal ray from the head's axis, how far out the face is in its direction. Returns, per
     control point, that radius r, a weight w (1 well inside the face, easing to 0 at its rim) and the
     height of the nose tip. */
  function faceFit(F, HN, HM, head) {
    var V = F.v, Tn = F.t, n = V.length / 3, k = 0.80 / 7.743, top = 0, i, j, a, m;
    for (i = 1; i < n; i++) if (Math.abs(V[i * 3]) < 0.01 && V[i * 3 + 1] > V[top * 3 + 1]) top = i;      // top of the forehead, on the centre line
    var y0 = 0.70 - V[top * 3 + 1] * k, z0 = 0.92 * Math.sqrt(1 - 0.70 * 0.70) - V[top * 3 + 2] * k;
    var P = new Float32Array(n * 3); for (i = 0; i < n; i++) { P[i * 3] = V[i * 3] * k; P[i * 3 + 1] = y0 + V[i * 3 + 1] * k; P[i * 3 + 2] = z0 + V[i * 3 + 2] * k; }
    var r = new Float32Array(HN * HM), hit = new Float32Array(HN * HM), w = new Float32Array(HN * HM);
    for (j = 0; j < HM; j++) for (i = 0; i < HN; i++) {
      var th = i / HN * TAU, h = head(th, Math.max(j / (HM - 1), 0.012), true), dx = Math.sin(th), dz = Math.cos(th), best = 0;
      for (a = 0; a < Tn.length; a += 3) {                  // ray (0, h.y, 0) + s (dx, 0, dz) against each triangle that spans this height
        var A = Tn[a] * 3, B = Tn[a + 1] * 3, Cc = Tn[a + 2] * 3, ya = P[A + 1], yb = P[B + 1], yc = P[Cc + 1];
        if ((ya < h.y && yb < h.y && yc < h.y) || (ya > h.y && yb > h.y && yc > h.y)) continue;
        var e1x = P[B] - P[A], e1y = yb - ya, e1z = P[B + 2] - P[A + 2], e2x = P[Cc] - P[A], e2y = yc - ya, e2z = P[Cc + 2] - P[A + 2];
        var px = -dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y, det = e1x * px + e1y * py + e1z * pz; if (Math.abs(det) < 1e-12) continue;
        var tx = -P[A], ty = h.y - ya, tz = -P[A + 2], u = (tx * px + ty * py + tz * pz) / det; if (u < 0 || u > 1) continue;
        var qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x, v = (dx * qx + dz * qz) / det; if (v < 0 || u + v > 1) continue;
        var sd = (e2x * qx + e2y * qy + e2z * qz) / det; if (sd > best) best = sd;
      }
      r[j * HN + i] = best; hit[j * HN + i] = best > 0 ? 1 : 0;
    }
    for (m = 0; m < 3; m++) {                               // soften the rim of the hit region: three small blurs
      var nx = new Float32Array(HN * HM);
      for (j = 0; j < HM; j++) for (i = 0; i < HN; i++) {
        var sum = 0, c = 0, di, dj;
        for (dj = -1; dj <= 1; dj++) for (di = -1; di <= 1; di++) { var jj = j + dj; if (jj < 0 || jj >= HM) continue; sum += hit[jj * HN + (i + di + HN) % HN]; c++; }
        nx[j * HN + i] = sum / c;
      }
      hit = nx;
    }
    for (i = 0; i < HN * HM; i++) w[i] = r[i] > 0 ? ss(0.5, 0.95, hit[i]) : 0;
    return { r: r, w: w, noseY: y0 + V[4 * 3 + 1] * k };
  }

  /* ================= BUST — DTU style ================= */
  function bust() {
    var C = [], HS = 0.74, HY = 2.86, boxes = [[-0.95, 0, -0.5, 0.95, 2.1, 0.5], [-0.55, 2.1, -0.6, 0.55, 3.55, 0.6]];
    var PLASTER = [0.86, 0.82, 0.75];
    function head(theta, t, plain) {               // head-local point + how far it was pushed out (plain: skull, ears and hair only)
      var sph = t < 0.82, phi = sph ? t / 0.82 * 2.62 : 2.62, sx = Math.sin(theta), cz = Math.cos(theta);
      var x = 0.80 * Math.sin(phi) * sx, y = Math.cos(phi), z = 0.92 * Math.sin(phi) * cz;
      if (!sph) { var q = (t - 0.82) / 0.18, r0 = Math.sin(2.62); y = Math.cos(2.62) - q * 0.72; var rn = KT.lerp(r0, 0.6, ss(0, 0.6, q)); x = 0.80 * rn * sx; z = 0.92 * rn * cz - 0.1 * q; }
      x *= 1 - 0.24 * ss(-0.2, -0.9, y); if (z < 0) z *= 1.1;
      var f = plain ? 0 : ss(0.0, 0.3, cz), d = 0;
      d += 0.27 * gs(x, y, 0, -0.10, 0.085, 0.21) + 0.15 * gs(x, y, 0, -0.24, 0.12, 0.085);                 // nose
      d += 0.07 * gs(x, y, 0, 0.27, 0.44, 0.07);                                                              // brow
      d -= 0.11 * (gs(x, y, 0.30, 0.10, 0.13, 0.085) + gs(x, y, -0.30, 0.10, 0.13, 0.085));                   // eye sockets
      d += 0.055 * (gs(x, y, 0.30, 0.09, 0.085, 0.05) + gs(x, y, -0.30, 0.09, 0.085, 0.05));                  // eyes
      d += 0.05 * (gs(x, y, 0.47, -0.16, 0.14, 0.13) + gs(x, y, -0.47, -0.16, 0.14, 0.13));                   // cheekbones
      d += 0.065 * gs(x, y, 0, -0.47, 0.19, 0.04) + 0.07 * gs(x, y, 0, -0.57, 0.17, 0.045);                   // lips
      d -= 0.035 * gs(x, y, 0, -0.67, 0.2, 0.04); d += 0.10 * gs(x, y, 0, -0.82, 0.17, 0.12);                 // chin
      d *= f;
      var ear = 0.15 * Math.exp(-0.5 * ((z + 0.06) * (z + 0.06) / 0.0064 + (y - 0.03) * (y - 0.03) / 0.03)) * ss(0.55, 0.8, Math.abs(sx));
      var hair = 0.045 * ss(0.15, 0.45, y - 0.9 * z + 0.25) * (0.6 + 0.8 * fbm(theta * 5, t * 14));           // a sculpted cap of hair
      var l = Math.hypot(x, z) || 1e-6, out = d + ear + hair, nose = 0.27 * gs(x, y, 0, -0.10, 0.085, 0.21) * f;
      if (plain) return { x: x, y: y + (sph ? 0.25 * hair : 0), z: z, d: 0, nose: 0, ear: ear, hair: hair };
      return { x: x + x / l * (ear + hair) * 0.9 + 0.15 * d * sx, y: y + (sph ? 0.25 * hair : 0), z: z + z / l * hair * 0.9 + d * (cz > 0 ? 1 : 0), d: d, nose: nose };
    }
    var F = KT.face, HN = F ? 104 : 76, HM = F ? 80 : 60, noseW = new Float32Array(HN * HM), fit = F && faceFit(F, HN, HM, head);
    chart(C, 'head', { nu: HN, nv: HM, periodicU: true, closed: true, tu: 8, tv: 6, minA: 3, ref: function () { return [0, HY, 0]; } }, function (s, t, i, j) {
      var h = head(s * TAU, Math.max(t, 0.012), !!fit), k = j * HN + i, c;
      if (fit) {                                   // the face: the measured radius where the mesh was hit, eased into the skull around it
        var w = fit.w[k], l = Math.hypot(h.x, h.z) || 1e-6, r = l + (fit.r[k] - l) * w, extra = (1 - w) * (h.ear + h.hair) * 0.9, sn = Math.sin(s * TAU), cs = Math.cos(s * TAU);
        h.x = sn * (r + extra); h.z = cs * (r + extra); h.d = (fit.r[k] - l) * w;
        noseW[k] = w * gs(h.x, h.y, 0, fit.noseY, 0.075, 0.13) * ss(0.2, 0.7, cs);
        c = mul(PLASTER, (0.93 + 0.1 * fbm(s * 9, t * 9)) * (1 + 0.9 * Math.min(0, h.d)) * (0.84 + 0.16 * ss(-1.6, -0.6, h.y)));
      } else {
        c = mul(PLASTER, (0.93 + 0.1 * fbm(s * 9, t * 9)) * (1 + 2.2 * Math.min(0, h.d)) * (0.84 + 0.16 * ss(-1.6, -0.6, h.y)));
        c = mul(c, 1 - 0.22 * gs(h.x, h.y, 0, -0.52, 0.17, 0.014) * ss(0, 0.3, Math.cos(s * TAU)));
        noseW[k] = h.nose;
      }
      return [h.x * HS, HY + h.y * HS, h.z * HS, c[0], c[1], c[2]];
    });
    revolve(C, 'shoulders', { c: [0, 0, -0.06], nu: 32, tu: 4, tv: 3, minA: 5,
      squash: function (j) { var w = ss(1, 5, j) * (1 - ss(7, 10, j) * 0.7); return [1 + 0.62 * w, 1 - 0.22 * w]; },   // round at the neck, broad at the shoulders
      prof: [[0.26, 2.10], [0.30, 1.98], [0.40, 1.86], [0.56, 1.74], [0.76, 1.62], [0.88, 1.42], [0.90, 1.1], [0.84, 0.72], [0.70, 0.44], [0.62, 0.3], [0.62, 0.12], [0.64, 0]] },
      function (r, h, a) { return mul(PLASTER, (0.9 + 0.14 * fbm(a * 2, h * 3)) * (0.8 + 0.2 * ss(0, 1.2, h))); });
    chart(C, 'table', { nu: 40, nv: 40, tu: 5, tv: 5 }, function (s, t) {
      var x = s * 11 - 5.5, z = t * 11 - 5.5, c = mul([0.36, 0.23, 0.13], 0.7 + 0.5 * h2(Math.floor(z / 0.55), 1.7));
      c = mul(c, (0.82 + 0.3 * fbm(x * 0.5, z * 3)) * (1 - 0.5 * ss(3, 6.5, Math.hypot(x, z))));
      c = mul(c, lit([x, 0.02, z], boxes));
      return [x, 0, z, c[0], c[1], c[2]];
    });
    chart(C, 'backdrop', { nu: 20, nv: 10, tu: 4, tv: 2, closed: true, ref: function () { return [0, 3, -12]; } }, function (s, t) {
      var x = s * 14 - 7, c = mul([0.20, 0.22, 0.26], 0.5 + 0.7 * (1 - t) + 0.1 * fbm(x * 0.4, t * 3));
      return [x, t * 7.4, -4.9 + 0.5 * Math.cos(s * 3.1416 * 2), c[0], c[1], c[2]];      // a gently curved studio backdrop
    });
    var dome = [], a, b;                            // DTU: a patch of a camera dome in front of the object
    for (b = 0; b < 5; b++) for (a = 0; a < 9; a++) {
      var az = (a - 4) / 4 * 0.95, el = 0.18 + b * 0.19, R = 7.2;
      dome.push([R * Math.cos(el) * Math.sin(az), 2.2 + R * Math.sin(el) * 0.75, R * Math.cos(el) * Math.cos(az)]);
    }
    return {
      charts: C,
      views: { wide: { th: 0.42, ph: 1.32, r: 9.6, tx: 0, ty: 2.1, tz: 0 }, close: { th: 0.5, ph: 1.42, r: 2.9, tx: 0, ty: 2.82, tz: 0.25 },
               a: { th: 1.5, ph: 1.4, r: 4.6, tx: 0, ty: 2.7, tz: 0.1 }, b: { th: -0.45, ph: 1.2, r: 6.2, tx: 0, ty: 2.3, tz: 0 } },
      tour: ['wide', 'close', 'a', 'b'],
      rig: { cams: dome, target: [0, 2.5, 0] },
      edit: function (k) {                          // every control point that carries the nose, pulled forward
        var c = C[0], P = c.patch.P, B = c.base, i;
        var fw = fit ? 0.85 : 3.2, up = fit ? 0.1 : 0.5;      // the measured face carries weights up to 1, the sculpted one up to 0.27
        for (i = 0; i < noseW.length; i++) { P[i * 3 + 2] = B[i * 3 + 2] + fw * k * noseW[i] * HS; P[i * 3 + 1] = B[i * 3 + 1] + up * k * noseW[i] * HS; }
        return [c];
      }
    };
  }

  /* ================= GARDEN — Mip-NeRF 360 style ================= */
  function garden() {
    var C = [], boxes = [[-1.15, 0.84, -1.15, 1.15, 0.94, 1.15], [-0.5, 0.94, -0.5, 0.5, 2.4, 0.5]], PC = [0, 1.95, 0];
    chart(C, 'ground', { nu: 66, nv: 66, tu: 6, tv: 6 }, function (s, t) {
      var x = s * 27 - 13.5, z = t * 27 - 13.5, d = Math.hypot(x, z), y = d > 5.4 ? 0.25 * (fbm(x * 0.2, z * 0.2) - 0.4) * ss(5.4, 9, d) : 0;
      var tile = (Math.floor(x / 1.5 + 0.5) + Math.floor(z / 1.5 + 0.5)) & 1, c = mul(tile ? [0.58, 0.55, 0.50] : [0.33, 0.33, 0.33], 0.85 + 0.25 * fbm(x * 1.2, z * 1.2));
      c = mix(c, mix([0.18, 0.34, 0.12], [0.36, 0.46, 0.16], fbm(x * 0.5, z * 0.5)), ss(5.0, 5.7, d));
      c = mul(c, lit([x, y + 0.02, z], boxes));
      return [x, y, z, c[0], c[1], c[2]];
    });
    revolve(C, 'table', { c: [0, 0, 0], nu: 28, tu: 4, tv: 4, minA: 5,
      prof: [[0, 0.935], [0.55, 0.94], [1.12, 0.94], [1.2, 0.93], [1.2, 0.87], [1.1, 0.85], [0.5, 0.84], [0.13, 0.80], [0.1, 0.45], [0.12, 0.1], [0.5, 0.05], [0.56, 0], [0, 0]] },
      function (r, h, a, p) {
        var c = mul([0.55, 0.38, 0.22], 0.7 + 0.5 * h2(Math.floor((p[0] + 3) / 0.19), 0.3));
        if (h > 0.9) c = mul(c, lit([p[0], 0.96, p[2]], [boxes[1]]));
        return mul(c, h < 0.86 ? 0.6 : 1);
      });
    revolve(C, 'pot', { c: [0, 0, 0], nu: 16, tu: 4, tv: 2, minA: 4,
      prof: [[0, 0.95], [0.2, 0.95], [0.27, 1.0], [0.33, 1.2], [0.38, 1.4], [0.36, 1.44], [0.3, 1.42], [0.28, 1.36]] },
      function (r, h, a) { return mul([0.72, 0.36, 0.21], 0.8 + 0.3 * fbm(a * 3, h * 6)); });
    chart(C, 'plant', { nu: 40, nv: 26, periodicU: true, closed: true, tu: 5, tv: 4, minA: 4, ref: function () { return PC; } }, function (s, t) {
      var a = s * TAU, ph = Math.max(0.03, Math.min(0.97, t)) * 3.1416, n = pfbm(s, 7, t * 9), r = 0.46 + 0.42 * n;
      var c = mix([0.06, 0.22, 0.07], [0.36, 0.55, 0.16], n * 1.4), fl = h2(Math.floor(s * 14), Math.floor(t * 9));
      if (fl > 0.8 && t > 0.08 && t < 0.8) c = fl > 0.9 ? [0.92, 0.36, 0.62] : [0.98, 0.84, 0.3];   // flowers
      return [PC[0] + r * Math.sin(ph) * Math.cos(a), PC[1] + r * 1.05 * Math.cos(ph), PC[2] + r * Math.sin(ph) * Math.sin(a), c[0], c[1], c[2]];
    });
    chart(C, 'hedge', { nu: 84, nv: 9, periodicU: true, tu: 14, tv: 1, cu: 3, cv: 5 }, function (s, t) {
      var a = s * TAU, top = 2.3 + 1.4 * pfbm(s, 26), r = 12 + 1.3 * Math.sin(t * 3.1416) * (0.5 + pfbm(s, 34, t * 3));
      var c = mul(mix([0.06, 0.17, 0.07], [0.22, 0.36, 0.12], pfbm(s, 50, t * 5)), 0.5 + 0.65 * t);
      return [r * Math.sin(a), t * top, r * Math.cos(a), c[0], c[1], c[2]];
    });
    return {
      charts: C,
      views: { wide: { th: 0.7, ph: 1.22, r: 9.4, tx: 0, ty: 1.1, tz: 0 }, close: { th: 0.3, ph: 1.3, r: 2.7, tx: 0, ty: 1.8, tz: 0 },
               a: { th: 2.2, ph: 1.05, r: 6.4, tx: 0, ty: 1.2, tz: 0 }, b: { th: -1.1, ph: 1.34, r: 12, tx: 0, ty: 1.4, tz: 0 } },
      tour: ['wide', 'close', 'a', 'b'],
      rig: ring(32, 6.2, 1.0, 3.8, [0, 1.2, 0]),
      edit: function (k) {                          // let the plant grow
        var c = C[3], P = c.patch.P, B = c.base, i, g = 1 + 0.8 * k;
        for (i = 0; i < P.length; i += 3) { P[i] = PC[0] + (B[i] - PC[0]) * g; P[i + 1] = PC[1] - 0.5 + (B[i + 1] - PC[1] + 0.5) * (1 + 1.1 * k); P[i + 2] = PC[2] + (B[i + 2] - PC[2]) * g; }
        return [c];
      }
    };
  }

  /* ================= STUDIO — the abstract still life ================= */
  function studio() {
    var C = [], rgb = KT.gl.rgb, PAL = [rgb(KT.color.surf), rgb(KT.color.splat), rgb(KT.color.ctrl)];
    chart(C, 'floor', { nu: 4, nv: 4 }, function (s, t) { var c = mix([0.19, 0.21, 0.27], [0.34, 0.33, 0.35], 0.5 * (s + t)); return [s * 6.4 - 3.2, 0, t * 6.4 - 3.2, c[0], c[1], c[2]]; });
    chart(C, 'wall A', { nu: 4, nv: 4, tu: 4, tv: 2 }, function (s, t) { var c = mix([0.13, 0.17, 0.25], [0.22, 0.27, 0.38], t); return [s * 6.4 - 3.2, t * 3.4, -3.2, c[0], c[1], c[2]]; });
    chart(C, 'wall B', { nu: 4, nv: 4, tu: 4, tv: 2 }, function (s, t) { var c = mix([0.17, 0.15, 0.24], [0.27, 0.23, 0.36], t); return [-3.2, t * 3.4, 3.2 - s * 6.4, c[0], c[1], c[2]]; });
    var VC = [[0.14, 0.30, 0.55], [0.14, 0.30, 0.55], [0.93, 0.87, 0.75], [0.82, 0.38, 0.24], [0.93, 0.87, 0.75], [0.14, 0.30, 0.55], [0.93, 0.87, 0.75], [0.82, 0.38, 0.24], [0.93, 0.87, 0.75]];
    var VP = [[0.30, 0], [0.34, 0.05], [0.52, 0.3], [0.60, 0.62], [0.50, 0.95], [0.31, 1.22], [0.20, 1.42], [0.24, 1.58], [0.31, 1.72]].map(function (q) { return [q[0] * 1.35, q[1] * 1.35]; });
    var vi = 0;
    revolve(C, 'vase', { c: [-1.6, 0, 0.9], nu: 10, prof: VP, tu: 5, tv: 3, cu: 6, cv: 8, minA: 5 }, function () { return VC[Math.floor(vi++ / 10) % 9]; });
    chart(C, 'cloth', { nu: 15, nv: 15, cu: 9, cv: 9 }, function (s, t, i, j) {
      var x = s * 2 - 1, z = t * 2 - 1, L = 1.2, h = 0.55, rho = Math.max(Math.abs(x), Math.abs(z)) || 1e-6;
      var f = KT.clamp((rho * L - h) / (L - h), 0, 1), ang = Math.atan2(z, x);
      var rad = rho * L <= h ? rho * L : h + 0.10 * f + 0.11 * Math.sin(ang * 7) * f;
      var c = mix([0.08, 0.40, 0.45], [0.80, 0.90, 0.86], ((i % 4 === 1 || j % 4 === 1) ? 1 : 0) * 0.8);
      return [1.3 + x / rho * rad, 0.92 * (1 - 0.97 * Math.pow(f, 0.8)) + 0.03, 0.6 + z / rho * rad, c[0], c[1], c[2]];
    });
    var TUBE = 0.105, KS = 0.34, KY = 1.82;
    function knotC(u) {
      var a = u * 2 * TAU, cs = Math.cos(1.5 * a), x = KS * (2 + cs) * Math.cos(a), y = KS * (2 + cs) * Math.sin(a), z = KS * Math.sin(1.5 * a);
      return [x * Math.cos(0.78) + z * Math.sin(0.78), y, -x * Math.sin(0.78) + z * Math.cos(0.78)];
    }
    chart(C, 'knot', { nu: 72, nv: 6, periodicU: true, periodicV: true, closed: true, tu: 36, tv: 1, cu: 4, cv: 12, minB: 14,
      ref: function (u) { var c = knotC(u); return [1.3 + c[0], KY + c[1], 0.6 + c[2]]; } }, function (s, t) {
      var p0 = knotC(s), p1 = knotC(s + 0.002), tx = p1[0] - p0[0], ty = p1[1] - p0[1], tz = p1[2] - p0[2];
      var sx = p1[0] + p0[0], sy = p1[1] + p0[1], sz = p1[2] + p0[2];
      var bx = ty * sz - tz * sy, by = tz * sx - tx * sz, bz = tx * sy - ty * sx, bl = Math.hypot(bx, by, bz); bx /= bl; by /= bl; bz /= bl;
      var nx = by * tz - bz * ty, ny = bz * tx - bx * tz, nz = bx * ty - by * tx, nl = Math.hypot(nx, ny, nz); nx /= nl; ny /= nl; nz /= nl;
      var a = t * TAU, ca = Math.cos(a) * TUBE, sa = Math.sin(a) * TUBE, h = s * 3, k = Math.floor(h) % 3, c = mix(PAL[k], PAL[(k + 1) % 3], h - Math.floor(h));
      return [1.3 + p0[0] + nx * ca + bx * sa, KY + p0[1] + ny * ca + by * sa, 0.6 + p0[2] + nz * ca + bz * sa, c[0], c[1], c[2]];
    });
    return {
      charts: C,
      views: { wide: { th: 0.78, ph: 1.14, r: 11.2, tx: -0.2, ty: 1.2, tz: 0.2 }, close: { th: 1.0, ph: 1.22, r: 1.75, tx: 1.45, ty: 2.25, tz: 0.75 },
               a: { th: 0.3, ph: 1.2, r: 5.0, tx: -1.6, ty: 1.15, tz: 0.9 }, b: { th: 1.5, ph: 1.0, r: 4.6, tx: 1.3, ty: 0.7, tz: 0.6 } },
      tour: ['wide', 'a', 'close', 'b'],
      rig: ring(24, 6.5, 1.0, 4.0, [0, 1.2, 0]),
      edit: function (k) {                          // let the knot breathe
        var c = C[5], Q = c.patch.P, B = c.base, i, s = 1 + 0.32 * (k - 0.5);
        for (i = 0; i < Q.length; i += 3) { Q[i] = 1.3 + (B[i] - 1.3) * s; Q[i + 1] = KY + (B[i + 1] - KY) * s; Q[i + 2] = 0.6 + (B[i + 2] - 0.6) * s; }
        return [c];
      }
    };
  }

  /* ================= BREAKFAST — a light, small scene for the short Hebrew deck ================= */
  function breakfast() {
    var C = [], RD = 0.74, rd = 0.33, CY = 0.05 + rd * 0.94, MX = 1.95, MZ = -0.55;
    var boxes = [[-RD - rd, 0.05, -RD - rd, RD + rd, CY + rd, RD + rd], [MX - 0.54, 0, MZ - 0.54, MX + 0.54, 1.0, MZ + 0.54]];
    function soft(p) { return 0.6 + 0.4 * lit(p, boxes); }
    chart(C, 'table', { nu: 38, nv: 38, tu: 5, tv: 5 }, function (s, t) {
      var x = s * 9 - 4.5, z = t * 9 - 4.5, c = mul([0.91, 0.79, 0.63], 0.88 + 0.2 * h2(Math.floor(z / 0.7), 2.2));
      c = mul(c, (0.93 + 0.12 * fbm(x * 0.6, z * 3)) * soft([x, 0.01, z]));
      return [x, 0, z, c[0], c[1], c[2]];
    });
    revolve(C, 'plate', { c: [0, 0, 0], nu: 30, tu: 4, tv: 2, minA: 5,
      prof: [[0, 0.035], [0.7, 0.035], [1.05, 0.05], [1.3, 0.14], [1.43, 0.2], [1.45, 0.19], [1.3, 0.1], [1.0, 0.0], [0, 0.0]] },
      function (r0, h, a, p) { return mul([0.99, 0.98, 0.95], h > 0.02 ? soft([p[0], p[1] + 0.02, p[2]]) : 0.82); });
    chart(C, 'doughnut', { nu: 76, nv: 28, periodicU: true, periodicV: true, closed: true, tu: 8, tv: 3, minA: 3, minB: 4,
      ref: function (u, v, pos) { var a = Math.atan2(pos[2], pos[0]); return [RD * Math.cos(a), CY, RD * Math.sin(a)]; } }, function (s, t) {
      var u = s * TAU, v = t * TAU, cv = Math.cos(v), sv = Math.sin(v), c = [0.87, 0.67, 0.42];
      if (sv > 0.07 + 0.09 * Math.sin(u * 5 + 0.6) + 0.05 * Math.sin(u * 11)) {
        c = [1.0, 0.54, 0.69];
        var k = h2(Math.floor(s * 26), Math.floor(t * 11));
        if (k > 0.78 && sv > 0.3) c = [[0.36, 0.78, 0.98], [1.0, 0.84, 0.04], [1, 1, 1], [0.4, 0.83, 0.64], [0.75, 0.47, 1.0]][Math.floor(h2(Math.floor(t * 11), Math.floor(s * 26)) * 5)];
      }
      return [(RD + rd * cv) * Math.cos(u), CY + rd * sv, (RD + rd * cv) * Math.sin(u), c[0], c[1], c[2]];
    });
    revolve(C, 'mug', { c: [MX, 0, MZ], nu: 24, tu: 4, tv: 3, minA: 5,
      prof: [[0, 0], [0.46, 0], [0.52, 0.04], [0.54, 0.5], [0.54, 0.98], [0.52, 1.02], [0.48, 1.0], [0.47, 0.88], [0.3, 0.86], [0, 0.86]] },
      function (r0, h) { return r0 < 0.475 && h > 0.8 && h < 0.95 ? [0.33, 0.19, 0.11] : mul([0.68, 0.81, 0.93], 0.92 + 0.1 * h); });
    return {
      charts: C,
      views: { wide: { th: 0.75, ph: 1.0, r: 7.4, tx: 0.7, ty: 0.35, tz: -0.2 }, close: { th: 0.25, ph: 1.12, r: 3.1, tx: 0, ty: 0.38, tz: 0 },
               a: { th: -0.7, ph: 0.82, r: 5.8, tx: 0.6, ty: 0.3, tz: -0.2 }, b: { th: 1.9, ph: 1.08, r: 5.4, tx: 1.0, ty: 0.45, tz: -0.3 } },
      tour: ['wide', 'close', 'a', 'b'],
      rig: ring(20, 4.2, 1.2, 3.2, [0.5, 0.4, 0]),
      edit: function (k) {                          // a fatter doughnut
        var c = C[2], P = c.patch.P, B = c.base, i;
        for (i = 0; i < P.length; i += 3) { var a = Math.atan2(B[i + 2], B[i]), cx = RD * Math.cos(a), cz = RD * Math.sin(a), g = 1 + 0.35 * k;
          P[i] = cx + (B[i] - cx) * g; P[i + 1] = CY + (B[i + 1] - CY) * g; P[i + 2] = cz + (B[i + 2] - cz) * g; }
        return [c];
      }
    };
  }

  /* ================= SCAN — a real DTU object (talk/dtu-data.js, baked by tools/dtu/fit.py) =================
     Not a stand-in and not a result of the thesis method: the scan's depth maps were fused and interpolated by
     cubic B-spline charts offline, and every control colour was taken from the scan's photographs. Without the
     data file this scene does not exist. Charts:
       the object (named by the data file: `owl`) — one open sheet over everything that was measured: the front
                and both sides. It ends where the measurements end, so it is open at the back and on top of the
                head. S_u x S_v points outwards; it is `closed` only to have its back faces culled.
       table  — the fitted table plane, as far as the photographs show it.
     `seen` on a chart (Uint8Array, one per control point) is 1 where the surface was measured and 0 where a
     small hole was inpainted. photo: true — the colours already hold the light of the photographs. */
  /* The owl of scan 122, alive: animate(t, paint, material, from, target, fromTarget) writes control points and control
     colours and returns the charts whose shape changed. Everything is an edit of the chart's own net. Regions are in the
     object's frame (scene units, 1 = 80 mm), read off the fitted surface:
       breathing — the chest moves out and back along the horizontal direction away from the body's axis;
       head      — every point above the shoulders tilts about an axis that points at the cameras (so the
                   sheet's open back never turns into view), now and then;
       blinking  — a lid colour closes over each eye socket from above;
       paint     — 0..1 wipes a MATERIAL down over its TARGET, from the top of what is painted to its foot.
     Breathing, head and blink are the owl's own: they stay on the owl's control points, whatever is painted.
     TARGETS (TARGETS below) say where a material goes, as a weight 0..1 per control point of the object's chart (the
     owl and the books are one sheet; the table is a chart of its own and is never painted):
       object    — every control point of the object: the owl and the books, all that is stone;
       owl       — the owl alone. Its mask (talk/dtu/owlmask122.js) was segmented OFFLINE from the DINOv3 features of
                   the scan's photographs and the geometry of the net (tools/dtu/owlmask.py); its edge is feathered over
                   two control points. Without that file: a line by hand between the owl and the books;
       selection — the control points selected on the Hands atlas (talk/hands.js, c.sel), feathered the same way.
     A material is one entry of MATS. Its colours are a rule, evaluated once per control point of the object;
     nothing is painted by hand or generated. The rule sees a sample s:
       s.x, s.y, s.z     where the point is (its position in the scan);
       s.nx, s.ny, s.nz  the unit normal of the control net there, pointing out of the object (the cameras are at +z);
       s.sh              0..1, how bright the photograph is there: the carving's relief (grooves are dark);
       s.head            whether the point is above the shoulders;
     and writes s.r, s.g, s.b. Optional: `eye(d, lx, ly)` -> [r, g, b] inside an eye socket (d = 0 centre .. 1
     rim, lx / ly = -1..1 across it), `beak` and `lid` colours; `colour` may also set s.a, an opacity (1 = solid:
     the renderer then shows what is behind, and the far side of the owl, through it; eyes and beak stay
     solid); and `move(s, t)` may add to s.x, s.y, s.z every frame: a material that is not rigid. In `move` only,
     s.h is the height of the point above where the painted part stands (the table for `object`, the feet for
     `owl`, the lowest selected point for `selection`) and s.H the height of the painted part: tie the motion to
     s.h and what is painted stays standing. Opacity and motion are scaled in by the wipe and by the target's
     weight. `view` names the camera the material is shown from (default: the step's own): a see-through one
     wants `through`, where the books lie behind the owl.
     KNOBS: `p` holds the material's own properties (opacity, gloss, stiffness, ...) at their defaults, and
     `knobs` says which of them the simulator may turn and how far: [{ k, label, min, max, step }]. `colour` and
     `move` get p as their last argument. Turning a knob that `colour` reads repaints the owl (tune(), below).
     animate(t, paint, material, from, target, fromTarget) wipes `material` on `target` in over `from` on
     `fromTarget` (defaults: `stone`, the photograph, and `object`). */
  var MATS = {
    stone: { label: 'stone' },                                    // the photograph's own colours: nothing is changed
    snowy: { label: 'snowy owl', beak: [0.09, 0.085, 0.08], lid: [0.80, 0.79, 0.76],
      p: { albedo: 0.9, barring: 0.5 },
      knobs: [{ k: 'albedo', label: 'albedo', min: 0.4, max: 1, step: 0.01 }, { k: 'barring', label: 'barring', min: 0, max: 1, step: 0.01 }],
      // white where the stone is bright, dark bars where the carving has grooves (fewer of them on the head)
      colour: function (s, p) {
        var b = 2 * p.barring, t = s.head ? ss(0.06 * b, 0.3 * b + 1e-3, s.sh) : ss(0.2 * b, 0.62 * b + 1e-3, s.sh), w = p.albedo * (0.45 + 0.55 * s.sh);
        s.r = 0.15 + (w - 0.15) * t; s.g = 0.12 + (0.99 * w - 0.12) * t; s.b = 0.10 + (0.955 * w - 0.10) * t;
      },
      eye: function (d) { return d < 0.42 ? [0.02, 0.02, 0.02] : d < 0.86 ? [0.95, 0.70, 0.08] : [0.08, 0.07, 0.06]; } },
    jelly: { label: 'alien jelly', view: 'through', beak: [0.03, 0.40, 0.12], lid: [0.12, 0.72, 0.22],
      p: { opacity: 0.34, gloss: 1, stiffness: 1 },
      knobs: [{ k: 'opacity', label: 'opacity', min: 0.08, max: 1, step: 0.01 }, { k: 'gloss', label: 'gloss', min: 0, max: 2, step: 0.02 },
              { k: 'stiffness', label: 'stiffness', min: 0.25, max: 4, step: 0.05 }],
      // a green gel you can see through: a third opaque where you look straight into it, more where it thins out
      // towards the rim (lime there), the carving's relief smoothed over, and two painted highlights (a lamp up
      // and to the left, a broad sheen from above), which are solid
      colour: function (s, p) {
        var f = Math.pow(1 - Math.max(0, Math.abs(s.nz)), 1.5), w = 0.65 + 0.35 * s.sh;
        var hl = p.gloss * (0.9 * Math.pow(Math.max(0, -0.22 * s.nx + 0.44 * s.ny + 0.87 * s.nz), 30) + 0.25 * Math.pow(Math.max(0, 0.7 * s.ny + 0.7 * s.nz), 10));
        s.r = (0.10 + 0.60 * f) * w + hl; s.g = (0.80 + 0.20 * f) * w + hl; s.b = (0.22 + 0.20 * f) * w + hl;
        s.a = Math.min(1, p.opacity * (1 + 0.76 * f) + 1.4 * hl);
      },
      eye: function (d, lx, ly) { return Math.hypot(lx + 0.32, ly - 0.34) < 0.24 ? [0.95, 1, 0.95] : d < 0.82 ? [0.01, 0.06, 0.02] : [0.45, 1, 0.35]; },
      // it wobbles: a sway that grows with the height above where it stands (s.h: the table, or the owl's feet), and a
      // ripple climbing the body. Like a mass on a spring, a stiffer jelly moves less (1 / stiffness) and faster (the
      // square root of it)
      move: function (s, t, p) {
        var q = Math.sqrt(p.stiffness), h = s.h, a = 0.045 / p.stiffness * Math.pow(h, 1.5) * (0.65 + 0.35 * Math.sin(0.9 * t));
        var rp = 0.016 / p.stiffness * Math.sin(9 * s.y - 7 * q * t);
        s.x += a * Math.sin(5.0 * q * t) + rp * s.nx; s.z += 0.7 * a * Math.sin(4.3 * q * t + 1.3) + rp * s.nz; s.y -= 0.012 / p.stiffness * h * (0.5 + 0.5 * Math.sin(10 * q * t));
      } }
  };
  /* crystal clear glass: almost nothing where you look straight through it, a mirror at the rim. How much of the
     light a surface reflects at a given angle is Schlick's approximation of Fresnel, from the refractive index;
     that reflectance is what makes glass visible, so it sets both the brightness and the opacity. On top: a soft
     box up and to the left as a sharp highlight, a bright "window" on what faces upwards, and the carving's
     grooves as faint lines, the way facets catch the light. All of it painted, none of it traced. */
  MATS.glass = { label: 'clear glass', view: 'through', beak: [0.80, 0.90, 1.0], lid: [0.82, 0.90, 0.97],
    p: { opacity: 0.07, index: 1.5, gloss: 1 },
    knobs: [{ k: 'opacity', label: 'opacity', min: 0.02, max: 0.6, step: 0.01 }, { k: 'index', label: 'refractive index', min: 1.05, max: 2.4, step: 0.01 },
            { k: 'gloss', label: 'gloss', min: 0, max: 2, step: 0.02 }],
    colour: function (s, p) {
      var c = Math.abs(s.nz), f0 = Math.pow((p.index - 1) / (p.index + 1), 2), F = f0 + (1 - f0) * Math.pow(1 - c, 5);
      var hl = p.gloss * Math.pow(Math.max(0, -0.22 * s.nx + 0.44 * s.ny + 0.87 * c), 90);
      var win = p.gloss * 0.22 * ss(0.25, 0.7, 2 * c * s.ny), line = 0.16 * (1 - s.sh) * (1 - s.sh);
      var k = Math.min(1, 6 * F) * 0.55 + win;                       // how much reflected light there is to see
      s.r = 0.62 + 0.38 * k + hl - line; s.g = 0.78 + 0.22 * k + hl - 0.6 * line; s.b = 0.90 + 0.10 * k + hl;
      s.a = Math.min(1, p.opacity + (1 - p.opacity) * Math.min(1, 1.5 * F) + 0.6 * win + line + 1.6 * hl);
    },
    eye: function (d, lx, ly) { return Math.hypot(lx + 0.3, ly - 0.32) < 0.22 ? [1, 1, 1] : d < 0.8 ? [0.04, 0.10, 0.22] : [0.75, 0.88, 1]; } };
  Object.keys(MATS).forEach(function (k) { MATS[k].rev = 0; });
  /* where a material may go; `selection` needs the Hands layer (talk/hands.js), which the host offers or not */
  var TARGETS = [{ v: 'object', label: 'object' }, { v: 'owl', label: 'owl' }, { v: 'selection', label: 'selection' }];
  var FEET = 1.45;                           // the owl stands on the book at this height: its motion is measured from here
  function ss(a, b, v) { v = KT.clamp((v - a) / (b - a), 0, 1); return v * v * (3 - 2 * v); }
  /* a 0..1 weight per control point with a soft edge: two 3 x 3 box blurs over the lattice, edges clamped (the way
     talk/dtu/owlmask122.js feathers the owl when it loads) */
  function feather(src, nu, nv, out) {
    var T = new Float32Array(src.length), i, j, a, q, v, pass;
    out.set(src);
    for (pass = 0; pass < 2; pass++) {
      for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) { for (v = 0, a = -1; a <= 1; a++) v += out[j * nu + Math.min(nu - 1, Math.max(0, i + a))]; T[j * nu + i] = v / 3; }
      for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) { for (v = 0, q = -1; q <= 1; q++) v += T[Math.min(nv - 1, Math.max(0, j + q)) * nu + i]; out[j * nu + i] = v / 3; }
    }
    return out;
  }
  function owlLife(c, OWL) {
    var P = c.patch.P, K = c.patch.C, B = c.base, K0 = new Float32Array(K), n = B.length / 3, nu = c.patch.nu, nv = c.patch.nv;
    var EYES = [[0.127, 3.214], [0.470, 3.262]], RE = 0.092, BEAK = [0.335, 3.02];
    var life = [], LW = new Float32Array(n * 2), eyes = [], N = new Float32Array(n * 3), SH = new Float32Array(n), BK = new Float32Array(n);
    var AL = new Float32Array(n).fill(1);                          // the opacity of every control point, as the renderer reads it (c.alpha)
    var MO = new Float32Array(n), masked = !!(OWL && OWL.length === n), Y0 = 1e9, Y1 = -1e9;
    var i, j, k, o, q, x, y, z, e, d, w, lid = [[0, 0, 0, 0], [0, 0, 0, 0]], s = {};
    c.baseC = K0;
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {            // normals of the net: (next - previous) along u, cross, along v
      var a = (j * nu + Math.max(0, i - 1)) * 3, b = (j * nu + Math.min(nu - 1, i + 1)) * 3, g = (Math.max(0, j - 1) * nu + i) * 3, h = (Math.min(nv - 1, j + 1) * nu + i) * 3;
      var ux = B[b] - B[a], uy = B[b + 1] - B[a + 1], uz = B[b + 2] - B[a + 2], vx = B[h] - B[g], vy = B[h + 1] - B[g + 1], vz = B[h + 2] - B[g + 2];
      x = uy * vz - uz * vy; y = uz * vx - ux * vz; z = ux * vy - uy * vx; d = Math.hypot(x, y, z) || 1; o = (j * nu + i) * 3;
      N[o] = x / d; N[o + 1] = y / d; N[o + 2] = z / d;
    }
    for (i = 0; i < n; i++) {
      o = i * 3; x = B[o]; y = B[o + 1]; z = B[o + 2];
      // the owl alone: the mask segmented offline where its file is loaded, else a line by hand that rises towards the open page
      MO[i] = masked ? OWL[i] : ss(0, 0.07, y - (x < 0.3 ? 1.56 + 0.25 * x : 1.635 + 1.2 * (x - 0.3)));
      if (y < Y0) Y0 = y; if (y > Y1) Y1 = y;
      // the owl's life, on the owl only: breathing (the chest) and the head
      var lb = ss(1.75, 2.2, y) * (1 - ss(2.6, 2.95, y)) * MO[i], lh = ss(2.3, 2.85, y) * MO[i];
      if (lb || lh) { life.push(i); LW[i * 2] = lb; LW[i * 2 + 1] = lh; }
      SH[i] = Math.min(1, (0.30 * K0[o] + 0.59 * K0[o + 1] + 0.11 * K0[o + 2]) / 0.30);
      d = Math.hypot((x - BEAK[0]) / 0.055, (y - BEAK[1]) / 0.09); BK[i] = d < 1.2 && z > 0.3 ? 1 - ss(0.8, 1.2, d) : 0;
      for (e = 0; e < 2; e++) {
        d = Math.hypot(x - EYES[e][0], y - EYES[e][1]) / RE; if (z < 0.2) continue;
        if (d < 1) eyes.push(i, e, (y - EYES[e][1]) / RE);
        else if (d < 1.5) { lid[e][0] += K0[o]; lid[e][1] += K0[o + 1]; lid[e][2] += K0[o + 2]; lid[e][3]++; }   // the brow: the lid's colour in stone
      }
    }
    lid.forEach(function (l) { l[0] /= l[3]; l[1] /= l[3]; l[2] /= l[3]; });
    /* the targets: a weight per control point, how low and how high it reaches, and the height it stands on */
    function extent(t) {
      var W = t.w, lo = 1e9, hi = -1e9, m;
      for (m = 0; m < n; m++) if (W[m] > 0.01) { if (B[m * 3 + 1] < lo) lo = B[m * 3 + 1]; if (B[m * 3 + 1] > hi) hi = B[m * 3 + 1]; }
      t.lo = lo; t.hi = hi; return t;
    }
    var TG = { object: extent({ w: new Float32Array(n).fill(1), v: 0 }), owl: extent({ w: MO, v: 0 }), selection: { w: new Float32Array(n), v: 0, rev: NaN } };
    TG.object.base = Y0; TG.owl.base = FEET;
    function target(name) {
      var t = TG[name] || TG.object;
      if (t === TG.selection) {                    // the Hands layer's selection, feathered again whenever it changed
        var S = c.sel && c.sel.length === n ? c.sel : null, rev = S ? c.selRev || 0 : -1;
        if (t.rev !== rev) { t.rev = rev; if (S) feather(S, nu, nv, t.w); else t.w.fill(0); extent(t); t.base = t.lo; t.v++; }
      }
      return t;
    }
    function sample(i, t) {
      var o = i * 3; s.x = B[o]; s.y = B[o + 1]; s.z = B[o + 2]; s.nx = N[o]; s.ny = N[o + 1]; s.nz = N[o + 2]; s.sh = SH[i]; s.head = B[o + 1] > 2.85;
      s.h = t ? Math.max(0, s.y - t.base) : 0; s.H = t ? Math.max(0, t.hi - t.base) : 0;
      return s;
    }
    /* the colours (and opacities) of a material on every control point of the object, before any target: worked out the
       first time it is asked for, and again after one of its knobs was turned. -> { K, A (or null), rev } */
    var baked = {};
    function bake(name) {
      var m = MATS[name], bk = baked[name], rgb;
      if (bk && bk.rev === m.rev) return bk;
      bk = baked[name] = bk || { K: new Float32Array(K0), A: null }; bk.rev = m.rev;
      if (!m.colour) return bk;
      var K1 = bk.K, A1 = bk.A; if (A1) A1.fill(1);
      for (i = 0; i < n; i++) {
        o = i * 3; s.a = 1; m.colour(sample(i, null), m.p); rgb = [s.r, s.g, s.b];
        if (BK[i] && m.beak) for (k = 0; k < 3; k++) rgb[k] += (m.beak[k] * (0.6 + 0.4 * SH[i]) - rgb[k]) * BK[i];
        K1[o] = rgb[0]; K1[o + 1] = rgb[1]; K1[o + 2] = rgb[2];
        if (s.a < 1) { A1 = A1 || (bk.A = new Float32Array(n).fill(1)); A1[i] = s.a + (1 - s.a) * BK[i]; }
      }
      if (m.eye) for (q = 0; q < eyes.length; q += 3) {
        o = eyes[q] * 3; e = EYES[eyes[q + 1]]; x = (B[o] - e[0]) / RE; y = eyes[q + 2];
        rgb = m.eye(Math.hypot(x, y), x, y); K1[o] = rgb[0]; K1[o + 1] = rgb[1]; K1[o + 2] = rgb[2];
        if (A1) A1[eyes[q]] = 1;
      }
      return bk;
    }
    var painted = -1, shut = -1, key = '', to = null, src = null, TA = null, TB = null, moved = false;
    /* one control point: `from` on its target above the wipe's front, `material` on its target below it; a target's
       weight blends the material with the photograph's own colour */
    function colour(i, front) {
      var o = i * 3, m = ss(front, front + 0.25, B[o + 1]), wa = TA.w[i], wb = TB.w[i], F = src.K, T2 = to.K, f0, t0;
      for (k = 0; k < 3; k++) { f0 = K0[o + k] + (F[o + k] - K0[o + k]) * wa; t0 = K0[o + k] + (T2[o + k] - K0[o + k]) * wb; K[o + k] = f0 + (t0 - f0) * m; }
      if (to.A || src.A) { f0 = src.A ? 1 + (src.A[i] - 1) * wa : 1; t0 = to.A ? 1 + (to.A[i] - 1) * wb : 1; AL[i] = f0 + (t0 - f0) * m; }
      return m;
    }
    function animate(t, paint, material, from, target1, target0) {
      var mat = MATS[material] || MATS.snowy, old = MATS[from] || MATS.stone;
      to = bake(MATS[material] ? material : 'snowy'); src = bake(MATS[from] ? from : 'stone');
      TB = target(target1); TA = target(target0);
      var ky = material + to.rev + ':' + target1 + TB.v + '<' + from + src.rev + ':' + target0 + TA.v; if (ky !== key) { key = ky; painted = -1; }
      // the wipe runs over what is painted, before or after: from its top to its foot
      var lo = 1e9, hi = -1e9;
      [[mat, TB], [old, TA]].forEach(function (l) { if (l[0].colour && l[1].hi >= l[1].lo) { lo = Math.min(lo, l[1].lo); hi = Math.max(hi, l[1].hi); } });
      if (hi < lo) { lo = Y0; hi = Y1; }
      var front = hi - (hi - lo + 0.25) * paint;
      var breath = 0.5 - 0.5 * Math.cos(t * 2 * Math.PI / 3.4), tc = t % 9, tb = t % 3.7;
      var tilt = 0.3 * ss(4, 4.7, tc) * (1 - ss(7, 7.7, tc)), blink = ss(3.0, 3.09, tb) * (1 - ss(3.12, 3.26, tb));
      var ca = Math.cos(tilt), sa = Math.sin(tilt), cw, sw, m;
      // what is not rigid moves: the new material below the front, the old one above it, each on its own target.
      // Then every control point is written (and once more after the motion ends, to put them back); else the owl's life only
      var soft = (paint > 0 && !!mat.move) || (paint < 1 && !!old.move), every = soft || moved, cnt = every ? n : life.length;
      moved = soft;
      for (q = 0; q < cnt; q++) {
        i = every ? q : life[q]; o = i * 3; x = B[o]; y = B[o + 1]; z = B[o + 2];
        if (soft) {
          m = ss(front, front + 0.25, y);
          if (mat.move && (w = TB.w[i] * m) > 0) { sample(i, TB); mat.move(s, t, mat.p); x += (s.x - B[o]) * w; y += (s.y - B[o + 1]) * w; z += (s.z - B[o + 2]) * w; }
          if (old.move && (w = TA.w[i] * (1 - m)) > 0) { sample(i, TA); old.move(s, t, old.p); x += (s.x - B[o]) * w; y += (s.y - B[o + 1]) * w; z += (s.z - B[o + 2]) * w; }
        }
        w = LW[i * 2];
        if (w) { var dx = B[o] - (-0.08 + 0.26 * (B[o + 1] - 1.7)), dz = B[o + 2] - 0.18, dl = 0.045 * breath * w / (Math.hypot(dx, dz) || 1); x += dx * dl; z += dz * dl; }
        w = LW[i * 2 + 1];
        if (w) {
          y += 0.012 * breath * w;
          if (tilt) { if (w < 1) { cw = Math.cos(tilt * w); sw = Math.sin(tilt * w); } else { cw = ca; sw = sa; }
            dx = x - 0.24; var dy = y - 2.45; x = 0.24 + dx * cw - dy * sw; y = 2.45 + dx * sw + dy * cw; }
        }
        P[o] = x; P[o + 1] = y; P[o + 2] = z;
      }
      if (paint !== painted) for (painted = paint, shut = -1, i = 0; i < n; i++) colour(i, front);
      c.alpha = (to.A && paint > 0) || (src.A && paint < 1) ? AL : null;       // see-through somewhere, right now
      if (blink !== shut) for (shut = blink, q = 0; q < eyes.length; q += 3) {      // the lid comes down from above
        i = eyes[q]; o = i * 3; m = colour(i, front);
        if (eyes[q + 2] > 1 - 2 * blink) {
          var l = lid[eyes[q + 1]], L0 = old.lid || l, L1 = mat.lid || l, wa = TA.w[i], wb = TB.w[i];
          for (k = 0; k < 3; k++) { var f0 = l[k] + (L0[k] - l[k]) * wa, t0 = l[k] + (L1[k] - l[k]) * wb; K[o + k] = f0 + (t0 - f0) * m; }
        }
      }
      return [c];
    }
    /* for tests and captions: whether the owl's mask is the one segmented offline, and a target's weights */
    animate.masked = masked;
    animate.weights = function (name) { return target(name).w; };
    return animate;
  }

  function scan(D) {
    var C = [];
    D.charts.forEach(function (d) {
      var P = d.P, K = d.C, nu = d.nu, o = d.plane ? { nu: nu, nv: d.nv, tu: 5, tv: 4 }
                                               : { nu: nu, nv: d.nv, tu: 9, tv: 12, cu: 4, cv: 4, closed: true };
      var c = chart(C, d.name, o, function (s, t, i, j) { var k = (j * nu + i) * 3; return [P[k], P[k + 1], P[k + 2], K[k], K[k + 1], K[k + 2]]; });
      c.seen = d.seen; c.ground = !!d.plane;                      // the table is ground: tools leave it alone, and it can be put away
      // features of a pretrained model baked onto the control points, when their file (talk/dtu/features<N>.js) is loaded
      var F = KT.dtuFeatures && KT.dtuFeatures[D.scan], fc = F && F.charts[d.name];
      c.feat = fc ? { dim: F.dim, scale: F.scale, lo: F.lo, hi: F.hi, nu: fc.nu, nv: fc.nv, step: fc.step || 1, data: fc.data } : null;
    });
    var T = D.target, ty = T[1], tz = T[2], M = KT.dtuMasks && KT.dtuMasks[D.scan];
    return {
      charts: C, photo: true,
      /* every view lies inside the dome of real cameras (azimuth -60..60 degrees, 16..70 degrees above the table);
         a scan baked with `--register scans` brings its own */
      views: D.views || { wide: { th: 0.30, ph: 1.22, r: 8.6, tx: 0, ty: ty, tz: tz }, close: { th: -0.15, ph: 1.20, r: 3.3, tx: 0.15, ty: ty * 1.5, tz: 0.2 },
               a: { th: -0.85, ph: 1.12, r: 6.8, tx: 0, ty: ty, tz: tz }, b: { th: 0.80, ph: 1.00, r: 6.8, tx: 0, ty: ty, tz: tz },
               through: { th: -0.55, ph: 1.14, r: 5.0, tx: 0.1, ty: 2.3, tz: 0.2 },        // close, three-quarter: the books behind the body
               near: { th: 0.30, ph: 1.2, r: 5.8, tx: 0.05, ty: 2.2, tz: 0.2 } },             // the owl and its books fill the frame, from the front
      tour: D.tour || ['wide', 'close', 'a', 'b'],
      rig: { cams: D.cams, target: T },              // the scan's own camera positions, in the object's frame
      /* scan 122 only: the owl alive and repainted (owlLife), on a target; another scan has none of it. The owl's mask
         (talk/dtu/owlmask122.js, segmented offline) is used where it is loaded and was made for this chart */
      animate: D.scan === 122 ? owlLife(C[0], M && M.chart === C[0].name && M.nu === C[0].patch.nu && M.nv === C[0].patch.nv ? M.mask : null) : null,
      materials: D.scan === 122 ? Object.keys(MATS).map(function (k) { return { v: k, label: MATS[k].label, view: MATS[k].view, knobs: MATS[k].knobs || [], p: MATS[k].p }; }) : null,
      targets: D.scan === 122 ? TARGETS.slice() : null,
      /* turn a knob of a material: its value changes for every owl on the page, and they repaint */
      tune: function (v, k, x) { var m = MATS[v]; if (m && m.p && m.p[k] !== x) { m.p[k] = x; m.rev++; } }
    };
  }

  /* ================= DUO — the talk's cover: the owl and the rabbit side by side =================
     Both real DTU scans (talk/dtu-data.js, talk/dtu/scan55.js), their objects only, without the tables. The owl
     stays in its own frame, so that its life (owlLife) still finds its eyes and shoulders. The rabbit is brought
     into the owl's units (its 71 mm to the owl's 80 mm a unit, so the two stand at their true relative sizes),
     turned a little towards the owl and set beside it. `box` = the centre and half extents of the pair, for framing.
     Registered only when both data files are loaded. */
  function duo() {
    var A = KT.dtu, B = KT.dtuScans[55], a = scan(A), b = scan(B), k = B.unit_mm / A.unit_mm, yaw = -0.2, DX = 2.95, DZ = -0.35;
    var cy = Math.cos(yaw), sy = Math.sin(yaw), lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    function place(p) { var x = p[0] * k, z = p[2] * k; return [DX + cy * x + sy * z, p[1] * k, DZ - sy * x + cy * z]; }
    var objs = a.charts.filter(function (c) { return !c.ground; }), rab = b.charts.filter(function (c) { return !c.ground; });
    rab.forEach(function (c, i) {
      var P = c.patch.P, j, q;
      for (j = 0; j < P.length; j += 3) { q = place([P[j], P[j + 1], P[j + 2]]); P[j] = q[0]; P[j + 1] = q[1]; P[j + 2] = q[2]; }
      c.base.set(P); c.hex = HEX[(objs.length + i) % HEX.length];
    });
    var C = objs.concat(rab);
    C.forEach(function (c) { var P = c.patch.P, j, q; for (j = 0; j < P.length; j += 3) for (q = 0; q < 3; q++) { lo[q] = Math.min(lo[q], P[j + q]); hi[q] = Math.max(hi[q], P[j + q]); } });
    var T = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    return {
      charts: C, photo: true, animate: a.animate,
      box: { c: T, h: [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2] },
      views: { wide: { th: 0.12, ph: 1.24, r: 9.6, tx: T[0], ty: T[1], tz: T[2] }, close: { th: 0.05, ph: 1.2, r: 5.2, tx: T[0] - 0.4, ty: 2.4, tz: T[2] },
               a: { th: -0.3, ph: 1.18, r: 8.8, tx: T[0], ty: T[1], tz: T[2] }, b: { th: 0.42, ph: 1.1, r: 8.8, tx: T[0], ty: T[1], tz: T[2] } },
      tour: ['wide', 'a', 'b'],
      rig: { cams: a.rig.cams.concat(b.rig.cams.map(place)), target: T }
    };
  }

  KT.scenes = {
    barn:   { label: 'Barn',   suite: 'Tanks and Temples style', build: barn },
    truck:  { label: 'Truck',  suite: 'Tanks and Temples style', build: truck },
    bust:   { label: 'Bust',   suite: 'DTU style', build: bust },
    garden: { label: 'Garden', suite: 'Mip-NeRF 360 style', build: garden },
    studio: { label: 'Studio', suite: 'abstract still life', build: studio },
    /* hidden: not offered in the lab's scene picker or on the landing page; the Hebrew deck loads it by name */
    breakfast: { label: 'Breakfast', suite: 'illustration', hidden: true, build: breakfast }
  };
  /* the real object, when its data file was loaded before this one. `note` is what a caption may say about it */
  var NOTE = 'fitted offline, not trained';
  if (KT.dtu) KT.scenes.scan = { label: KT.dtu.label || 'Scan', suite: 'DTU scan ' + KT.dtu.scan, photo: true, note: NOTE,
                                 build: function () { return scan(KT.dtu); } };
  /* further scans (talk/dtu/scan<N>.js), offered by the lab page's scene picker only */
  Object.keys(KT.dtuScans || {}).forEach(function (n) {
    var D = KT.dtuScans[n];
    KT.scenes['scan' + n] = { label: D.label, suite: 'DTU scan ' + n, photo: true, note: NOTE, build: function () { return scan(D); } };
  });
  /* the cover's pair: not offered in the lab page's scene picker */
  if (KT.dtu && KT.dtuScans && KT.dtuScans[55]) KT.scenes.duo = { label: 'Owl and rabbit', suite: 'DTU scans 122 and 55', photo: true, hidden: true, note: NOTE, build: duo };
})();
