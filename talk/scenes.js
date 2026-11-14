/* Knot-a-Surface talk — scene library for the `manifold` module.
   Every scene is nothing but B-spline charts: geometry in the control points, appearance in the
   control features (one colour per control point). The benchmark-style scenes are PROCEDURAL
   STAND-INS modelled after the kind of capture each dataset contains — they are not the datasets.
     barn, truck — Tanks and Temples style (outdoor object, handheld orbit)
     bust        — DTU style (scanned object on a table, camera dome)
     garden      — Mip-NeRF 360 style (object in the middle of an unbounded surround)
     studio      — the abstract still life (vase, cloth, torus knot)
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

  /* ================= BUST — DTU style ================= */
  function bust() {
    var C = [], HS = 0.74, HY = 2.86, boxes = [[-0.95, 0, -0.5, 0.95, 2.1, 0.5], [-0.55, 2.1, -0.6, 0.55, 3.55, 0.6]];
    var PLASTER = [0.86, 0.82, 0.75];
    function head(theta, t) {                      // head-local point + how far it was pushed out
      var sph = t < 0.82, phi = sph ? t / 0.82 * 2.62 : 2.62, sx = Math.sin(theta), cz = Math.cos(theta);
      var x = 0.80 * Math.sin(phi) * sx, y = Math.cos(phi), z = 0.92 * Math.sin(phi) * cz;
      if (!sph) { var q = (t - 0.82) / 0.18, r0 = Math.sin(2.62); y = Math.cos(2.62) - q * 0.72; var rn = KT.lerp(r0, 0.6, ss(0, 0.6, q)); x = 0.80 * rn * sx; z = 0.92 * rn * cz - 0.1 * q; }
      x *= 1 - 0.24 * ss(-0.2, -0.9, y); if (z < 0) z *= 1.1;
      var f = ss(0.0, 0.3, cz), d = 0;
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
      return { x: x + x / l * (ear + hair) * 0.9 + 0.15 * d * sx, y: y + (sph ? 0.25 * hair : 0), z: z + z / l * hair * 0.9 + d * (cz > 0 ? 1 : 0), d: d, nose: nose };
    }
    var HN = 76, HM = 60, noseW = new Float32Array(HN * HM);
    chart(C, 'head', { nu: HN, nv: HM, periodicU: true, closed: true, tu: 8, tv: 6, minA: 3, ref: function () { return [0, HY, 0]; } }, function (s, t, i, j) {
      var h = head(s * TAU, Math.max(t, 0.012)), c = mul(PLASTER, (0.93 + 0.1 * fbm(s * 9, t * 9)) * (1 + 2.2 * Math.min(0, h.d)) * (0.84 + 0.16 * ss(-1.6, -0.6, h.y)));
      c = mul(c, 1 - 0.22 * gs(h.x, h.y, 0, -0.52, 0.17, 0.014) * ss(0, 0.3, Math.cos(s * TAU)));
      noseW[j * HN + i] = h.nose;
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
        for (i = 0; i < noseW.length; i++) { P[i * 3 + 2] = B[i * 3 + 2] + 3.2 * k * noseW[i] * HS; P[i * 3 + 1] = B[i * 3 + 1] + 0.5 * k * noseW[i] * HS; }
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

  KT.scenes = {
    barn:   { label: 'Barn',   suite: 'Tanks and Temples style', build: barn },
    truck:  { label: 'Truck',  suite: 'Tanks and Temples style', build: truck },
    bust:   { label: 'Bust',   suite: 'DTU style', build: bust },
    garden: { label: 'Garden', suite: 'Mip-NeRF 360 style', build: garden },
    studio: { label: 'Studio', suite: 'abstract still life', build: studio }
  };
})();
