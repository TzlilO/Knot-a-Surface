/* Knot-a-Surface talk — "Free vs. bound" (flatland).

   A real, tiny differentiable-rendering experiment in a 2D world. The scene is a cross-section:
   the "surface" is a curve, cameras are 1D pinhole sensors that each see one strip of pixels.
   Two models are trained with the SAME photometric loss on the SAME training strips, from the
   SAME sparse noisy points, for the SAME number of Adam steps:

     FREE  (3DGS-like)  every Gaussian owns its mean, covariance, colour and opacity.
     BOUND (ours)       one cubic B-spline; every Gaussian is DERIVED from it
                        (mean = S(u), covariance = tangent frame, colour = basis-blended features).

   Nothing below is scripted: the picture is whatever the optimizer does. The one exception is the closing beat,
   scene `handoff`: training stops, everything but the bound curve fades out, and that curve eases onto a fixed
   spline fit of the true curve (HANDOFF), the curve chapter 02 (sims/basis.js) starts from.

   File layout: (1) pure section — scene, renderer, analytic gradients, Adam, metrics. It has no
   DOM access and loads under node (`require('./flatland.js')`) for gradient / convergence tests.
   (2) browser section — the KnotTalk module (canvas drawing, controls, scenes). */
(function (root) {
  'use strict';
  var NODE = typeof module !== 'undefined' && module.exports;
  var NURBS = NODE ? require('../nurbs.js') : root.KnotTalk.nurbs;

  /* =====================================================================================
     1. PURE SECTION
     ===================================================================================== */

  var CFG = {
    L: 1.4,              // the true curve spans x in [-L, L]
    W: 128,              // pixels per 1D camera
    R: 2.4, cy: 0.2,     // cameras sit on an arc of radius R around (0, cy), looking at the centre
    tanHalf: 0.78,       // tan of the half field of view
    thTrain: 42 * Math.PI / 180,   // training cameras are spread over [-thTrain, +thTrain]
    thNovel: 58 * Math.PI / 180,   // the held-out camera may be dragged over [-thNovel, +thNovel]
    G: 64,               // Gaussians per model (free: owned, bound: derived)
    M: 32,               // control points of the bound curve
    Ns: 32,              // sparse "SfM" points both models start from
    noise: 0.09,         // their position noise (world units; one pixel is ~0.026)
    fitLam: 0.02,        // ridge on second differences in the least-squares curve fit
    sigN: 0.012,         // bound: fixed std-dev along the normal
    kappa: 0.7,          // bound: sigma_tangent = kappa * |S'(u)| * du
    tau: 0.06,           // a visible Gaussian farther than this from the true curve is a floater
    visW: 0.05,          // "visible" = blending weight alpha*T above this in some view on the arc
    lrPos: 0.006, lrScale: 0.02, lrRot: 0.02, lrCol: 0.04, lrOp: 0.04,
    decaySteps: 3000, decayTo: 0.1,   // every learning rate decays exponentially to 10 % over 3000 steps
    maxSteps: 4000
  };

  var NEAR = 0.2, LOWPASS = 0.09 /* (0.3 px)^2, as in 3DGS */, ALPHA_MIN = 1 / 255, ALPHA_MAX = 0.99, T_MIN = 1e-4;

  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function logit(p) { return Math.log(p / (1 - p)); }
  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function sstep(a, b, x) { var t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

  /* ---------- ground truth: a curve y = h(x) with colour along it ---------- */
  function gtHeight(x) {
    var b = (x - 0.38) / 0.24;
    return 0.11 * Math.sin(1.7 * x + 0.5) + 0.05 * Math.sin(4.3 * x - 1.0) + 0.58 * Math.exp(-b * b);
  }
  /* smooth ramp + four distinct bands */
  var BANDS = [
    { a: -1.10, b: -0.84, c: [0.96, 0.95, 0.88] },
    { a: -0.42, b: -0.20, c: [0.10, 0.52, 0.30] },
    { a: 0.28, b: 0.48, c: [0.93, 0.22, 0.62] },
    { a: 0.86, b: 1.08, c: [0.97, 0.90, 0.22] }
  ];
  function gtColor(x, out) {
    var t = (x + CFG.L) / (2 * CFG.L), e = 0.025, r, g, b, i, k, B;
    r = 0.90 - 0.68 * t; g = 0.42 + 0.20 * Math.sin(3.1 * t); b = 0.22 + 0.72 * t;
    for (i = 0; i < BANDS.length; i++) {
      B = BANDS[i]; k = sstep(B.a - e, B.a + e, x) * (1 - sstep(B.b - e, B.b + e, x));
      r += (B.c[0] - r) * k; g += (B.c[1] - g) * k; b += (B.c[2] - b) * k;
    }
    out[0] = r; out[1] = g; out[2] = b;
    return out;
  }

  /* dense polyline of the true curve, for distance queries and drawing */
  var GT_N = 561, GT_X = new Float64Array(GT_N), GT_Y = new Float64Array(GT_N);
  (function () {
    for (var i = 0; i < GT_N; i++) { GT_X[i] = -CFG.L + 2 * CFG.L * i / (GT_N - 1); GT_Y[i] = gtHeight(GT_X[i]); }
  })();
  function distToCurve(x, y) {
    var best = Infinity, i, ax, ay, bx, by, t, dx, dy, d;
    for (i = 0; i < GT_N - 1; i++) {
      ax = GT_X[i]; ay = GT_Y[i]; bx = GT_X[i + 1] - ax; by = GT_Y[i + 1] - ay;
      t = clamp(((x - ax) * bx + (y - ay) * by) / (bx * bx + by * by), 0, 1);
      dx = x - ax - t * bx; dy = y - ay - t * by; d = dx * dx + dy * dy;
      if (d < best) best = d;
    }
    return Math.sqrt(best);
  }

  /* ---------- the hand-off curve ----------
     Chapter 01 ends (scene `handoff`) with one curve left on the stage, and chapter 02 (talk/sims/basis.js,
     scene `ducks`) starts from the same curve, in the same place and style. It is the true curve above, written
     as a clamped uniform cubic B-spline with 14 control points: least squares on the GT_N points of y = h(x),
     parametrized by x (u = (x + L) / 2L), both end points pinned to the ends of the curve. Computed once, with
     no randomness; its worst distance to the true curve is 0.0024 (half a pixel on a laptop-sized stage).
     y0 / y1 = the y range of that spline. Everything is rounded to 4 decimals because basis.js carries a literal
     copy (its HANDOFF): keep the two equal.
       node -e "console.log(JSON.stringify(require('./talk/sims/flatland.js').HANDOFF))"                      */
  var HANDOFF = (function () {
    var n = 14, p = 3, U = NURBS.clampedKnots(n, p), N = new Float64Array(p + 1), A = [], B = [], P = [];
    var i, j, k, sp, u, f, t, x, y, y0 = Infinity, y1 = -Infinity, q;
    function r4(v) { return Math.round(v * 1e4) / 1e4; }
    for (i = 0; i < n; i++) { A.push(new Float64Array(n)); B.push([0, 0]); }
    for (k = 0; k < GT_N; k++) {
      u = k / (GT_N - 1); sp = NURBS.findSpan(n, p, u, U); NURBS.basisFuns(sp, u, p, U, N);
      for (i = 0; i <= p; i++) {
        B[sp - p + i][0] += N[i] * GT_X[k]; B[sp - p + i][1] += N[i] * GT_Y[k];
        for (j = 0; j <= p; j++) A[sp - p + i][sp - p + j] += N[i] * N[j];
      }
    }
    A[0].fill(0); A[0][0] = 1; B[0] = [GT_X[0], GT_Y[0]];                            // pinned ends
    A[n - 1].fill(0); A[n - 1][n - 1] = 1; B[n - 1] = [GT_X[GT_N - 1], GT_Y[GT_N - 1]];
    for (i = 0; i < n; i++) {                                                          // Gaussian elimination, partial pivoting
      for (k = i + 1, j = i; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[j][i])) j = k;
      t = A[i]; A[i] = A[j]; A[j] = t; t = B[i]; B[i] = B[j]; B[j] = t;
      for (k = i + 1; k < n; k++) {
        f = A[k][i] / A[i][i]; if (!f) continue;
        for (j = i; j < n; j++) A[k][j] -= f * A[i][j];
        B[k][0] -= f * B[i][0]; B[k][1] -= f * B[i][1];
      }
    }
    for (i = n - 1; i >= 0; i--) {
      x = B[i][0]; y = B[i][1];
      for (j = i + 1; j < n; j++) { x -= A[i][j] * P[j][0]; y -= A[i][j] * P[j][1]; }
      P[i] = [x / A[i][i], y / A[i][i]];
    }
    P = P.map(function (c) { return [r4(c[0]), r4(c[1])]; });
    for (k = 0; k <= 2000; k++) { q = NURBS.curvePoint(P, null, p, U, k / 2000); y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]); }
    return { P: P, y0: r4(y0), y1: r4(y1) };
  })();

  /* ---------- 1D pinhole camera on the arc ---------- */
  function makeCamera(theta, cam) {
    cam = cam || {};
    var s = Math.sin(theta), c = Math.cos(theta);
    cam.theta = theta; cam.W = CFG.W; cam.f = (CFG.W / 2) / CFG.tanHalf;
    cam.px = CFG.R * s; cam.py = CFG.cy + CFG.R * c;
    cam.dx = -s; cam.dy = -c;          // forward (unit)
    cam.rx = c; cam.ry = -s;           // image axis (unit, perpendicular to forward)
    return cam;
  }
  function trainAngles(K) {
    var a = [], k;
    for (k = 0; k < K; k++) a.push(-CFG.thTrain + 2 * CFG.thTrain * k / (K - 1));
    return a;
  }
  /* default held-out angle: the middle of the gap just right of centre (never a training angle) */
  function defaultNovelAngle(K) {
    var a = trainAngles(K), g = Math.floor((K - 1) / 2);
    return 0.5 * (a[g] + a[g + 1]);
  }

  /* first hit of a ray with the true curve; returns x of the hit or NaN */
  function rayHit(ox, oy, vx, vy) {
    // only the slab y in [-0.5, 1.0] can contain the curve
    var t0 = (1.0 - oy) / vy, t1 = (-0.5 - oy) / vy, dt = 0.008, t, x, g, gp = 0, tp = 0, have = false;
    if (t0 < 0) t0 = 0;
    for (t = t0; t <= t1 + dt; t += dt) {
      x = ox + t * vx;
      if (x < -CFG.L || x > CFG.L) { have = false; continue; }
      g = oy + t * vy - gtHeight(x);
      if (have && ((g <= 0) !== (gp <= 0))) {
        var lo = tp, hi = t, glo = gp, k, mid, gm;
        for (k = 0; k < 24; k++) {
          mid = 0.5 * (lo + hi); gm = oy + mid * vy - gtHeight(ox + mid * vx);
          if ((gm <= 0) === (glo <= 0)) { lo = mid; glo = gm; } else hi = mid;
        }
        return ox + 0.5 * (lo + hi) * vx;
      }
      gp = g; tp = t; have = true;
    }
    return NaN;
  }
  /* ground-truth strip: ray–curve intersection, 8 sub-rays per pixel (box filter), black background.
     cov (optional): receives per pixel the share of its sub-rays that hit the curve (for drawing only) */
  var _c3 = [0, 0, 0];
  function renderGT(cam, out, cov) {
    var W = cam.W, S = 8, j, s, u, vx, vy, hx, r, g, b, k;
    out = out || new Float32Array(W * 3);
    for (j = 0; j < W; j++) {
      r = g = b = 0; k = 0;
      for (s = 0; s < S; s++) {
        u = (j + (s + 0.5) / S - W / 2) / cam.f;
        vx = cam.dx + u * cam.rx; vy = cam.dy + u * cam.ry;
        hx = rayHit(cam.px, cam.py, vx, vy);
        if (hx === hx) { gtColor(hx, _c3); r += _c3[0]; g += _c3[1]; b += _c3[2]; k++; }
      }
      out[j * 3] = r / S; out[j * 3 + 1] = g / S; out[j * 3 + 2] = b / S;
      if (cov) cov[j] = k / S;
    }
    return out;
  }

  /* ---------- a set of 2D Gaussians (struct of arrays) ----------
     mean (mx,my), covariance [[a,b],[b,c]], colour (cr,cg,cb) in [0,1], opacity op in (0,1).
     The same struct holds dL/d(each field) when used as a gradient buffer. */
  function makeGaussians(n) {
    var g = { n: n }, F = ['mx', 'my', 'a', 'b', 'c', 'cr', 'cg', 'cb', 'op'];
    for (var i = 0; i < F.length; i++) g[F[i]] = new Float64Array(n);
    return g;
  }
  function zeroGaussians(g) {
    g.mx.fill(0); g.my.fill(0); g.a.fill(0); g.b.fill(0); g.c.fill(0);
    g.cr.fill(0); g.cg.fill(0); g.cb.fill(0); g.op.fill(0);
  }

  /* ---------- the renderer (shared by both models) ----------
     Project every Gaussian to the camera's 1D image (EWA: var' = J Sigma J^T + low-pass), sort by
     depth, composite front to back:  C = sum_i c_i alpha_i prod_{j<i} (1 - alpha_j).
     If `tgt` is given returns the sum of squared errors; if `gg` is also given, accumulates
     scale * d(SSE)/d(every Gaussian field) into it — exact analytic gradients through the
     compositing, the projection and the projected variance. Only the depth ORDER is treated as
     constant (its derivative is zero almost everywhere). */
  var SC = { n: 0 }, VIS = null;   // VIS: optional per-Gaussian record of the peak blending weight alpha*T
  var COV = null;                  // COV: optional per-pixel record of the coverage 1 - T (for drawing over a backdrop)
  function scratch(n) {
    if (SC.n >= n) return;
    SC.n = n;
    ['z', 'u', 'v', 'jx', 'jy', 'xc', 'dU', 'dV', 'lA', 'lT', 'lG'].forEach(function (k) { SC[k] = new Float64Array(n); });
    SC.lo = new Int32Array(n); SC.hi = new Int32Array(n); SC.ord = new Int32Array(n); SC.lI = new Int32Array(n);
  }
  function renderCam(cam, gs, img, tgt, gg, scale) {
    var n = gs.n, W = cam.W, f = cam.f, i, j, k, cnt = 0;
    scratch(n);
    var sz = SC.z, su = SC.u, sv = SC.v, sjx = SC.jx, sjy = SC.jy, sxc = SC.xc, dU = SC.dU, dV = SC.dV;
    var lo = SC.lo, hi = SC.hi, ord = SC.ord, lI = SC.lI, lA = SC.lA, lT = SC.lT, lG = SC.lG;
    var ex, ey, z, xc, jx, jy, v, u, rad;
    for (i = 0; i < n; i++) {
      ex = gs.mx[i] - cam.px; ey = gs.my[i] - cam.py;
      z = ex * cam.dx + ey * cam.dy;
      if (z < NEAR || gs.op[i] < ALPHA_MIN) continue;
      xc = ex * cam.rx + ey * cam.ry;
      jx = f * (cam.rx / z - xc * cam.dx / (z * z)); jy = f * (cam.ry / z - xc * cam.dy / (z * z));
      v = jx * jx * gs.a[i] + 2 * jx * jy * gs.b[i] + jy * jy * gs.c[i] + LOWPASS;
      u = f * xc / z + W / 2;
      rad = 3.5 * Math.sqrt(v);
      if (u + rad < 0 || u - rad > W) continue;
      sz[i] = z; su[i] = u; sv[i] = v; sjx[i] = jx; sjy[i] = jy; sxc[i] = xc; dU[i] = 0; dV[i] = 0;
      lo[i] = Math.max(0, Math.floor(u - rad)); hi[i] = Math.min(W - 1, Math.floor(u + rad));
      // insertion sort by depth
      k = cnt++;
      while (k > 0 && sz[ord[k - 1]] > z) { ord[k] = ord[k - 1]; k--; }
      ord[k] = i;
    }
    var sse = 0, pc, T, R, Gc, B, m, d, g, al, w, hit;
    for (j = 0; j < W; j++) {
      pc = j + 0.5; T = 1; R = 0; Gc = 0; B = 0; m = 0;
      for (k = 0; k < cnt; k++) {
        i = ord[k];
        if (j < lo[i] || j > hi[i]) continue;
        d = pc - su[i]; g = Math.exp(-0.5 * d * d / sv[i]);
        al = gs.op[i] * g;
        if (al < ALPHA_MIN) continue;
        hit = al > ALPHA_MAX; if (hit) al = ALPHA_MAX;
        w = al * T;
        if (VIS && w > VIS[i]) VIS[i] = w;
        R += gs.cr[i] * w; Gc += gs.cg[i] * w; B += gs.cb[i] * w;
        lI[m] = i; lA[m] = al; lT[m] = T; lG[m] = hit ? -1 : g; m++;
        T *= 1 - al;
        if (T < T_MIN) break;
      }
      if (img) { img[j * 3] = R; img[j * 3 + 1] = Gc; img[j * 3 + 2] = B; }
      if (COV) COV[j] = 1 - T;
      if (!tgt) continue;
      var er = R - tgt[j * 3], eg = Gc - tgt[j * 3 + 1], eb = B - tgt[j * 3 + 2];
      sse += er * er + eg * eg + eb * eb;
      if (!gg) continue;
      // backward through the compositing, back to front. S = colour accumulated BEHIND Gaussian i.
      var qr = 2 * er * scale, qg = 2 * eg * scale, qb = 2 * eb * scale, Sr = 0, Sg = 0, Sb = 0, inv, dA, dG;
      for (k = m - 1; k >= 0; k--) {
        i = lI[k]; al = lA[k]; T = lT[k]; w = al * T;
        gg.cr[i] += qr * w; gg.cg[i] += qg * w; gg.cb[i] += qb * w;
        inv = 1 / (1 - al);
        dA = qr * (gs.cr[i] * T - Sr * inv) + qg * (gs.cg[i] * T - Sg * inv) + qb * (gs.cb[i] * T - Sb * inv);
        Sr += gs.cr[i] * w; Sg += gs.cg[i] * w; Sb += gs.cb[i] * w;
        g = lG[k];
        if (g < 0) continue;                 // alpha was clamped: no gradient
        gg.op[i] += dA * g;                  // alpha = op * g
        dG = dA * gs.op[i] * g;              // dL/dg * g
        d = pc - su[i];
        dU[i] += dG * d / sv[i];             // g = exp(-d^2 / 2v), d = pc - u
        dV[i] += dG * 0.5 * d * d / (sv[i] * sv[i]);
      }
    }
    if (gg) {
      // chain rule: (u, v) -> mean and covariance.  du/dm = J;  v = J Sigma J^T + low-pass.
      var gJx, gJy, gr, gd, z2, e1, e2;
      for (k = 0; k < cnt; k++) {
        i = ord[k]; jx = sjx[i]; jy = sjy[i]; z = sz[i]; xc = sxc[i]; z2 = z * z;
        gg.a[i] += dV[i] * jx * jx; gg.b[i] += dV[i] * 2 * jx * jy; gg.c[i] += dV[i] * jy * jy;
        gJx = 2 * (jx * gs.a[i] + jy * gs.b[i]); gJy = 2 * (jx * gs.b[i] + jy * gs.c[i]);   // dv/dJ
        gr = gJx * cam.rx + gJy * cam.ry; gd = gJx * cam.dx + gJy * cam.dy;
        // J = f r / z - f xc d / z^2  =>  dv/dm = -(f/z^2)(gJ.r) d - f (gJ.d) (r/z^2 - 2 xc d / z^3)
        e1 = -(f / z2) * gr + 2 * f * gd * xc / (z2 * z); e2 = -f * gd / z2;
        gg.mx[i] += dU[i] * jx + dV[i] * (e1 * cam.dx + e2 * cam.rx);
        gg.my[i] += dU[i] * jy + dV[i] * (e1 * cam.dy + e2 * cam.ry);
      }
    }
    return sse;
  }

  /* ---------- FREE model: 9 parameters per Gaussian ----------
     theta[i*9 ..] = x, y, log s1, log s2, phi, colour logits (3), opacity logit
     Sigma = R(phi) diag(s1^2, s2^2) R(phi)^T */
  var FREE_P = 9;
  function freeDerive(th, gs) {
    var n = gs.n, i, o, s1, s2, c, s;
    for (i = 0; i < n; i++) {
      o = i * FREE_P;
      gs.mx[i] = th[o]; gs.my[i] = th[o + 1];
      s1 = Math.exp(2 * th[o + 2]); s2 = Math.exp(2 * th[o + 3]); c = Math.cos(th[o + 4]); s = Math.sin(th[o + 4]);
      gs.a[i] = c * c * s1 + s * s * s2; gs.b[i] = c * s * (s1 - s2); gs.c[i] = s * s * s1 + c * c * s2;
      gs.cr[i] = sigmoid(th[o + 5]); gs.cg[i] = sigmoid(th[o + 6]); gs.cb[i] = sigmoid(th[o + 7]);
      gs.op[i] = sigmoid(th[o + 8]);
    }
  }
  function freeBackward(th, gs, gg, grad) {
    var n = gs.n, i, o, s1, s2, c, s, ga, gb, gc;
    for (i = 0; i < n; i++) {
      o = i * FREE_P;
      grad[o] = gg.mx[i]; grad[o + 1] = gg.my[i];
      s1 = Math.exp(2 * th[o + 2]); s2 = Math.exp(2 * th[o + 3]); c = Math.cos(th[o + 4]); s = Math.sin(th[o + 4]);
      ga = gg.a[i]; gb = gg.b[i]; gc = gg.c[i];
      grad[o + 2] = 2 * s1 * (ga * c * c + gb * c * s + gc * s * s);
      grad[o + 3] = 2 * s2 * (ga * s * s - gb * c * s + gc * c * c);
      grad[o + 4] = (gc - ga) * 2 * gs.b[i] + gb * (c * c - s * s) * (s1 - s2);
      grad[o + 5] = gg.cr[i] * gs.cr[i] * (1 - gs.cr[i]);
      grad[o + 6] = gg.cg[i] * gs.cg[i] * (1 - gs.cg[i]);
      grad[o + 7] = gg.cb[i] * gs.cb[i] * (1 - gs.cb[i]);
      grad[o + 8] = gg.op[i] * gs.op[i] * (1 - gs.op[i]);
    }
  }

  /* ---------- BOUND model: 5 parameters per control point + 1 ----------
     theta = [ P (M x 2) | F (M x 3 colour features) | opacity logit ]
     Derived, never optimized:   u_i = (i + 1/2) / G
       mean_i   = S(u_i)                          = sum_j N_j(u_i) P_j
       Sigma_i  = (kappa du)^2 S' S'^T + sigN^2 n n^T     (tangent-aligned, thin along the normal)
       colour_i = sigmoid( sum_j N_j(u_i) F_j )
     Gradients reach the control points by the chain rule:
       dL/dP_j = sum_i N_j(u_i) dL/dmean_i  +  N'_j(u_i) dL/dS'_i
     The second term carries the gradient through the DERIVED covariance (it depends on S'), so the
     gradient is exact — verified against finite differences. `covGrad = false` stop-gradients
     through the covariance instead; in this experiment that variant does not converge (the curve
     no longer feels that stretching a span widens its Gaussians), so the exact gradient is used. */
  function boundTable(M, G) {
    var p = 3, U = NURBS.clampedKnots(M, p), us = [], i;
    for (i = 0; i < G; i++) us.push((i + 0.5) / G);
    var T = NURBS.axisTable(U, M, p, us);
    T.p = p; T.M = M; T.G = G; T.knots = U; T.du = 1 / G;
    T.dx = new Float64Array(G); T.dy = new Float64Array(G);
    return T;
  }
  function boundDerive(th, T, gs) {
    var G = T.G, M = T.M, w = T.p + 1, A = CFG.kappa * T.du, i, r, j, N, D, x, y, dx, dy, fr, fg, fb, q, k;
    var sn2 = CFG.sigN * CFG.sigN, op = sigmoid(th[5 * M]);
    A *= A;
    for (i = 0; i < G; i++) {
      x = y = dx = dy = fr = fg = fb = 0;
      for (r = 0; r < w; r++) {
        j = T.span[i] - T.p + r; N = T.N[i * w + r]; D = T.D[i * w + r];
        x += N * th[2 * j]; y += N * th[2 * j + 1]; dx += D * th[2 * j]; dy += D * th[2 * j + 1];
        k = 2 * M + 3 * j; fr += N * th[k]; fg += N * th[k + 1]; fb += N * th[k + 2];
      }
      q = dx * dx + dy * dy;
      gs.mx[i] = x; gs.my[i] = y; T.dx[i] = dx; T.dy[i] = dy;
      gs.a[i] = A * dx * dx + sn2 * (1 - dx * dx / q);
      gs.b[i] = A * dx * dy - sn2 * dx * dy / q;
      gs.c[i] = A * dy * dy + sn2 * (1 - dy * dy / q);
      gs.cr[i] = sigmoid(fr); gs.cg[i] = sigmoid(fg); gs.cb[i] = sigmoid(fb);
      gs.op[i] = op;
    }
  }
  function boundBackward(th, T, gs, gg, grad, covGrad) {
    var G = T.G, M = T.M, w = T.p + 1, A = CFG.kappa * T.du, i, r, j, N, D, dx, dy, q, q2, gdx, gdy, k;
    var sn2 = CFG.sigN * CFG.sigN, gop = 0, hr, hg, hb, ga, gb, gc;
    A *= A;
    grad.fill(0);
    for (i = 0; i < G; i++) {
      gdx = gdy = 0;
      if (covGrad) {
        dx = T.dx[i]; dy = T.dy[i]; q = dx * dx + dy * dy; q2 = q * q; ga = gg.a[i]; gb = gg.b[i]; gc = gg.c[i];
        gdx = ga * (2 * A * dx - 2 * sn2 * dx * dy * dy / q2) + gb * (A * dy - sn2 * dy * (dy * dy - dx * dx) / q2) +
              gc * (2 * sn2 * dy * dy * dx / q2);
        gdy = ga * (2 * sn2 * dx * dx * dy / q2) + gb * (A * dx - sn2 * dx * (dx * dx - dy * dy) / q2) +
              gc * (2 * A * dy - 2 * sn2 * dy * dx * dx / q2);
      }
      hr = gg.cr[i] * gs.cr[i] * (1 - gs.cr[i]); hg = gg.cg[i] * gs.cg[i] * (1 - gs.cg[i]); hb = gg.cb[i] * gs.cb[i] * (1 - gs.cb[i]);
      for (r = 0; r < w; r++) {
        j = T.span[i] - T.p + r; N = T.N[i * w + r]; D = T.D[i * w + r];
        grad[2 * j] += N * gg.mx[i] + D * gdx; grad[2 * j + 1] += N * gg.my[i] + D * gdy;
        k = 2 * M + 3 * j; grad[k] += N * hr; grad[k + 1] += N * hg; grad[k + 2] += N * hb;
      }
      gop += gg.op[i];
    }
    grad[5 * M] = gop * gs.op[0] * (1 - gs.op[0]);
  }

  /* ---------- Adam ---------- */
  function makeAdam(n) { return { m: new Float64Array(n), v: new Float64Array(n), t: 0 }; }
  function adamStep(th, grad, st, lr, k) {
    var b1 = 0.9, b2 = 0.999, i, n = th.length, c1, c2, g;
    st.t++;
    c1 = 1 - Math.pow(b1, st.t); c2 = 1 - Math.pow(b2, st.t);
    for (i = 0; i < n; i++) {
      g = grad[i];
      st.m[i] = b1 * st.m[i] + (1 - b1) * g; st.v[i] = b2 * st.v[i] + (1 - b2) * g * g;
      th[i] -= k * lr[i] * (st.m[i] / c1) / (Math.sqrt(st.v[i] / c2) + 1e-10);
    }
  }
  function lrDecay(step) { return Math.pow(CFG.decayTo, Math.min(step, CFG.decaySteps) / CFG.decaySteps); }

  /* ---------- the shared starting information: sparse noisy coloured points ----------
     A stand-in for an SfM point cloud: Ns points on the true surface, each displaced by
     isotropic Gaussian noise, each carrying the surface colour it was triangulated from. */
  function gauss(rnd) { return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }
  function makePoints(rnd) {
    var n = CFG.Ns, pts = [], i, x, c;
    for (i = 0; i < n; i++) {
      x = -CFG.L + 2 * CFG.L * (i + 0.15 + 0.7 * rnd()) / n;
      c = gtColor(x, [0, 0, 0]);
      pts.push({ x: x + CFG.noise * gauss(rnd), y: gtHeight(x) + CFG.noise * gauss(rnd), c: c });
    }
    return pts;
  }

  /* FREE init: one Gaussian per point, the remainder are jittered copies; isotropic, size from the
     nearest-neighbour spacing (as 3DGS does), colour from the point, opacity 0.5. */
  function initFree(pts, rnd) {
    var G = CFG.G, n = pts.length, th = new Float64Array(G * FREE_P), i, k, o, p, d, d1, d2, s, jit;
    for (i = 0; i < G; i++) {
      p = pts[i % n]; o = i * FREE_P;
      d1 = d2 = Infinity;
      for (k = 0; k < n; k++) if (k !== i % n) {
        d = Math.hypot(pts[k].x - p.x, pts[k].y - p.y);
        if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) d2 = d;
      }
      s = clamp(0.5 * (d1 + d2) * n / G, 0.02, 0.12);
      jit = i < n ? 0 : 0.5 * (d1 + d2) * 0.5;
      th[o] = p.x + jit * gauss(rnd); th[o + 1] = p.y + jit * gauss(rnd);
      th[o + 2] = th[o + 3] = Math.log(s); th[o + 4] = 0;
      th[o + 5] = logit(clamp(p.c[0], 0.03, 0.97)); th[o + 6] = logit(clamp(p.c[1], 0.03, 0.97)); th[o + 7] = logit(clamp(p.c[2], 0.03, 0.97));
      th[o + 8] = 0;
    }
    return th;
  }

  /* BOUND init: least-squares cubic B-spline through the same points. They are parametrized by
     their projection on the principal axis of the cloud; a small second-difference ridge keeps
     the system well posed. Colour features are fitted the same way. */
  function initBound(pts) {
    var M = CFG.M, p = 3, U = NURBS.clampedKnots(M, p), n = pts.length, i, j, k, r;
    var mx = 0, my = 0, sxx = 0, sxy = 0, syy = 0;
    for (i = 0; i < n; i++) { mx += pts[i].x; my += pts[i].y; }
    mx /= n; my /= n;
    for (i = 0; i < n; i++) { var ex = pts[i].x - mx, ey = pts[i].y - my; sxx += ex * ex; sxy += ex * ey; syy += ey * ey; }
    var ang = 0.5 * Math.atan2(2 * sxy, sxx - syy), ax = Math.cos(ang), ay = Math.sin(ang);
    if (ax < 0) { ax = -ax; ay = -ay; }
    var t = [], tmin = Infinity, tmax = -Infinity;
    for (i = 0; i < n; i++) { t[i] = (pts[i].x - mx) * ax + (pts[i].y - my) * ay; tmin = Math.min(tmin, t[i]); tmax = Math.max(tmax, t[i]); }
    var A = [], Bv = [], N = new Float64Array(p + 1);
    for (i = 0; i < M; i++) { A.push(new Float64Array(M)); Bv.push(new Float64Array(5)); }
    for (k = 0; k < n; k++) {
      var u = (t[k] - tmin) / (tmax - tmin), sp = NURBS.findSpan(M, p, u, U), y = [pts[k].x, pts[k].y, pts[k].c[0], pts[k].c[1], pts[k].c[2]];
      NURBS.basisFuns(sp, u, p, U, N);
      for (i = 0; i <= p; i++) {
        for (j = 0; j <= p; j++) A[sp - p + i][sp - p + j] += N[i] * N[j];
        for (r = 0; r < 5; r++) Bv[sp - p + i][r] += N[i] * y[r];
      }
    }
    var lam = CFG.fitLam, st = [1, -2, 1];
    for (k = 0; k + 2 < M; k++) for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) A[k + i][k + j] += lam * st[i] * st[j];
    // Gaussian elimination with partial pivoting
    for (i = 0; i < M; i++) {
      var piv = i;
      for (k = i + 1; k < M; k++) if (Math.abs(A[k][i]) > Math.abs(A[piv][i])) piv = k;
      var tmp = A[i]; A[i] = A[piv]; A[piv] = tmp; tmp = Bv[i]; Bv[i] = Bv[piv]; Bv[piv] = tmp;
      for (k = i + 1; k < M; k++) {
        var fct = A[k][i] / A[i][i];
        if (!fct) continue;
        for (j = i; j < M; j++) A[k][j] -= fct * A[i][j];
        for (r = 0; r < 5; r++) Bv[k][r] -= fct * Bv[i][r];
      }
    }
    for (i = M - 1; i >= 0; i--) for (r = 0; r < 5; r++) {
      var s = Bv[i][r];
      for (j = i + 1; j < M; j++) s -= A[i][j] * Bv[j][r];
      Bv[i][r] = s / A[i][i];
    }
    var th = new Float64Array(5 * M + 1);
    for (i = 0; i < M; i++) {
      th[2 * i] = Bv[i][0]; th[2 * i + 1] = Bv[i][1];
      for (r = 0; r < 3; r++) th[2 * M + 3 * i + r] = logit(clamp(Bv[i][2 + r], 0.03, 0.97));
    }
    th[5 * M] = 0;   // opacity 0.5, like the free Gaussians
    return th;
  }

  /* ---------- a model = parameters + optimizer state ---------- */
  function makeModel(kind, th) {
    var n = CFG.G, lr = new Float64Array(th.length), i, o;
    var m = { kind: kind, th: th, th0: new Float64Array(th), grad: new Float64Array(th.length), lr: lr,
              adam: makeAdam(th.length), gs: makeGaussians(n), gg: makeGaussians(n), step: 0, loss: NaN,
              tab: kind === 'bound' ? boundTable(CFG.M, CFG.G) : null, covGrad: true };
    if (kind === 'free') {
      for (i = 0; i < n; i++) {
        o = i * FREE_P;
        lr[o] = lr[o + 1] = CFG.lrPos; lr[o + 2] = lr[o + 3] = CFG.lrScale; lr[o + 4] = CFG.lrRot;
        lr[o + 5] = lr[o + 6] = lr[o + 7] = CFG.lrCol; lr[o + 8] = CFG.lrOp;
      }
    } else {
      for (i = 0; i < 2 * CFG.M; i++) lr[i] = CFG.lrPos;
      for (i = 2 * CFG.M; i < 5 * CFG.M; i++) lr[i] = CFG.lrCol;
      lr[5 * CFG.M] = CFG.lrOp;
    }
    derive(m);
    return m;
  }
  function derive(m) { if (m.kind === 'free') freeDerive(m.th, m.gs); else boundDerive(m.th, m.tab, m.gs); }

  /* photometric loss (mean squared error over all training pixels) and its gradient */
  function lossGrad(m, cams, tgts, wantGrad) {
    var k, sse = 0, scale = 1 / (cams.length * CFG.W * 3);
    derive(m);
    if (wantGrad) zeroGaussians(m.gg);
    for (k = 0; k < cams.length; k++) sse += renderCam(cams[k], m.gs, null, tgts[k], wantGrad ? m.gg : null, scale);
    if (wantGrad) {
      if (m.kind === 'free') freeBackward(m.th, m.gs, m.gg, m.grad);
      else boundBackward(m.th, m.tab, m.gs, m.gg, m.grad, m.covGrad);
    }
    return sse * scale;
  }
  function trainStep(m, cams, tgts) {
    m.loss = lossGrad(m, cams, tgts, true);
    adamStep(m.th, m.grad, m.adam, m.lr, lrDecay(m.step));
    m.step++;
  }

  /* ---------- metrics ---------- */
  function psnrOf(mse) { return -10 * Math.log(Math.max(mse, 1e-12)) / Math.LN10; }
  function psnr(m, cams, tgts) {
    var k, sse = 0;
    for (k = 0; k < cams.length; k++) sse += renderCam(cams[k], m.gs, null, tgts[k], null, 0);
    return psnrOf(sse / (cams.length * CFG.W * 3));
  }
  /* Which Gaussians can actually be seen? Render from a fan of audit cameras over the whole arc and
     keep each Gaussian's peak blending weight alpha*T. Hidden ones (buried behind opaque
     neighbours, or faded out) are not counted for or against a model. */
  var AUDIT = null;
  function visibility(m, vis) {
    var k;
    if (!AUDIT) { AUDIT = []; for (k = 0; k < 9; k++) AUDIT.push(makeCamera(-CFG.thNovel + 2 * CFG.thNovel * k / 8)); }
    vis.fill(0); VIS = vis;
    for (k = 0; k < AUDIT.length; k++) renderCam(AUDIT[k], m.gs, null, null, null, 0);
    VIS = null;
  }
  /* geometry error: opacity-weighted mean distance of the visible centres to the true curve;
     floaters: visible Gaussians farther than tau. `dist` / `vis` receive per-Gaussian values. */
  function geometry(m, dist, vis) {
    var gs = m.gs, i, d, sw = 0, sd = 0, fl = 0, nv = 0;
    vis = vis || new Float64Array(gs.n);
    visibility(m, vis);
    for (i = 0; i < gs.n; i++) {
      d = distToCurve(gs.mx[i], gs.my[i]);
      if (dist) dist[i] = d;
      if (vis[i] < CFG.visW) continue;
      nv++; sw += gs.op[i]; sd += gs.op[i] * d;
      if (d > CFG.tau) fl++;
    }
    return { err: sw > 0 ? sd / sw : 0, floaters: fl, visible: nv };
  }

  /* ---------- depth-ambiguity probe (the `rays` scene) ----------
     Move Gaussian i by `along` on the ray from `cam` through it and by `across` perpendicular to
     that ray. An along-ray move also rescales the covariance with depth^2, so the Gaussian's
     footprint in `cam` itself is exactly unchanged: whatever the loss still feels comes from the
     other cameras and from changed occlusion. Returns the saved state for unshift(). */
  function shiftGaussian(gs, i, cam, along, across) {
    var sv = [gs.mx[i], gs.my[i], gs.a[i], gs.b[i], gs.c[i]];
    var rx = sv[0] - cam.px, ry = sv[1] - cam.py, rl = Math.sqrt(rx * rx + ry * ry), z0, z1, k;
    rx /= rl; ry /= rl;
    z0 = (sv[0] - cam.px) * cam.dx + (sv[1] - cam.py) * cam.dy;
    gs.mx[i] = sv[0] + along * rx - across * ry; gs.my[i] = sv[1] + along * ry + across * rx;
    z1 = (gs.mx[i] - cam.px) * cam.dx + (gs.my[i] - cam.py) * cam.dy;
    if (along) { k = z1 / z0; k *= k; gs.a[i] *= k; gs.b[i] *= k; gs.c[i] *= k; }
    return sv;
  }
  function unshift(gs, i, sv) { gs.mx[i] = sv[0]; gs.my[i] = sv[1]; gs.a[i] = sv[2]; gs.b[i] = sv[3]; gs.c[i] = sv[4]; }
  /* For every visible floater of model m: the training camera it matters to most (`cam`), the
     share of its image influence that falls in that camera (`share`), and the training PSNR after
     sliding it +-d along / across that camera's ray (worst of the two signs). */
  function rayProbe(E, m, dist, vis, d) {
    var gs = m.gs, K = E.K, W = CFG.W, n3 = W * 3, i, k, j, res = [], base;
    var full = [], img = new Float32Array(n3);
    base = psnr(m, E.cams, E.tgts);
    for (k = 0; k < K; k++) { full.push(new Float32Array(n3)); renderCam(E.cams[k], gs, full[k], null, null, 0); }
    for (i = 0; i < gs.n; i++) {
      if (vis[i] < CFG.visW || dist[i] <= CFG.tau) continue;
      var op = gs.op[i], infl = [], tot = 0, dom = 0, s, sv, al, ac;
      gs.op[i] = 0;                              // influence = how much each strip changes without it
      for (k = 0; k < K; k++) {
        renderCam(E.cams[k], gs, img, null, null, 0);
        for (s = 0, j = 0; j < n3; j++) s += Math.abs(img[j] - full[k][j]);
        infl.push(s); tot += s; if (s > infl[dom]) dom = k;
      }
      gs.op[i] = op;
      if (tot < 1e-6) continue;
      sv = shiftGaussian(gs, i, E.cams[dom], d, 0); al = psnr(m, E.cams, E.tgts); unshift(gs, i, sv);
      sv = shiftGaussian(gs, i, E.cams[dom], -d, 0); al = Math.min(al, psnr(m, E.cams, E.tgts)); unshift(gs, i, sv);
      sv = shiftGaussian(gs, i, E.cams[dom], 0, d); ac = psnr(m, E.cams, E.tgts); unshift(gs, i, sv);
      sv = shiftGaussian(gs, i, E.cams[dom], 0, -d); ac = Math.min(ac, psnr(m, E.cams, E.tgts)); unshift(gs, i, sv);
      res.push({ i: i, cam: dom, share: infl[dom] / tot, tot: tot, along: al, across: ac, dist: dist[i] });
    }
    return { base: base, list: res };
  }

  /* ---------- an experiment: scene + cameras + both models ---------- */
  function makeExperiment(o) {
    var K = o.K, rnd = o.rng(o.seed), E = { K: K, seed: o.seed };
    E.cams = trainAngles(K).map(function (a) { return makeCamera(a); });
    // o.cover: also keep each true strip's coverage (renderGT's cov), for drawing it over a backdrop; null otherwise
    E.cov = o.cover ? { train: E.cams.map(function () { return new Float32Array(CFG.W); }), novel: new Float32Array(CFG.W) } : null;
    E.tgts = E.cams.map(function (c, k) { return renderGT(c, null, E.cov && E.cov.train[k]); });
    E.novel = makeCamera(o.novel != null ? o.novel : defaultNovelAngle(K));
    E.novelTgt = renderGT(E.novel, null, E.cov && E.cov.novel);
    E.pts = makePoints(rnd);
    E.free = makeModel('free', initFree(E.pts, rnd));
    E.bound = makeModel('bound', initBound(E.pts));
    return E;
  }
  function setNovel(E, theta) { makeCamera(theta, E.novel); renderGT(E.novel, E.novelTgt, E.cov && E.cov.novel); }

  var core = {
    CFG: CFG, FREE_P: FREE_P, GT_N: GT_N, GT_X: GT_X, GT_Y: GT_Y, HANDOFF: HANDOFF,
    sigmoid: sigmoid, gtHeight: gtHeight, gtColor: gtColor, distToCurve: distToCurve,
    makeCamera: makeCamera, trainAngles: trainAngles, defaultNovelAngle: defaultNovelAngle, renderGT: renderGT,
    makeGaussians: makeGaussians, renderCam: renderCam, derive: derive, lossGrad: lossGrad, trainStep: trainStep,
    psnr: psnr, psnrOf: psnrOf, geometry: geometry, shiftGaussian: shiftGaussian, unshift: unshift, rayProbe: rayProbe, makeExperiment: makeExperiment, setNovel: setNovel,
    makePoints: makePoints, initFree: initFree, initBound: initBound, makeModel: makeModel, lrDecay: lrDecay
  };
  if (NODE) { module.exports = core; return; }

  /* =====================================================================================
     2. BROWSER SECTION
     ===================================================================================== */
  var KT = root.KnotTalk, COL = KT.color, LIGHT = KT.theme === 'light';
  /* T: a string in the page's language (talk/i18n.js; the English itself on a page without languages). A sentence with a
     number in it is one template, never pieces joined in English order. In Hebrew the stage stays laid out left to right
     (kit.css) and a string from T carries its own direction, so the canvas text below needs nothing more. Never compare
     or parse what T returns. */
  var T = KT.t;
  var SEED = 6, K_DEF = 3, K_MIN = 2, K_MAX = 8;
  var RATE = 360;                       // optimizer steps per second (time-based, not per frame)
  var SLIDE = 0.12;                     // probe distance of the `rays` scene (world units = metres)
  var VIEW = { x0: -2.2, x1: 2.2, y0: -0.6, y1: 2.66 };   // world window of one view
  var ARC_N = 33, ARC_LO = 18, ARC_HI = 34;               // held-out PSNR sampled along the arc

  var SCENES = {
    free:   { model: 'free',  reset: true,  k: 'user', sel: 'train', pulse: false, hl: false, rays: false },
    novel:  { model: 'free',  reset: false, k: null,   sel: 'novel', pulse: true,  hl: true,  rays: false },
    rays:   { model: 'free',  reset: true,  k: 'min',  sel: 'auto',  pulse: false, hl: true,  rays: true },
    bound:  { model: 'bound', reset: true,  k: 'user', sel: 'novel', pulse: false, hl: false, rays: false },
    versus: { model: 'both',  reset: true,  k: 'user', sel: 'novel', pulse: false, hl: false, rays: false },
    /* the closing beat: the picture of the scene before it, then only the bound curve stays (see `drawHandoff`) */
    handoff: { model: 'both', reset: false, k: null,   sel: null,    pulse: false, hl: false, rays: false, handoff: true }
  };

  /* ---------- the hand-off: chapter 01's last picture is chapter 02's first ----------
     HANDOFF LAYOUT, the same rule in talk/sims/basis.js (scene `ducks`), so the two pictures coincide on stages
     of the same size: the curve's x range, -1.4 .. 1.4, spans the middle 76 % of the stage width; y uses the same
     scale; the middle of the curve's y range (HANDOFF.y0 .. y1) sits at the vertical centre of the stage. Should
     that make the curve taller than 40 % of the stage height, the height sets the scale instead (still centred).
     HANDOFF STYLE, also shared: the curve as a lit tube in the surface colour. Core width w = clamp(0.02 s, 2.5, 6)
     px, s = px per world unit of that layout; round caps and joins. Under the core a soft glow of nine strokes in
     the surface colour, 8 / 7 / 6 / 5.1 / 4.3 / 3.6 / 3 / 2.4 / 1.9 w wide at alpha .018 / .022 / .028 / .034 /
     .042 / .052 / .064 / .08 / .11 (HO_GLOW: plain strokes, no blur filter, so Canvas and SVG draw it alike); over
     the core a highlight 0.4 w wide in #dcf6ff at alpha .9.
     On a light page (KnotTalk.theme, the talk) the tube lies on paper instead: a glow of the surface colour would only
     tint the paper into a pale band, so the same nine strokes become a soft shadow under the curve, in the text colour
     at HO_SHADE times their alpha, moved down HO_DROP w; the highlight is the surface colour taken HO_TINT of the way
     to the card colour (bg), at alpha HO_HA, moved up HO_LIFT w (lit from above). */
  var HO_W = 0.76, HO_H = 0.40;
  var HO_GLOW = [[8, 0.018], [7, 0.022], [6, 0.028], [5.1, 0.034], [4.3, 0.042], [3.6, 0.052], [3, 0.064], [2.4, 0.08], [1.9, 0.11]];
  var HO_SHADE = 0.45, HO_DROP = 0.8, HO_TINT = 0.45, HO_LIFT = 0.18;
  var HO_HI = LIGHT ? hexMix(COL.surf, COL.bg, HO_TINT) : '#dcf6ff', HO_HA = LIGHT ? 0.35 : 0.9;
  var HO_FADE = 1.5, HO_MOVE = [0.3, 1.8];      // s: everything else fades over [0, 1.5]; the curve moves over [0.3, 1.8]
  function handoffFrame(W, H) {                 // screen point of world (x, y) = (ox + s x, oy - s y)
    var s = Math.min(HO_W * W / (2 * CFG.L), HO_H * H / (HANDOFF.y1 - HANDOFF.y0));
    return { s: s, ox: W / 2, oy: H / 2 + s * (HANDOFF.y0 + HANDOFF.y1) / 2 };
  }
  function handoffWidth(s) { return clamp(0.02 * s, 2.5, 6); }
  function smoother(t) { t = clamp(t, 0, 1); return t * t * t * (t * (6 * t - 15) + 10); }

  function rgba(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
  }
  function c255(v) { return Math.round(clamp(v, 0, 1) * 255); }
  function mixHex(h1, h2, t, a) {
    var p = parseInt(h1.slice(1), 16), q = parseInt(h2.slice(1), 16);
    function ch(s) { return Math.round(((p >> s) & 255) * (1 - t) + ((q >> s) & 255) * t); }
    return 'rgba(' + ch(16) + ',' + ch(8) + ',' + ch(0) + ',' + a + ')';
  }
  /* the colour t of the way from h1 to h2, as #rrggbb (the same function in basis.js: the hand-off's colours agree) */
  function hexMix(h1, h2, t) {
    var p = parseInt(h1.slice(1), 16), q = parseInt(h2.slice(1), 16), s, c, o = '#';
    for (s = 16; s >= 0; s -= 8) { c = Math.round(((p >> s) & 255) * (1 - t) + ((q >> s) & 255) * t); o += (c < 16 ? '0' : '') + c.toString(16); }
    return o;
  }
  /* The 1-D photos are taken against black: that is the experiment, on every page. On a light page each pixel is drawn
     over a light backdrop instead (the card colour taken BACK_K of the way to the line colour, so that "nothing there"
     stays apart from the true curve's white band), by its coverage: 1 - T as the renderer records it, for the truth
     the share of its sub-rays that hit the curve. The red underline still measures the error the loss sees. */
  var BACK_K = 0.7, BACK = LIGHT ? (function (h) { var n = parseInt(h.slice(1), 16); return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]; })(hexMix(COL.bg, COL.line, BACK_K)) : null;

  function injectCSS() {
    if (document.getElementById('kt-flatland-css')) return;
    var st = document.createElement('style');
    st.id = 'kt-flatland-css';
    st.textContent = [
      '.kt-sim-flatland .fl-cap{max-width:none;line-height:1.5;font-size:12px}',
      '.kt-sim-flatland .fl-cap b{font-family:var(--f-display);font-size:15px;letter-spacing:0;font-weight:600}',
      '.kt-sim-flatland .fl-cap i{display:block;font-style:normal;color:var(--mute);font-size:11px;margin-top:3px}',
      /* the readouts are not a .kt-panel on purpose: the kit folds panels away on small embedded stages */
      '.kt-sim-flatland .fl-metrics{position:absolute;z-index:5;display:flex;flex-direction:column;padding:10px 14px 9px;gap:6px;'+
        'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}',
      '.kt-sim-flatland .fl-table{display:grid;grid-template-columns:auto repeat(5,auto);gap:5px 16px;align-items:baseline}',
      '.kt-sim-flatland .fl-th{font:600 11px/1.15 var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--mute);text-align:right}',
      '.kt-sim-flatland .fl-th small{font-size:10px;letter-spacing:.02em;text-transform:none;color:var(--mute);opacity:.85}',
      '.kt-sim-flatland .fl-name{font:700 12px/1 var(--f-mono);letter-spacing:.14em}',
      '.kt-sim-flatland .fl-name.free{color:var(--err)}.kt-sim-flatland .fl-name.bound{color:var(--surf)}',
      '.kt-sim-flatland .fl-num{font:600 18px/1.1 var(--f-mono);font-variant-numeric:tabular-nums;text-align:right;color:var(--text)}',
      '.kt-sim-flatland .fl-num.ok{color:var(--ok)}.kt-sim-flatland .fl-num.err{color:var(--err)}.kt-sim-flatland .fl-num.dim{color:var(--dim)}',
      '.kt-sim-flatland .fl-off{display:none}',
      '.kt-sim-flatland .fl-metrics .kt-readout{min-height:0;border-top:1px solid var(--line);padding-top:6px}',
      '.kt-sim-flatland .fl-metrics .kt-readout .kt-val{font-size:12px;color:var(--dim)}',
      '.kt-sim-flatland .fl-ctl .kt-slider{width:190px}',
      /* centre with auto margins: left:50% would cap the panel at half the stage and wrap it */
      '.kt-sim-flatland:not(.kt-compact) .fl-ctl{left:12px;right:12px;margin:0 auto;width:fit-content;transform:none}',
      '.kt-sim-flatland.fl-mid .fl-num{font-size:15px}',
      '.kt-sim-flatland.fl-mid .fl-table{gap:4px 11px}',
      '.kt-sim-flatland.fl-mid .fl-th{font-size:10px;letter-spacing:.06em}',
      '.kt-sim-flatland.fl-mid .fl-metrics{padding:8px 11px 7px}',
      '.kt-sim-flatland .fl-handle{position:absolute;z-index:6;left:0;top:0;width:52px;height:52px;margin:-26px 0 0 -26px;border-radius:50%;touch-action:none;cursor:grab}',
      '.kt-sim-flatland .fl-handle:active{cursor:grabbing}',
      '.kt-sim-flatland.kt-embed .fl-num{font-size:16px}',
      '.kt-sim-flatland.fl-compact .fl-cap{left:10px;top:8px;font-size:11px;line-height:1.35}',
      '.kt-sim-flatland.fl-compact .fl-cap b{font-size:13px}',
      '.kt-sim-flatland .fl-cap i.fl-short,.kt-sim-flatland.fl-compact .fl-cap i.fl-long{display:none}',
      '.kt-sim-flatland.fl-compact .fl-cap i.fl-short{display:block;font-size:10px;margin-top:1px}',
      '.kt-sim-flatland.fl-compact .fl-metrics{padding:6px 10px;gap:3px;border-radius:8px}',
      '.kt-sim-flatland.fl-compact .fl-table{gap:2px 8px;grid-template-columns:auto repeat(5,1fr)}',
      '.kt-sim-flatland.fl-compact .fl-th{font-size:10px;letter-spacing:.04em}',
      '.kt-sim-flatland.fl-compact .fl-th small{display:none}',
      '.kt-sim-flatland.fl-compact .fl-name{font-size:11px;letter-spacing:.06em}',
      '.kt-sim-flatland.fl-compact .fl-num{font-size:13px}',
      '.kt-sim-flatland.fl-compact .fl-metrics .kt-readout{display:none}',
      '.kt-sim-flatland.fl-compact:not(.kt-compact) .fl-ctl{left:8px;right:8px;bottom:8px;padding:6px 8px;gap:5px 8px;justify-content:center;width:auto}',
      '.kt-sim-flatland.fl-compact .fl-ctl .kt-slider{width:150px}',
      /* scene `handoff`: the module's own chrome fades with the picture (--fl-keep, set every frame) and stops taking input */
      '.kt-sim-flatland.fl-handoff .fl-cap,.kt-sim-flatland.fl-handoff .fl-metrics,.kt-sim-flatland.fl-handoff .fl-ctl,' +
        '.kt-sim-flatland.fl-handoff .kt-gear{opacity:var(--fl-keep,1);pointer-events:none}',
      '.kt-sim-flatland.fl-handoff .fl-handle{pointer-events:none}',
      '.kt-sim-flatland.fl-gone .fl-cap,.kt-sim-flatland.fl-gone .fl-metrics,.kt-sim-flatland.fl-gone .fl-ctl,' +
        '.kt-sim-flatland.fl-gone .kt-gear{visibility:hidden}',
      /* a light page (the talk): the readouts as one of the kit's glass panels, its numbers in the page's type */
      ':where(:root[data-theme="light"]) .kt-sim-flatland .fl-metrics{border:0;border-radius:1.2rem;padding:12px 16px 10px;' +
        'box-shadow:var(--glass-edge),var(--glass-drop);backdrop-filter:blur(22px) saturate(180%);-webkit-backdrop-filter:blur(22px) saturate(180%)}',
      ':where(:root[data-theme="light"]) .kt-sim-flatland.fl-mid .fl-metrics{padding:10px 14px 8px}',
      ':where(:root[data-theme="light"]) .kt-sim-flatland.fl-compact .fl-metrics{padding:7px 12px;border-radius:1rem}',
      ':where(:root[data-theme="light"]) .kt-sim-flatland :is(.fl-th,.fl-name,.fl-num){font-family:var(--f-ui)}',
      ':where(:root[data-theme="light"]) .kt-sim-flatland .fl-th{letter-spacing:.04em}',
      ':where(:root[data-theme="light"]) .kt-sim-flatland .fl-name{letter-spacing:.02em}',
      /* Hebrew (talk/i18n.js sets <html lang>) has no capitals and is not letter-spaced */
      ':root[lang="he"] .kt-sim-flatland :is(.fl-th,.fl-name){letter-spacing:0;text-transform:none}'
    ].join('\n');
    document.head.appendChild(st);
  }

  KT.register('flatland', {
    title: 'Free vs. bound',
    blurb: 'A real differentiable-rendering experiment in a 2D world, running live. Same photos, same start, ' +
           'same optimizer: free Gaussians (3DGS-like) against Gaussians derived from one B-spline curve.',
    tool: 'Canvas 2D · live optimizer',
    scenes: [
      { id: 'free', label: 'Free Gaussians' },
      { id: 'novel', label: 'A new viewpoint' },
      { id: 'rays', label: 'Depth ambiguity' },
      { id: 'bound', label: 'Bound to a curve' },
      { id: 'versus', label: 'Free vs. bound' },
      { id: 'handoff', label: 'Only the curve' }
    ],

    create: function (env) {
      var ui = env.ui, rootEl = env.root;
      injectCSS();
      var MONO = (getComputedStyle(rootEl).getPropertyValue('--f-mono') || '').trim() || 'ui-monospace, Menlo, monospace';

      /* ---------- state ---------- */
      var opt = env.opts || {};                    // ?seed=N&views=K on the lab page
      var seed = (opt.seed != null && isFinite(+opt.seed)) ? +opt.seed : SEED;
      var K = clamp(Math.round(+opt.views) || K_DEF, K_MIN, K_MAX), userK = K, userTheta = null, E = null;
      var scene = null, model = 'free', sel = 'novel', selUser = false;
      var running = true, acc = 0, pulse = false, hl = false, raysOn = false;
      var Wd = 0, Hd = 0, compact = false, views = [], stripBox = null, legendY = 0, raysBox = null;
      var dirty = true, novelDirty = false, tMet = 0, tArc = 0, tProbe = 0, now = 0;
      var n = CFG.G;
      var MET = {
        free: { train: NaN, novel: NaN, geo: NaN, fl: 0, dist: new Float64Array(n), vis: new Float64Array(n) },
        bound: { train: NaN, novel: NaN, geo: NaN, fl: 0, dist: new Float64Array(n), vis: new Float64Array(n) }
      };
      var IMG = { free: new Float32Array(CFG.W * 3), bound: new Float32Array(CFG.W * 3) }, ROWP = { free: NaN, bound: NaN };
      // the models' coverage of the shown strip, for drawing it over the light backdrop (BACK); the truth's is E.cov
      var CVR = BACK && { free: new Float32Array(CFG.W), bound: new Float32Array(CFG.W) };
      var arcCam = [], arcGT = [], ARC = { free: new Float64Array(ARC_N), bound: new Float64Array(ARC_N) }, arcStep = -1;
      var probe = null, probeStep = -1, slideTrain = NaN;
      var drawTab = null;                // basis table for drawing the bound curve
      /* scene `handoff`: t = seconds since it started (time-based, paused off screen like every loop), keep = how
         much of everything else is still drawn, k = how far the curve has come (pane -> hand-off frame, plain ->
         hand-off style). from = the bound curve's control points when it started; goal = the hand-off curve at the
         same 241 parameters that drawTab uses. */
      var HO = { on: false, t: 0, keep: 1, k: 0, done: false, back: null, from: new Float64Array(2 * CFG.M), goal: new Float64Array(2 * 241) };
      (function () {
        var us = [], i, q, Uh = NURBS.clampedKnots(HANDOFF.P.length, 3);
        for (i = 0; i <= 240; i++) us.push(i / 240);
        drawTab = NURBS.axisTable(NURBS.clampedKnots(CFG.M, 3), CFG.M, 3, us);
        for (i = 0; i <= 240; i++) { q = NURBS.curvePoint(HANDOFF.P, null, 3, Uh, us[i]); HO.goal[2 * i] = q[0]; HO.goal[2 * i + 1] = q[1]; }
        for (i = 0; i < ARC_N; i++) { arcCam.push(makeCamera(-CFG.thNovel + 2 * CFG.thNovel * i / (ARC_N - 1))); arcGT.push(null); }
      })();

      /* ---------- DOM ---------- */
      var cvs = ui.canvas(rootEl), ctx = cvs.ctx;
      var cap = ui.el('div', 'kt-caption fl-cap'); rootEl.appendChild(cap);
      var met = ui.el('div', 'fl-metrics'); rootEl.appendChild(met);
      var table = ui.el('div', 'fl-table'); met.appendChild(table);
      // the column heads, each with its unit: one template apiece
      var HEAD = [T('train <small>dB</small>'), T('held-out <small>dB</small>'), T('geom err <small>cm</small>'),
                  T('floaters <small>>{d} cm</small>', { d: Math.round(CFG.tau * 100) }), T('params')];
      var NAME = { free: T('free'), bound: T('bound') };       // the two models, as the table and the control name them
      var cells = { free: {}, bound: {} }, rowEls = { free: [], bound: [] };
      table.appendChild(ui.el('span', 'fl-th', ''));
      HEAD.forEach(function (h) { table.appendChild(ui.el('span', 'fl-th', h)); });
      ['free', 'bound'].forEach(function (kind) {
        var nm = ui.el('span', 'fl-name ' + kind, NAME[kind]); table.appendChild(nm); rowEls[kind].push(nm);
        ['train', 'novel', 'geo', 'fl', 'par'].forEach(function (k) {
          var c = ui.el('span', 'fl-num', '–'); table.appendChild(c); cells[kind][k] = c; rowEls[kind].push(c);
        });
      });
      var stepOut = ui.readout(met, { label: T('step') });

      var ctl = ui.panel(rootEl, { pos: 'bottom' }); ctl.classList.add('fl-ctl');
      var segModel = ui.seg(ctl, { label: T('model'), value: model,
        options: [{ v: 'free', label: NAME.free }, { v: 'bound', label: NAME.bound }, { v: 'both', label: T('both') }],
        onChange: function (v) { model = v; showRows(); layout(); dirty = true; } });
      var sldK = ui.slider(ctl, { label: T('views'), min: K_MIN, max: K_MAX, step: 1, value: K,
        onInput: function (v) { userK = v; if (v !== K) { K = v; reset(); setRun(true); } } });
      ui.button(ctl, { label: T('reset'), onClick: function () { reset(); setRun(true); } });
      var tglRun = ui.toggle(ctl, { label: T('run'), value: true, onChange: function (v) { running = v; } });
      function setRun(v) { running = v; tglRun.set(v); }
      /* the deck owns arrows and space: never leave keyboard focus on a control after a pointer click */
      function dropFocus() { var a = document.activeElement; if (a && a !== document.body && ctl.contains(a) && a.blur) a.blur(); }
      ctl.addEventListener('pointerup', function () { setTimeout(dropFocus, 0); });
      sldK.input.addEventListener('change', dropFocus);

      var handles = [0, 1].map(function (vi) {
        var h = ui.el('div', 'fl-handle'); h.title = T('drag the held-out camera'); rootEl.appendChild(h);
        var drag = false;
        function move(e) {
          var v = views[vi], r = rootEl.getBoundingClientRect();
          if (!v) return;
          var wx = (e.clientX - r.left - v.ox) / v.s, wy = (v.oy - (e.clientY - r.top)) / v.s;
          userTheta = clamp(Math.atan2(wx, Math.max(0.05, wy - CFG.cy)), -CFG.thNovel, CFG.thNovel);
          novelDirty = true;                    // applied once per frame (re-rendering the true strip costs a few ms)
        }
        h.addEventListener('pointerdown', function (e) {
          drag = true; pulse = false; sel = 'novel'; selUser = true; dirty = true;
          try { h.setPointerCapture(e.pointerId); } catch (x) {}
          e.preventDefault();
        });
        h.addEventListener('pointermove', function (e) { if (drag) move(e); });
        function end(e) { drag = false; try { h.releasePointerCapture(e.pointerId); } catch (x) {} }
        h.addEventListener('pointerup', end); h.addEventListener('pointercancel', end);
        return h;
      });

      /* click a training camera to see its strip */
      var downAt = null;
      function onDown(e) { downAt = [e.clientX, e.clientY]; }
      function onUp(e) {
        if (HO.on || !downAt || Math.abs(e.clientX - downAt[0]) + Math.abs(e.clientY - downAt[1]) > 8) { downAt = null; return; }
        downAt = null;
        var r = rootEl.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, best = -1, bd = 26 * 26, vi, k, v, d;
        for (vi = 0; vi < views.length; vi++) for (k = 0; k < E.K; k++) {
          v = views[vi]; d = Math.pow(v.ox + E.cams[k].px * v.s - x, 2) + Math.pow(v.oy - E.cams[k].py * v.s - y, 2);
          if (d < bd) { bd = d; best = k; }
        }
        if (best >= 0) { sel = best; selUser = true; dirty = true; }
      }
      cvs.cv.addEventListener('pointerdown', onDown);
      cvs.cv.addEventListener('pointerup', onUp);

      /* ---------- experiment control ---------- */
      function reset() {
        E = makeExperiment({ K: K, seed: seed, rng: KT.rng, novel: userTheta, cover: !!BACK });
        acc = 0; probe = null; probeStep = -1; arcStep = -1; slideTrain = NaN;
        ARC.free.fill(NaN); ARC.bound.fill(NaN);
        if (typeof sel === 'number' && sel >= K) sel = K - 1;
        sldK.set(K); dirty = true;
      }
      function trainSel() { return Math.floor((K - 1) / 2); }
      function shown(kind) { return model === 'both' || model === kind; }
      function showRows() {
        ['free', 'bound'].forEach(function (kind) {
          rowEls[kind].forEach(function (c) { c.classList.toggle('fl-off', !shown(kind)); });
        });
      }

      var CAPS = {
        free: function () { return T('<b>Free Gaussians</b><br>{g} splats, each with its own position, shape, colour, opacity — fitted to {k} photos.', { g: CFG.G, k: K }); },
        novel: function () { return T('<b>Now a camera it never trained on</b><br>Same run. Drag the held-out camera along the arc.'); },
        rays: function () { return T('<b>Depth is not observed</b><br>With {k} views many floaters matter to one camera only: slide one along that ray and the photo barely changes.', { k: K }); },
        bound: function () { return T('<b>Bound Gaussians</b><br>One cubic B-spline, {m} control points. Every splat is derived from the curve — none can leave it.', { m: CFG.M }); },
        versus: function () { return T('<b>Same photos, same start, same steps</b><br>Only the parametrization differs.'); }
      };
      CAPS.handoff = CAPS.versus;                  // it fades out with the picture it belongs to
      function paintCaption() {
        var f = CAPS[scene] || CAPS.free;
        cap.innerHTML = f() +
          '<i class="fl-long">' + T('start: both from the same {n} noisy points (+) · loss: MSE on the training strips · Adam, {steps} steps', { n: CFG.Ns, steps: CFG.maxSteps }) + '</i>' +
          '<i class="fl-short">' + T('same {n} noisy start points (+) · same loss · same {steps} steps', { n: CFG.Ns, steps: CFG.maxSteps }) + '</i>';
      }

      function setScene(id) {
        var S = SCENES[id], back;
        // idempotent: re-selecting a scene never restarts it. The hand-off is the exception: picked again, it replays.
        if (!S || (id === scene && !S.handoff)) return;
        // stepping back out of the hand-off to the scene it started from carries on with the same run (no reset)
        back = HO.on && id === HO.back;
        if (S.handoff) HO.back = scene === 'handoff' ? HO.back : scene;
        scene = id;
        model = S.handoff && (model === 'bound' || model === 'both') ? model : S.model; segModel.set(model); showRows();
        pulse = S.pulse; hl = S.hl; raysOn = S.rays; selUser = false;
        if (S.k === 'min') K = K_MIN; else if (S.k === 'user') K = userK;
        if (S.reset && !back) reset();
        if (S.sel) sel = S.sel === 'train' ? trainSel() : S.sel === 'auto' ? 0 : 'novel';
        if (typeof sel === 'number' && sel >= E.K) sel = E.K - 1;
        setRun(!S.handoff);                         // the hand-off stops training: the curve it carries away is fixed
        handoff(!!S.handoff);
        paintCaption(); layout(); dirty = true;
      }
      /* start (or restart) / end the hand-off */
      function handoff(on) {
        HO.on = on; HO.t = 0; HO.keep = 1; HO.k = 0; HO.done = false;
        if (on) for (var j = 0; j < 2 * CFG.M; j++) HO.from[j] = E.bound.th[j];
        rootEl.classList.toggle('fl-handoff', on); rootEl.classList.remove('fl-gone');
        rootEl.style.setProperty('--fl-keep', '1');
      }

      /* ---------- layout ---------- */
      function fitView(x, y, w, h, kind) {
        var padT = 26, padB = 4, s = Math.min(w / (VIEW.x1 - VIEW.x0), (h - padT - padB) / (VIEW.y1 - VIEW.y0));
        return { kind: kind, x: x, y: y, w: w, h: h, s: s,
                 ox: x + w / 2 - 0.5 * (VIEW.x0 + VIEW.x1) * s,
                 oy: y + padT + (h - padT - padB - (VIEW.y1 - VIEW.y0) * s) / 2 + VIEW.y1 * s };
      }
      function layout() {
        if (!Wd || !Hd) return;
        var w = Wd, h = Hd, rr = rootEl.getBoundingClientRect(), pad, cr, mr, botLimit, capR, two = model === 'both';
        // on small embedded stages the kit (.kt-compact) folds the control panel behind a button
        var folded = rootEl.classList.contains('kt-compact'), ctlTop;
        compact = w < 760 || h < 520;
        rootEl.classList.toggle('fl-compact', compact);
        rootEl.classList.toggle('fl-mid', !compact && w < 1000);
        pad = compact ? 8 : 14;
        cr = ctl.getBoundingClientRect();
        ctlTop = folded || !cr.height ? h - 2 : cr.top - rr.top;
        if (compact) {
          met.style.cssText = 'left:8px;right:8px;top:auto;bottom:' + Math.round(h - ctlTop + 6) + 'px';
          cap.style.maxWidth = (w - (folded ? 112 : 20)) + 'px';
        } else {
          met.style.cssText = 'top:12px;right:12px;left:auto;bottom:auto';
        }
        mr = met.getBoundingClientRect();
        if (!compact) cap.style.maxWidth = Math.max(180, mr.left - rr.left - 34) + 'px';
        capR = cap.getBoundingClientRect();
        botLimit = (compact ? mr.top - rr.top : ctlTop) - 8;

        // image strips: header + one row per shown model + ground truth
        var rows = two ? 3 : 2, rowH = compact ? 11 : clamp(Math.round(h * 0.024), 14, 26), errH = compact ? 3 : 5, gap = compact ? 4 : 7;
        var headH = compact ? 15 : 20, sH = headH + rowH + gap + (rows - 1) * (rowH + errH + gap);
        var labW = compact ? 44 : 78, valW = compact ? 54 : 78;
        stripBox = { x: pad, y: botLimit - sH, w: w - 2 * pad, h: sH, rowH: rowH, errH: errH, gap: gap, headH: headH, labW: labW, valW: valW };
        legendY = compact ? 0 : stripBox.y - 16;
        var wBot = stripBox.y - (compact ? 6 : 34);

        // world view(s): slide the top edge down until the camera arc clears the caption and the
        // metrics panel (they may sit over the empty corners), with 1 pane or 2 (side by side / stacked)
        var obs = [{ l: capR.left - rr.left, r: capR.right - rr.left, b: capR.bottom - rr.top }], top, best = null;
        if (!compact) obs.push({ l: mr.left - rr.left, r: mr.right - rr.left, b: mr.bottom - rr.top });
        raysBox = null;
        if (raysOn) {              // the measured numbers of the `rays` scene sit under the caption
          raysBox = { x: pad + 2, y: capR.bottom - rr.top + (compact ? 6 : 14), w: compact ? 250 : 250, h: compact ? 3 * 15 : 4 * 36 };
          obs.push({ l: raysBox.x, r: raysBox.x + raysBox.w, b: raysBox.y + raysBox.h - (compact ? 14 : 30) });
        }
        function clear(vs) {
          return vs.every(function (v) {
            var half = CFG.R * Math.sin(CFG.thNovel) * v.s + 34;
            return obs.every(function (o) {
              var dx = o.l > v.ox ? o.l - v.ox : o.r < v.ox ? v.ox - o.r : 0, u;
              if (dx >= half) return true;                      // entirely beside the arc
              u = Math.min(dx / v.s, CFG.R * 0.999);
              return o.b + 36 <= v.oy - (CFG.cy + Math.sqrt(CFG.R * CFG.R - u * u)) * v.s;
            });
          });
        }
        function arrange(type, top) {
          var hh = wBot - top;
          if (type === 0) return [fitView(0, top, w, hh, model)];
          if (type === 1) return [fitView(0, top, w / 2, hh, 'free'), fitView(w / 2, top, w / 2, hh, 'bound')];
          return [fitView(0, top, w, hh / 2, 'free'), fitView(0, top + hh / 2, w, hh / 2, 'bound')];
        }
        (two ? [1, 2] : [0]).forEach(function (type) {
          var vs;
          for (top = 2; top < wBot - 60; top += 6) {
            vs = arrange(type, top);
            if (clear(vs)) { if (!best || vs[0].s > best[0].s * 1.02) best = vs; break; }
          }
        });
        views = best || [fitView(0, Math.max(0, wBot - 80), w, 80, two ? 'free' : model)];
        handles[1].style.display = views.length > 1 ? '' : 'none';
      }
      function resize(w, h) {
        Wd = w; Hd = h; cvs.fit(); layout(); dirty = true; draw();
      }

      /* ---------- measurements ---------- */
      function selCam() {
        if (sel === 'novel') return { cam: E.novel, tgt: E.novelTgt, novel: true, cov: E.cov && E.cov.novel };
        return { cam: E.cams[sel], tgt: E.tgts[sel], novel: false, cov: E.cov && E.cov.train[sel] };
      }
      function fmt(v, d) { return v === v ? v.toFixed(d) : '–'; }
      function updateMetrics() {
        ['free', 'bound'].forEach(function (kind) {
          var m = E[kind], R = MET[kind], g;
          derive(m);
          R.train = psnr(m, E.cams, E.tgts);
          R.novel = psnr(m, [E.novel], [E.novelTgt]);
          g = geometry(m, R.dist, R.vis); R.geo = g.err; R.fl = g.floaters;
        });
        var both = model === 'both';
        ['free', 'bound'].forEach(function (kind) {
          var R = MET[kind], O = MET[kind === 'free' ? 'bound' : 'free'], c = cells[kind];
          c.train.textContent = fmt(R.train, 1);
          c.novel.textContent = fmt(R.novel, 1); c.novel.className = 'fl-num' + (both && R.novel > O.novel + 0.5 ? ' ok' : '');
          c.geo.textContent = fmt(R.geo * 100, 1); c.geo.className = 'fl-num' + (both && R.geo < O.geo * 0.8 ? ' ok' : '');
          c.fl.textContent = R.fl; c.fl.className = 'fl-num ' + (R.fl > 0 ? 'err' : 'ok');
          c.par.textContent = E[kind].th.length; c.par.className = 'fl-num dim';
        });
        showRows();
        var sv = { step: E.free.step, max: CFG.maxSteps, k: E.K, w: CFG.W };
        stepOut.set(E.free.step >= CFG.maxSteps ? T('{step} / {max} · {k} views × {w} px · done', sv) :
                    running ? T('{step} / {max} · {k} views × {w} px', sv) : T('{step} / {max} · {k} views × {w} px · paused', sv));
      }
      /* held-out PSNR from 33 viewpoints along the arc — drawn as the colour of the arc itself */
      function updateArc() {
        var i, made = 0, done = true;
        for (i = 0; i < ARC_N; i++) {
          if (!arcGT[i]) { if (made >= 6) { done = false; continue; } arcGT[i] = renderGT(arcCam[i]); made++; }
          ARC.free[i] = renderCam(arcCam[i], E.free.gs, null, arcGT[i], null, 0);
          ARC.bound[i] = renderCam(arcCam[i], E.bound.gs, null, arcGT[i], null, 0);
          ARC.free[i] = psnrOf(ARC.free[i] / (CFG.W * 3)); ARC.bound[i] = psnrOf(ARC.bound[i] / (CFG.W * 3));
        }
        if (done) arcStep = E.free.step;
      }
      /* `rays` scene: measure the depth ambiguity on the free model (see rayProbe) */
      function updateProbe() {
        var P = rayProbe(E, E.free, MET.free.dist, MET.free.vis, SLIDE), L = P.list, tots, med, one, ex, al, ac;
        probeStep = E.free.step;
        if (!L.length) { probe = null; return; }
        function median(a) { a = a.slice().sort(function (x, y) { return x - y; }); return a[a.length >> 1]; }
        tots = L.map(function (r) { return r.tot; }); med = median(tots);
        one = L.filter(function (r) { return r.share >= 0.9; });
        // exemplars: the floaters that matter (influence above the median) and are most free to slide
        ex = one.filter(function (r) { return r.tot >= med; }).sort(function (a, b) { return b.along - a.along; }).slice(0, 2);
        al = median(L.map(function (r) { return r.along; })); ac = median(L.map(function (r) { return r.across; }));
        probe = { base: P.base, list: L, one: one, ex: ex, medAlong: al, medAcross: ac };
        if (ex.length && !selUser) sel = ex[0].cam;
      }
      /* strips of the selected camera; in the `rays` scene the free strip is rendered with the
         exemplar floaters displaced along their rays, so the audience sees it (not) change */
      function renderStrips() {
        var S = selCam(), sse, saved = null, off = 0, i, gsF = E.free.gs;
        if (raysOn && probe && probe.ex.length && shown('free')) {
          off = SLIDE * Math.sin(now * 2 * Math.PI / 3.6);
          saved = probe.ex.map(function (r) { return shiftGaussian(gsF, r.i, E.cams[r.cam], off, 0); });
          slideTrain = psnr(E.free, E.cams, E.tgts);
        } else slideTrain = NaN;
        COV = CVR && CVR.free;
        sse = renderCam(S.cam, gsF, IMG.free, S.tgt, null, 0); ROWP.free = psnrOf(sse / (CFG.W * 3));
        if (saved) for (i = 0; i < saved.length; i++) unshift(gsF, probe.ex[i].i, saved[i]);
        COV = CVR && CVR.bound;
        sse = renderCam(S.cam, E.bound.gs, IMG.bound, S.tgt, null, 0); ROWP.bound = psnrOf(sse / (CFG.W * 3));
        COV = null;
        return off;
      }

      /* ---------- drawing ---------- */
      function font(px, wt) { return (wt || 500) + ' ' + px + 'px ' + MONO; }
      /* the canvas labels are set in capitals; Hebrew has none, and there a unit inside a label keeps its case (dB, px, cm) */
      function caps(s) { return KT.rtl ? s : s.toUpperCase(); }
      function drawCamera(v, cam, on, label, isNovel) {
        var X = v.ox + cam.px * v.s, Y = v.oy - cam.py * v.s, sz = compact ? 8 : 12, t = CFG.tanHalf * 0.75;
        // screen-space forward / image axis (y flips)
        var fx = cam.dx, fy = -cam.dy, ax = cam.rx, ay = -cam.ry, bx = X - fx * sz, by = Y - fy * sz;
        ctx.beginPath();
        ctx.moveTo(X, Y); ctx.lineTo(bx - ax * sz * t, by - ay * sz * t); ctx.lineTo(bx + ax * sz * t, by + ay * sz * t); ctx.closePath();
        ctx.fillStyle = on ? rgba(COL.text, 0.9) : COL.bg; ctx.fill();
        ctx.lineWidth = on ? 1.6 : 1.1; ctx.strokeStyle = on || isNovel ? COL.text : COL.dim; ctx.stroke();
        if (isNovel) {
          ctx.beginPath(); ctx.arc(X, Y, compact ? 10 : 14, 0, 6.2832); ctx.lineWidth = 1.2; ctx.strokeStyle = rgba(COL.text, 0.75); ctx.stroke();
          if (pulse) {
            var ph = (now % 1.5) / 1.5;
            ctx.beginPath(); ctx.arc(X, Y, 14 + 22 * ph, 0, 6.2832); ctx.lineWidth = 2; ctx.strokeStyle = rgba(COL.text, 0.7 * (1 - ph)); ctx.stroke();
          }
        }
        ctx.font = font(11, 600); ctx.textBaseline = 'alphabetic';
        var tw = ctx.measureText(label).width, lx = X - fx * (sz + 16) - tw / 2, ly = Y - fy * (sz + 16) + 4;
        if (isNovel) { lx = X - tw / 2; ly = Y - (compact ? 16 : 22); }
        lx = clamp(lx, v.x + 3, v.x + v.w - tw - 3); ly = Math.max(ly, v.y + 11);
        ctx.fillStyle = on || isNovel ? COL.text : COL.dim; ctx.fillText(label, lx, ly);
      }
      function drawFrustum(v, cam, a, fill) {
        var X = v.ox + cam.px * v.s, Y = v.oy - cam.py * v.s, k, vx, vy, t, pts = [];
        for (k = -1; k <= 1; k += 2) {
          vx = cam.dx + k * CFG.tanHalf * cam.rx; vy = cam.dy + k * CFG.tanHalf * cam.ry;
          t = vy < -0.05 ? (cam.py - VIEW.y0) / -vy : 6;
          t = Math.min(t, 6);
          pts.push([v.ox + (cam.px + t * vx) * v.s, v.oy - (cam.py + t * vy) * v.s]);
        }
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); ctx.lineTo(X, Y); ctx.lineTo(pts[1][0], pts[1][1]);
        if (fill) { ctx.fillStyle = rgba(COL.text, fill); ctx.fill(); }
        ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.dim, a); ctx.stroke();
      }
      function ellipseOf(gs, i, v, k, out, minR) {
        var a = gs.a[i], b = gs.b[i], c = gs.c[i], tr = 0.5 * (a + c), dd = Math.sqrt(Math.max(0, tr * tr - (a * c - b * b)));
        minR = minR || 1.3;
        out.x = v.ox + gs.mx[i] * v.s; out.y = v.oy - gs.my[i] * v.s;
        out.r1 = Math.max(minR, k * Math.sqrt(tr + dd) * v.s); out.r2 = Math.max(minR, k * Math.sqrt(Math.max(tr - dd, 0)) * v.s);
        out.rot = -0.5 * Math.atan2(2 * b, a - c);
        return out;
      }
      var _e = {};
      /* 1-sigma outline + soft fill in the learned colour. The bound Gaussians are only ~1 px thick
         along the normal at laptop scale, so their drawn half-thickness has a floor (minR) — a
         legibility aid for the picture only; the renderer uses the true covariance. */
      function drawGaussians(v, kind) {
        var gs = E[kind].gs, R = MET[kind], i, fl, hid, col, minR = kind === 'bound' ? (compact ? 2.2 : 3) : 1.3;
        for (i = 0; i < gs.n; i++) {
          hid = R.vis[i] < CFG.visW; fl = !hid && R.dist[i] > CFG.tau;
          col = c255(gs.cr[i]) + ',' + c255(gs.cg[i]) + ',' + c255(gs.cb[i]);
          if (kind === 'free') {
            ellipseOf(gs, i, v, 2, _e, minR);
            ctx.beginPath(); ctx.ellipse(_e.x, _e.y, _e.r1, _e.r2, _e.rot, 0, 6.2832);
            ctx.fillStyle = 'rgba(' + col + ',' + (hid ? 0.05 : 0.10 * gs.op[i]).toFixed(3) + ')'; ctx.fill();   // soft 2-sigma skirt
          }
          ellipseOf(gs, i, v, 1, _e, minR);
          ctx.beginPath(); ctx.ellipse(_e.x, _e.y, _e.r1, _e.r2, _e.rot, 0, 6.2832);
          ctx.fillStyle = 'rgba(' + col + ',' + (hid ? 0.10 : 0.2 + 0.6 * gs.op[i]).toFixed(3) + ')'; ctx.fill();
          if (fl) { ctx.lineWidth = hl ? 2 : 1.5; ctx.strokeStyle = COL.err; }
          else if (hid) { ctx.lineWidth = 0.8; ctx.strokeStyle = rgba(COL.mute, 0.6); }
          else { ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.splat, 0.85); }
          ctx.stroke();
          if (fl && hl) {                      // halo, so floaters read from the back of the room
            ctx.beginPath(); ctx.arc(_e.x, _e.y, Math.max(_e.r1, 5) + 5, 0, 6.2832);
            ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.err, 0.45); ctx.stroke();
          }
        }
      }
      function drawBoundCurve(v) {
        var th = E.bound.th, M = CFG.M, T = drawTab, i, r, j, x, y, N;
        // control polygon
        ctx.beginPath();
        for (j = 0; j < M; j++) { x = v.ox + th[2 * j] * v.s; y = v.oy - th[2 * j + 1] * v.s; if (j) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.ctrl, 0.45); ctx.stroke();
        if (HO.on) return;                   // the hand-off draws the curve itself, on top of everything (drawHandoff)
        // the curve S(u)
        ctx.beginPath();
        for (i = 0; i < T.n; i++) {
          x = y = 0;
          for (r = 0; r < 4; r++) { j = T.span[i] - 3 + r; N = T.N[i * 4 + r]; x += N * th[2 * j]; y += N * th[2 * j + 1]; }
          x = v.ox + x * v.s; y = v.oy - y * v.s;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
        }
        ctx.lineWidth = compact ? 1.3 : 1.6; ctx.strokeStyle = COL.surf; ctx.lineJoin = 'round'; ctx.stroke();
      }
      function drawControlPoints(v) {
        var th = E.bound.th, j, r = compact ? 1.8 : 2.6;
        ctx.fillStyle = COL.ctrl;
        for (j = 0; j < CFG.M; j++) { ctx.beginPath(); ctx.arc(v.ox + th[2 * j] * v.s, v.oy - th[2 * j + 1] * v.s, r, 0, 6.2832); ctx.fill(); }
      }
      function drawArc(v, kind) {
        var A = ARC[kind], i, a0, a1, t, cx = v.ox, cy = v.oy - CFG.cy * v.s, r = CFG.R * v.s;
        // canvas angle of a camera at arc angle th (measured from straight up): th - 90deg
        ctx.lineCap = 'butt';
        for (i = 0; i < ARC_N; i++) {
          a0 = -CFG.thNovel + 2 * CFG.thNovel * (i - 0.5) / (ARC_N - 1); a1 = a0 + 2 * CFG.thNovel / (ARC_N - 1);
          a0 = Math.max(a0, -CFG.thNovel); a1 = Math.min(a1, CFG.thNovel);
          ctx.beginPath(); ctx.arc(cx, cy, r, a0 - Math.PI / 2, a1 - Math.PI / 2 + 0.004);
          if (A[i] === A[i]) {
            t = clamp((A[i] - ARC_LO) / (ARC_HI - ARC_LO), 0, 1);
            ctx.lineWidth = compact ? 2.5 : 4; ctx.strokeStyle = mixHex(COL.err, COL.ok, t, 0.85);
          } else { ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.mute, 0.5); }
          ctx.stroke();
        }
      }
      function drawRays(v, off) {
        var gs = E.free.gs, i, r, cam, X, Y, CX, CY, dx, dy, l, t, k, P = probe, o;
        if (!P) return;
        // every floater that matters to ONE camera only: its single ray
        ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.err, 0.28);
        for (i = 0; i < P.one.length; i++) {
          r = P.one[i]; cam = E.cams[r.cam];
          CX = v.ox + cam.px * v.s; CY = v.oy - cam.py * v.s; X = v.ox + gs.mx[r.i] * v.s; Y = v.oy - gs.my[r.i] * v.s;
          ctx.beginPath(); ctx.moveTo(CX, CY); ctx.lineTo(X, Y); ctx.stroke();
        }
        for (i = 0; i < P.ex.length; i++) {
          r = P.ex[i]; cam = E.cams[r.cam];
          CX = v.ox + cam.px * v.s; CY = v.oy - cam.py * v.s; X = v.ox + gs.mx[r.i] * v.s; Y = v.oy - gs.my[r.i] * v.s;
          dx = X - CX; dy = Y - CY; l = Math.sqrt(dx * dx + dy * dy); dx /= l; dy /= l;
          t = (v.oy - VIEW.y0 * v.s - CY) / Math.max(dy, 0.05);          // down to the bottom of the view
          ctx.beginPath(); ctx.moveTo(CX, CY); ctx.lineTo(CX + dx * t, CY + dy * t);
          ctx.lineWidth = 1.6; ctx.strokeStyle = rgba(COL.err, 0.95); ctx.stroke();
          // the other cameras' sight lines to the same Gaussian (they hardly see it)
          ctx.setLineDash([3, 5]); ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.dim, 0.5);
          for (k = 0; k < E.K; k++) if (k !== r.cam) {
            ctx.beginPath(); ctx.moveTo(v.ox + E.cams[k].px * v.s, v.oy - E.cams[k].py * v.s); ctx.lineTo(X, Y); ctx.stroke();
          }
          ctx.setLineDash([]);
          // the slide range, and the ghost at its current offset
          l = SLIDE * v.s;
          ctx.beginPath(); ctx.moveTo(X - dx * l, Y - dy * l); ctx.lineTo(X + dx * l, Y + dy * l);
          ctx.lineWidth = 5; ctx.strokeStyle = rgba(COL.text, 0.28); ctx.lineCap = 'round'; ctx.stroke(); ctx.lineCap = 'butt';
          o = shiftGaussian(gs, r.i, cam, off, 0);
          ellipseOf(gs, r.i, v, 1, _e); unshift(gs, r.i, o);
          ctx.beginPath(); ctx.ellipse(_e.x, _e.y, Math.max(_e.r1, 4), Math.max(_e.r2, 4), _e.rot, 0, 6.2832);
          ctx.setLineDash([4, 3]); ctx.lineWidth = 1.6; ctx.strokeStyle = COL.text; ctx.stroke(); ctx.setLineDash([]);
        }
      }
      /* the measured numbers of the `rays` scene */
      function drawRaysInfo() {
        var P = probe, B = raysBox, i, lines = [], cm = Math.round(SLIDE * 100), base;
        if (!P || !B) return;
        base = P.base.toFixed(1);
        ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
        if (compact) {
          if (P.ex.length) {
            lines.push([T('ALONG RAY ±{cm} CM', { cm: cm }), base + ' → ' + P.ex[0].along.toFixed(1) + ' dB', COL.text]);
            lines.push([T('ACROSS RAY'), base + ' → ' + P.ex[0].across.toFixed(1) + ' dB', COL.err]);
          }
          lines.push([T('ONE CAMERA ONLY'), T('{a} of {b} floaters', { a: P.one.length, b: P.list.length }), COL.dim]);
          var col = 118;                     // where the values start; a translated name that is longer moves them out
          ctx.font = font(11, 600);
          if (KT.rtl) for (i = 0; i < lines.length; i++) col = Math.max(col, ctx.measureText(lines[i][0]).width + 10);
          for (i = 0; i < lines.length; i++) {
            ctx.font = font(11, 600); ctx.fillStyle = COL.mute; ctx.fillText(lines[i][0], B.x, B.y + 11 + i * 15);
            ctx.fillStyle = lines[i][2]; ctx.fillText(lines[i][1], B.x + col, B.y + 11 + i * 15);
          }
          return;
        }
        if (P.ex.length) {
          lines.push([T('slide ±{cm} cm along its ray', { cm: cm }), T('train {a} → {b} dB', { a: base, b: P.ex[0].along.toFixed(1) }), COL.text]);
          lines.push([T('same slide across the ray'), T('train {a} → {b} dB', { a: base, b: P.ex[0].across.toFixed(1) }), COL.err]);
        }
        lines.push([T('all {n} floaters, median', { n: P.list.length }), T('{a} along · {b} across', { a: P.medAlong.toFixed(1), b: P.medAcross.toFixed(1) }), COL.dim]);
        lines.push([T('matter to one camera only'), T('{a} of {b} floaters', { a: P.one.length, b: P.list.length }), COL.dim]);
        for (i = 0; i < lines.length; i++) {
          ctx.font = font(11, 600); ctx.fillStyle = COL.mute; ctx.fillText(caps(lines[i][0]), B.x, B.y + 11 + i * 36);
          ctx.font = font(13, 600); ctx.fillStyle = lines[i][2]; ctx.fillText(lines[i][1], B.x, B.y + 27 + i * 36);
        }
      }
      function drawView(v, off) {
        var kind = v.kind, k, i, x, y;
        ctx.save();
        ctx.beginPath(); ctx.rect(v.x, v.y, v.w, v.h); ctx.clip();
        for (k = 0; k < E.K; k++) drawFrustum(v, E.cams[k], sel === k ? 0.5 : 0.13, sel === k ? 0.045 : 0);
        drawFrustum(v, E.novel, sel === 'novel' ? 0.55 : 0.2, sel === 'novel' ? 0.045 : 0);
        drawArc(v, kind);
        // the true surface: thin, dim, dashed
        ctx.beginPath();
        for (i = 0; i < GT_N; i += 4) { x = v.ox + GT_X[i] * v.s; y = v.oy - GT_Y[i] * v.s; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
        ctx.setLineDash([5, 4]); ctx.lineWidth = 1.2; ctx.strokeStyle = rgba(COL.dim, 0.85); ctx.stroke(); ctx.setLineDash([]);
        // the shared starting points
        ctx.beginPath();
        for (i = 0; i < E.pts.length; i++) {
          x = v.ox + E.pts[i].x * v.s; y = v.oy - E.pts[i].y * v.s;
          ctx.moveTo(x - 3, y); ctx.lineTo(x + 3, y); ctx.moveTo(x, y - 3); ctx.lineTo(x, y + 3);
        }
        ctx.lineWidth = 1; ctx.strokeStyle = rgba(COL.dim, 0.5); ctx.stroke();
        if (kind === 'bound') { drawGaussians(v, 'bound'); drawBoundCurve(v); drawControlPoints(v); }
        else drawGaussians(v, 'free');
        if (kind === 'free' && raysOn) drawRays(v, off);
        for (k = 0; k < E.K; k++) drawCamera(v, E.cams[k], sel === k, String(k + 1), false);
        drawCamera(v, E.novel, sel === 'novel', T('HELD-OUT'), true);
        // pane title
        if (views.length > 1) {
          ctx.font = font(compact ? 11 : 13, 700); ctx.textBaseline = 'alphabetic';
          ctx.fillStyle = kind === 'free' ? COL.err : COL.surf;
          ctx.fillText(kind === 'free' ? T('FREE · 3DGS-LIKE') : T('BOUND · OURS'), v.x + (compact ? 10 : 16),
                       Math.min(v.y + v.h - (compact ? 6 : 10), v.oy - VIEW.y0 * v.s + 16));
        }
        ctx.restore();
      }
      function drawStrips() {
        var B = stripBox, S = selCam(), W = CFG.W, x0 = B.x + B.labW, bw = B.w - B.labW - B.valW, pw = bw / W, y = B.y, j, rows, r, img, e, a;
        ctx.textBaseline = 'middle';
        ctx.font = font(11, 600); ctx.fillStyle = COL.dim;
        var title = S.novel ? T('HELD-OUT VIEW — never trained on') : T('TRAINING VIEW — camera {i} of {k}', { i: sel + 1, k: E.K });
        if (compact) title = S.novel ? T('HELD-OUT VIEW') : T('TRAINING VIEW {i}/{k}', { i: sel + 1, k: E.K });
        title = caps(title);
        ctx.fillText(title, x0, y + B.headH / 2 - 1);
        if (!compact) {
          // the note on the right, in the longest wording that leaves the title clear (a narrow stage gets a shorter one)
          var room = bw - ctx.measureText(title).width - 18, notes, nt;
          if (raysOn && slideTrain === slideTrain) {      // live probe: the strip below is rendered with the floater displaced
            ctx.fillStyle = COL.text; nt = { v: slideTrain.toFixed(1) + ' dB' };
            notes = [T('LIVE · FLOATER SLIDING ALONG ITS RAY · ALL TRAINING VIEWS {v}', nt), T('LIVE · FLOATER SLIDING · ALL TRAINING VIEWS {v}', nt), T('FLOATER SLIDING · TRAINING VIEWS {v}', nt), T('TRAINING VIEWS {v}', nt)];
          } else {
            ctx.fillStyle = COL.mute;
            notes = [T('1D IMAGE, {w} PX · RED UNDERLINE = ERROR', { w: W }), T('{w} PX · RED UNDERLINE = ERROR', { w: W }), T('RED = ERROR')];
          }
          for (j = 0; j < notes.length && ctx.measureText(notes[j]).width > room; j++) {}
          ctx.textAlign = 'right';
          if (j < notes.length) ctx.fillText(notes[j], x0 + bw, y + B.headH / 2 - 1);
          ctx.textAlign = 'left';
        }
        y += B.headH;
        rows = [['gt', T('TRUE'), COL.dim]];
        if (shown('free')) rows.push(['free', T('FREE'), COL.err]);
        if (shown('bound')) rows.push(['bound', T('BOUND'), COL.surf]);
        for (r = 0; r < rows.length; r++) {
          img = rows[r][0] === 'gt' ? S.tgt : IMG[rows[r][0]];
          var cv = CVR && (rows[r][0] === 'gt' ? S.cov : CVR[rows[r][0]]);
          for (j = 0; j < W; j++) {
            if (cv) {                          // light page: over the backdrop, by coverage (see BACK)
              a = 1 - cv[j];
              ctx.fillStyle = 'rgb(' + c255(img[j * 3] + a * BACK[0]) + ',' + c255(img[j * 3 + 1] + a * BACK[1]) + ',' + c255(img[j * 3 + 2] + a * BACK[2]) + ')';
            } else ctx.fillStyle = 'rgb(' + c255(img[j * 3]) + ',' + c255(img[j * 3 + 1]) + ',' + c255(img[j * 3 + 2]) + ')';
            ctx.fillRect(x0 + j * pw, y, pw + 0.6, B.rowH);
          }
          ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.strokeRect(x0 - 0.5, y - 0.5, bw + 1, B.rowH + 1);
          ctx.font = font(compact ? 11 : 12, 700); ctx.fillStyle = rows[r][2]; ctx.fillText(rows[r][1], B.x + 2, y + B.rowH / 2 + 0.5);
          if (rows[r][0] !== 'gt') {
            ctx.textAlign = 'right'; ctx.font = font(compact ? 11 : 13, 600); ctx.fillStyle = COL.text;
            ctx.fillText(fmt(ROWP[rows[r][0]], 1) + ' dB', B.x + B.w - 2, y + B.rowH / 2 + 0.5); ctx.textAlign = 'left';
            for (j = 0; j < W; j++) {
              e = (Math.abs(img[j * 3] - S.tgt[j * 3]) + Math.abs(img[j * 3 + 1] - S.tgt[j * 3 + 1]) + Math.abs(img[j * 3 + 2] - S.tgt[j * 3 + 2])) / 3;
              a = clamp(e * 5, 0, 1);
              if (a < 0.04) continue;
              ctx.fillStyle = rgba(COL.err, a.toFixed(3)); ctx.fillRect(x0 + j * pw, y + B.rowH + 1, pw + 0.6, B.errH);
            }
            y += B.errH;
          }
          y += B.rowH + B.gap;
        }
        ctx.textBaseline = 'alphabetic';
      }
      function drawLegend() {
        if (!legendY) return;
        var x = 16, y = legendY, items, i, tw;
        ctx.font = font(11, 500); ctx.textBaseline = 'middle';
        items = [['dash', T('true surface')], ['plus', T('shared start points')], ['ring', T('Gaussian, 1σ')], ['red', T('floater')],
                 ['arc', T('held-out PSNR from that viewpoint, {lo}–{hi} dB', { lo: ARC_LO, hi: ARC_HI })]];
        if (shown('bound')) items.splice(3, 0, ['ctrl', T('control point')], ['surf', T('spline')]);
        for (i = 0; i < items.length; i++) {
          items[i][1] = caps(items[i][1]).replace('Σ', 'σ');     // capitals turn σ into Σ: put it back
          tw = ctx.measureText(items[i][1]).width;
          if (x + 26 + tw > Wd - 12) break;
          switch (items[i][0]) {
            case 'dash': ctx.setLineDash([4, 3]); ctx.strokeStyle = COL.dim; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 18, y); ctx.stroke(); ctx.setLineDash([]); break;
            case 'plus': ctx.strokeStyle = COL.dim; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x + 5, y); ctx.lineTo(x + 13, y); ctx.moveTo(x + 9, y - 4); ctx.lineTo(x + 9, y + 4); ctx.stroke(); break;
            case 'ring': ctx.strokeStyle = COL.splat; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.ellipse(x + 9, y, 8, 4, 0, 0, 6.2832); ctx.stroke(); break;
            case 'red': ctx.strokeStyle = COL.err; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.ellipse(x + 9, y, 8, 4, 0, 0, 6.2832); ctx.stroke(); break;
            case 'ctrl': ctx.fillStyle = COL.ctrl; ctx.beginPath(); ctx.arc(x + 9, y, 3, 0, 6.2832); ctx.fill(); break;
            case 'surf': ctx.strokeStyle = COL.surf; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 18, y); ctx.stroke(); break;
            case 'arc': var g = ctx.createLinearGradient(x, 0, x + 18, 0); g.addColorStop(0, COL.err); g.addColorStop(1, COL.ok);
              ctx.strokeStyle = g; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 18, y); ctx.stroke(); break;
          }
          ctx.fillStyle = COL.mute; ctx.fillText(items[i][1], x + 24, y + 0.5);
          x += 24 + tw + 20;
        }
        ctx.textBaseline = 'alphabetic';
      }
      /* scene `handoff`: the bound curve, from its place in the BOUND pane (k = 0) to the hand-off curve in the
         hand-off frame (k = 1), sample by sample at the same parameter u; its style goes from the plain line to
         the HANDOFF STYLE on the way. Drawn last and unclipped, at full opacity, while the rest fades. */
      function drawHandoff() {
        var F = handoffFrame(Wd, Hd), k = HO.k, th = HO.from, G = HO.goal, T = drawTab, v = null, i, r, j, N, x, y, ax, ay, w0, w;
        for (i = 0; i < views.length; i++) if (views[i].kind === 'bound') v = views[i];
        if (!v) v = F;                       // no BOUND pane (never in the deck): start in the hand-off frame
        var path = LIGHT ? new Path2D() : null;   // a light page also strokes the curve moved (shadow, highlight)
        ctx.beginPath();
        for (i = 0; i < T.n; i++) {
          x = y = 0;
          for (r = 0; r < 4; r++) { j = T.span[i] - 3 + r; N = T.N[i * 4 + r]; x += N * th[2 * j]; y += N * th[2 * j + 1]; }
          ax = v.ox + x * v.s; ay = v.oy - y * v.s;
          x = ax + (F.ox + G[2 * i] * F.s - ax) * k; y = ay + (F.oy - G[2 * i + 1] * F.s - ay) * k;
          if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
          if (path) { if (i) path.lineTo(x, y); else path.moveTo(x, y); }
        }
        w0 = compact ? 1.3 : 1.6; w = w0 + (handoffWidth(F.s) - w0) * k;
        ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        if (k > 0 && path) {                 // light: the glow's strokes as a soft shadow, moved down
          ctx.save(); ctx.translate(0, HO_DROP * w);
          for (i = 0; i < HO_GLOW.length; i++) {
            ctx.lineWidth = w * HO_GLOW[i][0]; ctx.strokeStyle = rgba(COL.text, (HO_GLOW[i][1] * HO_SHADE * k).toFixed(4)); ctx.stroke(path);
          }
          ctx.restore();
        } else if (k > 0) for (i = 0; i < HO_GLOW.length; i++) {
          ctx.lineWidth = w * HO_GLOW[i][0]; ctx.strokeStyle = rgba(COL.surf, (HO_GLOW[i][1] * k).toFixed(4)); ctx.stroke();
        }
        ctx.lineWidth = w; ctx.strokeStyle = COL.surf; ctx.stroke();
        if (k > 0) {
          ctx.save(); if (path) ctx.translate(0, -HO_LIFT * w);
          ctx.lineWidth = 0.4 * w; ctx.strokeStyle = rgba(HO_HI, (HO_HA * k).toFixed(4));
          if (path) ctx.stroke(path); else ctx.stroke();
          ctx.restore();
        }
        ctx.lineCap = 'butt'; ctx.lineJoin = 'miter';
      }
      function draw() {
        if (!E || !Wd || !stripBox) return;
        var off = renderStrips(), i, v, hx, hy;
        ctx.clearRect(0, 0, Wd, Hd);
        ctx.globalAlpha = HO.on ? HO.keep : 1;
        if (ctx.globalAlpha > 0.002) {
          for (i = 0; i < views.length; i++) drawView(views[i], off);
          if (views.length > 1) {            // hairline between the two panes
            v = views[1]; ctx.beginPath();
            if (v.x > views[0].x) { ctx.moveTo(v.x + 0.5, v.y + 8); ctx.lineTo(v.x + 0.5, v.y + v.h - 8); }
            else { ctx.moveTo(v.x + 16, v.y + 0.5); ctx.lineTo(v.x + v.w - 16, v.y + 0.5); }
            ctx.lineWidth = 1; ctx.strokeStyle = COL.line; ctx.stroke();
          }
          if (raysOn) drawRaysInfo();
          drawLegend();
          drawStrips();
        }
        ctx.globalAlpha = 1;
        if (HO.on) drawHandoff();
        for (i = 0; i < views.length; i++) {
          v = views[i]; hx = v.ox + E.novel.px * v.s; hy = v.oy - E.novel.py * v.s;
          handles[i].style.transform = 'translate(' + hx.toFixed(1) + 'px,' + hy.toFixed(1) + 'px)';
        }
      }

      /* ---------- the loop: time-based optimization, then measure, then draw ---------- */
      env.loop(function (t, dt) {
        now = t;
        var stepped = false;
        if (novelDirty) { setNovel(E, userTheta); novelDirty = false; dirty = true; }
        if (running && E.free.step < CFG.maxSteps) {
          acc = Math.min(acc + RATE * dt, RATE * 0.05);          // per-frame cap: at most 50 ms worth of steps
          var todo = Math.floor(acc), t0 = performance.now();
          acc -= todo;
          while (todo-- > 0 && E.free.step < CFG.maxSteps) {
            trainStep(E.free, E.cams, E.tgts); trainStep(E.bound, E.cams, E.tgts);   // always in lockstep
            stepped = true;
            if (performance.now() - t0 > 14) { acc = 0; break; }                      // and at most ~14 ms of work
          }
        }
        if (stepped || dirty) { derive(E.free); derive(E.bound); }
        if (dirty || (stepped && t - tMet > 0.12) || (!stepped && MET.free.step !== E.free.step)) {
          updateMetrics(); MET.free.step = E.free.step; tMet = t;
        }
        if (arcStep !== E.free.step && (t - tArc > 0.35 || dirty)) { updateArc(); tArc = t; }
        if (raysOn && probeStep !== E.free.step && E.free.step >= 300 && (t - tProbe > 2 || !stepped)) { updateProbe(); tProbe = t; }
        var redraw = dirty || !HO.on || !HO.done;
        if (HO.on && !HO.done) {                 // the hand-off: time-based; once it has ended the picture holds still
          HO.t += dt;
          HO.keep = 1 - smoother(HO.t / HO_FADE);
          HO.k = smoother((HO.t - HO_MOVE[0]) / (HO_MOVE[1] - HO_MOVE[0]));
          HO.done = HO.t >= Math.max(HO_FADE, HO_MOVE[1]);
          rootEl.style.setProperty('--fl-keep', HO.keep.toFixed(3));
          rootEl.classList.toggle('fl-gone', HO.keep < 0.002);
        }
        dirty = false;
        if (redraw) draw();
      });

      /* ---------- boot ---------- */
      reset(); showRows(); paintCaption();
      (function () { var r = rootEl.getBoundingClientRect(); if (r.width > 0 && r.height > 0) resize(r.width, r.height); })();

      return {
        setScene: setScene,
        resize: resize,
        destroy: function () { cvs.cv.removeEventListener('pointerdown', onDown); cvs.cv.removeEventListener('pointerup', onUp); },
        /* for tests and for the deck's speaker notes: the live numbers */
        state: function () {
          return { scene: scene, K: E.K, step: E.free.step, model: model, sel: sel, running: running,
                   handoff: HO.on ? { t: HO.t, keep: HO.keep, k: HO.k, done: HO.done, frame: handoffFrame(Wd, Hd) } : null,
                   free: { train: MET.free.train, novel: MET.free.novel, geo: MET.free.geo, floaters: MET.free.fl, params: E.free.th.length },
                   bound: { train: MET.bound.train, novel: MET.bound.novel, geo: MET.bound.geo, floaters: MET.bound.fl, params: E.bound.th.length },
                   probe: probe && { one: probe.one.length, all: probe.list.length, base: probe.base, medAlong: probe.medAlong, medAcross: probe.medAcross,
                                     ex: probe.ex.map(function (r) { return { along: r.along, across: r.across, share: r.share }; }) } };
        }
      };
    }
  });
})(typeof window !== 'undefined' ? window : globalThis);
