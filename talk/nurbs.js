/* Knot-a-Surface talk — spline math shared by every simulation module.
   Pure functions, no dependencies. Algorithm numbers refer to Piegl & Tiller, "The NURBS Book".
   Exposed as window.KnotTalk.nurbs (and module.exports under node, for tests). */
(function (root) {
  'use strict';

  /* Clamped uniform knot vector: n control points, degree p, domain [0,1]. */
  function clampedKnots(n, p) {
    var U = [], i, inner = n - p;
    for (i = 0; i <= p; i++) U.push(0);
    for (i = 1; i < inner; i++) U.push(i / inner);
    for (i = 0; i <= p; i++) U.push(1);
    return U;
  }

  /* Periodic (closed) uniform knots: n UNIQUE control points, wrapped to n+p. Domain [0,1]. */
  function periodicKnots(n, p) {
    var U = [], i;
    for (i = 0; i <= n + 2 * p; i++) U.push((i - p) / n);
    return U;
  }

  /* A2.1 — knot span index k with U[k] <= u < U[k+1]; m = number of control points. */
  function findSpan(m, p, u, U) {
    if (u >= U[m]) return m - 1;
    if (u <= U[p]) return p;
    var lo = p, hi = m, mid = (lo + hi) >> 1;
    while (u < U[mid] || u >= U[mid + 1]) {
      if (u < U[mid]) hi = mid; else lo = mid;
      mid = (lo + hi) >> 1;
    }
    return mid;
  }

  /* A2.2 + first derivative — the p+1 non-zero basis functions at u, and d/du of each.
     Writes N[0..p] and (if D given) D[0..p] starting at `off`. */
  var _l = new Float64Array(8), _r = new Float64Array(8), _prev = new Float64Array(8);
  function basisFuns(k, u, p, U, N, D, off) {
    off = off || 0;
    var j, r, saved, tmp;
    N[off] = 1;
    for (j = 1; j <= p; j++) {
      if (j === p && D) for (r = 0; r < p; r++) _prev[r] = N[off + r];   // degree p-1 values
      _l[j] = u - U[k + 1 - j];
      _r[j] = U[k + j] - u;
      saved = 0;
      for (r = 0; r < j; r++) {
        tmp = N[off + r] / (_r[r + 1] + _l[j - r]);
        N[off + r] = saved + _r[r + 1] * tmp;
        saved = _l[j - r] * tmp;
      }
      N[off + j] = saved;
    }
    if (D) {
      if (p === 0) { D[off] = 0; return; }
      // N'_{i,p} = p/(U[i+p]-U[i]) N_{i,p-1} - p/(U[i+p+1]-U[i+1]) N_{i+1,p-1}
      for (r = 0; r <= p; r++) {
        var i = k - p + r, a = 0, b = 0, d;
        if (r > 0) { d = U[i + p] - U[i]; if (d > 0) a = p / d * _prev[r - 1]; }
        if (r < p) { d = U[i + p + 1] - U[i + 1]; if (d > 0) b = p / d * _prev[r]; }
        D[off + r] = a - b;
      }
    }
  }

  /* One basis function N_{i,p}(u) by the Cox–de Boor recursion (for plotting). */
  function oneBasis(i, p, u, U) {
    if (p === 0) {
      if (u >= U[i] && u < U[i + 1]) return 1;
      // close the last non-empty span on the right so the plot reaches u = max
      return (u === U[U.length - 1] && U[i] < u && U[i + 1] === u) ? 1 : 0;
    }
    var a = 0, b = 0, d1 = U[i + p] - U[i], d2 = U[i + p + 1] - U[i + 1];
    if (d1 > 0) a = (u - U[i]) / d1 * oneBasis(i, p - 1, u, U);
    if (d2 > 0) b = (U[i + p + 1] - u) / d2 * oneBasis(i + 1, p - 1, u, U);
    return a + b;
  }

  /* Rational curve point. P = [[x,y],...], w = weights (or null), returns [x,y]. */
  var _N = new Float64Array(8);
  function curvePoint(P, w, p, U, u, out) {
    var m = P.length, k = findSpan(m, p, u, U), x = 0, y = 0, W = 0, r, i, b;
    basisFuns(k, u, p, U, _N);
    for (r = 0; r <= p; r++) {
      i = k - p + r; b = _N[r] * (w ? w[i] : 1);
      x += b * P[i][0]; y += b * P[i][1]; W += b;
    }
    out = out || [0, 0];
    out[0] = x / W; out[1] = y / W;
    return out;
  }

  /* A5.1 (Boehm) — insert one knot; the curve does not change. Works in homogeneous coords. */
  function insertKnot(P, w, p, U, ub) {
    var m = P.length, k = findSpan(m, p, ub, U), Q = [], W = [], i, a, wa, wb, ww;
    for (i = 0; i <= m; i++) {
      if (i <= k - p) a = 1;
      else if (i >= k + 1) a = 0;
      else a = (ub - U[i]) / (U[i + p] - U[i]);
      wa = a > 0 ? (w ? w[i] : 1) : 0;
      wb = a < 1 ? (w ? w[i - 1] : 1) : 0;
      ww = a * wa + (1 - a) * wb;
      Q.push([
        (a * wa * (a > 0 ? P[i][0] : 0) + (1 - a) * wb * (a < 1 ? P[i - 1][0] : 0)) / ww,
        (a * wa * (a > 0 ? P[i][1] : 0) + (1 - a) * wb * (a < 1 ? P[i - 1][1] : 0)) / ww
      ]);
      W.push(ww);
    }
    var V = U.slice(0, k + 1).concat([ub], U.slice(k + 1));
    return { P: Q, w: W, U: V };
  }

  /* ---------- tensor-product patches (B-spline; weights = 1) ---------- */

  /* A patch: nu x nv UNIQUE control points, degree p, optional wrap per axis.
     P = positions (xyz), C = control features shown as colour (rgb). */
  function makePatch(o) {
    var p = o.p || 3, nu = o.nu, nv = o.nv;
    return {
      nu: nu, nv: nv, p: p, periodicU: !!o.periodicU, periodicV: !!o.periodicV,
      mu: o.periodicU ? nu + p : nu, mv: o.periodicV ? nv + p : nv,
      Uk: o.periodicU ? periodicKnots(nu, p) : clampedKnots(nu, p),
      Vk: o.periodicV ? periodicKnots(nv, p) : clampedKnots(nv, p),
      P: new Float32Array(nu * nv * 3), C: new Float32Array(nu * nv * 3)
    };
  }

  /* Precompute basis + derivative for a list of parameter values on one axis. */
  function axisTable(knots, m, p, ts) {
    var n = ts.length, w = p + 1;
    var T = { n: n, span: new Int32Array(n), N: new Float64Array(n * w), D: new Float64Array(n * w) };
    for (var a = 0; a < n; a++) {
      var k = findSpan(m, p, ts[a], knots);
      T.span[a] = k;
      basisFuns(k, ts[a], p, knots, T.N, T.D, a * w);
    }
    return T;
  }

  /* Evaluate a patch on the grid us x vs (two separable passes).
     out.pos / out.su / out.sv / out.col are Float32Array(us.length*vs.length*3), row = v index. */
  function patchGrid(patch, us, vs, out) {
    var p = patch.p, w = p + 1, nu = patch.nu, nv = patch.nv;
    var TU = axisTable(patch.Uk, patch.mu, p, us), TV = axisTable(patch.Vk, patch.mv, p, vs);
    var na = us.length, nb = vs.length, P = patch.P, C = patch.C;
    // pass 1: contract along v for every control column i -> A (value), B (d/dv), K (colour)
    var A = new Float64Array(nu * nb * 3), B = new Float64Array(nu * nb * 3), K = new Float64Array(nu * nb * 3);
    var i, a, b, r, j, o, q, n, d;
    for (b = 0; b < nb; b++) {
      var kv = TV.span[b] - p;
      for (i = 0; i < nu; i++) {
        o = (i * nb + b) * 3;
        for (r = 0; r < w; r++) {
          j = kv + r; if (j >= nv) j -= nv;
          q = (j * nu + i) * 3; n = TV.N[b * w + r]; d = TV.D[b * w + r];
          A[o] += n * P[q]; A[o + 1] += n * P[q + 1]; A[o + 2] += n * P[q + 2];
          B[o] += d * P[q]; B[o + 1] += d * P[q + 1]; B[o + 2] += d * P[q + 2];
          K[o] += n * C[q]; K[o + 1] += n * C[q + 1]; K[o + 2] += n * C[q + 2];
        }
      }
    }
    // pass 2: contract along u
    var pos = out.pos, su = out.su, sv = out.sv, col = out.col, t;
    for (b = 0; b < nb; b++) {
      for (a = 0; a < na; a++) {
        var ku = TU.span[a] - p, x = 0, y = 0, z = 0, ux = 0, uy = 0, uz = 0, vx = 0, vy = 0, vz = 0, cr = 0, cg = 0, cb = 0;
        for (r = 0; r < w; r++) {
          i = ku + r; if (i >= nu) i -= nu;
          o = (i * nb + b) * 3; n = TU.N[a * w + r]; d = TU.D[a * w + r];
          x += n * A[o]; y += n * A[o + 1]; z += n * A[o + 2];
          ux += d * A[o]; uy += d * A[o + 1]; uz += d * A[o + 2];
          vx += n * B[o]; vy += n * B[o + 1]; vz += n * B[o + 2];
          cr += n * K[o]; cg += n * K[o + 1]; cb += n * K[o + 2];
        }
        t = (b * na + a) * 3;
        pos[t] = x; pos[t + 1] = y; pos[t + 2] = z;
        if (su) { su[t] = ux; su[t + 1] = uy; su[t + 2] = uz; }
        if (sv) { sv[t] = vx; sv[t + 1] = vy; sv[t + 2] = vz; }
        if (col) { col[t] = cr; col[t + 1] = cg; col[t + 2] = cb; }
      }
    }
    return out;
  }

  /* Single-point evaluation: returns {pos, su, sv, col} as plain arrays. */
  function patchPoint(patch, u, v) {
    var o = { pos: new Float32Array(3), su: new Float32Array(3), sv: new Float32Array(3), col: new Float32Array(3) };
    patchGrid(patch, [u], [v], o);
    return o;
  }

  /* Which control points (unique indices) influence (u,v), with their weights N_i(u)N_j(v). */
  function patchSupport(patch, u, v) {
    var p = patch.p, w = p + 1, Nu = new Float64Array(w), Nv = new Float64Array(w);
    var ku = findSpan(patch.mu, p, u, patch.Uk), kv = findSpan(patch.mv, p, v, patch.Vk), res = [];
    basisFuns(ku, u, p, patch.Uk, Nu); basisFuns(kv, v, p, patch.Vk, Nv);
    for (var s = 0; s < w; s++) for (var r = 0; r < w; r++) {
      res.push({ i: (ku - p + r) % patch.nu, j: (kv - p + s) % patch.nv, w: Nu[r] * Nv[s] });
    }
    return res;
  }

  var api = {
    clampedKnots: clampedKnots, periodicKnots: periodicKnots, findSpan: findSpan,
    basisFuns: basisFuns, oneBasis: oneBasis, curvePoint: curvePoint, insertKnot: insertKnot,
    makePatch: makePatch, axisTable: axisTable, patchGrid: patchGrid, patchPoint: patchPoint,
    patchSupport: patchSupport
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.KnotTalk = root.KnotTalk || {};
  root.KnotTalk.nurbs = api;
})(typeof window !== 'undefined' ? window : globalThis);
