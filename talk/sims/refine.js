/* Knot-a-Surface talk — module `refine`: "Two-layer density".
   A real least-squares fit of a cubic tensor-product B-spline height field to a procedural terrain.
     Layer A — control density: Boehm knot insertion where the per-span error is worst (exact: the
               surface is unchanged at the instant of insertion), and remove + re-fit where a knot is
               redundant.
     Layer B — sampling density: Gaussians sampled on the (u,v) domain; never touches the surface.
   Tool: Canvas 2D. Standalone: talk/sims/refine.html. Depends on talk/nurbs.js + talk/kit.js only.
   The maths below (PART 1) is pure and is also exported under node for tests. */
(function (root) {
  'use strict';
  var NODE = typeof module !== 'undefined' && module.exports;
  var NB = NODE ? require('../nurbs.js') : root.KnotTalk.nurbs;
  var P = 3, W = 4;                       // cubic everywhere

  /* =====================================================================================
     PART 1 — pure maths
     Layout conventions: a grid field F[b * n + a] has a along u (x, to the right) and b along v
     (y, downwards). The control net is Z[j * nu + i], i along u, j along v.
     ===================================================================================== */

  /* ---------- the target: smooth rolling ground + one crater field ---------- */
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
  function targetRaw(x, y) {
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
  /* Sampled target at cell centres, normalised so the relief spans exactly [0, 1]. */
  function makeTarget(G) {
    var T = new Float64Array(G * G), lo = Infinity, hi = -Infinity, a, b, v;
    for (b = 0; b < G; b++) for (a = 0; a < G; a++) {
      v = targetRaw((a + 0.5) / G, (b + 0.5) / G); T[b * G + a] = v;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    var k = 1 / (hi - lo);
    for (a = 0; a < T.length; a++) T[a] = (T[a] - lo) * k;
    return { G: G, T: T, lo: lo, k: k, xs: centres(G) };
  }
  function targetAt(tg, x, y) { return (targetRaw(x, y) - tg.lo) * tg.k; }
  function centres(n) { var xs = new Float64Array(n); for (var a = 0; a < n; a++) xs[a] = (a + 0.5) / n; return xs; }

  /* ---------- basis tables / evaluation ---------- */
  function table(knots, n, xs) { return NB.axisTable(knots, n, P, xs); }

  /* z on the grid xs(u) x ys(v) from tables TU, TV. */
  function evalGrid(Z, nu, TU, TV, out) {
    var na = TU.n, nb = TV.n, a, b, r, s, o, acc, row, kv, ku, Nu = TU.N, Nv = TV.N;
    for (b = 0; b < nb; b++) {
      kv = TV.span[b] - P;
      for (a = 0; a < na; a++) {
        ku = TU.span[a] - P; acc = 0;
        for (s = 0; s < W; s++) {
          o = (kv + s) * nu + ku; row = 0;
          for (r = 0; r < W; r++) row += Nu[a * W + r] * Z[o + r];
          acc += Nv[b * W + s] * row;
        }
        out[b * na + a] = acc;
      }
    }
    return out;
  }
  /* Gram matrix B^T B of one axis (dense n x n, banded). */
  function gram(T, n) {
    var Gm = new Float64Array(n * n), a, r, s, k;
    for (a = 0; a < T.n; a++) {
      k = T.span[a] - P;
      for (r = 0; r < W; r++) for (s = 0; s < W; s++) Gm[(k + r) * n + k + s] += T.N[a * W + r] * T.N[a * W + s];
    }
    return Gm;
  }
  /* rhs = Bu^T T Bv  (nv rows x nu cols). */
  function rhsOf(T, TU, TV, nu, nv) {
    var na = TU.n, nb = TV.n, out = new Float64Array(nu * nv), a, b, r, s, ku, kv, t, w;
    for (b = 0; b < nb; b++) {
      kv = TV.span[b] - P;
      for (a = 0; a < na; a++) {
        ku = TU.span[a] - P; t = T[b * na + a];
        for (s = 0; s < W; s++) {
          w = TV.N[b * W + s] * t;
          for (r = 0; r < W; r++) out[(kv + s) * nu + ku + r] += w * TU.N[a * W + r];
        }
      }
    }
    return out;
  }

  /* ---------- the model ---------- */
  /* A model bundles knots + net with everything derived from the knots for a given target. */
  function makeModel(tg, U, V, Z) {
    var nu = U.length - W, nv = V.length - W;
    var m = { tg: tg, U: U, V: V, nu: nu, nv: nv, Z: Z || new Float64Array(nu * nv) };
    m.TU = table(U, nu, tg.xs); m.TV = table(V, nv, tg.xs);
    m.Gu = gram(m.TU, nu); m.Gv = gram(m.TV, nv);
    m.rhs = rhsOf(tg.T, m.TU, m.TV, nu, nv);
    m.fit = new Float64Array(tg.G * tg.G);
    m.cg = null;
    return m;
  }
  function uniformModel(tg, nu, nv) { return makeModel(tg, NB.clampedKnots(nu, P), NB.clampedKnots(nv, P)); }

  /* A X = Gu X Gv on nets (the normal-equations operator of the separable least-squares problem). */
  function applyA(m, X, out, tmp) {
    var nu = m.nu, nv = m.nv, Gu = m.Gu, Gv = m.Gv, i, j, k, lo, hi, acc;
    for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
      lo = i - P < 0 ? 0 : i - P; hi = i + P >= nu ? nu - 1 : i + P; acc = 0;
      for (k = lo; k <= hi; k++) acc += Gu[i * nu + k] * X[j * nu + k];
      tmp[j * nu + i] = acc;
    }
    for (j = 0; j < nv; j++) {
      lo = j - P < 0 ? 0 : j - P; hi = j + P >= nv ? nv - 1 : j + P;
      for (i = 0; i < nu; i++) {
        acc = 0;
        for (k = lo; k <= hi; k++) acc += Gv[j * nv + k] * tmp[k * nu + i];
        out[j * nu + i] = acc;
      }
    }
    return out;
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  /* Jacobi-preconditioned conjugate gradients on the normal equations — the optimizer.
     Each step lowers the squared error monotonically (CG minimises the energy norm). */
  function cgInit(m) {
    var n = m.nu * m.nv, i, j;
    var c = { r: new Float64Array(n), z: new Float64Array(n), p: new Float64Array(n), Ap: new Float64Array(n),
              tmp: new Float64Array(n), Mi: new Float64Array(n), rz: 0, rz0: 0, it: 0, done: false };
    for (j = 0; j < m.nv; j++) for (i = 0; i < m.nu; i++) c.Mi[j * m.nu + i] = 1 / (m.Gu[i * m.nu + i] * m.Gv[j * m.nv + j]);
    applyA(m, m.Z, c.Ap, c.tmp);
    for (i = 0; i < n; i++) { c.r[i] = m.rhs[i] - c.Ap[i]; c.z[i] = c.Mi[i] * c.r[i]; c.p[i] = c.z[i]; }
    c.rz = dot(c.r, c.z);
    var b2 = 0; for (i = 0; i < n; i++) b2 += m.rhs[i] * m.rhs[i] * c.Mi[i];
    c.b2 = b2; c.done = c.rz <= CG_TOL * CG_TOL * b2;
    m.cg = c; return c;
  }
  var CG_TOL = 1e-10;
  function cgStep(m) {
    var c = m.cg || cgInit(m);
    if (c.done) return false;
    var n = c.r.length, i, pAp, alpha, rzNew, beta, Z = m.Z;
    applyA(m, c.p, c.Ap, c.tmp);
    pAp = dot(c.p, c.Ap);
    if (!(pAp > 0)) { c.done = true; return false; }
    alpha = c.rz / pAp;
    for (i = 0; i < n; i++) { Z[i] += alpha * c.p[i]; c.r[i] -= alpha * c.Ap[i]; c.z[i] = c.Mi[i] * c.r[i]; }
    rzNew = dot(c.r, c.z); beta = rzNew / c.rz; c.rz = rzNew; c.it++;
    for (i = 0; i < n; i++) c.p[i] = c.z[i] + beta * c.p[i];
    if (rzNew <= CG_TOL * CG_TOL * c.b2 || c.it > 20 * n + 200) c.done = true;
    return true;
  }
  function cgConverge(m, maxIt) {
    var k = 0; maxIt = maxIt || 1e5;
    if (!m.cg) cgInit(m);
    while (k < maxIt && cgStep(m)) k++;
    return k;
  }

  /* Dense Cholesky for the small per-axis Gram matrices; solves in place for many right-hand sides. */
  function cholesky(A, n) {
    var L = new Float64Array(A), i, j, k, s;
    for (j = 0; j < n; j++) {
      s = L[j * n + j]; for (k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
      s = Math.sqrt(s > 0 ? s : 1e-300); L[j * n + j] = s;
      for (i = j + 1; i < n; i++) {
        var t = L[i * n + j]; for (k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = t / s;
      }
    }
    return L;
  }
  function cholSolve(L, n, x, off, stride) {       // x[off + i*stride], i = 0..n-1
    var i, k, s;
    for (i = 0; i < n; i++) { s = x[off + i * stride]; for (k = 0; k < i; k++) s -= L[i * n + k] * x[off + k * stride]; x[off + i * stride] = s / L[i * n + i]; }
    for (i = n - 1; i >= 0; i--) { s = x[off + i * stride]; for (k = i + 1; k < n; k++) s -= L[k * n + i] * x[off + k * stride]; x[off + i * stride] = s / L[i * n + i]; }
  }
  /* Closed-form optimum of the same problem: Z = Gu^-1 (Bu^T T Bv) Gv^-1. Used to score removal
     candidates and to cross-check the optimizer; the animated fit is always CG. */
  function solveDirect(m) {
    var nu = m.nu, nv = m.nv, Z = new Float64Array(m.rhs), Lu = cholesky(m.Gu, nu), Lv = cholesky(m.Gv, nv), i, j;
    for (j = 0; j < nv; j++) cholSolve(Lu, nu, Z, j * nu, 1);
    for (i = 0; i < nu; i++) cholSolve(Lv, nv, Z, i, nu);
    return Z;
  }

  /* ---------- error measures ---------- */
  /* Evaluates the fit on the target grid and returns RMSE, max |e| and the marginal span errors:
     for each U-span and each V-span, { a, b, n = sample lines inside, mse = mean squared residual
     over the samples inside }. mse * n / G is the span's share of the total mean squared error. */
  function measure(m) {
    var tg = m.tg, G = tg.G, T = tg.T, fit = evalGrid(m.Z, m.nu, m.TU, m.TV, m.fit);
    var col = new Float64Array(G), row = new Float64Array(G), a, b, e, ss = 0, mx = 0;
    for (b = 0; b < G; b++) for (a = 0; a < G; a++) {
      e = fit[b * G + a] - T[b * G + a]; e *= e;
      col[a] += e; row[b] += e; ss += e; if (e > mx) mx = e;
    }
    return { rmse: Math.sqrt(ss / (G * G)), max: Math.sqrt(mx),
             su: spanMeans(col, m.TU, m.U, m.nu, G), sv: spanMeans(row, m.TV, m.V, m.nv, G) };
  }
  function spanMeans(line, T, knots, n, G) {
    var ns = n - P, mse = new Float64Array(ns), cnt = new Int32Array(ns), a, k, spans = [];
    for (a = 0; a < T.n; a++) { k = T.span[a] - P; mse[k] += line[a]; cnt[k]++; }
    for (k = 0; k < ns; k++) spans.push({ k: k + P, a: knots[k + P], b: knots[k + P + 1], n: cnt[k],
                                          mse: cnt[k] ? mse[k] / (cnt[k] * G) : 0 });
    return spans;
  }
  /* Only the RMSE, from the normal equations: ||T||^2 - 2 rhs.Z + Z.AZ (cheap, no grid pass). */
  function rmseOf(m, Z) {
    var n = m.nu * m.nv, AZ = applyA(m, Z, new Float64Array(n), new Float64Array(n)), G = m.tg.G;
    if (m.tg.TT == null) m.tg.TT = dot(m.tg.T, m.tg.T);
    var ss = m.tg.TT - 2 * dot(m.rhs, Z) + dot(Z, AZ);
    return Math.sqrt(Math.max(0, ss) / (G * G));
  }

  /* ---------- Layer A, up: Boehm knot insertion (The NURBS Book, A5.1), scalar, per line ---------- */
  /* Inserts ub once into `knots`; every line of the net along that axis gets one more control
     height. Returns { knots, Z, nu, nv, k } — the surface it describes is the same function. */
  function insertKnot(Z, nu, nv, U, V, axis, ub) {
    var knots = axis === 'u' ? U : V, n = axis === 'u' ? nu : nv, k = NB.findSpan(n, P, ub, knots), i, j;
    var al = new Float64Array(n + 1);
    for (i = 0; i <= n; i++) al[i] = i <= k - P ? 1 : i >= k + 1 ? 0 : (ub - knots[i]) / (knots[i + P] - knots[i]);
    var nk = Array.prototype.slice.call(knots, 0, k + 1).concat([ub], Array.prototype.slice.call(knots, k + 1));
    var Q;
    if (axis === 'u') {
      Q = new Float64Array((nu + 1) * nv);
      for (j = 0; j < nv; j++) for (i = 0; i <= nu; i++)
        Q[j * (nu + 1) + i] = (al[i] > 0 ? al[i] * Z[j * nu + i] : 0) + (al[i] < 1 ? (1 - al[i]) * Z[j * nu + i - 1] : 0);
      return { U: nk, V: V, Z: Q, nu: nu + 1, nv: nv, k: k };
    }
    Q = new Float64Array(nu * (nv + 1));
    for (j = 0; j <= nv; j++) for (i = 0; i < nu; i++)
      Q[j * nu + i] = (al[j] > 0 ? al[j] * Z[j * nu + i] : 0) + (al[j] < 1 ? (1 - al[j]) * Z[(j - 1) * nu + i] : 0);
    return { U: U, V: nk, Z: Q, nu: nu, nv: nv + 1, k: k };
  }
  /* Model-level insertion; also MEASURES that the surface did not move (max |dz| on the grid). */
  function insertInto(m, axis, ub) {
    var before = new Float64Array(evalGrid(m.Z, m.nu, m.TU, m.TV, m.fit));
    var r = insertKnot(m.Z, m.nu, m.nv, m.U, m.V, axis, ub), m2 = makeModel(m.tg, r.U, r.V, r.Z);
    var after = evalGrid(m2.Z, m2.nu, m2.TU, m2.TV, m2.fit), d = 0, i, e;
    for (i = 0; i < after.length; i++) { e = Math.abs(after[i] - before[i]); if (e > d) d = e; }
    return { model: m2, axis: axis, knot: ub, k: r.k, dz: d, added: m2.nu * m2.nv - m.nu * m.nv };
  }
  /* The rule: among all U- and V-spans that still hold at least `minSamples` sample lines, take
     the one that holds the most squared error (mean squared residual x samples in the span, i.e.
     the span's share of the total) and put the new knot at its midpoint.
     CFG.SCORE = 'mean' ranks by the mean alone instead. On a tensor-product net that starves one
     axis: a narrow span crossing the crater keeps the highest MEAN however finely its own axis is
     cut, because the error left in it is the OTHER axis's to fix (see the node test). */
  function worstSpan(ms, minSamples) {
    var best = null, pick = function (spans, axis) {
      for (var s = 0; s < spans.length; s++) {
        var sp = spans[s];
        var sc = CFG.SCORE === 'mean' ? sp.mse : sp.mse * sp.n;
        if (sp.n >= minSamples && (!best || sc > best.score)) best = { axis: axis, idx: s, a: sp.a, b: sp.b, mse: sp.mse, score: sc, knot: 0.5 * (sp.a + sp.b) };
      }
    };
    pick(ms.su, 'u'); pick(ms.sv, 'v');
    return best;
  }

  /* ---------- Layer A, down: knot removal = remove + re-fit ---------- */
  /* Exact removal generally does not exist (the surface would have to be C^3 across the knot).
     So: drop interior knot `idx` of the axis, start from the least-squares projection of the
     CURRENT surface onto the smaller spline space, and let the optimizer re-fit the target. */
  function removeFrom(m, axis, idx) {
    var U = m.U, V = m.V, nu = m.nu, nv = m.nv, i, j, r, s, a;
    var knots = (axis === 'u' ? U : V).slice(); knots.splice(idx, 1);
    var n = (axis === 'u' ? nu : nv), n2 = n - 1, Told = axis === 'u' ? m.TU : m.TV;
    var Tnew = table(knots, n2, m.tg.xs), L = cholesky(gram(Tnew, n2), n2);
    var C = new Float64Array(n2 * n);                 // cross Gram  B'^T B
    for (a = 0; a < Tnew.n; a++) {
      var kn = Tnew.span[a] - P, ko = Told.span[a] - P;
      for (r = 0; r < W; r++) for (s = 0; s < W; s++) C[(kn + r) * n + ko + s] += Tnew.N[a * W + r] * Told.N[a * W + s];
    }
    var Z = m.Z, Q, line = new Float64Array(n2);
    if (axis === 'u') {
      Q = new Float64Array(n2 * nv);
      for (j = 0; j < nv; j++) {
        for (i = 0; i < n2; i++) { s = 0; for (r = 0; r < n; r++) s += C[i * n + r] * Z[j * nu + r]; line[i] = s; }
        cholSolve(L, n2, line, 0, 1);
        for (i = 0; i < n2; i++) Q[j * n2 + i] = line[i];
      }
      return makeModel(m.tg, knots, V, Q);
    }
    Q = new Float64Array(nu * n2);
    for (i = 0; i < nu; i++) {
      for (j = 0; j < n2; j++) { s = 0; for (r = 0; r < n; r++) s += C[j * n + r] * Z[r * nu + i]; line[j] = s; }
      cholSolve(L, n2, line, 0, 1);
      for (j = 0; j < n2; j++) Q[j * nu + i] = line[j];
    }
    return makeModel(m.tg, U, knots, Q);
  }
  /* Score every interior knot by the RMSE of the re-fitted surface without it; cheapest first. */
  function removalCandidates(m) {
    var out = [], push = function (axis) {
      var knots = axis === 'u' ? m.U : m.V, n = axis === 'u' ? m.nu : m.nv, idx, k2, m2;
      if (n <= W) return;
      for (idx = W; idx < n; idx++) {
        k2 = knots.slice(); k2.splice(idx, 1);
        m2 = axis === 'u' ? makeModel(m.tg, k2, m.V) : makeModel(m.tg, m.U, k2);
        out.push({ axis: axis, idx: idx, knot: knots[idx], rmse: rmseOf(m2, solveDirect(m2)),
                   removed: m.nu * m.nv - m2.nu * m2.nv });
      }
    };
    push('u'); push('v');
    out.sort(function (a, b) { return a.rmse - b.rmse; });
    return out;
  }

  /* ---------- the canonical pipeline (deterministic; the animation walks exactly these steps) ---------- */
  var CFG = {
    G: 128,              // target samples per axis
    N0: 5,               // coarse net: N0 x N0 control heights
    BUDGET: 196,         // control-point budget for auto-insertion
    MIN_SAMPLES: 4,      // a span must hold this many sample lines to be split
    TOL: 1.05,           // pruning keeps RMSE <= TOL x (RMSE before pruning)
    SCORE: 'sum'         // span score: 'sum' of squared residuals in the span, or their 'mean'
  };
  function coarseModel(tg) {
    var m = uniformModel(tg, CFG.N0, CFG.N0), mean = 0, i;
    for (i = 0; i < tg.T.length; i++) mean += tg.T[i];
    mean /= tg.T.length;
    for (i = 0; i < m.Z.length; i++) m.Z[i] = mean;      // start flat at the mean height
    return m;
  }
  /* Next insertion under the budget, or null. Model must be converged. */
  function nextInsertion(m, ms, budget) {
    var w = worstSpan(ms, CFG.MIN_SAMPLES);
    if (!w) return null;
    var cost = w.axis === 'u' ? m.nv : m.nu;
    if (budget != null && m.nu * m.nv + cost > budget) return null;
    return w;
  }
  function converged(m) { m.Z = solveDirect(m); m.cg = null; return m; }
  /* Coarse -> auto-insert to the budget. Returns the model and the history of steps. */
  function runAuto(tg, budget, onStep) {
    var m = converged(coarseModel(tg)), ms = measure(m), hist = [{ nu: m.nu, nv: m.nv, rmse: ms.rmse, max: ms.max }], w, r;
    while ((w = nextInsertion(m, ms, budget))) {
      r = insertInto(m, w.axis, w.knot);
      m = converged(r.model); ms = measure(m);
      hist.push({ nu: m.nu, nv: m.nv, rmse: ms.rmse, max: ms.max, axis: w.axis, knot: w.knot, dz: r.dz });
      if (onStep) onStep(m, ms, r);
    }
    return { model: m, hist: hist };
  }
  /* The cheapest acceptable removal under an RMSE ceiling, or { cand: null, next } when none. */
  function nextRemoval(m, ceiling) {
    var c = removalCandidates(m);
    if (!c.length) return { cand: null, next: null };
    return c[0].rmse <= ceiling ? { cand: c[0], next: c[1] || null } : { cand: null, next: c[0] };
  }
  function runPrune(m, ceiling) {
    var hist = [], r;
    while ((r = nextRemoval(m, ceiling)).cand) {
      m = converged(removeFrom(m, r.cand.axis, r.cand.idx));
      hist.push({ axis: r.cand.axis, knot: r.cand.knot, nu: m.nu, nv: m.nv, rmse: rmseOf(m, m.Z) });
    }
    return { model: m, hist: hist, blocked: r.next };
  }
  /* Uniform-knot baseline with the same net shape, trained by the same optimizer. */
  function uniformBaseline(tg, nu, nv) {
    var m = uniformModel(tg, nu, nv), mean = 0, i;
    for (i = 0; i < tg.T.length; i++) mean += tg.T[i];
    mean /= tg.T.length;
    for (i = 0; i < m.Z.length; i++) m.Z[i] = mean;
    m.iters = cgConverge(m);
    return m;
  }

  /* The strongest near-square uniform net around N control points: for every width a, both the
     largest b with a*b <= N and the smallest with a*b >= N ("nearest >="). Uniform RMSE jumps
     around with knot alignment, so this guards the comparison against a lucky or unlucky net —
     and it lets the uniform side have a few points MORE than the adaptive one. */
  function bestUniform(tg, N) {
    var best = null, a, b, k, m, r, seen = {};
    for (a = W; a * W <= N; a++) for (k = 0; k < 2; k++) {
      b = k ? Math.ceil(N / a) : Math.floor(N / a);
      if (b < W || Math.abs(a - b) > 6 || seen[a + 'x' + b]) continue;
      seen[a + 'x' + b] = 1;
      m = uniformModel(tg, a, b); r = rmseOf(m, solveDirect(m));
      if (!best || r < best.rmse) best = { nu: a, nv: b, n: a * b, rmse: r };
    }
    return best;
  }

  var MATH = {
    CFG: CFG, targetRaw: targetRaw, makeTarget: makeTarget, targetAt: targetAt, centres: centres, table: table,
    evalGrid: evalGrid, makeModel: makeModel, uniformModel: uniformModel, cgInit: cgInit, cgStep: cgStep,
    cgConverge: cgConverge, solveDirect: solveDirect, measure: measure, rmseOf: rmseOf, insertKnot: insertKnot,
    insertInto: insertInto, worstSpan: worstSpan, removeFrom: removeFrom, removalCandidates: removalCandidates,
    coarseModel: coarseModel, nextInsertion: nextInsertion, converged: converged, runAuto: runAuto,
    nextRemoval: nextRemoval, runPrune: runPrune, uniformBaseline: uniformBaseline, bestUniform: bestUniform
  };
  if (NODE) { module.exports = MATH; return; }

  /* =====================================================================================
     PART 2 — the instrument (Canvas 2D)
     ===================================================================================== */
  var KT = root.KnotTalk, C = KT.color;
  function rgbOf(hex) { var n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function rgba(key, a) { var c = rgbOf(C[key]); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
  function pct(x, d) { return (100 * x).toFixed(d == null ? 2 : d) + ' %'; }
  function expStr(x) { return x === 0 ? '0' : x.toExponential(1).replace('e-', 'e−'); }
  function thousands(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }

  /* colour ramps -> 256-entry LUTs */
  function ramp(stops) {
    var lut = new Float32Array(256 * 4), i, s, a, b, t, k;
    for (i = 0; i < 256; i++) {
      t = i / 255; s = 0;
      while (s < stops.length - 2 && t > stops[s + 1][0]) s++;
      a = stops[s]; b = stops[s + 1]; k = KT.clamp((t - a[0]) / (b[0] - a[0]), 0, 1);
      var ca = rgbOf(a[1]), cb = rgbOf(b[1]);
      lut[i * 4] = ca[0] + (cb[0] - ca[0]) * k; lut[i * 4 + 1] = ca[1] + (cb[1] - ca[1]) * k;
      lut[i * 4 + 2] = ca[2] + (cb[2] - ca[2]) * k;
      lut[i * 4 + 3] = (a[2] == null ? 1 : a[2]) + ((b[2] == null ? 1 : b[2]) - (a[2] == null ? 1 : a[2])) * k;
    }
    return lut;
  }
  var PAL_FIT = ramp([[0, '#0a2236'], [0.30, '#0f3c5c'], [0.58, '#1f739f'], [0.80, '#4fc3f7'], [1, '#dcf4ff']]);
  var PAL_TGT = ramp([[0, '#161c26'], [0.30, '#2c3644'], [0.58, '#58677b'], [0.80, '#96a5b9'], [1, '#ecf0f6']]);
  var PAL_ERR = ramp([[0, '#2a0c0e', 0], [0.14, '#521819', 0.46], [0.42, '#b93b39', 0.92], [0.72, '#ef5350', 1], [1, '#ffd9d2', 1]]);
  var EGAM = 0.9;                         // display gamma of the error heat
  var LX = -0.50, LY = -0.62, LZ = 0.62, ZS = 0.55, GSAT = 1.15, ZG = 0.35;
  (function () { var n = Math.sqrt(LX * LX + LY * LY + LZ * LZ); LX /= n; LY /= n; LZ /= n; })();

  /* Hill-shaded relief of H (n x n) into RGBA px. E (optional) = signed error field -> red heat. */
  function shade(px, H, n, pal, E, escale) {
    var x, y, i, o, gx, gy, inv, s, lum, t, li, r, g, b, k = n * 0.5 * ZS, e, a, hi, c0;
    for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
      i = y * n + x; o = i * 4;
      gx = (H[x < n - 1 ? i + 1 : i] - H[x > 0 ? i - 1 : i]) * (x > 0 && x < n - 1 ? k : 2 * k);
      gy = (H[y < n - 1 ? i + n : i] - H[y > 0 ? i - n : i]) * (y > 0 && y < n - 1 ? k : 2 * k);
      inv = 1 / Math.sqrt(1 + (gx * gx + gy * gy) / (GSAT * GSAT));    // soft-limit steep slopes so the
      gx *= inv; gy *= inv;                                             // rolling ground still reads
      inv = 1 / Math.sqrt(gx * gx + gy * gy + 1);
      s = (LZ - gx * LX - gy * LY) * inv / LZ;                 // 1 on flat ground
      lum = 0.24 + 0.54 * s; lum = lum < 0.18 ? 0.18 : lum > 1.16 ? 1.16 : lum;
      t = H[i]; t = t < 0 ? 0 : t > 1 ? 1 : t;
      // faint contour every 0.1 of relief
      if (!E && x < n - 1 && y < n - 1) { c0 = Math.floor(H[i] * 10); if (c0 !== Math.floor(H[i + 1] * 10) || c0 !== Math.floor(H[i + n] * 10)) lum *= 0.88; }
      li = ((t * 255) | 0) * 4;
      if (E) {
        r = 15 * lum; g = 20 * lum; b = 29 * lum;
        e = Math.abs(E[i]) / escale; e = e > 1 ? 1 : e; e = Math.pow(e, EGAM);
        li = ((e * 255) | 0) * 4; a = PAL_ERR[li + 3];
        r += (PAL_ERR[li] - r) * a; g += (PAL_ERR[li + 1] - g) * a; b += (PAL_ERR[li + 2] - b) * a;
      } else {
        r = pal[li] * lum; g = pal[li + 1] * lum; b = pal[li + 2] * lum;
        if (lum > 1) { hi = (lum - 1) * 1.3; r += (255 - r) * hi; g += (255 - g) * hi; b += (255 - b) * hi; }
      }
      px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = 255;
    }
  }

  var CSS_DONE = false;
  function injectCSS() {
    if (CSS_DONE) return; CSS_DONE = true;
    var s = document.createElement('style');
    s.textContent = [
      '.kt-sim-refine .rf-wrap{position:absolute;inset:0;display:flex}',
      '.kt-sim-refine .rf-wrap.rf-side{flex-direction:row}',
      '.kt-sim-refine .rf-wrap.rf-stack{flex-direction:column}',
      '.kt-sim-refine .rf-stage{position:relative;flex:1 1 0;min-width:0;min-height:0}',
      '.kt-sim-refine .rf-col{flex:none;display:flex;gap:10px;min-width:0}',
      '.kt-sim-refine .rf-side .rf-col{flex-direction:column;justify-content:center;width:var(--rf-col,268px);padding:14px 14px 14px 0}',
      '.kt-sim-refine .rf-stack .rf-col{flex-direction:row;align-items:stretch;padding:0 14px 12px}',
      '.kt-sim-refine .rf-col .kt-panel{position:static;transform:none;max-width:none;min-width:0}',
      '.kt-sim-refine .rf-stack .rf-col .kt-panel{flex:1 1 0}',
      '.kt-sim-refine .rf-read .kt-val{font-size:15px}',
      '.kt-sim-refine .rf-read .kt-panel-title{margin-top:2px}',
      '.kt-sim-refine .rf-read .rf-t2{margin-top:8px;padding-top:9px;border-top:1px solid var(--line)}',
      '.kt-sim-refine .rf-hide{display:none!important}',
      '.kt-sim-refine .rf-btns{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}',
      '.kt-sim-refine .rf-btns.rf-tog{grid-template-columns:repeat(2,minmax(0,1fr))}',
      '.kt-sim-refine .rf-btns .kt-btn,.kt-sim-refine .rf-btns .kt-toggle{padding-left:2px;padding-right:2px;text-align:center}',
      '.kt-sim-refine .kt-toggle.rf-g.on{color:var(--splat);border-color:rgba(var(--splat-rgb),.55);background:rgba(var(--splat-rgb),.1)}',
      '.kt-sim-refine .rf-dens input{accent-color:var(--splat)}',
      '.kt-sim-refine .rf-dens input::-webkit-slider-thumb{background:var(--splat)}',
      '.kt-sim-refine .rf-dens input::-moz-range-thumb{background:var(--splat)}',
      /* stacked: readouts become a grid of label-over-value cells */
      '.kt-sim-refine .rf-stack .rf-read{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 14px;align-content:center;flex:1.25 1 0}',
      '.kt-sim-refine .rf-stack .rf-read .kt-panel-title{display:none}',
      '.kt-sim-refine .rf-stack .rf-read .kt-readout{flex-direction:column;align-items:flex-start;gap:3px;min-height:0}',
      '.kt-sim-refine .rf-stack .rf-read .kt-val{margin-left:0}',
      '.kt-sim-refine .rf-stack .rf-ctrl{justify-content:center}',
      /* projector-sized stage: bigger type */
      '.kt-sim-refine .rf-big .kt-lab{font-size:13px}',
      '.kt-sim-refine .rf-big .rf-read .kt-val{font-size:19px}',
      '.kt-sim-refine .rf-big .kt-seg-btn,.kt-sim-refine .rf-big .kt-btn,.kt-sim-refine .rf-big .kt-toggle{font-size:13px;padding:9px 12px}',
      '.kt-sim-refine .rf-big .kt-panel-title{font-size:11px}',
      '.kt-sim-refine .rf-big .kt-panel{padding:13px 15px;gap:9px}',
      /* phone-sized stage: everything in two tight strips */
      '.kt-sim-refine .rf-tiny .rf-col{flex-direction:column;gap:5px;padding:0 8px 8px}',
      '.kt-sim-refine .rf-tiny .rf-col .kt-panel{flex:none;padding:6px 9px;border-radius:8px}',
      '.kt-sim-refine .rf-tiny .rf-read{gap:4px 8px;grid-template-columns:repeat(5,minmax(0,1fr))}',
      '.kt-sim-refine .rf-tiny .rf-read .kt-val{font-size:12px}',
      '.kt-sim-refine .rf-tiny .rf-read .kt-lab{font-size:10px}',
      '.kt-sim-refine .rf-tiny .rf-read .kt-readout{gap:1px}',
      '.kt-sim-refine .rf-tiny .rf-x{display:none}',
      '.kt-sim-refine .rf-tiny .rf-ctrl{flex-direction:row;flex-wrap:wrap;align-items:center;gap:5px 6px}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-panel-title,.kt-sim-refine .rf-tiny .rf-ctrl .kt-seg .kt-lab{display:none}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-row{min-height:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-seg-box{margin-left:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .rf-btns{flex:1 1 150px;gap:4px}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .rf-dens{flex:1 1 150px;min-width:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .rf-dens input{width:0;min-width:50px}',
      '.kt-sim-refine .rf-tiny.rf-vs .rf-x2{display:none}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .rf-tog{flex:0 0 132px}',
      '.kt-sim-refine .rf-tiny .kt-seg-btn,.kt-sim-refine .rf-tiny .kt-btn,.kt-sim-refine .rf-tiny .kt-toggle{padding:6px 6px}',
      /* kit "compact" (small embedded stage): the readout strip stays, the controls fold behind the
         kit's own "controls" button and open as an overlay so the figure never reflows */
      '.kt-sim-refine.kt-compact .rf-col .rf-read{display:grid}',
      '.kt-sim-refine.kt-compact .rf-col .rf-ctrl{position:absolute;left:8px;right:8px;top:40px;z-index:7}'
    ].join('\n');
    document.head.appendChild(s);
  }

  /* one target + canonical results per page, shared by every mounted instance */
  var SHARED = null;
  function shared() {
    if (SHARED) return SHARED;
    var tg = makeTarget(CFG.G), S = { tg: tg };
    function snapOf(m) { return { U: m.U.slice(), V: m.V.slice(), Z: new Float64Array(m.Z) }; }
    S.restore = function (s) { var m = makeModel(tg, s.U.slice(), s.V.slice(), new Float64Array(s.Z)); cgInit(m); return m; };
    S.snapOf = snapOf;
    S.coarse = function () { if (!S._c) { var c = converged(coarseModel(tg)); S._c = snapOf(c); S.coarseRmse = measure(c).rmse; } return S._c; };
    S.auto = function () {
      if (!S._a) { var r = runAuto(tg, CFG.BUDGET); S._a = snapOf(r.model); S.autoHist = r.hist; S.autoRmse = r.hist[r.hist.length - 1].rmse; }
      return S._a;
    };
    S.fin = function () {
      if (!S._f) {
        var a = S.restore(S.auto()), pr = runPrune(a, S.autoRmse * CFG.TOL);
        S._f = snapOf(pr.model); S.pruneHist = pr.hist; S.pruneBlocked = pr.blocked;
      }
      return S._f;
    };
    return (SHARED = S);
  }

  var SCENES = [
    { id: 'coarse', label: 'Coarse net' }, { id: 'insert', label: 'Insert one knot' },
    { id: 'auto', label: 'Auto-insert' }, { id: 'remove', label: 'Remove + re-fit' },
    { id: 'versus', label: 'Adaptive vs uniform' }, { id: 'sampling', label: 'Layer B · sampling' }
  ];
  var PACE = {
    sceneInsert: { delay: 1.0, hi: 1.5, hold: 2.0, rate: 9 },
    button: { delay: 0, hi: 0.55, hold: 0.9, rate: 36 },
    auto: { gap: 0.8, hi: 0.26, hold: 0.12, rate: 260 },
    prune: { gap: 1.15, hi: 0.5, rate: 200 }
  };

  KT.register('refine', {
    title: 'Two-layer density',
    blurb: 'A cubic B-spline height field fitted to a terrain, live. Layer A adds control where the per-span error ' +
           'is worst — knot insertion leaves the surface exactly unchanged — and prunes knots that buy nothing. ' +
           'Layer B only changes how many Gaussians are sampled.',
    tool: 'Canvas 2D · live least-squares fit',
    scenes: SCENES,
    create: function (env) {
      injectCSS();
      var rootEl = env.root, ui = env.ui, S = shared(), tg = S.tg;
      var cs = getComputedStyle(rootEl);
      var F_MONO = cs.getPropertyValue('--f-mono').trim() || 'ui-monospace,monospace';
      var F_DISP = cs.getPropertyValue('--f-display').trim() || 'system-ui,sans-serif';

      /* ---------- DOM ---------- */
      var wrap = ui.el('div', 'rf-wrap rf-side'), stage = ui.el('div', 'rf-stage'), col = ui.el('div', 'rf-col');
      rootEl.appendChild(wrap); wrap.appendChild(stage); wrap.appendChild(col);
      var cv = ui.canvas(stage), ctx = cv.ctx;

      var pr = ui.panel(col, { title: 'Layer A · control density' }); pr.classList.add('rf-read');
      var LABELS = [];
      function readout(full, short, tone, cls) {
        var r = ui.readout(pr, { label: full, tone: tone }); if (cls) r.el.classList.add(cls);
        LABELS.push({ el: r.el.querySelector('.kt-lab'), full: full, short: short });
        return r;
      }
      var rdNet = readout('control points', 'net', 'ctrl');
      var rdPar = readout('learnable params', 'params', 'ctrl');
      var rdKn = readout('interior knots', 'knots', 'ctrl', 'rf-x');
      var rdRmse = readout('RMSE', 'RMSE', 'err');
      var rdMax = readout('max error', 'max err', 'err', 'rf-x2');
      var rdUni = readout('uniform RMSE', 'uniform', '', 'rf-hide');
      var t2 = ui.el('div', 'kt-panel-title rf-t2', 'Layer B · sampling density'); pr.appendChild(t2);
      var rdG = readout('Gaussians', 'Gaussians', 'splat');

      var pc = ui.panel(col, { title: 'controls' }); pc.classList.add('rf-ctrl');
      var segMap = ui.seg(pc, { label: 'map', value: 'fit',
        options: [{ v: 'fit', label: 'fit' }, { v: 'target', label: 'target' }, { v: 'error', label: 'error' }],
        onChange: function (v) { st.view = v; dirty = true; } });
      var rowB = ui.el('div', 'kt-row rf-btns'); pc.appendChild(rowB);
      ui.button(rowB, { label: 'insert', onClick: function () { userInsert(); } });
      ui.button(rowB, { label: 'remove', onClick: function () { userRemove(); } });
      ui.button(rowB, { label: 'auto', onClick: function () { userAuto(); } });
      ui.button(rowB, { label: 'reset', onClick: function () { userReset(); } });
      var slDens = ui.slider(pc, { label: 'samples', min: 6, max: 64, step: 1, value: 24,
        fmt: function (v) { return v + '²'; },
        onInput: function (v) { st.dens = v; st.densAuto = false; if (!st.gauss) { st.gauss = true; tgG.set(true); } } });
      slDens.el.classList.add('rf-dens');
      var rowT = ui.el('div', 'kt-row rf-btns rf-tog'); pc.appendChild(rowT);
      var tgRun = ui.toggle(rowT, { label: 'run', value: true, onChange: function (v) { st.run = v; } });
      var tgG = ui.toggle(rowT, { label: 'gaussians', value: false, onChange: function (v) { st.gauss = v; st.densAuto = false; } });
      tgG.el.classList.add('rf-g');

      if (env.embed) {                    // inside the deck, never keep keyboard focus: space and arrows belong to it
        col.addEventListener('click', function () { var a = document.activeElement; if (a && col.contains(a) && a.blur) a.blur(); });
        slDens.input.addEventListener('change', function () { slDens.input.blur(); });
      }

      /* ---------- state ---------- */
      var st = { scene: '', run: true, view: 'fit', gauss: false, dens: 24, densAuto: false, densT: 0,
                 mode: 'idle', timer: 0, rate: 12, acc: 0, op: null, pending: null, versus: false,
                 ceiling: null, blocked: null, pruned: 0, ptsFrom: 0, inserted: 0, barScale: 0, holdScale: false,
                 dotsOff: false };
      var m = null, ms = null, base = null, baseMs = null, bestU = null;
      var fx = [], dirty = true, dirtyB = true, zVersion = 0;
      var status = { l1: '', l2: '', tone: 'text', live: null };
      var Wc = 0, Hc = 0, L = null, tiny = false, big = false;

      /* display buffers */
      var disp = { n: 0 };
      function setDispRes(n) {
        if (disp.n === n) return;
        disp.n = n; disp.xs = centres(n);
        disp.tgt = new Float32Array(n * n);
        for (var b = 0; b < n; b++) for (var a = 0; a < n; a++) disp.tgt[b * n + a] = targetAt(tg, disp.xs[a], disp.xs[b]);
        disp.fitA = new Float32Array(n * n); disp.fitB = new Float32Array(n * n); disp.err = new Float32Array(n * n);
        disp.cA = mkCanvas(n); disp.cB = mkCanvas(n);
        disp.tabA = null; disp.tabB = null; dirty = dirtyB = true;
      }
      function mkCanvas(n) {
        var c = document.createElement('canvas'); c.width = c.height = n;
        var x = c.getContext('2d'); return { c: c, x: x, img: x.createImageData(n, n) };
      }
      function tablesFor(mod) { return { m: mod, TU: table(mod.U, mod.nu, disp.xs), TV: table(mod.V, mod.nv, disp.xs) }; }
      function paintMap(cvs, mod, tab, fitBuf, view, escale) {
        var n = disp.n, i;
        evalGrid(mod.Z, mod.nu, tab.TU, tab.TV, fitBuf);
        if (view === 'target') shade(cvs.img.data, disp.tgt, n, PAL_TGT);
        else if (view === 'error') {
          for (i = 0; i < n * n; i++) disp.err[i] = fitBuf[i] - disp.tgt[i];
          shade(cvs.img.data, fitBuf, n, PAL_TGT, disp.err, escale);
        } else shade(cvs.img.data, fitBuf, n, PAL_FIT);
        cvs.x.putImageData(cvs.img, 0, 0);
      }

      /* ---------- model plumbing ---------- */
      function setModel(mod) {
        m = mod; if (!m.cg) cgInit(m);
        ms = measure(m); disp.tabA = null; dirty = true; st.acc = 0; zVersion++;
      }
      function snap() {                    // land exactly on the least-squares optimum before a decision
        m.Z = solveDirect(m); cgInit(m); m.cg.done = true; ms = measure(m); dirty = true; zVersion++;
      }
      function setStatus(l1, l2, tone, live) { status.l1 = l1; status.l2 = l2 || ''; status.tone = tone || 'text'; status.live = live || null; }
      function netStr(mod) { return mod.nu + ' × ' + mod.nv; }
      function clearRun() { st.op = null; st.pending = null; st.mode = 'idle'; st.timer = 0; st.acc = 0; fx.length = 0; st.holdScale = false; }

      /* ---------- Layer A beats ---------- */
      function beginInsert(pace, auto) {
        snap();
        var w = nextInsertion(m, ms, auto ? CFG.BUDGET : null);
        if (!w) return false;
        st.op = { type: 'ins', w: w, phase: 0, t: 0, pace: pace, auto: !!auto };
        if (!auto) {
          var share = w.mse * (w.axis === 'u' ? ms.su : ms.sv)[w.idx].n / tg.G / (ms.rmse * ms.rmse);
          setStatus('WORST SPAN · ' + w.axis + ' ∈ [' + w.a.toFixed(3) + ', ' + w.b.toFixed(3) + ']',
                    'holds ' + Math.round(100 * share) + ' % of the squared error · knot goes at its midpoint', 'err');
        }
        return true;
      }
      function doInsert(op) {
        var w = op.w, before = ms.rmse, r = insertInto(m, w.axis, w.knot);
        if (r.dz > 1e-9 && root.console) root.console.error('refine: knot insertion moved the surface by', r.dz);
        setModel(r.model); st.inserted++; st.ceiling = null;
        fx.push({ kind: 'new', axis: w.axis, knot: w.knot, age: 0, life: op.auto ? 1.6 : 3.2 });
        op.dz = r.dz; op.before = before; op.added = r.added;
        var where = w.axis + ' = ' + w.knot.toFixed(3);
        if (op.auto) {
          setStatus('AUTO-INSERT · KNOT ' + st.inserted + ' · ' + where + ' · +' + r.added + ' CONTROL POINTS', '', 'ctrl',
            function () { return 'surface unchanged (max |Δz| ' + expStr(op.dz) + ') · RMSE ' + pct(st.rmse0) + ' → ' + pct(ms.rmse); });
        } else {
          setStatus(r.dz <= 1e-9 ? 'SURFACE UNCHANGED · +' + r.added + ' CONTROL POINTS' : 'SURFACE MOVED — INSERTION BUG',
                    'Boehm insertion at ' + where + ' · max |Δz| = ' + expStr(r.dz) + ' · RMSE still ' + pct(before), r.dz <= 1e-9 ? 'ok' : 'err');
        }
      }
      function beginRemove(pace) {
        snap();
        if (st.ceiling == null) st.ceiling = ms.rmse * CFG.TOL;
        var r = nextRemoval(m, st.ceiling);
        if (!r.cand) { st.blocked = r.next; return false; }
        st.op = { type: 'rem', cand: r.cand, phase: 0, t: 0, pace: pace };
        setStatus('REDUNDANT KNOT · ' + r.cand.axis + ' = ' + r.cand.knot.toFixed(3),
                  're-fit without it → RMSE ' + pct(r.cand.rmse, 3) + ' · tolerance ' + pct(st.ceiling, 3), 'ctrl');
        return true;
      }
      function doRemove(op) {
        var c = op.cand, before = ms.rmse, n0 = m.nu * m.nv;
        fx.push({ kind: 'gone', axis: c.axis, knot: c.knot, age: 0, life: 1.1 });
        setModel(removeFrom(m, c.axis, c.idx)); st.pruned++;
        var ceil = st.ceiling;
        setStatus('REMOVE + RE-FIT · ' + c.axis + ' = ' + c.knot.toFixed(3) + ' · −' + (n0 - m.nu * m.nv) + ' CONTROL POINTS', '', 'ctrl',
          function () { return 'RMSE ' + pct(before, 3) + ' → ' + pct(ms.rmse, 3) + ' · tolerance ' + pct(ceil, 3); });
      }
      function blockedStatus(head) {
        var b = st.blocked;
        setStatus(head, b ? 'next cheapest removal → RMSE ' + pct(b.rmse, 3) + ' · over the ' + pct(st.ceiling, 3) + ' tolerance'
                          : 'no interior knots left', st.pruned ? 'ok' : 'text');
      }
      function stepOp(dt) {
        var op = st.op; op.t += dt;
        if (op.type === 'ins') {
          if (op.phase === 0 && op.t >= op.pace.hi) { doInsert(op); op.phase = 1; op.t = 0; }
          else if (op.phase === 1 && op.t >= op.pace.hold) {
            st.rate = op.pace.rate; st.op = null;
            if (!op.auto) setStatus('THE OPTIMIZER USES THE NEW FREEDOM', '', 'surf',
              function () { return 'RMSE ' + pct(op.before) + ' → ' + pct(ms.rmse) + (m.cg.done ? ' · converged' : ''); });
          }
        } else if (op.phase === 0 && op.t >= op.pace.hi) { doRemove(op); st.rate = op.pace.rate; st.op = null; }
      }
      function optimize(dt) {
        st.acc += st.rate * dt;
        var n = Math.floor(st.acc); if (n > 24) { n = 24; st.acc = 24; }
        if (n < 1) return;
        st.acc -= n;
        while (n-- > 0 && cgStep(m)) { /* one CG step */ }
        ms = measure(m); dirty = true; zVersion++;
        if (m.cg.done) onConverged();
      }
      function onConverged() {
        if (st.scene === 'coarse' && st.mode === 'idle' && m.nu === CFG.N0 && m.nv === CFG.N0 && !st.inserted)
          setStatus('SMOOTH GROUND FITS · THE CRATER DOES NOT', netStr(m) + ' net · RMSE ' + pct(ms.rmse) + ' · the error sits in one u-span and one v-span', 'text');
      }
      function autoNext() {
        st.timer = 0;
        if (!beginInsert(PACE.auto, true)) {
          st.mode = 'idle';
          setStatus('BUDGET REACHED · ' + m.nu * m.nv + ' / ' + CFG.BUDGET + ' CONTROL POINTS',
                    'RMSE ' + pct(st.rmse0) + ' → ' + pct(ms.rmse) + ' · ' + st.inserted + ' knots, clustered over the crater', 'ok');
        }
      }
      function pruneNext() {
        st.timer = 0;
        if (!beginRemove(PACE.prune)) {
          st.mode = 'idle';
          blockedStatus(st.pruned ? 'PRUNED ' + st.pruned + ' KNOT' + (st.pruned > 1 ? 'S' : '') + ' · ' + st.ptsFrom + ' → ' + m.nu * m.nv + ' CONTROL POINTS'
                                  : 'NOTHING TO PRUNE WITHIN TOLERANCE');
        }
      }

      /* ---------- user controls ---------- */
      function busy() { return !!st.op; }
      function userInsert() {
        if (busy()) return; st.mode = 'idle'; st.pending = null; st.rmse0 = st.rmse0 || ms.rmse;
        if (!beginInsert(PACE.button, false)) setStatus('NO SPAN LEFT TO SPLIT', 'every span is already down to ' + CFG.MIN_SAMPLES + ' sample lines', 'text');
      }
      function userRemove() {
        if (busy()) return; st.mode = 'idle'; st.pending = null;
        if (!st.pruned) st.ptsFrom = m.nu * m.nv;
        if (!beginRemove(PACE.button)) blockedStatus('NO KNOT CAN GO WITHIN TOLERANCE');
      }
      function userAuto() {
        if (busy()) return; st.pending = null; st.mode = 'auto'; st.timer = PACE.auto.gap; st.rmse0 = st.rmse0 || ms.rmse;
      }
      function userReset() { enterCoarse(true); }

      /* ---------- scenes ---------- */
      function setView(v) { st.view = v; segMap.set(v); dirty = dirtyB = true; }
      function setGauss(on) { st.gauss = on; tgG.set(on); }
      function enterCoarse(animate) {
        clearRun(); st.inserted = 0; st.pruned = 0; st.ceiling = null; st.blocked = null;
        if (animate) {
          var c = coarseModel(tg); setModel(c); st.rate = 9;
          setStatus('COARSE NET · ' + netStr(c) + ' CONTROL HEIGHTS', '', 'ctrl',
            function () { return 'conjugate gradients on the least-squares fit · RMSE ' + pct(ms.rmse); });
        } else setModel(S.restore(S.coarse()));
        S.coarse(); st.rmse0 = S.coarseRmse;
        st.barScale = 0;
      }
      function setScene(id) {
        var known = false, i; for (i = 0; i < SCENES.length; i++) if (SCENES[i].id === id) known = true;
        if (!known) return;
        st.scene = id; st.barScale = 0; st.versus = id === 'versus'; st.densAuto = false; st.dotsOff = id === 'sampling';
        rdUni.el.classList.toggle('rf-hide', !st.versus);
        if (!st.run) { st.run = true; tgRun.set(true); }
        if (id === 'coarse') { enterCoarse(true); setView('fit'); setGauss(false); }
        else if (id === 'insert') {
          enterCoarse(false); setView('fit'); setGauss(false); st.holdScale = true;
          setStatus('COARSE NET · ' + netStr(m) + ' · RMSE ' + pct(ms.rmse), 'where is the error? measured per u-span and per v-span', 'text');
          st.pending = { delay: PACE.sceneInsert.delay, fn: function () { beginInsert(PACE.sceneInsert, false); } };
        } else if (id === 'auto') {
          enterCoarse(false); setView('fit'); setGauss(false);
          setStatus('AUTO-INSERT · BUDGET ' + CFG.BUDGET + ' CONTROL POINTS', 'one knot every ' + PACE.auto.gap + ' s · midpoint of the span holding the most error', 'ctrl');
          st.mode = 'auto'; st.timer = PACE.auto.gap - 0.7;
        } else if (id === 'remove') {
          clearRun(); setModel(S.restore(S.auto())); setView('fit'); setGauss(false);
          st.inserted = S.autoHist.length - 1; st.pruned = 0; st.ptsFrom = m.nu * m.nv; st.rmse0 = S.autoHist[0].rmse;
          st.ceiling = S.autoRmse * CFG.TOL; st.blocked = null;
          setStatus('PRUNE · ' + m.nu * m.nv + ' CONTROL POINTS · RMSE ' + pct(ms.rmse, 3),
                    'drop the cheapest knot, re-fit · keep RMSE ≤ ' + pct(st.ceiling, 3) + ' (+' + Math.round((CFG.TOL - 1) * 100) + ' %)', 'ctrl');
          st.mode = 'prune'; st.timer = PACE.prune.gap - 1.0;
        } else {
          clearRun(); setModel(S.restore(S.fin()));
          st.inserted = S.autoHist.length - 1; st.pruned = S.pruneHist.length; st.rmse0 = S.autoHist[0].rmse; st.ceiling = null;
          if (id === 'versus') { setView('error'); setGauss(false); setStatus('', '', 'text'); }
          else {
            setView('fit'); setGauss(true); st.densAuto = true; st.densT = 0;
            setStatus('', '', 'splat');
          }
        }
        relayout();
      }

      function refreshBase() {
        var key = m.nu + 'x' + m.nv;
        if (base && base.key === key) return;
        base = uniformBaseline(tg, m.nu, m.nv); base.key = key; baseMs = measure(base);
        bestU = bestUniform(tg, m.nu * m.nv); disp.tabB = null; dirtyB = true;
      }

      /* ---------- per-frame step ---------- */
      function step(dt) {
        var i;
        for (i = fx.length - 1; i >= 0; i--) { fx[i].age += dt; if (fx[i].age > fx[i].life) fx.splice(i, 1); }
        if (st.op) stepOp(dt);
        else {
          if (!m.cg.done) optimize(dt);
          if (st.pending) { st.timer += dt; if (st.timer >= st.pending.delay) { var f = st.pending.fn; st.pending = null; st.timer = 0; f(); } }
          else if (st.mode === 'auto') { if (st.timer >= PACE.auto.gap) autoNext(); }
          else if (st.mode === 'prune') { if (st.timer >= PACE.prune.gap) pruneNext(); }
        }
        if (st.mode !== 'idle' && !st.pending) st.timer += dt;      // the cadence includes the beat itself
        if (st.densAuto) {
          st.densT += dt;
          var d = Math.round(KT.lerp(8, 56, 0.5 - 0.5 * Math.cos(st.densT * 2 * Math.PI / 7)));
          if (d !== st.dens) { st.dens = d; slDens.set(d); }
        }
      }

      /* ---------- layout ---------- */
      function relayout() { if (Wc > 0) layout(Wc, Hc); }
      function layout(w, h) {
        Wc = w; Hc = h;
        tiny = w < 560 || h < 470; big = !tiny && Math.min(w, h) >= 860 && w >= 1000;
        var colW = big ? 330 : env.embed ? 236 : 268;
        var squat = tiny && h < 470;      // phone-sized deck stage: one status line only
        var bar = tiny ? 28 : KT.clamp(Math.min(w, h) * 0.09, 30, 84), chrome = bar + (tiny ? 62 : 110);
        var sideMs = Math.min(w - colW - chrome, h - chrome), stackMs = Math.min(w - chrome, h - (big ? 150 : 128) - chrome);
        if (st.versus) { sideMs = Math.min((w - colW) / 2, h - 150); stackMs = Math.min(w / 2, h - 128 - 150); }
        var side = !tiny && sideMs >= stackMs;
        wrap.className = 'rf-wrap ' + (side ? 'rf-side' : 'rf-stack') + (tiny ? ' rf-tiny' : '') + (big ? ' rf-big' : '') + (st.versus ? ' rf-vs' : '');
        for (var q = 0; q < LABELS.length; q++) LABELS[q].el.textContent = tiny ? LABELS[q].short : LABELS[q].full;
        wrap.style.setProperty('--rf-col', colW + 'px');
        var sz = cv.fit(), W = sz.w, H = sz.h, top = rootEl.classList.contains('kt-compact') ? 30 : 0;   // kit's button
        var fs = tiny ? 11 : big ? 14 : 12, pad = tiny ? 8 : big ? 22 : 16;
        L = { W: W, H: H, fs: fs, pad: pad, f1: tiny ? 12 : big ? 18 : 14, f2: tiny ? 11 : big ? 14 : 12 };
        L.squat = squat; L.statusH = squat ? 22 : tiny ? 52 : L.f1 + 2 * L.f2 + 30;
        if (st.versus) {
          var gapM = tiny ? 10 : 28, head = fs + 12, foot = tiny ? 110 : big ? 190 : 156;
          var sp = Math.floor(Math.min((W - 2 * pad - gapM) / 2, H - top - 2 * pad - head - foot));
          L.sp = sp; L.gap = gapM; L.head = head; L.foot = foot;
          L.x0 = Math.round((W - (2 * sp + gapM)) / 2); L.y0 = top + Math.round((H - top - (head + sp + foot)) / 2) + head;
          setDispRes(pickRes(sp, sz.dpr));
        } else {
          var gap = tiny ? 5 : 8, lab = fs + 8, rot = fs + 8;
          L.bar = Math.round(bar); L.gap = gap;
          var msz = Math.floor(Math.min(W - 2 * pad - L.bar - gap - rot, H - top - 2 * pad - lab - L.bar - gap - L.statusH));
          L.ms = msz;
          var tw = msz + gap + L.bar + rot, th = lab + L.bar + gap + msz + L.statusH;
          L.mx = Math.round((W - tw) / 2); L.my = top + Math.round((H - top - th) / 2) + lab + L.bar + gap;
          setDispRes(pickRes(msz, sz.dpr));
        }
        dirty = dirtyB = true;
      }
      function pickRes(px, dpr) { return px * dpr > 620 ? 384 : px * dpr > 330 ? 320 : 224; }

      /* ---------- drawing ---------- */
      function font(w, px, fam) { return w + ' ' + px + 'px ' + (fam || F_MONO); }
      function txt(s, x, y, f, color, align, ls) {
        ctx.font = f; ctx.fillStyle = color; ctx.textAlign = align || 'left'; ctx.textBaseline = 'alphabetic';
        if ('letterSpacing' in ctx) ctx.letterSpacing = ls || '0px';
        ctx.fillText(s, x, y);
      }
      /* text that must fit `maxW`: drop tracking, then size, then truncate */
      function fitTxt(s, x, y, w, px, color, align, ls, maxW) {
        var f = font(w, px), p = px;
        ctx.font = f; if ('letterSpacing' in ctx) ctx.letterSpacing = ls || '0px';
        if (ctx.measureText(s).width > maxW) { ls = '0px'; ctx.letterSpacing = ls; }
        while (ctx.measureText(s).width > maxW && p > 11) { p -= 0.5; ctx.font = font(w, p); }
        if (ctx.measureText(s).width > maxW) { while (s.length > 4 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1); s += '…'; }
        txt(s, x, y, font(w, p), color, align, ls);
      }
      /* secondary text: one line if it fits, else broken once at a ' · ' separator */
      function flowTxt(s, x, y, px, color, maxW, lead) {
        ctx.font = font(500, px); if ('letterSpacing' in ctx) ctx.letterSpacing = '0px';
        if (ctx.measureText(s).width <= maxW) { fitTxt(s, x, y, 500, px, color, 'left', '0.02em', maxW); return; }
        var parts = s.split(' · '), a = parts[0], i = 1;
        while (i < parts.length - 1 && ctx.measureText(a + ' · ' + parts[i]).width <= maxW) a += ' · ' + parts[i++];
        if (parts.length < 2) { fitTxt(s, x, y, 500, px, color, 'left', '0.02em', maxW); return; }
        fitTxt(a, x, y, 500, px, color, 'left', '0.02em', maxW);
        fitTxt(parts.slice(i).join(' · '), x, y + lead, 500, px, color, 'left', '0.02em', maxW);
      }
      function toneCol(t) { return t === 'ok' ? C.ok : t === 'err' ? C.err : t === 'ctrl' ? C.ctrl : t === 'splat' ? C.splat : t === 'surf' ? C.surf : C.text; }

      function drawKnots(mod, x, y, s, alpha, lw) {
        var i, p;
        ctx.strokeStyle = rgba('ctrl', alpha); ctx.lineWidth = lw; ctx.beginPath();
        for (i = W; i < mod.nu; i++) { p = Math.round(x + mod.U[i] * s) + 0.5; ctx.moveTo(p, y); ctx.lineTo(p, y + s); }
        for (i = W; i < mod.nv; i++) { p = Math.round(y + mod.V[i] * s) + 0.5; ctx.moveTo(x, p); ctx.lineTo(x + s, p); }
        ctx.stroke();
      }
      function greville(k, n) { var g = new Float64Array(n); for (var i = 0; i < n; i++) g[i] = (k[i + 1] + k[i + 2] + k[i + 3]) / 3; return g; }
      function drawDots(mod, x, y, s, r, alpha) {
        var gu = greville(mod.U, mod.nu), gv = greville(mod.V, mod.nv), i, j;
        ctx.fillStyle = rgba('ctrl', alpha); ctx.beginPath();
        for (j = 0; j < mod.nv; j++) for (i = 0; i < mod.nu; i++) {
          var px = x + gu[i] * s, py = y + gv[j] * s;
          ctx.moveTo(px + r, py); ctx.arc(px, py, r, 0, 6.2832);
        }
        ctx.fill();
      }
      function frame(x, y, s) { ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(Math.round(x) - 0.5, Math.round(y) - 0.5, s + 1, s + 1); }
      function tag(s, x, y, color) {
        ctx.font = font(600, L.fs); if ('letterSpacing' in ctx) ctx.letterSpacing = '0.1em';
        var w = ctx.measureText(s).width + 14, h = L.fs + 10;
        ctx.fillStyle = 'rgba(7,9,15,.72)'; ctx.fillRect(x, y, w, h);
        txt(s, x + 7, y + h - 6, font(600, L.fs), color, 'left', '0.1em');
      }

      /* Layer B overlay: one Gaussian per (u,v) sample, drawn as the top-down footprint of a disc lying
         in the tangent plane — foreshortened along the gradient. Cached; redrawn only when it changes. */
      var gT = { s: 0, m: null, c: document.createElement('canvas'), key: '' };
      function paintGauss(g, s) {
        var n = st.dens;
        if (gT.s !== n || gT.m !== m) { gT.s = n; gT.m = m; gT.xs = centres(n); gT.TU = table(m.U, m.nu, gT.xs); gT.TV = table(m.V, m.nv, gT.xs); }
        var TU = gT.TU, TV = gT.TV, Z = m.Z, nu = m.nu, a, b, r, q, o, zu, zv, ru, rd, sp = s / n, r0 = sp * 0.42;
        g.beginPath();
        for (b = 0; b < n; b++) {
          var kv = TV.span[b] - P;
          for (a = 0; a < n; a++) {
            var ku = TU.span[a] - P; zu = 0; zv = 0;
            for (q = 0; q < W; q++) {
              o = (kv + q) * nu + ku; ru = 0; rd = 0;
              for (r = 0; r < W; r++) { ru += TU.N[a * W + r] * Z[o + r]; rd += TU.D[a * W + r] * Z[o + r]; }
              zu += TV.N[b * W + q] * rd; zv += TV.D[b * W + q] * ru;
            }
            zu *= ZG; zv *= ZG;
            var c = 1 / Math.sqrt(1 + zu * zu + zv * zv), ang = Math.atan2(zv, zu);
            var cx = gT.xs[a] * s, cy = gT.xs[b] * s, rx = Math.max(0.5, r0 * c);
            g.moveTo(cx + rx * Math.cos(ang), cy + rx * Math.sin(ang));
            g.ellipse(cx, cy, rx, r0, ang, 0, 6.2832);
          }
        }
        g.fillStyle = rgba('splat', sp > 6 ? 0.13 : 0.28); g.fill();
        g.strokeStyle = rgba('splat', sp > 11 ? 0.85 : 0.62); g.lineWidth = sp > 9 ? 1.1 : sp > 5 ? 0.8 : 0.5; g.stroke();
      }
      function drawGauss(x, y, s) {
        var dpr = cv.size.dpr, px = Math.round(s * dpr), key = st.dens + '|' + px + '|' + zVersion;
        if (gT.key !== key || gT.m !== m) {
          if (gT.c.width !== px) gT.c.width = gT.c.height = px;
          var g = gT.c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, s, s);
          paintGauss(g, s); gT.key = key;
        }
        ctx.drawImage(gT.c, x, y, s, s);
      }

      function barVal(sp) { return Math.sqrt(sp.mse * sp.n / tg.G); }      // the span's share, in RMSE units
      function drawBars(dt, t) {
        var x = L.mx, y = L.my, s = L.ms, bar = L.bar, gap = L.gap, i, sp, v, mx = 1e-9, worst = worstSpan(ms, CFG.MIN_SAMPLES), op = st.op;
        for (i = 0; i < ms.su.length; i++) mx = Math.max(mx, barVal(ms.su[i]));
        for (i = 0; i < ms.sv.length; i++) mx = Math.max(mx, barVal(ms.sv[i]));
        if (mx > st.barScale || !st.barScale) st.barScale = mx;
        else if (!st.holdScale && !(op && op.type === 'ins')) st.barScale += (Math.max(mx, 1e-6) - st.barScale) * (1 - Math.exp(-dt / 0.9));
        var sc = (bar - 3) / st.barScale, hiW = op && op.type === 'ins' && op.phase === 0 ? op.w : null;
        var pulse = 0.5 + 0.5 * Math.sin(t * 9);
        // baselines
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath();
        ctx.moveTo(x, y - gap + 0.5); ctx.lineTo(x + s, y - gap + 0.5);
        ctx.moveTo(x + s + gap - 0.5, y); ctx.lineTo(x + s + gap - 0.5, y + s); ctx.stroke();
        for (i = 0; i < ms.su.length; i++) {
          sp = ms.su[i]; v = Math.max(1.5, barVal(sp) * sc);
          var isW = worst && worst.axis === 'u' && worst.idx === i, x0 = x + sp.a * s, x1 = x + sp.b * s, inset = x1 - x0 > 5 ? 1 : 0.25;
          ctx.fillStyle = rgba('err', isW ? 0.95 : 0.4);
          ctx.fillRect(x0 + inset, y - gap - v, x1 - x0 - 2 * inset, v);
          if (hiW && hiW.axis === 'u' && hiW.idx === i) { ctx.strokeStyle = rgba('ctrl', 0.5 + 0.5 * pulse); ctx.lineWidth = 2; ctx.strokeRect(x0, y - gap - v - 2, x1 - x0, v + 2); }
        }
        for (i = 0; i < ms.sv.length; i++) {
          sp = ms.sv[i]; v = Math.max(1.5, barVal(sp) * sc);
          var isV = worst && worst.axis === 'v' && worst.idx === i, y0 = y + sp.a * s, y1 = y + sp.b * s, ins = y1 - y0 > 5 ? 1 : 0.25;
          ctx.fillStyle = rgba('err', isV ? 0.95 : 0.4);
          ctx.fillRect(x + s + gap, y0 + ins, v, y1 - y0 - 2 * ins);
          if (hiW && hiW.axis === 'v' && hiW.idx === i) { ctx.strokeStyle = rgba('ctrl', 0.5 + 0.5 * pulse); ctx.lineWidth = 2; ctx.strokeRect(x + s + gap, y0, v + 2, y1 - y0); }
        }
        // labels
        var lf = font(600, L.fs);
        txt(tiny ? 'ERROR IN u' : 'ERROR MEASURED IN u', x, y - gap - bar - 7, lf, C.dim, 'left', '0.12em');
        ctx.save(); ctx.translate(x + s + gap + bar + L.fs + 1, y); ctx.rotate(Math.PI / 2);
        txt(tiny ? 'ERROR IN v' : 'ERROR MEASURED IN v', 0, 0, lf, C.dim, 'left', '0.12em'); ctx.restore();
        if (!tiny) txt('full bar ' + pct(st.barScale, st.barScale < 0.01 ? 2 : 1), x + s, y - gap - bar - 7, font(500, L.fs), C.mute, 'right', '0.04em');
      }

      function drawStatus(x, y, maxW, align) {
        var l2 = status.live ? status.live() : status.l2;
        if (status.l1) fitTxt(status.l1, x, y + L.f1, 600, L.f1, toneCol(status.tone), align, '0.07em', maxW);
        if (l2 && !L.squat) flowTxt(l2, x, y + L.f1 + L.f2 + (tiny ? 5 : 9), L.f2, C.dim, maxW, L.f2 + 5);
      }

      function drawMain(t, dt) {
        var x = L.mx, y = L.my, s = L.ms, op = st.op, i, f, k, p;
        if (s < 40) return;
        if (!disp.tabA || disp.tabA.m !== m) { disp.tabA = tablesFor(m); dirty = true; }
        if (dirty) { paintMap(disp.cA, m, disp.tabA, disp.fitA, st.view, ESCALE); dirty = false; }
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(disp.cA.c, x, y, s, s);
        frame(x, y, s);

        // the span about to be split
        if (op && op.type === 'ins' && op.phase === 0) {
          var w = op.w, pu = 0.5 + 0.5 * Math.sin(t * 9);
          ctx.fillStyle = rgba('err', 0.10 + 0.08 * pu);
          if (w.axis === 'u') ctx.fillRect(x + w.a * s, y, (w.b - w.a) * s, s); else ctx.fillRect(x, y + w.a * s, s, (w.b - w.a) * s);
          ctx.strokeStyle = rgba('ctrl', 0.55 + 0.45 * pu); ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]); ctx.beginPath();
          if (w.axis === 'u') { p = x + w.knot * s; ctx.moveTo(p, y); ctx.lineTo(p, y + s); } else { p = y + w.knot * s; ctx.moveTo(x, p); ctx.lineTo(x + s, p); }
          ctx.stroke(); ctx.setLineDash([]);
        }
        var soft = st.gauss ? 0.3 : 1;
        drawKnots(m, x, y, s, 0.72 * soft, big ? 1.4 : 1);
        if (!st.dotsOff && !st.gauss && st.view !== 'error') drawDots(m, x, y, s, tiny ? 1.2 : big ? 2.3 : 1.7, 0.85);
        // knot about to be removed
        if (op && op.type === 'rem') {
          k = op.cand; var bl = 0.5 + 0.5 * Math.sin(t * 12), hw = tiny ? 4 : 7;
          ctx.fillStyle = 'rgba(255,255,255,' + (0.10 + 0.10 * bl) + ')';
          if (k.axis === 'u') ctx.fillRect(x + k.knot * s - hw, y, 2 * hw, s); else ctx.fillRect(x, y + k.knot * s - hw, s, 2 * hw);
          ctx.strokeStyle = 'rgba(255,255,255,' + (0.6 + 0.4 * bl) + ')'; ctx.lineWidth = 2.5; ctx.setLineDash([7, 5]); ctx.beginPath();
          if (k.axis === 'u') { p = x + k.knot * s; ctx.moveTo(p, y); ctx.lineTo(p, y + s); } else { p = y + k.knot * s; ctx.moveTo(x, p); ctx.lineTo(x + s, p); }
          ctx.stroke(); ctx.setLineDash([]);
        }
        // flashes: new knots glow, removed knots fade out dashed
        for (i = 0; i < fx.length; i++) {
          f = fx[i]; k = 1 - f.age / f.life; ctx.save();
          if (f.kind === 'new') { ctx.strokeStyle = rgba('ctrl', 0.35 + 0.65 * k); ctx.lineWidth = 1 + 2.6 * k; ctx.shadowColor = C.ctrl; ctx.shadowBlur = 16 * k; }
          else { ctx.strokeStyle = rgba('ctrl', 0.8 * k); ctx.lineWidth = 1.5; ctx.setLineDash([3, 6]); }
          ctx.beginPath();
          if (f.axis === 'u') { p = x + f.knot * s; ctx.moveTo(p, y); ctx.lineTo(p, y + s); } else { p = y + f.knot * s; ctx.moveTo(x, p); ctx.lineTo(x + s, p); }
          ctx.stroke(); ctx.restore();
        }
        if (st.gauss) drawGauss(x, y, s);

        tag(st.view === 'fit' ? 'FIT · z(u,v)' : st.view === 'target' ? 'TARGET · h(x,y)' : 'ERROR · |z − h|', x + 8, y + 8,
            st.view === 'fit' ? C.surf : st.view === 'target' ? C.text : C.err);
        if (st.view === 'error' && !tiny) legend(x + 8, y + s - 8 - (L.fs + 10), ESCALE);
        drawBars(dt, t);

        if (st.scene === 'sampling' || (st.gauss && !status.l1)) {
          status.tone = 'splat';
          status.l1 = 'LAYER B · ' + thousands(st.dens * st.dens) + ' GAUSSIANS SAMPLED ON (u,v)';
          status.l2 = 'surface untouched · still ' + m.nu * m.nv + ' parameters · RMSE ' + pct(ms.rmse); status.live = null;
        }
        drawStatus(x, y + s + (tiny ? 6 : 14), L.W - x - L.pad, 'left');
      }
      function legend(x, y, scale) {
        var w = 92, h = L.fs + 10, i;
        ctx.fillStyle = 'rgba(7,9,15,.72)'; ctx.fillRect(x, y, w + 86, h);
        for (i = 0; i < w; i++) {
          var li = ((Math.pow(i / (w - 1), EGAM) * 255) | 0) * 4;
          ctx.fillStyle = 'rgba(' + (PAL_ERR[li] | 0) + ',' + (PAL_ERR[li + 1] | 0) + ',' + (PAL_ERR[li + 2] | 0) + ',' + PAL_ERR[li + 3] + ')';
          ctx.fillRect(x + 7 + i, y + 6, 1, h - 12);
        }
        txt('0 – ' + pct(scale, 0), x + w + 15, y + h - 6, font(500, L.fs), C.dim, 'left', '0.04em');
      }

      function drawVersus(t) {
        refreshBase();
        var sp = L.sp, xa = L.x0, xb = L.x0 + sp + L.gap, y = L.y0, fs = L.fs;
        if (sp < 40) return;
        if (!disp.tabA || disp.tabA.m !== m) { disp.tabA = tablesFor(m); dirty = true; }
        if (!disp.tabB || disp.tabB.m !== base) { disp.tabB = tablesFor(base); dirtyB = true; }
        var esc = Math.max(1e-6, 0.8 * Math.max(ms.max, baseMs.max));
        if (dirty || esc !== disp.esc) { paintMap(disp.cA, m, disp.tabA, disp.fitA, st.view, esc); dirty = false; dirtyB = dirtyB || esc !== disp.esc; }
        if (dirtyB) { paintMap(disp.cB, base, disp.tabB, disp.fitB, st.view, esc); dirtyB = false; }
        disp.esc = esc;
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(disp.cA.c, xa, y, sp, sp); ctx.drawImage(disp.cB.c, xb, y, sp, sp);
        frame(xa, y, sp); frame(xb, y, sp);
        drawKnots(m, xa, y, sp, 0.6, 1); drawKnots(base, xb, y, sp, 0.6, 1);
        if (st.gauss) drawGauss(xa, y, sp);
        fitTxt((tiny ? 'ADAPTIVE · ' : 'ADAPTIVE KNOTS · ') + netStr(m), xa, y - 9, 600, fs, C.ctrl, 'left', '0.12em', sp);
        fitTxt((tiny ? 'UNIFORM · ' : 'UNIFORM KNOTS · ') + netStr(base), xb, y - 9, 600, fs, C.dim, 'left', '0.12em', sp);
        if (st.view === 'error' && !tiny) legend(xa + 8, y + sp - 8 - (fs + 10), esc);
        // the two numbers
        var better = ms.rmse < baseMs.rmse, fb = tiny ? 24 : KT.clamp(sp * 0.105, 28, 56), yb = y + sp + (tiny ? 8 : 16) + fb;
        var nd = ms.rmse < 0.0995 && baseMs.rmse < 0.0995 ? 2 : 1;
        txt(pct(ms.rmse, nd), xa, yb, font(600, fb, F_DISP), better ? C.ok : C.text, 'left', '-0.01em');
        txt(pct(baseMs.rmse, nd), xb, yb, font(600, fb, F_DISP), better ? C.err : C.text, 'left', '-0.01em');
        var yl = yb + fs + (tiny ? 5 : 9), pts = thousands(m.nu * m.nv);
        fitTxt('RMSE · ' + pts + ' CONTROL POINTS', xa, yl, 500, fs, C.dim, 'left', '0.1em', sp);
        fitTxt('RMSE · ' + pts + ' CONTROL POINTS', xb, yl, 500, fs, C.dim, 'left', '0.1em', sp);
        var ratio = baseMs.rmse / ms.rmse, same = Math.abs(ratio - 1) < 0.005;
        var head = same ? 'SAME ERROR · THE TWO NETS COINCIDE'
                 : better ? '×' + ratio.toFixed(1) + ' LOWER ERROR AT EQUAL PARAMETERS' : 'UNIFORM IS ×' + (1 / ratio).toFixed(1) + ' BETTER HERE';
        var ys = yl + (tiny ? 8 : 16), tw = 2 * sp + L.gap;
        fitTxt(head, xa, ys + L.f1, 600, L.f1, same ? C.text : better ? C.ok : C.err, 'left', '0.07em', tw);
        if (bestU) {
          var rb = bestU.rmse / ms.rmse;
          flowTxt((tiny ? 'best uniform near ' + pts + ' pts: ' : 'strongest uniform net near ' + pts + ' points: ') +
                  bestU.nu + ' × ' + bestU.nv + ' = ' + bestU.n + ' → ' + pct(bestU.rmse, nd) +
                  (same ? '' : rb >= 1 ? ' · still ×' + rb.toFixed(1) + ' behind' : ' · ×' + (1 / rb).toFixed(1) + ' ahead of adaptive'),
                  xa, ys + L.f1 + L.f2 + (tiny ? 5 : 9), L.f2, C.dim, tw, L.f2 + 5);
        }
      }

      var ESCALE = 0.30;                       // error heat: full red at 30 % of the relief
      var rdCache = {};
      function setRd(rd, key, s) { if (rdCache[key] !== s) { rdCache[key] = s; rd.set(s); } }
      function updateReadouts() {
        setRd(rdNet, 'net', netStr(m));
        setRd(rdPar, 'par', thousands(m.nu * m.nv));
        setRd(rdKn, 'kn', (m.nu - W) + ' u · ' + (m.nv - W) + ' v');
        setRd(rdRmse, 'rmse', pct(ms.rmse, ms.rmse < 0.02 ? 3 : 2));
        setRd(rdMax, 'max', pct(ms.max, ms.max < 0.1 ? 2 : 1));
        setRd(rdG, 'g', st.gauss ? thousands(st.dens * st.dens) : '—');
        if (st.versus && baseMs) setRd(rdUni, 'uni', pct(baseMs.rmse, baseMs.rmse < 0.02 ? 3 : 2));
      }

      env.loop(function (t, dt) {
        if (!m || !L) return;
        if (st.run) step(dt);
        ctx.clearRect(0, 0, L.W, L.H);
        if (st.versus) drawVersus(t); else drawMain(t, dt);
        updateReadouts();
      });

      var r0 = rootEl.getBoundingClientRect();
      enterCoarse(true); st.scene = 'coarse';
      if (r0.width > 0 && r0.height > 0) layout(r0.width, r0.height);

      return {
        setScene: setScene,
        resize: function (w, h) { layout(w, h); },
        destroy: function () { if (wrap.parentNode) wrap.parentNode.removeChild(wrap); },
        /* for tests / the deck: a read-only peek at the live numbers */
        peek: function () {
          return { scene: st.scene, nu: m.nu, nv: m.nv, rmse: ms.rmse, max: ms.max, done: m.cg.done, mode: st.mode,
                   busy: !!st.op, pending: !!st.pending,
                   inserted: st.inserted, pruned: st.pruned, gauss: st.gauss ? st.dens * st.dens : 0,
                   uniform: baseMs ? baseMs.rmse : null, best: bestU, status: [status.l1, status.live ? status.live() : status.l2],
                   layout: wrap.className, map: L && (L.ms || L.sp), U: m.U.slice(), V: m.V.slice() };
        }
      };
    }
  });
})(typeof window !== 'undefined' ? window : globalThis);
