/* Knot-a-Surface talk — module `refine`: "Two-layer density".
   A real least-squares fit of a cubic tensor-product B-spline height field to a procedural terrain.
     Layer A — control density: Boehm knot insertion where the per-span error is worst (exact: the
               surface is unchanged at the instant of insertion), and remove + re-fit where a knot is
               redundant.
     Layer B — sampling density: Gaussians sampled on the (u,v) domain; never touches the surface.
               Each one is derived: centre S(u,v), tangent axes S_u, S_v times the sampling interval,
               normal S_u x S_v. What they DRAW is rendered (a normalised blend of their tangent
               planes, seen from above) and measured against the target, so the two densities can be
               weighed against each other: the balance map.
   Two views of one state: the map from above, and a side elevation cut along u.
   Tool: Canvas 2D. Standalone: talk/sims/refine.html. Depends on talk/nurbs.js + talk/kit.js only.
   The maths below (PART 1) is pure and is also exported under node for tests. */
(function (root) {
  'use strict';
  var NODE = typeof module !== 'undefined' && module.exports;
  var NB = NODE ? require('../nurbs.js') : root.KnotTalk.nurbs;
  var targetRaw = NODE ? require('../terrain.js') : root.KnotTalk.terrain;   // talk/terrain.js (chapter 03 fits it too)
  if (typeof targetRaw !== 'function') throw new Error('refine: load talk/terrain.js before talk/sims/refine.js');
  var P = 3, W = 4;                       // cubic everywhere

  /* =====================================================================================
     PART 1 — pure maths
     Layout conventions: a grid field F[b * n + a] has a along u (x, to the right) and b along v
     (y, downwards). The control net is Z[j * nu + i], i along u, j along v.
     ===================================================================================== */

  /* ---------- the target: smooth rolling ground + one crater field ----------
     targetRaw is talk/terrain.js (KnotTalk.terrain): one definition, shared with chapter 03's patch. */
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

  /* Cholesky for the per-axis Gram matrices; solves in place for many right-hand sides. A cubic
     B-spline Gram matrix has half-bandwidth P and so has its factor: every loop stays inside the
     band, which keeps a factor + solve linear in n (dense storage, banded work). */
  function cholesky(A, n) {
    var L = new Float64Array(A), i, j, k, s, t, k0, i1;
    for (j = 0; j < n; j++) {
      k0 = j - P < 0 ? 0 : j - P;
      s = L[j * n + j]; for (k = k0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
      s = Math.sqrt(s > 0 ? s : 1e-300); L[j * n + j] = s;
      i1 = j + P >= n ? n - 1 : j + P;
      for (i = j + 1; i <= i1; i++) {
        t = L[i * n + j]; for (k = i - P < 0 ? 0 : i - P; k < j; k++) t -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = t / s;
      }
    }
    return L;
  }
  function cholSolve(L, n, x, off, stride) {       // x[off + i*stride], i = 0..n-1
    var i, k, s, k1;
    for (i = 0; i < n; i++) { s = x[off + i * stride]; for (k = i - P < 0 ? 0 : i - P; k < i; k++) s -= L[i * n + k] * x[off + k * stride]; x[off + i * stride] = s / L[i * n + i]; }
    for (i = n - 1; i >= 0; i--) { s = x[off + i * stride]; k1 = i + P >= n ? n - 1 : i + P; for (k = i + 1; k <= k1; k++) s -= L[k * n + i] * x[off + k * stride]; x[off + i * stride] = s / L[i * n + i]; }
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
  /* Score every interior knot by the RMSE of the re-fitted surface without it; cheapest first.
     Dropping a knot changes ONE axis, so the target is contracted with the other axis's basis once
     (`half`), and each candidate only needs its own axis table, a banded factor and one pass over
     `half`. At a least-squares optimum the squared error is ||T||^2 - rhs.Z, so no grid pass either.
     (The first version rebuilt the whole model per candidate: 150 ms a step at 37 x 37, 480 ms at
     64 x 64; this one is a few ms.) */
  function removalCandidates(m) {
    var tg = m.tg, G = tg.G, T = tg.T, out = [];
    if (tg.TT == null) tg.TT = dot(T, T);
    function run(axis) {
      var isU = axis === 'u', knots = isU ? m.U : m.V, n = isU ? m.nu : m.nv, no = isU ? m.nv : m.nu;
      if (n <= W) return;
      var To = isU ? m.TV : m.TU, Lo = cholesky(isU ? m.Gv : m.Gu, no);
      var half = new Float64Array(G * no), a, b, s, r, k, t, line, idx, i, j, w;
      for (b = 0; b < G; b++) for (a = 0; a < G; a++) {
        t = T[b * G + a];
        if (isU) { k = To.span[b] - P; for (s = 0; s < W; s++) half[a * no + k + s] += To.N[b * W + s] * t; }
        else { k = To.span[a] - P; for (s = 0; s < W; s++) half[b * no + k + s] += To.N[a * W + s] * t; }
      }
      for (idx = W; idx < n; idx++) {
        var k2 = knots.slice(); k2.splice(idx, 1);
        var n2 = n - 1, Tn = table(k2, n2, tg.xs), L = cholesky(gram(Tn, n2), n2);
        var R = new Float64Array(n2 * no);               // R[j * n2 + i]: j on the other axis, i on this one
        for (line = 0; line < G; line++) {
          k = Tn.span[line] - P;
          for (r = 0; r < W; r++) { w = Tn.N[line * W + r]; for (j = 0; j < no; j++) R[j * n2 + k + r] += w * half[line * no + j]; }
        }
        var Zc = new Float64Array(R);
        for (j = 0; j < no; j++) cholSolve(L, n2, Zc, j * n2, 1);
        for (i = 0; i < n2; i++) cholSolve(Lo, no, Zc, i, n2);
        out.push({ axis: axis, idx: idx, knot: knots[idx], rmse: Math.sqrt(Math.max(0, tg.TT - dot(R, Zc)) / (G * G)), removed: no });
      }
    }
    run('u'); run('v');
    out.sort(function (a, b) { return a.rmse - b.rmse; });
    return out;
  }

  /* ---------- the canonical pipeline (deterministic; the animation walks exactly these steps) ---------- */
  var CFG = {
    G: 128,              // target samples per axis
    N0: 5,               // coarse net: N0 x N0 control heights
    BUDGET: 196,         // control-point budget of the scripted auto-insertion (the deck's numbers)
    MAX_PTS: 4096,       // hard cap on control points (budget slider, insert button, balance map)
    MIN_SAMPLES: 4,      // a span must hold this many sample lines to be split
    TOL: 1.05,           // pruning keeps RMSE <= TOL x (RMSE before pruning)
    SCORE: 'sum',        // span score: 'sum' of squared residuals in the span, or their 'mean'
    GMIN: 6, GMAX: 256,  // Gaussians per axis: 36 ... 65 536
    SIG: 0.5, RCUT: 1.5, // a Gaussian's footprint: sigma and cut-off, in sample spacings
    ZW: 0.35             // world height of the relief (the map is 1 x 1): sets every normal
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

  /* =====================================================================================
     Layer B — the derived Gaussians, what they draw, and how far that is from the target
     ===================================================================================== */
  /* n x n Gaussians at the cell centres of (u,v). Each is derived from the surface, nothing is free:
     height z, slopes z_u, z_v (so its tangent axes are S_u = (1,0,ZW z_u) and S_v = (0,1,ZW z_v),
     scaled by the sampling interval) and unit normal S_u x S_v = (-ZW z_u, -ZW z_v, 1) / |.|. */
  function sampleGauss(m, n, gs) {
    var N2 = n * n, nu = m.nu, Z = m.Z, a, b, i, q, r, s, d, e, o, ku, kv, zz, len, ZW = CFG.ZW;
    if (!gs || gs.n !== n) {
      gs = { n: n, xs: centres(n), z: new Float32Array(N2), zu: new Float32Array(N2), zv: new Float32Array(N2),
             nx: new Float32Array(N2), ny: new Float32Array(N2), nz: new Float32Array(N2), U: null, V: null };
    }
    if (gs.U !== m.U || gs.V !== m.V) { gs.TU = table(m.U, m.nu, gs.xs); gs.TV = table(m.V, m.nv, gs.xs); gs.U = m.U; gs.V = m.V; }
    var TU = gs.TU, TV = gs.TV, tN = new Float64Array(nu), tD = new Float64Array(nu);
    for (b = 0; b < n; b++) {
      kv = TV.span[b] - P;
      for (i = 0; i < nu; i++) {                           // contract along v once per control column
        s = 0; d = 0;
        for (q = 0; q < W; q++) { zz = Z[(kv + q) * nu + i]; s += TV.N[b * W + q] * zz; d += TV.D[b * W + q] * zz; }
        tN[i] = s; tD[i] = d;
      }
      for (a = 0; a < n; a++) {
        ku = TU.span[a] - P; s = 0; d = 0; e = 0;
        for (r = 0; r < W; r++) { s += TU.N[a * W + r] * tN[ku + r]; d += TU.D[a * W + r] * tN[ku + r]; e += TU.N[a * W + r] * tD[ku + r]; }
        o = b * n + a; gs.z[o] = s; gs.zu[o] = d; gs.zv[o] = e;
        len = 1 / Math.sqrt(1 + ZW * ZW * (d * d + e * e));
        gs.nx[o] = -ZW * d * len; gs.ny[o] = -ZW * e * len; gs.nz[o] = len;
      }
    }
    return gs;
  }
  /* Footprint weights of an n-lattice at the positions xs (one axis; the footprint is separable).
     A Gaussian of sigma = SIG sample spacings, lowered so it reaches zero at RCUT spacings, then
     normalised over the Gaussians that reach the position: w sums to 1 everywhere. */
  function blendTable(n, xs) {
    var SIG = CFG.SIG, R = CFG.RCUT, K = Math.ceil(2 * R) + 1, E = xs.length, c = Math.exp(-R * R / (2 * SIG * SIG));
    var t = { n: n, E: E, K: K, i0: new Int32Array(E), cnt: new Int32Array(E), w: new Float32Array(E * K), d: new Float32Array(E * K) };
    for (var i = 0; i < E; i++) {
      var p = xs[i] * n - 0.5, a0 = Math.ceil(p - R), a1 = Math.floor(p + R), s = 0, k = 0, a, q, ww;
      if (a0 < 0) a0 = 0; if (a1 > n - 1) a1 = n - 1;
      for (a = a0; a <= a1; a++, k++) { q = p - a; ww = Math.exp(-q * q / (2 * SIG * SIG)) - c; if (ww < 0) ww = 0; t.w[i * K + k] = ww; t.d[i * K + k] = q / n; s += ww; }
      if (!(s > 0)) { a0 = p < 0 ? 0 : p > n - 1 ? n - 1 : Math.round(p); k = 1; t.w[i * K] = 1; t.d[i * K] = (p - a0) / n; s = 1; }
      for (q = 0; q < k; q++) t.w[i * K + q] /= s;
      t.i0[i] = a0; t.cnt[i] = k;
    }
    return t;
  }
  /* What the Gaussians draw, seen from above: at every position the normalised blend of the tangent
     planes of the Gaussians that reach it,
         D(x,y) = sum_ab w_a(x) w_b(y) [ z_ab + z_u,ab (x - u_a) + z_v,ab (y - v_b) ]
     (and, if outN is given, the same blend of their unit normals, re-normalised).
     tx, ty = blendTable of the lattice at the x and y positions. Two separable passes. */
  function renderGauss(gs, tx, ty, outH, outN) {
    var n = gs.n, Ex = tx.E, Ey = ty.E, K = tx.K, st = outN ? 5 : 2, i, j, q, r, b, o, w, a0, c, p;
    var b0 = n, b1 = -1;
    for (j = 0; j < Ey; j++) { if (ty.i0[j] < b0) b0 = ty.i0[j]; if (ty.i0[j] + ty.cnt[j] - 1 > b1) b1 = ty.i0[j] + ty.cnt[j] - 1; }
    var nb = b1 - b0 + 1, need = nb * Ex * st;
    if (!gs.tmp || gs.tmp.length < need) gs.tmp = new Float32Array(need);
    var tmp = gs.tmp, z = gs.z, zu = gs.zu, zv = gs.zv, nx = gs.nx, ny = gs.ny, nz = gs.nz, A, B, X, Y, Zn;
    for (b = b0; b <= b1; b++) for (i = 0; i < Ex; i++) {          // pass 1: along u, per lattice row
      a0 = b * n + tx.i0[i]; c = tx.cnt[i]; A = 0; B = 0; X = 0; Y = 0; Zn = 0;
      for (r = 0; r < c; r++) {
        w = tx.w[i * K + r]; o = a0 + r;
        A += w * (z[o] + zu[o] * tx.d[i * K + r]); B += w * zv[o];
        if (outN) { X += w * nx[o]; Y += w * ny[o]; Zn += w * nz[o]; }
      }
      p = ((b - b0) * Ex + i) * st; tmp[p] = A; tmp[p + 1] = B;
      if (outN) { tmp[p + 2] = X; tmp[p + 3] = Y; tmp[p + 4] = Zn; }
    }
    for (j = 0; j < Ey; j++) for (i = 0; i < Ex; i++) {            // pass 2: along v
      c = ty.cnt[j]; A = 0; X = 0; Y = 0; Zn = 0;
      for (q = 0; q < c; q++) {
        w = ty.w[j * K + q]; p = ((ty.i0[j] + q - b0) * Ex + i) * st;
        A += w * (tmp[p] + tmp[p + 1] * ty.d[j * K + q]);
        if (outN) { X += w * tmp[p + 2]; Y += w * tmp[p + 3]; Zn += w * tmp[p + 4]; }
      }
      outH[j * Ex + i] = A;
      if (outN) { w = 1 / Math.sqrt(X * X + Y * Y + Zn * Zn || 1); o = (j * Ex + i) * 3; outN[o] = X * w; outN[o + 1] = Y * w; outN[o + 2] = Zn * w; }
    }
    return outH;
  }
  /* The drawn error: render n x n Gaussians of the model on the target's own grid and compare.
       rmse = RMS (drawn - target)     what the picture is off by
       samp = RMS (drawn - surface)    the part the SAMPLING adds (zero with infinitely many Gaussians)
     `m.fit` must hold the surface on the target grid (measure(m) leaves it there). ws = workspace. */
  function drawnError(m, n, ws, gs) {
    var tg = m.tg, G = tg.G, T = tg.T, fit = m.fit, i, e, f, se = 0, sf = 0, mx = 0;
    ws = ws || {};
    gs = gs || (ws.gs = sampleGauss(m, n, ws.gs));
    if (!ws.bt || ws.bt.n !== n) ws.bt = blendTable(n, tg.xs);
    if (!ws.D) ws.D = new Float32Array(G * G);
    renderGauss(gs, ws.bt, ws.bt, ws.D, ws.N || null);
    for (i = 0; i < G * G; i++) { e = ws.D[i] - T[i]; f = ws.D[i] - fit[i]; se += e * e; sf += f * f; if (e * e > mx) mx = e * e; }
    return { rmse: Math.sqrt(se / (G * G)), samp: Math.sqrt(sf / (G * G)), max: Math.sqrt(mx) };
  }

  /* ---------- normals ---------- */
  /* Unit normals of the target on its grid (central differences of the analytic terrain). */
  function targetNormals(tg) {
    if (tg.tn) return tg.tn;
    var G = tg.G, tn = new Float32Array(G * G * 3), h = 1e-4, a, b, x, y, gx, gy, len, o, ZW = CFG.ZW;
    for (b = 0; b < G; b++) for (a = 0; a < G; a++) {
      x = tg.xs[a]; y = tg.xs[b];
      gx = (targetRaw(x + h, y) - targetRaw(x - h, y)) * tg.k / (2 * h);
      gy = (targetRaw(x, y + h) - targetRaw(x, y - h)) * tg.k / (2 * h);
      len = 1 / Math.sqrt(1 + ZW * ZW * (gx * gx + gy * gy)); o = (b * G + a) * 3;
      tn[o] = -ZW * gx * len; tn[o + 1] = -ZW * gy * len; tn[o + 2] = len;
    }
    return (tg.tn = tn);
  }
  /* Unit normals S_u x S_v of the fitted surface on the target grid (analytic derivatives). */
  function fitNormals(m, out) {
    var G = m.tg.G, nu = m.nu, Z = m.Z, TU = m.TU, TV = m.TV, a, b, r, s, o, ku, kv, zu, zv, ru, rd, len, ZW = CFG.ZW;
    out = out || new Float32Array(G * G * 3);
    for (b = 0; b < G; b++) {
      kv = TV.span[b] - P;
      for (a = 0; a < G; a++) {
        ku = TU.span[a] - P; zu = 0; zv = 0;
        for (s = 0; s < W; s++) {
          o = (kv + s) * nu + ku; ru = 0; rd = 0;
          for (r = 0; r < W; r++) { ru += TU.N[a * W + r] * Z[o + r]; rd += TU.D[a * W + r] * Z[o + r]; }
          zu += TV.N[b * W + s] * rd; zv += TV.D[b * W + s] * ru;
        }
        len = 1 / Math.sqrt(1 + ZW * ZW * (zu * zu + zv * zv)); o = (b * G + a) * 3;
        out[o] = -ZW * zu * len; out[o + 1] = -ZW * zv * len; out[o + 2] = len;
      }
    }
    return out;
  }
  /* Mean and max angle (degrees) between two fields of unit normals. */
  function angleStats(A, B) {
    var n = A.length / 3, i, d, s = 0, mx = 0, ang;
    for (i = 0; i < n; i++) {
      d = A[i * 3] * B[i * 3] + A[i * 3 + 1] * B[i * 3 + 1] + A[i * 3 + 2] * B[i * 3 + 2];
      ang = Math.acos(d > 1 ? 1 : d < -1 ? -1 : d); s += ang; if (ang > mx) mx = ang;
    }
    return { mean: s / n * 180 / Math.PI, max: mx * 180 / Math.PI };
  }

  var MATH = {
    CFG: CFG, targetRaw: targetRaw, makeTarget: makeTarget, targetAt: targetAt, centres: centres, table: table,
    evalGrid: evalGrid, makeModel: makeModel, uniformModel: uniformModel, cgInit: cgInit, cgStep: cgStep,
    cgConverge: cgConverge, solveDirect: solveDirect, measure: measure, rmseOf: rmseOf, insertKnot: insertKnot,
    insertInto: insertInto, worstSpan: worstSpan, removeFrom: removeFrom, removalCandidates: removalCandidates,
    coarseModel: coarseModel, nextInsertion: nextInsertion, converged: converged, runAuto: runAuto,
    nextRemoval: nextRemoval, runPrune: runPrune, uniformBaseline: uniformBaseline, bestUniform: bestUniform,
    sampleGauss: sampleGauss, blendTable: blendTable, renderGauss: renderGauss, drawnError: drawnError,
    targetNormals: targetNormals, fitNormals: fitNormals, angleStats: angleStats
  };
  if (NODE) { module.exports = MATH; return; }

  /* =====================================================================================
     PART 2 — the instrument (Canvas 2D)
     ===================================================================================== */
  var KT = root.KnotTalk, C = KT.color;
  function rgbOf(hex) { var n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function rgba(key, a) { var c = rgbOf(C[key]); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
  function pct(x, d) { return (100 * x).toFixed(d == null ? 2 : d) + ' %'; }
  function pctA(x) { return pct(x, x < 0.00095 ? 3 : x < 0.0995 ? 2 : 1); }       // as many digits as the size needs
  function expStr(x) { return x === 0 ? '0' : x.toExponential(1).replace('e-', 'e−'); }
  function thousands(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ' '); }
  function kfmt(n) { return n >= 10000 ? Math.floor(n / 1000) + 'k' : thousands(n); }
  function now() { return root.performance && root.performance.now ? root.performance.now() : Date.now(); }

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
  var PAL_GSS = ramp([[0, '#1a1230'], [0.30, '#35235a'], [0.58, '#74509b'], [0.80, '#ce93d8'], [1, '#f8e9fb']]);   // what the Gaussians draw
  var PAL_ERR = ramp([[0, '#2a0c0e', 0], [0.14, '#521819', 0.46], [0.42, '#b93b39', 0.92], [0.72, '#ef5350', 1], [1, '#ffd9d2', 1]]);
  var EGAM = 0.9;                         // display gamma of the error heat
  var LX = -0.50, LY = -0.62, LZ = 0.62, ZS = 0.55, GSAT = 1.15;
  (function () { var n = Math.sqrt(LX * LX + LY * LY + LZ * LZ); LX /= n; LY /= n; LZ /= n; })();
  var ZLO = -0.14, ZSPAN = 1.28;          // relief range of the side view (control heights overshoot a little)
  var MAP_LO = 1e-4, MAP_HI = 0.1;        // balance map colour range: 0.01 % ... 10 %, logarithmic
  var KSTEPS = [6, 16, 32, 64, 128, 256]; // rows of the balance map that get a tick label
  var ANG_SCALE = 45;                     // normal-error heat: full red at 45 degrees
  var REGIME_K = 2;                       // "balanced" = surface error and sampling error within this factor

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
  /* Unit normals of a height buffer (n x n, display resolution) by central differences. */
  function normalsOf(H, n, out) {
    var x, y, i, gx, gy, len, ZW = CFG.ZW;
    for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
      i = y * n + x;
      gx = (H[x < n - 1 ? i + 1 : i] - H[x > 0 ? i - 1 : i]) * (x > 0 && x < n - 1 ? n * 0.5 : n);
      gy = (H[y < n - 1 ? i + n : i] - H[y > 0 ? i - n : i]) * (y > 0 && y < n - 1 ? n * 0.5 : n);
      len = 1 / Math.sqrt(1 + ZW * ZW * (gx * gx + gy * gy));
      out[i * 3] = -ZW * gx * len; out[i * 3 + 1] = -ZW * gy * len; out[i * 3 + 2] = len;
    }
    return out;
  }
  /* The usual normal-map colouring, n * 0.5 + 0.5. The map's v axis runs down the page, so green is
     flipped: red = leaning right, green = leaning up the page, blue = facing the viewer. */
  function paintNormals(px, N3, n) {
    for (var i = 0, o = 0, q = 0; i < n * n; i++, o += 4, q += 3) {
      px[o] = 127.5 + 127.5 * N3[q]; px[o + 1] = 127.5 - 127.5 * N3[q + 1]; px[o + 2] = 127.5 + 127.5 * N3[q + 2]; px[o + 3] = 255;
    }
  }
  function normalCss(nx, ny, nz) { return 'rgb(' + Math.round(127.5 + 127.5 * nx) + ',' + Math.round(127.5 - 127.5 * ny) + ',' + Math.round(127.5 + 127.5 * nz) + ')'; }
  /* balance-map colour of a drawn error: half-decade bands, so equal errors read as one colour and the
     iso-error contours (L-shaped: neither knob helps alone) are the band edges */
  var MAP_BANDS = ['#0f2233', '#1c2c47', '#3a2643', '#6c2536', '#a3302f', '#de463f', '#ffb3a6'];
  function mapCss(e) {
    var k = Math.floor(2 * Math.log(Math.max(e, 1e-12) / MAP_LO) / Math.LN10 + 1e-9);
    return MAP_BANDS[k < 0 ? 0 : k > 6 ? 6 : k];
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
      '.kt-sim-refine .rf-btns.rf-tog,.kt-sim-refine .rf-btns.rf-pre{grid-template-columns:repeat(2,minmax(0,1fr))}',
      '.kt-sim-refine .rf-btns .kt-btn,.kt-sim-refine .rf-btns .kt-toggle{padding-left:2px;padding-right:2px;text-align:center;white-space:nowrap}',
      '.kt-sim-refine .kt-toggle.rf-g.on{color:var(--splat);border-color:rgba(var(--splat-rgb),.55);background:rgba(var(--splat-rgb),.1)}',
      '.kt-sim-refine .rf-pre .kt-btn.on{color:var(--text);border-color:var(--dim);background:rgba(255,255,255,.06)}',
      '.kt-sim-refine .rf-dens input{accent-color:var(--splat)}',
      '.kt-sim-refine .rf-dens input::-webkit-slider-thumb{background:var(--splat)}',
      '.kt-sim-refine .rf-dens input::-moz-range-thumb{background:var(--splat)}',
      '.kt-sim-refine .rf-bud input{accent-color:var(--ctrl)}',
      '.kt-sim-refine .rf-bud input::-webkit-slider-thumb{background:var(--ctrl)}',
      '.kt-sim-refine .rf-bud input::-moz-range-thumb{background:var(--ctrl)}',
      '.kt-sim-refine .rf-side .rf-ctrl .kt-slider .kt-lab{min-width:52px}',
      /* stacked: readouts become a grid of label-over-value cells, controls a two-column grid */
      '.kt-sim-refine .rf-stack .rf-read{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px 14px;align-content:center;flex:1 1 0}',
      '.kt-sim-refine .rf-stack .rf-read .kt-panel-title{display:none}',
      '.kt-sim-refine .rf-stack .rf-read .kt-readout{flex-direction:column;align-items:flex-start;gap:3px;min-height:0}',
      '.kt-sim-refine .rf-stack .rf-read .kt-val{margin-left:0}',
      '.kt-sim-refine .rf-stack.rf-gon .rf-x,.kt-sim-refine .rf-stack:not(.rf-gon) .rf-gx,.kt-sim-refine .rf-stack.rf-non .rf-x2{display:none}',
      '.kt-sim-refine .rf-stack:not(.rf-tiny) .rf-col .rf-ctrl{flex:1.5 1 0;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 16px;align-content:center}',
      '.kt-sim-refine .rf-stack:not(.rf-tiny) .rf-ctrl .kt-panel-title{grid-column:1/-1}',
      '.kt-sim-refine .rf-stack:not(.rf-tiny) .rf-ctrl .rf-pre{grid-row:span 2}',
      /* a short stage with the panel beside the figure: tighter rows so the whole panel stays on screen */
      '.kt-sim-refine .rf-dense .rf-col{gap:6px;padding-top:6px;padding-bottom:6px}',
      '.kt-sim-refine .rf-dense .rf-col .kt-panel{padding:7px 11px;gap:3px}',
      '.kt-sim-refine .rf-dense .kt-row{min-height:19px}',
      '.kt-sim-refine .rf-dense .rf-read .kt-val{font-size:13px}',
      '.kt-sim-refine .rf-dense .rf-read .rf-t2{margin-top:4px;padding-top:6px}',
      '.kt-sim-refine .rf-dense .kt-seg-btn,.kt-sim-refine .rf-dense .kt-btn,.kt-sim-refine .rf-dense .kt-toggle{padding-top:5px;padding-bottom:5px}',
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
      '.kt-sim-refine .rf-tiny:not(.rf-gon) .rf-gx,.kt-sim-refine .rf-tiny.rf-gon .rf-x2,.kt-sim-refine .rf-tiny.rf-non .rf-x3{display:none}',
      '.kt-sim-refine .rf-tiny .rf-ctrl{flex-direction:row;flex-wrap:wrap;align-items:center;gap:5px 6px}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-panel-title,.kt-sim-refine .rf-tiny .rf-ctrl .kt-seg .kt-lab{display:none}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-row{min-height:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-seg-box{margin-left:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .rf-btns{flex:1 1 150px;gap:4px}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-slider{flex:1 1 150px;min-width:0}',
      '.kt-sim-refine .rf-tiny .rf-ctrl .kt-slider input{width:0;min-width:50px}',
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
    S.coarseAngle = function () {         // mean angle between the coarse net's normals and the target's
      if (S._ca == null) S._ca = angleStats(fitNormals(S.restore(S.coarse())), targetNormals(tg)).mean;
      return S._ca;
    };

    /* The balance map: drawn error over (control points x Gaussians), MEASURED.
       Columns = NCOL nets taken from the auto-insertion sequence run up to CFG.MAX_PTS (log-spaced in
       control points); rows = Gaussian lattices, x2 Gaussians per row. Every cell renders that many
       Gaussians of that net and compares with the target (drawnError). Done in slices of a few ms
       per frame so the page never stalls; the map fills in as it is measured. */
    var NCOL = 12, B = S.bal = { rows: [6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256], cols: null, E: null, Sm: null,
                                 nstar: null, ready: false, phase: 0, done: 0, total: 1, ms: 0 };
    S.balStep = function (budgetMs) {
      if (B.ready) return true;
      var t0 = now(), nr = B.rows.length, i, k, w, r, d;
      if (B.phase === 0) {
        S.coarse(); B.m = S.restore(S.coarse()); B.mm = measure(B.m);
        B.snaps = [{ pts: B.m.nu * B.m.nv, nu: B.m.nu, nv: B.m.nv, rmse: B.mm.rmse, snap: S.coarse() }];
        B.phase = 1;
      }
      while (B.phase === 1 && now() - t0 < budgetMs) {                 // the auto-insertion sequence, to the cap
        w = nextInsertion(B.m, B.mm, CFG.MAX_PTS);
        if (!w) { B.phase = 2; break; }
        r = insertInto(B.m, w.axis, w.knot); B.m = converged(r.model); B.mm = measure(B.m);
        B.snaps.push({ pts: B.m.nu * B.m.nv, nu: B.m.nu, nv: B.m.nv, rmse: B.mm.rmse, snap: snapOf(B.m) });
        B.done = Math.log(B.m.nu * B.m.nv / 25) / Math.log(CFG.MAX_PTS / 25) * 0.5; B.total = 1;
      }
      if (B.phase === 2) {
        var last = B.snaps[B.snaps.length - 1].pts, cols = [], seen = {};
        for (k = 0; k < NCOL; k++) {
          var want = 25 * Math.pow(last / 25, k / (NCOL - 1)), best = 0;
          for (i = 1; i < B.snaps.length; i++) if (Math.abs(Math.log(B.snaps[i].pts / want)) < Math.abs(Math.log(B.snaps[best].pts / want))) best = i;
          if (!seen[best]) { seen[best] = 1; cols.push(B.snaps[best]); }
        }
        B.cols = cols; B.snaps = null; B.m = null;
        B.E = new Float32Array(cols.length * nr); B.Sm = new Float32Array(cols.length * nr);
        for (i = 0; i < B.E.length; i++) B.E[i] = B.Sm[i] = NaN;
        B.balCol = 0;
        for (i = 1; i < cols.length; i++) if (Math.abs(Math.log(cols[i].pts / 400)) < Math.abs(Math.log(cols[B.balCol].pts / 400))) B.balCol = i;
        B.ci = 0; B.ri = 0; B.cm = null; B.ws = {}; B.phase = 3;
      }
      while (B.phase === 3 && now() - t0 < budgetMs) {                 // one cell = one render + one comparison
        if (!B.cm) { B.cm = S.restore(B.cols[B.ci].snap); measure(B.cm); }
        d = drawnError(B.cm, B.rows[B.ri], B.ws);
        B.E[B.ci * nr + B.ri] = d.rmse; B.Sm[B.ci * nr + B.ri] = d.samp;
        B.done = 0.5 + 0.5 * (B.ci * nr + B.ri + 1) / (B.cols.length * nr);
        if (++B.ri >= nr) { B.ri = 0; B.cm = null; if (++B.ci >= B.cols.length) B.phase = 4; }
      }
      if (B.phase === 4) {
        /* the balance line: per net, the lattice at which the sampling error equals the surface error
           (log-interpolated between rows; below the first / above the last row when it never crosses) */
        B.nstar = [];
        for (i = 0; i < B.cols.length; i++) {
          var F = B.cols[i].rmse, ns = null;
          if (B.Sm[i * nr] <= F) ns = B.rows[0] * 0.6;
          else if (B.Sm[i * nr + nr - 1] > F) ns = B.rows[nr - 1] * 1.6;
          else for (k = 0; k < nr - 1; k++) {
            var a = B.Sm[i * nr + k], b = B.Sm[i * nr + k + 1];
            if (a > F && b <= F) { ns = B.rows[k] * Math.pow(B.rows[k + 1] / B.rows[k], Math.log(a / F) / Math.log(a / b)); break; }
          }
          B.nstar.push(ns);
        }
        B.ready = true; B.done = 1; B.phase = 5; B.ws = null;
      }
      B.ms += now() - t0;
      return B.ready;
    };
    return (SHARED = S);
  }

  var SCENES = [
    { id: 'coarse', label: 'Coarse net' }, { id: 'insert', label: 'Insert one knot' },
    { id: 'auto', label: 'Auto-insert' }, { id: 'remove', label: 'Remove + re-fit' },
    { id: 'versus', label: 'Adaptive vs uniform' }, { id: 'sampling', label: 'Layer B · sampling' },
    { id: 'balance', label: 'Balance the two' }, { id: 'normals', label: 'Normals' }
  ];
  var PACE = {
    sceneInsert: { delay: 1.0, hi: 1.5, hold: 2.0, rate: 9 },
    button: { delay: 0, hi: 0.55, hold: 0.9, rate: 36 },
    auto: { gap: 0.8, hi: 0.26, hold: 0.12, rate: 260 },
    prune: { gap: 1.15, hi: 0.5, rate: 200 },
    tour: 5.5                             // seconds on each regime of the `balance` scene
  };
  var BUDGETS = [49, 64, 100, 144, 196, 256, 400, 576, 784, 1024, 1600, 2500, 4096];
  var DSTEPS = 64;                        // positions of the (geometric) Gaussian slider
  function densOf(i) { return Math.round(CFG.GMIN * Math.pow(CFG.GMAX / CFG.GMIN, i / DSTEPS)); }
  function densIdx(n) { return Math.round(DSTEPS * Math.log(n / CFG.GMIN) / Math.log(CFG.GMAX / CFG.GMIN)); }
  var REGIMES = {
    ctrl: { n: 256, label: 'ctrl-starved', head: 'STARVED OF CONTROL POINTS' },
    gauss: { n: 8, label: 'gauss-starved', head: 'STARVED OF GAUSSIANS' },
    bal: { n: 128, label: 'balanced', head: 'BALANCED' }
  };

  KT.register('refine', {
    title: 'Two-layer density',
    blurb: 'A cubic B-spline height field fitted to a terrain, live, seen from above and from the side. Layer A adds ' +
           'control where the per-span error is worst and prunes knots that buy nothing. Layer B is how many Gaussians ' +
           'are sampled from the surface. The drawn error is measured, and it falls only when the two are balanced.',
    tool: 'Canvas 2D · live least-squares fit',
    scenes: SCENES,
    create: function (env) {
      injectCSS();
      var rootEl = env.root, ui = env.ui, S = shared(), tg = S.tg, B = S.bal;
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
        var r = ui.readout(pr, { label: full, tone: tone }); if (cls) r.el.className += ' ' + cls;
        LABELS.push({ el: r.el.querySelector('.kt-lab'), full: full, short: short });
        return r;
      }
      var rdNet = readout('control points', 'net', 'ctrl');
      var rdPar = readout('parameters', 'params', 'ctrl', 'rf-x3');
      var rdKn = readout('interior knots', 'knots', 'ctrl', 'rf-x');
      var rdRmse = readout('surface RMSE', 'surface', 'err');
      var rdMax = readout('max error', 'max err', 'err', 'rf-x2');
      var rdUni = readout('uniform RMSE', 'uniform', '', 'rf-hide');
      var t2 = ui.el('div', 'kt-panel-title rf-t2', 'Layer B · sampling density'); pr.appendChild(t2);
      var rdG = readout('Gaussians', 'Gaussians', 'splat', 'rf-gx');
      var rdDrawn = readout('drawn RMSE', 'drawn', 'err', 'rf-gx');
      var rdAng = readout('normal error', 'normals', 'err', 'rf-hide');

      var pc = ui.panel(col, { title: 'controls' }); pc.classList.add('rf-ctrl');
      var segMap = ui.seg(pc, { label: 'map', value: 'fit',
        options: [{ v: 'fit', label: 'fit' }, { v: 'target', label: 'target' }, { v: 'error', label: 'error' }],
        onChange: function (v) { st.view = v; dirty = true; } });
      var segShade = ui.seg(pc, { label: 'shade', value: 'height',
        options: [{ v: 'height', label: 'height' }, { v: 'normal', label: 'normals' }],
        onChange: function (v) { setShade(v); } });
      var rowB = ui.el('div', 'kt-row rf-btns'); pc.appendChild(rowB);
      ui.button(rowB, { label: 'insert', onClick: function () { userInsert(); } });
      ui.button(rowB, { label: 'remove', onClick: function () { userRemove(); } });
      ui.button(rowB, { label: 'auto', onClick: function () { userAuto(); } });
      ui.button(rowB, { label: 'reset', onClick: function () { userReset(); } });
      var rowP = ui.el('div', 'kt-row rf-btns rf-pre'); pc.appendChild(rowP);
      var preBtn = {};
      ['ctrl', 'gauss', 'bal'].forEach(function (k) { preBtn[k] = ui.button(rowP, { label: REGIMES[k].label, onClick: function () { userPreset(k); } }); });
      preBtn.tour = ui.button(rowP, { label: 'tour', onClick: function () { startTour(); } });
      var rowT = ui.el('div', 'kt-row rf-btns rf-tog'); pc.appendChild(rowT);
      /* one button, named for what a press does now: pause what is playing, resume it, or play this step again */
      var btRun = ui.button(rowT, { label: 'pause', onClick: function () {
        if (!st.run) st.run = true; else if (playing()) st.run = false; else setScene(st.scene);
      } });
      var tgG = ui.toggle(rowT, { label: 'gaussians', value: false, onChange: function (v) { st.gauss = v; st.densAuto = false; st.tour = null; relayout(); } });
      tgG.el.classList.add('rf-g');
      var slBud = ui.slider(pc, { label: 'budget', min: 0, max: BUDGETS.length - 1, step: 1, value: BUDGETS.indexOf(CFG.BUDGET),
        fmt: function (i) { return thousands(BUDGETS[i]); },
        onInput: function (i) { st.budget = BUDGETS[i]; } });
      slBud.el.classList.add('rf-bud');
      var slDens = ui.slider(pc, { label: 'samples', min: 0, max: DSTEPS, step: 1, value: densIdx(24),
        fmt: function (i) { return densOf(i) + '²'; },
        onInput: function (i) { st.dens = densOf(i); st.densAuto = false; st.tour = null; if (!st.gauss) { st.gauss = true; tgG.set(true); relayout(); } } });
      slDens.el.classList.add('rf-dens');
      var densVal = slDens.el.querySelector('.kt-val');

      if (env.embed) {                    // inside the deck, never keep keyboard focus: space and arrows belong to it
        col.addEventListener('click', function () { var a = document.activeElement; if (a && col.contains(a) && a.blur) a.blur(); });
        [slDens, slBud].forEach(function (s) { s.input.addEventListener('change', function () { s.input.blur(); }); });
      }

      /* ---------- state ---------- */
      var st = { scene: '', run: true, view: 'fit', shade: 'height', gauss: false, dens: 24, densAuto: false, densT: 0,
                 mode: 'idle', timer: 0, rate: 12, acc: 0, op: null, pending: null, versus: false,
                 ceiling: null, blocked: null, pruned: 0, ptsFrom: 0, inserted: 0, barScale: 0, holdScale: false,
                 budget: CFG.BUDGET, cut: 0.365, bal: false, tour: null, want: null, preset: '', batch: false, manual: false };
      var m = null, ms = null, base = null, baseMs = null, bestU = null;
      var fx = [], dirty = true, dirtyB = true, zVersion = 0;
      var status = { l1: '', l2: '', tone: 'text', live: null };
      var Wc = 0, Hc = 0, L = null, tiny = false, big = false;
      var perf = { top: 0, frame: 0 }, lastFrame = 0;

      /* display buffers */
      var disp = { n: 0 };
      function setDispRes(n) {
        if (disp.n === n) return;
        disp.n = n; disp.xs = centres(n);
        disp.tgt = new Float32Array(n * n);
        for (var b = 0; b < n; b++) for (var a = 0; a < n; a++) disp.tgt[b * n + a] = targetAt(tg, disp.xs[a], disp.xs[b]);
        disp.fitA = new Float32Array(n * n); disp.fitB = new Float32Array(n * n); disp.err = new Float32Array(n * n);
        disp.drawn = new Float32Array(n * n); disp.n3 = null; disp.tn3 = null; disp.bt = null;
        disp.cA = mkCanvas(n); disp.cB = mkCanvas(n);
        disp.tabA = null; disp.tabB = null; dirty = dirtyB = true;
      }
      function mkCanvas(n) {
        var c = document.createElement('canvas'); c.width = c.height = n;
        var x = c.getContext('2d'); return { c: c, x: x, img: x.createImageData(n, n) };
      }
      function tablesFor(mod) { return { m: mod, TU: table(mod.U, mod.nu, disp.xs), TV: table(mod.V, mod.nv, disp.xs) }; }

      /* ---------- Layer B: the Gaussians of the live model, cached per (density, surface) ---------- */
      var gsA = null, gsKey = '', dr = { key: '', v: null, ws: {} }, an = { key: '', v: null, buf: null };
      function ensureGauss() {
        var key = st.dens + '|' + zVersion;
        if (gsKey !== key || !gsA) { gsA = sampleGauss(m, st.dens, gsA); gsKey = key; }
        return gsA;
      }
      /* drawn error of the live state (null while the Gaussians are off) */
      function drawnNow() {
        if (!st.gauss) return null;
        var key = st.dens + '|' + zVersion + '|' + st.shade;
        if (dr.key !== key) {
          if (st.shade === 'normal' && !dr.N) dr.N = new Float32Array(tg.G * tg.G * 3);
          dr.ws.N = st.shade === 'normal' ? dr.N : null;
          dr.v = drawnError(m, st.dens, dr.ws, ensureGauss()); dr.key = key;
        }
        return dr.v;
      }
      /* mean angle (degrees) between the normals that are shown and the target's, on the target grid */
      function angleNow() {
        var key = (st.gauss ? st.dens : 0) + '|' + zVersion + '|' + st.shade;
        if (an.key !== key) {
          if (st.gauss) { drawnNow(); an.v = dr.ws.N ? angleStats(dr.ws.N, targetNormals(tg)).mean : null; }
          else { an.buf = fitNormals(m, an.buf); an.v = angleStats(an.buf, targetNormals(tg)).mean; }
          an.key = key;
        }
        return an.v;
      }
      /* which knob is starving the picture: 'ctrl' | 'gauss' | 'bal' (null while the Gaussians are off) */
      function regime() {
        var d = drawnNow(); if (!d) return null;
        return d.samp > REGIME_K * ms.rmse ? 'gauss' : ms.rmse > REGIME_K * d.samp ? 'ctrl' : 'bal';
      }

      /* the top view: continuous surface, or (Gaussians on) what the Gaussians draw */
      function paintTop(cvs, mod, tab, fitBuf, view, escale, useG) {
        var n = disp.n, px = cvs.img.data, H = fitBuf, pal = PAL_FIT, N3 = null, T3, i, d, q;
        evalGrid(mod.Z, mod.nu, tab.TU, tab.TV, fitBuf);
        if (useG) {
          var gs = ensureGauss();
          if (!disp.bt || disp.bt.n !== gs.n) disp.bt = blendTable(gs.n, disp.xs);
          if (st.shade === 'normal' && !disp.n3) disp.n3 = new Float32Array(n * n * 3);
          renderGauss(gs, disp.bt, disp.bt, disp.drawn, st.shade === 'normal' ? disp.n3 : null);
          H = disp.drawn; pal = PAL_GSS; if (st.shade === 'normal') N3 = disp.n3;
        }
        if (st.shade === 'normal') {
          if (!disp.tn3) disp.tn3 = normalsOf(disp.tgt, n, new Float32Array(n * n * 3));
          if (!disp.n3) disp.n3 = new Float32Array(n * n * 3);
          T3 = disp.tn3;
          if (view === 'target') paintNormals(px, T3, n);
          else {
            if (!N3) N3 = normalsOf(H, n, disp.n3);
            if (view === 'error') {
              for (i = 0, q = 0; i < n * n; i++, q += 3) {
                d = N3[q] * T3[q] + N3[q + 1] * T3[q + 1] + N3[q + 2] * T3[q + 2];
                disp.err[i] = Math.acos(d > 1 ? 1 : d < -1 ? -1 : d) * 57.29578;
              }
              shade(px, H, n, PAL_TGT, disp.err, ANG_SCALE);
            } else paintNormals(px, N3, n);
          }
        } else if (view === 'target') shade(px, disp.tgt, n, PAL_TGT);
        else if (view === 'error') {
          for (i = 0; i < n * n; i++) disp.err[i] = H[i] - disp.tgt[i];
          shade(px, H, n, PAL_TGT, disp.err, escale);
        } else shade(px, H, n, pal);
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
      function autoPace() {                // the scripted 18 knots keep their pace; a larger budget speeds up
        var k = st.inserted, f = k < 18 ? 1 : k < 40 ? 0.5 : 0.22, a = PACE.auto;
        return { gap: a.gap * f, hi: a.hi * f, hold: a.hold * f, rate: a.rate / f };
      }
      function beginInsert(pace, auto) {
        snap();
        var w = nextInsertion(m, ms, auto ? st.budget : CFG.MAX_PTS);
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
            function () { return 'surface unchanged (max |Δz| ' + expStr(op.dz) + ') · RMSE ' + pct(st.rmse0) + ' → ' + pctA(ms.rmse); });
        } else {
          setStatus(r.dz <= 1e-9 ? 'SURFACE UNCHANGED · +' + r.added + ' CONTROL POINTS' : 'SURFACE MOVED — INSERTION BUG',
                    'Boehm insertion at ' + where + ' · max |Δz| = ' + expStr(r.dz) + ' · RMSE still ' + pctA(before), r.dz <= 1e-9 ? 'ok' : 'err');
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
              function () { return 'RMSE ' + pctA(op.before) + ' → ' + pctA(ms.rmse) + (m.cg.done ? ' · converged' : ''); });
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
        if (!beginInsert(autoPace(), true)) {
          st.mode = 'idle';
          setStatus('BUDGET REACHED · ' + thousands(m.nu * m.nv) + ' / ' + thousands(st.budget) + ' CONTROL POINTS',
                    'RMSE ' + pct(st.rmse0) + ' → ' + pctA(ms.rmse) + ' · ' + st.inserted + (m.nu * m.nv > 400 ? ' knots inserted' : ' knots, clustered over the crater'), 'ok');
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
      function playing() { return !!(st.op || st.pending || st.mode !== 'idle' || st.tour || st.densAuto || !m.cg.done); }
      function userInsert() {
        if (busy()) return; st.mode = 'idle'; st.pending = null; manual(); st.rmse0 = st.rmse0 || ms.rmse;
        if (!beginInsert(PACE.button, false)) {
          var w = worstSpan(ms, CFG.MIN_SAMPLES);
          if (w) setStatus('CAP REACHED · ' + thousands(m.nu * m.nv) + ' OF ' + thousands(CFG.MAX_PTS) + ' CONTROL POINTS', 'one more knot would add ' + (w.axis === 'u' ? m.nv : m.nu) + ' control points', 'text');
          else setStatus('NO SPAN LEFT TO SPLIT', 'every span is already down to ' + CFG.MIN_SAMPLES + ' sample lines', 'text');
        }
      }
      function userRemove() {
        if (busy()) return; st.mode = 'idle'; st.pending = null; manual();
        if (!st.pruned) st.ptsFrom = m.nu * m.nv;
        if (!beginRemove(PACE.button)) blockedStatus('NO KNOT CAN GO WITHIN TOLERANCE');
      }
      function userAuto() {
        if (busy()) return; st.pending = null; manual(); st.mode = 'auto'; st.timer = autoPace().gap; st.rmse0 = st.rmse0 || ms.rmse;
        if (m.nu * m.nv >= st.budget) setStatus('BUDGET ' + thousands(st.budget) + ' CONTROL POINTS', 'raise the budget (up to ' + thousands(CFG.MAX_PTS) + ') to keep inserting', 'text');
      }
      function userReset() { manual(); enterCoarse(true); }
      function manual() { st.tour = null; st.preset = ''; st.want = null; st.manual = true; markPreset(); }

      /* ---------- the three regimes ---------- */
      function markPreset() { for (var k in preBtn) preBtn[k].classList.toggle('on', k === st.preset || (k === 'tour' && !!st.tour)); }
      function applyPreset(which) {
        var snapS = which === 'ctrl' ? S.coarse() : B.cols[which === 'gauss' ? B.cols.length - 1 : B.balCol].snap;
        clearRun(); setModel(S.restore(snapS));
        st.inserted = 0; st.pruned = 0; st.ceiling = null; st.blocked = null; st.rmse0 = S.coarseRmse; st.barScale = 0;
        st.gauss = true; tgG.set(true); st.densAuto = false; setDens(REGIMES[which].n);
        st.view = 'fit'; segMap.set('fit'); st.preset = which; st.want = null; st.bal = true; st.manual = false;
        setStatus('', '', 'text'); markPreset(); relayout();
      }
      function wantPreset(which) {         // the nets beyond 5 x 5 come from the measured sequence: wait for it
        if (which === 'ctrl' || B.cols) applyPreset(which); else { st.want = which; st.bal = true; relayout(); }
      }
      function userPreset(which) { st.tour = null; wantPreset(which); }
      function startTour() { st.tour = { i: 0, t: 0, order: ['ctrl', 'gauss', 'bal'] }; wantPreset('ctrl'); }

      /* ---------- scenes ---------- */
      function setView(v) { st.view = v; segMap.set(v); dirty = dirtyB = true; }
      function setGauss(on) { st.gauss = on; tgG.set(on); if (!st.batch) relayout(); }
      function setShade(v) {
        st.shade = v; segShade.set(v); dirty = dirtyB = true;
        rdAng.el.classList.toggle('rf-hide', v !== 'normal');
        if (!st.batch) relayout();
      }
      function setDens(n) { st.dens = n; slDens.set(densIdx(n)); densVal.textContent = n + '²'; }
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
        st.batch = true;
        st.scene = id; st.barScale = 0; st.versus = id === 'versus'; st.densAuto = false;
        st.bal = id === 'sampling' || id === 'balance'; st.tour = null; st.want = null; st.preset = ''; st.manual = false;
        st.budget = CFG.BUDGET; slBud.set(BUDGETS.indexOf(CFG.BUDGET));
        setShade(id === 'normals' ? 'normal' : 'height');
        rdUni.el.classList.toggle('rf-hide', !st.versus);
        st.run = true;
        if (id === 'coarse') { enterCoarse(true); setView('fit'); setGauss(false); }
        else if (id === 'insert') {
          enterCoarse(false); setView('fit'); setGauss(false); st.holdScale = true;
          setStatus('COARSE NET · ' + netStr(m) + ' · RMSE ' + pct(ms.rmse), 'where is the error? measured per u-span and per v-span', 'text');
          st.pending = { delay: PACE.sceneInsert.delay, fn: function () { beginInsert(PACE.sceneInsert, false); } };
        } else if (id === 'auto') {
          enterCoarse(false); setView('fit'); setGauss(false);
          setStatus('AUTO-INSERT · BUDGET ' + st.budget + ' CONTROL POINTS', 'one knot every ' + PACE.auto.gap + ' s · midpoint of the span holding the most error', 'ctrl');
          st.mode = 'auto'; st.timer = PACE.auto.gap - 0.7;
        } else if (id === 'remove') {
          clearRun(); setModel(S.restore(S.auto())); setView('fit'); setGauss(false);
          st.inserted = S.autoHist.length - 1; st.pruned = 0; st.ptsFrom = m.nu * m.nv; st.rmse0 = S.autoHist[0].rmse;
          st.ceiling = S.autoRmse * CFG.TOL; st.blocked = null;
          setStatus('PRUNE · ' + m.nu * m.nv + ' CONTROL POINTS · RMSE ' + pct(ms.rmse, 3),
                    'drop the cheapest knot, re-fit · keep RMSE ≤ ' + pct(st.ceiling, 3) + ' (+' + Math.round((CFG.TOL - 1) * 100) + ' %)', 'ctrl');
          st.mode = 'prune'; st.timer = PACE.prune.gap - 1.0;
        } else if (id === 'balance') {
          setGauss(true); startTour();
        } else {
          clearRun(); setModel(S.restore(S.fin()));
          st.inserted = S.autoHist.length - 1; st.pruned = S.pruneHist.length; st.rmse0 = S.autoHist[0].rmse; st.ceiling = null;
          if (id === 'versus') { setView('error'); setGauss(false); setStatus('', '', 'text'); }
          else if (id === 'normals') { setView('fit'); setGauss(true); setDens(40); setStatus('', '', 'splat'); }
          else {
            setView('fit'); setGauss(true); st.densAuto = true; st.densT = 0;
            setStatus('', '', 'splat');
          }
        }
        st.batch = false; markPreset();
        relayout();
      }

      function refreshBase() {
        var key = m.nu + 'x' + m.nv;
        if (base && base.key === key) return;
        base = uniformBaseline(tg, m.nu, m.nv); base.key = key; baseMs = measure(base);
        bestU = bestUniform(tg, m.nu * m.nv); disp.tabB = null; dirtyB = true; sdB.key = '';
      }

      /* ---------- per-frame step ---------- */
      function step(dt) {
        var i;
        for (i = fx.length - 1; i >= 0; i--) { fx[i].age += dt; if (fx[i].age > fx[i].life) fx.splice(i, 1); }
        if (st.op) stepOp(dt);
        else {
          if (!m.cg.done) optimize(dt);
          if (st.pending) { st.timer += dt; if (st.timer >= st.pending.delay) { var f = st.pending.fn; st.pending = null; st.timer = 0; f(); } }
          else if (st.mode === 'auto') { if (st.timer >= autoPace().gap) autoNext(); }
          else if (st.mode === 'prune') { if (st.timer >= PACE.prune.gap) pruneNext(); }
        }
        if (st.mode !== 'idle' && !st.pending) st.timer += dt;      // the cadence includes the beat itself
        if (st.densAuto) {                                          // 64 -> 16 384 Gaussians and back, geometric
          st.densT += dt;
          var d = Math.round(8 * Math.pow(16, 0.5 - 0.5 * Math.cos(st.densT * 2 * Math.PI / 9)));
          if (d !== st.dens) setDens(d);
        }
        if (st.tour && !st.want) {                                  // the three regimes, once each
          st.tour.t += dt;
          if (st.tour.t >= PACE.tour && B.cols) {
            st.tour.t = 0;
            if (++st.tour.i >= st.tour.order.length) { st.tour = null; markPreset(); }
            else { var keep = st.tour; applyPreset(keep.order[keep.i]); st.tour = keep.i < keep.order.length - 1 ? keep : null; markPreset(); }
          }
        }
      }

      /* ---------- layout ---------- */
      function relayout() { if (Wc > 0) layout(Wc, Hc); }
      /* Two views of one state. `col`: the side view sits under the map with the same width, so u lines
         up in both (narrow or square stages). `row`: the side view sits beside the map (wide stages).
         The balance map takes the place of the per-span error bars when it is shown. */
      function solveMain(CW, CH, g) {
        var pad = g.pad, fs = g.fs, showBal = g.showBal, bars = !showBal;
        var bar = bars ? Math.round(tiny ? 26 : KT.clamp(Math.min(CW, CH) * 0.075, 28, 72)) : 0;
        var gap = tiny ? 5 : 8, lab = fs + 8, rot = bars ? fs + 8 : 0, labS = tiny ? 0 : fs + 9, gapV = tiny ? 6 : 14, gapS = tiny ? 5 : 12;
        var SA = CFG.ZW * ZSPAN, Wa = CW - 2 * pad, Ha = CH - 2 * pad;
        var mapTop = lab + (bars ? bar + gap : 0), sideX = bars ? gap + bar + rot : 0;
        var gapB = tiny ? 10 : 24, bwMin = tiny ? 128 : 230, bhMin = tiny ? 120 : 210, gapM = tiny ? 10 : 28;
        var hFix = mapTop + gapV + labS + gapS + g.statusH, cands = [], i, c, s, ws, best = null;
        if (!showBal) cands.push({ arr: 'col', s: Math.min((Ha - hFix) / (1 + SA), Wa - sideX) });
        else {
          cands.push({ arr: 'col', balAt: 'right', s: Math.min((Ha - hFix) / (1 + SA), Wa - gapB - bwMin) });
          cands.push({ arr: 'col', balAt: 'below', s: Math.min((Ha - hFix - gapB - bhMin) / (1 + SA), Wa) });
        }
        if (!tiny) {
          s = Math.min(Ha - mapTop - gapS - g.statusH, (Wa - sideX - gapM) / 1.85);
          ws = Math.min(Wa - sideX - gapM - s, 1.3 * s);
          if (showBal) ws = Math.min(ws, (mapTop + s + gapS + g.statusH - labS - gapB - bhMin) / SA);
          cands.push({ arr: 'row', s: s, ws: ws });
        }
        for (i = 0; i < cands.length; i++) {
          c = cands[i]; c.s = Math.floor(c.s);
          c.score = c.arr === 'row' ? 0.96 * Math.min(c.s, c.ws) : c.s;
          if (!best || c.score > best.score) best = c;
        }
        var o = { score: best.score, arr: best.arr, bars: bars, bar: bar, gap: gap, ms: Math.max(0, best.s), fs: fs, pad: pad };
        s = o.ms;
        var x0, y0, hs, bw, bh, blockW, blockH;
        if (best.arr === 'col') {
          hs = Math.round(SA * s);
          bw = best.balAt === 'right' ? Math.floor(Math.min(Wa - s - gapB, tiny ? 190 : 560)) : 0;
          bh = best.balAt === 'below' ? Math.floor(Math.min(Ha - hFix - gapB - s - hs, 330)) : 0;
          blockW = s + sideX + (bw ? gapB + bw : 0); blockH = hFix + s + hs + (bh ? gapB + bh : 0);
          x0 = tiny ? pad : Math.round(pad + (Wa - blockW) / 2); y0 = Math.round(pad + Math.max(0, Ha - blockH) / 2);
          o.mx = x0; o.my = y0 + mapTop;
          o.side = { x: x0, y: o.my + s + gapV + labS, w: s, h: hs };
          if (bw) { o.balR = { x: x0 + s + gapB, y: y0, w: bw, h: o.side.y + hs - y0 }; if (g.compact) { o.balR.y += 40; o.balR.h -= 40; } }
          if (bh) o.balR = { x: x0, y: o.side.y + hs + gapB, w: Math.min(Wa, Math.max(s, 520)), h: bh };
          o.stat = { x: x0, y: (bh ? o.balR.y + bh : o.side.y + hs) + gapS, w: CW - x0 - pad };
        } else {
          ws = Math.floor(best.ws); hs = Math.round(SA * ws);
          blockW = s + sideX + gapM + ws; blockH = mapTop + s + gapS + g.statusH;
          x0 = Math.round(pad + (Wa - blockW) / 2); y0 = Math.round(pad + Math.max(0, Ha - blockH) / 2);
          o.mx = x0; o.my = y0 + mapTop;
          var sx = x0 + s + sideX + gapM;
          if (showBal) {
            o.side = { x: sx, y: y0 + labS, w: ws, h: hs };
            o.balR = { x: sx, y: o.side.y + hs + gapB, w: ws, h: y0 + blockH - (o.side.y + hs + gapB) };
          } else o.side = { x: sx, y: o.my + Math.round((s - hs) / 2), w: ws, h: hs };
          o.stat = { x: x0, y: o.my + s + gapS, w: s + sideX };
        }
        o.labS = labS;
        return o;
      }
      /* versus: two maps side by side, each with its own side view under it */
      function solveVersus(CW, CH, g) {
        var pad = g.pad, fs = g.fs, top = g.compact ? 36 : 0;
        var gapM = tiny ? 10 : 28, head = fs + 12, foot = tiny ? 88 : big ? 190 : 156, SA = CFG.ZW * ZSPAN, gapV = tiny ? 5 : 10;
        var sp = Math.floor(Math.min((CW - 2 * pad - gapM) / 2, (CH - top - 2 * pad - head - foot - gapV) / (1 + SA)));
        var hs = Math.round(SA * sp);
        return { score: sp, versus: true, sp: sp, hs: hs, gapV: gapV, gap: gapM, head: head, foot: foot, fs: fs, pad: pad,
                 x0: Math.round((CW - (2 * sp + gapM)) / 2), y0: top + Math.round((CH - top - (head + sp + gapV + hs + foot)) / 2) + head };
      }
      function layout(w, h) {
        Wc = w; Hc = h;
        tiny = w < 560 || h < 470; big = !tiny && Math.min(w, h) >= 860 && w >= 1000;
        var colW = big ? 330 : env.embed ? 236 : 268, squat = tiny && h < 470, i, r, c, best = null;
        var cls = 'rf-wrap' + (tiny ? ' rf-tiny' : '') + (big ? ' rf-big' : '') + (st.versus ? ' rf-vs' : '') +
                  (st.gauss ? ' rf-gon' : '') + (st.shade === 'normal' ? ' rf-non' : '');
        for (var q = 0; q < LABELS.length; q++) LABELS[q].el.textContent = tiny ? LABELS[q].short : LABELS[q].full;
        wrap.style.setProperty('--rf-col', colW + 'px');
        var g = { fs: tiny ? 11 : big ? 14 : 12, pad: tiny ? 8 : big ? 22 : 16, f1: tiny ? 12 : big ? 18 : 14, f2: tiny ? 11 : big ? 14 : 12,
                  squat: squat, compact: rootEl.classList.contains('kt-compact'), showBal: st.bal && !st.versus };
        g.statusH = squat ? 20 : tiny ? 48 : g.f1 + 2 * g.f2 + 26;
        var solve = st.versus ? solveVersus : solveMain, modes = tiny ? ['rf-stack'] : ['rf-side', 'rf-stack'];
        for (i = 0; i < modes.length; i++) {                 // try the panel beside and below the figure, keep the larger figure
          wrap.className = cls + ' ' + modes[i];
          if (modes[i] === 'rf-side' && col.scrollHeight > wrap.clientHeight + 1) wrap.className += ' rf-dense';
          r = stage.getBoundingClientRect(); c = solve(r.width, r.height, g); c.mode = wrap.className;
          if (modes[i] === 'rf-side' && col.scrollHeight > wrap.clientHeight + 1) c.score *= 0.6;      // the panel would be cut off
          if (!best || c.score > best.score) best = c;
        }
        wrap.className = best.mode;
        var sz = cv.fit();
        L = solve(sz.w, sz.h, g);
        L.W = sz.w; L.H = sz.h; L.f1 = g.f1; L.f2 = g.f2; L.squat = squat; L.statusH = g.statusH; L.mode = best.mode;
        setDispRes(pickRes(st.versus ? L.sp : L.ms, sz.dpr));
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
      /* text with a dark halo, for labels that sit on a picture */
      function halo(s, x, y, f, color, align, ls) {
        ctx.font = f; ctx.textAlign = align || 'left'; ctx.textBaseline = 'alphabetic';
        if ('letterSpacing' in ctx) ctx.letterSpacing = ls || '0px';
        ctx.lineJoin = 'round'; ctx.lineWidth = 3.5; ctx.strokeStyle = 'rgba(7,9,15,.88)'; ctx.strokeText(s, x, y);
        ctx.fillStyle = color; ctx.fillText(s, x, y);
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
        var gu = greville(mod.U, mod.nu), gv = greville(mod.V, mod.nv), i, j, small = mod.nu * mod.nv > 900;
        ctx.fillStyle = rgba('ctrl', alpha); ctx.beginPath();
        for (j = 0; j < mod.nv; j++) for (i = 0; i < mod.nu; i++) {
          var px = x + gu[i] * s, py = y + gv[j] * s;
          if (small) ctx.rect(px - 0.8, py - 0.8, 1.6, 1.6); else { ctx.moveTo(px + r, py); ctx.arc(px, py, r, 0, 6.2832); }
        }
        ctx.fill();
      }
      function frame(x, y, s, h) { ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.strokeRect(Math.round(x) - 0.5, Math.round(y) - 0.5, s + 1, (h == null ? s : h) + 1); }
      function tag(s, x, y, color, maxW) {
        var f = L.fs, ls = '0.1em';
        ctx.font = font(600, f); if ('letterSpacing' in ctx) ctx.letterSpacing = ls;
        if (maxW && ctx.measureText(s).width + 14 > maxW) { ls = '0px'; ctx.letterSpacing = ls; }
        while (maxW && ctx.measureText(s).width + 14 > maxW && f > 8.5) { f -= 0.5; ctx.font = font(600, f); }
        var w = ctx.measureText(s).width + 14, h = f + 10;
        ctx.fillStyle = 'rgba(7,9,15,.72)'; ctx.fillRect(x, y, w, h);
        txt(s, x + 7, y + h - 6, font(600, f), color, 'left', ls);
        return w;
      }

      /* Layer B overlay on the map: one Gaussian per (u,v) sample. Seen from above the footprint of a
         derived Gaussian is a circle (its tangent axes are S_u, S_v times the spacing); the tick is its
         normal S_u x S_v seen from above, so it points downhill and grows with the tilt.
         Cached; redrawn only when it changes. Too dense to draw one by one: the picture underneath is
         already what they render. */
      var gT = { c: document.createElement('canvas'), key: '', dot: document.createElement('canvas') };
      function paintGauss(g, s, dpr) {
        var gs = ensureGauss(), n = gs.n, sp = s / n, r0 = sp * 0.42, a, b, o, cx, cy, len = sp * 0.95;
        // every footprint is the same circle: draw one row of them, then stamp the row n times
        var lw = sp > 9 ? 1.1 : 0.8, d = gT.dot, dc, rh = Math.ceil(sp * dpr) + 2;
        d.width = Math.round(s * dpr); d.height = rh; dc = d.getContext('2d'); dc.setTransform(dpr, 0, 0, dpr, 0, 0);
        dc.beginPath();
        for (a = 0; a < n; a++) { cx = (a + 0.5) * sp; dc.moveTo(cx + r0, rh / dpr / 2); dc.arc(cx, rh / dpr / 2, r0, 0, 6.2832); }
        dc.strokeStyle = 'rgba(248,233,251,' + (sp > 11 ? 0.7 : 0.5) + ')'; dc.lineWidth = lw; dc.stroke();
        for (b = 0; b < n; b++) g.drawImage(d, 0, (b + 0.5) * sp - rh / dpr / 2, s, rh / dpr);
        if (sp < 9) return;
        // its normal S_u x S_v, seen from above
        g.beginPath();
        for (b = 0; b < n; b++) for (a = 0; a < n; a++) {
          o = b * n + a; cx = (a + 0.5) * sp; cy = (b + 0.5) * sp;
          g.moveTo(cx, cy); g.lineTo(cx + gs.nx[o] * len, cy + gs.ny[o] * len);
        }
        g.strokeStyle = 'rgba(7,9,15,.5)'; g.lineWidth = (sp > 14 ? 1.5 : 1.1) + 1.6; g.lineCap = 'round'; g.stroke();
        g.strokeStyle = '#ffffff'; g.lineWidth = sp > 14 ? 1.5 : 1.1; g.stroke();
      }
      function drawGauss(x, y, s) {
        if (s / st.dens < 6.5) return;
        var dpr = cv.size.dpr, px = Math.round(s * dpr), key = st.dens + '|' + px + '|' + zVersion;
        if (gT.key !== key) {
          if (gT.c.width !== px) gT.c.width = gT.c.height = px;
          var g = gT.c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, s, s);
          paintGauss(g, s, dpr); gT.key = key;
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
        var lf = font(600, L.fs), wide = s > 330;
        txt(wide ? 'ERROR MEASURED IN u' : 'ERROR IN u', x, y - gap - bar - 7, lf, C.dim, 'left', '0.12em');
        ctx.save(); ctx.translate(x + s + gap + bar + L.fs + 1, y); ctx.rotate(Math.PI / 2);
        txt(wide ? 'ERROR MEASURED IN v' : 'ERROR IN v', 0, 0, lf, C.dim, 'left', '0.12em'); ctx.restore();
        if (!tiny) txt('full bar ' + (st.barScale < 0.00995 ? pctA(st.barScale) : pct(st.barScale, 1)), x + s, y - gap - bar - 7, font(500, L.fs), C.mute, 'right', '0.04em');
      }

      function drawStatus(x, y, maxW, align) {
        var l2 = status.live ? status.live() : status.l2;
        if (status.l1) fitTxt(status.l1, x, y + L.f1, 600, L.f1, toneCol(status.tone), align, '0.07em', maxW);
        if (l2 && !L.squat) flowTxt(l2, x, y + L.f1 + L.f2 + (tiny ? 5 : 9), L.f2, C.dim, maxW, L.f2 + 5);
      }

      /* ---------- the side view: a cut along u at v = st.cut, height to scale ---------- */
      var sdA = { key: '' }, sdB = { key: '' }, Nv4 = new Float64Array(W), Dv4 = new Float64Array(W);
      function sideData(d, mod, K, useG) {
        var key = [zVersion, st.cut, K, useG ? st.dens : 0].join('|');
        if (d.key === key && d.m === mod) return d;
        d.key = key; d.m = mod;
        var v0 = st.cut, nu = mod.nu, Z = mod.Z, i, k, r, q, zz, kv, ku, a, b, e;
        if (!d.xs || d.xs.length !== K + 1) {
          d.xs = new Float64Array(K + 1); for (k = 0; k <= K; k++) d.xs[k] = k / K;
          d.fit = new Float32Array(K + 1); d.fu = new Float32Array(K + 1); d.fv = new Float32Array(K + 1);
          d.tgt = new Float32Array(K + 1); d.drawn = new Float32Array(K + 1); d.tU = null; d.tgtKey = ''; d.tx = null;
        }
        // the cut is itself a B-spline curve: its control heights are the net contracted with N_j(v0)
        kv = NB.findSpan(mod.nv, P, v0, mod.V); NB.basisFuns(kv, v0, P, mod.V, Nv4, Dv4);
        if (!d.c || d.c.length !== nu) { d.c = new Float64Array(nu); d.cd = new Float64Array(nu); }
        for (i = 0; i < nu; i++) {
          a = 0; b = 0;
          for (q = 0; q < W; q++) { zz = Z[(kv - P + q) * nu + i]; a += Nv4[q] * zz; b += Dv4[q] * zz; }
          d.c[i] = a; d.cd[i] = b;
        }
        if (!d.tU || d.tUk !== mod.U) { d.tU = table(mod.U, nu, d.xs); d.tUk = mod.U; d.gu = greville(mod.U, nu); }
        for (k = 0; k <= K; k++) {
          ku = d.tU.span[k] - P; a = 0; b = 0; e = 0;
          for (r = 0; r < W; r++) { a += d.tU.N[k * W + r] * d.c[ku + r]; b += d.tU.D[k * W + r] * d.c[ku + r]; e += d.tU.N[k * W + r] * d.cd[ku + r]; }
          d.fit[k] = a; d.fu[k] = b; d.fv[k] = e;
        }
        if (d.tgtKey !== v0 + '|' + K) { for (k = 0; k <= K; k++) d.tgt[k] = targetAt(tg, d.xs[k], v0); d.tgtKey = v0 + '|' + K; }
        if (useG) {
          var gs = ensureGauss();
          if (!d.tx || d.tx.n !== gs.n) d.tx = blendTable(gs.n, d.xs);
          renderGauss(gs, d.tx, blendTable(gs.n, [v0]), d.drawn);
        }
        var shown = useG ? d.drawn : d.fit; d.gap = 0; d.gapK = 0;
        for (k = 0; k <= K; k++) { e = Math.abs(shown[k] - d.tgt[k]); if (e > d.gap) { d.gap = e; d.gapK = k; } }
        return d;
      }
      function targetNormalAt(x, y) {      // unit normal of the target at (x, y)
        var h = 1e-4, gx = (targetAt(tg, x + h, y) - targetAt(tg, x - h, y)) / (2 * h), gy = (targetAt(tg, x, y + h) - targetAt(tg, x, y - h)) / (2 * h);
        var len = 1 / Math.sqrt(1 + CFG.ZW * CFG.ZW * (gx * gx + gy * gy));
        return [-CFG.ZW * gx * len, -CFG.ZW * gy * len, len];
      }
      function drawSide(R, mod, d, o) {
        if (R.w < 60 || R.h < 24) return;
        var K = Math.max(256, Math.min(512, Math.round(R.w / 1.5))), useG = !!o.gauss, k, i, j;   // >= 256 samples: the gap it reports does not depend on the stage size
        sideData(d, mod, K, useG);
        var xs = d.xs, shown = useG ? d.drawn : d.fit, normals = st.shade === 'normal';
        function X(u) { return R.x + u * R.w; }
        function Y(z) { return R.y + R.h - (z - ZLO) / ZSPAN * R.h; }
        function path(arr) { ctx.beginPath(); ctx.moveTo(X(0), Y(arr[0])); for (var q = 1; q <= K; q++) ctx.lineTo(X(xs[q]), Y(arr[q])); }
        ctx.save(); ctx.beginPath(); ctx.rect(R.x, R.y, R.w, R.h); ctx.clip();
        ctx.fillStyle = 'rgba(255,255,255,.02)'; ctx.fillRect(R.x, R.y, R.w, R.h);
        // relief 0 and 1
        ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.setLineDash([2, 5]); ctx.beginPath();
        ctx.moveTo(R.x, Math.round(Y(0)) + 0.5); ctx.lineTo(R.x + R.w, Math.round(Y(0)) + 0.5);
        ctx.moveTo(R.x, Math.round(Y(1)) + 0.5); ctx.lineTo(R.x + R.w, Math.round(Y(1)) + 0.5); ctx.stroke(); ctx.setLineDash([]);
        // u knots, as in the map above
        ctx.strokeStyle = rgba('ctrl', o.dimCtrl ? 0.1 : 0.2); ctx.beginPath();
        for (i = W; i < mod.nu; i++) { var kx = Math.round(X(mod.U[i])) + 0.5; ctx.moveTo(kx, R.y); ctx.lineTo(kx, R.y + R.h); }
        ctx.stroke();
        // every control height of the net, seen from the side
        if (!tiny) {
          var gu = d.gu, cs2 = mod.nu * mod.nv > 900 ? 1.4 : 2;
          ctx.fillStyle = rgba('ctrl', o.dimCtrl ? 0.12 : 0.26); ctx.beginPath();
          for (j = 0; j < mod.nv; j++) for (i = 0; i < mod.nu; i++) ctx.rect(X(gu[i]) - cs2 / 2, Y(mod.Z[j * mod.nu + i]) - cs2 / 2, cs2, cs2);
          ctx.fill();
        }
        // the error, as a gap between the target and what is shown
        ctx.beginPath(); ctx.moveTo(X(0), Y(d.tgt[0]));
        for (k = 1; k <= K; k++) ctx.lineTo(X(xs[k]), Y(d.tgt[k]));
        for (k = K; k >= 0; k--) ctx.lineTo(X(xs[k]), Y(shown[k]));
        ctx.closePath(); ctx.fillStyle = rgba('err', 0.42); ctx.fill();
        // target, then the surface
        path(d.tgt); ctx.strokeStyle = 'rgba(221,229,240,.82)'; ctx.lineWidth = tiny ? 1 : 1.3; ctx.stroke();
        path(d.fit); ctx.strokeStyle = o.color || C.surf; ctx.lineWidth = tiny ? 1.6 : 2.1; ctx.stroke();
        // the control polygon of this cut
        var ca = o.dimCtrl ? 0.4 : 1;
        ctx.strokeStyle = rgba('ctrl', 0.5 * ca); ctx.lineWidth = 1; ctx.beginPath();
        for (i = 0; i < mod.nu; i++) { if (i) ctx.lineTo(X(d.gu[i]), Y(d.c[i])); else ctx.moveTo(X(d.gu[i]), Y(d.c[i])); }
        ctx.stroke();
        var rr = mod.nu > 40 ? 1.5 : tiny ? 2 : 2.8;
        ctx.fillStyle = rgba('ctrl', 0.95 * ca); ctx.beginPath();
        for (i = 0; i < mod.nu; i++) { var cx0 = X(d.gu[i]), cy0 = KT.clamp(Y(d.c[i]), R.y + 2, R.y + R.h - 2); ctx.moveTo(cx0 + rr, cy0); ctx.arc(cx0, cy0, rr, 0, 6.2832); }
        ctx.fill();
        var Ln, step, nrm, tn;
        if (useG) {
          /* the Gaussians that reach the cut. Each is drawn where its tangent plane crosses the cut
             plane: a chord through (u_a, z + z_v (v0 - v_b)) with slope z_u, one sample spacing long. */
          var gs = ensureGauss(), n = gs.n, sp = R.w / n, p = st.cut * n - 0.5, half = 0.5 / n, b, a, oo, zc, dz, wsum = 0, wts = [], bd = -1, wb = 0;
          var b0 = Math.max(0, Math.ceil(p - CFG.RCUT)), b1 = Math.min(n - 1, Math.floor(p + CFG.RCUT)), cc = Math.exp(-CFG.RCUT * CFG.RCUT / (2 * CFG.SIG * CFG.SIG));
          for (b = b0; b <= b1; b++) { var ww = Math.max(0, Math.exp(-(p - b) * (p - b) / (2 * CFG.SIG * CFG.SIG)) - cc); wts.push(ww); wsum += ww; if (ww > wb) { wb = ww; bd = b; } }
          if (bd < 0) { bd = KT.clamp(Math.round(p), 0, n - 1); b0 = b1 = bd; wts = [1]; wsum = 1; }
          if (sp >= 3) for (b = b0; b <= b1; b++) {
            var wn = wts[b - b0] / wsum; if (wn < 0.12) continue;
            ctx.beginPath();
            for (a = 0; a < n; a++) {
              oo = b * n + a; zc = gs.z[oo] + gs.zv[oo] * (st.cut - gs.xs[b]); dz = gs.zu[oo] * half;
              ctx.moveTo(X(gs.xs[a] - half), Y(zc - dz)); ctx.lineTo(X(gs.xs[a] + half), Y(zc + dz));
            }
            ctx.strokeStyle = rgba('splat', 0.25 + 0.75 * wn); ctx.lineWidth = sp >= 9 ? 2.8 : 1.8; ctx.lineCap = 'butt'; ctx.stroke();
          }
          path(d.drawn); ctx.strokeStyle = sp >= 3 ? 'rgba(248,233,251,.75)' : C.splat; ctx.lineWidth = sp >= 3 ? 1 : 2; ctx.stroke();
          // normal needles on the row nearest the cut: S_u x S_v, projected into the cut plane
          step = Math.max(1, Math.ceil(7 / sp)); Ln = KT.clamp(sp * 1.1, 9, tiny ? 13 : 20);
          for (a = (step >> 1); a < n; a += step) {
            oo = bd * n + a; zc = gs.z[oo] + gs.zv[oo] * (st.cut - gs.xs[bd]);
            var px0 = X(gs.xs[a]), py0 = Y(zc);
            if (normals) {
              tn = targetNormalAt(gs.xs[a], st.cut);
              var ty0 = Y(targetAt(tg, gs.xs[a], st.cut));
              ctx.strokeStyle = 'rgba(221,229,240,.55)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px0, ty0); ctx.lineTo(px0 + tn[0] * Ln, ty0 - tn[2] * Ln); ctx.stroke();
            }
            ctx.strokeStyle = normals ? normalCss(gs.nx[oo], gs.ny[oo], gs.nz[oo]) : '#f3d9f7'; ctx.lineWidth = normals ? 2 : 1.2;
            ctx.beginPath(); ctx.moveTo(px0, py0); ctx.lineTo(px0 + gs.nx[oo] * Ln, py0 - gs.nz[oo] * Ln); ctx.stroke();
            if (sp >= 7) { ctx.fillStyle = C.splat; ctx.beginPath(); ctx.arc(px0, py0, sp >= 12 ? 2.3 : 1.6, 0, 6.2832); ctx.fill(); }
          }
        } else if (normals) {
          // no Gaussians: the surface's own normals along the cut, against the target's
          step = Math.max(1, Math.round(14 / (R.w / K))); Ln = tiny ? 11 : 17;
          for (k = (step >> 1); k <= K; k += step) {
            var len = 1 / Math.sqrt(1 + CFG.ZW * CFG.ZW * (d.fu[k] * d.fu[k] + d.fv[k] * d.fv[k]));
            nrm = [-CFG.ZW * d.fu[k] * len, -CFG.ZW * d.fv[k] * len, len]; tn = targetNormalAt(xs[k], st.cut);
            ctx.strokeStyle = 'rgba(221,229,240,.55)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(xs[k]), Y(d.tgt[k])); ctx.lineTo(X(xs[k]) + tn[0] * Ln, Y(d.tgt[k]) - tn[2] * Ln); ctx.stroke();
            ctx.strokeStyle = normalCss(nrm[0], nrm[1], nrm[2]); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(xs[k]), Y(d.fit[k])); ctx.lineTo(X(xs[k]) + nrm[0] * Ln, Y(d.fit[k]) - nrm[2] * Ln); ctx.stroke();
          }
        }
        // the largest gap on this cut, dimensioned
        if (d.gap > 0.012 && !o.noGap) {
          var gx0 = Math.round(X(xs[d.gapK])) + 0.5, ya = Y(d.tgt[d.gapK]), yb = Y(shown[d.gapK]);
          ctx.strokeStyle = '#ffd9d2'; ctx.lineWidth = 1.2; ctx.beginPath();
          ctx.moveTo(gx0, ya); ctx.lineTo(gx0, yb); ctx.moveTo(gx0 - 3, ya); ctx.lineTo(gx0 + 3, ya); ctx.moveTo(gx0 - 3, yb); ctx.lineTo(gx0 + 3, yb); ctx.stroke();
          var right = gx0 < R.x + R.w * 0.72;
          halo(pct(d.gap, d.gap < 0.0995 ? 1 : 0), gx0 + (right ? 6 : -6), KT.clamp((ya + yb) / 2 + 4, R.y + 12, R.y + R.h - 4), font(600, tiny ? 10 : L.fs), '#ffd9d2', right ? 'left' : 'right', '0.02em');
        }
        ctx.restore();
        frame(R.x, R.y, R.w, R.h);
      }
      /* caption of the side view: what the lines are, in their colours */
      function sideCaption(R, useG, y) {
        var f = font(600, L.fs), f5 = font(500, L.fs), x, parts, i, w, tot = 0, lw;
        var long = 'SIDE VIEW · CUT AT v = ' + st.cut.toFixed(3), short = 'SIDE · v = ' + st.cut.toFixed(2);
        parts = [['target', 'rgba(221,229,240,.9)'], ['surface', C.surf], ['control', C.ctrl]];
        if (useG) parts.push(['Gaussians', C.splat]);
        parts.push(['error', C.err]);
        ctx.font = f5; if ('letterSpacing' in ctx) ctx.letterSpacing = '0.02em';
        for (i = 0; i < parts.length; i++) tot += ctx.measureText(parts[i][0]).width + 12;
        ctx.font = f; if ('letterSpacing' in ctx) ctx.letterSpacing = '0.12em';
        lw = ctx.measureText(long).width;
        if (lw + 14 + tot > R.w) { long = short; lw = ctx.measureText(short).width; }
        txt(long, R.x, y, f, C.dim, 'left', '0.12em');
        if (lw + 14 + tot > R.w) return;
        x = R.x + R.w;
        for (i = parts.length - 1; i >= 0; i--) { ctx.font = f5; if ('letterSpacing' in ctx) ctx.letterSpacing = '0.02em'; w = ctx.measureText(parts[i][0]).width; txt(parts[i][0], x, y, f5, parts[i][1], 'right', '0.02em'); x -= w + 12; }
      }

      /* the cut line on the map, with its grip */
      function cutY() { return L.my + st.cut * L.ms; }
      function drawCut() {
        var x = L.mx, s = L.ms, y = Math.round(cutY()) + 0.5;
        ctx.strokeStyle = 'rgba(7,9,15,.55)'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + s, y); ctx.stroke();
        ctx.strokeStyle = 'rgba(255,255,255,.92)'; ctx.lineWidth = 1; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + s, y); ctx.stroke(); ctx.setLineDash([]);
        var gx = x + (tiny ? 9 : 12), r = tiny ? 4.5 : 5.5;
        ctx.fillStyle = drag != null ? '#ffffff' : 'rgba(255,255,255,.95)'; ctx.strokeStyle = 'rgba(7,9,15,.8)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(gx, y, r, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.strokeStyle = 'rgba(7,9,15,.8)'; ctx.lineWidth = 1.2; ctx.beginPath();
        ctx.moveTo(gx - 2.2, y - 1.2); ctx.lineTo(gx, y - 3.2); ctx.lineTo(gx + 2.2, y - 1.2);
        ctx.moveTo(gx - 2.2, y + 1.2); ctx.lineTo(gx, y + 3.2); ctx.lineTo(gx + 2.2, y + 1.2); ctx.stroke();
        if (!tiny) halo('CUT', x + s - 7, y - 5, font(600, L.fs - 1), 'rgba(255,255,255,.92)', 'right', '0.14em');
      }

      /* ---------- the balance map ---------- */
      function balX(g, pts) {               // x of a control-point count on the (log, column) axis
        var c = B.cols, f, j;
        if (!c) f = Math.log(pts / 25) / Math.log(CFG.MAX_PTS / 25) * (g.nc - 1);
        else if (pts <= c[0].pts) f = 0; else if (pts >= c[c.length - 1].pts) f = c.length - 1;
        else { for (j = 0; j < c.length - 2 && pts >= c[j + 1].pts; j++) { /* find the bracket */ } f = j + Math.log(pts / c[j].pts) / Math.log(c[j + 1].pts / c[j].pts); }
        return g.x + (f + 0.5) * g.cell;
      }
      function balY(g, n) { var r = B.rows; return g.y + g.h - (Math.log(n / r[0]) / Math.log(r[r.length - 1] / r[0]) * (r.length - 1) + 0.5) * g.cell; }
      function drawBalance(R, t) {
        var nr = B.rows.length, nc = B.cols ? B.cols.length : 12, fs = tiny ? 9.5 : L.fs - 1, i, ci, ri, v, f = font(500, fs), f6 = font(600, fs);
        var titleH = fs + 10, axL = Math.round(tiny ? fs * 3.3 : fs * 4.6), axB = fs + 6, axT = tiny ? fs + 3 : fs + 7;
        var d = drawnNow(), reg = regime(), vb = tiny || R.h < 300 ? 0 : L.f1 + L.f2 + 40;
        var cbw = R.w - axL >= 290 ? 54 : 0;
        var cell = Math.floor(Math.min((R.w - axL - cbw) / nc, (R.h - titleH - axB - axT - vb) / nr));
        cell = KT.clamp(cell, 5, big ? 44 : 36);
        var g = { x: Math.round(R.x + axL), y: Math.round(R.y + titleH), cell: cell, nc: nc, nr: nr, w: cell * nc, h: cell * nr };
        if (!vb && !tiny) g.y += Math.round(Math.max(0, R.h - titleH - axB - axT - g.h) / 2);
        L.bg = g;
        // title
        if (tiny) fitTxt(B.ready ? 'DRAWN ERROR' : 'MEASURING ' + Math.round(100 * B.done) + ' %', R.x, g.y - 8, 600, fs, C.dim, 'left', '0.1em', R.w);
        else {
          fitTxt('DRAWN ERROR · MEASURED', R.x, g.y - 8, 600, fs + 1, C.dim, 'left', '0.12em', B.ready ? R.w : R.w - 110);
          if (!B.ready) txt('measuring ' + Math.round(100 * B.done) + ' %', g.x + g.w, g.y - 8, f, C.mute, 'right', '0.04em');
        }
        // cells
        ctx.fillStyle = 'rgba(255,255,255,.025)'; ctx.fillRect(g.x, g.y, g.w, g.h);
        if (B.cols) for (ci = 0; ci < nc; ci++) for (ri = 0; ri < nr; ri++) {
          v = B.E[ci * nr + ri]; if (v !== v) continue;
          ctx.fillStyle = mapCss(v); ctx.fillRect(g.x + ci * cell, g.y + (nr - 1 - ri) * cell, cell - (cell > 9 ? 1 : 0.5), cell - (cell > 9 ? 1 : 0.5));
        }
        frame(g.x, g.y, g.w, g.h);
        // the balance line: sampling error = surface error
        if (B.ready) {
          ctx.save(); ctx.beginPath(); ctx.rect(g.x, g.y, g.w, g.h); ctx.clip();
          ctx.strokeStyle = 'rgba(7,9,15,.6)'; ctx.lineWidth = 4; ctx.beginPath();
          for (i = 0; i < nc; i++) { var lx = g.x + (i + 0.5) * cell, ly = balY(g, B.nstar[i]); if (i) ctx.lineTo(lx, ly); else ctx.moveTo(lx, ly); }
          ctx.stroke();
          ctx.strokeStyle = C.ok; ctx.lineWidth = 1.6; ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
          ctx.restore();
          // the three regimes
          var lf = font(600, tiny ? 9 : fs), ls = '0.08em', lh = (tiny ? 9 : fs) + 3;
          var two = g.w < 330;
          var yT = g.y + Math.round(g.h * (tiny ? 0.24 : 0.30)), yB = g.y + Math.round(g.h * (tiny ? 0.58 : 0.74));
          if (tiny) {
            halo('CTRL-STARVED', g.x + 4, yT, lf, C.ctrl, 'left', '0px');
            halo('GAUSS-STARVED', g.x + g.w - 4, yB + lh, lf, C.splat, 'right', '0px');
          } else if (two) {
            halo('STARVED OF', g.x + 7, yT, lf, C.ctrl, 'left', ls); halo('CONTROL POINTS', g.x + 7, yT + lh, lf, C.ctrl, 'left', ls);
            halo('STARVED OF', g.x + g.w - 7, yB, lf, C.splat, 'right', ls); halo('GAUSSIANS', g.x + g.w - 7, yB + lh, lf, C.splat, 'right', ls);
          } else {
            halo('STARVED OF CONTROL POINTS', g.x + 8, yT, lf, C.ctrl, 'left', ls);
            halo('STARVED OF GAUSSIANS', g.x + g.w - 8, yB + lh, lf, C.splat, 'right', ls);
          }
          // "balanced", written along the line
          var ia = Math.floor(nc * 0.36), ib = Math.min(nc - 1, ia + 2);
          var ax = g.x + (ia + 0.5) * cell, ay = balY(g, B.nstar[ia]), bx = g.x + (ib + 0.5) * cell, by = balY(g, B.nstar[ib]);
          if (!tiny) {
            ctx.save(); ctx.translate((ax + bx) / 2, (ay + by) / 2); ctx.rotate(Math.atan2(by - ay, bx - ax));
            halo('BALANCED', 0, -6, lf, C.ok, 'center', ls); ctx.restore();
          }
        }
        // axes: control points along x, Gaussians along y
        var lastX = -1e9, tw;
        ctx.font = f;
        for (ci = 0; ci < nc; ci++) {
          if (!B.cols) break;
          var lab = tiny && B.cols[ci].pts >= 1000 ? Math.floor(B.cols[ci].pts / 1000) + 'k' : kfmt(B.cols[ci].pts), xx = g.x + (ci + 0.5) * cell; tw = ctx.measureText(lab).width;
          if (ci !== 0 && ci !== nc - 1 && (xx - tw / 2 < lastX + 7 || xx + tw / 2 > g.x + g.w - (tiny ? 2 * fs : ctx.measureText(kfmt(B.cols[nc - 1].pts)).width) - 10)) continue;
          txt(lab, xx, g.y + g.h + fs + 3, f, C.dim, 'center'); ctx.font = f; lastX = xx + tw / 2;
        }
        var lastY = -1e9;
        for (ri = nr - 1; ri >= 0; ri--) {
          if (KSTEPS.indexOf(B.rows[ri]) < 0) continue;
          var yy = g.y + (nr - 1 - ri + 0.5) * cell + fs * 0.36;
          if (yy - lastY < fs + 3 && ri !== 0) continue;
          v = B.rows[ri] * B.rows[ri]; txt(tiny && v >= 1000 ? Math.floor(v / 1000) + 'k' : kfmt(v), g.x - 5, yy, f, C.dim, 'right'); lastY = yy;
        }
        txt(tiny ? 'control points →' : 'CONTROL POINTS →', g.x, g.y + g.h + axB + fs + (tiny ? 1 : 4), f6, C.ctrl, 'left', tiny ? '0px' : '0.1em');
        if (tiny) { txt('Gaussians ↑', g.x, g.y + g.h + axB + 2 * fs + 5, f6, C.splat, 'left', '0px'); if (B.ready) txt('┄ balanced', g.x + g.w, g.y + g.h + axB + 2 * fs + 5, f6, C.ok, 'right', '0px'); }
        else { ctx.save(); ctx.translate(g.x - axL + fs - 1, g.y + g.h); ctx.rotate(-Math.PI / 2); txt('GAUSSIANS →', 0, 0, f6, C.splat, 'left', '0.1em'); ctx.restore(); }
        // colour bar
        if (cbw) {
          var cx = g.x + g.w + 12, k;
          for (k = 0; k < 6; k++) { ctx.fillStyle = MAP_BANDS[5 - k]; ctx.fillRect(cx, g.y + k * g.h / 6, 7, g.h / 6 - 1); }
          [0.1, 0.01, 0.001, 0.0001].forEach(function (e, q) { txt(pct(e, q > 1 ? q - 1 : 0), cx + 11, g.y + (q / 3) * g.h + (q === 0 ? fs - 1 : q === 3 ? 0 : fs * 0.36), f, C.dim, 'left'); });
        }
        // where we are
        var ox = balX(g, m.nu * m.nv), pulse = 0.5 + 0.5 * Math.sin(t * 4);
        if (d) {
          var oy = balY(g, st.dens), rr = Math.max(tiny ? 4.5 : 6, Math.min(cell * 0.5, 11));
          ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1; ctx.beginPath();
          ctx.moveTo(ox, oy + rr); ctx.lineTo(ox, g.y + g.h); ctx.moveTo(ox - rr, oy); ctx.lineTo(g.x, oy); ctx.stroke();
          ctx.strokeStyle = 'rgba(7,9,15,.85)'; ctx.lineWidth = 4.5; ctx.beginPath(); ctx.arc(ox, oy, rr, 0, 6.2832); ctx.stroke();
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(ox, oy, rr + pulse * 1.5, 0, 6.2832); ctx.stroke();
          var onRight = ox > g.x + g.w * 0.62, below = oy < g.y + 26;
          halo(pctA(d.rmse), ox + (onRight ? -rr - 6 : rr + 6), oy + (below ? rr + fs + 2 : -rr - 3), font(600, fs + 2), '#ffffff', onRight ? 'right' : 'left', '0.02em');
        } else {
          ctx.strokeStyle = rgba('ctrl', 0.9); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(ox, g.y); ctx.lineTo(ox, g.y + g.h); ctx.stroke();
        }
        // the verdict: which layer the drawn error belongs to
        if (vb) {
          var vy = g.y + g.h + axB + axT + 16, vw = Math.min(R.w, Math.max(g.w + axL, 300)), vx = R.x;
          if (!d) fitTxt('GAUSSIANS OFF · NOTHING IS DRAWN FROM SAMPLES', vx, vy + L.f1, 600, L.f1, C.dim, 'left', '0.07em', vw);
          else {
            var F2 = ms.rmse * ms.rmse, S2 = d.samp * d.samp, share = F2 / (F2 + S2 || 1);
            fitTxt(REGIMES[reg].head, vx, vy + L.f1, 600, L.f1, reg === 'bal' ? C.ok : reg === 'ctrl' ? C.ctrl : C.splat, 'left', '0.07em', vw);
            var by0 = vy + L.f1 + 9, bw0 = vw, sw = Math.round(KT.clamp(share, 0.01, 0.99) * bw0);
            ctx.fillStyle = C.ctrl; ctx.fillRect(vx, by0, sw - 1, 7); ctx.fillStyle = C.splat; ctx.fillRect(vx + sw + 1, by0, bw0 - sw - 1, 7);
            var ty = by0 + 7 + L.f2 + 5;
            fitTxt('surface ' + pctA(ms.rmse), vx, ty, 500, L.f2, C.ctrl, 'left', '0.02em', vw * 0.5 - 6);
            fitTxt('sampling ' + pctA(d.samp), vx + vw, ty, 500, L.f2, C.splat, 'right', '0.02em', vw * 0.5 - 6);
          }
        }
      }

      /* ---------- the main figure ---------- */
      function topTag() {
        var n = st.shade === 'normal', g = st.gauss, G = thousands(st.dens * st.dens);
        if (st.view === 'target') return [n ? 'TARGET · NORMALS' : 'TARGET · h(x,y)', C.text];
        if (st.view === 'error') return [n ? 'NORMAL ERROR · ANGLE TO TARGET' : g ? 'ERROR · |drawn − h|' : 'ERROR · |z − h|', C.err];
        if (g) return [n ? 'NORMALS OF ' + G + ' GAUSSIANS' : 'DRAWN BY ' + G + ' GAUSSIANS', C.splat];
        return [n ? 'NORMALS · S_u × S_v' : 'FIT · z(u,v)', C.surf];
      }
      function liveStatus() {
        var d = drawnNow(), G = thousands(st.dens * st.dens), pts = thousands(m.nu * m.nv), reg, e2, ri, cc;
        if (st.want) { status.tone = 'text'; status.l1 = 'MEASURING THE NETS · ' + Math.round(100 * B.done) + ' %'; status.l2 = 'the auto-insertion sequence, run to ' + thousands(CFG.MAX_PTS) + ' control points'; status.live = null; return; }
        if (st.scene === 'normals' && d && st.shade === 'normal') {
          var a = angleNow();
          status.tone = 'splat'; status.l1 = 'GEOMETRIC NORMALS · n = S_u × S_v, PER GAUSSIAN'; status.live = null;
          status.l2 = 'mean angle to the target’s normals ' + (a == null ? '—' : a.toFixed(1) + '°') + ' · on the 5 × 5 net it is ' + S.coarseAngle().toFixed(1) + '°';
          return;
        }
        if (!d || (st.manual && status.l1)) return;
        if (st.preset) {
          reg = regime(); status.tone = reg === 'bal' ? 'ok' : reg === 'ctrl' ? 'ctrl' : 'splat'; status.live = null;
          status.l1 = REGIMES[reg].head + (tiny ? ' · ' + pts + ' × ' + G : ' · ' + pts + ' CONTROL POINTS × ' + G + ' GAUSSIANS');
          if (reg === 'ctrl') {
            ri = B.rows.indexOf(32); e2 = B.cols && B.cols[0].pts === m.nu * m.nv && ri >= 0 ? B.E[ri] : NaN;
            status.l2 = 'drawn error ' + pctA(d.rmse) + ' · the surface itself is off by ' + pctA(ms.rmse) +
                        (e2 === e2 && st.dens > 32 ? ' · 1 024 Gaussians draw it at ' + pctA(e2) + ': the other ' + thousands(st.dens * st.dens - 1024) + ' buy nothing' : ' · more Gaussians only redraw it');
          } else if (reg === 'gauss') {
            cc = B.cols && B.balCol != null ? B.cols[B.balCol] : null; ri = B.rows.indexOf(st.dens);
            e2 = cc && ri >= 0 ? B.E[B.balCol * B.rows.length + ri] : NaN;
            status.l2 = 'the surface is within ' + pctA(ms.rmse) + ' of the target · drawn it is off by ' + pctA(d.rmse) +
                        (e2 === e2 && cc.pts < m.nu * m.nv ? ' · ' + cc.pts + ' control points drawn the same way: ' + pctA(e2) : ' · the detail is there, unsampled');
          } else status.l2 = 'drawn error ' + pctA(d.rmse) + ' · surface ' + pctA(ms.rmse) + ' and sampling ' + pctA(d.samp) + ' are of one size: neither knob is wasted';
          return;
        }
        if (st.scene === 'sampling' || !status.l1) {
          status.tone = 'splat'; status.live = null;
          status.l1 = (tiny ? '' : 'LAYER B · ') + G + ' GAUSSIANS · DRAWN ERROR ' + pctA(d.rmse);
          status.l2 = 'surface untouched · still ' + pts + ' parameters · surface RMSE ' + pctA(ms.rmse);
        }
      }
      function drawMain(t, dt) {
        var x = L.mx, y = L.my, s = L.ms, op = st.op, i, f, k, p;
        if (s < 40) return;
        if (!disp.tabA || disp.tabA.m !== m) { disp.tabA = tablesFor(m); dirty = true; }
        var key = [zVersion, st.gauss ? st.dens : 0, st.shade, st.view, disp.n].join('|');
        if (dirty || key !== disp.keyA) {
          var t0 = now(); paintTop(disp.cA, m, disp.tabA, disp.fitA, st.view, ESCALE, st.gauss); perf.top = now() - t0;
          dirty = false; disp.keyA = key;
        }
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
        var soft = st.gauss ? 0.3 : m.nu * m.nv > 900 ? 0.6 : 1;
        drawKnots(m, x, y, s, 0.72 * soft, big ? 1.4 : 1);
        if (!st.gauss && st.view !== 'error') drawDots(m, x, y, s, tiny ? 1.2 : big ? 2.3 : 1.7, 0.85);
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
        if (st.gauss && st.view !== 'target') drawGauss(x, y, s);
        drawCut();

        var tg2 = topTag();
        tag(tiny && st.gauss && st.view === 'fit' ? thousands(st.dens * st.dens) + ' GAUSSIANS' : tg2[0], x + 8, y + 8, tg2[1], s - 16);
        if (st.view === 'error' && !tiny) legend(x + 8, y + s - 8 - (L.fs + 10), st.shade === 'normal' ? ANG_SCALE : ESCALE, st.shade === 'normal');
        if (L.bars) drawBars(dt, t); else if (!tiny) txt('TOP VIEW', x, y - 8, font(600, L.fs), C.dim, 'left', '0.12em');

        // the same state from the side
        drawSide(L.side, m, sdA, { gauss: st.gauss, dimCtrl: st.gauss });
        if (L.labS) sideCaption(L.side, st.gauss, L.side.y - 8);
        else halo('SIDE · v = ' + st.cut.toFixed(2), L.side.x + 5, L.side.y + L.side.h - 5, font(600, 9), C.dim, 'left', '0.06em');
        if (L.arr === 'col' && !tiny) {     // the cut, carried down from the map to the side view
          ctx.strokeStyle = 'rgba(255,255,255,.16)'; ctx.lineWidth = 1; ctx.setLineDash([2, 4]); ctx.beginPath();
          ctx.moveTo(x - 6.5, cutY()); ctx.lineTo(x - 6.5, L.side.y + L.side.h * 0.5); ctx.lineTo(x - 1, L.side.y + L.side.h * 0.5);
          ctx.moveTo(x - 6.5, cutY()); ctx.lineTo(x - 1, cutY()); ctx.stroke(); ctx.setLineDash([]);
        }
        if (L.balR) drawBalance(L.balR, t); else L.bg = null;

        liveStatus();
        drawStatus(L.stat.x, L.stat.y, L.stat.w, 'left');
      }
      function legend(x, y, scale, deg) {
        var w = 92, h = L.fs + 10, i;
        ctx.fillStyle = 'rgba(7,9,15,.72)'; ctx.fillRect(x, y, w + 86, h);
        for (i = 0; i < w; i++) {
          var li = ((Math.pow(i / (w - 1), EGAM) * 255) | 0) * 4;
          ctx.fillStyle = 'rgba(' + (PAL_ERR[li] | 0) + ',' + (PAL_ERR[li + 1] | 0) + ',' + (PAL_ERR[li + 2] | 0) + ',' + PAL_ERR[li + 3] + ')';
          ctx.fillRect(x + 7 + i, y + 6, 1, h - 12);
        }
        txt('0 – ' + (deg ? scale + '°' : pct(scale, 0)), x + w + 15, y + h - 6, font(500, L.fs), C.dim, 'left', '0.04em');
      }

      function drawVersus(t) {
        refreshBase();
        var sp = L.sp, xa = L.x0, xb = L.x0 + sp + L.gap, y = L.y0, fs = L.fs;
        if (sp < 40) return;
        if (!disp.tabA || disp.tabA.m !== m) { disp.tabA = tablesFor(m); dirty = true; }
        if (!disp.tabB || disp.tabB.m !== base) { disp.tabB = tablesFor(base); dirtyB = true; }
        var esc = Math.max(1e-6, 0.8 * Math.max(ms.max, baseMs.max));
        var key = [zVersion, st.gauss ? st.dens : 0, st.shade, st.view, disp.n].join('|');
        if (dirty || esc !== disp.esc || key !== disp.keyA) { paintTop(disp.cA, m, disp.tabA, disp.fitA, st.view, esc, st.gauss); dirty = false; dirtyB = dirtyB || esc !== disp.esc || key !== disp.keyA; disp.keyA = key; }
        if (dirtyB) { paintTop(disp.cB, base, disp.tabB, disp.fitB, st.view, esc, false); dirtyB = false; }
        disp.esc = esc;
        ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(disp.cA.c, xa, y, sp, sp); ctx.drawImage(disp.cB.c, xb, y, sp, sp);
        frame(xa, y, sp); frame(xb, y, sp);
        drawKnots(m, xa, y, sp, 0.6, 1); drawKnots(base, xb, y, sp, 0.6, 1);
        if (st.gauss && st.view !== 'target') drawGauss(xa, y, sp);
        // one cut through both
        var yc = Math.round(y + st.cut * sp) + 0.5;
        ctx.strokeStyle = 'rgba(255,255,255,.8)'; ctx.lineWidth = 1; ctx.setLineDash([6, 4]); ctx.beginPath();
        ctx.moveTo(xa, yc); ctx.lineTo(xa + sp, yc); ctx.moveTo(xb, yc); ctx.lineTo(xb + sp, yc); ctx.stroke(); ctx.setLineDash([]);
        fitTxt((tiny ? 'ADAPTIVE · ' : 'ADAPTIVE KNOTS · ') + netStr(m), xa, y - 9, 600, fs, C.ctrl, 'left', '0.12em', sp);
        fitTxt((tiny ? 'UNIFORM · ' : 'UNIFORM KNOTS · ') + netStr(base), xb, y - 9, 600, fs, C.dim, 'left', '0.12em', sp);
        if (st.view === 'error' && !tiny) legend(xa + 8, y + sp - 8 - (fs + 10), st.shade === 'normal' ? ANG_SCALE : esc, st.shade === 'normal');
        // the same cut from the side, under each map
        var ys = y + sp + L.gapV;
        drawSide({ x: xa, y: ys, w: sp, h: L.hs }, m, sdA, { gauss: st.gauss, dimCtrl: st.gauss });
        drawSide({ x: xb, y: ys, w: sp, h: L.hs }, base, sdB, { gauss: false });
        // the two numbers
        y = ys + L.hs - sp;                  // everything below hangs under the side views
        var better = ms.rmse < baseMs.rmse, fb = tiny ? 22 : KT.clamp(sp * 0.105, 28, 56), yb = y + sp + (tiny ? 6 : 16) + fb;
        var nd = ms.rmse < 0.0995 && baseMs.rmse < 0.0995 ? 2 : 1;
        txt(pct(ms.rmse, nd), xa, yb, font(600, fb, F_DISP), better ? C.ok : C.text, 'left', '-0.01em');
        txt(pct(baseMs.rmse, nd), xb, yb, font(600, fb, F_DISP), better ? C.err : C.text, 'left', '-0.01em');
        var yl = tiny ? yb : yb + fs + 9, pts = thousands(m.nu * m.nv);
        if (!tiny) {
          fitTxt('RMSE · ' + pts + ' CONTROL POINTS', xa, yl, 500, fs, C.dim, 'left', '0.1em', sp);
          fitTxt('RMSE · ' + pts + ' CONTROL POINTS', xb, yl, 500, fs, C.dim, 'left', '0.1em', sp);
        }
        var ratio = baseMs.rmse / ms.rmse, same = Math.abs(ratio - 1) < 0.005;
        var head = same ? 'SAME ERROR · THE TWO NETS COINCIDE'
                 : better ? '×' + ratio.toFixed(1) + ' LOWER ERROR AT EQUAL PARAMETERS' : 'UNIFORM IS ×' + (1 / ratio).toFixed(1) + ' BETTER HERE';
        var ysx = yl + (tiny ? 7 : 16), tw = 2 * sp + L.gap;
        fitTxt(head, xa, ysx + L.f1, 600, L.f1, same ? C.text : better ? C.ok : C.err, 'left', '0.07em', tw);
        if (bestU) {
          var rb = bestU.rmse / ms.rmse;
          flowTxt((tiny ? 'best uniform near ' + pts + ' pts: ' : 'strongest uniform net near ' + pts + ' points: ') +
                  bestU.nu + ' × ' + bestU.nv + ' = ' + bestU.n + ' → ' + pct(bestU.rmse, nd) +
                  (same ? '' : rb >= 1 ? ' · still ×' + rb.toFixed(1) + ' behind' : ' · ×' + (1 / rb).toFixed(1) + ' ahead of adaptive'),
                  xa, ysx + L.f1 + L.f2 + (tiny ? 5 : 9), L.f2, C.dim, tw, L.f2 + 5);
        }
      }

      var ESCALE = 0.30;                       // error heat: full red at 30 % of the relief
      var rdCache = {};
      function setRd(rd, key, s) { if (rdCache[key] !== s) { rdCache[key] = s; rd.set(s); } }
      function updateReadouts() {
        var d = drawnNow();
        setRd(rdNet, 'net', netStr(m));
        setRd(rdPar, 'par', thousands(m.nu * m.nv));
        setRd(rdKn, 'kn', (m.nu - W) + ' u · ' + (m.nv - W) + ' v');
        setRd(rdRmse, 'rmse', pct(ms.rmse, ms.rmse < 0.02 ? 3 : 2));
        setRd(rdMax, 'max', pct(ms.max, ms.max < 0.1 ? 2 : 1));
        setRd(rdG, 'g', st.gauss ? thousands(st.dens * st.dens) : '—');
        setRd(rdDrawn, 'dr', d ? pct(d.rmse, d.rmse < 0.02 ? 3 : 2) : '—');
        if (st.shade === 'normal') { var a = angleNow(); setRd(rdAng, 'ang', a == null ? '—' : a.toFixed(1) + '°'); }
        if (st.versus && baseMs) setRd(rdUni, 'uni', pct(baseMs.rmse, baseMs.rmse < 0.02 ? 3 : 2));
      }

      /* ---------- pointer: drag the cut, click a cell of the balance map ---------- */
      var drag = null;
      function local(e) { var r = cv.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
      function hitCut(x, y, touch) {
        if (!L || st.versus || !(L.ms > 40)) return false;
        var yc = cutY();
        if (touch) return Math.abs(y - yc) < KT.HIT && x > L.mx - KT.HIT && x < L.mx + 12 + KT.HIT;      // a finger takes the grip only
        return Math.abs(y - yc) < 7 && x > L.mx - 4 && x < L.mx + L.ms + 4;
      }
      function moveCut(y) { var v = KT.clamp((y - L.my) / L.ms, 0.02, 0.98); if (v !== st.cut) { st.cut = v; } }
      KT.touchGuard(cv.cv, { hit: function (x, y) { var r = cv.cv.getBoundingClientRect(); return hitCut(x - r.left, y - r.top, true); } });
      cv.cv.addEventListener('pointerdown', function (e) {
        var q = local(e);
        if (!hitCut(q[0], q[1], e.pointerType === 'touch')) return;
        drag = e.pointerId; try { cv.cv.setPointerCapture(e.pointerId); } catch (x) {}
        moveCut(q[1]); e.preventDefault();
      });
      cv.cv.addEventListener('pointermove', function (e) {
        var q = local(e);
        if (drag === e.pointerId) { moveCut(q[1]); return; }
        if (e.pointerType === 'mouse') cv.cv.style.cursor = hitCut(q[0], q[1], false) ? 'ns-resize' : cellAt(q[0], q[1]) ? 'pointer' : '';
      });
      function endDrag(e) { if (drag === e.pointerId) drag = null; }
      cv.cv.addEventListener('pointerup', endDrag); cv.cv.addEventListener('pointercancel', endDrag);
      cv.cv.addEventListener('lostpointercapture', endDrag);
      function cellAt(x, y) {
        var g = L && L.bg; if (!g || !B.cols || st.versus) return null;
        var ci = Math.floor((x - g.x) / g.cell), ri = g.nr - 1 - Math.floor((y - g.y) / g.cell);
        return ci >= 0 && ci < g.nc && ri >= 0 && ri < g.nr ? { ci: ci, ri: ri } : null;
      }
      cv.cv.addEventListener('click', function (e) {
        var q = local(e), c = cellAt(q[0], q[1]);
        if (!c) return;
        st.tour = null; clearRun(); setModel(S.restore(B.cols[c.ci].snap));
        st.inserted = 0; st.pruned = 0; st.ceiling = null; st.blocked = null; st.rmse0 = S.coarseRmse; st.barScale = 0;
        st.densAuto = false; setDens(B.rows[c.ri]); st.preset = 'cell'; st.manual = false; st.view = 'fit'; segMap.set('fit'); setStatus('', '', 'text'); markPreset();
        if (!st.gauss) setGauss(true);
      });

      function draw(t, dt) {
        ctx.clearRect(0, 0, L.W, L.H);
        if (st.versus) drawVersus(t); else drawMain(t, dt);
        updateReadouts();
      }
      env.loop(function (t, dt) {
        if (!m || !L) return;
        var t0 = now(), gapMs = lastFrame ? t0 - lastFrame : 16; lastFrame = t0;
        // the balance map is measured in slices: 8 ms a frame, more when frames are slow anyway
        if (st.bal || st.want) { if (!B.ready) S.balStep(KT.clamp(0.3 * gapMs, 8, 60)); if (st.want && B.cols) applyPreset(st.want); }
        if (st.run) step(dt);
        var lab = !st.run ? 'resume' : playing() ? 'pause' : 'replay';
        if (btRun.textContent !== lab) btRun.textContent = lab;
        t0 = now(); draw(t, dt); perf.frame = now() - t0;
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
          var d = drawnNow();
          return { scene: st.scene, nu: m.nu, nv: m.nv, rmse: ms.rmse, max: ms.max, done: m.cg.done, mode: st.mode,
                   busy: !!st.op, pending: !!st.pending,
                   inserted: st.inserted, pruned: st.pruned, gauss: st.gauss ? st.dens * st.dens : 0,
                   drawn: d ? d.rmse : null, sampling: d ? d.samp : null, regime: regime(), shade: st.shade, cut: st.cut,
                   budget: st.budget, preset: st.preset, tour: !!st.tour, angle: st.shade === 'normal' ? angleNow() : null,
                   balance: { ready: B.ready, done: B.done, ms: B.ms, cols: B.cols ? B.cols.map(function (c) { return c.pts; }) : null, rows: B.rows.slice(),
                              E: B.E ? Array.prototype.slice.call(B.E) : null, nstar: B.nstar },
                   uniform: baseMs ? baseMs.rmse : null, best: bestU, status: [status.l1, status.live ? status.live() : status.l2],
                   layout: wrap.className, arr: L && L.arr, map: L && (L.ms || L.sp), side: L && L.side, bal: L && L.balR, perf: { top: perf.top, frame: perf.frame },
                   U: m.U.slice(), V: m.V.slice() };
        },
        /* for tests: jump to a regime / time a refit and a full redraw of the live state */
        preset: function (k) { userPreset(k); },
        bench: function (reps) {
          var i, t0, out = {}; reps = reps || 5;
          t0 = now(); for (i = 0; i < reps; i++) { m.Z = solveDirect(m); measure(m); } out.refit = (now() - t0) / reps;
          var z0 = new Float64Array(m.Z), k = 0;
          for (i = 0; i < m.Z.length; i++) m.Z[i] = 0.5; cgInit(m);
          t0 = now(); k = cgConverge(m); out.cg = now() - t0; out.cgIters = k;
          m.Z = z0; cgInit(m); ms = measure(m);
          t0 = now(); for (i = 0; i < reps; i++) { zVersion++; dirty = true; gT.key = ''; draw(i, 0.016); } out.redraw = (now() - t0) / reps;
          t0 = now(); for (i = 0; i < reps; i++) draw(i, 0.016); out.idleFrame = (now() - t0) / reps;
          if (m.nu > W || m.nv > W) { t0 = now(); removalCandidates(m); out.removal = now() - t0; }
          out.pts = m.nu * m.nv; out.gauss = st.gauss ? st.dens * st.dens : 0; out.dispRes = disp.n;
          return out;
        }
      };
    }
  });
})(typeof window !== 'undefined' ? window : globalThis);
