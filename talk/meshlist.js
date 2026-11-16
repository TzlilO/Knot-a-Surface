/* Knot-a-Surface talk — a mesh from a LIST of Gaussians, and the same audit for a lattice (KnotTalk.meshList).
   What an unstructured 3DGS store has to do before it has triangles, run for real on the page:
     emulate(parts, seed)  the list: the samples of the charts' lattices, each moved to a random place inside its
                           cell, nudged off the surface along the normal, turned and resized at random, then
                           shuffled. EMULATED from the same surfaces, never trained. Each entry keeps its true
                           chart and lattice cell, used ONLY to grade the result (a list does not know them).
     mesher(list, o)       meshing it the way a list must: a spatial hash and a k-nearest-neighbour search for every
                           Gaussian (every distance computed is counted), a normal from the neighbours (PCA), the
                           neighbours projected to that tangent plane and triangulated there (the local Delaunay
                           star: the Voronoi cell clipped by each neighbour's bisector), and a triangle kept where at
                           least two of its three corners proposed it (consensus of the local triangulations, after
                           Gopi, Krishnan & Silva 2000). Neighbour rules in the spirit of PCL's greedy projection: of
                           the k = 12 nearest, those within mu = 1.6 x the distance to the 6th nearest (a radius that
                           follows the local density without trusting one lucky close pair) and whose normals differ
                           by less than 45 degrees. A job: step(ms) works for about ms
                           milliseconds and returns true when it is finished, run() to the end; the time spent in
                           each phase is measured inside the steps, so a page can spread the work over frames.
     lattice(parts)        the charts' own lattices, stitched: a 2 x 2 window, two triangles per stop, no search.
     audit(...)            one grading for both: edges, boundary edges, non-manifold edges (three or more
                           triangles on one edge), bridges (edges between points that are not neighbours on the
                           surface: on another chart, or across a fold of the same one), and boundary loops beyond
                           the charts' own rims (holes).
   parts = [{ patch, na, nb, chart }]: chart `chart` sampled on an even na x nb lattice (u = (a + 0.5) / na).
   Plain functions; also require()-able under node (tests). Depends on talk/nurbs.js. */
(function (root) {
  'use strict';
  var KT = root.KnotTalk = root.KnotTalk || {};
  var NB = null;            // talk/nurbs.js, looked up when first needed (so this file may load before it)
  function nurbs() { return NB || (NB = KT.nurbs || (typeof require === 'function' ? require('./nurbs.js') : null)); }
  var now = function () { return root.performance && root.performance.now ? root.performance.now() : Date.now(); };

  function rng(seed) {     // the kit's deterministic generator (KnotTalk.rng), here so the file stands alone under node
    var s = seed >>> 0;
    return function () { s = (s + 0x6D2B79F5) | 0; var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function wraps(p) { return { u: !!p.periodicU, v: !!p.periodicV }; }

  /* ---------- a point of a patch at any (u, v): position, both tangents, colour ---------- */
  var Nu = new Float64Array(8), Du = new Float64Array(8), Nv = new Float64Array(8), Dv = new Float64Array(8);
  function evalAt(patch, u, v, o) {
    var p = patch.p, nu = patch.nu, nv = patch.nv, P = patch.P, C = patch.C, r, s, q, w, wu, wv, i, j;
    var B = nurbs(), ku = B.findSpan(patch.mu, p, u, patch.Uk), kv = B.findSpan(patch.mv, p, v, patch.Vk);
    B.basisFuns(ku, u, p, patch.Uk, Nu, Du, 0); B.basisFuns(kv, v, p, patch.Vk, Nv, Dv, 0);
    for (q = 0; q < 12; q++) o[q] = 0;
    for (s = 0; s <= p; s++) {
      j = (kv - p + s) % nv;
      for (r = 0; r <= p; r++) {
        i = (ku - p + r) % nu; q = (j * nu + i) * 3; w = Nu[r] * Nv[s]; wu = Du[r] * Nv[s]; wv = Nu[r] * Dv[s];
        o[0] += w * P[q]; o[1] += w * P[q + 1]; o[2] += w * P[q + 2];
        o[3] += wu * P[q]; o[4] += wu * P[q + 1]; o[5] += wu * P[q + 2];
        o[6] += wv * P[q]; o[7] += wv * P[q + 1]; o[8] += wv * P[q + 2];
        o[9] += w * C[q]; o[10] += w * C[q + 1]; o[11] += w * C[q + 2];
      }
    }
    return o;
  }

  /* ---------- the list ---------- */
  function emulate(parts, seed) {
    var rnd = rng(seed == null ? 7 : seed), n = 0, k, a, b;
    parts.forEach(function (m) { n += m.na * m.nb; });
    var tmp = { pos: new Float32Array(n * 3), col: new Float32Array(n * 3), u: new Float32Array(n * 3), v: new Float32Array(n * 3),
                chart: new Uint16Array(n), part: new Uint16Array(n), ca: new Int32Array(n), cb: new Int32Array(n), sp: new Float32Array(n) };
    var e = new Float64Array(12), t = 0;
    parts.forEach(function (m, pi) {
      var w = wraps(m.patch);
      for (b = 0; b < m.nb; b++) for (a = 0; a < m.na; a++) {
        var u = (a + rnd()) / m.na, v = (b + rnd()) / m.nb;
        u = w.u ? u % 1 : Math.min(u, 0.99999); v = w.v ? v % 1 : Math.min(v, 0.99999);
        evalAt(m.patch, u, v, e);
        // the edges of this sample's cell, its normal, and its size
        var sx = e[3] / m.na, sy = e[4] / m.na, sz = e[5] / m.na, tx = e[6] / m.nb, ty = e[7] / m.nb, tz = e[8] / m.nb;
        var nx = sy * tz - sz * ty, ny = sz * tx - sx * tz, nz = sx * ty - sy * tx, nl = Math.hypot(nx, ny, nz) || 1e-12, sp = Math.sqrt(nl);
        var off = (rnd() - 0.5) * 0.5 * sp, an = rnd() * 6.283185, ca = Math.cos(an), sa = Math.sin(an);
        var f1 = sp * 0.72 * (0.6 + 0.7 * rnd()), f2 = sp * 0.72 * (0.6 + 0.7 * rnd());
        var sl = Math.hypot(sx, sy, sz) || 1e-12, e1x = sx / sl, e1y = sy / sl, e1z = sz / sl;
        nx /= nl; ny /= nl; nz /= nl;
        var e2x = ny * e1z - nz * e1y, e2y = nz * e1x - nx * e1z, e2z = nx * e1y - ny * e1x, o = t * 3;
        tmp.pos[o] = e[0] + nx * off; tmp.pos[o + 1] = e[1] + ny * off; tmp.pos[o + 2] = e[2] + nz * off;
        tmp.col[o] = e[9]; tmp.col[o + 1] = e[10]; tmp.col[o + 2] = e[11];
        tmp.u[o] = (e1x * ca + e2x * sa) * f1; tmp.u[o + 1] = (e1y * ca + e2y * sa) * f1; tmp.u[o + 2] = (e1z * ca + e2z * sa) * f1;
        tmp.v[o] = (e2x * ca - e1x * sa) * f2; tmp.v[o + 1] = (e2y * ca - e1y * sa) * f2; tmp.v[o + 2] = (e2z * ca - e1z * sa) * f2;
        tmp.chart[t] = m.chart; tmp.part[t] = pi; tmp.ca[t] = a; tmp.cb[t] = b; tmp.sp[t] = sp; t++;
      }
    });
    var ord = new Uint32Array(n);                     // no order at all
    for (k = 0; k < n; k++) ord[k] = k;
    for (k = n - 1; k > 0; k--) { var j = Math.floor(rnd() * (k + 1)), s = ord[k]; ord[k] = ord[j]; ord[j] = s; }
    var L = { n: n, pos: new Float32Array(n * 3), col: new Float32Array(n * 3), axU: new Float32Array(n * 3), axV: new Float32Array(n * 3),
              chart: new Uint16Array(n), part: new Uint16Array(n), ca: new Int32Array(n), cb: new Int32Array(n),
              grid: parts.map(function (m) { var w = wraps(m.patch); return { na: m.na, nb: m.nb, wu: w.u, wv: w.v, patch: m.patch }; }) };
    for (k = 0; k < n; k++) {
      var f = ord[k], q;
      for (q = 0; q < 3; q++) { L.pos[k * 3 + q] = tmp.pos[f * 3 + q]; L.col[k * 3 + q] = tmp.col[f * 3 + q]; L.axU[k * 3 + q] = tmp.u[f * 3 + q]; L.axV[k * 3 + q] = tmp.v[f * 3 + q]; }
      L.chart[k] = tmp.chart[f]; L.part[k] = tmp.part[f]; L.ca[k] = tmp.ca[f]; L.cb[k] = tmp.cb[f];
    }
    return L;
  }

  /* ---------- a spatial hash over the points: what a list needs before it can ask "who is near me" ---------- */
  function hashGrid(P, n, cell) {
    var lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30], i, q;
    for (i = 0; i < n; i++) for (q = 0; q < 3; q++) { var x = P[i * 3 + q]; if (x < lo[q]) lo[q] = x; if (x > hi[q]) hi[q] = x; }
    var DX = Math.floor((hi[0] - lo[0]) / cell) + 3, DY = Math.floor((hi[1] - lo[1]) / cell) + 3, DZ = Math.floor((hi[2] - lo[2]) / cell) + 3;
    var RMAX = Math.max(DX, DY, DZ);
    var S = 1; while (S < n * 2) S <<= 1;
    var mask = S - 1, keys = new Float64Array(S).fill(-1), cnt = new Int32Array(S), start = new Int32Array(S), slot = new Int32Array(n), ord = new Int32Array(n);
    function cx(i) { return Math.floor((P[i * 3] - lo[0]) / cell) + 1; }
    function cy(i) { return Math.floor((P[i * 3 + 1] - lo[1]) / cell) + 1; }
    function cz(i) { return Math.floor((P[i * 3 + 2] - lo[2]) / cell) + 1; }
    function find(x, y, z, add) {
      var key = x + DX * (y + DY * z), h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
      while (keys[h] !== -1) { if (keys[h] === key) return h; h = (h + 1) & mask; }
      if (!add) return -1;
      keys[h] = key; return h;
    }
    for (i = 0; i < n; i++) { slot[i] = find(cx(i), cy(i), cz(i), true); cnt[slot[i]]++; }
    var sum = 0;
    for (i = 0; i < S; i++) { start[i] = sum; sum += cnt[i]; }
    var fill = new Int32Array(S);
    for (i = 0; i < n; i++) { ord[start[slot[i]] + fill[slot[i]]++] = i; }
    return {
      tests: 0,
      /* the k nearest points to point i (not i itself), nearest first, into nbI / nbD (squared distances) at off;
         shells of cells around i's cell until the k-th distance is inside what was searched. Returns how many. */
      knn: function (i, k, nbI, nbD, off) {
        var x0 = cx(i), y0 = cy(i), z0 = cz(i), px = P[i * 3], py = P[i * 3 + 1], pz = P[i * 3 + 2], m = 0, r, a, b, c, h, j, t, d, e, s;
        for (r = 1; r <= RMAX; r++) {
          for (c = -r; c <= r; c++) for (b = -r; b <= r; b++) for (a = -r; a <= r; a++) {
            if (r > 1 && Math.max(Math.abs(a), Math.abs(b), Math.abs(c)) !== r) continue;     // shell 1 is the whole 3 x 3 x 3 block
            h = find(x0 + a, y0 + b, z0 + c, false); if (h < 0) continue;
            for (t = start[h], e = t + cnt[h]; t < e; t++) {
              j = ord[t]; if (j === i) continue;
              s = j * 3; d = (P[s] - px) * (P[s] - px) + (P[s + 1] - py) * (P[s + 1] - py) + (P[s + 2] - pz) * (P[s + 2] - pz);
              this.tests++;
              if (m < k) m++; else if (d >= nbD[off + k - 1]) continue;
              var w = m - 1;                                   // insertion into the sorted list
              while (w > 0 && nbD[off + w - 1] > d) { nbD[off + w] = nbD[off + w - 1]; nbI[off + w] = nbI[off + w - 1]; w--; }
              nbD[off + w] = d; nbI[off + w] = j;
            }
          }
          if (m >= k && nbD[off + k - 1] <= r * cell * r * cell) break;
        }
        return m;
      }
    };
  }

  /* the eigenvector of the smallest eigenvalue of a symmetric 3 x 3 matrix (the normal of a point cloud's
     covariance), into out[0..2] */
  function smallest(a00, a01, a02, a11, a12, a22, out) {
    var p1 = a01 * a01 + a02 * a02 + a12 * a12, q = (a00 + a11 + a22) / 3, b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
    var p = Math.sqrt((b00 * b00 + b11 * b11 + b22 * b22 + 2 * p1) / 6), lam;
    if (p < 1e-30) { out[0] = 0; out[1] = 0; out[2] = 1; return; }
    var c00 = b00 / p, c11 = b11 / p, c22 = b22 / p, c01 = a01 / p, c02 = a02 / p, c12 = a12 / p;
    var r = (c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02)) / 2;
    lam = q + 2 * p * Math.cos(Math.acos(r < -1 ? -1 : r > 1 ? 1 : r) / 3 + 2 * Math.PI / 3);
    var r0x = a00 - lam, r0y = a01, r0z = a02, r1x = a01, r1y = a11 - lam, r1z = a12, r2x = a02, r2y = a12, r2z = a22 - lam;
    var ax = r0y * r1z - r0z * r1y, ay = r0z * r1x - r0x * r1z, az = r0x * r1y - r0y * r1x;
    var bx = r0y * r2z - r0z * r2y, by = r0z * r2x - r0x * r2z, bz = r0x * r2y - r0y * r2x;
    var cx = r1y * r2z - r1z * r2y, cy = r1z * r2x - r1x * r2z, cz = r1x * r2y - r1y * r2x;
    var la = ax * ax + ay * ay + az * az, lb = bx * bx + by * by + bz * bz, lc = cx * cx + cy * cy + cz * cz;
    if (lb > la && lb >= lc) { ax = bx; ay = by; az = bz; la = lb; } else if (lc > la) { ax = cx; ay = cy; az = cz; la = lc; }
    la = Math.sqrt(la) || 1;
    out[0] = ax / la; out[1] = ay / la; out[2] = az / la;
  }

  /* ---------- meshing a list ---------- */
  function mesher(L, o) {
    o = o || {};
    var n = L.n, P = L.pos, K = o.k || 12, MU = o.mu || 1.6, RI = o.ri == null ? 5 : o.ri, DEV = Math.cos((o.dev == null ? 45 : o.dev) * Math.PI / 180), KEEP = o.keep || 2;
    var nbI = new Int32Array(n * K), nbD = new Float32Array(n * K), nbN = new Uint8Array(n), NRM = new Float32Array(n * 3);
    var star = new Int32Array(Math.max(1024, n * 14)), sStart = new Int32Array(n + 1), sLen = 0, grid = null;
    var job = { n: n, phase: 'index', done: false, i: 0, tests: 0, ms: { index: 0, search: 0, mesh: 0, audit: 0 }, tri: null, nTri: 0, stats: null };
    var X = new Float64Array(K + 1), Y = new Float64Array(K + 1), J = new Int32Array(K + 1);
    var PX = new Float64Array(2 * K + 16), PY = new Float64Array(2 * K + 16), PL = new Int32Array(2 * K + 16);
    var QX = new Float64Array(2 * K + 16), QY = new Float64Array(2 * K + 16), QL = new Int32Array(2 * K + 16), nv3 = [0, 0, 0];

    function index() {                  // the cell: about the size the list's own Gaussians say the gaps are
      var s = 0, i, m = Math.min(n, 4096);
      for (i = 0; i < m; i++) {
        var k = Math.floor(i * n / m) * 3;
        s += Math.hypot(L.axU[k], L.axU[k + 1], L.axU[k + 2]) + Math.hypot(L.axV[k], L.axV[k + 1], L.axV[k + 2]);
      }
      job.cell = (o.cellK || 2) * s / (2 * m * 0.72 * 0.95);
      grid = hashGrid(P, n, job.cell);
    }
    function search(i) { nbN[i] = grid.knn(i, K, nbI, nbD, i * K); }
    function normal(i) {
      var m = nbN[i], s, k, q, mx = P[i * 3], my = P[i * 3 + 1], mz = P[i * 3 + 2];
      for (k = 0; k < m; k++) { s = nbI[i * K + k] * 3; mx += P[s]; my += P[s + 1]; mz += P[s + 2]; }
      mx /= m + 1; my /= m + 1; mz /= m + 1;
      var c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0, dx, dy, dz;
      for (k = -1; k < m; k++) {
        s = (k < 0 ? i : nbI[i * K + k]) * 3; dx = P[s] - mx; dy = P[s + 1] - my; dz = P[s + 2] - mz;
        c00 += dx * dx; c01 += dx * dy; c02 += dx * dz; c11 += dy * dy; c12 += dy * dz; c22 += dz * dz;
      }
      smallest(c00, c01, c02, c11, c12, c22, nv3);
      for (q = 0; q < 3; q++) NRM[i * 3 + q] = nv3[q];
    }
    /* the local Delaunay star of point i in its tangent plane: clip a square around i by the bisector of each
       accepted neighbour; the neighbours that keep an edge of the cell are i's Delaunay neighbours, and two
       consecutive ones make a triangle with i. An edge of the square left over means a gap there: no triangle. */
    function fan(i) {
      var m = nbN[i], o3 = i * 3, nx = NRM[o3], ny = NRM[o3 + 1], nz = NRM[o3 + 2], k, j, s, c = 0;
      sStart[i] = sLen;
      if (m < 2) return;
      var ax = Math.abs(nx) < 0.9 ? 1 : 0, ay = ax ? 0 : 1;      // a tangent basis e1, e2
      var e1x = ay * nz - 0 * ny, e1y = 0 * nx - ax * nz, e1z = ax * ny - ay * nx, l1 = Math.hypot(e1x, e1y, e1z);
      e1x /= l1; e1y /= l1; e1z /= l1;
      var e2x = ny * e1z - nz * e1y, e2y = nz * e1x - nx * e1z, e2z = nx * e1y - ny * e1x;
      var ri = Math.min(m - 1, RI), r2 = MU * MU * nbD[i * K + ri], R = Math.max(1e-9, MU * Math.sqrt(nbD[i * K + ri]));
      for (k = 0; k < m; k++) {
        if (nbD[i * K + k] > r2) break;
        j = nbI[i * K + k]; s = j * 3;
        if (Math.abs(nx * NRM[s] + ny * NRM[s + 1] + nz * NRM[s + 2]) < DEV) continue;    // the surface turns too much in between
        var dx = P[s] - P[o3], dy = P[s + 1] - P[o3 + 1], dz = P[s + 2] - P[o3 + 2];
        X[c] = dx * e1x + dy * e1y + dz * e1z; Y[c] = dx * e2x + dy * e2y + dz * e2z; J[c] = j; c++;
      }
      if (c < 2) return;
      var np = 4, q, t;
      PX[0] = -R; PY[0] = -R; PX[1] = R; PY[1] = -R; PX[2] = R; PY[2] = R; PX[3] = -R; PY[3] = R; PL[0] = PL[1] = PL[2] = PL[3] = -1;
      for (k = 0; k < c; k++) {                                  // the half-plane x . q <= |q|^2 / 2, edge label k
        var qx = X[k], qy = Y[k], h = (qx * qx + qy * qy) / 2, nq = 0;
        if (h < 1e-30) continue;
        for (q = 0; q < np; q++) {
          var r = (q + 1) % np, fa = PX[q] * qx + PY[q] * qy - h, fb = PX[r] * qx + PY[r] * qy - h;
          if (fa <= 0) { QX[nq] = PX[q]; QY[nq] = PY[q]; QL[nq++] = PL[q]; }
          if ((fa <= 0) !== (fb <= 0)) {
            t = fa / (fa - fb);
            QX[nq] = PX[q] + (PX[r] - PX[q]) * t; QY[nq] = PY[q] + (PY[r] - PY[q]) * t; QL[nq++] = fa <= 0 ? k : PL[q];
          }
        }
        if (nq < 3) { np = 0; break; }
        for (q = 0; q < nq; q++) { PX[q] = QX[q]; PY[q] = QY[q]; PL[q] = QL[q]; }
        np = nq;
      }
      if (sLen + np * 2 > star.length) { var big = new Int32Array(star.length * 2); big.set(star); star = big; }
      for (q = 0; q < np; q++) {
        var a = PL[q], b = PL[(q + 1) % np];
        if (a >= 0 && b >= 0 && a !== b) { star[sLen++] = J[a]; star[sLen++] = J[b]; }
      }
    }
    function has(x, a, b) {             // does point x's star hold the triangle (x, a, b)?
      for (var t = sStart[x], e = sStart[x + 1]; t < e; t += 2) if ((star[t] === a && star[t + 1] === b) || (star[t] === b && star[t + 1] === a)) return true;
      return false;
    }
    function vote() {
      var out = new Uint32Array(Math.max(3, sLen * 3 / 2)), m = 0, i, t, a, b;
      for (i = 0; i < n; i++) for (t = sStart[i]; t < sStart[i + 1]; t += 2) {
        a = star[t]; b = star[t + 1];
        var va = has(a, i, b), vb = has(b, i, a);
        if (1 + va + vb < KEEP) continue;
        if ((va && a < i) || (vb && b < i)) continue;          // listed once: by the first of its proposers
        out[m * 3] = i; out[m * 3 + 1] = a; out[m * 3 + 2] = b; m++;
      }
      job.tri = out.subarray(0, m * 3); job.nTri = m;
    }
    var PH = ['index', 'search', 'normal', 'fan', 'vote', 'audit'];
    job.step = function (budget) {
      var t0 = now(), tEnd = t0 + (budget == null ? 1e9 : budget);
      while (!job.done) {
        var a = now(), ph = job.phase, i, e;
        if (ph === 'index') { index(); job.ms.index += now() - a; job.phase = 'search'; job.i = 0; }
        else if (ph === 'vote') { sStart[n] = sLen; vote(); job.ms.mesh += now() - a; job.phase = 'audit'; }
        else if (ph === 'audit') {
          job.stats = audit(job.tri, job.nTri, L);
          job.ms.audit += now() - a; job.done = true; job.phase = 'done'; job.progress = 1;
        } else {
          for (i = job.i, e = Math.min(n, i + 512); i < e; i++) (ph === 'search' ? search : ph === 'normal' ? normal : fan)(i);
          job.i = e;
          if (ph === 'search') { job.ms.search += now() - a; job.tests = grid.tests; } else job.ms.mesh += now() - a;
          if (e === n) { job.phase = PH[PH.indexOf(ph) + 1]; job.i = 0; }
        }
        if (!job.done) job.progress = Math.min(0.999, Math.max(0, (PH.indexOf(job.phase) - 1 + job.i / n) / 4));
        if (now() >= tEnd) break;
      }
      return job.done;
    };
    job.run = function () { while (!job.step(1e9)); return job; };
    job.normals = NRM;
    return job;
  }

  /* ---------- the charts' own lattices, stitched ----------
     parts in the window's order; vertex (a, b) of part m is m.off + b * na + a (offsets assigned here). Returns the
     triangles (two per window stop, in the window's order), three edges per stop for drawing, and per vertex its chart
     and lattice cell (for the audit). No search: the neighbours are the next addresses. */
  function lattice(parts) {
    var t0 = now(), V = 0, Q = 0, a, b, t = 0, e = 0;
    parts.forEach(function (m) {
      var w = wraps(m.patch); m.wu = w.u; m.wv = w.v;
      m.off = V; m.q0 = Q; m.qa = w.u ? m.na : m.na - 1; m.qb = w.v ? m.nb : m.nb - 1;
      V += m.na * m.nb; Q += Math.max(0, m.qa) * Math.max(0, m.qb);
    });
    var tri = new Uint32Array(Q * 6), edge = new Uint32Array(Q * 6), chart = new Uint16Array(V), part = new Uint16Array(V), ca = new Int32Array(V), cb = new Int32Array(V);
    parts.forEach(function (m, pi) {
      for (b = 0; b < m.nb; b++) for (a = 0; a < m.na; a++) { var v = m.off + b * m.na + a; chart[v] = m.chart; part[v] = pi; ca[v] = a; cb[v] = b; }
      for (b = 0; b < m.qb; b++) for (a = 0; a < m.qa; a++) {
        var v00 = m.off + b * m.na + a, v10 = m.off + b * m.na + (a + 1) % m.na, v01 = m.off + ((b + 1) % m.nb) * m.na + a, v11 = m.off + ((b + 1) % m.nb) * m.na + (a + 1) % m.na;
        tri[t++] = v00; tri[t++] = v10; tri[t++] = v11; tri[t++] = v00; tri[t++] = v11; tri[t++] = v01;
        edge[e++] = v00; edge[e++] = v10; edge[e++] = v00; edge[e++] = v01; edge[e++] = v00; edge[e++] = v11;
      }
    });
    return { n: V, V: V, Q: Q, tri: tri, nTri: Q * 2, edge: edge, chart: chart, part: part, ca: ca, cb: cb,
             grid: parts.map(function (m) { return { na: m.na, nb: m.nb, wu: m.wu, wv: m.wv, patch: m.patch }; }), ms: now() - t0 };
  }

  /* ---------- one audit for both ----------
     tri: nTri triangles over the points of M = { n, chart, part, ca, cb, grid }: per point its true chart, the part
     (lattice) it was sampled from and its cell (ca, cb) there; grid[part] = { na, nb, wu, wv, patch }. Edges are
     counted by how many triangles hold them. A BRIDGE joins two points that are not neighbours on the surface:
     on two charts, or on one chart but at least twice as far apart along it as through the air (the surface path
     is measured along the straight line in (u, v) between the two cells; pairs within two cells of each other are
     neighbours). A HOLE is a loop of boundary edges (edges on one triangle only) that does not reach the chart's own
     rim (the outermost two rows of cells): a sheet's border is not a hole, and a gap that runs into it is not counted.
     Returns the counts and, for drawing, edges as (a, b, first triangle holding it), in the order of that triangle:
     bridges and non-manifold edges in `bad`, the edges around holes in `hole`. */
  var ea = new Float64Array(12);
  function along(g, a0, b0, a1, b1) {        // [path along the surface, chord] between the centres of two cells of one lattice
    if (g.wu && Math.abs(a1 - a0) > g.na / 2) a1 += a1 < a0 ? g.na : -g.na;
    if (g.wv && Math.abs(b1 - b0) > g.nb / 2) b1 += b1 < b0 ? g.nb : -g.nb;
    var len = 0, x = 0, y = 0, z = 0, x0 = 0, y0 = 0, z0 = 0, k, S = 12;
    for (k = 0; k <= S; k++) {
      var u = (a0 + (a1 - a0) * k / S + 0.5) / g.na, v = (b0 + (b1 - b0) * k / S + 0.5) / g.nb;
      u = g.wu ? ((u % 1) + 1) % 1 : Math.min(Math.max(u, 0), 0.99999); v = g.wv ? ((v % 1) + 1) % 1 : Math.min(Math.max(v, 0), 0.99999);
      evalAt(g.patch, u, v, ea);
      if (k) len += Math.hypot(ea[0] - x, ea[1] - y, ea[2] - z); else { x0 = ea[0]; y0 = ea[1]; z0 = ea[2]; }
      x = ea[0]; y = ea[1]; z = ea[2];
    }
    return [len, Math.hypot(x - x0, y - y0, z - z0)];
  }
  function audit(tri, nTri, M) {
    var nV = M.n, chart = M.chart, part = M.part, ca = M.ca, cb = M.cb, grid = M.grid;
    var S = 1; while (S < nTri * 3) S <<= 1;
    var mask = S - 1, K1 = new Int32Array(S).fill(-1), K2 = new Int32Array(S), CNT = new Int32Array(S), FIRST = new Int32Array(S), used = [];
    function add(a, b, t) {
      if (a > b) { var s = a; a = b; b = s; }
      var h = (Math.imul(a, 0x9E3779B1) ^ Math.imul(b, 0x85EBCA77)) & mask;
      while (K1[h] !== -1) { if (K1[h] === a && K2[h] === b) { CNT[h]++; return; } h = (h + 1) & mask; }
      K1[h] = a; K2[h] = b; CNT[h] = 1; FIRST[h] = t; used.push(h);
    }
    var t, i;
    for (t = 0; t < nTri; t++) { add(tri[t * 3], tri[t * 3 + 1], t); add(tri[t * 3 + 1], tri[t * 3 + 2], t); add(tri[t * 3 + 2], tri[t * 3], t); }
    function far(a, b) {
      if (chart[a] !== chart[b] || part[a] !== part[b]) return true;
      var g = grid[part[a]], da = Math.abs(ca[a] - ca[b]), db = Math.abs(cb[a] - cb[b]);
      if (g.wu) da = Math.min(da, g.na - da);
      if (g.wv) db = Math.min(db, g.nb - db);
      if (Math.max(da, db) <= 2) return false;
      var w = along(g, ca[a], cb[a], ca[b], cb[b]);
      return w[0] > 2 * w[1];
    }
    var par = new Int32Array(nV).fill(-1), rimE = [];
    function root(x) { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    function join(a, b) { if (par[a] < 0) par[a] = a; if (par[b] < 0) par[b] = b; a = root(a); b = root(b); if (a !== b) par[a] = b; }
    function border(x) {
      var g = grid[part[x]];
      return (!g.wu && (ca[x] <= 1 || ca[x] >= g.na - 2)) || (!g.wv && (cb[x] <= 1 || cb[x] >= g.nb - 2));
    }
    var r = { points: nV, triangles: nTri, edges: used.length, boundary: 0, nonManifold: 0, bridges: 0, loops: 0, holes: 0, bad: [], hole: [] };
    for (i = 0; i < used.length; i++) {
      var h = used[i], a = K1[h], b = K2[h], c = CNT[h], br = far(a, b);
      if (c === 1) { r.boundary++; join(a, b); rimE.push(a, b, FIRST[h]); }
      if (c >= 3) r.nonManifold++;
      if (br) r.bridges++;
      if (br || c >= 3) r.bad.push(a, b, FIRST[h]);
    }
    var open = new Uint8Array(nV);            // 1: this loop reaches the rim
    for (i = 0; i < nV; i++) if (par[i] >= 0) { var q = root(i); if (border(i)) open[q] = 1; }
    for (i = 0; i < nV; i++) if (par[i] === i) { r.loops++; if (!open[i]) r.holes++; }
    for (i = 0; i < rimE.length; i += 3) if (!open[root(rimE[i])]) r.hole.push(rimE[i], rimE[i + 1], rimE[i + 2]);
    // the points that ended up in no triangle at all
    var inT = new Uint8Array(nV);
    for (t = 0; t < nTri * 3; t++) inT[tri[t]] = 1;
    r.unused = 0; for (i = 0; i < nV; i++) if (!inT[i]) r.unused++;
    return r;
  }

  var api = { emulate: emulate, mesher: mesher, lattice: lattice, audit: audit, evalAt: evalAt };
  KT.meshList = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
