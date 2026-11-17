/* Knot-a-Surface talk — a mesh from a LIST of Gaussians, and the same audit for a lattice (KnotTalk.meshList).
   What an unstructured 3DGS store has to do before it has triangles, run for real on the page:
     emulator(parts, seed)  the list: the samples of the charts' lattices, each moved to a random place inside its
                            cell, nudged off the surface along the normal, turned and resized at random, then
                            shuffled. EMULATED from the same surfaces, never trained. Each entry keeps its true
                            chart and lattice cell, used ONLY to grade the result (a list does not know them).
     mesher(list, o)        meshing it the way a list must: a spatial hash and a k-nearest-neighbour search for every
                            Gaussian (every distance computed is counted), a normal from the neighbours (PCA), the
                            neighbours projected to that tangent plane and triangulated there (the local Delaunay
                            star: the Voronoi cell clipped by each neighbour's bisector), and a triangle kept where at
                            least two of its three corners proposed it (consensus of the local triangulations, after
                            Gopi, Krishnan & Silva 2000). Neighbour rules in the spirit of PCL's greedy projection: of
                            the k = 12 nearest, those within mu = 1.6 x the distance to the 6th nearest (a radius that
                            follows the local density without trusting one lucky close pair) and whose normals differ
                            by less than 45 degrees. The hash cell is twice the gap the list's own Gaussians suggest
                            (their sizes), the fastest of the sizes tried.
     lattice(parts)         the charts' own lattices, stitched: a 2 x 2 window, two triangles per stop, no search.
     auditor(tri, nTri, M)  one grading for both: edges, boundary edges, non-manifold edges (three or more triangles
                            on one edge), bridges (edges between points that are not neighbours on the surface: on
                            another chart, or across a fold or groove of the same one) and holes.
     addresser(P, n, S)     addresses for a mesh whose points are not the samples (TSDF fusion, talk/meshtsdf.js): the
                            address of the nearest lattice sample, through a hash grid; the audit then grades it too.
     gauge(P, n, M)         how far a mesh's points are from the surface: each projected onto the chart of its address.
   emulator, mesher, auditor, addresser and gauge are JOBS: step(ms) works for about ms milliseconds and returns true
   when finished, run() goes to the end. Each measures the time spent inside its steps, so a page can spread the work
   over frames and still report what it cost. emulate(...) and audit(...) are the same, run at once.
   parts = [{ patch, na, nb, chart }]: chart `chart` sampled on an even na x nb lattice (u = (a + 0.5) / na).
   Plain functions; also require()-able under node (tests). Depends on talk/nurbs.js. */
(function (root) {
  'use strict';
  var KT = root.KnotTalk = root.KnotTalk || {};
  var NB = null;            // talk/nurbs.js, looked up when first needed (so this file may load before it)
  function nurbs() { return NB || (NB = KT.nurbs || (typeof require === 'function' ? require('./nurbs.js') : null)); }
  var now = function () { return root.performance && root.performance.now ? root.performance.now() : Date.now(); };

  function rng(seed) {      // the kit's deterministic generator (KnotTalk.rng), here so the file stands alone under node
    var s = seed >>> 0;
    return function () { s = (s + 0x6D2B79F5) | 0; var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function wraps(p) { return { u: !!p.periodicU, v: !!p.periodicV }; }
  /* a job from a list of phases: each phase is fn(budgetEnd) -> true when that phase is finished */
  function job(J, phases) {
    var k = 0;
    J.done = false; J.ms = J.ms || {};
    J.step = function (budget) {
      var end = now() + (budget == null ? 1e9 : budget);
      while (!J.done) {
        var ph = phases[k], a = now(), fin = ph.fn(end);
        J.ms[ph.name] = (J.ms[ph.name] || 0) + now() - a;
        if (fin && ++k === phases.length) J.done = true;
        J.phase = J.done ? 'done' : phases[k].name;
        if (now() >= end) break;
      }
      return J.done;
    };
    J.run = function () { while (!J.step(1e9)); return J; };
    J.phase = phases[0].name;
    return J;
  }

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
  function emulator(parts, seed) {
    var rnd = rng(seed == null ? 7 : seed), n = 0, t = 0, pi = 0, a = 0, b = 0, e = new Float64Array(12);
    parts.forEach(function (m) { n += m.na * m.nb; });
    var tmp = { pos: new Float32Array(n * 3), col: new Float32Array(n * 3), u: new Float32Array(n * 3), v: new Float32Array(n * 3),
                chart: new Uint16Array(n), part: new Uint16Array(n), ca: new Int32Array(n), cb: new Int32Array(n) };
    var J = { n: n, list: null, progress: 0 };
    function one(m) {
      var w = wraps(m.patch), u = (a + rnd()) / m.na, v = (b + rnd()) / m.nb;
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
      tmp.chart[t] = m.chart; tmp.part[t] = pi; tmp.ca[t] = a; tmp.cb[t] = b; t++;
    }
    return job(J, [
      { name: 'sample', fn: function (end) {
        while (pi < parts.length) {
          var m = parts[pi];
          for (var c = 0; c < 2048 && pi < parts.length; c++) { one(m); if (++a === m.na) { a = 0; if (++b === m.nb) { b = 0; m = parts[++pi]; } } }
          J.progress = 0.9 * t / Math.max(1, n);
          if (now() >= end) return pi >= parts.length;
        }
        return true;
      } },
      { name: 'shuffle', fn: function () {
        var ord = new Uint32Array(n), k, q;               // no order at all
        for (k = 0; k < n; k++) ord[k] = k;
        for (k = n - 1; k > 0; k--) { var j = Math.floor(rnd() * (k + 1)), s = ord[k]; ord[k] = ord[j]; ord[j] = s; }
        var L = { n: n, pos: new Float32Array(n * 3), col: new Float32Array(n * 3), axU: new Float32Array(n * 3), axV: new Float32Array(n * 3),
                  chart: new Uint16Array(n), part: new Uint16Array(n), ca: new Int32Array(n), cb: new Int32Array(n),
                  grid: parts.map(function (m) { var w = wraps(m.patch); return { na: m.na, nb: m.nb, wu: w.u, wv: w.v, patch: m.patch }; }) };
        for (k = 0; k < n; k++) {
          var f = ord[k];
          for (q = 0; q < 3; q++) { L.pos[k * 3 + q] = tmp.pos[f * 3 + q]; L.col[k * 3 + q] = tmp.col[f * 3 + q]; L.axU[k * 3 + q] = tmp.u[f * 3 + q]; L.axV[k * 3 + q] = tmp.v[f * 3 + q]; }
          L.chart[k] = tmp.chart[f]; L.part[k] = tmp.part[f]; L.ca[k] = tmp.ca[f]; L.cb[k] = tmp.cb[f];
        }
        J.list = L; J.progress = 1; tmp = null;
        return true;
      } }
    ]);
  }
  function emulate(parts, seed) { return emulator(parts, seed).run().list; }

  /* ---------- a spatial hash over the points: what a list needs before it can ask "who is near me" ---------- */
  function hashGrid(P, n, cell) {
    var lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30], i, q;
    for (i = 0; i < n; i++) for (q = 0; q < 3; q++) { var x = P[i * 3 + q]; if (x < lo[q]) lo[q] = x; if (x > hi[q]) hi[q] = x; }
    var DX = Math.floor((hi[0] - lo[0]) / cell) + 3, DY = Math.floor((hi[1] - lo[1]) / cell) + 3, DZ = Math.floor((hi[2] - lo[2]) / cell) + 3;
    var RMAX = Math.max(DX, DY, DZ), S = 1;
    while (S < n * 2) S <<= 1;
    var mask = S - 1, keys = new Float64Array(S).fill(-1), cnt = new Int32Array(S), start = new Int32Array(S), slot = new Int32Array(n), ord = new Int32Array(n);
    function cx(i) { return Math.floor((P[i * 3] - lo[0]) / cell) + 1; }
    function cy(i) { return Math.floor((P[i * 3 + 1] - lo[1]) / cell) + 1; }
    function cz(i) { return Math.floor((P[i * 3 + 2] - lo[2]) / cell) + 1; }
    function find(x, y, z, add) {
      if (x < 0 || y < 0 || z < 0 || x >= DX || y >= DY || z >= DZ) return -1;      // outside the grid: no cell (and no other cell's key)
      var key = x + DX * (y + DY * z), h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
      while (keys[h] !== -1) { if (keys[h] === key) return h; h = (h + 1) & mask; }
      if (!add) return -1;
      keys[h] = key; return h;
    }
    for (i = 0; i < n; i++) { slot[i] = find(cx(i), cy(i), cz(i), true); cnt[slot[i]]++; }
    var sum = 0, fill = new Int32Array(S);
    for (i = 0; i < S; i++) { start[i] = sum; sum += cnt[i]; }
    for (i = 0; i < n; i++) { ord[start[slot[i]] + fill[slot[i]]++] = i; }
    return {
      tests: 0,
      bytes: keys.byteLength + cnt.byteLength + start.byteLength + fill.byteLength + slot.byteLength + ord.byteLength,
      /* the k nearest points to point i (not i itself), nearest first, into nbI / nbD (squared distances) at off;
         shells of cells around i's cell until the k-th distance is inside what was searched. Returns how many. */
      knn: function (i, k, nbI, nbD, off) { return this.knnAt(P[i * 3], P[i * 3 + 1], P[i * 3 + 2], k, nbI, nbD, off, i); },
      /* the same from any position (skip: a point to leave out, or -1). A position outside the points' box starts at
         the nearest cell of the box and searches that much further */
      knnAt: function (px, py, pz, k, nbI, nbD, off, skip) {
        var x0 = Math.floor((px - lo[0]) / cell) + 1, y0 = Math.floor((py - lo[1]) / cell) + 1, z0 = Math.floor((pz - lo[2]) / cell) + 1, out = 0, m = 0, r, a, b, c, h, j, t, d, e, s, w;
        if (x0 < 1 || y0 < 1 || z0 < 1 || x0 > DX - 2 || y0 > DY - 2 || z0 > DZ - 2) {
          x0 = Math.min(Math.max(x0, 1), DX - 2); y0 = Math.min(Math.max(y0, 1), DY - 2); z0 = Math.min(Math.max(z0, 1), DZ - 2);
          a = lo[0] + (x0 - 1) * cell; b = lo[1] + (y0 - 1) * cell; c = lo[2] + (z0 - 1) * cell;        // how far outside that cell
          out = Math.hypot(Math.max(a - px, 0, px - a - cell), Math.max(b - py, 0, py - b - cell), Math.max(c - pz, 0, pz - c - cell));
        }
        for (r = 1; r <= RMAX; r++) {
          for (c = -r; c <= r; c++) for (b = -r; b <= r; b++) for (a = -r; a <= r; a++) {
            if (r > 1 && Math.max(Math.abs(a), Math.abs(b), Math.abs(c)) !== r) continue;     // shell 1 is the whole 3 x 3 x 3 block
            h = find(x0 + a, y0 + b, z0 + c, false); if (h < 0) continue;
            for (t = start[h], e = t + cnt[h]; t < e; t++) {
              j = ord[t]; if (j === skip) continue;
              s = j * 3; d = (P[s] - px) * (P[s] - px) + (P[s + 1] - py) * (P[s + 1] - py) + (P[s + 2] - pz) * (P[s + 2] - pz);
              this.tests++;
              if (m < k) m++; else if (d >= nbD[off + k - 1]) continue;
              w = m - 1;                                       // insertion into the sorted list
              while (w > 0 && nbD[off + w - 1] > d) { nbD[off + w] = nbD[off + w - 1]; nbI[off + w] = nbI[off + w - 1]; w--; }
              nbD[off + w] = d; nbI[off + w] = j;
            }
          }
          if (m >= k && (out ? Math.sqrt(nbD[off + k - 1]) <= r * cell - out : nbD[off + k - 1] <= r * cell * r * cell)) break;
        }
        return m;
      }
    };
  }
  /* the gap between neighbouring Gaussians that a list's own Gaussians suggest: their mean axis, over what the emulator
     makes of a gap (0.72 x 0.6..1.3) */
  function gap(L) {
    var s = 0, k, c = Math.min(L.n, 4096);
    for (k = 0; k < c; k++) { var q = Math.floor(k * L.n / c) * 3; s += Math.hypot(L.axU[q], L.axU[q + 1], L.axU[q + 2]) + Math.hypot(L.axV[q], L.axV[q + 1], L.axV[q + 2]); }
    return s / (2 * c * 0.72 * 0.95);
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

  /* ---------- meshing a list ----------
     J.tests = distances computed; J.ms = { index, search, normal, fan, vote, audit } in ms; J.tri / J.nTri = the kept
     triangles, each listed once, by the first Gaussian (in list order) that proposed it, in that order; J.stats = the audit.
     J.bytes = { work, mesh }: the bytes of the arrays it allocated (the hash grid, the neighbours and their distances,
     the normals, the stars, the buffer the triangles are collected in) and of the triangles it kept (12 each); the
     vertices are the list's own Gaussians. As implemented, with N Gaussians and k neighbours each:
       time    O(N k) expected: a hash grid whose cells hold a few Gaussians each, so a search looks at a number of
               candidates that grows with k and not with N (J.tests counts them); then, per Gaussian, a covariance over
               k neighbours, up to k cuts of a polygon that keeps about six corners, and a vote over its six triangles.
               (Worst cases: O(k^2) per Gaussian for the cuts; O(N) per search if the Gaussians crowd into one cell.
               A k-d tree in place of the grid searches in O(N log N).)
       memory  O(N k): 8 k bytes of neighbours a Gaussian, the stars (about 14 entries each), the grid and normals. */
  function mesher(L, o) {
    o = o || {};
    var n = L.n, P = L.pos, K = o.k || 12, MU = o.mu || 1.6, RI = o.ri == null ? 5 : o.ri, DEV = Math.cos((o.dev == null ? 45 : o.dev) * Math.PI / 180), KEEP = o.keep || 2;
    var nbI = new Int32Array(n * K), nbD = new Float32Array(n * K), nbN = new Uint8Array(n), NRM = new Float32Array(n * 3);
    var star = new Int32Array(Math.max(1024, n * 14)), sStart = new Int32Array(n + 1), sLen = 0, grid = null, i = 0, out = null, m = 0, aud = null;
    var J = { n: n, k: K, tests: 0, tri: null, nTri: 0, stats: null, progress: 0, cell: 0, normals: NRM, bytes: null };
    var X = new Float64Array(K + 1), Y = new Float64Array(K + 1), Jx = new Int32Array(K + 1);
    var PX = new Float64Array(2 * K + 16), PY = new Float64Array(2 * K + 16), PL = new Int32Array(2 * K + 16);
    var QX = new Float64Array(2 * K + 16), QY = new Float64Array(2 * K + 16), QL = new Int32Array(2 * K + 16), nv3 = [0, 0, 0];

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
      var e1x = ay * nz, e1y = -ax * nz, e1z = ax * ny - ay * nx, l1 = Math.hypot(e1x, e1y, e1z);
      e1x /= l1; e1y /= l1; e1z /= l1;
      var e2x = ny * e1z - nz * e1y, e2y = nz * e1x - nx * e1z, e2z = nx * e1y - ny * e1x;
      var ri = Math.min(m - 1, RI), r2 = MU * MU * nbD[i * K + ri], R = Math.max(1e-9, MU * Math.sqrt(nbD[i * K + ri]));
      for (k = 0; k < m; k++) {
        if (nbD[i * K + k] > r2) break;
        j = nbI[i * K + k]; s = j * 3;
        if (Math.abs(nx * NRM[s] + ny * NRM[s + 1] + nz * NRM[s + 2]) < DEV) continue;    // the surface turns too much in between
        var dx = P[s] - P[o3], dy = P[s + 1] - P[o3 + 1], dz = P[s + 2] - P[o3 + 2];
        X[c] = dx * e1x + dy * e1y + dz * e1z; Y[c] = dx * e2x + dy * e2y + dz * e2z; Jx[c] = j; c++;
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
        if (a >= 0 && b >= 0 && a !== b) { star[sLen++] = Jx[a]; star[sLen++] = Jx[b]; }
      }
    }
    function has(x, a, b) {             // does point x's star hold the triangle (x, a, b)?
      for (var t = sStart[x], e = sStart[x + 1]; t < e; t += 2) if ((star[t] === a && star[t + 1] === b) || (star[t] === b && star[t + 1] === a)) return true;
      return false;
    }
    function vote(i) {
      for (var t = sStart[i]; t < sStart[i + 1]; t += 2) {
        var a = star[t], b = star[t + 1], va = has(a, i, b), vb = has(b, i, a);
        if (1 + va + vb < KEEP) continue;
        if ((va && a < i) || (vb && b < i)) continue;          // listed once: by the first of its proposers
        out[m * 3] = i; out[m * 3 + 1] = a; out[m * 3 + 2] = b; m++;
      }
    }
    /* one pass over all points in chunks, as a phase */
    function each(fn, w, before) {
      return function (end) {
        if (i === 0 && before) before();
        while (i < n) {
          for (var e = Math.min(n, i + 256); i < e; i++) fn(i);
          J.progress = (w + i / n) / 4.4;
          if (now() >= end) break;
        }
        J.tests = grid ? grid.tests : 0;
        if (i < n) return false;
        i = 0; return true;
      };
    }
    return job(J, [
      { name: 'index', fn: function () {     // the cell: twice the gap that the list's own Gaussians suggest (their size)
        var s = 0, k, c = Math.min(n, 4096);
        for (k = 0; k < c; k++) {
          var q = Math.floor(k * n / c) * 3;
          s += Math.hypot(L.axU[q], L.axU[q + 1], L.axU[q + 2]) + Math.hypot(L.axV[q], L.axV[q + 1], L.axV[q + 2]);
        }
        J.cell = (o.cellK || 2) * s / (2 * c * 0.72 * 0.95);
        grid = hashGrid(P, n, J.cell);
        return true;
      } },
      { name: 'search', fn: each(search, 0) },
      { name: 'normal', fn: each(normal, 1) },
      { name: 'fan', fn: each(fan, 2) },
      { name: 'vote', fn: each(vote, 3, function () { sStart[n] = sLen; out = new Uint32Array(Math.max(3, sLen * 3 / 2)); m = 0; }) },
      { name: 'audit', fn: function (end) {
        if (!aud) {
          J.tri = out.subarray(0, m * 3); J.nTri = m; aud = auditor(J.tri, m, L, o.audit);
          J.bytes = { work: grid.bytes + nbI.byteLength + nbD.byteLength + nbN.byteLength + NRM.byteLength + star.byteLength + sStart.byteLength + out.byteLength - m * 12, mesh: m * 12 };
        }
        var fin = aud.step(Math.max(1, end - now()));
        J.progress = (4 + 0.4 * aud.progress) / 4.4;
        if (fin) { J.stats = aud.stats; J.progress = 1; }
        return fin;
      } }
    ]);
  }

  /* ---------- the charts' own lattices, stitched ----------
     parts in the window's order; vertex (a, b) of part m is m.off + b * na + a (offsets assigned here). Returns the
     triangles (two per window stop, in the window's order), three edges per stop for drawing, and per vertex its chart
     and lattice cell (for the audit). No search: the neighbours are the next addresses. ms = the time it took.
     As implemented: one pass over the N samples, O(N) time, and nothing is kept while it runs. bytes = { work: 0, mesh }:
     the triangles (12 bytes each) are the only thing it makes, and they could be read off the addresses instead of
     stored; the vertices are the samples themselves. (The edge list is for drawing and the addresses for the audit:
     neither is counted.) */
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
             grid: parts.map(function (m) { return { na: m.na, nb: m.nb, wu: m.wu, wv: m.wv, patch: m.patch }; }), ms: now() - t0,
             bytes: { work: 0, mesh: tri.byteLength } };
  }

  /* ---------- one audit for both ----------
     tri: nTri triangles over the points of M = { n, chart, part, ca, cb, grid }: per point its true chart, the part
     (lattice) it was sampled from and its cell (ca, cb) there; grid[part] = { na, nb, wu, wv, patch }. Edges are
     counted by how many triangles hold them. A BRIDGE joins two points that are not neighbours on the surface:
     on two charts, or on one chart but at least twice as far apart along it as through the air (the surface path
     is measured along the straight line in (u, v) between the two cells; pairs within two cells of each other are
     neighbours). A HOLE is a loop of boundary edges (edges on one triangle only) that does not reach the chart's own
     rim (the outermost two rows of cells): a sheet's border is not a hole, and a gap that runs into it is not counted.
     stats: the counts, and for drawing, edges as (a, b, first triangle holding it), in the order of that triangle:
     bridges and non-manifold edges in `bad`, the edges around holes in `hole`.
     A mesh whose points are not the samples (TSDF fusion makes its own vertices) is graded the same way once its
     points have addresses: addresser(), below, gives each the address of the lattice sample nearest to it. Its
     triangles are as large as its own cells, not the lattice's, so `o.rim` says how many lattice cells from the
     chart's rim still count as the rim: two of the mesh's own cells (default 2: the lattice's and the list's). */
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
  function auditor(tri, nTri, M, o) {
    var nV = M.n, chart = M.chart, part = M.part, ca = M.ca, cb = M.cb, grid = M.grid, S = 1, RIM = Math.max(2, Math.round((o && o.rim) || 2));
    while (S < nTri * 3) S <<= 1;
    var mask = S - 1, K1 = new Int32Array(S).fill(-1), K2 = new Int32Array(S), CNT = new Int32Array(S), FIRST = new Int32Array(S), used = [];
    var par = new Int32Array(nV).fill(-1), rimE = [], t = 0, i = 0;
    var r = { points: nV, triangles: nTri, edges: 0, boundary: 0, nonManifold: 0, bridges: 0, loops: 0, holes: 0, unused: 0, bad: [], hole: [] };
    var J = { stats: null, progress: 0 };
    function add(a, b, t) {
      if (a > b) { var s = a; a = b; b = s; }
      var h = (Math.imul(a, 0x9E3779B1) ^ Math.imul(b, 0x85EBCA77)) & mask;
      while (K1[h] !== -1) { if (K1[h] === a && K2[h] === b) { CNT[h]++; return; } h = (h + 1) & mask; }
      K1[h] = a; K2[h] = b; CNT[h] = 1; FIRST[h] = t; used.push(h);
    }
    function far(a, b) {
      if (chart[a] !== chart[b] || part[a] !== part[b]) return true;
      var g = grid[part[a]], da = Math.abs(ca[a] - ca[b]), db = Math.abs(cb[a] - cb[b]);
      if (g.wu) da = Math.min(da, g.na - da);
      if (g.wv) db = Math.min(db, g.nb - db);
      if (Math.max(da, db) <= 2) return false;
      var w = along(g, ca[a], cb[a], ca[b], cb[b]);
      return w[0] > 2 * w[1];
    }
    function root(x) { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    function join(a, b) { if (par[a] < 0) par[a] = a; if (par[b] < 0) par[b] = b; a = root(a); b = root(b); if (a !== b) par[a] = b; }
    function border(x) {
      var g = grid[part[x]];
      return (!g.wu && (ca[x] < RIM || ca[x] >= g.na - RIM)) || (!g.wv && (cb[x] < RIM || cb[x] >= g.nb - RIM));
    }
    return job(J, [
      { name: 'edges', fn: function (end) {
        while (t < nTri) {
          for (var e = Math.min(nTri, t + 4096); t < e; t++) { add(tri[t * 3], tri[t * 3 + 1], t); add(tri[t * 3 + 1], tri[t * 3 + 2], t); add(tri[t * 3 + 2], tri[t * 3], t); }
          J.progress = 0.4 * t / Math.max(1, nTri);
          if (now() >= end) break;
        }
        return t >= nTri;
      } },
      { name: 'grade', fn: function (end) {
        while (i < used.length) {
          for (var e = Math.min(used.length, i + 4096); i < e; i++) {
            var h = used[i], a = K1[h], b = K2[h], c = CNT[h], br = far(a, b);
            if (c === 1) { r.boundary++; join(a, b); rimE.push(a, b, FIRST[h]); }
            if (c >= 3) r.nonManifold++;
            if (br) r.bridges++;
            if (br || c >= 3) r.bad.push(a, b, FIRST[h]);
          }
          J.progress = 0.4 + 0.5 * i / Math.max(1, used.length);
          if (now() >= end) break;
        }
        return i >= used.length;
      } },
      { name: 'loops', fn: function () {
        var open = new Uint8Array(nV), inT = new Uint8Array(nV), k;     // open: this loop reaches the rim
        r.edges = used.length;
        for (k = 0; k < nV; k++) if (par[k] >= 0 && border(k)) open[root(k)] = 1;
        for (k = 0; k < nV; k++) if (par[k] === k) { r.loops++; if (!open[k]) r.holes++; }
        for (k = 0; k < rimE.length; k += 3) if (!open[root(rimE[k])]) r.hole.push(rimE[k], rimE[k + 1], rimE[k + 2]);
        for (k = 0; k < nTri * 3; k++) inT[tri[k]] = 1;                // the points that ended up in no triangle at all
        for (k = 0; k < nV; k++) if (!inT[k]) r.unused++;
        J.stats = r; J.progress = 1; K1 = K2 = CNT = FIRST = used = null;
        return true;
      } }
    ]);
  }
  function audit(tri, nTri, M, o) { return auditor(tri, nTri, M, o).run().stats; }

  /* ---------- addresses for points that have none ----------
     S = { n, pos, chart, part, ca, cb, grid }: the lattice's samples, where they are and their addresses. Each of the
     n points P gets the address of the sample nearest to it, found through a hash grid over the samples (cell: twice
     the mean step along a lattice row). Where one of its four nearest samples lies on another chart, that one is kept
     as a second address (alt), for the distance to the surface near a seam. A job.
     -> J.M = { n, chart, part, ca, cb, grid, alt: { part (-1: none), ca, cb }, near: the sample's index } */
  function addresser(P, n, S, o) {
    var K = 4, nbI = new Int32Array(K), nbD = new Float32Array(K), g = null, i = 0, k;
    var M = { n: n, chart: new Uint16Array(n), part: new Uint16Array(n), ca: new Int32Array(n), cb: new Int32Array(n), grid: S.grid, near: new Int32Array(n),
              alt: { part: new Int16Array(n).fill(-1), ca: new Int32Array(n), cb: new Int32Array(n) } };
    var J = { M: null, progress: 0, cell: 0 };
    return job(J, [
      { name: 'index', fn: function () {
        var s = 0, c = 0, t, q;
        for (t = 0; t < 4096 && S.n > 1; t++) {
          q = Math.floor(t * (S.n - 1) / 4096);
          if (S.part[q] !== S.part[q + 1] || S.cb[q] !== S.cb[q + 1]) continue;
          s += Math.hypot(S.pos[q * 3] - S.pos[q * 3 + 3], S.pos[q * 3 + 1] - S.pos[q * 3 + 4], S.pos[q * 3 + 2] - S.pos[q * 3 + 5]); c++;
        }
        J.cell = (o && o.cell) || 2 * s / Math.max(1, c) || 1;
        g = hashGrid(S.pos, S.n, J.cell);
        return true;
      } },
      { name: 'nearest', fn: function (end) {
        while (i < n) {
          for (var e = Math.min(n, i + 256); i < e; i++) {
            var m = g.knnAt(P[i * 3], P[i * 3 + 1], P[i * 3 + 2], K, nbI, nbD, 0, -1), s = nbI[0];
            M.near[i] = s; M.chart[i] = S.chart[s]; M.part[i] = S.part[s]; M.ca[i] = S.ca[s]; M.cb[i] = S.cb[s];
            for (k = 1; k < m; k++) if (S.part[nbI[k]] !== S.part[s]) { M.alt.part[i] = S.part[nbI[k]]; M.alt.ca[i] = S.ca[nbI[k]]; M.alt.cb[i] = S.cb[nbI[k]]; break; }
          }
          J.progress = i / Math.max(1, n);
          if (now() >= end) break;
        }
        if (i < n) return false;
        J.M = M; J.progress = 1; g = null;
        return true;
      } }
    ]);
  }

  /* ---------- how far a mesh's points are from the surface ----------
     The point of a chart nearest to p: Gauss-Newton on (u, v) from a start (S_u, S_v are exact: one evaluation gives
     the point and both tangents), at most 12 steps, each no longer than a tenth of the domain, the best point met
     kept. g = { wu, wv, patch }. -> the distance. */
  var ec = new Float64Array(12);
  function closest(g, u, v, px, py, pz) {
    var best = 1e30, it, rx, ry, rz, d, a, b, c, gu, gv, det, du, dv;
    for (it = 0; it < 12; it++) {
      evalAt(g.patch, u, v, ec);
      rx = ec[0] - px; ry = ec[1] - py; rz = ec[2] - pz; d = Math.sqrt(rx * rx + ry * ry + rz * rz);
      if (d < best) best = d;
      a = ec[3] * ec[3] + ec[4] * ec[4] + ec[5] * ec[5]; b = ec[3] * ec[6] + ec[4] * ec[7] + ec[5] * ec[8]; c = ec[6] * ec[6] + ec[7] * ec[7] + ec[8] * ec[8];
      gu = ec[3] * rx + ec[4] * ry + ec[5] * rz; gv = ec[6] * rx + ec[7] * ry + ec[8] * rz; det = a * c - b * b;
      if (!(det > 1e-30)) break;
      du = -(c * gu - b * gv) / det; dv = -(a * gv - b * gu) / det;
      if (du > 0.1) du = 0.1; else if (du < -0.1) du = -0.1;
      if (dv > 0.1) dv = 0.1; else if (dv < -0.1) dv = -0.1;
      var u1 = g.wu ? (((u + du) % 1) + 1) % 1 : Math.min(Math.max(u + du, 0), 0.99999), v1 = g.wv ? (((v + dv) % 1) + 1) % 1 : Math.min(Math.max(v + dv, 0), 0.99999);
      if (Math.abs(u1 - u) < 1e-9 && Math.abs(v1 - v) < 1e-9) break;       // at rest (or held by the chart's rim)
      u = u1; v = v1;
    }
    return best;
  }
  /* the distances of n points P from the surface, each measured to the chart its address names (M = { part, ca, cb,
     grid }, and M.alt for a second address: the nearer of the two counts), starting from the middle of its cell. A job.
     -> J.stats = { n, mean, rms, p95, max } in scene units; J.dist. */
  function gauge(P, n, M) {
    var D = new Float32Array(n), i = 0, sum = 0, sq = 0, J = { stats: null, dist: D, progress: 0 }, A = M.alt || null;
    return job(J, [
      { name: 'project', fn: function (end) {
        while (i < n) {
          for (var e = Math.min(n, i + 128); i < e; i++) {
            var g = M.grid[M.part[i]], d = closest(g, (M.ca[i] + 0.5) / g.na, (M.cb[i] + 0.5) / g.nb, P[i * 3], P[i * 3 + 1], P[i * 3 + 2]);
            if (A && A.part[i] >= 0) { g = M.grid[A.part[i]]; d = Math.min(d, closest(g, (A.ca[i] + 0.5) / g.na, (A.cb[i] + 0.5) / g.nb, P[i * 3], P[i * 3 + 1], P[i * 3 + 2])); }
            D[i] = d; sum += d; sq += d * d;
          }
          J.progress = 0.95 * i / Math.max(1, n);
          if (now() >= end) break;
        }
        return i >= n;
      } },
      { name: 'rank', fn: function () {
        var s = new Float32Array(D).sort();
        J.stats = { n: n, mean: n ? sum / n : 0, rms: n ? Math.sqrt(sq / n) : 0, p95: n ? s[Math.min(n - 1, Math.floor(0.95 * n))] : 0, max: n ? s[n - 1] : 0 };
        J.progress = 1;
        return true;
      } }
    ]);
  }

  var api = { emulator: emulator, emulate: emulate, mesher: mesher, lattice: lattice, auditor: auditor, audit: audit, evalAt: evalAt,
              gap: gap, addresser: addresser, gauge: gauge, closest: closest };
  KT.meshList = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
