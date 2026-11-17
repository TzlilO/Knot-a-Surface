/* Knot-a-Surface talk — a mesh from a list of Gaussians by TSDF FUSION (KnotTalk.meshTsdf).
   The route most 3DGS papers take from Gaussians to triangles, run for real on the page:
     1  depth maps   one per capture camera: a pinhole at the camera's centre, looking at the middle of the Gaussians'
                     box and exactly as wide as that box's bounding sphere, S x S pixels. Every Gaussian is drawn into a
                     z-buffer as a flat disc in its own plane (FOOT sigmas along each axis, and at least a pixel
                     and a half across), each pixel taking the depth of the nearest disc on its ray, and that Gaussian's colour.
                     A CPU rasteriser: the same picture on every machine and under node. A pixel no disc reaches is no
                     measurement.
     2  fusion       a DENSE voxel grid over the box. Every voxel is projected into every depth map; where the surface
                     seen there is no further than the truncation distance in front of the voxel, its signed distance
                     (depth seen - the voxel's depth, cut off at the truncation distance) and the colour seen go into
                     a running average, weight one per view: the update of Curless & Levoy 1996, as KinectFusion and
                     Open3D's uniform TSDF volume make it.
     3  extraction   the zero level set by MARCHING CUBES, slab by slab, over the cells whose eight corners were seen
                     by at least WMIN views. The 256 cases are not typed in: they are generated from one rule per
                     face of the cube (below), and checkCubes() proves, case by case, that any grid of values gives
                     a closed, consistently oriented, manifold surface.
   Views are taken one at a time (draw the depth map, fuse it, reuse its buffer), so the memory is the volume and one
   depth map. As implemented, with N Gaussians, V views, P = S x S pixels a view and R^3 voxels:
     time     depth maps O(V (N + P))  (a disc covers a bounded number of pixels, because the picture is only as fine
              as the voxels) + fusion O(V R^3) + extraction O(R^3);
     memory   O(R^3 + P): 6 bytes a voxel (distance int16, weight uint8, colour 3 x uint8), 8 bytes a pixel, and
              O(R^2) for the extractor's edge tables, besides the mesh that comes out.
   THE GRID: the voxel is the Gaussians' mean spacing, so that the grid could hold what the samples hold, unless the
   box would then need more than `maxVoxels` voxels: then the voxel is as large as that cap makes it (grid.capped,
   grid.wanted = the voxels the spacing would have needed). The picture's pixel is one voxel wide at the middle of
   the box.
   fuser(list, o) is a JOB like those of talk/meshlist.js: step(ms) works for about ms milliseconds and returns true
   when finished, run() goes to the end; the time spent inside the steps is measured per stage (ms.depth, ms.fuse,
   ms.extract), and the result does not depend on how the work was sliced. list = { n, pos, axU, axV, col } (the
   emulator of talk/meshlist.js makes one: EMULATED, never trained).
   Plain functions; also require()-able under node (tests). Depends on nothing. */
(function (root) {
  'use strict';
  var KT = root.KnotTalk = root.KnotTalk || {};
  var now = function () { return root.performance && root.performance.now ? root.performance.now() : Date.now(); };
  var BIG = 1e30;

  /* ---------- marching cubes: the 256 cases, generated ----------
     A cell's corner c sits at (c & 1, c >> 1 & 1, c >> 2 & 1) along the grid's (i, j, k); bit c of a case is set when
     that corner is INSIDE (its value is negative). Edge a * 4 + p + 2 q runs along axis a from the corner whose other
     two coordinates, in the cyclic order after a, are p and q.
     The rule, per face of the cube: walk its four corners around; every run of inside corners is cut off by ONE
     segment, from the edge where the walk enters the run to the edge where it leaves it. A face with two inside
     corners on a diagonal therefore gets two segments, one around each: the inside corners are kept apart. The rule
     reads the face's four signs only, so the two cells that share a face draw the same segments on it. Directed
     with the inside on the right as seen from outside the cell, the segments of a cell close into loops; each loop is
     filled with triangles, counter-clockwise seen from the outside (the positive side). */
  var CUBES = (function () {
    var E0 = new Uint8Array(12), E1 = new Uint8Array(12), a, p, q, m, s, faces = [];
    function edge(c0, c1) { var lo = c0 & c1, d = c0 ^ c1, ax = d === 1 ? 0 : d === 2 ? 1 : 2; return ax * 4 + ((lo >> (ax + 1) % 3) & 1) + 2 * ((lo >> (ax + 2) % 3) & 1); }
    for (a = 0; a < 3; a++) for (q = 0; q < 2; q++) for (p = 0; p < 2; p++) { E0[a * 4 + p + 2 * q] = (p << (a + 1) % 3) | (q << (a + 2) % 3); E1[a * 4 + p + 2 * q] = E0[a * 4 + p + 2 * q] | (1 << a); }
    // a face: axis a, side s; its corners in the order (0,0) (1,0) (1,1) (0,1) of the two other axes, which is
    // counter-clockwise seen from +a (from outside the cell when s = 1, from inside when s = 0)
    for (a = 0; a < 3; a++) for (s = 0; s < 2; s++) faces.push({ a: a, s: s, c: [[0, 0], [1, 0], [1, 1], [0, 1]].map(function (w) { return (s << a) | (w[0] << (a + 1) % 3) | (w[1] << (a + 2) % 3); }) });
    function segments(F, m) {              // the directed segments of case m on face F: [[from edge, to edge], ...]
      var ins = F.c.map(function (c) { return (m >> c) & 1; }), n = ins[0] + ins[1] + ins[2] + ins[3], out = [], k, j;
      if (n === 0 || n === 4) return out;
      for (k = 0; k < 4; k++) {
        if (!ins[k] || ins[(k + 3) % 4]) continue;                  // corner k starts a run of inside corners
        for (j = k; ins[(j + 1) % 4]; j = (j + 1) % 4);             // ... which ends at corner j
        var enter = edge(F.c[(k + 3) % 4], F.c[k]), leave = edge(F.c[j], F.c[(j + 1) % 4]);
        out.push(F.s ? [enter, leave] : [leave, enter]);
      }
      return out;
    }
    /* how squarely the triangle on edges (a, b, c) of case m faces the outside, with its corners at the middle of
       their edges: its unit normal . the sum of (outside end - inside end) over the three edges */
    function facing(a, b, c, m) {
      var P = [a, b, c].map(function (e) { var c0 = E0[e], c1 = E1[e]; return [((c0 & 1) + (c1 & 1)) / 2, ((c0 >> 1 & 1) + (c1 >> 1 & 1)) / 2, ((c0 >> 2 & 1) + (c1 >> 2 & 1)) / 2]; });
      var ux = P[1][0] - P[0][0], uy = P[1][1] - P[0][1], uz = P[1][2] - P[0][2], vx = P[2][0] - P[0][0], vy = P[2][1] - P[0][1], vz = P[2][2] - P[0][2];
      var n = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx], l = Math.hypot(n[0], n[1], n[2]) || 1, sum = 0;
      [a, b, c].forEach(function (e) { sum += ((m >> E0[e]) & 1 ? 1 : -1) * n[e >> 2]; });
      return sum / l;
    }
    /* a loop of 3 to 8 edges is filled with triangles: of all the ways to cut the polygon, the one whose worst
       triangle faces the outside most squarely (the first such, in a fixed order) */
    function fill(loop, m) {
      function cut(i, j) {                 // every triangulation of the polygon loop[i..j] -> [{ t: [edges...], s: its worst facing }]
        if (j - i < 2) return [{ t: [], s: 9 }];
        var out = [], k;
        for (k = i + 1; k < j; k++) {
          var f = facing(loop[i], loop[k], loop[j], m), L = cut(i, k), R = cut(k, j);
          L.forEach(function (l) { R.forEach(function (r) { out.push({ t: l.t.concat([loop[i], loop[k], loop[j]], r.t), s: Math.min(f, l.s, r.s) }); }); });
        }
        return out;
      }
      var all = cut(0, loop.length - 1), best = all[0];
      all.forEach(function (c) { if (c.s > best.s + 1e-9) best = c; });
      return best.t;
    }
    var cases = [], most = 0;
    for (m = 0; m < 256; m++) {
      var next = new Int8Array(12).fill(-1), seen = new Uint8Array(12), tri = [], e, x, loop;
      faces.forEach(function (F) { segments(F, m).forEach(function (g) { next[g[0]] = g[1]; }); });
      for (e = 0; e < 12; e++) {
        if (next[e] < 0 || seen[e]) continue;
        for (loop = [], x = e; !seen[x]; x = next[x]) { seen[x] = 1; loop.push(x); }
        fill(loop, m).forEach(function (t) { tri.push(t); });
      }
      cases.push(tri); most = Math.max(most, tri.length / 3);
    }
    var T3 = most * 3, tab = new Int8Array(256 * T3).fill(-1), cnt = new Uint8Array(256);
    cases.forEach(function (t, k) { cnt[k] = t.length / 3; tab.set(t, k * T3); });
    return { e0: E0, e1: E1, tri: tab, count: cnt, stride: T3, most: most, faces: faces, segments: segments, edge: edge };
  })();

  /* The proof that the table is right, case by case (node tests run it; it takes a millisecond):
       1  a triangle's corners lie on three different edges of the cell, each with one end inside and one outside,
          and every such edge is used;
       2  inside a cell every edge of the triangulation is either shared by exactly two triangles, which run along
          it in opposite directions, or it lies in a face of the cell and belongs to one triangle;
       3  the edges that lie in a face are exactly the segments the face rule gives for that face's four signs, in
          the rule's direction: the same for every case that shows the face those signs;
       4  seen from the cell on the other side, the same face with the same signs gets the same segments reversed.
     1-4 for all 256 cases mean: over any grid, every triangle edge inside the grid is shared by exactly two
     triangles that run along it in opposite directions (a closed, oriented surface), and the triangles around a
     vertex form one fan (each of the four faces around its grid edge holds exactly one segment at it).
     5  orientation: every triangle faces the outside. Checked with the corners at the middle of their edges: the
        sum over a triangle's three edges of normal . (outside end - inside end) is positive.
     -> { ok, cases, most (triangles in the largest case), errors: [text] } */
  function checkCubes() {
    var C = CUBES, err = [], m, t, k, q, face = {};
    function mid(e) { var c0 = C.e0[e], c1 = C.e1[e]; return [((c0 & 1) + (c1 & 1)) / 2, ((c0 >> 1 & 1) + (c1 >> 1 & 1)) / 2, ((c0 >> 2 & 1) + (c1 >> 2 & 1)) / 2]; }
    function inFace(F, e) { return ((C.e0[e] >> F.a) & 1) === F.s && ((C.e1[e] >> F.a) & 1) === F.s; }
    for (m = 0; m < 256; m++) {
      var n = C.count[m], used = {}, dir = {}, cross = 0, e;
      for (e = 0; e < 12; e++) if (((m >> C.e0[e]) & 1) !== ((m >> C.e1[e]) & 1)) cross |= 1 << e;
      for (t = 0; t < n; t++) {
        var v = [C.tri[m * C.stride + t * 3], C.tri[m * C.stride + t * 3 + 1], C.tri[m * C.stride + t * 3 + 2]];
        if (v[0] === v[1] || v[1] === v[2] || v[0] === v[2]) err.push('case ' + m + ': a triangle repeats an edge');
        for (k = 0; k < 3; k++) {
          if (!((cross >> v[k]) & 1)) err.push('case ' + m + ': a corner on edge ' + v[k] + ', which the surface does not cross');
          used[v[k]] = 1;
          var key = v[k] + '>' + v[(k + 1) % 3]; dir[key] = (dir[key] || 0) + 1;
        }
        // 5: the triangle faces the outside
        var A = mid(v[0]), B = mid(v[1]), D = mid(v[2]), nx = (B[1] - A[1]) * (D[2] - A[2]) - (B[2] - A[2]) * (D[1] - A[1]),
            ny = (B[2] - A[2]) * (D[0] - A[0]) - (B[0] - A[0]) * (D[2] - A[2]), nz = (B[0] - A[0]) * (D[1] - A[1]) - (B[1] - A[1]) * (D[0] - A[0]), sum = 0;
        for (k = 0; k < 3; k++) { var c0 = C.e0[v[k]], c1 = C.e1[v[k]], sg = (m >> c0) & 1 ? 1 : -1, ax = v[k] >> 2; sum += sg * (ax === 0 ? nx : ax === 1 ? ny : nz); }
        if (!(sum > 0)) err.push('case ' + m + ': a triangle faces the inside');
      }
      for (e = 0; e < 12; e++) if (((cross >> e) & 1) && !used[e]) err.push('case ' + m + ': crossed edge ' + e + ' is not used');
      // 2 and 3: what is left over after pairing opposite directions must be the face rule's segments, each once
      var want = {};
      C.faces.forEach(function (F, fi) {
        var sg = C.segments(F, m), pat = F.c.map(function (c) { return (m >> c) & 1; }).join('');
        sg.forEach(function (g) { want[g[0] + '>' + g[1]] = (want[g[0] + '>' + g[1]] || 0) + 1; if (!inFace(F, g[0]) || !inFace(F, g[1])) err.push('case ' + m + ': a segment leaves its face'); });
        // 3: one answer per face and pattern (the rule is a function of the four signs; kept as a table to compare across cells)
        var id = fi + ':' + pat, txt = sg.map(function (g) { return g.join('>'); }).sort().join(' ');
        if (face[id] == null) face[id] = txt; else if (face[id] !== txt) err.push('face ' + id + ' is cut two ways');
      });
      Object.keys(dir).forEach(function (key) {
        var ab = key.split('>'), back = dir[ab[1] + '>' + ab[0]] || 0, w = want[key] || 0;
        if (dir[key] > 1) err.push('case ' + m + ': edge ' + key + ' is in more than one triangle on the same side');
        if (back + w !== 1) err.push('case ' + m + ': edge ' + key + ' is neither shared by two triangles nor a segment of a face');
      });
      Object.keys(want).forEach(function (key) { if (dir[key] !== 1) err.push('case ' + m + ': face segment ' + key + ' is missing'); });
    }
    // 4: the face a = s of a cell is the face a = 1 - s of its neighbour; the edges in it keep their other two
    //    coordinates, so edge ids differ only in the bit that says on which side of the cell the face is
    C.faces.forEach(function (F, fi) {
      if (F.s !== 1) return;
      var G = C.faces[fi - 1], pat, id;                              // the same axis, side 0
      function over(e) {                                             // the edge of the neighbour that is this cell's edge e (in face a = 1)
        var ax = e >> 2, p = e & 1, q = (e >> 1) & 1;
        if ((ax + 1) % 3 === F.a) p = 0; else q = 0;
        return ax * 4 + p + 2 * q;
      }
      for (pat = 0; pat < 16; pat++) {
        id = [0, 1, 2, 3].map(function (k) { return (pat >> k) & 1; }).join('');
        var here = face[fi + ':' + id], there = face[(fi - 1) + ':' + id];
        if (here == null || there == null) { err.push('face pattern ' + id + ' never met'); continue; }
        var rev = here.split(' ').filter(Boolean).map(function (g) { var ab = g.split('>'); return over(+ab[1]) + '>' + over(+ab[0]); }).sort().join(' ');
        if (rev !== there) err.push('faces ' + F.a + ' of two neighbouring cells disagree on pattern ' + id);
      }
      void G; void q;
    });
    return { ok: !err.length, cases: 256, most: C.most, errors: err };
  }

  /* ---------- the zero level set of a grid of values, as a job ----------
     vol = { n: [n0, n1, n2], D, W, C, wmin, lo, h, ax }: value D[(k * n1 + j) * n0 + i] at grid point (i, j, k), which
     is at lo[ax[q]] + h * (i, j, k)[q] along world axis ax[q] (ax: an even permutation of 0 1 2, so that the grid's
     axes are right-handed; default 0 1 2). A point is inside where D < 0. W (optional): how many views saw each point;
     a cell with a corner below wmin is left out. C (optional): Uint8 r g b per point, interpolated like the position.
     -> J.pos, J.col (0..1, or null), J.tri, J.nV, J.nTri; triangles counter-clockwise seen from the positive side,
     in the order of the slabs k: J.slab[k] = how many triangles the slabs 0..k hold. */
  function extractor(vol) {
    var n0 = vol.n[0], n1 = vol.n[1], n2 = vol.n[2], D = vol.D, W = vol.W || null, C = vol.C || null, wmin = vol.wmin || 1, h = vol.h, lo = vol.lo, ax = vol.ax || [0, 1, 2];
    var s0 = n0 * n1, off = new Int32Array(8), c, k = 0, TB = CUBES.tri, CN = CUBES.count, ST = CUBES.stride, E0 = CUBES.e0;
    for (c = 0; c < 8; c++) off[c] = (c & 1) + ((c >> 1) & 1) * n0 + ((c >> 2) & 1) * s0;
    var vcap = 4096, tcap = 8192, pos = new Float32Array(vcap * 3), col = C ? new Float32Array(vcap * 3) : null, tri = new Uint32Array(tcap * 3), nV = 0, nT = 0;
    // one vertex per crossed grid edge. Tables of the edges met so far: along i and along j in the planes k and k + 1,
    // along k between them. 0 = no vertex yet, else its index + 1
    var tabs = [new Int32Array(s0), new Int32Array(s0), new Int32Array(s0), new Int32Array(s0), new Int32Array(s0)];   // i edges at k, at k + 1; j edges at k, at k + 1; k edges
    var eTab = new Uint8Array(12), eOff = new Int32Array(12), e, w3 = [0, 0, 0];
    for (e = 0; e < 12; e++) {
      var a = e >> 2, p = e & 1, q = (e >> 1) & 1;
      if (a === 0) { eTab[e] = q; eOff[e] = p * n0; }                 // along i, at (i, j + p, k + q)
      else if (a === 1) { eTab[e] = 2 + p; eOff[e] = q; }             // along j, at (i + q, j, k + p)
      else { eTab[e] = 4; eOff[e] = q * n0 + p; }                     // along k, at (i + p, j + q, k)
    }
    var J = { done: false, progress: 0, ms: 0, pos: null, col: null, tri: null, nV: 0, nTri: 0, slab: new Uint32Array(Math.max(1, n2 - 1)), bytes: 5 * s0 * 4 };
    function vertex(e, idx, i, j) {
      var c0 = E0[e], g0 = idx + off[c0], g1 = idx + off[CUBES.e1[e]], d0 = D[g0], d1 = D[g1], t = d0 / (d0 - d1), o = nV * 3, a = e >> 2;
      if (nV === vcap) { vcap *= 2; var np = new Float32Array(vcap * 3); np.set(pos); pos = np; if (col) { np = new Float32Array(vcap * 3); np.set(col); col = np; } }
      w3[0] = i + (c0 & 1); w3[1] = j + ((c0 >> 1) & 1); w3[2] = k + ((c0 >> 2) & 1); w3[a] += t;
      pos[o + ax[0]] = lo[ax[0]] + h * w3[0]; pos[o + ax[1]] = lo[ax[1]] + h * w3[1]; pos[o + ax[2]] = lo[ax[2]] + h * w3[2];
      if (col) { g0 *= 3; g1 *= 3; col[o] = (C[g0] + t * (C[g1] - C[g0])) / 255; col[o + 1] = (C[g0 + 1] + t * (C[g1 + 1] - C[g0 + 1])) / 255; col[o + 2] = (C[g0 + 2] + t * (C[g1 + 2] - C[g0 + 2])) / 255; }
      return nV++;
    }
    function slabK() {
      var i, j, idx, m, t, n, q, e, tb, at, v;
      for (j = 0; j < n1 - 1; j++) for (i = 0, idx = (k * n1 + j) * n0; i < n0 - 1; i++, idx++) {
        if (W && (W[idx] < wmin || W[idx + 1] < wmin || W[idx + n0] < wmin || W[idx + n0 + 1] < wmin ||
                  W[idx + s0] < wmin || W[idx + s0 + 1] < wmin || W[idx + s0 + n0] < wmin || W[idx + s0 + n0 + 1] < wmin)) continue;
        m = (D[idx] < 0 ? 1 : 0) | (D[idx + 1] < 0 ? 2 : 0) | (D[idx + n0] < 0 ? 4 : 0) | (D[idx + n0 + 1] < 0 ? 8 : 0) |
            (D[idx + s0] < 0 ? 16 : 0) | (D[idx + s0 + 1] < 0 ? 32 : 0) | (D[idx + s0 + n0] < 0 ? 64 : 0) | (D[idx + s0 + n0 + 1] < 0 ? 128 : 0);
        if (m === 0 || m === 255) continue;
        n = CN[m] * 3;
        if ((nT + CN[m]) > tcap) { tcap *= 2; var nt = new Uint32Array(tcap * 3); nt.set(tri); tri = nt; }
        for (t = 0, q = m * ST; t < n; t++) {
          e = TB[q + t]; tb = tabs[eTab[e]]; at = j * n0 + i + eOff[e]; v = tb[at];
          if (!v) { v = vertex(e, idx, i, j) + 1; tb[at] = v; }
          tri[nT * 3 + t] = v - 1;
        }
        nT += CN[m];
      }
      J.slab[k] = nT;
      var sw = tabs[0]; tabs[0] = tabs[1]; tabs[1] = sw; tabs[1].fill(0);
      sw = tabs[2]; tabs[2] = tabs[3]; tabs[3] = sw; tabs[3].fill(0); tabs[4].fill(0);
    }
    J.step = function (budget) {
      var t0 = now(), end = t0 + (budget == null ? 1e9 : budget);
      while (!J.done) {
        if (k < n2 - 1) { slabK(); k++; }
        if (k >= n2 - 1) { J.pos = pos.slice(0, nV * 3); J.col = col ? col.slice(0, nV * 3) : null; J.tri = tri.slice(0, nT * 3); J.nV = nV; J.nTri = nT; J.done = true; pos = col = tri = tabs = null; }
        J.progress = J.done ? 1 : k / Math.max(1, n2 - 1);
        if (now() >= end) break;
      }
      J.ms += now() - t0;
      return J.done;
    };
    J.run = function () { while (!J.step(1e9)); return J; };
    return J;
  }

  /* ---------- the grid ----------
     over the box lo..hi plus a margin of the truncation distance and a voxel: the voxel is `spacing`, unless that
     takes more than `cap` voxels: then it is as large as the cap makes it. `trv` = the truncation distance, in voxels.
     -> { h, n: [nx, ny, nz], voxels, lo: where grid point (0, 0, 0) is, capped, wanted: { h, n, voxels } } */
  function sizeGrid(lo, hi, spacing, cap, trv) {
    function at(h) { return [0, 1, 2].map(function (q) { return Math.ceil((hi[q] - lo[q] + 2 * (trv + 1) * h) / h) + 1; }); }
    var h = spacing, w = at(h), n = w, tot = n[0] * n[1] * n[2], it;
    var want = { h: h, n: w, voxels: tot };
    for (it = 0; it < 200 && tot > cap; it++) { h *= Math.max(1.002, Math.cbrt(tot / cap)); n = at(h); tot = n[0] * n[1] * n[2]; }
    return { h: h, n: n, voxels: tot, capped: tot !== want.voxels, wanted: want,
             lo: [0, 1, 2].map(function (q) { return (lo[q] + hi[q]) / 2 - (n[q] - 1) * h / 2; }) };
  }

  /* ---------- the cameras ----------
     one pinhole per centre, looking at `at` with +y up, S x S pixels, exactly as wide as the sphere of radius `rad`
     around `at` (at most MAXFOV degrees to either side when the camera is inside it or close to it).
     -> { c, r, u, f (right, up, forward: unit vectors), F (focal length in pixels), S } */
  var MAXFOV = 55;
  function camera(c, at, rad, S) {
    var fx = at[0] - c[0], fy = at[1] - c[1], fz = at[2] - c[2], d = Math.hypot(fx, fy, fz) || 1e-9;
    fx /= d; fy /= d; fz /= d;
    var ux = 0, uy = 1, uz = 0; if (Math.abs(fy) > 0.999) { ux = 1; uy = 0; }          // looking straight down: any other up
    var rx = fy * uz - fz * uy, ry = fz * ux - fx * uz, rz = fx * uy - fy * ux, rl = Math.hypot(rx, ry, rz);
    rx /= rl; ry /= rl; rz /= rl;
    ux = ry * fz - rz * fy; uy = rz * fx - rx * fz; uz = rx * fy - ry * fx;
    var tmax = Math.tan(MAXFOV * Math.PI / 180), tn = d > rad * 1.0001 ? Math.min(tmax, rad / Math.sqrt(d * d - rad * rad)) : tmax;
    return { c: [c[0], c[1], c[2]], r: [rx, ry, rz], u: [ux, uy, uz], f: [fx, fy, fz], F: S / 2 / tn, S: S, d: d };
  }

  /* the gap between neighbouring Gaussians that a list's own Gaussians suggest: their mean axis, over what the
     emulator of talk/meshlist.js makes of a gap (0.72 x 0.6..1.3). talk/meshlist.js has the same rule (gap()) */
  function gapOf(L) {
    var s = 0, k, c = Math.min(L.n, 4096);
    for (k = 0; k < c; k++) { var q = Math.floor(k * L.n / c) * 3; s += Math.hypot(L.axU[q], L.axU[q + 1], L.axU[q + 2]) + Math.hypot(L.axV[q], L.axV[q + 1], L.axV[q + 2]); }
    return s / (2 * c * 0.72 * 0.95);
  }

  /* ---------- the whole route, as a job ----------
     o = { cams: [[x, y, z], ...] camera centres, spacing (default: what the list's Gaussians suggest), maxVoxels (4e6),
           maxViews (64: of more cameras, that many, evenly), maxSide (512: pixels along a depth map's side),
           trunc (3: the truncation distance in voxels), minWeight (3: views that must have seen a cell's corners, the
           default of Open3D's voxel block grid), foot (1.6: a Gaussian's disc, in sigmas) }
     J.V, J.S, J.P = views, pixels along a side, pixels a view; J.grid (sizeGrid); J.trunc (in scene units);
     J.view = views fused so far; J.stage = 'depth' | 'fuse' | 'extract' | 'done'; J.ms = { depth, fuse, extract };
     J.bytes = { volume, maps, tables, mesh }: the bytes of the arrays allocated: the volume (6 a voxel), the depth
     map and its Gaussians (8 a pixel), the extractor's edge tables and the Gaussians' colours as bytes, the mesh
     (24 a vertex: position and colour; 12 a triangle), known when it is out;
     J.mesh = { pos, col, tri, nV, nTri, slab } when done. The volume is released as soon as the mesh is out;
     free() releases everything else too. */
  function fuser(L, o) {
    o = o || {};
    var n = L.n, P = L.pos, AU = L.axU, AV = L.axV, CL = L.col, i, q;
    var FOOT = o.foot || 1.6, MINR = o.minPixels == null ? 0.75 : o.minPixels, TRV = o.trunc || 3, WMIN = o.minWeight == null ? 3 : o.minWeight;
    var lo = [BIG, BIG, BIG], hi = [-BIG, -BIG, -BIG];
    for (i = 0; i < n; i++) for (q = 0; q < 3; q++) { var x = P[i * 3 + q]; if (x < lo[q]) lo[q] = x; if (x > hi[q]) hi[q] = x; }
    var spacing = o.spacing || gapOf(L), G = sizeGrid(lo, hi, spacing, o.maxVoxels || 4e6, TRV), h = G.h, tr = TRV * h;
    var mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], rad = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) / 2;
    var S = Math.max(32, Math.min(o.maxSide || 512, Math.ceil(2 * rad / h))), all = o.cams || [], VM = o.maxViews || 64, cams = [];
    for (i = 0; i < Math.min(VM, all.length); i++) cams.push(camera(all[all.length > VM ? Math.floor(i * all.length / VM) : i], mid, rad, S));
    var V = cams.length, AX = [2, 0, 1], n0 = G.n[AX[0]], n1 = G.n[AX[1]], n2 = G.n[AX[2]], NV = n0 * n1 * n2;      // slabs are horizontal: k runs along +y
    var D = null, W = null, C = null, C8 = null, depth = null, own = null, ex = null;
    var J = { n: n, done: false, stage: 'depth', progress: 0, view: 0, V: V, views: all.length, S: S, P: S * S, grid: G, spacing: spacing, trunc: tr, truncVoxels: TRV, minWeight: WMIN,
              foot: FOOT, cams: cams, ms: { depth: 0, fuse: 0, extract: 0 }, mesh: null, slabAxis: AX[2],
              bytes: { volume: NV * 6, maps: S * S * 8, tables: 5 * n0 * n1 * 4 + n * 3, mesh: 0 } };
    var v = 0, gi = 0, kz = 0, cam = null;

    /* a depth map: Gaussians gi.. of the list into the z-buffer of camera `cam`, until the clock says stop */
    function draw(end) {
      var c = cam.c, r = cam.r, u = cam.u, f = cam.f, F = cam.F, cx = S / 2, cy = S / 2, near = 1e-3 * cam.d, g, o3, e;
      var first = gi;
      if (gi === 0) { depth.fill(BIG); own.fill(-1); }
      for (; gi < n; gi++) {
        if (gi > first && (gi & 1023) === 0 && now() >= end) return false;
        g = gi; o3 = g * 3;
        var dx = P[o3] - c[0], dy = P[o3 + 1] - c[1], dz = P[o3 + 2] - c[2];
        var X = r[0] * dx + r[1] * dy + r[2] * dz, Y = u[0] * dx + u[1] * dy + u[2] * dz, Z = f[0] * dx + f[1] * dy + f[2] * dz;
        if (Z < near) continue;
        var iz = 1 / Z, fz = F * iz, x0 = cx + X * fz, y0 = cy - Y * fz;
        // its two axes, FOOT sigmas long and never shorter than MINR pixels are wide at its depth (a disc smaller than
        // a pixel would fall between the pixels' rays), in the camera's frame; and their pictures (the projection,
        // linear about the centre). The disc grows in its own plane only: seen edge-on it stays a sliver
        var ka = FOOT, kb = FOOT, la = AU[o3] * AU[o3] + AU[o3 + 1] * AU[o3 + 1] + AU[o3 + 2] * AU[o3 + 2], lb = AV[o3] * AV[o3] + AV[o3 + 1] * AV[o3 + 1] + AV[o3 + 2] * AV[o3 + 2], lm = MINR * Z / F;
        if (la * ka * ka < lm * lm) ka = lm / Math.sqrt(la || 1e-30);
        if (lb * kb * kb < lm * lm) kb = lm / Math.sqrt(lb || 1e-30);
        var Ax = (r[0] * AU[o3] + r[1] * AU[o3 + 1] + r[2] * AU[o3 + 2]) * ka, Ay = (u[0] * AU[o3] + u[1] * AU[o3 + 1] + u[2] * AU[o3 + 2]) * ka, Az = (f[0] * AU[o3] + f[1] * AU[o3 + 1] + f[2] * AU[o3 + 2]) * ka;
        var Bx = (r[0] * AV[o3] + r[1] * AV[o3 + 1] + r[2] * AV[o3 + 2]) * kb, By = (u[0] * AV[o3] + u[1] * AV[o3 + 1] + u[2] * AV[o3 + 2]) * kb, Bz = (f[0] * AV[o3] + f[1] * AV[o3 + 1] + f[2] * AV[o3 + 2]) * kb;
        var ax = fz * (Ax - X * iz * Az), ay = -fz * (Ay - Y * iz * Az), bx = fz * (Bx - X * iz * Bz), by = -fz * (By - Y * iz * Bz);
        // the disc's picture is the ellipse d' S^-1 d <= 1, S = a a' + b b'
        var sxx = ax * ax + bx * bx + 1e-6, sxy = ax * ay + bx * by, syy = ay * ay + by * by + 1e-6, hx = Math.sqrt(sxx), hy = Math.sqrt(syy);
        var u0 = Math.ceil(x0 - hx - 0.5), u1 = Math.floor(x0 + hx - 0.5), v0 = Math.ceil(y0 - hy - 0.5), v1 = Math.floor(y0 + hy - 0.5);
        if (u0 < 0) u0 = 0; if (v0 < 0) v0 = 0; if (u1 > S - 1) u1 = S - 1; if (v1 > S - 1) v1 = S - 1;
        if (u0 > u1 || v0 > v1) continue;
        var det = sxx * syy - sxy * sxy, qa = syy / det, qb = -sxy / det, qc = sxx / det;
        // the depth along a pixel's ray: where the ray meets the disc's plane, kept within the disc's own depth range;
        // a disc seen within 6 degrees of edge-on is flat at its centre's depth
        var nx = Ay * Bz - Az * By, ny = Az * Bx - Ax * Bz, nz = Ax * By - Ay * Bx, num = nx * X + ny * Y + nz * Z;
        var flat = num * num < 0.01 * (nx * nx + ny * ny + nz * nz) * (X * X + Y * Y + Z * Z), ez = Math.sqrt(Az * Az + Bz * Bz), zl = Z - ez, zh = Z + ez;
        for (var pv = v0; pv <= v1; pv++) {
          var py = pv + 0.5, ddy = py - y0, ry = (cy - py) / F, row = pv * S;
          for (var pu = u0; pu <= u1; pu++) {
            var ddx = pu + 0.5 - x0;
            if (qa * ddx * ddx + 2 * qb * ddx * ddy + qc * ddy * ddy > 1) continue;
            var z = Z;
            if (!flat) { z = num / (nx * (pu + 0.5 - cx) / F + ny * ry + nz); if (!(z > zl)) z = zl; else if (z > zh) z = zh; }
            e = row + pu;
            if (z < depth[e]) { depth[e] = z; own[e] = g; }
          }
        }
      }
      gi = 0;
      return true;
    }
    /* fusion: the slabs kz.. of the volume against the depth map of camera `cam` */
    function fuse(end) {
      var c = cam.c, r = cam.r, u = cam.u, f = cam.f, F = cam.F, cx = S / 2, cy = S / 2, near = 1e-3 * cam.d, GL = G.lo, k, j, i2, e, w, t, sdf, d, g, q3, px, py, iz, iw;
      var ox = GL[0] - c[0], oy = GL[1] - c[1], oz = GL[2] - c[2], K = 32767 / tr;
      var X0 = r[0] * ox + r[1] * oy + r[2] * oz, Y0 = u[0] * ox + u[1] * oy + u[2] * oz, Z0 = f[0] * ox + f[1] * oy + f[2] * oz;
      var dXi = h * r[AX[0]], dYi = h * u[AX[0]], dZi = h * f[AX[0]], dXj = h * r[AX[1]], dYj = h * u[AX[1]], dZj = h * f[AX[1]], dXk = h * r[AX[2]], dYk = h * u[AX[2]], dZk = h * f[AX[2]];
      for (var first = kz; kz < n2; kz++) {
        if (kz > first && now() >= end) return false;
        k = kz;
        for (j = 0; j < n1; j++) {
          var X = X0 + k * dXk + j * dXj, Y = Y0 + k * dYk + j * dYj, Z = Z0 + k * dZk + j * dZj, q0 = (k * n1 + j) * n0;
          for (i2 = 0; i2 < n0; i2++, X += dXi, Y += dYi, Z += dZi) {
            if (Z < near) continue;
            iz = F / Z; px = cx + X * iz; py = cy - Y * iz;
            if (!(px >= 0 && py >= 0 && px < S && py < S)) continue;
            e = (py | 0) * S + (px | 0); d = depth[e];
            if (!(d < BIG)) continue;                                  // nothing was seen there: no measurement
            sdf = d - Z;
            if (sdf < -tr) continue;                                   // far behind what was seen: unknown
            t = sdf >= tr ? 32767 : sdf * K;
            g = own[e] * 3; e = q0 + i2; w = W[e]; iw = 1 / (w + 1); q3 = e * 3;
            D[e] = Math.floor((D[e] * w + t) * iw + 0.5);
            C[q3] = (C[q3] * w + C8[g]) * iw + 0.5; C[q3 + 1] = (C[q3 + 1] * w + C8[g + 1]) * iw + 0.5; C[q3 + 2] = (C[q3 + 2] * w + C8[g + 2]) * iw + 0.5;
            if (w < 255) W[e] = w + 1;
          }
        }
      }
      kz = 0;
      return true;
    }
    J.step = function (budget) {
      var end = now() + (budget == null ? 1e9 : budget), t0;
      if (!D && !J.done) {
        D = new Int16Array(NV); W = new Uint8Array(NV); C = new Uint8Array(NV * 3); depth = new Float32Array(S * S); own = new Int32Array(S * S);
        C8 = new Uint8Array(n * 3); for (var c8 = 0; c8 < n * 3; c8++) C8[c8] = CL[c8] <= 0 ? 0 : CL[c8] >= 1 ? 255 : Math.round(CL[c8] * 255);    // the Gaussians' colours as bytes
      }
      while (!J.done) {
        t0 = now();
        if (J.stage === 'depth') {
          if (v >= V) J.stage = 'extract';
          else { cam = cams[v]; if (draw(end)) J.stage = 'fuse'; J.ms.depth += now() - t0; }
        } else if (J.stage === 'fuse') {
          if (fuse(end)) { v++; J.view = v; J.stage = v < V ? 'depth' : 'extract'; }
          J.ms.fuse += now() - t0;
        } else {
          if (!ex) ex = extractor({ n: [n0, n1, n2], D: D, W: W, C: C, wmin: WMIN, lo: G.lo, h: h, ax: AX });
          var fin = ex.step(Math.max(0.5, end - now()));
          J.ms.extract = ex.ms;
          if (fin) {
            J.mesh = { pos: ex.pos, col: ex.col, tri: ex.tri, nV: ex.nV, nTri: ex.nTri, slab: ex.slab };
            J.bytes.mesh = ex.pos.byteLength + ex.col.byteLength + ex.tri.byteLength;
            D = W = C = C8 = depth = own = ex = null; J.stage = 'done'; J.done = true;
          }
        }
        // two thirds of a view's work is its fusion (measured); the extraction is the last twentieth
        J.progress = J.done ? 1 : J.stage === 'extract' ? 0.95 + 0.05 * (ex ? ex.progress : 0) :
          0.95 * (v + (J.stage === 'fuse' ? 0.35 + 0.65 * kz / n2 : 0.35 * gi / Math.max(1, n))) / Math.max(1, V);
        if (now() >= end) break;
      }
      return J.done;
    };
    J.run = function () { while (!J.step(1e9)); return J; };
    J.free = function () { D = W = C = C8 = depth = own = ex = null; J.mesh = null; };
    /* for tests and pictures: the depth map of view k, drawn at once -> { depth (BIG = nothing), own, S } */
    J.depthOf = function (k) {
      var d0 = depth, o0 = own, g0 = gi, c0 = cam, out;
      depth = new Float32Array(S * S); own = new Int32Array(S * S); gi = 0; cam = cams[k]; draw(BIG);
      out = { depth: depth, own: own, S: S }; depth = d0; own = o0; gi = g0; cam = c0;
      return out;
    };
    return J;
  }

  var api = { fuser: fuser, extractor: extractor, sizeGrid: sizeGrid, camera: camera, gap: gapOf, cubes: CUBES, checkCubes: checkCubes,
              extractorName: 'marching cubes', BIG: BIG };
  KT.meshTsdf = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
