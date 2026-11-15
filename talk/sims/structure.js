/* Knot-a-Surface talk — module `structure`: "Structure pays".
   The same scene stored two ways, side by side:
     a list   Gaussians in no particular order, as an unstructured 3DGS keeps them. EMULATED here, not
              trained: the same surfaces sampled at jittered places, nudged off the surface, each with a
              random in-plane orientation and size, then shuffled.
     ours     Gaussians derived from the spline charts, so each one has an address (chart, i, j).
   and four elementary operations run on both: read the order, find neighbours, build a mesh, cut the
   scene into parts. Everything shown is computed in the page; the counts and times are measured.
   Tool: WebGL (three.js), two viewports of one renderer. Standalone: talk/sims/structure.html.
   Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js, scenes.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var N = 16000, KNN = 12, KEDGE = 3;
  var PART = ['#4db6ac', '#f06292', '#9ccc65', '#7986cb', '#ffb74d', '#ba68c8', '#90a4ae', '#e57373', '#a1887f'];

  var SCENES = {
    bag:        { cap: '<b>the same Gaussians, stored two ways</b> · colour = position in memory', mode: 'order' },
    neighbours: { cap: '<b>nearest neighbours</b> · search the list, or read the next addresses', mode: 'near', hint: 'move over the scene to probe' },
    mesh:       { cap: '<b>a mesh</b> · join each Gaussian to the next ones', mode: 'mesh' },
    segments:   { cap: '<b>parts</b> · cluster the list, or read the chart', mode: 'parts' }
  };

  KT.register('structure', {
    title: 'Structure pays',
    blurb: 'One scene as an unordered list of Gaussians and as Gaussians derived from spline charts. Neighbours, a mesh and a ' +
           'segmentation are computed on both, live. The list is emulated from the same surfaces, not a trained 3DGS model.',
    tool: 'WebGL · three.js · one scene, two stores',
    scenes: [{ id: 'bag', label: 'Two stores' }, { id: 'neighbours', label: 'Neighbours' }, { id: 'mesh', label: 'Mesh' }, { id: 'segments', label: 'Parts' }],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 160), R = G.renderer(root, cam), scene = new T.Scene();
      var st = { scene: 'bag', sc: SCENES.bag, t: 0, probe: -1, auto: 0, autoT: 9, hover: false }, size = { w: 2, h: 2 }, need = true;
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);
      var tagL = ui.el('div', 'kt-side'), tagR = ui.el('div', 'kt-side'), bar = ui.el('div', 'kt-divide');
      root.appendChild(tagL); root.appendChild(tagR); root.appendChild(bar);

      /* ---------- the scene, sampled once ---------- */
      var def = KT.scenes.truck.build(), charts = def.charts, rnd = KT.rng(7);
      var ours = { pos: new Float32Array(N * 3), col: new Float32Array(N * 3), chart: new Uint8Array(N), n: 0 }, grids = [];
      var list = { pos: new Float32Array(N * 3), col: new Float32Array(N * 3), chart: new Uint8Array(N), n: 0 };
      var fR = G.splats(N), fL = G.splats(N); scene.add(fR.mesh); scene.add(fL.mesh);
      (function build() {
        var area = charts.map(function (c) {                 // surface area and side lengths of each chart, coarsely
          var us = [], vs = [], i, g = { pos: new Float32Array(144 * 3), su: new Float32Array(144 * 3), sv: new Float32Array(144 * 3) }, a = 0, lu = 0, lv = 0;
          for (i = 0; i < 12; i++) { us.push((i + 0.5) / 12); vs.push((i + 0.5) / 12); }
          NB.patchGrid(c.patch, us, vs, g);
          for (i = 0; i < 144; i++) {
            var o = i * 3, ax = g.su[o], ay = g.su[o + 1], az = g.su[o + 2], bx = g.sv[o], by = g.sv[o + 1], bz = g.sv[o + 2];
            var nl = Math.hypot(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), l = Math.hypot(ax, ay, az);
            a += nl / 144; lu += l / 144; lv += nl / (l || 1e-9) / 144;
          }
          return { a: Math.pow(a, 0.6), lu: lu, lv: lv };     // well under proportional: the big street and wall need not win everything
        });
        var tot = area.reduce(function (s, q) { return s + q.a; }, 0), n = 0, m = 0, tmp = [], spSum = 0;
        charts.forEach(function (c, ci) {
          var want = N * area[ci].a / tot, nb = Math.max(3, Math.round(Math.sqrt(want * area[ci].lv / area[ci].lu))), na = Math.max(3, Math.floor(want / nb));
          if (n + na * nb > N) nb = Math.floor((N - n) / na);
          var us = new Float32Array(na), vs = new Float32Array(nb), du = new Float32Array(na), dv = new Float32Array(nb), i, j, k;
          for (i = 0; i < na; i++) { us[i] = (i + 0.5) / na; du[i] = 1 / na; }
          for (j = 0; j < nb; j++) { vs[j] = (j + 0.5) / nb; dv[j] = 1 / nb; }
          var g = { pos: new Float32Array(na * nb * 3), su: new Float32Array(na * nb * 3), sv: new Float32Array(na * nb * 3), col: new Float32Array(na * nb * 3) };
          NB.patchGrid(c.patch, us, vs, g);
          grids.push({ base: n, na: na, nb: nb, wrapU: c.patch.periodicU, wrapV: c.patch.periodicV });
          G.writePlanoids(fR, n, g, na, nb, du, dv, 0.72);
          // where a chart's rows crowd together (the ends of the truck body) a derived Gaussian gets very long: cap it
          var len = [], cap2; for (k = n; k < n + na * nb; k++) len.push(Math.hypot(fR.axU[k * 3], fR.axU[k * 3 + 1], fR.axU[k * 3 + 2]), Math.hypot(fR.axV[k * 3], fR.axV[k * 3 + 1], fR.axV[k * 3 + 2]));
          cap2 = len.slice().sort(function (p, q) { return p - q; })[len.length >> 1] * 1.4;
          for (k = n; k < n + na * nb; k++) [fR.axU, fR.axV].forEach(function (A, w) { var l = len[(k - n) * 2 + w]; if (l > cap2) { A[k * 3] *= cap2 / l; A[k * 3 + 1] *= cap2 / l; A[k * 3 + 2] *= cap2 / l; } });
          ours.pos.set(g.pos, n * 3); ours.col.set(g.col, n * 3); ours.chart.fill(ci, n, n + na * nb);
          // the list: the same count, each sample moved inside its cell, off the surface a little, turned and resized
          for (j = 0; j < nb; j++) for (i = 0; i < na; i++) {
            var q = NB.patchPoint(c.patch, KT.clamp((i + rnd()) / na, 0, 0.9999), KT.clamp((j + rnd()) / nb, 0, 0.9999));
            var sx = q.su[0] / na, sy = q.su[1] / na, sz = q.su[2] / na, tx = q.sv[0] / nb, ty = q.sv[1] / nb, tz = q.sv[2] / nb;
            var nx = sy * tz - sz * ty, ny = sz * tx - sx * tz, nz = sx * ty - sy * tx, nl = Math.hypot(nx, ny, nz) || 1e-9, sp = Math.sqrt(nl);
            spSum += sp;
            var off = (rnd() - 0.5) * 0.5 * sp, an = rnd() * 6.283, ca = Math.cos(an), sa = Math.sin(an), f1 = sp * 0.72 * (0.6 + 0.7 * rnd()), f2 = sp * 0.72 * (0.6 + 0.7 * rnd());
            var sl = Math.hypot(sx, sy, sz) || 1e-9, e1 = [sx / sl, sy / sl, sz / sl], e2 = [(ny * e1[2] - nz * e1[1]) / nl, (nz * e1[0] - nx * e1[2]) / nl, (nx * e1[1] - ny * e1[0]) / nl];
            tmp.push({ p: [q.pos[0] + nx / nl * off, q.pos[1] + ny / nl * off, q.pos[2] + nz / nl * off], c: [q.col[0], q.col[1], q.col[2]], ch: ci,
                       u: [0, 1, 2].map(function (k) { return (e1[k] * ca + e2[k] * sa) * f1; }), v: [0, 1, 2].map(function (k) { return (e2[k] * ca - e1[k] * sa) * f2; }) });
          }
          n += na * nb;
        });
        ours.n = n;
        for (var i = tmp.length - 1; i > 0; i--) { var j = Math.floor(rnd() * (i + 1)), s = tmp[i]; tmp[i] = tmp[j]; tmp[j] = s; }   // no order at all
        tmp.forEach(function (q, k) {
          list.pos.set(q.p, k * 3); list.col.set(q.c, k * 3); list.chart[k] = q.ch;
          fL.center.set(q.p, k * 3); fL.axU.set(q.u, k * 3); fL.axV.set(q.v, k * 3);
        });
        list.n = m = tmp.length; list.sp = spSum / m;          // the typical gap between neighbours
        fR.setCount(n); fL.setCount(m);
      })();

      /* ---------- what a list needs before it can answer "who is near me": an index ---------- */
      var grid = (function () {
        var t0 = performance.now(), lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], i, k, P = list.pos;
        for (i = 0; i < list.n; i++) for (k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i * 3 + k]); hi[k] = Math.max(hi[k], P[i * 3 + k]); }
        var cell = list.sp * 2.6, dim = [0, 1, 2].map(function (q) { return Math.max(1, Math.ceil((hi[q] - lo[q]) / cell)); }), buckets = {};
        function key(x, y, z) { return (x * 73856093) ^ (y * 19349663) ^ (z * 83492791); }
        for (i = 0; i < list.n; i++) { var kk = key(Math.floor((P[i * 3] - lo[0]) / cell), Math.floor((P[i * 3 + 1] - lo[1]) / cell), Math.floor((P[i * 3 + 2] - lo[2]) / cell)); (buckets[kk] = buckets[kk] || []).push(i); }
        return { ms: performance.now() - t0, looked: 0,
          /* the k nearest list entries to (x,y,z), nearest first; counts the distances it had to compute */
          knn: function (x, y, z, k, skip) {
            var cx = Math.floor((x - lo[0]) / cell), cy = Math.floor((y - lo[1]) / cell), cz = Math.floor((z - lo[2]) / cell), best = [], r, a, b, c, bk, j, d;
            this.looked = 0;
            for (r = 0; r < Math.max(dim[0], dim[1], dim[2]) + 1; r++) {
              for (a = -r; a <= r; a++) for (b = -r; b <= r; b++) for (c = -r; c <= r; c++) {
                if (Math.max(Math.abs(a), Math.abs(b), Math.abs(c)) !== r) continue;
                bk = buckets[key(cx + a, cy + b, cz + c)]; if (!bk) continue;
                for (j = 0; j < bk.length; j++) { if (bk[j] === skip) continue; d = Math.pow(P[bk[j] * 3] - x, 2) + Math.pow(P[bk[j] * 3 + 1] - y, 2) + Math.pow(P[bk[j] * 3 + 2] - z, 2); best.push([d, bk[j]]); this.looked++; }
              }
              if (best.length >= k) { best.sort(function (p, q) { return p[0] - q[0]; }); best.length = Math.min(best.length, k * 3); if (Math.sqrt(best[k - 1][0]) < r * cell) break; }
            }
            return best.slice(0, k).map(function (q) { return q[1]; });
          } };
      })();

      /* ---------- order: colour by position in memory ---------- */
      function ramp(t) { var a = t * 5; return [KT.clamp(Math.abs(a - 3) - 1, 0, 1) * 0.9 + 0.1, KT.clamp(2 - Math.abs(a - 2), 0, 1) * 0.85 + 0.1, KT.clamp(2 - Math.abs(a - 4), 0, 1) * 0.9 + 0.1]; }
      /* ---------- mesh: the list joins nearest neighbours; ours joins next addresses ---------- */
      var mesh = null, eOk, eBad, eOurs;
      function buildMesh() {
        var t0 = performance.now(), ok = [], bad = [], seen = {}, i, P = list.pos, nb, j, a, b;
        for (i = 0; i < list.n; i++) {
          nb = grid.knn(P[i * 3], P[i * 3 + 1], P[i * 3 + 2], KEDGE, i);
          for (j = 0; j < nb.length; j++) { a = Math.min(i, nb[j]); b = Math.max(i, nb[j]); if (seen[a * 65536 + b]) continue; seen[a * 65536 + b] = 1; (list.chart[a] === list.chart[b] ? ok : bad).push(a, b); }
        }
        var ms = performance.now() - t0 + grid.ms, t1 = performance.now(), ed = [], tri = 0;
        grids.forEach(function (g) {
          for (var y = 0; y < g.nb; y++) for (var x = 0; x < g.na; x++) {
            var k = g.base + y * g.na + x, rx = x + 1 < g.na ? x + 1 : g.wrapU ? 0 : -1, ry = y + 1 < g.nb ? y + 1 : g.wrapV ? 0 : -1;
            if (rx >= 0) ed.push(k, g.base + y * g.na + rx); if (ry >= 0) ed.push(k, g.base + ry * g.na + x); if (rx >= 0 && ry >= 0) tri += 2;
          }
        });
        mesh = { ok: ok, bad: bad, ms: ms, ed: ed, tri: tri, ms2: performance.now() - t1 };
        eOk = lines(mesh.ok, list.pos, C.text, 0.6); eBad = lines(mesh.bad, list.pos, C.err, 0.95); eOurs = lines(mesh.ed, ours.pos, C.surf, 0.8);
      }
      function lines(idx, P, colour, opacity) {
        var l = G.lines(idx.length, colour, opacity), k;
        for (k = 0; k < idx.length; k++) { l.arr[k * 3] = P[idx[k] * 3]; l.arr[k * 3 + 1] = P[idx[k] * 3 + 1]; l.arr[k * 3 + 2] = P[idx[k] * 3 + 2]; }
        l.set(idx.length); l.obj.visible = false; scene.add(l.obj); return l;
      }
      /* ---------- parts: k-means on position for the list; the chart for ours ---------- */
      var parts = null;
      function buildParts() {
        var t0 = performance.now(), k = charts.length, cen = [], lab = new Uint8Array(list.n), P = list.pos, i, c, it, r2 = KT.rng(3);
        for (c = 0; c < k; c++) { i = Math.floor(r2() * list.n); cen.push([P[i * 3], P[i * 3 + 1], P[i * 3 + 2]]); }
        for (it = 0; it < 14; it++) {
          var sum = cen.map(function () { return [0, 0, 0, 0]; });
          for (i = 0; i < list.n; i++) {
            var bd = 1e18, bc = 0;
            for (c = 0; c < k; c++) { var d = Math.pow(P[i * 3] - cen[c][0], 2) + Math.pow(P[i * 3 + 1] - cen[c][1], 2) + Math.pow(P[i * 3 + 2] - cen[c][2], 2); if (d < bd) { bd = d; bc = c; } }
            lab[i] = bc; sum[bc][0] += P[i * 3]; sum[bc][1] += P[i * 3 + 1]; sum[bc][2] += P[i * 3 + 2]; sum[bc][3]++;
          }
          sum.forEach(function (s, q) { if (s[3]) cen[q] = [s[0] / s[3], s[1] / s[3], s[2] / s[3]]; });
        }
        var right = 0;                                          // purity: how much of each cluster is its largest true part
        for (c = 0; c < k; c++) { var cnt = new Int32Array(k), mx = 0; for (i = 0; i < list.n; i++) if (lab[i] === c) cnt[list.chart[i]]++; for (i = 0; i < k; i++) mx = Math.max(mx, cnt[i]); right += mx; }
        parts = { lab: lab, purity: right / list.n, ms: performance.now() - t0 };
      }

      /* ---------- neighbours of a probe ---------- */
      var star = { L: G.lines(KNN * 2, C.text, 0.9), R: G.lines(KNN * 2, C.text, 0.9) }, near = { L: [], R: [], wrong: 0, looked: 0 };
      star.L.mat.depthTest = star.R.mat.depthTest = false; scene.add(star.L.obj); scene.add(star.R.obj);
      function dots(colour) {                                   // the neighbours themselves, as dots that stay visible
        var g = new T.BufferGeometry(); g.setAttribute('position', new T.BufferAttribute(new Float32Array(KNN * 3), 3));
        var o = new T.Points(g, new T.PointsMaterial({ color: colour, size: 9, sizeAttenuation: false, depthTest: false })); o.frustumCulled = false; o.visible = false; scene.add(o); return o;
      }
      var dot = { ok: dots(C.surf), bad: dots(C.err), R: dots(C.surf) };
      function setDots(o, P, idx) { var a = o.geometry.attributes.position; idx.forEach(function (j, k) { a.array[k * 3] = P[j * 3]; a.array[k * 3 + 1] = P[j * 3 + 1]; a.array[k * 3 + 2] = P[j * 3 + 2]; }); a.needsUpdate = true; o.geometry.setDrawRange(0, idx.length); }
      var spots = (function () {                               // places where another surface comes close: where a list gets confused
        var out = [], i, nb, w;
        for (i = 0; i < ours.n; i += 7) {
          if (charts[ours.chart[i]].name === 'street' || charts[ours.chart[i]].name === 'wall') continue;
          nb = grid.knn(ours.pos[i * 3], ours.pos[i * 3 + 1], ours.pos[i * 3 + 2], KNN, -1); w = nb.filter(function (q) { return list.chart[q] !== ours.chart[i]; }).length;
          if (w >= 3 && w <= 9) out.push(i);
        }
        return out.filter(function (q, k) { return k % Math.max(1, Math.floor(out.length / 9)) === 0; });
      })();
      function probe(i) {
        st.probe = i; var g = grids[ours.chart[i]], k = i - g.base, x = k % g.na, y = Math.floor(k / g.na), P = ours.pos, a, b, q, m = 0;
        near.L = grid.knn(P[i * 3], P[i * 3 + 1], P[i * 3 + 2], KNN, -1); near.looked = grid.looked;
        near.wrong = near.L.filter(function (j) { return list.chart[j] !== ours.chart[i]; }).length;
        near.R = [];
        [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1], [2, 0], [-2, 0], [0, 2], [0, -2]].forEach(function (d) {
          a = x + d[0]; b = y + d[1];
          if (g.wrapU) a = (a + g.na) % g.na; if (g.wrapV) b = (b + g.nb) % g.nb;
          if (a >= 0 && a < g.na && b >= 0 && b < g.nb) near.R.push(g.base + b * g.na + a);
        });
        [['L', list.pos, near.L], ['R', ours.pos, near.R]].forEach(function (s) {
          m = 0;
          s[2].forEach(function (j) { for (q = 0; q < 3; q++) star[s[0]].arr[m++] = P[i * 3 + q]; for (q = 0; q < 3; q++) star[s[0]].arr[m++] = s[1][j * 3 + q]; });
          star[s[0]].set(m / 3);
        });
        setDots(dot.ok, list.pos, near.L.filter(function (j) { return list.chart[j] === ours.chart[i]; }));
        setDots(dot.bad, list.pos, near.L.filter(function (j) { return list.chart[j] !== ours.chart[i]; })); setDots(dot.R, ours.pos, near.R);
        need = true;
      }

      /* ---------- colours for the current operation ---------- */
      function paint() {
        if (st.sc.mode === 'mesh' && !mesh) buildMesh();
        if (st.sc.mode === 'parts' && !parts) buildParts();
        var mode = st.sc.mode, i, k, c, L = fL.color, Rr = fR.color, dim = mode === 'near' ? 0.6 : mode === 'mesh' ? 0.16 : 1, set = {};
        for (i = 0; i < list.n; i++) {
          c = mode === 'order' ? ramp(i / list.n) : mode === 'parts' ? G.rgb(PART[parts.lab[i] % PART.length]) : [list.col[i * 3] * dim, list.col[i * 3 + 1] * dim, list.col[i * 3 + 2] * dim];
          L[i * 3] = c[0]; L[i * 3 + 1] = c[1]; L[i * 3 + 2] = c[2];
        }
        for (i = 0; i < ours.n; i++) {
          c = mode === 'order' ? ramp(i / ours.n) : mode === 'parts' ? G.rgb(PART[ours.chart[i] % PART.length]) : [ours.col[i * 3] * dim, ours.col[i * 3 + 1] * dim, ours.col[i * 3 + 2] * dim];
          Rr[i * 3] = c[0]; Rr[i * 3 + 1] = c[1]; Rr[i * 3 + 2] = c[2];
        }
        if (mode === 'near' && st.probe >= 0) {
          var bad = G.rgb(C.err), good = G.rgb(C.surf);
          near.L.forEach(function (j) { c = list.chart[j] !== ours.chart[st.probe] ? bad : good; L[j * 3] = c[0]; L[j * 3 + 1] = c[1]; L[j * 3 + 2] = c[2]; });
          near.R.forEach(function (j) { Rr[j * 3] = good[0]; Rr[j * 3 + 1] = good[1]; Rr[j * 3 + 2] = good[2]; });
        }
        fL.commit(list.n); fR.commit(ours.n);
        var flat = mode === 'order' || mode === 'parts' ? 0.55 : 0;
        fL.uniforms.uFlat.value = fR.uniforms.uFlat.value = flat;
        var n = list.n.toLocaleString('en-US');
        tagL.innerHTML = '<b>a list</b> · ' + n + ' Gaussians, unordered <i>(emulated)</i><br>' + ({
          order: 'neighbours in memory are strangers in space',
          near: 'computed <em>' + near.looked + '</em> distances through an index · <em class="bad">' + near.wrong + ' of ' + KNN + '</em> lie on another surface',
          mesh: mesh && 'join the ' + KEDGE + ' nearest: ' + ((mesh.ok.length + mesh.bad.length) / 2).toLocaleString('en-US') + ' edges, <em class="bad">' + (mesh.bad.length / 2).toLocaleString('en-US') + '</em> bridge two surfaces · ' + mesh.ms.toFixed(0) + ' ms of searching',
          parts: parts && 'k-means into ' + charts.length + ' clusters · <em class="bad">' + Math.round(parts.purity * 100) + ' %</em> of the Gaussians sit with their own part · ' + parts.ms.toFixed(0) + ' ms'
        })[mode];
        tagR.innerHTML = '<b>spline charts</b> · the same ' + ours.n.toLocaleString('en-US') + ', each with an address<br>' + ({
          order: 'memory order is surface order: chart by chart, row by row',
          near: 'read <em>' + near.R.length + '</em> addresses, computed <em>0</em> distances · <em>0</em> on another surface',
          mesh: mesh && 'join (i, j) to (i+1, j) and (i, j+1): ' + mesh.tri.toLocaleString('en-US') + ' triangles · ' + (mesh.ms2 < 1 ? 'under 1' : mesh.ms2.toFixed(0)) + ' ms, no search',
          parts: charts.length + ' parts, by construction: the chart is the label'
        })[mode];
      }

      /* ---------- camera, pointer ---------- */
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 2, maxRadius: 40 });
      var ray = new T.Raycaster(), ndc = new T.Vector2(), rects = [[0, 0, 1, 1], [0, 0, 1, 1]];
      R.canvas.addEventListener('pointermove', function (e) {
        if (st.sc.mode !== 'near' || e.pointerType === 'touch') return;
        var r = R.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = r.height - (e.clientY - r.top), q = x >= rects[1][0] && y >= rects[1][1] && x < rects[1][0] + rects[1][2] && y < rects[1][1] + rects[1][3] ? rects[1] : rects[0];
        cam.aspect = q[2] / q[3]; cam.updateProjectionMatrix();
        ndc.set((x - q[0]) / q[2] * 2 - 1, (y - q[1]) / q[3] * 2 - 1); ray.setFromCamera(ndc, cam);
        var o = ray.ray.origin, d = ray.ray.direction, P = ours.pos, best = -1, bt = 1e9, i, ex, ey, ez, t;
        for (i = 0; i < ours.n; i++) { ex = P[i * 3] - o.x; ey = P[i * 3 + 1] - o.y; ez = P[i * 3 + 2] - o.z; t = ex * d.x + ey * d.y + ez * d.z; if (t > 0 && t < bt && ex * ex + ey * ey + ez * ez - t * t < 0.012) { bt = t; best = i; } }
        if (best >= 0 && best !== st.probe) { st.hover = true; probe(best); }
      });
      R.canvas.addEventListener('pointerleave', function () { st.hover = false; st.autoT = 9; });

      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.autoT = 9; st.hover = false;
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        orbit.touched = false; orbit.to(view(), false); need = true;
      }
      function view() { var v = Object.assign({}, def.views.wide), a = rects[0][2] / rects[0][3]; v.r *= KT.clamp(1.25 / a, 0.62, 1.7); return v; }

      env.loop(function (t, dt) {
        st.t += dt;
        if (st.sc.mode === 'near' && !st.hover && spots.length && (st.autoT += dt) > 2.4) { st.autoT = 0; st.auto = (st.auto + 1) % spots.length; probe(spots[st.auto]); }
        if (!orbit.touched) {
          var home = view(), pr = st.sc.mode === 'near' && st.probe >= 0 ? st.probe : -1;      // probing: move in on the probe
          orbit.to(pr < 0 ? home : { r: home.r * 0.5, tx: ours.pos[pr * 3], ty: ours.pos[pr * 3 + 1], tz: ours.pos[pr * 3 + 2] });
          orbit.goal.th = def.views.wide.th + 0.22 * Math.sin(t * 0.21);
        }
        orbit.update(dt, 2.2);
        if (need) { need = false; paint(); }
        var rr = R.renderer, mode = st.sc.mode, k;
        rr.setScissorTest(true);
        for (k = 0; k < 2; k++) {
          var q = rects[k]; rr.setViewport(q[0], q[1], q[2], q[3]); rr.setScissor(q[0], q[1], q[2], q[3]);
          cam.aspect = q[2] / q[3]; cam.updateProjectionMatrix();
          fL.mesh.visible = k === 0; fR.mesh.visible = k === 1;
          if (mesh) { eOk.obj.visible = eBad.obj.visible = mode === 'mesh' && k === 0; eOurs.obj.visible = mode === 'mesh' && k === 1; }
          star.L.obj.visible = dot.ok.visible = dot.bad.visible = mode === 'near' && k === 0 && st.probe >= 0; star.R.obj.visible = dot.R.visible = mode === 'near' && k === 1 && st.probe >= 0;
          G.draw(R, scene, cam);
        }
        rr.setScissorTest(false);
      });

      if (!document.getElementById('kt-structure-css')) {
        var css = document.createElement('style'); css.id = 'kt-structure-css';
        css.textContent =
          '.kt-sim-structure .kt-side{position:absolute;z-index:4;pointer-events:none;font:500 11px/1.5 var(--f-mono);letter-spacing:.03em;color:var(--dim);' +
          'padding:8px 11px;background:var(--panel);border:1px solid var(--line);border-radius:8px;max-width:calc(100% - 20px)}' +
          '.kt-sim-structure .kt-side b{color:var(--text);font-weight:600;letter-spacing:.06em;text-transform:uppercase}' +
          '.kt-sim-structure .kt-side i{font-style:normal;color:var(--mute)}.kt-sim-structure .kt-side em{font-style:normal;color:var(--ok);font-weight:600}' +
          '.kt-sim-structure .kt-side em.bad{color:var(--err)}' +
          '.kt-sim-structure .kt-divide{position:absolute;z-index:3;background:var(--line2);pointer-events:none}' +
          '.kt-sim-structure.kt-embed .kt-hint{display:none}.kt-sim-structure .kt-gear{display:none}' +
          '@media (max-width:640px){.kt-sim-structure .kt-side{font-size:9.5px;padding:6px 8px}}';
        document.head.appendChild(css);
      }

      setScene(env.opts.scene || 'bag');
      return {
        setScene: setScene, orbit: orbit, stats: function () { return { near: near, mesh: mesh && { ok: mesh.ok.length / 2, bad: mesh.bad.length / 2, tri: mesh.tri, ms: mesh.ms }, purity: parts && parts.purity, spots: spots.length }; },
        resize: function (w, h) {
          R.resize(w, h); size.w = w; size.h = h;
          var wide = w / h > 1.45, s;                           // side by side on a wide stage, stacked on a tall one
          rects = wide ? [[0, 0, w / 2, h], [w / 2, 0, w / 2, h]] : [[0, h / 2, w, h / 2], [0, 0, w, h / 2]];
          s = bar.style; s.left = wide ? w / 2 + 'px' : '0'; s.top = wide ? '0' : h / 2 + 'px'; s.width = wide ? '1px' : '100%'; s.height = wide ? '100%' : '1px';
          tagL.style.cssText = wide ? 'left:10px;bottom:10px;max-width:' + (w / 2 - 20) + 'px' : 'left:10px;top:' + (h / 2 - 10) + 'px;transform:translateY(-100%)';
          tagR.style.cssText = wide ? 'left:' + (w / 2 + 10) + 'px;bottom:10px;max-width:' + (w / 2 - 20) + 'px' : 'left:10px;bottom:10px';
          if (!orbit.touched) orbit.to(view(), true);
          need = true;
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
