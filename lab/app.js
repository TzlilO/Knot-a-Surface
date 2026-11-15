/* Knot-a-Surface lab — the interactive figure (lab.html).
   Objects come from lab/objects.js as B-spline charts; lab/render.js draws the Gaussians derived from
   them. This file: view-adaptive sampling under a fixed Gaussian budget (as in talk/sims/manifold.js),
   control features (colour and, when baked, DINOv3 features in lab/features/<scene>.js), and the tools:
     sculpt  drag a surface; control points inside the brush follow
     select  click a surface; control points with a similar feature (or of the same chart) are selected,
             and the selection can be inflated, lifted, scaled, tinted or hidden
   Depends on vendor/three.min.js, talk/nurbs.js, talk/kit.js (gestures), talk/gl.js (orbit, planoids). */
(function () {
  'use strict';
  var T = window.THREE, KT = window.KnotTalk, NB = KT.nurbs, G = KT.gl, KL = window.KnotLab;
  var CAP = 420000, FLOOR = 0.10, PER_CP = 16, TAU = 6.283185307, PAGE = [0.969, 0.961, 0.941];
  var MODES = [
    { v: 'beauty', name: 'Rendering' }, { v: 'albedo', name: 'Appearance' }, { v: 'depth', name: 'Depth' }, { v: 'normal', name: 'Normals' },
    { v: 'gauss', name: 'Gaussians' }, { v: 'charts', name: 'Charts' }, { v: 'feat', name: 'DINOv3 features' }, { v: 'sim', name: 'Similarity' }];
  var ROLL = 0.025, SLANT = 0.2;
  var TINTS = [['cobalt', [0.10, 0.22, 0.62]], ['oxblood', [0.50, 0.08, 0.09]], ['sage', [0.44, 0.55, 0.40]], ['saffron', [0.90, 0.62, 0.10]], ['ink', [0.07, 0.07, 0.09]], ['bone', [0.92, 0.89, 0.82]]];
  function $(id) { return document.getElementById(id); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function wrap(x, m) { return ((x % m) + m) % m; }
  function fmt(n) { return Math.round(n).toLocaleString('en-US'); }
  var query = {}; location.search.slice(1).split('&').forEach(function (kv) { var p = kv.split('='); if (p[0]) query[p[0]] = decodeURIComponent(p[1] || ''); });

  var stage = $('stage'), small = window.matchMedia('(max-width: 900px)').matches;
  var cam = new T.PerspectiveCamera(36, 1, 0.1, 160), R;
  try { R = KL.renderer(stage, cam, { capacity: CAP, page: PAGE }); }
  catch (e) { stage.classList.add('dead'); stage.appendChild(el('p', 'lab-dead', 'This figure needs WebGL 2, which this browser does not provide.')); return; }
  var field = { center: R.center, axU: R.axU, axV: R.axV, color: R.color, capacity: CAP };
  var st = { scn: '', mode: 0, split: false, roll: 0, budget: +query.budget || (small ? 90000 : 200000), sampling: 'adaptive', k: 0.72,
             tool: 'orbit', brush: 0.5, by: 'feature', thr: 0.55, nets: false, hide: false, drawn: 0 };
  var def = null, charts = [], totalCP = 0, feat = null, need = true, camKey = '', lastAlloc = 0, zr = [1, 30];
  var sel = null;               // { n, cen:[x,y,z], seed, edit:{inflate,lift,scale,tint,amount} } while a selection exists

  /* ---------- per-chart helpers ---------- */
  function greville(p, n, knots, periodic) {
    var g = new Float32Array(n), i, r, s;
    for (i = 0; i < n; i++) { s = 0; for (r = 1; r <= p; r++) s += knots[i + r]; g[i] = periodic ? wrap(s / p, 1) : s / p; }
    return g;
  }
  /* B-spline blend of nch channels per control point (src) on the grid us x vs -> out, row = v index */
  function evalCh(patch, us, vs, src, nch, out) {
    var p = patch.p, w = p + 1, nu = patch.nu, nv = patch.nv, na = us.length, nb = vs.length, a, b, r, s, k, o, q, ww;
    var TU = NB.axisTable(patch.Uk, patch.mu, p, us), TV = NB.axisTable(patch.Vk, patch.mv, p, vs);
    for (b = 0; b < nb; b++) for (a = 0; a < na; a++) {
      o = (b * na + a) * nch; for (k = 0; k < nch; k++) out[o + k] = 0;
      for (s = 0; s < w; s++) for (r = 0; r < w; r++) {
        ww = TV.N[b * w + s] * TU.N[a * w + r]; q = (((TV.span[b] - p + s) % nv) * nu + (TU.span[a] - p + r) % nu) * nch;
        for (k = 0; k < nch; k++) out[o + k] += ww * src[q + k];
      }
    }
  }
  /* coarse cells: where each chart is, which way it faces, how big (drives the sampling) */
  function coarse(c) {
    var gu = c.gu, gv = c.gv, us = new Float32Array(gu), vs = new Float32Array(gv), i, n = gu * gv, o;
    for (i = 0; i < gu; i++) us[i] = (i + 0.5) / gu;
    for (i = 0; i < gv; i++) vs[i] = (i + 0.5) / gv;
    var g = { pos: new Float32Array(n * 3), su: new Float32Array(n * 3), sv: new Float32Array(n * 3) };
    NB.patchGrid(c.patch, us, vs, g);
    c.cpos = g.pos; c.cnrm = new Float32Array(n * 3); c.carea = new Float32Array(n);
    c.clu = new Float32Array(n); c.clv = new Float32Array(n); c.area = 0;
    for (i = 0; i < n; i++) {
      o = i * 3;
      var ax = g.su[o], ay = g.su[o + 1], az = g.su[o + 2], bx = g.sv[o], by = g.sv[o + 1], bz = g.sv[o + 2];
      var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, nl = Math.hypot(nx, ny, nz) || 1e-9;
      c.cnrm[o] = nx / nl; c.cnrm[o + 1] = ny / nl; c.cnrm[o + 2] = nz / nl;
      c.carea[i] = nl / n; c.area += nl / n;
      c.clu[i] = Math.hypot(ax, ay, az); c.clv[i] = nl / (c.clu[i] || 1e-9);
    }
    if (c.closed && c.ref) {
      var k = Math.floor(n * 0.37), rf = c.ref(us[k % gu], vs[Math.floor(k / gu)], [g.pos[k * 3], g.pos[k * 3 + 1], g.pos[k * 3 + 2]]);
      c.flip = ((g.pos[k * 3] - rf[0]) * c.cnrm[k * 3] + (g.pos[k * 3 + 1] - rf[1]) * c.cnrm[k * 3 + 1] + (g.pos[k * 3 + 2] - rf[2]) * c.cnrm[k * 3 + 2]) < 0 ? -1 : 1;
    }
  }

  /* ---------- allocation: importance -> per-tile counts -> warped (u,v) lattices ----------
     st.sampling: 'even' = by surface area · 'adaptive' = by what the camera sees, with a floor everywhere ·
     'locked' = the whole budget inside the view frustum, shared by area on screen (nothing outside it) */
  var M = new T.Matrix4();
  function invCdf(m, n, out, dOut) {
    var g = m.length, cdf = new Float64Array(g + 1), i, tot = 0, prev = 0;
    for (i = 0; i < g; i++) { tot += m[i]; cdf[i + 1] = tot; }
    function at(q) { var x = q * tot, k = 0; while (k < g - 1 && cdf[k + 1] < x) k++; return (k + (x - cdf[k]) / (m[k] || 1e-12)) / g; }
    for (i = 0; i < n; i++) { var e = at((i + 1) / n); out[i] = at((i + 0.5) / n); dOut[i] = e - prev; prev = e; }
  }
  function allocate() {
    cam.updateMatrixWorld(); M.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    var e = M.elements, cx = cam.position.x, cy = cam.position.y, cz = cam.position.z, totA = 0, totW = 0, z0 = 1e9, z1 = 0;
    var locked = st.sampling === 'locked', lim = locked ? 1.04 : 1.12;
    charts.forEach(function (c) { totA += c.area; });
    charts.forEach(function (c) {
      var n = c.gu * c.gv, i, o, w; c.w = c.w || new Float32Array(n);
      for (i = 0; i < n; i++) {
        o = i * 3; var x = c.cpos[o], y = c.cpos[o + 1], z = c.cpos[o + 2];
        var cw = e[3] * x + e[7] * y + e[11] * z + e[15], px = e[0] * x + e[4] * y + e[8] * z + e[12], py = e[1] * x + e[5] * y + e[9] * z + e[13];
        w = 0;
        if (cw > 0.05 && Math.abs(px) < cw * lim && Math.abs(py) < cw * lim) {
          var dx = cx - x, dy = cy - y, dz = cz - z, d2 = dx * dx + dy * dy + dz * dz;
          var cs = (dx * c.cnrm[o] + dy * c.cnrm[o + 1] + dz * c.cnrm[o + 2]) / Math.sqrt(d2) * c.flip;
          cs = c.closed ? (cs > -0.15 ? 1 : 0) : Math.max(0.25, Math.abs(cs));      // the far side of a closed chart asks for nothing
          w = c.carea[i] * cs / d2 * (locked && !c.lock ? 1 : c.gain);
          if (!c.lock) { if (cw < z0) z0 = cw; if (cw > z1) z1 = cw; }
        }
        c.w[i] = w; totW += w;
      }
    });
    var goal = st.budget * boost, total = sizeTiles(1);     // laid for a little more than the budget: back faces drop out when written
    if (Math.abs(total - goal) > 0.001 * goal) sizeTiles(goal / total);
    if (z1 > z0) zr = [z0 * 0.85, z1 * 1.5];

    function sizeTiles(scale) {
      var sum = 0;
      charts.forEach(function (c) {
        var gu = c.gu, cu = gu / c.tu, cv = c.gv / c.tv;
        c.count = 0;
        c.tiles.forEach(function (tl) {
          var mu = new Float64Array(cu), mv = new Float64Array(cv), share = 0, lu = 0, lv = 0, a, b, i, imp;
          for (b = 0; b < cv; b++) for (a = 0; a < cu; a++) {
            i = (tl.iv * cv + b) * gu + tl.iu * cu + a;
            imp = locked ? c.w[i] / (totW || 1) : st.sampling === 'adaptive' ? FLOOR * c.carea[i] / totA + (1 - FLOOR) * c.w[i] / (totW || 1) : c.carea[i] / totA;
            share += imp; mu[a] += imp; mv[b] += imp; lu += imp * c.clu[i] / c.tu; lv += imp * c.clv[i] / c.tv;
          }
          if (!share) { tl.na = tl.nb = 0; return; }             // locked: a tile outside the view gets nothing
          var N = Math.max(4, goal * share * scale), ratio = lu / (lv || 1e-12);
          tl.nb = Math.max(c.minB, Math.round(Math.sqrt(N / ratio))); tl.na = Math.max(c.minA, Math.round(N / tl.nb));
          for (a = 0; a < cu; a++) mu[a] += share / cu * (locked ? 0.12 : 0.3);
          for (b = 0; b < cv; b++) mv[b] += share / cv * (locked ? 0.12 : 0.3);
          tl.us = new Float32Array(tl.na); tl.du = new Float32Array(tl.na); tl.vs = new Float32Array(tl.nb); tl.dv = new Float32Array(tl.nb);
          invCdf(mu, tl.na, tl.us, tl.du); invCdf(mv, tl.nb, tl.vs, tl.dv);
          for (a = 0; a < tl.na; a++) { tl.us[a] = (tl.iu + tl.us[a]) / c.tu; tl.du[a] /= c.tu; }
          for (b = 0; b < tl.nb; b++) { tl.vs[b] = (tl.iv + tl.vs[b]) / c.tv; tl.dv[b] /= c.tv; }
          c.count += tl.na * tl.nb;
        });
        sum += c.count;
      });
      return sum;
    }
  }

  var sg = { pos: new Float32Array(CAP * 3), su: new Float32Array(CAP * 3), sv: new Float32Array(CAP * 3), col: new Float32Array(CAP * 3) };
  var ch = new Float32Array(60000 * 5);
  /* derive every Gaussian again: centre, tangent axes, colour, and the control features riding along.
     Back faces of closed charts are never written: they cannot be seen in this frame, so they should not
     spend the budget. What is written then falls short of it, so the next lattices are laid larger (boost). */
  var boost = 1, kept = new Int32Array(60000);
  function resample() {
    var n = emit();
    boost = KT.clamp(boost * st.budget / Math.max(1, n), 1, 4);
    if (Math.abs(n - st.budget) > 0.1 * st.budget) n = emit();
    st.drawn = n; R.commit(n); R.depthRange(zr[0], zr[1]);
    paintNumbers();
  }
  function emit() {
    allocate();
    var n = 0, meta = R.meta, aux = R.aux, eye = { x: cam.position.x, y: cam.position.y, z: cam.position.z, flip: 1, min: -0.12 };
    charts.forEach(function (c, ci) {
      var first = n; eye.flip = c.flip;
      c.tiles.forEach(function (tl) {
        var m = tl.na * tl.nb, n0 = n, a, b, i, j, o;
        if (!m || m > CAP) return;
        if (kept.length < m) kept = new Int32Array(m);
        NB.patchGrid(c.patch, tl.us, tl.vs, sg);
        n = G.writePlanoids(field, n, sg, tl.na, tl.nb, tl.du, tl.dv, st.k, null, c.closed ? eye : null, kept);
        if (ch.length < m * 5) ch = new Float32Array(m * 5);
        evalCh(c.patch, tl.us, tl.vs, c.A, 5, ch);
        for (j = n0; j < n; j++) {
          i = kept[j - n0]; a = i % tl.na; b = (i - a) / tl.na; o = j * 4;
          aux[o] = ch[i * 5]; aux[o + 1] = ch[i * 5 + 1]; aux[o + 2] = ch[i * 5 + 2]; aux[o + 3] = ch[i * 5 + 3];
          meta[o] = ci; meta[o + 1] = ch[i * 5 + 4]; meta[o + 2] = tl.us[a]; meta[o + 3] = tl.vs[b];
        }
      });
      c.count = n - first;
    });
    return n;
  }

  /* ---------- scene ---------- */
  var netL = null, pts = null, ptPos = null;
  function loadScene(key) {
    var s = KL.scenes[key]; if (!s) return;
    st.scn = key; def = s.build(); charts = def.charts.slice(0, R.maxCharts); sel = null; feat = null;
    totalCP = 0; var segs = 0;
    charts.forEach(function (c) {
      var p = c.patch, n = p.nu * p.nv; totalCP += n;
      segs += (p.periodicU ? p.nu : p.nu - 1) * p.nv + (p.periodicV ? p.nv : p.nv - 1) * p.nu;
      c.mat = c.mat || { rough: 0.6, metal: 0 }; c.baseC = new Float32Array(p.C);
      c.A = new Float32Array(n * 5);                // per control point: feature colour (3), similarity, selection
      c.gU = greville(p.p, p.nu, p.Uk, p.periodicU); c.gV = greville(p.p, p.nv, p.Vk, p.periodicV);
      coarse(c);
    });
    var rho = charts.map(function (c) { return c.patch.nu * c.patch.nv / c.area; }), rmax = Math.max.apply(null, rho.filter(function (r, i) { return !charts[i].lock; }));
    charts.forEach(function (c, i) { c.gain = c.lock ? 0.015 : KT.clamp(Math.pow(rho[i] / rmax, 0.35), 0.3, 1); });   // the flat floor needs very little
    R.setCharts(charts.map(function (c) { return { mat: c.mat, rgb: G.rgb(c.hex), ground: c.lock }; }));
    R.light(def.views.wide.th - 1.0, 0.92);                 // from the camera's upper left: shadows fall where they can be seen
    var tg = def.rig.target, rad = def.radius || 3;
    R.frame(tg[0], tg[1], tg[2], rad, rad * 1.7, rad * 3.2);
    [netL && netL.obj, pts].forEach(function (o) { if (o) { R.overlay.remove(o); o.geometry.dispose(); o.material.dispose(); } });
    netL = G.lines(segs * 2, KT.color.ctrl, 0.55); netL.mat.depthTest = false; R.overlay.add(netL.obj);
    var pg = new T.BufferGeometry(); ptPos = new Float32Array(totalCP * 3); pg.setAttribute('position', new T.BufferAttribute(ptPos, 3));
    pts = new T.Points(pg, new T.PointsMaterial({ color: KT.color.ctrl, size: totalCP > 3000 ? 3 : 5, sizeAttenuation: false, depthTest: false }));
    pts.frustumCulled = false; R.overlay.add(pts);
    updateNets(); showNets();
    orbit.touched = false; orbit.to(home(), true);
    [].forEach.call($('scenes').children, function (b) { b.classList.toggle('on', b._k === key); });
    loadFeatures(key); paintTable(); paintSel(); paintCaption(); need = true;
  }
  /* the scene's own viewing direction, at the distance that frames the subject in this stage */
  function home() {
    var v = Object.assign({}, def.views.wide), a = stage.clientWidth / Math.max(1, stage.clientHeight), half = cam.fov * Math.PI / 360, tg = def.rig.target;
    v.r = (def.radius || 3) * 0.92 / Math.sin(Math.atan(Math.tan(half) * Math.min(1, a)));
    v.tx = tg[0]; v.ty = tg[1]; v.tz = tg[2];
    return v;
  }
  function updateNets() {
    var A = netL.arr, n = 0, q = 0;
    charts.forEach(function (c) {
      var p = c.patch, P = p.P, nu = p.nu, nv = p.nv, i, j, o, o2;
      c.ptOff = q / 3;
      if (c.lock) return;
      ptPos.set(P, q); q += P.length;
      for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
        o = (j * nu + i) * 3;
        if (i < nu - 1 || p.periodicU) { o2 = (j * nu + (i + 1) % nu) * 3; A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o2]; A[n++] = P[o2 + 1]; A[n++] = P[o2 + 2]; }
        if (j < nv - 1 || p.periodicV) { o2 = (((j + 1) % nv) * nu + i) * 3; A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o2]; A[n++] = P[o2 + 1]; A[n++] = P[o2 + 2]; }
      }
    });
    netL.set(n / 3); pts.geometry.setDrawRange(0, q / 3); pts.geometry.attributes.position.needsUpdate = true;
  }
  function showNets() { netL.obj.visible = pts.visible = st.nets; R.invalidate(); }

  /* ---------- control features: DINOv3, baked per control point (lab/features/<scene>.js) ---------- */
  function loadFeatures(key) {
    function attach() {
      var f = KL.features && KL.features[key]; if (st.scn !== key || !f) return paintFeat();
      var ok = charts.every(function (c) { var e = f.charts[c.name]; return e && e.nu === c.patch.nu && e.nv === c.patch.nv; });
      if (!ok) { feat = null; return paintFeat('The baked features do not match this scene’s control nets (re-run tools/bake).'); }
      feat = f;
      charts.forEach(function (c) {
        var bin = atob(f.charts[c.name].data), n = c.patch.nu * c.patch.nv, i, k, d = f.dim;
        c.F = new Float32Array(n * d);
        for (i = 0; i < n * d; i++) c.F[i] = ((bin.charCodeAt(i) << 24) >> 24) * f.scale;        // int8
        for (i = 0; i < n; i++) for (k = 0; k < 3; k++) c.A[i * 5 + k] = KT.clamp((c.F[i * d + k] - f.lo[k]) / (f.hi[k] - f.lo[k]), 0, 1);
      });
      paintFeat(); paintCaption(); need = true;
    }
    if (KL.features && KL.features[key]) return attach();
    var s = document.createElement('script'); s.src = 'lab/features/' + key + '.js'; s.onload = attach; s.onerror = function () { paintFeat(); };
    document.head.appendChild(s); paintFeat();
  }
  function featureAt(c, u, v) {
    var d = feat.dim, f = new Float32Array(d), k;
    NB.patchSupport(c.patch, u, v).forEach(function (s) { var q = (s.j * c.patch.nu + s.i) * d; for (k = 0; k < d; k++) f[k] += s.w * c.F[q + k]; });
    return f;
  }

  /* ---------- picking: the nearest Gaussian along the ray under the pointer ---------- */
  var ray = new T.Raycaster(), ndc = new T.Vector2();
  function setRay(x, y) { var r = R.canvas.getBoundingClientRect(); ndc.set((x - r.left) / r.width * 2 - 1, -((y - r.top) / r.height * 2 - 1)); ray.setFromCamera(ndc, cam); }
  function pick(x, y) {
    setRay(x, y);
    var o = ray.ray.origin, d = ray.ray.direction, C = R.center, U = R.axU, V = R.axV, best = -1, bt = 1e9, i, q;
    for (i = 0; i < st.drawn; i++) {
      q = i * 3;
      var ex = C[q] - o.x, ey = C[q + 1] - o.y, ez = C[q + 2] - o.z, t = ex * d.x + ey * d.y + ez * d.z;
      if (t <= 0 || t >= bt) continue;
      var s2 = (U[q] * U[q] + U[q + 1] * U[q + 1] + U[q + 2] * U[q + 2] + V[q] * V[q] + V[q + 1] * V[q + 1] + V[q + 2] * V[q + 2]) * 1.6;
      if (ex * ex + ey * ey + ez * ez - t * t < s2) { bt = t; best = i; }
    }
    if (best < 0) return null;
    return { c: charts[R.meta[best * 4]], u: R.meta[best * 4 + 2], v: R.meta[best * 4 + 3], pos: [C[best * 3], C[best * 3 + 1], C[best * 3 + 2]] };
  }

  /* ---------- sculpt: control points inside the brush follow the drag ---------- */
  var grab = null, plane = new T.Plane(), hitP = new T.Vector3();
  function startSculpt(e) {
    if (st.tool !== 'sculpt') return false;
    var h = pick(e.clientX, e.clientY); if (!h || h.c.lock) return false;
    var s = st.brush * (def.radius || 3) * 0.5, list = [];
    charts.forEach(function (c) {
      if (c.lock) return;
      var P = c.patch.P, n = P.length / 3, idx = [], w = [], i, d2;
      for (i = 0; i < n; i++) {
        d2 = Math.pow(P[i * 3] - h.pos[0], 2) + Math.pow(P[i * 3 + 1] - h.pos[1], 2) + Math.pow(P[i * 3 + 2] - h.pos[2], 2);
        if (d2 < 9 * s * s) { idx.push(i); w.push(Math.exp(-0.5 * d2 / (s * s))); }
      }
      if (idx.length) list.push({ c: c, idx: idx, w: w, was: new Float32Array(P) });
    });
    plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), hitP.fromArray(h.pos));
    grab = { id: e.pointerId, p0: h.pos, list: list };
    if (sel) clearSel();
    return true;
  }
  function moveSculpt(e) {
    if (!grab || e.pointerId !== grab.id) return;
    setRay(e.clientX, e.clientY); if (!ray.ray.intersectPlane(plane, hitP)) return;
    var dx = hitP.x - grab.p0[0], dy = hitP.y - grab.p0[1], dz = hitP.z - grab.p0[2];
    grab.list.forEach(function (g) {
      var P = g.c.patch.P, k, i;
      for (k = 0; k < g.idx.length; k++) { i = g.idx[k] * 3; P[i] = g.was[i] + g.w[k] * dx; P[i + 1] = Math.max(0.002, g.was[i + 1] + g.w[k] * dy); P[i + 2] = g.was[i + 2] + g.w[k] * dz; }
      coarse(g.c);
    });
    updateNets(); need = true;
  }
  function endSculpt(e, cancel) {
    if (!grab || (e && e.pointerId !== grab.id)) return;
    if (cancel) { grab.list.forEach(function (g) { g.c.patch.P.set(g.was); coarse(g.c); }); updateNets(); need = true; }
    grab = null;
  }

  /* ---------- select by feature (or by chart), then edit the selection ---------- */
  function selectAt(x, y) {
    var h = pick(x, y); if (!h || h.c.lock) return clearSel();
    if (sel) clearSel();
    var byFeat = st.by === 'feature' && feat, f = byFeat ? featureAt(h.c, h.u, h.v) : null, fl = 0, k, d = feat ? feat.dim : 0;
    if (f) { for (k = 0; k < d; k++) fl += f[k] * f[k]; fl = Math.sqrt(fl) || 1; }
    charts.forEach(function (c) {
      var n = c.patch.nu * c.patch.nv, i, s, l;
      c.S = new Float32Array(n);
      for (i = 0; i < n; i++) {
        if (c.lock) s = 0;
        else if (f) { s = 0; l = 0; for (k = 0; k < d; k++) { s += f[k] * c.F[i * d + k]; l += c.F[i * d + k] * c.F[i * d + k]; } s /= fl * (Math.sqrt(l) || 1); }
        else s = c === h.c ? 1 : 0;
        c.S[i] = s;
      }
      c.P0 = new Float32Array(c.patch.P); c.C0 = new Float32Array(c.patch.C);
      // a normal per control point (at its Greville site), for inflating
      var g = { pos: new Float32Array(n * 3), su: new Float32Array(n * 3), sv: new Float32Array(n * 3) };
      NB.patchGrid(c.patch, c.gU, c.gV, g); c.N0 = new Float32Array(n * 3);
      for (i = 0; i < n; i++) {
        var o = i * 3, nx = g.su[o + 1] * g.sv[o + 2] - g.su[o + 2] * g.sv[o + 1], ny = g.su[o + 2] * g.sv[o] - g.su[o] * g.sv[o + 2], nz = g.su[o] * g.sv[o + 1] - g.su[o + 1] * g.sv[o];
        l = (Math.hypot(nx, ny, nz) || 1) * c.flip; c.N0[o] = nx / l; c.N0[o + 1] = ny / l; c.N0[o + 2] = nz / l;
      }
    });
    sel = { seed: h, byFeat: !!f, edit: { inflate: 0, lift: 0, scale: 1, tint: null, amount: 0.85 } };
    threshold(); paintSel();
  }
  /* similarity -> selection weights (soft edge), their centroid and count */
  function threshold() {
    if (!sel) return;
    var n = 0, cx = 0, cy = 0, cz = 0, sw = 0;
    charts.forEach(function (c) {
      var P = c.P0, i, w, m = c.S.length;
      for (i = 0; i < m; i++) {
        w = sel.byFeat ? KT.clamp((c.S[i] - st.thr) / 0.08 + 0.5, 0, 1) : c.S[i]; w = w * w * (3 - 2 * w);
        c.A[i * 5 + 3] = Math.max(0, c.S[i]); c.A[i * 5 + 4] = w;
        if (w > 0.5) n++;
        cx += w * P[i * 3]; cy += w * P[i * 3 + 1]; cz += w * P[i * 3 + 2]; sw += w;
      }
    });
    sel.n = n; sel.cen = sw ? [cx / sw, cy / sw, cz / sw] : [0, 0, 0];
    applyEdit();
  }
  function applyEdit() {
    if (!sel) return;
    var e = sel.edit, cen = sel.cen, unit = (def.radius || 3) * 0.25;
    charts.forEach(function (c) {
      if (c.lock) return;
      var P = c.patch.P, C = c.patch.C, i, o, w, k, s, m = c.S.length;
      for (i = 0; i < m; i++) {
        o = i * 3; w = c.A[i * 5 + 4]; s = 1 + w * (e.scale - 1);
        for (k = 0; k < 3; k++) {
          P[o + k] = cen[k] + (c.P0[o + k] + w * (e.inflate * unit * c.N0[o + k] + (k === 1 ? e.lift * unit : 0)) - cen[k]) * s;
          C[o + k] = e.tint ? c.C0[o + k] + (e.tint[k] * (0.55 + 0.9 * lum(c.C0, o)) - c.C0[o + k]) * w * e.amount : c.C0[o + k];
        }
        if (P[o + 1] < 0.002) P[o + 1] = 0.002;
      }
      coarse(c);
    });
    updateNets(); need = true;
  }
  function lum(C, o) { return 0.3 * C[o] + 0.59 * C[o + 1] + 0.11 * C[o + 2]; }
  function clearSel() {            // keeps what was edited; only the selection itself goes
    if (!sel) return;
    charts.forEach(function (c) { for (var i = 0; i < c.S.length; i++) { c.A[i * 5 + 3] = 0; c.A[i * 5 + 4] = 0; } });
    sel = null; st.hide = false; R.hide(false); paintSel(); need = true;
  }
  function resetScene() {
    charts.forEach(function (c) { c.patch.P.set(c.base); c.patch.C.set(c.baseC); coarse(c); });
    if (sel) { sel = null; charts.forEach(function (c) { for (var i = 0; i < c.S.length; i++) { c.A[i * 5 + 3] = 0; c.A[i * 5 + 4] = 0; } }); }
    st.hide = false; R.hide(false); updateNets(); paintSel(); need = true;
  }

  /* ---------- camera and pointer ---------- */
  var orbit = G.orbit(cam, R.canvas, { embed: small, minRadius: 1.2, maxRadius: 40, pick: startSculpt,
    hit: function (x, y) { if (st.tool !== 'sculpt') return false; var h = pick(x, y); return !!h && !h.c.lock; },
    release: function () { endSculpt(null, true); } });
  var down = null;
  R.canvas.addEventListener('pointerdown', function (e) { down = { x: e.clientX, y: e.clientY, id: e.pointerId }; });
  R.canvas.addEventListener('pointermove', function (e) { moveSculpt(e); cursor(e); });
  R.canvas.addEventListener('pointerup', function (e) {
    endSculpt(e);
    if (down && down.id === e.pointerId && st.tool === 'select' && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 6) selectAt(e.clientX, e.clientY);
    down = null;
  });
  R.canvas.addEventListener('pointercancel', function (e) { endSculpt(e, true); down = null; });
  R.canvas.addEventListener('pointerleave', function () { ring.style.opacity = 0; });
  var ring = el('div', 'lab-brush'); stage.appendChild(ring);
  function cursor(e) {
    if (st.tool !== 'sculpt' || e.pointerType === 'touch') { ring.style.opacity = 0; return; }
    var r = R.canvas.getBoundingClientRect(), px = st.brush * (def.radius || 3) * 0.5 * r.height / (2 * Math.tan(cam.fov * Math.PI / 360) * orbit.cur.r);
    ring.style.opacity = 1; ring.style.width = ring.style.height = 2 * px + 'px';
    ring.style.transform = 'translate(' + (e.clientX - r.left - px) + 'px,' + (e.clientY - r.top - px) + 'px)';
  }

  /* ---------- split screen: one divider and one label per band ---------- */
  var split = el('div', 'lab-split'), bands = [], size = { w: 1, h: 1 };
  stage.appendChild(split);
  function bandModes() { return [0, 2, 3, 4, 5].concat(feat ? [6] : []); }
  function showView() {
    var bm = bandModes();
    R.view(st.mode, st.split ? bm : null, st.roll, SLANT);
    stage.classList.toggle('splitting', st.split);
    R.live(st.split ? 0.07 : 0);
    if (st.split && bands.length !== bm.length) {
      split.innerHTML = '';
      bands = bm.map(function (m, j) {
        var b = { a: el('i'), b: el('i'), lab: el('span', '', '<b>(' + 'abcdefgh'[j] + ')</b> ' + MODES[m].name) };
        split.appendChild(b.a); split.appendChild(b.b); split.appendChild(b.lab); return b;
      });
    }
    [].forEach.call($('modes').children, function (b) { b.classList.toggle('on', st.split ? b._m === -1 : b._m === st.mode); });
    paintCaption();
  }
  function placeSplit() {
    var w = size.w, h = size.h, n = bands.length, tight = w / n < 150;
    bands.forEach(function (b, j) {
      var yl = h - 24 - (tight ? j % 3 * 28 : 0), x = w * wrap(j / n + st.roll, 1), sk = ' skewX(' + (-Math.atan(SLANT)) + 'rad)';
      b.a.style.transform = 'translateX(' + x + 'px)' + sk; b.b.style.transform = 'translateX(' + (x + w) + 'px)' + sk;
      var xc = wrap(w * ((j + 0.5) / n + st.roll) - SLANT * yl, w);
      b.lab.style.transform = 'translate(' + xc + 'px,' + yl + 'px) translate(-50%,-50%)';
      b.lab.style.opacity = KT.clamp(Math.min(xc, w - xc) / 50 - 0.5, 0, 1);
    });
  }

  /* ---------- interface ---------- */
  function slider(id, o) {
    var row = $(id), inp = row.querySelector('input'), val = row.querySelector('output');
    inp.min = o.min; inp.max = o.max; inp.step = o.step; inp.value = o.value;
    function show() { if (val) val.textContent = o.fmt ? o.fmt(+inp.value) : inp.value; }
    inp.addEventListener('input', function () { show(); o.on(+inp.value); }); show();
    return { set: function (v) { inp.value = v; show(); } };
  }
  function seg(id, value, on) {
    var box = $(id), btns = [].slice.call(box.querySelectorAll('button'));
    function paint(v) { btns.forEach(function (b) { b.classList.toggle('on', b.dataset.v === v); }); }
    btns.forEach(function (b) { b.addEventListener('click', function () { paint(b.dataset.v); on(b.dataset.v); }); });
    paint(value); return { set: paint };
  }
  Object.keys(KL.scenes).forEach(function (k) {
    var b = el('button', '', KL.scenes[k].label); b.type = 'button'; b._k = k; b.addEventListener('click', function () { loadScene(k); }); $('scenes').appendChild(b);
  });
  MODES.concat([{ v: 'split', name: 'All, side by side' }]).forEach(function (m, i) {
    var b = el('button', '', '<i>(' + 'abcdefghi'[i] + ')</i>' + m.name); b.type = 'button'; b._m = i < MODES.length ? i : -1;
    b.addEventListener('click', function () {
      st.split = b._m < 0; if (!st.split) st.mode = b._m;
      if (b._m === 7) setTool('select');           // similarity is measured from a clicked point
      showView();
    });
    $('modes').appendChild(b);
  });
  var slB = slider('budget', { min: 20000, max: CAP - 20000, step: 5000, value: st.budget, fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, on: function (v) { st.budget = v; need = true; } });
  seg('sampling', st.sampling, function (v) { st.sampling = v; need = true; });
  $('nets').addEventListener('change', function () { st.nets = this.checked; showNets(); });
  function setTool(v) { st.tool = v; document.body.dataset.tool = v; ring.style.opacity = 0; segTool.set(v); }
  var segTool = seg('tool', st.tool, setTool);
  document.body.dataset.tool = st.tool;
  slider('brush', { min: 0.15, max: 1.5, step: 0.05, value: st.brush, fmt: function (v) { return v.toFixed(2); }, on: function (v) { st.brush = v; } });
  var segBy = seg('by', st.by, function (v) { st.by = v; });
  slider('thr', { min: 0.1, max: 0.95, step: 0.01, value: st.thr, fmt: function (v) { return v.toFixed(2); }, on: function (v) { st.thr = v; threshold(); paintSel(); } });
  var ed = {
    inflate: slider('inflate', { min: -1, max: 1, step: 0.02, value: 0, fmt: sgn, on: function (v) { if (sel) { sel.edit.inflate = v; applyEdit(); } } }),
    lift: slider('lift', { min: -1, max: 2, step: 0.02, value: 0, fmt: sgn, on: function (v) { if (sel) { sel.edit.lift = v; applyEdit(); } } }),
    scale: slider('scale', { min: 0.4, max: 1.8, step: 0.01, value: 1, fmt: function (v) { return '×' + v.toFixed(2); }, on: function (v) { if (sel) { sel.edit.scale = v; applyEdit(); } } })
  };
  function sgn(v) { return (v > 0 ? '+' : '') + v.toFixed(2); }
  TINTS.forEach(function (t) {
    var b = el('button'); b.type = 'button'; b.title = t[0]; b.setAttribute('aria-label', 'tint ' + t[0]); b.style.background = 'rgb(' + t[1].map(function (x) { return Math.round(x * 255); }) + ')';
    b.addEventListener('click', function () { if (!sel) return; sel.edit.tint = sel.edit.tint === t[1] ? null : t[1]; applyEdit(); paintSel(); });
    b._t = t[1]; $('tints').appendChild(b);
  });
  $('hide').addEventListener('change', function () { st.hide = this.checked; R.hide(st.hide); });
  $('clear').addEventListener('click', clearSel);
  $('reset').addEventListener('click', resetScene);
  $('save').addEventListener('click', function () {
    R.present();                                   // the converged picture again, read before the browser presents it
    var a = document.createElement('a'); a.download = 'knot-a-surface-' + st.scn + '.png'; a.href = R.canvas.toDataURL('image/png'); a.click();
  });

  function paintSel() {
    document.body.classList.toggle('has-sel', !!sel);
    $('hide').checked = st.hide;
    if (sel) { ed.inflate.set(sel.edit.inflate); ed.lift.set(sel.edit.lift); ed.scale.set(sel.edit.scale); }
    [].forEach.call($('tints').children, function (b) { b.classList.toggle('on', !!sel && sel.edit.tint === b._t); });
    $('selinfo').innerHTML = sel
      ? '<b>' + fmt(sel.n) + '</b> control points selected ' + (sel.byFeat ? 'by similarity to the clicked point on <em>' + sel.seed.c.name + '</em>' : 'on <em>' + sel.seed.c.name + '</em>') + '.'
      : 'Click a surface in the figure to select by ' + (st.by === 'feature' && feat ? 'feature similarity' : 'chart') + '.';
  }
  function paintFeat(msg) {
    var on = !!feat; document.body.classList.toggle('no-feat', !on);
    [].forEach.call($('modes').children, function (b) { if (b._m === 6 || b._m === 7) b.disabled = !on; });
    if (!on && (st.mode === 6 || st.mode === 7)) st.mode = 0;
    if (!on) { st.by = 'chart'; segBy.set('chart'); } else { st.by = 'feature'; segBy.set('feature'); }
    $('featnote').innerHTML = on
      ? 'Features: <b>' + feat.model + '</b>, run on ' + feat.views + ' rendered views at ' + feat.res + ' px, distilled onto the control points and reduced to ' + feat.dim + ' dimensions by PCA (' + Math.round(feat.explained * 100) + ' % of the variance).'
      : (msg || 'No baked features for this scene: selection works per chart.');
    showView(); paintSel();
  }
  function paintNumbers() {
    $('n-g').textContent = fmt(st.drawn); $('n-cp').textContent = fmt(totalCP); $('n-par').textContent = fmt(totalCP * PER_CP);
    charts.forEach(function (c) { if (c.cell) c.cell.textContent = fmt(c.count); });
    paintCaption();
  }
  function paintTable() {
    var tb = $('atlas'); tb.innerHTML = '';
    charts.forEach(function (c) {
      var tr = el('tr'), p = c.patch;
      tr.appendChild(el('td', '', '<i style="background:' + c.hex + '"></i>' + c.name));
      tr.appendChild(el('td', '', p.nu + '×' + p.nv + (p.periodicU || p.periodicV ? '<sup>○</sup>' : '')));
      c.cell = el('td', '', '–'); tr.appendChild(c.cell);
      tr.appendChild(el('td', '', c.mat.metal ? 'metal' : c.mat.rough < 0.3 ? 'glazed' : c.mat.rough < 0.7 ? 'satin' : 'matte'));
      tb.appendChild(tr);
    });
  }
  function paintCaption() {
    if (!def) return;
    var s = KL.scenes[st.scn], what = st.split
      ? 'The same frame read ' + bandModes().length + ' ways: ' + bandModes().map(function (m, j) { return '(' + 'abcdef'[j] + ') ' + MODES[m].name; }).join(', ') + '.'
      : ['Lit with one key light, its shadow, ambient occlusion and a studio reflection.', 'The colour stored on the control points, unlit.', 'Distance from the camera, light near and dark far.',
         'Surface normals, read off the spline as S<sub>u</sub> × S<sub>v</sub>.', 'Each Gaussian drawn at reduced size.', 'The chart every Gaussian was derived from.',
         'DINOv3 features stored on the control points; colour is their first three principal components.', 'Cosine similarity of every control feature to the clicked point.'][st.mode];
    $('caption').innerHTML = '<b>Fig. 1.</b> <em>' + s.label + '.</em> ' + fmt(st.drawn) + ' Gaussians, none of them stored: each is derived on the fly from one of ' +
      charts.length + ' B-spline charts (' + fmt(totalCP) + ' control points, ' + fmt(totalCP * PER_CP) + ' numbers). ' + what + ' <span class="credit">' + (s.credit || '') + '</span>';
  }

  /* ---------- loop ---------- */
  var last = 0, bar = $('conv');
  function frame(ms) {
    requestAnimationFrame(frame);
    if (!def || document.hidden) return;
    var t = ms / 1000, dt = last ? Math.min(0.05, t - last) : 0.016; last = t;
    orbit.update(dt, 4);
    var e = cam.matrixWorld.elements, key = (e[12] + e[13] * 3 + e[14] * 7 + e[8] * 11 + e[9] * 13).toFixed(3);
    if (key !== camKey) { R.invalidate(); if (t - lastAlloc > 0.09) { camKey = key; need = true; } }
    if (need) { need = false; lastAlloc = t; resample(); }
    if (st.split) { st.roll = wrap(st.roll + ROLL * dt, 1); R.view(st.mode, bandModes(), st.roll, SLANT, true); placeSplit(); }
    R.render();
    bar.style.transform = 'scaleX(' + (st.split ? 0 : R.progress()) + ')';
  }
  new ResizeObserver(function () {
    var w = stage.clientWidth, h = stage.clientHeight; if (!w || !h) return;
    size.w = w; size.h = h; R.resize(w, h, +query.dpr || 0); need = true;
  }).observe(stage);

  KL.app = { R: R, camera: cam, orbit: orbit, st: st, charts: function () { return charts; }, scene: function () { return def; },
             load: loadScene, pick: pick, selectAt: selectAt, features: function () { return feat; } };   // for tests and tools/bake
  loadScene(KL.scenes[query.scene] ? query.scene : Object.keys(KL.scenes)[0]);
  showView();
  requestAnimationFrame(frame);
})();
