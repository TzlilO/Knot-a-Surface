/* Knot-a-Surface talk — KnotTalk.hands: work on a scene of spline charts with your hands, and on its atlas.
   A tool layer of its own, not part of any one module. It needs a HOST that can say what is under the
   pointer and redraw, and charts in the shape talk/scenes.js builds them. Everything it does is an edit of
   control points, control colours or control opacities, or a mask stored per control point; the Gaussians
   are never touched, they are derived again.

   Hands (mouse or touch, on the object in the 3D view; the ground is left alone):
     grab   drag a patch of the surface with the pointer, in the plane of the screen
     pinch  while held, the patch draws together around the point you hold
     pull   while held, the patch comes out along the surface normal there
     push   while held, it goes in
     hold   paint a region that the other actions then cannot move (in the control points' colour); "let go" frees it
   The patch is every control point within `reach` of the point you touched, with a smooth falloff.

   Atlas: one chart as a picture of its own (u, v) domain, one pixel per control point, in the channel you
   choose: colour, normals, opacity, height, baked features (when the host has them), or the masks alone.
   On it: wand (grow a selection from the pixel you click over the channel shown; on features, everything
   similar to it), brush, erase. The selection (in the surface colour) can then be tinted, hidden, inflated or held.

   show(true, 'select'): the atlas and its selection tools only, for a host that does something of its own with
   the selection (the manifold module's `paint` scene puts the chosen material on it). On the object the pointer
   then selects too: the wand from the control point under it, or the brush and the eraser around it.

   host = { root, ui, canvas, camera, charts() -> [chart], pick(x, y) -> { c, i, j, pos, n } | null,
            changed(chart, shape) }       shape: true when control points moved (false: colour, opacity, masks)
   A chart may carry: ground (never touched), feat { dim, scale, lo, hi, nu, nv, step, data (base64 int8) }.
   This layer adds to a chart: sel, pin (Float32Array, one per control point), selRev (counts every change of
   sel, so a host can tell), alpha (the same, when something was hidden; the renderer reads it), and keeps what it
   changed so that reset() puts it back.
   -> { claim(e), hit(x, y), undo(), step(dt), show(on, mode), reset(), el }   claim / hit / undo are for the host's
   orbit (KnotTalk.gestures): this layer adds no touch handling of its own. */
(function () {
  'use strict';
  var KT = window.KnotTalk, T = KT.t;      // T: a string in the page's language (talk/i18n.js; the English itself on a page without languages), for showing only
  var ACTS = [{ v: 'grab', label: T('grab') }, { v: 'pinch', label: T('pinch') }, { v: 'pull', label: T('pull') }, { v: 'push', label: T('push') }, { v: 'hold', label: T('hold') }];
  var TINTS = [[0.85, 0.20, 0.25], [0.95, 0.70, 0.10], [0.20, 0.70, 0.35], [0.20, 0.45, 0.90], [0.90, 0.90, 0.88]];
  /* the marks on the atlas: a selection tinted towards the surface colour, a hold towards the control points', and the ring
     around the pixel last touched. A dark page keeps them as they were written (cyan, amber, white); a light page takes
     its own surface and control colours, and a ring in the text colour */
  function rgbOf(hex) { var n = parseInt(hex.slice(1), 16); return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255]; }
  var LIGHT = KT.theme === 'light';
  var SELC = LIGHT ? rgbOf(KT.color.surf) : [0.31, 0.76, 0.97], HOLDC = LIGHT ? rgbOf(KT.color.ctrl) : [1.0, 0.65, 0.15], RING = LIGHT ? KT.color.text : '#fff';

  function create(host) {
    var THREE = window.THREE, ui = host.ui, cam = host.camera, cv = host.canvas;
    var st = { act: 'grab', reach: 0.22, power: 1, chan: 'rgb', tool: 'wand', tol: 0.16, tint: 0, on: false, mode: 'hands' };
    var stroke = null, dirty = true, last = 0, mark = null;
    var ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(), hitP = new THREE.Vector3();

    /* ---------- the chart being worked on: the first one that is not ground ---------- */
    function obj() { var cs = host.charts(), i; for (i = 0; i < cs.length; i++) if (!cs[i].ground) return cs[i]; return cs[0]; }
    function masks(c) {
      var n = c.patch.nu * c.patch.nv;
      if (!c.sel || c.sel.length !== n) { c.sel = new Float32Array(n); c.pin = new Float32Array(n); }
      return c;
    }
    /* unit normals of the control net, pointing the way S_u x S_v does */
    function normals(c) {
      var p = c.patch, P = p.P, nu = p.nu, nv = p.nv, N = c.hN && c.hN.length === P.length ? c.hN : (c.hN = new Float32Array(P.length)), i, j;
      for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
        var a = (j * nu + Math.max(0, i - 1)) * 3, b = (j * nu + Math.min(nu - 1, i + 1)) * 3, g = (Math.max(0, j - 1) * nu + i) * 3, h = (Math.min(nv - 1, j + 1) * nu + i) * 3;
        var ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[h] - P[g], vy = P[h + 1] - P[g + 1], vz = P[h + 2] - P[g + 2];
        var x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx, d = Math.hypot(x, y, z) || 1, o = (j * nu + i) * 3;
        N[o] = x / d * c.flip; N[o + 1] = y / d * c.flip; N[o + 2] = z / d * c.flip;
      }
      c.hNok = true;
      return N;
    }
    function nrm(c) { return c.hNok ? c.hN : normals(c); }

    /* ---------- hands ---------- */
    /* every control point of c within `reach` of `at`, around the control point (i0, j0): indices and weights */
    function patchAt(c, i0, j0, at, free) {
      var p = c.patch, P = p.P, nu = p.nu, nv = p.nv, R = st.reach, idx = [], w = [], i, j, o, d;
      function gap(a, b) { return Math.hypot(P[a] - P[b], P[a + 1] - P[b + 1], P[a + 2] - P[b + 2]) || 1e-3; }
      var o0 = (j0 * nu + i0) * 3, ru = Math.min(nu, Math.ceil(1.6 * R / gap(o0, (j0 * nu + Math.min(nu - 1, i0 + 1)) * 3 === o0 ? o0 - 3 : (j0 * nu + Math.min(nu - 1, i0 + 1)) * 3)) + 1);
      var o1 = (Math.min(nv - 1, j0 + 1) * nu + i0) * 3, rv = Math.min(nv, Math.ceil(1.6 * R / gap(o0, o1 === o0 ? o0 - nu * 3 : o1)) + 1);
      for (j = Math.max(0, j0 - rv); j <= Math.min(nv - 1, j0 + rv); j++) for (i = Math.max(0, i0 - ru); i <= Math.min(nu - 1, i0 + ru); i++) {
        o = (j * nu + i) * 3; d = Math.hypot(P[o] - at[0], P[o + 1] - at[1], P[o + 2] - at[2]) / R; if (d >= 1) continue;
        d = Math.cos(d * Math.PI / 2); d *= d; if (!free) d *= 1 - c.pin[j * nu + i];
        if (d > 1e-4) { idx.push(j * nu + i); w.push(d); }
      }
      return { idx: idx, w: w };
    }
    function setRay(e) { var r = cv.getBoundingClientRect(); ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1)); ray.setFromCamera(ndc, cam); }
    function holdAt(h) {                       // hold: the patch under the pointer is pinned, at full strength in its middle
      var b = patchAt(h.c, h.i, h.j, h.pos, true), k;
      for (k = 0; k < b.idx.length; k++) h.c.pin[b.idx[k]] = Math.max(h.c.pin[b.idx[k]], Math.min(1, 1.4 * b.w[k]));
      dirty = true; host.changed(h.c, false);
    }
    function claim(e) {
      if (!st.on) return false;
      var h = host.pick(e.clientX, e.clientY); if (!h || h.c.ground) return false;
      masks(h.c); mark = { c: h.c, i: h.i, j: h.j };
      if (st.mode === 'select') {                // the atlas's tools, used on the object
        stroke = { c: h.c, id: e.pointerId, sel: true, was: new Float32Array(h.c.sel) };
        selectAt(h.c, h.i, h.j, true);
        try { cv.setPointerCapture(e.pointerId); } catch (err) { /* a synthetic pointer */ }
        return true;
      }
      if (st.act === 'hold') { stroke = { c: h.c, id: e.pointerId, hold: true }; holdAt(h); }
      else {
        var b = patchAt(h.c, h.i, h.j, h.pos, false), P = h.c.patch.P, was = new Float32Array(b.idx.length * 3), k;
        for (k = 0; k < b.idx.length; k++) { was[k * 3] = P[b.idx[k] * 3]; was[k * 3 + 1] = P[b.idx[k] * 3 + 1]; was[k * 3 + 2] = P[b.idx[k] * 3 + 2]; }
        stroke = { c: h.c, id: e.pointerId, idx: b.idx, w: b.w, was: was, at: h.pos, n: h.n, d: [0, 0, 0], t: 0 };
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new THREE.Vector3()), hitP.set(h.pos[0], h.pos[1], h.pos[2]));
      }
      remember(h.c);
      try { cv.setPointerCapture(e.pointerId); } catch (err) { /* a synthetic pointer */ }
      return true;
    }
    /* where every point of the patch is now: grab = where the pointer took it; the others grow with the time held */
    function apply() {
      var S = stroke, P = S.c.patch.P, a = S.at, n = S.n, k, o, q, w, x, y, z, s;
      var out = st.act === 'pull' ? 1 : st.act === 'push' ? -1 : 0, go = out * 0.22 * st.power * S.t, sq = st.act === 'pinch' ? 1 - Math.exp(-1.3 * st.power * S.t) : 0;
      for (k = 0; k < S.idx.length; k++) {
        o = S.idx[k] * 3; q = k * 3; w = S.w[k]; x = S.was[q]; y = S.was[q + 1]; z = S.was[q + 2];
        if (sq) {                                // pinch: towards the held point, along the surface only
          var dx = x - a[0], dy = y - a[1], dz = z - a[2], dn = dx * n[0] + dy * n[1] + dz * n[2];
          s = w * sq; x -= (dx - dn * n[0]) * s; y -= (dy - dn * n[1]) * s; z -= (dz - dn * n[2]) * s;
        }
        P[o] = x + (S.d[0] + n[0] * go) * w; P[o + 1] = y + (S.d[1] + n[1] * go) * w; P[o + 2] = z + (S.d[2] + n[2] * go) * w;
      }
      S.c.hNok = false; dirty = true; host.changed(S.c, true);
    }
    cv.addEventListener('pointermove', function (e) {
      if (!stroke || e.pointerId !== stroke.id) return;
      if (stroke.sel) {                          // brush and eraser follow the pointer; the wand acted where it went down
        if (st.tool === 'wand') return;
        var hs = host.pick(e.clientX, e.clientY); if (hs && hs.c === stroke.c) { mark = { c: hs.c, i: hs.i, j: hs.j }; selectAt(hs.c, hs.i, hs.j, false); }
        return;
      }
      if (stroke.hold) { var h = host.pick(e.clientX, e.clientY); if (h && h.c === stroke.c) { mark = { c: h.c, i: h.i, j: h.j }; holdAt(h); } return; }
      if (st.act !== 'grab') return;
      setRay(e);
      if (ray.ray.intersectPlane(plane, hitP)) { stroke.d = [hitP.x - stroke.at[0], hitP.y - stroke.at[1], hitP.z - stroke.at[2]]; apply(); }
    });
    function end(e) { if (stroke && e.pointerId === stroke.id) stroke = null; }
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    function undo() {                          // a second finger arrived: this was the start of a pinch-zoom, not a stroke
      if (!stroke) return;
      if (stroke.sel) { stroke.c.sel.set(stroke.was); selChanged(stroke.c); stroke = null; return; }
      if (!stroke.hold) { stroke.d = [0, 0, 0]; stroke.t = 0; var a = st.act; st.act = 'grab'; apply(); st.act = a; }
      stroke = null;
    }

    /* ---------- what was changed, so that reset can put it back ---------- */
    function remember(c) { if (!c.h0) c.h0 = { P: new Float32Array(c.patch.P), C: new Float32Array(c.patch.C) }; }
    function reset() {
      host.charts().forEach(function (c) {
        if (c.h0) { c.patch.P.set(c.h0.P); c.patch.C.set(c.h0.C); c.h0 = null; }
        if (c.sel) { c.sel.fill(0); c.pin.fill(0); c.selRev = (c.selRev || 0) + 1; }
        c.alpha = null; c.hNok = false; host.changed(c, true);
      });
      stroke = null; mark = null; dirty = true;
    }

    /* ---------- the atlas ---------- */
    var CHANS = [{ v: 'rgb', label: 'rgb' }, { v: 'normal', label: T('normals') }, { v: 'alpha', label: T('opacity') }, { v: 'height', label: T('height') },
                 { v: 'feat', label: 'DINOv3' }, { v: 'mask', label: T('masks') }];
    var el = ui.el('div', 'kt-panel kt-hands'); host.root.appendChild(el);
    var only = [ui.el('div', 'kt-panel-title', T('hands'))]; el.appendChild(only[0]);     // `only`: not shown in the select mode
    only.push(ui.seg(el, { value: st.act, options: ACTS, onChange: function (v) { st.act = v; } }).el);
    only.push(ui.slider(el, { label: T('reach'), min: 0.06, max: 0.6, step: 0.01, value: st.reach, fmt: function (v) { return Math.round(v * 80) + ' mm'; }, onInput: function (v) { st.reach = v; } }).el);
    only.push(ui.slider(el, { label: T('strength'), min: 0.2, max: 3, step: 0.1, value: st.power, fmt: function (v) { return v.toFixed(1); }, onInput: function (v) { st.power = v; } }).el);
    var title = ui.el('div', 'kt-panel-title', T('atlas')); el.appendChild(title);
    var segC = ui.seg(el, { value: st.chan, options: CHANS, onChange: function (v) { st.chan = v; dirty = true; } });
    var box = ui.el('div', 'kt-hands-box'); el.appendChild(box);
    var pic = ui.el('canvas', 'kt-hands-cv'); box.appendChild(pic);
    var g = pic.getContext('2d'), img = null;
    ui.seg(el, { value: st.tool, options: [{ v: 'wand', label: T('wand') }, { v: 'brush', label: T('brush') }, { v: 'erase', label: T('erase') }], onChange: function (v) { st.tool = v; } });
    ui.slider(el, { label: T('tolerance'), min: 0.02, max: 0.6, step: 0.01, value: st.tol, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { st.tol = v; } });
    var row = ui.el('div', 'kt-row kt-hands-ops'); el.appendChild(row); only.push(row);
    var note = ui.el('div', 'kt-hands-note'); el.appendChild(note);
    function sel() { var c = masks(obj()); return c; }
    function count(c) { var n = 0, i; for (i = 0; i < c.sel.length; i++) if (c.sel[i] > 0.5) n++; return n; }
    function each(fn) { var c = sel(), i; remember(c); for (i = 0; i < c.sel.length; i++) if (c.sel[i] > 0.5) fn(c, i); dirty = true; return c; }
    ui.button(row, { label: T('tint'), onClick: function () {
      var t = TINTS[st.tint++ % TINTS.length], c = each(function (c, i) { var K = c.patch.C, B = c.h0.C, o = i * 3, l = 0.4 + 2.2 * (0.3 * B[o] + 0.59 * B[o + 1] + 0.11 * B[o + 2]);
        K[o] = t[0] * Math.min(1, l); K[o + 1] = t[1] * Math.min(1, l); K[o + 2] = t[2] * Math.min(1, l); });    // the tint, shaded by the original's brightness
      host.changed(c, false); } });
    ui.button(row, { label: T('hide'), onClick: function () {
      var c = sel(); if (!c.alpha) c.alpha = new Float32Array(c.sel.length).fill(1);
      each(function (c, i) { c.alpha[i] = c.alpha[i] < 0.5 ? 1 : 0; }); host.changed(c, false); } });
    function puff(s) { return function () { var c = sel(), N = nrm(c); each(function (c, i) { var P = c.patch.P, o = i * 3; P[o] += N[o] * s; P[o + 1] += N[o + 1] * s; P[o + 2] += N[o + 2] * s; }); c.hNok = false; host.changed(c, true); }; }
    ui.button(row, { label: T('inflate'), onClick: puff(0.025) });
    ui.button(row, { label: T('deflate'), onClick: puff(-0.025) });
    ui.button(row, { label: T('hold'), onClick: function () { var c = each(function (c, i) { c.pin[i] = 1; }); host.changed(c, false); } });
    var row2 = ui.el('div', 'kt-row kt-hands-ops'); el.appendChild(row2);
    ui.button(row2, { label: T('select none'), onClick: function () { var c = sel(); c.sel.fill(0); selChanged(c); } });
    only.push(ui.button(row2, { label: T('let go'), onClick: function () { var c = sel(); c.pin.fill(0); dirty = true; host.changed(c, false); } }));
    only.push(ui.button(row2, { label: T('reset'), onClick: reset }));
    only.forEach(function (e) { e.classList.add('kt-hands-only'); });
    /* the selection changed: count it on the chart (a host may follow it), redraw the atlas, tell the host */
    function selChanged(c) { c.selRev = (c.selRev || 0) + 1; dirty = true; host.changed(c, false); }
    /* the selection tool at control point (i, j): the wand only where the pointer went down */
    function selectAt(c, i, j, down) {
      if (st.tool === 'wand') { if (!down) return; wand(c, i, j); } else daub(c, i, j, st.tool === 'brush' ? 1 : 0);
      selChanged(c);
    }

    /* baked features of a chart, decoded once (a host may hand them over later): { F (Float32, dim per stored point), dim, nu, nv, step, lo, hi } */
    function feat(c) {
      if (c.hF !== undefined && (c.hF || !c.feat)) return c.hF;
      var f = c.feat; if (!f) return (c.hF = null);
      var bin = atob(f.data), F = new Float32Array(bin.length), i;
      for (i = 0; i < bin.length; i++) F[i] = ((bin.charCodeAt(i) << 24) >> 24) * f.scale;        // int8
      return (c.hF = { F: F, dim: f.dim, nu: f.nu, nv: f.nv, step: f.step || 1, lo: f.lo, hi: f.hi });
    }
    function featAt(f, c, i, j) { return (Math.min(f.nv - 1, Math.round(j / f.step)) * f.nu + Math.min(f.nu - 1, Math.round(i / f.step))) * f.dim; }
    /* the value of the shown channel at control point k of c, as three numbers in 0..1. In the select mode the host
       paints the chart's colours itself, so `rgb` shows (and the wand reads) the chart's own colours: c.baseC, the
       photograph's, where the chart keeps them */
    function value(c, k, out) {
      var o = k * 3, K = st.mode === 'select' && c.baseC ? c.baseC : c.patch.C, f;
      if (st.chan === 'normal') { var N = nrm(c); out[0] = N[o] * 0.5 + 0.5; out[1] = N[o + 1] * 0.5 + 0.5; out[2] = N[o + 2] * 0.5 + 0.5; }
      else if (st.chan === 'alpha') out[0] = out[1] = out[2] = c.alpha ? c.alpha[k] : 1;
      else if (st.chan === 'height') out[0] = out[1] = out[2] = (c.patch.P[o + 1] - c.hY[0]) / (c.hY[1] - c.hY[0] || 1);
      else if (st.chan === 'feat' && (f = feat(c))) { var q = featAt(f, c, k % c.patch.nu, (k / c.patch.nu) | 0); for (var a = 0; a < 3; a++) out[a] = KT.clamp((f.F[q + a] - f.lo[a]) / (f.hi[a] - f.lo[a]), 0, 1); }
      else if (st.chan === 'mask') out[0] = out[1] = out[2] = 0.12;
      else { out[0] = Math.min(1, K[o] * 1.6); out[1] = Math.min(1, K[o + 1] * 1.6); out[2] = Math.min(1, K[o + 2] * 1.6); }   // the photograph is dark: shown a little brighter
      return out;
    }
    function paint() {
      var c = masks(obj()), p = c.patch, nu = p.nu, nv = p.nv, i, j, k, v = [0, 0, 0], y0 = 1e9, y1 = -1e9;
      for (k = 1; k < p.P.length; k += 3) { if (p.P[k] < y0) y0 = p.P[k]; if (p.P[k] > y1) y1 = p.P[k]; }
      c.hY = [y0, y1];
      if (!img || img.width !== nu || img.height !== nv) { pic.width = nu; pic.height = nv; img = g.createImageData(nu, nv); box.style.aspectRatio = nu + ' / ' + nv; }
      var D = img.data;
      for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
        k = j * nu + i; value(c, k, v);
        var s = c.sel[k] > 0.5 ? 0.5 : 0, h = c.pin[k] * 0.55, o = ((nv - 1 - j) * nu + i) * 4;      // v runs up the picture
        D[o] = 255 * ((v[0] * (1 - s) + SELC[0] * s) * (1 - h) + HOLDC[0] * h); D[o + 1] = 255 * ((v[1] * (1 - s) + SELC[1] * s) * (1 - h) + HOLDC[1] * h);
        D[o + 2] = 255 * ((v[2] * (1 - s) + SELC[2] * s) * (1 - h) + HOLDC[2] * h); D[o + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      if (mark && mark.c === c) { g.strokeStyle = RING; g.lineWidth = Math.max(1, nu / 120); g.beginPath(); g.arc(mark.i + 0.5, nv - 1 - mark.j + 0.5, nu / 30, 0, 6.3); g.stroke(); }
      title.textContent = T('atlas · {name} · {size}', { name: T(c.name), size: nu + ' × ' + nv });     // a chart's name is an id: only what is shown of it is translated
      var f = feat(c), n = count(c);
      note.textContent = (st.chan === 'feat' && !f ? T('no features are baked for this object') + ' · ' : '') + T('{n} control points selected', { n: n.toLocaleString('en-US') }) +
        (st.mode === 'select' ? ' · ' + T('select here or on the object') : '');
    }
    /* wand: on features, every control point whose feature is close to the clicked one; on any other channel,
       grow from the clicked point over neighbours whose value stays within the tolerance of it */
    function wand(c, i0, j0) {
      var p = c.patch, nu = p.nu, nv = p.nv, n = nu * nv, k0 = j0 * nu + i0, f = st.chan === 'feat' && feat(c), k, a;
      if (f) {
        var q0 = featAt(f, c, i0, j0), n0 = 0, d = f.dim;
        for (a = 0; a < d; a++) n0 += f.F[q0 + a] * f.F[q0 + a];
        for (k = 0; k < n; k++) {
          var q = featAt(f, c, k % nu, (k / nu) | 0), dot = 0, n1 = 0;
          for (a = 0; a < d; a++) { dot += f.F[q + a] * f.F[q0 + a]; n1 += f.F[q + a] * f.F[q + a]; }
          if (dot / (Math.sqrt(n0 * n1) || 1) > 1 - 2 * st.tol) c.sel[k] = 1;
        }
        return;
      }
      var seed = value(c, k0, [0, 0, 0]).slice(), v = [0, 0, 0], seen = new Uint8Array(n), stack = [k0];
      seen[k0] = 1;
      while (stack.length) {
        k = stack.pop(); value(c, k, v);
        if (Math.hypot(v[0] - seed[0], v[1] - seed[1], v[2] - seed[2]) > st.tol) continue;
        c.sel[k] = 1;
        var i = k % nu, j = (k / nu) | 0;
        if (i > 0 && !seen[k - 1]) { seen[k - 1] = 1; stack.push(k - 1); } if (i < nu - 1 && !seen[k + 1]) { seen[k + 1] = 1; stack.push(k + 1); }
        if (j > 0 && !seen[k - nu]) { seen[k - nu] = 1; stack.push(k - nu); } if (j < nv - 1 && !seen[k + nu]) { seen[k + nu] = 1; stack.push(k + nu); }
      }
    }
    function daub(c, i0, j0, on) {
      var p = c.patch, r = Math.max(2, Math.round(p.nu / 24)), i, j;
      for (j = Math.max(0, j0 - r); j <= Math.min(p.nv - 1, j0 + r); j++) for (i = Math.max(0, i0 - r); i <= Math.min(p.nu - 1, i0 + r); i++)
        if (Math.hypot(i - i0, j - j0) <= r) c.sel[j * p.nu + i] = on;
    }
    var drawing = false;
    function atlasAt(e) {
      var c = masks(obj()), r = pic.getBoundingClientRect(), i = Math.floor((e.clientX - r.left) / r.width * c.patch.nu), j = c.patch.nv - 1 - Math.floor((e.clientY - r.top) / r.height * c.patch.nv);
      if (i < 0 || j < 0 || i >= c.patch.nu || j >= c.patch.nv) return;
      selectAt(c, i, j, drawing === 'down');
      mark = { c: c, i: i, j: j }; drawing = true; dirty = true;
    }
    pic.addEventListener('pointerdown', function (e) { drawing = 'down'; try { pic.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ } atlasAt(e); e.preventDefault(); });
    pic.addEventListener('pointermove', function (e) { if (drawing) atlasAt(e); });
    pic.addEventListener('pointerup', function () { drawing = false; }); pic.addEventListener('pointercancel', function () { drawing = false; });

    if (!document.getElementById('kt-hands-css')) {
      var css = document.createElement('style'); css.id = 'kt-hands-css';
      css.textContent =
        '.kt-hands{left:12px;bottom:12px;width:268px;max-height:calc(100% - 24px);overflow:auto;display:none}.kt-hands.on{display:block}' +
        '.kt-hands .kt-seg .kt-seg-box{margin-left:0;width:100%;flex-wrap:wrap}.kt-hands .kt-seg .kt-seg-btn{flex:1;padding:6px 4px}' +
        '.kt-hands-box{margin:6px auto;width:176px;border:1px solid var(--line2);border-radius:4px;overflow:hidden;background:var(--bg);touch-action:none}' +
        '.kt-hands-cv{display:block;width:100%;height:100%;image-rendering:pixelated;cursor:crosshair}' +
        '.kt-hands-ops{flex-wrap:wrap;gap:4px}.kt-hands-ops .kt-btn{flex:1;padding:6px 4px;font-size:10px}' +
        '.kt-hands-note{font:500 10px/1.4 var(--f-mono);color:var(--mute);margin-top:4px}' +
        '.kt-hands .kt-panel-title+.kt-row{margin-top:0}.kt-hands .kt-panel-title:not(:first-child){margin-top:8px}' +
        '.kt-hands.kt-select .kt-hands-only{display:none}.kt-hands.kt-select .kt-panel-title{margin-top:0}' +
        // folded panels (a narrow stage in the deck): this one folds with the others; opened, it keeps to the lower half and scrolls
        '.kt-hands:not(.on){display:none!important}.kt-compact .kt-hands.on{display:none}' +
        '.kt-compact.kt-open .kt-hands.on{display:block;top:auto;bottom:8px;left:8px;right:8px;width:auto;max-height:calc(50% - 12px)}' +
        '@media (max-width:640px){.kt-hands{width:auto;right:12px;max-height:calc(50% - 18px)}.kt-hands-box{width:110px}}';
      document.head.appendChild(css);
    }

    return {
      el: el, claim: claim, undo: undo, reset: reset, state: st,
      hit: function (x, y) { if (!st.on) return false; var h = host.pick(x, y); return !!h && !h.c.ground; },
      /* the host's frame: the actions that grow while the pointer is held, and the atlas when something changed */
      step: function (dt) {
        if (!st.on) return;
        if (stroke && !stroke.hold && st.act !== 'grab') { stroke.t += dt; apply(); }
        last += dt; if (dirty && last > 0.08) { dirty = false; last = 0; paint(); }
      },
      show: function (on, mode) {
        st.on = !!on; st.mode = mode === 'select' ? 'select' : 'hands'; el.classList.toggle('on', st.on); el.classList.toggle('kt-select', st.mode === 'select');
        stroke = null; dirty = true;
        var f = feat(masks(obj())); segC.el.querySelectorAll('.kt-seg-btn').forEach(function (b) { if (b._v === 'feat') b.style.display = f ? '' : 'none'; });
        if (!f && st.chan === 'feat') { st.chan = 'rgb'; segC.set('rgb'); }
      },
      /* for tests: do in code what a pointer would */
      wandAt: function (i, j) { var c = masks(obj()); wand(c, i, j); selChanged(c); return count(c); },
      /* for tests: the brush (on = 1) or the eraser (on = 0) at control point (i, j) */
      brushAt: function (i, j, on) { var c = masks(obj()); daub(c, i, j, on ? 1 : 0); selChanged(c); return count(c); },
      /* for tests: which channel the atlas shows ('rgb', 'normal', 'alpha', 'height', 'feat', 'mask') */
      channel: function (v) { st.chan = v; segC.set(v); dirty = true; },
      object: obj
    };
  }
  KT.hands = { create: create };
})();
