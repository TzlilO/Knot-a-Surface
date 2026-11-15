/* Knot-a-Surface talk — module `basis`: "Pluck a control point".
   A B-spline / NURBS curve you can take apart: control points, basis functions, local support,
   degree, knots, rational weights and the Cox–de Boor recursion. Tool: SVG, redrawn from state.
   Standalone: talk/sims/basis.html. Depends on talk/nurbs.js, talk/kit.js and talk/ink.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var NS = 'http://www.w3.org/2000/svg';

  /* world box is x in [0,2], y in [0,1] (y up) */
  /* ten control points: a cubic then has seven spans, so one basis function (4 spans) is visibly local */
  var FREE = [[0.10, 0.34], [0.28, 0.86], [0.48, 0.16], [0.68, 0.80], [0.88, 0.24], [1.10, 0.84], [1.30, 0.18], [1.50, 0.82], [1.70, 0.26], [1.90, 0.62]];
  var S2 = Math.SQRT1_2, CX = 1, CY = 0.5, R = 0.38;
  var CIRCLE = {
    P: [[CX + R, CY], [CX + R, CY + R], [CX, CY + R], [CX - R, CY + R], [CX - R, CY],
        [CX - R, CY - R], [CX, CY - R], [CX + R, CY - R], [CX + R, CY]],
    w: [1, S2, 1, S2, 1, S2, 1, S2, 1],
    U: [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1], p: 2
  };

  var TEXTS = KT.ink.texts, MISSILE = KT.ink.missile;       // the drawings (talk/ink.js)
  var morphSets = KT.ink.morphSets;
  var SHAPES = ['letters', 'handwriting', 'missile'];

  /* ---------- how close is the estimate? ----------
     Ground truth = one of the drawings. Estimate = a least-squares cubic with fewer control points,
     read out at a number of samples. Both numbers are budgets shared by the strokes in proportion to
     their length. The gap = distance from each ground-truth point to the sampled estimate. */
  function trace(c) {                      // points along a curve, a dozen per control point, + chord-length parameter
    var n = c.P.length * 12, pts = [], T = [0], i, L = 0;
    for (i = 0; i <= n; i++) pts.push(NB.curvePoint(c.P, null, 3, c.U, i / n));
    for (i = 1; i <= n; i++) { L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); T.push(L); }
    return { pts: pts, T: T.map(function (t) { return t / L; }), len: L };
  }
  var _F = new Float64Array(8);
  function lsq(g, n) {                     // normal equations (banded, so plain elimination is cheap)
    var U = NB.clampedKnots(n, 3), A = [], bx = new Float64Array(n), by = new Float64Array(n), i, j, k, a, c, f, P = [];
    for (i = 0; i < n; i++) A.push(new Float64Array(n));
    for (k = 0; k < g.pts.length; k++) {
      var sp = NB.findSpan(n, 3, g.T[k], U); NB.basisFuns(sp, g.T[k], 3, U, _F);
      for (a = 0; a < 4; a++) { i = sp - 3 + a; bx[i] += _F[a] * g.pts[k][0]; by[i] += _F[a] * g.pts[k][1]; for (c = 0; c < 4; c++) A[i][sp - 3 + c] += _F[a] * _F[c]; }
    }
    for (i = 0; i < n; i++) A[i][i] += 1e-7;
    for (i = 0; i < n; i++) for (j = i + 1; j < Math.min(n, i + 4); j++) {
      f = A[j][i] / A[i][i]; if (!f) continue;
      for (c = i; c < Math.min(n, i + 4); c++) A[j][c] -= f * A[i][c];
      bx[j] -= f * bx[i]; by[j] -= f * by[i];
    }
    for (i = n - 1; i >= 0; i--) {
      var x = bx[i], y = by[i];
      for (c = i + 1; c < Math.min(n, i + 4); c++) { x -= A[i][c] * P[c][0]; y -= A[i][c] * P[c][1]; }
      P[i] = [x / A[i][i], y / A[i][i]];
    }
    return { P: P, U: U };
  }
  /* fit every stroke of `truth` with nT control points in all and read it at mT samples; near = true also
     returns, per ground-truth point, the closest point of the estimate (for drawing the gap) */
  function estimate(truth, nT, mT, near) {
    var total = 0, sq = 0, cnt = 0, mx = 0, strokes = [];
    truth.forEach(function (g) { total += g.len; });
    truth.forEach(function (g) {
      var n = Math.max(4, Math.round(nT * g.len / total)), m = Math.max(2, Math.round(mT * g.len / total)), f = lsq(g, n), est = [], i, k;
      for (i = 0; i < m; i++) est.push(NB.curvePoint(f.P, null, 3, f.U, i / (m - 1)));
      var w = 3 + Math.ceil(m * 0.06), hit = near ? [] : null;
      for (k = 0; k < g.pts.length; k++) {
        var q = g.pts[k], c0 = Math.round(g.T[k] * (m - 1)), bd = 1e9, bq = null;
        for (i = Math.max(0, c0 - w); i < Math.min(m - 1, c0 + w); i++) {
          var a = est[i], b = est[i + 1], vx = b[0] - a[0], vy = b[1] - a[1], t = KT.clamp(((q[0] - a[0]) * vx + (q[1] - a[1]) * vy) / (vx * vx + vy * vy || 1), 0, 1);
          var d = Math.hypot(q[0] - a[0] - t * vx, q[1] - a[1] - t * vy);
          if (d < bd) { bd = d; if (near) bq = [a[0] + t * vx, a[1] + t * vy]; }
        }
        sq += bd * bd; cnt++; if (bd > mx) mx = bd; if (near) hit.push(bq);
      }
      strokes.push({ P: f.P, est: est, hit: hit });
    });
    return { strokes: strokes, rms: Math.sqrt(sq / cnt), max: mx };
  }
  var SWEEP = [12, 16, 22, 30, 42, 58, 80, 110, 150, 210];

  var SCENES = {
    ducks:     { cap: '<b>a spline</b> · a flexible strip, pinned by weights', wiggle: 4 },
    letters:   { cap: '<b>one spline per letter</b> · twelve short curves, each with its own control points', multi: 'letters', rate: 70,
                 hint: 'drag any amber point' },
    script:    { cap: '<b>handwriting</b> · three strokes · the pen is a point sliding along the curve', multi: 'script', rate: 26, again: 1 },
    morph:     { cap: '<b>morphing</b> · letters → handwriting → a missile · only the control points move', morph: 1 },
    lens:      { cap: '<b>Gaussians on the tracks</b> · a fixed number, spread over what is in view · move closer: same count, finer', morph: 1, lens: 1,
                 hint: 'zoom with the slider · drag to pan' },
    fit:       { cap: '<b>how close?</b> · control points set what the curve can say · samples set how finely it is read', fit: 1 },
    drag:      { cap: '<b>control points</b> steer · the curve follows', wiggle: 5, hint: 'drag any amber point' },
    blend:     { cap: '<b>every curve point</b> is a weighted average of nearby control points', basis: 1, blend: 1,
                 hint: 'scrub the plot to move u' },
    local:     { cap: '<b>local support</b> · one point moves, only p + 1 spans change', basis: 1, local: 1, sel: 4, wiggle: 4 },
    degree:    { cap: '<b>degree p</b> · smoothness C<sup>p−1</sup> across every knot', basis: 1, cycle: 1 },
    knots:     { cap: '<b>knots</b> · where the polynomial pieces join', basis: 1, ticks: 1, knotAuto: 1,
                 hint: 'drag the amber knots on the axis' },
    rational:  { cap: '<b>rational weights</b> · the R in NURBS — exact conics', basis: 1, circle: 1, sel: 1,
                 hint: 'select a point · change its weight' },
    recursion: { cap: '<b>Cox–de Boor</b> · boxes → hats → parabolas → cubics', basis: 1, pyramid: 1, sel: 4 }
  };

  KT.register('basis', {
    title: 'Pluck a control point',
    blurb: 'A B-spline curve taken apart: drag control points, watch the basis functions that blend them, ' +
           'slide knots, raise the degree, and add rational weights until the curve is an exact circle.',
    tool: 'SVG · direct manipulation',
    scenes: [
      { id: 'ducks', label: 'The draftsman’s spline' }, { id: 'letters', label: 'A spline per letter' },
      { id: 'script', label: 'Handwriting' }, { id: 'morph', label: 'Morph' }, { id: 'lens', label: 'Locked to the view' },
      { id: 'fit', label: 'How close?' },
      { id: 'drag', label: 'Control points' },
      { id: 'blend', label: 'Blending' }, { id: 'local', label: 'Local support' },
      { id: 'degree', label: 'Degree' }, { id: 'knots', label: 'Knots' },
      { id: 'rational', label: 'Rational weights' }, { id: 'recursion', label: 'Cox–de Boor' }
    ],
    create: function (env) {
      var root = env.root, ui = env.ui;
      var svg = document.createElementNS(NS, 'svg');
      svg.setAttribute('class', 'kt-canvas');
      root.appendChild(svg);
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);

      var st = { P: [], w: [], U: [], p: 3, sel: 3, u: 0.37, scene: 'ducks', sc: SCENES.ducks,
                 bh: 0, ghost: null, userU: false, touched: false, t: 0, cycleT: 0, circle: false };
      var W = 0, H = 0, dirty = true, drag = null, L = {};

      /* ---------- controls ---------- */
      var panel = ui.panel(root, { pos: 'tr', title: 'curve' });
      var segDeg = ui.seg(panel, { label: 'degree p', value: 3,
        options: [{ v: 1, label: '1' }, { v: 2, label: '2' }, { v: 3, label: '3' }],
        onChange: function (v) { st.touched = true; setDegree(v); } });
      var segM = ui.seg(panel, { label: 'shape', value: 0, options: SHAPES.map(function (n, i) { return { v: i, label: n }; }),
        onChange: function (v) { st.touched = true; if (st.m && v !== st.m.from) { st.m.to = v; st.m.k = 0; st.m.hold = 0; } } });
      var tgF = ui.toggle(panel, { label: 'freeze the shape', value: false, onChange: function (v) { st.frozen = v; } });
      /* the lens scene's controls: how many Gaussians, how close */
      var slG = ui.slider(panel, { label: 'gaussians', min: 100, max: 1600, step: 50, value: 400, fmt: function (v) { return v.toFixed(0); },
        onInput: function (v) { st.z.n = v; dirty = true; } });
      var slZ = ui.slider(panel, { label: 'zoom', min: 0, max: 1, step: 0.01, value: 0, fmt: function (v) { return '×' + Math.pow(12, v).toFixed(1); },
        onInput: function (v) { st.z.user = true; st.z.zoom = Math.pow(12, v); dirty = true; } });
      /* the fit scene's controls */
      var segT = ui.seg(panel, { label: 'truth', value: 'script', options: [{ v: 'script', label: 'handwriting' }, { v: 'missile', label: 'missile' }],
        onChange: function (v) { st.touched = true; setTruth(v); } });
      var slN = ui.slider(panel, { label: 'control points', min: 12, max: 210, step: 1, value: 14, fmt: function (v) { return v.toFixed(0); },
        onInput: function (v) { st.touched = true; st.f.n = v; refit(false); } });
      var slM = ui.slider(panel, { label: 'samples', min: 0, max: 1, step: 0.005, value: 0, fmt: function (v) { return samples(v).toFixed(0); },
        onInput: function (v) { st.touched = true; st.f.m = samples(v); refit(true); } });
      var segD = ui.seg(panel, { label: 'drawn as', value: 'lines', options: [{ v: 'lines', label: 'lines' }, { v: 'gauss', label: 'gaussians' }],
        onChange: function (v) { st.touched = true; st.f.mode = v; dirty = true; } });
      function samples(v) { return Math.round(16 * Math.pow(75, v)); }       // 16 .. 1200, on a log scale
      var slW = ui.slider(panel, { label: 'weight w', min: 0.1, max: 4, step: 0.05, value: 1,
        fmt: function (v) { return v.toFixed(2); },
        onInput: function (v) { st.touched = true; st.w[st.sel] = v; dirty = true; } });
      var rowB = ui.el('div', 'kt-row'); panel.appendChild(rowB);
      ui.button(rowB, { label: 'all w = 1', onClick: function () {
        for (var i = 0; i < st.w.length; i++) st.w[i] = 1; slW.set(1); dirty = true; } });
      ui.button(rowB, { label: 'reset', onClick: function () { load(st.circle, true); } });
      var rd = ui.readout(panel, { label: 'continuity', tone: 'surf' });
      var rdErr = ui.readout(panel, { label: 'circle error', tone: 'ok' });

      function load(circle, keepScene) {
        st.circle = circle;
        if (circle) {
          st.P = CIRCLE.P.map(function (q) { return q.slice(); });
          st.w = CIRCLE.w.slice(); st.U = CIRCLE.U.slice(); st.p = 2;
        } else {
          st.P = FREE.map(function (q) { return q.slice(); });
          st.w = FREE.map(function () { return 1; }); st.p = 3;
          st.U = NB.clampedKnots(st.P.length, st.p);
        }
        st.base = st.P.map(function (q) { return q.slice(); });
        segDeg.set(st.p); slW.set(st.w[st.sel] || 1);
        if (keepScene) st.touched = true;
        dirty = true;
      }
      function setDegree(p) {
        if (st.circle) load(false);
        st.p = p; st.U = NB.clampedKnots(st.P.length, p); segDeg.set(p); dirty = true;
        if (st.multi && !st.m) st.multi.forEach(knot);
      }
      /* the letter scenes: several curves, each { P, w, U, p, at } (at = where its points start in writing order) */
      function knot(c) { c.p = Math.min(st.p, c.P.length - 1); c.U = NB.clampedKnots(c.P.length, c.p); }
      function loadText(name) {
        var at = 0;
        st.multi = TEXTS[name]().map(function (P) {
          var c = { P: P, w: P.map(function () { return 1; }), at: at }; knot(c); at += P.length; return c;
        });
        st.pen = 0; st.total = at;
      }
      /* the fit scene: st.f = { truth, n, m, mode, now, curve, floor } */
      function setTruth(name) {
        var curves = name === 'missile' ? MISSILE : TEXTS.script(), lo = 1e9, hi = -1e9;
        st.f.truth = curves.map(function (P) { return trace({ P: P, U: NB.clampedKnots(P.length, 3) }); });
        st.f.truth.forEach(function (g) { g.pts.forEach(function (q) { lo = Math.min(lo, q[1]); hi = Math.max(hi, q[1]); }); });
        st.f.h = hi - lo; st.f.name = name; segT.set(name);
        st.f.floor = SWEEP.map(function (n) { return estimate(st.f.truth, n, 1500).rms; });     // the spline alone, read finely
        refit(true);
      }
      function refit(sweep) {
        var F = st.f; F.now = estimate(F.truth, F.n, F.m, true);
        if (sweep) F.curve = SWEEP.map(function (n) { return estimate(F.truth, n, F.m).rms; });
        dirty = true;
      }
      function loadFit() {
        st.f = { n: 14, m: 20, mode: segD.get() }; slN.set(14); slM.set(Math.log(20 / 16) / Math.log(75));
        setTruth(segT.get());
      }
      /* the morph scene: st.m.sets = the three shapes; the curves on screen are a blend of two of them */
      function loadMorph() {
        st.m = { sets: morphSets(), from: 0, to: 0, k: 0, hold: 1.6 };
        st.multi = st.m.sets[0].map(function (c) { return { P: c.P.map(function (q) { return q.slice(); }), w: c.w, U: c.U.slice(), p: 3, at: 0 }; });
        st.pen = st.total = 1e9; segM.set(0);       // nothing is being written: every curve is drawn in full
      }
      /* the lens scene: the morph's curves, drawn ONLY as Gaussians that sit on them. A fixed number is spread
         evenly along the part of the curves inside the view window, so moving closer spends the same Gaussians
         on less track. st.z = { n, zoom, cx, cy, user } */
      var uid = 'kb' + Math.floor(Math.random() * 1e6), PEN = 0.0055;
      function drawLens() {
        var Z = st.z, zoom = Z.zoom, hw = 1 / zoom, hh = 0.5 / zoom, S = L.s * zoom, X0 = L.ox + L.s, Y0 = L.oy + L.s / 2, h, i, k;
        Z.cx = KT.clamp(Z.cx, hw, 2 - hw); Z.cy = KT.clamp(Z.cy, hh, 1 - hh);
        function P(q) { return [X0 + (q[0] - Z.cx) * S, Y0 - (q[1] - Z.cy) * S]; }
        // the tracks as polylines, and which of their pieces lie inside the window
        var tracks = st.multi.map(function (c) { var n = c.P.length * 14, o = []; for (i = 0; i <= n; i++) o.push(NB.curvePoint(c.P, c.w, 3, c.U, i / n)); return o; });
        var segs = [], len = 0, all = 0;
        tracks.forEach(function (T) {
          for (i = 0; i + 1 < T.length; i++) {
            var a = T[i], b = T[i + 1], l = Math.hypot(b[0] - a[0], b[1] - a[1]), mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2; all += l;
            if (Math.abs(mx - Z.cx) <= hw && Math.abs(my - Z.cy) <= hh && l > 0) { segs.push([a, b, l]); len += l; }
          }
        });
        // N Gaussians, one per equal share of the visible track: centre on the track, long axis along it
        var G = [], N = Z.n, step = len / N, at = 0, need = step / 2, si = 0;
        for (k = 0; k < N && si < segs.length; k++) {
          while (si < segs.length && at + segs[si][2] < need) { at += segs[si][2]; si++; }
          if (si >= segs.length) break;
          var sg = segs[si], t = (need - at) / sg[2];
          G.push([sg[0][0] + (sg[1][0] - sg[0][0]) * t, sg[0][1] + (sg[1][1] - sg[0][1]) * t, Math.atan2(-(sg[1][1] - sg[0][1]), sg[1][0] - sg[0][0])]);
          need += step;
        }
        var bx = L.ox, by = L.oy, bw = 2 * L.s, bhh = L.s;
        h = '<defs><radialGradient id="' + uid + 'g"><stop offset="0" stop-color="' + C.splat + '"/><stop offset=".5" stop-color="' + C.splat + '" stop-opacity=".85"/>' +
            '<stop offset="1" stop-color="' + C.splat + '" stop-opacity="0"/></radialGradient>' +
            '<clipPath id="' + uid + 'c"><rect x="' + f(bx) + '" y="' + f(by) + '" width="' + f(bw) + '" height="' + f(bhh) + '"/></clipPath></defs>' +
            '<rect x="' + f(bx) + '" y="' + f(by) + '" width="' + f(bw) + '" height="' + f(bhh) + '" fill="none" stroke="' + C.line + '"/>' +
            '<g clip-path="url(#' + uid + 'c)">';
        tracks.forEach(function (T) {
          h += '<path d="' + T.map(function (q, j) { var a = P(q); return (j ? 'L' : 'M') + f(a[0]) + ' ' + f(a[1]); }).join('') + '" fill="none" stroke="' + C.surf + '" stroke-width="1" opacity=".3"/>';
        });
        var rx = Math.max(step * 0.62, PEN) * S * 1.55, ry = PEN * S * 1.55;
        G.forEach(function (g) { var a = P(g); h += '<ellipse cx="' + f(a[0]) + '" cy="' + f(a[1]) + '" rx="' + f(rx) + '" ry="' + f(ry) + '" transform="rotate(' + f(g[2] * 57.2958) + ' ' + f(a[0]) + ' ' + f(a[1]) + ')" fill="url(#' + uid + 'g)"/>'; });
        h += '</g>';
        // the outside view: the whole drawing, the window, and where the Gaussians are
        var iw = Math.min(bw * 0.3, 250), ih = iw / 2, ix = bx + 8, iy = by + bhh - ih - 8, m = iw / 2;
        function Q(q) { return [ix + q[0] * m, iy + (1 - q[1]) * m]; }
        h += '<rect x="' + f(ix) + '" y="' + f(iy) + '" width="' + f(iw) + '" height="' + f(ih) + '" fill="' + C.bg + '" stroke="' + C.dim + '" opacity=".94"/>';
        tracks.forEach(function (T) {
          h += '<path d="' + T.filter(function (q, j) { return j % 3 === 0 || j === T.length - 1; }).map(function (q, j) { var a = Q(q); return (j ? 'L' : 'M') + f(a[0]) + ' ' + f(a[1]); }).join('') + '" fill="none" stroke="' + C.surf + '" stroke-width=".8" opacity=".4"/>';
        });
        var dots = ''; G.forEach(function (g) { var a = Q(g); dots += 'M' + f(a[0] - 0.9) + ' ' + f(a[1]) + 'h1.8'; });
        var w0 = Q([Z.cx - hw, Z.cy + hh]);
        h += '<path d="' + dots + '" stroke="' + C.splat + '" stroke-width="1.8" stroke-linecap="round"/>' +
             '<rect x="' + f(w0[0]) + '" y="' + f(w0[1]) + '" width="' + f(2 * hw * m) + '" height="' + f(2 * hh * m) + '" fill="none" stroke="' + C.text + '" stroke-width="1.2"/>' +
             '<text x="' + f(ix + 6) + '" y="' + f(iy + 13) + '" fill="' + C.mute + '" class="kt-t">outside view</text>';
        rd.el.querySelector('.kt-lab').textContent = 'in view';
        rd.set(G.length + ' on ' + Math.round(len / all * 100) + ' % of the track');
        rdErr.el.style.display = 'none'; slW.el.style.display = rowB.style.display = 'none';
        return h;
      }
      function blend0(shape) {                 // show one shape as it is
        st.multi.forEach(function (c, i) { var a = st.m.sets[shape][i], j; for (j = 0; j < c.P.length; j++) { c.P[j][0] = a.P[j][0]; c.P[j][1] = a.P[j][1]; } for (j = 0; j < c.U.length; j++) c.U[j] = a.U[j]; });
      }
      function blend() {
        var m = st.m, e = m.k * m.k * m.k * (m.k * (6 * m.k - 15) + 10);
        st.multi.forEach(function (c, i) {
          var a = m.sets[m.from][i], b = m.sets[m.to][i], j;
          for (j = 0; j < c.P.length; j++) { c.P[j][0] = a.P[j][0] + (b.P[j][0] - a.P[j][0]) * e; c.P[j][1] = a.P[j][1] + (b.P[j][1] - a.P[j][1]) * e; }
          for (j = 0; j < c.U.length; j++) c.U[j] = a.U[j] + (b.U[j] - a.U[j]) * e;
        });
      }

      /* ---------- layout ---------- */
      function layout() {
        var padX = Math.max(22, W * 0.05), topPad = 46;
        var bh = st.bh * H, topH = H - bh;
        var s = Math.min((W - 2 * padX) / 2, (topH - topPad - 26) / 1);
        L.s = s; L.ox = (W - 2 * s) / 2; L.oy = topPad + (topH - topPad - 26 - s) / 2;
        if (st.sc.fit && W > 640) L.ox = Math.min(L.ox, Math.max(padX, W - 2 * s - 250));   // keep the drawing clear of the panel
        L.bx0 = padX + 8; L.bx1 = W - padX - 8; L.by0 = topH + 6; L.by1 = H - 44; L.bh = bh;
      }
      function px(q) { return [L.ox + q[0] * L.s, L.oy + (1 - q[1]) * L.s]; }
      function unpx(x, y) { return [KT.clamp((x - L.ox) / L.s, 0, 2), KT.clamp(1 - (y - L.oy) / L.s, 0, 1)]; }
      function bx(u) { return L.bx0 + u * (L.bx1 - L.bx0); }

      /* rational basis row at u: span k and the p+1 weights that blend P[k-p..k] */
      var _N = new Float64Array(8);
      function row(u) {
        var m = st.P.length, p = st.p, k = NB.findSpan(m, p, u, st.U), sum = 0, r, out = [];
        NB.basisFuns(k, u, p, st.U, _N);
        for (r = 0; r <= p; r++) { out[r] = _N[r] * st.w[k - p + r]; sum += out[r]; }
        for (r = 0; r <= p; r++) out[r] /= sum;
        return { k: k, R: out };
      }
      function pt(u) { return NB.curvePoint(st.P, st.w, st.p, st.U, u); }
      function f(v) { return v.toFixed(1); }
      function curvePath(u0, u1, n) {
        var d = '', i, q;
        for (i = 0; i <= n; i++) { q = px(pt(u0 + (u1 - u0) * i / n)); d += (i ? 'L' : 'M') + f(q[0]) + ' ' + f(q[1]); }
        return d;
      }

      /* ---------- drawing ---------- */
      /* several splines at once; each is drawn as far as the pen has got */
      function drawText() {
        var sc = st.sc, big = sc.multi === 'letters', h = '', tip = null, n = 0;
        var rp = KT.clamp(L.s * (big ? 0.011 : 0.006), 1.6, big ? 4.6 : 2.6), sw = KT.clamp(L.s * 0.008, 1.6, big ? 3.4 : 3);   // small stages: finer marks
        st.multi.forEach(function (c) {
          var u = KT.clamp((st.pen - c.at) / c.P.length, 0, 1), d = '', i, q, seg = c.P.length * 10;
          n += c.P.length;
          h += '<path d="' + c.P.map(function (k, j) { q = px(k); return (j ? 'L' : 'M') + f(q[0]) + ' ' + f(q[1]); }).join('') +
               '" fill="none" stroke="' + C.ctrl + '" stroke-width="1" stroke-dasharray="3 4" opacity="' + (big ? 0.55 : 0.3) + '"/>';
          if (u > 0) {
            for (i = 0; i <= seg; i++) { q = px(NB.curvePoint(c.P, c.w, c.p, c.U, u * i / seg)); d += (i ? 'L' : 'M') + f(q[0]) + ' ' + f(q[1]); }
            h += '<path d="' + d + '" fill="none" stroke="' + C.surf + '" stroke-width="' + f(sw) + '" stroke-linecap="round" stroke-linejoin="round"/>';
            if (u < 1) tip = q;
          }
          for (i = 0; i < c.P.length; i++) {
            q = px(c.P[i]);
            h += '<circle cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" r="' + f(rp) + '" fill="' + C.ctrl + '" stroke="' + C.bg + '" stroke-width="' + (rp > 3 ? 1.5 : 0.8) + '"' + (big ? '' : ' opacity=".7"') + '/>';
          }
        });
        if (tip) h += '<circle cx="' + f(tip[0]) + '" cy="' + f(tip[1]) + '" r="6" fill="' + C.text + '" stroke="' + C.bg + '" stroke-width="2"/>';
        rd.el.querySelector('.kt-lab').textContent = 'curves';
        rd.set(st.multi.length + ' splines · ' + n + ' pts');
        rdErr.el.style.display = 'none'; slW.el.style.display = rowB.style.display = 'none';
        return h;
      }

      /* ground truth, the sampled estimate, the gap between them, and the gap against the control-point count */
      function drawFit() {
        var F = st.f, h = '', i, q, a, whisk = '', every = Math.max(1, Math.round(F.truth.reduce(function (n, g) { return n + g.pts.length; }, 0) / 420));
        function path(P) { return P.map(function (k, j) { q = px(k); return (j ? 'L' : 'M') + f(q[0]) + ' ' + f(q[1]); }).join(''); }
        F.now.strokes.forEach(function (sk, si) {
          var g = F.truth[si];
          for (i = 0; i < g.pts.length; i += every) { q = px(g.pts[i]); a = px(sk.hit[i]); if (Math.hypot(q[0] - a[0], q[1] - a[1]) > 1.2) whisk += 'M' + f(q[0]) + ' ' + f(q[1]) + 'L' + f(a[0]) + ' ' + f(a[1]); }
          h += '<path d="' + path(g.pts) + '" fill="none" stroke="' + C.dim + '" stroke-width="1.6" stroke-linejoin="round" opacity=".8"/>';
        });
        h += '<path d="' + whisk + '" stroke="' + C.err + '" stroke-width="1.1" opacity=".8"/>';
        F.now.strokes.forEach(function (sk) {
          if (F.mode === 'lines') h += '<path d="' + path(sk.est) + '" fill="none" stroke="' + C.surf + '" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>';
          else for (i = 0; i < sk.est.length; i++) {       // one Gaussian per sample: centred on it, as long as the gap to its neighbours
            var e0 = px(sk.est[Math.max(0, i - 1)]), e1 = px(sk.est[Math.min(sk.est.length - 1, i + 1)]); q = px(sk.est[i]);
            var len = Math.hypot(e1[0] - e0[0], e1[1] - e0[1]) / (i && i < sk.est.length - 1 ? 2 : 1);
            h += '<ellipse cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" rx="' + f(Math.max(1.6, len * 0.62)) + '" ry="2.6" transform="rotate(' + f(Math.atan2(e1[1] - e0[1], e1[0] - e0[0]) * 57.2958) + ' ' + f(q[0]) + ' ' + f(q[1]) + ')" fill="' + C.splat + '" opacity=".72"/>';
          }
          if (F.n <= 110) {
            h += '<path d="' + path(sk.P) + '" fill="none" stroke="' + C.ctrl + '" stroke-width="1" stroke-dasharray="2 4" opacity=".45"/>';
            for (i = 0; i < sk.P.length; i++) { q = px(sk.P[i]); h += '<circle cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" r="2.6" fill="' + C.ctrl + '"/>'; }
          }
        });
        // the plot: mean gap (share of the drawing's height) against control points, log-log
        if (L.bh > 30 && F.curve) {
          var x0 = L.bx0 + 34, x1 = L.bx1, y0 = L.by0 + 8, y1 = L.by1, lo = Math.log(0.0005), hi = Math.log(0.25);
          var X = function (n) { return x0 + (Math.log(n) - Math.log(12)) / (Math.log(210) - Math.log(12)) * (x1 - x0); };
          var Y = function (e) { return y1 - (KT.clamp(Math.log(e / F.h), lo, hi) - lo) / (hi - lo) * (y1 - y0); };
          [0.001, 0.01, 0.1].forEach(function (e) {
            h += '<path d="M' + f(x0) + ' ' + f(Y(e * F.h)) + 'H' + f(x1) + '" stroke="' + C.line + '" stroke-width="1"/>' +
                 '<text x="' + f(x0 - 6) + '" y="' + f(Y(e * F.h) + 4) + '" text-anchor="end" fill="' + C.mute + '" class="kt-t">' + (e * 100) + '%</text>';
          });
          [12, 30, 80, 210].forEach(function (n) { h += '<text x="' + f(X(n)) + '" y="' + f(y1 + 16) + '" text-anchor="middle" fill="' + C.mute + '" class="kt-t">' + n + '</text>'; });
          var line = function (arr) { return arr.map(function (e, j) { return (j ? 'L' : 'M') + f(X(SWEEP[j])) + ' ' + f(Y(e)); }).join(''); };
          h += '<path d="' + line(F.floor) + '" fill="none" stroke="' + C.surf + '" stroke-width="1.4" stroke-dasharray="4 4" opacity=".8"/>';
          h += '<path d="' + line(F.curve) + '" fill="none" stroke="' + C.err + '" stroke-width="2"/>';
          h += '<circle cx="' + f(X(F.n)) + '" cy="' + f(Y(F.now.rms)) + '" r="5.5" fill="' + C.err + '" stroke="' + C.bg + '" stroke-width="2"/>';
          h += '<text x="' + f(x1) + '" y="' + f(y0 + 4) + '" text-anchor="end" fill="' + C.mute + '" class="kt-t">mean gap against control points · ' +
               '<tspan fill="' + C.err + '">with ' + F.m + ' samples</tspan> · <tspan fill="' + C.surf + '">with plenty</tspan></text>';
        }
        rd.el.querySelector('.kt-lab').textContent = 'mean gap';
        rd.set((F.now.rms / F.h * 100).toFixed(2) + ' % · worst ' + (F.now.max / F.h * 100).toFixed(1) + ' %'); rd.tone('err');
        rdErr.el.style.display = 'none'; slW.el.style.display = rowB.style.display = 'none';
        return h;
      }

      function draw() {
        layout();
        var sc = st.sc, p = st.p, m = st.P.length, U = st.U, i, q, h = '', a;
        var showBasis = L.bh > 30;

        // drafting grid behind the curve
        h += '<g stroke="' + C.line + '" stroke-width="1" opacity=".55">';
        for (i = 0; i <= 8; i++) { a = L.ox + i * L.s / 4; h += '<path d="M' + f(a) + ' ' + f(L.oy) + 'V' + f(L.oy + L.s) + '"/>'; }
        for (i = 0; i <= 4; i++) { a = L.oy + i * L.s / 4; h += '<path d="M' + f(L.ox) + ' ' + f(a) + 'H' + f(L.ox + 2 * L.s) + '"/>'; }
        h += '</g>';
        if (st.z) { svg.innerHTML = drawLens(); return; }
        if (st.f) { svg.innerHTML = h + drawFit(); return; }
        rd.tone('surf');
        if (st.multi) { svg.innerHTML = h + drawText(); return; }

        // control polygon
        var poly = st.P.map(function (c, j) { q = px(c); return (j ? 'L' : 'M') + f(q[0]) + ' ' + f(q[1]); }).join('');
        h += '<path d="' + poly + '" fill="none" stroke="' + C.ctrl + '" stroke-width="1.2" stroke-dasharray="3 5" opacity=".6"/>';

        // ghost of the curve before the edit (local support)
        if (sc.local && st.ghost) h += '<path d="' + st.ghost + '" fill="none" stroke="' + C.dim + '" stroke-width="1.5" stroke-dasharray="2 6"/>';

        // the curve
        var dim = sc.local ? 0.3 : 1;
        h += '<path d="' + curvePath(0, 1, 220) + '" fill="none" stroke="' + C.surf + '" stroke-width="3.2" stroke-linecap="round" opacity="' + dim + '"/>';
        if (sc.local) {
          var ua = U[st.sel], ub = U[st.sel + p + 1];
          h += '<path d="' + curvePath(ua, ub, 120) + '" fill="none" stroke="' + C.surf + '" stroke-width="5" stroke-linecap="round"/>';
        }

        // knot images + iso-parameter beads (parameter speed)
        if (sc.ticks) {
          for (i = 1; i < 28; i++) { q = px(pt(i / 28)); h += '<circle cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" r="2.2" fill="' + C.bg + '" stroke="' + C.text + '" stroke-width="1"/>'; }
          for (i = p + 1; i < m; i++) {
            var u = U[i], q0 = px(pt(Math.max(0, u - 0.002))), q1 = px(pt(Math.min(1, u + 0.002)));
            var tx = q1[0] - q0[0], ty = q1[1] - q0[1], tl = Math.hypot(tx, ty) || 1; q = px(pt(u));
            h += '<path d="M' + f(q[0] - ty / tl * 11) + ' ' + f(q[1] + tx / tl * 11) + 'L' + f(q[0] + ty / tl * 11) + ' ' + f(q[1] - tx / tl * 11) +
                 '" stroke="' + C.ctrl + '" stroke-width="2.4" stroke-linecap="round"/>';
          }
        }

        // blending: the point S(u) and the p+1 control points that make it
        var rw = null, act = {};
        if (sc.blend) {
          rw = row(st.u); var sq = px(pt(st.u));
          for (i = 0; i <= p; i++) {
            var ci = rw.k - p + i, wgt = rw.R[i]; act[ci] = wgt; q = px(st.P[ci]);
            h += '<path d="M' + f(sq[0]) + ' ' + f(sq[1]) + 'L' + f(q[0]) + ' ' + f(q[1]) + '" stroke="' + C.ctrl +
                 '" stroke-width="' + f(0.6 + 7 * wgt) + '" stroke-linecap="round" opacity="' + (0.2 + 0.7 * wgt).toFixed(2) + '"/>';
          }
          h += '<circle cx="' + f(sq[0]) + '" cy="' + f(sq[1]) + '" r="7" fill="' + C.text + '" stroke="' + C.bg + '" stroke-width="2"/>';
        }

        // control points
        for (i = 0; i < m; i++) {
          q = px(st.P[i]);
          var rr = 7 + (act[i] != null ? 13 * act[i] : 0) + (sc.circle ? 5 * (st.w[i] - 1) : 0);
          rr = KT.clamp(rr, 4, 26);
          var on = (sc.local || sc.circle || sc.pyramid) && i === st.sel;
          if (on) h += '<circle cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" r="' + f(rr + 7) + '" fill="none" stroke="' + C.ctrl + '" stroke-width="1.2" opacity=".7"/>';
          h += '<circle cx="' + f(q[0]) + '" cy="' + f(q[1]) + '" r="' + f(rr) + '" fill="' + C.ctrl + '" stroke="' + C.bg + '" stroke-width="2"' +
               (sc.blend && act[i] == null ? ' opacity=".35"' : '') + '/>';
          if (act[i] != null && act[i] > 0.02) {
            h += '<text x="' + f(q[0]) + '" y="' + f(q[1] - rr - 8) + '" text-anchor="middle" fill="' + C.ctrl + '" class="kt-t">' + Math.round(act[i] * 100) + '%</text>';
          } else if (sc.circle) {
            var mid = Math.abs(st.P[i][0] - CX) < 0.02, side = st.P[i][0] >= CX ? 1 : -1;
            h += '<text x="' + f(q[0] + (mid ? 0 : side * (rr + 8))) + '" y="' + f(q[1] + (mid ? (st.P[i][1] >= CY ? -rr - 9 : rr + 17) : 4)) +
                 '" text-anchor="' + (mid ? 'middle' : side > 0 ? 'start' : 'end') + '" fill="' + C.dim + '" class="kt-t">w ' + st.w[i].toFixed(2) + '</text>';
          }
        }

        if (showBasis) h += drawBasis(rw);
        svg.innerHTML = h;

        // readouts
        rd.el.querySelector('.kt-lab').textContent = 'continuity';
        rd.set('C' + (p - 1 < 0 ? 0 : p - 1) + ' · ' + m + ' pts · ' + U.length + ' knots');
        rdErr.el.style.display = sc.circle ? '' : 'none';
        slW.el.style.display = sc.circle || !env.embed ? '' : 'none';
        rowB.style.display = sc.circle || !env.embed ? '' : 'none';
        if (sc.circle) {
          var e = 0;
          for (i = 0; i <= 200; i++) { q = pt(i / 200); e = Math.max(e, Math.abs(Math.hypot(q[0] - CX, q[1] - CY) - R)); }
          rdErr.set((e / R * 100).toFixed(e < 1e-9 ? 0 : 2) + ' % of r'); rdErr.tone(e < 1e-9 ? 'ok' : 'err');
        }
      }

      function drawBasis(rw) {
        var sc = st.sc, p = st.p, m = st.P.length, U = st.U, h = '', i, s, q;
        var x0 = L.bx0, x1 = L.bx1, y0 = L.by0, y1 = L.by1, n = 180;
        var rows = sc.pyramid ? p + 1 : 1, gap = 12, rh = (y1 - y0 - gap * (rows - 1)) / rows;
        // family tree of the selected cubic (pyramid): N_{sel,p} needs N_{sel..sel+p-q, q}
        for (var r = 0; r < rows; r++) {
          var deg = sc.pyramid ? r : p, top = y0 + r * (rh + gap), bot = top + rh;
          var cnt = U.length - deg - 1;
          h += '<path d="M' + f(x0) + ' ' + f(bot) + 'H' + f(x1) + '" stroke="' + C.line + '" stroke-width="1"/>';
          var amp = rh - (sc.pyramid ? 22 : 14);
          if (sc.pyramid) h += '<text x="' + f(x0) + '" y="' + f(top + 10) + '" fill="' + C.dim + '" class="kt-t">p = ' + deg + '</text>';
          for (i = 0; i < cnt; i++) {
            var d = '', any = false, vals = [];
            for (s = 0; s <= n; s++) {
              var u = s / n, v;
              if (deg === p && !sc.pyramid) {
                // rational basis via the active row (cheap, and correct for weights != 1)
                var k = NB.findSpan(m, p, u, U); v = 0;
                if (i >= k - p && i <= k) { NB.basisFuns(k, u, p, U, _N); var sum = 0, t;
                  for (t = 0; t <= p; t++) sum += _N[t] * st.w[k - p + t];
                  v = _N[i - (k - p)] * st.w[i] / sum; }
              } else v = NB.oneBasis(i, deg, u, U);
              vals.push(v); if (v > 1e-6) any = true;
            }
            if (!any) continue;
            for (s = 0; s <= n; s++) d += (s ? 'L' : 'M') + f(x0 + s / n * (x1 - x0)) + ' ' + f(bot - vals[s] * amp);
            var hot = sc.pyramid ? (i >= st.sel && i <= st.sel + (p - deg)) :
                      (sc.local || sc.circle) ? i === st.sel : rw ? (i >= rw.k - p && i <= rw.k) : true;
            if (hot && (sc.local || sc.pyramid || sc.circle)) h += '<path d="' + d + 'L' + f(x1) + ' ' + f(bot) + 'L' + f(x0) + ' ' + f(bot) + 'Z" fill="' + C.ctrl + '" opacity=".14"/>';
            h += '<path d="' + d + '" fill="none" stroke="' + C.ctrl + '" stroke-width="' + (hot ? 2.2 : 1.2) + '" opacity="' + (hot ? 1 : 0.32) + '" stroke-linejoin="round"/>';
          }
        }
        // u scanner
        if (rw) {
          var xs = bx(st.u);
          h += '<path d="M' + f(xs) + ' ' + f(y0) + 'V' + f(y1) + '" stroke="' + C.text + '" stroke-width="1.2"/>';
          for (i = 0; i <= p; i++) h += '<circle cx="' + f(xs) + '" cy="' + f(y1 - rw.R[i] * (rh - 14)) + '" r="4.5" fill="' + C.ctrl + '" stroke="' + C.bg + '" stroke-width="1.5"/>';
          h += '<text x="' + f(xs) + '" y="' + f(y1 + 30) + '" text-anchor="middle" fill="' + C.text + '" class="kt-t">u = ' + st.u.toFixed(2) + '</text>';
        }
        // knots on the axis (stacked when repeated)
        var seen = {};
        for (i = 0; i < U.length; i++) {
          var key = U[i].toFixed(4), lvl = seen[key] = (seen[key] || 0) + 1, xk = bx(U[i]);
          var inner = i > p && i < m;
          h += '<path d="M' + f(xk) + ' ' + f(y1 + 3 + (lvl - 1) * 7) + 'l5 8h-10z" fill="' + (inner && sc.ticks ? C.ctrl : 'none') + '" stroke="' + C.ctrl + '" stroke-width="1.2" opacity="' + (inner ? 1 : 0.55) + '"/>';
        }
        if (!sc.pyramid) h += '<text x="' + f(x1) + '" y="' + f(y0 + 11) + '" text-anchor="end" fill="' + C.mute + '" class="kt-t">' +
             (st.circle || anyW() ? 'R' : 'N') + '<tspan baseline-shift="sub" font-size="9">i,' + p + '</tspan>(u) · basis functions</text>';
        return h;
      }
      function anyW() { for (var i = 0; i < st.w.length; i++) if (Math.abs(st.w[i] - 1) > 1e-6) return true; return false; }

      /* ---------- interaction ---------- */
      function local(e) { var r = svg.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
      function hit(x, y, touch) {        // touch widens the knot targets; points are already 52 px across
        var best = null, bd = 26, i, q, d;
        if (st.f) return null;
        if (st.multi) {
          bd = touch ? KT.HIT : 13;
          if (st.z) return touch || x < L.ox || x > L.ox + 2 * L.s || y < L.oy || y > L.oy + L.s ? null : { kind: 'pan' };   // a mouse drags the window; a finger scrolls the page
          if (st.m) return null;                 // the morph owns its control points
          st.multi.forEach(function (c, ci) {
            for (i = 0; i < c.P.length; i++) { q = px(c.P[i]); d = Math.hypot(q[0] - x, q[1] - y); if (d < bd) { bd = d; best = { kind: 'text', c: ci, i: i }; } }
          });
          return best;
        }
        for (i = 0; i < st.P.length; i++) { q = px(st.P[i]); d = Math.hypot(q[0] - x, q[1] - y); if (d < bd) { bd = d; best = { kind: 'pt', i: i }; } }
        if (best) return best;
        if (L.bh > 30) {
          if (st.sc.ticks && y > L.by1 - (touch ? 20 : 6) && y < L.by1 + (touch ? 36 : 30)) {
            bd = touch ? KT.HIT : 16;
            for (i = st.p + 1; i < st.P.length; i++) { d = Math.abs(bx(st.U[i]) - x); if (d < bd) { bd = d; best = { kind: 'knot', i: i }; } }
            if (best) return best;
          }
          if (st.sc.blend && y > L.by0 - 10 && x > L.bx0 - 10 && x < L.bx1 + 10) return { kind: 'scan' };
        }
        return null;
      }
      function apply(d, x, y) {
        if (d.kind === 'pan') {
          if (d.px != null) { st.z.cx -= (x - d.px) / (L.s * st.z.zoom); st.z.cy += (y - d.py) / (L.s * st.z.zoom); st.z.user = true; }
          d.px = x; d.py = y;
        }
        else if (d.kind === 'text') {            // a corner is three points on one spot: they move together
          var c = st.multi[d.c], was = d.was || (d.was = c.P[d.i].slice()), now = unpx(x, y);
          d.set = d.set || c.P.map(function (k, j) { return k[0] === was[0] && k[1] === was[1] ? j : -1; }).filter(function (j) { return j >= 0; });
          d.set.forEach(function (j) { c.P[j] = now.slice(); });
        }
        else if (d.kind === 'pt') { st.P[d.i] = unpx(x, y); st.base[d.i] = st.P[d.i].slice(); }
        else if (d.kind === 'knot') {
          var u = KT.clamp((x - L.bx0) / (L.bx1 - L.bx0), 0, 1);
          // interior knots may stack on each other, but never on the clamped end knots (that span would be empty)
          st.U[d.i] = KT.clamp(u, Math.max(st.U[d.i - 1], 0.004), Math.min(st.U[d.i + 1], 0.996));
        } else { st.u = KT.clamp((x - L.bx0) / (L.bx1 - L.bx0), 0, 0.9999); st.userU = true; }
        dirty = true;
      }
      svg.addEventListener('pointerdown', function (e) {
        if (drag) return;                  // one drag at a time: a second finger does not start another
        var q = local(e), d = hit(q[0], q[1], e.pointerType === 'touch');
        if (!d) return;
        drag = d; d.id = e.pointerId; if (d.kind !== 'pan') st.touched = true;
        if (d.kind === 'pt') { st.sel = d.i; slW.set(st.w[d.i]); }
        svg.setPointerCapture(e.pointerId); apply(d, q[0], q[1]); e.preventDefault();
      });
      svg.addEventListener('pointermove', function (e) {
        var q = local(e);
        if (drag) { if (e.pointerId === drag.id) apply(drag, q[0], q[1]); }   // a second finger does not steal the handle
        else svg.style.cursor = hit(q[0], q[1]) ? 'grab' : '';
      });
      function end(e) { if (drag && e.pointerId === drag.id) drag = null; }
      svg.addEventListener('pointerup', end); svg.addEventListener('pointercancel', end);
      // on touch screens, only claim the gesture when it starts on a handle (else let the page scroll)
      KT.touchGuard(svg, { hit: function (x, y) { var q = local({ clientX: x, clientY: y }); return !!hit(q[0], q[1], true); } });

      /* ---------- scenes ---------- */
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.touched = false; st.userU = false; st.t = 0; st.cycleT = 0;
        if (!!sc.circle !== st.circle) load(!!sc.circle);
        else if (sc.circle) load(true);
        else {                               // same control polygon, clean knots/weights/degree
          st.P = st.base.map(function (q) { return q.slice(); });
          st.w = st.P.map(function () { return 1; });
          st.p = 3; st.U = NB.clampedKnots(st.P.length, 3); segDeg.set(3);
        }
        st.m = st.f = st.z = null; st.multi = null;
        if (sc.fit) loadFit(); else if (sc.morph) loadMorph(); else if (sc.multi) loadText(sc.multi);
        if (sc.lens) { st.z = { n: slG.get(), zoom: 1, cx: 1, cy: 0.5, user: false }; slZ.set(0); st.m.from = st.m.to = 1; blend0(1); segM.set(1); }
        st.frozen = false; tgF.set(false); tgF.el.style.display = sc.morph ? '' : 'none';
        slG.el.style.display = slZ.el.style.display = sc.lens ? '' : 'none';
        segM.el.style.display = sc.morph ? '' : 'none'; segDeg.el.style.display = sc.morph || sc.fit ? 'none' : '';
        [segT, slN, slM, segD].forEach(function (c) { c.el.style.display = sc.fit ? '' : 'none'; });
        if (sc.sel != null) st.sel = sc.sel;
        slW.set(st.w[st.sel]);
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        st.ghost = null; dirty = true;
        if (sc.local) { st.bhSnap = true; }
      }

      env.loop(function (t, dt) {
        var sc = st.sc, target = sc.basis ? (sc.pyramid ? 0.56 : 0.32) : sc.fit ? 0.3 : 0;
        if (Math.abs(st.bh - target) > 0.001) { st.bh += (target - st.bh) * (1 - Math.exp(-7 * dt)); dirty = true; }
        else if (st.bh !== target) { st.bh = target; dirty = true; }
        st.t += dt;
        if (st.f && !st.touched) {               // refine both together, hold, start over
          var k = KT.clamp((st.t % 11) / 7.5, 0, 1), n = Math.round(14 * Math.pow(150 / 14, k)), mm2 = Math.round(20 * Math.pow(900 / 20, k));
          if (n !== st.f.n || mm2 !== st.f.m) { st.f.n = n; st.f.m = mm2; slN.set(n); slM.set(Math.log(mm2 / 16) / Math.log(75)); refit(true); }
        }
        if (st.z && !st.z.user) {                // move in on the writing and back out, until someone takes over
          var ph = (st.t % 16) / 16, zk = ph < 0.12 ? 0 : ph < 0.45 ? (ph - 0.12) / 0.33 : ph < 0.67 ? 1 : ph < 1 ? 1 - (ph - 0.67) / 0.33 : 0;
          zk = zk * zk * (3 - 2 * zk); st.z.zoom = Math.pow(3.6, zk); st.z.cx = KT.lerp(1, 0.95, zk); st.z.cy = KT.lerp(0.5, 0.74, zk); slZ.set(Math.log(st.z.zoom) / Math.log(12)); dirty = true;
        }
        if (st.m && !st.frozen) {                // hold a shape, then slide to the next one
          var mm = st.m;
          if (mm.to !== mm.from) {
            mm.k = Math.min(1, mm.k + dt / 2.6); blend(); dirty = true;
            if (mm.k >= 1) { mm.from = mm.to; mm.hold = 2.2; segM.set(mm.from); }
          } else if (!st.touched && (mm.hold -= dt) <= 0) { mm.to = st.z ? 3 - mm.from : (mm.from + 1) % 3; mm.k = 0; }     // the lens goes handwriting <-> missile
        }
        else if (st.multi) {                     // the pen writes the strokes in order; a touch finishes the writing
          var was = st.pen;
          if (st.touched) st.pen = st.total;
          else { st.pen += sc.rate * dt; if (sc.again && st.pen > st.total + sc.rate * 3) st.pen = 0; }
          if (was < st.total || st.pen < was) dirty = true;
        }
        if (!st.touched && !drag) {
          if (sc.wiggle != null) {             // idle motion so the stage is alive before anyone touches it
            var i = sc.wiggle, b = st.base[i];
            st.P[i] = [b[0] + 0.07 * Math.sin(st.t * 1.1), KT.clamp(b[1] + 0.16 * Math.sin(st.t * 1.7), 0, 1)];
            dirty = true;
          }
          if (sc.cycle) {
            st.cycleT += dt;
            if (st.cycleT > 1.9) { st.cycleT = 0; var np = st.p % 3 + 1; st.p = np; st.U = NB.clampedKnots(st.P.length, np); segDeg.set(np); dirty = true; }
          }
          if (sc.knotAuto) {                   // slide one interior knot towards its neighbour and back
            var k = st.p + 2, base = NB.clampedKnots(st.P.length, st.p);
            st.U[k] = KT.lerp(base[k], base[k - 1] + 0.004, 0.5 - 0.5 * Math.cos(st.t * 0.9));
            dirty = true;
          }
        }
        if (sc.blend && !st.userU) { st.u = 0.5 - 0.48 * Math.cos(st.t * 0.45); dirty = true; }
        if (sc.local && !st.ghost && st.bh > 0.3) {  // snapshot the rest curve once the layout settled
          var keep = st.P[st.sel]; st.P[st.sel] = st.base[st.sel]; layout(); st.ghost = curvePath(0, 1, 160); st.P[st.sel] = keep;
        }
        if (sc.local && Math.abs(st.bh - target) > 0.001) st.ghost = null;
        if (dirty) { dirty = false; draw(); }
      });

      if (!document.getElementById('kt-basis-css')) {
        var css = document.createElement('style'); css.id = 'kt-basis-css';
        css.textContent = '.kt-sim-basis .kt-t{font:600 11px/1 var(--f-mono);letter-spacing:.04em}' +
          '.kt-sim-basis.kt-embed .kt-panel{min-width:176px}';
        document.head.appendChild(css);
      }

      load(false); setScene('ducks');
      return {
        setScene: setScene,
        resize: function (w, h) { W = w; H = h; svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h); st.ghost = null; dirty = true; }
      };
    }
  });
})();
