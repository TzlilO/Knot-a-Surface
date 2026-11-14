/* Knot-a-Surface talk — module `basis`: "Pluck a control point".
   A B-spline / NURBS curve you can take apart: control points, basis functions, local support,
   degree, knots, rational weights and the Cox–de Boor recursion. Tool: SVG, redrawn from state.
   Standalone: talk/sims/basis.html. Depends on talk/nurbs.js + talk/kit.js only. */
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

  var SCENES = {
    ducks:     { cap: '<b>a spline</b> · a flexible strip, pinned by weights', wiggle: 4 },
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
      { id: 'ducks', label: 'The draftsman’s spline' }, { id: 'drag', label: 'Control points' },
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
      }

      /* ---------- layout ---------- */
      function layout() {
        var padX = Math.max(22, W * 0.05), topPad = 46;
        var bh = st.bh * H, topH = H - bh;
        var s = Math.min((W - 2 * padX) / 2, (topH - topPad - 26) / 1);
        L.s = s; L.ox = (W - 2 * s) / 2; L.oy = topPad + (topH - topPad - 26 - s) / 2;
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
      function draw() {
        layout();
        var sc = st.sc, p = st.p, m = st.P.length, U = st.U, i, q, h = '', a;
        var showBasis = L.bh > 30;

        // drafting grid behind the curve
        h += '<g stroke="' + C.line + '" stroke-width="1" opacity=".55">';
        for (i = 0; i <= 8; i++) { a = L.ox + i * L.s / 4; h += '<path d="M' + f(a) + ' ' + f(L.oy) + 'V' + f(L.oy + L.s) + '"/>'; }
        for (i = 0; i <= 4; i++) { a = L.oy + i * L.s / 4; h += '<path d="M' + f(L.ox) + ' ' + f(a) + 'H' + f(L.ox + 2 * L.s) + '"/>'; }
        h += '</g>';

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
        if (d.kind === 'pt') { st.P[d.i] = unpx(x, y); st.base[d.i] = st.P[d.i].slice(); }
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
        drag = d; d.id = e.pointerId; st.touched = true;
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
        if (sc.sel != null) st.sel = sc.sel;
        slW.set(st.w[st.sel]);
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        st.ghost = null; dirty = true;
        if (sc.local) { st.bhSnap = true; }
      }

      env.loop(function (t, dt) {
        var sc = st.sc, target = sc.basis ? (sc.pyramid ? 0.56 : 0.32) : 0;
        if (Math.abs(st.bh - target) > 0.001) { st.bh += (target - st.bh) * (1 - Math.exp(-7 * dt)); dirty = true; }
        else if (st.bh !== target) { st.bh = target; dirty = true; }
        st.t += dt;
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
