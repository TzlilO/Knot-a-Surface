/* Knot-a-Surface — simulators for the short Hebrew deck (talk-he.html). Canvas 2D, light theme.
   One motif throughout: a doughnut.
     he-shapes — explicit representations: the same shape as points, triangles or voxels
     he-field  — implicit representation: a function you query (signed distance), and why
                 rendering one means asking it many times (ray marching)
     he-fit    — Gaussians: a real little optimiser fitting soft blobs to a picture
   Registered with the kit (talk/kit.js) for its lifecycle; all on-screen text is DOM, in Hebrew. */
(function () {
  'use strict';
  var KT = window.KnotTalk, TAU = 6.283185307;
  var INK = '#1d1d1f', DIM = '#6e6e73', BLUE = '#0a84ff', PURPLE = '#af52de', ORANGE = '#ff7a3d';
  var DOUGH = [232, 184, 124], GLAZE = [255, 138, 176], SPR = [[92, 200, 250], [255, 214, 10], [255, 255, 255], [102, 212, 164], [191, 120, 255]];
  function he(n) { return Math.round(n).toLocaleString('he-IL'); }
  function h2(x, y) { var s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); }
  function rgb(c, k) { return 'rgb(' + Math.round(Math.min(255, c[0] * k)) + ',' + Math.round(Math.min(255, c[1] * k)) + ',' + Math.round(Math.min(255, c[2] * k)) + ')'; }
  function read(root) { var e = KT.ui.el('div', 'he-read'); root.appendChild(e); return function (big, small) { e.innerHTML = '<b>' + big + '</b>' + small; }; }

  /* ---------- the doughnut ---------- */
  var R = 1, r = 0.44;
  function torus(u, v) {
    var cu = Math.cos(u), su = Math.sin(u), cv = Math.cos(v), sv = Math.sin(v);
    return { p: [(R + r * cv) * cu, r * sv, (R + r * cv) * su], n: [cv * cu, sv, cv * su] };
  }
  function glazed(u, y) { return y > 0.07 + 0.09 * Math.sin(u * 5 + 0.6) + 0.05 * Math.sin(u * 11); }
  function skin(u, v, y) {
    if (!glazed(u, y)) return DOUGH;
    var a = Math.floor(u / TAU * 22), b = Math.floor(v / TAU * 9), k = h2(a, b);
    return k > 0.8 && y > 0.2 ? SPR[Math.floor(h2(b, a) * SPR.length)] : GLAZE;
  }
  function sdf(x, y, z) { var q = Math.hypot(x, z) - R; return Math.hypot(q, y) - r; }
  /* a tiny camera: yaw, pitch, soft perspective. Returns [sx, sy, depth]. */
  function cam(yaw, pitch, cx, cy, S) {
    var cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    function rot(p) { var x = p[0] * cyw + p[2] * syw, z = -p[0] * syw + p[2] * cyw; return [x, p[1] * cp - z * sp, p[1] * sp + z * cp]; }
    return { rot: rot, at: function (p) { var q = rot(p), k = 4.2 / (4.2 - q[2]); return [cx + q[0] * S * k, cy - q[1] * S * k, q[2]]; } };
  }
  var LIGHT = (function () { var l = [-0.42, 0.74, 0.52], n = Math.hypot(l[0], l[1], l[2]); return [l[0] / n, l[1] / n, l[2] / n]; })();
  function shade(n) { return 0.8 + 0.28 * Math.max(0, n[0] * LIGHT[0] + n[1] * LIGHT[1] + n[2] * LIGHT[2]); }
  /* turn the object, on the kit's shared gestures: one finger / mouse drag turns it (embedded, a
     finger turns it sideways only and a vertical swipe scrolls the deck); two fingers turn it in both
     axes and a pinch scales it (st.zoom) */
  function turn(cv, st, embed) {
    KT.gestures(cv, { embed: embed,
      start: function (e) { if (e.pointerType !== 'touch') st.touched = true; },
      drag: function (dx, dy) { st.yaw += dx * 0.008; st.pitch = KT.clamp(st.pitch + dy * 0.006, 0.15, 1.4); st.touched = true; },
      pinch: function (k) { st.zoom = KT.clamp(st.zoom * k, 0.6, 1.8); st.touched = true; } });
    cv.style.cursor = 'grab';
  }

  /* ================= he-shapes: points · triangles · voxels ================= */
  KT.register('he-shapes', {
    title: 'ייצוג מפורש', blurb: '', tool: 'Canvas 2D',
    scenes: [{ id: 'points', label: 'נקודות' }, { id: 'mesh', label: 'משולשים' }, { id: 'voxels', label: 'קוביות' }],
    create: function (env) {
      var ui = env.ui, root = env.root, C = ui.canvas(root), g = C.ctx, W = 1, H = 1;
      var st = { mode: 'points', detail: 0.4, yaw: 0.6, pitch: 0.72, zoom: 1, t: 0, touched: false, sweep: false, hero: false, cyc: 0 };
      var out = read(root);
      var panel = ui.panel(root, { pos: 'bottom' });
      var seg = ui.seg(panel, { value: 'points', options: [{ v: 'points', label: 'נקודות' }, { v: 'mesh', label: 'משולשים' }, { v: 'voxels', label: 'קוביות' }],
        onChange: function (v) { st.mode = v; st.sweep = false; } });
      var sl = ui.slider(panel, { label: 'פירוט', min: 0, max: 100, value: 40, fmt: function () { return ''; }, onInput: function (v) { st.detail = v / 100; st.sweep = false; } });
      turn(C.cv, st, env.embed);

      // a fixed, well-spread pool of surface samples; "more points" just takes more of it
      var POOL = 3600, pts = [];
      (function () {
        for (var i = 0; pts.length < POOL; i++) {         // R2 low-discrepancy sequence: any prefix is even
          var u = TAU * ((0.5 + i * 0.7548776662) % 1), v = TAU * ((0.5 + i * 0.5698402910) % 1);
          if (h2(i, 9.1) > (R + r * Math.cos(v)) / (R + r)) continue;                 // equal area on the torus
          var t = torus(u, v);
          pts.push({ p: t.p, n: t.n, c: skin(u, v, t.p[1]) });
        }
      })();

      function draw() {
        var S = Math.min(W, H) * (st.hero ? 0.3 : 0.245) * st.zoom, cm = cam(st.yaw, st.pitch, W / 2, H * 0.44, S), d = st.detail, i, j, q;
        g.clearRect(0, 0, W, H);
        // a soft contact shadow keeps the object on the "table"
        var sh = g.createRadialGradient(W / 2, H * 0.44 + S * 0.95, 0, W / 2, H * 0.44 + S * 0.95, S * 1.7);
        sh.addColorStop(0, 'rgba(120,80,40,.16)'); sh.addColorStop(1, 'rgba(120,80,40,0)');
        g.save(); g.translate(0, H * 0.44 + S * 0.95); g.scale(1, 0.26); g.translate(0, -(H * 0.44 + S * 0.95)); g.fillStyle = sh; g.fillRect(0, 0, W, H * 3); g.restore();

        if (st.mode === 'points') {
          var N = Math.round(KT.lerp(90, POOL, d * d)), list = [], rad = Math.sqrt(17.4 / N / Math.PI) * 0.5 * S;
          for (i = 0; i < N; i++) { q = cm.at(pts[i].p); list.push([q[2], q[0], q[1], pts[i]]); }
          list.sort(function (a, b) { return a[0] - b[0]; });
          for (i = 0; i < N; i++) { q = list[i]; g.fillStyle = rgb(q[3].c, shade(cm.rot(q[3].n))); g.beginPath(); g.arc(q[1], q[2], Math.max(1.3, rad * (1 + q[0] * 0.1)), 0, TAU); g.fill(); }
          out(he(N), 'נקודות');
        } else if (st.mode === 'mesh') {
          var nu = Math.round(KT.lerp(5, 44, d)), nv = Math.max(4, Math.round(nu * 0.5)), faces = [];
          for (i = 0; i < nu; i++) for (j = 0; j < nv; j++) {
            var u0 = i / nu * TAU, u1 = (i + 1) / nu * TAU, v0 = j / nv * TAU, v1 = (j + 1) / nv * TAU, m = torus((u0 + u1) / 2, (v0 + v1) / 2);
            var nn = cm.rot(m.n);                       // no culling: far faces are simply painted first
            var a = cm.at(torus(u0, v0).p), b = cm.at(torus(u1, v0).p), c = cm.at(torus(u1, v1).p), e = cm.at(torus(u0, v1).p);
            faces.push([cm.rot(m.p)[2], a, b, c, e, rgb(skin((u0 + u1) / 2, (v0 + v1) / 2, m.p[1]), shade(nn))]);
          }
          faces.sort(function (a, b) { return a[0] - b[0]; });
          g.lineWidth = 1; g.lineJoin = 'round'; g.strokeStyle = 'rgba(255,255,255,.75)';
          faces.forEach(function (f) {
            g.fillStyle = f[5]; g.beginPath(); g.moveTo(f[1][0], f[1][1]); g.lineTo(f[2][0], f[2][1]); g.lineTo(f[3][0], f[3][1]); g.lineTo(f[4][0], f[4][1]); g.closePath(); g.fill(); g.stroke();
            g.beginPath(); g.moveTo(f[1][0], f[1][1]); g.lineTo(f[3][0], f[3][1]); g.stroke();
          });
          out(he(nu * nv * 2), 'משולשים');
        } else {
          var n = Math.round(KT.lerp(6, 24, d)), h = 3 / n, ny = Math.max(2, Math.round(1.0 / h)), cubes = [], cells = n * n * ny;
          var F = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], fr = F.map(cm.rot);
          function full(ix, iy, iz) { return sdf(-1.5 + (ix + 0.5) * h, (iy + 0.5 - ny / 2) * h, -1.5 + (iz + 0.5) * h) < 0.2 * h; }
          for (i = 0; i < n; i++) for (j = 0; j < ny; j++) for (var k = 0; k < n; k++) {
            if (!full(i, j, k)) continue;
            var c0 = [-1.5 + (i + 0.5) * h, (j + 0.5 - ny / 2) * h, -1.5 + (k + 0.5) * h];
            cubes.push([cm.rot(c0)[2], c0, i, j, k]);
          }
          cubes.sort(function (a, b) { return a[0] - b[0]; });
          g.lineWidth = 0.8; g.strokeStyle = 'rgba(255,255,255,.6)';
          cubes.forEach(function (cb) {
            var c0 = cb[1], uu = Math.atan2(c0[2], c0[0]), col = glazed(uu < 0 ? uu + TAU : uu, c0[1] + (cb[3] === ny - 1 ? 0.2 : 0)) ? GLAZE : DOUGH;
            for (var f = 0; f < 6; f++) {
              if (fr[f][2] <= 0.02) continue;
              var nb = F[f]; if (full(cb[2] + nb[0], cb[3] + nb[1], cb[4] + nb[2])) continue;
              var ax = nb[0] ? [0, 1, 0] : [1, 0, 0], bx = nb[2] ? [0, 1, 0] : (nb[0] ? [0, 0, 1] : [0, 0, 1]), pp = [];
              [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (s) {
                pp.push(cm.at([c0[0] + (nb[0] + ax[0] * s[0] + bx[0] * s[1]) * h / 2, c0[1] + (nb[1] + ax[1] * s[0] + bx[1] * s[1]) * h / 2, c0[2] + (nb[2] + ax[2] * s[0] + bx[2] * s[1]) * h / 2]));
              });
              g.fillStyle = rgb(col, shade(fr[f])); g.beginPath(); g.moveTo(pp[0][0], pp[0][1]); g.lineTo(pp[1][0], pp[1][1]); g.lineTo(pp[2][0], pp[2][1]); g.lineTo(pp[3][0], pp[3][1]); g.closePath(); g.fill(); g.stroke();
            }
          });
          out(he(cells), 'תאים ברשת');
        }
      }

      function setScene(id) {
        st.hero = id === 'hero'; st.mode = st.hero ? 'points' : id; st.touched = false; st.t = 0; st.cyc = 0; st.zoom = 1;
        st.detail = id === 'points' ? 0.5 : 0.3; st.sweep = id === 'mesh' || id === 'voxels';
        seg.set(st.mode); sl.set(st.detail * 100);
        panel.style.display = root.querySelector('.he-read').style.display = st.hero ? 'none' : '';
      }
      env.loop(function (t, dt) {
        st.t += dt;
        if (!st.touched) st.yaw += dt * 0.22;
        if (st.sweep) { st.detail = 0.5 - 0.42 * Math.cos(st.t * 0.5); sl.set(st.detail * 100); }
        if (st.hero) { st.cyc += dt; if (st.cyc > 3.4) { st.cyc = 0; st.mode = st.mode === 'points' ? 'mesh' : st.mode === 'mesh' ? 'voxels' : 'points'; } st.detail = 0.55; }
        draw();
      });
      setScene(env.opts.scene || 'points');
      return { setScene: setScene, state: st, resize: function (w, h) { W = w; H = h; C.fit(); } };
    }
  });

  /* ================= he-field: a function you ask ================= */
  KT.register('he-field', {
    title: 'ייצוג סתום', blurb: '', tool: 'Canvas 2D',
    scenes: [{ id: 'probe', label: 'שאלה אחת' }, { id: 'march', label: 'הרבה שאלות' }],
    create: function (env) {
      var ui = env.ui, root = env.root, C = ui.canvas(root), g = C.ctx, W = 1, H = 1;
      var st = { scene: 'probe', p: [1.5, 0.7], t: 0, touched: false, n: 12, a: 0 };
      var out = read(root), bub = ui.el('div', 'he-bubble'); root.appendChild(bub);
      var panel = ui.panel(root, { pos: 'bottom' });
      var sl = ui.slider(panel, { label: 'דגימות לכל קרן', min: 2, max: 64, value: 12, fmt: function (v) { return v; }, onInput: function (v) { st.n = v; } });
      function f(x, y) { return Math.abs(Math.hypot(x, y) - R) - r; }       // the doughnut seen from above
      var fld = document.createElement('canvas'), S = 1, cx = 0, cy = 0;
      function X(x) { return cx + x * S; } function Y(y) { return cy - y * S; }
      function bake() {                                // the field as soft bands of distance
        var w = 220, h = Math.round(220 * H / W), im, d, i, j, o;
        fld.width = w; fld.height = h; var fg = fld.getContext('2d'); im = fg.createImageData(w, h); d = im.data;
        for (j = 0; j < h; j++) for (i = 0; i < w; i++) {
          var x = ((i + 0.5) / w * W - cx) / S, y = -((j + 0.5) / h * H - cy) / S, v = f(x, y), band = Math.abs(((v / 0.22) % 1 + 1) % 1 - 0.5) < 0.06 ? 0.75 : 1;
          o = (j * w + i) * 4;
          if (v < 0) { d[o] = 255; d[o + 1] = 150 + 60 * (1 + v / r); d[o + 2] = 185 + 40 * (1 + v / r); d[o + 3] = 210; }
          else { d[o] = 175; d[o + 1] = 82; d[o + 2] = 222; d[o + 3] = Math.max(0, (34 - v * 16)) * (band < 1 ? 2.2 : 1); }
        }
        fg.putImageData(im, 0, 0);
      }
      /* the probe: a mouse drags it from anywhere. A finger grabs it only on the probe itself (the rest
         of the stage scrolls the deck); a tap elsewhere puts it there. */
      var held = null, tap = null;
      function drag(e) { var b = C.cv.getBoundingClientRect(); st.p = [KT.clamp((e.clientX - held[0] - b.left - cx) / S, -2.3, 2.3), KT.clamp(-(e.clientY - held[1] - b.top - cy) / S, -1.6, 1.6)]; st.touched = true; }
      function onProbe(x, y) {
        if (st.scene !== 'probe') return null;
        var b = C.cv.getBoundingClientRect(), dx = x - b.left - X(st.p[0]), dy = y - b.top - Y(st.p[1]);
        return Math.hypot(dx, dy) < KT.HIT ? [dx, dy] : null;
      }
      KT.touchGuard(C.cv, { hit: function (x, y) { return !!onProbe(x, y); } });
      C.cv.addEventListener('pointerdown', function (e) {
        if (st.scene !== 'probe') return;
        var off = onProbe(e.clientX, e.clientY);
        if (e.pointerType === 'touch' && !off) { tap = [e.clientX, e.clientY, e.pointerId]; return; }
        held = off || [0, 0]; held.id = e.pointerId; C.cv.setPointerCapture(e.pointerId); drag(e);
      });
      C.cv.addEventListener('pointermove', function (e) { if (held && e.pointerId === held.id) drag(e); });
      C.cv.addEventListener('pointerup', function (e) {
        if (tap && tap[2] === e.pointerId && Math.hypot(e.clientX - tap[0], e.clientY - tap[1]) < 10) { held = [0, 0]; drag(e); }
        held = null; tap = null;
      });
      C.cv.addEventListener('pointercancel', function () { held = null; tap = null; });

      function draw() {
        g.clearRect(0, 0, W, H);
        g.imageSmoothingEnabled = true; g.drawImage(fld, 0, 0, W, H);
        g.strokeStyle = PURPLE; g.lineWidth = 3;                                  // f = 0: the surface itself
        g.beginPath(); g.arc(cx, cy, (R + r) * S, 0, TAU); g.stroke(); g.beginPath(); g.arc(cx, cy, (R - r) * S, 0, TAU); g.stroke();
        if (st.scene === 'probe') {
          var p = st.p, v = f(p[0], p[1]), px = X(p[0]), py = Y(p[1]);
          g.setLineDash([5, 6]); g.lineWidth = 1.6; g.strokeStyle = INK; g.beginPath(); g.arc(px, py, Math.max(2, Math.abs(v) * S), 0, TAU); g.stroke(); g.setLineDash([]);
          g.fillStyle = '#fff'; g.beginPath(); g.arc(px, py, 9, 0, TAU); g.fill(); g.fillStyle = v < 0 ? '#ff5c8a' : PURPLE; g.beginPath(); g.arc(px, py, 6, 0, TAU); g.fill();
          bub.style.display = ''; bub.style.transform = 'translate(' + (px + 16) + 'px,' + (py - 46) + 'px)';
          bub.innerHTML = '<span class="ltr">f = ' + (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2) + '</span><i style="background:' + (v < 0 ? '#ff5c8a' : PURPLE) + '"></i>' + (v < 0 ? 'בפנים' : 'בחוץ');
          out('שאלה אחת', 'תשובה אחת');
        } else {
          bub.style.display = 'none';
          var ex = 2.25, n = st.n, K = 44, strip = [], a, k, i2;
          function hit(ang) {                       // march one ray: is any sample inside?
            var dx = -Math.cos(ang), dy = Math.sin(ang), res = false;
            for (var s = 0; s < n; s++) { var t = 0.25 + (s + 0.5) / n * 4.0; if (f(ex + dx * t, dy * t) < 0) res = true; }
            return res;
          }
          var FOV = 1.76;
          for (k = 0; k < K; k++) strip.push(hit((k + 0.5) / K * FOV - FOV / 2));
          a = (FOV / 2 - 0.03) * Math.sin(st.t * 0.5);
          var dx = -Math.cos(a), dy = Math.sin(a);
          g.strokeStyle = 'rgba(29,29,31,.35)'; g.lineWidth = 1.4; g.beginPath(); g.moveTo(X(ex), Y(0)); g.lineTo(X(ex + dx * 4.25), Y(dy * 4.25)); g.stroke();
          for (i2 = 0; i2 < n; i2++) {
            var tt = 0.25 + (i2 + 0.5) / n * 4.0, qx = ex + dx * tt, qy = dy * tt, ins = f(qx, qy) < 0;
            g.beginPath(); g.arc(X(qx), Y(qy), ins ? 6 : 4, 0, TAU); g.fillStyle = ins ? '#ff5c8a' : '#fff'; g.fill(); g.lineWidth = 1.4; g.strokeStyle = ins ? '#fff' : 'rgba(29,29,31,.4)'; g.stroke();
          }
          g.fillStyle = INK; g.beginPath(); g.arc(X(ex), Y(0), 9, 0, TAU); g.fill(); g.fillStyle = '#fff'; g.beginPath(); g.arc(X(ex), Y(0), 3.5, 0, TAU); g.fill();
          // the 1-D "photo" this camera gets: one pixel per ray
          var sx = X(-2.42), ph = Math.min(H * 0.62, 3.0 * S), top = cy - ph / 2, cell = ph / K, now = KT.clamp(Math.floor((a + FOV / 2) / FOV * K), 0, K - 1);
          g.fillStyle = 'rgba(255,255,255,.85)'; rr(g, sx - 5, top - 5, 28, ph + 10, 9); g.fill();
          for (k = 0; k < K; k++) { g.fillStyle = strip[k] ? '#ff8ab0' : '#f1ebe2'; g.fillRect(sx, top + (K - 1 - k) * cell, 18, Math.ceil(cell)); }
          g.strokeStyle = INK; g.lineWidth = 2; g.strokeRect(sx - 2, top + (K - 1 - now) * cell - 1, 22, cell + 2);
          var q = 1920 * 1080 * n;
          out(q >= 1e6 ? he(q / 1e6) + ' מיליון' : he(q), 'שאלות לתמונה אחת');
        }
      }
      function rr(c, x, y, w, h, r0) { c.beginPath(); c.moveTo(x + r0, y); c.arcTo(x + w, y, x + w, y + h, r0); c.arcTo(x + w, y + h, x, y + h, r0); c.arcTo(x, y + h, x, y, r0); c.arcTo(x, y, x + w, y, r0); c.closePath(); }

      function setScene(id) { st.scene = id; st.touched = false; st.t = 0; panel.style.display = id === 'march' ? '' : 'none'; }
      env.loop(function (t, dt) {
        st.t += dt;
        if (st.scene === 'probe' && !st.touched) st.p = [1.35 * Math.cos(st.t * 0.45) + 0.5 * Math.cos(st.t * 1.1), 0.95 * Math.sin(st.t * 0.6)];
        draw();
      });
      if (!document.getElementById('he-bubble-css')) {
        var css = document.createElement('style'); css.id = 'he-bubble-css';
        css.textContent = '.he-bubble{position:absolute;z-index:4;left:0;top:0;display:flex;align-items:center;gap:.5rem;padding:.5rem .8rem;border-radius:999px;' +
          'background:rgba(255,255,255,.82);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:inset 0 1px 0 #fff,0 8px 22px -10px rgba(60,30,90,.4);' +
          'font-size:.92rem;font-weight:700;color:#1d1d1f;white-space:nowrap;pointer-events:none;direction:rtl;will-change:transform}' +
          '.he-bubble i{width:.5rem;height:.5rem;border-radius:50%}.he-bubble .ltr{font-variant-numeric:tabular-nums;font-weight:600;color:#6e6e73}';
        document.head.appendChild(css);
      }
      setScene(env.opts.scene || 'probe');
      return { setScene: setScene, state: st, resize: function (w, h) { W = w; H = h; C.fit(); S = Math.min(W / 5.6, H / 4.2); cx = W * 0.52; cy = H * 0.47; bake(); } };
    }
  });

  /* ================= he-fit: soft blobs learn a picture ================= */
  KT.register('he-fit', {
    title: 'גאוסיאנים', blurb: '', tool: 'Canvas 2D · אופטימיזציה חיה',
    scenes: [{ id: 'few', label: 'מעט כתמים' }, { id: 'many', label: 'הרבה כתמים' }],
    create: function (env) {
      var ui = env.ui, root = env.root, C = ui.canvas(root), g = C.ctx, W = 1, H = 1, TS = 76, NP = TS * TS;
      var st = { n: 14, show: true, step: 0, acc: 0 };
      var out = read(root);
      var panel = ui.panel(root, { pos: 'bottom' });
      panel.style.setProperty('--tint', ORANGE);
      var sl = ui.slider(panel, { label: 'כמה גאוסיאנים', min: 6, max: 260, value: 14, fmt: function (v) { return v; }, onInput: function (v) { st.n = v; init(); } });
      var tg = ui.toggle(panel, { label: 'הצג כתמים', value: true, onChange: function (v) { st.show = v; } });
      ui.button(panel, { label: 'מהתחלה', onClick: function () { init(); } });

      // the picture to learn: a doughnut on a plate, drawn large and shrunk
      var tgt = document.createElement('canvas'), T = new Float32Array(NP * 3);
      (function () {
        var big = document.createElement('canvas'), B = TS * 6, c = big.getContext('2d'), i;
        big.width = big.height = B; c.fillStyle = '#f7eddc'; c.fillRect(0, 0, B, B);
        c.fillStyle = 'rgba(150,110,70,.18)'; c.beginPath(); c.arc(B / 2, B / 2 + B * 0.02, B * 0.455, 0, TAU); c.fill();
        c.fillStyle = '#fffdf8'; c.beginPath(); c.arc(B / 2, B / 2, B * 0.44, 0, TAU); c.fill();
        c.strokeStyle = '#efe4d2'; c.lineWidth = B * 0.012; c.beginPath(); c.arc(B / 2, B / 2, B * 0.34, 0, TAU); c.stroke();
        c.fillStyle = '#d9a066'; c.beginPath(); c.arc(B / 2, B / 2, B * 0.30, 0, TAU); c.fill();
        c.fillStyle = '#ff8ab0'; c.beginPath();
        for (i = 0; i <= 90; i++) { var a = i / 90 * TAU, rr0 = B * (0.262 + 0.014 * Math.sin(a * 7) + 0.008 * Math.sin(a * 13 + 1)); c.lineTo(B / 2 + rr0 * Math.cos(a), B / 2 + rr0 * Math.sin(a)); }
        c.fill();
        c.lineCap = 'round'; c.lineWidth = B * 0.018;
        var cols = ['#5cc8fa', '#ffd60a', '#ffffff', '#66d4a4', '#bf78ff'];
        for (i = 0; i < 22; i++) { var an = h2(i, 1) * TAU, rd = B * (0.125 + 0.105 * h2(i, 2)), d = h2(i, 3) * TAU; c.strokeStyle = cols[i % 5]; c.beginPath(); c.moveTo(B / 2 + rd * Math.cos(an), B / 2 + rd * Math.sin(an)); c.lineTo(B / 2 + rd * Math.cos(an) + B * 0.03 * Math.cos(d), B / 2 + rd * Math.sin(an) + B * 0.03 * Math.sin(d)); c.stroke(); }
        c.fillStyle = '#fffdf8'; c.beginPath(); c.arc(B / 2, B / 2, B * 0.095, 0, TAU); c.fill();
        c.strokeStyle = 'rgba(255,255,255,.55)'; c.lineWidth = B * 0.02; c.beginPath(); c.arc(B / 2, B / 2, B * 0.215, 3.6, 4.6); c.stroke();
        tgt.width = tgt.height = TS; var tc = tgt.getContext('2d'); tc.imageSmoothingQuality = 'high'; tc.drawImage(big, 0, 0, TS, TS);
        var d0 = tc.getImageData(0, 0, TS, TS).data; for (i = 0; i < NP; i++) { T[i * 3] = d0[i * 4] / 255; T[i * 3 + 1] = d0[i * 4 + 1] / 255; T[i * 3 + 2] = d0[i * 4 + 2] / 255; }
      })();

      /* the model: n Gaussians x 8 numbers (x, y, log sx, log sy, angle, r, g, b); image = normalised blend */
      var NPAR = 8, P, M, V, Gr, Wt = new Float32Array(NP), Sm = new Float32Array(NP * 3), Im = new Float32Array(NP * 3);
      var LR = [0.22, 0.22, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], EPS = 0.02, BG = [0.97, 0.93, 0.86];
      var view = document.createElement('canvas'); view.width = view.height = TS; var vc = view.getContext('2d'), vim = vc.createImageData(TS, TS);
      function init() {
        var n = st.n, i, rng = KT.rng(7 + n);
        P = new Float32Array(n * NPAR); M = new Float32Array(n * NPAR); V = new Float32Array(n * NPAR); Gr = new Float32Array(n * NPAR);
        for (i = 0; i < n; i++) {
          var o = i * NPAR, s = Math.log(TS / Math.sqrt(n) * 0.55);
          P[o] = rng() * TS; P[o + 1] = rng() * TS; P[o + 2] = s; P[o + 3] = s; P[o + 4] = rng() * 3.14;
          P[o + 5] = 0.75 + 0.2 * rng(); P[o + 6] = 0.7 + 0.2 * rng(); P[o + 7] = 0.62 + 0.2 * rng();
        }
        st.step = 0;
      }
      function each(i, fn) {                           // visit the pixels one Gaussian can reach
        var o = i * NPAR, x = P[o], y = P[o + 1], sx = Math.exp(P[o + 2]), sy = Math.exp(P[o + 3]), c = Math.cos(P[o + 4]), s = Math.sin(P[o + 4]);
        var rad = 3 * Math.max(sx, sy), x0 = Math.max(0, Math.floor(x - rad)), x1 = Math.min(TS - 1, Math.ceil(x + rad)), y0 = Math.max(0, Math.floor(y - rad)), y1 = Math.min(TS - 1, Math.ceil(y + rad));
        for (var py = y0; py <= y1; py++) for (var px = x0; px <= x1; px++) {
          var dx = px + 0.5 - x, dy = py + 0.5 - y, a = (c * dx + s * dy), b = (-s * dx + c * dy), u = a / sx, v = b / sy, q = u * u + v * v;
          if (q < 9) fn(py * TS + px, Math.exp(-0.5 * q), u, v, a, b, sx, sy, c, s);
        }
      }
      function iterate() {
        var n = st.n, i, p, o, k;
        for (p = 0; p < NP; p++) { Wt[p] = EPS; Sm[p * 3] = EPS * BG[0]; Sm[p * 3 + 1] = EPS * BG[1]; Sm[p * 3 + 2] = EPS * BG[2]; }
        for (i = 0; i < n; i++) { o = i * NPAR; var cr = P[o + 5], cg = P[o + 6], cb = P[o + 7];
          each(i, function (px, gv) { Wt[px] += gv; Sm[px * 3] += gv * cr; Sm[px * 3 + 1] += gv * cg; Sm[px * 3 + 2] += gv * cb; }); }
        for (p = 0; p < NP; p++) for (k = 0; k < 3; k++) Im[p * 3 + k] = Sm[p * 3 + k] / Wt[p];
        Gr.fill(0);
        for (i = 0; i < n; i++) { o = i * NPAR;
          (function (o) { var cr = P[o + 5], cg = P[o + 6], cb = P[o + 7];
            each(i, function (px, gv, u, v, a, b, sx, sy, c, s) {
              var w = Wt[px], er = Im[px * 3] - T[px * 3], eg = Im[px * 3 + 1] - T[px * 3 + 1], eb = Im[px * 3 + 2] - T[px * 3 + 2];
              Gr[o + 5] += er * gv / w; Gr[o + 6] += eg * gv / w; Gr[o + 7] += eb * gv / w;
              var dg = (er * (cr - Im[px * 3]) + eg * (cg - Im[px * 3 + 1]) + eb * (cb - Im[px * 3 + 2])) / w, dq = -0.5 * gv * dg;   // dL/dq
              Gr[o] += dq * -2 * (u * c / sx - v * s / sy); Gr[o + 1] += dq * -2 * (u * s / sx + v * c / sy);
              Gr[o + 2] += dq * -2 * u * u; Gr[o + 3] += dq * -2 * v * v; Gr[o + 4] += dq * 2 * (u * b / sx - v * a / sy);
            }); })(o);
        }
        st.step++;
        var b1 = 0.9, b2 = 0.999, c1 = 1 - Math.pow(b1, st.step), c2 = 1 - Math.pow(b2, st.step), hi = Math.log(TS * 0.45);
        for (i = 0; i < n * NPAR; i++) {                 // Adam
          M[i] = b1 * M[i] + (1 - b1) * Gr[i]; V[i] = b2 * V[i] + (1 - b2) * Gr[i] * Gr[i];
          P[i] -= LR[i % NPAR] * (M[i] / c1) / (Math.sqrt(V[i] / c2) + 1e-8);
        }
        for (i = 0; i < n; i++) { o = i * NPAR; P[o] = KT.clamp(P[o], -2, TS + 2); P[o + 1] = KT.clamp(P[o + 1], -2, TS + 2);
          P[o + 2] = KT.clamp(P[o + 2], -0.4, hi); P[o + 3] = KT.clamp(P[o + 3], -0.4, hi); for (k = 5; k < 8; k++) P[o + k] = KT.clamp(P[o + k], 0, 1); }
      }
      function rr(c, x, y, w, h, r0) { c.beginPath(); c.moveTo(x + r0, y); c.arcTo(x + w, y, x + w, y + h, r0); c.arcTo(x + w, y + h, x, y + h, r0); c.arcTo(x, y + h, x, y, r0); c.arcTo(x, y, x + w, y, r0); c.closePath(); }
      function draw() {
        var d = vim.data, p, i;
        for (p = 0; p < NP; p++) { d[p * 4] = Im[p * 3] * 255; d[p * 4 + 1] = Im[p * 3 + 1] * 255; d[p * 4 + 2] = Im[p * 3 + 2] * 255; d[p * 4 + 3] = 255; }
        vc.putImageData(vim, 0, 0);
        g.clearRect(0, 0, W, H);
        var side = Math.min(W * 0.74, H * 0.66), x0 = (W - side) / 2, y0 = H * 0.46 - side / 2, k = side / TS;
        g.save(); g.shadowColor = 'rgba(88,56,24,.28)'; g.shadowBlur = 40; g.shadowOffsetY = 18; g.fillStyle = '#fff'; rr(g, x0, y0, side, side, side * 0.11); g.fill(); g.restore();
        g.save(); rr(g, x0, y0, side, side, side * 0.11); g.clip(); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(view, x0, y0, side, side);
        if (st.show) {
          g.lineWidth = 1.3;
          for (i = 0; i < st.n; i++) { var o = i * NPAR;
            g.strokeStyle = 'rgba(29,29,31,.55)'; g.beginPath(); g.ellipse(x0 + P[o] * k, y0 + P[o + 1] * k, Math.exp(P[o + 2]) * k * 1.1, Math.exp(P[o + 3]) * k * 1.1, P[o + 4], 0, TAU); g.stroke();
            g.strokeStyle = 'rgba(255,255,255,.7)'; g.beginPath(); g.ellipse(x0 + P[o] * k, y0 + P[o + 1] * k, Math.exp(P[o + 2]) * k * 1.1 + 1.3, Math.exp(P[o + 3]) * k * 1.1 + 1.3, P[o + 4], 0, TAU); g.stroke(); }
        }
        g.restore();
        var th = Math.max(54, side * 0.2), tx = x0 - th * 0.28, ty = y0 - th * 0.3;      // the target, as a small photo
        g.save(); g.shadowColor = 'rgba(88,56,24,.3)'; g.shadowBlur = 18; g.shadowOffsetY = 8; g.fillStyle = '#fff'; rr(g, tx - 4, ty - 4, th + 8, th + 8, th * 0.2); g.fill(); g.restore();
        g.save(); rr(g, tx, ty, th, th, th * 0.17); g.clip(); g.drawImage(tgt, tx, ty, th, th); g.restore();
        out(he(st.n), 'גאוסיאנים · ' + he(st.n * NPAR) + ' מספרים');
      }
      function setScene(id) { st.n = id === 'many' ? 180 : 14; st.show = id !== 'many'; sl.set(st.n); tg.set(st.show); init(); }
      env.loop(function (t, dt) {
        st.acc = Math.min(st.acc + dt * 70, 6);          // ~70 optimiser steps a second, whatever the frame rate
        var t0 = performance.now();
        while (st.acc >= 1 && performance.now() - t0 < 10) { iterate(); st.acc -= 1; }
        if (performance.now() - t0 >= 10) st.acc = 0;
        draw();
      });
      setScene(env.opts.scene || 'few'); iterate();
      return { setScene: setScene, resize: function (w, h) { W = w; H = h; C.fit(); } };
    }
  });
})();
