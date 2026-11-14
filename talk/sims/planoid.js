/* Knot-a-Surface talk — module `planoid`: "Calculus makes the splat".
   One B-spline patch. Pick a point (u,v) and watch a Gaussian fall out of the surface's own
   calculus: position = S(u,v), orientation = tangent frame, scale = |S_u| du x |S_v| dv.
   Then fill the chart and turn the resolution knob — the learnable parameters never change.
   Tool: WebGL (three.js) + a small 2D canvas for the (u,v) chart.
   Standalone: talk/sims/planoid.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var NU = 6, NV = 6, SPAN = 3.2, PER_CP = 16;      // 3 position + 12 SH1 colour + 1 opacity
  var NMAX = 160;

  var SCENES = {
    net:      { show: 'net wire', cap: '<b>tensor product</b> · a net of control points sweeps out a surface',
                hint: 'drag to orbit · drag an amber point to reshape' },
    probe:    { show: 'net wire probe uv', cap: '<b>position</b> · μ = S(u, v)', hint: 'drag the cross in the (u,v) chart' },
    tangents: { show: 'wire probe tan uv', cap: '<b>tangents</b> · S<sub>u</sub> and S<sub>v</sub>, in closed form' },
    normal:   { show: 'wire probe tan nor uv', cap: '<b>rotation</b> · R = [ Ŝ<sub>u</sub> · Ŝ<sub>v</sub><sup>⊥</sup> · n ]' },
    scale:    { show: 'wire probe tan nor ell uv', n: 9, cap: '<b>scale</b> · s = ( ‖S<sub>u</sub>‖Δu , ‖S<sub>v</sub>‖Δv , ε ) — a flat Gaussian' },
    derive:   { show: 'net wire probe tan nor ell', n: 9, wig: 1, fixed: [0.56, 0.44],
                cap: '<b>nothing is learned per Gaussian</b> · move the surface, the splat re-derives' },
    fill:     { show: 'wire field uv', n: 14, rim: 1, cap: '<b>fill the chart</b> · one Gaussian per (u,v) sample, sized to its cell' },
    knob:     { show: 'field', sweep: 1, rim: 0.7, cap: '<b>the resolution knob</b> · re-evaluate, don’t retrain' },
    features: { show: 'net field feat', n: 72, rim: 0.4, cap: '<b>control features</b> · colour and opacity ride the same basis' },
    hero:     { show: 'wire field feat', n: 64, rim: 0.5, hero: 1, cap: '' }
  };

  KT.register('planoid', {
    title: 'Calculus makes the splat',
    blurb: 'A Gaussian derived, not learned: position, orientation and scale read straight off a spline ' +
           'surface’s partial derivatives. Fill the (u,v) chart and change the sample count — the parameters stay put.',
    tool: 'WebGL · three.js',
    scenes: [
      { id: 'net', label: 'Control net' }, { id: 'probe', label: 'Position' },
      { id: 'tangents', label: 'Tangents' }, { id: 'normal', label: 'Rotation' },
      { id: 'scale', label: 'Scale' }, { id: 'derive', label: 'Derived, not learned' },
      { id: 'fill', label: 'Fill the chart' }, { id: 'knob', label: 'Resolution knob' },
      { id: 'features', label: 'Control features' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(34, 1, 0.1, 60);
      var R = G.renderer(root, cam), scene = new T.Scene();
      var W = 1, H = 1, fitR = 1;

      /* ---------- the patch ---------- */
      var patch = NB.makePatch({ nu: NU, nv: NV, p: 3 }), base = new Float32Array(NU * NV * 3);
      (function () {
        var A = [0.12, 0.47, 0.90], B = [1.0, 0.44, 0.26], D = [0.80, 0.58, 0.85], i, j, o, x, z, s, t;
        for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
          o = (j * NU + i) * 3; s = i / (NU - 1); t = j / (NV - 1); x = s * 2 - 1; z = t * 2 - 1;
          base[o] = x * SPAN / 2; base[o + 2] = z * SPAN / 2;
          base[o + 1] = 1.0 * Math.exp(-((x - 0.25) * (x - 0.25) + (z + 0.2) * (z + 0.2)) * 2.4) -
                        0.55 * Math.exp(-((x + 0.6) * (x + 0.6) + (z - 0.55) * (z - 0.55)) * 3.0) + 0.14 * x;
          for (var c = 0; c < 3; c++) patch.C[o + c] = KT.lerp(KT.lerp(A[c], B[c], s), D[c], t * 0.75);
        }
        o = (2 * NU + 3) * 3; patch.C[o] = 1; patch.C[o + 1] = 0.86; patch.C[o + 2] = 0.30;   // one loud control feature
        patch.P.set(base);
      })();

      /* ---------- scene objects ---------- */
      var grid = new T.GridHelper(9, 18, 0x1b2a42, 0x111a2b); grid.position.y = -1.15; scene.add(grid);

      var FN = 33, fillGeo = new T.BufferGeometry(), fillPos = new Float32Array(FN * FN * 3), idx = [], a, b;
      for (b = 0; b < FN - 1; b++) for (a = 0; a < FN - 1; a++) {
        var q = b * FN + a; idx.push(q, q + 1, q + FN, q + 1, q + FN + 1, q + FN);
      }
      fillGeo.setAttribute('position', new T.BufferAttribute(fillPos, 3)); fillGeo.setIndex(idx);
      var fill = new T.Mesh(fillGeo, new T.MeshBasicMaterial({ color: 0x0a1826, side: T.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 }));
      fill.frustumCulled = false; scene.add(fill);

      var WL = 13, WS = 40, wire = G.lines(2 * WL * WS * 2, C.surf, 0.55); scene.add(wire.obj);
      var net = G.lines(((NU - 1) * NV + NU * (NV - 1)) * 2, C.ctrl, 0.55); scene.add(net.obj);
      var balls = [], ballGeo = new T.SphereGeometry(0.06, 14, 10);
      for (a = 0; a < NU * NV; a++) {
        var m = new T.Mesh(ballGeo, new T.MeshBasicMaterial({ color: C.ctrl })); m.userData.i = a;
        balls.push(m); scene.add(m);
      }

      var field = G.splats(NMAX * NMAX); scene.add(field.mesh);
      var one = G.splats(1, { blend: true }); one.uniforms.uRim.value = 0; one.mesh.renderOrder = 10; scene.add(one.mesh);
      var probe = new T.Mesh(new T.SphereGeometry(0.05, 16, 12), new T.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
      probe.renderOrder = 21; scene.add(probe);
      var aU = G.arrow(C.surf), aV = G.arrow(C.surf), aN = G.arrow(0xffffff);
      scene.add(aU.obj); scene.add(aV.obj); scene.add(aN.obj);
      var EL = 72, ell = new T.LineLoop(new T.BufferGeometry().setAttribute('position',
        new T.BufferAttribute(new Float32Array(EL * 3), 3)), new T.LineBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true, opacity: 0.85 }));
      ell.frustumCulled = false; ell.renderOrder = 15; scene.add(ell);

      // floating labels
      function tag(html, tone) { var e = ui.el('div', 'kt-tag kt-tag-' + tone, html); root.appendChild(e); return e; }
      var tMu = tag('μ', 'text'), tU = tag('S<sub>u</sub>', 'surf'), tV = tag('S<sub>v</sub>', 'surf'), tN = tag('n', 'text');
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);

      // (u,v) chart
      var uvBox = ui.el('div', 'kt-uv'); root.appendChild(uvBox);
      uvBox.appendChild(ui.el('span', 'kt-uv-lab', '(u, v) chart'));
      var uvc = ui.canvas(uvBox, 'kt-uv-cv');

      /* ---------- controls ---------- */
      var st = { scene: 'net', sc: SCENES.net, show: {}, n: 14, gain: 0.75, u: 0.5, v: 0.5, t: 0,
                 touchedProbe: false, touchedN: false, wigI: -1 };
      var panel = ui.panel(root, { pos: 'tr', title: 'sampling' });
      var slN = ui.slider(panel, { label: 'samples', min: 4, max: NMAX, step: 1, value: st.n,
        fmt: function (v) { return v + '²'; },
        onInput: function (v) { st.n = v; st.touchedN = true; fieldDirty = true; uvDirty = true; } });
      var slG = ui.slider(panel, { label: 'σ / cell', min: 0.3, max: 1.1, step: 0.01, value: st.gain,
        fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { st.gain = v; fieldDirty = true; } });
      var rdG = ui.readout(panel, { label: 'Gaussians', tone: 'splat' });
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      var rdZ = ui.readout(panel, { label: 'params / Gaussian', tone: 'ok' });
      rdP.set((NU * NV * PER_CP).toLocaleString('en-US')); rdZ.set('0');

      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, theta: 0.72, phi: 1.0, radius: 7.2, target: [0, 0.05, 0],
        minRadius: 3, maxRadius: 20, pick: pickBall });
      var surfDirty = true, fieldDirty = true, uvDirty = true;

      /* ---------- drag a control point ---------- */
      var ray = new T.Raycaster(), ndc = new T.Vector2(), plane = new T.Plane(), hitP = new T.Vector3(), dragCP = null;
      function setNdc(e) {
        var r = R.canvas.getBoundingClientRect();
        ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1));
        ray.setFromCamera(ndc, cam);
      }
      function pickBall(e) {
        if (!st.show.net) return false;
        setNdc(e);
        var h = ray.intersectObjects(balls, false)[0];
        if (!h) return false;
        dragCP = { i: h.object.userData.i, id: e.pointerId };
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), h.object.position);
        R.canvas.setPointerCapture(e.pointerId); st.wigI = -1; st.sc = Object.assign({}, st.sc, { wig: 0 });
        return true;
      }
      R.canvas.addEventListener('pointermove', function (e) {
        if (!dragCP) { if (st.show.net) { setNdc(e); R.canvas.style.cursor = ray.intersectObjects(balls, false)[0] ? 'grab' : ''; } return; }
        setNdc(e);
        if (ray.ray.intersectPlane(plane, hitP)) {
          var o = dragCP.i * 3;
          patch.P[o] = KT.clamp(hitP.x, -3, 3); patch.P[o + 1] = KT.clamp(hitP.y, -1.1, 2.4); patch.P[o + 2] = KT.clamp(hitP.z, -3, 3);
          base.set(patch.P); surfDirty = fieldDirty = true;
        }
      });
      function endCP(e) { if (dragCP && e.pointerId === dragCP.id) dragCP = null; }
      R.canvas.addEventListener('pointerup', endCP); R.canvas.addEventListener('pointercancel', endCP);

      /* ---------- (u,v) chart interaction ---------- */
      var uvDrag = false;
      function uvSet(e) {
        var r = uvc.cv.getBoundingClientRect();
        st.u = KT.clamp((e.clientX - r.left) / r.width, 0.02, 0.98);
        st.v = KT.clamp(1 - (e.clientY - r.top) / r.height, 0.02, 0.98);
        st.touchedProbe = true; uvDirty = true;
      }
      uvc.cv.style.touchAction = 'none';
      uvc.cv.addEventListener('pointerdown', function (e) { uvDrag = true; uvc.cv.setPointerCapture(e.pointerId); uvSet(e); });
      uvc.cv.addEventListener('pointermove', function (e) { if (uvDrag) uvSet(e); });
      uvc.cv.addEventListener('pointerup', function () { uvDrag = false; });
      uvc.cv.addEventListener('pointercancel', function () { uvDrag = false; });

      /* ---------- geometry updates ---------- */
      function lin(n) { var o = new Float32Array(n); for (var i = 0; i < n; i++) o[i] = i / (n - 1); return o; }
      var wU = lin(WS + 1), wL = lin(WL), fL = lin(FN);
      var gA = { pos: new Float32Array((WS + 1) * WL * 3) }, gB = { pos: new Float32Array((WS + 1) * WL * 3) };
      function updateSurface() {
        var i, j, o, n = 0, A = wire.arr, P = patch.P;
        NB.patchGrid(patch, wU, wL, gA);            // lines along u (one per v)
        for (j = 0; j < WL; j++) for (i = 0; i < WS; i++) {
          o = (j * (WS + 1) + i) * 3;
          A[n++] = gA.pos[o]; A[n++] = gA.pos[o + 1]; A[n++] = gA.pos[o + 2];
          A[n++] = gA.pos[o + 3]; A[n++] = gA.pos[o + 4]; A[n++] = gA.pos[o + 5];
        }
        NB.patchGrid(patch, wL, wU, gB);            // lines along v (one per u)
        for (i = 0; i < WL; i++) for (j = 0; j < WS; j++) {
          o = (j * WL + i) * 3; var o2 = ((j + 1) * WL + i) * 3;
          A[n++] = gB.pos[o]; A[n++] = gB.pos[o + 1]; A[n++] = gB.pos[o + 2];
          A[n++] = gB.pos[o2]; A[n++] = gB.pos[o2 + 1]; A[n++] = gB.pos[o2 + 2];
        }
        wire.set(n / 3);
        NB.patchGrid(patch, fL, fL, { pos: fillPos }); fillGeo.attributes.position.needsUpdate = true;
        n = 0; A = net.arr;
        for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
          o = (j * NU + i) * 3; balls[j * NU + i].position.set(P[o], P[o + 1], P[o + 2]);
          if (i < NU - 1) { A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o + 3]; A[n++] = P[o + 4]; A[n++] = P[o + 5]; }
          if (j < NV - 1) { var p2 = o + NU * 3; A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[p2]; A[n++] = P[p2 + 1]; A[n++] = P[p2 + 2]; }
        }
        net.set(n / 3);
      }

      var fg = { pos: new Float32Array(NMAX * NMAX * 3), su: new Float32Array(NMAX * NMAX * 3),
                 sv: new Float32Array(NMAX * NMAX * 3), col: new Float32Array(NMAX * NMAX * 3) };
      var violet = G.rgb(C.splat);
      function updateField() {
        var n = st.n, us = new Float32Array(n), d = new Float32Array(n), i;
        for (i = 0; i < n; i++) { us[i] = (i + 0.5) / n; d[i] = 1 / n; }
        NB.patchGrid(patch, us, us, fg);
        G.writePlanoids(field, 0, fg, n, n, d, d, st.gain, st.show.feat ? null : violet);
        field.commit(n * n); st.fieldN = n;
        rdG.set((n * n).toLocaleString('en-US'));
      }

      var pr = null;
      function updateProbe() {
        pr = NB.patchPoint(patch, st.u, st.v);
        var p = pr.pos, su = pr.su, sv = pr.sv, k = 1 / st.n * st.gain;
        probe.position.set(p[0], p[1], p[2]);
        var lu = Math.hypot(su[0], su[1], su[2]), uv = (su[0] * sv[0] + su[1] * sv[1] + su[2] * sv[2]) / (lu * lu);
        var px = [sv[0] - uv * su[0], sv[1] - uv * su[1], sv[2] - uv * su[2]], lv = Math.hypot(px[0], px[1], px[2]);
        var nx = su[1] * px[2] - su[2] * px[1], ny = su[2] * px[0] - su[0] * px[2], nz = su[0] * px[1] - su[1] * px[0];
        var ln = Math.hypot(nx, ny, nz); if (ny < 0) { nx = -nx; ny = -ny; nz = -nz; }
        aU.aim(p, su[0], su[1], su[2], lu * 0.26); aV.aim(p, sv[0], sv[1], sv[2], Math.hypot(sv[0], sv[1], sv[2]) * 0.26);
        aN.aim(p, nx, ny, nz, 0.8);
        pr.tipU = [p[0] + su[0] * 0.26, p[1] + su[1] * 0.26, p[2] + su[2] * 0.26];
        pr.tipV = [p[0] + sv[0] * 0.26, p[1] + sv[1] * 0.26, p[2] + sv[2] * 0.26];
        pr.tipN = [p[0] + nx / ln * 0.8, p[1] + ny / ln * 0.8, p[2] + nz / ln * 0.8];
        // the planoid itself: 1-sigma ellipse + the Gaussian
        var E = ell.geometry.attributes.position.array, i, c, s;
        for (i = 0; i < EL; i++) {
          c = Math.cos(i / EL * 6.2832) * k; s = Math.sin(i / EL * 6.2832) * k;
          E[i * 3] = p[0] + su[0] * c + px[0] * s; E[i * 3 + 1] = p[1] + su[1] * c + px[1] * s + 0.004; E[i * 3 + 2] = p[2] + su[2] * c + px[2] * s;
        }
        ell.geometry.attributes.position.needsUpdate = true;
        one.center.set(p); one.axU.set([su[0] * k, su[1] * k, su[2] * k]); one.axV.set([px[0] * k, px[1] * k, px[2] * k]);
        one.color.set(violet); one.setCount(1); one.commit();
        pr.lu = lu; pr.lv = lv;
      }

      function place(el, w, dx, dy) {
        var q = w && G.project(cam, w[0], w[1], w[2], W, H);
        if (!q) { el.style.opacity = 0; return; }
        el.style.opacity = 1; el.style.transform = 'translate(' + (q[0] + dx).toFixed(1) + 'px,' + (q[1] + dy).toFixed(1) + 'px)';
      }

      function drawUV() {
        var s = uvc.fit(), g = uvc.ctx, w = s.w, h = s.h, i, n = st.n;
        g.clearRect(0, 0, w, h);
        g.strokeStyle = 'rgba(255,167,38,.38)'; g.lineWidth = 1;            // knot lines = patch seams of the basis
        var inner = NU - 3;
        for (i = 1; i < inner; i++) {
          g.beginPath(); g.moveTo(i / inner * w, 0); g.lineTo(i / inner * w, h); g.moveTo(0, i / inner * h); g.lineTo(w, i / inner * h); g.stroke();
        }
        if (st.show.field || st.show.ell) {
          g.fillStyle = C.splat;
          if (n <= 44) { var r = Math.max(1, Math.min(3, w / n * 0.22));
            for (var b2 = 0; b2 < n; b2++) for (var a2 = 0; a2 < n; a2++) { g.beginPath(); g.arc((a2 + 0.5) / n * w, (b2 + 0.5) / n * h, r, 0, 6.2832); g.fill(); } }
          else { g.globalAlpha = Math.min(0.75, 0.25 + n / 320); g.fillRect(0, 0, w, h); g.globalAlpha = 1; }
        }
        if (st.show.probe) {
          var x = st.u * w, y = (1 - st.v) * h;
          if (st.show.ell) { g.strokeStyle = C.splat; g.lineWidth = 1.5; g.strokeRect(x - w / n / 2, y - h / n / 2, w / n, h / n); }
          g.strokeStyle = C.text; g.lineWidth = 1.2;
          g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
          g.fillStyle = C.text; g.beginPath(); g.arc(x, y, 4, 0, 6.2832); g.fill();
        }
      }

      /* ---------- scenes ---------- */
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.t = 0; st.touchedProbe = false; st.touchedN = false; st.wigI = -1;
        st.show = {}; sc.show.split(' ').forEach(function (k) { st.show[k] = true; });
        patch.P.set(base);
        if (sc.n) { st.n = sc.n; slN.set(sc.n); }
        if (sc.fixed) { st.u = sc.fixed[0]; st.v = sc.fixed[1]; }
        field.uniforms.uRim.value = sc.rim == null ? 0.6 : sc.rim;
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        var s = st.show;
        net.obj.visible = !!s.net; balls.forEach(function (m, i) {
          m.visible = !!s.net;
          if (s.feat) m.material.color.setRGB(patch.C[i * 3], patch.C[i * 3 + 1], patch.C[i * 3 + 2], T.SRGBColorSpace);
          else m.material.color.set(C.ctrl);
          m.scale.setScalar(s.feat ? 1.25 : 1);
        });
        wire.obj.visible = !!s.wire; wire.mat.opacity = s.field && !sc.hero ? 0.22 : 0.55;
        field.mesh.visible = !!s.field;
        probe.visible = !!s.probe; aU.obj.visible = aV.obj.visible = !!s.tan; aN.obj.visible = !!s.nor;
        fill.visible = !s.field || !!sc.hero;
        ell.visible = one.mesh.visible = !!s.ell;
        tMu.style.display = s.probe ? '' : 'none'; tU.style.display = tV.style.display = s.tan ? '' : 'none';
        tN.style.display = s.nor ? '' : 'none';
        uvBox.style.display = s.uv ? '' : 'none';
        panel.style.display = sc.hero ? 'none' : ''; cap.style.display = sc.hero ? 'none' : '';
        slN.el.style.display = slG.el.style.display = rdG.el.style.display = (s.field || s.ell) ? '' : 'none';
        grid.visible = !sc.hero;
        if (!orbit.touched) orbit.to(sc.hero ? { th: 0.9, ph: 1.12, r: 5.6 * fitR } : { th: 0.72, ph: 1.0, r: 7.2 * fitR });
        surfDirty = fieldDirty = uvDirty = true;
      }

      env.loop(function (t, dt) {
        var sc = st.sc, s = st.show, i, j, o;
        st.t += dt;
        if (sc.hero) {                         // the cover: a breathing surface, splats sweeping across it
          for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
            o = (j * NU + i) * 3; patch.P[o + 1] = base[o + 1] + 0.2 * Math.sin(st.t * 0.7 + i * 0.9 + j * 0.6);
          }
          surfDirty = fieldDirty = true;
        }
        if (sc.wig && !dragCP) {               // wiggle the control point that weighs most on the probe
          if (st.wigI < 0) { var best = 0; NB.patchSupport(patch, st.u, st.v).forEach(function (c) { if (c.w > best) { best = c.w; st.wigI = c.j * NU + c.i; } }); }
          o = st.wigI * 3;
          patch.P[o + 1] = base[o + 1] + 0.55 * Math.sin(st.t * 1.5); patch.P[o] = base[o] + 0.3 * Math.sin(st.t * 0.9);
          surfDirty = true;
        }
        if (s.probe && !sc.fixed && !st.touchedProbe) {
          st.u = 0.5 + 0.3 * Math.sin(st.t * 0.5); st.v = 0.5 + 0.3 * Math.sin(st.t * 0.37 + 1); uvDirty = true;
        }
        if (sc.sweep && !st.touchedN) {
          var n = Math.round(Math.exp(KT.lerp(Math.log(6), Math.log(150), 0.5 - 0.5 * Math.cos(st.t * 0.55))));
          if (n !== st.n) { st.n = n; slN.set(n); fieldDirty = true; }
        }
        if (surfDirty) { updateSurface(); surfDirty = false; }
        if (s.field && fieldDirty) { updateField(); fieldDirty = false; }
        if (s.field) {
          var cnt = st.fieldN * st.fieldN;
          if (sc.hero) cnt = Math.round((0.5 - 0.42 * Math.cos(st.t * 0.33)) * st.fieldN) * st.fieldN;
          field.setCount(cnt);
        }
        if (s.probe) updateProbe();
        if (!orbit.touched) orbit.goal.th = (sc.hero ? 0.9 : 0.72) + 0.22 * Math.sin(t * 0.17);
        orbit.update(dt);
        R.renderer.render(scene, cam);
        if (s.probe && pr) {
          place(tMu, pr.pos, 10, -26);
          if (s.tan) { place(tU, pr.tipU, 8, -10); place(tV, pr.tipV, 8, -10); }
          if (s.nor) place(tN, pr.tipN, 8, -14);
        }
        if (s.uv && uvDirty) { drawUV(); uvDirty = false; }
      });

      if (!document.getElementById('kt-planoid-css')) {
        var css = document.createElement('style'); css.id = 'kt-planoid-css';
        css.textContent =
          '.kt-sim-planoid .kt-tag{position:absolute;left:0;top:0;z-index:3;pointer-events:none;font:600 15px/1 var(--f-mono);' +
          'padding:3px 6px;border-radius:5px;background:rgba(7,9,15,.7);will-change:transform}' +
          '.kt-tag-surf{color:var(--surf)}.kt-tag-text{color:var(--text)}' +
          '.kt-sim-planoid .kt-tag sub{font-size:10px}' +
          '.kt-sim-planoid .kt-uv{position:absolute;z-index:5;left:12px;bottom:12px;width:clamp(120px,22%,200px);aspect-ratio:1;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;backdrop-filter:blur(10px)}' +
          '.kt-sim-planoid .kt-uv-cv{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}' +
          '.kt-sim-planoid .kt-uv-lab{position:absolute;z-index:1;left:6px;top:6px;pointer-events:none;font:600 10px/1 var(--f-mono);' +
          'letter-spacing:.12em;text-transform:uppercase;color:var(--dim);background:rgba(7,9,15,.82);padding:3px 5px;border-radius:4px}';
        document.head.appendChild(css);
      }

      setScene(env.opts.scene || 'net');
      return {
        setScene: setScene,
        resize: function (w, h) {
          W = w; H = h; R.resize(w, h); uvDirty = true;
          fitR = Math.max(1, 1.32 / (w / h));               // back off on narrow stages so the patch stays in frame
          if (!orbit.touched) orbit.to({ r: (st.sc.hero ? 5.6 : 7.2) * fitR });
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
