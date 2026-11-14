/* Knot-a-Surface talk — module `manifold`: "A manifold of splines".
   A whole scene as an atlas of B-spline charts (a spline manifold). Every Gaussian on screen is
   derived from one of the charts; a FIXED Gaussian budget is re-sampled over the (u,v) domains
   as the camera moves (view-adaptive sampling), and editing a control point moves thousands of
   Gaussians at once.
   Tool: WebGL (three.js) instanced planoids + a 2D-canvas (u,v) atlas.
   Standalone: talk/sims/manifold.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var PER_CP = 16;                  // 3 position + 12 SH1 colour + 1 opacity per control point
  var CAP = 170000, FLOOR = 0.10;   // FLOOR = share of the budget kept as view-independent coverage

  var VIEWS = {
    wide: { th: 0.78, ph: 1.14, r: 11.2, tx: -0.2, ty: 1.2, tz: 0.2 },
    knot: { th: 1.05, ph: 1.28, r: 3.1, tx: 1.3, ty: 1.78, tz: 0.6 },
    macro: { th: 1.0, ph: 1.22, r: 1.75, tx: 1.45, ty: 2.25, tz: 0.75 },
    vase: { th: 0.3, ph: 1.2, r: 5.0, tx: -1.6, ty: 1.15, tz: 0.9 },
    cloth: { th: 1.5, ph: 1.0, r: 4.6, tx: 1.3, ty: 0.7, tz: 0.6 }
  };
  var SCENES = {
    scene:  { view: 'splats', cam: 'wide', cap: '<b>one scene</b> · six spline charts · every Gaussian derived',
              hint: 'drag to orbit' },
    charts: { view: 'charts', cam: 'wide', cap: '<b>an atlas of charts</b> · each coherent surface gets its own (u,v) domain' },
    nets:   { view: 'charts', cam: 'wide', nets: 1, cap: '<b>control nets</b> · dense where the geometry is, sparse where it isn’t' },
    budget: { view: 'splats', tour: 1, cap: '<b>fixed budget, moving camera</b> · samples flow to what is seen' },
    lod:    { view: 'splats', cam: 'macro', ab: 1, rim: 0.6, cap: '<b>level of detail</b> · the same budget, with and without view-adaptive sampling' },
    edit:   { view: 'splats', cam: 'cloth', nets: 1, breathe: 1, cap: '<b>editable by construction</b> · move control points, the Gaussians follow',
              hint: 'drag an amber point' }
  };

  /* ---------- the scene: six charts ---------- */
  function mix3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function buildCharts() {
    var out = [];
    function chart(name, hex, o, fn) {
      var p = NB.makePatch(o), i, j, q, r;
      for (j = 0; j < o.nv; j++) for (i = 0; i < o.nu; i++) {
        r = fn(o.periodicU ? i / o.nu : i / (o.nu - 1), o.periodicV ? j / o.nv : j / (o.nv - 1), i, j); q = (j * o.nu + i) * 3;
        p.P[q] = r[0]; p.P[q + 1] = r[1]; p.P[q + 2] = r[2]; p.C[q] = r[3]; p.C[q + 1] = r[4]; p.C[q + 2] = r[5];
      }
      // each chart's (u,v) domain is cut into tu x tv tiles; every tile gets its own sample lattice
      var c = { name: name, hex: hex, patch: p, closed: !!o.closed, flip: 1, ref: o.ref || null,
                tu: o.tu || 4, tv: o.tv || 4, gu: (o.tu || 4) * (o.cu || 6), gv: (o.tv || 4) * (o.cv || 6),
                minA: o.minA || 2, minB: o.minB || 2, tiles: [], count: 0 };
      for (j = 0; j < c.tv; j++) for (i = 0; i < c.tu; i++) c.tiles.push({ iu: i, iv: j, na: 0, nb: 0 });
      c.base = new Float32Array(p.P); out.push(c);
      return c;
    }
    // large planar regions: 4 x 4 is plenty
    chart('floor', '#8fa4c0', { nu: 4, nv: 4, tu: 4, tv: 4 }, function (s, t) {
      var c = mix3([0.19, 0.21, 0.27], [0.34, 0.33, 0.35], 0.5 * (s + t));
      return [s * 6.4 - 3.2, 0, t * 6.4 - 3.2, c[0], c[1], c[2]];
    });
    chart('wall A', '#4fc3f7', { nu: 4, nv: 4, tu: 4, tv: 2 }, function (s, t) {
      var c = mix3([0.13, 0.17, 0.25], [0.22, 0.27, 0.38], t);
      return [s * 6.4 - 3.2, t * 3.4, -3.2, c[0], c[1], c[2]];
    });
    chart('wall B', '#69f0ae', { nu: 4, nv: 4, tu: 4, tv: 2 }, function (s, t) {
      var c = mix3([0.17, 0.15, 0.24], [0.27, 0.23, 0.36], t);
      return [-3.2, t * 3.4, 3.2 - s * 6.4, c[0], c[1], c[2]];
    });
    // a vase: surface of revolution, closed around u
    var VR = [0.30, 0.34, 0.52, 0.60, 0.50, 0.31, 0.20, 0.24, 0.31], VY = [0, 0.05, 0.3, 0.62, 0.95, 1.22, 1.42, 1.58, 1.72];
    var VC = [[0.14, 0.30, 0.55], [0.14, 0.30, 0.55], [0.93, 0.87, 0.75], [0.82, 0.38, 0.24], [0.93, 0.87, 0.75],
              [0.14, 0.30, 0.55], [0.93, 0.87, 0.75], [0.82, 0.38, 0.24], [0.93, 0.87, 0.75]];
    var VX = -1.6, VZ = 0.9, VS = 1.35;
    chart('vase', '#ffa726', { nu: 10, nv: 9, periodicU: true, closed: true, tu: 5, tv: 3, cu: 6, cv: 8, minA: 5,
      ref: function (u, v, pos) { return [VX, pos[1], VZ]; } }, function (s, t, i, j) {
      var a = s * 6.2832;
      return [VX + VS * VR[j] * Math.cos(a), VS * VY[j], VZ + VS * VR[j] * Math.sin(a), VC[j][0], VC[j][1], VC[j][2]];
    });
    // a cloth draped over a plinth
    chart('cloth', '#ef5350', { nu: 15, nv: 15, tu: 4, tv: 4, cu: 9, cv: 9 }, function (s, t, i, j) {
      var x = s * 2 - 1, z = t * 2 - 1, L = 1.2, h = 0.55, rho = Math.max(Math.abs(x), Math.abs(z)) || 1e-6;
      var f = KT.clamp((rho * L - h) / (L - h), 0, 1), ang = Math.atan2(z, x);
      var rad = rho * L <= h ? rho * L : h + 0.10 * f + 0.11 * Math.sin(ang * 7) * f;
      var y = 0.92 * (1 - 0.97 * Math.pow(f, 0.8)) + 0.03;
      var stripe = (i % 4 === 1 || j % 4 === 1) ? 1 : 0;
      var c = mix3([0.08, 0.40, 0.45], [0.80, 0.90, 0.86], stripe * 0.8);
      return [1.3 + x / rho * rad, y, 0.6 + z / rho * rad, c[0], c[1], c[2]];
    });
    // the namesake: a tube around a (2,3) torus knot, closed both ways
    var KN = 72, KM = 6, TUBE = 0.105, KS = 0.34, KY = 1.82, PAL = [KT.gl.rgb(C.surf), KT.gl.rgb(C.splat), KT.gl.rgb(C.ctrl)];
    function knotC(u) {
      var a = u * 2 * 6.2832, cs = Math.cos(1.5 * a), x = KS * (2 + cs) * Math.cos(a), y = KS * (2 + cs) * Math.sin(a), z = KS * Math.sin(1.5 * a);
      var ca = Math.cos(0.78), sa = Math.sin(0.78);
      return [x * ca + z * sa, y, -x * sa + z * ca];
    }
    chart('knot', '#ce93d8', { nu: KN, nv: KM, periodicU: true, periodicV: true, closed: true, tu: 36, tv: 1, cu: 4, cv: 12, minB: 14,
      ref: function (u) { var c = knotC(u); return [1.3 + c[0], KY + c[1], 0.6 + c[2]]; } }, function (s, t) {
      var p0 = knotC(s), p1 = knotC(s + 0.002), tx = p1[0] - p0[0], ty = p1[1] - p0[1], tz = p1[2] - p0[2];
      var sx = p1[0] + p0[0], sy = p1[1] + p0[1], sz = p1[2] + p0[2];
      var bx = ty * sz - tz * sy, by = tz * sx - tx * sz, bz = tx * sy - ty * sx, bl = Math.hypot(bx, by, bz);
      bx /= bl; by /= bl; bz /= bl;
      var nx = by * tz - bz * ty, ny = bz * tx - bx * tz, nz = bx * ty - by * tx, nl = Math.hypot(nx, ny, nz);
      nx /= nl; ny /= nl; nz /= nl;
      var a = t * 6.2832, ca = Math.cos(a) * TUBE, sa = Math.sin(a) * TUBE;
      var h = s * 3, k = Math.floor(h) % 3, c = mix3(PAL[k], PAL[(k + 1) % 3], h - Math.floor(h));
      return [1.3 + p0[0] + nx * ca + bx * sa, KY + p0[1] + ny * ca + by * sa, 0.6 + p0[2] + nz * ca + bz * sa, c[0], c[1], c[2]];
    });
    out.knotY = KY;
    return out;
  }

  KT.register('manifold', {
    title: 'A manifold of splines',
    blurb: 'A scene as an atlas of spline charts. A fixed Gaussian budget is re-sampled across the charts as the ' +
           'camera moves, each chart keeps the control net it deserves, and editing a control point moves the splats with it.',
    tool: 'WebGL · three.js · instanced planoids',
    scenes: [
      { id: 'scene', label: 'The scene' }, { id: 'charts', label: 'Atlas of charts' },
      { id: 'nets', label: 'Control nets' }, { id: 'budget', label: 'Fixed budget' },
      { id: 'lod', label: 'Level of detail' }, { id: 'edit', label: 'Edit' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 80);
      var R = G.renderer(root, cam), scene = new T.Scene();
      var charts = buildCharts(), W = 1, H = 1;
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      var st = { scene: 'scene', sc: SCENES.scene, view: 'splats', adaptive: true, nets: false, t: 0,
                 budget: +env.opts.budget || (small ? 24000 : 60000), k: 0.72, tourI: 0, tourT: 0, abT: 0, touchedAB: false };

      var field = G.splats(CAP); field.uniforms.uRim.value = 0.35; scene.add(field.mesh);
      var totalCP = 0, segs = 0;
      charts.forEach(function (c) {
        var p = c.patch; totalCP += p.nu * p.nv;
        segs += (p.periodicU ? p.nu : p.nu - 1) * p.nv + (p.periodicV ? p.nv : p.nv - 1) * p.nu;
      });
      var netL = G.lines(segs * 2, C.ctrl, 0.5); netL.obj.renderOrder = 5; scene.add(netL.obj);
      var ptGeo = new T.BufferGeometry(), ptPos = new Float32Array(totalCP * 3);
      ptGeo.setAttribute('position', new T.BufferAttribute(ptPos, 3));
      var pts = new T.Points(ptGeo, new T.PointsMaterial({ color: C.ctrl, size: 5, sizeAttenuation: false }));
      pts.frustumCulled = false; scene.add(pts);

      /* ---------- chrome ---------- */
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);
      var badge = ui.el('div', 'kt-ab'); root.appendChild(badge);
      var panel = ui.panel(root, { pos: 'tr', title: 'budget' });
      var slB = ui.slider(panel, { label: 'budget', min: 4000, max: CAP, step: 1000, value: st.budget,
        fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, onInput: function (v) { st.budget = v; need = true; } });
      var segV = ui.seg(panel, { label: 'colour', value: 'splats',
        options: [{ v: 'splats', label: 'appearance' }, { v: 'charts', label: 'charts' }],
        onChange: function (v) { st.view = v; need = true; } });
      var rowT = ui.el('div', 'kt-row'); panel.appendChild(rowT);
      var tgA = ui.toggle(rowT, { label: 'view-adaptive', value: true, onChange: function (v) { st.adaptive = v; st.touchedAB = true; need = true; paintBadge(); } });
      var tgN = ui.toggle(rowT, { label: 'nets', value: false, onChange: function (v) { st.nets = v; showNets(); } });
      var rdG = ui.readout(panel, { label: 'Gaussians drawn', tone: 'splat' });
      var rdV = ui.readout(panel, { label: 'budget in view', tone: 'ok' });
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      rdP.set(totalCP + ' × ' + PER_CP + ' = ' + (totalCP * PER_CP).toLocaleString('en-US'));


      // (u,v) atlas: one tile per chart
      var atlas = ui.el('div', 'kt-atlas'); root.appendChild(atlas);
      charts.forEach(function (c) {
        var tile = ui.el('div', 'kt-tile');
        tile.style.setProperty('--c', c.hex);
        tile.appendChild(ui.el('div', 'kt-tile-name', c.name));
        var box = ui.el('div', 'kt-tile-box'); tile.appendChild(box);
        c.cv = ui.canvas(box, 'kt-tile-cv');
        c.lab = ui.el('div', 'kt-tile-n'); tile.appendChild(c.lab);
        c.sub = ui.el('div', 'kt-tile-s', c.patch.nu + '×' + c.patch.nv + ' ctrl'); tile.appendChild(c.sub);
        atlas.appendChild(tile);
      });

      var orbit = G.orbit(cam, R.canvas, Object.assign({ embed: env.embed, minRadius: 1.6, maxRadius: 16, pick: pickPoint,
        theta: VIEWS.wide.th, phi: VIEWS.wide.ph, radius: VIEWS.wide.r, target: [VIEWS.wide.tx, VIEWS.wide.ty, VIEWS.wide.tz] }));

      /* ---------- coarse cells: where each chart is, which way it faces, how big ---------- */
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
        if (c.closed && c.ref) {             // orient normals outwards once
          var k = Math.floor(n * 0.37), u = us[k % gu], v = vs[Math.floor(k / gu)];
          var rf = c.ref(u, v, [g.pos[k * 3], g.pos[k * 3 + 1], g.pos[k * 3 + 2]]);
          var d = (g.pos[k * 3] - rf[0]) * c.cnrm[k * 3] + (g.pos[k * 3 + 1] - rf[1]) * c.cnrm[k * 3 + 1] + (g.pos[k * 3 + 2] - rf[2]) * c.cnrm[k * 3 + 2];
          c.flip = d < 0 ? -1 : 1;
        }
      }

      /* ---------- view-adaptive allocation: importance -> per-chart counts -> warped (u,v) lattices ---------- */
      var M = new T.Matrix4(), need = true, lastAlloc = 0, camKey = '';
      function invCdf(m, n, out, dOut) {     // n samples from the density m (piecewise-constant), plus local spacing
        var g = m.length, cdf = new Float64Array(g + 1), i, tot = 0;
        for (i = 0; i < g; i++) { tot += m[i]; cdf[i + 1] = tot; }
        function at(q) { var x = q * tot, k = 0; while (k < g - 1 && cdf[k + 1] < x) k++; return (k + (x - cdf[k]) / (m[k] || 1e-12)) / g; }
        var prev = 0;
        for (i = 0; i < n; i++) { var e = at((i + 1) / n); out[i] = at((i + 0.5) / n); dOut[i] = e - prev; prev = e; }
      }
      function allocate() {
        cam.updateMatrixWorld(); M.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
        var e = M.elements, cx = cam.position.x, cy = cam.position.y, cz = cam.position.z;
        var totA = 0, totW = 0, inView = 0;
        charts.forEach(function (c) { totA += c.area; });
        charts.forEach(function (c) {
          var n = c.gu * c.gv, i, o, w; c.w = c.w || new Float32Array(n); c.vis = c.vis || new Uint8Array(n);
          for (i = 0; i < n; i++) {
            o = i * 3; var x = c.cpos[o], y = c.cpos[o + 1], z = c.cpos[o + 2];
            var cw = e[3] * x + e[7] * y + e[11] * z + e[15], px = e[0] * x + e[4] * y + e[8] * z + e[12], py = e[1] * x + e[5] * y + e[9] * z + e[13];
            w = 0; c.vis[i] = 0;
            if (cw > 0.05 && Math.abs(px) < cw * 1.12 && Math.abs(py) < cw * 1.12) {
              var dx = cx - x, dy = cy - y, dz = cz - z, d2 = dx * dx + dy * dy + dz * dz, dl = Math.sqrt(d2);
              var cs = (dx * c.cnrm[o] + dy * c.cnrm[o + 1] + dz * c.cnrm[o + 2]) / dl * c.flip;
              // open charts: foreshortening. closed charts: the near side counts in full (flat planoids must
              // stay small all the way to the silhouette), the hidden far side keeps a thin share
              if (c.closed) { c.vis[i] = cs > -0.15 ? 1 : 0; cs = c.vis[i] ? 1 : 0.3; }
              else { cs = Math.max(0.25, Math.abs(cs)); c.vis[i] = 1; }
              w = c.carea[i] * cs / d2;
            }
            c.w[i] = w; totW += w;
          }
        });
        charts.forEach(function (c) {
          var gu = c.gu, cu = gu / c.tu, cv = c.gv / c.tv;
          c.count = 0;
          c.tiles.forEach(function (tl) {
            var mu = new Float64Array(cu), mv = new Float64Array(cv), share = 0, lu = 0, lv = 0, a, b, i, imp;
            for (b = 0; b < cv; b++) for (a = 0; a < cu; a++) {
              i = (tl.iv * cv + b) * gu + tl.iu * cu + a;
              imp = st.adaptive ? FLOOR * c.carea[i] / totA + (1 - FLOOR) * c.w[i] / (totW || 1) : c.carea[i] / totA;
              share += imp; mu[a] += imp; mv[b] += imp; lu += imp * c.clu[i] / c.tu; lv += imp * c.clv[i] / c.tv;
              if (c.vis[i]) inView += imp;
            }
            var N = Math.max(4, st.budget * share), ratio = lu / (lv || 1e-12);     // aim for square cells in world space
            // curved closed charts keep a minimum count around the bend so flat planoids can follow it
            tl.nb = Math.max(c.minB, Math.round(Math.sqrt(N / ratio))); tl.na = Math.max(c.minA, Math.round(N / tl.nb));
            for (a = 0; a < cu; a++) mu[a] += share / cu * 0.3;                       // bounded contrast inside a tile:
            for (b = 0; b < cv; b++) mv[b] += share / cv * 0.3;                       // no giant cells on the far side
            tl.us = new Float32Array(tl.na); tl.du = new Float32Array(tl.na); tl.vs = new Float32Array(tl.nb); tl.dv = new Float32Array(tl.nb);
            invCdf(mu, tl.na, tl.us, tl.du); invCdf(mv, tl.nb, tl.vs, tl.dv);
            for (a = 0; a < tl.na; a++) { tl.us[a] = (tl.iu + tl.us[a]) / c.tu; tl.du[a] /= c.tu; }
            for (b = 0; b < tl.nb; b++) { tl.vs[b] = (tl.iv + tl.vs[b]) / c.tv; tl.dv[b] /= c.tv; }
            c.count += tl.na * tl.nb;
          });
        });
        rdV.set(Math.round(inView * 100) + ' %');
      }

      var sg = { pos: new Float32Array(CAP * 3), su: new Float32Array(CAP * 3), sv: new Float32Array(CAP * 3), col: new Float32Array(CAP * 3) };
      function resample() {
        allocate();
        var n = 0;
        charts.forEach(function (c) {
          var rgb = st.view === 'charts' ? G.rgb(c.hex) : null;
          c.tiles.forEach(function (tl) {
            if (n + tl.na * tl.nb > CAP) return;
            NB.patchGrid(c.patch, tl.us, tl.vs, sg);
            n = G.writePlanoids(field, n, sg, tl.na, tl.nb, tl.du, tl.dv, st.k, rgb);
          });
          c.lab.textContent = c.count.toLocaleString('en-US') + ' G';
          drawTile(c);
        });
        field.setCount(n); field.commit();
        rdG.set(n.toLocaleString('en-US'));
      }
      function drawTile(c) {
        var s = c.cv.fit(), g = c.cv.ctx, w = s.w, h = s.h, i;
        g.clearRect(0, 0, w, h); g.fillStyle = c.hex;
        // density of each tile's lattice = brightness; a few lattice lines where they are sparse enough to read
        c.tiles.forEach(function (tl) {
          var x0 = tl.iu / c.tu * w, y0 = (1 - (tl.iv + 1) / c.tv) * h, tw = w / c.tu, th = h / c.tv;
          var dens = tl.na * tl.nb / (tw * th);                 // samples per CSS px of the tile
          g.globalAlpha = KT.clamp(dens * 0.16, 0.05, 0.9); g.fillRect(x0 + 0.5, y0 + 0.5, tw - 1, th - 1);
          if (tw / tl.na > 3) { g.globalAlpha = 0.85;
            for (i = 0; i < tl.na; i++) g.fillRect(Math.round(tl.us[i] * w), y0, 1, th); }
          if (th / tl.nb > 3) { g.globalAlpha = 0.85;
            for (i = 0; i < tl.nb; i++) g.fillRect(x0, Math.round((1 - tl.vs[i]) * h), tw, 1); }
        });
        g.globalAlpha = 1;
      }

      /* ---------- control nets ---------- */
      function updateNets() {
        var A = netL.arr, n = 0, q = 0;
        charts.forEach(function (c) {
          var p = c.patch, P = p.P, nu = p.nu, nv = p.nv, i, j, o, o2;
          c.ptOff = q / 3;
          ptPos.set(P, q); q += P.length;
          for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
            o = (j * nu + i) * 3;
            if (i < nu - 1 || p.periodicU) { o2 = (j * nu + (i + 1) % nu) * 3; A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o2]; A[n++] = P[o2 + 1]; A[n++] = P[o2 + 2]; }
            if (j < nv - 1 || p.periodicV) { o2 = (((j + 1) % nv) * nu + i) * 3; A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o2]; A[n++] = P[o2 + 1]; A[n++] = P[o2 + 2]; }
          }
        });
        netL.set(n / 3); ptGeo.attributes.position.needsUpdate = true;
      }
      function showNets() { netL.obj.visible = pts.visible = st.nets; tgN.set(st.nets); }

      /* ---------- drag a control point ---------- */
      var ray = new T.Raycaster(), ndc = new T.Vector2(), plane = new T.Plane(), hitP = new T.Vector3(), dragCP = null;
      ray.params.Points.threshold = 0.07;
      function setNdc(e) {
        var r = R.canvas.getBoundingClientRect();
        ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1)); ray.setFromCamera(ndc, cam);
      }
      function pickPoint(e) {
        if (!st.nets) return false;
        setNdc(e);
        var h = ray.intersectObject(pts, false)[0];
        if (!h) return false;
        var gi = h.index, c = null;
        charts.forEach(function (k) { if (gi >= k.ptOff && gi < k.ptOff + k.patch.nu * k.patch.nv) c = k; });
        dragCP = { c: c, i: gi - c.ptOff, id: e.pointerId }; st.touchedEdit = true;
        var o = dragCP.i * 3;
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), hitP.set(c.patch.P[o], c.patch.P[o + 1], c.patch.P[o + 2]));
        R.canvas.setPointerCapture(e.pointerId);
        return true;
      }
      R.canvas.addEventListener('pointermove', function (e) {
        if (!dragCP) return;
        setNdc(e);
        if (ray.ray.intersectPlane(plane, hitP)) {
          var o = dragCP.i * 3, P = dragCP.c.patch.P;
          P[o] = hitP.x; P[o + 1] = hitP.y; P[o + 2] = hitP.z; dragCP.c.base.set(P);
          coarse(dragCP.c); updateNets(); need = true;
        }
      });
      function endCP(e) { if (dragCP && e.pointerId === dragCP.id) dragCP = null; }
      R.canvas.addEventListener('pointerup', endCP); R.canvas.addEventListener('pointercancel', endCP);

      /* ---------- scenes ---------- */
      function paintBadge() {
        badge.innerHTML = 'view-adaptive sampling <b class="' + (st.adaptive ? 'on' : 'off') + '">' + (st.adaptive ? 'ON' : 'OFF') + '</b>';
      }
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.t = 0; st.tourI = 0; st.tourT = 0; st.abT = 0; st.touchedAB = false; st.touchedEdit = false;
        st.view = sc.view; segV.set(sc.view); st.nets = !!sc.nets; showNets();
        st.adaptive = true; tgA.set(true); paintBadge();
        badge.style.display = sc.ab ? '' : 'none';
        field.uniforms.uRim.value = sc.rim || 0.35;
        charts.forEach(function (c) { c.patch.P.set(c.base); coarse(c); });
        updateNets();
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        orbit.touched = false;
        if (sc.cam) orbit.to(VIEWS[sc.cam]); else if (sc.tour) orbit.to(VIEWS.wide);
        need = true;
      }

      var TOUR = ['wide', 'vase', 'knot', 'cloth'];
      env.loop(function (t, dt) {
        var sc = st.sc;
        st.t += dt;
        if (sc.tour && !orbit.touched) {
          st.tourT += dt;
          if (st.tourT > 5.2) { st.tourT = 0; st.tourI = (st.tourI + 1) % TOUR.length; orbit.to(VIEWS[TOUR[st.tourI]]); }
        }
        if (sc.ab && !st.touchedAB) {
          st.abT += dt;
          if (st.abT > 3.4) { st.abT = 0; st.adaptive = !st.adaptive; tgA.set(st.adaptive); paintBadge(); need = true; }
        }
        if (sc.breathe && !st.touchedEdit && !dragCP) {     // lift a corner of the cloth; let the knot breathe
          var cl = charts[4], kn = charts[5], P = cl.patch.P, B = cl.base, i, j, o, nu = cl.patch.nu;
          var lift = 0.5 - 0.5 * Math.cos(st.t * 1.1);
          for (j = 0; j < 5; j++) for (i = nu - 5; i < nu; i++) {
            o = (j * nu + i) * 3; var f = (4 - j) / 4 * (i - (nu - 5)) / 4;
            P[o + 1] = B[o + 1] + lift * 0.9 * f; P[o] = B[o] + lift * 0.35 * f; P[o + 2] = B[o + 2] - lift * 0.35 * f;
          }
          var s = 1 + 0.16 * Math.sin(st.t * 1.6), Q = kn.patch.P, KB = kn.base;
          for (i = 0; i < Q.length; i += 3) { Q[i] = 1.3 + (KB[i] - 1.3) * s; Q[i + 1] = charts.knotY + (KB[i + 1] - charts.knotY) * s; Q[i + 2] = 0.6 + (KB[i + 2] - 0.6) * s; }
          coarse(cl); coarse(kn); updateNets(); need = true;
        }
        if (!orbit.touched && sc.cam === 'wide') orbit.goal.th = VIEWS.wide.th + 0.16 * Math.sin(t * 0.16);
        orbit.update(dt, sc.tour ? 1.5 : 3);
        var e = cam.matrixWorld.elements, key = (e[12] + e[13] * 3 + e[14] * 7 + e[8] * 11 + e[9] * 13).toFixed(3);
        if (key !== camKey && t - lastAlloc > 0.09) { camKey = key; need = true; }
        if (need) { need = false; lastAlloc = t; resample(); }
        R.renderer.render(scene, cam);
      });

      if (!document.getElementById('kt-manifold-css')) {
        var css = document.createElement('style'); css.id = 'kt-manifold-css';
        css.textContent =
          '.kt-sim-manifold .kt-atlas{position:absolute;z-index:5;left:12px;bottom:12px;display:grid;gap:6px;' +
          'grid-template-columns:repeat(6,minmax(0,84px));padding:8px;background:var(--panel);border:1px solid var(--line);' +
          'border-radius:10px;backdrop-filter:blur(10px);max-width:calc(100% - 24px)}' +
          '.kt-sim-manifold .kt-tile{min-width:0}' +
          '.kt-sim-manifold .kt-tile-name{font:600 10px/1 var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--c);margin-bottom:4px}' +
          '.kt-sim-manifold .kt-tile-box{position:relative;aspect-ratio:1.25;border:1px solid var(--line2);border-radius:4px;overflow:hidden;background:var(--bg)}' +
          '.kt-sim-manifold .kt-tile-cv{position:absolute;inset:0;width:100%;height:100%}' +
          '.kt-sim-manifold .kt-tile-n{font:600 11px/1 var(--f-mono);color:var(--text);margin-top:5px;font-variant-numeric:tabular-nums;white-space:nowrap}' +
          '.kt-sim-manifold .kt-tile-s{font:500 10px/1 var(--f-mono);color:var(--mute);margin-top:3px;white-space:nowrap}' +
          '.kt-sim-manifold .kt-ab{position:absolute;z-index:4;left:50%;top:52px;transform:translateX(-50%);white-space:nowrap;' +
          'font:600 13px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--dim);padding:9px 14px;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:999px}' +
          '.kt-sim-manifold .kt-ab b{margin-left:8px}.kt-sim-manifold .kt-ab .on{color:var(--ok)}.kt-sim-manifold .kt-ab .off{color:var(--err)}' +
          '@media (max-width:640px){.kt-sim-manifold .kt-atlas{grid-template-columns:repeat(6,minmax(0,1fr));gap:4px;padding:6px}' +
          '.kt-sim-manifold .kt-tile-s{display:none}.kt-sim-manifold .kt-tile-name{font-size:8px;letter-spacing:.04em}.kt-sim-manifold .kt-tile-n{font-size:9px}}';
        document.head.appendChild(css);
      }

      setScene(env.opts.scene || 'scene');
      return {
        setScene: setScene,
        resize: function (w, h) { W = w; H = h; R.resize(w, h); need = true; },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
