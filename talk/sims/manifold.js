/* Knot-a-Surface talk — module `manifold`: "A manifold of splines" (the full-power demo).
   A whole scene as an atlas of B-spline charts — a spline manifold. Every Gaussian on screen is
   derived from one of the charts; a FIXED Gaussian budget is re-sampled over the (u,v) domains
   as the camera moves (view-adaptive sampling), and editing control points moves thousands of
   Gaussians at once. Scenes come from talk/scenes.js: benchmark-style stand-ins (barn, truck,
   bust, garden) plus the abstract still life. Any scene can be read as appearance, depth, normals,
   single Gaussians or chart identity — or all five at once, in a rolling split screen.
   Tool: WebGL (three.js) instanced planoids + a 2D-canvas (u,v) atlas.
   Standalone: talk/sims/manifold.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js, scenes.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var PER_CP = 16;                  // 3 position + 12 SH1 colour + 1 opacity per control point
  var CAP = 220000, FLOOR = 0.10;   // FLOOR = share of the budget kept as view-independent coverage
  /* what a Gaussian shows; the order is the shader's uMode (gl.js). The view 'split' shows all of them */
  var MODES = [{ v: 'splats', label: 'rgb', name: 'appearance' }, { v: 'depth', label: 'depth', name: 'depth' },
               { v: 'normal', label: 'normal', name: 'normals' }, { v: 'gauss', label: 'gauss', name: 'Gaussians' },
               { v: 'charts', label: 'chart', name: 'charts' }];
  var ROLL = 0.03, SLANT = 0.2;     // split screen: band travel in stage widths per second, and the lean of its dividers
  function wrap(x, m) { return ((x % m) + m) % m; }

  var SCENES = {
    scene:  { scn: 'barn', view: 'splats', cam: 'wide', rig: 1, cap: 'every Gaussian derived from a spline chart', hint: 'drag to orbit' },
    modes:  { scn: 'barn', view: 'split', cam: 'wide', cap: 'one set of Gaussians, read five ways at once' },
    charts: { scn: 'barn', view: 'charts', cam: 'wide', cap: 'an atlas of charts · each coherent surface gets its own (u,v) domain' },
    nets:   { scn: 'barn', view: 'charts', cam: 'wide', nets: 1, cap: 'control nets · geometry and appearance live on the amber points' },
    budget: { scn: 'barn', view: 'splats', tour: 1, cap: 'fixed budget, moving camera · samples flow to what is seen' },
    lod:    { scn: 'barn', view: 'splats', cam: 'close', ab: 1, rim: 0.6, cap: 'level of detail · the same budget, with and without view-adaptive sampling' },
    truck:  { scn: 'truck', view: 'splats', tour: 1, rig: 1, cap: 'same method, another capture' },
    bust:   { scn: 'bust', view: 'splats', tour: 1, rig: 1, cap: 'a scanned object under a camera dome' },
    garden: { scn: 'garden', view: 'splats', tour: 1, rig: 1, cap: 'an object inside an unbounded surround' },
    edit:   { scn: 'bust', view: 'splats', cam: 'a', breathe: 1, cap: 'editable by construction · move control points, the Gaussians follow',
              hint: 'turn on nets to drag a control point' },
    studio: { scn: 'studio', view: 'splats', cam: 'wide', cap: 'the abstract still life' },
    /* chrome-free, for pages that embed the scene as a backdrop (mount with { bare: true }) */
    hero:   { scn: 'barn', view: 'splats', tour: 1, cap: '' }
  };

  KT.register('manifold', {
    title: 'A manifold of splines',
    blurb: 'Benchmark-style scenes — a barn, a truck, a bust, a garden — built entirely from spline charts. A fixed Gaussian ' +
           'budget is re-sampled across the charts as the camera moves, and editing a control point moves the splats with it. ' +
           'The scenes are procedural stand-ins modelled after Tanks and Temples, DTU and Mip-NeRF 360 captures, not the datasets.',
    tool: 'WebGL · three.js · instanced planoids',
    scenes: [
      { id: 'scene', label: 'Barn' }, { id: 'modes', label: 'All modalities' }, { id: 'charts', label: 'Atlas of charts' }, { id: 'nets', label: 'Control nets' },
      { id: 'budget', label: 'Fixed budget' }, { id: 'lod', label: 'Level of detail' }, { id: 'truck', label: 'Truck' },
      { id: 'bust', label: 'Bust' }, { id: 'garden', label: 'Garden' }, { id: 'edit', label: 'Edit' }, { id: 'studio', label: 'Still life' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 160);
      var R = G.renderer(root, cam, env.opts.clear), scene = new T.Scene();
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      var st = { scene: 'scene', sc: SCENES.scene, scn: '', view: 'splats', adaptive: true, nets: false, rig: false, t: 0,
                 budget: +env.opts.budget || (small ? 30000 : 90000), k: 0.72, tourI: 0, tourT: 0, abT: 0, touchedAB: false, touchedEdit: false };
      var def = null, charts = [], totalCP = 0, netL = null, pts = null, ptPos = null, rigL = null;
      var need = true, lastAlloc = 0, camKey = '', fit = 1;
      /* a named camera of the current scene, backed off on narrow stages so the subject stays in frame */
      function view(name) { var v = Object.assign({}, def.views[name]); v.r *= fit; return v; }

      var field = G.splats(CAP); scene.add(field.mesh);

      /* ---------- chrome ---------- */
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);
      var badge = ui.el('div', 'kt-ab'); root.appendChild(badge);
      var panel = ui.panel(root, { pos: 'tr', title: 'scene · budget' });
      var segS = ui.seg(panel, { value: 'barn', options: Object.keys(KT.scenes).filter(function (k) { return !KT.scenes[k].hidden; }).map(function (k) { return { v: k, label: KT.scenes[k].label.toLowerCase() }; }),
        onChange: function (v) {
          st.sc = Object.assign({}, st.sc, { scn: v, tour: 0, breathe: 0, ab: 0, cam: 'wide' });
          loadScene(v); orbit.touched = false; orbit.to(view('wide'), true); paintCap();
        } });
      var slB = ui.slider(panel, { label: 'budget', min: 6000, max: CAP - 20000, step: 2000, value: Math.min(st.budget, CAP - 20000),
        fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, onInput: function (v) { st.budget = v; need = true; } });
      var segV = ui.seg(panel, { value: 'splats', options: MODES.concat([{ v: 'split', label: 'all' }]),
        onChange: function (v) { st.view = v; showView(); } });
      var rowT = ui.el('div', 'kt-row'); panel.appendChild(rowT);
      var tgA = ui.toggle(rowT, { label: 'adaptive', value: true, onChange: function (v) { st.adaptive = v; st.touchedAB = true; need = true; paintBadge(); } });
      var tgN = ui.toggle(rowT, { label: 'nets', value: false, onChange: function (v) { st.nets = v; showNets(); } });
      var tgR = ui.toggle(rowT, { label: 'cameras', value: false, onChange: function (v) { st.rig = v; showNets(); } });
      var rdG = ui.readout(panel, { label: 'Gaussians drawn', tone: 'splat' });
      var rdV = ui.readout(panel, { label: 'budget in view', tone: 'ok' });
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      var atlas = ui.el('div', 'kt-atlas'); root.appendChild(atlas);
      /* split screen: one divider and one label per band, laid over the canvas and moved every frame */
      var split = ui.el('div', 'kt-split'), bands = MODES.map(function (m) {
        var b = { a: ui.el('i'), b: ui.el('i'), lab: ui.el('span', '', m.name) };   // a divider can wrap: two copies
        split.appendChild(b.a); split.appendChild(b.b); split.appendChild(b.lab);
        return b;
      });
      root.appendChild(split);
      var roll = 0, zr = [1, 10], size = { w: 1, h: 1 };
      function showView() {
        var i = MODES.map(function (m) { return m.v; }).indexOf(st.view), all = i < 0;
        field.uniforms.uMode.value = all ? 0 : i;
        field.uniforms.uSplit.value.set(all ? MODES.length : 0, roll, SLANT);
        root.classList.toggle('kt-splitting', all);
      }
      function placeSplit() {
        var w = size.w, h = size.h, n = MODES.length, tight = w / n < 130;
        bands.forEach(function (b, j) {
          var yl = h - 26 - (tight ? j % 3 * 30 : 0);           // narrow bands: stagger the labels so neighbours never touch
          var x = w * wrap(j / n + roll, 1), sk = ' skewX(' + (-Math.atan(SLANT)) + 'rad)';
          b.a.style.transform = 'translateX(' + x + 'px)' + sk; b.b.style.transform = 'translateX(' + (x + w) + 'px)' + sk;
          var xc = wrap(w * ((j + 0.5) / n + roll) - SLANT * yl, w);
          b.lab.style.transform = 'translate(' + xc + 'px,' + yl + 'px) translate(-50%,-50%)';
          b.lab.style.opacity = KT.clamp(Math.min(xc, w - xc) / 44 - 0.4, 0, 1);
        });
      }

      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 1.4, maxRadius: 46, pick: pickPoint,
        hit: function (x, y) { return findPoint(x, y, true) >= 0; },
        release: function () {
          if (!dragCP) return;
          dragCP.c.patch.P.set(dragCP.was, dragCP.i * 3); coarse(dragCP.c); updateNets(); need = true; dragCP = null;
        } });

      /* ---------- load a scene: charts, nets, atlas tiles, capture rig ---------- */
      function loadScene(key) {
        if (st.scn === key) { charts.forEach(function (c) { c.patch.P.set(c.base); coarse(c); }); updateNets(); need = true; return; }
        st.scn = key; def = KT.scenes[key].build(); charts = def.charts; segS.set(key);
        [netL && netL.obj, pts, rigL && rigL.obj].forEach(function (o) { if (o) { scene.remove(o); o.geometry.dispose(); o.material.dispose(); } });
        totalCP = 0; var segs = 0;
        charts.forEach(function (c) {
          var p = c.patch; totalCP += p.nu * p.nv;
          segs += (p.periodicU ? p.nu : p.nu - 1) * p.nv + (p.periodicV ? p.nv : p.nv - 1) * p.nu;
        });
        netL = G.lines(segs * 2, C.ctrl, totalCP > 3000 ? 0.3 : 0.5); netL.obj.renderOrder = 5; scene.add(netL.obj);
        var ptGeo = new T.BufferGeometry(); ptPos = new Float32Array(totalCP * 3);
        ptGeo.setAttribute('position', new T.BufferAttribute(ptPos, 3));
        pts = new T.Points(ptGeo, new T.PointsMaterial({ color: C.ctrl, size: totalCP > 3000 ? 3 : 5, sizeAttenuation: false }));
        pts.frustumCulled = false; scene.add(pts);
        // capture rig: one small frustum per camera of the dataset-style rig
        var cams = def.rig.cams, tg = def.rig.target, n = 0;
        rigL = G.lines(cams.length * 16, C.text, 0.4); scene.add(rigL.obj);
        cams.forEach(function (p) {
          var f = new T.Vector3(tg[0] - p[0], tg[1] - p[1], tg[2] - p[2]), d = f.length() * 0.04; f.normalize();
          var r = new T.Vector3().crossVectors(f, new T.Vector3(0, 1, 0)).normalize(), u = new T.Vector3().crossVectors(r, f);
          var k = [], a;
          [[1, 1], [-1, 1], [-1, -1], [1, -1]].forEach(function (s) {
            k.push([p[0] + f.x * d + (r.x * s[0] * 0.7 + u.x * s[1] * 0.5) * d, p[1] + f.y * d + (r.y * s[0] * 0.7 + u.y * s[1] * 0.5) * d, p[2] + f.z * d + (r.z * s[0] * 0.7 + u.z * s[1] * 0.5) * d]);
          });
          for (a = 0; a < 4; a++) {
            rigL.arr.set(p, n); rigL.arr.set(k[a], n + 3); rigL.arr.set(k[a], n + 6); rigL.arr.set(k[(a + 1) % 4], n + 9); n += 12;
          }
        });
        rigL.set(n / 3);
        atlas.innerHTML = '';
        atlas.style.gridTemplateColumns = 'repeat(' + Math.min(charts.length, small ? 4 : 8) + ',minmax(0,76px))';
        charts.forEach(function (c) {
          var tile = ui.el('div', 'kt-tile'); tile.style.setProperty('--c', c.hex);
          tile.appendChild(ui.el('div', 'kt-tile-name', c.name));
          var box = ui.el('div', 'kt-tile-box'); tile.appendChild(box);
          c.cv = ui.canvas(box, 'kt-tile-cv');
          c.lab = ui.el('div', 'kt-tile-n'); tile.appendChild(c.lab);
          tile.appendChild(ui.el('div', 'kt-tile-s', c.patch.nu + '×' + c.patch.nv + ' ctrl'));
          atlas.appendChild(tile); c.w = null; c.vis = null;
        });
        charts.forEach(coarse); updateNets(); showNets();
        // no point sampling finer than a chart's own detail: charts with a sparse control net per unit
        // area (ground, backdrops) get a smaller share of the budget than densely described ones
        var rho = charts.map(function (c) { return c.patch.nu * c.patch.nv / c.area; }), rmax = Math.max.apply(null, rho);
        charts.forEach(function (c, i) { c.gain = KT.clamp(Math.pow(rho[i] / rmax, 0.35), 0.3, 1); });
        rdP.set(totalCP.toLocaleString('en-US') + ' × ' + PER_CP + ' = ' + (totalCP * PER_CP).toLocaleString('en-US'));
        need = true;
      }

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

      /* ---------- view-adaptive allocation: importance -> per-tile counts -> warped (u,v) lattices ---------- */
      var M = new T.Matrix4();
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
        var totA = 0, totW = 0, inView = 0, z0 = 1e9, z1 = 0;
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
              w = c.carea[i] * cs / d2 * c.gain;
              if (cw < z0) z0 = cw; if (cw > z1) z1 = cw;
            }
            c.w[i] = w; totW += w;
          }
        });
        // size every tile's lattice; rounding to na x nb drifts off the budget, so a second pass rescales
        var total = sizeTiles(1, true);
        if (Math.abs(total - st.budget) > 0.001 * st.budget) sizeTiles(st.budget / total, false);
        rdV.set(Math.round(inView * 100) + ' %');
        if (z1 > z0) zr = [z0 * 0.9, z1 * 1.05];      // depth of what is in view: the range the depth colours span

        function sizeTiles(scale, first) {
        var sum = 0;
        charts.forEach(function (c) {
          var gu = c.gu, cu = gu / c.tu, cv = c.gv / c.tv;
          c.count = 0;
          c.tiles.forEach(function (tl) {
            var mu = new Float64Array(cu), mv = new Float64Array(cv), share = 0, lu = 0, lv = 0, a, b, i, imp;
            for (b = 0; b < cv; b++) for (a = 0; a < cu; a++) {
              i = (tl.iv * cv + b) * gu + tl.iu * cu + a;
              imp = st.adaptive ? FLOOR * c.carea[i] / totA + (1 - FLOOR) * c.w[i] / (totW || 1) : c.carea[i] / totA;
              share += imp; mu[a] += imp; mv[b] += imp; lu += imp * c.clu[i] / c.tu; lv += imp * c.clv[i] / c.tv;
              if (first && c.vis[i]) inView += imp;
            }
            var N = Math.max(4, st.budget * share * scale), ratio = lu / (lv || 1e-12);   // aim for square cells in world space
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
          sum += c.count;
        });
        return sum;
        }
      }

      var sg = { pos: new Float32Array(CAP * 3), su: new Float32Array(CAP * 3), sv: new Float32Array(CAP * 3), col: new Float32Array(CAP * 3) };
      function resample() {
        allocate();
        var n = 0;
        charts.forEach(function (c) {
          var rgb = G.rgb(c.hex), n0 = n, i;
          c.tiles.forEach(function (tl) {
            if (n + tl.na * tl.nb > CAP) return;         // cannot happen while the slider tops out below CAP
            NB.patchGrid(c.patch, tl.us, tl.vs, sg);
            n = G.writePlanoids(field, n, sg, tl.na, tl.nb, tl.du, tl.dv, st.k);
          });
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          c.lab.textContent = c.count.toLocaleString('en-US') + ' G';
          drawTile(c);
        });
        field.setCount(n); field.commit(n);
        rdG.set(n.toLocaleString('en-US'));
        if (env.opts.onStats) env.opts.onStats({ scene: st.scn, drawn: n, controlPoints: totalCP, params: totalCP * PER_CP, charts: charts.length });
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
        netL.set(n / 3); pts.geometry.attributes.position.needsUpdate = true;
      }
      function showNets() {
        if (!netL) return;
        netL.obj.visible = pts.visible = st.nets; rigL.obj.visible = st.rig; tgN.set(st.nets); tgR.set(st.rig);
      }

      /* ---------- drag a control point ---------- */
      var ray = new T.Raycaster(), ndc = new T.Vector2(), plane = new T.Plane(), hitP = new T.Vector3(), dragCP = null;
      function setNdc(e) {
        var r = R.canvas.getBoundingClientRect();
        ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1)); ray.setFromCamera(ndc, cam);
      }
      /* the control point under client (x, y), as an index into ptPos: the ray's point threshold for a
         mouse, a fingertip-sized screen radius for touch */
      function findPoint(x, y, touch) {
        if (!st.nets || !pts) return -1;
        if (touch) return G.nearest(cam, R.canvas, ptPos, totalCP, x, y, KT.HIT);
        setNdc({ clientX: x, clientY: y }); ray.params.Points.threshold = orbit.cur.r * 0.012;
        var h = ray.intersectObject(pts, false)[0];
        return h ? h.index : -1;
      }
      function pickPoint(e) {
        var gi = findPoint(e.clientX, e.clientY, e.pointerType === 'touch'), c = null;
        if (gi < 0) return false;
        charts.forEach(function (k) { if (gi >= k.ptOff && gi < k.ptOff + k.patch.nu * k.patch.nv) c = k; });
        var o = (gi - c.ptOff) * 3;
        dragCP = { c: c, i: gi - c.ptOff, id: e.pointerId, was: c.patch.P.slice(o, o + 3) };
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), hitP.set(c.patch.P[o], c.patch.P[o + 1], c.patch.P[o + 2]));
        R.canvas.setPointerCapture(e.pointerId);
        return true;
      }
      R.canvas.addEventListener('pointermove', function (e) {
        if (!dragCP || e.pointerId !== dragCP.id) return;
        setNdc(e);
        if (ray.ray.intersectPlane(plane, hitP)) {
          var o = dragCP.i * 3, P = dragCP.c.patch.P;
          st.touchedEdit = true;           // only a drag that actually moves a point stops the scene's own edit
          P[o] = hitP.x; P[o + 1] = hitP.y; P[o + 2] = hitP.z;
          coarse(dragCP.c); updateNets(); need = true;
        }
      });
      function endCP(e) { if (dragCP && e.pointerId === dragCP.id) dragCP = null; }
      R.canvas.addEventListener('pointerup', endCP); R.canvas.addEventListener('pointercancel', endCP);

      /* ---------- scenes ---------- */
      function paintBadge() {
        badge.innerHTML = 'view-adaptive sampling <b class="' + (st.adaptive ? 'on' : 'off') + '">' + (st.adaptive ? 'ON' : 'OFF') + '</b>';
      }
      function paintCap() {
        var s = KT.scenes[st.scn];
        cap.innerHTML = '<b>' + s.label + '</b> · ' + s.suite + (st.scn === 'studio' ? '' : ' · procedural stand-in') + '<br>' + st.sc.cap;
        badge.style.display = st.sc.ab ? '' : 'none';
      }
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        if (id === 'hero' && env.opts.scn && KT.scenes[env.opts.scn]) sc = Object.assign({}, sc, { scn: env.opts.scn });
        st.scene = id; st.sc = sc; st.t = 0; st.tourI = 0; st.tourT = 0; st.abT = 0; st.touchedAB = false; st.touchedEdit = false;
        st.view = sc.view; segV.set(sc.view); showView(); st.nets = !!sc.nets; st.rig = !!sc.rig;
        st.adaptive = true; tgA.set(true); paintBadge();
        field.uniforms.uRim.value = sc.rim || 0.3;
        var fresh = st.scn !== sc.scn;
        loadScene(sc.scn); showNets(); paintCap();
        hint.textContent = sc.hint || '';
        orbit.touched = false;
        orbit.to(view(sc.cam || 'wide'), fresh);
        need = true;
      }

      env.loop(function (t, dt) {
        var sc = st.sc;
        st.t += dt;
        if (sc.tour && !orbit.touched) {
          st.tourT += dt;
          if (st.tourT > 5.2) { st.tourT = 0; st.tourI = (st.tourI + 1) % def.tour.length; orbit.to(view(def.tour[st.tourI])); }
        }
        if (sc.ab && !st.touchedAB) {
          st.abT += dt;
          if (st.abT > 3.4) { st.abT = 0; st.adaptive = !st.adaptive; tgA.set(st.adaptive); paintBadge(); need = true; }
        }
        if (sc.breathe && !st.touchedEdit && !dragCP) {     // the scene's own edit, eased in and out
          def.edit(0.5 - 0.5 * Math.cos(st.t * 1.15)).forEach(coarse);
          if (st.nets) updateNets();
          need = true;
        }
        var home = def.views[sc.cam || 'wide'];
        if (!orbit.touched && !sc.tour && (sc.cam || 'wide') === 'wide') orbit.goal.th = home.th + 0.14 * Math.sin(t * 0.16);
        orbit.update(dt, sc.tour ? 1.5 : 3);
        var e = cam.matrixWorld.elements, key = (e[12] + e[13] * 3 + e[14] * 7 + e[8] * 11 + e[9] * 13).toFixed(3);
        if (key !== camKey && t - lastAlloc > 0.09) { camKey = key; need = true; }
        if (need) { need = false; lastAlloc = t; resample(); }
        var ud = field.uniforms.uDepth.value, kz = 1 - Math.exp(-5 * dt);
        ud.x += (zr[0] - ud.x) * kz; ud.y += (zr[1] - ud.y) * kz;
        if (st.view === 'split') { roll = wrap(roll + ROLL * dt, 1); field.uniforms.uSplit.value.y = roll; placeSplit(); }
        R.renderer.render(scene, cam);
      });

      if (!document.getElementById('kt-manifold-css')) {
        var css = document.createElement('style'); css.id = 'kt-manifold-css';
        css.textContent =
          '.kt-sim-manifold .kt-atlas{position:absolute;z-index:5;left:12px;bottom:12px;display:grid;gap:6px;' +
          'padding:8px;background:var(--panel);border:1px solid var(--line);' +
          'border-radius:10px;backdrop-filter:blur(10px);max-width:calc(100% - 24px)}' +
          '.kt-sim-manifold .kt-tile{min-width:0}' +
          '.kt-sim-manifold .kt-tile-name{font:600 10px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--c);margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}' +
          '.kt-sim-manifold .kt-tile-box{position:relative;aspect-ratio:1.4;border:1px solid var(--line2);border-radius:4px;overflow:hidden;background:var(--bg)}' +
          '.kt-sim-manifold .kt-tile-cv{position:absolute;inset:0;width:100%;height:100%}' +
          '.kt-sim-manifold .kt-tile-n{font:600 11px/1 var(--f-mono);color:var(--text);margin-top:5px;font-variant-numeric:tabular-nums;white-space:nowrap}' +
          '.kt-sim-manifold .kt-tile-s{font:500 10px/1 var(--f-mono);color:var(--mute);margin-top:3px;white-space:nowrap}' +
          '.kt-sim-manifold .kt-caption{line-height:1.6}.kt-sim-manifold.kt-embed .kt-hint{display:none}' +
          '.kt-sim-manifold.kt-compact .kt-atlas{display:none}' +
          '.kt-sim-manifold.kt-bare .kt-panel,.kt-sim-manifold.kt-bare .kt-atlas,.kt-sim-manifold.kt-bare .kt-caption,' +
          '.kt-sim-manifold.kt-bare .kt-hint,.kt-sim-manifold.kt-bare .kt-ab,.kt-sim-manifold.kt-bare .kt-gear{display:none!important}' +
          '.kt-sim-manifold .kt-panel .kt-seg .kt-seg-box{margin-left:0;width:100%}' +
          '.kt-sim-manifold .kt-panel .kt-seg .kt-seg-btn{flex:1;padding:7px 4px}' +
          '.kt-sim-manifold .kt-split{position:absolute;inset:0;z-index:3;overflow:hidden;pointer-events:none;display:none}' +
          '.kt-sim-manifold.kt-splitting .kt-split{display:block}.kt-sim-manifold.kt-splitting .kt-atlas{display:none}' +
          '.kt-sim-manifold .kt-split i{position:absolute;left:-1px;top:0;width:2px;height:100%;transform-origin:0 0;background:rgba(255,255,255,.6)}' +
          '.kt-sim-manifold .kt-split span{position:absolute;left:0;top:0;white-space:nowrap;font:600 11px/1 var(--f-mono);letter-spacing:.1em;' +
          'text-transform:uppercase;color:var(--text);padding:7px 10px;background:var(--panel);border:1px solid var(--line);border-radius:999px}' +
          '.kt-sim-manifold .kt-ab{position:absolute;z-index:4;left:50%;top:64px;transform:translateX(-50%);white-space:nowrap;' +
          'font:600 13px/1 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--dim);padding:9px 14px;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:999px}' +
          '.kt-sim-manifold .kt-ab b{margin-left:8px}.kt-sim-manifold .kt-ab .on{color:var(--ok)}.kt-sim-manifold .kt-ab .off{color:var(--err)}' +
          '@media (max-width:640px){.kt-sim-manifold .kt-atlas{gap:4px;padding:6px}.kt-sim-manifold .kt-tile-box{aspect-ratio:2}' +
          '.kt-sim-manifold .kt-tile-s{display:none}.kt-sim-manifold .kt-tile-name{font-size:8px;letter-spacing:.04em}.kt-sim-manifold .kt-tile-n{font-size:9px}}';
        document.head.appendChild(css);
      }

      if (env.opts.bare) root.classList.add('kt-bare');
      setScene(env.opts.scene || 'scene');
      return {
        setScene: setScene, orbit: orbit, camera: cam, points: function () { return ptPos; },   // for tests
        /* for host pages: switch the scene or the budget without the module's own panel */
        load: function (key) { st.sc = Object.assign({}, st.sc, { scn: key }); st.tourI = 0; st.tourT = 0; loadScene(key); orbit.touched = false; orbit.to(view('wide'), true); paintCap(); },
        setBudget: function (n) { st.budget = KT.clamp(n, 6000, CAP - 20000); slB.set(st.budget); need = true; },
        resize: function (w, h) {
          R.resize(w, h); need = true; size.w = w; size.h = h;
          R.renderer.getDrawingBufferSize(field.uniforms.uRes.value);
          var f = KT.clamp(1.4 / (w / h), 1, 1.7);
          if (f !== fit) { fit = f; if (!orbit.touched) orbit.to(view(st.sc.tour ? def.tour[st.tourI] : st.sc.cam || 'wide'), true); }
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
