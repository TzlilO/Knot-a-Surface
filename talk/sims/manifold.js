/* Knot-a-Surface talk — module `manifold`: "A manifold of splines" (the full-power demo).
   A whole scene as an atlas of B-spline charts — a spline manifold. Every Gaussian on screen is
   derived from one of the charts; a FIXED Gaussian budget is re-sampled over the (u,v) domains
   as the camera moves (view-adaptive sampling), and editing control points moves thousands of
   Gaussians at once. Scenes come from talk/scenes.js: the real objects (`scan`, `scan<N>`: DTU scans
   fitted offline). The hand-modelled stand-ins (barn, truck, bust, garden, studio) are no longer offered
   here; the barn is only what a page without the scan data falls back to. Any scene can be read as appearance, depth, normals,
   single Gaussians or chart identity — or all five at once, in a rolling split screen.
   Tool: WebGL (three.js) instanced planoids + a 2D-canvas (u,v) atlas.
   Standalone: talk/sims/manifold.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js, scenes.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var PER_CP = 16;                  // 3 position + 12 SH1 colour + 1 opacity per control point
  var CAP = 320000, FLOOR = 0.10;   // FLOOR = share of the budget kept as view-independent coverage
  var DENSE = 96;                   // a control net longer than this along either side is drawn thinned out
  /* what a Gaussian shows; the order is the shader's uMode (gl.js). The view 'split' shows all of them */
  var MODES = [{ v: 'splats', label: 'rgb', name: 'appearance' }, { v: 'depth', label: 'depth', name: 'depth' },
               { v: 'normal', label: 'normal', name: 'normals' }, { v: 'gauss', label: 'gauss', name: 'Gaussians' },
               { v: 'charts', label: 'chart', name: 'charts' }];
  var ROLL = 0.03, SLANT = 0.2;     // split screen: band travel in stage widths per second, and the lean of its dividers
  function wrap(x, m) { return ((x % m) + m) % m; }

  /* The talk runs on ONE subject: `scan`, a real DTU object (talk/dtu-data.js; without that file the barn
     stands in). atlas: 1 shows the (u,v) tiles, tools: 1 the toggles. */
  var SCENES = {
    scene:  { scn: 'scan', view: 'splats', cam: 'wide', cap: 'every Gaussian derived from a spline chart', hint: 'drag to orbit' },
    modes:  { scn: 'scan', view: 'split', cam: 'wide', cap: 'one set of Gaussians, read five ways at once' },
    charts: { scn: 'scan', view: 'charts', cam: 'wide', atlas: 1, cap: 'an atlas of charts · each coherent surface gets its own (u,v) domain' },
    nets:   { scn: 'scan', view: 'charts', cam: 'wide', atlas: 1, nets: 1, cap: 'control nets · geometry and appearance live on the amber points' },
    budget: { scn: 'scan', view: 'splats', tour: 1, atlas: 1, cap: 'fixed budget, moving camera · samples flow to what is seen' },
    lod:    { scn: 'scan', view: 'splats', cam: 'close', ab: 1, rim: 0.6, atlas: 1, cap: 'level of detail · the same budget, with and without view-adaptive sampling' },
    /* the samples of each chart, stitched into triangles by a running 2 x 2 window */
    mesh:   { scn: 'scan', view: 'gauss', cam: 'wide', mesh: 1, cap: 'tessellation · a 2 × 2 window stitches the samples into triangles' },
    edit:   { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, tools: 1, cap: 'editable by construction · move control points, the Gaussians follow',
              hint: 'turn on nets to drag a control point' },
    paint:  { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, paint: 1, cap: 'material edit · the same net, new control colours' },
    /* chrome-free, for pages that embed the scene as a backdrop (mount with { bare: true }) */
    hero:   { scn: 'barn', view: 'splats', tour: 1, cap: '' }
  };

  KT.register('manifold', {
    title: 'A manifold of splines',
    blurb: 'A scene stored only as spline charts. A fixed Gaussian budget is re-sampled across the charts as the camera moves, ' +
           'and editing a control point moves the splats with it. The owl and the rabbit are real DTU scans, fitted offline to ' +
           'the dataset’s depth maps and photographs. They were not trained.',
    tool: 'WebGL · three.js · instanced planoids',
    scenes: [
      { id: 'scene', label: 'The object' }, { id: 'modes', label: 'All modalities' }, { id: 'charts', label: 'Atlas of charts' }, { id: 'nets', label: 'Control nets' },
      { id: 'budget', label: 'Fixed budget' }, { id: 'lod', label: 'Level of detail' }, { id: 'mesh', label: 'Stitch a mesh' },
      { id: 'edit', label: 'Edit' }, { id: 'paint', label: 'Materials' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 160);
      var R = G.renderer(root, cam, env.opts.clear), scene = new T.Scene();
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      var st = { scene: 'scene', sc: SCENES.scene, scn: '', view: 'splats', adaptive: true, nets: false, rig: false, t: 0,
                 budget: +env.opts.budget || (small ? 30000 : KT.dtu ? 160000 : 90000),   // the scanned object has detail to show for a larger budget
                 k: 0.72, cull: true, paint: 0, mat: 'snowy', tourI: 0, tourT: 0, abT: 0, touchedAB: false, touchedEdit: false };
      var segM = null;                 // the material picker of a scene that has materials (shown on the `paint` scene)
      var def = null, charts = [], totalCP = 0, shownCP = 0, netL = null, pts = null, ptPos = null, rigL = null;
      var need = true, lastAlloc = 0, camKey = '', fit = 1;
      /* a named camera of the current scene, backed off on narrow stages so the subject stays in frame */
      function view(name) { var v = Object.assign({}, def.views[name]); v.r *= fit; return v; }

      var field = G.splats(CAP, { alpha: true }); scene.add(field.mesh);     // alpha: a chart may carry an opacity per control point (c.alpha)

      /* ---------- chrome ---------- */
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);
      var badge = ui.el('div', 'kt-ab'); root.appendChild(badge);
      var panel = ui.panel(root, { pos: 'tr', title: 'scene · budget' });
      /* the picker offers the scanned objects; only a page without their data offers the hand-modelled scenes */
      var pick = Object.keys(KT.scenes).filter(function (k) { return !KT.scenes[k].hidden; }), real = pick.filter(function (k) { return KT.scenes[k].photo; });
      var segS = ui.seg(panel, { value: 'barn', options: (real.length ? real : pick).map(function (k) { return { v: k, label: KT.scenes[k].label.toLowerCase() }; }),
        onChange: function (v) {
          st.sc = Object.assign({}, st.sc, { scn: v, tour: 0, breathe: 0, ab: 0, cam: 'wide' });
          loadScene(v); syncMesh(); orbit.touched = false; orbit.to(view('wide'), true); paintCap();
        } });
      segS.el.classList.add('mf-pick');
      var slB = ui.slider(panel, { label: 'budget', min: 6000, max: CAP - 20000, step: 2000, value: Math.min(st.budget, CAP - 20000),
        fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, onInput: function (v) { st.budget = v; need = true; } });
      var segV = ui.seg(panel, { value: 'splats', options: MODES.concat([{ v: 'split', label: 'all' }]),
        onChange: function (v) { st.view = v; showView(); } });
      var rowT = ui.el('div', 'kt-row mf-tog'); panel.appendChild(rowT);
      var tgA = ui.toggle(rowT, { label: 'adaptive', value: true, onChange: function (v) { st.adaptive = v; st.touchedAB = true; need = true; paintBadge(); } });
      var tgN = ui.toggle(rowT, { label: 'nets', value: false, onChange: function (v) { st.nets = v; showNets(); } });
      var tgR = ui.toggle(rowT, { label: 'cameras', value: false, onChange: function (v) { st.rig = v; showNets(); } });
      ui.toggle(rowT, { label: 'no back faces', value: true, onChange: function (v) { st.cull = v; need = true; } });
      var rdG = ui.readout(panel, { label: 'Gaussians drawn', tone: 'splat' });
      var rdV = ui.readout(panel, { label: 'budget in view', tone: 'ok' });
      [slB.el, rdV.el].forEach(function (e) { e.classList.add('mf-samp'); });   // the view-adaptive sampler's rows: not in the mesh scene
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      var rdM = ui.readout(panel, { label: 'triangles stitched', tone: 'surf' }); rdM.el.classList.add('mf-tri');
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
          moveCP(0, 0, 0); dragCP = null;
        } });

      /* ---------- load a scene: charts, nets, atlas tiles, capture rig ---------- */
      function loadScene(key) {
        if (st.scn === key) { charts.forEach(function (c) { c.patch.P.set(c.base); if (c.baseC) c.patch.C.set(c.baseC); c.alpha = null; coarse(c); }); updateNets(); need = true; return; }
        st.scn = key; def = KT.scenes[key].build(); charts = def.charts; segS.set(key);
        field.uniforms.uFlat.value = def.photo || KT.scenes[key].photo ? 1 : 0;   // colours from photographs are already lit
        if (segM) { segM.el.remove(); segM = null; }
        if (def.materials) {
          if (!def.materials.some(function (m) { return m.v === st.mat; })) st.mat = def.materials[0].v;
          segM = ui.seg(panel, { value: st.mat, options: def.materials, onChange: pickMat });
          segM.el.classList.add('mf-mat');
        }
        [netL && netL.obj, pts, rigL && rigL.obj].forEach(function (o) { if (o) { scene.remove(o); o.geometry.dispose(); o.material.dispose(); } });
        totalCP = 0; shownCP = 0; var segs = 0;
        charts.forEach(function (c) {
          var p = c.patch; totalCP += p.nu * p.nv;
          // a dense net (a scanned object) is drawn as every ns-th row and column, or it is one solid colour
          c.ns = Math.max(p.nu, p.nv) > DENSE ? Math.ceil(Math.max(p.nu, p.nv) / 48) : 1;
          c.pi = every(p.nu, c.ns); c.pj = every(p.nv, c.ns); shownCP += c.pi.length * c.pj.length;
          segs += (p.periodicU ? p.nu : p.nu - 1) * c.pj.length + (p.periodicV ? p.nv : p.nv - 1) * c.pi.length;
        });
        netL = G.lines(segs * 2, C.ctrl, shownCP > 3000 ? 0.3 : 0.5); netL.obj.renderOrder = 5; scene.add(netL.obj);
        var ptGeo = new T.BufferGeometry(); ptPos = new Float32Array(shownCP * 3);
        ptGeo.setAttribute('position', new T.BufferAttribute(ptPos, 3));
        pts = new T.Points(ptGeo, new T.PointsMaterial({ color: C.ctrl, size: shownCP > 3000 || shownCP < totalCP ? 3 : 5, sizeAttenuation: false }));
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
        var goal = st.budget * boost;          // lattices are laid for a little more than the budget: back faces drop out when written
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
              if (c.closed) { c.vis[i] = cs > -0.15 ? 1 : 0; cs = c.vis[i] ? 1 : c.alpha ? 0.6 : st.cull ? 0 : 0.3; }      // culled: the far side asks for nothing (a see-through chart shows it)
              else { cs = Math.max(0.25, Math.abs(cs)); c.vis[i] = 1; }
              w = c.carea[i] * cs / d2 * c.gain;
              if (cw < z0) z0 = cw; if (cw > z1) z1 = cw;
            }
            c.w[i] = w; totW += w;
          }
        });
        // size every tile's lattice; rounding to na x nb drifts off the budget, so a second pass rescales
        var total = sizeTiles(1, true);
        if (Math.abs(total - goal) > 0.001 * goal) sizeTiles(goal / total, false);
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
            var N = Math.max(4, goal * share * scale), ratio = lu / (lv || 1e-12);   // aim for square cells in world space
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
      /* Back faces of closed charts are never written: they cannot be seen in this frame, so they should not
         spend the budget. What is written then falls short of it, so the next lattices are laid larger (boost). */
      var boost = 1;
      function resample() {
        var n = emit();
        if (st.cull) { boost = KT.clamp(boost * st.budget / Math.max(1, n), 1, 4); if (Math.abs(n - st.budget) > 0.1 * st.budget) n = emit(); } else boost = 1;
        field.setCount(n); field.commit(n);
        rdG.set(n.toLocaleString('en-US'));
        if (env.opts.onStats) env.opts.onStats({ scene: st.scn, drawn: n, controlPoints: totalCP, params: totalCP * PER_CP, charts: charts.length });
      }
      function emit() {
        allocate();
        var n = 0, eye = { x: cam.position.x, y: cam.position.y, z: cam.position.z, flip: 1, min: -0.12 }, thin = false;
        charts.forEach(function (c) {
          var rgb = G.rgb(c.hex), n0 = n, i, AL = c.alpha, p = c.patch;
          eye.flip = c.flip;
          c.tiles.forEach(function (tl) {
            if (tl.na * tl.nb > CAP) return;
            NB.patchGrid(c.patch, tl.us, tl.vs, sg);
            var t0 = n, k, a, b;
            // a see-through chart keeps its far side, and each Gaussian takes the opacity of the control point nearest to it
            n = G.writePlanoids(field, n, sg, tl.na, tl.nb, tl.du, tl.dv, st.k, null, st.cull && c.closed && !AL ? eye : null);
            if (!AL) { field.alpha.fill(1, t0, n); return; }
            for (k = t0; k < n; k++) {
              a = (k - t0) % tl.na; b = ((k - t0) / tl.na) | 0;
              field.alpha[k] = AL[Math.round(tl.vs[b] * (p.nv - 1)) * p.nu + Math.round(tl.us[a] * (p.nu - 1))];
            }
            thin = true;
          });
          c.count = n - n0;
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          c.lab.textContent = c.count.toLocaleString('en-US') + ' G';
          drawTile(c);
        });
        field.mesh.userData.ktSplat.thin = thin;
        return n;
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

      /* ---------- scene `mesh`: every chart sampled on one even lattice; the Gaussians sit on its points, and a
         2 x 2 window running over it turns each stop into two triangles. A Gaussian goes when its point has
         been stitched, so the instances are written in the reverse of the window's order: what is left to
         stitch is always the first part of the buffer. ---------- */
      var mesh = null, MESH_T = 12;
      function dropMesh() {
        if (!mesh) return;
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].forEach(function (o) { scene.remove(o); o.material.dispose(); o.geometry.dispose(); });
        mesh = null; root.classList.remove('mf-mesh'); need = true;
      }
      function syncMesh() { if (st.sc.mesh) buildMesh(); else dropMesh(); }
      function buildMesh() {
        dropMesh();
        // dense nets (a scanned object): a sample at every second control point; sparse ones: two per span
        var parts = charts.map(function (c) {
          var p = c.patch, f = Math.max(p.nu, p.nv) > DENSE ? 0.5 : 2;
          return { c: c, na: Math.min(240, Math.round(p.nu * f)), nb: Math.min(240, Math.round(p.nv * f)) };
        }).sort(function (a, b) { return a.na * a.nb - b.na * b.nb; });            // the window starts on the coarsest chart
        var V = 0, Q = 0, n = 0, a, b;
        parts.forEach(function (m) {
          var p = m.c.patch; m.off = V; m.q0 = Q; m.qa = p.periodicU ? m.na : m.na - 1; m.qb = p.periodicV ? m.nb : m.nb - 1;
          V += m.na * m.nb; Q += m.qa * m.qb;
        });
        var pos = new Float32Array(V * 3), col = new Float32Array(V * 3), tri = new Uint32Array(Q * 6), edge = new Uint32Array(Q * 6), t = 0;
        function lattice(m, back) {
          var us = new Float32Array(m.na), vs = new Float32Array(m.nb), du = new Float32Array(m.na).fill(1 / m.na), dv = new Float32Array(m.nb).fill(1 / m.nb);
          for (a = 0; a < m.na; a++) us[back ? m.na - 1 - a : a] = (a + 0.5) / m.na;
          for (b = 0; b < m.nb; b++) vs[back ? m.nb - 1 - b : b] = (b + 0.5) / m.nb;
          NB.patchGrid(m.c.patch, us, vs, sg);
          return { du: du, dv: dv };
        }
        parts.forEach(function (m) {
          lattice(m, false); pos.set(sg.pos.subarray(0, m.na * m.nb * 3), m.off * 3); col.set(sg.col.subarray(0, m.na * m.nb * 3), m.off * 3);
          for (b = 0; b < m.qb; b++) for (a = 0; a < m.qa; a++) {
            var v00 = m.off + b * m.na + a, v10 = m.off + b * m.na + (a + 1) % m.na, v01 = m.off + ((b + 1) % m.nb) * m.na + a, v11 = m.off + ((b + 1) % m.nb) * m.na + (a + 1) % m.na;
            tri[t] = v00; tri[t + 1] = v10; tri[t + 2] = v11; tri[t + 3] = v00; tri[t + 4] = v11; tri[t + 5] = v01;
            edge[t] = v00; edge[t + 1] = v10; edge[t + 2] = v00; edge[t + 3] = v01; edge[t + 4] = v00; edge[t + 5] = v11; t += 6;
          }
        });
        parts.slice().reverse().forEach(function (m) {
          var d = lattice(m, true), n0 = n, rgb = G.rgb(m.c.hex), i;
          n = G.writePlanoids(field, n, sg, m.na, m.nb, d.du, d.dv, st.k, null, null);
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          m.c.lab.textContent = (n - n0).toLocaleString('en-US') + ' G';
        });
        field.alpha.fill(1, 0, n); field.mesh.userData.ktSplat.thin = false; field.commit(n);
        var geo = new T.BufferGeometry(); geo.setAttribute('position', new T.BufferAttribute(pos, 3)); geo.setAttribute('aCol', new T.BufferAttribute(col, 3));
        geo.setIndex(new T.BufferAttribute(tri, 1));
        var wgeo = new T.BufferGeometry(); wgeo.setAttribute('position', geo.attributes.position); wgeo.setIndex(new T.BufferAttribute(edge, 1));
        var kgeo = new T.BufferGeometry(); kgeo.setAttribute('position', new T.BufferAttribute(new Float32Array(12), 3));
        mesh = { parts: parts, V: V, Q: Q, pos: pos, t: 0, done: false,
          // the triangles carry the samples' own colours, unlit like the Gaussians; pushed back a little so their edges read
          tri: new T.Mesh(geo, new T.ShaderMaterial({ side: T.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
            vertexShader: 'attribute vec3 aCol; varying vec3 vC; void main(){ vC = aCol; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
            fragmentShader: 'varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0); }' })),
          // every edge faintly, the ones just stitched brightly, and the window itself: its four samples and a dot
          wire: new T.LineSegments(wgeo, new T.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.1 })),
          fresh: new T.LineSegments(wgeo.clone(), new T.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.6 })),
          win: new T.LineLoop(kgeo, new T.LineBasicMaterial({ color: C.text, depthTest: false })),
          dot: new T.Points(kgeo, new T.PointsMaterial({ color: C.text, size: 5, sizeAttenuation: false, depthTest: false })) };
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].forEach(function (o) { o.frustumCulled = false; scene.add(o); });
        mesh.win.renderOrder = mesh.dot.renderOrder = 6; root.classList.add('mf-mesh');
        stepMesh(0);
      }
      function stepMesh(dt) {
        mesh.t += dt;
        var f = Math.min(1, mesh.t / MESH_T), q = Math.min(mesh.Q, Math.floor(f * f * mesh.Q)), left = mesh.V, m, i, k;
        for (i = 0; i < mesh.parts.length; i++) {
          m = mesh.parts[i]; k = q - m.q0;
          if (k >= m.qa * m.qb) { left -= m.na * m.nb; continue; }
          if (k >= 0) {                                   // the window is on this chart, at column a of row b
            var a = k % m.qa, b = Math.floor(k / m.qa), W = mesh.win.geometry.attributes.position, P = mesh.pos;
            left -= b * m.na + a;
            [m.off + b * m.na + a, m.off + b * m.na + (a + 1) % m.na, m.off + ((b + 1) % m.nb) * m.na + (a + 1) % m.na, m.off + ((b + 1) % m.nb) * m.na + a]
              .forEach(function (v, j) { W.setXYZ(j, P[v * 3], P[v * 3 + 1], P[v * 3 + 2]); });
            W.needsUpdate = true;
          }
          break;
        }
        var band = Math.min(q, Math.ceil(mesh.Q * 0.035));
        mesh.tri.geometry.setDrawRange(0, q * 6); mesh.wire.geometry.setDrawRange(0, q * 6); mesh.fresh.geometry.setDrawRange((q - band) * 6, band * 6);
        mesh.win.visible = mesh.dot.visible = mesh.fresh.visible = q < mesh.Q; field.setCount(left);
        rdG.set(left.toLocaleString('en-US')); rdM.set((q * 2).toLocaleString('en-US') + ' / ' + (mesh.Q * 2).toLocaleString('en-US'));
        if (q === mesh.Q && !mesh.done) { mesh.done = true; if (!orbit.touched) orbit.to(view('close')); }   // finished: go and look at the triangles
      }

      /* ---------- control nets ---------- */
      function every(n, s) { var a = [], i; for (i = 0; i < n; i++) if (i % s === 0 || i === n - 1) a.push(i); return a; }
      /* the rows c.pj and columns c.pi of each net (all of them unless the net is dense), and the points where they cross */
      function updateNets() {
        var A = netL.arr, n = 0, q = 0;
        function seg(P, o, o2) { A[n++] = P[o]; A[n++] = P[o + 1]; A[n++] = P[o + 2]; A[n++] = P[o2]; A[n++] = P[o2 + 1]; A[n++] = P[o2 + 2]; }
        charts.forEach(function (c) {
          var p = c.patch, P = p.P, nu = p.nu, nv = p.nv, I = c.pi, J = c.pj, a, b, i, j, o;
          c.ptOff = q / 3;
          for (b = 0; b < J.length; b++) for (a = 0; a < I.length; a++) { o = (J[b] * nu + I[a]) * 3; ptPos[q++] = P[o]; ptPos[q++] = P[o + 1]; ptPos[q++] = P[o + 2]; }
          for (b = 0; b < J.length; b++) for (j = J[b], i = 0; i < nu; i++) if (i < nu - 1 || p.periodicU) seg(P, (j * nu + i) * 3, (j * nu + (i + 1) % nu) * 3);
          for (a = 0; a < I.length; a++) for (i = I[a], j = 0; j < nv; j++) if (j < nv - 1 || p.periodicV) seg(P, (j * nu + i) * 3, (((j + 1) % nv) * nu + i) * 3);
        });
        netL.set(n / 3); pts.geometry.attributes.position.needsUpdate = true;
      }
      function showNets() {
        if (!netL) return;
        netL.obj.visible = pts.visible = st.nets; rigL.obj.visible = st.rig; tgN.set(st.nets); tgR.set(st.rig);
        /* a thinned net lies on the surface it interpolates, so it would flicker in and out of the Gaussians:
           while it shows, the Gaussians' depth is written one sigma back (the pre-pass variant) */
        var k = field.mesh.userData.ktSplat;
        if (k) { k.front = k.front || k.core; k.core = st.nets && shownCP < totalCP ? k.depth : k.front; }
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
        if (touch) return G.nearest(cam, R.canvas, ptPos, shownCP, x, y, KT.HIT);
        setNdc({ clientX: x, clientY: y }); ray.params.Points.threshold = orbit.cur.r * 0.012;
        var h = ray.intersectObject(pts, false)[0];
        return h ? h.index : -1;
      }
      function pickPoint(e) {
        var gi = findPoint(e.clientX, e.clientY, e.pointerType === 'touch'), c = null;
        if (gi < 0) return false;
        charts.forEach(function (k) { if (gi >= k.ptOff && gi < k.ptOff + k.pi.length * k.pj.length) c = k; });
        /* a sparse net: that one control point. A dense net: a soft brush, the points within 3 drawn rows of it */
        var p = c.patch, P = p.P, l = gi - c.ptOff, i0 = c.pi[l % c.pi.length], j0 = c.pj[Math.floor(l / c.pi.length)];
        var rad = c.ns > 1 ? 3 * c.ns : 0, idx = [], w = [], was = [], i, j, d, o = (j0 * p.nu + i0) * 3;
        for (j = Math.max(0, j0 - rad); j <= Math.min(p.nv - 1, j0 + rad); j++) for (i = Math.max(0, i0 - rad); i <= Math.min(p.nu - 1, i0 + rad); i++) {
          d = Math.hypot(i - i0, j - j0) / (rad + 1); if (d >= 1) continue;
          d = Math.cos(d * Math.PI / 2); idx.push((j * p.nu + i) * 3); w.push(d * d); was.push(P[idx[idx.length - 1]], P[idx[idx.length - 1] + 1], P[idx[idx.length - 1] + 2]);
        }
        dragCP = { c: c, id: e.pointerId, idx: idx, w: w, was: was, at: [P[o], P[o + 1], P[o + 2]] };
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), hitP.set(P[o], P[o + 1], P[o + 2]));
        R.canvas.setPointerCapture(e.pointerId);
        return true;
      }
      R.canvas.addEventListener('pointermove', function (e) {
        if (!dragCP || e.pointerId !== dragCP.id) return;
        setNdc(e);
        if (ray.ray.intersectPlane(plane, hitP)) {
          st.touchedEdit = true;           // only a drag that actually moves a point stops the scene's own edit
          moveCP(hitP.x - dragCP.at[0], hitP.y - dragCP.at[1], hitP.z - dragCP.at[2]);
        }
      });
      function moveCP(dx, dy, dz) {
        var D = dragCP, P = D.c.patch.P, n, o;
        for (n = 0; n < D.idx.length; n++) { o = D.idx[n]; P[o] = D.was[n * 3] + dx * D.w[n]; P[o + 1] = D.was[n * 3 + 1] + dy * D.w[n]; P[o + 2] = D.was[n * 3 + 2] + dz * D.w[n]; }
        coarse(D.c); updateNets(); need = true;
      }
      function endCP(e) { if (dragCP && e.pointerId === dragCP.id) dragCP = null; }
      R.canvas.addEventListener('pointerup', endCP); R.canvas.addEventListener('pointercancel', endCP);

      /* ---------- scenes ---------- */
      /* a material is shown from the camera it names (a see-through one from where something lies behind it) */
      function matView() {
        var m = st.sc.paint && def.materials && def.materials.filter(function (k) { return k.v === st.mat; })[0];
        return m && m.view && def.views[m.view] ? m.view : st.sc.cam || 'wide';
      }
      function pickMat(v) { st.mat = v; st.paint = 0; if (segM) segM.set(v); if (!orbit.touched) orbit.to(view(matView())); }
      function paintBadge() {
        badge.innerHTML = 'view-adaptive sampling <b class="' + (st.adaptive ? 'on' : 'off') + '">' + (st.adaptive ? 'ON' : 'OFF') + '</b>';
      }
      function paintCap() {
        var s = KT.scenes[st.scn];
        cap.innerHTML = '<b>' + s.label + '</b> · ' + s.suite + (s.note != null ? s.note && ' · ' + s.note : st.scn === 'studio' ? '' : ' · procedural stand-in') + '<br>' + st.sc.cap;
        badge.style.display = st.sc.ab ? '' : 'none';
      }
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        if (id === 'hero' && env.opts.scn && KT.scenes[env.opts.scn]) sc = Object.assign({}, sc, { scn: env.opts.scn });
        if (!KT.scenes[sc.scn]) sc = Object.assign({}, sc, { scn: 'barn' });
        root.classList.toggle('mf-noatlas', !sc.atlas); root.classList.toggle('mf-tools', !!sc.tools); root.classList.toggle('mf-paint', !!sc.paint);
        st.scene = id; st.sc = sc; st.t = 0; st.tourI = 0; st.tourT = 0; st.abT = 0; st.touchedAB = false; st.touchedEdit = false;
        st.view = sc.view; segV.set(sc.view); showView(); st.nets = !!sc.nets; st.rig = !!sc.rig;
        st.adaptive = true; tgA.set(true); paintBadge();
        field.uniforms.uRim.value = sc.rim || 0.3;
        var fresh = st.scn !== sc.scn;
        loadScene(sc.scn); showNets(); paintCap(); syncMesh(); st.paint = 0;
        hint.textContent = sc.hint || '';
        orbit.touched = false;
        orbit.to(view(matView()), fresh);
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
        if (sc.breathe && !st.touchedEdit && !dragCP && (def.animate || def.edit)) {
          // the scene's own edit: a life of its own where the scene has one (and a repaint, wiped in), else one move eased in and out
          if (sc.paint && def.animate) st.paint = Math.min(1, st.paint + dt / 3.5);
          (def.animate ? def.animate(st.t, st.paint, st.mat) : def.edit(0.5 - 0.5 * Math.cos(st.t * 1.15))).forEach(coarse);
          if (st.nets) updateNets();
          need = true;
        }
        var home = def.views[matView()];
        if (!orbit.touched && !sc.tour && (sc.cam || 'wide') === 'wide') orbit.goal.th = home.th + 0.14 * Math.sin(t * 0.16);
        orbit.update(dt, sc.tour ? 1.5 : 3);
        var e = cam.matrixWorld.elements, key = (e[12] + e[13] * 3 + e[14] * 7 + e[8] * 11 + e[9] * 13).toFixed(3);
        if (key !== camKey && t - lastAlloc > 0.09) { camKey = key; need = true; }
        if (mesh) { need = false; stepMesh(dt); }
        else if (need) { need = false; lastAlloc = t; resample(); }
        var ud = field.uniforms.uDepth.value, kz = 1 - Math.exp(-5 * dt);
        ud.x += (zr[0] - ud.x) * kz; ud.y += (zr[1] - ud.y) * kz;
        if (st.view === 'split') { roll = wrap(roll + ROLL * dt, 1); field.uniforms.uSplit.value.y = roll; placeSplit(); }
        G.draw(R, scene, cam);
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
          '.kt-sim-manifold.kt-compact .kt-atlas,.kt-sim-manifold.mf-noatlas .kt-atlas{display:none}' +
          // in the deck the steps drive the scene: no picker, and the toggles only on a step that asks for them (tools: 1)
          '.kt-sim-manifold.kt-embed .mf-pick,.kt-sim-manifold.kt-embed:not(.mf-tools) .mf-tog{display:none}' +
          '.kt-sim-manifold:not(.mf-mesh) .mf-tri,.kt-sim-manifold.mf-mesh .mf-samp,.kt-sim-manifold:not(.mf-paint) .mf-mat{display:none}' +

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
        load: function (key) { st.sc = Object.assign({}, st.sc, { scn: key }); st.tourI = 0; st.tourT = 0; loadScene(key); syncMesh(); orbit.touched = false; orbit.to(view('wide'), true); paintCap(); },
        setMaterial: pickMat,
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
