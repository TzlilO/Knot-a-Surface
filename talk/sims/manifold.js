/* Knot-a-Surface talk — module `manifold`: "A manifold of splines" (the full-power demo).
   A whole scene as an atlas of B-spline charts — a spline manifold. Every Gaussian on screen is
   derived from one of the charts; a FIXED Gaussian budget is re-sampled over the (u,v) domains
   as the camera moves (view-adaptive sampling), and editing control points moves thousands of
   Gaussians at once. Scenes come from talk/scenes.js: the real objects (`scan`, `scan<N>`: DTU scans
   fitted offline). The hand-modelled stand-ins (barn, truck, bust, garden, studio) are no longer offered
   here; the barn is only what a page without the scan data falls back to. Any scene can be read as appearance, depth, normals,
   single Gaussians or chart identity — or all five at once, in a rolling split screen.
   Tool: WebGL (three.js) instanced planoids + a 2D-canvas (u,v) atlas.
   Standalone: talk/sims/manifold.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js, scenes.js; the scene
   `mesh` also on meshlist.js (the list's mesher and the audit) and meshio.js (the export), loaded before this file. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var PER_CP = 16;                  // 3 position + 12 SH1 colour + 1 opacity per control point
  var CAP0 = 320000, FLOOR = 0.10;  // the most Gaussians a mount holds (opts.cap may lower it); FLOOR = share of the budget kept as view-independent coverage
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
    /* the same Gaussians meshed twice: as an (emulated) list, by neighbour search, and on the charts' lattices, by a 2 x 2 window */
    mesh:   { scn: 'scan', view: 'gauss', cam: 'wide', mesh: 1, cap: 'tessellation · the same Gaussians as a list (emulated) and on the lattice' },
    edit:   { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, tools: 1, cap: 'editable by construction · move control points, the Gaussians follow',
              hint: 'turn on nets to drag a control point' },
    paint:  { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, paint: 1, cap: 'material edit · the same net, new control colours' },
    /* the tool layer of talk/hands.js: work on the object with the pointer, and on its atlas */
    hands:  { scn: 'scan', view: 'splats', cam: 'near', tools: 1, hands: 1, cap: 'hands · grab, pinch, pull, push or hold the surface: every action moves control points',
              hint: 'drag on the object to act, beside it to orbit' },
    /* chrome-free, for pages that embed the scene as a backdrop (mount with { bare: true }) */
    hero:   { scn: 'barn', view: 'splats', tour: 1, cap: '' },
    /* the landing page's loop (tease(), below): three readings side by side, then the picture, then the materials */
    teaser: { scn: 'scan', view: 'splats', teaser: 1, cap: '' },
    /* the talk's cover (coverLoop, below): the owl and the rabbit side by side, without chrome */
    cover:  { scn: 'duo', view: 'splats', cam: 'wide', breathe: 1, cover: 1, bare: 1, cap: '' }
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
      { id: 'edit', label: 'Edit' }, { id: 'paint', label: 'Materials' }, { id: 'hands', label: 'Hands' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 160);
      var R = G.renderer(root, cam, env.opts.clear), scene = new T.Scene();
      var CAP = Math.min(+env.opts.cap || CAP0, CAP0);       // this mount's Gaussian capacity: a page that mounts several can ask for less
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      var st = { scene: 'scene', sc: SCENES.scene, scn: '', view: 'splats', adaptive: true, nets: false, rig: false, t: 0,
                 budget: Math.min(+env.opts.budget || (small ? 30000 : KT.dtu ? 160000 : 90000), (Math.min(+env.opts.cap || CAP0, CAP0)) - 20000),   // the scanned object has detail to show for a larger budget
                 k: 0.72, cull: true, ground: true, drawn: 0, paint: 0, mat: 'glass', from: 'stone', tourI: 0, tourT: 0, abT: 0, touchedAB: false, touchedEdit: false,
                 target: 'object', fromT: 'object' };     // where the material goes (the scene's `targets`: object, owl, selection), and where the one before it went
      var segM = null, segT = null, knobs = null;   // the material and target pickers of a scene that has materials, and the chosen one's knobs (on the `paint` scene)
      var def = null, charts = [], totalCP = 0, shownCP = 0, netL = null, pts = null, ptPos = null, rigL = null;
      var need = true, lastAlloc = 0, camKey = '', fit = 1;
      /* a named camera of the current scene, backed off on narrow stages so the subject stays in frame */
      function view(name) { var v = Object.assign({}, def.views[name] || def.views.wide); v.r *= fit; return v; }

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
          if (hands) { hands.reset(); showHands(); }
        } });
      segS.el.classList.add('mf-pick');
      var slB = ui.slider(panel, { label: 'budget', min: 6000, max: CAP - 20000, step: 2000, value: Math.min(st.budget, CAP - 20000),
        fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, onInput: function (v) { st.budget = v; need = true; remesh(); } });
      var segV = ui.seg(panel, { value: 'splats', options: MODES.concat([{ v: 'split', label: 'all' }]),
        onChange: function (v) { st.view = v; showView(); } });
      var rowT = ui.el('div', 'kt-row mf-tog'); panel.appendChild(rowT);
      var tgA = ui.toggle(rowT, { label: 'adaptive', value: true, onChange: function (v) { st.adaptive = v; st.touchedAB = true; need = true; paintBadge(); } });
      var tgN = ui.toggle(rowT, { label: 'nets', value: false, onChange: function (v) { st.nets = v; showNets(); } });
      var tgR = ui.toggle(rowT, { label: 'cameras', value: false, onChange: function (v) { st.rig = v; showNets(); } });
      ui.toggle(rowT, { label: 'no back faces', value: true, onChange: function (v) { st.cull = v; need = true; } });
      /* the ground (a chart the scene marks `ground`) can be put away: the object alone, with the whole budget */
      ui.toggle(rowT, { label: 'ground', value: true, onChange: function (v) { st.ground = v; charts.forEach(function (c) { c.off = !!c.ground && !v; }); need = true; remesh(); } });
      var rdG = ui.readout(panel, { label: 'Gaussians drawn', tone: 'splat' });
      var rdV = ui.readout(panel, { label: 'budget in view', tone: 'ok' });
      rdV.el.classList.add('mf-samp');   // the view-adaptive sampler's row: not in the mesh scene (whose lattice the budget sizes)
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      var rdM = ui.readout(panel, { label: 'triangles stitched', tone: 'surf' }); rdM.el.classList.add('mf-tri');
      var atlas = ui.el('div', 'kt-atlas'); root.appendChild(atlas);
      /* split screen: one divider and one label per band, laid over the canvas and moved every frame */
      var split = ui.el('div', 'kt-split'), bands = MODES.concat(MODES[0]).map(function (m) {      // six: the most bands a split shows
        var b = { a: ui.el('i'), b: ui.el('i'), lab: ui.el('span', '', m.name) };   // a divider can wrap: two copies
        split.appendChild(b.a); split.appendChild(b.b); split.appendChild(b.lab);
        return b;
      });
      root.appendChild(split);
      var roll = 0, zr = [1, 10], size = { w: 1, h: 1 }, shown = MODES;
      /* which readings the split screen shows, left to right: all five, or up to three chosen ones ([{ v, name }]),
         which then repeat across `rep` times as many bands */
      function setBands(list, rep) {
        shown = list ? [].concat.apply([], Array(rep || 1).fill(list)) : MODES;
        var ix = (list || []).map(function (m) { return MODES.map(function (q) { return q.v; }).indexOf(m.v); });
        field.uniforms.uBand.value.set(list ? ix[0] : -1, ix[1] || 0, ix[2] || 0, list ? list.length : 1);
        bands.forEach(function (b, j) { var on = j < shown.length; b.a.style.display = b.b.style.display = b.lab.style.display = on ? '' : 'none'; if (on) b.lab.textContent = shown[j].name; });
      }
      function showView() {
        var i = MODES.map(function (m) { return m.v; }).indexOf(st.view), all = i < 0;
        field.uniforms.uMode.value = all ? 0 : i;
        field.uniforms.uSplit.value.set(all ? shown.length : 0, roll, SLANT);
        root.classList.toggle('kt-splitting', all);
      }
      function placeSplit() {
        var w = size.w, h = size.h, n = shown.length, tight = w / n < 130;
        bands.slice(0, n).forEach(function (b, j) {
          var yl = h - (st.sc.teaser ? (small ? 0.34 : 0.2) * h : 26) - (tight ? j % 3 * 30 : 0);     // narrow bands: stagger the labels so neighbours never touch (the teaser's page has a strip along the bottom)
          var x = w * wrap(j / n + roll, 1), sk = ' skewX(' + (-Math.atan(SLANT)) + 'rad)';
          b.a.style.transform = 'translateX(' + x + 'px)' + sk; b.b.style.transform = 'translateX(' + (x + w) + 'px)' + sk;
          var xc = wrap(w * ((j + 0.5) / n + roll) - SLANT * yl, w);
          b.lab.style.transform = 'translate(' + xc + 'px,' + yl + 'px) translate(-50%,-50%)';
          b.lab.style.opacity = KT.clamp(Math.min(xc, w - xc) / 44 - 0.4, 0, 1);
        });
      }

      var hands = null;                // the tool layer (talk/hands.js), where that file is loaded
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 1.4, maxRadius: 46,
        pick: function (e) { return pickPoint(e) || (!!hands && hands.claim(e)); },
        hit: function (x, y) { return findPoint(x, y, true) >= 0 || (!!hands && hands.hit(x, y)); },
        release: function () {
          if (hands) hands.undo();
          if (!dragCP) return;
          moveCP(0, 0, 0); dragCP = null;
        } });

      /* ---------- load a scene: charts, nets, atlas tiles, capture rig ---------- */
      function loadScene(key) {
        if (st.scn === key) { charts.forEach(function (c) { c.patch.P.set(c.base); if (c.baseC) c.patch.C.set(c.baseC); c.alpha = null; coarse(c); }); updateNets(); need = true; return; }
        st.scn = key; def = KT.scenes[key].build(); charts = def.charts; segS.set(key);
        charts.forEach(function (c) { c.off = !!c.ground && !st.ground; });
        field.uniforms.uFlat.value = def.photo || KT.scenes[key].photo ? 1 : 0;   // colours from photographs are already lit
        if (segM) { segM.el.remove(); segM = null; }
        if (segT) { segT.el.remove(); segT = null; }
        if (def.materials) {
          if (!def.materials.some(function (m) { return m.v === st.mat; })) st.mat = def.materials[0].v;
          segM = ui.seg(panel, { value: st.mat, options: def.materials, onChange: pickMat });
          segM.el.classList.add('mf-mat');
          // where it goes; a selection needs the tool layer (talk/hands.js)
          var tg = (def.targets || []).filter(function (t) { return t.v !== 'selection' || hands; });
          if (tg.length) {
            if (!tg.some(function (t) { return t.v === st.target; })) st.target = tg[0].v;
            segT = ui.seg(panel, { label: 'on', value: st.target, options: tg, onChange: pickTarget });
            segT.el.classList.add('mf-target');
          }
        }
        showKnobs();
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
        charts.forEach(function (c) { if (!c.off) totA += c.area; });
        charts.forEach(function (c) {
          var n = c.gu * c.gv, i, o, w; c.w = c.w || new Float32Array(n); c.vis = c.vis || new Uint8Array(n);
          if (c.off) { c.w.fill(0); c.vis.fill(0); return; }
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
          c.count = 0; if (c.off) return;
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
      /* per Gaussian of the last frame: its chart and the control point nearest to it (for picking, and for what
         a chart stores per control point: opacity, the tool layer's selection and holds) */
      var instC = new Uint8Array(CAP), instK = new Int32Array(CAP), keptL = new Int32Array(CAP);
      function emit() {
        allocate();
        var n = 0, eye = { x: cam.position.x, y: cam.position.y, z: cam.position.z, flip: 1, min: -0.12 }, thin = false, track = !!st.sc.hands || selecting();
        charts.forEach(function (c, ci) {
          // the selection and the holds are tinted on the `hands` scene; while a material goes on the selection, the material shows it
          var rgb = G.rgb(c.hex), n0 = n, i, AL = c.alpha, SE = !!st.sc.hands && c.sel, PN = !!st.sc.hands && c.pin, p = c.patch, K = field.color;
          if (c.off) { c.count = 0; c.lab.textContent = 'put away'; return; }
          eye.flip = c.flip;
          c.tiles.forEach(function (tl) {
            if (tl.na * tl.nb > CAP) return;
            NB.patchGrid(c.patch, tl.us, tl.vs, sg);
            var t0 = n, k, q, l, o, s;
            // a see-through chart keeps its far side, and each Gaussian takes the opacity of the control point nearest to it
            n = G.writePlanoids(field, n, sg, tl.na, tl.nb, tl.du, tl.dv, st.k, null, st.cull && c.closed && !AL ? eye : null, keptL);
            if (!AL) field.alpha.fill(1, t0, n);
            if (!AL && !track) return;
            for (k = t0; k < n; k++) {
              l = keptL[k - t0]; q = Math.round(tl.vs[(l / tl.na) | 0] * (p.nv - 1)) * p.nu + Math.round(tl.us[l % tl.na] * (p.nu - 1));
              instC[k] = ci; instK[k] = q;
              if (AL) field.alpha[k] = AL[q];
              if (SE && SE[q] > 0.5) { o = k * 3; K[o] = K[o] * 0.5 + 0.155; K[o + 1] = K[o + 1] * 0.5 + 0.38; K[o + 2] = K[o + 2] * 0.5 + 0.485; }   // selected: towards cyan
              if (PN && (s = PN[q] * 0.5) > 0) { o = k * 3; K[o] += (1.0 - K[o]) * s; K[o + 1] += (0.65 - K[o + 1]) * s; K[o + 2] += (0.15 - K[o + 2]) * s; }   // held: towards amber
            }
            if (AL) thin = true;
          });
          c.count = n - n0;
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          c.lab.textContent = c.count.toLocaleString('en-US') + ' G';
          drawTile(c);
        });
        field.mesh.userData.ktSplat.thin = thin; st.drawn = n;
        return n;
      }
      /* what is under the pointer: the Gaussian nearest to the eye among those within a few pixels of it
         -> { c: its chart, i, j: the control point nearest to it, pos, n: the unit normal there, pointing out } */
      function pickSurface(x, y) {
        if (!st.sc.hands && !selecting()) return null;
        var r = R.canvas.getBoundingClientRect(), px = (x - r.left) / r.width * 2 - 1, py = -((y - r.top) / r.height * 2 - 1), rx = 14 / r.width, ry = 14 / r.height;
        cam.updateMatrixWorld(); M.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
        var e = M.elements, P = field.center, A = field.alpha, best = -1, bw = 1e9, k, o, w;
        for (k = 0; k < st.drawn; k++) {
          o = k * 3; w = e[3] * P[o] + e[7] * P[o + 1] + e[11] * P[o + 2] + e[15];
          if (w <= 0.05 || w >= bw || A[k] < 0.05) continue;
          if (Math.abs((e[0] * P[o] + e[4] * P[o + 1] + e[8] * P[o + 2] + e[12]) / w - px) > rx || Math.abs((e[1] * P[o] + e[5] * P[o + 1] + e[9] * P[o + 2] + e[13]) / w - py) > ry) continue;
          best = k; bw = w;
        }
        if (best < 0) return null;
        o = best * 3;
        var c = charts[instC[best]], U = field.axU, V = field.axV, nx = U[o + 1] * V[o + 2] - U[o + 2] * V[o + 1], ny = U[o + 2] * V[o] - U[o] * V[o + 2], nz = U[o] * V[o + 1] - U[o + 1] * V[o];
        var s = (c.closed ? c.flip : (nx * (cam.position.x - P[o]) + ny * (cam.position.y - P[o + 1]) + nz * (cam.position.z - P[o + 2]) < 0 ? -1 : 1)) / (Math.hypot(nx, ny, nz) || 1);
        return { c: c, i: instK[best] % c.patch.nu, j: (instK[best] / c.patch.nu) | 0, pos: [P[o], P[o + 1], P[o + 2]], n: [nx * s, ny * s, nz * s] };
      }
      if (KT.hands) hands = KT.hands.create({ root: root, ui: ui, canvas: R.canvas, camera: cam, charts: function () { return charts; }, pick: pickSurface,
        changed: function (c, shape) { if (shape) { coarse(c); if (st.nets) updateNets(); } need = true; } });
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

      /* ---------- scene `mesh`: the same Gaussians meshed twice, at the same budget ----------
         THE LATTICE: every chart is sampled on one even lattice, as many points in all as the budget (shared over the
         charts by area x gain, the sampler's own share, and laid so that the cells are square on the surface), and the
         Gaussians sit on its points. A 2 x 2 window running over it turns each stop into two triangles. A Gaussian goes
         when its point has been stitched, so the instances are written in the reverse of the window's order: what is
         left to stitch is always the first part of the buffer.
         A LIST (talk/meshlist.js): the same number of Gaussians as an EMULATED unstructured list, meshed the way a list
         has to be: a hash grid and a k-nearest search for every Gaussian, a plane through its neighbours, a local
         triangulation in that plane, and a vote. The work is spread over the frames (SLICE ms each) and timed inside
         them. Its triangles then appear in the order the list made them, which is no order at all; bridges and
         non-manifold edges in red, the rims of its holes in faint red.
         The board compares the two, graded by the same audit; every number on it is measured on the page. The step plays
         the list first, then the lattice; picking one (`mesh from`) stops that. A new budget, or the ground put away,
         rebuilds both once the slider has rested for REST seconds. The stitched lattice downloads as PLY, GLB, OBJ or
         STL (talk/meshio.js), in metres where the scene has a physical scale. ---------- */
      var ML = KT.meshList, IO = KT.meshio, mesh = null;
      var MESH_T = 12, LIST_T = 6, HOLD_T = 2.5, REST = 0.45, SLICE = 12;    // seconds; SLICE: ms of the list's work per frame
      var board = ui.el('div', 'mf-board'); root.appendChild(board);
      var segH = null, segF = null, ioNote = null;
      if (ML) {
        segH = ui.seg(panel, { label: 'mesh from', value: 'list', options: [{ v: 'list', label: 'a list' }, { v: 'lattice', label: 'the lattice' }],
          onChange: function (v) { if (mesh) { mesh.auto = false; showPhase(v); } } });
        segH.el.classList.add('mf-tri');
      }
      if (IO) {
        var io = ui.el('div', 'mf-tri mf-io'); panel.appendChild(io);
        segF = ui.seg(io, { label: 'export', value: 'glb', options: IO.formats.map(function (f) { return { v: f.v, label: f.v }; }) });
        ui.button(io, { label: 'download the lattice mesh', kind: 'go', onClick: download });
        ioNote = ui.el('div', 'mf-io-note'); io.appendChild(ioNote);
      }
      function remesh() { if (mesh) mesh.due = mesh.age + REST; }    // the budget moved: rebuild once it rests
      function syncMesh() { if (st.sc.mesh) buildMesh(); else dropMesh(); }
      function dropMesh() {
        if (!mesh) return;
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].concat(mesh.lobj || []).forEach(function (o) { scene.remove(o); o.material.dispose(); o.geometry.dispose(); });
        mesh = null; root.classList.remove('mf-mesh'); need = true;
      }
      /* how many lattice points each chart gets (the budget, shared by area x gain like the sampler's share) and how they
         are laid: na / nb = |S_u| / |S_v across S_u|, averaged over the chart, so that the cells are square on the surface */
      function sizeLattices() {
        var on = charts.filter(function (c) { return !c.off; }), W = 0, N = Math.min(st.budget, CAP);
        on.forEach(function (c) { W += c.area * c.gain; });
        return on.map(function (c) {
          var n = N * c.area * c.gain / W, lu = 0, lv = 0, i;
          for (i = 0; i < c.carea.length; i++) { lu += c.clu[i] * c.carea[i]; lv += c.clv[i] * c.carea[i]; }
          var nb = Math.max(c.minB, 2, Math.round(Math.sqrt(n * lv / (lu || 1e-12)))), na = Math.max(c.minA, 2, Math.round(n / nb));
          return { c: c, patch: c.patch, chart: charts.indexOf(c), na: na, nb: nb };
        });
      }
      function evalLattice(m, back) {        // part m's even lattice into sg; back: in the reverse order
        var us = new Float32Array(m.na), vs = new Float32Array(m.nb), du = new Float32Array(m.na).fill(1 / m.na), dv = new Float32Array(m.nb).fill(1 / m.nb), a, b;
        for (a = 0; a < m.na; a++) us[back ? m.na - 1 - a : a] = (a + 0.5) / m.na;
        for (b = 0; b < m.nb; b++) vs[back ? m.nb - 1 - b : b] = (b + 0.5) / m.nb;
        NB.patchGrid(m.c.patch, us, vs, sg);
        return { du: du, dv: dv };
      }
      /* triangles in the samples' own colours, unlit like the Gaussians; pushed back a little so that edges drawn on them read */
      function surfaceMesh(pos, col, idx) {
        var geo = new T.BufferGeometry(); geo.setAttribute('position', new T.BufferAttribute(pos, 3)); geo.setAttribute('aCol', new T.BufferAttribute(col, 3));
        geo.setIndex(new T.BufferAttribute(idx, 1));
        return new T.Mesh(geo, new T.ShaderMaterial({ side: T.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
          vertexShader: 'attribute vec3 aCol; varying vec3 vC; void main(){ vC = aCol; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
          fragmentShader: 'varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0); }' }));
      }
      function edgeLines(E, P, colour, opacity) {    // edges as (a, b, first triangle) -> line segments, in that order
        var n = E.length / 3, arr = new Float32Array(n * 6), k, q;
        for (k = 0; k < n; k++) for (q = 0; q < 3; q++) { arr[k * 6 + q] = P[E[k * 3] * 3 + q]; arr[k * 6 + 3 + q] = P[E[k * 3 + 1] * 3 + q]; }
        var g = new T.BufferGeometry(); g.setAttribute('position', new T.BufferAttribute(arr, 3));
        var o = new T.LineSegments(g, new T.LineBasicMaterial({ color: colour, transparent: true, opacity: opacity })); o.renderOrder = 4;
        return o;
      }
      function upTo(E, q) {                         // how many of the edges E come with the first q triangles
        var lo = 0, hi = E.length / 3, m;
        while (lo < hi) { m = (lo + hi) >> 1; if (E[m * 3 + 2] < q) lo = m + 1; else hi = m; }
        return lo;
      }
      function buildMesh() {
        var was = mesh && mesh.due ? { auto: mesh.auto, phase: mesh.phase } : null;    // a rebuild for a new budget keeps what was picked
        dropMesh();
        if (!ML) return;
        var parts = sizeLattices().sort(function (a, b) { return a.na * a.nb - b.na * b.nb; });   // the window starts on the smallest chart
        if (!parts.length) return;
        var lat = ML.lattice(parts), V = lat.V, w = def.views.wide;
        var pos = new Float32Array(V * 3), col = new Float32Array(V * 3), nrm = new Float32Array(V * 3);
        var eye = [w.tx + w.r * Math.sin(w.ph) * Math.sin(w.th), w.ty + w.r * Math.cos(w.ph), w.tz + w.r * Math.sin(w.ph) * Math.cos(w.th)];
        parts.forEach(function (m) {
          var k = m.na * m.nb, i, o, s = 0;
          evalLattice(m, false); pos.set(sg.pos.subarray(0, k * 3), m.off * 3); col.set(sg.col.subarray(0, k * 3), m.off * 3);
          // the surface's own normal, S_u x S_v: out of a closed chart, towards the scene's camera on an open one
          for (i = 0; i < k; i++) {
            o = i * 3; var ax = sg.su[o], ay = sg.su[o + 1], az = sg.su[o + 2], bx = sg.sv[o], by = sg.sv[o + 1], bz = sg.sv[o + 2];
            var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, l = Math.hypot(nx, ny, nz) || 1;
            nrm[(m.off + i) * 3] = nx / l; nrm[(m.off + i) * 3 + 1] = ny / l; nrm[(m.off + i) * 3 + 2] = nz / l;
            s += nx * (eye[0] - sg.pos[o]) + ny * (eye[1] - sg.pos[o + 1]) + nz * (eye[2] - sg.pos[o + 2]);
          }
          m.sign = m.c.closed ? m.c.flip : s < 0 ? -1 : 1;
          if (m.sign < 0) for (i = m.off * 3; i < (m.off + k) * 3; i++) nrm[i] = -nrm[i];
        });
        var tri = surfaceMesh(pos, col, lat.tri), wgeo = new T.BufferGeometry(), kgeo = new T.BufferGeometry();
        wgeo.setAttribute('position', tri.geometry.attributes.position); wgeo.setIndex(new T.BufferAttribute(lat.edge, 1));
        kgeo.setAttribute('position', new T.BufferAttribute(new Float32Array(12), 3));
        mesh = { parts: parts, lat: lat, V: V, Q: lat.Q, pos: pos, col: col, nrm: nrm, t: 0, rt: 0, hold: 0, age: 0, due: 0, done: false, latA: null,
          auto: was ? was.auto : true, phase: 'lattice', hex: charts.map(function (c) { return G.rgb(c.hex); }), tri: tri,
          // every edge faintly, the ones just stitched brightly, and the window itself: its four samples and a dot
          wire: new T.LineSegments(wgeo, new T.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.1 })),
          fresh: new T.LineSegments(wgeo.clone(), new T.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.6 })),
          win: new T.LineLoop(kgeo, new T.LineBasicMaterial({ color: C.text, depthTest: false })),
          dot: new T.Points(kgeo, new T.PointsMaterial({ color: C.text, size: 5, sizeAttenuation: false, depthTest: false })) };
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].forEach(function (o) { o.frustumCulled = false; scene.add(o); });
        mesh.win.renderOrder = mesh.dot.renderOrder = 6; root.classList.add('mf-mesh');
        // the list (the same samples as a 3DGS list would hold them, emulated over a few frames, then meshed) and the lattice's audit
        mesh.L = null; mesh.job = null; mesh.ejob = ML.emulator(parts, 7); mesh.ajob = ML.auditor(lat.tri, lat.nTri, lat);
        showPhase(was ? was.phase : 'list');
        paintIO();
      }
      function writeLattice() {
        var n = 0;
        mesh.parts.slice().reverse().forEach(function (m) {
          var d = evalLattice(m, true), n0 = n, rgb = G.rgb(m.c.hex), i;
          n = G.writePlanoids(field, n, sg, m.na, m.nb, d.du, d.dv, st.k, null, null);
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          m.c.lab.textContent = (n - n0).toLocaleString('en-US') + ' G';
        });
        field.alpha.fill(1, 0, n); field.mesh.userData.ktSplat.thin = false; field.commit(n); field.setCount(n);
      }
      function writeList() {                         // reversed as well: the list entries already meshed are the end of the buffer
        var L = mesh.L, n = L.n, k, q, s, d, rgb;
        for (k = 0; k < n; k++) {
          s = (n - 1 - k) * 3; d = k * 3; rgb = mesh.hex[L.chart[n - 1 - k]];
          for (q = 0; q < 3; q++) { field.center[d + q] = L.pos[s + q]; field.axU[d + q] = L.axU[s + q]; field.axV[d + q] = L.axV[s + q]; field.color[d + q] = L.col[s + q]; field.aux[d + q] = rgb[q]; }
        }
        field.alpha.fill(1, 0, n); field.mesh.userData.ktSplat.thin = false; field.commit(n); field.setCount(n);
      }
      function showPhase(p) {
        var M = mesh, L = p === 'list';
        if (!M) return;
        M.phase = p; M.t = 0; M.rt = 0; M.hold = 0; M.done = false;
        if (segH) segH.set(p);
        if (!L) writeLattice(); else if (M.L) writeList(); else field.setCount(0);
        [M.tri, M.wire, M.fresh, M.win, M.dot].forEach(function (o) { o.visible = !L; });
        (M.lobj || []).forEach(function (o) { o.visible = L; });
        rdM.el.firstChild.textContent = L ? 'triangles made' : 'triangles stitched';
        if (L) stepList(0); else stepLattice(0);
        paintBoard();
      }
      function listReady() {                         // the list's mesh is done: its triangles, red edges and hole rims, to be revealed
        var M = mesh, J = M.job;
        M.ltri = surfaceMesh(M.L.pos, M.L.col, J.tri);
        M.lbad = edgeLines(J.stats.bad, M.L.pos, C.err, 0.95); M.lhole = edgeLines(J.stats.hole, M.L.pos, C.err, 0.45);
        M.lobj = [M.ltri, M.lbad, M.lhole];
        M.lobj.forEach(function (o) { o.frustumCulled = false; o.visible = M.phase === 'list'; o.geometry.setDrawRange(0, 0); scene.add(o); });
      }
      function stepMesh(dt) {
        var M = mesh;
        M.age += dt;
        if (M.due && M.age >= M.due) { buildMesh(); return; }        // a new budget, and the slider has rested
        // one piece of work per frame, SLICE ms at most: emulate the list, grade the lattice, mesh the list (graded alike)
        if (!M.L) { if (M.ejob.step(SLICE)) { M.L = M.ejob.list; M.job = ML.mesher(M.L); if (M.phase === 'list') writeList(); } paintBoard(); }
        else if (!M.latA) { if (M.ajob.step(SLICE)) M.latA = M.ajob.stats; paintBoard(); }
        else if (!M.job.done) { if (M.job.step(SLICE)) listReady(); paintBoard(); }
        if (M.phase === 'list') stepList(dt); else stepLattice(dt);
      }
      function stepList(dt) {
        var M = mesh, J = M.job, n = M.L ? M.L.n : 0;
        if (!J || !J.done) { field.setCount(n); rdG.set(n.toLocaleString('en-US')); rdM.set(J ? Math.round(J.progress * 100) + ' %' : '…'); return; }
        M.rt += dt;
        var f = Math.min(1, M.rt / LIST_T), q = Math.min(J.nTri, Math.floor(f * f * J.nTri)), gone = f >= 1 ? n : q ? J.tri[(q - 1) * 3] + 1 : 0;
        M.ltri.geometry.setDrawRange(0, q * 3);
        M.lbad.geometry.setDrawRange(0, upTo(J.stats.bad, q) * 2); M.lhole.geometry.setDrawRange(0, upTo(J.stats.hole, q) * 2);
        field.setCount(n - gone);                    // each triangle is listed by the first Gaussian that proposed it: those before it are done
        rdG.set((n - gone).toLocaleString('en-US')); rdM.set(q.toLocaleString('en-US') + ' / ' + J.nTri.toLocaleString('en-US'));
        if (f >= 1 && M.auto && (M.hold += dt) > HOLD_T) showPhase('lattice');
      }
      function stepLattice(dt) {
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
      /* the board: one column per store, the numbers as they are measured */
      function paintBoard() {
        if (!mesh || !ML) { board.innerHTML = ''; return; }
        var M = mesh, J = M.job, A = J && J.done ? J.stats : null, B = M.latA, E = J ? J.ms : {};
        function num(x) { return x.toLocaleString('en-US'); }
        function big(x) { return x >= 1e6 ? (x / 1e6).toFixed(1) + ' M' : x >= 1e4 ? Math.round(x / 1e3) + ' k' : num(x); }
        function ms(x) { return x < 1 ? '&lt; 1 ms' : num(Math.round(x)) + ' ms'; }
        function td(x, err) { return x == null ? '<td>…</td>' : '<td' + (err ? x > 0 ? ' class="bad"' : ' class="ok"' : '') + '>' + num(x) + '</td>'; }
        function row(name, k) { return '<tr><th>' + name + '</th>' + td(A && A[k], 1) + td(B && B[k], 1) + '</tr>'; }
        board.innerHTML = '<b>the same ' + num(M.V) + ' Gaussians, meshed two ways</b><table>' +
          '<thead><tr><th></th><th' + (M.phase === 'list' ? ' class="on"' : '') + '>a list <i>emulated</i></th><th' + (M.phase === 'list' ? '' : ' class="on"') + '>the lattice</th></tr></thead><tbody>' +
          '<tr><th>neighbour search</th><td>' + (!J ? 'emulating the list · ' + Math.round(M.ejob.progress * 100) + ' %' : J.phase === 'index' ? '…' :
            big(J.tests) + ' distances · ' + ms((E.index || 0) + (E.search || 0))) + '</td><td class="ok">none</td></tr>' +
          '<tr><th>triangulation</th><td>' + (A ? ms((E.normal || 0) + (E.fan || 0) + (E.vote || 0)) : J ? Math.round(J.progress * 100) + ' %' : '…') + '</td><td>' + ms(M.lat.ms) + '</td></tr>' +
          '<tr><th>triangles</th>' + td(A && A.triangles) + td(B && B.triangles) + '</tr>' +
          row('bridges', 'bridges') + row('non-manifold edges', 'nonManifold') + row('holes', 'holes') +
          '</tbody></table><span class="mf-key"><i></i>bridge or non-manifold edge <i class="f"></i>edge of a hole · times measured here</span>';
      }
      /* ---------- the stitched lattice as a file: every chart in the mesh, in metres where the scene has a scale ---------- */
      function scaleOf() {
        var k = /^scan(\d*)$/.exec(st.scn), D = k && (k[1] ? (KT.dtuScans || {})[k[1]] : KT.dtu);
        return D && D.unit_mm ? { scale: D.unit_mm / 1000, unit: 'm', note: '1 scene unit = ' + D.unit_mm + ' mm', tag: 'metres', scan: D.scan || k[1] }
                              : { scale: 1, unit: 'scene units', note: '', tag: 'scene-units', scan: null };
      }
      function paintIO() {
        if (!ioNote || !mesh) return;
        var s = scaleOf();
        ioNote.classList.remove('bad');
        ioNote.textContent = mesh.parts.map(function (m) { return m.c.name; }).join(' + ') + ' · ' + mesh.V.toLocaleString('en-US') + ' points · ' +
          (mesh.Q * 2).toLocaleString('en-US') + ' triangles · ' + (s.scan ? 'in metres (' + s.note + ')' : 'in scene units');
      }
      function meshFile(fmt) {
        var M = mesh, s = scaleOf(), S = KT.scenes[st.scn];
        return IO.write(fmt, { name: 'Knot-a-Surface: ' + S.label + ', the stitched lattice', unit: s.unit + (s.note ? ' (' + s.note + ')' : ''), scale: s.scale,
          notes: [S.label + ', ' + S.suite + (S.note ? ': ' + S.note : ''),
                  'spline charts sampled on even lattices and stitched by a 2 x 2 window: ' + M.V + ' points, ' + M.Q * 2 + ' triangles',
                  '+y up' + (s.scan ? ', the table at y = 0' : '')],
          parts: M.parts.map(function (m) {
            var k = m.na * m.nb, src = M.lat.tri, o = m.q0 * 6, tri = new Uint32Array(m.qa * m.qb * 6), j, r = m.sign < 0;
            for (j = 0; j < tri.length; j += 3) {      // counter-clockwise seen from the side the normals point to
              tri[j] = src[o + j] - m.off; tri[j + 1] = src[o + j + (r ? 2 : 1)] - m.off; tri[j + 2] = src[o + j + (r ? 1 : 2)] - m.off;
            }
            return { name: m.c.name, pos: M.pos.subarray(m.off * 3, (m.off + k) * 3), nrm: M.nrm.subarray(m.off * 3, (m.off + k) * 3),
                     col: M.col.subarray(m.off * 3, (m.off + k) * 3), tri: tri };
          }) });
      }
      function fileName(fmt) {
        var s = scaleOf(), S = KT.scenes[st.scn];
        return S.label.toLowerCase().replace(/[^a-z0-9]+/g, '-') + (s.scan ? '-dtu' + s.scan : '') + '-lattice-' + mesh.V + '-points-' + s.tag + '.' + fmt;
      }
      /* saved through KnotTalk.saveFile, looked up now: a page may have replaced it (talk/meshio.js) */
      function download() {
        if (!mesh || !IO) return;
        var fmt = segF.get(), name = fileName(fmt), b;
        ioNote.classList.remove('bad');
        try { b = IO.blob(meshFile(fmt)); } catch (e) { ioNote.textContent = 'could not write the file: ' + e.message; ioNote.classList.add('bad'); return; }
        ioNote.textContent = 'saving ' + name + ' …';
        Promise.resolve().then(function () { return window.KnotTalk.saveFile(name, b); }).then(function () {
          ioNote.textContent = name + ' · ' + (b.size / 1048576).toFixed(1) + ' MB';
        }, function (e) {
          ioNote.textContent = 'not saved: ' + ((e && e.message) || e || 'refused'); ioNote.classList.add('bad');
        });
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
      /* the new material wipes in over the one that is on the object now */
      function pickMat(v) { if (st.paint > 0.5) { st.from = st.mat; st.fromT = st.target; } st.mat = v; st.paint = 0; if (segM) segM.set(v); showKnobs(); if (!orbit.touched) orbit.to(view(matView())); }
      /* the same material wipes over its new target: object (owl and books), owl, or what is selected on the atlas */
      function pickTarget(v) {
        if (!def || !def.targets || !def.targets.some(function (t) { return t.v === v; }) || (v === 'selection' && !hands)) return;
        if (st.paint > 0.5) { st.from = st.mat; st.fromT = st.target; }
        st.target = v; st.paint = 0; if (segT) segT.set(v); showHands();
      }
      /* the tool layer's panel: all of it on the `hands` scene; its atlas and selection tools while a material goes on a selection */
      function selecting() { return !!(hands && st.sc.paint && st.target === 'selection' && def && def.targets); }
      function showHands() {
        if (!hands) return;
        var sel = selecting(), on = !!st.sc.hands || sel;
        hands.show(on, sel ? 'select' : 'hands'); root.classList.toggle('mf-handson', on);
        hint.textContent = sel ? 'select on the atlas or on the object: the material follows the selection' : st.sc.hint || '';
        need = true;
        // a page without the scan's DINOv3 features (the talk) fetches them the first time a selection is wanted; the atlas offers them then
        if (sel && def.features && !def.featAsked) { def.featAsked = true; def.features(showHands); }
      }
      /* one slider per property the chosen material lets the simulator turn */
      function showKnobs() {
        if (knobs) { knobs.remove(); knobs = null; }
        var m = def && def.materials && def.materials.filter(function (k) { return k.v === st.mat; })[0];
        if (!m || !m.knobs.length) return;
        knobs = ui.el('div', 'mf-mat mf-knobs'); panel.appendChild(knobs);
        m.knobs.forEach(function (k) {
          ui.slider(knobs, { label: k.label, min: k.min, max: k.max, step: k.step, value: m.p[k.k],
            fmt: function (v) { return (+v).toFixed(2); }, onInput: function (v) { def.tune(m.v, k.k, +v); } });
        });
      }
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
        if (!KT.scenes[sc.scn]) sc = Object.assign({}, sc, { scn: KT.scenes.scan && sc.scn !== 'scan' ? 'scan' : 'barn' });
        root.classList.toggle('mf-noatlas', !sc.atlas); root.classList.toggle('mf-tools', !!sc.tools); root.classList.toggle('mf-paint', !!sc.paint);
        root.classList.toggle('kt-bare', !!env.opts.bare || !!sc.bare);       // a scene may ask for no chrome at all (the cover)
        st.scene = id; st.sc = sc; st.t = 0; st.tourI = 0; st.tourT = 0; st.abT = 0; st.touchedAB = false; st.touchedEdit = false;
        setBands(sc.teaser || sc.cover ? TEASE_BANDS : null, small ? 1 : 2);      // a narrow stage has room for three bands, not six
        field.uniforms.uFade.value = 0; split.style.opacity = '';
        st.view = sc.view; segV.set(sc.view); showView(); st.nets = !!sc.nets; st.rig = !!sc.rig;
        st.adaptive = true; tgA.set(true); paintBadge();
        field.uniforms.uRim.value = sc.rim || 0.3;
        var fresh = st.scn !== sc.scn;
        loadScene(sc.scn); showNets(); paintCap(); syncMesh(); st.paint = 0; st.from = 'stone'; st.fromT = st.target;
        hint.textContent = sc.hint || '';
        if (hands) { hands.reset(); showHands(); }
        orbit.touched = false;
        orbit.to(view(sc.teaser && def.views.near ? 'near' : matView()), fresh || !!sc.teaser);
        need = true;
      }

      /* ---------- scene `teaser`: one loop that ends exactly where it begins ----------
         TEASE seconds = nine breaths of the owl, and its head tilt and its blink are at rest at both ends.
           0 .. 7.5    three readings in six bands (normals, Gaussians, the picture, twice), which roll across the
                       stage twice per loop: the pattern repeats every half stage, so the roll closes too
           .. 9        they dissolve into the picture (uFade), the split is switched off
           9 .. 14.5   a material wipes in (the snowy owl), holds
           .. 23       the next one wipes in over it (the jelly); its opacity and gloss swell and settle
           .. 28       the photograph's colours wipe back
           .. 29.5     the picture dissolves into the three readings again
         Everything is a function of the time in the loop, and the camera's path is closed. */
      var TEASE = 30.6, TEASE_BANDS = [{ v: 'normal', name: 'normals' }, { v: 'gauss', name: 'Gaussians' }, { v: 'splats', name: 'visual' }];
      var TEASE_ON = 'owl';                         // the teaser's materials go on the owl alone: a white owl and a green one on stone books
      function sstep(a, b, v) { v = KT.clamp((v - a) / (b - a), 0, 1); return v * v * (3 - 2 * v); }
      function tease() {
        var u = st.t % TEASE, f = u < 7.5 ? 0 : u < 28 ? sstep(7.5, 9, u) : 1 - sstep(28, 29.5, u), U = field.uniforms;
        var mat = 'stone', from = 'stone', paint = 0, sw = 0;
        if (u >= 9 && u < 14.5) { mat = 'snowy'; paint = (u - 9) / 3.5; }
        else if (u >= 14.5 && u < 23) { from = 'snowy'; mat = 'jelly'; paint = (u - 14.5) / 3.5; sw = u < 18 ? 0 : 0.5 - 0.5 * Math.cos((u - 18) / 5 * 2 * Math.PI); }
        else if (u >= 23) { from = 'jelly'; paint = (u - 23) / 3.5; }
        U.uFade.value = f; U.uMode.value = 0; roll = wrap(2 * u / TEASE, 1); U.uSplit.value.set(f < 1 ? shown.length : 0, roll, SLANT);
        root.classList.toggle('kt-splitting', f < 1); split.style.opacity = 1 - f; if (f < 1) placeSplit();
        if (def.animate) {
          if (def.tune) { def.tune('jelly', 'opacity', 0.34 + 0.4 * sw); def.tune('jelly', 'gloss', 1 + 0.8 * sw); }
          def.animate(u, Math.min(1, paint), mat, from, TEASE_ON, TEASE_ON).forEach(coarse); need = true;
        }
        if (!orbit.touched) {                       // close and swaying; round to the three-quarter view while the owl is see-through
          var a = view(def.views.near ? 'near' : 'wide'), b = def.views.through ? view('through') : a, g = sstep(14.5, 17.5, u) * (1 - sstep(23, 26.5, u)), o = orbit.goal;
          ['th', 'ph', 'r', 'tx', 'ty', 'tz'].forEach(function (k) { o[k] = a[k] + (b[k] - a[k]) * g; });
          o.th += 0.16 * Math.sin(u / TEASE * 2 * Math.PI);
        }
      }

      /* ---------- scene `cover`: the talk's first screen ----------
         The owl and the rabbit (scene `duo`). The camera drifts slowly, the owl lives, and every COVER seconds the
         picture dissolves for a while into three readings in rolling bands (normals, Gaussians, the picture):
           0 .. 3.5   the picture        .. 5     it dissolves into the bands
           .. 11      the bands roll     .. 12.5  back to the picture, which holds to the end of the loop
         The pair is framed from its bounding box (def.box) to fit the stage at any shape. */
      var COVER = 22;
      function coverLoop() {
        var u = st.t % COVER, f = u < 3.5 ? 1 : u < 11 ? 1 - sstep(3.5, 5, u) : sstep(11, 12.5, u), U = field.uniforms;
        U.uFade.value = f; U.uMode.value = 0; roll = wrap(st.t * 0.035, 1); U.uSplit.value.set(f < 1 ? shown.length : 0, roll, SLANT);
        root.classList.toggle('kt-splitting', f < 1); split.style.opacity = 1 - f; if (f < 1) placeSplit();
        if (orbit.touched) return;
        var w = def.views.wide, o = orbit.goal, B = def.box, asp = size.w / Math.max(1, size.h), k;
        for (k in w) o[k] = w[k];
        if (B) {                                   // the half extents plus a margin must fit the half field of view (36 degrees tall)
          var hw = B.h[0] * 1.08 + 0.25, hh = B.h[1] * 1.12 + 0.3;
          o.r = Math.max(hh, hw / asp) / Math.tan(18 * Math.PI / 180) + B.h[2];
          o.tx = B.c[0]; o.ty = B.c[1]; o.tz = B.c[2];
        }
        o.th += 0.26 * Math.sin(st.t * 2 * Math.PI / 34); o.ph += 0.04 * Math.sin(st.t * 2 * Math.PI / 26);
      }

      env.loop(function (t, dt) {
        var sc = st.sc;
        st.t += dt;
        if (sc.teaser) tease();
        if (sc.cover) coverLoop();
        if (hands && (sc.hands || selecting())) hands.step(dt);
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
          (def.animate ? def.animate(st.t, st.paint, st.mat, st.from, st.target, st.fromT) : def.edit(0.5 - 0.5 * Math.cos(st.t * 1.15))).forEach(coarse);
          if (st.nets) updateNets();
          need = true;
        }
        var home = def.views[matView()];
        if (!orbit.touched && !sc.tour && !sc.teaser && !sc.cover && (sc.cam || 'wide') === 'wide') orbit.goal.th = home.th + 0.14 * Math.sin(t * 0.16);
        orbit.update(dt, sc.tour ? 1.5 : 3);
        var e = cam.matrixWorld.elements, key = (e[12] + e[13] * 3 + e[14] * 7 + e[8] * 11 + e[9] * 13).toFixed(3);
        if (key !== camKey && t - lastAlloc > 0.09) { camKey = key; need = true; }
        if (mesh) { need = false; stepMesh(dt); }
        else if (need) { need = false; lastAlloc = t; resample(); }
        var ud = field.uniforms.uDepth.value, kz = 1 - Math.exp(-5 * dt);
        ud.x += (zr[0] - ud.x) * kz; ud.y += (zr[1] - ud.y) * kz;
        if (st.view === 'split' && !sc.teaser) { roll = wrap(roll + ROLL * dt, 1); field.uniforms.uSplit.value.y = roll; placeSplit(); }
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
          '.kt-sim-manifold:not(.mf-mesh) .mf-tri,.kt-sim-manifold.mf-mesh .mf-samp,.kt-sim-manifold:not(.mf-paint) .mf-mat,.kt-sim-manifold:not(.mf-paint) .mf-target{display:none}' +
          // scene `mesh`: the board (list against lattice, bottom left) and the export rows of the panel
          '.kt-sim-manifold .mf-board{position:absolute;z-index:5;left:12px;bottom:12px;max-width:calc(100% - 24px);padding:9px 11px 8px;pointer-events:none;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
          'font:500 11px/1.3 var(--f-mono);color:var(--dim)}.kt-sim-manifold:not(.mf-mesh) .mf-board,.kt-sim-manifold.kt-bare .mf-board{display:none}' +
          '.kt-sim-manifold .mf-board>b{display:block;color:var(--text);font-weight:600;margin-bottom:6px}' +
          '.kt-sim-manifold .mf-board table{border-collapse:collapse}' +
          '.kt-sim-manifold .mf-board th,.kt-sim-manifold .mf-board td{padding:2px 0 2px 16px;text-align:right;white-space:nowrap;font-weight:500;font-variant-numeric:tabular-nums}' +
          '.kt-sim-manifold .mf-board th:first-child{padding-left:0;text-align:left;color:var(--dim)}' +
          '.kt-sim-manifold .mf-board thead th{font:600 10px/1.2 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--mute);padding-bottom:4px}' +
          '.kt-sim-manifold .mf-board thead th.on{color:var(--text)}.kt-sim-manifold .mf-board thead i{font-style:normal;text-transform:none;letter-spacing:0;color:var(--mute)}' +
          '.kt-sim-manifold .mf-board td{color:var(--text)}.kt-sim-manifold .mf-board td.bad{color:var(--err)}.kt-sim-manifold .mf-board td.ok{color:var(--ok)}' +
          '.kt-sim-manifold .mf-key{display:block;margin-top:6px;font-size:10px;color:var(--mute)}' +
          '.kt-sim-manifold .mf-key i{display:inline-block;width:14px;height:2px;margin:0 5px 3px 0;vertical-align:middle;background:var(--err)}' +
          '.kt-sim-manifold .mf-key i.f{margin-left:8px;opacity:.45}' +
          '.kt-sim-manifold .mf-io{display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--line);margin-top:2px;padding-top:7px}' +
          '.kt-sim-manifold .mf-io-note{font:500 10px/1.4 var(--f-mono);color:var(--mute);max-width:280px}.kt-sim-manifold .mf-io-note.bad{color:var(--err)}' +
          '@media (max-width:640px){.kt-sim-manifold .mf-board{left:6px;bottom:6px;max-width:calc(100% - 12px);padding:6px 8px;font-size:9.5px}' +
          '.kt-sim-manifold .mf-board>b{margin-bottom:3px}.kt-sim-manifold .mf-board th,.kt-sim-manifold .mf-board td{padding:1px 0 1px 9px}' +
          '.kt-sim-manifold .mf-board thead th{font-size:8.5px;letter-spacing:.03em}.kt-sim-manifold .mf-key{display:none}}' +
          '.kt-sim-manifold .mf-knobs{border-top:1px solid var(--line);margin-top:6px;padding-top:4px}' +
          // with the tool layer's panel on a narrow stage, this panel keeps to the upper half (it scrolls), that one to the lower
          '.kt-sim-manifold.kt-compact.kt-open.mf-handson .kt-tr{max-height:calc(50% - 48px);overflow:auto}' +
          '@media (max-width:640px){.kt-sim-manifold.mf-handson .kt-tr{max-height:calc(50% - 24px);overflow:auto}}' +
          '.kt-sim-manifold.mf-handson .kt-tr>*{flex-shrink:0}' +          // its rows keep their height: the panel scrolls, they never overlap
          // on a narrow stage the paint scene's own controls come first: the material, where it goes, its knobs
          '.kt-sim-manifold.mf-paint.kt-compact .kt-tr>.kt-panel-title{order:-2}' +
          '.kt-sim-manifold.mf-paint.kt-compact .kt-tr>.mf-mat,.kt-sim-manifold.mf-paint.kt-compact .kt-tr>.mf-target,.kt-sim-manifold.mf-paint.kt-compact .kt-tr>.mf-knobs{order:-1}' +
          '@media (max-width:640px){.kt-sim-manifold.mf-paint .kt-tr>.kt-panel-title{order:-2}' +
          '.kt-sim-manifold.mf-paint .kt-tr>.mf-mat,.kt-sim-manifold.mf-paint .kt-tr>.mf-target,.kt-sim-manifold.mf-paint .kt-tr>.mf-knobs{order:-1}}' +
          '.kt-sim-manifold .mf-tog{flex-wrap:wrap}' +

          '.kt-sim-manifold.kt-bare .kt-panel,.kt-sim-manifold.kt-bare .kt-atlas,.kt-sim-manifold.kt-bare .kt-caption,' +
          '.kt-sim-manifold.kt-bare .kt-hint,.kt-sim-manifold.kt-bare .kt-ab,.kt-sim-manifold.kt-bare .kt-gear{display:none!important}' +
          '.kt-sim-manifold .kt-panel .kt-seg .kt-seg-box{margin-left:0;width:100%}' +
          '.kt-sim-manifold .kt-panel .kt-seg .kt-seg-btn{flex:1;padding:7px 4px}' +
          '.kt-sim-manifold .kt-split{position:absolute;inset:0;z-index:3;overflow:hidden;pointer-events:none;display:none}' +
          '.kt-sim-manifold.kt-splitting .kt-split{display:block}.kt-sim-manifold.kt-splitting .kt-atlas{display:none}' +
          '.kt-sim-manifold .kt-split i{position:absolute;left:-1px;top:0;width:2px;height:100%;transform-origin:0 0;background:rgba(255,255,255,.6)}' +
          '.kt-sim-manifold.kt-bare .kt-split i{background:rgba(255,255,255,.3)}' +
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
        load: function (key) { st.sc = Object.assign({}, st.sc, { scn: key }); st.tourI = 0; st.tourT = 0; loadScene(key); syncMesh(); orbit.touched = false; orbit.to(view('wide'), true); paintCap(); if (hands) { hands.reset(); showHands(); } },
        setMaterial: pickMat,
        setTarget: pickTarget,                                                            // 'object', 'owl' or 'selection' (the paint scene's control)
        hands: function () { return hands; },                                             // for tests
        pickSurface: pickSurface,
        setBudget: function (n) { st.budget = KT.clamp(n, 6000, CAP - 20000); slB.set(st.budget); need = true; remesh(); },
        /* scene `mesh`, for tests: what is meshed and every number on the board; the stitched lattice as a file */
        meshStats: function () {
          if (!mesh) return null;
          function lean(a) { if (!a) return null; var o = {}; Object.keys(a).forEach(function (k) { if (typeof a[k] === 'number') o[k] = a[k]; }); return o; }
          var J = mesh.job;
          return { phase: mesh.phase, auto: mesh.auto, points: mesh.V, budget: st.budget, triangles: mesh.Q * 2, stitchMs: mesh.lat.ms,
                   parts: mesh.parts.map(function (m) { return { name: m.c.name, na: m.na, nb: m.nb }; }), lattice: lean(mesh.latA),
                   list: J && { done: J.done, progress: J.progress, tests: J.tests, ms: J.ms, cell: J.cell, audit: lean(J.stats) } };
        },
        meshFile: function (fmt) { return mesh && IO ? meshFile(fmt) : null; },
        meshSeek: function (t) { if (mesh) { mesh.t = mesh.rt = t; } },                  // for tests: jump the running animation to t seconds
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
