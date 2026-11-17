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
   `mesh` also on meshlist.js (the list's mesher, the audit and the distances), meshtsdf.js (TSDF fusion; without it the
   comparison has two columns) and meshio.js (the export), loaded before this file. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  /* T: a string in the page's language (talk/i18n.js; the English itself on a page without languages). Every string a
     viewer reads goes through it, a sentence with a number or a name in it as ONE template, so that the sentence keeps
     its own word order and direction. What T returns is for showing only: ids (scene and material keys, chart names),
     file names and what is written into a file stay in English. */
  var T = KT.t;
  var CTRL_WORD = window.KnotTalk.theme === 'light' ? T('orange') : T('amber');   // the control points' colour, as the copy names it
  var PER_CP = 16;                  // 3 position + 12 SH1 colour + 1 opacity per control point
  var CAP0 = 320000, FLOOR = 0.10;  // the most Gaussians a mount holds (opts.cap may lower it); FLOOR = share of the budget kept as view-independent coverage
  var DENSE = 96;                   // a control net longer than this along either side is drawn thinned out
  /* what a Gaussian shows; the order is the shader's uMode (gl.js). The view 'split' shows all of them */
  var MODES = [{ v: 'splats', label: 'rgb', name: T('appearance') }, { v: 'depth', label: T('depth'), name: T('depth') },
               { v: 'normal', label: T('normal'), name: T('normals') }, { v: 'gauss', label: T('gauss'), name: T('Gaussians') },
               { v: 'charts', label: T('chart'), name: T('charts') }];
  var ROLL = 0.03, SLANT = 0.2;     // split screen: band travel in stage widths per second, and the lean of its dividers
  /* rim shading darkens the fringe of every Gaussian (uRim: 0.3 unless a scene asks for more). On a light page that fringe
     draws a dark outline around a light-coloured surface: there a light scene (KT.scenes[..].light) or a light material
     (talk/scenes.js, MATS[..].light) is shaded with LIGHT_RIM. A dark page keeps the scene's own rim */
  var LIGHT = KT.theme === 'light', LIGHT_RIM = 0.12;
  function wrap(x, m) { return ((x % m) + m) % m; }

  /* The talk runs on ONE subject: `scan`, a real DTU object (talk/dtu-data.js; without that file the barn
     stands in). atlas: 1 shows the (u,v) tiles, tools: 1 the toggles. */
  var SCENES = {
    scene:  { scn: 'scan', view: 'splats', cam: 'wide', cap: T('every Gaussian derived from a spline chart'), hint: T('drag to orbit') },
    modes:  { scn: 'scan', view: 'split', cam: 'wide', cap: T('one set of Gaussians, read five ways at once') },
    charts: { scn: 'scan', view: 'charts', cam: 'wide', atlas: 1, cap: T('an atlas of charts · each coherent surface gets its own (u,v) domain') },
    nets:   { scn: 'scan', view: 'charts', cam: 'wide', atlas: 1, nets: 1, cap: T('control nets · geometry and appearance live on the {colour} points', { colour: CTRL_WORD }) },
    budget: { scn: 'scan', view: 'splats', tour: 1, atlas: 1, cap: T('fixed budget, moving camera · samples flow to what is seen') },
    lod:    { scn: 'scan', view: 'splats', cam: 'close', ab: 1, rim: 0.6, atlas: 1, cap: T('level of detail · the same budget, with and without view-adaptive sampling') },
    /* the same Gaussians meshed three ways, side by side: on the charts' lattices, by a 2 x 2 window; as an (emulated)
       list, by neighbour search; and by TSDF fusion of that list's depth maps */
    mesh:   { scn: 'scan', view: 'gauss', cam: 'wide', mesh: 1, cmp: 1, cap: KT.meshTsdf ? T('tessellation · the same Gaussians, three ways to a mesh')
                                                                                         : T('tessellation · the same Gaussians as a list (emulated) and on the lattice') },
    /* one of them on the whole stage: the lattice, as it is stitched (the deck's second mesh step; all three are still on the board) */
    stitch: { scn: 'scan', view: 'gauss', cam: 'wide', mesh: 1, from: 'lattice', cap: T('tessellation · on the lattice: no search, two triangles per stop of a 2 × 2 window') },
    edit:   { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, tools: 1, cap: T('editable by construction · move control points, the Gaussians follow'),
              hint: T('turn on nets to drag a control point') },
    paint:  { scn: 'scan', view: 'splats', cam: 'wide', breathe: 1, paint: 1, cap: T('material edit · the same net, new control colours') },
    /* the tool layer of talk/hands.js: work on the object with the pointer, and on its atlas */
    hands:  { scn: 'scan', view: 'splats', cam: 'near', tools: 1, hands: 1, cap: T('hands · grab, pinch, pull, push or hold the surface: every action moves control points'),
              hint: T('drag on the object to act, beside it to orbit') },
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
      { id: 'budget', label: 'Fixed budget' }, { id: 'lod', label: 'Level of detail' }, { id: 'mesh', label: 'Three meshes' }, { id: 'stitch', label: 'Stitch a mesh' },
      { id: 'edit', label: 'Edit' }, { id: 'paint', label: 'Materials' }, { id: 'hands', label: 'Hands' }
    ],
    create: function (env) {
      var THREE = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new THREE.PerspectiveCamera(36, 1, 0.1, 160);
      var R = G.renderer(root, cam, env.opts.clear), scene = new THREE.Scene();
      var CAP = Math.min(+env.opts.cap || CAP0, CAP0);       // this mount's Gaussian capacity: a page that mounts several can ask for less
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      /* sampling: 'even' | 'adaptive' | 'window' (opts.sampling picks it). The standalone lab page (the shell sets body.kt-labpage; an
         embed passes `embed`) starts on the view window; every embedded use keeps the adaptive sampler it always had. st.adaptive is
         "follows the view" (off = even) and st.win picks the view window among the view-following modes. */
      var SAMPLINGS = ['even', 'adaptive', 'window'];
      var dflt = SAMPLINGS.indexOf(env.opts.sampling) >= 0 ? env.opts.sampling : !env.embed && document.body.classList.contains('kt-labpage') ? 'window' : 'adaptive';
      var st = { scene: 'scene', sc: SCENES.scene, scn: '', view: 'splats', adaptive: dflt !== 'even', win: dflt === 'window', buf: 0.04, nets: false, rig: false, t: 0,
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
      var panel = ui.panel(root, { pos: 'tr', title: T('scene · budget') });
      /* the picker offers the scanned objects; only a page without their data offers the hand-modelled scenes */
      var pick = Object.keys(KT.scenes).filter(function (k) { return !KT.scenes[k].hidden; }), real = pick.filter(function (k) { return KT.scenes[k].photo; });
      var segS = ui.seg(panel, { value: 'barn', options: (real.length ? real : pick).map(function (k) { return { v: k, label: KT.scenes[k].label.toLowerCase() }; }),
        onChange: function (v) {
          st.sc = Object.assign({}, st.sc, { scn: v, tour: 0, breathe: 0, ab: 0, cam: 'wide' });
          loadScene(v); syncMesh(); orbit.touched = false; orbit.to(view('wide'), true); paintCap();
          if (hands) { hands.reset(); showHands(); }
        } });
      segS.el.classList.add('mf-pick');
      var slB = ui.slider(panel, { label: T('budget'), min: 6000, max: CAP - 20000, step: 2000, value: Math.min(st.budget, CAP - 20000),
        fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; }, onInput: function (v) { st.budget = v; need = true; remesh(); } });
      var segV = ui.seg(panel, { value: 'splats', options: MODES.concat([{ v: 'split', label: T('all') }]),
        onChange: function (v) { st.view = v; showView(); } });
      var rowT = ui.el('div', 'kt-row mf-tog'); panel.appendChild(rowT);
      var tgA = ui.toggle(rowT, { label: T('adaptive'), value: dflt !== 'even', onChange: function (v) { st.adaptive = v; st.touchedAB = true; need = true; paintBadge(); paintMode(); } });
      var tgN = ui.toggle(rowT, { label: T('nets'), value: false, onChange: function (v) { st.nets = v; showNets(); } });
      var tgR = ui.toggle(rowT, { label: T('cameras'), value: false, onChange: function (v) { st.rig = v; showNets(); } });
      ui.toggle(rowT, { label: T('no back faces'), value: true, onChange: function (v) { st.cull = v; need = true; } });
      /* the ground (a chart the scene marks `ground`) can be put away: the object alone, with the whole budget */
      ui.toggle(rowT, { label: T('ground'), value: true, onChange: function (v) { st.ground = v; charts.forEach(function (c) { c.off = !!c.ground && !v; }); need = true; remesh(); } });
      /* the lab page (and a mount that asks for a sampling) picks it here: even, adaptive or the view window, with its buffer */
      var segW = null, slW = null;
      function mode() { return !st.adaptive ? 'even' : st.win ? 'window' : 'adaptive'; }
      function setMode(m) { st.adaptive = m !== 'even'; if (m !== 'even') st.win = m === 'window'; st.touchedAB = true; need = true; paintMode(); paintBadge(); }
      function paintMode() { tgA.set(st.adaptive); if (segW) segW.set(mode()); if (slW) slW.el.style.display = mode() === 'window' ? '' : 'none'; }
      if (!env.embed || env.opts.sampling) {
        segW = ui.seg(panel, { value: mode(), options: [{ v: 'even', label: T('even') }, { v: 'adaptive', label: T('adaptive') }, { v: 'window', label: T('view window') }], onChange: setMode });
        segW.el.classList.add('mf-samp');
        slW = ui.slider(panel, { label: T('buffer'), min: 0, max: 0.15, step: 0.005, value: st.buf,
          fmt: function (v) { return (v * 100).toFixed(1) + ' %'; }, onInput: function (v) { st.buf = v; need = true; } });
        slW.el.classList.add('mf-samp');
        tgA.el.style.display = 'none';                     // the three-way control above says the same, and more
      }
      var rdG = ui.readout(panel, { label: T('Gaussians drawn'), tone: 'splat' });
      var rdV = ui.readout(panel, { label: T('budget in view'), tone: 'ok' });
      rdV.el.classList.add('mf-samp');   // the view-adaptive sampler's row: not in the mesh scene (whose lattice the budget sizes)
      var rdP = ui.readout(panel, { label: T('learnable params'), tone: 'ctrl' });
      var rdM = ui.readout(panel, { label: T('triangles stitched'), tone: 'surf' }); rdM.el.classList.add('mf-tri');
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
            segT = ui.seg(panel, { label: T('on'), value: st.target, options: tg, onChange: pickTarget });
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
        var ptGeo = new THREE.BufferGeometry(); ptPos = new Float32Array(shownCP * 3);
        ptGeo.setAttribute('position', new THREE.BufferAttribute(ptPos, 3));
        pts = new THREE.Points(ptGeo, new THREE.PointsMaterial({ color: C.ctrl, size: shownCP > 3000 || shownCP < totalCP ? 3 : 5, sizeAttenuation: false }));
        pts.frustumCulled = false; scene.add(pts);
        // capture rig: one small frustum per camera of the dataset-style rig
        var cams = def.rig.cams, tg = def.rig.target, n = 0;
        rigL = G.lines(cams.length * 16, C.text, 0.4); scene.add(rigL.obj);
        cams.forEach(function (p) {
          var f = new THREE.Vector3(tg[0] - p[0], tg[1] - p[1], tg[2] - p[2]), d = f.length() * 0.04; f.normalize();
          var r = new THREE.Vector3().crossVectors(f, new THREE.Vector3(0, 1, 0)).normalize(), u = new THREE.Vector3().crossVectors(r, f);
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
          tile.appendChild(ui.el('div', 'kt-tile-name', c.label));        // what a viewer reads for the chart's name (talk/scenes.js); the name itself is an id
          var box = ui.el('div', 'kt-tile-box'); tile.appendChild(box);
          c.cv = ui.canvas(box, 'kt-tile-cv');
          c.lab = ui.el('div', 'kt-tile-n'); tile.appendChild(c.lab);
          tile.appendChild(ui.el('div', 'kt-tile-s', T('{n} ctrl', { n: c.patch.nu + '×' + c.patch.nv })));
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
      var M = new THREE.Matrix4();
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
        var win = mode() === 'window', lim = win ? 1 + st.buf : 1.12;      // lim: how far past the image edge (in half-images) a cell still asks for Gaussians
        charts.forEach(function (c) { if (!c.off) totA += c.area; });
        charts.forEach(function (c) {
          var n = c.gu * c.gv, i, o, w; c.w = c.w || new Float32Array(n); c.vis = c.vis || new Uint8Array(n);
          if (c.off) { c.w.fill(0); c.vis.fill(0); return; }
          for (i = 0; i < n; i++) {
            o = i * 3; var x = c.cpos[o], y = c.cpos[o + 1], z = c.cpos[o + 2];
            var cw = e[3] * x + e[7] * y + e[11] * z + e[15], px = e[0] * x + e[4] * y + e[8] * z + e[12], py = e[1] * x + e[5] * y + e[9] * z + e[13];
            w = 0; c.vis[i] = 0;
            if (cw > 0.05 && Math.abs(px) < cw * lim && Math.abs(py) < cw * lim) {
              var dx = cx - x, dy = cy - y, dz = cz - z, d2 = dx * dx + dy * dy + dz * dz, dl = Math.sqrt(d2);
              var cs = (dx * c.cnrm[o] + dy * c.cnrm[o + 1] + dz * c.cnrm[o + 2]) / dl * c.flip;
              // open charts: foreshortening. closed charts: the near side counts in full (flat planoids must
              // stay small all the way to the silhouette), the hidden far side keeps a thin share
              if (c.closed) { c.vis[i] = cs > -0.15 ? 1 : 0; cs = c.vis[i] ? 1 : c.alpha ? 0.6 : st.cull ? 0 : 0.3; }      // culled: the far side asks for nothing (a see-through chart shows it)
              else { cs = Math.max(0.25, Math.abs(cs)); c.vis[i] = 1; }
              if (win && (Math.abs(px) > cw || Math.abs(py) > cw)) c.vis[i] = 0;       // window: "in view" is the image itself, not its buffer
              w = c.carea[i] * cs / d2 * (win ? 1 : c.gain);                          // window: the whole budget goes by area on screen
              if (cw < z0) z0 = cw; if (cw > z1) z1 = cw;
            }
            c.w[i] = w; totW += w;
          }
        });
        // size every tile's lattice; rounding to na x nb drifts off the budget, so a second pass rescales
        var total = sizeTiles(1, true);
        if (Math.abs(total - goal) > 0.001 * goal) sizeTiles(goal / total, false);
        rdV.set(T('{n} %', { n: Math.round(inView * 100) }));
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
              imp = win ? c.w[i] / (totW || 1) : st.adaptive ? FLOOR * c.carea[i] / totA + (1 - FLOOR) * c.w[i] / (totW || 1) : c.carea[i] / totA;
              share += imp; mu[a] += imp; mv[b] += imp; lu += imp * c.clu[i] / c.tu; lv += imp * c.clv[i] / c.tv;
              if (first && c.vis[i]) inView += imp;
            }
            if (win && !share) { tl.na = tl.nb = 0; return; }                         // window: a tile outside it gets nothing
            var N = Math.max(4, goal * share * scale), ratio = lu / (lv || 1e-12);   // aim for square cells in world space
            // curved closed charts keep a minimum count around the bend so flat planoids can follow it
            tl.nb = Math.max(c.minB, Math.round(Math.sqrt(N / ratio))); tl.na = Math.max(c.minA, Math.round(N / tl.nb));
            for (a = 0; a < cu; a++) mu[a] += share / cu * (win ? 0.12 : 0.3);        // bounded contrast inside a tile:
            for (b = 0; b < cv; b++) mv[b] += share / cv * (win ? 0.12 : 0.3);        // no giant cells on the far side
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
      /* the tool layer's marks on the Gaussians: a selection tints towards the surface colour, a hold towards the control
         points' (on a dark page their cyan and amber, as they have always been written) */
      var SELT = LIGHT ? G.rgb(C.surf) : [0.31, 0.76, 0.97], HOLDT = LIGHT ? G.rgb(C.ctrl) : [1.0, 0.65, 0.15];
      function emit() {
        allocate();
        var n = 0, eye = { x: cam.position.x, y: cam.position.y, z: cam.position.z, flip: 1, min: -0.12 }, thin = false, track = !!st.sc.hands || selecting();
        charts.forEach(function (c, ci) {
          // the selection and the holds are tinted on the `hands` scene; while a material goes on the selection, the material shows it
          var rgb = G.rgb(c.hex), n0 = n, i, AL = c.alpha, SE = !!st.sc.hands && c.sel, PN = !!st.sc.hands && c.pin, p = c.patch, K = field.color;
          if (c.off) { c.count = 0; c.lab.textContent = T('put away'); return; }
          eye.flip = c.flip;
          c.tiles.forEach(function (tl) {
            if (tl.na * tl.nb > CAP || (!tl.na && st.win)) return;     // (window: a tile outside the view holds no lattice)
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
              if (SE && SE[q] > 0.5) { o = k * 3; K[o] = K[o] * 0.5 + SELT[0] * 0.5; K[o + 1] = K[o + 1] * 0.5 + SELT[1] * 0.5; K[o + 2] = K[o + 2] * 0.5 + SELT[2] * 0.5; }   // selected: towards the surface colour
              if (PN && (s = PN[q] * 0.5) > 0) { o = k * 3; K[o] += (HOLDT[0] - K[o]) * s; K[o + 1] += (HOLDT[1] - K[o + 1]) * s; K[o + 2] += (HOLDT[2] - K[o + 2]) * s; }   // held: towards the control points' colour
            }
            if (AL) thin = true;
          });
          c.count = n - n0;
          for (i = n0 * 3; i < n * 3; i += 3) { field.aux[i] = rgb[0]; field.aux[i + 1] = rgb[1]; field.aux[i + 2] = rgb[2]; }
          c.lab.textContent = T('{n} G', { n: c.count.toLocaleString('en-US') });
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

      /* ---------- scenes `mesh` and `stitch`: the same Gaussians meshed three ways, at the same budget ----------
         OURS, THE LATTICE: every chart is sampled on one even lattice, as many points in all as the budget (shared over the
         charts by area x gain, the sampler's own share, and laid so that the cells are square on the surface), and the
         Gaussians sit on its points. A 2 x 2 window running over it turns each stop into two triangles. A Gaussian goes
         when its point has been stitched, so the instances are written in the reverse of the window's order: what is
         left to stitch is always the first part of the buffer.
         A LIST (talk/meshlist.js): the same number of Gaussians as an EMULATED unstructured list, meshed the way a list
         has to be: a hash grid and a k-nearest search for every Gaussian, a plane through its neighbours, a local
         triangulation in that plane, and a vote. Its triangles appear in the order the list made them, which is no
         order at all.
         TSDF FUSION (talk/meshtsdf.js; without that file there are two techniques): the list's Gaussians again, the way
         most 3DGS papers get their mesh: a depth map of them from each of the scene's capture cameras (def.rig), fused
         into a dense voxel grid over their box, and the zero level set extracted by marching cubes. The voxel is the
         Gaussians' spacing unless the grid would then pass VOXELS voxels; then it is as large as that cap makes it, and
         the board says so. Its triangles appear slab by slab, from the table up.
         The three are JOBS that run side by side, a few milliseconds a frame each (SLICE), timed inside their own
         steps. One audit grades all three: bridges and non-manifold edges in red, the rims of holes in faint red (a
         vertex TSDF made gets the address of the lattice sample nearest to it first); and each mesh's vertices are
         measured against the spline charts. Every number on the board is measured on the page.
         Scene `mesh` shows the three side by side: three viewports of the one renderer, one camera, one orbit, each
         with its own Gaussians (those not meshed yet), triangles and defects, above the board. Scene `stitch` shows
         one of them on the whole stage: the lattice, or the one picked with `mesh from`; it has the downloads. Between
         the two scenes nothing is rebuilt or measured again. A new budget, or the ground put away, rebuilds all three
         once the slider has rested for REST seconds. The stitched lattice downloads as PLY, GLB, OBJ or STL
         (talk/meshio.js), in metres where the scene has a physical scale. ---------- */
      var ML = KT.meshList, TS = KT.meshTsdf, IO = KT.meshio, mesh = null;
      var MESH_T = 12, LIST_T = 6, TSDF_T = 6, REST = 0.45;                  // seconds
      /* the meshing gets SLICE ms of every frame: 12 at 60 frames a second, up to 30 where the drawing itself is slow (so
         that the jobs still get their share of the time); opts.slice fixes it (tests on a software renderer) */
      var SLICE = +env.opts.slice || 12, SLICE0 = +env.opts.slice || 0;
      var VOXELS = +env.opts.voxels || (small ? 1.5e6 : 4e6);                // the most voxels a TSDF grid gets on this screen
      var KINDS = TS ? ['lattice', 'list', 'tsdf'] : ['lattice', 'list'];
      var KNAME = { lattice: T('ours · the lattice'), list: T('a list · emulated'), tsdf: T('TSDF fusion') };            // over a viewport (text)
      var KHEAD = { lattice: T('the lattice <i>ours</i>'), list: T('a list <i>emulated</i>'), tsdf: T('TSDF fusion') };  // a column of the board (HTML)
      var board = ui.el('div', 'mf-board'); root.appendChild(board);
      /* compare view: a title and a status line over each viewport, a rule between two */
      var tags = KINDS.map(function (k) {
        var e = ui.el('div', 'mf-vp mf-vp-' + k); e.appendChild(ui.el('b', '', KNAME[k])); e.appendChild(ui.el('span')); root.appendChild(e); return e;
      });
      var bars = KINDS.slice(1).map(function () { var e = ui.el('i', 'mf-bar'); root.appendChild(e); return e; });
      rdG.el.classList.add('mf-one'); rdP.el.classList.add('mf-deck1'); segV.el.classList.add('mf-deck1');   // rows the compare view has no use for
      var segH = null, segF = null, ioNote = null;
      if (ML) {
        segH = ui.seg(panel, { label: T('mesh from'), value: 'lattice',
          options: [{ v: 'list', label: T('a list') }, { v: 'lattice', label: T('the lattice') }].concat(TS ? [{ v: 'tsdf', label: T('TSDF') }] : []),
          onChange: function (v) { if (mesh) showPhase(v); } });
        segH.el.classList.add('mf-tri');
      }
      if (IO) {
        var io = ui.el('div', 'mf-tri mf-io'); panel.appendChild(io);
        segF = ui.seg(io, { label: T('export'), value: 'glb', options: IO.formats.map(function (f) { return { v: f.v, label: f.v }; }) });
        ui.button(io, { label: T('download the lattice mesh'), kind: 'go', onClick: download });
        ioNote = ui.el('div', 'mf-io-note'); io.appendChild(ioNote);
      }
      function remesh() { if (mesh) mesh.due = mesh.age + REST; }    // the budget moved: rebuild once it rests
      /* what a mesh is built from: the scene, the budget and the charts that are on */
      function meshKey() { return st.scn + ' ' + Math.min(st.budget, CAP) + ' ' + charts.map(function (c) { return c.off ? 0 : 1; }).join(''); }
      /* a scene with `mesh` builds the three; `cmp` shows them side by side, else one fills the stage (`from`, or the
         lattice). Between such scenes on the same Gaussians nothing is rebuilt or measured again: the play starts over */
      function syncMesh() {
        if (!st.sc.mesh) { dropMesh(); return; }
        if (mesh && !mesh.due && mesh.key === meshKey()) play();
        else { if (mesh) mesh.due = 0; buildMesh(); }          // a new scene starts where it starts, even with a budget change pending
      }
      function dropField(f) {
        if (!f) return;
        var k = f.mesh.userData.ktSplat;
        scene.remove(f.mesh); f.mesh.geometry.dispose(); [k.plain, k.depth, k.core, k.accum].forEach(function (m) { if (m) m.dispose(); });
      }
      function dropMesh() {
        if (!mesh) return;
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].concat(mesh.lobj || [], mesh.tobj || []).forEach(function (o) { scene.remove(o); o.material.dispose(); o.geometry.dispose(); });
        dropField(mesh.fieldL); dropField(mesh.fieldT);
        if (mesh.tjob) mesh.tjob.free();                        // the volume, if the fusion was still under way
        mesh = null; field.mesh.visible = true;
        root.classList.remove('mf-mesh'); root.classList.remove('mf-cmp'); need = true;
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
        var geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('aCol', new THREE.BufferAttribute(col, 3));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        return new THREE.Mesh(geo, new THREE.ShaderMaterial({ side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
          vertexShader: 'attribute vec3 aCol; varying vec3 vC; void main(){ vC = aCol; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
          fragmentShader: 'varying vec3 vC; void main(){ gl_FragColor = vec4(vC, 1.0); }' }));
      }
      function edgeLines(E, P, colour, opacity) {    // edges as (a, b, first triangle) -> line segments, in that order
        var n = E.length / 3, arr = new Float32Array(n * 6), k, q;
        for (k = 0; k < n; k++) for (q = 0; q < 3; q++) { arr[k * 6 + q] = P[E[k * 3] * 3 + q]; arr[k * 6 + 3 + q] = P[E[k * 3 + 1] * 3 + q]; }
        var g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3));
        var o = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: opacity })); o.renderOrder = 4;
        return o;
      }
      function upTo(E, q) {                         // how many of the edges E come with the first q triangles
        var lo = 0, hi = E.length / 3, m;
        while (lo < hi) { m = (lo + hi) >> 1; if (E[m * 3 + 2] < q) lo = m + 1; else hi = m; }
        return lo;
      }
      function buildMesh() {
        var was = mesh && mesh.due ? mesh.phase : null;        // a rebuild for a new budget keeps what was picked
        dropMesh();
        if (!ML) return;
        var parts = sizeLattices().sort(function (a, b) { return a.na * a.nb - b.na * b.nb; });   // the window starts on the smallest chart
        if (!parts.length) return;
        var lat = ML.lattice(parts), V = lat.V, w = def.views.wide;
        var pos = new Float32Array(V * 3), col = new Float32Array(V * 3), nrm = new Float32Array(V * 3);
        var eye = [w.tx + w.r * Math.sin(w.ph) * Math.sin(w.th), w.ty + w.r * Math.cos(w.ph), w.tz + w.r * Math.sin(w.ph) * Math.cos(w.th)];
        var lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9], solo = parts.some(function (m) { return !m.c.ground; });
        parts.forEach(function (m) {
          var k = m.na * m.nb, i, o, s = 0, q;
          evalLattice(m, false); pos.set(sg.pos.subarray(0, k * 3), m.off * 3); col.set(sg.col.subarray(0, k * 3), m.off * 3);
          // the surface's own normal, S_u x S_v: out of a closed chart, towards the scene's camera on an open one
          for (i = 0; i < k; i++) {
            o = i * 3; var ax = sg.su[o], ay = sg.su[o + 1], az = sg.su[o + 2], bx = sg.sv[o], by = sg.sv[o + 1], bz = sg.sv[o + 2];
            var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, l = Math.hypot(nx, ny, nz) || 1;
            nrm[(m.off + i) * 3] = nx / l; nrm[(m.off + i) * 3 + 1] = ny / l; nrm[(m.off + i) * 3 + 2] = nz / l;
            s += nx * (eye[0] - sg.pos[o]) + ny * (eye[1] - sg.pos[o + 1]) + nz * (eye[2] - sg.pos[o + 2]);
            if (!m.c.ground || !solo) for (q = 0; q < 3; q++) { if (sg.pos[o + q] < lo[q]) lo[q] = sg.pos[o + q]; if (sg.pos[o + q] > hi[q]) hi[q] = sg.pos[o + q]; }
          }
          m.sign = m.c.closed ? m.c.flip : s < 0 ? -1 : 1;
          if (m.sign < 0) for (i = m.off * 3; i < (m.off + k) * 3; i++) nrm[i] = -nrm[i];
        });
        var tri = surfaceMesh(pos, col, lat.tri), wgeo = new THREE.BufferGeometry(), kgeo = new THREE.BufferGeometry();
        wgeo.setAttribute('position', tri.geometry.attributes.position); wgeo.setIndex(new THREE.BufferAttribute(lat.edge, 1));
        kgeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
        mesh = { parts: parts, lat: lat, V: V, Q: lat.Q, pos: pos, col: col, nrm: nrm, age: 0, due: 0, done: false, busy: true, cmp: false, tiny: false, rects: null, tagH: 0, left: 0, painted: -1, bh: 0,
          phase: was || 'lattice', key: meshKey(), hex: charts.map(function (c) { return G.rgb(c.hex); }), tri: tri,
          // the object's box (every chart but the ground): what a viewport of the compare view frames
          box: { c: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2], h: [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2] },
          clock: { lattice: 0, list: 0, tsdf: 0 }, fld: { lattice: field, list: field, tsdf: field }, note: { lattice: '', list: '', tsdf: '' },
          aud: { lattice: null, list: null, tsdf: null }, dist: { lattice: null, list: null, tsdf: null }, queue: [],
          // every edge faintly, the ones just stitched brightly, and the window itself: its four samples and a dot
          wire: new THREE.LineSegments(wgeo, new THREE.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.1 })),
          fresh: new THREE.LineSegments(wgeo.clone(), new THREE.LineBasicMaterial({ color: C.text, transparent: true, opacity: 0.6 })),
          win: new THREE.LineLoop(kgeo, new THREE.LineBasicMaterial({ color: C.text, depthTest: false })),
          dot: new THREE.Points(kgeo, new THREE.PointsMaterial({ color: C.text, size: 5, sizeAttenuation: false, depthTest: false })) };
        [mesh.tri, mesh.wire, mesh.fresh, mesh.win, mesh.dot].forEach(function (o) { o.frustumCulled = false; scene.add(o); });
        mesh.win.renderOrder = mesh.dot.renderOrder = 6; root.classList.add('mf-mesh');
        // the list (the same samples as a 3DGS list would hold them, emulated over a few frames; then meshed, and fused), and
        // what grades the lattice: its audit, and how far its points are from the charts
        mesh.L = null; mesh.job = null; mesh.tjob = null; mesh.tm = null; mesh.ejob = ML.emulator(parts, 7);
        mesh.sam = { n: V, pos: pos, chart: lat.chart, part: lat.part, ca: lat.ca, cb: lat.cb, grid: lat.grid };
        grade(ML.auditor(lat.tri, lat.nTri, lat), function (j) { mesh.aud.lattice = j.stats; });
        grade(ML.gauge(pos, V, mesh.sam), function (j) { mesh.dist.lattice = j.stats; });
        play();
        paintIO();
      }
      function grade(job, then) { mesh.queue.push({ job: job, then: then }); }     // grading waits in line: one job at a time
      /* (re)start the show of the scene: the three side by side, or one on the whole stage */
      function play() {
        var M = mesh, cmp = !!st.sc.cmp && KINDS.length > 1;
        if (!M) return;
        M.cmp = cmp; root.classList.toggle('mf-cmp', cmp); M.rects = null;
        if (cmp) {
          KINDS.forEach(function (k) { M.clock[k] = 0; });
          M.fld.lattice = field; M.fld.list = M.fld.tsdf = null; writeLattice(field); fields();
          layoutMesh(); paintBoard(true);
        } else showPhase(st.sc.from || M.phase);
      }
      /* the compare view's own Gaussians: the list twice more, as the list meshes them and as the level set passes them */
      function fields() {
        var M = mesh;
        if (!M.cmp || !M.L) return;
        if (!M.fieldL) { M.fieldL = G.splats(M.L.n); scene.add(M.fieldL.mesh); writeList(M.fieldL); }
        M.fld.list = M.fieldL;
        if (M.tjob) { if (!M.fieldT) { M.fieldT = G.splats(M.L.n); scene.add(M.fieldT.mesh); writeSlabs(M.fieldT); } M.fld.tsdf = M.fieldT; }
      }
      function mirror(f) {                           // an extra field is drawn like the main one
        var a = field.uniforms, b = f.uniforms;
        ['uGain', 'uRim', 'uFlat', 'uMode', 'uFade'].forEach(function (k) { b[k].value = a[k].value; });
        ['uSplit', 'uRes', 'uBand', 'uDepth'].forEach(function (k) { b[k].value.copy(a[k].value); });
      }
      function sealed(f, n, kind) { if (f.alpha) f.alpha.fill(1, 0, n); f.mesh.userData.ktSplat.thin = false; f.commit(n); f.setCount(n); f.kind = kind; }
      function writeLattice(f) {
        var n = 0;
        mesh.parts.slice().reverse().forEach(function (m) {
          var d = evalLattice(m, true), n0 = n, rgb = G.rgb(m.c.hex), i;
          n = G.writePlanoids(f, n, sg, m.na, m.nb, d.du, d.dv, st.k, null, null);
          for (i = n0 * 3; i < n * 3; i += 3) { f.aux[i] = rgb[0]; f.aux[i + 1] = rgb[1]; f.aux[i + 2] = rgb[2]; }
          m.c.lab.textContent = T('{n} G', { n: (n - n0).toLocaleString('en-US') });
        });
        sealed(f, n, 'lattice');
      }
      function putList(f, d, s) {                    // list entry s into instance d of field f
        var L = mesh.L, rgb = mesh.hex[L.chart[s]], q;
        s *= 3; d *= 3;
        for (q = 0; q < 3; q++) { f.center[d + q] = L.pos[s + q]; f.axU[d + q] = L.axU[s + q]; f.axV[d + q] = L.axV[s + q]; f.color[d + q] = L.col[s + q]; f.aux[d + q] = rgb[q]; }
      }
      function writeList(f) {                        // reversed as well: the list entries already meshed are the end of the buffer
        var n = mesh.L.n, k;
        for (k = 0; k < n; k++) putList(f, k, n - 1 - k);
        sealed(f, n, 'list');
      }
      /* the list once more, in the order the level set passes it: the grid's slabs from the top down, so that the
         Gaussians of the slabs already extracted are the end of the buffer. M.under[k] = Gaussians below slab k */
      function writeSlabs(f) {
        var M = mesh, L = M.L, n = L.n, J = M.tjob, ax = J.slabAxis, ns = Math.max(1, J.grid.n[ax] - 1), y0 = J.grid.lo[ax], h = J.grid.h, cnt = new Int32Array(ns + 1), at = new Int32Array(ns), k, s;
        function slab(i) { return Math.min(ns - 1, Math.max(0, Math.floor((L.pos[i * 3 + ax] - y0) / h))); }
        for (k = 0; k < n; k++) cnt[slab(k) + 1]++;
        for (k = 0; k < ns; k++) cnt[k + 1] += cnt[k];            // cnt[k] = Gaussians below slab k
        for (k = 0; k < n; k++) { s = slab(k); putList(f, n - 1 - (cnt[s] + at[s]++), k); }
        M.under = cnt; sealed(f, n, 'slabs');
      }
      /* one technique on the whole stage (scene `stitch`, and the picker) */
      function showPhase(p) {
        var M = mesh;
        if (!M) return;
        if (p === 'tsdf' && !TS) p = 'lattice';
        if (M.cmp) { M.cmp = false; root.classList.remove('mf-cmp'); M.rects = null; layoutMesh(); }
        M.phase = p; M.clock[p] = 0; M.done = false;
        KINDS.forEach(function (k) { M.fld[k] = field; });
        if (segH) segH.set(p);
        if (p === 'lattice') writeLattice(field); else if (!M.L) { field.setCount(0); field.kind = ''; } else if (p === 'list') writeList(field); else writeSlabs(field);
        rdM.el.firstChild.textContent = p === 'lattice' ? T('triangles stitched') : T('triangles made');
        step[p](0);
        paintBoard(true);
      }
      function listReady() {                         // the list's mesh is done: its triangles, red edges and hole rims, to be revealed
        var M = mesh, J = M.job;
        M.ltri = surfaceMesh(M.L.pos, M.L.col, J.tri);
        M.lbad = edgeLines(J.stats.bad, M.L.pos, C.err, 0.95); M.lhole = edgeLines(J.stats.hole, M.L.pos, C.err, 0.45);
        M.lobj = [M.ltri, M.lbad, M.lhole];
        M.lobj.forEach(function (o) { o.frustumCulled = false; o.visible = false; o.geometry.setDrawRange(0, 0); scene.add(o); });
        M.aud.list = J.stats;
        grade(ML.gauge(M.L.pos, M.L.n, M.L), function (j) { M.dist.list = j.stats; });
      }
      /* the fusion's mesh is out: its vertices get addresses (the nearest lattice sample), then the same audit and the
         same measure as the others; then it is revealed */
      function tsdfOut() {
        var M = mesh, tm = M.tjob.mesh;
        grade(ML.addresser(tm.pos, tm.nV, M.sam), function (a) {
          var rim = Math.ceil(2 * M.tjob.grid.h / (a.cell / 2));          // two voxels, in lattice cells: what counts as a chart's rim for a mesh this coarse
          M.tjob.near = a.M;
          grade(ML.auditor(tm.tri, tm.nTri, a.M, { rim: rim }), function (j) {
            M.aud.tsdf = j.stats; M.tm = tm;
            M.ttri = surfaceMesh(tm.pos, tm.col, tm.tri);
            M.tbad = edgeLines(j.stats.bad, tm.pos, C.err, 0.95); M.thole = edgeLines(j.stats.hole, tm.pos, C.err, 0.45);
            M.tobj = [M.ttri, M.tbad, M.thole];
            M.tobj.forEach(function (o) { o.frustumCulled = false; o.visible = false; o.geometry.setDrawRange(0, 0); scene.add(o); });
            grade(ML.gauge(tm.pos, tm.nV, a.M), function (g) { M.dist.tsdf = g.stats; });
          });
        });
      }
      function msText(x) { return x < 1 ? '< 1 ms' : Math.round(x).toLocaleString('en-US') + ' ms'; }
      function spent(k) {                            // the milliseconds a technique has worked so far, inside its own steps
        var M = mesh, E = M.job ? M.job.ms : {}, F = M.tjob ? M.tjob.ms : {};
        return k === 'lattice' ? M.lat.ms : k === 'list' ? (E.index || 0) + (E.search || 0) + (E.normal || 0) + (E.fan || 0) + (E.vote || 0) : F.depth + F.fuse + F.extract;
      }
      function stepMesh(dt) {
        var M = mesh, q;
        M.age += dt;
        if (!SLICE0) SLICE = KT.clamp(600 * dt, 12, 30);
        if (M.due && M.age >= M.due) { buildMesh(); return; }        // a new budget, and the slider has rested
        // the work of this frame: emulate the list; then the list's mesher and the fusion side by side, SLICE ms between
        // them; and one grading job, in what is left (a short slice while a mesher still runs)
        if (!M.L) {
          if (M.ejob.step(SLICE)) {
            M.L = M.ejob.list; M.job = ML.mesher(M.L);
            if (TS) M.tjob = TS.fuser(M.L, { cams: def.rig.cams, spacing: ML.gap(M.L), maxVoxels: VOXELS });
            if (M.cmp) fields(); else if (M.phase !== 'lattice') showPhase(M.phase);
          }
        } else {
          var a = !M.job.done, b = !!M.tjob && !M.tjob.done, each = a && b ? SLICE * 0.6 : SLICE;
          if (a && M.job.step(each)) listReady();
          if (b && M.tjob.step(each)) tsdfOut();
        }
        q = M.queue[0];
        if (q && q.job.step(M.L && M.job.done && (!M.tjob || M.tjob.done) ? SLICE : 3)) { M.queue.shift(); q.then(q.job); paintBoard(true); }
        M.tiny = M.cmp && root.classList.contains('mf-narrow');        // a narrow stage: a tag says the time only, the board has the rest
        (M.cmp ? KINDS : [M.phase]).forEach(function (k) { step[k](dt); });
        if (M.cmp) {
          KINDS.forEach(function (k, i) { if (tags[i].lastChild.textContent !== M.note[k]) tags[i].lastChild.textContent = M.note[k]; });
          if (!orbit.touched && M.rects) {            // the object, as large as a viewport holds it under its tag
            var B = M.box, asp = M.rects[0][2] / M.rects[0][3], o = orbit.goal, tan = Math.tan(18 * Math.PI / 180), tg = Math.min(0.3, M.tagH / M.rects[0][3]);
            o.r = Math.max((B.h[1] * 1.06 + 0.1) / (1 - tg), (B.h[0] * 1.08 + 0.1) / asp) / tan + B.h[2];
            o.tx = B.c[0]; o.ty = B.c[1] + tg * (o.r - B.h[2]) * tan; o.tz = B.c[2];
          }
        } else {
          rdG.set(M.left.toLocaleString('en-US')); rdM.set(M.note[M.phase]);
        }
        paintBoard(false);
      }
      var step = {
        lattice: function (dt) {
          var M = mesh, f = M.fld.lattice;
          M.clock.lattice += dt;
          var g = Math.min(1, M.clock.lattice / MESH_T), q = Math.min(M.Q, Math.floor(g * g * M.Q)), left = M.V, m, i, k;
          for (i = 0; i < M.parts.length; i++) {
            m = M.parts[i]; k = q - m.q0;
            if (k >= m.qa * m.qb) { left -= m.na * m.nb; continue; }
            if (k >= 0) {                                   // the window is on this chart, at column a of row b
              var a = k % m.qa, b = Math.floor(k / m.qa), W = M.win.geometry.attributes.position, P = M.pos;
              left -= b * m.na + a;
              [m.off + b * m.na + a, m.off + b * m.na + (a + 1) % m.na, m.off + ((b + 1) % m.nb) * m.na + (a + 1) % m.na, m.off + ((b + 1) % m.nb) * m.na + a]
                .forEach(function (v, j) { W.setXYZ(j, P[v * 3], P[v * 3 + 1], P[v * 3 + 2]); });
              W.needsUpdate = true;
            }
            break;
          }
          var band = Math.min(q, Math.ceil(M.Q * 0.035));
          M.tri.geometry.setDrawRange(0, q * 6); M.wire.geometry.setDrawRange(0, q * 6); M.fresh.geometry.setDrawRange((q - band) * 6, band * 6);
          M.busy = q < M.Q; f.setCount(left); M.left = left;
          M.note.lattice = M.tiny ? msText(M.lat.ms) : M.cmp ? T('{n} triangles · {t}', { n: (M.Q * 2).toLocaleString('en-US'), t: msText(M.lat.ms) })
                                 : (q * 2).toLocaleString('en-US') + ' / ' + (M.Q * 2).toLocaleString('en-US');
          if (q === M.Q && !M.done && !M.cmp) { M.done = true; if (!orbit.touched) orbit.to(view('close')); }   // finished: go and look at the triangles
        },
        list: function (dt) {
          var M = mesh, J = M.job, n = M.L ? M.L.n : 0, f = M.fld.list, pc;
          if (!J || !J.done) {
            if (f) f.setCount(f.kind === 'list' ? n : 0);
            M.left = n; pc = Math.round((J ? J.progress : M.ejob.progress) * 100);
            M.note.list = M.tiny || (J && !M.cmp) ? T('{n} %', { n: pc }) : !J ? T('emulating the list · {n} %', { n: pc }) :
              J.phase === 'index' || J.phase === 'search' ? T('searching · {p} % · {t}', { p: pc, t: msText(spent('list')) }) : T('triangulating · {p} % · {t}', { p: pc, t: msText(spent('list')) });
            return;
          }
          M.clock.list += dt;
          var g = Math.min(1, M.clock.list / LIST_T), q = Math.min(J.nTri, Math.floor(g * g * J.nTri)), gone = g >= 1 ? n : q ? J.tri[(q - 1) * 3] + 1 : 0;
          M.ltri.geometry.setDrawRange(0, q * 3);
          M.lbad.geometry.setDrawRange(0, upTo(J.stats.bad, q) * 2); M.lhole.geometry.setDrawRange(0, upTo(J.stats.hole, q) * 2);
          if (f) f.setCount(n - gone);                   // each triangle is listed by the first Gaussian that proposed it: those before it are done
          M.left = n - gone;
          M.note.list = M.tiny ? msText(spent('list')) : M.cmp ? T('{n} triangles · {t}', { n: J.nTri.toLocaleString('en-US'), t: msText(spent('list')) })
                              : q.toLocaleString('en-US') + ' / ' + J.nTri.toLocaleString('en-US');
        },
        tsdf: function (dt) {
          var M = mesh, J = M.tjob, n = M.L ? M.L.n : 0, f = M.fld.tsdf, tm = M.tm;
          if (!tm) {
            if (f) f.setCount(f.kind === 'slabs' ? n : 0);
            M.left = n;
            M.note.tsdf = M.tiny || (J && !M.cmp) ? T('{n} %', { n: Math.round((J || M.ejob).progress * 100) }) : !J ? T('emulating the list · {n} %', { n: Math.round(M.ejob.progress * 100) }) :
              J.stage === 'depth' || J.stage === 'fuse' ? T('depth map {a} of {b} · {t}', { a: Math.min(J.V, J.view + 1), b: J.V, t: msText(spent('tsdf')) }) :
              J.done ? T('grading · {t}', { t: msText(spent('tsdf')) }) : T('extracting · {t}', { t: msText(spent('tsdf')) });
            return;
          }
          M.clock.tsdf += dt;
          var ns = tm.slab.length, g = Math.min(1, M.clock.tsdf / TSDF_T), kf = Math.min(ns, Math.floor(g * ns + 1e-9)), q = kf ? tm.slab[kf - 1] : 0, gone = g >= 1 ? n : M.under ? M.under[kf] : 0;
          M.ttri.geometry.setDrawRange(0, q * 3);
          M.tbad.geometry.setDrawRange(0, upTo(M.aud.tsdf.bad, q) * 2); M.thole.geometry.setDrawRange(0, upTo(M.aud.tsdf.hole, q) * 2);
          if (f) f.setCount(n - gone);                   // the Gaussians of the slabs already extracted are done
          M.left = n - gone;
          M.note.tsdf = M.tiny ? msText(spent('tsdf')) : M.cmp ? T('{n} triangles · {t}', { n: tm.nTri.toLocaleString('en-US'), t: msText(spent('tsdf')) })
                              : q.toLocaleString('en-US') + ' / ' + tm.nTri.toLocaleString('en-US');
        }
      };
      /* what a view shows of the scene: one technique's Gaussians, triangles and marks */
      function showKind(k) {
        var M = mesh, f = M.fld[k];
        M.tri.visible = k === 'lattice'; M.wire.visible = k === 'lattice' && !M.cmp;      // side by side, the three meshes are drawn alike: no net over ours
        [M.fresh, M.win, M.dot].forEach(function (o) { o.visible = k === 'lattice' && M.busy; });
        (M.lobj || []).forEach(function (o) { o.visible = k === 'list'; });
        (M.tobj || []).forEach(function (o) { o.visible = k === 'tsdf'; });
        [field, M.fieldL, M.fieldT].forEach(function (x) { if (x) x.mesh.visible = x === f; });
        if (f && f !== field) mirror(f);
      }
      /* the frame: one view of the stage, or the three viewports of the compare view, each with the camera's aspect its own */
      function drawMesh() {
        var M = mesh, rr = R.renderer, q = M.cmp ? M.rects : null, i;
        if (!q) { showKind(M.phase); G.draw(R, scene, cam); return; }
        rr.setScissorTest(false); rr.setViewport(0, 0, size.w, size.h); rr.clear();
        rr.setScissorTest(true);
        for (i = 0; i < KINDS.length; i++) {
          rr.setViewport(q[i][0], q[i][1], q[i][2], q[i][3]); rr.setScissor(q[i][0], q[i][1], q[i][2], q[i][3]);
          cam.aspect = q[i][2] / q[i][3]; cam.updateProjectionMatrix();
          showKind(KINDS[i]); G.draw(R, scene, cam);
        }
        rr.setScissorTest(false); rr.setViewport(0, 0, size.w, size.h);
        cam.aspect = size.w / size.h; cam.updateProjectionMatrix();
      }
      /* where the viewports of the compare view go: side by side, under what lies on top of the stage (the caption; the
         panel, or the controls button) and above the board, which runs along the bottom. On a wide stage (the lab page)
         they keep to the left of the panel instead. Rectangles in CSS pixels, y from the bottom, as three.js counts */
      function layoutMesh() {
        var M = mesh, w = size.w, h = size.h, n = KINDS.length;
        board.style.right = ''; root.classList.toggle('mf-narrow', w < 620);
        if (!M || !M.cmp || w < 8 || h < 8) return;
        var rr = root.getBoundingClientRect(), pr = panel.getBoundingClientRect(), cr = cap.getBoundingClientRect(), gear = root.querySelector('.kt-gear'), gr = gear ? gear.getBoundingClientRect() : null;
        var compact = root.classList.contains('kt-compact'), shown = !compact && pr.width > 0 && pr.height > 0;
        var side = shown && w / h >= 1.25 && (pr.left - rr.left - 20) / n >= 200, x1 = side ? pr.left - rr.left - 8 : w, pad = w < 620 ? 6 : 12;
        // what lies on top: the caption, and the panel (when the viewports are under it) or the folded stage's controls
        // button; the viewports begin under all of it, except that the small button only pushes down the tag below it
        var over = side ? null : shown ? pr : gr && gr.height ? gr : null, capB = cr.height ? cr.bottom - rr.top : 0;
        var top = Math.max(capB, over && shown ? over.bottom - rr.top : 0) + 6;
        if (side) board.style.right = (w - x1 + 8) + 'px';
        M.bh = board.offsetHeight;
        var vw = x1 / n, y1 = h - M.bh - pad - 6, vh = Math.max(60, y1 - top), i;
        M.rects = [];
        for (i = 0; i < n; i++) {
          var under = over && !shown && over.right - rr.left > i * vw && over.left - rr.left < (i + 1) * vw ? over.bottom - rr.top + 4 : 0;
          M.rects.push([i * vw, h - top - vh, vw, vh]);
          tags[i].style.left = (i * vw + 5) + 'px'; tags[i].style.top = Math.max(top, under) + 'px'; tags[i].style.maxWidth = (vw - 10) + 'px';
          if (i) { bars[i - 1].style.left = (i * vw) + 'px'; bars[i - 1].style.top = top + 'px'; bars[i - 1].style.height = vh + 'px'; }
        }
        M.tagH = tags[0].offsetHeight || 0;
      }
      /* ---------- the board: one column per technique; every number as it is measured ----------
         complexity as the code here has it (talk/meshlist.js, talk/meshtsdf.js), then what was measured: the time inside
         each job, the bytes of the arrays it allocated (working + mesh), the mesh, its defects, and how far its vertices
         are from the spline charts. Repainted a few times a second while the jobs run. */
      function paintBoard(now) {
        var M = mesh;
        if (!M || !ML) { board.innerHTML = ''; return; }
        if (!now && M.age - M.painted < 0.15) return;
        if (!now && M.settled) return;                               // nothing left to change
        M.painted = M.age;
        var J = M.job, F = M.tjob, tm = M.tm, sc = scaleOf(), done = { lattice: true, list: !!J && J.done, tsdf: !!F && F.done };
        function num(x) { return x.toLocaleString('en-US'); }
        function ms(x) { return x < 1 ? '&lt; 1 ms' : num(Math.round(x)) + ' ms'; }
        function mb(x) { return x <= 0 ? '0' : (x / 1048576).toFixed(x < 104857600 ? 1 : 0); }          // megabytes (2^20 bytes)
        function len(x) { return sc.mm ? (x * sc.mm).toFixed(2) + ' mm' : x.toFixed(4); }
        function td(x, cls) { return '<td' + (cls ? ' class="' + cls + '"' : '') + '>' + (x == null ? '…' : x) + '</td>'; }
        function row(name, fn, err) {
          return '<tr><th>' + name + '</th>' + KINDS.map(function (k) { var x = fn(k); return td(x == null ? null : typeof x === 'number' ? num(x) : x, err && x != null ? x > 0 ? 'bad' : 'ok' : ''); }).join('') + '</tr>';
        }
        function aud(key) { return function (k) { return M.aud[k] ? M.aud[k][key] : null; }; }
        var BY = { lattice: M.lat.bytes, list: J && J.bytes, tsdf: F && F.done ? { work: F.bytes.volume + F.bytes.maps + F.bytes.tables, mesh: F.bytes.mesh } : null };
        var BIGO = { time: { lattice: 'O(N)', list: 'O(N·k)', tsdf: 'O(V·(N+P+R³))' }, mem: { lattice: 'O(1)', list: 'O(N·k)', tsdf: 'O(R³+P)' } };
        var html = '<b>' + T('the same {n} Gaussians, meshed {m} ways', { n: num(M.V), m: KINDS.length }) + '</b><div><table><thead><tr><th></th>' +
          KINDS.map(function (k) { return '<th class="mf-h-' + k + (M.cmp || M.phase === k ? ' on' : '') + '">' + KHEAD[k] + '</th>'; }).join('') + '</tr></thead><tbody>' +
          row(T('time'), function (k) { return '<span dir="ltr">' + BIGO.time[k] + '</span>'; }) +
          row(T('time, measured'), function (k) {
            return done[k] ? ms(spent(k)) : k === 'list' ? (J ? T('{n} %', { n: Math.round(J.progress * 100) }) : null) : F ? T('{n} %', { n: Math.round(F.progress * 100) }) : null; }) +
          row(T('memory'), function (k) { return '<span dir="ltr">' + BIGO.mem[k] + '</span>'; }) +
          row(T('memory, arrays'), function (k) { return BY[k] ? '<span dir="ltr">' + mb(BY[k].work) + ' + ' + mb(BY[k].mesh) + ' MB</span>' : null; }) +
          row(T('triangles'), function (k) { return k === 'lattice' ? M.Q * 2 : k === 'list' ? (done.list ? J.nTri : null) : done.tsdf ? F.mesh.nTri : null; }) +
          row(T('bridges'), aud('bridges'), 1) + row(T('non-manifold edges'), aud('nonManifold'), 1) + row(T('holes'), aud('holes'), 1) +
          row(T('off surface, mean'), function (k) { return M.dist[k] ? '<span dir="ltr">' + len(M.dist[k].mean) + '</span>' : null; }) +
          row(T('95th percentile'), function (k) { return M.dist[k] ? '<span dir="ltr">' + len(M.dist[k].p95) + '</span>' : null; }) +
          '</tbody></table><p class="mf-key">' + (F ? T('<i></i>bridge or non-manifold edge <i class="f"></i>edge of a hole · level set by marching cubes')
                                                     : T('<i></i>bridge or non-manifold edge <i class="f"></i>edge of a hole · times measured here')) + '</p></div><div class="mf-notes kt-text">';
        // the symbols, once, with what they are at this budget
        html += '<p>' + T('<span dir="ltr"><b>N</b> = {N}</span> Gaussians · <span dir="ltr"><b>k</b> = {k}</span> neighbours each', { N: num(M.V), k: J ? J.k : 12 }) +
          (F ? ' · ' + T('<span dir="ltr"><b>V</b> = {V}</span> views · <span dir="ltr"><b>P</b> = {S} × {S}</span> pixels a view · <span dir="ltr"><b>R³</b> = {R}</span> voxels',
                         { V: F.V, S: F.S, R: F.grid.n.join(' × ') }) : '') + '</p>';
        if (F) html += '<p class="mf-grid">' + (F.grid.capped
          ? T('voxel {a} against {b} between Gaussians: the grid is capped at {cap} voxels, {want} would hold what the samples hold',
              { a: '<span dir="ltr">' + len(F.grid.h) + '</span>', b: '<span dir="ltr">' + len(F.spacing) + '</span>', cap: '<span dir="ltr">' + (VOXELS / 1e6) + ' M</span>', want: '<span dir="ltr">' + (F.grid.wanted.voxels / 1e6).toFixed(1) + ' M</span>' })
          : T('voxel {a}, the spacing of the Gaussians; the grid may take up to {cap} voxels', { a: '<span dir="ltr">' + len(F.grid.h) + '</span>', cap: '<span dir="ltr">' + (VOXELS / 1e6) + ' M</span>' })) + '</p>';
        html += '<p class="mf-x">' + T('time: inside each job, not the replay · memory: array bytes, working + mesh, not process memory · off surface: vertices to the spline charts; ours is on them by construction') + '</p>';
        if (F) html += '<p class="mf-x">' + T('production fusion hashes voxel blocks near the surface: memory about <span dir="ltr">R²</span>, not <span dir="ltr">R³</span>; a k-d tree searches in <span dir="ltr">O(N log N)</span>') + '</p>';
        board.innerHTML = html + '</div>';
        M.settled = done.list && (!TS || done.tsdf) && !M.queue.length;
        if (M.cmp && board.offsetHeight !== M.bh) layoutMesh();
      }
      /* ---------- the stitched lattice as a file: every chart in the mesh, in metres where the scene has a scale ---------- */
      function scaleOf() {
        var k = /^scan(\d*)$/.exec(st.scn), D = k && (k[1] ? (KT.dtuScans || {})[k[1]] : KT.dtu);
        return D && D.unit_mm ? { scale: D.unit_mm / 1000, unit: 'm', note: '1 scene unit = ' + D.unit_mm + ' mm', mm: D.unit_mm, tag: 'metres', scan: D.scan || k[1] }
                              : { scale: 1, unit: 'scene units', note: '', tag: 'scene-units', scan: null };
      }
      function paintIO() {
        if (!ioNote || !mesh) return;
        var s = scaleOf();
        ioNote.classList.remove('bad');
        ioNote.textContent = T('{parts} · {points} points · {triangles} triangles · {unit}', {
          parts: mesh.parts.map(function (m) { return m.c.label; }).join(' + '), points: mesh.V.toLocaleString('en-US'), triangles: (mesh.Q * 2).toLocaleString('en-US'),
          unit: s.scan ? T('in metres (1 scene unit = {mm} mm)', { mm: s.mm }) : T('in scene units') });
      }
      /* what is written into the file, and the file's name, stay English (KT.scenes[..].en): the writers keep comment
         lines 7-bit, and a name made of the scene's label must come out Latin on a page in any language */
      function sceneEn() { var S = KT.scenes[st.scn]; return S.en || S; }
      function meshFile(fmt) {
        var M = mesh, s = scaleOf(), S = sceneEn();
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
        var s = scaleOf(), S = sceneEn();
        return S.label.toLowerCase().replace(/[^a-z0-9]+/g, '-') + (s.scan ? '-dtu' + s.scan : '') + '-lattice-' + mesh.V + '-points-' + s.tag + '.' + fmt;
      }
      /* saved through KnotTalk.saveFile, looked up now: a page may have replaced it (talk/meshio.js) */
      function download() {
        if (!mesh || !IO) return;
        var fmt = segF.get(), name = fileName(fmt), b;
        ioNote.classList.remove('bad');
        try { b = IO.blob(meshFile(fmt)); } catch (e) { ioNote.textContent = T('could not write the file: {why}', { why: '' + e.message }); ioNote.classList.add('bad'); return; }
        ioNote.textContent = T('saving {name} …', { name: name });
        Promise.resolve().then(function () { return window.KnotTalk.saveFile(name, b); }).then(function () {
          ioNote.textContent = T('{name} · {size} MB', { name: name, size: (b.size / 1048576).toFixed(1) });
        }, function (e) {
          ioNote.textContent = T('not saved: {why}', { why: (e && e.message) || e || T('refused') }); ioNote.classList.add('bad');
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
      var ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), plane = new THREE.Plane(), hitP = new THREE.Vector3(), dragCP = null;
      function setNdc(e) {
        var r = R.canvas.getBoundingClientRect();
        ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1)); ray.setFromCamera(ndc, cam);
      }
      /* the control point under client (x, y), as an index into ptPos: the ray's point threshold for a
         mouse, a fingertip-sized screen radius for touch */
      function findPoint(x, y, touch) {
        if (!st.nets || !pts || (mesh && mesh.cmp)) return -1;       // the compare view has three pictures of the net: none is picked
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
        plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new THREE.Vector3()), hitP.set(P[o], P[o + 1], P[o + 2]));
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
        hint.textContent = sel ? T('select on the atlas or on the object: the material follows the selection') : st.sc.hint || '';
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
        badge.innerHTML = st.adaptive ? T('view-adaptive sampling <b class="on">ON</b>') : T('view-adaptive sampling <b class="off">OFF</b>');
      }
      /* the rim shading of what is on the stage (see LIGHT_RIM); a material that wipes in takes over from the one before it */
      function rim() {
        var base = st.sc.rim || 0.3;
        if (!LIGHT || st.sc.rim) return base;                  // a scene that asks for its own rim keeps it (lod: the fringes show the sampling)
        if (KT.scenes[st.scn] && KT.scenes[st.scn].light) return LIGHT_RIM;
        if (!st.sc.paint || !def || !def.materials) return base;
        function of(v) { var m = def.materials.filter(function (k) { return k.v === v; })[0]; return m && m.light ? LIGHT_RIM : base; }
        return of(st.from) + (of(st.mat) - of(st.from)) * KT.clamp(st.paint, 0, 1);
      }
      function paintCap() {
        var s = KT.scenes[st.scn];
        cap.innerHTML = '<b>' + s.label + '</b> · ' + s.suite + (s.note != null ? s.note && ' · ' + s.note : st.scn === 'studio' ? '' : ' · ' + T('procedural stand-in')) + '<br>' + st.sc.cap;
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
        st.adaptive = dflt !== 'even'; st.win = dflt === 'window'; paintMode(); paintBadge();
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
      var TEASE = 30.6, TEASE_BANDS = [{ v: 'normal', name: T('normals') }, { v: 'gauss', name: T('Gaussians') }, { v: 'splats', name: T('visual') }];
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
          if (st.abT > 3.4) { st.abT = 0; st.adaptive = !st.adaptive; paintMode(); paintBadge(); need = true; }
        }
        if (sc.breathe && !st.touchedEdit && !dragCP && (def.animate || def.edit)) {
          // the scene's own edit: a life of its own where the scene has one (and a repaint, wiped in), else one move eased in and out
          if (sc.paint && def.animate) st.paint = Math.min(1, st.paint + dt / 3.5);
          (def.animate ? def.animate(st.t, st.paint, st.mat, st.from, st.target, st.fromT) : def.edit(0.5 - 0.5 * Math.cos(st.t * 1.15))).forEach(coarse);
          if (st.nets) updateNets();
          need = true;
        }
        if (LIGHT) field.uniforms.uRim.value = rim();
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
        if (mesh) drawMesh(); else G.draw(R, scene, cam);
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
          // the compare view has its numbers over the viewports and on the board: the readouts, the picker and the export
          // leave the panel (and in the deck what does not bear on a mesh: the reading, the parameter count)
          '.kt-sim-manifold.mf-cmp :is(.mf-tri,.mf-one),.kt-sim-manifold.mf-cmp.kt-embed .mf-deck1{display:none}' +
          // scenes `mesh` and `stitch`: the board (one column per technique; bottom left, or along the bottom under the
          // viewports of the compare view, its notes beside the table where there is room) and the export rows of the panel
          '.kt-sim-manifold .mf-board{position:absolute;z-index:4;left:12px;bottom:12px;max-width:calc(100% - 24px);padding:9px 11px 8px;pointer-events:none;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);' +
          'font:500 11px/1.3 var(--f-mono);color:var(--dim)}.kt-sim-manifold:not(.mf-mesh) .mf-board,.kt-sim-manifold.kt-bare .mf-board{display:none}' +
          '.kt-sim-manifold .mf-board>b{display:block;color:var(--text);font-weight:600;margin-bottom:6px}' +
          '.kt-sim-manifold .mf-board table{border-collapse:collapse}' +
          '.kt-sim-manifold .mf-board th,.kt-sim-manifold .mf-board td{padding:2px 0 2px 16px;text-align:right;white-space:nowrap;font-weight:500;font-variant-numeric:tabular-nums}' +
          '.kt-sim-manifold .mf-board th:first-child{padding-left:0;text-align:left;color:var(--dim)}' +
          '.kt-sim-manifold .mf-board thead th{font:600 10px/1.2 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--mute);padding-bottom:4px}' +
          '.kt-sim-manifold .mf-board thead th.on{color:var(--text)}.kt-sim-manifold .mf-board thead i{font-style:normal;text-transform:none;letter-spacing:0;color:var(--mute)}' +
          '.kt-sim-manifold .mf-board thead th.mf-h-lattice.on,.kt-sim-manifold .mf-vp-lattice b{color:var(--surf)}' +
          '.kt-sim-manifold .mf-board td{color:var(--text)}.kt-sim-manifold .mf-board td.bad{color:var(--err)}.kt-sim-manifold .mf-board td.ok{color:var(--ok)}' +
          '.kt-sim-manifold .mf-notes{font-size:10px;line-height:1.4;color:var(--mute);min-width:0}.kt-sim-manifold .mf-notes p{margin:0 0 5px}' +
          '.kt-sim-manifold .mf-notes p:last-child{margin:0}.kt-sim-manifold .mf-notes b{color:var(--text);font-weight:600}' +
          '.kt-sim-manifold .mf-notes .mf-grid{color:var(--dim)}.kt-sim-manifold:not(.mf-cmp) .mf-notes{display:none}' +
          '.kt-sim-manifold .mf-key{margin:6px 0 0;font-size:10px;color:var(--mute)}' +
          '.kt-sim-manifold .mf-key i{display:inline-block;width:14px;height:2px;margin:0 5px 3px 0;vertical-align:middle;background:var(--err)}' +
          '.kt-sim-manifold .mf-key i.f{margin-left:8px;opacity:.45}' +
          '.kt-sim-manifold.mf-cmp .mf-board{right:12px;max-width:none;display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:22px;align-items:start}' +
          '.kt-sim-manifold.mf-cmp .mf-board>b{grid-column:1 / -1}.kt-sim-manifold.mf-cmp .mf-board :is(th,td){padding-top:1px;padding-bottom:1px;padding-left:14px}' +
          '.kt-sim-manifold.mf-cmp .mf-board th:first-child{padding-left:0}' +
          // over each viewport of the compare view: the technique and how far it is; a hairline between two viewports
          '.kt-sim-manifold .mf-vp{position:absolute;z-index:4;display:none;pointer-events:none;padding:5px 8px;font:500 10.5px/1.35 var(--f-mono);color:var(--dim);' +
          'background:var(--panel);border:1px solid var(--line);border-radius:8px;font-variant-numeric:tabular-nums}' +
          '.kt-sim-manifold .mf-vp b{display:block;font:600 10px/1.3 var(--f-mono);letter-spacing:.08em;text-transform:uppercase;color:var(--text)}' +
          '.kt-sim-manifold .mf-bar{position:absolute;z-index:3;display:none;width:1px;background:var(--line2);pointer-events:none}' +
          '.kt-sim-manifold.mf-cmp:not(.kt-bare) :is(.mf-vp,.mf-bar){display:block}' +
          // a narrow stage (a phone): the table and under it the symbols and the grid; smaller type, tighter rows
          '.kt-sim-manifold.mf-cmp.mf-narrow .mf-board{display:block;left:6px;right:6px;bottom:6px;padding:5px 7px;font-size:9px;line-height:1.2}' +
          '.kt-sim-manifold.mf-cmp.mf-narrow .mf-board>b,.kt-sim-manifold.mf-cmp.mf-narrow :is(.mf-x,.mf-key){display:none}.kt-sim-manifold.mf-cmp .mf-key{margin-top:4px}' +
          '.kt-sim-manifold.mf-cmp.mf-narrow .mf-board :is(th,td){padding:1px 0 1px 6px}.kt-sim-manifold.mf-cmp.mf-narrow .mf-board th:first-child{padding-left:0}' +
          '.kt-sim-manifold.mf-cmp.mf-narrow .mf-board thead th{font-size:8px;letter-spacing:0;padding-bottom:2px}' +
          '.kt-sim-manifold.mf-cmp.mf-narrow .mf-notes{font-size:8.5px;line-height:1.3;margin-top:3px}.kt-sim-manifold.mf-cmp.mf-narrow .mf-notes p{margin:0 0 2px}' +
          '.kt-sim-manifold.mf-cmp.mf-narrow :is(.mf-deck1,.mf-tog){display:none}' +       // a narrow lab page: the panel keeps the scene and the budget
          '.kt-sim-manifold.mf-narrow .mf-vp{padding:3px 5px;font-size:8.5px;line-height:1.25;border-radius:6px}.kt-sim-manifold.mf-narrow .mf-vp b{font-size:8.5px;letter-spacing:.02em}' +
          '.kt-sim-manifold .kt-panel .mf-tri.kt-seg .kt-seg-btn{white-space:nowrap;padding-left:2px;padding-right:2px}' +
          '.kt-sim-manifold .mf-io{display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--line);margin-top:2px;padding-top:7px}' +
          '.kt-sim-manifold .mf-io-note{font:500 10px/1.4 var(--f-mono);color:var(--mute);max-width:280px}.kt-sim-manifold .mf-io-note.bad{color:var(--err)}' +
          '@media (max-width:640px){.kt-sim-manifold .mf-board{left:6px;bottom:6px;max-width:calc(100% - 12px);padding:6px 8px;font-size:9.5px}' +
          '.kt-sim-manifold .mf-board>b{margin-bottom:3px}.kt-sim-manifold .mf-board th,.kt-sim-manifold .mf-board td{padding:1px 0 1px 9px}' +
          '.kt-sim-manifold .mf-board thead th{font-size:8.5px;letter-spacing:.03em}.kt-sim-manifold .mf-key{display:none}}' +
          // the panel stays above the board: on a folded stage the board steps aside while the controls are open, and on a
          // narrow page the panel ends above it (and scrolls), so the export is never under the board
          '.kt-sim-manifold.kt-compact.kt-open .mf-board{display:none}' +
          '@media (max-width:640px){.kt-sim-manifold.mf-mesh:not(.kt-embed) .kt-tr{max-height:calc(100% - 160px);overflow:auto}' +
          '.kt-sim-manifold.mf-mesh:not(.kt-embed) .kt-tr>*{flex-shrink:0}}' +
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
          // the light theme: what lies over the object (the board, the atlas, the badge, the band labels) is glass that hides
          // what is behind it, since a clear one turns grey over the owl and its dark text stops reading; a chart's name in
          // its identity colour is taken towards the text colour, so that it reads on white glass
          ':root[data-theme="light"] .kt-sim-manifold :is(.mf-board,.mf-vp,.kt-atlas,.kt-ab,.kt-split span){background:rgba(255,255,255,.84);' +
          'border-color:transparent;box-shadow:var(--glass-edge),0 10px 28px -18px rgba(88,56,24,.45)}' +
          ':root[data-theme="light"] .kt-sim-manifold .kt-tile-name{color:color-mix(in srgb,var(--c) 62%,var(--text))}' +
          // and the dividers of the split screen are dark lines on the light card (the same rule is in talk/prelude.css); the
          // badge of `lod` goes under the caption, where the wider glass panel does not cover its ON / OFF
          ':root[data-theme="light"] .kt-sim-manifold .kt-split i{background:color-mix(in srgb,var(--text) 22%,transparent)}' +
          ':root[data-theme="light"] .kt-sim-manifold .kt-ab{left:14px;transform:none}' +
          '@media (max-width:640px){.kt-sim-manifold .kt-atlas{gap:4px;padding:6px}.kt-sim-manifold .kt-tile-box{aspect-ratio:2}' +
          '.kt-sim-manifold .kt-tile-s{display:none}.kt-sim-manifold .kt-tile-name{font-size:8px;letter-spacing:.04em}.kt-sim-manifold .kt-tile-n{font-size:9px}}' +
          // Hebrew (talk/i18n.js sets <html lang>): nothing moves, the stage is laid out as in English; only the labels set as
          // letter-spaced capitals are not letter-spaced
          ':root[lang="he"] .kt-sim-manifold :is(.kt-tile-name,.kt-split span,.kt-ab,.mf-board thead th,.mf-vp b){letter-spacing:0}';
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
        sampling: function (m) { if (SAMPLINGS.indexOf(m) >= 0) setMode(m); return mode(); },       // 'even' | 'adaptive' | 'window' (for tests and host pages)
        setBuffer: function (b) { st.buf = KT.clamp(+b, 0, 0.15); if (slW) slW.set(st.buf); need = true; return st.buf; },
        setBudget: function (n) { st.budget = KT.clamp(n, 6000, CAP - 20000); slB.set(st.budget); need = true; remesh(); },
        /* scenes `mesh` and `stitch`, for tests: what is meshed and every number on the board; the stitched lattice as a file */
        meshStats: function () {
          if (!mesh) return null;
          function lean(a) { if (!a) return null; var o = {}; Object.keys(a).forEach(function (k) { if (typeof a[k] === 'number') o[k] = a[k]; }); return o; }
          var J = mesh.job, F = mesh.tjob;
          return { phase: mesh.phase, auto: false, compare: mesh.cmp, kinds: KINDS.slice(), rects: mesh.rects, points: mesh.V, budget: st.budget, triangles: mesh.Q * 2, stitchMs: mesh.lat.ms,
                   parts: mesh.parts.map(function (m) { return { name: m.c.name, na: m.na, nb: m.nb }; }), lattice: lean(mesh.aud.lattice), settled: !!mesh.settled, shown: !!mesh.settled && (!F || !!mesh.tm),
                   bytes: { lattice: mesh.lat.bytes, list: J && J.bytes, tsdf: F && F.bytes }, dist: { lattice: lean(mesh.dist.lattice), list: lean(mesh.dist.list), tsdf: lean(mesh.dist.tsdf) }, unit_mm: scaleOf().mm || null,
                   list: J && { done: J.done, progress: J.progress, tests: J.tests, ms: J.ms, cell: J.cell, k: J.k, triangles: J.nTri, audit: lean(J.stats) },
                   tsdf: F && { done: F.done, progress: F.progress, stage: F.stage, ms: F.ms, V: F.V, cameras: F.views, S: F.S, voxel: F.grid.h, spacing: F.spacing, grid: F.grid.n, voxels: F.grid.voxels,
                                capped: F.grid.capped, cap: VOXELS, wanted: F.grid.wanted.voxels, trunc: F.truncVoxels, minWeight: F.minWeight, triangles: F.mesh ? F.mesh.nTri : null,
                                vertices: F.mesh ? F.mesh.nV : null, audit: lean(mesh.aud.tsdf) } };
        },
        meshFile: function (fmt) { return mesh && IO ? meshFile(fmt) : null; },
        meshSeek: function (t) { if (mesh) KINDS.forEach(function (k) { mesh.clock[k] = t; }); },   // for tests: jump the running animations to t seconds
        resize: function (w, h) {
          R.resize(w, h); need = true; size.w = w; size.h = h;
          R.renderer.getDrawingBufferSize(field.uniforms.uRes.value);
          layoutMesh();
          var f = KT.clamp(1.4 / (w / h), 1, 1.7);
          if (f !== fit) { fit = f; if (!orbit.touched) orbit.to(view(st.sc.tour ? def.tour[st.tourI] : st.sc.cam || 'wide'), true); }
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
