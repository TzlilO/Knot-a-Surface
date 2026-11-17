/* One take (talk-onetake.html) — the thesis on one object, in one continuous shot, with museum labels.
   The scanned owl (module `manifold`, bare) is on the stage from the first second; each beat is one placard (number,
   title, one sentence, a provenance line) and one stage state: the owl's scene, optionally one of its readings or its
   control nets, optionally a loupe (a second small simulation: `flatland`, `planoid`) and, for the finale, the `touch`
   module's `flight`, in which the owl stands too. Everything is driven through the kit's public handles: KT.mount,
   handle.setScene, handle.api (auto, for the throw) and the mount option onStats. The two things the manifold handle
   has no call for (a reading other than the scene's own, the nets) are asked of the module's own control panel, which
   is mounted but hidden (html.kt-bare): its view buttons and its nets toggle, clicked as a user would.
   Copy: the BEATS table below, in English, every string through KnotTalk.t; Hebrew in talk/onetake/he.js.
   Keys: → ↓ space PageDown next, ← ↑ PageUp back (in Hebrew ← is next), Home / End, T clock, L language, M mute,
   Enter throws on the finale. Taps on the left / right third of the window, vertical swipes, #7 deep links. */
(function () {
  'use strict';
  var KT = window.KnotTalk, T = KT.t, RTL = !!KT.rtl, root = document.documentElement;
  function $(s, r) { return (r || document).querySelector(s); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  var QB = +((/[?&]budget=(\d+)/.exec(location.search) || [])[1] || 0);   // ?budget=N: the owl's Gaussian budget (tests on a software renderer)

  /* ================= evidence: copied exactly from the constants at the top of talk/deck.js =================
     DTU: Table 1 of the paper (Chamfer distance, mm; PGSR = its best results); BENCH: the Surface-Prior bench,
     measured in-page on a procedural proxy capture, baselines emulated. Nothing here is a number of this page's own. */
  var DTU = {
    scans: [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122],
    pgsr: [0.33, 0.51, 0.29, 0.28, 0.75, 0.53, 0.46, 0.92, 0.62, 0.48, 0.45, 0.55, 0.29, 0.33, 0.31],
    ours: [0.303, 0.480, 0.316, 0.329, 0.643, 0.552, 0.467, 0.801, 0.626, 0.558, 0.821, 0.572, 0.296, 0.346, 0.341],
    printedMean: { pgsr: 0.47, ours: 0.466 }
  };
  var BENCH = [
    { name: 'Knot-a-Surface', params: 96.0e3, chamfer: 72.0, ours: true },
    { name: 'PGSR', params: 1.81e6, chamfer: 58.4 }, { name: '2DGS', params: 2.06e6, chamfer: 58.5 },
    { name: 'SuGaR', params: 1.65e6, chamfer: 216.3 }, { name: '3DGS', params: 3.69e6, chamfer: 59.7 },
    { name: 'NeuS / VolSDF', params: 2.10e6, chamfer: 676.6 }, { name: 'MVS', params: 116.6e3, chamfer: 201.2 }
  ];
  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }
  function evidence() {
    var n = DTU.scans.length, mo = mean(DTU.ours), mp = mean(DTU.pgsr), wins = 0, close = 0, i;
    for (i = 0; i < n; i++) { if (DTU.ours[i] < DTU.pgsr[i]) wins++; if (Math.abs(DTU.ours[i] - DTU.pgsr[i]) <= 0.05) close++; }
    // the bench as talk.html says it: the densifying baselines (the ones that reach about 58 mm), as many times our parameters
    var us = BENCH.filter(function (b) { return b.ours; })[0], dens = BENCH.filter(function (b) { return !b.ours && b.chamfer < 60; });
    var lo = Math.round(Math.min.apply(null, dens.map(function (b) { return b.params / us.params; }))),
        hi = Math.round(Math.max.apply(null, dens.map(function (b) { return b.params / us.params; })));
    var best = Math.round(Math.min.apply(null, dens.map(function (b) { return b.chamfer; })));
    var bad = [];
    if (Math.abs(mo - DTU.printedMean.ours) > 0.0015) bad.push(T('ours {v}', { v: DTU.printedMean.ours }));
    if (Math.abs(mp - DTU.printedMean.pgsr) > 0.006) bad.push(T('PGSR {v}', { v: DTU.printedMean.pgsr }));
    return [
      { big: T('{a} mm against {b} mm', { a: mo.toFixed(3), b: mp.toFixed(3) }),
        small: T('mean Chamfer distance on {n} DTU scans, ours against PGSR (the paper, Table 1)', { n: n }) },
      { big: T('{c} of {n} scans', { c: close, n: n }),
        small: T('within 0.05 mm of PGSR; ours is lower on {w}', { w: wins }) },
      { big: T('{p} mm from {k} k parameters', { p: Math.round(us.chamfer), k: Math.round(us.params / 1000) }),
        small: T('proxy bench: the densifying baselines reach ≈ {b} mm, more accurate, with {lo}–{hi}× as many; baselines are behavioural emulations', { b: best, lo: lo, hi: hi }) },
      bad.length ? { note: T('Means are computed from the 15 scans shown (the table prints {list}).', { list: bad.join(', ') }) } : null
    ].filter(Boolean);
  }

  /* ================= the beats =================
     owl: the manifold scene · view: one of its readings (its own view buttons) · nets: its nets toggle · loupe: [module,
     [[seconds, scene], ...]] · touch: the touch module's scene instead of the owl · dim / wide: the evidence look ·
     prov(stats): a provenance line that carries a live count from the module's stats */
  var stats = null;
  function fmt(n) { return Number(n).toLocaleString('en-US'); }
  var BEATS = [
    { owl: 'scene', delay: 2.5,
      title: T('This is not a photograph.'),
      text: T('Nothing here is a pixel that a camera recorded: it is being drawn now, from a description of the owl.'),
      prov: T('DTU scan 122 · surface fitted offline to the dataset’s depth maps · not trained') },
    { owl: 'scene', view: 'gauss',
      title: T('It is drawn from soft blobs.'),
      text: T('A Gaussian has a centre, a size, an orientation, an opacity and a colour; blend enough of them and you get the picture.'),
      prov: function (s) { return s ? T('{n} Gaussians on screen · the count is measured', { n: '<b>' + fmt(s.drawn) + '</b>' }) : T('the count on screen is measured'); } },
    { owl: 'scene', view: 'gauss', loupe: ['flatland', [[0, 'novel'], [5, 'rays']]],
      title: T('A list of blobs does not know where the surface is.'),
      text: T('Free Gaussians match the photographs and still float off the object: along a camera ray, depth is a free choice.'),
      prov: T('2D toy · optimiser running live in this page') },
    { owl: 'scene', nets: true,
      title: T('Under these blobs there is a surface.'),
      text: T('A spline surface: a net of control points, smooth between them, and the only thing that is stored.'),
      prov: function (s) { return s ? T('{c} charts · {n} control points', { c: s.charts, n: fmt(s.controlPoints) }) : T('two charts · 121,216 control points'); } },
    { owl: 'scene', nets: true, loupe: ['planoid', [[0, 'probe'], [4.5, 'normal'], [9, 'scale']]],
      title: T('Ask the surface, get a Gaussian.'),
      text: T('Position is a point of the surface; orientation is its two tangents and their normal; size is the distance to the next sample.'),
      prov: T('computed from derivatives · nothing learned per Gaussian') },
    { owl: 'edit', nets: true,
      title: T('Move the surface and every blob follows.'),
      text: T('The Gaussians are not stored and corrected: they are computed again from the control net, every frame.'),
      prov: T('the owl’s motion is scripted') },
    { owl: 'lod',
      title: T('How many blobs is a knob, not a property.'),
      text: T('The same surface can be drawn with a thousand Gaussians or a million, and a fixed budget can be spent where the camera looks.'),
      prov: T('same control points in both · occlusion is not tested') },
    { owl: 'scene', view: 'normal',
      title: T('Normals are read, not estimated.'),
      text: T('Depth, normal and the chart a blob belongs to come straight from the surface.'),
      prov: T('analytic · from S<sub>u</sub> × S<sub>v</sub>') },
    { owl: 'charts',
      title: T('A scene is an atlas of charts.'),
      text: T('Each chart is one spline patch with its own (u, v) square, like a page of a map.'),
      prov: T('at the seam between charts nothing is guaranteed') },
    { owl: 'stitch',
      title: T('The mesh was already there.'),
      text: T('Samples on a lattice know their neighbours, so a sliding 2 × 2 window stitches triangles with no search.'),
      prov: T('stitched live at the current budget') },
    { owl: 'paint',
      title: T('A material is an edit of the control points.'),
      text: T('Colour, opacity and shape live on the same points, so glass is a rule written on them.'),
      prov: T('a rule on the scan’s colours and normals · nothing painted by hand') },
    { touch: 'flight', act: 'throw',
      title: T('And a surface can be hit.'),
      text: T('On a spline, contact is a closest point and a normal; on a list of blobs there is nothing to hit.'),
      prov: T('contact computed · the headers are scripted · the lists are emulated · the smoke is an invented rule') },
    { owl: 'scene', dim: true, wide: true, facts: true,
      title: T('Does it still reconstruct real objects?'),
      text: '',
      prov: T('figures from the paper’s table and a proxy bench · the owl on this page is a fit, not a result of the method') },
    { owl: 'scene', dim: true,
      title: T('What is open.'),
      text: T('Seams between charts, thin structures, foliage and glass are open; image quality, training and rendering time, and ablations are not measured yet.'),
      prov: T('stated limits') },
    { owl: 'scene', last: true,
      title: T('The surface is not reconstructed at the end. It is the representation.'),
      text: '',
      prov: T('Knot-a-Surface · thesis talk') }
  ];
  var N = BEATS.length;

  /* ================= DOM ================= */
  var stageEl = $('.ot-stage'), loupeEl = $('.ot-loupe'), loupeHost = $('.ot-loupe-sim'), wire = $('.ot-wire');
  var plac = $('.ot-placard'), pNo = $('.ot-no'), pTitle = $('.ot-title'), pText = $('.ot-text'), pFacts = $('.ot-facts'), pProv = $('.ot-prov'), pAct = $('.ot-act');
  var bar = $('.ot-progress i'), hint = $('.ot-hint'), clock = $('.ot-clock'), msg = $('.ot-msg');
  var FWD = RTL ? '←' : '→', BACK = RTL ? '→' : '←';
  hint.innerHTML = T('{fwd} or space · next   {back} · back   T · clock   L · language', { fwd: FWD, back: BACK });
  function say(text) { msg.textContent = text; msg.classList.toggle('on', !!text); }

  /* ================= the stage: one module at a time, cross-faded ================= */
  var OWL_OK = !!(KT.sims.manifold && KT.dtu && KT.dtu.charts);
  var TOUCH_OK = !!(KT.sims.touch && KT.dtu && KT.dtuScans && KT.dtuScans[55]);
  var stage = null;                 // { sim, handle, layer, scene }
  function loseGL(node) {           // free the WebGL context at once: back-and-forth must not pile up contexts
    [].forEach.call(node.querySelectorAll('canvas'), function (c) {
      var gl = null; try { gl = c.getContext('webgl2') || c.getContext('webgl'); } catch (e) { gl = null; }
      var x = gl && gl.getExtension('WEBGL_lose_context'); if (x) x.loseContext();
    });
  }
  function drop(s, later) {
    if (!s) return;
    function go() { try { s.handle.destroy(); } catch (e) { /* already gone */ } loseGL(s.layer); s.layer.remove(); }
    if (later) setTimeout(go, later); else go();
  }
  function mountStage(sim, scene) {
    var layer = el('div', 'ot-layer'), mnt = el('div'); layer.appendChild(mnt); stageEl.appendChild(layer);
    var o = { embed: true, bare: true, scene: scene };
    if (sim === 'manifold') { o.cap = 200000; if (QB) o.budget = QB; o.onStats = function (s) { stats = s; liveProv(); }; }
    var h = KT.mount(sim, mnt, o);
    var s = { sim: sim, handle: h, layer: layer, scene: scene, root: mnt };
    requestAnimationFrame(function () { requestAnimationFrame(function () { layer.classList.add('on'); }); });
    return s;
  }
  function useStage(sim, scene) {
    if (stage && stage.sim === sim) { stage.scene = scene; stage.handle.setScene(scene); return stage; }
    var old = stage;
    stage = mountStage(sim, scene);
    if (old) { old.layer.classList.remove('on'); drop(old, 650); }      // the 0.6 s cross-fade, then the old module goes
    return stage;
  }
  /* the owl's readings and nets: the module's own (hidden) panel, as a user would press it */
  function pressView(v) {
    var b = [].filter.call(stage.root.querySelectorAll('.kt-seg-btn'), function (x) { return x._v === v; })[0];
    if (b && !b.classList.contains('on')) b.click();
  }
  function setNets(on) {
    var b = stage.root.querySelectorAll('.mf-tog .kt-toggle')[1];
    if (b && (b.getAttribute('aria-pressed') === 'true') !== on) b.click();
  }

  /* ================= the loupe ================= */
  var loupe = null, loupeT0 = 0, loupeStep = -1;     // { sim, handle, root }
  function openLoupe(spec) {
    var sim = spec[0];
    if (loupe && loupe.sim !== sim) { dropLoupe(); }
    if (!loupe) {
      var r = el('div'); loupeHost.appendChild(r);
      if (!KT.sims[sim]) { r.className = 'ot-loupe-msg'; r.textContent = T('This simulation could not be loaded.'); loupe = { sim: sim, handle: null, root: r }; }
      else loupe = { sim: sim, handle: KT.mount(sim, r, { embed: true, scene: spec[1][0][1] }), root: r };
    }
    loupe.root.style.display = '';
    loupeT0 = performance.now(); loupeStep = -1;
    loupeEl.classList.add('on'); root.classList.add('ot-loupe-on');
    placeWire(); wire.classList.add('on');
  }
  function closeLoupe() {
    loupeEl.classList.remove('on'); root.classList.remove('ot-loupe-on'); wire.classList.remove('on');
    var l = loupe; if (!l) return;
    setTimeout(function () { if (loupe === l && !loupeEl.classList.contains('on')) l.root.style.display = 'none'; }, 750);   // hidden = paused
  }
  function dropLoupe() {
    if (!loupe) return;
    if (loupe.handle) try { loupe.handle.destroy(); } catch (e) { /* gone */ }
    loseGL(loupe.root); loupe.root.remove(); loupe = null;
  }
  function placeWire() {
    var W = window.innerWidth, H = window.innerHeight, r = loupeEl.getBoundingClientRect();
    var sx = RTL ? 0.09 : -0.09;                                  // --ot-shift-loupe, in window widths
    var ox = W / 2 + sx * W, oy = H * 0.36;                       // about the owl's head while the loupe is open
    var lx = RTL ? r.right : r.left, ly = r.bottom - 14;
    if (!r.width) { lx = RTL ? W * 0.4 : W * 0.6; ly = H * 0.5; }
    var line = $('line', wire), dot = $('circle', wire), dx = ox - lx, dy = oy - ly, d = Math.hypot(dx, dy) || 1;
    line.setAttribute('x1', lx); line.setAttribute('y1', ly); line.setAttribute('x2', ox - dx / d * 6); line.setAttribute('y2', oy - dy / d * 6);
    dot.setAttribute('cx', ox); dot.setAttribute('cy', oy);
  }

  /* ================= the placard ================= */
  var cur = -1, beatT0 = 0, lastProv = '';
  function provOf(b) { return typeof b.prov === 'function' ? b.prov(stats) : b.prov; }
  function liveProv() {
    var b = BEATS[cur]; if (!b || typeof b.prov !== 'function') return;
    var p = provOf(b); if (p !== lastProv) { pProv.innerHTML = p; lastProv = p; }
  }
  function fill(i) {
    var b = BEATS[i];
    pNo.innerHTML = T('No. {n} / {total}', { n: String(i + 1).padStart(2, '0'), total: String(N).padStart(2, '0') });
    pTitle.innerHTML = b.title; pText.innerHTML = b.text || '';
    pFacts.innerHTML = '';
    if (b.facts) evidence().forEach(function (f) {
      if (f.note) { pFacts.appendChild(el('p', 'ot-note', f.note)); return; }
      var d = el('div', 'ot-fact'); d.appendChild(el('b', '', f.big)); d.appendChild(el('span', '', f.small)); pFacts.appendChild(d);
    });
    lastProv = provOf(b); pProv.innerHTML = lastProv;
    pAct.innerHTML = '';
    if (b.act === 'throw' && TOUCH_OK) {
      var btn = el('button', '', T('throw')); btn.type = 'button';
      btn.addEventListener('click', function (e) { e.stopPropagation(); btn.blur(); throwBall(); });
      pAct.appendChild(btn); pAct.appendChild(el('kbd', '', 'Enter'));
    }
    plac.classList.toggle('ot-last', !!b.last);
  }
  /* the swap is a CSS animation, not a timer: whatever the frames cost, the placard ends up shown */
  function showPlacard(i, first) {
    var b = BEATS[i];
    fill(i);
    plac.classList.remove('in', 'first');
    void plac.offsetWidth;                                          // restart the animation
    plac.classList.add(first && b.delay ? 'first' : 'in');
  }
  function throwBall() { if (stage && stage.sim === 'touch' && stage.handle.api.auto) stage.handle.api.auto(); }

  /* ================= a beat ================= */
  function go(i, first) {
    i = Math.max(0, Math.min(N - 1, i | 0));
    if (i === cur && !first) return;
    var b = BEATS[i];
    cur = i; beatT0 = performance.now();
    say('');
    if (b.touch && TOUCH_OK) useStage('touch', b.touch);
    else if (OWL_OK) {
      useStage('manifold', b.owl || 'scene');
      if (b.view) pressView(b.view);
      if (b.nets) setNets(true);
    } else say(T('The scanned owl or its simulation could not be loaded on this page.'));
    if (b.touch && !TOUCH_OK) say(T('The finale needs the touch module and the rabbit’s scan, which could not be loaded.'));
    root.classList.toggle('ot-dim', !!b.dim); root.classList.toggle('ot-wide', !!b.wide);
    root.classList.toggle('ot-center', !!(b.touch && TOUCH_OK));    // the finale frames itself for the whole window
    if (b.loupe) openLoupe(b.loupe); else {
      closeLoupe();
      var keep = loupe && [i - 1, i + 1].some(function (k) { return BEATS[k] && BEATS[k].loupe && BEATS[k].loupe[0] === loupe.sim; });
      if (loupe && !keep) { var l = loupe; setTimeout(function () { if (loupe === l && !loupeEl.classList.contains('on')) dropLoupe(); }, 800); }
    }
    showPlacard(i, first);
    bar.style.width = ((i + 1) / N * 100) + '%';
    if (!first) hint.classList.add('gone');
    try { history.replaceState(null, '', location.pathname + location.search + '#' + (i + 1)); } catch (e) { /* file:// in some browsers */ }
  }
  function next() { go(cur + 1); }
  function prev() { go(cur - 1); }

  /* the loupe's own timeline inside a beat (time-based) */
  function tick() {
    var b = BEATS[cur];
    if (b && b.loupe && loupe && loupe.handle) {
      var t = (performance.now() - loupeT0) / 1000, steps = b.loupe[1], k = 0, j;
      for (j = 0; j < steps.length; j++) if (t >= steps[j][0]) k = j;
      if (k !== loupeStep) { loupeStep = k; loupe.handle.setScene(steps[k][1]); }
    }
    if (clock.classList.contains('on')) {
      var s = Math.floor((performance.now() - clockT0) / 1000);
      clock.textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
    }
    requestAnimationFrame(tick);
  }
  var clockT0 = performance.now();

  /* ================= input ================= */
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var k = e.key, t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
    var fwd = RTL ? 'ArrowLeft' : 'ArrowRight', back = RTL ? 'ArrowRight' : 'ArrowLeft';
    if (k === fwd || k === 'ArrowDown' || k === 'PageDown' || k === ' ') { e.preventDefault(); next(); }
    else if (k === back || k === 'ArrowUp' || k === 'PageUp') { e.preventDefault(); prev(); }
    else if (k === 'Home') { e.preventDefault(); go(0); }
    else if (k === 'End') { e.preventDefault(); go(N - 1); }
    else if (k === 't' || k === 'T') { if (!clock.classList.contains('on') && e.shiftKey) clockT0 = performance.now(); clock.classList.toggle('on'); }
    else if (k === 'l' || k === 'L') { if (KT.i18n) KT.i18n.toggle(); }
    else if (k === 'm' || k === 'M') { KT.muted = !KT.muted; }
    else if (k === 'Enter') { if (BEATS[cur].act === 'throw') { e.preventDefault(); throwBall(); } }
  });
  /* a tap on the left / right third steps (the middle third is the owl's, to orbit); a vertical swipe steps too */
  var down = null;
  window.addEventListener('pointerdown', function (e) { down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId, touch: e.pointerType === 'touch', tgt: e.target }; }, true);
  window.addEventListener('pointerup', function (e) {
    var d = down; down = null;
    if (!d || d.id !== e.pointerId) return;
    if (d.tgt && d.tgt.closest && d.tgt.closest('button, a, .ot-loupe')) return;
    var dx = e.clientX - d.x, dy = e.clientY - d.y, dt = performance.now() - d.t;
    if (d.touch && Math.abs(dy) > 60 && Math.abs(dy) > 1.6 * Math.abs(dx) && dt < 900) { if (dy < 0) next(); else prev(); return; }
    if (d.touch && Math.abs(dx) > 60 && Math.abs(dx) > 1.6 * Math.abs(dy) && dt < 900 && d.tgt.closest && d.tgt.closest('.ot-placard')) {
      if ((dx < 0) !== RTL) next(); else prev(); return;
    }
    if (Math.hypot(dx, dy) > 8 || dt > 500) return;
    var W = window.innerWidth, x = e.clientX;
    if (x > W * 2 / 3) { if (RTL) prev(); else next(); }
    else if (x < W / 3) { if (RTL) next(); else prev(); }
  }, true);
  window.addEventListener('hashchange', function () { var n = parseInt(location.hash.slice(1), 10); if (n >= 1 && n <= N) go(n - 1); });
  window.addEventListener('resize', function () { if (loupeEl.classList.contains('on')) placeWire(); });
  loupeEl.addEventListener('transitionend', function () { if (loupeEl.classList.contains('on')) placeWire(); });

  /* ================= start ================= */
  var h0 = parseInt(location.hash.slice(1), 10);
  go(h0 >= 1 && h0 <= N ? h0 - 1 : 0, true);
  requestAnimationFrame(tick);

  /* for tests: where the page is, and what is mounted */
  KT.onetake = {
    beats: N, go: go, next: next, prev: prev,
    state: function () {
      return { beat: cur + 1, stage: stage && { sim: stage.sim, scene: stage.scene }, loupe: loupe && { sim: loupe.sim, open: loupeEl.classList.contains('on'), step: loupeStep },
               mounted: KT.mounted.map(function (m) { return m.name; }), canvases: document.querySelectorAll('canvas').length, stats: stats };
    },
    stage: function () { return stage && stage.handle; }, loupe: function () { return loupe && loupe.handle; }
  };
})();
