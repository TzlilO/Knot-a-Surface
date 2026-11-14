/* Knot-a-Surface talk — simulation kit.
   Every simulation is a self-contained module that registers itself here:

     KnotTalk.register('name', {
       title, blurb, tool,                       // shown by the standalone shell
       scenes: [{ id, label }],                  // named states the deck (or shell) can jump to
       create(env) -> { setScene(id), resize?(w, h), destroy?() }
     });

   env = { root, embed, loop(fn), ui, nurbs }
     root   — a positioned, sized element the module owns
     embed  — true inside the deck (keep chrome compact), false on the standalone page
     loop   — register a per-frame callback fn(t, dt) in seconds; it only runs while the
              module is on screen and the tab is visible

   Mount with KnotTalk.mount(name, element, { embed }) or, on a bare page,
   KnotTalk.standalone(name) which builds the lab shell around it. */
(function () {
  'use strict';
  var KT = window.KnotTalk = window.KnotTalk || {};
  KT.sims = KT.sims || {};
  KT.register = function (name, def) { KT.sims[name] = def; };

  /* ---------- tiny DOM helpers ---------- */
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  var ui = {
    el: el,
    /* floating control group; pos: 'tl' | 'tr' | 'bl' | 'br' | 'bottom' */
    panel: function (root, o) {
      o = o || {};
      var p = el('div', 'kt-panel kt-' + (o.pos || 'br'));
      if (o.title) p.appendChild(el('div', 'kt-panel-title', o.title));
      root.appendChild(p);
      return p;
    },
    slider: function (parent, o) {
      var row = el('label', 'kt-row kt-slider');
      var lab = el('span', 'kt-lab', o.label), val = el('span', 'kt-val');
      var inp = el('input');
      inp.type = 'range'; inp.min = o.min; inp.max = o.max; inp.step = o.step || 1; inp.value = o.value;
      var fmt = o.fmt || function (v) { return v; };
      function show() { val.textContent = fmt(+inp.value); }
      inp.addEventListener('input', function () { show(); if (o.onInput) o.onInput(+inp.value); });
      row.appendChild(lab); row.appendChild(inp); row.appendChild(val); parent.appendChild(row); show();
      return { el: row, input: inp, get: function () { return +inp.value; },
               set: function (v) { inp.value = v; show(); } };
    },
    /* segmented control; options: [{ v, label }] */
    seg: function (parent, o) {
      var row = el('div', 'kt-row kt-seg');
      if (o.label) row.appendChild(el('span', 'kt-lab', o.label));
      var box = el('div', 'kt-seg-box'), cur = o.value, btns = [];
      function paint() { btns.forEach(function (b) { b.classList.toggle('on', b._v === cur); }); }
      o.options.forEach(function (op) {
        var b = el('button', 'kt-seg-btn', op.label); b.type = 'button'; b._v = op.v;
        b.addEventListener('click', function () { cur = op.v; paint(); if (o.onChange) o.onChange(cur); });
        btns.push(b); box.appendChild(b);
      });
      row.appendChild(box); parent.appendChild(row); paint();
      return { el: row, get: function () { return cur; }, set: function (v) { cur = v; paint(); } };
    },
    toggle: function (parent, o) {
      var b = el('button', 'kt-toggle', o.label); b.type = 'button';
      var on = !!o.value;
      function paint() { b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); }
      b.addEventListener('click', function () { on = !on; paint(); if (o.onChange) o.onChange(on); });
      parent.appendChild(b); paint();
      return { el: b, get: function () { return on; }, set: function (v) { on = !!v; paint(); } };
    },
    button: function (parent, o) {
      var b = el('button', 'kt-btn' + (o.kind ? ' kt-btn-' + o.kind : ''), o.label); b.type = 'button';
      b.addEventListener('click', o.onClick); parent.appendChild(b);
      return b;
    },
    /* label + live value, for the numbers a module wants read aloud */
    readout: function (parent, o) {
      var row = el('div', 'kt-row kt-readout' + (o.tone ? ' kt-tone-' + o.tone : ''));
      var val = el('span', 'kt-val');
      row.appendChild(el('span', 'kt-lab', o.label)); row.appendChild(val); parent.appendChild(row);
      return { el: row, set: function (t) { val.textContent = t; },
               tone: function (t) { row.className = 'kt-row kt-readout' + (t ? ' kt-tone-' + t : ''); } };
    },
    /* DPR-aware 2D canvas filling `parent`; call fit() from resize, returns CSS-pixel size */
    canvas: function (parent, cls) {
      var cv = el('canvas', cls || 'kt-canvas'); parent.appendChild(cv);
      var ctx = cv.getContext('2d'), size = { w: 0, h: 0, dpr: 1 };
      function fit() {
        var r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
        size.w = Math.max(1, r.width); size.h = Math.max(1, r.height); size.dpr = dpr;
        cv.width = Math.round(size.w * dpr); cv.height = Math.round(size.h * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return size;
      }
      return { cv: cv, ctx: ctx, size: size, fit: fit };
    }
  };
  KT.ui = ui;

  /* the talk's colour code — one meaning per hue, everywhere */
  KT.color = {
    surf: '#4fc3f7',   // the continuous surface (what we optimise)
    ctrl: '#ffa726',   // control points & knots (the learnable parameters)
    splat: '#ce93d8',  // Gaussians (derived, never free)
    err: '#ef5350',    // error, floaters, free primitives
    ok: '#69f0ae',     // good news
    text: '#dde5f0', dim: '#8fa4c0', mute: '#5c6f8a', line: '#1b2a42', bg: '#07090f'
  };
  KT.clamp = function (x, a, b) { return x < a ? a : x > b ? b : x; };
  KT.lerp = function (a, b, t) { return a + (b - a) * t; };
  KT.ease = function (t) { t = KT.clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  /* deterministic RNG so every run of the talk shows the same thing */
  KT.rng = function (seed) {
    var s = seed >>> 0;
    return function () { s = (s + 0x6D2B79F5) | 0; var t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  };

  /* ---------- lifecycle ---------- */
  KT.mount = function (name, root, opts) {
    var def = KT.sims[name];
    if (!def) throw new Error('KnotTalk: no simulation named "' + name + '"');
    opts = opts || {};
    root.classList.add('kt-sim', 'kt-sim-' + name);
    if (opts.embed) root.classList.add('kt-embed');

    var loops = [], visible = false, raf = 0, last = 0;
    function frame(ms) {
      raf = 0;
      if (!visible || document.hidden) return;
      var t = ms / 1000, dt = last ? Math.min(0.05, t - last) : 0.016;
      last = t;
      for (var i = 0; i < loops.length; i++) loops[i](t, dt);
      raf = requestAnimationFrame(frame);
    }
    function kick() { if (!raf && visible && !document.hidden) { last = 0; raf = requestAnimationFrame(frame); } }

    var env = { root: root, embed: !!opts.embed, opts: opts, ui: ui, nurbs: KT.nurbs,
                loop: function (fn) { loops.push(fn); kick(); } };
    var api = def.create(env) || {};

    // small embedded stages: fold the control panels behind one button so they don't cover the figure
    var gear = el('button', 'kt-gear', 'controls'); gear.type = 'button';
    gear.addEventListener('click', function () { root.classList.toggle('kt-open'); });
    root.appendChild(gear);
    function hasPanel() {
      return [].some.call(root.querySelectorAll('.kt-panel'), function (p) { return p.style.display !== 'none'; });
    }
    if (opts.embed) {
      // inside the deck, a control used with the pointer must not keep focus: the deck's arrow and
      // space keys have to keep working afterwards (keyboard users who Tab to a control keep it)
      root.addEventListener('click', function (e) {
        var b = e.target.closest && e.target.closest('button');
        if (b && e.detail > 0) b.blur();
      });
      root.addEventListener('pointerup', function (e) { if (e.target.tagName === 'INPUT') e.target.blur(); });
    }
    var ro = new ResizeObserver(function () {
      var r = root.getBoundingClientRect();
      root.classList.toggle('kt-compact', !!opts.embed && r.width < 560 && hasPanel());
      if (api.resize && r.width > 0 && r.height > 0) api.resize(r.width, r.height);
    });
    ro.observe(root);
    var io = new IntersectionObserver(function (es) {
      visible = es[es.length - 1].isIntersecting; kick();
    }, { rootMargin: '0px' });           // strictly on screen: an off-screen WebGL stage must not keep rendering
    io.observe(root);
    var onVis = function () { kick(); };
    document.addEventListener('visibilitychange', onVis);

    return {
      name: name, def: def, api: api,
      setScene: function (id) { if (api.setScene) api.setScene(id); },
      destroy: function () {
        ro.disconnect(); io.disconnect(); document.removeEventListener('visibilitychange', onVis);
        if (raf) cancelAnimationFrame(raf);
        if (api.destroy) api.destroy();
      }
    };
  };

  /* ---------- standalone lab shell ---------- */
  KT.standalone = function (name) {
    var def = KT.sims[name];
    document.title = def.title + ' — Knot-a-Surface lab';
    document.body.classList.add('kt-labpage');
    var head = el('header', 'kt-lab-head',
      '<a class="kt-lab-back" href="../../talk.html">← the talk</a>' +
      '<div class="kt-lab-id"><span class="kt-lab-kicker">Knot-a-Surface · lab module</span>' +
      '<h1>' + def.title + '</h1></div>' +
      '<p class="kt-lab-blurb">' + def.blurb + '</p>' +
      '<span class="kt-lab-tool">' + def.tool + '</span>');
    var main = el('main', 'kt-lab-main'), stage = el('div', 'kt-lab-stage');
    main.appendChild(stage);
    var foot = el('footer', 'kt-lab-foot');
    document.body.appendChild(head); document.body.appendChild(main); document.body.appendChild(foot);
    var opts = { embed: false };                 // ?key=value in the URL reaches the module as env.opts
    new URLSearchParams(location.search).forEach(function (v, k) { opts[k] = v; });
    var sim = KT.mount(name, stage, opts);
    var btns = [];
    function go(id) {
      sim.setScene(id);
      btns.forEach(function (b) { b.classList.toggle('on', b._id === id); });
      try { history.replaceState(null, '', '#' + id); } catch (e) {}
    }
    (def.scenes || []).forEach(function (s, i) {
      var b = el('button', 'kt-lab-scene', '<i>' + String(i + 1).padStart(2, '0') + '</i>' + s.label);
      b.type = 'button'; b._id = s.id; b.addEventListener('click', function () { go(s.id); });
      btns.push(b); foot.appendChild(b);
    });
    var want = location.hash.slice(1), first = def.scenes && def.scenes[0];
    var ok = (def.scenes || []).some(function (s) { return s.id === want; });
    if (ok) go(want); else if (first) go(first.id);
    return sim;
  };
})();
