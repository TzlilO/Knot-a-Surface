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

  /* the page's theme: 'light' when <html data-theme="light"> (talk.html), else 'dark' (the lab pages, the landing page).
     Decided once, when this file runs: modules read KT.color when they are created. */
  KT.theme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  /* the talk's colour code — one meaning per hue, everywhere — in the page's theme (the same values as kit.css's tokens).
     bg = what a module draws on: the dark stage, or on a light page the white glass card (over the warm page). */
  KT.color = KT.theme === 'light' ? {
    surf: '#0068d9',   // the continuous surface (what we optimise)
    ctrl: '#c25e00',   // control points & knots (the learnable parameters)
    splat: '#9b3fcf',  // Gaussians (derived, never free)
    err: '#d93025',    // error, floaters, free primitives
    ok: '#16843a',     // good news
    text: '#1d1d1f', dim: '#636366', mute: '#7c7c80', line: '#dcd5cc', bg: '#fbf8f5'
  } : {
    surf: '#4fc3f7',
    ctrl: '#ffa726',
    splat: '#ce93d8',
    err: '#ef5350',
    ok: '#69f0ae',
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

  /* ---------- touch: who owns a gesture ----------
     A mounted root carries touch-action:pan-y (kit.css), so inside a deck a one-finger vertical
     swipe over a stage scrolls the page. A stage claims a touch only when it starts on a handle,
     or (for stages you orbit) when a second finger lands on it.
       KT.touchGuard(el, { hit(x, y), multi })
         hit   -> true when a touch starting at client (x, y) lands on a draggable handle
         multi -> also claim two fingers on el (no page scroll or zoom while they are down)
       KT.gestures(el, o) — the multi-pointer bookkeeping for a stage you turn (gl.orbit, he-shapes):
         o.embed       inside a deck: one finger on touch turns horizontally only, vertical scrolls
         o.claim(e)    -> true when the module takes this pointerdown itself (a handle drag)
         o.hit(x, y)   -> true when a touch there would land on such a handle
         o.drag(dx, dy, n, e)  one pointer (n = 1) or the centroid of two fingers (n = 2), CSS px
         o.pinch(k)    finger spread ratio since the previous event (k > 1: fingers apart)
         o.start(e)    an unclaimed pointer went down
         o.release(id) optional: a second finger landed while a finger held a handle — drop that
                       handle (put it back); the two fingers become a camera gesture instead
       Outside a deck (the lab pages) the element gets touch-action:none and owns every gesture. */
  KT.HIT = 24;                       // touch hit radius around a handle, CSS px (a ~48 px target)
  function onEl(el, list) { var k = 0; for (var i = 0; i < list.length; i++) if (el.contains(list[i].target)) k++; return k; }
  KT.touchGuard = function (el, o) {
    o = o || {};
    function stop(e) { if (e.cancelable) e.preventDefault(); }
    el.addEventListener('touchstart', function (e) {
      if (o.multi && onEl(el, e.touches) >= 2) { stop(e); return; }
      var t = e.changedTouches[0];
      if (o.hit && t && o.hit(t.clientX, t.clientY)) stop(e);   // a cancelled touchstart never scrolls
    }, { passive: false });
    if (o.multi) {
      el.addEventListener('touchmove', function (e) { if (onEl(el, e.touches) >= 2) stop(e); }, { passive: false });
      el.addEventListener('gesturestart', stop);                // iOS Safari's own pinch-zoom
    }
  };
  KT.gestures = function (el, o) {
    var pts = {}, ids = [], claimed = {}, pair = null, SLOP = 6;
    if (!o.embed) el.style.touchAction = 'none';
    KT.touchGuard(el, { hit: o.hit, multi: true });
    function spread() {
      if (ids.length < 2) return null;
      var a = pts[ids[0]], b = pts[ids[1]];
      return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, d: Math.hypot(a.x - b.x, a.y - b.y) };
    }
    function busy() { for (var k in claimed) return true; return false; }
    el.addEventListener('pointerdown', function (e) {
      var touch = e.pointerType === 'touch', id;
      // only a first finger can grab a handle: a second one joins the camera gesture (pinch, two-finger turn)
      if (o.claim && !ids.length && !busy() && o.claim(e)) { claimed[e.pointerId] = { x: e.clientX, y: e.clientY }; return; }
      if (touch && o.release) for (id in claimed) {         // a dense net: the first finger grabbed a point on its way to a pinch
        o.release(+id);
        pts[id] = { x: claimed[id].x, y: claimed[id].y, touch: true, ax: 0, ay: 0, live: true, dead: false };
        ids.push(+id); delete claimed[id];
      }
      // a pointer whose release never arrived (context menu, lost capture) must not linger
      ids = ids.filter(function (k) { return k !== e.pointerId && pts[k] && (touch || pts[k].touch); });
      for (id in pts) if (ids.indexOf(+id) < 0) delete pts[id];
      pts[e.pointerId] = { x: e.clientX, y: e.clientY, touch: touch, ax: 0, ay: 0, live: !touch, dead: false };
      ids.push(e.pointerId);
      try { el.setPointerCapture(e.pointerId); } catch (x) {}
      pair = spread();                   // new baseline: the second finger landing never jumps the camera
      if (o.start) o.start(e);
    });
    el.addEventListener('pointermove', function (e) {
      var p = pts[e.pointerId];
      if (claimed[e.pointerId]) { claimed[e.pointerId].x = e.clientX; claimed[e.pointerId].y = e.clientY; }
      if (!p) return;
      var dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (busy()) return;
      if (ids.length === 1) {
        if (!p.live) {                   // a finger has to travel a little before it turns anything
          p.ax += dx; p.ay += dy;
          if (Math.hypot(p.ax, p.ay) < SLOP) return;
          p.live = true;
          if (o.embed && Math.abs(p.ay) > Math.abs(p.ax)) p.dead = true;   // a vertical swipe: the page scrolls
          return;
        }
        if (p.dead) return;
        if (o.embed && p.touch) dy = 0;
        if (o.drag) o.drag(dx, dy, 1, e);
        return;
      }
      if (ids.indexOf(e.pointerId) > 1) return;
      var s = spread();
      if (pair) {
        if (o.drag) o.drag(s.x - pair.x, s.y - pair.y, 2, e);
        if (o.pinch && pair.d > 12 && s.d > 12) o.pinch(KT.clamp(s.d / pair.d, 0.5, 2));
      }
      pair = s;
    });
    function up(e) {
      delete claimed[e.pointerId];
      var i = ids.indexOf(e.pointerId); if (i < 0) return;
      delete pts[e.pointerId]; ids.splice(i, 1);
      ids = ids.filter(function (id) { return !!pts[id]; });
      ids.forEach(function (id) { pts[id].live = true; pts[id].dead = false; });  // the finger left behind keeps turning, from where it is
      pair = spread();
    }
    el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
    el.addEventListener('lostpointercapture', up);
    // desktop Safari reports a trackpad pinch as gesture events (no pointers down): zoom the stage with it.
    // On iOS the same events accompany a two-finger touch, which the pointer path above already handles.
    var gscale = 1;
    el.addEventListener('gesturestart', function () { gscale = 1; });
    el.addEventListener('gesturechange', function (e) {
      if (e.cancelable) e.preventDefault();
      if (!ids.length && o.pinch && e.scale > 0) { o.pinch(KT.clamp(e.scale / gscale, 0.5, 2)); gscale = e.scale; }
    });
    return { active: function () { return ids.length > 0; }, count: function () { return ids.length; } };
  };

  /* ---------- lifecycle ---------- */
  KT.mounted = [];                   // every live mount, newest last (debugging and tests)
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

    var handle = {
      name: name, def: def, api: api,
      setScene: function (id) { if (api.setScene) api.setScene(id); },
      destroy: function () {
        ro.disconnect(); io.disconnect(); document.removeEventListener('visibilitychange', onVis);
        if (raf) cancelAnimationFrame(raf);
        if (api.destroy) api.destroy();
        var i = KT.mounted.indexOf(handle); if (i >= 0) KT.mounted.splice(i, 1);
      }
    };
    KT.mounted.push(handle);
    return handle;
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
