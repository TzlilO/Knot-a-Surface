/* Knot-a-Surface — scroll engine for the short Hebrew deck (talk-he.html).
   One step is active at a time (the one nearest the middle of the screen); its data-scene goes to
   the chapter's simulator. Right-to-left: ← moves forward, → moves back; space / ↓ also advance. */
(function () {
  'use strict';
  var KT = window.KnotTalk;
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  var small = window.matchMedia('(max-width:900px)');

  /* ---------- simulators ---------- */
  function mount(sec) {
    if (sec._sim) return;
    var el = sec.querySelector('.sim-mount'), name = sec.dataset.sim;
    if (!el || !KT.sims[name]) return;
    if (name === 'manifold') {                       // the 3D scene: the finale module without its own chrome
      var num = sec.querySelector('.he-read'), sl = sec.querySelector('input[type=range]');
      if (Math.min(window.innerWidth, window.innerHeight) < 560) sl.value = 24000;
      sec._sim = KT.mount('manifold', el, { embed: true, bare: true, scene: 'hero', clear: 'transparent', budget: +sl.value,
        onStats: function (s) { num.innerHTML = '<b>' + Math.round(s.drawn).toLocaleString('he-IL') + '</b>גאוסיאנים על המסך'; } });
      sec._sim.api.load('breakfast');
      sl.addEventListener('input', function () { sec._sim.api.setBudget(+sl.value); });
      sl.addEventListener('pointerup', function () { sl.blur(); });
      return;
    }
    sec._sim = KT.mount(name, el, { embed: true, scene: sec._cur || sec.dataset.scene });
  }
  var near = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) mount(e.target); }); }, { rootMargin: '100% 0px' });
  $$('[data-sim]').forEach(function (s) { near.observe(s); });

  /* ---------- steps ---------- */
  var nav = $$('.nav'), cur = null, dots = document.querySelector('.dots');
  var prev = document.querySelector('.dock .prev'), next = document.querySelector('.dock .next');
  nav.forEach(function (n, i) {
    n._i = i;
    var d = document.createElement('i'); d.addEventListener('click', function () { go(n); }); dots.appendChild(d); n._dot = d;
  });
  function activate(n) {
    if (n === cur) return;
    if (cur) { cur.classList.remove('on'); cur._dot.classList.remove('on'); }
    cur = n; n.classList.add('on'); n._dot.classList.add('on');
    var sec = n.closest('[data-sim]'), sc = n.dataset.scene;
    if (sec && sc && sec._cur !== sc) { sec._cur = sc; if (sec._sim) sec._sim.setScene(sc); }
    prev.disabled = n._i === 0; next.disabled = n._i === nav.length - 1;
  }
  var queued = false;
  function onScroll() {
    if (queued) return; queued = true;
    requestAnimationFrame(function () {
      queued = false;
      var y = window.innerHeight * (small.matches ? 0.64 : 0.5), best = null, bd = 1e9;
      nav.forEach(function (n) {
        var r = n.getBoundingClientRect(), c = (small.matches || r.height > window.innerHeight) ? KT.clamp(y, r.top, r.bottom) : (r.top + r.bottom) / 2;
        var d = Math.abs(c - y); if (d < bd) { bd = d; best = n; }
      });
      if (best) activate(best);
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);
  function go(n) {
    var r = n.getBoundingClientRect(), top;
    if (n._i === 0 || r.height > window.innerHeight * 0.92) top = window.scrollY + r.top;
    else if (small.matches && n.closest('.chapter')) top = window.scrollY + r.top - window.innerHeight * 0.52;
    else top = window.scrollY + (r.top + r.bottom) / 2 - window.innerHeight / 2;
    window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  }
  function step(d) { var i = cur ? cur._i : 0; go(nav[KT.clamp(i + d, 0, nav.length - 1)]); }
  prev.addEventListener('click', function () { step(-1); });
  next.addEventListener('click', function () { step(1); });
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target && e.target.tagName;
    if (t === 'INPUT' || (t === 'BUTTON' && (e.key === ' ' || e.key === 'Enter'))) return;
    var k = e.key;
    if (k === ' ' || k === 'ArrowDown' || k === 'PageDown' || k === 'ArrowLeft') { e.preventDefault(); step(e.shiftKey && k === ' ' ? -1 : 1); }
    else if (k === 'ArrowUp' || k === 'PageUp' || k === 'ArrowRight') { e.preventDefault(); step(-1); }
    else if (k === 'Home') { e.preventDefault(); go(nav[0]); }
    else if (k === 'End') { e.preventDefault(); go(nav[nav.length - 1]); }
  });
  onScroll();
})();
