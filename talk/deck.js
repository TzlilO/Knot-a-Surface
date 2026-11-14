/* Knot-a-Surface talk — scroll-driven deck engine.
   Steps (.step, plus the cover) are the unit of navigation: the one nearest the reading line is
   active, and its data-scene is pushed to the chapter's simulation. Keys: → / space next,
   ← previous, F focus the stage, T pace clock, 0–8 chapters. */
(function () {
  'use strict';
  var KT = window.KnotTalk;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  var NS = 'http://www.w3.org/2000/svg';

  /* ================= evidence data — edit here =================
     DTU: Table 1 of the paper (Chamfer distance, mm, lower is better; PGSR = its best results).
     `printedMean` is the mean as printed in the table; the exhibit recomputes the mean from the
     row and shows a warning while the two disagree. */
  var DTU = {
    scans: [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122],
    pgsr: [0.33, 0.51, 0.29, 0.28, 0.75, 0.53, 0.46, 0.92, 0.62, 0.48, 0.45, 0.55, 0.29, 0.33, 0.31],
    ours: [0.303, 0.480, 0.316, 0.329, 0.643, 0.552, 0.467, 0.801, 0.626, 0.558, 0.821, 0.572, 0.296, 0.346, 0.341],
    printedMean: { pgsr: 0.47, ours: 0.466 }
  };
  /* Surface-Prior bench (surface-prior.html): measured in-page on a procedural proxy capture. */
  var BENCH = [
    { name: 'Knot-a-Surface', prim: 12000, params: 96.0e3, chamfer: 72.0, closed: 95.7, step: '5.6 k', ours: true, lx: 10, ly: -10 },
    { name: 'PGSR', prim: 34138, params: 1.81e6, chamfer: 58.4, closed: 87.7, step: '20.6 k', lx: -9, ly: 16, anchor: 'end' },
    { name: '2DGS', prim: 40326, params: 2.06e6, chamfer: 58.5, closed: 84.2, step: '24.4 k', lx: 0, ly: -11, anchor: 'middle' },
    { name: 'SuGaR', prim: 31071, params: 1.65e6, chamfer: 216.3, closed: 87.7, step: '18.8 k', lx: -10, ly: 4, anchor: 'end' },
    { name: '3DGS', prim: 62488, params: 3.69e6, chamfer: 59.7, closed: 56.1, step: '28.1 k', lx: 9, ly: 14 },
    { name: 'NeuS / VolSDF', prim: 16472, params: 2.10e6, chamfer: 676.6, closed: 33.2, step: '24.4 k', lx: -10, ly: 4, anchor: 'end' },
    { name: 'MVS', prim: 19437, params: 116.6e3, chamfer: 201.2, closed: 69.2, step: 'no loop', lx: 10, ly: 4 }
  ];

  /* ================= equations ================= */
  if (window.katex) $$('[data-tex]').forEach(function (e) {
    try { window.katex.render(e.dataset.tex, e, { displayMode: e.classList.contains('eq'), throwOnError: false }); } catch (err) { e.textContent = e.dataset.tex; }
  });

  /* ================= simulations: mount when near, bind scenes ================= */
  var simSections = $$('section[data-sim]');
  simSections.forEach(function (sec) {
    var frame = $('.stage-frame', sec);
    if (!frame) return;
    var tools = el('div', 'stage-tools');
    var link = el('a', '', 'open module ↗'); link.href = 'talk/sims/' + sec.dataset.sim + '.html'; link.target = '_blank'; link.rel = 'noopener';
    var fb = el('button', '', 'focus · F'); fb.type = 'button'; fb.addEventListener('click', function () { setFocus(!document.body.classList.contains('focus')); });
    tools.appendChild(link); tools.appendChild(fb); frame.appendChild(tools);
    sec._link = link;
  });
  function mountSim(sec) {
    if (sec._sim || !KT.sims[sec.dataset.sim]) return;
    sec._sim = KT.mount(sec.dataset.sim, $('.sim-mount', sec), { embed: true, scene: sec._cur || sec.dataset.scene });
    if (sec._cur) sec._sim.setScene(sec._cur);
  }
  var near = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) mountSim(e.target); });
  }, { rootMargin: '120% 0px' });
  simSections.forEach(function (s) { near.observe(s); });

  /* ================= chapter rail ================= */
  var chapters = $$('section[data-rail]'), rail = $('.rail');
  chapters.forEach(function (sec, i) {
    var a = el('a', '', sec.dataset.rail + '<span>' + sec.dataset.title + '</span>');
    a.href = '#'; a.addEventListener('click', function (e) { e.preventDefault(); go(firstNav(sec)); });
    rail.appendChild(a); sec._rail = a; sec._idx = i;
  });
  function firstNav(sec) { return sec.matches('.nav') ? sec : $('.nav', sec); }

  /* ================= steps ================= */
  var allNav = $$('.nav'), cur = null, mobile = window.matchMedia('(max-width:900px)');
  allNav.forEach(function (n, i) { n._i = i; });
  function visibleNav() { return allNav.filter(function (n) { return n.offsetParent !== null; }); }

  function activate(n) {
    if (n === cur) return;
    if (cur) cur.classList.remove('on');
    cur = n; n.classList.add('on');
    var sec = n.closest('section'), sc = n.dataset.scene;
    if (sc && sec.dataset.sim && (sec._cur !== sc || n.hasAttribute('data-replay'))) {
      sec._cur = sc;
      if (sec._sim) sec._sim.setScene(sc);
      if (sec._link) sec._link.href = 'talk/sims/' + sec.dataset.sim + '.html#' + sc;
    }
    chapters.forEach(function (c) { c._rail.classList.toggle('on', c === sec || c.contains(n)); });
    $$('.stage-frame.is-current').forEach(function (f) { f.classList.remove('is-current'); });
    var fr = $('.stage-frame', sec); if (fr) fr.classList.add('is-current'); else setFocus(false);
    try { history.replaceState(null, '', '#s' + n._i); } catch (e) {}
    tick();
  }

  var bar = $('.progress'), queued = false;
  function onScroll() {
    if (queued) return; queued = true;
    requestAnimationFrame(function () {
      queued = false;
      var small = mobile.matches, y = window.innerHeight * (small ? 0.62 : 0.5), best = null, bd = 1e9;
      visibleNav().forEach(function (n) {
        // wide screens: the step centred on the reading line. small screens (and over-tall steps): the step the line is inside.
        var r = n.getBoundingClientRect(), c = (small || r.height > window.innerHeight) ? KT.clamp(y, r.top, r.bottom) : (r.top + r.bottom) / 2;
        var d = Math.abs(c - y); if (d < bd) { bd = d; best = n; }
      });
      if (best) activate(best);
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = 'scaleX(' + (max > 0 ? window.scrollY / max : 0) + ')';
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);

  function go(n, instant) {
    if (!n) return;
    var r = n.getBoundingClientRect(), top;
    if (n.classList.contains('cover') || r.height > window.innerHeight * 0.92) top = window.scrollY + r.top;
    else if (mobile.matches && n.closest('.chapter')) top = window.scrollY + r.top - window.innerHeight * 0.52;   // just under the pinned stage
    else top = window.scrollY + (r.top + r.bottom) / 2 - window.innerHeight * 0.5;
    window.scrollTo({ top: Math.max(0, top), behavior: instant ? 'auto' : 'smooth' });
  }
  function stepBy(d) {
    var v = visibleNav(), i = v.indexOf(cur);
    if (i < 0) i = 0;
    var n = v[KT.clamp(i + d, 0, v.length - 1)];
    if (d > 0 && !clock.t0) startClock();
    go(n);
  }

  /* ================= cut length + pace clock ================= */
  var clock = { t0: 0, el: $('.clock') }, cutBtns = $$('.cut button');
  function cut() { return document.body.classList.contains('cut-20') ? 20 : 40; }
  function setCut(m) {
    document.body.classList.toggle('cut-20', m === 20);
    cutBtns.forEach(function (b) { b.classList.toggle('on', +b.dataset.cut === m); });
    try { localStorage.setItem('kt-cut', m); } catch (e) {}
    if (cur && cur.offsetParent === null) {            // the active step was a deep-dive: fall back to its predecessor
      var i = cur._i; while (i > 0 && allNav[i].offsetParent === null) i--;
      go(allNav[i], true);
    } else if (cur) go(cur, true);
    onScroll(); tick();
  }
  cutBtns.forEach(function (b) { b.addEventListener('click', function () { setCut(+b.dataset.cut); }); });
  function startClock() { clock.t0 = Date.now(); tick(); }
  clock.el.addEventListener('click', function () { if (clock.t0) { clock.t0 = 0; tick(); } else startClock(); });
  function mmss(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  /* planned time at the active step: chapter budgets (data-min40 / data-min20), linear inside a chapter */
  function plan() {
    var key = cut() === 20 ? 'min20' : 'min40', t = 0, done = false;
    chapters.forEach(function (c) {
      if (done) return;
      var m = +c.dataset[key] || 0;
      if (cur && (c === cur || c.contains(cur))) {
        var navs = $$('.nav', c).filter(function (n) { return n.offsetParent !== null; });
        var i = Math.max(0, navs.indexOf(cur));
        t += m * (navs.length ? (i + 1) / navs.length : 1); done = true;
      } else t += m;
    });
    return t * 60;
  }
  function tick() {
    var p = plan();
    if (!clock.t0) { clock.el.className = 'clock'; clock.el.innerHTML = 'T · start clock · plan ' + mmss(p); return; }
    var e = (Date.now() - clock.t0) / 1000;
    clock.el.className = 'clock ' + (e > p + 45 ? 'late' : 'ok');
    clock.el.innerHTML = '<b>' + mmss(e) + '</b> · plan ' + mmss(p) + ' / ' + cut() + '′';
  }
  setInterval(tick, 1000);

  /* ================= focus mode + keys ================= */
  function setFocus(on) { document.body.classList.toggle('focus', !!on && !!$('.stage-frame.is-current')); }
  var keys = $('.keys');
  $('.help').addEventListener('click', function () { keys.classList.toggle('on'); });
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;       // sliders keep their arrows
    if (tag === 'BUTTON' && (e.key === ' ' || e.key === 'Enter')) return;
    var k = e.key;
    if (k === 'ArrowRight' || k === 'ArrowDown' || k === 'PageDown' || k === ' ') { e.preventDefault(); stepBy(e.shiftKey && k === ' ' ? -1 : 1); }
    else if (k === 'ArrowLeft' || k === 'ArrowUp' || k === 'PageUp') { e.preventDefault(); stepBy(-1); }
    else if (k === 'Home') { e.preventDefault(); go(allNav[0]); }
    else if (k === 'End') { e.preventDefault(); var v = visibleNav(); go(v[v.length - 1]); }
    else if (k === 'f' || k === 'F') setFocus(!document.body.classList.contains('focus'));
    else if (k === 'Escape') { setFocus(false); keys.classList.remove('on'); }
    else if (k === 't' || k === 'T') { if (clock.t0) clock.t0 = 0; else startClock(); tick(); }
    else if (k === '?') keys.classList.toggle('on');
    else if (k >= '0' && k <= '9') { var c = chapters[+k]; if (c) go(firstNav(c)); }
  });

  /* ================= exhibits ================= */
  function svg(tag, at, parent, text) {
    var e = document.createElementNS(NS, tag), k;
    for (k in at) e.setAttribute(k, at[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  var INK = { text: '#dde5f0', dim: '#8fa4c0', mute: '#5c6f8a', line: '#1b2a42', bg: '#0b1017', ours: '#4fc3f7' };
  var MONO = 'font-family:var(--f-mono);font-size:10.5px;font-weight:500';
  function tipFor(box) {
    var t = el('div', 'tip'); box.appendChild(t);
    return { show: function (html, x, y) {
        t.innerHTML = html; t.classList.add('on');
        var bw = box.clientWidth, tw = t.offsetWidth;
        t.style.left = KT.clamp(x - tw / 2, 4, bw - tw - 4) + 'px'; t.style.top = (y - t.offsetHeight - 12) + 'px';
      }, hide: function () { t.classList.remove('on'); } };
  }
  function mean(a) { return a.reduce(function (s, v) { return s + v; }, 0) / a.length; }

  function dtuChart(box) {
    var n = DTU.scans.length, mo = mean(DTU.ours), mp = mean(DTU.pgsr), wins = 0, close = 0, i;
    for (i = 0; i < n; i++) { if (DTU.ours[i] < DTU.pgsr[i]) wins++; if (Math.abs(DTU.ours[i] - DTU.pgsr[i]) <= 0.05) close++; }
    $('.stats', box).innerHTML =
      '<div><b>' + mo.toFixed(3) + '</b>ours · mean mm</div><div><b>' + mp.toFixed(3) + '</b>PGSR · mean mm</div>' +
      '<div><b>' + close + ' / ' + n + '</b>scans within 0.05 mm</div><div><b>' + wins + ' / ' + n + '</b>scans where ours is lower</div>';
    var warn = $('.check', box), bad = [];
    if (Math.abs(mo - DTU.printedMean.ours) > 0.0015) bad.push('ours: the row averages ' + mo.toFixed(3) + ', the table prints ' + DTU.printedMean.ours);
    if (Math.abs(mp - DTU.printedMean.pgsr) > 0.006) bad.push('PGSR: the row averages ' + mp.toFixed(3) + ', the table prints ' + DTU.printedMean.pgsr);
    warn.style.display = bad.length ? '' : 'none';
    warn.innerHTML = '<b>Check before presenting</b> — ' + bad.join(' · ') + '. Fix the row or the mean in <span class="mono">talk/deck.js</span>.';

    var W = 660, H = 290, L = 40, Rm = 10, Tm = 12, B = 34, y0 = 0.2, y1 = 1.0;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Chamfer distance per DTU scan, ours versus PGSR' }, $('.plot', box));
    function X(k) { return L + (k + 0.5) * (W - L - Rm) / n; }
    function Y(v) { return Tm + (y1 - v) / (y1 - y0) * (H - Tm - B); }
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
      svg('path', { d: 'M' + L + ' ' + Y(v) + 'H' + (W - Rm), stroke: INK.line, 'stroke-width': 1 }, s);
      svg('text', { x: L - 8, y: Y(v) + 3.5, 'text-anchor': 'end', fill: INK.mute, style: MONO }, s, v.toFixed(1));
    });
    svg('text', { x: L, y: H - 4, fill: INK.mute, style: MONO }, s, 'DTU scan');
    var tip = tipFor(box), hot = svg('rect', { y: Tm, height: H - Tm - B, fill: '#ffffff', opacity: 0, rx: 4 }, s);
    for (i = 0; i < n; i++) {
      var x = X(i), a = DTU.ours[i], b = DTU.pgsr[i];
      svg('path', { d: 'M' + x + ' ' + Y(a) + 'V' + Y(b), stroke: INK.mute, 'stroke-width': 2, 'stroke-linecap': 'round' }, s);
      svg('circle', { cx: x, cy: Y(b), r: 5, fill: INK.bg, stroke: INK.dim, 'stroke-width': 2 }, s);
      svg('circle', { cx: x, cy: Y(a), r: 6, fill: INK.ours, stroke: INK.bg, 'stroke-width': 2 }, s);
      svg('text', { x: x, y: H - B + 16, 'text-anchor': 'middle', fill: INK.dim, style: MONO }, s, DTU.scans[i]);
    }
    var bw = (W - L - Rm) / n;
    s.addEventListener('pointermove', function (e) {
      var r = s.getBoundingClientRect(), k = Math.floor(((e.clientX - r.left) / r.width * W - L) / bw);
      if (k < 0 || k >= n) { tip.hide(); hot.setAttribute('opacity', 0); return; }
      hot.setAttribute('x', L + k * bw); hot.setAttribute('width', bw); hot.setAttribute('opacity', 0.05);
      var d = DTU.ours[k] - DTU.pgsr[k], pr = box.getBoundingClientRect();
      tip.show('scan <b>' + DTU.scans[k] + '</b><br>ours <b>' + DTU.ours[k].toFixed(3) + '</b> · PGSR <b>' + DTU.pgsr[k].toFixed(2) + '</b><br>Δ ' + (d > 0 ? '+' : '−') + Math.abs(d).toFixed(3) + ' mm',
        r.left - pr.left + X(k) / W * r.width, r.top - pr.top + Y(Math.max(DTU.ours[k], DTU.pgsr[k])) / H * r.height);
    });
    s.addEventListener('pointerleave', function () { tip.hide(); hot.setAttribute('opacity', 0); });
  }

  function benchChart(box) {
    var W = 560, H = 290, L = 46, Rm = 16, Tm = 12, B = 38;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Optimised parameters versus Chamfer distance on the proxy bench' }, $('.plot', box));
    var lx0 = Math.log10(5e4), lx1 = Math.log10(8e6), ly0 = Math.log10(40), ly1 = Math.log10(1000);
    function X(v) { return L + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - L - Rm); }
    function Y(v) { return H - B - (Math.log10(v) - ly0) / (ly1 - ly0) * (H - Tm - B); }
    [[1e5, '100 k'], [1e6, '1 M'], [5e6, '5 M']].forEach(function (t) {
      svg('path', { d: 'M' + X(t[0]) + ' ' + Tm + 'V' + (H - B), stroke: INK.line, 'stroke-width': 1 }, s);
      svg('text', { x: X(t[0]), y: H - B + 15, 'text-anchor': 'middle', fill: INK.mute, style: MONO }, s, t[1]);
    });
    [50, 100, 200, 500].forEach(function (v) {
      svg('path', { d: 'M' + L + ' ' + Y(v) + 'H' + (W - Rm), stroke: INK.line, 'stroke-width': 1 }, s);
      svg('text', { x: L - 8, y: Y(v) + 3.5, 'text-anchor': 'end', fill: INK.mute, style: MONO }, s, v);
    });
    svg('text', { x: W - Rm, y: H - 5, 'text-anchor': 'end', fill: INK.mute, style: MONO }, s, 'optimised parameters · log →');
    svg('text', { x: L, y: H - 5, fill: INK.mute, style: MONO }, s, '↓ Chamfer mm · log');
    var tip = tipFor(box);
    BENCH.forEach(function (m) {
      var x = X(m.params), y = Y(m.chamfer);
      svg('circle', m.ours ? { cx: x, cy: y, r: 7, fill: INK.ours, stroke: INK.bg, 'stroke-width': 2 }
                           : { cx: x, cy: y, r: 5, fill: INK.bg, stroke: INK.dim, 'stroke-width': 2 }, s);
      svg('text', { x: x + m.lx, y: y + m.ly, 'text-anchor': m.anchor || 'start', fill: m.ours ? INK.text : INK.dim,
        style: MONO + (m.ours ? ';font-weight:700;font-size:11.5px' : '') }, s, m.name);
      var h = svg('circle', { cx: x, cy: y, r: 16, fill: '#fff', opacity: 0 }, s);
      h.addEventListener('pointerenter', function () {
        var r = s.getBoundingClientRect(), pr = box.getBoundingClientRect();
        tip.show('<b>' + m.name + '</b><br>' + m.prim.toLocaleString('en-US') + ' primitives · <b>' + (m.params >= 1e6 ? (m.params / 1e6).toFixed(2) + ' M' : (m.params / 1e3).toFixed(1) + ' k') +
          '</b> params<br>Chamfer <b>' + m.chamfer.toFixed(1) + ' mm</b> · closed ' + m.closed + '%<br>converged at step ' + m.step,
          r.left - pr.left + x / W * r.width, r.top - pr.top + y / H * r.height);
      });
      h.addEventListener('pointerleave', tip.hide);
    });
  }
  var ex = $('#ex-dtu'); if (ex) dtuChart(ex);
  ex = $('#ex-bench'); if (ex) benchChart(ex);

  /* ================= boot ================= */
  var saved = 40;
  try { saved = +localStorage.getItem('kt-cut') === 20 ? 20 : 40; } catch (e) {}
  document.body.classList.toggle('cut-20', saved === 20);
  cutBtns.forEach(function (b) { b.classList.toggle('on', +b.dataset.cut === saved); });
  var m = /^#s(\d+)$/.exec(location.hash);
  if (m && allNav[+m[1]] && allNav[+m[1]].offsetParent !== null) {
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    requestAnimationFrame(function () { go(allNav[+m[1]], true); onScroll(); });
  }
  onScroll(); tick();
})();
