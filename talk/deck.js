/* Knot-a-Surface talk — scroll-driven deck engine.
   Steps (.step, plus the cover) are the unit of navigation: the one nearest the reading line is
   active, and its data-scene is pushed to the chapter's simulation. Keys: → / space next,
   ← previous, F focus the stage, T pace clock, S titles only (see "subtitles"), 0 the Knot-a-Surface cover, 1–8
   chapters, Home the opening, L the language (talk/i18n.js: English or Hebrew; in Hebrew the two arrows swap, ← is next).
   The opening ("How does a computer see in 3D?") and the cover sit in a light wrapper, .prelude
   (talk/prelude.css); while a step inside it is active, body.lightroom turns the fixed chrome light too.
   Parts of the talk can be switched off (data-off in talk.html, or the slide master, talk/master.js): see "parts"
   below. KnotTalk.deck is the handle the slide master (and tests) use. */
(function () {
  'use strict';
  var KT = window.KnotTalk;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  var NS = 'http://www.w3.org/2000/svg';
  /* talk.html?master (or #master) is the slide master's presenter view (talk/master.js): the page shows its panel only
     (master.css hides the rest under body.kt-presenter), so the deck here mounts no simulation and takes no step. Set
     before anything below can run a frame. */
  var presenter = /(?:^|&)(?:master|prompter)(?:[=&]|$)/.test(location.search.slice(1)) || location.hash === '#master' || location.hash === '#prompter';   // ?prompter: the same view as a teleprompter (master.js)
  if (presenter) document.body.classList.add('kt-presenter');

  /* ================= language =================
     talk/i18n.js decides it (KT.lang, KT.rtl) and swaps the page's copy: talk.html calls apply() right after its
     copy, and the call here takes whatever that one could not see, before anything below reads the page (the rail's
     titles, the KaTeX placeholders a translated block still holds). Every string this file puts on screen goes
     through KT.t. Hebrew reads right to left, so its two arrow keys swap (← next, → previous, as in talk-he.html)
     and the help panel shows them that way. */
  var I18N = KT.i18n || null, RTL = !!KT.rtl;
  if (I18N) I18N.apply();
  if (RTL) {
    $$('.keys .k-fwd').forEach(function (e) { e.textContent = '←'; });
    $$('.keys .k-back').forEach(function (e) { e.textContent = '→'; });
  }

  /* ================= evidence data — edit here =================
     DTU: Table 1 of the paper (Chamfer distance, mm, lower is better; PGSR = its best results).
     `printedMean` is the mean as printed in the table; the exhibit recomputes the mean from the
     row and adds a one-line note while the two disagree. */
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
    var link = el('a', '', KT.t('open module ↗')); link.href = 'talk/sims/' + sec.dataset.sim + '.html'; link.target = '_blank'; link.rel = 'noopener';
    var fb = el('button', '', KT.t('focus · F')); fb.type = 'button'; fb.addEventListener('click', function () { setFocus(!document.body.classList.contains('focus')); });
    tools.appendChild(link); tools.appendChild(fb); frame.appendChild(tools);
    sec._link = link;
  });
  function mountSim(sec) {
    if (sec._sim || !KT.sims[sec.dataset.sim]) return;
    // a section may add mount options as JSON in data-mount (no chrome, a transparent canvas, a smaller capacity...);
    // the numbers a module reports are passed on as a `kt-stats` event on the section
    var o = { embed: true, scene: sec._cur || sec.dataset.scene,
              onStats: function (st) { sec.dispatchEvent(new CustomEvent('kt-stats', { detail: st })); } };
    if (sec.dataset.mount) try { Object.assign(o, JSON.parse(sec.dataset.mount)); } catch (e) { /* malformed: mount as usual */ }
    sec._sim = KT.mount(sec.dataset.sim, $('.sim-mount', sec), o);
    if (sec._cur) sec._sim.setScene(sec._cur);
  }
  var near = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) mountSim(e.target); });
  }, { rootMargin: '120% 0px' });
  if (!presenter) simSections.forEach(function (s) { near.observe(s); });

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
  allNav.forEach(function (n, i) { n._i = i; n._sec = n.closest('section[data-rail]') || n.closest('section') || n; });
  function visibleNav() { return allNav.filter(function (n) { return n.offsetParent !== null; }); }
  function emit(type, detail) { document.dispatchEvent(new CustomEvent(type, { detail: detail })); }

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
    document.body.classList.toggle('lightroom', !!n.closest('.prelude'));
    $$('.stage-frame.is-current').forEach(function (f) { f.classList.remove('is-current'); });
    var fr = $('.stage-frame', sec); if (fr) fr.classList.add('is-current'); else setFocus(false);
    try { history.replaceState(null, '', '#s' + n._i); } catch (e) {}
    sound(n);
    tick();
    emit('kt-step', { step: n });
  }

  /* ================= a step may carry a sound =================
     <article ... data-audio="name" data-audio-len="30">: while such a step is active the first seconds of the
     track loop, faded in and out. Names are resolved in KnotTalk.tracks (talk/audio/tracks.js); a name that is
     not listed there is silent, and the mute button and the credit line (.credit) stay hidden. M mutes.
     <article ... data-sfx>: the step's simulation makes sound effects of its own (synthesized, e.g. chapter 09, "Can’t touch this");
     the button shows there too, and the same mute reaches them through KnotTalk.muted.
     Browsers only start sound after a key or a click. */
  var snd = { a: null, src: '', len: 30, ok: false, want: false, mute: false, vol: 0, raf: 0, btn: $('.sound') };
  function sound(n) {
    var src = (KT.tracks || {})[n.dataset.audio] || '';
    if (src && src !== snd.src) {
      if (snd.a) snd.a.pause();
      snd.src = src; snd.ok = false; snd.vol = 0; snd.len = +n.dataset.audioLen || 30;
      var a = snd.a = new Audio(src); a.preload = 'auto'; a.volume = 0;
      a.addEventListener('canplay', function () { if (a === snd.a) { snd.ok = true; paintSound(); fade(); } });
      a.addEventListener('timeupdate', function () { if (a.currentTime >= snd.len) a.currentTime = 0; });
      a.addEventListener('ended', function () { a.currentTime = 0; a.play().catch(function () {}); });
    }
    snd.want = !!src; paintSound(); fade();
  }
  function paintSound() {
    document.body.classList.toggle('has-sound', snd.ok);
    // the button also shows on a step whose simulation makes its own sound (data-sfx); KT.muted silences that too
    snd.btn.hidden = !((snd.ok && snd.want) || (cur && cur.hasAttribute('data-sfx'))); snd.btn.classList.toggle('off', snd.mute);
    KT.muted = snd.mute;
  }
  function fade() {
    cancelAnimationFrame(snd.raf);
    var a = snd.a, goal = snd.want && !snd.mute ? 0.6 : 0, last = performance.now();
    if (!a || !snd.ok) return;
    if (goal && a.paused) a.play().catch(function () {});     // refused before the first key or click: retried on the next one
    (function stepFade(now) {
      var d = Math.max(0, now - last) / 600; last = now;      // 0.6 s from silence to full
      snd.vol = goal > snd.vol ? Math.min(goal, snd.vol + d) : Math.max(goal, snd.vol - d);
      a.volume = snd.vol;
      if (snd.vol !== goal) snd.raf = requestAnimationFrame(stepFade); else if (!goal) a.pause();
    })(last);
  }
  snd.btn.addEventListener('click', function () { snd.mute = !snd.mute; paintSound(); fade(); });
  ['keydown', 'pointerdown'].forEach(function (ev) { document.addEventListener(ev, function () { if (snd.want && snd.ok && !snd.mute && snd.a.paused) fade(); }); });

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
      if (best && !presenter) activate(best);
      var max = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.transform = 'scaleX(' + (max > 0 ? window.scrollY / max : 0) + ')';
    });
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll);

  function go(n, instant) {
    if (n && n.offsetParent === null) n = nearest(n, 1);   // a part that is off: where it would be, the next step shown
    if (!n) return;
    var r = n.getBoundingClientRect(), top;
    if (n.classList.contains('cover') || r.height > window.innerHeight * 0.92) top = window.scrollY + r.top;
    else if (mobile.matches && n.closest('.chapter')) top = window.scrollY + r.top - window.innerHeight * 0.52;   // just under the pinned stage
    else top = window.scrollY + (r.top + r.bottom) / 2 - window.innerHeight * 0.5;
    window.scrollTo({ top: Math.max(0, top), behavior: instant ? 'auto' : 'smooth' });
  }
  function stepBy(d) {
    var c = cur && cur.offsetParent === null ? nearest(cur, -1) : cur;   // the active step was just switched off: from its predecessor
    var v = visibleNav(), i = v.indexOf(c);
    if (i < 0) i = 0;
    if (c && !c.classList.contains('cover')) {        // taller than the screen: show the rest of it first
      var r = c.getBoundingClientRect(), vh = window.innerHeight;
      if (d > 0 && r.bottom > vh + 12) { window.scrollBy({ top: Math.min(r.bottom - vh + 8, vh * 0.8), behavior: 'smooth' }); return; }
      if (d < 0 && r.top < -12) { window.scrollBy({ top: -Math.min(-r.top + 8, vh * 0.8), behavior: 'smooth' }); return; }
    }
    var n = v[KT.clamp(i + d, 0, v.length - 1)];
    if (d > 0 && !clock.t0) startClock();
    go(n);
  }

  /* ================= cut length + pace clock ================= */
  var clock = { t0: 0, el: $('.clock') }, cutBtns = $$('.cut button');
  function cut() { return document.body.classList.contains('cut-20') ? 20 : 40; }
  /* the planned length of a cut, from the chapter budgets: the buttons and the clock show it, so they stay true
     whatever the chapters add up to, and whatever is switched off (see "parts" below) */
  function total(m) { return chapters.reduce(function (t, c) { return t + budget(c, m); }, 0); }
  function setCut(m) {
    document.body.classList.toggle('cut-20', m === 20);
    cutBtns.forEach(function (b) { b.classList.toggle('on', +b.dataset.cut === m); });
    try { localStorage.setItem('kt-cut', m); } catch (e) {}
    layout(); settle(); onScroll(); tick();
    emit('kt-cut', { cut: m });
  }
  cutBtns.forEach(function (b) { b.addEventListener('click', function () { setCut(+b.dataset.cut); }); });
  function startClock() { clock.t0 = Date.now(); tick(); emit('kt-clock', { t0: clock.t0 }); }
  function toggleClock() { if (clock.t0) { clock.t0 = 0; tick(); emit('kt-clock', { t0: 0 }); } else startClock(); }
  clock.el.addEventListener('click', toggleClock);
  function mmss(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  /* planned time at the active step: chapter budgets (data-min40 / data-min20), linear inside a chapter */
  function plan() {
    var m = cut(), t = 0, done = false;
    chapters.forEach(function (c) {
      if (done) return;
      if (cur && (c === cur || c.contains(cur))) {
        var navs = navsOf(c).filter(function (n) { return shown(n, m); });
        var i = Math.max(0, navs.indexOf(cur));
        t += budget(c, m) * (navs.length ? (i + 1) / navs.length : 1); done = true;
      } else t += budget(c, m);
    });
    return t * 60;
  }

  /* ================= parts: what is on =================
     A step is on screen when the cut shows it (the short cut hides data-depth="deep" steps) AND it is on. Every
     step is on unless talk.html marks it, or its section, data-off (off until the presenter turns it on); the
     presenter's overrides, set in the slide master (talk/master.js), win over those defaults and are all it saves.
     A step that is off leaves the flow as a deep step does in the short cut: display:none (deck.css), so it is
     neither navigable nor counted by the clock and the cut totals. A section with nothing left on screen goes,
     with its sticky stage and its rail entry, and the light wrapper goes when none of its sections is left.
     A section's budget is shared evenly by the steps the cut shows, so what is off comes off the totals.
     A step's key is its section's rail label and its title: a saved choice survives edits elsewhere in the page.
     The key is made from the ENGLISH title in every language (the page as written: i18n.source, i18n.attrEn), so a
     choice made in one language holds in the other and windows in different languages understand each other; the
     title that is shown (n._title, the slide master's rows) is in the language of the page. */
  var over = {}, byKey = {}, prelude = $('.prelude');
  function navsOf(sec) { return sec._navs || (sec._navs = sec.matches('.nav') ? [sec] : $$('.nav', sec)); }
  function textOf(e) {
    var c = e.cloneNode(true);
    $$('[data-tex]', c).forEach(function (t) { t.textContent = t.dataset.tex; });      // the TeX, not KaTeX's rendering
    return c.textContent.replace(/\s+/g, ' ').trim();
  }
  allNav.forEach(function (n) {                    // a title: the step's heading, else its thesis line, else its label
    var h = $('h1, h2', n) || $('.thesis, .qa', n) || $('.lvl, .tag, .kick', n), t = h ? textOf(h) : '';
    var en = h ? (I18N ? textOf(I18N.source(h)) : t) : '', at = navsOf(n._sec).indexOf(n) + 1;
    n._title = t || (n._sec.dataset.title || KT.t('Step')) + ' · ' + at;
    if (!en) en = ((I18N ? I18N.attrEn(n._sec, 'data-title') : n._sec.dataset.title) || 'Step') + ' · ' + at;
    var base = (n._sec.dataset.rail || '') + ' · ' + en, k = base, j = 1;
    while (byKey[k]) k = base + ' #' + (++j);
    n._key = k; byKey[k] = n;
  });
  function defaultOn(n) { return !n.hasAttribute('data-off') && !n._sec.hasAttribute('data-off'); }
  /* design "knob" (talk/atlas.js, talk.html?design=knob): a sampled set stands in for the cut, smp = { has(step), cost(step)
     minutes }; it is transient (never saved here), the presenter's overrides still win, and data-off is the sampler's to
     decide. null (the classic talk): nothing below changes */
  var smp = null;
  function isOn(n) { return Object.prototype.hasOwnProperty.call(over, n._key) ? over[n._key] : smp ? true : defaultOn(n); }
  function inCut(n, m) { return smp ? smp.has(n) : m !== 20 || !n.matches('.step[data-depth="deep"]'); }
  function shown(n, m) { return isOn(n) && inCut(n, m || cut()); }
  function minutes(sec, m) { return +sec.dataset[m === 20 ? 'min20' : 'min40'] || 0; }
  function inCutOf(sec, m) { return navsOf(sec).filter(function (x) { return inCut(x, m); }); }
  function share(n, m) { if (smp) return smp.cost(n); var c = inCutOf(n._sec, m); return inCut(n, m) && c.length ? minutes(n._sec, m) / c.length : 0; }
  function budget(sec, m) {
    if (smp) return navsOf(sec).reduce(function (t, x) { return t + (shown(x) ? smp.cost(x) : 0); }, 0);
    var c = inCutOf(sec, m); return c.length ? minutes(sec, m) * c.filter(isOn).length / c.length : 0;
  }
  function set(n, on) { if (!!on === defaultOn(n)) delete over[n._key]; else over[n._key] = !!on; }
  function layout() {
    var m = cut();
    allNav.forEach(function (n) { var on = isOn(n); n.classList.toggle('kt-on', on); n.classList.toggle('kt-off', !on); n.classList.toggle('kt-unsampled', !!smp && !smp.has(n)); });
    chapters.forEach(function (sec) {
      var navs = navsOf(sec), gone = !navs.some(function (n) { return shown(n, m); });
      sec.classList.toggle('kt-on', navs.some(isOn));             // a data-off section that something of is on
      sec.classList.toggle('kt-gone', gone); sec._rail.classList.toggle('kt-gone', gone);
    });
    if (prelude) prelude.classList.toggle('kt-gone', !$$('section[data-rail]', prelude).some(function (s) { return !s.classList.contains('kt-gone'); }));
    cutBtns.forEach(function (b) { b.textContent = Math.round(total(+b.dataset.cut)) + '′'; });
  }
  /* the nearest step on screen from n, looking in direction d first (1 forward, -1 back), then the other way */
  function nearest(n, d) {
    for (var k = 0; k < 2; k++, d = -d)
      for (var i = n._i; i >= 0 && i < allNav.length; i += d) if (allNav[i].offsetParent !== null) return allNav[i];
    return null;
  }
  /* after the flow changed: keep the active step on the reading line; one that has just gone (a deep-dive in the
     short cut, a part switched off) falls back to its predecessor */
  function settle() { if (cur) go(cur.offsetParent === null ? nearest(cur, -1) : cur, true); }
  function refresh() { layout(); settle(); onScroll(); tick(); emit('kt-parts', { over: Object.assign({}, over) }); }
  function tick() {
    var p = plan();
    if (!clock.t0) { clock.el.className = 'clock'; clock.el.innerHTML = KT.t('T · start clock · plan {t}', { t: mmss(p) }); return; }
    var e = (Date.now() - clock.t0) / 1000;
    clock.el.className = 'clock ' + (e > p + 45 ? 'late' : 'ok');
    clock.el.innerHTML = KT.t('<b>{e}</b> · plan {p} / {tot}′', { e: mmss(e), p: mmss(p), tot: Math.round(total(cut())) });
  }
  setInterval(tick, 1000);

  /* ================= focus mode + keys ================= */
  function setFocus(on) { document.body.classList.toggle('focus', !!on && !!$('.stage-frame.is-current')); }
  var keys = $('.keys');
  $('.help').addEventListener('click', function () { keys.classList.toggle('on'); });
  /* the key of a keydown, a letter in lower case. While the keyboard types another script (a Hebrew layout gives
     'ך' for L and 'ג' for D) a letter shortcut goes by the key's position instead, so F, T, M, D and L still work */
  function letter(e) {
    var k = e.key || '';
    if (k.length !== 1) return k;
    return k.charCodeAt(0) > 127 && /^Key[A-Z]$/.test(e.code || '') ? e.code.charAt(3).toLowerCase() : k.toLowerCase();
  }
  /* the language: the same control in the HUD and in the slide master; L switches (the page reloads, see "boot") */
  var hud = $('.hud');
  if (I18N && hud) hud.insertBefore(I18N.switcher('lang'), $('.clock', hud));

  /* ================= subtitles: titles only =================
     A step's subtitles are its text about the picture: paragraphs (p.aside, p.fine, p.big, p.scope, p.credit…), lists,
     the .do / .try lines, an opener's .time; on the two title screens what follows the title block (the cover's
     p.fine, the opening's "scroll down" .meta). What stays: badges and labels (.lvl, .tag, .num, .kick), the
     headline (h1, h2, p.thesis, p.qa), a title screen's .lede and the cover's byline, and every exhibit (.eq, .flip,
     table.cmp, .exhibits, .legend, .props, .labs, .trio, .p-next). Titles only (html.kt-titles; deck.css hides SUB
     there) leaves every step its height, so the reading line, the scroll and the scenes behave as before. talk.html's
     head sets the class before the first paint (?subs=on|off, else localStorage 'kt-subs'); S, the HUD button, the
     slide master and the presenter view switch it (setSubs). The presenter view shows subtitles(step) as its notes. */
  var SUB = ':scope>:is(p:not(.thesis,.qa,.lede),ul,.do,.try,.time)', root = document.documentElement;
  function subtitles(n) {
    var host = n.matches('.p-hero') ? $('.p-text', n) : n, out = host ? $$(SUB, host) : [];
    if (host && n.matches('.p-open')) out = $$(':scope>:is(p:not(.lede),.meta)', host);
    return out;
  }
  var subsBtn = el('button', 'subs', '<svg viewBox="0 0 16 13" aria-hidden="true"><path d="M1.5 2.2h13" stroke-width="2.2"/><path class="sl" d="M1.5 7h10.5M1.5 11h7.5" stroke-width="1.4"/></svg>');
  subsBtn.type = 'button'; subsBtn.title = KT.t('S · titles only: hide / show the subtitles');
  subsBtn.setAttribute('aria-label', KT.t('Titles only: hide the subtitles of the slides (S)'));
  if (hud) hud.insertBefore(subsBtn, $('.help', hud));
  function subs() { return !root.classList.contains('kt-titles'); }
  function paintSubs() { subsBtn.setAttribute('aria-pressed', String(!subs())); }
  function setSubs(on) {
    on = !!on;
    if (on !== subs()) {
      root.classList.toggle('kt-titles', !on);
      settle(); onScroll();
    }
    try { localStorage.setItem('kt-subs', on ? 'on' : 'off'); } catch (e) {}
    // an address that says ?subs= would undo this choice at the next load: it goes, the saved choice holds
    if (/[?&]subs=/.test(location.search)) try {
      history.replaceState(null, '', location.pathname + location.search.replace(/([?&])subs=[^&]*&?/, '$1').replace(/[?&]$/, '') + location.hash);
    } catch (e) {}
    paintSubs();
    emit('kt-subs', { subs: on });
  }
  subsBtn.addEventListener('click', function () { setSubs(!subs()); });
  paintSubs();

  /* ================= info boxes: on / off =================
     The DOM boxes over a simulation stage that carry text, numbers or controls (the kit's caption, readout, panel and
     hint, and each module's own boards, tags, badges, labels, legends, plots) wear the class `kt-box`; with html.kt-bare
     (deck.css, "Info boxes off") they are hidden and only the picture stays. Same plumbing as the subtitles: the class
     is set by talk.html's head before the first paint (?info=on|off, else localStorage 'kt-info'), switched by I, the HUD
     button, the slide master and the presenter view (setInfo), event kt-info {info}. */
  var infoBtn = el('button', 'info', '<svg viewBox="0 0 16 13" aria-hidden="true"><rect class="fr" x="1.2" y="1.2" width="13.6" height="10.6" rx="2.2" stroke-width="1.3"/><rect class="bx" x="3.6" y="3.4" width="5.4" height="3" rx="1" stroke-width="1.3"/><path class="sl" d="M10.4 2.4 L2.6 10.6" stroke-width="1.5"/></svg>');
  infoBtn.type = 'button'; infoBtn.title = KT.t('I · info boxes: hide / show the captions and panels over the simulations');
  infoBtn.setAttribute('aria-label', KT.t('Info boxes off: hide the boxes over the simulations (I)'));
  if (hud) hud.insertBefore(infoBtn, subsBtn.nextSibling);
  function info() { return !root.classList.contains('kt-bare'); }
  function paintInfo() { infoBtn.setAttribute('aria-pressed', String(!info())); }
  function setInfo(on) {
    on = !!on;
    if (on !== info()) root.classList.toggle('kt-bare', !on);     // hidden, not removed: no module lays out again
    try { localStorage.setItem('kt-info', on ? 'on' : 'off'); } catch (e) {}
    if (/[?&]info=/.test(location.search)) try {
      history.replaceState(null, '', location.pathname + location.search.replace(/([?&])info=[^&]*&?/, '$1').replace(/[?&]$/, '') + location.hash);
    } catch (e) {}
    paintInfo();
    emit('kt-info', { info: on });
  }
  infoBtn.addEventListener('click', function () { setInfo(!info()); });
  paintInfo();

  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;       // sliders keep their arrows
    if (tag === 'BUTTON' && (e.key === ' ' || e.key === 'Enter')) return;
    var k = letter(e), fwd = RTL ? 'ArrowLeft' : 'ArrowRight', back = RTL ? 'ArrowRight' : 'ArrowLeft';   // right to left: ← is forward
    if (k === fwd || k === 'ArrowDown' || k === 'PageDown' || k === ' ') { e.preventDefault(); stepBy(e.shiftKey && k === ' ' ? -1 : 1); }
    else if (k === back || k === 'ArrowUp' || k === 'PageUp') { e.preventDefault(); stepBy(-1); }
    else if (k === 'Home') { e.preventDefault(); go(allNav[0]); }
    else if (k === 'End') { e.preventDefault(); var v = visibleNav(); go(v[v.length - 1]); }
    else if (k === 'f') setFocus(!document.body.classList.contains('focus'));
    else if (k === 'Escape') { setFocus(false); keys.classList.remove('on'); }
    else if (k === 't') toggleClock();
    else if (k === 'm') snd.btn.click();
    else if (k === 'l') { if (I18N && !e.repeat) I18N.toggle(); }
    else if (k === 's') { if (!e.repeat) setSubs(!subs()); }
    else if (k === 'i') { if (!e.repeat) setInfo(!info()); }
    else if (k === '?') keys.classList.toggle('on');
    else if (k >= '0' && k <= '9') {                 // 0 = the Knot-a-Surface cover, 1-9 = the chapter with that rail number; a key with none goes to the last part
      var c = k === '0' ? ($('section.cover[data-rail]') || chapters[0]) : chapters.filter(function (x) { return +x.dataset.rail === +k; })[0] || chapters[chapters.length - 1];
      go(firstNav(c));
    }
  });

  /* ================= exhibits ================= */
  function svg(tag, at, parent, text) {
    var e = document.createElementNS(NS, tag), k;
    for (k in at) e.setAttribute(k, at[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  /* the charts' ink, in the page's theme (KnotTalk.theme): on the light talk the kit's light colour code on a glass card */
  var C = KT.color, INK = KT.theme === 'light'
    ? { text: C.text, dim: C.dim, mute: C.mute, line: C.line, bg: C.bg, ours: C.surf, hot: '#3c2d1e' }
    : { text: '#dde5f0', dim: '#8fa4c0', mute: '#5c6f8a', line: '#1b2a42', bg: '#0b1017', ours: '#4fc3f7', hot: '#ffffff' };
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
      '<div><b>' + mo.toFixed(3) + '</b>' + KT.t('ours · mean mm') + '</div><div><b>' + mp.toFixed(3) + '</b>' + KT.t('PGSR · mean mm') + '</div>' +
      '<div><b>' + close + ' / ' + n + '</b>' + KT.t('scans within 0.05 mm') + '</div><div><b>' + wins + ' / ' + n + '</b>' + KT.t('scans where ours is lower') + '</div>';
    // the means above are computed from the rows; say so while a printed mean differs from its row
    var warn = $('.check', box), bad = [];
    if (Math.abs(mo - DTU.printedMean.ours) > 0.0015) bad.push(KT.t('ours {v}', { v: DTU.printedMean.ours }));
    if (Math.abs(mp - DTU.printedMean.pgsr) > 0.006) bad.push(KT.t('PGSR {v}', { v: DTU.printedMean.pgsr }));
    warn.style.display = bad.length ? '' : 'none';
    warn.textContent = KT.t('Means are computed from the 15 scans shown (the table prints {list}).', { list: bad.join(', ') });

    // the key's swatches (written in talk.html in the dark ink) take the chart's ink
    var key = $$('.chart-key i', box);
    if (key[0]) key[0].style.background = INK.ours;
    if (key[1]) key[1].style.borderColor = INK.dim;
    var W = 660, H = 290, L = 40, Rm = 10, Tm = 12, B = 34, y0 = 0.2, y1 = 1.0;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': KT.t('Chamfer distance per DTU scan, ours versus PGSR') }, $('.plot', box));
    function X(k) { return L + (k + 0.5) * (W - L - Rm) / n; }
    function Y(v) { return Tm + (y1 - v) / (y1 - y0) * (H - Tm - B); }
    [0.2, 0.4, 0.6, 0.8, 1.0].forEach(function (v) {
      svg('path', { d: 'M' + L + ' ' + Y(v) + 'H' + (W - Rm), stroke: INK.line, 'stroke-width': 1 }, s);
      svg('text', { x: L - 8, y: Y(v) + 3.5, 'text-anchor': 'end', fill: INK.mute, style: MONO }, s, v.toFixed(1));
    });
    svg('text', { x: L, y: H - 4, fill: INK.mute, style: MONO }, s, KT.t('DTU scan'));
    var tip = tipFor(box), hot = svg('rect', { y: Tm, height: H - Tm - B, fill: INK.hot, opacity: 0, rx: 4 }, s);
    for (i = 0; i < n; i++) {
      var x = X(i), a = DTU.ours[i], b = DTU.pgsr[i];
      svg('path', { d: 'M' + x + ' ' + Y(a) + 'V' + Y(b), stroke: INK.mute, 'stroke-width': 2, 'stroke-linecap': 'round' }, s);
      svg('circle', { cx: x, cy: Y(b), r: 5, fill: INK.bg, stroke: INK.dim, 'stroke-width': 2 }, s);
      svg('circle', { cx: x, cy: Y(a), r: 6, fill: INK.ours, stroke: INK.bg, 'stroke-width': 2 }, s);
      svg('text', { x: x, y: H - B + 16, 'text-anchor': 'middle', fill: INK.dim, style: MONO }, s, DTU.scans[i]);
    }
    var bw = (W - L - Rm) / n;
    function point(e) {
      var r = s.getBoundingClientRect(), k = Math.floor(((e.clientX - r.left) / r.width * W - L) / bw);
      if (k < 0 || k >= n) { tip.hide(); hot.setAttribute('opacity', 0); return; }
      hot.setAttribute('x', L + k * bw); hot.setAttribute('width', bw); hot.setAttribute('opacity', 0.05);
      var d = DTU.ours[k] - DTU.pgsr[k], pr = box.getBoundingClientRect();
      tip.show(KT.t('scan <b>{scan}</b><br>ours <b>{ours}</b> · PGSR <b>{pgsr}</b><br>Δ {d} mm',
          { scan: DTU.scans[k], ours: DTU.ours[k].toFixed(3), pgsr: DTU.pgsr[k].toFixed(2), d: (d > 0 ? '+' : '−') + Math.abs(d).toFixed(3) }),
        r.left - pr.left + X(k) / W * r.width, r.top - pr.top + Y(Math.max(DTU.ours[k], DTU.pgsr[k])) / H * r.height);
    }
    // hover with a mouse; on touch a tap shows a scan and the tip stays until the next tap
    s.addEventListener('pointermove', point); s.addEventListener('pointerdown', point);
    s.addEventListener('pointerleave', function (e) { if (e.pointerType === 'touch') return; tip.hide(); hot.setAttribute('opacity', 0); });
  }

  function benchChart(box) {
    var W = 560, H = 290, L = 46, Rm = 16, Tm = 12, B = 38;
    var s = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': KT.t('Optimised parameters versus Chamfer distance on the proxy bench') }, $('.plot', box));
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
    svg('text', { x: W - Rm, y: H - 5, 'text-anchor': 'end', fill: INK.mute, style: MONO }, s, KT.t('optimised parameters · log →'));
    svg('text', { x: L, y: H - 5, fill: INK.mute, style: MONO }, s, KT.t('↓ Chamfer mm · log'));
    var tip = tipFor(box);
    s.addEventListener('pointerdown', function (e) { if (e.target.getAttribute('opacity') !== '0') tip.hide(); });  // a tap off the points closes the tip
    BENCH.forEach(function (m) {
      var x = X(m.params), y = Y(m.chamfer);
      svg('circle', m.ours ? { cx: x, cy: y, r: 7, fill: INK.ours, stroke: INK.bg, 'stroke-width': 2 }
                           : { cx: x, cy: y, r: 5, fill: INK.bg, stroke: INK.dim, 'stroke-width': 2 }, s);
      svg('text', { x: x + m.lx, y: y + m.ly, 'text-anchor': m.anchor || 'start', fill: m.ours ? INK.text : INK.dim,
        style: MONO + (m.ours ? ';font-weight:700;font-size:11.5px' : '') }, s, m.name);
      var h = svg('circle', { cx: x, cy: y, r: 16, fill: '#fff', opacity: 0 }, s);
      h.addEventListener('pointerenter', function () {
        var r = s.getBoundingClientRect(), pr = box.getBoundingClientRect();
        tip.show(KT.t('<b>{name}</b><br>{prim} primitives · <b>{params}</b> params<br>Chamfer <b>{chamfer} mm</b> · closed {closed}%<br>converged at step {step}', {
            name: m.name, prim: m.prim.toLocaleString('en-US'),
            params: m.params >= 1e6 ? (m.params / 1e6).toFixed(2) + ' M' : (m.params / 1e3).toFixed(1) + ' k',
            chamfer: m.chamfer.toFixed(1), closed: m.closed, step: m.step === 'no loop' ? KT.t('no loop') : m.step }),
          r.left - pr.left + x / W * r.width, r.top - pr.top + y / H * r.height);
      });
      h.addEventListener('pointerleave', function (e) { if (e.pointerType !== 'touch') tip.hide(); });
    });
  }
  var ex = $('#ex-dtu'); if (ex) dtuChart(ex);
  ex = $('#ex-bench'); if (ex) benchChart(ex);

  /* ================= the handle: talk/master.js (the slide master) and tests =================
     events on document: kt-step {step} (a new active step), kt-cut {cut}, kt-clock {t0}, kt-parts {over} (after refresh),
     kt-subs {subs} (subtitles shown, or titles only), kt-info {info} (boxes over the simulations shown, or off) */
  KT.deck = {
    presenter: presenter,                                              // this page is the presenter view (talk.html?master)
    steps: allNav, sections: chapters, navs: navsOf,                   // every step (.nav) and every data-rail section, in order
    key: function (n) { return n._key; }, title: function (n) { return n._title; },   // the key is English in every language, the title is the page's
    letter: letter,                                                    // a keydown's key, by position on a keyboard that types another script
    section: function (n) { return n._sec; }, byKey: function (k) { return byKey[k] || null; },
    current: function () { return cur; }, cut: cut, setCut: setCut,
    info: info, setInfo: setInfo,                                      // info(): true = boxes over the simulations shown; setInfo(on)
    subs: subs, setSubs: setSubs, subtitles: subtitles,              // subs(): true = subtitles shown; subtitles(step): its subtitle elements
    defaultOn: defaultOn, isOn: isOn, inCut: inCut, shown: shown,      // shown(n, m): on, and in cut m (default: the current one)
    minutes: minutes, share: share, budget: budget, total: total, plan: plan,
    clock: function () { return clock.t0; }, toggleClock: toggleClock,
    go: go, step: stepBy, activate: activate,
    set: set,                                                          // set(step, on): one override; refresh() after a batch
    overrides: function () { return Object.assign({}, over); },
    setOverrides: function (o) {                                       // replace them all (unknown keys are dropped)
      over = {};
      for (var k in o || {}) if (byKey[k] && typeof o[k] === 'boolean') set(byKey[k], o[k]);
      refresh();
    },
    refresh: refresh,
    // design "knob" (talk/atlas.js): sample({ has, cost }) or sample(null); repaints like a change of cut (event kt-cut)
    sample: function (s) { smp = s || null; layout(); settle(); onScroll(); tick(); emit('kt-cut', { cut: cut() }); }
  };

  /* ================= boot =================
     A change of language reloads the page (talk/i18n.js). What the presenter had is handed over that reload and is
     back here: the cut, the parts that are on, the running clock, the mute, the step (the saved choices and the
     address hold most of it anyway; this also works where the browser gives the page no storage). */
  document.addEventListener('kt-lang', function (e) {
    e.detail.keep.deck = { cut: cut(), over: Object.assign({}, over), t0: clock.t0, mute: snd.mute, step: cur ? cur._i : -1, subs: subs(), info: info() };
  });
  var kept = (I18N && I18N.kept && I18N.kept.deck) || {};
  var saved = 40;
  try { saved = +localStorage.getItem('kt-cut') === 20 ? 20 : 40; } catch (e) {}
  if (kept.cut === 20 || kept.cut === 40) saved = kept.cut;
  document.body.classList.toggle('cut-20', saved === 20);
  cutBtns.forEach(function (b) { b.classList.toggle('on', +b.dataset.cut === saved); });
  // the slide master's saved choices (talk/master.js writes them, step key -> on): read here, before the first frame
  (function () {
    var o = null, k;
    try { o = JSON.parse(localStorage.getItem('kt-parts') || '{}'); } catch (e) { /* no storage: the page's own defaults */ }
    if (kept.over && typeof kept.over === 'object') o = kept.over;
    for (k in o || {}) if (byKey[k] && typeof o[k] === 'boolean') set(byKey[k], o[k]);
  })();
  if (kept.t0 > 0) clock.t0 = +kept.t0;
  if (kept.mute) snd.mute = true;
  layout();
  var m = /^#s(\d+)$/.exec(location.hash), start = kept.step >= 0 ? +kept.step : m ? +m[1] : -1;
  if (allNav[start]) {
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    // a step that is off by now gives way to its predecessor
    requestAnimationFrame(function () { var n = allNav[start]; go(n.offsetParent === null ? nearest(n, -1) : n, true); onScroll(); });
  }
  onScroll(); tick();
})();
