/* Knot-a-Surface talk — the slide master: switch parts of the talk on and off while it runs.
   D, or the list button in the HUD, opens it as a drawer over the deck; D or Esc closes it. talk.html?master (or
   #master) is the presenter view: the same list as a page of its own, for a second window of the same browser. It
   follows the talk window (the active step, the cut, the clock) and drives it: previous / next (its buttons, or
   ← → PageUp PageDown space), a click on a title, the cut, T for the clock.
   Every section is listed with its steps, each with a switch, its minutes in both cuts (the section's data-min40 /
   data-min20, shared evenly by its steps) and the totals of what is on. What a switch does is deck.js's ("parts",
   KnotTalk.deck): a step that is off leaves the deck as a deep step does in the short cut. Parts that talk.html marks
   data-off start off and carry an "off by default" tag. Saved per browser (localStorage 'kt-parts', every access
   guarded): only the presenter's overrides of the page's defaults, so "reset" returns to the page's own choice and a
   later change of a default still reaches every part the presenter never touched; "show all" turns everything on.
   Between windows: a BroadcastChannel where there is one, postMessage between a window and the one it opened, and
   the storage event for the overrides. Every message has an id, so one that arrives twice is applied once. With none
   of them (some file:// setups) the drawer in the talk window still works on its own. */
(function () {
  'use strict';
  var KT = window.KnotTalk || {}, D = KT.deck;
  if (!D) return;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function mmss(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  function mins(m) { return m > 0 ? mmss(m * 60) : '–'; }
  function own(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }

  var STORE = 'kt-parts', TAG = 'knot-talk-master';
  var presenter = /(?:^|&)master(?:[=&]|$)/.test(location.search.slice(1)) || location.hash === '#master';

  /* ---------- saved choices: only the overrides of the page's defaults ---------- */
  function parse(v) { try { var o = JSON.parse(v || '{}'); return o && typeof o === 'object' && !Array.isArray(o) ? o : {}; } catch (e) { return {}; } }
  var wipe = false;                                 // "reset": forget every saved override, also of steps this copy of the page lacks
  function save() {
    try {
      var ls = window.localStorage, o = wipe ? {} : parse(ls.getItem(STORE)), now = D.overrides(), k;
      // merge: a key this page does not know (a window with an older or newer talk.html) is left as it is
      for (k in o) if (own(o, k) && D.byKey(k) && !own(now, k)) delete o[k];
      for (k in now) if (own(now, k)) o[k] = now[k];
      if (Object.keys(o).length) ls.setItem(STORE, JSON.stringify(o)); else ls.removeItem(STORE);
    } catch (e) { /* no storage here (a private window, some file:// setups): the choice lasts as long as the page */ }
    wipe = false;
  }
  function same(a, b) {
    var k;
    for (k in a) if (own(a, k) && a[k] !== b[k]) return false;
    for (k in b) if (own(b, k) && !own(a, k)) return false;
    return true;
  }

  /* ---------- the bus between windows ---------- */
  var bus = (function () {
    var me = Math.random().toString(36).slice(2, 10), count = 0, seen = [], fns = [], peers = [], ch = null;
    function take(d) {
      if (!d || d.kt !== TAG || d.from === me || seen.indexOf(d.id) >= 0) return;
      seen.push(d.id); if (seen.length > 120) seen.shift();
      fns.forEach(function (f) { f(d); });
    }
    try { ch = new BroadcastChannel(TAG); ch.onmessage = function (e) { take(e.data); }; } catch (e) { ch = null; }
    window.addEventListener('message', function (e) {        // only from the window that opened this one, or one it opened
      var s = e.source;
      if (!s || s === window || !e.data || e.data.kt !== TAG) return;
      if (s !== window.opener && peers.indexOf(s) < 0) return;
      if (peers.indexOf(s) < 0) peers.push(s);
      take(e.data);
    });
    return {
      bc: !!ch,
      on: function (f) { fns.push(f); },
      peer: function (w) { if (w && w !== window && peers.indexOf(w) < 0) peers.push(w); },
      post: function (msg) {
        msg.kt = TAG; msg.from = me; msg.id = me + '.' + (++count);
        if (ch) try { ch.postMessage(msg); } catch (e) {}
        peers = peers.filter(function (w) { try { return !w.closed; } catch (e) { return false; } });
        peers.forEach(function (w) { try { w.postMessage(msg, '*'); } catch (e) {} });
      }
    };
  })();

  /* ---------- what the panel shows: this deck, or (presenter view) what the talk window last said ---------- */
  var view = { cut: D.cut(), cur: null, t0: 0, plan: 0, linked: false };
  function cutNow() { return presenter ? view.cut : D.cut(); }
  function curNow() { return presenter ? (view.cur && D.byKey(view.cur)) : D.current(); }
  function label(n) { var s = D.section(n); return (s.dataset.rail || '') + ' · ' + D.title(n); }
  function nextAfter(n, m) {
    for (var i = n ? D.steps.indexOf(n) + 1 : 0; i < D.steps.length; i++) if (D.shown(D.steps[i], m)) return D.steps[i];
    return null;
  }

  /* ---------- the panel ---------- */
  var panel = el('div', 'ktm' + (presenter ? ' ktm-page' : ''));
  panel.innerHTML =
    '<header class="ktm-head">' +
      '<div class="ktm-top"><div><div class="ktm-kick">Knot‑a‑Surface · ' + (presenter ? 'presenter view' : 'the talk') + '</div>' +
        '<h2 class="ktm-h" id="ktm-h" tabindex="-1">Slide master</h2></div>' +
        '<div class="ktm-tools">' + (presenter ? '' :
          '<button type="button" class="ktm-pop" title="The same list in a second window, synced with this one">Presenter view ↗</button>' +
          '<button type="button" class="ktm-x" aria-label="Close the slide master (Esc)" title="Close · D / esc">×</button>') + '</div></div>' +
      '<p class="ktm-note">Switch parts of the talk on and off. A part that is off leaves the deck: the arrows skip it, and ' +
        'the clock and the totals leave it out. ' + (presenter ? 'This window follows the talk window and drives it.' : '<kbd>D</kbd> or <kbd>esc</kbd> closes.') + '</p>' +
      '<div class="ktm-cuts" role="group" aria-label="Length of what is on, per cut; pick the cut in use">' +
        '<button type="button" data-cut="40"><span>long cut</span><b></b></button>' +
        '<button type="button" data-cut="20"><span>short cut</span><b></b></button></div>' +
      '<div class="ktm-acts"><button type="button" class="ktm-all">Show all</button>' +
        '<button type="button" class="ktm-reset">Reset to defaults<i></i></button></div>' +
    '</header>' +
    (presenter ?
      '<section class="ktm-live" aria-label="The talk window">' +
        '<div class="ktm-nav"><button type="button" class="ktm-prev" aria-label="Previous step">← prev</button>' +
          '<button type="button" class="ktm-next" aria-label="Next step">next →</button></div>' +
        '<div class="ktm-where"><div><span>now</span><b class="ktm-cur">–</b></div><div><span>next</span><b class="ktm-nxt">–</b></div></div>' +
        '<div class="ktm-clock"><button type="button" class="ktm-clk">start clock</button><span class="ktm-time"></span></div>' +
        '<p class="ktm-link"><span class="ktm-st" role="status"></span> <button type="button" class="ktm-open">Open the talk ↗</button></p>' +
      '</section>' : '') +
    '<div class="ktm-body"><div class="ktm-cols" aria-hidden="true"><span>part</span><span>long</span><span>short</span></div>' +
      '<ol class="ktm-list"></ol></div>';

  var list = $('.ktm-list', panel), secRows = [], stepRows = [], groups = [], byStep = new Map();
  function tag(t, cls, tip) { var s = el('span', 'ktm-tag' + (cls ? ' ' + cls : ''), t); if (tip) s.title = tip; return s; }
  function row(cls, lab, name, aria) {
    var r = el('div', 'ktm-row ' + cls);
    var tog = el('input', 'ktm-tog'); tog.type = 'checkbox'; tog.dataset.col = 'tog'; tog.setAttribute('aria-label', aria);
    var go = el('button', 'ktm-go'); go.type = 'button'; go.dataset.col = 'go'; go.textContent = name;
    var nm = el('span', 'ktm-name'); nm.appendChild(go);
    var m40 = el('span', 'ktm-m'), m20 = el('span', 'ktm-m');
    r.appendChild(tog); r.appendChild(el('span', 'ktm-rail', lab)); r.appendChild(nm); r.appendChild(m40); r.appendChild(m20);
    return { row: r, tog: tog, go: go, name: nm, m40: m40, m20: m20 };
  }
  var grp = -1;
  D.sections.forEach(function (sec) {
    var g = sec.closest('.prelude') ? 0 : 1;
    if (g !== grp) {                                       // the light opening and the cover, then the talk
      grp = g;
      var gl = el('li', 'ktm-grp', '<span>' + (g ? 'The talk' : 'Opening and cover · light') + '</span>');
      var G = { el: gl, g: g, secs: [], m40: el('span', 'ktm-m'), m20: el('span', 'ktm-m') };
      gl.appendChild(G.m40); gl.appendChild(G.m20); list.appendChild(gl); groups.push(G);
    }
    groups[groups.length - 1].secs.push(sec);
    var navs = D.navs(sec), single = navs.length === 1 && navs[0] === sec, title = sec.dataset.title || D.title(navs[0]);
    var li = el('li', 'ktm-sec'), R = row('ktm-srow' + (single ? ' ktm-one' : ''), sec.dataset.rail || '', title,
      (sec.dataset.rail || '') + ' · ' + title + (single ? '' : ', all its steps'));
    R.sec = sec; R.navs = navs; li.appendChild(R.row);
    if (sec.hasAttribute('data-off')) R.name.appendChild(tag('off by default', 'def', 'talk.html marks this part data-off: it stays off until you switch it on'));
    secRows.push(R);
    if (single) { R.name.appendChild(tag('now', 'now')); byStep.set(sec, R); }
    else {
      var ol = el('ol', 'ktm-steps');
      navs.forEach(function (n, k) {
        var S = row('ktm-step', String(k + 1), D.title(n), D.title(n));
        S.n = n; S.sec = sec;
        if (n.matches('.step[data-depth="deep"]')) S.name.appendChild(tag('deep', '', 'A deep-dive: the short cut leaves it out'));
        if (n.hasAttribute('data-off')) S.name.appendChild(tag('off by default', 'def', 'talk.html marks this step data-off: it stays off until you switch it on'));
        S.name.appendChild(tag('now', 'now'));
        var it = el('li'); it.appendChild(S.row); ol.appendChild(it);
        stepRows.push(S); byStep.set(n, S);
      });
      li.appendChild(ol);
    }
    list.appendChild(li);
  });

  var cutBtns = $$('.ktm-cuts button', panel), allBtn = $('.ktm-all', panel), resetBtn = $('.ktm-reset', panel);
  var live = presenter ? {
    cur: $('.ktm-cur', panel), nxt: $('.ktm-nxt', panel), clk: $('.ktm-clk', panel), time: $('.ktm-time', panel),
    link: $('.ktm-link', panel), st: $('.ktm-st', panel), open: $('.ktm-open', panel), was: null
  } : null;

  function setNum(span, v, full, cutName) {
    span.textContent = mins(v);
    span.classList.toggle('is-cut', v < full - 1e-9);
    span.title = cutName + ': ' + mins(v) + (v < full - 1e-9 ? ' of ' + mins(full) : '');
  }
  var lastCur = null;
  function paint() {
    var m = cutNow(), cur = curNow();
    cutBtns.forEach(function (b) {
      var c = +b.dataset.cut; $('b', b).textContent = mmss(D.total(c) * 60);
      b.setAttribute('aria-pressed', String(c === m));
    });
    var nOver = Object.keys(D.overrides()).length;
    resetBtn.disabled = !nOver; $('i', resetBtn).textContent = nOver ? String(nOver) : '';
    resetBtn.title = nOver ? nOver + ' step' + (nOver > 1 ? 's differ' : ' differs') + ' from the page’s defaults' : 'Everything is as the page sets it';
    allBtn.disabled = D.steps.every(D.isOn);
    groups.forEach(function (G) {
      var a = 0, b = 0;
      G.secs.forEach(function (s) { a += D.budget(s, 40); b += D.budget(s, 20); });
      G.m40.textContent = mins(a); G.m20.textContent = mins(b);
    });
    secRows.forEach(function (R) {
      var on = R.navs.filter(D.isOn).length, all = on === R.navs.length;
      var shown = R.navs.some(function (n) { return D.shown(n, m); });
      R.tog.checked = all; R.tog.indeterminate = on > 0 && !all;
      R.row.classList.toggle('is-off', !on);
      R.row.classList.toggle('is-out', on > 0 && !shown);
      R.row.classList.toggle('is-now', !!cur && (cur === R.sec || R.sec.contains(cur)));
      R.go.setAttribute('aria-disabled', String(!shown));
      R.go.title = shown ? 'Go to this part' : 'Off in this cut: switch it on to go there';
      setNum(R.m40, D.budget(R.sec, 40), D.minutes(R.sec, 40), 'long cut');
      setNum(R.m20, D.budget(R.sec, 20), D.minutes(R.sec, 20), 'short cut');
    });
    stepRows.forEach(function (S) {
      var on = D.isOn(S.n), shown = on && D.inCut(S.n, m);
      S.tog.checked = on;
      S.row.classList.toggle('is-off', !on);
      S.row.classList.toggle('is-out', on && !shown);
      S.row.classList.toggle('is-now', S.n === cur);
      S.go.setAttribute('aria-disabled', String(!shown));
      S.go.title = shown ? 'Go to this step' : on ? 'A deep-dive: the short cut leaves it out' : 'Off: switch it on to go there';
      var a = on ? D.share(S.n, 40) : 0, b = on && D.inCut(S.n, 20) ? D.share(S.n, 20) : 0;
      S.m40.textContent = mins(a); S.m20.textContent = mins(b);
    });
    if (live) {
      live.cur.textContent = cur ? label(cur) : '–';
      var nx = cur ? nextAfter(cur, m) : null;
      live.nxt.textContent = nx ? label(nx) : cur ? 'the end' : '–';
      if (live.was !== view.linked) {
        live.was = view.linked;
        live.link.classList.toggle('on', view.linked); live.open.hidden = view.linked;
        live.st.textContent = view.linked ? 'Linked to the talk window.' :
          'Waiting for the talk window: open talk.html in this browser and this page follows it.' +
          (bus.bc ? '' : ' This browser gives the page no BroadcastChannel: open this view from the talk’s slide master ' +
            '(D, then “Presenter view”), or serve the folder over http.');
      }
      paintClock();
    }
    if (cur !== lastCur) { lastCur = cur; follow(); }
  }
  function paintClock() {
    if (!live) return;
    var t0 = view.t0, p = view.plan, tot = mmss(D.total(cutNow()) * 60);
    live.clk.textContent = t0 ? 'reset clock' : 'start clock';
    if (!t0) { live.time.className = 'ktm-time'; live.time.innerHTML = 'plan <b>' + mmss(p) + '</b> / ' + tot; return; }
    var e = (Date.now() - t0) / 1000;
    live.time.className = 'ktm-time ' + (e > p + 45 ? 'late' : 'ok');
    live.time.innerHTML = '<b>' + mmss(e) + '</b> · plan ' + mmss(p) + ' / ' + tot;
  }
  if (presenter) setInterval(paintClock, 1000);
  /* keep the active step's row in sight (the drawer only while it is open) */
  function follow() {
    var R = byStep.get(curNow()); if (!R || (!presenter && !isOpen)) return;
    var body = $('.ktm-body', panel), r = R.row.getBoundingClientRect(), b = body.getBoundingClientRect(), pad = 48;
    if (r.top < b.top + pad || r.bottom > b.bottom - 8) body.scrollTop += r.top - b.top - b.height * 0.35;
  }

  /* ---------- acting: here, and (presenter view) in the talk window ---------- */
  function commit(steps, on) {
    steps.forEach(function (n) { D.set(n, on); });
    D.refresh();
    if (presenter) bus.post({ type: 'set', keys: steps.map(D.key), on: on });
  }
  function goTo(n) {
    if (!n) return;
    if (presenter) bus.post({ type: 'go', key: D.key(n) });
    else { close(); D.go(n); }
  }
  secRows.forEach(function (R) {
    R.tog.addEventListener('change', function () { commit(R.navs, R.tog.checked); });   // mixed or off: all on; all on: all off
    R.go.addEventListener('click', function () {
      if (R.go.getAttribute('aria-disabled') === 'true') return;
      var m = cutNow(); goTo(R.navs.filter(function (n) { return D.shown(n, m); })[0]);
    });
  });
  stepRows.forEach(function (S) {
    S.tog.addEventListener('change', function () { commit([S.n], S.tog.checked); });
    S.go.addEventListener('click', function () { if (S.go.getAttribute('aria-disabled') !== 'true') goTo(S.n); });
  });
  allBtn.addEventListener('click', function () {
    D.steps.forEach(function (n) { D.set(n, true); }); D.refresh();
    if (presenter) bus.post({ type: 'all' });
  });
  resetBtn.addEventListener('click', function () {
    wipe = true; D.setOverrides({});
    if (presenter) bus.post({ type: 'reset' });
  });
  cutBtns.forEach(function (b) {
    b.addEventListener('click', function () {
      var m = +b.dataset.cut;
      if (presenter) { view.cut = m; paint(); bus.post({ type: 'cut', m: m }); } else D.setCut(m);
    });
  });
  /* arrows move between the rows (switches with switches, titles with titles); ← → between the two (drawer only) */
  function rove(e, sideways) {
    var t = e.target, col = t && t.dataset && t.dataset.col, k = e.key;
    if (!col || !panel.contains(t)) return false;
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      if (!sideways) return false;
      var o = $('[data-col="' + (col === 'tog' ? 'go' : 'tog') + '"]', t.closest('.ktm-row')); if (o) o.focus();
      return true;
    }
    var all = $$('[data-col="' + col + '"]', list), i = all.indexOf(t), j;
    if (k === 'ArrowDown') j = i + 1; else if (k === 'ArrowUp') j = i - 1;
    else if (k === 'Home') j = 0; else if (k === 'End') j = all.length - 1; else return false;
    j = Math.max(0, Math.min(all.length - 1, j));
    all[j].focus(); all[j].scrollIntoView({ block: 'nearest' });
    return true;
  }

  /* ---------- the drawer in the talk window ---------- */
  var dlg = null, isOpen = false, back = null, hb = null;
  function open(byPointer) {
    if (!dlg || isOpen) return;
    back = byPointer ? null : document.activeElement;
    isOpen = true; paint();
    try { dlg.showModal(); } catch (e) { dlg.setAttribute('open', ''); }
    document.body.classList.add('ktm-on'); hb.setAttribute('aria-expanded', 'true');
    var R = byStep.get(D.current());
    follow();
    (R ? R.tog : $('.ktm-tog', list)).focus({ preventScroll: true });
  }
  function close() {
    if (!dlg || !isOpen) return;
    isOpen = false;
    try { dlg.close(); } catch (e) { dlg.removeAttribute('open'); }
    document.body.classList.remove('ktm-on'); hb.setAttribute('aria-expanded', 'false');
    // back to where the keys were; after a click, to nothing, so that space keeps stepping the deck
    if (back && back !== document.body && document.contains(back)) { try { back.focus({ preventScroll: true }); } catch (e) {} }
    else if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    back = null;
  }
  function openPresenter() {
    var url = location.href.split('#')[0].split('?')[0] + '?master', w = null;
    try { w = window.open(url, 'knot-talk-master', 'popup,width=600,height=900'); } catch (e) {}
    if (w) { bus.peer(w); try { w.focus(); } catch (e) {} }
  }

  if (!presenter) {
    dlg = el('dialog', 'ktm-dialog'); dlg.setAttribute('aria-labelledby', 'ktm-h');
    dlg.appendChild(panel); document.body.appendChild(dlg);
    hb = el('button', 'master', '<svg viewBox="0 0 16 13" aria-hidden="true"><path d="M1.5 2h2.5M1.5 6.5h2.5M1.5 11h2.5M6.5 2h8M6.5 6.5h8M6.5 11h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>');
    hb.type = 'button'; hb.title = 'D · slide master: show / hide parts';
    hb.setAttribute('aria-label', 'Slide master: show or hide parts of the talk (D)');
    hb.setAttribute('aria-haspopup', 'dialog'); hb.setAttribute('aria-expanded', 'false');
    var hud = $('.hud'), help = hud && $('.help', hud);
    if (hud) hud.insertBefore(hb, help || null);
    hb.addEventListener('click', function (e) { if (isOpen) close(); else open(e.detail > 0); });
    $('.ktm-x', panel).addEventListener('click', close);
    $('.ktm-pop', panel).addEventListener('click', openPresenter);
    dlg.addEventListener('cancel', function (e) { e.preventDefault(); close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) close(); });        // the dimmed deck around it
    dlg.addEventListener('keydown', function (e) {
      var k = e.key;
      if (!(e.metaKey || e.ctrlKey || e.altKey)) {
        if (k === 'Escape' || ((k === 'd' || k === 'D') && !e.repeat)) { e.preventDefault(); close(); }
        else if (rove(e, true)) e.preventDefault();
      }
      e.stopPropagation();                                  // the deck behind does not see keys while the drawer is open
    });
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat || (e.key !== 'd' && e.key !== 'D')) return;
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      e.preventDefault(); if (isOpen) close(); else open(false);
    });
  } else {
    /* ---------- the presenter view: this page is the panel ---------- */
    document.body.classList.add('kt-presenter');
    document.body.appendChild(panel);
    document.title = 'Slide master · Knot‑a‑Surface';
    $('.ktm-prev', panel).addEventListener('click', function () { bus.post({ type: 'step', d: -1 }); });
    $('.ktm-next', panel).addEventListener('click', function () { bus.post({ type: 'step', d: 1 }); });
    live.clk.addEventListener('click', function () { bus.post({ type: 'clock' }); });
    live.open.addEventListener('click', function () {
      var w = null;
      try { w = window.open(location.href.split('#')[0].split('?')[0], 'knot-talk'); } catch (x) {}
      if (w) bus.peer(w);
    });
    // a clicker or the keyboard drives the talk window; this page's own (hidden) deck never sees a key
    window.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target, k = e.key, ctl = t && (t.tagName === 'BUTTON' || t.tagName === 'INPUT');
      var fwd = k === 'ArrowRight' || k === 'PageDown' || (k === ' ' && !ctl && !e.shiftKey);
      var bwd = k === 'ArrowLeft' || k === 'PageUp' || (k === ' ' && !ctl && e.shiftKey);
      if (fwd || bwd) { e.preventDefault(); bus.post({ type: 'step', d: fwd ? 1 : -1 }); }
      else if ((k === 't' || k === 'T') && !e.repeat) bus.post({ type: 'clock' });
      else if (rove(e, false)) e.preventDefault();
      e.stopPropagation();
    }, true);
    if (window.opener) bus.peer(window.opener);
    var hello = function () { if (!view.linked) bus.post({ type: 'hello' }); };
    setInterval(hello, 2000);
  }

  /* ---------- keeping windows in step ---------- */
  function sendState() {
    if (presenter) return;
    var c = D.current();
    bus.post({ type: 'state', over: D.overrides(), cut: D.cut(), cur: c ? D.key(c) : null, t0: D.clock(), plan: D.plan() });
  }
  bus.on(function (d) {
    if (presenter) {
      if (d.type === 'state') {
        view.linked = true; view.cut = d.cut === 20 ? 20 : 40; view.cur = d.cur || null; view.t0 = +d.t0 || 0; view.plan = +d.plan || 0;
        var o = d.over && typeof d.over === 'object' ? d.over : {};
        if (!same(o, D.overrides())) D.setOverrides(o); else paint();          // setOverrides repaints through kt-parts
      } else if (d.type === 'bye') { view.linked = false; paint(); }
      return;
    }
    var n;
    if (d.type === 'hello') sendState();
    else if (d.type === 'set') {
      (d.keys || []).forEach(function (k) { if ((n = D.byKey(k))) D.set(n, !!d.on); });
      D.refresh();
    } else if (d.type === 'all') { D.steps.forEach(function (x) { D.set(x, true); }); D.refresh(); }
    else if (d.type === 'reset') D.setOverrides({});
    else if (d.type === 'go') { if ((n = D.byKey(d.key))) D.go(n); }
    else if (d.type === 'step') D.step(d.d < 0 ? -1 : 1);
    else if (d.type === 'cut') D.setCut(d.m === 20 ? 20 : 40);
    else if (d.type === 'clock') D.toggleClock();
  });
  /* another window of this browser saved a change: apply what changed (twice is harmless) */
  window.addEventListener('storage', function (e) {
    if (e.key !== STORE && e.key !== null) return;
    if (e.key === null) { D.setOverrides({}); return; }                       // storage cleared
    var a = parse(e.oldValue), b = parse(e.newValue), k, n, changed = false, keys = Object.assign({}, a, b);
    for (k in keys) if (a[k] !== b[k] && (n = D.byKey(k))) { D.set(n, own(b, k) ? !!b[k] : D.defaultOn(n)); changed = true; }
    if (changed) D.refresh();
  });
  ['kt-parts', 'kt-cut', 'kt-clock', 'kt-step'].forEach(function (t) {
    document.addEventListener(t, function () {
      if (t === 'kt-parts') save();
      paint(); sendState();
    });
  });
  if (!presenter) window.addEventListener('pagehide', function () { bus.post({ type: 'bye' }); });

  /* ---------- boot: deck.js has applied the saved choices already (before its first frame); show them, say hello ---------- */
  paint(); sendState();
  if (presenter) bus.post({ type: 'hello' });

  KT.master = { open: open, close: close, isOpen: function () { return isOpen; }, paint: paint, presenter: presenter,
                bus: bus, view: view };
})();
