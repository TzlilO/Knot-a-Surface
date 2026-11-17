/* Knot-a-Surface talk — the slide master: switch parts of the talk on and off while it runs.
   D, or the list button in the HUD, opens it as a drawer over the deck; D or Esc closes it. talk.html?master (or
   #master) is the presenter view: the same list as a page of its own, for a second window of the same browser. It
   follows the talk window (the active step, the cut, the clock) and drives it: previous / next (its buttons, or
   ← → PageUp PageDown space), a click on a title, the cut, T for the clock.
   Every section is listed with its steps (a section of one step is one row), each with a switch, its minutes in both
   cuts (the section's data-min40 / data-min20, shared evenly by its steps; struck through, what a part that is off
   would take) and the totals of what is on. What a switch does is deck.js's ("parts", KnotTalk.deck): a step that is
   off leaves the deck as a deep step does in the short cut. Parts that talk.html marks data-off start off and carry an
   "off by default" tag. Saved per browser (localStorage 'kt-parts', every access guarded): only the presenter's
   overrides of the page's defaults, so "reset" returns to the page's own choice and a later change of a default still
   reaches every part the presenter never touched; "show all" turns everything on. deck.js applies them at load.
   Between windows: a BroadcastChannel where there is one, postMessage between a window and the one it opened, and
   the storage event for the overrides. Every message has an id, so one that arrives twice is applied once; only the
   window where a switch was turned saves. The talk window is the authority: the presenter view shows what it last
   said. With none of these channels (some file:// setups) the drawer in the talk window still works on its own.
   Languages (talk/i18n.js): the panel is in the page's language and carries the language control (L switches here
   too); in Hebrew the presenter view's arrows swap as the deck's do (← next). What windows exchange and what is
   saved are the steps' English keys, so two windows in different languages agree; see "language" below for how a
   switch in one window reaches the other.
   Titles only (deck.js "subtitles", KnotTalk.deck.subs / setSubs): a switch under the cuts, here and in the presenter
   view, which flips the talk window as the cut does. The presenter view always shows the active step's subtitles as
   speaker notes, read from its own copy of the page (the same talk.html, in the same language) by the step's key:
   nothing but the key crosses between windows. A− / A+ set their size (localStorage 'kt-notes').
   Script tab / teleprompter (presenter view): beside "Notes", the script of the active step (talk/script/<lang>.js, loaded
   by talk.html in the presenter view only, for the page's language), large, with the next step's first sentence under it,
   a reading line, a mirror, its own size, and full screen (F; the panel alone). talk.html?prompter = the presenter view
   as that panel alone, filling the window. Up / Down scroll it; ← → PageUp PageDown space step, as always. */
(function () {
  'use strict';
  var KT = window.KnotTalk || {}, D = KT.deck;
  if (!D) return;
  /* languages (talk/i18n.js): every string here goes through T; the step keys windows exchange and save are English in
     every language (deck.js), the titles shown are the page's. A switch of language reloads the page: see "language" */
  var T = KT.t, I18N = KT.i18n || null, RTL = !!KT.rtl;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function mmss(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
  function mins(m) { return m > 0 ? mmss(m * 60) : '–'; }
  function own(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }

  var STORE = 'kt-parts', TAG = 'knot-talk-master';
  var presenter = !!D.presenter;                     // talk.html?master or #master: deck.js decides it, before its first frame
  var prompter = presenter && (/(?:^|&)prompter(?:[=&]|$)/.test(location.search.slice(1)) || location.hash === '#prompter');   // talk.html?prompter: the presenter view as a teleprompter, full window

  /* ---------- saved choices: only the overrides of the page's defaults ---------- */
  function parse(v) { try { var o = JSON.parse(v || '{}'); return o && typeof o === 'object' && !Array.isArray(o) ? o : {}; } catch (e) { return {}; } }
  var wipe = false;                                 // "reset": forget every saved override, also of steps this copy of the page lacks
  /* only the window where the presenter acted saves: a change that came from another window (its message, or its save
     seen through the storage event) is applied here without saving it again, so a late echo cannot overwrite a newer
     choice */
  var afar = 0;
  function fromAfar(f) { afar++; try { f(); } finally { afar--; } }
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
      var s = e.source, mine = false;
      if (!s || s === window || !e.data || e.data.kt !== TAG) return;
      try { mine = s.opener === window; } catch (x) { /* not ours to ask */ }   // a window this one opened, before a reload forgot it (a change of language reloads)
      if (s !== window.opener && peers.indexOf(s) < 0 && !mine) return;
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
  var view = { cut: D.cut(), cur: null, t0: 0, plan: 0, linked: false, subs: D.subs(), info: D.info() };
  function cutNow() { return presenter ? view.cut : D.cut(); }
  function subsNow() { return presenter ? view.subs : D.subs(); }
  function infoNow() { return presenter ? view.info : D.info(); }
  function curNow() { return presenter ? (view.cur && D.byKey(view.cur)) : D.current(); }
  function label(n) { var s = D.section(n); return (s.dataset.rail || '') + ' · ' + D.title(n); }
  function nextAfter(n, m) {
    for (var i = n ? D.steps.indexOf(n) + 1 : 0; i < D.steps.length; i++) if (D.shown(D.steps[i], m)) return D.steps[i];
    return null;
  }

  /* ---------- the panel ---------- */
  var panel = el('div', 'ktm' + (presenter ? ' ktm-page' : '') + (prompter ? ' ktm-prompter' : ''));
  panel.innerHTML = (presenter ? '<div class="ktm-side">' : '') +
    '<header class="ktm-head">' +
      '<div class="ktm-top"><div><div class="ktm-kick">' + (presenter ? T('Knot‑a‑Surface · presenter view') : T('Knot‑a‑Surface · the talk')) + '</div>' +
        '<h2 class="ktm-h" id="ktm-h" tabindex="-1">' + T('Slide master') + '</h2></div>' +
        '<div class="ktm-tools">' + (presenter ? '' :
          '<button type="button" class="ktm-pop">' + T('Presenter view ↗') + '</button>' +
          '<button type="button" class="ktm-x">×</button>') + '</div></div>' +
      '<p class="ktm-note">' + T('Switch parts of the talk on and off. A part that is off leaves the deck: the arrows skip it, and the clock and the totals leave it out.') +
        ' ' + (presenter ? T('This window follows the talk window and drives it.') : T('<kbd>D</kbd> or <kbd>esc</kbd> closes.')) + '</p>' +
      '<div class="ktm-cuts" role="group">' +
        '<button type="button" data-cut="40"><span>' + T('long cut') + '</span><b></b></button>' +
        '<button type="button" data-cut="20"><span>' + T('short cut') + '</span><b></b></button></div>' +
      '<label class="ktm-subs"><input type="checkbox" class="ktm-tog"><span><b>' + T('Titles only') + '</b> <kbd>S</kbd>' +
        '<small>' + T('Hide the subtitles on the slides; the presenter view keeps them as speaker notes.') + '</small></span></label>' +
      '<label class="ktm-subs ktm-infosw"><input type="checkbox" class="ktm-tog"><span><b>' + T('Info boxes off') + '</b> <kbd>I</kbd>' +
        '<small>' + T('Hide the captions, readouts, panels and labels over the simulations: the picture alone.') + '</small></span></label>' +
      '<div class="ktm-acts"><button type="button" class="ktm-all">' + T('Show all') + '</button>' +
        '<button type="button" class="ktm-reset">' + T('Reset to defaults') + '<i></i></button></div>' +
    '</header>' +
    '<div class="ktm-body"><div class="ktm-cols" aria-hidden="true"><span>' + T('part') + '</span><span>' + T('long') + '</span><span>' + T('short') + '</span></div>' +
      '<ol class="ktm-list"></ol></div>' +
    (presenter ? '</div>' +
      /* the presenter view: the talk window's controls and the speaker notes, beside the list (master.css) */
      '<div class="ktm-main"><section class="ktm-live">' +
        '<div class="ktm-nav"><button type="button" class="ktm-prev">' + T('← prev') + '</button>' +
          '<button type="button" class="ktm-next">' + T('next →') + '</button></div>' +
        '<div class="ktm-where"><div><span>' + T('now') + '</span><b class="ktm-cur">–</b></div><div><span>' + T('next') + '</span><b class="ktm-nxt">–</b></div></div>' +
        '<div class="ktm-clock"><button type="button" class="ktm-clk">' + T('start clock') + '</button><span class="ktm-time"></span></div>' +
        '<p class="ktm-link"><span class="ktm-st" role="status"></span> <button type="button" class="ktm-open">' + T('Open the talk ↗') + '</button></p>' +
      '</section>' +
      '<section class="ktm-notes" data-tab="notes"><div class="ktm-nhead"><span class="ktm-tabs" role="tablist">' +
          '<button type="button" role="tab" data-tab="notes">' + T('Notes') + '</button>' +
          '<button type="button" role="tab" data-tab="script">' + T('Script') + '</button></span>' +
        '<span class="ktm-size ktm-nsize" role="group"><button type="button" class="ktm-smaller">A−</button><button type="button" class="ktm-larger">A+</button></span></div>' +
        '<div class="ktm-ntext" tabindex="0"></div>' +
        /* the teleprompter: the script of the active step (talk/script/<lang>.js), large; its own toolbar so that it
           goes full screen with it */
        '<div class="ktm-prompt"><div class="ktm-ptools">' +
          '<button type="button" class="ktm-pprev">' + T('← prev') + '</button><button type="button" class="ktm-pnext">' + T('next →') + '</button>' +
          '<span class="ktm-size ktm-psize" role="group"><button type="button" class="ktm-psmaller">A−</button><button type="button" class="ktm-plarger">A+</button></span>' +
          '<label class="ktm-psw ktm-pline"><input type="checkbox" class="ktm-tog"><span>' + T('reading line') + '</span></label>' +
          '<label class="ktm-psw ktm-pmirror"><input type="checkbox" class="ktm-tog"><span>' + T('mirror') + '</span></label>' +
          '<button type="button" class="ktm-pfull"></button></div>' +
          '<div class="ktm-pbox"><div class="ktm-pmeta"></div>' +
            '<div class="ktm-pbody"><div class="ktm-pband" aria-hidden="true"></div><div class="ktm-ptext" tabindex="0"></div></div>' +
            '<div class="ktm-pnextl"><span class="ktm-pnl">' + T('next') + '</span><span class="ktm-pnt"></span></div>' +
            '<div class="ktm-pcorner"><i class="ktm-pdot"></i><span class="ktm-pcount"></span><button type="button" class="ktm-pclk"><span class="ktm-ptime"></span></button></div></div>' +
        '</div></section></div>' : '');
  /* what is said in an attribute is set here, not written into the markup above: a translation may hold a quote */
  function say(sel, name, text) { var e = $(sel, panel); if (e) e.setAttribute(name, text); }
  say('.ktm-pop', 'title', T('The same list in a second window, synced with this one'));
  say('.ktm-x', 'aria-label', T('Close the slide master (Esc)')); say('.ktm-x', 'title', T('Close · D / esc'));
  say('.ktm-cuts', 'aria-label', T('Length of what is on, per cut; pick the cut in use'));
  say('.ktm-live', 'aria-label', T('The talk window'));
  say('.ktm-prev', 'aria-label', T('Previous step')); say('.ktm-next', 'aria-label', T('Next step'));
  say('.ktm-subs .ktm-tog', 'aria-label', T('Titles only: hide the subtitles of the slides (S)'));
  say('.ktm-infosw .ktm-tog', 'aria-label', T('Info boxes off: hide the boxes over the simulations (I)'));
  say('.ktm-notes', 'aria-label', T('Speaker notes')); say('.ktm-size', 'aria-label', T('Text size'));
  say('.ktm-smaller', 'aria-label', T('Smaller text')); say('.ktm-larger', 'aria-label', T('Larger text'));
  /* "in use" on the cut that is on is written by the stylesheet (master.css): hand it the words of another language */
  if (T('in use') !== 'in use') panel.style.setProperty('--ktm-inuse', JSON.stringify(' · ' + T('in use')));
  /* the language: the HUD's control again, beside "show all" and "reset" (the drawer, and the presenter view) */
  if (I18N) $('.ktm-acts', panel).appendChild(I18N.switcher('ktm-lang'));

  var list = $('.ktm-list', panel), secRows = [], stepRows = [], groups = [], byStep = new Map();
  function tag(t, cls, tip) { var s = el('span', 'ktm-tag' + (cls ? ' ' + cls : ''), t); if (tip) s.title = tip; return s; }
  function row(cls, lab, name, aria) {
    var r = el('div', 'ktm-row ' + cls);
    var tog = el('input', 'ktm-tog'); tog.type = 'checkbox'; tog.dataset.col = 'tog'; tog.setAttribute('aria-label', aria);
    var hit = el('label', 'ktm-hit'); hit.appendChild(tog);         // the whole cell takes the tap, not just the switch
    var go = el('button', 'ktm-go'); go.type = 'button'; go.dataset.col = 'go'; go.textContent = name;
    var nm = el('span', 'ktm-name'); nm.appendChild(go);
    var m40 = el('span', 'ktm-m'), m20 = el('span', 'ktm-m');
    r.appendChild(hit); r.appendChild(el('span', 'ktm-rail', lab)); r.appendChild(nm); r.appendChild(m40); r.appendChild(m20);
    return { row: r, tog: tog, go: go, name: nm, m40: m40, m20: m20 };
  }
  var grp = -1;
  D.sections.forEach(function (sec) {
    var navs = D.navs(sec);
    if (!navs.length) return;                              // nothing in it to show or hide
    var g = sec.closest('.prelude') ? 0 : 1;
    if (g !== grp) {                                       // the light opening and the cover, then the talk
      grp = g;
      var gl = el('li', 'ktm-grp', '<span>' + (g ? T('The talk') : T('Opening and cover · light')) + '</span>');
      var G = { el: gl, g: g, secs: [], m40: el('span', 'ktm-m'), m20: el('span', 'ktm-m') };
      gl.appendChild(G.m40); gl.appendChild(G.m20); list.appendChild(gl); groups.push(G);
    }
    groups[groups.length - 1].secs.push(sec);
    // a part of one step (the cover, a single slide) is one row: its switch is the step's
    var single = navs.length === 1, title = sec.dataset.title || D.title(navs[0]);
    var li = el('li', 'ktm-sec'), part = (sec.dataset.rail || '') + ' · ' + title;
    var R = row('ktm-srow' + (single ? ' ktm-one' : ''), sec.dataset.rail || '', title, single ? part : T('{part}, all its steps', { part: part }));
    R.sec = sec; R.navs = navs; li.appendChild(R.row);
    if (single && navs[0].matches('.step[data-depth="deep"]')) R.name.appendChild(tag(T('deep'), '', T('A deep-dive: the short cut leaves it out')));
    if (sec.hasAttribute('data-off') || (single && navs[0].hasAttribute('data-off')))
      R.name.appendChild(tag(T('off by default'), 'def', T('talk.html marks this part data-off: it stays off until you switch it on')));
    secRows.push(R);
    if (single) { R.name.appendChild(tag(T('now'), 'now')); byStep.set(navs[0], R); }
    else {
      var ol = el('ol', 'ktm-steps');
      navs.forEach(function (n, k) {
        var S = row('ktm-step', String(k + 1), D.title(n), D.title(n));
        S.n = n; S.sec = sec;
        if (n.matches('.step[data-depth="deep"]')) S.name.appendChild(tag(T('deep'), '', T('A deep-dive: the short cut leaves it out')));
        if (n.hasAttribute('data-off')) S.name.appendChild(tag(T('off by default'), 'def', T('talk.html marks this step data-off: it stays off until you switch it on')));
        S.name.appendChild(tag(T('now'), 'now'));
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
  var subsTog = $('.ktm-subs:not(.ktm-infosw) .ktm-tog', panel), infoTog = $('.ktm-infosw .ktm-tog', panel);

  /* ---------- speaker notes (presenter view): the subtitles of the active step, from this page's own copy ---------- */
  var notes = presenter ? { box: $('.ktm-ntext', panel), size: 1 } : null;
  function noteSize(k) {
    if (!notes) return;
    if (k) notes.size = Math.max(0.6, Math.min(2.6, Math.round(notes.size * (k > 0 ? 1.15 : 1 / 1.15) * 100) / 100));
    try { if (k) localStorage.setItem('kt-notes', String(notes.size)); } catch (e) { /* the size lasts as long as the page */ }
    panel.style.setProperty('--ktm-ns', notes.size);
  }
  if (notes) { try { notes.size = Math.max(0.6, Math.min(2.6, +localStorage.getItem('kt-notes') || 1)); } catch (e) {} noteSize(0); }
  function showNotes(n) {
    if (!notes) return;
    var box = notes.box, subs = n ? D.subtitles(n) : [];
    box.textContent = '';
    subs.forEach(function (e) {
      var c = e.cloneNode(true);
      c.removeAttribute('id'); $$('[id]', c).forEach(function (x) { x.removeAttribute('id'); });
      $$('[data-tex]', c).forEach(function (t) { if (!t.firstChild) t.textContent = t.dataset.tex; });   // no KaTeX: the TeX itself
      box.appendChild(c);
    });
    if (!subs.length) box.appendChild(el('p', 'ktm-none', n ? T('No notes for this step.') : '–'));
    box.scrollTop = 0;
  }

  /* ---------- the teleprompter (presenter view): the script of the active step, from talk/script/<lang>.js ----------
     KnotTalk.script[lang] = { "<step key>": "paragraph.\n\nparagraph.", … }, hand-written, keyed by the English step key
     (KnotTalk.deck.key); a line wholly inside square brackets is a stage direction. Only the page's own language is shown.
     Saved in localStorage 'kt-prompt' {tab, size, line, mirror}; every access guarded. */
  var script = (function () { var all = KT.script, o = all && all[KT.lang]; return o && typeof o === 'object' ? o : null; })();
  var pr = null;
  if (presenter) {
    pr = { tab: '', size: 1, line: false, mirror: false, box: $('.ktm-prompt', panel), notes: $('.ktm-notes', panel), text: $('.ktm-ptext', panel),
           meta: $('.ktm-pmeta', panel), nxt: $('.ktm-pnt', panel), count: $('.ktm-pcount', panel), time: $('.ktm-ptime', panel),
           full: $('.ktm-pfull', panel), dot: $('.ktm-pdot', panel), shown: null };
    var sv = {}; try { sv = parse(localStorage.getItem('kt-prompt')); } catch (e) {}
    pr.tab = sv.tab === 'notes' || sv.tab === 'script' ? sv.tab : script ? 'script' : 'notes';
    if (prompter) pr.tab = 'script';
    pr.size = Math.max(0.6, Math.min(2.6, +sv.size || 1)); pr.line = sv.line === true; pr.mirror = sv.mirror === true;
  }
  function savePrompt() {
    if (!pr) return;
    try { localStorage.setItem('kt-prompt', JSON.stringify({ tab: prompter ? (parse(localStorage.getItem('kt-prompt')).tab || '') : pr.tab, size: pr.size, line: pr.line, mirror: pr.mirror })); } catch (e) { /* lasts as long as the page */ }
  }
  function isDir(l) { return /^\s*\[.*\]\s*$/.test(l); }
  function passage(n) { var v = script && n ? script[D.key(n)] : null; return typeof v === 'string' && v.trim() ? v : null; }
  /* the first sentence of a passage, for the hand-over; stage directions do not count */
  function firstSentence(txt) {
    var lines = txt.split(/\n/).map(function (l) { return l.trim(); }).filter(function (l) { return l && !isDir(l); }), t = lines.join(' ');
    var m = /^[\s\S]*?[.!?…:](?=\s|$)/.exec(t), r = m ? m[0] : t;
    return r.length > 170 ? r.slice(0, 167).replace(/\s+\S*$/, '') + '…' : r;
  }
  function showScript(n) {
    if (!pr) return;
    var box = pr.text; box.textContent = '';
    box.setAttribute('dir', KT.rtl ? 'rtl' : 'ltr'); box.setAttribute('lang', KT.lang || 'en');
    var txt = passage(n), msg = !script ? T('No script for this language.') : !n ? '–' : !txt ? T('No script for this step.') : null;
    if (msg) { var q = el('p', 'ktm-none'); q.textContent = msg; box.appendChild(q); }
    else txt.replace(/\r/g, '').split(/\n[ \t]*\n+/).forEach(function (para) {
      var run = [];
      function flush() { if (run.length) { var p = el('p', 'ktm-pp'); p.textContent = run.join('\n'); box.appendChild(p); run = []; } }
      para.split('\n').forEach(function (l) {
        if (!l.trim()) return;
        if (isDir(l)) { flush(); var d = el('div', 'ktm-dir'); d.textContent = l.trim(); box.appendChild(d); } else run.push(l.trim());
      });
      flush();
    });
    box.scrollTop = 0;
  }
  function paintPrompt(cur) {
    if (!pr) return;
    var m = cutNow(), nx = cur ? nextAfter(cur, m) : null;
    pr.meta.textContent = cur ? label(cur) : '–';
    var nt = '';
    if (nx) { var pt = passage(nx); nt = pt ? firstSentence(pt) : label(nx); } else nt = cur ? T('the end') : '–';
    pr.nxt.textContent = nt; pr.nxt.setAttribute('dir', script && nx && passage(nx) ? (KT.rtl ? 'rtl' : 'ltr') : 'auto');
    var vis = D.steps.filter(function (s) { return D.shown(s, m); }), i = cur ? vis.indexOf(cur) : -1;
    if (cur && i < 0) i = vis.filter(function (s) { return D.steps.indexOf(s) < D.steps.indexOf(cur); }).length - 1;   // a step the cut leaves out: after the last one before it
    pr.count.textContent = cur ? T('step {n} / {total}', { n: Math.max(1, i + 1), total: vis.length }) : '';
    pr.dot.className = 'ktm-pdot' + (view.linked ? ' on' : '');
    pr.dot.title = view.linked ? T('Linked to the talk window.') : T('Waiting for the talk window: open talk.html in this browser and this page follows it.');
  }
  function setTab(t, keep) {
    if (!pr) return;
    pr.tab = t; pr.notes.setAttribute('data-tab', t);
    $$('.ktm-tabs button', pr.notes).forEach(function (b) { b.setAttribute('aria-selected', String(b.dataset.tab === t)); });
    if (!keep) savePrompt();
  }
  function setPrompt() {
    if (!pr) return;
    pr.box.style.setProperty('--ktm-pz', pr.size);
    pr.box.classList.toggle('is-line', pr.line); pr.box.classList.toggle('is-mirror', pr.mirror);
    $('.ktm-pline .ktm-tog', pr.box).checked = pr.line; $('.ktm-pmirror .ktm-tog', pr.box).checked = pr.mirror;
    var f = !!(document.fullscreenElement || document.webkitFullscreenElement) || (!prompter && pr.box.classList.contains('ktm-fill'));
    pr.full.textContent = f ? T('exit full screen') : T('full screen');
    pr.full.title = f ? T('Exit full screen · F') : T('Full screen · F');
  }
  function promptSize(k) { pr.size = Math.max(0.6, Math.min(2.6, Math.round(pr.size * (k > 0 ? 1.12 : 1 / 1.12) * 100) / 100)); setPrompt(); savePrompt(); }
  /* full screen: the Fullscreen API on the panel; if the browser refuses, a class that fills the window (always on in ?prompter) */
  function inFull() { return !!(document.fullscreenElement || document.webkitFullscreenElement); }
  function toggleFull() {
    if (!pr || !script) return;
    var b = pr.box, on = prompter ? !inFull() : !(inFull() || b.classList.contains('ktm-fill'));
    if (on) {
      setTab('script', true);
      if (!prompter) b.classList.add('ktm-fill');
      var rq = b.requestFullscreen || b.webkitRequestFullscreen;
      if (rq) { try { var r = rq.call(b); if (r && r.catch) r.catch(function () {}); } catch (e) {} }
    } else {
      if (inFull()) { try { (document.exitFullscreen || document.webkitExitFullscreen).call(document); } catch (e) {} }
      if (!prompter) b.classList.remove('ktm-fill');
    }
    setPrompt();
  }
  if (pr) {
    ['fullscreenchange', 'webkitfullscreenchange'].forEach(function (t) {
      document.addEventListener(t, function () { if (!inFull() && !prompter) pr.box.classList.remove('ktm-fill'); setPrompt(); });
    });
    $$('.ktm-tabs button', pr.notes).forEach(function (b) { b.addEventListener('click', function () { setTab(b.dataset.tab); }); });
    $('.ktm-pprev', pr.box).addEventListener('click', function () { bus.post({ type: 'step', d: -1 }); });
    $('.ktm-pnext', pr.box).addEventListener('click', function () { bus.post({ type: 'step', d: 1 }); });
    $('.ktm-psmaller', pr.box).addEventListener('click', function () { promptSize(-1); });
    $('.ktm-plarger', pr.box).addEventListener('click', function () { promptSize(1); });
    $('.ktm-pline .ktm-tog', pr.box).addEventListener('change', function (e) { pr.line = e.target.checked; setPrompt(); savePrompt(); });
    $('.ktm-pmirror .ktm-tog', pr.box).addEventListener('change', function (e) { pr.mirror = e.target.checked; setPrompt(); savePrompt(); });
    pr.full.addEventListener('click', toggleFull);
    $('.ktm-pclk', pr.box).addEventListener('click', function () { bus.post({ type: 'clock' }); });
    say('.ktm-tabs', 'aria-label', T('Speaker notes or script'));
    $('.ktm-tabs [data-tab="script"]', panel).title = T('Teleprompter: the script of this step');
    $('.ktm-pline', panel).title = T('Reading line: a soft band a third of the way down');
    $('.ktm-pmirror', panel).title = T('Mirror: flip the text, for a glass teleprompter');
    say('.ktm-psize', 'aria-label', T('Text size')); say('.ktm-psmaller', 'aria-label', T('Smaller text')); say('.ktm-plarger', 'aria-label', T('Larger text'));
    if (!script) { $$('.ktm-ptools', pr.box).forEach(function (t) { t.hidden = true; }); }
    setTab(pr.tab, true); setPrompt();
  }

  /* minutes of a part in one cut: what is on of it; a part that is off shows what it would take, struck through (CSS) */
  function setNum(span, v, full, cutName, off) {
    var part = !off && v > 0 && v < full - 1e-9;
    span.textContent = mins(off ? full : v);
    span.classList.toggle('is-cut', part);
    span.title = off ? (full > 0 ? T('{cut}: off · {m} when on', { cut: cutName, m: mins(full) }) : T('{cut}: not in it', { cut: cutName }))
      : part ? T('{cut}: {m} of {full}', { cut: cutName, m: mins(v), full: mins(full) }) : T('{cut}: {m}', { cut: cutName, m: mins(v) });
  }
  var LONG = T('long cut'), SHORT = T('short cut');
  var lastCur = null;
  function paint() {
    var m = cutNow(), cur = curNow();
    cutBtns.forEach(function (b) {
      var c = +b.dataset.cut; $('b', b).textContent = mmss(D.total(c) * 60);
      b.setAttribute('aria-pressed', String(c === m));
    });
    var nOver = Object.keys(D.overrides()).length;
    resetBtn.disabled = !nOver; $('i', resetBtn).textContent = nOver ? String(nOver) : '';
    resetBtn.title = !nOver ? T('Everything is as the page sets it')
      : nOver > 1 ? T('{n} steps differ from the page’s defaults', { n: nOver }) : T('{n} step differs from the page’s defaults', { n: nOver });
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
      R.go.title = shown ? T('Go to this part') : on ? T('The short cut leaves it out') : T('Off: switch it on to go there');
      // a section's full minutes in a cut are what the cut shows of it (a section of deep steps only has none in the short one)
      var f40 = R.navs.some(function (n) { return D.inCut(n, 40); }) ? D.minutes(R.sec, 40) : 0;
      var f20 = R.navs.some(function (n) { return D.inCut(n, 20); }) ? D.minutes(R.sec, 20) : 0;
      setNum(R.m40, D.budget(R.sec, 40), f40, LONG, !on);
      setNum(R.m20, D.budget(R.sec, 20), f20, SHORT, !on);
    });
    stepRows.forEach(function (S) {
      var on = D.isOn(S.n), shown = on && D.inCut(S.n, m);
      S.tog.checked = on;
      S.row.classList.toggle('is-off', !on);
      S.row.classList.toggle('is-out', on && !shown);
      S.row.classList.toggle('is-now', S.n === cur);
      S.go.setAttribute('aria-disabled', String(!shown));
      S.go.title = shown ? T('Go to this step') : on ? T('A deep-dive: the short cut leaves it out') : T('Off: switch it on to go there');
      var a = D.share(S.n, 40), b = D.share(S.n, 20);          // share() is 0 where the cut leaves the step out
      setNum(S.m40, on ? a : 0, a, LONG, !on);
      setNum(S.m20, on ? b : 0, b, SHORT, !on);
    });
    if (live) {
      live.cur.textContent = cur ? label(cur) : '–';
      var nx = cur ? nextAfter(cur, m) : null;
      live.nxt.textContent = nx ? label(nx) : cur ? T('the end') : '–';
      if (live.was !== view.linked) {
        live.was = view.linked;
        live.link.classList.toggle('on', view.linked); live.open.hidden = view.linked;
        live.st.textContent = view.linked ? T('Linked to the talk window.') :
          T('Waiting for the talk window: open talk.html in this browser and this page follows it.') +
          (bus.bc ? '' : ' ' + T('This browser gives the page no BroadcastChannel: open this view from the talk’s slide master (D, then “Presenter view”), or serve the folder over http.'));
      }
      paintClock();
    }
    subsTog.checked = !subsNow(); infoTog.checked = !infoNow();
    if (cur !== lastCur) { lastCur = cur; follow(); showNotes(cur); showScript(cur); }
    paintPrompt(cur);
  }
  function paintClock() {
    if (!live) return;
    var t0 = view.t0, p = view.plan, tot = mmss(D.total(cutNow()) * 60);
    live.clk.textContent = t0 ? T('reset clock') : T('start clock');
    if (!t0) { live.time.className = 'ktm-time'; live.time.innerHTML = T('plan <b>{p}</b> / {tot}', { p: mmss(p), tot: tot }); }
    else {
      var e = (Date.now() - t0) / 1000;
      live.time.className = 'ktm-time ' + (e > p + 45 ? 'late' : 'ok');
      live.time.innerHTML = T('<b>{e}</b> · plan {p} / {tot}', { e: mmss(e), p: mmss(p), tot: tot });
    }
    if (pr) { pr.time.innerHTML = live.time.innerHTML; pr.time.className = 'ktm-ptime ' + live.time.className.replace('ktm-time', '').trim(); }
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
  subsTog.addEventListener('change', function () {
    var on = !subsTog.checked;                            // checked = titles only
    if (presenter) { view.subs = on; paint(); bus.post({ type: 'subs', subs: on }); } else D.setSubs(on);
  });
  infoTog.addEventListener('change', function () {
    var on = !infoTog.checked;                            // checked = boxes off
    if (presenter) { view.info = on; paint(); bus.post({ type: 'info', info: on }); } else D.setInfo(on);
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
  /* a link to this page in another role, in this window's language (said in the address: it holds without storage) */
  function pageUrl(query) {
    var u = location.href.split('#')[0].split('?')[0] + query;
    return I18N ? I18N.url(KT.lang, true, u) : u;
  }
  function openPresenter() {
    var url = pageUrl('?master'), w = null;
    try { w = window.open(url, 'knot-talk-master', 'popup,width=1180,height=860'); } catch (e) {}
    if (w) { bus.peer(w); try { w.focus(); } catch (e) {} }
  }

  if (!presenter) {
    dlg = el('dialog', 'ktm-dialog'); dlg.setAttribute('aria-labelledby', 'ktm-h');
    dlg.appendChild(panel); document.body.appendChild(dlg);
    hb = el('button', 'master', '<svg viewBox="0 0 16 13" aria-hidden="true"><path d="M1.5 2h2.5M1.5 6.5h2.5M1.5 11h2.5M6.5 2h8M6.5 6.5h8M6.5 11h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>');
    hb.type = 'button'; hb.title = T('D · slide master: show / hide parts');
    hb.setAttribute('aria-label', T('Slide master: show or hide parts of the talk (D)'));
    hb.setAttribute('aria-haspopup', 'dialog'); hb.setAttribute('aria-expanded', 'false');
    var hud = $('.hud'), help = hud && $('.help', hud);
    if (hud) hud.insertBefore(hb, help || null);
    hb.addEventListener('click', function (e) { if (isOpen) close(); else open(e.detail > 0); });
    $('.ktm-x', panel).addEventListener('click', close);
    $('.ktm-pop', panel).addEventListener('click', openPresenter);
    dlg.addEventListener('cancel', function (e) { e.preventDefault(); close(); });
    dlg.addEventListener('click', function (e) { if (e.target === dlg) close(); });        // the dimmed deck around it
    dlg.addEventListener('keydown', function (e) {
      var k = D.letter(e);                                  // D and L by position on a keyboard that types Hebrew
      if (!(e.metaKey || e.ctrlKey || e.altKey)) {
        if (k === 'Escape' || (k === 'd' && !e.repeat)) { e.preventDefault(); close(); }
        else if (k === 'l' && !e.repeat && I18N) { e.preventDefault(); I18N.toggle(); }
        else if (k === 's' && !e.repeat) { e.preventDefault(); D.setSubs(!D.subs()); }
        else if (k === 'i' && !e.repeat) { e.preventDefault(); D.setInfo(!D.info()); }
        else if (rove(e, true)) e.preventDefault();
      }
      e.stopPropagation();                                  // the deck behind does not see keys while the drawer is open
    });
    document.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey || e.repeat || D.letter(e) !== 'd') return;
      var t = e.target, tag = t && t.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
      e.preventDefault(); if (isOpen) close(); else open(false);
    });
  } else {
    /* ---------- the presenter view: this page is the panel ---------- */
    document.body.classList.add('kt-presenter');
    document.body.appendChild(panel);
    document.title = I18N ? I18N.strip(T('Slide master · Knot‑a‑Surface')) : T('Slide master · Knot‑a‑Surface');
    $('.ktm-prev', panel).addEventListener('click', function () { bus.post({ type: 'step', d: -1 }); });
    $('.ktm-next', panel).addEventListener('click', function () { bus.post({ type: 'step', d: 1 }); });
    live.clk.addEventListener('click', function () { bus.post({ type: 'clock' }); });
    $('.ktm-smaller', panel).addEventListener('click', function () { noteSize(-1); });
    $('.ktm-larger', panel).addEventListener('click', function () { noteSize(1); });
    live.open.addEventListener('click', function () {
      var w = null;
      try { w = window.open(pageUrl(''), 'knot-talk'); } catch (x) {}
      if (w) bus.peer(w);
    });
    // a clicker or the keyboard drives the talk window; this page's own (hidden) deck never sees a key
    window.addEventListener('keydown', function (e) {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target, k = D.letter(e), ctl = t && (t.tagName === 'BUTTON' || t.tagName === 'INPUT');
      var ahead = RTL ? 'ArrowLeft' : 'ArrowRight', behind = RTL ? 'ArrowRight' : 'ArrowLeft';   // right to left: ← is forward, as in the deck
      var fwd = k === ahead || k === 'PageDown' || (k === ' ' && !ctl && !e.shiftKey);
      var bwd = k === behind || k === 'PageUp' || (k === ' ' && !ctl && e.shiftKey);
      if (fwd || bwd) { e.preventDefault(); bus.post({ type: 'step', d: fwd ? 1 : -1 }); }
      else if (k === 't' && !e.repeat) bus.post({ type: 'clock' });
      else if (k === 's' && !e.repeat) { view.subs = !view.subs; paint(); bus.post({ type: 'subs', subs: view.subs }); }
      else if (k === 'i' && !e.repeat) { view.info = !view.info; paint(); bus.post({ type: 'info', info: view.info }); }
      else if (k === 'l' && !e.repeat && I18N) I18N.toggle();
      else if (k === 'f' && !e.repeat && script) { e.preventDefault(); toggleFull(); }
      else if (k === 'Escape' && pr.box.classList.contains('ktm-fill') && !prompter) toggleFull();
      else if (rove(e, false)) e.preventDefault();
      else if ((k === 'ArrowDown' || k === 'ArrowUp') && pr.tab === 'script' && script) {   // the teleprompter scrolls; Page Up / Down and ← → step
        e.preventDefault();
        var bx = pr.text, lh = parseFloat(getComputedStyle(bx).lineHeight) || 40;
        bx.scrollBy(0, (k === 'ArrowDown' ? 1 : -1) * lh * 2);
      }
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
    bus.post({ type: 'state', over: D.overrides(), cut: D.cut(), cur: c ? D.key(c) : null, t0: D.clock(), plan: D.plan(), subs: D.subs(), info: D.info() });
  }
  /* ---------- language ----------
     A switch reloads the window it is made in (talk/i18n.js, its `kt-lang` event comes first). That window says so on
     the bus, and the windows of the other kind follow without saying it again: the presenter view follows the talk
     window, a talk window follows the presenter view that drives it (two talk windows side by side keep their own
     languages). A presenter view that hears no talk window follows the saved language instead (the storage event).
     The drawer, if the switch was made in it, opens again after the reload. */
  var keptHere = (I18N && I18N.kept && I18N.kept.master) || {};
  function speak(l) { if (I18N && l !== KT.lang) fromAfar(function () { I18N.setLang(l); }); }   // this window in language l, as another window asked
  document.addEventListener('kt-lang', function (e) {
    e.detail.keep.master = { open: isOpen };
    if (!afar) bus.post({ type: 'lang', lang: e.detail.lang, presenter: presenter });
  });
  if (presenter) window.addEventListener('storage', function (e) {
    if (e.key === 'kt-lang' && e.newValue && !view.linked) speak(e.newValue);
  });
  bus.on(function (d) {
    if (d.type === 'lang') { if (!!d.presenter !== presenter) speak(d.lang); return; }
    if (presenter) {
      if (d.type === 'state') {
        view.linked = true; view.cut = d.cut === 20 ? 20 : 40; view.cur = d.cur || null; view.t0 = +d.t0 || 0; view.plan = +d.plan || 0;
        view.subs = d.subs !== false; view.info = d.info !== false;
        var o = d.over && typeof d.over === 'object' ? d.over : {};
        // the talk window's word on what is on (setOverrides repaints through kt-parts)
        if (!same(o, D.overrides())) fromAfar(function () { D.setOverrides(o); }); else paint();
      } else if (d.type === 'bye') { view.linked = false; paint(); }
      return;
    }
    var n;
    if (d.type === 'hello') sendState();
    else if (d.type === 'set') fromAfar(function () {
      (d.keys || []).forEach(function (k) { if ((n = D.byKey(k))) D.set(n, !!d.on); });
      D.refresh();
    });
    else if (d.type === 'all') fromAfar(function () { D.steps.forEach(function (x) { D.set(x, true); }); D.refresh(); });
    else if (d.type === 'reset') fromAfar(function () { D.setOverrides({}); });
    else if (d.type === 'go') { if ((n = D.byKey(d.key))) D.go(n); }
    else if (d.type === 'step') D.step(d.d < 0 ? -1 : 1);
    else if (d.type === 'cut') D.setCut(d.m === 20 ? 20 : 40);
    else if (d.type === 'subs') D.setSubs(d.subs !== false);
    else if (d.type === 'info') D.setInfo(d.info !== false);
    else if (d.type === 'clock') D.toggleClock();
  });
  /* another window of this browser saved a change: apply what changed (twice is harmless) */
  window.addEventListener('storage', function (e) {
    if (e.key !== STORE && e.key !== null) return;
    fromAfar(function () {
      if (e.key === null) { D.setOverrides({}); return; }                     // storage cleared
      var a = parse(e.oldValue), b = parse(e.newValue), k, n, changed = false, keys = Object.assign({}, a, b);
      for (k in keys) if (a[k] !== b[k] && (n = D.byKey(k))) { D.set(n, own(b, k) ? !!b[k] : D.defaultOn(n)); changed = true; }
      if (changed) D.refresh();
    });
  });
  ['kt-parts', 'kt-cut', 'kt-clock', 'kt-step', 'kt-subs', 'kt-info'].forEach(function (t) {
    document.addEventListener(t, function () {
      if (t === 'kt-parts' && !afar) save();
      paint(); sendState();
    });
  });
  if (!presenter) window.addEventListener('pagehide', function () { bus.post({ type: 'bye' }); });

  /* ---------- boot: deck.js has applied the saved choices already (before its first frame); show them, say hello ---------- */
  paint(); sendState();
  if (presenter) bus.post({ type: 'hello' });
  else if (keptHere.open) open(true);                  // the language was switched from the drawer: back in it

  KT.master = { open: open, close: close, isOpen: function () { return isOpen; }, paint: paint, presenter: presenter,
                bus: bus, view: view };
})();
