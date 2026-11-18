/* Knot-a-Surface talk — design variant B: "the talk is its own method".
   On only with talk.html?design=knob (kept for the session; ?design=classic turns it off) or window.KT_DESIGN = 'knob':
   talk.html's head then sets html.kt-knob, and this file does the rest. Without the class it returns at once (one
   exception: a presenter view, talk.html?master, follows a talk window that has it, see "between windows").

   The thesis says: store a small control net and DERIVE the samples at the resolution the moment needs, under a budget,
   spent where the viewer looks. Here the talk does that to itself.
   - CONTROL NET: twelve claims, data-claim="1".."12" on their anchor steps in talk.html (CLAIMS below, their sentences).
   - SAMPLES: every other step, with a weight data-w: 1 a claim, 2 what the story needs next (openers, set-ups and
     pay-offs of a claim), 3 other main steps, 4 deep steps, 5 parts that ship off (data-off, chapter 08).
   - COST: a step's minutes = its section's data-min40, split over the section's steps by their word counts (the English
     of the page, so both languages sample the same set).
   - THE KNOB: minutes, from 5 to the full length. The sampler takes every claim (the floor: always, whatever they cost),
     then the other steps in ORDER until the next one does not fit: weight classes in turn, and inside a class one step
     per column in turn (round robin over the atlas's columns, in the story's order), each column's steps in the page's
     order. Stopping at the first step that does not fit (a prefix of one fixed order) makes the set monotone: what is
     sampled at a budget is sampled at every larger one.
   - MORE HERE: each + on a column moves that column's steps one class earlier (weight - k, never before 1.5, so a
     boosted step comes after the claims and before every unboosted class-2 step); each − one class later (at most 6).
     The budget stays: the order changes, so minutes move to that column from the rest. "reset" forgets the presses and
     the raised steps.
   - RAISED: a tile clicked while it is not sampled is raised for the session: sampled like a claim, whatever the knob.
   - The presenter's slide-master overrides still win: a step switched off there is never sampled. data-off is the
     sampler's to decide (weight 5). Nothing here writes 'kt-parts': the sampled set is a transient layer of the deck
     (KnotTalk.deck.sample, talk/deck.js), stored for the session only ('kt-knob' in sessionStorage).
   - THE ATLAS (A, or the HUD button; Esc closes): the talk drawn as a chart in (u, v). u = the story, columns = the
     chapters (the opening's six short sections share one column); v = depth: row 1 the control claims, row 2 the samples
     (weights 2-3), row 3 the deep ones (4-5). Lit tiles are sampled, ghosts are not; the active step is ringed; seams
     are drawn between the columns. The claims are the control points (control-point colour) of a cubic B-spline drawn
     through the band (surface colour); every other step is a Gaussian on that curve (Gaussian colour), lit when sampled.
   - THE FLOOR: while only the claims are sampled, each claim step shows its number and sentence above its headline.
   Every string goes through KnotTalk.t; the Hebrew is talk/atlas-he.js (loaded by talk.html's head with the design). */
(function () {
  'use strict';
  var KT = window.KnotTalk, D = KT && KT.deck, root = document.documentElement;
  if (!D) return;
  var M = KT.master || null;
  if (!root.classList.contains('kt-knob')) {
    /* between windows: a presenter view opened without the design follows a talk window that has it */
    if (D.presenter && M && M.bus) M.bus.on(function (d) {
      if (d.type !== 'knob' || root.classList.contains('kt-knob')) return;
      root.classList.add('kt-knob'); boot(d);
    });
    return;
  }
  boot(null);

  function boot(remote) {
    var T = KT.t, I18N = KT.i18n || null, RTL = !!KT.rtl, presenter = !!D.presenter;
    function $(s, r) { return (r || document).querySelector(s); }
    function $$(s, r) { return [].slice.call((r || document).querySelectorAll(s)); }
    function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
    function own(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
    function mmss(m) { var s = Math.max(0, Math.round(m * 60)); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
    function num(v) { return String(Math.round(v * 10) / 10); }
    function ltr(s) { return '<span dir="ltr">' + s + '</span>'; }

    /* ---------- the words (English keys; talk/atlas-he.js has the Hebrew) ---------- */
    var CLAIMS = [
      'A computer describes a 3D world as a list or as a question.',
      'Gaussians are a list of soft items: learnable from photographs, fast, with no defined surface.',
      'Free Gaussians fit the photographs and still float off the surface.',
      'Invert the question: optimise a smooth surface and derive every Gaussian from it.',
      'A spline: few control points, local edits, knots where detail is needed.',
      'Position, orientation and size of a Gaussian follow from the surface and its derivatives.',
      'The number of Gaussians is a render-time knob, and a budget can follow the camera.',
      'Parameters live on control points: 16 each, zero per Gaussian.',
      'Capacity comes from knots, resolution from samples: balance the two.',
      'A real object: an owl as charts, fitted offline, with mesh, normals, edits and materials read from the control net.',
      'A surface can be touched: contact is a formula on a spline and undefined on a list.',
      'Evidence and limits: close to PGSR on DTU and behind on average; seams, thin structures, image quality and timing are open.'
    ];
    var S = {
      resample: 're-sample the talk',
      readout: '{min} min · {s} of {t} steps',
      floorNote: 'The floor: only the twelve claims, {m} min in all.',
      knobAria: 'Length of the talk in minutes: turning it re-samples the talk',
      atlasBtn: 'A · the talk as a chart: control points, samples, budget',
      atlasAria: 'The talk as a chart',
      kicker: 'Knot‑a‑Surface · the talk',
      title: 'The talk, as a chart',
      cp: 'control points', samples: 'samples', budget: 'budget',
      row1: 'control claims', row2: 'samples', row3: 'deep',
      seam: 'seam',
      more: 'More here: move minutes to {part}',
      less: 'Less here: move minutes away from {part}',
      reset: 'reset',
      resetTip: 'Spread the budget by weight again, and drop the raised steps',
      raised: 'raised for this session',
      notSampled: 'not sampled at {m} min',
      offHere: 'Switched off in the slide master',
      goTip: 'Go to this step',
      raiseTip: 'Not sampled: a click raises it for this session and goes there',
      close: 'Close the atlas (Esc)',
      hint: 'Click a tile to go there · A or Esc closes',
      opening: 'The opening',
      minutes: 'minutes',
      colMin: '{on} of {all} min sampled',
      keyA: 'the talk as a chart · the knob in the corner re-samples the talk'
    };

    /* ---------- the steps: weight, claim, cost, column ---------- */
    var steps = D.steps, secs = D.sections, info = new Map();
    function words(n) {
      var c = I18N ? I18N.source(n) : n.cloneNode(true);
      $$('[data-tex]', c).forEach(function (t) { t.textContent = ' x '; });
      $$('.kt-claimline', c).forEach(function (t) { t.remove(); });
      return Math.max(1, (c.textContent.match(/\S+/g) || []).length);
    }
    /* columns: one per data-rail section, except the opening's short sections (inside .prelude, before the cover), one column */
    var cols = [], colOf = new Map();
    secs.forEach(function (sec) {
      var navs = D.navs(sec);
      if (!navs.length) return;
      var opening = !!sec.closest('.prelude') && !sec.matches('.cover');
      var last = cols[cols.length - 1];
      if (opening && last && last.opening) { last.secs.push(sec); last.navs = last.navs.concat(navs); }
      else cols.push({ secs: [sec], navs: navs.slice(), opening: opening, i: cols.length });
    });
    cols.forEach(function (c) {
      c.rail = c.opening ? c.secs[0].dataset.rail + '–' + c.secs[c.secs.length - 1].dataset.rail : c.secs[0].dataset.rail;
      c.key = c.secs[0].dataset.rail;                                         // the column's name in the stored state
      c.title = c.opening ? T(S.opening) : (c.secs[0].dataset.title || '');
      c.secs.forEach(function (s) { colOf.set(s, c); });
    });
    secs.forEach(function (sec) {
      var navs = D.navs(sec), wd = navs.map(words), tot = wd.reduce(function (a, b) { return a + b; }, 0) || 1;
      navs.forEach(function (n, k) {
        var w = +n.dataset.w || (n.matches('.step[data-depth="deep"]') ? 4 : 3), cl = +n.dataset.claim || 0;
        if (n.hasAttribute('data-off') || sec.hasAttribute('data-off')) w = Math.max(w, 5);
        info.set(n, { w: cl ? 1 : Math.max(2, w), claim: cl, cost: D.minutes(sec, 40) * wd[k] / tot, col: colOf.get(sec), key: D.key(n) });
      });
    });
    var claimSteps = steps.filter(function (n) { return info.get(n).claim; }).sort(function (a, b) { return info.get(a).claim - info.get(b).claim; });

    /* ---------- the state, for the session ---------- */
    var STORE = 'kt-knob', st = { min: 20, more: {}, raised: {} };
    try { var o = JSON.parse(sessionStorage.getItem(STORE) || 'null'); if (o && typeof o === 'object') adopt(o); } catch (e) { /* no storage: defaults */ }
    function adopt(o) {
      if (isFinite(+o.min)) st.min = +o.min;
      st.more = {}; st.raised = {};
      var k;
      if (o.more && typeof o.more === 'object') for (k in o.more) if (own(o.more, k) && isFinite(+o.more[k]) && +o.more[k]) st.more[k] = Math.max(-4, Math.min(4, Math.round(+o.more[k])));
      if (o.raised && typeof o.raised === 'object') for (k in o.raised) if (own(o.raised, k) && D.byKey(k)) st.raised[k] = 1;
    }
    var qm = /[?&]min=([0-9.]+)(?:&|$)/.exec(location.search);
    if (qm) {                          // ?min=18 sets the knob once: kept for the session, and gone from the address so a reload keeps later turns
      st.min = +qm[1];
      try { history.replaceState(null, '', location.pathname + location.search.replace(/([?&])min=[^&]*&?/, '$1').replace(/[?&]$/, '') + location.hash); } catch (e) {}
    }
    if (remote) adopt(remote);
    function save() { try { sessionStorage.setItem(STORE, JSON.stringify(st)); } catch (e) {} }
    if (qm) save();

    /* ---------- the sampler ---------- */
    function excluded(n) { var o = D.overrides(); return own(o, info.get(n).key) && o[info.get(n).key] === false; }
    function forced(n) { var I = info.get(n); return !excluded(n) && (I.claim > 0 || own(st.raised, I.key)); }
    function prio(n) { var I = info.get(n); return I.claim ? 1 : Math.min(6, Math.max(1.5, I.w - (st.more[I.col.key] || 0))); }
    function order() {
      var classes = {}, ps = [], out = [];
      steps.forEach(function (n) {
        if (excluded(n) || forced(n)) return;
        var p = prio(n), c = info.get(n).col.i;
        if (!classes[p]) { classes[p] = {}; ps.push(p); }
        (classes[p][c] = classes[p][c] || []).push(n);
      });
      ps.sort(function (a, b) { return a - b; }).forEach(function (p) {
        var q = classes[p], left = true;
        while (left) {
          left = false;
          cols.forEach(function (c) { var a = q[c.i]; if (a && a.length) { out.push(a.shift()); left = true; } });
        }
      });
      return out;
    }
    function maxMin() { return steps.reduce(function (t, n) { return t + (excluded(n) ? 0 : info.get(n).cost); }, 0); }
    function knobMax() { return Math.max(6, Math.ceil(maxMin() * 2) / 2); }
    var set = new Set(), prev = null, used = 0, floorCost = 0, floor = false;
    function sample() {
      st.min = Math.max(5, Math.min(knobMax(), +st.min || 20));
      var s = new Set(), u = 0;
      steps.forEach(function (n) { if (forced(n)) { s.add(n); u += info.get(n).cost; } });
      floorCost = u;
      var nForced = s.size;
      order().every(function (n) {
        var c = info.get(n).cost;
        if (u + c > st.min + 1e-9) return false;
        s.add(n); u += c; return true;
      });
      prev = set; set = s; used = u; floor = s.size === nForced;
      root.classList.toggle('kt-floor', floor);
      D.sample({ has: function (n) { return set.has(n); }, cost: function (n) { var I = info.get(n); return I ? I.cost : 0; } });
    }
    function eligible() { return steps.filter(function (n) { return !excluded(n); }).length; }
    function readout() {
      var shown = used > st.min + 1e-9 || set.size === eligible() ? num(used) : num(st.min);
      return T(S.readout, { min: shown, s: set.size, t: eligible() });
    }

    /* ---------- the claim line, shown at the floor ---------- */
    claimSteps.forEach(function (n) {
      var c = info.get(n).claim, line = el('div', 'kt-claimline');
      line.innerHTML = '<span class="kt-cl-n">' + ltr(c + ' / 12') + '</span><span class="kt-cl-t">' + T(CLAIMS[c - 1]) + '</span>';
      var host = n.matches('.p-hero') ? $('.p-text', n) || n : n;
      var h = $(':scope>h1, :scope>h2, :scope>.thesis', host) || $('h1, h2', host);
      if (h && h.parentNode) h.parentNode.insertBefore(line, h); else host.insertBefore(line, host.firstChild);
    });

    /* ---------- the HUD knob (where the two cut buttons were) ---------- */
    var hudKnob = null, hudOut = null, hudRange = null;
    var hud = $('.hud');
    if (hud && !presenter) {
      hudKnob = el('div', 'kt-knobhud');
      hudKnob.innerHTML = '<label class="kt-kh-l"><small>' + T(S.resample) + '</small>' +
        '<input type="range" min="5" step="0.5"></label><output class="kt-kh-o" aria-live="off"></output>';
      hudRange = $('input', hudKnob); hudOut = $('output', hudKnob);
      hudRange.setAttribute('aria-label', T(S.knobAria));
      var ab = el('button', 'kt-atlasbtn', '<svg viewBox="0 0 16 13" aria-hidden="true"><path d="M1.4 9.6C4 2.4 6.6 2.4 8 6.5s4 4.1 6.6-3" fill="none" stroke-width="1.6" stroke-linecap="round"/>' +
        '<circle cx="1.6" cy="9.4" r="1.3"/><circle cx="8" cy="6.5" r="1.3"/><circle cx="14.4" cy="3.6" r="1.3"/></svg>');
      ab.type = 'button'; ab.title = T(S.atlasBtn); ab.setAttribute('aria-label', T(S.atlasAria) + ' (A)');
      ab.addEventListener('click', function () { toggleAtlas(); });
      hudKnob.appendChild(ab);
      hud.insertBefore(hudKnob, $('.cut', hud) || hud.firstChild);
      hudRange.addEventListener('input', function () { setMin(+hudRange.value); });
      var keys = $('.keys');
      if (keys) { var kl = el('div', 'kt-keyA'); kl.innerHTML = '<kbd>A</kbd>' + T(S.keyA); keys.appendChild(kl); }
    }
    function paintKnobs() {
      var mx = knobMax();
      [hudRange, atl && atl.range].forEach(function (r) {
        if (!r) return;
        r.max = mx; if (document.activeElement !== r || Math.abs(+r.value - st.min) > 0.26) r.value = st.min;
        r.style.setProperty('--kt-fill', ((st.min - 5) / (mx - 5) * 100).toFixed(1) + '%');
      });
      var txt = readout(), note = floor && floorCost > st.min ? T(S.floorNote, { m: num(floorCost) }) : '';
      if (hudOut) { hudOut.innerHTML = txt; hudOut.title = note; hudKnob.classList.toggle('is-floor', floor); }
      if (atl) {
        atl.out.innerHTML = txt; atl.note.textContent = note;
        var any = Object.keys(st.more).length + Object.keys(st.raised).length;
        atl.reset.disabled = !any;
      }
    }
    function setMin(v) { st.min = v; save(); update(); }

    /* ---------- the atlas ---------- */
    var atl = null, open = false;
    function buildAtlas() {
      var a = atl = {};
      var box = a.box = el('div', 'kta');
      box.setAttribute('role', 'dialog'); box.setAttribute('aria-modal', 'true'); box.setAttribute('aria-label', T(S.atlasAria)); box.hidden = true;
      box.innerHTML =
        '<div class="kta-top">' +
          '<svg class="kta-uv" viewBox="0 0 44 44" aria-hidden="true" dir="ltr"><path d="M6 6H36M31 2.5 36 6 31 9.5M6 6V36M2.5 31 6 36 9.5 31" fill="none" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>' +
            '<text x="38" y="17" text-anchor="end">u</text><text x="12" y="41">v</text></svg>' +
          '<div class="kta-ttl"><div class="kta-kick">' + T(S.kicker) + ' · <span>' + T(S.hint) + '</span></div><h2>' + T(S.title) + '</h2>' +
            '<p class="kta-legend"><span class="kta-lg-c"><i></i>' + T(S.cp) + '</span><span class="kta-lg-s"><i></i>' + T(S.samples) + '</span>' +
            '<span class="kta-lg-b"><i></i>' + T(S.budget) + '</span></p></div>' +
          '<div class="kta-knob"><label><span>' + T(S.resample) + '</span><input type="range" min="5" step="0.5"></label>' +
            '<div class="kta-kr"><output></output><button type="button" class="kta-reset"></button></div><small class="kta-note"></small></div>' +
          '<button type="button" class="kta-x">×</button>' +
        '</div>' +
        '<div class="kta-scroll"><div class="kta-board"><svg class="kta-svg" aria-hidden="true"></svg><div class="kta-grid"></div><div class="kta-claims"></div></div></div>';
      document.body.appendChild(box);
      a.range = $('.kta-knob input', box); a.out = $('.kta-knob output', box); a.note = $('.kta-note', box);
      a.range.setAttribute('aria-label', T(S.knobAria));
      a.reset = $('.kta-reset', box); a.reset.textContent = T(S.reset); a.reset.title = T(S.resetTip);
      a.reset.addEventListener('click', function () { st.more = {}; st.raised = {}; save(); update(); });
      a.range.addEventListener('input', function () { setMin(+a.range.value); });
      var x = $('.kta-x', box); x.setAttribute('aria-label', T(S.close)); x.title = T(S.close);
      x.addEventListener('click', function () { toggleAtlas(false); });
      a.scroll = $('.kta-scroll', box); a.board = $('.kta-board', box); a.grid = $('.kta-grid', box);
      a.claims = $('.kta-claims', box); a.svg = $('.kta-svg', box);
      a.tiles = new Map(); a.cols = [];
      var g = a.grid;
      g.style.gridTemplateColumns = 'var(--kta-gut) ' + cols.map(function (c) {
        var n2 = c.navs.filter(function (n) { var w = info.get(n).w; return w === 2 || w === 3; }).length;
        var n3 = c.navs.filter(function (n) { return info.get(n).w >= 4; }).length;
        var sub = Math.max(n2 > 6 ? 2 : 1, n3 > 6 ? 2 : 1), wide = 1;
        c.sub = sub;
        return 'minmax(calc(var(--kta-tw) * ' + (sub * wide).toFixed(2) + '), ' + (sub * wide).toFixed(2) + 'fr)';
      }).join(' ');
      ['row1', 'row2', 'row3'].forEach(function (r, k) {
        var lab = el('div', 'kta-rowlab kta-rl' + (k + 1), '<span>' + T(S[r]) + '</span>');
        lab.style.gridRow = String(k + 2); lab.style.gridColumn = '1'; g.appendChild(lab);
      });
      cols.forEach(function (c, ci) {
        var C = { c: c, ci: ci };
        var head = el('div', 'kta-head');
        head.style.gridColumn = String(ci + 2); head.style.gridRow = '1';
        head.innerHTML = '<div class="kta-h1"><b dir="ltr">' + esc(c.rail) + '</b>' +
          '<span class="kta-pm"><button type="button" class="kta-less">−</button><i class="kta-lvl"></i><button type="button" class="kta-more">+</button></span></div>' +
          '<div class="kta-ht">' + esc(c.title) + '</div><div class="kta-bar"><i></i></div>';
        var part = c.rail + ' · ' + c.title;
        var more = $('.kta-more', head), less = $('.kta-less', head);
        more.title = T(S.more, { part: part }); more.setAttribute('aria-label', more.title);
        less.title = T(S.less, { part: part }); less.setAttribute('aria-label', less.title);
        more.addEventListener('click', function () { bump(c, 1); });
        less.addEventListener('click', function () { bump(c, -1); });
        g.appendChild(head); C.head = head; C.bar = $('.kta-bar', head); C.lvl = $('.kta-lvl', head);
        var band = el('div', 'kta-bandcell'); band.style.gridColumn = String(ci + 2); band.style.gridRow = '2'; g.appendChild(band); C.band = band;
        [2, 3].forEach(function (row) {
          var cell = el('div', 'kta-cell kta-r' + row);
          cell.style.gridColumn = String(ci + 2); cell.style.gridRow = String(row + 1);
          cell.style.setProperty('--kta-sub', c.sub);
          c.navs.forEach(function (n) {
            var I = info.get(n);
            if (I.claim || (row === 2 ? I.w > 3 : I.w < 4)) return;
            var t = el('button', 'kta-tile' + (row === 3 ? ' kta-thin' : ''));
            t.type = 'button';
            t.innerHTML = '<span class="kta-min">' + mmss(I.cost) + '</span><span class="kta-tt">' + esc(D.title(n)) + '</span><span class="kta-rz">' + T(S.raised) + '</span>';
            t.addEventListener('click', function () { pick(n); });
            cell.appendChild(t); a.tiles.set(n, t);
          });
          g.appendChild(cell);
        });
        a.cols.push(C);
      });
      claimSteps.forEach(function (n) {
        var I = info.get(n), t = el('button', 'kta-claim kta-lane' + (I.claim % 2 ? 'a' : 'b'));
        t.type = 'button';
        t.innerHTML = '<span class="kta-cn">' + I.claim + '</span><span class="kta-ct">' + T(CLAIMS[I.claim - 1]) + '</span>' +
          '<span class="kta-cm"><bdi dir="ltr">' + esc(I.col.rail) + '</bdi> · <bdi>' + mmss(I.cost) + '</bdi></span>';
        t.title = D.title(n);
        t.addEventListener('click', function () { pick(n); });
        a.claims.appendChild(t); a.tiles.set(n, t);
      });
      box.addEventListener('click', function (e) { if (e.target === box) toggleAtlas(false); });
    }
    function bump(c, d) {
      var v = (st.more[c.key] || 0) + d;
      v = Math.max(-4, Math.min(4, v));
      if (v) st.more[c.key] = v; else delete st.more[c.key];
      save(); update();
    }
    function pick(n) {
      toggleAtlas(false);
      if (!set.has(n) && !excluded(n)) { st.raised[info.get(n).key] = 1; save(); update(); }
      D.go(n);
    }

    /* geometry: the band of claims, the curve through them, the samples on it, the seams (on open and on resize) */
    var SVGNS = 'http://www.w3.org/2000/svg';
    function svg(tag, at, parent) { var e = document.createElementNS(SVGNS, tag), k; for (k in at) e.setAttribute(k, at[k]); if (parent) parent.appendChild(e); return e; }
    function layoutAtlas() {
      var a = atl; if (!a) return;
      var bw = a.board.clientWidth, gut = a.grid.firstElementChild.getBoundingClientRect().width;
      var tw = Math.max(150, Math.min(250, (bw - gut - 6 * 12) / 6)), gap = 10;
      a.board.style.setProperty('--kta-cw', tw.toFixed(0) + 'px');
      /* lanes: odd claims above the curve, even below; each tile as close to its column as its neighbours allow */
      var bR = a.board.getBoundingClientRect();
      function u(r) { return RTL ? bR.right - r.right : r.left - bR.left; }     // distance along the story, from its start
      var colU = a.cols.map(function (C) { var r = C.head.getBoundingClientRect(); return { u0: u(r), w: r.width }; });
      var anchors = new Map();
      cols.forEach(function (c, ci) {
        var cl = c.navs.filter(function (n) { return info.get(n).claim; }), sm = c.navs.filter(function (n) { return !info.get(n).claim; });
        cl.forEach(function (n, k) { anchors.set(n, colU[ci].u0 + colU[ci].w * (k + 1) / (cl.length + 1)); });
        sm.forEach(function (n, k) { anchors.set(n, colU[ci].u0 + colU[ci].w * (k + 0.5) / sm.length); });
      });
      var lanes = { a: [], b: [] };
      claimSteps.forEach(function (n) { lanes[info.get(n).claim % 2 ? 'a' : 'b'].push(n); });
      var lo = gut + 4, hi = bw - 4, place = new Map(), laneH = { a: 0, b: 0 };
      Object.keys(lanes).forEach(function (L) {
        var ns = lanes[L], xs = ns.map(function (n) { return anchors.get(n) - tw / 2; }), i, it;
        for (it = 0; it < 40; it++) {
          for (i = 0; i < xs.length; i++) xs[i] = Math.max(lo, Math.min(hi - tw, xs[i]));
          for (i = 1; i < xs.length; i++) if (xs[i] < xs[i - 1] + tw + gap) { var m = (xs[i] + xs[i - 1] + tw + gap) / 2; xs[i - 1] = m - tw - gap; xs[i] = m; }
        }
        for (i = 0; i < xs.length; i++) xs[i] = Math.max(lo, Math.min(hi - tw, xs[i]));
        for (i = 1; i < xs.length; i++) xs[i] = Math.max(xs[i], xs[i - 1] + tw + gap);
        for (i = xs.length - 2; i >= 0; i--) xs[i] = Math.min(xs[i], xs[i + 1] - tw - gap);
        ns.forEach(function (n, k) {
          var t = a.tiles.get(n); t.style.width = tw + 'px'; t.style.insetInlineStart = xs[k] + 'px';
          place.set(n, xs[k]); laneH[L] = Math.max(laneH[L], t.offsetHeight);
        });
      });
      var mid = 34, bandH = laneH.a + mid + laneH.b + 12;
      a.board.style.setProperty('--kta-band', bandH + 'px');
      var band0 = a.cols[0].band.getBoundingClientRect().top - bR.top;
      a.claims.style.top = band0 + 'px'; a.claims.style.height = bandH + 'px';
      lanes.b.forEach(function (n) { a.tiles.get(n).style.top = (laneH.a + mid + 8) + 'px'; });
      lanes.a.forEach(function (n) { var t = a.tiles.get(n); t.style.top = (laneH.a - t.offsetHeight + 4) + 'px'; });
      /* the curve: a clamped cubic B-spline whose control points are the claims (at their columns, a little up or down) */
      var yMid = band0 + laneH.a + 8 + mid / 2, amp = mid * 0.36;
      var P = [[gut, yMid]];
      claimSteps.forEach(function (n) { P.push([anchors.get(n), yMid + (info.get(n).claim % 2 ? -amp : amp)]); });
      P.push([bw, yMid]);
      P.sort(function (p, q) { return p[0] - q[0]; });
      var curve = bspline(P, 480);
      function yAt(x) {
        var i = 1; while (i < curve.length - 1 && curve[i][0] < x) i++;
        var p = curve[i - 1], q = curve[i], t = q[0] > p[0] ? (x - p[0]) / (q[0] - p[0]) : 0;
        return p[1] + (q[1] - p[1]) * Math.max(0, Math.min(1, t));
      }
      var h = a.board.scrollHeight, s = a.svg;
      s.setAttribute('width', bw); s.setAttribute('height', h); s.setAttribute('viewBox', '0 0 ' + bw + ' ' + h);
      while (s.firstChild) s.removeChild(s.firstChild);
      var g = svg('g', RTL ? { transform: 'matrix(-1 0 0 1 ' + bw + ' 0)' } : {}, s);
      /* seams between the columns, from the top of the band to the foot of the board, and the word once */
      var top = a.cols[0].head.getBoundingClientRect().bottom - bR.top;
      colU.forEach(function (c, i) {
        if (!i) return;
        svg('path', { d: 'M' + (c.u0 - 0.5) + ' ' + top + 'V' + (h - 18), class: 'kta-seam' }, g);
      });
      if (colU[1]) {
        var t = svg('text', { x: colU[1].u0 - 0.5, y: h - 5, class: 'kta-seamw', 'text-anchor': 'middle' }, s);
        t.textContent = KT.i18n ? KT.i18n.strip(T(S.seam)) : T(S.seam);
        if (RTL) t.setAttribute('x', bw - colU[1].u0 + 0.5);
      }
      svg('path', { d: 'M' + P.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('L'), class: 'kta-poly' }, g);
      svg('path', { d: 'M' + curve.map(function (p) { return p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join('L'), class: 'kta-curve' }, g);
      a.dots = new Map();
      claimSteps.forEach(function (n) {
        var x = anchors.get(n), y = yMid + (info.get(n).claim % 2 ? -amp : amp), tile = a.tiles.get(n);
        var tx = place.get(n) + tw / 2, ty = info.get(n).claim % 2 ? band0 + laneH.a + 4 : band0 + laneH.a + mid + 8;
        svg('path', { d: 'M' + tx + ' ' + ty + 'L' + x + ' ' + y, class: 'kta-lead' }, g);
        var d = svg('circle', { cx: x, cy: y, r: 5.5, class: 'kta-cp' }, g); a.dots.set(n, d); tile._dot = d;
      });
      steps.forEach(function (n) {
        if (info.get(n).claim) return;
        var x = anchors.get(n), d = svg('circle', { cx: x, cy: yAt(x), r: 4.2, class: 'kta-gs' }, g);
        var ti = svg('title', {}, d); ti.textContent = D.title(n);
        a.dots.set(n, d);
      });
      paintAtlas(true);
    }
    /* a clamped uniform cubic B-spline through control points P (Cox–de Boor, as talk/nurbs.js), sampled at m points */
    function bspline(P, m) {
      var n = P.length - 1, p = Math.min(3, n), U = [], i, out = [];
      for (i = 0; i <= n + p + 1; i++) U.push(i <= p ? 0 : i >= n + 1 ? 1 : (i - p) / (n - p + 1));
      function N(i, k, t) {
        if (k === 0) return (U[i] <= t && t < U[i + 1]) || (t >= 1 && U[i] < 1 && U[i + 1] >= 1) ? 1 : 0;
        var a = U[i + k] - U[i], b = U[i + k + 1] - U[i + 1];
        return (a ? (t - U[i]) / a * N(i, k - 1, t) : 0) + (b ? (U[i + k + 1] - t) / b * N(i + 1, k - 1, t) : 0);
      }
      for (var j = 0; j <= m; j++) {
        var t = j / m, x = 0, y = 0;
        for (i = 0; i <= n; i++) { var w = N(i, p, t); x += w * P[i][0]; y += w * P[i][1]; }
        out.push([x, y]);
      }
      return out;
    }
    var flash = new Map();
    function paintAtlas(quiet) {
      var a = atl; if (!a) return;
      var cur = D.current();
      a.tiles.forEach(function (t, n) {
        var on = set.has(n), off = excluded(n), was = prev ? prev.has(n) : on, I = info.get(n);
        t.classList.toggle('is-lit', on); t.classList.toggle('is-ghost', !on); t.classList.toggle('is-off', off);
        t.classList.toggle('is-now', n === cur);
        t.classList.toggle('is-raised', own(st.raised, I.key) && !I.claim);
        t.title = (I.claim ? D.title(n) + ' · ' : '') + (off ? T(S.offHere) : on ? T(S.goTip) : T(S.raiseTip));
        if (!quiet && open && on !== was) {
          clearTimeout(flash.get(t)); t.classList.remove('kta-in', 'kta-out'); void t.offsetWidth;
          t.classList.add(on ? 'kta-in' : 'kta-out');
          flash.set(t, setTimeout(function () { t.classList.remove('kta-in', 'kta-out'); }, 1300));
        }
        var d = a.dots && a.dots.get(n);
        if (d) { d.classList.toggle('is-lit', on); d.classList.toggle('is-now', n === cur); }
      });
      a.cols.forEach(function (C) {
        var c = C.c, on = 0, all = 0;
        c.navs.forEach(function (n) { var k = info.get(n).cost; if (!excluded(n)) all += k; if (set.has(n)) on += k; });
        C.bar.firstChild.style.width = (all ? on / all * 100 : 0).toFixed(1) + '%';
        C.bar.title = T(S.colMin, { on: mmss(on), all: mmss(all) });
        var lv = st.more[c.key] || 0;
        C.lvl.textContent = lv ? (lv > 0 ? '+' + lv : '−' + (-lv)) : '';
        C.head.classList.toggle('is-more', lv > 0); C.head.classList.toggle('is-less', lv < 0);
        C.head.classList.toggle('is-now', !!cur && c.navs.indexOf(cur) >= 0);
      });
    }
    function toggleAtlas(on) {
      if (presenter) return;
      on = on == null ? !open : !!on;
      if (on === open) return;
      if (on && !atl) buildAtlas();
      open = on; atl.box.hidden = !on;
      root.classList.toggle('kt-atlas-open', on);
      if (on) {
        paintKnobs(); layoutAtlas();
        var cur = D.current(), t = cur && atl.tiles.get(cur);
        if (t && atl.scroll.scrollWidth > atl.scroll.clientWidth) t.scrollIntoView({ block: 'nearest', inline: 'center' });
        try { $('.kta-x', atl.box).focus({ preventScroll: true }); } catch (e) {}
      } else if (document.activeElement && atl.box.contains(document.activeElement)) document.activeElement.blur();
    }
    var rz = 0;
    window.addEventListener('resize', function () { if (!open) return; cancelAnimationFrame(rz); rz = requestAnimationFrame(layoutAtlas); });
    document.addEventListener('keydown', function (e) {
      if (presenter || e.metaKey || e.ctrlKey || e.altKey) return;
      var t = e.target, tag = t && t.tagName;
      if (tag === 'TEXTAREA' || tag === 'SELECT' || (tag === 'INPUT' && t.type !== 'range')) return;
      var k = D.letter ? D.letter(e) : (e.key || '').toLowerCase();
      if (k === 'a' && !e.repeat) { e.preventDefault(); toggleAtlas(); }
      else if (k === 'Escape' && open) toggleAtlas(false);
    });

    /* ---------- the slide master: a step the knob leaves out says so, distinct from "off" ---------- */
    function decorateMaster() {
      var panel = $('.ktm'); if (!panel) return;
      var lis = $$('.ktm-list > li.ktm-sec', panel), k = 0;
      var withNavs = secs.filter(function (s) { return D.navs(s).length; });
      var head = $$('.ktm-cols span', panel);
      if (head[1] && !head[1]._kt) { head[1]._kt = 1; head[1].textContent = T(S.minutes); }
      withNavs.forEach(function (sec) {
        var li = lis[k++]; if (!li) return;
        var navs = D.navs(sec), rows = navs.length === 1 ? [$('.ktm-srow', li)] : $$('.ktm-steps .ktm-row', li);
        navs.forEach(function (n, j) {
          var r = rows[j]; if (!r) return;
          var nm = $('.ktm-name', r), tg = $('.kt-ns', r), out = !set.has(n) && D.isOn(n);
          r.classList.toggle('kt-ns-row', out);
          if (out && !tg) { tg = el('span', 'ktm-tag kt-ns'); nm.insertBefore(tg, $('.ktm-tag.now', nm)); }
          if (tg) { tg.style.display = out ? '' : 'none'; if (out) tg.textContent = T(S.notSampled, { m: num(st.min) }); }
        });
      });
    }

    /* ---------- between windows: the talk window says where its knob is; a presenter view follows ---------- */
    function post() { if (M && M.bus && !presenter) M.bus.post({ type: 'knob', min: st.min, more: st.more, raised: st.raised }); }
    if (M && M.bus) M.bus.on(function (d) {
      if (presenter && d.type === 'knob') { adopt(d); save(); update(true); }
      else if (!presenter && d.type === 'hello') post();
    });

    /* ---------- one change, everywhere ---------- */
    function update(fromAfar) {
      sample(); paintKnobs(); if (open) paintAtlas(); decorateMaster();
      if (!fromAfar) post();
    }
    document.addEventListener('kt-step', function () { if (open) paintAtlas(true); });
    document.addEventListener('kt-parts', function () { update(true); });      // the slide master switched something: sample again
    document.addEventListener('kt-cut', function () {
      if (document.body.classList.contains('cut-20')) document.body.classList.remove('cut-20');   // the knob is the cut now
      decorateMaster();
    });
    document.body.classList.remove('cut-20');
    update(!!remote);
    KT.atlas = {
      state: function () { return JSON.parse(JSON.stringify(st)); },
      setMin: setMin, more: function (rail, d) { var c = cols.filter(function (x) { return x.key === rail || x.secs.some(function (s) { return s.dataset.rail === rail; }); })[0]; if (c) bump(c, d || 1); },
      reset: function () { st.more = {}; st.raised = {}; save(); update(); },
      sampled: function () { return steps.filter(function (n) { return set.has(n); }).map(D.key); },
      used: function () { return used; }, floorCost: function () { return floorCost; }, max: maxMin, knobMax: knobMax,
      cost: function (n) { return info.get(n).cost; }, weight: function (n) { return prio(n); }, claims: claimSteps,
      columns: cols, open: function () { toggleAtlas(true); }, close: function () { toggleAtlas(false); }, isOpen: function () { return open; },
      claimText: function (i) { return CLAIMS[i - 1]; }, strings: S, claimsEn: CLAIMS.slice()
    };
  }
})();
