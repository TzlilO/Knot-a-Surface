/* Knot-a-Surface talk — the core cut: the steps that carry the talk's one sentence, about twenty minutes.
   On only with talk.html?cut=core (every other address: the talk as it was, nothing here changes it). A step is in it when
   talk.html gives it data-core="<minutes>"; data-core-scene="<scene>" on a step is the scene its simulation is shown in
   while this cut is on (when the module has it; else data-scene). The set goes to the deck as a sample, the way the knob
   design (talk/atlas.js) does it: KnotTalk.deck.sample({ has, cost }); the slide master's overrides still win, and a step
   that is data-off is shown if it carries data-core. If ?design=knob is on too, that design wins and this file does nothing.
   The HUD's cut group has a link to it (a diamond and its minutes, the sum of data-core, computed here); in the cut the
   link is the one that is lit, and the 20′ / 40′ buttons leave it for the same page without ?cut=core, on that step.
   A language switch keeps ?cut=core (talk/i18n.js only rewrites lang); the presenter view opened from the slide master gets
   it too (talk/master.js pageUrl), so its now / next and the clock total agree with the talk window. */
(function () {
  'use strict';
  var KT = window.KnotTalk, D = KT && KT.deck, root = document.documentElement;
  if (!D) return;
  var on = /[?&]cut=core(?:&|$)/.test(location.search) && !root.classList.contains('kt-knob');
  var link = document.querySelector('.hud .cut a.core'), group = document.querySelector('.hud .cut');
  var navs = [].slice.call(document.querySelectorAll('.nav[data-core]'));
  function mins(n) { return +n.dataset.core || 0; }
  var total = navs.reduce(function (t, n) { return t + mins(n); }, 0);

  /* this address with ?cut=core put in or taken out (the language, the step and the rest stay) */
  function address(core) {
    var qs = location.search.slice(1).split('&').filter(function (p) { return p && !/^cut(=|$)/.test(p); });
    if (core) qs.push('cut=core');
    return location.pathname + (qs.length ? '?' + qs.join('&') : '') + location.hash;
  }
  if (link) {
    link.querySelector('b').textContent = String(Math.round(total));
    link.title = KT.t('Core cut: only the steps that carry the talk’s one sentence · {m} min', { m: Math.round(total) });
    link.setAttribute('aria-label', KT.t('Core cut'));
    link.setAttribute('aria-current', on ? 'true' : 'false');
    link.classList.toggle('on', on);
    link.addEventListener('click', function (e) { e.preventDefault(); if (!on) location.href = address(true); });
  }
  if (!on) return;

  root.classList.add('kt-core');
  var smpLabels = { 20: D.total(20), 40: D.total(40) };        // the two buttons keep the totals of their own cuts
  navs.forEach(function (n) {
    var sc = n.dataset.coreScene, sim = n.closest('section[data-sim]'), def = sim && KT.sims && KT.sims[sim.dataset.sim];
    if (!sc) return;
    // a module that does not know the scene (yet) keeps the step's own
    if (def && def.scenes && !def.scenes.some(function (s) { return s.id === sc; })) return;
    n._scene = sc;
  });
  D.sample({ has: function (n) { return n.hasAttribute('data-core'); }, cost: mins, labels: smpLabels });

  var btns = [].slice.call(document.querySelectorAll('.hud .cut button'));
  function lit() { btns.forEach(function (b) { b.classList.remove('on'); }); }
  lit(); document.addEventListener('kt-cut', lit);
  if (group) group.addEventListener('click', function (e) {     // before deck.js sees it: leave the core cut for that cut
    var b = e.target.closest && e.target.closest('button[data-cut]');
    if (!b) return;
    e.stopPropagation(); e.preventDefault();
    try { localStorage.setItem('kt-cut', b.dataset.cut); } catch (x) {}
    location.href = address(false);
  }, true);
})();
