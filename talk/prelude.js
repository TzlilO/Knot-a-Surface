/* Knot-a-Surface talk — the opening ("How does a computer see in 3D?"): the 3D step's own controls.
   Its scene is the `manifold` module without chrome (mounted by deck.js from the section's data-mount); this file
   gives it the slider for the number of Gaussians and the count in the corner of the card, as the Hebrew deck has.
   Loaded before deck.js, so that a small screen can ask for fewer Gaussians before the module is mounted. */
(function () {
  'use strict';
  var sec = document.querySelector('.prelude .p-3d');
  if (!sec) return;
  var sl = sec.querySelector('input[type=range]'), num = sec.querySelector('.p-read');
  if (Math.min(window.innerWidth, window.innerHeight) < 560) {          // a phone: start lighter
    sl.value = 24000;
    try { var o = JSON.parse(sec.dataset.mount || '{}'); o.budget = 24000; sec.dataset.mount = JSON.stringify(o); } catch (e) { /* keep the page's own */ }
  }
  sec.addEventListener('kt-stats', function (e) {
    num.innerHTML = '<b>' + Math.round(e.detail.drawn).toLocaleString('en-US') + '</b>Gaussians on screen';
  });
  sl.addEventListener('input', function () { if (sec._sim) sec._sim.api.setBudget(+sl.value); });
  sl.addEventListener('pointerup', function () { sl.blur(); });      // the deck's keys must not move the slider afterwards
})();
