/* Knot-a-Surface talk — music for steps that ask for it (<article data-audio="name">, see talk/deck.js).
   No recording is kept in this repository. A name maps to a file (or any URL the browser can play); a name
   that is not listed stays silent and nothing is requested. talk/audio/beat.js loads after this file and fills
   `touch` (chapter 07b) with an original loop it synthesizes in the page; a file named here for `touch` wins.

   KnotTalk.tracks = { name: 'talk/audio/your-own-file.mp3' };                                            */
window.KnotTalk = window.KnotTalk || {};
KnotTalk.tracks = {};

/* The presenter's own recording for chapter 09, kept OUT of the repository and of every published copy:
   talk/audio/local/ is in .gitignore. Put the file at talk/audio/local/touch.mp3 on the machine that presents and
   open talk.html?song=on once (remembered in localStorage `kt-song`; ?song=off goes back to the original loop).
   Off by default, so a copy of the page without the file asks for nothing. With it on and the file missing the
   chapter has no music (its sound effects stay). The steps' credit line then names the recording. */
(function () {
  var on = false;
  try {
    var q = /[?&]song=(on|off)\b/.exec(location.search);
    if (q) localStorage.setItem('kt-song', q[1]);
    on = localStorage.getItem('kt-song') === 'on';
  } catch (e) {}
  if (!on) return;
  KnotTalk.tracks.touch = 'talk/audio/local/touch.mp3';
  var t = KnotTalk.t || function (s) { return s; };
  [].forEach.call(document.querySelectorAll('[data-audio="touch"] .credit'), function (c) {
    c.textContent = t('♪ MC Hammer, “U Can’t Touch This” (1990) · the presenter’s own copy, not part of this page');
    c.setAttribute('data-i18n', 'off');
  });
})();
