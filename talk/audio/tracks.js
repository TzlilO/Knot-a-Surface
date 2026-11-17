/* Knot-a-Surface talk — music for steps that ask for it (<article data-audio="name">, see talk/deck.js).
   No recording is kept in this repository. A name maps to a file (or any URL the browser can play); a name
   that is not listed stays silent and nothing is requested. talk/audio/beat.js loads after this file and fills
   `touch` (chapter 07b) with an original loop it synthesizes in the page; a file named here for `touch` wins.

   KnotTalk.tracks = { name: 'talk/audio/your-own-file.mp3' };                                            */
window.KnotTalk = window.KnotTalk || {};
KnotTalk.tracks = {};
