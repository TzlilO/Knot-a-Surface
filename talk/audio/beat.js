/* Knot-a-Surface talk — the music under chapter 07b ("Can't touch this"): an ORIGINAL loop, made in the page.
   Not a recording and not a cover: a small drum machine (kick, clap, hats), a rubber bass and a marimba line
   written for this talk. D major, 96 BPM, swung sixteenths; I–vi–IV–V twice, then IV–V–vi–V. Every fourth bar
   the marimba falls through the chord with the gaps between its notes shrinking, like a ball settling.
   12 bars of 4/4 at 96 BPM are exactly 30 s, the length the steps loop (data-audio-len="30"). The last beat is a
   breath: everything has died away before the bar line, so the seam is silence into the downbeat (first and last
   samples are 0) and stays clean although talk/deck.js restarts the track by hand when it ends.
   Synthesized sample by sample when this script runs (a few tens of ms, no Web Audio), so the URL is in place
   before talk/deck.js first asks for it; encoded as a 16-bit mono WAV at 22.05 kHz (1.3 MB) and registered as a
   blob: URL in KnotTalk.tracks.touch, unless talk/audio/tracks.js names a file for `touch`. Peaks at -12 dBFS,
   to sit under the chapter's own sound effects (talk/sims/touch.js); M or the ♪ button mutes both.
   The score is the tables below, the balance is GAIN. Under node, require() it for render() / wav(); nothing
   is registered there. */
(function (root) {
  'use strict';
  var RATE = 22050, BPM = 96, BARS = 12, STEP = 60 / BPM / 4, LEN = BARS * 16 * STEP,     // 30 s exactly
      SWING = 0.16, PEAK = Math.pow(10, -12 / 20);    // the off sixteenths land 16 % of a step late; peak -12 dBFS

  /* ================= the score: 16 sixteenths a bar ================= */
  var ROOT = [38, 35, 31, 33, 38, 35, 31, 33, 31, 33, 35, 33];      // bass roots (MIDI): D B G A | D B G A | G A B A
  var KICK = [1, 0, 0, 0, 0, 0, 0, .55, 0, 0, .85, 0, 0, 0, 0, 0];
  var CLAP = [0, 0, 0, 0, .9, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
  var HAT = [.5, 0, .8, 0, .5, 0, .8, .3, .5, 0, .8, 0, .5, .25, .8, .3];
  var BASS = [[0, 0, 1.6], [3, 12, .8], [6, 0, .9], [10, 0, .9], [11, 12, .8], [14, 7, .9]];   // [step, semitones up, length in steps]
  var MAL = [                                                        // [step, MIDI] per bar, chord tones only
    [[2, 74], [5, 78], [6, 81], [9, 86]],                            // D   up the chord
    [[2, 83], [5, 81], [6, 78], [9, 74]],                            // Bm  and down
    [[2, 79], [5, 83], [6, 86], [9, 83]],                            // G
    [[0, 88], [6, 85], [10, 81], [12, 76], [13, 73]],                // A   the settling ball: gaps of 6, 4, 2, 1
    [[2, 74], [5, 78], [6, 81], [9, 86]],                            // D
    [[2, 83], [5, 86], [6, 83], [9, 78], [11, 81]],                  // Bm
    [[2, 79], [5, 83], [6, 86], [9, 83]],                            // G
    [[0, 88], [6, 85], [10, 81], [12, 76], [13, 73], [14, 69]],      // A   settles one bounce lower
    [[0, 83], [3, 79], [6, 74], [10, 79], [11, 83]],                 // G   the middle: 3 + 3 + 4 + 1
    [[0, 85], [3, 81], [6, 76], [10, 81], [11, 85]],                 // A
    [[0, 86], [3, 83], [6, 78], [10, 83], [11, 86]],                 // Bm
    [[0, 88], [6, 85], [10, 81], [12, 76]]                           // A   last bounce, then the breath
  ];
  var GAIN = { kick: 0.9, clap: 0.45, hat: 0.28, open: 0.22, bass: 0.5, mallet: 0.2 };

  /* ================= voices: each renders one hit, normalised to peak 1 ================= */
  function rng(seed) {                         // mulberry32: the same noise on every load
    return function () {
      seed = (seed + 0x6D2B79F5) | 0;
      var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function mtof(m) { return 440 * Math.pow(2, (m - 69) / 12); }
  function biquad(hp, f, q, fs) {              // RBJ cookbook: high-pass, or band-pass with 0 dB peak
    var w = 2 * Math.PI * f / fs, c = Math.cos(w), al = Math.sin(w) / (2 * q), a0 = 1 + al,
        b0 = (hp ? (1 + c) / 2 : al) / a0, b1 = (hp ? -(1 + c) : 0) / a0, b2 = (hp ? (1 + c) / 2 : -al) / a0,
        a1 = -2 * c / a0, a2 = (1 - al) / a0, x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    return function (x) { var y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2; x2 = x1; x1 = x; y2 = y1; y1 = y; return y; };
  }
  function blep(t, dt) {                       // polyBLEP: rounds off the saw's jump so it does not alias
    if (t < dt) { t /= dt; return t + t - t * t - 1; }
    if (t > 1 - dt) { t = (t - 1) / dt; return t * t + t + t + 1; }
    return 0;
  }
  function finish(x, fs) {                     // 10 ms cosine taper where the hit is cut off, then peak 1
    var n = x.length, m = Math.min(n, Math.round(0.01 * fs)), p = 0, i;
    for (i = 0; i < m; i++) x[n - 1 - i] *= 0.5 - 0.5 * Math.cos(Math.PI * i / m);
    for (i = 0; i < n; i++) p = Math.max(p, Math.abs(x[i]));
    if (p > 0) for (i = 0; i < n; i++) x[i] /= p;
    return x;
  }
  function kick(fs) {                          // a sine falling from 158 to 48 Hz, softly saturated, short
    var n = Math.round(0.36 * fs), x = new Float32Array(n), ph = 0, e = 1, p = 1,
        de = Math.exp(-1 / (0.11 * fs)), dp = Math.exp(-1 / (0.03 * fs)), i;
    for (i = 0; i < n; i++) {
      x[i] = Math.tanh(1.6 * Math.sin(2 * Math.PI * ph)) * e;
      ph += (48 + 110 * p) / fs; e *= de; p *= dp;
    }
    return finish(x, fs);
  }
  function clap(fs, seed) {                    // band-passed noise: three quick slaps, then a short room
    var n = Math.round(0.3 * fs), x = new Float32Array(n), r = rng(seed),
        bp = biquad(false, 1200, 1.1, fs), hp = biquad(true, 500, 0.7, fs), i, t, k;
    for (i = 0; i < n; i++) {
      t = i / fs; k = Math.floor(t / 0.0083);
      x[i] = hp(bp(2 * r() - 1)) * Math.min(1, t / 0.0005) *
             (k < 3 ? Math.exp(-(t - k * 0.0083) / 0.0025) : 0.7 * Math.exp(-(t - 0.0249) / 0.07));
    }
    return finish(x, fs);
  }
  function hat(fs, seed, decay) {              // noise above 7 kHz
    var n = Math.round(decay * 7 * fs), x = new Float32Array(n), r = rng(seed),
        h1 = biquad(true, 7000, 0.7, fs), h2 = biquad(true, 7000, 0.7, fs), e = 1, d = Math.exp(-1 / (decay * fs)), i;
    for (i = 0; i < n; i++) { x[i] = h2(h1(2 * r() - 1)) * Math.min(1, i / (0.0004 * fs)) * e; e *= d; }
    return finish(x, fs);
  }
  function bass(fs, midi, len) {               // saw + sine through a resonant low-pass that closes fast: a rubber pluck
    var f0 = mtof(midi), on = Math.round(len * fs), rel = Math.round(0.025 * fs), n = on + rel, x = new Float32Array(n),
        ph = 0, lo = 0, bd = 0, b = 1, c = 1, e = 1, db = Math.exp(-1 / (0.02 * fs)), dc = Math.exp(-1 / (0.05 * fs)),
        de = Math.exp(-1 / (0.5 * fs)), i, dt, s, g, hi;
    for (i = 0; i < n; i++) {
      dt = f0 * (1 + 0.0578 * b) / fs;         // starts a semitone sharp and drops into tune: the bounce
      s = 0.7 * (2 * ph - 1 - blep(ph, dt)) + 0.6 * Math.sin(2 * Math.PI * ph);
      g = 2 * Math.PI * (200 + 1300 * c) / fs;
      lo += g * bd; hi = s - lo - 0.6 * bd; bd += g * hi;     // Chamberlin state-variable filter, low-pass out
      x[i] = lo * Math.min(1, i / (0.004 * fs)) * e * (i < on ? 1 : 1 - (i - on) / rel);
      ph += dt; if (ph >= 1) ph -= 1;
      b *= db; c *= dc; e *= de;
    }
    return finish(x, fs);
  }
  function mallet(fs, midi) {                  // two sines, the second two octaves up and quick: a soft marimba
    var w = 2 * Math.PI * mtof(midi) / fs, n = Math.round(0.7 * fs), x = new Float32Array(n), e1 = 1,
        e2 = 4 * w < 0.9 * Math.PI ? 0.3 : 0, d1 = Math.exp(-1 / (0.16 * fs)), d2 = Math.exp(-1 / (0.035 * fs)), i;
    for (i = 0; i < n; i++) {
      x[i] = Math.min(1, i / (0.0015 * fs)) * (e1 * Math.sin(w * i) + e2 * Math.sin(4 * w * i));
      e1 *= d1; e2 *= d2;
    }
    return finish(x, fs);
  }

  /* ================= the loop ================= */
  function score(fs, gain) {                   // the raw mix; a part whose gain is missing is left out
    var N = Math.round(LEN * fs), mix = new Float32Array(N), r = rng(96), memo = {}, G = gain || GAIN, b,
        K = kick(fs), C = [clap(fs, 1), clap(fs, 2), clap(fs, 3)], O = hat(fs, 8, 0.09),
        H = [hat(fs, 4, 0.022), hat(fs, 5, 0.022), hat(fs, 6, 0.022), hat(fs, 7, 0.022)];
    function add(x, at, g) {
      if (!g) return;
      var o = Math.round(at * fs), n = Math.min(x.length, N - o), i;
      for (i = 0; i < n; i++) mix[o + i] += g * x[i];
    }
    function at(bar, step) { return (bar * 16 + step + (step % 2 ? SWING : 0)) * STEP; }
    function once(key, make) { return memo[key] || (memo[key] = make()); }
    function bar(b) {
      var last = b === BARS - 1, fill = b % 4 === 3, s, t, v;      // bars 3, 7, 11 end a phrase; 11 ends the loop
      for (s = 0; s < 16 && !(last && s > 12); s++) {              // the last bar stops after beat 4: the breath
        t = at(b, s); v = 0.85 + 0.3 * r();
        if (KICK[s] && !(fill && s === 7)) add(K, t, G.kick * KICK[s]);
        if (CLAP[s]) add(C[(b + s) % 3], t, G.clap * CLAP[s]);
        else if (fill && !last && s === 15) add(C[b % 3], t, G.clap * 0.35);   // a soft pickup into the next phrase
        if (s === 14 && b % 4 === 1) add(O, t, G.open);                        // an open hat ends bars 1, 5, 9
        else if (HAT[s]) add(H[r() * 4 | 0], t, G.hat * HAT[s] * v);
      }
      BASS.forEach(function (n) {
        var m = ROOT[b] + n[1];
        if (G.bass && !(last && n[0] > 12)) add(once('b' + m + ':' + n[2], function () { return bass(fs, m, n[2] * STEP); }), at(b, n[0]), G.bass);
      });
      MAL[b].forEach(function (n) {
        var v = 0.85 + 0.15 * r();
        if (G.mallet) add(once('m' + n[1], function () { return mallet(fs, n[1]); }), at(b, n[0]), G.mallet * v);
      });
    }
    for (b = 0; b < BARS; b++) bar(b);
    return mix;
  }
  function master(x, fs) {                     // peak to -12 dBFS; 1 ms in and 20 ms out (both ends are silence already)
    var n = x.length, p = 0, a = Math.round(0.001 * fs), z = Math.round(0.02 * fs), k, i;
    for (i = 0; i < n; i++) p = Math.max(p, Math.abs(x[i]));
    k = p > 0 ? PEAK / p : 0;
    for (i = 0; i < n; i++) x[i] *= k;
    for (i = 0; i < a; i++) x[i] *= i / a;
    for (i = 0; i < z; i++) x[n - 1 - i] *= 0.5 - 0.5 * Math.cos(Math.PI * i / z);
    return x;
  }
  function render(fs) { fs = fs || RATE; return master(score(fs, GAIN), fs); }
  function wav(x, fs) {                        // 16-bit mono PCM
    var n = x.length, buf = new ArrayBuffer(44 + 2 * n), d = new DataView(buf), i;
    function str(o, s) { for (var k = 0; k < s.length; k++) d.setUint8(o + k, s.charCodeAt(k)); }
    str(0, 'RIFF'); d.setUint32(4, 36 + 2 * n, true); str(8, 'WAVE');
    str(12, 'fmt '); d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
    d.setUint32(24, fs, true); d.setUint32(28, 2 * fs, true); d.setUint16(32, 2, true); d.setUint16(34, 16, true);
    str(36, 'data'); d.setUint32(40, 2 * n, true);
    for (i = 0; i < n; i++) d.setInt16(44 + 2 * i, Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767))), true);
    return buf;
  }

  var api = { render: render, wav: wav, score: score, master: master, GAIN: GAIN, rate: RATE, bpm: BPM, bars: BARS, seconds: LEN };
  if (typeof module !== 'undefined' && module.exports) { module.exports = api; return; }
  try {
    var KT = root.KnotTalk = root.KnotTalk || {}, T = KT.tracks = KT.tracks || {};
    if (!T.touch) T.touch = URL.createObjectURL(new Blob([wav(render(RATE), RATE)], { type: 'audio/wav' }));
  } catch (e) { /* no Blob or object URLs here: the steps stay silent, as they would without a track */ }
})(typeof window !== 'undefined' ? window : globalThis);
