/* Knot-a-Surface talk — module `touch`: "Can't touch this".
   One bowl, stored two ways, in one world, and a solid ball (SI units: 1 kg, radius 0.25 m, g = 9.81):
     a list   Gaussians in no particular order, as an unstructured 3DGS keeps them. EMULATED here, not
              trained, exactly as in the `structure` module: the same surface sampled at jittered places,
              nudged off the surface, each turned and resized at random, then shuffled. Each one is a 3D
              Gaussian: two axes along the surface, a third for its thickness (25 to 40 % of the other two),
              the frame tilted off the surface by up to 46 degrees, an opacity of 0.34 to 0.64. It is drawn
              the way 3DGS draws one: the ellipsoid carried to the screen through the camera's Jacobian, its
              opacity falling off as exp(-m^2 / 2), blended back to front (sorted every frame).
     ours     one cubic B-spline patch (13 x 13 control points); its Gaussians are derived from it.
   Six scenes:
     ghost    the ball is dropped on the list. There is no boundary to hit: with no rule it passes through.
              Then, without a cut, the list's half of `moment` (its drop, camera, slow motion, numbers and
              timeline, the same code: `run.close` makes `model()` answer 'moment'), then the pass again
              (`GHOST`). The lab page adds a third drop, the rule "every Gaussian is a loose particle": the
              ball knocks away the ones it touches, and since the Gaussians of a list overlap where they lie,
              as solid particles they push each other apart and the bowl comes loose. The talk never shows it
              in `ghost` (`flight` scatters the list that way).
     moment   the same drop on both bowls, close up and slowed down. When does the ball touch the list? It
              admits many answers, each measured on the fall: the ball's surface reaches the 3, 2 or 1 sigma
              shell of the nearest Gaussian (the exact distance from its centre to the ellipsoid), the summed
              density at the ball's lowest point crosses 0.1, 0.3 or 0.5, the ball reaches the nearest mean.
              A timeline gives their spread in ms of simulated time. Then the spline: one answer, the root of
              |S(u,v) - c| = r with S(u,v) the closest point, by bisection on the parabola, with its residual.
              SCRIPTED: the camera path and the slow motion (simulated time runs 1/SLOW as fast within the last
              centimetres of the surface, see `slowmo`), nothing else. The numbers are for this emulated list.
     solid    the same drop on the spline, an ideal rigid surface. Contact is computed ON THE SPLINE:
              closest point S(u,v) by Gauss-Newton steps in (u,v), normal S_u x S_v, the velocity
              reflected about the normal with a restitution coefficient.
     brittle  toughness: when the impulse the impact asks of the surface exceeds it, the surface breaks
              into pieces of the (u,v) domain; each piece is a rigid body carrying its Gaussians.
     elastic  the control points sit on damped springs (a membrane over the control net); the contact
              force is spread to them by the basis weights; the Gaussians are re-derived every frame.
     flight   two headers, then the lists. The cast are real objects stored as spline charts: the owl (DTU scan 122,
              on its books) and the rabbit (DTU scan 55) in the talk's clear glass, both fitted offline to the
              dataset (talk/dtu-data.js, talk/dtu/scan55.js; never trained). The ball comes down on the owl's head;
              the owl nods it to the rabbit, which jumps and heads it on into the list bowl, where list versions of
              the owl and the rabbit stand (EMULATED from the same charts, as the bowl's list is). There nothing
              scatters: the Gaussians near the ball's path drift aside like floating balloons and the ball falls
              through. COMPUTED: where and when the ball meets each figure (closest point by Gauss-Newton in (u, v)
              in a window of the chart around the head, normal S_u x S_v, the root of |S(u, v) - c| = r inside the
              sub-step). SCRIPTED: the owl's nod, the rabbit's jump, squash and nod, and where each header sends the
              ball (see `header`). INVENTED: the balloon drift (see `balloons`): a list defines no force.
              Without the scans' data (a page that does not load them) `flight` is the older finale: a hard bounce on
              the elastic bowl throws the ball across to the list, whose Gaussians it scatters as loose particles
              (only the direction of the launch is chosen, see `launch`).
   All motion runs on a fixed sub-step (H; H / 8 in `moment`), so a scene plays the same at any frame rate.
   Every number on screen is read off the simulation.
   Sound: synthesised in the page with Web Audio (oscillators, a seeded noise buffer through filters, envelopes,
   and short grain buffers written here sample by sample); nothing is loaded or sampled. Every sound is an event
   the simulation computes, scaled by what it measures: a knock per bounce on the spline (its impulse), a low roll
   while the ball rolls (its speed), a hiss while it sinks through a list with no rule (how many Gaussians it
   overlaps), a scatter and a rattle when Gaussians are knocked loose (the momentum they took, how many came
   loose), a pluck and a wobble on the elastic surface (pitch from the spring stiffness, loudness and wobble from
   the control points' motion), a crack and shards when it breaks (impulse over toughness, the pieces' sizes and
   speeds), a whoosh at the launch (its speed), a click for each candidate moment of contact, when it happened. All of it
   sits above the bass of the chapter's music (talk/audio/beat.js), so it carries over it. Nothing plays before a key
   or pointer press (browsers forbid it), while KnotTalk.muted is true, or while the module is paused.
   Words: every string a viewer reads goes through T = KnotTalk.t (talk/i18n.js), keyed by its English, one statement
   (numbers and all) as one template; a line of the numbers box is statements joined by " · ". In Hebrew the stage stays
   a left-to-right island: the timeline, the address pin and the boxes are where they are in English, T gives each
   string its own direction, and the sentences under the timeline are marked right-to-left as a block.
   Tool: WebGL (three.js), Web Audio. Standalone: talk/sims/touch.html.
   Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js; `flight` also on dtu-data.js, dtu/scan55.js and scenes.js
   (KnotTalk.materials.glass), loaded before this file (without the first two it falls back, without scenes.js the
   rabbit keeps its photograph's colours). */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  var T = KT.t, RTL = !!KT.rtl;                // the page's language: T(english, vars) -> that string in it (the English itself when there is no translation)

  var NA = 88, N = NA * NA;                    // Gaussians per bowl: an NA x NA lattice in (u, v)
  var NC = 13, SPANS = NC - 3;                 // control points per side: one cubic patch of 10 x 10 spans
  var RB = 1.6, DEPTH = 0.95, SQ = 0.9;        // the bowl: half-width, depth, roundness of the rim (0 square .. 1 disc)
  var X0 = 6;                                  // the spline bowl sits at x = +X0, the list at x = -X0
  var GRAV = 9.81, BR = 0.25, BM = 1;          // gravity, ball radius, ball mass
  var H = 1 / 480;                             // physics sub-step, seconds
  var VREST = 0.3, MU = 0.08, CROLL = 0.35;    // rigid: resting speed, rolling resistance (Coulomb + viscous)
  var MS = 4;                                  // brittle: mass of the whole surface, shared by area
  var MC = 0.02, MDAMP = 0.05, KA = 0.04;      // elastic: mass and damping per control point, anchor / membrane stiffness
  var KC = 6000, CC = 6, CT = 1.2;             // elastic: contact stiffness, damping along the normal, friction along the surface
  var MP = BM / 1500, EP = 0.9, EPP = 0.9;     // list as particles: mass of one Gaussian, restitution ball-Gaussian, Gaussian-Gaussian
  var W2 = 4000;                               // ... and how hard two overlapping Gaussians push apart: stiffness / mass, 1/s^2
  var DRAG = 0.35;                             // air drag on a loose Gaussian, 1/s
  var TILT = 0.8;                              // the list: largest tilt of a Gaussian off the surface, radians
  var HM = H / 8, SLOW = 20;                   // moment: its sub-step, and how much slower time runs near contact (scripted)
  var FSPR = 280;                              // sound: the springs' pitch at 160 N/m, Hz (it goes as the square root of the stiffness)
  var LIGHT = [0.35, 0.86, 0.36];
  /* flight with the scans: where the cast stands (the base of each figure: the bottom of the owl's books, the rabbit's
     feet), how big (FS: metres per unit of the owl's scan, 80 mm; the rabbit keeps its true size beside it, 71 mm a
     unit), and the script's times */
  var FS = 0.5;                                // the owl is 1.75 m tall and its head about as wide as the ball
  var OWL_AT = [5.3, -0.75, -0.15], RAB_AT = [1.35, -0.75, 0.35], RFACE = 0.32;   // RFACE: the rabbit faces this far (rad) from the eye towards the owl
  var T1 = 1.0, T2 = 1.3;                      // SCRIPTED: how long the ball flies after each header
  var HJ = 0.5, CROUCH = 0.2, LAND = 0.24;     // SCRIPTED: the rabbit's jump: its height (m), its crouch and its landing (s)
  /* INVENTED, the balloon rule (`balloons`): reach beyond the ball (m), push (m/s^2), drag along with the ball (1/s),
     the spring home (1/s^2), its damping (1/s), lift while displaced (1/s^2), sway (1/s^2), top speed (m/s) */
  var BREACH = 0.14, BKP = 22, BKD = 2.4, BK = 3.5, BC = 1.5, BLIFT = 0.9, BSWAY = 0.7, BVMAX = 1.5;

  var SCENES = {
    ghost:   { cap: T('<b>a ball meets a list of Gaussians</b> · a density, not a boundary'), model: 'ghost', side: -1 },
    moment:  { cap: T('<b>the moment of contact</b> · close up, slowed down'), model: 'moment', side: -1 },
    solid:   { cap: T('<b>the same drop on the spline</b> · closest point, normal, reflection'), model: 'solid', side: 1 },
    brittle: { cap: T('<b>toughness</b> · the surface breaks when the impact asks too much of it'), model: 'brittle', side: 1 },
    elastic: { cap: T('<b>an elastic surface</b> · the control points sit on springs'), model: 'elastic', side: 1 },
    flight:  { cap: T('<b>out of bounds</b> · from the spline bowl to the list of Gaussians'), model: 'flight', side: 1,
               capF: T('<b>out of bounds</b> · two headers, then the lists in the bowl') }
  };
  /* where the ball is let go: x and z from the bowl's centre, height y, and the speed it is thrown down with */
  var DROPS = {
    ghost:   [[0.25, 2.6, -0.5, 0], [0.25, 2.6, -0.5, 0]],
    moment:  [[0.1, 0.68, -0.1, 0]],                              // the same place over both bowls: list, then spline
    solid:   [[-0.35, 2.2, -0.6, 0], [0.5, 2.0, -0.45, 0], [-0.7, 2.4, 0.1, 0]],
    brittle: [[-0.2, 0.62, -0.45, 0], [-0.2, 3.3, -0.45, 0]],
    elastic: [[-0.1, 2.3, -0.15, 0], [0.4, 1.9, -0.3, 0], [-0.3, 2.7, 0.2, 0]],
    flight:  [[0.8, 3.4, 0, 12]]
  };
  /* moment: the definitions of "contact" a list admits, in the order the timeline names them */
  var CANDS = [{ lab: '3σ', k: 3 }, { lab: '2σ', k: 2 }, { lab: '1σ', k: 1 },
               { lab: 'ρ.1', lev: 0.1 }, { lab: 'ρ.3', lev: 0.3 }, { lab: 'ρ.5', lev: 0.5 }, { lab: 'μ' }];

  /* ---------- the list's Gaussians, drawn as 3DGS draws them ----------
     The covariance of the three axes is carried to the screen through the Jacobian of the projection at the
     centre (an ellipse, plus a 0.3 px^2 low-pass), and the quad over it is shaded out to 3 sigma with opacity
     alpha exp(-m^2 / 2). Alpha-blended, no depth written, drawn after the solid things (sorted on the CPU). */
  var FUZZ_V = [
    'attribute vec3 iC; attribute vec3 iU; attribute vec3 iV; attribute vec3 iW; attribute vec3 iCol; attribute float iA;',
    'uniform vec2 uView; uniform vec3 uLight;',
    'varying vec2 vQ; varying vec3 vCol; varying float vA;',
    'void main(){',
    '  vec4 e = viewMatrix * vec4(iC, 1.0); float z = -e.z;',
    '  if (z < 0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }',
    '  vec3 a = (viewMatrix * vec4(iU, 0.0)).xyz, b = (viewMatrix * vec4(iV, 0.0)).xyz, c = (viewMatrix * vec4(iW, 0.0)).xyz;',
    '  float fx = projectionMatrix[0][0] * 0.5 * uView.x, fy = projectionMatrix[1][1] * 0.5 * uView.y;',
    '  vec3 j0 = vec3(fx / z, 0.0, fx * e.x / (z * z)), j1 = vec3(0.0, fy / z, fy * e.y / (z * z));',
    '  vec2 pa = vec2(dot(j0, a), dot(j1, a)), pb = vec2(dot(j0, b), dot(j1, b)), pc = vec2(dot(j0, c), dot(j1, c));',
    '  float sxx = pa.x * pa.x + pb.x * pb.x + pc.x * pc.x + 0.3, syy = pa.y * pa.y + pb.y * pb.y + pc.y * pc.y + 0.3;',
    '  float sxy = pa.x * pa.y + pb.x * pb.y + pc.x * pc.y;',
    '  float mid = 0.5 * (sxx + syy), dis = sqrt(max(mid * mid - (sxx * syy - sxy * sxy), 0.0));',
    '  float l1 = mid + dis, l2 = max(mid - dis, 0.3);',
    '  vec2 v1 = abs(sxy) > 1e-6 ? normalize(vec2(sxy, l1 - sxx)) : (sxx >= syy ? vec2(1.0, 0.0) : vec2(0.0, 1.0));',
    '  vec2 off = 3.0 * (position.x * sqrt(l1) * v1 + position.y * sqrt(l2) * vec2(-v1.y, v1.x));',
    '  vec4 p = projectionMatrix * e; p.xy += off / (0.5 * uView) * p.w; gl_Position = p;',
    '  vQ = 3.0 * position.xy;',
    '  vCol = iCol * (0.74 + 0.46 * abs(dot(normalize(iW), uLight))); vA = iA;',      // a turned Gaussian catches the light differently
    '}'].join('\n');
  var FUZZ_F = [
    'varying vec2 vQ; varying vec3 vCol; varying float vA;',
    'void main(){',
    '  float q = dot(vQ, vQ); if (q > 9.0) discard;',
    '  float a = vA * exp(-0.5 * q); if (a < 0.004) discard;',
    '  gl_FragColor = vec4(vCol, a);',
    '}'].join('\n');

  /* ---------- sound: one engine for every mount of this module ----------
     Web Audio only: oscillators, a seeded noise buffer through biquad filters, gain envelopes, and grain buffers
     written here sample by sample (damped sines: the ticks of a rattle, the ring of a shard). Nothing is loaded.
     The context is made on the first key or pointer press anywhere and resumed on later ones; nothing starts
     while KnotTalk.muted is true (read at play time). One-shot voices: at most VOICES at once; a newcomer
     quieter than every voice still sounding is dropped, a louder one replaces the quietest; two of one kind start
     at least 30 ms apart (clicks 15 ms), so frames that come in a burst cannot pile them into one loud hit. Sustained voices
     (roll, hiss, wobble): one each per mount, renewed every frame, and if no frame renews them (the module is
     paused off screen, the tab is hidden) they fade out within HOLD seconds and stop by themselves. */
  var SFX = (function () {
    var ctx = null, bus = null, noise = null, dead = false, armed = false, live = [], cnt = {}, next = {}, rr = KT.rng(4242);
    var VOICES = 8, HOLD = 0.22, LAG = 0.012;
    ['knock', 'roll', 'hiss', 'scatter', 'rattle', 'wobble', 'crack', 'shards', 'whoosh', 'tick', 'glass', 'puff'].forEach(function (k) { cnt[k] = 0; });
    function wake() {
      if (dead) return;
      try {
        if (!ctx) {
          var AC = window.AudioContext || window.webkitAudioContext;
          if (!AC) { dead = true; return; }
          ctx = new AC(); build();
        }
        if (ctx.state !== 'running' && ctx.resume) { var p = ctx.resume(); if (p && p.catch) p.catch(function () {}); }
      } catch (e) { dead = true; ctx = null; }
    }
    function build() {
      var sr = ctx.sampleRate, n = Math.floor(sr * 2.5), rn = KT.rng(99), x, i;
      noise = ctx.createBuffer(1, n, sr); x = noise.getChannelData(0);
      for (i = 0; i < n; i++) x[i] = rn() * 2 - 1;
      bus = ctx.createGain(); bus.gain.value = 0.22;                          // a room, not a game: a hard knock peaks near -16 dBFS, the crack -12
      var lp = biq('lowpass', 5000, 0.5), cp = ctx.createDynamicsCompressor();  // no harsh highs; a pile-up is held down
      cp.threshold.value = -18; cp.knee.value = 8; cp.ratio.value = 4; cp.attack.value = 0.003; cp.release.value = 0.25;
      bus.connect(lp); lp.connect(cp); cp.connect(ctx.destination);
    }
    function arm() {
      if (armed) return; armed = true;
      ['pointerdown', 'pointerup', 'touchend', 'click', 'keydown'].forEach(function (k) { window.addEventListener(k, wake, { capture: true, passive: true }); });
      try { if (navigator.userActivation && navigator.userActivation.hasBeenActive) wake(); } catch (e) {}   // a key or click came before this mount
    }
    function ok() { var K = window.KnotTalk; return !!ctx && ctx.state === 'running' && !(K && K.muted); }

    /* ---- one-shot voices ---- */
    function voice(kind, peak, tau, pan, delay) {
      var t0 = ctx.currentTime + LAG + (delay || 0), t = Math.max(t0, next[kind] || 0), i, w = -1, lo = 1e9, l, v, node;
      if (t > t0 + 0.25) return null;          // frames that came in a burst (a stall): one kind never stacks on itself, nor lags far
      next[kind] = t + (kind === 'tick' ? 0.015 : 0.03);
      for (i = live.length - 1; i >= 0; i--) if (live[i].end <= t) live.splice(i, 1);
      if (live.length >= VOICES) {
        for (i = 0; i < live.length; i++) { l = live[i].peak * Math.exp(-Math.max(0, t - live[i].t) / live[i].tau); if (l < lo) { lo = l; w = i; } }
        if (lo >= peak) return null;
        cut(live[w], t); live.splice(w, 1);
      }
      v = { t: t, peak: peak, tau: tau, end: t + 0.05, srcs: [], left: 0, out: ctx.createGain() };
      node = v.out;
      if (ctx.createStereoPanner) { var p = ctx.createStereoPanner(); p.pan.value = pan || 0; v.out.connect(p); node = p; }
      node.connect(bus); live.push(v); cnt[kind]++;
      return v;
    }
    function cut(v, t) { try { v.out.gain.setTargetAtTime(0, t, 0.008); v.srcs.forEach(function (s) { s.stop(t + 0.06); }); } catch (e) {} }
    function track(v, s, end) {
      v.srcs.push(s); v.left++; v.end = Math.max(v.end, end);
      s.onended = function () { if (--v.left <= 0) try { v.out.disconnect(); } catch (e) {} };
    }
    function env(p, t, a, att, tau) { p.setValueAtTime(0, t); p.linearRampToValueAtTime(a, t + att); p.setTargetAtTime(0, t + att, tau); }
    function osc(v, f, t, len) { var o = ctx.createOscillator(); o.frequency.setValueAtTime(f, t); o.start(t); o.stop(t + len); track(v, o, t + len); return o; }
    function hiss(v, t, len) { var s = ctx.createBufferSource(); s.buffer = noise; s.loop = true; s.start(t, rr() * 2); s.stop(t + len); track(v, s, t + len); return s; }
    function play(v, b, t) { var s = ctx.createBufferSource(); s.buffer = b; s.start(t); track(v, s, t + b.duration); return s; }
    function biq(type, f, q) { var b = ctx.createBiquadFilter(); b.type = type; b.frequency.value = f; b.Q.value = q; return b; }
    function amp(t, a, att, tau) { var g = ctx.createGain(); if (t != null) env(g.gain, t, a, att, tau); else g.gain.value = a; return g; }
    function chain() { for (var i = 0; i < arguments.length - 1; i++) arguments[i].connect(arguments[i + 1]); }
    /* damped sines [start s, Hz, decay s, amplitude] summed into one buffer: y_k = d^k sin(w k) by its recurrence */
    function grains(L, dur) {
      var sr = ctx.sampleRate, n = Math.ceil(dur * sr), b = ctx.createBuffer(1, n, sr), x = b.getChannelData(0), g, k, i0, m, d, c, a, y0, y1, y2, w;
      for (g = 0; g < L.length; g++) {
        i0 = Math.floor(L[g][0] * sr); if (i0 >= n - 2) continue;
        w = 2 * Math.PI * L[g][1] / sr; d = Math.exp(-1 / (L[g][2] * sr)); m = Math.min(n - i0, Math.ceil(L[g][2] * sr * 8));
        c = 2 * d * Math.cos(w); a = L[g][3]; y1 = 0; y0 = d * Math.sin(w);
        for (k = 1; k < m; k++) { x[i0 + k] += a * y0; y2 = c * y0 - d * d * y1; y1 = y0; y0 = y2; }
      }
      return b;
    }

    /* a bounce on the spline: the bowl's first modes, struck. J (N·s) sets loudness, brightness and, a little, pitch;
       tone shifts it with the place on the bowl; st > 1 is slow motion (lower, longer). The fundamental sits above the
       bass line of the chapter's music (talk/audio/beat.js), and the upper modes and the click carry over its drums. */
    function knock(J, tone, pan, st) {
      if (!ok()) return;
      var h = Math.min(J / 12, 1), a = 0.06 + 0.3 * Math.pow(h, 0.85), w = Math.pow(st || 1, 0.2), br = 0.45 + 0.55 * h;
      var v = voice('knock', a, 0.1 * w, pan); if (!v) return;
      var t = v.t, f = 230 * Math.pow(Math.max(J, 0.3) / 4, 0.1) * (tone || 1) / w;
      [[1, 1, 0.1], [2.32, 0.62 * br, 0.055], [4.1, 0.36 * br, 0.032], [6.3, 0.18 * br, 0.018]].forEach(function (md, i) {
        var o = osc(v, f * md[0], t, md[2] * w * 7 + 0.02);
        if (!i) { o.frequency.setValueAtTime(f * 1.2, t); o.frequency.exponentialRampToValueAtTime(f, t + 0.025 * w); }
        chain(o, amp(t, a * md[1], 0.0015, md[2] * w), v.out);
      });
      chain(hiss(v, t, 0.05 * w), biq('bandpass', 1800 + 1400 * br, 0.9), amp(t, a * 0.8 * br, 0.0008, 0.006 * w), v.out);   // the contact itself
    }
    /* the surface breaks: r = impulse asked / toughness; np pieces */
    function crack(r, np, pan) {
      if (!ok()) return;
      var a = KT.clamp(0.3 + 0.22 * Math.log2(Math.max(r, 1)), 0.3, 0.68), v = voice('crack', a, 0.08, pan); if (!v) return;
      var t = v.t, o = osc(v, 100, t, 0.5), L = [], m = KT.clamp(Math.round(4 + np / 5), 4, 14), i, x;
      chain(hiss(v, t, 0.25), biq('bandpass', 2300, 0.7), amp(t, a * 0.75, 0.0006, 0.024), v.out);   // the break
      chain(hiss(v, t, 0.4), biq('lowpass', 420, 0.7), amp(t, a * 0.6, 0.002, 0.06), v.out);           // its body
      o.frequency.exponentialRampToValueAtTime(48, t + 0.12); chain(o, amp(t, a * 0.55, 0.002, 0.08), v.out);
      for (i = 0; i < m; i++) { x = i / m; L.push([0.003 + 0.14 * Math.pow(x, 1.7) + 0.004 * rr(), 1200 + 1800 * rr(), 0.0003 + 0.0006 * rr(), a * (0.9 - 0.5 * x) * (0.6 + 0.4 * rr())]); }
      chain(play(v, grains(L, 0.2), t), v.out);                                                       // the crack running through
    }
    /* its pieces flying apart: P = [{ n Gaussians, sp m/s, rho m from the impact }]; the fastest ring, small ones higher,
       those far from the impact later (the crack reaches them later) */
    function shards(P, r, pan) {
      if (!ok() || !P.length) return;
      var A = KT.clamp(0.1 + 0.08 * Math.log2(Math.max(r, 1)), 0.1, 0.24), v = voice('shards', A, 0.2, pan); if (!v) return;
      var vmax = 0, rmax = 0, nref = 0, L = [];
      P.forEach(function (q) { vmax = Math.max(vmax, q.sp); rmax = Math.max(rmax, q.rho); nref += q.n / P.length; });
      P.slice().sort(function (p, q) { return q.sp - p.sp; }).slice(0, KT.clamp(Math.round(P.length / 4), 4, 12)).forEach(function (q) {
        var f = KT.clamp(1100 * Math.pow(nref / q.n, 0.35), 650, 2600), d = KT.clamp(0.035 * Math.pow(q.n / nref, 0.2), 0.02, 0.06);
        var a = A * Math.pow(q.sp / (vmax || 1), 0.8), t0 = 0.03 + 0.38 * q.rho / (rmax || 1) + 0.02 * rr();
        L.push([t0, f, d, a]); L.push([t0, f * 2.76, d * 0.5, a * 0.35]);
      });
      chain(play(v, grains(L, 0.75), v.t), v.out);
    }
    /* the ball meets a list of loose particles: j = momentum the Gaussians took (N·s), n = how many it hit */
    function scatter(j, n, pan) {
      if (!ok()) return;
      var h = KT.clamp(j / 1.5, 0, 1), A = 0.12 + 0.2 * h, v = voice('scatter', A, 0.12, pan); if (!v) return;
      var t = v.t, L = [], m = KT.clamp(Math.round(10 + 6 * Math.log2(1 + n)), 10, 56), i;
      for (i = 0; i < m; i++) L.push([0.4 * Math.pow(rr(), 2.4), 800 + 1600 * rr(), 0.0004 + 0.001 * rr(), A * (0.3 + 0.7 * rr())]);
      chain(play(v, grains(L, 0.45), t), biq('lowpass', 3600, 0.5), v.out);
      chain(hiss(v, t, 0.4), biq('lowpass', 650, 0.6), amp(t, A * 0.9, 0.004, 0.07), v.out);          // the ball pushing through
    }
    /* dn more Gaussians knocked loose since the last burst: dry ticks, more of them for more */
    function rattle(dn, pan) {
      if (!ok() || dn < 1) return;
      var A = KT.clamp(0.06 + 0.035 * Math.log2(1 + dn), 0.06, 0.3), v = voice('rattle', A, 0.04, pan); if (!v) return;
      var L = [], m = KT.clamp(Math.round(1 + 2.2 * Math.log2(1 + dn)), 1, 18), i;
      for (i = 0; i < m; i++) L.push([0.085 * rr(), 900 + 1700 * rr(), 0.00035 + 0.0008 * rr(), A * (0.35 + 0.65 * rr())]);
      chain(play(v, grains(L, 0.1), v.t), biq('lowpass', 3400, 0.5), v.out);
    }
    /* a contact begins on the elastic surface: sp = approach speed (m/s), k = spring stiffness (N/m); the pitch goes
       as the square root of the stiffness, as a spring's frequency does (FSPR at 160 N/m) */
    function pluck(sp, k, pan) {
      if (!ok()) return;
      var h = KT.clamp(sp / 8, 0, 1), a = 0.045 + 0.16 * Math.pow(h, 0.8), F = FSPR * Math.sqrt(k / 160), v = voice('wobble', a, 0.16, pan); if (!v) return;
      var t = v.t, o = osc(v, F * 1.6, t, 1.1), o2 = osc(v, F * 3.2, t, 0.6);
      o.frequency.exponentialRampToValueAtTime(F, t + 0.11); chain(o, amp(t, a, 0.003, 0.16), v.out);
      o2.frequency.exponentialRampToValueAtTime(F * 2, t + 0.11); chain(o2, amp(t, a * 0.3, 0.003, 0.07), v.out);
      chain(osc(v, 95, t, 0.3), amp(t, a * 0.4, 0.002, 0.04), v.out);                // the push of the ball into it
    }
    /* the ball leaves the bowl at sp m/s */
    function whoosh(sp, pan) {
      if (!ok()) return;
      var h = KT.clamp(sp / 14, 0, 1), A = 0.1 + 0.26 * h, f = 420 + 75 * sp, v = voice('whoosh', A, 0.3, pan); if (!v) return;
      var t = v.t, bp = biq('bandpass', f * 0.6, 1.1), g = amp(null, 0);                // above the ringing springs and the music's bass
      bp.frequency.setValueAtTime(f * 0.6, t); bp.frequency.exponentialRampToValueAtTime(f, t + 0.12); bp.frequency.exponentialRampToValueAtTime(f * 0.45, t + 0.95);
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(A, t + 0.08); g.gain.setTargetAtTime(0, t + 0.1, 0.24);
      chain(hiss(v, t, 1.3), bp, g, v.out);
      chain(hiss(v, t, 1.0), biq('lowpass', 300, 0.7), amp(t, A * 0.3, 0.06, 0.2), v.out);
    }
    /* moment: candidate i says "contact": a dry click, a little lower for each later definition (a click, not a note:
       the chapter's music has its own) */
    function tick(i, pan, delay) {
      if (!ok()) return;
      var f = 3000 * Math.pow(0.92, i), v = voice('tick', 0.13, 0.003, pan, delay); if (!v) return;
      chain(play(v, grains([[0, f, 0.0024, 0.13], [0, f * 1.47, 0.0011, 0.08]], 0.03), v.t), v.out);
    }
    /* flight: the glass rabbit heads the ball. J = the header's impulse, m |v_out - v_in| (N·s), sets the loudness and,
       a little, the brightness: a struck glass, a few inharmonic modes, each a slowly beating pair (the glass is not
       perfectly round), the higher ones shorter, over the click of the tap */
    function glass(J, pan) {
      if (!ok()) return;
      var h = Math.min(J / 12, 1), A = 0.05 + 0.2 * Math.pow(h, 0.8), v = voice('glass', A, 0.45, pan); if (!v) return;
      var f = 1240 * Math.pow(Math.max(J, 0.5) / 6, 0.04), br = 0.5 + 0.5 * h, L = [];
      [[1, 1, 0.85], [2.32, 0.55 * br, 0.42], [3.88, 0.3 * br, 0.2]].forEach(function (m) {
        L.push([0, f * m[0], m[2], A * m[1]]); L.push([0.002, f * m[0] * 1.0037, m[2] * 0.9, A * m[1] * 0.55]);
      });
      chain(play(v, grains(L, 1.6), v.t), v.out);
      chain(hiss(v, v.t, 0.03), biq('highpass', 2600, 0.7), amp(v.t, A * 0.45, 0.0005, 0.004), v.out);   // the tap
    }
    /* flight: the ball meets the lists in the bowl, whose Gaussians drift aside like balloons (the invented rule): a soft,
       rubbery push, louder for more of them (n = how many the ball reached at once) */
    function puff(n, pan) {
      if (!ok() || n < 1) return;
      var h = KT.clamp(Math.log2(1 + n) / 9, 0, 1), A = 0.05 + 0.11 * h, v = voice('puff', A, 0.2, pan); if (!v) return;
      var t = v.t, o = osc(v, 300, t, 0.7);
      o.frequency.exponentialRampToValueAtTime(175, t + 0.35); chain(o, biq('lowpass', 700, 0.7), amp(t, A, 0.035, 0.15), v.out);
      chain(hiss(v, t, 0.5), biq('lowpass', 520, 0.6), amp(t, A * 0.6, 0.05, 0.12), v.out);
    }
    function hush() { if (!ctx) return; var t = ctx.currentTime; live.forEach(function (v) { cut(v, t); }); live = []; }

    /* ---- sustained voices ---- */
    function glide(p, x, t, tc, fresh) { p.cancelScheduledValues(t); if (fresh) p.setValueAtTime(x, t); else p.setTargetAtTime(x, t, tc); }
    function channel(kind) {
      var c = { g: null, srcs: [], until: 0, lvl: 0, quiet: -9, p: null };
      function make(t, f0) {
        var g = c.g = ctx.createGain(), s, f;
        g.gain.value = 0; g.connect(bus); c.srcs = []; c.p = {};                 // silent until set() raises it
        if (kind === 'wobble') {                                                 // a soft tone and its octave
          var o1 = ctx.createOscillator(), o2 = ctx.createOscillator(), lp = biq('lowpass', 1400, 0.6);
          o1.frequency.value = f0 || FSPR; o2.frequency.value = 2 * (f0 || FSPR);
          o1.start(t); o2.start(t); c.srcs.push(o1, o2);
          chain(o1, lp, g); chain(o2, amp(null, 0.22), lp); c.p.f = o1.frequency; c.p.f2 = o2.frequency;
        } else {
          s = ctx.createBufferSource(); s.buffer = noise; s.loop = true; s.start(t, rr() * 2); c.srcs.push(s);
          if (kind === 'roll') { f = biq('lowpass', f0 || 200, 0.9); chain(s, f, biq('highpass', 70, 0.7), g); }
          else { f = biq('bandpass', f0 || 1700, 0.55); chain(s, f, biq('lowpass', 3800, 0.5), g); }
          c.p.f = f.frequency;
        }
        c.srcs[0].onended = function () { try { g.disconnect(); } catch (e) {} };
      }
      c.set = function (level, f) {
        if (!ctx) return;
        if (!ok()) { c.stop(); return; }
        var t = ctx.currentTime, alive = c.g && t < c.until - 0.03, p;
        if (!(level > 1e-4)) {
          if (alive && c.lvl > 0) { p = c.g.gain; p.cancelScheduledValues(t); p.setTargetAtTime(0, t, 0.08); }
          if (c.lvl > 0) c.quiet = t;
          c.lvl = 0; return;
        }
        if (!alive) { c.stop(); make(t, f); }
        if (c.lvl === 0 && t - c.quiet > 0.25 && kind !== 'wobble') cnt[kind]++;   // a wobble is counted by its pluck
        p = c.g.gain; p.cancelScheduledValues(t);
        if (!alive) p.setValueAtTime(0, t);                                      // a new voice rises from silence
        p.setTargetAtTime(level, t, level >= c.lvl ? 0.02 : 0.08);
        p.setTargetAtTime(0, t + HOLD, 0.05);                                    // the next frame renews this; if none comes, it fades
        c.until = t + HOLD + 0.45;
        c.srcs.forEach(function (s) { try { s.stop(c.until); } catch (e) {} });
        if (f) { glide(c.p.f, f, t, kind === 'wobble' ? 0.012 : 0.04, !alive); if (c.p.f2) glide(c.p.f2, 2 * f, t, 0.012, !alive); }
        c.lvl = level;
      };
      c.stop = function () {
        if (!c.g) return;
        var t = ctx.currentTime, g = c.g;
        try { g.gain.cancelScheduledValues(t); g.gain.setTargetAtTime(0, t, 0.012); c.srcs.forEach(function (s) { s.stop(t + 0.1); }); } catch (e) {}
        c.g = null; c.srcs = []; if (c.lvl > 0) c.quiet = t; c.lvl = 0;
      };
      return c;
    }

    function safe(fn) { return function () { try { if (ctx) fn.apply(null, arguments); } catch (e) {} }; }   // sound never breaks the simulation
    return {
      arm: arm, wake: wake, ok: function () { try { return ok(); } catch (e) { return false; } }, now: function () { return ctx ? ctx.currentTime : 0; },
      counts: function () { return Object.assign({}, cnt); },
      knock: safe(knock), crack: safe(crack), shards: safe(shards), scatter: safe(scatter), rattle: safe(rattle),
      pluck: safe(pluck), whoosh: safe(whoosh), tick: safe(tick), glass: safe(glass), puff: safe(puff), hush: safe(hush),
      channel: function (kind) {
        var c = channel(kind);
        return { set: safe(c.set), stop: safe(c.stop) };
      }
    };
  })();

  KT.register('touch', {
    title: 'Can’t touch this',
    blurb: 'A ball meets one bowl stored two ways. A list of Gaussians has no boundary to collide with. A spline surface has a closest ' +
           'point and a normal, and with three material constants it is rigid, brittle or elastic. The list is emulated from the same surface, not a trained 3DGS model.',
    tool: 'WebGL · three.js · contact on the spline',
    scenes: [{ id: 'ghost', label: 'No surface' }, { id: 'moment', label: 'Moment of contact' }, { id: 'solid', label: 'Rigid' },
             { id: 'brittle', label: 'Brittle' }, { id: 'elastic', label: 'Elastic' }, { id: 'flight', label: 'Out of bounds' }],
    create: function (env) {
      var THREE = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new THREE.PerspectiveCamera(36, 1, 0.1, 400), R = G.renderer(root, cam), scene = new THREE.Scene();
      var st = { scene: 'ghost', sc: SCENES.ghost, acc: 0, clock: 0, off: null }, size = { w: 2, h: 1 };
      /* ghost: its drops in turn. pass = no rule, close = the list's close-up of `moment`, loose = the particle rule (lab page only) */
      var GHOST = env.embed ? ['pass', 'close'] : ['pass', 'close', 'loose'];
      var mat = { e: 0.75, tough: 5, stiff: 160 };
      ['e', 'tough', 'stiff'].forEach(function (k) { if (env.opts[k] != null && isFinite(+env.opts[k])) mat[k] = +env.opts[k]; });
      var cap = ui.el('div', 'kt-box kt-caption'); root.appendChild(cap);
      var tag = ui.el('div', 'kt-box kt-side'); root.appendChild(tag);
      var tlm = ui.el('div', 'kt-mo'), tagT = ui.el('div'); tag.appendChild(tlm); tag.appendChild(tagT);   // moment's timeline, then the words
      if (RTL) tagT.setAttribute('dir', 'rtl');                  // the words run right to left on a right-to-left page; the timeline and the stage do not
      var pin = ui.el('div', 'kt-box kt-pin'); root.appendChild(pin);
      SFX.arm();

      scene.add(new THREE.HemisphereLight(0xe6efff, 0x2a3650, 1.5));
      var sun = new THREE.DirectionalLight(0xffffff, 2.6); sun.position.set(LIGHT[0], LIGHT[1], LIGHT[2]); scene.add(sun);

      /* ---------- the bowl: one cubic B-spline patch ----------
         u runs along z and v along x, so that S_u x S_v points up, into the bowl. */
      function bowl(a, b) {
        var z = RB * a * Math.sqrt(1 - SQ * b * b / 2), x = RB * b * Math.sqrt(1 - SQ * a * a / 2);
        return [x, DEPTH * (x * x + z * z) / (RB * RB), z];
      }
      var patch = NB.makePatch({ nu: NC, nv: NC, p: 3 }), P0 = new Float32Array(NC * NC * 3);
      (function () {
        var U = patch.Uk, i, j, q;
        for (j = 0; j < NC; j++) for (i = 0; i < NC; i++) {      // control points at the Greville abscissae
          q = bowl(2 * (U[i + 1] + U[i + 2] + U[i + 3]) / 3 - 1, 2 * (U[j + 1] + U[j + 2] + U[j + 3]) / 3 - 1);
          patch.P[(j * NC + i) * 3] = q[0] + X0; patch.P[(j * NC + i) * 3 + 1] = q[1]; patch.P[(j * NC + i) * 3 + 2] = q[2];
        }
        P0.set(patch.P);
      })();
      /* S, S_u and S_v at one address, without allocating */
      var Nu = new Float64Array(4), Du = new Float64Array(4), Nv = new Float64Array(4), Dv = new Float64Array(4);
      var E = { p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0] };
      function evalS(u, v, o) {
        var ku = NB.findSpan(NC, 3, u, patch.Uk), kv = NB.findSpan(NC, 3, v, patch.Vk), P = patch.P, r, s, q, a, da, db, k;
        NB.basisFuns(ku, u, 3, patch.Uk, Nu, Du); NB.basisFuns(kv, v, 3, patch.Vk, Nv, Dv);
        for (k = 0; k < 3; k++) o.p[k] = o.su[k] = o.sv[k] = 0;
        for (s = 0; s < 4; s++) for (r = 0; r < 4; r++) {
          q = ((kv - 3 + s) * NC + ku - 3 + r) * 3; a = Nu[r] * Nv[s]; da = Du[r] * Nv[s]; db = Nu[r] * Dv[s];
          for (k = 0; k < 3; k++) { o.p[k] += a * P[q + k]; o.su[k] += da * P[q + k]; o.sv[k] += db * P[q + k]; }
        }
        return o;
      }
      var YB = evalS(0.5, 0.5, E).p[1];                          // the bottom of the bowl

      /* ---------- its Gaussians, twice ---------- */
      /* flight's cast needs both scans; its lists in the bowl share the list's field, after the bowl's own N Gaussians */
      var FIG = !!(KT.dtu && KT.dtu.charts && KT.dtuScans && KT.dtuScans[55] && KT.dtuScans[55].charts);
      var NLO = FIG ? 3200 : 0, NLR = FIG ? 3200 : 0, NF = NLO + NLR, NT = N + NF;
      var fR = G.splats(N), fL = fuzzy(NT); scene.add(fR.mesh); scene.add(fL.mesh);
      var us = new Float32Array(NA), du = new Float32Array(NA), baseR = new Float32Array(N * 3), baseL = new Float32Array(N * 3);
      var gr = { pos: new Float32Array(N * 3), su: new Float32Array(N * 3), sv: new Float32Array(N * 3) };
      var grW = { pos: gr.pos, su: gr.su, sv: gr.sv, col: fR.color };   // colours are painted separately, every frame
      var home = { c: new Float32Array(N * 3) }, pr = new Float32Array(N), prMax = 0.02;
      var lax = new Float32Array(N * 9), lsg = new Float32Array(N * 3), lsm = new Float32Array(N);   // the list: unit axes, sigmas, largest sigma
      function tone(u, v) { return ((Math.floor(u * SPANS) + Math.floor(v * SPANS)) & 1) ? 1 : 0.8; }   // the knot spans, as a checker
      function derive() { NB.patchGrid(patch, us, us, gr); G.writePlanoids(fR, 0, grW, NA, NA, du, du, 0.72); }
      (function build() {
        var i, j, k, cs = G.rgb(C.surf), cg = G.rgb(C.splat), rnd = KT.rng(7), tmp = [], spSum = 0, q;
        var rt = KT.rng(23), uo = new Float32Array(N * 3), vo = new Float32Array(N * 3);   // the tilt, thickness and opacity draw from a stream of their own
        for (i = 0; i < NA; i++) { us[i] = (i + 0.5) / NA; du[i] = 1 / NA; }
        for (j = 0; j < NA; j++) for (i = 0; i < NA; i++) for (k = 0; k < 3; k++) baseR[(j * NA + i) * 3 + k] = cs[k] * 0.8 * tone(us[i], us[j]);
        derive(); fR.setCount(N);
        // the list: the same count, each sample moved inside its cell, off the surface a little, turned and resized
        for (j = 0; j < NA; j++) for (i = 0; i < NA; i++) {
          var uu = KT.clamp((i + rnd()) / NA, 0, 0.9999), vv = KT.clamp((j + rnd()) / NA, 0, 0.9999); evalS(uu, vv, E);
          var sx = E.su[0] / NA, sy = E.su[1] / NA, sz = E.su[2] / NA, tx = E.sv[0] / NA, ty = E.sv[1] / NA, tz = E.sv[2] / NA;
          var nx = sy * tz - sz * ty, ny = sz * tx - sx * tz, nz = sx * ty - sy * tx, nl = Math.hypot(nx, ny, nz) || 1e-9, sp = Math.sqrt(nl);
          spSum += sp;
          var off = (rnd() - 0.5) * 0.5 * sp, an = rnd() * 6.283, ca = Math.cos(an), sa = Math.sin(an), f1 = sp * 0.72 * (0.6 + 0.7 * rnd()), f2 = sp * 0.72 * (0.6 + 0.7 * rnd());
          var sl = Math.hypot(sx, sy, sz) || 1e-9, e1 = [sx / sl, sy / sl, sz / sl], e2 = [(ny * e1[2] - nz * e1[1]) / nl, (nz * e1[0] - nx * e1[2]) / nl, (nx * e1[1] - ny * e1[0]) / nl];
          tmp.push({ p: [E.p[0] - 2 * X0 + nx / nl * off, E.p[1] + ny / nl * off, E.p[2] + nz / nl * off], t: tone(uu, vv), n: [nx / nl, ny / nl, nz / nl],
                     u: [0, 1, 2].map(function (k) { return (e1[k] * ca + e2[k] * sa) * f1; }), v: [0, 1, 2].map(function (k) { return (e2[k] * ca - e1[k] * sa) * f2; }) });
        }
        for (i = tmp.length - 1; i > 0; i--) { j = Math.floor(rnd() * (i + 1)); q = tmp[i]; tmp[i] = tmp[j]; tmp[j] = q; }   // no order at all
        tmp.forEach(function (q, k) {
          fL.center.set(q.p, k * 3); uo.set(q.u, k * 3); vo.set(q.v, k * 3);
          baseL[k * 3] = cg[0] * 0.8 * q.t; baseL[k * 3 + 1] = cg[1] * 0.8 * q.t; baseL[k * 3 + 2] = cg[2] * 0.8 * q.t;
          // tilted off the surface: the whole frame (both axes along it, and its normal) turned about a random axis in it
          var lu = Math.hypot(q.u[0], q.u[1], q.u[2]), lv = Math.hypot(q.v[0], q.v[1], q.v[2]), ph = rt() * 6.2832, th = (rt() * 2 - 1) * TILT;
          var ax = [0, 1, 2].map(function (j) { return Math.cos(ph) * q.u[j] / lu + Math.sin(ph) * q.v[j] / lv; }), c0 = Math.cos(th), s0 = Math.sin(th);
          function turn(x) {                                          // Rodrigues: x cos + (a x x) sin + a (a . x)(1 - cos)
            var d = ax[0] * x[0] + ax[1] * x[1] + ax[2] * x[2];
            return [x[0] * c0 + (ax[1] * x[2] - ax[2] * x[1]) * s0 + ax[0] * d * (1 - c0), x[1] * c0 + (ax[2] * x[0] - ax[0] * x[2]) * s0 + ax[1] * d * (1 - c0),
                    x[2] * c0 + (ax[0] * x[1] - ax[1] * x[0]) * s0 + ax[2] * d * (1 - c0)];
          }
          var nn = turn(q.n), sw = (0.25 + 0.15 * rt()) * 0.5 * (lu + lv);
          fL.axU.set(turn(q.u), k * 3); fL.axV.set(turn(q.v), k * 3); fL.axW.set([nn[0] * sw, nn[1] * sw, nn[2] * sw], k * 3);
          fL.alpha[k] = 0.34 + 0.3 * rt();
        });
        home.c.set(fL.center.subarray(0, N * 3));
        for (k = 0; k < N; k++) {                                   // as a particle, a Gaussian is a sphere of its own size: the mean of its two 1-sigma axes
          pr[k] = 0.5 * (Math.hypot(uo[k * 3], uo[k * 3 + 1], uo[k * 3 + 2]) + Math.hypot(vo[k * 3], vo[k * 3 + 1], vo[k * 3 + 2]));   // (as stored before the tilt, which keeps lengths)
          prMax = Math.max(prMax, pr[k]);
          [fL.axU, fL.axV, fL.axW].forEach(function (A, j) {
            var o = k * 3, s = Math.hypot(A[o], A[o + 1], A[o + 2]) || 1e-9;
            lsg[o + j] = s; lax[k * 9 + j * 3] = A[o] / s; lax[k * 9 + j * 3 + 1] = A[o + 1] / s; lax[k * 9 + j * 3 + 2] = A[o + 2] / s;
            lsm[k] = Math.max(lsm[k], s);
          });
        }
        fR.uniforms.uFlat.value = 0.22;
      })();
      /* the list's own field: physics reads and writes `center`; every frame, draw() sorts the Gaussians far to near
         and uploads them in that order */
      function fuzzy(n) {
        var g = new THREE.InstancedBufferGeometry(), base = new THREE.PlaneGeometry(2, 2), A = {}, vb = new THREE.Vector2();
        g.index = base.index; g.setAttribute('position', base.getAttribute('position'));
        [['iC', 3], ['iU', 3], ['iV', 3], ['iW', 3], ['iCol', 3], ['iA', 1]].forEach(function (d) {
          var at = new THREE.InstancedBufferAttribute(new Float32Array(n * d[1]), d[1]); at.setUsage(THREE.DynamicDrawUsage); g.setAttribute(d[0], at); A[d[0]] = at;
        });
        g.instanceCount = 0;
        var m = new THREE.ShaderMaterial({ transparent: true, depthWrite: false, side: THREE.DoubleSide, vertexShader: FUZZ_V, fragmentShader: FUZZ_F,
          uniforms: { uView: { value: new THREE.Vector2(1, 1) }, uLight: { value: new THREE.Vector3(LIGHT[0], LIGHT[1], LIGHT[2]).normalize() } } });
        var mesh = new THREE.Mesh(g, m); mesh.frustumCulled = false; mesh.renderOrder = 3;
        var f = { mesh: mesh, center: new Float32Array(n * 3), axU: new Float32Array(n * 3), axV: new Float32Array(n * 3), axW: new Float32Array(n * 3),
                  color: new Float32Array(n * 3), alpha: new Float32Array(n), key: new Float64Array(n), ord: new Int32Array(n), cnt: new Int32Array(4097) };
        f.draw = function (camera, renderer) {
          var e = camera.matrixWorld.elements, fx = -e[8], fy = -e[9], fz = -e[10], cx = e[12], cy = e[13], cz = e[14];
          var Cc = f.center, K = f.key, O = f.ord, B = 4096, cn = f.cnt, lo = 1e18, hi = -1e18, i, k, o, s, q, tot = 0, sc;
          for (i = 0; i < n; i++) {
            o = i * 3; if (Cc[o + 1] < -60) { K[i] = NaN; continue; }            // long gone
            k = (Cc[o] - cx) * fx + (Cc[o + 1] - cy) * fy + (Cc[o + 2] - cz) * fz; K[i] = k; tot++;
            if (k < lo) lo = k; if (k > hi) hi = k;
          }
          cn.fill(0); sc = (B - 1) / Math.max(hi - lo, 1e-9);
          for (i = 0; i < n; i++) if (K[i] === K[i]) cn[1 + Math.floor((hi - K[i]) * sc)]++;   // bucket 0 = the farthest
          for (k = 0; k < B; k++) cn[k + 1] += cn[k];
          for (i = 0; i < n; i++) if (K[i] === K[i]) O[cn[Math.floor((hi - K[i]) * sc)]++] = i;
          var aC = A.iC.array, aU = A.iU.array, aV = A.iV.array, aW = A.iW.array, aK = A.iCol.array, aA = A.iA.array, U = f.axU, V = f.axV, W = f.axW, Kc = f.color;
          for (q = 0; q < tot; q++) {
            i = O[q]; o = i * 3; s = q * 3;
            aC[s] = Cc[o]; aC[s + 1] = Cc[o + 1]; aC[s + 2] = Cc[o + 2]; aU[s] = U[o]; aU[s + 1] = U[o + 1]; aU[s + 2] = U[o + 2];
            aV[s] = V[o]; aV[s + 1] = V[o + 1]; aV[s + 2] = V[o + 2]; aW[s] = W[o]; aW[s + 1] = W[o + 1]; aW[s + 2] = W[o + 2];
            aK[s] = Kc[o]; aK[s + 1] = Kc[o + 1]; aK[s + 2] = Kc[o + 2]; aA[q] = f.alpha[i];
          }
          g.instanceCount = tot;
          for (k in A) { A[k].updateRange.offset = 0; A[k].updateRange.count = tot * A[k].itemSize; A[k].needsUpdate = true; }
          renderer.getDrawingBufferSize(vb); m.uniforms.uView.value.copy(vb);
        };
        return f;
      }

      /* ---------- the ball, and what is drawn about a contact ---------- */
      var ballMesh = new THREE.Mesh(new THREE.SphereGeometry(BR, 48, 32), new THREE.MeshStandardMaterial({ color: 0xe4e9f0, roughness: 0.5, metalness: 0 }));
      scene.add(ballMesh);
      var arrow = G.vector(C.surf, { radius: 0.02, ghost: 0.22 }); scene.add(arrow.obj);
      var dot = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshBasicMaterial({ color: C.text, depthTest: false, transparent: true }));
      dot.renderOrder = 30; scene.add(dot);
      var NS = 40, iso = G.lines(4 * NS, C.text, 0.85), drop = G.lines(2, C.text, 0.55); scene.add(iso.obj); scene.add(drop.obj);
      var net = G.lines(4 * NC * (NC - 1), C.ctrl, 0.5); net.mat.depthTest = false; net.obj.renderOrder = 12; scene.add(net.obj);
      function points(n, px, opacity) {
        var g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
        var o = new THREE.Points(g, new THREE.PointsMaterial({ color: C.ctrl, size: px, sizeAttenuation: false, depthTest: false, transparent: true, opacity: opacity }));
        o.frustumCulled = false; o.renderOrder = 13; scene.add(o); return o;
      }
      var cpts = points(NC * NC, 4.5, 0.85), sup = points(16, 10, 1);
      var stem = G.lines(2 * NC * NC, C.ctrl, 1); stem.mat.depthTest = false; stem.obj.renderOrder = 12; scene.add(stem.obj);   // each spring, from its anchor to its control point
      var TR = 220, trail = G.lines(2 * TR, C.text, 0.4), tr = { p: new Float32Array(TR * 3), n: 0, k: 0 }; scene.add(trail.obj);
      // moment: where the ball's lowest point was when each definition said "contact" (red: the list; cyan: the spline), on its vertical
      var mkL = G.lines(2 * CANDS.length, C.err, 0.95), mkS = G.lines(2, C.surf, 1), mkG = G.lines(2, C.text, 0.3);
      [mkL, mkS, mkG].forEach(function (l) { l.mat.depthTest = false; l.obj.renderOrder = 14; scene.add(l.obj); });
      var low = new THREE.Mesh(new THREE.SphereGeometry(0.011, 12, 8), new THREE.MeshBasicMaterial({ color: C.text, depthTest: false, transparent: true }));   // the ball's lowest point
      low.renderOrder = 31; scene.add(low); ballMesh.renderOrder = 4;                   // (drawn after the list, for when it is see-through)

      /* ---------- state ---------- */
      var ball = { p: [0, 0, 0], v: [0, 0, 0], on: false, rest: false };
      var cp = { ok: false, u: 0.5, v: 0.5, p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], n: [0, 1, 0], nc: [0, 1, 0], d: 9, sd: 9, edge: false };
      var run = { k: 0, t: 0, tr: 0, hover: 0.55, still: 0, E0: 1, rule: 0, close: false, ph: '' };
      /* the model the drop on screen runs: 'moment' through all of `moment` and in ghost's close-up, else the scene's own */
      function model() { return run.close ? 'moment' : st.sc.model; }
      var hit = { n: 0, J: 0, t: -9, ask: 0, held: true, vout: 0 };
      var surf = { broken: false, tb: 0 };
      var el = { touch: false, f: 0, fmax: 0, J: 0, lost: 0, pen: 0, sup: [] };
      var fl = { phase: 'drop', tt: 0, th: 0, vsim: 0, vset: 0, boost: false, vin: 0,
                 t1: null, ta: null, tf: 0, tb: 0, miss: false, hit: [null, null], cps: [null, null], loops: 0 };   // (the cast's: see `meet`)
      var prt = { v: new Float32Array(N * 3), aw: new Uint8Array(N), n: 0, seen: new Uint8Array(N), nSeen: 0, sub: 0 };
      function newCp() { return { ok: false, u: 0.5, v: 0.5, p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], n: [0, 1, 0], nc: [0, 1, 0], d: 9, sd: 9, edge: false }; }
      var mo = { side: -1, D: 9, slow: 1, near: [], top: 0, spot: [0, 0, 0], val: null, t: [], y: [], who: [], root: null, end: 0, cpm: newCp(), cpr: newCp(), cps: newCp() };
      /* what the sound hears: the physics only ever adds to these, and they never feed back */
      var au = { simT: 0, J: 0, Jn: 0, tone: 1, rest: 0, restV: 0, inside: 0, calls: 0, hits: 0, jl: 0, crack: 0, pluck: 0, launch: 0, tick: [],
                 first: true, lastN: 0, rat: 0, ratT: -9, env: 0, glass: 0, puff: 0 };

      /* ---------- closest point on the spline ----------
         Minimise |S(u,v) - c|^2 over the address: a few Gauss-Newton steps in (u,v), started from the last
         answer (or from the nearest of a coarse lattice of samples when there is none, or it went stale). */
      var SG = 12, seedP = new Float32Array(SG * SG * 3);
      (function () { for (var j = 0; j < SG; j++) for (var i = 0; i < SG; i++) { evalS((i + 0.5) / SG, (j + 0.5) / SG, E); seedP.set(E.p, (j * SG + i) * 3); } })();
      function closest(c, o) {
        var k, best = 1e18, bi = 0, dx, dy, dz, d2, it, a, b, g, gu, gv, det;
        for (k = 0; k < SG * SG; k++) { dx = seedP[k * 3] - c[0]; dy = seedP[k * 3 + 1] - c[1]; dz = seedP[k * 3 + 2] - c[2]; d2 = dx * dx + dy * dy + dz * dz; if (d2 < best) { best = d2; bi = k; } }
        if (o.ok) { evalS(o.u, o.v, E); dx = E.p[0] - c[0]; dy = E.p[1] - c[1]; dz = E.p[2] - c[2]; if (dx * dx + dy * dy + dz * dz > best * 1.05 + 0.25) o.ok = false; }
        if (!o.ok) { o.u = (bi % SG + 0.5) / SG; o.v = (Math.floor(bi / SG) + 0.5) / SG; }
        for (it = 0; it < 5; it++) {
          evalS(o.u, o.v, E); dx = E.p[0] - c[0]; dy = E.p[1] - c[1]; dz = E.p[2] - c[2];
          gu = dx * E.su[0] + dy * E.su[1] + dz * E.su[2]; gv = dx * E.sv[0] + dy * E.sv[1] + dz * E.sv[2];
          a = E.su[0] * E.su[0] + E.su[1] * E.su[1] + E.su[2] * E.su[2]; b = E.su[0] * E.sv[0] + E.su[1] * E.sv[1] + E.su[2] * E.sv[2];
          g = E.sv[0] * E.sv[0] + E.sv[1] * E.sv[1] + E.sv[2] * E.sv[2]; det = a * g - b * b || 1e-12;
          o.u = KT.clamp(o.u - (g * gu - b * gv) / det, 0, 1); o.v = KT.clamp(o.v - (a * gv - b * gu) / det, 0, 1);
        }
        evalS(o.u, o.v, o);
        var nx = o.su[1] * o.sv[2] - o.su[2] * o.sv[1], ny = o.su[2] * o.sv[0] - o.su[0] * o.sv[2], nz = o.su[0] * o.sv[1] - o.su[1] * o.sv[0], nl = Math.hypot(nx, ny, nz) || 1e-9;
        o.n[0] = nx / nl; o.n[1] = ny / nl; o.n[2] = nz / nl;
        dx = c[0] - o.p[0]; dy = c[1] - o.p[1]; dz = c[2] - o.p[2];
        o.d = Math.hypot(dx, dy, dz); o.sd = dx * o.n[0] + dy * o.n[1] + dz * o.n[2];
        o.edge = o.u <= 0 || o.u >= 1 || o.v <= 0 || o.v >= 1;
        if (o.edge) { o.nc[0] = dx / (o.d || 1e-9); o.nc[1] = dy / (o.d || 1e-9); o.nc[2] = dz / (o.d || 1e-9); }   // off the rim: the nearest point is on the edge
        else { k = o.sd < 0 ? -1 : 1; o.nc[0] = o.n[0] * k; o.nc[1] = o.n[1] * k; o.nc[2] = o.n[2] * k; }            // inside: along the normal, on the ball's side
        o.ok = true; return o;
      }

      /* ---------- the ball in free flight: the exact parabola over one sub-step ---------- */
      function fly(h) { var p = ball.p, v = ball.v; p[0] += v[0] * h; p[1] += v[1] * h - 0.5 * GRAV * h * h; p[2] += v[2] * h; v[1] -= GRAV * h; }
      function speed() { return Math.hypot(ball.v[0], ball.v[1], ball.v[2]); }
      function energy() { var v = ball.v; return 0.5 * BM * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) + BM * GRAV * (ball.p[1] - YB - BR); }

      /* ---------- rigid contact ----------
         When the ball overlaps the surface and approaches it: go back along the parabola to the moment of
         impact, give the ball the impulse J = m (1 + e) |v.n| along the normal, and come forward again.
         No position is corrected by hand, so with e = 1 the energy is conserved to rounding. A slow contact
         (|v.n| < VREST) is a resting one: no bounce, the ball is held on the surface and rolls with a little
         resistance. In the brittle scene the surface delivers at most its toughness and then breaks.
         h: the sub-step (H, or HM in `moment`). */
      function rigid(brittle, h) {
        h = h || H;
        closest(ball.p, cp);
        if (cp.d >= BR) { ball.rest = false; return; }
        var n = cp.nc, p = ball.p, v = ball.v, vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2], pen = BR - cp.d, k, J;
        if (vn >= 0) return;                                                        // already leaving
        if (-vn < VREST && mat.e < 0.999) {                                         // resting contact
          for (k = 0; k < 3; k++) { v[k] -= vn * n[k]; p[k] += pen * n[k]; }
          var s = speed(), dec = (MU * GRAV + CROLL * s) * h;
          if (s <= dec) v[0] = v[1] = v[2] = 0; else for (k = 0; k < 3; k++) v[k] *= (s - dec) / s;
          ball.rest = true; au.rest += h; au.restV += speed() * h; return;
        }
        ball.rest = false;
        var tau = Math.min(h, pen / -vn);
        fly(-tau);
        vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        J = BM * (1 + mat.e) * -vn;
        hit.ask = J; hit.held = true; hit.t = run.t;
        if (brittle && J > mat.tough) { hit.held = false; J = mat.tough; shatter(cp.u, cp.v, n, J, cp.p); }
        for (k = 0; k < 3; k++) v[k] += J / BM * n[k];
        hit.J = J; hit.n++; hit.vout = speed();
        if (hit.held) { au.J = Math.max(au.J, J); au.Jn++; au.tone = 0.97 + 0.06 * cp.v; } else au.crack = hit.ask / mat.tough;
        fly(tau);
      }

      /* ---------- brittle: pieces of the (u,v) domain ----------
         Cells in polar coordinates of the address plane around the impact: rings that widen outwards, cut by
         radial cracks at jittered angles. Each piece is a rigid body. The impulse the surface took, J along
         -n, is shared as a velocity field that falls off with distance from the impact (so momentum is exact);
         a piece also opens outwards and tumbles with the gradient of that field. Then gravity. */
      var frag = { id: new Uint16Array(N), loc: new Float32Array(N * 3), au: new Float32Array(N * 3), av: new Float32Array(N * 3), list: [], n: 0 };
      function shatter(u0, v0, n, J, hp) {
        var rr = KT.rng(31 + run.k), RINGS = [0.045, 0.11, 0.2, 0.33, 0.5, 9], KS = [3, 6, 8, 10, 11, 12], cuts = [], base = [], total = 0, r, k, s, a, b, f, th, rho;
        var p1 = rr() * 6.283, p2 = rr() * 6.283, map = [], L = frag.list = [], q, o;
        for (r = 0; r < KS.length; r++) { base[r] = total; cuts[r] = []; for (k = 0; k < KS[r]; k++) cuts[r].push((k + 0.2 + 0.6 * rr()) / KS[r]); total += KS[r]; }
        for (b = 0; b < NA; b++) for (a = 0; a < NA; a++) {
          k = b * NA + a; th = Math.atan2(us[b] - v0, us[a] - u0);
          rho = Math.hypot(us[a] - u0, us[b] - v0) * (1 + 0.16 * Math.sin(3 * th + p1) + 0.09 * Math.sin(7 * th + p2));
          r = 0; while (rho > RINGS[r]) r++;
          f = ((th / 6.2832) % 1 + 1) % 1; s = 0; while (s < KS[r] && f >= cuts[r][s]) s++; s %= KS[r];
          if (map[base[r] + s] == null) { map[base[r] + s] = L.length; L.push({ n: 0, c: [0, 0, 0], v: [0, 0, 0], ax: [0, 1, 0], w: 0, shade: 0.72 + 0.5 * rr(), heat: 0, m: 0 }); }
          q = L[frag.id[k] = map[base[r] + s]]; q.n++; q.c[0] += fR.center[k * 3]; q.c[1] += fR.center[k * 3 + 1]; q.c[2] += fR.center[k * 3 + 2];
        }
        var sig = 0.5, sum = 0, mr = [0, 0, 0], V0, d = [-n[0], -n[1], -n[2]];
        L.forEach(function (q) {
          q.c[0] /= q.n; q.c[1] /= q.n; q.c[2] /= q.n; q.m = MS * q.n / N;
          var e = [q.c[0] - hp[0], q.c[1] - hp[1], q.c[2] - hp[2]], en = e[0] * n[0] + e[1] * n[1] + e[2] * n[2], l;
          q.rho = Math.hypot(e[0], e[1], e[2]); q.wt = Math.exp(-q.rho * q.rho / (2 * sig * sig)); sum += q.m * q.wt;
          e[0] -= en * n[0]; e[1] -= en * n[1]; e[2] -= en * n[2]; l = Math.hypot(e[0], e[1], e[2]) || 1e-9; q.e = [e[0] / l, e[1] / l, e[2] / l];
        });
        V0 = J / sum;
        L.forEach(function (q) {                                    // opening: outwards along the surface, zero net momentum
          var op = 0.35 * V0 * q.wt * q.rho / (q.rho + 0.3); q.o = [q.e[0] * op, q.e[1] * op, q.e[2] * op];
          mr[0] += q.m * q.o[0]; mr[1] += q.m * q.o[1]; mr[2] += q.m * q.o[2];
        });
        L.forEach(function (q) {
          var k, wx, wy, wz, g = -0.5 * V0 * q.wt * q.rho / (sig * sig), e = q.e, l;
          for (k = 0; k < 3; k++) q.v[k] = d[k] * V0 * q.wt + q.o[k] - mr[k] / MS;
          wx = g * (e[1] * d[2] - e[2] * d[1]) + (rr() - 0.5) * 2.4; wy = g * (e[2] * d[0] - e[0] * d[2]) + (rr() - 0.5) * 2.4; wz = g * (e[0] * d[1] - e[1] * d[0]) + (rr() - 0.5) * 2.4;
          l = Math.hypot(wx, wy, wz) || 1e-9; q.ax = [wx / l, wy / l, wz / l]; q.w = l; q.heat = q.wt;
        });
        for (k = 0; k < N; k++) {
          q = L[frag.id[k]]; o = k * 3;
          frag.loc[o] = fR.center[o] - q.c[0]; frag.loc[o + 1] = fR.center[o + 1] - q.c[1]; frag.loc[o + 2] = fR.center[o + 2] - q.c[2];
        }
        frag.au.set(fR.axU); frag.av.set(fR.axV); frag.n = L.length;
        surf.broken = true; surf.tb = run.t;
      }
      function writeFragments() {                                   // every piece: a parabola and a steady spin, in closed form
        var t = run.t - surf.tb, L = frag.list, M = [], i, k, o, q, m, c = fR.center, au = fR.axU, av = fR.axV, lc = frag.loc, x, y, z;
        for (i = 0; i < L.length; i++) {
          q = L[i]; var a = q.w * t, ca = Math.cos(a), sa = Math.sin(a), ic = 1 - ca, X = q.ax[0], Y = q.ax[1], Z = q.ax[2];
          M.push([ca + X * X * ic, X * Y * ic - Z * sa, X * Z * ic + Y * sa, Y * X * ic + Z * sa, ca + Y * Y * ic, Y * Z * ic - X * sa, Z * X * ic - Y * sa, Z * Y * ic + X * sa, ca + Z * Z * ic,
                  q.c[0] + q.v[0] * t, q.c[1] + q.v[1] * t - 0.5 * GRAV * t * t, q.c[2] + q.v[2] * t]);
        }
        for (k = 0; k < N; k++) {
          m = M[frag.id[k]]; o = k * 3;
          x = lc[o]; y = lc[o + 1]; z = lc[o + 2];
          c[o] = m[9] + m[0] * x + m[1] * y + m[2] * z; c[o + 1] = m[10] + m[3] * x + m[4] * y + m[5] * z; c[o + 2] = m[11] + m[6] * x + m[7] * y + m[8] * z;
          x = frag.au[o]; y = frag.au[o + 1]; z = frag.au[o + 2];
          au[o] = m[0] * x + m[1] * y + m[2] * z; au[o + 1] = m[3] * x + m[4] * y + m[5] * z; au[o + 2] = m[6] * x + m[7] * y + m[8] * z;
          x = frag.av[o]; y = frag.av[o + 1]; z = frag.av[o + 2];
          av[o] = m[0] * x + m[1] * y + m[2] * z; av[o + 1] = m[3] * x + m[4] * y + m[5] * z; av[o + 2] = m[6] * x + m[7] * y + m[8] * z;
        }
      }
      function mend() { surf.broken = false; frag.n = 0; frag.list = []; derive(); }

      /* ---------- elastic: a membrane over the control net ----------
         Every inner control point has a mass, is tied to its rest position (stiffness KA k) and to its four
         neighbours (stiffness k) and is damped; the rim is pinned. The ball and the surface push on each other
         with a stiff, damped contact spring along the normal at the closest point; the surface's share goes to
         the 16 control points that shape that address, each by its basis weight N_i(u) N_j(v). */
      var MX = new Float32Array(NC * NC * 3), MV = new Float32Array(NC * NC * 3), MF = new Float32Array(NC * NC * 3);
      function membrane(contact) {
        var km = mat.stiff, ka = km * KA, i, j, q, k, w3 = NC * 3, n, pen, s;
        MF.fill(0); el.f = 0; el.pen = 0;
        if (contact && ball.on) {
          closest(ball.p, cp);
          pen = cp.edge ? BR - cp.d : BR - cp.sd; n = cp.edge ? cp.nc : cp.n;
          if (pen > 0 && pen < 2.5 * BR) {
            var S = NB.patchSupport(patch, cp.u, cp.v), vs = [0, 0, 0], fn, vr;
            for (s = 0; s < S.length; s++) { q = (S[s].j * NC + S[s].i) * 3; for (k = 0; k < 3; k++) vs[k] += S[s].w * MV[q + k]; }
            vr = (ball.v[0] - vs[0]) * n[0] + (ball.v[1] - vs[1]) * n[1] + (ball.v[2] - vs[2]) * n[2];
            fn = Math.max(0, KC * pen - CC * vr);
            for (k = 0; k < 3; k++) {
              var vt = ball.v[k] - vs[k] - vr * n[k];                 // sliding velocity: a viscous friction, counted as damping
              ball.v[k] += (fn * n[k] - CT * vt) / BM * H; el.lost += CT * vt * vt * H;
            }
            for (s = 0; s < S.length; s++) { q = (S[s].j * NC + S[s].i) * 3; for (k = 0; k < 3; k++) MF[q + k] -= fn * n[k] * S[s].w; }
            if (!el.touch) { el.touch = true; el.J = 0; el.fmax = 0; hit.t = run.t; au.pluck = Math.max(au.pluck, -vr); }
            el.f = fn; el.fmax = Math.max(el.fmax, fn); el.J += fn * H; el.sup = S; el.pen = pen;
            if (fn > 0) el.lost += CC * vr * vr * H;
          } else if (el.touch) { el.touch = false; hit.n++; hit.J = el.J; hit.vout = speed(); }
        }
        for (j = 1; j < NC - 1; j++) for (i = 1; i < NC - 1; i++) {
          q = (j * NC + i) * 3;
          for (k = 0; k < 3; k++) MF[q + k] += km * (MX[q + k - 3] + MX[q + k + 3] + MX[q + k - w3] + MX[q + k + w3] - 4 * MX[q + k]) - ka * MX[q + k] - MDAMP * MV[q + k];
        }
        for (j = 1; j < NC - 1; j++) for (i = 1; i < NC - 1; i++) {
          q = (j * NC + i) * 3;
          for (k = 0; k < 3; k++) { el.lost += MDAMP * MV[q + k] * MV[q + k] * H; MV[q + k] += MF[q + k] / MC * H; MX[q + k] += MV[q + k] * H; patch.P[q + k] = P0[q + k] + MX[q + k]; }
        }
      }
      function membraneEnergy() {
        var km = mat.stiff, ka = km * KA, e = 0, i, j, q, k, d;
        for (j = 0; j < NC; j++) for (i = 0; i < NC; i++) {
          q = (j * NC + i) * 3;
          for (k = 0; k < 3; k++) {
            e += 0.5 * MC * MV[q + k] * MV[q + k] + 0.5 * ka * MX[q + k] * MX[q + k];
            if (i + 1 < NC) { d = MX[q + k + 3] - MX[q + k]; e += 0.5 * km * d * d; }
            if (j + 1 < NC) { d = MX[q + k + NC * 3] - MX[q + k]; e += 0.5 * km * d * d; }
          }
        }
        return e;
      }
      function flatten() { MX.fill(0); MV.fill(0); patch.P.set(P0); el.touch = false; el.f = el.pen = 0; el.lost = 0; el.sup = []; }

      /* ---------- the list as loose particles ----------
         The only way to make a list of Gaussians answer a ball without building a surface first: each Gaussian
         is a small sphere of its own size. The ball hits it as a body 1,500 times heavier would. A Gaussian that
         has been hit is a free body (gravity, a little drag) and meets the others as soft spheres: an impulse
         along the line of centres when two approach (equal masses), and, because the Gaussians of a list overlap
         where they lie, a spring that pushes every overlapping pair apart. One that has not been touched yet
         stays asleep where it was stored, until one that has moved (by 0.4 of its radius) overlaps it; nothing ties it
         to its neighbours. */
      function wake(i) { if (!prt.aw[i]) { prt.aw[i] = 1; prt.n++; } }
      function ballVsList(push) {
        var P = fL.center, V = prt.v, c = ball.p, bv = ball.v, i, o, dx, dy, dz, d2, d, vr, j, RR = BR + prMax, rr, hits = 0;
        if (Math.abs(c[0] + X0) > RB + 2 || c[1] > DEPTH + 2 || c[1] < -3) return 0;
        au.calls++;
        for (i = 0; i < N; i++) {
          o = i * 3; dx = P[o] - c[0]; if (dx > RR || dx < -RR) continue;
          dy = P[o + 1] - c[1]; if (dy > RR || dy < -RR) continue;
          dz = P[o + 2] - c[2]; d2 = dx * dx + dy * dy + dz * dz; rr = push ? BR + pr[i] : BR; if (d2 >= rr * rr) continue;
          if (!prt.seen[i]) { prt.seen[i] = 1; prt.nSeen++; }
          if (!push) { au.inside++; continue; }
          d = Math.sqrt(d2) || 1e-6; dx /= d; dy /= d; dz /= d;
          vr = (V[o] - bv[0]) * dx + (V[o + 1] - bv[1]) * dy + (V[o + 2] - bv[2]) * dz; if (vr >= 0) continue;
          j = -(1 + EP) * vr / (1 / MP + 1 / BM);
          V[o] += j / MP * dx; V[o + 1] += j / MP * dy; V[o + 2] += j / MP * dz;
          bv[0] -= j / BM * dx; bv[1] -= j / BM * dy; bv[2] -= j / BM * dz;
          wake(i); hits++; au.jl += j;
        }
        au.hits += hits;
        return hits;
      }
      var HB = 1 << 16;                                           // a hashed grid: cells of the largest particle diameter
      function hkey(x, y, z) { return ((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)) & (HB - 1); }
      var grid = { head: new Int32Array(HB + 1), fill: new Int32Array(HB), item: new Int32Array(N), key: new Int32Array(N),
                   ix: new Int16Array(N), iy: new Int16Array(N), iz: new Int16Array(N),
        build: function () {
          var P = fL.center, cs = 2 * prMax, x0 = -X0 - RB - 1.2, y0 = YB - 1.6, z0 = -RB - 1.2, i, k, x, y, z, head = this.head, fill = this.fill;
          var nx = Math.ceil((2 * RB + 2.4) / cs), ny = Math.ceil((DEPTH + 3.2) / cs);
          head.fill(0);
          for (i = 0; i < N; i++) {
            x = Math.floor((P[i * 3] - x0) / cs); y = Math.floor((P[i * 3 + 1] - y0) / cs); z = Math.floor((P[i * 3 + 2] - z0) / cs);
            if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nx) { this.key[i] = -1; continue; }   // far from the bowl: on its own
            this.ix[i] = x; this.iy[i] = y; this.iz[i] = z; k = this.key[i] = hkey(x, y, z); head[k + 1]++;
          }
          for (k = 0; k < HB; k++) { head[k + 1] += head[k]; fill[k] = head[k]; }
          for (i = 0; i < N; i++) if (this.key[i] >= 0) this.item[fill[this.key[i]]++] = i;
        } };
      var PSTEP = 4;                                               // Gaussian against Gaussian: every fourth sub-step
      function particles() {
        var P = fL.center, V = prt.v, aw = prt.aw, i, o, k = Math.exp(-DRAG * H);
        for (i = 0; i < N; i++) {
          if (!aw[i]) continue; o = i * 3;
          if (P[o + 1] < -60) continue;                             // long gone
          V[o] *= k; V[o + 1] = V[o + 1] * k - GRAV * H; V[o + 2] *= k;
          P[o] += V[o] * H; P[o + 1] += V[o + 1] * H; P[o + 2] += V[o + 2] * H;
        }
        if (++prt.sub % PSTEP) return;
        grid.build();
        var head = grid.head, item = grid.item, HC = home.c, cx, cy, cz, a, b, g, q, s, j, p, dx, dy, dz, d2, d, vr, rr, f, far, hp = PSTEP * H;
        for (i = 0; i < N; i++) {
          if (!aw[i] || grid.key[i] < 0) continue;
          o = i * 3; cx = grid.ix[i]; cy = grid.iy[i]; cz = grid.iz[i];
          dx = P[o] - HC[o]; dy = P[o + 1] - HC[o + 1]; dz = P[o + 2] - HC[o + 2]; far = dx * dx + dy * dy + dz * dz > 0.16 * pr[i] * pr[i];
          for (g = -1; g <= 1; g++) for (b = -1; b <= 1; b++) for (a = -1; a <= 1; a++) {
            q = hkey(cx + a, cy + b, cz + g);
            for (s = head[q]; s < head[q + 1]; s++) {
              j = item[s]; if (j === i || (aw[j] && j < i)) continue; p = j * 3;       // a pair of awake ones is met once
              dx = P[p] - P[o]; dy = P[p + 1] - P[o + 1]; dz = P[p + 2] - P[o + 2]; d2 = dx * dx + dy * dy + dz * dz; rr = pr[i] + pr[j];
              if (d2 >= rr * rr || d2 < 1e-12) continue;
              if (!aw[j] && !far) continue;                                          // a sleeper is woken by one that has really moved
              d = Math.sqrt(d2); dx /= d; dy /= d; dz /= d;
              vr = (V[p] - V[o]) * dx + (V[p + 1] - V[o + 1]) * dy + (V[p + 2] - V[o + 2]) * dz;
              f = W2 * (rr - d) * hp - (vr < 0 ? vr * (1 + EPP) / 2 : 0);              // the overlap spring, and the impulse of an approach
              V[o] -= f * dx; V[o + 1] -= f * dy; V[o + 2] -= f * dz; V[p] += f * dx; V[p + 1] += f * dy; V[p + 2] += f * dz;
              wake(j);
            }
          }
        }
      }
      /* how many others a Gaussian of the list overlaps, as stored: measured once, on the same grid */
      var overlaps = (function () {
        var P = fL.center, n = 0, i, a, b, g, q, s, j, dx, dy, dz, rr;
        grid.build();
        for (i = 0; i < N; i++) for (g = -1; g <= 1; g++) for (b = -1; b <= 1; b++) for (a = -1; a <= 1; a++) {
          q = hkey(grid.ix[i] + a, grid.iy[i] + b, grid.iz[i] + g);
          for (s = grid.head[q]; s < grid.head[q + 1]; s++) {
            j = grid.item[s]; if (j <= i) continue; dx = P[j * 3] - P[i * 3]; dy = P[j * 3 + 1] - P[i * 3 + 1]; dz = P[j * 3 + 2] - P[i * 3 + 2]; rr = pr[i] + pr[j];
            if (dx * dx + dy * dy + dz * dz < rr * rr) n += 2;
          }
        }
        return n / N;
      })();
      function restoreList() { fL.center.set(home.c); prt.v.fill(0); prt.aw.fill(0); prt.seen.fill(0); prt.n = prt.nSeen = prt.sub = 0; }

      /* ---------- flight: the owl and the rabbit, two real objects stored as spline charts ----------
         Each is the object's one cubic chart from its scan (the owl with its books; never the table), in metres (FS), set
         where the script has it. Its Gaussians are derived from it on an even (u, v) lattice, as the bowl's are, with
         gl.writePlanoids: the owl's far side is left out (a closed chart, solid), the rabbit keeps it (it is see-through).
         The rabbit wears the talk's clear glass: talk/scenes.js's rule (KnotTalk.materials.glass, the one the owl's
         material scene uses), evaluated once per control point, colour and opacity, then carried to the Gaussians by the
         same basis as everything else. In the list bowl stand list versions of both: EMULATED, never trained, made from
         the same charts the way the bowl's own list is made from the bowl (below, `listOf`). */
      var FIGS = small() ? 0.62 : 0.8;            // lattice scale: about 19,000 Gaussians on a laptop, 11,000 on a phone (the cover: 160,000 and 30,000)
      function small() { return Math.min(window.innerWidth, window.innerHeight) < 560; }
      var FNu = new Float64Array(4), FDu = new Float64Array(4), FNv = new Float64Array(4), FDv = new Float64Array(4);
      var FE = { p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], c: [0, 0, 0] };
      function figure(D, o) {
        var d = D.charts.filter(function (c) { return !c.plane; })[0], nu = d.nu, nv = d.nv, n = nu * nv, i, j, q, x, y, z;
        var k, patch = NB.makePatch({ nu: nu, nv: nv, p: 3 }), P = patch.P, sc = FS * D.unit_mm / KT.dtu.unit_mm, cy = Math.cos(o.yaw), sy = Math.sin(o.yaw);
        for (i = 0; i < n; i++) {
          q = i * 3; x = d.P[q] * sc; y = d.P[q + 1] * sc; z = d.P[q + 2] * sc;
          P[q] = o.at[0] + cy * x + sy * z; P[q + 1] = o.at[1] + y; P[q + 2] = o.at[2] - sy * x + cy * z;
        }
        patch.C.set(d.C);
        var F = { d: d, patch: patch, B: new Float32Array(P), sc: sc, nu: nu, nv: nv, W: null, piv: null, th: 0, flip: 1, at: o.at, yaw: o.yaw, key: '' };
        // the window of the closest-point search: from the Greville abscissa of the lowest control row whose mean height
        // passes vcut (in the scan's units) up to the top of the chart
        for (j = 0; j < nv; j++) { for (y = 0, i = 0; i < nu; i++) y += d.P[(j * nu + i) * 3 + 1]; if (y / nu > o.vcut) break; }
        F.vlo = KT.clamp((patch.Vk[j + 1] + patch.Vk[j + 2] + patch.Vk[j + 3]) / 3, 0, 1);
        var na = F.na = Math.max(8, Math.round(o.na)), nb = F.nb = Math.max(8, Math.round(o.nb));
        F.us = new Float32Array(na); F.vs = new Float32Array(nb); F.du = new Float32Array(na); F.dv = new Float32Array(nb);
        for (i = 0; i < na; i++) { F.us[i] = (i + 0.5) / na; F.du[i] = 1 / na; }
        for (j = 0; j < nb; j++) { F.vs[j] = (j + 0.5) / nb; F.dv[j] = 1 / nb; }
        F.grid = { pos: new Float32Array(na * nb * 3), su: new Float32Array(na * nb * 3), sv: new Float32Array(na * nb * 3), col: new Float32Array(na * nb * 3) };
        NB.patchGrid(patch, F.us, F.vs, F.grid);
        F.seed = []; for (j = 0; j < nb; j++) if (F.vs[j] >= F.vlo) for (i = 0; i < na; i++) F.seed.push(j * na + i);
        // which way is out: S_u x S_v against the way from the chart's vertical axis (its control points' mean), by a vote of
        // every lattice sample
        var g = F.grid, mx = 0, mz = 0, vote = 0, t;
        for (i = 0; i < n; i++) { mx += P[i * 3]; mz += P[i * 3 + 2]; }
        mx /= n; mz /= n;
        for (k = 0; k < na * nb; k++) {
          t = k * 3; x = g.su[t + 1] * g.sv[t + 2] - g.su[t + 2] * g.sv[t + 1]; z = g.su[t] * g.sv[t + 1] - g.su[t + 1] * g.sv[t];
          vote += (x * (g.pos[t] - mx) + z * (g.pos[t + 2] - mz)) > 0 ? 1 : -1;
        }
        F.flip = vote < 0 ? -1 : 1;
        return F;
      }
      /* S, S_u, S_v (and the colour, when o.c is there) of a figure at (u, v). th turns the owl's head: every control point
         by th times its weight W, about the neck (piv), in the plane of the eye: the same edit the drawing makes */
      function evalF(F, u, v, o, th) {
        var pt = F.patch, nu = F.nu, ku = NB.findSpan(pt.mu, 3, u, pt.Uk), kv = NB.findSpan(pt.mv, 3, v, pt.Vk), B = F.B, W = F.W, C = pt.C;
        var r, s, q, a, da, db, k, x, y, z, w, cs, sn, ex, ey, oc = o.c;
        NB.basisFuns(ku, u, 3, pt.Uk, FNu, FDu); NB.basisFuns(kv, v, 3, pt.Vk, FNv, FDv);
        for (k = 0; k < 3; k++) { o.p[k] = o.su[k] = o.sv[k] = 0; if (oc) oc[k] = 0; }
        for (s = 0; s < 4; s++) for (r = 0; r < 4; r++) {
          q = (kv - 3 + s) * nu + ku - 3 + r; a = FNu[r] * FNv[s]; da = FDu[r] * FNv[s]; db = FNu[r] * FDv[s];
          x = B[q * 3]; y = B[q * 3 + 1]; z = B[q * 3 + 2];
          if (th && W && (w = W[q])) { cs = Math.cos(th * w); sn = Math.sin(th * w); ex = x - F.piv[0]; ey = y - F.piv[1]; x = F.piv[0] + ex * cs - ey * sn; y = F.piv[1] + ex * sn + ey * cs; }
          o.p[0] += a * x; o.p[1] += a * y; o.p[2] += a * z; o.su[0] += da * x; o.su[1] += da * y; o.su[2] += da * z; o.sv[0] += db * x; o.sv[1] += db * y; o.sv[2] += db * z;
          if (oc) { oc[0] += a * C[q * 3]; oc[1] += a * C[q * 3 + 1]; oc[2] += a * C[q * 3 + 2]; }
        }
        return o;
      }
      /* the closest point of a figure's chart to c, inside its window: Gauss-Newton in (u, v) as `closest` does on the bowl,
         started from the nearer of the last answer and the nearest lattice sample of the window. These charts are far less
         even than the bowl (a scan's sheet, curved about as tightly as the ball is wide), so the steps are damped: at most
         0.02 in u or v, and kept only when they bring the point nearer (else halved) */
      function descend(F, c, u, v, th, o) {
        var dx, dy, dz, it, a, b, g, gu, gv, det, su, sv, m, h, f, f2, u2, v2;
        evalF(F, u, v, FE, th); f = dist2(FE.p, c);
        for (it = 0; it < 60; it++) {
          dx = FE.p[0] - c[0]; dy = FE.p[1] - c[1]; dz = FE.p[2] - c[2];
          gu = dx * FE.su[0] + dy * FE.su[1] + dz * FE.su[2]; gv = dx * FE.sv[0] + dy * FE.sv[1] + dz * FE.sv[2];
          a = FE.su[0] * FE.su[0] + FE.su[1] * FE.su[1] + FE.su[2] * FE.su[2]; b = FE.su[0] * FE.sv[0] + FE.su[1] * FE.sv[1] + FE.su[2] * FE.sv[2];
          g = FE.sv[0] * FE.sv[0] + FE.sv[1] * FE.sv[1] + FE.sv[2] * FE.sv[2]; det = a * g - b * b || 1e-12;
          su = -(g * gu - b * gv) / det; sv = -(a * gv - b * gu) / det;
          if ((v >= 1 && sv > 0) || (v <= F.vlo && sv < 0)) { sv = 0; su = -gu / (a || 1e-12); }      // on the edge of the window: along it
          if ((u >= 1 && su > 0) || (u <= 0 && su < 0)) { su = 0; sv = -gv / (g || 1e-12); }
          for (var pass = 0; pass < 2; pass++) {                  // the Gauss-Newton step; where it stalls, steepest descent
            if (pass) { su = -gu / (a || 1e-12); sv = -gv / (g || 1e-12); if (v >= 1 && sv > 0 || v <= F.vlo && sv < 0) sv = 0; if (u >= 1 && su > 0 || u <= 0 && su < 0) su = 0; }
            m = Math.max(Math.abs(su), Math.abs(sv)); if (m > 0.02) { su *= 0.02 / m; sv *= 0.02 / m; }
            for (h = 1; h > 1 / 4096; h *= 0.5) {
              u2 = KT.clamp(u + su * h, 0, 1); v2 = KT.clamp(v + sv * h, F.vlo, 1); evalF(F, u2, v2, FE, th); f2 = dist2(FE.p, c);
              if (f2 < f) break;
            }
            if (f2 < f) break;
          }
          if (!(f2 < f)) break;
          m = Math.abs(u2 - u) + Math.abs(v2 - v); u = u2; v = v2; f = f2;
          if (m < 1e-10) break;
        }
        o.u = u; o.v = v; return f;
      }
      function closestF(F, c, o, th) {
        var G = F.grid.pos, S = F.seed, k, i, best = 1e18, bi = S[0], dx, dy, dz, d2;
        for (k = 0; k < S.length; k++) { i = S[k] * 3; dx = G[i] - c[0]; dy = G[i + 1] - c[1]; dz = G[i + 2] - c[2]; d2 = dx * dx + dy * dy + dz * dz; if (d2 < best) { best = d2; bi = S[k]; } }
        var wu = o.u, wv = o.v, warm = o.ok;
        d2 = descend(F, c, F.us[bi % F.na], F.vs[(bi / F.na) | 0], th, o);    // from the nearest sample, and from the last answer: the nearer end wins
        if (warm) { k = o.u; i = o.v; if (descend(F, c, wu, wv, th, o) > d2) { o.u = k; o.v = i; } else d2 = dist2(FE.p, c); }
        if (d2 < (BR + 0.05) * (BR + 0.05)) {                    // near a contact: the scan's fine relief holds neighbouring minima, so look around
          var bu = o.u, bv = o.v, j, e;
          for (j = 0; j < 4; j++) {
            e = descend(F, c, KT.clamp(bu + (j & 1 ? 0.012 : -0.012), 0, 1), KT.clamp(bv + (j & 2 ? 0.012 : -0.012), F.vlo, 1), th, o);
            if (e < d2) { d2 = e; k = o.u; i = o.v; } else { k = bu; i = bv; }
            bu = k; bv = i;
          }
          o.u = bu; o.v = bv;
        }
        evalF(F, o.u, o.v, o, th);
        var nx = o.su[1] * o.sv[2] - o.su[2] * o.sv[1], ny = o.su[2] * o.sv[0] - o.su[0] * o.sv[2], nz = o.su[0] * o.sv[1] - o.su[1] * o.sv[0], nl = (Math.hypot(nx, ny, nz) || 1e-9) * F.flip;
        o.n[0] = nx / nl; o.n[1] = ny / nl; o.n[2] = nz / nl;
        dx = c[0] - o.p[0]; dy = c[1] - o.p[1]; dz = c[2] - o.p[2];
        o.d = Math.hypot(dx, dy, dz); o.sd = dx * o.n[0] + dy * o.n[1] + dz * o.n[2];
        o.edge = o.u <= 0 || o.u >= 1 || o.v >= 1;
        if (o.edge) { o.nc[0] = dx / (o.d || 1e-9); o.nc[1] = dy / (o.d || 1e-9); o.nc[2] = dz / (o.d || 1e-9); }
        else { k = o.sd < 0 ? -1 : 1; o.nc[0] = o.n[0] * k; o.nc[1] = o.n[1] * k; o.nc[2] = o.n[2] * k; }
        o.ok = true; return o;
      }
      var FO = null, FR = null, fF = null, RP = { T: [0, 0, 0], M: new Float64Array(9), R: new Float64Array(9), S: [1, 1, 1] };
      var LH = new Float32Array(NT * 3), LSZ = new Float32Array(NT);          // the lists' homes and their Gaussians' largest sigma
      var BAIM = [0, 0, 0];                                                    // where the rabbit sends the ball: the head of the rabbit's list
      if (FIG) (function cast() {
        var A = KT.dtu, Bn = KT.dtuScans[55], i, j, k, o, x, y, z, w;
        /* the owl, facing the eye; its head (control points above the shoulders, softly: talk/scenes.js's owlLife takes
           the same band) turns about the neck */
        FO = figure(A, { yaw: 0, at: OWL_AT, na: 150 * FIGS, nb: 116 * FIGS, vcut: 2.1 });
        var dO = FO.d, n = FO.nu * FO.nv; FO.W = new Float32Array(n); FO.hi = [];
        for (i = 0; i < n; i++) { w = ss(2.25, 2.8, dO.P[i * 3 + 1]); if (w > 0) { FO.W[i] = w; FO.hi.push(i); } }
        FO.piv = [OWL_AT[0] + 0.24 * FO.sc, OWL_AT[1] + 2.45 * FO.sc];
        FO.hc = [OWL_AT[0] + 0.24 * FO.sc, OWL_AT[1] + 2.95 * FO.sc, OWL_AT[2] + 0.2 * FO.sc]; FO.reach = BR + 0.62 * FO.sc + 0.3;
        for (i = 0, y = -1e9; i < n; i++) y = Math.max(y, FO.B[i * 3 + 1]); FO.top = y;
        /* the rabbit: which way its face looks, in the scan (from the middle of its head to its nose, the frontmost point of the
           head), so that it can be turned RFACE from the eye towards the owl. It is built around its own feet; the script
           moves it (rabbitPose) */
        var dR = Bn.charts.filter(function (c) { return !c.plane; })[0], m = dR.nu * dR.nv, hx = 0, hz = 0, hn = 0, nose = null;
        for (i = 0; i < m; i++) {
          y = dR.P[i * 3 + 1]; if (y < 2.4) continue; hx += dR.P[i * 3]; hz += dR.P[i * 3 + 2]; hn++;
          if (y < 3.2 && (!nose || dR.P[i * 3 + 2] > nose[1])) nose = [dR.P[i * 3], dR.P[i * 3 + 2]];
        }
        var yaw = RFACE - Math.atan2(nose[0] - hx / hn, nose[1] - hz / hn);
        FR = figure(Bn, { yaw: yaw, at: [0, 0, 0], na: 150 * FIGS, nb: 84 * FIGS, vcut: 2.0 });
        FR.face = [Math.sin(RFACE), 0, Math.cos(RFACE)];
        var hs = [0, 0, 0], ns = 0, Bp = FR.B, top = -1e9;
        for (i = 0; i < m; i++) if (dR.P[i * 3 + 1] > 2.4) { hs[0] += Bp[i * 3]; hs[1] += Bp[i * 3 + 1]; hs[2] += Bp[i * 3 + 2]; ns++; top = Math.max(top, Bp[i * 3 + 1]); }
        FR.hc = [hs[0] / ns, hs[1] / ns, hs[2] / ns]; FR.reach = BR + 0.9 * FR.sc + 0.35;
        /* its glass: the material's rule on every control point (the net's normal, out of the object, and how bright the
           photograph is there: the carving's relief), colour and opacity; both carried to the lattice by the basis */
        var MG = KT.materials && KT.materials.glass, nu = dR.nu, nv = dR.nv, Pd = dR.P, Kd = dR.C;
        FR.glass = !!MG;
        if (MG) {
          var Cg = FR.patch.C, AC = new Float32Array(m * 3), smp = {};
          for (j = 0; j < nv; j++) for (i = 0; i < nu; i++) {
            var a = (j * nu + Math.max(0, i - 1)) * 3, b = (j * nu + Math.min(nu - 1, i + 1)) * 3, g = (Math.max(0, j - 1) * nu + i) * 3, h = (Math.min(nv - 1, j + 1) * nu + i) * 3;
            var ux = Pd[b] - Pd[a], uy = Pd[b + 1] - Pd[a + 1], uz = Pd[b + 2] - Pd[a + 2], vx = Pd[h] - Pd[g], vy = Pd[h + 1] - Pd[g + 1], vz = Pd[h + 2] - Pd[g + 2];
            x = uy * vz - uz * vy; y = uz * vx - ux * vz; z = ux * vy - uy * vx; w = (Math.hypot(x, y, z) || 1) * FR.flip; o = (j * nu + i) * 3;
            smp.nx = x / w; smp.ny = y / w; smp.nz = z / w; smp.x = Pd[o]; smp.y = Pd[o + 1]; smp.z = Pd[o + 2]; smp.head = Pd[o + 1] > 2.4;
            smp.sh = Math.min(1, (0.30 * Kd[o] + 0.59 * Kd[o + 1] + 0.11 * Kd[o + 2]) / 0.30); smp.a = 1;
            MG.colour(smp, MG.p);
            Cg[o] = smp.r; Cg[o + 1] = smp.g; Cg[o + 2] = smp.b; AC[o] = AC[o + 1] = AC[o + 2] = smp.a;
          }
          var ag = { pos: new Float32Array(FR.na * FR.nb * 3), col: new Float32Array(FR.na * FR.nb * 3) };
          NB.patchGrid(Object.assign({}, FR.patch, { C: AC }), FR.us, FR.vs, ag);
          FR.ag = new Float32Array(FR.na * FR.nb); for (i = 0; i < FR.ag.length; i++) FR.ag[i] = KT.clamp(ag.col[i * 3], 0, 1);
          NB.patchGrid(FR.patch, FR.us, FR.vs, FR.grid);                       // the lattice again, with the glass colours
        }
        /* its forehead, where the owl aims: of the samples on the front of the head, below the chart's open top, the one whose
           outward normal looks most towards where the ball comes from (up, and towards the owl); the ball's centre is aimed
           one radius out from it */
        var gp = FR.grid.pos, gu = FR.grid.su, gv = FR.grid.sv, uin = [0.62, 0.62, 0.48], bestF = -9, fi = 0;
        for (k = 0; k < FR.na * FR.nb; k++) {
          o = k * 3; if (gp[o + 1] < FR.hc[1] || FR.vs[(k / FR.na) | 0] > 0.95) continue;
          x = gu[o + 1] * gv[o + 2] - gu[o + 2] * gv[o + 1]; y = gu[o + 2] * gv[o] - gu[o] * gv[o + 2]; z = gu[o] * gv[o + 1] - gu[o + 1] * gv[o];
          w = (Math.hypot(x, y, z) || 1) * FR.flip; x /= w; y /= w; z /= w;
          if (x * FR.face[0] + z * FR.face[2] < 0.3) continue;                 // facing forwards
          var sc2 = x * uin[0] + y * uin[1] + z * uin[2]; if (sc2 > bestF) { bestF = sc2; fi = k; FR.fn = [x, y, z]; }
        }
        FR.fp = [gp[fi * 3], gp[fi * 3 + 1], gp[fi * 3 + 2]];
        FR.aimL = [FR.fp[0] + FR.fn[0] * BR, FR.fp[1] + FR.fn[1] * BR, FR.fp[2] + FR.fn[2] * BR];

        /* both figures in one field, the rabbit first (its count is fixed), then the owl's front: one field with an opacity
           per Gaussian, solid and see-through together, as the manifold module draws its glass owl over the table */
        fF = G.splats(FR.na * FR.nb + FO.na * FO.nb, { alpha: !!MG }); fF.uniforms.uFlat.value = 1;   // photographs: their colours already hold the light
        fF.uniforms.uRim.value = KT.theme === 'light' && MG ? 0.12 : 0.3;      // a light material: no dark outline on a light page
        if (MG) { fF.alpha.fill(1); fF.alpha.set(FR.ag); fF.mesh.userData.ktSplat.thin = true; }
        fF.mesh.visible = false; scene.add(fF.mesh);

        /* the lists in the bowl: the same two charts, a little apart on the bowl's floor */
        var bowlY = function (x, z) { return DEPTH * ((x + X0) * (x + X0) + z * z) / (RB * RB); };
        var LO = figure(A, { yaw: 0.25, at: [-X0 - 0.5, 0, -0.4], na: 8, nb: 8, vcut: 9 }), LR = figure(Bn, { yaw: yaw, at: [-X0 + 0.5, 0.06, 0.12], na: 8, nb: 8, vcut: 9 });
        [LO, LR].forEach(function (L) { var dy = bowlY(L.at[0], L.at[2]) - 0.04; for (var i = 1; i < L.B.length; i += 3) L.B[i] += dy; L.patch.P.set(L.B); });
        var tmp = [], cg = G.rgb(C.splat), rnd = KT.rng(57), rt = KT.rng(91);
        listOf(LO, 64, 50, rnd, tmp); listOf(LR, 80, 40, rnd, tmp);
        for (i = tmp.length - 1; i > 0; i--) { j = Math.floor(rnd() * (i + 1)); o = tmp[i]; tmp[i] = tmp[j]; tmp[j] = o; }   // one list: no order at all
        var hl = [0, 0, 0], hn2 = 0, rtop = -1e9;
        tmp.forEach(function (q) { if (q.r) rtop = Math.max(rtop, q.p[1]); });
        tmp.forEach(function (q, t) {
          var k = N + t, lu = Math.hypot(q.u[0], q.u[1], q.u[2]), lv = Math.hypot(q.v[0], q.v[1], q.v[2]), ph = rt() * 6.2832, th = (rt() * 2 - 1) * TILT;
          var ax = [0, 1, 2].map(function (j) { return Math.cos(ph) * q.u[j] / lu + Math.sin(ph) * q.v[j] / lv; }), c0 = Math.cos(th), s0 = Math.sin(th);
          function turn(x) {
            var d = ax[0] * x[0] + ax[1] * x[1] + ax[2] * x[2];
            return [x[0] * c0 + (ax[1] * x[2] - ax[2] * x[1]) * s0 + ax[0] * d * (1 - c0), x[1] * c0 + (ax[2] * x[0] - ax[0] * x[2]) * s0 + ax[1] * d * (1 - c0),
                    x[2] * c0 + (ax[0] * x[1] - ax[1] * x[0]) * s0 + ax[2] * d * (1 - c0)];
          }
          var nn = turn(q.n), sw = (0.25 + 0.15 * rt()) * 0.5 * (lu + lv);
          LH.set(q.p, k * 3); fL.axU.set(turn(q.u), k * 3); fL.axV.set(turn(q.v), k * 3); fL.axW.set([nn[0] * sw, nn[1] * sw, nn[2] * sw], k * 3);
          fL.alpha[k] = 0.34 + 0.3 * rt(); LSZ[k] = Math.max(lu, lv);
          // the scan's own colours, a third of the way towards the colour of Gaussians: these are lists too
          fL.color[k * 3] = q.c[0] * 0.66 + cg[0] * 0.3; fL.color[k * 3 + 1] = q.c[1] * 0.66 + cg[1] * 0.3; fL.color[k * 3 + 2] = q.c[2] * 0.66 + cg[2] * 0.3;
          if (q.r && q.p[1] > rtop - 0.32) { hl[0] += q.p[0]; hl[1] += q.p[1]; hl[2] += q.p[2]; hn2++; }
        });
        BAIM = [hl[0] / hn2, hl[1] / hn2 - 0.1, hl[2] / hn2];                  // through the head of the rabbit's list
      })();
      LH.set(home.c); LSZ.set(lsm);
      /* a list of na x nb Gaussians from a figure's chart, as the bowl's list is made from the bowl: one jittered sample in
         every (u, v) cell, nudged off the surface, turned and resized at random within its cell's size (tilted and
         thickened by the caller) */
      function listOf(F, na, nb, rnd, out) {
        var i, j, e = { p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], c: [0, 0, 0] }, isR = F.d.name !== 'owl';
        for (j = 0; j < nb; j++) for (i = 0; i < na; i++) {
          evalF(F, KT.clamp((i + rnd()) / na, 0, 0.9999), KT.clamp((j + rnd()) / nb, 0, 0.9999), e, 0);
          var sx = e.su[0] / na, sy = e.su[1] / na, sz = e.su[2] / na, tx = e.sv[0] / nb, ty = e.sv[1] / nb, tz = e.sv[2] / nb;
          var nx = sy * tz - sz * ty, ny = sz * tx - sx * tz, nz = sx * ty - sy * tx, nl = Math.hypot(nx, ny, nz) || 1e-9, sp = Math.sqrt(nl);
          var off = (rnd() - 0.5) * 0.5 * sp, an = rnd() * 6.283, ca = Math.cos(an), sa = Math.sin(an), f1 = sp * 0.72 * (0.6 + 0.7 * rnd()), f2 = sp * 0.72 * (0.6 + 0.7 * rnd());
          var sl = Math.hypot(sx, sy, sz) || 1e-9, e1 = [sx / sl, sy / sl, sz / sl], e2 = [(ny * e1[2] - nz * e1[1]) / nl, (nz * e1[0] - nx * e1[2]) / nl, (nx * e1[1] - ny * e1[0]) / nl];
          out.push({ p: [e.p[0] + nx / nl * off, e.p[1] + ny / nl * off, e.p[2] + nz / nl * off], c: e.c.slice(), r: isR, n: [nx / nl, ny / nl, nz / nl],
                     u: [0, 1, 2].map(function (k) { return (e1[k] * ca + e2[k] * sa) * f1; }), v: [0, 1, 2].map(function (k) { return (e2[k] * ca - e1[k] * sa) * f2; }) });
        }
      }
      function ss(a, b, v) { v = KT.clamp((v - a) / (b - a), 0, 1); return v * v * (3 - 2 * v); }
      function dist2(p, c) { var x = p[0] - c[0], y = p[1] - c[1], z = p[2] - c[2]; return x * x + y * y + z * z; }

      /* SCRIPTED, the owl's nod, x seconds from the contact it is timed to: a wind-up away from the rabbit, a swing through
         the ball towards it, and back. Radians, positive = the head's top towards the rabbit */
      function owlNod(x) {
        if (x < -0.55 || x > 1.0) return 0;
        if (x < -0.15) return -0.16 * KT.ease((x + 0.55) / 0.4);
        if (x < 0.1) return -0.16 + 0.46 * KT.ease((x + 0.15) / 0.25);
        return 0.3 * (1 - KT.ease((x - 0.1) / 0.9));
      }
      function owlTh(t) { return fl.t1 == null ? 0 : owlNod(t - fl.t1); }
      /* SCRIPTED, the rabbit at time t: a crouch, a jump whose top is the moment the ball is due (fl.ta), a landing that
         squashes it (volume kept), and a nod that turns it towards the bowl and tips it forwards through the header. A rigid
         motion of its chart (and the squash, while the ball is far): world = T + R S local */
      var VJ = Math.sqrt(2 * GRAV * HJ), TJ = VJ / GRAV;
      function rabbitPose(t, o) {
        var ta = fl.ta, y = 0, sy = 1, psi = 0, phi = 0, x;
        if (ta != null) {
          var t0 = ta - TJ, t1 = ta + TJ;                               // take-off and landing
          if (t >= t0 - CROUCH && t < t0) sy = 1 - 0.13 * Math.sin(Math.PI * (t - t0 + CROUCH) / CROUCH);
          else if (t >= t0 && t < t1) { x = t - t0; y = VJ * x - 0.5 * GRAV * x * x; }
          else if (t >= t1 && t < t1 + LAND) sy = 1 - 0.1 * Math.sin(Math.PI * (t - t1) / LAND);
          x = t - ta;
          psi = x < -0.45 || x > 1 ? 0 : x < -0.12 ? 0.18 * KT.ease((x + 0.45) / 0.33) : x < 0.12 ? 0.18 - 0.48 * KT.ease((x + 0.12) / 0.24) : -0.3 * (1 - KT.ease((x - 0.12) / 0.88));
          phi = 0.14 * Math.sin(Math.PI * KT.clamp((x + 0.3) / 0.7, 0, 1));
        }
        var f = FR.face, ax = f[2], az = -f[0], cp = Math.cos(phi), sp = Math.sin(phi), cs = Math.cos(psi), sn = Math.sin(psi);
        // tip forwards: about the horizontal axis a = up x face (Rodrigues), then turn about the vertical
        var A = [cp + ax * ax * (1 - cp), -az * sp, ax * az * (1 - cp), az * sp, cp, -ax * sp, az * ax * (1 - cp), ax * sp, cp + az * az * (1 - cp)];
        var Y = [cs, 0, sn, 0, 1, 0, -sn, 0, cs], R = o.R, M = o.M, i, j, sx = 1 / Math.sqrt(sy);
        for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) R[i * 3 + j] = Y[i * 3] * A[j] + Y[i * 3 + 1] * A[3 + j] + Y[i * 3 + 2] * A[6 + j];
        o.S[0] = sx; o.S[1] = sy; o.S[2] = sx;
        for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) M[i * 3 + j] = R[i * 3 + j] * o.S[j];
        o.T[0] = RAB_AT[0]; o.T[1] = RAB_AT[1] + y; o.T[2] = RAB_AT[2];
        return o;
      }
      function toWorld(o, l, w) { var M = o.M; for (var i = 0; i < 3; i++) w[i] = o.T[i] + M[i * 3] * l[0] + M[i * 3 + 1] * l[1] + M[i * 3 + 2] * l[2]; return w; }
      function toLocal(o, w, l) {                                       // S^-1 R^T (w - T)
        var R = o.R, x = w[0] - o.T[0], y = w[1] - o.T[1], z = w[2] - o.T[2];
        for (var i = 0; i < 3; i++) l[i] = (R[i] * x + R[3 + i] * y + R[6 + i] * z) / o.S[i];
        return l;
      }
      function rot(o, l, w) { var R = o.R; for (var i = 0; i < 3; i++) w[i] = R[i * 3] * l[0] + R[i * 3 + 1] * l[1] + R[i * 3 + 2] * l[2]; return w; }

      /* ---------- where and when the ball meets a figure (computed) ---------- */
      var CL = [0, 0, 0], HCW = [0, 0, 0], PW = [0, 0, 0], NW = [0, 0, 0], BP0 = [0, 0, 0], BV0 = [0, 0, 0], CQ = [0, 0, 0];
      /* |S(u, v) - c| for figure `who` (0 the owl, 1 the rabbit) with the figure as the script has it at time t; o receives
         the closest point (o.wp, o.wn in the world) */
      function gapAt(who, c, t, o) {
        if (!who) { closestF(FO, c, o, owlTh(t)); o.wp = o.p.slice(); o.wn = o.nc.slice(); return o.d; }
        rabbitPose(t, RP); toLocal(RP, c, CL); closestF(FR, CL, o, 0);
        o.wp = toWorld(RP, o.p, [0, 0, 0]); o.wn = rot(RP, o.nc, [0, 0, 0]);
        return Math.hypot(c[0] - o.wp[0], c[1] - o.wp[1], c[2] - o.wp[2]);
      }
      function along(t, c) { c[0] = BP0[0] + BV0[0] * t; c[1] = BP0[1] + BV0[1] * t - 0.5 * GRAV * t * t; c[2] = BP0[2] + BV0[2] * t; return c; }
      /* after a sub-step: if the ball now overlaps the figure, find the moment inside the sub-step when it first touched (the
         root of |S(u, v) - c(t)| - r, by bisection, the figure moving with its script), put the ball there, and head it */
      function meet(who) {
        var F = who ? FR : FO, o = fl.cps[who], cen = who ? toWorld(rabbitPose(run.t, RP), FR.hc, HCW) : FO.hc;
        if (Math.hypot(ball.p[0] - cen[0], ball.p[1] - cen[1], ball.p[2] - cen[2]) > F.reach) { o.near = false; return; }
        o.near = true;
        var d = gapAt(who, ball.p, run.t, o), t0 = run.t - H, lo = 0, hi = H, it, m, g, k = { u: o.u, v: o.v, wp: o.wp, wn: o.wn, edge: o.edge };
        if (d > BR) return;
        for (it = 0; it < 48 && hi - lo > 1e-15; it++) {
          m = 0.5 * (lo + hi); g = gapAt(who, along(m, CQ), t0 + m, o);
          if (g > BR) lo = m; else { hi = m; d = g; k = { u: o.u, v: o.v, wp: o.wp, wn: o.wn, edge: o.edge }; }   // the answer at the touching end
        }
        along(hi, CQ); o.u = k.u; o.v = k.v; o.wp = k.wp; o.wn = k.wn; o.edge = k.edge;
        if (who) rabbitPose(t0 + hi, RP);
        // an independent look at the same answer: the nearest of a fine (u, v) grid around it, with the figure as it was then
        var chk = 1e9, a, b, uu, vv, th = who ? 0 : owlTh(t0 + hi), cl = who ? toLocal(RP, CQ, [0, 0, 0]) : CQ;
        for (b = -12; b <= 12; b++) for (a = -12; a <= 12; a++) {
          uu = KT.clamp(o.u + a * 0.0025, 0, 1); vv = KT.clamp(o.v + b * 0.0025, F.vlo, 1); evalF(F, uu, vv, FE, th);
          chk = Math.min(chk, Math.hypot(cl[0] - FE.p[0], cl[1] - FE.p[1], cl[2] - FE.p[2]) * (who ? RP.S[0] : 1));
        }
        ball.p[0] = CQ[0]; ball.p[1] = CQ[1]; ball.p[2] = CQ[2]; ball.v[0] = BV0[0]; ball.v[1] = BV0[1] - GRAV * hi; ball.v[2] = BV0[2];
        var rec = { who: who, t: t0 + hi - run.tr, at: t0 + hi, u: o.u, v: o.v, d: d, gap: d - BR, check: chk - BR, p: o.wp.slice(), n: o.wn.slice(), c: CQ.slice(), vin: ball.v.slice(), edge: o.edge };
        header(who, rec);
        fly(H - hi);
      }
      /* SCRIPTED, where a header sends the ball: the parabola from where it is to the next target in a set time (T1 to the
         rabbit's forehead, one radius out, at the top of its jump; T2 to the head of the rabbit's list in the bowl). The
         contact before it is computed; the speed and direction after it are chosen. The impulse is what that takes,
         m |v_out - v_in|, and it sets the loudness of the sound */
      function header(who, rec) {
        var A = [0, 0, 0], T = who ? T2 : T1, c = ball.p, k, J;
        if (!who) { fl.ta = rec.at + T1; toWorld(rabbitPose(fl.ta, RP), FR.aimL, A); }
        else A = BAIM.slice();
        for (k = 0; k < 3; k++) ball.v[k] = (A[k] - c[k]) / T;
        ball.v[1] += 0.5 * GRAV * T;
        J = BM * Math.hypot(ball.v[0] - rec.vin[0], ball.v[1] - rec.vin[1], ball.v[2] - rec.vin[2]);
        rec.vout = ball.v.slice(); rec.J = J; rec.aim = A; fl.hit[who] = rec; fl.th = rec.at;
        if (who) au.glass = Math.max(au.glass, J); else { au.J = Math.max(au.J, J); au.Jn++; au.tone = 0.82; }   // glass rings; stone knocks
        fl.phase = who ? 'air2' : 'air1';
      }
      /* at the drop: the moment the ball will meet the owl's head with the head at the middle of its swing, so that the nod
         can be timed to it (the actual contact is computed as it happens, with the head where it is) */
      function predictOwl() {
        var o = newCp(), c = ball.p.slice(), th = owlNod(0), y, lo, hi, it, vy = ball.v[1];
        for (y = c[1]; y > OWL_AT[1]; y -= 0.01) { c[1] = y; if (closestF(FO, c, o, th).d < BR) break; }
        lo = y + 0.01; hi = y;
        for (it = 0; it < 40; it++) { c[1] = 0.5 * (lo + hi); if (closestF(FO, c, o, th).d < BR) hi = c[1]; else lo = c[1]; }
        return (vy + Math.sqrt(vy * vy + 2 * GRAV * (ball.p[1] - hi))) / GRAV;
      }

      /* ---------- INVENTED: the lists answer the ball like floating balloons ----------
         A list defines no force: there is no surface to push back and nothing ties a Gaussian to another. To show the ball
         passing through without breaking anything, each Gaussian near the ball's path is given this rule: inside a reach of
         BREACH beyond the ball (and its own size) it is pushed away from the ball's centre and dragged a little along with
         the ball (both falling off as (1 - d / reach)^2); once displaced it is pulled home by a soft, under-damped spring,
         lifted a little while away (it floats up) and swayed a little (each with its own phase): it bobs and wanders home.
         The ball feels nothing back (the rule is one-way) and falls through. Every fourth sub-step, on the bowl's list and
         the figures' lists alike. */
      var BV = new Float32Array(NT * 3), BA = new Uint8Array(NT), BPU = new Uint8Array(NT), BSTEP = 4;
      var bal = { sub: 0, moved: 0, inside: 0, reach: 0, dmax: 0, dnow: 0 };
      function balReset() { BV.fill(0); BA.fill(0); BPU.fill(0); bal.sub = bal.moved = bal.inside = bal.reach = bal.dmax = bal.dnow = 0; }
      function balloons() {
        if (++bal.sub % BSTEP) return;
        var h = BSTEP * H, P = fL.center, c = ball.p, bv = ball.v, i, o, dx, dy, dz, d2, d, R, q, ins = 0, rch = 0, sp, w;
        if (ball.on && Math.abs(c[0] + X0) < RB + 1.4 && c[1] > -1.6 && c[1] < 3.4) {
          au.calls++;
          for (i = 0; i < NT; i++) {
            o = i * 3; R = BR + BREACH + LSZ[i];
            dx = P[o] - c[0]; if (dx > R || dx < -R) continue; dy = P[o + 1] - c[1]; if (dy > R || dy < -R) continue;
            dz = P[o + 2] - c[2]; d2 = dx * dx + dy * dy + dz * dz; if (d2 >= R * R) continue;
            d = Math.sqrt(d2) || 1e-6; q = 1 - d / R; q *= q; rch++;
            if (d < BR) ins++;
            if (!BPU[i]) { BPU[i] = 1; bal.moved++; }
            BA[i] = 1;
            BV[o] += (BKP * q * dx / d + BKD * q * (bv[0] - BV[o])) * h; BV[o + 1] += (BKP * q * dy / d + BKD * q * (bv[1] - BV[o + 1])) * h;
            BV[o + 2] += (BKP * q * dz / d + BKD * q * (bv[2] - BV[o + 2])) * h;
          }
          au.inside += ins;
        }
        bal.inside = ins; bal.reach = rch;
        var dm = 0, ts = run.t;
        for (i = 0; i < NT; i++) {
          if (!BA[i]) continue; o = i * 3;
          dx = P[o] - LH[o]; dy = P[o + 1] - LH[o + 1]; dz = P[o + 2] - LH[o + 2]; d = Math.sqrt(dx * dx + dy * dy + dz * dz); w = 3.1 * ts + 2.39996 * i;
          BV[o] += (-BK * dx - BC * BV[o] + BSWAY * d * Math.sin(w)) * h;
          BV[o + 1] += (-BK * dy - BC * BV[o + 1] + BLIFT * d) * h;
          BV[o + 2] += (-BK * dz - BC * BV[o + 2] + BSWAY * d * Math.cos(w)) * h;
          sp = Math.hypot(BV[o], BV[o + 1], BV[o + 2]); if (sp > BVMAX) { BV[o] *= BVMAX / sp; BV[o + 1] *= BVMAX / sp; BV[o + 2] *= BVMAX / sp; }
          P[o] += BV[o] * h; P[o + 1] += BV[o + 1] * h; P[o + 2] += BV[o + 2] * h;
          if (d < 2e-4 && sp < 2e-3) { P[o] = LH[o]; P[o + 1] = LH[o + 1]; P[o + 2] = LH[o + 2]; BV[o] = BV[o + 1] = BV[o + 2] = 0; BA[i] = 0; }   // home: asleep again
          if (d > dm) dm = d;
        }
        bal.dnow = dm; if (dm > bal.dmax) bal.dmax = dm;
      }
      /* the cast on stage or off it; the lists of the figures stay out of the list's field (parked far below, where its
         drawing skips them) in every other scene */
      function figShow(on) {
        if (!FIG) return;
        fF.mesh.visible = on; fR.mesh.visible = !on;
        var P = fL.center, i;
        if (on) P.set(LH.subarray(N * 3), N * 3);
        else for (i = N; i < NT; i++) { P[i * 3] = 0; P[i * 3 + 1] = -100; P[i * 3 + 2] = 0; }
        balReset();
      }
      /* every frame on stage: the owl's head where the script has it (its control points turned, the lattice evaluated
         again only when they moved), its front half written as Gaussians for this eye; the rabbit's lattice carried by its
         pose (written again only when it moved) */
      var RW = FIG ? { pos: new Float32Array(FR.na * FR.nb * 3), su: new Float32Array(FR.na * FR.nb * 3), sv: new Float32Array(FR.na * FR.nb * 3), col: FR.grid.col } : null;
      function figDraw() {
        var th = owlTh(run.t), F = FO, P = F.patch.P, B = F.B, k, i, q, w, cs, sn, ex, ey, n;
        if (th !== F.th) {
          for (k = 0; k < F.hi.length; k++) {
            i = F.hi[k]; q = i * 3; w = F.W[i] * th; cs = Math.cos(w); sn = Math.sin(w); ex = B[q] - F.piv[0]; ey = B[q + 1] - F.piv[1];
            P[q] = F.piv[0] + ex * cs - ey * sn; P[q + 1] = F.piv[1] + ex * sn + ey * cs;
          }
          F.th = th; NB.patchGrid(F.patch, F.us, F.vs, F.grid);
        }
        rabbitPose(run.t, RP);
        var key = RP.T[1].toFixed(5) + ',' + Array.prototype.join.call(RP.M, ','), g = FR.grid, m = FR.na * FR.nb, M = RP.M, x, y, z, o;
        if (key !== FR.key) {
          FR.key = key;
          for (k = 0; k < m; k++) {
            o = k * 3; toWorld(RP, [g.pos[o], g.pos[o + 1], g.pos[o + 2]], PW); RW.pos[o] = PW[0]; RW.pos[o + 1] = PW[1]; RW.pos[o + 2] = PW[2];
            x = g.su[o]; y = g.su[o + 1]; z = g.su[o + 2]; for (i = 0; i < 3; i++) RW.su[o + i] = M[i * 3] * x + M[i * 3 + 1] * y + M[i * 3 + 2] * z;
            x = g.sv[o]; y = g.sv[o + 1]; z = g.sv[o + 2]; for (i = 0; i < 3; i++) RW.sv[o + i] = M[i * 3] * x + M[i * 3 + 1] * y + M[i * 3 + 2] * z;
          }
          n = G.writePlanoids(fF, 0, RW, FR.na, FR.nb, FR.du, FR.dv, 0.72, null, null);
          if (FR.flip < 0) for (k = 0; k < n * 3; k++) fF.axV[k] = -fF.axV[k];          // S_u x S_v out of the object, as the layers want it
          fig.nR = n;
        }
        n = G.writePlanoids(fF, fig.nR, F.grid, F.na, F.nb, F.du, F.dv, 0.72, null, { x: cam.position.x, y: cam.position.y, z: cam.position.z, flip: F.flip, min: -0.12 });
        fF.setCount(n); fF.commit(n); fig.nO = n - fig.nR;
      }
      var fig = { nO: 0, nR: 0 };

      /* ---------- moment: when does the ball touch? ----------
         Over the list, every sub-step measures, for the Gaussians near the ball's vertical: the gap between the
         ball's surface and each one's 3, 2 and 1 sigma ellipsoid (exact point-to-ellipsoid distance minus the
         radius), the gap to the nearest mean, and the summed density alpha exp(-m^2 / 2) at the ball's lowest point.
         A definition says "contact" at the first sub-step where its gap reaches 0 (or its density its level); the
         time is interpolated inside that sub-step. Over the spline: the bounce's sub-step is searched by bisection
         for the root of |S(u,v) - c| - r, with S(u,v) from the closest-point solve. */
      function dEll(px, py, pz, i, K) {                             // distance from a point (relative to the mean) to the K-sigma ellipsoid of Gaussian i
        var o = i * 9, s = i * 3, q0 = px * lax[o] + py * lax[o + 1] + pz * lax[o + 2], q1 = px * lax[o + 3] + py * lax[o + 4] + pz * lax[o + 5],
            q2 = px * lax[o + 6] + py * lax[o + 7] + pz * lax[o + 8], a0 = K * lsg[s], a1 = K * lsg[s + 1], a2 = K * lsg[s + 2], A0 = a0 * a0, A1 = a1 * a1, A2 = a2 * a2;
        if (q0 * q0 / A0 + q1 * q1 / A1 + q2 * q2 / A2 <= 1) return 0;
        // the nearest point is x_j = A_j q_j / (A_j + l), l the root of F(l) = sum (a_j q_j / (A_j + l))^2 - 1: F falls and is
        // convex, so Newton from below never overshoots; it starts at a_min |q| - A_max, where F is still >= 0
        var l = Math.max(0, Math.min(a0, a1, a2) * Math.sqrt(q0 * q0 + q1 * q1 + q2 * q2) - Math.max(A0, A1, A2)), it, t0, t1, t2, F;
        for (it = 0; it < 60; it++) {
          t0 = a0 * q0 / (A0 + l); t1 = a1 * q1 / (A1 + l); t2 = a2 * q2 / (A2 + l);
          F = t0 * t0 + t1 * t1 + t2 * t2 - 1; if (F < 1e-13) break;
          l += F / (2 * (t0 * t0 / (A0 + l) + t1 * t1 / (A1 + l) + t2 * t2 / (A2 + l)));
        }
        t0 = q0 * l / (A0 + l); t1 = q1 * l / (A1 + l); t2 = q2 * l / (A2 + l);
        return Math.sqrt(t0 * t0 + t1 * t1 + t2 * t2);
      }
      var gap = { v: [0, 0, 0, 0, 0, 0, 0], who: [-1, -1, -1, -1, -1, -1, -1] };   // per candidate: its measure now, and the Gaussian that decides it
      function listGaps(c) {
        var P = fL.center, A = fL.alpha, nr = mo.near, best = [9, 9, 9], who = [-1, -1, -1], gm = 9, wm = -1, rho = 0, yl = c[1] - BR, idx, i, o, px, py, pz, d, k, K, g;
        for (idx = 0; idx < nr.length; idx++) {
          i = nr[idx]; o = i * 3; px = c[0] - P[o]; py = c[1] - P[o + 1]; pz = c[2] - P[o + 2]; d = Math.sqrt(px * px + py * py + pz * pz);
          if (d - BR < gm) { gm = d - BR; wm = i; }
          for (k = 0; k < 3; k++) {
            K = 3 - k; if (d - K * lsm[i] - BR >= best[k]) continue;        // its ellipsoid lies inside a sphere of K s_max: it cannot be nearer
            g = dEll(px, py, pz, i, K) - BR; if (g < best[k]) { best[k] = g; who[k] = i; }
          }
          py = yl - P[o + 1];                                               // the density at the lowest point
          if (px * px + py * py + pz * pz < 25 * lsm[i] * lsm[i]) {
            var s = i * 3, q0 = (px * lax[i * 9] + py * lax[i * 9 + 1] + pz * lax[i * 9 + 2]) / lsg[s], q1 = (px * lax[i * 9 + 3] + py * lax[i * 9 + 4] + pz * lax[i * 9 + 5]) / lsg[s + 1],
                q2 = (px * lax[i * 9 + 6] + py * lax[i * 9 + 7] + pz * lax[i * 9 + 8]) / lsg[s + 2];
            rho += A[i] * Math.exp(-0.5 * (q0 * q0 + q1 * q1 + q2 * q2));
          }
        }
        gap.v[0] = best[0]; gap.v[1] = best[1]; gap.v[2] = best[2]; gap.v[3] = gap.v[4] = gap.v[5] = rho; gap.v[6] = gm;
        gap.who[0] = who[0]; gap.who[1] = who[1]; gap.who[2] = who[2]; gap.who[6] = wm;
      }
      function measureList(tNow, yPrev) {
        var i, prev = mo.val, cur, thr, f, up;
        listGaps(ball.p);
        if (prev) for (i = 0; i < CANDS.length; i++) {
          if (mo.t[i] != null) continue;
          cur = gap.v[i]; up = CANDS[i].lev != null; thr = up ? CANDS[i].lev : 0;
          if (up ? cur < thr : cur > thr) continue;
          f = KT.clamp((prev[i] - thr) / ((prev[i] - cur) || 1e-30), 0, 1);   // where in the sub-step it crossed
          mo.t[i] = tNow - HM + f * HM; mo.y[i] = yPrev + f * (ball.p[1] - yPrev) - BR; mo.who[i] = gap.who[i]; au.tick.push([i, mo.t[i]]);
        }
        mo.val = gap.v.slice();
      }
      function contactRoot(p0, v0, ts) {                             // the root of |S(u,v) - c(t)| - r inside one sub-step, by bisection
        var c = [0, 0, 0], lo = 0, hi = HM, it, f;
        function at(t) { c[0] = p0[0] + v0[0] * t; c[1] = p0[1] + v0[1] * t - 0.5 * GRAV * t * t; c[2] = p0[2] + v0[2] * t; return closest(c, mo.cpr).d - BR; }
        mo.cpr.ok = false; if (at(0) <= 0) return null;
        for (it = 0; it < 80 && hi - lo > 1e-16; it++) { var m = 0.5 * (lo + hi); if (at(m) > 0) lo = m; else hi = m; }
        f = at(0.5 * (lo + hi));
        return { t: ts + 0.5 * (lo + hi), res: Math.abs(f), u: mo.cpr.u, v: mo.cpr.v, p: c.slice(), y: c[1] - BR };
      }
      function substepMoment() {
        var t0 = run.t - run.tr, p0 = ball.p.slice(), v0 = ball.v.slice(), nb = hit.n, was = ball.on;
        run.t += HM; au.simT += HM;
        if (!ball.on && run.t >= run.hover) { ball.on = true; run.tr = run.t - HM; run.E0 = energy(); }
        if (ball.on) fly(HM);
        if (mo.side > 0) {
          if (ball.on) rigid(false, HM);
          if (hit.n > nb && !mo.root && was) mo.root = contactRoot(p0, v0, t0);
          if ((mo.root && run.t - run.tr - mo.root.t > 2.4) || run.t > 14) letGo(run.k + 1);
        } else {
          if (ball.on && ball.p[1] - BR < mo.top + 0.03 && ball.p[1] > mo.spot[1] - 0.6) {
            ballVsList(false);                                      // the Gaussians it overlaps, painted as in `ghost`
            measureList(run.t - run.tr, p0[1]);
          }
          if (ball.on && !mo.end && ball.p[1] < mo.spot[1] - 0.9) mo.end = run.t;
          if ((mo.end && run.t - mo.end > 1.2) || run.t > 14) letGo(run.k + 1);
        }
        pace();
      }
      /* SCRIPTED: how fast simulated time runs, from the ball's height D over the surface the list was made from (the
         same height over the spline): real time far away, 1/SLOW from 10 cm above to 5 cm below, eased between. On the
         spline, once the ball has bounced 4 cm clear, time speeds up again over 60 ms (it would bounce in the zone for long).
         Set after every sub-step, so the slowing starts in the right sub-step at any frame rate. */
      function slowmo(D) { return 1 - (1 - 1 / SLOW) * Math.min(KT.ease((0.24 - D) / 0.14), KT.ease((D + 0.17) / 0.12)); }
      function pace() {
        var c = [ball.p[0] + (mo.side < 0 ? 2 * X0 : 0), ball.p[1], ball.p[2]], s;
        closest(c, mo.cpm); mo.D = mo.cpm.sd - BR; s = slowmo(mo.D);
        if (mo.root) { if (!mo.left && mo.D > 0.04) mo.left = run.t; if (mo.left) s = Math.max(s, 1 / SLOW + (1 - 1 / SLOW) * KT.ease((run.t - mo.left) / 0.06)); }
        mo.slow = s;
      }
      function momentDrop(side) {                                   // a new approach: where it lands, and which Gaussians could decide it
        var q = DROPS.moment[0], x = side * X0 + q[0], z = q[2], c, i, o, dx, dz, r;
        mo.side = side; mo.root = null; mo.end = 0; mo.left = 0; mo.slow = 1;
        c = closest([q[0] + X0, YB + 0.05, z], mo.cps);
        mo.spot = [c.p[0] - (side < 0 ? 2 * X0 : 0), c.p[1], c.p[2]];
        if (side > 0) return;
        mo.val = null; mo.t = CANDS.map(function () { return null; }); mo.y = mo.t.slice(); mo.who = CANDS.map(function () { return -1; });
        mo.near = []; mo.top = -9;
        for (i = 0; i < N; i++) {
          o = i * 3; dx = home.c[o] - x; dz = home.c[o + 2] - z; r = BR + 3 * lsm[i] + 0.03;
          if (dx * dx + dz * dz < r * r && Math.abs(home.c[o + 1] - mo.spot[1]) < 0.4) { mo.near.push(i); mo.top = Math.max(mo.top, home.c[o + 1] + 3 * lsm[i]); }
        }
      }

      /* ---------- the script: every scene lets the ball go by itself, again and again ---------- */
      function letGo(k) {
        var m = st.sc.model, ph = m === 'ghost' ? GHOST[k % GHOST.length] : '', close = m === 'moment' || ph === 'close', was = run.close;
        var d = close ? DROPS.moment : DROPS[st.scene], q = d[k % d.length], side = m === 'moment' ? (k % 2 ? 1 : -1) : st.sc.side;
        run.k = k; run.t = 0; run.still = 0; run.ph = ph; run.close = close; run.rule = ph === 'loose' ? 1 : 0;
        run.hover = m === 'flight' || close ? 1.3 : was && m === 'ghost' ? 1.1 : 0.55;   // back from the close-up: the camera has the way out to travel
        cap.innerHTML = close ? SCENES.moment.cap : FIG && m === 'flight' ? st.sc.capF : st.sc.cap;
        root.classList.toggle('tc-close', close);                  // close up: the bowl fills the stage, under the caption too
        ball.p = [side * X0 + q[0], q[1], q[2]]; ball.v = [0, -q[3], 0]; ball.on = false; ball.rest = false;
        cp.ok = false; hit.n = 0; hit.J = 0; hit.t = -9; hit.ask = 0; hit.held = true; hit.vout = 0; tr.n = tr.k = 0;
        run.E0 = energy(); el.lost = 0;
        if (m === 'ghost' || m === 'flight' || m === 'moment') restoreList();
        if (surf.broken) mend();
        if (m === 'flight') { flatten(); derive(); fl.phase = 'drop'; fl.tt = fl.th = 0; fl.boost = false; }
        figShow(FIG && m === 'flight');
        if (FIG && m === 'flight') {                               // the ball is let go over the owl's head, from rest
          ball.p = [FO.hc[0] - 0.04, FO.top + 1.7, FO.hc[2]]; ball.v = [0, 0, 0]; run.E0 = energy();
          fl.t1 = fl.ta = null; fl.tb = 0; fl.miss = false; fl.hit = [null, null]; fl.cps = [newCp(), newCp()]; fl.loops++;
          FO.th = NaN; FR.key = '';                                // drawn afresh
          fl.tf = predictOwl();
        }
        if (close) momentDrop(side); else mo.slow = 1;             // (no slow motion is carried out of a close-up)
        au.first = true; au.lastN = 0;
      }
      /* SCRIPTED, the one thing in the finale that is: when the ball leaves the elastic surface, its velocity is
         set so that the parabola that follows comes down inside the list's bowl. The speed is the one the
         simulated bounce gave it whenever that is enough to cover the distance (otherwise the least speed that
         is); only the direction is aimed. From there on the flight is ballistic and the scatter is dynamics. */
      var AIM = [0.55, 0.42, 0.05];
      function launch() {
        var dx = -X0 + AIM[0] - ball.p[0], dy = AIM[1] - ball.p[1], dz = AIM[2] - ball.p[2], vs = speed(), T, best = 1e9, bt = 1.5, s, vy;
        fl.vsim = vs;
        for (T = 2.3; T > 1.2; T -= 0.01) {                         // the highest arc that needs no more than the measured speed
          vy = dy / T + 0.5 * GRAV * T; s = Math.sqrt((dx * dx + dz * dz) / (T * T) + vy * vy);
          if (s < best) { best = s; bt = T; }
          if (s <= vs) { bt = T; best = s; break; }
        }
        fl.boost = Math.abs(best - vs) > 0.01 * vs;                 // the measured speed could not be kept
        ball.v = [dx / bt, dy / bt + 0.5 * GRAV * bt, dz / bt]; fl.vset = speed(); fl.phase = 'air'; fl.tl = run.t; el.touch = false; el.f = 0;
        au.launch = fl.vset;
      }
      function substep() {
        var m = model(), s;
        if (m === 'moment') { for (s = 0; s < 8 && run.close; s++) substepMoment(); return; }   // (ghost's close-up may hand back to its pass inside)
        run.t += H; au.simT += H;
        if (!ball.on && run.t >= run.hover) { ball.on = true; run.tr = run.t - H; run.E0 = energy(); }
        BP0[0] = ball.p[0]; BP0[1] = ball.p[1]; BP0[2] = ball.p[2]; BV0[0] = ball.v[0]; BV0[1] = ball.v[1]; BV0[2] = ball.v[2];   // (flight's contacts look back into the sub-step)
        if (ball.on) fly(H);
        if (m === 'ghost') {
          if (ball.on) ballVsList(run.rule === 1);
          if (run.t > (run.rule ? 5.2 : 3.0)) letGo(run.k + 1);
        } else if (m === 'solid') {
          if (ball.on) rigid(false);
          run.still = ball.rest && speed() < 0.03 ? run.still + H : 0;
          if (run.still > 1.8 || run.t > 18 || ball.p[1] < -6) letGo(run.k + 1);
        } else if (m === 'brittle') {
          if (ball.on && !surf.broken) rigid(true);
          if (surf.broken ? run.t - surf.tb > 3.6 : run.t > (run.k % 2 ? 4.4 : 2.5)) letGo(run.k + 1);
        } else if (m === 'elastic') {
          membrane(true);
          s = speed(); run.still = s < 0.12 && el.touch ? run.still + H : 0;
          if (run.still > 1.3 || run.t > 9.5 || ball.p[1] < -6) letGo(run.k + 1);
        } else if (FIG) {                                          // flight, with the cast
          if (ball.on && fl.t1 == null) fl.t1 = run.tr + fl.tf;    // the nod is timed to the predicted contact
          if (fl.phase === 'drop' && ball.on) meet(0);
          else if (fl.phase === 'air1') { meet(1); if (fl.phase === 'air1' && run.t > fl.ta + 0.5) { fl.phase = 'air2'; fl.miss = true; } }
          balloons();
          if (fl.phase === 'air2' && bal.inside) { fl.phase = 'bowl'; fl.tb = run.t; fl.vin = speed(); au.puff = bal.reach; }
          if ((fl.phase === 'bowl' && run.t - fl.tb > 4.6) || run.t > 18 || (fl.phase !== 'bowl' && ball.p[1] < -12)) letGo(0);   // the lists have time to settle
        } else {
          if (fl.phase === 'drop') {
            membrane(true);
            if (el.touch && !fl.tt) fl.tt = run.t;
            if (fl.tt && ((!el.touch && ball.v[1] > 0) || run.t - fl.tt > 1)) launch();
          } else {
            membrane(false);
            if (ball.on) { s = speed(); if (ballVsList(true) && fl.phase === 'air') { fl.phase = 'hit'; fl.th = run.t; fl.vin = s; } }
            if ((fl.phase === 'hit' && run.t - fl.th > 5) || run.t > 16) letGo(0);
          }
        }
        if (prt.n) particles();
        if (m === 'flight' && fl.phase !== 'drop' && ++tr.k % 6 === 0 && ball.p[1] > -8) { tr.p.set(ball.p, (tr.n % TR) * 3); tr.n++; }
      }

      /* ---------- sound: one frame's worth of events, heard ---------- */
      var chRoll = SFX.channel('roll'), chHiss = SFX.channel('hiss'), chWob = SFX.channel('wobble');
      var SINW = []; (function () { for (var i = 0; i < NC; i++) SINW.push(Math.sin(Math.PI * i / (NC - 1))); })();   // the membrane's first mode
      function quiet() { au.simT = au.J = au.Jn = au.rest = au.restV = au.inside = au.calls = au.hits = au.jl = au.crack = au.pluck = au.launch = au.glass = au.puff = 0; au.tick.length = 0; }
      function silence() { chRoll.stop(); chHiss.stop(); chWob.stop(); SFX.hush(); au.env = 0; au.rat = 0; }
      function listen(dt) {
        var m = model(), s, pan = 0, x, w;
        if (!SFX.ok()) { quiet(); chRoll.stop(); chHiss.stop(); chWob.stop(); au.lastN = prt.n; if (prt.n || prt.nSeen) au.first = false; return; }
        s = G.project(cam, ball.p[0], ball.p[1], ball.p[2], size.w, size.h);
        if (s) pan = KT.clamp((s[0] / size.w - 0.5) * 0.9, -0.45, 0.45);
        var stretch = m === 'moment' ? 1 / mo.slow : 1;
        if (au.Jn) SFX.knock(au.J, au.tone, pan, stretch);                           // a bounce on the spline
        if (au.crack) {                                                               // it broke
          SFX.crack(au.crack, frag.n, pan);
          SFX.shards(frag.list.map(function (q) { return { n: q.n, sp: Math.hypot(q.v[0], q.v[1], q.v[2]), rho: q.rho }; }), au.crack, pan);
        }
        x = au.rest ? au.restV / au.rest : 0;                                         // rolling: the share of the frame in resting contact, at its speed
        chRoll.set(au.rest && au.simT ? 0.11 * Math.pow(KT.clamp(x / 2.5, 0, 1), 1.2) * Math.min(1, au.rest / au.simT) : 0, 200 + 130 * x);
        x = au.calls ? au.inside / au.calls : 0;                                       // sinking through a list with no rule
        chHiss.set(x > 0 ? 0.13 * Math.pow(KT.clamp(x / 100, 0, 1), 0.6) * KT.clamp(speed() / 7, 0.5, 1.2) : 0, (1200 + 90 * speed()) / Math.pow(stretch, 0.3));
        if (au.hits && au.first) { SFX.scatter(au.jl, au.hits, pan); au.first = false; au.lastN = prt.n; au.ratT = SFX.now(); }   // the ball meets loose particles
        else if (!au.first) {                                                         // and every Gaussian knocked loose since rattles
          x = prt.n - au.lastN; if (x < 0) x = 0; au.lastN = prt.n; au.rat += x;
          if (au.rat && SFX.now() - au.ratT > 0.06) { SFX.rattle(au.rat, pan); au.rat = 0; au.ratT = SFX.now(); }
        }
        if (au.pluck > 0.2) SFX.pluck(au.pluck, mat.stiff, pan);                      // a contact begins on the springs
        if (au.glass) SFX.glass(au.glass, pan);                                       // the glass rabbit heads the ball
        if (au.puff) SFX.puff(au.puff, pan);                                          // the lists in the bowl give way
        if (m === 'elastic' || (m === 'flight' && !FIG)) {                            // and the net rings: loudness from its motion, the wobble from its first mode
          var e2 = 0, vm = 0, ws = 0, q, i, j;
          for (j = 1; j < NC - 1; j++) for (i = 1; i < NC - 1; i++) {
            q = (j * NC + i) * 3; e2 += MV[q] * MV[q] + MV[q + 1] * MV[q + 1] + MV[q + 2] * MV[q + 2];
            w = SINW[i] * SINW[j]; vm += w * MV[q + 1]; ws += w;
          }
          x = Math.sqrt(e2 / ((NC - 2) * (NC - 2)));                                  // rms speed of the control points, m/s: loudness ~ their kinetic energy
          au.env = Math.max(x, au.env * Math.exp(-dt / 0.15));
          chWob.set(au.env > 0.08 ? 0.11 * Math.pow(KT.clamp(au.env / 1.4, 0, 1), 2) : 0, FSPR * Math.sqrt(mat.stiff / 160) * (1 + 0.05 * KT.clamp(vm / ws / 0.6, -1, 1)));
        } else { chWob.set(0); au.env = 0; }
        if (au.launch) SFX.whoosh(au.launch, pan);                                    // out of the bowl
        if (au.tick.length) {                                                         // moment: a definition says "contact", heard when it
          var T1 = run.t - run.tr, T0 = T1 - au.simT, last = -1;                      // did within the frame (a frame late), 15 ms apart at least
          au.tick.forEach(function (q) { last = Math.max(au.simT > 0 ? KT.clamp((q[1] - T0) / au.simT, 0, 1) * dt : 0, last + 0.015); SFX.tick(q[0], pan, last); });
        }
        quiet();
      }

      /* ---------- colours: the checker of knot spans, the ball's shadow, and red where things go wrong ---------- */
      var ERR = G.rgb(C.err);
      /* moment: the Gaussians that decided a definition are taken 60 % of the way towards a mark that stands out from the
         page behind the see-through list: near white on the dark stage, the text colour on a light page */
      var MARK = KT.theme === 'light' ? G.rgb(C.text).map(function (v) { return 0.6 * v; }) : [0.6, 0.58, 0.56];
      function paint() {
        var bx = ball.p[0], by = ball.p[1], bz = ball.p[2], k, o, dx, dy, dz, t, d2, s, w, f, P, cl, L = LIGHT, ll = Math.hypot(L[0], L[1], L[2]);
        var lx = L[0] / ll, ly = L[1] / ll, lz = L[2] / ll, m = model(), heat = surf.broken ? Math.exp(-(run.t - surf.tb) / 0.7) : 0, q, r;
        if (fR.mesh.visible) {                                    // (not while flight's cast stands where the spline bowl was)
          P = fR.center; cl = fR.color;
          for (k = 0; k < N; k++) {
            o = k * 3; dx = bx - P[o]; dy = by - P[o + 1]; dz = bz - P[o + 2]; t = dx * lx + dy * ly + dz * lz; f = 1; r = 0;
            if (t > 0 && t < 6) { d2 = dx * dx + dy * dy + dz * dz - t * t; s = BR * (0.85 + 0.22 * t); f = 1 - 0.62 * Math.exp(-d2 / (s * s)) / (1 + 0.3 * t); }
            if (surf.broken) { q = frag.list[frag.id[k]]; f *= q.shade; r = q.heat * heat * 0.8; }
            cl[o] = baseR[o] * f * (1 - r) + ERR[0] * r; cl[o + 1] = baseR[o + 1] * f * (1 - r) + ERR[1] * r; cl[o + 2] = baseR[o + 2] * f * (1 - r) + ERR[2] * r;
          }
          fR.commit(N);
        }
        {
          P = fL.center; cl = fL.color; var V = prt.v, in2 = BR * BR, pass = (m === 'ghost' && !run.rule) || (m === 'moment' && mo.side < 0);
          for (k = 0; k < N; k++) {
            o = k * 3; dx = bx - P[o]; dy = by - P[o + 1]; dz = bz - P[o + 2]; t = dx * lx + dy * ly + dz * lz; f = 1; r = 0; d2 = dx * dx + dy * dy + dz * dz;
            if (t > 0 && t < 6) { s = BR * (0.85 + 0.22 * t); f = 1 - 0.62 * Math.exp(-(d2 - t * t) / (s * s)) / (1 + 0.3 * t); }
            if (prt.aw[k]) r = Math.min(0.85, 0.3 + 0.25 * Math.hypot(V[o], V[o + 1], V[o + 2]));
            else if (pass && prt.seen[k]) r = d2 < in2 * 1.3 ? 0.9 : 0.45;        // inside the ball right now, or was
            cl[o] = baseL[o] * f * (1 - r) + ERR[0] * r; cl[o + 1] = baseL[o + 1] * f * (1 - r) + ERR[1] * r; cl[o + 2] = baseL[o + 2] * f * (1 - r) + ERR[2] * r;
          }
          if (m === 'moment' && mo.side < 0) mo.who.forEach(function (i) {          // the Gaussians that decided a definition: lit up (inked, on a light page)
            if (i < 0) return; o = i * 3; cl[o] = 0.4 * cl[o] + MARK[0]; cl[o + 1] = 0.4 * cl[o + 1] + MARK[1]; cl[o + 2] = 0.4 * cl[o + 2] + MARK[2];
          });
        }
      }

      /* ---------- what is drawn about the contact ---------- */
      function overlays() {
        var m = model(), onSpline = m === 'solid' || m === 'elastic' || (m === 'brittle' && !surf.broken) || (m === 'flight' && !FIG && fl.phase === 'drop') || (m === 'moment' && mo.side > 0);
        var soft = m === 'elastic' || (m === 'flight' && !FIG), k, i, o, q, bm = ballMesh.material, see = m === 'moment';
        ballMesh.position.set(ball.p[0], ball.p[1], ball.p[2]);
        if (bm.transparent !== see) { bm.transparent = see; bm.opacity = see ? 0.3 : 1; bm.depthWrite = !see; bm.needsUpdate = true; }   // close up: see under the ball
        dot.scale.setScalar(see ? 0.35 : 1);
        low.visible = see && ball.on; low.position.set(ball.p[0], ball.p[1] - BR, ball.p[2]);
        if (onSpline) closest(ball.p, cp);
        var near = onSpline && cp.d < BR + 0.75 && (soft || !cp.edge || cp.d < BR * 1.6), since = run.t - hit.t;
        iso.obj.visible = dot.visible = arrow.obj.visible = drop.obj.visible = near;
        var pinAt = near ? cp.p : null, pinU = cp.u, pinV = cp.v;
        if (FIG && m === 'flight') { var fo = figMarks(); if (fo) { pinAt = fo.p; pinU = fo.u; pinV = fo.v; } }
        if (near) {
          var up = 0.012, n = cp.nc, len;
          for (i = 0; i < NS; i++) for (k = 0; k < 2; k++) {
            evalS(cp.u, (i + k) / NS, E); o = (i * 2 + k) * 3; iso.arr[o] = E.p[0] + n[0] * up; iso.arr[o + 1] = E.p[1] + n[1] * up; iso.arr[o + 2] = E.p[2] + n[2] * up;
            evalS((i + k) / NS, cp.v, E); o += NS * 6; iso.arr[o] = E.p[0] + n[0] * up; iso.arr[o + 1] = E.p[1] + n[1] * up; iso.arr[o + 2] = E.p[2] + n[2] * up;
          }
          iso.set(4 * NS);
          dot.position.set(cp.p[0], cp.p[1], cp.p[2]);
          drop.arr.set(ball.p, 0); drop.arr.set(cp.p, 3); drop.set(2);
          len = soft ? 0.5 + Math.min(1.1, el.f / 260) : 0.55 + (since < 0.45 ? Math.min(1.2, hit.J * 0.085) * (1 - since / 0.45) : 0);
          if (m === 'moment') len *= 0.45;                          // close up: the normal at a size that leaves the contact in view
          arrow.aim(cp.p, n[0], n[1], n[2], len, soft ? 1 + Math.min(1, el.f / 300) : since < 0.45 ? 1.6 : 1);
        }
        net.obj.visible = cpts.visible = stem.obj.visible = soft; sup.visible = soft && el.touch;
        if (soft) {
          var P = patch.P, a = cpts.geometry.attributes.position, v = 0, j;
          a.array.set(P); a.needsUpdate = true;
          for (j = 0; j < NC; j++) for (i = 0; i < NC; i++) {
            q = (j * NC + i) * 3;
            if (i + 1 < NC) { for (k = 0; k < 3; k++) { net.arr[v * 3 + k] = P[q + k]; net.arr[v * 3 + 3 + k] = P[q + 3 + k]; } v += 2; }
            if (j + 1 < NC) { for (k = 0; k < 3; k++) { net.arr[v * 3 + k] = P[q + k]; net.arr[v * 3 + 3 + k] = P[q + NC * 3 + k]; } v += 2; }
          }
          net.set(v);
          for (v = 0, q = 0; q < NC * NC * 3; q += 3) if (Math.abs(MX[q]) + Math.abs(MX[q + 1]) + Math.abs(MX[q + 2]) > 0.006) {
            for (k = 0; k < 3; k++) { stem.arr[v * 3 + k] = P0[q + k]; stem.arr[v * 3 + 3 + k] = P[q + k]; } v += 2;
          }
          stem.set(v);
          if (el.touch) {
            a = sup.geometry.attributes.position;
            el.sup.forEach(function (s, k) { q = (s.j * NC + s.i) * 3; a.array[k * 3] = P[q]; a.array[k * 3 + 1] = P[q + 1]; a.array[k * 3 + 2] = P[q + 2]; });
            a.needsUpdate = true; sup.geometry.setDrawRange(0, el.sup.length);
          }
        }
        trail.obj.visible = m === 'flight' && tr.n > 1;
        if (trail.obj.visible) {
          var n0 = Math.max(0, tr.n - TR), c = 0;
          for (i = n0; i < tr.n - 1; i++) { for (k = 0; k < 3; k++) { trail.arr[c * 3 + k] = tr.p[(i % TR) * 3 + k]; trail.arr[c * 3 + 3 + k] = tr.p[((i + 1) % TR) * 3 + k]; } c += 2; }
          trail.set(c);
        }
        // moment: a short mark, facing the eye, at each height where a definition said "contact"
        mkL.obj.visible = mkS.obj.visible = mkG.obj.visible = m === 'moment';
        if (m === 'moment') {
          var ce = cam.matrixWorld.elements, rx = ce[0], ry = ce[1], rz = ce[2], h = 0.07, xl = -X0 + DROPS.moment[0][0], z0 = DROPS.moment[0][2], cnt = 0, yt = -9, yb = 9;
          var mark = function (L, at, x, y, z, w) { L.arr[at] = x - rx * w; L.arr[at + 1] = y - ry * w; L.arr[at + 2] = z - rz * w; L.arr[at + 3] = x + rx * w; L.arr[at + 4] = y + ry * w; L.arr[at + 5] = z + rz * w; };
          for (i = 0; i < CANDS.length; i++) if (mo.t[i] != null) { mark(mkL, cnt * 6, xl, mo.y[i], z0, h); cnt++; yt = Math.max(yt, mo.y[i]); yb = Math.min(yb, mo.y[i]); }
          mkL.set(cnt * 2);
          if (mo.root) { mark(mkS, 0, mo.root.p[0], mo.root.y, mo.root.p[2], h * 1.3); mkS.set(2); } else mkS.set(0);
          if (cnt > 1) { mkG.arr.set([xl, yt + 0.03, z0, xl, yb - 0.03, z0]); mkG.set(2); } else mkG.set(0);
        }
        // the address, written next to the contact
        var s = pinAt ? G.project(cam, pinAt[0], pinAt[1], pinAt[2], size.w, size.h) : null;
        if (s && s[0] > 40 && s[0] < size.w - 150 && s[1] > 40 && s[1] < size.h - 30) {
          pin.style.display = 'block'; pin.style.transform = 'translate(' + (s[0] + 16).toFixed(1) + 'px,' + (s[1] + 10).toFixed(1) + 'px)';
          setText(pin, '(u, v) = (' + pinU.toFixed(2) + ', ' + pinV.toFixed(2) + ')');
        } else pin.style.display = 'none';
      }
      /* flight: on the figure the ball is coming down on, while it is near, the closest point S(u, v), the line to the ball's
         centre, the normal and the chart's two parameter lines through the point; for a moment after a header, the contact
         itself (its normal drawn longer with the impulse). -> { p, u, v } for the address pin, or null */
      var IE = { p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0] }, IW = [0, 0, 0];
      function figMarks() {
        var who = fl.phase === 'drop' ? 0 : fl.phase === 'air1' ? 1 : -1, rec = null, o = null, P, Nn, u, v, F, th, k, i, a, since = 9, len;
        for (k = 0; k < 2; k++) if (fl.hit[k] && run.t - fl.hit[k].at < 0.7) { rec = fl.hit[k]; since = run.t - rec.at; }
        if (rec) { who = rec.who; P = rec.p; Nn = rec.n; u = rec.u; v = rec.v; }
        else if (who >= 0 && (o = fl.cps[who]) && o.near && o.ok && o.wp && o.d < BR + 0.75) { P = o.wp; Nn = o.wn; u = o.u; v = o.v; }
        else return null;
        F = who ? FR : FO; th = who ? 0 : owlTh(run.t); if (who) rabbitPose(run.t, RP);
        iso.obj.visible = dot.visible = arrow.obj.visible = drop.obj.visible = true;
        for (i = 0; i < NS; i++) for (k = 0; k < 2; k++) for (a = 0; a < 2; a++) {     // a = 0: along u at this v; a = 1: along v at this u
          var x = (i + k) / NS - 0.5;
          evalF(F, a ? u : KT.clamp(u + 0.24 * x, 0, 1), a ? KT.clamp(v + 0.2 * x, F.vlo, 1) : v, IE, th);
          if (who) toWorld(RP, IE.p, IW); else { IW[0] = IE.p[0]; IW[1] = IE.p[1]; IW[2] = IE.p[2]; }
          var q = (a * NS * 2 + i * 2 + k) * 3; iso.arr[q] = IW[0] + Nn[0] * 0.012; iso.arr[q + 1] = IW[1] + Nn[1] * 0.012; iso.arr[q + 2] = IW[2] + Nn[2] * 0.012;
        }
        iso.set(4 * NS);
        dot.position.set(P[0], P[1], P[2]);
        drop.arr.set(rec ? rec.c : ball.p, 0); drop.arr.set(P, 3); drop.set(2);
        len = 0.45 + (rec ? Math.min(1, rec.J * 0.06) * (1 - since / 0.7) : 0);
        arrow.aim(P, Nn[0], Nn[1], Nn[2], len, rec ? 1.5 : 1);
        return { p: P, u: u, v: v };
      }
      function setText(e, html) { if (e._h !== html) { e._h = html; e.innerHTML = html; } }

      /* ---------- moment: the timeline of the candidate moments (ms after release), the list's and the spline's ---------- */
      var tl = { key: '', w: 420, cur: null, a: 0, b: 1, x0: 52 };
      function timeline() {
        var on = model() === 'moment', spl = st.sc.model === 'moment', Tt = mo.t, idx = [], i, lo = 1e9, hi = -1e9, r = mo.root;
        tlm.classList.toggle('on', on);
        if (!on) return;
        for (i = 0; i < CANDS.length; i++) if (Tt[i] != null) { idx.push(i); lo = Math.min(lo, Tt[i]); hi = Math.max(hi, Tt[i]); }
        var W = tl.w, L0 = tl.x0, R1 = W - 12;
        function ln(x1, x2, y1, y2, col, wd, op) { return '<line x1="' + x1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y1="' + y1 + '" y2="' + y2 + '" stroke="' + col + '"' + (wd ? ' stroke-width="' + wd + '"' : '') + (op ? ' stroke-opacity="' + op + '"' : '') + '/>'; }
        function tx(x, y, col, s, anc) { return '<text x="' + x.toFixed(1) + '" y="' + y + '" fill="' + col + '"' + (anc ? ' text-anchor="' + anc + '"' : '') + '>' + s + '</text>'; }
        var rows = '<svg width="' + W + '" height="84" viewBox="0 0 ' + W + ' 84">' + ln(L0, R1, 70, 70, C.mute, 0, 0.6) + tx(0, 82, C.mute, 'ms') +
                   tx(0, 35, C.splat, T('list')) + ln(L0, R1, 31, 31, C.line, 1) +
                   (spl ? tx(0, 58, C.surf, T('spline')) + ln(L0, R1, 54, 54, C.line, 1) : '');   // ghost's close-up has no spline drop: no row for it
        if (!idx.length) {                                          // nothing has said "contact" yet: the empty rows, at their full height
          if (tl.key !== 'none,' + spl + W) { tl.key = 'none,' + spl + W; tlm.innerHTML = rows + '</svg>'; tl.cur = null; }
          return;
        }
        var llo = lo, lhi = hi;
        if (r) { lo = Math.min(lo, r.t); hi = Math.max(hi, r.t); }
        var key = idx.join(',') + (r ? ',r' : '') + ',' + spl + W, pad = Math.max(0.0015, 0.1 * (hi - lo));
        function X(t) { return L0 + (t - tl.a) / (tl.b - tl.a) * (R1 - L0); }
        if (key !== tl.key) {
          tl.key = key; tl.a = lo - pad; tl.b = hi + pad;
          var span = (tl.b - tl.a) * 1000, steps = [0.5, 1, 2, 5, 10, 20, 50, 100], sv = 100, k, x, ends = [-99, -99];
          for (k = 0; k < steps.length; k++) if (span / steps[k] <= Math.max(3, Math.floor((R1 - L0) / 48))) { sv = steps[k]; break; }
          var s = rows;
          for (k = Math.ceil(tl.a * 1000 / sv) * sv; k <= tl.b * 1000 + 1e-9; k += sv) { x = X(k / 1000); s += ln(x, x, 67, 73, C.mute) + tx(x, 82, C.mute, +k.toFixed(1), 'middle'); }
          idx.slice().sort(function (p, q) { return Tt[p] - Tt[q]; }).forEach(function (c) {
            var x = X(Tt[c]), lab = CANDS[c].lab, w = lab.length * 6 + 6, row = x - w / 2 > ends[0] ? 0 : x - w / 2 > ends[1] ? 1 : ends[0] <= ends[1] ? 0 : 1;
            ends[row] = x + w / 2;
            s += ln(x, x, 25, 37, C.err, 2) + tx(x, row ? 9 : 20, C.err, lab, 'middle');
          });
          if (idx.length > 1) s += ln(X(llo), X(lhi), 42, 42, C.err, 1, 0.7) + ln(X(llo), X(llo), 39, 45, C.err, 1, 0.7) + ln(X(lhi), X(lhi), 39, 45, C.err, 1, 0.7);
          if (r) { x = X(r.t); s += ln(x, x, 48, 60, C.surf, 2) + tx(x + 6 > R1 - 40 ? x - 6 : x + 6, 58, C.surf, T('one root'), x + 6 > R1 - 40 ? 'end' : ''); }
          s += '<line class="kt-mo-now" x1="0" x2="0" y1="2" y2="62" stroke="' + C.text + '" stroke-opacity="0.6"/></svg>';
          tlm.innerHTML = s; tl.cur = tlm.querySelector('.kt-mo-now');
        }
        var now = run.t - run.tr, xn = X(now);
        if (tl.cur) { tl.cur.setAttribute('x1', xn.toFixed(1)); tl.cur.setAttribute('x2', xn.toFixed(1)); tl.cur.style.display = ball.on && now >= tl.a && now <= tl.b ? '' : 'none'; }
      }

      /* ---------- the words: every number is read off the simulation ---------- */
      var NUM = N.toLocaleString('en-US'), LIST = T('<b>a list</b> · {n} Gaussians, unordered <i>(emulated)</i>', { n: NUM }) + '<br>';
      var SPL = T('<b>a spline chart</b> · {c} × {c} control points → {n} Gaussians', { c: NC, n: NUM }) + '<br>';
      function pct(x) { return T('{n} %', { n: (100 * x).toFixed(Math.abs(1 - x) < 0.02 ? 2 : 0).replace('-', '−') }); }   // below zero: lower than at rest in the bowl
      function ms(t) { return (1000 * t).toFixed(1) + ' ms'; }
      function sci(x) { if (!(x > 0)) return '0'; var e = Math.floor(Math.log10(x)); return (x / Math.pow(10, e)).toFixed(1) + ' × 10<sup>' + String(e).replace('-', '−') + '</sup>'; }
      /* one template per statement: a line is statements joined by " · ", in the same order in every language */
      function words() {
        var m = model(), s, e = energy() / run.E0, addr = T('closest point S(<em>{u}</em>, <em>{v}</em>)', { u: cp.u.toFixed(2), v: cp.v.toFixed(2) });
        if (m === 'ghost') {
          var dv = Math.hypot(ball.v[0], ball.v[1] + (ball.on ? GRAV * (run.t - run.tr) : 0), ball.v[2]);
          s = LIST + (run.rule
            ? T('rule: every Gaussian is a loose particle · the ball touched <em class="bad">{seen}</em> · <em class="bad">{loose}</em> knocked loose · it lost {dv} m/s<br>a Gaussian overlaps {ov} others where it lies: as solid particles they push each other apart',
                { seen: prt.nSeen.toLocaleString('en-US'), loose: prt.n.toLocaleString('en-US'), dv: dv.toFixed(2), ov: overlaps.toFixed(1) })
            : T('no rule: the ball overlapped <em class="bad">{seen}</em> Gaussians and touched <em class="bad">0</em> · its velocity changed by {dv} m/s<br>a list has a density, not a boundary: a contact needs a threshold, or a mesh extracted first',
                { seen: prt.nSeen.toLocaleString('en-US'), dv: dv.toFixed(2) }));
        } else if (m === 'moment') {
          var nf = 0, lo = 1e9, hi = -1e9, ylo = 0, yhi = 0, slow = mo.slow < 0.985 ? ' · ' + T('time slowed ×{k} <i>(scripted)</i>', { k: Math.round(1 / mo.slow) }) : '';
          mo.t.forEach(function (t, i) { if (t == null) return; nf++; if (t < lo) { lo = t; yhi = mo.y[i]; } if (t > hi) { hi = t; ylo = mo.y[i]; } });
          if (mo.side < 0 || !mo.root) {
            s = mo.side < 0 ? LIST : SPL;
            if (mo.side < 0) s += nf
              ? T('{n} of {m} definitions say “contact” · from {a} to {b} after release · spread <em class="bad">{d}</em>, {cm} cm of fall',
                  { n: nf, m: CANDS.length, a: ms(lo), b: ms(hi), d: ms(hi - lo), cm: (100 * (yhi - ylo)).toFixed(1) })
              : T('when does the ball touch it? A list does not say, so each definition is measured as the ball comes down');
            else s += T('contact is the root of |S(u, v) − c| = r, with S(u, v) the closest point · nothing to choose');
            s += '<br>' + (mo.side < 0 ? T('σ: shell of the nearest Gaussian · ρ: summed density at the ball’s lowest point · μ: nearest mean · for this emulated list') : T('the list’s {n} answers stay on the timeline', { n: nf })) + slow;
          } else {
            var r = mo.root;
            s = SPL + T('contact: |S(u, v) − c| = r at <em>{t}</em> after release, at S(<em>{u}</em>, <em>{v}</em>) · residual <em>{res} m</em>', { t: ms(r.t), u: r.u.toFixed(2), v: r.v.toFixed(2), res: sci(r.res) }) +
                '<br>' + T('one definition, one time') + (nf > 1 ? ' · ' + T('the list’s {n} answers spread over {d}', { n: nf, d: ms(hi - lo) }) : '') + slow;
          }
        } else if (m === 'solid') {
          s = SPL + T('{addr} · distance <em>{d} m</em> · normal n = S<sub>u</sub> × S<sub>v</sub>', { addr: addr, d: Math.max(0, cp.d - BR).toFixed(2) }) + '<br>' +
            (hit.n ? (ball.rest ? T('rolling on the surface after {n} bounces', { n: hit.n }) : T('bounce {n}: impulse <em>{J} N·s</em> along n', { n: hit.n, J: hit.J.toFixed(1) })) : T('ideal rigid surface')) +
            ' · ' + T('restitution e = {e} · energy <em>{p}</em> of the drop', { e: mat.e.toFixed(2), p: pct(e) });
        } else if (m === 'brittle') {
          s = SPL + (surf.broken
            ? T('the impact asked <em class="bad">{ask} N·s</em> &gt; toughness {tough} N·s: <em class="bad">broken</em> into <em>{n}</em> pieces of the (u, v) domain<br>each piece is a rigid body and its Gaussians ride on it · the ball kept {v} m/s',
                { ask: hit.ask.toFixed(1), tough: mat.tough.toFixed(1), n: frag.n, v: hit.vout.toFixed(1) })
            : (hit.n ? T('the impact asked <em>{ask} N·s</em> ≤ toughness {tough} N·s: <em>it holds</em>', { ask: hit.ask.toFixed(1), tough: mat.tough.toFixed(1) })
                     : T('toughness {tough} N·s: the largest impulse the surface can return', { tough: mat.tough.toFixed(1) })) +
              '<br>' + T('{addr} · impulse asked = m (1 + e) |v · n|', { addr: addr }));
        } else if (m === 'elastic') {
          var em = (membraneEnergy() + 0.5 * KC * el.pen * el.pen) / run.E0;       // the springs of the net, and the contact spring
          s = SPL + T('control points on damped springs, {k} N/m', { k: mat.stiff.toFixed(0) }) + ' · ' +
            (el.touch ? T('contact force <em>{f} N</em> on {n} control points, by basis weight', { f: el.f.toFixed(0), n: el.sup.length }) : T('the Gaussians are derived again every frame')) +
            '<br>' + T('energy: ball <em>{b}</em> · surface <em>{s}</em> · damped {d}', { b: pct(e), s: pct(em), d: pct(el.lost / run.E0) }) + (hit.n ? ' · ' + T('bounce {n}', { n: hit.n }) : '');
        } else if (FIG) {                                          // flight, with the cast
          var h0 = fl.hit[0], h1 = fl.hit[1], o0 = fl.cps[0];
          if (fl.phase === 'drop') {
            s = T('<b>two spline charts</b> · the owl, {a} × {b} control points', { a: FO.nu, b: FO.nv }) + ' · ' +
              (FR.glass ? T('the rabbit, {a} × {b}, in clear glass', { a: FR.nu, b: FR.nv }) : T('the rabbit, {a} × {b}', { a: FR.nu, b: FR.nv })) + ' · ' +
              T('{n} Gaussians derived', { n: (fig.nO + fig.nR).toLocaleString('en-US') }) + '<br>' +
              (!ball.on ? T('the ball is let go over the owl’s head')
                : o0 && o0.near && o0.ok ? T('closest point on the owl S(<em>{u}</em>, <em>{v}</em>) by Gauss–Newton in (u, v) · {d} m to go', { u: o0.u.toFixed(2), v: o0.v.toFixed(2), d: Math.max(0, o0.d - BR).toFixed(2) })
                : T('falling at {v} m/s', { v: speed().toFixed(1) }));
          } else if (fl.phase === 'air1') {
            s = T('header 1 · the ball met the owl at S(<em>{u}</em>, <em>{v}</em>), {t} after release · |S − c| − r = <em>{g} m</em> · impulse {J} N·s',
                  { u: h0.u.toFixed(2), v: h0.v.toFixed(2), t: ms(h0.t), g: sci(Math.abs(h0.gap)), J: h0.J.toFixed(1) }) + '<br>' +
              T('computed: where and when they meet · scripted: the owl’s nod, and the ball sent to the rabbit at {v} m/s', { v: Math.hypot(h0.vout[0], h0.vout[1], h0.vout[2]).toFixed(1) });
          } else if (fl.phase === 'air2') {
            s = (h1 ? T('header 2 · the ball met the rabbit at S(<em>{u}</em>, <em>{v}</em>), {t} after release · |S − c| − r = <em>{g} m</em> · impulse {J} N·s',
                        { u: h1.u.toFixed(2), v: h1.v.toFixed(2), t: ms(h1.t), g: sci(Math.abs(h1.gap)), J: h1.J.toFixed(1) })
                    : T('the ball missed the rabbit')) + '<br>' +
              (h1 ? T('computed: where and when they meet · scripted: the rabbit’s jump and nod, and the ball sent into the bowl at {v} m/s', { v: Math.hypot(h1.vout[0], h1.vout[1], h1.vout[2]).toFixed(1) })
                  : T('speed {v} m/s', { v: speed().toFixed(1) }));
          } else {
            s = T('<b>three lists</b> · the bowl, the owl and the rabbit · {n} Gaussians, unordered <i>(emulated)</i>', { n: NT.toLocaleString('en-US') }) + '<br>' +
              T('the ball passes through · <em>{k}</em> Gaussians drift aside, the farthest {d} cm · <em>{loose}</em> knocked loose', { k: bal.moved.toLocaleString('en-US'), d: (100 * bal.dmax).toFixed(0), loose: prt.n }) +
              ' · ' + T('the drift is an invented rule: a list defines no force');
          }
        } else if (fl.phase === 'drop') {
          s = SPL + T('elastic, {k} N/m · the ball is thrown down at {v} m/s', { k: mat.stiff.toFixed(0), v: DROPS.flight[0][3] }) +
            '<br>' + (el.touch ? T('contact force <em>{f} N</em> on {n} control points', { f: el.f.toFixed(0), n: el.sup.length }) : T('speed {v} m/s', { v: speed().toFixed(1) }));
        } else if (fl.phase === 'air') {
          s = T('<b>out of bounds</b> · ballistic flight, {v} m/s, {h} m up', { v: speed().toFixed(1), h: (ball.p[1] - YB).toFixed(1) }) + '<br>' +
            (fl.boost ? T('left the surface at <em>{v0} m/s</em>, set to {v1} m/s to reach the list · aimed at the list of Gaussians', { v0: fl.vsim.toFixed(1), v1: fl.vset.toFixed(1) })
                      : T('left the surface at <em>{v0} m/s</em> · aimed at the list of Gaussians', { v0: fl.vsim.toFixed(1) }));
        } else {
          s = LIST + T('the ball came in at {v} m/s and touched <em class="bad">{seen}</em> Gaussians · <em class="bad">{loose}</em> of {n} knocked loose<br>every Gaussian is a loose particle: nothing pushes back, nothing holds the bowl together',
                       { v: fl.vin.toFixed(1), seen: prt.nSeen.toLocaleString('en-US'), loose: prt.n.toLocaleString('en-US'), n: NUM });
        }
        setText(tagT, s);
      }

      /* ---------- controls ---------- */
      var panel = ui.panel(root, { pos: 'tr', title: T('material') });
      var sl = {
        e: ui.slider(panel, { label: T('restitution'), min: 0, max: 1, step: 0.05, value: mat.e, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { mat.e = v; } }),
        tough: ui.slider(panel, { label: T('toughness'), min: 1, max: 16, step: 0.5, value: mat.tough, fmt: function (v) { return v.toFixed(1) + ' N·s'; }, onInput: function (v) { mat.tough = v; } }),
        stiff: ui.slider(panel, { label: T('stiffness'), min: 60, max: 400, step: 10, value: mat.stiff, fmt: function (v) { return v + ' N/m'; }, onInput: function (v) { mat.stiff = v; } })
      };
      ui.button(panel, { label: T('drop again'), kind: 'go', onClick: function () { var m = st.sc.model; letGo(m === 'flight' || m === 'moment' ? 0 : run.k + (m === 'solid' || m === 'elastic' ? 1 : 0)); } });
      // on the lab page there is no deck to carry the mute: the module has its own switch (KnotTalk.muted, as in the deck)
      var sndT = env.embed ? null : ui.toggle(panel, { label: 'sound', value: !KT.muted, onChange: function (on) { KT.muted = !on; if (on) SFX.wake(); } });
      function dimRows() {
        var m = st.sc.model, use = { e: m === 'solid' || m === 'brittle' || m === 'moment', tough: m === 'brittle', stiff: m === 'elastic' || (m === 'flight' && !FIG) }, k;
        for (k in sl) sl[k].el.style.opacity = use[k] ? 1 : 0.35;
      }

      /* ---------- camera ---------- */
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 1.2, maxRadius: 34, target: [-X0, 0.2, 0], radius: 8 });
      function fit(k) { return KT.clamp((k || 1.15) / (size.w / size.h), 1, 2.4); }     // a narrow stage stands further back
      function view() {
        var m = model(), f = fit(), x = st.sc.side * X0;
        if (m === 'ghost') return { tx: x, ty: -0.45, tz: 0, r: 8.6 * f, th: -0.42, ph: 1.06 };
        if (m === 'moment') {                                       // SCRIPTED: in to the spot as the ball is let go, low enough to see under it
          var k = KT.ease((run.t - 0.1) / 1.1), sp = mo.spot, yb = KT.clamp((ball.p[1] - sp[1] - 0.35) / 0.35, 0, 1);
          // 30 degrees above the horizon at the closest: inside the bowl, above its wall, with the gap under the ball in sight
          return { tx: sp[0], ty: sp[1] + 0.1 + 0.28 * yb + 0.25 * (1 - k), tz: sp[2], r: (4.4 - 2.7 * k) * f, th: mo.side * (0.32 + 0.3 * (1 - k)), ph: 0.9 + 0.15 * k };
        }
        if (m === 'solid') return { tx: x, ty: 0.55, tz: 0, r: 7.4 * f, th: 0.45, ph: 0.82 };
        if (m === 'brittle') return { tx: x, ty: -0.1, tz: 0, r: 8.6 * f, th: 0.45, ph: 1.0 };
        if (m === 'elastic') return { tx: x, ty: 0.6, tz: 0, r: 7.6 * f, th: 0.5, ph: 1.0 };
        if (FIG) {                                                // the finale with the cast: both figures, the owl, the flight, the rabbit, the bowl
          var bx = ball.p[0], lo = size.w / size.h < 1.1 ? 0.45 : 0;   // a narrow stage (a phone) has the numbers over its lower third: the cast stands higher
          if (!ball.on) return { tx: 3.3, ty: 1.45 - lo, tz: 0, r: 10.4 * f, th: 0.06, ph: 1.36 };
          if (fl.phase === 'drop') return { tx: OWL_AT[0] - 0.8, ty: 1.0 - lo, tz: 0, r: 7.6 * f, th: 0.1, ph: 1.34 };
          if (fl.phase === 'air1') return { tx: 0.5 * (bx + RAB_AT[0]), ty: 1.4 - lo, tz: 0, r: 10 * f, th: 0.05, ph: 1.34 };
          if (fl.phase === 'air2') return { tx: KT.clamp(bx, -X0 + 0.8, RAB_AT[0]), ty: Math.max(0.9, ball.p[1] * 0.7) - lo, tz: 0, r: 10.5 * fit(1.36), th: 0, ph: 1.3 };
          return { tx: -X0, ty: 0.55 - 0.6 * lo, tz: 0, r: 8.4 * f, th: -0.1, ph: 1.2 };
        }
        // the finale: over the spline bowl, then with the ball, then over what is left of the list
        if (fl.phase === 'drop') return { tx: X0 - 0.3, ty: 0.9, tz: 0, r: 9 * f, th: 0.16, ph: 1.2 };
        if (fl.phase === 'air') return { tx: ball.p[0], ty: Math.max(0.6, ball.p[1] * 0.62), tz: 0, r: 15 * fit(1.36), th: 0.05, ph: 1.24 };
        return { tx: -X0, ty: -0.4, tz: 0, r: 10.5 * f, th: -0.14, ph: 1.16 };
      }

      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        var first = !st.frames;
        st.scene = id; st.sc = sc; st.acc = 0;
        if (surf.broken) mend();
        flatten(); derive(); restoreList(); letGo(0);
        if (!first) run.hover = Math.max(run.hover, 1.1);          // the camera may have a bowl's distance to travel
        dimRows(); st.off = null;                                  // (letGo set the caption and the close-up layout; the picture's offset snaps)
        silence(); quiet(); au.lastN = 0; au.first = true;        // a new scene starts in silence
        orbit.touched = false; orbit.to(view(), first);
      }

      env.loop(function (t, dt) {
        var n = 0, m = model(), v, g = orbit.goal, k, mom = m === 'moment', cap = mom ? 400 : 40, off;
        // moment: the panel of numbers covers the bottom of the stage (half of it on a phone); measured before this frame writes to it
        var cover = mom && size.h > 1 ? Math.min(0.7, (tag.offsetHeight + 10) / size.h) : 0;
        st.clock += dt; st.acc += dt; st.frames = (st.frames || 0) + 1;
        while (n < cap) {                                          // the kind of sub-step is asked each time: ghost hands over between its pass and its close-up
          if (run.close) { if (st.acc < HM / mo.slow) break; st.acc -= HM / mo.slow; substepMoment(); }   // a sub-step costs HM / slow of real time
          else { if (st.acc < H) break; substep(); st.acc -= H; }
          n++;
        }
        if (n === cap) st.acc = 0;
        listen(dt);
        if (sndT && sndT.get() === !!KT.muted) sndT.set(!KT.muted);   // the switch follows KnotTalk.muted, whoever set it
        m = model(); mom = m === 'moment'; v = view();
        if (m === 'flight') {                                     // follow the ball; a drag still turns the camera, nothing snaps
          g.tx = v.tx; g.ty = v.ty; g.tz = v.tz; g.r = v.r;
          if (!orbit.touched) { g.th = v.th; g.ph = v.ph; }
          orbit.update(dt, fl.phase === 'air' ? 3.4 : 2.4);
        } else {
          if (!orbit.touched) { orbit.to(v); g.th = v.th + 0.16 * Math.sin(st.clock * 0.23); }
          orbit.update(dt, 2.4);
        }
        // the picture moves up, so that the contact stays above the panel; eased, as ghost goes in and out of its close-up (snaps on a new scene)
        off = mom && size.h > 1 ? Math.max(0, cover - 0.24) * 0.75 * size.h : 0;
        st.off = st.off == null ? off : st.off + (off - st.off) * (1 - Math.exp(-4 * dt));
        if (st.off > 0.5 && size.h > 1) cam.setViewOffset(size.w, size.h, 0, st.off, size.w, size.h);
        else { st.off = off ? st.off : 0; if (cam.view && cam.view.enabled) cam.clearViewOffset(); }
        if (surf.broken) writeFragments(); else if (m === 'elastic' || (m === 'flight' && !FIG)) derive();
        if (FIG && m === 'flight') figDraw();
        paint(); overlays(); words(); timeline();
        fL.draw(cam, R.renderer);
        (G.draw || function (R, s, c) { R.renderer.render(s, c); })(R, scene, cam);
      });

      if (!document.getElementById('kt-touch-css')) {
        var css = document.createElement('style'); css.id = 'kt-touch-css';
        css.textContent =
          '.kt-sim-touch .kt-side{position:absolute;z-index:4;left:10px;bottom:10px;pointer-events:none;font:500 11px/1.5 var(--f-mono);letter-spacing:.03em;color:var(--dim);' +
          'padding:8px 11px;background:var(--panel);border:1px solid var(--line);border-radius:8px;max-width:min(680px,calc(100% - 20px))}' +
          '.kt-sim-touch .kt-side b{color:var(--text);font-weight:600;letter-spacing:.06em;text-transform:uppercase}' +
          '.kt-sim-touch .kt-side i{font-style:normal;color:var(--mute)}.kt-sim-touch .kt-side em{font:inherit;font-weight:600;letter-spacing:inherit;color:var(--ok)}' +   // (the deck sets every em in a serif italic)
          '.kt-sim-touch .kt-side em.bad{color:var(--err)}.kt-sim-touch .kt-side sub,.kt-sim-touch .kt-side sup{font-size:.8em;line-height:0}' +
          '.kt-sim-touch .kt-mo{display:none;margin:0 0 4px}.kt-sim-touch .kt-mo.on{display:block}.kt-sim-touch .kt-mo svg{display:block;overflow:visible}' +
          '.kt-sim-touch .kt-mo text{font:600 9.5px/1 var(--f-mono);letter-spacing:.02em}' +
          '.kt-sim-touch .kt-pin{position:absolute;z-index:3;left:0;top:0;pointer-events:none;white-space:nowrap;font:600 11px/1 var(--f-mono);color:var(--text);' +
          'padding:4px 6px;background:var(--panel);border:1px solid var(--line2);border-radius:5px;display:none}' +
          '.kt-sim-touch.kt-compact .kt-pin{font-size:10px}' +
          // the light theme: the words lie over the bowls (in `moment` the list, then the spline bowl, fill the stage), so
          // they sit on glass that hides what is behind them; a clear one takes the bowl's colour and dark text stops reading
          ':root[data-theme="light"] .kt-sim-touch :is(.kt-side,.kt-pin){background:rgba(255,255,255,.86);border-color:transparent;' +
          'box-shadow:var(--glass-edge),0 10px 28px -18px rgba(88,56,24,.45)}' +
          ':root[data-theme="light"] .kt-sim-touch.tc-close .kt-caption{border-radius:6px;background:rgba(255,255,255,.86);' +
          'box-shadow:0 0 0 6px rgba(255,255,255,.86),0 10px 26px -16px rgba(88,56,24,.45)}' +   // its margin as a spread: the caption keeps its place and size
          '@media (max-width:640px){.kt-sim-touch .kt-side{font-size:9.5px;padding:6px 8px}}' +
          // Hebrew is neither letter-spaced nor set in capitals, and a left-to-right run inside a Hebrew sentence (a number
          // with its unit, a formula, an address) is never broken over two lines
          ':root[lang="he"] .kt-sim-touch :is(.kt-side,.kt-side b,.kt-mo text){letter-spacing:0;text-transform:none}' +
          ':root[lang="he"] .kt-sim-touch .kt-side span[dir="ltr"]{white-space:nowrap}';
        document.head.appendChild(css);
      }

      setScene(env.opts.scene || 'ghost');
      return {
        setScene: setScene, orbit: orbit,
        /* handles for the deck and for tests */
        set: function (k, v) { if (sl[k]) { mat[k] = +v; sl[k].set(+v); } },
        advance: function (sec) {                                  // fast-forwarded time is not heard
          for (var i = 0, n = Math.round(sec / H); i < n; i++) substep();
          quiet(); au.lastN = prt.n; if (prt.n || prt.nSeen) au.first = false;
        },
        sfx: function () { return SFX.counts(); },
        stats: function () {
          return { scene: st.scene, t: run.t, drop: run.k, rule: run.rule, ph: run.ph, close: run.close, model: model(), on: ball.on, rest: ball.rest, p: ball.p.slice(), v: ball.v.slice(), speed: speed(),
                   energy: energy() / run.E0, bounces: hit.n, J: hit.J, ask: hit.ask, held: hit.held, u: cp.u, v2: cp.v, d: cp.d,
                   broken: surf.broken, pieces: frag.n, touch: el.touch, force: el.f, fmax: el.fmax, membrane: membraneEnergy() / run.E0, damped: el.lost / run.E0,
                   phase: fl.phase, vsim: fl.vsim, vset: fl.vset, boost: fl.boost, vin: fl.vin, reached: prt.nSeen, awake: prt.n, n: N, overlaps: overlaps,
                   moment: { side: mo.side, slow: mo.slow, D: mo.D, labels: CANDS.map(function (c) { return c.lab; }), t: mo.t.slice(), y: mo.y.slice(), who: mo.who.slice(),
                             near: mo.near.length, root: mo.root && { t: mo.root.t, res: mo.root.res, u: mo.root.u, v: mo.root.v, y: mo.root.y } },
                   fig: FIG ? { on: st.sc.model === 'flight', loops: fl.loops, t1: fl.t1, ta: fl.ta, tb: fl.tb, miss: fl.miss, owlTh: owlTh(run.t), rab: RP.T.slice(), lists: NF,
                                hits: fl.hit.map(function (h) { return h && { t: h.t, at: h.at, u: h.u, v: h.v, d: h.d, gap: h.gap, check: h.check, J: h.J, edge: h.edge, p: h.p, n: h.n, c: h.c, vin: h.vin, vout: h.vout, aim: h.aim }; }),
                                drawn: { owl: fig.nO, rabbit: fig.nR, owlGrid: FO.na * FO.nb, rabbitGrid: FR.na * FR.nb }, glass: FR.glass,
                                bal: { moved: bal.moved, inside: bal.inside, dmax: bal.dmax, dnow: bal.dnow, active: (function () { var c = 0; for (var i = 0; i < NT; i++) c += BA[i]; return c; })() },
                                drift: function () { var m = 0, P = fL.center, i, d; for (i = 0; i < NT * 3; i += 3) { d = Math.hypot(P[i] - LH[i], P[i + 1] - LH[i + 1], P[i + 2] - LH[i + 2]); if (d > m) m = d; } return m; }(),
                                reds: (function () { var c = 0, K = fL.color, i; for (i = 0; i < NT * 3; i += 3) if (Math.hypot(K[i] - ERR[0], K[i + 1] - ERR[1], K[i + 2] - ERR[2]) < 0.25) c++; return c; })(),
                                visible: { figures: fF.mesh.visible, bowl: fR.mesh.visible },
                                parked: (function () { var c = 0; for (var i = N; i < NT; i++) if (fL.center[i * 3 + 1] < -60) c++; return c; })() } : null };
        },
        resize: function (w, h) {
          R.resize(w, h); size.w = w; size.h = h;
          cpts.material.size = KT.clamp(w / 220, 2.5, 4.5); sup.material.size = cpts.material.size * 2.2;
          tl.w = Math.round(KT.clamp(w - 44, 220, 600)); tl.key = '';
          if (!orbit.touched) orbit.to(view(), true);
        },
        destroy: function () { silence(); R.renderer.dispose(); }
      };
    }
  });
})();
