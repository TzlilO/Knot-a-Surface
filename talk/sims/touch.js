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
     ghost    the ball is dropped on the list. There is no boundary to hit. Two honest rules alternate:
              no rule (it passes through), and "every Gaussian is a loose particle": the ball knocks away
              the ones it touches, and since the Gaussians of a list overlap where they lie, as solid
              particles they push each other apart and the bowl comes loose.
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
     flight   a hard bounce on the elastic bowl throws the ball out; the camera follows it across to the
              list, whose Gaussians it scatters as loose particles. Only the direction of the launch is
              chosen (see `launch`); the bounce, the flight and the scatter are simulated.
   All motion runs on a fixed sub-step (H; H / 8 in `moment`), so a scene plays the same at any frame rate.
   Every number on screen is read off the simulation.
   Sound: synthesised in the page with Web Audio (oscillators, a seeded noise buffer through filters, envelopes,
   and short grain buffers written here sample by sample); nothing is loaded or sampled. Every sound is an event
   the simulation computes, scaled by what it measures: a knock per bounce on the spline (its impulse), a low roll
   while the ball rolls (its speed), a hiss while it sinks through a list with no rule (how many Gaussians it
   overlaps), a scatter and a rattle when Gaussians are knocked loose (the momentum they took, how many came
   loose), a pluck and a wobble on the elastic surface (pitch from the spring stiffness, loudness and wobble from
   the control points' motion), a crack and shards when it breaks (impulse over toughness, the pieces' sizes and
   speeds), a whoosh at the launch (its speed), a tick for each candidate moment of contact. Nothing plays before
   a key or pointer press (browsers forbid it), while KnotTalk.muted is true, or while the module is paused.
   Tool: WebGL (three.js), Web Audio. Standalone: talk/sims/touch.html.
   Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;

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
  var LIGHT = [0.35, 0.86, 0.36];

  var SCENES = {
    ghost:   { cap: '<b>a ball meets a list of Gaussians</b> · a density, not a boundary', model: 'ghost', side: -1 },
    moment:  { cap: '<b>the moment of contact</b> · close up, slowed down', model: 'moment', side: -1 },
    solid:   { cap: '<b>the same drop on the spline</b> · closest point, normal, reflection', model: 'solid', side: 1 },
    brittle: { cap: '<b>toughness</b> · the surface breaks when the impact asks too much of it', model: 'brittle', side: 1 },
    elastic: { cap: '<b>an elastic surface</b> · the control points sit on springs', model: 'elastic', side: 1 },
    flight:  { cap: '<b>out of bounds</b> · from the spline bowl to the list of Gaussians', model: 'flight', side: 1 }
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
     quieter than every voice still sounding is dropped, a louder one replaces the quietest. Sustained voices
     (roll, hiss, wobble): one each per mount, renewed every frame, and if no frame renews them (the module is
     paused off screen, the tab is hidden) they fade out within HOLD seconds and stop by themselves. */
  var SFX = (function () {
    var ctx = null, bus = null, noise = null, dead = false, armed = false, live = [], cnt = {}, rr = KT.rng(4242);
    var VOICES = 8, HOLD = 0.22, LAG = 0.012;
    ['knock', 'roll', 'hiss', 'scatter', 'rattle', 'wobble', 'crack', 'shards', 'whoosh', 'tick'].forEach(function (k) { cnt[k] = 0; });
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
      bus = ctx.createGain(); bus.gain.value = 0.22;                          // a room, not a game: the hardest knock peaks near -14 dBFS
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
    function voice(kind, peak, tau, pan) {
      var t = ctx.currentTime + LAG, i, w = -1, lo = 1e9, l, v, node;
      for (i = live.length - 1; i >= 0; i--) if (live[i].end <= t) live.splice(i, 1);
      if (live.length >= VOICES) {
        for (i = 0; i < live.length; i++) { l = live[i].peak * Math.exp(-Math.max(0, t - live[i].t) / live[i].tau); if (l < lo) { lo = l; w = i; } }
        if (lo >= peak) return null;
        cut(live[w], t); live.splice(w, 1);
      }
      v = { t: t, peak: peak, tau: tau, end: t + 0.05, srcs: [], left: 0, out: ctx.createGain() };
      node = v.out;
      if (pan && ctx.createStereoPanner) { var p = ctx.createStereoPanner(); p.pan.value = pan; v.out.connect(p); node = p; }
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
      var h = KT.clamp(sp / 8, 0, 1), a = 0.035 + 0.13 * Math.pow(h, 0.8), F = FSPR * Math.sqrt(k / 160), v = voice('wobble', a, 0.16, pan); if (!v) return;
      var t = v.t, o = osc(v, F * 1.6, t, 1.1), o2 = osc(v, F * 3.2, t, 0.6);
      o.frequency.exponentialRampToValueAtTime(F, t + 0.11); chain(o, amp(t, a, 0.003, 0.16), v.out);
      o2.frequency.exponentialRampToValueAtTime(F * 2, t + 0.11); chain(o2, amp(t, a * 0.3, 0.003, 0.07), v.out);
      chain(osc(v, 95, t, 0.3), amp(t, a * 0.4, 0.002, 0.04), v.out);                // the push of the ball into it
    }
    /* the ball leaves the bowl at sp m/s */
    function whoosh(sp, pan) {
      if (!ok()) return;
      var h = KT.clamp(sp / 14, 0, 1), A = 0.08 + 0.22 * h, f = 260 + 55 * sp, v = voice('whoosh', A, 0.3, pan); if (!v) return;
      var t = v.t, bp = biq('bandpass', f * 0.55, 1.1), g = amp(null, 0);
      bp.frequency.setValueAtTime(f * 0.55, t); bp.frequency.exponentialRampToValueAtTime(f, t + 0.12); bp.frequency.exponentialRampToValueAtTime(f * 0.4, t + 0.95);
      g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(A, t + 0.08); g.gain.setTargetAtTime(0, t + 0.1, 0.24);
      chain(hiss(v, t, 1.3), bp, g, v.out);
      chain(hiss(v, t, 1.0), biq('lowpass', 300, 0.7), amp(t, A * 0.5, 0.06, 0.2), v.out);
    }
    /* moment: candidate i says "contact": a dry click, a little lower for each later definition (a click, not a note:
       the chapter's music has its own) */
    function tick(i, pan) {
      if (!ok()) return;
      var f = 3000 * Math.pow(0.92, i), v = voice('tick', 0.16, 0.003, pan); if (!v) return;
      chain(play(v, grains([[0, f, 0.0024, 0.16], [0, f * 1.47, 0.0011, 0.1]], 0.03), v.t), v.out);
    }
    function hush() { if (!ctx) return; var t = ctx.currentTime; live.forEach(function (v) { cut(v, t); }); live = []; }

    /* ---- sustained voices ---- */
    function glide(p, x, t, tc) { p.cancelScheduledValues(t); p.setTargetAtTime(x, t, tc); }
    function channel(kind) {
      var c = { g: null, srcs: [], until: 0, lvl: 0, quiet: -9, p: null };
      function make(t) {
        var g = c.g = ctx.createGain(), s, f;
        g.gain.setValueAtTime(0, t); g.connect(bus); c.srcs = []; c.p = {};
        if (kind === 'wobble') {                                                 // a soft tone and its octave
          var o1 = ctx.createOscillator(), o2 = ctx.createOscillator(), lp = biq('lowpass', 1400, 0.6);
          o1.start(t); o2.start(t); c.srcs.push(o1, o2);
          chain(o1, lp, g); chain(o2, amp(null, 0.22), lp); c.p.f = o1.frequency; c.p.f2 = o2.frequency;
        } else {
          s = ctx.createBufferSource(); s.buffer = noise; s.loop = true; s.start(t, rr() * 2); c.srcs.push(s);
          if (kind === 'roll') { f = biq('lowpass', 160, 0.9); chain(s, f, biq('highpass', 45, 0.7), g); }
          else { f = biq('bandpass', 1700, 0.55); chain(s, f, biq('lowpass', 3800, 0.5), g); }
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
        if (!alive) { c.stop(); make(t); }
        if (c.lvl === 0 && t - c.quiet > 0.25 && kind !== 'wobble') cnt[kind]++;   // a wobble is counted by its pluck
        p = c.g.gain; p.cancelScheduledValues(t);
        p.setTargetAtTime(level, t, level >= c.lvl ? 0.02 : 0.08);
        p.setTargetAtTime(0, t + HOLD, 0.05);                                    // the next frame renews this; if none comes, it fades
        c.until = t + HOLD + 0.45;
        c.srcs.forEach(function (s) { try { s.stop(c.until); } catch (e) {} });
        if (f) { glide(c.p.f, f, t, kind === 'wobble' ? 0.012 : 0.04); if (c.p.f2) glide(c.p.f2, 2 * f, t, 0.012); }
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
      pluck: safe(pluck), whoosh: safe(whoosh), tick: safe(tick), hush: safe(hush),
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
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 400), R = G.renderer(root, cam), scene = new T.Scene();
      var st = { scene: 'ghost', sc: SCENES.ghost, acc: 0, clock: 0 }, size = { w: 2, h: 1 };
      var mat = { e: 0.75, tough: 5, stiff: 160 };
      ['e', 'tough', 'stiff'].forEach(function (k) { if (env.opts[k] != null && isFinite(+env.opts[k])) mat[k] = +env.opts[k]; });
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var tag = ui.el('div', 'kt-side'); root.appendChild(tag);
      var tlm = ui.el('div', 'kt-mo'), tagT = ui.el('div'); tag.appendChild(tlm); tag.appendChild(tagT);   // moment's timeline, then the words
      var pin = ui.el('div', 'kt-pin'); root.appendChild(pin);
      SFX.arm();

      scene.add(new T.HemisphereLight(0xe6efff, 0x2a3650, 1.5));
      var sun = new T.DirectionalLight(0xffffff, 2.6); sun.position.set(LIGHT[0], LIGHT[1], LIGHT[2]); scene.add(sun);

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
      var fR = G.splats(N), fL = fuzzy(N); scene.add(fR.mesh); scene.add(fL.mesh);
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
        home.c.set(fL.center);
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
        var g = new T.InstancedBufferGeometry(), base = new T.PlaneGeometry(2, 2), A = {}, vb = new T.Vector2();
        g.index = base.index; g.setAttribute('position', base.getAttribute('position'));
        [['iC', 3], ['iU', 3], ['iV', 3], ['iW', 3], ['iCol', 3], ['iA', 1]].forEach(function (d) {
          var at = new T.InstancedBufferAttribute(new Float32Array(n * d[1]), d[1]); at.setUsage(T.DynamicDrawUsage); g.setAttribute(d[0], at); A[d[0]] = at;
        });
        g.instanceCount = 0;
        var m = new T.ShaderMaterial({ transparent: true, depthWrite: false, side: T.DoubleSide, vertexShader: FUZZ_V, fragmentShader: FUZZ_F,
          uniforms: { uView: { value: new T.Vector2(1, 1) }, uLight: { value: new T.Vector3(LIGHT[0], LIGHT[1], LIGHT[2]).normalize() } } });
        var mesh = new T.Mesh(g, m); mesh.frustumCulled = false; mesh.renderOrder = 3;
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
      var ballMesh = new T.Mesh(new T.SphereGeometry(BR, 48, 32), new T.MeshStandardMaterial({ color: 0xe4e9f0, roughness: 0.5, metalness: 0 }));
      scene.add(ballMesh);
      var arrow = G.vector(C.surf, { radius: 0.02, ghost: 0.22 }); scene.add(arrow.obj);
      var dot = new T.Mesh(new T.SphereGeometry(0.045, 16, 12), new T.MeshBasicMaterial({ color: C.text, depthTest: false, transparent: true }));
      dot.renderOrder = 30; scene.add(dot);
      var NS = 40, iso = G.lines(4 * NS, C.text, 0.85), drop = G.lines(2, C.text, 0.55); scene.add(iso.obj); scene.add(drop.obj);
      var net = G.lines(4 * NC * (NC - 1), C.ctrl, 0.5); net.mat.depthTest = false; net.obj.renderOrder = 12; scene.add(net.obj);
      function points(n, px, opacity) {
        var g = new T.BufferGeometry(); g.setAttribute('position', new T.BufferAttribute(new Float32Array(n * 3), 3));
        var o = new T.Points(g, new T.PointsMaterial({ color: C.ctrl, size: px, sizeAttenuation: false, depthTest: false, transparent: true, opacity: opacity }));
        o.frustumCulled = false; o.renderOrder = 13; scene.add(o); return o;
      }
      var cpts = points(NC * NC, 4.5, 0.85), sup = points(16, 10, 1);
      var stem = G.lines(2 * NC * NC, C.ctrl, 1); stem.mat.depthTest = false; stem.obj.renderOrder = 12; scene.add(stem.obj);   // each spring, from its anchor to its control point
      var TR = 220, trail = G.lines(2 * TR, C.text, 0.4), tr = { p: new Float32Array(TR * 3), n: 0, k: 0 }; scene.add(trail.obj);
      // moment: where the ball's lowest point was when each definition said "contact" (red: the list; cyan: the spline), on its vertical
      var mkL = G.lines(2 * CANDS.length, C.err, 0.95), mkS = G.lines(2, C.surf, 1), mkG = G.lines(2, C.text, 0.3);
      [mkL, mkS, mkG].forEach(function (l) { l.mat.depthTest = false; l.obj.renderOrder = 14; scene.add(l.obj); });
      var low = new T.Mesh(new T.SphereGeometry(0.011, 12, 8), new T.MeshBasicMaterial({ color: C.text, depthTest: false, transparent: true }));   // the ball's lowest point
      low.renderOrder = 31; scene.add(low); ballMesh.renderOrder = 4;                   // (drawn after the list, for when it is see-through)

      /* ---------- state ---------- */
      var ball = { p: [0, 0, 0], v: [0, 0, 0], on: false, rest: false };
      var cp = { ok: false, u: 0.5, v: 0.5, p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], n: [0, 1, 0], nc: [0, 1, 0], d: 9, sd: 9, edge: false };
      var run = { k: 0, t: 0, tr: 0, hover: 0.55, still: 0, E0: 1, rule: 0 };
      var hit = { n: 0, J: 0, t: -9, ask: 0, held: true, vout: 0 };
      var surf = { broken: false, tb: 0 };
      var el = { touch: false, f: 0, fmax: 0, J: 0, lost: 0, pen: 0, sup: [] };
      var fl = { phase: 'drop', tt: 0, th: 0, vsim: 0, vset: 0, boost: false, vin: 0 };
      var prt = { v: new Float32Array(N * 3), aw: new Uint8Array(N), n: 0, seen: new Uint8Array(N), nSeen: 0, sub: 0 };
      function newCp() { return { ok: false, u: 0.5, v: 0.5, p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], n: [0, 1, 0], nc: [0, 1, 0], d: 9, sd: 9, edge: false }; }
      var mo = { side: -1, D: 9, slow: 1, near: [], top: 0, spot: [0, 0, 0], val: null, t: [], y: [], who: [], root: null, end: 0, cpm: newCp(), cpr: newCp(), cps: newCp() };
      /* what the sound hears: the physics only ever adds to these, and they never feed back */
      var au = { simT: 0, J: 0, Jn: 0, tone: 1, rest: 0, restV: 0, inside: 0, calls: 0, hits: 0, jl: 0, crack: 0, pluck: 0, launch: 0, tick: [],
                 first: true, lastN: 0, rat: 0, ratT: -9, env: 0 };

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
          mo.t[i] = tNow - HM + f * HM; mo.y[i] = yPrev + f * (ball.p[1] - yPrev) - BR; mo.who[i] = gap.who[i]; au.tick.push(i);
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
        var d = DROPS[st.scene], q = d[k % d.length], m = st.sc.model, side = m === 'moment' ? (k % 2 ? 1 : -1) : st.sc.side;
        run.k = k; run.t = 0; run.still = 0; run.rule = k % 2; run.hover = m === 'flight' ? 1.3 : m === 'moment' ? 1.3 : 0.55;
        ball.p = [side * X0 + q[0], q[1], q[2]]; ball.v = [0, -q[3], 0]; ball.on = false; ball.rest = false;
        cp.ok = false; hit.n = 0; hit.J = 0; hit.t = -9; hit.ask = 0; hit.held = true; hit.vout = 0; tr.n = tr.k = 0;
        run.E0 = energy(); el.lost = 0;
        if (m === 'ghost' || m === 'flight' || m === 'moment') restoreList();
        if (surf.broken) mend();
        if (m === 'flight') { flatten(); derive(); fl.phase = 'drop'; fl.tt = fl.th = 0; fl.boost = false; }
        if (m === 'moment') momentDrop(side);
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
        var m = st.sc.model, s;
        if (m === 'moment') { for (s = 0; s < 8; s++) substepMoment(); return; }
        run.t += H; au.simT += H;
        if (!ball.on && run.t >= run.hover) { ball.on = true; run.tr = run.t - H; run.E0 = energy(); }
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
      function quiet() { au.simT = au.J = au.Jn = au.rest = au.restV = au.inside = au.calls = au.hits = au.jl = au.crack = au.pluck = au.launch = 0; au.tick.length = 0; }
      function silence() { chRoll.stop(); chHiss.stop(); chWob.stop(); SFX.hush(); au.env = 0; au.rat = 0; }
      function listen(dt) {
        var m = st.sc.model, s, pan = 0, x, w;
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
        chRoll.set(au.rest && au.simT ? 0.11 * Math.pow(KT.clamp(x / 2.5, 0, 1), 1.2) * Math.min(1, au.rest / au.simT) : 0, 110 + 70 * x);
        x = au.calls ? au.inside / au.calls : 0;                                       // sinking through a list with no rule
        chHiss.set(x > 0 ? 0.16 * Math.pow(KT.clamp(x / 140, 0, 1), 0.6) * KT.clamp(speed() / 7, 0.35, 1.2) : 0, (1200 + 90 * speed()) / Math.pow(stretch, 0.3));
        if (au.hits && au.first) { SFX.scatter(au.jl, au.hits, pan); au.first = false; au.lastN = prt.n; au.ratT = SFX.now(); }   // the ball meets loose particles
        else if (!au.first) {                                                         // and every Gaussian knocked loose since rattles
          x = prt.n - au.lastN; if (x < 0) x = 0; au.lastN = prt.n; au.rat += x;
          if (au.rat && SFX.now() - au.ratT > 0.06) { SFX.rattle(au.rat, pan); au.rat = 0; au.ratT = SFX.now(); }
        }
        if (au.pluck > 0.2) SFX.pluck(au.pluck, mat.stiff, pan);                      // a contact begins on the springs
        if (m === 'elastic' || m === 'flight') {                                      // and the net rings: loudness from its motion, the wobble from its first mode
          var e2 = 0, vm = 0, ws = 0, q, i, j;
          for (j = 1; j < NC - 1; j++) for (i = 1; i < NC - 1; i++) {
            q = (j * NC + i) * 3; e2 += MV[q] * MV[q] + MV[q + 1] * MV[q + 1] + MV[q + 2] * MV[q + 2];
            w = SINW[i] * SINW[j]; vm += w * MV[q + 1]; ws += w;
          }
          x = Math.sqrt(e2 / ((NC - 2) * (NC - 2)));                                  // rms speed of the control points, m/s: loudness ~ their kinetic energy
          au.env = Math.max(x, au.env * Math.exp(-dt / 0.15));
          chWob.set(au.env > 0.08 ? 0.14 * Math.pow(KT.clamp(au.env / 1.4, 0, 1), 2) : 0, 150 * Math.sqrt(mat.stiff / 160) * (1 + 0.05 * KT.clamp(vm / ws / 0.6, -1, 1)));
        } else { chWob.set(0); au.env = 0; }
        if (au.launch) SFX.whoosh(au.launch, pan);                                    // out of the bowl
        au.tick.forEach(function (i) { SFX.tick(i, pan); });                          // moment: a definition says "contact"
        quiet();
      }

      /* ---------- colours: the checker of knot spans, the ball's shadow, and red where things go wrong ---------- */
      var ERR = G.rgb(C.err);
      function paint() {
        var bx = ball.p[0], by = ball.p[1], bz = ball.p[2], k, o, dx, dy, dz, t, d2, s, w, f, P, cl, L = LIGHT, ll = Math.hypot(L[0], L[1], L[2]);
        var lx = L[0] / ll, ly = L[1] / ll, lz = L[2] / ll, m = st.sc.model, heat = surf.broken ? Math.exp(-(run.t - surf.tb) / 0.7) : 0, q, r;
        {
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
          if (m === 'moment' && mo.side < 0) mo.who.forEach(function (i) {          // the Gaussians that decided a definition: lit up
            if (i < 0) return; o = i * 3; cl[o] = 0.4 * cl[o] + 0.6; cl[o + 1] = 0.4 * cl[o + 1] + 0.58; cl[o + 2] = 0.4 * cl[o + 2] + 0.56;
          });
        }
      }

      /* ---------- what is drawn about the contact ---------- */
      function overlays() {
        var m = st.sc.model, onSpline = m === 'solid' || m === 'elastic' || (m === 'brittle' && !surf.broken) || (m === 'flight' && fl.phase === 'drop') || (m === 'moment' && mo.side > 0);
        var soft = m === 'elastic' || m === 'flight', k, i, o, q, bm = ballMesh.material, see = m === 'moment';
        ballMesh.position.set(ball.p[0], ball.p[1], ball.p[2]);
        if (bm.transparent !== see) { bm.transparent = see; bm.opacity = see ? 0.3 : 1; bm.depthWrite = !see; bm.needsUpdate = true; }   // close up: see under the ball
        dot.scale.setScalar(see ? 0.35 : 1);
        low.visible = see && ball.on; low.position.set(ball.p[0], ball.p[1] - BR, ball.p[2]);
        if (onSpline) closest(ball.p, cp);
        var near = onSpline && cp.d < BR + 0.75 && (soft || !cp.edge || cp.d < BR * 1.6), since = run.t - hit.t;
        iso.obj.visible = dot.visible = arrow.obj.visible = drop.obj.visible = near;
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
        var s = near ? G.project(cam, cp.p[0], cp.p[1], cp.p[2], size.w, size.h) : null;
        if (s && s[0] > 40 && s[0] < size.w - 150 && s[1] > 40 && s[1] < size.h - 30) {
          pin.style.display = 'block'; pin.style.transform = 'translate(' + (s[0] + 16).toFixed(1) + 'px,' + (s[1] + 10).toFixed(1) + 'px)';
          setText(pin, '(u, v) = (' + cp.u.toFixed(2) + ', ' + cp.v.toFixed(2) + ')');
        } else pin.style.display = 'none';
      }
      function setText(e, html) { if (e._h !== html) { e._h = html; e.innerHTML = html; } }

      /* ---------- moment: the timeline of the candidate moments (ms after release), the list's and the spline's ---------- */
      var tl = { key: '', w: 420, cur: null, a: 0, b: 1, x0: 52 };
      function timeline() {
        var on = st.sc.model === 'moment', Tt = mo.t, idx = [], i, lo = 1e9, hi = -1e9, r = mo.root;
        tlm.classList.toggle('on', on);
        if (!on) return;
        for (i = 0; i < CANDS.length; i++) if (Tt[i] != null) { idx.push(i); lo = Math.min(lo, Tt[i]); hi = Math.max(hi, Tt[i]); }
        if (!idx.length) { if (tl.key) { tlm.innerHTML = ''; tl.key = ''; tl.cur = null; } return; }
        var llo = lo, lhi = hi;
        if (r) { lo = Math.min(lo, r.t); hi = Math.max(hi, r.t); }
        var key = idx.join(',') + (r ? ',r' : '') + ',' + tl.w, W = tl.w, L0 = tl.x0, R1 = W - 12, pad = Math.max(0.0015, 0.1 * (hi - lo));
        function X(t) { return L0 + (t - tl.a) / (tl.b - tl.a) * (R1 - L0); }
        function ln(x1, x2, y1, y2, col, wd, op) { return '<line x1="' + x1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y1="' + y1 + '" y2="' + y2 + '" stroke="' + col + '"' + (wd ? ' stroke-width="' + wd + '"' : '') + (op ? ' stroke-opacity="' + op + '"' : '') + '/>'; }
        function tx(x, y, col, s, anc) { return '<text x="' + x.toFixed(1) + '" y="' + y + '" fill="' + col + '"' + (anc ? ' text-anchor="' + anc + '"' : '') + '>' + s + '</text>'; }
        if (key !== tl.key) {
          tl.key = key; tl.a = lo - pad; tl.b = hi + pad;
          var span = (tl.b - tl.a) * 1000, steps = [0.5, 1, 2, 5, 10, 20, 50, 100], sv = 100, k, x, ends = [-99, -99];
          for (k = 0; k < steps.length; k++) if (span / steps[k] <= Math.max(3, Math.floor((R1 - L0) / 48))) { sv = steps[k]; break; }
          var s = '<svg width="' + W + '" height="84" viewBox="0 0 ' + W + ' 84">' + ln(L0, R1, 70, 70, C.mute, 0, 0.6);
          for (k = Math.ceil(tl.a * 1000 / sv) * sv; k <= tl.b * 1000 + 1e-9; k += sv) { x = X(k / 1000); s += ln(x, x, 67, 73, C.mute) + tx(x, 82, C.mute, +k.toFixed(1), 'middle'); }
          s += tx(0, 82, C.mute, 'ms') + tx(0, 35, C.splat, 'list') + ln(L0, R1, 31, 31, C.line, 1) + tx(0, 58, C.surf, 'spline') + ln(L0, R1, 54, 54, C.line, 1);
          idx.slice().sort(function (p, q) { return Tt[p] - Tt[q]; }).forEach(function (c) {
            var x = X(Tt[c]), lab = CANDS[c].lab, w = lab.length * 6 + 6, row = x - w / 2 > ends[0] ? 0 : x - w / 2 > ends[1] ? 1 : ends[0] <= ends[1] ? 0 : 1;
            ends[row] = x + w / 2;
            s += ln(x, x, 25, 37, C.err, 2) + tx(x, row ? 9 : 20, C.err, lab, 'middle');
          });
          if (idx.length > 1) s += ln(X(llo), X(lhi), 42, 42, C.err, 1, 0.7) + ln(X(llo), X(llo), 39, 45, C.err, 1, 0.7) + ln(X(lhi), X(lhi), 39, 45, C.err, 1, 0.7);
          if (r) { x = X(r.t); s += ln(x, x, 48, 60, C.surf, 2) + tx(x + 6 > R1 - 40 ? x - 6 : x + 6, 58, C.surf, 'one root', x + 6 > R1 - 40 ? 'end' : ''); }
          s += '<line class="kt-mo-now" x1="0" x2="0" y1="2" y2="62" stroke="' + C.text + '" stroke-opacity="0.6"/></svg>';
          tlm.innerHTML = s; tl.cur = tlm.querySelector('.kt-mo-now');
        }
        var now = run.t - run.tr, xn = X(now);
        if (tl.cur) { tl.cur.setAttribute('x1', xn.toFixed(1)); tl.cur.setAttribute('x2', xn.toFixed(1)); tl.cur.style.display = ball.on && now >= tl.a && now <= tl.b ? '' : 'none'; }
      }

      /* ---------- the words: every number is read off the simulation ---------- */
      var NUM = N.toLocaleString('en-US'), LIST = '<b>a list</b> · ' + NUM + ' Gaussians, unordered <i>(emulated)</i><br>';
      var SPL = '<b>a spline chart</b> · ' + NC + ' × ' + NC + ' control points → ' + NUM + ' Gaussians<br>';
      function pct(x) { return (100 * x).toFixed(Math.abs(1 - x) < 0.02 ? 2 : 0).replace('-', '−') + ' %'; }   // below zero: lower than at rest in the bowl
      function ms(t) { return (1000 * t).toFixed(1) + ' ms'; }
      function sci(x) { if (!(x > 0)) return '0'; var e = Math.floor(Math.log10(x)); return (x / Math.pow(10, e)).toFixed(1) + ' × 10<sup>' + String(e).replace('-', '−') + '</sup>'; }
      function words() {
        var m = st.sc.model, s, e = energy() / run.E0, addr = 'closest point S(<em>' + cp.u.toFixed(2) + '</em>, <em>' + cp.v.toFixed(2) + '</em>)';
        if (m === 'ghost') {
          var dv = Math.hypot(ball.v[0], ball.v[1] + (ball.on ? GRAV * (run.t - run.tr) : 0), ball.v[2]);
          s = LIST + (run.rule
            ? 'rule: every Gaussian is a loose particle · the ball touched <em class="bad">' + prt.nSeen.toLocaleString('en-US') + '</em> · <em class="bad">' + prt.n.toLocaleString('en-US') + '</em> knocked loose · it lost ' + dv.toFixed(2) + ' m/s<br>a Gaussian overlaps ' + overlaps.toFixed(1) + ' others where it lies: as solid particles they push each other apart'
            : 'no rule: the ball overlapped <em class="bad">' + prt.nSeen.toLocaleString('en-US') + '</em> Gaussians and touched <em class="bad">0</em> · its velocity changed by ' + dv.toFixed(2) + ' m/s<br>a list has a density, not a boundary: a contact needs a threshold, or a mesh extracted first');
        } else if (m === 'moment') {
          var nf = 0, lo = 1e9, hi = -1e9, ylo = 0, yhi = 0, slow = mo.slow < 0.985 ? ' · time slowed ×' + Math.round(1 / mo.slow) + ' <i>(scripted)</i>' : '';
          mo.t.forEach(function (t, i) { if (t == null) return; nf++; if (t < lo) { lo = t; yhi = mo.y[i]; } if (t > hi) { hi = t; ylo = mo.y[i]; } });
          if (mo.side < 0 || !mo.root) {
            s = mo.side < 0 ? LIST : SPL;
            if (mo.side < 0) s += nf
              ? nf + ' of ' + CANDS.length + ' definitions say “contact” · from ' + ms(lo) + ' to ' + ms(hi) + ' after release · spread <em class="bad">' + ms(hi - lo) + '</em>, ' + (100 * (yhi - ylo)).toFixed(1) + ' cm of fall'
              : 'when does the ball touch it? A list does not say, so each definition is measured as the ball comes down';
            else s += 'contact is the root of |S(u, v) − c| = r, with S(u, v) the closest point · nothing to choose';
            s += '<br>' + (mo.side < 0 ? 'σ: shell of the nearest Gaussian · ρ: summed density at the ball’s lowest point · μ: nearest mean · for this emulated list' : 'the list’s ' + nf + ' answers stay on the timeline') + slow;
          } else {
            var r = mo.root;
            s = SPL + 'contact: |S(u, v) − c| = r at <em>' + ms(r.t) + '</em> after release, at S(<em>' + r.u.toFixed(2) + '</em>, <em>' + r.v.toFixed(2) + '</em>) · residual <em>' + sci(r.res) + ' m</em>' +
                '<br>one definition, one time' + (nf > 1 ? ' · the list’s ' + nf + ' answers spread over ' + ms(hi - lo) : '') + slow;
          }
        } else if (m === 'solid') {
          s = SPL + addr + ' · distance <em>' + Math.max(0, cp.d - BR).toFixed(2) + ' m</em> · normal n = S<sub>u</sub> × S<sub>v</sub><br>' +
            (hit.n ? (ball.rest ? 'rolling on the surface after ' + hit.n + ' bounces' : 'bounce ' + hit.n + ': impulse <em>' + hit.J.toFixed(1) + ' N·s</em> along n') : 'ideal rigid surface') +
            ' · restitution e = ' + mat.e.toFixed(2) + ' · energy <em>' + pct(e) + '</em> of the drop';
        } else if (m === 'brittle') {
          s = SPL + (surf.broken
            ? 'the impact asked <em class="bad">' + hit.ask.toFixed(1) + ' N·s</em> &gt; toughness ' + mat.tough.toFixed(1) + ' N·s: <em class="bad">broken</em> into <em>' + frag.n + '</em> pieces of the (u, v) domain<br>each piece is a rigid body and its Gaussians ride on it · the ball kept ' + hit.vout.toFixed(1) + ' m/s'
            : (hit.n ? 'the impact asked <em>' + hit.ask.toFixed(1) + ' N·s</em> ≤ toughness ' + mat.tough.toFixed(1) + ' N·s: <em>it holds</em>' : 'toughness ' + mat.tough.toFixed(1) + ' N·s: the largest impulse the surface can return') +
              '<br>' + addr + ' · impulse asked = m (1 + e) |v · n|');
        } else if (m === 'elastic') {
          var em = (membraneEnergy() + 0.5 * KC * el.pen * el.pen) / run.E0;       // the springs of the net, and the contact spring
          s = SPL + 'control points on damped springs, ' + mat.stiff.toFixed(0) + ' N/m · ' +
            (el.touch ? 'contact force <em>' + el.f.toFixed(0) + ' N</em> on ' + el.sup.length + ' control points, by basis weight' : 'the Gaussians are derived again every frame') +
            '<br>energy: ball <em>' + pct(e) + '</em> · surface <em>' + pct(em) + '</em> · damped ' + pct(el.lost / run.E0) + (hit.n ? ' · bounce ' + hit.n : '');
        } else if (fl.phase === 'drop') {
          s = SPL + 'elastic, ' + mat.stiff.toFixed(0) + ' N/m · the ball is thrown down at ' + DROPS.flight[0][3] + ' m/s' +
            '<br>' + (el.touch ? 'contact force <em>' + el.f.toFixed(0) + ' N</em> on ' + el.sup.length + ' control points' : 'speed ' + speed().toFixed(1) + ' m/s');
        } else if (fl.phase === 'air') {
          s = '<b>out of bounds</b> · ballistic flight, ' + speed().toFixed(1) + ' m/s, ' + (ball.p[1] - YB).toFixed(1) + ' m up<br>left the surface at <em>' + fl.vsim.toFixed(1) + ' m/s</em>' +
            (fl.boost ? ', set to ' + fl.vset.toFixed(1) + ' m/s to reach the list' : '') + ' · aimed at the list of Gaussians';
        } else {
          s = LIST + 'the ball came in at ' + fl.vin.toFixed(1) + ' m/s and touched <em class="bad">' + prt.nSeen.toLocaleString('en-US') + '</em> Gaussians · <em class="bad">' + prt.n.toLocaleString('en-US') + '</em> of ' + NUM + ' knocked loose' +
            '<br>every Gaussian is a loose particle: nothing pushes back, nothing holds the bowl together';
        }
        setText(tagT, s);
      }

      /* ---------- controls ---------- */
      var panel = ui.panel(root, { pos: 'tr', title: 'material' });
      var sl = {
        e: ui.slider(panel, { label: 'restitution', min: 0, max: 1, step: 0.05, value: mat.e, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { mat.e = v; } }),
        tough: ui.slider(panel, { label: 'toughness', min: 1, max: 16, step: 0.5, value: mat.tough, fmt: function (v) { return v.toFixed(1) + ' N·s'; }, onInput: function (v) { mat.tough = v; } }),
        stiff: ui.slider(panel, { label: 'stiffness', min: 60, max: 400, step: 10, value: mat.stiff, fmt: function (v) { return v + ' N/m'; }, onInput: function (v) { mat.stiff = v; } })
      };
      ui.button(panel, { label: 'drop again', kind: 'go', onClick: function () { var m = st.sc.model; letGo(m === 'flight' || m === 'moment' ? 0 : run.k + (m === 'solid' || m === 'elastic' ? 1 : 0)); } });
      // on the lab page there is no deck to carry the mute: the module has its own switch (KnotTalk.muted, as in the deck)
      var sndT = env.embed ? null : ui.toggle(panel, { label: 'sound', value: !KT.muted, onChange: function (on) { KT.muted = !on; if (on) SFX.wake(); } });
      function dimRows() {
        var m = st.sc.model, use = { e: m === 'solid' || m === 'brittle' || m === 'moment', tough: m === 'brittle', stiff: m === 'elastic' || m === 'flight' }, k;
        for (k in sl) sl[k].el.style.opacity = use[k] ? 1 : 0.35;
      }

      /* ---------- camera ---------- */
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 1.2, maxRadius: 34, target: [-X0, 0.2, 0], radius: 8 });
      function fit(k) { return KT.clamp((k || 1.15) / (size.w / size.h), 1, 2.4); }     // a narrow stage stands further back
      function view() {
        var m = st.sc.model, f = fit(), x = st.sc.side * X0;
        if (m === 'ghost') return { tx: x, ty: -0.45, tz: 0, r: 8.6 * f, th: -0.42, ph: 1.06 };
        if (m === 'moment') {                                       // SCRIPTED: in to the spot as the ball is let go, low enough to see under it
          var k = KT.ease((run.t - 0.1) / 1.1), sp = mo.spot, yb = KT.clamp((ball.p[1] - sp[1] - 0.35) / 0.35, 0, 1);
          // 30 degrees above the horizon at the closest: inside the bowl, above its wall, with the gap under the ball in sight
          return { tx: sp[0], ty: sp[1] + 0.1 + 0.28 * yb + 0.25 * (1 - k), tz: sp[2], r: (4.4 - 2.7 * k) * f, th: mo.side * (0.32 + 0.3 * (1 - k)), ph: 0.9 + 0.15 * k };
        }
        if (m === 'solid') return { tx: x, ty: 0.55, tz: 0, r: 7.4 * f, th: 0.45, ph: 0.82 };
        if (m === 'brittle') return { tx: x, ty: -0.1, tz: 0, r: 8.6 * f, th: 0.45, ph: 1.0 };
        if (m === 'elastic') return { tx: x, ty: 0.6, tz: 0, r: 7.6 * f, th: 0.5, ph: 1.0 };
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
        cap.innerHTML = sc.cap; dimRows();
        silence(); quiet(); au.lastN = 0; au.first = true;        // a new scene starts in silence
        orbit.touched = false; orbit.to(view(), first);
      }

      env.loop(function (t, dt) {
        var n = 0, m = st.sc.model, v, g = orbit.goal, k, mom = m === 'moment', cap = mom ? 400 : 40;
        st.clock += dt; st.acc += dt; st.frames = (st.frames || 0) + 1;
        if (mom) while (n < cap && st.acc >= HM / mo.slow) { st.acc -= HM / mo.slow; substepMoment(); n++; }   // a sub-step costs HM / slow of real time
        else while (st.acc >= H && n < cap) { substep(); st.acc -= H; n++; }
        if (n === cap) st.acc = 0;
        listen(dt);
        if (sndT && sndT.get() === !!KT.muted) sndT.set(!KT.muted);   // the switch follows KnotTalk.muted, whoever set it
        m = st.sc.model; v = view();
        if (m === 'flight') {                                     // follow the ball; a drag still turns the camera, nothing snaps
          g.tx = v.tx; g.ty = v.ty; g.tz = v.tz; g.r = v.r;
          if (!orbit.touched) { g.th = v.th; g.ph = v.ph; }
          orbit.update(dt, fl.phase === 'air' ? 3.4 : 2.4);
        } else {
          if (!orbit.touched) { orbit.to(v); g.th = v.th + 0.16 * Math.sin(st.clock * 0.23); }
          orbit.update(dt, 2.4);
        }
        if (surf.broken) writeFragments(); else if (m === 'elastic' || m === 'flight') derive();
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
          '.kt-sim-touch .kt-side i{font-style:normal;color:var(--mute)}.kt-sim-touch .kt-side em{font-style:normal;color:var(--ok);font-weight:600}' +
          '.kt-sim-touch .kt-side em.bad{color:var(--err)}.kt-sim-touch .kt-side sub,.kt-sim-touch .kt-side sup{font-size:.8em;line-height:0}' +
          '.kt-sim-touch .kt-mo{display:none;margin:0 0 4px}.kt-sim-touch .kt-mo.on{display:block}.kt-sim-touch .kt-mo svg{display:block;overflow:visible}' +
          '.kt-sim-touch .kt-mo text{font:600 9.5px/1 var(--f-mono);letter-spacing:.02em}' +
          '.kt-sim-touch .kt-pin{position:absolute;z-index:3;left:0;top:0;pointer-events:none;white-space:nowrap;font:600 11px/1 var(--f-mono);color:var(--text);' +
          'padding:4px 6px;background:var(--panel);border:1px solid var(--line2);border-radius:5px;display:none}' +
          '.kt-sim-touch.kt-compact .kt-pin{font-size:10px}' +
          '@media (max-width:640px){.kt-sim-touch .kt-side{font-size:9.5px;padding:6px 8px}}';
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
          return { scene: st.scene, t: run.t, drop: run.k, rule: run.rule, on: ball.on, rest: ball.rest, p: ball.p.slice(), v: ball.v.slice(), speed: speed(),
                   energy: energy() / run.E0, bounces: hit.n, J: hit.J, ask: hit.ask, held: hit.held, u: cp.u, v2: cp.v, d: cp.d,
                   broken: surf.broken, pieces: frag.n, touch: el.touch, force: el.f, fmax: el.fmax, membrane: membraneEnergy() / run.E0, damped: el.lost / run.E0,
                   phase: fl.phase, vsim: fl.vsim, vset: fl.vset, boost: fl.boost, vin: fl.vin, reached: prt.nSeen, awake: prt.n, n: N, overlaps: overlaps,
                   moment: { side: mo.side, slow: mo.slow, D: mo.D, labels: CANDS.map(function (c) { return c.lab; }), t: mo.t.slice(), y: mo.y.slice(), who: mo.who.slice(),
                             near: mo.near.length, root: mo.root && { t: mo.root.t, res: mo.root.res, u: mo.root.u, v: mo.root.v, y: mo.root.y } } };
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
