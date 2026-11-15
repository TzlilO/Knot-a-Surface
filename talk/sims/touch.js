/* Knot-a-Surface talk — module `touch`: "Can't touch this".
   One bowl, stored two ways, in one world, and a solid ball (SI units: 1 kg, radius 0.25 m, g = 9.81):
     a list   Gaussians in no particular order, as an unstructured 3DGS keeps them. EMULATED here, not
              trained, exactly as in the `structure` module: the same surface sampled at jittered places,
              nudged off the surface, each turned and resized at random, then shuffled.
     ours     one cubic B-spline patch (13 x 13 control points); its Gaussians are derived from it.
   Five scenes:
     ghost    the ball is dropped on the list. There is no boundary to hit. Two honest rules alternate:
              no rule (it passes through), and "every Gaussian is a loose particle": the ball knocks away
              the ones it touches, and since the Gaussians of a list overlap where they lie, as solid
              particles they push each other apart and the bowl comes loose.
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
   All motion runs on a fixed sub-step, so a scene plays the same at any frame rate. Every number on screen
   is read off the simulation.
   Tool: WebGL (three.js). Standalone: talk/sims/touch.html.
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
  var LIGHT = [0.35, 0.86, 0.36];

  var SCENES = {
    ghost:   { cap: '<b>a ball meets a list of Gaussians</b> · a density, not a boundary', model: 'ghost', side: -1 },
    solid:   { cap: '<b>the same drop on the spline</b> · closest point, normal, reflection', model: 'solid', side: 1 },
    brittle: { cap: '<b>toughness</b> · the surface breaks when the impact asks too much of it', model: 'brittle', side: 1 },
    elastic: { cap: '<b>an elastic surface</b> · the control points sit on springs', model: 'elastic', side: 1 },
    flight:  { cap: '<b>out of bounds</b> · from the spline bowl to the list of Gaussians', model: 'flight', side: 1 }
  };
  /* where the ball is let go: x and z from the bowl's centre, height y, and the speed it is thrown down with */
  var DROPS = {
    ghost:   [[0.25, 2.6, -0.5, 0], [0.25, 2.6, -0.5, 0]],
    solid:   [[-0.35, 2.2, -0.6, 0], [0.5, 2.0, -0.45, 0], [-0.7, 2.4, 0.1, 0]],
    brittle: [[-0.2, 0.62, -0.45, 0], [-0.2, 3.3, -0.45, 0]],
    elastic: [[-0.1, 2.3, -0.15, 0], [0.4, 1.9, -0.3, 0], [-0.3, 2.7, 0.2, 0]],
    flight:  [[0.8, 3.4, 0, 12]]
  };

  KT.register('touch', {
    title: 'Can’t touch this',
    blurb: 'A ball meets one bowl stored two ways. A list of Gaussians has no boundary to collide with. A spline surface has a closest ' +
           'point and a normal, and with three material constants it is rigid, brittle or elastic. The list is emulated from the same surface, not a trained 3DGS model.',
    tool: 'WebGL · three.js · contact on the spline',
    scenes: [{ id: 'ghost', label: 'No surface' }, { id: 'solid', label: 'Rigid' }, { id: 'brittle', label: 'Brittle' },
             { id: 'elastic', label: 'Elastic' }, { id: 'flight', label: 'Out of bounds' }],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.1, 400), R = G.renderer(root, cam), scene = new T.Scene();
      var st = { scene: 'ghost', sc: SCENES.ghost, acc: 0, clock: 0 }, size = { w: 2, h: 1 };
      var mat = { e: 0.75, tough: 5, stiff: 160 };
      ['e', 'tough', 'stiff'].forEach(function (k) { if (env.opts[k] != null && isFinite(+env.opts[k])) mat[k] = +env.opts[k]; });
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var tag = ui.el('div', 'kt-side'); root.appendChild(tag);
      var pin = ui.el('div', 'kt-pin'); root.appendChild(pin);

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
      var fR = G.splats(N), fL = G.splats(N); scene.add(fR.mesh); scene.add(fL.mesh);
      var us = new Float32Array(NA), du = new Float32Array(NA), baseR = new Float32Array(N * 3), baseL = new Float32Array(N * 3);
      var gr = { pos: new Float32Array(N * 3), su: new Float32Array(N * 3), sv: new Float32Array(N * 3) };
      var grW = { pos: gr.pos, su: gr.su, sv: gr.sv, col: fR.color };   // colours are painted separately, every frame
      var home = { c: new Float32Array(N * 3) }, pr = new Float32Array(N), prMax = 0.02;
      function tone(u, v) { return ((Math.floor(u * SPANS) + Math.floor(v * SPANS)) & 1) ? 1 : 0.8; }   // the knot spans, as a checker
      function derive() { NB.patchGrid(patch, us, us, gr); G.writePlanoids(fR, 0, grW, NA, NA, du, du, 0.72); }
      (function build() {
        var i, j, k, cs = G.rgb(C.surf), cg = G.rgb(C.splat), rnd = KT.rng(7), tmp = [], spSum = 0, q;
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
          tmp.push({ p: [E.p[0] - 2 * X0 + nx / nl * off, E.p[1] + ny / nl * off, E.p[2] + nz / nl * off], t: tone(uu, vv),
                     u: [0, 1, 2].map(function (k) { return (e1[k] * ca + e2[k] * sa) * f1; }), v: [0, 1, 2].map(function (k) { return (e2[k] * ca - e1[k] * sa) * f2; }) });
        }
        for (i = tmp.length - 1; i > 0; i--) { j = Math.floor(rnd() * (i + 1)); q = tmp[i]; tmp[i] = tmp[j]; tmp[j] = q; }   // no order at all
        tmp.forEach(function (q, k) {
          fL.center.set(q.p, k * 3); fL.axU.set(q.u, k * 3); fL.axV.set(q.v, k * 3);
          baseL[k * 3] = cg[0] * 0.8 * q.t; baseL[k * 3 + 1] = cg[1] * 0.8 * q.t; baseL[k * 3 + 2] = cg[2] * 0.8 * q.t;
        });
        home.c.set(fL.center);
        for (k = 0; k < N; k++) {                                   // as a particle, a Gaussian is a sphere of its own size: the mean of its two 1-sigma axes
          pr[k] = 0.5 * (Math.hypot(fL.axU[k * 3], fL.axU[k * 3 + 1], fL.axU[k * 3 + 2]) + Math.hypot(fL.axV[k * 3], fL.axV[k * 3 + 1], fL.axV[k * 3 + 2]));
          prMax = Math.max(prMax, pr[k]);
        }
        fL.setCount(N);
        fL.uniforms.uGain.value = 0.86; fL.uniforms.uFlat.value = fR.uniforms.uFlat.value = 0.22;   // the list a little smaller, so its Gaussians read as separate things
      })();

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

      /* ---------- state ---------- */
      var ball = { p: [0, 0, 0], v: [0, 0, 0], on: false, rest: false };
      var cp = { ok: false, u: 0.5, v: 0.5, p: [0, 0, 0], su: [0, 0, 0], sv: [0, 0, 0], n: [0, 1, 0], nc: [0, 1, 0], d: 9, sd: 9, edge: false };
      var run = { k: 0, t: 0, tr: 0, hover: 0.55, still: 0, E0: 1, rule: 0 };
      var hit = { n: 0, J: 0, t: -9, ask: 0, held: true, vout: 0 };
      var surf = { broken: false, tb: 0 };
      var el = { touch: false, f: 0, fmax: 0, J: 0, lost: 0, pen: 0, sup: [] };
      var fl = { phase: 'drop', tt: 0, th: 0, vsim: 0, vset: 0, boost: false, vin: 0 };
      var prt = { v: new Float32Array(N * 3), aw: new Uint8Array(N), n: 0, seen: new Uint8Array(N), nSeen: 0, sub: 0 };

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
         resistance. In the brittle scene the surface delivers at most its toughness and then breaks. */
      function rigid(brittle) {
        closest(ball.p, cp);
        if (cp.d >= BR) { ball.rest = false; return; }
        var n = cp.nc, p = ball.p, v = ball.v, vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2], pen = BR - cp.d, k, J;
        if (vn >= 0) return;                                                        // already leaving
        if (-vn < VREST && mat.e < 0.999) {                                         // resting contact
          for (k = 0; k < 3; k++) { v[k] -= vn * n[k]; p[k] += pen * n[k]; }
          var s = speed(), dec = (MU * GRAV + CROLL * s) * H;
          if (s <= dec) v[0] = v[1] = v[2] = 0; else for (k = 0; k < 3; k++) v[k] *= (s - dec) / s;
          ball.rest = true; return;
        }
        ball.rest = false;
        var tau = Math.min(H, pen / -vn);
        fly(-tau);
        vn = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
        J = BM * (1 + mat.e) * -vn;
        hit.ask = J; hit.held = true; hit.t = run.t;
        if (brittle && J > mat.tough) { hit.held = false; J = mat.tough; shatter(cp.u, cp.v, n, J, cp.p); }
        for (k = 0; k < 3; k++) v[k] += J / BM * n[k];
        hit.J = J; hit.n++; hit.vout = speed();
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
            if (!el.touch) { el.touch = true; el.J = 0; el.fmax = 0; hit.t = run.t; }
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
        for (i = 0; i < N; i++) {
          o = i * 3; dx = P[o] - c[0]; if (dx > RR || dx < -RR) continue;
          dy = P[o + 1] - c[1]; if (dy > RR || dy < -RR) continue;
          dz = P[o + 2] - c[2]; d2 = dx * dx + dy * dy + dz * dz; rr = push ? BR + pr[i] : BR; if (d2 >= rr * rr) continue;
          if (!prt.seen[i]) { prt.seen[i] = 1; prt.nSeen++; }
          if (!push) continue;
          d = Math.sqrt(d2) || 1e-6; dx /= d; dy /= d; dz /= d;
          vr = (V[o] - bv[0]) * dx + (V[o + 1] - bv[1]) * dy + (V[o + 2] - bv[2]) * dz; if (vr >= 0) continue;
          j = -(1 + EP) * vr / (1 / MP + 1 / BM);
          V[o] += j / MP * dx; V[o + 1] += j / MP * dy; V[o + 2] += j / MP * dz;
          bv[0] -= j / BM * dx; bv[1] -= j / BM * dy; bv[2] -= j / BM * dz;
          wake(i); hits++;
        }
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

      /* ---------- the script: every scene lets the ball go by itself, again and again ---------- */
      function letGo(k) {
        var d = DROPS[st.scene], q = d[k % d.length], m = st.sc.model;
        run.k = k; run.t = 0; run.still = 0; run.rule = k % 2; run.hover = m === 'flight' ? 1.3 : 0.55;
        ball.p = [st.sc.side * X0 + q[0], q[1], q[2]]; ball.v = [0, -q[3], 0]; ball.on = false; ball.rest = false;
        cp.ok = false; hit.n = 0; hit.J = 0; hit.t = -9; hit.ask = 0; hit.held = true; hit.vout = 0; tr.n = tr.k = 0;
        run.E0 = energy(); el.lost = 0;
        if (m === 'ghost' || m === 'flight') restoreList();
        if (surf.broken) mend();
        if (m === 'flight') { flatten(); derive(); fl.phase = 'drop'; fl.tt = fl.th = 0; fl.boost = false; }
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
      }
      function substep() {
        var m = st.sc.model, s;
        run.t += H;
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
          P = fL.center; cl = fL.color; var V = prt.v, in2 = BR * BR, pass = m === 'ghost' && !run.rule;
          for (k = 0; k < N; k++) {
            o = k * 3; dx = bx - P[o]; dy = by - P[o + 1]; dz = bz - P[o + 2]; t = dx * lx + dy * ly + dz * lz; f = 1; r = 0; d2 = dx * dx + dy * dy + dz * dz;
            if (t > 0 && t < 6) { s = BR * (0.85 + 0.22 * t); f = 1 - 0.62 * Math.exp(-(d2 - t * t) / (s * s)) / (1 + 0.3 * t); }
            if (prt.aw[k]) r = Math.min(0.85, 0.3 + 0.25 * Math.hypot(V[o], V[o + 1], V[o + 2]));
            else if (pass && prt.seen[k]) r = d2 < in2 * 1.3 ? 0.9 : 0.45;        // inside the ball right now, or was
            cl[o] = baseL[o] * f * (1 - r) + ERR[0] * r; cl[o + 1] = baseL[o + 1] * f * (1 - r) + ERR[1] * r; cl[o + 2] = baseL[o + 2] * f * (1 - r) + ERR[2] * r;
          }
          fL.commit(N);
        }
      }

      /* ---------- what is drawn about the contact ---------- */
      function overlays() {
        var m = st.sc.model, onSpline = m === 'solid' || m === 'elastic' || (m === 'brittle' && !surf.broken) || (m === 'flight' && fl.phase === 'drop');
        var soft = m === 'elastic' || m === 'flight', k, i, o, q;
        ballMesh.position.set(ball.p[0], ball.p[1], ball.p[2]);
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
        // the address, written next to the contact
        var s = near ? G.project(cam, cp.p[0], cp.p[1], cp.p[2], size.w, size.h) : null;
        if (s && s[0] > 40 && s[0] < size.w - 150 && s[1] > 40 && s[1] < size.h - 30) {
          pin.style.display = 'block'; pin.style.transform = 'translate(' + (s[0] + 16).toFixed(1) + 'px,' + (s[1] + 10).toFixed(1) + 'px)';
          setText(pin, '(u, v) = (' + cp.u.toFixed(2) + ', ' + cp.v.toFixed(2) + ')');
        } else pin.style.display = 'none';
      }
      function setText(e, html) { if (e._h !== html) { e._h = html; e.innerHTML = html; } }

      /* ---------- the words: every number is read off the simulation ---------- */
      var NUM = N.toLocaleString('en-US'), LIST = '<b>a list</b> · ' + NUM + ' Gaussians, unordered <i>(emulated)</i><br>';
      var SPL = '<b>a spline chart</b> · ' + NC + ' × ' + NC + ' control points → ' + NUM + ' Gaussians<br>';
      function pct(x) { return (100 * x).toFixed(Math.abs(1 - x) < 0.02 ? 2 : 0).replace('-', '−') + ' %'; }   // below zero: lower than at rest in the bowl
      function words() {
        var m = st.sc.model, s, e = energy() / run.E0, addr = 'closest point S(<em>' + cp.u.toFixed(2) + '</em>, <em>' + cp.v.toFixed(2) + '</em>)';
        if (m === 'ghost') {
          var dv = Math.hypot(ball.v[0], ball.v[1] + (ball.on ? GRAV * (run.t - run.tr) : 0), ball.v[2]);
          s = LIST + (run.rule
            ? 'rule: every Gaussian is a loose particle · the ball touched <em class="bad">' + prt.nSeen.toLocaleString('en-US') + '</em> · <em class="bad">' + prt.n.toLocaleString('en-US') + '</em> knocked loose · it lost ' + dv.toFixed(2) + ' m/s<br>a Gaussian overlaps ' + overlaps.toFixed(1) + ' others where it lies: as solid particles they push each other apart'
            : 'no rule: the ball overlapped <em class="bad">' + prt.nSeen.toLocaleString('en-US') + '</em> Gaussians and touched <em class="bad">0</em> · its velocity changed by ' + dv.toFixed(2) + ' m/s<br>a list has a density, not a boundary: a contact needs a threshold, or a mesh extracted first');
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
        setText(tag, s);
      }

      /* ---------- controls ---------- */
      var panel = ui.panel(root, { pos: 'tr', title: 'material' });
      var sl = {
        e: ui.slider(panel, { label: 'restitution', min: 0, max: 1, step: 0.05, value: mat.e, fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { mat.e = v; } }),
        tough: ui.slider(panel, { label: 'toughness', min: 1, max: 16, step: 0.5, value: mat.tough, fmt: function (v) { return v.toFixed(1) + ' N·s'; }, onInput: function (v) { mat.tough = v; } }),
        stiff: ui.slider(panel, { label: 'stiffness', min: 60, max: 400, step: 10, value: mat.stiff, fmt: function (v) { return v + ' N/m'; }, onInput: function (v) { mat.stiff = v; } })
      };
      ui.button(panel, { label: 'drop again', kind: 'go', onClick: function () { letGo(st.sc.model === 'flight' ? 0 : run.k + (st.sc.model === 'solid' || st.sc.model === 'elastic' ? 1 : 0)); } });
      function dimRows() {
        var m = st.sc.model, use = { e: m === 'solid' || m === 'brittle', tough: m === 'brittle', stiff: m === 'elastic' || m === 'flight' }, k;
        for (k in sl) sl[k].el.style.opacity = use[k] ? 1 : 0.35;
      }

      /* ---------- camera ---------- */
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 2.5, maxRadius: 34, target: [-X0, 0.2, 0], radius: 8 });
      function fit(k) { return KT.clamp((k || 1.15) / (size.w / size.h), 1, 2.4); }     // a narrow stage stands further back
      function view() {
        var m = st.sc.model, f = fit(), x = st.sc.side * X0;
        if (m === 'ghost') return { tx: x, ty: -0.45, tz: 0, r: 8.6 * f, th: -0.42, ph: 1.06 };
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
        orbit.touched = false; orbit.to(view(), first);
      }

      env.loop(function (t, dt) {
        var n = 0, m, v, g = orbit.goal, k;
        st.clock += dt; st.acc += dt; st.frames = (st.frames || 0) + 1;
        while (st.acc >= H && n < 40) { substep(); st.acc -= H; n++; }
        if (n === 40) st.acc = 0;
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
        paint(); overlays(); words();
        (G.draw || function (R, s, c) { R.renderer.render(s, c); })(R, scene, cam);
      });

      if (!document.getElementById('kt-touch-css')) {
        var css = document.createElement('style'); css.id = 'kt-touch-css';
        css.textContent =
          '.kt-sim-touch .kt-side{position:absolute;z-index:4;left:10px;bottom:10px;pointer-events:none;font:500 11px/1.5 var(--f-mono);letter-spacing:.03em;color:var(--dim);' +
          'padding:8px 11px;background:var(--panel);border:1px solid var(--line);border-radius:8px;max-width:min(680px,calc(100% - 20px))}' +
          '.kt-sim-touch .kt-side b{color:var(--text);font-weight:600;letter-spacing:.06em;text-transform:uppercase}' +
          '.kt-sim-touch .kt-side i{font-style:normal;color:var(--mute)}.kt-sim-touch .kt-side em{font-style:normal;color:var(--ok);font-weight:600}' +
          '.kt-sim-touch .kt-side em.bad{color:var(--err)}.kt-sim-touch .kt-side sub{font-size:.8em}' +
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
        advance: function (sec) { for (var i = 0, n = Math.round(sec / H); i < n; i++) substep(); },
        stats: function () {
          return { scene: st.scene, t: run.t, drop: run.k, rule: run.rule, on: ball.on, rest: ball.rest, p: ball.p.slice(), v: ball.v.slice(), speed: speed(),
                   energy: energy() / run.E0, bounces: hit.n, J: hit.J, ask: hit.ask, held: hit.held, u: cp.u, v2: cp.v, d: cp.d,
                   broken: surf.broken, pieces: frag.n, touch: el.touch, force: el.f, fmax: el.fmax, membrane: membraneEnergy() / run.E0, damped: el.lost / run.E0,
                   phase: fl.phase, vsim: fl.vsim, vset: fl.vset, boost: fl.boost, vin: fl.vin, reached: prt.nSeen, awake: prt.n, n: N, overlaps: overlaps };
        },
        resize: function (w, h) {
          R.resize(w, h); size.w = w; size.h = h;
          cpts.material.size = KT.clamp(w / 220, 2.5, 4.5); sup.material.size = cpts.material.size * 2.2;
          if (!orbit.touched) orbit.to(view(), true);
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
