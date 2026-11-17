/* Knot-a-Surface talk — module `planoid`: "Calculus makes the splat".
   One B-spline patch: a 6 x 6 net fitted by least squares to the terrain of chapter 05 (talk/terrain.js; scene
   `net` shows the conjugate-gradient fit run from a flat net). Pick a point (u,v) and watch a Gaussian fall out of
   the surface's own calculus: position = S(u,v), orientation = tangent frame, scale = |S_u| du x |S_v| dv.
   Then fill the chart — every Gaussian's 1-sigma outline drawn crisp, one size for all against each sized to its
   own cell, with the uncovered share measured (`scale`, `fill`) — stitch the samples into a mesh with a sliding
   2 x 2 window (`stitch`), turn the resolution knob from 64 to 1,048,576 Gaussians with the frame time measured
   (`knob`), warp a fixed budget towards what a second camera sees (`adaptive`), or spend all of it inside the
   camera's view plus a margin and none outside (`window`). The learnable parameters never change.

   Reading the picture in 3D:
     - the surface is a lit sheet (darker underneath), it casts a soft shadow on the floor, and distance
       dims everything; what the sheet hides (control points, net, arrows, the tangent plane) shows through
       faintly instead of being drawn on top;
     - S_u, S_v and n are solid arrows; the two tangents have their own iso-curve each, span a visible
       parallelogram of the tangent plane, and the same plane is drawn face-on in a 2D inset with the
       angle between them. The camera rocks gently for parallax and steps aside when the plane would be
       seen edge-on;
     - scale has its spatial context: the probe is a cell of the n x n sampling lattice, drawn with its
       neighbours. The size is measured three ways and compared (see measure()): from 4 neighbours
       (central differences), from 8 neighbours (second moment of the offsets), and from the derivatives.

   Tool: WebGL (three.js) + small 2D canvases for the (u,v) chart and the tangent plane.
   Standalone: talk/sims/planoid.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color;
  /* Words: every string a viewer reads goes through T = KnotTalk.t (talk/i18n.js): that string in the page's language,
     or the English itself where there is no translation (the lab page). A sentence with a number or a name in it is
     one template with {placeholders}; the numbers are formatted as before and passed in.
     On a right-to-left page the stage stays a left-to-right island (the kit sets it): nothing in it moves, and what T
     returns carries its own direction, so it reads right in a label, a tag or on a canvas as it is. What is left to
     this module: LTRH around a formula with no words in it where it sits in the kit's panel (the one part of the stage
     that runs right to left), the numbers under the caption as a right-to-left block, and no tracking on Hebrew (the
     end of the stylesheet). What T returns is for display only: never a key, an id or a test. */
  var T = KT.t;
  function LTRH(html) { return KT.rtl ? '<span dir="ltr">' + html + '</span>' : html; }
  var CTRL_WORD = window.KnotTalk.theme === 'light' ? T('orange') : T('amber');   // the control points' colour, as the copy names it
  var NU = 6, NV = 6, SPAN = 3.2, PER_CP = 16;      // 3 position + 12 SH1 colour + 1 opacity
  var NMAX = 160, FLOOR = -1.15;
  var VEC = 0.26;            // a tangent arrow is drawn at this fraction of its true length
  var LATMAX = 40;           // up to this many samples per axis the whole sampling lattice is drawn
  var EDGE = 0.24;           // |cos| between the view ray and the normal under which the tangent plane counts as edge-on

  /* ---------- colours: KnotTalk.color, in the page's theme ----------
     The dark stage (the lab page) keeps the values it always had. On a light page (the talk) every colour is taken
     from KT.color, or mixed from two of its entries where the dark stage used a shade the palette has no entry for:
     the marks that are white on the dark stage (the probe, the normal, the derivative ring, the window) are ink,
     the pale tints of the surface and Gaussian colours become deep ones, and what was a dark chip is light glass. */
  var LIGHT = KT.theme === 'light';
  function hex3(c) { var n = parseInt(c.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
  function mix(a, b, t) {                           // '#rrggbb' between a (t = 0) and b (t = 1)
    var p = hex3(a), q = hex3(b);
    return '#' + p.map(function (x, i) { return ('0' + Math.round(x + (q[i] - x) * t).toString(16)).slice(-2); }).join('');
  }
  function rgba(c, a) { var p = hex3(c); return 'rgba(' + p[0] + ',' + p[1] + ',' + p[2] + ',' + a + ')'; }
  var INK = LIGHT ? C.text : '#ffffff';                              // the neutral marks: white on dark, ink on paper
  var SURF_HI = LIGHT ? mix(C.surf, C.text, 0.3) : '#d7f1ff';        // the surface colour, set off from it
  var SPLAT_HI = LIGHT ? mix(C.splat, C.text, 0.45) : '#f6e9fa';     // the Gaussian colour, set off from it
  var OUTL_ON = LIGHT ? mix(C.splat, C.text, 0.6) : '#2e1440';       // 1-sigma outlines drawn over the violet Gaussians: a deep violet
  // two shades of the surface colour, S_u and S_v, must not look alike: S_v leans to violet (on paper an indigo, ~35 degrees of hue
  // from the surface's azure and as far from the Gaussians' purple)
  var COL_U = C.surf, COL_V = LIGHT ? mix(C.surf, C.splat, 0.55) : '#6d8dff';
  function rim(v) { return LIGHT ? Math.min(v, 0.2) : v; }           // splat rim shading outlines each Gaussian in a dark line: faint on paper

  var OUTMAX = 48;           // up to this many samples per axis every Gaussian's 1-sigma outline is drawn
  var KN_LEVELS = [8, 16, 32, 64, 128, 256, 512, 1024];   // the knob, x4 Gaussians per step
  var KN_BUDGET = 50;        // ms: the knob turns back at a step whose frame time is above this and 3x the first step's
  var STITCH_T = 10, STITCH_LEAD = 0.8;                   // the 2 x 2 window: seconds for the whole lattice, and a pause first

  var SCENES = {
    net:      { show: 'net wire fit', fit: 1, cap: T('<b>a spline fitted to a terrain</b> · 36 control heights, least squares'),
                hint: T('drag to orbit · drag an {colour} point to reshape', { colour: CTRL_WORD }) },
    probe:    { show: 'net wire probe uv', cap: T('<b>position</b> · μ = S(u, v)'), hint: T('drag the cross in the (u,v) chart') },
    tangents: { show: 'wire probe tan iso plane face uv', rock: 1,
                cap: T('<b>tangents</b> · S<sub>u</sub> and S<sub>v</sub>, in closed form — two directions that span the tangent plane') },
    normal:   { show: 'wire probe tan nor iso plane face uv', rock: 1,
                cap: T('<b>rotation</b> · R = [ Ŝ<sub>u</sub> · Ŝ<sub>v</sub><sup>⊥</sup> · n ]') },
    scale:    { show: 'probe tan nor ell nbr face uv tile', n: 9, rock: 0.55, fixed: [0.28, 0.5],
                cap: T('<b>scale</b> · s = ( ‖S<sub>u</sub>‖Δu , ‖S<sub>v</sub><sup>⊥</sup>‖Δv , ε ) — the step to the neighbouring samples'),
                hint: T('drag in the (u,v) chart · 4 / 8 neighbours · one size / own cell') },
    derive:   { show: 'net probe tan nor ell nbr face', n: 9, wig: 1, fixed: [0.56, 0.44], rock: 0.4,
                cap: T('<b>nothing is learned per Gaussian</b> · move the surface, the splat and its neighbours re-derive') },
    fill:     { show: 'field uv tile', n: 14, rim: 1, cap: T('<b>fill the chart</b> · one Gaussian per (u,v) sample · outline = 1σ'),
                hint: T('one size / own cell · take the samples slider') },
    stitch:   { show: 'field uv stitch', n: 12, rim: 1, view: { th: 0.72, ph: 0.86, r: 6.9 },
                cap: T('<b>stitch</b> · a 2 × 2 window over the (u,v) lattice: two triangles per stop'),
                hint: T('take the samples slider to run it again') },
    knob:     { show: 'field', sweep: 1, rim: 0.7, cap: T('<b>the resolution knob</b> · re-evaluate, don’t retrain'),
                hint: T('take the Gaussians slider') },
    adaptive: { show: 'uv adapt', n: 32, view: { th: 1.52, ph: 1.06, r: 8.4, ty: 0.55 },
                cap: T('<b>adaptive sampling</b> · a fixed budget, warped towards what a camera sees'),
                hint: T('drag to orbit · we watch from the side') },
    window:   { show: 'uv adapt win', n: 32, view: { th: 1.52, ph: 1.06, r: 8.4, ty: 0.55 },
                cap: T('<b>view window</b> · every Gaussian inside what a camera sees, plus a margin'),
                hint: T('drag to orbit · take the buffer slider') },
    features: { show: 'net field feat', n: 72, rim: 0.4, cap: T('<b>control features</b> · colour and opacity ride the same basis') },
    hero:     { show: 'wire field feat', n: 64, rim: 0.5, hero: 1, cap: '' }
  };

  /* ---------- small vector / symmetric-matrix helpers ---------- */
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function len(a) { return Math.sqrt(dot(a, a)); }
  function mul(a, k) { return [a[0] * k, a[1] * k, a[2] * k]; }
  function add(a, b, k) { if (k == null) k = 1; return [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  /* S += k a a^T, S = [xx, xy, xz, yy, yz, zz] */
  function sym(S, a, k) { S[0] += k * a[0] * a[0]; S[1] += k * a[0] * a[1]; S[2] += k * a[0] * a[2]; S[3] += k * a[1] * a[1]; S[4] += k * a[1] * a[2]; S[5] += k * a[2] * a[2]; }
  function fro(S) { return Math.sqrt(S[0] * S[0] + S[3] * S[3] + S[5] * S[5] + 2 * (S[1] * S[1] + S[2] * S[2] + S[4] * S[4])); }
  /* how far two covariances are apart, relative to the second: ||A - B||_F / ||B||_F */
  function gap(A, B) { var D = [], i; for (i = 0; i < 6; i++) D.push(A[i] - B[i]); return fro(D) / (fro(B) || 1e-30); }
  /* eigen-decomposition of a symmetric 3x3 (cyclic Jacobi): { l: eigenvalues, largest first, v: unit eigenvectors } */
  function eig3(S) {
    var a = [[S[0], S[1], S[2]], [S[1], S[3], S[4]], [S[2], S[4], S[5]]], v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], it, p, q, k;
    for (it = 0; it < 30; it++) {
      p = 0; q = 1;
      if (Math.abs(a[0][2]) > Math.abs(a[p][q])) { p = 0; q = 2; }
      if (Math.abs(a[1][2]) > Math.abs(a[p][q])) { p = 1; q = 2; }
      if (Math.abs(a[p][q]) < 1e-15 * (Math.abs(a[0][0]) + Math.abs(a[1][1]) + Math.abs(a[2][2])) + 1e-300) break;
      var th = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]), c = Math.cos(th), s = Math.sin(th), x, y;
      for (k = 0; k < 3; k++) { x = a[k][p]; y = a[k][q]; a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
      for (k = 0; k < 3; k++) { x = a[p][k]; y = a[q][k]; a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
      for (k = 0; k < 3; k++) { x = v[k][p]; y = v[k][q]; v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
    }
    var o = [0, 1, 2].sort(function (i, j) { return a[j][j] - a[i][i]; });
    return { l: o.map(function (i) { return Math.max(0, a[i][i]); }),
             v: o.map(function (i) { return [v[0][i], v[1][i], v[2][i]]; }) };
  }

  /* ---------- the size of one Gaussian, measured three ways ----------
     The sample at (u, v) sits in a lattice with spacing D in u and in v. With d(a,b) = S(u + aD, v + bD) - S(u, v):

       derivatives (the method)   s = ( |S_u| D , |S_v^perp| D ) in the frame [ S_u^ , S_v^perp^ ]:
                                  covariance GS = D^2 ( S_u S_u^T + S_v^perp S_v^perp^T )
       4 neighbours               central differences: a_u = ( d(1,0) - d(-1,0) ) / 2 , a_v = ( d(0,1) - d(0,-1) ) / 2
                                  — half the chord between opposite neighbours — then the same two lengths
                                  ( |a_u| , |a_v^perp| ). a_u = D S_u + O(D^3), so this converges to GS.
       8 neighbours               the second moment of all eight offsets, S8 = (1/6) sum d d^T. The factor 6 is
                                  what the same sum gives in the chart (sum of a^2 over the stencil), so to first
                                  order S8 = J diag(D, D)^2 J^T with J = [ S_u S_v ]: the cell pushed through the
                                  Jacobian, shear included. It converges to JF = D^2 ( S_u S_u^T + S_v S_v^T ),
                                  NOT to GS: the two-length formula keeps the area (det GS = det JF) and drops
                                  the shear S_u . S_v. `shear` is that gap; it does not shrink with D.
                                  S8 also has a small third axis along n: neighbours on a curved surface sag
                                  off the tangent plane, so a neighbour fit is never exactly flat.

     Every gap is ||A - B||_F / ||B||_F over 3x3 covariances. Returns the frame, the offsets and each
     estimate as two in-surface axes (1 sigma before the sigma/cell factor) plus its lengths s. */
  function measure(patch, u, v, D, G9) {
    NB.patchGrid(patch, [u - D, u, u + D], [v - D, v, v + D], G9);
    var P = G9.pos, c = [P[12], P[13], P[14]], su = [G9.su[12], G9.su[13], G9.su[14]], sv = [G9.sv[12], G9.sv[13], G9.sv[14]];
    var lu = len(su), lsv = len(sv), svp = add(sv, su, -dot(su, sv) / (lu * lu)), lvp = len(svp);
    var nr = cross(su, svp), ln = len(nr) || 1; if (nr[1] < 0) ln = -ln; nr = mul(nr, 1 / ln);      // the normal that points up
    var e1 = mul(su, 1 / lu), e2 = cross(nr, e1), d = [], k;
    for (k = 0; k < 9; k++) d.push([P[k * 3] - c[0], P[k * 3 + 1] - c[1], P[k * 3 + 2] - c[2]]);      // k = 3 (b + 1) + (a + 1)
    var GS = [0, 0, 0, 0, 0, 0], JF = [0, 0, 0, 0, 0, 0], S4 = [0, 0, 0, 0, 0, 0], S8 = [0, 0, 0, 0, 0, 0];
    sym(GS, su, D * D); sym(GS, svp, D * D); sym(JF, su, D * D); sym(JF, sv, D * D);
    var au = mul(add(d[5], d[3], -1), 0.5), av = mul(add(d[7], d[1], -1), 0.5), avp = add(av, au, -dot(au, av) / dot(au, au));
    sym(S4, au, 1); sym(S4, avp, 1);
    for (k = 0; k < 9; k++) if (k !== 4) sym(S8, d[k], 1 / 6);
    var eJ = eig3(JF), e8 = eig3(S8), cs = KT.clamp(dot(su, sv) / (lu * lsv), -1, 1);
    function axes(e) { return [mul(e.v[0], Math.sqrt(e.l[0])), mul(e.v[1], Math.sqrt(e.l[1]))]; }
    return { c: c, su: su, sv: sv, svp: svp, n: nr, e1: e1, e2: e2, lu: lu, lsv: lsv, lvp: lvp, D: D, d: d,
      ang: Math.acos(cs) * 180 / Math.PI,
      gs: { ax: [mul(su, D), mul(svp, D)], s: [lu * D, lvp * D, 0] },
      jf: { ax: axes(eJ), s: [Math.sqrt(eJ.l[0]), Math.sqrt(eJ.l[1]), 0] },
      n4: { ax: [au, avp], s: [len(au), len(avp), 0] },
      n8: { ax: axes(e8), s: [Math.sqrt(e8.l[0]), Math.sqrt(e8.l[1]), Math.sqrt(e8.l[2])] },
      gap4: gap(S4, GS), gap8: gap(S8, JF), shear: gap(JF, GS) };
  }

  KT.register('planoid', {
    title: 'Calculus makes the splat',
    blurb: 'A Gaussian derived, not learned: position, orientation and scale read straight off a spline ' +
           'surface’s partial derivatives. Fill the (u,v) chart and change the sample count — the parameters stay put.',
    tool: 'WebGL · three.js',
    scenes: [
      { id: 'net', label: 'Control net' }, { id: 'probe', label: 'Position' },
      { id: 'tangents', label: 'Tangents' }, { id: 'normal', label: 'Rotation' },
      { id: 'scale', label: 'Scale' }, { id: 'derive', label: 'Derived, not learned' },
      { id: 'fill', label: 'Fill the chart' }, { id: 'stitch', label: 'Stitch a mesh' },
      { id: 'knob', label: 'Resolution knob' }, { id: 'adaptive', label: 'Adaptive sampling' },
      { id: 'window', label: 'View window' }, { id: 'features', label: 'Control features' }
    ],
    create: function (env) {
      var THREE = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new THREE.PerspectiveCamera(34, 1, 0.1, 60);
      var R = G.renderer(root, cam), scene = new THREE.Scene();
      var W = 1, H = 1, fitR = 1;

      /* light and air: the same light direction as the splat shader, and a fog that follows the camera */
      var LIGHT_DIR = new THREE.Vector3(0.35, 0.86, 0.36).normalize();
      scene.fog = new THREE.Fog(LIGHT ? C.bg : 0x07090f, 5, 14);          // distance fades into the stage: dark, or the glass card
      scene.add(new THREE.HemisphereLight(0xe6efff, 0x2a3650, 1.5));
      var sun = new THREE.DirectionalLight(0xffffff, 2.6); sun.position.copy(LIGHT_DIR); scene.add(sun);

      /* ---------- the patch: the terrain of chapter 05 (talk/terrain.js), held by this 6 x 6 net ----------
         The control points stand on a regular grid in x and z; only their heights are fitted. ground(x, z) is the
         target: the terrain on [0,1]^2 laid over the patch's square, normalised so its relief spans HS (0.35 of the
         width, the proportions chapter 05 uses). The fit is linear least squares over FG x FG samples at the cell
         centres of (u, v): the height of S(u, v) against the terrain at the same (x, z), each sample weighted by the
         area of the map it stands for, |x_u z_v - x_v z_u| (the even control points crowd the samples in the middle
         of the map; unweighted, the crater there would count for more). So the error is the RMS over the map, the
         measure chapter 05 takes on an even grid of it. Scene `net` shows it run —
         Jacobi-preconditioned conjugate gradients from a flat net, each step a glide along its search direction to
         the minimum on that line; every other scene starts from the converged net (`base`). Six control points a
         side hold the rolling ground and blur the crater: that is where chapter 05 starts. */
      var TERR = typeof KT.terrain === 'function' ? KT.terrain : null;
      var HS = 0.35 * SPAN, HB = -0.45, tLo = 0, tHi = 1;
      if (TERR) (function () {
        tLo = 1e9; tHi = -1e9;
        for (var b = 0; b < 128; b++) for (var a = 0; a < 128; a++) { var h = TERR((a + 0.5) / 128, (b + 0.5) / 128); if (h < tLo) tLo = h; if (h > tHi) tHi = h; }
      })();
      /* the area of the map (x, z) a sample stands for: |x_u z_v - x_v z_u| from the tangents at float offset t */
      function mapArea(SU, SV, t) { return Math.abs(SU[t] * SV[t + 2] - SU[t + 2] * SV[t]); }
      function ground(x, z) {
        if (TERR) return HB + HS * (TERR(x / SPAN + 0.5, z / SPAN + 0.5) - tLo) / (tHi - tLo);
        x = x / SPAN * 2; z = z / SPAN * 2;          // without talk/terrain.js: the module's old analytic bump stands in
        return Math.exp(-((x - 0.25) * (x - 0.25) + (z + 0.2) * (z + 0.2)) * 2.4) - 0.55 * Math.exp(-((x + 0.6) * (x + 0.6) + (z - 0.55) * (z - 0.55)) * 3) + 0.14 * x;
      }
      var patch = NB.makePatch({ nu: NU, nv: NV, p: 3 }), base = new Float32Array(NU * NV * 3), fitBase = new Float32Array(NU * NV * 3);
      (function () {
        var A = [0.12, 0.47, 0.90], B = [1.0, 0.44, 0.26], D = [0.80, 0.58, 0.85], i, j, o, s, t;
        for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
          o = (j * NU + i) * 3; s = i / (NU - 1); t = j / (NV - 1);
          base[o] = (s * 2 - 1) * SPAN / 2; base[o + 2] = (t * 2 - 1) * SPAN / 2;
          for (var c = 0; c < 3; c++) patch.C[o + c] = KT.lerp(KT.lerp(A[c], B[c], s), D[c], t * 0.75);
        }
        o = (2 * NU + 3) * 3; patch.C[o] = 1; patch.C[o + 1] = 0.86; patch.C[o + 2] = 0.30;   // one loud control feature
        patch.P.set(base);
      })();
      var FG = 40, FIT = (function () {          // the normal equations, once: A = B^T W B / N, rhs = B^T W t / N, tt = t.W t / N
        var n = NU * NV, us = new Float64Array(FG), a, b, i, j, q, r, s, N = FG * FG;
        for (a = 0; a < FG; a++) us[a] = (a + 0.5) / FG;
        var gp = { pos: new Float32Array(N * 3), su: new Float32Array(N * 3), sv: new Float32Array(N * 3) };
        NB.patchGrid(patch, us, us, gp);          // x(u), z(v): heights do not enter
        var TB = NB.axisTable(patch.Uk, NU, 3, us), A = new Float64Array(n * n), rhs = new Float64Array(n), tg = new Float64Array(N), wt = new Float64Array(N);
        var idx = new Int32Array(16), w = new Float64Array(16), tt = 0, mean = 0, ws = 0, k;
        for (s = 0; s < N; s++) { wt[s] = mapArea(gp.su, gp.sv, s * 3); ws += wt[s]; }
        for (s = 0; s < N; s++) wt[s] *= N / ws;                         // mean weight 1
        for (b = 0; b < FG; b++) for (a = 0; a < FG; a++) {
          s = b * FG + a; k = wt[s]; tg[s] = ground(gp.pos[s * 3], gp.pos[s * 3 + 2]); tt += k * tg[s] * tg[s]; mean += k * tg[s];
          for (q = 0; q < 4; q++) for (r = 0; r < 4; r++) { idx[q * 4 + r] = (TB.span[b] - 3 + q) * NU + TB.span[a] - 3 + r; w[q * 4 + r] = TB.N[a * 4 + r] * TB.N[b * 4 + q]; }
          for (i = 0; i < 16; i++) { rhs[idx[i]] += k * w[i] * tg[s]; for (j = 0; j < 16; j++) A[idx[i] * n + idx[j]] += k * w[i] * w[j]; }
        }
        for (i = 0; i < n * n; i++) A[i] /= N;
        for (i = 0; i < n; i++) rhs[i] /= N;
        return { n: n, A: A, b: rhs, tt: tt / N, mean: mean / N, us: us, pos: gp.pos, tg: tg };
      })();
      function mulA(X, out) { var n = FIT.n, i, j, s; for (i = 0; i < n; i++) { s = 0; for (j = 0; j < n; j++) s += FIT.A[i * n + j] * X[j]; out[i] = s; } return out; }
      function dotN(x, y) { var s = 0; for (var i = 0; i < x.length; i++) s += x[i] * y[i]; return s; }
      /* RMS of (surface - terrain) over the FG x FG samples, from the normal equations: tt - 2 b.Z + Z.AZ */
      var fitTmp = new Float64Array(NU * NV);
      function fitRms(Z) { return Math.sqrt(Math.max(0, FIT.tt - 2 * dotN(FIT.b, Z) + dotN(Z, mulA(Z, fitTmp)))); }
      function heights(out) { for (var i = 0; i < NU * NV; i++) out[i] = patch.P[i * 3 + 1]; return out; }
      /* conjugate gradients, one line search at a time: cgAim picks the step (al along p), cgLand takes it */
      function cgNew(Z) {
        var n = FIT.n, c = { Z: Z, r: new Float64Array(n), z: new Float64Array(n), p: new Float64Array(n), Ap: new Float64Array(n),
                             from: new Float64Array(n), it: 0, tau: 0, done: false }, i;
        mulA(Z, c.Ap);
        for (i = 0; i < n; i++) { c.r[i] = FIT.b[i] - c.Ap[i]; c.z[i] = c.r[i] / FIT.A[i * n + i]; c.p[i] = c.z[i]; }
        c.rz = c.rz0 = dotN(c.r, c.z); cgAim(c);
        return c;
      }
      function cgAim(c) {
        mulA(c.p, c.Ap); var pAp = dotN(c.p, c.Ap);
        c.al = pAp > 0 ? c.rz / pAp : 0; c.from.set(c.Z); c.tau = 0;
        c.slice = 0.05 + 0.9 * Math.pow(0.7, c.it);               // how long this step is shown taking, in seconds (about 4.3 s in all)
        if (!(pAp > 0) || c.rz <= 1e-18 * c.rz0) c.done = true;
      }
      function cgLand(c) {
        var n = FIT.n, i, rz, be;
        for (i = 0; i < n; i++) { c.Z[i] = c.from[i] + c.al * c.p[i]; c.r[i] -= c.al * c.Ap[i]; c.z[i] = c.r[i] / FIT.A[i * n + i]; }
        rz = dotN(c.r, c.z); be = rz / c.rz; c.rz = rz; c.it++;
        for (i = 0; i < n; i++) c.p[i] = c.z[i] + be * c.p[i];
        if (c.rz <= 1e-18 * c.rz0 || c.it >= 4 * n) c.done = true; else cgAim(c);
      }
      (function () {                             // the converged net: the rest pose of every scene but `net`
        var c = cgNew(new Float64Array(FIT.n).fill(FIT.mean)), i;
        while (!c.done) cgLand(c);
        for (i = 0; i < FIT.n; i++) base[i * 3 + 1] = c.Z[i];
        FIT.rms = fitRms(c.Z); FIT.steps = c.it; fitBase.set(base); patch.P.set(base);
      })();
      /* scene net: one CG line search at a time, shown as a glide (eased) from where it starts to where it lands */
      function fitTick(dt) {
        var c = st.fit, i;
        if (!c || c.done || c.user) return;
        c.tau += dt / c.slice;
        if (c.tau >= 1) cgLand(c);
        var e = c.done ? 1 : KT.ease(c.tau);
        for (i = 0; i < FIT.n; i++) patch.P[i * 3 + 1] = c.done ? c.Z[i] : c.from[i] + e * c.al * c.p[i];
        surfDirty = true;
      }
      /* measured on the net as shown: RMS (weighted by the map's area, as the fit) and largest |surface - terrain| over all
         FG x FG samples, as a share of the relief, against the terrain under each sample where it is now (a dragged
         control point moves samples in x and z too); a red tick at every other sample */
      var rgFull = { pos: new Float32Array(FG * FG * 3), su: new Float32Array(FG * FG * 3), sv: new Float32Array(FG * FG * 3) },
          rgT = new Float64Array(FG * FG), fitStat = { rms: 0, max: 0 };
      function updateResid() {
        NB.patchGrid(patch, FIT.us, FIT.us, rgFull);
        var P = rgFull.pos, A = resid.arr, se = 0, ws = 0, mx = 0, e, s, a, b, k, o = 0, N = FG * FG, c = st.fit;
        for (s = 0; s < N; s++) {
          rgT[s] = ground(P[s * 3], P[s * 3 + 2]); e = Math.abs(P[s * 3 + 1] - rgT[s]); k = mapArea(rgFull.su, rgFull.sv, s * 3);
          se += k * e * e; ws += k; if (e > mx) mx = e;
        }
        for (b = 0; b < TD; b++) for (a = 0; a < TD; a++) {
          s = (b * 2 + 1) * FG + a * 2 + 1;
          A[o] = P[s * 3]; A[o + 1] = P[s * 3 + 1]; A[o + 2] = P[s * 3 + 2]; A[o + 3] = P[s * 3]; A[o + 4] = rgT[s]; A[o + 5] = P[s * 3 + 2]; o += 6;
        }
        resid.set(o / 3);
        fitStat.rms = Math.sqrt(se / (ws || 1)) / HS; fitStat.max = mx / HS;
        rdFit.set(pct(fitStat.rms)); rdMax.set(pct(fitStat.max));
        rdIt.set(!c ? '—' : c.user ? T('stopped: a point was moved') : c.done ? T('converged · {n} CG steps', { n: c.it }) : T('CG step {n}', { n: c.it + 1 }));
        hudDirty = true;
      }
      function fmtFps(ms) { var f = 1000 / ms; return (f < 9.95 ? f.toFixed(1) : Math.round(f)) + ' fps'; }
      function fmtMs(ms) { return ms == null ? '—' : ms < 10 ? ms.toFixed(1) + ' ms' : Math.round(ms) + ' ms'; }
      function pctW(x) { return x == null ? '—' : (x * 100).toFixed(x < 0.1 ? 1 : 0) + ' %'; }
      /* the numbers under the caption, for the scene on stage */
      function drawHud() {
        var s = st.show, sc = st.sc, h = '', A, B, a;
        if (s.fit && st.fit) {
          var c = st.fit;
          h = (c.user ? T('least squares · stopped: a control point was moved') : c.done ? T('least squares · converged in <b>{n}</b> conjugate-gradient steps', { n: c.it })
               : T('least squares · conjugate gradients, step <b>{n}</b>', { n: c.it + 1 })) +
              '<br>' + T('RMS error <b class="e">{rms}</b> of the relief · largest <b class="e">{max}</b>', { rms: pct(fitStat.rms), max: pct(fitStat.max) });
        } else if (s.tile && tl.own) {
          var one = tl.b >= 0.5; A = one ? tl.one : tl.own; B = one ? tl.own : tl.one;
          h = T('<b class="s">{size}</b> · uncovered <b class="e">{unc}</b> · 3+ deep <b>{pile}</b>',
                { size: one ? T('one size for all') : T('each sized to its cell'), unc: pct(A.unc), pile: pct(A.pile) }) +
              '<br>' + T('<span class="m">{size}: uncovered {unc} · 3+ deep {pile}</span>', { size: one ? T('sized to cells') : T('one size'), unc: pct(B.unc), pile: pct(B.pile) });
        } else if (s.stitch && sx.n) {
          h = T('<b class="u">{done}</b> / {all} triangles · <b class="s">{left}</b> / {n} Gaussians still drawn', { done: sx.q * 2, all: sx.Q * 2, left: sx.V - sx.passed, n: sx.V });
        } else if (sc.sweep) {
          var ft = kn.ft[kn.cur];
          h = T('<span class="big s">{n}</span> Gaussians from <span class="big c">{stored}</span> stored numbers', { n: (kn.cur * kn.cur).toLocaleString('en-US'), stored: NU * NV * PER_CP }) +
              '<br>' + (ft == null ? T('frame time measuring… · re-evaluated in <b>{t}</b>', { t: fmtMs(kn.ev[kn.cur]) })
                        : T('frame time <b>{ms}</b> ({fps}) · re-evaluated in <b>{t}</b>', { ms: ft.toFixed(1) + ' ms', fps: fmtFps(ft), t: fmtMs(kn.ev[kn.cur]) }));
        } else if (s.adapt && ad.stat) {
          a = ad.stat;
          var eq = a.sIn && a.uIn ? Math.round(a.n * (a.uIn / a.sIn) * (a.uIn / a.sIn)) : null;
          h = T('<b class="s">{n}</b> Gaussians, fixed · <b class="s">{share}</b> of them in the camera’s view', { n: a.n.toLocaleString('en-US'), share: Math.round(100 * a.ins / a.n) + ' %' }) +
              '<br>' + T('the view covers <b>{share}</b> of the surface', { share: Math.round(100 * a.view) + ' %' }) +
              '<br>' + T('cell side there <b>{side}</b> of the width · even lattice <b>{even}</b>', { side: pctW(a.sIn), even: pctW(a.uIn) });
          rdIn.set(a.ins.toLocaleString('en-US') + ' · ' + Math.round(100 * a.ins / a.n) + ' %'); rdFp.set(T('{share} of the surface', { share: Math.round(100 * a.view) + ' %' }));
          rdSpc.el.lastChild.innerHTML = pctW(a.sIn) + ' <i>·</i> ' + pctW(a.sOut); rdUni.set(pctW(a.uIn));
          rdEq.set(eq ? T('≈ {n} Gaussians', { n: eq.toLocaleString('en-US') }) : '—');
        }
        h = h.replace(/ %/g, '\u00a0%');                  // a number keeps its unit on its line
        if (h !== hudHtml) { hudHtml = h; hud.innerHTML = h; hud.style.display = h ? '' : 'none'; hudPlace(); }
      }
      function hudPlace() {                       // caption and numbers clear of the panel while that is open, the numbers under the caption
        var pw = panel.offsetParent ? panel.offsetWidth + 30 : 10, mw = Math.round(KT.clamp(root.clientWidth - 14 - pw, 160, 620)) + 'px';
        hud.style.maxWidth = mw; cap.style.maxWidth = root.classList.contains('kt-compact') ? '' : mw;
        hud.style.top = Math.round(cap.offsetTop + cap.offsetHeight + 7) + 'px';
      }

      /* ---------- scene objects ----------
         Draw order: opaque things (control points, arrows, splat field) first, then by renderOrder:
         floor glow -6, shadow -4, sheet 0, lines on it 1-2, "hidden" copies 3+, Gaussians 9-10. */
      var a, b;
      function own(o, order) { o.frustumCulled = false; if (order != null) o.renderOrder = order; scene.add(o); return o; }
      function ghostLines(l, opacity, order) {              // the part of a line set that lies behind something
        return own(new THREE.LineSegments(l.obj.geometry, G.hidden(l.mat.color, opacity, 'line')), order);
      }

      // the floor: a grid (on paper the line colour of the page, its centre lines a shade darker) and a soft pool of the
      // surface colour under the patch (on the dark stage a pale glow)
      var grid = LIGHT ? new THREE.GridHelper(9, 18, mix(C.line, C.dim, 0.3), C.line) : new THREE.GridHelper(9, 18, 0x243652, 0x162238);
      grid.position.y = FLOOR; scene.add(grid);
      var glowC = LIGHT ? G.rgb(C.surf).map(function (x) { return x.toFixed(3); }).join(', ') + ', a * a * 0.06' : '0.20, 0.33, 0.52, a * a * 0.30';
      var glow = own(new THREE.Mesh(new THREE.CircleGeometry(5.4, 48).rotateX(-Math.PI / 2), new THREE.ShaderMaterial({
        transparent: true, depthWrite: false,
        vertexShader: 'varying vec2 vP; void main(){ vP = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: 'varying vec2 vP; void main(){ float a = 1.0 - clamp(length(vP) / 5.4, 0.0, 1.0); gl_FragColor = vec4(' + glowC + '); }'
      })), -6);
      glow.position.y = FLOOR - 0.004;

      var FN = 33, sheet = G.sheet(FN, FN), fill = own(sheet.mesh, 0);
      var sg = { pos: sheet.pos, su: new Float32Array(FN * FN * 3), sv: new Float32Array(FN * FN * 3) };
      // the sheet's shadow: the same geometry flattened onto the floor along the light, three times for a soft edge
      // (black on the dark stage; on paper a light shade of ink, or the three layers make a slab)
      var shadowMat = new THREE.MeshBasicMaterial({ color: LIGHT ? C.text : 0x000000, transparent: true, opacity: LIGHT ? 0.045 : 0.2, depthWrite: false, side: THREE.DoubleSide, fog: false });
      var shadows = [0.93, 1.0, 1.08].map(function (k, i) {
        var m = own(new THREE.Mesh(sheet.geo, shadowMat), -4), sx = -LIGHT_DIR.x / LIGHT_DIR.y, sz = -LIGHT_DIR.z / LIGHT_DIR.y;
        m.matrixAutoUpdate = false; m.matrixWorldNeedsUpdate = true;
        m.matrix.set(k, sx, 0, -sx * FLOOR,   0, 1e-4, 0, FLOOR + 0.004 + 0.002 * i,   0, sz, k, -sz * FLOOR,   0, 0, 0, 1);
        return m;
      });

      var WL = 13, WS = 40, wire = G.lines(2 * WL * WS * 2, C.surf, 0.55); own(wire.obj, 1);
      var net = G.lines(((NU - 1) * NV + NU * (NV - 1)) * 2, C.ctrl, 0.7); own(net.obj, 2);
      var netGhost = ghostLines(net, 0.2, 3);
      var balls = [], ballGeo = new THREE.SphereGeometry(0.06, 20, 14), ballHid = G.hidden(C.ctrl, 0.3);
      for (a = 0; a < NU * NV; a++) {
        var m = new THREE.Mesh(ballGeo, new THREE.MeshLambertMaterial({ color: C.ctrl })); m.userData.i = a;
        var mh = new THREE.Mesh(ballGeo, ballHid); mh.renderOrder = 3; m.add(mh);
        balls.push(m); scene.add(m);
      }

      var field = G.splats(NMAX * NMAX); scene.add(field.mesh);
      var violet = G.rgb(C.splat);

      // the probe and its frame
      var probe = new THREE.Mesh(new THREE.SphereGeometry(0.05, 20, 14), new THREE.MeshLambertMaterial({ color: INK, emissive: LIGHT ? 0x000000 : 0x6a6a6a }));
      var probeHid = new THREE.Mesh(probe.geometry, G.hidden(INK, 0.4)); probeHid.renderOrder = 8; probe.add(probeHid); scene.add(probe);
      var aU = G.vector(COL_U), aV = G.vector(COL_V), aN = G.vector(INK);
      scene.add(aU.obj); scene.add(aV.obj); scene.add(aN.obj);
      var IS = 48, isoU = G.lines(IS * 2, COL_U, 0.95), isoV = G.lines(IS * 2, COL_V, 0.95); own(isoU.obj, 2); own(isoV.obj, 2);
      var ig = { pos: new Float32Array((IS + 1) * 3) }, isoT = lin(IS + 1);

      // the tangent plane: a faint patch of it, and the parallelogram the two tangents span
      var plPos = new Float32Array(8 * 3), plGeo = new THREE.BufferGeometry();
      plGeo.setAttribute('position', new THREE.BufferAttribute(plPos, 3)); plGeo.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
      plGeo.addGroup(0, 6, 0); plGeo.addGroup(6, 6, 1);
      function flat(color, opacity) { return new THREE.MeshBasicMaterial({ color: color, transparent: true, opacity: opacity, side: THREE.DoubleSide, depthWrite: false }); }
      var plane = own(new THREE.Mesh(plGeo, [flat(C.surf, 0.09), flat(C.surf, 0.22)]), 4);
      var planeHid = own(new THREE.Mesh(plGeo, G.hidden(C.surf, 0.045, 'flat')), 5);
      var plGrid = G.lines(40, C.surf, 0.26), plEdge = G.lines(8, SURF_HI, 0.85), mark = G.lines(8, INK, 0.9);
      own(plGrid.obj, 5); own(plEdge.obj, 6); own(mark.obj, 6);
      var plEdgeHid = ghostLines(plEdge, 0.25, 7);

      // the Gaussian at the probe is painted on the sheet (sheet.gauss); these are the 1-sigma outlines of the
      // estimates, flat in the tangent plane (ribbons: a 1 px line would vanish)
      var EL = 72;
      function ribbon(color, opacity, order) {
        var pos = new Float32Array((EL + 1) * 2 * 3), g = new THREE.BufferGeometry(), ix = [], i;
        for (i = 0; i < EL; i++) ix.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
        g.setAttribute('position', new THREE.BufferAttribute(pos, 3)); g.setIndex(ix);
        var mesh = own(new THREE.Mesh(g, flat(color, opacity)), order), hid = own(new THREE.Mesh(g, G.hidden(color, opacity * 0.3, 'flat')), order);
        return { mesh: mesh, hid: hid,
          /* centre c, axes A and B (1 sigma), lifted by `lift` along nr, half-width hw as a fraction of the radius */
          set: function (c, A, B, nr, lift, hw) {
            for (var i = 0; i <= EL; i++) {
              var co = Math.cos(i / EL * 6.2832), si = Math.sin(i / EL * 6.2832), k, r, o;
              for (k = 0; k < 2; k++) {
                r = k ? 1 + hw : 1 - hw; o = (i * 2 + k) * 3;
                pos[o] = c[0] + (A[0] * co + B[0] * si) * r + nr[0] * lift;
                pos[o + 1] = c[1] + (A[1] * co + B[1] * si) * r + nr[1] * lift;
                pos[o + 2] = c[2] + (A[2] * co + B[2] * si) * r + nr[2] * lift;
              }
            }
            g.attributes.position.needsUpdate = true;
          } };
      }
      var ringN = ribbon(C.splat, 0.95, 11), ringA = ribbon(INK, 0.95, 12);

      // the neighbourhood: the lattice on the surface, the 3 x 3 block, the eight neighbours and their Gaussians
      var LS = 36, lat = G.lines(2 * (LATMAX + 1) * LS * 2, C.surf, 0.3); own(lat.obj, 1);
      var lg = { pos: new Float32Array((LS + 1) * (LATMAX + 1) * 3) };
      var dotTex = (function () {
        var cv = document.createElement('canvas'); cv.width = cv.height = 32;
        var g = cv.getContext('2d'); g.fillStyle = '#fff'; g.beginPath(); g.arc(16, 16, 14, 0, 6.2832); g.fill();
        return new THREE.CanvasTexture(cv);
      })();
      var dotPos = new Float32Array(LATMAX * LATMAX * 3), dotGeo = new THREE.BufferGeometry();
      dotGeo.setAttribute('position', new THREE.BufferAttribute(dotPos, 3)); dotGeo.setDrawRange(0, 0);
      var dots = own(new THREE.Points(dotGeo, new THREE.PointsMaterial({ color: C.splat, size: 0.05, map: dotTex, alphaTest: 0.5, transparent: true, opacity: 0.75 })), 2);
      var BS = 4, BN = 3 * BS + 1, bgd = { pos: new Float32Array(BN * BN * 3) };
      var blk = G.lines(8 * (BN - 1) * 2, SURF_HI, 0.8); own(blk.obj, 2);
      var nbGeo = new THREE.SphereGeometry(0.036, 16, 12), cs = new THREE.Color(C.splat);
      var nbOn = new THREE.MeshLambertMaterial({ color: cs, emissive: cs.clone().multiplyScalar(0.35) });
      var nbOff = new THREE.MeshLambertMaterial({ color: cs.clone().multiplyScalar(0.42), emissive: cs.clone().multiplyScalar(0.1) });
      var nbHid = G.hidden(C.splat, 0.3), nbDot = [];
      for (a = 0; a < 8; a++) {
        var nd = new THREE.Mesh(nbGeo, nbOn), nh = new THREE.Mesh(nbGeo, nbHid); nh.renderOrder = 8; nd.add(nh); nbDot.push(nd); scene.add(nd);
      }
      var NR = 40, nbRingA = G.lines(8 * NR * 2, C.splat, 0.5), nbRingI = G.lines(4 * NR * 2, C.splat, 0.18);
      own(nbRingA.obj, 11); own(nbRingI.obj, 11);
      var span = G.lines(16, SPLAT_HI, 0.95), err = G.lines(4, C.err, 1); own(span.obj, 12); own(err.obj, 13);
      var spanHid = ghostLines(span, 0.3, 13);
      var G9 = { pos: new Float32Array(27), su: new Float32Array(27), sv: new Float32Array(27) };
      var G9b = { pos: new Float32Array(27), su: new Float32Array(27), sv: new Float32Array(27) };

      // the terrain the net is fitted to (scene `net`): every other fit sample as a dot, and in red how far the
      // surface is from it there (target = the neutral colour, as in chapter 05)
      var TD = FG / 2, tdUV = new Float64Array(TD), tgtPos = new Float32Array(TD * TD * 3), rg = { pos: new Float32Array(TD * TD * 3) };
      (function () {
        for (var a = 0; a < TD; a++) tdUV[a] = FIT.us[a * 2 + 1];
        for (var b = 0; b < TD; b++) for (var a2 = 0; a2 < TD; a2++) {
          var s = (b * 2 + 1) * FG + a2 * 2 + 1, o = (b * TD + a2) * 3;
          tgtPos[o] = FIT.pos[s * 3]; tgtPos[o + 1] = FIT.tg[s]; tgtPos[o + 2] = FIT.pos[s * 3 + 2];
        }
      })();
      var tgtGeo = new THREE.BufferGeometry(); tgtGeo.setAttribute('position', new THREE.BufferAttribute(tgtPos, 3));
      var tgtDots = own(new THREE.Points(tgtGeo, new THREE.PointsMaterial({ color: C.text, size: 0.055, map: dotTex, alphaTest: 0.5, transparent: true, opacity: 0.95 })), 2);
      var tgtHid = own(new THREE.Points(tgtGeo, new THREE.PointsMaterial({ color: C.text, size: 0.055, map: dotTex, alphaTest: 0.5, transparent: true, opacity: 0.3,
        depthFunc: THREE.GreaterDepth, depthWrite: false })), 3);
      var resid = G.lines(TD * TD * 2, C.err, 0.85); own(resid.obj, 2);
      var residHid = ghostLines(resid, 0.25, 3);

      /* ---------- the overlay: drawn after the frame, over the smooth Gaussians ----------
         The depth is cleared and the surface's own depth written again (ovDepth), so what lies ON the surface — the
         Gaussians' 1-sigma outlines, the red of what they leave uncovered, a camera's footprint — shows over the
         rendered Gaussians and is hidden only by the surface itself. */
      var ovl = new THREE.Scene(); ovl.fog = scene.fog;
      function ovOwn(o, order) { o.frustumCulled = false; o.renderOrder = order; ovl.add(o); return o; }
      var ovDepth = ovOwn(new THREE.Mesh(sheet.geo, new THREE.MeshBasicMaterial({ colorWrite: false, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2 })), -10);
      var maskTex = null, maskF = 0, maskData = null;
      var ovMat = new THREE.ShaderMaterial({
        transparent: true, depthWrite: false, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
        uniforms: { tMask: { value: null }, uBlend: { value: 0 }, uGap: { value: 0 }, uTint: { value: 0 }, uOut: { value: 0 }, uFog: { value: new THREE.Vector2(4, 14) },
                    uRed: { value: new THREE.Vector3().fromArray(G.rgb(C.err)) }, uWhite: { value: new THREE.Vector3().fromArray(G.rgb(C.text)) },
                    uBg: { value: new THREE.Vector3().fromArray(G.rgb(C.bg)) } },
        vertexShader: 'varying vec2 vUv; varying float vZ; void main(){ vUv = uv; vec4 e = modelViewMatrix * vec4(position, 1.0); vZ = -e.z; gl_Position = projectionMatrix * e; }',
        fragmentShader: [
          'uniform sampler2D tMask; uniform float uBlend; uniform float uGap; uniform float uTint; uniform float uOut; uniform vec2 uFog; uniform vec3 uRed; uniform vec3 uWhite; uniform vec3 uBg;',
          'varying vec2 vUv; varying float vZ;',
          'void main(){',
          '  vec4 m = texture2D(tMask, vUv);',
          '  float g = 0.85 * uGap * smoothstep(0.3, 0.7, mix(m.g, m.r, uBlend));',    // uncovered: own cell (g) or one size (r)
          '  float t = 0.17 * uTint * m.b;',                                          // inside a camera's footprint
          '  float o = uOut * m.g;',                                                   // outside a view window: the bare surface, veiled
          '  if (max(max(g, t), o) < 0.01) discard;',
          '  float f = clamp((vZ - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0);',
          '  gl_FragColor = o > max(g, t) ? vec4(uBg, o * (1.0 - 0.3 * f)) : vec4(g >= t ? uRed : uWhite, max(g, t) * (1.0 - 0.55 * f));',
          '}'].join('\n')
      });
      var ovMesh = ovOwn(new THREE.Mesh(sheet.geo, ovMat), 0);
      function maskSize(F) {
        if (F === maskF) return;
        if (maskTex) maskTex.dispose();
        maskF = F; maskData = new Uint8Array(F * F * 4);
        maskTex = new THREE.DataTexture(maskData, F, F, THREE.RGBAFormat); maskTex.magFilter = maskTex.minFilter = THREE.LinearFilter;
        maskTex.needsUpdate = true; ovMat.uniforms.tMask.value = maskTex;
      }
      var EM = 36;                                // segments per outline
      var outl = G.lines(OUTMAX * OUTMAX * EM * 2, C.splat, 0.95); ovOwn(outl.obj, 2);

      // stitch: the triangles (flat-shaded, the surface's colour), their edges, the newest pair, and the window
      var SQ = (OUTMAX - 1) * (OUTMAX - 1);
      var triGeo = new THREE.BufferGeometry();
      triGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(SQ * 18), 3));
      triGeo.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(SQ * 18), 3));
      triGeo.setDrawRange(0, 0);
      // (on paper the triangles take a light shade of the surface colour, so that its edges read on them)
      var triCol = LIGHT ? new THREE.Color(mix(C.surf, C.bg, 0.55)) : new THREE.Color(0x2f8fbf);
      var tri = own(new THREE.Mesh(triGeo, new THREE.MeshLambertMaterial({ color: triCol, emissive: LIGHT ? triCol.clone().multiplyScalar(0.12) : 0x0c3448, side: THREE.DoubleSide,
        polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })), 1);
      var triEdge = G.lines(SQ * 10, C.surf, 0.9); own(triEdge.obj, 2);
      var triNew = G.lines(10, INK, 1); own(triNew.obj, 3);
      var winGeo = new THREE.BufferGeometry(); winGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(12), 3));
      var win = ovOwn(new THREE.LineLoop(winGeo, new THREE.LineBasicMaterial({ color: INK, depthTest: false, transparent: true, fog: false })), 6);
      var winDot = ovOwn(new THREE.Points(winGeo, new THREE.PointsMaterial({ color: INK, size: 6, sizeAttenuation: false, depthTest: false, transparent: true, fog: false })), 6);

      // adaptive: a second camera (body + frustum, in the neutral colour) and the outline of what it sees on the surface
      var obs = new THREE.PerspectiveCamera(21, 1.45, 0.05, 30);
      var obsBody = G.lines(48, C.text, 0.95); ovOwn(obsBody.obj, -5);
      var obsRay = G.lines(16, C.text, 0.55); ovOwn(obsRay.obj, -5);
      var fpLine = G.lines(4096, C.text, 0.95); ovOwn(fpLine.obj, 3);
      // view window: its polygon on the surface, dashed, in the Gaussians' colour (they live inside it)
      var wnLine = G.lines(4096, C.splat, 1); ovOwn(wnLine.obj, 4);

      // floating labels
      function tag(html, tone, color) {
        var e = ui.el('div', 'kt-tag kt-tag-' + tone, html); if (color) e.style.color = color;
        root.appendChild(e); return e;
      }
      var tMu = tag('μ', 'text'), tU = tag('', 'vec', COL_U), tV = tag('', 'vec', COL_V), tN = tag('n', 'vec', INK);
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);

      // (u,v) chart, the tangent plane seen face-on, and the 4 / 8 switch above them
      var uvBox = ui.el('div', 'kt-uv'); root.appendChild(uvBox);
      var uvLabHtml = T('(u, v) chart'), uvLab = ui.el('span', 'kt-uv-lab', uvLabHtml); uvBox.appendChild(uvLab);
      var uvc = ui.canvas(uvBox, 'kt-uv-cv');
      var fcBox = ui.el('div', 'kt-uv kt-face'); root.appendChild(fcBox);
      fcBox.appendChild(ui.el('span', 'kt-uv-lab', T('tangent plane<span class="kt-wide"> · face-on</span>')));
      var fc = ui.canvas(fcBox, 'kt-uv-cv'); fc.cv.style.cursor = 'default';
      var nbBar = ui.el('div', 'kt-nbsw'); root.appendChild(nbBar);

      /* ---------- controls ---------- */
      var st = { scene: 'net', sc: SCENES.net, show: {}, n: 14, gain: 0.75, u: 0.5, v: 0.5, t: 0, nb: 4, ci: 0, cj: 0,
                 touchedProbe: false, touchedN: false, wigI: -1,
                 nudge: true, nud: { th: 0, ph: 0 }, nudT: 0, autoR: null, userZoom: false, sparkT: 0, ins: 160, lift: 0, viewKey: '' };
      var panel = ui.panel(root, { pos: 'tr', title: T('sampling') });
      var slN = ui.slider(panel, { label: T('samples'), min: 4, max: NMAX, step: 1, value: st.n,
        fmt: function (v) { return v + '²'; },
        onInput: function (v) { st.n = v; st.touchedN = true; fieldDirty = true; uvDirty = true; latDirty = probeDirty = true; } });
      var slG = ui.slider(panel, { label: T('σ / cell'), min: 0.3, max: 1.1, step: 0.01, value: st.gain,
        fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { st.gain = v; fieldDirty = true; probeDirty = true; } });
      var rdG = ui.readout(panel, { label: T('Gaussians'), tone: 'splat' });
      var rdP = ui.readout(panel, { label: T('learnable params'), tone: 'ctrl' });
      var rdZ = ui.readout(panel, { label: T('params / Gaussian'), tone: 'ok' });
      rdP.set((NU * NV * PER_CP).toLocaleString('en-US')); rdZ.set('0');
      var rdAng = ui.readout(panel, { label: LTRH('∠ ( S<sub>u</sub> , S<sub>v</sub> )'), tone: 'surf' });
      var rdSn = ui.readout(panel, { label: T('σ from {n} neighbours', { n: st.nb }), tone: 'splat' });
      var rdSa = ui.readout(panel, { label: T('σ from derivatives') });
      var rdGap = ui.readout(panel, { label: T('they differ by'), tone: 'ok' });
      var rdSh = ui.readout(panel, { label: T('shear the two lengths drop'), tone: 'err' });
      var spBox = ui.el('div', 'kt-pl-spark'); panel.appendChild(spBox);
      var spc = ui.canvas(spBox, 'kt-pl-spark-cv');
      // scene net: the optimiser
      var rdFit = ui.readout(panel, { label: T('fit error · RMS'), tone: 'err' });
      var rdMax = ui.readout(panel, { label: T('largest miss') });
      var rdIt = ui.readout(panel, { label: T('optimiser') });
      // scenes scale and fill: one size for all, or each Gaussian sized to its own cell
      var segSz = ui.seg(panel, { label: T('size'), value: 'own', options: [{ v: 'one', label: T('one for all') }, { v: 'own', label: T('own cell') }],
        onChange: function (v) { tl.auto = false; tl.goal = v === 'one' ? 1 : 0; } });
      segSz.el.classList.add('kt-pl-sz');
      var rdUnc = ui.readout(panel, { label: T('uncovered <i>one · own</i>'), tone: 'err' });
      var rdPile = ui.readout(panel, { label: T('piled 3+ deep <i>one · own</i>') });
      [segSz, rdUnc, rdPile].forEach(function (r) { panel.insertBefore(r.el, rdP.el); });
      // scene stitch
      var rdTri = ui.readout(panel, { label: T('triangles'), tone: 'surf' });
      panel.insertBefore(rdTri.el, rdP.el);
      // scene knob: its own slider, Gaussians on a log scale, and what it costs, measured
      var slK = ui.slider(panel, { label: T('Gaussians'), min: 0, max: 1000, step: 1, value: 0,
        fmt: function (v) { var n = knN(v); return (n * n).toLocaleString('en-US'); },
        onInput: function (v) { st.touchedN = true; knSet(knN(v)); } });
      panel.insertBefore(slK.el, slN.el);
      var rdFt = ui.readout(panel, { label: T('frame time · measured') });
      var rdEv = ui.readout(panel, { label: T('re-evaluate · measured') });
      var rdLim = ui.readout(panel, { label: T('turns back at') });
      var knBox = ui.el('div', 'kt-pl-spark'); panel.appendChild(knBox);
      var knc = ui.canvas(knBox, 'kt-pl-spark-cv');
      // scene adaptive
      var rdIn = ui.readout(panel, { label: T('in the camera’s view'), tone: 'splat' });
      var rdFp = ui.readout(panel, { label: T('the view covers') });
      var rdSpc = ui.readout(panel, { label: T('cell side <i>in · out</i>') });
      var rdUni = ui.readout(panel, { label: T('even lattice, in view') });
      var rdEq = ui.readout(panel, { label: T('that spacing everywhere') });
      [rdIn, rdFp, rdSpc, rdUni, rdEq].forEach(function (r) { panel.insertBefore(r.el, rdP.el); });
      // scene window: the margin around the view, and what the window costs
      var slB = ui.slider(panel, { label: T('buffer'), min: 0, max: 0.15, step: 0.005, value: 0.04,
        fmt: function (v) { return (v * 100).toFixed(1) + ' %'; }, onInput: function (v) { wd.buf = v; } });
      var rdWin = ui.readout(panel, { label: T('the window covers') });
      var rdWs = ui.readout(panel, { label: T('cell side in view'), tone: 'splat' });
      var rdSpr = ui.readout(panel, { label: T('spring net, in view') });
      var rdBare = ui.readout(panel, { label: T('no Gaussians at all'), tone: 'err' });
      panel.insertBefore(slB.el, rdIn.el);
      [rdWin, rdWs, rdSpr].forEach(function (r) { panel.insertBefore(r.el, rdUni.el); });
      panel.insertBefore(rdBare.el, rdP.el);
      // the numbers that must stay readable when the panel is folded away (a phone, the deck): under the caption
      var hud = ui.el('div', 'kt-pl-hud'); root.appendChild(hud);
      if (KT.rtl) hud.classList.add('kt-text');   // a few lines of running text: on a right-to-left page they take its direction (kit.css)
      var hudDirty = true, hudHtml = '';
      var segNb = ui.seg(nbBar, { label: T('size from'), value: st.nb,
        options: [{ v: 4, label: T('{n} neighbours', { n: 4 }) }, { v: 8, label: '8' }],
        onChange: function (v) { st.nb = v; probeDirty = uvDirty = latDirty = true; st.sparkT = 0; } });

      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, theta: 0.72, phi: 1.0, radius: 7.2, target: [0, 0.05, 0],
        minRadius: 1.2, maxRadius: 20, pick: pickBall, hit: function (x, y) { return findBall(x, y, true) >= 0; },
        release: function () { if (!dragCP) return; if (dragCP.moved) { patch.P.set(dragCP.was, dragCP.i * 3); base.set(patch.P); surfDirty = fieldDirty = true; } dragCP = null; } });
      var surfDirty = true, fieldDirty = true, uvDirty = true, latDirty = true, probeDirty = true, faceDirty = true;

      /* ---------- drag a control point ---------- */
      var ray = new THREE.Raycaster(), ndc = new THREE.Vector2(), dragPlane = new THREE.Plane(), hitP = new THREE.Vector3(), dragCP = null;
      function setNdc(e) {
        var r = R.canvas.getBoundingClientRect();
        ndc.set((e.clientX - r.left) / r.width * 2 - 1, -((e.clientY - r.top) / r.height * 2 - 1));
        ray.setFromCamera(ndc, cam);
      }
      /* the control point under client (x, y): the exact sphere for a mouse, a fingertip-sized
         screen radius for touch */
      function findBall(x, y, touch) {
        if (!st.show.net) return -1;
        if (touch) return G.nearest(cam, R.canvas, patch.P, NU * NV, x, y, KT.HIT);
        setNdc({ clientX: x, clientY: y });
        var h = ray.intersectObjects(balls, false)[0];
        return h ? h.object.userData.i : -1;
      }
      function pickBall(e) {
        var i = findBall(e.clientX, e.clientY, e.pointerType === 'touch');
        if (i < 0) return false;
        dragCP = { i: i, id: e.pointerId, was: patch.P.slice(i * 3, i * 3 + 3) };
        dragPlane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new THREE.Vector3()), balls[i].position);
        R.canvas.setPointerCapture(e.pointerId);
        return true;
      }
      R.canvas.addEventListener('pointermove', function (e) {
        if (!dragCP) { if (st.show.net && e.pointerType === 'mouse') { setNdc(e); R.canvas.style.cursor = ray.intersectObjects(balls, false)[0] ? 'grab' : ''; } return; }
        if (e.pointerId !== dragCP.id) return;
        setNdc(e);
        if (ray.ray.intersectPlane(dragPlane, hitP)) {
          var o = dragCP.i * 3;
          if (!dragCP.moved) {             // a real drag begins: the idle wiggle stops, and its point goes home
            dragCP.moved = true;
            if (st.wigI >= 0 && st.wigI !== dragCP.i) patch.P.set(base.subarray(st.wigI * 3, st.wigI * 3 + 3), st.wigI * 3);
            st.wigI = -1; st.sc = Object.assign({}, st.sc, { wig: 0 });
            if (st.fit && !st.fit.done) {                 // the optimiser stops; the other points take their fitted heights
              for (var fi = 0; fi < FIT.n; fi++) if (fi !== dragCP.i) patch.P[fi * 3 + 1] = fitBase[fi * 3 + 1];
              st.fit.done = true;
            }
            if (st.fit) st.fit.user = true;
          }
          patch.P[o] = KT.clamp(hitP.x, -3, 3); patch.P[o + 1] = KT.clamp(hitP.y, -1.1, 2.4); patch.P[o + 2] = KT.clamp(hitP.z, -3, 3);
          base.set(patch.P); surfDirty = fieldDirty = true;
        }
      });
      function endCP(e) { if (dragCP && e.pointerId === dragCP.id) dragCP = null; }
      R.canvas.addEventListener('pointerup', endCP); R.canvas.addEventListener('pointercancel', endCP);

      /* ---------- (u,v) chart interaction ---------- */
      var uvDrag = false;
      function uvSet(e) {
        var r = uvc.cv.getBoundingClientRect();
        st.u = KT.clamp((e.clientX - r.left) / r.width, 0.02, 0.98);
        st.v = KT.clamp(1 - (e.clientY - r.top) / r.height, 0.02, 0.98);
        st.touchedProbe = true; uvDirty = probeDirty = true;
      }
      uvc.cv.style.touchAction = 'none';
      uvc.cv.addEventListener('pointerdown', function (e) { uvDrag = true; uvc.cv.setPointerCapture(e.pointerId); uvSet(e); });
      uvc.cv.addEventListener('pointermove', function (e) { if (uvDrag) uvSet(e); });
      uvc.cv.addEventListener('pointerup', function () { uvDrag = false; });
      uvc.cv.addEventListener('pointercancel', function () { uvDrag = false; });

      /* ---------- geometry updates ---------- */
      function lin(n, lo, hi) {
        var o = new Float32Array(n), i; if (lo == null) { lo = 0; hi = 1; }
        for (i = 0; i < n; i++) o[i] = lo + (hi - lo) * i / (n - 1);
        return o;
      }
      /* append the segment a -> b to a line array at float offset k; returns the next offset */
      function seg(A, k, p, q) { A[k] = p[0]; A[k + 1] = p[1]; A[k + 2] = p[2]; A[k + 3] = q[0]; A[k + 4] = q[1]; A[k + 5] = q[2]; return k + 6; }
      function segAt(A, k, P, o, o2) { A[k] = P[o]; A[k + 1] = P[o + 1]; A[k + 2] = P[o + 2]; A[k + 3] = P[o2]; A[k + 4] = P[o2 + 1]; A[k + 5] = P[o2 + 2]; return k + 6; }

      var wU = lin(WS + 1), wL = lin(WL), fL = lin(FN);
      var gA = { pos: new Float32Array((WS + 1) * WL * 3) }, gB = { pos: new Float32Array((WS + 1) * WL * 3) };
      function updateSurface() {
        var i, j, o, n = 0, A = wire.arr, P = patch.P;
        NB.patchGrid(patch, wU, wL, gA);            // lines along u (one per v)
        for (j = 0; j < WL; j++) for (i = 0; i < WS; i++) { o = (j * (WS + 1) + i) * 3; n = segAt(A, n, gA.pos, o, o + 3); }
        NB.patchGrid(patch, wL, wU, gB);            // lines along v (one per u)
        for (i = 0; i < WL; i++) for (j = 0; j < WS; j++) n = segAt(A, n, gB.pos, (j * WL + i) * 3, ((j + 1) * WL + i) * 3);
        wire.set(n / 3);
        NB.patchGrid(patch, fL, fL, sg); sheet.set(sg, -1);     // S_u x S_v points down on this patch: the lit side is the other one
        n = 0; A = net.arr;
        for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
          o = (j * NU + i) * 3; balls[j * NU + i].position.set(P[o], P[o + 1], P[o + 2]);
          if (i < NU - 1) n = segAt(A, n, P, o, o + 3);
          if (j < NV - 1) n = segAt(A, n, P, o, o + NU * 3);
        }
        net.set(n / 3);
      }

      /* the sampling lattice itself: every cell border and every sample centre (when there are few enough to read) */
      function updateLattice() {
        var n = st.n, i, j, k = 0, A = lat.arr, M = LS;
        if (n > LATMAX) { lat.set(0); dotGeo.setDrawRange(0, 0); return; }
        var bd = lin(n + 1), f = lin(M + 1), us = new Float32Array(n);
        NB.patchGrid(patch, f, bd, lg);
        for (j = 0; j <= n; j++) for (i = 0; i < M; i++) k = segAt(A, k, lg.pos, (j * (M + 1) + i) * 3, (j * (M + 1) + i + 1) * 3);
        NB.patchGrid(patch, bd, f, lg);
        for (i = 0; i <= n; i++) for (j = 0; j < M; j++) k = segAt(A, k, lg.pos, (j * (n + 1) + i) * 3, ((j + 1) * (n + 1) + i) * 3);
        lat.set(k / 3);
        for (i = 0; i < n; i++) us[i] = (i + 0.5) / n;
        NB.patchGrid(patch, us, us, { pos: dotPos });
        dotGeo.setDrawRange(0, n * n); dotGeo.attributes.position.needsUpdate = true;
      }

      var fg = { pos: new Float32Array(NMAX * NMAX * 3), su: new Float32Array(NMAX * NMAX * 3),
                 sv: new Float32Array(NMAX * NMAX * 3), col: new Float32Array(NMAX * NMAX * 3) };
      var memo = {};
      function centres(n) { var k = 'c' + n, a, i; if (!memo[k]) { a = memo[k] = new Float32Array(n); for (i = 0; i < n; i++) a[i] = (i + 0.5) / n; } return memo[k]; }
      function evenD(n) { var k = 'd' + n; return memo[k] || (memo[k] = new Float32Array(n).fill(1 / n)); }
      /* the n x n lattice of planoids sized to their cells (gl.writePlanoids), evaluated in strips of rows that fit fg */
      function writeLattice(fd, n, k, rgb) {
        var us = centres(n), d = evenD(n), rows = Math.max(1, Math.floor(NMAX * NMAX / n)), b0, nb, cnt = 0;
        for (b0 = 0; b0 < n; b0 += rows) {
          nb = Math.min(rows, n - b0);
          NB.patchGrid(patch, us, us.subarray(b0, b0 + nb), fg);
          cnt = G.writePlanoids(fd, cnt, fg, n, nb, d, d, k, rgb);
        }
        fd.commit(cnt);
        return cnt;
      }
      function updateField() { writeLattice(field, st.n, st.gain, st.show.feat ? null : violet); st.fieldN = st.n; }

      /* ---------- footprints ----------
         A Gaussian lies in its tangent plane: centre S(u_i, v_i), axes along S_u^ and S_v^perp^, half-lengths LA, LB
         (1 sigma). Carried into the chart through the Jacobian [S_u S_v] at its centre, its 1-sigma ellipse is
             { (u_i + a, v_i + b) :  ( (|S_u| a + (S_v . S_u^) b) / LA )^2 + ( |S_v^perp| b / LB )^2 <= 1 } ,
         kept as (a1 a + b1 b)^2 + (b2 b)^2 <= 1. That set's boundary, laid on the surface, is the outline that is
         drawn; the set is what the coverage counts. It is first order (exact for a flat patch): the flat Gaussian's
         own ellipse floats a little above or below the curved surface, the outline is drawn on the surface.
         Two sizings of the same samples:
           own cell   LA = |S_u| du k,  LB = |S_v^perp| dv k        the derivative rule, as gl.writePlanoids
           one size   LA = LB = k sqrt(mean cell area)              cell area = |S_u| |S_v^perp| du dv at each sample:
                                                                    the same total area, one size for all
         `blend` goes from the first (0) to the second (1).
         Coverage (measured once per change of samples, sigma / cell or surface, never per frame): an F x F grid of
         (u,v) points ((i + 0.5) / F, (j + 0.5) / F), each weighted by its own surface area |S_u x S_v| / F^2 (the
         weights depend on the surface alone and are kept while its control points stay put). F = COVF = 1283 for
         every sample count: a prime above 8 x NMAX, so a lattice of up to NMAX samples a side gets at least 8 test
         points per cell along each axis, and as n and F share no factor the points fall at offsets that drift
         across the cells instead of repeating in each one. (A grid of 4 n points a side, F = 512 at n = 128, sits
         at the same four offsets in every cell and over-counts what lies there: it read 22 % three deep, own cell,
         at 128^2 against 5-7 % at the counts around it.)
           uncovered = area of the points inside no footprint / area of the patch
           piled     = area of the points inside three or more / area of the patch
         The red masks, on the surface and in the chart, are the same counts: per texel the share of its 3 x 3 points
         left uncovered (COVD texels a side).                                                                         */
      function footSet(cap) {
        function f(k) { return new Float32Array(cap * (k || 1)); }
        return { n: 0, u: f(), v: f(), du: f(), dv: f(), lu: f(), pr: f(), lvp: f(), cx: f(3), e1: f(3), e2: f(3) };
      }
      function fillSet(S, gd, us, vs, na, nb, du, dv) {
        var P = gd.pos, SU = gd.su, SV = gd.sv, a, b, i = 0, t, ux, uy, uz, vx, vy, vz, lu, pr, px, py, pz, lv;
        for (b = 0; b < nb; b++) for (a = 0; a < na; a++, i++) {
          t = i * 3; ux = SU[t]; uy = SU[t + 1]; uz = SU[t + 2]; vx = SV[t]; vy = SV[t + 1]; vz = SV[t + 2];
          lu = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1e-9; pr = (ux * vx + uy * vy + uz * vz) / lu;
          px = vx - pr * ux / lu; py = vy - pr * uy / lu; pz = vz - pr * uz / lu; lv = Math.sqrt(px * px + py * py + pz * pz) || 1e-9;
          S.u[i] = us[a]; S.v[i] = vs[b]; S.du[i] = du[a]; S.dv[i] = dv[b]; S.lu[i] = lu; S.pr[i] = pr; S.lvp[i] = lv;
          S.cx[t] = P[t]; S.cx[t + 1] = P[t + 1]; S.cx[t + 2] = P[t + 2];
          S.e1[t] = ux / lu; S.e1[t + 1] = uy / lu; S.e1[t + 2] = uz / lu; S.e2[t] = px / lv; S.e2[t + 1] = py / lv; S.e2[t + 2] = pz / lv;
        }
        S.n = i;
      }
      function sizeA(S, i, k, s1, bl) { return KT.lerp(S.lu[i] * S.du[i] * k, s1, bl); }
      function sizeB(S, i, k, s1, bl) { return KT.lerp(S.lvp[i] * S.dv[i] * k, s1, bl); }
      /* the set as planoids; rev writes them in reverse order (setCount then retires the first ones) */
      function writeSet(fd, S, k, s1, bl, rgb, rev) {
        var i, j, o, t, LA, LB;
        for (i = 0; i < S.n; i++) {
          j = rev ? S.n - 1 - i : i; o = j * 3; t = i * 3; LA = sizeA(S, i, k, s1, bl); LB = sizeB(S, i, k, s1, bl);
          fd.center[o] = S.cx[t]; fd.center[o + 1] = S.cx[t + 1]; fd.center[o + 2] = S.cx[t + 2];
          fd.axU[o] = S.e1[t] * LA; fd.axU[o + 1] = S.e1[t + 1] * LA; fd.axU[o + 2] = S.e1[t + 2] * LA;
          fd.axV[o] = S.e2[t] * LB; fd.axV[o + 1] = S.e2[t + 1] * LB; fd.axV[o + 2] = S.e2[t + 2] * LB;
          fd.color[o] = rgb[0]; fd.color[o + 1] = rgb[1]; fd.color[o + 2] = rgb[2];
        }
        fd.commit(S.n);
      }

      // the surface on a fine vertex grid, for laying lines on it (bilinear, then lifted along the upward normal)
      var VF = 128, surfVer = 0, LIFT = 0.006;
      var vg = { pos: new Float32Array((VF + 1) * (VF + 1) * 3), su: new Float32Array((VF + 1) * (VF + 1) * 3),
                 sv: new Float32Array((VF + 1) * (VF + 1) * 3), nrm: new Float32Array((VF + 1) * (VF + 1) * 3), ver: -1 };
      function vgUpdate() {
        if (vg.ver === surfVer) return;
        var f = lin(VF + 1), i, t, x, y, z, l;
        NB.patchGrid(patch, f, f, vg);
        for (i = 0; i < (VF + 1) * (VF + 1); i++) {
          t = i * 3;
          x = vg.su[t + 1] * vg.sv[t + 2] - vg.su[t + 2] * vg.sv[t + 1]; y = vg.su[t + 2] * vg.sv[t] - vg.su[t] * vg.sv[t + 2]; z = vg.su[t] * vg.sv[t + 1] - vg.su[t + 1] * vg.sv[t];
          l = Math.sqrt(x * x + y * y + z * z) || 1; if (y < 0) l = -l;
          vg.nrm[t] = x / l; vg.nrm[t + 1] = y / l; vg.nrm[t + 2] = z / l;
        }
        vg.ver = surfVer;
      }
      function onSurf(u, v, lift, A, o) {
        u = u < 0 ? 0 : u > 1 ? 1 : u; v = v < 0 ? 0 : v > 1 ? 1 : v;
        var x = u * VF, y = v * VF, i = Math.min(VF - 1, x | 0), j = Math.min(VF - 1, y | 0), s = x - i, t = y - j, c;
        var k00 = (j * (VF + 1) + i) * 3, k10 = k00 + 3, k01 = k00 + (VF + 1) * 3, k11 = k01 + 3, P = vg.pos, N = vg.nrm;
        for (c = 0; c < 3; c++) {
          A[o + c] = (P[k00 + c] * (1 - s) + P[k10 + c] * s) * (1 - t) + (P[k01 + c] * (1 - s) + P[k11 + c] * s) * t +
                     lift * ((N[k00 + c] * (1 - s) + N[k10 + c] * s) * (1 - t) + (N[k01 + c] * (1 - s) + N[k11 + c] * s) * t);
        }
      }
      /* the 1-sigma outlines of footprints from..S.n-1, on the surface */
      function outlineSet(S, k, s1, bl, from) {
        var A = outl.arr, o = 0, i, m, LA, LB, a1, b1, b2, th, a, b, pu = 0, pv = 0, qu, qv;
        for (i = from || 0; i < S.n; i++) {
          LA = sizeA(S, i, k, s1, bl); LB = sizeB(S, i, k, s1, bl); a1 = S.lu[i] / LA; b1 = S.pr[i] / LA; b2 = S.lvp[i] / LB;
          for (m = 0; m <= EM; m++) {
            th = m / EM * 6.283185; b = Math.sin(th) / b2; a = (Math.cos(th) - b1 * b) / a1; qu = S.u[i] + a; qv = S.v[i] + b;
            if (m) { onSurf(pu, pv, LIFT, A, o); onSurf(qu, qv, LIFT, A, o + 3); o += 6; }
            pu = qu; pv = qv;
          }
        }
        outl.set(o / 3);
      }

      // coverage: the area weights of the F x F grid (cw.net = the control points they were measured on), and the
      // count of footprints over each point
      var COVF = 1283, COVD = Math.ceil(COVF / 3);
      var cw = { F: 0, net: null, w: null, tot: 0 }, cntA = null, cntB = null;
      function coverW(F) {
        var P = patch.P, b0, nb, i, t, x, y, z, w, tot = 0;
        if (cw.F === F && cw.net) { for (i = 0; i < P.length && P[i] === cw.net[i]; i++); if (i === P.length) return; }
        var fs = centres(F), rows = Math.max(1, Math.floor(NMAX * NMAX / F));
        if (cw.F !== F || !cw.w) { cw.w = new Float32Array(F * F); cntA = new Uint8Array(F * F); cntB = new Uint8Array(F * F); }
        for (b0 = 0; b0 < F; b0 += rows) {
          nb = Math.min(rows, F - b0); NB.patchGrid(patch, fs, fs.subarray(b0, b0 + nb), fg);
          for (i = 0; i < nb * F; i++) {
            t = i * 3;
            x = fg.su[t + 1] * fg.sv[t + 2] - fg.su[t + 2] * fg.sv[t + 1]; y = fg.su[t + 2] * fg.sv[t] - fg.su[t] * fg.sv[t + 2]; z = fg.su[t] * fg.sv[t + 1] - fg.su[t + 1] * fg.sv[t];
            w = Math.sqrt(x * x + y * y + z * z); cw.w[b0 * F + i] = w; tot += w;
          }
        }
        cw.F = F; cw.tot = tot; (cw.net || (cw.net = new Float32Array(P.length))).set(P);
      }
      function coverCount(S, k, s1, bl, F, cnt) {
        var i, fu, fv, f0, f1, g0, g1, a, b, p1, p2, LA, LB, a1, b1, b2, ea, eb, uc, vc, row, w = cw.w, un = 0, pile = 0;
        cnt.fill(0);
        for (i = 0; i < S.n; i++) {
          LA = sizeA(S, i, k, s1, bl); LB = sizeB(S, i, k, s1, bl); a1 = S.lu[i] / LA; b1 = S.pr[i] / LA; b2 = S.lvp[i] / LB;
          eb = 1 / b2; ea = Math.sqrt(b1 * b1 + b2 * b2) / (a1 * b2); uc = S.u[i]; vc = S.v[i];
          f0 = Math.max(0, Math.floor((uc - ea) * F - 0.5)); f1 = Math.min(F - 1, Math.ceil((uc + ea) * F - 0.5));
          g0 = Math.max(0, Math.floor((vc - eb) * F - 0.5)); g1 = Math.min(F - 1, Math.ceil((vc + eb) * F - 0.5));
          for (fv = g0; fv <= g1; fv++) {
            b = (fv + 0.5) / F - vc; p2 = b2 * b; p2 *= p2;
            if (p2 > 1) continue;
            row = fv * F;
            for (fu = f0; fu <= f1; fu++) {
              a = (fu + 0.5) / F - uc; p1 = a1 * a + b1 * b;
              if (p1 * p1 + p2 <= 1 && cnt[row + fu] < 255) cnt[row + fu]++;
            }
          }
        }
        for (i = 0; i < F * F; i++) { if (!cnt[i]) un += w[i]; else if (cnt[i] >= 3) pile += w[i]; }
        return { unc: un / cw.tot, pile: pile / cw.tot };
      }

      /* ---------- scenes scale and fill: one size for all, or each Gaussian sized to its own cell ---------- */
      var tset = footSet(NMAX * NMAX);
      var tl = { key: '', b: 0, goal: 0, auto: true, mode: 'own', s1: 0, own: null, one: null, bDirty: true, imgOwn: null, imgOne: null };
      /* own cell for 4.5 s, a 1.2 s glide, one size for 4.5 s, a glide back: 11.4 s a round */
      function autoBlend(t) { var x = t % 11.4; return x < 4.5 ? 0 : x < 5.7 ? KT.ease((x - 4.5) / 1.2) : x < 10.2 ? 1 : 1 - KT.ease((x - 10.2) / 1.2); }
      function tileTick(dt) {
        var g = tl.auto ? autoBlend(st.t) : tl.goal, b = tl.auto ? g : tl.b + (g - tl.b) * (1 - Math.exp(-7 * dt));
        if (Math.abs(b - g) < 2e-3) b = g;
        if (b !== tl.b) { tl.b = b; tl.bDirty = true; }
        var m = tl.b < 0.5 ? 'own' : 'one';
        if (m !== tl.mode) { tl.mode = m; segSz.set(m); hudDirty = true; }
      }
      /* the masks: per texel of a D x D image (D = ceil(F / 3)), the share of its 3 x 3 test points that no footprint
         covers: one size in red, own cell in green (the overlay texture), and the same shares as red images for the chart */
      function maskBuild(F, D) {
        var X, Y, fu, fv, i, m, ua, ub, o, s, r = G.rgb(C.err).map(function (x) { return Math.round(x * 255); });
        maskSize(D);
        for (Y = 0; Y < D; Y++) for (X = 0; X < D; X++) {
          m = ua = ub = 0;
          for (fv = Y * 3; fv < Y * 3 + 3 && fv < F; fv++) for (fu = X * 3; fu < X * 3 + 3 && fu < F; fu++) {
            i = fv * F + fu; m++; if (!cntA[i]) ua++; if (!cntB[i]) ub++;
          }
          o = (Y * D + X) * 4; maskData[o] = Math.round(255 * ub / m); maskData[o + 1] = Math.round(255 * ua / m); maskData[o + 2] = 0; maskData[o + 3] = 255;
        }
        maskTex.needsUpdate = true;
        function image(cv, ch) {                         // chart row 0 is v = 1: flipped
          if (!cv || cv.width !== D) { cv = document.createElement('canvas'); cv.width = cv.height = D; }
          var g = cv.getContext('2d'), im = g.createImageData(D, D);
          for (Y = 0; Y < D; Y++) for (X = 0; X < D; X++) {
            s = (Y * D + X) * 4; o = ((D - 1 - Y) * D + X) * 4;
            im.data[o] = r[0]; im.data[o + 1] = r[1]; im.data[o + 2] = r[2]; im.data[o + 3] = Math.round(maskData[s + ch] * 200 / 255);
          }
          g.putImageData(im, 0, 0); return cv;
        }
        tl.imgOne = image(tl.imgOne, 0); tl.imgOwn = image(tl.imgOwn, 1);
      }
      function tileConfig() {
        var n = st.n, us = centres(n), d = evenD(n), i, area = 0, F = COVF;
        NB.patchGrid(patch, us, us, fg); fillSet(tset, fg, us, us, n, n, d, d);
        for (i = 0; i < tset.n; i++) area += tset.lu[i] * tset.lvp[i] * tset.du[i] * tset.dv[i];
        tl.s1 = st.gain * Math.sqrt(area / tset.n);
        coverW(F);
        tl.own = coverCount(tset, st.gain, tl.s1, 0, F, cntA); tl.one = coverCount(tset, st.gain, tl.s1, 1, F, cntB);
        maskBuild(F, COVD);
        rdUnc.el.lastChild.innerHTML = pct(tl.one.unc) + ' <i>·</i> ' + pct(tl.own.unc);
        rdPile.el.lastChild.innerHTML = pct(tl.one.pile) + ' <i>·</i> ' + pct(tl.own.pile);
        hudDirty = true;
      }
      function tileUpdate() {
        var key = st.n + '|' + st.gain + '|' + surfVer;
        if (key !== tl.key) { tl.key = key; vgUpdate(); tileConfig(); tl.bDirty = true; }
        if (!tl.bDirty) return;
        tl.bDirty = false;
        if (st.show.field) { writeSet(field, tset, st.gain, tl.s1, tl.b, violet); st.fieldN = st.n; }
        if (st.n <= OUTMAX) outlineSet(tset, st.gain, tl.s1, tl.b, 0);
        ovMat.uniforms.uBlend.value = tl.b; uvDirty = true; hudDirty = true;
        if (st.show.nbr) probeDirty = true;
      }

      /* ---------- scene stitch: a 2 x 2 window over the lattice of samples, two triangles at every stop ----------
         The samples are the Gaussians' centres on a uniform n x n lattice. Window (a, b) takes the four samples
         (a, b), (a+1, b), (a+1, b+1), (a, b+1) and stitches the triangles (a,b)(a+1,b)(a+1,b+1) and (a,b)(a+1,b+1)(a,b+1):
         the lattice says who the neighbours are, nothing is searched. Reading order in the chart — left to right,
         from the top row (v = 1, the near side in the default view) down — STITCH_T seconds for the (n-1)^2 stops
         after a STITCH_LEAD pause; the Gaussians the window has passed are retired (written in reverse order of
         passing, so setCount drops them), as the full-scene mesh of chapter 07 does. Then it holds. */
      function quadAt(n, k) { return { a: k % (n - 1), b: n - 2 - Math.floor(k / (n - 1)) }; }      // stop k: the window's lower-left sample
      var sset = footSet(OUTMAX * OUTMAX), sx = { n: 0, Q: 0, q: -1, V: 0, t0: 0, ver: -1, k: 0, passed: 0, P: null };
      function triAt(TP, TN, k, P, i0, i1, i2) {
        var ax = P[i1] - P[i0], ay = P[i1 + 1] - P[i0 + 1], az = P[i1 + 2] - P[i0 + 2], bx = P[i2] - P[i0], by = P[i2 + 1] - P[i0 + 1], bz = P[i2 + 2] - P[i0 + 2];
        var nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1, v, c;
        if (ny < 0) l = -l;
        // the lattice's winding faces down (S_u x S_v points down on this patch), so from above the camera sees back faces, lit
        // by the hemisphere's ground colour: a dark band, the dark stage's look. On paper the winding is turned over and the
        // triangles are lit from above, a light shade of the surface colour.
        (LIGHT ? [i0, i2, i1] : [i0, i1, i2]).forEach(function (i) { for (c = 0; c < 3; c++) { TP[k + c] = P[i + c]; } TN[k] = nx / l; TN[k + 1] = ny / l; TN[k + 2] = nz / l; k += 3; });
        return k;
      }
      function stitchBuild() {
        var n = Math.min(st.n, OUTMAX), us = centres(n), d = evenD(n), a, b, k = 0, e = 0, P, q;
        vgUpdate();
        // the set in the order the window passes the samples: rows from v = 1 down (vs reversed), each left to right
        var vr = memo['r' + n] || (memo['r' + n] = Float32Array.from(us).reverse());
        NB.patchGrid(patch, us, vr, fg); fillSet(sset, fg, us, vr, n, n, d, d);
        writeSet(field, sset, st.gain, 0, 0, violet, true); st.fieldN = n;
        NB.patchGrid(patch, us, us, fg);
        P = sx.P = fg.pos.slice(0, n * n * 3);                 // the lattice itself, row b = v index
        var TP = triGeo.attributes.position.array, TN = triGeo.attributes.normal.array, E = triEdge.arr;
        for (k = 0, q = 0; q < (n - 1) * (n - 1); q++) {
          a = quadAt(n, q).a; b = quadAt(n, q).b;
          var v00 = (b * n + a) * 3, v10 = v00 + 3, v01 = v00 + n * 3, v11 = v01 + 3;
          k = triAt(TP, TN, k, P, v00, v10, v11); k = triAt(TP, TN, k, P, v00, v11, v01);
          e = segAt(E, e, P, v00, v10); e = segAt(E, e, P, v10, v11); e = segAt(E, e, P, v11, v01); e = segAt(E, e, P, v01, v00); e = segAt(E, e, P, v00, v11);
        }
        triGeo.attributes.position.needsUpdate = triGeo.attributes.normal.needsUpdate = true;
        sx.n = n; sx.Q = (n - 1) * (n - 1); sx.V = n * n; sx.q = -1; sx.t0 = st.t; sx.ver = surfVer; sx.k = st.gain;
        if (st.n !== n) { st.n = n; slN.set(n); layers(); }
      }
      function stitchTick() {
        if (sx.n !== st.n || sx.ver !== surfVer || sx.k !== st.gain) stitchBuild();
        var p = (st.t - sx.t0 - STITCH_LEAD) / STITCH_T * sx.Q, q = p < 0 ? 0 : Math.min(sx.Q, Math.floor(p) + 1);
        if (q === sx.q) return;
        sx.q = q;
        var n = sx.n, k = Math.max(0, q - 1), qa = quadAt(n, k), a = qa.a, b = qa.b, done = q >= sx.Q, P = sx.P, W = winGeo.attributes.position.array, i, o;
        triGeo.setDrawRange(0, q * 6); triEdge.set(q * 10);
        sx.passed = done ? sx.V : q ? (n - 2 - b) * n + a : 0;      // in passing order: the rows above the window's top row, and its left
        field.setCount(sx.V - sx.passed);
        [(b * n + a), (b * n + a + 1), ((b + 1) * n + a + 1), ((b + 1) * n + a)].forEach(function (v, j) {
          for (o = 0; o < 3; o++) W[j * 3 + o] = P[v * 3 + o] + (o === 1 ? 0.02 : 0);
        });
        winGeo.attributes.position.needsUpdate = true;
        win.visible = winDot.visible = !done;
        if (q) { for (i = 0; i < 30; i++) triNew.arr[i] = triEdge.arr[k * 30 + i]; triNew.set(10); } else triNew.set(0);
        triNew.obj.visible = !done;
        outlineSet(sset, st.gain, 0, 0, sx.passed);
        rdTri.set((q * 2).toLocaleString('en-US') + ' / ' + (sx.Q * 2).toLocaleString('en-US'));
        rdG.set((sx.V - sx.passed).toLocaleString('en-US') + ' / ' + sx.V.toLocaleString('en-US'));     // still drawn, of the lattice
        uvDirty = true; hudDirty = true;
      }

      /* ---------- scene knob: x4 Gaussians a step, stored numbers fixed, frame time measured ----------
         Each step holds at least 1.4 s and 6 frames. Its frame time is the median interval between frames, from the
         third frame after the change on (the first two carry the re-evaluation and the upload). The sweep turns
         back at a step whose frame time is above both KN_BUDGET and 3x that of the first step (what a frame costs
         anyway on this machine: on a slow one the fixed cost alone can pass 50 ms), or at the last step
         (1024^2 = 1,048,576 Gaussians): that step is held a little longer, then it starts again from the first.
         The holds run on the clock (performance.now), not on the kit's dt: that stops at 50 ms a frame, so slow
         frames would stretch them (a 1.4 s hold took 11 s at 400 ms a frame).
         The ceiling is the field's buffer: 60 bytes a Gaussian (five 3-float attributes), 63 MB at the top,
         allocated only while this scene is on stage: freed when the scene is left, and when the stage leaves the
         screen (see `onStage`); knTick makes it again when the knob is back on screen. */
      var small = field, big = null;
      var kn = { li: 0, t: 0, k: 0, iv: [], last: 0, ft: {}, ev: {}, lim: null, cur: 0, top: false, plotT: 0, redo: false };
      function knBig(on) {
        if (on && !big) {
          var top = KN_LEVELS[KN_LEVELS.length - 1];
          big = G.splats(top * top); big.uniforms.uRim.value = small.uniforms.uRim.value; scene.add(big.mesh);
        }
        if (on) { small.mesh.visible = false; big.mesh.visible = true; field = big; return; }
        if (big) {
          scene.remove(big.mesh); big.mesh.geometry.dispose();
          var u = big.mesh.userData.ktSplat; [u.plain, u.depth, u.core, u.accum].forEach(function (m) { m.dispose(); });
          big = null; small.mesh.visible = !!st.show.field;       // knBig(true) had hidden it
        }
        field = small;
      }
      function knN(v) { return Math.round(Math.exp(KT.lerp(Math.log(4), Math.log(1024), v / 1000))); }
      function knV(n) { return Math.round((Math.log(n) - Math.log(4)) / (Math.log(1024) - Math.log(4)) * 1000); }
      function knSet(n) {                    // re-evaluate the patch at n x n: the only work turning the knob takes
        knBig(n * n > small.capacity || !!big);
        var t0 = performance.now();
        st.n = n; writeLattice(field, n, st.gain, violet); st.fieldN = n; field.setCount(n * n);
        kn.ev[n] = performance.now() - t0; kn.cur = n; kn.k = 0; kn.t = 0; kn.iv = [];
        slK.set(knV(n)); layers(); rdEv.set(kn.ev[n].toFixed(kn.ev[n] < 10 ? 1 : 0) + ' ms'); hudDirty = true; kn.plotT = 0;
      }
      function knCap() { var f0 = kn.ft[KN_LEVELS[0]]; return Math.max(KN_BUDGET, f0 != null ? 3 * f0 : 0); }
      function median(a) { var s = a.slice().sort(function (x, y) { return x - y; }), m = s.length >> 1; return s.length & 1 ? s[m] : 0.5 * (s[m - 1] + s[m]); }
      function knTick(dt) {
        if (kn.redo) { kn.redo = false; knSet(kn.cur); return; }     // back on screen after `onStage` freed the field
        var now = performance.now(), iv = kn.last ? now - kn.last : 0, ft = null;
        // the step's clock: the time since the last frame, as measured (a frame of more than a second counts as one)
        kn.last = now; kn.t += iv > 0 ? Math.min(iv, 1000) / 1000 : dt; kn.k++;
        if (kn.k > 2 && iv > 0 && iv < 4000) { kn.iv.push(iv); if (kn.iv.length > 120) kn.iv.shift(); }
        if (kn.iv.length >= 3) { ft = median(kn.iv); kn.ft[kn.cur] = ft; }
        if (!st.touchedN && kn.iv.length >= 6) {
          if (!kn.top && kn.t >= 1.4) {
            var cap = knCap();
            if (ft > cap || kn.li >= KN_LEVELS.length - 1) {
              kn.top = true; kn.lim = { n: kn.cur, ft: ft, cap: cap, why: ft > cap ? 'frame' : 'buffer' };
            } else { kn.li++; knSet(KN_LEVELS[kn.li]); return; }
          } else if (kn.top && kn.t >= 2.8) { kn.top = false; kn.li = 0; knSet(KN_LEVELS[0]); return; }
        }
        kn.plotT -= dt;
        if (kn.plotT <= 0) {
          kn.plotT = 0.25;
          rdFt.set(ft == null ? T('measuring…') : ft.toFixed(1) + ' ms · ' + fmtFps(ft));
          rdLim.set(!kn.lim ? '—' : kn.lim.why === 'frame' ? T('{n} · frame > {ms} ms', { n: (kn.lim.n * kn.lim.n).toLocaleString('en-US'), ms: Math.round(kn.lim.cap) })
                    : T('{n} · buffer full', { n: (kn.lim.n * kn.lim.n).toLocaleString('en-US') }));
          drawKnob(ft); hudDirty = true;
        }
      }
      function drawKnob(ft) {
        var z = knc.fit(), g = knc.ctx, w = z.w, h = z.h, lx0 = Math.log(16), lx1 = Math.log(1048576), ym = knCap() * 1.25, pts = [];
        g.clearRect(0, 0, w, h);
        KN_LEVELS.forEach(function (n) { if (kn.ft[n] != null) { pts.push([n * n, kn.ft[n]]); ym = Math.max(ym, kn.ft[n] * 1.12); } });
        if (kn.ft[kn.cur] != null && KN_LEVELS.indexOf(kn.cur) < 0) ym = Math.max(ym, kn.ft[kn.cur] * 1.12);
        function X(c) { return 8 + (Math.log(c) - lx0) / (lx1 - lx0) * (w - 16); }
        function Y(ms) { return h - 13 - Math.min(ms, ym) / ym * (h - 28); }
        g.font = '500 9px JetBrains Mono, ui-monospace, monospace'; g.textBaseline = 'middle';
        [[1000 / 60, '60 fps', rgba(C.dim, 0.35)], [knCap(), T('turns back'), rgba(C.err, 0.4)]].forEach(function (l) {
          g.strokeStyle = l[2]; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(8, Y(l[0])); g.lineTo(w - 8, Y(l[0])); g.stroke(); g.setLineDash([]);
          g.fillStyle = C.mute; g.textAlign = 'right'; g.fillText(l[1], w - 8, Y(l[0]) - 6);
        });
        g.fillStyle = C.mute; g.textAlign = 'center';
        [[64, '64'], [1024, '1k'], [16384, '16k'], [262144, '262k']].forEach(function (l) { g.fillText(l[1], X(l[0]), h - 5); });
        g.textAlign = 'right'; g.fillText('1M', w - 6, h - 5);
        if (pts.length > 1) {
          g.strokeStyle = rgba(C.text, 0.75); g.lineWidth = 1.5; g.beginPath();
          pts.forEach(function (p, i) { if (i) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1])); }); g.stroke();
        }
        g.fillStyle = C.text; pts.forEach(function (p) { g.beginPath(); g.arc(X(p[0]), Y(p[1]), 2.4, 0, 6.2832); g.fill(); });
        if (ft != null) { g.fillStyle = C.splat; g.beginPath(); g.arc(X(kn.cur * kn.cur), Y(ft), 4, 0, 6.2832); g.fill(); }
        g.fillStyle = C.dim; g.textAlign = 'left'; g.fillText(T('frame time vs Gaussians · measured'), 8, 8);
      }

      /* ---------- scene adaptive: a fixed budget, warped towards what a camera sees ----------
         The camera `obs` flies a closed path (AD_T s a round) above the patch, looking across it. On a QA x QA grid
         of (u,v) cell centres a point is in its view when it projects inside the image; there its importance is the
         image area its piece of surface covers, |S_u x S_v| cos(angle to the view ray) / distance^2 (foreshortening
         and distance), elsewhere none — as in the manifold module's allocation, here for one tile.
         The n x n cells of the lattice are warped towards it as a net of springs: the (n+1)^2 cell corners, those
         on the border sliding along it, each pulled to the stiffness-weighted mean of its four neighbours, a
         spring's stiffness being 1 + AD_ALPHA x the importance (normalised to its maximum, blurred) at its middle.
         Gauss-Seidel sweeps at AD_RATE a second, from the even lattice when the scene opens: with positive
         stiffnesses and the border fixed on the square this is a barycentric (Tutte) embedding, so cells cannot
         fold. A separable warp, one density per axis, would also make every row and column through the view dense.
         Each Gaussian sits at its cell's centre and is sized to its own cell by the chapter's rule, now through the
         warp: with e_s, e_t the cell's mid-edge spans in (u,v) and J = [S_u S_v], its axes are k J e_s and
         k (J e_t)^perp. Measured: how many Gaussians have their centre in the view; the share of the surface's area
         in the view (on the QA grid); cell side = sqrt(|J e_s x J e_t|), the side of a Gaussian's cell on the
         surface, averaged in the view and out of it, against the same budget on an even lattice. */
      var QA = 64, AD_T = 28, AD_ALPHA = 12, AD_RATE = 600;
      var qa = { pos: new Float32Array(QA * QA * 3), su: new Float32Array(QA * QA * 3), sv: new Float32Array(QA * QA * 3), area: new Float32Array(QA * QA), nrm: new Float32Array(QA * QA * 3) };
      var ad = { ver: -1, fresh: true, w: new Float64Array(QA * QA), m: new Float64Array(QA * QA), tmp: new Float64Array(QA * QA), acc: 0,
                 n: 0, X: null, inG: null, fv: new Float32Array((VF + 1) * (VF + 1)), seg: [], uniVer: -1, uniIn: null, uniS: null, stat: null };
      var aset = footSet(NMAX * NMAX), M4 = new THREE.Matrix4(), q4 = [0, 0, 0];
      aset.m = new Float32Array(NMAX * NMAX * 4);       // per Gaussian: the 1-sigma ellipse in the chart, (a, b) = M (cos, sin)
      function proj(x, y, z) {                        // NDC x, y and w of a world point in the observer camera
        var e = M4.elements, w = e[3] * x + e[7] * y + e[11] * z + e[15];
        q4[0] = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w; q4[1] = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w; q4[2] = w;
        return q4;
      }
      function seen(x, y, z) { var q = proj(x, y, z); return q[2] > 0 && Math.abs(q[0]) <= 1 && Math.abs(q[1]) <= 1; }
      function obsAt(t) {
        var f = t / AD_T * 6.283185, lx = 0.6 * Math.cos(f + 2.25), lz = 0.52 * Math.sin(f + 2.25);
        obs.position.set(1.32 * Math.cos(f), 2.25 + 0.2 * Math.sin(2 * f), 1.08 * Math.sin(f));
        obs.lookAt(lx, 0.12, lz); obs.updateMatrixWorld(); obs.updateProjectionMatrix();
        M4.multiplyMatrices(obs.projectionMatrix, obs.matrixWorldInverse);
      }
      function stiff(u, v) {                          // 1 + AD_ALPHA x the blurred importance, bilinear on the QA grid
        var x = KT.clamp(u * QA - 0.5, 0, QA - 1.001), y = KT.clamp(v * QA - 0.5, 0, QA - 1.001), i = x | 0, j = y | 0, s = x - i, t = y - j, m = ad.m;
        return 1 + AD_ALPHA * ((m[j * QA + i] * (1 - s) + m[j * QA + i + 1] * s) * (1 - t) + (m[(j + 1) * QA + i] * (1 - s) + m[(j + 1) * QA + i + 1] * s) * t);
      }
      function relax() {                              // one Gauss-Seidel sweep over the corners
        var n = ad.n, N = n + 1, X = ad.X, a, b, i, j, su, sv, sw, w, onU, onV, q;
        var nb = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        for (b = 0; b < N; b++) for (a = 0; a < N; a++) {
          onU = a === 0 || a === n; onV = b === 0 || b === n;
          if (onU && onV) continue;
          i = b * N + a; su = 0; sv = 0; sw = 0;
          for (q = 0; q < 4; q++) {
            var p2 = a + nb[q][0], q2 = b + nb[q][1];
            if (p2 < 0 || p2 > n || q2 < 0 || q2 > n || (onU && p2 !== a) || (onV && q2 !== b)) continue;
            j = q2 * N + p2; w = stiff((X[i * 2] + X[j * 2]) / 2, (X[i * 2 + 1] + X[j * 2 + 1]) / 2);
            su += w * X[j * 2]; sv += w * X[j * 2 + 1]; sw += w;
          }
          if (!onU) X[i * 2] = su / sw;
          if (!onV) X[i * 2 + 1] = sv / sw;
        }
      }
      var bl3 = [0, 0, 0];
      function bil(A, u, v, out) {                    // a field of the fine vertex grid at (u, v)
        var x = KT.clamp(u, 0, 1) * VF, y = KT.clamp(v, 0, 1) * VF, i = Math.min(VF - 1, x | 0), j = Math.min(VF - 1, y | 0), s = x - i, t = y - j, c;
        var k00 = (j * (VF + 1) + i) * 3, k10 = k00 + 3, k01 = k00 + (VF + 1) * 3, k11 = k01 + 3;
        for (c = 0; c < 3; c++) out[c] = (A[k00 + c] * (1 - s) + A[k10 + c] * s) * (1 - t) + (A[k01 + c] * (1 - s) + A[k11 + c] * s) * t;
        return out;
      }
      function adaptTick(dt) {
        var n = st.n, N = n + 1, i, a, b, j, t, x, y, z, w, s, W = ad.w;
        vgUpdate();
        if (ad.ver !== surfVer) {                     // the importance grid's surface: positions, area elements, normals
          NB.patchGrid(patch, centres(QA), centres(QA), qa);
          for (i = 0; i < QA * QA; i++) {
            t = i * 3; x = qa.su[t + 1] * qa.sv[t + 2] - qa.su[t + 2] * qa.sv[t + 1]; y = qa.su[t + 2] * qa.sv[t] - qa.su[t] * qa.sv[t + 2]; z = qa.su[t] * qa.sv[t + 1] - qa.su[t + 1] * qa.sv[t];
            w = Math.sqrt(x * x + y * y + z * z) || 1e-9; qa.area[i] = w; qa.nrm[t] = x / w; qa.nrm[t + 1] = y / w; qa.nrm[t + 2] = z / w;
          }
          ad.ver = surfVer; ad.uniVer = -1;
        }
        if (ad.n !== n || ad.fresh) {                 // the even lattice: where the net starts
          ad.n = n; ad.X = new Float64Array(N * N * 2); ad.inG = new Uint8Array(n * n);
          for (b = 0; b < N; b++) for (a = 0; a < N; a++) { ad.X[(b * N + a) * 2] = a / n; ad.X[(b * N + a) * 2 + 1] = b / n; }
          ad.fresh = false; ad.acc = 0; ad.uniVer = -1;
        }
        obsAt(st.t);
        var E = obs.position, mx = 0, seenA = 0, allA = 0, ins = 0;
        for (i = 0; i < QA * QA; i++) {
          t = i * 3; allA += qa.area[i]; W[i] = 0;
          if (!seen(qa.pos[t], qa.pos[t + 1], qa.pos[t + 2])) continue;
          x = E.x - qa.pos[t]; y = E.y - qa.pos[t + 1]; z = E.z - qa.pos[t + 2]; s = x * x + y * y + z * z;
          W[i] = w = qa.area[i] * Math.abs(qa.nrm[t] * x + qa.nrm[t + 1] * y + qa.nrm[t + 2] * z) / (s * Math.sqrt(s));
          if (w > mx) mx = w; seenA += qa.area[i];
        }
        // normalised and blurred (two 5-wide box passes) into the stiffness field
        var M = ad.m, Tm = ad.tmp, r = 2;
        for (i = 0; i < QA * QA; i++) M[i] = mx > 0 ? W[i] / mx : 0;
        for (var pass = 0; pass < 2; pass++) {
          for (b = 0; b < QA; b++) for (a = 0; a < QA; a++) { s = 0; for (j = -r; j <= r; j++) s += M[b * QA + KT.clamp(a + j, 0, QA - 1)]; Tm[b * QA + a] = s / (2 * r + 1); }
          for (b = 0; b < QA; b++) for (a = 0; a < QA; a++) { s = 0; for (j = -r; j <= r; j++) s += Tm[KT.clamp(b + j, 0, QA - 1) * QA + a]; M[b * QA + a] = s / (2 * r + 1); }
        }
        ad.acc += AD_RATE * dt;
        var sw = Math.min(40, Math.floor(ad.acc)); ad.acc -= sw;
        for (i = 0; i < sw; i++) relax();
        // the Gaussians: one per cell, at its centre, sized to it
        var X = ad.X, P = [0, 0, 0], SU = [0, 0, 0], SV = [0, 0, 0], S = aset;
        for (b = 0; b < n; b++) for (a = 0; a < n; a++) {
          var c00 = (b * N + a) * 2, c10 = c00 + 2, c01 = c00 + N * 2, c11 = c01 + 2, k = b * n + a;
          var u = (X[c00] + X[c10] + X[c01] + X[c11]) / 4, v = (X[c00 + 1] + X[c10 + 1] + X[c01 + 1] + X[c11 + 1]) / 4;
          var esu = (X[c10] + X[c11] - X[c00] - X[c01]) / 2, esv = (X[c10 + 1] + X[c11 + 1] - X[c00 + 1] - X[c01 + 1]) / 2;
          var etu = (X[c01] + X[c11] - X[c00] - X[c10]) / 2, etv = (X[c01 + 1] + X[c11 + 1] - X[c00 + 1] - X[c10 + 1]) / 2;
          bil(vg.pos, u, v, P); bil(vg.su, u, v, SU); bil(vg.sv, u, v, SV);
          var As = add(mul(SU, esu), SV, esv), At = add(mul(SU, etu), SV, etv), ls = len(As) || 1e-9, lam = dot(At, As) / (ls * ls);
          var Ap = add(At, As, -lam), lp = len(Ap) || 1e-9, kk = st.gain;
          S.u[k] = u; S.v[k] = v; t = k * 3;
          S.cx[t] = P[0]; S.cx[t + 1] = P[1]; S.cx[t + 2] = P[2];
          S.e1[t] = As[0] / ls; S.e1[t + 1] = As[1] / ls; S.e1[t + 2] = As[2] / ls; S.e2[t] = Ap[0] / lp; S.e2[t + 1] = Ap[1] / lp; S.e2[t + 2] = Ap[2] / lp;
          S.lu[k] = ls * kk; S.lvp[k] = lp * kk; S.du[k] = 1; S.dv[k] = 1; S.pr[k] = 0;        // so that sizeA / sizeB give k |J e_s| and k |(J e_t)^perp|
          S.m[k * 4] = kk * esu; S.m[k * 4 + 1] = kk * (etu - lam * esu); S.m[k * 4 + 2] = kk * esv; S.m[k * 4 + 3] = kk * (etv - lam * esv);
        }
        S.n = n * n;
        if (st.show.field) { writeSet(field, S, 1, 0, 0, violet); st.fieldN = n; }
        if (n <= OUTMAX) outlineM(S);
        // which Gaussians are in the view, and how big their cells are, in it and out of it
        for (i = 0; i < n * n; i++) { t = i * 3; ad.inG[i] = seen(S.cx[t], S.cx[t + 1], S.cx[t + 2]) ? 1 : 0; ins += ad.inG[i]; }
        var sp = cellSide(S, ad.inG, st.gain);
        if (ad.uniVer !== surfVer) {                  // the same budget on an even lattice
          NB.patchGrid(patch, centres(n), centres(n), fg); ad.uniS = footSet(n * n); fillSet(ad.uniS, fg, centres(n), centres(n), n, n, evenD(n), evenD(n));
          ad.uniIn = new Uint8Array(n * n); ad.uniVer = surfVer;
        }
        for (i = 0; i < n * n; i++) { t = i * 3; ad.uniIn[i] = seen(ad.uniS.cx[t], ad.uniS.cx[t + 1], ad.uniS.cx[t + 2]) ? 1 : 0; }
        var su = cellSide(ad.uniS, ad.uniIn, st.gain);
        ad.stat = { n: n * n, ins: ins, view: seenA / allA, sIn: sp.in, sOut: sp.out, uIn: su.in, uAll: su.all };
        footprint();
        uvDirty = true; hudDirty = true;
      }
      /* the outlines of a set whose chart ellipses are stored as M (scene adaptive) */
      function outlineM(S) {
        var A = outl.arr, o = 0, i, m, th, c, sn, pu = 0, pv = 0, qu, qv, Mm = S.m;
        for (i = 0; i < S.n; i++) for (m = 0; m <= EM; m++) {
          th = m / EM * 6.283185; c = Math.cos(th); sn = Math.sin(th);
          qu = S.u[i] + Mm[i * 4] * c + Mm[i * 4 + 1] * sn; qv = S.v[i] + Mm[i * 4 + 2] * c + Mm[i * 4 + 3] * sn;
          if (m) { onSurf(pu, pv, LIFT, A, o); onSurf(qu, qv, LIFT, A, o + 3); o += 6; }
          pu = qu; pv = qv;
        }
        outl.set(o / 3);
      }
      /* the side of each Gaussian's cell on the surface, averaged in the view / out of it / all, as a share of the width.
         The set's sizes are k times the cell's spans, so the cell's area is sizeA sizeB / k^2 */
      function cellSide(S, inF, k) {
        var i, d, si = 0, ni = 0, so = 0, no = 0, sa = 0;
        for (i = 0; i < S.n; i++) {
          d = Math.sqrt(S.lu[i] * S.lvp[i] * S.du[i] * S.dv[i]) / (S.m ? k : 1); sa += d;
          if (inF[i]) { si += d; ni++; } else { so += d; no++; }
        }
        return { in: ni ? si / ni / SPAN : null, out: no ? so / no / SPAN : null, all: S.n ? sa / S.n / SPAN : null };
      }
      /* the part of the surface in the camera's view: marching squares on the fine vertex grid of
         f = 1 - max(|x|, |y|) in the camera's image (f > 0 inside), laid on the surface; a tint inside (mask B) */
      function footprint() {
        var N = VF + 1, f = ad.fv, i, j, t, q, A = fpLine.arr, o = 0, seg = ad.seg, u0, v0;
        for (i = 0; i < N * N; i++) { t = i * 3; q = proj(vg.pos[t], vg.pos[t + 1], vg.pos[t + 2]); f[i] = q[2] > 0 ? 1 - Math.max(Math.abs(q[0]), Math.abs(q[1])) : -1; }
        maskSize(VF);
        for (j = 0; j < VF; j++) for (i = 0; i < VF; i++) {
          var m = (f[j * N + i] + f[j * N + i + 1] + f[(j + 1) * N + i] + f[(j + 1) * N + i + 1]) > 0;
          t = (j * VF + i) * 4; maskData[t] = 0; maskData[t + 1] = 0; maskData[t + 2] = m ? 255 : 0; maskData[t + 3] = 255;
        }
        maskTex.needsUpdate = true;
        seg.length = 0;
        function cut(fa, fb) { return fa / (fa - fb); }
        for (j = 0; j < VF; j++) for (i = 0; i < VF; i++) {
          var f00 = f[j * N + i], f10 = f[j * N + i + 1], f01 = f[(j + 1) * N + i], f11 = f[(j + 1) * N + i + 1], pts = [];
          if ((f00 > 0) !== (f10 > 0)) pts.push(i + cut(f00, f10), j);
          if ((f10 > 0) !== (f11 > 0)) pts.push(i + 1, j + cut(f10, f11));
          if ((f01 > 0) !== (f11 > 0)) pts.push(i + cut(f01, f11), j + 1);
          if ((f00 > 0) !== (f01 > 0)) pts.push(i, j + cut(f00, f01));
          for (var k = 0; k + 3 < pts.length; k += 4) seg.push(pts[k] / VF, pts[k + 1] / VF, pts[k + 2] / VF, pts[k + 3] / VF);
        }
        for (i = 0; i + 3 < seg.length && o + 6 <= A.length; i += 4) { onSurf(seg[i], seg[i + 1], LIFT * 1.5, A, o); onSurf(seg[i + 2], seg[i + 3], LIFT * 1.5, A, o + 3); o += 6; }
        fpLine.set(o / 3);
        // the camera: a small body, the start of its frustum, and four rays to where its corners meet the surface
        var L = [], hw = Math.tan(obs.fov * Math.PI / 360) * obs.aspect, hh = Math.tan(obs.fov * Math.PI / 360), dn = 0.4, bx = 0.17, by = 0.115;
        function box(x0, x1, y0, y1, z0, z1) {
          var c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
          [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]].forEach(function (e) { L.push(c[e[0]], c[e[1]]); });
        }
        box(-bx, bx, -by, by, 0.05, 0.33);
        var nr = [[-hw * dn, -hh * dn, -dn], [hw * dn, -hh * dn, -dn], [hw * dn, hh * dn, -dn], [-hw * dn, hh * dn, -dn]];
        nr.forEach(function (p, i2) { L.push([0, 0, 0], p, p, nr[(i2 + 1) % 4]); });
        L.push([-0.05, by, 0.19], [0, by + 0.08, 0.19], [0, by + 0.08, 0.19], [0.05, by, 0.19]);      // which way is up
        var v3 = new THREE.Vector3(); o = 0;
        L.forEach(function (p) { v3.set(p[0], p[1], p[2]).applyMatrix4(obs.matrixWorld); obsBody.arr[o++] = v3.x; obsBody.arr[o++] = v3.y; obsBody.arr[o++] = v3.z; });
        obsBody.set(L.length);
        o = 0;
        [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (c) {
          var h = rayHit(c[0], c[1]); if (!h) return;
          obsRay.arr[o++] = obs.position.x; obsRay.arr[o++] = obs.position.y; obsRay.arr[o++] = obs.position.z;
          obsRay.arr[o++] = h[0]; obsRay.arr[o++] = h[1]; obsRay.arr[o++] = h[2];
        });
        obsRay.set(o / 3);
      }
      /* where the ray through the image corner (x, y) first meets the surface (marched, then bisected), or null */
      var xu = null;
      function heightAt(x, z) {                        // the surface's height over world (x, z): invert x(u), z(v)
        if (!xu) { xu = new Float64Array(VF + 1); for (var i = 0; i <= VF; i++) xu[i] = vg.pos[i * 3]; }
        function inv(X) { var lo = 0, hi = VF; if (X <= xu[0]) return 0; if (X >= xu[VF]) return 1; while (hi - lo > 1) { var m = (lo + hi) >> 1; if (xu[m] <= X) lo = m; else hi = m; } return (lo + (X - xu[lo]) / (xu[hi] - xu[lo])) / VF; }
        var p = [0, 0, 0]; onSurf(inv(x), inv(z), 0, p, 0); return p[1];
      }
      function rayHit(nx, ny) {
        var v = new THREE.Vector3(nx, ny, 0.5).unproject(obs).sub(obs.position).normalize(), E = obs.position, t = 0, step = 0.03, lo, hi, m, k;
        function under(t) { var x = E.x + v.x * t, z = E.z + v.z * t; return Math.abs(x) <= SPAN / 2 && Math.abs(z) <= SPAN / 2 && E.y + v.y * t <= heightAt(x, z); }
        for (k = 0; k < 400; k++) { t += step; if (under(t)) break; }
        if (k >= 400) return null;
        lo = t - step; hi = t;
        for (k = 0; k < 20; k++) { m = (lo + hi) / 2; if (under(m)) hi = m; else lo = m; }
        return [E.x + v.x * hi, E.y + v.y * hi, E.z + v.z * hi];
      }

      function fmt3(x) { return x >= 1e-3 ? x.toPrecision(3) : x.toExponential(1); }
      function pct(x) { x *= 100; return (x < 0.095 ? x.toFixed(3) : x < 9.95 ? x.toFixed(2) : x.toFixed(1)) + ' %'; }
      function label(rd, html) { rd.el.firstChild.innerHTML = html; }

      /* everything that hangs on the probe: its frame, the tangent plane, its Gaussian, and — in the scale
         scenes — its lattice cell with the eight neighbours and the size estimates */
      var pr = null;
      function updateProbe() {
        var s = st.show, n = st.n, D = 1 / n, u = st.u, v = st.v, k = st.gain, i, j, o, A, q;
        if (s.nbr) {                       // the probe is a sample of the lattice; border cells have no full neighbourhood
          st.ci = KT.clamp(Math.floor(u * n), 1, n - 2); st.cj = KT.clamp(Math.floor(v * n), 1, n - 2);
          u = (st.ci + 0.5) / n; v = (st.cj + 0.5) / n;
        } else D = Math.min(D, u, 1 - u, v, 1 - v) * 0.999;
        var m = pr = measure(patch, u, v, D, G9), p = m.c;
        pr.u = u; pr.v = v;
        var stepU = m.lu * D, stepV = m.lsv * D, gz = s.nbr ? KT.clamp(Math.min(stepU, stepV) / 0.34, 0.06, 1) : 1;
        pr.gz = gz; pr.block = 1.5 * (stepU + stepV);
        probe.position.set(p[0], p[1], p[2]); probe.scale.setScalar(gz);

        // the frame: tangents at a fixed fraction of their length, or — next to the neighbours — as the step they predict
        var LU = s.nbr ? stepU : m.lu * VEC, LV = s.nbr ? stepV : m.lsv * VEC, LN = 0.8 * gz;
        aU.aim(p, m.su[0], m.su[1], m.su[2], LU, gz); aV.aim(p, m.sv[0], m.sv[1], m.sv[2], LV, gz);
        aN.aim(p, m.n[0], m.n[1], m.n[2], LN, gz);
        pr.tipU = add(p, m.su, LU / m.lu); pr.tipV = add(p, m.sv, LV / m.lsv); pr.tipN = add(p, m.n, LN);

        if (s.plane) {
          var PA = mul(m.su, VEC), PB = mul(m.sv, VEC), po = add(p, m.n, 0.006), lo = -0.55, hi = 1.3;
          var pt = function (a, b) { return add(add(po, PA, a), PB, b); };
          [pt(lo, lo), pt(hi, lo), pt(hi, hi), pt(lo, hi), pt(0, 0), pt(1, 0), pt(1, 1), pt(0, 1)].forEach(function (w, i) { plPos.set(w, i * 3); });
          plGeo.attributes.position.needsUpdate = true;
          o = 0; A = plEdge.arr;
          o = seg(A, o, pt(0, 0), pt(1, 0)); o = seg(A, o, pt(1, 0), pt(1, 1)); o = seg(A, o, pt(1, 1), pt(0, 1)); o = seg(A, o, pt(0, 1), pt(0, 0));
          plEdge.set(o / 3);
          o = 0; A = plGrid.arr;           // lines of the plane, parallel to S_u and to S_v
          [lo, -0.5, 0, 0.5, 1, hi].forEach(function (t) { o = seg(A, o, pt(lo, t), pt(hi, t)); o = seg(A, o, pt(t, lo), pt(t, hi)); });
          plGrid.set(o / 3);
        }
        if (s.nor) {                       // right angles: n against each tangent
          var e = 0.13 * gz, tu = mul(m.su, e / m.lu), tv = mul(m.sv, e / m.lsv), tn = mul(m.n, e);
          o = 0; A = mark.arr;
          o = seg(A, o, add(p, tu), add(add(p, tu), tn)); o = seg(A, o, add(add(p, tu), tn), add(p, tn));
          o = seg(A, o, add(p, tv), add(add(p, tv), tn)); o = seg(A, o, add(add(p, tv), tn), add(p, tn));
          mark.set(o / 3);
        }
        if (s.iso) {                       // the two curves the tangents are tangent to
          NB.patchGrid(patch, isoT, [v], ig);
          for (i = 0, o = 0; i < IS; i++) o = segAt(isoU.arr, o, ig.pos, i * 3, i * 3 + 3);
          isoU.set(o / 3);
          NB.patchGrid(patch, [u], isoT, ig);
          for (i = 0, o = 0; i < IS; i++) o = segAt(isoV.arr, o, ig.pos, i * 3, i * 3 + 3);
          isoV.set(o / 3);
        }

        var ref = s.nbr && st.nb === 8 ? m.jf : m.gs, est = st.nb === 8 ? m.n8 : m.n4;
        pr.ref = ref; pr.est = est; pr.gap = st.nb === 8 ? m.gap8 : m.gap4;
        // in scene scale the painted Gaussians take the size the lattice gives them: own cell, or one size for all (tl.b);
        // the white ring stays the size the derivatives ask for
        var bl = s.tile ? tl.b : 0, s1 = tl.s1, re = function (A) { var l = len(A) || 1; return mul(A, KT.lerp(1, s1 / l, bl)); };
        if (s.ell) {                       // the planoid itself, and its 1-sigma outline
          sheet.gauss(4, p, re(mul(ref.ax[0], k)), re(mul(ref.ax[1], k)), violet, s.tile ? 0.5 : 0.9);
          ringA.set(p, mul(ref.ax[0], k), mul(ref.ax[1], k), m.n, 0.014 * gz, 0.022);
        }
        if (s.nbr) {
          ringN.set(p, mul(est.ax[0], k), mul(est.ax[1], k), m.n, 0.010 * gz, 0.06);
          var P = G9.pos, SU = G9.su, SV = G9.sv, ra = 0, ri = 0, w = 0, on, t;
          for (j = 0; j < 9; j++) {
            if (j === 4) continue;
            on = st.nb === 8 || (j & 1) === 1; t = j * 3;
            var nd = nbDot[w++]; nd.position.set(P[t], P[t + 1], P[t + 2]); nd.scale.setScalar(gz * (on ? 1 : 0.8)); nd.material = on ? nbOn : nbOff;
            // each neighbour is a planoid of its own, from its own derivatives
            var sx = [SU[t], SU[t + 1], SU[t + 2]], sy = [SV[t], SV[t + 1], SV[t + 2]], c0 = [P[t], P[t + 1], P[t + 2]];
            sy = add(sy, sx, -dot(sx, sy) / dot(sx, sx)); sx = mul(sx, D * k); sy = mul(sy, D * k);
            sheet.gauss(j, c0, re(sx), re(sy), violet, s.tile ? (on ? 0.3 : 0.1) : (on ? 0.42 : 0.13));
            var nn = cross(sx, sy), l = len(nn) || 1; if (nn[1] < 0) l = -l; c0 = add(c0, nn, 0.006 * gz / l);
            A = on ? nbRingA.arr : nbRingI.arr; o = on ? ra : ri;
            var prev = add(c0, sx), cur;
            for (i = 1; i <= NR; i++) {
              cur = add(add(c0, sx, Math.cos(i / NR * 6.2832)), sy, Math.sin(i / NR * 6.2832)); o = seg(A, o, prev, cur); prev = cur;
            }
            if (on) ra = o; else ri = o;
          }
          nbRingA.set(ra / 3); nbRingI.set(ri / 3);
          // what the size is read from: the two chords (4), or all eight offsets (8)
          o = 0; A = span.arr; q = function (j) { return [P[j * 3], P[j * 3 + 1], P[j * 3 + 2]]; };
          if (st.nb === 4) { o = seg(A, o, q(3), q(5)); o = seg(A, o, q(1), q(7)); }
          else for (j = 0; j < 9; j++) if (j !== 4) o = seg(A, o, p, q(j));
          span.set(o / 3);
          // red: where the first-order step lands short of the real neighbour (the surface curves away)
          o = 0; A = err.arr; o = seg(A, o, add(p, m.su, D), q(5)); o = seg(A, o, add(p, m.sv, D), q(7)); err.set(o / 3);

          // the 3 x 3 block of cells on the surface
          NB.patchGrid(patch, lin(BN, (st.ci - 1) / n, (st.ci + 2) / n), lin(BN, (st.cj - 1) / n, (st.cj + 2) / n), bgd);
          o = 0; A = blk.arr;
          for (j = 0; j < BN; j += BS) for (i = 0; i < BN - 1; i++) {
            o = segAt(A, o, bgd.pos, (j * BN + i) * 3, (j * BN + i + 1) * 3); o = segAt(A, o, bgd.pos, (i * BN + j) * 3, ((i + 1) * BN + j) * 3);
          }
          blk.set(o / 3);
          sheet.uniforms.uCell.value.set(st.ci, st.cj, n, st.nb);
          dots.material.size = 0.05 * gz;

          var eight = st.nb === 8;
          label(rdSn, T('σ from {n} neighbours', { n: st.nb }));
          label(rdSa, eight ? T('σ from J·diag(Δu,Δv)') : T('σ from ‖S<sub>u</sub>‖Δu , ‖S<sub>v</sub><sup>⊥</sup>‖Δv'));
          rdSn.set(fmt3(est.s[0] * k) + ' × ' + fmt3(est.s[1] * k));
          rdSa.set(fmt3(ref.s[0] * k) + ' × ' + fmt3(ref.s[1] * k));
          rdGap.set(pct(pr.gap)); rdGap.tone(pr.gap < 0.01 ? 'ok' : pr.gap < 0.08 ? '' : 'err');
          rdSh.set(pct(m.shear));
        }
        if (s.tan) rdAng.set(m.ang.toFixed(1) + '°');
        faceDirty = true;
      }

      /* ---------- labels that follow the arrow tips ---------- */
      function place(el, w, dx, dy) {
        var q = w && G.project(cam, w[0], w[1], w[2], W, H);
        if (!q) { el.style.opacity = 0; return; }
        el.style.opacity = 1;
        el.style.transform = 'translate(' + (q[0] + dx).toFixed(1) + 'px,' + (q[1] + dy).toFixed(1) + 'px) translate(-50%,-50%)';
      }
      /* a label just beyond the tip, along the arrow as it appears on screen; it also says when the arrow
         points at the viewer or away (the one case where the drawing cannot) */
      var camDir = new THREE.Vector3();
      function placeTip(el, from, tip, name, clear) {
        var qa = G.project(cam, from[0], from[1], from[2], W, H), qb = G.project(cam, tip[0], tip[1], tip[2], W, H);
        if (!qa || !qb) { el.style.opacity = 0; return null; }
        var dx = qb[0] - qa[0], dy = qb[1] - qa[1], L = Math.hypot(dx, dy);
        if (L < 12) { dx = 0.62; dy = -0.78; } else { dx /= L; dy /= L; }
        var d = add(tip, from, -1), l = len(d) || 1;
        camDir.copy(cam.position).sub(probe.position).normalize();
        var z = (d[0] * camDir.x + d[1] * camDir.y + d[2] * camDir.z) / l;
        var html = name + (z > 0.9 ? '<i>' + T('⊙ towards you') + '</i>' : z < -0.9 ? '<i>' + T('⊗ away') + '</i>' : '');
        if (el._html !== html) { el.innerHTML = html; el._html = html; }
        var x = qb[0] + dx * 20, y = qb[1] + dy * 15;
        if (clear && Math.abs(x - clear[0]) < 64 && Math.abs(y - clear[1]) < 24) y = clear[1] + (y < clear[1] ? -24 : 24);   // not on top of the other label
        el.style.opacity = 1;
        el.style.transform = 'translate(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px) translate(-50%,-50%)';
        return [x, y];
      }

      /* ---------- the (u,v) chart ---------- */
      function drawUV() {
        var z = uvc.fit(), g = uvc.ctx, w = z.w, h = z.h, i, n = st.n, s = st.show, a2, b2;
        g.clearRect(0, 0, w, h);
        g.strokeStyle = rgba(C.ctrl, 0.38); g.lineWidth = 1;               // knot lines = patch seams of the basis
        var inner = NU - 3;
        for (i = 1; i < inner; i++) {
          g.beginPath(); g.moveTo(i / inner * w, 0); g.lineTo(i / inner * w, h); g.moveTo(0, i / inner * h); g.lineTo(w, i / inner * h); g.stroke();
        }
        if (s.tile && tl.imgOwn) {                         // red: the points no footprint covers, own cell -> one size
          g.imageSmoothingEnabled = true;
          if (tl.b < 1) { g.globalAlpha = 1 - tl.b; g.drawImage(tl.imgOwn, 0, 0, w, h); }
          if (tl.b > 0) { g.globalAlpha = tl.b; g.drawImage(tl.imgOne, 0, 0, w, h); }
          g.globalAlpha = 1;
        }
        function ellipses(S, k, s1, bl, from, alpha) {   // the footprints, in the chart where they are defined
          g.strokeStyle = rgba(C.splat, alpha); g.lineWidth = 1; g.beginPath();
          for (var q = from; q < S.n; q++) {
            var LA = sizeA(S, q, k, s1, bl), LB = sizeB(S, q, k, s1, bl), e1 = S.lu[q] / LA, e2 = S.pr[q] / LA, e3 = S.lvp[q] / LB, m, th, eb, ea, x, y;
            for (m = 0; m <= 28; m++) {
              th = m / 28 * 6.283185; eb = Math.sin(th) / e3; ea = (Math.cos(th) - e2 * eb) / e1;
              x = (S.u[q] + ea) * w; y = (1 - S.v[q] - eb) * h;
              if (m) g.lineTo(x, y); else g.moveTo(x, y);
            }
          }
          g.stroke();
        }
        if (s.stitch && sx.n) {
          var sn = sx.n, SX = function (q) { return (q + 0.5) / sn * w; }, SY = function (q) { return (1 - (q + 0.5) / sn) * h; }, qk, qa, qb, rr2 = Math.max(1.2, Math.min(3, w / sn * 0.2));
          g.lineWidth = 1;
          for (qk = 0; qk < sx.q; qk++) {
            qa = quadAt(sn, qk).a; qb = quadAt(sn, qk).b;
            g.fillStyle = qk === sx.q - 1 && sx.q < sx.Q ? (LIGHT ? rgba(INK, 0.2) : 'rgba(255,255,255,.34)') : rgba(C.surf, 0.2);
            g.beginPath(); g.moveTo(SX(qa), SY(qb)); g.lineTo(SX(qa + 1), SY(qb)); g.lineTo(SX(qa + 1), SY(qb + 1)); g.lineTo(SX(qa), SY(qb + 1)); g.closePath(); g.fill();
            g.strokeStyle = rgba(C.surf, 0.85); g.stroke();
            g.beginPath(); g.moveTo(SX(qa), SY(qb)); g.lineTo(SX(qa + 1), SY(qb + 1)); g.stroke();
          }
          for (qb = 0; qb < sn; qb++) for (qa = 0; qa < sn; qa++) {
            var gone = (sn - 1 - qb) * sn + qa < sx.passed;
            g.fillStyle = gone ? C.surf : C.splat; g.beginPath(); g.arc(SX(qa), SY(qb), gone ? rr2 * 0.7 : rr2, 0, 6.2832); g.fill();
          }
          if (sx.q < sx.Q) {                                 // the window around the four samples it holds
            qk = Math.max(0, sx.q - 1); qa = quadAt(sn, qk).a; qb = quadAt(sn, qk).b;
            var pad = Math.max(3, w / sn * 0.3);
            g.strokeStyle = INK; g.lineWidth = 1.6;
            g.strokeRect(SX(qa) - pad, SY(qb + 1) - pad, SX(qa + 1) - SX(qa) + 2 * pad, SY(qb) - SY(qb + 1) + 2 * pad);
          }
        }
        if (s.adapt && ad.stat) {
          if (!ad.cv) { ad.cv = document.createElement('canvas'); ad.cv.width = ad.cv.height = VF; ad.im = ad.cv.getContext('2d').createImageData(VF, VF); }
          var D = ad.im.data, fu, fv, o2, VC = LIGHT ? hex3(C.text) : [230, 230, 230], VA = LIGHT ? 22 : 52;     // the camera's view: a veil, pale on dark, ink on paper
          for (fv = 0; fv < VF; fv++) for (fu = 0; fu < VF; fu++) {
            o2 = ((VF - 1 - fv) * VF + fu) * 4; D[o2] = VC[0]; D[o2 + 1] = VC[1]; D[o2 + 2] = VC[2]; D[o2 + 3] = maskData[(fv * VF + fu) * 4 + 2] ? VA : 0;
          }
          ad.cv.getContext('2d').putImageData(ad.im, 0, 0); g.imageSmoothingEnabled = true; g.drawImage(ad.cv, 0, 0, w, h);
          var an = ad.n, AN = an + 1, X2 = ad.X, aa, ab, o3;
          // the net of cell corners, as the springs have pulled it, and a dot per Gaussian
          g.strokeStyle = rgba(C.splat, 0.38); g.lineWidth = 1; g.beginPath();
          for (ab = 0; ab < AN; ab++) for (aa = 0; aa < AN; aa++) {
            o3 = (ab * AN + aa) * 2;
            if (aa) { g.moveTo(X2[o3 - 2] * w, (1 - X2[o3 - 1]) * h); g.lineTo(X2[o3] * w, (1 - X2[o3 + 1]) * h); }
            if (ab) { g.moveTo(X2[o3 - AN * 2] * w, (1 - X2[o3 - AN * 2 + 1]) * h); g.lineTo(X2[o3] * w, (1 - X2[o3 + 1]) * h); }
          }
          g.stroke();
          var rd2 = Math.max(0.8, Math.min(1.8, w / an * 0.14));
          for (aa = 0; aa < an * an; aa++) {
            g.fillStyle = ad.inG[aa] ? C.splat : rgba(C.splat, 0.6);
            g.fillRect(aset.u[aa] * w - rd2, (1 - aset.v[aa]) * h - rd2, 2 * rd2, 2 * rd2);
          }
          g.strokeStyle = C.text; g.lineWidth = 1.3; g.beginPath();
          for (aa = 0; aa + 3 < ad.seg.length; aa += 4) { g.moveTo(ad.seg[aa] * w, (1 - ad.seg[aa + 1]) * h); g.lineTo(ad.seg[aa + 2] * w, (1 - ad.seg[aa + 3]) * h); }
          g.stroke();
        }
        function cells(x0, y0, cw, ch, ci, cj) {          // the cell (ci, cj) and its neighbourhood, cell size cw x ch at (x0, y0) for cell (0, 0)
          for (b2 = -1; b2 <= 1; b2++) for (a2 = -1; a2 <= 1; a2++) {
            var x = x0 + (ci + a2) * cw, y = y0 - (cj + b2 + 1) * ch, mid = !a2 && !b2, on = mid || st.nb === 8 || !a2 || !b2;
            g.fillStyle = mid ? (LIGHT ? rgba(INK, 0.85) : 'rgba(255,255,255,.92)') : on ? rgba(C.splat, 0.62) : rgba(C.splat, 0.10);
            g.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
            if (!on) { g.strokeStyle = rgba(C.splat, 0.5); g.lineWidth = 1; g.setLineDash([2, 2]); g.strokeRect(x + 1, y + 1, cw - 2, ch - 2); g.setLineDash([]); }
          }
        }
        if (s.nbr && n <= LATMAX) {                        // the lattice of cells
          g.strokeStyle = rgba(C.splat, 0.22); g.lineWidth = 1; g.beginPath();
          for (i = 1; i < n; i++) { g.moveTo(i / n * w, 0); g.lineTo(i / n * w, h); g.moveTo(0, i / n * h); g.lineTo(w, i / n * h); }
          g.stroke();
        }
        if (s.nbr) cells(0, h, w / n, h / n, st.ci, st.cj);
        if (s.tile && tset.n && n <= OUTMAX) ellipses(tset, st.gain, tl.s1, tl.b, 0, s.nbr ? 0.75 : 0.9);
        if ((s.field || s.ell) && !s.stitch && !s.adapt) {
          g.fillStyle = C.splat;
          if (n <= 44) { var r = Math.max(1, Math.min(3, w / n * 0.22));
            for (b2 = 0; b2 < n; b2++) for (a2 = 0; a2 < n; a2++) { g.beginPath(); g.arc((a2 + 0.5) / n * w, (b2 + 0.5) / n * h, r, 0, 6.2832); g.fill(); } }
          else { g.globalAlpha = Math.min(0.75, 0.25 + n / 320) * (s.nbr ? 0.35 : 1); g.fillRect(0, 0, w, h); g.globalAlpha = 1; }
        }
        if (s.probe) {
          var pu = s.nbr ? (st.ci + 0.5) / n : st.u, pv = s.nbr ? (st.cj + 0.5) / n : st.v, x = pu * w, y = (1 - pv) * h;
          if (s.ell && !s.nbr) { g.strokeStyle = C.splat; g.lineWidth = 1.5; g.strokeRect(x - w / n / 2, y - h / n / 2, w / n, h / n); }
          g.strokeStyle = s.nbr ? rgba(C.text, 0.45) : C.text; g.lineWidth = 1.2;
          g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
          g.fillStyle = s.nbr ? C.bg : C.text; g.beginPath(); g.arc(x, y, s.nbr ? Math.min(3, w / n * 0.3) : 4, 0, 6.2832); g.fill();
          if (s.nbr && w / n < 11) {                       // cells too small to read: a magnified copy of the 5 x 5 window
            var L = Math.round(w * 0.5), lx = pu < 0.5 ? w - L - 5 : 5, ly = pv < 0.5 ? 20 : h - L - 5, cs2 = L / 5;
            g.strokeStyle = rgba(C.text, 0.5); g.lineWidth = 1;
            g.strokeRect(x - 2.5 * w / n, y - 2.5 * h / n, 5 * w / n, 5 * h / n);
            g.beginPath(); g.moveTo(x, y); g.lineTo(lx + L / 2, ly + L / 2); g.stroke();
            g.fillStyle = LIGHT ? rgba(C.bg, 0.96) : 'rgba(11,16,26,.96)'; g.fillRect(lx, ly, L, L); g.strokeRect(lx + 0.5, ly + 0.5, L - 1, L - 1);
            g.save(); g.beginPath(); g.rect(lx, ly, L, L); g.clip();
            cells(lx + (2 - st.ci) * cs2, ly + (st.cj + 3) * cs2, cs2, cs2, st.ci, st.cj);
            g.restore();
          }
        }
        // the label is set in capitals: a lower-case σ would turn into Σ, so it keeps its case
        var lab = s.nbr ? (w < 150 ? T('(u, v) · {n} nbrs', { n: st.nb }) : T('(u, v) · {n} neighbours', { n: st.nb })) : s.stitch ? (w < 150 ? T('2 × 2 window') : T('(u, v) · 2 × 2 window')) :
                  s.adapt ? (w < 150 ? T('(u, v) · warped') : T('(u, v) · warped lattice')) : s.tile ? (w < 150 ? '(u, v) · 1<i>σ</i>' : T('(u, v) · 1<i>σ</i> footprints')) : T('(u, v) chart');
        if (lab !== uvLabHtml) { uvLabHtml = lab; uvLab.innerHTML = lab; }
      }

      /* ---------- the tangent plane, face-on ----------
         Drawn in the plane's own orthonormal frame (x = S_u^, y = n x S_u^), seen from the side n points to:
         whatever the 3D camera does, S_u and S_v are shown here undistorted, with the true angle between them. */
      function drawFace() {
        var z = fc.fit(), g = fc.ctx, w = z.w, h = z.h, m = pr, s = st.show, i, j;
        g.clearRect(0, 0, w, h);
        if (!m) return;
        var small = w < 150, top = 24, bot = small ? 20 : 24, cx, cy, sc, F = small ? 10 : 11;
        function X(d) { return cx + dot(d, m.e1) * sc; }
        function Y(d) { return cy - dot(d, m.e2) * sc; }
        function line(p, q) { g.beginPath(); g.moveTo(X(p), Y(p)); g.lineTo(X(q), Y(q)); g.stroke(); }
        function arrow(d, col, wd) {
          var x1 = X(d), y1 = Y(d), an = Math.atan2(y1 - cy, x1 - cx), hl = Math.min(8, Math.hypot(x1 - cx, y1 - cy) * 0.4);
          g.strokeStyle = g.fillStyle = col; g.lineWidth = wd;
          g.beginPath(); g.moveTo(cx, cy); g.lineTo(x1 - Math.cos(an) * hl * 0.8, y1 - Math.sin(an) * hl * 0.8); g.stroke();
          g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - Math.cos(an - 0.42) * hl, y1 - Math.sin(an - 0.42) * hl);
          g.lineTo(x1 - Math.cos(an + 0.42) * hl, y1 - Math.sin(an + 0.42) * hl); g.closePath(); g.fill();
          return an;
        }
        function sub(txt, lo, x, y, col) {                // "S" with a subscript, centred on (x, y)
          g.fillStyle = col; g.font = '600 ' + F + 'px JetBrains Mono, ui-monospace, monospace'; g.textAlign = 'left'; g.textBaseline = 'middle';
          var w1 = g.measureText(txt).width; g.fillText(txt, x - w1 / 2 - 2, y);
          g.font = '600 ' + (F - 3) + 'px JetBrains Mono, ui-monospace, monospace'; g.fillText(lo, x + w1 / 2 - 2, y + 3);
        }
        function ellipse(A, B, col, wd) {
          g.strokeStyle = col; g.lineWidth = wd; g.beginPath();
          for (i = 0; i <= 64; i++) {
            var d = add(mul(A, Math.cos(i / 64 * 6.2832)), B, Math.sin(i / 64 * 6.2832));
            if (i) g.lineTo(X(d), Y(d)); else g.moveTo(X(d), Y(d));
          }
          g.stroke();
        }
        function foot(txt, col) {
          g.font = '600 ' + F + 'px JetBrains Mono, ui-monospace, monospace'; g.textAlign = 'left'; g.textBaseline = 'alphabetic';
          g.fillStyle = col; g.fillText(txt, 7, h - 7);
        }
        var su = m.su, sv = m.sv;
        if (!s.nbr) {
          // fit the parallelogram { 0, S_u, S_v, S_u + S_v }
          var pts = [[0, 0, 0], su, sv, add(su, sv)], x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
          pts.forEach(function (p) { var x = dot(p, m.e1), y = dot(p, m.e2); x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); });
          var mx = small ? 14 : 25, my = small ? 8 : 15, aw = w - 2 * mx - (small ? 12 : 0), ah = h - top - bot - 2 * my;
          sc = Math.min(aw / (x1 - x0), ah / (y1 - y0));
          cx = mx + (aw - (x1 - x0) * sc) / 2 - x0 * sc; cy = top + my + (ah - (y1 - y0) * sc) / 2 + y1 * sc;
          g.fillStyle = rgba(C.surf, 0.13); g.strokeStyle = rgba(SURF_HI, 0.5); g.lineWidth = 1;
          g.beginPath(); pts = [pts[0], su, add(su, sv), sv];
          pts.forEach(function (p, i) { if (i) g.lineTo(X(p), Y(p)); else g.moveTo(X(p), Y(p)); });
          g.closePath(); g.fill(); g.stroke();
          var aV2 = Math.atan2(Y(sv) - cy, X(sv) - cx), rr = small ? 13 : 17;
          if (s.nor) {                                   // the axis the rotation actually uses: S_v made perpendicular to S_u
            g.setLineDash([3, 3]); g.strokeStyle = COL_V; g.lineWidth = 1.2; line([0, 0, 0], m.svp); g.setLineDash([]);
            var sgn = Y(m.svp) > cy ? 1 : -1;
            g.strokeStyle = rgba(C.text, 0.8); g.lineWidth = 1; g.strokeRect(cx, sgn > 0 ? cy : cy - 7, 7, 7);
            if (!small && Math.abs(m.ang - 90) > 3) sub('S', 'v⊥', X(m.svp) + (m.ang > 90 ? 16 : -18), Y(m.svp) - sgn * 12, rgba(COL_V, 0.85));
          }
          g.strokeStyle = rgba(C.text, 0.9); g.lineWidth = 1.2; g.beginPath(); g.arc(cx, cy, rr, 0, aV2, aV2 < 0); g.stroke();
          arrow(su, COL_U, 2.2); arrow(sv, COL_V, 2.2);
          sub('S', 'u', X(su) + 9, Y(su) + (aV2 < 0 ? 9 : -9), COL_U); sub('S', 'v', X(sv) + 11, Y(sv), COL_V);
          g.fillStyle = C.text; g.beginPath(); g.arc(cx, cy, 2.6, 0, 6.2832); g.fill();
          if (s.nor) {                                   // n comes straight out of the picture
            g.strokeStyle = INK; g.lineWidth = 1.4; g.beginPath(); g.arc(cx, cy, 5.5, 0, 6.2832); g.stroke();
            if (!small) sub('n', '', cx - 10, cy + (aV2 < 0 ? 12 : -12), INK);
          }
          foot((small ? '∠ ' : '∠(Su, Sv) = ') + m.ang.toFixed(1) + '°', C.text);
          return;
        }
        // the neighbourhood, to scale: the lattice the Jacobian predicts, the real neighbours, the estimates
        var D = m.D, A = mul(su, D), B = mul(sv, D), k = st.gain, ext = 0, d;
        for (j = 0; j < 9; j++) ext = Math.max(ext, Math.abs(dot(m.d[j], m.e1)), Math.abs(dot(m.d[j], m.e2)));
        sc = Math.min(w - 12, h - top - bot) / 2 / (ext * 1.12); cx = w / 2; cy = top + (h - top - bot) / 2;
        g.fillStyle = rgba(C.splat, 0.16); g.beginPath();
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].forEach(function (q, i) {
          d = add(mul(A, q[0]), B, q[1]); if (i) g.lineTo(X(d), Y(d)); else g.moveTo(X(d), Y(d));
        });
        g.closePath(); g.fill();
        g.strokeStyle = rgba(C.surf, 0.32); g.lineWidth = 1;
        for (i = -1.5; i <= 1.5; i++) { line(add(mul(A, i), B, -1.5), add(mul(A, i), B, 1.5)); line(add(mul(B, i), A, -1.5), add(mul(B, i), A, 1.5)); }
        g.strokeStyle = rgba(SPLAT_HI, 0.7); g.lineWidth = 1;
        if (st.nb === 4) { line(m.d[3], m.d[5]); line(m.d[1], m.d[7]); }
        else for (j = 0; j < 9; j++) if (j !== 4) line([0, 0, 0], m.d[j]);
        ellipse(mul(m.est.ax[0], k), mul(m.est.ax[1], k), C.splat, 3.2);
        ellipse(mul(m.ref.ax[0], k), mul(m.ref.ax[1], k), INK, 1.3);
        arrow(A, COL_U, 2); arrow(B, COL_V, 2);
        for (j = 0; j < 9; j++) {
          var on = j === 4 || st.nb === 8 || (j & 1) === 1;
          g.fillStyle = j === 4 ? INK : on ? C.splat : rgba(C.splat, 0.4);
          g.beginPath(); g.arc(X(m.d[j]), Y(m.d[j]), j === 4 ? 3 : on ? 3.4 : 2.4, 0, 6.2832); g.fill();
        }
        foot(small ? pct(m.gap) : T('differ by {gap}', { gap: pct(m.gap) }), m.gap < 0.01 ? C.ok : C.text);
        if (!small) { g.textAlign = 'right'; g.fillStyle = C.dim; g.fillText('∠ ' + m.ang.toFixed(0) + '°', w - 7, h - 7); }
      }

      /* measured, not asserted: the gap between the neighbour estimate and its derivative twin at this very
         sample, for a range of lattice densities (log-log). It falls as 1 / n^2. */
      var SPN = [4, 5, 6, 8, 10, 13, 17, 22, 28, 36, 46, 60, 78, 100, 128, 160];
      function drawSpark() {
        var z = spc.fit(), g = spc.ctx, w = z.w, h = z.h, i, pts = [], eight = st.nb === 8;
        g.clearRect(0, 0, w, h);
        if (!pr) return;
        var lx0 = Math.log(4), lx1 = Math.log(160), ly0 = Math.log(5e-5), ly1 = Math.log(0.6);
        function X(n) { return 6 + (Math.log(n) - lx0) / (lx1 - lx0) * (w - 12); }
        function Y(e) { return 4 + (ly1 - Math.log(Math.max(e, 5e-5))) / (ly1 - ly0) * (h - 8); }
        for (i = 0; i < SPN.length; i++) {
          var D = 1 / SPN[i];
          if (pr.u - D < 0 || pr.u + D > 1 || pr.v - D < 0 || pr.v + D > 1) continue;
          var m = measure(patch, pr.u, pr.v, D, G9b); pts.push([SPN[i], eight ? m.gap8 : m.gap4]);
        }
        g.strokeStyle = rgba(C.dim, 0.22); g.lineWidth = 1; g.font = '500 9px JetBrains Mono, ui-monospace, monospace';
        g.fillStyle = C.mute; g.textBaseline = 'middle'; g.textAlign = 'right';
        [1e-1, 1e-2, 1e-3, 1e-4].forEach(function (e) {
          g.beginPath(); g.moveTo(6, Y(e)); g.lineTo(w - 6, Y(e)); g.stroke(); g.fillText(+(e * 100).toPrecision(1) + '%', w - 7, Y(e) - 6);
        });
        if (pts.length > 1) {
          var last = pts[pts.length - 1];
          g.setLineDash([3, 3]); g.strokeStyle = rgba(C.text, 0.45); g.beginPath();       // a 1 / n^2 line through the finest point
          g.moveTo(X(4), Y(last[1] * last[0] * last[0] / 16)); g.lineTo(X(last[0]), Y(last[1])); g.stroke(); g.setLineDash([]);
          g.strokeStyle = C.splat; g.lineWidth = 2; g.beginPath();
          pts.forEach(function (p, i) { if (i) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1])); });
          g.stroke();
        }
        g.fillStyle = INK; g.beginPath(); g.arc(X(st.n), Y(pr.gap), 3.2, 0, 6.2832); g.fill();
        g.fillStyle = C.dim; g.textAlign = 'left'; g.fillText(T('differ by, vs samples · dashed = 1/n²'), 6, 9);
      }

      /* ---------- scenes ---------- */
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.t = 0; st.touchedProbe = false; st.touchedN = false; st.wigI = -1;
        st.show = {}; sc.show.split(' ').forEach(function (k) { st.show[k] = true; });
        if (!sc.sweep) knBig(false);                     // leaving the knob frees its big buffers
        if (sc.fit) base.set(fitBase);                   // the fit starts over: from a flat net, towards the terrain
        patch.P.set(base);
        st.fit = null;
        if (sc.fit) {
          var Z0 = new Float64Array(FIT.n).fill(FIT.mean);
          for (var zi = 0; zi < FIT.n; zi++) patch.P[zi * 3 + 1] = Z0[zi];
          st.fit = cgNew(Z0);
        }
        if (sc.n) { st.n = sc.n; slN.set(sc.n); }
        if (sc.fixed) { st.u = sc.fixed[0]; st.v = sc.fixed[1]; }
        small.uniforms.uRim.value = rim(sc.rim == null ? 0.6 : sc.rim);
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        var s = st.show, on = function (o, v) { o.visible = !!v; };
        // restart what runs on its own: the size alternation, the stitching, the camera's path, the knob
        tl.auto = true; tl.goal = 0; tl.b = 0; tl.key = ''; tl.mode = 'own'; segSz.set('own');
        sx.n = 0; ad.fresh = true; kn.li = 0; kn.top = false; kn.lim = null; kn.ft = {}; kn.last = 0; kn.redo = false;
        if (sc.sweep) knSet(KN_LEVELS[0]);
        on(tgtDots, s.fit); on(tgtHid, s.fit); on(resid.obj, s.fit); on(residHid, s.fit);
        on(ovMesh, s.tile || s.adapt); ovMat.uniforms.uGap.value = s.tile ? 1 : 0; ovMat.uniforms.uTint.value = s.adapt ? 1 : 0;
        on(ovDepth, s.tile || s.stitch || s.adapt);
        outl.mat.color.set(s.field ? OUTL_ON : C.splat); outl.mat.opacity = s.field ? 0.85 : 0.95;   // on the violet Gaussians: a deep violet
        [tri, triEdge.obj, triNew.obj, win, winDot].forEach(function (o) { on(o, s.stitch); });
        triGeo.setDrawRange(0, 0); triEdge.set(0);
        [obsBody.obj, obsRay.obj, fpLine.obj].forEach(function (o) { on(o, s.adapt); });
        rdFit.el.style.display = rdMax.el.style.display = rdIt.el.style.display = s.fit ? '' : 'none';
        segSz.el.style.display = rdUnc.el.style.display = rdPile.el.style.display = s.tile ? '' : 'none';
        rdTri.el.style.display = s.stitch ? '' : 'none';
        slK.el.style.display = rdFt.el.style.display = rdEv.el.style.display = rdLim.el.style.display = knBox.style.display = sc.sweep ? '' : 'none';
        [rdIn, rdFp, rdSpc, rdUni, rdEq].forEach(function (r) { r.el.style.display = s.adapt ? '' : 'none'; });
        label(rdP, sc.sweep ? T('stored numbers') : T('learnable params'));
        hudHtml = null; hudDirty = true;
        on(net.obj, s.net); on(netGhost, s.net);
        balls.forEach(function (m, i) {
          m.visible = !!s.net;
          if (s.feat) m.material.color.setRGB(patch.C[i * 3], patch.C[i * 3 + 1], patch.C[i * 3 + 2], THREE.SRGBColorSpace);
          else m.material.color.set(C.ctrl);
          m.material.emissive.copy(m.material.color).multiplyScalar(0.32);
          m.children[0].visible = !s.feat;
          m.scale.setScalar(s.feat ? 1.25 : 1);
        });
        wire.mat.opacity = s.field && !sc.hero ? 0.22 : s.tan ? 0.34 : 0.55;
        on(field.mesh, s.field); if (field !== small) small.mesh.visible = false;
        on(probe, s.probe); on(aU.obj, s.tan); on(aV.obj, s.tan); on(aN.obj, s.nor);
        on(isoU.obj, s.iso); on(isoV.obj, s.iso);
        on(plane, s.plane); on(planeHid, s.plane); on(plGrid.obj, s.plane); on(plEdge.obj, s.plane); on(plEdgeHid, s.plane);
        on(mark.obj, s.nor);
        on(fill, !s.field || sc.hero);
        for (var gi = 0; gi < 9; gi++) sheet.gauss(gi, null, null, null, null, 0);
        sheet.uniforms.uCell.value.z = 0;
        on(ringA.mesh, s.ell); on(ringA.hid, s.ell);
        on(ringN.mesh, s.nbr); on(ringN.hid, s.nbr);
        [blk.obj, span.obj, spanHid, err.obj].concat(nbDot).forEach(function (o) { on(o, s.nbr); });
        on(nbRingA.obj, s.nbr && !s.tile); on(nbRingI.obj, s.nbr && !s.tile);      // with every outline drawn, these would be twice
        tMu.style.display = s.probe && !s.nbr ? '' : 'none'; tU.style.display = tV.style.display = s.tan ? '' : 'none';
        tN.style.display = s.nor ? '' : 'none';
        uvBox.style.display = s.uv ? '' : 'none';
        fcBox.style.display = s.face ? '' : 'none'; fcBox.classList.toggle('kt-solo', !s.uv);
        root.classList.toggle('kt-has-face', !!s.face); root.classList.toggle('kt-nbr', !!s.nbr);
        nbBar.style.display = s.nbr ? '' : 'none';
        panel.style.display = sc.hero ? 'none' : ''; cap.style.display = sc.hero ? 'none' : '';
        slG.el.style.display = rdG.el.style.display = (s.field || s.ell || s.adapt) ? '' : 'none';
        slN.el.style.display = (s.field || s.ell || s.adapt) && !sc.sweep ? '' : 'none';
        rdAng.el.style.display = s.tan ? '' : 'none';
        rdSn.el.style.display = rdSa.el.style.display = rdGap.el.style.display = spBox.style.display = s.nbr ? '' : 'none';
        on(grid, !sc.hero); on(glow, !sc.hero); shadows.forEach(function (m) { on(m, !sc.hero); });
        st.nud.th = st.nud.ph = 0; st.sparkT = 0;
        surfDirty = fieldDirty = uvDirty = latDirty = probeDirty = true;
        layers();
      }
      /* what depends on the sample count as well as on the scene */
      function layers() {
        var s = st.show, fine = s.nbr && st.n <= LATMAX;
        lat.obj.visible = dots.visible = !!fine;
        outl.obj.visible = !!(s.tile || s.stitch || s.adapt) && st.n <= OUTMAX;
        wire.obj.visible = !!s.wire || (!!s.nbr && !fine);
        rdSh.el.style.display = s.nbr && st.nb === 8 ? '' : 'none';
        rdG.set((st.n * st.n).toLocaleString('en-US'));
      }

      function wrap(a) { return a - 6.283185307 * Math.round(a / 6.283185307); }

      env.loop(function (t, dt) {
        var sc = st.sc, s = st.show, i, j, o;
        st.t += dt;
        if (sc.hero) {                         // the cover: a breathing surface, splats sweeping across it
          for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
            o = (j * NU + i) * 3; patch.P[o + 1] = base[o + 1] + 0.2 * Math.sin(st.t * 0.7 + i * 0.9 + j * 0.6);
          }
          surfDirty = fieldDirty = true;
        }
        if (sc.wig && !dragCP) {               // wiggle the control point that weighs most on the probe
          if (st.wigI < 0) {
            var best = 0, pu = s.nbr ? (KT.clamp(Math.floor(st.u * st.n), 1, st.n - 2) + 0.5) / st.n : st.u,
                pv = s.nbr ? (KT.clamp(Math.floor(st.v * st.n), 1, st.n - 2) + 0.5) / st.n : st.v;
            NB.patchSupport(patch, pu, pv).forEach(function (c) { if (c.w > best) { best = c.w; st.wigI = c.j * NU + c.i; } });
          }
          o = st.wigI * 3;
          patch.P[o + 1] = base[o + 1] + 0.55 * Math.sin(st.t * 1.5); patch.P[o] = base[o] + 0.3 * Math.sin(st.t * 0.9);
          surfDirty = true;
        }
        if (s.probe && !sc.fixed && !st.touchedProbe) {
          st.u = 0.5 + 0.3 * Math.sin(st.t * 0.5); st.v = 0.5 + 0.3 * Math.sin(st.t * 0.37 + 1); uvDirty = probeDirty = true;
        }
        if (sc.fit) fitTick(dt);                                  // the optimiser pulls the net onto the terrain
        if (sc.sweep) knTick(dt);                                 // the knob steps on, and measures
        if (s.tile) tileTick(dt);                                 // one size for all <-> own cell
        if (surfDirty) {
          updateSurface(); surfVer++; probeDirty = true; if (s.nbr) latDirty = true; surfDirty = false;
          if (s.fit) updateResid();
        }
        if (latDirty) { layers(); if (s.nbr) updateLattice(); latDirty = false; }
        if (s.tile) tileUpdate();
        if (s.stitch) stitchTick();
        if (s.adapt) adaptTick(dt);
        if (s.field && fieldDirty && !sc.sweep && !s.tile && !s.stitch && !s.adapt) updateField();
        fieldDirty = false;
        if (s.field && !s.stitch && !sc.sweep) {
          var cnt = st.fieldN * st.fieldN;
          if (sc.hero) cnt = Math.round((0.5 - 0.42 * Math.cos(st.t * 0.33)) * st.fieldN) * st.fieldN;
          field.setCount(cnt);
        }
        if (s.probe && probeDirty) { updateProbe(); probeDirty = false; }

        /* ---- camera: a base view per scene, a rocking for parallax, and a step aside when the plane is edge-on ---- */
        var g = orbit.goal, view = sc.hero ? { th: 0.9, ph: 1.12, r: 5.6 } : sc.view ? { th: sc.view.th, ph: sc.view.ph, r: sc.view.r } : { th: 0.72, ph: 1.0, r: 7.2 },
            tg = [0, sc.view && sc.view.ty != null ? sc.view.ty : 0.05, 0];
        if (s.tan && pr) {                     // look at the probe; next to the neighbours, as close as the 3 x 3 block allows
          var fo = s.nbr ? (sc.wig ? 0.8 : 1) : 0.35;
          tg = [pr.c[0] * fo, 0.05 + (pr.c[1] - 0.05) * fo, pr.c[2] * fo];
          view.r = sc.wig ? 6.0 : s.nbr ? KT.clamp(4.6 * pr.block, 0.42, 5.4) : 6.5;
          if (s.nbr) {                         // and from above its own tangent plane: lean the base view towards the normal
            view.ph = 0.8;
            if (!sc.wig) {
              var lx = Math.sin(view.ph) * Math.sin(view.th) * 1.5 + pr.n[0], ly = Math.cos(view.ph) * 1.5 + pr.n[1],
                  lz = Math.sin(view.ph) * Math.cos(view.th) * 1.5 + pr.n[2];
              view.th = Math.atan2(lx, lz); view.ph = Math.acos(KT.clamp(ly / Math.hypot(lx, ly, lz), -1, 1));
            }
          }
        }
        view.th = orbit.cur.th + wrap(view.th - orbit.cur.th);
        view.r *= fitR;
        if (st.autoR != null && g.r !== st.autoR) st.userZoom = true;      // the user has zoomed: the distance is theirs now
        if (!st.userZoom) g.r = st.autoR = view.r;
        g.tx = tg[0]; g.ty = tg[1]; g.tz = tg[2];
        var busy = orbit.dragging() || !!dragCP;
        if (s.tan && pr && st.nudge && !busy) {
          camDir.copy(cam.position).sub(probe.position).normalize();
          var c = pr.n[0] * camDir.x + pr.n[1] * camDir.y + pr.n[2] * camDir.z, sg2 = c < 0 ? -1 : 1, ac = Math.abs(c);
          if (ac < EDGE) {                     // turn towards the side of the plane we are already on, and stop as soon as it reads
            var dth = wrap(Math.atan2(sg2 * pr.n[0], sg2 * pr.n[2]) - orbit.cur.th);
            var dph = KT.clamp(Math.acos(KT.clamp(sg2 * pr.n[1], -1, 1)), 0.12, 1.5) - orbit.cur.ph, kk;
            if (orbit.touched) {               // the user's view: aim a little ahead of where the camera is
              kk = 0.1 + 0.1 * (1 - ac / EDGE); g.th = orbit.cur.th + dth * kk; g.ph = KT.clamp(orbit.cur.ph + dph * kk, 0.12, 1.5);
            } else { kk = 1 - Math.exp(-1.1 * dt); st.nud.th += dth * kk; st.nud.ph += dph * kk; }
            st.nudT = 1.6;
          } else if (ac > EDGE * 1.25) { var dk = Math.exp(-0.35 * dt); st.nud.th *= dk; st.nud.ph *= dk; }
        }
        if (!orbit.touched) {                  // idle: sway, or rock in the scenes where the vectors must be read
          var rk = sc.rock || 0;
          g.th = view.th + (rk ? 0.36 * rk * Math.sin(t * 0.52) : 0.22 * Math.sin(t * 0.17)) + st.nud.th;
          g.ph = KT.clamp(view.ph + (rk ? 0.11 * rk * Math.sin(t * 0.37 + 1) : 0) + st.nud.ph, 0.12, 1.5);
        }
        if (st.nudT > 0) { st.nudT -= dt; hint.textContent = st.nudT > 0 ? T('tangent plane edge-on · stepping aside') : (sc.hint || ''); }
        orbit.update(dt);
        // on a small stage the insets cover the lower part: lift the picture into the free area above them
        var occ = s.face ? st.ins + 14 + (s.nbr ? 46 : 0) : 0, span = Math.max((s.uv ? 2 : 1) * (st.ins + 8) + 16, s.nbr ? 226 : 0) / W;
        st.lift += (0.5 * occ * KT.ease((span - 0.3) / 0.3) - st.lift) * (1 - Math.exp(-4 * dt));
        var vk = W + 'x' + H + ':' + Math.round(st.lift);
        if (vk !== st.viewKey) { st.viewKey = vk; cam.setViewOffset(W, H, 0, Math.round(st.lift), W, H); }
        var near = Math.max(0.2, orbit.cur.r - 1.3), far = orbit.cur.r + 5.2;
        scene.fog.near = near; scene.fog.far = far; sheet.uniforms.uFog.value.set(near, far + 1.5); ovMat.uniforms.uFog.value.set(near, far + 1.5);
        G.draw(R, scene, cam);
        if (s.tile || s.stitch || s.adapt) {      // what lies on the surface, over the Gaussians (see `ovl`)
          var rr = R.renderer, ac = rr.autoClear;
          rr.autoClear = false;
          if (field.mesh.visible) rr.clearDepth();
          G.draw(R, ovl, cam);
          rr.autoClear = ac;
        }

        if (s.probe && pr) {
          if (!s.nbr) {
            var q0 = G.project(cam, pr.c[0], pr.c[1], pr.c[2], W, H), ox = 0, oy = 1;
            if (q0 && s.tan) [pr.tipU, pr.tipV, s.nor && pr.tipN].forEach(function (w) {      // away from the arrows
              var q = w && G.project(cam, w[0], w[1], w[2], W, H), l; if (!q) return;
              l = Math.hypot(q[0] - q0[0], q[1] - q0[1]) || 1; ox -= (q[0] - q0[0]) / l; oy -= (q[1] - q0[1]) / l;
            });
            var ol = Math.hypot(ox, oy) || 1; place(tMu, pr.c, ox / ol * 22, oy / ol * 20);
          }
          if (s.tan) {
            var at = placeTip(tU, pr.c, pr.tipU, s.nbr ? 'S<sub>u</sub>Δu' : 'S<sub>u</sub>');
            placeTip(tV, pr.c, pr.tipV, s.nbr ? 'S<sub>v</sub>Δv' : 'S<sub>v</sub>', at);
          }
          if (s.nor) placeTip(tN, pr.c, pr.tipN, 'n');
        }
        if (s.uv && uvDirty) { drawUV(); uvDirty = false; }
        if (s.face && faceDirty) { drawFace(); faceDirty = false; }
        if (s.nbr) { st.sparkT -= dt; if (st.sparkT <= 0) { drawSpark(); st.sparkT = sc.wig ? 0.3 : 1e9; } }
        if (hudDirty) { hudDirty = false; drawHud(); }
      });
      /* Off screen the kit pauses the loop; the module then lets go of what only the stage on screen needs: the knob's
         big buffer (63 MB, and its copy on the GPU) and the coverage grid of scenes scale and fill (about 10 MB).
         Leaving chapter 03 by a jump (keys, the rail, the slide master hiding the next steps) changes no scene, so
         setScene alone would keep them for the rest of the talk. Both come back on demand: knTick re-evaluates the
         level it shows (kn.redo), tileUpdate measures again (tl.key). */
      var onStage = new IntersectionObserver(function (es) {
        if (es[es.length - 1].isIntersecting) return;
        kn.last = 0;                                   // the knob's clock: time off screen is not a frame
        if (big) { knBig(false); kn.redo = true; }
        if (cw.w) { cw.w = cntA = cntB = cw.net = null; cw.F = 0; tl.key = ''; }
      });
      onStage.observe(root);
      // anything that changes the estimate redraws the convergence plot (throttled while the surface is moving)
      function resetSpark() { if (st.sparkT > 1) st.sparkT = 0; }
      slN.input.addEventListener('input', resetSpark); uvc.cv.addEventListener('pointermove', resetSpark);
      uvc.cv.addEventListener('pointerdown', resetSpark); R.canvas.addEventListener('pointerup', resetSpark);

      if (!document.getElementById('kt-planoid-css')) {
        var css = document.createElement('style'); css.id = 'kt-planoid-css';
        css.textContent =
          '.kt-sim-planoid{--kt-ins:160px}' +
          '.kt-sim-planoid .kt-tag{position:absolute;left:0;top:0;z-index:3;pointer-events:none;font:600 15px/1 var(--f-mono);' +
          'padding:3px 6px;border-radius:5px;background:rgba(7,9,15,.72);will-change:transform;white-space:nowrap}' +
          '.kt-tag-text{color:var(--text)}' +
          '.kt-sim-planoid.kt-nbr .kt-tag{font-size:12px;padding:2px 5px;background:rgba(7,9,15,.6)}.kt-sim-planoid.kt-nbr .kt-tag sub{font-size:8px}' +
          '.kt-sim-planoid .kt-tag sub{font-size:10px}' +
          '.kt-sim-planoid .kt-tag i{font:500 10px/1 var(--f-mono);font-style:normal;letter-spacing:.04em;margin-left:6px;opacity:.85}' +
          '.kt-sim-planoid .kt-uv{position:absolute;z-index:5;left:12px;bottom:12px;width:var(--kt-ins);height:var(--kt-ins);' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;backdrop-filter:blur(10px)}' +
          '.kt-sim-planoid .kt-face{left:calc(20px + var(--kt-ins))}.kt-sim-planoid .kt-face.kt-solo{left:12px}' +
          '.kt-sim-planoid.kt-has-face .kt-hint{left:auto;right:14px;transform:none}' +
          '.kt-sim-planoid .kt-uv-cv{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}' +
          '.kt-sim-planoid .kt-uv-lab{position:absolute;z-index:1;left:6px;top:6px;pointer-events:none;font:600 10px/1 var(--f-mono);' +
          'letter-spacing:.12em;text-transform:uppercase;color:var(--dim);background:rgba(7,9,15,.82);padding:3px 5px;border-radius:4px;white-space:nowrap}' +
          '.kt-sim-planoid .kt-uv-lab i{font-style:normal;text-transform:none}' +
          '.kt-sim-planoid.kt-ins-small .kt-wide{display:none}.kt-sim-planoid.kt-ins-small .kt-uv-lab{letter-spacing:.04em;font-size:9px}' +
          '.kt-sim-planoid .kt-nbsw{position:absolute;z-index:5;left:12px;bottom:calc(20px + var(--kt-ins));padding:4px 4px 4px 10px;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px)}' +
          '.kt-sim-planoid .kt-nbsw .kt-seg-btn.on{background:var(--splat)}' +
          /* (a plot in the kit's panel: left to right, also where the panel runs right to left) */
          '.kt-sim-planoid .kt-pl-spark{position:relative;height:92px;margin-top:2px;border-top:1px solid var(--line);direction:ltr}' +
          '.kt-sim-planoid .kt-pl-spark-cv{position:absolute;inset:0;width:100%;height:100%}' +
          '.kt-sim-planoid .kt-readout .kt-lab sub,.kt-sim-planoid .kt-readout .kt-lab sup{font-size:8px;line-height:0}' +
          '.kt-sim-planoid .kt-readout .kt-lab i,.kt-sim-planoid .kt-readout .kt-val i{font-style:normal;color:var(--mute)}' +
          '.kt-sim-planoid .kt-pl-sz .kt-seg-btn.on{background:var(--splat)}' +
          '.kt-sim-planoid .kt-pl-hud{position:absolute;z-index:4;left:7px;top:34px;max-width:min(64%,620px);pointer-events:none;' +
          'padding:2px 7px 3px;border-radius:6px;background:rgba(7,9,15,.55);' +
          'font:500 11px/1.6 var(--f-mono);letter-spacing:.03em;color:var(--dim);text-shadow:0 1px 3px rgba(7,9,15,.9)}' +
          '.kt-sim-planoid .kt-pl-hud b{font-weight:600;color:var(--text)}.kt-sim-planoid .kt-pl-hud .m{color:var(--mute)}' +
          '.kt-sim-planoid .kt-pl-hud .s{color:var(--splat)}.kt-sim-planoid .kt-pl-hud .c{color:var(--ctrl)}' +
          '.kt-sim-planoid .kt-pl-hud .e{color:var(--err)}.kt-sim-planoid .kt-pl-hud .u{color:var(--surf)}' +
          '.kt-sim-planoid .kt-pl-hud .big{font:600 21px/1.25 var(--f-mono);letter-spacing:0}' +
          '.kt-sim-planoid.kt-compact .kt-pl-hud,.kt-sim-planoid.kt-ins-small .kt-pl-hud{left:3px;font-size:10px;line-height:1.5;max-width:calc(100% - 20px)}' +
          '.kt-sim-planoid.kt-compact .kt-pl-hud .big,.kt-sim-planoid.kt-ins-small .kt-pl-hud .big{font-size:16px}' +
          /* a light page (the talk): the dark chips become the kit's glass, the type the page's (Inter); colours stay tokens */
          '[data-theme="light"] .kt-sim-planoid .kt-tag{font-family:var(--f-ui);background:rgba(255,255,255,.74);' +
          'box-shadow:var(--glass-edge),0 4px 14px -6px rgba(88,56,24,.32);border-radius:7px}' +
          '[data-theme="light"] .kt-sim-planoid.kt-nbr .kt-tag{background:rgba(255,255,255,.68)}' +
          '[data-theme="light"] .kt-sim-planoid .kt-tag i{font-family:var(--f-ui);color:var(--dim)}' +
          '[data-theme="light"] .kt-sim-planoid .kt-uv{border:0;border-radius:14px;background:rgba(255,255,255,.6);' +
          'backdrop-filter:blur(22px) saturate(180%);-webkit-backdrop-filter:blur(22px) saturate(180%);' +
          'box-shadow:var(--glass-edge),0 14px 34px -18px rgba(88,56,24,.35)}' +
          '[data-theme="light"] .kt-sim-planoid .kt-uv-lab{font:600 9.5px/1 var(--f-ui);letter-spacing:.08em;color:var(--dim);' +
          'background:rgba(255,255,255,.8);box-shadow:var(--glass-edge)}' +
          '[data-theme="light"] .kt-sim-planoid .kt-nbsw{border:0;border-radius:999px;background:rgba(255,255,255,.62);' +
          'backdrop-filter:blur(22px) saturate(180%);-webkit-backdrop-filter:blur(22px) saturate(180%);' +
          'box-shadow:var(--glass-edge),0 10px 26px -16px rgba(88,56,24,.35);padding:4px 4px 4px 14px}' +
          '[data-theme="light"] .kt-sim-planoid .kt-nbsw .kt-seg-btn.on,[data-theme="light"] .kt-sim-planoid .kt-pl-sz .kt-seg-btn.on{color:#fff}' +
          '[data-theme="light"] .kt-sim-planoid .kt-pl-hud{font-family:var(--f-ui);letter-spacing:0;font-variant-numeric:tabular-nums;' +
          'padding:4px 10px 5px;border-radius:12px;background:rgba(255,255,255,.66);text-shadow:none;' +
          'backdrop-filter:blur(18px) saturate(170%);-webkit-backdrop-filter:blur(18px) saturate(170%);' +
          'box-shadow:var(--glass-edge),0 10px 26px -16px rgba(88,56,24,.3)}' +
          '[data-theme="light"] .kt-sim-planoid .kt-pl-hud .big{font-family:var(--f-ui);font-weight:700;letter-spacing:-.01em}' +
          /* a right-to-left page (talk.html in Hebrew; the same mark as kit.css asks for): this module's own labels are set
             without the tracking of the capitals they replace; and the kit lays its row the other way round (label on the
             right), so the box around the 4 / 8 switch keeps its wider padding on the label's side */
          ':root[dir="rtl"][data-lang] .kt-sim-planoid .kt-uv-lab,:root[dir="rtl"][data-lang] .kt-sim-planoid .kt-pl-hud,' +
          ':root[dir="rtl"][data-lang] .kt-sim-planoid .kt-tag i{letter-spacing:0}' +
          ':root[dir="rtl"][data-lang] .kt-sim-planoid .kt-nbsw{padding:4px 10px 4px 4px}' +
          ':root[dir="rtl"][data-lang][data-theme="light"] .kt-sim-planoid .kt-nbsw{padding:4px 14px 4px 4px}';
        document.head.appendChild(css);
      }

      setScene(env.opts.scene || 'net');
      return {
        setScene: setScene, orbit: orbit, camera: cam, patch: patch,      // camera + net: for tests and host pages
        state: st,                                                        // st.nudge = false keeps an edge-on view (tests)
        /* the numbers on screen: { n, nb, u, v, sigmaNeighbours, sigmaDerivatives, gap, shear, angle, edge } */
        probe: function () {
          if (!pr) return null;
          camDir.copy(cam.position).sub(probe.position).normalize();
          return { n: st.n, nb: st.nb, u: pr.u, v: pr.v, sigmaNeighbours: pr.est.s.map(function (x) { return x * st.gain; }),
                   sigmaDerivatives: pr.ref.s.map(function (x) { return x * st.gain; }), gap: pr.gap, gap4: pr.gap4, gap8: pr.gap8,
                   shear: pr.shear, angle: pr.ang, edge: Math.abs(pr.n[0] * camDir.x + pr.n[1] * camDir.y + pr.n[2] * camDir.z) };
        },
        setNeighbours: function (k) { st.nb = k === 8 ? 8 : 4; segNb.set(st.nb); probeDirty = uvDirty = latDirty = true; st.sparkT = 0; },
        /* scenes scale / fill: 'one' or 'own' pins the size (stops the alternation); the measured coverage */
        setSize: function (m) { tl.auto = false; tl.goal = m === 'one' ? 1 : 0; segSz.set(m === 'one' ? 'one' : 'own'); },
        tiling: function () { return tl.own && { n: st.n, k: st.gain, blend: tl.b, sigmaOne: tl.s1, own: tl.own, one: tl.one, F: cw.F }; },
        fit: function () { return st.fit && { step: st.fit.it, done: st.fit.done, rms: fitStat.rms, max: fitStat.max, rmsConverged: FIT.rms / HS, steps: FIT.steps }; },
        stitch: function () { return sx.n && { n: sx.n, triangles: sx.q * 2, of: sx.Q * 2, gaussiansLeft: sx.V - sx.passed }; },
        knob: function () { return { n: kn.cur, gaussians: kn.cur * kn.cur, frame: Object.assign({}, kn.ft), evaluate: Object.assign({}, kn.ev), limit: kn.lim, big: !!big }; },
        adaptive: function () { return ad.stat; },
        resize: function (w, h) {
          W = w; H = h; R.resize(w, h); uvDirty = faceDirty = true; st.sparkT = Math.min(st.sparkT, 0.05);
          fitR = Math.max(1, 1.32 / (w / h));               // back off on narrow stages so the patch stays in frame
          var ins = Math.round(KT.clamp(Math.min(w * 0.22, h * 0.3), 108, 200));
          st.ins = ins; root.style.setProperty('--kt-ins', ins + 'px'); root.classList.toggle('kt-ins-small', ins < 150);
          // a narrow stage folds the panel behind the kit's `controls` button — on the lab page too, not only in the deck
          if (!env.embed) root.classList.toggle('kt-compact', w < 560);
          hudPlace(); if (st.sc.sweep) kn.plotT = 0;
        },
        destroy: function () { onStage.disconnect(); R.renderer.dispose(); }
      };
    }
  });
})();
