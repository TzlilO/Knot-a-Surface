/* Knot-a-Surface talk — module `planoid`: "Calculus makes the splat".
   One B-spline patch. Pick a point (u,v) and watch a Gaussian fall out of the surface's own
   calculus: position = S(u,v), orientation = tangent frame, scale = |S_u| du x |S_v| dv.
   Then fill the chart and turn the resolution knob — the learnable parameters never change.

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
  var NU = 6, NV = 6, SPAN = 3.2, PER_CP = 16;      // 3 position + 12 SH1 colour + 1 opacity
  var NMAX = 160, FLOOR = -1.15;
  var VEC = 0.26;            // a tangent arrow is drawn at this fraction of its true length
  var LATMAX = 40;           // up to this many samples per axis the whole sampling lattice is drawn
  var EDGE = 0.24;           // |cos| between the view ray and the normal under which the tangent plane counts as edge-on
  var COL_U = C.surf, COL_V = '#6d8dff';            // two shades of the surface colour: S_u and S_v must not look alike

  var SCENES = {
    net:      { show: 'net wire', cap: '<b>tensor product</b> · a net of control points sweeps out a surface',
                hint: 'drag to orbit · drag an amber point to reshape' },
    probe:    { show: 'net wire probe uv', cap: '<b>position</b> · μ = S(u, v)', hint: 'drag the cross in the (u,v) chart' },
    tangents: { show: 'wire probe tan iso plane face uv', rock: 1,
                cap: '<b>tangents</b> · S<sub>u</sub> and S<sub>v</sub>, in closed form — two directions that span the tangent plane' },
    normal:   { show: 'wire probe tan nor iso plane face uv', rock: 1,
                cap: '<b>rotation</b> · R = [ Ŝ<sub>u</sub> · Ŝ<sub>v</sub><sup>⊥</sup> · n ]' },
    scale:    { show: 'probe tan nor ell nbr face uv', n: 9, rock: 0.55, fixed: [0.28, 0.5],
                cap: '<b>scale</b> · s = ( ‖S<sub>u</sub>‖Δu , ‖S<sub>v</sub><sup>⊥</sup>‖Δv , ε ) — the step to the neighbouring samples',
                hint: 'drag in the (u,v) chart · switch 4 / 8 neighbours · take the samples slider' },
    derive:   { show: 'net probe tan nor ell nbr face', n: 9, wig: 1, fixed: [0.56, 0.44], rock: 0.4,
                cap: '<b>nothing is learned per Gaussian</b> · move the surface, the splat and its neighbours re-derive' },
    fill:     { show: 'wire field uv', n: 14, rim: 1, cap: '<b>fill the chart</b> · one Gaussian per (u,v) sample, sized to its cell' },
    knob:     { show: 'field', sweep: 1, rim: 0.7, cap: '<b>the resolution knob</b> · re-evaluate, don’t retrain' },
    features: { show: 'net field feat', n: 72, rim: 0.4, cap: '<b>control features</b> · colour and opacity ride the same basis' },
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
      { id: 'fill', label: 'Fill the chart' }, { id: 'knob', label: 'Resolution knob' },
      { id: 'features', label: 'Control features' }
    ],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(34, 1, 0.1, 60);
      var R = G.renderer(root, cam), scene = new T.Scene();
      var W = 1, H = 1, fitR = 1;

      /* light and air: the same light direction as the splat shader, and a fog that follows the camera */
      var LIGHT = new T.Vector3(0.35, 0.86, 0.36).normalize();
      scene.fog = new T.Fog(0x07090f, 5, 14);
      scene.add(new T.HemisphereLight(0xe6efff, 0x2a3650, 1.5));
      var sun = new T.DirectionalLight(0xffffff, 2.6); sun.position.copy(LIGHT); scene.add(sun);

      /* ---------- the patch ---------- */
      var patch = NB.makePatch({ nu: NU, nv: NV, p: 3 }), base = new Float32Array(NU * NV * 3);
      (function () {
        var A = [0.12, 0.47, 0.90], B = [1.0, 0.44, 0.26], D = [0.80, 0.58, 0.85], i, j, o, x, z, s, t;
        for (j = 0; j < NV; j++) for (i = 0; i < NU; i++) {
          o = (j * NU + i) * 3; s = i / (NU - 1); t = j / (NV - 1); x = s * 2 - 1; z = t * 2 - 1;
          base[o] = x * SPAN / 2; base[o + 2] = z * SPAN / 2;
          base[o + 1] = 1.0 * Math.exp(-((x - 0.25) * (x - 0.25) + (z + 0.2) * (z + 0.2)) * 2.4) -
                        0.55 * Math.exp(-((x + 0.6) * (x + 0.6) + (z - 0.55) * (z - 0.55)) * 3.0) + 0.14 * x;
          for (var c = 0; c < 3; c++) patch.C[o + c] = KT.lerp(KT.lerp(A[c], B[c], s), D[c], t * 0.75);
        }
        o = (2 * NU + 3) * 3; patch.C[o] = 1; patch.C[o + 1] = 0.86; patch.C[o + 2] = 0.30;   // one loud control feature
        patch.P.set(base);
      })();

      /* ---------- scene objects ----------
         Draw order: opaque things (control points, arrows, splat field) first, then by renderOrder:
         floor glow -6, shadow -4, sheet 0, lines on it 1-2, "hidden" copies 3+, Gaussians 9-10. */
      var a, b;
      function own(o, order) { o.frustumCulled = false; if (order != null) o.renderOrder = order; scene.add(o); return o; }
      function ghostLines(l, opacity, order) {              // the part of a line set that lies behind something
        return own(new T.LineSegments(l.obj.geometry, G.hidden(l.mat.color, opacity, 'line')), order);
      }

      var grid = new T.GridHelper(9, 18, 0x243652, 0x162238); grid.position.y = FLOOR; scene.add(grid);
      var glow = own(new T.Mesh(new T.CircleGeometry(5.4, 48).rotateX(-Math.PI / 2), new T.ShaderMaterial({
        transparent: true, depthWrite: false,
        vertexShader: 'varying vec2 vP; void main(){ vP = position.xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
        fragmentShader: 'varying vec2 vP; void main(){ float a = 1.0 - clamp(length(vP) / 5.4, 0.0, 1.0); gl_FragColor = vec4(0.20, 0.33, 0.52, a * a * 0.30); }'
      })), -6);
      glow.position.y = FLOOR - 0.004;

      var FN = 33, sheet = G.sheet(FN, FN), fill = own(sheet.mesh, 0);
      var sg = { pos: sheet.pos, su: new Float32Array(FN * FN * 3), sv: new Float32Array(FN * FN * 3) };
      // the sheet's shadow: the same geometry flattened onto the floor along the light, three times for a soft edge
      var shadowMat = new T.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.2, depthWrite: false, side: T.DoubleSide, fog: false });
      var shadows = [0.93, 1.0, 1.08].map(function (k, i) {
        var m = own(new T.Mesh(sheet.geo, shadowMat), -4), sx = -LIGHT.x / LIGHT.y, sz = -LIGHT.z / LIGHT.y;
        m.matrixAutoUpdate = false; m.matrixWorldNeedsUpdate = true;
        m.matrix.set(k, sx, 0, -sx * FLOOR,   0, 1e-4, 0, FLOOR + 0.004 + 0.002 * i,   0, sz, k, -sz * FLOOR,   0, 0, 0, 1);
        return m;
      });

      var WL = 13, WS = 40, wire = G.lines(2 * WL * WS * 2, C.surf, 0.55); own(wire.obj, 1);
      var net = G.lines(((NU - 1) * NV + NU * (NV - 1)) * 2, C.ctrl, 0.7); own(net.obj, 2);
      var netGhost = ghostLines(net, 0.2, 3);
      var balls = [], ballGeo = new T.SphereGeometry(0.06, 20, 14), ballHid = G.hidden(C.ctrl, 0.3);
      for (a = 0; a < NU * NV; a++) {
        var m = new T.Mesh(ballGeo, new T.MeshLambertMaterial({ color: C.ctrl })); m.userData.i = a;
        var mh = new T.Mesh(ballGeo, ballHid); mh.renderOrder = 3; m.add(mh);
        balls.push(m); scene.add(m);
      }

      var field = G.splats(NMAX * NMAX); scene.add(field.mesh);
      var violet = G.rgb(C.splat);

      // the probe and its frame
      var probe = new T.Mesh(new T.SphereGeometry(0.05, 20, 14), new T.MeshLambertMaterial({ color: 0xffffff, emissive: 0x6a6a6a }));
      var probeHid = new T.Mesh(probe.geometry, G.hidden(0xffffff, 0.4)); probeHid.renderOrder = 8; probe.add(probeHid); scene.add(probe);
      var aU = G.vector(COL_U), aV = G.vector(COL_V), aN = G.vector(0xffffff);
      scene.add(aU.obj); scene.add(aV.obj); scene.add(aN.obj);
      var IS = 48, isoU = G.lines(IS * 2, COL_U, 0.95), isoV = G.lines(IS * 2, COL_V, 0.95); own(isoU.obj, 2); own(isoV.obj, 2);
      var ig = { pos: new Float32Array((IS + 1) * 3) }, isoT = lin(IS + 1);

      // the tangent plane: a faint patch of it, and the parallelogram the two tangents span
      var plPos = new Float32Array(8 * 3), plGeo = new T.BufferGeometry();
      plGeo.setAttribute('position', new T.BufferAttribute(plPos, 3)); plGeo.setIndex([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
      plGeo.addGroup(0, 6, 0); plGeo.addGroup(6, 6, 1);
      function flat(color, opacity) { return new T.MeshBasicMaterial({ color: color, transparent: true, opacity: opacity, side: T.DoubleSide, depthWrite: false }); }
      var plane = own(new T.Mesh(plGeo, [flat(C.surf, 0.09), flat(C.surf, 0.22)]), 4);
      var planeHid = own(new T.Mesh(plGeo, G.hidden(C.surf, 0.045, 'flat')), 5);
      var plGrid = G.lines(40, C.surf, 0.26), plEdge = G.lines(8, '#d7f1ff', 0.85), mark = G.lines(8, '#ffffff', 0.9);
      own(plGrid.obj, 5); own(plEdge.obj, 6); own(mark.obj, 6);
      var plEdgeHid = ghostLines(plEdge, 0.25, 7);

      // the Gaussian at the probe is painted on the sheet (sheet.gauss); these are the 1-sigma outlines of the
      // estimates, flat in the tangent plane (ribbons: a 1 px line would vanish)
      var EL = 72;
      function ribbon(color, opacity, order) {
        var pos = new Float32Array((EL + 1) * 2 * 3), g = new T.BufferGeometry(), ix = [], i;
        for (i = 0; i < EL; i++) ix.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
        g.setAttribute('position', new T.BufferAttribute(pos, 3)); g.setIndex(ix);
        var mesh = own(new T.Mesh(g, flat(color, opacity)), order), hid = own(new T.Mesh(g, G.hidden(color, opacity * 0.3, 'flat')), order);
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
      var ringN = ribbon(C.splat, 0.95, 11), ringA = ribbon('#ffffff', 0.95, 12);

      // the neighbourhood: the lattice on the surface, the 3 x 3 block, the eight neighbours and their Gaussians
      var LS = 36, lat = G.lines(2 * (LATMAX + 1) * LS * 2, C.surf, 0.3); own(lat.obj, 1);
      var lg = { pos: new Float32Array((LS + 1) * (LATMAX + 1) * 3) };
      var dotTex = (function () {
        var cv = document.createElement('canvas'); cv.width = cv.height = 32;
        var g = cv.getContext('2d'); g.fillStyle = '#fff'; g.beginPath(); g.arc(16, 16, 14, 0, 6.2832); g.fill();
        return new T.CanvasTexture(cv);
      })();
      var dotPos = new Float32Array(LATMAX * LATMAX * 3), dotGeo = new T.BufferGeometry();
      dotGeo.setAttribute('position', new T.BufferAttribute(dotPos, 3)); dotGeo.setDrawRange(0, 0);
      var dots = own(new T.Points(dotGeo, new T.PointsMaterial({ color: C.splat, size: 0.05, map: dotTex, alphaTest: 0.5, transparent: true, opacity: 0.75 })), 2);
      var BS = 4, BN = 3 * BS + 1, bgd = { pos: new Float32Array(BN * BN * 3) };
      var blk = G.lines(8 * (BN - 1) * 2, '#d7f1ff', 0.8); own(blk.obj, 2);
      var nbGeo = new T.SphereGeometry(0.036, 16, 12), cs = new T.Color(C.splat);
      var nbOn = new T.MeshLambertMaterial({ color: cs, emissive: cs.clone().multiplyScalar(0.35) });
      var nbOff = new T.MeshLambertMaterial({ color: cs.clone().multiplyScalar(0.42), emissive: cs.clone().multiplyScalar(0.1) });
      var nbHid = G.hidden(C.splat, 0.3), nbDot = [];
      for (a = 0; a < 8; a++) {
        var nd = new T.Mesh(nbGeo, nbOn), nh = new T.Mesh(nbGeo, nbHid); nh.renderOrder = 8; nd.add(nh); nbDot.push(nd); scene.add(nd);
      }
      var NR = 40, nbRingA = G.lines(8 * NR * 2, C.splat, 0.5), nbRingI = G.lines(4 * NR * 2, C.splat, 0.18);
      own(nbRingA.obj, 11); own(nbRingI.obj, 11);
      var span = G.lines(16, '#f6e9fa', 0.95), err = G.lines(4, C.err, 1); own(span.obj, 12); own(err.obj, 13);
      var spanHid = ghostLines(span, 0.3, 13);
      var G9 = { pos: new Float32Array(27), su: new Float32Array(27), sv: new Float32Array(27) };
      var G9b = { pos: new Float32Array(27), su: new Float32Array(27), sv: new Float32Array(27) };

      // floating labels
      function tag(html, tone, color) {
        var e = ui.el('div', 'kt-tag kt-tag-' + tone, html); if (color) e.style.color = color;
        root.appendChild(e); return e;
      }
      var tMu = tag('μ', 'text'), tU = tag('', 'vec', COL_U), tV = tag('', 'vec', COL_V), tN = tag('n', 'vec', '#ffffff');
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);

      // (u,v) chart, the tangent plane seen face-on, and the 4 / 8 switch above them
      var uvBox = ui.el('div', 'kt-uv'); root.appendChild(uvBox);
      var uvLab = ui.el('span', 'kt-uv-lab', '(u, v) chart'); uvBox.appendChild(uvLab);
      var uvc = ui.canvas(uvBox, 'kt-uv-cv');
      var fcBox = ui.el('div', 'kt-uv kt-face'); root.appendChild(fcBox);
      fcBox.appendChild(ui.el('span', 'kt-uv-lab', 'tangent plane<span class="kt-wide"> · face-on</span>'));
      var fc = ui.canvas(fcBox, 'kt-uv-cv'); fc.cv.style.cursor = 'default';
      var nbBar = ui.el('div', 'kt-nbsw'); root.appendChild(nbBar);

      /* ---------- controls ---------- */
      var st = { scene: 'net', sc: SCENES.net, show: {}, n: 14, gain: 0.75, u: 0.5, v: 0.5, t: 0, nb: 4, ci: 0, cj: 0,
                 touchedProbe: false, touchedN: false, wigI: -1,
                 nudge: true, nud: { th: 0, ph: 0 }, nudT: 0, autoR: null, userZoom: false, sparkT: 0, ins: 160, lift: 0, viewKey: '' };
      var panel = ui.panel(root, { pos: 'tr', title: 'sampling' });
      var slN = ui.slider(panel, { label: 'samples', min: 4, max: NMAX, step: 1, value: st.n,
        fmt: function (v) { return v + '²'; },
        onInput: function (v) { st.n = v; st.touchedN = true; fieldDirty = true; uvDirty = true; latDirty = probeDirty = true; } });
      var slG = ui.slider(panel, { label: 'σ / cell', min: 0.3, max: 1.1, step: 0.01, value: st.gain,
        fmt: function (v) { return v.toFixed(2); }, onInput: function (v) { st.gain = v; fieldDirty = true; probeDirty = true; } });
      var rdG = ui.readout(panel, { label: 'Gaussians', tone: 'splat' });
      var rdP = ui.readout(panel, { label: 'learnable params', tone: 'ctrl' });
      var rdZ = ui.readout(panel, { label: 'params / Gaussian', tone: 'ok' });
      rdP.set((NU * NV * PER_CP).toLocaleString('en-US')); rdZ.set('0');
      var rdAng = ui.readout(panel, { label: '∠ ( S<sub>u</sub> , S<sub>v</sub> )', tone: 'surf' });
      var rdSn = ui.readout(panel, { label: 'σ from 4 neighbours', tone: 'splat' });
      var rdSa = ui.readout(panel, { label: 'σ from derivatives' });
      var rdGap = ui.readout(panel, { label: 'they differ by', tone: 'ok' });
      var rdSh = ui.readout(panel, { label: 'shear the two lengths drop', tone: 'err' });
      var spBox = ui.el('div', 'kt-pl-spark'); panel.appendChild(spBox);
      var spc = ui.canvas(spBox, 'kt-pl-spark-cv');
      var segNb = ui.seg(nbBar, { label: 'size from', value: st.nb,
        options: [{ v: 4, label: '4 neighbours' }, { v: 8, label: '8' }],
        onChange: function (v) { st.nb = v; probeDirty = uvDirty = latDirty = true; st.sparkT = 0; } });

      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, theta: 0.72, phi: 1.0, radius: 7.2, target: [0, 0.05, 0],
        minRadius: 1.2, maxRadius: 20, pick: pickBall, hit: function (x, y) { return findBall(x, y, true) >= 0; },
        release: function () { if (!dragCP) return; if (dragCP.moved) { patch.P.set(dragCP.was, dragCP.i * 3); base.set(patch.P); surfDirty = fieldDirty = true; } dragCP = null; } });
      var surfDirty = true, fieldDirty = true, uvDirty = true, latDirty = true, probeDirty = true, faceDirty = true;

      /* ---------- drag a control point ---------- */
      var ray = new T.Raycaster(), ndc = new T.Vector2(), dragPlane = new T.Plane(), hitP = new T.Vector3(), dragCP = null;
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
        dragPlane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(new T.Vector3()), balls[i].position);
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
      function updateField() {
        var n = st.n, us = new Float32Array(n), d = new Float32Array(n), i;
        for (i = 0; i < n; i++) { us[i] = (i + 0.5) / n; d[i] = 1 / n; }
        NB.patchGrid(patch, us, us, fg);
        G.writePlanoids(field, 0, fg, n, n, d, d, st.gain, st.show.feat ? null : violet);
        field.commit(n * n); st.fieldN = n;
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
        if (s.ell) {                       // the planoid itself, and its 1-sigma outline
          sheet.gauss(4, p, mul(ref.ax[0], k), mul(ref.ax[1], k), violet, 0.9);
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
            sheet.gauss(j, c0, sx, sy, violet, on ? 0.42 : 0.13);
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
          label(rdSn, 'σ from ' + st.nb + ' neighbours');
          label(rdSa, eight ? 'σ from J·diag(Δu,Δv)' : 'σ from ‖S<sub>u</sub>‖Δu , ‖S<sub>v</sub><sup>⊥</sup>‖Δv');
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
      var camDir = new T.Vector3();
      function placeTip(el, from, tip, name, clear) {
        var qa = G.project(cam, from[0], from[1], from[2], W, H), qb = G.project(cam, tip[0], tip[1], tip[2], W, H);
        if (!qa || !qb) { el.style.opacity = 0; return null; }
        var dx = qb[0] - qa[0], dy = qb[1] - qa[1], L = Math.hypot(dx, dy);
        if (L < 12) { dx = 0.62; dy = -0.78; } else { dx /= L; dy /= L; }
        var d = add(tip, from, -1), l = len(d) || 1;
        camDir.copy(cam.position).sub(probe.position).normalize();
        var z = (d[0] * camDir.x + d[1] * camDir.y + d[2] * camDir.z) / l;
        var html = name + (z > 0.9 ? '<i>⊙ towards you</i>' : z < -0.9 ? '<i>⊗ away</i>' : '');
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
        g.strokeStyle = 'rgba(255,167,38,.38)'; g.lineWidth = 1;            // knot lines = patch seams of the basis
        var inner = NU - 3;
        for (i = 1; i < inner; i++) {
          g.beginPath(); g.moveTo(i / inner * w, 0); g.lineTo(i / inner * w, h); g.moveTo(0, i / inner * h); g.lineTo(w, i / inner * h); g.stroke();
        }
        function cells(x0, y0, cw, ch, ci, cj) {          // the cell (ci, cj) and its neighbourhood, cell size cw x ch at (x0, y0) for cell (0, 0)
          for (b2 = -1; b2 <= 1; b2++) for (a2 = -1; a2 <= 1; a2++) {
            var x = x0 + (ci + a2) * cw, y = y0 - (cj + b2 + 1) * ch, mid = !a2 && !b2, on = mid || st.nb === 8 || !a2 || !b2;
            g.fillStyle = mid ? 'rgba(255,255,255,.92)' : on ? 'rgba(206,147,216,.62)' : 'rgba(206,147,216,.10)';
            g.fillRect(x + 0.5, y + 0.5, cw - 1, ch - 1);
            if (!on) { g.strokeStyle = 'rgba(206,147,216,.5)'; g.lineWidth = 1; g.setLineDash([2, 2]); g.strokeRect(x + 1, y + 1, cw - 2, ch - 2); g.setLineDash([]); }
          }
        }
        if (s.nbr && n <= LATMAX) {                        // the lattice of cells
          g.strokeStyle = 'rgba(206,147,216,.22)'; g.lineWidth = 1; g.beginPath();
          for (i = 1; i < n; i++) { g.moveTo(i / n * w, 0); g.lineTo(i / n * w, h); g.moveTo(0, i / n * h); g.lineTo(w, i / n * h); }
          g.stroke();
        }
        if (s.nbr) cells(0, h, w / n, h / n, st.ci, st.cj);
        if (s.field || s.ell) {
          g.fillStyle = C.splat;
          if (n <= 44) { var r = Math.max(1, Math.min(3, w / n * 0.22));
            for (b2 = 0; b2 < n; b2++) for (a2 = 0; a2 < n; a2++) { g.beginPath(); g.arc((a2 + 0.5) / n * w, (b2 + 0.5) / n * h, r, 0, 6.2832); g.fill(); } }
          else { g.globalAlpha = Math.min(0.75, 0.25 + n / 320) * (s.nbr ? 0.35 : 1); g.fillRect(0, 0, w, h); g.globalAlpha = 1; }
        }
        if (s.probe) {
          var pu = s.nbr ? (st.ci + 0.5) / n : st.u, pv = s.nbr ? (st.cj + 0.5) / n : st.v, x = pu * w, y = (1 - pv) * h;
          if (s.ell && !s.nbr) { g.strokeStyle = C.splat; g.lineWidth = 1.5; g.strokeRect(x - w / n / 2, y - h / n / 2, w / n, h / n); }
          g.strokeStyle = s.nbr ? 'rgba(221,229,240,.45)' : C.text; g.lineWidth = 1.2;
          g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
          g.fillStyle = s.nbr ? C.bg : C.text; g.beginPath(); g.arc(x, y, s.nbr ? Math.min(3, w / n * 0.3) : 4, 0, 6.2832); g.fill();
          if (s.nbr && w / n < 11) {                       // cells too small to read: a magnified copy of the 5 x 5 window
            var L = Math.round(w * 0.5), lx = pu < 0.5 ? w - L - 5 : 5, ly = pv < 0.5 ? 20 : h - L - 5, cs2 = L / 5;
            g.strokeStyle = 'rgba(221,229,240,.5)'; g.lineWidth = 1;
            g.strokeRect(x - 2.5 * w / n, y - 2.5 * h / n, 5 * w / n, 5 * h / n);
            g.beginPath(); g.moveTo(x, y); g.lineTo(lx + L / 2, ly + L / 2); g.stroke();
            g.fillStyle = 'rgba(11,16,26,.96)'; g.fillRect(lx, ly, L, L); g.strokeRect(lx + 0.5, ly + 0.5, L - 1, L - 1);
            g.save(); g.beginPath(); g.rect(lx, ly, L, L); g.clip();
            cells(lx + (2 - st.ci) * cs2, ly + (st.cj + 3) * cs2, cs2, cs2, st.ci, st.cj);
            g.restore();
          }
        }
        uvLab.textContent = !s.nbr ? '(u, v) chart' : w < 150 ? '(u, v) · ' + st.nb + ' nbrs' : '(u, v) · ' + st.nb + ' neighbours';
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
          g.fillStyle = 'rgba(79,195,247,.13)'; g.strokeStyle = 'rgba(215,241,255,.5)'; g.lineWidth = 1;
          g.beginPath(); pts = [pts[0], su, add(su, sv), sv];
          pts.forEach(function (p, i) { if (i) g.lineTo(X(p), Y(p)); else g.moveTo(X(p), Y(p)); });
          g.closePath(); g.fill(); g.stroke();
          var aV2 = Math.atan2(Y(sv) - cy, X(sv) - cx), rr = small ? 13 : 17;
          if (s.nor) {                                   // the axis the rotation actually uses: S_v made perpendicular to S_u
            g.setLineDash([3, 3]); g.strokeStyle = COL_V; g.lineWidth = 1.2; line([0, 0, 0], m.svp); g.setLineDash([]);
            var sgn = Y(m.svp) > cy ? 1 : -1;
            g.strokeStyle = 'rgba(221,229,240,.8)'; g.lineWidth = 1; g.strokeRect(cx, sgn > 0 ? cy : cy - 7, 7, 7);
            if (!small && Math.abs(m.ang - 90) > 3) sub('S', 'v⊥', X(m.svp) + (m.ang > 90 ? 16 : -18), Y(m.svp) - sgn * 12, 'rgba(109,141,255,.85)');
          }
          g.strokeStyle = 'rgba(221,229,240,.9)'; g.lineWidth = 1.2; g.beginPath(); g.arc(cx, cy, rr, 0, aV2, aV2 < 0); g.stroke();
          arrow(su, COL_U, 2.2); arrow(sv, COL_V, 2.2);
          sub('S', 'u', X(su) + 9, Y(su) + (aV2 < 0 ? 9 : -9), COL_U); sub('S', 'v', X(sv) + 11, Y(sv), COL_V);
          g.fillStyle = C.text; g.beginPath(); g.arc(cx, cy, 2.6, 0, 6.2832); g.fill();
          if (s.nor) {                                   // n comes straight out of the picture
            g.strokeStyle = '#fff'; g.lineWidth = 1.4; g.beginPath(); g.arc(cx, cy, 5.5, 0, 6.2832); g.stroke();
            if (!small) sub('n', '', cx - 10, cy + (aV2 < 0 ? 12 : -12), '#fff');
          }
          foot((small ? '∠ ' : '∠(Su, Sv) = ') + m.ang.toFixed(1) + '°', C.text);
          return;
        }
        // the neighbourhood, to scale: the lattice the Jacobian predicts, the real neighbours, the estimates
        var D = m.D, A = mul(su, D), B = mul(sv, D), k = st.gain, ext = 0, d;
        for (j = 0; j < 9; j++) ext = Math.max(ext, Math.abs(dot(m.d[j], m.e1)), Math.abs(dot(m.d[j], m.e2)));
        sc = Math.min(w - 12, h - top - bot) / 2 / (ext * 1.12); cx = w / 2; cy = top + (h - top - bot) / 2;
        g.fillStyle = 'rgba(206,147,216,.16)'; g.beginPath();
        [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].forEach(function (q, i) {
          d = add(mul(A, q[0]), B, q[1]); if (i) g.lineTo(X(d), Y(d)); else g.moveTo(X(d), Y(d));
        });
        g.closePath(); g.fill();
        g.strokeStyle = 'rgba(79,195,247,.32)'; g.lineWidth = 1;
        for (i = -1.5; i <= 1.5; i++) { line(add(mul(A, i), B, -1.5), add(mul(A, i), B, 1.5)); line(add(mul(B, i), A, -1.5), add(mul(B, i), A, 1.5)); }
        g.strokeStyle = 'rgba(246,233,250,.7)'; g.lineWidth = 1;
        if (st.nb === 4) { line(m.d[3], m.d[5]); line(m.d[1], m.d[7]); }
        else for (j = 0; j < 9; j++) if (j !== 4) line([0, 0, 0], m.d[j]);
        ellipse(mul(m.est.ax[0], k), mul(m.est.ax[1], k), C.splat, 3.2);
        ellipse(mul(m.ref.ax[0], k), mul(m.ref.ax[1], k), '#ffffff', 1.3);
        arrow(A, COL_U, 2); arrow(B, COL_V, 2);
        for (j = 0; j < 9; j++) {
          var on = j === 4 || st.nb === 8 || (j & 1) === 1;
          g.fillStyle = j === 4 ? '#fff' : on ? C.splat : 'rgba(206,147,216,.4)';
          g.beginPath(); g.arc(X(m.d[j]), Y(m.d[j]), j === 4 ? 3 : on ? 3.4 : 2.4, 0, 6.2832); g.fill();
        }
        foot((small ? '' : 'differ by ') + pct(m.gap), m.gap < 0.01 ? C.ok : C.text);
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
        g.strokeStyle = 'rgba(143,164,192,.22)'; g.lineWidth = 1; g.font = '500 9px JetBrains Mono, ui-monospace, monospace';
        g.fillStyle = C.mute; g.textBaseline = 'middle'; g.textAlign = 'right';
        [1e-1, 1e-2, 1e-3, 1e-4].forEach(function (e) {
          g.beginPath(); g.moveTo(6, Y(e)); g.lineTo(w - 6, Y(e)); g.stroke(); g.fillText(+(e * 100).toPrecision(1) + '%', w - 7, Y(e) - 6);
        });
        if (pts.length > 1) {
          var last = pts[pts.length - 1];
          g.setLineDash([3, 3]); g.strokeStyle = 'rgba(221,229,240,.45)'; g.beginPath();       // a 1 / n^2 line through the finest point
          g.moveTo(X(4), Y(last[1] * last[0] * last[0] / 16)); g.lineTo(X(last[0]), Y(last[1])); g.stroke(); g.setLineDash([]);
          g.strokeStyle = C.splat; g.lineWidth = 2; g.beginPath();
          pts.forEach(function (p, i) { if (i) g.lineTo(X(p[0]), Y(p[1])); else g.moveTo(X(p[0]), Y(p[1])); });
          g.stroke();
        }
        g.fillStyle = '#fff'; g.beginPath(); g.arc(X(st.n), Y(pr.gap), 3.2, 0, 6.2832); g.fill();
        g.fillStyle = C.dim; g.textAlign = 'left'; g.fillText('differ by, vs samples · dashed = 1/n²', 6, 9);
      }

      /* ---------- scenes ---------- */
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.scene = id; st.sc = sc; st.t = 0; st.touchedProbe = false; st.touchedN = false; st.wigI = -1;
        st.show = {}; sc.show.split(' ').forEach(function (k) { st.show[k] = true; });
        patch.P.set(base);
        if (sc.n) { st.n = sc.n; slN.set(sc.n); }
        if (sc.fixed) { st.u = sc.fixed[0]; st.v = sc.fixed[1]; }
        field.uniforms.uRim.value = sc.rim == null ? 0.6 : sc.rim;
        cap.innerHTML = sc.cap; hint.textContent = sc.hint || '';
        var s = st.show, on = function (o, v) { o.visible = !!v; };
        on(net.obj, s.net); on(netGhost, s.net);
        balls.forEach(function (m, i) {
          m.visible = !!s.net;
          if (s.feat) m.material.color.setRGB(patch.C[i * 3], patch.C[i * 3 + 1], patch.C[i * 3 + 2], T.SRGBColorSpace);
          else m.material.color.set(C.ctrl);
          m.material.emissive.copy(m.material.color).multiplyScalar(0.32);
          m.children[0].visible = !s.feat;
          m.scale.setScalar(s.feat ? 1.25 : 1);
        });
        wire.mat.opacity = s.field && !sc.hero ? 0.22 : s.tan ? 0.34 : 0.55;
        on(field.mesh, s.field);
        on(probe, s.probe); on(aU.obj, s.tan); on(aV.obj, s.tan); on(aN.obj, s.nor);
        on(isoU.obj, s.iso); on(isoV.obj, s.iso);
        on(plane, s.plane); on(planeHid, s.plane); on(plGrid.obj, s.plane); on(plEdge.obj, s.plane); on(plEdgeHid, s.plane);
        on(mark.obj, s.nor);
        on(fill, !s.field || sc.hero);
        for (var gi = 0; gi < 9; gi++) sheet.gauss(gi, null, null, null, null, 0);
        sheet.uniforms.uCell.value.z = 0;
        on(ringA.mesh, s.ell); on(ringA.hid, s.ell);
        on(ringN.mesh, s.nbr); on(ringN.hid, s.nbr);
        [blk.obj, nbRingA.obj, nbRingI.obj, span.obj, spanHid, err.obj].concat(nbDot).forEach(function (o) { on(o, s.nbr); });
        tMu.style.display = s.probe && !s.nbr ? '' : 'none'; tU.style.display = tV.style.display = s.tan ? '' : 'none';
        tN.style.display = s.nor ? '' : 'none';
        uvBox.style.display = s.uv ? '' : 'none';
        fcBox.style.display = s.face ? '' : 'none'; fcBox.classList.toggle('kt-solo', !s.uv);
        root.classList.toggle('kt-has-face', !!s.face); root.classList.toggle('kt-nbr', !!s.nbr);
        nbBar.style.display = s.nbr ? '' : 'none';
        panel.style.display = sc.hero ? 'none' : ''; cap.style.display = sc.hero ? 'none' : '';
        slN.el.style.display = slG.el.style.display = rdG.el.style.display = (s.field || s.ell) ? '' : 'none';
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
        if (sc.sweep && !st.touchedN) {
          var n = Math.round(Math.exp(KT.lerp(Math.log(6), Math.log(150), 0.5 - 0.5 * Math.cos(st.t * 0.55))));
          if (n !== st.n) { st.n = n; slN.set(n); fieldDirty = latDirty = true; }
        }
        if (surfDirty) { updateSurface(); probeDirty = true; if (s.nbr) latDirty = true; surfDirty = false; }
        if (latDirty) { layers(); if (s.nbr) updateLattice(); latDirty = false; }
        if (s.field && fieldDirty) { updateField(); fieldDirty = false; }
        if (s.field) {
          var cnt = st.fieldN * st.fieldN;
          if (sc.hero) cnt = Math.round((0.5 - 0.42 * Math.cos(st.t * 0.33)) * st.fieldN) * st.fieldN;
          field.setCount(cnt);
        }
        if (s.probe && probeDirty) { updateProbe(); probeDirty = false; }

        /* ---- camera: a base view per scene, a rocking for parallax, and a step aside when the plane is edge-on ---- */
        var g = orbit.goal, view = sc.hero ? { th: 0.9, ph: 1.12, r: 5.6 } : { th: 0.72, ph: 1.0, r: 7.2 }, tg = [0, 0.05, 0];
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
        if (st.nudT > 0) { st.nudT -= dt; hint.textContent = st.nudT > 0 ? 'tangent plane edge-on · stepping aside' : (sc.hint || ''); }
        orbit.update(dt);
        // on a small stage the insets cover the lower part: lift the picture into the free area above them
        var occ = s.face ? st.ins + 14 + (s.nbr ? 46 : 0) : 0, span = Math.max((s.uv ? 2 : 1) * (st.ins + 8) + 16, s.nbr ? 226 : 0) / W;
        st.lift += (0.5 * occ * KT.ease((span - 0.3) / 0.3) - st.lift) * (1 - Math.exp(-4 * dt));
        var vk = W + 'x' + H + ':' + Math.round(st.lift);
        if (vk !== st.viewKey) { st.viewKey = vk; cam.setViewOffset(W, H, 0, Math.round(st.lift), W, H); }
        var near = Math.max(0.2, orbit.cur.r - 1.3), far = orbit.cur.r + 5.2;
        scene.fog.near = near; scene.fog.far = far; sheet.uniforms.uFog.value.set(near, far + 1.5);
        R.renderer.render(scene, cam);

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
      });
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
          '.kt-sim-planoid.kt-ins-small .kt-wide{display:none}.kt-sim-planoid.kt-ins-small .kt-uv-lab{letter-spacing:.04em;font-size:9px}' +
          '.kt-sim-planoid .kt-nbsw{position:absolute;z-index:5;left:12px;bottom:calc(20px + var(--kt-ins));padding:4px 4px 4px 10px;' +
          'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px)}' +
          '.kt-sim-planoid .kt-nbsw .kt-seg-btn.on{background:var(--splat)}' +
          '.kt-sim-planoid .kt-pl-spark{position:relative;height:92px;margin-top:2px;border-top:1px solid var(--line)}' +
          '.kt-sim-planoid .kt-pl-spark-cv{position:absolute;inset:0;width:100%;height:100%}' +
          '.kt-sim-planoid .kt-readout .kt-lab sub,.kt-sim-planoid .kt-readout .kt-lab sup{font-size:8px;line-height:0}';
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
        resize: function (w, h) {
          W = w; H = h; R.resize(w, h); uvDirty = faceDirty = true; st.sparkT = Math.min(st.sparkT, 0.05);
          fitR = Math.max(1, 1.32 / (w / h));               // back off on narrow stages so the patch stays in frame
          var ins = Math.round(KT.clamp(Math.min(w * 0.22, h * 0.3), 108, 200));
          st.ins = ins; root.style.setProperty('--kt-ins', ins + 'px'); root.classList.toggle('kt-ins-small', ins < 150);
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });
})();
