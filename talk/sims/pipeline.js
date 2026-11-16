/* Knot-a-Surface talk — module `pipeline`: "From photographs to charts".
   What comes before chapter 06's owl, on the real data of DTU scan 122: talk/dtu/pipeline122.js, written offline by
   tools/dtu/pipeline.py in the scene frame and units of talk/dtu-data.js (+y up, the table at y = 0, 1 unit = 80 mm).
     photos      the dataset's 64 calibrated cameras as frusta (pose and pinhole from the file). The photographs themselves
                 are not shown: their terms of use could not be checked (`photos: null` in the file).
     sfm         the dataset's sparse COLMAP points, each appearing when its SECOND camera is added (a point needs two
                 views). COLMAP matched and triangulated them WITH the dataset's calibrated poses: no pose was estimated.
     init3dgs    the 3DGS initialisation from those points as the reference implementation (graphdeco-inria) makes it:
                 one isotropic Gaussian per point, variance = the mean squared distance to its three nearest neighbours
                 (simple-knn's distCUDA2; done here exactly, with a k-d tree), the point's colour, opacity 0.1. Drawn the
                 way 3DGS draws: EWA-projected, sorted, alpha-composited (a material of this module, inside gl.draw's
                 frame). An inset magnifies one Gaussian and its three neighbours. Nothing is trained.
     initspline  a 12 x 16 cubic chart fitted by pipeline.py to the sparse points only (robust least squares, no depth
                 map): its amber control net and Gaussians derived from it (gl.writePlanoids), with the file's numbers.
     levels      the coarse-to-fine levels of fit.py's GEOMETRIC fit of the owl's chart to the fused depth maps (layout,
                 levels 1 to 3, then the shipped net of talk/dtu-data.js), morphing at fixed (u, v), each with its
                 measured distance to the 303,155 measured points. Not training, not iterations of an optimiser. Level 1
                 is a little further from the surface than the layout, and the rms rises from level 3 to level 4
                 (numbers only, no net): both real, both shown.
     train       with talk/dtu/train122.js (KnotTalk.dtuTrain[122], exported from a real 3DGS run): its snapshots one
                 after another (they cannot be interpolated: Gaussians are cloned and pruned between them), with the
                 Gaussians in each checkpoint, those kept in the file, and the logged loss and PSNR. Without that file the
                 run is PENDING, said so on screen, and the stage shows the stand-in: the 3DGS initialisation beside the
                 levels of the geometric fit.
   Every number on screen comes from those files or is measured in the page.
   Tool: WebGL (three.js). Standalone: talk/sims/pipeline.html. Depends on vendor/three.min.js, nurbs.js, kit.js, gl.js and
   dtu/pipeline122.js; on dtu-data.js for the shipped owl; on dtu/train122.js when it exists. */
(function () {
  'use strict';
  var KT = window.KnotTalk, NB = KT.nurbs, C = KT.color, SCAN = 122;
  var FRUST = 1.2;               // depth at which a camera's image rectangle is drawn (scene units: 96 mm)
  var NUM = 59;                  // numbers per Gaussian in the reference model: position 3, scale 3, rotation 4, opacity 1, colour 48 (SH to degree 3)
  var RAYS = 26;                 // triangulation rays drawn for the camera just added
  var RATE = { photos: 9, sfm: 4.6 };          // cameras added per second
  var HOLD = 3.6, MORPH = 1.2, SNAP = 2.6;     // levels: seconds a level holds, seconds a morph takes; train: seconds per snapshot
  var KS = 0.72;                 // sigma of a derived Gaussian as a fraction of its sampling interval (as in the manifold module)

  var SCENES = {
    photos: { view: 'rig', cap: '<b>the photographs</b> · DTU scan 122 · 64 calibrated cameras<br>each frustum is one camera: its pose and pinhole as the dataset gives them. The photographs themselves are not shown' },
    sfm: { view: 'rig', cap: '<b>sparse points</b> · the dataset’s own COLMAP model<br>features matched and triangulated with the dataset’s calibrated poses: no pose was estimated · a point appears when a second camera sees it' },
    init3dgs: { view: 'owl', cap: '<b>3DGS initialisation</b> · as the reference implementation makes it<br>one round Gaussian per sparse point · σ from its three nearest neighbours · the point’s colour · opacity 0.1 · nothing trained' },
    initspline: { view: 'owl', cap: '<b>our initialisation</b> · one cubic chart fitted to the same points<br>12 × 16 control points, robust least squares on the sparse points only (no depth map) · every Gaussian derived from the chart' },
    levels: { view: 'owl', cap: '<b>coarse to fine</b> · the owl’s chart fitted offline to the fused depth maps<br>a geometric fit, not training: each level is a finer grid of nodes pulled onto the measured surface' },
    train: { view: 'owl', cap: '' }
  };

  function fmt(n) { return Math.round(n).toLocaleString('en-US'); }
  function f2(v) { return (+v).toFixed(2); }
  function pc(v) { return (v * 100).toFixed(1) + ' %'; }
  function every(n, s) { var a = [], i; for (i = 0; i < n; i++) if (i % s === 0 || i === n - 1) a.push(i); return a; }
  function finite(v) { return typeof v === 'number' && isFinite(v); }

  KT.register('pipeline', {
    title: 'From photographs to charts',
    blurb: 'DTU scan 122 from its 64 calibrated photographs to the owl’s spline chart: the cameras, the dataset’s sparse COLMAP points, ' +
           'the standard 3DGS initialisation from them beside a chart fitted to the same points, and the coarse-to-fine levels of the ' +
           'offline geometric fit to the depth maps. Nothing here is trained; a real 3DGS run is replayed once its snapshots are exported.',
    tool: 'WebGL · three.js · cameras, points, two kinds of Gaussians',
    scenes: [{ id: 'photos', label: 'Photographs' }, { id: 'sfm', label: 'Sparse points' }, { id: 'init3dgs', label: '3DGS init' },
             { id: 'initspline', label: 'Chart init' }, { id: 'levels', label: 'Coarse to fine' }, { id: 'train', label: 'Training' }],
    create: function (env) {
      var T = window.THREE, G = KT.gl, root = env.root, ui = env.ui;
      var cam = new T.PerspectiveCamera(36, 1, 0.02, 400), R = G.renderer(root, cam, env.opts.clear), scene = new T.Scene();
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);
      addCss();
      var D = KT.dtuPipeline && KT.dtuPipeline[SCAN];
      if (!D) {                                   // nothing to show without the data file: say which one
        cap.innerHTML = '<b>no data</b> · this module needs talk/dtu/pipeline122.js (written by tools/dtu/pipeline.py)';
        return { setScene: function () {}, resize: function (w, h) { R.resize(w, h); R.renderer.render(scene, cam); }, destroy: function () { R.renderer.dispose(); } };
      }
      var TR = KT.dtuTrain && KT.dtuTrain[SCAN] && KT.dtuTrain[SCAN].snaps && KT.dtuTrain[SCAN].snaps.length ? KT.dtuTrain[SCAN] : null;
      var OWL = KT.dtu && KT.dtu.scan === SCAN ? KT.dtu.charts.filter(function (c) { return !c.plane; })[0] : null;   // the shipped chart
      var CAMS = D.cameras.list, NC = CAMS.length, IN = D.cameras.intrinsics, SF = D.sfm, NP = SF.count, MM = D.unit_mm;
      var TGT = KT.dtu && KT.dtu.target ? KT.dtu.target.slice() : [0, 1.75, 0.2];
      var small = Math.min(window.innerWidth, window.innerHeight) < 560;
      var pr = function () { return R.renderer.getPixelRatio(); };
      var st = { scene: '', sc: SCENES.photos, t: 0, k: 0, kGo: NC, play: true, w: 2, h: 2, split: false, rects: null, narrow: false,
                 rayJ: -1, rayT: 9, sortKey: '', sortT: 0, nets: true, pts: true, opa: 'init', nInit: small ? 10000 : 20000 };

      /* ================= the rig: 64 frusta from the file's poses and pinhole ================= */
      function axes(c) { var r = c.R; return [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]]; }   // camera x (right), y (down), z (forward) in scene axes
      function pix(c, px, py, d) {               // the scene point that pixel (px, py) of camera c sees at depth d
        var a = axes(c), x = (px - IN.cx) / IN.fx * d, y = (py - IN.cy) / IN.fy * d, o = [], q;
        for (q = 0; q < 3; q++) o.push(c.c[q] + a[0][q] * x + a[1][q] * y + a[2][q] * d);
        return o;
      }
      function camMat(colour, base, hi) {        // a camera's lines or image rectangle: shown once camera aCam is added (uK > aCam), flashing as it is
        return new T.ShaderMaterial({
          transparent: true, depthWrite: false, side: T.DoubleSide,
          uniforms: { uK: { value: 0 }, uAll: { value: 0 }, uCol: { value: new T.Color(colour) }, uBase: { value: base }, uHi: { value: hi } },
          vertexShader: 'attribute float aCam; uniform float uK; uniform float uAll; varying float vF; varying float vIn;\n' +
            'void main(){ float age = uK - aCam; vIn = age > 0.0 ? 1.0 : 0.0; vF = age > 0.0 ? clamp(1.0 - age / 2.5, 0.0, 1.0) : 0.0;\n' +
            '  if (age <= 0.0 && uAll < 0.5) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }\n' +
            '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
          fragmentShader: 'uniform vec3 uCol; uniform float uBase; uniform float uHi; varying float vF; varying float vIn;\n' +
            'void main(){ gl_FragColor = vec4(uCol, min(1.0, uBase * mix(0.3, 1.0, vIn) + uHi * vF)); }'
        });
      }
      function camGeo(list, ids, index) {
        var g = new T.BufferGeometry(), P = new Float32Array(list.length * 3), A = new Float32Array(ids), i;
        for (i = 0; i < list.length; i++) P.set(list[i], i * 3);
        g.setAttribute('position', new T.BufferAttribute(P, 3)); g.setAttribute('aCam', new T.BufferAttribute(A, 1));
        if (index) g.setIndex(index);
        return g;
      }
      var rig = (function () {
        var L = [], LI = [], Q = [], QI = [], X = [], XI = [], AX = [], AI = [];
        CAMS.forEach(function (c, i) {
          var W = IN.width, H = IN.height, k = [pix(c, 0, 0, FRUST), pix(c, W, 0, FRUST), pix(c, W, H, FRUST), pix(c, 0, H, FRUST)], q, n = Q.length;
          for (q = 0; q < 4; q++) { L.push(c.c, k[q], k[q], k[(q + 1) % 4]); LI.push(i, i, i, i); }
          // a notch on the top edge (row 0 of the photograph), so the camera's roll reads
          var a = pix(c, W * 0.4, 0, FRUST), b = pix(c, W * 0.5, -H * 0.17, FRUST), e = pix(c, W * 0.6, 0, FRUST);
          L.push(a, b, b, e); LI.push(i, i, i, i);
          k.forEach(function (p) { Q.push(p); QI.push(i); });
          X.push(n, n + 1, n + 2, n, n + 2, n + 3);
          // the optical axis, to its closest approach to the owl's centre
          var f = axes(c)[2], t = (TGT[0] - c.c[0]) * f[0] + (TGT[1] - c.c[1]) * f[1] + (TGT[2] - c.c[2]) * f[2];
          AX.push(c.c, [c.c[0] + f[0] * t, c.c[1] + f[1] * t, c.c[2] + f[2] * t]); AI.push(i, i);
        });
        var lines = new T.LineSegments(camGeo(L, LI), camMat(C.text, 0.62, 0.38));
        var planes = new T.Mesh(camGeo(Q, QI, X), camMat(C.text, 0.07, 0.5));
        var optic = new T.LineSegments(camGeo(AX, AI), camMat(C.dim, 0.13, 0.25));
        // the table plane under the owl: a ring and a cross where the optical axes meet
        var ring = [], s;
        for (s = 0; s < 96; s++) {
          var a0 = s / 96 * Math.PI * 2, a1 = (s + 1) / 96 * Math.PI * 2;
          ring.push([TGT[0] + 1.5 * Math.cos(a0), 0, TGT[2] + 1.5 * Math.sin(a0)], [TGT[0] + 1.5 * Math.cos(a1), 0, TGT[2] + 1.5 * Math.sin(a1)]);
        }
        ring.push([TGT[0] - 0.25, 0, TGT[2]], [TGT[0] + 0.25, 0, TGT[2]], [TGT[0], 0, TGT[2] - 0.25], [TGT[0], 0, TGT[2] + 0.25]);
        var gr = new T.BufferGeometry(), gp = new Float32Array(ring.length * 3);
        ring.forEach(function (p, i) { gp.set(p, i * 3); });
        gr.setAttribute('position', new T.BufferAttribute(gp, 3));
        var floor = new T.LineSegments(gr, new T.LineBasicMaterial({ color: C.mute, transparent: true, opacity: 0.45 }));
        [lines, planes, optic, floor].forEach(function (o) { o.frustumCulled = false; scene.add(o); });
        lines.renderOrder = 4; planes.renderOrder = 3;
        var mats = [lines.material, planes.material, optic.material];
        return { objs: [lines, planes, optic, floor], set: function (k, all) { mats.forEach(function (m) { m.uniforms.uK.value = k; m.uniforms.uAll.value = all ? 1 : 0; }); },
                 show: function (v) { this.objs.forEach(function (o) { o.visible = v; }); } };
      })();

      /* ================= the sparse points: a point appears when its second camera is added ================= */
      var sfm = (function () {
        var g = new T.BufferGeometry(), sec = new Float32Array(NP), i;
        for (i = 0; i < NP; i++) sec[i] = SF.second[i];
        g.setAttribute('position', new T.BufferAttribute(SF.P, 3)); g.setAttribute('aCol', new T.BufferAttribute(SF.C, 3, true));
        g.setAttribute('aSec', new T.BufferAttribute(sec, 1));
        var m = new T.ShaderMaterial({
          uniforms: { uK: { value: NC }, uSize: { value: 2.6 }, uDim: { value: 1 } },
          vertexShader: 'attribute vec3 aCol; attribute float aSec; uniform float uK; uniform float uSize; varying vec3 vCol; varying float vPop;\n' +
            'void main(){ float age = uK - aSec; vCol = aCol; vPop = clamp(1.0 - age / 1.6, 0.0, 1.0);\n' +
            '  if (age <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); gl_PointSize = 0.0; return; }\n' +
            '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); gl_PointSize = uSize * (1.0 + 1.3 * vPop); }',
          fragmentShader: 'uniform float uDim; varying vec3 vCol; varying float vPop;\n' +
            'void main(){ vec2 q = gl_PointCoord * 2.0 - 1.0; if (dot(q, q) > 1.0) discard;\n' +
            '  gl_FragColor = vec4(mix(vCol * uDim, vec3(1.0), 0.75 * vPop), 1.0); }'
        });
        var o = new T.Points(g, m); o.frustumCulled = false; o.renderOrder = 1; scene.add(o);
        return o;
      })();
      /* per camera: the points it completes, and the counts and reprojection errors as cameras are added */
      var byCam = [], ebin = new Int32Array(NC * 256), cum = new Int32Array(NC + 1), medC = {};
      (function () {
        var i, j, b;
        for (j = 0; j < NC; j++) byCam.push([]);
        for (i = 0; i < NP; i++) { j = SF.second[i]; byCam[j].push(i); b = Math.min(255, Math.round(SF.error[i] / SF.error_scale)); ebin[j * 256 + b]++; }
        for (j = 0; j < NC; j++) cum[j + 1] = cum[j] + byCam[j].length;
      })();
      function medErr(k) {                         // median reprojection error (px) of the points whose second camera is among the first k
        if (medC[k] != null) return medC[k];
        var N = cum[k], h = new Int32Array(256), j, b, s = 0, r1 = (N - 1) >> 1, r2 = N >> 1, v1 = -1, v2 = -1;
        if (!N) return (medC[k] = NaN);
        for (j = 0; j < k; j++) for (b = 0; b < 256; b++) h[b] += ebin[j * 256 + b];
        for (b = 0; b < 256 && v2 < 0; b++) { s += h[b]; if (v1 < 0 && s > r1) v1 = b; if (s > r2) v2 = b; }   // errors are stored in steps of error_scale: exact
        return (medC[k] = (v1 + v2) / 2 * SF.error_scale);
      }
      var rays = G.lines(RAYS * 4, C.text, 0.4); rays.obj.renderOrder = 5; scene.add(rays.obj);
      function setRays(j) {                        // the newest camera's points: a ray from each of its two cameras
        st.rayJ = j; st.rayT = 0;
        var L = j >= 0 ? byCam[j] : [], step = Math.max(1, Math.floor(L.length / RAYS)), n = 0, A = rays.arr, i, p, a, b;
        for (i = 0; i < L.length && n < RAYS * 4; i += step) {
          p = L[i]; a = CAMS[SF.first[p]].c; b = CAMS[j].c;
          A.set(a, n * 3); A.set([SF.P[p * 3], SF.P[p * 3 + 1], SF.P[p * 3 + 2]], n * 3 + 3);
          A.set(b, n * 3 + 6); A.set([SF.P[p * 3], SF.P[p * 3 + 1], SF.P[p * 3 + 2]], n * 3 + 9); n += 4;
        }
        rays.set(n);
      }

      /* ================= 3DGS Gaussians, drawn as 3DGS draws them =================
         Each Gaussian: centre, 3D covariance (upper triangle), colour, opacity. The vertex shader projects the covariance
         with the Jacobian of the perspective map (EWA, as the reference rasterizer; + 0.3 px^2 on the diagonal, its low-pass)
         and spans a quad to 3 sigma along the 2D axes; the fragment is opacity x exp(-d^2 / 2), cut below 1/255, blended
         back to front over what is behind (the instances are sorted by depth on the CPU when the camera turns). */
      var GS = (function () {
        var cap = NP, i;
        if (TR) TR.snaps.forEach(function (s) { cap = Math.max(cap, s.n); });
        var base = new T.PlaneGeometry(2, 2), g = new T.InstancedBufferGeometry(), at = {};
        g.index = base.index; g.setAttribute('position', base.getAttribute('position'));
        [['iC', 3], ['iA', 3], ['iB', 3], ['iK', 4]].forEach(function (s) {
          var a = new T.InstancedBufferAttribute(new Float32Array(cap * s[1]), s[1]); a.setUsage(T.DynamicDrawUsage); g.setAttribute(s[0], a); at[s[0]] = a;
        });
        g.instanceCount = 0;
        var mat = new T.ShaderMaterial({
          transparent: true, depthWrite: false, depthTest: true,
          uniforms: { uFocal: { value: new T.Vector2(1, 1) }, uView: { value: new T.Vector2(1, 1) }, uAlpha: { value: 1 }, uOpa: { value: -1 } },
          vertexShader: [
            'attribute vec3 iC; attribute vec3 iA; attribute vec3 iB; attribute vec4 iK;',
            'uniform vec2 uFocal; uniform vec2 uView; uniform float uAlpha; uniform float uOpa;',
            'varying vec2 vQ; varying vec4 vK;',
            'void main(){',
            '  vec4 e = viewMatrix * vec4(iC, 1.0); vec4 p = projectionMatrix * e; float z = -e.z;',
            '  vQ = vec2(0.0); vK = vec4(0.0);',
            '  if (z < 0.01 || abs(p.x) > 1.3 * p.w || abs(p.y) > 1.3 * p.w) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }',
            '  mat3 V = mat3(iA.x, iA.y, iA.z, iA.y, iB.x, iB.y, iA.z, iB.y, iB.z);',
            '  mat3 J = mat3(uFocal.x / z, 0.0, 0.0, 0.0, uFocal.y / z, 0.0, uFocal.x * e.x / (z * z), uFocal.y * e.y / (z * z), 0.0);',
            '  mat3 M = J * mat3(viewMatrix); mat3 TV = M * V;',
            '  vec3 m0 = vec3(M[0][0], M[1][0], M[2][0]), m1 = vec3(M[0][1], M[1][1], M[2][1]);',
            '  vec3 t0 = vec3(TV[0][0], TV[1][0], TV[2][0]), t1 = vec3(TV[0][1], TV[1][1], TV[2][1]);',
            '  float a = dot(t0, m0) + 0.3, b = dot(t0, m1), c = dot(t1, m1) + 0.3;',
            '  float mid = 0.5 * (a + c), rad = length(vec2(0.5 * (a - c), b)), l1 = mid + rad, l2 = max(mid - rad, 0.1);',
            '  vec2 d = vec2(b, l1 - a); float dl = length(d);',
            '  vec2 v1 = dl > 1e-6 ? d / dl : (a >= c ? vec2(1.0, 0.0) : vec2(0.0, 1.0)), v2 = vec2(-v1.y, v1.x);',
            '  float r1 = 3.0 * sqrt(l1), r2 = 3.0 * sqrt(l2);',
            '  if (r1 > 3.0 * max(uView.x, uView.y)) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }',
            '  vec2 off = position.x * r1 * v1 + position.y * r2 * v2;',
            '  vQ = position.xy * 3.0; vK = iK; if (uOpa >= 0.0) vK.a = uOpa; vK.a *= uAlpha;',
            '  gl_Position = vec4(p.xy / p.w + off * 2.0 / uView, p.z / p.w, 1.0);',
            '}'].join('\n'),
          fragmentShader: [
            'precision highp float;',
            'varying vec2 vQ; varying vec4 vK;',
            'void main(){',
            '  float a = vK.a * exp(-0.5 * dot(vQ, vQ));',
            '  if (a < 1.0 / 255.0) discard;',
            '  gl_FragColor = vec4(vK.rgb, min(a, 0.99));',
            '}'].join('\n')
        });
        var mesh = new T.Mesh(g, mat); mesh.frustumCulled = false; mesh.renderOrder = 2; scene.add(mesh);
        var src = { C: new Float32Array(cap * 3), A: new Float32Array(cap * 3), B: new Float32Array(cap * 3), K: new Float32Array(cap * 4) };
        var depth = new Float32Array(cap), key = new Uint16Array(cap), order = new Uint32Array(cap), cnt = new Uint32Array(65536);
        return {
          mesh: mesh, mat: mat, cap: cap, n: 0, src: src, what: '',
          /* back to front along the camera's view direction (the order does not depend on where the camera is) */
          sort: function (camera) {
            var n = this.n, e = camera.matrixWorld.elements, fx = -e[8], fy = -e[9], fz = -e[10], mn = Infinity, mx = -Infinity, j, s, v;
            for (j = 0; j < n; j++) { v = src.C[j * 3] * fx + src.C[j * 3 + 1] * fy + src.C[j * 3 + 2] * fz; depth[j] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
            var sc = mx > mn ? 65535 / (mx - mn) : 0;
            cnt.fill(0);
            for (j = 0; j < n; j++) { v = ((mx - depth[j]) * sc) | 0; key[j] = v; cnt[v]++; }
            for (s = 0, j = 0; j < 65536; j++) { v = cnt[j]; cnt[j] = s; s += v; }
            for (j = 0; j < n; j++) order[cnt[key[j]]++] = j;
            var oc = at.iC.array, oa = at.iA.array, ob = at.iB.array, ok = at.iK.array, q, t;
            for (j = 0; j < n; j++) {
              q = order[j]; t = q * 3;
              oc[j * 3] = src.C[t]; oc[j * 3 + 1] = src.C[t + 1]; oc[j * 3 + 2] = src.C[t + 2];
              oa[j * 3] = src.A[t]; oa[j * 3 + 1] = src.A[t + 1]; oa[j * 3 + 2] = src.A[t + 2];
              ob[j * 3] = src.B[t]; ob[j * 3 + 1] = src.B[t + 1]; ob[j * 3 + 2] = src.B[t + 2];
              ok[j * 4] = src.K[q * 4]; ok[j * 4 + 1] = src.K[q * 4 + 1]; ok[j * 4 + 2] = src.K[q * 4 + 2]; ok[j * 4 + 3] = src.K[q * 4 + 3];
            }
            ['iC', 'iA', 'iB', 'iK'].forEach(function (k) { at[k].updateRange.offset = 0; at[k].updateRange.count = n * at[k].itemSize; at[k].needsUpdate = true; });
            g.instanceCount = n;
          },
          /* the camera's focal length and the viewport, in device pixels (call before each draw) */
          prep: function (camera, vw, vh) {
            var P = camera.projectionMatrix.elements;
            mat.uniforms.uFocal.value.set(P[0] * vw / 2, P[5] * vh / 2); mat.uniforms.uView.value.set(vw, vh);
          }
        };
      })();

      /* the reference initialisation: sigma^2 = mean squared distance to the three nearest other points (exact, k-d tree) */
      var INIT = null;
      function initGauss() {
        if (INIT) return INIT;
        var t0 = performance.now(), P = SF.P, n = NP, idx = new Int32Array(n), ax = new Uint8Array(n), i;
        for (i = 0; i < n; i++) idx[i] = i;
        function select(lo, hi, k, a) {          // Wirth: idx[lo..hi] around its k-th element by coordinate a
          while (lo < hi) {
            var x = P[idx[k] * 3 + a], l = lo, r = hi, t;
            do {
              while (P[idx[l] * 3 + a] < x) l++;
              while (x < P[idx[r] * 3 + a]) r--;
              if (l <= r) { t = idx[l]; idx[l] = idx[r]; idx[r] = t; l++; r--; }
            } while (l <= r);
            if (r < k) lo = l;
            if (k < l) hi = r;
          }
        }
        function build(lo, hi) {                 // [lo, hi): the median along the longest side splits
          if (hi - lo < 2) return;
          var mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity], j, q, v, a = 0;
          for (j = lo; j < hi; j++) for (q = 0; q < 3; q++) { v = P[idx[j] * 3 + q]; if (v < mn[q]) mn[q] = v; if (v > mx[q]) mx[q] = v; }
          if (mx[1] - mn[1] > mx[a] - mn[a]) a = 1; if (mx[2] - mn[2] > mx[a] - mn[a]) a = 2;
          var mid = (lo + hi) >> 1; select(lo, hi - 1, mid, a); ax[mid] = a;
          build(lo, mid); build(mid + 1, hi);
        }
        build(0, n);
        var best = new Float64Array(3), nb = new Int32Array(3), qx, qy, qz, qi;
        function visit(lo, hi) {
          if (hi <= lo) return;
          var mid = (lo + hi) >> 1, p = idx[mid], dx, dy, dz, d;
          if (p !== qi) {
            dx = P[p * 3] - qx; dy = P[p * 3 + 1] - qy; dz = P[p * 3 + 2] - qz; d = dx * dx + dy * dy + dz * dz;
            if (d < best[2]) {
              if (d < best[1]) { best[2] = best[1]; nb[2] = nb[1]; if (d < best[0]) { best[1] = best[0]; nb[1] = nb[0]; best[0] = d; nb[0] = p; } else { best[1] = d; nb[1] = p; } }
              else { best[2] = d; nb[2] = p; }
            }
          }
          if (hi - lo === 1) return;
          var a = ax[mid], diff = (a === 0 ? qx : a === 1 ? qy : qz) - P[p * 3 + a];
          if (diff < 0) { visit(lo, mid); if (diff * diff < best[2]) visit(mid + 1, hi); }
          else { visit(mid + 1, hi); if (diff * diff < best[2]) visit(lo, mid); }
        }
        var sig = new Float32Array(n), near = new Int32Array(n * 3);
        for (i = 0; i < n; i++) {
          qi = i; qx = P[i * 3]; qy = P[i * 3 + 1]; qz = P[i * 3 + 2]; best[0] = best[1] = best[2] = Infinity;
          visit(0, n);
          sig[i] = Math.sqrt(Math.max((best[0] + best[1] + best[2]) / 3, 1e-14)); near[i * 3] = nb[0]; near[i * 3 + 1] = nb[1]; near[i * 3 + 2] = nb[2];
        }
        var sorted = Array.prototype.slice.call(sig).sort(function (a, b) { return a - b; });
        INIT = { sig: sig, near: near, med: (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2, ms: performance.now() - t0 };
        return INIT;
      }
      function gsInit() {                          // fill the 3DGS field with the initialisation
        if (GS.what === 'init') return;
        var I = initGauss(), s = GS.src, i, v;
        for (i = 0; i < NP; i++) {
          s.C[i * 3] = SF.P[i * 3]; s.C[i * 3 + 1] = SF.P[i * 3 + 1]; s.C[i * 3 + 2] = SF.P[i * 3 + 2];
          v = I.sig[i] * I.sig[i];
          s.A[i * 3] = v; s.A[i * 3 + 1] = 0; s.A[i * 3 + 2] = 0; s.B[i * 3] = v; s.B[i * 3 + 1] = 0; s.B[i * 3 + 2] = v;
          s.K[i * 4] = SF.C[i * 3] / 255; s.K[i * 4 + 1] = SF.C[i * 3 + 1] / 255; s.K[i * 4 + 2] = SF.C[i * 3 + 2] / 255; s.K[i * 4 + 3] = 0.1;
        }
        GS.n = NP; GS.what = 'init'; st.sortKey = '';
      }
      function gsSnap(k) {                         // fill it with snapshot k of the real run: covariance = R diag(s^2) R^T
        var S = TR.snaps[k], n = Math.min(S.n, GS.cap), s = GS.src, i, w, x, y, z, l, sx, sy, sz;
        for (i = 0; i < n; i++) {
          s.C[i * 3] = S.p[i * 3]; s.C[i * 3 + 1] = S.p[i * 3 + 1]; s.C[i * 3 + 2] = S.p[i * 3 + 2];
          w = S.q[i * 4]; x = S.q[i * 4 + 1]; y = S.q[i * 4 + 2]; z = S.q[i * 4 + 3]; l = Math.hypot(w, x, y, z) || 1; w /= l; x /= l; y /= l; z /= l;
          var r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
          var r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
          var r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
          sx = S.s[i * 3] * S.s[i * 3]; sy = S.s[i * 3 + 1] * S.s[i * 3 + 1]; sz = S.s[i * 3 + 2] * S.s[i * 3 + 2];
          s.A[i * 3] = r00 * r00 * sx + r01 * r01 * sy + r02 * r02 * sz;
          s.A[i * 3 + 1] = r00 * r10 * sx + r01 * r11 * sy + r02 * r12 * sz;
          s.A[i * 3 + 2] = r00 * r20 * sx + r01 * r21 * sy + r02 * r22 * sz;
          s.B[i * 3] = r10 * r10 * sx + r11 * r11 * sy + r12 * r12 * sz;
          s.B[i * 3 + 1] = r10 * r20 * sx + r11 * r21 * sy + r12 * r22 * sz;
          s.B[i * 3 + 2] = r20 * r20 * sx + r21 * r21 * sy + r22 * r22 * sz;
          s.K[i * 4] = S.c[i * 3] / 255; s.K[i * 4 + 1] = S.c[i * 3 + 1] / 255; s.K[i * 4 + 2] = S.c[i * 3 + 2] / 255; s.K[i * 4 + 3] = S.a[i] / 255;
        }
        GS.n = n; GS.what = 'snap' + k; st.sortKey = '';
      }

      /* ================= Gaussians derived from charts (the talk's own renderer) ================= */
      var CAPS = small ? 26000 : 46000, field = G.splats(CAPS); scene.add(field.mesh);
      field.uniforms.uRim.value = 0.3;
      var VIOLET = G.rgb(C.splat), dcache = {};
      function grid(n) { return { pos: new Float32Array(n * 3), su: new Float32Array(n * 3), sv: new Float32Array(n * 3), col: new Float32Array(n * 3) }; }
      function sample(patch, na, nb, out) {
        var us = new Float32Array(na), vs = new Float32Array(nb), i;
        for (i = 0; i < na; i++) us[i] = (i + 0.5) / na;
        for (i = 0; i < nb; i++) vs[i] = (i + 0.5) / nb;
        NB.patchGrid(patch, us, vs, out);
        return out;
      }
      function dims(patch, N) {                    // a lattice of about N samples whose cells are square on the surface
        var g = sample(patch, 12, 12, grid(144)), lu = 0, lv = 0, i;
        for (i = 0; i < 144; i++) { lu += Math.hypot(g.su[i * 3], g.su[i * 3 + 1], g.su[i * 3 + 2]); lv += Math.hypot(g.sv[i * 3], g.sv[i * 3 + 1], g.sv[i * 3 + 2]); }
        var nb = Math.max(4, Math.round(Math.sqrt(N * lv / lu))), na = Math.max(4, Math.round(N / nb));
        while (na * nb > CAPS) nb--;
        return [na, nb];
      }
      function writeField(g, na, nb, rgb) {
        var key = na + 'x' + nb, d = dcache[key];
        if (!d) d = dcache[key] = [new Float32Array(na).fill(1 / na), new Float32Array(nb).fill(1 / nb)];
        var n = G.writePlanoids(field, 0, g, na, nb, d[0], d[1], KS, rgb || null);
        field.setCount(n); field.commit(n); st.drawn = n;
        return n;
      }
      function paint(g, rgb, n) { for (var i = 0; i < n; i++) { g.col[i * 3] = rgb[0]; g.col[i * 3 + 1] = rgb[1]; g.col[i * 3 + 2] = rgb[2]; } }

      /* the chart fitted to the sparse points only */
      var initPatch = NB.makePatch({ nu: D.init.nu, nv: D.init.nv }); initPatch.P.set(D.init.P);
      function writeInit() {
        var d = dims(initPatch, st.nInit), g = sample(initPatch, d[0], d[1], grid(d[0] * d[1]));
        writeField(g, d[0], d[1], VIOLET); field.uniforms.uFlat.value = 0;
        rdGs.set(fmt(st.drawn));
      }

      /* the levels of the geometric fit, then the shipped net: each sampled once on the same (u, v) lattice */
      var LV = null;
      function levels() {
        if (LV) return LV;
        var S = D.levels.map(function (l) { var p = NB.makePatch({ nu: l.nu, nv: l.nv }); p.P.set(l.P); return { name: l.name, patch: p, d: l, photo: false }; });
        if (OWL) { var p = NB.makePatch({ nu: OWL.nu, nv: OWL.nv }); p.P.set(OWL.P); p.C.set(OWL.C); S.push({ name: 'shipped', patch: p, d: D.shipped, photo: true }); }
        var dm = dims(S[0].patch, small ? 13000 : 24000), na = dm[0], nb = dm[1];
        S.forEach(function (s) { s.g = sample(s.patch, na, nb, grid(na * nb)); if (!s.photo) paint(s.g, VIOLET, na * nb); });
        LV = { S: S, na: na, nb: nb, work: grid(na * nb), from: grid(na * nb), fromFlat: 0, i: 0, f: 1, t: 0, auto: true, dirty: true };
        LV.from.pos.set(S[0].g.pos); LV.from.su.set(S[0].g.su); LV.from.sv.set(S[0].g.sv); LV.from.col.set(S[0].g.col);
        return LV;
      }
      function flatOf(L) { var f = KT.ease(L.f); return L.fromFlat + ((L.S[L.i].photo ? 1 : 0) - L.fromFlat) * f; }
      function levelGo(i) {                        // morph from whatever is drawn now to level i
        var L = levels(), w = L.f < 1 ? L.work : L.S[L.i].g, A = L.from;
        if (L.f < 1) blend(L);                     // mid-morph: start from the blend on screen
        A.pos.set(w.pos); A.su.set(w.su); A.sv.set(w.sv); A.col.set(w.col);
        L.fromFlat = flatOf(L); L.i = i; L.f = 0; L.t = 0; L.dirty = true;
        if (st.nets && !st.split) setNet(L.S[i].patch);
        paintLevel(); plotDirty = true;
      }
      function blend(L) {
        var a = L.from, b = L.S[L.i].g, w = L.work, f = KT.ease(L.f), n = L.na * L.nb * 3, j;
        for (j = 0; j < n; j++) {
          w.pos[j] = a.pos[j] + (b.pos[j] - a.pos[j]) * f; w.su[j] = a.su[j] + (b.su[j] - a.su[j]) * f;
          w.sv[j] = a.sv[j] + (b.sv[j] - a.sv[j]) * f; w.col[j] = a.col[j] + (b.col[j] - a.col[j]) * f;
        }
        return w;
      }
      function levelsStep(dt) {
        var L = levels();
        if (L.f < 1) { L.f = Math.min(1, L.f + dt / MORPH); L.dirty = true; }
        else if (L.auto) { L.t += dt; if (L.t > (L.i === L.S.length - 1 ? HOLD * 1.7 : HOLD)) levelGo((L.i + 1) % L.S.length); }
        if (L.dirty) {
          L.dirty = false;
          writeField(L.f < 1 ? blend(L) : L.S[L.i].g, L.na, L.nb);
          field.uniforms.uFlat.value = flatOf(L);
          rdGs.set(fmt(st.drawn));
        }
      }

      /* ================= control nets (amber); a dense net is drawn thinned ================= */
      var net = G.lines(44000 * 2, C.ctrl, 0.55); net.obj.renderOrder = 6; scene.add(net.obj);
      var netG = new T.BufferGeometry(); netG.setAttribute('position', new T.BufferAttribute(new Float32Array(3000 * 3), 3));
      var netP = new T.Points(netG, new T.PointsMaterial({ color: C.ctrl, size: 5, sizeAttenuation: false })); netP.frustumCulled = false; netP.renderOrder = 7; scene.add(netP);
      function setNet(patch) {
        var nu = patch.nu, nv = patch.nv, P = patch.P, ns = Math.max(1, Math.ceil(Math.max(nu, nv) / 50)), I = every(nu, ns), J = every(nv, ns), A = net.arr, n = 0, i, j, a, b, o;
        function seg(p, q) { A[n++] = P[p]; A[n++] = P[p + 1]; A[n++] = P[p + 2]; A[n++] = P[q]; A[n++] = P[q + 1]; A[n++] = P[q + 2]; }
        for (b = 0; b < J.length; b++) for (j = J[b], i = 0; i < nu - 1; i++) seg((j * nu + i) * 3, (j * nu + i + 1) * 3);
        for (a = 0; a < I.length; a++) for (i = I[a], j = 0; j < nv - 1; j++) seg((j * nu + i) * 3, ((j + 1) * nu + i) * 3);
        net.set(n / 3);
        var Q = netG.attributes.position.array, m = 0;
        for (b = 0; b < J.length; b++) for (a = 0; a < I.length; a++) { o = (J[b] * nu + I[a]) * 3; Q[m++] = P[o]; Q[m++] = P[o + 1]; Q[m++] = P[o + 2]; }
        netG.setDrawRange(0, m / 3); netG.attributes.position.needsUpdate = true;
        netP.material.size = ns > 1 ? 3 : nu * nv > 400 ? 4 : 6;
        net.mat.opacity = ns > 1 ? 0.42 : nu * nv > 400 ? 0.5 : 0.75;
        st.thin = ns > 1;
        // a net that lies on the surface (dense, interpolating) would flicker in and out of the Gaussians: while it shows,
        // their depth is written one sigma back (gl.js's pre-pass variant), as the manifold module does
        var k = field.mesh.userData.ktSplat; if (k) { k.front = k.front || k.core; k.core = st.thin ? k.depth : k.front; }
      }
      function showNet(v) { net.obj.visible = netP.visible = !!v; var k = field.mesh.userData.ktSplat; if (k && !v) { k.front = k.front || k.core; k.core = k.front; } }

      /* ================= one Gaussian under a magnifier (init3dgs) ================= */
      var cam2 = new T.PerspectiveCamera(8, 1, 0.005, 60);
      var probe = (function () {
        var lines = G.lines(6, C.text, 0.85), ringG = new T.BufferGeometry(), dots = new T.BufferGeometry();
        ringG.setAttribute('position', new T.BufferAttribute(new Float32Array(64 * 3), 3));
        dots.setAttribute('position', new T.BufferAttribute(new Float32Array(4 * 3), 3));
        var ring = new T.LineLoop(ringG, new T.LineBasicMaterial({ color: C.splat, transparent: true, opacity: 0.95, depthTest: false }));
        var pts = new T.Points(dots, new T.PointsMaterial({ color: C.text, size: 6, sizeAttenuation: false, depthTest: false, transparent: true }));
        lines.mat.depthTest = false;
        [lines.obj, ring, pts].forEach(function (o) { o.frustumCulled = false; o.renderOrder = 12; o.visible = false; scene.add(o); });
        // where to look: well-observed points on the front of the owl, from its feet to its head
        var cand = [], i, P = SF.P;
        for (i = 0; i < NP; i++) {
          var x = P[i * 3], y = P[i * 3 + 1], z = P[i * 3 + 2];
          if (SF.track[i] >= 8 && Math.abs(x) < 0.7 && y > 0.5 && y < 3.2 && z > 0.3 && z < 0.9) cand.push(i);
        }
        cand.sort(function (a, b) { return P[a * 3 + 1] - P[b * 3 + 1]; });
        var list = [], m = 9;
        for (i = 0; i < m && cand.length; i++) list.push(cand[Math.floor((i + 0.5) / m * cand.length)]);
        list = [4, 1, 7, 2, 6, 0, 8, 3, 5].map(function (k) { return list[k]; }).filter(function (k) { return k != null; });   // jump around the owl
        return { lines: lines, ring: ring, pts: pts, list: list, i: 0, t: 0, show: function (v) { lines.obj.visible = ring.visible = pts.visible = v; } };
      })();
      var inset = ui.el('div', 'pl-inset'), insetLab = ui.el('span'), mark = ui.el('div', 'pl-mark');
      inset.appendChild(insetLab); root.appendChild(inset); root.appendChild(mark);
      function probeAim() {                        // put the magnifier's camera on the probe, looking the way the main camera looks
        var I = initGauss(), p = probe.list[probe.i], s = I.sig[p], P = SF.P, e = cam.matrixWorld.elements;
        var fx = -e[8], fy = -e[9], fz = -e[10], c = [P[p * 3], P[p * 3 + 1], P[p * 3 + 2]], dist = 7 * s / Math.tan(cam2.fov / 2 * Math.PI / 180);
        cam2.position.set(c[0] - fx * dist, c[1] - fy * dist, c[2] - fz * dist); cam2.up.set(e[4], e[5], e[6]); cam2.lookAt(c[0], c[1], c[2]); cam2.updateMatrixWorld();
        var ra = probe.ring.geometry.attributes.position, k, a, rx = e[0], ry = e[1], rz = e[2], ux = e[4], uy = e[5], uz = e[6];
        for (k = 0; k < 64; k++) { a = k / 64 * Math.PI * 2; ra.array[k * 3] = c[0] + s * (Math.cos(a) * rx + Math.sin(a) * ux); ra.array[k * 3 + 1] = c[1] + s * (Math.cos(a) * ry + Math.sin(a) * uy); ra.array[k * 3 + 2] = c[2] + s * (Math.cos(a) * rz + Math.sin(a) * uz); }
        ra.needsUpdate = true;
        var L = probe.lines.arr, D3 = probe.pts.geometry.attributes.position.array, q, nb;
        D3.set(c, 0);
        for (q = 0; q < 3; q++) { nb = I.near[p * 3 + q]; L.set(c, q * 6); L.set([P[nb * 3], P[nb * 3 + 1], P[nb * 3 + 2]], q * 6 + 3); D3.set([P[nb * 3], P[nb * 3 + 1], P[nb * 3 + 2]], q * 3 + 3); }
        probe.lines.set(6); probe.pts.geometry.attributes.position.needsUpdate = true;
        return { c: c, s: s, p: p, dist: dist };
      }

      /* ================= chrome ================= */
      var parts = [];
      function only(el, ids) { parts.push({ el: el, on: ids.split(' ') }); return el; }
      var hint = ui.el('div', 'kt-hint'); root.appendChild(hint);
      var what = only(ui.el('div', 'pl-what'), 'levels'); root.appendChild(what);
      var pend = ui.el('div', 'pl-pend'); root.appendChild(pend);
      var note = ui.el('div', 'pl-note'); root.appendChild(note);
      var tagL = ui.el('div', 'pl-side'), tagR = ui.el('div', 'pl-side'), bar = ui.el('div', 'pl-divide');
      root.appendChild(tagL); root.appendChild(tagR); root.appendChild(bar);
      var panel = ui.panel(root, { pos: 'tr', title: 'scan 122 · from photographs to charts' });
      function row(label, ids, tone, more) { var r = ui.readout(panel, { label: label, tone: tone }); only(r.el, ids); if (more) r.el.classList.add('pl-more'); return r; }
      /* photos */
      var rdPh = row('photographs', 'photos'), rdFo = row('focal length · field of view', 'photos', '', true), rdDi = row('distance to the owl', 'photos'), rdPo = row('poses', 'photos', '', true);
      /* sfm */
      var rdK = row('cameras added', 'sfm'), rdN = row('points triangulated', 'sfm'), rdE = row('reprojection error, median', 'sfm', 'err'), rdTr = row('track length', 'sfm', '', true);
      var slK = ui.slider(panel, { label: 'cameras', min: 1, max: NC, step: 1, value: NC, fmt: function (v) { return v + ' / ' + NC; },
        onInput: function (v) { st.play = false; st.k = v; } });
      only(slK.el, 'sfm');
      var rowBtn = only(ui.el('div', 'kt-row pl-btns'), 'photos sfm'); panel.appendChild(rowBtn);
      ui.button(rowBtn, { label: 'replay', onClick: function () { st.k = 0; st.play = true; setRays(-1); } });
      /* init3dgs */
      var rdG = row('Gaussians', 'init3dgs', 'splat'), rdS = row('σ, median', 'init3dgs'), rdO = row('opacity', 'init3dgs'), rdNu = row('numbers stored', 'init3dgs', '', true);
      var segO = ui.seg(panel, { label: 'opacity', value: 'init', options: [{ v: 'init', label: '0.1 (init)' }, { v: 'one', label: '1' }],
        onChange: function (v) { st.opa = v; GS.mat.uniforms.uOpa.value = v === 'one' ? 1 : -1; } });
      only(segO.el, 'init3dgs');
      /* initspline */
      var rdCP = row('control points', 'initspline', 'ctrl'), rdFit = row('fitted to', 'initspline'), rdTo = row('to those points', 'initspline', 'err'),
          rdTd = row('to the dense surface', 'initspline', 'err'), rdRf = row('same net, dense points', 'initspline', '', true), rdNs = row('numbers stored', 'initspline', '', true);
      var rdGs = row('Gaussians drawn', 'initspline levels', 'splat');
      var slN = ui.slider(panel, { label: 'Gaussians', min: 2000, max: CAPS - 2000, step: 1000, value: st.nInit, fmt: function (v) { return (v / 1000).toFixed(0) + 'k'; },
        onInput: function (v) { st.nInit = v; writeInit(); } });
      only(slN.el, 'initspline');
      var rowT = only(ui.el('div', 'kt-row pl-btns'), 'initspline levels'); panel.appendChild(rowT);
      var tgN = ui.toggle(rowT, { label: 'control net', value: true, onChange: function (v) { st.nets = v; netFor(); } });
      var tgP = ui.toggle(rowT, { label: 'sparse points', value: true, onChange: function (v) { st.pts = v; sfm.visible = v && st.scene === 'initspline'; } });
      only(tgP.el, 'initspline');
      /* levels */
      var rdL = row('level', 'levels'), rdLe = row('rms · p95 · median', 'levels', 'err'), rdL1 = row('measured points within 1 mm', 'levels', '', true), rdLo = row('nodes on the measured surface', 'levels', '', true);
      var lvOpts = D.levels.map(function (l, i) { return { v: i, label: i ? 'L' + i : 'layout' }; });
      if (OWL) lvOpts.push({ v: D.levels.length, label: 'shipped' });
      var segL = ui.seg(panel, { value: 0, options: lvOpts, onChange: function (v) { levels().auto = false; levelGo(v); } });
      only(segL.el, 'levels');
      /* train */
      var rdRun = TR ? null : row('3DGS run', 'train', 'err'), rdIt, rdTot, rdKept, rdLoss, rdPs, slS = null;
      if (TR) {
        rdIt = row('iteration', 'train'); rdTot = row('Gaussians in the checkpoint', 'train', 'splat'); rdKept = row('kept in the file, drawn', 'train', 'splat');
        rdLoss = row('training loss', 'train'); rdPs = row('PSNR', 'train', 'ok');
        slS = ui.slider(panel, { label: 'snapshot', min: 0, max: TR.snaps.length - 1, step: 1, value: 0,
          fmt: function (v) { return fmt(TR.snaps[v].iter); }, onInput: function (v) { trn.auto = false; snapGo(v); } });
        only(slS.el, 'train');
      }

      /* a small plot, bottom left: new points per camera (sfm), errors per level (levels), the run's log (train) */
      var plot = ui.el('div', 'pl-plot'), plotT = ui.el('div', 'pl-plot-t'), plotB = ui.el('div', 'pl-plot-box');
      plot.appendChild(plotT); plot.appendChild(plotB); root.appendChild(plot); only(plot, 'sfm levels train');
      var pcv = ui.canvas(plotB), plotDirty = true;

      /* ================= photos and sfm: numbers ================= */
      (function () {
        var fx = 2 * Math.atan(IN.width / 2 / IN.fx) * 180 / Math.PI, fy = 2 * Math.atan(IN.height / 2 / IN.fy) * 180 / Math.PI, d0 = Infinity, d1 = 0;
        CAMS.forEach(function (c) { var d = Math.hypot(c.c[0] - TGT[0], c.c[1] - TGT[1], c.c[2] - TGT[2]); d0 = Math.min(d0, d); d1 = Math.max(d1, d); });
        rdPh.set(NC + ' · ' + IN.width + ' × ' + IN.height + ' px');
        rdFo.set(fmt(IN.fx) + ' px · ' + fx.toFixed(1) + '° × ' + fy.toFixed(1) + '°');
        rdDi.set((d0 * MM / 1000).toFixed(2) + ' – ' + (d1 * MM / 1000).toFixed(2) + ' m');
        rdPo.set('calibrated, given');
        rdTr.set('median ' + SF.stats.track_median + ' · max ' + SF.stats.track_max + ' photos');
        var I = D.init, nn = I.nu * I.nv;
        rdCP.set(I.nu + ' × ' + I.nv + ' = ' + fmt(nn));
        rdFit.set(fmt(I.fit.points) + ' points · ' + fmt(I.fit.inliers) + ' inliers');
        rdTo.set('rms ' + f2(I.to_sparse.inliers.rms_mm) + ' · median ' + f2(I.to_sparse.inliers.median_mm) + ' mm');
        rdTd.set('rms ' + f2(I.to_dense.rms_mm) + ' · median ' + f2(I.to_dense.median_mm) + ' mm');
        rdRf.set('rms ' + f2(I.reference.rms_mm) + ' mm');
        rdNs.set(fmt(nn) + ' × 3 = ' + fmt(nn * 3));
        rdO.set('0.1 each');
        rdG.set(fmt(NP));
        rdNu.set(fmt(NP) + ' × ' + NUM + ' = ' + fmt(NP * NUM));
        if (rdRun) rdRun.set('pending: no snapshots loaded');
      })();
      function paintK() {
        var k = Math.max(0, Math.min(NC, Math.ceil(st.k - 1e-9))), m = medErr(k);
        rdK.set(k + ' / ' + NC); rdN.set(fmt(cum[k]) + ' / ' + fmt(NP)); rdE.set(isFinite(m) ? f2(m) + ' px' : '–');
        if (st.play) slK.set(Math.max(1, k));
        if (k !== st.kShown) { st.kShown = k; plotDirty = true; }
      }
      function paintLevel() {
        var L = levels(), s = L.S[L.i], d = s.d;
        rdL.set((s.name === 'shipped' ? 'shipped' : s.name) + ' · ' + s.patch.nu + ' × ' + s.patch.nv);
        rdLe.set(f2(d.rms_mm) + ' · ' + (d.p95_mm < 0.1 ? d.p95_mm.toFixed(3) : f2(d.p95_mm)) + ' · ' + (d.median_mm < 0.1 ? d.median_mm.toFixed(3) : f2(d.median_mm)) + ' mm');
        rdL1.set(pc(d.within_1mm)); rdLo.set(pc(d.on_measured));
        segL.set(L.i);
        what.innerHTML = '<b>' + (s.name === 'shipped' ? 'the shipped net' : s.name) + '</b> · ' + d.what;
        if (st.split) tagR.innerHTML = sideR();
      }

      /* ================= train ================= */
      var trn = { i: 0, t: 0, auto: true, fade: 1, want: -1 };
      function logAt(it) {                         // the run's logged loss and PSNR at iteration it (the nearest logged one at or before it)
        var L = TR.log || {}, I = L.iter || [], o = { loss: null, li: null, psnr: null, pi: null }, j;
        for (j = 0; j < I.length; j++) {
          if (I[j] > it) break;
          if (L.loss && finite(L.loss[j])) { o.loss = L.loss[j]; o.li = I[j]; }
          if (L.psnr && finite(L.psnr[j])) { o.psnr = L.psnr[j]; o.pi = I[j]; }
        }
        return o;
      }
      function snapGo(k) { trn.want = k; trn.t = 0; if (slS) slS.set(k); }
      function paintSnap() {
        var s = TR.snaps[trn.i], lg = logAt(s.iter);
        rdIt.set(fmt(s.iter)); rdTot.set(fmt(s.ntotal)); rdKept.set(fmt(s.n));
        rdLoss.set(lg.loss == null ? '–' : lg.loss.toPrecision(3) + (lg.li !== s.iter ? ' @ ' + fmt(lg.li) : ''));
        rdPs.set(lg.psnr == null ? '–' : lg.psnr.toFixed(2) + ' dB' + (lg.pi !== s.iter ? ' @ ' + fmt(lg.pi) : ''));
        plotDirty = true;
      }
      function trainStep(dt) {
        if (!TR) { levelsStep(dt); return; }
        if (trn.want >= 0) {                       // a short dissolve: out, swap, in
          trn.fade = Math.max(0, trn.fade - dt / 0.22);
          if (trn.fade === 0) { trn.i = trn.want; trn.want = -1; gsSnap(trn.i); paintSnap(); }
        } else trn.fade = Math.min(1, trn.fade + dt / 0.22);
        GS.mat.uniforms.uAlpha.value = trn.fade;
        if (trn.auto && trn.want < 0) { trn.t += dt; if (trn.t > (trn.i === TR.snaps.length - 1 ? SNAP * 1.8 : SNAP)) snapGo((trn.i + 1) % TR.snaps.length); }
      }
      function sideL() { return '<b>3DGS</b> · iteration 0 of a run that has not been made yet<br>the initialisation, untrained: ' + fmt(NP) + ' Gaussians, opacity 0.1'; }
      function sideR() {
        var L = levels(), s = L.S[L.i];
        return '<b>stand-in</b> · our chart, coarse to fine: <em>' + (s.name === 'shipped' ? 'the shipped net' : s.name) + '</em> · ' + s.patch.nu + ' × ' + s.patch.nv +
               '<br>a geometric fit to the depth maps, not training · rms ' + f2(s.d.rms_mm) + ' mm';
      }

      /* ================= plots ================= */
      function drawPlot() {
        plotDirty = false;
        var s = pcv.fit(), g = pcv.ctx, w = s.w, h = s.h, id = st.scene;
        g.clearRect(0, 0, w, h); g.font = '500 9px "JetBrains Mono", ui-monospace, monospace'; g.textBaseline = 'alphabetic';
        if (id === 'sfm') plotSfm(g, w, h); else if (id === 'levels') plotLevels(g, w, h); else if (id === 'train') { if (TR) plotTrain(g, w, h); else plotLevels(g, w, h); }
      }
      function plotSfm(g, w, h) {                  // the points each camera completes, and the running total
        plotT.textContent = 'points completed by each camera';
        var k = Math.ceil(st.k - 1e-9), mx = 0, j, x0 = 4, bw = (w - 8) / NC, H = h - 14;
        for (j = 0; j < NC; j++) mx = Math.max(mx, byCam[j].length);
        for (j = 0; j < NC; j++) {
          var bh = byCam[j].length / mx * (H - 4);
          g.fillStyle = j < k ? (j === k - 1 ? C.text : 'rgba(221,229,240,.55)') : 'rgba(143,164,192,.18)';
          g.fillRect(x0 + j * bw + 0.5, H - bh, Math.max(1, bw - 1), bh);
        }
        g.strokeStyle = 'rgba(221,229,240,.9)'; g.lineWidth = 1.2; g.beginPath();
        for (j = 0; j <= Math.min(k, NC); j++) { var x = x0 + j * bw, y = H - cum[j] / NP * (H - 4); if (j) g.lineTo(x, y); else g.moveTo(x, y); }
        g.stroke();
        g.fillStyle = C.mute; g.fillText('camera 1', x0, h - 2); g.textAlign = 'right'; g.fillText(NC + '', w - 4, h - 2);
        g.fillStyle = C.dim; g.fillText('running total: ' + fmt(cum[Math.min(k, NC)]), w - 4, 10); g.textAlign = 'left';
      }
      function plotLevels(g, w, h) {               // rms, p95, median of every level on a log axis; level 4 has numbers only
        plotT.textContent = 'distance to the measured surface, mm (log)';
        var rows = D.levels.map(function (l, i) { return { lab: i ? 'L' + i : 'lay', d: l, st: i }; });
        rows.push({ lab: 'L4', d: D.level_last, st: -1 });
        if (OWL) rows.push({ lab: 'ship', d: D.shipped, st: D.levels.length });
        var L = levels(), cur = L.i, lo = Math.log10(0.008), hi = Math.log10(3), x0 = 30, x1 = w - 8, H = h - 13, top = 12;
        function X(i) { return x0 + (i + 0.5) / rows.length * (x1 - x0); }
        function Y(v) { return top + (1 - (Math.log10(Math.max(v, 0.008)) - lo) / (hi - lo)) * (H - top); }
        g.strokeStyle = 'rgba(143,164,192,.18)'; g.lineWidth = 1; g.fillStyle = C.mute;
        [0.01, 0.1, 1].forEach(function (v) { var y = Y(v); g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); g.fillText(v + '', 2, y + 3); });
        rows.forEach(function (r, i) {
          if (r.st === cur) { g.fillStyle = 'rgba(221,229,240,.09)'; g.fillRect(X(i) - (x1 - x0) / rows.length / 2, top - 4, (x1 - x0) / rows.length, H - top + 6); }
          g.fillStyle = r.st === cur ? C.text : C.mute; g.textAlign = 'center'; g.fillText(r.lab, X(i), h - 2);
        });
        g.textAlign = 'left';
        [['rms_mm', C.err, 'rms', 1.6], ['p95_mm', '#ff8a80', 'p95', 1.1], ['median_mm', '#ffcdd2', 'median', 1.1]].forEach(function (s, k) {
          g.strokeStyle = s[1]; g.lineWidth = s[3]; g.beginPath();
          rows.forEach(function (r, i) { var y = Y(r.d[s[0]]); if (i) g.lineTo(X(i), y); else g.moveTo(X(i), y); });
          g.stroke();
          rows.forEach(function (r, i) {
            var y = Y(r.d[s[0]]); g.beginPath(); g.arc(X(i), y, 2.4, 0, Math.PI * 2);
            if (r.st < 0) { g.fillStyle = '#07090f'; g.fill(); g.strokeStyle = s[1]; g.stroke(); } else { g.fillStyle = s[1]; g.fill(); }
          });
          g.fillStyle = s[1]; g.fillText(s[2], x0 + 4 + k * 46, 9);
        });
        g.fillStyle = C.mute; g.textAlign = 'right'; g.fillText('○ no net', x1, 9); g.textAlign = 'left';
      }
      function plotTrain(g, w, h) {                // the run's loss (log) and PSNR against the iteration (log), snapshots marked
        plotT.textContent = 'the run: training loss and PSNR';
        var L = TR.log || {}, I = L.iter || [], maxIt = 1, j, x0 = 26, x1 = w - 26, top = 12, H = h - 13;
        TR.snaps.forEach(function (s) { maxIt = Math.max(maxIt, s.iter); }); I.forEach(function (v) { maxIt = Math.max(maxIt, v); });
        function X(it) { return x0 + Math.log10(1 + it) / Math.log10(1 + maxIt) * (x1 - x0); }
        var ls = [], ps = [];
        for (j = 0; j < I.length; j++) { if (L.loss && finite(L.loss[j]) && L.loss[j] > 0) ls.push([I[j], L.loss[j]]); if (L.psnr && finite(L.psnr[j])) ps.push([I[j], L.psnr[j]]); }
        g.strokeStyle = 'rgba(143,164,192,.18)'; g.lineWidth = 1;
        TR.snaps.forEach(function (s, k) { var x = X(s.iter); g.strokeStyle = k === trn.i ? C.text : 'rgba(143,164,192,.25)'; g.beginPath(); g.moveTo(x, top - 2); g.lineTo(x, H); g.stroke(); });
        if (ls.length > 1) {
          var a = Infinity, b = -Infinity; ls.forEach(function (p) { a = Math.min(a, Math.log10(p[1])); b = Math.max(b, Math.log10(p[1])); }); if (b - a < 1e-6) { a -= 0.5; b += 0.5; }
          g.strokeStyle = C.err; g.lineWidth = 1.3; g.beginPath();
          ls.forEach(function (p, i) { var y = top + (1 - (Math.log10(p[1]) - a) / (b - a)) * (H - top); if (i) g.lineTo(X(p[0]), y); else g.moveTo(X(p[0]), y); });
          g.stroke(); g.fillStyle = C.err; g.fillText('loss', 2, 9);
        }
        if (ps.length) {
          var c = Infinity, d = -Infinity; ps.forEach(function (p) { c = Math.min(c, p[1]); d = Math.max(d, p[1]); }); if (d - c < 1) { c -= 0.5; d += 0.5; }
          g.fillStyle = C.ok; g.strokeStyle = C.ok; g.lineWidth = 1.2; g.beginPath();
          ps.forEach(function (p, i) { var y = top + (1 - (p[1] - c) / (d - c)) * (H - top); if (i) g.lineTo(X(p[0]), y); else g.moveTo(X(p[0]), y); });
          if (ps.length > 1) g.stroke();
          ps.forEach(function (p) { var y = top + (1 - (p[1] - c) / (d - c)) * (H - top); g.beginPath(); g.arc(X(p[0]), y, 2.3, 0, Math.PI * 2); g.fill(); });
          g.textAlign = 'right'; g.fillText('PSNR ' + c.toFixed(0) + '–' + d.toFixed(0) + ' dB', w - 2, 9); g.textAlign = 'left';
        }
        g.fillStyle = C.mute; g.fillText('0', x0 - 3, h - 2); g.textAlign = 'right'; g.fillText(fmt(maxIt), x1 + 22, h - 2); g.textAlign = 'center'; g.fillText('iteration (log)', (x0 + x1) / 2, h - 2); g.textAlign = 'left';
      }

      /* ================= cameras: views fitted to the stage ================= */
      var orbit = G.orbit(cam, R.canvas, { embed: env.embed, minRadius: 0.6, maxRadius: 90, target: TGT });
      var fitCam = new T.PerspectiveCamera(36, 1, 0.02, 400), V3 = new T.Vector3();
      var OWLBOX = [[-1.0, 0.0, -0.3], [0.95, 3.5, 0.7]], box = function (b) { var o = [], i; for (i = 0; i < 8; i++) o.push([b[i & 1][0], b[(i >> 1) & 1][1], b[(i >> 2) & 1][2]]); return o; };
      var RIGPTS = CAMS.map(function (c) { return c.c; }).concat(box(OWLBOX)), OWLPTS = box(OWLBOX);
      var RIGC = (function () { var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity]; RIGPTS.forEach(function (p) { for (var q = 0; q < 3; q++) { lo[q] = Math.min(lo[q], p[q]); hi[q] = Math.max(hi[q], p[q]); } }); return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]; })();
      function place(v, r) {
        var s = Math.sin(v.ph), c = Math.cos(v.ph);
        fitCam.position.set(v.tx + r * s * Math.sin(v.th), v.ty + r * c, v.tz + r * s * Math.cos(v.th)); fitCam.lookAt(v.tx, v.ty, v.tz); fitCam.updateMatrixWorld();
      }
      /* the distance at which every point lies within m of the frame's edge, with the target moved so that their
         projection is centred in the frame (v.tx, v.ty, v.tz change; returns the distance) */
      function fitView(pts, v, aspect, m) {
        var lo, hi, it, r = 10, ok, i, pass, b;
        fitCam.aspect = aspect; fitCam.updateProjectionMatrix();
        for (pass = 0; pass < 3; pass++) {
          lo = 0.5; hi = 150;
          for (it = 0; it < 26; it++) {
            r = (lo + hi) / 2; ok = true; place(v, r);
            for (i = 0; i < pts.length && ok; i++) { V3.set(pts[i][0], pts[i][1], pts[i][2]).project(fitCam); if (V3.z > 1 || Math.abs(V3.x) > m || Math.abs(V3.y) > m) ok = false; }
            if (ok) hi = r; else lo = r;
          }
          r = hi; place(v, r); b = [Infinity, -Infinity, Infinity, -Infinity];
          for (i = 0; i < pts.length; i++) { V3.set(pts[i][0], pts[i][1], pts[i][2]).project(fitCam); b[0] = Math.min(b[0], V3.x); b[1] = Math.max(b[1], V3.x); b[2] = Math.min(b[2], V3.y); b[3] = Math.max(b[3], V3.y); }
          var e = fitCam.matrixWorld.elements, ty = Math.tan(fitCam.fov / 2 * Math.PI / 180) * r, tx = ty * aspect, cx = (b[0] + b[1]) / 2 * tx, cy = (b[2] + b[3]) / 2 * ty;
          v.tx += e[0] * cx + e[4] * cy; v.ty += e[1] * cx + e[5] * cy; v.tz += e[2] * cx + e[6] * cy;     // shift along the camera's right and up
        }
        return r;
      }
      function aspect() { var q = st.split ? st.rects[0] : null; return q ? q[2] / q[3] : st.w / Math.max(1, st.h); }
      function view(name) {
        var v;
        if (name === 'rig') { v = { th: 0.62, ph: 0.98, tx: RIGC[0], ty: RIGC[1] - 0.6, tz: RIGC[2] }; v.r = fitView(RIGPTS, v, aspect(), 0.86); }
        else { v = { th: 0.30, ph: 1.24, tx: TGT[0], ty: TGT[1] + 0.05, tz: TGT[2] }; v.r = fitView(OWLPTS, v, aspect(), st.split ? 0.84 : 0.74); }
        return v;
      }
      function home() {                           // sfm: from the rig in towards the cloud as it fills
        var v = view(st.sc.view);
        if (st.scene === 'sfm') {
          var o = view('owl'), f = KT.ease(KT.clamp(st.k / NC * 1.25, 0, 1)) * 0.75, k;
          o.r *= 2.1; for (k in v) v[k] += (o[k] - v[k]) * f;
        }
        return v;
      }

      /* ================= scenes ================= */
      function netFor() {                          // which net shows: the chart of the step
        var on = st.nets && !st.split && (st.scene === 'initspline' || st.scene === 'levels');
        if (on) setNet(st.scene === 'initspline' ? initPatch : levels().S[levels().i].patch);
        showNet(on);
      }
      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        var first = !st.scene;
        st.scene = id; st.sc = sc; st.t = 0; st.split = id === 'train' && !TR;
        root.className = root.className.replace(/\bpl-s-\w+/g, '').trim() + ' pl-s-' + id;
        parts.forEach(function (p) { p.el.style.display = p.on.indexOf(id) >= 0 ? '' : 'none'; });
        // what is drawn
        rig.show(id === 'photos' || id === 'sfm'); rig.set(NC, false);
        sfm.visible = id === 'sfm' || (id === 'initspline' && st.pts);
        sfm.material.uniforms.uK.value = NC; sfm.material.uniforms.uDim.value = id === 'initspline' ? 0.8 : 1; sfm.material.uniforms.uSize.value = (id === 'initspline' ? 2.2 : 2.6) * pr();
        rays.obj.visible = id === 'sfm'; probe.show(false);
        GS.mesh.visible = id === 'init3dgs' || id === 'train';
        field.mesh.visible = id === 'initspline' || id === 'levels' || st.split;
        GS.mat.uniforms.uOpa.value = id === 'init3dgs' && st.opa === 'one' ? 1 : -1; GS.mat.uniforms.uAlpha.value = 1;
        inset.style.display = mark.style.display = 'none';
        if (id === 'photos' || id === 'sfm') { st.k = 0; st.play = true; setRays(-1); st.kShown = -1; }
        if (id === 'init3dgs' || (id === 'train' && !TR)) { gsInit(); rdS.set(f2(INIT.med * MM) + ' mm'); probe.i = 0; probe.t = 0; }
        if (id === 'initspline') writeInit();
        if (id === 'levels' || st.split) { var L = levels(); L.auto = true; L.t = 0; levelGo(0); L.f = 1; L.dirty = true; L.fromFlat = 0; }
        if (id === 'train' && TR) { trn.auto = true; trn.t = 0; trn.want = -1; trn.fade = 1; trn.i = 0; gsSnap(0); paintSnap(); if (slS) slS.set(0); }
        netFor();
        // words
        var c = sc.cap;
        if (id === 'train') c = TR ? '<b>3DGS training</b> · one real run of the reference implementation on the same photographs and points<br>snapshots, one after another: Gaussians are cloned and pruned between them, so they cannot be blended'
                                   : '<b>3DGS training</b> · <span class="pl-red">the run is pending</span>: no snapshots are loaded<br>left, the 3DGS initialisation, untrained · right, a stand-in: the levels of the offline geometric fit, which is not training';
        cap.innerHTML = c;
        pend.style.display = id === 'train' && !TR ? '' : 'none';
        pend.innerHTML = '<b>real 3DGS run: pending</b><br>tools/train trains the reference implementation on a GPU; this step replays its snapshots once they are exported';
        hint.textContent = { photos: 'drag to orbit', sfm: 'drag the cameras slider to scrub', init3dgs: 'drag to orbit', initspline: 'drag to orbit', levels: 'pick a level, or let them run', train: TR ? 'drag the snapshot slider' : '' }[id];
        tagL.style.display = tagR.style.display = bar.style.display = st.split ? '' : 'none';
        if (st.split) { tagL.innerHTML = sideL(); tagR.innerHTML = sideR(); }
        if (id === 'train' && TR) note.innerHTML = TR.note ? 'the run: ' + TR.note : '';
        note.style.display = id === 'train' && TR && TR.note ? '' : 'none';
        layout();
        orbit.touched = false; orbit.to(home(), first);
        plotDirty = true;
        if (env.opts.onStats) env.opts.onStats(stats());
      }
      function stats() {
        var k = Math.ceil(st.k - 1e-9);
        return { scene: st.scene, cameras: NC, added: k, points: cum[Math.min(NC, Math.max(0, k))], gaussians3dgs: GS.mesh.visible ? GS.n : 0,
                 derived: field.mesh.visible ? st.drawn : 0, sigmaMedianMm: INIT ? INIT.med * MM : null, knnMs: INIT ? INIT.ms : null,
                 level: LV ? LV.S[LV.i].name : null, train: !!TR, snapshot: TR ? trn.i : null, split: st.split };
      }

      /* ================= frame ================= */
      function layout() {
        var w = st.w, h = st.h;
        st.narrow = w < 560;
        root.classList.toggle('pl-narrow', st.narrow);
        if (st.split) {
          var wide = w / h > 1.25;
          st.rects = wide ? [[0, 0, w / 2, h], [w / 2, 0, w / 2, h]] : [[0, h / 2, w, h / 2], [0, 0, w, h / 2]];
          var s = bar.style; s.left = wide ? w / 2 + 'px' : '0'; s.top = wide ? '0' : h / 2 + 'px'; s.width = wide ? '1px' : '100%'; s.height = wide ? '100%' : '1px';
          tagL.style.cssText = wide ? 'left:10px;bottom:150px;max-width:' + (w / 2 - 20) + 'px' : 'left:10px;top:' + (h / 2 - 10) + 'px;transform:translateY(-100%);max-width:' + (w - 20) + 'px';
          tagR.style.cssText = wide ? 'left:' + (w / 2 + 10) + 'px;bottom:12px;max-width:' + (w / 2 - 20) + 'px' : 'left:10px;bottom:10px;max-width:' + (w - 20) + 'px';
        } else st.rects = null;
        plotDirty = true;
      }
      function sortIfNeeded(t) {
        var e = cam.matrixWorld.elements, key = (e[8] * 97).toFixed(1) + (e[9] * 97).toFixed(1) + (e[10] * 97).toFixed(1) + GS.what + GS.n;
        if (key !== st.sortKey && (t - st.sortT > 0.08 || !st.sortKey)) { st.sortKey = key; st.sortT = t; GS.sort(cam); }
      }
      env.loop(function (t, dt) {
        if (st.w < 4 || st.h < 4) return;
        var id = st.scene;
        st.t += dt;
        /* the scene's own clock */
        if (id === 'photos' || id === 'sfm') {
          if (st.play) st.k = Math.min(NC, st.k + dt * RATE[id]);
          rig.set(st.k, id === 'sfm'); sfm.material.uniforms.uK.value = id === 'sfm' ? st.k : NC;
          if (id === 'sfm') {
            var j = Math.ceil(st.k - 1e-9) - 1;
            if (j !== st.rayJ) setRays(j);
            st.rayT += dt; rays.mat.opacity = 0.5 * Math.exp(-st.rayT * 1.4); rays.obj.visible = j >= 0;
            paintK();
          }
        }
        if (id === 'levels') levelsStep(dt);
        if (id === 'train') trainStep(dt);
        if (id === 'init3dgs' && probe.list.length) { probe.t += dt; if (probe.t > 3.4) { probe.t = 0; probe.i = (probe.i + 1) % probe.list.length; } }
        /* camera */
        if (!orbit.touched) { var v = home(); v.th += 0.12 * Math.sin(st.t * 0.17); orbit.to(v); }
        orbit.update(dt, 2.4);
        if (GS.mesh.visible) sortIfNeeded(t);
        if (plotDirty && plot.style.display !== 'none') drawPlot();
        /* draw */
        var rr = R.renderer, p = pr();
        if (st.split) {
          rr.setScissorTest(true);
          for (var k = 0; k < 2; k++) {
            var q = st.rects[k]; rr.setViewport(q[0], q[1], q[2], q[3]); rr.setScissor(q[0], q[1], q[2], q[3]);
            cam.aspect = q[2] / q[3]; cam.updateProjectionMatrix();
            GS.mesh.visible = k === 0; field.mesh.visible = k === 1;
            GS.prep(cam, q[2] * p, q[3] * p);
            G.draw(R, scene, cam);
          }
          rr.setScissorTest(false); rr.setViewport(0, 0, st.w, st.h);
          GS.mesh.visible = field.mesh.visible = true;
          return;
        }
        cam.aspect = st.w / st.h; cam.updateProjectionMatrix();
        GS.prep(cam, st.w * p, st.h * p);
        if (id === 'init3dgs') probe.show(false);
        G.draw(R, scene, cam);
        if (id === 'init3dgs' && probe.list.length) drawInset(p);
      });
      function drawInset(p) {                      // the magnifier: the same Gaussians seen through a narrow lens on one of them
        var rr = R.renderer, iw = Math.round(Math.min(st.w * (st.narrow ? 0.46 : 0.3), 300)), ih = Math.round(iw * 0.78), ix = 12;
        var top = st.narrow ? Math.min(st.h - ih - 12, 92) : st.h - ih - 12, iy = st.h - top - ih;      // CSS top of the inset, and its GL y (from the bottom)
        var a = probeAim(), e = cam.matrixWorld.elements;
        var zMain = (a.c[0] - cam.position.x) * -e[8] + (a.c[1] - cam.position.y) * -e[9] + (a.c[2] - cam.position.z) * -e[10];
        var mag = (ih / (2 * a.dist * Math.tan(cam2.fov / 2 * Math.PI / 180))) / (st.h / (2 * zMain * Math.tan(cam.fov / 2 * Math.PI / 180)));
        cam2.aspect = iw / ih; cam2.updateProjectionMatrix();
        rr.setScissorTest(true); rr.setViewport(ix, iy, iw, ih); rr.setScissor(ix, iy, iw, ih);
        GS.prep(cam2, iw * p, ih * p); probe.show(true);
        G.draw(R, scene, cam2);
        probe.show(false); rr.setScissorTest(false); rr.setViewport(0, 0, st.w, st.h); GS.prep(cam, st.w * p, st.h * p);
        var I = INIT, P = SF.P, d = [0, 1, 2].map(function (q) { var n = I.near[a.p * 3 + q]; return Math.hypot(P[n * 3] - a.c[0], P[n * 3 + 1] - a.c[1], P[n * 3 + 2] - a.c[2]) * MM; });
        var s = inset.style; s.display = ''; s.left = ix + 'px'; s.top = top + 'px'; s.width = iw + 'px'; s.height = ih + 'px';
        insetLab.innerHTML = '<b>×' + Math.round(mag) + '</b> · one Gaussian: <em>σ ' + f2(a.s * MM) + ' mm</em><br>its 3 nearest: ' + d.map(f2).join(', ') + ' mm';
        var sp = G.project(cam, a.c[0], a.c[1], a.c[2], st.w, st.h);
        mark.style.display = sp ? '' : 'none'; if (sp) { mark.style.left = sp[0] + 'px'; mark.style.top = sp[1] + 'px'; }
      }

      setScene(env.opts.scene && SCENES[env.opts.scene] ? env.opts.scene : 'photos');
      return {
        setScene: setScene, orbit: orbit, stats: stats,
        resize: function (w, h) {
          R.resize(w, h); st.w = w; st.h = h; layout();
          if (!orbit.touched) orbit.to(home(), true);
          plotDirty = true;
        },
        destroy: function () { R.renderer.dispose(); }
      };
    }
  });

  function addCss() {
    if (document.getElementById('kt-pipeline-css')) return;
    var css = document.createElement('style'); css.id = 'kt-pipeline-css';
    css.textContent =
      '.kt-sim-pipeline .kt-caption{max-width:min(62%,640px);line-height:1.55}' +
      '.kt-sim-pipeline .kt-caption .pl-red{color:var(--err);font-weight:600}' +
      '.kt-sim-pipeline .kt-panel{max-width:min(330px,calc(100% - 24px))}' +
      '.kt-sim-pipeline .kt-panel .kt-readout .kt-val{white-space:normal;text-align:right}' +
      '.kt-sim-pipeline .pl-btns{flex-wrap:wrap;gap:6px}' +
      '.kt-sim-pipeline .kt-panel .kt-seg .kt-seg-box{margin-left:0;width:100%}.kt-sim-pipeline .kt-panel .kt-seg .kt-seg-btn{flex:1;padding:7px 3px}' +
      '.kt-sim-pipeline .pl-plot{position:absolute;z-index:5;left:12px;bottom:12px;width:min(300px,calc(100% - 24px));padding:8px 10px 6px;' +
      'background:var(--panel);border:1px solid var(--line);border-radius:10px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px)}' +
      '.kt-sim-pipeline .pl-plot-t{font:600 9.5px/1 var(--f-mono);letter-spacing:.1em;text-transform:uppercase;color:var(--mute);margin-bottom:6px}' +
      '.kt-sim-pipeline .pl-plot-box{position:relative;height:96px}.kt-sim-pipeline .pl-plot-box canvas{position:absolute;inset:0;width:100%;height:100%}' +
      '.kt-sim-pipeline .pl-what{position:absolute;z-index:4;left:14px;top:58px;max-width:min(40%,420px);pointer-events:none;font:400 12px/1.5 var(--f-ui);color:var(--dim)}' +
      '.kt-sim-pipeline .pl-what b{color:var(--text);font-weight:600}' +
      '.kt-sim-pipeline .pl-side{position:absolute;z-index:4;pointer-events:none;font:500 11px/1.5 var(--f-mono);letter-spacing:.02em;color:var(--dim);' +
      'padding:8px 11px;background:var(--panel);border:1px solid var(--line);border-radius:8px}' +
      '.kt-sim-pipeline .pl-side b{color:var(--text);font-weight:600;letter-spacing:.06em;text-transform:uppercase}.kt-sim-pipeline .pl-side em{font-style:normal;color:var(--splat)}' +
      '.kt-sim-pipeline .pl-divide{position:absolute;z-index:3;background:var(--line2);pointer-events:none}' +
      '.kt-sim-pipeline .pl-pend{position:absolute;z-index:4;left:50%;top:50%;transform:translate(-50%,-50%);pointer-events:none;text-align:center;' +
      'font:500 11px/1.55 var(--f-mono);color:var(--dim);padding:10px 14px;background:rgba(7,9,15,.82);border:1px solid rgba(var(--err-rgb),.55);border-radius:10px;max-width:min(360px,80%)}' +
      '.kt-sim-pipeline .pl-pend b{color:var(--err);font-weight:600;letter-spacing:.08em;text-transform:uppercase}' +
      '.kt-sim-pipeline .pl-note{position:absolute;z-index:4;right:12px;bottom:12px;max-width:min(46%,420px);pointer-events:none;font:500 10px/1.45 var(--f-mono);color:var(--mute);text-align:right}' +
      '.kt-sim-pipeline .pl-inset{position:absolute;z-index:3;pointer-events:none;border:1px solid rgba(var(--splat-rgb),.65);border-radius:6px}' +
      '.kt-sim-pipeline .pl-inset span{position:absolute;left:7px;top:6px;right:6px;font:500 10px/1.4 var(--f-mono);color:var(--dim)}' +
      '.kt-sim-pipeline .pl-inset b{color:var(--text)}.kt-sim-pipeline .pl-inset em{font-style:normal;color:var(--splat)}' +
      '.kt-sim-pipeline .pl-mark{position:absolute;z-index:3;width:22px;height:22px;margin:-11px 0 0 -11px;border:1.5px solid var(--splat);border-radius:50%;pointer-events:none}' +
      '.kt-sim-pipeline.kt-embed .kt-hint{display:none}' +
      '.kt-sim-pipeline.kt-compact .pl-plot,.kt-sim-pipeline.kt-compact .pl-what,.kt-sim-pipeline.kt-compact .pl-note{display:none!important}' +
      '.kt-sim-pipeline.pl-narrow .pl-more,.kt-sim-pipeline.pl-narrow .pl-what,.kt-sim-pipeline.pl-narrow .kt-panel-title{display:none!important}' +
      '.kt-sim-pipeline.pl-narrow .kt-caption{max-width:calc(100% - 24px);font-size:10px}' +
      '.kt-sim-pipeline.pl-narrow .kt-panel{top:auto;bottom:12px;right:12px;min-width:0;width:min(230px,calc(100% - 24px))}' +
      '.kt-sim-pipeline.pl-narrow .pl-plot{bottom:auto;top:84px;width:min(210px,calc(100% - 24px))}.kt-sim-pipeline.pl-narrow .pl-plot-box{height:64px}' +
      '.kt-sim-pipeline.pl-narrow .pl-side{font-size:9.5px;padding:6px 8px}.kt-sim-pipeline.pl-narrow .pl-pend{font-size:10px}';
    document.head.appendChild(css);
  }
})();
