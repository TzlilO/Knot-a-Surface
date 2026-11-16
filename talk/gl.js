/* Knot-a-Surface talk — three.js helpers shared by the WebGL modules (planoid, manifold).
   Needs THREE (talk/vendor/three.min.js) + talk/kit.js. Exposed as KnotTalk.gl. */
(function () {
  'use strict';
  var KT = window.KnotTalk, T = window.THREE;
  var gl = KT.gl = {};

  /* renderer bound to a module root; returns { renderer, canvas, resize(w,h) } */
  /* clear: omit for the theme's stage: on a dark page the dark instrument background (0x07090f, opaque), on a light page
     (KT.theme) a see-through canvas, so the glass card behind it shows; 'transparent' = see-through on any page;
     'opaque' = the theme's KT.color.bg, opaque; or a colour (0xrrggbb or '#rrggbb'), opaque.
     A see-through canvas holds premultiplied colour: normal blending composites right over the page, additive
     blending only ever brightens it (a glow vanishes on a light page). */
  gl.renderer = function (root, camera, clear) {
    if (clear == null || clear === '') clear = KT.theme === 'light' ? 'transparent' : 0x07090f;
    var see = clear === 'transparent';
    var r = new T.WebGLRenderer({ antialias: true, alpha: see, powerPreference: 'high-performance' });
    r.setClearColor(see ? 0x000000 : clear === 'opaque' ? KT.color.bg : clear, see ? 0 : 1);
    r.domElement.className = 'kt-canvas';
    root.insertBefore(r.domElement, root.firstChild);
    function resize(w, h) {
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    return { renderer: r, canvas: r.domElement, resize: resize };
  };

  /* Minimal orbit camera, on the kit's shared gesture handling (KnotTalk.gestures):
       mouse: drag = rotate; wheel zooms on the standalone page, ctrl+wheel (= trackpad pinch) when
              embedded, so the deck keeps the scroll wheel.
       touch: two-finger pinch = zoom, two-finger drag = rotate (yaw + pitch). One finger rotates in
              both axes on the standalone page; embedded it turns horizontally only, so a vertical
              swipe still scrolls the deck.
     pick(e) may return true to claim a pointerdown for the module (e.g. a control-point drag);
     hit(x, y) says whether a touch at client (x, y) would be picked, so it does not scroll the page;
     release(id) drops that drag when a second finger turns the touch into a pinch. */
  gl.orbit = function (camera, dom, o) {
    o = o || {};
    var cur = { th: o.theta || 0.7, ph: o.phi || 1.05, r: o.radius || 6, tx: 0, ty: 0, tz: 0 };
    if (o.target) { cur.tx = o.target[0]; cur.ty = o.target[1]; cur.tz = o.target[2]; }
    var goal = Object.assign({}, cur), api;
    var lim = { r0: o.minRadius || 1.2, r1: o.maxRadius || 14 };
    function place() {
      var s = Math.sin(cur.ph), c = Math.cos(cur.ph);
      camera.position.set(cur.tx + cur.r * s * Math.sin(cur.th), cur.ty + cur.r * c, cur.tz + cur.r * s * Math.cos(cur.th));
      camera.lookAt(cur.tx, cur.ty, cur.tz);
      camera.updateMatrixWorld();
    }
    var gs = KT.gestures(dom, {
      embed: o.embed, claim: o.pick, hit: o.hit, release: o.release,
      start: function (e) { if (e.pointerType !== 'touch') api.touched = true; },
      drag: function (dx, dy) {
        goal.th -= dx * 0.006;
        goal.ph = KT.clamp(goal.ph - dy * 0.005, 0.12, 1.5);
        api.touched = true;
      },
      pinch: function (k) { goal.r = KT.clamp(goal.r / k, lim.r0, lim.r1); api.touched = true; }
    });
    dom.addEventListener('wheel', function (e) {
      if (o.embed && !e.ctrlKey) return;           // the deck owns the wheel
      e.preventDefault();
      goal.r = KT.clamp(goal.r * Math.exp(e.deltaY * 0.0012), lim.r0, lim.r1); api.touched = true;
    }, { passive: false });
    api = {
      cur: cur, goal: goal, touched: false,
      /* time-based easing towards the goal; returns true while still moving */
      update: function (dt, rate) {
        var k = 1 - Math.exp(-(rate || 6) * dt), moving = false, key;
        for (key in goal) {
          var d = goal[key] - cur[key];
          if (Math.abs(d) > 1e-4) { cur[key] += d * k; moving = true; } else cur[key] = goal[key];
        }
        place();
        return moving;
      },
      to: function (g, snap) { Object.assign(goal, g); if (snap) Object.assign(cur, goal); place(); },
      dragging: gs.active
    };
    place();
    return api;
  };

  /* index of the point (xyz triples in P, the first n) whose screen image is nearest to client
     (x, y) within px CSS pixels, or -1. Touch picking: a fingertip needs a target far larger than
     a control point's dot, and a screen-space radius stays the same size at any zoom. */
  var _w = null;
  gl.nearest = function (camera, dom, P, n, x, y, px) {
    _w = _w || new T.Vector3();
    var r = dom.getBoundingClientRect(), best = -1, bd = px * px, second = 1e12, i, sx, sy, d;
    for (i = 0; i < n; i++) {
      _w.set(P[i * 3], P[i * 3 + 1], P[i * 3 + 2]).project(camera);
      if (_w.z > 1 || _w.z < -1) continue;
      sx = r.left + (_w.x * 0.5 + 0.5) * r.width; sy = r.top + (-_w.y * 0.5 + 0.5) * r.height;
      d = (sx - x) * (sx - x) + (sy - y) * (sy - y);
      if (d < bd) { second = bd < px * px ? bd : second; bd = d; best = i; }
      else if (d < second) second = d;
    }
    // A fingertip cannot single out one point of a dense net (the barn has 11,188): when a second
    // point is nearly as close, nothing is grabbed, so the touch scrolls or turns the stage instead.
    // Zoom in until the points are a finger apart and they become grabbable.
    if (best >= 0 && Math.sqrt(second) < Math.max(px * 0.6, 1.6 * Math.sqrt(bd))) return -1;
    return best;
  };

  /* ---------- derived Gaussians ("planoids") as instanced, tangent-aligned quads ----------
     Each instance is a flat Gaussian: centre + two tangent axes (1 sigma). No sorting: the
     Gaussian fringe is resolved with a screen-space dither, the core writes depth. */
  var EXT = 2.15;
  var VERT = [
    'attribute vec3 iCenter; attribute vec3 iU; attribute vec3 iV; attribute vec3 iColor; attribute vec3 iAux;',
    'uniform float uGain;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN; varying vec3 vW; varying vec3 vAux; varying float vZ;',
    '#ifdef LAYERS',                                             // a field with per-Gaussian opacity (gl.splats, opts.alpha)
    'attribute float iAlpha; uniform float uLayer; varying float vA;',
    '#endif',
    'void main(){',
    '  vQ = position.xy; vCol = iColor; vAux = iAux; vN = normalize(cross(iU, iV));',
    '#ifdef LAYERS',
    // gl.draw draws such a field in layers: 0 the opaque Gaussians, 1 the see-through ones that face away from
    // the eye, 2 the see-through ones that face it (-1: all of them at once, the plain render)
    '  vA = iAlpha;',
    '  if (uLayer > -0.5) {',
    '    bool thin = iAlpha < 0.999, near = dot(cross(iU, iV), cameraPosition - iCenter) > 0.0;',
    '    if (uLayer < 0.5 ? thin : (!thin || (uLayer < 1.5) == near)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }',
    '  }',
    '#endif',
    '  vec3 p = iCenter + (iU * position.x + iV * position.y) * uGain; vW = p;',
    '  vec4 e = viewMatrix * vec4(p, 1.0); vZ = -e.z;',
    '#ifdef PUSH',                                               // gl.draw, pass 1: one sigma behind the surface
    '  e.xyz *= 1.0 + uGain * max(length(iU), length(iV)) / length(e.xyz);',
    '#endif',
    '  gl_Position = projectionMatrix * e;',
    '}'].join('\n');
  /* uMode picks what a Gaussian shows: 0 appearance, 1 depth, 2 normal, 3 the Gaussian itself (drawn
     smaller, so each one reads), 4 its aux colour (the chart it belongs to). With uSplit = (bands,
     roll, slant) the mode is chosen per pixel instead: slanted screen bands, one mode each, shifted
     by roll (a fraction of the width) — every mode of the same frame, side by side. uBand (when its x
     is not negative) = (mode, mode, mode, how many of them): the bands show those modes in turn, over
     and over; otherwise band k shows mode k.
     uFade 0..1 dissolves whatever is shown into appearance: at 1 every band is the plain picture, so
     the split can be switched off (or on) without a jump. */
  var FRAG = [
    'precision highp float;',
    '#if defined(DEPTH) || defined(ACCUM)',                       // gl.draw cuts at weight 0.236 where the dither cut near 0.42:
    '#define SHARP 1.66',                                         // a steeper falloff keeps every Gaussian the size it had
    '#else',
    '#define SHARP 1.0',
    '#endif',
    'uniform vec3 uLight; uniform float uRim; uniform float uFlat;',
    'uniform float uMode; uniform vec3 uSplit; uniform vec2 uRes; uniform vec2 uDepth; uniform vec3 uTint; uniform vec4 uBand; uniform float uFade;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN; varying vec3 vW; varying vec3 vAux; varying float vZ;',
    '#ifdef OUT2',                                               // a second target: (opacity x weight) beside (colour x weight, weight)
    'varying float vA; layout(location = 1) out highp vec4 oAlpha;',
    '#endif',
    'vec3 ramp(float t){',                                         // near = light and warm, far = dark
    '  vec3 a = vec3(0.99, 0.95, 0.72), b = vec3(0.98, 0.55, 0.38), c = vec3(0.70, 0.21, 0.48), d = vec3(0.26, 0.09, 0.44), e = vec3(0.09, 0.06, 0.24);',
    '  t *= 4.0;',
    '  return t < 1.0 ? mix(a, b, t) : t < 2.0 ? mix(b, c, t - 1.0) : t < 3.0 ? mix(c, d, t - 2.0) : mix(d, e, t - 3.0);',
    '}',
    'void main(){',
    '  float m = uMode;',
    '  if (uSplit.x > 0.5) m = floor(fract((gl_FragCoord.x + uSplit.z * (uRes.y - gl_FragCoord.y)) / uRes.x - uSplit.y) * uSplit.x);',
    '  if (uSplit.x > 0.5 && uBand.x > -0.5) { m = mod(m, uBand.w); m = m < 0.5 ? uBand.x : m < 1.5 ? uBand.y : uBand.z; }',
    '  bool own = m > 2.5 && m < 3.5;',
    '  float ow = own ? 1.0 - uFade : 0.0, kk = mix(1.0, 3.6, ow);',      // a single Gaussian grows back to its full footprint as it fades
    '  float q2 = dot(vQ, vQ), K = kk * SHARP, g0 = exp(-0.5 * q2 * kk);',
    '#if defined(DEPTH) || defined(ACCUM)',
    // band-limit each Gaussian to the pixel grid: widen it by half a pixel (measured in its own units, from
    // the screen derivatives) and lower it so its total stays the same. A Gaussian much smaller than a pixel,
    // or seen edge-on as a sliver, then fades instead of flickering, and dense dots do not beat into moire.
    '  vec2 qx = dFdx(vQ), qy = dFdy(vQ);',
    '  float lp = 1.0 / (1.0 + K * 0.125 * (dot(qx, qx) + dot(qy, qy)));',
    '  float g = lp * exp(-0.5 * q2 * K * lp);',
    '#else',
    '  float g = exp(-0.5 * q2 * K);',
    '#endif',
    '  float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));',
    '  float a = smoothstep(0.28, 0.56, g);',
    '#if defined(DEPTH)',
    '  if (g < 0.236) discard;',                                  // the core of each Gaussian: what counts as the surface
    '  gl_FragColor = vec4(0.0); return;',
    '#elif defined(ACCUM)',
    '  float w = max(g - 0.1 * lp, 0.0) / 0.9;',                  // weight: falls to zero before the edge of the quad
    '  if (w <= 0.0) discard;',
    '  w *= mix(1.0, 0.15, ow);',                                 // single Gaussians keep their soft profile (see the resolve)
    '#elif defined(BLEND)',
    '  a = g;',
    '#else',
    '  if (a <= n) discard;',
    '#endif',
    '  vec3 nn = normalize(vN);',                                  // a planoid has two faces: light the one we see
    '  if (dot(nn, cameraPosition - vW) < 0.0) nn = -nn;',
    '  float d = max(dot(nn, uLight), 0.0);',
    '  float lit = mix(0.40 + 0.16 * nn.y + 0.66 * d, 1.0, uFlat), rimv = 0.55 + 0.45 * smoothstep(0.12, 0.75, g0);',
    '  vec3 c = (own ? uTint : m > 3.5 ? vAux : vCol) * lit;',
    '  c *= mix(1.0, rimv, own ? 1.0 : uRim);',
    '  if (m > 0.5 && m < 1.5) c = ramp(clamp(log(vZ / uDepth.x) / log(uDepth.y / uDepth.x), 0.0, 1.0));',
    '  else if (m > 1.5 && m < 2.5) c = nn * 0.5 + 0.5;',
    '  if (uFade > 0.0) c = mix(c, vCol * lit * mix(1.0, rimv, uRim), uFade);',
    '#if defined(ACCUM)',
    '  gl_FragColor = vec4(c * w, w);',
    '#ifdef OUT2',
    '  oAlpha = vec4(vA * w, 0.0, 0.0, w);',
    '#endif',
    '#elif defined(BLEND)',
    '  gl_FragColor = vec4(c, a * 0.92);',
    '#else',
    '  gl_FragColor = vec4(c, 1.0);',
    '#endif',
    '}'].join('\n');

  /* opts.blend: true alpha-blended Gaussians (smooth, unsorted — for a handful of showcase splats).
     opts.alpha: the field carries an opacity per Gaussian (`alpha`, 1 = solid; S_u x S_v must point out of the
     object). While `mesh.userData.ktSplat.thin` is true, gl.draw draws the see-through ones as two more layers. */
  gl.splats = function (capacity, opts) {
    var blend = !!(opts && opts.blend), lay = opts && opts.alpha ? '#define LAYERS\n' : '';
    var base = new T.PlaneGeometry(2 * EXT, 2 * EXT);
    var g = new T.InstancedBufferGeometry();
    g.index = base.index; g.setAttribute('position', base.getAttribute('position'));
    var A = {};
    ['iCenter', 'iU', 'iV', 'iColor', 'iAux'].forEach(function (n) {
      var at = new T.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      at.setUsage(T.DynamicDrawUsage); g.setAttribute(n, at); A[n] = at;
    });
    g.instanceCount = 0;
    if (lay) { A.iAlpha = new T.InstancedBufferAttribute(new Float32Array(capacity).fill(1), 1); A.iAlpha.setUsage(T.DynamicDrawUsage); g.setAttribute('iAlpha', A.iAlpha); }
    var U = { uGain: { value: 1 }, uRim: { value: 0.6 }, uFlat: { value: 0 }, uLayer: { value: -1 },
              uLight: { value: new T.Vector3(0.35, 0.86, 0.36).normalize() },
              uMode: { value: 0 }, uSplit: { value: new T.Vector3(0, 0, 0) }, uRes: { value: new T.Vector2(1, 1) },
              uBand: { value: new T.Vector4(-1, 0, 0, 0) }, uFade: { value: 0 },
              uDepth: { value: new T.Vector2(1, 10) }, uTint: { value: new T.Vector3().fromArray(gl.rgb(KT.color.splat)) } };
    function variant(def, o) {            // the same shader and the same uniforms, compiled for another pass
      return new T.ShaderMaterial(Object.assign({ vertexShader: lay + def + VERT, fragmentShader: lay + def + FRAG, side: T.DoubleSide, uniforms: U }, o));
    }
    var sum = { transparent: true, depthWrite: false, blending: T.CustomBlending, blendSrc: T.OneFactor, blendDst: T.OneFactor, blendSrcAlpha: T.OneFactor, blendDstAlpha: T.OneFactor };
    var mat = variant(blend ? '#define BLEND\n' : '', { transparent: blend, depthWrite: !blend, depthTest: !blend });
    var mesh = new T.Mesh(g, mat); mesh.frustumCulled = false;
    if (!blend) mesh.userData.ktSplat = {     // what gl.draw swaps in
      plain: mat,
      depth: variant('#define DEPTH\n#define PUSH\n', { colorWrite: false }),
      core: variant('#define DEPTH\n', { colorWrite: false }),
      accum: variant('#define ACCUM\n', sum),
      accum2: lay ? variant('#define ACCUM\n#define OUT2\n', sum) : null, thin: false
    };
    return {
      mesh: mesh, capacity: capacity, uniforms: mat.uniforms,
      center: A.iCenter.array, axU: A.iU.array, axV: A.iV.array, color: A.iColor.array, aux: A.iAux.array, alpha: lay ? A.iAlpha.array : null,
      setCount: function (n) { g.instanceCount = Math.min(n, capacity); },
      /* upload the first n instances (all of them when n is omitted) */
      commit: function (n) {
        [A.iCenter, A.iU, A.iV, A.iColor, A.iAux].forEach(function (at) {
          at.updateRange.offset = 0; at.updateRange.count = n == null ? -1 : Math.min(n, capacity) * 3; at.needsUpdate = true;
        });
        if (lay) { A.iAlpha.updateRange.offset = 0; A.iAlpha.updateRange.count = n == null ? -1 : Math.min(n, capacity); A.iAlpha.needsUpdate = true; }
      }
    };
  };

  /* ---------- a frame with smooth Gaussians ----------
     gl.draw(R, scene, camera) stands in for R.renderer.render(scene, camera). The splats of gl.splats are
     drawn as a surface instead of dithered dots:
       1  off screen, depth only: the core of each Gaussian, pushed one sigma back. Which layer is in front;
       2  off screen: every Gaussian fragment in front of that depth, summed as (colour x weight, weight);
       3  on screen: colour = sum / weight, opacity from the weight (a soft edge where the Gaussians thin out);
       4  on screen, depth only: the cores again, where they are, so that the rest of the scene, rendered
          last and as usual, still hides behind the surface or draws over it.
     The current viewport and scissor are respected. Without float render targets (or with no splats in the
     scene) it is a plain render: the dithered look.
     See-through Gaussians (a field made with opts.alpha, while its ktSplat.thin is set): steps 1 to 4 are done for
     the solid ones only; then twice more, for the see-through ones that face away and for those that face the
     eye, steps 1 to 3 into a target that also sums (opacity x weight), blended over the screen with that
     opacity. They hide behind the solid surface, show it through themselves, and write no depth: the rest of
     the scene is drawn over them. */
  var _v4 = null, _c = null;
  function smooth(R) {
    if (R.smooth !== undefined) return R.smooth;
    var r = R.renderer;
    if (!r.capabilities.isWebGL2 || !(r.extensions.has('EXT_color_buffer_float') || r.extensions.has('EXT_color_buffer_half_float'))) return (R.smooth = null);
    var rt = new T.WebGLRenderTarget(4, 4, { type: T.HalfFloatType, minFilter: T.NearestFilter, magFilter: T.NearestFilter, stencilBuffer: false });
    var tri = new T.BufferGeometry();
    tri.setAttribute('position', new T.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3));
    var mat = new T.ShaderMaterial({
      transparent: true, depthTest: false, depthWrite: false,
      uniforms: { tAcc: { value: rt.texture }, uSize: { value: new T.Vector2(4, 4) } },
      vertexShader: 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: [
        'uniform sampler2D tAcc; uniform vec2 uSize;',
        'void main(){',
        '  vec4 s = texture2D(tAcc, gl_FragCoord.xy / uSize);',
        '  if (s.a < 0.02) discard;',
        '  gl_FragColor = vec4(s.rgb / s.a, smoothstep(0.02, 0.15, s.a));',
        '}'].join('\n')
    });
    var quad = new T.Mesh(tri, mat); quad.frustumCulled = false;
    var sc = new T.Scene(); sc.add(quad);
    return (R.smooth = { rt: rt, mat: mat, quad: quad, scene: sc, cam: new T.Camera(), size: new T.Vector2() });
  }
  function layered(S) {                    // the second target and its resolve, made the first time they are needed
    if (S.rt2) return;
    S.rt2 = new T.WebGLMultipleRenderTargets(4, 4, 2, { type: T.HalfFloatType, minFilter: T.NearestFilter, magFilter: T.NearestFilter, stencilBuffer: false });
    S.mat2 = new T.ShaderMaterial({
      transparent: true, depthTest: false, depthWrite: false,
      uniforms: { tAcc: { value: S.rt2.texture[0] }, tAlpha: { value: S.rt2.texture[1] }, uSize: S.mat.uniforms.uSize, uThin: { value: 0 } },
      vertexShader: 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: [
        'uniform sampler2D tAcc; uniform sampler2D tAlpha; uniform vec2 uSize; uniform float uThin;',
        'void main(){',
        '  vec4 s = texture2D(tAcc, gl_FragCoord.xy / uSize);',
        '  if (s.a < 0.02) discard;',
        '  gl_FragColor = vec4(s.rgb / s.a, smoothstep(0.02, 0.15, s.a) * mix(1.0, texture2D(tAlpha, gl_FragCoord.xy / uSize).r / s.a, uThin));',
        '}'].join('\n')
    });
  }
  gl.draw = function (R, scene, camera) {
    var r = R.renderer, S = smooth(R), list = [], i;
    if (S) scene.traverseVisible(function (o) { if (o.userData.ktSplat) list.push(o); });
    if (!list.length) { r.render(scene, camera); return; }
    _v4 = _v4 || new T.Vector4(); _c = _c || new T.Color();
    var pr = r.getPixelRatio(), rt = S.rt, size = r.getDrawingBufferSize(S.size);
    if (rt.width !== size.x || rt.height !== size.y) { rt.setSize(size.x, size.y); S.mat.uniforms.uSize.value.copy(size); }
    rt.viewport.copy(r.getViewport(_v4)).multiplyScalar(pr).floor();
    rt.scissor.copy(r.getScissor(_v4)).multiplyScalar(pr).floor(); rt.scissorTest = r.getScissorTest();
    var auto = r.autoClear, target = r.getRenderTarget(), alpha = r.getClearAlpha(), mask = camera.layers.mask, bg = scene.background;
    var masks = list.map(function (o) { return o.layers.mask; });
    function use(k) { list.forEach(function (o) { o.material = o.userData.ktSplat[k]; }); }
    r.getClearColor(_c);
    list.forEach(function (o) { o.layers.set(31); });
    camera.layers.set(31); scene.background = null; r.autoClear = false;
    var thin = list.filter(function (o) { return o.userData.ktSplat.thin && o.userData.ktSplat.accum2; });
    if (thin.length) {
      layered(S); rt = S.rt2;
      if (rt.width !== size.x || rt.height !== size.y) rt.setSize(size.x, size.y);
      rt.viewport.copy(S.rt.viewport); rt.scissor.copy(S.rt.scissor); rt.scissorTest = S.rt.scissorTest;
      S.quad.material = S.mat2;
    }
    function layer(v) { thin.forEach(function (o) { o.material.uniforms.uLayer.value = v; }); }
    function sums() { list.forEach(function (o) { o.material = o.userData.ktSplat[thin.indexOf(o) < 0 ? 'accum' : 'accum2']; }); }
    layer(0);
    r.setRenderTarget(rt); r.setClearColor(0x000000, 0); r.clear(true, true, false);
    use('depth'); r.render(scene, camera);                                    // 1
    sums(); r.render(scene, camera);                                          // 2
    r.setRenderTarget(target); r.setClearColor(_c, alpha);
    if (auto) r.clear();
    if (thin.length) S.mat2.uniforms.uThin.value = 0;
    r.render(S.scene, S.cam);                                                 // 3
    use('core'); r.render(scene, camera);                                     // 4
    if (thin.length) {
      var rest = list.filter(function (o) { return thin.indexOf(o) < 0; });
      rest.forEach(function (o) { o.visible = false; });
      S.mat2.uniforms.uThin.value = 1;
      [1, 2].forEach(function (v) {                                           // far side, then near side
        use('depth'); layer(v);
        r.setRenderTarget(rt); r.setClearColor(0x000000, 0); r.clear(true, false, false);   // the depth of the solid surface stays
        r.render(scene, camera);
        sums(); r.render(scene, camera);
        r.setRenderTarget(target); r.setClearColor(_c, alpha);
        r.render(S.scene, S.cam);
      });
      layer(-1); rest.forEach(function (o) { o.visible = true; });
      S.quad.material = S.mat;
    }
    camera.layers.mask = mask; scene.background = bg;
    r.render(scene, camera);                                                  // everything that is not a splat
    r.autoClear = auto;
    for (i = 0; i < list.length; i++) { list[i].layers.mask = masks[i]; list[i].material = list[i].userData.ktSplat.plain; }
  };

  /* Turn an evaluated (u,v) grid into planoids — the whole "method" in one loop:
       mean  = S(u,v)
       frame = [ S_u^ , S_v^perp ]            (rotation, from the tangents)
       scale = ( |S_u| du , |S_v^perp| dv )   (from the local sampling interval)
     grid = { pos, su, sv, col } with row = v index; du[a], dv[b] = sampling interval at each
     sample; k = sigma as a fraction of the interval; rgb (optional) overrides the colour.
     cull (optional) = { x, y, z, flip, min }: leave out samples whose outward normal (S_u x S_v, times
     flip) points away from the eye at (x, y, z) by more than min (a cosine, e.g. -0.1) — back faces.
     kept (optional Int32Array) receives, per written instance, which sample (b * na + a) it came from.
     Returns the next free instance index. */
  gl.writePlanoids = function (sp, start, grid, na, nb, du, dv, k, rgb, cull, kept) {
    var P = grid.pos, SU = grid.su, SV = grid.sv, CL = grid.col, n = start, a, b, t, o;
    var cx = sp.center, ux = sp.axU, vx = sp.axV, cl = sp.color;
    for (b = 0; b < nb; b++) for (a = 0; a < na; a++) {
      if (n >= sp.capacity) return n;
      t = (b * na + a) * 3; o = n * 3;
      var sux = SU[t], suy = SU[t + 1], suz = SU[t + 2], svx = SV[t], svy = SV[t + 1], svz = SV[t + 2];
      if (cull) {
        var nx = suy * svz - suz * svy, ny = suz * svx - sux * svz, nz = sux * svy - suy * svx, ex = cull.x - P[t], ey = cull.y - P[t + 1], ez = cull.z - P[t + 2];
        if ((nx * ex + ny * ey + nz * ez) * cull.flip < cull.min * Math.sqrt((nx * nx + ny * ny + nz * nz) * (ex * ex + ey * ey + ez * ez))) continue;
      }
      if (kept) kept[n - start] = b * na + a;
      var uu = sux * sux + suy * suy + suz * suz || 1e-12, uv = (sux * svx + suy * svy + suz * svz) / uu;
      svx -= uv * sux; svy -= uv * suy; svz -= uv * suz;        // S_v orthogonalised against S_u
      var ku = du[a] * k, kv = dv[b] * k;
      cx[o] = P[t]; cx[o + 1] = P[t + 1]; cx[o + 2] = P[t + 2];
      ux[o] = sux * ku; ux[o + 1] = suy * ku; ux[o + 2] = suz * ku;
      vx[o] = svx * kv; vx[o + 1] = svy * kv; vx[o + 2] = svz * kv;
      if (rgb) { cl[o] = rgb[0]; cl[o + 1] = rgb[1]; cl[o + 2] = rgb[2]; }
      else { cl[o] = CL[t]; cl[o + 1] = CL[t + 1]; cl[o + 2] = CL[t + 2]; }
      n++;
    }
    return n;
  };

  /* line geometry helpers */
  gl.lines = function (maxVerts, color, opacity) {
    var g = new T.BufferGeometry();
    var at = new T.BufferAttribute(new Float32Array(maxVerts * 3), 3); at.setUsage(T.DynamicDrawUsage);
    g.setAttribute('position', at); g.setDrawRange(0, 0);
    var m = new T.LineBasicMaterial({ color: color, transparent: opacity < 1, opacity: opacity == null ? 1 : opacity });
    var obj = new T.LineSegments(g, m); obj.frustumCulled = false;
    return { obj: obj, arr: at.array, mat: m,
             set: function (nVerts) { g.setDrawRange(0, nVerts); at.needsUpdate = true; } };
  };

  /* a fat arrow gizmo that always draws on top; aim(origin, dir, length) */
  gl.arrow = function (color, radius) {
    var r = radius || 0.018, m = new T.MeshBasicMaterial({ color: color, depthTest: false, transparent: true });
    var shaft = new T.Mesh(new T.CylinderGeometry(r, r, 1, 10).translate(0, 0.5, 0), m);
    var head = new T.Mesh(new T.ConeGeometry(r * 3.4, r * 9, 14).translate(0, r * 4.5, 0), m);
    var g = new T.Group(); g.add(shaft); g.add(head); shaft.renderOrder = head.renderOrder = 20;
    var up = new T.Vector3(0, 1, 0), d = new T.Vector3();
    return { obj: g, aim: function (o, x, y, z, len) {
      g.position.set(o[0], o[1], o[2]); g.quaternion.setFromUnitVectors(up, d.set(x, y, z).normalize());
      shaft.scale.y = Math.max(0.01, len - r * 9); head.position.y = Math.max(0.01, len - r * 9);
    } };
  };

  /* ---------- depth cues: things that are hidden stay visible, but faint ----------
     A material that draws only where something nearer has already written depth. Put it on a second object
     that shares the geometry (and the transform) of the real one and render it afterwards (renderOrder):
     the part of a control net, an arrow or a plane that lies behind the surface shows through, dimmed.
     kind: 'line' for line sets, 'flat' for open sheets (both faces), omitted for closed solids (front faces
     only, so a solid does not show through itself). */
  gl.hidden = function (color, opacity, kind) {
    var M = kind === 'line' ? T.LineBasicMaterial : T.MeshBasicMaterial;
    return new M({ color: color, transparent: true, opacity: opacity, depthFunc: T.GreaterDepth, depthWrite: false,
                   side: kind === 'flat' ? T.DoubleSide : T.FrontSide });
  };

  /* A vector as a solid object: a lit cylinder and cone (needs lights in the scene), depth-tested, with a dark
     cap under the cone (seen from behind, an arrow shows a dark disc) and a faint copy where it is hidden.
     aim(origin, x, y, z, length, k): k scales the thickness; the head never takes more than ~45% of the length. */
  gl.vector = function (color, o) {
    o = o || {};
    var r = o.radius || 0.02, hl = r * 8, hr = r * 3.1, c = new T.Color(color);
    var lit = new T.MeshLambertMaterial({ color: c, emissive: c.clone().multiplyScalar(0.3) });
    var cap = new T.MeshBasicMaterial({ color: c.clone().multiplyScalar(KT.theme === 'light' ? 0.5 : 0.2) });
    var ghost = gl.hidden(color, o.ghost == null ? 0.3 : o.ghost);
    var sg = new T.CylinderGeometry(r, r, 1, 16).translate(0, 0.5, 0);
    var hg = new T.ConeGeometry(hr, hl, 24).translate(0, hl / 2, 0);     // groups: 0 = side, 2 = base cap
    var shaft = new T.Mesh(sg, lit), head = new T.Mesh(hg, [lit, lit, cap]);
    var gs = new T.Mesh(sg, ghost), gh = new T.Mesh(hg, ghost);
    gs.renderOrder = gh.renderOrder = o.order == null ? 8 : o.order;
    shaft.add(gs); head.add(gh);
    var g = new T.Group(); g.add(shaft); g.add(head);
    var up = new T.Vector3(0, 1, 0), d = new T.Vector3();
    return { obj: g, aim: function (p, x, y, z, len, k) {
      var f = Math.min(k || 1, len / (hl * 2.2));
      g.position.set(p[0], p[1], p[2]); g.quaternion.setFromUnitVectors(up, d.set(x, y, z).normalize());
      head.scale.setScalar(f); head.position.y = len - hl * f;
      shaft.scale.set(f, Math.max(1e-5, len - hl * f), f);
    } };
  };

  /* A lit sheet over an evaluated (u,v) grid of na x nb points (row = v index): the surface as an object.
     Shaded by its normal under the splats' light, the face turned away from the eye darker, a rim of the
     surface colour at grazing angles, fading with distance (uFog = near, far along the view axis).
     set(grid, flip): grid = { pos, su, sv } from nurbs.patchGrid, pos being the sheet's own `pos` array;
     flip = +1 / -1 picks which side of S_u x S_v is the lit "top".
     gauss(i, c, A, B, rgb, w): paint Gaussian i (of up to 9) on the sheet — centre c, 1-sigma axes A and B, peak
     opacity w (0 = off). A flat Gaussian lying in a tangent plane cannot be drawn as its own translucent quad
     next to a curved, depth-writing sheet without the sheet cutting pieces out of it; its footprint on the
     sheet (the density taken along the normal, fading within about one sigma off the plane) has no such seam.
     uniforms.uCell = (i, j, n, k): tint cell (i, j) of an n x n lattice over the sheet's (u,v) square and its
     k = 4 or 8 neighbours with uCellCol (n = 0: off). Done per pixel from the interpolated (u,v), so the cell
     edges are exact whatever the tessellation. */
  gl.sheet = function (na, nb, o) {
    o = o || {};
    var g = new T.BufferGeometry(), pos = new Float32Array(na * nb * 3), nor = new Float32Array(na * nb * 3), idx = [], a, b, q;
    var uv = new Float32Array(na * nb * 2);
    for (b = 0; b < nb; b++) for (a = 0; a < na; a++) { uv[(b * na + a) * 2] = a / (na - 1); uv[(b * na + a) * 2 + 1] = b / (nb - 1); }
    for (b = 0; b < nb - 1; b++) for (a = 0; a < na - 1; a++) { q = b * na + a; idx.push(q, q + 1, q + na, q + 1, q + na + 1, q + na); }
    g.setAttribute('position', new T.BufferAttribute(pos, 3)); g.setAttribute('normal', new T.BufferAttribute(nor, 3));
    g.setAttribute('uv', new T.BufferAttribute(uv, 2)); g.setIndex(idx);
    function v3s() { var l = [], i; for (i = 0; i < 9; i++) l.push(new T.Vector3()); return l; }
    // the sheet's own colours, top and underside: deep blue on the dark stage, a light blue on a light page's glass
    var light = KT.theme === 'light';
    var mat = new T.ShaderMaterial({
      side: T.DoubleSide, transparent: true, polygonOffset: true, polygonOffsetFactor: 2, polygonOffsetUnits: 2,
      uniforms: { uLight: { value: new T.Vector3(0.35, 0.86, 0.36).normalize() },
                  uTop: { value: light ? new T.Vector3(0.400, 0.470, 0.540) : new T.Vector3(0.050, 0.150, 0.225) },
                  uUnder: { value: light ? new T.Vector3(0.440, 0.470, 0.520) : new T.Vector3(0.030, 0.062, 0.100) },
                  uRim: { value: new T.Vector3().fromArray(gl.rgb(KT.color.surf)) }, uBg: { value: new T.Vector3().fromArray(gl.rgb(KT.color.bg)) },
                  uFog: { value: new T.Vector2(4, 14) }, uAlpha: { value: o.alpha == null ? 0.94 : o.alpha },
                  uCell: { value: new T.Vector4(0, 0, 0, 4) }, uCellCol: { value: new T.Vector3().fromArray(gl.rgb(KT.color.splat)) },
                  uGc: { value: v3s() }, uGa: { value: v3s() }, uGb: { value: v3s() }, uGn: { value: v3s() },
                  uGk: { value: v3s().map(function () { return new T.Vector4(); }) } },
      vertexShader: [
        'varying vec3 vN; varying vec3 vW; varying float vZ; varying vec2 vUv;',
        'void main(){ vN = normal; vW = position; vUv = uv; vec4 e = viewMatrix * vec4(position, 1.0); vZ = -e.z; gl_Position = projectionMatrix * e; }'].join('\n'),
      fragmentShader: [
        'uniform vec3 uLight; uniform vec3 uTop; uniform vec3 uUnder; uniform vec3 uRim; uniform vec3 uBg; uniform vec2 uFog; uniform float uAlpha;',
        'uniform vec3 uGc[9]; uniform vec3 uGa[9]; uniform vec3 uGb[9]; uniform vec3 uGn[9]; uniform vec4 uGk[9];',
        'uniform vec4 uCell; uniform vec3 uCellCol;',
        'varying vec3 vN; varying vec3 vW; varying float vZ; varying vec2 vUv;',
        'void main(){',
        '  vec3 n = normalize(vN), V = normalize(cameraPosition - vW);',
        '  float f = dot(n, V); bool under = f < 0.0;',
        '  float d = max(dot(n, uLight), 0.0);',                    // the light is on the top side only
        '  float wrap = 0.5 + 0.5 * dot(n, uLight);',
        '  vec3 c = under ? uUnder * (0.75 + 0.55 * wrap) : uTop * (0.42 + 0.50 * wrap * wrap + 1.25 * d * d);',
        '  if (!under) c += uRim * 0.22 * pow(max(dot(n, normalize(uLight + V)), 0.0), 28.0);',
        '  c += uRim * (under ? 0.05 : 0.11) * pow(1.0 - abs(f), 3.0);',
        '  if (uCell.z > 0.5) {',
        '    vec2 q = abs(floor(vUv * uCell.z) - uCell.xy); float far = max(q.x, q.y);',
        '    float t = far < 0.5 ? 0.36 : far > 1.5 ? 0.0 : (uCell.w > 6.0 || q.x + q.y < 1.5) ? 0.17 : 0.035;',
        '    c = mix(c, uCellCol * (0.62 + 0.38 * wrap), t);',
        '  }',
        '  for (int i = 0; i < 9; i++) {',
        '    vec3 r = vW - uGc[i]; vec3 q = vec3(dot(r, uGa[i]), dot(r, uGb[i]), dot(r, uGn[i]));',
        '    c = mix(c, uGk[i].rgb * (0.62 + 0.38 * wrap), uGk[i].a * exp(-0.5 * dot(q, q)));',
        '  }',
        '  c = mix(c, uBg, 0.72 * clamp((vZ - uFog.x) / (uFog.y - uFog.x), 0.0, 1.0));',
        '  gl_FragColor = vec4(c, uAlpha);',
        '}'].join('\n')
    });
    var mesh = new T.Mesh(g, mat); mesh.frustumCulled = false;
    var U = mat.uniforms;
    return { mesh: mesh, geo: g, pos: pos, uniforms: mat.uniforms,
      gauss: function (i, c, A, B, rgb, w) {
        if (!w) { U.uGk.value[i].w = 0; return; }
        var aa = A[0] * A[0] + A[1] * A[1] + A[2] * A[2], bb = B[0] * B[0] + B[1] * B[1] + B[2] * B[2];
        var nx = A[1] * B[2] - A[2] * B[1], ny = A[2] * B[0] - A[0] * B[2], nz = A[0] * B[1] - A[1] * B[0];
        var k = 1 / ((Math.sqrt(nx * nx + ny * ny + nz * nz) || 1) * Math.sqrt(Math.min(aa, bb)));
        U.uGc.value[i].set(c[0], c[1], c[2]); U.uGa.value[i].set(A[0] / aa, A[1] / aa, A[2] / aa);
        U.uGb.value[i].set(B[0] / bb, B[1] / bb, B[2] / bb); U.uGn.value[i].set(nx * k, ny * k, nz * k);
        U.uGk.value[i].set(rgb[0], rgb[1], rgb[2], w);
      },
      set: function (grid, flip) {
        var SU = grid.su, SV = grid.sv, i, t, x, y, z, l, s = flip || 1;
        for (i = 0; i < na * nb; i++) {
          t = i * 3;
          x = SU[t + 1] * SV[t + 2] - SU[t + 2] * SV[t + 1]; y = SU[t + 2] * SV[t] - SU[t] * SV[t + 2]; z = SU[t] * SV[t + 1] - SU[t + 1] * SV[t];
          l = s / (Math.sqrt(x * x + y * y + z * z) || 1); nor[t] = x * l; nor[t + 1] = y * l; nor[t + 2] = z * l;
        }
        g.attributes.position.needsUpdate = true; g.attributes.normal.needsUpdate = true;
      } };
  };

  /* project a world point to CSS pixels inside the module root; returns null when behind */
  var _v = null;
  gl.project = function (camera, x, y, z, w, h) {
    _v = _v || new T.Vector3();
    _v.set(x, y, z).project(camera);
    if (_v.z > 1) return null;
    return [(_v.x * 0.5 + 0.5) * w, (-_v.y * 0.5 + 0.5) * h];
  };

  /* '#rrggbb' -> [r,g,b] in display space (the splat shader writes colours straight out) */
  gl.rgb = function (hex) {
    var n = parseInt(hex.slice(1), 16);
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  };
})();
