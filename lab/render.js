/* Knot-a-Surface lab — deferred renderer for derived Gaussians.
   Every Gaussian is an instanced, tangent-aligned quad (centre + two tangent axes, as in talk/gl.js).
   Pipeline, per frame:
     1. shadow pass   the same instances from the key light, into a depth map
     2. G-buffer pass albedo + chart id | world normal | control features (DINOv3 PCA colour, similarity)
     3. lighting pass key light with shadows, sky/ground ambient with screen-space occlusion, a studio
                      environment for reflections, tone mapping — or one of the data views
     4. accumulation  while nothing moves, frames are averaged: the Gaussian fringe is resolved by a
                      new dither pattern each frame (stochastic transparency), the camera and the light
                      are jittered sub-pixel / over a small disc, so edges, shadows and occlusion converge
   Views (uMode): 0 beauty, 1 appearance (unlit control colour), 2 depth, 3 normals, 4 single Gaussians,
   5 charts, 6 features, 7 similarity. uSplit shows several at once in rolling slanted bands.
   Needs THREE r158 with WebGL2. Exposed as KnotLab.renderer(host, camera, opts). */
(function () {
  'use strict';
  var T = window.THREE, KL = window.KnotLab = window.KnotLab || {};
  var EXT = 2.15, MAXC = 32, ACC = 64, SHADOW = 2048;

  var COMMON = [
    'float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }',
    'uniform float uMode; uniform vec3 uSplit; uniform float uBand[8]; uniform vec2 uRes; uniform float uFrame;',
    'float viewMode(){',
    '  if (uSplit.x < 0.5) return uMode;',
    '  float q = fract((gl_FragCoord.x + uSplit.z * (uRes.y - gl_FragCoord.y)) / uRes.x - uSplit.y);',
    '  return uBand[int(q * uSplit.x)];',
    '}'].join('\n');

  var G_VERT = [
    'attribute vec3 iCenter; attribute vec3 iU; attribute vec3 iV; attribute vec3 iColor; attribute vec4 iAux; attribute vec4 iMeta;',
    'uniform float uHide;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN; varying vec4 vAux; varying vec4 vMeta;',
    'void main(){',
    '  vQ = position.xy; vCol = iColor; vAux = iAux; vMeta = iMeta; vN = normalize(cross(iU, iV));',
    '  gl_Position = projectionMatrix * viewMatrix * vec4(iCenter + iU * position.x + iV * position.y, 1.0);',
    '  if (uHide > 0.5 && iMeta.y > 0.5) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);',   // hidden selection: off screen
    '}'].join('\n');

  /* iMeta = (chart index, selection weight, u, v); iAux = (feature colour, similarity) */
  var G_FRAG = [
    'precision highp float;',
    'layout(location = 0) out vec4 g0; layout(location = 1) out vec4 g1; layout(location = 2) out vec4 g2;',
    COMMON,
    'uniform float uBake;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN; varying vec4 vAux; varying vec4 vMeta;',
    'void main(){',
    '  float m = viewMode(); bool own = m > 3.5 && m < 4.5;',
    '  float g = exp(-0.5 * dot(vQ, vQ) * (own ? 3.6 : 1.0));',
    '  if (smoothstep(0.28, 0.56, g) <= (uBake > 0.5 ? 0.5 : ign(gl_FragCoord.xy + 5.588238 * uFrame))) discard;',
    '  g0 = vec4(pow(max(vCol, 0.0), vec3(2.2)), vMeta.x / 255.0);',
    '  g1 = vec4(normalize(vN), own ? smoothstep(0.12, 0.75, g) : vMeta.y);',
    '  g2 = uBake > 0.5 ? vec4(vMeta.z, vMeta.w, vMeta.x + 1.0, 1.0) : vAux;',
    '}'].join('\n');

  var S_FRAG = [
    'precision highp float;',
    'layout(location = 0) out vec4 o;',
    'uniform float uFrame;',
    'varying vec2 vQ;',
    'float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }',
    'void main(){',
    '  if (smoothstep(0.28, 0.56, exp(-0.5 * dot(vQ, vQ))) <= ign(gl_FragCoord.xy + 3.17 * uFrame)) discard;',
    '  o = vec4(1.0);',
    '}'].join('\n');

  var FS_VERT = 'void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }';

  var L_FRAG = [
    'precision highp float;',
    'layout(location = 0) out vec4 o;',
    COMMON,
    'uniform sampler2D tG0; uniform sampler2D tG1; uniform sampler2D tG2; uniform sampler2D tDepth; uniform sampler2D tShadow;',
    'uniform mat4 uInvProj; uniform mat4 uProj; uniform mat4 uCamWorld; uniform mat4 uView; uniform mat4 uLightVP;',
    'uniform vec3 uEye; uniform vec3 uLight; uniform vec3 uPage; uniform vec4 uFade; uniform vec2 uDepthR; uniform float uAlpha; uniform float uAoR;',
    'uniform vec4 uMat[' + MAXC + ']; uniform vec3 uPal[' + MAXC + ']; uniform vec3 uTint; uniform vec3 uSelCol;',
    'const float PI = 3.14159265;',
    'vec3 lin(vec3 c){ return pow(c, vec3(2.2)); }',
    'vec3 viewPos(vec2 uv){ vec4 c = uInvProj * vec4(uv * 2.0 - 1.0, texture(tDepth, uv).x * 2.0 - 1.0, 1.0); return c.xyz / c.w; }',
    'vec3 ramp(float t){',                                          // near = light and warm, far = dark
    '  vec3 a = vec3(0.99, 0.95, 0.72), b = vec3(0.98, 0.55, 0.38), c = vec3(0.70, 0.21, 0.48), d = vec3(0.26, 0.09, 0.44), e = vec3(0.09, 0.06, 0.24);',
    '  t *= 4.0;',
    '  return t < 1.0 ? mix(a, b, t) : t < 2.0 ? mix(b, c, t - 1.0) : t < 3.0 ? mix(c, d, t - 2.0) : mix(d, e, t - 3.0);',
    '}',
    'vec3 heat(float t){',                                          // similarity: paper -> ultramarine -> ink
    '  return t < 0.5 ? mix(vec3(0.93, 0.92, 0.89), vec3(0.45, 0.62, 0.92), t * 2.0) : mix(vec3(0.45, 0.62, 0.92), vec3(0.08, 0.14, 0.52), t * 2.0 - 2.0 + 1.0);',
    '}',
    /* a photo studio: a bright dome, one large overhead softbox on the key side, a cool strip opposite */
    'vec3 studio(vec3 r, float rough){',
    '  vec3 c = mix(vec3(0.34, 0.33, 0.32), vec3(0.95, 0.96, 1.0), smoothstep(-0.25, 0.9, r.y));',
    '  float b = mix(0.015, 0.55, rough);',
    '  c += vec3(5.2, 5.0, 4.6) * smoothstep(0.90 - b, 0.985, dot(r, uLight));',
    '  c += vec3(1.3, 1.5, 1.9) * smoothstep(0.955 - b, 0.995, dot(r, normalize(vec3(-uLight.x, 0.25, -uLight.z))));',
    '  return c * mix(0.5, 1.0, smoothstep(-0.35, 0.05, r.y));',
    '}',
    'float shadow(vec3 P, vec3 N){',
    '  vec4 l = uLightVP * vec4(P + N * 0.03, 1.0); vec3 s = l.xyz / l.w * 0.5 + 0.5;',
    '  if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 1.0;',
    '  float a = ign(gl_FragCoord.xy + 13.0 * uFrame) * 6.2832, sum = 0.0;',
    '  for (int i = 0; i < 6; i++) {',
    '    float r = sqrt((float(i) + 0.5) / 6.0) * 1.6 / ' + SHADOW + '.0, t = a + float(i) * 2.39996;',
    '    sum += step(s.z - 0.0012, texture(tShadow, s.xy + r * vec2(cos(t), sin(t))).x);',
    '  }',
    '  return sum / 6.0;',
    '}',
    'float occlusion(vec2 uv, vec3 P, vec3 N){',                    // N in view space
    '  float a = ign(gl_FragCoord.xy + 31.0 * uFrame) * 6.2832, occ = 0.0, rp = uAoR * uProj[1][1] / -P.z * 0.5;',
    '  for (int i = 0; i < 9; i++) {',
    '    float r = sqrt((float(i) + 0.5) / 9.0) * rp, t = a + float(i) * 2.39996;',
    '    vec3 v = viewPos(uv + r * vec2(cos(t) * uRes.y / uRes.x, sin(t))) - P; float d2 = dot(v, v);',
    '    occ += max(0.0, dot(v, N) - 0.02 * uAoR) / (d2 + 0.02 * uAoR * uAoR) * smoothstep(uAoR * uAoR * 1.6, uAoR * uAoR * 0.4, d2);',
    '  }',
    '  return clamp(1.0 - 1.15 * uAoR * occ / 9.0, 0.0, 1.0);',
    '}',
    'vec3 tone(vec3 x){ return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }',
    'void main(){',
    '  vec2 uv = gl_FragCoord.xy / uRes;',
    '  if (texture(tDepth, uv).x >= 1.0) { o = vec4(uPage, uAlpha); return; }',
    '  vec4 a0 = texture(tG0, uv), a1 = texture(tG1, uv), a2 = texture(tG2, uv);',
    '  int id = int(a0.a * 255.0 + 0.5); float m = viewMode();',
    '  vec3 Pv = viewPos(uv), P = (uCamWorld * vec4(Pv, 1.0)).xyz, V = normalize(uEye - P), N = normalize(a1.xyz);',
    '  if (dot(N, V) < 0.0) N = -N;',
    '  vec3 Nv = mat3(uView) * N, c;',
    '  float ao = occlusion(uv, Pv, Nv), nl = max(dot(N, uLight), 0.0), sh = shadow(P, N);',
    '  float clay = (0.42 + 0.20 * N.y) * mix(0.55, 1.0, ao) + 0.62 * nl * mix(0.45, 1.0, sh);',   // plain shading for the data views
    '  if (m < 0.5 || m > 7.5) {',
    '    float rough = uMat[id].x, metal = uMat[id].y;',
    '    vec3 F0 = mix(vec3(0.04), a0.rgb, metal), dif = a0.rgb * (1.0 - metal), H = normalize(uLight + V);',
    '    float nh = max(dot(N, H), 0.0), nv = max(dot(N, V), 1e-3), vh = max(dot(V, H), 0.0);',
    '    float al = max(rough * rough, 0.02), D = al * al / (PI * pow(nh * nh * (al * al - 1.0) + 1.0, 2.0));',
    '    float k = (rough + 1.0) * (rough + 1.0) / 8.0, G = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);',
    '    vec3 F = F0 + (1.0 - F0) * pow(1.0 - vh, 5.0);',
    '    c = (dif / PI + min(D * G, 40.0) * F / (4.0 * nl * nv + 1e-3)) * nl * sh * vec3(3.05, 2.9, 2.62);',
    '    vec3 Fr = F0 + (max(vec3(1.0 - rough), F0) - F0) * pow(1.0 - nv, 5.0);',
    '    c += dif * mix(vec3(0.30, 0.28, 0.26), vec3(0.74, 0.77, 0.84), N.y * 0.5 + 0.5) * ao;',
    '    c += studio(reflect(-V, N), rough) * Fr * mix(ao, 1.0, 0.35) * mix(1.0, 0.6, rough);',
    '    c = tone(c * 0.92);',
    // the floor only catches light: far from any object it is exactly the page, so the figure has no plate
    '    vec3 irr = mix(vec3(0.30, 0.28, 0.26), vec3(0.74, 0.77, 0.84), N.y * 0.5 + 0.5), key = nl * vec3(3.05, 2.9, 2.62) / PI;',
    '    if (uMat[id].z > 0.5) c = uPage * (key * sh + irr * ao) / (key + irr);',
    '  }',
    '  else if (m < 1.5) c = a0.rgb;',
    '  else if (m < 2.5) c = lin(ramp(clamp(log(-Pv.z / uDepthR.x) / log(uDepthR.y / uDepthR.x), 0.0, 1.0)));',
    '  else if (m < 3.5) c = lin(N * 0.5 + 0.5);',
    '  else if (m < 4.5) c = lin(uTint) * clay * (0.4 + 0.6 * a1.w);',
    '  else if (m < 5.5) c = lin(uPal[id]) * clay;',
    '  else if (m < 6.5) c = lin(a2.rgb) * mix(0.82, 1.0, ao);',
    '  else c = lin(heat(clamp(a2.a, 0.0, 1.0))) * mix(0.8, 1.0, ao);',
    '  if (m > 5.5 && m < 7.5 && uMat[id].z > 0.5) c = uPage * mix(0.88, 1.0, ao);',   // feature views: the floor stays paper
    '  if (m < 3.5 || m > 4.5) {',                                 // the selection, hatched (a1.w is the splat profile in view 4)
    '    float hatch = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) / 9.0));',
    '    c = mix(c, uSelCol, a1.w * (0.12 + 0.26 * hatch) * step(m, 5.5));',
    '  }',
    '  float fade = smoothstep(uFade.z, uFade.w, length(P.xz - uFade.xy));',         // the floor dissolves into the page,
    '  float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));',             // and before it reaches the figure\'s edge
    '  c = mix(c, uPage, max(fade, uMat[id].z * (1.0 - smoothstep(0.0, 0.14, edge))));',
    '  o = vec4(c, uAlpha);',
    '}'].join('\n');

  var OUT_FRAG = [
    'precision highp float;',
    'layout(location = 0) out vec4 o;',
    'uniform sampler2D tAcc; uniform vec2 uRes; uniform float uRaw;',
    'float ign(vec2 p){ return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }',
    'void main(){',
    '  vec4 t = texture(tAcc, gl_FragCoord.xy / uRes);',
    '  if (uRaw > 0.5) { o = t; return; }',
    '  vec3 c = max(t.rgb, 0.0);',
    '  c = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));',
    '  o = vec4(c + (ign(gl_FragCoord.xy) - 0.5) / 255.0, 1.0);',
    '}'].join('\n');

  function halton(i, b) { var f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; }
  function srgb(c) { return c.map(function (x) { return Math.pow(x, 2.2); }); }

  /* opts: { capacity, page: [r,g,b] (sRGB, the colour the scene fades into) } */
  KL.renderer = function (host, camera, opts) {
    var r = new T.WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'high-performance' });
    if (!r.capabilities.isWebGL2) throw new Error('WebGL2 is required');
    var gl = r.getContext(), half = !!gl.getExtension('EXT_color_buffer_float');
    r.autoClear = false; r.domElement.className = 'lab-canvas'; host.insertBefore(r.domElement, host.firstChild);
    var cap = opts.capacity, W = 2, H = 2, count = 0, dirty = true, quality = ACC, live = 0;

    /* ---------- instances ---------- */
    var base = new T.PlaneGeometry(2 * EXT, 2 * EXT), geo = new T.InstancedBufferGeometry(), A = {};
    geo.index = base.index; geo.setAttribute('position', base.getAttribute('position'));
    [['iCenter', 3], ['iU', 3], ['iV', 3], ['iColor', 3], ['iAux', 4], ['iMeta', 4]].forEach(function (d) {
      var at = new T.InstancedBufferAttribute(new Float32Array(cap * d[1]), d[1]);
      at.setUsage(T.DynamicDrawUsage); geo.setAttribute(d[0], at); A[d[0]] = at;
    });
    geo.instanceCount = 0;

    var view = { uMode: { value: 0 }, uSplit: { value: new T.Vector3() }, uBand: { value: [0, 1, 2, 3, 4, 5, 6, 7] },
                 uRes: { value: new T.Vector2(2, 2) }, uFrame: { value: 0 } };
    var gMat = new T.ShaderMaterial({ glslVersion: T.GLSL3, vertexShader: G_VERT, fragmentShader: G_FRAG, side: T.DoubleSide,
      uniforms: Object.assign({ uHide: { value: 0 }, uBake: { value: 0 } }, view) });
    var sMat = new T.ShaderMaterial({ glslVersion: T.GLSL3, vertexShader: G_VERT, fragmentShader: S_FRAG, side: T.DoubleSide,
      uniforms: { uHide: gMat.uniforms.uHide, uFrame: view.uFrame } });
    var mesh = new T.Mesh(geo, gMat); mesh.frustumCulled = false;
    var scene = new T.Scene(); scene.add(mesh);

    /* ---------- targets ---------- */
    var gbuf = new T.WebGLMultipleRenderTargets(2, 2, 3, { minFilter: T.NearestFilter, magFilter: T.NearestFilter });
    gbuf.texture[1].type = gbuf.texture[2].type = half ? T.HalfFloatType : T.UnsignedByteType;
    gbuf.depthTexture = new T.DepthTexture(2, 2); gbuf.depthTexture.type = T.UnsignedIntType;
    var acc = new T.WebGLRenderTarget(2, 2, { type: half ? T.HalfFloatType : T.UnsignedByteType, minFilter: T.NearestFilter, magFilter: T.NearestFilter, depthBuffer: false });
    var shadow = new T.WebGLRenderTarget(SHADOW, SHADOW, { minFilter: T.NearestFilter, magFilter: T.NearestFilter });
    shadow.depthTexture = new T.DepthTexture(SHADOW, SHADOW); shadow.depthTexture.type = T.UnsignedIntType;
    var light = new T.OrthographicCamera(-1, 1, 1, -1, 0.1, 10), L0 = new T.Vector3(0.42, 0.84, 0.34).normalize(), fit = { c: new T.Vector3(), r: 4 };

    /* ---------- full-screen passes ---------- */
    var quad = new T.Mesh(new T.PlaneGeometry(2, 2)), fsScene = new T.Scene(), fsCam = new T.Camera(); quad.frustumCulled = false; fsScene.add(quad);
    var mats = [], pal = [], i;
    for (i = 0; i < MAXC; i++) { mats.push(new T.Vector4(0.6, 0, 0, 0)); pal.push(new T.Vector3(0.5, 0.5, 0.5)); }
    var page = srgb(opts.page || [1, 1, 1]);
    var lMat = new T.ShaderMaterial({ glslVersion: T.GLSL3, vertexShader: FS_VERT, fragmentShader: L_FRAG, depthTest: false, depthWrite: false,
      transparent: true, blending: T.CustomBlending, blendSrc: T.SrcAlphaFactor, blendDst: T.OneMinusSrcAlphaFactor,
      blendSrcAlpha: T.OneFactor, blendDstAlpha: T.ZeroFactor,
      uniforms: Object.assign({
        tG0: { value: gbuf.texture[0] }, tG1: { value: gbuf.texture[1] }, tG2: { value: gbuf.texture[2] }, tDepth: { value: gbuf.depthTexture },
        tShadow: { value: shadow.depthTexture }, uInvProj: { value: new T.Matrix4() }, uProj: { value: new T.Matrix4() },
        uCamWorld: { value: new T.Matrix4() }, uView: { value: new T.Matrix4() }, uLightVP: { value: new T.Matrix4() },
        uEye: { value: new T.Vector3() }, uLight: { value: L0.clone() }, uPage: { value: new T.Vector3().fromArray(page) }, uFade: { value: new T.Vector4(0, 0, 8, 12) },
        uDepthR: { value: new T.Vector2(1, 30) }, uAlpha: { value: 1 }, uAoR: { value: 0.55 }, uMat: { value: mats }, uPal: { value: pal },
        uTint: { value: new T.Vector3(0.81, 0.58, 0.85) }, uSelCol: { value: new T.Vector3().fromArray(srgb([0.16, 0.32, 0.86])) } }, view) });
    var oMat = new T.ShaderMaterial({ glslVersion: T.GLSL3, vertexShader: FS_VERT, fragmentShader: OUT_FRAG, depthTest: false, depthWrite: false,
      uniforms: { tAcc: { value: acc.texture }, uRes: view.uRes, uRaw: { value: 0 } } });
    var overlay = new T.Scene(), tmp = new T.Vector3();

    function pass(mat, target) { quad.material = mat; r.setRenderTarget(target); r.render(fsScene, fsCam); }
    function drawShadow(jx, jy) {
      // an orthographic box around the subject, seen from the key light (moved a little: an area light, once averaged)
      var d = tmp.copy(L0); d.x += jx; d.z += jy; d.normalize();
      var R = fit.r * 1.9;
      light.left = light.bottom = -R; light.right = light.top = R; light.near = 0.1; light.far = fit.r * 8;
      light.position.copy(fit.c).addScaledVector(d, fit.r * 4); light.lookAt(fit.c); light.updateProjectionMatrix(); light.updateMatrixWorld();
      lMat.uniforms.uLightVP.value.multiplyMatrices(light.projectionMatrix, light.matrixWorldInverse);
      mesh.material = sMat; r.setRenderTarget(shadow); r.clear(); r.render(scene, light);
    }
    function drawG() {
      mesh.material = gMat; r.setRenderTarget(gbuf); r.setClearColor(0x000000, 0); r.clear(); r.render(scene, camera);
      var u = lMat.uniforms;
      u.uInvProj.value.copy(camera.projectionMatrixInverse); u.uProj.value.copy(camera.projectionMatrix);
      u.uCamWorld.value.copy(camera.matrixWorld); u.uView.value.copy(camera.matrixWorldInverse);
      u.uEye.value.setFromMatrixPosition(camera.matrixWorld);     // not cameraPosition: the lighting pass is drawn with its own camera
    }

    var api = {
      canvas: r.domElement, three: r, overlay: overlay, half: half, maxCharts: MAXC,
      center: A.iCenter.array, axU: A.iU.array, axV: A.iV.array, color: A.iColor.array, aux: A.iAux.array, meta: A.iMeta.array,
      /* upload the first n instances; names = which attributes changed (all when omitted) */
      commit: function (n, names) {
        geo.instanceCount = Math.min(n, cap);
        (names || Object.keys(A)).forEach(function (k) {
          var at = A[k]; at.updateRange.offset = 0; at.updateRange.count = geo.instanceCount * at.itemSize; at.needsUpdate = true;
        });
        dirty = true;
      },
      /* charts: [{ mat: {rough, metal}, rgb: [r,g,b], ground }] — material and identity colour by chart index */
      setCharts: function (list) {
        list.forEach(function (c, k) { if (k < MAXC) { mats[k].set(c.mat.rough, c.mat.metal, c.ground ? 1 : 0, 0); pal[k].fromArray(c.rgb); } });
        dirty = true;
      },
      /* key light: azimuth (radians, like the orbit camera's) and elevation above the horizon */
      light: function (az, el) {
        L0.set(Math.cos(el) * Math.sin(az), Math.sin(el), Math.cos(el) * Math.cos(az)); lMat.uniforms.uLight.value.copy(L0); dirty = true;
      },
      /* subject bounds (for the shadow box and the floor fade) */
      frame: function (cx, cy, cz, radius, fade0, fade1) {
        fit.c.set(cx, cy, cz); fit.r = radius; lMat.uniforms.uFade.value.set(cx, cz, fade0, fade1);
        lMat.uniforms.uAoR.value = radius * 0.2; dirty = true;
      },
      /* soft = only the roll changed: keep averaging (see live) instead of starting over */
      view: function (mode, bands, roll, slant, soft) {
        view.uMode.value = mode;
        view.uSplit.value.set(bands ? bands.length : 0, roll || 0, slant || 0);
        if (bands) bands.forEach(function (b, k) { view.uBand.value[k] = b; });
        if (!soft) dirty = true;
      },
      /* k > 0: never stop, and blend each new frame in with at least weight k (for a picture that keeps moving a little) */
      live: function (k) { live = k; },
      progress: function () { return Math.min(1, count / quality); },
      depthRange: function (near, far) { lMat.uniforms.uDepthR.value.set(near, far); },
      hide: function (on) { gMat.uniforms.uHide.value = on ? 1 : 0; dirty = true; },
      quality: function (n) { quality = n; },
      invalidate: function () { dirty = true; },
      count: function () { return count; },
      resize: function (w, h, dpr) {
        r.setPixelRatio(dpr || Math.min(window.devicePixelRatio || 1, 2)); r.setSize(w, h, false);
        var s = r.getDrawingBufferSize(new T.Vector2()); W = s.x; H = s.y;
        gbuf.setSize(W, H); acc.setSize(W, H); view.uRes.value.set(W, H);
        camera.aspect = w / h; camera.updateProjectionMatrix(); dirty = true;
      },
      /* draws one frame if the picture is not converged yet; returns false when there was nothing to do */
      render: function () {
        if (dirty) { count = 0; dirty = false; }
        if (count >= quality && !live) return false;
        var jx = count ? halton(count % 256, 2) - 0.5 : 0, jy = count ? halton(count % 256, 3) - 0.5 : 0;
        view.uFrame.value = count % 256;
        camera.setViewOffset(W, H, jx, jy, W, H); camera.updateMatrixWorld();
        var a = halton(count + 1, 5) * 6.2832, d = count ? Math.sqrt(halton(count + 1, 7)) * 0.045 : 0;
        drawShadow(d * Math.cos(a), d * Math.sin(a));
        drawG();
        lMat.uniforms.uAlpha.value = Math.max(1 / (count + 1), live);
        pass(lMat, acc);
        camera.clearViewOffset();
        r.setRenderTarget(null); pass(oMat, null);
        r.render(overlay, camera);
        count++;
        return true;
      },
      /* draw the picture as it stands again, without adding a frame (e.g. right before reading the canvas) */
      present: function () { r.setRenderTarget(null); pass(oMat, null); r.render(overlay, camera); },
      /* bake helper: per pixel (u, v, chart index + 1, 1) of the nearest surface, hard-edged, as floats (rows bottom-up) */
      readIds: function () {
        var rt = new T.WebGLRenderTarget(W, H, { type: T.FloatType, minFilter: T.NearestFilter, magFilter: T.NearestFilter, depthBuffer: false });
        var out = new Float32Array(W * H * 4);
        gMat.uniforms.uBake.value = 1; camera.clearViewOffset(); camera.updateMatrixWorld(); drawG(); gMat.uniforms.uBake.value = 0;
        oMat.uniforms.tAcc.value = gbuf.texture[2]; oMat.uniforms.uRaw.value = 1; pass(oMat, rt);
        r.readRenderTargetPixels(rt, 0, 0, W, H, out);
        oMat.uniforms.tAcc.value = acc.texture; oMat.uniforms.uRaw.value = 0; rt.dispose(); dirty = true;
        return { w: W, h: H, data: out };
      },
      size: function () { return { w: W, h: H }; }
    };
    return api;
  };
})();
