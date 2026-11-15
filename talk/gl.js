/* Knot-a-Surface talk — three.js helpers shared by the WebGL modules (planoid, manifold).
   Needs THREE (talk/vendor/three.min.js) + talk/kit.js. Exposed as KnotTalk.gl. */
(function () {
  'use strict';
  var KT = window.KnotTalk, T = window.THREE;
  var gl = KT.gl = {};

  /* renderer bound to a module root; returns { renderer, canvas, resize(w,h) } */
  /* clear: omit for the dark instrument background, or 'transparent' to let the page show through */
  gl.renderer = function (root, camera, clear) {
    var see = clear === 'transparent';
    var r = new T.WebGLRenderer({ antialias: true, alpha: see, powerPreference: 'high-performance' });
    r.setClearColor(see ? 0x000000 : 0x07090f, see ? 0 : 1);
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
    'void main(){',
    '  vQ = position.xy; vCol = iColor; vAux = iAux; vN = normalize(cross(iU, iV));',
    '  vec3 p = iCenter + (iU * position.x + iV * position.y) * uGain; vW = p;',
    '  vec4 e = viewMatrix * vec4(p, 1.0); vZ = -e.z;',
    '  gl_Position = projectionMatrix * e;',
    '}'].join('\n');
  /* uMode picks what a Gaussian shows: 0 appearance, 1 depth, 2 normal, 3 the Gaussian itself (drawn
     smaller, so each one reads), 4 its aux colour (the chart it belongs to). With uSplit = (bands,
     roll, slant) the mode is chosen per pixel instead: slanted screen bands, one mode each, shifted
     by roll (a fraction of the width) — every mode of the same frame, side by side. */
  var FRAG = [
    'precision highp float;',
    'uniform vec3 uLight; uniform float uRim; uniform float uFlat;',
    'uniform float uMode; uniform vec3 uSplit; uniform vec2 uRes; uniform vec2 uDepth; uniform vec3 uTint;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN; varying vec3 vW; varying vec3 vAux; varying float vZ;',
    'vec3 ramp(float t){',                                         // near = light and warm, far = dark
    '  vec3 a = vec3(0.99, 0.95, 0.72), b = vec3(0.98, 0.55, 0.38), c = vec3(0.70, 0.21, 0.48), d = vec3(0.26, 0.09, 0.44), e = vec3(0.09, 0.06, 0.24);',
    '  t *= 4.0;',
    '  return t < 1.0 ? mix(a, b, t) : t < 2.0 ? mix(b, c, t - 1.0) : t < 3.0 ? mix(c, d, t - 2.0) : mix(d, e, t - 3.0);',
    '}',
    'void main(){',
    '  float m = uMode;',
    '  if (uSplit.x > 0.5) m = floor(fract((gl_FragCoord.x + uSplit.z * (uRes.y - gl_FragCoord.y)) / uRes.x - uSplit.y) * uSplit.x);',
    '  bool own = m > 2.5 && m < 3.5;',
    '  float g = exp(-0.5 * dot(vQ, vQ) * (own ? 3.6 : 1.0));',
    '  float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));',
    '  float a = smoothstep(0.28, 0.56, g);',
    '#ifdef BLEND',
    '  a = g;',
    '#else',
    '  if (a <= n) discard;',
    '#endif',
    '  vec3 nn = normalize(vN);',                                  // a planoid has two faces: light the one we see
    '  if (dot(nn, cameraPosition - vW) < 0.0) nn = -nn;',
    '  float d = max(dot(nn, uLight), 0.0);',
    '  vec3 c = (own ? uTint : m > 3.5 ? vAux : vCol) * mix(0.40 + 0.16 * nn.y + 0.66 * d, 1.0, uFlat);',
    '  c *= mix(1.0, 0.55 + 0.45 * smoothstep(0.12, 0.75, g), own ? 1.0 : uRim);',
    '  if (m > 0.5 && m < 1.5) c = ramp(clamp(log(vZ / uDepth.x) / log(uDepth.y / uDepth.x), 0.0, 1.0));',
    '  else if (m > 1.5 && m < 2.5) c = nn * 0.5 + 0.5;',
    '#ifdef BLEND',
    '  gl_FragColor = vec4(c, a * 0.92);',
    '#else',
    '  gl_FragColor = vec4(c, 1.0);',
    '#endif',
    '}'].join('\n');

  /* opts.blend: true alpha-blended Gaussians (smooth, unsorted — for a handful of showcase splats) */
  gl.splats = function (capacity, opts) {
    var blend = !!(opts && opts.blend);
    var base = new T.PlaneGeometry(2 * EXT, 2 * EXT);
    var g = new T.InstancedBufferGeometry();
    g.index = base.index; g.setAttribute('position', base.getAttribute('position'));
    var A = {};
    ['iCenter', 'iU', 'iV', 'iColor', 'iAux'].forEach(function (n) {
      var at = new T.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      at.setUsage(T.DynamicDrawUsage); g.setAttribute(n, at); A[n] = at;
    });
    g.instanceCount = 0;
    var mat = new T.ShaderMaterial({
      vertexShader: VERT, fragmentShader: (blend ? '#define BLEND\n' : '') + FRAG, side: T.DoubleSide,
      transparent: blend, depthWrite: !blend, depthTest: !blend,
      uniforms: { uGain: { value: 1 }, uRim: { value: 0.6 }, uFlat: { value: 0 },
                  uLight: { value: new T.Vector3(0.35, 0.86, 0.36).normalize() },
                  uMode: { value: 0 }, uSplit: { value: new T.Vector3(0, 0, 0) }, uRes: { value: new T.Vector2(1, 1) },
                  uDepth: { value: new T.Vector2(1, 10) }, uTint: { value: new T.Vector3().fromArray(gl.rgb(KT.color.splat)) } }
    });
    var mesh = new T.Mesh(g, mat); mesh.frustumCulled = false;
    return {
      mesh: mesh, capacity: capacity, uniforms: mat.uniforms,
      center: A.iCenter.array, axU: A.iU.array, axV: A.iV.array, color: A.iColor.array, aux: A.iAux.array,
      setCount: function (n) { g.instanceCount = Math.min(n, capacity); },
      /* upload the first n instances (all of them when n is omitted) */
      commit: function (n) {
        [A.iCenter, A.iU, A.iV, A.iColor, A.iAux].forEach(function (at) {
          at.updateRange.offset = 0; at.updateRange.count = n == null ? -1 : Math.min(n, capacity) * 3; at.needsUpdate = true;
        });
      }
    };
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
