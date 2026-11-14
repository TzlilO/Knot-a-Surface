/* Knot-a-Surface talk — three.js helpers shared by the WebGL modules (planoid, manifold).
   Needs THREE (talk/vendor/three.min.js) + talk/kit.js. Exposed as KnotTalk.gl. */
(function () {
  'use strict';
  var KT = window.KnotTalk, T = window.THREE;
  var gl = KT.gl = {};

  /* renderer bound to a module root; returns { renderer, canvas, resize(w,h) } */
  gl.renderer = function (root, camera) {
    var r = new T.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    r.setClearColor(0x07090f, 1);
    r.domElement.className = 'kt-canvas';
    root.insertBefore(r.domElement, root.firstChild);
    function resize(w, h) {
      r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      r.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    return { renderer: r, canvas: r.domElement, resize: resize };
  };

  /* Minimal orbit camera. Drag = rotate. Zoom only by pinch / ctrl+wheel when embedded so the
     deck keeps the scroll wheel; plain wheel zooms on the standalone page.
     pick(e) may return true to claim a pointerdown for the module (e.g. a control-point drag). */
  gl.orbit = function (camera, dom, o) {
    o = o || {};
    var cur = { th: o.theta || 0.7, ph: o.phi || 1.05, r: o.radius || 6, tx: 0, ty: 0, tz: 0 };
    if (o.target) { cur.tx = o.target[0]; cur.ty = o.target[1]; cur.tz = o.target[2]; }
    var goal = Object.assign({}, cur), down = null, api;
    var lim = { r0: o.minRadius || 1.2, r1: o.maxRadius || 14 };
    function place() {
      var s = Math.sin(cur.ph), c = Math.cos(cur.ph);
      camera.position.set(cur.tx + cur.r * s * Math.sin(cur.th), cur.ty + cur.r * c, cur.tz + cur.r * s * Math.cos(cur.th));
      camera.lookAt(cur.tx, cur.ty, cur.tz);
      camera.updateMatrixWorld();
    }
    dom.addEventListener('pointerdown', function (e) {
      if (o.pick && o.pick(e)) return;
      down = { x: e.clientX, y: e.clientY, id: e.pointerId }; dom.setPointerCapture(e.pointerId);
      api.touched = true;
    });
    dom.addEventListener('pointermove', function (e) {
      if (!down || e.pointerId !== down.id) return;
      goal.th -= (e.clientX - down.x) * 0.006;
      goal.ph = KT.clamp(goal.ph - (e.clientY - down.y) * 0.005, 0.12, 1.5);
      down.x = e.clientX; down.y = e.clientY;
    });
    function up(e) { if (down && e.pointerId === down.id) down = null; }
    dom.addEventListener('pointerup', up); dom.addEventListener('pointercancel', up);
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
      dragging: function () { return !!down; }
    };
    place();
    return api;
  };

  /* ---------- derived Gaussians ("planoids") as instanced, tangent-aligned quads ----------
     Each instance is a flat Gaussian: centre + two tangent axes (1 sigma). No sorting: the
     Gaussian fringe is resolved with a screen-space dither, the core writes depth. */
  var EXT = 2.15;
  var VERT = [
    'attribute vec3 iCenter; attribute vec3 iU; attribute vec3 iV; attribute vec3 iColor;',
    'uniform float uGain;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN;',
    'void main(){',
    '  vQ = position.xy; vCol = iColor; vN = normalize(cross(iU, iV));',
    '  vec3 p = iCenter + (iU * position.x + iV * position.y) * uGain;',
    '  gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);',
    '}'].join('\n');
  var FRAG = [
    'precision highp float;',
    'uniform vec3 uLight; uniform float uRim; uniform float uFlat;',
    'varying vec2 vQ; varying vec3 vCol; varying vec3 vN;',
    'void main(){',
    '  float g = exp(-0.5 * dot(vQ, vQ));',
    '  float n = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));',
    '  float a = smoothstep(0.28, 0.56, g);',
    '#ifdef BLEND',
    '  a = g;',
    '#else',
    '  if (a <= n) discard;',
    '#endif',
    '  float d = abs(dot(normalize(vN), uLight));',
    '  vec3 c = vCol * mix(0.40 + 0.60 * d, 1.0, uFlat);',
    '  c *= mix(1.0, 0.55 + 0.45 * smoothstep(0.12, 0.75, g), uRim);',
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
    ['iCenter', 'iU', 'iV', 'iColor'].forEach(function (n) {
      var at = new T.InstancedBufferAttribute(new Float32Array(capacity * 3), 3);
      at.setUsage(T.DynamicDrawUsage); g.setAttribute(n, at); A[n] = at;
    });
    g.instanceCount = 0;
    var mat = new T.ShaderMaterial({
      vertexShader: VERT, fragmentShader: (blend ? '#define BLEND\n' : '') + FRAG, side: T.DoubleSide,
      transparent: blend, depthWrite: !blend, depthTest: !blend,
      uniforms: { uGain: { value: 1 }, uRim: { value: 0.6 }, uFlat: { value: 0 },
                  uLight: { value: new T.Vector3(0.35, 0.86, 0.36).normalize() } }
    });
    var mesh = new T.Mesh(g, mat); mesh.frustumCulled = false;
    return {
      mesh: mesh, capacity: capacity, uniforms: mat.uniforms,
      center: A.iCenter.array, axU: A.iU.array, axV: A.iV.array, color: A.iColor.array,
      setCount: function (n) { g.instanceCount = Math.min(n, capacity); },
      commit: function () { A.iCenter.needsUpdate = A.iU.needsUpdate = A.iV.needsUpdate = A.iColor.needsUpdate = true; }
    };
  };

  /* Turn an evaluated (u,v) grid into planoids — the whole "method" in one loop:
       mean  = S(u,v)
       frame = [ S_u^ , S_v^perp ]            (rotation, from the tangents)
       scale = ( |S_u| du , |S_v^perp| dv )   (from the local sampling interval)
     grid = { pos, su, sv, col } with row = v index; du[a], dv[b] = sampling interval at each
     sample; k = sigma as a fraction of the interval; rgb (optional) overrides the colour.
     Returns the next free instance index. */
  gl.writePlanoids = function (sp, start, grid, na, nb, du, dv, k, rgb) {
    var P = grid.pos, SU = grid.su, SV = grid.sv, CL = grid.col, n = start, a, b, t, o;
    var cx = sp.center, ux = sp.axU, vx = sp.axV, cl = sp.color;
    for (b = 0; b < nb; b++) for (a = 0; a < na; a++) {
      if (n >= sp.capacity) return n;
      t = (b * na + a) * 3; o = n * 3;
      var sux = SU[t], suy = SU[t + 1], suz = SU[t + 2], svx = SV[t], svy = SV[t + 1], svz = SV[t + 2];
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
