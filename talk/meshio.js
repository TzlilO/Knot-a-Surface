/* Knot-a-Surface talk — small mesh writers (KnotTalk.meshio): PLY, GLB, OBJ and STL, with no library.
   A mesh to write = {
     name,                          what it is (file comments)
     unit,                          what one unit of the output is, e.g. 'm'
     scale,                         output units per scene unit (positions are multiplied by it)
     notes: [lines],                provenance, written into the file as comments / glTF asset extras
     parts: [{ name, pos, nrm, col, tri }]   one chart each: positions and unit normals (Float32Array, xyz per
                                    vertex), colours (Float32Array, rgb 0..1 as displayed, i.e. sRGB), triangles
                                    (Uint32Array, three indices into this part's vertices, counter-clockwise seen
                                    from the side the normals point to) }
   write(fmt, mesh) -> { data: ArrayBuffer | [string], mime, ext }; blob(out) -> a Blob of it.
   KnotTalk.saveFile(filename, blob) -> a promise: the ONE way the talk hands a file to the person. Defined here only
   when nothing defined it before; the default is a Blob URL + a[download], revoked later. A page may replace it (a
   review build wraps the file for its platform's save), so callers look it up when they save, not when they load.
     ply  binary little-endian: float x y z, float nx ny nz, uchar red green blue; faces as uchar count + int indices
     glb  glTF 2.0 binary: one node and mesh per part with POSITION (min / max), NORMAL, COLOR_0 (linear, as glTF
          wants) and uint32 indices; an unlit, double-sided material (the colours come from photographs and already
          hold their light); JSON and BIN chunks padded to 4 bytes
     obj  text: `v x y z r g b` (the common vertex-colour extension), `vn`, one `o` per part, 1-based `f v//vn`
     stl  binary: 80-byte header, triangle count, a facet normal and three corners per triangle (no colour)
   Plain functions; also require()-able under node (tests). */
(function (root) {
  'use strict';
  var KT = root.KnotTalk = root.KnotTalk || {};

  function count(mesh) {
    var v = 0, t = 0;
    mesh.parts.forEach(function (p) { v += p.pos.length / 3; t += p.tri.length / 3; });
    return { v: v, t: t };
  }
  function utf8(s) {
    if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(s);
    var b = unescape(encodeURIComponent(s)), u = new Uint8Array(b.length), i;
    for (i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
    return u;
  }
  function ascii(s) { return String(s).replace(/[^\x20-\x7e]/g, '?'); }   // header and comment lines stay 7-bit
  function byte(c) { return Math.max(0, Math.min(255, Math.round(c * 255))); }
  function toLinear(c) { return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }

  /* ---------- PLY ---------- */
  function ply(mesh) {
    var n = count(mesh), s = mesh.scale || 1;
    var head = 'ply\nformat binary_little_endian 1.0\n' +
      (mesh.notes || []).map(function (l) { return 'comment ' + ascii(l) + '\n'; }).join('') +
      'comment units: ' + ascii(mesh.unit || 'scene units') + '\n' +
      'element vertex ' + n.v + '\nproperty float x\nproperty float y\nproperty float z\n' +
      'property float nx\nproperty float ny\nproperty float nz\n' +
      'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
      'element face ' + n.t + '\nproperty list uchar int vertex_indices\nend_header\n';
    var hb = utf8(head), buf = new ArrayBuffer(hb.length + n.v * 27 + n.t * 13), dv = new DataView(buf), o = hb.length, base = 0;
    new Uint8Array(buf).set(hb, 0);
    mesh.parts.forEach(function (p) {
      var nv = p.pos.length / 3, i, k;
      for (i = 0; i < nv; i++) {
        for (k = 0; k < 3; k++) { dv.setFloat32(o, p.pos[i * 3 + k] * s, true); o += 4; }
        for (k = 0; k < 3; k++) { dv.setFloat32(o, p.nrm[i * 3 + k], true); o += 4; }
        for (k = 0; k < 3; k++) dv.setUint8(o++, byte(p.col[i * 3 + k]));
      }
    });
    mesh.parts.forEach(function (p) {
      var nt = p.tri.length / 3, i;
      for (i = 0; i < nt; i++) {
        dv.setUint8(o++, 3);
        dv.setInt32(o, base + p.tri[i * 3], true); dv.setInt32(o + 4, base + p.tri[i * 3 + 1], true); dv.setInt32(o + 8, base + p.tri[i * 3 + 2], true); o += 12;
      }
      base += p.pos.length / 3;
    });
    return buf;
  }

  /* ---------- GLB (glTF 2.0 binary) ---------- */
  function glb(mesh) {
    var s = mesh.scale || 1, views = [], acc = [], bins = [], off = 0;
    function view(arr, target) {
      views.push({ buffer: 0, byteOffset: off, byteLength: arr.byteLength, target: target });
      bins.push(arr); off += arr.byteLength;           // every array here holds 4-byte values: offsets stay aligned
      return views.length - 1;
    }
    var json = {
      asset: { version: '2.0', generator: 'Knot-a-Surface talk (talk/meshio.js)', extras: { name: mesh.name, unit: mesh.unit || 'scene units', notes: mesh.notes || [] } },
      extensionsUsed: ['KHR_materials_unlit'],
      scene: 0, scenes: [{ name: mesh.name, nodes: [] }], nodes: [], meshes: [],
      materials: [{ name: 'vertex colours, unlit', doubleSided: true,
                    pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 1 },
                    extensions: { KHR_materials_unlit: {} } }],
      buffers: [], bufferViews: views, accessors: acc
    };
    mesh.parts.forEach(function (p) {
      var nv = p.pos.length / 3, i, k, P = new Float32Array(nv * 3), N = new Float32Array(p.nrm), Cl = new Float32Array(nv * 3);
      var lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (i = 0; i < nv * 3; i++) { P[i] = p.pos[i] * s; Cl[i] = toLinear(Math.max(0, Math.min(1, p.col[i]))); }
      for (i = 0; i < nv; i++) for (k = 0; k < 3; k++) { var x = P[i * 3 + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x; }   // min / max of the float32 values written
      acc.push({ bufferView: view(P, 34962), componentType: 5126, count: nv, type: 'VEC3', min: lo, max: hi });
      acc.push({ bufferView: view(N, 34962), componentType: 5126, count: nv, type: 'VEC3' });
      acc.push({ bufferView: view(Cl, 34962), componentType: 5126, count: nv, type: 'VEC3' });
      acc.push({ bufferView: view(new Uint32Array(p.tri), 34963), componentType: 5125, count: p.tri.length, type: 'SCALAR' });
      var a = acc.length - 4;
      json.meshes.push({ name: p.name, primitives: [{ attributes: { POSITION: a, NORMAL: a + 1, COLOR_0: a + 2 }, indices: a + 3, material: 0, mode: 4 }] });
      json.nodes.push({ name: p.name, mesh: json.meshes.length - 1 });
      json.scenes[0].nodes.push(json.nodes.length - 1);
    });
    json.buffers.push({ byteLength: off });
    var jb = utf8(JSON.stringify(json)), jl = (jb.length + 3) & ~3, bl = (off + 3) & ~3;
    var buf = new ArrayBuffer(12 + 8 + jl + 8 + bl), dv = new DataView(buf), u8 = new Uint8Array(buf), o;
    dv.setUint32(0, 0x46546C67, true); dv.setUint32(4, 2, true); dv.setUint32(8, buf.byteLength, true);       // 'glTF', version 2, length
    dv.setUint32(12, jl, true); dv.setUint32(16, 0x4E4F534A, true); u8.set(jb, 20);                          // 'JSON'
    for (o = 20 + jb.length; o < 20 + jl; o++) u8[o] = 0x20;                                                // padded with spaces
    o = 20 + jl; dv.setUint32(o, bl, true); dv.setUint32(o + 4, 0x004E4942, true); o += 8;                   // 'BIN\0', padded with zeros
    bins.forEach(function (a) { u8.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), o); o += a.byteLength; });
    return buf;
  }

  /* ---------- OBJ (text, in chunks: a Blob takes the list as it is) ---------- */
  function obj(mesh) {
    var s = mesh.scale || 1, out = [], lines = [], base = 1;
    function flush() { out.push(lines.join('\n') + '\n'); lines = []; }
    lines.push('# ' + ascii(mesh.name));
    (mesh.notes || []).forEach(function (l) { lines.push('# ' + ascii(l)); });
    lines.push('# units: ' + ascii(mesh.unit || 'scene units') + '; vertex colours as v x y z r g b (0..1)');
    mesh.parts.forEach(function (p) {
      var nv = p.pos.length / 3, nt = p.tri.length / 3, i, P = p.pos, N = p.nrm, Cl = p.col;
      lines.push('o ' + ascii(p.name).replace(/\s+/g, '_'));
      for (i = 0; i < nv; i++) {
        lines.push('v ' + (P[i * 3] * s).toFixed(6) + ' ' + (P[i * 3 + 1] * s).toFixed(6) + ' ' + (P[i * 3 + 2] * s).toFixed(6) + ' ' +
                   Cl[i * 3].toFixed(4) + ' ' + Cl[i * 3 + 1].toFixed(4) + ' ' + Cl[i * 3 + 2].toFixed(4));
        if (lines.length > 4096) flush();
      }
      for (i = 0; i < nv; i++) {
        lines.push('vn ' + N[i * 3].toFixed(5) + ' ' + N[i * 3 + 1].toFixed(5) + ' ' + N[i * 3 + 2].toFixed(5));
        if (lines.length > 4096) flush();
      }
      for (i = 0; i < nt; i++) {
        var a = base + p.tri[i * 3], b = base + p.tri[i * 3 + 1], c = base + p.tri[i * 3 + 2];
        lines.push('f ' + a + '//' + a + ' ' + b + '//' + b + ' ' + c + '//' + c);
        if (lines.length > 4096) flush();
      }
      base += nv;
    });
    flush();
    return out;
  }

  /* ---------- STL (binary) ---------- */
  function stl(mesh) {
    var n = count(mesh), s = mesh.scale || 1, buf = new ArrayBuffer(84 + n.t * 50), dv = new DataView(buf), u8 = new Uint8Array(buf), o = 84, k;
    var head = ascii(mesh.name + ' | units: ' + (mesh.unit || 'scene units') + ' | ' + (mesh.notes || []).join(' | ')).slice(0, 80);
    for (k = 0; k < 80; k++) u8[k] = k < head.length ? head.charCodeAt(k) : 0x20;
    if (head.slice(0, 5).toLowerCase() === 'solid') u8[0] = 0x53;     // a binary STL must not start with "solid"
    dv.setUint32(80, n.t, true);
    mesh.parts.forEach(function (p) {
      var P = p.pos, nt = p.tri.length / 3, i;
      for (i = 0; i < nt; i++) {
        var a = p.tri[i * 3] * 3, b = p.tri[i * 3 + 1] * 3, c = p.tri[i * 3 + 2] * 3;
        var ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
        var nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, l = Math.hypot(nx, ny, nz) || 1;
        dv.setFloat32(o, nx / l, true); dv.setFloat32(o + 4, ny / l, true); dv.setFloat32(o + 8, nz / l, true); o += 12;
        [a, b, c].forEach(function (q) { dv.setFloat32(o, P[q] * s, true); dv.setFloat32(o + 4, P[q + 1] * s, true); dv.setFloat32(o + 8, P[q + 2] * s, true); o += 12; });
        dv.setUint16(o, 0, true); o += 2;
      }
    });
    return buf;
  }

  var FORMATS = {
    ply: { write: ply, mime: 'application/octet-stream', label: 'PLY · binary, vertex colours' },
    glb: { write: glb, mime: 'model/gltf-binary', label: 'GLB · glTF 2.0' },
    obj: { write: obj, mime: 'model/obj', label: 'OBJ · text' },
    stl: { write: stl, mime: 'model/stl', label: 'STL · binary, no colour' }
  };
  function write(fmt, mesh) { var f = FORMATS[fmt]; return { data: f.write(mesh), mime: f.mime, ext: fmt }; }
  function blob(out) { return new Blob(Array.isArray(out.data) ? out.data : [out.data], { type: out.mime }); }
  /* the default way to hand a file to the person: a link with a download name, clicked; its URL is revoked later */
  if (typeof KT.saveFile !== 'function') KT.saveFile = function (filename, b) {
    return new Promise(function (ok) {
      var url = URL.createObjectURL(b), a = document.createElement('a');
      a.href = url; a.download = filename; a.rel = 'noopener'; a.style.display = 'none';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 30000);
      ok();
    });
  };

  var api = { formats: Object.keys(FORMATS).map(function (k) { return { v: k, label: FORMATS[k].label }; }), write: write, blob: blob,
              ply: ply, glb: glb, obj: obj, stl: stl };
  KT.meshio = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
