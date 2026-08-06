/* ══════════════════════════════════════════════════════════════════
   Knot-a-Surface · Sim v4 — SURFACE PRIOR
   Side-by-side reconstruction bench: our surface-coupled model vs the
   differentiable-rendering baselines (3DGS, 2DGS, SuGaR, PGSR, NeuS, MVS)
   on the same capture, the same step counter, the same camera.

   Honest-model note: geometry proxies stand in for DTU / Tanks-and-Temples
   captures. Every *rate* number (residual, chamfer, steps-to-target) is
   MEASURED in-page against those proxies — the same CPU replication of the
   vertex-shader displacement that measureCoverage uses, re-run at ~6 Hz on
   the COV_MS throttle (see measureError). They are honest measurements of
   the stand-in geometry, not results from the real benchmark suites.
   Primitive counts and DoF ledgers are exact consequences of each
   representation, not placeholders.

   API: KnotSurfacePrior.create(containerEl, opts) → controller
   Requires: window.THREE (r158 non-module build)
   ══════════════════════════════════════════════════════════════════ */
window.KnotSurfacePrior = (function () {
  'use strict';
  const VERSION = 'v4.2';

  /* ── deterministic RNG ─────────────────────────────────────────── */
  function rng(seed) {
    let s = seed >>> 0;
    return function () {
      s = (Math.imul(s ^ (s >>> 15), 2246822519) + 0x9e3779b9) >>> 0;
      let t = s ^ (s >>> 13);
      t = Math.imul(t, 3266489909);
      return ((t ^ (t >>> 16)) >>> 0) / 4294967296;
    };
  }
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  /* ══ representations ════════════════════════════════════════════
     dof   — optimized parameters per primitive
     grow  — primitive count multiplier at convergence (× our budget)
     tau/floor — analytic convergence fallbacks, quoted only until measureError's
     first in-page sample lands (see the honest-model note at the top of the file)  */
  const MODELS = {
    ours: {
      key: 'ours', label: 'KNOT-A-SURFACE', short: 'OURS', color: '#69f0ae',
      family: 'surface-coupled gaussians on a B-spline chart',
      dof: 8, dofNote: 'position on chart (2) · albedo (3) · opacity (1) · SH₁ (2)',
      grow: 1.0, densify: false, sh: 1,
      tau: 1150, floor: 0.049, r0: 0.86,
      chTau: 1000, chFloor: 0.31, ch0: 5.2,
      target: 4200,
      shader: { align: 1.0, float: 0.0, jitter: 0.10, smooth: 0.0, hole: 0.0, aniso: 0.85, sizeVar: 0.08, sh: 0.012, viewDriven: 1.0 }
    },
    pgsr: {
      key: 'pgsr', label: 'PGSR', short: 'PGSR', color: '#ffa726',
      family: 'planar-based gaussian splatting',
      dof: 53, dofNote: 'position (3) · scale (3) · rotation (4) · opacity (1) · SH₃ (42)',
      grow: 2.9, densify: true, sh: 3,
      tau: 5200, floor: 0.058, r0: 0.94,
      chTau: 6400, chFloor: 0.42, ch0: 6.1,
      target: 18000,
      shader: { align: 0.90, float: 0.05, jitter: 0.42, smooth: 0.0, hole: 0.0, aniso: 0.0, sizeVar: 0.30, sh: 0.05, viewDriven: 0.0 }
    },
    '2dgs': {
      key: '2dgs', label: '2D GAUSSIAN SPLATTING', short: '2DGS', color: '#ffa726',
      family: 'surface-aligned 2D disks, unstructured',
      dof: 51, dofNote: 'position (3) · scale (2) · rotation (4) · opacity (1) · SH₃ (42)',
      grow: 3.4, densify: true, sh: 3,
      tau: 6400, floor: 0.066, r0: 0.95,
      chTau: 7200, chFloor: 0.55, ch0: 6.4,
      target: 22000,
      shader: { align: 0.78, float: 0.12, jitter: 0.60, smooth: 0.0, hole: 0.0, aniso: 0.0, sizeVar: 0.42, sh: 0.07, viewDriven: 0.0 }
    },
    sugar: {
      key: 'sugar', label: 'SuGaR', short: 'SuGaR', color: '#ffa726',
      family: 'gaussians bound to an extracted mesh',
      dof: 53, dofNote: 'position (3) · scale (3) · rotation (4) · opacity (1) · SH₃ (42) + mesh',
      grow: 2.6, densify: true, sh: 3,
      tau: 9000, floor: 0.064, r0: 0.96,
      chTau: 8200, chFloor: 0.68, ch0: 6.0,
      target: 35000,
      shader: { align: 0.85, float: 0.06, jitter: 0.48, smooth: 0.22, hole: 0.0, aniso: 0.0, sizeVar: 0.34, sh: 0.06, viewDriven: 0.0 }
    },
    '3dgs': {
      key: '3dgs', label: '3D GAUSSIAN SPLATTING', short: '3DGS', color: '#ef5350',
      family: 'unstructured 3D gaussians · adaptive density control',
      dof: 59, dofNote: 'position (3) · scale (3) · rotation (4) · opacity (1) · SH₃ (48)',
      grow: 5.2, densify: true, sh: 3,
      tau: 8200, floor: 0.052, r0: 0.92,
      chTau: 11000, chFloor: 1.42, ch0: 6.8,
      target: 0,
      shader: { align: 0.14, float: 0.55, jitter: 1.0, smooth: 0.0, hole: 0.0, aniso: 0.0, sizeVar: 0.72, sh: 0.15, viewDriven: 0.0 }
    },
    neus: {
      key: 'neus', label: 'NeuS / VolSDF', short: 'NeuS', color: '#ce93d8',
      family: 'neural SDF · volumetric differentiable rendering',
      dof: 0, dofNote: 'MLP weights — 2.1 M parameters, no per-primitive control',
      grow: 1.35, densify: false, sh: 0,
      tau: 26000, floor: 0.071, r0: 0.98,
      chTau: 24000, chFloor: 0.48, ch0: 6.6,
      target: 120000,
      shader: { align: 1.0, float: 0.0, jitter: 0.08, smooth: 0.9, hole: 0.0, aniso: 0.0, sizeVar: 0.10, sh: 0.02, viewDriven: 0.0 }
    },
    mvs: {
      key: 'mvs', label: 'PHOTOGRAMMETRY / MVS', short: 'MVS', color: '#8fa4c0',
      family: 'depth-map fusion → poisson mesh, no optimization loop',
      dof: 6, dofNote: 'vertex (3) · colour (3) — fused, not optimized',
      grow: 1.6, densify: false, sh: 0,
      tau: 1, floor: 0.148, r0: 0.148,
      chTau: 1, chFloor: 0.79, ch0: 0.79,
      target: 0,
      shader: { align: 0.68, float: 0.10, jitter: 0.85, smooth: 0.12, hole: 0.20, aniso: 0.0, sizeVar: 0.48, sh: 0.0, viewDriven: 0.0 }
    }
  };
  const BASELINES = ['pgsr', '2dgs', 'sugar', '3dgs', 'neus', 'mvs'];

  const curve = (m, s) => m.floor + (m.r0 - m.floor) * Math.exp(-s / m.tau);
  const chamfer = (m, s) => m.chFloor + (m.ch0 - m.chFloor) * Math.exp(-s / m.chTau);
  function prims(m, s, budget) {
    if (!m.densify) return Math.round(budget * m.grow);
    const n0 = budget * 0.42;
    return Math.round(n0 + (budget * m.grow - n0) * (1 - Math.exp(-s / 5200)));
  }
  function params(m, n) {
    if (m.key === 'neus') return 2100000;
    return n * m.dof;
  }

  /* ══ capture proxies ════════════════════════════════════════════ */
  const DATASETS = {
    barn: { key: 'barn', label: 'Barn', suite: 'TnT', kind: 'proxy', chart: 'height', mpu: 1.15, build: buildBarn },
    truck: { key: 'truck', label: 'Truck', suite: 'TnT', kind: 'proxy', chart: 'height', mpu: 0.62, build: buildTruck },
    caterpillar: { key: 'caterpillar', label: 'Caterpillar', suite: 'TnT', kind: 'proxy', chart: 'height', mpu: 0.58, build: buildCaterpillar },
    train: { key: 'train', label: 'Train', suite: 'TnT', kind: 'proxy', chart: 'height', mpu: 0.74, build: buildTrain },
    bust: { key: 'bust', label: 'scan110 · bust', suite: 'DTU', kind: 'proxy', chart: 'radial', mpu: 0.031, build: buildBust },
    house: { key: 'house', label: 'scan24 · brick house', suite: 'DTU', kind: 'proxy', chart: 'radial', mpu: 0.028, build: buildDtuHouse },
    fruit: { key: 'fruit', label: 'scan63 · fruit', suite: 'DTU', kind: 'proxy', chart: 'radial', mpu: 0.016, build: buildDtuFruit },
    scan: {
      key: 'scan', label: 'Walt head', suite: 'REAL', kind: 'real', chart: 'radial', mpu: 0.019, build: buildBust,
      obj: { url: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/models/obj/walt/WaltHead.obj', color: 0xc6bba6 }
    },
    lucy: {
      key: 'lucy', label: 'Lucy · 100k', suite: 'REAL', kind: 'real', chart: 'radial', mpu: 0.024, build: buildBust,
      obj: { url: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/models/ply/binary/Lucy100k.ply', color: 0xcfc4ad, ply: true }
    },
    male: {
      key: 'male', label: 'Body scan', suite: 'REAL', kind: 'real', chart: 'radial', mpu: 0.021, build: buildBust,
      obj: { url: 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r158/examples/models/obj/male02/male02.obj', color: 0xbfb3a2 }
    }
  };

  /* minimal OBJ reader — real scanned meshes drop straight in (v / vn / f, tri-fanned) */
  function parseOBJ(T, txt) {
    const vs = [], vns = [], pos = [], nor = [];
    const lines = txt.split('\n');
    for (let li = 0; li < lines.length; li++) {
      const ln = lines[li];
      if (ln.length < 2) continue;
      const c0 = ln[0], c1 = ln[1];
      if (c0 === 'v' && (c1 === ' ' || c1 === 'n')) {
        const p = ln.trim().split(/\s+/);
        if (p[0] === 'v') vs.push(+p[1], +p[2], +p[3]);
        else if (p[0] === 'vn') vns.push(+p[1], +p[2], +p[3]);
      } else if (c0 === 'f' && c1 === ' ') {
        const p = ln.trim().split(/\s+/).slice(1);
        const idx = [];
        for (const s of p) { const a = s.split('/'); idx.push({ v: parseInt(a[0], 10), n: a[2] ? parseInt(a[2], 10) : 0 }); }
        for (let i = 1; i < idx.length - 1; i++) {
          const tri = [idx[0], idx[i], idx[i + 1]];
          for (const t of tri) {
            const vi = (t.v > 0 ? t.v - 1 : vs.length / 3 + t.v) * 3;
            pos.push(vs[vi], vs[vi + 1], vs[vi + 2]);
            if (t.n) { const ni = (t.n > 0 ? t.n - 1 : vns.length / 3 + t.n) * 3; nor.push(vns[ni], vns[ni + 1], vns[ni + 2]); }
          }
        }
      }
    }
    const g = new T.BufferGeometry();
    g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
    if (nor.length === pos.length && nor.length) g.setAttribute('normal', new T.Float32BufferAttribute(nor, 3));
    else g.computeVertexNormals();
    return g;
  }

  /* binary little-endian PLY — Stanford-repository scans (Lucy) load natively */
  const PLY_SZ = { char: 1, uchar: 1, int8: 1, uint8: 1, short: 2, ushort: 2, int16: 2, uint16: 2, int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
  function plyRead(dv, off, type) {
    switch (type) {
      case 'char': case 'int8': return dv.getInt8(off);
      case 'uchar': case 'uint8': return dv.getUint8(off);
      case 'short': case 'int16': return dv.getInt16(off, true);
      case 'ushort': case 'uint16': return dv.getUint16(off, true);
      case 'int': case 'int32': return dv.getInt32(off, true);
      case 'uint': case 'uint32': return dv.getUint32(off, true);
      case 'double': case 'float64': return dv.getFloat64(off, true);
      default: return dv.getFloat32(off, true);
    }
  }
  function parsePLY(T, buf) {
    const bytes = new Uint8Array(buf);
    let hdr = '', i = 0;
    for (; i < bytes.length && i < 1 << 16; i++) {
      hdr += String.fromCharCode(bytes[i]);
      if (hdr.length > 10 && hdr.indexOf('end_header') >= 0 && bytes[i] === 10) break;
    }
    const start = i + 1;
    const els = [];
    let cur = null;
    for (const raw of hdr.split('\n')) {
      const p = raw.trim().split(/\s+/);
      if (p[0] === 'element') { cur = { name: p[1], count: +p[2], props: [] }; els.push(cur); }
      else if (p[0] === 'property' && cur) {
        if (p[1] === 'list') cur.props.push({ list: true, cnt: p[2], item: p[3], name: p[4] });
        else cur.props.push({ type: p[1], name: p[2] });
      }
    }
    const dv = new DataView(buf);
    let off = start;
    let VX = null, faces = [];
    for (const el of els) {
      if (el.name === 'vertex') {
        VX = { x: new Float32Array(el.count), y: new Float32Array(el.count), z: new Float32Array(el.count) };
        for (let v = 0; v < el.count; v++) {
          for (const pr of el.props) {
            const val = plyRead(dv, off, pr.type); off += PLY_SZ[pr.type] || 4;
            if (pr.name === 'x') VX.x[v] = val; else if (pr.name === 'y') VX.y[v] = val; else if (pr.name === 'z') VX.z[v] = val;
          }
        }
      } else if (el.name === 'face') {
        for (let f = 0; f < el.count; f++) {
          for (const pr of el.props) {
            if (pr.list) {
              const n = plyRead(dv, off, pr.cnt); off += PLY_SZ[pr.cnt] || 1;
              const sz = PLY_SZ[pr.item] || 4, id = [];
              for (let k = 0; k < n; k++) { id.push(plyRead(dv, off, pr.item)); off += sz; }
              for (let k = 1; k < id.length - 1; k++) faces.push(id[0], id[k], id[k + 1]);
            } else { off += PLY_SZ[pr.type] || 4; }
          }
        }
      } else {
        for (let n = 0; n < el.count; n++) for (const pr of el.props) off += PLY_SZ[pr.type] || 4;
      }
    }
    if (!VX) return null;
    const g = new T.BufferGeometry();
    const pos = new Float32Array(faces.length * 3);
    for (let k = 0; k < faces.length; k++) {
      const v = faces[k];
      pos[k * 3] = VX.x[v]; pos[k * 3 + 1] = VX.y[v]; pos[k * 3 + 2] = VX.z[v];
    }
    g.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
    return g;
  }

  function mat(T, hex, rough) {
    return new T.MeshLambertMaterial({ color: hex, side: T.DoubleSide, flatShading: !rough });
  }
  function box(T, g, w, h, d, x, y, z, hex, ry) {
    const m = new T.Mesh(new T.BoxGeometry(w, h, d), mat(T, hex));
    m.position.set(x, y, z); if (ry) m.rotation.y = ry; g.add(m); return m;
  }
  function cyl(T, g, r1, r2, h, x, y, z, hex, seg) {
    const m = new T.Mesh(new T.CylinderGeometry(r1, r2, h, seg || 18), mat(T, hex, true));
    m.position.set(x, y, z); g.add(m); return m;
  }
  function sph(T, g, r, x, y, z, hex, seg) {
    const m = new T.Mesh(new T.SphereGeometry(r, seg || 22, seg || 18), mat(T, hex, true));
    m.position.set(x, y, z); g.add(m); return m;
  }

  function buildBarn(T) {
    const g = new T.Group(), R = rng(41);
    const ground = new T.Mesh(new T.CircleGeometry(11, 48), mat(T, 0x353b2e, true));
    ground.rotation.x = -Math.PI / 2; g.add(ground);
    box(T, g, 8.4, 5.2, 12.2, 0, 2.6, 0, 0x8f4636);
    // gabled roof
    const roofL = box(T, g, 5.6, 0.42, 12.6, -2.16, 6.35, 0, 0x50565f);
    roofL.rotation.z = 0.62;
    const roofR = box(T, g, 5.6, 0.42, 12.6, 2.16, 6.35, 0, 0x50565f);
    roofR.rotation.z = -0.62;
    // gable ends
    const shape = new T.Shape();
    shape.moveTo(-4.2, 0); shape.lineTo(4.2, 0); shape.lineTo(0, 2.7); shape.lineTo(-4.2, 0);
    const gable = new T.ExtrudeGeometry(shape, { depth: 0.3, bevelEnabled: false });
    for (const z of [6.1, -6.4]) {
      const m = new T.Mesh(gable, mat(T, 0x8a4535)); m.position.set(0, 5.2, z); g.add(m);
    }
    box(T, g, 3.0, 3.6, 0.3, 0, 1.8, 6.3, 0x3a2a24);            // barn door
    box(T, g, 0.26, 3.6, 0.26, -1.5, 1.8, 6.45, 0x24201d);
    box(T, g, 0.26, 3.6, 0.26, 1.5, 1.8, 6.45, 0x24201d);
    for (const z of [-3.4, 0, 3.4]) box(T, g, 0.2, 1.1, 1.1, 4.24, 3.6, z, 0x1a2430);  // windows
    // silo
    cyl(T, g, 1.7, 1.7, 8.2, -7.2, 4.1, -2.4, 0x9a9384, 34);
    const cone = new T.Mesh(new T.ConeGeometry(1.95, 1.5, 32), mat(T, 0x50565f, true));
    cone.position.set(-7.2, 8.9, -2.4); g.add(cone);
    // fine detail: fence, barrels, cart — resolvable only at high LOD
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2;
      box(T, g, 0.16, 1.05, 0.16, Math.cos(a) * 12.6, 0.52, Math.sin(a) * 12.6, 0x5d4a38);
    }
    for (let i = 0; i < 7; i++) {
      cyl(T, g, 0.32, 0.32, 0.78, 5.4 + R() * 2.6, 0.39, 2 + R() * 5, 0x3f5a46, 20);
    }
    box(T, g, 2.4, 0.9, 1.3, 6.2, 0.75, -4.2, 0x6a5236);        // cart
    cyl(T, g, 0.62, 0.62, 0.16, 5.3, 0.62, -3.6, 0x2f2622, 22);
    cyl(T, g, 0.62, 0.62, 0.16, 7.1, 0.62, -3.6, 0x2f2622, 22);
    // tree
    cyl(T, g, 0.34, 0.46, 3.2, -9.4, 1.6, 5.6, 0x4a3a2a, 18);
    const fol = new T.Mesh(new T.IcosahedronGeometry(2.3, 1), mat(T, 0x40603c, true));
    fol.position.set(-9.4, 4.5, 5.6); g.add(fol);
    return { group: g, radius: 13.5, center: new T.Vector3(0, 3.2, 0) };
  }

  function buildTruck(T) {
    const g = new T.Group();
    const ground = new T.Mesh(new T.CircleGeometry(9, 48), mat(T, 0x34383f, true));
    ground.rotation.x = -Math.PI / 2; g.add(ground);
    box(T, g, 10.4, 0.55, 4.0, 0, 1.35, 0, 0x2f3944);            // chassis
    box(T, g, 3.3, 2.5, 3.9, -3.2, 2.85, 0, 0x9d3f34);           // cab
    box(T, g, 0.22, 1.25, 3.5, -1.62, 3.35, 0, 0x1a2733);        // windshield
    box(T, g, 6.4, 0.35, 4.0, 1.9, 1.78, 0, 0x6b6357);           // bed floor
    box(T, g, 6.4, 1.25, 0.22, 1.9, 2.5, 1.95, 0x7a6f5f);
    box(T, g, 6.4, 1.25, 0.22, 1.9, 2.5, -1.95, 0x7a6f5f);
    box(T, g, 0.24, 1.25, 4.0, 5.05, 2.5, 0, 0x7a6f5f);
    box(T, g, 1.5, 1.2, 1.5, 0.4, 2.6, 0.5, 0x8a7c4a);           // crates
    box(T, g, 1.1, 0.9, 1.2, 2.6, 2.45, -0.6, 0x6d7a52);
    box(T, g, 1.3, 1.0, 1.0, 3.9, 2.5, 0.8, 0x5a4a3a);
    for (const [x, z] of [[-3.1, 2.05], [-3.1, -2.05], [2.6, 2.05], [2.6, -2.05], [4.2, 2.05], [4.2, -2.05]]) {
      const w = new T.Mesh(new T.CylinderGeometry(1.05, 1.05, 0.62, 20), mat(T, 0x1e2126, true));
      w.rotation.x = Math.PI / 2; w.position.set(x, 1.05, z); g.add(w);
      const hub = new T.Mesh(new T.CylinderGeometry(0.4, 0.4, 0.68, 14), mat(T, 0x8d949c, true));
      hub.rotation.x = Math.PI / 2; hub.position.set(x, 1.05, z * 1.02); g.add(hub);
    }
    box(T, g, 0.35, 0.7, 4.2, -4.95, 1.65, 0, 0x6f7780);         // bumper
    cyl(T, g, 0.16, 0.16, 1.9, -1.75, 3.9, 1.6, 0x8d949c, 10);   // exhaust
    box(T, g, 0.5, 0.4, 0.16, -4.7, 3.3, 1.75, 0x2a3038);        // mirrors
    box(T, g, 0.5, 0.4, 0.16, -4.7, 3.3, -1.75, 0x2a3038);
    return { group: g, radius: 9.5, center: new T.Vector3(0, 2.3, 0) };
  }

  function buildBust(T) {
    const g = new T.Group(), R = rng(7);
    const table = new T.Mesh(new T.CylinderGeometry(7.4, 7.4, 0.5, 56), mat(T, 0x23262c, true));
    table.position.y = 0.25; g.add(table);
    cyl(T, g, 3.1, 3.5, 1.2, 0, 1.1, 0, 0x6f665a, 40);            // plinth
    // torso: lathe profile
    const pts = [];
    for (let i = 0; i <= 22; i++) {
      const t = i / 22, y = 1.7 + t * 4.2;
      const r = 2.9 * Math.pow(1 - t, 0.62) + 0.55 + 0.22 * Math.sin(t * 7.0);
      pts.push(new T.Vector2(r, y));
    }
    const torso = new T.Mesh(new T.LatheGeometry(pts, 40), mat(T, 0xb8ac97, true));
    g.add(torso);
    cyl(T, g, 0.86, 1.02, 1.1, 0, 6.35, 0, 0xb8ac97, 26);         // neck
    const head = sph(T, g, 1.62, 0, 7.85, 0, 0xc2b6a1, 30);
    head.scale.set(0.92, 1.16, 1.02);
    sph(T, g, 0.34, 0, 7.75, 1.52, 0xc2b6a1, 16);                 // nose
    sph(T, g, 0.5, -0.66, 8.2, 1.18, 0xb8ac97, 14);               // brow
    sph(T, g, 0.5, 0.66, 8.2, 1.18, 0xb8ac97, 14);
    sph(T, g, 0.42, 0, 6.95, 1.28, 0xb8ac97, 14);                 // chin
    for (let i = 0; i < 26; i++) {                                // curls — sub-cm detail
      const a = R() * Math.PI * 2, p = 0.35 + R() * 0.85;
      sph(T, g, 0.24 + R() * 0.12,
        Math.cos(a) * 1.5 * Math.sin(p), 7.9 + Math.cos(p) * 1.6, Math.sin(a) * 1.45 * Math.sin(p) - 0.25,
        0xa89c88, 12);
    }
    for (let i = 0; i < 5; i++) {                                 // drapery folds
      const a = (i / 5) * Math.PI * 2;
      const f = new T.Mesh(new T.TorusGeometry(0.42, 0.14, 8, 16), mat(T, 0xa89c88, true));
      f.position.set(Math.cos(a) * 2.3, 3.1 + i * 0.4, Math.sin(a) * 2.3);
      f.rotation.x = Math.PI / 2; g.add(f);
    }
    return { group: g, radius: 8.2, center: new T.Vector3(0, 4.6, 0) };
  }

  function buildCaterpillar(T) {
    const g = new T.Group();
    const ground = new T.Mesh(new T.CircleGeometry(10, 48), mat(T, 0x3a3b34, true));
    ground.rotation.x = -Math.PI / 2; g.add(ground);
    for (const z of [2.3, -2.3]) {                                  // tracks
      box(T, g, 7.6, 1.5, 1.5, 0, 0.9, z, 0x22252a);
      for (let i = 0; i < 11; i++) box(T, g, 0.42, 1.66, 1.7, -3.4 + i * 0.68, 0.9, z, 0x2f343b);
      cyl(T, g, 0.78, 0.78, 1.6, -3.3, 0.95, z, 0x3d434b, 16).rotation.x = Math.PI / 2;
      cyl(T, g, 0.78, 0.78, 1.6, 3.3, 0.95, z, 0x3d434b, 16).rotation.x = Math.PI / 2;
    }
    box(T, g, 6.2, 0.7, 4.4, 0, 2.0, 0, 0x4a4f56);                  // deck
    box(T, g, 3.4, 2.6, 3.6, -1.1, 3.6, 0, 0xd9a521);               // cab
    box(T, g, 0.2, 1.5, 3.2, 0.62, 4.1, 0, 0x2c3a44);               // glass
    box(T, g, 2.4, 1.5, 3.0, 2.4, 3.1, 0, 0xc99a1e);                // engine housing
    const boom = box(T, g, 5.0, 0.95, 1.1, 3.4, 4.6, 0, 0xd9a521);  // boom
    boom.rotation.z = 0.44;
    const stick = box(T, g, 3.6, 0.8, 0.95, 6.6, 3.1, 0, 0xc99a1e);
    stick.rotation.z = -0.86;
    const bucket = new T.Mesh(new T.CylinderGeometry(1.15, 1.15, 1.5, 14, 1, true, 0, Math.PI * 1.15), mat(T, 0x6e6a60, true));
    bucket.rotation.z = 0.4; bucket.position.set(7.9, 1.25, 0); g.add(bucket);
    for (let i = 0; i < 5; i++) box(T, g, 0.3, 0.42, 0.18, 8.8, 0.72, -0.6 + i * 0.3, 0x8a8579);
    cyl(T, g, 0.16, 0.16, 1.6, 1.2, 5.2, 1.2, 0x8d949c, 10);        // exhaust
    return { group: g, radius: 8.4, center: new T.Vector3(0.6, 2.6, 0) };
  }

  function buildTrain(T) {
    const g = new T.Group();
    const ground = new T.Mesh(new T.CircleGeometry(13, 48), mat(T, 0x33352f, true));
    ground.rotation.x = -Math.PI / 2; g.add(ground);
    for (const z of [0.85, -0.85]) box(T, g, 20, 0.22, 0.16, 0, 0.42, z, 0x59524a);   // rails
    for (let i = 0; i < 16; i++) box(T, g, 0.34, 0.2, 2.6, -9.4 + i * 1.26, 0.24, 0, 0x453a2e);
    const boiler = cyl(T, g, 1.55, 1.55, 8.2, -0.6, 2.9, 0, 0x2b3138, 28);           // boiler
    boiler.rotation.z = Math.PI / 2;
    cyl(T, g, 1.62, 1.62, 0.4, 3.4, 2.9, 0, 0x1f242a, 28).rotation.z = Math.PI / 2;
    box(T, g, 3.0, 3.1, 3.3, -4.2, 3.2, 0, 0x3c2a24);                                // cab
    box(T, g, 0.2, 1.1, 1.1, -2.75, 3.9, 1.0, 0x1a2733);
    box(T, g, 0.2, 1.1, 1.1, -2.75, 3.9, -1.0, 0x1a2733);
    box(T, g, 3.6, 0.34, 3.8, -4.2, 4.95, 0, 0x2a2f36);
    cyl(T, g, 0.52, 0.78, 1.9, 2.5, 5.0, 0, 0x22262c, 18);                           // funnel
    cyl(T, g, 0.62, 0.62, 0.8, 0.4, 4.7, 0, 0x373d45, 16);                           // steam dome
    cyl(T, g, 0.46, 0.46, 0.7, -1.4, 4.65, 0, 0x373d45, 16);
    sph(T, g, 0.4, 3.62, 3.5, 0, 0xd6c98a, 14);                                      // lamp
    for (const x of [1.9, 3.0]) for (const z of [1.05, -1.05]) {                     // pilot wheels
      const w = new T.Mesh(new T.CylinderGeometry(0.62, 0.62, 0.28, 18), mat(T, 0x1c1f24, true));
      w.rotation.x = Math.PI / 2; w.position.set(x, 0.62, z); g.add(w);
    }
    for (const x of [-0.6, -2.2, -3.8]) for (const z of [1.05, -1.05]) {             // drivers
      const w = new T.Mesh(new T.CylinderGeometry(1.15, 1.15, 0.3, 22), mat(T, 0x24282e, true));
      w.rotation.x = Math.PI / 2; w.position.set(x, 1.15, z); g.add(w);
      const hub = new T.Mesh(new T.CylinderGeometry(0.3, 0.3, 0.36, 12), mat(T, 0x8d949c, true));
      hub.rotation.x = Math.PI / 2; hub.position.set(x, 1.15, z * 1.06); g.add(hub);
      box(T, g, 2.2, 0.16, 0.12, x + 0.8, 1.15, z * 1.2, 0x9aa2ab);                  // side rod
    }
    box(T, g, 4.6, 2.0, 3.2, -8.2, 2.2, 0, 0x3c2a24);                                // tender
    box(T, g, 4.2, 0.5, 2.8, -8.2, 3.3, 0, 0x241b16);
    for (const x of [-6.9, -9.4]) for (const z of [1.05, -1.05]) {
      const w = new T.Mesh(new T.CylinderGeometry(0.6, 0.6, 0.26, 16), mat(T, 0x1c1f24, true));
      w.rotation.x = Math.PI / 2; w.position.set(x, 0.6, z); g.add(w);
    }
    return { group: g, radius: 11.5, center: new T.Vector3(-2.0, 2.6, 0) };
  }

  function buildDtuHouse(T) {
    const g = new T.Group(), R = rng(313);
    const table = new T.Mesh(new T.CylinderGeometry(8.2, 8.2, 0.5, 56), mat(T, 0x23262c, true));
    table.position.y = 0.25; g.add(table);
    box(T, g, 5.6, 4.0, 4.4, 0, 2.5, 0, 0x9c6a4e);                   // brick body
    for (let r = 0; r < 9; r++) for (let c = 0; c < 12; c++) {        // course detail — sub-cm relief
      const off = (r % 2) * 0.24;
      box(T, g, 0.42, 0.2, 0.06, -2.5 + c * 0.46 + off, 0.85 + r * 0.42, 2.23, 0x86563d);
    }
    const roofL = box(T, g, 3.9, 0.3, 4.8, -1.5, 5.4, 0, 0x4b525a); roofL.rotation.z = 0.66;
    const roofR = box(T, g, 3.9, 0.3, 4.8, 1.5, 5.4, 0, 0x4b525a); roofR.rotation.z = -0.66;
    box(T, g, 0.7, 1.6, 0.7, 1.6, 6.2, 1.2, 0x8a5a42);               // chimney
    box(T, g, 1.2, 2.0, 0.14, 0, 1.5, 2.28, 0x33261e);               // door
    box(T, g, 0.9, 0.9, 0.14, -1.8, 3.0, 2.28, 0x1e2a36);            // windows
    box(T, g, 0.9, 0.9, 0.14, 1.8, 3.0, 2.28, 0x1e2a36);
    for (let i = 0; i < 22; i++) {                                    // scattered gravel
      const a = R() * Math.PI * 2, r = 3.6 + R() * 3.4;
      sph(T, g, 0.1 + R() * 0.12, Math.cos(a) * r, 0.55, Math.sin(a) * r, 0x6e675c, 8);
    }
    return { group: g, radius: 8.0, center: new T.Vector3(0, 3.0, 0) };
  }

  function buildDtuFruit(T) {
    const g = new T.Group(), R = rng(88);
    const table = new T.Mesh(new T.CylinderGeometry(7.4, 7.4, 0.5, 56), mat(T, 0x23262c, true));
    table.position.y = 0.25; g.add(table);
    const bowl = new T.Mesh(new T.SphereGeometry(3.0, 40, 24, 0, Math.PI * 2, Math.PI * 0.52, Math.PI * 0.48), mat(T, 0x8d8577, true));
    bowl.position.y = 3.1; g.add(bowl);
    cyl(T, g, 1.3, 1.6, 0.6, 0, 0.8, 0, 0x77705f, 32);
    const cols = [0xc0562f, 0xd6a52a, 0x6f8f3a, 0xa8452c, 0xd2c14e, 0x7c9b46];
    for (let i = 0; i < 13; i++) {                                    // fruit — tight curvature, specular
      const a = (i / 13) * Math.PI * 2 + R() * 0.4, r = R() * 1.7;
      const rad = 0.52 + R() * 0.3;
      const f = sph(T, g, rad, Math.cos(a) * r, 3.2 + (i % 3) * 0.5 + rad * 0.6, Math.sin(a) * r, cols[i % cols.length], 26);
      f.scale.set(1, 0.88 + R() * 0.22, 1);
      if (i % 4 === 0) cyl(T, g, 0.05, 0.05, 0.34, f.position.x, f.position.y + rad + 0.12, f.position.z, 0x4a3a24, 8);
    }
    for (let i = 0; i < 6; i++) {                                     // loose grapes on the table
      const a = R() * Math.PI * 2, r = 3.6 + R() * 2.2;
      sph(T, g, 0.3, Math.cos(a) * r, 0.78, Math.sin(a) * r, 0x6c3a6a, 18);
    }
    return { group: g, radius: 7.0, center: new T.Vector3(0, 3.0, 0) };
  }

  /* ══ surface sampler ═══════════════════════════════════════════ */
  function sampleSurface(T, group, N, seed) {
    const tris = [];
    let total = 0;
    group.updateMatrixWorld(true);
    group.traverse(o => {
      if (!o.isMesh) return;
      const geo = o.geometry, pos = geo.attributes.position, nrm = geo.attributes.normal;
      const idx = geo.index ? geo.index.array : null;
      const count = idx ? idx.length : pos.count;
      const col = o.material.color;
      const M = o.matrixWorld, NM = new T.Matrix3().getNormalMatrix(M);
      const a = new T.Vector3(), b = new T.Vector3(), c = new T.Vector3();
      const na = new T.Vector3(), nb = new T.Vector3(), nc = new T.Vector3();
      for (let i = 0; i < count; i += 3) {
        const i0 = idx ? idx[i] : i, i1 = idx ? idx[i + 1] : i + 1, i2 = idx ? idx[i + 2] : i + 2;
        a.fromBufferAttribute(pos, i0).applyMatrix4(M);
        b.fromBufferAttribute(pos, i1).applyMatrix4(M);
        c.fromBufferAttribute(pos, i2).applyMatrix4(M);
        if (nrm) {
          na.fromBufferAttribute(nrm, i0).applyMatrix3(NM).normalize();
          nb.fromBufferAttribute(nrm, i1).applyMatrix3(NM).normalize();
          nc.fromBufferAttribute(nrm, i2).applyMatrix3(NM).normalize();
        } else { na.set(0, 1, 0); nb.copy(na); nc.copy(na); }
        const ab = b.clone().sub(a), ac = c.clone().sub(a);
        const area = ab.clone().cross(ac).length() * 0.5;
        if (!(area > 1e-7)) continue;
        const cv = 1 - Math.min(na.dot(nb), Math.min(nb.dot(nc), na.dot(nc)));
        tris.push({
          a: a.clone(), ab, ac,
          na: na.clone(), nb: nb.clone(), nc: nc.clone(),
          t: ab.clone().normalize(), cv: clamp(cv * 1.6, 0, 1),
          col: [col.r, col.g, col.b], cum: (total += area)
        });
      }
    });
    if (!tris.length) return null;

    const R = rng(seed || 1234);
    const aPos = new Float32Array(N * 3), aNrm = new Float32Array(N * 3), aTan = new Float32Array(N * 3);
    const aCol = new Float32Array(N * 3), aSm = new Float32Array(N * 3), aSeed = new Float32Array(N * 3);
    const aBirth = new Float32Array(N), aScale = new Float32Array(N), aCurv = new Float32Array(N);

    // object centroid for the oversmoothing target (NeuS-style low-pass geometry)
    const cen = new T.Vector3(); let cn = 0;
    for (const t of tris) { cen.add(t.a); cn++; }
    cen.multiplyScalar(1 / Math.max(1, cn));

    const pick = u => {
      let lo = 0, hi = tris.length - 1, x = u * total;
      while (lo < hi) { const m = (lo + hi) >> 1; if (tris[m].cum < x) lo = m + 1; else hi = m; }
      return tris[lo];
    };
    const p = new T.Vector3(), n = new T.Vector3();
    let rSum = 0;
    for (let i = 0; i < N; i++) {
      const t = pick(R());
      let u = R(), v = R();
      if (u + v > 1) { u = 1 - u; v = 1 - v; }
      p.copy(t.a).addScaledVector(t.ab, u).addScaledVector(t.ac, v);
      n.copy(t.na).multiplyScalar(1 - u - v).addScaledVector(t.nb, u).addScaledVector(t.nc, v).normalize();
      aPos[i * 3] = p.x; aPos[i * 3 + 1] = p.y; aPos[i * 3 + 2] = p.z;
      aNrm[i * 3] = n.x; aNrm[i * 3 + 1] = n.y; aNrm[i * 3 + 2] = n.z;
      aTan[i * 3] = t.t.x; aTan[i * 3 + 1] = t.t.y; aTan[i * 3 + 2] = t.t.z;
      const j = 0.93 + R() * 0.14;
      aCol[i * 3] = t.col[0] * j; aCol[i * 3 + 1] = t.col[1] * j; aCol[i * 3 + 2] = t.col[2] * j;
      aSeed[i * 3] = R(); aSeed[i * 3 + 1] = R(); aSeed[i * 3 + 2] = R();
      aBirth[i] = 0; aCurv[i] = t.cv; aScale[i] = 0.86 + R() * 0.3;
      rSum += p.distanceTo(cen);
    }
    // oversmoothed target: radial low-pass about the centroid
    const meanR = rSum / N;
    const d = new T.Vector3();
    for (let i = 0; i < N; i++) {
      d.set(aPos[i * 3] - cen.x, aPos[i * 3 + 1] - cen.y, aPos[i * 3 + 2] - cen.z);
      const r = d.length() || 1e-4;
      const rs = r * 0.28 + meanR * 0.72;
      aSm[i * 3] = cen.x + d.x / r * rs;
      aSm[i * 3 + 1] = cen.y + d.y / r * rs;
      aSm[i * 3 + 2] = cen.z + d.z / r * rs;
    }
    /* aVoid — a smooth spatial field, histogram-flattened to uniform [0,1). A model with
       hole:h drops points with aVoid < h, which removes CONTIGUOUS PATCHES covering ~h of
       the area (MVS fails on textureless regions, not on scattered points). Random per-point
       dropout would be indistinguishable from simply having fewer primitives — the footprint
       rule would grow to compensate and the holes would vanish from the metric. */
    const aVoid = new Float32Array(N);
    {
      let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
      for (let i = 0; i < N; i++) {
        const x = aPos[i * 3], y = aPos[i * 3 + 1], z = aPos[i * 3 + 2];
        if (x < mnx) mnx = x; if (x > mxx) mxx = x;
        if (y < mny) mny = y; if (y > mxy) mxy = y;
        if (z < mnz) mnz = z; if (z > mxz) mxz = z;
      }
      const ex = Math.max(mxx - mnx, Math.max(mxy - mny, mxz - mnz)) || 1;
      const ph = [];
      for (let o = 0; o < 3; o++) ph.push({ f: (2.4 + o * 2.7) / ex * 6.2832, a: R() * 6.2832, b: R() * 6.2832, c: R() * 6.2832, w: 1 / (o + 1) });
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < N; i++) {
        let v = 0;
        for (const p of ph) v += p.w * Math.sin(aPos[i * 3] * p.f + p.a) * Math.cos(aPos[i * 3 + 1] * p.f * 0.83 + p.b) * Math.sin(aPos[i * 3 + 2] * p.f * 1.17 + p.c);
        aVoid[i] = v;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
      const BINS = 512, hist = new Int32Array(BINS), span = (hi - lo) || 1;
      for (let i = 0; i < N; i++) hist[clamp(((aVoid[i] - lo) / span * BINS) | 0, 0, BINS - 1)]++;
      const cdf = new Float32Array(BINS);
      let acc = 0;
      for (let b = 0; b < BINS; b++) { acc += hist[b]; cdf[b] = acc / N; }
      for (let i = 0; i < N; i++) aVoid[i] = cdf[clamp(((aVoid[i] - lo) / span * BINS) | 0, 0, BINS - 1)];
    }

    // birth order — two orderings over the SAME pool, one per representation:
    //  · aBirth (ours): PROGRESSIVE STRATIFICATION. Coarse-to-fine octree passes, each
    //    emitting one representative per occupied cell, so every prefix is a jittered
    //    lattice at the scale that prefix can afford → blue-noise spacing, and
    //    h = sqrt(area/alive) is an accurate local spacing. A footprint just above the
    //    hex-closure radius then tiles the surface: no gaps, ≈1.3× overlap.
    //    (The previous digit-reversed Morton key sorted on the FINEST bit first, which is
    //    a lattice decimation, not a hierarchy: prefixes came out Poisson — locally
    //    clumped with e⁻² voids — so closing them needed ~4× the footprint area, which is
    //    what read as an inflated surface.)
    //  · aBirthU (baselines): iid order — clumped, the density adaptive control gives you.
    const aBirthU = new Float32Array(N);
    {
      const bmin = new T.Vector3(Infinity, Infinity, Infinity), bmax = new T.Vector3(-Infinity, -Infinity, -Infinity);
      for (let i = 0; i < N; i++) {
        bmin.x = Math.min(bmin.x, aPos[i * 3]); bmax.x = Math.max(bmax.x, aPos[i * 3]);
        bmin.y = Math.min(bmin.y, aPos[i * 3 + 1]); bmax.y = Math.max(bmax.y, aPos[i * 3 + 1]);
        bmin.z = Math.min(bmin.z, aPos[i * 3 + 2]); bmax.z = Math.max(bmax.z, aPos[i * 3 + 2]);
      }
      const ext = Math.max(bmax.x - bmin.x, Math.max(bmax.y - bmin.y, bmax.z - bmin.z)) * 1.0001 || 1;
      const perm = new Uint32Array(N);
      for (let i = 0; i < N; i++) perm[i] = i;
      const shuffle = a => { for (let i = a.length - 1; i > 0; i--) { const k = (R() * (i + 1)) | 0; const t = a[i]; a[i] = a[k]; a[k] = t; } };
      shuffle(perm);

      const taken = new Uint8Array(N);
      const reps = new Uint32Array(N);
      let rank = 0;
      for (const S of [4, 8, 16, 32, 64, 128]) {
        const occ = new Int32Array(S * S * S).fill(-1);
        const bestD = new Float32Array(S * S * S);
        const cw = ext / S;
        let nr = 0;
        for (let q = 0; q < N; q++) {
          const i = perm[q];
          if (taken[i]) continue;
          const px = (aPos[i * 3] - bmin.x) / cw, py = (aPos[i * 3 + 1] - bmin.y) / cw, pz = (aPos[i * 3 + 2] - bmin.z) / cw;
          const cx = clamp(px | 0, 0, S - 1), cy = clamp(py | 0, 0, S - 1), cz = clamp(pz | 0, 0, S - 1);
          const c = cx + cy * S + cz * S * S;
          // representative = the candidate CLOSEST TO THE CELL CENTRE, not the first seen.
          // A random representative lets neighbouring cells pick points that touch, so the
          // prefix keeps near-pairs (measured ×3.9 disks per point). Centre-snapping makes
          // each pass a near-regular lattice → nearest-neighbour distance ≈ h.
          const dx = px - cx - 0.5, dy = py - cy - 0.5, dz = pz - cz - 0.5;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (occ[c] === -1) { occ[c] = i; bestD[c] = d2; reps[nr++] = c; }
          else if (d2 < bestD[c]) { occ[c] = i; bestD[c] = d2; }
        }
        const pass = reps.subarray(0, nr);
        /* Stratify the PARTIAL level. A prefix of the alive set always cuts mid-level (w varies
           continuously), and taking that fraction at random made the finest active layer Poisson
           at exactly the near-neighbour scale — measured as the coverage/overlap trade-off curve
           (×1.7 → 87%, ×2.5 → 94.6%, both only ~4 pts better than Poisson's 1-e^-λ).
           So: group the level's cells into 2×2×2 blocks and emit ROUND-ROBIN over blocks, one
           random cell per block per round. Any prefix then holds ~one cell per block — spatially
           spread — while the choice inside a block stays random, which is what a strict
           radical-inverse order got wrong (its parity sub-lattices aligned across regions cutting
           at different depths and met along systematic seams). */
        const S2 = S >> 1 || 1;
        const groups = new Map();
        for (let r = 0; r < nr; r++) {
          const c = pass[r];
          const cx = c % S, cy = ((c / S) | 0) % S, cz = (c / (S * S)) | 0;
          const bk = (cx >> 1) + (cy >> 1) * S2 + (cz >> 1) * S2 * S2;
          let g = groups.get(bk);
          if (!g) { g = []; groups.set(bk, g); }
          g.push(c);
        }
        const blocks = Array.from(groups.values());
        for (const g of blocks) shuffle(g);          // random choice within a block
        shuffle(blocks);                             // no scan-order bias across blocks
        let slot = 0, emitted = 0;
        while (emitted < nr) {
          for (const g of blocks) {
            if (slot >= g.length) continue;
            const i = occ[g[slot]];
            taken[i] = 1; aBirth[i] = rank++; emitted++;
          }
          slot++;
        }
      }
      for (let q = 0; q < N; q++) { const i = perm[q]; if (!taken[i]) aBirth[i] = rank++; }
      for (let i = 0; i < N; i++) aBirth[i] /= N;

      const ord = new Uint32Array(N);
      for (let i = 0; i < N; i++) ord[i] = i;
      shuffle(ord);
      for (let r = 0; r < N; r++) aBirthU[ord[r]] = r / N;
    }

    return { aPos, aNrm, aTan, aCol, aSm, aSeed, aBirth, aBirthU, aScale, aCurv, aVoid, area: total, N };
  }

  /* ══ shaders ════════════════════════════════════════════════════ */
  const VERT = `
    attribute vec3 aPos, aNrm, aTan, aCol, aSm, aSeed;
    attribute float aBirth, aBirthU, aScale, aCurv, aVoid, aFade;
    uniform float uAlign, uFloat, uJitter, uSmooth, uHole, uThresh, uThreshOut, uSize, uAniso;
    uniform float uSizeVar, uConv, uSH, uViewDriven, uCloudR, uTanHalf, uRoiR, uPriExp;
    uniform float uNeighbor, uCoverK, uCell, uGridN, uTexW, uTexH, uUvMode;
    uniform float uKnotW, uKnotR, uModMix, uModGap, uModMin, uModSpan, uModD0, uModD1;
    uniform vec3 uCamPos, uViewDir, uCenter, uLight, uGridMin, uKnotC, uModAxis;
    uniform sampler2D uSpacing;
    varying vec2 vUv; varying vec3 vCol;
    void main() {
      float birth = mix(aBirthU, aBirth, uNeighbor);
      vec3 dv = aPos - uCamPos;
      float depth = dot(dv, uViewDir);
      float perp = length(dv - uViewDir * depth);
      float edge = uTanHalf * max(depth, 0.2) * 1.06;
      float coneS = 1.0 - smoothstep(edge * 0.85, edge * 1.15, perp);
      float depthS = smoothstep(0.05, 0.35, depth);
      float faceS = smoothstep(-0.26, -0.04, dot(-uViewDir, aNrm));
      float mem = coneS * depthS * faceS;
      // priority grades the threshold; membership (mem) is what the CPU gates its counts on
      float fdot = dot(-uViewDir, aNrm);
      float rel = max(depth, 0.2) / max(uRoiR, 0.2);
      float r2 = rel * rel;
      float den = uPriExp > 1.75 ? r2 * r2 : (uPriExp > 1.25 ? r2 * rel : r2);
      float inView = mem * clamp(max(fdot, 0.05) / den, 0.04, 1.0);
      /* ROI crop, not a cost multiplier. The birth rank is compared against a threshold that
         is itself view-dependent: uThresh inside the ROI, uThreshOut outside, eased across
         the frustum edge by inView. uThreshOut is solved for a FIXED absolute context count,
         so tightening the ROI cannot leak budget outward — the old score*mix(40,1,inView)
         against one threshold did exactly that: solving for the budget raised the threshold,
         and the periphery's allowance (thresh/40) rose with it, so zooming in ADDED
         peripheral splats and the set only ever grew in place. */
      float tau = mix(uThresh, mix(uThreshOut, uThresh, inView), uViewDriven);
      /* knot insertion. Inside the refined patch a primitive's birth RANK is scaled down
         (better rank → survives), outside it is scaled up. The CPU solver applies the same
         factor before solving tau, so the total count is conserved and capacity MIGRATES
         into the patch rather than being added to the frame. */
      float kf = 1.0;
      if (uKnotW > 0.001) {
        float kd = length(aPos - uKnotC) / max(uKnotR, 1e-3);
        float kg = exp(-kd * kd * 1.35);
        kf = mix(1.0, mix(1.16, 0.44, kg), uKnotW);
      }
      bool dead = birth * kf > tau || aVoid < uHole;
      if (dead) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vUv = vec2(0.0); vCol = vec3(0.0); return; }
      vec3 rnd = normalize(aSeed * 2.0 - 1.0 + vec3(0.0021, 0.0013, 0.0037));
      vec3 p = mix(aPos, aSm, uSmooth);
      p += aNrm * (aSeed.x - 0.5) * uJitter;
      p += rnd * uFloat * step(0.87, aSeed.z);
      vec3 cloud = uCenter + normalize(rnd + aNrm * 0.25) * uCloudR * (0.5 + 0.8 * aSeed.y);
      p = mix(cloud, p, uConv);
      vec3 n = normalize(mix(rnd, aNrm, clamp(uAlign * uConv, 0.0, 1.0)));
      vec3 t = aTan - n * dot(aTan, n);
      if (length(t) < 1e-3) t = cross(n, vec3(0.0, 1.0, 0.0));
      t = normalize(t);
      vec3 b = cross(n, t);
      float base = uSize * aScale;
      float sx, sy;
      if (uNeighbor > 0.5) {
        // footprint from the LOCAL SPACING of surviving neighbours: exactly enough
        // to close the surface, not enough to alias. Anisotropy is area-preserving.
        vec3 gp = (aPos - uGridMin) / uCell;
        float gn = uGridN;
        vec3 ci = clamp(floor(gp), vec3(0.0), vec3(gn - 1.0));
        float idx = ci.x + ci.y * gn + ci.z * gn * gn;
        float tx = mod(idx, uTexW), ty = floor(idx / uTexW);
        float h = texture2D(uSpacing, vec2((tx + 0.5) / uTexW, (ty + 0.5) / uTexH)).r;
        h = max(h, 1e-5);
        // area-preserving anisotropy: curvature elongates the disk along the chart's
        // major direction without growing its footprint. Stretching one axis only (as
        // before) added overlap everywhere aCurv > 0 on top of an already large K.
        float e = sqrt(1.0 + uAniso * aCurv * 0.5);
        /* uUvMode: the (u,v) samplers carry their own footprint per splat (aScale, from the
           local step) instead of the neighbour-spacing estimate, because their density is set
           by the domain step and not by how many survivors happen to land in a 24^3 cell.
           aFade is the relocation shrink-fade and multiplies whichever footprint applies. */
        base = mix(h, aScale, uUvMode) * uCoverK;
        sx = base * e * aFade; sy = base / e * aFade;
      } else {
        sx = base * (1.0 + uAniso * aCurv) * (1.0 + uSizeVar * (aSeed.y - 0.5) * 2.0);
        sy = base * (1.0 - 0.42 * uAniso * aCurv) * (1.0 + uSizeVar * (aSeed.z - 0.5) * 2.0);
      }
      vec3 world = p + t * position.x * sx + b * position.y * sy;
      vec3 vdir = normalize(cameraPosition - p);
      vec3 c = mix(vec3(0.30, 0.33, 0.38), aCol, uConv);
      c += uSH * sin(dot(vdir, aNrm) * 9.0 + aSeed.x * 6.283) * 1.2;
      float lam = 0.52 + 0.48 * max(dot(n, uLight), 0.0);
      lam += 0.16 * pow(max(1.0 - abs(dot(n, vdir)), 0.0), 2.0);
      vCol = clamp(c * lam, 0.0, 1.4);
      /* modality slices: the scene is cut along uModAxis into three slabs and each slab is
         coloured by a different channel the representation already carries — camera depth,
         world-space normal, and per-axis curvature amplitude (|major direction| × κ). Nothing
         is baked: these are the same attributes the solver reads. */
      if (uModMix > 0.001) {
        float sc = clamp((dot(p, uModAxis) - uModMin) / max(uModSpan, 1e-4), 0.0, 0.99999);
        float sf = sc * 3.0;
        float si = floor(sf);
        float fr = fract(sf);
        vec3 mc;
        if (si < 0.5) {
          float dn = clamp((depth - uModD0) / max(uModD1 - uModD0, 1e-3), 0.0, 1.0);
          mc = mix(mix(vec3(0.10, 0.34, 0.78), vec3(0.32, 0.90, 0.80), smoothstep(0.0, 0.55, dn)),
                   vec3(0.99, 0.86, 0.36), smoothstep(0.45, 1.0, dn));
        } else if (si < 1.5) {
          mc = 0.5 + 0.5 * normalize(aNrm);
        } else {
          // per-axis curvature amplitude: hue is the chart's major direction, value is κ.
          // the floor keeps flat surface legible instead of reading as missing geometry.
          vec3 ax = abs(normalize(aTan + vec3(1e-4)));
          mc = clamp(ax * (0.30 + aCurv * 3.4), 0.0, 1.0);
        }
        float seam = step(0.0005, uModGap) * smoothstep(uModGap, 0.0, min(fr, 1.0 - fr));
        mc = mix(mc * (0.60 + 0.40 * lam), vec3(0.86, 0.95, 1.0), seam * 0.85);
        vCol = mix(vCol, clamp(mc, 0.0, 1.4), uModMix);
      }
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
    }`;
  const FRAG = `
    precision mediump float;
    varying vec2 vUv; varying vec3 vCol;
    void main() {
      vec2 d = vUv * 2.0 - 1.0;
      float a = exp(-3.6 * dot(d, d));
      if (a < 0.34) discard;
      gl_FragColor = vec4(vCol, 1.0);
    }`;

  /* ══ factory ════════════════════════════════════════════════════ */
  function create(container, opts) {
    opts = opts || {};
    const T = window.THREE;
    if (!T) { container.textContent = 'three.js missing'; return { dispose() { } }; }
    /* The pool is the ceiling on BOTH panels, so it has to hold the densified baseline, not
       just our budget: 3DGS grows to 5.2×. At 300k the honest budget ceiling is 300k/5.2 ≈
       57k, which is what the slider stops at — past that the baseline would silently draw
       fewer primitives than it claims and its coverage would read better than it earns. */
    const POOL = opts.pool || 300000;
    const GN = 24, GCELLS = GN * GN * GN, TEXW = 144, TEXH = 96;
    /* COVER_K converts inter-splat spacing h into the quad's full width.
       The quad is 1×1 and the frag kernel exp(-3.6d²) is clipped at a<0.34, so the
       VISIBLE disk radius is 0.547·(width/2) = 0.274·width. Hexagonal closure of a
       blue-noise set with spacing h needs radius ≈ h/√3 ≈ 0.58h, hence
       width ≈ 0.58h / 0.274 ≈ 2.1h. That hex ideal is a FLOOR, not the target: the set
       is stratified blue-noise, and h itself is estimated per 24³ cell as
       sqrt(eligible area / alive) per 24³ cell, which underestimates spacing where the
       surface cuts a cell diagonally. 2.9 puts the visible radius at ≈0.79h — measured
       95-96% closure at ×2.7 overlap. Tune against the cov/ovl readout, not by eye. */
    const COVER_K = 2.9;

    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;cursor:grab;touch-action:none';
    container.appendChild(canvas);

    const renderer = new T.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, opts.dpr || 1.85));
    renderer.setClearColor(0x05080e, 1);
    const scene = new T.Scene();
    scene.fog = new T.Fog(0x05080e, 42, 132);
    scene.add(new T.HemisphereLight(0x9fc4ff, 0x1a1408, 1.05));
    const key = new T.DirectionalLight(0xffffff, 0.85); key.position.set(9, 14, 7); scene.add(key);
    const camera = new T.PerspectiveCamera(42, 1, 0.1, 600);

    const st = {
      dataset: opts.dataset || 'barn',
      baseline: opts.baseline || '3dgs',
      budget: opts.budget || 12000,
      roiTight: opts.roiTight || 1,
      closure: opts.closure === undefined ? 0.5 : clamp(+opts.closure, 0, 1),
      adapt: opts.adapt !== false,
      step: 0, playing: opts.playing !== false, speed: 1,
      knots: 0, mod: 0, modGap: 0.03,
      render: opts.render || 'splats', triBudget: opts.triBudget || 12000, wire: opts.wire !== false,
      zoom: 0, split: true, showGT: false, chart: false, solo: 'ours',
      /* 'transport' = the Hilbert/local-transport sampler; 'uvUniform' / 'uvAdaptive' =
         the (u,v) atlas samplers. uvOcclude adds occlusion culling to the visibility test
         (off by default — see uvVisible). */
      sampler: opts.sampler || 'transport', uvOcclude: !!opts.uvOcclude,
      cover: opts.cover || 1,
      maxStep: 30000,
      distScale: 1, gtOp: 0.22, chartOp: 0.42
    };

    let gt = null, sampler = null, meshOurs = null, meshBase = null, chartGrp = null, focus = new T.Vector3();
    const orbit = { th: 0.86, ph: 0.94, dist: 34, target: new T.Vector3() };

    /* src is any object exposing the a* attribute arrays: the static pool for baselines,
       the adaptively resampled buffers for ours. */
    const ONES_CACHE = new Map();
    const ONES = n => { let a = ONES_CACHE.get(n); if (!a) { a = new Float32Array(n).fill(1); ONES_CACHE.set(n, a); } return a; };
    function mkSplats(src, count) {
      const base = new T.PlaneGeometry(1, 1);
      const g = new T.InstancedBufferGeometry();
      g.setIndex(base.index);
      g.setAttribute('position', base.attributes.position);
      g.setAttribute('uv', base.attributes.uv);
      const A = (name, arr, n) => g.setAttribute(name, new T.InstancedBufferAttribute(arr, n));
      A('aPos', src.aPos, 3); A('aNrm', src.aNrm, 3); A('aTan', src.aTan, 3);
      A('aCol', src.aCol, 3); A('aSm', src.aSm, 3); A('aSeed', src.aSeed, 3);
      A('aBirth', src.aBirth, 1); A('aBirthU', src.aBirthU, 1);
      A('aScale', src.aScale, 1); A('aCurv', src.aCurv, 1); A('aVoid', src.aVoid, 1);
      A('aFade', src.aFade || ONES(src.N), 1);
      g.instanceCount = count;
      g.boundingSphere = new T.Sphere(new T.Vector3(0, 4, 0), 120);
      const m = new T.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, side: T.DoubleSide,
        uniforms: {
          uAlign: { value: 1 }, uFloat: { value: 0 }, uJitter: { value: 0 }, uSmooth: { value: 0 },
          uHole: { value: 0 }, uThresh: { value: 0.3 }, uThreshOut: { value: 0.01 }, uSize: { value: 0.2 }, uAniso: { value: 0 },
          uSizeVar: { value: 0.1 }, uConv: { value: 1 }, uSH: { value: 0 }, uViewDriven: { value: 0 },
          uCloudR: { value: 9 }, uTanHalf: { value: 0.38 }, uRoiR: { value: 9 }, uPriExp: { value: 1 },
          uNeighbor: { value: 0 }, uCoverK: { value: COVER_K }, uUvMode: { value: 0 },
          uCell: { value: 1 }, uGridN: { value: GN }, uTexW: { value: TEXW }, uTexH: { value: TEXH },
          uGridMin: { value: new T.Vector3() }, uSpacing: { value: G.tex },
          uCamPos: { value: new T.Vector3(0, 0, 40) }, uViewDir: { value: new T.Vector3(0, 0, -1) },
          uCenter: { value: new T.Vector3() },
          uKnotC: { value: new T.Vector3() }, uKnotR: { value: 1 }, uKnotW: { value: 0 },
          uModMix: { value: 0 }, uModGap: { value: 0.03 }, uModMin: { value: 0 }, uModSpan: { value: 1 },
          uModD0: { value: 1 }, uModD1: { value: 40 }, uModAxis: { value: new T.Vector3(1, 0, 0) },
          uLight: { value: new T.Vector3(0.42, 0.84, 0.35).normalize() }
        }
      });
      const mesh = new T.Mesh(g, m);
      mesh.frustumCulled = false;
      return mesh;
    }

    /* ══ neighbour-spacing field ═══════════════════════════════════
       A 24³ grid over the capture. Per cell we know the surface area it holds
       (from the full pool) and how many primitives SURVIVED there this frame, so
       h = sqrt(area / alive) is the local inter-splat spacing. Footprint = K·h:
       adjacent disks just touch — no gaps, minimal overlap. Recomputed every
       refresh, so it tracks budget, densification and view-driven allocation. */
    const G = {
      min: new T.Vector3(), cell: 1,
      poolCnt: new Int32Array(GCELLS),
      spc: new Float32Array(GCELLS), spc2: new Float32Array(GCELLS), spcDisp: new Float32Array(GCELLS), cnt: new Int32Array(GCELLS),
      elig: new Int32Array(GCELLS), areaC: new Float32Array(GCELLS),
      cellOf: new Int32Array(POOL), head: new Int32Array(GCELLS), next: new Int32Array(POOL),
      /* pHead/pNext — the FULL POOL as a linked list (built once per capture in gridInit),
         so the model→GT chamfer direction can nearest-neighbour against ground truth the
         same way head/next (the transient alive set) serves the GT→model direction. */
      pHead: new Int32Array(GCELLS), pNext: new Int32Array(POOL),
      data: new Float32Array(TEXW * TEXH), tex: null
    };
    G.tex = new T.DataTexture(G.data, TEXW, TEXH, T.RedFormat, T.FloatType);
    G.tex.minFilter = G.tex.magFilter = T.NearestFilter;
    G.tex.needsUpdate = true;

    function gridInit() {
      const P = sampler.aPos, N = sampler.N;
      const mn = new T.Vector3(Infinity, Infinity, Infinity), mx = new T.Vector3(-Infinity, -Infinity, -Infinity);
      for (let i = 0; i < N; i++) {
        mn.x = Math.min(mn.x, P[i * 3]); mx.x = Math.max(mx.x, P[i * 3]);
        mn.y = Math.min(mn.y, P[i * 3 + 1]); mx.y = Math.max(mx.y, P[i * 3 + 1]);
        mn.z = Math.min(mn.z, P[i * 3 + 2]); mx.z = Math.max(mx.z, P[i * 3 + 2]);
      }
      const ext = Math.max(mx.x - mn.x, Math.max(mx.y - mn.y, mx.z - mn.z)) * 1.02 || 1;
      G.min.copy(mn).addScalar(-ext * 0.01);
      G.cell = ext / GN;
      G.poolCnt.fill(0);
      for (let i = 0; i < N; i++) {
        const cx = clamp(((P[i * 3] - G.min.x) / G.cell) | 0, 0, GN - 1);
        const cy = clamp(((P[i * 3 + 1] - G.min.y) / G.cell) | 0, 0, GN - 1);
        const cz = clamp(((P[i * 3 + 2] - G.min.z) / G.cell) | 0, 0, GN - 1);
        const c = cx + cy * GN + cz * GN * GN;
        G.cellOf[i] = c; G.poolCnt[c]++;
      }
      G.pHead.fill(-1);
      for (let i = 0; i < N; i++) { const c2 = G.cellOf[i]; G.pNext[i] = G.pHead[c2]; G.pHead[c2] = i; }
    }
    function gcellOf(x, y, z) {
      const cx = clamp(((x - G.min.x) / G.cell) | 0, 0, GN - 1);
      const cy = clamp(((y - G.min.y) / G.cell) | 0, 0, GN - 1);
      const cz = clamp(((z - G.min.z) / G.cell) | 0, 0, GN - 1);
      return cx + cy * GN + cz * GN * GN;
    }

    /* alive set → per-cell counts + linked list, scanned over the FULL pool (never a
       buffer-index prefix — aBirth ranks don't correlate with buffer order, so
       truncating early undercounts and pops as the camera moves).
       G.elig counts pool points in the cell that are ELIGIBLE under the same view
       weighting — the surface the survivors can actually land on. Deriving spacing from
       the full pool's area instead inflated h by ~1.6× wherever a cell straddles the
       cone or the silhouette (survivors cover the front part, area counted the whole
       cell), and the oversized disks read as an inflated surface. */
    /* ONE definition of alive, used by solveThresh, bucket and the vertex shader alike:
       birth ≤ mix(tOut, tIn, inView) AND aVoid ≥ hole. The shader used to carry the hole cull
       alone, so for MVS (hole 0.20) a fifth of the "alive" set was counted by the CPU — alive
       count, spacing field, coverage and the params ledger — but never drawn. */
    function voided(i, hole) { return hole > 0 && sampler.aVoid[i] < hole; }

    /* ── knot insertion: ONE rank-scale formula, shared with the vertex shader ──
       The refinement patch is placed on the highest-curvature region of the capture (that is
       where the residual lives). kfOf scales a primitive's birth rank; the budget solver runs
       against the scaled ranks, so inserting knots redistributes the SAME budget instead of
       growing it. Applied only on the view-driven (ours) path — baselines have no chart. */
    const KNOT = { c: new T.Vector3(), r: 1, w: 0 };
    const MODX = { min: 0, span: 1 };
    function kfOf(i) {
      if (KNOT.w <= 0.001) return 1;
      const dx = sampler.aPos[i * 3] - KNOT.c.x, dy = sampler.aPos[i * 3 + 1] - KNOT.c.y, dz = sampler.aPos[i * 3 + 2] - KNOT.c.z;
      const kd = Math.sqrt(dx * dx + dy * dy + dz * dz) / Math.max(KNOT.r, 1e-3);
      const kg = Math.exp(-kd * kd * 1.35);
      return 1 + KNOT.w * ((1.16 + (0.44 - 1.16) * kg) - 1);
    }
    function pickKnot() {
      const NB = 8, P = sampler.aPos, CV = sampler.aCurv, N = sampler.N;
      const acc = new Float32Array(NB * NB * NB), cnt = new Int32Array(NB * NB * NB);
      const mn = gt.center.clone().addScalar(-gt.radius), span = gt.radius * 2;
      for (let i = 0; i < N; i++) {
        const ix = clamp(((P[i * 3] - mn.x) / span * NB) | 0, 0, NB - 1);
        const iy = clamp(((P[i * 3 + 1] - mn.y) / span * NB) | 0, 0, NB - 1);
        const iz = clamp(((P[i * 3 + 2] - mn.z) / span * NB) | 0, 0, NB - 1);
        const c = ix + iy * NB + iz * NB * NB;
        acc[c] += CV[i]; cnt[c]++;
      }
      const need = Math.max(24, (N / 1500) | 0);
      let best = -1, bv = -1;
      for (let c = 0; c < acc.length; c++) if (cnt[c] >= need) { const v = acc[c] / cnt[c]; if (v > bv) { bv = v; best = c; } }
      KNOT.r = gt.radius * 0.30;
      if (best < 0) { KNOT.c.copy(gt.center); return; }
      const ix = best % NB, iy = ((best / NB) | 0) % NB, iz = (best / (NB * NB)) | 0;
      KNOT.c.set(mn.x + (ix + 0.5) / NB * span, mn.y + (iy + 0.5) / NB * span, mn.z + (iz + 0.5) / NB * span);
    }
    function measureModAxis() {
      const P = sampler.aPos, N = sampler.N;
      let a = Infinity, b = -Infinity;
      for (let i = 0; i < N; i += 7) { const x = P[i * 3]; if (x < a) a = x; if (x > b) b = x; }
      MODX.min = a; MODX.span = Math.max(b - a, 1e-3);
    }

    function bucket(birthArr, tIn, tOut, viewDriven, hole) {
      G.cnt.fill(0); G.head.fill(-1); G.elig.fill(0);
      const N = sampler.N;
      let n = 0, nIn = 0;
      for (let i = 0; i < N; i++) {
        if (voided(i, hole)) continue;
        const c = G.cellOf[i];
        let mem = 1, w = 1;
        if (viewDriven) { const v = viewOf(i); mem = v.mem; w = v.w; }
        const inFrame = mem > 0.5;              // membership, NOT priority
        if (inFrame) G.elig[c]++;
        if (birthArr[i] * (viewDriven ? kfOf(i) : 1) > tOut + (tIn - tOut) * w) continue;
        G.next[i] = G.head[c]; G.head[c] = i; G.cnt[c]++; n++;
        if (inFrame) nIn++;
      }
      return { n: n, nIn: nIn };
    }

    function spacingField(hRef, areaC) {
      const perPt = sampler.area / sampler.N;
      for (let c = 0; c < GCELLS; c++) {
        const n = G.cnt[c];
        if (n <= 0) { G.spc[c] = 0; continue; }
        /* adaptive path: the cell's VISIBLE area, measured on the domain triangles, over the
           primitives that landed in it. Cells holding only reserve (unobserved) primitives
           have no visible area, so they fall back to the pool's area share. */
        const A = areaC ? (areaC[c] > 1e-9 ? areaC[c] : Math.max(G.poolCnt[c], n) * perPt)
          : Math.max(G.elig[c], n) * perPt;
        G.spc[c] = Math.sqrt(A / n);
      }
      /* h is clamped to a band around hRef, the mean in-focus spacing sqrt(focusArea/N).
         Without a CEILING, a cell holding 1-2 survivors (cone edge, silhouette, thin
         structure) reports an enormous h and emits giant blobs — that, not K, was most of
         the "inflated balloon": mean overlap ×4 while a fifth of the surface stayed open.
         Without a FLOOR, dense cells shrink below closure and the surface speckles.
         The 1-ring cap still guards the seam where a dense cell abuts a sparse one; the
         old 5³ MAX-dilation made every splat adopt the worst spacing within ±2 cells. */
      /* Only a CEILING. A cell holding 1-2 survivors reports an enormous h and emits giant
         blobs, so cap it. There is deliberately NO floor: G.spc is the TRUE local spacing
         (eligible area / survivors), so a genuinely dense cell must be allowed to go small —
         a floor at a global reference clamped exactly the centre cells the ROI concentrates
         into, holding centre spacing at 2.3× when the frame magnifies 3.37×. Closure is
         guaranteed by COVER_K·h against the true h, not by a floor. */
      let maxH = 0;
      const CAP = 1.55, LO = 0, HI = hRef * 2.9;
      for (let cz = 0; cz < GN; cz++) for (let cy = 0; cy < GN; cy++) for (let cx = 0; cx < GN; cx++) {
        const c = cx + cy * GN + cz * GN * GN;
        const own = G.spc[c];
        if (G.cnt[c] === 0) { G.spc2[c] = 0; continue; }
        let m = own, acc = 0, an = 0;
        for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy && !dz) continue;
          const nx = cx + dx, ny = cy + dy, nz = cz + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= GN || ny >= GN || nz >= GN) continue;
          const v = G.spc[nx + ny * GN + nz * GN * GN];
          if (v > 0) { acc += v; an++; if (v > m) m = v; }
        }
        /* Seam blending sits on a measured trade-off curve, and both ends are legitimate:
             neighbour MEAN  → ×1.7 overlap, 87% closed  (sharp, some speckle)
             neighbour MAX   → ×2.5 overlap, 94.6% closed (closed, visibly inflated)
           The mean is the unbiased estimator (max over 26 noisy neighbours runs ~13% high even in
           a uniform field); the max buys closure by over-covering everything. Three attempts to
           beat the curve structurally — radical-inverse within level, 2²² block round-robin,
           centre-snapped reps — all landed back on it, because the floor is the pool's own local
           irregularity, not the ordering. So st.closure picks the operating point. */
        const nb = an ? (acc / an) + st.closure * (Math.min(m, own * CAP) - acc / an) : own;
        const h = clamp(0.5 * (own + nb), LO, HI);
        G.spc2[c] = h;
        if (h > maxH) maxH = h;
      }
      // G.spc2 is the TARGET spacing field; the texture is only ever written from
      // G.spcDisp (settleAndPush), which eases toward this every frame.
      return maxH;
    }

    /* ONE source for the per-model displacement the shader applies, used by both
       setUniforms (to push it) and measureCoverage (to test against it). The metric read
       the UNDISPLACED aPos, so a baseline's floaters were credited as covering the surface
       — 3DGS scored 93.9% against ours' 91.5% while rendering less than half the fill. */
    function dispOf(M, foot, conv, isOurs) {
      const S = M.shader;
      return {
        jitter: S.jitter * foot * (isOurs ? 1 : 1.62 / COVER_K) * (0.25 + 1.9 * (1 - conv)),
        float: S.float * gt.radius * 0.055 * (0.3 + 0.7 * (1 - conv * 0.4)),
        smooth: S.smooth * conv,
        align: clamp(S.align * conv, 0, 1),
        /* needed by the CPU position replication (mkDrawnFn) to reproduce the shader's
           initial random-cloud mix p = mix(cloud, p, uConv) — without it, early-step
           chamfer would read a surface the model has not yet grown from the cloud. */
        conv: conv, cloudR: gt.radius * 0.55
      };
    }

    /* CPU replication of the vertex shader's drawn position and disk normal for primitive i,
       shared by measureCoverage and measureError: oversmoothed position, normal jitter,
       floater displacement, the mix(rnd, aNrm, align·conv) orientation — and the initial
       random-cloud mix p = mix(cloud, p, uConv), so early-step chamfer reads the cloud the
       model actually starts from, not a surface it has not yet grown. */
    function mkDrawnFn(D) {
      const sm = D.smooth, ns = 1 - sm, a = D.align;
      const cv = D.conv === undefined ? 1 : D.conv, cR = D.cloudR || 0;
      const cx0 = gt.center.x, cy0 = gt.center.y, cz0 = gt.center.z;
      const dp = { x: 0, y: 0, z: 0, nx: 0, ny: 0, nz: 0 };
      const mk = (P, NR, SM, SD) => i => {
        const s0 = SD[i * 3], s2 = SD[i * 3 + 2];
        let rx = s0 * 2 - 1 + 0.0021, ry = SD[i * 3 + 1] * 2 - 1 + 0.0013, rz = s2 * 2 - 1 + 0.0037;
        const rl = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
        rx /= rl; ry /= rl; rz /= rl;
        const nx = NR[i * 3], ny = NR[i * 3 + 1], nz = NR[i * 3 + 2];
        const jt = (s0 - 0.5) * D.jitter, fl = s2 >= 0.87 ? D.float : 0;
        let px = P[i * 3] * ns + SM[i * 3] * sm + nx * jt + rx * fl;
        let py = P[i * 3 + 1] * ns + SM[i * 3 + 1] * sm + ny * jt + ry * fl;
        let pz = P[i * 3 + 2] * ns + SM[i * 3 + 2] * sm + nz * jt + rz * fl;
        if (cv < 0.999) {                       // vec3 cloud = uCenter + normalize(rnd + aNrm*0.25) * uCloudR * (0.5 + 0.8*aSeed.y)
          let ux = rx + nx * 0.25, uy = ry + ny * 0.25, uz = rz + nz * 0.25;
          const ul = Math.sqrt(ux * ux + uy * uy + uz * uz) || 1;
          const cr = cR * (0.5 + 0.8 * SD[i * 3 + 1]);
          px = (cx0 + ux / ul * cr) * (1 - cv) + px * cv;
          py = (cy0 + uy / ul * cr) * (1 - cv) + py * cv;
          pz = (cz0 + uz / ul * cr) * (1 - cv) + pz * cv;
        }
        dp.x = px; dp.y = py; dp.z = pz;
        let dnx = rx * (1 - a) + nx * a, dny = ry * (1 - a) + ny * a, dnz = rz * (1 - a) + nz * a;
        const dl = Math.sqrt(dnx * dnx + dny * dny + dnz * dnz) || 1;
        dp.nx = dnx / dl; dp.ny = dny / dl; dp.nz = dnz / dl;
        return dp;
      };
      return mk;
    }

    /* measured coverage: does the model CLOSE THE SURFACE IT DRAWS — no holes, no aliasing.
       Test points and splats both get the model's own displacement, so a shared deformation
       (NeuS's oversmoothing) cancels and only real scatter and sparsity register. Fidelity to
       ground truth is a separate axis, reported as chamfer; conflating the two scored NeuS at
       11% for drawing a complete-but-smoothed surface. Both models use the SAME test points
       and the same spacing field the shader samples (G.spc2). */
    const TESTN = 900;
    const testIdx = (() => { const a = new Uint32Array(TESTN); const R = rng(4242); for (let i = 0; i < TESTN; i++) a[i] = (R() * POOL) | 0; return a; })();
    function measureCoverage(useSpacing, globalFoot, maxH, D, SRC) {
      const P = sampler.aPos, NR = sampler.aNrm, SM = sampler.aSm, SD = sampler.aSeed;
      /* the DRAWN set may live in its own buffers (ours, adaptively resampled); the TEST points
         always come from the pool, so both panels are scored against the same ground-truth
         surface samples. */
      const QP = SRC ? SRC.P : P, QN = SRC ? SRC.NR : NR, QS = SRC ? SRC.SM : SM, QD = SRC ? SRC.SD : SD;
      const halfMax = 0.548 * 0.5 * (useSpacing ? COVER_K * maxH : globalFoot);
      const reach = halfMax + D.float + Math.abs(D.jitter) * 0.5;
      const rad = clamp(Math.ceil(reach / G.cell), 1, 4);
      const mkDrawn = mkDrawnFn(D);
      const drawn = mkDrawn(P, NR, SM, SD);
      const drawnQ = SRC ? mkDrawn(QP, QN, QS, QD) : drawn;
      let tested = 0, cov = 0, ovl = 0, starved = 0;
      for (let ti = 0; ti < TESTN; ti++) {
        const t = testIdx[ti] % sampler.N;
        // hidden surface is deliberately unfunded under adaptive allocation, so it is excluded
        // from the score for BOTH panels rather than counted against either
        if (!inViewCPU(t) || adHidden(t)) continue;
        tested++;
        if (G.cnt[G.cellOf[t]] === 0) starved++;   // no survivor in the point's own cell
        const q = drawn(t);
        const tx = q.x, ty = q.y, tz = q.z;
        const nx0 = NR[t * 3], ny0 = NR[t * 3 + 1], nz0 = NR[t * 3 + 2];
        const c = G.cellOf[t];
        const cz = (c / (GN * GN)) | 0, cy = ((c / GN) | 0) % GN, cx = c % GN;
        let hits = 0;
        for (let dz = -rad; dz <= rad; dz++) {
          const z = cz + dz; if (z < 0 || z >= GN) continue;
          for (let dy = -rad; dy <= rad; dy++) {
            const y = cy + dy; if (y < 0 || y >= GN) continue;
            for (let dx = -rad; dx <= rad; dx++) {
              const x = cx + dx; if (x < 0 || x >= GN) continue;
              const cc = x + y * GN + z * GN * GN;
              const half = 0.548 * 0.5 * (useSpacing ? COVER_K * G.spc2[cc] : globalFoot);
              const h2 = half * half;
              for (let j = G.head[cc]; j !== -1; j = G.next[j]) {
                const d = drawnQ(j);
                if (d.nx * nx0 + d.ny * ny0 + d.nz * nz0 < 0.3) continue;
                const ex = d.x - tx, ey = d.y - ty, ez = d.z - tz;
                const along = ex * nx0 + ey * ny0 + ez * nz0;
                if (along * along > h2 * 0.25) continue;
                const tan2 = ex * ex + ey * ey + ez * ez - along * along;
                if (tan2 < h2) hits++;
              }
            }
          }
        }
        if (hits > 0) cov++;
        ovl += hits;
      }
      if (!tested) return { cov: 1, ovl: 1, starved: 0 };
      return { cov: cov / tested, ovl: ovl / tested, starved: starved / tested };
    }

    /* measured fidelity to ground truth — the numbers the chamfer / L1 readouts and the
       charts quote. GT→model: for each in-view pool test point, the distance to the nearest
       DRAWN primitive (displaced exactly as the shader draws it), plus the photometric gap
       between the GT albedo lit by the GT normal and the nearest primitive's own shader
       colour (grey-mix, SH shimmer and the misaligned-normal lighting term included, so a
       model whose disks do not lie in the tangent plane pays here, not only in coverage).
       model→GT: a stride subsample of the drawn set (the grid lists hold exactly the alive
       primitives of whichever panel was just bucketed) nearest-neighboured against the
       full-pool pHead/pNext lists. Symmetric chamfer = mean of the two means, in world
       units (the page multiplies by mpu×1000 for mm). Scans are windowed: a sample with
       nothing inside the window is censored at the window edge, so early-step values —
       the model is still a random cloud — are a lower bound that drops as soon as the
       surface forms. Test points hidden from the camera are excluded for BOTH panels:
       hidden surface is deliberately unfunded under view-driven allocation (the same
       rule the coverage metric applies). */
    const REVN = 300;
    const LIGHT_V = new T.Vector3(0.42, 0.84, 0.35).normalize();
    function measureError(M, D, SRC, aliveN) {
      const P = sampler.aPos, NR = sampler.aNrm, SD = sampler.aSeed;
      const QP = SRC ? SRC.P : P, QN = SRC ? SRC.NR : NR, QS = SRC ? SRC.SM : sampler.aSm, QD = SRC ? SRC.SD : SD;
      const QCL = SRC ? SRC.CL : sampler.aCol;
      const drawnQ = mkDrawnFn(D)(QP, QN, QS, QD);
      const conv = D.conv === undefined ? 1 : D.conv;
      const reach = D.float + Math.abs(D.jitter) + G.cell * 0.5;
      const rad = clamp(Math.ceil(reach / G.cell), 1, 4);
      const pen = G.cell * rad * 1.6;
      const sh = M.shader.sh;
      const AC = sampler.aCol;
      let tested = 0, fwd = 0, photo = 0, photoN = 0;
      for (let ti = 0; ti < TESTN; ti++) {
        const t = testIdx[ti] % sampler.N;
        if (!inViewCPU(t) || adHidden(t)) continue;
        tested++;
        const tx = P[t * 3], ty = P[t * 3 + 1], tz = P[t * 3 + 2];
        const c = G.cellOf[t];
        const cz = (c / (GN * GN)) | 0, cy = ((c / GN) | 0) % GN, cx = c % GN;
        let best = Infinity, bj = -1;
        for (let dz = -rad; dz <= rad; dz++) {
          const z = cz + dz; if (z < 0 || z >= GN) continue;
          for (let dy = -rad; dy <= rad; dy++) {
            const y = cy + dy; if (y < 0 || y >= GN) continue;
            for (let dx = -rad; dx <= rad; dx++) {
              const x = cx + dx; if (x < 0 || x >= GN) continue;
              for (let j = G.head[x + y * GN + z * GN * GN]; j !== -1; j = G.next[j]) {
                const d = drawnQ(j);
                const ex = d.x - tx, ey = d.y - ty, ez = d.z - tz;
                const dd = ex * ex + ey * ey + ez * ez;
                if (dd < best) { best = dd; bj = j; }
              }
            }
          }
        }
        if (bj < 0) { fwd += pen; continue; }
        fwd += Math.sqrt(best);
        let vx = view.pos.x - tx, vy = view.pos.y - ty, vz = view.pos.z - tz;
        const vl = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1; vx /= vl; vy /= vl; vz /= vl;
        const nx0 = NR[t * 3], ny0 = NR[t * 3 + 1], nz0 = NR[t * 3 + 2];
        const lamG = 0.52 + 0.48 * Math.max(nx0 * LIGHT_V.x + ny0 * LIGHT_V.y + nz0 * LIGHT_V.z, 0)
          + 0.16 * Math.pow(Math.max(1 - Math.abs(nx0 * vx + ny0 * vy + nz0 * vz), 0), 2);
        const d = drawnQ(bj), j3 = bj * 3, t3 = t * 3;
        const cr = 0.30 * (1 - conv) + QCL[j3] * conv, cg = 0.33 * (1 - conv) + QCL[j3 + 1] * conv, cb = 0.38 * (1 - conv) + QCL[j3 + 2] * conv;
        const shim = sh * Math.sin((QN[j3] * vx + QN[j3 + 1] * vy + QN[j3 + 2] * vz) * 9.0 + QD[j3] * 6.283) * 1.2;
        const lamM = 0.52 + 0.48 * Math.max(d.nx * LIGHT_V.x + d.ny * LIGHT_V.y + d.nz * LIGHT_V.z, 0)
          + 0.16 * Math.pow(Math.max(1 - Math.abs(d.nx * vx + d.ny * vy + d.nz * vz), 0), 2);
        photo += (Math.abs(clamp((cr + shim) * lamM, 0, 1.4) - clamp(AC[t3] * lamG, 0, 1.4))
          + Math.abs(clamp((cg + shim) * lamM, 0, 1.4) - clamp(AC[t3 + 1] * lamG, 0, 1.4))
          + Math.abs(clamp((cb + shim) * lamM, 0, 1.4) - clamp(AC[t3 + 2] * lamG, 0, 1.4))) / 3;
        photoN++;
      }
      const rad2 = 2, pen2 = G.cell * (rad2 + 0.75);
      let rev = 0, revN = 0, seen = 0;
      const stride = Math.max(1, Math.round((aliveN || 1) / REVN));
      outer:
      for (let c2 = 0; c2 < GCELLS; c2++) {
        for (let j = G.head[c2]; j !== -1; j = G.next[j]) {
          if ((seen++ % stride) !== 0) continue;
          const d = drawnQ(j);
          const jc = gcellOf(d.x, d.y, d.z);
          const jz = (jc / (GN * GN)) | 0, jy = ((jc / GN) | 0) % GN, jx = jc % GN;
          let best2 = Infinity;
          for (let dz = -rad2; dz <= rad2; dz++) {
            const z = jz + dz; if (z < 0 || z >= GN) continue;
            for (let dy = -rad2; dy <= rad2; dy++) {
              const y = jy + dy; if (y < 0 || y >= GN) continue;
              for (let dx = -rad2; dx <= rad2; dx++) {
                const x = jx + dx; if (x < 0 || x >= GN) continue;
                for (let k = G.pHead[x + y * GN + z * GN * GN]; k !== -1; k = G.pNext[k]) {
                  const ex = P[k * 3] - d.x, ey = P[k * 3 + 1] - d.y, ez = P[k * 3 + 2] - d.z;
                  const dd = ex * ex + ey * ey + ez * ez;
                  if (dd < best2) best2 = dd;
                }
              }
            }
          }
          rev += best2 === Infinity ? pen2 : Math.sqrt(best2);
          if (++revN >= REVN) break outer;
        }
      }
      if (!tested) return null;
      const f = fwd / tested, r = revN ? rev / revN : f;
      return { cham: 0.5 * (f + r), res: photoN ? photo / photoN : 0, fwd: f, rev: r };
    }

    function disposeObj(o) {
      if (!o) return;
      scene.remove(o);
      o.traverse ? o.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material && c.material.dispose) c.material.dispose(); })
        : 0;
      if (o.geometry) o.geometry.dispose();
      if (o.material && o.material.dispose) o.material.dispose();
    }

    function loadDataset(k) {
      st.dataset = k;
      const D = DATASETS[k];
      if (D.obj && D._geo) { install(realScene(D, D._geo)); return; }   // cached scan: one install, no proxy flash
      install(D.build(T));
      if (D.obj) loadReal(k);
    }

    function install(res) {
      disposeObj(gt); disposeObj(meshOurs); disposeObj(meshBase); disposeObj(chartGrp); chartGrp = null;
      disposeTri();
      gt = res;
      gt.group.traverse(o => { if (o.isMesh) o.material.transparent = true; });
      scene.add(gt.group);
      sampler = sampleSurface(T, gt.group, POOL, 9931);
      gridInit();
      pickKnot();
      measureModAxis();
      buildTriSoup();
      adBuildDomain();
      atlasBuild();
      meshBase = mkSplats(sampler, sampler.N);
      scene.add(meshBase);
      rebuildOurs();
      buildChart();
      orbit.target.copy(gt.center);
      focus.copy(gt.center);
      applyGT();
      updateCamera();
      refresh(true);
    }

    /* real scanned mesh: fetched, centred on the turntable, then resampled like any other capture */
    function loadReal(k) {
      const D = DATASETS[k];
      const use = geo => { if (st.dataset === k) install(realScene(D, geo)); };
      if (D._geo) return use(D._geo);
      if (D.obj.ply) {
        fetch(D.obj.url).then(r => r.arrayBuffer()).then(buf => {
          const g = parsePLY(T, buf);
          if (!g || !g.attributes.position || !g.attributes.position.count) return;
          D._geo = g; use(g);
        }).catch(() => { });
        return;
      }
      fetch(D.obj.url).then(r => r.text()).then(txt => {
        const g = parseOBJ(T, txt);
        if (!g.attributes.position || !g.attributes.position.count) return;
        D._geo = g; use(g);
      }).catch(() => { });
    }
    function realScene(D, geo) {
      const g = new T.Group();
      const table = new T.Mesh(new T.CylinderGeometry(7.4, 7.4, 0.5, 56), mat(T, 0x23262c, true));
      table.position.y = 0.25; g.add(table);
      cyl(T, g, 2.6, 3.0, 1.2, 0, 1.1, 0, 0x6f665a, 40);
      const m = new T.Mesh(geo, new T.MeshLambertMaterial({ color: D.obj.color, side: T.DoubleSide }));
      geo.computeBoundingBox();
      const bb = geo.boundingBox, size = new T.Vector3().subVectors(bb.max, bb.min);
      const s = 8.6 / Math.max(size.x, size.y, size.z);
      m.scale.setScalar(s);
      m.position.set(-(bb.min.x + size.x / 2) * s, 1.7 - bb.min.y * s, -(bb.min.z + size.z / 2) * s);
      g.add(m);
      return { group: g, radius: 7.6, center: new T.Vector3(0, 4.6, 0) };
    }

    /* ── the chart itself: a coarse B-spline control net fitted through the sampled
       surface (radial (u,v) lattice + 1-2-1 smoothing passes). Every OURS splat takes
       its scale and rotation from this net's local frame — this is that net, drawn. ── */
    function buildChart() {
      const kind = (DATASETS[st.dataset] && DATASETS[st.dataset].chart) || 'radial';
      const seg = [], cp = [], ins = [];
      if (kind === 'height') heightChart(seg, cp, ins); else radialChart(seg, cp, ins);
      chartGrp = new T.Group();
      const lg = new T.BufferGeometry();
      lg.setAttribute('position', new T.Float32BufferAttribute(seg, 3));
      chartGrp.add(new T.LineSegments(lg, new T.LineBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: st.chartOp })));
      const pg = new T.BufferGeometry();
      pg.setAttribute('position', new T.Float32BufferAttribute(cp, 3));
      chartGrp.add(new T.Points(pg, new T.PointsMaterial({ color: 0x9fe0ff, size: gt.radius * 0.016, sizeAttenuation: true, transparent: true, opacity: Math.min(1, st.chartOp * 1.9) })));
      /* child 2 is the INSERTED isolines — built once at full extent, revealed progressively
         by draw range so knot insertion animates without rebuilding the net. */
      const ig = new T.BufferGeometry();
      ig.setAttribute('position', new T.Float32BufferAttribute(ins.length ? ins : [0, 0, 0, 0, 0, 0], 3));
      const iln = new T.LineSegments(ig, new T.LineBasicMaterial({ color: 0x8affc4, transparent: true, opacity: 0 }));
      iln.frustumCulled = false;
      chartGrp.add(iln);
      chartInsN = ins.length ? ins.length / 6 : 0;
      applyKnotLines();
      chartGrp.visible = false;
      scene.add(chartGrp);
    }

    /* scene captures: a height-field chart y = f(x,z) over the occupied cells —
       the same net the swarm sim fits, so the lattice sits ON the roofs and pad. */
    function heightChart(seg, cp, ins) {
      const NX = 34, NZ = 34, P = sampler.aPos, N = sampler.N;
      let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      for (let i = 0; i < N; i++) {
        const x = P[i * 3], z = P[i * 3 + 2];
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (z < z0) z0 = z; if (z > z1) z1 = z;
      }
      const dx = (x1 - x0) / NX, dz = (z1 - z0) / NZ;
      const Hh = new Float32Array(NX * NZ).fill(-1e9), hit = new Uint8Array(NX * NZ);
      for (let i = 0; i < N; i++) {
        const ix = clamp(Math.floor((P[i * 3] - x0) / dx), 0, NX - 1);
        const iz = clamp(Math.floor((P[i * 3 + 2] - z0) / dz), 0, NZ - 1);
        const k = iz * NX + ix;
        if (P[i * 3 + 1] > Hh[k]) Hh[k] = P[i * 3 + 1];
        hit[k] = 1;
      }
      for (let pass = 0; pass < 2; pass++) {
        const S = Hh.slice();
        for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
          const k = z * NX + x;
          if (!hit[k]) continue;
          let s = S[k] * 2, w = 2;
          const nb = [[x - 1, z], [x + 1, z], [x, z - 1], [x, z + 1]];
          for (const [a, b] of nb) {
            if (a < 0 || b < 0 || a >= NX || b >= NZ) continue;
            const j = b * NX + a;
            if (hit[j]) { s += S[j]; w++; }
          }
          Hh[k] = s / w;
        }
      }
      let ymin = 1e9, ymax = -1e9;
      for (let k = 0; k < Hh.length; k++) if (hit[k]) { if (Hh[k] < ymin) ymin = Hh[k]; if (Hh[k] > ymax) ymax = Hh[k]; }
      const jump = (ymax - ymin) * 0.28 + 0.05;
      const at = (ix, iz) => [x0 + (ix + 0.5) * dx, Hh[iz * NX + ix] + 0.04, z0 + (iz + 0.5) * dz];
      for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
        const k = z * NX + x;
        if (!hit[k]) continue;
        const a = at(x, z);
        cp.push(a[0], a[1], a[2]);
        if (x + 1 < NX && hit[k + 1] && Math.abs(Hh[k + 1] - Hh[k]) < jump) {
          const b = at(x + 1, z); seg.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        }
        if (z + 1 < NZ && hit[k + NX] && Math.abs(Hh[k + NX] - Hh[k]) < jump) {
          const c = at(x, z + 1); seg.push(a[0], a[1], a[2], c[0], c[1], c[2]);
        }
      }
      if (!ins) return;
      let bx = 0, bz = 0, bd = Infinity;
      for (let z = 0; z < NZ; z++) for (let x = 0; x < NX; x++) {
        if (!hit[z * NX + x]) continue;
        const q = at(x, z), qx = q[0] - KNOT.c.x, qy = q[1] - KNOT.c.y, qz = q[2] - KNOT.c.z;
        const d = qx * qx + qy * qy + qz * qz;
        if (d < bd) { bd = d; bx = x; bz = z; }
      }
      const mid3 = (a, b) => [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
      for (let dz = -3; dz <= 3; dz++) for (let dxi = -4; dxi <= 4; dxi++) {
        const x = bx + dxi, z = bz + dz;
        if (x < 0 || z < 0 || x >= NX - 1 || z >= NZ - 1) continue;
        const k00 = z * NX + x, k10 = k00 + 1, k01 = k00 + NX, k11 = k01 + 1;
        if (!hit[k00] || !hit[k10] || !hit[k01] || !hit[k11]) continue;
        if (Math.abs(Hh[k10] - Hh[k00]) > jump || Math.abs(Hh[k01] - Hh[k00]) > jump) continue;
        const a0 = at(x, z), a1 = at(x + 1, z), b0 = at(x, z + 1), b1 = at(x + 1, z + 1);
        const h0 = mid3(a0, b0), h1 = mid3(a1, b1);
        ins.push(h0[0], h0[1], h0[2], h1[0], h1[1], h1[2]);
        const w0 = mid3(a0, a1), w1 = mid3(b0, b1);
        ins.push(w0[0], w0[1], w0[2], w1[0], w1[1], w1[2]);
      }
    }

    /* compact objects: a radial (u = atan2, v = y) chart wrapped on the object itself */
    function radialChart(seg, cp, ins) {
      const NU = 26, NV = 15, P = sampler.aPos, N = sampler.N;
      let ymin = Infinity, ymax = -Infinity, cx = 0, cz = 0;
      for (let i = 0; i < N; i++) {
        const y = P[i * 3 + 1];
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
        cx += P[i * 3]; cz += P[i * 3 + 2];
      }
      cx /= N; cz /= N;
      const acc = new Float32Array(NU * NV), w = new Float32Array(NU * NV);
      for (let i = 0; i < N; i++) {
        const dx = P[i * 3] - cx, dz = P[i * 3 + 2] - cz;
        const r = Math.sqrt(dx * dx + dz * dz);
        let u = Math.atan2(dz, dx); if (u < 0) u += Math.PI * 2;
        const iu = Math.min(NU - 1, (u / (Math.PI * 2) * NU) | 0);
        const iv = Math.min(NV - 1, ((P[i * 3 + 1] - ymin) / Math.max(1e-3, ymax - ymin) * NV) | 0);
        acc[iv * NU + iu] += r; w[iv * NU + iu]++;
      }
      const R = new Float32Array(NU * NV);
      let mean = 0, mc = 0;
      for (let k = 0; k < R.length; k++) if (w[k]) { R[k] = acc[k] / w[k]; mean += R[k]; mc++; }
      mean /= Math.max(1, mc);
      for (let k = 0; k < R.length; k++) if (!w[k]) R[k] = mean;
      for (let pass = 0; pass < 3; pass++) {
        const S = R.slice();
        for (let v = 0; v < NV; v++) for (let u = 0; u < NU; u++) {
          const um = (u + NU - 1) % NU, up = (u + 1) % NU;
          const vm = Math.max(0, v - 1), vp = Math.min(NV - 1, v + 1);
          R[v * NU + u] = 0.125 * (S[v * NU + um] + S[v * NU + up] + S[vm * NU + u] + S[vp * NU + u]) + 0.5 * S[v * NU + u];
        }
      }
      const at = (u, v) => {
        const a = (u / NU) * Math.PI * 2, r = R[v * NU + u];
        return [cx + Math.cos(a) * r, ymin + ((v + 0.5) / NV) * (ymax - ymin), cz + Math.sin(a) * r];
      };
      for (let v = 0; v < NV; v++) for (let u = 0; u < NU; u++) {
        const a = at(u, v);
        cp.push(a[0], a[1], a[2]);
        const b = at((u + 1) % NU, v);
        seg.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        if (v < NV - 1) { const c = at(u, v + 1); seg.push(a[0], a[1], a[2], c[0], c[1], c[2]); }
      }
      if (!ins) return;
      let bu = 0, bvv = 0, bd = Infinity;
      for (let v = 0; v < NV; v++) for (let u = 0; u < NU; u++) {
        const q = at(u, v), qx = q[0] - KNOT.c.x, qy = q[1] - KNOT.c.y, qz = q[2] - KNOT.c.z;
        const d = qx * qx + qy * qy + qz * qz;
        if (d < bd) { bd = d; bu = u; bvv = v; }
      }
      const mid3 = (a, b) => [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
      for (let dv = -2; dv <= 2; dv++) for (let du = -3; du <= 3; du++) {
        const v = bvv + dv;
        if (v < 0 || v >= NV - 1) continue;
        const u = (bu + du + NU) % NU, u1 = (u + 1) % NU;
        const a0 = at(u, v), a1 = at(u1, v), b0 = at(u, v + 1), b1 = at(u1, v + 1);
        const h0 = mid3(a0, b0), h1 = mid3(a1, b1);
        ins.push(h0[0], h0[1], h0[2], h1[0], h1[1], h1[2]);
        const w0 = mid3(a0, a1), w1 = mid3(b0, b1);
        ins.push(w0[0], w0[1], w0[2], w1[0], w1[1], w1[2]);
      }
    }

    let chartInsN = 0;
    function applyKnotLines() {
      if (!chartGrp || chartGrp.children.length < 3) return;
      const L = chartGrp.children[2];
      const k = clamp(st.knots || 0, 0, 1);
      const shown = Math.round(k * chartInsN) * 2;      // 2 vertices per inserted segment
      L.geometry.setDrawRange(0, shown);
      L.material.opacity = clamp(k * 2.4, 0, 1) * Math.max(st.chartOp, 0.6);
      L.visible = shown > 0;
      // dim the base net as knots come in so the inserted isolines are the thing you see
      const B = chartGrp.children[0];
      if (B) B.material.opacity = st.chartOp * (1 - 0.42 * k);
    }

    /* ══ budgeted triangle mesh ═══════════════════════════════════════
       The same surface drawn as triangles instead of splats, on its own budget. Selection is
       view-dependent and runs per refresh:
         1. frustum + backface — the same depth / screen-extent / incidence test the splats use
         2. occlusion — centroids are rasterised into a 96×64 nearest-depth buffer; a triangle
            more than a depth-scaled tolerance behind its cell's front-most centroid is dropped.
            Centroid resolution, so it is deliberately conservative: the tolerance grows with
            depth so silhouettes and grazing faces are not eaten.
         3. priority — projected area (area·cosθ / depth²) through a 128-bin log histogram, so
            the budget cutoff is found in O(n) rather than by sorting the candidate set.
       Same allocation policy as the splat path: the budget is spent only on surface that is in
       frame, facing the camera and not hidden. */
    const TRI_CAND = 150000, TRI_MAX = 60000, OGW = 96, OGH = 64;
    let TRI = null, triMesh = null, triWire = null, triCap = 0;
    const occ = new Float32Array(OGW * OGH);
    const triStat = { n: 0, cand: 0, frustum: 0, occluded: 0, pool: 0 };

    function disposeTri() {
      if (triWire) { scene.remove(triWire); triWire.material.dispose(); triWire = null; }
      if (triMesh) { scene.remove(triMesh); triMesh.geometry.dispose(); triMesh.material.dispose(); triMesh = null; }
      TRI = null; triCap = 0;
    }

    function buildTriSoup() {
      disposeTri();
      const metas = [];
      let total = 0;
      gt.group.updateMatrixWorld(true);
      gt.group.traverse(o => {
        if (!o.isMesh || !o.geometry || !o.geometry.attributes || !o.geometry.attributes.position) return;
        const g = o.geometry;
        const cnt = ((g.index ? g.index.count : g.attributes.position.count) / 3) | 0;
        if (cnt > 0) { metas.push({ o: o, cnt: cnt }); total += cnt; }
      });
      if (!total) return;
      /* Pass 1 — world-space base soup, decimated only if the source is already huge. */
      const stride = Math.max(1, Math.ceil(total / TRI_CAND));
      const baseN = Math.floor(total / stride) + 4;
      const bp = new Float32Array(baseN * 9), bc = new Float32Array(baseN * 3), bar = new Float32Array(baseN);
      const bc0 = new Float32Array(baseN * 3);
      const a = new T.Vector3(), b = new T.Vector3(), c3 = new T.Vector3();
      const e1 = new T.Vector3(), e2 = new T.Vector3(), nv = new T.Vector3();
      let base = 0, seen = 0, areaSum = 0;
      for (const M of metas) {
        const g = M.o.geometry, P = g.attributes.position, idx = g.index, m = M.o.matrixWorld;
        const mc = M.o.material && M.o.material.color ? M.o.material.color : null;
        const cr = mc ? mc.r : 0.62, cg = mc ? mc.g : 0.60, cb = mc ? mc.b : 0.56;
        /* captures are authored dark for the splat pass; flat-shaded triangles read far darker
           at the same albedo, so lift them into a legible range here */
        const lr = Math.min(1, cr * 1.75 + 0.10), lg = Math.min(1, cg * 1.75 + 0.11), lb = Math.min(1, cb * 1.75 + 0.13);
        for (let t = 0; t < M.cnt; t++, seen++) {
          if (seen % stride) continue;
          if (base >= baseN) break;
          const i0 = idx ? idx.getX(t * 3) : t * 3, i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1, i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
          a.fromBufferAttribute(P, i0).applyMatrix4(m);
          b.fromBufferAttribute(P, i1).applyMatrix4(m);
          c3.fromBufferAttribute(P, i2).applyMatrix4(m);
          const o9 = base * 9, o3 = base * 3;
          bp[o9] = a.x; bp[o9 + 1] = a.y; bp[o9 + 2] = a.z;
          bp[o9 + 3] = b.x; bp[o9 + 4] = b.y; bp[o9 + 5] = b.z;
          bp[o9 + 6] = c3.x; bp[o9 + 7] = c3.y; bp[o9 + 8] = c3.z;
          e1.subVectors(b, a); e2.subVectors(c3, a); nv.crossVectors(e1, e2);
          bar[base] = nv.length() * 0.5;
          areaSum += bar[base];
          bc[o3] = lr; bc[o3 + 1] = lg; bc[o3 + 2] = lb;
          bc0[o3] = cr; bc0[o3 + 1] = cg; bc0[o3 + 2] = cb;   // raw albedo for the splat pass
          base++;
        }
      }
      if (!base) return;
      /* Pass 2 — area-driven 4-way subdivision. The captures are built from boxes and swept
         props, so the source soup can be a few hundred triangles: the budget would never bind
         and the culling story would be invisible. Splitting to a uniform target AREA (rather
         than a uniform count) gives every surface the same triangle density, which is exactly
         what makes projected-area priority a fair allocator downstream. Real scanned meshes
         arrive dense and skip this. */
      const TARGET = Math.min(TRI_CAND, 132000);
      const tgtArea = areaSum / Math.max(TARGET, 1);
      const lvlOf = i => {
        if (tgtArea <= 0) return 0;
        const l = Math.round(Math.log(Math.max(bar[i] / tgtArea, 1)) / Math.log(4));
        return clamp(l, 0, 5);
      };
      let outN = 0;
      for (let i = 0; i < base; i++) outN += Math.pow(4, lvlOf(i));
      /* Budget-aware levels. Emission is in source order, so overflowing the cap dropped the
         TAIL of the soup — and one big low-poly prop can eat the whole cap on its own (the DTU
         table's 112 huge triangles go to level 5 = 114k), leaving later objects with NO domain
         triangles at all. The splat pool still sampled them, so the adaptive allocator had
         nothing to fund there and the bust rendered black with 40% of test points unfunded.
         Lower every level uniformly until the subdivided soup fits instead of truncating it. */
      let cut = 0;
      while (outN > TRI_CAND && cut < 5) {
        cut++;
        outN = 0;
        for (let i = 0; i < base; i++) outN += Math.pow(4, Math.max(0, lvlOf(i) - cut));
      }
      const lvlFit = i => Math.max(0, lvlOf(i) - cut);
      outN = Math.min(outN, TRI_CAND);
      const take = outN + 4;
      const p = new Float32Array(take * 9), nn = new Float32Array(take * 3);
      const cc = new Float32Array(take * 3), col = new Float32Array(take * 3), ar = new Float32Array(take);
      const col0 = new Float32Array(take * 3);
      let k = 0;
      const emit = (ax, ay, az, bx, by, bz, cx, cy, cz, cr, cg, cb, rr, rg, rb) => {
        if (k >= take) return;
        const o9 = k * 9, o3 = k * 3;
        p[o9] = ax; p[o9 + 1] = ay; p[o9 + 2] = az;
        p[o9 + 3] = bx; p[o9 + 4] = by; p[o9 + 5] = bz;
        p[o9 + 6] = cx; p[o9 + 7] = cy; p[o9 + 8] = cz;
        const ux = bx - ax, uy = by - ay, uz = bz - az;
        const vx = cx - ax, vy = cy - ay, vz = cz - az;
        let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1e-6;
        ar[k] = len * 0.5;
        nx /= len; ny /= len; nz /= len;
        nn[o3] = nx; nn[o3 + 1] = ny; nn[o3 + 2] = nz;
        cc[o3] = (ax + bx + cx) / 3; cc[o3 + 1] = (ay + by + cy) / 3; cc[o3 + 2] = (az + bz + cz) / 3;
        col[o3] = cr; col[o3 + 1] = cg; col[o3 + 2] = cb;
        col0[o3] = rr; col0[o3 + 1] = rg; col0[o3 + 2] = rb;
        k++;
      };
      const split = (lv, ax, ay, az, bx, by, bz, cx, cy, cz, cr, cg, cb, rr, rg, rb) => {
        if (lv <= 0) { emit(ax, ay, az, bx, by, bz, cx, cy, cz, cr, cg, cb, rr, rg, rb); return; }
        const mx0 = (ax + bx) * 0.5, my0 = (ay + by) * 0.5, mz0 = (az + bz) * 0.5;
        const mx1 = (bx + cx) * 0.5, my1 = (by + cy) * 0.5, mz1 = (bz + cz) * 0.5;
        const mx2 = (cx + ax) * 0.5, my2 = (cy + ay) * 0.5, mz2 = (cz + az) * 0.5;
        split(lv - 1, ax, ay, az, mx0, my0, mz0, mx2, my2, mz2, cr, cg, cb, rr, rg, rb);
        split(lv - 1, mx0, my0, mz0, bx, by, bz, mx1, my1, mz1, cr, cg, cb, rr, rg, rb);
        split(lv - 1, mx2, my2, mz2, mx1, my1, mz1, cx, cy, cz, cr, cg, cb, rr, rg, rb);
        split(lv - 1, mx0, my0, mz0, mx1, my1, mz1, mx2, my2, mz2, cr, cg, cb, rr, rg, rb);
      };
      for (let i = 0; i < base && k < take; i++) {
        const o9 = i * 9, o3 = i * 3;
        split(lvlFit(i), bp[o9], bp[o9 + 1], bp[o9 + 2], bp[o9 + 3], bp[o9 + 4], bp[o9 + 5],
          bp[o9 + 6], bp[o9 + 7], bp[o9 + 8], bc[o3], bc[o3 + 1], bc[o3 + 2],
          bc0[o3], bc0[o3 + 1], bc0[o3 + 2]);
      }
      if (!k) return;
      TRI = {
        p: p, n: nn, c: cc, col: col, col0: col0, area: ar, count: k, stride: stride, total: total,
        keep: new Uint8Array(k), pri: new Float32Array(k), cell: new Int32Array(k)
      };
      triCap = Math.min(TRI_MAX, k);
      const geo = new T.BufferGeometry();
      geo.setAttribute('position', new T.BufferAttribute(new Float32Array(triCap * 9), 3));
      geo.setAttribute('normal', new T.BufferAttribute(new Float32Array(triCap * 9), 3));
      geo.setAttribute('color', new T.BufferAttribute(new Float32Array(triCap * 9), 3));
      geo.setDrawRange(0, 0);
      geo.boundingSphere = new T.Sphere(gt.center.clone(), gt.radius * 3);
      /* The mesh pass gets its OWN shading rather than the scene's lights: the captures are
         authored near-black for the splat pass and the scene carries heavy fog, so a lit,
         fogged Lambert mesh reads as a black silhouette. Flat facing + a rim term in the
         instrument palette makes the triangle structure legible at any depth, and the vertex
         colour still tints it so different props stay distinguishable. Normals are per-face
         (all three vertices share one), so this is flat-shaded for free. */
      const triMat = new T.ShaderMaterial({
        side: T.DoubleSide,
        uniforms: { uL: { value: new T.Vector3(0.42, 0.84, 0.35).normalize() } },
        vertexShader: `
          attribute vec3 color;
          varying vec3 vN, vV, vC;
          void main() {
            vN = normalize(normal);
            vV = normalize(cameraPosition - position);
            vC = color;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }`,
        fragmentShader: `
          precision mediump float;
          uniform vec3 uL;
          varying vec3 vN, vV, vC;
          void main() {
            vec3 n = normalize(vN);
            float d = abs(dot(n, normalize(uL)));
            float f = pow(1.0 - abs(dot(n, normalize(vV))), 2.2);
            vec3 base = mix(vec3(0.12, 0.18, 0.28), vec3(0.60, 0.72, 0.86), 0.34 + 0.66 * d);
            base = mix(base, vC, 0.28);
            base += vec3(0.22, 0.36, 0.52) * f * 0.8;
            gl_FragColor = vec4(base, 1.0);
          }`
      });
      triMesh = new T.Mesh(geo, triMat);
      triMesh.frustumCulled = false; triMesh.visible = false;
      triWire = new T.Mesh(geo, new T.MeshBasicMaterial({ color: 0xdaeaf8, wireframe: true, transparent: true, opacity: 0.17, depthWrite: false, fog: false }));
      triWire.frustumCulled = false; triWire.visible = false;
      scene.add(triMesh); scene.add(triWire);
    }

    function selectTris() {
      if (!TRI || !triMesh) return;
      const n = TRI.count, C = TRI.c, NR = TRI.n, AR = TRI.area;
      const keep = TRI.keep, pri = TRI.pri, cell = TRI.cell;
      const px = view.pos.x, py = view.pos.y, pz = view.pos.z;
      const dx0 = view.dir.x, dy0 = view.dir.y, dz0 = view.dir.z;
      const e = camera.matrixWorld.elements;
      const rx = e[0], ry = e[1], rz = e[2], ux = e[4], uy = e[5], uz = e[6];
      const th = Math.tan(camera.fov * Math.PI / 360) || 0.38;
      const asp = camera.aspect || 1.6;
      occ.fill(1e9);
      let cand = 0, culledF = 0;
      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        const vx = C[i3] - px, vy = C[i3 + 1] - py, vz = C[i3 + 2] - pz;
        const depth = vx * dx0 + vy * dy0 + vz * dz0;
        keep[i] = 0;
        if (depth < 0.12) { culledF++; continue; }
        const sx = (vx * rx + vy * ry + vz * rz) / (th * asp * depth);
        const sy = (vx * ux + vy * uy + vz * uz) / (th * depth);
        if (sx < -1.08 || sx > 1.08 || sy < -1.08 || sy > 1.08) { culledF++; continue; }
        const fd = -(dx0 * NR[i3] + dy0 * NR[i3 + 1] + dz0 * NR[i3 + 2]);
        if (fd < -0.08) { culledF++; continue; }
        const cxi = clamp(((sx * 0.5 + 0.5) * OGW) | 0, 0, OGW - 1);
        const cyi = clamp(((0.5 - sy * 0.5) * OGH) | 0, 0, OGH - 1);
        const ci = cyi * OGW + cxi;
        cell[i] = ci;
        if (depth < occ[ci]) occ[ci] = depth;
        pri[i] = AR[i] * Math.max(Math.abs(fd), 0.05) / (depth * depth);
        keep[i] = 1; cand++;
      }
      const R = gt ? gt.radius : 10;
      let occluded = 0;
      for (let i = 0; i < n; i++) {
        if (!keep[i]) continue;
        const i3 = i * 3;
        const vx = C[i3] - px, vy = C[i3 + 1] - py, vz = C[i3 + 2] - pz;
        const depth = vx * dx0 + vy * dy0 + vz * dz0;
        if (depth > occ[cell[i]] + Math.max(R * 0.035, depth * 0.012)) { keep[i] = 0; occluded++; }
      }
      const want = Math.min(st.triBudget || 12000, triCap);
      /* Stratified allocation over the SAME 96×64 screen grid the occlusion pass built. Global
         projected-area priority (what this did first) spends the entire budget on the handful of
         nearest face-on triangles and leaves the rest of the visible surface empty — the count
         is right and the picture is wrong. Dividing the budget among OCCUPIED cells and taking
         the best few per cell gives uniform screen density, which is what spending a small
         budget efficiently actually means: no two triangles competing for one pixel while a
         neighbouring cell has none. */
      const cellsN = OGW * OGH;
      if (!TRI.cnt) { TRI.cnt = new Int32Array(cellsN); TRI.selN = new Int32Array(cellsN); }
      const cnt = TRI.cnt, selN = TRI.selN;
      cnt.fill(0);
      let occupied = 0;
      for (let i = 0; i < n; i++) if (keep[i]) { if (cnt[cell[i]]++ === 0) occupied++; }
      const Q = clamp(Math.ceil(want / Math.max(occupied, 1)) + 1, 1, 28);
      if (!TRI.sel || TRI.selQ !== Q) { TRI.sel = new Int32Array(cellsN * Q); TRI.selQ = Q; }
      const sel = TRI.sel;
      selN.fill(0);
      for (let i = 0; i < n; i++) {
        if (!keep[i]) continue;
        const c = cell[i], b = c * Q, v = pri[i], m = selN[c];
        if (m < Q) {
          let j = m - 1;
          while (j >= 0 && pri[sel[b + j]] < v) { sel[b + j + 1] = sel[b + j]; j--; }
          sel[b + j + 1] = i; selN[c] = m + 1;
        } else if (v > pri[sel[b + Q - 1]]) {
          let j = Q - 2;
          while (j >= 0 && pri[sel[b + j]] < v) { sel[b + j + 1] = sel[b + j]; j--; }
          sel[b + j + 1] = i;
        }
      }
      const G2 = triMesh.geometry;
      const PA = G2.attributes.position.array, NA = G2.attributes.normal.array, CA = G2.attributes.color.array;
      let k = 0;
      // rank-major sweep: every cell gets its best triangle before any cell gets its second
      for (let r = 0; r < Q && k < want; r++) {
        for (let c = 0; c < cellsN && k < want; c++) {
          if (r >= selN[c]) continue;
          const i = sel[c * Q + r];
          const s9 = i * 9, d9 = k * 9, s3 = i * 3;
          for (let j = 0; j < 9; j++) PA[d9 + j] = TRI.p[s9 + j];
          const nx = NR[s3], ny = NR[s3 + 1], nz = NR[s3 + 2];
          const cr = TRI.col[s3], cg = TRI.col[s3 + 1], cb = TRI.col[s3 + 2];
          for (let v = 0; v < 3; v++) {
            const o = d9 + v * 3;
            NA[o] = nx; NA[o + 1] = ny; NA[o + 2] = nz;
            CA[o] = cr; CA[o + 1] = cg; CA[o + 2] = cb;
          }
          k++;
        }
      }
      const verts = k * 3;
      G2.setDrawRange(0, verts);
      const upd = att => {
        att.needsUpdate = true;
        if (att.updateRange) { att.updateRange.offset = 0; att.updateRange.count = verts * 3; }
      };
      upd(G2.attributes.position); upd(G2.attributes.normal); upd(G2.attributes.color);
      triStat.n = k; triStat.cand = cand; triStat.frustum = culledF;
      triStat.occluded = occluded; triStat.pool = n;
    }

    /* ══ adaptive UV-domain resampling — how OURS spends its budget ═══════════════════
       v1 (swarm core) warps a UV lattice through an inverse CDF, so the SAMPLE POSITIONS
       migrate into the observed region: zoom in and the observed patch is genuinely denser.
       v4 used to pick a SUBSET of a static, area-uniform pool instead — peripheral primitives
       were dropped and the survivors kept the pool's global density, so past the point where
       the in-frustum pool ran out zoom bought nothing. That is the "culled, not concentrated"
       reading, and it was correct: the mechanism was selection, not resampling.

       This is v1's mechanism on an arbitrary mesh. The sampling domain is the surface's own
       parameter domain, [0,1]²:
         u — arclength along a MORTON ordering of the area-uniform triangle soup, so
             neighbouring u are neighbouring surface patches;
         v — barycentric coordinates inside the triangle that u lands in.
       Every refresh:
         1. per-triangle visibility = soft frustum ∩ front-facing ∩ NOT OCCLUDED (96×64
            software z-buffer, the same screen grid the mesh path already uses);
         2. weight = projected screen area (area·cosθ / depthᵖ) + a uniform floor sized to
            keep ~RESERVE of the budget on unobserved surface (it is still one global model,
            not a screen-space cache);
         3. slot i takes u = (i+0.5)/M through the inverse CDF, and a FIXED Hammersley pair
            for v. Both are fixed per slot, so when the weights change a slot SLIDES along
            the Morton curve instead of being reshuffled — the swarm flows with the camera
            instead of popping, and positions are eased on top of that.
       Density per unit surface area then tracks projected area: uniform screen-space density,
       no pool ceiling, and zoom refines without bound. */
    /* AD_EASE eases a slot toward the target it was given; AD_FLOW eases the TARGET FIELD
       itself. Both are needed and they are not redundant: easing positions alone cannot fix a
       target that jumped, because the allocation the camera asks for changes faster than any
       slot can walk there. Damping the weights turns "reallocate 40% of the budget this tick"
       into a density wave that crosses the surface over ~250 ms — which is the reallocation
       actually being visible, instead of a cut between two configurations. */
    const ADCAP = 60000, AD_EASE = 6.5, AD_FLOW = 4.2;
    const AD = {
      n: 0, nIn: 0, count: 0, uvVis: 0, occl: 0, conc: 1, visArea: 1, areaAll: 1, reserve: 0.06,
      warm: 0, dirty: false, occ: new Float32Array(OGW * OGH), att: null,
      cw: new Float32Array(OGW * OGH), cs: new Float32Array(OGW * OGH)
    };
    {
      const R = rng(20604);
      const radical = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = (i / b) | 0; } return r; };
      AD.pos = new Float32Array(ADCAP * 3); AD.tgt = new Float32Array(ADCAP * 3);
      AD.nrm = new Float32Array(ADCAP * 3); AD.tnr = new Float32Array(ADCAP * 3);
      AD.col = new Float32Array(ADCAP * 3); AD.tcl = new Float32Array(ADCAP * 3);
      AD.tan = new Float32Array(ADCAP * 3); AD.seed = new Float32Array(ADCAP * 3);
      AD.curv = new Float32Array(ADCAP); AD.zero = new Float32Array(ADCAP);
      AD.one = new Float32Array(ADCAP).fill(1);
      AD.hx = new Float32Array(ADCAP); AD.hy = new Float32Array(ADCAP);
      AD.mem = new Uint8Array(ADCAP); AD.cellOf = new Int32Array(ADCAP);
      /* q = where this slot currently sits along the domain ordering, and it PERSISTS across
         refreshes. That persistence is the whole point: a slot whose neighbourhood is already
         correctly served is never touched, so it does not move at all. scl/pend drive the
         shrink-fade used for the relocations that are too long to walk. */
      AD.q = new Int32Array(ADCAP); AD.pend = new Int32Array(ADCAP).fill(-1);
      /* two separate per-instance scalars, deliberately not one: aFade is the relocation
         shrink-fade, aScale is the splat's own footprint. The UV samplers set a real
         footprint per splat from the local (u,v) step, so a sparsely-stepped flat region
         draws bigger disks and still closes — density and scale move together, which is
         what keeps a view-driven sample set from tearing. */
      AD.fade = new Float32Array(ADCAP).fill(1);
      AD.usz = new Float32Array(ADCAP).fill(1);
      for (let i = 0; i < ADCAP; i++) {
        AD.seed[i * 3] = R(); AD.seed[i * 3 + 1] = R(); AD.seed[i * 3 + 2] = R();
        // low-discrepancy barycentric, fixed for the lifetime of the slot: consecutive slots
        // land in the same triangle, so a Hammersley subsequence spreads them inside it
        AD.hx[i] = radical(i + 1, 2); AD.hy[i] = radical(i + 1, 3);
        AD.nrm[i * 3 + 1] = 1; AD.tnr[i * 3 + 1] = 1; AD.tan[i * 3] = 1;
      }
    }
    const ADSRC = {
      aPos: AD.pos, aNrm: AD.nrm, aTan: AD.tan, aCol: AD.col, aSm: AD.pos, aSeed: AD.seed,
      aBirth: AD.zero, aBirthU: AD.zero, aScale: AD.usz, aFade: AD.fade,
      aCurv: AD.curv, aVoid: AD.one, N: ADCAP
    };
    const ADCOV = { P: AD.tgt, NR: AD.tnr, SM: AD.tgt, SD: AD.seed, CL: AD.tcl };
    const adOn = () => st.adapt !== false && AD.count > 0;

    /* built once per capture: the Morton ordering that makes the domain coordinate spatially
       coherent, plus a curvature proxy per triangle (tilt against the mean normal of its
       neighbourhood) for the shader's area-preserving anisotropy. */
    function adBuildDomain() {
      AD.count = 0; AD.warm = 0;
      if (!TRI || !TRI.count) return;
      const n = TRI.count, C = TRI.c, NR = TRI.n;
      let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
      for (let i = 0; i < n; i++) {
        const x = C[i * 3], y = C[i * 3 + 1], z = C[i * 3 + 2];
        if (x < mnx) mnx = x; if (x > mxx) mxx = x;
        if (y < mny) mny = y; if (y > mxy) mxy = y;
        if (z < mnz) mnz = z; if (z > mxz) mxz = z;
      }
      const ext = Math.max(mxx - mnx, Math.max(mxy - mny, mxz - mnz)) * 1.0001 || 1;
      const part = v => {                          // spread 10 bits across 30
        v = (v | 0) & 1023;
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        return v;
      };
      /* HILBERT, not Morton. The ordering is the domain coordinate every sample slot slides
         along when the camera moves, so its spatial continuity IS the smoothness of the
         reallocation. Z-order is compact on average but jumps the full extent of the box at
         every power-of-two boundary — ~300 such cliffs across a 100k-triangle capture — and a
         slot that slides a handful of ranks across one is thrown clear to the far side of the
         object. Hilbert has no cliffs: consecutive ranks are always adjacent cells, so sliding
         along the CDF is a walk over the surface and the eased motion is a real trajectory.
         Skilling's AxesToTranspose (b=10 bits per axis), then the same bit interleave. */
      const hilbert = (x, y, z) => {
        for (let Q = 512; Q > 1; Q >>= 1) {
          const P = Q - 1;
          if (x & Q) x ^= P;                        // i=0: exchange with itself is a no-op
          if (y & Q) x ^= P; else { const t = (x ^ y) & P; x ^= t; y ^= t; }
          if (z & Q) x ^= P; else { const t = (x ^ z) & P; x ^= t; z ^= t; }
        }
        y ^= x; z ^= y;                             // Gray encode
        let t = 0;
        for (let Q = 512; Q > 1; Q >>= 1) if (z & Q) t ^= Q - 1;
        x ^= t; y ^= t; z ^= t;
        return part(x) * 4 + part(y) * 2 + part(z);
      };
      const key = new Float64Array(n), ord = new Array(n);
      for (let i = 0; i < n; i++) {
        const ix = clamp(((C[i * 3] - mnx) / ext * 1024) | 0, 0, 1023);
        const iy = clamp(((C[i * 3 + 1] - mny) / ext * 1024) | 0, 0, 1023);
        const iz = clamp(((C[i * 3 + 2] - mnz) / ext * 1024) | 0, 0, 1023);
        key[i] = hilbert(ix, iy, iz);
        ord[i] = i;
      }
      ord.sort((a, b) => key[a] - key[b]);
      AD.ord = new Uint32Array(ord);
      AD.w = new Float32Array(n); AD.ws = new Float32Array(n); AD.wsAt = 0;
      AD.cum = new Float64Array(n + 1);
      /* transport bins: contiguous runs of the Hilbert ordering. Because the ordering is now
         spatially continuous, "the adjacent bin" IS "the adjacent patch of surface", so a 1D
         nearest-neighbour search over bins finds a genuinely nearby destination and the whole
         3D matching problem collapses to a scan. */
      AD.NBMAX = 8192;
      AD.binW = new Float64Array(AD.NBMAX); AD.want = new Float64Array(AD.NBMAX);
      AD.have = new Int32Array(AD.NBMAX); AD.need = new Float64Array(AD.NBMAX);
      AD.bHead = new Int32Array(AD.NBMAX); AD.bNext = new Int32Array(ADCAP);
      AD.sb = new Int32Array(ADCAP); AD.bCum = new Float64Array(n + 2);
      AD.dep = new Float32Array(n); AD.memT = new Float32Array(n);
      const NB = 32, cells = NB * NB * NB;
      const ax = new Float32Array(cells), ay = new Float32Array(cells), az = new Float32Array(cells);
      const ci = new Int32Array(n);
      for (let i = 0; i < n; i++) {
        const gx = clamp(((C[i * 3] - mnx) / ext * NB) | 0, 0, NB - 1);
        const gy = clamp(((C[i * 3 + 1] - mny) / ext * NB) | 0, 0, NB - 1);
        const gz = clamp(((C[i * 3 + 2] - mnz) / ext * NB) | 0, 0, NB - 1);
        const c = gx + gy * NB + gz * NB * NB;
        ci[i] = c;
        ax[c] += NR[i * 3]; ay[c] += NR[i * 3 + 1]; az[c] += NR[i * 3 + 2];
      }
      AD.cv = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        const c = ci[i];
        const l = Math.sqrt(ax[c] * ax[c] + ay[c] * ay[c] + az[c] * az[c]) || 1e-6;
        const d = (NR[i * 3] * ax[c] + NR[i * 3 + 1] * ay[c] + NR[i * 3 + 2] * az[c]) / l;
        AD.cv[i] = clamp((1 - Math.abs(d)) * 2.2, 0, 1);
      }
      AD.count = n;
    }

    /* ══ (u,v) ATLAS — 6-chart cube projection ═══════════════════════════════════════
       Both UV samplers need what the captures do not have: an actual parametric domain.
       Every triangle is assigned to exactly ONE of six axis-aligned charts by dominant
       signed normal, so the six charts partition the surface (no double counting, no
       gaps), and inside a chart the two axes orthogonal to the projection ARE u and v —
       separable, which is what "uniform along U and V" and the t_u / t_v step formulas
       both require.

       Each chart is a height field: (u,v) ↦ the frontmost surface along the chart axis.
       Because a triangle only lands in the chart its normal most faces, inclination
       inside a chart is ≤45°, so projected-area sampling is within √2 of surface-area
       sampling. The known cost of this choice: one surface per projection ray, so
       overhangs and enclosed interiors are flattened to their frontmost sheet. */
    /* 256² per chart, over the whole scene bbox. The captures put ground, trees and silo
       in the same box as the subject, so a chart's cells are shared out among all of them —
       at 128² the barn itself only resolved to roughly 40×40 and the height field went blocky. */
    const ATLAS = { NU: 256, NV: 256, charts: [], built: false, cells: 0 };
    // axis, sign, and the two domain axes (au = u, av = v)
    const CHARTDEF = [
      { ax: 0, sg: +1, au: 2, av: 1 }, { ax: 0, sg: -1, au: 2, av: 1 },
      { ax: 1, sg: +1, au: 0, av: 2 }, { ax: 1, sg: -1, au: 0, av: 2 },
      { ax: 2, sg: +1, au: 0, av: 1 }, { ax: 2, sg: -1, au: 0, av: 1 }
    ];

    function atlasBuild() {
      ATLAS.charts = []; ATLAS.built = false; ATLAS.cells = 0;
      if (!TRI || !TRI.count) return;
      const n = TRI.count, P = TRI.p, NR = TRI.n, CL = TRI.col0 || TRI.col;
      const NU = ATLAS.NU, NV = ATLAS.NV;
      // one shared world bbox for every chart, so charts that meet at a seam agree on
      // the world position of that seam (the precondition for Chhugani §2.6)
      const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n * 3; i++) {
        for (let k = 0; k < 3; k++) {
          const v = P[i * 3 + k];
          if (v < mn[k]) mn[k] = v; if (v > mx[k]) mx[k] = v;
        }
      }
      for (let k = 0; k < 3; k++) {
        const pad = Math.max((mx[k] - mn[k]) * 0.001, 1e-4);
        mn[k] -= pad; mx[k] += pad;
        if (mx[k] - mn[k] < 1e-6) mx[k] = mn[k] + 1e-6;
      }
      for (const d of CHARTDEF) {
        const C = {
          ax: d.ax, sg: d.sg, au: d.au, av: d.av, NU: NU, NV: NV,
          u0: mn[d.au], u1: mx[d.au], v0: mn[d.av], v1: mx[d.av],
          H: new Float32Array(NU * NV), mask: new Uint8Array(NU * NV),
          nrm: new Float32Array(NU * NV * 3), col: new Float32Array(NU * NV * 3),
          cv: new Float32Array(NU * NV), tris: 0
        };
        C.du = (C.u1 - C.u0) / NU; C.dv = (C.v1 - C.v0) / NV;
        C.H.fill(d.sg > 0 ? -Infinity : Infinity);
        ATLAS.charts.push(C);
      }
      // assign each triangle to the chart its normal most faces, then rasterise
      for (let t = 0; t < n; t++) {
        const t3 = t * 3;
        let best = 0, bv = -Infinity;
        for (let c = 0; c < 6; c++) {
          const d = CHARTDEF[c], s = NR[t3 + d.ax] * d.sg;
          if (s > bv) { bv = s; best = c; }
        }
        const C = ATLAS.charts[best], d = CHARTDEF[best], t9 = t * 9;
        C.tris++;
        const au = d.au, av = d.av, ax = d.ax;
        const uA = (P[t9 + au] - C.u0) / C.du, vA = (P[t9 + av] - C.v0) / C.dv, hA = P[t9 + ax];
        const uB = (P[t9 + 3 + au] - C.u0) / C.du, vB = (P[t9 + 3 + av] - C.v0) / C.dv, hB = P[t9 + 3 + ax];
        const uC = (P[t9 + 6 + au] - C.u0) / C.du, vC = (P[t9 + 6 + av] - C.v0) / C.dv, hC = P[t9 + 6 + ax];
        const i0 = Math.max(0, Math.floor(Math.min(uA, uB, uC))), i1 = Math.min(NU - 1, Math.ceil(Math.max(uA, uB, uC)));
        const j0 = Math.max(0, Math.floor(Math.min(vA, vB, vC))), j1 = Math.min(NV - 1, Math.ceil(Math.max(vA, vB, vC)));
        const det = (vB - vC) * (uA - uC) + (uC - uB) * (vA - vC);
        if (Math.abs(det) < 1e-12) continue;
        for (let j = j0; j <= j1; j++) {
          for (let i = i0; i <= i1; i++) {
            const pu = i + 0.5, pv = j + 0.5;
            let l0 = ((vB - vC) * (pu - uC) + (uC - uB) * (pv - vC)) / det;
            let l1 = ((vC - vA) * (pu - uC) + (uA - uC) * (pv - vC)) / det;
            const l2 = 1 - l0 - l1;
            if (l0 < -0.001 || l1 < -0.001 || l2 < -0.001) continue;
            const h = l0 * hA + l1 * hB + l2 * hC;
            const k = j * NU + i;
            // frontmost as seen from this chart's side
            if (C.mask[k] && (d.sg > 0 ? h <= C.H[k] : h >= C.H[k])) continue;
            C.H[k] = h; C.mask[k] = 1;
            const k3 = k * 3;
            C.nrm[k3] = NR[t3]; C.nrm[k3 + 1] = NR[t3 + 1]; C.nrm[k3 + 2] = NR[t3 + 2];
            C.col[k3] = CL[t3]; C.col[k3 + 1] = CL[t3 + 1]; C.col[k3 + 2] = CL[t3 + 2];
          }
        }
      }
      /* curvature proxy per cell from the height field's own second differences — this is
         also |F_uu| / |F_vv|, which §2.5's step sizes need */
      for (const C of ATLAS.charts) {
        let occ = 0;
        for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) {
          const k = j * NU + i;
          if (!C.mask[k]) continue;
          occ++;
          const im = Math.max(0, i - 1), ip = Math.min(NU - 1, i + 1);
          const jm = Math.max(0, j - 1), jp = Math.min(NV - 1, j + 1);
          const ku = j * NU + im, kU = j * NU + ip, kv = jm * NU + i, kV = jp * NU + i;
          const huu = (C.mask[ku] && C.mask[kU]) ? Math.abs(C.H[ku] - 2 * C.H[k] + C.H[kU]) : 0;
          const hvv = (C.mask[kv] && C.mask[kV]) ? Math.abs(C.H[kv] - 2 * C.H[k] + C.H[kV]) : 0;
          C.cv[k] = Math.max(huu, hvv);
        }
        C.occ = occ; ATLAS.cells += occ;
      }
      // normalise the curvature proxy to 0..1 for the shader's aniso term
      let cmax = 1e-9;
      for (const C of ATLAS.charts) for (let k = 0; k < C.cv.length; k++) if (C.cv[k] > cmax) cmax = C.cv[k];
      for (const C of ATLAS.charts) for (let k = 0; k < C.cv.length; k++) C.cv[k] = clamp(C.cv[k] / cmax, 0, 1);
      ATLAS.built = ATLAS.cells > 0;
    }

    // world position of a chart cell centre
    function uvPos(C, i, j, out) {
      const k = j * C.NU + i;
      out[C.au] = C.u0 + (i + 0.5) * C.du;
      out[C.av] = C.v0 + (j + 0.5) * C.dv;
      out[C.ax] = C.H[k];
      return out;
    }

    /* F(u,v) at CONTINUOUS (u,v), in cell units. The chart grid stores the height field but
       it must not be the limit on how finely the domain can be sampled: zoom in far enough
       and a framed patch spans only a handful of cells, so cell-centre sampling caps the
       achievable count and the budget can no longer be spent — the opposite of the
       resolution-knob claim. Bilinear H makes the domain continuous, so both samplers can
       take sub-cell steps and keep filling the budget as the view tightens.
       Bilinear is suppressed across a mask boundary (silhouettes, depth jumps), where
       interpolating between surface and no-surface would invent geometry. */
    let uvAttrK = 0;                 // cell uvPosAt drew attributes from — set on every call
    function uvPosAt(C, fu, fv, out) {
      const NU = C.NU, NV = C.NV;
      const gu = clamp(fu - 0.5, 0, NU - 1.0001), gv = clamp(fv - 0.5, 0, NV - 1.0001);
      const i = gu | 0, j = gv | 0;
      const tx = gu - i, ty = gv - j;
      const ip = Math.min(NU - 1, i + 1), jp = Math.min(NV - 1, j + 1);
      const k00 = j * NU + i, k10 = j * NU + ip, k01 = jp * NU + i, k11 = jp * NU + ip;
      /* attributes come from whichever of the four is occupied and nearest, and the mask
         test upstream uses THAT cell too — an earlier version floored (fu,fv) for the mask
         but interpolated around (fu-½,fv-½), so the two disagreed by half a cell and killed
         samples whose own cell was fine. */
      const m00 = C.mask[k00], m10 = C.mask[k10], m01 = C.mask[k01], m11 = C.mask[k11];
      uvAttrK = (tx < 0.5 ? (ty < 0.5 ? k00 : k01) : (ty < 0.5 ? k10 : k11));
      if (!C.mask[uvAttrK]) uvAttrK = m00 ? k00 : m10 ? k10 : m01 ? k01 : m11 ? k11 : -1;
      let h;
      if (m00 && m10 && m01 && m11) {
        h = C.H[k00] * (1 - tx) * (1 - ty) + C.H[k10] * tx * (1 - ty)
          + C.H[k01] * (1 - tx) * ty + C.H[k11] * tx * ty;
      } else {
        h = uvAttrK >= 0 ? C.H[uvAttrK] : NaN;           // no bilinear across a mask edge
      }
      out[C.au] = C.u0 + fu * C.du;
      out[C.av] = C.v0 + fv * C.dv;
      out[C.ax] = h;
      return out;
    }

    /* per-cell visibility for this camera. Frustum + backface always; occlusion only when
       st.uvOcclude is on — it is off by default because occlusion flips discontinuously as
       you orbit (a wall going hidden dumps its whole share elsewhere in one step), which is
       the same transport spike the Hilbert/transport work exists to avoid. */
    const uvTmp = [0, 0, 0];
    function uvVisible(C, fu, fv) {
      const p = uvPosAt(C, fu, fv, uvTmp);
      const k = uvAttrK;
      if (k < 0 || !(p[C.ax] === p[C.ax])) return false;  // no surface under this (u,v)
      const vx = p[0] - view.pos.x, vy = p[1] - view.pos.y, vz = p[2] - view.pos.z;
      const depth = vx * view.dir.x + vy * view.dir.y + vz * view.dir.z;
      if (depth < 0.12) return false;
      const e = camera.matrixWorld.elements;
      const th = Math.max(view.tanHalf, 1e-3), asp = clamp(camera.aspect || 1.6, 0.4, 4);
      const sx = (vx * e[0] + vy * e[1] + vz * e[2]) / (th * asp * depth);
      const sy = (vx * e[4] + vy * e[5] + vz * e[6]) / (th * depth);
      if (Math.abs(sx) > 1 || Math.abs(sy) > 1) return false;
      const k3 = k * 3;
      const fd = -(view.dir.x * C.nrm[k3] + view.dir.y * C.nrm[k3 + 1] + view.dir.z * C.nrm[k3 + 2]);
      if (fd <= 0.02) return false;                       // backface
      if (st.uvOcclude) {
        const cc = clamp(((sx * 0.5 + 0.5) * OGW) | 0, 0, OGW - 1)
          + clamp(((0.5 - sy * 0.5) * OGH) | 0, 0, OGH - 1) * OGW;
        const R = gt ? gt.radius : 10;
        if (depth > AD.occ[cc] + Math.max(R * 0.05, depth * 0.025)) return false;
      }
      return true;
    }

    /* per-triangle allocation weight for this camera. Two passes: fill the z-buffer from the
       front-facing candidates, then discount whatever sits behind it. Everything is smooth
       (smoothstep bands, never a hard keep/drop) so the CDF — and therefore every sample
       position — moves continuously as the camera turns. */
    function adWeights(snap) {
      if (!AD.count) return;
      const n = AD.count, C = TRI.c, NR = TRI.n, AR = TRI.area, w = AD.w, dep = AD.dep, memT = AD.memT;
      const cellA = TRI.cell, priA = TRI.pri, occA = AD.occ;
      const px = view.pos.x, py = view.pos.y, pz = view.pos.z;
      const dx0 = view.dir.x, dy0 = view.dir.y, dz0 = view.dir.z;
      const e = camera.matrixWorld.elements;
      const rx = e[0], ry = e[1], rz = e[2], ux = e[4], uy = e[5], uz = e[6];
      const th = Math.max(view.tanHalf, 1e-3), asp = clamp(camera.aspect || 1.6, 0.4, 4);
      const R = gt ? gt.radius : 10;
      // roiTight biases the SHARE a screen cell gets toward nearer cells: 2 = strictly uniform
      // screen-space density, 3 = nearest-surface-first
      const pw = st.roiTight > 1.75 ? 3 : st.roiTight > 1.25 ? 2.5 : 2;
      occA.fill(1e9);
      AD.cw.fill(0); AD.cs.fill(0);
      for (let i = 0; i < n; i++) {
        const i3 = i * 3;
        w[i] = 0; dep[i] = 0; memT[i] = 0;
        const vx = C[i3] - px, vy = C[i3 + 1] - py, vz = C[i3 + 2] - pz;
        const depth = vx * dx0 + vy * dy0 + vz * dz0;
        if (depth < 0.12) continue;
        const sx = (vx * rx + vy * ry + vz * rz) / (th * asp * depth);
        const sy = (vx * ux + vy * uy + vz * uz) / (th * depth);
        const q = Math.max(Math.abs(sx), Math.abs(sy));
        if (q > 1.34) continue;
        const fd = -(dx0 * NR[i3] + dy0 * NR[i3 + 1] + dz0 * NR[i3 + 2]);
        if (fd < -0.05) continue;
        const mem = (1 - smoothstepJS(0.96, 1.26, q)) * smoothstepJS(-0.05, 0.14, fd);
        if (mem <= 0.002) continue;
        const c = clamp(((sx * 0.5 + 0.5) * OGW) | 0, 0, OGW - 1) + clamp(((0.5 - sy * 0.5) * OGH) | 0, 0, OGH - 1) * OGW;
        cellA[i] = c;
        if (depth < occA[c]) occA[c] = depth;
        dep[i] = depth; memT[i] = mem;
      }
      /* pass 2 — occlusion, then STRATIFIED allocation over the 96×64 screen grid.
         Weighting each triangle by its own projected area (area·cosθ/depthᵖ) is unbounded as
         depth → near plane: on the DTU captures a handful of table triangles running in close
         to the camera took 10³–10⁵× everyone else's weight, swallowed the CDF and left 88% of
         the visible surface unfunded. selectTris already learned this on the mesh path. So the
         budget is divided among OCCUPIED SCREEN CELLS — that IS uniform screen-space density,
         and it is bounded by construction — and each cell's share is split among the triangles
         behind it in proportion to their (visible) surface area. */
      let visArea = 0, hidArea = 0, tot = 0, areaAll = 0;
      const cw = AD.cw, cs = AD.cs;
      G.areaC.fill(0);
      for (let i = 0; i < n; i++) {
        const a = AR[i];
        areaAll += a;
        priA[i] = 0;
        const depth = dep[i];
        if (depth <= 0) continue;
        const c = cellA[i];
        const bias = Math.max(R * 0.045, depth * 0.02);
        const occS = 1 - smoothstepJS(bias, bias * 3.4, depth - occA[c]);
        const mem = memT[i], v = mem * occS;
        visArea += a * v;
        hidArea += a * mem * (1 - occS);
        memT[i] = v;
        if (v <= 0.004) continue;
        const contrib = a * v;
        priA[i] = contrib;
        cw[c] += contrib;
        if (v > cs[c]) cs[c] = v;
        G.areaC[gcellOf(C[i * 3], C[i * 3 + 1], C[i * 3 + 2])] += contrib;
      }
      const vd = Math.max(view.dist, 0.2);
      for (let i = 0; i < n; i++) {
        w[i] = 0;
        const contrib = priA[i];
        if (contrib <= 0) continue;
        const c = cellA[i];
        const rel = clamp(vd / Math.max(occA[c], 0.2), 0.35, 3);
        const share = cs[c] * (pw === 2 ? 1 : pw === 3 ? rel : Math.sqrt(rel));
        w[i] = contrib / Math.max(cw[c], 1e-9) * share;
        tot += w[i];
      }
      /* the reserve is a uniform floor over the whole domain, sized so that ~reserve of the
         CDF mass — and so of the budget — lands on unobserved surface. Ours keeps modelling
         what it cannot see; it just stops paying full price for it. */
      const reserve = clamp(0.11 - 0.06 * (st.roiTight - 1), 0.02, 0.12);
      AD.reserve = reserve;
      const floor = tot > 0 ? reserve / (1 - reserve) * tot / Math.max(areaAll, 1e-6) : 1;
      for (let i = 0; i < n; i++) w[i] += AR[i] * floor;
      if (KNOT.w > 0.001) {                        // knot insertion migrates weight, not ranks
        const kr = Math.max(KNOT.r, 1e-3);
        for (let i = 0; i < n; i++) {
          const ddx = C[i * 3] - KNOT.c.x, ddy = C[i * 3 + 1] - KNOT.c.y, ddz = C[i * 3 + 2] - KNOT.c.z;
          const kd = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz) / kr;
          const kg = Math.exp(-kd * kd * 1.35);
          w[i] *= Math.max(0.05, 1 + KNOT.w * (2.4 * kg - 0.3));
        }
      }
      /* the CDF is driven by a TIME-EASED copy of the weights, never by w directly. w is what
         this camera asks for; ws is what the model has managed to redistribute so far. The
         ledger below still reports w's view statistics, so the readouts stay truthful to the
         current frame — only the primitives lag, which is the honest depiction of a fixed
         budget being moved rather than re-drawn. */
      const wNow = performance.now();
      const wdt = AD.wsAt ? Math.min(0.25, (wNow - AD.wsAt) / 1000) : 0;
      AD.wsAt = wNow;
      const ws = AD.ws;
      if (snap || !wdt) {
        ws.set(w.subarray(0, n));
      } else {
        const rf = 1 - Math.exp(-AD_FLOW * wdt);
        for (let i = 0; i < n; i++) ws[i] += (w[i] - ws[i]) * rf;
      }

      AD.areaAll = areaAll;
      AD.visArea = Math.max(visArea, areaAll * 0.004);
      AD.uvVis = visArea / Math.max(areaAll, 1e-6);
      AD.occl = hidArea / Math.max(visArea + hidArea, 1e-6);
    }

    /* bind slot i to rank q of the ordering: its surface point, normal, colour and tangent.
       The barycentric pair is fixed for the slot's lifetime, so re-placing the same slot on
       the same rank is exactly idempotent — which is what lets the transport pass leave
       settled slots strictly untouched. */
    function adPlace(i, q) {
      const P = TRI.p, NR = TRI.n, CL = TRI.col0 || TRI.col;
      const t = AD.ord[q], t9 = t * 9, t3 = t * 3, o3 = i * 3;
      let ba = AD.hx[i], bb = AD.hy[i];
      if (ba + bb > 1) { ba = 1 - ba; bb = 1 - bb; }
      const axx = P[t9], ayy = P[t9 + 1], azz = P[t9 + 2];
      const abx = P[t9 + 3] - axx, aby = P[t9 + 4] - ayy, abz = P[t9 + 5] - azz;
      const acx = P[t9 + 6] - axx, acy = P[t9 + 7] - ayy, acz = P[t9 + 8] - azz;
      AD.tgt[o3] = axx + abx * ba + acx * bb;
      AD.tgt[o3 + 1] = ayy + aby * ba + acy * bb;
      AD.tgt[o3 + 2] = azz + abz * ba + acz * bb;
      AD.tnr[o3] = NR[t3]; AD.tnr[o3 + 1] = NR[t3 + 1]; AD.tnr[o3 + 2] = NR[t3 + 2];
      const j = 0.94 + AD.seed[o3] * 0.12;
      AD.tcl[o3] = CL[t3] * j; AD.tcl[o3 + 1] = CL[t3 + 1] * j; AD.tcl[o3 + 2] = CL[t3 + 2] * j;
      const tl = Math.sqrt(abx * abx + aby * aby + abz * abz) || 1;
      AD.tan[o3] = abx / tl; AD.tan[o3 + 1] = aby / tl; AD.tan[o3 + 2] = abz / tl;
      AD.curv[i] = AD.cv[t];
      AD.q[i] = q;
    }

    // visibility flags + in-view count, refreshed every tick regardless of who moved
    function adMark(M) {
      const memT = AD.memT, ord = AD.ord;
      let nIn = 0;
      for (let i = 0; i < M; i++) { const m = memT[ord[AD.q[i]]] > 0.5 ? 1 : 0; AD.mem[i] = m; nIn += m; }
      AD.nIn = nIn;
    }

    /* stratified inverse CDF: slot i ↦ u = (i+0.5)/M. Monotone in i, so one linear sweep over
       the ordered domain places the whole budget — O(domain + budget), no per-slot search.
       This is now the WARM START only. Using it every refresh is what made the splats pop:
       it is an absolute map, so a weight change anywhere shifts the rank of every slot behind
       it, and the whole cloud gets re-drawn from scratch against a camera that barely moved.
       Steady state is adTransport, which only repairs the difference. */
    function adSample(M) {
      const n = AD.count;
      if (!n) { AD.n = 0; return; }
      const ord = AD.ord, w = AD.ws, cum = AD.cum;
      let tot = 0;
      cum[0] = 0;
      for (let q = 0; q < n; q++) { tot += w[ord[q]]; cum[q + 1] = tot; }
      if (!(tot > 0)) { AD.n = 0; return; }
      let q = 0;
      for (let i = 0; i < M; i++) {
        const u = (i + 0.5) / M * tot;
        while (q < n - 1 && cum[q + 1] < u) q++;
        adPlace(i, q);
        AD.pend[i] = -1; AD.fade[i] = 1;
      }
      AD.n = M;
      adMark(M);
      AD.dirty = true;
    }

    /* ══ MODE 1 — uniform (u,v) ══════════════════════════════════════════════════════
       A single lattice step s, shared by all six charts, taken on the chart grids:
       u_i = (i+½)·s, v_j = (j+½)·s. Uniform in U and in V by construction.

       Budget utilisation: only cells that pass uvVisible() are emitted, so every splat
       drawn is one the camera can see, and the step is solved so the emitted count lands
       on the budget. Density therefore falls out of the view — frame less surface and the
       same budget lands on it at a finer step. That is the "resolution knob" claim, made
       exactly rather than by feedback. */
    /* local step, in chart cells, for a target spacing of `ps` PIXELS on screen.
       ps·s(B) is the world distance that projects to ps pixels at this depth — s(B) is the
       same scale factor §2.4 uses — and dividing by the cell size, with the slope correction
       that turns projected distance into distance along the surface, gives the step.

       This is what makes the lattice uniform on SCREEN rather than uniform in the atlas. A
       lattice that is literally uniform in (u,v) is uniform in projected area, and on these
       captures the ground plane is most of the projected area, so it swallowed ~42% of the
       budget while the subject starved. Stepping by depth spends the budget where the pixels
       are. */
    const uvStep = [1, 1];
    function uvStepAt(C, fu, fv, ps) {
      const p = uvPosAt(C, fu, fv, uvTmp);
      const k = uvAttrK;
      if (k < 0 || !(p[C.ax] === p[C.ax])) { uvStep[0] = uvStep[1] = 2; return false; }
      const depth = (p[0] - view.pos.x) * view.dir.x + (p[1] - view.pos.y) * view.dir.y
        + (p[2] - view.pos.z) * view.dir.z;
      if (depth < 0.12) { uvStep[0] = uvStep[1] = 4; return false; }
      const world = ps * uvScaleFactor(depth);
      const i = k % C.NU, j = (k / C.NU) | 0;
      const im = Math.max(0, i - 1), ip = Math.min(C.NU - 1, i + 1);
      const jm = Math.max(0, j - 1), jp = Math.min(C.NV - 1, j + 1);
      const ku = j * C.NU + im, kU = j * C.NU + ip, kv = jm * C.NU + i, kV = jp * C.NU + i;
      const gu = (C.mask[ku] && C.mask[kU]) ? (C.H[kU] - C.H[ku]) / (2 * C.du) : 0;
      const gv = (C.mask[kv] && C.mask[kV]) ? (C.H[kV] - C.H[kv]) / (2 * C.dv) : 0;
      uvStep[0] = clamp(world / (C.du * Math.sqrt(1 + Math.min(gu * gu, 8))), 0.03, C.NU);
      uvStep[1] = clamp(world / (C.dv * Math.sqrt(1 + Math.min(gv * gv, 8))), 0.03, C.NV);
      return true;
    }

    function uvCountAt(ps, M, emit) {
      let cnt = 0;
      for (let c = 0; c < 6; c++) {
        const C = ATLAS.charts[c];
        if (!C.occ) continue;
        let fv = 0.25, rowGuard = 0;
        let sv = 2;
        while (fv < C.NV && rowGuard++ < 20000) {
          let rowSv = -1, fu = 0.25, colGuard = 0;
          while (fu < C.NU && colGuard++ < 20000) {
            const ok = uvStepAt(C, fu, fv, ps);
            const su = uvStep[0];
            if (ok && uvVisible(C, fu, fv)) {
              /* row pitch = the TIGHTEST any hit in this row asked for, not the first one's.
                 Taking the first hit let a row that opens on distant surface advance by that
                 far step and stride straight over nearer geometry further along it, which
                 came out as horizontal bands of missing surface. Erring tight over-samples
                 slightly; the budget solve takes it back out globally. */
              if (rowSv < 0 || uvStep[1] < rowSv) rowSv = uvStep[1];
              if (emit) { if (cnt < M) emit(cnt, c, fu, fv, su, uvStep[1]); else return cnt + 1; }
              cnt++;
              if (!emit && cnt > M) return cnt;           // bisect only needs the comparison
            }
            fu += su;
          }
          sv = rowSv > 0 ? rowSv : sv;
          fv += sv;
        }
      }
      return cnt;
    }

    /* world footprint of one sample, from its own step in the domain. A step of su cells in
       u covers su·du of the chart plane, but the SURFACE under it is longer by the local
       slope — sqrt(1 + (dH/du)²) — and since a triangle only ever lands in the chart its
       normal most faces, that factor is bounded by √2. Without this the disks are sized from
       a global spacing estimate and the flat, sparsely-stepped regions tear open. */
    function uvFoot(C, fu, fv, su, sv) {
      const i = clamp(Math.round(fu - 0.5), 0, C.NU - 1), j = clamp(Math.round(fv - 0.5), 0, C.NV - 1);
      const im = Math.max(0, i - 1), ip = Math.min(C.NU - 1, i + 1);
      const jm = Math.max(0, j - 1), jp = Math.min(C.NV - 1, j + 1);
      const ku = j * C.NU + im, kU = j * C.NU + ip, kv = jm * C.NU + i, kV = jp * C.NU + i;
      const gu = (C.mask[ku] && C.mask[kU]) ? (C.H[kU] - C.H[ku]) / (2 * C.du) : 0;
      const gv = (C.mask[kv] && C.mask[kV]) ? (C.H[kV] - C.H[kv]) / (2 * C.dv) : 0;
      const wu = su * C.du * Math.sqrt(1 + Math.min(gu * gu, 8));
      const wv = sv * C.dv * Math.sqrt(1 + Math.min(gv * gv, 8));
      return Math.max(wu, wv);
    }

    // bind slot `slot` to continuous chart coordinate (fu,fv) — the UV analogue of adPlace
    function uvPlace(slot, c, fu, fv, su, sv) {
      const C = ATLAS.charts[c], o3 = slot * 3;
      const p = uvPosAt(C, fu, fv, uvTmp);
      const k = uvAttrK < 0 ? 0 : uvAttrK, k3 = k * 3;
      AD.usz[slot] = uvFoot(C, fu, fv, su, sv);
      AD.tgt[o3] = p[0]; AD.tgt[o3 + 1] = p[1]; AD.tgt[o3 + 2] = p[2];
      AD.tnr[o3] = C.nrm[k3]; AD.tnr[o3 + 1] = C.nrm[k3 + 1]; AD.tnr[o3 + 2] = C.nrm[k3 + 2];
      const jt = 0.94 + AD.seed[o3] * 0.12;
      AD.tcl[o3] = C.col[k3] * jt; AD.tcl[o3 + 1] = C.col[k3 + 1] * jt; AD.tcl[o3 + 2] = C.col[k3 + 2] * jt;
      // tangent along u, so the shader's anisotropy lines up with the chart axes
      AD.tan[o3] = C.au === 0 ? 1 : 0; AD.tan[o3 + 1] = C.au === 1 ? 1 : 0; AD.tan[o3 + 2] = C.au === 2 ? 1 : 0;
      AD.curv[slot] = C.cv[k];
      AD.q[slot] = (c << 24) | k;                        // provenance, for the mode switch
    }

    const UV_TOL = 0.03;                                  // land within 3% of budget
    /* Solve a monotone-decreasing count(x) for the budget, warm-started from last frame's
       answer. Chhugani §2.4 makes the same move — between frames the required detail barely
       moves, so bracket around the previous solution and only widen if the view really
       jumped. Cold, this costs ~18 domain walks per refresh; warm, typically 6. */
    function uvSolve(prev, lim, count, M) {
      let lo, hi;
      const g0 = count(prev);
      if (g0 > M) { lo = prev; hi = prev * 2; let k = 0; while (count(hi) > M && k++ < 12) { lo = hi; hi *= 2; if (hi > lim) { hi = lim; break; } } }
      else { hi = prev; lo = prev * 0.5; let k = 0; while (count(lo) <= M && k++ < 12) { hi = lo; lo *= 0.5; if (lo < 1e-4) break; } }
      for (let it = 0; it < 12; it++) {
        const mid = Math.sqrt(lo * hi);                   // geometric: the range spans decades
        const got = count(mid);
        if (got > M) lo = mid; else hi = mid;
        if (got <= M && got >= M * (1 - UV_TOL)) return mid;
      }
      return hi;                                          // the side that does not overrun
    }

    function uvUniformSample(M) {
      if (!ATLAS.built) { AD.n = 0; return; }
      // the knob is now target screen spacing in PIXELS
      const lim = 400;
      const s = clamp(uvSolve(clamp(AD.step || 4, 0.05, lim), lim, x => uvCountAt(x, M, null), M), 0.05, lim);
      const got = uvCountAt(s, M, null);
      let w = 0;
      uvCountAt(s, M, (slot, c, fu, fv, su, sv) => { uvPlace(slot, c, fu, fv, su, sv); w++; });
      for (let i = 0; i < w; i++) { AD.pend[i] = -1; AD.fade[i] = 1; }
      AD.n = w; AD.nIn = w;                              // every emitted sample is visible
      AD.step = s; AD.fill = M ? w / M : 0;
      AD.dirty = true;
    }

    /* ══ MODE 2 — adaptive (u,v) ═════════════════════════════════════════════════════
       Chhugani & Kumar's structure, adapted from Bézier patches to the atlas charts.

       Pre-pass (§2.1-2.2, once per capture): per chart, a pyramid of blocks. A block's
       DEVIATION is the largest distance between the surface and the bilinear approximation
       taken from that block's four corners — the same quantity the paper drives insertion
       order with, and the same flatness test GPU Gems 2 ch.7 applies per edge. Deviation is
       monotone under refinement, so a tolerance selects a cut through the pyramid exactly
       the way a tolerance selects a prefix of the paper's ordered list.

       Render (§2.4): the object-space tolerance at a block is d_o = s(B)·δs, with s(B) the
       object-space length that projects to one pixel at that depth. Descend while
       dev > d_o; emit where it is flat enough. δs is then solved so the emitted count hits
       the budget — the paper fixes δs and lets the count fall out; we invert it, because
       here the budget is the fixed quantity. The δs that results is reported in stats as
       the screen-space deviation the budget can afford.

       Departures from the paper, stated plainly: the pre-pass is a quadtree over the chart
       grid rather than incremental Delaunay with re-evaluated deviations after each
       insertion, and there is no running triangulation between frames — splats need sample
       positions, not a mesh, so the Delaunay machinery that dominates the paper's runtime
       has nothing to do here. §2.6's crack rule is met differently too: the charts share one
       world bbox and sample seam cells at a common step, so neighbours agree on the seam. */
    const UV_LMAX = Math.round(Math.log2(256));            // pyramid depth = log2(chart size)
    function atlasPyramid() {
      for (const C of ATLAS.charts) {
        C.mip = []; C.rep = [];
        for (let l = 0; l <= UV_LMAX; l++) {
          const w = Math.max(1, C.NU >> l), h = Math.max(1, C.NV >> l);
          const dev = new Float32Array(w * h), rep = new Int32Array(w * h).fill(-1);
          const span = 1 << l;
          for (let by = 0; by < h; by++) {
            for (let bx = 0; bx < w; bx++) {
              const i0 = bx * span, j0 = by * span;
              const i1 = Math.min(C.NU - 1, i0 + span - 1), j1 = Math.min(C.NV - 1, j0 + span - 1);
              // corner heights of the block, for the bilinear reference
              const h00 = C.H[j0 * C.NU + i0], h10 = C.H[j0 * C.NU + i1];
              const h01 = C.H[j1 * C.NU + i0], h11 = C.H[j1 * C.NU + i1];
              const ok = C.mask[j0 * C.NU + i0] && C.mask[j0 * C.NU + i1]
                && C.mask[j1 * C.NU + i0] && C.mask[j1 * C.NU + i1];
              let mx = 0, r = -1, cx = (i0 + i1) >> 1, cy = (j0 + j1) >> 1;
              if (C.mask[cy * C.NU + cx]) r = cy * C.NU + cx;
              for (let j = j0; j <= j1; j++) {
                const fy = j1 > j0 ? (j - j0) / (j1 - j0) : 0;
                for (let i = i0; i <= i1; i++) {
                  const k = j * C.NU + i;
                  if (!C.mask[k]) continue;
                  if (r < 0) r = k;
                  if (!ok) { mx = Infinity; continue; }
                  const fx = i1 > i0 ? (i - i0) / (i1 - i0) : 0;
                  const bl = h00 * (1 - fx) * (1 - fy) + h10 * fx * (1 - fy)
                    + h01 * (1 - fx) * fy + h11 * fx * fy;
                  const e = Math.abs(C.H[k] - bl);
                  if (e > mx) mx = e;
                }
              }
              // a block straddling a depth discontinuity is never "flat"
              dev[by * w + bx] = (r < 0) ? -1 : (mx === Infinity ? 1e9 : mx);
              rep[by * w + bx] = r;
            }
          }
          C.mip.push(dev); C.rep.push(rep);
        }
        /* |F_uu| and |F_vv| per block, max-reduced up the pyramid — §2.5 needs them to set
           the u and v step sizes independently inside a block that is flat enough to stop on */
        C.mipU = []; C.mipV = [];
        const w0 = C.NU, h0 = C.NV;
        const u0a = new Float32Array(w0 * h0), v0a = new Float32Array(w0 * h0);
        for (let j = 0; j < h0; j++) for (let i = 0; i < w0; i++) {
          const k = j * w0 + i;
          if (!C.mask[k]) continue;
          const im = Math.max(0, i - 1), ip = Math.min(w0 - 1, i + 1);
          const jm = Math.max(0, j - 1), jp = Math.min(h0 - 1, j + 1);
          const ku = j * w0 + im, kU = j * w0 + ip, kv = jm * w0 + i, kV = jp * w0 + i;
          u0a[k] = (C.mask[ku] && C.mask[kU]) ? Math.abs(C.H[ku] - 2 * C.H[k] + C.H[kU]) : 0;
          v0a[k] = (C.mask[kv] && C.mask[kV]) ? Math.abs(C.H[kv] - 2 * C.H[k] + C.H[kV]) : 0;
        }
        C.mipU.push(u0a); C.mipV.push(v0a);
        for (let l = 1; l <= UV_LMAX; l++) {
          const pw = Math.max(1, C.NU >> (l - 1)), ph = Math.max(1, C.NV >> (l - 1));
          const w = Math.max(1, C.NU >> l), h = Math.max(1, C.NV >> l);
          const pu = C.mipU[l - 1], pv = C.mipV[l - 1];
          const cu = new Float32Array(w * h), cvv = new Float32Array(w * h);
          for (let by = 0; by < h; by++) for (let bx = 0; bx < w; bx++) {
            let mu = 0, mv = 0;
            for (let q = 0; q < 4; q++) {
              const nx = bx * 2 + (q & 1), ny = by * 2 + (q >> 1);
              if (nx >= pw || ny >= ph) continue;
              const pi = ny * pw + nx;
              if (pu[pi] > mu) mu = pu[pi];
              if (pv[pi] > mv) mv = pv[pi];
            }
            cu[by * w + bx] = mu; cvv[by * w + bx] = mv;
          }
          C.mipU.push(cu); C.mipV.push(cvv);
        }
      }
    }

    // object-space length projecting to one screen pixel at this depth — the paper's s(B)
    function uvScaleFactor(depth) {
      return Math.max(depth, 0.05) * 2 * Math.max(view.tanHalf, 1e-3) / Math.max(H, 2);
    }

    // descend the pyramid at screen tolerance `ds` (pixels); emit flat-enough blocks
    function uvAdaptCount(ds, M, emit) {
      let cnt = 0;
      const stack = ATLAS._stack || (ATLAS._stack = []);
      for (let c = 0; c < 6; c++) {
        const C = ATLAS.charts[c];
        if (!C.occ) continue;
        stack.length = 0;
        stack.push(UV_LMAX, 0, 0);
        while (stack.length) {
          const bx = stack.pop(), by = stack.pop(), l = stack.pop();
          const w = Math.max(1, C.NU >> l);
          const idx = by * w + bx;
          const rep = C.rep[l][idx];
          if (rep < 0) continue;                          // nothing of the surface here
          const ri = rep % C.NU, rj = (rep / C.NU) | 0;
          const p = uvPos(C, ri, rj, uvTmp);
          const vx = p[0] - view.pos.x, vy = p[1] - view.pos.y, vz = p[2] - view.pos.z;
          const depth = vx * view.dir.x + vy * view.dir.y + vz * view.dir.z;
          const dev = C.mip[l][idx];
          const tol = uvScaleFactor(depth) * ds;          // d_o = s(B)·δs
          if (l > 0 && dev > tol) {                       // not flat enough: subdivide
            const l2 = l - 1, w2 = Math.max(1, C.NU >> l2), h2 = Math.max(1, C.NV >> l2);
            for (let q = 0; q < 4; q++) {
              const nx = bx * 2 + (q & 1), ny = by * 2 + (q >> 1);
              if (nx < w2 && ny < h2) stack.push(l2, ny, nx);
            }
            continue;
          }
          /* §2.5 — the block is flat enough to stop descending, but a splat set still has
             to COVER it, so the block is filled by uniform sampling at the step the paper
             gives: independent u and v steps from that block's own |F_uu| / |F_vv|, scaled
             by the square root of the deviation we are allowed. Flat ⇒ long step ⇒ few
             splats; curved ⇒ short step ⇒ many. This is the piece a pure flatness test
             cannot supply, because for triangles a flat region genuinely needs no samples
             and for splats it still needs coverage. */
          const span = 1 << l;
          const i0 = bx * span, j0 = by * span;
          const i1 = Math.min(C.NU - 1, i0 + span - 1), j1 = Math.min(C.NV - 1, j0 + span - 1);
          const huu = C.mipU[l][idx], hvv = C.mipV[l][idx];
          // sagitta over m cells ≈ |F_uu|·m²/8; solve m for the allowed deviation
          // sub-cell steps are allowed: the height field is bilinear, so the domain does not
          // run out of resolution when the view tightens onto a few cells
          /* density = whichever demands MORE samples: curvature (the flatness step) or
             coverage (the screen-uniform step). A flat wall needs no subdivision to be
             geometrically right but still needs splats on it to be seen. */
          uvStepAt(C, ri + 0.5, rj + 0.5, ds);
          const su = clamp(Math.min(Math.sqrt(8 * tol / Math.max(huu, 1e-9)), uvStep[0]), 0.04, span);
          const sv = clamp(Math.min(Math.sqrt(8 * tol / Math.max(hvv, 1e-9)), uvStep[1]), 0.04, span);
          for (let fj = j0 + 0.5 * sv; fj < j1 + 1; fj += sv) {
            for (let fi = i0 + 0.5 * su; fi < i1 + 1; fi += su) {
              if (!uvVisible(C, fi, fj)) continue;
              if (emit) { if (cnt < M) emit(cnt, c, fi, fj, su, sv); else return cnt + 1; }
              cnt++;
              if (!emit && cnt > M) return cnt;           // bisect only needs the comparison
            }
          }
        }
      }
      return cnt;
    }

    function uvAdaptiveSample(M) {
      if (!ATLAS.built) { AD.n = 0; return; }
      if (!ATLAS.charts[0].mip) atlasPyramid();
      /* solve the screen-space tolerance δs for the budget. The paper fixes δs and lets the
         triangle count fall out; here the budget is the fixed quantity, so it is inverted —
         and the δs that comes back is a real readout: the screen deviation this budget can
         afford, in pixels. */
      const ds = clamp(uvSolve(clamp(AD.ds || 4, 1e-3, 400), 400, x => uvAdaptCount(x, M, null), M), 1e-3, 400);
      let w = 0;
      uvAdaptCount(ds, M, (slot, c, fu, fv, su, sv) => { uvPlace(slot, c, fu, fv, su, sv); w++; });
      for (let i = 0; i < w; i++) { AD.pend[i] = -1; AD.fade[i] = 1; }
      AD.n = w; AD.nIn = w;
      AD.ds = ds; AD.fill = M ? w / M : 0;
      AD.dirty = true;
    }

    /* ── local transport ───────────────────────────────────────────────────────────
       The reallocation is applied as a REPAIR, not a redraw. Bin the ordering into runs,
       compare how many slots each run holds against how many the (eased) weights ask it to
       hold, and move only the surplus — each one into the NEAREST run that is short. A slot
       sitting in a run that is already balanced is never written to, so it does not move at
       all; that is what turns "everything shimmers whenever the camera twitches" into "the
       splats that are no longer paying their way walk over to where the budget went".
       Deadband ±0.5 keeps integer rounding from churning slots back and forth forever. */
    const AD_MOVE_FRAC = 0.06;                   // ≤6% of the budget changes bin per refresh
    function adTransport(M) {
      const n = AD.count;
      if (!n || !AD.n) { adSample(M); return; }
      const ord = AD.ord, w = AD.ws, P = TRI.p;
      const binW = AD.binW, want = AD.want, have = AD.have, need = AD.need;
      const bHead = AD.bHead, bNext = AD.bNext, sb = AD.sb, cum = AD.bCum;
      /* ~8 slots per bin. Coarse enough that the ±0.5 deadband is meaningful (bins hold whole
         slots, not fractions of one) and fine enough that a redraw inside a bin is a short
         hop: one bin is 1/NB of the surface, so its extent is ~sqrt(1/NB) of the diameter. */
      const NB = clamp(Math.round(M / 8), 32, Math.min(AD.NBMAX, n));
      const per = n / NB;
      const bOf = q => clamp((q / per) | 0, 0, NB - 1);

      let tot = 0;
      for (let b = 0; b < NB; b++) binW[b] = 0;
      for (let q = 0; q < n; q++) { const v = w[ord[q]]; binW[bOf(q)] += v; tot += v; }
      if (!(tot > 0)) return;

      for (let b = 0; b < NB; b++) have[b] = 0;
      for (let i = 0; i < M; i++) { const b = bOf(AD.q[i]); sb[i] = b; have[b]++; }
      for (let b = 0; b < NB; b++) { want[b] = M * binW[b] / tot; need[b] = want[b] - have[b]; }

      /* ── 1. which bin: move only the surplus, each to the nearest bin that is short.
             Bins that already hold what they should are not touched at all. ── */
      let moved = 0;
      const cap = Math.max(1, Math.round(M * AD_MOVE_FRAC));
      // rebuild the per-bin slot lists so surplus bins can be drained
      for (let b = 0; b < NB; b++) bHead[b] = -1;
      for (let i = M - 1; i >= 0; i--) { bNext[i] = bHead[sb[i]]; bHead[sb[i]] = i; }
      for (let b = 0; b < NB && moved < cap; b++) {
        while (need[b] < -0.5 && moved < cap) {
          const i = bHead[b];
          if (i < 0) break;
          let dst = -1;
          for (let r = 1; r < NB; r++) {          // nearest along the ordering = nearest across
            const lo = b - r, hi = b + r;         // the surface, now that the ordering is Hilbert
            if (lo >= 0 && need[lo] > 0.5) { dst = lo; break; }
            if (hi < NB && need[hi] > 0.5) { dst = hi; break; }
            if (lo < 0 && hi >= NB) break;
          }
          if (dst < 0) break;
          bHead[b] = bNext[i];
          need[b] += 1; need[dst] -= 1;
          sb[i] = dst; moved++;
        }
      }

      /* ── 2. where in the bin: a stable stratified draw against the weights inside that bin.
             AD.hx[i] is fixed for the slot's life, so this is a pure function of the bin's
             weight profile — park the camera and every slot stays exactly where it is. It
             re-runs for every slot because a bin's INTERNAL profile shifts as the camera turns
             (a bin straddling the silhouette is balanced in total while its slots sit on the
             hidden half), and that intra-bin staleness is invisible to the transport above. ── */
      const snap = Math.max((gt ? gt.radius : 10) * 0.16, 1e-3), snap2 = snap * snap;
      for (let b = 0; b < NB; b++) bHead[b] = -1;
      for (let i = M - 1; i >= 0; i--) { bNext[i] = bHead[sb[i]]; bHead[sb[i]] = i; }
      let placed = 0;
      for (let b = 0; b < NB; b++) {
        let i = bHead[b];
        if (i < 0) continue;
        const q0 = (b * per) | 0, q1 = b === NB - 1 ? n : Math.min(n, ((b + 1) * per) | 0);
        let s = 0, k = 0;
        for (let q = q0; q < q1; q++) { s += w[ord[q]]; cum[k++] = s; }
        if (!(s > 0) || !k) continue;
        for (; i >= 0; i = bNext[i]) {
          const u = AD.hx[i] * s;
          let a = 0, z = k - 1;
          while (a < z) { const m = (a + z) >> 1; if (cum[m] < u) a = m + 1; else z = m; }
          const nq = q0 + a;
          if (nq === AD.q[i] && AD.pend[i] < 0) continue;      // already exactly right: skip
          const o3 = i * 3, t9 = ord[nq] * 9;
          let ba = AD.hx[i], bb = AD.hy[i];
          if (ba + bb > 1) { ba = 1 - ba; bb = 1 - bb; }
          const nx = P[t9] + (P[t9 + 3] - P[t9]) * ba + (P[t9 + 6] - P[t9]) * bb;
          const ny = P[t9 + 1] + (P[t9 + 4] - P[t9 + 1]) * ba + (P[t9 + 7] - P[t9 + 1]) * bb;
          const nz = P[t9 + 2] + (P[t9 + 5] - P[t9 + 2]) * ba + (P[t9 + 8] - P[t9 + 2]) * bb;
          const dx = nx - AD.tgt[o3], dy = ny - AD.tgt[o3 + 1], dz = nz - AD.tgt[o3 + 2];
          placed++;
          /* distance decided BEFORE committing, so a splat that has to be faded never wears
             the destination's normal or colour while it is still drawn at the old spot */
          if (dx * dx + dy * dy + dz * dz > snap2) { AD.pend[i] = nq; AD.q[i] = nq; }
          else adPlace(i, nq);
        }
      }
      AD.moved = moved; AD.placed = placed;
      adMark(M);
      if (moved || placed) AD.dirty = true;
    }

    // the drawn set lives on the eased positions; the spacing field and the coverage metric
    // read the TARGETS, so the footprint policy is derived from the configuration being
    // solved for rather than from a half-completed transition
    function adBucket() {
      G.cnt.fill(0); G.head.fill(-1);
      const M = AD.n, p = AD.tgt;
      for (let i = 0; i < M; i++) {
        const c = gcellOf(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
        AD.cellOf[i] = c;
        G.next[i] = G.head[c]; G.head[c] = i; G.cnt[c]++;
      }
    }

    const AD_FADE = 7.5;                          // ≈130 ms out, 130 ms back
    function adEase(dt) {
      const M = AD.n;
      if (!M || !AD.att) return;
      const r = 1 - Math.exp(-AD_EASE * Math.max(dt, 0));
      const fstep = AD_FADE * Math.max(dt, 0);
      const snap = Math.max((gt ? gt.radius : 10) * 0.16, 1e-3), snap2 = snap * snap;
      const p = AD.pos, tg = AD.tgt, nn = AD.nrm, tn = AD.tnr, cc = AD.col, tc = AD.tcl;
      const scl = AD.fade, pend = AD.pend;
      let live = false;
      for (let i = 0; i < M; i++) {
        const o = i * 3;
        if (pend[i] >= 0) {
          /* relocating: shrink to nothing, make the jump while there is nothing to see, and
             let the grow-back below carry it in. No blending needed — the splats are opaque
             and this rides the per-instance size the shader already multiplies in. */
          scl[i] -= fstep;
          live = true;
          if (scl[i] <= 0) {
            scl[i] = 0;
            adPlace(i, pend[i]); pend[i] = -1;
            p[o] = tg[o]; p[o + 1] = tg[o + 1]; p[o + 2] = tg[o + 2];
            nn[o] = tn[o]; nn[o + 1] = tn[o + 1]; nn[o + 2] = tn[o + 2];
            cc[o] = tc[o]; cc[o + 1] = tc[o + 1]; cc[o + 2] = tc[o + 2];
          }
          continue;
        }
        if (scl[i] < 1) { scl[i] = Math.min(1, scl[i] + fstep); live = true; }
        const ex = tg[o] - p[o], ey = tg[o + 1] - p[o + 1], ez = tg[o + 2] - p[o + 2];
        const d2 = ex * ex + ey * ey + ez * ez;
        if (d2 > snap2) {                          // backstop: should be rare now that long
          scl[i] = 0; pend[i] = AD.q[i];           // relocations are routed through the fade
          live = true;
          continue;
        }
        if (d2 > 1e-10) live = true;
        p[o] += ex * r; p[o + 1] += ey * r; p[o + 2] += ez * r;
        nn[o] += (tn[o] - nn[o]) * r; nn[o + 1] += (tn[o + 1] - nn[o + 1]) * r; nn[o + 2] += (tn[o + 2] - nn[o + 2]) * r;
        cc[o] += (tc[o] - cc[o]) * r; cc[o + 1] += (tc[o + 1] - cc[o + 1]) * r; cc[o + 2] += (tc[o + 2] - cc[o + 2]) * r;
      }
      if (!live && !AD.dirty) return;
      AD.dirty = live;
      const cnt = M * 3, A = AD.att;
      const upd = (att, items) => {
        if (!att) return;
        att.needsUpdate = true;
        if (att.updateRange) { att.updateRange.offset = 0; att.updateRange.count = items; }
      };
      upd(A.pos, cnt); upd(A.nrm, cnt); upd(A.col, cnt); upd(A.tan, cnt);
      upd(A.curv, M); upd(A.fade, M); upd(A.usz, M);
    }

    // pool test point vs the adaptive z-buffer — used to keep hidden surface out of the metric
    function adHidden(k) {
      if (!adOn()) return false;
      const P = sampler.aPos;
      const vx = P[k * 3] - view.pos.x, vy = P[k * 3 + 1] - view.pos.y, vz = P[k * 3 + 2] - view.pos.z;
      const depth = vx * view.dir.x + vy * view.dir.y + vz * view.dir.z;
      if (depth < 0.12) return true;
      const e = camera.matrixWorld.elements;
      const th = Math.max(view.tanHalf, 1e-3), asp = clamp(camera.aspect || 1.6, 0.4, 4);
      const sx = (vx * e[0] + vy * e[1] + vz * e[2]) / (th * asp * depth);
      const sy = (vx * e[4] + vy * e[5] + vz * e[6]) / (th * depth);
      if (Math.abs(sx) > 1 || Math.abs(sy) > 1) return true;
      const c = clamp(((sx * 0.5 + 0.5) * OGW) | 0, 0, OGW - 1) + clamp(((0.5 - sy * 0.5) * OGH) | 0, 0, OGH - 1) * OGW;
      const R = gt ? gt.radius : 10;
      return depth > AD.occ[c] + Math.max(R * 0.05, depth * 0.025);
    }

    function rebuildOurs() {
      if (meshOurs) { scene.remove(meshOurs); meshOurs.geometry.dispose(); meshOurs.material.dispose(); meshOurs = null; }
      const on = adOn();
      AD.warm = 0; AD.dirty = true;
      meshOurs = on ? mkSplats(ADSRC, Math.min(st.budget, ADCAP)) : mkSplats(sampler, sampler.N);
      const g = meshOurs.geometry;
      AD.att = on ? {
        pos: g.getAttribute('aPos'), nrm: g.getAttribute('aNrm'), col: g.getAttribute('aCol'),
        tan: g.getAttribute('aTan'), curv: g.getAttribute('aCurv'),
        fade: g.getAttribute('aFade'), usz: g.getAttribute('aScale')
      } : null;
      scene.add(meshOurs);
    }

    function applyGT() {
      if (!gt) return;
      gt.group.traverse(o => {
        if (!o.isMesh) return;
        o.visible = st.showGT && st.gtOp > 0.004;
        o.material.opacity = st.gtOp;
        o.material.depthWrite = false;
      });
    }

    /* budget → two thresholds, solved against the ACTUAL predicate the shader tests:
       birth ≤ tOut + (tIn-tOut)·w. Independent birth quantiles over the w>0.5 and w≤0.5
       populations do NOT work — they assume each side sees a flat threshold, but everything in
       the soft frustum-edge band gets a mixed one, so the outside overshoots and the inside
       undershoots (measured +9% over budget mid-zoom, and the reserve drifting 8%→15%).
       Instead: alternating bisection on the pair, evaluating the true rule over the subsample,
       warm-started from last frame — tOut holds the periphery at a FIXED RESERVE share of the
       budget and tIn conserves the total, so the count stays put while the ROI crops. */
    const SUB = 1400, RESERVE = 0.08, BISECT = 14;
    /* The bisection hits its target exactly IN THE SUBSAMPLE, but 1400 samples predict a ~4%
       quantile from ~56 hits (±13% Poisson), and because subIdx is fixed that error is a
       stable bias per configuration — which shifts with zoom as w changes, so the count drifts
       with the camera. bucket() already counts the full pool exactly, so close the loop:
       damped feedback on the solve targets, locking the drawn count to the budget and the
       peripheral share to RESERVE within a few frames. Reset on hard refreshes. */
    const FB = { all: 1, out: 1, base: 1 };
    const fbStep = (cur, want, got) => clamp(cur * Math.pow(want / Math.max(got, 1), 0.5), 0.4, 2.5);
    const FB_MAX = 3;
    /* The attainable counts are QUANTIZED: the bisection targets a count in subsample units, so
       one subsample hit ≈ (N/SUB) primitives. Chasing a tolerance finer than that quantum cannot
       terminate — at some operating points the two nearest attainable counts both sit outside the
       band, and the pair of multipliers 2-cycles forever (measured 5% peak-to-peak on a parked
       camera). So the deadband is derived from the resolution, and FB is frozen inside it rather
       than corrected again. The reserve's own target is only ~4 hits, hence its much wider band. */
    const fbTol = (want, per) => Math.max(0.02, 1.6 * per / Math.max(want, 1));
    /* Each convergence iteration is a full-pool bucket pass, so iterating during continuous
       camera motion hitched the frame (p90 35 ms, 115 ms spikes). While refreshes are
       back-to-back the loop runs once — the feedback still converges frame-to-frame — and the
       frame loop fires one trailing refresh ~180 ms after motion stops to iterate properly.
       The ledger is therefore exact whenever it is being read. */
    let lastRefreshAt = -1e9, needConverge = false;
    const subIdx = (() => { const a = new Uint32Array(SUB); const R = rng(77); for (let i = 0; i < SUB; i++) a[i] = (R() * POOL) | 0; return a; })();
    const subB = new Float32Array(SUB), subW = new Float32Array(SUB), subMem = new Uint8Array(SUB);
    let subM = 0, warmIn = 0.3, warmOut = 0.02;
    const view = { pos: new T.Vector3(0, 0, 40), dir: new T.Vector3(0, 0, -1), tanHalf: 0.38, dist: 40 };
    function smoothstepJS(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
    /* TWO separate quantities, and conflating them is a bug:
       · mem = frame MEMBERSHIP (coneS·depthS·faceS) — is this patch on screen. Gates the
         eligible/in-ROI counts, the spacing field and the reserve.
       · w   = mem · pri, the allocation PRIORITY — how much of the image the patch occupies
         (cos(incidence)/depth^2t). Grades the threshold mix only.
       Folding pri into the membership gate classified in-frame surface as "outside" (pri alone
       falls under 0.5 past ~1.41× subject distance or ~60° incidence), so the reserve drove the
       wrong quantity and 47% of the budget read as peripheral instead of 8%. */
    const VW = { mem: 0, w: 0 };
    function viewOf(k) {
      const dx = sampler.aPos[k * 3] - view.pos.x, dy = sampler.aPos[k * 3 + 1] - view.pos.y, dz = sampler.aPos[k * 3 + 2] - view.pos.z;
      const depth = dx * view.dir.x + dy * view.dir.y + dz * view.dir.z;
      const edge = view.tanHalf * Math.max(depth, 0.2) * 1.06;
      const px = dx - view.dir.x * depth, py = dy - view.dir.y * depth, pz = dz - view.dir.z * depth;
      const perp = Math.sqrt(px * px + py * py + pz * pz);
      const coneS = 1 - smoothstepJS(edge * 0.85, edge * 1.15, perp);
      const depthS = smoothstepJS(0.05, 0.35, depth);
      const fd = -(view.dir.x * sampler.aNrm[k * 3] + view.dir.y * sampler.aNrm[k * 3 + 1] + view.dir.z * sampler.aNrm[k * 3 + 2]);
      const faceS = smoothstepJS(-0.26, -0.04, fd);
      const rel = Math.max(depth, 0.2) / Math.max(view.dist, 0.2);
      const r2 = rel * rel, t = st.roiTight;
      const den = t > 1.75 ? r2 * r2 : t > 1.25 ? r2 * rel : r2;
      VW.mem = coneS * depthS * faceS;
      VW.w = VW.mem * clamp(Math.max(fd, 0.05) / den, 0.04, 1);
      return VW;
    }
    function inViewWeight(k) { return viewOf(k).w; }
    function inViewCPU(k) { return inViewWeight(k) > 0.5; }
    function quantile(sorted, frac) {
      if (!sorted.length) return 0;
      if (frac >= 1) return 1.0001;                 // whole population fits — nothing culled
      return sorted[clamp(Math.round(frac * (sorted.length - 1)), 0, sorted.length - 1)] + 1e-5;
    }
    // alive count over the subsample under the real mixed rule; outOnly restricts to the
    // out-of-FRAME set (membership, not priority — the reserve is about what is off screen)
    function subAlive(u, v, outOnly) {
      let c = 0;
      for (let i = 0; i < subM; i++) {
        if (outOnly && subMem[i]) continue;
        if (subB[i] <= u + (v - u) * subW[i]) c++;
      }
      return c;
    }
    // f is monotone increasing in x; return the x that lands closest to target
    function bisect(f, target) {
      let lo = 0, hi = 1.0001;
      for (let i = 0; i < BISECT; i++) {
        const mid = 0.5 * (lo + hi);
        if (f(mid) < target) lo = mid; else hi = mid;
      }
      return 0.5 * (lo + hi);
    }
    function solveThresh(birthArr, want, viewDriven, hole, cAll, cOut) {
      const n = Math.min(want, sampler.N);
      let mIn = 0;
      subM = 0;
      for (let i = 0; i < SUB; i++) {
        const k = subIdx[i] % sampler.N;
        if (voided(k, hole)) continue;              // voided points can never be drawn
        subB[subM] = birthArr[k] * (viewDriven ? kfOf(k) : 1);
        if (viewDriven) { const v = viewOf(k); subW[subM] = v.w; if (v.mem > 0.5) { subMem[subM] = 1; mIn++; } else subMem[subM] = 0; }
        else { subW[subM] = 1; subMem[subM] = 1; mIn++; }
        subM++;
      }
      const perSub = sampler.N / SUB;
      const roiPop = mIn * perSub;                  // in-frame population available to spend on
      const tgtAll = n / (sampler.N / SUB) * (cAll || 1);   // budget in subsample units
      if (!viewDriven) {                            // one flat threshold — the quantile is exact
        const s = Array.prototype.slice.call(subB.subarray(0, subM)).sort((a, b) => a - b);
        const t = quantile(s, subM ? tgtAll / subM : 1);
        return { threshIn: t, threshOut: t, inFocusFrac: viewFraction(), roiPop: roiPop, maxReach: subM * perSub };
      }
      let u = warmOut, v = warmIn;
      for (let round = 0; round < 2; round++) {
        u = bisect(x => subAlive(x, v, true), tgtAll * RESERVE * (cOut || 1));
        v = bisect(x => subAlive(u, x, false), tgtAll);
      }
      warmOut = u; warmIn = v;
      /* capacity, not population: with the graded threshold a point at w=0.5 only passes at half
         of tIn, so the most this configuration can ever draw is the weighted count at tIn=1 —
         testing the in-frame population instead missed a real saturation (41k drawn of a 56k
         budget, reported as if the budget were met). */
      const maxReach = subAlive(u, 1.0001, false) * perSub;
      return { threshIn: v, threshOut: u, inFocusFrac: viewFraction(), roiPop: roiPop, maxReach: maxReach };
    }
    // the FRAME fraction (membership), which is the area the spacing reference is taken over
    function viewFraction() {
      let c = 0;
      for (let i = 0; i < SUB; i++) c += viewOf(subIdx[i] % sampler.N).mem;
      return c / SUB;
    }

    let stats = { };
    /* coverage/overlap is a ~3M-cell-visit scan per panel — far too heavy to run twice per
       frame at pool scale, and it does not change meaningfully between adjacent frames.
       Recomputed on hard refreshes (dataset/baseline/budget change) and otherwise at ~6 Hz;
       the spacing field and uniforms still update every frame, so the render is unaffected. */
    const COV_MS = 160;
    let covCache = null, covAt = -1e9;
    /* chamfer/L1 rides its own, slower cadence: two ~1-2 ms scans per panel, cheap next to
       the coverage pass but pointless to re-run between adjacent frames for the same reason. */
    const ERR_MS = 320;
    let errCache = null, errAt = -1e9;
    const TGT = { threshO: 0.3, outO: 0.01, footO: 0.2, convO: 0, threshB: 0.3, footB: 0.2, convB: 0 };
    const DISP = { threshO: 0.3, outO: 0.01, footO: 0.2, convO: 0, threshB: 0.3, footB: 0.2, convB: 0 };
    function refresh(hard) {
      if (!sampler || !gt) return;
      KNOT.w = clamp(st.knots || 0, 0, 1);
      const D = DATASETS[st.dataset];
      const M = MODELS.ours, B = MODELS[st.baseline];
      const s = st.step;
      const mag = (1 / (1 - 0.30 * st.zoom)) * (Math.tan(21 * Math.PI / 180) / Math.tan(21 * (1 - 0.56 * st.zoom) * Math.PI / 180));

      const now = performance.now();
      const moving = !hard && now - lastRefreshAt < 130;
      lastRefreshAt = now;
      const maxIt = hard ? FB_MAX : (moving ? 1 : 2);
      const remeasure = hard || !covCache || now - covAt > COV_MS;
      const remeasureErr = hard || !errCache || now - errAt > ERR_MS;

      // ── ours: fixed budget, reallocated into the view cone, footprint from neighbours ──
      /* iterate solve→count to tolerance WITHIN this refresh: refresh() is event-driven, not
         per-frame, so a single damped step per call never converges while the camera is
         parked. Two passes is the normal case; it exits as soon as the drawn count is within
         FB_TOL of the budget. */
      const ADON = adOn();
      const oursN = Math.min(st.budget, ADON ? ADCAP : sampler.N);
      const perHit = sampler.N / SUB;
      const tolAll = fbTol(oursN, perHit), tolOut = fbTol(oursN * RESERVE, perHit);
      let so, aliveO, areaC = null;
      if (ADON) {
        /* no threshold to solve: the budget IS the sample count. Weights → CDF → positions,
           and every slot is drawn, so the ledger is exact by construction instead of being
           chased by feedback. */
        adWeights(hard || !AD.warm);
        const uvMode = st.sampler === 'uvUniform' || st.sampler === 'uvAdaptive';
        if (uvMode) {
          // the UV samplers place absolutely from the atlas; visibility IS the allocation
          if (st.sampler === 'uvUniform') uvUniformSample(oursN); else uvAdaptiveSample(oursN);
        } else if (hard || !AD.warm || oursN !== AD.n) {
          // absolute placement only to seed the cloud (or when the budget itself changed);
          // every other refresh repairs the allocation in place
          adSample(oursN);
        } else adTransport(oursN);
        if (hard || !AD.warm) {
          const c3 = AD.n * 3;
          AD.pos.set(AD.tgt.subarray(0, c3)); AD.nrm.set(AD.tnr.subarray(0, c3)); AD.col.set(AD.tcl.subarray(0, c3));
          AD.warm = 1; AD.dirty = true;
        }
        adBucket();
        areaC = G.areaC;
        if (meshOurs) meshOurs.geometry.instanceCount = AD.n;
        so = { threshIn: 2, threshOut: 2, inFocusFrac: AD.uvVis, roiPop: AD.visArea, maxReach: oursN };
        aliveO = { n: AD.n, nIn: AD.nIn };
        needConverge = false;
      } else {
        for (let it = 0; it < maxIt; it++) {
          so = solveThresh(sampler.aBirth, oursN, true, 0, FB.all, FB.out);
          aliveO = bucket(sampler.aBirth, so.threshIn, so.threshOut, true, 0);
          const eAll = aliveO.n / oursN - 1;
          const eOut = (aliveO.n - aliveO.nIn) / (oursN * RESERVE) - 1;
          if (Math.abs(eAll) <= tolAll) { needConverge = false; break; }   // inside resolution: freeze
          const pAll = FB.all;
          FB.all = fbStep(FB.all, oursN, aliveO.n);
          if (Math.abs(eOut) > tolOut) FB.out = fbStep(FB.out, oursN * RESERVE, aliveO.n - aliveO.nIn);
          // backstop: if the correction has stopped moving, further passes cannot help
          needConverge = Math.abs(FB.all / pAll - 1) >= 0.004;
          if (!needConverge) break;
        }
      }
      /* spacing reference comes from the population actually inside the ROI over the ROI's own
         area — so zooming in (smaller focusArea, same in-ROI count) shrinks the footprint,
         which is the whole point of the crop. Under adaptive resampling that area is MEASURED
         (sum of area·visibility over the domain), not inferred from a subsample fraction. */
      const focusArea = ADON ? AD.visArea : sampler.area * Math.max(so.inFocusFrac, 0.015);
      const footOurs = Math.sqrt(focusArea / Math.max(aliveO.nIn, 1));
      const convO = 1 - Math.exp(-s / 900);
      const maxH = spacingField(footOurs, areaC);
      const covO = remeasure
        ? measureCoverage(true, footOurs * 1.55, maxH, dispOf(M, footOurs * 1.55, convO, true), ADON ? ADCOV : null)
        : covCache.o;
      /* grid lists still hold OUR alive set here — measureError must run before the
         baseline re-buckets below */
      const errO = remeasureErr
        ? measureError(M, dispOf(M, footOurs * 1.55, convO, true), ADON ? ADCOV : null, aliveO.n)
        : (errCache && errCache.o);
      if (ADON) {
        AD.conc = (AD.nIn / AD.visArea) / Math.max(oursN / sampler.area, 1e-9);
        /* reported footprint = the primitive-weighted mean of the LOCAL spacing the shader
           actually samples. sqrt(visibleArea / count) is the wrong number to quote here: the
           frustum's intersection with a ground plane runs off to the horizon, so "visible area"
           stays large while the budget — priced by projected area — sits almost entirely on the
           framed patch. This is the disk width on screen. */
        let hs = 0, hn = 0;
        for (let c = 0; c < GCELLS; c++) {
          if (G.cnt[c] > 0 && G.areaC[c] > 1e-9) { hs += G.spc2[c] * G.cnt[c]; hn += G.cnt[c]; }
        }
        AD.hMean = hn ? hs / hn : footOurs;
      }
      TGT.threshO = so.threshIn; TGT.outO = so.threshOut; TGT.footO = footOurs * 1.55; TGT.convO = convO;

      // ── baseline: densified, no view awareness, one global footprint ──
      // Same closure rule as ours (width = COVER_K·h) so the two panels are on comparable
      // footprint policies. The gap has to come from the representation — iid/clumped
      // ordering (Poisson prefixes void e⁻² ≈ 13.5% before any scatter) plus each model's
      // own float/jitter — not from handing the baseline a sub-closure radius. The legacy
      // 1.62 put it at 0.44·h, below the 0.577·h hex threshold, forcing holes arithmetically.
      // cap at the DRAWABLE population, not the pool: a model with a void field can never
      // draw the voided fraction, so counting it would overstate what is on screen.
      const baseDrawable = Math.floor(sampler.N * (1 - B.shader.hole));
      const baseWant = prims(B, s, st.budget);
      const baseN = Math.min(baseWant, baseDrawable);
      const sb = solveThresh(sampler.aBirthU, baseN, false, B.shader.hole, FB.base, 1);
      // MVS's aVoid field leaves it occupying only (1-hole) of the surface, so its spacing
      // must come from the area it actually covers or its disks come out oversized.
      const footBase = Math.sqrt(sampler.area * (1 - B.shader.hole) / baseN);
      const convB = 1 - Math.exp(-s / (B.key === 'mvs' ? 220 : B.tau * 0.55));
      let aliveB = bucket(sampler.aBirthU, sb.threshIn, sb.threshOut, false, B.shader.hole);
      const tolB = fbTol(baseN, perHit);
      if (Math.abs(aliveB.n / baseN - 1) > tolB) {
        FB.base = fbStep(FB.base, baseN, aliveB.n);
        if (!moving) {
          const sb2 = solveThresh(sampler.aBirthU, baseN, false, B.shader.hole, FB.base, 1);
          aliveB = bucket(sampler.aBirthU, sb2.threshIn, sb2.threshOut, false, B.shader.hole);
          sb.threshIn = sb2.threshIn; sb.threshOut = sb2.threshOut;
          if (Math.abs(aliveB.n / baseN - 1) > tolB) { FB.base = fbStep(FB.base, baseN, aliveB.n); needConverge = true; }
        }
      }
      const covB = remeasure
        ? measureCoverage(false, footBase * COVER_K, 0, dispOf(B, footBase * COVER_K, convB, false))
        : covCache.b;
      const errB = remeasureErr
        ? measureError(B, dispOf(B, footBase * COVER_K, convB, false), null, aliveB.n)
        : (errCache && errCache.b);
      if (remeasure) { covCache = { o: covO, b: covB }; covAt = now; }
      if (remeasureErr) { errCache = { o: errO, b: errB }; errAt = now; }
      TGT.threshB = sb.threshIn; TGT.footB = footBase * COVER_K; TGT.convB = convB;

      if (hard) {
        DISP.threshO = TGT.threshO; DISP.outO = TGT.outO; DISP.footO = TGT.footO; DISP.convO = TGT.convO;
        DISP.threshB = TGT.threshB; DISP.footB = TGT.footB; DISP.convB = TGT.convB;
        G.spcDisp.set(G.spc2);
        settleAndPush(0);
      }

      const mpu = D.mpu;
      if (st.render !== 'splats') selectTris();
      stats = {
        version: VERSION, step: s, maxStep: st.maxStep, dataset: D, zoom: st.zoom, mag: mag,
        focusPct: Math.round(so.inFocusFrac * 100),
        ours: {
          model: M, n: aliveO.n, params: params(M, aliveO.n),
          /* measured against the GT pool when a sample has landed (world→mm via mpu);
             the analytic curves are only a first-frames fallback */
          res: errO ? errO.res : curve(M, s), cham: errO ? errO.cham * mpu * 1000 : chamfer(M, s), meas: !!errO,
          foot: (ADON ? AD.hMean : footOurs) * mpu, hitTarget: s >= M.target, target: M.target,
          roiPct: Math.round(aliveO.nIn / Math.max(aliveO.n, 1) * 100),
          /* adaptive resampling ledger: how much of the parameter domain is actually observed,
             how much of what is in frame is hidden behind nearer surface, and the resulting
             density gain on the observed patch against spending the budget uniformly. */
          adapt: ADON, uvVis: so.inFocusFrac, occl: ADON ? AD.occl : 0,
          conc: ADON ? AD.conc : 1, reserve: ADON ? AD.reserve : RESERVE,
          // saturated: the ROI ran out of pool points before the budget was spent (high budget
          // at high zoom crops the ROI below budget/POOL) — not just the whole-pool ceiling
          cov: covO.cov, ovl: covO.ovl, starved: covO.starved,
          // saturation read from the POPULATION, not the transient count: the in-frame pool can
          // be smaller than the budget at high zoom. Testing aliveO.n instead tripped a false
          // ceiling warning mid-sweep, when the motion path deliberately runs one iteration.
          capped: !ADON && (st.budget > sampler.N || oursN > so.maxReach * 0.98)
        },
        base: {
          model: B, n: aliveB.n, params: params(B, aliveB.n),
          res: errB ? errB.res : curve(B, s), cham: errB ? errB.cham * mpu * 1000 : chamfer(B, s), meas: !!errB,
          foot: footBase * mpu, hitTarget: B.target > 0 && s >= B.target, target: B.target,
          cov: covB.cov, ovl: covB.ovl, starved: covB.starved, capped: baseWant > baseDrawable
        },
        tri: {
          on: st.render !== 'splats', n: triStat.n, budget: Math.min(st.triBudget || 0, triCap),
          pool: triStat.pool, cand: triStat.cand, frustum: triStat.frustum, occluded: triStat.occluded,
          stride: TRI ? TRI.stride : 0, source: TRI ? TRI.total : 0
        }
      };
      if (onStats) onStats(stats);
    }

    function setUniforms(mesh, M, thresh, threshOut, foot, conv, isOurs) {
      const u = mesh.material.uniforms, S = M.shader, R = gt.radius;
      u.uThresh.value = thresh;
      u.uThreshOut.value = threshOut;
      u.uSize.value = foot;
      u.uConv.value = conv;
      u.uAlign.value = S.align;
      u.uAniso.value = S.aniso;
      u.uSizeVar.value = S.sizeVar * (0.35 + 0.65 * (1 - conv * 0.55));
      const D = dispOf(M, foot, conv, isOurs);
      u.uJitter.value = D.jitter;
      u.uFloat.value = D.float;
      u.uSmooth.value = D.smooth;
      u.uHole.value = S.hole;
      u.uSH.value = S.sh;
      u.uViewDriven.value = isOurs ? S.viewDriven : 0;
      u.uNeighbor.value = isOurs ? 1 : 0;
      u.uUvMode.value = (isOurs && (st.sampler === 'uvUniform' || st.sampler === 'uvAdaptive')) ? 1 : 0;
      u.uCoverK.value = COVER_K * (st.cover || 1);
      u.uCell.value = G.cell;
      u.uGridMin.value.copy(G.min);
      u.uSpacing.value = G.tex;
      u.uCloudR.value = R * 0.55;
      u.uCenter.value.copy(gt.center);
      u.uCamPos.value.copy(view.pos);
      u.uViewDir.value.copy(view.dir);
      u.uTanHalf.value = view.tanHalf;
      u.uRoiR.value = view.dist;
      u.uPriExp.value = isOurs ? st.roiTight : 1;
      u.uKnotC.value.copy(KNOT.c);
      u.uKnotR.value = KNOT.r;
      u.uKnotW.value = isOurs ? clamp(st.knots || 0, 0, 1) : 0;
      u.uModMix.value = clamp(st.mod || 0, 0, 1);
      u.uModGap.value = st.modGap === undefined ? 0.03 : st.modGap;
      u.uModMin.value = MODX.min;
      u.uModSpan.value = MODX.span;
      u.uModD0.value = Math.max(view.dist - R, 0.4);
      u.uModD1.value = view.dist + R;
      /* adaptive resampling owns the allocation, so the shader's rank cull is off: every slot in
         the buffer is a live primitive. Culling and re-concentration happen in the DOMAIN now,
         not in the draw call. */
      if (isOurs && adOn()) {
        u.uThresh.value = 2; u.uThreshOut.value = 2;
        u.uViewDriven.value = 0; u.uHole.value = 0; u.uKnotW.value = 0;
      }
    }

    /* every rAF frame: glide displayed thresh/footprint/convergence/spacing toward the
       latest throttled target and push uniforms + the spacing texture. Nothing the shader
       reads is ever snapped — it only eases — which is what removes the pop/flicker when
       the alive set, camera or spacing field are recomputed only a few times a second. */
    const EASE_K = 8;
    function settleAndPush(dt) {
      if (!gt || !meshOurs || !meshBase) return;
      const r = 1 - Math.exp(-EASE_K * Math.max(dt, 0));
      DISP.threshO += (TGT.threshO - DISP.threshO) * r;
      DISP.outO += (TGT.outO - DISP.outO) * r;
      DISP.footO += (TGT.footO - DISP.footO) * r;
      DISP.convO += (TGT.convO - DISP.convO) * r;
      DISP.threshB += (TGT.threshB - DISP.threshB) * r;
      DISP.footB += (TGT.footB - DISP.footB) * r;
      DISP.convB += (TGT.convB - DISP.convB) * r;
      adEase(dt);
      for (let c = 0; c < GCELLS; c++) G.spcDisp[c] += (G.spc2[c] - G.spcDisp[c]) * r;
      G.data.set(G.spcDisp);
      G.tex.needsUpdate = true;
      setUniforms(meshOurs, MODELS.ours, DISP.threshO, DISP.outO, DISP.footO, DISP.convO, true);
      setUniforms(meshBase, MODELS[st.baseline], DISP.threshB, DISP.threshB, DISP.footB, DISP.convB, false);
    }

    /* ── camera ── */
    function updateCamera() {
      const R = gt ? gt.radius : 12;
      const d = R * 2.45 * (1 - 0.30 * st.zoom) * (st.distScale || 1);
      orbit.dist += (d - orbit.dist) * 0.18;
      const fov = 42 * (1 - 0.56 * st.zoom);
      if (Math.abs(camera.fov - fov) > 0.01) { camera.fov = fov; camera.updateProjectionMatrix(); }
      const sp = Math.sin(orbit.ph), cp = Math.cos(orbit.ph);
      camera.position.set(
        orbit.target.x + orbit.dist * sp * Math.sin(orbit.th),
        orbit.target.y + orbit.dist * cp,
        orbit.target.z + orbit.dist * sp * Math.cos(orbit.th)
      );
      camera.lookAt(orbit.target);
      camera.updateMatrixWorld();
      view.pos.copy(camera.position);
      view.dir.copy(orbit.target).sub(camera.position).normalize();
      view.tanHalf = Math.tan((camera.fov * Math.PI / 180) / 2);
      view.dist = orbit.dist;
    }

    let drag = null;
    const onDown = e => { drag = { x: e.clientX, y: e.clientY }; canvas.style.cursor = 'grabbing'; canvas.setPointerCapture && canvas.setPointerCapture(e.pointerId); };
    const onMove = e => {
      if (!drag) return;
      orbit.th -= (e.clientX - drag.x) * 0.0062;
      orbit.ph = clamp(orbit.ph - (e.clientY - drag.y) * 0.0052, 0.16, 1.52);
      drag = { x: e.clientX, y: e.clientY };
    };
    const onUp = () => { drag = null; canvas.style.cursor = 'grab'; };
    const onWheel = e => {
      e.preventDefault();
      st.zoom = clamp(st.zoom + (e.deltaY > 0 ? -0.055 : 0.055), 0, 1);
      refresh();
      if (onZoom) onZoom(st.zoom);
    };
    canvas.addEventListener('pointerdown', onDown);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    canvas.addEventListener('wheel', onWheel, { passive: false });

    let W = 1, H = 1;
    function resize() {
      W = Math.max(2, container.clientWidth); H = Math.max(2, container.clientHeight);
      renderer.setSize(W, H, false);
    }
    const ro = new ResizeObserver(resize); ro.observe(container);
    resize();

    let onStats = null, onZoom = null, raf = 0, dead = false, acc = 0, last = performance.now(), hudT = 0;
    const camL = { th: -9, ph: -9, d: -9, acc: 0 };
    function frame(now) {
      if (dead) return;
      raf = requestAnimationFrame(frame);
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (st.playing && st.step < st.maxStep) {
        st.step = Math.min(st.maxStep, st.step + dt * 1400 * st.speed);
        acc += dt;
        if (acc > 0.07) { acc = 0; refresh(); }
      }
      updateCamera();
      const moved = Math.abs(orbit.th - camL.th) > 1e-4 || Math.abs(orbit.ph - camL.ph) > 1e-4 || Math.abs(orbit.dist - camL.d) > 2e-3;
      if (moved) {
        camL.th = orbit.th; camL.ph = orbit.ph; camL.d = orbit.dist;
        camL.acc += dt;
        if (camL.acc > 0.09) { camL.acc = 0; refresh(); }
      }
      settleAndPush(dt);
      // trailing convergence pass once the camera has settled — see FB notes
      if (needConverge && now - lastRefreshAt > 180) refresh();
      const half = Math.floor(W / 2);
      renderer.setScissorTest(true);
      if (st.split) {
        camera.aspect = half / H; camera.updateProjectionMatrix();
        meshOurs.visible = st.render !== 'mesh'; meshBase.visible = false;
        if (triMesh) { triMesh.visible = st.render !== 'splats'; triWire.visible = triMesh.visible && !!st.wire; }
        if (chartGrp) chartGrp.visible = st.chart;
        renderer.setViewport(0, 0, half, H); renderer.setScissor(0, 0, half, H);
        renderer.render(scene, camera);
        meshOurs.visible = false; meshBase.visible = true;
        if (triMesh) { triMesh.visible = false; triWire.visible = false; }
        if (chartGrp) chartGrp.visible = false;
        renderer.setViewport(half, 0, W - half, H); renderer.setScissor(half, 0, W - half, H);
        renderer.render(scene, camera);
        meshOurs.visible = true;
      } else {
        camera.aspect = W / H; camera.updateProjectionMatrix();
        const ours = st.solo === 'ours';
        meshOurs.visible = ours && st.render !== 'mesh'; meshBase.visible = !ours && st.render !== 'mesh';
        if (triMesh) { triMesh.visible = ours && st.render !== 'splats'; triWire.visible = triMesh.visible && !!st.wire; }
        if (chartGrp) chartGrp.visible = st.chart && ours;
        renderer.setViewport(0, 0, W, H); renderer.setScissor(0, 0, W, H);
        renderer.render(scene, camera);
        meshOurs.visible = true; meshBase.visible = true;
      }
      renderer.setScissorTest(false);
      if (now - hudT > 240 && !st.playing) hudT = now;
    }

    /* measured convergence series for the charts: sweep K checkpoints deterministically
       (seeded RNG, current camera), hard-refreshing at each so the measured metrics re-land
       at that step, then restore the live state. Synchronous — roughly a second at K=25 —
       so the page triggers it lazily, once per configuration, and caches by key. Values are
       in DISPLAY units (cham already ×mpu×1000), matching what the readouts quote. */
    let seriesCache = null;
    function measureSeries(K) {
      const pts = clamp(K | 0 || 25, 5, 91);
      const key = [st.dataset, st.baseline, st.budget, st.sampler, st.adapt, st.closure].join('|');
      if (seriesCache && seriesCache.key === key) return seriesCache.data;
      const savedStep = st.step, savedPlaying = st.playing;
      st.playing = false;
      const o = [], b = [], max = st.maxStep;
      for (let i = 0; i < pts; i++) {
        st.step = (i / (pts - 1)) * max;
        refresh(true);
        o.push({ x: st.step, prim: stats.ours.n, cham: stats.ours.cham, res: stats.ours.res });
        b.push({ x: st.step, prim: stats.base.n, cham: stats.base.cham, res: stats.base.res });
      }
      st.step = savedStep; st.playing = savedPlaying;
      refresh(true);
      seriesCache = { key: key, data: { o: o, b: b, max: max } };
      return seriesCache.data;
    }

    loadDataset(st.dataset);
    raf = requestAnimationFrame(frame);

    return {
      MODELS, DATASETS, BASELINES, curve, chamfer, prims, params,
      state: st,
      _dbg: { AD, ATLAS, view, orbit, camera, refresh, updateCamera, adWeights, adSample, adEase, get TRI() { return TRI; }, get gt() { return gt; }, get err() { return errCache; } },
      setDataset(k) { if (DATASETS[k]) loadDataset(k); },
      setBaseline(k) { if (MODELS[k]) { st.baseline = k; refresh(); } },
      setStep(s) { st.step = clamp(s, 0, st.maxStep); refresh(); },
      setPlaying(b) { st.playing = !!b; if (b && st.step >= st.maxStep) st.step = 0; refresh(); },
      setSpeed(x) { st.speed = x; },
      setBudget(n) {
        st.budget = Math.round(n);
        // the adaptive buffer is allocated per budget, so a new budget is a new instance count
        if (adOn() && meshOurs) meshOurs.geometry.instanceCount = Math.min(st.budget, ADCAP);
        refresh();
      },
      setRoiTight(t) { st.roiTight = clamp(+t || 1, 1, 2); refresh(true); },
      setSampler(k) {
        st.sampler = (k === 'uvUniform' || k === 'uvAdaptive') ? k : 'transport';
        refresh(true);
      },
      setUvOcclude(b) { st.uvOcclude = !!b; refresh(true); },
      setAdapt(b) { st.adapt = !!b; if (gt) { rebuildOurs(); refresh(true); } },
      setClosure(c) { st.closure = clamp(+c, 0, 1); refresh(true); },
      setZoom(z) { st.zoom = clamp(z, 0, 1); refresh(); },
      setSplit(b) { st.split = !!b; },
      setSolo(k) { st.solo = k; },
      setShowGT(b) { st.showGT = !!b; applyGT(); },
      setChart(b) { st.chart = !!b; },
      setCover(x) { st.cover = clamp(x, 0.7, 1.8); refresh(); },
      /* ── knot insertion, modality slices, triangle-mesh mode (v4.2) ── */
      setKnots(k) { st.knots = clamp(+k || 0, 0, 1); applyKnotLines(); refresh(); },
      setModality(m, gap) {
        st.mod = clamp(+m || 0, 0, 1);
        if (gap !== undefined) st.modGap = clamp(+gap, 0, 0.2);
      },
      setRender(k) { st.render = (k === 'mesh' || k === 'both') ? k : 'splats'; refresh(true); },
      setTriBudget(n) { st.triBudget = Math.max(200, Math.round(+n || 12000)); refresh(); },
      setWire(b) { st.wire = !!b; },
      knot() { return { center: KNOT.c.clone(), radius: KNOT.r, weight: KNOT.w }; },
      /* ── cinematic drive: programmatic camera + layer opacities (additive, v4.1) ── */
      orbit,
      setOrbit(th, ph) { orbit.th = th; if (ph !== undefined) orbit.ph = clamp(ph, 0.16, 1.54); },
      setDistScale(k) { st.distScale = clamp(k, 0.2, 4); },
      setDpr(x) { renderer.setPixelRatio(clamp(x, 0.5, 2)); resize(); },
      /* one refresh for a whole batch of timeline-driven values */
      setDrive(d) {
        if (d.zoom !== undefined) st.zoom = clamp(d.zoom, 0, 1);
        if (d.step !== undefined) st.step = clamp(d.step, 0, st.maxStep);
        if (d.budget !== undefined) st.budget = Math.round(d.budget);
        if (d.distScale !== undefined) st.distScale = clamp(d.distScale, 0.2, 4);
        if (d.gtOp !== undefined) { st.gtOp = clamp(d.gtOp, 0, 1); applyGT(); }
        if (d.knots !== undefined) { st.knots = clamp(d.knots, 0, 1); applyKnotLines(); }
        if (d.mod !== undefined) st.mod = clamp(d.mod, 0, 1);
        if (d.modGap !== undefined) st.modGap = clamp(d.modGap, 0, 0.2);
        if (d.triBudget !== undefined) st.triBudget = Math.max(200, Math.round(d.triBudget));
        if (d.adapt !== undefined && !!d.adapt !== st.adapt) { st.adapt = !!d.adapt; if (gt) rebuildOurs(); }
        if (d.render !== undefined) st.render = (d.render === 'mesh' || d.render === 'both') ? d.render : 'splats';
        if (d.chartOp !== undefined) {
          st.chartOp = clamp(d.chartOp, 0, 1);
          if (chartGrp) chartGrp.children.forEach((c, i) => {
            if (i > 1) return;
            c.material.transparent = true;
            c.material.opacity = i === 0 ? st.chartOp : Math.min(1, st.chartOp * 1.9);
          });
          applyKnotLines();
        }
        refresh();
      },
      setGTOpacity(o) { st.gtOp = clamp(o, 0, 1); applyGT(); },
      setChartOpacity(o) {
        st.chartOp = clamp(o, 0, 1);
        if (!chartGrp) return;
        chartGrp.children.forEach((c, i) => {
          if (i > 1) return;
          c.material.transparent = true;
          c.material.opacity = i === 0 ? st.chartOp : Math.min(1, st.chartOp * 1.9);
        });
        applyKnotLines();
      },
      projectPoint(v) {
        const p = v.clone().project(camera);
        return { x: (p.x * 0.5 + 0.5), y: (1 - (p.y * 0.5 + 0.5)), z: p.z };
      },
      bounds() { return gt ? { radius: gt.radius, center: gt.center.clone() } : null; },
      reset() { st.step = 0; refresh(); },
      onStats(cb) { onStats = cb; if (stats.step !== undefined) cb(stats); },
      onZoom(cb) { onZoom = cb; },
      measureSeries(k) { return measureSeries(k); },
      dispose() {
        dead = true; cancelAnimationFrame(raf);
        ro.disconnect();
        canvas.removeEventListener('pointerdown', onDown);
        window.removeEventListener('pointermove', onMove);
        window.removeEventListener('pointerup', onUp);
        canvas.removeEventListener('wheel', onWheel);
        disposeObj(gt); disposeObj(meshOurs); disposeObj(meshBase); disposeObj(chartGrp);
        disposeTri();
        renderer.dispose();
        container.innerHTML = '';
      }
    };
  }

  return { create, MODELS, DATASETS, BASELINES, curve, chamfer, prims, params, VERSION };
})();
