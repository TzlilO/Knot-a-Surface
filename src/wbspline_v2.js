/* ═══════════════════ WIDGET: B-spline playground v2 ═══════════════════
   Full NURBS: adjustable degree, per-CP weight/color/splat-scale (dbl-click
   a control point), draggable CPs, exact midpoint knot insertion.        */
// extended CP record: 8 channels [x,y,z, w, r,g,b, sMul]
const BS_STRIDE = 8;
function evalNURBS(cps, nu, nv, KU, KV, p, q, u, v, out) {
  const Nu = basisFuncs(KU, p, nu, u), Nv = basisFuncs(KV, q, nv, v);
  let X = 0, Y = 0, Z = 0, W = 0, R = 0, G = 0, B = 0, SM = 0;
  for (let j = 0; j < nv; j++) {
    const bv = Nv[j]; if (!bv) continue;
    for (let i = 0; i < nu; i++) {
      const bu = Nu[i]; if (!bu) continue;
      const k = (j * nu + i) * BS_STRIDE;
      const w = cps[k + 3], bw = bu * bv * w;
      X += bw * cps[k]; Y += bw * cps[k + 1]; Z += bw * cps[k + 2]; W += bw;
      R += bw * cps[k + 4]; G += bw * cps[k + 5]; B += bw * cps[k + 6]; SM += bw * cps[k + 7];
    }
  }
  const iw = W !== 0 ? 1 / W : 0;
  out.x = X * iw; out.y = Y * iw; out.z = Z * iw;
  out.r = R * iw; out.g = G * iw; out.b = B * iw; out.s = SM * iw;
  return out;
}
// exact midpoint knot insertion in U on the extended grid (rational: weights ride along)
function bsInsertU(cps, nu, nv, KU, p) {
  let curK = KU.slice(), cur = cps, curNu = nu;
  const spans = [];
  for (let i = p; i < curK.length - p - 1; i++)
    if (curK[i + 1] > curK[i]) spans.push((curK[i] + curK[i + 1]) / 2);
  for (const ub of spans) {
    let k = p; while (k < curK.length - p - 2 && ub >= curK[k + 1]) k++;
    const nn = curNu + 1, out = new Float32Array(nn * nv * BS_STRIDE);
    for (let j = 0; j < nv; j++) for (let i = 0; i < nn; i++) {
      let a;
      if (i <= k - p) a = 1; else if (i >= k + 1) a = 0;
      else a = (ub - curK[i]) / (curK[i + p] - curK[i]);
      const o = (j * nn + i) * BS_STRIDE;
      const iA = (j * curNu + Math.min(i, curNu - 1)) * BS_STRIDE;
      const iB = (j * curNu + Math.max(i - 1, 0)) * BS_STRIDE;
      for (let c = 0; c < BS_STRIDE; c++) out[o + c] = a * cur[iA + c] + (1 - a) * cur[iB + c];
    }
    curK = curK.slice(0, k + 1).concat([ub], curK.slice(k + 1));
    cur = out; curNu = nn;
  }
  return { cps: cur, nu: curNu, K: curK };
}
function bsTranspose(cps, nu, nv) {
  const out = new Float32Array(nu * nv * BS_STRIDE);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const a = (j * nu + i) * BS_STRIDE, b = (i * nv + j) * BS_STRIDE;
    for (let c = 0; c < BS_STRIDE; c++) out[b + c] = cps[a + c];
  }
  return out;
}

function wBspline(section) {
  const cv = section.querySelector('#bspline-canvas');
  const rd = makeRenderer(cv);
  const scene = new T3.Scene();
  const cam = new T3.PerspectiveCamera(46, 16 / 9, 0.1, 200);
  scene.add(new T3.AmbientLight(0xffffff, 0.7),
    (() => { const L = new T3.DirectionalLight(0xbfd8ff, 0.9); L.position.set(6, 10, 4); return L; })());
  const orbit = { th: -0.9, ph: 0.5, r: 15 };
  const applyCam = () => {
    cam.position.set(orbit.r * Math.cos(orbit.ph) * Math.cos(orbit.th), orbit.r * Math.sin(orbit.ph), orbit.r * Math.cos(orbit.ph) * Math.sin(orbit.th));
    cam.lookAt(0, 0.4, 0);
  };
  applyCam();
  let stopOrbit = null;

  const SHAPES = {
    wave: (x, z) => 1.1 * Math.sin(x * 0.9) * Math.cos(z * 0.9),
    dome: (x, z) => 2.4 - 0.16 * (x * x + z * z),
    saddle: (x, z) => 0.11 * (x * x - z * z),
    terrain: (x, z) => 1.5 * (hash1(Math.round(x * 2) + 9, Math.round(z * 2) + 7) - 0.3) + 0.7 * Math.sin(x) * Math.cos(z * 1.3)
  };
  let deg = 3, nu = 6, nv = 6, KU = clampedKnots(6, 3), KV = clampedKnots(6, 3);
  let cps = null, SN = 10;
  const SURF_RES = 28;
  function heightColor(y) {          // default control-feature color: height gradient
    const f = Math.max(0, Math.min(1, (y + 1.6) / 3.6));
    return [0.16 + 0.35 * f, 0.42 + 0.42 * f, 0.85 - 0.25 * f];
  }
  function fillShape(name) {
    cps = new Float32Array(nu * nv * BS_STRIDE);
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const x = (i / (nu - 1) - 0.5) * 8, z = (j / (nv - 1) - 0.5) * 8;
      const y = SHAPES[name](x, z);
      const k = (j * nu + i) * BS_STRIDE;
      const c = heightColor(y);
      cps[k] = x; cps[k + 1] = y; cps[k + 2] = z;
      cps[k + 3] = 1;                                  // NURBS weight
      cps[k + 4] = c[0]; cps[k + 5] = c[1]; cps[k + 6] = c[2];
      cps[k + 7] = 1;                                  // splat-scale multiplier
    }
  }
  let shapeName = 'wave';
  fillShape(shapeName);

  let cpMesh = null, netLines = null, surfMesh = null, splats = null, selRing = null;
  const _o = {}, _o2 = {}, _o3 = {};
  const _v = new T3.Vector3(), _v2 = new T3.Vector3(), _v3 = new T3.Vector3(),
        _q = new T3.Quaternion(), _m = new T3.Matrix4(), _up = new T3.Vector3(0, 0, 1);

  function rebuildObjects() {
    for (const o of [cpMesh, netLines, surfMesh, splats, selRing]) if (o) { scene.remove(o); if (o.geometry) o.geometry.dispose(); }
    cpMesh = new T3.InstancedMesh(new T3.SphereGeometry(0.22, 12, 9),
      new T3.MeshBasicMaterial({ color: 0xffa726, transparent: true, opacity: 0.95 }), nu * nv);
    scene.add(cpMesh);
    const segs = (nu - 1) * nv + (nv - 1) * nu;
    netLines = new T3.LineSegments(
      new T3.BufferGeometry().setAttribute('position', new T3.BufferAttribute(new Float32Array(segs * 6), 3)),
      new T3.LineBasicMaterial({ color: 0xffa726, transparent: true, opacity: 0.3 }));
    scene.add(netLines);
    const g = new T3.PlaneGeometry(1, 1, SURF_RES, SURF_RES);
    surfMesh = new T3.Mesh(g, new T3.MeshLambertMaterial({
      color: 0x4a7bd0, transparent: true, opacity: 0.28, side: T3.DoubleSide, depthWrite: false }));
    scene.add(surfMesh);
    splats = new T3.InstancedMesh(new T3.CircleGeometry(0.5, 12),
      new T3.MeshBasicMaterial({ transparent: true, opacity: 0.72, side: T3.DoubleSide, depthWrite: false }), 40 * 40);
    splats.instanceColor = new T3.InstancedBufferAttribute(new Float32Array(40 * 40 * 3), 3);
    scene.add(splats);
    selRing = new T3.Mesh(new T3.RingGeometry(0.3, 0.36, 20),
      new T3.MeshBasicMaterial({ color: 0x4fc3f7, side: T3.DoubleSide }));
    selRing.visible = false;
    scene.add(selRing);
  }

  function refresh() {
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
      const k = j * nu + i, b = k * BS_STRIDE;
      const wScale = 0.75 + 0.35 * Math.min(2.5, cps[b + 3]);   // weight shown as CP size
      _m.compose(_v.set(cps[b], cps[b + 1], cps[b + 2]), _q.identity(), _v2.set(wScale, wScale, wScale));
      cpMesh.setMatrixAt(k, _m);
    }
    cpMesh.instanceMatrix.needsUpdate = true;
    const ap = netLines.geometry.attributes.position.array;
    let w = 0;
    const put = k => { const b = k * BS_STRIDE; ap[w++] = cps[b]; ap[w++] = cps[b + 1]; ap[w++] = cps[b + 2]; };
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu - 1; i++) { put(j * nu + i); put(j * nu + i + 1); }
    for (let i = 0; i < nu; i++) for (let j = 0; j < nv - 1; j++) { put(j * nu + i); put((j + 1) * nu + i); }
    netLines.geometry.attributes.position.needsUpdate = true;
    const pos = surfMesh.geometry.attributes.position;
    let idx = 0;
    for (let j = 0; j <= SURF_RES; j++) for (let i = 0; i <= SURF_RES; i++) {
      evalNURBS(cps, nu, nv, KU, KV, deg, deg, i / SURF_RES, j / SURF_RES, _o);
      pos.setXYZ(idx++, _o.x, _o.y, _o.z);
    }
    pos.needsUpdate = true;
    surfMesh.geometry.computeVertexNormals();
    const e = 0.012;
    let k2 = 0;
    for (let j = 0; j < SN; j++) for (let i = 0; i < SN; i++) {
      const u = (i + 0.5) / SN, v = (j + 0.5) / SN;
      evalNURBS(cps, nu, nv, KU, KV, deg, deg, u, v, _o);
      evalNURBS(cps, nu, nv, KU, KV, deg, deg, Math.min(1, u + e), v, _o2);
      evalNURBS(cps, nu, nv, KU, KV, deg, deg, u, Math.min(1, v + e), _o3);
      _v2.set(_o2.x - _o.x, _o2.y - _o.y, _o2.z - _o.z);
      _v3.set(_o3.x - _o.x, _o3.y - _o.y, _o3.z - _o.z);
      const n = _v2.clone().cross(_v3).normalize();
      _q.setFromUnitVectors(_up, n);
      const sm = Math.max(0.15, _o.s);
      const su = _v2.length() / e / SN * 1.2 * sm, sv = _v3.length() / e / SN * 1.2 * sm;
      _m.compose(_v.set(_o.x, _o.y, _o.z).addScaledVector(n, 0.03), _q, new T3.Vector3(su, sv, 1));
      splats.setMatrixAt(k2, _m);
      splats.instanceColor.setXYZ(k2, _o.r, _o.g, _o.b);
      k2++;
    }
    splats.count = SN * SN;
    splats.instanceMatrix.needsUpdate = true;
    splats.instanceColor.needsUpdate = true;
    section.querySelector('#bs-grid-label').textContent = nu + '×' + nv + ' CPs · degree ' + deg;
  }
  rebuildObjects(); refresh();

  /* CP dragging + selection */
  const ray = new T3.Raycaster(); const ndc = new T3.Vector2();
  let dragCP = -1, selCP = -1; const plane = new T3.Plane(); const hitP = new T3.Vector3();
  function toNDC(e) {
    const r = cv.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }
  function pickCP(e) {
    toNDC(e); ray.setFromCamera(ndc, cam);
    const hit = ray.intersectObject(cpMesh)[0];
    return hit && hit.instanceId !== undefined ? hit.instanceId : -1;
  }
  function cpDown(e) {
    const id = pickCP(e);
    if (id >= 0) {
      dragCP = id;
      const b = id * BS_STRIDE;
      plane.setFromNormalAndCoplanarPoint(cam.getWorldDirection(_v).negate(), _v2.set(cps[b], cps[b + 1], cps[b + 2]));
      e.stopImmediatePropagation();
    }
  }
  function cpMove(e) {
    if (dragCP < 0) return;
    toNDC(e); ray.setFromCamera(ndc, cam);
    if (ray.ray.intersectPlane(plane, hitP)) {
      const b = dragCP * BS_STRIDE;
      cps[b] = hitP.x; cps[b + 1] = hitP.y; cps[b + 2] = hitP.z;
      if (selCP === dragCP) placeSelRing();
      refresh();
    }
  }
  const cpUp = () => dragCP = -1;
  function placeSelRing() {
    if (selCP < 0) { selRing.visible = false; return; }
    const b = selCP * BS_STRIDE;
    selRing.position.set(cps[b], cps[b + 1], cps[b + 2]);
    selRing.lookAt(cam.position);
    selRing.visible = true;
  }
  /* per-CP editor panel (dbl-click) */
  const panel = section.querySelector('#bs-cp-panel');
  const pw = panel.querySelector('#bs-cp-w'), pc = panel.querySelector('#bs-cp-color'),
        ps = panel.querySelector('#bs-cp-s'), pTitle = panel.querySelector('#bs-cp-title');
  function openPanel(id) {
    selCP = id;
    const b = id * BS_STRIDE;
    pTitle.textContent = 'CP (' + (id % nu) + ',' + Math.floor(id / nu) + ')';
    pw.value = cps[b + 3]; ps.value = cps[b + 7];
    const hx = c => ('0' + Math.round(Math.max(0, Math.min(1, c)) * 255).toString(16)).slice(-2);
    pc.value = '#' + hx(cps[b + 4]) + hx(cps[b + 5]) + hx(cps[b + 6]);
    panel.style.display = 'block';
    placeSelRing();
  }
  function closePanel() { selCP = -1; selRing.visible = false; panel.style.display = 'none'; }
  function cpDbl(e) {
    const id = pickCP(e);
    if (id >= 0) { openPanel(id); e.stopImmediatePropagation(); } else closePanel();
  }
  const onW = () => { if (selCP >= 0) { cps[selCP * BS_STRIDE + 3] = +pw.value; refresh(); } };
  const onS = () => { if (selCP >= 0) { cps[selCP * BS_STRIDE + 7] = +ps.value; refresh(); } };
  const onC = () => {
    if (selCP < 0) return;
    const b = selCP * BS_STRIDE, h = pc.value;
    cps[b + 4] = parseInt(h.slice(1, 3), 16) / 255;
    cps[b + 5] = parseInt(h.slice(3, 5), 16) / 255;
    cps[b + 6] = parseInt(h.slice(5, 7), 16) / 255;
    refresh();
  };
  const onClose = () => closePanel();
  pw.addEventListener('input', onW); ps.addEventListener('input', onS); pc.addEventListener('input', onC);
  panel.querySelector('#bs-cp-close').addEventListener('click', onClose);

  cv.addEventListener('pointerdown', cpDown);
  cv.addEventListener('dblclick', cpDbl);
  window.addEventListener('pointermove', cpMove);
  window.addEventListener('pointerup', cpUp);
  stopOrbit = simpleOrbit(cv, orbit, applyCam);

  /* UI */
  const onClick = [];
  function wire(sel, fn) {
    section.querySelectorAll(sel).forEach(b => { const h = () => fn(b); b.addEventListener('click', h); onClick.push([b, h]); });
  }
  function resetGrid() {
    nu = 6; nv = 6;
    KU = clampedKnots(nu, deg); KV = clampedKnots(nv, deg);
    fillShape(shapeName); closePanel(); rebuildObjects(); refresh();
  }
  wire('.bs-shape', b => {
    section.querySelectorAll('.bs-shape').forEach(x => x.classList.toggle('on', x === b));
    shapeName = b.dataset.s; resetGrid();
  });
  wire('.bs-deg', b => {
    section.querySelectorAll('.bs-deg').forEach(x => x.classList.toggle('on', x === b));
    deg = +b.dataset.d; resetGrid();
  });
  wire('.bs-n', b => {
    section.querySelectorAll('.bs-n').forEach(x => x.classList.toggle('on', x === b));
    SN = +b.dataset.n; refresh();
  });
  wire('#bs-subdiv', () => {
    if (nu > 24) return;
    let r = bsInsertU(cps, nu, nv, KU, deg);
    cps = r.cps; nu = r.nu; KU = r.K;
    let tr = bsTranspose(cps, nu, nv);
    r = bsInsertU(tr, nv, nu, KV, deg);
    cps = bsTranspose(r.cps, r.nu, nu); nv = r.nu; KV = r.K;
    closePanel(); rebuildObjects(); refresh();
  });
  wire('#bs-shuffle', () => {
    for (let k = 0; k < nu * nv; k++) {
      const b = k * BS_STRIDE;
      cps[b + 1] += (Math.random() - 0.5) * 1.6;
    }
    refresh();
  });

  let raf = 0, dead = false;
  (function loop() {
    if (dead) return;
    raf = requestAnimationFrame(loop);
    if (selRing.visible) selRing.lookAt(cam.position);
    sizeRenderer(rd, cv, cam);
    rd.render(scene, cam);
  })();
  return { dispose() {
    dead = true; cancelAnimationFrame(raf); if (stopOrbit) stopOrbit();
    cv.removeEventListener('pointerdown', cpDown);
    cv.removeEventListener('dblclick', cpDbl);
    window.removeEventListener('pointermove', cpMove);
    window.removeEventListener('pointerup', cpUp);
    pw.removeEventListener('input', onW); ps.removeEventListener('input', onS); pc.removeEventListener('input', onC);
    for (const [b, h] of onClick) b.removeEventListener('click', h);
    rd.dispose();
  } };
}
