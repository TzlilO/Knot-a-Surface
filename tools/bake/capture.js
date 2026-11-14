/* Step 1 of the feature bake: render every scene of lab.html from its capture rig.
   For each view it saves the rendered picture (what DINOv3 will look at) and, for every second pixel,
   which chart and which (u, v) is visible there (what the features get attached to).
     node tools/bake/capture.js <outdir> [--page <url>] [scene ...]
   No GPU is needed; the page is stepped on a virtual clock, so slow software WebGL only costs time. */
const fs = require('fs'), path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT || '/opt/node-tools/node_modules/playwright');
const args = process.argv.slice(2), out = args.shift();
let page = 'file://' + path.resolve(__dirname, '../../lab.html');
if (args[0] === '--page') { args.shift(); page = args.shift(); }
const RES = 768, FRAMES = 6, STRIDE = 2, BUDGET = 150000;

(async () => {
  const b = await chromium.launch(), p = await b.newPage({ viewport: { width: 1280, height: 900 } });
  p.on('pageerror', e => { console.error('page error:', e.message); process.exit(1); });
  await p.addInitScript(() => {
    let q = new Map(), id = 0, t = 0;
    window.requestAnimationFrame = cb => { q.set(++id, cb); return id; };
    window.cancelAnimationFrame = i => { q.delete(i); };
    window.__step = () => { t += 16; const run = q; q = new Map(); run.forEach(cb => cb(t)); };
  });
  await p.goto(page + '?dpr=1&budget=' + BUDGET);
  await p.waitForFunction(() => window.KnotLab && KnotLab.app);
  const scenes = args.length ? args : await p.evaluate(() => Object.keys(KnotLab.scenes));
  for (const scn of scenes) {
    const meta = await p.evaluate(([scn, RES]) => {
      const A = KnotLab.app; A.load(scn); A.R.resize(RES, RES, 1);
      const d = A.scene();
      return { scene: scn, res: RES, views: d.rig.cams.length, charts: A.charts().map(c => ({ name: c.name, nu: c.patch.nu, nv: c.patch.nv,
        periodicU: c.patch.periodicU, periodicV: c.patch.periodicV, lock: !!c.lock, gU: Array.from(c.gU), gV: Array.from(c.gV) })) };
    }, [scn, RES]);
    const dir = path.join(out, scn); fs.mkdirSync(dir, { recursive: true });
    for (let i = 0; i < meta.views; i++) {
      const r = await p.evaluate(([i, FRAMES, STRIDE]) => {
        const A = KnotLab.app, d = A.scene(), c = d.rig.cams[i], t = d.rig.target;
        const dx = c[0] - t[0], dy = c[1] - t[1], dz = c[2] - t[2], rr = Math.hypot(dx, dy, dz);
        A.orbit.to({ th: Math.atan2(dx, dz), ph: Math.acos(dy / rr), r: rr, tx: t[0], ty: t[1], tz: t[2] }, true);
        for (let k = 0; k < FRAMES; k++) window.__step();
        const png = A.R.canvas.toDataURL('image/png'), ids = A.R.readIds(), w = ids.w / STRIDE, h = ids.h / STRIDE;
        const o = new Float32Array(w * h * 3);
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
          const s = ((y * STRIDE) * ids.w + x * STRIDE) * 4, q = (y * w + x) * 3;
          o[q] = ids.data[s]; o[q + 1] = ids.data[s + 1]; o[q + 2] = ids.data[s + 2];
        }
        let bin = ''; const u8 = new Uint8Array(o.buffer);
        for (let k = 0; k < u8.length; k += 32768) bin += String.fromCharCode.apply(null, u8.subarray(k, k + 32768));
        return { png, ids: btoa(bin), w, h };
      }, [i, FRAMES, STRIDE]);
      fs.writeFileSync(path.join(dir, 'v' + String(i).padStart(2, '0') + '.png'), Buffer.from(r.png.split(',')[1], 'base64'));
      fs.writeFileSync(path.join(dir, 'v' + String(i).padStart(2, '0') + '.ids'), Buffer.from(r.ids, 'base64'));
      meta.idw = r.w; meta.idh = r.h; meta.stride = STRIDE;
      process.stdout.write(scn + ' view ' + (i + 1) + '/' + meta.views + '\r');
    }
    fs.writeFileSync(path.join(dir, 'meta.json'), JSON.stringify(meta));
    console.log(scn + ': ' + meta.views + ' views captured        ');
  }
  await b.close();
})();
