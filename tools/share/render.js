/* node tools/share/render.js [out.jpg] [layout]      (default: talk/share.jpg, layout "pair")
   The picture a shared link to the talk shows (og:image in talk.html), 1200 x 630.
     pair   two owls: the scan cut in two (stone | normals), and its copy cut in three (stone | glass | jelly)
     row    three owls: glass, the scan cut in two (stone | normals), jelly
   tools/share/page.html draws it: every band is the `manifold` module itself, in scene `paint`, with one material
   fully wiped in (or the normals reading), from one camera and at one moment; the page only cuts the bands.
   The moment T: all materials are in, the owl's eyes are open and its head is straight; for "pair" it is also an
   instant when the jelly's sideways sway is zero, so its band lines up with its neighbours.
   Drawn at twice the size and halved. Needs playwright (the talk's tests use the same one); without a GPU it
   takes several minutes (one module instance per band). */
const path = require('path'), fs = require('fs');
let pw; try { pw = require('playwright'); } catch (e) { pw = require('/opt/node-tools/node_modules/playwright'); }
const W = 1200, H = 630, BUDGET = 200000;
const LAYOUTS = {
  pair: { t: 8125, cam: { th: 0.30, ph: 1.32, r: 3.0, tx: 0.16, ty: 2.75, tz: 0.2 }, owls: [
    { left: 50,  w: 600, cuts: [300],      bands: [{ mat: 'stone', label: 'stone' }, { mat: 'stone', view: 'normal', label: 'normals' }] },
    { left: 550, w: 600, cuts: [232, 368], bands: [{ mat: 'stone', label: 'stone' }, { mat: 'glass', label: 'glass' }, { mat: 'jelly', label: 'jelly' }] }] },
  row: { t: 8000, cam: { th: 0.30, ph: 1.32, r: 3.9, tx: 0.16, ty: 2.6, tz: 0.2 }, owls: [
    { left: 0,   w: 400, cuts: [],    bands: [{ mat: 'glass', label: 'glass' }] },
    { left: 400, w: 400, cuts: [200], bandw: 130, bands: [{ mat: 'stone', label: 'stone' }, { mat: 'stone', view: 'normal', label: 'normals' }] },
    { left: 800, w: 400, cuts: [],    bands: [{ mat: 'jelly', label: 'jelly' }] }] }
};
(async () => {
  const out = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', 'talk', 'share.jpg'));
  const L = LAYOUTS[process.argv[3] || 'pair']; if (!L) { console.error('layouts: ' + Object.keys(LAYOUTS).join(', ')); process.exit(1); }
  const budget = +(process.env.SHARE_BUDGET || BUDGET);
  const b = await pw.chromium.launch(), p = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  p.setDefaultTimeout(3600000);
  p.on('pageerror', e => { console.error('page error: ' + e.message); process.exitCode = 1; });
  // a clock the script steps itself, shared by every instance, so the frame is the same every time
  await p.addInitScript(() => { let q = [], now = 0, id = 0; window.requestAnimationFrame = f => { q.push([++id, f]); return id; };
    window.cancelAnimationFrame = i => { q = q.filter(e => e[0] !== i); }; performance.now = () => now;
    window.__step = ms => { now += ms; const r = q; q = []; r.forEach(e => e[1](now)); }; });
  await p.goto('file://' + path.join(__dirname, 'page.html') + '?budget=' + budget + '&owls=' + encodeURIComponent(JSON.stringify(L.owls)));
  await p.waitForTimeout(2500);
  await p.evaluate(([t, cam]) => {
    const G = KnotTalk.gl, draw = G.draw;
    window.sims.forEach(s => { s.api.orbit.touched = true; Object.assign(s.api.orbit.goal, cam); });
    G.draw = () => {};                                                   // run the clock to t without drawing
    const n = Math.floor(t / 33); for (let i = 0; i < n; i++) window.__step(33); window.__step(t - n * 33 - 0.02);
    for (let i = 0; i < 200; i++) window.__step(0.0001);                 // the cameras settle on their goal; nothing else moves
    G.draw = draw; window.__step(0.0001); window.__step(0.0001);
  }, [L.t, L.cam]);
  await p.waitForTimeout(1000);
  const png = await p.screenshot();
  const q = await b.newPage();                                           // halve it in a canvas and write a JPEG
  const jpg = await q.evaluate(async ([src, w, h]) => {
    const im = new Image(); im.src = src; await im.decode();
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const x = c.getContext('2d'); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high'; x.drawImage(im, 0, 0, w, h);
    return c.toDataURL('image/jpeg', 0.9).split(',')[1];
  }, ['data:image/png;base64,' + png.toString('base64'), W, H]);
  fs.writeFileSync(out, Buffer.from(jpg, 'base64'));
  await b.close();
  console.log('wrote ' + out + ' · ' + W + ' x ' + H + ' · ' + fs.statSync(out).size + ' bytes');
})();
