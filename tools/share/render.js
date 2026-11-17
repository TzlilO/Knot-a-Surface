/* node tools/share/render.js [out.jpg]      (default: talk/share.jpg)
   The picture a shared link to the talk shows (og:image in talk.html): the scanned owl in the teaser's split screen,
   its face in the picture band, its Gaussians to the left and its normals to the right. tools/share/page.html is
   drawn once, at a fixed moment of the teaser's loop and from a fixed camera, at twice the size, then halved.
   The three bands that cross the owl keep their labels, larger and at the foot of the frame, placed inside the
   centre square (some previews crop to it); the outer labels and dividers are left out.
   Needs playwright (the talk's tests use the same one) and takes a minute or two without a GPU. */
const path = require('path'), fs = require('fs');
let pw; try { pw = require('playwright'); } catch (e) { pw = require('/opt/node-tools/node_modules/playwright'); }
const W = 1200, H = 630;
const T = 2.0783;                                                        // seconds into the teaser's loop: the bands sit centred on the owl, eyes open, head straight
const CAM = { th: 0.30, ph: 1.32, r: 3.0, tx: 0.16, ty: 2.75, tz: 0.2 }; // head and chest, from the front
(async () => {
  const out = path.resolve(process.argv[2] || path.join(__dirname, '..', '..', 'talk', 'share.jpg'));
  const b = await pw.chromium.launch(), p = await b.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  p.setDefaultTimeout(900000);
  p.on('pageerror', e => { console.error('page error: ' + e.message); process.exitCode = 1; });
  // a clock the script steps itself, so the frame is the same every time
  await p.addInitScript(() => { let q = [], now = 0, id = 0; window.requestAnimationFrame = f => { q.push([++id, f]); return id; };
    window.cancelAnimationFrame = i => { q = q.filter(e => e[0] !== i); }; performance.now = () => now;
    window.__step = ms => { now += ms; const r = q; q = []; r.forEach(e => e[1](now)); }; });
  await p.goto('file://' + path.join(__dirname, 'page.html')); await p.waitForTimeout(1500);
  await p.evaluate(([t, cam]) => {
    const G = KnotTalk.gl, draw = G.draw, A = window.sim.api;
    A.orbit.touched = true; Object.assign(A.orbit.goal, cam);
    G.draw = () => {};                                                   // run the clock to t without drawing
    for (let i = 0, n = Math.round(t * 1000 / 33); i < n; i++) window.__step(33);
    for (let i = 0; i < 200; i++) window.__step(0.0001);                 // the camera settles on its goal; the loop does not move
    G.draw = draw; window.__step(0.0001); window.__step(0.0001);
  }, [T, CAM]);
  const kept = await p.evaluate(([w]) => {
    const split = document.querySelector('.kt-split'), names = [], inner = (w - 630) / 2 + 9;
    split.querySelectorAll('span').forEach(l => {
      if (l.style.display === 'none') return;
      const r = l.getBoundingClientRect(), x = r.left + r.width / 2;
      if (x < 330 || x > w - 330) { l.style.display = 'none'; return; }
      const m = /translate\(([-\d.]+)px,\s*([-\d.]+)px\)/.exec(l.style.transform), y = 578, x1 = +m[1] - 0.2 * (y - +m[2]);   // down along its band (the dividers lean 0.2)
      l.style.fontSize = '18px'; l.style.padding = '9px 16px'; l.style.letterSpacing = '.1em'; l.style.opacity = 1;
      const half = l.getBoundingClientRect().width / 2, xc = Math.max(inner + half, Math.min(w - inner - half, x1));
      l.style.transform = 'translate(' + xc + 'px,' + y + 'px) translate(-50%,-50%)'; names.push(l.textContent);
    });
    split.querySelectorAll('i').forEach(d => { const r = d.getBoundingClientRect(), x = r.left + r.width / 2; if (d.style.display !== 'none' && (x < 180 || x > w - 180)) d.style.display = 'none'; });
    return names;
  }, [W]);
  await p.waitForTimeout(800);
  const png = await p.screenshot();
  // halve it in a canvas and write a JPEG
  const q = await b.newPage();
  const jpg = await q.evaluate(async ([src, w, h]) => {
    const im = new Image(); im.src = src; await im.decode();
    const c = document.createElement('canvas'); c.width = w; c.height = h;
    const x = c.getContext('2d'); x.imageSmoothingEnabled = true; x.imageSmoothingQuality = 'high'; x.drawImage(im, 0, 0, w, h);
    return c.toDataURL('image/jpeg', 0.9).split(',')[1];
  }, ['data:image/png;base64,' + png.toString('base64'), W, H]);
  fs.writeFileSync(out, Buffer.from(jpg, 'base64'));
  await b.close();
  console.log('wrote ' + out + ' · ' + W + ' x ' + H + ' · ' + fs.statSync(out).size + ' bytes · bands: ' + kept.join(' | '));
})();
