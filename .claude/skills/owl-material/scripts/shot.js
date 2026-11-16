// Screenshot one state of a talk page on a virtual clock (software WebGL: no GPU needed).
//   node shot.js <out.png> <url-or-repo-path> [steps=40] [width=1280] [height=800] [scale=1] [js]
// `steps` frames of 33 ms are run; only the last two are really drawn (software WebGL rasterises every frame it
// is handed, so drawing them all takes minutes). `js` runs in the page before the first step, e.g.
//   "KnotTalk.mounted[0].api.setMaterial('jelly')"
// The paint wipe takes 3.5 s (106 steps). Console errors and page errors are printed; no output = clean.
const { chromium } = require('/opt/node-tools/node_modules/playwright');
const path = require('path');
(async () => {
  let [out, url, steps = 40, w = 1280, h = 800, dsf = 1, js = ''] = process.argv.slice(2);
  if (!/^[a-z]+:\/\//.test(url)) { const [f, hash] = url.split('#'); url = 'file://' + path.resolve(__dirname, '../../../..', f) + (hash ? '#' + hash : ''); }
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: +dsf });
  p.setDefaultTimeout(180000);
  p.on('pageerror', e => console.log('PAGE ERROR', e.message));
  p.on('console', m => { if (m.type() === 'error') console.log('CONSOLE ERROR', m.text().slice(0, 300)); });
  await p.addInitScript(() => {
    let q = [], now = 0, id = 0;
    window.requestAnimationFrame = f => { q.push([++id, f]); return id; };
    window.cancelAnimationFrame = i => { q = q.filter(e => e[0] !== i); };
    performance.now = () => now;
    window.__step = ms => { now += ms; const run = q; q = []; run.forEach(e => { try { e[1](now); } catch (err) { console.error('frame: ' + err.message + ' ' + ((err.stack || '').split('\n')[1] || '')); } }); };
  });
  await p.goto(url);
  await p.waitForTimeout(600);
  if (js) await p.evaluate(js);
  await p.evaluate(n => {
    const G = window.KnotTalk && window.KnotTalk.gl, d = G && G.draw;
    if (d) G.draw = () => {};
    for (let i = 0; i < n - 2; i++) window.__step(33);
    if (d) G.draw = d;
    for (let i = 0; i < Math.min(2, n); i++) window.__step(33);
  }, +steps);
  await p.screenshot({ path: out });
  await b.close();
})();
