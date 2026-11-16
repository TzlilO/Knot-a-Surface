// Error sweep after a material change: the manifold lab page (every scene, every material) and the whole deck,
// at a desktop and a phone size. Prints one line per page and ALL CLEAN, or the errors.
//   node sweep.js            everything
//   node sweep.js lab        the lab page only (quick)
const { chromium } = require('/opt/node-tools/node_modules/playwright');
const path = require('path');
const ROOT = 'file://' + path.resolve(__dirname, '../../../..') + '/';
(async () => {
  const b = await chromium.launch(), only = process.argv[2];
  let bad = 0;
  async function page(url, w, h, fn) {
    const p = await b.newPage({ viewport: { width: w, height: h } });
    p.setDefaultTimeout(600000);
    const errs = [];
    p.on('pageerror', e => errs.push('PAGE ERROR ' + e.message));
    p.on('console', m => { if (m.type() === 'error') errs.push('CONSOLE ERROR ' + m.text().slice(0, 200)); });
    await p.addInitScript(() => {
      let q = [], now = 0, id = 0;
      window.requestAnimationFrame = f => { q.push([++id, f]); return id; };
      window.cancelAnimationFrame = i => { q = q.filter(e => e[0] !== i); };
      performance.now = () => now;
      window.__step = ms => { now += ms; const run = q; q = []; run.forEach(e => { try { e[1](now); } catch (err) { console.error('frame: ' + err.message + ' ' + ((err.stack || '').split('\n')[1] || '')); } }); };
    });
    await p.goto(ROOT + url); await p.waitForTimeout(400);
    const info = await fn(p);
    bad += errs.length;
    console.log((errs.length ? 'FAIL ' : 'ok   ') + url + ' ' + w + 'x' + h + ' ' + info + (errs.length ? '\n   ' + [...new Set(errs)].slice(0, 6).join('\n   ') : ''));
    await p.close();
  }
  const step = (p, n) => p.evaluate(k => { for (let i = 0; i < k; i++) window.__step(33); }, n);
  for (const [w, h] of [[1280, 800], [390, 780]]) {
    await page('talk/sims/manifold.html', w, h, async p => {
      const ids = await p.evaluate(() => KnotTalk.sims.manifold.scenes.map(x => x.id));
      for (const id of ids) { await p.evaluate(i => KnotTalk.mounted[0].api.setScene(i), id); await step(p, 5); }
      // every material: wipe it in, let it move, then leave and come back (colours and shape must be restored)
      const mats = await p.evaluate(() => { KnotTalk.mounted[0].api.setScene('paint'); return [...document.querySelectorAll('.mf-mat .kt-seg-btn')].map(x => x.textContent); });
      const names = await p.evaluate(() => (KnotTalk.scenes.scan.build().materials || []).map(m => m.v));
      for (const m of names) {
        await p.evaluate(v => { const A = KnotTalk.mounted[0].api, G = KnotTalk.gl, d = G.draw; A.setMaterial(v); G.draw = () => {}; for (let i = 0; i < 130; i++) window.__step(33); G.draw = d; }, m);
        await step(p, 3);
      }
      // every target: the whole object, the owl alone, a selection made with the wand on the atlas (on the DINOv3 features)
      const tgs = await p.evaluate(() => [...document.querySelectorAll('.mf-target .kt-seg-btn')].map(x => x.textContent));
      for (const t of tgs) {
        await p.evaluate(v => { const A = KnotTalk.mounted[0].api, G = KnotTalk.gl, d = G.draw; A.setTarget(v);
          if (v === 'selection') { const H = A.hands(); H.channel('feat'); H.wandAt(150, 300); }
          G.draw = () => {}; for (let i = 0; i < 130; i++) window.__step(33); G.draw = d; }, t);
        await step(p, 3);
      }
      await p.evaluate(() => { const A = KnotTalk.mounted[0].api; A.setScene('scene'); A.setScene('paint'); A.setScene('edit'); }); await step(p, 3);
      return ids.length + ' scenes, materials: ' + (mats.join(', ') || 'none') + '; targets: ' + (tgs.join(', ') || 'none');
    });
    if (only === 'lab') continue;
    await page('talk.html', w, h, async p => {
      const k = await p.evaluate(() => document.querySelectorAll('.step').length);
      for (let i = 0; i < k; i++) {
        await p.evaluate(j => { const s = document.querySelectorAll('.step')[j]; if (s.offsetParent) { s.scrollIntoView({ block: 'center' }); window.dispatchEvent(new Event('scroll')); } }, i);
        await step(p, 3); await p.waitForTimeout(30);
      }
      // the talk (everything after the opening, .prelude) is budgeted to at most 40 / 20 minutes; the opening comes on top
      const sum = await p.evaluate(() => ['min40', 'min20'].map(a => [...document.querySelectorAll('[data-' + a + ']')].reduce((t, e) => t + (e.closest('.prelude') ? 0 : +e.dataset[a]), 0)));
      return k + ' steps, time budgets of the talk ' + sum.join(' / ') + (sum[0] <= 40 && sum[1] <= 20 ? '' : '  <-- must be at most 40 / 20');
    });
  }
  console.log(bad ? 'ERRORS: ' + bad : 'ALL CLEAN');
  await b.close();
})();
