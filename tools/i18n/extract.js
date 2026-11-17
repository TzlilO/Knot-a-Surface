/* node tools/i18n/extract.js [out.json]
   Lists what talk.html asks a translation for: every block of copy and every translatable attribute, with the key
   the page will look up (talk/i18n.js: KnotTalk.i18n.blocks / key / attrs), grouped by section. The page is loaded
   with every script but talk/i18n.js held back, so the copy is as written.
   Needs playwright (the same one the talk's tests use: /opt/node-tools/node_modules/playwright, or NODE_PATH). */
const path = require('path'), fs = require('fs');
let pw; try { pw = require('playwright'); } catch (e) { pw = require('/opt/node-tools/node_modules/playwright'); }
(async () => {
  const root = path.resolve(__dirname, '..', '..'), out = process.argv[2];
  const b = await pw.chromium.launch(), p = await b.newPage();
  await p.route('**/*', r => { const u = r.request().url(); return /\.js(\?|$)/.test(u) && !/talk\/i18n\.js/.test(u) ? r.abort() : /^file:/.test(u) ? r.continue() : r.abort(); });
  await p.goto('file://' + path.join(root, 'talk.html'));
  const groups = await p.evaluate(() => {
    const I = KnotTalk.i18n, groups = [], seen = {};
    const name = el => { const s = el.closest('section[data-rail]'); return s ? s.getAttribute('data-rail') + ' · ' + s.getAttribute('data-title') : 'page'; };
    const add = (g, key, kind) => { let o = groups.find(x => x.section === g); if (!o) groups.push(o = { section: g, items: [] }); o.items.push({ key, kind, repeat: !!seen[key] }); seen[key] = 1; };
    I.blocks(document.body).forEach(el => add(name(el), I.key(el), el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(/\s+/)[0] : '')));
    I.attrs.forEach(a => document.querySelectorAll('[' + a + ']').forEach(el => { const v = el.getAttribute(a); if (/[A-Za-z]/.test(v)) add(name(el), v, '@' + a); }));
    return groups;
  });
  await b.close();
  const n = groups.reduce((t, g) => t + g.items.length, 0), uniq = groups.reduce((t, g) => t + g.items.filter(i => !i.repeat).length, 0);
  if (out) fs.writeFileSync(out, JSON.stringify(groups, null, 1));
  console.log(groups.map(g => g.section + ': ' + g.items.filter(i => !i.repeat).length + ' new of ' + g.items.length).join('\n'));
  console.log(n + ' items, ' + uniq + ' distinct keys, ' + groups.reduce((t, g) => t + g.items.filter(i => !i.repeat).reduce((s, i) => s + i.key.length, 0), 0) + ' characters to translate');
})();
