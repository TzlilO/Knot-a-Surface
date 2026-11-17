/* node tools/i18n/extract.js [out.json] [--modules]
   Lists what talk.html asks a translation for: every block of copy, every translatable attribute and the title, with
   the key the page will look up (talk/i18n.js: KnotTalk.i18n.items), grouped by section. The page is loaded with every
   script but talk/i18n.js held back, so the copy is as written.
   --modules adds a last group, "scripts": the keys the scripts ask for (KnotTalk.t('…') with a literal argument,
   found by tools/i18n/lib.js scan()), with the file and line of the first call.
   Needs playwright (the same one the talk's tests use: /opt/node-tools/node_modules/playwright, or NODE_PATH). */
const fs = require('fs'), L = require('./lib');
(async () => {
  const args = process.argv.slice(2), modules = args.indexOf('--modules') >= 0, out = args.filter(a => !/^--/.test(a))[0];
  const groups = await L.pageItems();
  if (modules) {
    const seen = {}; groups.forEach(g => g.items.forEach(i => { seen[i.key] = 1; }));
    const s = L.scan();
    groups.push({ section: 'scripts', items: Object.keys(s.keys).map(k => ({ key: k, kind: 't ' + s.keys[k][0], repeat: !!seen[k] })) });
  }
  const n = groups.reduce((t, g) => t + g.items.length, 0), uniq = groups.reduce((t, g) => t + g.items.filter(i => !i.repeat).length, 0);
  if (out) fs.writeFileSync(out, JSON.stringify(groups, null, 1));
  console.log(groups.map(g => g.section + ': ' + g.items.filter(i => !i.repeat).length + ' new of ' + g.items.length).join('\n'));
  console.log(n + ' items, ' + uniq + ' distinct keys, ' + groups.reduce((t, g) => t + g.items.filter(i => !i.repeat).reduce((s, i) => s + i.key.length, 0), 0) + ' characters to translate');
})();
