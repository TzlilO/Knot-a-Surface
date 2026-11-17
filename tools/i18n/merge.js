/* node tools/i18n/merge.js <part.json | dir>… [--out <file.js>] [--keep]
   Writes the Hebrew dictionary (talk/i18n/he.js, or --out) from translators' parts: JSON files of the form
   { "English, exactly as the page or the module has it": "עברית" }. Parts are taken in the order given and a later
   part wins; inside a directory the order is by name with chrome.json last (the page's own chrome: the keys help,
   the HUD, the slide master). Where two parts disagree the one that lost is reported.
   --keep   start from what the dictionary file holds now (the parts win), instead of from nothing
   The direction marks nobody can see (U+2066 … U+2069 and their kind) are written as \uXXXX escapes.
   Run tools/i18n/check.js afterwards. */
const fs = require('fs'), path = require('path'), L = require('./lib');
const argv = process.argv.slice(2), parts = [];
let out = path.join(L.ROOT, 'talk/i18n/he.js'), keep = false;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--out') out = argv[++i]; else if (argv[i] === '--keep') keep = true;
  else if (/^--/.test(argv[i])) { console.error('unknown option ' + argv[i]); process.exit(2); } else parts.push(argv[i]);
}
if (!parts.length) { console.error('usage: node tools/i18n/merge.js <part.json | dir>… [--out <file.js>] [--keep]'); process.exit(2); }
const r = L.readParts(parts), groups = [], dict = {};
if (keep && fs.existsSync(out)) {
  const old = L.dictionary(out), left = Object.keys(old).filter(k => !Object.prototype.hasOwnProperty.call(r.dict, k));
  left.forEach(k => { dict[k] = old[k]; });
  if (left.length) groups.push({ name: 'kept from the file as it was', keys: left });
}
r.parts.forEach(p => {
  const keys = Object.keys(r.dict).filter(k => r.from[k] === p.name);
  keys.forEach(k => { dict[k] = r.dict[k]; });
  if (keys.length) groups.push({ name: p.name + ' · ' + keys.length, keys });
});
fs.writeFileSync(out, L.dictSource(dict, groups));
console.log(Object.keys(dict).length + ' entries -> ' + path.relative(process.cwd(), out) + '  (' + groups.map(g => g.name).join(', ') + ')');
r.conflicts.forEach(c => console.log('  two parts disagree on ' + JSON.stringify(c.key.length > 70 ? c.key.slice(0, 67) + '…' : c.key) + ': ' + c.kept + ' wins over ' + c.dropped));
