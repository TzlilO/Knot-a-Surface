#!/usr/bin/env node
/* Checks the teleprompter scripts (talk/script/<lang>.js) against the steps of talk.html. See tools/script/README.md.
   For every language that has a file: ERROR for a step key without a passage, a passage whose key no step has, an empty
   passage, HTML tags in a passage; then words per section for the long and the short cut against the section's minutes.
   Exit code 1 on an error. Playwright: /opt/node-tools/node_modules/playwright (or NODE_PATH). */
'use strict';
const fs = require('fs'), path = require('path'), vm = require('vm');
const ROOT = path.resolve(__dirname, '../..'), DIR = process.argv[2] ? path.resolve(process.argv[2]) : path.join(ROOT, 'talk/script');   // an optional directory, for tests
let chromium;
try { chromium = require('playwright').chromium; } catch (e) { chromium = require('/opt/node-tools/node_modules/playwright').chromium; }

const files = fs.existsSync(DIR) ? fs.readdirSync(DIR).filter(f => /^[a-z]{2}\.js$/.test(f)).sort() : [];
if (!files.length) { console.log('no talk/script/<lang>.js yet: nothing to check'); process.exit(0); }

const words = t => t.split(/\n/).filter(l => l.trim() && !/^\s*\[.*\]\s*$/.test(l)).join(' ').split(/\s+/).filter(Boolean).length;

(async () => {
  const b = await chromium.launch({ args: ['--no-sandbox', '--headless=new', '--disable-gpu'] });
  const pg = await b.newPage();
  await pg.goto('file://' + path.join(ROOT, 'talk.html') + '?master&lang=en');     // the presenter view mounts no simulation
  await pg.waitForFunction(() => window.KnotTalk && KnotTalk.deck && KnotTalk.deck.steps.length);
  const steps = await pg.evaluate(() => KnotTalk.deck.steps.map(n => {
    const D = KnotTalk.deck, s = D.section(n);
    return { key: D.key(n), sec: (s.dataset.rail || '') + ' · ' + (s.dataset.title || ''), min40: +s.dataset.min40 || 0, min20: +s.dataset.min20 || 0,
             long: D.inCut(n, 40), short: D.inCut(n, 20), off: !D.defaultOn(n) };
  }));
  await b.close();
  let errors = 0;
  const err = m => { errors++; console.log('ERROR ' + m); };
  for (const f of files) {
    const lang = f.slice(0, 2), sb = {};
    sb.window = sb;                                                             // a browser's global: window.KnotTalk is KnotTalk
    try { vm.runInNewContext(fs.readFileSync(path.join(DIR, f), 'utf8'), sb, { filename: f }); } catch (e) { err(f + ': does not run: ' + e.message); continue; }
    const S = sb.KnotTalk && sb.KnotTalk.script && sb.KnotTalk.script[lang];
    if (!S || typeof S !== 'object') { err(f + ': no KnotTalk.script.' + lang); continue; }
    console.log('\n== ' + f + ' (' + Object.keys(S).length + ' passages, ' + steps.length + ' steps)');
    const known = new Set(steps.map(s => s.key));
    steps.forEach(s => { if (!(s.key in S)) err(lang + ': step without a passage: "' + s.key + '"' + (s.off ? ' (off by default)' : '')); });
    Object.keys(S).forEach(k => {
      const v = S[k];
      if (!known.has(k)) err(lang + ': passage whose key no step has: "' + k + '"');
      if (typeof v !== 'string' || !v.trim()) err(lang + ': empty passage: "' + k + '"');
      else if (/<\/?[a-zA-Z][^>]*>/.test(v)) err(lang + ': HTML tags in the passage of "' + k + '"');
    });
    const sec = new Map();
    steps.forEach(s => {
      const w = typeof S[s.key] === 'string' ? words(S[s.key]) : 0;
      if (!sec.has(s.sec)) sec.set(s.sec, { min40: s.min40, min20: s.min20, w40: 0, w20: 0 });
      const r = sec.get(s.sec); if (s.long) r.w40 += w; if (s.short) r.w20 += w;
    });
    const pad = (x, n) => String(x).padEnd(n), num = (x, n) => String(x).padStart(n);
    console.log(pad('section', 46) + num('min40', 6) + num('words', 7) + num('w/min', 7) + '   ' + num('min20', 6) + num('words', 7) + num('w/min', 7));
    let t = { m40: 0, w40: 0, m20: 0, w20: 0 };
    sec.forEach((r, k) => {
      const a = r.min40 ? Math.round(r.w40 / r.min40) : '–', c = r.min20 ? Math.round(r.w20 / r.min20) : '–';
      console.log(pad(k.slice(0, 45), 46) + num(r.min40, 6) + num(r.w40, 7) + num(a, 7) + '   ' + num(r.min20, 6) + num(r.w20, 7) + num(c, 7));
      t.m40 += r.min40; t.w40 += r.w40; t.m20 += r.min20; t.w20 += r.w20;
    });
    console.log(pad('total', 46) + num(t.m40, 6) + num(t.w40, 7) + num(t.m40 ? Math.round(t.w40 / t.m40) : '–', 7) + '   ' + num(t.m20, 6) + num(t.w20, 7) + num(t.m20 ? Math.round(t.w20 / t.m20) : '–', 7));
    console.log('(words per minute: Hebrew speech runs about 110-140 for a talk; stage directions are not counted)');
  }
  console.log(errors ? '\n' + errors + ' error(s)' : '\nno errors');
  process.exit(errors ? 1 : 0);
})();
