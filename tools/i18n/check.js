/* node tools/i18n/check.js [options]
   Checks the Hebrew dictionary of the talk (talk/i18n/he.js) against what the page and the scripts ask for.

     (a) coverage   every key talk.html asks for (tools/i18n/extract.js) and every KnotTalk.t('…') of the scripts has a
                    translation                                                                        ERROR
     (b) markup     for a key that holds markup: the same tags with the same attributes in key and value (a formula
                    placeholder <span data-tex="…"> is such a tag), not counting <span dir="ltr"> / <bdi> the
                    translation adds; tags closed in order; the same {placeholders}; no markup in a value that is
                    shown as plain text (an attribute, the title)                                      ERROR
     (c) numbers    every number of the key is in the value, as often (1D / 2D / 3D are words)          ERROR
     (d) direction  in values, what the bidi algorithm will reorder unless it is wrapped in <span dir="ltr"> or
                    U+2066 … U+2069: a digit, a space and %; an arrow, ×, a range with a dash, or = between Latin
                    letters or digits (a {placeholder} counts as a digit); a Latin name followed by a bracket
                    (S(u, v)); a number and its unit; also direction marks that do not pair           warning
     (e) stale      keys of the dictionary that neither the page nor any string of the scripts asks for  warning
     (f) keys       the slide master's step keys are the same in English and in Hebrew (the page is loaded in both,
                    with the dictionary), and nothing is asked for at load that the dictionary lacks    ERROR
     also           values equal to their English, values without a Hebrew letter, calls of KnotTalk.t whose
                    argument is not a literal (they cannot be checked here)                           notes

   Options
     --parts <dir|file>…   check these parts ({ "English": "Hebrew" } JSON) merged over the dictionary, a later one
                           winning (chrome.json last within a directory), and list where two parts disagree
     --only-parts          … without talk/i18n/he.js under them
     --dict <file.js>      another dictionary file
     --no-page             skip what needs a browser: the page's keys, (f)
     --all                 list every finding (default: the first 25 of a kind)
     --json <file>         write the findings as JSON
   Exit code 1 when there is an ERROR. Needs playwright for the page (see lib.js). */
const fs = require('fs'), path = require('path'), L = require('./lib');

const argv = process.argv.slice(2), opt = { parts: [], all: false, page: true, onlyParts: false, dict: path.join(L.ROOT, 'talk/i18n/he.js'), json: null };
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--parts') { while (argv[i + 1] && !/^--/.test(argv[i + 1])) opt.parts.push(argv[++i]); }
  else if (a === '--only-parts') opt.onlyParts = true;
  else if (a === '--dict') opt.dict = argv[++i];
  else if (a === '--no-page') opt.page = false;
  else if (a === '--all') opt.all = true;
  else if (a === '--json') opt.json = argv[++i];
  else if (a === '--help' || a === '-h') { console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0].replace(/^\/\* ?/, '')); process.exit(0); }
  else { console.error('unknown option ' + a); process.exit(2); }
}

/* ---------- reading a value ---------- */
const HEB = /[\u0590-\u05ff\ufb1d-\ufb4f]/, TAG = /<\/?[A-Za-z][A-Za-z0-9-]*(?:\s+[^\s"'<>\/=]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?)*\s*\/?>/g;
const VOID = { br: 1, hr: 1, img: 1, input: 1, wbr: 1 };
function tagsOf(s) {
  return (s.match(TAG) || []).map(t => {
    const close = t[1] === '/', name = /^<\/?([A-Za-z][A-Za-z0-9-]*)/.exec(t)[1].toLowerCase(), attrs = [];
    t.replace(/\s([^\s"'<>\/=]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g, (m, n, a, b, c) => { attrs.push(n.toLowerCase() + '=' + JSON.stringify(a !== undefined ? a : b !== undefined ? b : c !== undefined ? c : '')); return m; });
    return { close, name, sig: name + (attrs.length ? ' ' + attrs.sort().join(' ') : '') };
  });
}
const WRAP = { 'span dir="ltr"': 'span', 'bdi': 'bdi' };
const bag = list => { const m = {}; list.forEach(x => { m[x] = (m[x] || 0) + 1; }); return m; };
const text = s => s.replace(TAG, ' ').replace(/&(?:[a-z]+|#\d+|#x[0-9a-f]+);/gi, ' ');
const holes = s => (s.match(/\{\w+\}/g) || []);
function numbers(s) {
  s = text(s).replace(/\{\w+\}/g, ' ').replace(/(^|[^0-9.,])\d[-\u2011\u2010]?D(?![A-Za-z0-9])/g, '$1 ');        // 3D, 2\u2011D: a word
  return s.match(/\d+(?:[.,]\d+)*/g) || [];
}
/* the value with everything that is already left to right taken out: wrapped runs, key caps, formulas, the tags */
function loose(v) {
  let s = v, prev;
  do { prev = s; s = s.replace(/<(span|b|i|em|strong|small|a)\b[^>]*\b(?:dir="ltr"|class="[^"]*\bltr\b[^"]*")[^>]*>(?:(?!<\1\b)[^])*?<\/\1>/g, ' \ufffc '); } while (s !== prev);
  s = s.replace(/<(bdi|kbd|code)\b[^>]*>[^]*?<\/\1>/g, ' \ufffc ').replace(/<span\b[^>]*\bdata-tex=[^>]*>[^]*?<\/span>/g, ' \ufffc ');
  s = s.replace(/\u2066[^\u2066-\u2069]*\u2069/g, ' \ufffc ');
  return s.replace(TAG, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/\{\w+\}/g, '0');   // a {placeholder} is most often a number
}
const LINT = [
  [/\d\s+%/, 'a digit, a space and %: write 5.7% (the sign attached)'],
  [/[A-Za-z0-9%)°]\s*[←-↙⇄⇒⟶]\s*[A-Za-z0-9(]/, 'an arrow between Latin letters or digits'],
  [/[A-Za-z0-9)]\s*×\s*[A-Za-z0-9(]/, '× between Latin letters or digits'],
  [/\d\s*[–—]\s*\d/, 'a range of numbers with a dash'],
  [/[A-Za-z0-9)]\s*=\s*[A-Za-z0-9(−+-]/, '= between Latin letters or digits'],
  [/[A-Za-z0-9]\([^()]*\)/, 'a Latin name followed by a bracket (a formula written as text)'],
  [/\d\s+(?:mm|cm|ms|px|dB|fps|Hz|rms|RMS|min|[kKMm])(?![A-Za-z])/, 'a number and its unit']
];

(async () => {
  /* ---------- the dictionary ---------- */
  let dict = {}, conflicts = [], partsRead = [], from = {};
  if (!opt.onlyParts) { if (!fs.existsSync(opt.dict)) { console.error('no dictionary at ' + opt.dict); process.exit(2); } dict = L.dictionary(opt.dict); }
  const inFile = Object.keys(dict).length;
  if (opt.parts.length) { const r = L.readParts(opt.parts); Object.assign(dict, r.dict); conflicts = r.conflicts; partsRead = r.parts; from = r.from; }
  const has = k => Object.prototype.hasOwnProperty.call(dict, k);

  /* ---------- what is asked for ---------- */
  const kinds = {}, where = {}, note = (k, kind, at) => { (kinds[k] = kinds[k] || {})[kind] = 1; if (!where[k]) where[k] = at; };
  let browser = null, groups = [];
  if (opt.page) {
    browser = await L.playwright().chromium.launch();
    groups = await L.pageItems(browser);
    groups.forEach(g => g.items.forEach(i => note(i.key, i.kind === 'title' ? 'title' : i.kind[0] === '@' ? 'attr' : 'block', g.section + ' [' + i.kind + ']')));
  }
  const sc = L.scan();
  Object.keys(sc.keys).forEach(k => note(k, 't', sc.keys[k][0]));
  const asked = Object.keys(kinds);

  const out = { errors: {}, warnings: {}, notes: {} };
  const add = (level, name, key, msg) => { (out[level][name] = out[level][name] || []).push({ key, msg, at: where[key] || (from[key] ? 'part ' + from[key] : '') }); };

  /* (a) coverage */
  asked.forEach(k => { if (!has(k)) add('errors', '(a) no translation', k, ''); });

  Object.keys(dict).forEach(k => {
    const v = dict[k], kd = kinds[k] || {}, plainOnly = (kd.attr || kd.title) && !kd.block && !kd.t;
    /* (b) markup */
    const kt = tagsOf(k), vt = tagsOf(v), ko = bag(kt.filter(t => !t.close).map(t => t.sig)), vo = bag(vt.filter(t => !t.close).map(t => t.sig));
    const extra = {}, bad = [];
    Object.keys(WRAP).forEach(w => { const d = (vo[w] || 0) - (ko[w] || 0); if (d > 0) { extra[WRAP[w]] = (extra[WRAP[w]] || 0) + d; vo[w] -= d; } if (!vo[w]) delete vo[w]; if (!ko[w]) delete ko[w]; });
    Object.keys(ko).forEach(s => { if ((vo[s] || 0) < ko[s]) bad.push('missing <' + s + '>' + (ko[s] - (vo[s] || 0) > 1 ? ' ×' + (ko[s] - (vo[s] || 0)) : '')); });
    Object.keys(vo).forEach(s => { if ((ko[s] || 0) < vo[s]) bad.push('added <' + s + '>'); });
    const st = []; let order = true;
    vt.forEach(t => { if (VOID[t.name]) return; if (!t.close) st.push(t.name); else if (st.pop() !== t.name) order = false; });
    if (!order || st.length) bad.push('tags not closed in order');
    const kh = bag(holes(k)), vh = bag(holes(v));
    Object.keys(kh).forEach(h => { if (!vh[h]) bad.push('missing ' + h); });
    Object.keys(vh).forEach(h => { if (!kh[h]) bad.push('unknown ' + h); });
    if (plainOnly && vt.length) bad.push('markup in a value that is shown as plain text');
    else if (!kt.length && vt.length && kd.t && !kd.block) add('warnings', '(b) markup added to a string of a script (right only where it is set as HTML)', k, v);
    if (bad.length) add('errors', '(b) markup', k, bad.join('; '));
    /* (c) numbers */
    const kn = bag(numbers(k)), vn = bag(numbers(v)), lost = Object.keys(kn).filter(n => (vn[n] || 0) < kn[n]);
    if (lost.length) add('errors', '(c) numbers', k, 'not in the value: ' + lost.map(n => n + (kn[n] - (vn[n] || 0) > 1 ? ' ×' + (kn[n] - (vn[n] || 0)) : '')).join(', '));
    /* (d) direction */
    if (HEB.test(v)) {
      const s = loose(v), hits = LINT.filter(r => r[0].test(s)).map(r => r[1] + ' («' + (r[0].exec(s)[0]).trim() + '»)');
      if (hits.length) add('warnings', '(d) direction', k, hits.join('; ') + '  ← ' + v.slice(0, 110));
    }
    const li = (v.match(/[\u2066-\u2068]/g) || []).length, pd = (v.match(/\u2069/g) || []).length;
    if (li !== pd) add('warnings', '(d) direction', k, 'direction marks do not pair: ' + li + ' opened, ' + pd + ' closed');
    /* (e) stale */
    if (opt.page && !kinds[k] && !sc.literals.has(k)) add('warnings', '(e) stale: nothing asks for it', k, '');     // without the page its keys are unknown
    /* notes */
    if (v === k && /[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}/.test(text(k))) add('notes', 'same as the English', k, '');
    else if (!HEB.test(v) && v !== k) add('notes', 'no Hebrew letter in the value', k, v);
  });
  sc.dynamic.forEach(d => { (out.notes['KnotTalk.t with an argument that is not a literal'] = out.notes['KnotTalk.t with an argument that is not a literal'] || []).push({ key: d.text, msg: '', at: d.where }); });
  conflicts.forEach(c => { (out.warnings['two parts disagree'] = out.warnings['two parts disagree'] || []).push({ key: c.key, msg: c.kept + ' wins over ' + c.dropped + ': ' + c.is.slice(0, 70) + '  ≠  ' + c.was.slice(0, 70), at: '' }); });

  /* (f) the page in both languages */
  let runtime = null;
  if (opt.page) {
    const load = async lang => {
      const page = await browser.newPage(), errs = [];
      await page.route('**/*', r => {
        const u = r.request().url();
        if (/talk\/i18n\/\w+\.js(\?|$)/.test(u)) return r.fulfill({ status: 200, contentType: 'text/javascript', body: L.dictSource(dict) });
        if (/talk\/sims\/|talk\/vendor\/three|talk\/(dtu|audio|he)\/|talk\/(dtu-data|face-data|scenes|gl|hands|meshlist|meshio|terrain|ink)\.js/.test(u)) return r.fulfill({ status: 200, contentType: 'text/javascript', body: '' });
        return /^file:/.test(u) ? r.continue() : r.abort();
      });
      page.on('pageerror', e => errs.push(String(e && e.message || e)));
      await page.goto(L.fileUrl('talk.html', '?lang=' + lang));
      await page.waitForFunction(() => window.KnotTalk && KnotTalk.deck && KnotTalk.master, null, { timeout: 20000 });
      const r = await page.evaluate(() => ({ keys: KnotTalk.deck.steps.map(KnotTalk.deck.key), titles: KnotTalk.deck.steps.map(KnotTalk.deck.title), missing: KnotTalk.i18n.missing, dir: document.documentElement.dir }));
      await page.close(); r.errs = errs; return r;
    };
    const en = await load('en'), he = await load('he');
    runtime = { steps: en.keys.length, differ: 0, missing: Object.keys(he.missing).length };
    en.keys.forEach((k, i) => { if (he.keys[i] !== k) { runtime.differ++; add('errors', '(f) a step’s key differs between the languages', k, 'in Hebrew: ' + he.keys[i]); } });
    if (en.keys.length !== he.keys.length) add('errors', '(f) a step’s key differs between the languages', '(count)', en.keys.length + ' steps in English, ' + he.keys.length + ' in Hebrew');
    en.errs.concat(he.errs).forEach(e => add('errors', '(f) the page throws', e, ''));
    if (he.dir !== 'rtl') add('errors', '(f) the page throws', '<html dir>', 'the Hebrew page is not right to left');
    Object.keys(he.missing).forEach(k => { if (has(k)) return; if (!kinds[k]) add('errors', '(a) no translation', k, 'asked at load (' + he.missing[k] + ')'); });
    runtime.untranslatedTitles = he.titles.filter((t, i) => t === en.titles[i] && /[A-Za-z]{3,}/.test(t) && !HEB.test(t)).length;
    await browser.close();
  }

  /* ---------- report ---------- */
  const count = o => Object.keys(o).reduce((t, k) => t + o[k].length, 0), short = s => { s = String(s).replace(/\s+/g, ' '); return s.length > 96 ? s.slice(0, 93) + '…' : s; };
  const show = (title, o) => Object.keys(o).sort().forEach(name => {
    const list = o[name]; console.log('\n' + title + ' ' + name + ': ' + list.length);
    list.slice(0, opt.all ? list.length : 25).forEach(x => console.log('  ' + JSON.stringify(short(x.key)) + (x.msg ? '\n      ' + (opt.all ? x.msg : short(x.msg)) : '') + (x.at ? '   [' + x.at + ']' : '')));
    if (!opt.all && list.length > 25) console.log('  … ' + (list.length - 25) + ' more (--all)');
  });
  const pageKeys = groups.reduce((t, g) => t + g.items.filter(i => !i.repeat).length, 0);
  console.log('dictionary: ' + Object.keys(dict).length + ' entries' + (opt.parts.length ? ' (' + (opt.onlyParts ? 'parts only' : inFile + ' in ' + path.relative(L.ROOT, opt.dict)) + '; parts: ' + partsRead.map(p => p.name + ' ' + p.entries).join(', ') + ')' : ' in ' + path.relative(L.ROOT, opt.dict)));
  console.log('asked for: ' + asked.length + ' keys (' + (opt.page ? pageKeys + ' by the page' : 'page not read') + ', ' + Object.keys(sc.keys).length + ' by ' + sc.files.length + ' scripts), ' + asked.filter(has).length + ' translated');
  if (runtime) console.log('page: ' + runtime.steps + ' steps, ' + runtime.differ + ' keys that differ between the languages, ' + runtime.missing + ' strings asked at load without a translation, ' + runtime.untranslatedTitles + ' step titles still English');
  show('ERROR', out.errors); show('warning', out.warnings); show('note', out.notes);
  const e = count(out.errors), w = count(out.warnings);
  console.log('\n' + (e ? e + ' error' + (e > 1 ? 's' : '') : 'no errors') + ', ' + w + ' warning' + (w === 1 ? '' : 's') + ', ' + count(out.notes) + ' note' + (count(out.notes) === 1 ? '' : 's'));
  if (opt.json) fs.writeFileSync(opt.json, JSON.stringify({ entries: Object.keys(dict).length, asked: asked.length, runtime, errors: out.errors, warnings: out.warnings, notes: out.notes }, null, 1));
  process.exit(e ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
