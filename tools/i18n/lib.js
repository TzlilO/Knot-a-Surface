/* Shared by tools/i18n/extract.js, check.js and merge.js: where the keys of a translation come from, and the
   dictionary. Node only; the page's keys need playwright (the one the talk's tests use, or NODE_PATH).

     pageItems()        what talk.html asks for: every block of copy, every translatable attribute, the title, as
                        talk/i18n.js lists them in the page itself (KnotTalk.i18n.items), grouped by section
     scan()             what the scripts ask for: every KnotTalk.t('…') / KT.t('…') / T('…') call with a literal
                        first argument, where T is a name the file gave KnotTalk.t (var T = KT.t; a function that
                        only passes its arguments on to it). Also the calls whose argument is not a literal
                        (reported, they cannot be checked from here) and every string literal of the sources
                        (a dictionary key that is none of them is stale)
     dictionary(file)   KnotTalk.i18nData.he, read by running the dictionary file
     readParts(paths)   parts ({ "English": "Hebrew" } JSON files, or directories of them) merged in order, a later
                        part winning, chrome.json last within a directory; with the conflicts found */
const fs = require('fs'), path = require('path'), vm = require('vm');
const ROOT = path.resolve(__dirname, '..', '..');
function playwright() { try { return require('playwright'); } catch (e) { return require('/opt/node-tools/node_modules/playwright'); } }
const fileUrl = (rel, query) => 'file://' + path.join(ROOT, rel) + (query || '');

/* ---------- the page ---------- */
/* a page of talk.html with only talk/i18n.js run (and the dictionary given, as Hebrew, if any): the copy as written */
async function bare(browser, dict) {
  const page = await browser.newPage();
  await page.route('**/*', r => {
    const u = r.request().url();
    if (/talk\/i18n\/\w+\.js(\?|$)/.test(u)) return r.fulfill({ status: 200, contentType: 'text/javascript', body: dict ? dictSource(dict) : '' });
    return /\.js(\?|$)/.test(u) && !/talk\/i18n\.js/.test(u) ? r.abort() : /^file:/.test(u) ? r.continue() : r.abort();
  });
  return page;
}
async function pageItems(browser) {
  const own = !browser; if (own) browser = await playwright().chromium.launch();
  const page = await bare(browser, null);
  await page.goto(fileUrl('talk.html', '?lang=en'));
  const groups = await page.evaluate(() => {
    const I = KnotTalk.i18n, groups = [], seen = {};
    const name = el => { const s = el && el.closest('section[data-rail]'); return s ? s.getAttribute('data-rail') + ' · ' + s.getAttribute('data-title') : 'page'; };
    I.items(document.body).forEach(it => {
      const g = name(it.el); let o = groups.find(x => x.section === g);
      if (!o) groups.push(o = { section: g, items: [] });
      o.items.push({ key: it.key, kind: it.kind, repeat: !!seen[it.key] }); seen[it.key] = 1;
    });
    return groups;
  });
  await page.close(); if (own) await browser.close();
  return groups;
}

/* ---------- the scripts ---------- */
const SOURCES = ['talk', 'talk/sims', 'talk/he'];                      // directories whose .js files are scanned (not below them)
const DATA = ['talk/dtu', 'talk/audio'];                                // data: no calls, but strings a script may pass on to KnotTalk.t
const NOT_SOURCE = /(^|\/)(dtu-data|face-data)\.js$/;                   // numbers only
function sources(dirs) {
  const out = [];
  (dirs || SOURCES).forEach(d => {
    const dir = path.join(ROOT, d);
    if (!fs.existsSync(dir)) return;
    fs.readdirSync(dir).filter(f => /\.js$/.test(f)).sort().forEach(f => {
      const rel = d + '/' + f, full = path.join(dir, f);
      if (!NOT_SOURCE.test(rel) && fs.statSync(full).size < 1e6) out.push(rel);
    });
  });
  return out;
}
/* a small JavaScript tokenizer: enough to tell strings, comments and regular expressions apart */
function tokens(src) {
  const out = [], n = src.length; let i = 0, line = 1, last = null;
  const push = (type, value, at) => { last = { type, value, line: at }; out.push(last); };
  const regexOk = () => !last || (last.type === 'p' && !/[)\]}]/.test(last.value)) ||
    (last.type === 'id' && /^(return|typeof|case|in|of|delete|void|throw|new|else|do|instanceof|yield|await)$/.test(last.value));
  function str(q) {                                   // src[i] is the opening quote; returns the string's value
    let s = '', j = i + 1;
    for (; j < n && src[j] !== q; j++) {
      const c = src[j];
      if (c === '\n') line++;
      if (c !== '\\') { s += c; continue; }
      const e = src[++j];
      if (e === 'n') s += '\n'; else if (e === 't') s += '\t'; else if (e === 'r') s += '\r'; else if (e === 'b') s += '\b'; else if (e === 'f') s += '\f'; else if (e === 'v') s += '\v';
      else if (e === '0' && !/[0-9]/.test(src[j + 1])) s += '\0';
      else if (e === 'x') { s += String.fromCharCode(parseInt(src.substr(j + 1, 2), 16)); j += 2; }
      else if (e === 'u' && src[j + 1] === '{') { const k = src.indexOf('}', j); s += String.fromCodePoint(parseInt(src.slice(j + 2, k), 16)); j = k; }
      else if (e === 'u') { s += String.fromCharCode(parseInt(src.substr(j + 1, 4), 16)); j += 4; }
      else if (e === '\r') { if (src[j + 1] === '\n') j++; line++; } else if (e === '\n') line++;        // a line continuation
      else s += e;
    }
    i = j + 1; return s;
  }
  function template() {                               // src[i] is the opening backtick; returns { raw, plain }
    let j = i + 1, plain = true, depth = 0; const from = j;
    for (; j < n; j++) {
      const c = src[j];
      if (c === '\n') line++;
      if (c === '\\') { j++; continue; }
      if (!depth && c === '`') break;
      if (c === '$' && src[j + 1] === '{') { plain = false; depth++; j++; continue; }
      if (depth && c === '{') depth++; else if (depth && c === '}') depth--;
    }
    const raw = src.slice(from, j); i = j + 1; return { raw, plain };
  }
  while (i < n) {
    const c = src[i], at = line;
    if (c === '\n') { line++; i++; continue; }
    if (c === ' ' || c === '\t' || c === '\r' || c === '\f' || c === '\v' || c === '\u00a0' || c === '\ufeff') { i++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < n && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { const e = src.indexOf('*/', i + 2), end = e < 0 ? n : e + 2; line += (src.slice(i, end).match(/\n/g) || []).length; i = end; continue; }
    if (c === '"' || c === "'") { push('str', str(c), at); continue; }
    if (c === '`') { const t = template(); push(t.plain ? 'str' : 'tmpl', t.raw, at); continue; }
    if (c === '/' && regexOk()) {
      let j = i + 1, cls = false;
      for (; j < n && src[j] !== '\n'; j++) { const d = src[j]; if (d === '\\') { j++; continue; } if (d === '[') cls = true; else if (d === ']') cls = false; else if (d === '/' && !cls) break; }
      j++; while (j < n && /[a-z]/i.test(src[j])) j++;
      push('re', src.slice(i, j), at); i = j; continue;
    }
    if (/[A-Za-z_$]/.test(c)) { let j = i + 1; while (j < n && /[\w$]/.test(src[j])) j++; push('id', src.slice(i, j), at); i = j; continue; }
    if (/[0-9]/.test(c)) { let j = i + 1; while (j < n && /[\w.]/.test(src[j])) j++; push('num', src.slice(i, j), at); i = j; continue; }
    push('p', c, at); i++;
  }
  return out;
}
/* the calls of KnotTalk.t in one file: { keys: [{ key, line, cond }], dynamic: [{ line, text }], literals: Set } */
function scanSource(src) {
  const T = tokens(src), keys = [], dynamic = [], literals = new Set(), objs = { KnotTalk: 1, KT: 1 }, fns = {}, inside = new Set();
  const is = (k, type, value) => T[k] && T[k].type === type && (value === undefined || T[k].value === value);
  T.forEach(t => { if (t.type === 'str') literals.add(t.value); });
  /* k points at an object that is KnotTalk (maybe after `window.`), followed by .t : returns the index after `t`, else -1 */
  function tAt(k) {
    if (is(k, 'id', 'window') && is(k + 1, 'p', '.')) k += 2;
    return T[k] && T[k].type === 'id' && objs[T[k].value] && is(k + 1, 'p', '.') && is(k + 2, 'id', 't') ? k + 3 : -1;
  }
  for (let pass = 0; pass < 2; pass++) for (let k = 0; k < T.length; k++) {
    if (!is(k, 'id')) continue;
    /* X = [window.]KnotTalk (not KnotTalk.something): X is the object too */
    if (is(k + 1, 'p', '=') && !is(k + 2, 'p', '=')) {
      let j = k + 2; if (is(j, 'id', 'window') && is(j + 1, 'p', '.')) j += 2;
      if (T[j] && T[j].type === 'id' && objs[T[j].value] && !is(j + 1, 'p', '.')) objs[T[k].value] = 1;
      const e = tAt(k + 2);
      if (e > 0 && !is(e, 'p', '(')) fns[T[k].value] = 1;                    // X = KT.t
      /* X = function (a, …) { return KT.t(a, …) }   or   X = (a, …) => KT.t(a, …) */
      let b = k + 2; if (is(b, 'id', 'function')) b++;
      if (is(b, 'p', '(') && is(b + 1, 'id')) {
        const a = T[b + 1].value; let c = b; while (T[c] && !is(c, 'p', ')')) c++;
        c++; if (is(c, 'p', '=') && is(c + 1, 'p', '>')) c += 2;
        if (is(c, 'p', '{')) c++; if (is(c, 'id', 'return')) c++;
        const e2 = tAt(c);
        if (e2 > 0 && is(e2, 'p', '(') && is(e2 + 1, 'id', a)) { fns[T[k].value] = 1; inside.add(e2); }
      }
    }
    /* function X(a, …) { return KT.t(a, …) } */
    if (T[k].value === 'function' && is(k + 1, 'id') && is(k + 2, 'p', '(') && is(k + 3, 'id')) {
      const a = T[k + 3].value; let c = k + 2; while (T[c] && !is(c, 'p', ')')) c++;
      c++; if (is(c, 'p', '{')) c++; if (is(c, 'id', 'return')) c++;
      const e3 = tAt(c);
      if (e3 > 0 && is(e3, 'p', '(') && is(e3 + 1, 'id', a)) { fns[T[k + 1].value] = 1; inside.add(e3); }
    }
  }
  const lit = seg => seg.length && seg.every((t, i) => i % 2 ? t.type === 'p' && t.value === '+' : t.type === 'str') ? seg.filter(t => t.type === 'str').map(t => t.value).join('') : null;
  for (let k = 0; k < T.length; k++) {
    let open = -1;
    if (is(k, 'id') && !is(k - 1, 'p', '.')) {
      const e = tAt(k);
      if (e > 0 && is(e, 'p', '(')) open = e;
      else if (fns[T[k].value] && is(k + 1, 'p', '(') && !is(k - 1, 'id', 'function')) open = k + 1;
    }
    if (open < 0 || inside.has(open)) continue;
    /* the first argument: up to the comma or the bracket that closes the call */
    const arg = []; let depth = 0, j = open + 1;
    for (; j < T.length; j++) {
      const t = T[j];
      if (t.type === 'p') { if (/[([{]/.test(t.value)) depth++; else if (/[)\]}]/.test(t.value)) { if (!depth) break; depth--; } else if (t.value === ',' && !depth) break; }
      arg.push(t);
    }
    const whole = lit(arg);
    if (whole !== null) { keys.push({ key: whole, line: T[k].line, cond: false }); continue; }
    /* a choice between literals: cond ? 'a' : 'b' (nested too). The piece before a ? is a condition */
    const segs = [[]], ends = []; depth = 0;
    arg.forEach(t => {
      if (t.type === 'p' && /[([{]/.test(t.value)) depth++; else if (t.type === 'p' && /[)\]}]/.test(t.value)) depth--;
      if (!depth && t.type === 'p' && (t.value === '?' || t.value === ':')) { ends.push(t.value); segs.push([]); } else segs[segs.length - 1].push(t);
    });
    const branches = segs.filter((s, i) => ends[i] !== '?').map(lit);
    if (ends.length && branches.every(b => b !== null)) { branches.forEach(b => keys.push({ key: b, line: T[k].line, cond: true })); continue; }
    dynamic.push({ line: T[k].line, text: arg.map(t => t.type === 'str' ? JSON.stringify(t.value) : t.value).join(' ').slice(0, 90) });
  }
  return { keys, dynamic, literals, names: Object.keys(fns) };
}
function scan() {
  const keys = {}, dynamic = [], literals = new Set(), files = sources();
  files.forEach(rel => {
    const r = scanSource(fs.readFileSync(path.join(ROOT, rel), 'utf8'));
    r.keys.forEach(k => { (keys[k.key] = keys[k.key] || []).push(rel + ':' + k.line); });
    r.dynamic.forEach(d => dynamic.push({ where: rel + ':' + d.line, text: d.text }));
    r.literals.forEach(l => literals.add(l));
  });
  sources(DATA).forEach(rel => tokens(fs.readFileSync(path.join(ROOT, rel), 'utf8')).forEach(t => { if (t.type === 'str' && t.value.length < 400) literals.add(t.value); }));
  return { keys, dynamic, literals, files };
}

/* ---------- the dictionary ---------- */
function dictionary(file) {
  const w = { KnotTalk: {} };
  vm.runInNewContext(fs.readFileSync(file, 'utf8'), { window: w }, { filename: file });
  return ((w.KnotTalk || {}).i18nData || {}).he || {};
}
function readParts(paths) {
  const files = [];
  paths.forEach(p => {
    if (fs.statSync(p).isDirectory()) fs.readdirSync(p).filter(f => /\.json$/.test(f)).sort((a, b) => (a === 'chrome.json') - (b === 'chrome.json') || a.localeCompare(b)).forEach(f => files.push(path.join(p, f)));
    else files.push(p);
  });
  const dict = {}, from = {}, conflicts = [], parts = [];
  files.forEach(f => {
    const o = JSON.parse(fs.readFileSync(f, 'utf8')), name = path.basename(f); let n = 0;
    if (!o || typeof o !== 'object' || Array.isArray(o)) throw new Error(f + ': a part is one object, { "English": "Hebrew" }');
    Object.keys(o).forEach(k => {
      if (typeof o[k] !== 'string') throw new Error(f + ': the value of ' + JSON.stringify(k.slice(0, 60)) + ' is not a string');
      if (Object.prototype.hasOwnProperty.call(dict, k) && dict[k] !== o[k]) conflicts.push({ key: k, kept: name, dropped: from[k], was: dict[k], is: o[k] });
      dict[k] = o[k]; from[k] = name; n++;
    });
    parts.push({ file: f, name, entries: n });
  });
  return { dict, from, conflicts, parts };
}
/* a string as JavaScript source, with the characters nobody can see written out (direction marks, odd spaces) */
function js(s) { return JSON.stringify(s).replace(/[\u00ad\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]/g, c => '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4)); }
function dictSource(dict, groups) {
  const head = '/* Knot-a-Surface talk — Hebrew. KnotTalk.i18nData.he = { "English, exactly as the page or the module has it": "עברית" }.\n' +
    '   See talk/i18n.js for how a key is formed, and tools/i18n/ for the checks (coverage, markup, numbers) and for\n' +
    '   merge.js, which writes this file from the translators\' parts. Direction marks are written as \\u2066 … \\u2069. */\n';
  let body = '';
  (groups || [{ name: null, keys: Object.keys(dict) }]).forEach(g => {
    if (g.name) body += '  /* ' + g.name + ' */\n';
    g.keys.forEach(k => { body += '  ' + js(k) + ': ' + js(dict[k]) + ',\n'; });
  });
  return head + '(function () {\n  var KT = window.KnotTalk = window.KnotTalk || {}; KT.i18nData = KT.i18nData || {};\n' +
    '  var D = KT.i18nData.he = KT.i18nData.he || {};\n  Object.assign(D, {\n' + body.replace(/,\n$/, '\n') + '  });\n})();\n';
}

module.exports = { ROOT, playwright, fileUrl, bare, pageItems, sources, tokens, scanSource, scan, dictionary, readParts, js, dictSource };
