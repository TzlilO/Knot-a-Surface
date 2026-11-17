/* Knot-a-Surface talk — languages (English, Hebrew).
   The talk is written in English: talk.html holds the copy, the modules hold their own strings. A translation is a
   dictionary keyed by the English itself, talk/i18n/<lang>.js -> KnotTalk.i18nData[lang] = { "English": "translation" },
   so a sentence whose English has changed falls back to English instead of showing an outdated translation.
   Loaded first, in <head>, followed by the dictionaries. A page without this file (the lab pages, the landing page,
   talk-he.html) gets English from the fallback KnotTalk.t in talk/kit.js.

   Which language    Decided once, before anything is drawn: ?lang=he|en in the address, else the saved choice, else
                     English. A language in the address is saved too. KnotTalk.lang = 'en' | 'he', KnotTalk.rtl = true
                     for Hebrew; <html lang>, <html dir> and <html data-lang> are set here (data-lang marks a page
                     whose language this file set: talk/kit.css asks for it, so its right-to-left rules leave
                     talk-he.html alone).
   Switching         i18n.setLang(l) / i18n.toggle() reload the page: every module reads its strings once, when it is
                     created, so nothing on screen can be left in the other language. Hebrew goes into the address
                     (?lang=he: the link can be shared, and it holds without storage), English leaves it. Before the
                     reload a `kt-lang` event on document, detail { lang, was, keep }: what a listener puts into `keep`
                     (the deck: its cut, parts, clock, step) is back after the reload as i18n.kept, once.
   KnotTalk.t(en, vars)   that string in the current language, the English itself when there is no translation, with
                     {name} filled in from vars. Keys are exact strings. In Hebrew the result carries its own
                     direction: it comes wrapped in a bidi isolate (U+2067 ... U+2069 when it has Hebrew letters,
                     U+2066 ... U+2069 otherwise; zero width), so it reads right on a canvas, in SVG text and inside a
                     left-to-right stage as well as in the page. Markup that holds blocks, or is nothing but
                     elements, is left alone: its direction is the element's it goes into. So: one sentence, one call
                     with a template (never pieces glued around a number); and a result is for the screen, not for a
                     key, an id or a file name (i18n.strip takes the marks off).
   The page's copy   i18n.apply() swaps every BLOCK (the outermost element that holds text of its own; its key is its
                     innerHTML with whitespace collapsed), the attributes in ATTRS and the document's title.
                     data-i18n="off" keeps an element and what is in it out. A block without a translation stays
                     English and is marked dir="ltr" lang="en", so that it still reads. talk.html calls apply() right
                     after its copy, before the heavy scripts load (no flash of English); deck.js calls it again
                     before it reads the page (blocks already swapped are skipped).
   English, kept     i18n.source(el) = a detached copy of el as the page wrote it, i18n.en(el) its text,
                     i18n.attrEn(el, attr) an attribute's English: the slide master's keys stay English in both
                     languages (talk/deck.js).
   i18n.missing      { key: where it was asked: 't' | 'block' | '@attr' | 'title' }, what was asked for and not found.
   Checks and how to add a sentence: tools/i18n/README.md. */
(function () {
  'use strict';
  var KT = window.KnotTalk = window.KnotTalk || {};
  var LANGS = { en: { dir: 'ltr', name: 'English', short: 'EN' }, he: { dir: 'rtl', name: 'עברית', short: 'עב' } };
  var BASE = 'en', STORE = 'kt-lang', KEEP = 'kt-keep', NAMED = '|kt-keep|';
  var ATTRS = ['data-title', 'title', 'aria-label', 'placeholder'];
  var SKIP = { SCRIPT: 1, STYLE: 1, CANVAS: 1, TEMPLATE: 1, NOSCRIPT: 1, svg: 1, SVG: 1 };
  function own(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }

  /* ---------- storage: every access guarded (private windows, some file:// setups have none) ---------- */
  function get(store, k) { try { return window[store].getItem(k); } catch (e) { return null; } }
  function put(store, k, v) { try { window[store].setItem(k, v); return window[store].getItem(k) === v; } catch (e) { return false; } }
  function drop(store, k) { try { window[store].removeItem(k); } catch (e) { /* nothing to drop */ } }

  /* ---------- which language ---------- */
  function saved() { var v = get('localStorage', STORE) || get('sessionStorage', STORE); return v && own(LANGS, v) ? v : null; }
  function save(l) { return put('localStorage', STORE, l) || put('sessionStorage', STORE, l); }   // true: it will be read back
  var q = /[?&]lang=([\w-]+)/.exec(window.location.search || ''), asked = q && own(LANGS, q[1]) ? q[1] : null;
  var lang = asked || saved() || BASE;
  if (asked) save(asked);
  KT.lang = lang; KT.rtl = LANGS[lang].dir === 'rtl';
  var html = document.documentElement;
  html.lang = lang; html.dir = LANGS[lang].dir; html.setAttribute('data-lang', lang);

  /* the address of this page (or of `from`) in language l. explicit: with ?lang= in it */
  function url(l, explicit, from) {
    var u = from || window.location.href, i = u.indexOf('#'), hash = i < 0 ? '' : u.slice(i), rest = i < 0 ? u : u.slice(0, i);
    var j = rest.indexOf('?'), path = j < 0 ? rest : rest.slice(0, j);
    var qs = (j < 0 ? '' : rest.slice(j + 1)).split('&').filter(function (p) { return p && !/^lang(=|$)/.test(p); });
    if (explicit) qs.push('lang=' + l);
    return path + (qs.length ? '?' + qs.join('&') : '') + hash;
  }

  /* ---------- what a switch carries over its reload ----------
     One-shot and short-lived: read here, then gone. sessionStorage where there is one, else the tail of window.name
     (put back as it was: a window may have been opened by name). */
  var kept = {};
  (function () {
    var raw = get('sessionStorage', KEEP), name = '', i, o;
    drop('sessionStorage', KEEP);
    try { name = window.name || ''; } catch (e) { /* a frame may not own its name */ }
    i = name.indexOf(NAMED);
    if (i >= 0) {
      if (!raw) try { raw = decodeURIComponent(name.slice(i + NAMED.length)); } catch (e) { raw = null; }
      try { window.name = name.slice(0, i); } catch (e) { /* see above */ }
    }
    try { o = JSON.parse(raw || 'null'); } catch (e) { o = null; }
    if (o && typeof o === 'object' && o.lang === lang && Math.abs(Date.now() - o.at) < 30000 && o.keep && typeof o.keep === 'object') kept = o.keep;
  })();
  function stash(l, keep) {
    var s;
    try { s = JSON.stringify({ at: Date.now(), lang: l, keep: keep }); } catch (e) { return; }
    if (!put('sessionStorage', KEEP, s)) try { window.name = (window.name || '') + NAMED + encodeURIComponent(s); } catch (e) { /* nothing carried */ }
  }

  /* ---------- strings ---------- */
  var missing = {};
  function dict() { return lang === BASE ? null : (KT.i18nData || {})[lang] || {}; }
  function fill(s, vars) { return vars ? s.replace(/\{(\w+)\}/g, function (m, k) { return vars[k] != null ? vars[k] : m; }) : s; }
  /* a string that carries its own direction (right-to-left pages only) */
  var LRI = '\u2066', RLI = '\u2067', PDI = '\u2069', HEB = /[\u0590-\u05ff\ufb1d-\ufb4f]/;
  var MARKED = /^[\u2066-\u2068\u200e\u200f\u202a\u202b]/;                                  // it says its direction itself
  var BLOCKS = /<(?:div|p|ul|ol|li|dl|dt|dd|table|thead|tbody|tfoot|tr|td|th|h[1-6]|section|article|header|footer|nav|pre|blockquote|figure|hr|svg|canvas|label|button|input|select|option|textarea)\b/i;
  function isolate(s) {
    if (!KT.rtl || !s || !/[0-9A-Za-z\u0590-\u05ff\ufb1d-\ufb4f]/.test(s) || MARKED.test(s)) return s;
    if (s.indexOf('<') >= 0 && (BLOCKS.test(s) || (s.charAt(0) === '<' && s.charAt(s.length - 1) === '>'))) return s;
    return (HEB.test(s) ? RLI : LRI) + s + PDI;
  }
  KT.t = function (en, vars) {
    var d = dict(), s = en;
    if (typeof en !== 'string') return en;
    if (!d) return fill(en, vars);
    if (own(d, en)) s = d[en]; else if (!own(missing, en)) missing[en] = 't';
    return isolate(fill(s, vars));
  };

  /* ---------- the copy of the page ---------- */
  var EN = new WeakMap(), AT = new WeakMap(), enTitle = null;   // a swapped block -> its English innerHTML; element -> { attr: English }
  function ownText(el) { for (var n = el.firstChild; n; n = n.nextSibling) if (n.nodeType === 3 && /\S/.test(n.nodeValue)) return true; return false; }
  function isOff(el) { return el.getAttribute('data-i18n') === 'off'; }
  /* the blocks under root that are still as written, in document order. A block holds text of its own; what has no
     letter (a number, a sign) is not a block */
  function blocks(root, out) {
    out = out || [];
    for (var el = root.firstElementChild; el; el = el.nextElementSibling) {
      if (SKIP[el.tagName] || isOff(el) || EN.has(el)) continue;
      if (ownText(el)) { if (/[A-Za-z]/.test(el.textContent)) out.push(el); } else blocks(el, out);
    }
    return out;
  }
  function key(el) { return el.innerHTML.replace(/\s+/g, ' ').trim(); }
  /* the translatable attributes under root (root's own included) that are still as written: [{ el, attr, key }] */
  function attrs(root, out) {
    out = out || [];
    ATTRS.forEach(function (a) {
      var list = [].slice.call(root.querySelectorAll('[' + a + ']'));
      if (root.hasAttribute && root.hasAttribute(a)) list.unshift(root);
      list.forEach(function (el) {
        var v = el.getAttribute(a), done = AT.get(el);
        if (!/[A-Za-z]/.test(v) || (done && own(done, a)) || el.closest('[data-i18n="off"]')) return;
        out.push({ el: el, attr: a, key: v });
      });
    });
    return out;
  }
  /* everything the page asks a translation for, as written: [{ key, kind, el }] (tools/i18n/extract.js lists this) */
  function items(root) {
    root = root || document.body;
    var out = blocks(root).map(function (el) {
      return { key: key(el), kind: el.tagName.toLowerCase() + (el.className ? '.' + String(el.className).split(/\s+/)[0] : ''), el: el };
    });
    attrs(root).forEach(function (o) { out.push({ key: o.key, kind: '@' + o.attr, el: o.el }); });
    if (root === document.body && enTitle === null && /[A-Za-z]/.test(document.title)) out.push({ key: document.title, kind: 'title', el: null });
    return out;
  }
  function apply(root) {
    var d = dict(), n = 0;
    root = root || document.body;
    if (!d || !root) return 0;
    blocks(root).forEach(function (el) {
      var k = key(el);
      EN.set(el, el.innerHTML);
      if (own(d, k)) { el.innerHTML = d[k]; n++; }
      else {                                               // stays English, and says so: it reads left to right
        if (!own(missing, k)) missing[k] = 'block';
        if (!el.hasAttribute('dir')) el.setAttribute('dir', LANGS[BASE].dir);
        if (!el.hasAttribute('lang')) el.setAttribute('lang', BASE);
      }
    });
    attrs(root).forEach(function (o) {
      var m = AT.get(o.el);
      if (!m) AT.set(o.el, m = {});
      m[o.attr] = o.key;
      if (own(d, o.key)) { o.el.setAttribute(o.attr, d[o.key]); n++; } else if (!own(missing, o.key)) missing[o.key] = '@' + o.attr;
    });
    if (root === document.body && enTitle === null && /[A-Za-z]/.test(document.title)) {
      enTitle = document.title;
      if (own(d, enTitle)) { document.title = d[enTitle]; n++; } else if (!own(missing, enTitle)) missing[enTitle] = 'title';
    }
    return n;
  }
  /* a detached copy of el as the page wrote it: every swapped block inside it (or el itself) back in English. An
     element INSIDE a swapped block has no English of its own: its copy is what it shows now */
  function source(el) {
    var c = el.cloneNode(true);
    (function walk(a, b) {
      if (EN.has(a)) { b.innerHTML = EN.get(a); return; }
      for (a = a.firstElementChild, b = b.firstElementChild; a && b; a = a.nextElementSibling, b = b.nextElementSibling) walk(a, b);
    })(el, c);
    return c;
  }

  /* ---------- switching ---------- */
  var switching = false;
  function setLang(l) {
    if (!own(LANGS, l) || l === lang || switching) return false;
    switching = true; setTimeout(function () { switching = false; }, 4000);     // should the reload be refused
    var keep = {};
    try { document.dispatchEvent(new CustomEvent('kt-lang', { detail: { lang: l, was: lang, keep: keep } })); } catch (e) { /* nothing carried */ }
    var ok = save(l);
    stash(l, keep);
    var u = url(l, l !== BASE || !ok);                                           // English needs no word in the address, unless nothing can be saved
    if (u === window.location.href) window.location.reload(); else window.location.replace(u);
    return true;
  }
  /* the control: one button per language, the current one marked. cls = its class (the HUD's, the slide master's) */
  function switcher(cls) {
    var box = document.createElement('div');
    box.className = cls || 'kt-langs'; box.setAttribute('role', 'group'); box.setAttribute('aria-label', KT.t('Language'));
    box.setAttribute('data-i18n', 'off');                            // the names of the languages are not copy
    Object.keys(LANGS).forEach(function (l) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = LANGS[l].short; b.title = LANGS[l].name; b.lang = l;
      b.setAttribute('data-lang', l); b.setAttribute('aria-pressed', String(l === lang));
      if (l === lang) b.className = 'on';
      b.addEventListener('click', function () { setLang(l); });
      box.appendChild(b);
    });
    return box;
  }

  KT.i18n = {
    langs: LANGS, base: BASE, attrs: ATTRS, missing: missing, kept: kept,
    blocks: blocks, key: key, attrsOf: attrs, items: items, apply: apply,
    source: source,
    /* an element's English text, whatever language it shows now */
    en: function (el) { return source(el).textContent.replace(/\s+/g, ' ').trim(); },
    /* the English of a translated attribute */
    attrEn: function (el, a) { var m = AT.get(el); return (m && own(m, a) ? m[a] : el.getAttribute(a)) || ''; },
    /* the document's title in English */
    titleEn: function () { return enTitle === null ? document.title : enTitle; },
    setLang: setLang,
    toggle: function () { var ks = Object.keys(LANGS); return setLang(ks[(ks.indexOf(lang) + 1) % ks.length]); },
    switcher: switcher, url: url,
    /* s as one left-to-right piece (a number with its unit, a formula written as text) inside right-to-left text */
    ltr: function (s) { return KT.rtl ? LRI + s + PDI : s; },
    /* s without the direction marks KnotTalk.t and i18n.ltr add */
    strip: function (s) { return String(s).replace(/[\u2066-\u2069]/g, ''); }
  };
})();
