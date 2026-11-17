/* Knot-a-Surface talk — languages (English, Hebrew).
   The talk is written in English: talk.html holds the copy, the modules hold their own strings. A translation is a
   dictionary keyed by the English itself, talk/i18n/<lang>.js -> KnotTalk.i18nData[lang] = { "English": "translation" },
   so a sentence whose English has changed falls back to English instead of showing an outdated translation.
     KnotTalk.lang          'en' | 'he', decided once, before anything is drawn: ?lang= in the address, else the saved
                            choice, else English. Changing it reloads the page (i18n.setLang): every module reads its
                            strings once, when it is created, so nothing on screen can be left in the other language.
     KnotTalk.rtl           true for Hebrew. <html lang> and <html dir> are set here.
     KnotTalk.t(en, vars)   that string in the current language (the English itself when there is no translation),
                            with {name} filled in from vars. Keys are exact strings.
     KnotTalk.i18n.apply()  swaps the copy of the page: every BLOCK (the outermost element that holds text of its
                            own; its key is its innerHTML with whitespace collapsed) and the attributes in ATTRS.
                            deck.js calls it before it reads the page. i18n.en(el) = a block's English text.
     KnotTalk.i18n.missing  what was asked for and not found, for the checks in tools/i18n/.
   Loaded first, in <head>, followed by the dictionaries. A page without this file (the lab pages) gets English from
   the fallback KnotTalk.t in kit.js. */
(function () {
  'use strict';
  var KT = window.KnotTalk = window.KnotTalk || {};
  var LANGS = { en: { dir: 'ltr', name: 'English' }, he: { dir: 'rtl', name: 'עברית' } };
  var STORE = 'kt-lang', ATTRS = ['data-title', 'title', 'aria-label', 'placeholder'];
  var SKIP = { SCRIPT: 1, STYLE: 1, CANVAS: 1, TEMPLATE: 1, NOSCRIPT: 1, svg: 1, SVG: 1 };

  /* ---------- which language ---------- */
  function saved() {
    var v = null;
    try { v = window.localStorage.getItem(STORE); } catch (e) { /* storage can be closed: private windows, file:// */ }
    if (!v) try { v = window.sessionStorage.getItem(STORE); } catch (e) { /* the same */ }
    if (!v) { var m = /(?:^|;)kt-lang=(\w+)/.exec(window.name || ''); v = m && m[1]; }     // survives a reload in any case
    return v;
  }
  function save(l) {
    try { window.localStorage.setItem(STORE, l); } catch (e) { /* see saved() */ }
    try { window.sessionStorage.setItem(STORE, l); } catch (e) { /* see saved() */ }
    try { window.name = (window.name || '').replace(/(?:^|;)kt-lang=\w+/, '') + ';kt-lang=' + l; } catch (e) { /* a frame may not own its name */ }
  }
  var asked = /[?&]lang=(\w+)/.exec(window.location.search || ''), lang = (asked && asked[1]) || saved() || 'en';
  if (!LANGS[lang]) lang = 'en';
  KT.lang = lang; KT.rtl = LANGS[lang].dir === 'rtl';
  document.documentElement.lang = lang; document.documentElement.dir = LANGS[lang].dir;

  /* ---------- strings ---------- */
  var missing = {};
  function dict() { return lang === 'en' ? null : (KT.i18nData || {})[lang] || {}; }
  function fill(s, vars) { return vars ? s.replace(/\{(\w+)\}/g, function (m, k) { return vars[k] != null ? vars[k] : m; }) : s; }
  KT.t = function (en, vars) {
    var d = dict(), s = en;
    if (d) { if (Object.prototype.hasOwnProperty.call(d, en)) s = d[en]; else missing[en] = 1; }
    return fill(s, vars);
  };

  /* ---------- the copy of the page ---------- */
  function ownText(el) { for (var n = el.firstChild; n; n = n.nextSibling) if (n.nodeType === 3 && /\S/.test(n.nodeValue)) return true; return false; }
  /* the blocks under root, in document order. A block holds text of its own; what has no letter (a number, a sign)
     is not a block; data-i18n="off" keeps a subtree out (a name, a formula written as text). */
  function blocks(root, out) {
    out = out || [];
    for (var el = root.firstElementChild; el; el = el.nextElementSibling) {
      if (SKIP[el.tagName] || el.getAttribute('data-i18n') === 'off') continue;
      if (ownText(el)) { if (/[A-Za-z]/.test(el.textContent)) out.push(el); } else blocks(el, out);
    }
    return out;
  }
  function key(el) { return el.innerHTML.replace(/\s+/g, ' ').trim(); }
  var EN = typeof WeakMap === 'function' ? new WeakMap() : null;
  function apply(root) {
    var d = dict(); if (!d) return;
    root = root || document.body;
    blocks(root).forEach(function (el) {
      var k = key(el);
      if (EN) EN.set(el, el.textContent.replace(/\s+/g, ' ').trim());
      if (Object.prototype.hasOwnProperty.call(d, k)) el.innerHTML = d[k]; else missing[k] = 1;
    });
    ATTRS.forEach(function (a) {
      Array.prototype.forEach.call(root.querySelectorAll('[' + a + ']'), function (el) {
        var v = el.getAttribute(a); if (!/[A-Za-z]/.test(v)) return;
        el.setAttribute(a + '-en', v);
        if (Object.prototype.hasOwnProperty.call(d, v)) el.setAttribute(a, d[v]); else missing[v] = 1;
      });
    });
  }

  KT.i18n = {
    langs: LANGS, blocks: blocks, key: key, apply: apply, attrs: ATTRS, missing: missing,
    /* a block's English text, whatever language it shows now (keys kept in English stay stable: the slide master's) */
    en: function (el) { return (EN && EN.get(el)) || el.textContent.replace(/\s+/g, ' ').trim(); },
    /* the English of a translated attribute */
    attrEn: function (el, a) { return el.getAttribute(a + '-en') || el.getAttribute(a) || ''; },
    setLang: function (l) {
      if (!LANGS[l] || l === lang) return;
      save(l);
      var u = window.location.href;
      if (/[?&]lang=\w+/.test(u)) window.location.href = u.replace(/([?&]lang=)\w+/, '$1' + l); else window.location.reload();
    }
  };
})();
