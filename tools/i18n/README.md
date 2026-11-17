# The talk in two languages

`talk.html` is written in English. Hebrew is a dictionary keyed by the English itself:
`talk/i18n/he.js` sets `KnotTalk.i18nData.he = { "English, exactly as written": "עברית" }`.
A sentence whose English has changed no longer matches its key, so it falls back to English (and is marked
`dir="ltr" lang="en"`, so it still reads) instead of showing an outdated translation. The checks here report it.

## How it works

- `talk/i18n.js` is loaded first, in `<head>`, then the dictionary. It decides the language once per page load:
  `?lang=he|en` in the address, else the saved choice (`localStorage` `kt-lang`), else English. It sets
  `KnotTalk.lang`, `KnotTalk.rtl`, and `<html lang dir data-lang>`.
- The page's copy: `KnotTalk.i18n.apply()` swaps every block (the outermost element that holds text of its own; its
  key is its `innerHTML` with whitespace collapsed), the attributes `data-title`, `title`, `aria-label`,
  `placeholder`, and the document's title. `talk.html` calls it right after its copy, before the heavy scripts
  load; `talk/deck.js` calls it again before it reads the page. `data-i18n="off"` keeps an element out.
- Strings of the scripts: `KnotTalk.t('English', vars)`, with `{name}` filled in from `vars`. Pages without
  `talk/i18n.js` (the lab pages, the landing page, `talk-he.html`) get the English back from a fallback in
  `talk/kit.js`.
- Switching reloads the page, because every module reads its strings once, when it is created. The control is in the
  HUD and in the slide master, the key is L. The reload keeps the step, the cut, the parts that are on, the running
  clock and the mute. Hebrew goes into the address (`talk.html?lang=he`), so the link can be shared.
- The slide master's keys (`rail · step title`, saved in `kt-parts` and exchanged between windows) are the English
  titles in both languages; the titles it shows are the page's.

Right to left: the deck mirrors (`deck.css`, `prelude.css`, `master.css` name sides as inline-start / inline-end,
and each ends with what Hebrew needs besides). A simulation stage does not mirror: inside `.kt-sim` everything
keeps its English direction (module layout, canvas and SVG text, numbers); only the kit's panels, captions and hints
read right to left (`kit.css`). So that a Hebrew string still reads right there, `KnotTalk.t` wraps its result in a
bidi isolate (U+2067 … U+2069 around text with Hebrew letters, U+2066 … U+2069 otherwise) on a right-to-left page.

## Adding or changing a sentence

1. Write the English: in `talk.html` for copy, as `KnotTalk.t('…')` in a script. One sentence is one call, with a
   template for anything that varies: `KnotTalk.t('error {e}%', { e: x })`, never pieces glued around a number.
2. Add the translation to `talk/i18n/he.js` under exactly that English (for copy: run `node tools/i18n/extract.js`
   to see the key the page will look up). Rules for the Hebrew:
   - keep every inline element of the key with the same attributes (`<b>`, `<em class="c-surf">`, `<kbd>`,
     `<span data-tex="…"></span>` verbatim), every number and every `{placeholder}`;
   - anything that must read left to right as one piece gets a wrapper: `<span dir="ltr">12 × 16</span>` in copy
     and in strings set as HTML, U+2066 … U+2069 in plain strings (canvas, attributes; write them `\u2066`,
     `\u2069`). That is a number with its unit, a product, a range, an arrow, a formula written as text;
   - percent attached (`5.7%`), Western digits.
3. `node tools/i18n/check.js`, then look at the step in `talk.html?lang=he`.

A changed English sentence needs its dictionary key changed with it; `check.js` lists the sentence under (a) and
the old key under (e).

## The tools (node; playwright from `/opt/node-tools/node_modules/playwright` or `NODE_PATH`)

| | |
|---|---|
| `node tools/i18n/check.js` | the checks below, on `talk/i18n/he.js`; exit code 1 on an ERROR |
| `node tools/i18n/check.js --parts <dir or files>` | the same with translators' parts merged over the dictionary (`--only-parts`: without it) |
| `node tools/i18n/extract.js [out.json] [--modules]` | what the page asks for, by section (`--modules`: and the scripts' keys) |
| `node tools/i18n/merge.js <parts…> [--out file] [--keep]` | writes the dictionary from parts (`{ "English": "Hebrew" }` JSON), a later part winning, `chrome.json` last in a directory |

`check.js` reports:

- **(a) coverage** (ERROR): a key the page or a `KnotTalk.t('…')` call asks for has no translation. Calls are found by
  scanning `talk/*.js`, `talk/sims/*.js`, `talk/he/*.js` for `KnotTalk.t(`, `KT.t(` and any name a file gave that
  function (`var T = KT.t`); a choice between literals (`a ? 'x' : 'y'`) counts.
- **(b) markup** (ERROR): tags or attributes differ between key and value (added `<span dir="ltr">` / `<bdi>` are
  allowed), tags not closed in order, a `{placeholder}` lost, markup in a value that is shown as plain text.
- **(c) numbers** (ERROR): a number of the key is not in the value.
- **(d) direction** (warning): what will be reordered on screen unless wrapped: `5.7 %`, an arrow, ×, a range or `=`
  between Latin letters or digits, `S(u, v)`, a number and its unit.
- **(e) stale** (warning): a key nothing asks for any more.
- **(f) keys** (ERROR): the slide master's step keys differ between the two languages, or the page throws.
- notes: values equal to their English or without a Hebrew letter; `KnotTalk.t` calls whose argument is not a
  literal (their keys cannot be checked from the source: check them on screen, or in
  `KnotTalk.i18n.missing` in the browser's console after walking the talk in Hebrew).

`KnotTalk.i18n.missing` in the page lists what was asked for and not found, with where: `t`, `block`, `@attr`,
`title`.
