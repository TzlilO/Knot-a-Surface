# The teleprompter's script

`talk/script/<lang>.js` (today `he.js`) is the speaker's script, written by hand, one passage per step:

```js
window.KnotTalk = window.KnotTalk || {};
KnotTalk.script = KnotTalk.script || {};
KnotTalk.script.he = { "<step key>": "paragraph one.\n\nparagraph two.\n\n[stage direction]", … };
```

Plain text, never HTML. Paragraphs are separated by a blank line; a line wholly inside square brackets is a stage direction
(shown muted on its own line, and not counted as words). The key is the slide master's key for the step, in English in every
language: the section's rail label, " · ", the step's English heading (`KnotTalk.deck.key(step)`; the slide master's list shows
the same title; a second step of the same title gets " #2"). Open `talk.html?master` in a browser and run
`KnotTalk.deck.steps.map(KnotTalk.deck.key)` to list them. Renaming a step changes its key, so the passage must follow.
Only presenter windows (`talk.html?master`, `talk.html?prompter`) load the script, and only for the page's own language.

Check: `node tools/script/check.js`. For each `talk/script/<lang>.js` it reports ERROR (exit code 1) for a step without a
passage, a passage whose key no step has, an empty passage and HTML tags, then prints the words per section in the long and the
short cut next to the section's minutes (`data-min40` / `data-min20`) so that a passage that cannot be said in its time shows.
