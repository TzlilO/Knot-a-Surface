/* Knot-a-Surface talk — module `ledger`: "The parameter ledger".
   Exact arithmetic, no benchmark: what one free 3DGS Gaussian stores versus what one control
   point stores, and what that does to the size of a scene. The ratio is whatever your sliders
   make it; the paper's 10–50x claim is drawn as a band for reference.
   Tool: plain DOM + CSS. Standalone: talk/sims/ledger.html. Depends on kit.js only. */
(function () {
  'use strict';
  var KT = window.KnotTalk;
  /* T: a string in the page's language (talk/i18n.js; the English itself on a page without languages). A line with a
     number in it is one template, never pieces joined in English order. In Hebrew the stage stays laid out left to
     right (kit.css) and a string from T carries its own direction. Never compare or parse what T returns. */
  var T = KT.t;

  /* per-primitive float counts */
  var GS = [['μ', 3, T('position')], ['s', 3, T('scale')], ['q', 4, T('rotation')], ['α', 1, T('opacity')], ['SH₃', 48, T('colour, 16 × RGB')]];
  var CP = [['P', 3, T('position')], ['α̃', 1, T('opacity')], ['SH₁', 12, T('colour, 4 × RGB')]];
  var PER_G = 59, PER_CP = 16;
  var NMAX = 5e6;

  var SCENES = {
    gaussian: { show: 1, n: 1e6, m: 1e5, cap: T('<b>3DGS</b> · every Gaussian is its own set of free numbers') },
    control:  { show: 2, n: 1e6, m: 1e5, cap: T('<b>Knot-a-Surface</b> · the numbers live on control points; Gaussians carry none') },
    scale:    { show: 3, n: 1e6, m: 1e5, cap: T('<b>a scene</b> · same Gaussians on screen, two very different ledgers') },
    knob:     { show: 3, n: 1e5, m: 1e5, sweep: 1, cap: T('<b>more Gaussians</b> · one ledger grows, the other does not move') }
  };

  function fmt(n) {
    if (n >= 1e6) return parseFloat((n / 1e6).toFixed(n >= 1e7 ? 0 : 2)) + ' M';
    if (n >= 1e3) return parseFloat((n / 1e3).toFixed(n >= 1e5 ? 0 : 1)) + ' k';
    return String(Math.round(n));
  }
  function mb(params) { var b = params * 4 / 1048576; return b >= 100 ? b.toFixed(0) + ' MB' : b >= 1 ? b.toFixed(1) + ' MB' : (b * 1024).toFixed(0) + ' kB'; }

  KT.register('ledger', {
    title: 'The parameter ledger',
    blurb: 'What each representation actually has to store and optimise. Exact arithmetic over your own ' +
           'configuration — Gaussians rendered, control points, float32 — with the paper’s 10–50× band marked for reference.',
    tool: 'DOM + CSS · exact arithmetic',
    scenes: [
      { id: 'gaussian', label: 'One free Gaussian' }, { id: 'control', label: 'One control point' },
      { id: 'scale', label: 'A whole scene' }, { id: 'knob', label: 'Turn the knob' }
    ],
    create: function (env) {
      var ui = env.ui, root = env.root;
      var st = { n: 1e6, m: 1e5, sc: SCENES.gaussian, t: 0, touched: false };
      var wrap = ui.el('div', 'lg-wrap'); root.appendChild(wrap);
      var cap = ui.el('div', 'kt-caption'); root.appendChild(cap);

      function record(cls, title, sub, groups, total, note) {
        var r = ui.el('div', 'lg-rec ' + cls);
        r.appendChild(ui.el('div', 'lg-rec-h', '<span>' + title + '</span><b>' + total + '</b>'));
        r.appendChild(ui.el('div', 'lg-rec-s', sub));
        var row = ui.el('div', 'lg-cells');
        groups.forEach(function (g) {
          var grp = ui.el('div', 'lg-grp'), cells = ui.el('div', 'lg-grp-c');
          for (var i = 0; i < g[1]; i++) cells.appendChild(ui.el('i'));
          grp.appendChild(cells);
          grp.appendChild(ui.el('div', 'lg-grp-l', T('<b>{sym}</b> {n} · {what}', { sym: g[0], n: g[1], what: g[2] })));
          row.appendChild(grp);
        });
        if (note) row.appendChild(ui.el('div', 'lg-none', note));
        r.appendChild(row);
        return r;
      }
      var recs = ui.el('div', 'lg-recs'); wrap.appendChild(recs);
      var rG = record('lg-free', T('one free Gaussian'), T('3DGS · optimised independently'), GS, T('{n} floats', { n: PER_G }));
      var rC = record('lg-ctrl', T('one control point'), T('Knot-a-Surface · the only thing optimised'), CP, T('{n} floats', { n: PER_CP }));
      var rD = record('lg-der', T('one Gaussian'), T('Knot-a-Surface · derived from S(u, v)'), [], T('{n} floats', { n: 0 }),
        'μ = S(u,v) &nbsp;·&nbsp; R = [Ŝ<sub>u</sub> Ŝ<sub>v</sub><sup>⊥</sup> n] &nbsp;·&nbsp; s = (‖S<sub>u</sub>‖Δu, ‖S<sub>v</sub>‖Δv, ε)');
      recs.appendChild(rG); recs.appendChild(rC); recs.appendChild(rD);

      // the scene-scale comparison
      var scale = ui.el('div', 'lg-scale'); wrap.appendChild(scale);
      function bar(cls, label) {
        var b = ui.el('div', 'lg-bar ' + cls);
        b.appendChild(ui.el('div', 'lg-bar-l', label));
        var track = ui.el('div', 'lg-bar-t'), fill = ui.el('div', 'lg-bar-f'); track.appendChild(fill); b.appendChild(track);
        var val = ui.el('div', 'lg-bar-v'); b.appendChild(val); scale.appendChild(b);
        return { fill: fill, val: val };
      }
      var bG = bar('lg-free', '3DGS'), bC = bar('lg-ctrl', 'Knot-a-Surface');
      var big = ui.el('div', 'lg-ratio'); scale.appendChild(big);
      var bigN = ui.el('div', 'lg-ratio-n'), bigL = ui.el('div', 'lg-ratio-l', T('fewer optimised parameters')); big.appendChild(bigN); big.appendChild(bigL);
      var gauge = ui.el('div', 'lg-gauge', '<div class="lg-band"></div><div class="lg-pin"></div>' +
        '<span style="left:0">1×</span><span style="left:33.3%">10×</span><span style="left:56.6%">50×</span><span style="left:100%">1000×</span>');
      big.appendChild(gauge);
      var pin = gauge.querySelector('.lg-pin');

      var ctl = ui.el('div', 'lg-ctl'); scale.appendChild(ctl);
      function logSlider(label, lo, hi, get, set) {
        return ui.slider(ctl, { label: label, min: 0, max: 1000, step: 1, value: Math.round(Math.log(get() / lo) / Math.log(hi / lo) * 1000),
          fmt: function (v) { return fmt(lo * Math.pow(hi / lo, v / 1000)); },
          onInput: function (v) { st.touched = true; set(lo * Math.pow(hi / lo, v / 1000)); paint(); } });
      }
      var slN = logSlider(T('Gaussians rendered'), 1e4, 1e7, function () { return st.n; }, function (v) { st.n = v; });
      var slM = logSlider(T('control points'), 1e3, 1e6, function () { return st.m; }, function (v) { st.m = v; });
      function setLog(sl, v, lo, hi) { sl.set(Math.round(Math.log(v / lo) / Math.log(hi / lo) * 1000)); }
      var fine = ui.el('div', 'lg-fine', T('float32 · 3DGS at SH degree 3 (59 per Gaussian) · ours at SH degree 1 on the control net (16 per control point) · ' +
        'shaded band = the 10–50× the paper projects'));
      if (KT.rtl) fine.dir = 'rtl';                   // several lines of running text: in Hebrew they are set from the right
      scale.appendChild(fine);

      function paint() {
        var pg = st.n * PER_G, pc = st.m * PER_CP, ratio = pg / pc;
        var ref = st.sc.sweep ? NMAX * PER_G : Math.max(pg, pc);
        bG.fill.style.width = Math.max(0.4, pg / ref * 100) + '%';
        bC.fill.style.width = Math.max(0.4, pc / ref * 100) + '%';
        // two lines under each bar, each its own template (<small> is a line of its own)
        bG.val.innerHTML = T('<b>{p}</b> params · {mb}', { p: fmt(pg), mb: mb(pg) }) +
          '<small>' + T('{n} Gaussians × {k}', { n: fmt(st.n), k: PER_G }) + '</small>';
        bC.val.innerHTML = T('<b>{p}</b> params · {mb}', { p: fmt(pc), mb: mb(pc) }) +
          '<small>' + T('{m} control points × {k} &nbsp;+&nbsp; {n} Gaussians × 0', { m: fmt(st.m), k: PER_CP, n: fmt(st.n) }) + '</small>';
        bigN.textContent = (ratio >= 100 ? Math.round(ratio) : ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1)) + '×';
        bigL.textContent = ratio >= 1 ? T('fewer optimised parameters') : T('of the 3DGS parameter count');
        pin.style.left = KT.clamp(Math.log10(Math.max(1, ratio)) / 3, 0, 1) * 100 + '%';
      }

      function setScene(id) {
        var sc = SCENES[id]; if (!sc) return;
        st.sc = sc; st.t = 0; st.touched = false; st.n = sc.n; st.m = sc.m;
        setLog(slN, st.n, 1e4, 1e7); setLog(slM, st.m, 1e3, 1e6);
        wrap.className = 'lg-wrap lg-show-' + sc.show;
        cap.innerHTML = sc.cap; paint();
      }
      env.loop(function (t, dt) {
        if (!st.sc.sweep || st.touched) return;
        st.t += dt;                                   // sweep the Gaussian count; the control net stays put
        st.n = Math.exp(KT.lerp(Math.log(1e5), Math.log(NMAX), 0.5 - 0.5 * Math.cos(st.t * 0.7)));
        setLog(slN, st.n, 1e4, 1e7); paint();
      });

      if (!document.getElementById('kt-ledger-css')) {
        var css = document.createElement('style'); css.id = 'kt-ledger-css';
        css.textContent = [
          '.kt-sim-ledger{container-type:size}',
          '.lg-wrap{position:absolute;inset:0;display:flex;flex-direction:column;justify-content:center;gap:clamp(14px,3.2cqh,34px);padding:54px clamp(16px,5cqw,56px) 22px;overflow:auto}',
          '.lg-recs{display:grid;gap:clamp(8px,1.6cqh,16px)}',
          '.lg-rec{--c:var(--err);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:10px;background:var(--bg2);padding:12px 14px;transition:opacity .5s,transform .5s}',
          '.lg-ctrl{--c:var(--ctrl)}.lg-der{--c:var(--splat)}',
          '.lg-rec-h{display:flex;align-items:baseline;justify-content:space-between;gap:12px;font:600 clamp(15px,2cqw,21px)/1.1 var(--f-display)}',
          '.lg-rec-h b{font:600 clamp(15px,2cqw,21px)/1 var(--f-mono);color:var(--c);white-space:nowrap}',
          '.lg-rec-s{font:500 11px/1.3 var(--f-mono);letter-spacing:.06em;color:var(--mute);margin:4px 0 10px;text-transform:uppercase}',
          '.lg-cells{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:flex-start}',
          '.lg-grp-c{display:flex;flex-wrap:wrap;gap:2px;max-width:calc(16 * 13px)}',
          '.lg-grp-c i{width:11px;height:11px;border-radius:2px;background:var(--c)}',
          '.lg-grp-l{font:500 11px/1.2 var(--f-mono);color:var(--dim);margin-top:5px;white-space:nowrap}',
          '.lg-grp-l b{color:var(--text);font-weight:600}',
          '.lg-none{font:500 clamp(12px,1.5cqw,15px)/1.5 var(--f-mono);color:var(--dim);border:1px dashed rgba(var(--splat-rgb),.5);border-radius:8px;padding:9px 12px}',
          '.lg-none sub,.lg-none sup{font-size:.7em}',
          '.lg-scale{display:grid;gap:10px;transition:opacity .5s}',
          '.lg-bar{--c:var(--err);display:grid;grid-template-columns:118px minmax(0,1fr);gap:4px 12px;align-items:center}',
          '.lg-bar.lg-ctrl{--c:var(--ctrl)}',
          '.lg-bar-l{font:600 12px/1.2 var(--f-mono);color:var(--text)}',
          '.lg-bar-t{height:16px;border-radius:0 4px 4px 0;background:var(--bg2);border-left:1px solid var(--dim)}',
          '.lg-bar-f{height:100%;min-width:3px;border-radius:0 4px 4px 0;background:var(--c)}',
          '.lg-bar-v{grid-column:2;font:500 12px/1.3 var(--f-mono);color:var(--dim)}',
          '.lg-bar-v b{color:var(--text);font-size:14px}.lg-bar-v small{display:block;font-size:11px;color:var(--mute)}',
          '.lg-ratio{display:grid;grid-template-columns:auto minmax(0,1fr);align-items:center;gap:4px 18px;margin-top:4px}',
          '.lg-ratio-n{grid-row:span 2;font:700 clamp(44px,9cqw,104px)/.9 var(--f-display);letter-spacing:-.04em;color:var(--text);font-variant-numeric:tabular-nums;min-width:3.2ch}',
          '.lg-ratio-l{font:500 clamp(13px,1.7cqw,18px)/1.2 var(--f-ui);color:var(--dim);align-self:end}',
          '.lg-gauge{position:relative;height:30px;align-self:start;margin-right:18px}',
          '.lg-gauge:before{content:"";position:absolute;left:0;right:0;top:5px;height:4px;border-radius:2px;background:var(--line2)}',
          '.lg-band{position:absolute;left:33.3%;width:23.3%;top:5px;height:4px;background:var(--ok);opacity:.75}',
          '.lg-pin{position:absolute;top:0;width:3px;height:14px;margin-left:-1.5px;border-radius:2px;background:var(--text);box-shadow:0 0 0 2px var(--bg)}',
          '.lg-gauge span{position:absolute;top:16px;transform:translateX(-50%);font:500 10px/1 var(--f-mono);color:var(--mute)}',
          '.lg-gauge span:first-of-type{transform:none}.lg-gauge span:last-of-type{transform:translateX(-100%)}',
          '.lg-ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:4px 26px;margin-top:6px}',
          '.lg-fine{font:500 10.5px/1.5 var(--f-mono);color:var(--mute);letter-spacing:.02em}',
          /* staged reveal */
          '.lg-show-1 .lg-ctrl,.lg-show-1 .lg-der,.lg-show-1 .lg-scale,.lg-show-2 .lg-scale{opacity:0;pointer-events:none}',
          '.lg-show-1 .lg-rec.lg-ctrl,.lg-show-1 .lg-rec.lg-der{transform:translateY(10px)}',
          '.lg-show-2 .lg-free,.lg-show-3 .lg-rec-s{opacity:.55}',
          '.lg-show-3 .lg-grp-l{display:none}.lg-show-3 .lg-rec{padding:9px 14px}.lg-show-3 .lg-rec-s{margin-bottom:7px}',
          '.lg-show-3 .lg-grp-c i{width:7px;height:7px}.lg-show-3 .lg-grp-c{max-width:none}.lg-show-3 .lg-none{padding:5px 10px;font-size:11px}',
          '@container (max-width:560px){.lg-bar{grid-template-columns:1fr}.lg-bar-v{grid-column:1}.lg-grp-c i{width:8px;height:8px}.lg-wrap{justify-content:flex-start}}',
          /* Hebrew (talk/i18n.js sets <html lang>) has no capitals and is not letter-spaced: a name or a formula inside
             a label keeps its case */
          ':root[lang="he"] .lg-rec-s{letter-spacing:0;text-transform:none}:root[lang="he"] .lg-fine{letter-spacing:0}'
        ].join('\n');
        document.head.appendChild(css);
      }
      setScene(env.opts.scene || 'gaussian');
      return { setScene: setScene };
    }
  });
})();
