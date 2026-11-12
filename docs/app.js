/* TG-LiteNet project site, edition 2. Vanilla JS + GSAP/ScrollTrigger. All numbers come from data/results.js (scripts/make_website_data.py). */
(function () {
  "use strict";
  const R = window.RESULTS;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const DS = ["drive", "stare", "chase_db1", "hrf"];
  const DS_NAME = { drive: "DRIVE", stare: "STARE", chase_db1: "CHASE_DB1", hrf: "HRF", unseen: "Unseen average" };
  const COL = { A: "#6b7280", B: "#8b93a1", C: "#aab2bf", D: "#cdd3dc", E: "#ff6a4d", "E-MC": "#ff6a4d" };
  const CORAL = "#ff6a4d", MUTE = "#6b7280";
  const ASSET = "assets/viewer/";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = typeof window.gsap !== "undefined" && typeof window.ScrollTrigger !== "undefined";
  if (hasGsap) gsap.registerPlugin(ScrollTrigger);

  const h = (tag, attrs = {}, text) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; return e; };
  const el = (tag, attrs = {}, html) => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (html !== undefined) e.innerHTML = html; return e; };
  const f3 = (v) => v.toFixed(3);
  const stageBy = (t) => R.stages.find((s) => s.tag === t);

  /* ---------------------------------------------------------------- shared helpers */
  function group(container, items, initial, onPick) {
    const btns = items.map(([key, label]) => {
      const b = el("button", { class: "tab", type: "button", "aria-pressed": "false" }, label);
      b.addEventListener("click", () => pick(key));
      container.appendChild(b);
      return [key, b];
    });
    function pick(key) { btns.forEach(([k, b]) => b.setAttribute("aria-pressed", String(k === key))); onPick(key); }
    pick(initial);
    return pick;
  }
  const tip = $("#tip");
  function showTip(html, x, y) {
    tip.innerHTML = html; tip.style.display = "block";
    const w = tip.offsetWidth, hh = tip.offsetHeight;
    tip.style.left = Math.min(innerWidth - w - 8, Math.max(8, x + 14)) + "px";
    tip.style.top = Math.min(innerHeight - hh - 8, Math.max(8, y + 14)) + "px";
  }
  const hideTip = () => (tip.style.display = "none");
  $$(".term").forEach((t) => {
    t.tabIndex = 0;
    t.addEventListener("mousemove", (e) => showTip(t.dataset.def, e.clientX, e.clientY));
    t.addEventListener("mouseleave", hideTip);
    t.addEventListener("focus", () => { const r = t.getBoundingClientRect(); showTip(t.dataset.def, r.left, r.bottom); });
    t.addEventListener("blur", hideTip);
  });
  function key(container, items) {
    container.innerHTML = "";
    items.forEach(([c, t]) => container.appendChild(el("span", {}, `<i style="background:${c}"></i>${t}`)));
  }
  function niceTicks(lo, hi, n = 5) {
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function barChart(svg, o) {
    svg.innerHTML = "";
    const W = 900, H = o.H || 360, m = { l: 64, r: 16, t: 16, b: o.bottom || 56 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    const tf = (v) => (o.log ? Math.log10(v) : v);
    const y = (v) => m.t + ih - ((tf(Math.min(Math.max(v, o.yMin), o.yMax)) - tf(o.yMin)) / (tf(o.yMax) - tf(o.yMin))) * ih;
    (o.ticks || niceTicks(o.yMin, o.yMax)).forEach((t) => {
      svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "gridline" }));
      svg.appendChild(h("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, o.fmtTick ? o.fmtTick(t) : t));
    });
    svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }));
    const gw = iw / o.cats.length, ns = o.series.length, bw = Math.min(46, (gw * 0.78) / ns);
    o.cats.forEach((cat, i) => {
      const cx = m.l + gw * i + gw / 2;
      (Array.isArray(cat) ? cat : [cat]).forEach((line, j) => svg.appendChild(h("text", { x: cx, y: m.t + ih + 20 + j * 14, "text-anchor": "middle" }, line)));
      o.series.forEach((s, k) => {
        const v = s.vals[i]; if (v == null) return;
        const x0 = cx - (ns * bw) / 2 + k * bw, top = y(v), base = m.t + ih;
        const r = h("rect", { x: x0 + 1.5, y: top, width: bw - 3, height: Math.max(1, base - top), rx: 5, fill: s.color(i), class: "bar" });
        const info = `<b>${Array.isArray(cat) ? cat.join(" ") : cat}</b><br>${s.name}: ${o.fmtVal(v)}`;
        r.addEventListener("mousemove", (e) => showTip(info, e.clientX, e.clientY)); r.addEventListener("mouseleave", hideTip);
        svg.appendChild(r);
        if (s.err && s.err[i]) {
          const lo = y(Math.max(o.yMin, v - s.err[i])), hi = y(Math.min(o.yMax, v + s.err[i])), mx = x0 + bw / 2;
          [["line", { x1: mx, x2: mx, y1: lo, y2: hi }], ["line", { x1: mx - 4, x2: mx + 4, y1: hi, y2: hi }], ["line", { x1: mx - 4, x2: mx + 4, y1: lo, y2: lo }]]
            .forEach(([t, a]) => svg.appendChild(h(t, { ...a, stroke: "rgba(255,255,255,.75)", "stroke-width": 1.5 })));
        }
        if (o.valLabels !== false) svg.appendChild(h("text", { x: x0 + bw / 2, y: top - 7 - (s.err && s.err[i] ? 8 : 0), "text-anchor": "middle", style: "fill:#e6e3dd;font-weight:700;" + (ns > 2 ? "font-size:10px" : "") }, o.fmtVal(v)));
      });
    });
  }

  /* ---------------------------------------------------------------- motion layer */
  (function motion() {
    const bar = $("#progress");
    const onScroll = () => { const d = document.documentElement; bar.style.transform = `scaleX(${d.scrollTop / Math.max(1, d.scrollHeight - innerHeight)})`; };
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
    $$(".spot").forEach((n) => n.addEventListener("mousemove", (e) => { const r = n.getBoundingClientRect(); n.style.setProperty("--mx", e.clientX - r.left + "px"); n.style.setProperty("--my", e.clientY - r.top + "px"); }));
    // active nav link
    const links = $$(".nav-link"), secs = links.map((a) => $(a.getAttribute("href")));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a, i) => { const on = secs[i] === e.target; a.classList.toggle("bg-white/10", on); a.classList.toggle("text-white", on); }); }), { rootMargin: "-45% 0px -50% 0px" });
    secs.forEach((s) => s && io.observe(s));
    if (!hasGsap || reduce) { $$(".word").forEach((w) => (w.style.opacity = 1)); return; }

    // hero intro and parallax
    gsap.from("#heroTitle", { y: 46, opacity: 0, duration: 1.2, ease: "power3.out", delay: 0.15 });
    gsap.from(".hero-sub", { y: 26, opacity: 0, duration: 1, ease: "power3.out", delay: 0.55 });
    gsap.from(".hero-cta > *", { y: 22, opacity: 0, duration: 0.9, ease: "power3.out", delay: 0.8, stagger: 0.12 });
    gsap.fromTo("#heroBg", { scale: 1.18 }, { scale: 1, ease: "none", scrollTrigger: { trigger: "#top", start: "top top", end: "bottom top", scrub: true } });
    gsap.to("#heroBg", { yPercent: 14, ease: "none", scrollTrigger: { trigger: "#top", start: "top top", end: "bottom top", scrub: true } });

    // generic reveal
    $$(".reveal").forEach((n, i) => gsap.from(n, { y: 44, opacity: 0, duration: 1, ease: "power3.out", delay: (i % 4) * 0.06, scrollTrigger: { trigger: n, start: "top 88%", once: true } }));

    // scrubbing text reveal: words go from 12% to full opacity as the paragraph scrolls through
    const p = $("#scrubText");
    if (p) {
      p.innerHTML = p.textContent.trim().split(/\s+/).map((w) => `<span class="word">${w}</span>`).join(" ");
      gsap.fromTo($$(".word", p), { opacity: 0.12 }, { opacity: 1, ease: "none", stagger: 0.12, scrollTrigger: { trigger: p, start: "top 78%", end: "bottom 46%", scrub: true } });
    }

    // image scale and fade: grow from 0.8 to 1 on the way in, darken and fade on the way out
    $$(".sf-img").forEach((n) => {
      gsap.fromTo(n, { scale: 0.8, opacity: 0.55 }, { scale: 1, opacity: 1, ease: "none", scrollTrigger: { trigger: n, start: "top 92%", end: "top 38%", scrub: true } });
      gsap.to(n, { opacity: 0.2, filter: "brightness(0.45)", ease: "none", scrollTrigger: { trigger: n, start: "bottom 42%", end: "bottom 4%", scrub: true } });
    });
  })();

  /* ---------------------------------------------------------------- bento numbers */
  (function bento() {
    const A = stageBy("A"), E = stageBy("E-MC"), ratio = Math.round(A.params_m / E.params_m);
    $("#bentoE").textContent = f3(E.unseen_dice); $("#bentoA").textContent = f3(A.unseen_dice);
    $("#bentoLite").textContent = E.params_m.toFixed(2) + " M"; $("#bentoUnet").textContent = A.params_m.toFixed(0) + " M";
    const num = $("#bentoRatio"), bars = $$(".bento-bar");
    bars[1].dataset.w = String((E.params_m / A.params_m) * 100);
    const set = () => bars.forEach((b) => (b.style.width = Math.max(1.5, +b.dataset.w) + "%"));
    bars.forEach((b) => { b.style.width = "0%"; b.style.transition = "width 1.4s cubic-bezier(.2,.8,.2,1)"; });
    if (!hasGsap || reduce) { num.textContent = ratio; set(); return; }
    const obj = { v: 0 };
    ScrollTrigger.create({ trigger: "#bento", start: "top 75%", once: true, onEnter: () => { gsap.to(obj, { v: ratio, duration: 1.8, ease: "power2.out", onUpdate: () => (num.textContent = Math.round(obj.v)) }); set(); } });
  })();

  /* ---------------------------------------------------------------- dataset explorer */
  (function datasets() {
    const insight = {
      drive: "The <b>training</b> dataset. The model learns from 16 of these 40 photos (4 more pick the best checkpoint).",
      stare: "<b>Brighter</b> photos: the green channel averages 109 against 97 for DRIVE. Never seen during training.",
      chase_db1: "<b>Much darker</b> photos (green average 62) and bigger images. Vessels are 6.1 px wide before resizing and 3.6 px after.",
      hrf: "<b>Huge</b> images (3504x2336, about 6 times wider than DRIVE). After resizing the vessels are only 1.4 px wide because the labels are thin, so HRF also tests a different labelling style.",
    };
    const cv = $("#dsCanvas"), ctx = cv.getContext("2d"), imgs = {};
    let cur = "drive", view = "img";
    const load = (src) => new Promise((res) => { const i = new Image(); i.onload = () => res(i); i.src = src; });
    async function ensure(ds) { if (!imgs[ds]) imgs[ds] = { img: await load(`${ASSET}${ds}_img.jpg`), gt: await load(`${ASSET}${ds}_gt.png`) }; return imgs[ds]; }
    async function draw() {
      const { img, gt } = await ensure(cur);
      cv.width = img.width; cv.height = img.height;
      if (view === "gt") { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height); ctx.drawImage(gt, 0, 0); }
      else {
        ctx.drawImage(img, 0, 0);
        if (view === "both") {
          const t = document.createElement("canvas"); t.width = img.width; t.height = img.height;
          const tc = t.getContext("2d"); tc.drawImage(gt, 0, 0);
          try {
            const d = tc.getImageData(0, 0, t.width, t.height);
            for (let i = 0; i < d.data.length; i += 4) { const on = d.data[i] > 127; d.data[i] = 255; d.data[i + 1] = 106; d.data[i + 2] = 77; d.data[i + 3] = on ? 255 : 0; }
            tc.putImageData(d, 0, 0); ctx.drawImage(t, 0, 0);
          } catch (e) { /* opened from file:// */ }
        }
      }
      const s = R.dataset_info[cur];
      $("#dsStats").innerHTML = [["Images", s.n], ["Original size", `${s.w}x${s.h}`], ["Vessel pixels", s.vessel_pct + "%"], ["Vessel width, before and after resizing", `${s.calibre_native} to ${s.calibre_norm} px`], ["Brightness (green)", s.green], ["Role", cur === "drive" ? "training" : "unseen test"]]
        .map(([k, v]) => `<div><dt>${k}</dt><dd style="font-size:1.25rem">${v}</dd></div>`).join("");
      $("#dsInsight").innerHTML = insight[cur];
    }
    group($("#dsTabs"), DS.map((d) => [d, DS_NAME[d]]), "drive", (d) => { cur = d; draw(); });
    group($("#dsView"), [["img", "Photo"], ["gt", "Traced vessels"], ["both", "Overlay"]], "img", (v) => { view = v; draw(); });
  })();

  /* ---------------------------------------------------------------- horizontal accordion */
  (function accordion() {
    const panels = $$(".acc-panel");
    const open = (p) => panels.forEach((q) => { const on = q === p; q.classList.toggle("active", on); q.setAttribute("aria-expanded", String(on)); });
    panels.forEach((p) => {
      p.addEventListener("click", () => open(p));
      p.addEventListener("focus", () => open(p));
      p.addEventListener("mouseenter", () => { if (matchMedia("(hover: hover) and (min-width: 1024px)").matches) open(p); });
    });
  })();

  /* ---------------------------------------------------------------- stage stepper */
  (function stages() {
    const text = {
      A: ["Plain U-Net", "The standard baseline: a 31-million-parameter network trained with an overlap loss. Everything else is compared against this."],
      B: ["Add the topology loss", "Adds the centreline loss and a second output that predicts vessel centrelines. Goal: fewer gaps and fewer broken pieces."],
      C: ["Add DG augmentation", "Adds messy-photo augmentation and a consistency loss. Goal: stay accurate on other cameras."],
      D: ["Swap in the compact backbone", "Replaces the 31 M network with a tiny depthwise-separable one (0.21 M parameters). Goal: run on cheap hardware."],
      E: ["Add dropout: TG-LiteNet", "Adds dropout so the model can also say how unsure it is. This is the full TG-LiteNet."],
    };
    const order = ["A", "B", "C", "D", "E"];
    group($("#stageTabs"), order.map((t) => [t, `${t}  ${stageBy(t).label}`]), "A", render);
    function bar(name, val, frac, hot, note) { return `<div class="bar-row"><div class="top"><span>${name}</span><b>${val}</b></div><div class="track"><div class="fill ${hot ? "hot" : ""}" style="width:${Math.max(2, Math.min(100, frac * 100))}%"></div></div><div class="note">${note}</div></div>`; }
    function render(tag) {
      const s = stageBy(tag), i = order.indexOf(tag), p = i > 0 ? stageBy(order[i - 1]) : null;
      const d = (a, b, dd = 3) => (p ? ` (${a - b >= 0 ? "+" : "-"}${Math.abs(a - b).toFixed(dd)} vs ${order[i - 1]})` : "");
      $("#stageText").innerHTML = `<div class="text-coral font-bold text-sm">Stage ${tag}</div><h4 class="text-3xl font-black tracking-tight mt-2">${text[tag][0]}</h4><p class="mt-4 text-white/70 leading-relaxed">${text[tag][1]}</p>
        <p class="mt-6 text-sm text-white/50">${s.seeds} seeds, ${s.params_m.toFixed(2)} M parameters, ${s.gflops} GFLOPs, ${s.cpu_ms.toFixed(0)} ms on a laptop CPU</p>`;
      $("#stageBars").innerHTML =
        bar("Dice (overlap), higher is better", f3(s.unseen_dice) + d(s.unseen_dice, p ? p.unseen_dice : 0), (s.unseen_dice - 0.7) / 0.1, tag === "E", "bar scale starts at 0.70") +
        bar("clDice (connectivity), higher is better", f3(s.unseen_cldice) + d(s.unseen_cldice, p ? p.unseen_cldice : 0), (s.unseen_cldice - 0.8) / 0.08, tag === "E", "bar scale starts at 0.80") +
        bar("Broken pieces (Betti-0), lower is better", s.unseen_betti0_err.toFixed(1) + d(s.unseen_betti0_err, p ? p.unseen_betti0_err : 0, 1), s.unseen_betti0_err / 90, tag === "E", "full bar is 90 pieces") +
        bar("Gap between DRIVE and unseen, lower is better", f3(s.gap) + d(s.gap, p ? p.gap : 0), s.gap / 0.1, tag === "E", "full bar is 0.10");
    }
  })();

  /* ---------------------------------------------------------------- live viewer */
  (function viewer() {
    const cv = $("#vCanvas"), ctx = cv.getContext("2d", { willReadFrequently: true });
    const MODELS = [["A", "A  U-Net"], ["C", "C  Topology + DG"], ["E", "E  TG-LiteNet"]];
    const MODES = [["err", "Errors"], ["pred", "Prediction"], ["gt", "Ground truth"], ["photo", "Photo"], ["unc", "Uncertainty"]];
    const cache = {};
    let ds = "drive", model = "A", mode = "err", pending = false;
    const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
    const pixels = (img) => { const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(img, 0, 0); return x.getImageData(0, 0, c.width, c.height).data; };
    async function ensure(d) {
      if (cache[d]) return cache[d];
      const b = `${ASSET}${d}_`;
      const [img, gt, fov, pA, pC, pE, unc] = await Promise.all([load(b + "img.jpg"), load(b + "gt.png"), load(b + "fov.png"), load(b + "prob_A.png"), load(b + "prob_C.png"), load(b + "prob_E.png"), load(b + "unc_E.png")]);
      const w = img.width, hh = img.height, ch = (im) => { const p = pixels(im), o = new Uint8Array(w * hh); for (let i = 0; i < o.length; i++) o[i] = p[i * 4]; return o; };
      const c = { w, h: hh, photo: pixels(img), gt: ch(gt), fov: ch(fov), prob: { A: ch(pA), C: ch(pC), E: ch(pE) }, unc: ch(unc) };
      c.gtb = new Uint8Array(w * hh); for (let i = 0; i < c.gtb.length; i++) c.gtb[i] = c.gt[i] > 127 && c.fov[i] > 127 ? 1 : 0;
      c.gtPieces = label(c.gtb, w, hh).n;
      return (cache[d] = c);
    }
    function label(mask, w, hh) {   // 8-connected components
      const lab = new Int32Array(w * hh), stack = new Int32Array(w * hh), sizes = [0]; let n = 0;
      for (let s = 0; s < mask.length; s++) {
        if (!mask[s] || lab[s]) continue;
        n++; let top = 0, size = 0; stack[top++] = s; lab[s] = n;
        while (top) {
          const p = stack[--top]; size++; const x = p % w, y = (p - x) / w;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx, ny = y + dy; if (nx < 0 || ny < 0 || nx >= w || ny >= hh) continue;
            const q = ny * w + nx; if (mask[q] && !lab[q]) { lab[q] = n; stack[top++] = q; }
          }
        }
        sizes.push(size);
      }
      return { lab, n, sizes };
    }
    function update() {
      pending = false;
      const c = cache[ds]; if (!c) return;
      const thr = +$("#thr").value, minsz = +$("#minsz").value;
      $("#thrOut").textContent = thr.toFixed(2); $("#minszOut").textContent = minsz + " px";
      const t = thr * 255, w = c.w, hh = c.h, p = c.prob[model], pred = new Uint8Array(w * hh);
      for (let i = 0; i < pred.length; i++) pred[i] = p[i] >= t && c.fov[i] > 127 ? 1 : 0;
      const L = label(pred, w, hh); let pieces = L.n;
      if (minsz > 0) { for (let i = 0; i < pred.length; i++) if (pred[i] && L.sizes[L.lab[i]] < minsz) pred[i] = 0; pieces = L.sizes.reduce((a, s, k) => a + (k > 0 && s >= minsz ? 1 : 0), 0); }
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < pred.length; i++) { const g = c.gtb[i]; if (pred[i] && g) tp++; else if (pred[i]) fp++; else if (g) fn++; }
      const dice = (2 * tp) / Math.max(1, 2 * tp + fp + fn), rec = tp / Math.max(1, tp + fn), pre = tp / Math.max(1, tp + fp), thin = c.gtPieces > 100;
      $("#rDice").textContent = dice.toFixed(3); $("#rPieces").textContent = pieces; $("#rGt").textContent = c.gtPieces;
      $("#rExtra").textContent = thin ? "n/a" : Math.abs(pieces - c.gtPieces); $("#rRec").textContent = rec.toFixed(2); $("#rPre").textContent = pre.toFixed(2);
      const name = MODELS.find((m) => m[0] === model)[1].replace(/^\w\s+/, "");
      $("#vInsight").innerHTML = thin
        ? `<b>${name}</b> on ${DS_NAME[ds]}: <b>${pieces}</b> predicted vessel pieces. The ground-truth count (${c.gtPieces}) is not meaningful here: the very thin HRF labels shatter when resized to the working scale, which is why the paper counts pieces at full resolution.`
        : `<b>${name}</b> on ${DS_NAME[ds]}: <b>${pieces}</b> vessel pieces against <b>${c.gtPieces}</b> in the ground truth` + (minsz === 0 && pieces > 3 * c.gtPieces ? ". Most of the extra pieces are tiny specks: drag the fragment slider up." : minsz > 0 ? ` after deleting fragments under ${minsz} px.` : ".");
      cv.width = w; cv.height = hh;
      const out = ctx.createImageData(w, hh), o = out.data;
      for (let i = 0; i < pred.length; i++) {
        const k = i * 4; let r = 0, g = 0, b = 0;
        if (mode === "photo") { r = c.photo[k]; g = c.photo[k + 1]; b = c.photo[k + 2]; }
        else if (mode === "gt") r = g = b = c.gtb[i] ? 255 : 0;
        else if (mode === "pred") r = g = b = pred[i] ? 255 : 0;
        else if (mode === "unc") { const u = Math.min(1, c.unc[i] / 150); r = 10 + u * 245; g = 10 + u * (u > 0.65 ? 190 : 40); b = 14 + u * 70; }
        else { const base = 0.4 * (c.photo[k] + c.photo[k + 1] + c.photo[k + 2]) / 3; r = g = b = base; if (c.fov[i] > 127) { if (pred[i] && c.gtb[i]) r = g = b = 255; else if (pred[i]) { r = 255; g = 77; b = 58; } else if (c.gtb[i]) { r = 59; g = 130; b = 246; } } }
        o[k] = r; o[k + 1] = g; o[k + 2] = b; o[k + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }
    const sched = () => { if (!pending) { pending = true; requestAnimationFrame(update); } };
    async function setDs(d) {
      ds = d;
      try { await ensure(d); update(); } catch (e) {
        cv.width = 640; cv.height = 200; ctx.fillStyle = "#05070a"; ctx.fillRect(0, 0, 640, 200); ctx.fillStyle = "#fff"; ctx.font = "16px sans-serif";
        ctx.fillText("Open this page through a web server", 24, 90); ctx.fillText("(python -m http.server in docs/) or GitHub Pages.", 24, 118);
      }
    }
    group($("#vDs"), DS.map((d) => [d, DS_NAME[d]]), "drive", setDs);
    group($("#vModel"), MODELS, "A", (m) => { model = m; sched(); });
    group($("#vMode"), MODES, "err", (m) => { mode = m; sched(); });
    $("#thr").addEventListener("input", sched); $("#minsz").addEventListener("input", sched);
  })();

  /* ---------------------------------------------------------------- results explorer */
  (function results() {
    const stages = ["A", "B", "C", "D", "E"].map(stageBy);
    const METRICS = { dice: ["Dice", "higher is better"], cldice: ["clDice", "higher is better"], betti0_err: ["Betti-0 error", "lower is better"] };
    let metric = "dice", dsel = "unseen";
    key($("#rKey"), stages.map((s) => [COL[s.tag], `${s.tag}  ${s.label}`]));
    group($("#rMetric"), Object.entries(METRICS).map(([k, v]) => [k, v[0]]), "dice", (m) => { metric = m; draw(); });
    group($("#rDs"), [["unseen", "Unseen average"], ...DS.map((d) => [d, DS_NAME[d]])], "unseen", (d) => { dsel = d; draw(); });
    function draw() {
      const vals = stages.map((s) => s[`${dsel}_${metric}`]), errs = stages.map((s) => s[`${dsel}_${metric}_std`]);
      const lo = Math.min(...vals.map((v, i) => v - errs[i])), hi = Math.max(...vals.map((v, i) => v + errs[i]));
      let yMin, yMax;
      if (metric === "betti0_err") { yMin = 0; yMax = Math.ceil((hi * 1.12) / 10) * 10; }
      else { const span = Math.max(0.02, hi - lo); yMin = Math.floor((lo - span * 0.6) * 100) / 100; yMax = Math.ceil((hi + span * 0.35) * 100) / 100; }
      barChart($("#stageChart"), { cats: stages.map((s) => [s.tag, s.label]), series: [{ name: METRICS[metric][0], vals, err: errs, color: (i) => COL[stages[i].tag] }], yMin, yMax,
        fmtVal: (v) => (metric === "betti0_err" ? v.toFixed(1) : v.toFixed(3)), fmtTick: (t) => (metric === "betti0_err" ? t : t.toFixed(2)) });
      $("#rCap").textContent = `${METRICS[metric][0]}, ${DS_NAME[dsel]}, ${METRICS[metric][1]}` + (metric !== "betti0_err" ? `. Axis starts at ${yMin.toFixed(2)}` : "");
      const best = vals.indexOf(metric === "betti0_err" ? Math.min(...vals) : Math.max(...vals));
      $("#rInsight").innerHTML = `<b>Best here: stage ${stages[best].tag} (${stages[best].label})</b>, ${metric === "betti0_err" ? vals[best].toFixed(1) : vals[best].toFixed(3)}. ` +
        (dsel === "unseen" && metric === "dice" ? "At the standard 0.5 threshold the plain U-Net has the best Dice, so the topology and compact stages cost a little overlap. The checks below explain why."
          : dsel === "unseen" && metric === "betti0_err" ? "Topology-trained stages cut the broken pieces by more than half, but see the clean-up check below."
          : dsel === "hrf" ? "HRF is the hardest: huge images with thin labels. The compact stages (D, E) lose the most here." : "Hover a bar for exact numbers.");
    }
  })();

  (function checks() {
    const C = R.controls;
    const names = { A_unet: ["A", "U-Net"], B_unet_topo: ["B", "+ topology"], C_unet_topo_dg: ["C", "+ DG aug."], D_lite_topo_dg: ["D", "compact"], E_tglitenet: ["E", "TG-LiteNet"], X_unet_dg: ["U-Net", "DG only"], S_unet_b8: ["U-Net", "base 8"], S_unet_b16: ["U-Net", "base 16"], X_lite_dg: ["compact", "DG only"] };
    const cats = C.map((c) => names[c.exp]);
    const MODES = {
      pieces: { label: "Broken pieces", a: "fixed", b: "clean", an: "as predicted", bn: "after deleting fragments", met: "betti0_err", log: true, yMin: 3, yMax: 150, ticks: [3, 10, 30, 100], fmt: (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0)), cap: "Betti-0 error on unseen data (log scale, lower is better)" },
      dice: { label: "Dice vs threshold", a: "fixed", b: "tuned", an: "threshold 0.5", bn: "threshold tuned on validation", met: "dice", yMin: 0.74, yMax: 0.795, fmt: (v) => v.toFixed(3), cap: "Dice on unseen data (higher is better). Axis starts at 0.74" },
      cldice: { label: "Connectivity", a: "fixed", b: "clean", an: "as predicted", bn: "after deleting fragments", met: "cldice", yMin: 0.8, yMax: 0.86, fmt: (v) => v.toFixed(3), cap: "clDice on unseen data (higher is better). Axis starts at 0.80" },
    };
    group($("#cMode"), Object.entries(MODES).map(([k, v]) => [k, v.label]), "pieces", (k) => {
      const m = MODES[k];
      key($("#cKey"), [[MUTE, m.an], [CORAL, m.bn]]);
      barChart($("#checkChart"), { H: 380, bottom: 60, cats, yMin: m.yMin, yMax: m.yMax, log: !!m.log, ticks: m.ticks, fmtVal: m.fmt, fmtTick: (t) => (m.log ? t : t.toFixed(2)),
        series: [{ name: m.an, vals: C.map((c) => c[m.a][m.met]), color: () => MUTE }, { name: m.bn, vals: C.map((c) => c[m.b][m.met]), color: () => CORAL }] });
      $("#cCap").textContent = m.cap;
    });
  })();

  (function factorial() {
    const chips = [["unet", "U-Net"], ["lite", "Compact"], ["topo", "Topology loss"], ["dg", "DG augmentation"]];
    const on = new Set(), root = $("#facChips"), tag = { A_unet: "A", B_unet_topo: "B", C_unet_topo_dg: "C", D_lite_topo_dg: "D" };
    chips.forEach(([k, t]) => { const b = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, t); b.addEventListener("click", () => { on.has(k) ? on.delete(k) : on.add(k); b.setAttribute("aria-pressed", String(on.has(k))); draw(); }); root.appendChild(b); });
    function draw() {
      const rows = R.factorial.slice().sort((a, b) => (a.backbone === b.backbone ? 0 : a.backbone === "unet" ? -1 : 1) || a.topology - b.topology || a.dg - b.dg);
      const hit = (r) => on.size && [...on].every((k) => (k === "unet" ? r.backbone === "unet" : k === "lite" ? r.backbone === "lite" : k === "topo" ? r.topology : r.dg));
      $("#facTable").innerHTML = `<tr><th>Backbone</th><th>Topology</th><th>DG aug.</th><th>Dice</th><th>clDice</th><th>Betti-0</th><th>Gap</th><th>Stage</th></tr>` +
        rows.map((r) => `<tr class="${hit(r) ? "hl" : ""}"><td>${r.backbone === "unet" ? "U-Net" : "Compact"}</td><td>${r.topology ? "yes" : "-"}</td><td>${r.dg ? "yes" : "-"}</td><td>${r.dice.toFixed(3)}</td><td>${r.cldice.toFixed(3)}</td><td>${r.betti0.toFixed(1)}</td><td>${r.gap.toFixed(3)}</td><td>${tag[r.exp] || ""}</td></tr>`).join("");
    }
    draw();
  })();

  /* ---------------------------------------------------------------- findings carousel */
  (function carousel() {
    const slides = $$("#slides .slide"), nav = $("#dotNav"); let i = 0;
    const dots = slides.map((_, k) => { const b = el("button", { class: "dot-nav", type: "button", role: "tab", "aria-label": `Finding ${k + 1}` }); b.addEventListener("click", () => go(k)); nav.appendChild(b); return b; });
    function go(k) { i = (k + slides.length) % slides.length; slides.forEach((s, j) => s.classList.toggle("on", j === i)); dots.forEach((d, j) => { d.classList.toggle("on", j === i); d.setAttribute("aria-selected", String(j === i)); }); }
    $("#prevSlide").addEventListener("click", () => go(i - 1)); $("#nextSlide").addEventListener("click", () => go(i + 1));
    $("#slides").addEventListener("keydown", (e) => { if (e.key === "ArrowRight") go(i + 1); if (e.key === "ArrowLeft") go(i - 1); });
    let x0 = null;
    $("#slides").addEventListener("touchstart", (e) => (x0 = e.touches[0].clientX), { passive: true });
    $("#slides").addEventListener("touchend", (e) => { if (x0 === null) return; const dx = e.changedTouches[0].clientX - x0; if (Math.abs(dx) > 50) go(i + (dx < 0 ? 1 : -1)); x0 = null; });
    go(0);
  })();

  /* ---------------------------------------------------------------- cost scatter */
  (function tradeoff() {
    let score = "dice", show = "main"; const svg = $("#tradeChart");
    group($("#tScore"), [["dice", "Dice (overlap)"], ["cldice", "clDice (connectivity)"]], "dice", (s) => { score = s; draw(); });
    group($("#tShow"), [["main", "Main models"], ["all", "All configurations"]], "main", (s) => { show = s; draw(); });
    function draw() {
      svg.innerHTML = "";
      const W = 900, H = 440, m = { l: 70, r: 30, t: 20, b: 58 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
      const pts = R.tradeoff.filter((p) => show === "all" || p.main), ys = pts.map((p) => p[score]);
      const lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.18 + 0.002, y0 = lo - pad, y1 = hi + pad, x0 = Math.log10(45), x1 = Math.log10(7000);
      const X = (v) => m.l + ((Math.log10(v) - x0) / (x1 - x0)) * iw, Y = (v) => m.t + ih - ((v - y0) / (y1 - y0)) * ih;
      niceTicks(y0, y1, 5).forEach((t) => { svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" })); svg.appendChild(h("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, t.toFixed(3))); });
      [50, 100, 200, 500, 1000, 2000].forEach((t) => { svg.appendChild(h("line", { x1: X(t), x2: X(t), y1: m.t, y2: m.t + ih, class: "gridline" })); svg.appendChild(h("text", { x: X(t), y: m.t + ih + 20, "text-anchor": "middle" }, t)); });
      svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }));
      svg.appendChild(h("text", { x: m.l + iw / 2, y: H - 10, "text-anchor": "middle" }, "CPU time per image, ms (log scale)"));
      svg.appendChild(h("text", { transform: `translate(16 ${m.t + ih / 2}) rotate(-90)`, "text-anchor": "middle" }, score === "dice" ? "Unseen Dice" : "Unseen clDice"));
      const col = { A_unet: COL.A, B_unet_topo: COL.B, C_unet_topo_dg: COL.C, D_lite_topo_dg: COL.D, E_tglitenet: COL.E };
      const lab = { A_unet: [1, -9, "A  U-Net, 31 M"], B_unet_topo: [1, 4, "B"], C_unet_topo_dg: [1, 12, "C"], D_lite_topo_dg: [-1, 14, "D"], E_tglitenet: [1, 14, "E  TG-LiteNet, 0.21 M"], S_unet_b8: [0, 0, "plain U-Net, base 8"], S_unet_b16: [-1, -46, "plain U-Net, base 16"] };
      pts.slice().sort((a, b) => b.params_m - a.params_m).forEach((p) => {
        const r = 7 + 5 * Math.log10(1 + 30 * p.params_m);
        const c = h("circle", { cx: X(p.cpu_ms), cy: Y(p[score]), r, fill: col[p.exp] || "#2a303a", stroke: p.main ? "#fff" : "rgba(255,255,255,.4)", "stroke-width": p.main ? 1.5 : 1, class: "dot" });
        const html = `<b>${p.name}</b><br>${p.params_m} M parameters<br>${p.cpu_ms} ms per image<br>Dice ${p.dice.toFixed(3)}, clDice ${p.cldice.toFixed(3)}<br>Broken pieces ${p.betti0}`;
        c.addEventListener("mousemove", (e) => showTip(html, e.clientX, e.clientY)); c.addEventListener("mouseleave", hideTip);
        svg.appendChild(c);
        const l = lab[p.exp];
        if (l) svg.appendChild(h("text", { x: X(p.cpu_ms) + l[0] * (r + 6), y: Y(p[score]) + l[1] + (l[0] === 0 ? -r - 8 : 4), "text-anchor": l[0] > 0 ? "start" : l[0] < 0 ? "end" : "middle", style: "fill:#fff;font-weight:700;font-size:12.5px" }, l[2]));
      });
    }
    draw();
  })();

  /* ---------------------------------------------------------------- uncertainty */
  (function uncertainty() {
    const svg = $("#riskChart"), RK = R.risk;
    const LINES = { drive: ["#e8e6e1", ""], stare: ["#e8e6e1", "7 5"], chase_db1: ["#8b93a1", ""], hrf: [CORAL, ""] };
    group($("#uMode"), [["pixel", "pixels"], ["image", "images"]], "pixel", draw);
    function draw(mode) {
      svg.innerHTML = "";
      const cov = mode === "pixel" ? RK.pixel_cov : RK.image_cov;
      const W = 900, H = 360, m = { l: 70, r: 20, t: 16, b: 54 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
      const all = DS.flatMap((d) => RK.datasets[d][mode]), y1 = mode === "pixel" ? Math.ceil(Math.max(...all) * 100 + 0.5) / 100 : Math.ceil(Math.max(...all) * 20) / 20 + 0.02;
      const X = (c) => m.l + ((1 - c) / 0.5) * iw, Y = (v) => m.t + ih - (v / y1) * ih;
      niceTicks(0, y1, 5).forEach((t) => { svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" })); svg.appendChild(h("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, mode === "pixel" ? (t * 100).toFixed(0) + "%" : t.toFixed(2))); });
      [1, 0.9, 0.8, 0.7, 0.6, 0.5].forEach((c) => svg.appendChild(h("text", { x: X(c), y: m.t + ih + 20, "text-anchor": "middle" }, Math.round(c * 100) + "%")));
      svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "axis" }));
      svg.appendChild(h("text", { x: m.l + iw / 2, y: H - 10, "text-anchor": "middle" }, "share kept (most uncertain removed first)"));
      svg.appendChild(h("text", { transform: `translate(16 ${m.t + ih / 2}) rotate(-90)`, "text-anchor": "middle" }, mode === "pixel" ? "pixel error rate" : "average 1 - Dice of kept images"));
      DS.forEach((d) => {
        const v = RK.datasets[d][mode], [c, dash] = LINES[d];
        if (mode === "image") svg.appendChild(h("line", { x1: X(1), x2: X(0.5), y1: Y(v[0]), y2: Y(v[0]), stroke: c, "stroke-width": 1, "stroke-dasharray": "2 5", opacity: 0.55 }));
        svg.appendChild(h("polyline", { points: cov.map((cc, i) => `${X(cc)},${Y(v[i])}`).join(" "), fill: "none", stroke: c, "stroke-width": 2.8, "stroke-dasharray": dash, "stroke-linejoin": "round", "stroke-linecap": "round" }));
        cov.forEach((cc, i) => { const dot = h("circle", { cx: X(cc), cy: Y(v[i]), r: 5, fill: c, stroke: "#06080b", "stroke-width": 2, class: "dot" }); dot.addEventListener("mousemove", (e) => showTip(`<b>${DS_NAME[d]}</b><br>keep ${Math.round(cc * 100)}%<br>${mode === "pixel" ? (v[i] * 100).toFixed(2) + "% error" : "1 - Dice " + v[i].toFixed(3)}`, e.clientX, e.clientY)); dot.addEventListener("mouseleave", hideTip); svg.appendChild(dot); });
      });
      key($("#uKey"), DS.map((d) => [LINES[d][0], `${DS_NAME[d]} (correlation ${mode === "pixel" ? RK.datasets[d].rho_px : RK.datasets[d].rho_img})`]));
      $("#uCap").textContent = mode === "pixel" ? "Good uncertainty: the lines fall steeply as uncertain pixels are removed. They do." : "Good uncertainty: the lines fall as uncertain images are removed (dotted: random removal). They do not.";
    }
  })();

  /* ---------------------------------------------------------------- failure analysis */
  (function failures() {
    const F = R.failure || [];
    const get = (exp, kind, grp) => { const r = F.find((x) => x.exp === exp && x.kind === kind && x.group === grp); return r ? r.value : null; };
    const EX = [["A_unet", "A", "A  U-Net"], ["B_unet_topo", "B", "B  + topology"], ["C_unet_topo_dg", "C", "C  + DG aug."], ["D_lite_topo_dg", "D", "D  compact"], ["E_tglitenet", "E", "E  TG-LiteNet"]];
    const thin = F.filter((x) => x.kind === "recall_by_width" && x.group.startsWith("thin")).map((x) => x.value), wide = F.filter((x) => x.kind === "recall_by_width" && !x.group.startsWith("thin")).map((x) => x.value);
    const far = F.filter((x) => x.kind === "fp_share_far").map((x) => x.value), sA = get("A_unet", "stray_pieces_per_image", "pieces touching no true vessel"), sB = get("B_unet_topo", "stray_pieces_per_image", "pieces touching no true vessel");
    const rng = (a) => (a.length ? `${(Math.min(...a) * 100).toFixed(0)}-${(Math.max(...a) * 100).toFixed(0)}%` : "-");
    $("#failStats").innerHTML = [
      [rng(thin), "of the thinnest vessels (2 px or less) are found", "ph-line-segment"], [rng(wide), "of wider vessels are found", "ph-line-segments"],
      [far.length ? `${(100 - Math.max(...far) * 100).toFixed(0)}-${(100 - Math.min(...far) * 100).toFixed(0)}%` : "-", "of false-alarm pixels sit within 2 px of a real vessel", "ph-ruler"],
      [sA ? `${sA.toFixed(0)} to ${sB.toFixed(0)}` : "-", "stray pieces per image, U-Net to topology loss", "ph-scribble-loop"],
    ].map(([b, t, ic]) => `<div class="spot glass rounded-[24px] p-6"><i class="ph-bold ${ic} text-2xl text-coral"></i><div class="text-4xl font-black tracking-tight mt-3">${b}</div><p class="text-sm text-white/65 mt-1 leading-snug">${t}</p></div>`).join("");
    const KINDS = {
      width: { label: "Vessel width", kind: "recall_by_width", groups: ["thin  (<= 2 px)", "medium (2-4 px)", "thick (> 4 px)"], cats: ["Thin, 2 px or less", "Medium, 2 to 4 px", "Thick, over 4 px"], yMax: 1, fmt: (v) => (v * 100).toFixed(0) + "%", cap: "Share of true vessel pixels found, by vessel width (higher is better)" },
      zone: { label: "Image zone", kind: "recall_by_zone", groups: ["centre", "middle", "edge"], cats: ["Centre", "Middle", "Edge"], yMax: 1, fmt: (v) => (v * 100).toFixed(0) + "%", cap: "Share of true vessel pixels found, by distance from the image centre (higher is better)" },
      fp: { label: "False alarms", kind: "fp_rate_by_zone", groups: ["centre", "middle", "edge"], cats: ["Centre", "Middle", "Edge"], yMax: 0.07, fmt: (v) => (v * 100).toFixed(1) + "%", cap: "False-positive pixels as a share of all pixels in the zone (lower is better)" },
    };
    key($("#fKey"), EX.map(([, t, l]) => [COL[t], l]));
    group($("#fKind"), Object.entries(KINDS).map(([k, v]) => [k, v.label]), "width", (k) => {
      const K = KINDS[k];
      barChart($("#failChart"), { H: 340, cats: K.cats, yMin: k === "fp" ? 0 : 0.6, yMax: K.yMax, fmtVal: K.fmt, fmtTick: (t) => (t * 100).toFixed(0) + "%",
        series: EX.map(([exp, t, l]) => ({ name: l, vals: K.groups.map((g) => get(exp, K.kind, g)), color: () => COL[t] })) });
      $("#fCap").textContent = K.cap + (k === "fp" ? "" : ". Axis starts at 60%");
    });
  })();

  /* ---------------------------------------------------------------- copy buttons */
  $$("pre[data-copy]").forEach((p) => {
    const b = el("button", { type: "button" }, "Copy");
    b.addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(p.innerText.replace(/\nCopy$/, "")); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200); });
    p.appendChild(b);
  });
})();
