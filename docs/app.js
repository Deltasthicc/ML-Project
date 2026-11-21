/* TG-LiteNet project site, edition 3. Vanilla JS + GSAP. All numbers come from data/site.js (scripts/make_site_data.py), which reads results/. */
(function () {
  "use strict";
  const S = window.SITE;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const NS = "http://www.w3.org/2000/svg";
  const INK = "#101512", RED = "#e4432c", GREEN = "#1f6f54", MUTE = "#62665f";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const hasGsap = typeof window.gsap !== "undefined";
  const h = (tag, a = {}, text) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (text !== undefined) e.textContent = text; return e; };
  const el = (tag, a = {}, html) => { const e = document.createElement(tag); for (const k in a) e.setAttribute(k, a[k]); if (html !== undefined) e.innerHTML = html; return e; };
  const f3 = (v) => v.toFixed(3);
  const model = (key) => S.models.find((m) => m.key === key);
  const DSN = { drive: "DRIVE", stare: "STARE", chase_db1: "CHASE_DB1", hrf: "HRF", fives: "FIVES", unseen: "Unseen average" };
  const sgn = (v, d = 3) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

  const stackCards = [];   // filled by the novelties block, read by the scroll loop

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
  function key(container, items) { container.innerHTML = ""; items.forEach(([c, t]) => container.appendChild(el("span", {}, `<i style="background:${c}"></i>${t}`))); }
  function niceTicks(lo, hi, n = 5) {
    const raw = (hi - lo) / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p, step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p, out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function barChart(svg, o) {
    svg.innerHTML = "";
    const W = o.W || 900, H = o.H || 360, m = { l: 64, r: 16, t: 16, b: o.bottom || 52 }, iw = W - m.l - m.r, ih = H - m.t - m.b;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const y = (v) => m.t + ih - ((Math.min(Math.max(v, o.yMin), o.yMax) - o.yMin) / (o.yMax - o.yMin)) * ih;
    niceTicks(o.yMin, o.yMax).forEach((t) => {
      svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "gridline" }));
      svg.appendChild(h("text", { x: m.l - 8, y: y(t) + 4, "text-anchor": "end" }, o.fmtTick ? o.fmtTick(t) : t));
    });
    svg.appendChild(h("line", { x1: m.l, x2: W - m.r, y1: y(Math.max(o.yMin, 0)), y2: y(Math.max(o.yMin, 0)), class: "axis" }));
    const gw = iw / o.cats.length, ns = o.series.length, bw = Math.min(44, (gw * 0.8) / ns), zero = y(Math.max(o.yMin, 0));
    o.cats.forEach((cat, i) => {
      const cx = m.l + gw * i + gw / 2;
      (Array.isArray(cat) ? cat : [cat]).forEach((line, j) => svg.appendChild(h("text", { x: cx, y: m.t + ih + 20 + j * 14, "text-anchor": "middle" }, line)));
      o.series.forEach((s, k) => {
        const v = s.vals[i]; if (v == null) return;
        const x0 = cx - (ns * bw) / 2 + k * bw, top = y(v), a = Math.min(top, zero), hgt = Math.max(1, Math.abs(zero - top));
        const r = h("rect", { x: x0 + 1.5, y: a, width: bw - 3, height: hgt, rx: 4, fill: s.color(i), class: "bar" });
        const info = `<b>${Array.isArray(cat) ? cat.join(" ") : cat}</b><br>${s.name}: ${o.fmtVal(v)}`;
        r.addEventListener("mousemove", (e) => showTip(info, e.clientX, e.clientY)); r.addEventListener("mouseleave", hideTip);
        svg.appendChild(r);
        if (s.err && s.err[i]) {
          const lo = y(Math.max(o.yMin, v - s.err[i])), hi = y(Math.min(o.yMax, v + s.err[i])), mx = x0 + bw / 2;
          [{ x1: mx, x2: mx, y1: lo, y2: hi }, { x1: mx - 4, x2: mx + 4, y1: hi, y2: hi }, { x1: mx - 4, x2: mx + 4, y1: lo, y2: lo }].forEach((a_) => svg.appendChild(h("line", { ...a_, stroke: INK, "stroke-width": 1.4 })));
        }
        if (o.valLabels !== false) svg.appendChild(h("text", { x: x0 + bw / 2, y: (v >= 0 ? top - 7 : top + 14) - (s.err && s.err[i] && v >= 0 ? 8 : 0), "text-anchor": "middle", style: "fill:#101512;font-weight:800;" + (ns > 3 ? "font-size:10px" : "") }, o.fmtVal(v)));
      });
    });
  }

  /* ---------------------------------------------------------------- motion layer */
  (function motion() {
    const bar = $("#progress");
    const onScroll = () => { const d = document.documentElement; bar.style.transform = `scaleX(${d.scrollTop / Math.max(1, d.scrollHeight - innerHeight)})`; };
    addEventListener("scroll", onScroll, { passive: true }); onScroll();
    $$(".spot").forEach((n) => n.addEventListener("mousemove", (e) => { const r = n.getBoundingClientRect(); n.style.setProperty("--mx", e.clientX - r.left + "px"); n.style.setProperty("--my", e.clientY - r.top + "px"); }));
    const links = $$(".nav-link"), secs = links.map((a) => $(a.getAttribute("href")));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a, i) => { const on = secs[i] === e.target; a.classList.toggle("bg-black/5", on); a.classList.toggle("text-black", on); }); }), { rootMargin: "-45% 0px -50% 0px" });
    secs.forEach((s) => s && io.observe(s));
    if (!hasGsap || reduce) return;

    gsap.from("#heroTitle", { y: 46, opacity: 0, duration: 1.2, ease: "power3.out", delay: 0.1 });
    gsap.from(".hero-sub", { y: 24, opacity: 0, duration: 1, ease: "power3.out", delay: 0.5 });
    gsap.from(".hero-cta > *", { y: 20, opacity: 0, duration: 0.9, ease: "power3.out", delay: 0.75, stagger: 0.12 });
    gsap.from(".fcard", { y: 80, opacity: 0, scale: 0.9, duration: 1.3, ease: "power3.out", delay: 0.4, stagger: 0.18 });

    // reveal (position independent)
    const rio = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { gsap.to(e.target, { y: 0, opacity: 1, duration: 1, ease: "power3.out", overwrite: "auto" }); rio.unobserve(e.target); } }), { rootMargin: "0px 0px -6% 0px" });
    $$(".reveal").forEach((n) => { gsap.set(n, { y: 44, opacity: 0 }); rio.observe(n); });

    // image scale and fade: state derived from the element's live position on every frame (never remembered)
    const sf = $$(".sf");
    const apply = (n, st) => {
      if (st === "in") gsap.to(n, { scale: 1, opacity: 1, filter: "brightness(1)", duration: 0.9, ease: "power3.out", overwrite: "auto" });
      else if (st === "out") gsap.to(n, { scale: 0.97, opacity: 0.3, filter: "brightness(0.6)", duration: 0.7, overwrite: "auto" });
      else gsap.to(n, { scale: 0.88, opacity: 0.5, filter: "brightness(1)", duration: 0.5, overwrite: "auto" });
    };
    const sfUpdate = () => sf.forEach((n) => {
      const r = n.getBoundingClientRect(), want = r.top > innerHeight * 0.92 ? "pre" : r.bottom < innerHeight * 0.06 ? "out" : "in";
      if (n._st !== want) { n._st = want; apply(n, want); }
    });
    let raf = 0;
    const sched = () => { if (!raf) raf = setTimeout(() => { raf = 0; sfUpdate(); stackUpdate(); }, 16); };
    addEventListener("scroll", sched, { passive: true }); addEventListener("resize", sched);
    setInterval(() => { sfUpdate(); stackUpdate(); }, 400);
    sfUpdate();
    window.__sched = sched;
  })();

  /* ---------------------------------------------------------------- hero captions, marquee, bento */
  (function top() {
    const v = S.viewer.hrf.maps;
    $("#heroDiceA").textContent = "Dice " + v.v21_H_tglite70k.dice_normalised.toFixed(3);
    $("#heroDiceB").textContent = "Dice " + v.v21_H_tglite70k_scalesel.dice_normalised.toFixed(3);
    const H = model("H_tglite70k[det]"), total = S.n_trainings.v2 + S.n_trainings.v21;
    $("#bTuned").textContent = f3(H.tuned); $("#bSel").textContent = f3(H.sel); $("#bRuns").textContent = total; $("#bKb").textContent = S.int8.int8_kb;
    $("#kvV2").textContent = S.n_trainings.v2; $("#kvV21").textContent = S.n_trainings.v21;
    $("#bRatio").textContent = Math.round(31036481 / 66125);
    const items = [["ph-database", "Trained on 15 DRIVE images"], ["ph-lock-key", "Rules hashed before any test"], ["ph-arrows-in-line-horizontal", "66K parameters"], ["ph-chart-bar", `${total} trained models`],
      ["ph-hard-drives", `${S.int8.int8_kb} KB as INT8`], ["ph-flask", "FIVES held out"], ["ph-scales", "Holm-corrected statistics"], ["ph-ruler", `Scale selection ${sgn(H.sel - H.tuned)} Dice`], ["ph-lightbulb", "4 new ideas tested, 1 supported"]];
    const row = $("#marqueeA"), html = items.map(([ic, t], i) => `<span class="marquee-item ${i % 2 ? "out" : ""}"><i class="ph-bold ${ic}"></i>${t}</span>`).join("");
    row.innerHTML = html + html;
    const bio = new IntersectionObserver((es) => es.forEach((e) => {
      if (!e.isIntersecting) return;
      bio.disconnect();
      $$("#bento .bar-mini > i").forEach((b) => (b.style.width = Math.max(1.2, +b.dataset.w) + "%"));
      if (hasGsap && !reduce) { const o = { v: 0 }; gsap.to(o, { v: 66, duration: 1.6, ease: "power2.out", onUpdate: () => ($("#bParams").textContent = Math.round(o.v)) }); }
    }), { threshold: 0.25 });
    bio.observe($("#bento"));
  })();

  /* ---------------------------------------------------------------- versions accordion */
  (function accordion() {
    const panels = $$(".acc-panel");
    const open = (p) => panels.forEach((q) => { const on = q === p; q.classList.toggle("active", on); q.setAttribute("aria-expanded", String(on)); });
    panels.forEach((p) => {
      p.addEventListener("click", () => open(p)); p.addEventListener("focus", () => open(p));
      p.addEventListener("mouseenter", () => { if (matchMedia("(hover: hover) and (min-width: 1024px)").matches) open(p); });
    });
  })();

  /* ---------------------------------------------------------------- explorer (scatter) */
  (function explorer() {
    const svg = $("#xChart");
    let metric = "tuned", axis = "params", show = "main", pinned = "H_tglite70k[det]";
    const colour = (m) => (m.kind === "headline" ? RED : m.kind === "reference" ? INK : m.kind === "ablation" ? "#f0b4a8" : MUTE);
    const AX = { params: ["parameters (log scale)", [3e4, 3e8], (m) => m.params], cpu_ms: ["CPU time per 512x512 image, ms (log scale)", [15, 6000], (m) => m.cpu_ms], gflops: ["GFLOPs at 512x512 (log scale)", [1.5, 4000], (m) => m.gflops] };
    const MET = { raw: ["Dice, threshold 0.5", "raw"], tuned: ["Dice, tuned threshold", "tuned"], sel: ["Dice, tuned + scale selection", "sel"] };
    group($("#xMetric"), [["raw", "Raw (0.5)"], ["tuned", "Tuned threshold"], ["sel", "Tuned + scale selection"]], "tuned", (v) => { metric = v; draw(); });
    group($("#xAxis"), [["params", "Parameters"], ["cpu_ms", "CPU time"], ["gflops", "GFLOPs"]], "params", (v) => { axis = v; draw(); });
    group($("#xShow"), [["main", "Stages and headline models"], ["all", "Include ablations"]], "main", (v) => { show = v; draw(); });
    key($("#xKey"), [[MUTE, "v2 stages"], [RED, "TG-LiteNet v2.1 (66K)"], [INK, "v2.1 references"], ["#f0b4a8", "v2.1 ablations"]]);
    function draw() {
      svg.innerHTML = "";
      const W = 900, H = 460, mg = { l: 70, r: 24, t: 20, b: 56 }, iw = W - mg.l - mg.r, ih = H - mg.t - mg.b;
      const pts = S.models.filter((m) => (show === "all" || m.kind !== "ablation") && m[MET[metric][1]] != null);
      const ys = pts.map((m) => m[MET[metric][1]]), lo = Math.min(...ys), hi = Math.max(...ys), pad = (hi - lo) * 0.2 + 0.002;
      let y0 = lo - pad, y1 = hi + pad; if (axis === "params" && metric !== "raw") y1 = Math.max(y1, 0.82);
      const [, [xl, xh], xf] = AX[axis], lx0 = Math.log10(xl), lx1 = Math.log10(xh);
      const X = (v) => mg.l + ((Math.log10(v) - lx0) / (lx1 - lx0)) * iw, Y = (v) => mg.t + ih - ((v - y0) / (y1 - y0)) * ih;
      niceTicks(y0, y1, 5).forEach((t) => { svg.appendChild(h("line", { x1: mg.l, x2: W - mg.r, y1: Y(t), y2: Y(t), class: "gridline" })); svg.appendChild(h("text", { x: mg.l - 8, y: Y(t) + 4, "text-anchor": "end" }, t.toFixed(3))); });
      for (let e = Math.ceil(lx0); e <= Math.floor(lx1); e++) {
        [1, 3].forEach((mult) => { const v = mult * Math.pow(10, e); if (v >= xl && v <= xh) { svg.appendChild(h("line", { x1: X(v), x2: X(v), y1: mg.t, y2: mg.t + ih, class: "gridline" })); svg.appendChild(h("text", { x: X(v), y: mg.t + ih + 20, "text-anchor": "middle" }, v >= 1e6 ? v / 1e6 + "M" : v >= 1e3 ? v / 1e3 + "K" : v)); } });
      }
      svg.appendChild(h("line", { x1: mg.l, x2: W - mg.r, y1: mg.t + ih, y2: mg.t + ih, class: "axis" }));
      svg.appendChild(h("text", { x: mg.l + iw / 2, y: H - 10, "text-anchor": "middle" }, AX[axis][0]));
      svg.appendChild(h("text", { transform: `translate(16 ${mg.t + ih / 2}) rotate(-90)`, "text-anchor": "middle" }, MET[metric][0]));
      if (axis === "params" && metric !== "raw") {
        const sx = X(78e3), sy = Y(0.812);
        svg.appendChild(h("path", { d: `M${sx} ${sy - 11} l3 8 9 .5 -7 6 2.5 9 -7.5 -5 -7.5 5 2.5 -9 -7 -6 9 -.5z`, fill: "#7a4fd0", stroke: INK, "stroke-width": 1 }));
        svg.appendChild(h("text", { x: sx + 14, y: sy + 4, style: "font-weight:700" }, "DSAE-Net, 0.812 (published, other protocol)"));
      }
      const labelled = new Set(["H_tglite70k[det]", "A_unet[det]", "C_unet_topo_dg[det]", "E_tglitenet[det]", "R_lite_w16[det]"]);
      pts.slice().sort((a, b) => b.params - a.params).forEach((m) => {
        const cx = X(xf(m)), cy = Y(m[MET[metric][1]]), r = m.kind === "headline" ? 11 : 8, ens = m.kind === "ensemble";
        const node = ens ? h("rect", { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, transform: `rotate(45 ${cx} ${cy})`, fill: colour(m), stroke: "#fff", "stroke-width": 1.5, class: "dot" })
          : h("circle", { cx, cy, r, fill: colour(m), stroke: "#fff", "stroke-width": 1.5, class: "dot" + (m.key === pinned ? " on" : "") });
        const html = `<b>${m.label}</b><br>${m.params >= 1e6 ? (m.params / 1e6).toFixed(2) + " M" : Math.round(m.params / 1e3) + " K"} parameters, ${m.cpu_ms.toFixed(0)} ms CPU<br>unseen Dice ${f3(m[MET[metric][1]])}`;
        node.addEventListener("mousemove", (e) => { showTip(html, e.clientX, e.clientY); detail(m); });
        node.addEventListener("mouseleave", () => { hideTip(); detail(model(pinned)); });
        node.addEventListener("click", () => { pinned = m.key; draw(); detail(m); });
        svg.appendChild(node);
        if (labelled.has(m.key)) svg.appendChild(h("text", { x: cx + (m.key === "H_tglite70k[det]" ? 15 : 12), y: cy + (m.key === "A_unet[det]" ? 18 : -10), style: "fill:#101512;font-weight:800;font-size:12px" }, m.key === "H_tglite70k[det]" ? "66K" : m.label.split("  ")[0].replace("Depthwise 212K (size-rule headline)", "Depthwise 212K").replace("TG-LiteNet v2 (486K)", "E")));
      });
      $("#xCap").textContent = `${MET[metric][0]} against ${AX[axis][0].replace(" (log scale)", "")}`;
      detail(model(pinned));
    }
    function detail(m) {
      const bars = [["DRIVE", m.drive], ["STARE", m.stare], ["CHASE_DB1", m.chase], ["HRF", m.hrf], ["FIVES", m.fives]].map(([n, v]) => `<div class="mb-2"><div class="flex justify-between text-xs font-bold text-black/60"><span>${n}</span><span>${f3(v)}</span></div><div class="bar-mini"><i class="${n === "HRF" ? "hot" : ""}" style="width:${Math.max(2, ((v - 0.65) / 0.25) * 100)}%"></i></div></div>`).join("");
      $("#xDetail").innerHTML = `<div class="flex items-center gap-2 text-xs font-bold"><span class="rounded-full px-3 py-1 ${m.version === "v2.1" ? "bg-vermilion text-white" : "bg-black/10"}">${m.version}</span><span class="text-black/50">${m.n} seed${m.n > 1 ? "s" : ""}</span></div>
        <h3 class="mt-3 text-2xl font-black tracking-tight leading-tight">${m.label}</h3>
        <div class="grid grid-cols-3 gap-2 mt-4 text-center"><div class="rounded-2xl bg-black/5 py-3"><b class="block text-lg font-black">${m.params >= 1e6 ? (m.params / 1e6).toFixed(2) + "M" : Math.round(m.params / 1e3) + "K"}</b><span class="text-[11px] font-semibold text-black/55">params</span></div><div class="rounded-2xl bg-black/5 py-3"><b class="block text-lg font-black">${m.gflops}</b><span class="text-[11px] font-semibold text-black/55">GFLOPs</span></div><div class="rounded-2xl bg-black/5 py-3"><b class="block text-lg font-black">${m.cpu_ms.toFixed(0)}</b><span class="text-[11px] font-semibold text-black/55">ms CPU</span></div></div>
        <div class="mt-5 text-sm font-bold text-black/60 mb-2">Dice per dataset (tuned threshold)</div>${bars}
        <div class="grid grid-cols-2 gap-2 mt-4 text-sm"><div>Unseen raw <b>${f3(m.raw)}</b></div><div>Unseen tuned <b>${f3(m.tuned)}</b></div><div>+ scale sel. <b>${f3(m.sel)}</b></div><div>clDice <b>${f3(m.cldice)}</b></div><div>Betti-0 raw <b>${m.betti_raw.toFixed(1)}</b></div><div>Betti-0 clean <b>${m.betti_clean.toFixed(1)}</b></div></div>`;
    }
    draw();
  })();

  /* ---------------------------------------------------------------- live viewer */
  (function viewer() {
    const cv = $("#vCanvas"), ctx = cv.getContext("2d", { willReadFrequently: true });
    const MODELS = [["v21_H_tglite70k", "66K (v2.1)"], ["v21_H_tglite70k_scalesel", "66K + scale selection"], ["v21_R_lite_w16", "Depthwise 212K"], ["v21_Z_v2recipe", "66K, v2 recipe"], ["v2_E_tglitenet", "TG-LiteNet v2 (486K)"], ["v2_C_unet_topo_dg", "v2 stage C (31M)"]];
    const MODES = [["err", "Errors"], ["pred", "Prediction"], ["prob", "Probability"], ["gt", "Ground truth"], ["photo", "Photo"]];
    const cache = {}, order = ["drive", "stare", "chase_db1", "hrf", "fives"];
    let ds = "hrf", mdl = "v21_H_tglite70k", mode = "err", pending = false;
    const load = (src) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
    const pix = (img) => { const c = document.createElement("canvas"); c.width = img.width; c.height = img.height; const x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(img, 0, 0); return x.getImageData(0, 0, c.width, c.height).data; };
    async function ensure(d) {
      if (cache[d]) return cache[d];
      const base = `assets/${d}/`;
      const imgs = await Promise.all([load(base + "image.jpg"), load(base + "gt.png"), load(base + "fov.png"), ...MODELS.map(([k]) => load(`${base}prob_${k}.png`))]);
      const w = imgs[0].width, hh = imgs[0].height, ch = (im) => { const p = pix(im), o = new Uint8Array(w * hh); for (let i = 0; i < o.length; i++) o[i] = p[i * 4]; return o; };
      const c = { w, h: hh, photo: pix(imgs[0]), gt: ch(imgs[1]), fov: ch(imgs[2]), prob: {} };
      MODELS.forEach(([k], i) => (c.prob[k] = ch(imgs[3 + i])));
      c.gtb = new Uint8Array(w * hh); for (let i = 0; i < c.gtb.length; i++) c.gtb[i] = c.gt[i] > 127 && c.fov[i] > 127 ? 1 : 0;
      c.gtPieces = label(c.gtb, w, hh).n;
      return (cache[d] = c);
    }
    function label(mask, w, hh) {
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
    const chosen = () => S.viewer[ds].maps[mdl].threshold;
    function update() {
      pending = false;
      const c = cache[ds]; if (!c) return;
      const thr = +$("#thr").value, minsz = +$("#minsz").value;
      $("#thrOut").textContent = thr.toFixed(2); $("#minszOut").textContent = minsz + " px";
      const t = thr * 255, w = c.w, hh = c.h, p = c.prob[mdl], pred = new Uint8Array(w * hh);
      for (let i = 0; i < pred.length; i++) pred[i] = p[i] >= t && c.fov[i] > 127 ? 1 : 0;
      const L = label(pred, w, hh); let pieces = L.n;
      if (minsz > 0) { for (let i = 0; i < pred.length; i++) if (pred[i] && L.sizes[L.lab[i]] < minsz) pred[i] = 0; pieces = L.sizes.reduce((a, s, k) => a + (k > 0 && s >= minsz ? 1 : 0), 0); }
      let tp = 0, fp = 0, fn = 0;
      for (let i = 0; i < pred.length; i++) { const g = c.gtb[i]; if (pred[i] && g) tp++; else if (pred[i]) fp++; else if (g) fn++; }
      const dice = (2 * tp) / Math.max(1, 2 * tp + fp + fn);
      $("#rDice").textContent = dice.toFixed(3); $("#rPieces").textContent = pieces; $("#rRec").textContent = (tp / Math.max(1, tp + fn)).toFixed(2); $("#rPre").textContent = (tp / Math.max(1, tp + fp)).toFixed(2);
      const name = MODELS.find((m) => m[0] === mdl)[1], thin = c.gtPieces > 100;
      $("#vInsight").innerHTML = `<b>${name}</b> on ${DSN[ds]}: <b>${pieces}</b> predicted vessel pieces` + (thin ? ". The ground-truth piece count is not meaningful here: thin labels split into many fragments at the working scale." : ` against <b>${c.gtPieces}</b> in the ground truth` + (minsz === 0 && pieces > 3 * c.gtPieces ? ". Most extra pieces are tiny specks: drag the fragment slider up." : "."))
        + (mdl === "v21_H_tglite70k_scalesel" ? ` The scale rule chose ${S.viewer[ds].selected_scale === 1 ? "scale 1 (unchanged)" : "scale " + S.viewer[ds].selected_scale} for this image.` : "");
      cv.width = w; cv.height = hh;
      const out = ctx.createImageData(w, hh), o = out.data;
      for (let i = 0; i < pred.length; i++) {
        const k = i * 4; let r = 242, g = 238, b = 230;
        if (mode === "photo") { r = c.photo[k]; g = c.photo[k + 1]; b = c.photo[k + 2]; }
        else if (mode === "gt") { const v = c.gtb[i] ? 16 : c.fov[i] > 127 ? 255 : 220; r = g = b = v; }
        else if (mode === "pred") { const v = pred[i] ? 16 : c.fov[i] > 127 ? 255 : 220; r = g = b = v; }
        else if (mode === "prob") { const v = 255 - p[i]; r = g = b = c.fov[i] > 127 ? v : 220; }
        else {
          if (c.fov[i] <= 127) { r = g = b = 220; }
          else if (pred[i] && c.gtb[i]) { r = 16; g = 21; b = 18; }
          else if (pred[i]) { r = 228; g = 67; b = 44; }
          else if (c.gtb[i]) { r = 59; g = 130; b = 246; }
          else { r = g = b = 255; }
        }
        o[k] = r; o[k + 1] = g; o[k + 2] = b; o[k + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
    }
    const sched = () => { if (!pending) { pending = true; setTimeout(update, 0); } };   // setTimeout: also runs while the tab is in the background
    const reset = () => { $("#thr").value = chosen(); sched(); };
    async function setDs(d) { ds = d; try { await ensure(d); reset(); } catch (e) { cv.width = 640; cv.height = 200; ctx.fillStyle = "#0b0f0c"; ctx.fillRect(0, 0, 640, 200); ctx.fillStyle = "#fff"; ctx.font = "16px sans-serif"; ctx.fillText("Open this page through a web server (python -m http.server in docs/) or GitHub Pages.", 20, 100); } }
    group($("#vDs"), order.map((d) => [d, DSN[d]]), "hrf", setDs);
    group($("#vModel"), MODELS, "v21_H_tglite70k", (k) => { mdl = k; reset(); });
    group($("#vMode"), MODES, "err", (k) => { mode = k; sched(); });
    $("#thr").addEventListener("input", sched); $("#minsz").addEventListener("input", sched); $("#vReset").addEventListener("click", reset);
  })();

  /* ---------------------------------------------------------------- scale rule: compare slider + chart */
  (function scaleRule() {
    const order = ["hrf", "fives", "stare", "chase_db1", "drive"];
    const A = $("#cmpA"), B = $("#cmpB"), cmp = $("#cmp"), range = $("#cmpRange");
    range.addEventListener("input", () => cmp.style.setProperty("--cut", range.value + "%"));
    group($("#cDs"), order.map((d) => [d, DSN[d]]), "hrf", (d) => {
      A.src = `assets/${d}/err_v21_H_tglite70k.png`; B.src = `assets/${d}/err_v21_H_tglite70k_scalesel.png`;
      const v = S.viewer[d], a = v.maps.v21_H_tglite70k.dice_normalised, b = v.maps.v21_H_tglite70k_scalesel.dice_normalised, sh = S.scale_share[d];
      const pick = Object.entries(sh).sort((x, y) => y[1] - x[1])[0];
      $("#cmpCap").innerHTML = `${DSN[d]}: Dice ${a.toFixed(3)} at scale 1, ${b.toFixed(3)} with the rule (this image). Across the dataset the rule keeps scale ${pick[0]} for ${Math.round(100 * pick[1])}% of images.`;
    });
    const sets = [["H_tglite70k[det]", "66K", RED], ["R_lite_w16[det]", "Depthwise 212K", INK], ["E_tglitenet[det]", "v2 stage E", "#8f8a7c"], ["C_unet_topo_dg[det]", "v2 stage C (31M)", GREEN]];
    key($("#sKey"), sets.map(([, l, c]) => [c, l]));
    const d4 = ["stare", "chase_db1", "hrf", "fives"];
    barChart($("#sChart"), { W: 520, H: 300, bottom: 40, cats: d4.map((d) => DSN[d]), yMin: -0.14, yMax: 0.05, fmtVal: (v) => sgn(v, 3), fmtTick: (t) => t.toFixed(2),
      series: sets.map(([k, l, c]) => ({ name: l, vals: d4.map((d) => S.scale_delta[k][d]), color: () => c })), valLabels: false });
  })();

  /* ---------------------------------------------------------------- novelties: stacking cards */
  function stackUpdate() {
    if (typeof gsap === "undefined" || reduce) return;
    stackCards.forEach((c, i) => {
      const next = stackCards[i + 1]; if (!next) return;
      const top = parseFloat(getComputedStyle(c).top) || 0, nTop = next.getBoundingClientRect().top;
      const t = Math.min(1, Math.max(0, 1 - (nTop - top) / (innerHeight * 0.55)));
      gsap.set(c, { scale: 1 - 0.05 * t, filter: `brightness(${1 - 0.08 * t})`, y: -6 * t });
    });
  }
  (function novelties() {
    const root = $("#stack"), A = S.ablations, n1 = S.novelty1;
    const abl = (run, m) => { const v = A[run][m]; return `${sgn(v.delta, m === "betti" ? 1 : 4)} [±${v.ci.toFixed(m === "betti" ? 1 : 4)}] ${v.clear ? "clear" : "not clear"}`; };
    const cards = [
      { id: "N1", owner: "Abhiraj Agarwal", bg: "var(--paper2)", fg: INK, title: "Connectivity reliability score", verdict: ["mid", "Promising lead", "ph-lightbulb"],
        idea: "Flag an unreliable prediction by counting how fragmented it is: label-free, single pass, no extra cost.",
        res: [`Against MC-dropout entropy (stage E): ${sgn(n1.E_vs_mc.delta)} [±${n1.E_vs_mc.ci.toFixed(3)}], clear.`, `Against single-pass entropy (stage E): ${sgn(n1.E_det.delta)}, not clear.`, `For the 66K model, entropy is the stronger score: ${sgn(n1.H.delta)} [±${n1.H.ci.toFixed(3)}].`],
        note: "Two secondary connectivity scores (end-point density, small-fragment share) ranked failures better than entropy for most models. A promising lead for follow-up work." },
      { id: "N2", owner: "Tanish Shitanshu", bg: INK, fg: "#fff", title: "Calibre-guided scale selection", verdict: ["ok", "Supported", "ph-check-circle"],
        idea: "Run each image at four scales and keep the one whose predicted vessel width matches what the model saw on DRIVE.",
        res: [`Unseen Dice, 66K model: ${S.scale_unseen["H_tglite70k[det]"].text}.`, `HRF ${sgn(S.scale_delta["H_tglite70k[det]"].hrf)}, STARE and CHASE_DB1 unchanged.`, `FIVES ${sgn(S.scale_delta["H_tglite70k[det]"].fives)}: on thick vessels the default scale is already the better choice.`],
        note: "The rule was set on DRIVE only, and it passed its validation safety check for every model." },
      { id: "N3", owner: "Shashwat Rajan", bg: "var(--card)", fg: INK, title: "Topology-aware distillation", verdict: ["mid", "Open question", "ph-compass"],
        idea: "Let the small student copy a large teacher's soft centrelines, not only its pixels.",
        res: [`Dice ${abl("Z_no_kdskel", "dice")}.`, `clDice ${abl("Z_no_kdskel", "cldice")}.`, `Betti-0 ${abl("Z_no_kdskel", "betti")} (fewer pieces, but not clear).`],
        note: "With 3 seeds the test has modest power; more seeds would settle it." },
      { id: "N4", owner: "Tanish Shitanshu", bg: "#e7ece0", fg: INK, title: "Skeleton-level consistency", verdict: ["mid", "Dice trade-off", "ph-scales"],
        idea: "Ask the augmented view to reproduce the clean view's centrelines instead of its pixels.",
        res: [`Dice ${abl("Z_pixcons", "dice")}: a clear Dice trade-off against plain pixel consistency.`, `clDice ${abl("Z_pixcons", "cldice")}.`, `Betti-0 ${abl("Z_pixcons", "betti")}.`],
        note: "Pixel-level consistency stays the recommended choice, which narrows the design space for future work." },
    ];
    cards.forEach((c, i) => {
      const art = el("article", { class: "stack-card", style: `--i:${i}; background:${c.bg}; color:${c.fg}` });
      art.innerHTML = `<div class="flex flex-wrap items-start justify-between gap-6"><div><div class="big-n" style="color:#e4432c">${c.id}</div><div class="mt-1 text-sm font-bold" style="opacity:.62">Owner: ${c.owner}</div></div><span class="verdict ${c.verdict[0]}"><i class="ph-bold ${c.verdict[2]} text-base"></i>${c.verdict[1]}</span></div>
        <h3 class="mt-6 text-3xl md:text-4xl font-black tracking-tight leading-tight">${c.title}</h3>
        <p class="mt-3 text-lg max-w-2xl" style="opacity:.78">${c.idea}</p>
        <ul class="mt-6 space-y-2 text-[15.5px] font-semibold max-w-3xl">${c.res.map((r) => `<li class="flex gap-3"><i class="ph-bold ph-caret-right mt-1 text-vermilion"></i><span>${r}</span></li>`).join("")}</ul>
        <p class="mt-6 text-sm max-w-2xl" style="opacity:.65">${c.note}</p>`;
      root.appendChild(art); stackCards.push(art);
    });
    root.appendChild(el("div", { style: "height: 12vh" }));
  })();

  /* ---------------------------------------------------------------- controls: stage explorer, factorial, bars */
  (function controls() {
    const tags = ["A", "B", "C", "D", "E"], COLS = { A: "#9a9a90", B: "#6d9bc7", C: GREEN, D: "#e8a13a", E: RED };
    const MET = { dice: ["Dice", "higher is better"], cldice: ["clDice", "higher is better"], betti0_err: ["Betti-0 error", "lower is better"] };
    let mode = "raw", metric = "dice", dsel = "unseen";
    key($("#kKey"), tags.map((t) => [COLS[t], S.stage_labels[t]]));
    group($("#kMode"), [["raw", "Raw (threshold 0.5)"], ["clean", "Clean (delete fragments)"], ["tuned", "Tuned threshold"]], "raw", (v) => { mode = v; draw(); });
    group($("#kMetric"), Object.entries(MET).map(([k, v]) => [k, v[0]]), "dice", (v) => { metric = v; draw(); });
    group($("#kDs"), ["unseen", "drive", "stare", "chase_db1", "hrf", "fives"].map((d) => [d, DSN[d]]), "unseen", (v) => { dsel = v; draw(); });
    function draw() {
      const vals = tags.map((t) => S.stages[mode][t][`${dsel}_${metric}`][0]), errs = tags.map((t) => S.stages[mode][t][`${dsel}_${metric}`][1]);
      const lo = Math.min(...vals.map((v, i) => v - errs[i])), hi = Math.max(...vals.map((v, i) => v + errs[i]));
      let yMin, yMax;
      if (metric === "betti0_err") { yMin = 0; yMax = Math.ceil((hi * 1.15) / 10) * 10; } else { const span = Math.max(0.02, hi - lo); yMin = Math.floor((lo - span * 0.6) * 100) / 100; yMax = Math.ceil((hi + span * 0.35) * 100) / 100; }
      barChart($("#kChart"), { cats: tags.map((t) => [t, S.stage_labels[t].replace(/^\w\s+/, "")]), series: [{ name: MET[metric][0], vals, err: errs, color: (i) => COLS[tags[i]] }], yMin, yMax,
        fmtVal: (v) => (metric === "betti0_err" ? v.toFixed(1) : v.toFixed(3)), fmtTick: (t) => (metric === "betti0_err" ? t : t.toFixed(2)) });
      $("#kCap").textContent = `${MET[metric][0]}, ${DSN[dsel]}, ${mode}, ${MET[metric][1]}` + (metric !== "betti0_err" ? `. Axis starts at ${yMin.toFixed(2)}` : "");
      const best = vals.indexOf(metric === "betti0_err" ? Math.min(...vals) : Math.max(...vals));
      let msg = `<b>Best here: stage ${tags[best]}</b> (${metric === "betti0_err" ? vals[best].toFixed(1) : vals[best].toFixed(3)}). `;
      if (metric === "betti0_err" && dsel === "unseen") msg += mode === "raw" ? "The topology loss roughly halves the broken pieces (about 73 to 34)." : "After deleting fragments, every stage sits at about 4.5 pieces, the plain U-Net included. Raw fragment counts are not a reliable topology measure.";
      else if (metric === "dice" && dsel === "unseen") msg += mode === "tuned" ? "With the threshold chosen on validation images the topology-trained stages gain 0.013 to 0.027 Dice while the plain U-Net does not: much of the Dice 'cost' of topology at 0.5 is calibration." : "At the threshold-0.5 headline the stages are within about 0.012 Dice of each other.";
      else msg += "Hover a bar for exact values.";
      $("#kInsight").innerHTML = msg;
    }
    // factorial
    const chips = [["unet", "U-Net"], ["compact", "Compact"], ["topo", "Topology loss"], ["dg", "DG augmentation"]], on = new Set(), root = $("#fChips");
    chips.forEach(([k, t]) => { const b = el("button", { class: "chip", type: "button", "aria-pressed": "false" }, t); b.addEventListener("click", () => { on.has(k) ? on.delete(k) : on.add(k); b.setAttribute("aria-pressed", String(on.has(k))); fdraw(); }); root.appendChild(b); });
    function fdraw() {
      const hit = (r) => on.size && [...on].every((k) => (k === "unet" ? r.backbone === "U-Net" : k === "compact" ? r.backbone === "Compact" : k === "topo" ? r.topology : r.dg));
      $("#fTable").innerHTML = `<tr><th>Backbone</th><th>Topology</th><th>DG aug.</th><th>Seeds</th><th>Dice (raw)</th><th>clDice</th><th>Betti-0</th><th>Dice (tuned)</th><th>Gap (tuned)</th></tr>` +
        S.factorial.map((r) => `<tr class="${hit(r) ? "hl" : ""}"><td>${r.backbone}</td><td>${r.topology ? "yes" : "-"}</td><td>${r.dg ? "yes" : "-"}</td><td>${r.n}</td><td>${f3(r.dice_raw)}</td><td>${f3(r.cldice_raw)}</td><td>${r.betti_raw.toFixed(1)}</td><td>${f3(r.dice_tuned)}</td><td>${f3(r.gap_tuned)}</td></tr>`).join("");
    }
    // cpu bars and uncertainty bars
    const E = S.efficiency, H = model("H_tglite70k[det]");
    const cpu = [["31 M U-Net (stage A)", E.A_unet.cpu_4t, 0], ["Depthwise 212K", E.K_lite_w16.cpu_4t, 0], ["Plain U-Net 1.9 M", E.K_unet_w16.cpu_4t, 0], ["Plain U-Net 486K (v2)", E.K_unet_w8.cpu_4t, 0], ["TG-LiteNet v2.1 (66K)", H.cpu_ms, 1]];
    $("#cpuBars").innerHTML = cpu.map(([n, v, hot]) => `<div><div class="flex justify-between text-sm font-bold"><span>${n}</span><span>${v.toFixed(0)} ms</span></div><div class="bar-mini mt-1"><i class="${hot ? "hot" : ""}" style="width:${Math.max(2, (v / E.A_unet.cpu_4t) * 100)}%"></i></div></div>`).join("");
    const U = S.uncertainty, Ee = U["E_tglitenet[det]"], Em = U["E_tglitenet[mc]"];
    const rows = [["Pixel error vs uncertainty (stage E)", Ee.rho_px], ["Image error vs uncertainty, single pass", Ee.rho_img], ["Image error vs uncertainty, MC dropout", Em.rho_img]];
    $("#uncBars").innerHTML = rows.map(([n, v]) => `<div><div class="flex justify-between text-sm font-bold"><span>${n}</span><span>ρ ${sgn(v, 2)}</span></div><div class="bar-mini mt-1"><i class="${v < 0 ? "hot" : ""}" style="width:${Math.max(2, Math.abs(v) * 100 * 2.2)}%"></i></div></div>`).join("") +
      `<p class="text-sm text-black/55">Calibration (ECE): ${Ee.ece.toFixed(3)} raw, ${Ee.ece_ts.toFixed(3)} after temperature scaling fitted on DRIVE validation.</p>`;
    draw(); fdraw();
  })();

  /* ---------------------------------------------------------------- copy buttons */
  $$("pre[data-copy]").forEach((p) => {
    const b = el("button", { type: "button" }, "Copy");
    b.addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(p.innerText.replace(/\nCopy$/, "")); b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1200); });
    p.appendChild(b);
  });
})();
