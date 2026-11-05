/* Agentic GP replay player (needs agp_core.js first).
 *
 *   const player = new AgpReplayPlayer(rootElement, {height: 560, autoplay: true});
 *   player.setData(replayDict);          // replay schema v1 dict (+ optional `view` blocks from Python)
 *   player.onReport = (state) => ...;    // called on pause / scrub release / follow change if opts.report
 *
 * The whole replay is loaded once; playback, speed, scrubbing and interpolation all happen here
 * in the browser (requestAnimationFrame), driven by the race clock `t`. Python is not involved.
 * Keys (when the player has focus): Space play/pause, <- -> 5 s (Shift: 1 s), + - speed,
 * Home restart, 1-9 follow the car in that position, 0 / Esc overview.
 */

const AGP_SPEEDS = [0.25, 0.5, 1, 2, 4, 8, 16];
const AGP_TRAIL_S = 4.0;            // seconds of trail behind every car

class AgpReplayPlayer {
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = Object.assign({ height: 560, autoplay: false, speed: 1, report: false, labels: "auto",
                                follow: null, fill: false, vmax: false, trails: true }, opts);
    this.onReport = null;
    this.data = null; this.cars = []; this.track = null; this.id = null;
    this.t = 0; this.duration = 0; this.playing = false; this.speed = this.opts.speed;
    this.followId = null; this.zoom = 2.6; this.overlayClosed = false;
    this.cam = new AgpCamera(); this.last = 0; this.towerRows = []; this.dragging = false;
    this._build();
    this._frame = this._frame.bind(this);
    this.raf = requestAnimationFrame(this._frame);
  }

  /* ---------------------------------------------------------------- DOM */
  _build() {
    const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; };
    this.root.innerHTML = "";
    const wrap = this.wrap = el("div", "agp agp-replay");
    wrap.tabIndex = 0;
    const stage = this.stage = el("div", "agp-stage");
    this.canvas = el("canvas"); stage.append(this.canvas);
    const bar = this.bar = el("div", "agp-bar");
    this.btnRestart = el("button", "agp-btn agp-restart", "⏮"); this.btnRestart.title = "Restart (Home)";
    this.btnPlay = el("button", "agp-btn agp-play", "▶ Play"); this.btnPlay.title = "Play / pause (Space)";
    const sw = el("div", "agp-scrub-wrap");
    this.marks = el("canvas", "agp-marks");
    this.scrub = el("input", "agp-scrub"); Object.assign(this.scrub, { type: "range", min: 0, max: 1000, step: 1, value: 0 });
    this.scrub.title = "Scrub (← →)";
    sw.append(this.marks, this.scrub);
    this.clock = el("span", "agp-clock", "0.0 / 0.0");
    this.selSpeed = el("select", "agp-select agp-speed"); this.selSpeed.title = "Playback speed (+ -)";
    for (const s of AGP_SPEEDS) { const o = el("option"); o.value = s; o.textContent = `${s}×`; this.selSpeed.append(o); }
    this.selFollow = el("select", "agp-select agp-follow"); this.selFollow.title = "Follow a car (click it, or 1-9)";
    bar.append(this.btnRestart, this.btnPlay, sw, this.clock, this.selSpeed, this.selFollow);
    wrap.append(stage, bar);
    this.root.append(wrap);
    this.ctx = this.canvas.getContext("2d");

    this.btnPlay.onclick = () => this.toggle();
    this.btnRestart.onclick = () => { this.seek(0); this.overlayClosed = false; if (!this.playing) this.toggle(); };
    this.scrub.oninput = () => { this.dragging = true; this.seek(this.scrub.value / 1000 * this.duration); };
    this.scrub.onchange = () => { this.dragging = false; this._report(); };
    this.selSpeed.onchange = () => { this.speed = +this.selSpeed.value; };
    this.selFollow.onchange = () => this.follow(this.selFollow.value || null);
    wrap.onkeydown = (e) => { if (this.handleKey(e)) { e.preventDefault(); e.stopPropagation(); } };
    this.canvas.onclick = (e) => this._click(e.offsetX, e.offsetY);
    this.canvas.onwheel = (e) => {
      if (!this.followId) return;                          // only zoom while following (page scroll stays free)
      e.preventDefault(); this.zoom = agpClamp(this.zoom * Math.exp(-e.deltaY * 0.0015), 0.6, 8);
    };
    this.ro = new ResizeObserver(() => this._layout()); this.ro.observe(this.wrap);
    if (this.opts.fill) { this._onWinResize = () => this._layout(); window.addEventListener("resize", this._onWinResize); }
    this._layout();
  }

  _layout() {
    if (this.opts.fill) {
      const h = Math.max(260, window.innerHeight - this.bar.offsetHeight);
      this.stage.style.height = `${h}px`;
    } else if (this.opts.fit) {                            // presenter mode: fill the window below the canvas top
      const h = agpWindowHeight(this.stage, this.bar.offsetHeight + 16, 320);
      if (this.stage.style.height !== `${h}px`) this.stage.style.height = `${h}px`;
      if (!this._onWinResize) { this._onWinResize = () => this._layout(); window.addEventListener("resize", this._onWinResize); }
    } else {
      this.stage.style.height = `${Math.max(200, +this.opts.height || 560)}px`;
    }
    this._drawMarks();
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    if (this.ro) this.ro.disconnect();
    if (this._onWinResize) window.removeEventListener("resize", this._onWinResize);
  }

  /* ---------------------------------------------------------------- data */
  setData(replay, options) {
    if (options) Object.assign(this.opts, options);
    this._layout();
    if (!replay || !replay.track || !replay.cars) {         // nothing to show: an empty canvas
      this.data = null; this.track = null; this.cars = []; this.id = null; this.duration = 0; this.playing = false;
      this._syncButtons(); return;
    }
    const id = (replay.view && replay.view.id) || `${replay.title}|${replay.created_utc}|${replay.cars.length}`;
    const isNew = id !== this.id;
    this.data = replay; this.id = id;
    if (!this.track || this.track.id !== ((replay.track.view || {}).id || replay.track.key) || isNew)
      this.track = new AgpTrack(replay.track);
    this.cars = replay.cars.map((c, i) => agpPrepareCar(c, i));
    const seen = {};                                       // 3-letter codes must be unique: "PL1", "PL2", ...
    for (const pc of this.cars) seen[pc.code] = (seen[pc.code] || 0) + 1;
    const used = {};
    for (const pc of this.cars) if (seen[pc.code] > 1) {
      const num = (/(\d+)\s*$/.exec(pc.name) || [])[1] || String((used[pc.code] = (used[pc.code] || 0) + 1));
      pc.code = pc.code.slice(0, 3 - Math.min(2, num.length)) + num.slice(-2);
    }
    this.duration = Math.max(0, ...this.cars.map((c) => c.end));
    this.laps = replay.laps || 1;
    this._ticker();
    this.selFollow.innerHTML = "";
    const o0 = document.createElement("option"); o0.value = ""; o0.textContent = "Overview"; this.selFollow.append(o0);
    for (const pc of this.cars) {
      const o = document.createElement("option"); o.value = pc.id; o.textContent = `Follow ${pc.name}`;
      this.selFollow.append(o);
    }
    if (options && "speed" in options) this.speed = +options.speed || 1;
    this.selSpeed.value = String(this.speed);
    if (isNew) {
      this.t = 0; this.overlayClosed = false; this.cam.ready = false;
      this.followId = null;
      this.playing = !!this.opts.autoplay;
    }
    if (options && "follow" in options) this.followId = options.follow || null;
    if (this.followId && !this.cars.some((c) => c.id === this.followId)) this.followId = null;
    this.selFollow.value = this.followId || "";
    this._syncButtons();
    this._drawMarks();
  }

  _ticker() {                                              // radio + race events, by race time
    const items = [];
    for (const pc of this.cars) {
      const c = pc.car;
      for (const m of c.messages || []) items.push({ t: m.t, pc, kind: "radio", text: `“${m.text}”` });
      for (const e of c.events || []) {
        let text = null;
        if (e.kind === "crash") text = `hits the wall  +${(+e.value).toFixed(0)}s`;
        else if (e.kind === "lap" && this.laps > 1) text = `lap ${e.lap}  ${agpFmtLap(e.value)}`;
        else if (e.kind === "finish") text = `finishes  ${agpFmtLap(e.value)}`;
        else if (e.kind === "dnf") text = (e.text || "DNF").toUpperCase();
        else if (e.kind === "zone") text = `too fast in a zone  +${(+e.value || 1).toFixed(0)}s`;
        else if (e.kind === "stall") text = "drive() too slow - coasting";
        else if (e.kind === "driver_error") text = `error: ${e.text}`;
        if (text) items.push({ t: e.t, pc, kind: e.kind, text });
      }
    }
    items.sort((a, b) => a.t - b.t);
    this.items = items;
    this.itemT = Float64Array.from(items, (i) => i.t);
  }

  /* ---------------------------------------------------------------- control */
  toggle() {
    if (!this.data) return;
    this.playing = !this.playing;
    if (this.playing && this.t >= this.duration - 1e-6) { this.t = 0; this.overlayClosed = false; }
    this._syncButtons();
    if (!this.playing) this._report();
  }
  seek(t) { this.t = agpClamp(t, 0, this.duration); if (this.t < this.duration) this.overlayClosed = false; }
  follow(id) {
    this.followId = id || null; this.selFollow.value = this.followId || "";
    this._report();
  }
  setSpeed(s) { this.speed = s; this.selSpeed.value = String(s); }
  _syncButtons() { this.btnPlay.textContent = this.playing ? "❚❚ Pause" : "▶ Play"; }
  _report() {
    if (this.opts.report && this.onReport) this.onReport({ t: +this.t.toFixed(2), follow: this.followId });
  }
  handleKey(e) {
    const k = e.key;
    if (k === " " || k === "k") { this.toggle(); return true; }
    if (k === "ArrowRight" || k === "ArrowLeft") {
      const step = (e.shiftKey ? 1 : 5) * (k === "ArrowRight" ? 1 : -1);
      this.seek(this.t + step); if (!this.playing) this._report(); return true;
    }
    if (k === "+" || k === "=" || k === "-" || k === "_") {
      const i = AGP_SPEEDS.indexOf(this.speed), up = k === "+" || k === "=";
      this.setSpeed(AGP_SPEEDS[agpClamp((i < 0 ? 2 : i) + (up ? 1 : -1), 0, AGP_SPEEDS.length - 1)]); return true;
    }
    if (k === "Home") { this.seek(0); return true; }
    if (k === "Escape" || k === "0") { this.follow(null); return true; }
    if (/^[1-9]$/.test(k) && this.order && this.order[+k - 1]) { this.follow(this.order[+k - 1].pc.id); return true; }
    return false;
  }
  _click(x, y) {
    this.wrap.focus({ preventScroll: true });
    if (this.overlayOpen) { this.overlayClosed = true; return; }
    for (const r of this.towerRows) {
      if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) { this.follow(this.followId === r.id ? null : r.id); return; }
    }
    let best = null, bestD = Infinity;
    for (const s of this.states || []) {
      const d = Math.hypot(this.cam.X(s.st.x) - x, this.cam.Y(s.st.y) - y);
      if (d < bestD) { bestD = d; best = s; }
    }
    if (best && bestD < Math.max(22, this.carL)) { this.follow(best.pc.id); return; }
    if (!this.playing && this.t === 0) { this.toggle(); return; }
    if (this.followId) this.follow(null);
  }

  /* ---------------------------------------------------------------- frame loop */
  _frame(now) {
    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 0; this.last = now;
    if (this.playing && this.data) {
      this.t += dt * this.speed;
      if (this.t >= this.duration) { this.t = this.duration; this.playing = false; this._syncButtons(); this._report(); }
    }
    try { this._draw(dt, now); } catch (err) { console.error("[agp replay]", err); }
    this.raf = requestAnimationFrame(this._frame);
  }

  _standings(t) {
    const rows = this.cars.filter((pc) => pc.n).map((pc) => {
      const st = agpStateAt(pc, t);
      const finished = pc.finishT !== null && t >= pc.finishT - 1e-9;
      const retired = !finished && t > pc.end + 1e-9 && pc.end < this.duration;
      const hold = pc.holds.find(([a, b]) => t >= a && t < b) || null;
      return { pc, st, finished, retired, hold };
    });
    const grp = (r) => (r.finished ? 0 : r.retired ? 2 : 1);
    rows.sort((a, b) => grp(a) - grp(b) || (grp(a) === 0 ? a.pc.finishT - b.pc.finishT : b.st.p - a.st.p));
    return rows;
  }

  _gap(r, k, leader, t) {
    if (r.retired) return { text: "DNF", ink: AGP.muted };
    if (this.order.length === 1) return { text: r.finished ? agpFmtLap(r.pc.finishT) : agpFmtTime(Math.min(t, r.pc.end)), ink: AGP.ink };
    if (k === 0) return r.finished ? { text: agpFmtLap(r.pc.finishT), ink: AGP.ink } : { text: "Leader", ink: AGP.ink2 };
    if (r.finished && leader.finished) return { text: `+${(r.pc.finishT - leader.pc.finishT).toFixed(1)}`, ink: AGP.ink };
    const down = Math.floor(leader.st.p - r.st.p + 1e-9);
    if (down >= 1) return { text: `+${down} LAP${down > 1 ? "S" : ""}`, ink: AGP.ink2 };
    const tl = agpTimeAtProgress(leader.pc, r.st.p);
    return tl === null ? { text: "--", ink: AGP.muted } : { text: `+${Math.max(0, t - tl).toFixed(1)}`, ink: AGP.ink };
  }

  _draw(dt, now) {
    const { w: W, h: H, dpr } = agpFitCanvas(this.canvas);
    if (W < 40 || H < 40) return;                         // not visible (e.g. an inactive tab): skip
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = AGP.grass; ctx.fillRect(0, 0, W, H);
    if (!this.track) {
      ctx.fillStyle = AGP.muted; ctx.font = agpFont(16); ctx.textAlign = "center";
      ctx.fillText("No replay loaded", W / 2, H / 2); return;
    }
    const fs = agpClamp(W / 62, 12, 20);
    this.fs = fs;
    const t = this.t;
    const rows = this.order = this._standings(t);
    this.states = rows;
    const docked = W >= 760;
    const TW = docked ? agpClamp(W * 0.25, 230, 330) : agpClamp(fs * 9, 120, 170);
    const vp = docked ? { x: TW + 24, y: 14, w: W - TW - 38, h: H - 28 } : { x: 10, y: 10, w: W - 20, h: H - 20 };
    // camera: whole track (leaving room for corner labels), or glide after the followed car
    const m = fs * 1.8;
    const fit = AgpCamera.fit(this.track.bounds, { x: vp.x + m, y: vp.y + m, w: vp.w - 2 * m, h: vp.h - 2 * m });
    const fol = this.followId ? rows.find((r) => r.pc.id === this.followId) : null;
    const target = fol ? { cx: fol.st.x, cy: fol.st.y, s: Math.max(fit.s * this.zoom, fit.s), ox: fit.ox, oy: fit.oy } : fit;
    this.cam.ease(target, dt);
    const cam = this.cam;
    const L = this.carL = agpCarSize(cam, fs);

    // ---- world layer (clipped to the right of the docked timing tower)
    ctx.save();
    if (docked) { ctx.beginPath(); ctx.rect(TW + 16, 0, W - TW - 16, H); ctx.clip(); }
    ctx.save(); cam.apply(ctx, dpr);
    this.track.drawWorld(ctx, cam, { vmax: this.opts.vmax });
    ctx.restore();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.track.drawLabels(ctx, cam, fs);
    this._drawCrashMarks(ctx, cam, t, fs);
    if (this.opts.trails) for (const r of rows) this._drawTrail(ctx, cam, r, t);
    const blinkOn = Math.floor(now / 160) % 2 === 0;
    const drawOrder = rows.slice().reverse().sort((a, b) => (a.pc.id === this.followId) - (b.pc.id === this.followId));
    for (const r of drawOrder) {
      let alpha = 1;
      if (r.retired) alpha = 0.35;
      else if (r.finished && t > r.pc.finishT + 2) alpha = 0.55;
      if (r.hold) alpha = blinkOn ? 1 : 0.25;
      agpDrawCar(ctx, cam.X(r.st.x), cam.Y(r.st.y), r.st.h, r.pc.color, L, alpha, r.pc.id === this.followId);
    }
    const mode = this.opts.labels === "auto" ? (this.cars.length <= 6 || this.followId ? "names" : "codes") : this.opts.labels;
    if (mode !== "none") {
      const tags = drawOrder.map((r) => ({ r, x: cam.X(r.st.x), y: cam.Y(r.st.y) - L * 0.55 - 4,
        txt: mode === "codes" && r.pc.id !== this.followId ? r.pc.code : r.pc.name }));
      agpPlaceTags(ctx, tags, fs);
      for (const g of tags) agpCarTag(ctx, g.x, g.y, g.txt, g.r.pc.color, fs, g.r.retired ? 0.45 : 0.95);
    }
    for (const r of rows) if (r.hold) this._drawPenalty(ctx, cam, r, t, fs, L);
    ctx.restore();

    // ---- overlays
    const towerH = this._drawTower(ctx, rows, t, 10, 10, TW, H - 20, fs, docked);
    const free = H - 20 - towerH - 10;                    // room left under the tower for the radio
    if (docked && free >= fs * 7) this._drawTicker(ctx, t, { x: 10, y: 20 + towerH, w: TW, h: free }, fs, true);
    else this._drawTicker(ctx, t, { x: vp.x + 4, y: vp.y, w: Math.min(vp.w - 8, fs * 32), h: vp.h - 4 }, fs, false);
    if (fol) this._drawHud(ctx, fol, rows.indexOf(fol) + 1, vp, fs);
    this._drawStartHint(ctx, W, H, fs);
    this._drawFinish(ctx, W, H, fs, t);
    this.clock.textContent = `${agpFmtTime(t)} / ${agpFmtTime(this.duration)}`;
    if (!this.dragging) this.scrub.value = String(Math.round(1000 * t / (this.duration || 1)));
  }

  _drawCrashMarks(ctx, cam, t, fs) {
    const win = 1.4 * Math.max(1, this.speed / 2);         // flash length in race seconds
    for (const pc of this.cars) for (const e of pc.car.events || []) {
      if (e.kind !== "crash" || e.t > t) continue;
      const x = cam.X(e.x), y = cam.Y(e.y), r = fs * 0.32;
      ctx.strokeStyle = agpHexAlpha(AGP.bad, 0.75); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); ctx.stroke();
      const age = t - e.t;
      if (age < win) {                                      // expanding ring = the impact
        const k = age / win;
        ctx.strokeStyle = agpHexAlpha(AGP.bad, 1 - k); ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(x, y, fs * (0.6 + 2.6 * k), 0, AGP_TAU); ctx.stroke();
        ctx.fillStyle = agpHexAlpha(AGP.bad, 0.35 * (1 - k));
        ctx.beginPath(); ctx.arc(x, y, fs * (0.6 + 1.2 * k), 0, AGP_TAU); ctx.fill();
      }
    }
  }

  _drawTrail(ctx, cam, r, t) {
    const pc = r.pc, j = Math.min(r.st.i, pc.n - 1);
    const i0 = Math.max(0, agpBsearch(pc.T, t - AGP_TRAIL_S) - 1);
    if (j - i0 < 1) return;
    ctx.lineWidth = Math.max(2, this.fs * 0.2); ctx.lineCap = "round";
    let px = cam.X(pc.X[i0]), py = cam.Y(pc.Y[i0]);
    for (let i = i0 + 1; i <= j + 1; i++) {
      const head = i === j + 1;
      const wx = head ? r.st.x : pc.X[i], wy = head ? r.st.y : pc.Y[i];
      const x = cam.X(wx), y = cam.Y(wy);
      const age = t - (head ? t : pc.T[i]);
      const jump = Math.hypot(wx - (head ? pc.X[j] : pc.X[i - 1]), wy - (head ? pc.Y[j] : pc.Y[i - 1]));
      if (jump < 30) {                                      // never draw across a marshal reset
        ctx.strokeStyle = agpHexAlpha(pc.color, 0.55 * (1 - age / AGP_TRAIL_S));
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke();
      }
      px = x; py = y;
    }
  }

  _drawPenalty(ctx, cam, r, t, fs, L) {
    const [a, b] = r.hold, k = (t - a) / Math.max(1e-6, b - a);
    ctx.font = agpFont(fs * 0.85, 800);
    agpPill(ctx, cam.X(r.st.x) + L * 0.9, cam.Y(r.st.y) + L * 0.4 - k * fs, `+${Math.round(b - a)}s`, "#fff", AGP.bad, fs * 0.85);
  }

  _drawTower(ctx, rows, t, x, y, w, maxH, fs, docked) {
    this.towerRows = [];
    const rh = docked ? fs * 1.85 : fs * 1.55, head = docked ? fs * 3.3 : fs * 2.0;
    const n = Math.max(1, Math.min(rows.length, Math.floor((maxH - head - 8) / rh)));
    const h = head + n * rh + 8;
    agpPanel(ctx, x, y, w, h, 10);
    const leader = rows[0];
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    if (docked) {
      ctx.font = agpFont(fs * 0.72, 600); ctx.fillStyle = AGP.muted;
      ctx.fillText(agpEllipsis(ctx, (this.data.title || "Replay").toUpperCase(), w - 24), x + 12, y + fs * 0.95);
    }
    const ly = y + (docked ? fs * 2.3 : fs * 1.0);
    ctx.font = agpFont(fs * (docked ? 1.15 : 0.9), 800); ctx.fillStyle = AGP.ink;
    let lapTxt;
    if (leader && leader.finished) lapTxt = "FINISH";
    else lapTxt = `LAP ${Math.min(this.laps, Math.max(1, leader ? leader.st.lap : 1))}/${this.laps}`;
    ctx.fillText(lapTxt, x + 12, ly);
    if (leader && leader.finished) agpChequer(ctx, x + 14 + ctx.measureText(lapTxt).width + 8, ly - fs * 0.45, fs * 1.3, fs * 0.9, 4);
    ctx.textAlign = "right"; ctx.font = agpFont(fs * (docked ? 1.0 : 0.8), 600); ctx.fillStyle = AGP.ink2;
    ctx.fillText(agpFmtTime(t), x + w - 12, ly);
    ctx.fillStyle = AGP.panelLine; ctx.fillRect(x + 8, y + head - 2, w - 16, 1);
    const blinkOn = Math.floor(performance.now() / 300) % 2 === 0;
    for (let k = 0; k < n; k++) {
      const r = rows[k], ry = y + head + k * rh + 2, fol = r.pc.id === this.followId;
      if (fol) { agpRoundRect(ctx, x + 4, ry, w - 8, rh - 2, 6); ctx.fillStyle = agpHexAlpha(AGP.accent, 0.3); ctx.fill(); }
      const cy = ry + rh / 2;
      ctx.textAlign = "center"; ctx.font = agpFont(fs * 0.9, 800); ctx.fillStyle = r.retired ? AGP.muted : AGP.ink;
      ctx.fillText(String(k + 1), x + 12 + fs * 0.6, cy);
      ctx.fillStyle = r.pc.color; ctx.fillRect(x + 12 + fs * 1.4, cy - rh * 0.3, 4, rh * 0.6);
      const gap = this._gap(r, k, leader, t);
      ctx.font = agpFont(fs * 0.85, 600); ctx.textAlign = "right";
      const gw = ctx.measureText(gap.text).width;
      ctx.fillStyle = gap.ink; ctx.fillText(gap.text, x + w - 12, cy);
      let nameRight = x + w - 20 - gw;
      if (r.hold && blinkOn) {
        ctx.font = agpFont(fs * 0.7, 800);
        const pw = ctx.measureText("PEN").width + 8;
        agpRoundRect(ctx, nameRight - pw, cy - fs * 0.45, pw, fs * 0.9, 3); ctx.fillStyle = AGP.bad; ctx.fill();
        ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText("PEN", nameRight - pw / 2, cy + 0.5);
        nameRight -= pw + 6;
      }
      if (r.finished) { agpChequer(ctx, nameRight - fs * 0.9, cy - fs * 0.32, fs * 0.9, fs * 0.64, 3); nameRight -= fs * 0.9 + 6; }
      ctx.textAlign = "left"; ctx.font = agpFont(fs * 0.92, fol ? 800 : 600);
      ctx.fillStyle = r.retired ? AGP.muted : AGP.ink;
      const nx = x + 12 + fs * 1.4 + 10;
      ctx.fillText(agpEllipsis(ctx, docked ? r.pc.name : r.pc.code, nameRight - nx), nx, cy);
      this.towerRows.push({ x: x + 4, y: ry, w: w - 8, h: rh - 2, id: r.pc.id });
    }
    return h;
  }

  /* Radio + race events. column=true: a "RADIO" panel filling `box` from the top, two lines per
     entry (under the timing tower). column=false: a compact overlay at the bottom of `box`. */
  _drawTicker(ctx, t, box, fs, column) {
    if (!this.items || !this.items.length) return;
    const lh = column ? fs * 2.55 : fs * 1.55, head = column ? fs * 1.7 : 0;
    const maxN = Math.max(1, Math.min(column ? 6 : 4, Math.floor((box.h - head - 12) / lh)));
    const end = agpBsearch(this.itemT, t), shown = [];
    for (let i = end - 1; i >= 0 && shown.length < maxN; i--) {
      if (t - this.items[i].t > 40) break;
      shown.unshift(this.items[i]);
    }
    if (!shown.length && !column) return;
    const h = column ? Math.min(box.h, head + maxN * lh + 12) : shown.length * lh + 12;
    const x = box.x, y = column ? box.y : box.y + box.h - h, w = box.w;
    agpPanel(ctx, x, y, w, h, column ? 10 : 8);
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    if (column) {
      ctx.font = agpFont(fs * 0.72, 700); ctx.fillStyle = AGP.muted;
      ctx.fillText("RADIO & RACE CONTROL", x + 12, y + fs * 0.95);
      if (!shown.length) { ctx.font = agpFont(fs * 0.8, 400); ctx.fillText("(quiet so far)", x + 12, y + head + fs); }
    }
    shown.forEach((it, k) => {
      const top = y + head + 6 + k * lh;
      ctx.globalAlpha = agpClamp(1 - (t - it.t) / 40, 0.45, 1);
      const ink = it.kind === "crash" || it.kind === "dnf" || it.kind === "driver_error" ? "#f08080"
        : it.kind === "zone" || it.kind === "stall" ? AGP.warn : it.kind === "radio" ? AGP.ink2 : AGP.ink;
      ctx.fillStyle = it.pc.color; ctx.fillRect(x + 10, top + lh * 0.16, 4, lh * 0.68);
      ctx.font = agpFont(fs * 0.85, 800); ctx.fillStyle = AGP.ink;
      if (column) {
        ctx.fillText(agpEllipsis(ctx, it.pc.name, w - fs * 4.5), x + 22, top + lh * 0.3);
        ctx.font = agpFont(fs * 0.7, 600); ctx.fillStyle = AGP.muted; ctx.textAlign = "right";
        ctx.fillText(agpFmtTime(it.t), x + w - 12, top + lh * 0.3); ctx.textAlign = "left";
        ctx.font = agpFont(fs * 0.82, it.kind === "radio" ? 400 : 700); ctx.fillStyle = ink;
        ctx.fillText(agpEllipsis(ctx, it.text, w - 34), x + 22, top + lh * 0.72);
      } else {
        const name = agpEllipsis(ctx, it.pc.name, fs * 7), cy = top + lh / 2;
        ctx.fillText(name, x + 22, cy);
        const nx = x + 22 + ctx.measureText(name).width + fs * 0.6;
        ctx.font = agpFont(fs * 0.85, it.kind === "radio" ? 400 : 700); ctx.fillStyle = ink;
        ctx.fillText(agpEllipsis(ctx, it.text, x + w - 12 - nx), nx, cy);
      }
      ctx.globalAlpha = 1;
    });
  }

  _drawHud(ctx, r, pos, vp, fs) {
    const w = fs * 13, h = fs * 7.6, x = vp.x + vp.w - w - 4, y = vp.y + 4;
    agpPanel(ctx, x, y, w, h, 10);
    ctx.fillStyle = r.pc.color; ctx.fillRect(x, y + 10, 4, h - 20);
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = agpFont(fs * 0.95, 800); ctx.fillStyle = AGP.ink;
    ctx.fillText(agpEllipsis(ctx, `P${pos}  ${r.pc.name}`, w - 28), x + 14, y + fs * 1.0);
    const spd = Math.round(Math.abs(r.st.v)).toString();
    ctx.font = agpFont(fs * 2.1, 800); ctx.fillText(spd, x + 14, y + fs * 2.75);
    const sw = ctx.measureText(spd).width;
    ctx.font = agpFont(fs * 0.72, 600); ctx.fillStyle = AGP.muted;
    ctx.fillText("u/s", x + 20 + sw, y + fs * 3.1);
    ctx.textAlign = "right"; ctx.fillStyle = AGP.ink2; ctx.font = agpFont(fs * 0.78, 600);
    const crashes = r.pc.car.crashes || 0;
    ctx.fillText(`lap ${Math.min(this.laps, r.st.lap)}/${this.laps}`, x + w - 14, y + fs * 2.35);
    ctx.fillStyle = crashes ? "#f08080" : AGP.ink2;
    ctx.fillText(`${crashes} crash${crashes === 1 ? "" : "es"}`, x + w - 14, y + fs * 3.25);
    const bx = x + 14, bw = w - 28, bh = fs * 0.5;
    ctx.textAlign = "left"; ctx.font = agpFont(fs * 0.7, 600); ctx.fillStyle = AGP.muted;
    ctx.fillText("throttle / brake", bx, y + fs * 4.35);
    agpCenterBar(ctx, bx, y + fs * 4.85, bw, bh, r.st.thr, r.st.thr >= 0 ? AGP.good : AGP.bad);
    ctx.fillStyle = AGP.muted; ctx.fillText("steering  (+ = right)", bx, y + fs * 5.95);
    agpCenterBar(ctx, bx, y + fs * 6.45, bw, bh, r.st.str, AGP.accent);
  }

  _drawStartHint(ctx, W, H, fs) {
    if (this.playing || this.t > 0 || !this.data) return;
    const r = fs * 2.2, cx = W / 2 + (W >= 760 ? agpClamp(W * 0.25, 230, 330) / 2 : 0), cy = H / 2;
    ctx.beginPath(); ctx.arc(cx, cy, r, 0, AGP_TAU); ctx.fillStyle = "rgba(57,135,229,0.85)"; ctx.fill();
    ctx.beginPath(); ctx.moveTo(cx - r * 0.3, cy - r * 0.45); ctx.lineTo(cx + r * 0.5, cy); ctx.lineTo(cx - r * 0.3, cy + r * 0.45);
    ctx.closePath(); ctx.fillStyle = "#fff"; ctx.fill();
    ctx.font = agpFont(fs * 0.85, 600); ctx.textAlign = "center"; ctx.fillStyle = AGP.ink;
    agpPill(ctx, cx, cy + r + fs * 1.1, "Play (Space) · click a car to follow it", AGP.ink, "rgba(13,13,13,0.75)", fs * 0.85);
  }

  _drawFinish(ctx, W, H, fs, t) {
    this.overlayOpen = false;
    if (!this.data || this.duration <= 0 || t < this.duration - 1e-6 || this.overlayClosed || this.dragging) return;
    this.overlayOpen = true;
    ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(0, 0, W, H);
    const cars = this.cars.slice().sort((a, b) => (a.car.position || 99) - (b.car.position || 99));
    const rh = fs * 1.75, n = Math.min(cars.length, Math.floor((H - fs * 9) / rh));
    const w = Math.min(W - 30, fs * 36), h = fs * 7.6 + n * rh, x = (W - w) / 2, y = Math.max(10, (H - h) / 2);
    agpPanel(ctx, x, y, w, h, 12, "rgba(24,24,23,0.98)");
    ctx.save(); agpRoundRect(ctx, x, y, w, h, 12); ctx.clip();
    agpChequer(ctx, x, y, w, fs * 0.9, Math.round(w / (fs * 0.9))); ctx.restore();
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = agpFont(fs * 1.5, 800); ctx.fillStyle = AGP.ink;
    ctx.fillText(this.data.kind === "session" ? "SESSION COMPLETE" : "CHEQUERED FLAG", W / 2, y + fs * 2.1);
    ctx.font = agpFont(fs * 0.8, 600); ctx.fillStyle = AGP.muted;
    ctx.fillText(agpEllipsis(ctx, this.data.title || "", w - 30), W / 2, y + fs * 3.4);
    // columns: position, colour + name, then right-aligned time / best lap / crashes
    const rCrash = x + w - fs * 1.2, rBest = rCrash - fs * 5.0, rTime = rBest - fs * 5.6;
    const cols = [x + fs * 1.6, x + fs * 3.2, rTime - fs * 5.0];
    const hy = y + fs * 4.7;
    ctx.font = agpFont(fs * 0.7, 700); ctx.fillStyle = AGP.muted; ctx.textAlign = "right";
    ctx.fillText("TIME", rTime, hy); ctx.fillText("BEST LAP", rBest, hy); ctx.fillText("CRASHES", rCrash, hy);
    for (let k = 0; k < n; k++) {
      const c = cars[k].car, cy = y + fs * 5.6 + k * rh + rh / 2;
      if (k === 0 && c.finished) { agpRoundRect(ctx, x + 8, cy - rh / 2 + 1, w - 16, rh - 2, 6); ctx.fillStyle = "rgba(250,178,25,0.16)"; ctx.fill(); }
      ctx.textAlign = "center"; ctx.font = agpFont(fs, 800); ctx.fillStyle = AGP.ink; ctx.fillText(String(c.position || k + 1), cols[0], cy);
      ctx.fillStyle = c.color; ctx.fillRect(cols[1] - fs * 0.6, cy - rh * 0.3, 4, rh * 0.6);
      ctx.textAlign = "left"; ctx.font = agpFont(fs * 0.95, 700); ctx.fillStyle = AGP.ink;
      ctx.fillText(agpEllipsis(ctx, c.name, cols[2] - cols[1] - fs * 0.5), cols[1], cy);
      ctx.textAlign = "right"; ctx.font = agpFont(fs * 0.95, 600);
      ctx.fillStyle = c.finished ? AGP.ink : "#f08080";
      ctx.fillText(c.finished ? agpFmtLap(c.race_time) : agpEllipsis(ctx, c.status || "DNF", fs * 7), rTime, cy);
      ctx.fillStyle = AGP.ink2; ctx.fillText(c.best_lap != null ? agpFmtLap(c.best_lap) : "--", rBest, cy);
      ctx.fillText(String(c.crashes || 0), rCrash, cy);
    }
    ctx.textAlign = "center"; ctx.font = agpFont(fs * 0.7, 600); ctx.fillStyle = AGP.muted;
    ctx.fillText("click to close  ·  ⏮ or Home to watch again", W / 2, y + h - fs * 0.9);
  }

  /* little marks above the scrub bar: crashes (red) and finishes (white) */
  _drawMarks() {
    const c = this.marks; if (!c || !c.getContext) return;
    const { w, h, dpr } = agpFitCanvas(c);
    const ctx = c.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    if (!this.duration) return;
    for (const pc of this.cars) for (const e of pc.car.events || []) {
      if (e.kind !== "crash" && e.kind !== "finish") continue;
      ctx.fillStyle = e.kind === "crash" ? AGP.bad : AGP.ink;
      ctx.fillRect(8 + (w - 16) * e.t / this.duration - 1, 0, 2, h);
    }
  }
}
