/* Agentic GP live view (needs agp_core.js first).
 *
 *   const view = new AgpLiveView(rootElement, {height: 480});
 *   view.setData(state);    // called again for every new tick; the car glides to the new pose
 *
 * `state` is ONE tick snapshot built in Python (app/components/race_canvas.py: live_state()):
 *   {track, car: {x, y, heading_deg, speed, throttle, steering}, name, color, seq, tick, t, lap, laps,
 *    attempt, attempts, phase, status, crashes, rays: [{angle, dist}], ray_max, target: [x, y] | null,
 *    waypoints: [[x, y], [x, y]] | null, say: [{t, text}], trail: [[x, y]...], crash_points: [[x, y]...],
 *    ghosts: [{x, y, heading_deg, color, name}], tween_ms, follow, think_s, message}
 * Rays use the driver frame: angle + = right of the heading, like steering.
 * Click the track to toggle the chase camera; mouse wheel zooms while chasing.
 */

class AgpLiveView {
  constructor(root, opts = {}) {
    this.root = root;
    this.opts = Object.assign({ height: 480 }, opts);
    this.data = null; this.track = null; this.seq = null;
    this.from = null; this.to = null; this.t0 = 0; this.dur = 250; this.lastArrival = 0; this.gap = 250;
    this.follow = null; this.zoom = 3; this.cam = new AgpCamera(); this.last = 0;
    this.root.innerHTML = "";
    this.wrap = document.createElement("div"); this.wrap.className = "agp agp-live"; this.wrap.tabIndex = 0;
    this.stage = document.createElement("div"); this.stage.className = "agp-stage";
    this.canvas = document.createElement("canvas"); this.stage.append(this.canvas);
    this.wrap.append(this.stage); this.root.append(this.wrap);
    this.ctx = this.canvas.getContext("2d");
    this.canvas.onclick = () => { this.follow = !this._following(); this.wrap.focus({ preventScroll: true }); };
    this.icam = null; this.inset = null;
    this.canvas.onwheel = (e) => {
      if (!this._following()) return;
      e.preventDefault(); this.zoom = agpClamp(this.zoom * Math.exp(-e.deltaY * 0.0015), 0.8, 10);
    };
    this._frame = this._frame.bind(this);
    this.raf = requestAnimationFrame(this._frame);
    this._size();
  }
  _size() {
    const h = this.opts.fit ? agpWindowHeight(this.stage, 16, 320) : Math.max(200, +this.opts.height || 480);
    if (this.stage.style.height !== `${h}px`) this.stage.style.height = `${h}px`;
    if (this.opts.fit && !this._onWin) { this._onWin = () => this._size(); window.addEventListener("resize", this._onWin); }
  }
  _following() { return this.follow === null ? !!(this.data && this.data.follow) : this.follow; }
  destroy() { cancelAnimationFrame(this.raf); if (this._onWin) window.removeEventListener("resize", this._onWin); }

  setData(d, options) {
    if (options) { Object.assign(this.opts, options); this._size(); }
    if (!d) return;
    if (d.track && (!this.track || this.track.id !== ((d.track.view || {}).id || d.track.key)))
      this.track = new AgpTrack(d.track);
    const now = performance.now();
    if (d.car && (d.seq !== this.seq || !this.to)) {      // a new tick (or a first pose): tween from where we are
      const cur = this._pose(now);
      if (this.lastArrival) this.gap = agpClamp(now - this.lastArrival, 60, 1200);
      this.lastArrival = now;
      const sameAttempt = this.data && this.data.attempt === d.attempt && d.tick >= (this.data.tick || 0);
      this.from = cur && sameAttempt ? cur : null;
      this.to = { x: d.car.x, y: d.car.y, h: d.car.heading_deg, target: d.target || null };
      this.t0 = now;
      this.dur = d.tween_ms !== undefined && d.tween_ms !== null ? +d.tween_ms : this.gap;
      this.seq = d.seq;
    }
    if (!d.car) { this.from = this.to = null; }
    this.data = d;
  }

  _pose(now) {
    if (!this.to) return null;
    if (!this.from || this.dur <= 0) return Object.assign({}, this.to);
    const w = agpClamp((now - this.t0) / this.dur, 0, 1), e = w * w * (3 - 2 * w);   // smoothstep
    const a = this.from, b = this.to;
    const tg = a.target && b.target ? [agpLerp(a.target[0], b.target[0], e), agpLerp(a.target[1], b.target[1], e)] : b.target;
    return { x: agpLerp(a.x, b.x, e), y: agpLerp(a.y, b.y, e), h: agpLerpDeg(a.h, b.h, e), target: tg };
  }

  _frame(now) {
    const dt = this.last ? Math.min(0.1, (now - this.last) / 1000) : 0; this.last = now;
    try { this._draw(dt, now); } catch (err) { console.error("[agp live]", err); }
    this.raf = requestAnimationFrame(this._frame);
  }

  _draw(dt, now) {
    const { w: W, h: H, dpr } = agpFitCanvas(this.canvas);
    if (W < 40 || H < 40) return;                         // not visible (e.g. an inactive tab): skip
    const ctx = this.ctx, d = this.data || {};
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = AGP.grass; ctx.fillRect(0, 0, W, H);
    const fs = agpClamp(W / 60, 12, 19);
    if (!this.track) { this._center(ctx, W, H, fs, d.message || "Waiting for the first tick…"); return; }
    const docked = W >= 700;
    const HW = docked ? agpClamp(fs * 12.5, 190, 250) : 0;
    const vp = { x: 12, y: 12, w: W - HW - (docked ? 36 : 24), h: H - 24 };
    const pose = this._pose(now);
    const fit = AgpCamera.fit(this.track.bounds, vp);
    const chase = this._following() && pose;
    this.cam.ease(chase ? { cx: pose.x, cy: pose.y, s: Math.max(fit.s, fit.s * this.zoom), ox: fit.ox, oy: fit.oy } : fit, dt);
    const cam = this.cam;
    const clip = docked ? { x: 0, y: 0, w: W - HW - 22, h: H } : { x: 0, y: 0, w: W, h: H };
    this._scene(ctx, cam, clip, d, pose, fs, dpr, true);
    // picture-in-picture: a close-up of the car (rays, target) while the main view shows the track
    this.inset = null;
    if (!chase && pose && d.inset !== false && vp.w > 420) {
      const iw = agpClamp(vp.w * 0.34, 200, 330), ih = iw * 0.68, ir = { x: vp.x + 2, y: vp.y + 2, w: iw, h: ih };
      this.inset = ir;
      this.icam = this.icam || new AgpCamera();
      this.icam.ease({ cx: pose.x, cy: pose.y, s: Math.max(fit.s * 3.5, 2.2), ox: ir.x + iw / 2, oy: ir.y + ih / 2 }, dt, 12);
      ctx.save(); agpRoundRect(ctx, ir.x, ir.y, iw, ih, 10); ctx.fillStyle = AGP.grass; ctx.fill(); ctx.restore();
      this._scene(ctx, this.icam, ir, d, pose, fs * 0.85, dpr, false);
      ctx.save(); agpRoundRect(ctx, ir.x, ir.y, iw, ih, 10); ctx.strokeStyle = AGP.panelLine; ctx.lineWidth = 1.5; ctx.stroke();
      ctx.font = agpFont(fs * 0.68, 700); ctx.fillStyle = AGP.muted; ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText("CLOSE-UP · click the track to chase", ir.x + 10, ir.y + ih - fs * 1.1); ctx.restore();
    }
    if (!docked) this._drawSay(ctx, d, vp, fs);            // docked: the radio lives in the HUD panel
    if (docked) this._drawHud(ctx, d, W - HW - 12, 12, HW, H - 24, fs, now);
    else this._drawMiniHud(ctx, d, W, fs, now);
    if (d.message && !d.car) this._center(ctx, W - HW, H, fs, d.message);
    else if (d.message && (d.status === "paused" || d.status === "loading" || d.status === "planning"))
      this._banner(ctx, W - HW, fs, d.message);          // with a car: a pill at the top, the car stays visible
  }

  /* Track + trail + crashes + ghosts + the car with rays/target, clipped to rectangle r. */
  _scene(ctx, cam, r, d, pose, fs, dpr, labels) {
    ctx.save();
    agpRoundRect(ctx, r.x, r.y, r.w, r.h, labels ? 0 : 10); ctx.clip();
    ctx.save(); cam.apply(ctx, dpr); this.track.drawWorld(ctx, cam, { vmax: !!d.vmax }); ctx.restore();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (labels) this.track.drawLabels(ctx, cam, fs, { gateNumbers: !!d.gate_numbers });
    const L = agpCarSize(cam, fs);
    this._drawTrail(ctx, cam, d, pose, fs);
    for (const [x, y] of d.crash_points || []) {
      const sx = cam.X(x), sy = cam.Y(y), q = fs * 0.32;
      ctx.strokeStyle = agpHexAlpha(AGP.bad, 0.85); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx - q, sy - q); ctx.lineTo(sx + q, sy + q); ctx.moveTo(sx + q, sy - q); ctx.lineTo(sx - q, sy + q); ctx.stroke();
    }
    for (const g of d.ghosts || []) {
      agpDrawCar(ctx, cam.X(g.x), cam.Y(g.y), g.heading_deg, g.color || "#8d99ae", L, 0.4);
      if (g.name && labels) agpCarTag(ctx, cam.X(g.x), cam.Y(g.y) - L * 0.55 - 4, g.name, g.color || "#8d99ae", fs, 0.5);
    }
    if (pose) {
      if (d.waypoints) this._drawWaypoints(ctx, cam, pose, d.waypoints, fs);
      if (d.rays && d.show_rays !== false) this._drawRays(ctx, cam, pose, d, fs, !labels || cam.s > 1.6 || d.ray_labels);
      if (pose.target) this._drawTarget(ctx, cam, pose, fs);
      agpDrawCar(ctx, cam.X(pose.x), cam.Y(pose.y), pose.h, d.color || AGP.accent, L, 1, false);
      if (labels) agpCarTag(ctx, cam.X(pose.x), cam.Y(pose.y) - L * 0.55 - 6, d.name || "car", d.color || AGP.accent, fs);
    }
    ctx.restore();
  }

  _banner(ctx, W, fs, text) {
    ctx.font = agpFont(fs, 700);
    agpPill(ctx, W / 2, fs * 2.2, text, AGP.ink, "rgba(13,13,13,0.85)", fs);
  }

  _center(ctx, W, H, fs, text) {
    ctx.font = agpFont(fs, 600);
    agpPill(ctx, W / 2, H / 2, text, AGP.ink, "rgba(13,13,13,0.8)", fs);
  }

  _drawTrail(ctx, cam, d, pose, fs) {
    const tr = d.trail || [];
    if (tr.length < 2) return;
    ctx.lineWidth = Math.max(2, fs * 0.2); ctx.lineCap = "round"; ctx.lineJoin = "round";
    const n = tr.length;
    for (let i = 1; i < n; i++) {
      const [x0, y0] = tr[i - 1], [x1, y1] = tr[i];
      if (Math.hypot(x1 - x0, y1 - y0) > 30) continue;     // a marshal reset: no line across the infield
      ctx.strokeStyle = agpHexAlpha(d.color || AGP.accent, 0.12 + 0.5 * i / n);
      ctx.beginPath(); ctx.moveTo(cam.X(x0), cam.Y(y0)); ctx.lineTo(cam.X(x1), cam.Y(y1)); ctx.stroke();
    }
  }

  _drawRays(ctx, cam, pose, d, fs, showNumbers) {
    const maxD = d.ray_max || 200;
    for (const r of d.rays) {
      const a = (pose.h - r.angle) * Math.PI / 180;        // driver frame "+ = right" -> world angle
      const len = Math.min(r.dist, maxD);
      const ex = pose.x + Math.cos(a) * len, ey = pose.y + Math.sin(a) * len;
      const col = agpRayColor(r.dist, maxD);
      ctx.strokeStyle = col; ctx.lineWidth = r.dist < 25 ? 2.5 : 1.6; ctx.globalAlpha = 0.95;
      ctx.beginPath(); ctx.moveTo(cam.X(pose.x), cam.Y(pose.y)); ctx.lineTo(cam.X(ex), cam.Y(ey)); ctx.stroke();
      if (r.dist < maxD - 0.5) { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(cam.X(ex), cam.Y(ey), 3.5, 0, AGP_TAU); ctx.fill(); }
      ctx.globalAlpha = 1;
      if (showNumbers) {
        ctx.font = agpFont(fs * 0.7, 700);
        agpPill(ctx, cam.X(ex) + Math.cos(a) * 12, cam.Y(ey) - Math.sin(a) * 12, r.dist >= maxD - 0.5 ? `${Math.round(maxD)}+` : r.dist.toFixed(1),
                "#fff", agpHexAlpha("#141414", 0.8), fs * 0.7);
      }
    }
  }

  _drawWaypoints(ctx, cam, pose, wps, fs) {
    wps.forEach((p, k) => {
      if (!p) return;
      const sx = cam.X(p[0]), sy = cam.Y(p[1]);
      ctx.strokeStyle = k === 0 ? "rgba(144,133,233,0.95)" : "rgba(144,133,233,0.5)"; ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]); ctx.beginPath(); ctx.moveTo(cam.X(pose.x), cam.Y(pose.y)); ctx.lineTo(sx, sy); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(sx, sy, fs * 0.45, 0, AGP_TAU); ctx.stroke();
      ctx.font = agpFont(fs * 0.7, 700);
      agpPill(ctx, sx, sy - fs * 1.1, k === 0 ? "next" : "after next", "#fff", "rgba(98,80,214,0.85)", fs * 0.7);
    });
  }

  _drawTarget(ctx, cam, pose, fs) {
    const [tx, ty] = pose.target, sx = cam.X(tx), sy = cam.Y(ty), r = fs * 0.55;
    ctx.strokeStyle = "#e87ba4"; ctx.lineWidth = 2;
    ctx.setLineDash([3, 4]); ctx.beginPath(); ctx.moveTo(cam.X(pose.x), cam.Y(pose.y)); ctx.lineTo(sx, sy); ctx.stroke(); ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(sx, sy, r, 0, AGP_TAU); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(sx - r * 1.6, sy); ctx.lineTo(sx + r * 1.6, sy); ctx.moveTo(sx, sy - r * 1.6); ctx.lineTo(sx, sy + r * 1.6); ctx.stroke();
  }

  _drawSay(ctx, d, vp, fs) {
    const say = (d.say || []).slice(-3);
    if (!say.length) return;
    const lh = fs * 1.55, w = Math.min(vp.w - 8, fs * 30), h = say.length * lh + 12, x = vp.x + 4, y = vp.y + vp.h - h - 4;
    agpPanel(ctx, x, y, w, h, 8);
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    say.forEach((m, k) => {
      const cy = y + 6 + k * lh + lh / 2;
      ctx.globalAlpha = k === say.length - 1 ? 1 : 0.6;
      ctx.font = agpFont(fs * 0.75, 600); ctx.fillStyle = AGP.muted;
      const tt = m.t !== undefined && m.t !== null ? `${(+m.t).toFixed(1)}s` : "";
      ctx.fillText(tt, x + 12, cy);
      ctx.font = agpFont(fs * 0.88, k === say.length - 1 ? 600 : 400); ctx.fillStyle = AGP.ink;
      ctx.fillText(agpEllipsis(ctx, `“${m.text}”`, w - fs * 4.2), x + 12 + fs * 3.2, cy);
      ctx.globalAlpha = 1;
    });
  }

  _statusPill(d, now) {
    const st = String(d.status || "").toLowerCase();
    const age = this.lastArrival ? (now - this.lastArrival) / 1000 : 0;
    const map = {
      thinking: ["THINKING " + age.toFixed(1) + "s", AGP.warn, "#141414"], driving: ["DRIVING", AGP.good, "#fff"],
      planning: ["PLANNING", "#9085e9", "#fff"], reflecting: ["REFLECTING", "#9085e9", "#fff"],
      paused: ["PAUSED", "#5c5b57", "#fff"], finished: ["FINISHED", "#f0efec", "#141414"],
      stopped: ["STOPPED", "#5c5b57", "#fff"], error: ["ERROR", AGP.bad, "#fff"], loading: ["LOADING", "#5c5b57", "#fff"],
      dnf: ["DNF", AGP.bad, "#fff"], waiting: ["READY", "#5c5b57", "#fff"],
    };
    return map[st] || [st ? st.toUpperCase() : "READY", "#5c5b57", "#fff"];
  }

  _drawHud(ctx, d, x, y, w, h, fs, now) {
    agpPanel(ctx, x, y, w, h, 10);
    const c = d.car || {};
    ctx.fillStyle = d.color || AGP.accent; ctx.fillRect(x, y + 10, 4, fs * 1.6);
    ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.font = agpFont(fs, 800); ctx.fillStyle = AGP.ink;
    ctx.fillText(agpEllipsis(ctx, d.name || "car", w - 24), x + 14, y + 10 + fs * 0.8);
    const [label, bg, ink] = this._statusPill(d, now);
    ctx.font = agpFont(fs * 0.78, 800);
    const pw = ctx.measureText(label).width + fs, py = y + fs * 3.0;
    agpRoundRect(ctx, x + 14, py - fs * 0.65, pw, fs * 1.3, fs * 0.65); ctx.fillStyle = bg; ctx.fill();
    ctx.fillStyle = ink; ctx.fillText(label, x + 14 + fs * 0.5, py + 0.5);
    let yy = y + fs * 5.2;
    ctx.font = agpFont(fs * 2.6, 800); ctx.fillStyle = AGP.ink;
    const spd = c.speed !== undefined ? Math.round(Math.abs(c.speed)).toString() : "--";
    ctx.fillText(spd, x + 14, yy);
    const sw = ctx.measureText(spd).width;
    ctx.font = agpFont(fs * 0.75, 600); ctx.fillStyle = AGP.muted; ctx.fillText("speed (u/s)", x + 20 + sw, yy + fs * 0.4);
    yy += fs * 1.9;
    const bx = x + 14, bw = w - 28, bh = fs * 0.55;
    ctx.fillText("throttle / brake", bx, yy); yy += fs * 0.75;
    agpCenterBar(ctx, bx, yy, bw, bh, c.throttle, (c.throttle || 0) >= 0 ? AGP.good : AGP.bad);
    ctx.fillStyle = AGP.ink2; ctx.textAlign = "right"; ctx.fillText((+(c.throttle || 0)).toFixed(2), x + w - 14, yy - fs * 0.75);
    ctx.textAlign = "left"; yy += fs * 1.6; ctx.fillStyle = AGP.muted;
    ctx.fillText("steering  (+ = right)", bx, yy); yy += fs * 0.75;
    agpCenterBar(ctx, bx, yy, bw, bh, c.steering, AGP.accent);
    ctx.fillStyle = AGP.ink2; ctx.textAlign = "right"; ctx.fillText((+(c.steering || 0)).toFixed(2), x + w - 14, yy - fs * 0.75);
    ctx.textAlign = "left"; yy += fs * 2.0;
    const lines = [];
    if (d.attempts) lines.push(["attempt", `${d.attempt || 1} / ${d.attempts}${d.phase && d.phase !== "session" ? "  " + d.phase : ""}`]);
    if (d.laps) lines.push(["lap", `${Math.min(d.lap || 1, d.laps)} / ${d.laps}`]);
    if (d.tick !== undefined && d.tick !== null) lines.push(["tick", `${d.tick}`]);
    if (d.t !== undefined && d.t !== null) lines.push(["race clock", agpFmtTime(d.t)]);
    lines.push(["crashes", `${d.crashes || 0}`]);
    if (d.think_s !== undefined && d.think_s !== null) lines.push(["think", d.think_s < 1 ? `${Math.round(d.think_s * 1000)} ms` : `${(+d.think_s).toFixed(2)} s`]);
    for (const [k, v] of lines) {
      if (yy > y + h - fs) break;
      ctx.font = agpFont(fs * 0.8, 600); ctx.fillStyle = AGP.muted; ctx.textAlign = "left"; ctx.fillText(k, bx, yy);
      ctx.font = agpFont(fs * 0.9, 700); ctx.fillStyle = k === "crashes" && d.crashes ? "#f08080" : AGP.ink; ctx.textAlign = "right";
      ctx.fillText(v, x + w - 14, yy);
      yy += fs * 1.45;
    }
    // radio: the newest lines at the bottom of the panel, as much as fits
    const say = (d.say || []).slice(-4);
    if (say.length && yy + fs * 3 < y + h) {
      yy += fs * 0.4;
      ctx.fillStyle = AGP.panelLine; ctx.fillRect(bx, yy - fs * 0.6, w - 28, 1);
      ctx.textAlign = "left"; ctx.font = agpFont(fs * 0.7, 700); ctx.fillStyle = AGP.muted;
      ctx.fillText("RADIO", bx, yy + fs * 0.1); yy += fs * 1.2;
      const blocks = say.map((m) => { ctx.font = agpFont(fs * 0.8, 500); return { m, lines: agpWrap(ctx, `“${m.text}”`, w - 28, 2) }; });
      while (blocks.length > 1 && yy + blocks.reduce((a, b) => a + b.lines.length * fs * 1.1 + fs * 0.5, 0) > y + h - 8) blocks.shift();
      blocks.forEach((b, k) => {
        const last = k === blocks.length - 1;
        ctx.globalAlpha = last ? 1 : 0.6;
        ctx.font = agpFont(fs * 0.8, last ? 600 : 400); ctx.fillStyle = AGP.ink;
        for (const line of b.lines) { if (yy > y + h - 6) break; ctx.fillText(line, bx, yy); yy += fs * 1.1; }
        yy += fs * 0.4;
        ctx.globalAlpha = 1;
      });
    }
  }

  _drawMiniHud(ctx, d, W, fs, now) {
    const c = d.car || {};
    const [label, bg, ink] = this._statusPill(d, now);
    const w = fs * 9.5, h = fs * 4.6, x = W - w - 10, y = 10;
    agpPanel(ctx, x, y, w, h, 8);
    ctx.textBaseline = "middle"; ctx.textAlign = "left"; ctx.font = agpFont(fs * 0.75, 800);
    const pw = ctx.measureText(label).width + fs * 0.8;
    agpRoundRect(ctx, x + 10, y + 8, pw, fs * 1.2, fs * 0.6); ctx.fillStyle = bg; ctx.fill();
    ctx.fillStyle = ink; ctx.fillText(label, x + 10 + fs * 0.4, y + 8 + fs * 0.6);
    ctx.font = agpFont(fs * 1.4, 800); ctx.fillStyle = AGP.ink; ctx.textAlign = "right";
    ctx.fillText(c.speed !== undefined ? Math.round(Math.abs(c.speed)).toString() : "--", x + w - 10, y + 8 + fs * 0.6);
    agpCenterBar(ctx, x + 10, y + fs * 2.4, w - 20, fs * 0.45, c.throttle, (c.throttle || 0) >= 0 ? AGP.good : AGP.bad);
    agpCenterBar(ctx, x + 10, y + fs * 3.4, w - 20, fs * 0.45, c.steering, AGP.accent);
  }
}
