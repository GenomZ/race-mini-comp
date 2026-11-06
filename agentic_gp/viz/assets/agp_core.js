/* Agentic GP canvas core - shared drawing code (track, cars, camera, panels).
 *
 * Used by agp_replay.js (replay player), agp_live.js (live view) and, through them, by
 * the standalone HTML player (agentic_gp/viz/player.py) and the Streamlit components
 * (app/components/race_canvas.py). Plain script: no imports, no exports, no globals other
 * than the `agp*` / `Agp*` names below (the HTML player wraps everything in a function).
 *
 * Coordinates: the sim's world is y-up (x right, y up, heading in degrees counter-clockwise
 * from +x). The canvas is y-down, so the camera flips y. Distances are "track units".
 */

const AGP = {
  page: "#0d0d0d", grass: "#111311", runoff: "#1b1d1b", asphalt: "#34373c", wall: "#e9e7e0",
  center: "rgba(255,255,255,0.13)", gate: "rgba(255,255,255,0.07)", kerbRed: "#d03b3b", kerbWhite: "#f0efec",
  zone: "#fab219", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", panel: "rgba(22,22,21,0.88)",
  panelLine: "rgba(255,255,255,0.10)", accent: "#3987e5", good: "#0ca30c", warn: "#fab219", bad: "#d03b3b",
  font: 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
};
const AGP_TAU = Math.PI * 2;

/* ------------------------------------------------------------------ small helpers */
const agpLerp = (a, b, w) => a + (b - a) * w;
const agpClamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function agpLerpDeg(a, b, w) {                          // shortest way round, in degrees
  const d = ((((b - a + 180) % 360) + 360) % 360) - 180;
  return a + d * w;
}
function agpFmtTime(sec) {                               // 83.4 -> "1:23.4"
  if (sec === null || sec === undefined || !isFinite(sec)) return "--";
  const neg = sec < 0; sec = Math.abs(sec);
  const m = Math.floor(sec / 60), s = sec - 60 * m;
  const txt = m > 0 ? `${m}:${s.toFixed(1).padStart(4, "0")}` : s.toFixed(1);
  return (neg ? "-" : "") + txt;
}
function agpFmtLap(sec) {                                // 52.317 -> "52.32", 83.4 -> "1:23.40"
  if (sec === null || sec === undefined || !isFinite(sec)) return "--";
  const m = Math.floor(sec / 60), s = sec - 60 * m;
  return m > 0 ? `${m}:${s.toFixed(2).padStart(5, "0")}` : s.toFixed(2);
}
function agpCode(name) {                                 // "Planner" -> "PLA" (F1 style label)
  const letters = String(name || "?").replace(/[^A-Za-z0-9]/g, "");
  return (letters || "CAR").slice(0, 3).toUpperCase();
}
function agpFont(px, weight = 400) { return `${weight} ${Math.round(px)}px ${AGP.font}`; }
function agpHexAlpha(hex, a) {                           // "#rrggbb" + alpha -> rgba()
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex || ""));
  if (!m) return `rgba(141,153,174,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}
function agpBsearch(arr, t) {                            // first index with arr[i] > t
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] <= t) lo = mid + 1; else hi = mid; }
  return lo;
}
function agpRayColor(dist, maxDist) {                    // near = red, mid = amber, far = green
  if (dist < 10) return AGP.bad;
  if (dist < 25) return AGP.warn;
  return dist >= (maxDist || 200) - 0.5 ? "rgba(12,163,12,0.55)" : AGP.good;
}

/* Presenter mode ("fit"): a canvas height that fills the browser window below `el`.
 * top = el's distance from the top of the page content (independent of the current scroll), so the
 * canvas ends at the window's bottom edge when the page is scrolled to the top. A canvas that starts
 * too low for that gets one full window (minus `reserve`): scroll down once and it fills the screen. */
function agpWindowHeight(el, reserve, min) {
  const vh = window.innerHeight || 720;
  let top = el.getBoundingClientRect().top;
  for (let p = el.parentElement; p && p !== document.documentElement; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if (oy === "auto" || oy === "scroll") { top += p.scrollTop - p.getBoundingClientRect().top; break; }
  }
  top += window.scrollY || 0;
  const below = vh - top - reserve;
  return Math.round(Math.max(min, below >= min + 80 ? below : vh - reserve - 24));
}

/* Keep the canvas backing store in sync with its CSS size (crisp on HiDPI screens). */
function agpFitCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
  const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
  return { w, h, dpr };
}

/* ------------------------------------------------------------------ camera */
class AgpCamera {
  constructor() { this.cx = 0; this.cy = 0; this.s = 1; this.ox = 0; this.oy = 0; this.ready = false; }
  /* Target view that fits `bounds` [x0, y0, x1, y1] into the screen rectangle `r` (CSS px). */
  static fit(bounds, r) {
    const [x0, y0, x1, y1] = bounds;
    const s = Math.max(1e-6, Math.min(Math.max(1, r.w) / Math.max(1e-6, x1 - x0), Math.max(1, r.h) / Math.max(1e-6, y1 - y0)));
    return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s, ox: r.x + r.w / 2, oy: r.y + r.h / 2 };
  }
  /* Glide toward a target view (exponential smoothing; `rate` per second). */
  ease(target, dt, rate = 7) {
    const bad = (v) => !isFinite(v);
    if (bad(target.s) || target.s <= 0) return;           // hidden / zero-size canvas: keep the old view
    if (!this.ready || dt <= 0 || dt > 0.5 || bad(this.s) || this.s <= 0 || bad(this.cx) || bad(this.cy)) {
      Object.assign(this, target); this.ready = true; return;
    }
    const k = 1 - Math.exp(-rate * dt);
    this.cx = agpLerp(this.cx, target.cx, k); this.cy = agpLerp(this.cy, target.cy, k);
    this.s = Math.exp(agpLerp(Math.log(this.s), Math.log(target.s), k));
    this.ox = agpLerp(this.ox, target.ox, k); this.oy = agpLerp(this.oy, target.oy, k);
  }
  X(x) { return this.ox + (x - this.cx) * this.s; }
  Y(y) { return this.oy - (y - this.cy) * this.s; }
  toWorld(sx, sy) { return [this.cx + (sx - this.ox) / this.s, this.cy - (sy - this.oy) / this.s]; }
  /* Draw in world units from here on (y flipped). */
  apply(ctx, dpr) {
    const s = this.s * dpr;
    ctx.setTransform(s, 0, 0, -s, dpr * (this.ox - this.cx * this.s), dpr * (this.oy + this.cy * this.s));
  }
}

/* ------------------------------------------------------------------ the track */
/* `tr` is a replay track dict (center_x, left_x, ..., gate_x1, ..., zones) plus an optional
   `view` block from Python: {id, bounds, corners: [{id, apex_index, start_index, end_index,
   direction, v_max}], vmax: [per centerline point]}. Paths are built once and reused. */
class AgpTrack {
  constructor(tr) {
    this.tr = tr;
    const v = tr.view || {};
    this.id = v.id || tr.key || tr.name;
    this.n = tr.center_x.length;
    this.halfWidth = +tr.half_width || 9;
    this.corners = v.corners || [];
    this.vmax = v.vmax || null;
    this.maxSpeed = v.max_speed || 40;
    const L = (i) => [tr.left_x[i], tr.left_y[i]], R = (i) => [tr.right_x[i], tr.right_y[i]];
    const C = (i) => [tr.center_x[i], tr.center_y[i]];
    this.L = L; this.R = R; this.C = C;
    this.bounds = v.bounds || this._bounds();
    this.asphalt = new Path2D();
    this._loop(this.asphalt, L); this._loop(this.asphalt, R);
    this.leftWall = new Path2D(); this._loop(this.leftWall, L);
    this.rightWall = new Path2D(); this._loop(this.rightWall, R);
    this.center = new Path2D(); this._loop(this.center, C);
    this.gates = [];
    for (let g = 0; g < (tr.gate_x1 || []).length; g++)
      this.gates.push([tr.gate_x1[g], tr.gate_y1[g], tr.gate_x2[g], tr.gate_y2[g]]);
    this.kerbs = new Path2D();
    for (const c of this.corners) {                      // kerbs on both edges of every corner
      for (const P of [L, R]) this._run(this.kerbs, P, c.start_index - 1, c.end_index + 1);
    }
    this.zones = (tr.zones || []).map((z) => {
      const p = new Path2D(); this._run(p, C, z.from_index, z.to_index);
      const span = (z.to_index - z.from_index + this.n) % this.n;
      return { path: p, mid: (z.from_index + Math.floor(span / 2)) % this.n, z };
    });
  }
  _bounds() {
    const tr = this.tr, xs = tr.left_x.concat(tr.right_x), ys = tr.left_y.concat(tr.right_y);
    const pad = 6;
    return [Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad];
  }
  _loop(path, P) {
    const [x0, y0] = P(0); path.moveTo(x0, y0);
    for (let i = 1; i < this.n; i++) { const [x, y] = P(i); path.lineTo(x, y); }
    path.closePath();
  }
  _run(path, P, from, to) {                              // open polyline from index `from` to `to` (wraps)
    const n = this.n, span = ((to - from) % n + n) % n;
    for (let k = 0; k <= span; k++) {
      const [x, y] = P(((from + k) % n + n) % n);
      if (k === 0) path.moveTo(x, y); else path.lineTo(x, y);
    }
  }
  /* Everything that lives in world space (call after camera.apply). */
  drawWorld(ctx, cam, opt = {}) {
    const px = 1 / cam.s;                                // one CSS pixel in world units
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.strokeStyle = AGP.runoff; ctx.lineWidth = Math.max(5, 14 * px);
    ctx.stroke(this.leftWall); ctx.stroke(this.rightWall);
    ctx.fillStyle = AGP.asphalt; ctx.fill(this.asphalt, "evenodd");
    for (const z of this.zones) {                        // speed-limit zones (yellow flags)
      ctx.strokeStyle = agpHexAlpha(AGP.zone, 0.26); ctx.lineWidth = this.halfWidth * 1.9; ctx.lineCap = "butt";
      ctx.stroke(z.path);
      ctx.strokeStyle = agpHexAlpha(AGP.zone, 0.9); ctx.lineWidth = 2 * px; ctx.setLineDash([3, 3]);
      ctx.stroke(z.path); ctx.setLineDash([]); ctx.lineCap = "round";
    }
    if (opt.vmax && this.vmax) this._drawVmax(ctx, px);
    else {
      ctx.strokeStyle = AGP.center; ctx.lineWidth = 1.2 * px; ctx.setLineDash([6 * px, 7 * px]);
      ctx.stroke(this.center); ctx.setLineDash([]);
    }
    ctx.strokeStyle = AGP.gate; ctx.lineWidth = 1 * px;
    for (const [x1, y1, x2, y2] of this.gates) { ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
    const kw = Math.max(1.4, 4 * px);                    // kerbs: red with white dashes
    ctx.lineCap = "butt";
    ctx.strokeStyle = AGP.kerbRed; ctx.lineWidth = kw; ctx.stroke(this.kerbs);
    ctx.strokeStyle = AGP.kerbWhite; ctx.setLineDash([2.2, 2.2]); ctx.stroke(this.kerbs); ctx.setLineDash([]);
    ctx.lineCap = "round";
    ctx.strokeStyle = AGP.wall; ctx.lineWidth = Math.max(0.35, 1.6 * px);
    ctx.stroke(this.leftWall); ctx.stroke(this.rightWall);
    this._drawStartLine(ctx);
  }
  _drawVmax(ctx, px) {                                   // centerline coloured by corner speed
    ctx.lineWidth = Math.max(1.2, 3 * px); ctx.lineCap = "butt";
    for (let i = 0; i < this.n; i++) {
      const j = (i + 1) % this.n, [x0, y0] = this.C(i), [x1, y1] = this.C(j);
      ctx.strokeStyle = agpVmaxColor(this.vmax[i], this.maxSpeed);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    }
    ctx.lineCap = "round";
  }
  _drawStartLine(ctx) {
    if (!this.gates.length) return;
    const [x1, y1, x2, y2] = this.gates[0];
    const len = Math.hypot(x2 - x1, y2 - y1); if (len < 1e-6) return;
    const ux = (x2 - x1) / len, uy = (y2 - y1) / len, vx = -uy, vy = ux;  // along the line / across
    const n = Math.max(4, Math.round(len / 1.6)), sq = len / n;
    for (let row = 0; row < 2; row++) for (let k = 0; k < n; k++) {
      ctx.fillStyle = (k + row) % 2 ? "#111" : "#f4f4f2";
      const bx = x1 + ux * k * sq + vx * (row - 1) * sq, by = y1 + uy * k * sq + vy * (row - 1) * sq;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + ux * sq, by + uy * sq);
      ctx.lineTo(bx + ux * sq + vx * sq, by + uy * sq + vy * sq); ctx.lineTo(bx + vx * sq, by + vy * sq);
      ctx.closePath(); ctx.fill();
    }
  }
  /* Labels drawn in screen space (so text is never mirrored): corner ids, zone labels, gates. */
  drawLabels(ctx, cam, fs, opt = {}) {
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    if (opt.corners !== false) {
      ctx.font = agpFont(fs * 0.78, 700);
      for (const c of this.corners) {
        const a = c.apex_index, [cx, cy] = this.C(a);
        const [wx, wy] = c.direction === "right" ? this.L(a) : this.R(a);   // outside of the turn
        const dx = wx - cx, dy = wy - cy, d = Math.hypot(dx, dy) || 1;
        const off = (d + 6) + 16 / cam.s;
        const sx = cam.X(cx + dx / d * off), sy = cam.Y(cy + dy / d * off);
        agpPill(ctx, sx, sy, c.id, AGP.ink2, "rgba(13,13,13,0.72)", fs * 0.78);
      }
    }
    for (const z of this.zones) {
      const [x, y] = this.C(z.mid);
      const txt = `${(z.z.label || "Zone").toUpperCase()}  ≤${Math.round(z.z.max_speed)}`;
      ctx.font = agpFont(fs * 0.74, 700);
      agpPill(ctx, cam.X(x), cam.Y(y) - this.halfWidth * cam.s - fs, txt, "#141414", agpHexAlpha(AGP.zone, 0.95), fs * 0.74);
    }
    if (opt.gateNumbers) {
      ctx.font = agpFont(fs * 0.62, 600); ctx.fillStyle = AGP.muted;
      this.gates.forEach(([x1, y1, x2, y2], g) => {
        const dx = x2 - x1, dy = y2 - y1, d = Math.hypot(dx, dy) || 1;
        ctx.fillText(String(g), cam.X(x2 + dx / d * 5), cam.Y(y2 + dy / d * 5));
      });
    }
  }
}

function agpVmaxColor(v, vMaxTrack) {                  // one hue: slow corner = bright orange, flat out = dim
  const w = agpClamp(1 - v / (vMaxTrack || 40), 0, 1) ** 0.7;
  const a = [84, 76, 70], b = [255, 150, 60];
  return `rgb(${Math.round(agpLerp(a[0], b[0], w))},${Math.round(agpLerp(a[1], b[1], w))},${Math.round(agpLerp(a[2], b[2], w))})`;
}

/* ------------------------------------------------------------------ cars */
/* A small top-down race car, nose along +x, drawn in SCREEN space at (sx, sy).
   headingDeg is the world heading (counter-clockwise), so the screen angle is -heading. */
function agpDrawCar(ctx, sx, sy, headingDeg, color, L, alpha = 1, highlight = false) {
  const W = L * 0.5;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(sx, sy); ctx.rotate(-headingDeg * Math.PI / 180);
  if (highlight) {
    ctx.beginPath(); ctx.arc(0, 0, L * 0.95, 0, AGP_TAU);
    ctx.fillStyle = agpHexAlpha(AGP.accent, 0.22); ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = AGP.accent; ctx.stroke();
  }
  ctx.fillStyle = "rgba(0,0,0,0.45)";                    // shadow
  ctx.beginPath(); ctx.ellipse(1.5, 2, L * 0.55, W * 0.62, 0, 0, AGP_TAU); ctx.fill();
  const dark = "#121212";
  ctx.fillStyle = dark;                                   // wheels
  for (const [wx, wy] of [[-0.3, -0.47], [-0.3, 0.47], [0.25, -0.44], [0.25, 0.44]]) {
    ctx.fillRect(wx * L - L * 0.09, wy * W - W * 0.13, L * 0.18, W * 0.26);
  }
  ctx.fillStyle = dark; ctx.fillRect(-0.52 * L, -0.48 * W, 0.12 * L, 0.96 * W);   // rear wing
  ctx.fillRect(0.4 * L, -0.5 * W, 0.1 * L, W);                                     // front wing
  ctx.beginPath();                                        // body: sidepods + nose
  ctx.moveTo(-0.44 * L, -0.26 * W); ctx.lineTo(-0.12 * L, -0.36 * W); ctx.lineTo(0.14 * L, -0.3 * W);
  ctx.lineTo(0.47 * L, -0.1 * W); ctx.lineTo(0.47 * L, 0.1 * W); ctx.lineTo(0.14 * L, 0.3 * W);
  ctx.lineTo(-0.12 * L, 0.36 * W); ctx.lineTo(-0.44 * L, 0.26 * W); ctx.closePath();
  ctx.fillStyle = color; ctx.fill();
  ctx.lineWidth = Math.max(1, L / 22); ctx.strokeStyle = "rgba(0,0,0,0.65)"; ctx.stroke();
  ctx.fillStyle = color; ctx.fillRect(-0.53 * L, -0.48 * W, 0.05 * L, 0.96 * W);  // wing colour stripes
  ctx.fillRect(0.45 * L, -0.5 * W, 0.04 * L, W);
  ctx.beginPath(); ctx.arc(-0.04 * L, 0, W * 0.17, 0, AGP_TAU);                  // helmet
  ctx.fillStyle = "#f4f4f2"; ctx.fill(); ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = 1; ctx.stroke();
  ctx.restore();
}

/* Car length on screen: real size when zoomed in, never smaller than a readable minimum. */
function agpCarSize(cam, fs) { return agpClamp(5.2 * cam.s, fs * 1.25, fs * 3.2); }

/* Name tag above a car: colour bar + text. */
function agpCarTag(ctx, sx, sy, text, color, fs, alpha = 1) {
  ctx.save(); ctx.globalAlpha = alpha;
  ctx.font = agpFont(fs * 0.8, 700); ctx.textAlign = "left"; ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width + fs * 0.9, h = fs * 1.35;
  const x = sx - w / 2, y = sy - h;
  agpRoundRect(ctx, x, y, w, h, 4); ctx.fillStyle = "rgba(10,10,10,0.78)"; ctx.fill();
  ctx.fillStyle = color; ctx.fillRect(x, y, 3.5, h);
  ctx.fillStyle = AGP.ink; ctx.fillText(text, x + fs * 0.5, y + h / 2 + 0.5);
  ctx.restore();
}

/* Move name tags up until they no longer overlap (tags: [{x, y, txt}], y = bottom of the tag). */
function agpPlaceTags(ctx, tags, fs) {
  ctx.font = agpFont(fs * 0.8, 700);
  const h = fs * 1.35 + 2, placed = [];
  tags.sort((a, b) => b.y - a.y);
  for (const g of tags) {
    const w = ctx.measureText(g.txt).width + fs * 0.9;
    for (let guard = 0; guard < 12; guard++) {
      const hit = placed.find((p) => Math.abs(p.x - g.x) < (p.w + w) / 2 && Math.abs(p.y - g.y) < h);
      if (!hit) break;
      g.y = hit.y - h;
    }
    placed.push({ x: g.x, y: g.y, w });
  }
}

/* ------------------------------------------------------------------ panels */
function agpRoundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
function agpPanel(ctx, x, y, w, h, r = 8, fill = AGP.panel) {
  agpRoundRect(ctx, x, y, w, h, r);
  ctx.fillStyle = fill; ctx.fill();
  ctx.strokeStyle = AGP.panelLine; ctx.lineWidth = 1; ctx.stroke();
}
function agpPill(ctx, cx, cy, text, ink, bg, fs) {      // centred label with a rounded background
  const w = ctx.measureText(text).width + fs * 0.8, h = fs * 1.45;
  agpRoundRect(ctx, cx - w / 2, cy - h / 2, w, h, h / 2); ctx.fillStyle = bg; ctx.fill();
  ctx.fillStyle = ink; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(text, cx, cy + 0.5);
}
function agpEllipsis(ctx, text, maxW) {                 // shorten text to fit maxW pixels
  text = String(text);
  if (ctx.measureText(text).width <= maxW) return text;
  while (text.length > 1 && ctx.measureText(text + "…").width > maxW) text = text.slice(0, -1);
  return text + "…";
}
function agpWrap(ctx, text, maxW, maxLines = 2) {       // word-wrap into at most maxLines lines
  const words = String(text).split(/\s+/), lines = [];
  let cur = "";
  for (const w of words) {
    const t = cur ? cur + " " + w : w;
    if (ctx.measureText(t).width <= maxW || !cur) cur = t;
    else { lines.push(cur); cur = w; if (lines.length === maxLines) break; }
  }
  if (lines.length < maxLines && cur) lines.push(cur);
  else if (cur && lines.length === maxLines) lines[maxLines - 1] = agpEllipsis(ctx, lines[maxLines - 1] + " " + cur, maxW);
  return lines.map((l) => agpEllipsis(ctx, l, maxW));
}
function agpChequer(ctx, x, y, w, h, n = 4) {           // a little chequered flag
  const cw = w / n, ch = h / Math.max(1, Math.round(n * h / w));
  for (let i = 0; i < n; i++) for (let j = 0; j * ch < h - 0.01; j++) {
    ctx.fillStyle = (i + j) % 2 ? "#111" : "#f4f4f2"; ctx.fillRect(x + i * cw, y + j * ch, cw, ch);
  }
}
/* Horizontal bar gauge: value in [-1, 1] drawn from the centre (steering, "+ = right"). */
function agpCenterBar(ctx, x, y, w, h, value, color) {
  agpRoundRect(ctx, x, y, w, h, h / 2); ctx.fillStyle = "rgba(255,255,255,0.08)"; ctx.fill();
  const mid = x + w / 2, v = agpClamp(value || 0, -1, 1);
  ctx.fillStyle = color; ctx.fillRect(Math.min(mid, mid + v * w / 2), y, Math.abs(v * w / 2), h);
  ctx.fillStyle = AGP.ink; ctx.fillRect(mid - 1, y - 2, 2, h + 4);
}

/* ------------------------------------------------------------------ replay cars */
/* Turn a replay car (schema v1) into typed arrays for fast interpolation. A crash frame shows
   the car where the marshals put it back; we add a "contact" frame at the wall (event x, y)
   just before it, so the car visibly reaches the wall before it is reset. */
function agpPrepareCar(c, index) {
  const n = (c.t || []).length;
  const crashes = (c.events || []).filter((e) => e.kind === "crash").sort((a, b) => a.t - b.t);
  const rows = [];
  let ci = 0;
  for (let i = 0; i < n; i++) {
    while (ci < crashes.length && crashes[ci].t < c.t[i] - 1e-6) ci++;
    const e = crashes[ci];
    if (e && i > 0 && Math.abs(e.t - c.t[i]) < 1e-6 && i + 1 < n && c.speed[i + 1] === 0) {
      rows.push([c.t[i] - 0.02, e.x, e.y, c.heading_deg[i - 1], c.speed[i - 1], c.race_progress[i - 1],
                 c.lap[i - 1], c.throttle[i], c.steering[i]]);
      ci++;
    }
    rows.push([c.t[i], c.x[i], c.y[i], c.heading_deg[i], c.speed[i], c.race_progress[i], c.lap[i],
               c.throttle[i], c.steering[i]]);
  }
  const col = (k) => Float64Array.from(rows, (r) => r[k]);
  const pc = { car: c, index, id: c.id, name: c.name, color: c.color || "#8d99ae", T: col(0), X: col(1), Y: col(2),
               H: col(3), V: col(4), P: col(5), LAP: col(6), THR: col(7), STR: col(8), n: rows.length };
  pc.PM = new Float64Array(pc.n);                         // running max of race progress (for gaps)
  let m = 0; for (let i = 0; i < pc.n; i++) { m = Math.max(m, pc.P[i]); pc.PM[i] = m; }
  pc.holds = crashes.map((e) => [e.t, e.t + (e.value || 0)]);
  pc.end = pc.n ? pc.T[pc.n - 1] : 0;
  pc.finishT = c.finished && c.race_time != null ? +c.race_time : null;
  pc.code = agpCode(c.name);
  return pc;
}

/* Interpolated state of a prepared car at race time t. */
function agpStateAt(pc, t) {
  const n = pc.n;
  if (!n) return null;
  let j = agpBsearch(pc.T, t), i = j - 1, w;
  if (i < 0) { i = 0; j = 0; w = 0; }
  else if (j >= n) { i = n - 1; j = n - 1; w = 0; }
  else { const span = pc.T[j] - pc.T[i]; w = span > 0 ? (t - pc.T[i]) / span : 1; }
  return {
    x: agpLerp(pc.X[i], pc.X[j], w), y: agpLerp(pc.Y[i], pc.Y[j], w), h: agpLerpDeg(pc.H[i], pc.H[j], w),
    v: agpLerp(pc.V[i], pc.V[j], w), p: agpLerp(pc.P[i], pc.P[j], w), lap: w < 1 ? pc.LAP[i] : pc.LAP[j],
    thr: pc.THR[j], str: pc.STR[j], i,
  };
}

/* Race time at which a car first reached race progress p (for interval gaps), or null. */
function agpTimeAtProgress(pc, p) {
  const PM = pc.PM;
  if (!pc.n || p > PM[pc.n - 1]) return null;
  let lo = 0, hi = pc.n - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (PM[mid] >= p) hi = mid; else lo = mid + 1; }
  if (lo === 0) return pc.T[0];
  const p0 = PM[lo - 1], p1 = PM[lo];
  const w = p1 > p0 ? (p - p0) / (p1 - p0) : 1;
  return agpLerp(pc.T[lo - 1], pc.T[lo], w);
}
