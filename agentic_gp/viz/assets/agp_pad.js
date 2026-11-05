/* Agentic GP drive pad: arrow keys / WASD (or the on-screen keys) -> {up, down, left, right}.
 *
 *   const pad = new AgpDrivePad(rootElement, (pressed) => send(pressed));
 *
 * `send` is called ONLY when a key changes state (key auto-repeat is ignored), so a Streamlit
 * component built on it causes one rerun per press and one per release - never a flood.
 * Avoid the letters r and c: they are Streamlit's own hotkeys.
 */

const AGP_PAD_KEYS = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right",
                       w: "up", s: "down", a: "left", d: "right", W: "up", S: "down", A: "left", D: "right" };

class AgpDrivePad {
  constructor(root, send, opts = {}) {
    this.send = send;
    this.keys = { up: false, down: false, left: false, right: false };
    root.innerHTML = "";
    const pad = this.el = document.createElement("div");
    pad.className = "agp agp-pad"; pad.tabIndex = 0;
    pad.innerHTML = `
      <div class="agp-keys">
        <div class="agp-key agp-key-up" data-k="up">▲</div>
        <div class="agp-key agp-key-left" data-k="left">◀</div>
        <div class="agp-key agp-key-down agp-brake" data-k="down">▼</div>
        <div class="agp-key agp-key-right" data-k="right">▶</div>
      </div>
      <div class="agp-pad-text">
        <div class="agp-pad-title"></div>
        <div class="agp-pad-sub"></div>
      </div>`;
    root.append(pad);
    this.title = pad.querySelector(".agp-pad-title");
    this.sub = pad.querySelector(".agp-pad-sub");
    this.setLabels(opts.title, opts.subtitle);
    const set = (k, v) => {
      if (!k || this.keys[k] === v) return;
      this.keys[k] = v; this._paint(); this.send(Object.assign({}, this.keys));
    };
    pad.onkeydown = (e) => { const k = AGP_PAD_KEYS[e.key]; if (!k) return; e.preventDefault(); e.stopPropagation(); if (!e.repeat) set(k, true); };
    pad.onkeyup = (e) => { const k = AGP_PAD_KEYS[e.key]; if (!k) return; e.preventDefault(); e.stopPropagation(); set(k, false); };
    pad.onfocus = () => { pad.classList.add("agp-active"); this.setLabels(); };
    pad.onblur = () => {                                   // leaving the pad releases every key
      pad.classList.remove("agp-active"); this.setLabels();
      let changed = false;
      for (const k in this.keys) if (this.keys[k]) { this.keys[k] = false; changed = true; }
      if (changed) { this._paint(); this.send(Object.assign({}, this.keys)); }
    };
    for (const cap of pad.querySelectorAll(".agp-key")) {   // mouse / touch
      cap.onpointerdown = (e) => { e.preventDefault(); pad.focus(); set(cap.dataset.k, true); };
      cap.onpointerup = cap.onpointerleave = () => set(cap.dataset.k, false);
    }
  }
  setLabels(title, subtitle) {
    if (title !== undefined) this._t = title;
    if (subtitle !== undefined) this._s = subtitle;
    const active = this.el.classList.contains("agp-active");
    this.title.textContent = this._t || (active ? "Driving - keys are live" : "Click here to take the wheel");
    this.sub.textContent = this._s || "↑ / W throttle · ↓ / S brake · ← → / A D steer";
  }
  _paint() {
    for (const cap of this.el.querySelectorAll(".agp-key")) cap.classList.toggle("agp-on", !!this.keys[cap.dataset.k]);
  }
}
