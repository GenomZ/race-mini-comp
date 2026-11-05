"""Race canvases for Streamlit: the replay player, the live view and the keyboard drive pad.

    from app.components import replay_canvas, live_canvas, live_state, drive_pad, pad_controls

    replay_canvas("races/demo/class_demo.json", key="replay")          # whole replay, plays in the browser
    live_canvas(live_state(track, obs=obs, action=action), key="live")   # one tick; call again every tick
    pressed = drive_pad(key="pad")                                       # {"up": bool, "down": ..., ...}

How it works (Streamlit Custom Components v2, no iframe): Python sends DATA, the browser DRAWS.
Each component is registered once with HTML + CSS + JavaScript; the JavaScript lives in
agentic_gp/viz/assets/ and is the same code as the standalone HTML player. Because every call
passes a `key`, Streamlit updates the existing canvas in place on a rerun (no flicker, no
remount): the replay keeps playing, and the live car glides from its last pose to the new one.

Rules of thumb
- ALWAYS pass a unique `key` (without it, every data change remounts the canvas).
- Send plain Python numbers/lists (numpy arrays are converted for you by the helpers here).
- The replay canvas never reruns your script while it plays. It only reports back (one rerun)
  when you ask for it with `report=True`: on pause, scrub release and follow changes.
See app/components/README.md for every parameter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import streamlit as st

from agentic_gp.viz.player import asset, player_data, track_payload

# ---------------------------------------------------------------------------------- JavaScript
# CCv2 wants an ES module with a default export. We glue: shared core + one widget + a mount
# function. The mount function runs again on every data change, so it must be idempotent: the
# widget instance is kept in a WeakMap keyed by the component's parent element.
_MOUNT_REPLAY = """
const agpReplays = new WeakMap();
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let p = agpReplays.get(parentElement);
  if (!p) {
    p = new AgpReplayPlayer(parentElement.querySelector(".agp-root"), (data && data.options) || {});
    agpReplays.set(parentElement, p);
  }
  p.onReport = (s) => { setStateValue("t", s.t); setStateValue("follow", s.follow); };
  if (data) p.setData(data.replay, data.options);
  return () => { p.destroy(); agpReplays.delete(parentElement); };
}
"""

_MOUNT_LIVE = """
const agpLives = new WeakMap();
export default function (component) {
  const { data, parentElement } = component;
  let v = agpLives.get(parentElement);
  if (!v) {
    v = new AgpLiveView(parentElement.querySelector(".agp-root"), (data && data.options) || {});
    agpLives.set(parentElement, v);
  }
  if (data) v.setData(data.state, data.options);
  return () => { v.destroy(); agpLives.delete(parentElement); };
}
"""

_MOUNT_PAD = """
const agpPads = new WeakMap();
export default function (component) {
  const { data, parentElement, setStateValue } = component;
  let p = agpPads.get(parentElement);
  if (!p) {
    p = new AgpDrivePad(parentElement.querySelector(".agp-root"), (pressed) => p.sendFn("pressed", pressed));
    agpPads.set(parentElement, p);
  }
  p.sendFn = setStateValue;                     // always the newest callback
  if (data) p.setLabels(data.title || undefined, data.subtitle || undefined);
  return () => agpPads.delete(parentElement);
}
"""

_HTML = '<div class="agp-root"></div>'
_CSS = asset("agp.css")

_replay_component = st.components.v2.component(
    "agp_replay_canvas", html=_HTML, css=_CSS,
    js=asset("agp_core.js") + "\n" + asset("agp_replay.js") + "\n" + _MOUNT_REPLAY)
_live_component = st.components.v2.component(
    "agp_live_canvas", html=_HTML, css=_CSS,
    js=asset("agp_core.js") + "\n" + asset("agp_live.js") + "\n" + _MOUNT_LIVE)
_pad_component = st.components.v2.component(
    "agp_drive_pad", html=_HTML, css=_CSS, js=asset("agp_pad.js") + "\n" + _MOUNT_PAD)


def _noop() -> None:
    """CCv2 needs a callback for every state value it reports; we only read the values."""


# ---------------------------------------------------------------------------------- replay
@st.cache_data(show_spinner=False, max_entries=32)
def _payload_from_file(path: str, mtime: float) -> dict:
    return player_data(path)


def replay_payload(replay: Any) -> dict:
    """The data the replay canvas needs (cached for files). Accepts a path, Replay, dict or result."""
    if isinstance(replay, (str, Path)) and not str(replay).lstrip().startswith("{"):
        p = Path(replay)
        return _payload_from_file(str(p.resolve()), p.stat().st_mtime)
    if isinstance(replay, dict) and "view" in replay and "view" in replay.get("track", {}):
        return replay                                    # already a payload
    return player_data(replay)


def replay_canvas(replay: Any, *, key: str, height: int = 560, autoplay: bool = False, speed: float = 1.0,
                  follow: str | None = None, report: bool = False, labels: str = "auto", vmax: bool = False,
                  trails: bool = True, fit: bool = False) -> Any:
    """Multi-car replay player (whole replay sent once; plays at 60 fps in the browser).

    replay   path to a replay JSON, Replay object, replay dict, SessionResult or GrandPrixResult
             (None: an empty canvas that says "No replay loaded")
    key      REQUIRED unique key (keeps the player alive across reruns)
    height   canvas height in px (the control bar adds ~50 px)
    autoplay start playing as soon as a NEW replay is loaded
    speed    initial playback speed (0.25 .. 16)
    follow   car id to follow initially (None = whole track)
    report   True: on pause / scrub release / follow change the browser sends `t` and `follow`
             back (one rerun each). False (default): the player never reruns your script.
    labels   "auto" | "names" | "codes" | "none" (car name tags)
    vmax     colour the centerline by corner speed (slow = bright orange)
    trails   draw a 4 s trail behind every car
    fit      size the canvas to the browser window (from its top edge to the bottom of the window,
             at least 320 px) - presenter mode on a projector; `height` is then ignored

    Returns the component result: `res.t` (race time in s, or None) and `res.follow` (car id or
    None) - only filled when report=True. Also in st.session_state[key].
    """
    options = {"height": int(height), "autoplay": bool(autoplay), "speed": float(speed), "report": bool(report),
               "labels": labels, "vmax": bool(vmax), "trails": bool(trails), "fit": bool(fit)}
    if follow is not None:
        options["follow"] = follow
    payload = None if replay is None else replay_payload(replay)        # None: an empty "No replay loaded" canvas
    return _replay_component(key=key, data={"replay": payload, "options": options},
                             default={"t": None, "follow": None}, on_t_change=_noop, on_follow_change=_noop)


# ---------------------------------------------------------------------------------- live view
@st.cache_data(show_spinner=False, max_entries=32)
def cached_track_payload(spec: str) -> dict:
    """track_payload() for a registry key / "random:<seed>" / "file:..." spec, cached."""
    return track_payload(spec)


def _num(v: Any, nd: int = 3) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return round(f, nd) if math.isfinite(f) else 0.0


def _point(p: Any) -> list[float] | None:
    if p is None:
        return None
    try:
        return [_num(p[0], 2), _num(p[1], 2)]
    except (TypeError, IndexError, KeyError):
        return None


def rays_from_obs(obs: dict) -> list[dict]:
    """The 7 wall rays of an observation as [{"angle": deg (+ = right), "dist": units}]."""
    out = []
    for k, v in obs.items():
        if not k.startswith("distance_to_wall_"):
            continue
        name = k[len("distance_to_wall_"):]
        if name == "front":
            angle = 0.0
        else:
            side, _, deg = name.partition("_")
            try:
                angle = float(deg) * (1 if side == "right" else -1)
            except ValueError:
                continue
        out.append({"angle": angle, "dist": _num(v, 1)})
    return sorted(out, key=lambda r: r["angle"])


def waypoints_from_obs(obs: dict) -> list[list[float]]:
    """World positions of the next and after-next waypoints, rebuilt from distance + signed angle."""
    x, y = obs["position"]
    h = math.radians(float(obs["heading_deg"]))
    pts = []
    for d_key, a_key in (("distance_to_next_waypoint", "angle_to_next_waypoint"),
                         ("distance_to_waypoint_after_next", "angle_to_waypoint_after_next")):
        if d_key in obs and a_key in obs:
            a = h - math.radians(float(obs[a_key]))          # "+ = right" -> world angle
            d = float(obs[d_key])
            pts.append([round(x + d * math.cos(a), 2), round(y + d * math.sin(a), 2)])
    return pts


def live_state(track: Any, *, obs: dict | None = None, car: dict | None = None, action: Any = None,
               debug: dict | None = None, name: str = "car", color: str = "#3987e5", seq: Any = None,
               status: str = "driving", say: list | None = None, trail: list | None = None,
               crash_points: list | None = None, ghosts: list | None = None, attempt: int | None = None,
               attempts: int | None = None, phase: str | None = None, think_s: float | None = None,
               tween_ms: int | None = None, follow: bool = False, show_rays: bool = True,
               show_waypoints: bool = False, vmax: bool = False, gate_numbers: bool = False,
               message: str | None = None, ray_max: float = 200.0) -> dict:
    """Build one live-canvas snapshot (plain JSON) from an observation and/or a car pose.

    track    track spec / Track / track map / replay track, or a payload from cached_track_payload()
    obs      the observation the driver saw (gives pose, rays, lap, tick, crashes, ...)
    car      pose override {x, y, heading_deg, speed} (e.g. a replay frame); default: from obs
    action   Action / (throttle, steering[, say]) / dict - drawn as the gauges
    debug    the driver's debug dict; debug["target"] = [x, y] is drawn as the target point
    seq      changes -> the canvas starts a new tween (default: obs tick)
    status   "thinking" | "driving" | "planning" | "reflecting" | "paused" | "finished" | "stopped" | "error"
    say      radio lines [{"t": s, "text": "..."}] (last 3 shown)
    ghosts   other cars to draw faded [{"x", "y", "heading_deg", "color", "name"}]
    follow   start with the chase camera (the user can click to toggle)
    """
    payload = track if isinstance(track, dict) and "view" in track else (
        cached_track_payload(track) if isinstance(track, str) else track_payload(track))
    obs = obs or {}
    if car is None and obs:
        px, py = obs.get("position", (0.0, 0.0))
        car = {"x": px, "y": py, "heading_deg": obs.get("heading_deg", 0.0), "speed": obs.get("speed", 0.0)}
    thr, steer = _controls(action)
    if car is not None:
        thr = car.get("throttle", thr) if thr is None else thr
        steer = car.get("steering", steer) if steer is None else steer
    state: dict = {
        "track": payload, "name": str(name), "color": str(color), "status": status,
        "seq": seq if seq is not None else (f"{attempt}:{obs.get('tick')}" if obs else None),
        "tick": int(obs.get("tick", 0)) if obs else None, "t": _num(obs.get("race_time", 0.0), 2) if obs else None,
        "lap": int(obs.get("lap", 1)) if obs else None, "laps": int(obs.get("laps_total", 1)) if obs else None,
        "crashes": int(obs.get("crashes", 0)) if obs else 0, "attempt": attempt, "attempts": attempts,
        "phase": phase or obs.get("phase"), "think_s": None if think_s is None else _num(think_s, 4),
        "say": [{"t": _num(m.get("t"), 2) if isinstance(m, dict) else None,
                 "text": str(m.get("text") if isinstance(m, dict) else m)} for m in (say or [])][-5:],
        "trail": [p for p in (_point(q) for q in (trail or [])) if p][-240:],
        "crash_points": [p for p in (_point(q) for q in (crash_points or [])) if p][-200:],
        "ghosts": [{"x": _num(g["x"], 2), "y": _num(g["y"], 2), "heading_deg": _num(g.get("heading_deg", 0), 1),
                    "color": g.get("color", "#8d99ae"), "name": g.get("name", "")} for g in (ghosts or [])],
        "follow": bool(follow), "show_rays": bool(show_rays), "vmax": bool(vmax), "gate_numbers": bool(gate_numbers),
        "ray_max": float(ray_max), "message": message,
    }
    if car is not None:
        state["car"] = {"x": _num(car.get("x"), 2), "y": _num(car.get("y"), 2),
                        "heading_deg": _num(car.get("heading_deg"), 2), "speed": _num(car.get("speed"), 2),
                        "throttle": _num(thr or 0.0, 3), "steering": _num(steer or 0.0, 3)}
    if obs and any(k.startswith("distance_to_wall_") for k in obs):
        state["rays"] = rays_from_obs(obs)
    if obs and show_waypoints and "position" in obs:
        state["waypoints"] = waypoints_from_obs(obs)
    target = (debug or {}).get("target") if isinstance(debug, dict) else None
    state["target"] = _point(target)
    if tween_ms is not None:
        state["tween_ms"] = int(tween_ms)
    return state


def _controls(action: Any) -> tuple[float | None, float | None]:
    if action is None:
        return None, None
    if isinstance(action, dict):
        return action.get("throttle"), action.get("steering")
    if hasattr(action, "throttle"):
        return action.throttle, action.steering
    try:
        return float(action[0]), float(action[1])
    except (TypeError, IndexError, ValueError):
        return None, None


def live_canvas(state: dict | None, *, key: str, height: int = 480, fit: bool = False) -> None:
    """Live view of ONE car: pass a fresh live_state(...) every tick (e.g. from a fragment).

    Draws the track, the car (gliding from its previous pose), rays coloured by distance, the
    debug target point, throttle/steering gauges, a status pill with the "thinking" clock, the
    radio ticker, the trail and crash marks. Click the canvas for the chase camera.
    fit=True sizes the canvas to the browser window (presenter mode); `height` is then ignored.
    """
    data = {"state": state or {"message": "Waiting for the first tick..."},
            "options": {"height": int(height), "fit": bool(fit)}}
    _live_component(key=key, data=data)


# ---------------------------------------------------------------------------------- drive pad
def drive_pad(*, key: str, title: str | None = None, subtitle: str | None = None) -> dict:
    """Keyboard pad: click it, then hold arrow keys / WASD. Returns {"up","down","left","right"} bools.

    Every press and every release reruns the script ONCE (inside a fragment: only the fragment).
    Typical use: a @st.fragment(run_every=0.1) that reads the pad and steps the sim.
    """
    res = _pad_component(key=key, data={"title": title, "subtitle": subtitle},
                         default={"pressed": {}}, on_pressed_change=_noop)
    pressed = (res["pressed"] if res is not None and "pressed" in res else None) or {}
    return {k: bool(pressed.get(k)) for k in ("up", "down", "left", "right")}


def pad_controls(pressed: dict, steer: float = 0.6, brake: float = 1.0, coast_throttle: float = 0.0) -> tuple[float, float]:
    """Arrow keys -> (throttle, steering). up = full throttle, down = brake, left/right = -/+ steer."""
    throttle = 1.0 if pressed.get("up") else (-brake if pressed.get("down") else coast_throttle)
    steering = (steer if pressed.get("right") else 0.0) - (steer if pressed.get("left") else 0.0)
    return throttle, steering
