"""Standalone HTML replay player: one self-contained .html file per replay (no internet needed).

    from agentic_gp.viz.player import save_replay_html, replay_html, player_data
    save_replay_html("races/demo/class_demo.json", "class_demo.html")   # open it in any browser
    html = replay_html(replay, height=640)       # the same page as a string (e.g. for st.iframe)

    python race.py watch races/demo/class_demo.json --html             # the CLI way

The page embeds the replay JSON and the canvas player from `agentic_gp/viz/assets/`:
agp_core.js (track, cars, camera) + agp_replay.js (the player) + agp.css. The Streamlit app's
replay component (app/components/race_canvas.py) runs exactly the same JavaScript, so what you
see in the app is what you share as a file.

Player features: track with kerbs, gates, start line, corner labels and yellow-flag zones; cars
in their colours with name tags and trails; play/pause, 0.25x-16x, scrub bar, restart; click a
car (or a row of the timing tower) to follow it; timing tower with gaps; radio ticker; crash
flashes and "+5s" penalty holds; final classification. Keys: Space, <- ->, + -, Home, 1-9, Esc.

`player_data(replay)` is the replay dict (schema v1, see agentic_gp/race/replay.py) plus two
`view` blocks the browser cannot compute cheaply: corners + v_max per centerline point
(track["view"]) and an id/duration (data["view"]). `track_payload(track)` gives the same track
dict for the live view.
"""

from __future__ import annotations

import hashlib
import html
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"


@lru_cache(maxsize=None)
def asset(name: str) -> str:
    """Text of one file in agentic_gp/viz/assets/ (agp.css, agp_core.js, agp_replay.js, ...)."""
    return (ASSETS / name).read_text(encoding="utf-8")


# ============================================================================ inputs
def _replay_dict(replay: Any) -> dict:
    """Replay object / replay dict / path to a replay JSON / SessionResult / GrandPrixResult -> dict."""
    if isinstance(replay, (str, Path)):
        text = str(replay)
        if not text.lstrip().startswith("{"):
            text = Path(replay).read_text(encoding="utf-8")
        return json.loads(text)
    if isinstance(replay, dict):
        return replay
    if hasattr(replay, "to_replay"):                       # SessionResult
        replay = replay.to_replay("all" if len(getattr(replay, "attempts", [])) > 1 else "best")
    elif hasattr(replay, "replay") and not hasattr(replay, "to_dict"):
        replay = replay.replay
    elif hasattr(replay, "replay") and hasattr(replay, "qualifying_replay"):   # GrandPrixResult
        replay = replay.replay
    if hasattr(replay, "to_dict"):
        return replay.to_dict(extended=True)
    raise TypeError(f"cannot make a replay from {type(replay).__name__}")


def _replay_track_dict(track: Any) -> dict:
    """Any track-like input -> replay track dict (center_x, left_x, ..., gate_x1, ..., zones)."""
    from ..race.replay import ReplayTrack
    from ..sim.track import Track, make_track

    if isinstance(track, ReplayTrack):
        return track.to_dict(extended=True)
    if isinstance(track, dict) and "center_x" in track:
        return dict(track)
    if isinstance(track, dict) and "track" in track and "cars" in track:     # a whole replay
        return dict(track["track"])
    if isinstance(track, dict) and "centerline" in track:                    # a plan() track map
        key = str(track.get("key", ""))
        try:
            return ReplayTrack.from_track(make_track(key)).to_dict(extended=True)
        except Exception:                                  # noqa: BLE001 - custom track: rebuild from the map
            return _map_to_replay_track(track)
    if isinstance(track, (str, Track)) or track is None:
        return ReplayTrack.from_track(make_track(track)).to_dict(extended=True)
    raise TypeError(f"cannot draw a track from {type(track).__name__}")


def _map_to_replay_track(m: dict) -> dict:
    c, lw, rw = (np.asarray(m[k], float) for k in ("centerline", "left_wall", "right_wall"))
    idx = [int(w["centerline_index"]) for w in m.get("waypoints", [])]
    r2 = lambda a: [round(float(v), 2) for v in a]
    sx, sy = m.get("start_position", c[0])
    return {"key": str(m.get("key", "custom")), "name": str(m.get("name", "track")),
            "half_width": float(m.get("half_width", 9.0)), "length": float(m.get("lap_length", 0.0)),
            "center_x": r2(c[:, 0]), "center_y": r2(c[:, 1]), "left_x": r2(lw[:, 0]), "left_y": r2(lw[:, 1]),
            "right_x": r2(rw[:, 0]), "right_y": r2(rw[:, 1]),
            "gate_x1": r2(lw[idx, 0]), "gate_y1": r2(lw[idx, 1]), "gate_x2": r2(rw[idx, 0]), "gate_y2": r2(rw[idx, 1]),
            "start_x": float(sx), "start_y": float(sy), "start_heading_deg": float(m.get("start_heading_deg", 0.0)),
            "zones": []}


@lru_cache(maxsize=64)
def _registry_shape(key: str, n: int) -> tuple[tuple, tuple] | None:
    """(radii, corners) of a built-in / random track whose centerline has n points, else None."""
    from ..sim.track import TRACKS, make_track

    if not (key in TRACKS or key.startswith("random:")):
        return None
    try:
        tr = make_track(key)
    except Exception:                                      # noqa: BLE001
        return None
    if tr.n_points != n:
        return None
    return tuple(float(r) for r in tr.radii), tuple(tr.corners())


def _track_view(rt: dict, max_speed: float = 40.0, a_lat: float = 20.0) -> dict:
    """Extras for the browser: corners (T1, T2, ...), v_max per centerline point, bounds, id."""
    from .. import driverkit as dk

    pts = np.column_stack([np.asarray(rt["center_x"], float), np.asarray(rt["center_y"], float)])
    shape = _registry_shape(str(rt.get("key", "")), len(pts))
    if shape is not None:
        radii, corners = np.asarray(shape[0]), [dict(c) for c in shape[1]]
    else:
        ds = float(np.mean(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 1.0
        radii = dk.curvature_radii(pts, max(1, int(round(4.0 / max(ds, 1e-6)))))
        corners = dk.find_corners(pts, a_lat, max_speed)
    radii = np.where(np.isfinite(radii), radii, 1e9)
    vmax = np.minimum(max_speed, np.sqrt(a_lat * np.maximum(radii, 0.0)))
    xs = list(rt["left_x"]) + list(rt["right_x"])
    ys = list(rt["left_y"]) + list(rt["right_y"])
    pad = 8.0
    ident = hashlib.sha1(json.dumps([rt.get("key"), rt.get("name"), len(pts), rt["center_x"][:5],
                                     rt["center_y"][:5]]).encode()).hexdigest()[:12]
    keep = ("id", "start_index", "apex_index", "end_index", "direction", "v_max", "min_radius")
    return {
        "id": ident, "max_speed": float(max_speed),
        "bounds": [round(min(xs) - pad, 2), round(min(ys) - pad, 2), round(max(xs) + pad, 2), round(max(ys) + pad, 2)],
        "corners": [{k: (round(float(c[k]), 1) if isinstance(c.get(k), float) else c.get(k)) for k in keep}
                    for c in corners],
        "vmax": [round(float(v), 1) for v in vmax],
    }


def track_payload(track: Any, max_speed: float = 40.0) -> dict:
    """The track as the canvases want it: a replay track dict + a `view` block (corners, v_max, bounds).

    `track` may be a registry key ("grand_prix", "random:7"), a sim Track, a plan() track map,
    a replay track (object or dict) or a whole replay dict.
    """
    rt = _replay_track_dict(track)
    rt["view"] = _track_view(rt, max_speed)
    return rt


def player_data(replay: Any, title: str | None = None) -> dict:
    """Replay dict + view blocks: exactly what the HTML player and the Streamlit replay canvas load."""
    d = dict(_replay_dict(replay))
    if title:
        d["title"] = title
    d["track"] = track_payload(d["track"], float(d.get("max_speed", 40.0)))
    duration = max((c["t"][-1] for c in d.get("cars", []) if c.get("t")), default=0.0)
    sig = json.dumps([d.get("title"), d.get("created_utc"), d["track"]["view"]["id"],
                      [(c.get("id"), len(c.get("t", [])), c.get("t", [0])[-1:]) for c in d.get("cars", [])]])
    d["view"] = {"id": hashlib.sha1(sig.encode()).hexdigest()[:12], "duration": round(float(duration), 2)}
    return d


# ============================================================================ the HTML page
_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="agentic_gp.viz.player">
<title>{title}</title>
<style>
html, body {{ margin: 0; background: #0d0d0d; color: #f0efec; }}
body {{ padding: {pad}; }}
{css}
</style>
</head>
<body>
<div id="agp-player"></div>
<script id="agp-replay" type="application/json">{data}</script>
<script>
(function () {{
"use strict";
{core}
{replay}
const data = JSON.parse(document.getElementById("agp-replay").textContent);
const player = new AgpReplayPlayer(document.getElementById("agp-player"), {opts});
player.setData(data);
window.agpPlayer = player;
document.addEventListener("keydown", (e) => {{
  if (e.target && /INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName) && e.key !== " ") return;
  if (player.handleKey(e)) e.preventDefault();
}});
}})();
</script>
</body>
</html>
"""


def replay_html(replay: Any, *, height: int | None = 640, autoplay: bool = True, title: str | None = None,
                speed: float = 1.0, follow: str | None = None) -> str:
    """A complete, self-contained HTML page that plays `replay` (no CDN, works offline).

    `replay`: Replay object, replay dict, path to a replay .json, SessionResult or GrandPrixResult.
    `height`: canvas height in px; None = fill the browser window (best for shared files).
    """
    data = player_data(replay, title=title)
    opts = {"height": height or 560, "fill": height is None, "autoplay": bool(autoplay), "speed": float(speed),
            "follow": follow}
    text = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return _PAGE.format(title=html.escape(data.get("title") or "Agentic GP replay"), css=asset("agp.css"),
                        data=text, core=asset("agp_core.js"), replay=asset("agp_replay.js"),
                        opts=json.dumps(opts), pad="0" if height is None else "8px")


def save_replay_html(replay: Any, path: str | Path, *, height: int | None = None, autoplay: bool = True,
                     title: str | None = None, speed: float = 1.0) -> Path:
    """Write the standalone player page to `path` (folders are created). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(replay_html(replay, height=height, autoplay=autoplay, title=title, speed=speed), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """python -m agentic_gp.viz.player REPLAY.json [OUT.html]"""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(main.__doc__)
        return 2
    src = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else src.with_suffix(".html")
    print(save_replay_html(src, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
