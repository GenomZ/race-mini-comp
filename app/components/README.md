# Shared Streamlit layer: canvases, helpers, demo data

This folder (`app/components/`) and `app/lib/` hold everything the Race Control pages share. The
page teams build on them. Read this file first; the docstrings in each module have more examples.

| Module | What it gives you |
|---|---|
| `app/components/race_canvas.py` | `replay_canvas`, `live_canvas` + `live_state`, `drive_pad` + `pad_controls` (Custom Components v2) |
| `app/components/browser_check.py` | real-browser smoke check (Playwright): screenshots, "did the cars move?", console errors |
| `app/lib/paths.py` | `REPO_ROOT`, `RACES_DIR`, `DEMO_DIR`, `RESULTS_DIR`, `DRIVER_FOLDERS`, `ensure_repo_on_path()`, `rel()` |
| `app/lib/loaders.py` | cached loaders: replays, replay payloads, tracks, JSON, sessions, results, standings tables |
| `app/lib/drivers.py` | driver discovery + static inspection (`DriverInfo`); student code is never executed |
| `app/lib/checkpoints.py` | "Where am I?": `where_am_i(path)` and the sidebar widget `render_where_am_i()` |
| `app/lib/params_ui.py` | `params_sliders(PARAMS)`: one slider per numeric knob |
| `app/lib/live_runner.py` | `LiveRun` (a session in a background thread) and `BackgroundJob` (any long function) |
| `app/lib/llm_trace.py` | read `llm_logs/*.jsonl`: `timeline()` for `st.chat_message`, `call_table()`, `latency_stats()` |
| `agentic_gp/viz/player.py` | standalone HTML player: `replay_html()`, `save_replay_html()`, `player_data()`, `track_payload()` |
| `agentic_gp/viz/plotly_figs.py` | Plotly figures: track map, overlays, telemetry, learning curve, heatmap, latency |
| `agentic_gp/viz/assets/` | the shared JavaScript and CSS: `agp_core.js`, `agp_replay.js`, `agp_live.js`, `agp_pad.js`, `agp.css` |
| `races/demo/make_demo.py` | regenerates the demo data (see the last section) |
| `app/_components_demo.py` | one page that uses every canvas: `streamlit run app/_components_demo.py` |

## 0. Imports and paths (every page)

`agentic_gp` is not pip-installed; it is imported from the repo root. `streamlit run app/race_control.py`
only puts `app/` on `sys.path`, so the entry script must add the repo root once (pages run in the
same process, so they inherit it):

```python
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]          # app/race_control.py -> repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Then import as `from app.components import replay_canvas` and `from app.lib.loaders import ...`.
Run Streamlit from the repo root so the repo's `.streamlit/config.toml` applies: dark theme,
`runOnSave = true`, and no reruns when `races/`, `results/` or `logs/` change.

## 1. The architecture in one paragraph

Python owns the simulation and the drivers. Each view is **one keyed canvas**: a Custom Components
v2 element (no iframe), whose JavaScript is the same code as the standalone HTML player. Python
sends **data only**. For a replay that is the whole race, sent once; for the live view it is the
newest tick. The browser animates at 60 fps with `requestAnimationFrame` and interpolates by race
time `t`. Always pass a unique `key`: with a key, a rerun updates the canvas in place, so the
replay keeps playing and the live car glides to its new pose. Without a key, every data change
remounts the canvas.

## 2. `replay_canvas`: the multi-car replay player

```python
from app.components import replay_canvas

res = replay_canvas("races/demo/class_demo.json", key="replay")             # that's it
res = replay_canvas(gp.replay, key="gp", height=600, autoplay=True, follow="max", report=True)
if res.t is not None:                                                         # only with report=True
    st.dataframe(standings_table(load_replay(path), res.t))
```

`replay_canvas(replay, *, key, height=560, autoplay=False, speed=1.0, follow=None, report=False, labels="auto", vmax=False, trails=True, fit=False)`

| Argument | Meaning |
|---|---|
| `replay` | Any of: a path to a replay JSON, a `Replay`, a replay dict, a `SessionResult` (all attempts become ghost cars), a `GrandPrixResult` (its race replay), or a payload from `load_replay_payload(path)`. Paths are cached per file and modification time. |
| `key` | **Required.** A unique string per canvas on the page. |
| `height` | Canvas height in px. The control bar adds about 50 px; the width always fills the container. |
| `autoplay` | Start playing when a **new** replay is loaded. A rerun with the same replay keeps the current time and play state. |
| `speed` | Initial playback speed: 0.25, 0.5, 1, 2, 4, 8 or 16. |
| `follow` | Car `id` to follow from the start; `None` shows the whole track. |
| `report` | `False` (default): the canvas **never** reruns your script. `True`: on pause, scrub release, follow change and at the end, the browser sends `t` (race seconds) and `follow` (car id or None). Each report is one rerun. |
| `labels` | Name tags: `"auto"` (full names for 6 cars or fewer, 3-letter codes for more), `"names"`, `"codes"` or `"none"`. |
| `vmax` | Colour the centerline by corner speed: bright orange = slow corner. |
| `trails` | Draw a 4 s fading trail behind each car. |
| `fit` | Presenter mode: size the canvas to the browser window (from its top edge down, at least 320 px); `height` is ignored. |

**Returns** the component result. `res.t` and `res.follow` are `None` until the browser reports.
The same values are in `st.session_state[key]` as `{"t": ..., "follow": ...}`. In AppTest you can
simulate a pause with `at.session_state["replay"] = {"t": 12.0, "follow": None}; at.run()`.

**What the viewer sees:**
- The track: asphalt, walls, kerbs on every corner, gate lines, a chequered start line, corner labels T1.., and yellow-flag zones with their speed limit.
- The cars: shaped, in their own colours, with name tags placed so they don't overlap, and trails.
- A **timing tower**: position, colour, name, and the gap to the leader computed from race progress ("Leader", "+3.4", "+1 LAP", "DNF", or the finish time with a chequered flag). It also shows the leader's lap and the race clock. A red "PEN" badge blinks while a car serves a penalty.
- A **radio and race-control ticker**: driver `say` messages, crashes, laps, finishes, zone penalties and stalls. On a wide canvas it sits in the left column; otherwise it is a strip at the bottom.
- **Crashes**: the car is drawn reaching the wall at the contact point (event x, y), then an expanding red ring, a persistent red X, and a blinking car with a "+5s" pill for the length of the penalty hold.
- When following a car: a HUD with position, speed, lap and crashes, throttle/brake and steering bars, and a chase camera (zoom with the mouse wheel).
- At the end: the chequered flag and the final classification (time, best lap, crashes). Click to close it.
- Controls: restart, play/pause, a scrub bar marked with crashes (red) and finishes (white), speed, and follow.
- Keys, while the player has focus (click it): Space play/pause, ←/→ ±5 s (Shift: ±1 s), +/- speed, Home restart, 1-9 follow the car in that position, 0/Esc overview. Clicking a car or a tower row follows it; clicking empty track goes back to the overview.

`replay_payload(replay)` returns exactly what the canvas loads. `load_replay_payload(path)` in
`app/lib/loaders.py` is the cached file version.

## 3. `live_canvas` + `live_state`: one car, tick by tick

```python
from app.components import live_canvas, live_state

live_canvas(live_state("grand_prix", obs=obs, action=(thr, steer), debug=dbg, name="Max",
                       color="#d4af37", status="thinking", say=[{"t": 12.5, "text": "box box"}]),
            key="live", height=480)
```

`live_canvas(state, *, key, height=480, fit=False)` draws one snapshot (`fit=True`: sized to the window, as above). Call it again with a new state on every
tick, usually inside a `@st.fragment(run_every=0.25)`. When `seq` changes, the car glides from
where it is to the new pose over `tween_ms` (default: the measured time between updates). It shows:
- the 7 rays, red under 10 units, amber under 25, green beyond, with distances in the close-up;
- the debug target point, as a pink crosshair with a dashed line from the car;
- the trail and the crash marks;
- a HUD with speed, throttle/brake and steering bars ("+ = right"), attempt, lap, tick, race clock, crashes and think time;
- a status pill, which counts the seconds while `status="thinking"`;
- the latest radio lines (inside the HUD panel on wide canvases, a strip at the bottom on narrow ones).

In overview a **close-up inset** follows the car. Clicking the canvas switches to a chase camera
(mouse wheel zooms); clicking again goes back to the overview.

`live_state(track, *, obs=None, car=None, action=None, debug=None, name="car", color="#3987e5", seq=None, status="driving", say=None, trail=None, crash_points=None, ghosts=None, attempt=None, attempts=None, phase=None, think_s=None, tween_ms=None, follow=False, show_rays=True, show_waypoints=False, vmax=False, gate_numbers=False, message=None, ray_max=200.0) -> dict`

| Argument | Meaning |
|---|---|
| `track` | A spec string (cached), a `Track`, a track map, a replay track, or a payload from `cached_track_payload(spec)`. |
| `obs` | The observation the driver saw. It supplies the pose, the 7 rays, tick, lap, race clock and crashes. |
| `car` | Pose override `{"x", "y", "heading_deg", "speed"[, "throttle", "steering"]}`, e.g. a replay frame. |
| `action` | `Action`, `(throttle, steering)`, or a dict. Drawn as the gauges. |
| `debug` | The driver's debug dict. `debug["target"] = [x, y]` is drawn as the target point. `driverkit.PathFollower.debug` already contains it. |
| `seq` | Anything that changes on every tick (default `"<attempt>:<tick>"`). A new value starts a tween. |
| `status` | `"driving"`, `"thinking"`, `"planning"`, `"reflecting"`, `"paused"`, `"finished"`, `"stopped"`, `"error"`, `"loading"` or `"dnf"`. |
| `say` | Radio lines `[{"t", "text"}]`, or plain strings. The last 3 are shown. |
| `trail`, `crash_points` | `[[x, y], ...]` in world coordinates. |
| `ghosts` | Other cars, drawn faded: `[{"x", "y", "heading_deg", "color", "name"}]` (compare mode, previous attempt). |
| `follow` | Start in the chase camera. |
| `show_waypoints` | Dashed arrows to the next and after-next waypoints, rebuilt from the obs angles. For Sensor X-ray. |
| `vmax`, `gate_numbers` | Colour the centerline by corner speed; number the gates. |
| `message` | A text in the middle of the canvas when there is no car yet (e.g. "Press Start"). |

Helpers: `rays_from_obs(obs) -> [{"angle", "dist"}]` (angle + = right), `waypoints_from_obs(obs)`,
`cached_track_payload(spec)`.

## 4. `drive_pad` + `pad_controls`: drive with the keyboard

```python
from app.components import drive_pad, pad_controls

@st.fragment(run_every=0.1)                        # 10 Hz polling; step the sim every 0.5 s
def hand_drive():
    pressed = drive_pad(key="pad")                 # {"up": bool, "down": bool, "left": bool, "right": bool}
    throttle, steering = pad_controls(pressed)     # up = 1.0, down = -1.0 (brake), left/right = -/+0.6
    ...
```

`drive_pad(*, key, title=None, subtitle=None) -> dict`: the user clicks the pad, then holds the
arrow keys or WASD. The on-screen keys also work with a mouse or touch. Each press and each release
causes one rerun (only the fragment, if the pad is inside one). Key auto-repeat is ignored, and
leaving the pad releases every key. The letters r and c are not used, because they are
Streamlit's own hotkeys.

`pad_controls(pressed, steer=0.6, brake=1.0, coast_throttle=0.0) -> (throttle, steering)`.

The full hand-drive loop, a `RaceEnv` in `st.session_state` stepped once per 0.5 s, is in
`app/_components_demo.py` (tab "Hand-drive"). For turn-based driving (Physics Lab: one click is
one tick), use `st.button(..., shortcut="Up")` instead.

## 5. `app/lib/live_runner.py`: run a driver live in a background thread

```python
from app.lib.live_runner import LiveRun

def start():                                                  # a button callback
    st.session_state.run = LiveRun("drivers/examples/l2_planner.py",
                                   SessionConfig(track="sprint", attempts=3), params=overrides, pace=4.0).start()

run = st.session_state.get("run")
polling = run is not None and not run.done                    # decided on every FULL rerun

@st.fragment(run_every=0.25 if polling else None)
def live():
    run = st.session_state.get("run")
    if run is None:
        live_canvas(live_state("sprint", message="Press Start"), key="live"); return
    run.heartbeat()                                           # "still watching"; it stops after 120 s without one
    live_canvas(run.canvas_state(), key="live")               # or run.canvas_state(follow=True)
    st.line_chart(run.telemetry_frame(), x="t", y="speed")
    for rep in run.reports(): st.code(rep["summary_text"])    # what reflect() was told
    if run.done and polling: st.rerun()                       # one full rerun turns polling off
live()
```

`LiveRun(driver, config=None, *, params=None, isolate=False, pace=4.0, idle_timeout_s=120)`

| Argument | Meaning |
|---|---|
| `driver` | A file path, or anything `load_driver` accepts. `isolate=True` runs it in its own process, which is slower to start but can be killed. |
| `pace` | Minimum wall time per tick = `tick_seconds / pace`. 1.0 = real time, 4.0 = 4x; `None` = as fast as possible (code drivers do about 1000 ticks/s). LLM drivers are slower than any pace anyway. |
| Control | `start()`, `pause()`, `resume()`, `toggle_pause()`, `step()` (one tick while paused), `stop(kill=False)`, `heartbeat()`, `join(timeout)`. |
| Reading | `snapshot() -> LiveSnapshot` (seq, status, message, name, color, kind, attempt, attempts, phase, tick, t, lap, laps, crashes, car, obs, action, debug, think_s, say, trail, crash_points, done, error, wall_s); `telemetry()` / `telemetry_frame()` (one row per tick: attempt, phase, tick, t, x, y, speed, throttle, steering, lap, progress, race_progress, offset, crashed, think_ms); `reports()` (one per finished attempt: race_time, lap_times, crashes, status, summary_text, ...); `events()` (a text log); `result` (the `SessionResult` when done); `replay(which="all")`; `canvas_state(**live_state_kw)`; `track_payload()`. |

The worker thread never calls `st.*` (Streamlit's multithreading rule). It only publishes
immutable snapshots under a lock.

`BackgroundJob(fn, key_field="driver")` runs any long function in a thread, for example a grand prix:

```python
job = BackgroundJob(lambda j: run_grand_prix(files, cfg, progress=j.post, stop_event=j.stop_event)).start()
s = job.state()        # status (running|done|error|stopped), latest (newest update per driver), log, result, error, elapsed_s
job.stop()             # sets stop_event
```

## 6. Other helpers in `app/lib`

- **loaders**:
  - `load_json(path)`, `file_kind(data)` (`"replay"` | `"session"` | `"result"` | `"grand_prix"` | `"ticks"`), `list_json(root, kind)`.
  - Replays: `list_replays(root="races")` (newest first), `replay_label(path)` (for selectboxes), `load_replay(path) -> Replay`, `replay_from_upload(bytes)`, `load_replay_payload(path)`.
  - `standings_table(replay, t=None)`: official classification, or the live order at time `t`.
  - Tracks: `load_track(spec)` (shared `Track`, don't modify), `track_map(spec)`, `track_choices(n_random=3)`.
  - Sessions and results: `load_session(path)`, `list_results()`, `attempts_table(result)`, `telemetry_rows(session)`, `demo_file(name)` (raises a helpful error if the demo data is missing).
- **drivers**:
  - `discover_drivers()` returns a list of `DriverInfo`: path, group (submissions / examples / checkpoints / template), stem, name, author, color, kind, model, motto, anatomy, style (`"function"` | `"class"` | `"none"`), has_plan, has_drive, has_reflect, params, param_help (the comment next to each PARAM), llm_in (which of plan/drive/reflect call the LLM, also through helper methods), features, syntax_error, label, rel_path, uses_llm.
  - `inspect_driver(path)`, `find_driver(name_or_path)`.
- **checkpoints**: `where_am_i(path)` returns a `WhereAmI` with level ("CP0"..."CP8" or "-"), checks (cp, title, ok, hint, side), issues (from `race.py check`), next_step and summary. `render_where_am_i(container=None, key="where_am_i", expanded=False)` is the sidebar expander. The main ladder is CP0 → CP1 → CP2 → CP3 → CP5 → CP6 → CP8; CP4 (LLM pilot) and CP7 (fly) are side quests. It is a static check of the code's parts, not of lap times.
- **params_ui**: `params_sliders(params, *, key, help=None, ranges=None, container=None, columns=2) -> dict` gives numbers a slider (range guessed from the default, or `ranges={"aggression": (0.5, 1.5, 0.05)}`), bools a checkbox, and shows anything else read-only. Also `slider_spec(default)`, `changed_params(defaults, chosen)` and `params_label(defaults, chosen)`.
- **llm_trace**: `find_llm_logs()` searches `races/**/llm_logs/*.jsonl` and `llm_logs/`. `read_llm_log(path)`. `timeline(records)` returns `Step`s (role, kind = prompt / answer / tool_call / tool_result / error, text, name, args, think, seconds, tokens, call, phases, label) and shows only what is new in each request of a tool loop. Also `call_table(records)` and `latency_stats(seconds)` (n, mean, p50, p95, max).

## 7. Plotly figures (`agentic_gp.viz.plotly_figs`)

Display them with `st.plotly_chart(fig, theme=None)`; `theme=None` keeps the dark broadcast colours.
- `track_figure(track, *, vmax=True, corners=True, gates=True, zones=True, gate_numbers=False, title=None, height=560)`: asphalt, walls, gates, start/finish, T1.. labels, zones, and the centerline coloured by v_max = sqrt(20 R) (hover shows point, v_max and segment).
- Overlays:
  - `add_car(fig, x, y, heading_deg, color=, name=)`
  - `add_rays(fig, obs)`, coloured by distance
  - `add_target(fig, target, car=(x, y))`
  - `add_trajectory(fig, telemetry, by_speed=True)`
  - `add_crashes(fig, points)`
  - `add_turning_circle(fig, x, y, heading_deg, speed, steering)`: R = v²/(20|s|) (Physics Lab)
  - `car_polygon(...)`
- `telemetry_figure(data, x="t", compare=None)`: speed, throttle/brake and steering with a shared x axis. `data` is a report, rows, a DataFrame, a session dict (its last attempt) or a replay car.
- `speed_profile_figure(track, actual=None)`: grip limit → after the braking pass → actual (Ch2).
- `learning_curve_figure(results)`: race time per attempt per driver, DNF as an X, best-so-far dotted. `results` is a session/result dict, a list of them, or `{"name": [t, None, ...]}`.
- `segment_heatmap(session, value="time_lost")`: attempts × segments, seconds lost against the best attempt; X marks a crash.
- `latency_histogram(seconds, tick=0.5)`: calls vs the red one-tick line, with p50 and p95 marked.
- `blind_distance_figure(track, x, y, heading_deg, latency_s, speed=30)`: the stretch driven blind.
- `style(fig)`: apply the theme to your own figures.

## 8. Standalone HTML player (`agentic_gp.viz.player`)

```python
from agentic_gp.viz.player import save_replay_html, replay_html
save_replay_html("races/final.json", "final.html")          # fills the browser window; works offline
st.download_button("HTML player", replay_html(replay, height=None), "race.html", "text/html")
```

From the command line: `python race.py watch races/demo/class_demo.json --html` (or `python -m agentic_gp.viz.player X.json`).
The file embeds the replay JSON (with `<` escaped) and the same JavaScript as the replay canvas.
It loads nothing from the internet. Size is about 60 KB plus the replay.

## 9. Demo data (`races/demo/`)

Regenerate it with `python races/demo/make_demo.py` (about 30 s, mock LLM; `--only class_demo`, `--real-llm`).
It uses whatever drivers exist in `drivers/examples/`.

| File | Content |
|---|---|
| `l2_session.json` | `SessionResult.to_dict()`: the planner on grand_prix, 3 attempts. Reports carry telemetry, segments, crash_log and summary_text. Telemetry and Learning pages. |
| `l2_session_replay.json` | The same session as a replay, one ghost car per attempt. |
| `l2_ticks.json` | One attempt tick by tick, `{"ticks": [{"tick", "obs", "action", "debug", "think_seconds", "after"}]}`. Sensor X-ray. |
| `learner_session.json` (+ `_replay`) | l3_learner, 5 attempts on sprint. Learning curve and segment heatmap. |
| `class_demo.json` | Race replay: every example driver except l0_dummy and the flies on grand_prix, 2 laps (mock LLM; two planner variants "Careful"/"Brave" are added when fewer than 7 drivers exist). Also `class_demo_qualifying.json`, `class_demo_results.json`, `class_demo_standings.md` and `html/class_demo.html`. |
| `crash_loop.json` | l0_dummy under the classic crash rule: 109 crashes in 240 ticks (Ch1). |
| `llm_logs/class_demo/*.jsonl` | LLM traces of the LLM drivers in the class demo (mock replies by default). |

## 10. Testing and verifying

- Unit and AppTest coverage: `.venv/Scripts/python.exe -m pytest -q tests/test_viz_player.py`. In AppTest a CCv2 component is `at.get("bidi_component")`. Read its data with `json.loads(c.proto.json)` and simulate the browser with `at.session_state[key] = {...}`.
- Real browser: `python app/components/browser_check.py app/race_control.py --pages replay live --play --out shots`. It prints load times, `cars_moved`, console errors and screenshot paths. Look at the PNGs.
- Gotchas:
  - Never call `st.*` from a thread.
  - Every canvas needs a `key`.
  - Send plain floats and lists (the helpers already do).
  - A canvas inside an inactive `st.tabs` tab has zero width. It skips drawing and catches up when shown.
  - `st.line_chart` in a hidden tab logs harmless Vega "Infinite extent" warnings.
