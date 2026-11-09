# Cheat sheet: Agentic Grand Prix

One page to print and keep next to your keyboard. All field names come from `agentic_gp/driver.py`
(`OBS_FIELDS`, `TRACK_FIELDS`, `REPORT_FIELDS`). Back to the [lesson overview](README.md).

## 1. "+ = right"

**Every angle, ray, offset and steering value your driver sees is positive to the RIGHT and negative to
the LEFT.** `angle_to_next_waypoint = +20` means the gate is 20° to your right, so steer positive.
`offset_from_center = -3` means you are 3 units left of the centerline.
The one exception is **world** angles (`heading_deg`, `start_heading_deg`, `track_heading_deg`): they are
counter-clockwise from +x, like a maths plot. Poster: [assets/driver_frame.svg](assets/driver_frame.svg).

## 2. Your file

```python
from agentic_gp import driverkit as dk

NAME = "Rocket"; AUTHOR = "ada"; COLOR = "#e63946"; KIND = "code"      # code | llm | fly
MOTTO = "See wall, turn early."; PARAMS = {"cap": 22.0}                # PARAMS: --param cap=25

def drive(obs):                                   # TICK clock: every 0.5 s, no LLM at races
    steering = dk.steer_towards(obs["angle_to_next_waypoint"], obs["speed"])   # + = right
    throttle = 1.0 if obs["speed"] < PARAMS["cap"] else 0.0
    return throttle, steering                     # both in [-1, 1]
```

Your file is `drivers/submissions/<handle>.py` (`<handle>` = your handle, lower case, no spaces; Day-1
chapters use the example `ada`). Keep `NAME` your own: the house cars Max, Planner and Fly Toy are taken.
`COLOR` is a hex code like `"#e63946"`; no emoji in `NAME`.

Class style (from Ch2): `class Driver` with `plan(self, track)` (SESSION, once per track),
`drive(self, obs)` (TICK) and `reflect(self, report)` (LAP, after every attempt). The same instance drives
every attempt, so `self.` attributes are your memory. `drive` may return `(t, s)`, `(t, s, say)` or
`{"throttle", "steering", "say", "debug"}`; `None` = coast.

## 3. What you receive

**`obs`, every tick**

| Group | Fields |
|---|---|
| speed | `speed` (0..40 units/s, negative when reversing) |
| 7 wall rays (max 200) | `distance_to_wall_front`, `distance_to_wall_left_20` / `_right_20`, `_left_45` / `_right_45`, `_left_90` / `_right_90` |
| waypoints | `distance_to_next_waypoint`, `angle_to_next_waypoint` (+ = right), `distance_to_waypoint_after_next`, `angle_to_waypoint_after_next`, `next_waypoint_index`, `waypoints_passed`, `waypoints_total` |
| where | `position` [x, y], `heading_deg` (world), `offset_from_center` (+ = right), `segment`, `progress` (this lap, 0..1), `lap`, `laps_total`, `race_progress` |
| clocks | `sim_time`, `penalty_time`, `race_time`, `tick`, `ticks_remaining`, `dt` (0.5) |
| status | `crashes`, `finished`, `attempt_over`, `zone_speed_limit` (None outside a zone), `phase`, `events` (text list) |

**`track`, in `plan()`:** `key`, `name`, `track_width`, `half_width`, `lap_length`, `n_waypoints`,
`start_position`, `start_heading_deg`, `waypoints` [{`index`, `center`, `track_heading_deg`,
`min_turn_radius_until_next`, `centerline_index`}], `centerline` (dense, about 4 units apart: plan on this),
`left_wall`, `right_wall`, `corners` [{`id` "T1", `start_index`, `apex_index`, `end_index`, `min_radius`,
`direction`, `v_max`, `from_wp`, `to_wp`}], `briefing` (yellow flags appear **only** here), `notes`,
`physics`, `laps`.

**`report`, in `reflect()`:** `attempt`, `phase`, `next_phase` (`"practice"` / `"qualifying"` / `"race"` /
None), `completed`, `race_time`, `lap_times`, `best_lap`, `best_lap_so_far`, `sim_time`, `penalty_time`,
`crashes`, `zone_violations`, `waypoints_passed`, `laps`, `laps_completed`, `ticks`, `ended_by`,
`crash_log` (with rays and corner), `segments`, `corners`, `telemetry`, `stalls`, `driver_errors`,
`llm_usage`, `summary_text` (paste it into a prompt).

## 4. Four physics facts

| Fact | Formula | Remember | In code |
|---|---|---|---|
| **The tick** | one `drive()` answer is held for 0.5 s | at 30 u/s you move 15 units blind; the track is 18 wide | `obs["dt"]` |
| **Stopping distance** | d = v² / (2·25) = v²/50 | 30 → 0 needs 18 units; plus one blind tick = 33 | `dk.braking_distance(v, v_to)`, `dk.stopping_speed(d)` |
| **Corner speed** | v_max = √(20·R) | R 20 → 20 u/s; home track's tightest corner (T5) 16.5 u/s | `dk.speed_for_radius(R)`, `track["corners"][i]["v_max"]` |
| **Turn radius** | R = v² / (20·\|steering\|), never tighter than v / (1.5·\|steering\|) (yaw limit, below 13.3 u/s) | steering 0.5 at 10 / 20 / 30 u/s → R ≈ 13 / 40 / 90 | `dk.steer_towards(angle, speed)` |

Also: top speed 40, full throttle +12 u/s², full brake −25 u/s². A crash costs **+5 s** and the marshals put
you back on the centerline at speed 0. 500 ticks per lap without finishing = DNF.

## 5. Commands (`python race.py <command> --help` shows all options)

| Do this | Command |
|---|---|
| check your setup | `python race.py doctor` (Day 2: `python race.py doctor --llm`) |
| list tracks | `python race.py tracks` |
| contract check (must say PASSED) | `python race.py check drivers/submissions/<handle>.py` |
| drive one lap on the home track | `python race.py drive drivers/submissions/<handle>.py` |
| other track / random track | `... --track sprint` · `... --track random:7` |
| learn over attempts, save result | `... --track sprint --attempts 6 --save` → `results/<name>_sprint.json`, named after your `NAME` (`"Rocket 2"` → `results/rocket-2_sprint.json`) |
| try a knob | `... --param cap=25` |
| picture / replay / HTML player | `... --plot races/me.png` · `--replay races/me.json` · `--html races/me.html` |
| the old crash rule (crash-loop demo) | `... --crash-rule classic` |
| all public + 3 random tracks | `python race.py validate drivers/submissions/<handle>.py --random 3` |
| LLM pilot vs reflex exam (Ch4) | `python race.py exam drivers/submissions/<handle>_pilot.py --states races/demo/exam_states.json --n 10` |
| A/B race (Ch5) | `python race.py grand-prix drivers/submissions/<handle>.py races/<handle>_ch3.py --track briefing_demo --practice 0 --laps 1` (`races/<handle>_ch3.py` = your Ch3 file with a different `NAME`) |
| watch a replay | `python race.py watch races/demo/class_demo.json` (`--html` browser, `--unity` 3D) |
| leaderboard from `results/` | `python race.py leaderboard` |
| test without an LLM | PowerShell: `$env:LLM_PROVIDER="mock"; python race.py drive ...` · macOS: `LLM_PROVIDER=mock python race.py drive ...` |

With uv, write `uv run race.py ...` instead of `python race.py ...`.

## 6. Streamlit

| Do this | Command |
|---|---|
| open Race Control | `python -m streamlit run app/race_control.py` → http://localhost:8501 (`?presenter=1` for a projector) |
| run a lesson | `python -m streamlit run streamlit_lessons/s01_hello_pitwall.py` (… `s10_race_replay.py`) |
| check the version (needs 1.66+) | `python -m streamlit version` |
| stop an app | **Ctrl+C** in its terminal (one app at a time) |

`st.write` / magic · `st.metric` · `st.slider` · `st.button` · `st.session_state` (memory) ·
`@st.cache_data` (do slow work once) · `st.columns` / `st.tabs` / `st.sidebar` ·
`@st.fragment(run_every=0.15)` (animate) · `st.navigation` (pages). Full table:
[streamlit_lessons/README.md](../streamlit_lessons/README.md#cheat-sheet).

## 7. LLM and driverkit helpers

```python
from agentic_gp import llm, driverkit as dk

text = llm.chat("Say hi in 5 words")                          # -> str  (effort="none" = fast, no thinking)
plan = llm.chat_json(prompt, system=SYSTEM)                   # -> dict; one retry if the answer is not JSON
plan = llm.chat_json(prompt, tools=[corner_speed])            # the model may call your Python functions first
run = llm.ask_with_tools(report["summary_text"], tools, max_rounds=3)   # run.final_text, run.calls, run.rounds
try:
    plan = llm.chat_json(prompt)
except llm.LLMError:                                          # no key, timeout, budget, LLMNotAllowed ...
    plan = physics_plan                                       # ALWAYS fall back to code
llm.llm_stats()        # calls, tokens, seconds        llm.describe_llm()   llm.last_reasoning()

dk.track_brief(track)                  # corner table + physics + briefing, ready for a prompt
dk.summarize_report(report, track)     # lap summary for an LLM
dk.PathFollower.from_track(track)      # .control(obs) -> (throttle, steering); .set_speed_limit(9, 11, 15.0)
dk.racing_line(track, margin=3.0)      # dk.curvature_radii(points), dk.speed_profile(points)
dk.recover_if_stuck(obs, state)        # dk.wall_repulsion(obs)
```

A tool is a plain function; its name, type hints and docstring become the schema the model sees:

```python
def corner_speed(radius: float) -> float:
    """Max speed (units/s) through a corner of this radius."""
    return dk.speed_for_radius(radius)
```

**Grand Prix LLM budget:** ≤ 2 calls in `plan()`, ≤ 2 per practice `reflect()`, ≤ 8 per track; none in
`drive()`, qualifying or the race.
