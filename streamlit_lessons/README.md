# Streamlit in two hours: build your own Race Control

**Day 1, 13:30 to 15:30.** Ten tiny apps, one new idea each. You start with a title and four
numbers. Two hours later you own a **multi-car race replay viewer** and a **"My learning curve"**
page. You will use both at 16:35 in the Day-1 Club Race and again in Chapter 3.

Nothing in this block depends on your own driver working. Every lesson reads demo data through
`streamlit_lessons/_data.py`. If a demo file is missing, `_data.py` computes it with the simulator
(code drivers run a lap in under a second).

This file has two parts:

- **Part A, for the instructor**: the run sheet, with timings, what to say, and what to cut when you
  fall behind.
- **Part B, for students**: a step-by-step guide. For each lesson it gives what to type, what you
  should see, the exercise, and a hint.

At the end there is a cheat sheet, a section on common errors and fixes (Windows and macOS), and a
list of files.

---

## The one idea behind everything (show this first)

```
   you click / drag / type / save the file
                 |
                 v
   Streamlit RERUNS your whole script, top to bottom
                 |
                 v
   every st.* call draws something  ->  the page you see
                 |
                 +----> waits for the next event ...
```

It looks a lot like the driver loop you wrote this morning:

| Streamlit | Your driver |
|---|---|
| a widget value (`st.slider` returns a number) | a **percept**: `obs["speed"]` |
| `st.session_state` (survives reruns) | **memory**: `self.history` |
| `@st.cache_data` (do the slow work once) | **experience**: the plan you computed in `plan()` |
| a rerun | a **tick** |

---

# Part A - Instructor run sheet

## Before the block (10 minutes, ideally during lunch)

Every command below runs from the repo root **with the course venv active** (see "Start here" in
Part B). Nothing in this block downloads packages.

1. Run the lesson tests (pytest comes with `requirements.txt`):
   ```
   python -m pytest -q tests/test_streamlit_lessons.py
   ```
   It should finish green in about a minute. If your venv was made with `uv sync` (no dev extra),
   pytest is missing: `uv pip install pytest` once, or `uv sync --extra dev`.
2. Check the Streamlit version: `python -m streamlit version` must say **1.66 or newer**. An older
   one (the repo once allowed 1.50) breaks S06, S07, S08 and S10; the lessons then stop with
   "These lessons need Streamlit >= 1.66" and the upgrade command.
3. Optional: regenerate the demo data with `python races/demo/make_demo.py` (about a minute).
   The lessons also work without it.
4. Open the final app once on the projector machine. This loads Plotly and the canvas JavaScript
   into the browser cache:
   ```
   python -m streamlit run streamlit_lessons/s10_race_replay.py
   ```
   On Windows, accept the firewall prompt ("Private networks"). Tick **Play** once to check it.
5. Zoom the browser to 125 % for the projector (Ctrl/Cmd and +). The dark theme comes from
   `.streamlit/config.toml`.
6. Write on the board: **"Stuck for 2 minutes? Red sticky note up."** and
   **"Compare with `streamlit_lessons/solutions/`."**

## Timeline (120 minutes)

| Clock | Min | Step | Concept | What students do, and what they should see |
|---|---|---|---|---|
| 13:30 | 5 | **Opener** | rerun model | The loop diagram above. Live-code `st.write("hi")` and save: the page changes. |
| 13:35 | 6 | **S01** `s01_hello_pitwall.py` | `streamlit run`, magic, `st.metric` | Four lap tiles (DNF / 65.0 / 50.8 / Max 46.9 s with the committed mock demo) and the real `obs` dict. |
| 13:41 | 9 | **S02** `s02_corner_calculator.py` | widgets return values | Drag R: the speed changes. Change speed and steering: the red arc changes. |
| 13:50 | 12 | **S03** `s03_track_map.py` | `st.plotly_chart`, `@st.cache_data` | The track coloured by corner speed. Back on grand_prix, the caption says "from the CACHE". |
| 14:02 | 12 | **S04** `s04_telemetry.py` | pandas, charts, download | 4 practice runs. Speed and controls charts. A segment table. A CSV download. |
| 14:14 | 10 | **S05** `s05_driver_card.py` | sidebar, columns, tabs, expander | A driver card with the lap on the map, coloured by speed. |
| 14:24 | 5 | **Stretch** | | Stand up. Water. Close old Streamlit terminals (Ctrl+C). |
| 14:29 | 12 | **S06** `s06_run_my_driver.py` | button, `st.status`, cache keys, toast | A slider per `PARAMS` knob, then Race: 2 ghost cars replay. |
| 14:41 | 12 | **S07** `s07_be_the_driver.py` | `st.session_state`, callbacks, `st.rerun` | Students drive the car by clicking: one click = one tick. |
| 14:53 | 14 | **S08** `s08_live_view.py` | `@st.fragment(run_every=...)`, `st.progress` | The driver laps live: rays, target point, gauges, radio. Students step tick by tick. |
| 15:07 | 5 | **S09** `s09_track_lab.py` (demo only) | `data_editor`, `form`, `on_select` | Instructor demo. Students do it as homework (CURRICULUM.md: "S09 Track Lab becomes homework"). |
| 15:12 | 18 | **S10** `s10_race_replay.py` | `st.navigation`, scrubbing, interpolation | Their own multi-car viewer plus the pro canvas. The exercise is the "My learning curve" page. |
| 15:30 | | **End** | | Everyone has S10 open. Break. |

**If you have time**, give S09 its full 10 minutes and finish at 15:35. That eats 5 minutes of
the break, so only do it if the room is ahead.

**If you are behind**, cut in this order:

1. S09 becomes a 2-minute "look at this" or homework.
2. S05's exercise becomes homework.
3. The S08 exercise becomes "read the solution together" (`solutions/s08_live_view.py`, 3 min).
4. Never cut S07, S08 or S10. They are the three that pay off later (Physics Lab, Live, the
   Club Race replay).

## What to say at each step (one or two sentences)

- **S01:** "A Streamlit app is just a Python script. Every time anything happens, it runs again
  from the top. That's the whole trick."
- **S02:** "A slider is a variable that your user sets. Move it and the script reruns with the new
  value. Notice that nothing is 'event-driven' here: there are no callbacks and no handlers."
- **S03:** "Drawing the Grand Prix takes about 250 ms, and a rerun happens on every click.
  `@st.cache_data` means 'I've seen these arguments before, here's the answer': 10 to 40 ms. It is
  the plan-once idea from Chapter 2." Show it on grand_prix: the other tracks draw in 40 to 100 ms,
  so the difference is small there.
- **S04:** "Telemetry is just a table. pandas plus three chart calls get you most of a data
  dashboard."
- **S05:** "Layout is cheap: sidebar, columns and tabs are `with` blocks. Same data, a much better
  page."
- **S06:** "A button is True for exactly ONE rerun. Anything you want to keep goes into
  `st.session_state`. The cache key includes the file's modification time, so saving your driver
  forces a new run."
- **S07:** "You are now the `drive()` function. Every click is one tick of 0.5 s. Feel how little
  you can do per tick. That is what your driver lives with."
- **S08:** "Compute first, animate second. The whole lap is driven in under a second, then we play
  it back. The fragment reruns only itself, 7 times a second."
- **S09:** "A form batches edits. A chart with `on_select` sends a click back to Python."
- **S10:** "`st.navigation` turns ten scripts into one app. Your viewer interpolates every car
  between its 0.5 s frames, which is exactly what the pro player and the Unity viewer do."

## Room logistics

- One app at a time. Each `streamlit run` takes a port (8501, then 8502 ...). Before starting the
  next lesson, press Ctrl+C in the old terminal. Keep a single terminal per student.
- Students edit the lesson files in place. To start one over:
  `git checkout -- streamlit_lessons/s04_telemetry.py` (the file you changed). To start all of them
  over: `git checkout -- streamlit_lessons/`.
- Solutions run as they are: `python -m streamlit run streamlit_lessons/solutions/s04_telemetry.py`.
- To see what changed in a solution:
  `git diff --no-index streamlit_lessons/s04_telemetry.py streamlit_lessons/solutions/s04_telemetry.py`.
  A solution is the lesson file plus one `SOLUTION` line in the docstring and the changed lines
  (marked `# SOLUTION`), so the diff shows only the exercise. All files use LF line endings
  (`streamlit_lessons/.gitattributes`), so there are no line-ending warnings.

---

# Part B - Student guide

**Start here (once per terminal)**

Open a terminal in the **repo root** (the folder with `race.py`) and activate the course venv:

```
.venv\Scripts\activate            (Windows)
source .venv/bin/activate          (macOS)
```

Your prompt now starts with `(.venv)`. Check the version once: `python -m streamlit version`
must say 1.66 or newer. Every command in this guide is then `python -m streamlit run ...`.

Don't want to activate? Use the venv's Python directly:
`.venv\Scripts\python -m streamlit run ...` (Windows) or `.venv/bin/python -m streamlit run ...`
(macOS). uv users can also write `uv run --no-sync streamlit run ...`: it uses the venv as it is and
never touches the network, which is what you want on the class Wi-Fi. (The main README's plain
`uv run` form also works right after `uv sync`; it first checks the venv against `uv.lock`.)

**How every step works**

1. Open the file in your editor. Read the docstring at the top: Concept, Run, You should see,
   Exercise.
2. Start it from the repo root:
   ```
   python -m streamlit run streamlit_lessons/s01_hello_pitwall.py
   ```
3. The browser opens at http://localhost:8501. If it doesn't, open that address yourself.
4. Check "You should see". Then do the exercise: search the file for `# TODO(exercise)`.
5. **Save the file and watch the browser.** The page updates by itself (`runOnSave = true`).
6. Stop the app with **Ctrl+C** in the terminal before you start the next one.
7. Stuck for 2 minutes? Look at `streamlit_lessons/solutions/<same file name>`, or put your red
   sticky note up.

---

### S01 Hello, Pit Wall (6 min): your first app

**Teaches:** `streamlit run`, the rerun model, `st.title`, `st.markdown`, `st.metric`, and "magic"
(a value alone on a line gets displayed).

```
python -m streamlit run streamlit_lessons/s01_hello_pitwall.py
```

**You should see:**
- the title "Pit Wall";
- four tiles: L0 Dummy **DNF**, L1 Reflex **65.0 s**, L2 Planner **50.8 s**, Max (house) **46.9 s**.
  The times are the *best lap of the 2-lap class demo race* (`races/demo/class_demo.json`). Lap 2
  starts with a flying start, so they are faster than the 1-lap `race.py drive` times you measured
  in Ch1 and Ch2 (66.70 s and 52.50 s). The committed demo race was made with the offline mock LLM;
  if your instructor regenerated it with the real LLM, Max shows its real best lap instead;
- the full `obs` dict your `drive()` receives every tick, as a JSON tree that is fully expanded (more
  than 40 lines: scroll down for the blue box);
- a blue box that reads the dict for you ("+ = RIGHT ...").

**Try:** change `st.title("Pit Wall")` to your team name and save. Then put a bare `obs["speed"]`
on a line of its own and save.

**Exercise:** add a 5th tile "Me" with your best lap, and its gap to Max as the delta.
*Hint:* `st.columns(len(laps) + 1)`, then `gap = my_lap - laps["Max (house)"]` and
`tiles[-1].metric("Me", f"{my_lap:.1f} s", delta=f"{gap:+.1f} s", delta_color="inverse")`.
`"inverse"` turns "bigger is worse" into red, which is right for lap times.

---

### S02 Corner Calculator (9 min): every widget is a variable

**Teaches:** `st.slider`, `st.number_input`, `st.radio`. Each one returns a value, and changing it
reruns the script.

```
python -m streamlit run streamlit_lessons/s02_corner_calculator.py
```

**You should see:**
- "Corner radius R" (40) and "Fastest speed through it" **28.3 u/s**. That is `sqrt(20 * 40)`.
  Drag it to 80 and you get 40.0.
- "To stop you need 18 units of braking + 15 blind = 33 units" at 30 u/s. Braking takes `v²/50`,
  and you drive `0.5 v` blind before your driver can even react.
- A chart: **your path** (red), the circle your car drives with this speed and steering, against
  **the corner** (grey). Its radius is `max(v²/20, v/1.5) / |s|`: full steering gives the tightest
  circle, half steering one twice as wide (30 u/s at steering 0.5: R = 90). If your path is wider
  than the corner, you hit the wall.

**Exercise:** "Can I make it?" Add a `number_input` for the front-ray distance, then show
`st.success` if `stop + blind <= front`, otherwise `st.error`.
*Check:* at 30 u/s, a front distance of 40 shows green and 20 shows red.

---

### S03 Track Map (12 min): Plotly + caching

**Teaches:** `st.plotly_chart` (zoom, pan, hover), `st.selectbox`, `@st.cache_data`.

```
python -m streamlit run streamlit_lessons/s03_track_map.py
```

**You should see:**
- the Grand Prix circuit: walls, gates, start line, T1 ... T10;
- the centreline dots coloured by corner speed (bright orange = slow corner);
- a caption with the build time;
- a corner table (`min_radius`, `v_max`).

Pick "sprint", then "grand_prix" again. The caption now says **"Figure from the CACHE"**: drawing
took about 250 ms once, getting it back takes 10 to 40 ms.

**Try:** hover a centreline dot. It shows the point, the segment and v_max. Zoom with the mouse
wheel. Double-click to reset the view.

**Exercise:** add four checkboxes (v_max colours, corner labels, gates, zones) in `st.columns(4)`.
Pass them to `build_map(spec, layers)` and on to `pf.track_figure(..., **layers)`.
*Why must the layers be arguments of the cached function?* Because the cache key is made of the
arguments. If you leave them out, you get the old picture back.

---

### S04 Telemetry (12 min): pandas, charts, downloads

**Teaches:** turning a list of dicts into a `DataFrame`, then `st.dataframe`, `st.line_chart`,
`st.bar_chart` and `st.download_button`.

```
python -m streamlit run streamlit_lessons/s04_telemetry.py
```

**You should see:**
- a "practice day" of the Reflex driver on the sprint track: 4 runs with different PARAMS
  (73.8 s with 4 crashes, 71.7 s with 3, 50.0 s, 43.95 s);
- the speed chart. On runs 1 and 2 the speed drops to 0 at each crash and stays flat for the 5 s
  penalty (`_data.telemetry_rows` adds one `crash=True` row at the moment of each impact);
- throttle (red) and steering (blue);
- time per segment as bars and as a table;
- a CSV download button.

**Exercise:**
1. Show only the **crash ticks** of the selected run:
   `pd.DataFrame(session["reports"][run - 1]["crash_log"])`. Show `st.success` when it is empty.
2. Add a **"time lost"** column: this run's segment time minus the best run's time in the same
   segment. Take only two columns of the best run, rename its time, then merge:
   `best = pd.DataFrame(session["reports"][best_run - 1]["segments"])[["segment", "time"]]` and
   `segments.merge(best.rename(columns={"time": "best run"}), on="segment")`. (Merging the whole
   table gives 16 columns with `_x` / `_y` names.)

---

### S05 Driver Card (10 min): layout

**Teaches:** `st.sidebar`, `st.columns`, `st.tabs`, `st.expander`, `st.chat_message`.

```
python -m streamlit run streamlit_lessons/s05_driver_card.py
```

**You should see:**
- the run picker in the sidebar;
- three tiles (race time, crashes, top speed);
- the lap on the track map, each dot coloured by speed (hover a dot for t and speed);
- Speed / Controls / Radio tabs;
- the raw report folded inside an expander. This is exactly what `reflect()` receives.

**Exercise:** add a **"Crashes"** tab. In it, put a table of `report["crash_log"]` (tick, corner,
speed_before), and draw red X markers on the map with
`pf.add_crashes(fig, [c["position"] for c in crashes])`. Pick run 1 in the sidebar to see 4
crashes.

**5-minute stretch.** Stand up, drink some water, and Ctrl+C your old apps.

---

### S06 Run My Driver (12 min): buttons, status, cache keys

**Teaches:** `st.button` is True for **one** rerun, so results go into `st.session_state`. Also
`st.status`, `st.toast`, and a cache keyed on (file modification time, track, params).

```
python -m streamlit run streamlit_lessons/s06_run_my_driver.py
```

**You should see:**
- a driver picker. Your `drivers/submissions/*.py` come first; otherwise `l2_planner.py` is
  selected.
- **One slider per `PARAMS` knob**, generated from the file. The (?) help text is the comment next
  to each knob in your code.
- Press **Race!**: a status box, a toast with the best lap, then a replay with your 2 attempts as
  ghost cars.
- Race again without changing anything: the answer is instant (cache). Edit and save your driver
  file, race again: it drives again, because the file's modification time is part of the key.
- An int knob (like `racing_line = 0`) gets an int slider, a float knob a float slider, so your
  driver gets back the same type it declared (`3`, not `3.0`).

**Exercise:** compare mode. Before you overwrite `s.s06_run` (`s` is `st.session_state`), save the
old run in `s.s06_prev`. Then put both runs' cars into one replay, and add an `st.metric` with the
time difference.
*The data:* `race()` returns `{"best": 52.1, "replay": {...}}`. A replay is a dict with `"track"`
and `"cars"`, a list of cars like `{"id": "...", "name": "...", "color": "#...", "t": [...],
"x": [...], ...}`. Copy each old car with a new `id`, a name like `"prev ..."` and a grey colour
(`dict(car, id=..., name=..., color="#9a9a9a")`), and add them to a copy of the new replay's
`"cars"` list.

---

### S07 Be the Driver (12 min): session_state and callbacks

**Teaches:**
- `st.session_state` keeps a live `RaceEnv` (the simulator) between reruns.
- `on_click=` callbacks run **before** the rerun.
- `st.rerun()` restarts the script right now.

```
python -m streamlit run streamlit_lessons/s07_be_the_driver.py
```

**You should see:**
- five buttons: Left, Gas, Coast, Brake, Right;
- your red car on the sprint track, in a chase camera, with rays coloured by distance and dashed
  arrows to the next gates;
- one click = **one tick = 0.5 s**. The HUD shows speed, throttle and steering. The arrow keys
  and Space work too: click the page background first.
- "New car" puts you back on the grid.

**Try:** brake into T1 at 30 u/s. Count how many clicks it takes. That is why drivers must look
ahead.

**Exercise:**
1. **Autopilot.** Add a toggle. While it is on, each click asks a driver file for the action.
   Import it with `from agentic_gp import RaceEnv, load_driver`, pick the file with
   `st.selectbox(..., _data.driver_files(), ...)`, then `load_driver(str(path))`,
   `.plan(s.s07_env.track_map())` once, and `.drive(s.s07_obs).as_tuple()` on each click.
2. **Undo.** Keep a list of every action. Undo builds a new `RaceEnv` and replays all actions
   except the last one. This works because the simulator is **deterministic**: same actions,
   same race.

---

### S08 Live View (14 min): fragments and animation

**Teaches:**
- `@st.fragment(run_every=0.15)`: a function that reruns **by itself**, without the rest of the
  page;
- `st.progress`;
- "compute first, animate second".

```
python -m streamlit run streamlit_lessons/s08_live_view.py
```

**You should see:**
- the planner lapping the sprint track;
- 7 rays: red means a wall closer than 10, amber closer than 25, green beyond;
- the **pink target point** from the driver's `debug` dict;
- throttle and steering gauges and the radio lines ("Braking for T1");
- a progress bar "tick n of 82 - driving". At the end of the lap the HUD and the bar say
  **finished**. Restart shows the car on the grid at tick 0.

Untick **Play** and press **Step one tick** to go through the lap. Pick your own driver in the box.

**The old way (do not use it):**
```python
box = st.empty()
for f in frames:                 # blocks the whole script; any click restarts it from the top
    box.write(f["speed"])
    time.sleep(0.15)
```
A click during this loop restarts the script, so the click acts as a Stop button. A fragment does
the same job without blocking the page.

**Exercise:**
1. A playback-speed slider (ticks per refresh: 1, 2, 4 or 8).
2. A **realtime** toggle and a think-time slider. The command your car is executing was decided
   `think` seconds ago, that is `lag = math.ceil(think / 0.5)` ticks. Draw a grey **ghost** car
   where it was then: `live_state(..., ghosts=[{"x": ..., "y": ..., "heading_deg": ...,
   "color": "#888888", "name": "decided here"}])`, using `frames[s.s08_i - lag]["obs"]`
   (its `"position"` is `[x, y]`). Add an `st.progress` gauge: "command age 1.5 s = 3 ticks,
   drove ~45 units blind". This is a preview of Chapter 4: LLM drivers think for 1 to 3 s.

---

### S09 Track Lab (10 min, or homework): edit, submit, click

**Teaches:** `st.data_editor`, `st.form` (no rerun while you type), and
`st.plotly_chart(on_select="rerun")`.

```
python -m streamlit run streamlit_lessons/s09_track_lab.py
```

**You should see:**
- the Grand Prix control points as an editable table. Change a few numbers, or add rows at the
  bottom, then press **Build track**.
- Then either a green "Valid! 1554 units long, tightest corner R = 14" or a red list of problems
  (tight corner, crossing walls, start not on a straight);
- your circuit coloured by corner speed. Click a centreline dot: its data appears below the chart.
- **Race here** sends the L2 planner around your track. It does the original layout in 53.0 s.

**Exercise:** design a track that breaks your neighbour's driver. Pick their file in a selectbox,
race it, keep the result in `st.session_state`, and draw its trajectory and crash markers.

---

### S10 Race Replay (18 min): your own multi-car viewer

**Teaches:** a multipage app with `st.navigation` and `st.Page` (S01 to S09 become pages of one
app), loading a big JSON once with `@st.cache_data`, a time scrubber, and interpolation.

```
python -m streamlit run streamlit_lessons/s10_race_replay.py
```

**You should see:**
- a sidebar menu: "Race replay", then S01 to S09. Click through them: it is now one app.
- "Race replay" loads the class demo race (8 cars, 2 laps of the Grand Prix circuit).
  On the map a car right next to an already-labelled car gets no name label, so the names never
  pile up (at the start, on the grid, only a few cars are labelled; hover a car, or pick it in
  Follow). *Identical cars?* If the demo race was made with the offline **mock** LLM (the
  default of `make_demo.py`, and of the fallback race `_data.py` builds when `races/demo/` is
  missing), the LLM cars (Strategist, Max, Pit Crew) fall back to the racing line: they drive
  exactly like Racing Line, share its spot and show +0.00 in the tower. The instructor makes a
  real race with `python races/demo/make_demo.py --only class_demo --real-llm` (spends LLM
  budget, about a minute).
  - **My viewer** is the one you own: a race-clock slider, the cars on the map, a timing tower
    and a radio feed. Tick **Play** and the clock advances by itself. Pick a car in **Follow**
    and the map zooms in on it.
  - **Pro player** is the shared 60 fps canvas: the same player as Race Control and the HTML
    export. Press Space to play or pause, click a car to follow it, and press 1 to 9 to follow
    the car in that position.

How your viewer works: a replay stores each car's frames every 0.5 s. `car_state_at(car, t)`
interpolates the position at any time `t`, and `standings_at(replay, t)` sorts the cars by race
progress. That is the whole multi-car viewer, in about 30 lines.

**Exercise: the "My learning curve" page.** You will use it in Chapter 3.
1. Write a function `my_learning_curve()` above the `pages = ...` line:
   - `st.title("My learning curve")`
   - `files = _data.result_files()`. These are `results/*.json` files, written by
     `python race.py drive drivers/submissions/<handle>.py --track sprint --attempts 6 --save`
     and named after your car's `NAME` (`"Ada"` → `results/ada_sprint.json`).
     If you have none yet, you get the demo learner session. Old files from the v1 course are
     skipped.
   - an `st.multiselect` of the files, then load them with `_data.load_json`
   - `st.plotly_chart(pf.learning_curve_figure(results), theme=None)`
   - a metric "gain vs attempt 1 in %". **Checkpoint CP3** asks for at least 3 % and no DNF.
     The lap times are in each result's `attempts` list:
     `times = [a["race_time"] if a["completed"] else None for a in res["attempts"]]`
     (`None` = that attempt did not finish, a DNF).
2. Add `st.Page(my_learning_curve, title="My learning curve")` to the "My app" list.

**You did it.** You built ten apps, drove the car by hand, watched a driver think live, and
scrubbed through a multi-car race in a viewer you wrote yourself. At 16:35 the Club Race is
replayed in this app.

---

## Cheat sheet

| I want to ... | Streamlit |
|---|---|
| show text / numbers / anything | `st.write(x)`, or just `x` on its own line ("magic") |
| a big number with a delta | `st.metric("Lap", "52.1 s", "-0.4 s", delta_color="inverse")` |
| get a number from the user | `st.slider`, `st.number_input`, `st.select_slider` |
| pick one of many | `st.selectbox`, `st.radio`, `st.multiselect` |
| on/off | `st.toggle`, `st.checkbox` |
| do something on click | `if st.button("Go"):` (True for ONE rerun), or `st.button("Go", on_click=fn)` |
| remember between reruns | `st.session_state.x = ...` / `st.session_state.get("x")` |
| do slow work once | `@st.cache_data` (data: dicts, frames, figures) / `@st.cache_resource` (shared objects) |
| side by side | `a, b = st.columns(2)` then `a.metric(...)` or `with a: ...` |
| hide detail | `st.tabs([...])`, `st.expander("...")`, `st.sidebar` |
| animate | `@st.fragment(run_every=0.15)` around the part that changes |
| several pages | `st.navigation([st.Page("a.py"), st.Page(func)]).run()` |
| a race canvas | `from app.components import replay_canvas, live_canvas, live_state` (`app/components/README.md`) |
| a track figure | `from agentic_gp.viz import plotly_figs as pf` then `pf.track_figure("sprint")` |

**What `_data` gives you:** `reference_laps()`, `sample_obs()`, `practice()` (a session dict),
`telemetry_rows(session)`, `track(spec)`, `TRACKS`, `driver_files()`, `default_index()`,
`driver_info(path)` (PARAMS without running the file), `frames(driver, track, mtime)` (one lap
tick by tick), `load_json(path, mtime)`, `class_demo()`, `replay_files()`, `result_files()`,
`rel(path)`.

---

## Common errors and fixes

| You see | Why | Fix |
|---|---|---|
| `No module named streamlit` / `'streamlit' is not recognized ...` (Windows) / `zsh: command not found: streamlit` (macOS) | The venv is not active. | Activate it: `.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS, then `python -m streamlit run ...`. Or without activating: `.venv\Scripts\python -m streamlit run ...` (Windows), `.venv/bin/python -m streamlit run ...` (macOS). |
| `RuntimeError: These lessons need Streamlit >= 1.66, you have 1.50.0` | An old Streamlit in your venv. | `python -m pip install -U "streamlit>=1.66"` (uv: `uv pip install -U "streamlit>=1.66"`), then start the app again. |
| `uv run` starts downloading packages, or creates a `uv.lock` | Plain `uv run` syncs the environment first. | Use `uv run --no-sync ...`, or activate the venv and use `python -m streamlit run ...`. |
| `ModuleNotFoundError: No module named '_data'` | You ran a copy of the lesson from another folder. | Run the file where it is, from the repo root: `python -m streamlit run streamlit_lessons/s0X_....py`. Solutions find it through `solutions/_data.py`. |
| `ModuleNotFoundError: No module named 'agentic_gp'` (or `app`) | `import _data` must come before them, because `_data` puts the repo root on `sys.path`. | Keep `import _data` above the `agentic_gp` / `app` imports. |
| `Port 8501 is already in use` | An old app is still running. | Ctrl+C it in its terminal, or start with `--server.port 8502`. |
| Windows Defender firewall pop-up | Streamlit opens a network port. | Click "Allow" on private networks, or run with `--server.address localhost`. |
| The browser does not open | It is a headless or remote session. | Open http://localhost:8501 yourself. |
| Saving the file does not update the page | You did not start from the repo root, so `.streamlit/config.toml` (`runOnSave = true`) was not read. | Ctrl+C, `cd` to the repo root, run again. You can always click "Rerun" (top right) or press **R** in the page. |
| `StreamlitAPIException: st.session_state.x cannot be modified after the widget with key x is instantiated` | You set a widget's value after drawing the widget. | Set it **before** the widget, or inside an `on_click` callback (S07 / S08 do that). |
| `StreamlitDuplicateElementId` / `DuplicateWidgetID` | Two widgets have the same label. | Give one of them `key="something_unique"`. |
| My button's result disappears on the next click | A button is True for one rerun only. | Store the result: `st.session_state.result = ...` (S06). |
| I edited my driver, but S06 shows the old lap | The cache key did not change. | Include the file's modification time in the arguments (S06 does), or press **C** in the page and choose "Clear caches". |
| `UnhashableParamError` from `@st.cache_data` | An argument cannot be hashed (for example a Track object). | Pass a string or number instead, or name the parameter with a leading underscore (`_track`) so the cache ignores it. |
| The Plotly chart is an empty grey box for a moment | Plotly loads lazily the first time. | Wait a second. Make sure you passed `theme=None` when you use the dark track figures. |
| The canvas inside a tab is blank | A canvas in a hidden tab has zero width. | Click the tab: it draws as soon as it is visible. |
| The arrow keys do nothing in S07 | The keyboard focus is in a text box or in the sidebar. | Click an empty part of the page, then press the arrows. **C** and **R** are Streamlit's own keys. |
| The S08 / S10 animation stutters | The laptop is busy (several apps open, or a running race). | Close the other Streamlit apps. Pause and use Step, or the slider. |
| `UnicodeEncodeError: 'charmap' codec ...` in the Windows terminal | The cp1252 console cannot show a character you printed. | Use `st.write` instead of `print`, or set `PYTHONIOENCODING` before you start: PowerShell `$env:PYTHONIOENCODING="utf-8"`, macOS `export PYTHONIOENCODING=utf-8`. |
| Very slow first start (20 s or more) on Windows | Antivirus scans the venv, or the repo is in OneDrive. | Clone to a short local path (`C:\work\race-mini-comp`). It is faster after the first start. |
| macOS: "python" not found | The venv is not active (macOS itself only has `python3`). | `source .venv/bin/activate` first, or call `.venv/bin/python -m streamlit run ...`. |
| macOS Safari shows a blank page | An old Safari version or a blocked WebSocket. | Use Chrome or Firefox. |

---

## Files

| File | What it is |
|---|---|
| `_data.py` | Cached demo data for every lesson, plus the Streamlit version check and `sys.path` setup. It makes missing data on the fly. |
| `s01_hello_pitwall.py` ... `s10_race_replay.py` | The ten lessons. Each is at most 40 lines (S10 at most 90) of at most 100 characters. The docstring has Concept / Run / You should see / Exercise, and each file has exactly one `# TODO(exercise)`. |
| `solutions/` | One runnable solution per exercise, with the same file names: the lesson plus one `SOLUTION` docstring line and the changed lines, marked `# SOLUTION`. `solutions/_data.py` forwards to `../_data.py`. `solutions/s10_race_replay.py` is the complete app: the solution pages plus "My learning curve". |
| `.cache/` | Created only if `races/demo/` is missing. It holds the generated class demo race (the same 8 cars, Max included, with the mock LLM) and learner session (git-ignored). |
| `.gitattributes` | LF line endings for every file here, so `git diff --no-index` is clean on Windows too. |
| `../tests/test_streamlit_lessons.py` | The conventions above, plus an AppTest of every lesson and solution and each lesson's key interaction. |

Notes:

- The lessons use the mock LLM provider (`LLM_PROVIDER=mock`, set by `_data.py` unless you set
  something else). They list code drivers only, so this block never spends the class LLM budget.
- The canvases (`replay_canvas`, `live_canvas`) and the Plotly figures are shared with the Race
  Control app (`app/`). Their full API is in `app/components/README.md`.
