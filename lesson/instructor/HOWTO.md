# Agentic Grand Prix 2.0 - the instructor's HOWTO

One page to run the course, checked on 2026-10-22 on the course laptop (Windows 11, Python 3.12, a busy
machine), from the repo root. The race-day commands in (d) were re-run end to end in a scratch copy with the
real MiniMax key; the setup, Streamlit and Unity-build commands are the ones the course team verified (the
Unity self-test was re-run). The long versions are in
`lesson/instructor/instructor_guide.md` (setup, key, fly data, mystery track, FAQ) and
`lesson/instructor/RUN_OF_SHOW.md` (both days minute by minute, command cards [C0]-[C14]).

Commands are written as `python ...` with the course venv active. With uv, write `uv run race.py ...`
and `uv run streamlit run ...` instead (same arguments). PowerShell and macOS differ only where shown.

---

## (a) Setup in 5 commands (T-14, once)

Clone to a short local path without spaces, outside OneDrive (e.g. `C:\code\race-mini-comp`).

```
git clone https://github.com/GenomZ/race-mini-comp.git; cd race-mini-comp
uv sync --extra fly                          # Python 3.12 + every pinned package (+ the fly extra)
Copy-Item .env.example .env                  # macOS: cp .env.example .env  -> paste the key after MINIMAX_API_KEY=
uv run race.py doctor --llm                  # ALL GREEN, or OK with N warning(s); one real MiniMax call
uv run python -m pytest -q                   # the whole suite, green (about 10 min on a busy laptop)
```

- `doctor --llm` prints the SDK start-up time and the call time separately (the first `import openai`
  can take 1-2 minutes on a loaded laptop). A "spaces in the path" WARN is harmless.
- `.env` is git-ignored; never put the key in `.env.example`. The key never appears in any output.
- Fly data (Ch7 connectome): `python -m agentic_gp.fly.extract` once (downloads ~130 MB, writes
  `data/fly/steering_circuit_783.npz` + `.json`). Hand both to students for their `data/fly/` (the driver
  loads the `.npz`; the `.json` carries the FlyWire attribution, CC BY-NC 4.0).
- Without uv: `python -m venv .venv`, activate it (`.venv\Scripts\activate` / `source .venv/bin/activate`),
  `pip install -r requirements.txt`.
- Day-2 morning (08:15): `python race.py llm-check` must print `PASS`.

---

## (b) Streamlit: Race Control and the 2-hour tutorial

### Race Control (the course app, keep it running all day in terminal T2)

```
python -m streamlit run app/race_control.py           # http://localhost:8501
```
Start it from the repo root (so `.streamlit/config.toml` applies). Projector: sidebar toggle
"Presenter mode (projector)" or open `http://localhost:8501/?presenter=1`.

| Page | Use it for |
|---|---|
| Home | the ladder poster, Three Clocks, the latest grand prix |
| Live | run any driver and watch it lap live |
| Replay | any replay under `races/`; opens on the newest one you made (the class demo until then) |
| Grand Prix | start a race (the grid defaults to `races/grid/`, the scrutineered files) and show results |
| Sensor X-ray, Physics Lab | Ch1 (Physics Lab has the hand-drive tab) |
| Track Lab, Telemetry | Ch2 |
| Learning Curves | Ch3 (reads `results/`; the "Learning gone wrong" tab uses the Ch3 measurements) |
| Latency Lab | Ch4 (the latency cliff) |
| Pit Wall (LLM trace) | Ch5-Ch6: every prompt, tool call and answer of a driver's LLM log |
| Fly Lab | Ch7 |
| Inbox (instructor) | collect and check uploaded files |

**Mystery track secrecy:** every page hides files with "mystery" in their path (replays, results, LLM logs)
until you switch on the sidebar toggle **"Show Mystery GP files"** (or open the app with `?mystery=1`).
Leave it off until Show 2 (15:50); a grey line says how many files are hidden.

### The 2-hour Streamlit tutorial (Day 1, 13:30-15:30)

Run sheet with minute plan, what to say and what to cut: `streamlit_lessons/README.md` (Part A).
Before the block (lunch): `python -m pytest -q tests/test_streamlit_lessons.py` (green) and
`python -m streamlit version` (1.66 or newer); open S10 once on the projector machine.

```
python -m streamlit run streamlit_lessons/s01_hello_pitwall.py       # 13:35 S01 ... one app per lesson
python -m streamlit run streamlit_lessons/s10_race_replay.py         # 15:12 S10: their multi-car replay viewer
python -m streamlit run streamlit_lessons/solutions/s10_race_replay.py   # any solution: same name under solutions/
```

| Clock | Lesson | Clock | Lesson |
|---|---|---|---|
| 13:30 | opener: the rerun model | 14:24 | stretch |
| 13:35 | S01 hello pit wall | 14:29 | S06 run my driver |
| 13:41 | S02 corner calculator | 14:41 | S07 be the driver |
| 13:50 | S03 track map | 14:53 | S08 live view |
| 14:02 | S04 telemetry | 15:07 | S09 track lab (demo only) |
| 14:14 | S05 driver card | 15:12 | S10 race replay (+ "My learning curve" exercise) |

Behind schedule? Cut S09 first, then S05's exercise, then S08's exercise. Never cut S07, S08 or S10.
Stop Race Control (T2) during the block if two Streamlit servers confuse students; restart it at 15:30.

---

## (c) The Unity 3D viewer (Windows) for the finale

**Build once** (Unity 6, 6000.2.x from Unity Hub; 2-4 min, plus a one-off package import):
```
powershell -ExecutionPolicy Bypass -File unity/build_viewer.ps1
unity/RaceViewer/Build/Windows/RaceViewer.exe --replay unity/RaceViewer/Assets/StreamingAssets/replays/demo.json
```
The exe is `unity/RaceViewer/Build/Windows/RaceViewer.exe` (git-ignored; copy the whole `Build/Windows`
folder to another PC if you build elsewhere). `python race.py doctor` lists it as "Unity 3D viewer".

**Use it** (one window; a second `--unity` hands the new race to the open viewer, which restarts from the grid):
```
python race.py watch races/flycup/flycup.json --unity        # Show 1 warm-up (Fly Cup)
python race.py watch races/final/home.json --unity           # Show 1: Home GP
python race.py watch races/final/mystery.json --unity        # Show 2: Mystery GP
```
Never add `--unity` to `race.py final` (it would push the Mystery GP into the viewer during Show 1).

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | pause / play (pause on the grid for the grid walk) | `C` | next camera (TV, chase, top, orbit, heli) |
| `Home` / `End` | restart from the grid / podium + classification | `1`-`9`, `Tab` | follow position 1-9 / next car |
| `+` / `-` | speed 0.25x-32x | `0` | back to the TV director |
| `Left` / `Right` | -5 s / +5 s (`Shift` = 30 s) | `L` / `F` / `H` / `Esc` | labels / full screen / help / quit |

**Troubleshooting**
- Check a replay without a window: `python unity/tools/viewer_bridge.py races/final/home.json --selftest`
  (exit 0 = fine; prints "self-test PASSED").
- Weak laptop or wrong monitor: `RaceViewer.exe --replay races/final/home.json -screen-width 1280 -screen-height 720 -screen-fullscreen 0`; projector in "Duplicate" mode.
- "Could not load the replay" banner: the text says why; fix the file, the viewer reloads it within 2 s (`R`).
- Pink objects: rebuild with `build_viewer.ps1`. "Another Unity instance": close the editor first.
- Log: `%USERPROFILE%\AppData\LocalLow\Agentic Grand Prix\Agentic Grand Prix\Player.log`.
- No Unity (or macOS): `python race.py watch races/final/home.json --html` (browser player), or Race Control → Replay.

---

## (d) The final Grand Prix with Max on MiniMax (Day 2, 15:15-16:35)

Prerequisites: `.env` with the key, `python race.py llm-check` PASS in the morning, the sealed mystery track
`lesson_private/mystery.py` (+ `mystery_backup.py`) made and self-checked at T-7 (instructor_guide section 5;
`lesson_private/` is git-ignored).

```
# 15:15 [C1] scrutineering: static check, smoke test, key grep, house names, duplicates -> races/grid/
python lesson/instructor/grade.py scrutineer --grid races/grid
# 15:20 [C10] THE FINAL (T1): Home GP, then Mystery GP, then the championship. Max (house) races on MiniMax.
python race.py final races/grid --house --home grand_prix --mystery file:lesson_private/mystery.py --jobs 4 --out races/final --title "Agentic Grand Prix 2026"
# 15:20 [C9] Fly Cup (T3)
python lesson/instructor/grade.py scrutineer --fly --grid races/flycup
python race.py grand-prix races/flycup --practice 0 --no-qualifying --laps 2 --title "Fly Cup" --out races/flycup/flycup.json
# 16:15 [C13] awards (house cars never win)
python lesson/instructor/awards.py --fly-cup races/flycup/flycup_results.json --out races/final/awards.md
```

- **Scrutineering** fails a file for: static errors (forbidden import, syntax, NAME > 16 chars), a smoke test
  that fails or loops (an endless `drive()` loop is caught in ~15 s), an API key in the file, a house car's
  NAME (Max, Planner, Fly Toy), or a NAME already used by a file uploaded earlier (the first upload keeps it).
  Warnings (car still races): AUTHOR still `your_handle`, unchanged handrail copy, LLM calls in `drive()`,
  emoji in NAME, non-hex COLOR, odd file names (`Ann Lee (v3).py` races as `races/grid/ann_lee_v3.py`).
  Exit code 1 = at least one FAIL; the others still race. One-fix window, then run [C1] again.
- **The final** prints a line per car and phase, `Saved: ...home.json ...` when the Home GP is done (start
  Show 1 then), the championship (house cars race but are not scored) and one sha256 per replay for the
  results slide. Output: `races/final/{home,mystery}.json`, `*_results.json`, `championship.md`, LLM logs in
  `races/final/llm_logs/home|mystery/` (Max's log is always `Max.jsonl`).
- **Measured dry run** (2026-10-22, this laptop under load, real MiniMax key, a synthetic class of 22 main
  files): [C1] 35 s (8 FAIL as intended, 14 to the grid); go/no-go 30 s, 14/14 finished; Fly Cup 8 s + 19 s;
  **[C10] 6 min 0 s** (Home GP 165 s, Mystery GP 186 s), 17/17 cars finished with 0 crashes, no loading
  timeouts; Max won both races on MiniMax (134.45 s and 145.10 s); awards 2 s; Unity self-test on
  `mystery.json` PASSED in 17 s. The key appeared in no output file.
- Cars stuck in `DNF (timeout: loading)`? The machine is overloaded: close other programs, keep `--jobs 4`,
  and if needed add `--load-timeout 600`.
- **Plan B, no LLM** (MiniMax down): PowerShell `$env:LLM_PROVIDER="mock"` before the command (macOS: prefix
  `LLM_PROVIDER=mock`). Announce it first; Max and the LLM cars then race on their code fallbacks.
- **Mystery go/no-go** (quiet, before the final, never projected): `$env:LLM_PROVIDER="mock"; python race.py grand-prix races/grid --track-file lesson_private/mystery.py --practice 1 --no-qualifying --laps 1 --quiet --out lesson_private/gonogo/gonogo.json; Remove-Item Env:LLM_PROVIDER`. Go if at least 80 % finish, else use `mystery_backup.py`.
- After class: `python lesson/instructor/grade.py grade` -> `races/grading/grades.md` + `.csv` (optional).
  Rotate the MiniMax key the same evening.

---

## (e) Where everything lives

| Path | What |
|---|---|
| `race.py`, `agentic_gp/` | the CLI (`python race.py --help`) and the package: `sim/` (physics, tracks), `race/` (loader, session, grand prix, replays), `llm.py` (MiniMax + mock + budgets), `driverkit.py`, `fly/`, `viz/` (HTML player, Unity bridge) |
| `drivers/template.py` | every student's starting file |
| `drivers/examples/` | the ladder `l0`-`l6b` (Max = `l6_llm_crew_chief.py`) and the flies (`fly_toy`, `fly_broken`, `fly_connectome`) |
| `drivers/submissions/` | uploads: `<handle>.py`, `<handle>_pilot.py` (Ch4), `<handle>_fly.py` (Ch7); git-ignored except its README |
| `lesson/` | the chapters `00`-`08`, `README.md` (agenda), `cheatsheet.md`, `streamlit_block.md`, `assets/` (figures), `checkpoints/` (handrails `ch0`-`ch7`) |
| `lesson/instructor/` | `instructor_guide.md`, `RUN_OF_SHOW.md`, `CALIBRATION.md` (every measured number), `grade.py`, `awards.py`, `awards.md` |
| `app/race_control.py`, `app/views/` | Race Control (Streamlit) |
| `streamlit_lessons/` | S01-S10 + `solutions/` + `README.md` (run sheet) |
| `unity/` | the 3D viewer project, `build_viewer.ps1`, `tools/viewer_bridge.py`, `README.md` |
| `races/demo/` | committed demo races (`class_demo.json` = mock LLM version), `make_demo.py` |
| `races/grid/`, `races/flycup/`, `races/final/`, `races/grading/` | race-day output (git-ignored) |
| `lesson_private/` | the sealed mystery track (git-ignored) |
| `.env` / `.env.example` | the key (git-ignored) / the template |
| `results/` | `race.py drive --save` results (git-ignored except the shipped reference results) |
| `LICENSE` | MIT (code); the FlyWire data is CC BY-NC 4.0, see `agentic_gp/fly/ATTRIBUTION.md` |
