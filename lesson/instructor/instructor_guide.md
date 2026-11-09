# Instructor guide: Agentic Grand Prix 2.0

You run a two-day course for 15-30 solo students with **one shared MiniMax key**. Every student grows one
driver file through 8 chapters. Rolling snapshots happen during the days and a Day-1 Club Race closes Day 1.
The finale on Day 2 has two races: the Home GP and a Mystery GP on a sealed track. Both are shown in the Unity
3D viewer and in the Streamlit app "Race Control".

This page covers everything you prepare and everything that can go wrong. **[RUN_OF_SHOW.md](RUN_OF_SHOW.md)**
is the minute-by-minute script with the exact commands: print it.

| You need | Where |
|---|---|
| Minute-by-minute script, what to project, every race command | [RUN_OF_SHOW.md](RUN_OF_SHOW.md) |
| Measured lap times, checkpoint bands, latency cliff (all chapter numbers come from here) | [CALIBRATION.md](CALIBRATION.md) |
| Awards and how each is computed | [awards.md](awards.md) (`awards.py`) |
| Scrutineering, the race grid, the optional grade | `grade.py` (section 8 and 9 below) |
| Recording LLM answers for offline demos | [../cassettes/README.md](../cassettes/README.md) |
| What students read | [../README.md](../README.md) (agenda), [../cheatsheet.md](../cheatsheet.md), chapters `../00_pit_lane.md` ... `../08_garage_gp.md` |
| The 2-hour Streamlit block (its own run sheet) | [../../streamlit_lessons/README.md](../../streamlit_lessons/README.md) |
| Driver ladder, Max, measured MiniMax weekends | [../../drivers/README.md](../../drivers/README.md) |
| Unity viewer details | [../../unity/README.md](../../unity/README.md) |
| Fly brain data and licence | [../../agentic_gp/fly/README.md](../../agentic_gp/fly/README.md), [ATTRIBUTION.md](../../agentic_gp/fly/ATTRIBUTION.md) |

**Conventions on this page.** Every command runs from the **repo root** in a terminal where the course venv
is active. Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS. After
that, `python` is the venv's Python. Without activating you can write `uv run race.py ...` instead of
`python race.py ...`. Code blocks work in PowerShell and in a macOS shell unless they are labelled.

---

## 1. The instructor laptop (set up once, T-14)

**Use a Windows 11 laptop if you can.** Everything works on macOS too, except the Unity 3D viewer: it is
built for Windows only. On a Mac, show races in the HTML player or in Race Control instead (section 6).
Plan for 16 GB RAM: the final runs one Python process per driver, 6 at a time.

1. Clone the repo to a short local path with no spaces, outside OneDrive. For example `C:\code\race-mini-comp`
   or `~/code/race-mini-comp`. `race.py doctor` warns about OneDrive, spaces and non-ASCII characters in the path.
2. Install the packages. Python 3.12 is pinned in `.python-version` and every version is pinned in `uv.lock`:
   ```
   uv sync --extra fly          # + numba / pyarrow / pandas for the fly chapter (section 4)
   ```
   Without uv:
   ```
   python -m venv .venv
   .venv\Scripts\activate                      # macOS: source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e ".[fly]"                     # the fly extra (optional)
   ```
3. Put the key in `.env`. **Never** put it in `.env.example`, which is committed.
   ```
   Copy-Item .env.example .env       # PowerShell
   cp .env.example .env              # macOS
   ```
   Then open `.env` and paste the key after `MINIMAX_API_KEY=`, with no quotes.
4. Check everything:
   ```
   python race.py doctor --llm
   ```
   You should see `[OK  ] .env: present, not tracked by git`, `[OK  ] LLM config: minimax:MiniMax-M3, effort
   low, key set` and `[OK  ] LLM call: ... answered in ...`. The doctor reports the SDK start-up time and the
   call time separately. The first call in a fresh process pays for importing the `openai` package.
5. Run the test suite once: `python -m pytest -q`. Before the lesson tests were added, the integration run
   measured 662 tests (8 skipped) in about 160 s on a quiet machine. Expect a few minutes now. The skips are
   the live-LLM tests (opt-in with `AGP_LIVE_LLM=1`) and, without the fly extra, the numba test.
6. Build the Unity viewer once (section 6) and extract the fly data (section 4).

**Race-day terminal layout.**

| Terminal | Use |
|---|---|
| T1 | race commands: `grade.py scrutineer`, `race.py grand-prix`, `race.py final` |
| T2 | `python -m streamlit run app/race_control.py` (keep it running all day) |
| T3 | viewers (`race.py watch ... --unity` / `--html`) and the Fly Cup |

---

## 2. Prep checklist (T-14 to Day 2 08:15)

This is the CURRICULUM checklist rewritten for what the code actually has. Planned items that do not exist
are listed in section 12.

### T-14 days
- [ ] Latest `main` pulled; `python -m pytest -q` green on the instructor laptop. CI (`.github/workflows/tests.yml`)
      runs a pip job and a uv job on Ubuntu, Windows and macOS. The ladder test is `tests/test_drivers_ladder.py`
      and the checkpoint bands are `tests/test_checkpoints.py`.
- [ ] Calibration still true on this laptop: `python lesson/instructor/calibrate.py` takes about 5 minutes with
      the mock LLM. Then open [CALIBRATION.md](CALIBRATION.md): every band in the table must say "yes".
      Code drivers are deterministic, so your numbers must equal the ones in the chapters.
- [ ] **MiniMax key: a dedicated pay-as-you-go key, not a token-plan (M-Plan) key.** Plan keys hit
      usage windows (error **2056**: "usage-window limit reached for this key (5-hour / weekly window)").
      Prepay about $50, set a balance alert, and create a **backup key** that you keep in your password
      manager, not on the laptop.
- [ ] Decide whether the event is **non-commercial**. The FlyWire data may only be used non-commercially
      (section 4). If the event is paid or commercial, run only the toy fly.

### T-7 days
- [ ] Send the setup email: the "Before Day 1" pre-work in [../README.md](../README.md). Students install uv,
      clone, run `uv sync` and `uv run race.py doctor`, and send `doctor_report.txt` by T-2.
- [ ] Mystery track made, sealed and self-checked, plus a gentler backup track (section 5).
- [ ] Fly data extracted and tested (section 4): `python -m pytest -q tests/test_fly_brain.py tests/test_fly_drivers.py`.
- [ ] Confirm that real wiring beats scrambled wiring on this laptop: CALIBRATION.md, CP7 row and "Fly" section.

### T-3 days
- [ ] Real round trips: `python race.py llm-check`. You should see the steps `chat`, `chat_json` and
      `ask_with_tools` as PASS, with seconds and tokens, then `PASS`.
- [ ] Load test of the shared key, with 30 simulated students, 5 tiny calls each, one more student every 2 s:
      ```
      python race.py llm-check --load 30 --calls 5 --ramp 2
      ```
      Write down the `latency p50 / p95` line and the `errors: {...}` line. 2045, 1002 or 429 are rate limits:
      use smaller waves in Ch4 (RUN_OF_SHOW Day 2 09:15), or raise `--ramp` for the snapshot races.
- [ ] Record cassettes for the LLM demos you will show (Ch4 pilot, Ch5 strategist, Ch6 crew chief):
      [../cassettes/README.md](../cassettes/README.md). They are your Plan B if the network or the key fails.

### T-2 days
- [ ] Go through the `doctor_report.txt` files and chase every `FAIL` and every missing report. Typical WARNs:
      OneDrive path, spaces in path, no key yet (expected before Day 2).
- [ ] Optional offline install kit. On one machine per OS, run
      `python -m pip download -r requirements.txt -d wheelhouse`. Install from it with
      `pip install --no-index --find-links wheelhouse -r requirements.txt`. Put it on USB sticks.
- [ ] Print the posters at A2 from `lesson/assets/`: `loop.svg`, `three_clocks.svg`, `driver_frame.svg`
      ("+ = right"), `ladder.svg` and `anatomy_cards.svg`. Optional: `max_architecture.svg` and
      `llm_call_anatomy.svg` for Day 2.

### T-1 evening
- [ ] Demo replays present in `races/demo/` (committed): `class_demo.json` (8 cars, mock LLM), `crash_loop.json`,
      `l2_session_replay.json`, `learner_session_replay.json`, `latency_cliff.json`, `exam_states.json`,
      `llm_gp_oval.json` (real MiniMax). `python races/demo/make_demo.py` regenerates most of them with the
      mock LLM. With the mock LLM, four cars tie in `class_demo` (96.85 s). `python races/demo/make_demo.py
      --real-llm` gives a livelier race. It does **not** regenerate `exam_states.json` or `latency_cliff.json`,
      so keep those two.
- [ ] Opening "Lights out" race recorded: the commands are in RUN_OF_SHOW, "Day 1 09:00".
- [ ] Projector test: `python race.py watch races/demo/class_demo.json --unity`. Then a 30-car stress test:
      ```
      python unity/tools/make_sample_replay.py --cars 30 --out races/sample30.json
      python race.py watch races/sample30.json --unity
      ```
      Both take several minutes the first time: on a machine at 100 % CPU the 30-car replay took 551 s and
      the viewer's first start 267 s (later starts about 40 s). Start them early and do something else.
      HTML fallback: `python race.py watch races/demo/class_demo.json --html`.
- [ ] **Full rehearsal of the final on this laptop.** The handrails make a ready-made fake class with unique
      names, LLM drivers included. The output goes to `lesson_private/` so the mystery replay never shows
      up in Race Control's race lists:
      ```
      python race.py final lesson/checkpoints --house --mystery file:lesson_private/mystery.py --jobs 4 --out lesson_private/t1_rehearsal
      ```
      The dry run of this command on the development machine:
      - with the mock LLM: 8 handrails plus 3 house cars, both GPs, 78 s, every car clean;
      - with the real MiniMax, the final command on 6 students plus 3 house cars:
        - `--jobs 4`: Home 168 s and Mystery 163 s, every car clean;
        - `--jobs 6`: the LLM cars of the Mystery GP timed out while loading.

      That development machine was at 100 % CPU from unrelated jobs, and a cold `import openai` alone took
      about 2 minutes per process there. A second dry run on the same kind of machine had all 21 cars clean
      with `--jobs 6`, so the limit depends on the day's load. The commands in this guide and in
      RUN_OF_SHOW use `--jobs 4`; switch to `--jobs 6` only if this rehearsal and the Day-2 dress rehearsal
      ran clean with it. Write down your own `(wall time ...)` lines.
- [ ] Max: in this rehearsal (real key), Max must finish both races clean. Max is not precomputed: it races live
      in the final, as a house car with the same rules and budget as the students.
- [ ] Go/no-go for the mystery track: in the rehearsal, at least the handrails ch1, ch2, ch3, ch5 and ch6 must
      finish it. Otherwise switch to the backup track.
- [ ] Award certificates and vote forms (Best Agent Design, Best Radio) prepared; [awards.md](awards.md) printed.
- [ ] [RUN_OF_SHOW.md](RUN_OF_SHOW.md) printed. Commands for Day 2 15:15 copied into a text file you can paste from.

### Day 2, 08:15
- [ ] `python race.py llm-check`: PASS.
- [ ] Race Control running on the second screen (section 7).
- [ ] Key hand-out message ready in the LMS (section 3).
- [ ] TA briefed on scrutineering: `python lesson/instructor/grade.py scrutineer --grid races/grid` (section 8).
- [ ] The final command rehearsed on yesterday's rehearsal output (RUN_OF_SHOW, Day 2 15:16).

---

## 3. The shared MiniMax key

**The rules students follow** (they are enforced in code; see `drivers/submissions/README.md`):

| Context | Limit | Enforced by |
|---|---|---|
| Lessons (anything a student runs) | soft cap `LLM_DAILY_CAP` = 250 calls per process (prints `llm calls n/cap`), one request in flight per process, retries with jittered backoff on 429 / 1002 / 2045 / 5xx | `agentic_gp/llm.py` |
| Ch4 pilot (`drive()` calls the LLM) | at most 20 LLM ticks per run | the handrail `ch4_pilot.py` and `l4_llm_rookie.py` |
| Official events (snapshots, Club Race, rehearsal, final) | LLM only in `plan()` and in `reflect()` after a **practice** lap; at most 2 calls in `plan()`, 2 per practice `reflect()`, 8 per driver per track; `LLMNotAllowed` in `drive()`, qualifying and the race; every prompt and answer logged to `<out folder>/llm_logs/<race>/` | the runner's `official` LLM policy (the default for `grand-prix` and `final`) |

**Cost and load, measured.** One Max weekend is 6 calls and about 26k tokens (CALIBRATION.md, real MiniMax).
The handrails use 1-2 calls (ch5) and 4-6 calls (ch6) per weekend. A class of 30 LLM drivers on 2 tracks
therefore makes at most 30 × 8 × 2 = 480 calls in the final, and fewer in practice.

**Hand-out (Day 2, 09:00).** Post the key through the LMS as one line, not in the projected slides and not in a
chat that is recorded:
```
MINIMAX_API_KEY=<the key>
```
Students copy `.env.example` to `.env` and paste the line in (Ch4 has the steps). `.env` is git-ignored, and
`race.py doctor` FAILs if `.env` is tracked by git. In a grand prix every driver runs in its own process,
and the key never enters that process: the runner keeps it and relays the driver's LLM calls through a local
relay on 127.0.0.1 with a per-driver token (`agentic_gp/race/llm_proxy.py`). The race guard
(`agentic_gp/race/guard.py`) also stops the driver process from reading `.env` or any other file outside
Python and `agentic_gp`, from writing files, from starting processes and from opening network connections.

**During the day.**
- Watch the balance in the MiniMax platform console at each break.
- Rate-limit errors (2045, 1002, 429): run Ch4's live exam in waves by rows (RUN_OF_SHOW).
- A student whose key fails sees error 1004 or 2049 (wrong key) or 1008 (no balance) in plain words.
- Scrutineering greps every submission for the key and for anything that looks like `sk-...`
  (`grade.py scrutineer`). It never prints the key.

**If the key leaks** (pasted into a driver file, a chat or a commit):
1. Delete it in the MiniMax platform console.
2. Put the backup key in your `.env`.
3. Post the new line to the class.
4. Check that the leaked text is gone: remove it from the driver file. If it was committed, the key is burned
   whatever you do to the history.

**Rotate after the course (end of Day 2, announced at the retro).** In the MiniMax console, delete the course
key, so every copy in students' `.env` files stops working. Then create a new key for your own use. Check the
final balance and turn off any auto top-up. On the instructor laptop, delete the key line from `.env`.

**Plan B without the key.**
- Offline demos: cassettes (`LLM_PROVIDER=replay`).
- The final: `LLM_PROVIDER=mock`. Every file must still finish with the mock LLM (it is part of
  scrutineering), so this is fair. Announce it before the race.
- Chapter commands keep working with `LLM_PROVIDER=mock`: the LLM drivers then race their code fallback.

---

## 4. Fly brains: data, licence, the broken fly

**Licence first.** The connectome circuit is **derived from FlyWire** and licensed **CC BY-NC 4.0**. It may be
used only for **non-commercial** teaching. The repo's MIT licence covers the code only, not the data. A paid
bootcamp or company training is probably commercial use. In that case, skip Ch7 part B (the connectome): do not
run it, do not show its plots and do not hand out the file. Run only the toy fly (`drivers/examples/fly_toy.py`),
which uses no FlyWire data. The details and the required citations are in
[agentic_gp/fly/ATTRIBUTION.md](../../agentic_gp/fly/ATTRIBUTION.md). It supersedes the CURRICULUM line that
calls the data "CC BY 4.0".

**Extract the circuit (T-7, once).** It needs the `fly` extra (`uv sync --extra fly`, or `pip install pyarrow pandas`):
```
python -m agentic_gp.fly.extract
python -m pytest -q tests/test_fly_brain.py
```
The extractor:
- downloads two pinned inputs (about 132 MB) into `data/flywire/` and checks their sha256;
- writes `data/fly/steering_circuit_783.npz` (1.1 MB) and its `.json` sidecar, which records the recipe,
  the sources, the checksums and the licence;
- takes 10-20 s once the downloads are there. If you already have the inputs, add `--reuse-from <folder>`.

`data/` is git-ignored, so the circuit is never committed.

**Hand it out.** Give students `data/fly/steering_circuit_783.npz` and `.json` together with `ATTRIBUTION.md`.
They go into `data/fly/` in each student's copy, or anywhere if the environment variable `AGENTIC_GP_FLY_CIRCUIT`
points to the `.npz`. Without the file, `fly_connectome.py` says so and the car coasts. The toy fly needs no
data. Optional speed-up for the race laptop: `pip install numba` (included in the fly extra).

**Say the honesty statement out loud** in Ch7. It is the box "Honesty statement" in ATTRIBUTION.md. In short:
- the neurons and synapse counts are real;
- the eyes, the readout gains and the "spinal cord" speed rule are engineered by us;
- what the connectome contributes is shown by the scrambled control.

**The broken fly (Ch7 exercise).** `drivers/examples/fly_broken.py` is the toy with one wrong wiring row. The
answer is not written anywhere students read. To see it, diff the broken file against the toy:
```
git diff --no-index drivers/examples/fly_broken.py drivers/examples/fly_toy.py
```
Fixed means two things:
- the mirror self-check prints "SYMMETRIC": `python drivers/submissions/<handle>_fly.py` (the student's copy of
  `fly_broken.py`, which carries the self-check);
- grand_prix gives exactly 63.30 s with 0 crashes.

The handrail `lesson/checkpoints/ch7_fly_fixed.py` is the fixed file. Fly-class files can be linted with
`python -m agentic_gp.fly.rules drivers/submissions/<handle>_fly.py`.

---

## 5. The mystery track

The Mystery GP is the held-out test set. It must not exist in the repo and must not appear on any screen
before Show 2.

**Make it (T-7).**
1. `mkdir lesson_private`.
2. Keep it out of git, locally, without touching the committed `.gitignore`:
   - PowerShell: `Add-Content .git/info/exclude "lesson_private/"`
   - macOS: `echo "lesson_private/" >> .git/info/exclude`
3. `git status` must not list `lesson_private/`.
4. Save this file as `lesson_private/mystery.py` and change `SECRET`, the zone and the briefing:

```python
"""SEALED mystery track - instructor only. Never commit it, never show it before Show 2.

    python lesson_private/mystery.py                 # self-check: envelope, blind corners, straight, zone
    python race.py final ... --mystery file:lesson_private/mystery.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repo root (for the self-check)

from agentic_gp.sim.geometry import segment_hits  # noqa: E402
from agentic_gp.sim.track import Zone, make_random_track  # noqa: E402

SECRET = "silverstone-at-dawn"       # any text: same text -> same track on every machine
ZONE = (5, 7, 15.0)                  # yellow flag: from waypoint, to waypoint, max speed
BRIEFING = "A car is being recovered after waypoint 5. Expect the yellow flag all weekend."


def make_track():
    tr = make_random_track(SECRET)
    tr.key = "mystery"               # otherwise the replay shows "random:<SECRET>"
    tr.name = "Mystery Circuit"
    tr.zones = [Zone(*ZONE, "Yellow flag")]
    tr.briefing_text = BRIEFING
    return tr


if __name__ == "__main__":
    tr = make_track()
    corners = tr.corners()
    n = tr.n_gates
    blind = [g for g in range(n) if segment_hits(tr.gate_centers[g], tr.gate_centers[(g + 1) % n],
                                                  tr.wall_segments).any()]
    straight = np.isinf(tr.radii) | (tr.radii > 200)
    run, best = 0, 0
    for s in np.concatenate([straight, straight]):
        run = run + 1 if s else 0
        best = max(best, run)
    step = tr.length / len(tr.centerline)
    print(f"{tr.name}: length {tr.length:.0f} (1300-1700), min R {float(np.min(tr.radii)):.1f} (>= 12), "
          f"{n} gates, {len(corners)} corners")
    print(f"blind gate-to-gate chords: {len(blind)} (want >= 2): {blind}")
    print(f"longest straight: about {min(best, len(tr.centerline)) * step:.0f} units")
    print("corners: " + ", ".join(f"{c['id']} R{c['min_radius']:.0f} v{c['v_max']:.0f}" for c in corners))
    print("problems:", tr.validate(min_radius=12) or "none")
    print("briefing:", tr.briefing())
```

**Check it:** `python lesson_private/mystery.py`. With the example `SECRET` it prints:

```
Mystery Circuit: length 1510 (1300-1700), min R 26.0 (>= 12), 16 gates, 7 corners
blind gate-to-gate chords: 5 (want >= 2): [1, 5, 10, 13, 14]
longest straight: about 223 units
corners: T1 R38 v28, T2 R56 v33, T3 R55 v33, T4 R77 v39, T5 R27 v23, T6 R72 v38, T7 R26 v23
problems: none
briefing: Race control briefing - Mystery Circuit. Yellow flag between waypoints 5 and 7: max speed 15. ...
```

The CURRICULUM envelope is:
- length 1300-1700, half-width 9, min R ≥ 12, 16 gates;
- at least 2 blind corners (the chord to a gate crosses a wall);
- one long straight and at least one twisty sequence;
- one yellow-flag zone in the briefing.

Random tracks always satisfy the size rules. Try a few `SECRET` values until you like the corners. To see
the track, plot it (the PNG stays in `lesson_private/`):
```
python -c "import sys; sys.path.insert(0, '.'); from agentic_gp.sim.track import load_track_file; from agentic_gp.viz.mpl import plot_track; plot_track(load_track_file('lesson_private/mystery.py'), 'lesson_private/mystery.png')"
```

**Why a text secret?** Students validate on `random:1`, `random:2` ... Use a text that nobody can guess.
`tr.key` and `tr.name` are overwritten so the secret never appears in a replay. Do not use a `random:<n>` the
chapters mention (7, 12, 21, 101, 2026).

**Backup track.** Make `lesson_private/mystery_backup.py` the same way, with a gentler secret: no corner
under R 30, but still at least 2 blind chords (the self-check asks for them). Only about 1 random secret in 4
qualifies, so search instead of guessing. Save this as `lesson_private/find_secret.py`, put your own text in
`BASE`, and run `python lesson_private/find_secret.py` (about 40 s for 60 secrets):

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the repo root
from agentic_gp.sim.geometry import segment_hits  # noqa: E402
from agentic_gp.sim.track import make_random_track  # noqa: E402

BASE = "change-this-text"            # your own text: nobody may guess the secrets
MIN_R, MIN_BLIND = 30.0, 2           # backup envelope (main track: MIN_R = 12)
for i in range(60):
    tr = make_random_track(f"{BASE}-{i}")
    n = tr.n_gates
    blind = sum(bool(segment_hits(tr.gate_centers[g], tr.gate_centers[(g + 1) % n], tr.wall_segments).any())
                for g in range(n))
    min_r = float(np.min(tr.radii))
    verdict = "PASS" if min_r >= MIN_R and blind >= MIN_BLIND and 1300 <= tr.length <= 1700 else "fail"
    if verdict == "PASS":
        print(f"PASS {BASE}-{i}: min R {min_r:.1f}, blind chords {blind}, length {tr.length:.0f}")
```

Pick a PASS line you like, put that text in `SECRET` of `mystery_backup.py`, and run its self-check. Delete
`find_secret.py` afterwards (it is in `lesson_private/`, so it is never committed). In our test, 16 of 60
secrets passed with `MIN_R = 30` (23 with 25).

**Go/no-go.**
- T-1: the rehearsal in section 2 with the handrails.
- Day 2 lunch: a quiet code-only check of the real class files on the real mystery track. Never project it, and
  keep the output in `lesson_private/` (PowerShell shown; on macOS write `LLM_PROVIDER=mock python race.py ...`):

```
$env:LLM_PROVIDER="mock"; python race.py grand-prix races/grid --track-file lesson_private/mystery.py --practice 1 --no-qualifying --laps 1 --quiet --out lesson_private/gonogo/gonogo.json; Remove-Item Env:LLM_PROVIDER
```

Go if at least 80 % of the cars finish. Otherwise use the backup track in the final command.

---

## 6. The Unity 3D viewer and the projector

**Build once (Windows, T-14).** Install Unity 6 (6000.2.x) from Unity Hub, then:
```
powershell -ExecutionPolicy Bypass -File unity/build_viewer.ps1
unity/RaceViewer/Build/Windows/RaceViewer.exe --replay unity/RaceViewer/Assets/StreamingAssets/replays/demo.json
```
The first command builds `unity/RaceViewer/Build/Windows/RaceViewer.exe`. It takes 2-4 minutes, plus a one-off
package import of several minutes on a clean machine. The second command plays the bundled 12-car demo race;
`Esc` quits. `Build/` is git-ignored: copy the whole `Build/Windows` folder to the race laptop if you build
elsewhere. `python race.py doctor` shows "Unity 3D viewer: <path>" once it is found.

**Show a race:** `python race.py watch races/final/home.json --unity`.
- It copies the replay to the viewer's `latest.json` and starts the viewer, or hands the replay to the open
  window, which restarts from the grid.
- An open viewer re-reads its file every 2 s.
- Do **not** add `--unity` to `race.py final`. It would push the Mystery GP into the viewer while you are still
  showing the Home GP. Use `watch` for each race.

**Keys you will use:**

| Key | Action |
|---|---|
| `Space` | pause on the grid to introduce the line-up |
| `Home` / `End` | restart from the grid / jump to the podium and the classification |
| `C` | change camera |
| `1`-`9` | follow the car in that position |
| `0` | back to the TV director |
| `L` | hide the name labels |
| `F` | full screen on or off |
| `H` | help overlay |

After the winner finishes, stragglers play at up to 32x, so you never wait for the last car. Every finish is
still shown at 1x.

**Projector tips (from unity/README.md):**
- Use "Duplicate" display mode, or `-monitor 2` with the projector as the main display.
- The layout is for 16:9 (1920x1080 or 1280x720).
- On a weak laptop, run it in a window:
  `RaceViewer.exe --replay races/final/home.json -screen-width 1280 -screen-height 720 -screen-fullscreen 0`.
- Check any replay headless, without a window:
  `python unity/tools/viewer_bridge.py races/final/home.json --selftest` (exit 0 = fine).

**Fallbacks, in order:**
1. Unity.
2. The standalone HTML player: `python race.py watch races/final/home.json --html`. It writes `home.html` next to
   the replay and opens it in a browser. This works on macOS too.
3. Race Control, Replay page.

---

## 7. Race Control on the second screen

```
python -m streamlit run app/race_control.py
```
Start it from the repo root, so `.streamlit/config.toml` applies (dark theme, no rerun when `races/` changes).
On the projector, use **presenter mode**: the sidebar toggle "Presenter mode (projector)", or open
`http://localhost:8501/?presenter=1`.

| Page | When |
|---|---|
| Home, Live, Replay | all day: replays of any file under `races/` (the list is the newest first) |
| Grand Prix | snapshots and the races: it lists every grand prix result under `races/` |
| Sensor X-ray, Physics Lab | Ch1 (Physics Lab = the 8-minute hand-drive) |
| Track Lab, Telemetry | Ch2 |
| Learning Curves | Ch3 (reads `results/`) |
| Latency Lab | Ch4 (the latency cliff) |
| Pit Wall (LLM trace) | Ch6 (reads the LLM JSONL logs) |
| Fly Lab | Ch7 |
| Inbox (instructor) | collecting files (section 8) |

- The sidebar "Where am I?" widget tells a student which checkpoint their file has reached. It is static
  (parses, never runs the file).
- Race Control lists **every** JSON under `races/`. Anything about the mystery track must stay out of
  `races/` until Show 2. That is why the rehearsal and the go/no-go write to `lesson_private/`.
- **Mystery exposure during the final batch (15:15-15:50).** Three pages can show the mystery track before
  Show 2, because they all open on the newest file:
  - **Grand Prix** defaults to `mystery_results.json`;
  - **Pit Wall** opens on the newest LLM log, e.g. `final/llm_logs/mystery/<car>.jsonl`, whose first prompt
    contains "Track: Mystery Circuit", its corner table and the briefing;
  - **Replay** lists `final/mystery.json` near the top.

  From 15:15 to 15:50 keep the second screen on the **Home GP results** of the Grand Prix page only (pick
  `final/home_results.json` by hand), or switch it off. Never open Pit Wall or Replay on a projected screen in
  that window.
- **Replay opens on the class demo**, not on the newest race: it defaults to `demo/class_demo.json` (the mock
  version, four cars tied at 96.85 s). Pick the race you want in the "Replay file" box. Students see the same
  default in Ch0.
- **The first start is slow** on a busy laptop: the first render of a page can take 1-5 minutes, and charts
  appear 30-60 s after the text. Wait before you refresh; refreshing starts the work again.
- **The Grand Prix page form** (Start the grand prix) defaults to every main file in `drivers/submissions/`,
  unscrutineered, plus the `_pilot` and `_fly` side files. Use it only after you untick everything and pick
  the files from `races/grid/`, or race from T1 with the command cards instead.
- Windows asks about the firewall the first time Streamlit starts. "Private networks" is enough. Or start it
  with `--server.address localhost` so it is only reachable from your laptop.

---

## 8. Collecting files and scrutineering

**Folder conventions** (also in `drivers/submissions/README.md`):

| File in `drivers/submissions/` | Role | Races in |
|---|---|---|
| `<handle>.py` | main file | snapshots, Club Race, rehearsal, final |
| `<handle>_pilot.py` | Ch4 pilot (LLM every tick) | never raced |
| `<handle>_fly.py` | Ch7 fly | Fly Cup |
| `_anything.py`, `_old/` | ignored | - |

**Never race `drivers/submissions` directly.** `race.py grand-prix` and `final` race **every** `*.py` in a folder,
including the pilot and fly files. Always build the grid first: it is scrutineered and has main files only.

**Collecting.** Pick one channel and say it on Day 1 at 09:00:
- **LMS assignment** (recommended): download all, then copy the files into `drivers/submissions/` as `<handle>.py`.
- **Race Control → Inbox**: drop many files at once. It shows the static check and the smoke test, renames them
  to safe names (`Ada Lovelace (v3).py` → `ada_lovelace_v3.py`), and moves a replaced file to
  `drivers/submissions/_old/`.

Do not open Race Control to the class network just for uploads: every page, including Grand Prix and Inbox,
would be reachable from every student laptop. To lock the Inbox, set `AGP_INSTRUCTOR_PASSWORD` (in `.env` or
the environment) and unlock it with that password. On a hosted server (Railway, or `AGP_HOSTED=1`) the Inbox
stays locked until that variable is set. The other pages are not locked; anyone with the URL can use them, and
real-LLM races spend the key's budget.

The smoke test runs the uploaded file with `race.py check --isolate`: a separate process with the mock LLM,
no API key, and the race guard. The static check is not a sandbox on its own, so the runtime guard is the
backstop. Neither is an OS sandbox: on a public server, keep the key's budget small and the host disposable.

**Scrutineering + grid (one command; smoke tests run 4 at a time):**
```
python lesson/instructor/grade.py scrutineer --grid races/grid
```
For every main file it runs:
- the static check (imports whitelist, interface, metadata, 100 KB);
- `race.py check` in its own process with the mock LLM;
- a grep for the course key and for anything that looks like `sk-...`;
- a duplicate-NAME check (the championship adds points up by NAME).

It prints a table and writes `races/grading/scrutineering.md`. It then empties `races/grid/` and copies the
PASSED files into it. Exit code 1 means at least one file failed. Read its line, fix it with the student
(15:15-15:20 one-fix window on Day 2), and re-run.

**What scrutineering does not catch (yet), and what to do by hand.** Found in a 21-file dry run:

| Case | What happens | What you do |
|---|---|---|
| a student's `NAME` is a house car's ("Max", "Planner", "Fly Toy", any case) | passes; the podium, awards and championship then show two "Max", and the house car's log becomes `Max-2.jsonl` | announce on Day 1: house names are taken. Check the NAME column before every race; rename with the student |
| two files share a `NAME` | **both** files fail, including the student who had it first | find the copier, rename in their file, re-run [C1]; the first student is back on the grid |
| `drive()` loops forever | `race.py check` without `--isolate` hangs; in scrutineering it shows `TIMEOUT` after 180 s, and every re-run waits 3 minutes again while the file is there | move the file to `drivers/submissions/_old/` until it is fixed; tell the student to run `race.py check --isolate` |
| an LLM call inside `drive()` | passes (it is blocked at races), but writes hundreds of identical "LLM calls are not allowed during drive" rows per race into `llm_logs` | tell the student; it costs them nothing in time, only log noise |
| `COLOR = "red"` (not hex) or emoji in `NAME` | the colour is silently replaced by a palette colour; emoji are dropped in Unity and show as boxes in Pit Wall | Day 1 rule: hex colours only (`"#e63946"`), no emoji in `NAME` |
| an LMS file name like `Ann Lee (v3).py` | races, but the grading handle becomes `ann lee (v3)` (evidence screenshots must match) | rename to `<handle>.py` when you collect, or use the Inbox page, which renames |
| a fly file that is an untouched example, or `AUTHOR = "your_handle"` | passes the Fly Cup scrutineering; awards may then name "Fly Toy" (the house car's name) | check NAME and AUTHOR of every `_fly.py` before [C9] |

Measured on the synthetic dry-run class (6 main files, 1 pilot, 1 fly, 1 file with `import requests`):
- with `--all`: 8 PASS and 1 FAIL ("static: line 5: import requests is not allowed - no raw HTTP in a driver")
  in 57 s on the loaded development machine;
- the 7 main files alone: 13 s.

For the Fly Cup: `python lesson/instructor/grade.py scrutineer --fly --grid races/flycup`.

---

## 9. Grading (optional, separate from the race)

```
python lesson/instructor/grade.py grade
```
It reads `drivers/submissions/` (main files, plus pilot and fly files for the process score) and
`races/snapshots/*_results.json` (the snapshot races). It also reads `races/final/home_results.json` and
`mystery_results.json`, `results/*.json` (`race.py drive --save`) and screenshots in
`drivers/submissions/evidence/` named `<handle>...png|jpg|pdf`. For each main file it runs `race.py check`
and `race.py validate --tracks <5 public tracks> --random 0` with the mock LLM. Runs are cached in
`races/grading/runs/` until the file changes. The output is `races/grading/grades.md` (one breakdown per
student) and `grades.csv`.

| Weight | Criterion (CURRICULUM) | What grade.py measures |
|---|---|---|
| 40 | Working driver | `check` PASSED (15); finishes the 5 public tracks with the LLM off (10 × share); clean = 0 crashes (15 × share) |
| 30 | Design | plan() uses the map (8); reflect() keeps memory (5) and keeps the best / locks in (2); a measured improvement in some session (3); LLM inside try/except (3) + fed the briefing or tools (3), **or** no LLM with a filled-in "LLM: none, because ..." card line (6); the 5-line driver card (6) |
| 20 | Process | share of snapshot races the student's car was in (12); a race-engineer screenshot (5); a pilot or fly side file (3) |
| 10 | Result | per final race: at or better than the class median time (2.5) and ahead of Max (2.5) |

- A main file that is a handrail or example with only its metadata changed is flagged and gets 0 design
  points. Students may race it, but it is not their design.
- The static design facts come from `app/lib/checkpoints.py`, which also drives the "Where am I?" widget.
- Correct anything by hand with `--overrides overrides.csv`. The columns are `handle,criterion,points,note`;
  the points replace the computed criterion and the note shows up in the report.
- `--no-run` uses the cached runs only.

`grade.py grade` on the dry-run class took 39 s; the full output is in RUN_OF_SHOW, "Dry run". The totals:

| File | Total | Why |
|---|---|---|
| ch3-based, edited, driver card | 94.5 | |
| ch5-based, edited, driver card | 92.5 | |
| ch2-based, edited, driver card | 72.0 | no reflect() and no result points |
| ch6 / ch1 / ch0 handrails with only the metadata changed | 62.5 / 52.0 / 52.0 | flagged: design 0 |
| the file that failed scrutineering (static check) | 0.0 | |

`grade.py` zeroes the working score only for a **static** failure (forbidden import, syntax error, missing
metadata). A file rejected for another reason, for example a key-like `sk-...` string, still gets its full
working points (40/40 in the dry run) although it raced nowhere. Check `races/grading/scrutineering.md` and
correct such rows with `--overrides`.

---

## 10. Troubleshooting FAQ

**Setup (Windows)**

| Symptom | Fix |
|---|---|
| `uv` is not recognised | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"`, then open a new terminal |
| `python` opens the Microsoft Store | use the venv (`.venv\Scripts\activate`) or `uv run`; or turn off Settings → Apps → Advanced app settings → App execution aliases → python |
| `.venv\Scripts\activate` "running scripts is disabled" | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or call `.venv\Scripts\python.exe race.py ...` directly |
| doctor WARN "inside OneDrive" or "spaces in the path" | move the clone, e.g. to `C:\code\race-mini-comp` (OneDrive locks files while it syncs) |
| `LLM_PROVIDER=mock python ...` fails in PowerShell | PowerShell syntax: `$env:LLM_PROVIDER="mock"; python race.py ...`, and `Remove-Item Env:LLM_PROVIDER` afterwards |
| garbled characters in the console | `race.py` switches its own output to UTF-8; for other scripts run `$env:PYTHONIOENCODING="utf-8"` first |
| `git` warns about LF/CRLF | harmless |

**Setup (macOS)**

| Symptom | Fix |
|---|---|
| `uv` not found | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then open a new terminal |
| `python: command not found` | activate the venv (`source .venv/bin/activate`), or use `uv run race.py ...` / `python3` |
| no Unity viewer | expected: the viewer is built for Windows. Use `race.py watch FILE --html` or Race Control |
| Streamlit app looks different from the screenshots | the figures were made with Segoe UI; macOS uses a fallback font |

**Streamlit**

| Symptom | Fix |
|---|---|
| "Port 8501 is already in use" / the app opens on 8502 | an old app is still running: Ctrl+C in its terminal; one app at a time |
| "These lessons need Streamlit >= 1.66" | `uv sync` (or `pip install -U "streamlit>=1.66"`); `python -m streamlit version` |
| a student's page does not update | the file must be saved; `runOnSave` only works when started from the repo root |
| lessons reset | `git checkout -- streamlit_lessons/<file>.py`; solutions are in `streamlit_lessons/solutions/` |

**The LLM**

| Symptom | Fix |
|---|---|
| error 1004 or 2049 | wrong key in `.env` (spaces, quotes, an old key): paste the line again |
| error 1008 | the account balance is empty: top it up (backup key meanwhile) |
| error 2056 | a token-plan key hit its usage window: use the pay-as-you-go key |
| 2045 / 1002 / 429 retries, slow answers | rate limit: smaller waves; `llm-check --load` tells you the safe size |
| `LLMNotAllowed` | expected at official events: LLM in `drive()`, in qualifying or in the race; the driver must fall back to code |
| `LLMBudgetExceeded` | the per-phase budget is used up (2 in plan, 2 per practice reflect, 8 per track) or the soft daily cap of 250 |
| the very first call takes long | a cold `import openai`; `doctor --llm` shows "SDK start-up" separately from the call |

**Races**

| Symptom | Fix |
|---|---|
| `ERROR: no driver files found` | the grid folder is empty: run `grade.py scrutineer --grid races/grid` first |
| a car is `DNF (timeout: loading)` "still importing the LLM SDK after 600s" | the machine is overloaded: close Unity, browsers and other Python processes, use `--jobs 4`, and re-run (a documented infrastructure fault). In the real-key dry run on a machine at 100 % CPU from unrelated jobs, all three LLM cars of the Mystery GP hit this limit with `--jobs 6`; with `--jobs 4` every car finished |
| `DNF (stalled)` | `drive()` was too slow (10 stalls in a row or 30 in total); that is the driver's fault |
| `DNF (timeout)` in plan/reflect | over 120 s per call or 300 s of thinking per driver per track |
| `DNF (max ticks)` | the car did not finish within 500 ticks per lap (crash loop); the marshal rule usually prevents it |
| a pilot or fly file shows up in the race | you raced `drivers/submissions` directly; race `races/grid` |
| the Unity viewer shows an old race | `python race.py watch FILE --unity`; red banner: you gave a `*_results.json` instead of the replay JSON |
| `CircuitNotFoundError` (fly) | the circuit file is missing: section 4 |

---

## 11. Cut list (when the class is more than 15 minutes behind)

Apply in this order (CURRICULUM section 3):
1. S09 Track Lab becomes homework. The Streamlit run sheet already schedules it as a 5-minute demo.
2. Ch2 Gold (racing line) becomes homework.
3. Ch4 Silver is dropped.
4. The Ch6 LangGraph variant (`drivers/examples/l6b_pit_crew_langgraph.py`) becomes reading only.
5. The Ch7 connectome becomes an instructor demo. Keep the scrambled control: run `fly_connectome.py` with
   `--param SCRAMBLE=0` next to the real wiring.
6. Unity is skipped: show the races in the HTML player (`race.py watch FILE --html`).
7. The Day-1 Club Race drops to 1 lap (`--laps 1`).

Never cut the snapshots, the dress rehearsal or the LLM-off check in Ch8. They are what makes the final work.

---

## 12. What the CURRICULUM planned that does not exist (and what to do instead)

| Planned | Reality |
|---|---|
| `race.py collect` | not a command: LMS / Race Control Inbox + `grade.py scrutineer --grid races/grid` |
| Max precomputed the evening before and merged into the final replay; sealed `house_config.json` | Max races live in the final (`--house`), same rules and budget; its knobs are in `drivers/examples/l6_llm_crew_chief.py` |
| `results/snapshots/snap_<n>.json` | a snapshot is a `race.py grand-prix` into `races/snapshots/` (RUN_OF_SHOW) |
| a "Validation column" | a second snapshot race on a `random:<seed>` track, in `races/snapshots/validation/` |
| a Devcontainer / Codespaces fallback | not set up; fallbacks: the USB wheelhouse, or pair with a neighbour |
| a global LLM semaphore with a priority tier | the runner caps concurrent LLM calls at 8 per grand prix and starts one more driver every `--ramp` seconds (default 2) |
| a per-student anatomy radar at the retro | use the championship table (Home vs Mystery positions: who generalised) and the Learning Curves page |
