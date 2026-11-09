# Run of show: Agentic Grand Prix 2.0 (2 days)

Print this page. It is the minute-by-minute script: what happens, what is on the **projector** (screen 1)
and on the **second screen** (Race Control), and the exact command for every race. Background, setup and
troubleshooting are in [instructor_guide.md](instructor_guide.md). The numbers students should see are in
[CALIBRATION.md](CALIBRATION.md).

**Every command** runs from the repo root on the instructor laptop, in a terminal with the venv active
(`.venv\Scripts\activate`, macOS `source .venv/bin/activate`). The terminals are:
- **T1**: races;
- **T2**: `python -m streamlit run app/race_control.py`, running all day;
- **T3**: viewers and the Fly Cup.

The command cards (**[C1]**, **[C2]**, ...) are at the bottom of this page and are referenced from the schedules.
The chapter rows below count teaching time only; the lesson README's agenda gives each chapter its upload and
snapshot slot too (Ch1 65 min = 55 + upload + snapshot), so both tables end at the same clock times.

**Where files live on race days**

| Path | What | In git? |
|---|---|---|
| `drivers/submissions/` | what students hand in (`<handle>.py`, `_pilot.py`, `_fly.py`) | do not commit |
| `races/grid/`, `races/flycup/` | the scrutineered grid, rebuilt by [C1] / [C9] | ignored |
| `races/snapshots/` | Snapshot races (the Grand Prix page lists them) | ignored |
| `races/club/`, `races/rehearsal/`, `races/final/` | Club Race, dress rehearsal, the final | ignored |
| `lesson_private/` | the sealed mystery track + anything raced on it before Show 2 | excluded locally |

Demo files in `races/demo/` are committed and always there.

---

## Day 1: "From reflexes to plans"

| Time | Min | Block | Projector (screen 1) | Second screen / you |
|---|---|---|---|---|
| 08:30 | 30 | **Pit desk** (optional): TA fixes red doctors; offline install from the USB wheelhouse (instructor_guide section 2) | the doctor command, big: `uv run race.py doctor` | T2 started; Race Control on screen 2 in presenter mode |
| 09:00 | 3 | **Lights out**: welcome | Three Clocks poster (`lesson/assets/three_clocks.svg`) | |
| 09:03 | 7 | Ghost race of the house grid on the home track, paused on the grid: "Which one is smartest?" Then play it and reveal the times | Unity: `python race.py watch races/show/lights_out.json --unity` ([C0], made T-1) | Race Control → Grand Prix → "Lights out" (standings) |
| 09:10 | 4 | Same field on a track nobody tuned for. Then the overfitting matrix: the "Memorizer", the fastest reflex at home, crashes on 5 of 6 other tracks | `python race.py watch races/show/lights_out_random7.json --unity`, then `lesson/assets/overfitting_matrix.png` | |
| 09:14 | 6 | Rules: one file, classes (Open / LLM / Fly), snapshots at 10:45 / 12:20 / 16:30, Club Race 16:35, Day-2 final = Home GP + **sealed** Mystery GP. Upload channel. Sticky-note rule (red = stuck for 2 minutes) | `lesson/assets/ladder.svg`, `lesson/assets/loop.svg` | write the upload channel on the board |
| 09:20 | 25 | **Ch0 Pit Lane** ([00_pit_lane.md](../00_pit_lane.md)) | L0 crash replay: Race Control → Replay → `races/demo/crash_loop.json` | walk the room: `race.py check` green? |
| 09:45 | 55 | **Ch1 Reflexes** ([01_reflexes.md](../01_reflexes.md)), incl. **8-min Physics Lab hand-drive** around 10:00 | Race Control → Physics Lab, then Sensor X-ray; `lesson/assets/driver_frame.svg` ("+ = right"), `turn_radius.png`, `stopping_distance.png` | stuck students: handrail `lesson/checkpoints/ch1_reflex.py` |
| 10:40 | 5 | "Upload your file now" | upload channel on screen | collect into `drivers/submissions/` |
| 10:45 | 5 | **Snapshot #1** (CP1): T1 runs [C1] then [C2] with `N=1`. Start the home race at 10:45 sharp and the unseen-track race in T3 at the same time; with the real key the pair took 7-11 min in the dry run (Max is always last to finish), so the results land during the break | | |
| 10:50 | 15 | **Break**, Snapshot #1 on the projector | Race Control → Grand Prix → "Snapshot 1" (home) and "Snapshot 1 - unseen track"; Max's line is the target | |
| 11:05 | 70 | **Ch2 The Map** ([02_the_map.md](../02_the_map.md)) | `overfitting_matrix.png` first; `vmax_grand_prix.png`, `speed_profile.png`; Race Control → Track Lab | handrail `ch2_planner.py` |
| 12:15 | 5 | Upload | | |
| 12:20 | 10 | **Snapshot #2** (CP2): [C1], [C2] `N=2`. The unseen-track column is now the one to watch: "the map works on tracks it has never seen" | Grand Prix page: Snapshot 2 home vs unseen track | |
| 12:30 | 60 | **Lunch** | Snapshot #2 standings stay up | eat |
| 13:30 | 120 | **Streamlit block S01-S10**: follow Part A of [streamlit_lessons/README.md](../../streamlit_lessons/README.md) (its own minute plan; stretch at 14:24; S09 is a 5-min demo) | the lesson apps (`python -m streamlit run streamlit_lessons/s01_hello_pitwall.py` ...) | stop T2 (Race Control) during the block if ports get confusing; restart at 15:30 |
| 15:30 | 15 | **Break** | | restart T2 |
| 15:45 | 40 | **Ch3 Practice Makes Perfect** ([03_practice.md](../03_practice.md)) | `learning_curve.png`, `learning_gone_wrong.png`; Race Control → Learning Curves | handrail `ch3_learner.py` |
| 16:25 | 5 | Upload | | |
| 16:30 | 5 | **Snapshot #3** (CP3): [C1], [C2] `N=3` | | |
| 16:35 | 25 | **Day-1 Club Race**: [C3] (about 3 min). Share `races/club/club_race.json` through the upload channel; students open it in their own S10 viewer. You name 3 patterns you saw (e.g. "braked for a wall that wasn't there", "fast but crashed at T5", "never learned"). Reveal Max's home time as "the number to beat". **Exit ticket**: "which clock does each part of your driver live on?" | Unity: `python race.py watch races/club/club_race.json --unity` | Race Control → Grand Prix → "Day-1 Club Race" |
| 17:00 | | End of Day 1 | | back up `races/` (copy to a USB stick) |

---

## Day 2: "Brains that think"

| Time | Min | Block | Projector (screen 1) | Second screen / you |
|---|---|---|---|---|
| 08:15 | | Day-2 prep checklist (instructor_guide section 2): `python race.py llm-check` PASS | | |
| 09:00 | 15 | Overnight board (Snapshot #3 + Club Race). **Key hand-out** through the LMS (instructor_guide section 3). Everybody: copy `.env.example` to `.env`, paste the line, `python race.py doctor --llm` | `max_architecture.svg`; one budget slide: "2 calls in plan(), 2 per practice reflect(), 8 per track; never in drive() or the race" | watch for 1004/2049 (wrong key) |
| 09:15 | 55 | **Ch4 Talking to MiniMax** ([04_minimax.md](../04_minimax.md)). The live exam runs in **waves by rows**: rows A 09:35-09:40, B 09:40-09:45, C 09:45-09:50 (only that row presses Enter) | `llm_call_anatomy.svg`; **your demo** [C4]: the pilot driving with the LLM (cassette first, then one live run); `latency_cliff.png` + Race Control → Latency Lab | balance check in the MiniMax console |
| 10:10 | 15 | **Break** | | |
| 10:25 | 60 | **Ch5 The Strategist** ([05_strategist.md](../05_strategist.md)) | **your demo** [C5]: A/B race, strategist vs learner on `briefing_demo` | handrail `ch5_strategist.py` |
| 11:25 | 5 | Upload | | |
| 11:30 | 5 | **Snapshot #4** (CP5): [C1], [C2] `N=4`. LLM drivers now call MiniMax in `plan()` and practice `reflect()` | Grand Prix page: Snapshot 4 (class column) | LLM calls appear in the "LLM" column |
| 11:35 | 60 | **Ch6 The Crew Chief** ([06_crew_chief.md](../06_crew_chief.md)) | **your demo** [C6]: Max on an unseen track, then Race Control → Pit Wall (LLM trace) | handrail `ch6_crew_chief.py` |
| 12:35 | 55 | **Lunch.** You: the **dress rehearsal** [C7] with every CP6 upload (official LLM rules, real concurrency, stand-in mystery `random:2026`). Write down both `(wall time ...)` lines: that is your final's duration. Then the quiet mystery **go/no-go** [C8] (never projected) | | if the rehearsal shows LLM cars with `DNF (timeout: loading)`: close every other program and use `--jobs 4` in the final |
| 13:30 | 10 | Dress-rehearsal replay: who generalised (Home vs `random:2026` positions), who overfit. **Snapshot #5** = the rehearsal Home GP vs Max | `python race.py watch races/rehearsal/home.json --unity`; `races/rehearsal/championship.md` | |
| 13:40 | 50 | **Ch7 Fly Brains** ([07_fly_brains.md](../07_fly_brains.md)): toy fly + the broken fly (25), connectome + scrambled control (20), discussion (5) | Race Control → Fly Lab; `fly_circuit.svg`, `fly_tuning_curve.png`, `fly_real_vs_scrambled.png`; say the honesty statement (instructor_guide section 4) | connectome needs `data/fly/` (or demo it yourself, cut list item 5) |
| 14:30 | 45 | **Ch8 Garage** ([08_garage_gp.md](../08_garage_gp.md)): validate, LLM-off test, driver card, upload. **15:10: last LLM call** (quiet zone until the end of the show). **15:15: hard lock** | countdown on screen; the three Ch8 commands (`check`, `validate --random 3`, mock `drive`) | collect files; TA reads the scrutineering table |
| 15:15 | 5 | **Scrutineering**: T1 runs [C1]. Failures get the 15:15-15:20 one-fix window, then [C1] again; still failing = races in a later exhibition heat. A file whose `drive()` never returns makes every [C1] wait 3 min: move it to `drivers/submissions/_old/` first | `races/grading/scrutineering.md` (TA screen, not the projector) | students: break + vote on design cards (forms) |
| 15:20 | 15 | **The final batch** [C10] in T1 (Home GP, then Mystery GP; pipelined with the show). T3: Fly Cup [C9] | nothing race-related yet: second screen on the Home GP results only (Grand Prix page, `final/home_results.json` picked by hand) or off. Pit Wall, Replay and the Grand Prix default all show the mystery track (instructor_guide section 7) | T1 prints `Saved: races\final\home.json ...` when the Home GP is done |
| 15:35 | 15 | **Show 1: Home GP.** Fly Cup first as the warm-up during the grid walk; then the grid walk (name + motto per car), qualifying, the 3-lap race | [C11] Fly Cup, then Home GP in Unity: `Space` on the grid, `1`-`9` to follow cars, `End` for the classification | T1 is still computing the Mystery GP |
| 15:50 | 25 | **Show 2: Mystery GP.** Track reveal (`lesson_private/mystery.png`), read the briefing aloud (the `BRIEFING` line), practice learning curves, qualifying, 3-lap race with radio | [C12] | write the two sha256 lines from T1 on the results slide |
| 16:15 | 20 | **Championship + awards** [C13] | `races/final/championship.md`, then `races/final/awards.md`; Unity `End` for the podium | certificates; jury awards from the votes |
| 16:35 | 25 | **Retro**: Home vs Mystery positions (who generalised), "where the LLM belonged" (Pit Wall page on the final's logs), learning curves. **Announce the key rotation**: "the key stops working tonight". Feedback form | `races/final/championship.md` | |
| 17:00 | | End | | after class: [C14] |

---

## Command cards

**[C0] Lights-out race (record T-1 evening, show Day 1 09:03)**. The house grid is L0, L1, L2 (Planner, house),
L3, L5, Max (house) and the toy fly (house). Mock dry run: 34 s and 28 s. With the real key: 295 s and 220 s on a
loaded machine (one Max `reflect` call timed out after 115 s and Max finished on its code fallback), so record
it early on T-1.
```
python race.py grand-prix drivers/examples/l0_dummy.py drivers/examples/l1_reflex.py drivers/examples/l3_learner.py drivers/examples/l5_llm_strategist.py --house --practice 2 --laps 1 --title "Lights out" --out races/show/lights_out.json
python race.py grand-prix drivers/examples/l0_dummy.py drivers/examples/l1_reflex.py drivers/examples/l3_learner.py drivers/examples/l5_llm_strategist.py --house --practice 2 --laps 1 --track random:7 --title "Lights out - unseen track" --out races/show/lights_out_random7.json
```

**[C1] Scrutineer + build the grid (before every race)**
```
python lesson/instructor/grade.py scrutineer --grid races/grid
```
- It prints one row per file: `PASS` / `FAIL`, NAME, AUTHOR, KIND, smoke test, problems.
- It writes `races/grading/scrutineering.md`.
- It copies only the PASSED **main** files into `races/grid/`; pilot and fly files never race there.
- Exit code 1 = at least one FAIL. The others still race; fix the failures with the student.

**[C2] Snapshot N** (N = 1, 2, 3 on Day 1; 4 on Day 2). The format is `plan` → 2 practice laps with `reflect` →
1 timed lap, on the home track with Max as the reference, plus the same field on an unseen track (the
"Validation column"). Keep `random:101` for every snapshot so the unseen-track column is comparable.
```
python race.py grand-prix races/grid --house --practice 2 --no-qualifying --laps 1 --title "Snapshot 1" --out races/snapshots/snap1.json
python race.py grand-prix races/grid --practice 2 --no-qualifying --laps 1 --track random:101 --title "Snapshot 1 - unseen track" --out races/snapshots/validation/snap1_random101.json
```
Run the second line in T3 at the same time as the first. Real-key timings from the 21-file dry run on a loaded
machine: Snapshot 1 home 425 s + unseen 267 s, Snapshot 2 386 s + 159 s. The code cars were done after
70-190 s; Max (house) was always last (347-391 s: SDK start-up plus 4 real calls). If that is too slow for the
5-minute slot, leave `--house` out of the Day-1 snapshots and quote Max's Lights-out time as the target.
Change `1` to N in the title and in both file names. Show it with Race Control → Grand Prix (it lists every result
under `races/`), or `python race.py watch races/snapshots/snap1.json --html`. `snap1_results.json` is also
the input for the "Biggest Improvement" award.

**[C3] Day-1 Club Race** (code drivers + the house cars; 1 practice, qualifying, a 2-lap race on the home track)
```
python lesson/instructor/grade.py scrutineer --grid races/grid
python race.py grand-prix races/grid --house --practice 1 --laps 2 --title "Day-1 Club Race" --out races/club/club_race.json
python race.py watch races/club/club_race.json --unity
```
Cut-list item 7: `--laps 1`. Students: save `club_race.json` anywhere under their own `races/` folder, run
`python -m streamlit run streamlit_lessons/s10_race_replay.py` and pick it in "Replay file".

**[C4] Ch4 demo: the LLM drives every tick** (20 LLM calls per run on the shared key)
```
python race.py drive lesson/checkpoints/ch4_pilot.py
python race.py drive lesson/checkpoints/ch4_pilot.py --realtime
```
- Run the first one from the cassette if the network is shaky ([../cassettes/README.md](../cassettes/README.md)).
- Run the second one live: the world does not wait for the LLM.
- Measured with MiniMax: 20 calls, p50 1.04 s / p95 1.45 s, lap 65.75 s (turn-based, CALIBRATION.md).

**[C5] Ch5 demo: strategist vs learner on the yellow-flag track**
```
python race.py grand-prix lesson/checkpoints/ch5_strategist.py lesson/checkpoints/ch3_learner.py --track briefing_demo --practice 0 --laps 1 --title "Ch5 A/B" --out races/show/ch5_ab.json
python race.py watch races/show/ch5_ab.json --unity
```
- With MiniMax the strategist reads the briefing: briefing_demo 61.15 s with 0 zone violations, 1 call.
- The learner collects zone penalties.
- The yellow hatching shows in Unity.

**[C6] Ch6 demo: Max debriefs on an unseen track, with the trace for the Pit Wall page**
```
$env:LLM_LOG_DIR = "races/show/llm_logs"            # macOS: export LLM_LOG_DIR=races/show/llm_logs
python race.py drive drivers/examples/l6_llm_crew_chief.py --track random:7 --attempts 4 --replay races/show/max_random7.json
Remove-Item Env:LLM_LOG_DIR                          # macOS: unset LLM_LOG_DIR
```
Then open Race Control → Pit Wall and pick `races/show/llm_logs/Max.jsonl` (if a student car is also named
"Max", the house car's log is `Max-2.jsonl`: one more reason to keep house names out of the class). Measured with MiniMax on random:7:
practice 45.05 → 43.35 → 43.30 s (CALIBRATION.md).

**[C7] Dress rehearsal** (Day 2 lunch). This is exactly the final command with a stand-in mystery track:
```
python lesson/instructor/grade.py scrutineer --grid races/grid
python race.py final races/grid --house --home grand_prix --mystery random:2026 --jobs 4 --out races/rehearsal --title "Dress rehearsal"
```
Write down the two `(wall time ...)` lines and the LLM column. `--jobs 4` is the safe default; use `--jobs 6` in
the final only if this rehearsal also ran clean with `--jobs 6` (one real-key dry run with 21 cars was clean at 6,
another lost every LLM car to `DNF (timeout: loading)`). On the development machine, the mock dry run
for 6 students + 3 house cars took 64-76 s. With the real key, the Home GP alone took 318 s, because the machine
was at 100 % CPU from unrelated jobs and the SDK import alone took minutes. Your laptop's numbers are the ones
that count.

**[C8] Mystery go/no-go** (quiet; never on the projector; output stays in `lesson_private/`)
```
$env:LLM_PROVIDER="mock"; python race.py grand-prix races/grid --track-file lesson_private/mystery.py --practice 1 --no-qualifying --laps 1 --quiet --out lesson_private/gonogo/gonogo.json; Remove-Item Env:LLM_PROVIDER
```
On macOS: `LLM_PROVIDER=mock python race.py grand-prix ...` with the same arguments. Go if at least 80 % finish;
otherwise use `file:lesson_private/mystery_backup.py` in [C10].

**[C9] Fly Cup** (T3, 15:20; 2 laps on the home track, no practice, no qualifying)
```
python lesson/instructor/grade.py scrutineer --fly --grid races/flycup
python race.py grand-prix races/flycup --practice 0 --no-qualifying --laps 2 --title "Fly Cup" --out races/flycup/flycup.json
```
For "Best Fly Brain", the winner must beat its own scrambled control. Run that check on the winner only:
`python race.py drive races/flycup/<handle>_fly.py --param SCRAMBLE=0`. It must be slower or DNF.

**[C10] THE FINAL** (T1, 15:20, after [C1])
```
python lesson/instructor/grade.py scrutineer --grid races/grid
python race.py final races/grid --house --home grand_prix --mystery file:lesson_private/mystery.py --jobs 4 --out races/final --title "Agentic Grand Prix 2026"
```
It runs the Home GP, then the Mystery GP:
- Home GP: 2 practice laps, qualifying, a 3-lap race.
- Mystery GP: 3 practice laps, qualifying, a 3-lap race.

Official LLM rules apply, and LLM logs go to `races/final/llm_logs/home/` and `.../mystery/`. At the end it
prints the championship and one sha256 per replay: write them on the results slide. **One official run.** Re-run
only after a documented infrastructure fault (for example `DNF (timeout: loading)` for every LLM car), and say so.

Do not add `--unity` here: it would switch the open viewer to the Mystery GP during Show 1.

Plan B, no LLM: `$env:LLM_PROVIDER="mock"` before the command (macOS: prefix `LLM_PROVIDER=mock`).
Announce it first.

**[C11] Show 1** (T3)
```
python race.py watch races/flycup/flycup.json --unity
python race.py watch races/final/home.json --unity
```
- Run the second command once T1 has printed `Saved: races\final\home.json`.
- The qualifying lap is a replay of its own: `races/final/home_qualifying.json`.
- No Unity? Use `python race.py watch races/final/home.json --html`.

**[C12] Show 2** (T3)
```
python race.py watch races/final/mystery_qualifying.json --unity
python race.py watch races/final/mystery.json --unity
```
- Practice times per driver: Race Control → Grand Prix → the Mystery GP result (qualifying table: best practice
  vs qualifying lap).
- Radio messages scroll in the Unity ticker.

**[C13] Awards** (see [awards.md](awards.md))
```
python lesson/instructor/awards.py --snapshot1 races/snapshots/snap1_results.json --fly-cup races/flycup/flycup_results.json --out races/final/awards.md
```
Fill in the three jury lines (Most Creative LLM Use, Best Agent Design, Best Radio) by hand. Check the LLM-class
winner in `races/final/llm_logs/` and the fly winner's scrambled control ([C9]) before you announce them.

**[C14] After the course**
```
python lesson/instructor/grade.py grade
```
- This is the optional grade: `races/grading/grades.md` and `grades.csv`. Review the flags; corrections go
  in `--overrides overrides.csv`.
- **Rotate the key** in the MiniMax console (instructor_guide section 3).
- Archive `races/` (final, snapshots, llm_logs) and `drivers/submissions/` outside the repo.

---

## Dry run: every command on this page, measured

Every command card was run in this order on the development machine. The class was synthetic:
- 6 main files built from the handrails (ch0, ch1, ch2, ch3, ch5, ch6) with student names;
- 1 Ch4 pilot, 1 fly file;
- 1 file with a forbidden `import requests`.

The machine was at 100 % CPU from unrelated jobs the whole time, so a quiet laptop will be faster. "Cars"
counts the house cars (Max, Planner, Fly Toy). Every step exited 0 except the scrutineering, which exits 1 by
design because of the bad file.

| Card | Cars | Mock LLM | Notes |
|---|---|---|---|
| [C0] Lights out, home / random:7 | 7 | 34 s / 28 s | |
| T-1 rehearsal (`final lesson/checkpoints ...`) | 11 | 78 s | all 11 clean on both tracks |
| [C1] scrutineer + grid | 7 files | 13 s (3 s cached) | 6 PASS, `gus.py` FAIL: "import requests is not allowed" |
| [C2] Snapshot, home / unseen | 9 / 6 | 34 s / 27 s | |
| [C3] Club Race | 9 | 35 s | |
| [C7] dress rehearsal | 9 | 64 s | |
| [C8] go/no-go | 6 | 18 s | 6/6 finish: go |
| [C9] Fly Cup (scrutineer + race) | 1 | 8 s + 11 s | `--practice 0` works |
| **[C10] the final** | 9 | **58 s** (Home 28 s, Mystery 27 s) | two sha256 lines printed |
| `watch races/final/home.json --html --no-open` | | 5 s | writes `races/final/home.html` |
| `python unity/tools/viewer_bridge.py races/final/home.json --selftest` | | 18 s | "self-test PASSED" |
| [C13] awards | | 4 s | |
| [C14] grade | 7 files | 39 s | check + validate (5 tracks, mock) per file, then cached |

**With the real MiniMax** (same class, same final command):

| Run | Home GP | Mystery GP | LLM calls (Home / Mystery) |
|---|---|---|---|
| `--jobs 6` | 318 s; all finished | **all three LLM cars `DNF (timeout: loading)`**: "still importing the LLM SDK after 600s" | Eve 4, Dee 2 / - |
| `--jobs 4` (re-run) | 168 s; all finished, 0 crashes | 163 s; all finished, 0 crashes | Max 4 / 6, Eve 4 / 6, Dee 2 / 2 |

On that machine the 6 code cars were done after about 40 s in every run. The LLM cars spent 2-4 minutes
just importing the `openai` package, then raced. A bare `python -c "import openai"` took 2 min 2 s there.

The `--jobs 4` re-run:
- Home GP: Max (house) won in 126.60 s. Cyclone (the ch3 learner) was the best student in 152.10 s.
- Mystery GP: Cyclone won in 157.35 s, ahead of Eve (ch6, 158.40 s), Dee (ch5, 159.00 s) and Max
  (159.00 s, P4). That gave three "Beat the House" badges.
- `awards.py` on that run: Champion Cyclone (12 points), Cleanest Driver Eve Crew, LLM Class winner Eve Crew,
  Most Efficient Dee Strat (2 calls).

This is exactly the failure the dress rehearsal exists to catch on your laptop: close everything else during
the batch, and use `--jobs 4` if the rehearsal shows `DNF (timeout: loading)`.
