# Calibration: measured numbers for the lesson

Every number the chapters, the checkpoint bands and the instructor docs quote comes from this page. It is generated: do not edit it by hand, re-measure instead.

```
python lesson/instructor/calibrate.py                 # mock LLM: drivers, handrails, curves, cliff, fly (~5 min)
python lesson/instructor/calibrate.py --real-llm      # + MiniMax: Max, ch5, ch6, ch4 pilot (~40-60 calls)
python lesson/instructor/calibrate.py --parts real    # only the MiniMax part (keeps the other numbers)
```

Raw data: [`calibration.json`](calibration.json). Measured on Windows 11, Python 3.14.2, commit `befd421`. Sections: bulk 2026-10-22 (14 s), curves 2026-10-22 (3 s), fly 2026-10-22 (18 s), cliff 2026-10-22 (212 s), real 2026-10-22 (209 s).

How to read the times: one-lap attempts, default **marshal** crash rule, race time = sim time + 5 s per crash + 1 s per tick too fast in a yellow-flag zone. `(2c)` = 2 crashes, `(2z)` = 2 zone violations (both already inside the time). `a -> b` = attempt 1 -> best attempt. Code drivers are deterministic: students get exactly these numbers. LLM drivers in the big tables ran with `LLM_PROVIDER=mock` (= their code fallback); the MiniMax numbers are in their own section.

## Checkpoint bands (CP0-CP8)

What a student should see at each checkpoint, and what the handrail file in `lesson/checkpoints/` measures. Chapters quote this table. "Clean" = finished with 0 crashes.

| CP | Chapter | Handrail | Band (what counts as on track) | Handrail measured | OK |
|---|---|---|---|---|---|
| CP0 | Ch0 Pit Lane | `ch0_pit_lane.py` | `race.py check` PASSED; your car in the viewer; grand_prix finishes (the template drives slowly but cleanly) | grand_prix 85.85; clean on 5/5 public tracks | yes |
| CP1 | Ch1 Reflexes | `ch1_reflex.py` | grand_prix 60-80 s **clean**; clean on every public track. Silver < 75 s, Gold < 55 s and clean on sprint | grand_prix 66.70, sprint 50.00; clean on 5/5 public, 3/3 random | yes |
| CP2 | Ch2 The Map | `ch2_planner.py` | grand_prix 48-55 s clean; clean on every public AND random track without retuning (briefing_demo still collects zone penalties: no briefing reader yet). Gold: racing line | grand_prix 52.50; clean on 5/5 public, 3/3 random; random mean 43.7 s (Gold reference l2b_racing_line 49.95) | yes |
| CP3 | Ch3 Practice | `ch3_learner.py` | `--track sprint --attempts 6`: best >= 3 % faster than attempt 1, no DNF, clean best | sprint 41.40 -> 39.40 (4.8 %); grand_prix 52.50 -> 47.45 (9.6 %) | yes |
| CP4 | Ch4 MiniMax | `ch4_pilot.py` | not raced: exam table + p50/p95 of your calls + one sentence on where the LLM belongs; <= 20 LLM ticks per run; with the mock LLM the reflexes drive the whole lap | mock: grand_prix 66.70 (= the Ch1 reflexes); MiniMax: 20 calls, p50 1.04 s / p95 1.45 s, lap 65.75 | yes |
| CP5 | Ch5 Strategist | `ch5_strategist.py` | within +-5 % of your own Ch3 time; 0 zone violations on briefing_demo (real LLM); <= 3 LLM calls in plan(); still finishes with `LLM_PROVIDER=mock` | mock = physics plan: grand_prix best-of-4 49.45 (ch3: 49.45); MiniMax briefing_demo 61.15 with 1 call(s); MiniMax grand_prix qualifying 50.95 | yes |
| CP6 | Ch6 Crew Chief | `ch6_crew_chief.py` | improving curve over 4 attempts on a `random:<seed>` track; <= 3 LLM calls per reflect (official: 2); finishes with the mock LLM (then the Ch3 rules learn) | mock random:7: 44.45 -> 44.45 -> 42.60 -> 42.25; MiniMax random:7: practice 45.05 -> 43.35 -> 43.30, qualifying 43.30, 5 calls | yes |
| CP7 | Ch7 Fly Brains | `ch7_fly_fixed.py` | toy fly finishes a lap (fixed fly = exactly fly_toy, mirror self-check SYMMETRIC); scrambled wiring: DNF or far slower; connectome real wiring >= 8 of 16 gates | grand_prix 63.30; SCRAMBLE 0-4: 4/5 DNF; connectome oval real 49.20, scrambled: 162.60 (17c), 154.95 (16c), 149.50 (15c), 126.35 (9c), DNF (0 gates) (4c) | yes |
| CP8 | Ch8 Garage | `ch6_crew_chief.py` | final file: `check` green, clean on every public track, finishes with `LLM_PROVIDER=mock` (`validate --random 3` posts a score) | ch6 (mock) clean on 5/5 public, 3/3 random; public mean 46.0 s | yes |

## Lap times: every driver on every track (mock LLM)

Public tracks + random seeds 7, 12, 21. Learners (l3, l6, l6b, ch3, ch5, ch6): 4 attempts, shown as attempt 1 -> best.

| Driver | grand_prix | oval | sprint | speedway | briefing_demo | random:7 | random:12 | random:21 |
|---|---|---|---|---|---|---|---|---|
| template | 85.85 | 44.40 | 64.80 | 106.30 | 91.85 (1z) | 73.80 | 72.20 | 77.45 |
| l0_dummy | 307.40 (31c) | 151.60 (15c) | 207.30 (21c) | 284.35 (26c) | 316.40 (31c, 9z) | 234.75 (23c) | 212.85 (20c) | 217.35 (20c) |
| l1_reflex | 66.70 | 31.65 | 50.00 | 74.30 | 71.70 (1z) | 55.10 | 52.80 | 54.60 |
| l2_planner | 52.50 | 25.95 | 41.40 | 60.60 | 59.75 (2z) | 44.45 | 42.65 | 44.00 |
| l2b_racing_line | 49.95 | 25.00 | 38.40 | 58.20 | 57.65 (2z) | 44.15 | 39.15 | 42.85 |
| l3_learner | 52.50 -> 48.30 | 25.95 -> 24.70 | 41.40 -> 39.50 | 60.60 -> 57.00 | 59.75 (2z) -> 57.20 (2z) | 44.45 -> 41.50 | 42.65 -> 39.65 | 44.00 -> 42.25 |
| l4_llm_rookie | 71.80 | 34.50 | 49.35 | 82.35 | 76.65 (1z) | 59.25 | 56.95 | 60.05 |
| l5_llm_strategist | 49.95 | 25.00 | 38.40 | 58.20 | 57.65 (2z) | 44.15 | 39.15 | 42.85 |
| l6_llm_crew_chief (Max) | 49.95 -> 49.95 | 25.00 -> 25.00 | 38.40 -> 38.40 | 58.20 -> 58.20 | 57.30 -> 57.30 | 44.15 -> 44.15 | 39.15 -> 39.15 | 42.85 -> 42.85 |
| l6b_pit_crew_langgraph | 49.95 -> 49.95 | 25.00 -> 25.00 | 38.40 -> 38.40 | 58.20 -> 58.20 | 57.30 -> 57.30 | 44.15 -> 44.15 | 39.15 -> 39.15 | 42.85 -> 42.85 |
| fly_toy | 63.30 | 29.00 | 47.50 | 69.40 | 76.30 (13z) | 50.30 | 47.65 | 50.15 |
| fly_broken | 93.05 (2c) | 29.00 | 47.55 | 70.80 | 108.05 (2c, 15z) | 50.30 | 47.75 | 51.40 |
| fly_connectome | 117.10 (2c) | 49.20 | 78.55 (1c) | 122.80 | 134.10 (2c, 17z) | 86.75 | 84.75 (1c) | 87.05 |
| **ch0_pit_lane** | 85.85 | 44.40 | 64.80 | 106.30 | 91.85 (1z) | 73.80 | 72.20 | 77.45 |
| **ch1_reflex** | 66.70 | 31.65 | 50.00 | 74.30 | 72.75 (1z) | 55.10 | 52.80 | 54.60 |
| **ch2_planner** | 52.50 | 25.95 | 41.40 | 60.60 | 59.75 (2z) | 44.45 | 42.65 | 44.00 |
| **ch3_learner** | 52.50 -> 49.45 | 25.95 -> 25.10 | 41.40 -> 40.40 | 60.60 -> 56.75 | 59.75 (2z) -> 58.05 (2z) | 44.45 -> 42.25 | 42.65 -> 39.70 | 44.00 -> 42.45 |
| **ch4_pilot** | 66.70 | 31.65 | 50.00 | 74.30 | 72.75 (1z) | 55.10 | 52.80 | 54.60 |
| **ch5_strategist** | 52.50 -> 49.45 | 25.95 -> 25.10 | 41.40 -> 40.40 | 60.60 -> 56.75 | 59.75 (2z) -> 58.05 (2z) | 44.45 -> 42.25 | 42.65 -> 39.70 | 44.00 -> 42.45 |
| **ch6_crew_chief** | 52.50 -> 49.45 | 25.95 -> 25.10 | 41.40 -> 40.40 | 60.60 -> 56.75 | 59.75 (2z) -> 58.05 (2z) | 44.45 -> 42.25 | 42.65 -> 39.70 | 44.00 -> 42.45 |
| **ch7_fly_fixed** | 63.30 | 29.00 | 47.50 | 69.40 | 76.30 (13z) | 50.30 | 47.65 | 50.15 |

### Generalisation summary (best attempt)

| Driver | clean on public tracks | mean public (s) | clean on random seeds | mean random (s) |
|---|---|---|---|---|
| template | 5/5 | 78.6 | 3/3 | 74.5 |
| l0_dummy | 0/5 | 253.4 | 0/3 | 221.7 |
| l1_reflex | 5/5 | 58.9 | 3/3 | 54.2 |
| l2_planner | 5/5 | 48.0 | 3/3 | 43.7 |
| l2b_racing_line | 5/5 | 45.8 | 3/3 | 42.0 |
| l3_learner | 5/5 | 45.3 | 3/3 | 41.1 |
| l4_llm_rookie | 5/5 | 62.9 | 3/3 | 58.8 |
| l5_llm_strategist | 5/5 | 45.8 | 3/3 | 42.0 |
| l6_llm_crew_chief (Max) | 5/5 | 45.8 | 3/3 | 42.0 |
| l6b_pit_crew_langgraph | 5/5 | 45.8 | 3/3 | 42.0 |
| fly_toy | 5/5 | 57.1 | 3/3 | 49.4 |
| fly_broken | 3/5 | 69.7 | 3/3 | 49.8 |
| fly_connectome | 2/5 | 100.3 | 2/3 | 86.2 |
| ch0_pit_lane | 5/5 | 78.6 | 3/3 | 74.5 |
| ch1_reflex | 5/5 | 59.1 | 3/3 | 54.2 |
| ch2_planner | 5/5 | 48.0 | 3/3 | 43.7 |
| ch3_learner | 5/5 | 46.0 | 3/3 | 41.5 |
| ch4_pilot | 5/5 | 59.1 | 3/3 | 54.2 |
| ch5_strategist | 5/5 | 46.0 | 3/3 | 41.5 |
| ch6_crew_chief | 5/5 | 46.0 | 3/3 | 41.5 |
| ch7_fly_fixed | 5/5 | 57.1 | 3/3 | 49.4 |

### House cars

`--house` adds them to a grand prix; they race but score no points. Same numbers as above (Max under the mock LLM = its code plan on the racing line).

| House car | File | grand_prix | oval | sprint | speedway | briefing_demo | random:7 | random:12 | random:21 |
|---|---|---|---|---|---|---|---|---|---|
| Max | `drivers/examples/l6_llm_crew_chief.py` | 49.95 -> 49.95 | 25.00 -> 25.00 | 38.40 -> 38.40 | 58.20 -> 58.20 | 57.30 -> 57.30 | 44.15 -> 44.15 | 39.15 -> 39.15 | 42.85 -> 42.85 |
| Planner | `drivers/examples/l2_planner.py` | 52.50 | 25.95 | 41.40 | 60.60 | 59.75 (2z) | 44.45 | 42.65 | 44.00 |
| Fly Toy | `drivers/examples/fly_toy.py` | 63.30 | 29.00 | 47.50 | 69.40 | 76.30 (13z) | 50.30 | 47.65 | 50.15 |

## Learning curves (6 attempts, mock LLM)

`python race.py drive <file> --track <track> --attempts 6`. Gain = (attempt 1 - best) / attempt 1.

| Driver | Track | Attempts 1..6 | Best | Gain | Clean |
|---|---|---|---|---|---|
| l3_learner | grand_prix | 52.50 / 50.55 / 49.30 / 48.30 / 47.60 / 46.30 | 46.30 | 11.8 % | yes |
| l3_learner | oval | 25.95 / 25.75 / 25.10 / 24.70 / 24.70 / 24.90 | 24.70 | 4.8 % | yes |
| l3_learner | sprint | 41.40 / 40.90 / 39.50 / 39.80 / 39.30 / 41.50 | 39.30 | 5.1 % | yes |
| l3_learner | speedway | 60.60 / 57.40 / 57.00 / 57.00 / 55.35 / 55.15 | 55.15 | 9.0 % | yes |
| l3_learner | briefing_demo | 59.75 (2z) / 58.05 (2z) / 57.25 (2z) / 57.20 (2z) / 56.40 (2z) / 55.00 (2z) | 55.00 (2z) | 7.9 % | yes |
| l3_learner | random:7 | 44.45 / 42.60 / 41.50 / 41.55 / 40.85 / 39.95 | 39.95 | 10.1 % | yes |
| l3_learner | random:12 | 42.65 / 40.55 / 39.75 / 39.65 / 39.00 / 39.75 | 39.00 | 8.6 % | yes |
| l3_learner | random:21 | 44.00 / 42.70 / 42.35 / 42.25 / 41.80 / 42.00 | 41.80 | 5.0 % | yes |
| ch3_learner | grand_prix | 52.50 / 51.70 / 49.80 / 49.45 / 48.05 / 47.45 | 47.45 | 9.6 % | yes |
| ch3_learner | oval | 25.95 / 25.85 / 25.35 / 25.10 / 25.00 / 24.80 | 24.80 | 4.4 % | yes |
| ch3_learner | sprint | 41.40 / 41.65 / 40.90 / 40.40 / 40.35 / 39.40 | 39.40 | 4.8 % | yes |
| ch3_learner | speedway | 60.60 / 60.35 / 56.75 / 57.00 / 55.45 / 55.30 | 55.30 | 8.7 % | yes |
| ch3_learner | briefing_demo | 59.75 (2z) / 60.50 (2z) / 58.05 (2z) / 58.25 (2z) / 58.20 (2z) / 58.35 (2z) | 58.05 (2z) | 2.8 % | yes |
| ch3_learner | random:7 | 44.45 / 44.45 / 42.60 / 42.25 / 42.15 / 41.65 | 41.65 | 6.3 % | yes |
| ch3_learner | random:12 | 42.65 / 41.90 / 40.25 / 39.70 / 39.15 / 38.65 | 38.65 | 9.4 % | yes |
| ch3_learner | random:21 | 44.00 / 44.10 / 42.70 / 42.45 / 42.20 / 42.15 | 42.15 | 4.2 % | yes |

### Learning gone wrong (the keep-best ablation)

Keep-best **off** = ch3_learner whose reflect() forgets how good its best lap was, so every lap becomes the starting point of the next experiment (no revert, no smaller step after a bad change). "greedy" = `--param push=1.3 --param crash_cut=1.0` (big pushes, crashes not punished). The "last" column is what a learner without lock-in would take into qualifying. Where the on/off rows are identical, every change was an improvement, so keep-best never had to act. The PathFollower's stopping-distance safety net keeps even greedy knobs out of crash loops: here "learning gone wrong" means ending on a worse lap than the best one, not a DNF.

| Knobs | Keep-best | Track | Attempts 1..6 | Best | Last |
|---|---|---|---|---|---|
| default | on | grand_prix | 52.50 / 51.70 / 49.80 / 49.45 / 48.05 / 47.45 | 47.45 | 47.45 |
| default | **off** | grand_prix | 52.50 / 51.70 / 49.80 / 49.45 / 48.05 / 47.45 | 47.45 | 47.45 |
| greedy push=1.3 crash_cut=1.0 | on | grand_prix | 52.50 / 52.30 / 47.75 / 47.70 / 47.00 / 47.00 | 47.00 | 47.00 |
| greedy push=1.3 crash_cut=1.0 | **off** | grand_prix | 52.50 / 52.30 / 47.75 / 47.70 / 47.00 / 47.00 | 47.00 | 47.00 |
| default | on | sprint | 41.40 / 41.65 / 40.90 / 40.40 / 40.35 / 39.40 | 39.40 | 39.40 |
| default | **off** | sprint | 41.40 / 41.65 / 39.70 / 39.50 / 39.35 / 40.50 | 39.35 | 40.50 |
| greedy push=1.3 crash_cut=1.0 | on | sprint | 41.40 / 40.60 / 38.35 / 37.40 / 41.10 / 37.40 | 37.40 | 37.40 |
| greedy push=1.3 crash_cut=1.0 | **off** | sprint | 41.40 / 40.60 / 38.35 / 37.40 / 41.10 / 41.10 | 37.40 | 41.10 |

## The latency cliff (realtime mode)

A wrapper driver sleeps `think` seconds inside every `drive()` call, raced with `SessionConfig(realtime=True)` (= `race.py drive --realtime`): while it thinks, the car keeps executing its previous command for floor(think / 0.5) ticks, then the (stale) new command is applied. Track grand_prix, tick limit 500. Blind distance = speed x think time (3 s at 30 u/s = 90 units = 5 track widths). The marshal crash rule puts the car back on track after every crash, which is why some blind runs still reach the line - with dozens of crashes and several times the lap time. Which latency DNFs and which limps home is noisy; the cliff between 0 s and 0.5 s is not.

| Driver | think (s) | blind ticks per decision | Result | Gates | Late ticks | Wall time (s) |
|---|---|---|---|---|---|---|
| ch1_reflex | 0 | 0 | 66.70 | 16 | 0 | 0 |
| ch1_reflex | 0.5 | 1 | 220.25 (17c) | 16 | 136 | 69 |
| ch1_reflex | 1 | 2 | 445.95 (46c) | 16 | 288 | 145 |
| ch1_reflex | 2 | 4 | DNF (15 gates) (70c) | 15 | 400 | 201 |
| ch1_reflex | 3 | 6 | 573.40 (66c) | 16 | 418 | 211 |
| l2_planner | 0 | 0 | 52.50 | 16 | 0 | 0 |
| l2_planner | 0.5 | 1 | 434.15 (42c) | 16 | 225 | 114 |
| l2_planner | 1 | 2 | DNF (14 gates) (57c) | 14 | 334 | 168 |
| l2_planner | 2 | 4 | DNF (8 gates) (81c) | 8 | 400 | 201 |
| l2_planner | 3 | 6 | 465.65 (58c) | 16 | 302 | 154 |

## Fly brains: real wiring vs scrambled controls (no LLM)

`--param SCRAMBLE=<seed>` keeps every WIRING row (and, for the connectome, every neuron's number of connections) but sends it to a random target. A brain that beats its scrambled copies shows that the wiring carries the skill. DNF shows the gates reached (of 16).

| Driver | Track | real wiring | SCRAMBLE=0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|
| ch7_fly_fixed | grand_prix | 63.30 | DNF (0 gates) (100c) | DNF (5 gates) (4c) | DNF (0 gates) | DNF (0 gates) (63c) | 198.80 (22c) |
| ch7_fly_fixed | oval | 29.00 | DNF (0 gates) (83c) | 80.65 (8c) | DNF (0 gates) | DNF (0 gates) (58c) | 65.30 (6c) |
| fly_broken | grand_prix | 93.05 (2c) | DNF (0 gates) (100c) | DNF (5 gates) (3c) | DNF (0 gates) | DNF (0 gates) (74c) | 249.60 (29c) |
| fly_broken | oval | 29.00 | DNF (0 gates) (83c) | 79.90 (8c) | DNF (0 gates) | DNF (0 gates) (61c) | 124.60 (14c) |
| fly_connectome | oval | 49.20 | 162.60 (17c) | 154.95 (16c) | 149.50 (15c) | 126.35 (9c) | DNF (0 gates) (4c) |
| fly_toy | grand_prix | 63.30 | DNF (0 gates) (100c) | DNF (5 gates) (4c) | DNF (0 gates) | DNF (0 gates) (63c) | 198.80 (22c) |
| fly_toy | oval | 29.00 | DNF (0 gates) (83c) | 80.65 (8c) | DNF (0 gates) | DNF (0 gates) (58c) | 65.30 (6c) |

## With the real LLM (MiniMax)

Provider `minimax:MiniMax-M3`. Official weekend format, the same as an event: `plan()` -> 3 practice laps with `reflect()` -> qualifying lap (plan locked in) -> 1-lap race; official LLM rules (<= 2 calls in plan, <= 2 per practice reflect, none in drive / qualifying / race). Drivers run in parallel child processes like a real grand prix. LLM answers vary from run to run: treat these as one sample.

| Track | Driver | Practice 1 -> 2 -> 3 | Qualifying | Race | LLM calls | Tokens (prompt + completion) | LLM time (s) | per call p50 / p95 / max (s) |
|---|---|---|---|---|---|---|---|---|
| grand_prix | Max | 46.50 -> 45.40 -> 44.90 | 44.90 | 44.90 | 6 | 20281 + 6070 | 43.172 | 5.07 / 19.88 / 19.88 |
| grand_prix | CP5 Strategist | 53.85 -> 52.85 -> 50.95 | 50.95 | 50.95 | 2 | 3415 + 1319 | 7.82 | 3.91 / 5.9 / 5.9 |
| grand_prix | CP6 Crew Chief | 52.50 -> 52.55 -> 51.30 | 51.30 | 51.30 | 6 | 13457 + 4245 | 37.432 | 3.35 / 18.4 / 18.4 |
| random:7 | Max | 41.45 -> 41.60 -> 41.95 | 41.45 | 41.45 | 6 | 19142 + 6543 | 67.77 | 7.07 / 39.83 / 39.83 |
| random:7 | CP5 Strategist | 44.85 -> 44.40 -> 42.05 | 42.05 | 42.05 | 2 | 2623 + 606 | 4.156 | 2.08 / 2.7 / 2.7 |
| random:7 | CP6 Crew Chief | 45.05 -> 43.35 -> 43.30 | 43.30 | 43.30 | 5 | 12546 + 11216 | 101.861 | 9.24 / 44.5 / 44.5 |

Wall time per weekend (3 cars in parallel): grand_prix 49 s, random:7 110 s.

**ch5 on briefing_demo** (one practice lap, official plan budget): 61.15, 1 LLM call(s), 1189 + 2060 tokens, 22.7 s of LLM time. 0 zone violations: it read the yellow flag from the briefing (the mock gets 2).

**ch4 pilot** (turn-based, grand_prix, MAX_LLM_TICKS 20, `effort="none"`): 65.75; 20 LLM calls, per call p50 1.04 s / p95 1.45 s / max 4.2 s; 8410 + 521 tokens. Ticks per call at p50 = 2.1: in realtime the car would drive that many ticks blind per decision.

