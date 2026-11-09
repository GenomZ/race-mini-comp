# Awards: the list and how each one is computed

The awards ceremony is 20 minutes on Day 2 at 16:15. Before it, run [`awards.py`](awards.py); it takes about
4 seconds. It reads the result files that `race.py final` wrote and prints a markdown table you can project.
It also prints a "How each award was computed" section with the runner-up data.

```
python lesson/instructor/awards.py --snapshot1 races/snapshots/snap1_results.json --fly-cup races/flycup/flycup_results.json --out races/final/awards.md
```

| Input | Written by | Used for |
|---|---|---|
| `races/final/home_results.json`, `mystery_results.json` | `race.py final ... --out races/final` | almost everything |
| `races/final/championship.json` | the same command; recomputed with `agentic_gp.race.leaderboard.championship` when missing | Champion, LLM Class |
| `races/final/home.json`, `mystery.json` (the replays) | the same command | zone penalties (Cleanest Driver), radio messages (Best Radio) |
| `--snapshot1 races/snapshots/snap1_results.json` | Snapshot #1 (RUN_OF_SHOW [C2]) | Biggest Improvement |
| `--fly-cup races/flycup/flycup_results.json` | the Fly Cup (RUN_OF_SHOW [C9]) | Best Fly Brain |
| `--final DIR` | any `race.py final --out DIR` (default `races/final`) | try it on the dress rehearsal: `--final races/rehearsal` |

**Ground rules used by every award**
- **House cars never win.** Max, Planner and Fly Toy (`--house`) race but are left out, exactly like the
  championship. When a house car was faster, the detail column says so, e.g. "Max (house) finished ahead"
  or "real pole: Max (house), 44.00 s".
- **Race time** is the official classification time: sim time + 5 s per crash + 1 s per tick too fast in a
  yellow-flag zone. A car "finished" when it has a race time and an empty status. `DNF (...)` statuses
  never finish.
- **Matching a student across files** (Snapshot #1 vs the final) uses the driver file name first, then AUTHOR,
  then NAME. That is why `grade.py scrutineer` refuses duplicate NAMEs.
- A **catch-up / example file raced verbatim** is classified and can win race awards. It is **not**
  eligible for the design awards (Best Agent Design, Most Creative LLM Use). `grade.py grade` flags such
  files ("verbatim copy of lesson/checkpoints/...").
- **Display:** project the top 10 and the award table. Pseudonyms are allowed (NAME is up to 16 characters).

## The awards

| # | Award | Computed by `awards.py` as |
|---|---|---|
| 1 | **Champion** | most championship points: N+1-position per race for N scored cars; a DNF that passed at least 8 gates scores 1. Ties go to the better Mystery GP result (`championship.json`) |
| 2 | **Home GP winner** | first scored car in the Home GP standings |
| 3 | **Mystery GP winner** | first scored car in the Mystery GP standings |
| 4 | **Pole** (one per GP) | fastest qualifying lap among scored cars |
| 5 | **Fastest race lap** (one per GP) | lowest `best_lap` (single race lap) among scored cars |
| 6 | **Beat the House** (a badge, any number of winners) | every scored car classified ahead of Max in either race. If Max DNFs in a race, everybody classified ahead of it gets the badge, and the detail column shows Max's status |
| 7 | **Best Learner** | largest gain from practice lap 1 to the qualifying lap on the **mystery** track, among scored cars whose qualifying lap is within 10 % of pole (pole includes Max). A gain of 0 wins nothing |
| 8 | **Cleanest Driver** | among cars that finished **both** races: fewest crashes + zone penalties in the two races (zone penalties are counted from the race replays); ties go to the lower total race time |
| 9 | **LLM Class winner** | best championship position among `KIND="llm"` cars that made at least one LLM call. **You confirm** in `races/final/llm_logs/home/` and `.../mystery/` (or Race Control → Pit Wall) that a validated LLM answer changed the plan (CURRICULUM's class rule) |
| 10 | **Most Efficient** | best Mystery GP race time among finishers that used the LLM with **1 to 3** calls over the whole mystery weekend (the standings' LLM column). Code-only cars are not in this award: they are in the Open class |
| 11 | **Best Fly Brain** | fastest finished `KIND="fly"` car in the Fly Cup (or in the GPs if no Fly Cup file is given). **You confirm** that it beats its own scrambled control: `python race.py drive races/flycup/<handle>_fly.py --param SCRAMBLE=0` must be slower or DNF. The fly-class rules (`python -m agentic_gp.fly.rules FILE`) decide the "Biologically Honest" ribbon. Two checks by hand: on a **tie** `awards.py` names only one fly (unlike Pole and Fastest Lap, which list all), so look at the Fly Cup results; and a fly file that is an untouched example still carries the name "Fly Toy" and AUTHOR "Agentic GP", so check NAME and AUTHOR before you announce it |
| 12 | **Most Creative LLM Use** | jury: the logs plus a 60-second pit interview. `awards.py` lists every car's LLM calls and tokens per race as a shortlist |
| 13 | **Best Agent Design** | shortlist of 5 driver cards (the 5-line DRIVER CARD docstring) → class vote |
| 14 | **Biggest Improvement** | Snapshot #1 lap time (a DNF counts as 300 s) minus the final Home GP qualifying lap. Both are one lap of the home track; a gain of 0 wins nothing. Needs `--snapshot1` |
| 15 | **Best Radio** | peer vote. `awards.py` lists the scored cars with the most radio messages (`say`) per race as a reminder |

Jury and vote awards print as `(jury / vote: ____________)`: write the winner in by hand.

## Checked on a dry run

The real-key dry run of the final used a synthetic class of 6 handrail-based students and the 3 house cars
(`--jobs 4`, every car clean). `awards.py --final races/final_jobs4` printed:

| Award | Winner | Detail |
|---|---|---|
| Champion | Cyclone | 12 points |
| Home GP winner | Cyclone | 152.10 s (Max (house) finished ahead) |
| Mystery GP winner | Cyclone | 157.35 s |
| Pole (Home) | Cyclone | 51.70 s (real pole: Max (house), 43.75 s) |
| Pole (Mystery) | Cyclone | 53.85 s (real pole: Max (house), 53.50 s) |
| Fastest race lap (Home) | Cyclone | 50.10 s |
| Fastest race lap (Mystery) | Cyclone | 51.55 s |
| Beat the House | Cyclone, Eve Crew, Dee Strat | Home: Max P1 126.60 s, 0 ahead; Mystery: Max P4 159.00 s, 3 ahead |
| Best Learner | Cyclone | 56.20 s -> 53.85 s (gain 2.35 s, 4.2%) |
| Cleanest Driver | Eve Crew | 0 crashes + zone penalties, 312.90 s total |
| LLM Class winner | Eve Crew | championship P2, 8 points |
| Most Efficient | Dee Strat | 159.00 s with 2 LLM call(s) |
| Best Fly Brain | - | no finished fly-class car (no Fly Cup file given; the house Fly Toy does not count) |
| Biggest Improvement | - | pass --snapshot1 races/snapshots/snap1_results.json |

In the first real-key run (`--jobs 6`, an overloaded machine), all three LLM cars hit `DNF (timeout: loading)`
in the Mystery GP. Max was among them, so everybody classified ahead of Max got "Beat the House". When the
house car DNFs for an infrastructure reason, do the one documented re-run before you hand out badges.

With the mock LLM, every LLM car races its code fallback. Cars built on the same handrail then tie exactly,
and `awards.py` names all of them, e.g. "Pole (Home): Cyclone, Dee Strat, Eve Crew". Real races do not tie
like that. The dry-run timings are in [RUN_OF_SHOW.md](RUN_OF_SHOW.md), "Dry run".
