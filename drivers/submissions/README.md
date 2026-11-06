# Submissions

Put **one file** here: `drivers/submissions/<handle>.py`, where `<handle>` is your handle in lower case with
no spaces, e.g. `ada.py` or `ada_lovelace.py`.
Start from `drivers/template.py`. Optional extra files, never raced in the main event:
`<handle>_pilot.py` (Ch4 LLM pilot) and `<handle>_fly.py` (Fly Cup). The instructor never races this
folder directly: scrutineering copies the passed main files into a separate grid folder, so the side files
stay out of the main events.

## Before you submit (all three must pass)

```bash
python race.py check    drivers/submissions/<handle>.py            # contract + smoke test: must say PASSED
python race.py validate drivers/submissions/<handle>.py --random 3 # clean on the public + random tracks?
LLM_PROVIDER=mock python race.py drive drivers/submissions/<handle>.py   # must still finish without an LLM
```

On Windows PowerShell the last line is: `$env:LLM_PROVIDER="mock"; python race.py drive drivers/submissions/<handle>.py`.

## The rules (short version)

**Your file**
- One `.py` file, at most 100 KB.
- Metadata: `NAME` (at most 16 characters, pseudonyms welcome, no emoji, unique in the class, and not a
  house car's name: Max, Planner and Fly Toy are taken), `AUTHOR` (your handle), `COLOR` (hex, like
  `"#e63946"`; a colour word such as `"red"` is silently replaced),
  `KIND` = `"code"`, `"llm"` or `"fly"`. Optional: `MOTTO`, `PARAMS` (numeric knobs), `ANATOMY`.
- Interface: either a function `drive(obs) -> (throttle, steering)`, or a `class Driver` with
  `plan(track)`, `drive(obs)` and `reflect(report)`. `drive` may return a dict with `"throttle"`,
  `"steering"`, `"say"` (radio message) and `"debug"`.
- Imports: the Python standard library, `numpy`, `agentic_gp.driverkit`, `agentic_gp.llm`,
  `agentic_gp.fly` (fly drivers), `agentic_gp.driver` (the field lists and types), and `langgraph` /
  `langchain_core` / `langchain_openai` for a crew variant. Not allowed: `agentic_gp.sim`, `.race`, `.cli`,
  `.viz`, `openai`, `requests` and other network or process libraries. `race.py check` tells you.
- No file I/O (`open()`, `Path.read_text()`, `np.load()`, ...: `race.py check` fails them), no API key in the
  file. Each session starts with fresh memory: only what is in your file persists. In a race your driver runs
  in its own process under a guard that also refuses file access, network connections and new processes at run
  time.

**Timing**
- `drive()` must average under 50 ms and never take more than 2 s. A late answer = the car coasts that
  tick (a "stall"); 10 stalls in a row or 30 in total = DNF.
- An exception or a NaN in `drive()` = the car coasts that tick and the error is logged.
- `plan()` and `reflect()` may take up to 120 s each; 300 s of thinking time per driver per track.

**LLM budget (official events)**
- LLM calls only inside `plan()` and inside `reflect()` after a **practice** lap:
  at most **2 calls in `plan()`**, **2 per practice `reflect()`**, **8 per track**.
- No LLM in `drive()`, in qualifying or in the race: `agentic_gp.llm` raises `LLMNotAllowed` there.
  Catch `llm.LLMError` and fall back to code: "the LLM advises, code keeps the car safe".
- Every prompt and answer is logged (`races/<event>/llm_logs/`).

**Events**
- Grand prix: `plan` -> 2 practice laps (with `reflect`) -> 1 qualifying lap (the grid) -> 3-lap race.
  The Mystery GP gets 3 practice laps. `report["next_phase"]` tells you what comes next: lock in your
  best plan when it says `"qualifying"` or `"race"`.
- Classification: race time = sim time + 5 s per crash + 1 s per tick too fast in a yellow-flag zone.
  Ties go to fewer crashes; DNFs are ranked by progress. Yellow flags are announced **only** in
  `track["briefing"]`.
- Championship: N+1-position points per race; a DNF that passed at least 8 gates scores 1.
  House cars (Max) race but score no points - beat them for the "Beat the House" badge.

**Classes**: Open; LLM class (`KIND="llm"` and the logs show a validated LLM output that changed your
plan); Fly class (controls decoded from neuron activity only).

## Submitting

Upload the file the way your instructor told you (LMS or a pull request into this folder) by the
deadline. Fill in the 5-line driver card in your docstring: approach, what the LLM does, what
`reflect` learns, anatomy, what you would do with another day.
