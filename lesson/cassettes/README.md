# LLM cassettes: record real MiniMax answers, replay them offline

A **cassette** is a JSONL file of real LLM answers. You record it once before class with the real key
(`LLM_RECORD`). In class you play it back with no network and no key (`LLM_PROVIDER=replay`). The driver
then makes "the same" LLM calls and gets the recorded answers back, so the run reproduces the recorded one.

Use cassettes for the **live LLM demos** you want to be able to show even when the Wi-Fi, the key or MiniMax fails:
- the Ch4 pilot driving tick by tick;
- the Ch5 strategist writing its plan from the briefing;
- the Ch6 crew chief debriefing between laps.

**Never fake a cassette.** Do not write or edit the answers by hand. A cassette is evidence of what the model
really said, and students will compare it with their own calls. To show "what if the LLM answers garbage",
use the mock provider (`LLM_PROVIDER=mock` answers `{}`) or `llm.set_mock_responder(...)` in a test.

**You do not need a cassette to show a race.** A replay JSON (`races/demo/*.json`, `races/final/home.json`
...) already contains the whole race. `race.py watch FILE --html | --unity` plays it without an LLM.
Cassettes are for **re-running a driver** live.

## How it works (from `agentic_gp/llm.py`)

- `LLM_RECORD=<file.jsonl>`, with a real provider: every real answer is **appended** to the file. Each line
  holds:
  - `key`, a hash of the messages and tools;
  - `model`;
  - the request `messages` and `tools`;
  - the `reply`: content, reasoning, tool calls, finish reason and token usage.

  The API key is never written; it is not part of the request messages.
- `LLM_PROVIDER=replay` + `LLM_CASSETTE=<file.jsonl>`: each call returns a recorded reply. The cassette is
  matched by the same request hash first, otherwise by **the next unused entry in order**. Replay the
  **same command** you recorded: a different driver or track gets the old answers in order, which is
  nonsense.
- Replayed calls count in the statistics with the recorded token numbers. They take no network time.
- A missing cassette raises `LLMError` ("LLM_PROVIDER=replay needs LLM_CASSETTE=<recorded .jsonl>").
  A cassette with fewer answers than the run needs raises "cassette ... is exhausted". Drivers with a
  code fallback (all handrails and examples) keep driving on code in both cases.

## Record (T-3, instructor laptop, real key in `.env`)

Start every recording from an empty file: `LLM_RECORD` appends.

**PowerShell**
```powershell
New-Item -ItemType Directory -Force lesson/cassettes | Out-Null
Remove-Item lesson/cassettes/l5_briefing_demo.jsonl -ErrorAction SilentlyContinue
$env:LLM_RECORD = "lesson/cassettes/l5_briefing_demo.jsonl"
python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo --replay races/demo_local/l5_live.json
Remove-Item Env:LLM_RECORD
```

**macOS**
```bash
mkdir -p lesson/cassettes
rm -f lesson/cassettes/l5_briefing_demo.jsonl
LLM_RECORD=lesson/cassettes/l5_briefing_demo.jsonl python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo --replay races/demo_local/l5_live.json
```

The end of the output should look like this (measured with the real MiniMax-M3, effort low):
```
Attempt 1: FINISHED 57.80 s | laps [57.80] | crashes 0 | penalties 0.0 s | 16 waypoints | 116 ticks
drive() time: mean 1.1 ms, max 43.5 ms | LLM: 2 calls, 7423 tokens | wall 220.4 s
```
The cassette then has 2 lines (28 KB). The strategist read the yellow flag: 0 s of penalties. With the mock
LLM the same driver collects 2.0 s of zone penalties.

The wall time of 220 s comes from the development machine, which was at 100 % CPU and needed minutes for a
cold `import openai`. The LLM part of `plan()` took 36 s.

## Replay (in class, offline)

**PowerShell**
```powershell
$env:LLM_PROVIDER = "replay"; $env:LLM_CASSETTE = "lesson/cassettes/l5_briefing_demo.jsonl"
python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo
Remove-Item Env:LLM_PROVIDER, Env:LLM_CASSETTE
```

**macOS**
```bash
LLM_PROVIDER=replay LLM_CASSETTE=lesson/cassettes/l5_briefing_demo.jsonl python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo
```

Measured: `FINISHED 57.80 s`, `LLM: 2 calls, 7423 tokens`, the same lap as the recording. The wall time was
18.7 s instead of 220 s, and there was no network.

**Show the trace on the Pit Wall page.** Add `LLM_LOG_DIR` while you replay. The page reads every `*.jsonl`
inside an `llm_logs` folder under `races/`:
```powershell
$env:LLM_LOG_DIR = "races/demo_local/llm_logs"     # macOS: LLM_LOG_DIR=races/demo_local/llm_logs ... in front of the command
```

## Which cassettes to record

| Demo | Record with (same command later for replay) | Calls |
|---|---|---|
| Ch4: the LLM drives tick by tick | `python race.py drive lesson/checkpoints/ch4_pilot.py` | 20 (its `MAX_LLM_TICKS`) |
| Ch5: plan from the briefing | `python race.py drive drivers/examples/l5_llm_strategist.py --track briefing_demo` | 1-2 |
| Ch6: Max debriefs on an unseen track | `python race.py drive drivers/examples/l6_llm_crew_chief.py --track random:7 --attempts 4` | about 6 (the calibration run used 5) |

Use one file per demo (`lesson/cassettes/<demo>.jsonl`).

**Not with cassettes: the latency wall.** A replayed answer arrives instantly. A `--realtime` run from a cassette
therefore shows no latency at all, which is the opposite of the lesson. For Ch4's latency demo, use:
- the measured data in `races/demo/latency_cliff.json`;
- the Latency Lab page;
- `lesson/assets/latency_cliff.png`.

**Keep or commit?** A cassette holds prompts and model answers, never the key. You may commit the files to
this folder for the next course, or keep them local. Re-record them when a driver's prompt changes: replay
falls back to "next answer in order" and would quietly give stale answers.
