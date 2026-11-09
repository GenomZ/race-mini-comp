"""Calibration: measure every driver the lesson quotes, write CALIBRATION.md + calibration.json.

One command (from the repo root) re-measures everything the chapters and the instructor docs quote:

    python lesson/instructor/calibrate.py                 # mock LLM, ~5 min (most of it: the latency cliff)
    python lesson/instructor/calibrate.py --real-llm      # + Max, ch5, ch6 (and the ch4 pilot) with MiniMax
    python lesson/instructor/calibrate.py --parts real    # ONLY the real-LLM part; the other sections are kept
    python lesson/instructor/calibrate.py --parts bulk,curves --jobs 8

Parts (each one is a section of calibration.json; parts you do not run keep their previous numbers):
    bulk    every example driver, the template, every handrail (lesson/checkpoints) on every public
            track + random seeds (default 7, 12, 21), one-lap attempts; learners get 4 attempts.
            LLM drivers run with LLM_PROVIDER=mock here (= their code fallback).
    curves  learning curves (6 attempts) of l3_learner and ch3_learner on every track, plus the
            "learning gone wrong" ablation: ch3 without keep-best (it always continues from its last lap).
    cliff   the realtime latency cliff: a "sleepy" wrapper sleeps 0 / 0.5 / 1 / 2 / 3 s in every drive()
            call around l2_planner and ch1_reflex, raced with the runner's realtime mode (the world keeps
            moving while the driver thinks). Real sleeps: about 4 minutes, run in parallel.
    fly     fly_toy / fly_broken / ch7_fly_fixed vs scrambled wiring (SCRAMBLE 0..4) on grand_prix and oval;
            fly_connectome real vs scrambled on oval (needs data/fly/steering_circuit_783.npz).
    real    (only with --real-llm or --parts real) Max (house), ch5 and ch6 in an official grand-prix
            weekend (plan -> 3 practice laps with reflect -> qualifying -> 1-lap race, official LLM rules)
            on grand_prix and one random track; ch5 on briefing_demo (one lap: does it read the yellow
            flag?); the ch4 pilot for its 20 LLM ticks. Records calls, tokens and per-call latency.
            Costs roughly 40-60 MiniMax calls. The key comes from .env and is never printed.

Outputs: lesson/instructor/calibration.json (all numbers) and lesson/instructor/CALIBRATION.md (tables +
the checkpoint bands CP0-CP8 the chapters quote). The Markdown is always re-rendered from the JSON.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

JSON_OUT = HERE / "calibration.json"
MD_OUT = HERE / "CALIBRATION.md"
PUBLIC = ["grand_prix", "oval", "sprint", "speedway", "briefing_demo"]
SEEDS = [7, 12, 21]
CLIFF_LATENCIES = [0.0, 0.5, 1.0, 2.0, 3.0]
REAL_TRACKS = ["grand_prix", "random:7"]

EX, CP = "drivers/examples", "lesson/checkpoints"
# (label, file, group, attempts)  - learners get 4 attempts (attempt 1 -> best of 4)
DRIVERS = [
    ("template", "drivers/template.py", "template", 1),
    ("l0_dummy", f"{EX}/l0_dummy.py", "example", 1),
    ("l1_reflex", f"{EX}/l1_reflex.py", "example", 1),
    ("l2_planner", f"{EX}/l2_planner.py", "example", 1),
    ("l2b_racing_line", f"{EX}/l2b_racing_line.py", "example", 1),
    ("l3_learner", f"{EX}/l3_learner.py", "example", 4),
    ("l4_llm_rookie", f"{EX}/l4_llm_rookie.py", "example", 1),
    ("l5_llm_strategist", f"{EX}/l5_llm_strategist.py", "example", 1),
    ("l6_llm_crew_chief (Max)", f"{EX}/l6_llm_crew_chief.py", "example", 4),
    ("l6b_pit_crew_langgraph", f"{EX}/l6b_pit_crew_langgraph.py", "example", 4),
    ("fly_toy", f"{EX}/fly_toy.py", "example", 1),
    ("fly_broken", f"{EX}/fly_broken.py", "example", 1),
    ("fly_connectome", f"{EX}/fly_connectome.py", "example", 1),
    ("ch0_pit_lane", f"{CP}/ch0_pit_lane.py", "handrail", 1),
    ("ch1_reflex", f"{CP}/ch1_reflex.py", "handrail", 1),
    ("ch2_planner", f"{CP}/ch2_planner.py", "handrail", 1),
    ("ch3_learner", f"{CP}/ch3_learner.py", "handrail", 4),
    ("ch4_pilot", f"{CP}/ch4_pilot.py", "handrail", 1),
    ("ch5_strategist", f"{CP}/ch5_strategist.py", "handrail", 4),
    ("ch6_crew_chief", f"{CP}/ch6_crew_chief.py", "handrail", 4),
    ("ch7_fly_fixed", f"{CP}/ch7_fly_fixed.py", "handrail", 1),
]
HOUSE = [("Max", f"{EX}/l6_llm_crew_chief.py"), ("Planner", f"{EX}/l2_planner.py"), ("Fly Toy", f"{EX}/fly_toy.py")]


# ============================================================================ workers (child processes)
def _mock_env() -> None:
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["LLM_NO_DOTENV"] = "1"
    os.environ["LLM_QUIET"] = "1"
    for var in ("LLM_LOG_DIR", "LLM_RECORD", "LLM_CASSETTE"):
        os.environ.pop(var, None)


def _attempt_row(a) -> dict:
    r = a.result
    return {"phase": a.phase, "completed": bool(r.completed), "time": round(r.race_time, 2) if r.completed else None,
            "crashes": int(r.crashes), "zone_violations": len(a.zone_violations), "gates": int(r.waypoints_passed),
            "status": a.status or "", "late_ticks": int(a.late_ticks),
            "llm_calls_cum": int((a.llm_usage or {}).get("calls", 0))}


def run_job(job: dict) -> dict:
    """One session of one driver on one track (mock LLM). Returns a JSON-friendly row."""
    _mock_env()
    from agentic_gp.race import SessionConfig, load_driver, run_session

    t0 = time.perf_counter()
    try:
        drv = load_driver(str(REPO / job["file"]), params=job.get("params"))
        res = run_session(drv, SessionConfig(track=job["track"], attempts=job["attempts"]))
        attempts = [_attempt_row(a) for a in res.attempts]
        error = res.status or res.plan_error or None
    except Exception as exc:                      # noqa: BLE001 - a broken driver is a result, not a crash
        attempts, error = [], f"{type(exc).__name__}: {exc}"[:300]
    return dict(job, attempts=attempts, error=error, wall_s=round(time.perf_counter() - t0, 2))


def run_no_keep_best(job: dict) -> dict:
    """The "learning gone wrong" ablation: ch3_learner, but every lap becomes the new "best" (no keep-best)."""
    _mock_env()
    import importlib.util

    from agentic_gp.race import SessionConfig, run_session

    spec = importlib.util.spec_from_file_location("calib_ch3_nokeep", REPO / job["file"])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PARAMS.update(job.get("params") or {})

    class NoKeepBest(mod.Driver):
        def reflect(self, report: dict) -> None:
            if self.best:                         # forget how good the best was: the last lap always "wins"
                self.best["time"] = math.inf
            super().reflect(report)

    t0 = time.perf_counter()
    res = run_session(NoKeepBest(), SessionConfig(track=job["track"], attempts=job["attempts"]))
    return dict(job, attempts=[_attempt_row(a) for a in res.attempts], error=res.status or None,
                wall_s=round(time.perf_counter() - t0, 2))


def run_sleepy(job: dict) -> dict:
    """Realtime mode with a driver that sleeps `think` seconds in every drive() call."""
    _mock_env()
    from agentic_gp.race import SessionConfig, load_driver, run_session

    inner = load_driver(str(REPO / job["file"]))
    think = float(job["think"])

    class Sleepy:
        NAME, AUTHOR, KIND = f"Sleepy {think:g}s", "calibrate", "code"

        def plan(self, track: dict) -> None:
            inner.plan(track)

        def drive(self, obs: dict) -> tuple[float, float]:
            if think > 0:
                time.sleep(think + 0.005)         # +5 ms: floor(seconds / 0.5) must not round down
            a = inner.drive(obs)
            return a.throttle, a.steering

    t0 = time.perf_counter()
    res = run_session(Sleepy(), SessionConfig(track=job["track"], attempts=1, realtime=True,
                                              reflect_after_last=False))
    a = res.attempts[0]
    return dict(job, attempt=_attempt_row(a), ticks=int(a.result.ticks), wall_s=round(time.perf_counter() - t0, 1),
                think_max_s=round(a.think_seconds_max, 3))


def _pool(fn, jobs: list[dict], workers: int, label: str) -> list[dict]:
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            out.append(f.result())
            if k % 10 == 0 or k == len(futs):
                print(f"  {label}: {k}/{len(futs)}", flush=True)
    return out


# ============================================================================ parts
def tracks(seeds: list[int]) -> list[str]:
    return PUBLIC + [f"random:{s}" for s in seeds]


def part_bulk(seeds: list[int], workers: int) -> dict:
    jobs = [{"label": lab, "file": f, "group": g, "attempts": n, "track": t}
            for lab, f, g, n in DRIVERS for t in tracks(seeds)]
    rows = _pool(run_job, jobs, workers, "bulk")
    order = {(lab, t): i for i, (lab, t) in enumerate((j["label"], j["track"]) for j in jobs)}
    rows.sort(key=lambda r: order[(r["label"], r["track"])])
    return {"llm": "mock", "tracks": tracks(seeds), "rows": rows}


def part_curves(seeds: list[int], workers: int) -> dict:
    jobs = [{"label": lab, "file": f, "attempts": 6, "track": t}
            for lab, f in (("l3_learner", f"{EX}/l3_learner.py"), ("ch3_learner", f"{CP}/ch3_learner.py"))
            for t in tracks(seeds)]
    rows = _pool(run_job, jobs, workers, "curves")
    order = {(j["label"], j["track"]): i for i, j in enumerate(jobs)}
    rows.sort(key=lambda r: order[(r["label"], r["track"])])
    ablation = [{"label": "ch3_learner", "keep_best": False, "file": f"{CP}/ch3_learner.py", "attempts": 6,
                 "track": t, "params": p, "variant": v}
                for t in ("grand_prix", "sprint")
                for v, p in (("default", None), ("greedy push=1.3 crash_cut=1.0", {"push": 1.3, "crash_cut": 1.0}))]
    keep = [{"label": "ch3_learner", "keep_best": True, "file": f"{CP}/ch3_learner.py", "attempts": 6, "track": t,
             "params": p, "variant": v}
            for t in ("grand_prix", "sprint")
            for v, p in (("greedy push=1.3 crash_cut=1.0", {"push": 1.3, "crash_cut": 1.0}),)]
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        wrong = list(ex.map(run_no_keep_best, ablation))
        greedy = list(ex.map(run_job, keep))
    return {"llm": "mock", "rows": rows, "no_keep_best": wrong, "greedy_with_keep_best": greedy}


def part_cliff(workers: int) -> dict:
    jobs = [{"label": lab, "file": f, "track": "grand_prix", "think": lat}
            for lab, f in (("l2_planner", f"{EX}/l2_planner.py"), ("ch1_reflex", f"{CP}/ch1_reflex.py"))
            for lat in CLIFF_LATENCIES]
    rows = _pool(run_sleepy, jobs, max(workers, len(jobs)), "cliff")
    rows.sort(key=lambda r: (r["label"], r["think"]))
    return {"mode": "realtime (SessionConfig(realtime=True)), real time.sleep() in drive()", "rows": rows}


def part_fly(workers: int) -> dict:
    jobs = []
    for lab, f in (("fly_toy", f"{EX}/fly_toy.py"), ("fly_broken", f"{EX}/fly_broken.py"),
                   ("ch7_fly_fixed", f"{CP}/ch7_fly_fixed.py")):
        for t in ("grand_prix", "oval"):
            for s in [None, 0, 1, 2, 3, 4]:
                jobs.append({"label": lab, "file": f, "attempts": 1, "track": t, "scramble": s,
                             "params": None if s is None else {"SCRAMBLE": s}})
    from agentic_gp.fly.connectome import circuit_path

    connectome = circuit_path().exists()
    if connectome:
        for s in [None, 0, 1, 2, 3, 4]:
            jobs.append({"label": "fly_connectome", "file": f"{EX}/fly_connectome.py", "attempts": 1, "track": "oval",
                         "scramble": s, "params": None if s is None else {"SCRAMBLE": s}})
    rows = _pool(run_job, jobs, workers, "fly")
    rows.sort(key=lambda r: (r["label"], r["track"], -1 if r["scramble"] is None else r["scramble"]))
    return {"connectome_data": connectome, "rows": rows}


# ---------------------------------------------------------------------------- real LLM (separate process)
def _llm_log_stats(log_dir: Path) -> dict:
    """Per log file (= driver label): per-call seconds and tokens from the LLM JSONL logs."""
    out = {}
    for f in sorted(log_dir.glob("*.jsonl")):
        secs, prompt, completion, errors = [], 0, 0, 0
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("error"):
                errors += 1
            if rec.get("cached"):
                continue
            secs.append(float(rec.get("seconds") or 0.0))
            usage = ((rec.get("reply") or {}).get("usage")) or {}
            prompt += int(usage.get("prompt_tokens") or 0)
            completion += int(usage.get("completion_tokens") or 0)
        out[f.stem] = {"calls": len(secs), "errors": errors, "prompt_tokens": prompt,
                       "completion_tokens": completion, **_lat(secs)}
    return out


def _lat(secs: list[float]) -> dict:
    if not secs:
        return {"p50_s": None, "p95_s": None, "max_s": None, "total_s": 0.0}
    s = sorted(secs)
    p95 = s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]
    return {"p50_s": round(statistics.median(s), 2), "p95_s": round(p95, 2), "max_s": round(s[-1], 2),
            "total_s": round(sum(s), 1)}


def part_real() -> dict:
    """Runs in its own process with the real provider from .env (never printed)."""
    for var in ("LLM_PROVIDER", "LLM_NO_DOTENV", "LLM_CASSETTE", "LLM_RECORD"):
        os.environ.pop(var, None)
    os.environ["LLM_QUIET"] = "1"
    from agentic_gp import llm
    from agentic_gp.race import DriverEntry, GrandPrixConfig, SessionConfig, load_driver, run_grand_prix, run_session

    provider = llm.describe_llm()
    if provider.startswith("mock") or not llm.llm_available():
        return {"error": f"no real LLM configured ({provider}); put MINIMAX_API_KEY in .env"}
    result = {"provider": provider, "weekends": [], "pilot": None, "briefing": None}
    root = Path(tempfile.mkdtemp(prefix="agp_calib_llm_"))

    # 1. official weekends: Max (house), ch5, ch6
    for track in REAL_TRACKS:
        log_dir = root / track.replace(":", "_")
        entries = [DriverEntry(str(REPO / f"{EX}/l6_llm_crew_chief.py"), house=True),
                   DriverEntry(str(REPO / f"{CP}/ch5_strategist.py")),
                   DriverEntry(str(REPO / f"{CP}/ch6_crew_chief.py"))]
        cfg = GrandPrixConfig(track=track, practice_attempts=3, race_laps=1, jobs=3, llm_policy="official",
                              llm_log_dir=str(log_dir), title=f"Calibration {track}")
        print(f"  real: weekend on {track} (Max, ch5, ch6) ...", flush=True)
        t0 = time.perf_counter()
        gp = run_grand_prix(entries, cfg)
        wall = time.perf_counter() - t0
        logs = _llm_log_stats(log_dir) if log_dir.exists() else {}
        drivers = []
        for name, sess in gp.sessions.items():
            atts = [_attempt_row(a) for a in sess.attempts]
            stats = logs.get(Path(llm.log_file_name(name)).stem, {})
            drivers.append({"name": name, "attempts": atts, "llm": {k: sess.llm.get(k) for k in
                                                                     ("calls", "errors", "prompt_tokens",
                                                                      "completion_tokens", "seconds")},
                            "per_call": stats, "status": sess.status})
        result["weekends"].append({"track": track, "wall_s": round(wall, 1), "drivers": drivers,
                                   "standings": [{k: s.get(k) for k in ("position", "name", "race_time", "crashes",
                                                                         "status")} for s in gp.standings]})

    # 2. ch5 on briefing_demo: one practice lap, official plan budget - does the LLM read the yellow flag?
    os.environ["LLM_LOG_DIR"] = str(root / "briefing")
    llm.reset_llm_stats()
    res = run_session(load_driver(str(REPO / f"{CP}/ch5_strategist.py")),
                      SessionConfig(track="briefing_demo", attempts=1, phases=["practice"], llm_policy="official",
                                    reflect_after_last=False))
    result["briefing"] = {"attempt": _attempt_row(res.attempts[0]), "llm": res.llm,
                          "per_call": _lat([r["seconds"] for r in _log_records(root / "briefing")])}

    # 3. the ch4 pilot: turn-based, 20 LLM ticks, then the Ch1 reflexes
    os.environ["LLM_LOG_DIR"] = str(root / "pilot")
    llm.reset_llm_stats()
    res = run_session(load_driver(str(REPO / f"{CP}/ch4_pilot.py")), SessionConfig(track="grand_prix", attempts=1))
    recs = _log_records(root / "pilot")
    result["pilot"] = {"attempt": _attempt_row(res.attempts[0]), "llm": res.llm,
                       "per_call": _lat([r["seconds"] for r in recs]),
                       "unusable": sum(1 for r in recs if r.get("error"))}
    os.environ.pop("LLM_LOG_DIR", None)
    return result


def _log_records(folder: Path) -> list[dict]:
    out = []
    for f in folder.glob("*.jsonl"):
        out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in out if not r.get("cached")]


# ============================================================================ Markdown
def fmt_t(a: dict | None) -> str:
    if not a:
        return "-"
    s = f"{a['time']:.2f}" if a["completed"] else f"DNF ({a['gates']} gates)"
    extra = []
    if a["crashes"]:
        extra.append(f"{a['crashes']}c")
    if a["zone_violations"]:
        extra.append(f"{a['zone_violations']}z")
    return s + (f" ({', '.join(extra)})" if extra else "")


def best(atts: list[dict]) -> dict | None:
    done = [a for a in atts if a["completed"]]
    return min(done, key=lambda a: a["time"]) if done else (atts[-1] if atts else None)


def cell(row: dict) -> str:
    atts = row["attempts"]
    if not atts:
        return f"ERROR {row.get('error') or ''}"[:60]
    if len(atts) == 1:
        return fmt_t(atts[0])
    return f"{fmt_t(atts[0])} -> {fmt_t(best(atts))}"


def lookup(rows: list[dict], label: str, track: str, **kw) -> dict | None:
    for r in rows:
        if r["label"] == label and r["track"] == track and all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def clean_everywhere(rows: list[dict], label: str, tracks_: list[str]) -> bool:
    for t in tracks_:
        r = lookup(rows, label, t)
        if not r or not r["attempts"] or not all(a["completed"] and a["crashes"] == 0 for a in r["attempts"]):
            return False
    return True


def render(data: dict) -> str:
    L: list[str] = []
    w = L.append
    meta = data.get("meta", {})
    bulk = data.get("bulk") or {}
    rows = bulk.get("rows", [])
    trk = bulk.get("tracks", tracks(SEEDS))
    seeds_ = [t for t in trk if t.startswith("random:")]
    w("# Calibration: measured numbers for the lesson")
    w("")
    w("Every number the chapters, the checkpoint bands and the instructor docs quote comes from this page. "
      "It is generated: do not edit it by hand, re-measure instead.")
    w("")
    w("```")
    w("python lesson/instructor/calibrate.py                 # mock LLM: drivers, handrails, curves, cliff, fly (~5 min)")
    w("python lesson/instructor/calibrate.py --real-llm      # + MiniMax: Max, ch5, ch6, ch4 pilot (~40-60 calls)")
    w("python lesson/instructor/calibrate.py --parts real    # only the MiniMax part (keeps the other numbers)")
    w("```")
    w("")
    w(f"Raw data: [`calibration.json`](calibration.json). Measured on {meta.get('platform', '?')}, "
      f"Python {meta.get('python', '?')}, commit `{meta.get('git_commit', '?')}`. Sections: "
      + ", ".join(f"{k} {v}" for k, v in (meta.get("measured") or {}).items()) + ".")
    w("")
    w("How to read the times: one-lap attempts, default **marshal** crash rule, race time = sim time + 5 s per "
      "crash + 1 s per tick too fast in a yellow-flag zone. `(2c)` = 2 crashes, `(2z)` = 2 zone violations "
      "(both already inside the time). `a -> b` = attempt 1 -> best attempt. Code drivers are deterministic: "
      "students get exactly these numbers. LLM drivers in the big tables ran with `LLM_PROVIDER=mock` "
      "(= their code fallback); the MiniMax numbers are in their own section.")
    w("")

    # ------------------------------------------------------------------ bands
    w("## Checkpoint bands (CP0-CP8)")
    w("")
    w("What a student should see at each checkpoint, and what the handrail file in `lesson/checkpoints/` "
      "measures. Chapters quote this table. \"Clean\" = finished with 0 crashes.")
    w("")
    w("| CP | Chapter | Handrail | Band (what counts as on track) | Handrail measured | OK |")
    w("|---|---|---|---|---|---|")
    for b in bands(data):
        w(f"| {b['cp']} | {b['chapter']} | `{b['file']}` | {b['band']} | {b['measured']} | {'yes' if b['ok'] else '**NO**'} |")
    w("")

    # ------------------------------------------------------------------ big table
    if rows:
        w("## Lap times: every driver on every track (mock LLM)")
        w("")
        w(f"Public tracks + random seeds {', '.join(s.split(':')[1] for s in seeds_)}. Learners "
          "(l3, l6, l6b, ch3, ch5, ch6): 4 attempts, shown as attempt 1 -> best.")
        w("")
        w("| Driver | " + " | ".join(trk) + " |")
        w("|---|" + "---|" * len(trk))
        for lab, _f, group, _n in DRIVERS:
            cells = [cell(r) if (r := lookup(rows, lab, t)) else "-" for t in trk]
            name = f"**{lab}**" if group == "handrail" else lab
            w(f"| {name} | " + " | ".join(cells) + " |")
        w("")
        w("### Generalisation summary (best attempt)")
        w("")
        w("| Driver | clean on public tracks | mean public (s) | clean on random seeds | mean random (s) |")
        w("|---|---|---|---|---|")
        for lab, _f, _g, _n in DRIVERS:
            pub = [lookup(rows, lab, t) for t in PUBLIC]
            rnd = [lookup(rows, lab, t) for t in seeds_]
            w(f"| {lab} | {_clean_count(pub)} | {_mean(pub)} | {_clean_count(rnd)} | {_mean(rnd)} |")
        w("")
        w("### House cars")
        w("")
        w("`--house` adds them to a grand prix; they race but score no points. Same numbers as above "
          "(Max under the mock LLM = its code plan on the racing line).")
        w("")
        w("| House car | File | " + " | ".join(trk) + " |")
        w("|---|---|" + "---|" * len(trk))
        lab_of = {f: lab for lab, f, _g, _n in DRIVERS}
        for name, f in HOUSE:
            w(f"| {name} | `{f}` | " + " | ".join(cell(r) if (r := lookup(rows, lab_of[f], t)) else "-"
                                                 for t in trk) + " |")
        w("")

    # ------------------------------------------------------------------ curves
    curves = data.get("curves")
    if curves:
        w("## Learning curves (6 attempts, mock LLM)")
        w("")
        w("`python race.py drive <file> --track <track> --attempts 6`. Gain = (attempt 1 - best) / attempt 1.")
        w("")
        w("| Driver | Track | Attempts 1..6 | Best | Gain | Clean |")
        w("|---|---|---|---|---|---|")
        for r in curves["rows"]:
            a = r["attempts"]
            b = best(a)
            gain = _gain(a)
            w(f"| {r['label']} | {r['track']} | {' / '.join(fmt_t(x) for x in a)} | {fmt_t(b)} | "
              f"{gain} | {'yes' if all(x['completed'] and not x['crashes'] for x in a) else 'no'} |")
        w("")
        w("### Learning gone wrong (the keep-best ablation)")
        w("")
        w("Keep-best **off** = ch3_learner whose reflect() forgets how good its best lap was, so every lap "
          "becomes the starting point of the next experiment (no revert, no smaller step after a bad change). "
          "\"greedy\" = `--param push=1.3 --param crash_cut=1.0` (big pushes, crashes not punished). The "
          "\"last\" column is what a learner without lock-in would take into qualifying. Where the on/off rows "
          "are identical, every change was an improvement, so keep-best never had to act. The PathFollower's "
          "stopping-distance safety net keeps even greedy knobs out of crash loops: here \"learning gone "
          "wrong\" means ending on a worse lap than the best one, not a DNF.")
        w("")
        w("| Knobs | Keep-best | Track | Attempts 1..6 | Best | Last |")
        w("|---|---|---|---|---|---|")
        default = [r for r in curves.get("rows", []) if r["label"] == "ch3_learner" and r["track"] in ("grand_prix", "sprint")]
        for r in sorted([dict(r, variant="default", keep_best=True) for r in default]
                        + curves.get("greedy_with_keep_best", []) + curves.get("no_keep_best", []),
                        key=lambda r: (r["track"], r["variant"], not r.get("keep_best"))):
            a = r["attempts"]
            w(f"| {r['variant']} | {'on' if r.get('keep_best') else '**off**'} | {r['track']} | "
              f"{' / '.join(fmt_t(x) for x in a)} | {fmt_t(best(a))} | {fmt_t(a[-1]) if a else '-'} |")
        w("")

    # ------------------------------------------------------------------ cliff
    cliff = data.get("cliff")
    if cliff:
        w("## The latency cliff (realtime mode)")
        w("")
        w("A wrapper driver sleeps `think` seconds inside every `drive()` call, raced with "
          "`SessionConfig(realtime=True)` (= `race.py drive --realtime`): while it thinks, the car keeps "
          "executing its previous command for floor(think / 0.5) ticks, then the (stale) new command is applied. "
          "Track grand_prix, tick limit 500. Blind distance = speed x think time "
          "(3 s at 30 u/s = 90 units = 5 track widths). The marshal crash rule puts the car back on track "
          "after every crash, which is why some blind runs still reach the line - with dozens of crashes "
          "and several times the lap time. Which latency DNFs and which limps home is noisy; the cliff "
          "between 0 s and 0.5 s is not.")
        w("")
        w("| Driver | think (s) | blind ticks per decision | Result | Gates | Late ticks | Wall time (s) |")
        w("|---|---|---|---|---|---|---|")
        for r in cliff["rows"]:
            a = r["attempt"]
            w(f"| {r['label']} | {r['think']:g} | {int(r['think'] / 0.5)} | {fmt_t(a)} | {a['gates']} | "
              f"{a['late_ticks']} | {r['wall_s']:.0f} |")
        w("")

    # ------------------------------------------------------------------ fly
    fly = data.get("fly")
    if fly:
        w("## Fly brains: real wiring vs scrambled controls (no LLM)")
        w("")
        w("`--param SCRAMBLE=<seed>` keeps every WIRING row (and, for the connectome, every neuron's number of "
          "connections) but sends it to a random target. A brain that beats its scrambled copies shows that the "
          "wiring carries the skill. DNF shows the gates reached (of 16).")
        w("")
        w("| Driver | Track | real wiring | SCRAMBLE=0 | 1 | 2 | 3 | 4 |")
        w("|---|---|---|---|---|---|---|---|")
        seen = []
        for r in fly["rows"]:
            key = (r["label"], r["track"])
            if key in seen:
                continue
            seen.append(key)
            cells = [cell(x) if (x := lookup(fly["rows"], r["label"], r["track"], scramble=s)) else "-"
                     for s in [None, 0, 1, 2, 3, 4]]
            w(f"| {r['label']} | {r['track']} | " + " | ".join(cells) + " |")
        if not fly.get("connectome_data"):
            w("")
            w("fly_connectome: not measured (no `data/fly/steering_circuit_783.npz`; "
              "instructor: `python -m agentic_gp.fly.extract`).")
        w("")

    # ------------------------------------------------------------------ real
    real = data.get("real")
    w("## With the real LLM (MiniMax)")
    w("")
    if not real:
        w("Not measured yet: `python lesson/instructor/calibrate.py --parts real`.")
    elif real.get("error"):
        w(f"Not measured: {real['error']}")
    else:
        w(f"Provider `{real['provider']}`. Official weekend format, the same as an event: `plan()` -> 3 practice laps "
          "with `reflect()` -> qualifying lap (plan locked in) -> 1-lap race; official LLM rules (<= 2 calls in "
          "plan, <= 2 per practice reflect, none in drive / qualifying / race). Drivers run in parallel child "
          "processes like a real grand prix. LLM answers vary from run to run: treat these as one sample.")
        w("")
        w("| Track | Driver | Practice 1 -> 2 -> 3 | Qualifying | Race | LLM calls | Tokens (prompt + completion) "
          "| LLM time (s) | per call p50 / p95 / max (s) |")
        w("|---|---|---|---|---|---|---|---|---|")
        for wk in real["weekends"]:
            for d in wk["drivers"]:
                pr = [a for a in d["attempts"] if a["phase"] == "practice"]
                q = next((a for a in d["attempts"] if a["phase"] == "qualifying"), None)
                race = next((a for a in d["attempts"] if a["phase"] == "race"), None)
                llm_ = d["llm"]
                pc = d.get("per_call") or {}
                w(f"| {wk['track']} | {d['name']} | {' -> '.join(fmt_t(a) for a in pr)} | {fmt_t(q)} | {fmt_t(race)} | "
                  f"{llm_.get('calls')} | {llm_.get('prompt_tokens')} + {llm_.get('completion_tokens')} | "
                  f"{llm_.get('seconds')} | {pc.get('p50_s')} / {pc.get('p95_s')} / {pc.get('max_s')} |")
        w("")
        w("Wall time per weekend (3 cars in parallel): " + ", ".join(f"{wk['track']} {wk['wall_s']:.0f} s"
                                                                    for wk in real["weekends"]) + ".")
        w("")
        b = real.get("briefing")
        if b:
            pc = b["per_call"]
            w(f"**ch5 on briefing_demo** (one practice lap, official plan budget): {fmt_t(b['attempt'])}, "
              f"{b['llm'].get('calls')} LLM call(s), {b['llm'].get('prompt_tokens')} + "
              f"{b['llm'].get('completion_tokens')} tokens, {pc.get('total_s')} s of LLM time. "
              + ("0 zone violations: it read the yellow flag from the briefing (the mock gets 2)."
                 if b["attempt"]["zone_violations"] == 0 else "It missed the yellow flag this time."))
            w("")
        p = real.get("pilot")
        if p:
            pc = p["per_call"]
            w(f"**ch4 pilot** (turn-based, grand_prix, MAX_LLM_TICKS 20, `effort=\"none\"`): {fmt_t(p['attempt'])}; "
              f"{p['llm'].get('calls')} LLM calls, per call p50 {pc.get('p50_s')} s / p95 {pc.get('p95_s')} s / "
              f"max {pc.get('max_s')} s; {p['llm'].get('prompt_tokens')} + {p['llm'].get('completion_tokens')} "
              f"tokens. Ticks per call at p50 = {(pc.get('p50_s') or 0) / 0.5:.1f}: in realtime the car would "
              f"drive that many ticks blind per decision.")
            w("")
    return "\n".join(L) + "\n"


def _clean_count(rs: list) -> str:
    ok = sum(1 for r in rs if r and r["attempts"] and best(r["attempts"])["completed"]
             and best(r["attempts"])["crashes"] == 0)
    return f"{ok}/{len(rs)}"


def _mean(rs: list) -> str:
    ts = [best(r["attempts"])["time"] for r in rs if r and r["attempts"] and best(r["attempts"])["completed"]]
    return f"{statistics.mean(ts):.1f}" + ("" if len(ts) == len(rs) else f" ({len(rs) - len(ts)} DNF)") if ts else "-"


def _gain(atts: list[dict]) -> str:
    b = best(atts)
    if not atts or not atts[0]["completed"] or not b or not b["completed"]:
        return "-"
    return f"{100 * (atts[0]['time'] - b['time']) / atts[0]['time']:.1f} %"


# ---------------------------------------------------------------------------- bands
def bands(data: dict) -> list[dict]:
    rows = (data.get("bulk") or {}).get("rows", [])
    trk = (data.get("bulk") or {}).get("tracks", [])
    curves = (data.get("curves") or {}).get("rows", [])
    fly = (data.get("fly") or {}).get("rows", [])
    real = data.get("real") or {}
    out = []

    def a1(label: str, track: str = "grand_prix") -> dict | None:
        r = lookup(rows, label, track)
        return r["attempts"][0] if r and r["attempts"] else None

    def bst(label: str, track: str = "grand_prix") -> dict | None:
        r = lookup(rows, label, track)
        return best(r["attempts"]) if r and r["attempts"] else None

    def n_clean(label: str, ts: list[str]) -> str:
        return f"{sum(1 for t in ts if clean_everywhere(rows, label, [t]))}/{len(ts)}"

    rnd = [t for t in trk if t.startswith("random:")]
    pub_no_brief = [t for t in PUBLIC if t != "briefing_demo"]

    # CP0
    x = a1("ch0_pit_lane")
    out.append({"cp": "CP0", "chapter": "Ch0 Pit Lane", "file": "ch0_pit_lane.py",
                "band": "`race.py check` PASSED; your car in the viewer; grand_prix finishes (the template drives "
                        "slowly but cleanly)",
                "measured": f"grand_prix {fmt_t(x)}; clean on {n_clean('ch0_pit_lane', PUBLIC)} public tracks",
                "ok": bool(x and x["completed"])})
    # CP1
    x = a1("ch1_reflex")
    ok = bool(x and x["completed"] and x["crashes"] == 0 and 60 <= x["time"] <= 80) and \
        clean_everywhere(rows, "ch1_reflex", pub_no_brief)
    out.append({"cp": "CP1", "chapter": "Ch1 Reflexes", "file": "ch1_reflex.py",
                "band": "grand_prix 60-80 s **clean**; clean on every public track. Silver < 75 s, Gold < 55 s and "
                        "clean on sprint",
                "measured": f"grand_prix {fmt_t(x)}, sprint {fmt_t(a1('ch1_reflex', 'sprint'))}; clean on "
                            f"{n_clean('ch1_reflex', PUBLIC)} public, {n_clean('ch1_reflex', rnd)} random",
                "ok": ok})
    # CP2
    x = a1("ch2_planner")
    ok = bool(x and x["completed"] and x["crashes"] == 0 and 48 <= x["time"] <= 55) and \
        clean_everywhere(rows, "ch2_planner", PUBLIC + rnd)
    l2b = a1("l2b_racing_line")
    out.append({"cp": "CP2", "chapter": "Ch2 The Map", "file": "ch2_planner.py",
                "band": "grand_prix 48-55 s clean; clean on every public AND random track without retuning "
                        "(briefing_demo still collects zone penalties: no briefing reader yet). Gold: racing line",
                "measured": f"grand_prix {fmt_t(x)}; clean on {n_clean('ch2_planner', PUBLIC)} public, "
                            f"{n_clean('ch2_planner', rnd)} random; random mean "
                            f"{_mean([lookup(rows, 'ch2_planner', t) for t in rnd])} s (Gold reference "
                            f"l2b_racing_line {fmt_t(l2b)})",
                "ok": ok})
    # CP3
    r = lookup(curves, "ch3_learner", "sprint")
    a = r["attempts"] if r else []
    g = _gain(a)
    ok = bool(a) and all(x["completed"] for x in a) and g != "-" and float(g.split()[0]) >= 3.0
    gp = lookup(curves, "ch3_learner", "grand_prix")
    out.append({"cp": "CP3", "chapter": "Ch3 Practice", "file": "ch3_learner.py",
                "band": "`--track sprint --attempts 6`: best >= 3 % faster than attempt 1, no DNF, clean best",
                "measured": (f"sprint {fmt_t(a[0])} -> {fmt_t(best(a))} ({g}); grand_prix "
                             f"{fmt_t(gp['attempts'][0])} -> {fmt_t(best(gp['attempts']))} ({_gain(gp['attempts'])})")
                if a and gp else "-",
                "ok": ok})
    # CP4
    x = a1("ch4_pilot")
    p = real.get("pilot") if isinstance(real, dict) else None
    pm = (f"; MiniMax: {p['llm'].get('calls')} calls, p50 {p['per_call'].get('p50_s')} s / p95 "
          f"{p['per_call'].get('p95_s')} s, lap {fmt_t(p['attempt'])}") if p else "; MiniMax: not measured"
    out.append({"cp": "CP4", "chapter": "Ch4 MiniMax", "file": "ch4_pilot.py",
                "band": "not raced: exam table + p50/p95 of your calls + one sentence on where the LLM belongs; "
                        "<= 20 LLM ticks per run; with the mock LLM the reflexes drive the whole lap",
                "measured": f"mock: grand_prix {fmt_t(x)} (= the Ch1 reflexes){pm}",
                "ok": bool(x and x["completed"])})
    # CP5
    x5, x3 = bst("ch5_strategist"), bst("ch3_learner")
    b = real.get("briefing") if isinstance(real, dict) else None
    real5 = _real_driver(real, "grand_prix", "CP5 Strategist")
    bm = (f"; MiniMax briefing_demo {fmt_t(b['attempt'])} with {b['llm'].get('calls')} call(s)") if b else ""
    rm = (f"; MiniMax grand_prix qualifying {fmt_t(_phase(real5, 'qualifying'))}") if real5 else ""
    ok = bool(x5 and x3 and x5["completed"] and abs(x5["time"] - x3["time"]) <= 0.05 * x3["time"]) and \
        (b is None or b["attempt"]["zone_violations"] == 0)
    out.append({"cp": "CP5", "chapter": "Ch5 Strategist", "file": "ch5_strategist.py",
                "band": "within +-5 % of your own Ch3 time; 0 zone violations on briefing_demo (real LLM); "
                        "<= 3 LLM calls in plan(); still finishes with `LLM_PROVIDER=mock`",
                "measured": f"mock = physics plan: grand_prix best-of-4 {fmt_t(x5)} (ch3: {fmt_t(x3)}){bm}{rm}",
                "ok": ok})
    # CP6
    r6 = lookup(rows, "ch6_crew_chief", rnd[0]) if rnd else None
    a6 = r6["attempts"] if r6 else []
    real6 = _real_driver(real, REAL_TRACKS[1], "CP6 Crew Chief")
    rm = ""
    if real6:
        pr = [x for x in real6["attempts"] if x["phase"] == "practice"]
        rm = (f"; MiniMax {REAL_TRACKS[1]}: practice {' -> '.join(fmt_t(x) for x in pr)}, qualifying "
              f"{fmt_t(_phase(real6, 'qualifying'))}, {real6['llm'].get('calls')} calls")
    ok = bool(a6) and all(x["completed"] for x in a6) and best(a6)["time"] < a6[0]["time"]
    out.append({"cp": "CP6", "chapter": "Ch6 Crew Chief", "file": "ch6_crew_chief.py",
                "band": "improving curve over 4 attempts on a `random:<seed>` track; <= 3 LLM calls per reflect "
                        "(official: 2); finishes with the mock LLM (then the Ch3 rules learn)",
                "measured": (f"mock {rnd[0]}: {' -> '.join(fmt_t(x) for x in a6)}{rm}" if a6 else "-"),
                "ok": ok})
    # CP7
    fx = lookup(fly, "ch7_fly_fixed", "grand_prix", scramble=None)
    scr = [lookup(fly, "ch7_fly_fixed", "grand_prix", scramble=s) for s in range(5)]
    scr_dnf = sum(1 for s in scr if s and s["attempts"] and not s["attempts"][0]["completed"])
    con = lookup(fly, "fly_connectome", "oval", scramble=None)
    con_s = [lookup(fly, "fly_connectome", "oval", scramble=s) for s in range(5)]
    cm = ""
    if con:
        cm = (f"; connectome oval real {cell(con)}, scrambled: "
              + ", ".join(cell(s) for s in con_s if s))
    fa = fx["attempts"][0] if fx and fx["attempts"] else None
    out.append({"cp": "CP7", "chapter": "Ch7 Fly Brains", "file": "ch7_fly_fixed.py",
                "band": "toy fly finishes a lap (fixed fly = exactly fly_toy, mirror self-check SYMMETRIC); "
                        "scrambled wiring: DNF or far slower; connectome real wiring >= 8 of 16 gates",
                "measured": f"grand_prix {fmt_t(fa)}; SCRAMBLE 0-4: {scr_dnf}/5 DNF{cm}",
                "ok": bool(fa and fa["completed"] and fa["crashes"] == 0)})
    # CP8
    ok = clean_everywhere(rows, "ch6_crew_chief", pub_no_brief) and bool(a6)
    out.append({"cp": "CP8", "chapter": "Ch8 Garage", "file": "ch6_crew_chief.py",
                "band": "final file: `check` green, clean on every public track, finishes with "
                        "`LLM_PROVIDER=mock` (`validate --random 3` posts a score)",
                "measured": f"ch6 (mock) clean on {n_clean('ch6_crew_chief', PUBLIC)} public, "
                            f"{n_clean('ch6_crew_chief', rnd)} random; public mean "
                            f"{_mean([lookup(rows, 'ch6_crew_chief', t) for t in PUBLIC])} s",
                "ok": ok})
    return out


def _real_driver(real: dict, track: str, name: str) -> dict | None:
    for wk in (real or {}).get("weekends", []) if isinstance(real, dict) else []:
        if wk["track"] == track:
            return next((d for d in wk["drivers"] if d["name"] == name), None)
    return None


def _phase(d: dict, phase: str) -> dict | None:
    return next((a for a in d["attempts"] if a["phase"] == phase), None)


# ============================================================================ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure every lesson driver; write CALIBRATION.md + calibration.json")
    ap.add_argument("--parts", default="bulk,curves,cliff,fly",
                    help="comma list of bulk,curves,cliff,fly,real (default: all but real)")
    ap.add_argument("--real-llm", action="store_true", help="also run the real-LLM part (MiniMax key from .env)")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="random track seeds (default 7,12,21)")
    ap.add_argument("--jobs", type=int, default=max(2, min(8, (os.cpu_count() or 4) - 1)))
    ap.add_argument("--render-only", action="store_true", help="only re-write CALIBRATION.md from calibration.json")
    ap.add_argument("--_real-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args._real_child:                        # the real-LLM part runs here, in its own process
        print(json.dumps(part_real()))
        return 0

    data = json.loads(JSON_OUT.read_text(encoding="utf-8")) if JSON_OUT.exists() else {}
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    if args.real_llm and "real" not in parts:
        parts.append("real")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    data.setdefault("meta", {}).setdefault("measured", {})
    if not args.render_only:
        _mock_env()
        stamp = dt.datetime.now().strftime("%Y-%m-%d")
        for part in parts:
            t0 = time.perf_counter()
            print(f"[calibrate] {part} ...", flush=True)
            if part == "bulk":
                data["bulk"] = part_bulk(seeds, args.jobs)
            elif part == "curves":
                data["curves"] = part_curves(seeds, args.jobs)
            elif part == "cliff":
                data["cliff"] = part_cliff(args.jobs)
            elif part == "fly":
                data["fly"] = part_fly(args.jobs)
            elif part == "real":
                env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_real-child"], cwd=REPO,
                                      env=env, capture_output=True, text=True, encoding="utf-8")
                sys.stderr.write(proc.stderr[-3000:])
                last = [x for x in proc.stdout.splitlines() if x.startswith("{")]
                data["real"] = json.loads(last[-1]) if last else {"error": f"real part failed (exit {proc.returncode})"}
            else:
                print(f"unknown part {part!r}", file=sys.stderr)
                return 2
            data["meta"]["measured"][part] = f"{stamp} ({time.perf_counter() - t0:.0f} s)"
            print(f"[calibrate] {part} done in {time.perf_counter() - t0:.0f} s", flush=True)
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                    text=True).stdout.strip()
        except OSError:
            commit = "?"
        data["meta"].update({"format": "agp-calibration", "version": 1, "git_commit": commit or "?",
                             "python": platform.python_version(),
                             "platform": f"{platform.system()} {platform.release()}", "seeds": seeds})
        # merge into the file as it is NOW (another calibrate run may have written other parts meanwhile)
        disk = json.loads(JSON_OUT.read_text(encoding="utf-8")) if JSON_OUT.exists() else {}
        disk.setdefault("meta", {}).setdefault("measured", {})
        for part in parts:
            disk[part] = data[part]
            disk["meta"]["measured"][part] = data["meta"]["measured"][part]
        disk["meta"].update({k: v for k, v in data["meta"].items() if k != "measured"})
        data = disk
        JSON_OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    MD_OUT.write_text(render(data), encoding="utf-8")
    print(f"wrote {MD_OUT.relative_to(REPO)} and {JSON_OUT.relative_to(REPO)}")
    for b in bands(data):
        print(f"  {b['cp']}: {'ok ' if b['ok'] else 'NO '} {b['measured']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
