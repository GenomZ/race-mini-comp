"""(Re)generate the demo data used by the Streamlit app and the Streamlit lessons.

    python races/demo/make_demo.py                 # everything (about a minute with code drivers)
    python races/demo/make_demo.py --only class_demo crash_loop
    python races/demo/make_demo.py --real-llm      # LLM drivers use the real key (default: mock)

It uses whatever drivers exist when you run it (drivers/examples/*.py), so integration can simply
re-run it at the end. Every file is small (tens to hundreds of KB) and written to races/demo/:

  l2_session.json          SessionResult.to_dict() of l2_planner on grand_prix, 3 attempts
                           (reports with telemetry, segments, crash_log, summary_text)
  l2_session_replay.json   the same session as a replay, one ghost car per attempt
  l2_ticks.json            one attempt tick by tick: the obs the driver saw, its action and debug
                           dict (Sensor X-ray, live-view demos)
  learner_session.json     (+ _replay) l3_learner, 5 attempts on sprint - only if l3_learner exists
  class_demo.json          multi-car race replay: the example drivers (no l0_dummy, no flies) on grand_prix, 2 laps
  class_demo_qualifying.json, class_demo_results.json, class_demo_standings.md
  crash_loop.json          l0_dummy under the CLASSIC crash rule (stop + nudge back): the crash
                           loop that the marshal rule fixes (chapter 1)
  html/class_demo.html     standalone HTML player of class_demo.json (open it in a browser)
  llm_logs/class_demo/     LLM traces (JSONL) of the LLM drivers in the class demo (mock by default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXAMPLES = ROOT / "drivers" / "examples"
TRACK = "grand_prix"
# Not raced in the class demo: the open-loop dummy and the fly (they would add 5-10 minutes of DNF to the
# replay; the dummy has its own crash_loop file, the fly its Fly Lab) and drivers that need data files.
CLASS_DEMO_SKIP = {"l0_dummy.py"}            # + every fly_*.py (see make_class_demo)


def _say(msg: str) -> None:
    print(msg, flush=True)


def _write_json(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False, default=_json_default), encoding="utf-8")
    _say(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size // 1024} KB)")
    return path


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


# ------------------------------------------------------------------------------ the demos
def make_l2_session(out: Path) -> None:
    from agentic_gp.race.session import SessionConfig, iter_session

    driver = EXAMPLES / "l2_planner.py"
    if not driver.exists():
        _say("  (skipped: drivers/examples/l2_planner.py not found)")
        return
    ticks, result = [], None
    for ev in iter_session(str(driver), SessionConfig(track=TRACK, attempts=3)):
        if ev.type == "tick" and ev.attempt == 1:
            f = ev.data
            act = f["action"]
            ticks.append({"tick": int(f["obs"]["tick"]), "obs": f["obs"],
                          "action": {"throttle": round(act.throttle, 4), "steering": round(act.steering, 4),
                                     "say": act.say},
                          "debug": f.get("debug"), "think_seconds": f.get("think_seconds"),
                          "after": {k: round(float(f[k]), 3) for k in ("x", "y", "heading_deg", "speed", "t")}
                          | {"crashed": bool(f.get("crashed")), "events": list(f.get("events", []))}})
        elif ev.type == "session_end":
            result = ev.data
    _write_json(out / "l2_session.json", result.to_dict())
    result.to_replay("all", title="Planner - 3 attempts").to_json(out / "l2_session_replay.json")
    _say(f"  wrote races/demo/l2_session_replay.json")
    _write_json(out / "l2_ticks.json", {"format": "agentic-gp-ticks", "version": 1, "track": TRACK,
                                        "driver": result.driver, "attempt": 1, "ticks": ticks})


def make_learner_session(out: Path) -> None:
    from agentic_gp.race.session import SessionConfig, run_session

    driver = EXAMPLES / "l3_learner.py"
    if not driver.exists():
        _say("  (skipped: drivers/examples/l3_learner.py not found yet)")
        return
    result = run_session(str(driver), SessionConfig(track="sprint", attempts=5))
    _write_json(out / "learner_session.json", result.to_dict())
    result.to_replay("all", title="Learner - 5 attempts on sprint").to_json(out / "learner_session_replay.json")
    _say("  wrote races/demo/learner_session_replay.json")


def make_class_demo(out: Path, real_llm: bool) -> None:
    from agentic_gp.race.grand_prix import DriverEntry, GrandPrixConfig, run_grand_prix
    from agentic_gp.race.leaderboard import standings_markdown

    files = [f for f in sorted(EXAMPLES.glob("*.py"))
             if not f.name.startswith(("_", "fly_")) and f.name not in CLASS_DEMO_SKIP]
    entries = [DriverEntry(f, house="crew_chief" in f.stem) for f in files]
    variants = []
    if len(entries) < 7 and (EXAMPLES / "l2_planner.py").exists():
        # Few drivers yet: add two "students" (the planner with other settings) so the race is a race.
        for aggr, label in ((0.92, "Careful"), (1.12, "Brave")):
            entries.append(DriverEntry(EXAMPLES / "l2_planner.py", params={"aggression": aggr}))
            variants.append(label)
    if not entries:
        _say("  (skipped: no drivers in drivers/examples)")
        return
    log_dir = out / "llm_logs" / "class_demo"
    for old in log_dir.glob("*.jsonl"):        # logs are appended: start clean, or two runs mix in one file
        old.unlink()
    # 2 practice laps: the learner and the LangGraph pit crew only learn when a practice lap follows (lock-in)
    cfg = GrandPrixConfig(track=TRACK, practice_attempts=2, race_laps=2, title="Class demo - Grand Prix circuit",
                          llm_policy="official", llm_log_dir=str(log_dir))
    _say(f"  racing {len(entries)} cars in-process ({', '.join(e.source.stem if hasattr(e.source, 'stem') else str(e.source) for e in entries)})")
    gp = run_grand_prix(entries, cfg, inprocess=True)
    # Rename the planner variants ("planner-2", "planner-3") so the timing tower shows who is who.
    names = {f"planner-{k + 2}": label for k, label in enumerate(variants)}
    colors = {"Careful": "#e87ba4", "Brave": "#eda100"}
    for rep in (gp.replay, gp.qualifying_replay):
        for c in (rep.cars if rep else []):
            if c.id in names:
                c.name, c.color = names[c.id], colors[names[c.id]]
    for row in gp.standings + gp.qualifying:
        if row.get("id") in names:
            row["name"] = names[row["id"]]
    paths = gp.save(out, "class_demo")
    for p in paths.values():
        _say(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")
    (out / "class_demo_standings.md").write_text(standings_markdown(gp), encoding="utf-8")
    from agentic_gp.viz.player import save_replay_html
    html = save_replay_html(paths["replay"], out / "html" / "class_demo.html")
    _say(f"  wrote {html.relative_to(ROOT)}  ({html.stat().st_size // 1024} KB)")
    for row in gp.standings:
        rt = row.get("race_time")
        _say(f"    P{row['position']:<2} {row['name']:<14} {rt if rt is not None else row.get('status')}")


def make_crash_loop(out: Path) -> None:
    from agentic_gp.race.session import SessionConfig, run_session
    from agentic_gp.sim.config import PhysicsConfig

    driver = EXAMPLES / "l0_dummy.py"
    if not driver.exists():
        _say("  (skipped: drivers/examples/l0_dummy.py not found)")
        return
    cfg = SessionConfig(track=TRACK, attempts=1, max_ticks_per_lap=240, physics=PhysicsConfig(crash_rule="classic"))
    result = run_session(str(driver), cfg)
    rep = result.to_replay("best", title="Crash loop - classic crash rule (stop + nudge back)")
    rep.to_json(out / "crash_loop.json")
    a = result.attempts[0].result
    _say(f"  wrote races/demo/crash_loop.json  ({a.crashes} crashes in {a.ticks} ticks, {a.waypoints_passed} gates)")


DEMOS = {"l2_session": make_l2_session, "learner_session": make_learner_session, "class_demo": make_class_demo,
         "crash_loop": make_crash_loop}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", choices=sorted(DEMOS), help="build only these demos")
    ap.add_argument("--out", default=str(HERE), help="output folder (default races/demo)")
    ap.add_argument("--real-llm", action="store_true", help="LLM drivers call the real provider (default: mock)")
    args = ap.parse_args(argv)
    if not args.real_llm:
        os.environ["LLM_PROVIDER"] = "mock"         # demo data must never depend on the network
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(DEMOS):
        t0 = time.perf_counter()
        _say(f"[{name}]")
        fn = DEMOS[name]
        try:
            fn(out, args.real_llm) if name == "class_demo" else fn(out)
        except Exception as exc:                     # noqa: BLE001 - one broken demo must not stop the rest
            _say(f"  FAILED: {type(exc).__name__}: {exc}")
        _say(f"  ({time.perf_counter() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
