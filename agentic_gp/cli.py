"""The `race.py` command line (also `python -m agentic_gp`).

    python race.py drive DRIVER.py [--track grand_prix] [--attempts 1] [--laps 1] [--realtime] [--quiet]
                         [--plot PNG] [--replay JSON] [--html HTML] [--unity] [--save] [--param key=value]
                         [--dump-states FILE]
    python race.py check DRIVER.py                         # static check + 40-tick smoke test (exit 1 on errors)
    python race.py validate DRIVER.py [--random 2]         # every public track + random tracks -> score
    python race.py exam DRIVER.py --states FILE [--n 20] [--reference drivers/examples/l1_reflex.py]
    python race.py grand-prix PATH... [--track ..] [--practice 2] [--laps 3] [--jobs N] [--house]
                         [--out races/final.json] [--html races/final.html] [--unity] [--inprocess]
    python race.py final PATH... --home grand_prix --mystery TRACKSPEC [--house] [--out races/final]
    python race.py watch REPLAY.json [--unity | --html [OUT.html]]
    python race.py tracks [--plot DIR] [--random 3]
    python race.py leaderboard [--results results] [--out LEADERBOARD.md]
    python race.py doctor [--llm]                          # setup check -> doctor_report.txt
    python race.py llm-check [--load N --calls K --ramp S] # instructor: real MiniMax round trips + load test

Track specs (every --track / --home / --mystery): a registry key (grand_prix, oval, sprint, speedway,
briefing_demo), "random:<seed>", or a track file ("file:path.py", or --track-file path.py).
Every command prints plain text and never prints an API key. Mistakes (unknown track, bad --param
value, missing file) print "ERROR: ..." and exit 2 - set AGP_DEBUG=1 to see the Python traceback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
HOUSE_DRIVERS = ("drivers/examples/l6_llm_crew_chief.py", "drivers/examples/l2_planner.py",
                 "drivers/examples/fly_toy.py")
DEFAULT_REFERENCE = ("drivers/examples/l1_reflex.py", "drivers/examples/l2_planner.py")
DNF_SCORE_S = 300.0          # validate: a DNF counts as a 300 s lap
STRICT_HELP = "write strict replay schema v1 (without the additive zones/motto/anatomy/house fields)"


# ============================================================================ small helpers
def _utf8_console() -> None:
    """Windows consoles default to cp1252; driver names and radio messages may not fit in it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _secs(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _table(headers: Sequence[str], rows: list[Sequence[Any]]) -> str:
    """A plain aligned text table (numbers right-aligned)."""
    cells = [[str(h) for h in headers]] + [["" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    numeric = [all(_is_number(r[i]) for r in cells[1:] if r[i] not in ("", "-")) for i in range(len(headers))]

    def line(r: list[str]) -> str:
        return "  ".join(c.rjust(w) if num else c.ljust(w) for c, w, num in zip(r, widths, numeric)).rstrip()

    return "\n".join([line(cells[0]), line(["-" * w for w in widths])] + [line(r) for r in cells[1:]])


def _is_number(text: str) -> bool:
    try:
        float(text.rstrip("%s"))
        return True
    except ValueError:
        return False


def _track_spec(args: argparse.Namespace, attr: str = "track") -> str:
    """--track-file wins over --track (make_track also accepts "file:x.py" and paths ending in .py)."""
    if getattr(args, "track_file", None):
        return f"file:{args.track_file}"
    return getattr(args, attr)


def _physics(args: argparse.Namespace):
    if not getattr(args, "crash_rule", None):
        return None
    from .sim.config import PhysicsConfig
    return PhysicsConfig(crash_rule=args.crash_rule)


def _params(items: list[str] | None) -> dict:
    from .race.loader import parse_param_args
    return parse_param_args(items)


def _repo_path(rel: str) -> Path:
    """A repo file by its repo-relative name (works from any working directory)."""
    p = Path(rel)
    return p if p.exists() else REPO_ROOT / rel


def _save_html(replay: Any, path: str | Path) -> bool:
    """Standalone HTML replay player (agentic_gp.viz.player, written by the app team)."""
    try:
        from .viz.player import save_replay_html
    except ImportError:
        print("HTML player not installed yet (agentic_gp/viz/player.py is missing) - skipped --html.")
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_replay_html(replay, path)
    print(f"HTML replay: {path}  (open it in any browser)")
    return True


def _to_unity(replay_path: str | Path) -> None:
    from .viz.unity import show_in_unity
    show_in_unity(replay_path)


def _fail(message: str, code: int = 1) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return code


def _seconds_arg(text: str) -> float:
    """argparse type for durations: "30", "30s", "1.5 s", "2m", "500ms"."""
    t = str(text).strip().lower().replace(" ", "")
    for suffix, scale in (("ms", 0.001), ("min", 60.0), ("sec", 1.0), ("s", 1.0), ("m", 60.0)):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
            break
    else:
        scale = 1.0
    try:
        value = float(t) * scale
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"expected a duration, e.g. 30, 30s, 2m or 500ms, got {text!r}")
    return value


def _track_problem(spec: str) -> str | None:
    """Build the track once up front: a typo or a broken track file is one ERROR line, before any
    header is printed or any driver loads (None = the track is fine)."""
    from .sim.track import make_track
    try:
        make_track(spec)
    except SyntaxError as exc:
        return f"track file {exc.filename or spec}: syntax error on line {exc.lineno}: {exc.msg}"
    except (KeyError, ValueError, OSError, TypeError, AttributeError) as exc:
        return str(exc.args[0] if exc.args else exc)
    return None


def _warn_unknown_params(drv: Any) -> None:
    unknown = list(getattr(drv, "unknown_params", None) or [])
    if unknown:
        known = ", ".join(sorted(k for k in (getattr(drv, "params", None) or {}) if k not in unknown))             or "none - the driver has no PARAMS"
        print(f"WARNING: unknown PARAMS key(s) {', '.join(repr(k) for k in unknown)} - the driver does not use "
              f"them (known: {known})", file=sys.stderr)


def _result_file(results_dir: str, name: str, track: str) -> Path:
    """results/<driver>_<track>.json: one file per driver AND track, so tracks do not overwrite each other."""
    import re
    from .race.replay import slug
    track_part = re.sub(r"[^A-Za-z0-9_-]+", "_", str(track).replace(":", "-")).strip("_") or "track"
    return Path(results_dir) / f"{slug(name)}_{track_part}.json"


def _status_text(report: Any) -> str:
    r = report.result
    if r.completed:
        return f"FINISHED {r.race_time:.2f} s"
    return report.status or f"DNF ({r.ended_by})"


# ============================================================================ drive
def cmd_drive(args: argparse.Namespace) -> int:
    from .race import DriverLoadError, SessionConfig, load_driver, run_session

    problem = _track_problem(_track_spec(args))
    if problem:
        return _fail(problem, 2)
    try:
        drv = load_driver(args.driver, params=_params(args.param) or None, isolate=args.isolate,
                          llm_mode=args.llm_policy)
    except DriverLoadError as exc:
        return _fail(str(exc))
    config = SessionConfig(track=_track_spec(args), attempts=args.attempts, laps=args.laps, realtime=args.realtime,
                           max_ticks_per_lap=args.max_ticks, physics=_physics(args), llm_policy=args.llm_policy)
    _warn_unknown_params(drv)
    print(f"{drv.name} by {drv.author} ({drv.kind}) on {config.track}: {args.attempts} attempt(s) x "
          f"{args.laps} lap(s){' [realtime]' if args.realtime else ''}")
    states: list[dict] = []
    try:
        result = run_session(drv, config, on_event=_drive_printer(args, states))
    finally:
        drv.close()
    _print_session_summary(result)

    if args.dump_states:
        _dump_states(args.dump_states, states, config.track, drv.name, args.dump_n)
    if args.save:
        out = _result_file(args.results, drv.name, result.track_key)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result.to_result_dict(), indent=1, allow_nan=False), encoding="utf-8")
        print(f"Result saved: {out}  (python race.py leaderboard)")
    if args.plot:
        from .viz.mpl import plot_session
        plot_session(result, args.plot, which=args.which)
        print(f"Plot: {args.plot}")
    replay_path = args.replay
    if args.unity and not replay_path:
        replay_path = "races/latest_session.json"
    if replay_path or args.html:
        replay = result.to_replay(args.which)
        if replay_path:
            Path(replay_path).parent.mkdir(parents=True, exist_ok=True)
            replay.to_json(replay_path, extended=not args.strict_v1)
            print(f"Replay: {replay_path}  (python race.py watch {replay_path} --html | --unity)")
        if args.html:
            _save_html(replay, args.html)
        if args.unity:
            _to_unity(replay_path)
    return 0


def _drive_printer(args: argparse.Namespace, states: list[dict]) -> Callable:
    """The per-tick / per-attempt log of `drive` (events, radio, errors; --quiet = summaries only)."""
    def on_event(ev: Any) -> None:
        if ev.type == "tick":
            f = ev.data
            if args.dump_states and f.get("obs") is not None:
                states.append(f["obs"])
            if args.quiet:
                return
            notes = [e for e in f.get("events", []) if args.verbose or not e.startswith("waypoint_passed")]
            if f.get("say"):
                notes.append(f'radio: "{f["say"]}"')
            if f.get("driver_error"):
                notes.append(f"driver error: {f['driver_error']}")
            if f.get("stalled"):
                notes.append("stall: drive() too slow, coasting")
            if f.get("late_ticks"):
                notes.append(f"late: +{f['late_ticks']} tick(s) with the previous command")
            if notes or args.verbose:
                print(f"  t={f['t']:7.2f}s tick {f['tick']:4d} lap {f['lap']} seg {f['segment']:2d} "
                      f"v={f['speed']:5.1f} | " + " | ".join(notes))
        elif ev.type == "plan":
            err = ev.data.get("error")
            print(f"plan(): {ev.data['seconds']:.2f} s" + (f" - ERROR {err}" if err else ""))
        elif ev.type == "attempt_start" and not args.quiet:
            print(f"Attempt {ev.attempt} [{ev.phase}]")
        elif ev.type == "attempt_end":
            rep, r = ev.data, ev.data.result
            laps = ", ".join(f"{t:.2f}" for t in r.lap_times) or "-"
            print(f"Attempt {ev.attempt}: {_status_text(rep)} | laps [{laps}] | crashes {r.crashes} | "
                  f"penalties {r.penalty_time:.1f} s | {r.waypoints_passed} waypoints | {r.ticks} ticks"
                  + (f" | {rep.stalls} stalls" if rep.stalls else "")
                  + (f" | {rep.driver_error_count} driver errors" if rep.driver_error_count else ""))
            if rep.reflect_error:
                print(f"  reflect() error: {rep.reflect_error}")
    return on_event


def _print_session_summary(result: Any) -> None:
    best = result.best_lap
    print(f"Best lap: {_secs(best)} s" if best is not None else "Best lap: none (no finished lap)")
    think = result.think_seconds_mean * 1000
    line = f"drive() time: mean {think:.1f} ms, max {result.think_seconds_max * 1000:.1f} ms"
    if result.llm.get("calls"):
        line += (f" | LLM: {result.llm['calls']} calls, "
                 f"{result.llm.get('prompt_tokens', 0) + result.llm.get('completion_tokens', 0)} tokens")
    print(line + f" | wall {result.wall_seconds:.1f} s")
    if result.plan_error:
        print(f"plan() error: {result.plan_error}")


def _dump_states(path: str, states: list[dict], track: Any, driver: str, n: int) -> None:
    """Save up to n evenly spaced obs dicts (for `race.py exam`)."""
    if not states:
        print("No states recorded - nothing dumped.")
        return
    step = max(1, len(states) // max(1, n))
    picked = states[::step][:n]
    out = {"format": "agentic-gp-states", "version": 1, "track": str(track), "driver": driver,
           "states": picked}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=1, default=_json_default), encoding="utf-8")
    print(f"States: {len(picked)} obs saved to {path}  (python race.py exam DRIVER.py --states {path})")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# ============================================================================ check
def cmd_check(args: argparse.Namespace) -> int:
    from .race import DriverLoadError, SessionConfig, check_driver_source, load_driver, run_session

    path = Path(args.driver)
    print(f"Checking {path}")
    if not path.is_file():
        return _fail(f"file not found: {path}")
    issues = check_driver_source(path)
    errors = [i for i in issues if i.level == "error"]
    for i in issues:
        print(f"  {i.level.upper():7s} {i.message}")
    if errors:
        print(f"FAILED: {len(errors)} error(s) in the static check.")
        return 1
    print("  static check: OK" + (f" ({len(issues)} warning(s))" if issues else ""))

    if not args.real_llm:
        os.environ["LLM_PROVIDER"] = "mock"          # the smoke test checks the contract, not the LLM
    try:
        drv = load_driver(path, isolate=args.isolate)
    except DriverLoadError as exc:
        print(f"FAILED: {exc}")
        return 1
    meta = drv.meta()
    print(f"  driver: {meta['name']} by {meta['author']} | kind {meta['kind']} | color {meta['color']}"
          + (f" | model {meta['model']}" if meta.get("model") else "")
          + (f" | PARAMS {meta['params']}" if meta.get("params") else ""))
    uses_llm = bool(meta.get("uses_llm"))
    if uses_llm:
        print("  LLM: " + ("real (--real-llm)" if args.real_llm else
                           "mock (offline: chat() answers 'OK (mock LLM)', chat_json() returns {}; "
                           "use --real-llm to test with MiniMax)"))
    config = SessionConfig(track=args.track, attempts=1, max_ticks_per_lap=args.ticks)
    if args.isolate:
        print("  isolated: the driver runs in its own process, with the grand prix time limits")
        try:
            result = run_session(drv, config)
        finally:
            drv.close()
    else:
        result, hung = _run_check_session(drv, config, real_llm=args.real_llm)
        if hung:
            print(f"  ERROR   {hung}")
            print("FAILED")
            return 1
        drv.close()
    rep = result.attempts[0] if result.attempts else None
    problems: list[str] = []
    warnings: list[str] = []
    if result.plan_error:
        problems.append(f"plan() raised: {result.plan_error}")
    if rep is None:
        problems.append(result.status or "the driver never drove")
    else:
        if rep.driver_error_count:
            problems.append(f"drive() raised {rep.driver_error_count} time(s): {rep.driver_errors[0]['error']}")
        if rep.reflect_error:
            problems.append(f"reflect() raised: {rep.reflect_error}")
    mean_ms = result.think_seconds_mean * 1000
    max_ms = result.think_seconds_max * 1000
    if max_ms > 2000:
        problems.append(f"drive() took {max_ms:.0f} ms once (limit 2000 ms: that tick would be a stall)")
    if rep is not None:
        print(f"  smoke test: {rep.result.ticks} ticks on {args.track}, {rep.result.waypoints_passed} waypoints, "
              f"{rep.result.crashes} crashes, drive() mean {mean_ms:.1f} ms / max {max_ms:.1f} ms")
        top_speed = max((float(f["speed"]) for f in rep.frames), default=0.0)
        if top_speed < 1.0:
            warnings.append(f"the car never moved (top speed {top_speed:.1f}) - is the throttle positive? "
                            "drive() should return (throttle, steering), e.g. (0.6, 0.0)")
        elif rep.result.waypoints_passed == 0 and not problems:
            warnings.append(f"no waypoint passed in {rep.result.ticks} ticks - check the steering sign "
                            "(+ = right: steer toward angle_to_next_waypoint)")
    if mean_ms > 50:
        warnings.append(f"drive() averages {mean_ms:.0f} ms (race rule: < 50 ms on average)")
    for w in warnings:
        print(f"  WARNING {w}")
    for p in problems:
        print(f"  ERROR   {p}")
    if problems:
        if result.first_error_traceback:
            print("\nFirst error, full traceback:\n" + _indent(_traceback_tail(result.first_error_traceback)))
        if uses_llm and not args.real_llm:
            print("  hint: the LLM was mocked - your code must survive an empty or invalid LLM answer (use a "
                  "fallback: try/except llm.LLMError, data.get(key, default)), or run with --real-llm")
        print("FAILED")
        return 1
    print("PASSED - ready to race (python race.py validate " + str(path) + ")")
    return 0


CHECK_SILENCE_S = {"plan": 150.0, "drive": 15.0, "reflect": 150.0}   # check: longest silence per phase


def _run_check_session(drv: Any, config: Any, real_llm: bool = False) -> tuple[Any, str | None]:
    """run_session in a helper thread with a watchdog: an endless loop in plan(), drive() or reflect()
    ends the check with a clear message after CHECK_SILENCE_S instead of hanging forever.
    Returns (result, None) or (None, "what hung")."""
    from .race import run_session

    limits = dict(CHECK_SILENCE_S)
    if real_llm:
        limits["drive"] = 120.0                      # a real LLM call per tick may take a while
    state: dict = {"phase": "plan", "t": time.monotonic(), "result": None, "error": None, "tick": 0}

    def on_event(ev: Any) -> None:
        state["t"] = time.monotonic()
        if ev.type in ("plan", "attempt_start", "tick"):
            state["phase"] = "drive"
            if ev.type == "tick":
                state["tick"] = int(ev.data.get("tick", 0))
        elif ev.type == "attempt_end":
            state["phase"] = "reflect"

    def work() -> None:
        try:
            state["result"] = run_session(drv, config, on_event=on_event)
        except BaseException as exc:                 # noqa: BLE001 - re-raised in the main thread
            state["error"] = exc

    th = threading.Thread(target=work, name="agp-check", daemon=True)
    th.start()
    while th.is_alive():
        th.join(0.25)
        silent = time.monotonic() - state["t"]
        limit = limits[state["phase"]]
        if th.is_alive() and silent > limit:
            where = f"{state['phase']}()" + (f" (after tick {state['tick']})" if state["phase"] == "drive" else "")
            return None, (f"{where} did not return within {limit:.0f} s - an endless loop? (in a race this car "
                          "would stall or time out; `race.py check --isolate` runs it under the race's own limits)")
    if state["error"] is not None:
        raise state["error"]
    return state["result"], None


# ============================================================================ validate
def _traceback_tail(tb: str, lines: int = 12) -> str:
    return "\n".join(tb.rstrip().splitlines()[-lines:])


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def cmd_validate(args: argparse.Namespace) -> int:
    from .race import DriverLoadError, SessionConfig, load_driver, run_session
    from .sim.track import PUBLIC_TRACKS

    tracks = list(args.tracks or PUBLIC_TRACKS) + [f"random:{args.seed + k}" for k in range(args.random)]
    for spec in tracks:
        problem = _track_problem(spec)
        if problem:
            return _fail(problem, 2)
    rows, scores = [], []
    print(f"Validating {args.driver} on {len(tracks)} tracks: plan -> 1 practice + reflect -> 1 timed lap "
          f"(DNF counts as {DNF_SCORE_S:.0f} s)")
    for spec in tracks:
        try:
            drv = load_driver(args.driver, params=_params(args.param) or None, llm_mode=args.llm_policy)
        except DriverLoadError as exc:
            return _fail(str(exc))
        config = SessionConfig(track=spec, attempts=2, phases=["practice", "qualifying"],
                               llm_policy=args.llm_policy, physics=_physics(args))
        t0 = time.perf_counter()
        try:
            result = run_session(drv, config)
        finally:
            drv.close()
        timed = result.report_for("qualifying")
        practice = result.report_for("practice")
        r = timed.result if timed else None
        score = r.race_time if r is not None and r.completed else DNF_SCORE_S
        scores.append(score)
        rows.append([spec, _secs(practice.result.race_time) if practice and practice.result.completed else "DNF",
                     _secs(r.race_time) if r is not None and r.completed else "DNF",
                     r.crashes if r else "-", _secs(r.penalty_time, 1) if r else "-",
                     (timed.status if timed else result.status) or "ok", f"{time.perf_counter() - t0:.1f}"])
        if not args.quiet:
            print(f"  {spec:14s} timed lap {rows[-1][2]:>7s}  crashes {rows[-1][3]}")
    print()
    print(_table(["track", "practice", "timed lap", "crashes", "penalties", "status", "wall s"], rows))
    print(f"\nValidation score (mean timed lap, DNF = {DNF_SCORE_S:.0f} s): {statistics.fmean(scores):.2f} s"
          f"   | finished {sum(1 for s in scores if s < DNF_SCORE_S)}/{len(scores)}")
    return 0


# ============================================================================ exam
def cmd_exam(args: argparse.Namespace) -> int:
    from .race import DriverLoadError, load_driver
    from .sim.env import RaceEnv
    from .sim.track import make_track

    states_path = Path(args.states)
    if not states_path.is_file():
        return _fail(f"states file not found: {states_path} (make one with: race.py drive DRIVER.py "
                     f"--dump-states {states_path})")
    try:
        data = json.loads(states_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _fail(f"{states_path} is not a states file (JSON error: {exc}); make one with: race.py drive "
                     f"DRIVER.py --dump-states {states_path}")
    if isinstance(data, list):
        data = {"states": data}
    if not isinstance(data, dict):
        return _fail(f"{states_path}: expected {{\"states\": [obs, ...]}} or a list of obs dicts")
    states = [s for s in (data.get("states") or []) if isinstance(s, dict)][: args.n]
    if not states:
        return _fail(f"{args.states}: no states (make some with: race.py drive X.py --dump-states FILE)")
    reference = args.reference or next((r for r in DEFAULT_REFERENCE if _repo_path(r).is_file()), None)
    if reference is None:
        return _fail("no reference driver found - pass --reference FILE")
    track_map = RaceEnv(make_track(args.track or data.get("track") or "grand_prix")).track_map()
    try:
        cand = load_driver(args.driver)
        ref = load_driver(_repo_path(reference))
    except DriverLoadError as exc:
        return _fail(str(exc))
    if cand.uses_llm or ref.uses_llm:
        from . import llm
        try:
            llm.warm_up()                            # latency below = the call, not the SDK import
        except Exception:                            # noqa: BLE001
            pass
    for d in (cand, ref):
        try:
            d.plan(track_map)
        except Exception as exc:                     # noqa: BLE001 - an exam still runs without a plan
            print(f"  {d.name}: plan() raised {type(exc).__name__}: {exc}")
    print(f"Exam: {cand.name} vs reference {ref.name} on {len(states)} recorded states "
          f"(from {data.get('driver', '?')} on {data.get('track', '?')})")
    rows, lat, agree_s, agree_t = [], [], 0, 0
    for k, obs in enumerate(states, 1):
        t0 = time.perf_counter()
        try:
            a = cand.drive(obs)
            err = ""
        except Exception as exc:                     # noqa: BLE001
            a, err = None, f"{type(exc).__name__}: {exc}"[:40]
        lat.append(time.perf_counter() - t0)
        try:
            b = ref.drive(obs)
        except Exception:                            # noqa: BLE001
            b = None
        ct, cs = (a.throttle, a.steering) if a else (0.0, 0.0)
        rt, rs = (b.throttle, b.steering) if b else (0.0, 0.0)
        s_ok = _same_sign(cs, rs, 0.05)
        t_ok = _same_sign(ct, rt, 0.05)
        agree_s += s_ok
        agree_t += t_ok
        rows.append([k, f"{obs.get('speed', 0):.1f}", f"{obs.get('angle_to_next_waypoint', 0):+.0f}",
                     f"{obs.get('distance_to_wall_front', 0):.0f}", f"{ct:+.2f}", f"{cs:+.2f}", f"{rt:+.2f}",
                     f"{rs:+.2f}", "ok" if s_ok else "WRONG", "ok" if t_ok else "WRONG",
                     f"{lat[-1] * 1000:.0f}", err or (a.say or "")[:30]])
    cand.close()
    ref.close()
    print(_table(["#", "speed", "angle", "front", "thr", "steer", "ref thr", "ref steer", "steer sign",
                  "thr sign", "ms", "note"], rows))
    n = len(states)
    lat_sorted = sorted(lat)
    print(f"\nSteering sign agreement: {100 * agree_s / n:.0f} %   Throttle sign agreement: {100 * agree_t / n:.0f} %")
    print(f"Latency per drive() call: p50 {_pct(lat_sorted, 50) * 1000:.0f} ms, p95 {_pct(lat_sorted, 95) * 1000:.0f} ms,"
          f" max {lat_sorted[-1] * 1000:.0f} ms  (one tick = 500 ms; the race limit is 2000 ms)")
    return 0


def _same_sign(a: float, b: float, dead: float) -> bool:
    """Same direction ("+ = right"). Near zero (|x| < dead) means "straight": it agrees with a small value."""
    if abs(a) < dead or abs(b) < dead:
        return abs(a - b) < 2 * dead
    return (a > 0) == (b > 0)


def _pct(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = min(len(sorted_values) - 1, max(0, math.ceil(p / 100 * len(sorted_values)) - 1))
    return sorted_values[k]


# ============================================================================ grand prix
def _gp_entries(paths: list[str], house: bool, params: list[str] | None) -> list:
    """Driver files from paths (+ the house team), with --param overrides ("key=v" or "stem:key=v")."""
    from .race import DriverEntry, find_driver_files

    if not paths and not house:
        paths = ["drivers/submissions"]
    skipped: list = []
    files = find_driver_files(paths, skipped=skipped) if paths else []
    if skipped:
        print(f"note: side files not raced (pilot / fly files next to main files): "
              f"{', '.join(p.name for p in skipped)} - give a file path to race one anyway")
    house_files = [_repo_path(h) for h in HOUSE_DRIVERS if _repo_path(h).is_file()] if house else []
    house_set = {p.resolve() for p in house_files}
    seen, ordered = set(), []
    for f in list(files) + house_files:
        r = Path(f).resolve()
        if r not in seen:
            seen.add(r)
            ordered.append(Path(f))
    common, per_file = {}, {}
    for item in params or []:
        key, sep, value = item.partition("=")
        stem, colon, name = key.partition(":")
        if not sep:
            raise ValueError(f"--param expects key=value, got {item!r}")
        if colon:
            per_file.setdefault(stem.strip(), {})[name.strip()] = value
        else:
            common[key.strip()] = value
    entries = [DriverEntry(f, house=f.resolve() in house_set,
                           params={**common, **per_file.get(f.stem, {})} or None) for f in ordered]
    _note_duplicate_names(entries)
    return entries


def _note_duplicate_names(entries: list) -> None:
    """Two cars with the same NAME race as "Name" and "Name-2" (results, logs, viewers): say so up front."""
    from .race.grand_prix import plan_labels
    planned = plan_labels(entries)["planned"]
    for i, (name, label) in sorted(planned.items()):
        if label != name:
            first = next((j for j, (n, lab) in planned.items() if lab == name), i)
            me = Path(entries[i].source).name
            print(f"note: {me} and {Path(entries[first].source).name} are both called {name!r} - {me} races as "
                  f"{label!r} (give it its own NAME to tell them apart)")


def _gp_config(args: argparse.Namespace, track: str, practice: int, out_dir: Path, stem: str):
    from .race import GrandPrixConfig

    return GrandPrixConfig(
        track=track, practice_attempts=practice, race_laps=args.laps, jobs=args.jobs,
        driver_timeout_s=args.timeout, load_timeout_s=args.load_timeout, title=args.title,
        llm_policy=args.llm_policy, physics=_physics(args),
        llm_log_dir=str(out_dir / "llm_logs" / stem), log_dir=str(Path("logs") / "drivers" / stem),
        ramp_s=args.ramp, qualifying=not args.no_qualifying,
    )


class _GPPrinter:
    """Friendly grand prix progress: one line per phase change / attempt result per driver."""

    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self.t0 = time.perf_counter()
        self.phase: dict[str, tuple] = {}

    def __call__(self, u: dict) -> None:
        if self.quiet:
            return
        name = u.get("driver", "?")
        stamp = f"[{time.perf_counter() - self.t0:6.1f}s] {name:16s}"
        if u.get("type") == "progress":
            key = (u.get("phase"), u.get("attempt"))
            if self.phase.get(name) != key:
                self.phase[name] = key
                labels = {"load": "loading", "warm-up": "importing the LLM SDK", "plan": "planning"}
                label = labels.get(u.get("phase"), u.get("phase"))
                extra = f" #{u['attempt']}" if u.get("phase") == "practice" else ""
                if u.get("phase") == "plan" and u.get("error"):
                    extra = f" - plan() error: {u['error']}"
                print(f"{stamp} {label}{extra}", flush=True)
        elif u.get("type") == "attempt":
            res = f"{u['race_time']:.2f} s" if u.get("completed") else (u.get("status") or "DNF")
            print(f"{stamp}   {u['phase']} #{u['attempt']}: {res}, {u.get('crashes', 0)} crashes", flush=True)
        elif u.get("type") == "error":
            print(f"{stamp} ERROR: {str(u.get('traceback', ''))[-300:]}", flush=True)
        elif u.get("type") == "done":
            res = f"{u['race_time']:.2f} s" if u.get("race_time") is not None else u.get("status")
            print(f"{stamp} done: race {res}", flush=True)


def _print_gp(gp: Any) -> None:
    if gp.qualifying:
        print("\nQualifying (grid):")
        print(_table(["grid", "driver", "kind", "lap", "best practice", "status"],
                     [[q["position"], q["name"] + (" (house)" if q.get("house") else ""), q.get("kind"),
                       _secs(q.get("qualifying_time")), _secs(min([t for t in q.get("practice_times", []) if t],
                                                                   default=None)),
                       q.get("status") or "ok"] for q in gp.qualifying]))
    print(f"\nRace: {gp.title} - {gp.track_name}, {gp.replay.laps} laps")
    print(_table(["pos", "driver", "kind", "grid", "race time", "gap", "laps", "best lap", "crashes", "LLM",
                  "status"],
                 [[s["position"], s["name"] + (" (house)" if s.get("house") else ""), s.get("kind"),
                   s.get("grid") or "-", _secs(s.get("race_time")), s.get("gap_text") or "-", s["laps_completed"],
                   _secs(s.get("best_lap")), s["crashes"], s.get("llm_calls", 0), s.get("status") or "finished"]
                  for s in gp.standings]))
    for e in gp.errors:
        print(f"  note: {e['driver']}: {e['status'] or 'error'} - {str(e['error'])[:160]}")
    print(f"(wall time {gp.wall_seconds:.1f} s)")


def _run_one_gp(args: argparse.Namespace, entries: list, track: str, practice: int, out: Path,
                title: str | None = None) -> Any:
    from .race import run_grand_prix

    out_dir, stem = out.parent, out.stem
    config = _gp_config(args, track, practice, out_dir, stem)
    if title:
        config.title = title
    print(f"Grand prix '{config.title}' on {track}: {len(entries)} drivers, {practice} practice + "
          f"{'qualifying + ' if config.qualifying else ''}{args.laps}-lap race, "
          + ("in-process" if args.inprocess else f"jobs={args.jobs or 'auto'}"))
    stop = threading.Event()
    try:
        gp = run_grand_prix(entries, config, progress=_GPPrinter(args.quiet), stop_event=stop,
                            inprocess=args.inprocess)
    except KeyboardInterrupt:
        stop.set()
        raise
    _print_gp(gp)
    paths = gp.save(out_dir, stem, extended=not args.strict_v1)
    from .race.leaderboard import standings_markdown
    (out_dir / f"{stem}_standings.md").write_text(standings_markdown(gp), encoding="utf-8")
    print("Saved: " + ", ".join(str(p) for p in paths.values()) + f", {out_dir / (stem + '_standings.md')}")
    return gp


def _gp_out(out: str) -> Path:
    p = Path(out)
    return p if p.suffix == ".json" else p / "grand_prix.json"


def cmd_grand_prix(args: argparse.Namespace) -> int:
    try:
        entries = _gp_entries(args.paths, args.house, args.param)
    except ValueError as exc:
        return _fail(str(exc))
    if not entries:
        return _fail("no driver files found (give files or folders, e.g. drivers/submissions)")
    problem = _track_problem(_track_spec(args))
    if problem:
        return _fail(problem, 2)
    out = _gp_out(args.out)
    gp = _run_one_gp(args, entries, _track_spec(args), args.practice, out)
    if args.html:
        _save_html(gp.replay, args.html)
    if args.unity:
        _to_unity(out)
    return 0


def cmd_final(args: argparse.Namespace) -> int:
    from .race.leaderboard import championship, championship_markdown, standings_markdown

    try:
        entries = _gp_entries(args.paths, args.house, args.param)
    except ValueError as exc:
        return _fail(str(exc))
    if not entries:
        return _fail("no driver files found (give files or folders, e.g. drivers/submissions)")
    for label, spec in (("--home", args.home), ("--mystery", args.mystery)):
        problem = _track_problem(spec)               # before the Home GP: a typo must not waste a race
        if problem:
            return _fail(f"{label} {spec}: {problem}", 2)
    out_dir = Path(args.out)
    home = _run_one_gp(args, entries, args.home, args.practice, out_dir / "home.json",
                       title=f"{args.title} - Home GP")
    print()
    mystery = _run_one_gp(args, entries, args.mystery, args.mystery_practice, out_dir / "mystery.json",
                          title=f"{args.title} - Mystery GP")
    rows = championship([home, mystery])
    titles = [f"Home ({home.track_name})", f"Mystery ({mystery.track_name})"]
    (out_dir / "championship.json").write_text(json.dumps({"format": "agentic-gp-championship", "version": 1,
                                                           "races": titles, "standings": rows}, indent=1),
                                               encoding="utf-8")
    md = championship_markdown(rows, titles) + "\n" + standings_markdown(home) + "\n" + standings_markdown(mystery)
    (out_dir / "championship.md").write_text(md, encoding="utf-8")
    print("\nChampionship (house cars race but are not scored):")
    print(_table(["pos", "driver", "points", *titles],
                 [[r["position"], r["name"], r["points"],
                   *[("-" if x is None else f"P{x['position']} ({x['points']})") for x in r["results"]]]
                  for r in rows]))
    print(f"Saved: {out_dir / 'championship.json'}, {out_dir / 'championship.md'}")
    for path in (out_dir / "home.json", out_dir / "mystery.json"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  {path.name} sha256 {digest}  (record it on the results slide)")
    if args.html:
        for gp, name in ((home, "home"), (mystery, "mystery")):
            _save_html(gp.replay, out_dir / f"{name}.html")
    if args.unity:
        _to_unity(out_dir / "mystery.json")
    return 0


# ============================================================================ watch
def cmd_watch(args: argparse.Namespace) -> int:
    from .race.replay import Replay

    path = Path(args.replay)
    if not path.is_file():
        return _fail(f"replay not found: {path}")
    try:
        replay = Replay.from_json(path)
    except (ValueError, KeyError, TypeError) as exc:
        return _fail(f"{path} is not a replay: {exc}")
    where = "" if replay.track.name.lower() in replay.title.lower() else f" on {replay.track.name}"
    print(f"{replay.title} ({replay.kind}, {replay.laps} lap(s)){where}: {len(replay.cars)} car(s)")
    for c in replay.cars:
        res = f"{c.race_time:.2f} s" if c.finished and c.race_time is not None else (c.status or "DNF")
        print(f"  P{c.position:<2d} {c.name:16s} {res:>14s}  laps {c.laps_completed}  crashes {c.crashes}")
    problems = replay.validate()
    if problems:
        print(f"  ({len(problems)} schema problem(s): {problems[0]})")
    if args.html is not None:
        out = args.html or str(path.with_suffix(".html"))
        if _save_html(replay, out) and not args.no_open:
            import webbrowser
            opened = False
            try:
                opened = webbrowser.open(Path(out).resolve().as_uri())
            except Exception:                        # noqa: BLE001 - no browser is not an error
                pass
            if opened:
                print("Opening it in your browser ...")
            else:
                cmd = "start" if sys.platform == "win32" else "open" if sys.platform == "darwin" else "xdg-open"
                print(f"Open it yourself:  {cmd} {out}")
    if args.unity:
        _to_unity(path)
    if args.html is None and not args.unity:
        print("Watch it: --html (browser player), --unity (3D viewer), or "
              "`streamlit run app/race_control.py` -> Replay page.")
    return 0


# ============================================================================ tracks
def cmd_tracks(args: argparse.Namespace) -> int:
    from .sim.config import DEFAULT_CONFIG
    from .sim.track import TRACKS, make_track

    specs = [k for k in TRACKS] + [f"random:{args.seed + k}" for k in range(args.random)]
    rows = []
    for spec in specs:
        tr = make_track(spec)
        corners = tr.corners(DEFAULT_CONFIG)
        name = f"{tr.name} (alias of grand_prix)" if spec == "default" else tr.name
        rows.append([spec, name, f"{tr.length:.0f}", f"{float(min(tr.radii)):.1f}", len(corners),
                     min((c["v_max"] for c in corners), default="-"), tr.n_gates, tr.half_width,
                     len(tr.zones) or "-"])
        if args.plot:
            from .viz.mpl import plot_track
            Path(args.plot).mkdir(parents=True, exist_ok=True)
            plot_track(tr, Path(args.plot) / f"{spec.replace(':', '_')}.png")
    print(_table(["track", "name", "length", "min R", "corners", "slowest v", "gates", "half width", "zones"],
                 rows))
    if args.plot:
        print(f"Plots in {args.plot}/")
    print("Also: --track random:<seed> (any seed) and --track-file my_track.py (make_track() or CONTROL_POINTS).")
    return 0


# ============================================================================ leaderboard
def cmd_leaderboard(args: argparse.Namespace) -> int:
    from .race.leaderboard import main as leaderboard_main
    argv = ["--results", args.results, "--out", args.out, "--baseline", args.baseline]
    return leaderboard_main(argv + (["--track", args.track] if args.track else []))


# ============================================================================ doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    checks: list[tuple[str, str, str]] = []           # (OK | WARN | FAIL, what, detail)

    def add(status: str, what: str, detail: str = "") -> None:
        checks.append((status, what, detail))
        print(f"  [{status:4s}] {what}" + (f": {detail}" if detail else ""), flush=True)

    print("Agentic Grand Prix doctor")
    v = sys.version_info
    add("OK" if v >= (3, 11) else "FAIL", "Python", f"{platform.python_version()} ({sys.executable})"
        + ("" if v >= (3, 11) else " - need 3.11+ (Streamlit); install 3.12 with uv"))
    add("OK", "Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    _doctor_packages(add)
    _doctor_sim(add)
    _doctor_paths(add)
    _doctor_env(add)
    if args.llm:
        _doctor_llm_call(add)
    viewer = None
    try:
        from .viz.unity import find_viewer
        viewer = find_viewer()
    except Exception:                                  # noqa: BLE001 - optional
        pass
    add("OK", "Unity 3D viewer", str(viewer) if viewer else "not built (optional; unity/README.md)")
    app = REPO_ROOT / "app" / "race_control.py"
    add("OK" if app.is_file() else "WARN", "Streamlit app", "streamlit run app/race_control.py" if app.is_file()
        else "app/race_control.py not found")

    fails = sum(1 for c in checks if c[0] == "FAIL")
    warns = sum(1 for c in checks if c[0] == "WARN")
    verdict = "ALL GREEN" if not fails and not warns else (f"{fails} FAIL, {warns} WARN" if fails else
                                                         f"OK with {warns} warning(s)")
    report = ["Agentic Grand Prix - doctor report", f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
              f"repo: {REPO_ROOT}", ""] + [f"[{s:4s}] {w}" + (f": {d}" if d else "") for s, w, d in checks] + \
             ["", f"verdict: {verdict}"]
    Path(args.out).write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n{verdict} - report written to {args.out} (send it to your instructor)")
    return 1 if fails else 0


def _doctor_packages(add: Callable) -> None:
    from importlib import metadata, util

    needed = {"numpy": True, "matplotlib": True, "openai": True, "streamlit": True, "plotly": True,
              "scipy": False, "pandas": False, "langgraph": False, "langchain_openai": False, "pytest": False}
    dist = {"langchain_openai": "langchain-openai"}
    for mod, required in needed.items():
        if util.find_spec(mod) is None:
            add("FAIL" if required else "WARN", f"package {mod}", "missing - run: uv sync (or pip install -r "
                "requirements.txt)")
            continue
        try:
            version = metadata.version(dist.get(mod, mod))
        except metadata.PackageNotFoundError:
            version = "?"
        add("OK", f"package {mod}", version)


def _doctor_sim(add: Callable) -> None:
    try:
        from .sim.env import RaceEnv
        from .sim.track import make_track
        env = RaceEnv(make_track("grand_prix"))
        obs = env.reset()
        t0 = time.perf_counter()
        for _ in range(20):
            obs = env.step(0.6, obs["angle_to_next_waypoint"] / 30.0)
        ms = (time.perf_counter() - t0) / 20 * 1000
        add("OK", "Simulator", f"20 ticks, {obs['waypoints_passed']} waypoints, {ms:.2f} ms/tick")
    except Exception as exc:                           # noqa: BLE001
        add("FAIL", "Simulator", f"{type(exc).__name__}: {exc}")


def _doctor_paths(add: Callable) -> None:
    p = str(REPO_ROOT)
    problems = []
    if "onedrive" in p.lower():
        problems.append("inside OneDrive (sync locks files - move the repo, e.g. to C:/code)")
    if not p.isascii():
        problems.append("non-ASCII characters in the path")
    if " " in p:
        problems.append("spaces in the path")
    add("WARN" if problems else "OK", "Repo path", p + ("" if not problems else " - " + "; ".join(problems)))


def _doctor_env(add: Callable) -> None:
    env_file = REPO_ROOT / ".env"
    tracked = False
    try:
        r = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=REPO_ROOT, capture_output=True,
                           timeout=20)
        tracked = r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        pass
    if tracked:
        add("FAIL", ".env", "TRACKED BY GIT - run: git rm --cached .env (and ask for a new key)")
    else:
        add("OK", ".env", "present, not tracked by git" if env_file.is_file() else
            "not present (copy .env.example to .env on Day 2)")
    try:
        from . import llm
        cfg = llm.get_config()
        has_key = llm.llm_available()
        add("OK" if has_key else "WARN", "LLM config", f"{llm.describe_llm()}, effort {cfg.effort}, key "
            + ("set" if cfg.api_key else "not set") + ("" if has_key else " - ask your instructor (Day 2)"))
    except Exception as exc:                           # noqa: BLE001
        add("FAIL", "LLM config", f"{type(exc).__name__}: {exc}")


def _doctor_llm_call(add: Callable) -> None:
    from . import llm
    if not llm.llm_available():
        add("WARN", "LLM call", "skipped: no API key (set MINIMAX_API_KEY in .env)")
        return
    t0 = time.perf_counter()
    try:
        llm.warm_up()                                  # SDK import: slow once per process, not per call
        t1 = time.perf_counter()
        text = llm.chat("Reply with the single word OK.", effort="none", max_tokens=256)
        add("OK" if text.strip() else "FAIL", "LLM call", f"{llm.describe_llm()} answered in "
            f"{time.perf_counter() - t1:.1f} s (SDK start-up {t1 - t0:.1f} s, {len(text)} chars)")
    except Exception as exc:                           # noqa: BLE001
        add("FAIL", "LLM call", f"{type(exc).__name__}: {str(exc)[:160]}")


# ============================================================================ llm-check
def cmd_llm_check(args: argparse.Namespace) -> int:
    from . import llm

    print(f"LLM: {llm.describe_llm()} | key {'set' if llm.get_config().api_key else 'NOT set'}")
    if not llm.llm_available():
        print("FAIL: no API key - put MINIMAX_API_KEY=... in .env (never commit it)")
        return 1
    print("warming up the SDK ...", flush=True)
    llm.warm_up()
    rows = llm.llm_smoke_test(args.effort)
    print(_table(["step", "result", "seconds", "tokens", "detail"],
                 [[r["step"], "PASS" if r["ok"] else "FAIL", f"{r['seconds']:.2f}", r["tokens"], r["detail"]]
                  for r in rows]))
    ok = all(r["ok"] for r in rows)
    if args.load:
        ok = _llm_load_test(args.load, args.calls, args.ramp, args.effort) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _llm_load_test(workers: int, calls: int, ramp: float, effort: str | None) -> bool:
    """`workers` threads x `calls` tiny calls, one more worker every `ramp` s: latency + error codes."""
    from . import llm

    lat: list[float] = []
    errors: list[str] = []
    lock = threading.Lock()
    inflight = {"now": 0, "max": 0}

    def worker(w: int) -> None:
        for k in range(calls):
            t0 = time.perf_counter()
            with lock:
                inflight["now"] += 1
                inflight["max"] = max(inflight["max"], inflight["now"])
            try:
                llm.chat(f"Worker {w} call {k}: reply with OK.", effort=effort or "none", max_tokens=256)
                with lock:
                    lat.append(time.perf_counter() - t0)
            except Exception as exc:                   # noqa: BLE001
                code = llm.minimax_code(exc) or getattr(exc, "status", None) or type(exc).__name__
                with lock:
                    errors.append(str(code))
            finally:
                with lock:
                    inflight["now"] -= 1

    print(f"Load test: {workers} workers x {calls} calls, +1 worker every {ramp:g} s ...", flush=True)
    t0 = time.perf_counter()
    threads = []
    # Every worker is one student: let all of them have a request in flight at once (the default
    # semaphore allows 1 per process, which would turn the load test into a queue).
    llm.set_llm_semaphore(threading.BoundedSemaphore(max(1, workers)))
    try:
        for w in range(workers):
            th = threading.Thread(target=worker, args=(w,), daemon=True)
            th.start()
            threads.append(th)
            if w < workers - 1:
                time.sleep(ramp)
        for th in threads:
            th.join()
    finally:
        llm.set_llm_semaphore(None)
    wall = time.perf_counter() - t0
    lat.sort()
    n = len(lat) + len(errors)
    print(f"  {len(lat)}/{n} ok in {wall:.1f} s ({60 * n / max(wall, 1e-9):.0f} requests/min, "
          f"up to {inflight['max']} in flight)")
    if lat:
        print(f"  latency p50 {_pct(lat, 50):.2f} s, p95 {_pct(lat, 95):.2f} s, max {lat[-1]:.2f} s")
    if errors:
        counts = {e: errors.count(e) for e in sorted(set(errors))}
        print(f"  errors: {counts}  (2045/1002/429 = rate limit: lower --load or raise --ramp)")
    return not errors


# ============================================================================ argparse
def _add_track_args(p: argparse.ArgumentParser, default: str = "grand_prix") -> None:
    p.add_argument("--track", default=default, help="grand_prix | oval | sprint | speedway | briefing_demo | "
                                                    "random:<seed> | file:path.py (default %(default)s)")
    p.add_argument("--track-file", help="a track file (make_track() or CONTROL_POINTS); overrides --track")
    p.add_argument("--crash-rule", choices=["marshal", "classic"], help="default marshal (respawn on track)")


def _add_gp_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("paths", nargs="*", default=[],
                   help="driver files or folders (default drivers/submissions; with --house: none)")
    p.add_argument("--laps", type=int, default=3, help="race laps (default 3)")
    p.add_argument("--jobs", type=int, default=None, help="drivers running at once (default: CPUs - 1)")
    p.add_argument("--timeout", type=float, default=300.0, help="thinking-time budget per driver (s)")
    p.add_argument("--load-timeout", type=_seconds_arg, default=300.0,
                   help="seconds a driver may take to load, and again to import the LLM SDK (default 300; "
                        "raise it on a slow, busy laptop if cars end with 'DNF (timeout: loading)')")
    p.add_argument("--ramp", type=_seconds_arg, default=2.0,
                   help="start one more driver every RAMP s (default 2; '2s' works too)")
    p.add_argument("--house", action="store_true", help="add the house cars (Max, Planner, fly toy)")
    p.add_argument("--inprocess", action="store_true",
                   help="run drivers in this process (debugging, TRUSTED code only: a hung driver cannot be "
                        "killed and processes a driver starts are not cleaned up)")
    p.add_argument("--no-qualifying", action="store_true", help="skip the qualifying lap")
    p.add_argument("--llm-policy", default="official", choices=["official", "open"])
    p.add_argument("--param", action="append", help="key=value or file_stem:key=value (repeatable)")
    p.add_argument("--html", help="also write a standalone HTML replay player")
    p.add_argument("--unity", action="store_true", help="install the replay for the Unity viewer and start it")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--crash-rule", choices=["marshal", "classic"])
    p.add_argument("--strict-v1", action="store_true", help=STRICT_HELP)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="race.py", description="Agentic Grand Prix - race AI drivers.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Examples:\n  python race.py drive drivers/examples/l2_planner.py --attempts 2\n"
                                        "  python race.py check my_driver.py\n"
                                        "  python race.py grand-prix drivers/examples --track oval --practice 1")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("drive", help="run a session with one driver")
    p.add_argument("driver")
    _add_track_args(p)
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--laps", type=int, default=1)
    p.add_argument("--realtime", action="store_true", help="the world keeps moving while drive() thinks")
    p.add_argument("--max-ticks", type=int, default=None, help="tick limit per lap (default 500)")
    p.add_argument("--quiet", action="store_true", help="only attempt summaries")
    p.add_argument("--verbose", action="store_true", help="a line for every tick")
    p.add_argument("--plot", help="PNG of the best attempt")
    p.add_argument("--replay", help="replay JSON for the viewers")
    p.add_argument("--which", default="best", choices=["best", "last", "all"], help="attempts in the replay")
    p.add_argument("--strict-v1", action="store_true", help=STRICT_HELP)
    p.add_argument("--html", help="standalone HTML replay player")
    p.add_argument("--unity", action="store_true", help="install the replay for the Unity viewer and start it")
    p.add_argument("--save", action="store_true",
                   help="write results/<NAME>_<track>.json for the leaderboard (NAME 'Rocket 2' on sprint -> "
                        "results/rocket-2_sprint.json)")
    p.add_argument("--results", default="results")
    p.add_argument("--param", action="append", help="override a PARAMS knob: key=value (repeatable)")
    p.add_argument("--dump-states", help="save sampled obs dicts for `race.py exam`")
    p.add_argument("--dump-n", type=int, default=40, help="how many states to dump (default 40)")
    p.add_argument("--isolate", action="store_true", help="run the driver in a child process (race rules)")
    p.add_argument("--llm-policy", default="open", choices=["open", "official"])
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser("check", help="static check + 40-tick smoke test (exit 1 on errors)")
    p.add_argument("driver")
    p.add_argument("--track", default="grand_prix")
    p.add_argument("--ticks", type=int, default=40)
    p.add_argument("--isolate", action="store_true",
                   help="run the driver in its own process with the race time limits (like a grand prix)")
    p.add_argument("--real-llm", action="store_true", help="use the real LLM (default: mock)")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("validate", help="every public track + random tracks -> validation score")
    p.add_argument("driver")
    p.add_argument("--tracks", nargs="+", help="default: all public tracks")
    p.add_argument("--random", type=int, default=2, help="number of random tracks (default 2)")
    p.add_argument("--seed", type=int, default=1, help="first random seed (default 1)")
    p.add_argument("--param", action="append")
    p.add_argument("--llm-policy", default="official", choices=["official", "open"])
    p.add_argument("--crash-rule", choices=["marshal", "classic"])
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("exam", help="compare drive() decisions with a reference on recorded states")
    p.add_argument("driver")
    p.add_argument("--states", required=True, help="from: race.py drive X.py --dump-states FILE")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--reference", help="default drivers/examples/l1_reflex.py (else l2_planner.py)")
    p.add_argument("--track", help="track for plan() (default: the one in the states file)")
    p.set_defaults(func=cmd_exam)

    p = sub.add_parser("grand-prix", help="plan, practice, qualifying and a race for many drivers")
    _add_gp_args(p)
    p.add_argument("--track", default="grand_prix")
    p.add_argument("--track-file")
    p.add_argument("--practice", type=int, default=2, help="practice attempts (default 2)")
    p.add_argument("--title", default="Agentic Grand Prix")
    p.add_argument("--out", default="races/grand_prix.json", help="replay path (+ _qualifying/_results files)")
    p.set_defaults(func=cmd_grand_prix)

    p = sub.add_parser("final", help="Home GP + Mystery GP + championship")
    _add_gp_args(p)
    p.add_argument("--home", default="grand_prix", help="home track (default grand_prix)")
    p.add_argument("--mystery", required=True, help="mystery track spec, e.g. file:lesson_private/mystery.py")
    p.add_argument("--practice", type=int, default=2, help="home practice attempts (default 2)")
    p.add_argument("--mystery-practice", type=int, default=3, help="mystery practice attempts (default 3)")
    p.add_argument("--title", default="Agentic Grand Prix")
    p.add_argument("--out", default="races/final", help="output folder (default races/final)")
    p.set_defaults(func=cmd_final)

    p = sub.add_parser("watch", help="show a replay: summary, --html player or --unity 3D viewer")
    p.add_argument("replay")
    p.add_argument("--html", nargs="?", const="", default=None, help="write (and open) an HTML player")
    p.add_argument("--unity", action="store_true", help="install for the Unity viewer and start it")
    p.add_argument("--no-open", action="store_true", help="do not open the browser")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("tracks", help="list the tracks (+ PNGs with --plot DIR)")
    p.add_argument("--plot")
    p.add_argument("--random", type=int, default=0, help="also list N random tracks")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_tracks)

    p = sub.add_parser("leaderboard", help="results/*.json -> LEADERBOARD.md")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="LEADERBOARD.md")
    p.add_argument("--baseline", default="Max")
    p.add_argument("--track", help="only this track (default: one table per track)")
    p.set_defaults(func=cmd_leaderboard)

    p = sub.add_parser("doctor", help="check your setup -> doctor_report.txt")
    p.add_argument("--llm", action="store_true", help="also make one tiny LLM call")
    p.add_argument("--out", default="doctor_report.txt")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("llm-check", help="instructor: real LLM round trips (+ --load test)")
    p.add_argument("--effort", default=None, help="none | low | medium | high (default LLM_EFFORT)")
    p.add_argument("--load", type=int, default=0, help="concurrent workers for a load test")
    p.add_argument("--calls", type=int, default=3, help="calls per worker (default 3)")
    p.add_argument("--ramp", type=_seconds_arg, default=2.0,
                   help="seconds between starting workers (default 2; '30s' works too)")
    p.set_defaults(func=cmd_llm_check)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_console()
    ap = build_parser()
    args = ap.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except FileNotFoundError as exc:
        if os.getenv("AGP_DEBUG"):
            raise
        return _fail(str(exc))
    except SyntaxError as exc:                     # e.g. a track file that does not parse
        if os.getenv("AGP_DEBUG"):
            raise
        return _fail(f"syntax error in {exc.filename} line {exc.lineno}: {exc.msg}  "
                     "(AGP_DEBUG=1 shows the traceback)", 2)
    except OSError as exc:
        if os.getenv("AGP_DEBUG"):
            raise
        return _fail(str(exc))
    except (KeyError, ValueError) as exc:          # unknown track, bad --param value, ...
        if os.getenv("AGP_DEBUG"):
            raise
        text = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return _fail(f"{text}  (AGP_DEBUG=1 shows the traceback)", 2)


if __name__ == "__main__":
    raise SystemExit(main())
