"""Run a grand prix in its OWN Python process and stream its progress back to the app.

    from app.lib.gp_worker import GrandPrixSpec, run_gp_job
    spec = GrandPrixSpec(entries=[{"source": "drivers/examples/l1_reflex.py"}], config={"track": "oval"},
                         out_dir="races", stem="gp_1530", llm_env={"LLM_PROVIDER": "mock"})
    job = BackgroundJob(lambda j: run_gp_job(j, spec, on_update)).start()

Why a separate process? The Streamlit server is one process for every browser:
* the LLM provider of THIS race (mock or real) is set in the worker's environment only - another
  browser's choice never changes it, and this race never changes the server's;
* even "all drivers in one process" mode (faster for a demo) runs inside the worker, so a hanging
  or os._exit()-ing driver can only stop this race, never the app;
* Stop asks the worker to stop (it creates <spec>.stop, which the worker polls - NOT stdin: on Windows
  a thread blocked reading a pipe on stdin deadlocks every child process the worker starts); a worker
  that does not end in time is killed (its
  isolated driver processes notice that their parent is gone and exit too).

Protocol: the worker prints lines "@@AGP {json}" on stdout: {"kind": "update", "u": {...}} for
every progress update of run_grand_prix, then {"kind": "done", "paths": {...}, "standings": [...]}
or {"kind": "stopped"} or {"kind": "error", "error": "..."}. Driver print() output goes to stderr
(a log file), so it never gets in the way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .paths import REPO_ROOT

MARK = "@@AGP "
STOP_GRACE_S = 20.0                     # after "stop": kill the worker if it is not gone by then


@dataclass
class GrandPrixSpec:
    entries: list[dict]                 # [{"source": path, "house": bool, "params": dict|None}]
    config: dict                        # GrandPrixConfig keyword arguments (JSON-friendly)
    out_dir: str                        # where <stem>.json, _qualifying, _results, _standings.md go
    stem: str
    inprocess: bool = False             # all drivers in the WORKER process, one after another
    llm_env: dict = field(default_factory=dict)    # e.g. {"LLM_PROVIDER": "mock"} - this race only


def run_gp_job(job: Any, spec: GrandPrixSpec, on_update: Callable[[dict], None]) -> dict | None:
    """BackgroundJob body: start the worker, forward its updates, honour job.stop_event.

    Returns {"paths": {...}, "standings": [...]} when the race finished and was saved, None when it
    was stopped. Raises RuntimeError when the worker failed (the job's status becomes "error").
    """
    tmp = Path(tempfile.gettempdir()) / "agp_gp_worker"
    tmp.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time())}_{os.getpid()}_{threading.get_ident() % 100000}"
    spec_file, log_file = tmp / f"{stamp}_spec.json", tmp / f"{stamp}_stderr.log"
    stop_file = Path(str(spec_file) + ".stop")
    spec_file.write_text(json.dumps(asdict(spec)), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.update({k: str(v) for k, v in (spec.llm_env or {}).items()})
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    final: dict = {}
    with open(log_file, "w", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen([sys.executable, "-m", "app.lib.gp_worker", str(spec_file)], cwd=str(REPO_ROOT),
                                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)

        def watch_stop() -> None:
            while not job.stop_event.wait(0.5):
                if proc.poll() is not None:
                    return                               # finished by itself
            if proc.poll() is not None:
                return
            try:
                stop_file.write_text("stop", encoding="utf-8")
            except OSError:
                pass
            try:
                proc.wait(STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()

        threading.Thread(target=watch_stop, name="gp-worker-stop", daemon=True).start()
        try:
            for line in proc.stdout:
                if not line.startswith(MARK):
                    continue
                try:
                    msg = json.loads(line[len(MARK):])
                except ValueError:
                    continue
                if msg.get("kind") == "update":
                    on_update(msg["u"])
                else:
                    final = msg
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
    for f in (spec_file, stop_file):
        _drop(f)
    kind = final.get("kind")
    if kind == "done":
        _drop(log_file)
        return {"paths": final.get("paths", {}), "standings": final.get("standings", [])}
    if kind == "stopped" or job.stop_event.is_set():
        _drop(log_file)
        return None
    if kind == "error":
        raise RuntimeError(final.get("error") or "the grand prix failed")
    tail = _tail(log_file)
    raise RuntimeError(f"the grand prix process ended with exit code {proc.returncode}" + (f": {tail}" if tail else ""))


def _tail(path: Path, n: int = 600) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()[-n:]
    except OSError:
        return ""


def _drop(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def unique_stem(out_dir: str | Path, stem: str) -> str:
    """`stem`, or stem_2, stem_3, ... when a race with that name already exists (never overwrite)."""
    out = Path(out_dir)

    def taken(s: str) -> bool:
        return any((out / f"{s}{suffix}").exists() for suffix in (".json", "_results.json", "_qualifying.json"))

    if not taken(stem):
        return stem
    k = 2
    while taken(f"{stem}_{k}"):
        k += 1
    return f"{stem}_{k}"


# ============================================================================ the worker process
def _worker(spec_path: str) -> int:
    proto = sys.stdout                                   # protocol lines only
    sys.stdout = sys.stderr                              # every other print() (drivers!) -> the log
    lock = threading.Lock()

    def send(msg: dict) -> None:
        with lock:
            proto.write(MARK + json.dumps(msg, default=str) + "\n")
            proto.flush()

    try:
        from agentic_gp.race.grand_prix import DriverEntry, GrandPrixConfig, run_grand_prix

        spec = GrandPrixSpec(**json.loads(Path(spec_path).read_text(encoding="utf-8")))
        stop = threading.Event()

        stop_file = Path(spec_path + ".stop")

        def watch_stop_file() -> None:                   # (never read stdin here: see the module docstring)
            while not stop.is_set():
                if stop_file.exists():
                    stop.set()
                    return
                time.sleep(0.25)

        threading.Thread(target=watch_stop_file, name="gp-stop-file", daemon=True).start()
        entries = [DriverEntry(e["source"], house=bool(e.get("house")), params=e.get("params")) for e in spec.entries]
        cfg = GrandPrixConfig(**spec.config)
        gp = run_grand_prix(entries, cfg, progress=lambda u: send({"kind": "update", "u": u}), stop_event=stop,
                            inprocess=spec.inprocess)
        if stop.is_set():
            send({"kind": "stopped"})
            return 0
        paths = gp.save(spec.out_dir, spec.stem)
        try:
            from agentic_gp.race.leaderboard import standings_markdown

            md = Path(spec.out_dir) / f"{spec.stem}_standings.md"
            md.write_text(standings_markdown(gp), encoding="utf-8")
            paths["standings"] = md
        except Exception:                                # noqa: BLE001 - the markdown is a bonus
            pass
        standings = gp.to_dict().get("standings", [])
        send({"kind": "done", "paths": {k: str(v) for k, v in paths.items()},
              "standings": [{k: r.get(k) for k in ("position", "name", "race_time", "status", "house")}
                            for r in standings]})
        return 0
    except BaseException as exc:                         # noqa: BLE001 - report every failure to the app
        import traceback

        traceback.print_exc()
        send({"kind": "error", "error": f"{type(exc).__name__}: {exc}"[:1500]})
        return 1


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    os._exit(_worker(sys.argv[1]))                       # do not wait for daemon threads / leftover children
