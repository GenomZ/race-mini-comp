"""Run a driver experiment in a separate Python process that can be killed (student files never run in the app).

    from app.lib.sandbox import is_trusted, run_driver_job, SandboxError

    data = run_driver_job("app.lib.experiments:record_ticks", path,
                          {"driver": str(path), "track": "sprint", "max_ticks": 240}, timeout_s=30)

Why: the Streamlit server is ONE process shared by every browser tab. A student's drive() with an
endless loop, or an `os._exit()`, would freeze or kill Race Control for everybody if it ran here.
So every file outside drivers/examples/ runs in a child process:

* the child gets the job (a module:function name + JSON keyword arguments) in a temp file,
* it writes the JSON result (or the error) to another temp file,
* the parent waits at most `timeout_s` seconds, then kills it (SandboxError "did not finish").

The reference drivers in drivers/examples/ are ours and fast, so they run in-process (no 1-2 s
start-up). The child always uses the offline mock LLM unless `env` says otherwise.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .paths import DRIVERS_DIR, REPO_ROOT

EXAMPLES_DIR = DRIVERS_DIR / "examples"
API_KEY_VARS = ("MINIMAX_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")   # agentic_gp.llm.API_KEYS


class SandboxError(RuntimeError):
    """The sandboxed job failed: timed out, the process died, or the function raised."""


def is_trusted(path: str | Path | None) -> bool:
    """True for the reference drivers in drivers/examples/ (safe to run inside the app process)."""
    if path is None:
        return False
    try:
        p = Path(path).resolve()
        return p.parent == EXAMPLES_DIR.resolve() and p.suffix == ".py" and p.is_file()
    except OSError:
        return False


def _call(target: str, kwargs: dict) -> Any:
    module, _, name = target.partition(":")
    return getattr(importlib.import_module(module), name)(**kwargs)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)


def run_sandboxed(target: str, kwargs: dict, *, timeout_s: float = 30.0, env: dict | None = None) -> Any:
    """Call `module:function(**kwargs)` in a child Python process; return its JSON result.

    Raises SandboxError with a short, human message (timeout, exit code, or the exception).
    """
    tmp = Path(tempfile.gettempdir()) / "agp_sandbox"
    tmp.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    job, out = tmp / f"{job_id}_job.json", tmp / f"{job_id}_out.json"
    job.write_text(json.dumps({"target": target, "kwargs": kwargs}, default=_json_default), encoding="utf-8")
    child_env = dict(os.environ, LLM_PROVIDER="mock", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", LLM_NO_DOTENV="1")
    child_env.update({k: str(v) for k, v in (env or {}).items()})
    for var in API_KEY_VARS:                             # student code never shares a process with a key
        child_env.pop(var, None)
    child_env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([child_env["PYTHONPATH"]]
                                                                  if child_env.get("PYTHONPATH") else []))
    try:
        try:
            p = subprocess.run([sys.executable, "-m", "app.lib.sandbox", str(job), str(out)], cwd=str(REPO_ROOT),
                               env=child_env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise SandboxError(f"did not finish within {timeout_s:.0f} s - stopped it (an endless loop in "
                               f"plan() or drive()?)") from None
        if out.exists():
            res = json.loads(out.read_text(encoding="utf-8"))
            if res.get("ok"):
                return res.get("result")
            raise SandboxError(res.get("error") or "the job failed")
        tail = (p.stderr or p.stdout or "").strip()[-400:]
        raise SandboxError(f"the driver process exited with code {p.returncode} before it finished"
                           + (f": {tail}" if tail else " (os._exit() or sys.exit() in the driver?)"))
    finally:
        for f in (job, out):
            try:
                f.unlink()
            except OSError:
                pass


def run_driver_job(target: str, driver_path: str | Path | None, kwargs: dict, *, timeout_s: float = 30.0,
                   env: dict | None = None) -> Any:
    """In-process for drivers/examples/ (fast), in a killable child process for every other file."""
    if is_trusted(driver_path) and not env:
        return json.loads(json.dumps(_call(target, kwargs), default=_json_default))
    return run_sandboxed(target, kwargs, timeout_s=timeout_s, env=env)


def _guard(job: dict, out_file: Path) -> None:
    """The race guard (agentic_gp.race.guard) for this job: the driver may read itself and its track file."""
    from agentic_gp.race.guard import install_guard

    importlib.import_module(job["target"].partition(":")[0])   # our own module: imported before the guard
    kw = job.get("kwargs") or {}
    track = kw.get("track")
    reads = [track[5:]] if isinstance(track, str) and track.startswith("file:") else []
    install_guard(kw.get("driver"), write_files=[out_file], read_paths=reads)


def _main(argv: list[str]) -> int:
    job_file, out_file = Path(argv[0]), Path(argv[1])
    job = json.loads(job_file.read_text(encoding="utf-8"))
    sys.stdout = sys.stderr                              # the driver's print() must not get in the way
    _guard(job, out_file)
    try:
        result = {"ok": True, "result": _call(job["target"], job["kwargs"])}
    except BaseException as exc:                         # noqa: BLE001 - report everything, incl. SystemExit
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]}
    out_file.write_text(json.dumps(result, default=_json_default), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(_main(sys.argv[1:]))
