import multiprocessing as mp
import time
import textwrap
from pathlib import Path

import pytest

from agentic_gp.driver import Action
from agentic_gp.race.loader import (
    PALETTE, DriverDied, DriverError, DriverLoadError, DriverTimeout, IsolatedDriver, check_driver_source,
    coerce_params, color_for, load_driver, parse_param_args,
)

FUNCTION_STYLE = """
NAME = "Rocket"
AUTHOR = "ada"
COLOR = "#E63946"
KIND = "code"
PARAMS = {"gain": 1.0, "laps": 2, "brave": False}
seen = []

def plan(track):
    seen.append(track["name"])

def drive(obs):
    return PARAMS["gain"] * 0.5, 0.1

def reflect(report):
    seen.append(report["attempt"])
"""

CLASS_STYLE = """
from agentic_gp.driver import Driver as Base

class Driver(Base):
    NAME = "Max"
    AUTHOR = "house"
    KIND = "llm"
    MODEL = "MiniMax-M3"
    MOTTO = "plan once, drive fast"
    ANATOMY = ["map", "plan", "llm"]
    PARAMS = {"aggression": 1.0}

    def __init__(self):
        self.memory = []

    def drive(self, obs):
        self.memory.append(obs["tick"])
        return {"throttle": self.PARAMS["aggression"], "steering": 0, "say": "hi", "debug": {"n": len(self.memory)}}
"""

PUPPET = """
import os
import time

NAME = "Puppet"
AUTHOR = "tests"

class Driver:
    def __init__(self):
        self.calls = 0

    def plan(self, track):
        print("planning on", track["name"])

    def drive(self, obs):
        self.calls += 1
        cmd = obs.get("cmd")
        if cmd == "sleep":
            time.sleep(obs["seconds"])
        if cmd == "raise":
            raise ValueError("boom")
        if cmd == "exit":
            os._exit(7)
        return {"throttle": 0.5, "steering": -0.25, "say": f"call {self.calls}"}
"""


def write(dirpath: Path, name: str, source: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(textwrap.dedent(source), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_QUIET", "1")


# ------------------------------------------------------------------ in-process loading
def test_function_style(tmp_path):
    drv = load_driver(write(tmp_path, "rocket.py", FUNCTION_STYLE))
    assert (drv.name, drv.author, drv.kind, drv.color) == ("Rocket", "ada", "code", "#e63946")
    assert drv.params == {"gain": 1.0, "laps": 2, "brave": False}
    assert drv.drive({"tick": 0}) == Action(0.5, 0.1)
    drv.plan({"name": "Oval"})
    drv.reflect({"attempt": 1})
    assert drv.source_path.endswith("rocket.py")


def test_class_style_metadata_and_debug(tmp_path):
    drv = load_driver(write(tmp_path, "max.py", CLASS_STYLE))
    assert (drv.name, drv.kind, drv.model, drv.motto, drv.anatomy) == (
        "Max", "llm", "MiniMax-M3", "plan once, drive fast", ["map", "plan", "llm"])
    assert drv.color in PALETTE                       # no COLOR: stable palette colour by name
    assert drv.color == load_driver(write(tmp_path / "b", "max.py", CLASS_STYLE)).color
    a = drv.drive({"tick": 3})
    assert a == Action(1.0, 0.0, "hi", {"n": 1})
    drv.plan({})                                      # inherited no-op
    assert drv.uses_llm is False


def test_params_override_before_instantiation(tmp_path):
    drv = load_driver(write(tmp_path, "max.py", CLASS_STYLE), params={"aggression": "0.25"})
    assert drv.params == {"aggression": 0.25}
    assert drv.drive({"tick": 0}).throttle == 0.25
    fn = load_driver(write(tmp_path, "rocket.py", FUNCTION_STYLE), params={"gain": "2", "extra": "[1, 2]"})
    assert fn.params == {"gain": 2.0, "laps": 2, "brave": False, "extra": [1, 2]}
    assert fn.drive({}).throttle == 1.0


def test_coerce_and_parse_params():
    assert coerce_params({"a": 1.0, "n": 2, "b": True, "s": "x"},
                         {"a": "1.5", "n": "3", "b": "no", "s": "y", "new": "7"}) == \
        {"a": 1.5, "n": 3, "b": False, "s": "y", "new": 7}
    assert parse_param_args(["a=1", "line = racing"]) == {"a": "1", "line": " racing"}
    with pytest.raises(ValueError):
        parse_param_args(["oops"])


def test_same_file_name_in_different_folders_does_not_clash(tmp_path):
    a = load_driver(write(tmp_path / "a", "driver.py", "NAME='A'\ndef drive(obs):\n    return 1, 0\n"))
    b = load_driver(write(tmp_path / "b", "driver.py", "NAME='B'\ndef drive(obs):\n    return 0, 1\n"))
    assert (a.name, b.name) == ("A", "B")
    assert a.drive({}) == Action(1.0, 0.0) and b.drive({}) == Action(0.0, 1.0)


def test_loading_the_same_file_twice_gives_independent_drivers(tmp_path):
    p = write(tmp_path, "max.py", CLASS_STYLE)
    d1, d2 = load_driver(p), load_driver(p)
    d1.drive({"tick": 0})
    d1.drive({"tick": 1})
    assert d2.drive({"tick": 0}).debug == {"n": 1}     # its own memory
    assert d1.drive({"tick": 2}).debug == {"n": 3}


def test_other_sources(tmp_path):
    fn = load_driver(lambda obs: (1, 0))
    assert fn.drive({}) == Action(1.0, 0.0) and fn.color.startswith("#")

    class Mine:
        NAME = "Inst"

        def drive(self, obs):
            return None

    inst = load_driver(Mine())
    assert inst.name == "Inst" and inst.drive({}) == Action()
    assert load_driver(Mine).name == "Inst"
    assert load_driver(inst) is inst


def test_file_without_name_uses_the_stem_and_hex_shorthand(tmp_path):
    drv = load_driver(write(tmp_path, "speedy_gonzales.py", "COLOR='#abc'\ndef drive(obs):\n    return 1, 0\n"))
    assert drv.name == "speedy_gonzales" and drv.color == "#aabbcc"
    assert color_for("x", "red") in PALETTE


@pytest.mark.parametrize("source, needle", [
    ("x = 1\n", "no drive"),
    ("def drive(obs)\n    return 1, 0\n", "syntax"),
    ("from agentic_gp.sim import RaceEnv\ndef drive(obs):\n    return 1, 0\n", "agentic_gp.sim"),
    ("import agentic_gp.race.session\ndef drive(obs):\n    return 1, 0\n", "agentic_gp.race"),
    ("raise RuntimeError('at import')\ndef drive(obs):\n    return 1, 0\n", "RuntimeError"),
])
def test_load_errors_are_friendly(tmp_path, source, needle):
    with pytest.raises(DriverLoadError, match=needle):
        load_driver(write(tmp_path, "bad.py", source))


# ------------------------------------------------------------------ static checks
def _errors(tmp_path, source, name="d.py"):
    issues = check_driver_source(write(tmp_path, name, source))
    return [i.message for i in issues if i.level == "error"], [i.message for i in issues if i.level == "warning"]


BODY = "\nNAME = 'Ok'\nAUTHOR = 'me'\ndef drive(obs):\n    return 1, 0\n"


@pytest.mark.parametrize("imports", [
    "import math, json, random, collections, os, time",
    "import numpy as np",
    "from agentic_gp import driverkit as dk, llm",
    "from agentic_gp.driverkit import PathFollower",
    "from agentic_gp.driver import Driver",
    "import agentic_gp.fly",
    "from langgraph.graph import StateGraph",
    "from langchain_core.messages import HumanMessage",
    "from __future__ import annotations",
])
def test_whitelisted_imports_pass(tmp_path, imports):
    errors, warnings = _errors(tmp_path, imports + BODY)
    assert errors == [] and warnings == []


@pytest.mark.parametrize("imports, needle", [
    ("from agentic_gp.sim import RaceEnv", "agentic_gp.sim"),
    ("from agentic_gp import race", "agentic_gp.race"),
    ("import agentic_gp.cli", "agentic_gp.cli"),
    ("from agentic_gp.viz import mpl", "agentic_gp.viz"),
    ("import agentic_gp", "import the helper"),
    ("import openai", "agentic_gp.llm"),
    ("import requests", "raw HTTP"),
    ("import httpx", "raw HTTP"),
    ("import anthropic", "agentic_gp.llm"),
    ("import subprocess", "not allowed"),
    ("import socket", "not allowed"),
    ("from urllib.request import urlopen", "not allowed"),
    ("import importlib", "not allowed"),
    ("import shutil", "not allowed"),
    ("from multiprocessing import Process", "not allowed"),
    ("import pandas", "whitelist"),
    ("from . import helper", "relative"),
    ("x = RaceEnv", "RaceEnv"),
])
def test_forbidden_imports_are_errors(tmp_path, imports, needle):
    errors, _ = _errors(tmp_path, imports + BODY)
    assert any(needle in e for e in errors), errors


def test_metadata_and_size_rules(tmp_path):
    errors, warnings = _errors(tmp_path, "NAME = 'A' * 1\nNAME2 = 1\ndef drive(obs):\n    return 0, 0\n")
    assert errors == [] and any("AUTHOR" in w for w in warnings)
    errors, _ = _errors(tmp_path, "NAME = 'Seventeen chars!!'\nAUTHOR = 'x'\ndef drive(obs):\n    return 0, 0\n")
    assert any("longer than 16" in e for e in errors)
    errors, warnings = _errors(tmp_path, "class Driver:\n    NAME = 'Cls'\n    KIND = 'robot'\n    ANATOMY = ['map', "
                                         "'wings']\n    def drive(self, obs):\n        return open('x').read()\n")
    assert any("open()" in e and "file I/O" in e for e in errors)          # file I/O: an error
    joined = " ".join(warnings)
    assert "KIND" in joined and "wings" in joined and "AUTHOR" in joined
    errors, _ = _errors(tmp_path, BODY + "\nx = eval('1')\nexec('y = 2')\n")
    assert sum("eval()" in e or "exec()" in e for e in errors) == 2       # dynamic code: errors
    big = write(tmp_path, "big.py", BODY + "#" * 101_000)
    assert any("limit is 100 KB" in i.message for i in check_driver_source(big))
    assert check_driver_source(tmp_path / "missing.py")[0].level == "error"


# ------------------------------------------------------------------ isolation (child process)
def test_isolated_driver_round_trip_timeouts_errors_and_death(tmp_path):
    path = write(tmp_path, "puppet.py", PUPPET)
    drv = load_driver(path, isolate=True, log_path=tmp_path / "puppet.log")
    try:
        assert drv.name == "Puppet" and drv.isolated and drv.is_alive()
        drv.begin_session("open", "Puppet")
        drv.plan({"name": "Oval"})
        assert drv.drive({"tick": 0}, timeout=30) == Action(0.5, -0.25, "call 1")
        with pytest.raises(DriverError, match="ValueError: boom"):
            drv.drive({"cmd": "raise"}, timeout=30)
        with pytest.raises(DriverTimeout):
            drv.drive({"cmd": "sleep", "seconds": 1.0}, timeout=0.2)
        # the late answer ("call 3") arrives later and is discarded; the next call gets its own answer
        assert drv.drive({"tick": 1}, timeout=30).say == "call 4"
        assert drv.wait_idle(1.0)
        assert drv.llm_usage()["calls"] == 0
        with pytest.raises(DriverDied, match="exit code 7"):
            drv.drive({"cmd": "exit"}, timeout=30)
        assert drv.dead and not drv.is_alive()
        with pytest.raises(DriverDied):
            drv.drive({}, timeout=1)
    finally:
        drv.close()
    log = (tmp_path / "puppet.log").read_text(encoding="utf-8")
    assert "planning on Oval" in log and "ValueError: boom" in log
    assert mp.active_children() == []


def test_isolation_works_from_an_unguarded_main_script(tmp_path):
    """Streamlit pages, notebooks and quick scripts have no `if __name__ == "__main__":` guard.
    The child must not re-run them (that would recurse or crash)."""
    import subprocess
    import sys

    drv = write(tmp_path, "rocket.py", FUNCTION_STYLE)
    script = write(tmp_path, "page.py", f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        from agentic_gp.race import load_driver
        d = load_driver({str(drv)!r}, isolate=True)
        print("drove", d.drive({{"tick": 0}}, timeout=60).throttle)
        d.close()
    """)
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=240, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "drove 0.5"


def test_isolated_load_error_leaves_no_process(tmp_path):
    bad = write(tmp_path, "bad.py", "NAME='Bad'\nAUTHOR='x'\nraise RuntimeError('broken at import')\n"
                                    "def drive(obs):\n    return 1, 0\n")
    with pytest.raises(DriverLoadError, match="broken at import"):
        load_driver(bad, isolate=True, log_path=tmp_path / "bad.log")
    with pytest.raises(DriverLoadError, match="agentic_gp.sim"):     # static check runs before spawning
        load_driver(write(tmp_path, "cheat.py", "import agentic_gp.sim" + BODY), isolate=True)
    assert mp.active_children() == []


# ------------------------------------------------------------------ review fixes
@pytest.mark.parametrize("source, needle", [
    ("import os\nos.system('calc')", "os.system starts processes"),
    ("import os\nos.spawnv(os.P_NOWAIT, 'x', ['x'])", "os.spawnv starts processes"),
    ("from os import startfile", "os.startfile starts processes"),
    ("import asyncio\nasyncio.create_subprocess_exec('x')", "asyncio.create_subprocess_exec starts processes"),
    ("from concurrent.futures import ProcessPoolExecutor", "ProcessPoolExecutor starts processes"),
    ("from agentic_gp import llm\nllm._swap_fallback(())", "llm._swap_fallback is private"),
    ("from agentic_gp.llm import _POLICY", "agentic_gp.llm internals"),
    # review round 3: ways around the LLM rules and the name checks
    ("from agentic_gp import llm\nwith llm.llm_policy(phase='plan'):\n    pass", "llm.llm_policy"),
    ("from agentic_gp.llm import llm_policy", "llm_policy are off limits"),
    ("import contextvars", "contextvars is not allowed"),
    ("from contextvars import Context", "contextvars is not allowed"),
    ("x = __import__('open' + 'ai')", "__import__() - dynamic code"),
    ("x = compile('1', 'f', 'eval')", "compile() - dynamic code"),
    ("import os\nf = getattr(os, 'spawn' + 'v')", "getattr(os, <computed name>)"),
    ("import os\nf = getattr(os, 'system')", "os.system starts processes"),
    ("from agentic_gp import llm\nf = getattr(llm, '_POL' + 'ICY')", "getattr(llm, <computed name>)"),
    ("from agentic_gp import llm\nf = getattr(llm, '_POLICY')", "llm._POLICY is private"),
    ("import sys\nm = sys.modules['os']", "sys.modules"),
    ("import os\nd = os.__dict__", "os.__dict__"),
    ("import os\nd = vars(os)", "vars(os)"),
    ("b = __builtins__", "__builtins__"),
    ("import builtins", "import builtins is not allowed"),
    ("import _winapi", "import _winapi is not allowed"),
    ("x = ().__class__.__subclasses__()", "__subclasses__"),
])
def test_static_check_refuses_processes_and_llm_internals(tmp_path, source, needle):
    errors, _ = _errors(tmp_path, source + BODY)
    assert any(needle in e for e in errors), errors


def test_static_check_warns_about_threads_and_computed_names(tmp_path):
    errors, warnings = _errors(tmp_path, "import threading\nfrom agentic_gp import llm\n"
                                         "x = getattr(llm, 'chat')\nkey = 'a'\ny = getattr(dict, key, None)\n"
                                         "NAME = 'Rocket' * 50\nAUTHOR = 'x'\ndef drive(obs):\n    return 1, 0\n")
    assert errors == []                              # getattr by name / on non-modules is fine
    joined = " ".join(warnings)
    assert "import threading" in joined and "NAME is computed" in joined


def test_computed_long_name_is_cut_to_16_chars(tmp_path):
    drv = load_driver(write(tmp_path, "long.py", "NAME = 'Rocket\\n' * 50\nAUTHOR = 'x'\n"
                                                 "def drive(obs):\n    return 1, 0\n"))
    assert drv.name == "Rocket Rocket Ro" and len(drv.name) == 16


def test_sys_exit_at_import_or_in_init_is_a_load_error(tmp_path):
    with pytest.raises(DriverLoadError, match="SystemExit"):
        load_driver(write(tmp_path, "quit.py", "import sys\nsys.exit('bye')\n" + BODY))
    with pytest.raises(DriverLoadError, match="Driver\\(\\) failed: SystemExit"):
        load_driver(write(tmp_path, "quit2.py", "import sys\nclass Driver:\n    def __init__(self):\n"
                                                "        sys.exit(3)\n    def drive(self, obs):\n        return 1, 0\n"))


def test_unknown_params_are_reported_and_bad_values_explained(tmp_path):
    path = write(tmp_path, "rocket.py", FUNCTION_STYLE)
    drv = load_driver(path, params={"gian": "2", "gain": "1.5"})
    assert drv.unknown_params == ["gian"] and drv.params["gain"] == 1.5
    with pytest.raises(ValueError, match="--param gain expects a number \\(default 1.0\\), got 'fast'"):
        load_driver(path, params={"gain": "fast"})
    with pytest.raises(ValueError, match="--param laps expects a whole number"):
        coerce_params({"laps": 2}, {"laps": "two"})


def test_non_finite_params_stay_json_safe_in_meta(tmp_path):
    import json

    drv = load_driver(write(tmp_path, "w.py", "PARAMS = {'limit': float('inf'), 'n': float('nan'), 'v': [1.0]}"
                                              + BODY))
    assert drv.params["limit"] == float("inf")                   # the driver still sees its own values
    meta = drv.meta()
    assert meta["params"] == {"limit": "inf", "n": "nan", "v": [1.0]}
    json.dumps(meta, allow_nan=False)


def test_bad_drive_answers_are_driver_errors(tmp_path):
    from agentic_gp.race.loader import BadAction

    drv = load_driver(write(tmp_path, "s.py", "NAME='S'\nAUTHOR='x'\ndef drive(obs):\n    return 'full throttle'\n"))
    with pytest.raises(BadAction, match="returned str 'full throttle'"):
        drv.drive({})


def test_official_load_refuses_llm_calls(tmp_path):
    src = "from agentic_gp import llm\nANSWER = llm.chat('at import')\n" + BODY
    with pytest.raises(DriverLoadError, match="LLMNotAllowed"):
        load_driver(write(tmp_path, "imp.py", src), llm_mode="official")
    assert load_driver(write(tmp_path, "imp2.py", src)).name == "Ok"              # open mode: fine


def test_plan_gets_an_llm_deadline_before_its_time_limit(tmp_path):
    src = ("from agentic_gp import llm\nNAME='D'\nAUTHOR='x'\nSEEN = {}\n"
           "def plan(track):\n    SEEN['left'] = llm.current_llm_policy()['seconds_left']\n"
           "def drive(obs):\n    return 1, 0, str(SEEN.get('left'))\n")
    drv = load_driver(write(tmp_path, "dl.py", src))
    drv.plan({}, timeout=20.0)
    assert 17.0 < float(drv.drive({}).say) <= 18.0                # 20 s - 2 s margin
    iso = load_driver(write(tmp_path, "dl2.py", src), isolate=True)
    try:
        iso.plan({}, timeout=60.0)                                # the timeout travels to the child
        assert 53.0 < float(iso.drive({}, timeout=30).say) <= 55.0
    finally:
        iso.close()


def test_child_log_is_capped_and_tail_reads_little(tmp_path):
    from agentic_gp.race.loader import _CappedLog, _log_tail

    log = _CappedLog(str(tmp_path / "spam.log"), limit=200_000, keep=50_000)
    log.write("FIRST LINE\n")
    for i in range(20_000):
        log.write(f"spam line {i:06d} ........................................\n")
    log.write("Traceback: the last words\n")
    for _ in range(2):                                            # the next check runs after 64 KB more
        log.write("x" * 70_000 + "\n")
    log.write("the very end\n")
    log.flush()
    size = (tmp_path / "spam.log").stat().st_size
    text = (tmp_path / "spam.log").read_text(encoding="utf-8")
    assert size < 400_000 and log.trims >= 1
    assert text.startswith("FIRST LINE") and "log trimmed" in text and text.rstrip().endswith("the very end")
    assert _log_tail(str(tmp_path / "spam.log")).endswith("the very end")
    log.fh.close()


def test_killed_child_gives_back_its_llm_slots(tmp_path):
    """A driver killed in the middle of an LLM request must not keep a slot of the shared semaphore."""
    ctx = mp.get_context("spawn")
    sem = ctx.BoundedSemaphore(1)
    src = ("import time\nfrom agentic_gp import llm\nNAME='Hog'\nAUTHOR='x'\n"
           "def plan(track):\n    llm._semaphore().acquire()\n    time.sleep(60)\n"
           "def drive(obs):\n    return 1, 0\n")
    drv = load_driver(write(tmp_path, "hog.py", src), isolate=True, check=False, llm_semaphore=sem)
    with pytest.raises(DriverTimeout):
        drv.plan({}, timeout=1.5)                                 # it now holds the only slot
    assert not sem.acquire(timeout=0.1)
    drv.kill()
    assert sem.acquire(timeout=2.0)                               # given back by the parent
    sem.release()


def test_processes_a_driver_starts_die_with_it(tmp_path):
    import subprocess
    import sys

    pidfile = tmp_path / "grandchild.pid"
    src = (f"import subprocess, sys\nNAME='Spawner'\nAUTHOR='x'\n"
           f"def plan(track):\n    p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
           f"    open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
           f"def drive(obs):\n    return 1, 0\n")
    path = write(tmp_path, "spawner.py", src)
    with pytest.raises(DriverError, match="race guard"):                # the guard refuses it first ...
        drv = load_driver(path, isolate=True, check=False)              # (and the static check before it)
        try:
            drv.plan({}, timeout=60)
        finally:
            drv.close()
    drv = IsolatedDriver(path, check=False, guard=False)                # ... and the process containment
    drv.plan({}, timeout=60)                                            # behind the guard still holds
    pid = int(pidfile.read_text())
    drv.close()
    deadline = time.monotonic() + 5
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = _pid_alive(pid)
    if alive:                                                     # clean up before failing
        subprocess.run(["taskkill", "/F", "/PID", str(pid)] if sys.platform == "win32" else ["kill", "-9", str(pid)],
                       capture_output=True)
    assert not alive


def _pid_alive(pid: int) -> bool:
    import os
    import subprocess
    import sys

    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:                                                          # a zombie child is dead for our purposes
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def test_child_exits_when_the_parent_dies(tmp_path):
    import subprocess
    import sys

    hang = write(tmp_path, "hang.py", "NAME='Hang'\nAUTHOR='x'\ndef drive(obs):\n    while True:\n        pass\n")
    root = str(Path(__file__).resolve().parent.parent)
    runner = write(tmp_path, "runner.py",
                   "import os, sys\n"
                   f"sys.path.insert(0, {root!r})\n"
                   "from agentic_gp.race import load_driver, DriverTimeout\n"
                   "if __name__ == '__main__':\n"
                   f"    d = load_driver({str(hang)!r}, isolate=True)\n"
                   "    print(d.pid, flush=True)\n"
                   "    try:\n"
                   "        d.drive({}, timeout=0.5)\n"
                   "    except DriverTimeout:\n"
                   "        pass\n"
                   "    os._exit(1)                       # hard death: no atexit, no close()\n")
    out = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, timeout=240, cwd=tmp_path)
    pid = int(out.stdout.split()[0])
    deadline = time.monotonic() + 10
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not _pid_alive(pid)


def test_llm_warm_up_does_not_count_against_the_load_timeout(tmp_path, monkeypatch):
    """A cold `import openai` can take minutes on a busy machine: the child says hello first."""
    fake = tmp_path / "fakesdk" / "openai"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("import time, types\ntime.sleep(10)\nclass OpenAI:\n"
                                      "    def __init__(self, **kw):\n"
                                      "        self.chat = types.SimpleNamespace(completions=None)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "fakesdk"))       # spawned children copy sys.path
    src = "from agentic_gp import llm\n" + BODY
    # the 10 s import is longer than load_timeout: before the fix this was a load error (DNF). (8 s leaves
    # a busy laptop time to start the child and say hello.)
    drv = load_driver(write(tmp_path, "llmdrv.py", src), isolate=True, load_timeout=8.0,
                      env={"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "not-a-key"})
    try:
        assert drv.uses_llm and drv.is_alive()
        assert drv.drive({}, timeout=20) == Action(1.0, 0.0)  # answered once the warm-up is done
    finally:
        drv.close()


# ------------------------------------------------------------------ review round 3
def _fake_sdk(tmp_path, monkeypatch, seconds: float) -> None:
    fake = tmp_path / f"fakesdk{seconds:g}" / "openai"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(f"import time, types\ntime.sleep({seconds})\nclass OpenAI:\n"
                                      "    def __init__(self, **kw):\n"
                                      "        self.chat = types.SimpleNamespace(completions=None)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(fake.parent))                 # spawned children copy sys.path


@pytest.mark.parametrize("load_timeout, finishes", [(10.0, True), (6.0, False)])
def test_begin_session_waits_for_the_sdk_warm_up(tmp_path, monkeypatch, load_timeout, finishes):
    """The first call (begin_session, 30 s limit) used to time out behind a slow SDK import and was
    reported as a crash. Now it waits for the warm-up's second window; past 2 x load_timeout the car
    is "DNF (timeout: loading)". (The 15 s import vs 2 x 10 s / 2 x 6 s windows; the windows are this
    long because starting the child and saying hello alone took over 3 s on a busy laptop.)"""
    from agentic_gp.race import SessionConfig, run_session

    _fake_sdk(tmp_path, monkeypatch, 15)
    src = "from agentic_gp import llm\n" + BODY
    stages: list[str] = []
    drv = load_driver(write(tmp_path, "llmdrv.py", src), isolate=True, load_timeout=load_timeout,
                      env={"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "not-a-key"}, on_stage=stages.append)
    try:
        result = run_session(drv, SessionConfig(track="oval", attempts=1, laps=1, max_ticks_per_lap=60))
    finally:
        drv.close()
    assert stages[:2] == ["warm-up", "warm-up"]                    # both windows announced themselves
    if finishes:
        assert not result.status.startswith("DNF (timeout"), result.status
    else:
        assert result.status == "DNF (timeout: loading)"
        assert any("still importing the LLM SDK" in e["error"] for e in result.driver_errors)


CHEATS = {
    "thread": ("import threading\nfrom agentic_gp import llm\nNAME='T'\nAUTHOR='x'\nN = {'ok': 0, 'no': 0}\n"
               "def _bg():\n    try:\n        with llm.llm_policy(phase='plan'):\n            llm.chat('x')\n"
               "        N['ok'] += 1\n    except llm.LLMError:\n        N['no'] += 1\n"
               "def drive(obs):\n    if obs['tick'] < 3:\n        t = threading.Thread(target=_bg)\n"
               "        t.start()\n        t.join()\n    return 0.4, 0.0, f\"{N['ok']}/{N['no']}\"\n"),
    "context": ("import contextvars\nfrom agentic_gp import llm\nNAME='C'\nAUTHOR='x'\nN = {'ok': 0, 'no': 0}\n"
                "def _ask():\n    try:\n        with llm.llm_policy(phase='plan'):\n            llm.chat('x')\n"
                "        N['ok'] += 1\n    except llm.LLMError:\n        N['no'] += 1\n"
                "def drive(obs):\n    if obs['tick'] < 3:\n        contextvars.Context().run(_ask)\n"
                "    return 0.4, 0.0, f\"{N['ok']}/{N['no']}\"\n"),
}


@pytest.mark.parametrize("isolate", [False, True])
@pytest.mark.parametrize("cheat", sorted(CHEATS))
def test_own_scopes_cannot_escape_official_rules_at_run_time(tmp_path, cheat, isolate):
    """With the static check bypassed: an own llm_policy scope in a thread or a fresh Context made
    uncounted LLM calls during the race. Now every one is refused."""
    from agentic_gp import llm
    from agentic_gp.race import SessionConfig, run_session

    path = write(tmp_path, f"{cheat}.py", CHEATS[cheat])
    assert any(i.level == "error" for i in check_driver_source(path))      # the static check refuses it
    drv = load_driver(path, check=False, isolate=isolate, llm_mode="official")
    try:
        res = run_session(drv, SessionConfig(track="oval", attempts=2, laps=1, llm_policy="official",
                                             phases=["practice", "race"], max_ticks_per_lap=40))
    finally:
        drv.close()
    texts = [m["text"] for a in res.attempts for m in a.messages]
    assert texts and texts[-1] == "0/6" and all(t.startswith("0/") for t in texts), texts   # 6 tried, 0 got through
    assert not llm.official_session_active() and llm.llm_allowed()


def test_overlapping_official_driver_sessions_release_the_process(tmp_path):
    """Two drivers in official sessions that begin A, B and end A, B left every LLM call refused."""
    from agentic_gp import llm

    src = "from agentic_gp import llm\n" + BODY
    a, b = load_driver(write(tmp_path, "a.py", src)), load_driver(write(tmp_path, "b.py", src))
    a.begin_session("official", "A")
    b.begin_session("official", "B")
    a.drive({}), b.drive({}), a.plan({}), b.reflect({})
    a.end_session()
    assert not llm.llm_allowed()                                  # B is still between calls
    b.end_session()
    assert llm.llm_allowed() and llm.chat("free") == llm.MOCK_TEXT


def test_isolated_driver_cannot_read_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-should-stay-hidden")
    _fake_sdk(tmp_path, monkeypatch, 0)                           # the warm-up imports a fake SDK
    src = ("import os\nfrom agentic_gp import llm\nNAME='K'\nAUTHOR='x'\n"
           "def drive(obs):\n    return 0, 0, f\"{os.environ.get('MINIMAX_API_KEY')}|{llm.get_config().api_key}\"\n")
    drv = load_driver(write(tmp_path, "k.py", src), isolate=True, env={"LLM_PROVIDER": "minimax"})
    try:
        assert drv.drive({}, timeout=10).say == f"None|{llm_redacted()}"
    finally:
        drv.close()


def llm_redacted() -> str:
    from agentic_gp import llm
    return llm.REDACTED_KEY
