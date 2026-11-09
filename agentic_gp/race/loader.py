"""Load driver files: static checks, fresh modules, and the isolated (child process) proxy.

    from agentic_gp.race import load_driver, check_driver_source
    issues = check_driver_source("drivers/submissions/rocket.py")    # [Issue(level, message), ...]
    drv = load_driver("drivers/examples/l2_planner.py")              # runs in this process
    drv = load_driver("rocket.py", isolate=True)                     # runs in a child process
    drv.plan(track_map); action = drv.drive(obs); drv.reflect(report)
    drv.close()                                                      # stops the child (isolated only)

Every load executes the file in a FRESH, uniquely named module: two files called driver.py in
different folders never clash, and loading the same file twice gives two independent drivers.

Isolation (`isolate=True`): the driver file runs in a `spawn`ed child process (works on Windows,
macOS and Linux); the simulation stays in the parent. Requests and answers travel over a Pipe and
are matched by a sequence id, so an answer that arrives after its timeout is simply discarded.
The child's print() output goes to a log file. A driver that hangs or kills its own process can
never freeze the race: calls time out (DriverTimeout) or report the death (DriverDied).

LLM rules: the runner tells the driver which phase it is in (`drv.phase = "race"`) and which policy
applies (`drv.begin_session(mode="official")`); every call is wrapped in `agentic_gp.llm.llm_policy`
scopes (track > phase > plan|drive|reflect) - in this process or inside the child.
"""

from __future__ import annotations

import ast
import atexit
import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import llm
from ..driver import ANATOMY_TAGS, KINDS, Action, action_problem, normalize_action
from .guard import install_guard

PALETTE = (
    "#e63946", "#457b9d", "#2a9d8f", "#f4a261", "#8338ec", "#ff006e",
    "#3a86ff", "#fb5607", "#06d6a0", "#ffbe0b", "#8d99ae", "#b5179e",
)
MAX_FILE_BYTES = 100_000
MAX_NAME_CHARS = 16
# The house cars' NAMEs (Max = l6_llm_crew_chief, Planner = l2_planner, Fly Toy = fly_toy): a student car
# with one of these names would be mixed up with the house car in standings, awards and LLM logs.
HOUSE_NAMES = ("Max", "Planner", "Fly Toy")
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "drivers" / "examples"

# --- import whitelist (A11) ---------------------------------------------------------------
DENIED_STDLIB = {"socket", "ssl", "subprocess", "multiprocessing", "http", "urllib", "ftplib", "smtplib",
                 "ctypes", "importlib", "shutil", "builtins", "runpy", "pty", "nt", "posix", "_winapi",
                 "_posixsubprocess", "_socket", "_ssl", "_ctypes", "_multiprocessing", "_imp",
                 # ways around the name checks: arbitrary calls on load, live frames and objects, code objects
                 "pickle", "_pickle", "marshal", "shelve", "copyreg", "gc", "inspect", "code", "codeop"}
# The runner owns the LLM rules: a driver may not open its own policy scopes or contexts.
DENIED_RULE_MODULES = {"contextvars": "contextvars is not allowed in a driver (the runner owns the LLM rules)",
                       "_contextvars": "contextvars is not allowed in a driver (the runner owns the LLM rules)"}
LLM_RULE_NAMES = ("llm_policy", "LLMPolicy")
DENIED_PACKAGES = {"openai": "call the LLM through agentic_gp.llm (it enforces the race rules)",
                   "anthropic": "call the LLM through agentic_gp.llm",
                   "requests": "no raw HTTP in a driver", "httpx": "no raw HTTP in a driver",
                   "aiohttp": "no raw HTTP in a driver"}
DENIED_AGP = {"agentic_gp.sim": "the runner owns the simulation - use the obs/track dicts you are given",
              "agentic_gp.race": "the runner owns the race", "agentic_gp.cli": "the CLI is not for drivers",
              "agentic_gp.viz": "plotting is not part of a driver"}
ALLOWED_PACKAGES = ("numpy", "langgraph", "langchain_core", "langchain_openai")
ALLOWED_AGP = ("agentic_gp.driverkit", "agentic_gp.llm", "agentic_gp.fly", "agentic_gp.driver")
FILE_CALLS = ("open",)
# Methods that read or write files (pathlib, io, os, numpy): no file I/O in a driver, also not through a Path.
FILE_METHODS = ("read_text", "read_bytes", "write_text", "write_bytes", "open", "FileIO", "fdopen", "loadtxt",
                "genfromtxt", "fromfile", "tofile", "savez", "savez_compressed", "savetxt", "memmap")
NUMPY_FILE_FUNCS = ("load", "save")
DYNAMIC_CALLS = ("eval", "exec", "compile", "__import__", "globals", "breakpoint")   # errors: they get around every other check
# Attributes that reach interpreter internals (frames, builtins, raw attribute lookup) or call by computed name.
INTERNAL_ATTRS = ("__subclasses__", "__globals__", "__builtins__", "__code__", "__getattribute__", "__import__",
                  "__loader__", "__spec__", "__reduce__", "__reduce_ex__", "f_globals", "f_locals", "f_builtins",
                  "f_back", "gi_frame", "cr_frame", "ag_frame", "tb_frame", "_getframe", "ctypeslib",
                  "attrgetter", "methodcaller")
MODULE_INTROSPECTION = ("vars", "dir")
# Starting processes: these outlive the race and escape every time limit (error, like `subprocess`).
PROCESS_CALLS = {
    "os": ("system", "popen", "startfile", "fork", "forkpty", "kill", "killpg", "posix_spawn", "posix_spawnp",
           "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
           "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"),
    "asyncio": ("create_subprocess_exec", "create_subprocess_shell"),
}
PROCESS_NAMES = ("ProcessPoolExecutor",)
WARN_MODULES = {"threading": "threads", "concurrent": "threads", "_thread": "threads", "asyncio": "event loops"}


class DriverLoadError(Exception):
    """The driver file cannot be raced (syntax error, missing drive(), forbidden import, ...)."""


class DriverError(Exception):
    """An exception raised by driver code inside the isolated child (message = 'Type: text')."""

    def __init__(self, message: str, traceback_text: str = "") -> None:
        super().__init__(message)
        self.traceback_text = traceback_text


class DriverTimeout(Exception):
    """An isolated driver did not answer in time (a stall for drive(); a timeout for plan/reflect)."""


class DriverLoadTimeout(DriverTimeout):
    """An isolated driver was still loading (importing the LLM SDK) when its first call came due."""


class DriverDied(Exception):
    """The isolated driver's process is gone (it crashed, exited or was killed)."""


# ============================================================================ static check
@dataclass(frozen=True)
class Issue:
    level: str        # "error" | "warning"
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.message}"


def check_driver_source(path: str | Path) -> list[Issue]:
    """Static (AST) scrutineering of a driver file. Errors mean the file cannot race.

    Errors: file > 100 KB, syntax error, file I/O (open, Path.read_text / write_text, np.load / save, ...), no drive() function and no Driver class, imports outside
    the whitelist (stdlib minus network/process modules, numpy, agentic_gp.driverkit / llm / fly,
    langgraph, langchain_core, langchain_openai), any use of RaceEnv, starting processes (os.system,
    os.spawn*, os.exec*, os.startfile, asyncio.create_subprocess_*, ProcessPoolExecutor, ...), private
    agentic_gp.llm internals (llm._anything), llm.llm_policy / contextvars (the runner owns the LLM
    rules), dynamic code (eval / exec / compile / __import__ / globals), getattr(module, <computed name>),
    sys.modules / module.__dict__ / vars(module), interpreter internals (__getattribute__, frames,
    __subclasses__, pickle, gc, inspect, operator.attrgetter), NAME longer than 16 chars.
    Warnings: threads, a NAME that is not a plain string, missing NAME or AUTHOR, unknown KIND /
    ANATOMY tags, a house car's NAME (Max, Planner, Fly Toy), emoji in NAME, a COLOR that is not hex.

    This is scrutineering, not a sandbox: isolated drivers are additionally contained by their
    process (a Job Object / process group kills everything they start), by the race guard (an audit
    hook that refuses processes, network, native code and file access at run time, see guard.py) and
    by the LLM relay (the API key never enters their process, see llm_proxy.py). In-process runs (--inprocess, the app's fast mode) are for trusted code.
    """
    path = Path(path)
    issues: list[Issue] = []
    if not path.is_file():
        return [Issue("error", f"file not found: {path}")]
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        issues.append(Issue("error", f"file is {size // 1000} KB - the limit is {MAX_FILE_BYTES // 1000} KB"))
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return issues + [Issue("error", f"syntax error on line {exc.lineno}: {exc.msg}")]
    except (UnicodeDecodeError, ValueError) as exc:
        return issues + [Issue("error", f"cannot read the file: {exc}")]

    modules = _module_names(tree)
    top_funcs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if "drive" not in top_funcs and "Driver" not in classes:
        issues.append(Issue("error", "no drive(obs) function and no Driver class found"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                issues += _check_import(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                issues.append(Issue("error", f"line {node.lineno}: relative imports are not allowed "
                                             "(a driver is ONE file)"))
                continue
            mod = node.module or ""
            for alias in node.names:                 # from operator import attrgetter, from numpy import load
                fake = ast.Attribute(value=ast.Name(id=mod.split(".")[0]), attr=alias.name)
                fake.lineno = node.lineno
                issues += [i for i in _check_attribute(fake, modules) if "internals" in i.message
                           or "file I/O" in i.message]
            if mod == "agentic_gp":
                for alias in node.names:
                    issues += _check_import(f"agentic_gp.{alias.name}", node.lineno)
            else:
                issues += _check_import(mod, node.lineno)
        elif isinstance(node, ast.Name) and node.id == "RaceEnv" or isinstance(node, ast.Attribute) \
                and node.attr == "RaceEnv":
            issues.append(Issue("error", f"line {node.lineno}: RaceEnv is not for drivers - the runner owns "
                                         "the simulation"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FILE_CALLS:
            issues.append(Issue("error", f"line {node.lineno}: {node.func.id}() - file I/O is not allowed in a "
                                         "driver (only what is in your file persists)"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_CALLS:
            issues.append(Issue("error", f"line {node.lineno}: {node.func.id}() - dynamic code and imports are "
                                         "not allowed in a driver"))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            issues += _check_dynamic_call(node, modules)
        elif isinstance(node, ast.Name) and node.id == "__builtins__":
            issues.append(Issue("error", f"line {node.lineno}: __builtins__ is not for drivers"))
        if isinstance(node, ast.Attribute):
            issues += _check_attribute(node, modules)
        elif isinstance(node, ast.Name) and node.id in PROCESS_NAMES:
            issues.append(Issue("error", f"line {node.lineno}: {node.id} starts processes - a driver may not "
                                         "start processes (they would outlive the race)"))
        elif isinstance(node, ast.ImportFrom) and node.module == "agentic_gp.llm" and (
                any(a.name.startswith("_") or a.name in LLM_RULE_NAMES for a in node.names)):
            issues.append(Issue("error", f"line {node.lineno}: agentic_gp.llm internals and llm_policy are off "
                                         "limits (the runner sets the LLM rules)"))
        elif isinstance(node, ast.ImportFrom) and any(a.name in PROCESS_NAMES for a in node.names):
            issues.append(Issue("error", f"line {node.lineno}: ProcessPoolExecutor starts processes - a driver "
                                         "may not start processes (they would outlive the race)"))
        elif isinstance(node, ast.ImportFrom) and node.module in PROCESS_CALLS:
            for alias in node.names:
                if alias.name in PROCESS_CALLS[node.module]:
                    issues.append(Issue("error", f"line {node.lineno}: {node.module}.{alias.name} starts processes "
                                                 "- not allowed in a driver"))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for mod in mods:
                what = WARN_MODULES.get(mod.split(".")[0])
                if what:
                    issues.append(Issue("warning", f"line {node.lineno}: import {mod} - {what} you start run outside "
                                                   "the runner's time limits; LLM calls from them follow the "
                                                   "official rules (refused outside plan/reflect)"))

    meta = _literal_metadata(tree, classes.get("Driver"))
    name = meta.get("NAME")
    if name is None:
        issues.append(Issue("warning", "no NAME - the file name will be used"))
    elif name == _COMPUTED:
        issues.append(Issue("warning", f"NAME is computed, not a plain string - it is cut to {MAX_NAME_CHARS} "
                                       "characters when the driver loads"))
    elif len(str(name)) > MAX_NAME_CHARS:
        issues.append(Issue("error", f"NAME {name!r} is longer than {MAX_NAME_CHARS} characters"))
    if isinstance(name, str) and name != _COMPUTED:
        if name.strip().lower() in {h.lower() for h in HOUSE_NAMES} and path.resolve().parent != EXAMPLES_DIR:
            issues.append(Issue("warning", f"NAME {name!r} is a house car's name - pick your own (scrutineering "
                                           "rejects it: standings, awards and LLM logs would mix you up)"))
        if _has_symbols(name):
            issues.append(Issue("warning", f"NAME {name!r} has emoji / symbols - the 3D viewer and some fonts "
                                           "drop them; use letters, digits and spaces"))
    color = meta.get("COLOR")
    if color is not None and color != _COMPUTED and not (
            isinstance(color, str) and re.fullmatch(r"\s*#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})\s*", color)):
        issues.append(Issue("warning", f"COLOR {color!r} is not a hex colour like \"#e63946\" - your car gets a "
                                       "palette colour instead"))
    if meta.get("AUTHOR") is None:
        issues.append(Issue("warning", "no AUTHOR - add your handle, e.g. AUTHOR = \"ada\""))
    kind = meta.get("KIND")
    if kind is not None and kind not in KINDS:
        issues.append(Issue("warning", f"KIND {kind!r} is not one of {KINDS}"))
    anatomy = meta.get("ANATOMY")
    if isinstance(anatomy, (list, tuple)):
        unknown = [a for a in anatomy if a not in ANATOMY_TAGS]
        if unknown:
            issues.append(Issue("warning", f"unknown ANATOMY tags {unknown}; use {ANATOMY_TAGS}"))
    return _dedupe(issues)


def _check_import(module: str, lineno: int) -> list[Issue]:
    root = module.split(".")[0]
    where = f"line {lineno}: import {module}"
    if root == "agentic_gp":
        for denied, why in DENIED_AGP.items():
            if module == denied or module.startswith(denied + "."):
                return [Issue("error", f"{where} is not allowed - {why}")]
        if any(module == ok or module.startswith(ok + ".") for ok in ALLOWED_AGP):
            return []
        if module == "agentic_gp":
            return [Issue("error", f"{where}: import the helper you need, e.g. `from agentic_gp import driverkit`")]
        return [Issue("error", f"{where} is not on the whitelist (allowed: {', '.join(ALLOWED_AGP)})")]
    if root in DENIED_PACKAGES:
        return [Issue("error", f"{where} is not allowed - {DENIED_PACKAGES[root]}")]
    if root in DENIED_RULE_MODULES:
        return [Issue("error", f"{where}: {DENIED_RULE_MODULES[root]}")]
    if root in DENIED_STDLIB:
        return [Issue("error", f"{where} is not allowed in a driver (no network, processes or dynamic imports)")]
    if root in ALLOWED_PACKAGES or root in sys.stdlib_module_names:
        return []
    return [Issue("error", f"{where} is not on the whitelist (stdlib, numpy, agentic_gp.driverkit/llm/fly, "
                           "langgraph, langchain_core, langchain_openai)")]


def _module_names(tree: ast.Module) -> set[str]:
    """Names bound to modules by `import x` / `import x as y` / `from agentic_gp import llm` (+ os, sys, llm)."""
    names = {"os", "sys", "llm"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module in ("agentic_gp", "os"):
            names.update(a.asname or a.name for a in node.names)
    return names


def _check_dynamic_call(node: ast.Call, modules: set[str]) -> list[Issue]:
    """getattr(os, "spawn" + "v"), getattr(llm, name), vars(os), ... - ways around the name checks."""
    fn = node.func.id
    first = node.args[0] if node.args else None
    on_module = isinstance(first, ast.Name) and first.id in modules
    if fn in ("getattr", "setattr", "delattr") and len(node.args) >= 2 and not on_module:
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):    # getattr(x, "__getattribute__")
            fake = ast.Attribute(value=first if first is not None else ast.Name(id="_"), attr=attr.value)
            fake.lineno = node.lineno
            return _check_attribute(fake, modules)
        return []
    if fn in ("getattr", "setattr", "delattr") and on_module and len(node.args) >= 2:
        attr = node.args[1]
        if not (isinstance(attr, ast.Constant) and isinstance(attr.value, str)):
            return [Issue("error", f"line {node.lineno}: {fn}({first.id}, <computed name>) - a driver may only "
                                   f"use {first.id}'s attributes by name")]
        fake = ast.Attribute(value=ast.Name(id=first.id), attr=attr.value)
        fake.lineno = node.lineno
        return _check_attribute(fake, modules)
    if fn in MODULE_INTROSPECTION and on_module:
        return [Issue("error", f"line {node.lineno}: {fn}({first.id}) - module introspection is not allowed "
                               "in a driver")]
    return []


def _check_attribute(node: ast.Attribute, modules: set[str] = frozenset()) -> list[Issue]:
    """`os.system(...)`-style process starters, private `llm._x` internals, llm_policy, sys.modules."""
    base = node.value.id if isinstance(node.value, ast.Name) else         node.value.attr if isinstance(node.value, ast.Attribute) else None
    if base in PROCESS_CALLS and node.attr in PROCESS_CALLS[base]:
        return [Issue("error", f"line {node.lineno}: {base}.{node.attr} starts processes - a driver may not "
                               "start processes (they would outlive the race)")]
    if node.attr in PROCESS_NAMES:
        return [Issue("error", f"line {node.lineno}: {node.attr} starts processes - not allowed in a driver")]
    if base == "llm" and node.attr.startswith("_"):
        return [Issue("error", f"line {node.lineno}: llm.{node.attr} is private - use chat/chat_json/"
                               "ask_with_tools (the runner sets the LLM rules)")]
    if base == "llm" and node.attr in LLM_RULE_NAMES:
        return [Issue("error", f"line {node.lineno}: llm.{node.attr} - the runner sets the LLM rules; a "
                               "driver may not open its own policy scopes")]
    if base == "sys" and node.attr == "modules" or node.attr == "__dict__" and base in modules:
        return [Issue("error", f"line {node.lineno}: {base}.{node.attr} - module internals are not for drivers")]
    if node.attr in INTERNAL_ATTRS:
        return [Issue("error", f"line {node.lineno}: .{node.attr} - interpreter internals are not for drivers")]
    if node.attr in FILE_METHODS or node.attr in NUMPY_FILE_FUNCS and base in ("np", "numpy"):
        return [Issue("error", f"line {node.lineno}: .{node.attr} - file I/O is not allowed in a driver (only "
                               "what is in your file persists)")]
    return []


_COMPUTED = "<expression>"


def _literal_metadata(tree: ast.Module, driver_class: ast.ClassDef | None) -> dict:
    """Constant NAME/AUTHOR/KIND/ANATOMY assignments at module level or in class Driver (class wins)."""
    out: dict = {}
    for body in (tree.body, driver_class.body if driver_class else []):
        for node in body:
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else []
            for t in targets:
                if isinstance(t, ast.Name) and t.id in ("NAME", "AUTHOR", "KIND", "ANATOMY", "COLOR"):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        out[t.id] = _COMPUTED
    return out


def static_metadata(path: str | Path) -> dict:
    """Literal NAME / AUTHOR / KIND / ANATOMY / COLOR of a driver file, read without running it
    ({} if the file cannot be parsed; a computed value is the string "<expression>")."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return {}
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    return _literal_metadata(tree, classes.get("Driver"))


def _has_symbols(text: str) -> bool:
    """Emoji and other pictographs (outside the Basic Multilingual Plane, or Unicode category So)."""
    import unicodedata
    return any(ord(ch) > 0xFFFF or unicodedata.category(ch) == "So" for ch in text)


def _dedupe(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for i in issues:
        if (i.level, i.message) not in seen:
            seen.add((i.level, i.message))
            out.append(i)
    return out


# ============================================================================ in-process driver
@dataclass
class LoadedDriver:
    """A driver ready to race, whatever style the file used.

    plan/drive/reflect call the driver's code inside LLM policy scopes and let exceptions through:
    the SESSION catches and records them (a crash in drive() = coast for one tick).
    """

    name: str
    author: str = "anonymous"
    color: str = "#8d99ae"
    kind: str = "code"
    model: str | None = None
    source_path: str | None = None
    motto: str = ""
    anatomy: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    house: bool = False
    uses_llm: bool = False
    _plan: Callable | None = field(default=None, repr=False)
    _drive: Callable | None = field(default=None, repr=False)
    _reflect: Callable | None = field(default=None, repr=False)
    phase: str = "session"                     # set by the runner before each attempt
    isolated: bool = False
    log_path: str | None = None
    unknown_params: list[str] = field(default_factory=list)   # --param keys the driver's PARAMS lacks

    def __post_init__(self) -> None:
        self._track_scope: llm.LLMPolicy | None = None
        self._scopes: dict[str, llm.LLMPolicy] = {}  # reusable phase / drive scopes (no budgets)
        self._official = False
        self._between: tuple = ()                  # official: the rules for calls outside plan/reflect
        self._session_token: int | None = None     # official: this session's entry in llm's registry

    # -- the runner's hooks
    def begin_session(self, mode: str = "open", label: str | None = None) -> None:
        """New LLM budget scope for one session on one track (official: <= 8 calls).

        Official sessions are also registered with the llm module: calls from contexts the runner
        did not set up (threads the driver starts, a fresh contextvars.Context, a scope the driver
        opens itself) are judged by this session's rules first - refused outside plan()/reflect(),
        counted against the running plan()/reflect() budget inside them.
        """
        self.end_session()
        self._track_scope = llm.llm_policy(phase="track", mode=mode, label=label or self.name)
        self._scopes = {}
        self._official = mode == "official"
        if self._official:
            self._between = (self._track_scope, llm.llm_policy(phase="between"))
            self._session_token = llm._open_session(self._track_scope, self._between)

    def end_session(self) -> None:
        """Unregister begin_session's official rules (safe to call twice; other sessions keep theirs)."""
        if self._session_token is not None:
            llm._close_session(self._session_token)
            self._session_token = None
        self._official = False

    def llm_usage(self) -> dict:
        """Calls/tokens/seconds of LLM use since begin_session (+ mode and calls left)."""
        scope = self._track_scope
        if scope is None:
            return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0,
                    "cache_hits": 0, "mode": "open", "calls_left": None}
        mode = scope.mode or "open"
        limit = scope.limit(mode)
        u = scope.usage
        return {"calls": int(u["calls"]), "prompt_tokens": int(u["prompt_tokens"]),
                "completion_tokens": int(u["completion_tokens"]), "seconds": round(float(u["seconds"]), 3),
                "errors": int(u["errors"]), "cache_hits": int(u["cache_hits"]), "mode": mode,
                "calls_left": None if limit is None else max(0, limit - int(u["calls"]))}

    def _scope(self, phase: str) -> llm.LLMPolicy:
        if phase not in self._scopes:
            self._scopes[phase] = llm.llm_policy(phase=phase)
        return self._scopes[phase]

    def _scoped(self, method: str, fn: Callable, arg: Any, timeout: float | None = None) -> Any:
        """Run driver code inside track > phase > method policy scopes. plan/reflect get a FRESH
        scope per call (their budgets are per call) with a deadline a little before `timeout`, so a
        slow LLM raises LLMError (-> the driver's fallback) instead of the runner killing the driver;
        drive reuses one scope (it is called every tick)."""
        if self._track_scope is None:
            self.begin_session()
        if method == "drive":
            inner = self._scope("drive")
        else:
            deadline = None if timeout is None else time.monotonic() + max(0.0, timeout - deadline_margin(timeout))
            inner = llm.llm_policy(phase=method, deadline=deadline)
        token = self._session_token if self._official else None
        if token is not None:                      # threads started in here follow the same rules
            llm._set_session(token, (self._track_scope, self._scope(self.phase), inner))
        try:
            with self._track_scope, self._scope(self.phase), inner:
                return fn(arg)
        finally:
            if token is not None:
                llm._set_session(token, self._between)

    # -- the driver's code
    def plan(self, track: dict, timeout: float | None = None) -> None:
        if self._plan is not None:
            self._scoped("plan", self._plan, track, timeout)

    def drive(self, obs: dict, timeout: float | None = None) -> Action:
        if self._drive is None:
            return Action()
        raw = self._scoped("drive", self._drive, obs)
        problem = action_problem(raw)
        if problem:
            raise BadAction(problem)
        return normalize_action(raw)

    def reflect(self, report: dict, timeout: float | None = None) -> None:
        if self._reflect is not None:
            self._scoped("reflect", self._reflect, report, timeout)

    def close(self) -> None:
        """Nothing to stop in-process (isolated drivers stop their child here); undoes begin_session."""
        self.end_session()

    def meta(self) -> dict:
        return {"name": self.name, "author": self.author, "color": self.color, "kind": self.kind,
                "model": self.model, "source": self.source_path, "motto": self.motto,
                "anatomy": list(self.anatomy), "params": json_safe(self.params), "house": self.house,
                "uses_llm": self.uses_llm}


class BadAction(ValueError):
    """drive() returned something that is not an action (the car coasts that tick)."""


def deadline_margin(timeout: float) -> float:
    """How long before a plan()/reflect() time limit the LLM deadline falls (answer + fallback time)."""
    return min(5.0, 0.1 * max(0.0, timeout))


def json_safe(value: Any) -> Any:
    """A copy that json.dumps(allow_nan=False) accepts: non-finite floats and odd objects become text."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return repr(value)


_MODULE_COUNTER = itertools.count(1)


def load_driver(source: Any, *, isolate: bool = False, params: dict | None = None, check: bool = True,
                house: bool = False, log_path: str | Path | None = None, llm_semaphore: Any = None,
                env: dict | None = None, load_timeout: float = 300.0, llm_mode: str | None = None,
                cancel: threading.Event | None = None, on_stage: Callable[[str], None] | None = None):
    """Load a driver from a .py path, a module, a Driver instance/class or a plain drive(obs) callable.

    isolate=True (paths only) runs the file in a child process and returns an IsolatedDriver with the
    same interface. `params` overrides the driver's PARAMS before it is instantiated (strings from the
    CLI are converted to the type of the default; keys the driver does not have end up in
    `drv.unknown_params`). `check` runs check_driver_source first. llm_mode="official" refuses LLM
    calls while the file loads; `cancel` (isolated) stops waiting for the child when it is set;
    `on_stage(stage)` (isolated) is told when a slow loading stage starts ("warm-up": the LLM SDK import).
    Raises DriverLoadError with a friendly message if the driver cannot race (ValueError for a
    --param value of the wrong type).
    """
    if isinstance(source, LoadedDriver) or isinstance(source, IsolatedDriver):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix != ".py":
            raise DriverLoadError(f"{path}: a driver is a .py file")
        if not path.is_file():
            raise DriverLoadError(f"driver file not found: {path}")
        if isolate:
            return IsolatedDriver(path, params=params, check=check, house=house, log_path=log_path,
                                  llm_semaphore=llm_semaphore, env=env, load_timeout=load_timeout,
                                  llm_mode=llm_mode, cancel=cancel, on_stage=on_stage)
        if check:
            errors = [i for i in check_driver_source(path) if i.level == "error"]
            if errors:
                raise DriverLoadError(f"{path.name}: " + "; ".join(i.message for i in errors))
        with _load_scope(llm_mode):
            module = _exec_file(path)
            drv = _from_module(module, params, default_name=path.stem)
        drv.source_path = str(path)
        drv.uses_llm = _mentions_llm(path.read_text(encoding="utf-8", errors="replace"))
    elif isolate:
        raise DriverLoadError("isolate=True needs a driver file path")
    elif isinstance(source, types.ModuleType):
        drv = _from_module(source, params, default_name=source.__name__.rsplit(".", 1)[-1])
        drv.source_path = getattr(source, "__file__", None)
        # a module that imported agentic_gp.llm: the runner warms the LLM SDK up before plan() is timed
        drv.uses_llm = any(v is llm for v in vars(source).values())
    elif isinstance(source, type):
        drv = _from_object(_instantiate(source, params), None, None)
    elif hasattr(source, "drive") and callable(source.drive):
        drv = _from_object(source, None, params)
    elif callable(source):
        drv = LoadedDriver(name=getattr(source, "__name__", "driver"), _drive=source)
    else:
        raise DriverLoadError(f"cannot load a driver from {type(source).__name__}")
    drv.house = house or drv.house
    drv.color = color_for(drv.name, drv.color)
    return drv


@contextmanager
def _load_scope(llm_mode: str | None):
    """Official events: no LLM calls while a driver file is imported / instantiated (threads it
    starts included: the load scope is registered as an official session while the file loads)."""
    if llm_mode == "official":
        scope = llm.llm_policy(phase="load", mode="official")
        token = llm._open_session(scope, (scope,))
        try:
            with scope:
                yield
        finally:
            llm._close_session(token)
    else:
        yield


def _exec_file(path: Path) -> types.ModuleType:
    """Execute a file as a brand-new module (unique name, registered so pickling/dataclasses work)."""
    stem = re.sub(r"\W", "_", path.stem)
    name = f"agp_driver_{stem}_{next(_MODULE_COUNTER)}_{uuid.uuid4().hex[:6]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DriverLoadError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        sys.modules.pop(name, None)
        raise DriverLoadError(f"{path.name}: syntax error on line {exc.lineno}: {exc.msg}") from exc
    except (Exception, SystemExit) as exc:          # sys.exit() at import time must not end the race
        sys.modules.pop(name, None)
        raise DriverLoadError(f"{path.name}: error while loading the file: {type(exc).__name__}: {exc}") from exc
    return module


def _from_module(module: types.ModuleType, params: dict | None, default_name: str) -> LoadedDriver:
    cls = getattr(module, "Driver", None)
    own_class = isinstance(cls, type) and cls.__module__ == module.__name__
    # class style if the file defines `class Driver` (an imported base class alone does not count,
    # unless there is no drive() function at all)
    if own_class or (isinstance(cls, type) and not callable(getattr(module, "drive", None))):
        defaults = getattr(cls, "PARAMS", None) or getattr(module, "PARAMS", None)
        merged = _merge_params(defaults, params)
        if merged is not None:
            cls.PARAMS = merged
            if isinstance(getattr(module, "PARAMS", None), dict):
                module.PARAMS = merged
        try:
            instance = cls()
        except (Exception, SystemExit) as exc:
            raise DriverLoadError(f"Driver() failed: {type(exc).__name__}: {exc}") from exc
        drv = _from_object(instance, module, None, default_name)
    else:
        if not callable(getattr(module, "drive", None)):
            raise DriverLoadError(f"{default_name}: no drive(obs) function and no Driver class found")
        defaults = getattr(module, "PARAMS", None)
        merged = _merge_params(defaults, params)
        if merged is not None:
            module.PARAMS = merged
        drv = _from_object(module, module, None, default_name)
    drv.unknown_params = unknown_param_keys(defaults, params)
    return drv


def unknown_param_keys(defaults: Any, overrides: dict | None) -> list[str]:
    """Override keys that are not in the driver's PARAMS (usually a typo: --param agression=1.5)."""
    known = defaults if isinstance(defaults, dict) else {}
    return sorted(k for k in (overrides or {}) if k not in known)


def _instantiate(cls: type, params: dict | None) -> Any:
    merged = _merge_params(getattr(cls, "PARAMS", None), params)
    if merged is not None:
        cls.PARAMS = merged
    return cls()


def _from_object(obj: Any, module: types.ModuleType | None, params: dict | None,
                 default_name: str | None = None) -> LoadedDriver:
    def meta(key: str, default: Any = None) -> Any:
        val = getattr(obj, key, None)
        if val is None and module is not None:
            val = getattr(module, key, None)
        return default if val is None else val

    unknown = []
    if params:                                   # an instance was passed: override on the instance
        unknown = unknown_param_keys(meta("PARAMS"), params)
        merged = _merge_params(meta("PARAMS"), params)
        try:
            obj.PARAMS = merged
        except AttributeError:
            pass
    fallback = default_name or type(obj).__name__
    # one line, at most MAX_NAME_CHARS (a computed NAME can be anything; it names log files too)
    name = " ".join(str(meta("NAME", fallback)).split())[:MAX_NAME_CHARS].strip() or fallback[:MAX_NAME_CHARS]
    kind = str(meta("KIND", "code"))
    anatomy = meta("ANATOMY", [])
    p = meta("PARAMS", {})
    return LoadedDriver(
        name=name, author=str(meta("AUTHOR", "anonymous")), color=meta("COLOR"),
        kind=kind if kind in KINDS else "other", model=None if meta("MODEL") is None else str(meta("MODEL")),
        motto=str(meta("MOTTO", "")), anatomy=[str(a) for a in anatomy] if isinstance(anatomy, (list, tuple)) else [],
        params=dict(p) if isinstance(p, dict) else {},
        _plan=_method(obj, "plan"), _drive=_method(obj, "drive"), _reflect=_method(obj, "reflect"),
        unknown_params=unknown,
    )


def _method(obj: Any, name: str) -> Callable | None:
    fn = getattr(obj, name, None)
    return fn if callable(fn) else None


def _merge_params(defaults: Any, overrides: dict | None) -> dict | None:
    if overrides is None:
        return dict(defaults) if isinstance(defaults, dict) else None
    base = dict(defaults) if isinstance(defaults, dict) else {}
    return {**base, **coerce_params(base, overrides)}


def coerce_params(defaults: dict, overrides: dict) -> dict:
    """Convert override values (often CLI strings) to the type of the default value.

        coerce_params({"aggression": 1.0, "laps": 2, "line": True}, {"aggression": "1.2", "line": "no"})
        -> {"aggression": 1.2, "line": False}
    """
    out = {}
    for key, value in overrides.items():
        default = defaults.get(key)
        try:
            out[key] = _coerce(value, default)
        except ValueError:
            kind = "a whole number" if isinstance(default, int) else "a number"
            raise ValueError(f"--param {key} expects {kind} (default {default!r}), got {value!r}") from None
    return out


def _coerce(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if isinstance(default, bool):
        return text.lower() in ("1", "true", "yes", "on", "y")
    if isinstance(default, int):
        try:
            return int(text)
        except ValueError:
            return float(text)
    if isinstance(default, float):
        return float(text)
    if isinstance(default, str):
        return value
    try:
        return json.loads(text)                  # unknown key: numbers, lists, true/false, null
    except ValueError:
        return value


def parse_param_args(items: list[str] | None) -> dict:
    """CLI helper: ["aggression=1.1", "line=racing"] -> {"aggression": "1.1", "line": "racing"}."""
    out = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--param expects key=value, got {item!r}")
        out[key.strip()] = value
    return out


def _mentions_llm(source: str) -> bool:
    return bool(re.search(r"agentic_gp\.llm|from\s+agentic_gp\s+import[^\n]*\bllm\b", source))


def _is_hex(color: Any) -> bool:
    return isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) is not None


def color_for(name: str, color: Any = None) -> str:
    """A '#rrggbb' colour: the driver's own (also '#rgb'), else a stable palette pick by name."""
    if isinstance(color, str):
        c = color.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{3}", c):
            return "#" + "".join(ch * 2 for ch in c[1:]).lower()
        if _is_hex(c):
            return c.lower()
    return PALETTE[zlib.crc32(str(name).encode("utf-8")) % len(PALETTE)]


# ============================================================================ isolated driver
_LIVE: "set[IsolatedDriver]" = set()
_LIVE_LOCK = threading.Lock()
_SPAWN_LOCK = threading.Lock()


@contextmanager
def _main_module_hidden():
    """Hide the parent's __main__ while a child starts.

    A `spawn`ed child normally re-runs the parent's main script first (to find pickled functions).
    Our child only needs agentic_gp, and re-running a Streamlit page, a notebook, `python -` or an
    unguarded script in every driver process would be slow at best and recursive at worst.
    """
    main = sys.modules.get("__main__")
    sys.modules["__main__"] = types.ModuleType("__main__")
    try:
        yield
    finally:
        if main is not None:
            sys.modules["__main__"] = main


class IsolatedDriver:
    """Proxy for a driver file running in its own child process (same interface as LoadedDriver).

    Each call sends (seq, method, argument, phase, timeout) and waits at most `timeout` seconds for
    the answer with the same seq. Exceptions in the driver come back as DriverError; no answer in
    time raises DriverTimeout (the late answer is discarded when it arrives); a dead child raises
    DriverDied, and so does a set `cancel` event (the runner is stopping).

    Clean-up guarantees: the child and every process it starts are killed together (a Windows Job
    Object / a POSIX process group), the child exits by itself when this process dies, LLM slots of
    the shared semaphore that a killed child still held are given back, and its log is size-capped.
    """

    isolated = True

    def __init__(self, path: str | Path, *, params: dict | None = None, check: bool = True, house: bool = False,
                 log_path: str | Path | None = None, llm_semaphore: Any = None, env: dict | None = None,
                 load_timeout: float = 300.0, llm_mode: str | None = None,
                 cancel: threading.Event | None = None, on_stage: Callable[[str], None] | None = None,
                 guard: bool = True) -> None:
        path = Path(path).resolve()
        if check:
            errors = [i for i in check_driver_source(path) if i.level == "error"]
            if errors:
                raise DriverLoadError(f"{path.name}: " + "; ".join(i.message for i in errors))
        if log_path is None:
            log_dir = Path(tempfile.gettempdir()) / "agentic_gp_driver_logs"
            log_path = log_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', path.stem)}_{uuid.uuid4().hex[:8]}.log"
        self.log_path = str(log_path)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.source_path = str(path)
        self.phase = "session"
        self.house = house
        self.unknown_params: list[str] = []
        self.dead: str | None = None             # reason, once the child is gone
        self._seq = 0
        self._outstanding: set[int] = set()      # calls that timed out and have not answered yet
        self._usage: dict = LoadedDriver(name="?").llm_usage()
        self._lock = threading.Lock()
        self._cancel = cancel
        self._on_stage = on_stage
        self._load_timeout = load_timeout
        self._warm_deadline: float | None = None  # set while the LLM SDK import outlasts its first window
        self._sem = llm_semaphore
        ctx = mp.get_context("spawn")
        self._held = ctx.Value("i", 0) if llm_semaphore is not None else None   # LLM slots the child holds
        self._conn, child_conn = ctx.Pipe(duplex=True)
        # The child never sees the API key: it gets a token for this process's local LLM relay instead.
        from . import llm_proxy                  # parent only: driver processes do not pay for http.server

        self._llm_env = llm_proxy.child_llm_env(env)
        self._proc = ctx.Process(
            target=_child_main, name=f"agp-driver-{path.stem}", daemon=True,
            args=(child_conn, str(path), params, self.log_path, self._llm_env, llm_semaphore, self._held,
                  llm_mode, guard),
        )
        with _SPAWN_LOCK, _main_module_hidden(), llm.api_keys_hidden():
            self._proc.start()
        self._job = _contain(self._proc)         # Windows: the child and its children live in one job
        child_conn.close()
        with _LIVE_LOCK:
            _LIVE.add(self)
        try:
            meta = self._request(0, None, load_timeout)
        except (DriverTimeout, DriverDied, DriverError) as exc:
            self.close()
            raise DriverLoadError(f"{path.name}: {exc}") from exc
        for key in ("name", "author", "color", "kind", "model", "motto", "anatomy", "params", "uses_llm"):
            setattr(self, key, meta[key])
        self.unknown_params = list(meta.get("unknown_params") or [])
        self.house = house or bool(meta.get("house"))
        if self.uses_llm:
            # The child imports the LLM SDK AFTER saying hello, so a slow cold import (minutes on a
            # busy machine) does not count against load_timeout; it gets its own window of the same
            # size here, and one more before the first call (_await_warm_up).
            self._stage("warm-up")
            try:
                self._request(0, "llm warm-up", load_timeout)
            except DriverTimeout:
                self._warm_deadline = time.monotonic() + load_timeout
            except DriverDied as exc:
                self.close()
                raise DriverLoadError(f"{path.name}: {exc}") from exc

    # -- the LoadedDriver interface
    def begin_session(self, mode: str = "open", label: str | None = None) -> None:
        self._call("begin_session", (mode, label), 30.0)

    def end_session(self) -> None:
        """Nothing to undo in this process (the child's LLM rules live and die with the child)."""

    def llm_usage(self) -> dict:
        return dict(self._usage)

    def plan(self, track: dict, timeout: float | None = 120.0) -> None:
        self._call("plan", track, timeout)

    def drive(self, obs: dict, timeout: float | None = 2.0) -> Action:
        return self._call("drive", obs, timeout)

    def reflect(self, report: dict, timeout: float | None = 120.0) -> None:
        self._call("reflect", report, timeout)

    def meta(self) -> dict:
        return {"name": self.name, "author": self.author, "color": self.color, "kind": self.kind,
                "model": self.model, "source": self.source_path, "motto": self.motto,
                "anatomy": list(self.anatomy), "params": dict(self.params), "house": self.house,
                "uses_llm": self.uses_llm}

    @property
    def pid(self) -> int | None:
        return self._proc.pid

    def is_alive(self) -> bool:
        return self._proc.is_alive()

    # -- plumbing
    def _stage(self, stage: str) -> None:
        if self._on_stage is not None:
            try:
                self._on_stage(stage)
            except Exception:                    # noqa: BLE001 - a UI callback must not stop loading
                pass

    def _await_warm_up(self) -> None:
        """The child still imports the LLM SDK: wait for it (its second window) before the first call,
        so that call's own time limit is not spent waiting in the pipe."""
        if self._warm_deadline is None:
            return
        self._stage("warm-up")
        try:
            self._request(0, "llm warm-up", max(0.0, self._warm_deadline - time.monotonic()))
        except DriverTimeout:
            self._mark_dead(f"still importing the LLM SDK after {2 * self._load_timeout:g}s (2 x load_timeout)")
            self.kill()
            raise DriverLoadTimeout(self.dead) from None
        finally:
            self._warm_deadline = None

    def _call(self, method: str, arg: Any, timeout: float | None) -> Any:
        with self._lock:
            if self.dead:
                raise DriverDied(self.dead)
            self._await_warm_up()
            self._seq += 1
            seq = self._seq
            try:
                self._conn.send(("call", seq, method, arg, self.phase, timeout))
            except (OSError, EOFError, BrokenPipeError) as exc:
                self._mark_dead(f"driver process is gone ({type(exc).__name__})")
                raise DriverDied(self.dead) from exc
            return self._request(seq, method, timeout)

    def _request(self, seq: int, method: str | None, timeout: float | None) -> Any:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self._cancel is not None and self._cancel.is_set():
                self._outstanding.add(seq)
                raise DriverDied("stopped by the runner")
            wait = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
            if wait <= 0:
                self._outstanding.add(seq)
                raise DriverTimeout(f"{method or 'load'}() did not answer within {timeout:g}s")
            try:
                ready = self._conn.poll(wait)
                msg = self._conn.recv() if ready else None
            except (EOFError, OSError, BrokenPipeError):
                self._mark_dead(self._death_reason())
                raise DriverDied(self.dead)
            if msg is None:
                if not self._proc.is_alive() and not self._conn.poll(0):
                    self._mark_dead(self._death_reason())
                    raise DriverDied(self.dead)
                continue
            kind, reply_seq, ok, payload, usage = msg
            self._outstanding.discard(reply_seq)
            if reply_seq != seq:
                continue                         # a late answer to a call that already timed out
            if usage:
                self._usage = usage
            if ok:
                return payload
            raise DriverError(payload.get("error", "driver error"), payload.get("traceback", ""))

    def wait_idle(self, timeout: float) -> bool:
        """Wait until every timed-out call has answered (late answers are dropped). False = still busy."""
        with self._lock:
            deadline = time.monotonic() + timeout
            while self._outstanding and not self.dead:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    return False
                try:
                    if self._conn.poll(min(wait, 0.25)):
                        self._outstanding.discard(self._conn.recv()[1])
                    elif not self._proc.is_alive():
                        self._mark_dead(self._death_reason())
                except (EOFError, OSError, BrokenPipeError):
                    self._mark_dead(self._death_reason())
            return not self.dead

    def _death_reason(self) -> str:
        self._proc.join(0.5)
        code = self._proc.exitcode
        tail = _log_tail(self.log_path)
        return f"driver process exited (exit code {code})" + (f": {tail}" if tail else "")

    def _mark_dead(self, reason: str) -> None:
        if not self.dead:
            self.dead = reason

    def kill(self) -> None:
        """Stop the child (and every process it started) right now (no goodbye)."""
        self._mark_dead(self.dead or "killed by the runner")
        _kill_tree(self._proc, self._job)
        _stop_process(self._proc, graceful=False)
        self._after_exit()

    def close(self) -> None:
        """Ask the child to exit, then make sure it is gone (terminate / kill). Safe to call twice."""
        if self._proc.is_alive() and not self.dead:
            try:
                self._conn.send(("exit", 0, None, None, None, None))
            except (OSError, EOFError, BrokenPipeError):
                pass
        _stop_process(self._proc, graceful=not self.dead)
        _kill_tree(self._proc, self._job)        # processes the driver started do not outlive it
        self._mark_dead("closed")
        self._after_exit()

    def _after_exit(self) -> None:
        from . import llm_proxy

        llm_proxy.revoke(self._llm_env)
        self._give_back_llm_slots()
        _close_job(self._job)
        self._job = None
        try:
            self._conn.close()
        except OSError:
            pass
        with _LIVE_LOCK:
            _LIVE.discard(self)

    def _give_back_llm_slots(self) -> None:
        """A child killed in the middle of an LLM request never released its semaphore slot(s):
        release them here, or every other driver's LLM calls would wait forever."""
        if self._held is None or self._sem is None or self._proc.is_alive():
            return
        raw = self._held.get_obj()               # no lock: the dead child may have died holding it
        n, raw.value = int(raw.value), 0
        for _ in range(max(0, n)):
            try:
                self._sem.release()
            except (ValueError, OSError):        # already at its maximum
                break

    def __enter__(self) -> "IsolatedDriver":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IsolatedDriver({getattr(self, 'name', '?')!r}, pid={self.pid}, dead={self.dead!r})"


def _stop_process(proc: Any, graceful: bool) -> None:
    if graceful:
        proc.join(1.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
    if proc.is_alive():
        proc.kill()
        proc.join(2.0)


# ---------------------------------------------------------------------------- process trees
def _contain(proc: Any) -> Any:
    """Windows: put the child in a Job Object that kills every process in it when it is closed or
    terminated - also when THIS process dies, because the OS then closes the handle. None elsewhere
    (POSIX children make themselves process-group leaders instead, see _child_main)."""
    if sys.platform != "win32":
        return None
    try:
        return _win_job(proc)
    except Exception:                            # noqa: BLE001 - containment is best effort
        return None


def _win_job(proc: Any) -> Any:
    import ctypes
    from ctypes import wintypes

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

    class _Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _Basic), ("IoInfo", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _Extended()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # 9 = extended limits
    handle = k32.OpenProcess(0x0100 | 0x0001, False, proc.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if ok and handle:
        ok = k32.AssignProcessToJobObject(job, handle)
    if handle:
        k32.CloseHandle(handle)
    if not ok:
        k32.CloseHandle(job)
        return None
    return job


def _kill_tree(proc: Any, job: Any) -> None:
    """Kill every process the child started (and the child itself, if it still runs)."""
    if job is not None:
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            k32.TerminateJobObject(job, 1)
        except Exception:                        # noqa: BLE001
            pass
    elif sys.platform != "win32" and proc.pid:
        import signal
        try:
            if os.getpgid(proc.pid) == proc.pid:     # the child leads its own group (see _child_main)
                os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:                                     # the child is gone: its group may still exist
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass


def _close_job(job: Any) -> None:
    if job is None:
        return
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        k32.CloseHandle(job)
    except Exception:                            # noqa: BLE001
        pass


# ---------------------------------------------------------------------------- logs
MAX_LOG_BYTES = 5_000_000          # a child's log is trimmed to its first and last LOG_KEEP_BYTES above this
LOG_KEEP_BYTES = 1_000_000


def _log_tail(path: str, n_chars: int = 300) -> str:
    """The last lines of a log (reads only its last 4 KB, whatever its size)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            text = f.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-3:])[-n_chars:] if lines else ""


class _CappedLog:
    """The child's stdout/stderr: a text file that never grows past ~MAX_LOG_BYTES. Above it, the file
    is cut to its first and last LOG_KEEP_BYTES with a note in between (the start and the latest
    output, e.g. a final traceback, survive a print() loop)."""

    CHECK_EVERY = 65536

    def __init__(self, path: str, limit: int = MAX_LOG_BYTES, keep: int = LOG_KEEP_BYTES) -> None:
        self.path, self.limit, self.keep = path, limit, keep
        self.fh = open(path, "a", encoding="utf-8", buffering=1, errors="replace")
        self.encoding, self.errors = "utf-8", "replace"
        self._since = 0
        self.trims = 0

    def write(self, text: str) -> int:
        n = self.fh.write(text)
        self._since += len(text)
        if self._since >= self.CHECK_EVERY:
            self._since = 0
            self._maybe_trim()
        return n

    def _maybe_trim(self) -> None:
        try:
            self.fh.flush()
            fd = self.fh.fileno()
            size = os.fstat(fd).st_size
            if size <= self.limit:
                return
            with open(self.path, "rb") as r:
                head = r.read(self.keep)
                r.seek(max(self.keep, size - self.keep))
                tail = r.read()
            self.trims += 1
            note = (f"\n[agentic_gp] ... log trimmed ({self.trims}x): it passed {self.limit // 1_000_000} MB - "
                    "print less in drive() ...\n").encode("utf-8")
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, head + note + tail)
        except OSError:
            pass

    def flush(self) -> None:
        self.fh.flush()

    def fileno(self) -> int:
        return self.fh.fileno()

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fh, name)


@atexit.register
def _close_all() -> None:
    with _LIVE_LOCK:
        live = list(_LIVE)
    for drv in live:
        try:
            drv.kill()
        except Exception:
            pass


def close_all_drivers() -> None:
    """Kill every isolated driver process this process started (UI stop buttons, test cleanup)."""
    _close_all()


# ============================================================================ the child process
def _redirect_output(log_path: str) -> _CappedLog:
    """Send print() output and C-level stdout/stderr of the child to the (size-capped) log file."""
    log = _CappedLog(log_path)
    for fd in (1, 2):
        try:
            os.dup2(log.fileno(), fd)
        except OSError:
            pass
    sys.stdout = sys.stderr = log
    return log


class _TrackedSemaphore:
    """The child's view of the grand prix's shared LLM semaphore. It counts the slots this process
    holds in shared memory, so the parent can give them back if it has to kill the child mid-request."""

    def __init__(self, sem: Any, held: Any) -> None:
        self._sem, self._held = sem, held

    def acquire(self, block: bool = True, timeout: float | None = None) -> bool:
        ok = self._sem.acquire(block, timeout) if timeout is not None else self._sem.acquire(block)
        if ok:
            with self._held.get_lock():
                self._held.value += 1
        return ok

    def release(self) -> None:
        with self._held.get_lock():
            self._held.value -= 1
        self._sem.release()


def _watch_parent() -> None:
    """Exit when the parent process dies (crash, taskkill, closed terminal), instead of running on."""
    parent = mp.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        if sys.platform != "win32":
            import signal
            try:
                os.killpg(0, signal.SIGKILL)     # this process and everything it started
            except OSError:
                pass
        os._exit(1)

    threading.Thread(target=watch, name="agp-parent-watch", daemon=True).start()


def _child_main(conn: Any, path: str, params: dict | None, log_path: str, env: dict, llm_semaphore: Any,
                held: Any = None, llm_mode: str | None = None, guard: bool = True) -> None:
    """Entry point of an isolated driver process (spawn target: must stay a module-level function).

    guard=False skips the race guard (tests of the process containment behind it only)."""
    if sys.platform != "win32":
        try:
            os.setpgrp()                         # lead a process group: the parent kills it as a whole
        except OSError:
            pass
    _watch_parent()
    _redirect_output(log_path)
    os.environ.update({k: str(v) for k, v in (env or {}).items()})
    llm.seal_api_keys()                          # the driver cannot read the key from os.environ
    print(f"[agentic_gp] driver process {os.getpid()} loading {path}", flush=True)
    try:
        if guard:
            install_guard(path, write_files=[log_path],
                          write_dirs=[os.path.abspath(os.getenv("LLM_CACHE_DIR", ".cache/llm"))])
        drv = load_driver(path, params=params, check=False, llm_mode=llm_mode)
        if llm_semaphore is not None:
            llm.set_llm_semaphore(_TrackedSemaphore(llm_semaphore, held) if held is not None else llm_semaphore)
        hello = {**drv.meta(), "unknown_params": list(drv.unknown_params)}
        conn.send(("hello", 0, True, hello, drv.llm_usage()))
    except BaseException as exc:                 # noqa: BLE001 - report every load problem to the parent
        _safe_send(conn, ("hello", 0, False, _error_payload(exc), None))
        return
    if drv.uses_llm:
        try:
            llm.warm_up()                        # pay the SDK import before anything is timed
        except Exception as exc:                 # noqa: BLE001 - the driver may still race without it
            print(f"[agentic_gp] llm warm-up failed: {type(exc).__name__}: {exc}", flush=True)
        if not _safe_send(conn, ("warm", 0, True, None, drv.llm_usage())):
            return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError, KeyboardInterrupt):
            break                                # the parent is gone: exit quietly
        kind, seq, method, arg, phase, *rest = msg
        if kind == "exit":
            break
        timeout = rest[0] if rest else None
        drv.phase = phase
        t0 = time.perf_counter()
        try:
            if method == "begin_session":
                result = drv.begin_session(*arg)
            elif method in ("plan", "reflect"):
                result = getattr(drv, method)(arg, timeout=timeout)
            elif method == "drive":
                result = drv.drive(arg)
            else:
                raise ValueError(f"unknown method {method!r}")
            reply = ("reply", seq, True, result, drv.llm_usage())
        except BaseException as exc:             # noqa: BLE001 - SystemExit from a driver is an error too
            reply = ("reply", seq, False, _error_payload(exc), drv.llm_usage())
        if not _safe_send(conn, reply):
            break
        if time.perf_counter() - t0 > 1.0:
            sys.stdout.flush()
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _error_payload(exc: BaseException) -> dict:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    print(tb, flush=True)
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": tb[-4000:]}


def _safe_send(conn: Any, msg: tuple) -> bool:
    try:
        conn.send(msg)
        return True
    except (OSError, EOFError, BrokenPipeError):
        return False
    except Exception as exc:                     # e.g. an unpicklable result
        try:
            conn.send(("reply", msg[1], False, {"error": f"cannot send the answer: {exc}", "traceback": ""},
                       None))
            return True
        except Exception:
            return False
