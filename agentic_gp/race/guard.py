"""Runtime guard for driver processes: an audit hook (PEP 578) that refuses what the rules forbid.

    from agentic_gp.race.guard import install_guard
    install_guard(driver_path, write_files=[log_path])   # once, in the child, before the driver file runs

The static check (loader.check_driver_source) reads the code; this hook watches what the process
DOES, so a trick the AST check misses (`os.__getattribute__("system")`, a path built at run time,
pickle) still cannot:

* start a process (subprocess, os.system / exec / spawn / fork / startfile, pty),
* call native code or read raw memory through ctypes,
* open a network connection, except to the runner's LLM relay (and a local Ollama when that is the
  provider),
* read files outside Python itself, the installed packages, the agentic_gp package, the fly data,
  the driver's own file and the LLM cassette - so no `.env`, no other students' files, no sealed
  mystery track,
* write, delete or rename files, except the driver's log and the LLM log folder.

A refused action raises PermissionError ("blocked by the race guard: ...") inside the driver, which
the runner reports like any driver error. Audit hooks cannot be removed once installed. They are
defence in depth, not an OS sandbox: run public uploads on a machine with nothing else to lose.
"""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "agentic_gp"
FLY_DATA_DIR = REPO_ROOT / "data" / "fly"

# ctypes.dlopen / dlsym stay allowed: the HTTP stack loads system libraries when it starts (truststore);
# calling native code (ctypes.call_function) and reading raw memory (ctypes.cdata) are what is refused.
PROCESS_EVENTS = ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.startfile",
                  "os.fork", "os.forkpty", "pty.spawn", "ctypes.call_function", "ctypes.cdata",
                  "winreg.CreateKey", "winreg.SetValue", "winreg.DeleteKey", "winreg.DeleteValue", "webbrowser.open")
FILE_CHANGE_EVENTS = ("os.remove", "os.rmdir", "os.rename", "os.truncate", "os.chmod", "os.chown", "os.link",
                      "os.symlink", "os.mkdir", "os.utime", "shutil.rmtree", "shutil.copyfile", "shutil.copymode",
                      "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.make_archive",
                      "shutil.unpack_archive")
PICKLE_DENIED = {"os", "nt", "posix", "subprocess", "sys", "importlib", "ctypes", "socket", "shutil", "io",
                 "pathlib", "runpy", "code", "pty", "multiprocessing", "_io", "_posixsubprocess", "_winapi",
                 "webbrowser", "urllib", "http"}
PICKLE_DENIED_BUILTINS = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "delattr",
                          "globals", "locals", "vars", "breakpoint", "input"}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

_INSTALLED = False


def _norm(p: str | os.PathLike | bytes) -> str:
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    return os.path.normcase(os.path.realpath(os.fspath(p)))


def _inside(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def _read_roots(driver_path: str | None, extra: list | None = None) -> list[str]:
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, str(PACKAGE_DIR), str(FLY_DATA_DIR)}
    for name in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            roots.add(sysconfig.get_paths()[name])
        except KeyError:
            pass
    for entry in sys.path:                        # installed packages; the repo root itself is NOT a read root
        if entry and Path(entry).is_dir() and _norm(entry) != _norm(REPO_ROOT):
            roots.add(entry)
    for var in ("AGENTIC_GP_FLY_CIRCUIT", "LLM_CASSETTE"):
        if os.getenv(var):
            roots.add(os.environ[var])
    if driver_path:
        roots.add(driver_path)
    roots.update(str(p) for p in extra or [] if p)
    return sorted({_norm(r) for r in roots if r})


def _allowed_hosts() -> set[tuple[str, int]]:
    """(host, port) pairs a driver process may connect to: the LLM relay, and Ollama when that is the provider."""
    urls = [os.getenv("AGP_LLM_PROXY_URL")]
    if os.getenv("LLM_PROVIDER", "").strip().lower() == "ollama":
        urls.append(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1")
    hosts = set()
    for url in filter(None, urls):
        u = urlparse(url)
        if u.hostname:
            port = u.port or (443 if u.scheme == "https" else 80)
            names = ("localhost", "127.0.0.1", "::1") if u.hostname == "localhost" else (u.hostname,)
            hosts.update((name, port) for name in names)
    return hosts


def install_guard(driver_path: str | os.PathLike | None = None, *, write_files: list | None = None,
                  write_dirs: list | None = None, read_paths: list | None = None) -> None:
    """Install the audit hook in this process (once; later calls do nothing).

    `write_files` / `write_dirs`: what the process may write besides the LLM logs; `read_paths`: extra
    files or folders it may read (e.g. a `file:` track).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    read_roots = _read_roots(str(driver_path) if driver_path else None, read_paths)
    w_dirs = [_norm(d) for d in (write_dirs or []) if d]
    for var in ("LLM_LOG_DIR",):
        if os.getenv(var):
            w_dirs.append(_norm(os.environ[var]))
    w_files = {_norm(f) for f in (write_files or []) if f}
    if os.getenv("LLM_RECORD"):
        w_files.add(_norm(os.environ["LLM_RECORD"]))
    hosts = _allowed_hosts()

    def may_write(path: str) -> bool:
        return path in w_files or _inside(path, w_dirs)

    def may_read(path: str) -> bool:
        return _inside(path, read_roots) or may_write(path)

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            target = args[0] if args else None
            if target is None or isinstance(target, int):
                return
            mode, flags = (args[1] if len(args) > 1 else None), (args[2] if len(args) > 2 else 0)
            writing = bool(mode and any(c in str(mode) for c in "wax+")) or bool((flags or 0) & _WRITE_FLAGS)
            try:
                path = _norm(target)
            except (TypeError, ValueError, OSError):
                raise PermissionError(f"blocked by the race guard: open({target!r})") from None
            if writing and not may_write(path):
                raise PermissionError(f"blocked by the race guard: a driver may not write files ({target})")
            if not writing and not may_read(path):
                raise PermissionError(f"blocked by the race guard: a driver may not read {target}")
        elif event.startswith(PROCESS_EVENTS):
            raise PermissionError(f"blocked by the race guard: {event} (no processes or native code in a driver)")
        elif event in FILE_CHANGE_EVENTS:
            paths = [a for a in args if isinstance(a, (str, bytes, os.PathLike))]
            if not paths or not all(may_write(_norm(p)) for p in paths):
                raise PermissionError(f"blocked by the race guard: {event} (a driver may not change files)")
        elif event == "socket.connect":
            addr = args[1] if len(args) > 1 else None
            if not (isinstance(addr, tuple) and len(addr) >= 2 and (str(addr[0]), int(addr[1])) in hosts):
                raise PermissionError(f"blocked by the race guard: network connection to {addr!r} (LLM calls go "
                                      "through agentic_gp.llm)")
        elif event == "pickle.find_class":
            module, name = (str(args[0]) if args else ""), (str(args[1]) if len(args) > 1 else "")
            if module.split(".")[0] in PICKLE_DENIED or module == "builtins" and name in PICKLE_DENIED_BUILTINS:
                raise PermissionError(f"blocked by the race guard: unpickling {module}.{name}")

    sys.addaudithook(hook)
    _INSTALLED = True
