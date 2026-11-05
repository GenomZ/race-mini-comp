"""Bridge to the Unity 3D race viewer (unity/RaceViewer) - standard library only.

The viewer shows any replay JSON (schema v1). It loads, in this order: ``--replay <file>`` on its
command line, ``StreamingAssets/replays/latest.json``, the newest ``*.json`` in that folder, and
finally the bundled ``demo.json``. It re-reads the file whenever it changes on disk, so the class
workflow is: run the grand prix -> ``install_replay(path)`` -> the open viewer reloads by itself.

    from agentic_gp.viz.unity import find_viewer, install_replay, launch_viewer, show_in_unity
    install_replay("races/final.json")        # copy to .../StreamingAssets/replays/latest.json (+ build)
    launch_viewer("races/final.json")         # start RaceViewer.exe --replay races/final.json
    show_in_unity("races/final.json")         # both, with friendly messages (what `race.py --unity` does)

    python race.py watch races/final.json --unity
    python unity/tools/viewer_bridge.py races/final.json [--install-only | --selftest]

Build the viewer once with: powershell -ExecutionPolicy Bypass -File unity/build_viewer.ps1
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_REL = Path("unity") / "RaceViewer"
REPLAYS_REL = Path("Assets") / "StreamingAssets" / "replays"
BUILD_CANDIDATES = (
    Path("Build") / "Windows" / "RaceViewer.exe",
    Path("Build") / "Mac" / "RaceViewer.app" / "Contents" / "MacOS" / "Agentic Grand Prix",
    Path("Build") / "Linux" / "RaceViewer.x86_64",
)
BUILD_HINT = "powershell -ExecutionPolicy Bypass -File unity/build_viewer.ps1"


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """Walk up from `start` (default: this file, then the working directory) to the folder containing unity/RaceViewer."""
    starts = [Path(start)] if start else [Path(__file__).resolve().parent, Path.cwd()]
    for s in starts:
        for d in [s.resolve(), *s.resolve().parents]:
            if (d / PROJECT_REL / "Assets").is_dir():
                return d
    return None


def project_dir(repo_root: str | Path | None = None) -> Path | None:
    """unity/RaceViewer (the Unity project), or None if it is not in this checkout."""
    root = Path(repo_root) if repo_root else find_repo_root()
    if root is None:
        return None
    p = root / PROJECT_REL
    return p if p.is_dir() else None


def find_viewer(repo_root: str | Path | None = None) -> Path | None:
    """Path of the built viewer (unity/RaceViewer/Build/Windows/RaceViewer.exe), or None if not built yet."""
    proj = project_dir(repo_root)
    if proj is None:
        return None
    for rel in BUILD_CANDIDATES:
        exe = proj / rel
        if exe.exists():
            return exe
    return None


def install_replay(replay_path: str | Path, repo_root: str | Path | None = None) -> list[Path]:
    """Copy a replay to StreamingAssets/replays/latest.json in the project and, if built, in the build.

    A running viewer notices the new file within ~2 s and reloads it. Returns the written paths.
    """
    src = Path(replay_path)
    if not src.is_file():
        raise FileNotFoundError(f"replay not found: {src}")
    proj = project_dir(repo_root)
    if proj is None:
        raise FileNotFoundError("unity/RaceViewer not found - pass repo_root=")
    targets = [proj / REPLAYS_REL / "latest.json"]
    exe = find_viewer(repo_root)
    if exe is not None and exe.suffix.lower() == ".exe":
        targets.append(exe.parent / f"{exe.stem}_Data" / "StreamingAssets" / "replays" / "latest.json")
    written = []
    for dst in targets:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".json.tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)            # atomic swap: the viewer never reads a half-written file
        written.append(dst)
    return written


def launch_viewer(replay_path: str | Path | None = None, extra_args: Sequence[str] = (),
                  repo_root: str | Path | None = None, wait: bool = False) -> subprocess.Popen:
    """Start the viewer, optionally with ``--replay <file>`` and extra arguments
    (e.g. ``["--camera", "top", "--autoplay-speed", "2", "--title", "Class Final"]``).

    Returns the Popen handle; with ``wait=True`` waits for it to exit.
    """
    exe = find_viewer(repo_root)
    if exe is None:
        raise FileNotFoundError(f"Unity viewer not built. Run: {BUILD_HINT}")
    args: list[str] = [str(exe)]
    if replay_path is not None:
        args += ["--replay", str(Path(replay_path).resolve())]
    args += list(extra_args)
    proc = subprocess.Popen(args, cwd=str(exe.parent))
    if wait:
        proc.wait()
    return proc


def run_selftest(replay_path: str | Path, out_json: str | Path | None = None,
                 repo_root: str | Path | None = None, timeout: float = 600) -> int:
    """Headless check of a replay in the real viewer (no window). Returns the exit code (0 = pass)."""
    exe = find_viewer(repo_root)
    if exe is None:
        raise FileNotFoundError("Unity viewer not built")
    args = [str(exe), "-batchmode", "-nographics", "--selftest", "--replay", str(Path(replay_path).resolve())]
    if out_json:
        args += ["--selftest-out", str(Path(out_json).resolve())]
    return subprocess.run(args, cwd=str(exe.parent), timeout=timeout).returncode


def show_in_unity(replay_path: str | Path, *, launch: bool = True, extra_args: Sequence[str] = (),
                  repo_root: str | Path | None = None, log=print) -> bool:
    """Install a replay as latest.json and (if the viewer is built) start it. Never raises: prints
    what happened and what to do next. Returns True if the replay was installed."""
    try:
        written = install_replay(replay_path, repo_root)
    except FileNotFoundError as exc:
        log(f"Unity: {exc}")
        return False
    log(f"Unity: installed as {written[0]}" + (f" (+ {len(written) - 1} build copy)" if len(written) > 1 else ""))
    exe = find_viewer(repo_root)
    if exe is None:
        log(f"Unity: viewer not built yet - build it once with: {BUILD_HINT}")
        log("       (or open unity/RaceViewer in Unity Hub and press Play: it loads latest.json)")
    elif launch:
        already_open = viewer_running(exe)
        try:
            # Always launch: an open viewer takes the new replay over from the new instance (which then
            # quits), so there is still one window - unless --new-window is in extra_args.
            launch_viewer(replay_path, extra_args, repo_root)
            if already_open and "--new-window" not in extra_args:
                log(f"Unity: handed the replay to the open {exe.name} (still one window)")
            else:
                log(f"Unity: started {exe.name} (an open viewer also reloads latest.json by itself)")
        except OSError as exc:
            log(f"Unity: could not start the viewer: {exc}")
    return True


def viewer_running(exe: str | Path) -> bool:
    """Is a viewer with this executable name already running? (best effort: False when unsure)"""
    name = Path(exe).name
    try:
        if sys.platform == "win32":
            out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {name}", "/NH"], capture_output=True,
                                 text=True, timeout=10).stdout
            return name.lower() in out.lower()
        stem = Path(exe).stem
        return subprocess.run(["pgrep", "-x", stem], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Show an Agentic GP replay in the Unity 3D viewer.")
    ap.add_argument("replay", nargs="?", help="replay JSON (schema v1)")
    ap.add_argument("--install-only", action="store_true", help="copy as latest.json, do not launch")
    ap.add_argument("--selftest", action="store_true", help="run the viewer's headless self-test on the replay")
    ap.add_argument("--repo", help="repository root (default: auto-detect)")
    # anything else (e.g. --camera top --autoplay-speed 2 --title "Final") is passed on to the viewer
    args, extra = ap.parse_known_args(list(argv) if argv is not None else None)
    extra = [a for a in extra if a != "--"]
    if args.selftest:
        if not args.replay:
            ap.error("--selftest needs a replay")
        code = run_selftest(args.replay, repo_root=args.repo)
        print("self-test", "PASSED" if code == 0 else f"FAILED (exit {code})")
        return code
    if args.replay:
        for p in install_replay(args.replay, args.repo):
            print("installed", p)
    if args.install_only:
        return 0
    exe = find_viewer(args.repo)
    if exe is None:
        print(f"Viewer not built yet - run: {BUILD_HINT}", file=sys.stderr)
        return 1
    launch_viewer(args.replay, extra, args.repo)
    print("started", exe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
