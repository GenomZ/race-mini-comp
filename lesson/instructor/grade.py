"""Instructor autograder: scrutineering at the deadline, the race grid, and the optional course grade.

Run from the repo root (any OS; the commands never print an API key):

    python lesson/instructor/grade.py scrutineer                      # check every file in drivers/submissions
    python lesson/instructor/grade.py scrutineer --grid races/grid    # ... and copy the PASSED main files there
    python lesson/instructor/grade.py scrutineer --fly --grid races/flycup   # the <handle>_fly.py files instead
    python lesson/instructor/grade.py grade                           # the rubric -> races/grading/grades.md + .csv

File roles in the submissions folder: `<handle>.py` is the main file (it races), `<handle>_pilot.py` (Ch4)
and `<handle>_fly.py` (Ch7, Fly Cup) are side files, names starting with "_" are ignored.

scrutineer (per file): the static check (`agentic_gp.race.loader.check_driver_source`), the smoke test
(`race.py check FILE` in its own process with the mock LLM; an endless loop is reported after ~15 s), an
API-key grep (the key from .env / the environment and anything that looks like `sk-...`), a house car's
NAME (Max, Planner, Fly Toy: FAIL), and duplicate NAMEs among the main files (the championship adds points
up by NAME: the file uploaded FIRST keeps the name, later copies FAIL). Warnings: AUTHOR still
"your_handle", an unchanged copy of a handrail/example, LLM calls in drive() (refused in official races),
a file name that is not a plain handle (the grid gets a cleaned copy, e.g. "Ann Lee (v3).py" -> ann_lee_v3.py).
Exit code 1 if any file failed.

grade (CURRICULUM section 6, "Optional grading"; each criterion = weight x earned / possible, items that
do not apply are left out of "possible"):

    40 Working  `race.py check` PASSED (15); finishes every public track with the LLM off (10);
                clean = finished with 0 crashes on every public track (15). Measured with
                `race.py validate FILE --tracks <public> --random 0` under LLM_PROVIDER=mock.
    30 Design   a map: plan() uses the track (8); reflect() keeps memory (5), keeps the best / locks in (2),
                measured improvement: a later practice/attempt beat the first one in a snapshot, the
                final or results/*.json (3); the LLM: called inside try/except (3) and fed the briefing or
                tools (3) - or, without an LLM, a filled-in "LLM: none, because ..." card line (6);
                the 5-line DRIVER CARD filled in (6). A main file that is a handrail or example with
                only its metadata changed gets 0 design points (flagged; override if you disagree).
    20 Process  appears in the snapshot grand prix results in --snapshots (12 x share of snapshots);
                a screenshot / PDF in --evidence whose name starts with the handle (5);
                a Ch4 pilot or Ch7 fly side file (3).
    10 Result   per final race (home, mystery): race time at or better than the class median (2.5)
                and finished ahead of Max (2.5).

Static facts come from app/lib/checkpoints.py (the "Where am I?" widget: parses, never runs student
code). Runs are cached in <out>/runs/ and reused while the file is unchanged (--rerun forces, --no-run
uses only the cache). Manual corrections: --overrides FILE.csv with columns handle,criterion,points,note
(criterion = working | design | process | result; points replace the computed ones).
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PUBLIC_TRACKS = ("grand_prix", "oval", "sprint", "speedway", "briefing_demo")
WEIGHTS = {"working": 40, "design": 30, "process": 20, "result": 10}
META_NAMES = {"NAME", "AUTHOR", "COLOR", "KIND", "MOTTO", "MODEL", "ANATOMY"}
CARD_FIELDS = ("Approach", "LLM", "reflect()", "Anatomy", "Another day")
EVIDENCE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}
KEY_VARS = ("MINIMAX_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{16,}")
CHECK_TIMEOUT_S = 90.0        # race.py check stops an endless loop itself after ~15 s; this is the backstop
VALIDATE_TIMEOUT_S = 900.0
HOUSE_NAME = "Max"
HOUSE_NAMES = ("Max", "Planner", "Fly Toy")      # the house cars (agentic_gp.race.loader.HOUSE_NAMES)
PLACEHOLDER_AUTHORS = ("your_handle", "agentic gp", "")


# ============================================================================ files and roles
def role_of(path: Path) -> str:
    """'main' | 'pilot' | 'fly' | 'ignore'."""
    if path.suffix != ".py" or path.name.startswith("_"):
        return "ignore"
    if path.stem.endswith("_pilot"):
        return "pilot"
    if path.stem.endswith("_fly"):
        return "fly"
    return "main"


def handle_of(path: Path) -> str:
    """'ada' for ada.py / ada_pilot.py / ada_fly.py. Odd LMS file names are cleaned up the same way the
    grid copies are named: "Ann Lee (v3).py" -> "ann_lee_v3" (letters of any script are kept)."""
    stem = path.stem.lower()
    for suffix in ("_pilot", "_fly"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return re.sub(r"\W+", "_", stem).strip("_") or "driver"


def grid_name(path: Path) -> str:
    """The file name a driver gets in the race grid: <handle>.py, or <handle>_fly.py for a Fly Cup file."""
    return handle_of(path) + {"fly": "_fly", "pilot": "_pilot"}.get(role_of(path), "") + ".py"


def submission_files(folder: Path, roles: tuple[str, ...] = ("main", "pilot", "fly")) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"ERROR: submissions folder not found: {folder}")
    return sorted(p for p in folder.glob("*.py") if role_of(p) in roles)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_meta(path: Path) -> dict:
    """Literal NAME / AUTHOR / KIND / COLOR (module level, else class Driver) - parsed, never run."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}
    out: dict = {}
    scopes = [tree.body] + [n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver"]
    for body in scopes:
        for node in body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id in ("NAME", "AUTHOR", "KIND", "COLOR"):
                        out.setdefault(t.id, node.value.value)
    return out


# ============================================================================ secrets
def known_keys(env_file: Path | None = None) -> list[str]:
    """API key values from the environment and .env (only used for comparison, never printed)."""
    values = [os.environ.get(v, "") for v in KEY_VARS]
    env_file = env_file or REPO / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() in KEY_VARS:
                values.append(value.split("#")[0].strip().strip("'\""))
    return sorted({v for v in values if len(v) >= 8})


def key_leak(text: str, keys: list[str]) -> str | None:
    """A reason (without the key itself) if the text contains an API key, else None."""
    if any(k in text for k in keys):
        return "contains the course API key - remove it, and rotate the key"
    if KEY_PATTERN.search(text):
        return "contains something that looks like an API key (sk-...) - remove it"
    return None


# ============================================================================ handrails and examples
def _normalized(source: str) -> str | None:
    """AST dump without the docstring and the metadata assignments (comments are gone anyway)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    def clean(body: list) -> list:
        out = []
        for i, node in enumerate(body):
            if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if all(isinstance(t, ast.Name) and t.id in META_NAMES for t in targets):
                    continue
            if isinstance(node, ast.ClassDef):
                node.body = clean(node.body) or [ast.Pass()]
            out.append(node)
        return out

    tree.body = clean(tree.body)
    return ast.dump(tree, include_attributes=False)


_REFERENCES: dict[str, str] | None = None


def reference_copies() -> dict[str, str]:
    """normalized AST -> repo path, for every handrail, example and the template."""
    global _REFERENCES
    if _REFERENCES is None:
        _REFERENCES = {}
        files = sorted((REPO / "lesson" / "checkpoints").glob("*.py")) + \
            sorted((REPO / "drivers" / "examples").glob("*.py")) + [REPO / "drivers" / "template.py"]
        for f in files:
            if f.is_file():
                norm = _normalized(f.read_text(encoding="utf-8", errors="replace"))
                if norm:
                    _REFERENCES.setdefault(norm, f.relative_to(REPO).as_posix())
    return _REFERENCES


def verbatim_copy_of(path: Path) -> str | None:
    norm = _normalized(path.read_text(encoding="utf-8", errors="replace"))
    return reference_copies().get(norm) if norm else None


# ============================================================================ the driver card
def driver_card(path: Path) -> dict[str, str]:
    """{"Approach": "...", ...} from the DRIVER CARD lines of the module docstring (filled-in only)."""
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8", errors="replace"))) or ""
    except SyntaxError:
        return {}
    card: dict[str, str] = {}
    for line in doc.splitlines():
        label, sep, value = line.strip().partition(":")
        value = value.strip()
        if sep and label.strip() in CARD_FIELDS and value and not value.startswith("<"):
            card.setdefault(label.strip(), value)
    return card


# ============================================================================ running race.py
def _run(args: list[str], timeout: float) -> tuple[int, str]:
    env = dict(os.environ, LLM_PROVIDER="mock", PYTHONIOENCODING="utf-8", LLM_QUIET="1", LLM_NO_DOTENV="1")
    for var in KEY_VARS:                         # student files never run in a process that holds a key
        env.pop(var, None)
    try:
        p = subprocess.run([sys.executable, str(REPO / "race.py"), *args], cwd=str(REPO), env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout:.0f} s (an endless loop at import, in plan() or in drive()?)"


def cached_run(kind: str, path: Path, args: list[str], timeout: float, runs_dir: Path, *, rerun: bool = False,
               no_run: bool = False, tag: str = "") -> dict | None:
    """Run race.py once per file version (sha256 + tag); the output is kept in runs_dir."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    cache = runs_dir / f"{path.stem}.{kind}.json"
    sha = file_sha256(path)
    if cache.is_file() and not rerun:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            # reuse a run of this exact file version; a failed or timed-out run is retried (it may have been
            # a busy machine), unless --no-run says never start race.py
            if data.get("sha256") == sha and data.get("tag", "") == tag and (data.get("returncode") == 0 or no_run):
                return data
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    if no_run:
        return None
    code, out = _run(args, timeout)
    data = {"file": str(path), "sha256": sha, "tag": tag, "returncode": code, "output": out[-20000:]}
    cache.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def check_passed(run: dict | None) -> bool:
    return bool(run) and run["returncode"] == 0 and "PASSED" in run["output"]


def parse_validate(output: str) -> dict[str, dict]:
    """The table printed by `race.py validate` -> {track: {"finished", "time", "crashes", "status"}}."""
    rows: dict[str, dict] = {}
    lines = output.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.split()[:3] == ["track", "practice", "timed"]) + 2
    except StopIteration:
        return rows
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) < 7:
            break
        track, timed, crashes = parts[0], parts[2], parts[3]
        status = " ".join(parts[5:-1])
        finished = timed != "DNF"
        rows[track] = {"finished": finished, "time": float(timed) if finished else None,
                       "crashes": int(crashes) if crashes.isdigit() else None, "status": status}
    return rows


# ============================================================================ scrutineering
def scrutineer_file(path: Path, keys: list[str], runs_dir: Path, *, rerun: bool = False) -> dict:
    from agentic_gp.race.loader import check_driver_source

    meta = read_meta(path)
    row = {"file": path.name, "role": role_of(path), "handle": handle_of(path), "name": meta.get("NAME"),
           "author": meta.get("AUTHOR"), "kind": meta.get("KIND"), "problems": [], "warnings": [],
           "smoke": "skipped", "mtime": path.stat().st_mtime}
    row["problems"] += static_problems(path, keys)
    issues = check_driver_source(path)
    row["warnings"] += [i.message for i in issues if i.level != "error" and "house car's name" not in i.message]
    row["warnings"] += extra_warnings(path, meta)
    if not row["problems"]:
        run = cached_run("check", path, ["check", "--isolate", str(path.resolve())], CHECK_TIMEOUT_S, runs_dir, rerun=rerun)
        row["smoke"] = "PASSED" if check_passed(run) else ("TIMEOUT" if run["returncode"] == -1 else "FAILED")
        if row["smoke"] != "PASSED":
            lines = [ln.strip() for ln in run["output"].splitlines()]
            tail = [ln for ln in lines if "ERROR" in ln or "TIMEOUT" in ln] or \
                [ln for ln in lines if "FAILED" in ln]
            row["problems"].append("race.py check: " + (tail[0] if tail else row["smoke"])[:200])
    row["verdict"] = "FAIL" if row["problems"] else "PASS"
    return row


def static_problems(path: Path, keys: list[str]) -> list[str]:
    """Reasons a file may not race that need no run: static-check errors, an API key, a house car's NAME."""
    from agentic_gp.race.loader import check_driver_source

    out = [f"static: {i.message}" for i in check_driver_source(path) if i.level == "error"]
    leak = key_leak(path.read_text(encoding="utf-8", errors="replace"), keys)
    if leak:
        out.append(leak)
    name = read_meta(path).get("NAME")
    if isinstance(name, str) and name.strip().lower() in {h.lower() for h in HOUSE_NAMES}:
        out.append(f"NAME {name!r} is a house car's name - ask the student to rename (standings, awards and "
                   "LLM logs would mix the two cars up)")
    return out


def extra_warnings(path: Path, meta: dict) -> list[str]:
    """Things the instructor should see but that do not stop a car from racing."""
    out = []
    author = str(meta.get("AUTHOR") or "").strip().lower()
    if author in PLACEHOLDER_AUTHORS:
        out.append(f"AUTHOR is {meta.get('AUTHOR')!r} - not the student's handle yet")
    copy_of = verbatim_copy_of(path)
    if copy_of:
        out.append(f"unchanged copy of {copy_of} (only metadata differs)")
    if role_of(path) == "main":
        try:
            from app.lib.checkpoints import where_am_i
            if "drive" in tuple(where_am_i(path).info.llm_in):
                out.append("drive() calls the LLM - every such call is refused in official races (the car races "
                           "on its fallback, and its LLM log fills with refusals)")
        except Exception:                                # noqa: BLE001 - a warning only
            pass
    if grid_name(path) != path.name:
        out.append(f"file name is not a plain handle - the grid copy is named {grid_name(path)}")
    return out


def flag_duplicates(rows: list[dict]) -> None:
    """Two main (or two fly) files with the same NAME (or AUTHOR) would be merged in the championship.

    Same NAME: the file uploaded FIRST (oldest modification time; the file name breaks a tie) keeps it
    and races; every later copy FAILS until its student renames. Same AUTHOR: a warning on all of them."""
    for role in ("main", "fly"):
        group = [r for r in rows if r["role"] == role]
        for field_name, why in (("name", "the championship adds points up by NAME"),
                                ("author", "grading and awards match by handle")):
            seen: dict[str, list[dict]] = {}
            for r in group:
                value = str(r.get(field_name) or "").strip().lower()
                if value and value not in PLACEHOLDER_AUTHORS:
                    seen.setdefault(value, []).append(r)
            for same in seen.values():
                if len(same) < 2:
                    continue
                same.sort(key=lambda r: (r.get("mtime") or 0.0, r["file"]))
                first = same[0]
                for r in same:
                    others = ", ".join(o["file"] for o in same if o is not r)
                    msg = f"duplicate {field_name.upper()} {r.get(field_name)!r} (also {others}): {why}"
                    if field_name != "name":
                        r["warnings"].append(msg)
                    elif r is first:
                        r["warnings"].append(msg + f" - {r['file']} was uploaded first and keeps the name")
                    else:
                        r["problems"].append(msg + f" - {first['file']} was uploaded first: ask this student "
                                                   "to rename")
                        r["verdict"] = "FAIL"


def cmd_scrutineer(args: argparse.Namespace) -> int:
    folder = Path(args.submissions)
    roles = ("fly",) if args.fly else ("main", "pilot", "fly") if args.all else ("main",)
    files = submission_files(folder, roles)
    if not files:
        print(f"No {'/'.join(roles)} files in {folder}.")
        return 1
    out = Path(args.out)
    runs = out / "runs"
    keys = known_keys()
    print(f"Scrutineering {len(files)} file(s) in {folder} ({', '.join(roles)}); smoke tests run with the mock LLM, "
          f"{args.jobs} at a time ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = list(pool.map(lambda p: scrutineer_file(p, keys, runs, rerun=args.rerun), files))
    flag_duplicates(rows)
    for r in rows:
        r.pop("mtime", None)
    table = [[r["verdict"], r["file"], r["name"] or "-", r["author"] or "-", r["kind"] or "-", r["smoke"],
              "; ".join(r["problems"] + [f"warning: {w}" for w in r["warnings"]])[:110]] for r in rows]
    print(_text_table(["verdict", "file", "NAME", "AUTHOR", "KIND", "smoke", "problems / warnings"], table))
    n_fail = sum(r["verdict"] == "FAIL" for r in rows)
    out.mkdir(parents=True, exist_ok=True)
    stem = "scrutineering_fly" if args.fly else "scrutineering"
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    md = [f"# Scrutineering - {folder}", "", f"{len(rows) - n_fail} passed, {n_fail} failed.", "",
          "| Verdict | File | NAME | AUTHOR | KIND | Smoke | Problems / warnings |", "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in table]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n{len(rows) - n_fail} passed, {n_fail} failed - report: {out / (stem + '.md')}")
    if args.grid:
        grid = Path(args.grid)
        grid.mkdir(parents=True, exist_ok=True)
        for old in grid.glob("*.py"):
            old.unlink()
        grid_role = "fly" if args.fly else "main"      # pilot files never race
        passed = [folder / r["file"] for r in rows if r["verdict"] == "PASS" and r["role"] == grid_role]
        for p in passed:
            shutil.copy2(p, grid / grid_name(p))       # "Ann Lee (v3).py" races as ann_lee_v3.py
        renamed = [f"{p.name} -> {grid_name(p)}" for p in passed if grid_name(p) != p.name]
        print(f"Grid: {len(passed)} file(s) copied to {grid} (old .py files there removed)"
              + (f"; renamed: {', '.join(renamed)}" if renamed else "") + ". Race it:")
        print(f"    python race.py grand-prix {grid.as_posix()} --house ...")
    return 1 if n_fail else 0


# ============================================================================ grand prix results
def load_gp_results(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) and data.get("format") == "agentic-gp-grand-prix" else None


def gp_cars(gp: dict) -> list[dict]:
    """One row per car: the standings row + qualifying row + the driver's attempts and source stem."""
    quali = {q.get("name"): q for q in gp.get("qualifying") or []}
    drivers = {d.get("name"): d for d in gp.get("drivers") or []}
    rows = []
    for s in gp.get("standings") or []:
        d = drivers.get(s.get("name"), {})
        src = d.get("source") or ""
        rows.append({**s, "handle": Path(src.replace("\\", "/")).stem.lower() if src else None,
                     "qualifying": quali.get(s.get("name"), {}), "attempts": d.get("attempts") or [],
                     "llm": d.get("llm") or {}})
    return rows


def find_car(cars: list[dict], handle: str, author: str | None) -> dict | None:
    """The student's car: by file name first, then by AUTHOR (never the template's 'your_handle')."""
    scored = [c for c in cars if not c.get("house")]
    hit = next((c for c in scored if c.get("handle") == handle), None)
    if hit is None and author and author.lower() != "your_handle":
        hit = next((c for c in scored if str(c.get("author") or "").lower() == author.lower()), None)
    return hit


def improved_in_gp(car: dict) -> bool:
    """A later practice / qualifying lap beat the first practice lap."""
    times = [a.get("race_time") for a in car.get("attempts", [])
             if a.get("phase") in ("practice", "qualifying") and a.get("completed")]
    first = [a for a in car.get("attempts", []) if a.get("phase") == "practice"][:1]
    if not first or not first[0].get("completed") or len(times) < 2:
        return False
    return min(times[1:]) < first[0]["race_time"] - 1e-9


def improved_in_results(results_dir: Path, handle: str, author: str | None) -> bool:
    """results/*.json (race.py drive --save): a session where a later attempt beat attempt 1."""
    if not results_dir.is_dir():
        return False
    for p in results_dir.glob("*.json"):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(r, dict) or r.get("format") != "agentic-gp-result":
            continue
        src = Path(str(r.get("source") or "").replace("\\", "/")).stem.lower()
        if src != handle and not (author and str(r.get("author") or "").lower() == author.lower()):
            continue
        atts = r.get("attempts") or []
        done = [a.get("race_time") for a in atts if a.get("completed")]
        if len(atts) >= 2 and atts[0].get("completed") and len(done) >= 2 and min(done[1:]) < atts[0]["race_time"]:
            return True
    return False


# ============================================================================ grading
@dataclass
class Item:
    label: str
    earned: float
    possible: float | None          # None = does not apply (left out of the criterion)
    note: str = ""


@dataclass
class Student:
    handle: str
    file: Path
    name: str | None = None
    author: str | None = None
    kind: str | None = None
    items: dict[str, list[Item]] = field(default_factory=lambda: {k: [] for k in WEIGHTS})
    flags: list[str] = field(default_factory=list)
    override: dict[str, tuple[float, str]] = field(default_factory=dict)

    def score(self, criterion: str) -> float:
        if criterion in self.override:
            return self.override[criterion][0]
        items = [i for i in self.items[criterion] if i.possible]
        possible = sum(i.possible for i in items)
        if not possible:
            return 0.0
        return round(WEIGHTS[criterion] * sum(i.earned for i in items) / possible, 1)

    @property
    def total(self) -> float:
        return round(sum(self.score(c) for c in WEIGHTS), 1)


def grade_working(st: Student, check: dict | None, validate: dict | None, tracks: tuple[str, ...]) -> None:
    items = st.items["working"]
    if check is None:
        items.append(Item("race.py check PASSED", 0, 15, "not run (--no-run and no cached output)"))
    else:
        items.append(Item("race.py check PASSED", 15 if check_passed(check) else 0, 15,
                          "" if check_passed(check) else "check failed"))
    rows = parse_validate(validate["output"]) if validate else {}
    n = len(tracks)
    finished = [t for t in tracks if rows.get(t, {}).get("finished")]
    clean = [t for t in finished if rows[t].get("crashes") == 0]
    missing = "" if validate else "not run"
    items.append(Item(f"finishes with the LLM off ({len(finished)}/{n})", 10 * len(finished) / n, 10,
                      missing or ", ".join(f"{t} DNF" for t in tracks if t not in finished)))
    items.append(Item(f"clean on the public tracks ({len(clean)}/{n})", 15 * len(clean) / n, 15,
                      missing or ", ".join(f"{t} {rows[t]['crashes']} crash(es)" for t in finished if t not in clean)))


def grade_design(st: Student, info, improved: bool, copy_of: str | None) -> None:
    items = st.items["design"]
    f = info.features if info else frozenset()
    has_plan = bool(info and info.has_plan)
    has_reflect = bool(info and info.has_reflect)
    llm_in = tuple(info.llm_in) if info else ()
    card = driver_card(st.file)
    items.append(Item("map: plan() uses the track", 8 if has_plan and "map" in f else 0, 8))
    items.append(Item("reflect() keeps memory", 5 if has_reflect and "memory" in f else 0, 5))
    items.append(Item("keeps the best / locks in", 2 if has_reflect and ({"keep_best", "lock_in"} & f) else 0, 2))
    items.append(Item("measured improvement (practice/attempts)", 3 if improved else 0, 3,
                      "" if improved else "no later lap beat the first one in the snapshots/final/results"))
    if llm_in:
        items.append(Item("LLM call inside try/except (fallback)", 3 if "llm_fallback" in f else 0, 3))
        items.append(Item("LLM gets the briefing or tools", 3 if {"briefing", "tools"} & f else 0, 3))
    else:
        why = card.get("LLM", "")
        items.append(Item("no LLM: justified on the driver card", 6 if why else 0, 6,
                          why[:60] if why else "card line 'LLM: none, because ...' missing"))
    items.append(Item(f"driver card ({len(card)}/5 lines)", 6 * len(card) / 5, 6))
    if copy_of:
        st.flags.append(f"verbatim copy of {copy_of} (only metadata changed): design 0")
        for it in items:
            it.earned = 0


def grade_process(st: Student, snapshots: list[tuple[str, list[dict]]], evidence_dir: Path, side_files: set[str]) -> None:
    items = st.items["process"]
    if snapshots:
        seen = [label for label, cars in snapshots if find_car(cars, st.handle, st.author)]
        items.append(Item(f"in {len(seen)}/{len(snapshots)} snapshots", 12 * len(seen) / len(snapshots), 12,
                          ", ".join(seen)))
    else:
        items.append(Item("snapshots", 0, None, "no snapshot results found"))
    shots = sorted(p.name for p in evidence_dir.glob("*") if p.suffix.lower() in EVIDENCE_SUFFIXES
                   and p.stem.lower().startswith(st.handle)) if evidence_dir.is_dir() else []
    items.append(Item("race-engineer report screenshot", 5 if shots else 0, 5, ", ".join(shots)))
    items.append(Item("Ch4 pilot or Ch7 fly file", 3 if side_files else 0, 3, ", ".join(sorted(side_files))))


def grade_result(st: Student, finals: list[tuple[str, list[dict]]]) -> None:
    items = st.items["result"]
    if not finals:
        items.append(Item("final races", 0, None, "no final results yet"))
        return
    for label, cars in finals:
        scored = [c for c in cars if not c.get("house")]
        times = [c["race_time"] for c in scored if c.get("race_time") is not None and not c.get("status")]
        car = find_car(cars, st.handle, st.author)
        finished = bool(car) and car.get("race_time") is not None and not car.get("status")
        median = statistics.median(times) if times else None
        ok = finished and median is not None and car["race_time"] <= median + 1e-9
        if not car:
            note = "did not race"
        elif not finished:
            note = car.get("status") or "DNF"
        else:
            note = f"{car['race_time']:.2f} s vs median {median:.2f} s" if median is not None else ""
        items.append(Item(f"{label}: at or better than the median", 2.5 if ok else 0, 2.5, note))
        house = next((c for c in cars if c.get("house") and c.get("name") == HOUSE_NAME), None)
        if house is None:
            items.append(Item(f"{label}: ahead of {HOUSE_NAME}", 0, None, f"{HOUSE_NAME} did not race"))
        else:
            ahead = bool(car) and finished and car["position"] < house["position"]
            items.append(Item(f"{label}: ahead of {HOUSE_NAME}", 2.5 if ahead else 0, 2.5,
                              f"P{car['position']} vs {HOUSE_NAME} P{house['position']}" if car else ""))


def load_overrides(path: Path | None) -> dict[str, dict[str, tuple[float, str]]]:
    out: dict[str, dict[str, tuple[float, str]]] = {}
    if not path:
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            crit = (row.get("criterion") or "").strip().lower()
            if crit not in WEIGHTS:
                raise SystemExit(f"ERROR: {path}: unknown criterion {crit!r} (use {', '.join(WEIGHTS)})")
            pts = max(0.0, min(float(row["points"]), WEIGHTS[crit]))
            out.setdefault(row["handle"].strip().lower(), {})[crit] = (pts, (row.get("note") or "").strip())
    return out


def _gp_files(folder: Path) -> list[tuple[str, list[dict]]]:
    out = []
    if folder.is_dir():
        for p in sorted(folder.glob("*_results.json")):
            gp = load_gp_results(p)
            if gp is not None:
                out.append((p.name[: -len("_results.json")], gp_cars(gp)))
    return out


def grade_all(args: argparse.Namespace) -> list[Student]:
    try:
        from app.lib.checkpoints import where_am_i
    except Exception:                                    # noqa: BLE001 - grading still works, design = 0
        where_am_i = None
    folder = Path(args.submissions)
    files = submission_files(folder)
    mains = [p for p in files if role_of(p) == "main"]
    if not mains:
        raise SystemExit(f"ERROR: no main driver files (<handle>.py) in {folder}")
    tracks = tuple(args.tracks)
    out = Path(args.out)
    runs = out / "runs"
    snapshots = _gp_files(Path(args.snapshots))
    finals = [(label, cars) for label, cars in _gp_files(Path(args.final)) if label in ("home", "mystery")]
    all_gps = snapshots + finals
    overrides = load_overrides(Path(args.overrides) if args.overrides else None)
    tag = ",".join(tracks)

    keys = known_keys()
    rejected = {p: static_problems(p, keys) for p in mains}
    rows = [{"file": p.name, "role": "main", "name": read_meta(p).get("NAME"), "author": read_meta(p).get("AUTHOR"),
             "mtime": p.stat().st_mtime, "problems": [], "warnings": [], "verdict": "PASS"} for p in mains]
    flag_duplicates(rows)                                # a later copy of a NAME never raced either
    for p, r in zip(mains, rows):
        rejected[p] += r["problems"]

    def runs_for(p: Path) -> tuple[dict | None, dict | None]:
        if rejected[p]:
            return None, None                            # failed scrutineering: never raced
        check = cached_run("check", p, ["check", "--isolate", str(p.resolve())], CHECK_TIMEOUT_S, runs, rerun=args.rerun,
                           no_run=args.no_run)
        if not check_passed(check):
            return check, None
        validate = cached_run("validate", p, ["validate", str(p.resolve()), "--tracks", *tracks, "--random", "0",
                                              "--quiet"], VALIDATE_TIMEOUT_S, runs, rerun=args.rerun,
                              no_run=args.no_run, tag=tag)
        return check, validate

    if not args.no_run:
        print(f"Running check + validate (mock LLM, {len(tracks)} tracks) for {len(mains)} file(s), {args.jobs} at a "
              f"time; unchanged files reuse {runs} ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        measured = dict(zip(mains, pool.map(runs_for, mains)))

    students = []
    for p in mains:
        meta = read_meta(p)
        st = Student(handle_of(p), p, meta.get("NAME"), meta.get("AUTHOR"), meta.get("KIND"))
        check, validate = measured[p]
        if rejected[p]:
            st.flags.append("failed scrutineering (" + "; ".join(rejected[p])[:120] + "): working 0")
            st.items["working"] = [Item("passed scrutineering", 0, 40, "; ".join(rejected[p])[:200])]
        else:
            grade_working(st, check, validate, tracks)
        info = None
        if where_am_i is not None:
            try:
                info = where_am_i(p).info
            except Exception as exc:                     # noqa: BLE001
                st.flags.append(f"static inspection failed: {exc}")
        improved = any(improved_in_gp(c) for _, cars in all_gps
                       for c in [find_car(cars, st.handle, st.author)] if c) or \
            improved_in_results(Path(args.results), st.handle, st.author)
        grade_design(st, info, improved, verbatim_copy_of(p))
        side = {f.name for f in files if role_of(f) in ("pilot", "fly") and handle_of(f) == st.handle}
        grade_process(st, snapshots, Path(args.evidence), side)
        grade_result(st, finals)
        st.override = overrides.get(st.handle, {})
        students.append(st)
    return students


def write_reports(students: list[Student], out: Path) -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out / "grades.csv", out / "grades.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["handle", "file", "NAME", "AUTHOR", "KIND", *WEIGHTS, "total", "flags"])
        for s in students:
            w.writerow([s.handle, s.file.name, s.name, s.author, s.kind, *[s.score(c) for c in WEIGHTS], s.total,
                        " | ".join(s.flags + [f"override {c}: {n}" for c, (_, n) in s.override.items()])])
    lines = ["# Grades (autograded - review before you publish)", "",
             "| Handle | NAME | Working /40 | Design /30 | Process /20 | Result /10 | Total | Flags |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for s in sorted(students, key=lambda s: -s.total):
        lines.append(f"| {s.handle} | {s.name or '-'} | " + " | ".join(
            f"{s.score(c)}" + ("*" if c in s.override else "") for c in WEIGHTS)
                     + f" | **{s.total}** | {'; '.join(s.flags)} |")
    lines += ["", "`*` = manual override. Per student:", ""]
    for s in sorted(students, key=lambda s: s.handle):
        lines += [f"## {s.handle} ({s.file.name}) - {s.total}", ""]
        for c in WEIGHTS:
            ov = f" - OVERRIDE {s.override[c][0]}: {s.override[c][1]}" if c in s.override else ""
            lines.append(f"- **{c} {s.score(c)}/{WEIGHTS[c]}**{ov}")
            for i in s.items[c]:
                pts = "n/a" if i.possible is None else f"{i.earned:g}/{i.possible:g}"
                lines.append(f"  - {i.label}: {pts}" + (f" ({i.note})" if i.note else ""))
        lines += [f"- flag: {f}" for f in s.flags] + [""]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def cmd_grade(args: argparse.Namespace) -> int:
    students = grade_all(args)
    csv_path, md_path = write_reports(students, Path(args.out))
    rows = [[s.handle, s.name or "-", *[s.score(c) for c in WEIGHTS], s.total, "; ".join(s.flags)[:60]]
            for s in sorted(students, key=lambda s: -s.total)]
    print(_text_table(["handle", "NAME", *[f"{c}/{w}" for c, w in WEIGHTS.items()], "total", "flags"], rows))
    print(f"\nWritten: {csv_path}, {md_path}")
    return 0


# ============================================================================ CLI
def _text_table(headers: list[str], rows: list[list]) -> str:
    cells = [[str(h) for h in headers]] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    fmt = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()   # noqa: E731
    return "\n".join([fmt(cells[0]), fmt(["-" * w for w in widths])] + [fmt(r) for r in cells[1:]])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="grade.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scrutineer", help="check every submission (static + smoke test + key grep)")
    p.add_argument("--submissions", default="drivers/submissions")
    p.add_argument("--grid", help="copy the PASSED files into this folder (its old .py files are removed)")
    p.add_argument("--fly", action="store_true", help="the <handle>_fly.py files (Fly Cup) instead of the main files")
    p.add_argument("--all", action="store_true", help="main, pilot and fly files")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rerun", action="store_true", help="ignore cached smoke tests")
    p.add_argument("--out", default="races/grading")
    p.set_defaults(func=cmd_scrutineer)

    p = sub.add_parser("grade", help="the grading rubric -> grades.md + grades.csv")
    p.add_argument("--submissions", default="drivers/submissions")
    p.add_argument("--snapshots", default="races/snapshots", help="folder of snapshot *_results.json")
    p.add_argument("--final", default="races/final", help="folder with home_results.json / mystery_results.json")
    p.add_argument("--results", default="results", help="race.py drive --save results")
    p.add_argument("--evidence", default="drivers/submissions/evidence", help="screenshots named <handle>*.png/pdf")
    p.add_argument("--tracks", nargs="+", default=list(PUBLIC_TRACKS))
    p.add_argument("--overrides", help="CSV: handle,criterion,points,note")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--rerun", action="store_true", help="ignore cached check/validate runs")
    p.add_argument("--no-run", action="store_true", help="use cached runs only (never start race.py)")
    p.add_argument("--out", default="races/grading")
    p.set_defaults(func=cmd_grade)
    return ap


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
