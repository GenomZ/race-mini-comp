"""Find driver files and read what is in them WITHOUT running them (static AST inspection).

    from app.lib.drivers import discover_drivers, inspect_driver
    for info in discover_drivers():
        print(info.label, info.style, info.params, info.uses_llm)

Folders, in this order: drivers/submissions, drivers/examples, lesson/checkpoints (+ the template).
Nothing is imported or executed here: a student file with a bug (or an endless loop) can never
break the app page that lists it. Running a driver is the runner's job (agentic_gp.race).
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .paths import DRIVER_FOLDERS, TEMPLATE, rel

LLM_CALLS = {"chat", "chat_json", "ask_with_tools", "complete", "get_langchain_model"}
METHODS = ("plan", "drive", "reflect")


@dataclass(frozen=True)
class DriverInfo:
    path: Path
    group: str                       # "submissions" | "examples" | "checkpoints" | "template" | "other"
    stem: str
    name: str                        # NAME (or the file name)
    author: str | None = None
    color: str | None = None
    kind: str | None = None
    model: str | None = None
    motto: str | None = None
    anatomy: tuple = ()
    style: str = "none"              # "function" | "class" | "none" (no drive found)
    has_plan: bool = False
    has_drive: bool = False
    has_reflect: bool = False
    params: dict = field(default_factory=dict)       # literal PARAMS (numbers / bools / strings)
    param_help: dict = field(default_factory=dict)   # the comment written after each PARAMS entry
    llm_in: tuple = ()               # which of plan/drive/reflect (directly or via helpers) call the LLM
    imports: tuple = ()              # imported module names
    features: frozenset = frozenset()  # see _features(): "sensors", "map", "tools", "memory", ...
    syntax_error: str | None = None

    @property
    def uses_llm(self) -> bool:
        return bool(self.llm_in) or "llm_import" in self.features

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.group}/{self.path.name})"

    @property
    def rel_path(self) -> str:
        return rel(self.path)


# ------------------------------------------------------------------------------ discovery
def discover_drivers(folders: list[tuple[str, Path]] | None = None, include_template: bool = True) -> list[DriverInfo]:
    """Every driver file in the driver folders (files starting with "_" are skipped)."""
    out: list[DriverInfo] = []
    for group, folder in folders or DRIVER_FOLDERS:
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.py")):
            if p.name.startswith("_"):
                continue
            out.append(inspect_driver(p, group))
    if include_template and TEMPLATE.exists():
        out.append(inspect_driver(TEMPLATE, "template"))
    return out


def find_driver(name_or_path: str, drivers: list[DriverInfo] | None = None) -> DriverInfo | None:
    """Look a driver up by NAME, file stem or path."""
    p = Path(name_or_path)
    if p.suffix == ".py" and p.exists():
        return inspect_driver(p)
    for d in drivers or discover_drivers():
        if name_or_path in (d.name, d.stem, d.path.name, d.rel_path):
            return d
    return None


def inspect_driver(path: str | Path, group: str | None = None) -> DriverInfo:
    """Static facts about one driver file (cached until the file changes)."""
    p = Path(path).resolve()
    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0
    return _inspect(str(p), group or _group_of(p), mtime)


def _group_of(p: Path) -> str:
    for group, folder in DRIVER_FOLDERS:
        try:
            p.relative_to(folder.resolve())
            return group
        except ValueError:
            continue
    return "template" if p == TEMPLATE.resolve() else "other"


@lru_cache(maxsize=256)
def _inspect(path: str, group: str, mtime: float) -> DriverInfo:
    p = Path(path)
    try:
        source = p.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return DriverInfo(p, group, p.stem, p.stem, syntax_error=f"cannot read: {exc}")
    except SyntaxError as exc:
        return DriverInfo(p, group, p.stem, p.stem, syntax_error=f"line {exc.lineno}: {exc.msg}")

    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver"), None)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    methods = {n.name: n for n in (cls.body if cls else []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    scope = methods if cls is not None else funcs
    meta = _literals(tree, cls)
    params = meta.get("PARAMS") if isinstance(meta.get("PARAMS"), dict) else {}
    imports = _imports(tree)
    llm_names = _llm_names(tree)
    llm_in = tuple(m for m in METHODS if m in scope and _reaches_llm(scope[m], scope, funcs, llm_names))
    style = "class" if cls is not None and "drive" in methods else "function" if "drive" in funcs else "none"
    return DriverInfo(
        path=p, group=group, stem=p.stem, name=str(meta.get("NAME") or p.stem),
        author=_s(meta.get("AUTHOR")), color=_s(meta.get("COLOR")), kind=_s(meta.get("KIND")),
        model=_s(meta.get("MODEL")), motto=_s(meta.get("MOTTO")),
        anatomy=tuple(meta.get("ANATOMY") or ()) if isinstance(meta.get("ANATOMY"), (list, tuple)) else (),
        style=style, has_plan="plan" in scope, has_drive="drive" in scope, has_reflect="reflect" in scope,
        params=dict(params), param_help=_param_comments(source, params), llm_in=llm_in, imports=tuple(sorted(imports)),
        features=frozenset(_features(source, tree, scope, imports, llm_names, meta)),
    )


def _s(v: Any) -> str | None:
    return None if v is None else str(v)


def _literals(tree: ast.Module, cls: ast.ClassDef | None) -> dict:
    """Constant assignments at module level and in class Driver (the class wins)."""
    out: dict = {}
    for body in (tree.body, cls.body if cls else []):
        for node in body:
            targets = node.targets if isinstance(node, ast.Assign) else \
                [node.target] if isinstance(node, ast.AnnAssign) and node.value is not None else []
            for t in targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                        out.setdefault(t.id, None)
    return out


def _imports(tree: ast.Module) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def _llm_names(tree: ast.Module) -> set[str]:
    """Local names that mean "the LLM module" (llm, L, ...) or an LLM function (chat, chat_json, ...)."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "agentic_gp":
            names |= {a.asname or a.name for a in node.names if a.name == "llm"}
        elif isinstance(node, ast.ImportFrom) and node.module == "agentic_gp.llm":
            names |= {a.asname or a.name for a in node.names if a.name in LLM_CALLS}
        elif isinstance(node, ast.Import):
            names |= {a.asname for a in node.names if a.name == "agentic_gp.llm" and a.asname}
    return names


def _calls_llm(fn: ast.AST, llm_names: set[str]) -> bool:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name) and f.id in llm_names:
            return True
        if isinstance(f, ast.Attribute) and f.attr in LLM_CALLS:
            base = f.value
            if isinstance(base, ast.Name) and base.id in llm_names:
                return True
            if isinstance(base, ast.Attribute) and base.attr == "llm":      # agentic_gp.llm.chat(...)
                return True
    return False


def _called_helpers(fn: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self":
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


def _reaches_llm(fn: ast.AST, scope: dict, funcs: dict, llm_names: set[str]) -> bool:
    """Does `fn` call the LLM, directly or through helper methods / module functions?"""
    seen, todo = set(), [fn]
    while todo:
        f = todo.pop()
        if id(f) in seen:
            continue
        seen.add(id(f))
        if _calls_llm(f, llm_names):
            return True
        for name in _called_helpers(f):
            nxt = scope.get(name) or funcs.get(name)
            if nxt is not None:
                todo.append(nxt)
    return False


def _code_only(tree: ast.Module) -> str:
    """The program text without comments and docstrings (so a docstring that MENTIONS the sensors
    does not count as using them)."""
    import copy

    t = copy.deepcopy(tree)
    for node in ast.walk(t):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr)                 and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.unparse(t)


def _features(source: str, tree: ast.Module, scope: dict, imports: set[str], llm_names: set[str], meta: dict) -> set[str]:
    full = source
    source = _code_only(tree)
    low = source.lower()
    feats = set()
    if any(i.startswith("agentic_gp.llm") or i == "agentic_gp.llm" for i in imports) or llm_names:
        feats.add("llm_import")
    if "distance_to_wall" in source or "wall_repulsion" in source:
        feats.add("sensors")
    if "steer_towards" in source:
        feats.add("steer_towards")
    if "recover_if_stuck" in source:
        feats.add("recovery")
    if any(k in source for k in ("centerline", "PathFollower", "curvature_radii", "speed_profile", "racing_line")):
        feats.add("map")
    if "racing_line" in source:
        feats.add("racing_line")
    if "briefing" in source:
        feats.add("briefing")
    if "ask_with_tools" in source or "tools=" in source:
        feats.add("tools")
    if "lesson" in low or "notebook" in low:
        feats.add("notebook")
    if "steward" in low:
        feats.add("steward")
    if "next_phase" in source:
        feats.add("lock_in")
    if any(i.startswith("agentic_gp.fly") for i in imports) or meta.get("KIND") == "fly":
        feats.add("fly")
    if "langgraph" in " ".join(imports):
        feats.add("langgraph")
    if "PARAMS" in meta:
        feats.add("params")
    if meta.get("NAME"):
        feats.add("name")
    refl = scope.get("reflect")
    if refl is not None:
        if any(isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) for n in ast.walk(refl)) or \
                any(isinstance(n, ast.Attribute) and n.attr in ("append", "update") for n in ast.walk(refl)):
            feats.add("memory")
        body = ast.get_source_segment(full, refl) or ""
        if "best" in body.lower():
            feats.add("keep_best")
    for node in ast.walk(tree):                 # an LLM call inside try/except = a fallback
        if isinstance(node, ast.Try) and any(_calls_llm(b, llm_names) for b in node.body):
            feats.add("llm_fallback")
            break
    return feats


def _param_comments(source: str, params: dict) -> dict:
    """{"aggression": "x physics corner speed"} from `"aggression": 1.0,  # x physics corner speed`."""
    if not params:
        return {}
    out: dict = {}
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}
    last_key: tuple[str, int] | None = None
    for tok in toks:
        if tok.type == tokenize.STRING:
            try:
                val = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(val, str) and val in params:
                last_key = (val, tok.start[0])
        elif tok.type == tokenize.COMMENT and last_key and tok.start[0] == last_key[1]:
            out.setdefault(last_key[0], tok.string.lstrip("#").strip())
    return out
