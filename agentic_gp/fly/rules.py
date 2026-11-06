"""Mechanical checks of the Fly-class rule (Fly Cup, "Biologically Honest" ribbon).

    from agentic_gp.fly.rules import check_fly_rules
    for issue in check_fly_rules("drivers/my_fly.py"):        # [] = nothing found
        print(issue)                                            # "warning: Fly-class rule 2: ..."

    python -m agentic_gp.fly.rules drivers/my_fly.py            # same, from the command line

The rule (README.md, "The Fly-class rule"):

1. Throttle and steering are decoded from neuron activity with a fixed decoder.
2. Sensors enter only through the encoder, as smooth functions (no if/else on sensors), and the
   encoder drives only INPUT (sensory) neurons - never a neuron the decoder reads.
3. SPINAL_CORD exception: a speed rule may combine neuron outputs with the car's own ``speed``,
   only inside a section marked ``SPINAL_CORD`` and declared in ``SPINAL_CORD = "..."``.  It never
   reads walls, waypoints or the map.
4. REFLEX exception: a contact reflex (wall closer than a few units) may drive a decoded neuron
   directly only if declared in ``REFLEX = "..."``.  Nothing else may.
5. Best Fly Brain: the fly must beat its own scrambled control (``PARAMS["SCRAMBLE"]``).

These checks are a LINT: they catch the common ways to break rules 2, 3 and 5 (a branch on a sensor
value, a spinal cord that reads walls, a missing declaration or SCRAMBLE param).  Stewards still read
the encoder and the SPINAL_CORD / REFLEX sections in the pit interview (rule 2's "decoded neurons get
no encoder input" cannot be checked statically).  Every issue is a "warning": a file that breaks the
Fly-class rule can still race, just not in the Fly class.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["FlyIssue", "check_fly_rules", "sensor_branches", "SENSOR_KEYS"]

#: observation keys that are sensors (walls, waypoints, the map); "speed" is proprioception
SENSOR_KEYS = re.compile(r"^(distance_to_|angle_to_|position|heading|track|waypoint|next_waypoint|lap_progress)")
_PROPRIO = {"speed"}
_SECTION = re.compile(r"^\s*#\s*=+\s*(end\s+)?SPINAL_CORD\b", re.IGNORECASE)


@dataclass(frozen=True)
class FlyIssue:
    level: str        # "warning" (Fly-class only; the file can still race)
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.message}"


def _tainted_expr(node: ast.AST, tainted: set[str]) -> bool:
    """Does `node` carry sensor information (anything read from ``obs``, speed included)?  Taint
    flows through arithmetic and calls of plain functions (max, np.clip, a local lambda ...), but
    NOT through ``self.<method>(...)`` / ``self.<obj>.<method>(...)`` calls: those are the brain
    (encoder -> neurons -> decoder), whose outputs are neuron activity."""
    if isinstance(node, ast.Call):
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id == "self" and isinstance(node.func, ast.Attribute):
            return False
        parts = [node.func] + list(node.args) + [k.value for k in node.keywords]
        return any(_tainted_expr(p, tainted) for p in parts)
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return f"self.{node.attr}" in tainted
    return any(_tainted_expr(c, tainted) for c in ast.iter_child_nodes(node))


def _targets(t: ast.AST) -> list[str]:
    if isinstance(t, ast.Name):
        return [t.id]
    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
        return [f"self.{t.attr}"]
    if isinstance(t, (ast.Tuple, ast.List)):
        return [x for e in t.elts for x in _targets(e)]
    if isinstance(t, ast.Subscript):
        return _targets(t.value)
    return []


def _taint(fn: ast.AST, start: set[str], shared: set[str]) -> set[str]:
    """Fixpoint of names assigned from sensor-carrying expressions inside `fn` (plus `shared`
    self attributes tainted elsewhere in the file)."""
    tainted = set(start) | shared
    changed = True
    while changed:
        changed = False
        for n in ast.walk(fn):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and n.value is not None:
                tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                if _tainted_expr(n.value, tainted):
                    for name in (x for t in tg for x in _targets(t)):
                        if name not in tainted:
                            tainted.add(name)
                            changed = True
            elif isinstance(n, (ast.For, ast.comprehension)) and _tainted_expr(n.iter, tainted):
                for name in _targets(n.target):
                    if name not in tainted:
                        tainted.add(name)
                        changed = True
    return tainted


def sensor_branches(source: str, functions=None, sensors=("obs",), class_name: str | None = None) -> list[tuple[int, str, str]]:
    """``(line, function, condition)`` for every if / ternary / match / while whose condition depends
    on a sensor.  ``functions``: names to check (default: every function); ``sensors``: names that
    hold sensor values on entry (``obs`` plus, e.g., a ``bearing_deg`` parameter);
    ``class_name``: only methods of that class."""
    tree = ast.parse(source)
    scope = tree
    if class_name:
        scope = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if scope is None:
            return []
    fns = [n for n in ast.walk(scope) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    start = set(sensors) | {"obs"}
    shared: set[str] = set()
    for _ in range(3):                                     # self.<attr> taint across methods
        for fn in fns:
            shared |= {t for t in _taint(fn, start, shared) if t.startswith("self.")}
    found = []
    for fn in fns:
        if functions is not None and fn.name not in functions:
            continue
        tainted = _taint(fn, start, shared)
        for n in ast.walk(fn):
            test = None
            if isinstance(n, (ast.If, ast.IfExp, ast.While)):
                test = n.test
            elif isinstance(n, ast.Match):
                test = n.subject
            elif isinstance(n, ast.comprehension) and n.ifs:
                test = ast.BoolOp(op=ast.And(), values=n.ifs)
            if test is not None and _tainted_expr(test, tainted):
                seg = ast.get_source_segment(source, test) if hasattr(test, "lineno") else "comprehension filter"
                found.append((getattr(test, "lineno", getattr(n, "lineno", 0)), fn.name, (seg or "").strip()))
    return sorted(found)


def _declared(tree: ast.Module, name: str):
    """Value of a plain-string declaration `name = "..."` at module level or in class Driver
    (None = not declared, ... = declared but not a plain string)."""
    scopes = [tree.body]
    drv = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver"), None)
    if drv is not None:
        scopes.append(drv.body)
    value = None
    for body in scopes:
        for n in body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
                v = n.value
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    value = v.value
                elif isinstance(v, ast.Name) and v.id == name:
                    continue                                  # SPINAL_CORD = SPINAL_CORD in the class
                else:
                    try:
                        value = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        value = ...
    return value


def _kind(tree: ast.Module) -> str | None:
    drv = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver"), None)
    for body in ([drv.body] if drv else []) + [tree.body]:
        for n in body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KIND" for t in n.targets):
                if isinstance(n.value, ast.Constant):
                    return str(n.value.value)
    return None


def _params_keys(tree: ast.Module) -> set[str]:
    drv = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Driver"), None)
    for body in ([drv.body] if drv else []) + [tree.body]:
        for n in body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in n.targets) \
                    and isinstance(n.value, ast.Dict):
                return {k.value for k in n.value.keys if isinstance(k, ast.Constant)}
    return set()


def check_fly_rules(path: str | Path, *, only_if_fly: bool = True) -> list[FlyIssue]:
    """Lint a driver file against the Fly-class rule.  With ``only_if_fly`` (default) a file whose
    ``KIND`` is not "fly" returns [] (the rule only applies to fly brains)."""
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return []                                            # check_driver_source reports these
    if only_if_fly and _kind(tree) != "fly":
        return []
    issues: list[FlyIssue] = []
    warn = lambda msg: issues.append(FlyIssue("warning", f"Fly-class {msg}"))  # noqa: E731

    # rule 2: no if/else on sensors (outside the declared SPINAL_CORD and REFLEX sections and the radio)
    lines = source.splitlines()
    section = [False] * (len(lines) + 2)
    inside = False
    for i, line in enumerate(lines, 1):
        m = _SECTION.match(line)
        if m:
            inside = not m.group(1)
        section[i] = inside
    for lineno, fn, cond in sensor_branches(source):
        if "radio" in fn.lower() or section[min(lineno, len(section) - 1)]:
            continue
        warn(f"rule 2: line {lineno}, {fn}(): `if {cond}` branches on a sensor - sensors may enter only "
             "through the encoder, as smooth functions (np.clip instead of if/else)")

    # rule 3: SPINAL_CORD declared; its section reads only `speed`
    spinal = _declared(tree, "SPINAL_CORD")
    if spinal is None:
        warn('rule 3: no SPINAL_CORD declaration - add  SPINAL_CORD = ""  if every control comes from '
             'neurons, or describe your speed rule in it')
    elif spinal is ...:
        warn("rule 3: SPINAL_CORD must be a plain string")
    elif spinal.strip():
        if not any(section):
            warn("rule 3: SPINAL_CORD is declared but no section is marked  # ==== SPINAL_CORD ... "
                 "# ==== end SPINAL_CORD")
        for i, line in enumerate(lines, 1):
            if section[i]:
                for key in re.findall(r"['\"]([a-z_0-9]+)['\"]", line):
                    if key not in _PROPRIO and SENSOR_KEYS.match(key):
                        warn(f"rule 3: line {i}: the SPINAL_CORD section reads {key!r} - it may read only "
                             "neuron outputs and the car's own speed")

    # rule 4: a reflex must be declared (we can only check that the declaration is a string)
    reflex = _declared(tree, "REFLEX")
    if reflex is ...:
        warn("rule 4: REFLEX must be a plain string")

    # rule 5: a scrambled control
    if "SCRAMBLE" not in _params_keys(tree):
        warn('rule 5: no PARAMS["SCRAMBLE"] - a Fly-class finisher must beat its own scrambled control to '
             "win Best Fly Brain (see fly_toy.py)")
    return issues


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:])
    if not paths:
        print("usage: python -m agentic_gp.fly.rules DRIVER.py [...]")
        return 2
    bad = 0
    for p in paths:
        issues = check_fly_rules(p, only_if_fly=False)
        print(f"{p}: {'Fly-class OK (lint)' if not issues else f'{len(issues)} issue(s)'}")
        for i in issues:
            print(f"  {i}")
        bad += bool(issues)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
