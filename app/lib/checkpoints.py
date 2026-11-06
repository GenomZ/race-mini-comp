"""'Where am I?': look at a student's driver file and tell which checkpoint (CP0-CP8) it reached.

    from app.lib.checkpoints import where_am_i, render_where_am_i
    w = where_am_i("drivers/submissions/ada.py")
    w.level, w.next_step, [(c.cp, c.ok) for c in w.checks]
    render_where_am_i()                # the sidebar widget (any page)

Static only (the file is parsed, never run - see app/lib/drivers.py), so it is instant and safe.
It cannot see lap times: "clean on every track" checkpoints are about the CODE having the parts
of that chapter. The race itself (`python race.py validate`) is the real exam.

Ladder (CURRICULUM section 4): CP0 pit lane -> CP1 reflexes -> CP2 map -> CP3 practice ->
CP5 strategist -> CP6 crew chief -> CP8 garage. Side quests: CP4 (LLM pilot file) and CP7 (fly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .drivers import DriverInfo, discover_drivers, inspect_driver


@dataclass
class Check:
    cp: str
    title: str
    ok: bool
    hint: str                       # what to add when it is not ok yet
    side: bool = False              # side quest (not on the main ladder)


@dataclass
class WhereAmI:
    info: DriverInfo
    level: str                      # "CP0".."CP8", or "-" when even CP0 is missing
    checks: list[Check] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)   # race.py check errors / warnings
    next_step: str = ""

    @property
    def summary(self) -> str:
        i = self.info
        parts = [f"{i.style} driver" if i.style != "none" else "no drive() yet"]
        if i.has_plan:
            parts.append("plan")
        if i.has_reflect:
            parts.append("reflect")
        if i.llm_in:
            parts.append("LLM in " + "/".join(i.llm_in))
        if i.params:
            parts.append(f"{len(i.params)} PARAMS")
        return ", ".join(parts)


MAIN = ("CP0", "CP1", "CP2", "CP3", "CP5", "CP6", "CP8")


def _check_issues(path: Path) -> list[tuple[str, str]]:
    try:
        from agentic_gp.race.loader import check_driver_source
    except ImportError:                                   # pragma: no cover - core always present
        return []
    return [(i.level, i.message) for i in check_driver_source(path)]


def where_am_i(path: str | Path) -> WhereAmI:
    info = inspect_driver(path)
    f = info.features
    if info.syntax_error:
        return WhereAmI(info, "-", [], [f"error: {info.syntax_error}"], "Fix the syntax error first (see the line number).")
    issues = _check_issues(info.path)
    errors = [m for lvl, m in issues if lvl == "error"]
    llm = set(info.llm_in)
    checks = [
        Check("CP0", "Pit lane: drive(), NAME and AUTHOR, `race.py check` green",
              info.has_drive and "name" in f and info.author is not None and not errors,
              "set NAME / AUTHOR / COLOR and make `python race.py check <file>` pass"),
        Check("CP1", "Reflexes: reads the sensors, knobs in PARAMS",
              info.has_drive and bool(info.params) and bool({"sensors", "steer_towards", "recovery", "map"} & f),
              "steer to the waypoint, brake for the wall in front, put your numbers in PARAMS"),
        Check("CP2", "The map: class Driver with plan() using the track",
              info.style == "class" and info.has_plan and "map" in f,
              "class Driver: plan(track) -> speeds from track['centerline'], drive with PathFollower"),
        Check("CP3", "Practice: reflect() learns between attempts",
              info.has_reflect and "memory" in f,
              "add reflect(report): keep the best settings, slow down where you crashed"),
        Check("CP4", "LLM pilot: the LLM drives tick by tick (separate _pilot file)", "drive" in llm,
              "optional: <you>_pilot.py with chat_json() in drive() - never raced", side=True),
        Check("CP5", "Strategist: the LLM plans once, code drives (+ briefing)",
              "plan" in llm and "briefing" in f,
              "in plan(): chat_json() with the corner table + track['briefing'], validate, clamp, fall back"),
        Check("CP6", "Crew chief: LLM + tools in reflect(), memory, steward",
              "reflect" in llm and "tools" in f and bool({"notebook", "steward"} & f),
              "in reflect(): ask_with_tools(report['summary_text'], tools), a lessons notebook, steward checks"),
        Check("CP7", "Fly brain: a spiking neural network drives (side quest)", "fly" in f,
              "optional: <you>_fly.py with agentic_gp.fly", side=True),
        Check("CP8", "Garage: final file - plan + reflect, lock-in, survives without the LLM",
              not errors and info.has_plan and info.has_reflect and "lock_in" in f and (not llm or "llm_fallback" in f),
              "lock in on report['next_phase'] and wrap every LLM call in try/except (LLM off must still finish)"),
    ]
    reached = [c.cp for c in checks if c.ok and not c.side]
    level = reached[-1] if reached else "-"
    if level != "-":                                      # everything below the reached level counts as done
        top = MAIN.index(level)
        for c in checks:
            if not c.side and MAIN.index(c.cp) < top and not c.ok and c.cp != "CP0":
                c.ok = True
    nxt = next((c for c in checks if not c.side and not c.ok), None)
    next_step = f"Next ({nxt.cp}): {nxt.hint}" if nxt else "All checkpoints reached - race it: python race.py validate <file>"
    return WhereAmI(info, level, checks, [f"{lvl}: {m}" for lvl, m in issues], next_step)


# ------------------------------------------------------------------------------ the sidebar widget
def render_where_am_i(container=None, key: str = "where_am_i", expanded: bool = False) -> WhereAmI | None:
    """'Where am I?' expander: pick your file, see your checkpoint and the next step.

    Defaults to the sidebar; pass any container (st.container(), a column, ...) to put it elsewhere.
    """
    import streamlit as st

    box = container if container is not None else st.sidebar
    drivers = discover_drivers()
    if not drivers:
        box.caption("Where am I? - no driver files found yet (copy drivers/template.py to drivers/submissions/).")
        return None
    with box.expander("Where am I?", expanded=expanded, icon=":material/flag:"):
        labels = [d.label for d in drivers]
        choice = st.selectbox("Your driver file", labels, key=key, label_visibility="collapsed")
        w = where_am_i(drivers[labels.index(choice)].path)
        n = sum(1 for c in w.checks if c.ok and not c.side)
        st.markdown(f"**{w.level}** reached  \n:gray[{w.summary}]")
        st.progress(n / len(MAIN))
        lines = []
        for c in w.checks:
            icon = ":material/check_circle:" if c.ok else ":material/radio_button_unchecked:"
            color = "green" if c.ok else "gray"
            lines.append(f"{icon} :{color}[**{c.cp}** {c.title}]" + (" :gray[(side quest)]" if c.side else ""))
        st.markdown("  \n".join(lines))
        st.info(w.next_step, icon=":material/arrow_forward:")
        for msg in w.issues[:4]:
            (st.error if msg.startswith("error") else st.caption)(msg)
    return w
