"""PARAMS -> sliders: every numeric knob of a driver becomes a slider, automatically.

    from app.lib.params_ui import params_sliders
    info = inspect_driver("drivers/examples/l2_planner.py")
    overrides = params_sliders(info.params, key="live", help=info.param_help)
    run_session(load_driver(info.path, params=overrides), ...)

A driver declares its knobs as `PARAMS = {"aggression": 1.0, ...}` (see agentic_gp/driver.py).
Numbers get sliders whose range is guessed from the default value (0 .. 2x the default, finer
steps for small numbers); True/False gets a checkbox; anything else is shown but not editable.
Pass `ranges={"aggression": (0.5, 1.5, 0.05)}` to choose a range yourself.
"""

from __future__ import annotations

import math
from typing import Any


def _nice_step(span: float) -> float:
    """A round step that gives ~100 positions over `span` (0.01, 0.02, 0.05, 0.1, ...)."""
    raw = max(span, 1e-9) / 100.0
    exp = math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * 10 ** exp >= raw:
            return float(m * 10 ** exp)
    return float(10 ** (exp + 1))


def slider_spec(default: Any) -> tuple[Any, Any, Any] | None:
    """(min, max, step) for a default value, or None if it is not a plain number."""
    if isinstance(default, bool) or not isinstance(default, (int, float)) or not math.isfinite(default):
        return None
    if isinstance(default, int):
        hi = max(10, 2 * abs(default))
        lo = 0 if default >= 0 else -hi
        return int(lo), int(hi), 1
    d = float(default)
    if d == 0.0:
        return -1.0, 1.0, 0.01
    hi = 2.0 * abs(d) if d > 0 else 0.0
    lo = 0.0 if d > 0 else 2.0 * d
    if 0 < d <= 1.0:
        hi = max(hi, 1.0)                                  # factors like 0.92 can go above 1
    step = _nice_step(hi - lo)
    lo, hi = round(lo / step) * step, round(hi / step) * step
    return round(lo, 10), round(hi, 10), step


def params_sliders(params: dict, *, key: str, help: dict | None = None, ranges: dict | None = None,
                   container: Any = None, columns: int = 2) -> dict:
    """Draw one widget per PARAM and return the chosen values ({name: value}, all params).

    key        prefix for the widget keys (use a different one per page / driver)
    help       {name: tooltip} - e.g. DriverInfo.param_help (the comments next to PARAMS)
    ranges     {name: (min, max, step)} to override the guessed slider range
    container  where to draw (default: the main area); e.g. st.sidebar or an st.expander
    """
    import streamlit as st

    box = container if container is not None else st
    values: dict = {}
    if not params:
        box.caption("This driver has no PARAMS (add `PARAMS = {\"cap\": 30.0}` to get sliders here).")
        return values
    cols = box.columns(max(1, int(columns)))
    for i, (name, default) in enumerate(params.items()):
        col = cols[i % len(cols)]
        tip = (help or {}).get(name)
        wkey = f"{key}__param__{name}"
        spec = (ranges or {}).get(name) or slider_spec(default)
        if isinstance(default, bool):
            values[name] = col.checkbox(name, value=default, key=wkey, help=tip)
        elif spec is not None:
            lo, hi, step = spec
            if isinstance(default, int) and not isinstance(step, float):
                values[name] = col.slider(name, int(lo), int(hi), int(default), int(step), key=wkey, help=tip)
            else:
                v = min(max(float(default), float(lo)), float(hi))
                values[name] = col.slider(name, float(lo), float(hi), v, float(step), key=wkey, help=tip)
        else:
            col.text_input(name, value=repr(default), disabled=True, key=wkey, help=tip)
            values[name] = default
    return values


def changed_params(defaults: dict, chosen: dict) -> dict:
    """Only the PARAMS that differ from the driver's defaults (nice for labels: 'aggression=1.1')."""
    return {k: v for k, v in chosen.items() if k in defaults and v != defaults[k]}


def params_label(defaults: dict, chosen: dict) -> str:
    diff = changed_params(defaults, chosen)
    return ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
                     for k, v in diff.items()) or "default PARAMS"
