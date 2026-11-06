"""Small building blocks every Race Control page uses (header, empty states, pickers, LLM status).

    from app.lib.ui import page_header, empty_state, driver_picker, track_picker, fast_mode
    page_header("Replay", "Watch any race again: scrub, pause, follow a car.", chapter="Ch0")

Kept separate from app/components (the canvases) so a page stays short and readable: the page
says WHAT it shows, these helpers deal with the boring parts (missing files, picking a driver).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import streamlit as st

from .drivers import DriverInfo, discover_drivers
from .paths import REPO_ROOT, rel


def fast_mode() -> bool:
    """AGP_FAST=1 (set by the tests): pages shorten every heavy computation."""
    return os.environ.get("AGP_FAST", "") not in ("", "0")


def show_mystery() -> bool:
    """The Mystery GP stays secret until the instructor reveals it: files with "mystery" in their path
    (replays, results, LLM logs - their prompts describe the track) are hidden on every page until the
    sidebar toggle "Show Mystery GP files" is on (or the app is opened with ?mystery=1)."""
    if "show_mystery" not in st.session_state and \
            str(st.query_params.get("mystery", "")).lower() in ("1", "true", "on"):
        st.session_state.show_mystery = True
    return bool(st.session_state.get("show_mystery", False))


def hide_mystery(paths: Iterable[Path]) -> tuple[list[Path], int]:
    """(the paths without mystery-track files unless they are revealed, how many were hidden)."""
    paths = list(paths)
    if show_mystery():
        return paths, 0
    keep = [p for p in paths if "mystery" not in rel(p).lower()]
    return keep, len(paths) - len(keep)


def mystery_note(hidden: int) -> None:
    """One grey line when mystery-track files are hidden on this page."""
    if hidden:
        st.caption(f":material/visibility_off: {hidden} Mystery GP file(s) hidden until the reveal "
                   "(sidebar: Show Mystery GP files).")


def presenter_mode() -> bool:
    """Presenter mode (sidebar toggle, or ?presenter=1): compact header, canvases sized to the window,
    setup panels folded - for a 1280x720 projector."""
    return bool(st.session_state.get("presenter", False))


# ------------------------------------------------------------------------------ text blocks
def page_header(title: str, learn: str, *, chapter: str | None = None, icon: str | None = None) -> None:
    """Page title + the 1-3 line "what you learn here" caption (+ the chapter it belongs to)."""
    st.title(title, anchor=False)
    badge = f":blue-badge[{chapter}] " if chapter else ""
    if presenter_mode():                    # projector: the screen belongs to the race, the caption is a tooltip
        st.markdown(f"{badge}:gray[**What you learn here** (hover)]", help=learn)
        return
    st.markdown(f"{badge}:gray[**What you learn here:** {learn}]")


INSTRUCTOR_PASSWORD_VAR = "AGP_INSTRUCTOR_PASSWORD"
HOSTED_VARS = ("AGP_HOSTED", "RAILWAY_ENVIRONMENT", "RAILWAY_ENVIRONMENT_NAME", "RAILWAY_PROJECT_ID")


def require_instructor() -> None:
    """Stop the page until the instructor password is entered (env var AGP_INSTRUCTOR_PASSWORD).

    No password set: open on a laptop (the classroom default), locked on a hosted deployment
    (Railway, or AGP_HOSTED=1) - a public URL must never accept code uploads without one.
    """
    import hmac

    from agentic_gp import llm

    llm.load_env()                                       # the password may live in .env like the key
    password = os.environ.get(INSTRUCTOR_PASSWORD_VAR, "")
    if not password:
        if any(os.environ.get(v) for v in HOSTED_VARS):
            st.error(f"This page is locked: set {INSTRUCTOR_PASSWORD_VAR} in the deployment's environment "
                     "variables to use it on a hosted server.", icon=":material/lock:")
            st.stop()
        return
    if st.session_state.get("instructor_ok"):
        if st.sidebar.button("Lock instructor pages", icon=":material/lock:"):
            st.session_state.instructor_ok = False
            st.rerun()
        return
    with st.form("instructor_login"):
        entered = st.text_input("Instructor password", type="password")
        if st.form_submit_button("Unlock", icon=":material/lock_open:"):
            if hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8")):
                st.session_state.instructor_ok = True
                st.rerun()
            st.error("Wrong password.", icon=":material/block:")
    st.stop()


def empty_state(message: str, how: str | None = None, *, icon: str = ":material/info:") -> None:
    """A friendly 'nothing here yet' box with the command that fixes it."""
    st.info(message, icon=icon)
    if how:
        st.code(how, language="bash")


def demo_hint() -> None:
    empty_state("The demo data is missing.", ".venv/Scripts/python.exe races/demo/make_demo.py")


# ------------------------------------------------------------------------------ pickers
def driver_picker(label: str, *, key: str, default_stem: str = "l2_planner", container=None,
                  only: str | None = None, exclude_llm: bool = False) -> DriverInfo | None:
    """Selectbox over every driver file (submissions first). only="code" | "llm" filters by LLM use."""
    box = container if container is not None else st
    drivers = [d for d in discover_drivers() if d.style != "none" and d.group != "template"]
    if only == "code" or exclude_llm:
        drivers = [d for d in drivers if not d.uses_llm]
    elif only == "llm":
        drivers = [d for d in drivers if d.uses_llm]
    if not drivers:
        box.warning("No driver files found (drivers/examples, drivers/submissions).")
        return None
    labels = [d.label for d in drivers]
    index = next((i for i, d in enumerate(drivers) if d.stem == default_stem), 0)
    choice = box.selectbox(label, labels, index=index, key=key,
                           help="Your file goes in drivers/submissions/ (copy drivers/template.py).")
    return drivers[labels.index(choice)]


def track_picker(label: str = "Track", *, key: str, default: str = "grand_prix", container=None,
                 n_random: int = 3, custom_seed: bool = True) -> str:
    """Selectbox of the public tracks + random:<seed>; with custom_seed a number input for any seed."""
    from .loaders import track_choices

    box = container if container is not None else st
    choices = track_choices(n_random)
    if custom_seed:
        choices = choices + ["random:<seed>"]
    spec = box.selectbox(label, choices, index=choices.index(default) if default in choices else 0, key=key,
                         help="random:<seed> builds a new track from a number - same seed, same track, everywhere.")
    if spec == "random:<seed>":
        seed = box.number_input("Seed", 0, 10_000_000, 42, key=f"{key}_seed")
        spec = f"random:{int(seed)}"
    return spec


# ------------------------------------------------------------------------------ LLM status
def llm_status() -> dict:
    """Which LLM a call from this process would use: provider, model, key present (never the key)."""
    try:
        from agentic_gp import llm

        cfg = llm.get_config()
        return {"provider": cfg.provider, "model": cfg.model, "effort": cfg.effort,
                "available": llm.llm_available(), "real": cfg.provider not in ("mock", "replay"),
                "label": f"{cfg.provider}:{cfg.model}"}
    except Exception as exc:  # noqa: BLE001 - a broken config must not break the page
        return {"provider": "?", "model": "?", "effort": "?", "available": False, "real": False,
                "label": f"not configured ({type(exc).__name__})"}


def set_llm_provider(mock: bool) -> None:
    """DEPRECATED - do not use in pages: it changes LLM_PROVIDER for the WHOLE server (every browser).

    Pass the choice per run instead: llm_run_env(real=...) as the `env` of a child process
    (LiveRun(env=...), the grand prix worker). Kept so old code keeps working.
    """
    if mock:
        if os.environ.get("LLM_PROVIDER") != "mock":
            os.environ["AGP_APP_PREV_PROVIDER"] = os.environ.get("LLM_PROVIDER", "")
        os.environ["LLM_PROVIDER"] = "mock"
    elif os.environ.get("LLM_PROVIDER") == "mock":
        prev = os.environ.pop("AGP_APP_PREV_PROVIDER", "")
        if prev and prev != "mock":
            os.environ["LLM_PROVIDER"] = prev
        else:
            os.environ.pop("LLM_PROVIDER", None)


def real_provider() -> str:
    """The real provider a "Real LLM" run uses: LLM_PROVIDER (from the environment / .env) unless that is
    mock/replay, then AGP_REAL_PROVIDER or minimax. Never changes os.environ's provider."""
    try:
        from agentic_gp import llm

        llm.load_env()                                   # .env -> os.environ once (keys; env vars win)
    except Exception:                                    # noqa: BLE001
        pass
    p = os.environ.get("LLM_PROVIDER", "minimax").strip().lower() or "minimax"
    if p in ("mock", "replay"):
        p = (os.environ.get("AGP_REAL_PROVIDER") or "minimax").strip().lower()
    return p


def real_llm_possible() -> bool:
    """True when a real (non-mock) provider has a key configured (.env or environment). Read-only."""
    try:
        from agentic_gp import llm

        p = real_provider()
        if p not in llm.DEFAULT_MODELS or p in ("mock", "replay"):
            return False
        if p == "ollama":
            return True
        key_var = llm.API_KEYS.get(p)
        return bool(key_var and os.environ.get(key_var))
    except Exception:                                    # noqa: BLE001 - a broken config: no real calls
        return False


def llm_run_env(real: bool, log_dir: str | Path | None = None) -> dict:
    """Environment for ONE run's child process: the provider this browser picked (+ its LLM log folder).

    Per run, never process-wide: another browser's choice cannot change it.
    """
    env = {"LLM_PROVIDER": real_provider() if real else "mock"}
    if log_dir is not None:
        env["LLM_LOG_DIR"] = str(log_dir)
    return env


def sidebar_status() -> None:
    """The small LLM line at the bottom of the sidebar."""
    s = llm_status()
    if s["real"] and s["available"]:
        st.sidebar.caption(f":material/cloud_done: LLM: **{s['label']}** (real calls, key loaded)")
    elif s["provider"] == "mock":
        st.sidebar.caption(":material/cloud_off: LLM: **mock** (offline, free)")
    else:
        st.sidebar.caption(f":material/cloud_off: LLM: {s['label']} - no key (LLM drivers fall back)")


# ------------------------------------------------------------------------------ misc
def file_choice(label: str, paths: Iterable[Path], *, key: str, default_name: str | None = None,
                fmt=None, container=None) -> Path | None:
    """Selectbox over files with repo-relative labels (None when the list is empty)."""
    box = container if container is not None else st
    paths = list(paths)
    if not paths:
        return None
    index = next((i for i, p in enumerate(paths) if default_name and p.name == default_name), 0)
    return box.selectbox(label, paths, index=index, key=key, format_func=fmt or (lambda p: rel(p)))


def repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def fmt_time(t: float | None, crashes: int | None = None) -> str:
    if t is None:
        return "DNF"
    return f"{t:.2f} s" + (f" ({crashes} crash{'es' if crashes != 1 else ''})" if crashes is not None else "")


def mark_dead_heats(rows: list[dict], time_key: str = "time", status_key: str = "status") -> list[dict]:
    """Rows with the SAME finishing time (to 0.01 s) get "dead heat" in their status (a tie, not a bug)."""
    seen: dict = {}
    for r in rows:
        t = r.get(time_key)
        if isinstance(t, (int, float)):
            seen.setdefault(round(float(t), 2), []).append(r)
    for group in seen.values():
        if len(group) > 1:
            for r in group:
                st_ = str(r.get(status_key) or "finished")
                r[status_key] = st_ if "dead heat" in st_ else f"{st_} - dead heat ({len(group)} cars)"
    return rows


def short_status(status: str | None, limit: int = 18) -> str:
    """'DNF (driver crashed: KeyError ...)' -> 'DNF (driver crashed)' style short form for metrics/tables."""
    s = str(status or "").strip()
    if len(s) <= limit:
        return s
    head = s.split(":")[0]
    if "(" in head and not head.endswith(")"):
        head += ")"
    return head if len(head) <= limit + 4 else head[:limit].rstrip() + "..."
