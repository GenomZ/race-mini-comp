"""Inbox (instructor): collect student driver files, scrutineer them, file them in drivers/submissions/."""

import hashlib
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from app.lib.inbox_tools import SMOKE_TIMEOUT_S, sanitize, smoke_check, status_of
from app.lib.paths import DRIVERS_DIR, rel
from app.lib.ui import page_header, require_instructor

page_header("Inbox", "Scrutineering before the race: every file is checked statically (imports, interface, "
            "metadata), then driven for 40 ticks in its own process with the mock LLM. Only clean files go "
            "into drivers/submissions/ - the folder the grand prix races.", chapter="Instructor")
require_instructor()

SUBMISSIONS = Path(os.environ.get("AGP_SUBMISSIONS_DIR") or DRIVERS_DIR / "submissions")   # tests point this elsewhere
STAGING = Path(tempfile.gettempdir()) / "agp_inbox"
CHECK_TIMEOUT_S = SMOKE_TIMEOUT_S                  # 60 s: a clean check takes 2-10 s


@st.cache_data(show_spinner=False, max_entries=256)
def status_cached(path: str, mtime: float) -> dict:
    """status_of() cached until the file changes."""
    return status_of(path)


# ------------------------------------------------------------------------------ 1. upload
st.subheader("1. Upload", anchor=False)
ups = st.file_uploader("Student driver files (.py) - drop several at once", type=["py"], accept_multiple_files=True,
                       key="inbox_up")
staged = st.session_state.setdefault("inbox_staged", {})       # target name -> {"path", "orig", "smoke"}
handled = st.session_state.setdefault("inbox_handled", set())   # accepted / discarded uploads (by content)
for up in ups or []:
    target = sanitize(up.name)
    data = up.getvalue()
    digest = hashlib.sha1(data).hexdigest()[:12]
    if digest in handled:                                     # still in the uploader, but already dealt with
        continue
    folder = STAGING / digest
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / target
    if not p.exists():
        p.write_bytes(data)
    if staged.get(target, {}).get("path") != str(p):
        staged[target] = {"path": str(p), "orig": up.name, "smoke": None, "digest": digest}

if not staged:
    st.caption("Nothing uploaded yet. Students can also copy their file straight into drivers/submissions/.")

# ------------------------------------------------------------------------------ 2. check + accept
for target, item in list(staged.items()):
    p = Path(item["path"])
    if not p.exists():
        staged.pop(target)
        continue
    s = status_cached(str(p), p.stat().st_mtime)
    exists = (SUBMISSIONS / target).exists()
    with st.container(border=True):
        h1, h2 = st.columns([3, 2], vertical_alignment="center")
        h1.markdown(f"**{item['orig']}** -> `drivers/submissions/{target}`" + ("  :orange-badge[replaces a file]" if exists else ""))
        h2.markdown(f"NAME **{s['name']}** · AUTHOR {s['author'] or '?'} · {s['kind'] or '?'} · "
                    f"code has the parts of **{s['level']}**")
        # Two separate verdicts, side by side: the static check READS the code, the smoke test RUNS it.
        smoke = item.get("smoke")
        v1, v2 = st.columns(2)
        with v1:
            if s["errors"]:
                st.error("Static check: FAILED\n\n" + "\n".join(f"- {m}" for m in s["errors"]), icon=":material/block:")
            else:
                st.success("Static check: OK" + (f" ({len(s['warnings'])} warning(s))" if s["warnings"] else ""),
                           icon=":material/check_circle:")
        with v2:
            if s["errors"]:
                st.info("Smoke test: fix the static errors first.", icon=":material/science:")
            elif smoke is None:
                st.warning("Smoke test: not run yet - run it before accepting.", icon=":material/science:")
            elif smoke[0]:
                st.success("Smoke test: PASSED (it drives)", icon=":material/check:")
            else:
                st.error("Smoke test: FAILED - see the output below", icon=":material/close:")
        for w in s["warnings"]:
            st.caption(f":material/warning: {w}")
        b1, b2, b3, b4 = st.columns([1, 1, 1, 2], vertical_alignment="center")
        if b1.button("Smoke test", key=f"inbox_smoke_{target}", icon=":material/science:", width="stretch",
                     disabled=bool(s["errors"])):
            with st.spinner(f"Driving 40 ticks in a separate process (mock LLM, stopped after {CHECK_TIMEOUT_S:.0f} s)..."):
                item["smoke"] = smoke_check(p, CHECK_TIMEOUT_S)
            st.rerun()
        passed = bool(smoke and smoke[0])
        force = False
        if not s["errors"] and not passed:
            force = b4.checkbox("Accept anyway (smoke test not passed)", key=f"inbox_force_{target}",
                                help="For a file you have read yourself. It races in its own process like every "
                                     "student file, but it may DNF.")
        accept = b2.button("Accept", key=f"inbox_accept_{target}", type="primary" if passed else "secondary",
                           icon=":material/move_to_inbox:", width="stretch",
                           disabled=bool(s["errors"]) or not (passed or force),
                           help="Copy into drivers/submissions/ (an older file with that name goes to _old/). "
                                "Needs a PASSED smoke test, or 'Accept anyway'.")
        if b3.button("Discard", key=f"inbox_drop_{target}", icon=":material/delete:", width="stretch"):
            handled.add(item.get("digest"))
            staged.pop(target)
            st.rerun()
        if smoke is not None:
            with st.expander("Smoke test output", expanded=not smoke[0], icon=":material/terminal:"):
                st.code(smoke[1] or "(no output)", language=None, wrap_lines=True)
        if accept:
            SUBMISSIONS.mkdir(parents=True, exist_ok=True)
            dest = SUBMISSIONS / target
            if dest.exists():
                old = SUBMISSIONS / "_old"
                old.mkdir(exist_ok=True)
                dest.replace(old / f"{dest.stem}_{time.strftime('%H%M%S')}.py")
            dest.write_bytes(p.read_bytes())
            handled.add(item.get("digest"))
            staged.pop(target)
            st.toast(f"Accepted {rel(dest)}", icon=":material/check:")
            st.rerun()
        with st.expander("Source", icon=":material/code:"):
            st.code(p.read_text(encoding="utf-8", errors="replace")[:20000], language="python", line_numbers=True)

# ------------------------------------------------------------------------------ 3. the submissions folder
st.subheader("2. drivers/submissions/ (these race in the grand prix)", anchor=False)
files = sorted(f for f in SUBMISSIONS.glob("*.py") if not f.name.startswith("_")) if SUBMISSIONS.exists() else []
if not files:
    st.info("No submissions yet.", icon=":material/info:")
else:
    rows = []
    for f in files:
        s = status_cached(str(f), f.stat().st_mtime)
        rows.append({"file": f.name, "NAME": s["name"], "AUTHOR": s["author"], "kind": s["kind"], "LLM in": s["llm"],
                     "checkpoint": s["level"], "check": "error" if s["errors"] else ("warning" if s["warnings"] else "ok"),
                     "problems": "; ".join(s["errors"] + s["warnings"])[:200],
                     "modified": time.strftime("%H:%M:%S", time.localtime(f.stat().st_mtime))})
    n_bad = sum(r["check"] == "error" for r in rows)
    m = st.columns(3)
    m[0].metric("Submissions", len(rows))
    m[1].metric("Static errors", n_bad)
    m[2].metric("Use an LLM", sum(r["LLM in"] != "-" for r in rows))
    st.dataframe(rows, hide_index=True, width="stretch")
    st.code("python lesson/instructor/grade.py scrutineer --grid races/grid   # check every file, copy the PASSED ones\n"
            "python race.py validate drivers/submissions/<handle>.py --random 3  # the full exam, one file\n"
            "python race.py grand-prix races/grid --house --out races/class_gp.json --unity", language="bash")
