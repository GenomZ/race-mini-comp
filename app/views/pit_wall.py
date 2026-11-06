"""Pit Wall (LLM trace): read the conversation between your driver and the model, step by step."""

import json

import streamlit as st

from app.lib.llm_trace import call_table, find_llm_logs, latency_stats, read_llm_log, timeline
from app.lib.paths import rel
from app.lib.ui import empty_state, hide_mystery, mystery_note, page_header

page_header("Pit Wall", "An agent is a loop: prompt -> answer -> tool call -> tool result -> answer. Here you read "
            "every turn your driver had with the model, with the time and tokens each one cost. Did it use the "
            "tools? Did the Steward have to reject a patch?", chapter="Ch5-Ch6")

AVATAR = {"user": ":material/person:", "assistant": ":material/smart_toy:", "tool": ":material/build:",
          "system": ":material/settings:"}
CLIP = 700


def as_json(text: str):
    """The answer as a JSON object/list, or None (plain text)."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").removeprefix("json").strip()
    if not t.startswith(("{", "[")):
        return None
    try:
        return json.loads(t)
    except ValueError:
        return None


def log_label(p) -> str:
    try:
        n = sum(1 for line in open(p, encoding="utf-8") if line.strip())
    except OSError:
        n = 0
    return f"{rel(p).removeprefix('races/')}  ({n} call{'s' if n != 1 else ''})"


logs, n_hidden = hide_mystery(find_llm_logs())
mystery_note(n_hidden)
if not logs:
    empty_state("No LLM logs yet. They appear when an LLM driver runs with logging on - the Live page and the "
                "Grand Prix page do it for you, or from the terminal:",
                "# Git Bash / macOS / Linux:\n"
                "LLM_LOG_DIR=races/me/llm_logs python race.py drive drivers/examples/l5_llm_strategist.py\n"
                "# PowerShell:\n"
                '$env:LLM_LOG_DIR="races/me/llm_logs"; python race.py drive drivers/examples/l5_llm_strategist.py\n'
                "# cmd.exe:\n"
                "set LLM_LOG_DIR=races/me/llm_logs&& python race.py drive drivers/examples/l5_llm_strategist.py")
    st.stop()

c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
def is_real(p) -> bool:
    """Prefer a log of real model calls as the default (mock logs are canned)."""
    first = read_llm_log(p, limit=1)
    return bool(first) and "mock" not in str(first[0].get("llm", ""))


default = next((i for i, p in enumerate(logs) if is_real(p)), 0)
path = c1.selectbox("Log (one file per driver)", logs, index=default, format_func=log_label, key="pw_log")
records = read_llm_log(path)
if not records:
    st.warning("This log is empty.")
    st.stop()
phases = sorted({p for r in records for p in (r.get("phases") or [])}) or ["(none)"]
with c2:
    pick = st.multiselect("Phases", phases, default=phases, key="pw_phases",
                          help="plan = the Strategist (SESSION clock); reflect = the Engineer between laps (LAP clock).")
keep = [r for r in records if not r.get("phases") or set(r["phases"]) & set(pick)]

# ------------------------------------------------------------------------------ overview
secs = [r.get("seconds") for r in keep if r.get("seconds") is not None]
s = latency_stats(secs)
usage = [(r.get("reply") or {}).get("usage") or {} for r in keep]
def k(n: int) -> str:
    """12368 -> '12.4k' (metric values must stay short on a projector)."""
    return f"{n / 1000:.1f}k" if n >= 10_000 else str(n)


tok_in, tok_out = sum(u.get("prompt_tokens", 0) for u in usage), sum(u.get("completion_tokens", 0) for u in usage)
m = st.columns(6)
m[0].metric("LLM calls", len(keep))
m[1].metric("p50 latency", f"{s['p50']:.2f} s" if s["n"] else "-")
m[2].metric("Total wait", f"{sum(secs):.1f} s")
m[3].metric("Tokens in", k(tok_in), help=f"{tok_in} prompt tokens")
m[4].metric("Tokens out", k(tok_out), help=f"{tok_out} completion tokens (thinking included)")
m[5].metric("Tool calls", sum(len((r.get("reply") or {}).get("tool_calls") or []) for r in keep))
st.caption(f"Model: **{(keep[0].get('llm') or '?') if keep else '?'}**")
with st.expander("Call table", icon=":material/table:"):
    st.dataframe(call_table(keep), hide_index=True, width="stretch",
                 column_config={"seconds": st.column_config.NumberColumn(format="%.2f s"),
                                "time": st.column_config.TextColumn("time (UTC)")})
if any("mock" in str(r.get("llm", "")) for r in keep):
    st.caption(":material/info: These calls used the free offline **mock** model, so answers are canned and fast.")

# ------------------------------------------------------------------------------ the timeline
o1, o2, o3 = st.columns(3)
show_system = o1.toggle("Show system prompts", key="pw_sys")
show_think = o2.toggle("Show the model's thinking", value=True, key="pw_think")
steps = timeline(keep, include_system=show_system)
limit = (o3.number_input("Steps to show", 1, len(steps), min(len(steps), 60), 5,
                         key=f"pw_limit_{abs(hash(str(path))) % 10**8}_{len(steps)}") if steps else 0)

st.divider()
call_no = None
for step in steps[: int(limit)]:
    if step.call != call_no:
        call_no = step.call
        rec = keep[step.call]
        st.caption(f"**LLM call {step.call + 1}**  ·  {', '.join(step.phases) or 'no phase'}  ·  {step.label}  ·  "
                   f"{rec.get('ts', '')[11:19]} UTC")
    with st.chat_message(step.role if step.role in ("user", "assistant") else step.role, avatar=AVATAR.get(step.role)):
        if step.kind == "tool_call":
            st.markdown(f"calls tool **`{step.name}`**")
            st.code(json.dumps(step.args, indent=1) if not isinstance(step.args, str) else step.args, language="json")
        elif step.kind == "tool_result":
            st.markdown(f"result of **`{step.name or 'tool'}`**")
            st.code(step.text[:CLIP] + (" ..." if len(step.text) > CLIP else ""), language=None, wrap_lines=True)
        elif step.kind == "error":
            st.error(step.text)
        elif step.kind == "answer" and as_json(step.text) is not None:
            st.code(json.dumps(as_json(step.text), indent=1)[:4000], language="json")   # a plan / patch: pretty JSON
        else:
            text = step.text or "(empty)"
            if len(text) > CLIP:
                st.markdown(text[:CLIP] + " ...")
                with st.expander(f"full {step.kind} ({len(text)} characters)"):
                    st.text(text)
            else:
                st.markdown(text)
        if step.kind == "answer":
            tok = step.tokens or {}
            st.caption(f":material/timer: {step.seconds or 0:.2f} s  ·  {tok.get('prompt_tokens', 0)} tokens in, "
                       f"{tok.get('completion_tokens', 0)} out")
            if show_think and step.think:
                with st.expander("thinking", icon=":material/psychology:"):
                    st.text(step.think[:4000])

if steps and len(steps) > limit:
    st.caption(f"{len(steps) - int(limit)} more step(s) - raise 'Steps to show'.")
