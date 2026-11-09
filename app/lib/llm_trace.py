"""Read LLM traces (the JSONL logs agentic_gp.llm writes when LLM_LOG_DIR is set) for the Pit Wall page.

    from app.lib.llm_trace import find_llm_logs, read_llm_log, timeline, call_table
    for path in find_llm_logs():                      # races/<event>/llm_logs/*.jsonl, llm_logs/*.jsonl
        records = read_llm_log(path)                   # one record per LLM request + answer
        for step in timeline(records):                 # chat-style steps for st.chat_message
            with st.chat_message(step.role): st.markdown(step.text)

One record = one call to the model: {"ts", "llm", "label", "mode", "phases", "seconds",
"messages" (the whole conversation sent), "tools" (tool names offered), "reply" ({content,
think, tool_calls, usage}), "cached", "error"}. A tool loop (ask_with_tools) sends a growing
conversation, so timeline() shows only what is NEW in each request, then the model's answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import REPO_ROOT


def find_llm_logs(roots: list[str | Path] | None = None) -> list[Path]:
    """Every *.jsonl inside an `llm_logs` folder under races/ (and ./llm_logs), newest first."""
    roots = roots or [REPO_ROOT / "races", REPO_ROOT / "llm_logs"]
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.name == "llm_logs":
            found |= set(root.rglob("*.jsonl"))
        found |= {p for p in root.rglob("*.jsonl") if "llm_logs" in p.parts}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def read_llm_log(path: str | Path, limit: int | None = None) -> list[dict]:
    """Records of one JSONL log (broken lines are skipped)."""
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-limit:] if limit else out


@dataclass
class Step:
    """One bubble of the timeline."""

    role: str                        # "user" | "assistant" | "system" | "tool"
    kind: str                        # "prompt" | "answer" | "tool_call" | "tool_result" | "system" | "error"
    text: str
    name: str = ""                   # tool name for tool calls / results
    args: Any = None                 # tool call arguments (parsed JSON when possible)
    think: str = ""                  # the model's reasoning (when the provider returns it)
    seconds: float | None = None     # latency of the call that produced this answer
    tokens: dict = field(default_factory=dict)   # prompt_tokens / completion_tokens
    call: int = 0                    # index of the record (LLM call) this step belongs to
    phases: list = field(default_factory=list)
    label: str = ""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):                      # OpenAI "parts"
        return "\n".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def timeline(records: list[dict], include_system: bool = False) -> list[Step]:
    """Records -> chat steps (new prompt messages, tool results, then each answer + its tool calls)."""
    steps: list[Step] = []
    prev: list = []
    for i, r in enumerate(records):
        msgs = r.get("messages") or []
        k = 0
        while k < min(len(prev), len(msgs)) and prev[k] == msgs[k]:   # skip what was already shown
            k += 1
        new = msgs[k:]                            # (k == 0: a fresh conversation - show all of it)
        base = {"call": i, "phases": r.get("phases") or [], "label": r.get("label", "")}
        for m in new:
            role = m.get("role", "user")
            if role == "assistant":
                continue                          # answers come from the previous record's reply
            if role == "system" and not include_system:
                continue
            if role == "tool":
                steps.append(Step("tool", "tool_result", _content_text(m.get("content")), name=m.get("name", ""), **base))
            else:
                steps.append(Step(role, "system" if role == "system" else "prompt", _content_text(m.get("content")), **base))
        reply = r.get("reply") or {}
        usage = reply.get("usage") or {}
        if r.get("error"):
            steps.append(Step("assistant", "error", str(r["error"]), seconds=r.get("seconds"), **base))
        else:
            steps.append(Step("assistant", "answer", _content_text(reply.get("content")),
                              think=reply.get("think") or reply.get("reasoning") or "", seconds=r.get("seconds"),
                              tokens={"prompt_tokens": usage.get("prompt_tokens", 0),
                                      "completion_tokens": usage.get("completion_tokens", 0)}, **base))
            for tc in reply.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                steps.append(Step("assistant", "tool_call", "", name=fn.get("name", "?"), args=_args(fn.get("arguments")),
                                  **base))
        prev = msgs
    return steps


def call_table(records: list[dict]) -> list[dict]:
    """One row per LLM call: label, phases, seconds, tokens, tool calls, cached, error."""
    rows = []
    for i, r in enumerate(records):
        reply = r.get("reply") or {}
        usage = reply.get("usage") or {}
        rows.append({"call": i + 1, "time": r.get("ts", ""), "label": r.get("label", ""),
                     "phase": ",".join(r.get("phases") or []), "model": r.get("llm", ""), "seconds": r.get("seconds"),
                     "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                     "tool_calls": len(reply.get("tool_calls") or []), "cached": bool(r.get("cached")),
                     "error": r.get("error")})
    return rows


def latency_stats(seconds: list[float]) -> dict:
    """n, mean, p50, p95, max of a list of call latencies (seconds)."""
    xs = sorted(float(s) for s in seconds if s is not None)
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    pick = lambda q: xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 3), "p50": round(pick(0.5), 3),
            "p95": round(pick(0.95), 3), "max": round(xs[-1], 3)}
