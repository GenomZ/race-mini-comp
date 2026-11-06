"""Did this run's LLM calls work? Read the run's LLM log (JSONL) and count the failed calls.

    from app.lib.llm_health import llm_failures
    bad = llm_failures("races/live/llm_logs", since_utc=run.started_utc, label="Strategist")
    if bad["failed"]:
        st.warning(f"{bad['failed']} of {bad['calls']} LLM calls failed: {bad['first_error']}")

An LLM driver that loses its connection (wrong key, no network, rate limit) usually falls back to
its physics plan and finishes the lap anyway - so the race result alone hides the problem. The log
written by agentic_gp.llm (LLM_LOG_DIR) has one record per call with an "error" field.
"""

from __future__ import annotations

import json
from pathlib import Path


def llm_failures(log_dir: str | Path, *, since_utc: str | None = None, label: str | None = None) -> dict:
    """{"calls", "failed", "first_error", "errors": [..up to 5..], "files"} for the records at/after since_utc.

    since_utc is "YYYY-MM-DDTHH:MM:SSZ" (the log's own "ts" format, so string order = time order).
    label: only records of this log label (the driver name) when given.
    """
    out = {"calls": 0, "failed": 0, "first_error": None, "errors": [], "files": []}
    folder = Path(log_dir)
    if not folder.is_dir():
        return out
    for f in sorted(folder.glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        used = False
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if since_utc and str(rec.get("ts", "")) < since_utc:
                continue
            if label and rec.get("label") not in (None, "", label) and not str(rec.get("label", "")).startswith(label):
                continue
            used = True
            out["calls"] += 1
            err = rec.get("error")
            if err:
                out["failed"] += 1
                if len(out["errors"]) < 5:
                    out["errors"].append(str(err)[:300])
                if out["first_error"] is None:
                    out["first_error"] = str(err)[:300]
        if used:
            out["files"].append(str(f))
    return out
