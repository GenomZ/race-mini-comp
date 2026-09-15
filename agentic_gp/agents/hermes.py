"""The "Hermes" baseline: a zero-shot LLM driver.

Every tick the model receives the raw sensor JSON and a generic prompt, and is
asked to call the `drive` tool. There is no memory, no plan and no reflection -
which is exactly the weakness students are meant to exploit.

The model comes from `agentic_gp.llm.get_chat_model()` (Hermes 3 on Ollama by
default; switch with RACE_LLM_PROVIDER / RACE_LLM_MODEL).

Logging:
    Pass log_path="hermes.log" (or use run_hermes.py which defaults to
    results/hermes_trace.log) to write a per-tick trace in the same block
    format as my_agent_simple:

        === tick 12 (attempt 1) t=6.0s ===
        state:
          tick 12, t=6.0s, ...
        prompt-to-llm:
          [system]
          You are driving a virtual car...
          [user]
          Current sensor data:
          { ... }
        llm-reply:
          "<raw text from the model>"
        parsed-from:
          tool_call | json | regex | none
        decision:
          throttle = +0.60
          steering = -0.12
        latency_ms: 412.5
        events: []

    Useful for debugging why an LLM-driven agent crashed or strayed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import get_chat_model, describe_model
from ..physics import clamp
from ..tools import tools_by_name

HERMES_SYSTEM_PROMPT = (
    "You are driving a virtual car. Analyze the sensor data and output the optimal throttle and "
    "steering values to complete the track quickly without hitting walls.\n"
    "Throttle ranges from -1.0 (full brake) to 1.0 (full acceleration). Steering ranges from -1.0 "
    "(full left) to 1.0 (full right). A negative angle_to_next_waypoint means the waypoint is to "
    "your left, positive means it is to your right. Distances are in track units; the track is "
    "about 18 units wide. Hitting a wall costs a 5 second penalty and stops the car.\n"
    "Respond by calling the `drive` tool exactly once with your chosen throttle and steering."
)

_NUM = r"(-?\d+(?:\.\d+)?)"

LOGGER_NAME = "agentic_gp.agents.hermes"
logger = logging.getLogger(LOGGER_NAME)


def extract_controls(ai: AIMessage) -> tuple[float, float] | None:
    """Pull (throttle, steering) out of a tool call or, failing that, the text."""
    for call in getattr(ai, "tool_calls", None) or []:
        if call.get("name") == "drive":
            args = call.get("args") or {}
            try:
                return float(args["throttle"]), float(args["steering"])
            except (KeyError, TypeError, ValueError):
                continue
    text = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            return float(obj["throttle"]), float(obj["steering"])
    except (ValueError, KeyError, TypeError):
        pass
    t = re.search(r"throttle\D{0,12}" + _NUM, text, re.I)
    s = re.search(r"steer(?:ing)?\D{0,12}" + _NUM, text, re.I)
    if t and s:
        return float(t.group(1)), float(s.group(1))
    return None


def _parse_source(ai, parsed: tuple[float, float] | None) -> str:
    """How did we get the (throttle, steering) tuple out of the AI message?"""
    if parsed is None or ai is None:
        return "none"
    if getattr(ai, "tool_calls", None):
        for call in ai.tool_calls:
            if call.get("name") == "drive":
                return "tool_call"
    text = ai.content if isinstance(ai.content, str) else json.dumps(ai.content)
    if re.search(r"\{.*\}", text, re.S):
        return "json"
    return "regex"


def setup_log_file(log_path: str | None) -> logging.Handler | None:
    """Attach a FileHandler to the Hermes module logger. See my_agent_simple."""
    if not log_path:
        return None
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def log_tick(
    obs: dict,
    throttle: float,
    steering: float,
    parsed_from: str,
    raw_reply: str,
    latency_ms: float,
    events: list[str],
    attempt_idx: int,
    llm_error: str | None = None,
) -> None:
    """Write one structured per-tick record to the Hermes log."""
    lines = [
        f"=== tick {obs['tick']} (attempt {attempt_idx}) t={obs['sim_time']:.1f}s ===",
        "state:",
        f"  tick {obs['tick']}, t={obs['sim_time']:.1f}s, speed={obs['speed']:.1f}, "
        f"gate {obs['next_waypoint_index']} (passed {obs['waypoints_passed']}/{obs['waypoints_total']}), "
        f"next waypoint {obs['distance_to_next_waypoint']:.1f}m at {obs['angle_to_next_waypoint']:+.1f} deg",
        "prompt-to-llm:",
        "  [system]",
        f"  {HERMES_SYSTEM_PROMPT.splitlines()[0]} ... ({len(HERMES_SYSTEM_PROMPT)} chars)",
        "  [user]",
        "  Current sensor data:",
    ]
    # Truncate the user payload if it's huge (full obs JSON can be ~1 KB).
    user_payload = json.dumps({k: obs[k] for k in sorted(obs.keys()) if k != "position"})
    if len(user_payload) > 800:
        user_payload = user_payload[:800] + "..."
    lines.append(f"  {user_payload}")
    if llm_error is not None:
        lines += [
            "llm-error:",
            f"  {llm_error!r}",
        ]
    else:
        lines += [
            "llm-reply:",
            f"  {raw_reply[:1200]!r}",
        ]
    lines += [
        "parsed-from:",
        f"  {parsed_from}",
        "decision:",
        f"  throttle = {throttle:+.2f}",
        f"  steering = {steering:+.2f}",
        f"latency_ms: {latency_ms:.1f}",
        f"events: {events}",
        "",
    ]
    logger.info("\n".join(lines))


def run(env, tools, attempts: int = 1, verbose: bool = True, history: int = 0,
        model=None, log_path: str | None = None, **_) -> None:
    """Zero-shot LLM driver with per-tick logging.

    Args:
        env, tools, attempts, verbose, history, model: see README.
        log_path: if given, every tick's prompt + reply + decision is
                  appended to this file in the block format documented at
                  the top of the module. None (default) skips file
                  logging; the terminal trace still appears.
    """
    by = tools_by_name(tools)
    llm = get_chat_model(model)
    try:
        driver = llm.bind_tools([by["drive"]])
    except NotImplementedError:
        driver = llm
    print(f"[hermes] model={describe_model() if model is None else model} history={history}")
    if log_path:
        print(f"[hermes] logging per-tick trace to {log_path}")

    log_handler = setup_log_file(log_path)
    if log_handler is not None and verbose:
        # Stream to stderr so a student watching the terminal can follow.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)

    try:
        for attempt_idx in range(attempts):
            obs = json.loads(by["reset_environment"].invoke({}))
            past: deque = deque(maxlen=2 * max(history, 0))
            n_calls = 0
            n_parse_fail = 0
            n_errors = 0
            t_llm = 0.0
            attempt_label = attempt_idx + 1
            while not obs["attempt_over"]:
                human = HumanMessage(content="Current sensor data:\n" + json.dumps(obs))
                messages = [SystemMessage(content=HERMES_SYSTEM_PROMPT), *past, human]
                t0 = time.perf_counter()
                controls = None
                llm_error: str | None = None
                ai = None
                try:
                    ai = driver.invoke(messages)
                    controls = extract_controls(ai)
                except Exception as exc:
                    n_errors += 1
                    llm_error = repr(exc)
                    if verbose:
                        print(f"[hermes] LLM error: {exc}")
                latency_ms = (time.perf_counter() - t0) * 1000.0
                t_llm += time.perf_counter() - t0
                n_calls += 1
                if controls is None:
                    n_parse_fail += 1
                    controls = (0.0, 0.0)
                throttle, steering = (clamp(controls[0], -1, 1), clamp(controls[1], -1, 1))
                # Capture the raw reply text BEFORE we drive (drive() advances
                # time and we want the trace to show what the model said about
                # the state we saw).
                raw_reply = (
                    ai.content if ai is not None and isinstance(ai.content, str)
                    else json.dumps(ai.content) if ai is not None else ""
                )
                obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
                if history > 0:
                    past.append(human)
                    past.append(AIMessage(content=f"drive(throttle={throttle}, steering={steering})"))
                if verbose:
                    ev = f" -> {obs['events']}" if obs["events"] else ""
                    print(f"tick {obs['tick']:3d} t={obs['sim_time']:6.1f}s v={obs['speed']:5.1f} "
                          f"thr={throttle:+.2f} str={steering:+.2f}{ev}")
                # Per-tick log entry.
                log_tick(
                    obs, throttle, steering,
                    parsed_from=_parse_source(ai, controls),
                    raw_reply=raw_reply,
                    latency_ms=latency_ms,
                    events=list(obs.get("events", [])),
                    attempt_idx=attempt_label,
                    llm_error=llm_error,
                )
            res = env.attempts[-1]
            print(f"[hermes] attempt {res.attempt}: {'lap %.2fs' % res.lap_time if res.completed else 'DNF'} "
                  f"crashes={res.crashes} waypoints={res.waypoints_passed}/{env.track.n_gates} "
                  f"llm_calls={n_calls} parse_failures={n_parse_fail} errors={n_errors} "
                  f"avg_llm_latency={t_llm / max(n_calls, 1):.2f}s")
    finally:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
