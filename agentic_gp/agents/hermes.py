"""The "Hermes" baseline: a zero-shot LLM driver.

Every tick the model receives the raw sensor JSON and a generic prompt, and is
asked to call the `drive` tool. There is no memory, no plan and no reflection -
which is exactly the weakness students are meant to exploit.

The model comes from `agentic_gp.llm.get_chat_model()` (Hermes 3 on Ollama by
default; switch with RACE_LLM_PROVIDER / RACE_LLM_MODEL).
"""

from __future__ import annotations

import json
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


def run(env, tools, attempts: int = 1, verbose: bool = True, history: int = 0, model=None, **_) -> None:
    by = tools_by_name(tools)
    llm = get_chat_model(model)
    try:
        driver = llm.bind_tools([by["drive"]])
    except NotImplementedError:
        driver = llm
    print(f"[hermes] model={describe_model() if model is None else model} history={history}")

    for _ in range(attempts):
        obs = json.loads(by["reset_environment"].invoke({}))
        past: deque = deque(maxlen=2 * max(history, 0))
        n_calls = 0
        n_parse_fail = 0
        n_errors = 0
        t_llm = 0.0
        while not obs["attempt_over"]:
            human = HumanMessage(content="Current sensor data:\n" + json.dumps(obs))
            messages = [SystemMessage(content=HERMES_SYSTEM_PROMPT), *past, human]
            t0 = time.perf_counter()
            controls = None
            try:
                ai = driver.invoke(messages)
                controls = extract_controls(ai)
            except Exception as exc:  # network / provider errors: coast this tick
                n_errors += 1
                ai = AIMessage(content=f"[error: {exc}]")
                if verbose:
                    print(f"[hermes] LLM error: {exc}")
            t_llm += time.perf_counter() - t0
            n_calls += 1
            if controls is None:
                n_parse_fail += 1
                controls = (0.0, 0.0)
            throttle, steering = (clamp(controls[0], -1, 1), clamp(controls[1], -1, 1))
            obs = json.loads(by["drive"].invoke({"throttle": throttle, "steering": steering}))
            if history > 0:
                past.append(human)
                past.append(AIMessage(content=f"drive(throttle={throttle}, steering={steering})"))
            if verbose:
                ev = f" -> {obs['events']}" if obs["events"] else ""
                print(f"tick {obs['tick']:3d} t={obs['sim_time']:6.1f}s v={obs['speed']:5.1f} "
                      f"thr={throttle:+.2f} str={steering:+.2f}{ev}")
        res = env.attempts[-1]
        print(f"[hermes] attempt {res.attempt}: {'lap %.2fs' % res.lap_time if res.completed else 'DNF'} "
              f"crashes={res.crashes} waypoints={res.waypoints_passed}/{env.track.n_gates} "
              f"llm_calls={n_calls} parse_failures={n_parse_fail} errors={n_errors} "
              f"avg_llm_latency={t_llm / max(n_calls, 1):.2f}s")
