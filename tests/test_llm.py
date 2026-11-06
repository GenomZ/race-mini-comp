"""Tests for agentic_gp.llm: offline only (mock provider + a fake MiniMax-shaped client).

Real MiniMax checks are opt-in:  AGP_LIVE_LLM=1 python -m pytest -q -m llm tests/test_llm.py
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Literal, Optional

import pytest

from agentic_gp import llm


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Mock provider, no .env, no keys, no sleeping, fresh stats/caches for every test."""
    monkeypatch.setenv("LLM_NO_DOTENV", "1")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_QUIET", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("MINIMAX_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "LLM_MODEL", "LLM_EFFORT",
                "LLM_TIMEOUT", "LLM_LOG_DIR", "LLM_DAILY_CAP", "LLM_CASSETTE", "LLM_RECORD", "LLM_MAX_RPM",
                "MINIMAX_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    monkeypatch.setattr(llm, "_CLIENTS", {})
    llm.reset_llm_stats()
    llm.set_mock_responder(None)
    llm.set_llm_semaphore(None)
    yield
    llm.set_mock_responder(None)
    llm.set_llm_semaphore(None)


def completion(content="", *, reasoning=None, tool_calls=None, finish="stop", prompt_tokens=10, completion_tokens=5):
    """A MiniMax-shaped Chat Completions response (extra fields included, as the real API sends them)."""
    from openai.types.chat import ChatCompletion

    message = {"role": "assistant", "content": content, "name": "MiniMax AI", "audio_content": ""}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [
            {"id": f"call_function_abc_{i + 1}", "type": "function", "index": i,
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)
        ]
    return ChatCompletion.model_validate({
        "id": "resp1", "object": "chat.completion", "created": 1, "model": "MiniMax-M3",
        "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
        "base_resp": {"status_code": 0, "status_msg": ""},
    })


def api_error(status: int, message: str):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.minimax.io/v1/chat/completions")
    cls = {401: openai.AuthenticationError, 429: openai.RateLimitError, 400: openai.BadRequestError}.get(
        status, openai.InternalServerError)
    return cls(message, response=httpx.Response(status, request=request),
               body={"type": "error", "message": message, "http_code": str(status)})


class FakeClient:
    """Stands in for openai.OpenAI: records every request, replays scripted responses/exceptions."""

    def __init__(self, *script):
        self.script = list(script)
        self.requests: list[dict] = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs, default=str)))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def minimax(monkeypatch):
    """Switch to the minimax provider with a fake key; returns a function that installs a FakeClient."""
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-not-real")

    def install(*script) -> FakeClient:
        fake = FakeClient(*script)
        monkeypatch.setattr(llm, "_client", lambda cfg: fake)
        return fake

    return install


# ------------------------------------------------------------------ .env parsing + config
def test_parse_env_handles_comments_quotes_and_export():
    text = """
    # comment
    LLM_PROVIDER=minimax
    export MINIMAX_API_KEY = "abc def"
    LLM_MODEL='MiniMax-M3.1-Flash'
    LLM_EFFORT=low   # trailing comment
    EMPTY=
    not a pair
    """
    env = llm.parse_env(text)
    assert env == {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "abc def",
                   "LLM_MODEL": "MiniMax-M3.1-Flash", "LLM_EFFORT": "low", "EMPTY": ""}


def test_load_env_real_environment_wins(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("AGP_TEST_A=from_file\nAGP_TEST_B=from_file\n", encoding="utf-8")
    monkeypatch.setenv("AGP_TEST_A", "real")
    monkeypatch.delenv("AGP_TEST_B", raising=False)
    loaded = llm.load_env(path)
    assert os.environ["AGP_TEST_A"] == "real" and os.environ["AGP_TEST_B"] == "from_file"
    assert loaded == {"AGP_TEST_B": "from_file"}
    monkeypatch.delenv("AGP_TEST_B")


def test_config_defaults_describe_and_availability(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    cfg = llm.get_config()
    assert (cfg.model, cfg.effort, cfg.base_url) == ("MiniMax-M3", "low", "https://api.minimax.io/v1")
    assert llm.describe_llm() == "minimax:MiniMax-M3"
    assert not llm.llm_available()                       # no key
    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    assert llm.llm_available()
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert llm.llm_available() and llm.describe_llm() == "mock:mock"
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(llm.LLMError):
        llm.get_config()


def test_normalize_effort():
    assert llm.normalize_effort("OFF") == "none" and llm.normalize_effort(None) == "low"
    assert llm.normalize_effort("medium") == "medium"
    with pytest.raises(llm.LLMError):
        llm.normalize_effort("turbo")


# ------------------------------------------------------------------ text helpers
@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Sure! Here is the plan:\n```json\n{"a": 1}\n```\nGood luck.',
    '<think>maybe {"a": 2}?</think>{"a": 1}',
    'prose {"a": 1,} trailing comma',
    'first {not json} then {"a": 1}',
])
def test_extract_json_is_robust(text):
    assert llm.extract_json(text) == {"a": 1}


def test_extract_json_nested_and_failure():
    assert llm.extract_json('x {"cap": 30, "corners": [{"id": "T1"}]} y') == {"cap": 30, "corners": [{"id": "T1"}]}
    with pytest.raises(ValueError):
        llm.extract_json("no json here")


def test_strip_think_variants():
    assert llm.strip_think("<think>a</think> answer") == "answer"
    assert llm.strip_think("thinking without start</think>answer") == "answer"
    assert llm.strip_think("answer<think>cut off") == "answer"
    assert llm.strip_think(None) == ""


def test_minimax_code_from_message_and_exception():
    assert llm.minimax_code("login fail: carry the key (1004)") == 1004
    assert llm.minimax_code(api_error(429, "5-hour usage limit reached (2056)")) == 2056
    assert llm.minimax_code("plain failure") is None


# ------------------------------------------------------------------ function_schema
def corner_speed(radius: float, grip: float = 20.0, unit: Literal["u/s", "km/h"] = "u/s",
                 tags: Optional[list[str]] = None) -> float:
    """Fastest speed through a corner.

    Args:
        radius: corner radius in units
        grip (float): lateral grip in u/s^2
    Returns:
        the speed
    """
    return (grip * radius) ** 0.5


def test_function_schema_from_hints_and_docstring():
    schema = llm.function_schema(corner_speed)
    fn = schema["function"]
    assert schema["type"] == "function" and fn["name"] == "corner_speed"
    assert fn["description"] == "Fastest speed through a corner."
    props = fn["parameters"]["properties"]
    assert props["radius"] == {"type": "number", "description": "corner radius in units"}
    assert props["grip"]["default"] == 20.0 and props["grip"]["description"] == "lateral grip in u/s^2"
    assert props["unit"] == {"type": "string", "enum": ["u/s", "km/h"], "default": "u/s"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert fn["parameters"]["required"] == ["radius"]


# ------------------------------------------------------------------ mock provider
def test_mock_chat_and_chat_json_defaults():
    assert llm.chat("hi") == llm.MOCK_TEXT
    assert llm.chat_json("plan please") == {}
    assert llm.llm_stats()["calls"] == 2


def test_mock_responder_scripts_replies():
    seen = []
    llm.set_mock_responder(lambda messages, tools: seen.append(messages) or {"cap": 31})
    assert llm.chat_json("plan", system="You are a strategist") == {"cap": 31}
    assert seen[0][0]["role"] == "system" and llm.JSON_RULE in seen[0][0]["content"]


def test_chat_json_retries_once_with_nudge_then_fails():
    replies = iter(["not json at all", '{"ok": true}'])
    llm.set_mock_responder(lambda m, t: next(replies))
    assert llm.chat_json("x") == {"ok": True}
    llm.set_mock_responder(lambda m, t: "still prose")
    with pytest.raises(llm.LLMError):
        llm.chat_json("x")


def test_ask_with_tools_runs_parallel_calls_and_reports_errors():
    def braking_distance(v_from: float, v_to: float) -> float:
        """Distance to brake."""
        return (v_from ** 2 - v_to ** 2) / 50

    def explode() -> str:
        """Always fails."""
        raise RuntimeError("boom")

    def responder(messages, tools):
        if messages[-1]["role"] != "tool":
            return [{"tool": "braking_distance", "args": {"v_from": 40, "v_to": 10}},
                    {"tool": "explode", "args": {}}, {"tool": "missing", "args": {}}]
        return "Brake 30 units early."

    llm.set_mock_responder(responder)
    run = llm.ask_with_tools("When to brake?", [braking_distance, explode], system="Be brief.")
    assert run.final_text == "Brake 30 units early." and run.rounds == 2
    assert [c["name"] for c in run.calls] == ["braking_distance", "explode", "missing"]
    assert run.calls[0]["result"] == 30.0 and run.calls[0]["args"] == {"v_from": 40, "v_to": 10}
    assert run.calls[1]["result"].startswith("error: RuntimeError")
    assert "unknown tool" in run.calls[2]["result"]
    tool_msgs = [m for m in run.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 3 and len({m["tool_call_id"] for m in tool_msgs}) == 3
    assert run.messages[0]["role"] == "system"


def test_ask_with_tools_stops_after_max_rounds():
    def ping() -> str:
        """Ping."""
        return "pong"

    llm.set_mock_responder(lambda m, t: {"tool": "ping", "args": {}})
    run = llm.ask_with_tools("loop forever", [ping], max_rounds=3)
    assert run.rounds == 3 and len(run.calls) == 3


def test_chat_json_with_tools():
    def corner(radius: float) -> float:
        """v_max for a radius."""
        return (20 * radius) ** 0.5

    llm.set_mock_responder(lambda m, t: {"tool": "corner", "args": {"radius": 45}}
                           if m[-1]["role"] != "tool" else '{"T1": 30}')
    assert llm.chat_json("plan", tools=[corner]) == {"T1": 30}


def test_smoke_test_offline():
    rows = llm.llm_smoke_test()
    assert [r["step"] for r in rows] == ["chat", "chat_json", "ask_with_tools"]
    assert all(r["ok"] for r in rows), rows


# ------------------------------------------------------------------ policy
def test_open_mode_allows_everything():
    with llm.llm_policy(phase="drive"):
        assert llm.chat("x") == llm.MOCK_TEXT
    assert llm.current_llm_policy()["mode"] == "open"


def test_official_mode_forbids_drive_and_race_and_logs(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    with llm.llm_policy(phase="track", mode="official", label="ada"):
        with llm.llm_policy(phase="drive"):
            assert not llm.llm_allowed()
            with pytest.raises(llm.LLMNotAllowed):
                llm.chat("steer?")
        with llm.llm_policy(phase="race"), llm.llm_policy(phase="reflect"):
            with pytest.raises(llm.LLMNotAllowed):
                llm.chat("x")
    record = json.loads((tmp_path / "logs" / "ada.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "not allowed" in record["error"] and record["mode"] == "official"


def test_official_budgets_per_scope_and_per_track():
    track = llm.llm_policy(phase="track", mode="official")
    with track:
        with llm.llm_policy(phase="plan"):
            llm.chat("1")
            llm.chat("2")
            assert llm.current_llm_policy()["calls_left"] == 0
            with pytest.raises(llm.LLMBudgetExceeded):
                llm.chat("3")
    for _ in range(3):                                   # re-entering the track scope keeps counting
        with track, llm.llm_policy(phase="practice"), llm.llm_policy(phase="reflect"):
            llm.chat("a")
            llm.chat("b")
    assert track.usage["calls"] == 8
    with track, llm.llm_policy(phase="reflect"):
        with pytest.raises(llm.LLMBudgetExceeded):
            llm.chat("one too many")


def test_explicit_budget_and_usage():
    with llm.llm_policy(budget=1) as scope:
        llm.chat("x")
        with pytest.raises(llm.LLMBudgetExceeded):
            llm.chat("y")
    assert scope.usage["calls"] == 1
    assert isinstance(llm.LLMBudgetExceeded("x"), llm.LLMError)


def test_policy_set_in_a_thread_does_not_leak():
    seen = {}

    def worker():                                       # e.g. one grand-prix driver thread
        with llm.llm_policy(phase="drive", mode="official"):
            seen["inside"] = llm.llm_allowed()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["inside"] is False and llm.llm_allowed()


# ------------------------------------------------------------------ MiniMax requests (fake client)
def test_minimax_m3_effort_mapping(minimax):
    fake = minimax(completion("hi"), completion("hi"))
    llm.chat("a", effort="off")
    llm.chat("b")
    fast, normal = fake.requests
    assert fast["extra_body"] == {"reasoning_split": True, "thinking": {"type": "disabled"}}
    assert normal["extra_body"] == {"reasoning_split": True}
    for req in (fast, normal):
        assert req["model"] == "MiniMax-M3" and req["max_completion_tokens"] == 4096 and req["timeout"] == 30.0
        assert "reasoning_effort" not in req and "temperature" not in req and "response_format" not in req


def test_minimax_m31_uses_reasoning_effort(minimax, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "MiniMax-M3.1-Flash")
    fake = minimax(completion("hi"), completion("hi"))
    llm.chat("a")
    llm.chat("b", effort="none")                        # Flash cannot switch thinking off: use low
    assert [r["reasoning_effort"] for r in fake.requests] == ["low", "low"]
    assert "extra_body" not in fake.requests[0]


def test_planning_phase_gets_more_tokens_and_time(minimax):
    fake = minimax(completion("plan"))
    with llm.llm_policy(phase="plan"):
        llm.chat("plan the race")
    assert fake.requests[0]["max_completion_tokens"] == 16384 and fake.requests[0]["timeout"] == 120.0


def test_reasoning_is_separated_and_kept_in_history(minimax):
    minimax(completion("<think>inline thoughts</think>Brake at T3.", reasoning="split thoughts"))
    reply = llm.complete([{"role": "user", "content": "x"}])
    assert reply.text == "Brake at T3."
    assert "split thoughts" in llm.last_reasoning() and "inline thoughts" in llm.last_reasoning()
    assert reply.message["reasoning_content"] == "split thoughts"
    assert reply.message["content"].startswith("<think>")            # raw content stays in history
    assert llm.llm_stats()["prompt_tokens"] == 10 and llm.llm_stats()["completion_tokens"] == 5


def test_tool_loop_against_minimax_shaped_responses(minimax):
    def corner_speed(radius: float) -> float:
        """v_max."""
        return round((20 * radius) ** 0.5, 1)

    def braking_distance(v_from: float, v_to: float) -> float:
        """Braking distance."""
        return (v_from ** 2 - v_to ** 2) / 50

    fake = minimax(
        completion("I'll check.", reasoning="need both", finish="tool_calls",
                   tool_calls=[("corner_speed", {"radius": 45}), ("braking_distance", {"v_from": 40, "v_to": 30})]),
        completion("Take T3 at 30, brake 14 units early."),
    )
    run = llm.ask_with_tools("T3?", [corner_speed, braking_distance])
    assert run.final_text.startswith("Take T3 at 30") and run.rounds == 2
    assert [c["result"] for c in run.calls] == [30.0, 14.0]
    first, second = fake.requests
    assert first["tool_choice"] == "auto" and [t["function"]["name"] for t in first["tools"]] == \
        ["corner_speed", "braking_distance"]
    assistant = second["messages"][1]
    assert assistant["reasoning_content"] == "need both" and len(assistant["tool_calls"]) == 2
    tools = second["messages"][2:]
    assert [m["tool_call_id"] for m in tools] == ["call_function_abc_1", "call_function_abc_2"]


def test_finish_reason_length_retries_with_double_budget(minimax):
    fake = minimax(completion("", reasoning="long thinking...", finish="length"), completion('{"a": 1}'))
    assert llm.chat_json("x", max_tokens=1000) == {"a": 1}
    assert [r["max_completion_tokens"] for r in fake.requests] == [1000, 2000]


def test_fatal_minimax_code_is_not_retried(minimax):
    fake = minimax(api_error(429, "5-hour usage limit reached (2056)"), completion("never"))
    with pytest.raises(llm.LLMError) as info:
        llm.chat("x")
    assert info.value.code == 2056 and len(fake.requests) == 1
    assert llm.llm_stats()["errors"] == 1


def test_transient_errors_are_retried(minimax):
    fake = minimax(api_error(429, "rate growth limit (2045)"), api_error(500, "busy"), completion("ok"))
    assert llm.chat("x") == "ok" and len(fake.requests) == 3


def test_retries_give_up_after_max(minimax):
    minimax(api_error(500, "busy"), api_error(500, "busy"), api_error(500, "busy"))
    with pytest.raises(llm.LLMError) as info:
        llm.chat("x")
    assert info.value.status == 500


def test_auth_error_is_friendly(minimax):
    minimax(api_error(401, "login fail: Please carry the API secret key (1004)"))
    with pytest.raises(llm.LLMError, match="MINIMAX_API_KEY"):
        llm.chat("x")


def test_missing_key_is_friendly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    with pytest.raises(llm.LLMError, match="Set MINIMAX_API_KEY in .env"):
        llm.chat("x")


def test_empty_choices_with_base_resp_error(minimax):
    from openai.types.chat import ChatCompletion

    bad = ChatCompletion.model_validate({"id": "r", "object": "chat.completion", "created": 1, "model": "m",
                                         "choices": [], "base_resp": {"status_code": 1008,
                                                                      "status_msg": "insufficient balance"}})
    minimax(bad)
    with pytest.raises(llm.LLMError) as info:
        llm.chat("x")
    assert info.value.code == 1008


def test_daily_cap(minimax, monkeypatch):
    monkeypatch.setenv("LLM_DAILY_CAP", "2")
    minimax(completion("1"), completion("2"), completion("3"))
    llm.chat("a")
    llm.chat("b")
    with pytest.raises(llm.LLMBudgetExceeded, match="LLM_DAILY_CAP"):
        llm.chat("c")


def test_semaphore_wraps_every_request(minimax):
    class CountingSemaphore:
        def __init__(self):
            self.acquired = self.released = 0

        def acquire(self):
            self.acquired += 1

        def release(self):
            self.released += 1

    sem = CountingSemaphore()
    llm.set_llm_semaphore(sem)
    minimax(api_error(500, "busy"), completion("ok"))
    llm.chat("x")
    assert sem.acquired == sem.released == 2


def test_disk_cache_makes_reruns_free(minimax):
    fake = minimax(completion("cached answer"))
    assert llm.chat("same question", cache=True) == "cached answer"
    assert llm.chat("same question", cache=True) == "cached answer"
    assert len(fake.requests) == 1 and llm.llm_stats()["cache_hits"] == 1


def test_log_dir_writes_jsonl_without_the_key(minimax, monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path))
    minimax(completion("logged", reasoning="why"))
    with llm.llm_policy(phase="plan", label="max"):
        llm.chat("hello")
    text = (tmp_path / "max.jsonl").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["llm"] == "minimax:MiniMax-M3" and record["phases"] == ["plan"]
    assert record["reply"]["content"] == "logged" and record["messages"][-1]["content"] == "hello"
    assert "test-key-not-real" not in text


def test_record_then_replay_cassette(minimax, monkeypatch, tmp_path):
    tape = tmp_path / "tape.jsonl"
    monkeypatch.setenv("LLM_RECORD", str(tape))
    minimax(completion("first", reasoning="r1"), completion("second"))
    llm.chat("q1")
    llm.chat("q2")
    monkeypatch.delenv("LLM_RECORD")
    monkeypatch.setenv("LLM_PROVIDER", "replay")
    monkeypatch.setenv("LLM_CASSETTE", str(tape))
    monkeypatch.setattr(llm, "_CASSETTES", {})
    assert llm.llm_available()
    assert llm.chat("q2") == "second"            # matched by request hash
    assert llm.chat("unknown") == "first"        # fallback: next unused entry
    with pytest.raises(llm.LLMError, match="exhausted"):
        llm.chat("q3")


def test_stats_reset():
    llm.chat("x")
    assert llm.llm_stats()["calls"] == 1
    llm.reset_llm_stats()
    assert llm.llm_stats() == {"calls": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0,
                               "seconds": 0, "cache_hits": 0}


# ------------------------------------------------------------------ LangChain adapter
def test_langchain_adapter_uses_the_same_pipeline():
    pytest.importorskip("langchain_core")

    def corner(radius: float) -> float:
        """v_max for a radius."""
        return (20 * radius) ** 0.5

    model = llm.get_langchain_model()
    assert model.invoke("hi").content == llm.MOCK_TEXT
    llm.set_mock_responder(lambda m, t: {"tool": "corner", "args": {"radius": 20}} if t else "no tools")
    msg = model.bind_tools([corner]).invoke("T1?")
    assert msg.tool_calls[0]["name"] == "corner" and msg.tool_calls[0]["args"] == {"radius": 20}
    with llm.llm_policy(phase="drive", mode="official"):
        with pytest.raises(llm.LLMNotAllowed):
            model.invoke("steer")


# ------------------------------------------------------------------ live (opt-in)
@pytest.mark.llm
@pytest.mark.skipif(os.getenv("AGP_LIVE_LLM") != "1", reason="live MiniMax test: set AGP_LIVE_LLM=1")
def test_live_minimax_smoke(monkeypatch):
    path = llm.REPO_ROOT / ".env"
    for key, value in (llm.parse_env(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}).items():
        monkeypatch.setenv(key, value)                  # restored after the test
    if os.getenv("LLM_PROVIDER", "minimax") == "mock":
        monkeypatch.setenv("LLM_PROVIDER", "minimax")
    if not llm.llm_available():
        pytest.skip("no MiniMax key in .env")
    rows = llm.llm_smoke_test(effort="none")
    assert all(r["ok"] for r in rows), rows


# ------------------------------------------------------------------ review fixes (policy, deadline, tools)
def test_inner_scope_cannot_loosen_official_rules():
    with llm.llm_policy(phase="track", mode="official"), llm.llm_policy(phase="drive"):
        with llm.llm_policy(mode="open"):                     # a driver trying to switch the rules off
            assert llm.current_llm_policy()["mode"] == "official"
            with pytest.raises(llm.LLMNotAllowed):
                llm.chat("nested open")
    with llm.llm_policy(phase="track", mode="official"):      # official: only plan()/reflect() may call
        with pytest.raises(llm.LLMNotAllowed, match="only allowed inside plan"):
            llm.chat("between calls")


def test_threads_without_a_policy_use_the_runner_fallback():
    track = llm.llm_policy(phase="track", mode="official")
    plan = llm.llm_policy(phase="plan")
    results = []

    def worker():                                             # a thread does not inherit the ContextVar
        try:
            llm.chat("from a thread")
            results.append("ok")
        except llm.LLMError as exc:
            results.append(type(exc).__name__)

    prev = llm._swap_fallback((track, llm.llm_policy(phase="between")))
    try:
        t = threading.Thread(target=worker)
        t.start()
        t.join()
        llm._swap_fallback((track, plan))                     # while plan() runs: counts against plan
        for _ in range(3):
            t = threading.Thread(target=worker)
            t.start()
            t.join()
    finally:
        llm._swap_fallback(prev)
    assert results == ["LLMNotAllowed", "ok", "ok", "LLMBudgetExceeded"]
    assert plan.usage["calls"] == 2 and track.usage["calls"] == 2
    assert llm.llm_allowed()                                   # restored: open again


def _call_in(how: str) -> str:
    """One llm.chat from a context the runner did not set up, inside the driver's OWN policy scope."""
    import contextvars

    out: list[str] = []

    def own_scope():
        try:
            with llm.llm_policy(phase="plan"):                # a driver trying to declare itself "in plan"
                llm.chat("sneaky")
            out.append("ok")
        except llm.LLMError as exc:
            out.append(type(exc).__name__)

    if how == "thread":
        t = threading.Thread(target=own_scope)
        t.start()
        t.join()
    else:
        contextvars.Context().run(own_scope)                  # no thread: a fresh, empty context
    return out[0]


@pytest.mark.parametrize("how", ["thread", "fresh_context"])
def test_own_scope_cannot_escape_an_official_session(how):
    """Review round 3: an own llm_policy scope in a thread / fresh Context used to hide the session's rules."""
    track = llm.llm_policy(phase="track", mode="official")
    between = (track, llm.llm_policy(phase="between"))
    token = llm._open_session(track, between)
    try:
        assert _call_in(how) == "LLMNotAllowed"                       # between calls
        race = (track, llm.llm_policy(phase="race"), llm.llm_policy(phase="drive"))
        llm._set_session(token, race)
        assert _call_in(how) == "LLMNotAllowed"                       # during the race
        plan = llm.llm_policy(phase="plan")
        llm._set_session(token, (track, llm.llm_policy(phase="session"), plan))
        assert [_call_in(how) for _ in range(3)] == ["ok", "ok", "LLMBudgetExceeded"]   # counted against plan
        assert plan.usage["calls"] == 2 and track.usage["calls"] == 2
        with track, llm.llm_policy(phase="race"):                     # the runner's own context: its rules
            with pytest.raises(llm.LLMNotAllowed, match="during race"):
                llm.chat("x")
    finally:
        llm._close_session(token)
    assert _call_in(how) == "ok" and llm.llm_allowed()               # session over: open again


def test_overlapping_official_sessions_do_not_leave_the_process_locked():
    """Review round 3: A then B begin, A then B end -> the process must be open again (it stayed refused)."""
    a_root, b_root = llm.llm_policy(phase="track", mode="official"), llm.llm_policy(phase="track", mode="official")
    a = llm._open_session(a_root, (a_root, llm.llm_policy(phase="between")))
    b = llm._open_session(b_root, (b_root, llm.llm_policy(phase="between")))
    try:
        a_plan = llm.llm_policy(phase="plan")
        llm._set_session(a, (a_root, a_plan))                         # A's runner is inside plan()
        with pytest.raises(llm.LLMNotAllowed, match="several official sessions"):
            llm.chat("whose call is this?")                           # cannot be told apart: refused
        with a_root, a_plan:                                          # A's runner context: A's rules apply
            assert llm.chat("plan for A") == llm.MOCK_TEXT
            with llm.llm_policy(phase="reflect"):                     # nested own scope: still fine
                assert llm.chat("again") == llm.MOCK_TEXT
        with a_root, llm.llm_policy(phase="plan"):                    # a plan scope A's runner did not open
            with pytest.raises(llm.LLMNotAllowed):
                llm.chat("forged")
        llm._close_session(a)
        assert not llm.llm_allowed()                                  # B still runs (between calls)
    finally:
        llm._close_session(a)
        llm._close_session(b)
        llm._close_session(b)                                         # twice: harmless
    assert llm.llm_allowed() and not llm.official_session_active()
    assert llm.chat("free again") == llm.MOCK_TEXT


def test_get_config_hides_the_key_and_sealed_keys_still_work(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test-secret")
    assert llm.get_config().api_key == llm.REDACTED_KEY and "sk-test" not in repr(llm.get_config())
    assert llm._config().api_key == "sk-test-secret"
    try:
        llm.seal_api_keys()                                           # what an isolated driver process does
        assert "MINIMAX_API_KEY" not in os.environ
        assert llm._config().api_key == "sk-test-secret" and llm.llm_available()
        assert llm.get_config().api_key == llm.REDACTED_KEY
    finally:
        llm._SEALED.clear()
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    assert llm.get_config().api_key is None


def test_native_langchain_model_is_refused_in_official_events():
    track = llm.llm_policy(phase="track", mode="official")
    token = llm._open_session(track, (track, llm.llm_policy(phase="plan")))
    try:
        with pytest.raises(llm.LLMNotAllowed, match="native=True"):
            llm.get_langchain_model(native=True)
    finally:
        llm._close_session(token)
    with llm.llm_policy(phase="plan", mode="official"):
        with pytest.raises(llm.LLMNotAllowed, match="native=True"):
            llm.get_langchain_model(native=True)


def test_deadline_cuts_request_timeouts_and_retries(minimax, monkeypatch):
    import time as _time

    fake = minimax(api_error(500, "busy"), completion("never"))
    with llm.llm_policy(phase="plan", deadline=_time.monotonic() + 1.8):
        with pytest.raises(llm.LLMDeadline, match="no time left to retry"):
            llm.chat("slow")
    assert len(fake.requests) == 1 and fake.requests[0]["timeout"] <= 1.3 + 1e-6   # cut from 120 s to the time left
    with llm.llm_policy(phase="reflect", deadline=_time.monotonic() + 0.5):
        assert not llm.llm_allowed()
        with pytest.raises(llm.LLMDeadline):
            llm.chat("too late")
    assert isinstance(llm.LLMDeadline("x"), llm.LLMError)


def test_waiting_for_a_busy_slot_stops_at_the_deadline(minimax):
    import time as _time

    sem = threading.BoundedSemaphore(1)
    sem.acquire()                                             # another driver holds the only slot
    llm.set_llm_semaphore(sem)
    minimax(completion("never"))
    t0 = _time.monotonic()
    with llm.llm_policy(phase="plan", deadline=_time.monotonic() + 1.6):
        with pytest.raises(llm.LLMDeadline, match="no free LLM slot"):
            llm.chat("x")
    assert _time.monotonic() - t0 < 1.5


def test_ask_with_tools_returns_partial_run_when_the_budget_ends():
    applied = []

    def set_speed(segment: int, speed: float) -> str:
        """Set the planned speed of one segment."""
        applied.append((segment, speed))
        return "ok"

    def responder(messages, tools):
        n = sum(1 for m in messages if m.get("role") == "tool")
        return {"tool": "set_speed", "args": {"segment": n, "speed": 20}} if n < 3 else "done"

    llm.set_mock_responder(responder)
    with llm.llm_policy(phase="track", mode="official"), llm.llm_policy(phase="plan"):   # 2 calls
        run = llm.ask_with_tools("tune", [set_speed])
    assert run.stopped == "budget" and run.rounds == 2 and applied == [(0, 20), (1, 20)]
    assert [c["result"] for c in run.calls] == ["ok", "ok"]
    with llm.llm_policy(phase="track", mode="official"), llm.llm_policy(phase="drive"):
        with pytest.raises(llm.LLMNotAllowed):                # a refused FIRST call still raises
            llm.ask_with_tools("tune", [set_speed])
    llm.set_mock_responder(responder)
    assert llm.ask_with_tools("tune", [set_speed], max_rounds=1).stopped == "rounds"


def test_effort_none_keeps_thinking_for_tool_turns(minimax):
    def corner_speed(radius: float) -> float:
        """v_max."""
        return radius

    fake = minimax(completion("done"))
    llm.ask_with_tools("T3?", [corner_speed], effort="none")
    assert fake.requests[0]["extra_body"] == {"reasoning_split": True}     # no thinking: disabled


def test_extract_json_refuses_a_top_level_array():
    with pytest.raises(ValueError, match="JSON array with 2 item"):
        llm.extract_json('[{"segment": 3}, {"segment": 5}]')
    llm.set_mock_responder(lambda m, t: '[{"a": 1}]' if len(m) < 3 else '{"items": [{"a": 1}]}')
    assert llm.chat_json("list please") == {"items": [{"a": 1}]}           # the nudge asks for one object


def test_log_file_names_are_safe_and_logging_never_breaks_a_call(monkeypatch, tmp_path):
    assert llm.log_file_name("Rocket" * 50) == "Rocket" * 10 + ".jsonl"
    assert llm.log_file_name("Ракета") != llm.log_file_name("Молния")    # non-ASCII names stay apart
    assert llm.log_file_name("../..") == "llm.jsonl"
    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("LLM_LOG_DIR", str(blocker / "sub"))               # cannot be created
    monkeypatch.setattr(llm, "_LOG_WARNED", [])
    with llm.llm_policy(label="x" * 300):
        assert llm.chat("still works") == llm.MOCK_TEXT


# ------------------------------------------------------------------ live MiniMax (opt-in, one call each)
@pytest.fixture
def live(monkeypatch):
    """The real provider from .env (never printed). Skips unless AGP_LIVE_LLM=1 and a key is present."""
    if os.getenv("AGP_LIVE_LLM") != "1":
        pytest.skip("live MiniMax test: set AGP_LIVE_LLM=1")
    path = llm.REPO_ROOT / ".env"
    for key, value in (llm.parse_env(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}).items():
        monkeypatch.setenv(key, value)                  # restored after the test
    if os.getenv("LLM_PROVIDER", "minimax") == "mock":
        monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.delenv("LLM_NO_DOTENV", raising=False)
    monkeypatch.setattr(llm, "_CLIENTS", {})
    if not llm.llm_available():
        pytest.skip("no MiniMax key in .env")
    llm.warm_up()
    return llm.get_config()


@pytest.mark.llm
def test_live_chat(live):
    text = llm.chat("Say hi to a race driver in five words.", max_tokens=1024)
    assert text.strip() and "<think>" not in text
    assert llm.llm_stats()["calls"] == 1 and llm.llm_stats()["completion_tokens"] > 0


@pytest.mark.llm
def test_live_chat_json(live):
    out = llm.chat_json('Return {"throttle": <float -1..1>, "steering": <float -1..1>} for a car at speed 10 '
                        "with the next waypoint 30 degrees to the RIGHT (+ = right).")
    assert isinstance(out.get("throttle"), (int, float)) and isinstance(out.get("steering"), (int, float))
    assert out["steering"] > 0                           # + = right


@pytest.mark.llm
@pytest.mark.parametrize("effort", ["low", "none"])
def test_live_ask_with_tools(live, effort):
    def corner_speed(radius: float) -> float:
        """Fastest speed (u/s) through a corner of this radius (grip 20 u/s^2)."""
        return round((20.0 * radius) ** 0.5, 2)

    def braking_distance(v_from: float, v_to: float) -> float:
        """Distance (u) to brake from v_from to v_to at 25 u/s^2."""
        return round(max(0.0, (v_from ** 2 - v_to ** 2) / 50.0), 2)

    run = llm.ask_with_tools("Use the tools: corner speed for radius 45, and braking distance from 40 to that "
                             "speed. Then answer in one sentence.", [corner_speed, braking_distance],
                             effort=effort, max_rounds=4)
    names = [c["name"] for c in run.calls]
    assert "corner_speed" in names and run.final_text and run.stopped == ""
    assert next(c for c in run.calls if c["name"] == "corner_speed")["result"] == 30.0
    history = [m for m in run.messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert history and all("content" in m for m in history)


@pytest.mark.llm
def test_live_official_plan_budget_and_deadline(live):
    import time as _time

    with llm.llm_policy(phase="track", mode="official", label="live"):
        with llm.llm_policy(phase="plan", deadline=_time.monotonic() + 90):
            assert llm.chat("Reply with the single word GO.", max_tokens=512).strip()
            llm.chat("Reply with the single word GO.", max_tokens=512)
            with pytest.raises(llm.LLMBudgetExceeded):
                llm.chat("one too many")
        with llm.llm_policy(phase="race"):
            with pytest.raises(llm.LLMNotAllowed):
                llm.chat("not in the race")


def test_identical_refusals_are_logged_once_then_counted(monkeypatch, tmp_path):
    """An LLM call in every drive() tick of an official race: 1 log row, not hundreds."""
    monkeypatch.setenv("LLM_LOG_DIR", str(tmp_path / "logs"))
    with llm.llm_policy(phase="track", mode="official", label="kai"):
        with llm.llm_policy(phase="drive"):
            for _ in range(12):
                with pytest.raises(llm.LLMNotAllowed):
                    llm.chat("steer?")
    rows = (tmp_path / "logs" / "kai.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2                                                    # the 1st and the 10th
    assert "refused 10 times" in rows[1]


def test_ask_with_tools_last_round_asks_for_the_answer():
    def lookup(key: str) -> str:
        """Look something up."""
        return "42"

    seen = []

    def responder(messages, tools):
        seen.append(messages[-1])
        return {"tool": "lookup", "args": {"key": "x"}} if len(seen) == 1 else '{"answer": 42}'

    llm.set_mock_responder(responder)
    run = llm.ask_with_tools("answer in JSON", [lookup], max_rounds=2)
    assert run.rounds == 2 and llm.extract_json(run.final_text) == {"answer": 42}
    assert seen[1]["role"] == "tool" and seen[1]["content"] == "42\n\n" + llm.FINAL_ROUND_NUDGE


def test_stats_seconds_leave_out_the_sdk_start_up(monkeypatch):
    """The first real call of a process imports the SDK (seconds to minutes): that is not the call's time."""
    import types

    reply = completion("OK")                                                  # (imports the SDK now)

    class SlowClient:
        def __init__(self):
            time.sleep(0.4)                                                   # the "cold import"
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: reply))

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-key")
    clients: list = []
    monkeypatch.setattr(llm, "_client", lambda cfg: clients[0] if clients else clients.append(SlowClient()) or clients[0])
    assert llm.chat("hi") == "OK"
    assert llm.llm_stats()["seconds"] < 0.3
