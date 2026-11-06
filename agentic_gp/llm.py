"""LLM helper for drivers: MiniMax first, any OpenAI-compatible provider, plus an offline mock.

Configuration comes from environment variables. A `.env` file in the current directory or the repo
root is loaded automatically, but real environment variables always win:

    LLM_PROVIDER     = minimax (default) | openai | ollama | openrouter | mock | replay
    LLM_MODEL        = model id                       (default MiniMax-M3; see DEFAULT_MODELS)
    LLM_EFFORT       = none | low | medium | high | xhigh | max   (default low; "none"/"off" = thinking
                       off on MiniMax-M3, ~0.5 s answers; MiniMax-M3.1-* gets reasoning_effort, none -> low)
    MINIMAX_API_KEY  / MINIMAX_BASE_URL               (default https://api.minimax.io/v1)
    OPENAI_API_KEY   / OLLAMA_BASE_URL / OPENROUTER_API_KEY
    LLM_TIMEOUT      = seconds per request            (default 30; 120 inside plan/reflect)
    LLM_LOG_DIR      = folder: every request/response is appended there as JSONL
    LLM_DAILY_CAP    = soft per-process call cap      (default 250, then LLMBudgetExceeded)
    LLM_CASSETTE     = path.jsonl replayed by LLM_PROVIDER=replay; LLM_RECORD = path.jsonl to record

Usage:
    from agentic_gp import llm
    llm.chat("Say hi in five words")                                   # -> "Hi from the pit wall!"
    plan = llm.chat_json("Corner speeds for this track as JSON ...")   # -> dict (robust parsing)
    run = llm.ask_with_tools("How fast through T3?", [corner_speed])    # the model calls your function
    print(run.final_text, run.calls)
    llm.last_reasoning()                                                # the model's thinking, for UIs

Rules of the road (official events): the runner wraps driver calls in `llm_policy(...)`. LLM calls are
allowed only inside plan()/reflect() (not in drive(), not during qualifying/race, not at import time),
plan/reflect have small budgets (LLMBudgetExceeded) and a deadline: a call that cannot finish before the
plan()/reflect() time limit raises LLMError instead of hanging. All are LLMError subclasses, so
`except LLMError: fallback()` keeps you racing. An inner scope cannot loosen an "official" one, and
calls from threads (or contexts) a driver starts are judged by the running session's rules first.
In an isolated driver process the API key is not in os.environ, and get_config() hides it.

Runner / instructor hooks: llm_policy(phase=, mode=, budget=, label=, deadline=), current_llm_policy(),
official_session_active(), seal_api_keys(),
llm_allowed(), set_llm_semaphore(sem), llm_stats() / reset_llm_stats(), warm_up(), llm_smoke_test() (CLI llm-check).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import threading
import time
import types
import typing
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Union

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {
    "minimax": "MiniMax-M3",              # instructor's choice; MiniMax-M3.1-Flash is cheaper (always thinks)
    "openai": "gpt-4o-mini",
    "ollama": "hermes3",                  # local Hermes 3 via `ollama pull hermes3`
    "openrouter": "minimax/minimax-m3",
    "mock": "mock",
    "replay": "MiniMax-M3",
}
BASE_URLS = {                             # (env var, default)
    "minimax": ("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
    "openai": ("OPENAI_BASE_URL", None),
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    "openrouter": ("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
}
API_KEYS = {"minimax": "MINIMAX_API_KEY", "openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
EFFORT_ALIASES = {"off": "none", "disabled": "none", "false": "none", "0": "none", "": "low"}

MAX_TOKENS = 4096             # max_completion_tokens; thinking tokens count toward it
MAX_TOKENS_PLANNING = 16384   # inside plan()/reflect()
TIMEOUT = 30.0                # seconds per request (LLM_TIMEOUT overrides)
TIMEOUT_PLANNING = 120.0      # inside plan()/reflect() in open mode (60 s in official mode)
RETRY_BASE_SECONDS = 1.0      # backoff 1 s, 2 s (+ jitter)

# MiniMax error codes arrive as a "(NNNN)" suffix of the error message.
FATAL_CODES = {
    1004: "authentication failed - check MINIMAX_API_KEY in .env (ask your instructor)",
    1008: "insufficient balance on the MiniMax account - tell your instructor",
    2013: "invalid request parameters",
    2049: "invalid API key - check MINIMAX_API_KEY in .env (ask your instructor)",
    2056: "usage-window limit reached for this key (5-hour / weekly window)",
}
RETRY_CODES = {1000, 1001, 1002, 1013, 1024, 1033, 1041, 2045}   # transient: timeout, rate limit, busy

JSON_RULE = "Reply with ONLY one valid JSON object. No prose, no code fences."
MOCK_TEXT = "OK (mock LLM)"


# ------------------------------------------------------------------ errors
class LLMError(RuntimeError):
    """Any LLM failure. `code` = MiniMax error code (if any), `status` = HTTP status (if any)."""

    def __init__(self, message: str, *, code: int | None = None, status: int | None = None,
                 retryable: bool = False, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code, self.status, self.retryable, self.retry_after = code, status, retryable, retry_after


class LLMNotAllowed(LLMError):
    """Raised for an LLM call where the rules forbid one (official events: drive(), qualifying, race)."""


class LLMBudgetExceeded(LLMError):
    """Raised when a call budget (policy scope or LLM_DAILY_CAP) is used up."""


class LLMDeadline(LLMError):
    """Raised when a call cannot finish before the plan()/reflect() time limit (policy deadline)."""


# ------------------------------------------------------------------ .env + config
def parse_env(text: str) -> dict[str, str]:
    """Parse `.env` text: KEY=VALUE lines, `#` comments, optional quotes and `export ` prefix."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value.endswith(value[0]):
            value = value[1:-1]                          # quoted: keep everything inside
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()     # unquoted: strip a trailing comment
        if key:
            out[key] = value
    return out


_ENV_LOADED = False


def load_env(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Load `.env` into os.environ (real env vars win unless override). Returns the parsed values.

    Without `path`: `./.env` then `<repo>/.env`, once per process. LLM_NO_DOTENV=1 disables this.
    """
    global _ENV_LOADED
    if path is None:
        if _ENV_LOADED or os.getenv("LLM_NO_DOTENV"):
            return {}
        _ENV_LOADED = True
        paths = [Path.cwd() / ".env", REPO_ROOT / ".env"]
    else:
        paths = [Path(path)]
    loaded: dict[str, str] = {}
    for p in paths:
        if not p.is_file():
            continue
        for key, value in parse_env(p.read_text(encoding="utf-8-sig")).items():
            if override or key not in os.environ:
                os.environ[key] = value
                loaded.setdefault(key, value)
    return loaded


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    effort: str
    base_url: str | None
    api_key: str | None
    max_retries: int


REDACTED_KEY = "(set - hidden)"
# API keys moved out of os.environ (seal_api_keys): an isolated driver process keeps its key here, so
# driver code cannot read it from the environment and call the provider around the race rules.
_SEALED: dict[str, str] = {}
_KEYS_LOCK = threading.Lock()


def get_config(model: str | None = None) -> LLMConfig:
    """Current provider settings (re-read from the environment on every call). The API key is not
    included: `api_key` is REDACTED_KEY when a key is set, None otherwise."""
    cfg = _config(model)
    return LLMConfig(**{**cfg.__dict__, "api_key": REDACTED_KEY if cfg.api_key else None})


def seal_api_keys() -> None:
    """Move the provider API keys from os.environ into this module (isolated driver processes call
    this before the driver file loads; the llm helpers keep working, `os.environ` no longer shows them)."""
    load_env()
    for var in API_KEYS.values():
        value = os.environ.pop(var, None)
        if value:
            _SEALED[var] = value


def sealed_key(var: str) -> str | None:
    """The key in env var `var`, also after seal_api_keys() (runner use: the LLM relay)."""
    load_env()
    return os.getenv(var) or _SEALED.get(var)


@contextlib.contextmanager
def api_keys_hidden():
    """Take the API keys out of os.environ for the block (the runner starts driver processes in it, so
    they do not inherit a key). This process keeps using them through _SEALED meanwhile."""
    load_env()
    with _KEYS_LOCK:
        hidden = {var: os.environ.pop(var) for var in API_KEYS.values() if var in os.environ}
        added = [var for var in hidden if var not in _SEALED]
        for var in added:
            _SEALED[var] = hidden[var]
    try:
        yield
    finally:
        with _KEYS_LOCK:
            for var, value in hidden.items():
                os.environ.setdefault(var, value)
            for var in added:                     # nothing stays behind: os.environ is the source again
                _SEALED.pop(var, None)


def _config(model: str | None = None) -> LLMConfig:
    """get_config() with the real API key (internal)."""
    load_env()
    provider = os.getenv("LLM_PROVIDER", "minimax").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise LLMError(f"Unknown LLM_PROVIDER {provider!r}; choose one of {sorted(DEFAULT_MODELS)}")
    env_url, default_url = BASE_URLS.get(provider, (None, None))
    key_var = API_KEYS.get(provider)
    relay = os.getenv("AGP_LLM_PROXY_URL")
    if relay and key_var:                        # an isolated driver: the runner's relay holds the key
        return LLMConfig(provider=provider, model=model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
                         effort=os.getenv("LLM_EFFORT", "low").strip().lower(), base_url=relay,
                         api_key=os.getenv("AGP_LLM_PROXY_TOKEN"),
                         max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")))
    return LLMConfig(
        provider=provider,
        model=model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider],
        effort=os.getenv("LLM_EFFORT", "low").strip().lower(),
        base_url=(os.getenv(env_url) if env_url else None) or default_url,
        api_key=(os.getenv(key_var) or _SEALED.get(key_var) if key_var else None)
        or ("ollama" if provider == "ollama" else None),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


def normalize_effort(effort: str | None) -> str:
    """'off'/'disabled' -> 'none'; unknown values raise LLMError (typos would silently cost money)."""
    value = str(effort if effort is not None else "low").strip().lower()
    value = EFFORT_ALIASES.get(value, value)
    if value not in EFFORTS:
        raise LLMError(f"Unknown effort {effort!r}; choose one of {list(EFFORTS)} ('off' = 'none')")
    return value


def warm_up() -> None:
    """Import the OpenAI SDK and build the client now (no request), so the first real call is fast.

    A cold `import openai` can take seconds (minutes on a busy machine); runners call this before
    timing plan()/drive(). Does nothing for mock/replay or when no key is set.
    """
    cfg = _config()
    if cfg.provider not in ("mock", "replay") and cfg.api_key:
        _client(cfg).chat.completions          # touching the resource triggers the SDK's lazy imports


def describe_llm() -> str:
    """'provider:model', e.g. 'minimax:MiniMax-M3' (handy for a driver's MODEL metadata)."""
    cfg = get_config()
    return f"{cfg.provider}:{cfg.model}"


def llm_available() -> bool:
    """True if a call can be attempted: key present, or mock / ollama / an existing cassette."""
    cfg = get_config()
    if cfg.provider in ("mock", "ollama"):
        return True
    if cfg.provider == "replay":
        return Path(os.getenv("LLM_CASSETTE", "")).is_file()
    return bool(cfg.api_key)


# ------------------------------------------------------------------ stats + policy
_STATS_KEYS = ("calls", "errors", "prompt_tokens", "completion_tokens", "seconds", "cache_hits")
_STATS: dict[str, float] = dict.fromkeys(_STATS_KEYS, 0)
_LOCK = threading.Lock()
_DAILY = {"calls": 0}                # real network requests made by this process
_LAST = {"reasoning": ""}


def llm_stats() -> dict:
    """Process-wide usage: calls, errors, prompt_tokens, completion_tokens, seconds, cache_hits."""
    with _LOCK:
        out = {k: (round(v, 3) if k == "seconds" else int(v)) for k, v in _STATS.items()}
    return out


def reset_llm_stats() -> None:
    with _LOCK:
        _STATS.update(dict.fromkeys(_STATS_KEYS, 0))
        _DAILY["calls"] = 0
        _REFUSALS.clear()                           # the refused-call log counters too


def last_reasoning() -> str:
    """The model's reasoning text from the most recent call in this process ('' if none)."""
    return _LAST["reasoning"]


FORBIDDEN_IN_OFFICIAL = ("drive", "qualifying", "race")
ALLOWED_IN_OFFICIAL = ("plan", "reflect")       # official events: a call needs one of these phases
MIN_CALL_SECONDS = 1.0                          # less time than this before a deadline: no new call
OFFICIAL_BUDGETS = {"plan": 2, "reflect": 2, "track": 8}    # default budgets per scope in official mode


@dataclass
class LLMPolicy:
    """One scope of LLM rules. Scopes nest; a call must satisfy every active scope.

    phase   : "track" | "plan" | "drive" | "reflect" | "practice" | "qualifying" | "race" | "session" | "load" | None
    mode    : "open" (lessons) | "official" (grand prix) | None = inherit. "official" anywhere in the stack
              wins: an inner scope cannot loosen it.
    budget  : max calls inside this scope (None = unlimited; official plan/reflect/track get defaults)
    deadline: time.monotonic() by which every call in this scope must be done (None = none); request
              timeouts, retries and the wait for a free slot are cut to it, then LLMError is raised
    The object can be re-entered (`with track_scope:` many times); its usage counters accumulate.
    """

    phase: str | None = None
    mode: str | None = None
    budget: int | None = None
    label: str = ""
    deadline: float | None = None
    usage: dict = field(default_factory=lambda: dict.fromkeys(_STATS_KEYS, 0))
    _tokens: list = field(default_factory=list, repr=False, compare=False)

    def __enter__(self) -> "LLMPolicy":
        self._tokens.append(_POLICY.set(_POLICY.get() + (self,)))
        return self

    def __exit__(self, *exc) -> None:
        _POLICY.reset(self._tokens.pop())

    def limit(self, mode: str) -> int | None:
        if self.budget is not None:
            return self.budget
        return OFFICIAL_BUDGETS.get(self.phase or "") if mode == "official" else None


_POLICY: contextvars.ContextVar[tuple] = contextvars.ContextVar("agentic_gp_llm_policy", default=())

# Official sessions running in this process. The runner registers each one with its ROOT scope (the
# track scope it opens around every driver call) and keeps its current rules up to date: "between
# calls", or the plan()/reflect()/drive() stack while one runs. A call whose context stack does not
# start with a registered root - a thread the driver started, a fresh contextvars.Context, a scope the
# driver opened itself - is judged by those rules FIRST (see _stack()), so it cannot escape them.
_SESSIONS: dict[int, list] = {}          # token -> [root scope, current stack]
_SESSIONS_LOCK = threading.Lock()
_SESSION_IDS = iter(range(1, 1 << 62))
_FALLBACK: list = [()]                   # extra fallback stack set by _swap_fallback (tests, tools)
# Several official sessions at once (e.g. two in-process grand prix): a call from an unknown context
# cannot be told apart, so it is refused.
_AMBIGUOUS = LLMPolicy(phase="between", mode="official", label="(several official sessions)")


def llm_policy(phase: str | None = None, mode: str | None = None, budget: int | None = None,
               label: str = "", deadline: float | None = None) -> LLMPolicy:
    """Rules for the calls made inside a `with` block, e.g.

        with llm_policy(phase="track", mode="official", label="max"):        # <= 8 calls on this track
            with llm_policy(phase="plan", deadline=time.monotonic() + 115):  # <= 2 calls in plan()
                driver.plan(track)
    """
    return LLMPolicy(phase=phase, mode=mode, budget=budget, label=label, deadline=deadline)


def _open_session(root: LLMPolicy, stack: tuple) -> int:
    """Runner hook: register an official session (root = its track/load scope). Returns its token."""
    with _SESSIONS_LOCK:
        token = next(_SESSION_IDS)
        _SESSIONS[token] = [root, tuple(stack)]
    return token


def _set_session(token: int, stack: tuple) -> None:
    """Runner hook: the rules for calls from unknown contexts while this session is in `stack`."""
    with _SESSIONS_LOCK:
        if token in _SESSIONS:
            _SESSIONS[token][1] = tuple(stack)


def _close_session(token: int | None) -> None:
    """Runner hook: the session ended (safe to call twice; other sessions keep their rules)."""
    with _SESSIONS_LOCK:
        _SESSIONS.pop(token, None)


def _reset_sessions() -> None:
    """Forget every registered session and the extra fallback (test isolation)."""
    with _SESSIONS_LOCK:
        _SESSIONS.clear()
    _FALLBACK[0] = ()


def official_session_active() -> bool:
    """True while an official session (or an official driver load) runs in this process."""
    return bool(_fallback())


def _session_stacks() -> list[tuple]:
    """Current rules of every registered official session (+ the _swap_fallback stack, if any)."""
    with _SESSIONS_LOCK:
        stacks = [stack for _, stack in _SESSIONS.values()]
    if _FALLBACK[0]:
        stacks.append(_FALLBACK[0])
    return stacks


def _fallback() -> tuple:
    """The rules for a call from a context the runner did not set up (() = none)."""
    stacks = _session_stacks()
    if not stacks:
        return ()
    if len(stacks) == 1:
        return stacks[0]
    return (_AMBIGUOUS,)


class _Stack(tuple):
    """A policy stack plus how many of its scopes the RUNNER opened (`trusted`). Only those can make a
    call "inside plan()/reflect()" in official mode: a driver's own `llm_policy(phase="plan")` adds
    limits but never permission."""

    trusted: int = 0


def _make_stack(scopes: tuple, trusted: int) -> _Stack:
    out = _Stack(scopes)
    out.trusted = trusted
    return out


def _stack() -> _Stack:
    """The active policy stack. A context the runner set up (its stack starts with a registered
    session's root scope) uses its own stack; any other context - a thread or a fresh
    contextvars.Context the driver started, a scope it opened itself - gets the running official
    session's rules in FRONT of its own scopes, so it can only add limits, never escape them."""
    ctx = _POLICY.get()
    with _SESSIONS_LOCK:
        sessions = [(root, stack) for root, stack in _SESSIONS.values()]
    if _FALLBACK[0]:
        sessions.append((_FALLBACK[0][0], _FALLBACK[0]))
    if not sessions:
        return _make_stack(ctx, len(ctx))
    if ctx:
        for root, current in sessions:
            if ctx[0] is root:                    # the runner's context: trust what it opened
                n = 0
                while n < min(len(ctx), len(current)) and ctx[n] is current[n]:
                    n += 1
                return _make_stack(ctx, max(n, 1))
    fb = _fallback()
    return _make_stack(fb + ctx, len(fb))


def _swap_fallback(stack: tuple) -> tuple:
    """Set an extra fallback policy stack (the rules for calls from unknown contexts), return the
    previous one. The runner registers sessions with _open_session instead."""
    prev, _FALLBACK[0] = _FALLBACK[0], tuple(stack)
    return prev


def current_llm_policy() -> dict:
    """Effective rules now: mode, active phases, calls left (None = unlimited), label, seconds_left."""
    stack = _stack()
    mode = _mode(stack)
    left = [p.limit(mode) - p.usage["calls"] for p in stack if p.limit(mode) is not None]
    secs = _seconds_left(stack)
    return {"mode": mode, "phases": [p.phase for p in stack if p.phase], "calls_left": min(left) if left else None,
            "label": next((p.label for p in reversed(stack) if p.label), ""),
            "seconds_left": None if secs is None else round(secs, 1)}


def llm_allowed() -> bool:
    """True if one more call would pass the active policy (use it to choose LLM vs. fallback)."""
    stack = _stack()
    try:
        _check_policy(stack, count=False)
    except LLMError:
        return False
    return True


def _mode(stack: tuple) -> str:
    """'official' if ANY scope says so (an inner scope cannot loosen it), else the innermost mode."""
    if any(p.mode == "official" for p in stack):
        return "official"
    return next((p.mode for p in reversed(stack) if p.mode), "open")


def _seconds_left(stack: tuple) -> float | None:
    """Seconds until the nearest deadline in the stack (None = no deadline)."""
    deadlines = [p.deadline for p in stack if p.deadline is not None]
    return None if not deadlines else min(deadlines) - time.monotonic()


def _phase(stack: tuple) -> str | None:
    return next((p.phase for p in reversed(stack) if p.phase), None)


def _check_policy(stack: tuple, count: bool = True) -> None:
    mode = _mode(stack)
    if any(p is _AMBIGUOUS for p in stack):
        raise LLMNotAllowed("several official sessions run in this process: LLM calls from threads or "
                            "contexts the runner did not start are refused")
    if mode == "official":
        for p in stack:
            if p.phase in FORBIDDEN_IN_OFFICIAL:
                raise LLMNotAllowed(f"LLM calls are not allowed during {p.phase} in official events "
                                    "- drive with code there (plan/reflect in practice may use the LLM)")
        if not any(p.phase in ALLOWED_IN_OFFICIAL for p in stack[:getattr(stack, "trusted", len(stack))]):
            raise LLMNotAllowed("LLM calls are only allowed inside plan() and reflect() in official events "
                                "(not at import time and not between calls)")
    secs = _seconds_left(stack)
    if secs is not None and secs < MIN_CALL_SECONDS:
        raise LLMDeadline(f"no time left for an LLM call ({max(secs, 0.0):.1f}s before the "
                          f"{_phase(stack) or 'call'}() time limit) - use your fallback")
    for p in stack:
        limit = p.limit(mode)
        if limit is not None and p.usage["calls"] >= limit:
            raise LLMBudgetExceeded(f"LLM budget used up: {p.usage['calls']}/{limit} calls in {p.phase or 'scope'}")
    if count:
        with _LOCK:
            for p in stack:
                p.usage["calls"] += 1


def _record(stack: tuple, usage: dict, seconds: float, *, error: bool = False, cached: bool = False) -> None:
    delta = {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
             "seconds": seconds, "errors": int(error), "cache_hits": int(cached)}
    with _LOCK:
        _STATS["calls"] += 1
        for target in (_STATS, *(p.usage for p in stack)):
            for k, v in delta.items():
                target[k] += v


# ------------------------------------------------------------------ concurrency + rate limits
_SEMAPHORE: Any = None
_RECENT: deque = deque()


def set_llm_semaphore(sem: Any) -> None:
    """Limit in-flight requests with any object that has acquire([timeout=])/release() (e.g. a
    multiprocessing BoundedSemaphore shared by all grand-prix workers). None = default
    (LLM_MAX_INFLIGHT, 1 call in flight per process; `llm-check --load N` uses N)."""
    global _SEMAPHORE
    _SEMAPHORE = sem


def _semaphore() -> Any:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = threading.BoundedSemaphore(int(os.getenv("LLM_MAX_INFLIGHT", "1")))
    return _SEMAPHORE


def _sleep(seconds: float) -> None:          # tests monkeypatch this
    time.sleep(seconds)


def _count_daily_call(describe: str) -> None:
    """Soft per-process cap (LLM_DAILY_CAP) + optional requests/minute limit (LLM_MAX_RPM)."""
    cap = int(os.getenv("LLM_DAILY_CAP", "250"))
    with _LOCK:
        if _DAILY["calls"] >= cap:
            raise LLMBudgetExceeded(f"LLM_DAILY_CAP reached ({cap} calls in this process) - take a break or ask your instructor")
        _DAILY["calls"] += 1
        n = _DAILY["calls"]
    rpm = int(os.getenv("LLM_MAX_RPM", "0"))
    while rpm > 0:
        now = time.monotonic()
        with _LOCK:
            while _RECENT and now - _RECENT[0] > 60.0:
                _RECENT.popleft()
            if len(_RECENT) < rpm:
                _RECENT.append(now)
                break
            wait = 60.0 - (now - _RECENT[0])
        print(f"[llm] {rpm} calls/min limit: waiting {wait:.0f}s", file=sys.stderr)
        _sleep(wait + 0.05)
    if not os.getenv("LLM_QUIET"):
        print(f"[llm] llm calls {n}/{cap} ({describe})", file=sys.stderr)


# ------------------------------------------------------------------ text helpers
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def strip_think(text: str | None) -> str:
    """Remove <think>...</think> blocks (also unterminated ones) and surrounding whitespace."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:                       # opening tag missing
        text = text.split("</think>")[-1]
    if "<think>" in text:                        # cut off while thinking
        text = text.split("<think>")[0]
    return text.strip()


def _think_text(text: str | None) -> str:
    return "\n".join(m.strip() for m in _THINK_RE.findall(text or ""))


def extract_json(text: str | None) -> dict:
    """First JSON object in `text`: ignores <think>, code fences and prose; forgives trailing commas.

    Raises ValueError if there is none.
    """
    clean = strip_think(text)
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", clean, flags=re.S)
    decoder = json.JSONDecoder()
    for candidate in [*fenced, clean]:
        if candidate.lstrip().startswith("["):         # a JSON array: do not silently take its first item
            try:
                value, _ = decoder.raw_decode(candidate.lstrip())
            except ValueError:
                value = None
            if isinstance(value, list):
                raise ValueError(f"got a JSON array with {len(value)} item(s), need ONE JSON object "
                                 '(wrap the list, e.g. {"items": [...]})')
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            for m in re.finditer(r"\{", variant):
                try:
                    obj, _ = decoder.raw_decode(variant, m.start())
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    return obj
    raise ValueError(f"no JSON object in: {clean[:200]!r}")


def minimax_code(error: Any) -> int | None:
    """MiniMax error code from an exception or message: '... (2045)' -> 2045."""
    body = getattr(error, "body", None)
    msg = body.get("message", "") if isinstance(body, dict) else ""
    if not msg and isinstance(body, dict) and isinstance(body.get("error"), dict):
        msg = body["error"].get("message", "")
    msg = msg or str(error)
    m = re.search(r"\((\d{4,5})\)\s*\W*$", msg.strip())
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ tool schemas
_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", dict: "object", list: "array"}


def _json_type(tp: Any) -> dict:
    """Python type hint -> JSON schema fragment (the common cases students write)."""
    if tp in _JSON_TYPES:
        return {"type": _JSON_TYPES[tp]}
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return {"type": _JSON_TYPES.get(type(args[0]), "string"), "enum": list(args)}
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _json_type(options[0]) if len(options) == 1 else {"anyOf": [_json_type(a) for a in options]}
    if origin in (list, tuple, set, frozenset) or tp in (tuple, set):
        return {"type": "array", "items": _json_type(args[0])} if args else {"type": "array"}
    if origin is dict:
        return {"type": "object"}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Summary text + {param: description} from 'Args:' / ':param x:' style docstrings."""
    summary, params, section = [], {}, None
    for line in doc.splitlines():
        s = line.strip()
        m = re.match(r":param\s+(\w+)\s*:\s*(.*)", s)
        if m:
            params[m.group(1)] = m.group(2)
            continue
        if re.fullmatch(r"(args|arguments|parameters|params)\s*:", s.lower()):
            section = "args"
            continue
        if re.fullmatch(r"(returns?|raises|yields|examples?|notes?)\s*:", s.lower()) or s.startswith(":return"):
            section = "other"                            # not part of the tool description
            continue
        if section == "args":
            m = re.match(r"(\w+)\s*(\([^)]*\))?\s*:\s*(.+)", s)
            if m:
                params[m.group(1)] = m.group(3)
        if section is None:
            summary.append(s)
    return " ".join(x for x in summary if x).strip(), params


def function_schema(fn: Callable) -> dict:
    """OpenAI tool schema from a Python function's name, type hints and docstring."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    summary, param_docs = _parse_docstring(inspect.getdoc(fn) or "")
    properties, required = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        schema = _json_type(hints.get(name, str)) or {"type": "string"}
        if name in param_docs:
            schema["description"] = param_docs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
        elif isinstance(p.default, (int, float, str, bool)):
            schema["default"] = p.default
        properties[name] = schema
    return {"type": "function", "function": {
        "name": fn.__name__, "description": summary or fn.__name__,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


# ------------------------------------------------------------------ the request pipeline
@dataclass
class LLMReply:
    """One model turn. `message` is what goes back into the history (full: content + reasoning + tool_calls)."""

    text: str                                   # the answer with <think> removed
    reasoning: str = ""                         # reasoning_content and/or <think> text
    tool_calls: list = field(default_factory=list)   # [{"id", "type": "function", "function": {"name", "arguments"}}]
    finish_reason: str = "stop"
    usage: dict = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    message: dict = field(default_factory=dict)
    cached: bool = False


def _make_reply(content: str | None, api_reasoning: str | None, tool_calls: list, finish_reason: str,
                usage: dict, keep_reasoning: bool = True) -> LLMReply:
    content = content or ""
    message: dict = {"role": "assistant", "content": content}      # raw content: <think> kept in history
    if api_reasoning and keep_reasoning:
        message["reasoning_content"] = api_reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    reasoning = "\n".join(x for x in (api_reasoning or "", _think_text(content)) if x)
    return LLMReply(strip_think(content), reasoning, tool_calls, finish_reason or "stop", usage, message)


def _messages(prompt: str | list, system: str | None = None) -> list[dict]:
    msgs = [dict(m) for m in prompt] if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def _with_json_rule(msgs: list[dict]) -> list[dict]:
    """Ask for JSON in the system prompt (MiniMax ignores response_format, so never rely on it)."""
    if msgs and msgs[0].get("role") == "system":
        return [{**msgs[0], "content": f"{msgs[0]['content']}\n\n{JSON_RULE}"}, *msgs[1:]]
    return [{"role": "system", "content": JSON_RULE}, *msgs]


def _request_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def complete(messages: list[dict], *, tools: list[dict] | None = None, model: str | None = None,
             effort: str | None = None, max_tokens: int | None = None, json_mode: bool = False,
             cache: bool = False, timeout: float | None = None) -> LLMReply:
    """One model turn on a message list (low level; chat/chat_json/ask_with_tools build on it).

    Applies the policy, cache, provider quirks, retries, the finish_reason=length retry, stats and logs.
    """
    cfg = _config(model)
    stack = _stack()
    try:
        _check_policy(stack)
    except LLMError as exc:                         # refused calls are logged too (scrutineering)
        if _first_refusals(stack, exc):             # ... but not 800 identical rows from a drive() loop
            _log_call(cfg, stack, messages, tools, None, exc, 0.0)
        raise
    if cfg.provider not in ("mock", "replay") and cfg.api_key and not cache:
        try:
            _client(cfg)                            # a cold SDK import is start-up time, not this call's time
        except Exception:                           # noqa: BLE001 - the request below reports the problem
            pass
    if json_mode:
        messages = _with_json_rule(messages)
    phase = _phase(stack)
    planning = phase in ("plan", "reflect")
    max_tokens = max_tokens or (MAX_TOKENS_PLANNING if planning else MAX_TOKENS)
    effort = normalize_effort(effort or cfg.effort)
    t0 = time.perf_counter()
    reply, error = None, None
    try:
        cache_file = _cache_file(cfg, messages, tools, effort) if cache and cfg.provider != "mock" else None
        if cache_file and cache_file.is_file():
            reply = _reply_from_dict(json.loads(cache_file.read_text(encoding="utf-8")), cfg.provider)
            reply.cached = True
        else:
            if timeout is None:
                timeout = _default_timeout(planning, _mode(stack))
            deadline = _deadline(stack)
            reply = _provider_turn(cfg, messages, tools, effort, max_tokens, json_mode, timeout, deadline)
            if reply.finish_reason == "length" and not reply.text and not reply.tool_calls and \
                    _time_ok(deadline):
                # Thinking used up the whole output budget: retry once with twice the room.
                reply = _provider_turn(cfg, messages, tools, effort, 2 * max_tokens, json_mode, timeout, deadline)
            if cache_file and (reply.text or reply.tool_calls):
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(_reply_to_dict(reply)), encoding="utf-8")
        _LAST["reasoning"] = reply.reasoning
        return reply
    except LLMError as exc:
        error = exc
        raise
    except Exception as exc:                         # e.g. a mock responder that raised
        error = LLMError(f"{type(exc).__name__}: {exc}")
        raise error from exc
    finally:
        seconds = time.perf_counter() - t0
        _record(stack, reply.usage if reply else {}, seconds, error=error is not None,
                cached=bool(reply and reply.cached))
        _log_call(cfg, stack, messages, tools, reply, error, seconds)


_REFUSALS: dict = {}
REFUSAL_LOG_AT = {1, 10, 100, 1000, 10000}          # log the 1st, 10th, 100th ... identical refusal


def _first_refusals(stack: tuple, exc: Exception) -> bool:
    """True if this refused call should be logged: the same refusal (label + phases + reason) is logged
    on its 1st, 10th, 100th ... occurrence only; the record's error text carries the count."""
    label = next((p.label for p in reversed(stack) if p.label), "")
    key = (label, tuple(p.phase for p in stack if p.phase), str(exc))
    with _LOCK:
        n = _REFUSALS[key] = _REFUSALS.get(key, 0) + 1
    if n > 1 and n in REFUSAL_LOG_AT:
        exc.args = (f"{exc} [refused {n} times so far; only the 1st, 10th, 100th ... are logged]",)
    return n in REFUSAL_LOG_AT


def _deadline(stack: tuple) -> float | None:
    deadlines = [p.deadline for p in stack if p.deadline is not None]
    return min(deadlines) if deadlines else None


def _time_ok(deadline: float | None) -> bool:
    return deadline is None or deadline - time.monotonic() >= MIN_CALL_SECONDS


def _left(deadline: float | None, what: str) -> float | None:
    """Seconds left before the deadline (None = no deadline); LLMDeadline when too little is left."""
    if deadline is None:
        return None
    left = deadline - time.monotonic()
    if left < MIN_CALL_SECONDS:
        raise LLMDeadline(f"no time left for {what} before the plan()/reflect() time limit - use your fallback")
    return left


def _default_timeout(planning: bool, mode: str) -> float:
    if os.getenv("LLM_TIMEOUT"):
        return float(os.environ["LLM_TIMEOUT"])
    if planning:
        return 60.0 if mode == "official" else TIMEOUT_PLANNING
    return TIMEOUT


def _provider_turn(cfg: LLMConfig, messages: list[dict], tools: list[dict] | None, effort: str,
                   max_tokens: int, json_mode: bool, timeout: float, deadline: float | None = None) -> LLMReply:
    if cfg.provider == "mock":
        return _mock_turn(messages, tools, json_mode)
    if cfg.provider == "replay":
        return _replay_turn(messages, tools)
    if not cfg.api_key:
        raise LLMError(f"Set {API_KEYS.get(cfg.provider, 'the API key')} in .env - ask your instructor "
                       f"(or use LLM_PROVIDER=mock to work offline)")
    _count_daily_call(f"{cfg.provider}:{cfg.model}")
    reply = _call_api(cfg, _request_kwargs(cfg, messages, tools, effort, max_tokens, timeout), deadline)
    if os.getenv("LLM_RECORD"):
        _append_jsonl(Path(os.environ["LLM_RECORD"]), {"key": _request_key(messages, tools), "model": cfg.model,
                                                       "messages": messages, "tools": tools,
                                                       "reply": _reply_to_dict(reply)})
    return reply


def _request_kwargs(cfg: LLMConfig, messages: list[dict], tools: list[dict] | None, effort: str,
                    max_tokens: int, timeout: float) -> dict:
    """Provider quirks live here. No temperature (MiniMax ignores it), no response_format."""
    if cfg.provider != "minimax":                       # reasoning_content is a MiniMax extra field
        messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
    kw: dict = {"model": cfg.model, "messages": messages, "timeout": timeout}
    if cfg.provider in ("minimax", "openai"):
        kw["max_completion_tokens"] = max_tokens
    else:
        kw["max_tokens"] = max_tokens
    if tools:
        kw["tools"], kw["tool_choice"] = tools, "auto"   # MiniMax supports "auto" only
    name = cfg.model.lower()
    if cfg.provider == "minimax" and name.startswith("minimax-m3.1"):
        kw["reasoning_effort"] = effort if effort != "none" else "low"   # M3.1 always thinks; default is max (slow)
    elif cfg.provider == "minimax":
        extra: dict = {"reasoning_split": True}         # thinking -> reasoning_content, not <think> in content
        if effort == "none" and not tools:
            # fast answers for per-tick calls. Not with tools: without thinking M3 often SAYS it used a
            # tool instead of calling it (live-verified), so tool turns keep the default (low) thinking.
            extra["thinking"] = {"type": "disabled"}
        kw["extra_body"] = extra
    elif cfg.provider == "openai" and name.startswith(("o1", "o3", "o4", "gpt-5")) and effort != "none":
        kw["reasoning_effort"] = effort
    return kw


_CLIENTS: dict = {}


def _client(cfg: LLMConfig) -> Any:
    key = (cfg.base_url, cfg.api_key)
    if key not in _CLIENTS:
        from openai import OpenAI          # lazy: mock-only users do not need the SDK
        # max_retries=0: we retry ourselves so fatal MiniMax codes (e.g. 2056 on a 429) stop at once.
        _CLIENTS[key] = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url, max_retries=0, timeout=TIMEOUT)
    return _CLIENTS[key]


def _acquire(sem: Any, timeout: float | None) -> bool:
    """sem.acquire with a timeout when the object supports one."""
    if timeout is None:
        return sem.acquire() is not False
    try:
        return bool(sem.acquire(timeout=timeout))
    except TypeError:                              # an acquire() without a timeout parameter
        return sem.acquire() is not False


def _call_api(cfg: LLMConfig, kwargs: dict, deadline: float | None = None) -> LLMReply:
    """Send one request with up to `cfg.max_retries` retries (exponential backoff + jitter).

    With a deadline (plan/reflect), the wait for a free slot, every request timeout and every retry
    are cut to the time left, and LLMDeadline is raised when it is used up.
    """
    sem = _semaphore()
    base_timeout = float(kwargs.get("timeout") or TIMEOUT)
    for attempt in range(cfg.max_retries + 1):
        left = _left(deadline, "an LLM request")
        # wait for a slot only while a request could still be sent in time
        if not _acquire(sem, None if left is None else max(0.0, left - MIN_CALL_SECONDS)):
            raise LLMDeadline("no free LLM slot before the plan()/reflect() time limit (the class is busy)")
        try:
            left = _left(deadline, "an LLM request")
            if left is not None:
                kwargs = {**kwargs, "timeout": max(0.5, min(base_timeout, left - 0.5))}
            response = _client(cfg).chat.completions.create(**kwargs)
            return _reply_from_response(response, keep_reasoning=cfg.provider == "minimax")
        except LLMDeadline:
            raise
        except Exception as exc:
            err = _as_llm_error(exc, cfg)
            if not err.retryable or attempt == cfg.max_retries:
                raise err from exc
        finally:
            sem.release()
        delay = min(err.retry_after or RETRY_BASE_SECONDS * 2 ** attempt * (1.0 + 0.5 * random.random()), 20.0)
        if deadline is not None and deadline - time.monotonic() < delay + MIN_CALL_SECONDS:
            raise LLMDeadline(f"{err} - no time left to retry before the plan()/reflect() time limit") from None
        print(f"[llm] {err} - retry {attempt + 1}/{cfg.max_retries} in {delay:.1f}s", file=sys.stderr)
        _sleep(delay)
    raise AssertionError("unreachable")


def _reply_from_response(response: Any, keep_reasoning: bool) -> LLMReply:
    extra = getattr(response, "model_extra", None) or {}
    if not response.choices:                          # MiniMax may answer HTTP 200 + base_resp error
        base = extra.get("base_resp") or {}
        raise _code_error(base.get("status_code"), base.get("status_msg") or "empty response")
    choice = response.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning_content", None)
    details = getattr(msg, "reasoning_details", None)       # older MiniMax field name
    if not reasoning and isinstance(details, list):
        reasoning = "\n".join(str(d.get("text", "")) for d in details if isinstance(d, dict))
    calls = [{"id": tc.id, "type": "function",
              "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
             for tc in (msg.tool_calls or [])]
    usage = {"prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
             "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0}
    return _make_reply(msg.content, reasoning, calls, choice.finish_reason, usage, keep_reasoning)


def _code_error(code: int | None, message: str, status: int | None = None) -> LLMError:
    if code in FATAL_CODES:
        return LLMError(f"MiniMax error {code}: {FATAL_CODES[code]} ({message})", code=code, status=status)
    retryable = code in RETRY_CODES or (status is not None and (status in (408, 409, 429) or status >= 500))
    label = f"MiniMax error {code}" if code else f"HTTP {status}" if status else "LLM error"
    return LLMError(f"{label}: {message}", code=code, status=status, retryable=retryable)


def _as_llm_error(exc: Exception, cfg: LLMConfig) -> LLMError:
    import openai

    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.APITimeoutError):
        return LLMError(f"request timed out ({cfg.provider})", retryable=True)
    if isinstance(exc, openai.APIConnectionError):
        return LLMError(f"cannot reach {cfg.base_url} - check your network / base URL", retryable=True)
    if isinstance(exc, openai.APIStatusError):
        body = exc.body if isinstance(exc.body, dict) else {}
        message = str(body.get("message") or (body.get("error") or {}).get("message", "") or exc.message)
        err = _code_error(minimax_code(exc), message, exc.status_code)
        if exc.status_code in (401, 403) and err.code is None:
            err = LLMError(f"HTTP {exc.status_code}: {message} - check {API_KEYS.get(cfg.provider, 'your key')}",
                           status=exc.status_code)
        try:
            err.retry_after = float(exc.response.headers.get("retry-after"))
        except (TypeError, ValueError, AttributeError):
            pass
        return err
    return LLMError(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ mock + replay providers
_MOCK: dict = {"responder": None, "turn": 0}


def set_mock_responder(fn: Callable[[list, list | None], Any] | None) -> None:
    """Script the mock provider: fn(messages, tools) -> str (answer text) | dict | list[dict].

    A dict {"tool": name, "args": {...}} (or a list of them) is a tool call; any other dict is
    returned as JSON text. None restores the default ("OK (mock LLM)", or "{}" in JSON mode).
    """
    _MOCK["responder"], _MOCK["turn"] = fn, 0


def _mock_turn(messages: list[dict], tools: list[dict] | None, json_mode: bool) -> LLMReply:
    responder = _MOCK["responder"]
    out = responder(messages, tools) if responder else ("{}" if json_mode else MOCK_TEXT)
    _MOCK["turn"] += 1
    if isinstance(out, dict) and "tool" in out:
        out = [out]
    if isinstance(out, list):
        calls = [{"id": f"call_mock_{_MOCK['turn']}_{i}", "type": "function",
                  "function": {"name": c["tool"], "arguments": json.dumps(c.get("args", {}))}}
                 for i, c in enumerate(out)]
        return _make_reply("", None, calls, "tool_calls", {"prompt_tokens": 0, "completion_tokens": 0})
    text = json.dumps(out) if isinstance(out, dict) else str(out)
    return _make_reply(text, None, [], "stop", {"prompt_tokens": 0, "completion_tokens": 0})


_CASSETTES: dict = {}


def _replay_turn(messages: list[dict], tools: list[dict] | None) -> LLMReply:
    """Replay a recorded response: same request hash first, else the next unused entry in order."""
    path = os.getenv("LLM_CASSETTE", "")
    if not Path(path).is_file():
        raise LLMError(f"LLM_PROVIDER=replay needs LLM_CASSETTE=<recorded .jsonl>; not found: {path!r}")
    with _LOCK:
        tape = _CASSETTES.get(path)
        if tape is None:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            tape = _CASSETTES[path] = {"entries": [json.loads(x) for x in lines if x.strip()], "used": set()}
        key = _request_key(messages, tools)
        free = [i for i in range(len(tape["entries"])) if i not in tape["used"]]
        match = next((i for i in free if tape["entries"][i].get("key") == key), free[0] if free else None)
        if match is None:
            raise LLMError(f"cassette {path} is exhausted ({len(tape['entries'])} responses)")
        tape["used"].add(match)
    return _reply_from_dict(tape["entries"][match]["reply"], "minimax")


def _reply_to_dict(reply: LLMReply) -> dict:
    return {"content": reply.message.get("content", ""), "reasoning": reply.message.get("reasoning_content"),
            "think": reply.reasoning, "tool_calls": reply.tool_calls, "finish_reason": reply.finish_reason,
            "usage": reply.usage}


def _reply_from_dict(d: dict, provider: str) -> LLMReply:
    return _make_reply(d.get("content"), d.get("reasoning"), d.get("tool_calls") or [], d.get("finish_reason", "stop"),
                       d.get("usage") or {}, keep_reasoning=provider == "minimax")


def _cache_file(cfg: LLMConfig, messages: list, tools: list | None, effort: str) -> Path:
    key = _request_key(cfg.provider, cfg.model, effort, messages, tools)
    return Path(os.getenv("LLM_CACHE_DIR", ".cache/llm")) / f"{key}.json"


_LOG_LOCK = threading.Lock()


def _append_jsonl(path: Path, record: dict) -> None:
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _log_call(cfg: LLMConfig, stack: tuple, messages: list, tools: list | None, reply: LLMReply | None,
              error: Exception | None, seconds: float) -> None:
    log_dir = os.getenv("LLM_LOG_DIR")
    if not log_dir:
        return
    try:
        _write_log(log_dir, cfg, stack, messages, tools, reply, error, seconds)
    except (OSError, TypeError, ValueError) as exc:      # logging must never break a call
        if not _LOG_WARNED:
            _LOG_WARNED.append(True)
            print(f"[llm] could not write the LLM log in {log_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)


_LOG_WARNED: list = []


def log_file_name(label: str) -> str:
    """Safe, short JSONL file name for a log label (letters of any script kept, max 60 chars)."""
    stem = re.sub(r"[^\w.-]", "_", str(label)).strip("._")[:60]
    return f"{stem or 'llm'}.jsonl"


def _write_log(log_dir: str, cfg: LLMConfig, stack: tuple, messages: list, tools: list | None,
               reply: LLMReply | None, error: Exception | None, seconds: float) -> None:
    label = next((p.label for p in reversed(stack) if p.label), "") or f"llm_{os.getpid()}"
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "llm": f"{cfg.provider}:{cfg.model}",
              "label": label, "mode": _mode(stack), "phases": [p.phase for p in stack if p.phase],
              "seconds": round(seconds, 3), "messages": messages, "tools": [t["function"]["name"] for t in tools or []],
              "reply": _reply_to_dict(reply) if reply else None, "cached": bool(reply and reply.cached),
              "error": str(error) if error else None}
    _append_jsonl(Path(log_dir) / log_file_name(label), record)


# ------------------------------------------------------------------ the student-facing API
def chat(prompt: str | list[dict], system: str | None = None, *, model: str | None = None,
         effort: str | None = None, json_mode: bool = False, max_tokens: int | None = None,
         cache: bool = False, timeout: float | None = None) -> str:
    """Ask once, get the answer text (thinking removed; see last_reasoning())."""
    return complete(_messages(prompt, system), model=model, effort=effort, json_mode=json_mode,
                    max_tokens=max_tokens, cache=cache, timeout=timeout).text


def chat_json(prompt: str | list[dict], system: str | None = None, *, tools: list[Callable] | None = None,
              max_rounds: int = 3, **kw: Any) -> dict:
    """Ask for a JSON object and parse it robustly; one retry with a 'ONLY valid JSON' nudge.

    With `tools=[...]` the model may call your functions first (see ask_with_tools).
    Raises LLMError if no valid JSON comes back.
    """
    messages = _messages(prompt, system)
    if tools:
        run = ask_with_tools(messages, tools, max_rounds=max_rounds, json_mode=True, **kw)
        text, history = run.final_text, run.messages
    else:
        reply = complete(messages, json_mode=True, **kw)
        text, history = reply.text, [*messages, reply.message]
    try:
        return extract_json(text)
    except ValueError as exc:
        problem = str(exc)
    nudge = {"role": "user", "content": f"That was not valid JSON ({problem}). {JSON_RULE}"}
    reply = complete([*history, nudge], json_mode=True, **kw)
    try:
        return extract_json(reply.text)
    except ValueError as exc:
        raise LLMError(f"the model did not return valid JSON after a retry: {reply.text[:200]!r}") from exc


@dataclass
class ToolRun:
    final_text: str              # the model's last answer (after all tool calls)
    calls: list[dict]            # [{"name", "args", "result"}] in the order they ran
    rounds: int                  # model turns used (= LLM calls)
    messages: list[dict]         # the full conversation, ready to continue
    stopped: str = ""            # "" = the model answered | "rounds" | "budget" | "time" (stopped early)


def ask_with_tools(prompt: str | list[dict], tools: list[Callable], system: str | None = None, *,
                   max_rounds: int = 8, **kw: Any) -> ToolRun:
    """Let the model call your Python functions until it answers in plain text.

    Each function's name, type hints and docstring become its tool schema (function_schema).
    Every round is one LLM call. If the policy budget or the plan()/reflect() deadline runs out after
    the first round, the loop stops and returns what it has (`run.stopped` = "budget" | "time"; the
    tool results are in run.calls). A refused FIRST call raises as usual (LLMError -> your fallback).
    In official plan(): budget 2 calls -> one tool round + the answer. Before the last round (when tools
    were called) the last tool result tells the model to answer now, in the format the task asked for
    (FINAL_ROUND_NUDGE).
    """
    by_name = {fn.__name__: fn for fn in tools}
    schemas = [function_schema(fn) for fn in tools]
    messages = _messages(prompt, system)
    run = ToolRun(final_text="", calls=[], rounds=0, messages=messages)
    for k in range(max_rounds):
        if k > 0 and k == max_rounds - 1 and run.calls and messages[-1].get("role") == "tool":
            # Last round after tool calls: say so (in the last tool result, so the conversation keeps its
            # shape), or the model may spend it on one more lookup and never answer (seen live with
            # max_rounds=2: two tool rounds, no final answer).
            messages[-1] = {**messages[-1], "content": f"{messages[-1]['content']}\n\n{FINAL_ROUND_NUDGE}"}
        try:
            reply = complete(messages, tools=schemas, **kw)
        except (LLMBudgetExceeded, LLMNotAllowed, LLMDeadline) as exc:
            if run.rounds == 0:
                raise
            run.stopped = "time" if isinstance(exc, LLMDeadline) else "budget"
            return run
        run.rounds += 1
        messages.append(reply.message)          # the WHOLE assistant message (reasoning + tool_calls)
        run.final_text = reply.text
        if not reply.tool_calls:                # no tool requested: that was the final answer
            return run
        for call in reply.tool_calls:           # the model may request several calls at once
            name = call["function"]["name"]
            args, result = _run_tool(by_name, name, call["function"]["arguments"])
            run.calls.append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _as_text(result)})
    run.stopped = "rounds"
    return run                                  # out of rounds: final_text is the last thing it said


FINAL_ROUND_NUDGE = ("This is your last turn: do not call any more tools. Give your final answer now, "
                     "in exactly the format the task asked for.")


def _run_tool(by_name: dict, name: str, arguments: str) -> tuple[dict, Any]:
    """Run one requested call. Problems become error text the model can read and fix."""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"error: arguments are not valid JSON ({exc})"
    if name not in by_name:
        return args, f"error: unknown tool {name!r}; available: {sorted(by_name)}"
    try:
        return args, by_name[name](**args)
    except Exception as exc:
        return args, f"error: {type(exc).__name__}: {exc}"


def _as_text(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


# ------------------------------------------------------------------ pre-class check (CLI `llm-check`)
def llm_smoke_test(effort: str | None = None) -> list[dict]:
    """Three tiny real calls: plain chat, chat_json, one tool round trip with 2 tools.

    Returns one row per step: {"step", "ok", "seconds", "tokens", "detail"}. Never raises and
    never includes the API key. With LLM_PROVIDER=mock it runs offline (scripted answers).
    """
    def corner_speed(radius: float) -> float:
        """Fastest speed (u/s) through a corner of this radius (grip 20 u/s^2)."""
        return round(math.sqrt(20.0 * radius), 2)

    def braking_distance(v_from: float, v_to: float) -> float:
        """Distance (u) to brake from v_from to v_to at 25 u/s^2."""
        return round(max(0.0, (v_from ** 2 - v_to ** 2) / 50.0), 2)

    mock = get_config().provider == "mock"
    if mock:
        set_mock_responder(_smoke_mock_responder)
    steps = [
        ("chat", lambda: chat("Say hi to a race driver in five words.", effort=effort)),
        ("chat_json", lambda: chat_json('Return {"throttle": <float -1..1>, "steering": <float -1..1>} for a '
                                        "car at speed 10 with the next waypoint 30 degrees to the RIGHT "
                                        "(+ = right).", effort=effort)),
        ("ask_with_tools", lambda: ask_with_tools(
            "Use the tools: corner speed for radius 45, and braking distance from 40 to that speed. "
            "Then answer in one sentence.", [corner_speed, braking_distance], effort=effort, max_rounds=4)),
    ]
    rows = []
    try:
        for name, fn in steps:
            before, t0 = llm_stats(), time.perf_counter()
            try:
                out = fn()
                if isinstance(out, ToolRun):
                    ok = len(out.calls) >= 1 and bool(out.final_text)
                    detail = f"{out.rounds} rounds, tools {[c['name'] for c in out.calls]}"
                elif isinstance(out, dict):
                    ok = isinstance(out.get("throttle"), (int, float)) and isinstance(out.get("steering"), (int, float))
                    detail = json.dumps(out)[:80]
                else:
                    ok, detail = bool(out.strip()), f"{len(out)} chars"
            except Exception as exc:                 # report, do not crash the checker
                ok, detail = False, f"{type(exc).__name__}: {exc}"[:200]
            after = llm_stats()
            tokens = (after["prompt_tokens"] + after["completion_tokens"]
                      - before["prompt_tokens"] - before["completion_tokens"])
            rows.append({"step": name, "ok": ok, "seconds": round(time.perf_counter() - t0, 2),
                         "tokens": tokens, "detail": detail})
    finally:
        if mock:
            set_mock_responder(None)
    return rows


def _smoke_mock_responder(messages: list, tools: list | None) -> Any:
    if tools and not any(m.get("role") == "tool" for m in messages):
        return [{"tool": "corner_speed", "args": {"radius": 45}},
                {"tool": "braking_distance", "args": {"v_from": 40, "v_to": 30}}]
    if tools:
        return "Take it at 30, start braking 14 units early."
    if "JSON" in str(messages[0].get("content", "")):
        return {"throttle": 0.5, "steering": 0.4}
    return MOCK_TEXT


# ------------------------------------------------------------------ LangChain (LangGraph example only)
def get_langchain_model(*, model: str | None = None, effort: str | None = None, max_tokens: int | None = None,
                        native: bool = False) -> Any:
    """A LangChain chat model (supports .invoke and .bind_tools) for the same provider.

    Default: an adapter that routes through `complete()`, so policy, budgets, logs, cache, the mock and
    MiniMax quirks all apply. `native=True` returns langchain_openai.ChatOpenAI configured per MiniMax
    docs instead (bypasses the policy and drops reasoning_content between turns).
    """
    if native and (official_session_active() or _mode(_stack()) == "official"):
        raise LLMNotAllowed("get_langchain_model(native=True) bypasses the race rules - not allowed in "
                            "official events (the default adapter follows them)")
    cfg = _config(model)
    if native and cfg.provider not in ("mock", "replay"):
        from langchain_openai import ChatOpenAI

        kw: dict = {"model": cfg.model, "base_url": cfg.base_url, "api_key": cfg.api_key or "missing",
                    "max_tokens": max_tokens or MAX_TOKENS, "timeout": TIMEOUT, "max_retries": 2}
        if cfg.provider == "minimax" and cfg.model.lower().startswith("minimax-m3.1"):
            wanted = normalize_effort(effort or cfg.effort)
            kw["reasoning_effort"] = wanted if wanted != "none" else "low"
        elif cfg.provider == "minimax":
            kw["extra_body"] = {"reasoning_split": True}
            if normalize_effort(effort or cfg.effort) == "none":
                kw["extra_body"]["thinking"] = {"type": "disabled"}
        return ChatOpenAI(**kw)
    return _langchain_adapter()(model_id=model, effort=effort, max_tokens=max_tokens)


_ADAPTER: list = []


def _langchain_adapter() -> type:
    if _ADAPTER:
        return _ADAPTER[0]
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, convert_to_openai_messages
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    class AgpChatModel(BaseChatModel):
        """LangChain wrapper around agentic_gp.llm.complete()."""

        model_id: str | None = None
        effort: str | None = None
        max_tokens: int | None = None

        @property
        def _llm_type(self) -> str:
            return "agentic_gp"

        def bind_tools(self, tools: list, **kwargs: Any) -> Any:
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

        def _generate(self, messages: list, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
            reply = complete(convert_to_openai_messages(messages), tools=kwargs.get("tools"), model=self.model_id,
                             effort=self.effort, max_tokens=self.max_tokens)
            calls = []
            for c in reply.tool_calls:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"name": c["function"]["name"], "args": args, "id": c["id"], "type": "tool_call"})
            extra = {"reasoning_content": reply.reasoning} if reply.reasoning else {}
            msg = AIMessage(content=reply.text, tool_calls=calls, additional_kwargs=extra)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    _ADAPTER.append(AgpChatModel)
    return AgpChatModel

