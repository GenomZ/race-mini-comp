"""A local LLM relay that keeps the API key out of driver processes.

    from agentic_gp.race.llm_proxy import child_llm_env, revoke
    env = child_llm_env(env)       # parent, before it starts an isolated driver
    ...                            # the child's agentic_gp.llm talks to http://127.0.0.1:<port> with a token
    revoke(env)                    # when that driver closes

Why: a driver file runs in a child process, and anything in that process (os.environ, the llm
module, /proc/self/environ) is within reach of a clever enough file. So the child never gets the
key. The parent runs a tiny OpenAI-compatible relay on 127.0.0.1: it accepts POST /chat/completions
with a per-driver random token, swaps the token for the real key and forwards the request to the
provider. A stolen token is worth nothing outside this machine and dies with its driver. Each token
also has a hard call cap (AGP_LLM_PROXY_MAX_CALLS, default 200), a backstop behind the race rules
that agentic_gp.llm enforces in the child.
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .. import llm

PROXY_URL_VAR = "AGP_LLM_PROXY_URL"
PROXY_TOKEN_VAR = "AGP_LLM_PROXY_TOKEN"
MAX_CALLS = int(os.getenv("AGP_LLM_PROXY_MAX_CALLS", "200"))
UPSTREAM_TIMEOUT_S = 300.0
MAX_BODY_BYTES = 8_000_000
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"
FORWARD_HEADERS = ("Retry-After", "retry-after")


@dataclass
class _Grant:
    base_url: str
    api_key: str
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Relay(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.grants: dict[str, _Grant] = {}
        self.url = f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    server: _Relay

    def log_message(self, *args: Any) -> None:    # quiet: the llm module logs the calls itself
        pass

    def _answer(self, status: int, body: bytes, content_type: str = "application/json",
                headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._answer(status, json.dumps({"error": {"message": message, "type": "agp_proxy"}}).encode())

    def do_GET(self) -> None:
        self._error(404, "only POST /chat/completions is relayed")

    def do_POST(self) -> None:
        token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        grant = self.server.grants.get(token)
        if grant is None:
            return self._error(401, "unknown or revoked driver token")
        if self.path.rstrip("/") != "/chat/completions":
            return self._error(404, "only POST /chat/completions is relayed")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            return self._error(413, "request body missing or too large")
        with grant.lock:
            if grant.calls >= MAX_CALLS:
                return self._error(403, f"this driver used its {MAX_CALLS} relayed LLM calls")
            grant.calls += 1
        body = self.rfile.read(length)
        req = urllib.request.Request(grant.base_url.rstrip("/") + "/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {grant.api_key}",
                                              "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as r:
                self._answer(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            extra = {h: exc.headers[h] for h in FORWARD_HEADERS if exc.headers.get(h)}
            self._answer(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"), extra)
        except Exception as exc:                  # noqa: BLE001 - network trouble upstream
            self._error(502, f"relay cannot reach the provider: {type(exc).__name__}")


_RELAY: _Relay | None = None
_LOCK = threading.Lock()


def _relay() -> _Relay:
    global _RELAY
    with _LOCK:
        if _RELAY is None:
            _RELAY = _Relay()
            threading.Thread(target=_RELAY.serve_forever, name="agp-llm-relay", daemon=True).start()
        return _RELAY


def child_llm_env(env: dict | None) -> dict:
    """The child's extra environment: the parent's provider settings with the key replaced by a relay token.

    Without a key (mock, replay, ollama, or no key set) nothing is relayed and the env comes back as is,
    plus LLM_NO_DOTENV=1 so the child never reads .env itself.
    """
    out = {**(env or {}), "LLM_NO_DOTENV": "1"}
    out.pop(PROXY_URL_VAR, None)
    out.pop(PROXY_TOKEN_VAR, None)
    llm.load_env()                                # the child inherits the non-secret LLM_* settings
    provider = str(out.get("LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "minimax")).strip().lower()
    key_var = llm.API_KEYS.get(provider)
    for var in llm.API_KEYS.values():
        out.pop(var, None)
    key = (env or {}).get(key_var) if key_var else None
    key = key or (llm.sealed_key(key_var) if key_var else None)
    if not key:
        return out
    url_var, default_url = llm.BASE_URLS.get(provider, (None, None))
    base_url = ((out.get(url_var) if url_var else None) or (os.getenv(url_var) if url_var else None) or default_url
                or OPENAI_DEFAULT_URL)            # no base URL: the OpenAI SDK's own default
    relay = _relay()
    token = "agp-" + secrets.token_urlsafe(24)
    relay.grants[token] = _Grant(base_url=base_url, api_key=key)
    out[PROXY_URL_VAR], out[PROXY_TOKEN_VAR] = relay.url, token
    return out


def revoke(env: dict | None) -> None:
    """Invalidate the token handed out in this env (the driver is closed)."""
    token = (env or {}).get(PROXY_TOKEN_VAR)
    if token and _RELAY is not None:
        _RELAY.grants.pop(token, None)
