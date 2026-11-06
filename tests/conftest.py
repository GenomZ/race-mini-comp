"""Shared test setup: never touch a real LLM, never read the real .env, never write into the repo.

Live MiniMax tests are opt-in: AGP_LIVE_LLM=1 python -m pytest -m llm  (they read .env themselves).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:                  # `pytest` from any folder finds agentic_gp
    sys.path.insert(0, str(REPO))

# Set before any test module imports agentic_gp (spawned child processes inherit os.environ too).
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_NO_DOTENV"] = "1"
os.environ["LLM_QUIET"] = "1"
for _var in ("LLM_LOG_DIR", "LLM_RECORD", "LLM_CASSETTE"):
    os.environ.pop(_var, None)


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch, tmp_path):
    """Every test: mock provider, private LLM cache, no scripted mock replies left over."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_NO_DOTENV", "1")
    monkeypatch.setenv("LLM_QUIET", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "llm_cache"))
    for var in ("LLM_LOG_DIR", "LLM_RECORD", "LLM_CASSETTE"):
        monkeypatch.delenv(var, raising=False)
    yield
    from agentic_gp import llm
    llm.set_mock_responder(None)
    llm._reset_sessions()                      # no official session rules leak into the next test


@pytest.fixture
def repo() -> Path:
    """The repository root (example drivers live in repo / "drivers" / "examples")."""
    return REPO
