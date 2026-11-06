"""Upload hardening: static-check bypasses, the runtime race guard, the LLM relay and the Inbox password.

    .venv/Scripts/python.exe -m pytest -q tests/test_guard.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from agentic_gp.race import load_driver
from agentic_gp.race.loader import DriverLoadError, check_driver_source

HEAD = "NAME = 'Probe'\nAUTHOR = 'x'\n"
DRIVE = "def drive(obs):\n    return 1.0, 0.0\n"


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(HEAD + src, encoding="utf-8")
    return p


@pytest.mark.parametrize("src, needle", [
    ("import os\nf = os.__getattribute__('sy' + 'stem')\n", "__getattribute__"),
    ("x = getattr(object, '__getattribute__')\n", "__getattribute__"),
    ("from pathlib import Path\nt = Path('.env').read_text()\n", "file I/O"),
    ("t = open('.env').read()\n", "file I/O"),
    ("import numpy as np\na = np.load('x.npy')\n", "file I/O"),
    ("from numpy import load\n", "file I/O"),
    ("import pickle\n", "pickle"),
    ("import gc\n", "gc"),
    ("import inspect\n", "inspect"),
    ("from operator import attrgetter\n", "attrgetter"),
    ("import sys\nf = sys._getframe()\n", "_getframe"),
    ("g = globals()\n", "globals()"),
])
def test_static_check_closes_known_bypasses(tmp_path, src, needle):
    errors = [i.message for i in check_driver_source(_write(tmp_path, "d.py", src + DRIVE)) if i.level == "error"]
    assert any(needle in e for e in errors), errors


def _drive_isolated(path: Path):
    drv = load_driver(path, isolate=True, check=False)       # skip the static check: test the guard itself
    try:
        return drv.drive({"speed": 0.0})
    finally:
        drv.close()


@pytest.mark.parametrize("body, needle", [
    ("import os\ndef drive(obs):\n    os.__getattribute__('system')('echo hi')\n    return 1.0, 0.0\n", "os.system"),
    ("def drive(obs):\n    open(SECRET).read()\n    return 1.0, 0.0\n", "may not read"),
    ("def drive(obs):\n    open(OUT, 'w').write('x')\n    return 1.0, 0.0\n", "may not write"),
    ("import socket\ndef drive(obs):\n    socket.create_connection(('127.0.0.1', 9), timeout=2)\n    return 1.0, 0.0\n",
     "network"),
])
def test_race_guard_blocks_at_run_time(tmp_path, body, needle):
    secret = tmp_path / "secret.env"
    secret.write_text("MINIMAX_API_KEY=sk-not-for-drivers\n", encoding="utf-8")
    consts = f"SECRET = {str(secret)!r}\nOUT = {str(tmp_path / 'out.txt')!r}\n"
    with pytest.raises(Exception, match=needle):
        _drive_isolated(_write(tmp_path, "d.py", consts + body))
    assert not (tmp_path / "out.txt").exists()


def test_race_guard_blocks_at_import(tmp_path):
    p = _write(tmp_path, "d.py", "import os\nos.__getattribute__('system')('echo hi')\n" + DRIVE)
    with pytest.raises(DriverLoadError, match="race guard"):
        load_driver(p, isolate=True, check=False)


def test_a_normal_driver_still_drives_under_the_guard(tmp_path):
    p = _write(tmp_path, "d.py", "import numpy as np\nfrom agentic_gp import driverkit as dk\n"
                                  "def drive(obs):\n    return 0.5, float(np.clip(2.0, -1, 1))\n")
    action = _drive_isolated(p)
    assert action.throttle == 0.5 and action.steering == 1.0


class _FakeProvider(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeProvider.seen.append({"auth": self.headers.get("Authorization"), "path": self.path, "body": body})
        out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body.get("model", "m"),
                          "choices": [{"index": 0, "finish_reason": "stop",
                                       "message": {"role": "assistant", "content": "relayed hello"}}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def test_llm_relay_keeps_the_key_out_of_the_driver_process(tmp_path, monkeypatch):
    pytest.importorskip("openai")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProvider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeProvider.seen.clear()
    monkeypatch.setenv("LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-real-secret-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    src = ("import os\nfrom agentic_gp import llm\nKIND = 'llm'\n"
           "class Driver:\n"
           "    def plan(self, track):\n"
           "        self.answer = llm.chat('hi')\n"
           "        self.env = [k for k, v in os.environ.items() if 'sk-real' in v]\n"
           "    def drive(self, obs):\n"
           "        return {'throttle': 1.0, 'steering': 0.0, 'say': f'{self.answer}|{self.env}'}\n")
    drv = load_driver(_write(tmp_path, "d.py", src), isolate=True, check=False)
    try:
        drv.plan({"name": "t"}, timeout=60)
        say = drv.drive({"speed": 0.0}).say
    finally:
        drv.close()
        server.shutdown()
    assert say == "relayed hello|[]"                                    # the call worked, no key in the child
    assert [s["auth"] for s in _FakeProvider.seen] == ["Bearer sk-real-secret-key"]   # the relay added it


def test_inbox_needs_the_instructor_password(monkeypatch, tmp_path):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("AGP_SUBMISSIONS_DIR", str(tmp_path / "subs"))
    monkeypatch.setenv("AGP_INSTRUCTOR_PASSWORD", "pit-wall")
    entry = str(Path(__file__).resolve().parents[1] / "app" / "race_control.py")
    at = st_testing.AppTest.from_file(entry, default_timeout=60).run()
    at.switch_page("views/inbox.py").run()
    assert not at.exception and not any("Upload" in s.value for s in at.subheader)    # locked
    at.text_input[0].input("pit-wall")
    at.get("form_submit_button")[0].click().run()
    assert any("Upload" in s.value for s in at.subheader)


def test_inbox_is_locked_on_a_host_without_a_password(monkeypatch, tmp_path):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.delenv("AGP_INSTRUCTOR_PASSWORD", raising=False)
    monkeypatch.setenv("RAILWAY_PROJECT_ID", "p1")
    entry = str(Path(__file__).resolve().parents[1] / "app" / "race_control.py")
    at = st_testing.AppTest.from_file(entry, default_timeout=60).run()
    at.switch_page("views/inbox.py").run()
    assert any("AGP_INSTRUCTOR_PASSWORD" in e.value for e in at.error)
    assert not any("Upload" in s.value for s in at.subheader)
