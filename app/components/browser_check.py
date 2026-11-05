"""Real-browser smoke check for Streamlit pages (Playwright + headless Chromium).

    python app/components/browser_check.py app/_components_demo.py --out shots
    python app/components/browser_check.py app/race_control.py --pages replay live --play
    python app/components/browser_check.py races/demo/html/class_demo.html --play     # standalone player

What it does: starts `streamlit run <script>` headless on a free port (not for .html files),
opens the page (and each `--pages` url path), waits until Streamlit is idle, takes a full-page
screenshot, optionally presses Play on the first replay canvas and screenshots again 2 s later
(the cars must have moved: it compares canvas pixels), collects browser console errors and
warnings, prints a JSON summary, and stops the server.

One-time setup:
    uv pip install --python .venv/Scripts/python.exe playwright
    .venv/Scripts/python.exe -m playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CANVAS_HASH = """(idx) => {
  const found = [];
  const walk = (root) => root.querySelectorAll('*').forEach((e) => {
    if (e.shadowRoot) walk(e.shadowRoot);
    if (e.tagName === 'CANVAS' && e.closest('.agp-stage')) found.push(e);
  });
  walk(document);
  const c = found[idx]; if (!c || !c.width) return null;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let h = 0; for (let i = 0; i < d.length; i += 97) h = (h * 31 + d[i]) | 0; return h;
}"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(port: int, timeout: float = 180.0) -> bool:
    import urllib.request

    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=3) as r:
                if r.read().decode().strip() == "ok":
                    return True
        except OSError:
            time.sleep(1.0)
    return False


def _wait_idle(pg, timeout_ms: int = 60_000) -> None:
    """Wait until the page rendered elements and Streamlit's "running" indicator stayed away for 1 s.
    (A page with a run_every fragment is never fully idle: then we give up after timeout_ms.)"""
    pg.wait_for_selector('[data-testid="stElementContainer"]', timeout=180_000, state="attached")
    end, quiet_since = time.time() + timeout_ms / 1000, None
    while time.time() < end:
        busy = pg.locator('[data-testid="stStatusWidget"]').count() > 0
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since >= 1.0:
            return
        pg.wait_for_timeout(200)


def check(target: str, out: Path, pages: list[str], play: bool, width: int, height: int,
          on_done=None) -> dict:
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    server = None
    is_html = target.lower().endswith(".html")
    if is_html:
        urls = {"page": Path(target).resolve().as_uri()}
    else:
        port = _free_port()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        server = subprocess.Popen([sys.executable, "-m", "streamlit", "run", target, "--server.headless", "true",
                                   "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                  cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not _wait_health(port):
            server.kill()
            return {"error": "streamlit did not start"}
        urls = {"home": f"http://localhost:{port}/"} | {p: f"http://localhost:{port}/{p}" for p in pages}
    report: dict = {"target": target, "pages": {}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height})
            logs: list[dict] = []
            page.on("console", lambda m: logs.append({"type": m.type, "text": m.text[:300]}))
            page.on("pageerror", lambda e: logs.append({"type": "pageerror", "text": str(e)[:300]}))
            for name, url in urls.items():
                logs.clear()
                t0 = time.time()
                page.goto(url, wait_until="domcontentloaded", timeout=180_000)
                if not is_html:
                    _wait_idle(page)
                page.wait_for_timeout(1500)
                info = {"load_s": round(time.time() - t0, 2)}
                shot = out / f"{name}.png"
                page.screenshot(path=str(shot), full_page=True)
                info["screenshot"] = str(shot)
                if play:
                    try:
                        page.locator(".agp-play").first.wait_for(timeout=20_000)
                    except Exception:                # noqa: BLE001 - no replay canvas on this page
                        pass
                if play and page.locator(".agp-play").count():
                    h0 = page.evaluate(CANVAS_HASH, 0)
                    if "Pause" not in page.locator(".agp-play").first.inner_text():   # autoplay pages already run
                        page.locator(".agp-play").first.click()
                    page.wait_for_timeout(2000)
                    h1 = page.evaluate(CANVAS_HASH, 0)
                    shot2 = out / f"{name}_played.png"
                    page.screenshot(path=str(shot2), full_page=True)
                    info.update(played_screenshot=str(shot2), cars_moved=h0 != h1)
                info["console"] = [l for l in logs if l["type"] in ("error", "warning", "pageerror")
                                   and "willReadFrequently" not in l["text"]]       # caused by our pixel check
                report["pages"][name] = info
            # Playwright's browser.close() can hang on Windows: stop the server, report, and exit hard.
            _stop(server)
            if on_done is not None:
                on_done(report)                  # (main() prints and calls os._exit)
            browser.close()
    finally:
        _stop(server)
    return report


def _stop(server) -> None:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Screenshot Streamlit pages / HTML players in headless Chromium.")
    ap.add_argument("target", help="a Streamlit script (app/race_control.py) or a standalone .html player")
    ap.add_argument("--pages", nargs="*", default=[], help="url paths of st.navigation pages to visit too")
    ap.add_argument("--out", default="browser_shots", help="screenshot folder (default ./browser_shots)")
    ap.add_argument("--play", action="store_true", help="press Play on the first replay canvas, compare pixels")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args(argv)
    def done(rep: dict) -> None:
        print(json.dumps(rep, indent=1), flush=True)
        bad = rep.get("error") or any(e["type"] in ("error", "pageerror") for p in rep.get("pages", {}).values()
                                      for e in p.get("console", []))
        os._exit(1 if bad else 0)

    done(check(args.target, Path(args.out), args.pages, args.play, args.width, args.height, on_done=done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
