# Handover: Agentic Grand Prix 2.0

Written 2026-11-09 for the next agent or developer. Read this first, then [ARCHITECTURE.md](ARCHITECTURE.md)
and the [documentation index](README.md). Section 4 lists the remaining work as staged tasks. Each task says
what to change, where, how to check it, and whether it needs the owner's decision first.

Owner: Krzysztof Buzar (GitHub `GenomZ/race-mini-comp`).

---

## 1. Where things stand

| Item | State |
|---|---|
| 2.0 restructure plus hardened uploads | Merged to `main` as [PR #1](https://github.com/GenomZ/race-mini-comp/pull/1), commit `2c324b5`. All 12 CI checks passed on Ubuntu. |
| Sept 15 v1 agents (fruit fly, `my_agent.py` 68.50 s, ...) | Archived unchanged in [`legacy/v1/`](../legacy/v1/README.md). They do not run on 2.0 and are outside the tests. |
| Unity WebGL viewer for Railway | Built and tested locally. **Not committed, not deployed** (Task 1 and Task 2). |
| Lesson audit, Ch0-Ch8 and the Streamlit block | All numbers reproduce. Small text fixes and missing cassettes are still open (Task 3 and Task 4). |
| Flaky timing tests on a busy Windows machine | Open (Task 5). They pass in CI. |

**Local checkout at handover:**
- It is on `main` at `2c324b5`, up to date with `origin/main`. The local `agentic-gp-2.0` branch is deleted; the remote one can go too.
- `unity/` and `deploy/` are untracked: this is the WebGL work.
- This file and its link in `docs/README.md` are staged but not committed.
- A Docker container `agp-viewer` may still be serving the viewer on <http://127.0.0.1:8091/?replay=replays/demo.json>.

---

## 2. What was done (and why)

**2.0 restructure** (PR #1):
- package split: `agentic_gp/{sim,race,viz,fly}`
- `race.py` CLI: `check`, `drive`, `watch`, `doctor`, `leaderboard`, Grand Prix
- driver ladder in `drivers/examples` (L0-L6, plus fly drivers) and the student `drivers/template.py`
- Race Control Streamlit app in `app/`
- lesson chapters with checkpoint handrails in `lesson/`
- instructor guide and autograder in `lesson/instructor/`
- Streamlit mini-lessons in `streamlit_lessons/`

**Hardened student uploads.** Students upload driver `.py` files through the Race Control **Inbox** page. Four
layers protect the server, from cheapest to deepest:

1. **Static scrutineering**, `check_driver_source` in [`agentic_gp/race/loader.py`](../agentic_gp/race/loader.py).
   - It rejects denied imports (`os`, `subprocess`, `socket`, `pickle`, `marshal`, `gc`, `inspect`, ...).
   - It rejects dynamic tricks: `eval`/`exec`, `globals()`, `getattr` with a literal dunder name, `__getattribute__`, `__subclasses__`, frame attributes, `operator.attrgetter`.
   - It rejects **all file I/O**: `open()`, `Path.read_text`, `np.load`, ... `open()` used to be a warning and is now an error. The lessons and `drivers/submissions/README.md` say so.
2. **The API key never enters a driver process**, [`agentic_gp/race/llm_proxy.py`](../agentic_gp/race/llm_proxy.py).
   - The race process runs an OpenAI-compatible relay on `127.0.0.1:<random port>`.
   - Each driver gets a throwaway token through `AGP_LLM_PROXY_URL` / `AGP_LLM_PROXY_TOKEN`. The relay swaps it for the real key.
   - Each token is capped at `AGP_LLM_PROXY_MAX_CALLS` calls (default 200) and revoked when the driver exits.
   - `llm.api_keys_hidden()` removes the keys from `os.environ` while the child process starts, and the child has `LLM_NO_DOTENV=1`.
3. **Runtime guard**, [`agentic_gp/race/guard.py`](../agentic_gp/race/guard.py). A PEP 578 audit hook installed in every driver child process before the driver is imported. It blocks:
   - starting processes and `ctypes` calls (`dlopen`/`dlsym` are allowed because `truststore` needs them)
   - network connections other than the relay (and Ollama when that is the provider)
   - reads outside Python, the installed packages, `agentic_gp/`, `data/fly` and the driver's own file
   - file writes outside the driver's log and the LLM log/cache folders
   - dangerous `pickle` globals
4. **Inbox password**, `require_instructor()` in [`app/lib/ui.py`](../app/lib/ui.py).
   - Set `AGP_INSTRUCTOR_PASSWORD`.
   - When hosted (`AGP_HOSTED=1` or any `RAILWAY_*` variable), the page stays locked until that variable is set.
   - The Inbox smoke test (`app/lib/inbox_tools.py`) and `lesson/instructor/grade.py` run `race.py check --isolate` with no key in the environment.

Tests: [`tests/test_guard.py`](../tests/test_guard.py) has 21 tests. They cover static bypasses, runtime blocks, a relay round
trip where the driver holds no key, and the Inbox lock.

**Not an OS sandbox.** An audit hook is defence in depth: a determined attacker with a C-level exploit gets
past it. On a public server, keep the key's budget small. Only the Inbox page is password-locked.

**Unity WebGL viewer.** This was done in a separate session, and its files are not in git yet.
- The Unity 6 (6000.2.5f1) race viewer now also builds for WebGL.
- [`deploy/viewer/`](../deploy/viewer) contains:
  - a Caddy-based Dockerfile
  - `railway.json`
  - `stage.py`, which copies the WebGL build and replays into `deploy/viewer/site/` (git-ignored)
- Railway only serves static files; each browser renders the 3D race. Full instructions: [`unity/README.md`, "Host the viewer on Railway"](../unity/README.md#host-the-viewer-on-railway).
- Verified in Chromium from the Docker image:
  - the 12-car demo plays
  - URL options work
  - the missing-replay banner shows
  - hot reload, full screen and `?selftest` work (11/11 PASS)
- Bug fixed along the way: WebGL stripped the physics code, so `link.xml` keeps the collider classes.

---

## 3. Working in this repo

```bash
uv sync                                   # or: python -m venv .venv && pip install -r requirements.txt
uv run python -m pytest -q                # full suite; live MiniMax tests (-m llm) skip without AGP_LIVE_LLM=1
uv run race.py check drivers/examples/l2_planner.py
uv run race.py doctor
uv run streamlit run app/race_control.py  # Race Control at http://localhost:8501
```

**Things that will bite you:**
- **Never commit `.env`.** It holds the shared MiniMax key; `.env.example` is the template.
- `data/`, `races/*` (except `races/demo/`), `results/*` (except the reference `*_grand_prix.json`), `drivers/submissions/*` and `lesson_private/` are git-ignored on purpose. Course-day files must not land in git.
- **Do not weaken the guard to make a driver work.**
  - If a legitimate driver is blocked, find the exact audit event (the `PermissionError` says "blocked by the race guard: ...").
  - Add the narrowest allowance, plus a test in `tests/test_guard.py`.
  - Tests that need an unguarded child use `IsolatedDriver(path, check=False, guard=False)`.
- **Child start-up is slow on Windows** (3-12 s per driver process under load). Keep heavy imports, such as `http.server` in `llm_proxy`, lazy in the parent.
- `legacy/v1/` is an archive and is not collected by pytest (`testpaths = ["tests"]`). Don't "fix" its imports.
- `lesson/instructor/CALIBRATION.md` is **generated** by `lesson/instructor/calibrate.py`. Don't hand-edit it.

---

## 4. Remaining work, staged

Work top to bottom. **[OWNER]** marks a step that needs the owner's explicit yes because it publishes something,
spends the key's budget, or is a product choice. Everything else can be done and opened as a PR directly.

### Task 0: Sync the checkout (2 min)

Done at handover. On another machine, or if time has passed:

```bash
git switch main
git pull --ff-only
```

**Done when** `git status` shows `main` up to date with `origin/main`. In this checkout, `unity/` and `deploy/` are
also untracked until Task 1.

### Task 1: Commit the Unity WebGL viewer as its own PR (30 min)

**Why:** the viewer exists only in this one checkout. Losing the folder loses the work.

**Steps:**
1. `git switch -c unity-webgl-viewer`
2. Review what will be added with `git status --porcelain -uall unity deploy`. Expect about 111 files and about 1 MB:
   - C# scripts, `.meta` files, the scene, materials, `ProjectSettings/`, `Packages/manifest.json` and `packages-lock.json`
   - the WebGL template, `link.xml` and `StreamingAssets/replays/demo.json`
   - `unity/build_viewer.ps1`, `unity/tools/*.py` and `unity/README.md`
   - `deploy/viewer/{Dockerfile,Caddyfile,railway.json,stage.py,.gitignore}`
3. Make sure none of these are staged:
   - `unity/RaceViewer/Library/`, `Temp/`, `Logs/`, `UserSettings/`, `Build/`, `Builds/`
   - `deploy/viewer/site/`

   The `.gitignore` files already exclude them. Check with `git diff --cached --name-only | grep -E 'Library/|/Build|site/'` (expect no output).
4. Make sure no file is larger than about 1 MB and nothing holds a key: `grep -rIl -E 'sk-|API_KEY=' unity deploy` should print nothing.
5. Run `uv run python -m pytest -q tests/test_viz_player.py` and any test that imports `agentic_gp.viz.unity`. Also run `python -m py_compile deploy/viewer/stage.py unity/tools/*.py`.
6. Commit, push and open the PR against `main`. **[OWNER]** Push and PR only with the owner's go-ahead.

**Done when** the PR is open and CI is green.

### Task 2: [OWNER] Deploy the viewer to Railway (20 min)

**Why:** the owner may not have a Unity PC at the venue.

**Before deploying:** this publishes a public URL. Get a yes first. The owner must run `railway login` themselves (it is an interactive sign-in).

1. A WebGL build already exists in `unity/RaceViewer/Build/WebGL/`, and `deploy/viewer/site/` is staged. If either is missing, rebuild on a machine with Unity 6 plus Web Build Support:
   ```powershell
   powershell -ExecutionPolicy Bypass -File unity/build_viewer.ps1 -WebGL
   ```
   The first build takes about 45 minutes and later builds 3-4 minutes. Then stage the build:
   ```bash
   python deploy/viewer/stage.py
   ```
2. Deploy and get a public domain:
   ```bash
   railway up deploy/viewer --path-as-root --no-gitignore
   railway domain
   ```
3. Verify: open `https://<app>.up.railway.app/?replay=replays/demo.json`, and open the same URL with `&selftest` added. The browser console must show `[SelfTest] RESULT: PASS`.
4. Not tested yet: Safari, Firefox and a real GPU machine. About 250 `INVALID_ENUM bindTexture` warnings appeared in headless Chromium, but the picture was correct. Check these on the venue laptop.
5. Clean up the local container: `docker rm -f agp-viewer`.

**Done when** the public URL plays the demo and the self-test passes. Write the URL into `lesson/instructor/RUN_OF_SHOW.md`.

### Task 3: Lesson text fixes (30 min, no owner decision)

Found by the lesson audit. All are cosmetic, and every number in the chapters reproduces.

| # | Where | Problem | Fix |
|---|---|---|---|
| a | [`lesson/07_fly_brains.md`](../lesson/07_fly_brains.md) ~line 175 | Quotes the missing-file message as "No fly connectome file. Instructor: run python -m agentic_gp.fly.extract". | Quote the real message from `drivers/examples/fly_connectome.py:123-124` instead: "No data/fly/steering_circuit_783.npz: ask your instructor for that one file (instructor: python -m agentic_gp.fly.extract). The fly coasts." |
| b | [`lesson/checkpoints/ch3_learner.py:27`](../lesson/checkpoints/ch3_learner.py) | The exercise hint says `self.history[-1]["crash_segments"]`, but the current attempt is appended at line 117, so `[-1]` is "this attempt". The chapter text correctly uses `[-2]`. | Read lines 100-125 to confirm where the exercise rule runs. If it runs after the append, change the hint to compare `self.history[-1]` with `self.history[-2]`; otherwise leave it and note why. Then run `uv run race.py check lesson/checkpoints/ch3_learner.py` and `uv run python -m pytest -q tests/test_checkpoints.py`. |
| c | [`lesson/07_fly_brains.md`](../lesson/07_fly_brains.md) ~line 241 | Says the lateralisation index is **+0.83**, measured with the figure's settings (10° steps, 5 seeds). The Fly Lab page defaults (`app/views/fly_lab.py`, 15° steps, 3 seeds) show about +0.84 to +0.85. | Add "(about +0.84 on the Fly Lab page's default settings)" after +0.83. Do **not** change the figure. |
| d | `lesson/instructor/CALIBRATION.md:11` | Says "Python 3.14.2", but the project pins 3.12 (`.python-version`). | Generated file: re-run `uv run python lesson/instructor/calibrate.py` under 3.12. Its "real" section needs the MiniMax key, which makes this **[OWNER]**. Otherwise leave it; all numbers still reproduce. |

**Done when** a-c are fixed, `uv run python -m pytest -q tests/test_checkpoints.py tests/test_streamlit_lessons.py` passes, and the PR is open.

### Task 4: [OWNER] Record the LLM cassettes (45 min, spends the key's budget)

**Why:** `lesson/cassettes/` has only its README. Without recordings, the live LLM demos for Ch4 (pilot), Ch5
(strategist) and Ch6 (crew chief) have no offline fallback if the Wi-Fi, the key or MiniMax fails in class.

**Steps:**
1. Follow the "Record" section of [`lesson/cassettes/README.md`](../lesson/cassettes/README.md). Use the real key in `.env`, start each recording from an empty file (`LLM_RECORD` appends), and record one cassette per demo command.
2. Replay each one with `LLM_PROVIDER=replay LLM_CASSETTE=<file>` and the **same command**. Check that it runs offline.
3. **Never edit or fake a cassette by hand.**
4. Decide with the owner whether cassettes go in git or into `lesson_private/`. They contain model answers, never the key.

**Done when** each of the three demos replays offline with no network.

### Task 5: Make the timing-sensitive tests robust on slow Windows machines (1-2 h)

**Symptom:** under heavy load (Unity open, other processes busy), these failed in a full local run but passed on
re-run and in CI:
- `tests/test_grade.py::test_scrutineer_runs_the_real_check`
- `tests/test_loader.py::test_isolation_works_from_an_unguarded_main_script`
- `tests/test_loader.py::test_llm_warm_up_does_not_count_against_the_load_timeout`
- `tests/test_loader.py::test_begin_session_waits_for_the_sdk_warm_up[10.0-True]`
- `tests/test_loader.py::test_begin_session_waits_for_the_sdk_warm_up[6.0-False]` (also flaky when run alone)

**Cause:** driver child processes take 3-12 s to start on Windows under load. The tests assert tight wall-clock windows.

**Approach:**
- Widen the windows, or derive them from a measured baseline start-up (spawn one trivial child first and scale by it). Avoid sleeps.
- For `[6.0-False]` the test asserts that a 6 s budget is *too short*. Make the warm-up it simulates clearly longer than the budget, so it does not depend on machine speed.
- **Do not** change the production timeouts in `loader.py` to make the tests pass.

**Done when** `uv run python -m pytest -q tests/test_loader.py tests/test_grade.py` passes 5 times in a row while another CPU-heavy process runs, and CI stays green.

### Task 6 (optional): [OWNER] Host Race Control and lock the other pages

**Context:**
- The Streamlit app has no Railway/Docker config yet.
- The Inbox lock already detects Railway (`RAILWAY_*` variables).
- On a public host, every other page (races with real LLM calls, Fly Lab, ...) is open to anyone with the URL, and real-LLM races spend the key.

**Decision needed:** should Race Control be hosted at all, and which pages should be instructor-only?

**If yes:**
- Add a `deploy/app/` Dockerfile and `railway.json`, modelled on `deploy/viewer/`.
- Call `require_instructor()` at the top of each page that can spend the key.
- Set `AGP_INSTRUCTOR_PASSWORD`, `MINIMAX_API_KEY` and a low `AGP_LLM_PROXY_MAX_CALLS` in Railway's variables, never in git.
- Extend `tests/test_guard.py`'s Inbox tests to cover the newly locked pages.

### Task 7 (optional, only if asked): Port the legacy v1 agents to 2.0 drivers

[`legacy/v1/README.md`](../legacy/v1/README.md) lists them. Port one at a time to `drivers/examples/` on the 2.0
API (`drivers/template.py`). It must pass `race.py check`, and its lap time goes into `drivers/README.md`.
The v1 leaderboard numbers are not comparable to 2.0 Grand Prix numbers without a re-run.

---

## 5. Decisions already made (don't re-open them without the owner)

- `open()` and all file I/O in student drivers are **errors**, not warnings.
- The key stays in the race process; drivers only ever see relay tokens.
- The Inbox needs `AGP_INSTRUCTOR_PASSWORD`; when hosted without it, the page is locked.
- The 3D viewer reaches the venue as a **WebGL build served as static files**, not as a Unity install. Nothing is deployed until the owner says so.
- The v1 agents from Sept 15 are **archived** in `legacy/v1/`, not ported and not deleted.
- Commits end with a `Co-Authored-By:` line for the AI agent that wrote them. Push, PR, deploy and key-spending steps need the owner's go-ahead.
