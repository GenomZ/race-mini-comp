"""Shared helpers of the Race Control app (no page code here).

    paths        REPO_ROOT, RACES_DIR, DRIVER_FOLDERS, ensure_repo_on_path()
    loaders      cached loaders: replays, tracks, sessions, results, JSON
    drivers      driver discovery + static inspection (never runs student code)
    checkpoints  "Where am I?" - which checkpoint CP0-CP8 a driver file reached (+ sidebar widget)
    params_ui    PARAMS -> sliders
    live_runner  LiveRun (a session in a background thread) and BackgroundJob
    llm_trace    reading LLM JSONL logs for the Pit Wall page
"""
