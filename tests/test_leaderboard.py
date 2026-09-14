import json

from agentic_gp.leaderboard import load_results, render_markdown, sort_key


def _entry(name, best_lap, crashes=0, waypoints=16, completed=True):
    return {
        "name": name,
        "best_lap": best_lap,
        "total_crashes": crashes,
        "max_waypoints_passed": waypoints,
        "attempts": [{"completed": completed}],
        "tool_calls": 100,
        "model": "-",
        "timestamp": "2026-09-14T00:00:00",
        "_file": f"{name}.json",
    }


def test_finishers_rank_before_dnfs_and_by_time():
    rows = [
        _entry("slow", 120.0),
        _entry("dnf-far", None, crashes=3, waypoints=12, completed=False),
        _entry("fast", 70.0),
        _entry("dnf-near", None, crashes=1, waypoints=2, completed=False),
        _entry("hermes", 100.0, crashes=2),
    ]
    ordered = [r["name"] for r in sorted(rows, key=sort_key)]
    assert ordered == ["fast", "hermes", "slow", "dnf-far", "dnf-near"]


def test_markdown_marks_baseline_and_deltas():
    md = render_markdown([_entry("fast", 70.0), _entry("hermes", 100.0)])
    assert "**hermes** (baseline)" in md
    assert "-30.00" in md
    assert md.splitlines()[0].startswith("# Agentic Grand Prix")


def test_load_results_skips_bad_json(tmp_path):
    (tmp_path / "ok.json").write_text(json.dumps(_entry("ok", 80.0)))
    (tmp_path / "bad.json").write_text("{not json")
    rows = load_results(tmp_path)
    assert [r["name"] for r in rows] == ["ok"]
    assert "(no results yet)" in render_markdown([])
