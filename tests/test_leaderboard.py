import json

from agentic_gp.race.leaderboard import load_results, main, render_markdown, sort_key


def _entry(name, best_lap, crashes=0, waypoints=16, completed=True, kind="code", llm_calls=0):
    return {
        "format": "agentic-gp-result", "version": 1, "name": name, "kind": kind, "model": None, "track": "grand_prix",
        "best_lap": best_lap, "total_crashes": crashes, "max_waypoints_passed": waypoints,
        "attempts": [{"completed": completed}], "llm": {"calls": llm_calls}, "llm_calls": llm_calls,
        "timestamp": "2026-10-21T10:00:00Z",
    }


def test_finishers_rank_before_dnfs_and_by_time():
    rows = [
        _entry("slow", 120.0),
        _entry("dnf-far", None, crashes=3, waypoints=12, completed=False),
        _entry("fast", 50.0),
        _entry("dnf-near", None, crashes=1, waypoints=2, completed=False),
        _entry("Max", 55.0, crashes=2, kind="llm"),
        _entry("fast-but-crashy", 50.0, crashes=1),
    ]
    ordered = [r["name"] for r in sorted(rows, key=sort_key)]
    assert ordered == ["fast", "fast-but-crashy", "Max", "slow", "dnf-far", "dnf-near"]


def test_markdown_compares_with_the_house_racer():
    md = render_markdown([_entry("fast", 50.0, llm_calls=3), _entry("Max", 55.0, kind="llm"),
                          _entry("dnf", None, completed=False)])
    lines = md.splitlines()
    assert lines[0].startswith("# Agentic Grand Prix")
    assert "vs Max" in md and "**Max** (house)" in md
    assert "| 1 | fast | code | grand_prix | 50.00 | -5.00 | 1 (1) | 0 | 16 | 3 |" in md
    assert "| DNF | - |" in md
    assert "(no results yet)" in render_markdown([])
    assert "beats DNF" in render_markdown([_entry("a", 60.0), _entry("Max", None, completed=False)])


def test_load_results_skips_bad_files_and_main_writes_markdown(tmp_path, capsys):
    res = tmp_path / "results"
    res.mkdir()
    (res / "ok.json").write_text(json.dumps(_entry("ok", 80.0)), encoding="utf-8")
    (res / "bad.json").write_text("{not json", encoding="utf-8")
    (res / "list.json").write_text("[1, 2]", encoding="utf-8")
    assert [r["name"] for r in load_results(res)] == ["ok"]
    out = tmp_path / "LEADERBOARD.md"
    assert main(["--results", str(res), "--out", str(out)]) == 0
    assert "| 1 | ok |" in out.read_text(encoding="utf-8")
    assert "ok" in capsys.readouterr().out


def test_session_results_feed_the_leaderboard(tmp_path):
    from agentic_gp.race import SessionConfig, run_session

    res = run_session(lambda obs: (0.5, 0.0), SessionConfig(track="oval", attempts=1, max_ticks_per_lap=5))
    (tmp_path / "coast.json").write_text(json.dumps(res.to_result_dict()), encoding="utf-8")
    rows = load_results(tmp_path)
    assert rows[0]["name"] == "<lambda>" and rows[0]["best_lap"] is None
    assert "| 1 | <lambda> | code | oval | DNF |" in render_markdown(rows)


def test_one_table_per_track_and_a_track_filter():
    sprint = dict(_entry("quick", 41.4), track="sprint")
    rows = [_entry("slow", 307.4), sprint, _entry("Max", 52.5, kind="llm")]
    md = render_markdown(rows)
    assert "## grand_prix" in md and "## sprint" in md and md.index("## grand_prix") < md.index("## sprint")
    gp_table = md.split("## grand_prix")[1].split("## sprint")[0]
    assert "| 1 | **Max** (house) |" in gp_table and "quick" not in gp_table      # never ranked across tracks
    only = render_markdown(rows, track="sprint")
    assert "## " not in only and "| 1 | quick | code | sprint | 41.40 | - |" in only and "slow" not in only
