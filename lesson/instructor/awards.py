"""Compute the Grand Prix awards from the final's result files (see awards.md for the rules).

Run from the repo root after `race.py final ... --out races/final`:

    python lesson/instructor/awards.py                                     # reads races/final/
    python lesson/instructor/awards.py --snapshot1 races/snapshots/snap1_results.json \
                                       --fly-cup races/flycup/flycup_results.json --out races/final/awards.md

Inputs (all written by race.py): <final>/home_results.json, <final>/mystery_results.json,
<final>/championship.json (computed from the two results files when missing), and the replays
<final>/home.json / mystery.json (zone penalties and radio messages). House cars (Max, Planner, Fly Toy)
race but never win an award; "Beat the House" compares everyone with Max.

Jury / vote awards (Most Creative LLM Use, Best Agent Design, Best Radio) cannot be computed: the
script lists the data the jury needs and leaves a blank to fill in.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HOUSE_NAME = "Max"
DNF_SECONDS = 300.0           # a DNF in a snapshot counts as 300 s (the race.py validate convention)
LEARNER_WINDOW = 0.10         # Best Learner: only drivers within 10 % of pole
EFFICIENT_MAX_CALLS = 3       # Most Efficient: at most 3 LLM calls on the mystery weekend


@dataclass
class Award:
    title: str
    winners: list[str]
    detail: str = ""
    how: str = ""
    jury: bool = False
    data: list[str] = field(default_factory=list)


# ============================================================================ loading
def load_json(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def cars(gp: dict | None) -> list[dict]:
    """Standings rows joined with the qualifying row, the driver's attempts, LLM usage and file stem."""
    if not gp:
        return []
    quali = {q.get("name"): q for q in gp.get("qualifying") or []}
    drivers = {d.get("name"): d for d in gp.get("drivers") or []}
    out = []
    for s in gp.get("standings") or []:
        d = drivers.get(s.get("name"), {})
        src = str(d.get("source") or "").replace("\\", "/")
        out.append({**s, "stem": Path(src).stem.lower() if src else None, "q": quali.get(s.get("name"), {}),
                    "attempts": d.get("attempts") or [], "llm": d.get("llm") or {}})
    return out


def finished(c: dict) -> bool:
    return c.get("race_time") is not None and not c.get("status")


def scored(rows: list[dict]) -> list[dict]:
    return [c for c in rows if not c.get("house")]


def house_car(rows: list[dict], name: str = HOUSE_NAME) -> dict | None:
    return next((c for c in rows if c.get("house") and c.get("name") == name), None)


def replay_counts(replay: dict | None) -> dict[str, dict]:
    """car name -> {"zone": n zone penalties, "radio": n radio messages} from a replay JSON."""
    out: dict[str, dict] = {}
    for car in (replay or {}).get("cars") or []:
        out[car.get("name")] = {"zone": sum(1 for e in car.get("events") or [] if e.get("kind") == "zone"),
                                "radio": len(car.get("messages") or [])}
    return out


def who(c: dict) -> str:
    """'Fly Toy (ada_fly.py)': the NAME plus the file it came from (two students may share an example's NAME)."""
    return f"{c['name']} ({c['stem']}.py)" if c.get("stem") else str(c.get("name"))


def _t(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f} s"


def same_driver(a: dict, b: dict) -> bool:
    """The same student in two result files: by file stem, else by AUTHOR, else by NAME."""
    if a.get("stem") and a.get("stem") == b.get("stem"):
        return True
    au, bu = str(a.get("author") or "").lower(), str(b.get("author") or "").lower()
    if au and au == bu and au != "your_handle":
        return True
    return a.get("name") == b.get("name")


# ============================================================================ the awards
def race_winner(label: str, rows: list[dict]) -> Award:
    how = "first scored car (house cars race but are not scored)"
    s = scored(rows)
    if not s:
        return Award(f"{label} GP winner", [], "no results", how)
    w = s[0]
    house = [c for c in rows if c.get("house") and c["position"] < w["position"]]
    detail = f"{_t(w.get('race_time'))}" + (f" ({', '.join(c['name'] for c in house)} (house) finished ahead)"
                                             if house else "")
    return Award(f"{label} GP winner", [w["name"]], detail, how)


def pole(label: str, rows: list[dict]) -> Award:
    how = "fastest qualifying lap among scored cars"
    q = [c for c in scored(rows) if c["q"].get("qualifying_time") is not None]
    if not q:
        return Award(f"Pole ({label})", [], "no qualifying lap", how)
    best = min(c["q"]["qualifying_time"] for c in q)
    winners = [c["name"] for c in q if c["q"]["qualifying_time"] == best]
    real = min((c for c in rows if c["q"].get("qualifying_time") is not None),
               key=lambda c: c["q"]["qualifying_time"], default=None)
    note = f" (real pole: {real['name']} (house), {_t(real['q']['qualifying_time'])})" if real and real.get("house") \
        and real["q"]["qualifying_time"] < best else ""
    return Award(f"Pole ({label})", winners, _t(best) + note, how)


def fastest_lap(label: str, rows: list[dict]) -> Award:
    how = "best single race lap (standings 'best_lap') among scored cars"
    laps = [c for c in scored(rows) if c.get("best_lap") is not None]
    if not laps:
        return Award(f"Fastest race lap ({label})", [], "no complete race lap", how)
    best = min(c["best_lap"] for c in laps)
    return Award(f"Fastest race lap ({label})", [c["name"] for c in laps if c["best_lap"] == best], _t(best), how)


def beat_the_house(races: list[tuple[str, list[dict]]]) -> Award:
    how = f"badge for every scored car that finished ahead of {HOUSE_NAME} in either race"
    names: list[str] = []
    notes = []
    for label, rows in races:
        h = house_car(rows)
        if h is None:
            notes.append(f"{label}: {HOUSE_NAME} did not race")
            continue
        ahead = [c["name"] for c in scored(rows) if c["position"] < h["position"]]
        notes.append(f"{label}: {HOUSE_NAME} P{h['position']} {_t(h.get('race_time'))}, {len(ahead)} ahead")
        names += [n for n in ahead if n not in names]
    return Award("Beat the House", names, "; ".join(notes), how)


def best_learner(rows: list[dict]) -> Award:
    how = (f"largest gain from practice 1 to the qualifying lap on the mystery track, among scored cars "
           f"within {LEARNER_WINDOW:.0%} of pole")
    times = [c["q"]["qualifying_time"] for c in rows if c["q"].get("qualifying_time") is not None]
    if not times:
        return Award("Best Learner", [], "no qualifying laps", how)
    pole_time = min(times)
    cands = []
    for c in scored(rows):
        q = c["q"].get("qualifying_time")
        p1 = (c["q"].get("practice_times") or [None])[0]
        if q is None or p1 is None or q > pole_time * (1 + LEARNER_WINDOW):
            continue
        cands.append((p1 - q, c, p1, q))
    if not cands:
        return Award("Best Learner", [], f"nobody within {LEARNER_WINDOW:.0%} of pole ({_t(pole_time)})", how)
    cands.sort(key=lambda x: -x[0])
    gain, c, p1, q = cands[0]
    data = [f"{x[1]['name']}: {_t(x[2])} -> {_t(x[3])} ({x[0]:+.2f} s)" for x in cands]
    winners = [x[1]["name"] for x in cands if x[0] == gain and gain > 0]
    return Award("Best Learner", winners, f"{_t(p1)} -> {_t(q)} (gain {gain:.2f} s, {gain / p1:.1%})"
                 if winners else "no driver improved", how, data=data)


def cleanest(races: list[tuple[str, list[dict]]], zones: dict[str, dict[str, dict]]) -> Award:
    how = "finished both races; fewest crashes + zone penalties in the races; ties: lower total race time"
    if len(races) < 2:
        how = "finished the race; fewest crashes + zone penalties; ties: lower race time"
    table: dict[str, dict] = {}
    for label, rows in races:
        for c in scored(rows):
            t = table.setdefault(c["name"], {"n": 0, "faults": 0, "time": 0.0, "ok": True})
            t["n"] += 1
            t["ok"] = t["ok"] and finished(c)
            t["faults"] += int(c.get("crashes") or 0) + zones.get(label, {}).get(c["name"], {}).get("zone", 0)
            t["time"] += c.get("race_time") or 0.0
    ok = {n: t for n, t in table.items() if t["ok"] and t["n"] == len(races)}
    if not ok:
        return Award("Cleanest Driver", [], "nobody finished every race", how)
    best = min(ok.values(), key=lambda t: (t["faults"], t["time"]))
    winners = [n for n, t in ok.items() if (t["faults"], t["time"]) == (best["faults"], best["time"])]
    return Award("Cleanest Driver", winners, f"{best['faults']} crashes + zone penalties, "
                                             f"{best['time']:.2f} s total", how)


def llm_class(champ: list[dict], races: list[tuple[str, list[dict]]]) -> Award:
    how = ('best championship position among KIND="llm" cars that made at least one LLM call; confirm in '
           "races/final/llm_logs/ that a validated LLM answer changed the plan")
    used = {c["name"] for _, rows in races for c in scored(rows)
            if c.get("kind") == "llm" and int(c.get("llm_calls") or 0) > 0}
    for row in champ:
        if row.get("name") in used:
            return Award("LLM Class winner", [row["name"]], f"championship P{row['position']}, "
                                                           f"{row['points']} points", how)
    return Award("LLM Class winner", [], "no LLM-class car made an LLM call", how)


def most_efficient(rows: list[dict]) -> Award:
    how = (f"best mystery race time among finishers that used the LLM (1 to {EFFICIENT_MAX_CALLS} calls "
           f"over the whole weekend)")
    c = [x for x in scored(rows) if finished(x) and 1 <= int(x.get("llm_calls") or 0) <= EFFICIENT_MAX_CALLS]
    if not c:
        return Award("Most Efficient", [], "no finisher with 1-3 LLM calls", how)
    w = min(c, key=lambda x: x["race_time"])
    return Award("Most Efficient", [w["name"]], f"{_t(w['race_time'])} with {w['llm_calls']} LLM call(s)", how)


def best_fly(fly_rows: list[dict], races: list[tuple[str, list[dict]]]) -> Award:
    how = ('fastest finished KIND="fly" car (Fly Cup if given, else the GPs); it must also beat its own '
           "scrambled control: python race.py drive <file> --param SCRAMBLE=0 (and 1, 2) must be slower or DNF")
    pool = fly_rows or [c for _, rows in races for c in rows]
    fl = [c for c in scored(pool) if c.get("kind") == "fly" and finished(c)]
    if not fl:
        return Award("Best Fly Brain", [], "no finished fly-class car", how)
    best = min(c["race_time"] for c in fl)
    winners = [c for c in fl if c["race_time"] == best]            # a tie names every tied car
    tie = " (tie)" if len(winners) > 1 else ""
    return Award("Best Fly Brain", [who(c) for c in winners], f"{_t(best)}{tie} - check the scrambled control "
                 "before announcing", how, data=[f"{who(c)}: {_t(c['race_time'])}" for c in fl])


def biggest_improvement(snap1: list[dict], home: list[dict]) -> Award:
    how = (f"Snapshot #1 lap time (DNF = {DNF_SECONDS:.0f} s) minus the final Home GP qualifying lap; "
           "both are one lap of the home track")
    if not snap1:
        return Award("Biggest Improvement", [], "pass --snapshot1 races/snapshots/snap1_results.json", how)
    rows = []
    for c in scored(home):
        q = c["q"].get("qualifying_time")
        before = next((s for s in scored(snap1) if same_driver(s, c)), None)
        if q is None or before is None:
            continue
        t0 = before["race_time"] if finished(before) else DNF_SECONDS
        rows.append((t0 - q, c["name"], t0, q))
    if not rows:
        return Award("Biggest Improvement", [], "nobody raced both Snapshot #1 and the Home GP qualifying", how)
    rows.sort(key=lambda r: -r[0])
    gain, name, t0, q = rows[0]
    data = [f"{r[1]}: {r[2]:.2f} -> {r[3]:.2f} s ({r[0]:+.2f})" for r in rows]
    if gain <= 0:
        return Award("Biggest Improvement", [], "nobody got faster", how, data=data)
    winners = [r[1] for r in rows if r[0] == gain]
    return Award("Biggest Improvement", winners, f"{t0:.2f} s -> {q:.2f} s ({gain:.2f} s faster)", how, data=data)


def jury(title: str, how: str, data: list[str]) -> Award:
    return Award(title, [], "", how, jury=True, data=data)


# ============================================================================ assemble
def compute_awards(final: Path, snapshot1: Path | None = None, fly_cup: Path | None = None) -> list[Award]:
    home_gp = load_json(final / "home_results.json")
    myst_gp = load_json(final / "mystery_results.json")
    if not home_gp and not myst_gp:
        raise SystemExit(f"ERROR: no home_results.json / mystery_results.json in {final} (run race.py final first)")
    races = [(label, cars(gp)) for label, gp in (("Home", home_gp), ("Mystery", myst_gp)) if gp]
    zones = {label: replay_counts(load_json(final / f"{stem}.json"))
             for label, stem in (("Home", "home"), ("Mystery", "mystery"))}
    champ_doc = load_json(final / "championship.json")
    if champ_doc:
        champ = champ_doc.get("standings") or []
    else:
        from agentic_gp.race.leaderboard import championship
        champ = championship([gp for gp in (home_gp, myst_gp) if gp])
    home = dict(races).get("Home", [])
    myst = dict(races).get("Mystery", [])
    snap1 = cars(load_json(snapshot1)) if snapshot1 else []
    fly_rows = cars(load_json(fly_cup)) if fly_cup else []

    out = [Award("Champion", [champ[0]["name"]] if champ else [], f"{champ[0]['points']} points" if champ else
                 "no championship", "most championship points (N+1-position per race; mystery breaks ties)")]
    out += [race_winner(label, rows) for label, rows in races]
    out += [pole(label, rows) for label, rows in races]
    out += [fastest_lap(label, rows) for label, rows in races]
    out.append(beat_the_house(races))
    out.append(best_learner(myst) if myst else Award("Best Learner", [], "no mystery results"))
    out.append(cleanest(races, zones))
    out.append(llm_class(champ, races))
    out.append(most_efficient(myst) if myst else Award("Most Efficient", [], "no mystery results"))
    out.append(best_fly(fly_rows, races))
    out.append(biggest_improvement(snap1, home))
    llm_rows = [f"{c['name']}: {c.get('llm_calls', 0)} calls, {c.get('llm_tokens', 0)} tokens ({label})"
                for label, rows in races for c in scored(rows) if int(c.get("llm_calls") or 0) > 0]
    out.append(jury("Most Creative LLM Use", "jury: logs in races/final/llm_logs/ + 60-second pit interview",
                    llm_rows or ["no LLM calls"]))
    out.append(jury("Best Agent Design", "shortlist of 5 driver cards -> class vote", []))
    radio = []
    house_names = {c["name"] for _, rows in races for c in rows if c.get("house")}
    for label, _ in races:
        for name, n in sorted(zones.get(label, {}).items(), key=lambda kv: -kv[1]["radio"]):
            if n["radio"] and name not in house_names:
                radio.append(f"{name}: {n['radio']} radio messages ({label})")
    out.append(jury("Best Radio", "peer vote (the replays' radio feed)", radio[:10]))
    return out


def to_markdown(awards: list[Award], final: Path) -> str:
    lines = [f"# Awards - {final.as_posix()}", "", "| Award | Winner | Detail |", "|---|---|---|"]
    for a in awards:
        who = "(jury / vote: ____________)" if a.jury else (", ".join(a.winners) or "-")
        lines.append(f"| {a.title} | {who} | {a.detail} |")
    lines += ["", "## How each award was computed", ""]
    for a in awards:
        lines.append(f"- **{a.title}**: {a.how}")
        lines += [f"  - {d}" for d in a.data[:30]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--final", default="races/final", help="the race.py final --out folder")
    ap.add_argument("--snapshot1", help="Snapshot #1 *_results.json (Biggest Improvement)")
    ap.add_argument("--fly-cup", help="Fly Cup *_results.json (Best Fly Brain)")
    ap.add_argument("--out", help="also write the markdown here (e.g. races/final/awards.md)")
    args = ap.parse_args(argv)
    final = Path(args.final)
    md = to_markdown(compute_awards(final, Path(args.snapshot1) if args.snapshot1 else None,
                                    Path(args.fly_cup) if args.fly_cup else None), final)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
