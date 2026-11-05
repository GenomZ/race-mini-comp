"""Home: the ladder poster, the Three Clocks, where to go next, and the latest race."""

import streamlit as st

from app.lib.loaders import list_json, list_replays, load_json
from app.lib.paths import DRIVERS_DIR, rel
from app.lib.ui import hide_mystery, mark_dead_heats, page_header

page_header("Agentic Grand Prix - Race Control",
            "Where every part of the course lives. Climb the ladder one level at a time; each level adds ONE "
            "agent idea to your single driver file, and each idea lives on one of three clocks.")

# ------------------------------------------------------------------------------ the ladder poster
# (level, title, idea, example file, clock, expected band, side quest?)
LADDER = [
    ("CP0", "Pit Lane", "the loop: obs in, (throttle, steering) out", "l0_dummy", "TICK", "check PASSED; template ~86 s", False),
    ("CP1", "Reflexes", "condition -> action rules, stopping distance", "l1_reflex", "TICK", "60-80 s clean", False),
    ("CP2", "The Map", "world model + plan: v = sqrt(20 R)", "l2_planner", "SESSION", "48-55 s clean", False),
    ("CP3", "Practice", "learn between attempts: keep-best", "l3_learner", "LAP", "best >= 3 % better", False),
    ("CP4", "LLM Pilot", "the LLM drives every tick... too slow", "l4_llm_rookie", "TICK", "the latency wall", True),
    ("CP5", "Strategist", "LLM plans once, code executes", "l5_llm_strategist", "SESSION", "+-5 % of CP3, 0 zone hits", False),
    ("CP6", "Crew Chief", "LLM + tools + memory in reflect()", "l6_llm_crew_chief", "LAP", "improves on a new track", False),
    ("CP7", "Fly Brain", "4,494 real neurons steer the car", "fly_connectome", "TICK", "finishes a lap", True),
    ("CP8", "Garage", "robust final file, survives LLM-off", "your file", "ALL", "clean everywhere", False),
]
CLOCK_COLOR = {"TICK": "#3987e5", "LAP": "#fab219", "SESSION": "#199e70", "ALL": "#c3c2b7"}

STYLE = """
<style>
.agp-ladder{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin:4px 0 6px}
.agp-step{background:#1a1a19;border:1px solid #2c2c2a;border-radius:10px;padding:10px 12px;position:relative}
.agp-step.side{border-style:dashed;opacity:.88}
.agp-step .cp{font:700 13px system-ui;color:#898781;letter-spacing:.06em}
.agp-step .t{font:700 17px system-ui;color:#f0efec;margin:2px 0 4px}
.agp-step .i{font:13px/1.35 system-ui;color:#c3c2b7;min-height:36px}
.agp-step .f{font:12px ui-monospace,Consolas,monospace;color:#898781;margin-top:6px}
.agp-step .b{font:12px system-ui;color:#f0efec;margin-top:4px}
.agp-clock{display:inline-block;font:700 11px system-ui;padding:1px 7px;border-radius:9px;color:#0e0f11;margin-top:6px}
.agp-side{font:600 11px system-ui;color:#9085e9;position:absolute;top:9px;right:10px}
</style>
"""


def ladder_html() -> str:
    cards = []
    for cp, title, idea, example, clock, band, side in LADDER:
        cards.append(
            f'<div class="agp-step{" side" if side else ""}"><div class="cp">{cp}</div>'
            + ('<div class="agp-side">side quest</div>' if side else "")
            + f'<div class="t">{title}</div><div class="i">{idea}</div>'
            f'<div class="f">{example}</div><div class="b">{band}</div>'
            f'<span class="agp-clock" style="background:{CLOCK_COLOR[clock]}">{clock} clock</span></div>')
    return STYLE + '<div class="agp-ladder">' + "".join(cards) + "</div>"


def three_clocks_svg() -> str:
    """Session > attempts (laps) > ticks: which method runs on which clock, and how long it may think."""
    x0, x1 = 150, 1080                                   # timeline span (labels sit left of x0)
    ticks = "".join(f'<line x1="{x0 + k * 16}" y1="262" x2="{x0 + k * 16}" y2="286" stroke="#3987e5" stroke-width="4"/>'
                    for k in range(22))
    w = (x1 - x0 - 2 * 40) / 3
    attempts = "".join(
        f'<rect x="{x0 + k * (w + 40):.0f}" y="146" width="{w:.0f}" height="40" rx="7" fill="#2b2410" stroke="#fab219"/>'
        f'<text x="{x0 + k * (w + 40) + w / 2:.0f}" y="172" fill="#f0efec" font-size="18" text-anchor="middle">attempt {k + 1}</text>'
        + (f'<text x="{x0 + k * (w + 40) + w + 20:.0f}" y="173" fill="#fab219" font-size="20" font-weight="700" '
           f'text-anchor="middle">R</text>' if k < 2 else "")
        for k in range(3))
    label = lambda y, color, text: (f'<text x="24" y="{y}" fill="{color}" font-size="19" font-weight="800" '
                                    f'letter-spacing="1">{text}</text>')
    body = lambda y, text, color="#f0efec", size=18: f'<text x="{x0}" y="{y}" fill="{color}" font-size="{size}">{text}</text>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 380" width="1120" height="380"
     font-family="system-ui,Segoe UI,sans-serif">
  <rect width="1120" height="380" rx="14" fill="#141413"/>
  <text x="24" y="44" fill="#f0efec" font-size="26" font-weight="700">The Three Clocks</text>
  <text x="262" y="44" fill="#898781" font-size="18">every part of your driver lives on exactly one of them</text>
  {label(104, "#199e70", "SESSION")}
  <rect x="{x0}" y="94" width="{x1 - x0}" height="12" rx="6" fill="#123d2f"/>
  <circle cx="{x0}" cy="100" r="12" fill="#199e70"/>
  {body(82, "<tspan font-weight='700' font-family='Consolas,monospace'>plan(track)</tspan>: once, before the first lap - seconds of thinking are fine (LLM OK)")}
  {label(173, "#fab219", "LAP")}
  {attempts}
  {body(214, "<tspan font-weight='700' font-family='Consolas,monospace'>reflect(report)</tspan> (the R marks): between attempts - learn from the lap report (LLM OK)")}
  {label(282, "#3987e5", "TICK")}
  {ticks}
  <text x="{x0 + 22 * 16 + 20}" y="272" fill="#f0efec" font-size="18"><tspan font-weight="700" font-family="Consolas,monospace">drive(obs)</tspan>: every 0.5 s of race time - answer in milliseconds</text>
  <text x="{x0 + 22 * 16 + 20}" y="296" fill="#898781" font-size="16">an LLM takes ~1-3 s: the car drives blind for 2-6 ticks (Latency Lab)</text>
  <text x="24" y="350" fill="#c3c2b7" font-size="17">Exit ticket: which clock does each part of YOUR driver live on?</text>
</svg>"""


LINKS = [
    ("views/replay.py", "Replay", ":material/replay:", "Ch0 - watch any race, follow a car"),
    ("views/physics_lab.py", "Physics Lab", ":material/speed:", "Ch1 - turning circle, braking, drive by hand"),
    ("views/sensor_xray.py", "Sensor X-ray", ":material/radar:", "Ch1 - exactly what your driver sees"),
    ("views/live.py", "Live", ":material/videocam:", "Ch1-6 - run YOUR driver, watch every tick"),
    ("views/track_lab.py", "Track Lab", ":material/route:", "Ch2 - corner speeds, build a track"),
    ("views/telemetry.py", "Telemetry", ":material/monitoring:", "Ch2-3 - where the time goes"),
    ("views/learning_curves.py", "Learning Curves", ":material/trending_down:", "Ch3 - does practice pay?"),
    ("views/latency_lab.py", "Latency Lab", ":material/timer:", "Ch4 - why the LLM cannot drive"),
    ("views/pit_wall.py", "Pit Wall", ":material/forum:", "Ch5-6 - read the LLM's conversation"),
    ("views/fly_lab.py", "Fly Lab", ":material/bug_report:", "Ch7 - a real fly brain steers"),
    ("views/grand_prix.py", "Grand Prix", ":material/emoji_events:", "Ch8 - the race"),
    ("views/inbox.py", "Inbox", ":material/inbox:", "Instructor - collect and check files"),
]

st.html(ladder_html())
st.image(three_clocks_svg(), width=1120)

st.subheader("Where to go", anchor=False)
cols = st.columns(4)
for k, (target, label, icon, why) in enumerate(LINKS):
    with cols[k % 4]:
        with st.container(border=True, height=104):
            st.page_link(target, label=label, icon=icon)
            st.caption(why)

# ------------------------------------------------------------------------------ latest snapshot
st.divider()
st.subheader("Latest snapshot", anchor=False)
results, _ = hide_mystery(list_json(kind="grand_prix"))
submissions = sorted(p for p in (DRIVERS_DIR / "submissions").glob("*.py") if not p.name.startswith("_"))
replays, _ = hide_mystery(list_replays())
m = st.columns(3)
m[0].metric("Student drivers submitted", len(submissions))
m[1].metric("Replays in races/", len(replays))
m[2].metric("Grand prix results", len(results))

if not results:
    st.info("No grand prix has been run yet. Run one on the Grand Prix page, or: "
            "`python race.py grand-prix drivers/examples --laps 2 --out races/today.json`", icon=":material/info:")
else:
    latest = results[0]
    gp = load_json(latest)
    st.markdown(f"**{gp.get('title', 'Grand prix')}** on `{gp.get('track')}` - :gray[{rel(latest)}, "
                f"{gp.get('created_utc', '')[:16].replace('T', ' ')} UTC]")
    rows = mark_dead_heats([{"pos": r.get("position"), "driver": r.get("name"), "author": r.get("author"),
                             "time": r.get("race_time"), "gap": r.get("gap"), "best lap": r.get("best_lap"),
                             "crashes": r.get("crashes"), "status": r.get("status") or "finished"}
                            for r in gp.get("standings", [])])
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"time": st.column_config.NumberColumn(format="%.2f s"),
                                "gap": st.column_config.NumberColumn(format="+%.2f"),
                                "best lap": st.column_config.NumberColumn(format="%.2f s")})
    replay_file = latest.with_name(latest.name.replace("_results.json", ".json"))
    if replay_file.exists():
        st.page_link("views/replay.py", label="Watch this race", icon=":material/play_circle:",
                     query_params={"file": rel(replay_file)})
