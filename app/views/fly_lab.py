"""Fly Lab: 4,494 real fruit-fly neurons (FlyWire) steer the car - and a scrambled brain cannot."""

import numpy as np
import streamlit as st

from app.components import live_canvas, live_state
from app.lib.experiments import fly_available, fly_race
from app.lib.figs import raster_figure, sankey_figure, tuning_curve_figure
from app.lib.ui import fast_mode, page_header

page_header("Fly Lab", "A brain is also an agent: sensors -> neurons -> motor commands, with no code that says "
            "'steer right'. Look at the real wiring, probe it like an electrophysiologist, and compare it with "
            "the same neurons wired at random (the scientific control).", chapter="Ch7")

ok, where = fly_available()
if not ok:
    st.warning("The FlyWire circuit file is not on this machine, so the Fly Lab has nothing to simulate yet.",
               icon=":material/bug_report:")
    st.markdown("**Instructor, once (about a minute, needs the pinned FlyWire inputs):**")
    st.code("python -m agentic_gp.fly.extract          # writes data/fly/steering_circuit_783.npz\n"
            "# or point to a copy:  set AGENTIC_GP_FLY_CIRCUIT=D:\\path\\steering_circuit_783.npz", language="bash")
    st.markdown("Meanwhile the toy spiking fly needs no data: `python race.py drive drivers/examples/fly_toy.py`. "
                "Details: `agentic_gp/fly/README.md`.")
    st.caption(where)
    st.stop()

from agentic_gp.fly import load_circuit  # noqa: E402  (only when the data exists)
from agentic_gp.fly.viz import (bearing_sweep, circuit_table, drive_summary, lateralization,  # noqa: E402
                                raster_from_snapshots, sankey_data)


@st.cache_resource(show_spinner="Loading the FlyWire circuit...")
def circuit():
    return load_circuit()


c = circuit()
m = st.columns(4)
m[0].metric("Neurons", f"{c.n:,}")
m[1].metric("Connections", f"{c.n_edges:,}")
m[2].metric("Synapses", f"{int(np.abs(c.syn).sum()):,}")
m[3].metric("Source", "FlyWire v783")
st.caption("Data derived from the FlyWire connectome (CC BY-NC 4.0, non-commercial teaching only; "
           "agentic_gp/fly/ATTRIBUTION.md). Spiking model after Shiu et al. 2024.")

tab_wiring, tab_ephys, tab_drive = st.tabs([":material/hub: The wiring", ":material/electric_bolt: Electrophysiology",
                                            ":material/directions_car: The fly drives"])

# ------------------------------------------------------------------------------ wiring
with tab_wiring:
    c1, c2 = st.columns([2, 1])
    min_syn = c1.slider("Hide links with fewer synapses than", 10, 500, 80, 10, key="fly_min")
    other = c2.toggle("Include all other neurons (pooled)", value=False, key="fly_other")
    st.plotly_chart(sankey_figure(sankey_data(c, min_synapses=min_syn, include_other=other)), theme=None,
                    key="fly_sankey")
    st.caption("Blue = eyes (LC10a sees a small target, LC16/LPLC2/LC4 see walls and looming), purple = hubs "
               "(AOTU019/025), gold = descending neurons that drive the body (DNa02 steers, MDN walks backwards, "
               "DNp01 = giant fibre escape). Green links excite, red links inhibit. _L/_R = brain side. "
               "Only forward links are drawn (eyes -> hubs -> body); the table below has everything.")
    with st.expander("Neurons per cell type and side", icon=":material/table:"):
        st.dataframe(circuit_table(c), hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ electrophysiology
with tab_ephys:
    st.markdown("Put a target at each bearing (no walls), let the brain run, read the steering neurons. "
                "If steering follows the target side, the **wiring** knows how to pursue.")
    c1, c2, c3 = st.columns(3)
    step = c1.select_slider("Bearing step (deg)", [15, 30, 45], value=45 if fast_mode() else 15, key="fly_step")
    seeds = c2.slider("Scrambled brains", 0, 5, 1 if fast_mode() else 3, key="fly_seeds",
                      help="Each keeps every neuron and its number of connections, but rewires them at random.")
    ms = c3.select_slider("Brain time per bearing (ms)", [100, 250, 500], value=100 if fast_mode() else 250, key="fly_ms")

    @st.cache_data(show_spinner="Stimulating the brain at every bearing...", max_entries=8)
    def sweep(step: int, seeds: int, ms: int) -> dict:
        return bearing_sweep(circuit(), angles=range(-90, 91, step), scramble_seeds=range(seeds), ms=float(ms),
                             warmup_ms=50.0)

    if st.button("Run the experiment", type="primary", key="fly_sweep_go", icon=":material/science:"):
        st.session_state.fly_sweep = sweep(step, seeds, ms)
    sw = st.session_state.get("fly_sweep")
    if sw:
        left, right = st.columns([3, 1], gap="medium")
        left.plotly_chart(tuning_curve_figure(sw), theme=None, key="fly_tuning")
        with right:
            st.metric("Lateralization, real", f"{lateralization(sw['real'], sw['angles']):+.2f}",
                      help="+1: always turns fully toward the target; 0: flat or random; negative: turns away.")
            for seed, curve in sw["scrambled"].items():
                st.metric(f"scrambled seed {seed}", f"{lateralization(curve, sw['angles']):+.2f}")
        st.caption("Read it like a lab result: which side does the response live on? What does the control prove - "
                   "and what does it NOT prove (our eyes, readout gains and speed rule are engineered)?")
    else:
        st.info("Press 'Run the experiment' (a few seconds per brain).", icon=":material/info:")

# ------------------------------------------------------------------------------ driving
with tab_drive:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1], vertical_alignment="bottom")
    track = c1.selectbox("Track", ["oval", "grand_prix", "sprint"], key="fly_track")
    wiring = c2.selectbox("Brain", ["real FlyWire wiring"] + [f"scrambled seed {s}" for s in range(5)]
                          + ["lesion DNa02", "lesion AOTU019", "lesion LC10a"], key="fly_wiring")
    ticks = c3.slider("Ticks to drive", 10, 300, 20 if fast_mode() else 160, 10, key="fly_ticks",
                      help="Each tick simulates 100 ms of brain time (about 20-75 ms of compute).")
    go = c4.button("Drive", type="primary", key="fly_go", icon=":material/play_arrow:")
    scramble = int(wiring.rsplit(" ", 1)[1]) if wiring.startswith("scrambled") else None
    lesion = wiring.split(" ", 1)[1] if wiring.startswith("lesion") else ""

    @st.cache_data(show_spinner="The fly is driving (100 ms of brain per tick)...", max_entries=12)
    def drive(track: str, scramble, lesion: str, ticks: int) -> dict:
        return fly_race(track, scramble, ticks, lesion)

    if go:
        try:
            st.session_state.fly_run = (wiring, drive(track, scramble, lesion, int(ticks)))
        except Exception as exc:                          # noqa: BLE001
            st.error(f"The fly could not drive: {type(exc).__name__}: {exc}")
    got = st.session_state.get("fly_run")
    if not got:
        st.info("Pick a brain and press Drive. Then scrub through the ticks: the raster shows which neurons fired "
                "while the car did what it did.", icon=":material/info:")
    else:
        label, run = got
        frames, snaps = run["frames"], run["snapshots"]
        res = (f"finished in {run['race_time']:.1f} s" if run["finished"]
               else f"{run['gates']}/{run['gates_total']} gates in {len(frames)} ticks")
        st.markdown(f"**{label}** on `{track}`: {res}, {run['crashes']} crash(es)")
        turn = next((i for i, g in enumerate(frames) if abs(g["steering"]) > 0.3), min(len(frames) - 1, 12))
        k = st.slider("Tick (start at the first real turn)", 0, len(frames) - 1, turn, key=f"fly_tick_{label}_{track}")             if len(frames) > 1 else 0
        f = frames[k]
        left, right = st.columns([1, 1], gap="medium")
        with left:
            trail = [[g["x"], g["y"]] for g in frames[: k + 1]]
            say = [{"t": g["t"], "text": g["say"]} for g in frames[: k + 1] if g.get("say")][-3:]
            live_canvas(live_state(track, car={"x": f["x"], "y": f["y"], "heading_deg": f["heading_deg"],
                                               "speed": f["speed"]},
                                   action=(f["throttle"], f["steering"]), name="Fly", color="#8338ec",
                                   status="paused", trail=trail, crash_points=run["crash_points"], say=say,
                                   seq=k, tween_ms=200, follow=True), key="fly_canvas", height=420)
        with right:
            groups = ["DNa02_L", "DNa02_R", "AOTU019_L", "AOTU019_R", "MDN_L", "MDN_R", "DNp01_L", "DNp01_R"]
            raster = raster_from_snapshots(snaps, groups)
            snap = snaps[k]
            cursor = float(snap["t_ms"]) - float(snap["window_ms"])
            st.plotly_chart(raster_figure(raster, groups, cursor_ms=cursor, window_ms=1500.0, height=300), theme=None,
                            key="fly_raster")
            r = snap["rates"]
            mm = st.columns(4)
            mm[0].metric("waypoint", f"{round(f['angle']):+d} deg")
            mm[1].metric("DNa02 L / R (Hz)", f"{r['DNa02_L']:.0f} / {r['DNa02_R']:.0f}")
            mm[2].metric("steering", f"{f['steering']:+.2f}")
            mm[3].metric("MDN brake", f"{snap['motor']['brake']:.2f}")
        summ = drive_summary(snaps)
        st.markdown("**Why it drove like that** (share of ticks with the waypoint off-centre)")
        s = st.columns(5)
        s[0].metric("steered toward", f"{summ.get('toward', 0) * 100:.0f} %")
        s[1].metric("steered away", f"{summ.get('away', 0) * 100:.0f} %")
        s[2].metric("too weak to steer", f"{summ.get('weak_steer', 0) * 100:.0f} %")
        s[3].metric("braking (MDN)", f"{summ.get('braking', 0) * 100:.0f} %")
        s[4].metric("emergency stops", f"{summ.get('emergency', 0) * 100:.0f} %")
        st.caption("Run the real brain, then a scrambled one: the scrambled fly loses by steering the wrong way, "
                   "stopping, or circling - the same neurons, different wiring.")
