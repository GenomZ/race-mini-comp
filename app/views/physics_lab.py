"""Physics Lab: the three formulas behind every lap, and driving the car yourself (one tick per click)."""

import json
import math
import time

import streamlit as st

from agentic_gp.viz import plotly_figs as pf
from app.components import drive_pad, live_canvas, live_state, pad_controls
from app.lib.figs import stopping_figure, vmax_figure
from app.lib.loaders import track_choices, track_map
from app.lib.ui import page_header

page_header("Physics Lab", "Three formulas run the whole sim. Turning: R = v² / (20·|steering|), so twice the "
            "speed needs a 4x bigger circle. Braking: v² / 50 units to stop, plus v × 0.5 s you drive before you "
            "can even react. Corners: the fastest safe speed is √(20·R).", chapter="Ch1")

A_LAT, BRAKE, TICK, VMAX, YAW = 20.0, 25.0, 0.5, 40.0, 1.5

tab_turn, tab_stop, tab_corner, tab_drive = st.tabs(
    [":material/rotate_right: Turning circle", ":material/do_not_step: Stopping distance",
     ":material/speed: Corner speed", ":material/sports_esports: Hand-Drive"])


def turn_radius(v: float, s: float) -> float:
    """R = v^2 / (a_lat * |s|), never tighter than v / max_yaw_rate (slow cars turn on the spot-ish)."""
    if abs(s) < 1e-6 or v <= 0:
        return math.inf
    return max(v * v / (A_LAT * abs(s)), v / YAW)


# ------------------------------------------------------------------------------ turning circle
with tab_turn:
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    tspec = c1.selectbox("Track", track_choices(0), key="pl_track")
    tm = track_map(tspec)
    corners = tm["corners"]
    spots = ["start line"] + [f"before {c['id']} (R {c['min_radius']:.0f}, {c['direction']})" for c in corners]
    hairpin = 1 + min(range(len(corners)), key=lambda i: corners[i]["min_radius"]) if corners else 0
    spot = c2.selectbox("Put the car", spots, index=hairpin, key="pl_spot")
    speed = c3.slider("Speed (u/s)", 0.0, 40.0, 12.0, 1.0, key="pl_speed")
    steering = c4.slider("Steering (+ = right)", -1.0, 1.0, 0.5 if not corners or corners[hairpin - 1]["direction"] == "right"
                         else -0.5, 0.05, key="pl_steer")
    center = tm["centerline"]
    idx = 0 if spots.index(spot) == 0 else max(0, corners[spots.index(spot) - 1]["start_index"] - 6)
    (x0, y0), (x1, y1) = center[idx], center[(idx + 1) % len(center)]
    heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
    r = turn_radius(speed, steering)

    left, right = st.columns([3, 1], gap="medium")
    with left:
        fig = pf.track_figure(tspec, vmax=False, gates=False, height=560)
        pf.add_turning_circle(fig, x0, y0, heading, speed, steering)
        pf.add_car(fig, x0, y0, heading, color="#e63946", name="you")
        span = max(60.0, min(r if math.isfinite(r) else 60, 220) * 1.25)
        fig.update_xaxes(range=[x0 - span * 1.6, x0 + span * 1.6])
        fig.update_yaxes(range=[y0 - span, y0 + span])
        st.plotly_chart(fig, theme=None, key="pl_turn_fig")
    with right:
        st.metric("Turn radius R", "straight" if not math.isfinite(r) else f"{r:.0f} units")
        st.latex(r"R = \frac{v^2}{20\,|s|}")
        if corners and spots.index(spot) > 0:
            c = corners[spots.index(spot) - 1]
            room = c["min_radius"] + 0.5 * tm["half_width"]       # the centerline radius + some track width
            fits = math.isfinite(r) and r <= room
            (st.success if fits else st.error)(
                f"{c['id']} has a centerline radius of {c['min_radius']:.0f} (about {room:.0f} using the track width). "
                + ("This circle is tight enough." if fits else "Your circle is too wide: you hit the outside wall. "
                   "Slow down (R falls with v²) or steer more."), icon=":material/check:" if fits else ":material/close:")
        st.markdown("**Turn-radius explorer** (your steering)")
        st.dataframe([{"speed": v, "R": round(turn_radius(v, steering)) if math.isfinite(turn_radius(v, steering)) else None}
                      for v in (10, 20, 30, 40)], hide_index=True, width="stretch",
                     column_config={"R": st.column_config.NumberColumn("R (units)")})
        st.caption("Steering 0.5 at 10 / 20 / 30 u/s gives R = 10 / 40 / 90: double the speed, 4x the circle.")

# ------------------------------------------------------------------------------ stopping distance
with tab_stop:
    c1, c2 = st.columns(2)
    v = c1.slider("Speed (u/s)", 0.0, 40.0, 30.0, 1.0, key="pl_stop_v")
    front = c2.slider("Wall ahead (distance_to_wall_front)", 5.0, 200.0, 60.0, 5.0, key="pl_front")
    need = v * v / (2 * BRAKE) + v * TICK
    left, right = st.columns([3, 1], gap="medium")
    with left:
        st.plotly_chart(stopping_figure(v, brake=BRAKE, tick=TICK, front=front), theme=None, key="pl_stop_fig")
    with right:
        st.metric("Braking distance", f"{v * v / (2 * BRAKE):.1f} u")
        st.metric("+ blind tick", f"{v * TICK:.1f} u", help="You decide once per 0.5 s: the car covers v x 0.5 before your brake starts.")
        st.metric("Total to stop", f"{need:.1f} u")
        if need <= front:
            st.success(f"You can stop: {front - need:.0f} units to spare.", icon=":material/check:")
        else:
            st.error(f"Crash: you need {need - front:.0f} more units. Brake one tick earlier.", icon=":material/car_crash:")
        safe = max(0.0, -BRAKE * TICK + math.sqrt((BRAKE * TICK) ** 2 + 2 * BRAKE * front))
        st.caption(f"Fastest speed that can still stop before this wall: **{safe:.1f} u/s** "
                   f"(the Ch1 rule: throttle toward √(2·25·(front − margin))).")

# ------------------------------------------------------------------------------ corner speed
with tab_corner:
    c1, c2 = st.columns([1, 3])
    with c1:
        R = st.slider("Corner radius R", 5.0, 150.0, 30.0, 1.0, key="pl_R")
        st.metric("v_max = √(20·R)", f"{min(VMAX, math.sqrt(A_LAT * R)):.1f} u/s")
        st.caption("At v_max the car needs full steering (s = 1) to hold the line: any faster and R = v²/20 "
                   "is wider than the corner.")
        tspec2 = st.selectbox("Corners of track", track_choices(0), key="pl_track2")
    with c2:
        cs = track_map(tspec2)["corners"]
        st.plotly_chart(vmax_figure(cs, radius=R), theme=None, key="pl_vmax_fig")
    st.dataframe([{"corner": c["id"], "direction": c["direction"], "min radius": c["min_radius"], "v_max": c["v_max"],
                   "gates": f"{c['from_wp']}->{c['to_wp']}"} for c in cs], hide_index=True, width="stretch")

# ------------------------------------------------------------------------------ hand-drive
with tab_drive:
    st.caption("You are the driver: the same rules as your code. One decision = one 0.5 s tick. "
               "Everyone crashes once - watch the speed before the hairpin.")
    c1, c2, c3, c4, c5 = st.columns([2, 2.6, 1.8, 2, 1.5], vertical_alignment="bottom")
    hd_track = c1.selectbox("Track", track_choices(0), key="hd_track")
    mode = c2.segmented_control("Mode", ["Turn-based", "Real-time"], default="Turn-based", key="hd_mode",
                                help="Turn-based: each click (or key) = one tick. Real-time: hold the arrow keys; "
                                     "the sim steps every 0.5 s.")
    blind = c3.toggle("Blind mode", key="hd_blind", help="Hide the map: drive from the obs numbers only, like your code.")
    steer_amt = c4.slider("Steering per click", 0.1, 1.0, 0.6, 0.1, key="hd_steer")

    def _new_car() -> None:
        from agentic_gp.sim.env import RaceEnv

        env = RaceEnv(st.session_state.get("hd_track", "grand_prix"))
        st.session_state.hd_env = env
        st.session_state.hd_obs = env.reset()
        st.session_state.hd_trail = []
        st.session_state.hd_last = 0.0
        st.session_state.hd_action = (0.0, 0.0)
        st.session_state.hd_for = st.session_state.get("hd_track", "grand_prix")

    if "hd_env" not in st.session_state or st.session_state.get("hd_for") != hd_track:
        _new_car()
    c5.button("New car", icon=":material/restart_alt:", key="hd_reset", width="stretch", on_click=_new_car)

    def _tick(throttle: float, steering: float) -> None:
        obs = st.session_state.hd_obs
        if obs["attempt_over"]:
            return
        obs = st.session_state.hd_env.step(throttle, steering)
        st.session_state.hd_obs = obs
        st.session_state.hd_trail.append(obs["position"])
        st.session_state.hd_action = (throttle, steering)

    def view(status: str) -> None:
        env, obs = st.session_state.hd_env, st.session_state.hd_obs
        thr, steer = st.session_state.hd_action
        m = st.columns(5)
        m[0].metric("Speed", f"{obs['speed']:.1f}")
        m[1].metric("Gates", f"{obs['waypoints_passed']}/{obs['waypoints_total']}")
        m[2].metric("Crashes", obs["crashes"])
        m[3].metric("Race clock", f"{obs['race_time']:.1f} s")
        m[4].metric("Next waypoint", f"{obs['angle_to_next_waypoint']:+.0f} deg")
        if obs["finished"]:
            st.success(f"Finished in {obs['race_time']:.1f} s with {obs['crashes']} crash(es). The L1 reflex driver does "
                       "about 60-80 s clean. Beat it?", icon=":material/sports_score:")
        if blind:
            keys = ["speed", "distance_to_wall_front", "distance_to_wall_left_45", "distance_to_wall_right_45",
                    "distance_to_wall_left_90", "distance_to_wall_right_90", "angle_to_next_waypoint",
                    "distance_to_next_waypoint", "angle_to_waypoint_after_next", "crashes"]
            st.code(json.dumps({k: obs[k] for k in keys}, indent=1), language="json")
        else:
            live_canvas(live_state(hd_track, obs=obs, action=(thr, steer), name="You", color="#e63946", status=status,
                                   trail=st.session_state.hd_trail[-240:], crash_points=list(env.crash_points),
                                   tween_ms=450, follow=True, show_waypoints=True), key="hd_canvas", height=440)

    if mode == "Real-time":
        @st.fragment(run_every=0.1)
        def realtime_drive() -> None:
            pressed = drive_pad(key="hd_pad", title="Click here, then hold the arrow keys (or WASD)")
            throttle, steering = pad_controls(pressed, steer=steer_amt)
            engaged = any(pressed.values())
            if engaged and time.monotonic() - st.session_state.hd_last >= TICK:
                st.session_state.hd_last = time.monotonic()
                _tick(throttle, steering)
            view("driving" if engaged else "paused")

        realtime_drive()
    else:
        gas_turn = 0.4
        b = st.columns(5)
        b[0].button("Left", icon=":material/arrow_back:", shortcut="Left", key="hd_left", width="stretch",
                    on_click=_tick, args=(gas_turn, -steer_amt), help="Left arrow: steer left (with a little gas)")
        b[1].button("Gas", icon=":material/arrow_upward:", shortcut="Up", key="hd_gas", width="stretch",
                    on_click=_tick, args=(1.0, 0.0), help="Up arrow: full throttle", type="primary")
        b[2].button("Coast", icon=":material/pause:", shortcut="Space", key="hd_coast", width="stretch",
                    on_click=_tick, args=(0.0, 0.0), help="Space: no throttle, no steering")
        b[3].button("Brake", icon=":material/arrow_downward:", shortcut="Down", key="hd_brake", width="stretch",
                    on_click=_tick, args=(-1.0, 0.0), help="Down arrow: full brake")
        b[4].button("Right", icon=":material/arrow_forward:", shortcut="Right", key="hd_right", width="stretch",
                    on_click=_tick, args=(gas_turn, steer_amt), help="Right arrow: steer right (with a little gas)")
        view("paused")
        st.caption("Keyboard: arrows = Left / Gas / Brake / Right, Space = coast (click the page first). "
                   "Each press is exactly one 0.5 s tick, like one call of drive(obs).")
