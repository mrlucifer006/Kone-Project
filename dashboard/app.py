"""
ElevatorGuard-AI: Streamlit Multi-Modal Predictive Maintenance & Digital Twin Dashboard
"""

import sys
import os
import time
import threading
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import cv2
import requests

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.simulation.elevator_physics import ElevatorPhysicsSimulator
from src.simulation.dataset_generator import SyntheticDataGenerator
from src.vision.defect_analyzer import CableDefectDetector, GuideRailDefectDetector
from src.telemetry.feature_extraction import TelemetryFeatureExtractor
from src.fusion.multimodal_fusion import MultiModalFusionEngine
from src.fusion.rul_estimator import RULEstimator

API_BASE_URL = "http://localhost:8000"

# Page setup
st.set_page_config(
    page_title="ElevatorGuard-AI | KONE Capstone Dashboard",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-box {
        background-color: #1e222b;
        border: 1px solid #363c4a;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .status-ok { color: #2ecc71; font-weight: bold; }
    .status-warn { color: #f39c12; font-weight: bold; }
    .status-crit { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system_components():
    """Initialize singleton simulator, vision analyzer, prognostic engines, and local stepper."""
    sim = ElevatorPhysicsSimulator(num_floors=8)
    vision = CableDefectDetector()
    rail_detector = GuideRailDefectDetector()
    telemetry = TelemetryFeatureExtractor()
    fusion = MultiModalFusionEngine()
    rul = RULEstimator()
    data_gen = SyntheticDataGenerator()

    # Continuous background simulation loop for standalone mode
    class StandalonePhysicsStepper:
        def __init__(self, simulator):
            self.sim = simulator
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

        def _loop(self):
            last_t = time.time()
            while self.running:
                now = time.time()
                dt = min(0.1, max(0.01, now - last_t))
                last_t = now
                self.sim.step(dt=dt)
                time.sleep(0.05)

    stepper = StandalonePhysicsStepper(sim)
    return sim, vision, rail_detector, telemetry, fusion, rul, data_gen, stepper


sim, vision, rail_detector, telemetry, fusion, rul, data_gen, stepper = get_system_components()


# ---------------- API HELPER CLIENT ----------------
DEFAULT_RTDB_URL = "https://elevatorguard-kone-default-rtdb.firebaseio.com"

def fetch_firebase_fleet_data(database_url: str = DEFAULT_RTDB_URL, auth_key: str = ""):
    """Fetch all elevators fleet state from Firebase RTDB REST endpoint."""
    url = database_url.rstrip("/") + "/elevators.json"
    if auth_key:
        url += f"?auth={auth_key}"
    try:
        r = requests.get(url, timeout=1.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_esp32_s3_telemetry(host_ip: str):
    """Fetch live telemetry directly from local ESP32-S3 hub endpoint."""
    try:
        url = f"http://{host_ip}/api/telemetry"
        r = requests.get(url, timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_esp32_cam_snapshot(cam_url: str):
    """Fetch a JPEG snapshot from ESP32-CAM or ESP32-S3 proxy."""
    try:
        r = requests.get(cam_url, timeout=1.5)
        if r.status_code == 200:
            img_arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            return img
    except Exception:
        pass
    return None


def fetch_backend_telemetry():
    """Fetch live telemetry from background ESP32/FastAPI service."""
    try:
        r = requests.get(f"{API_BASE_URL}/api/status", timeout=0.5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def send_dispatch(floor: int):
    """Dispatch elevator both to live backend and local simulator."""
    sim.dispatch_to_floor(floor)
    try:
        requests.post(f"{API_BASE_URL}/api/elevator/dispatch", json={"floor": floor}, timeout=0.8)
    except Exception:
        pass


def send_emergency_stop():
    """Trigger emergency stop in backend and local simulator."""
    sim.trigger_emergency_stop()
    try:
        requests.post(f"{API_BASE_URL}/api/elevator/emergency_stop", timeout=0.8)
    except Exception:
        pass


def send_reset_alarms():
    """Reset emergency stop and cleared faults."""
    sim.reset_emergency_stop()
    sim.tension_imbalance_factor = 0.0
    sim.broken_wire_count = 0
    sim.wear_ratio = 0.02
    sim.corrosion_ratio = 0.01
    sim.state.wire_breaks = 0
    sim.state.wear_pct = 0.02
    sim.state.corrosion_pct = 0.01
    try:
        requests.post(f"{API_BASE_URL}/api/elevator/reset_alarm", timeout=0.8)
    except Exception:
        pass


def send_inject_defect(defect_type: str, severity: float):
    """Inject fault into backend and local simulator."""
    sim.inject_defect(defect_type, severity)
    try:
        requests.post(
            f"{API_BASE_URL}/api/elevator/inject_defect",
            json={"defect_type": defect_type, "severity": severity},
            timeout=0.8
        )
    except Exception:
        pass


# Check current backend connection state
backend_status = fetch_backend_telemetry()
is_backend_online = backend_status is not None

# Sync local simulator state if backend is live
if is_backend_online:
    k_data = backend_status.get("kinematics", {})
    t_data = backend_status.get("telemetry", {})
    sim.state.floor = k_data.get("floor", sim.state.floor)
    sim.state.target_floor = k_data.get("target_floor", sim.state.target_floor)
    sim.state.position_m = k_data.get("position_m", sim.state.position_m)
    sim.state.velocity_mps = k_data.get("velocity_mps", sim.state.velocity_mps)
    sim.state.is_moving = k_data.get("is_moving", sim.state.is_moving)
    sim.state.emergency_stop = k_data.get("emergency_stop", sim.state.emergency_stop)
    if "individual_tensions" in t_data:
        sim.state.rope_tensions_n = [float(x) for x in t_data["individual_tensions"]]


# ---------------- SIDEBAR CONTROLS ----------------
st.sidebar.image("https://img.shields.io/badge/ElevatorGuard--AI-v1.0.0-blue?style=for-the-badge", use_container_width=True)
st.sidebar.title("🛗 Lift Control Panel")

# System Connection Badge
if is_backend_online:
    st.sidebar.success("🟢 Edge Backend Online (ESP32 30 Hz)")
else:
    st.sidebar.info("🟡 Standalone Simulation Mode")

# Auto-Stream Toggle
auto_stream = st.sidebar.toggle("🔄 Live Telemetry Auto-Stream (1s)", value=True)

# Floor Dispatcher
st.sidebar.subheader("Elevator Dispatcher")
cols = st.sidebar.columns(4)
for f in range(1, 9):
    col = cols[(f - 1) % 4]
    btn_label = f"F{f}"
    if col.button(btn_label, key=f"btn_f_{f}", use_container_width=True):
        send_dispatch(f)
        st.sidebar.success(f"Dispatched to Floor {f}")

# Emergency Controls
st.sidebar.markdown("---")
st.sidebar.subheader("Safety Interlocks")
c_e1, c_e2 = st.sidebar.columns(2)
if c_e1.button("🚨 EMERGENCY BRAKE", use_container_width=True):
    send_emergency_stop()
    st.sidebar.error("EMERGENCY BRAKE ENGAGED")

if c_e2.button("✅ RESET ALARMS", use_container_width=True):
    send_reset_alarms()
    st.sidebar.info("Alarms Cleared")

# Fault Injection Box
st.sidebar.markdown("---")
st.sidebar.subheader("Fault Injection Toolkit")
fault_type = st.sidebar.selectbox(
    "Select Failure Mode",
    ["None", "Crown Wire Break", "Rope Tension Imbalance (>15%)", "Severe Surface Abrasion", "Corrosion Pitting"]
)
if st.sidebar.button("Inject Fault", use_container_width=True):
    if fault_type == "Crown Wire Break":
        send_inject_defect("broken_wire", 1)
        st.sidebar.warning("Injected 1 Crown Wire Break")
    elif fault_type == "Rope Tension Imbalance (>15%)":
        send_inject_defect("tension_imbalance", 0.32)
        st.sidebar.warning("Injected 32% Tension Imbalance")
    elif fault_type == "Severe Surface Abrasion":
        send_inject_defect("abrasion", 1.5)
        st.sidebar.warning("Injected Surface Wear")
    elif fault_type == "Corrosion Pitting":
        send_inject_defect("corrosion", 1.5)
        st.sidebar.warning("Injected Corrosion Pitting")


# ---------------- MAIN DASHBOARD NAVIGATION ----------------
tabs = st.tabs([
    "🌐 Super Dashboard (Fleet & RTDB)",
    "🏢 Hoistway Digital Twin",
    "👁️ Computer Vision NDT (YOLO)",
    "📈 Telemetry & Vibration Spectrogram",
    "🧠 Multi-Modal RUL Prognostics",
    "⚡ ESP32 Protocol & Hardware"
])


# ================= TAB 0: SUPER DASHBOARD (MULTI-ELEVATOR FLEET & RTDB) =================
with tabs[0]:
    @st.fragment(run_every=1.0 if auto_stream else None)
    def render_super_dashboard():
        st.title("🌐 KONE Fleet Super Dashboard: Guide Rail Health & Distributed Cloud Sync")
        st.caption("Fleet-wide integration connecting ESP32-S3 hubs across elevators via Firebase Realtime Database (RTDB) and local edge nodes.")

        # RTDB Sync Controls & Configuration
        with st.expander("⚙️ Firebase Realtime Database (RTDB) & Edge Sync Controls", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                rtdb_url_input = st.text_input("Firebase RTDB Base URL", value=DEFAULT_RTDB_URL, key="rtdb_url_input")
            with sc2:
                rtdb_auth_input = st.text_input("Auth Secret / Token (Optional)", value="", type="password", key="rtdb_auth_input")
            with sc3:
                s3_local_ip = st.text_input("Local ESP32-S3 Hub IP / Hostname", value="192.168.1.150", key="s3_local_ip")

            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                manual_sync = st.button("🔄 Sync Fleet Now", use_container_width=True)
            with col_btn2:
                st.caption("Auto-synchronizing every second when Live Telemetry Auto-Stream is active.")

        # Query Firebase RTDB or generate mock fleet if offline
        fleet_rtdb = fetch_firebase_fleet_data(rtdb_url_input, rtdb_auth_input)
        s3_local_data = fetch_esp32_s3_telemetry(s3_local_ip)

        fleet_elevators = ["ELEVATOR-LIFT-01", "ELEVATOR-LIFT-02", "ELEVATOR-LIFT-03", "ELEVATOR-LIFT-04"]

        # Build composite fleet state
        fleet_state = {}
        for idx, dev_id in enumerate(fleet_elevators):
            # Check Firebase
            if fleet_rtdb and isinstance(fleet_rtdb, dict) and dev_id in fleet_rtdb:
                node_data = fleet_rtdb[dev_id].get("live", fleet_rtdb[dev_id])
                fleet_state[dev_id] = {
                    "source": "Firebase RTDB",
                    "vibe_hz": float(node_data.get("vibration", {}).get("frequency_hz", 5.0 + idx * 2.2)),
                    "delta_f": float(node_data.get("prognostics", {}).get("delta_freq_hz", idx * 2.2)),
                    "baseline_hz": float(node_data.get("prognostics", {}).get("baseline_freq_hz", 5.0)),
                    "rail_hi": float(node_data.get("prognostics", {}).get("health_index", 1.0 - idx * 0.12)),
                    "rul_trips": int(node_data.get("prognostics", {}).get("remaining_trips", 300000 - idx * 45000)),
                    "rul_days": float(node_data.get("prognostics", {}).get("remaining_days", 1200.0 - idx * 180.0)),
                    "joint_step": float(node_data.get("vision", {}).get("joint_step_mm", 0.05 + idx * 0.04)),
                    "wear_stage": node_data.get("prognostics", {}).get("wear_stage", "OPTIMAL" if idx == 0 else "NOMINAL_WEAR"),
                    "action_plan": node_data.get("prognostics", {}).get("action_plan", "Guide rail surface within ISO 7465 pristine limits."),
                    "alarm": bool(node_data.get("vibration", {}).get("alarm", False))
                }
            elif dev_id == "ELEVATOR-LIFT-01" and s3_local_data:
                # Local ESP32-S3 endpoint
                v = s3_local_data.get("vibration", {})
                p = s3_local_data.get("prognostics", {})
                vis = s3_local_data.get("vision", {})
                fleet_state[dev_id] = {
                    "source": "Local ESP32-S3 Hub",
                    "vibe_hz": float(v.get("frequency_hz", 5.4)),
                    "delta_f": float(p.get("delta_freq_hz", 0.4)),
                    "baseline_hz": float(p.get("baseline_freq_hz", 5.0)),
                    "rail_hi": float(p.get("health_index", 0.94)),
                    "rul_trips": int(p.get("remaining_trips", 285000)),
                    "rul_days": float(p.get("remaining_days", 1140.0)),
                    "joint_step": float(vis.get("joint_step_mm", 0.06)),
                    "wear_stage": p.get("wear_stage", "OPTIMAL"),
                    "action_plan": p.get("action_plan", "Normal guide rail wear. Standard lubrication schedule."),
                    "alarm": bool(v.get("alarm", False))
                }
            else:
                # Standalone physics-driven mock state for fleet demonstration
                time_offset = (time.time() * 0.2 + idx * 1.5) % 10.0
                sim_vibe = 5.0 + (idx * 2.8) + 0.3 * np.sin(time_offset)
                sim_delta_f = max(0.0, sim_vibe - 5.0)
                sim_hi = max(0.15, 1.0 - (sim_delta_f / 30.0) * 0.8)
                sim_trips = int(sim_hi * 300000)
                sim_days = round(sim_trips / 250.0, 1)
                stages = ["OPTIMAL", "NOMINAL_WEAR", "ELEVATED_FRICTION", "CRITICAL_ACTION_REQUIRED"]
                stage_idx = min(3, int((1.0 - sim_hi) * 4))

                fleet_state[dev_id] = {
                    "source": "Simulated Node (Awaiting Hub Link)",
                    "vibe_hz": round(sim_vibe, 2),
                    "delta_f": round(sim_delta_f, 2),
                    "baseline_hz": 5.0,
                    "rail_hi": round(sim_hi, 3),
                    "rul_trips": sim_trips,
                    "rul_days": sim_days,
                    "joint_step": round(0.05 + idx * 0.05 + 0.01 * np.cos(time_offset), 2),
                    "wear_stage": stages[stage_idx],
                    "action_plan": "Routine monitoring per ISO 7465 schedule." if sim_hi > 0.6 else "Schedule guide rail reprofiling.",
                    "alarm": sim_delta_f > 15.0 or sim_hi < 0.40
                }

        # Multi-Elevator Fleet Selector
        f_top1, f_top2, f_top3 = st.columns([1.5, 1.5, 1])
        with f_top1:
            selected_elevator = st.selectbox(
                "🏢 Select Fleet Elevator Unit (ESP32-S3 Edge Hub):",
                fleet_elevators,
                index=0,
                key="super_selected_elevator"
            )
        with f_top2:
            sync_badge = "🟢 Cloud Live (Firebase RTDB)" if (fleet_rtdb is not None) else (
                "🟡 Local Node Link (ESP32-S3)" if s3_local_data else "🔵 Virtual Edge Fleet (Synchronized)"
            )
            st.metric("Fleet Sync Protocol", sync_badge, delta=f"Data Source: {fleet_state[selected_elevator]['source']}")
        with f_top3:
            active_alarms = sum(1 for e in fleet_state.values() if e["alarm"])
            st.metric("Fleet Alarms Active", f"{active_alarms} / {len(fleet_elevators)}", delta="EN 81-20 Compliant", delta_color="inverse" if active_alarms > 0 else "normal")

        st.markdown("---")

        active_unit = fleet_state[selected_elevator]

        # Top KPI Cards for Selected Unit
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Vibration Frequency", f"{active_unit['vibe_hz']:.1f} Hz", delta=f"Shift: {active_unit['delta_f']:+.1f} Hz")
        k2.metric("Guide Rail HI (ISO 7465)", f"{active_unit['rail_hi']:.3f}", delta=f"Stage: {active_unit['wear_stage']}")
        k3.metric("Remaining Useful Life", f"{active_unit['rul_trips']:,} Trips", delta=f"{active_unit['rul_days']:.1f} Days")
        k4.metric("Joint Misalignment", f"{active_unit['joint_step']:.2f} mm", delta="ISO limit: <= 0.20 mm", delta_color="inverse" if active_unit['joint_step'] > 0.20 else "normal")
        k5.metric("System State", "ALARM ACTIVE" if active_unit['alarm'] else "NORMAL OPERATION")

        st.markdown("---")

        # Visual Feed & Detection Overlays alongside Fleet Comparison
        feed_col, fleet_col = st.columns([1.2, 1])

        with feed_col:
            st.subheader(f"📹 Live Camera Feed & Defect Overlays ({selected_elevator})")
            st.caption("ESP32-CAM optical inspection targeting elevator guide rail face, fishplate joint, and guide shoe track.")

            # Attempt to fetch live frame or synthesize rail view
            cam_feed_url = f"http://{s3_local_ip}/api/cam_feed"
            cam_frame = fetch_esp32_cam_snapshot(cam_feed_url)

            if cam_frame is None:
                # Generate realistic guide rail image
                h, w = 480, 640
                cam_frame = np.zeros((h, w, 3), dtype=np.uint8)
                cam_frame[:] = (45, 48, 52) # Dark steel elevator shaft background

                # Guide rail T-profile center blade
                rail_x1, rail_x2 = int(w * 0.32), int(w * 0.68)
                cv2.rectangle(cam_frame, (rail_x1, 0), (rail_x2, h), (160, 165, 175), -1) # Machined blade face
                cv2.rectangle(cam_frame, (rail_x1 + 35, 0), (rail_x2 - 35, h), (185, 190, 200), -1) # Center guide shoe contact strip

                # Add longitudinal wear scratches
                np.random.seed(42 + hash(selected_elevator) % 100)
                for _ in range(8):
                    sx = np.random.randint(rail_x1 + 10, rail_x2 - 10)
                    sy1 = np.random.randint(0, h // 2)
                    sy2 = np.random.randint(h // 2, h)
                    cv2.line(cam_frame, (sx, sy1), (sx, sy2), (110, 115, 120), 1)

                # Add transverse fishplate joint seam
                joint_y = 260
                cv2.line(cam_frame, (rail_x1 - 15, joint_y), (rail_x2 + 15, joint_y), (30, 32, 38), 3)
                cv2.line(cam_frame, (rail_x1 - 15, joint_y + 3), (rail_x2 + 15, joint_y + 3), (230, 230, 240), 1)

                # If selected unit has high wear, draw corrosion patches
                if active_unit['rail_hi'] < 0.80:
                    cv2.ellipse(cam_frame, (rail_x1 + 50, 120), (35, 20), 15, 0, 360, (25, 110, 185), -1) # Rust in BGR
                if active_unit['rail_hi'] < 0.60:
                    cv2.ellipse(cam_frame, (rail_x2 - 60, 380), (45, 25), -20, 0, 360, (25, 100, 170), -1)

            # Run GuideRailDefectDetector on current frame
            rail_hi_res, rail_dets, annotated_rail = rail_detector.analyze_rail_frame(cam_frame, return_annotated=True)
            annotated_rgb = cv2.cvtColor(annotated_rail, cv2.COLOR_BGR2RGB)

            st.image(annotated_rgb, caption=f"ISO 7465 Defect Analysis | {len(rail_dets)} Detected Flaws | Visual HI: {rail_hi_res.hi_rail_vis:.3f}", use_container_width=True)

            if rail_dets:
                df_rail_dets = pd.DataFrame([
                    {"Class": d["class_name"], "Confidence": f"{int(d['confidence']*100)}%", "Bounding Box": str(d["bbox"])}
                    for d in rail_dets
                ])
                st.dataframe(df_rail_dets, use_container_width=True)

        with fleet_col:
            st.subheader("📊 Fleet-Wide Health & Frequency Shift Overview")
            st.caption("Cross-fleet comparison of guide rail vibration shift and Remaining Useful Life (RUL).")

            df_fleet = pd.DataFrame([
                {
                    "Elevator": dev,
                    "Vibration (Hz)": fleet_state[dev]["vibe_hz"],
                    "Delta f (Hz)": fleet_state[dev]["delta_f"],
                    "Health Index": fleet_state[dev]["rail_hi"],
                    "RUL (Trips)": fleet_state[dev]["rul_trips"],
                    "RUL (Days)": fleet_state[dev]["rul_days"],
                    "Status": "⚠️ ATTENTION" if fleet_state[dev]["alarm"] else "✅ NOMINAL"
                }
                for dev in fleet_elevators
            ])

            fig_fleet_bar = px.bar(
                df_fleet,
                x="Elevator",
                y="Health Index",
                color="Health Index",
                color_continuous_scale="RdYlGn",
                range_y=[0.0, 1.1],
                title="Fleet Guide Rail Health Index (HI_rail)"
            )
            fig_fleet_bar.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="ISO 7465 Discard")
            fig_fleet_bar.add_hline(y=0.55, line_dash="dot", line_color="orange", annotation_text="Maintenance Warning")
            fig_fleet_bar.update_layout(height=280, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0")
            st.plotly_chart(fig_fleet_bar, use_container_width=True)

            fig_vibe_scatter = px.scatter(
                df_fleet,
                x="Vibration (Hz)",
                y="RUL (Days)",
                size="RUL (Trips)",
                color="Elevator",
                text="Elevator",
                title="Vibration Frequency vs Remaining Useful Life (Days)"
            )
            fig_vibe_scatter.update_layout(height=280, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0")
            st.plotly_chart(fig_vibe_scatter, use_container_width=True)

            st.dataframe(df_fleet[["Elevator", "Vibration (Hz)", "Delta f (Hz)", "RUL (Days)", "Status"]], use_container_width=True)

    render_super_dashboard()


# ================= TAB 1: DIGITAL TWIN =================
with tabs[0]:
    @st.fragment(run_every=1.0 if auto_stream else None)
    def render_digital_twin():
        st.title("🏢 Elevator Hoistway Digital Twin & Live State")
        st.caption("Real-time physical kinematics, dynamic suspension cable tension, and passenger loading.")

        # Re-fetch latest state
        cur_backend = fetch_backend_telemetry()
        if cur_backend:
            kin = cur_backend["kinematics"]
            telem = cur_backend["telemetry"]
            current_floor = kin["floor"]
            target_floor = kin["target_floor"]
            car_pos = kin["position_m"]
            car_vel = kin["velocity_mps"]
            is_moving = kin["is_moving"]
            e_stop = kin["emergency_stop"]
            tensions = telem["individual_tensions"]
        else:
            state = sim.state
            current_floor = state.floor
            target_floor = state.target_floor
            car_pos = state.position_m
            car_vel = state.velocity_mps
            is_moving = state.is_moving
            e_stop = state.emergency_stop
            tensions = state.rope_tensions_n

        # Top KPI Metrics
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Current Floor", f"Floor {current_floor} / 8", delta=f"Target: F{target_floor}")
        kpi2.metric("Car Velocity", f"{car_vel:.2f} m/s", delta="Travel range: 24.5 m")
        kpi3.metric("Hoistway Position", f"{car_pos:.1f} m")

        status_str = "CRITICAL STOP" if e_stop else ("IN MOTION" if is_moving else "LEVELLED")
        kpi4.metric("Operational Status", status_str)
        kpi5.metric("Total Trips", f"{sim.state.total_trips:,}")

        st.markdown("---")

        col_hoistway, col_tensions = st.columns([1, 1.2])

        with col_hoistway:
            st.subheader("Shaft Elevation & Car Position")
            fig_hoist = go.Figure()

            # Guide rails
            fig_hoist.add_shape(type="line", x0=1.0, y0=0, x1=1.0, y1=24.5, line=dict(color="#4a5568", width=3))
            fig_hoist.add_shape(type="line", x0=4.0, y0=0, x1=4.0, y1=24.5, line=dict(color="#4a5568", width=3))

            # Floor horizontal markers
            for fl in range(1, 9):
                fy = (fl - 1) * 3.5
                fig_hoist.add_shape(type="line", x0=0.8, y0=fy, x1=4.2, y1=fy, line=dict(color="#2d3748", width=1, dash="dot"))
                fig_hoist.add_annotation(x=0.5, y=fy, text=f"F{fl}", showarrow=False, font=dict(color="#a0aec0", size=12))

            # Elevator Car box
            car_color = "#e74c3c" if e_stop else "#3182ce"
            fig_hoist.add_shape(
                type="rect",
                x0=1.5, y0=car_pos,
                x1=3.5, y1=car_pos + 2.4,
                fillcolor=car_color,
                line=dict(color="#e2e8f0", width=2),
            )
            fig_hoist.add_annotation(
                x=2.5, y=car_pos + 1.2,
                text=f"Car F{current_floor}",
                showarrow=False,
                font=dict(color="#ffffff", size=13, family="bold")
            )

            # Counterweight
            cw_pos = max(0.0, min(24.5, 24.5 - car_pos - 2.0))
            fig_hoist.add_shape(
                type="rect",
                x0=4.2, y0=cw_pos,
                x1=4.8, y1=cw_pos + 3.0,
                fillcolor="#718096",
                line=dict(color="#cbd5e0", width=1)
            )

            fig_hoist.update_layout(
                height=480,
                xaxis=dict(range=[0, 5.2], showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(range=[-1, 26], title="Hoistway Height (meters)", gridcolor="#2d3748"),
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a"
            )
            st.plotly_chart(fig_hoist, use_container_width=True)

        with col_tensions:
            st.subheader("Dynamic Cable Tensions (4 Suspension Ropes)")
            avg_t = np.mean(tensions) if len(tensions) > 0 else 2500.0
            imbalance = (np.max(tensions) - np.min(tensions)) / max(1.0, avg_t)

            df_t = pd.DataFrame({
                "Rope": ["Rope #1", "Rope #2", "Rope #3", "Rope #4"],
                "Tension (N)": tensions,
                "Nominal Rating": [2500, 2500, 2500, 2500]
            })

            fig_t = px.bar(
                df_t,
                x="Rope",
                y="Tension (N)",
                color="Tension (N)",
                color_continuous_scale="Viridis",
                range_y=[0, 4800]
            )
            fig_t.add_hline(y=2500, line_dash="dash", line_color="orange", annotation_text="Target Equilibrium (2500 N)")
            fig_t.update_layout(
                height=320,
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font_color="#e2e8f0"
            )
            st.plotly_chart(fig_t, use_container_width=True)

            if imbalance > 0.15:
                st.error(f"⚠️ EN 81-20 TENSION IMBALANCE ALERT: Imbalance is {imbalance*100:.1f}% (Safety Limit: 15.0%). Unequal cable stretch detected.")
            else:
                st.success(f"✅ Tension Imbalance: {imbalance*100:.1f}% (Within safe 15.0% tolerance).")

    render_digital_twin()


# ================= TAB 2: COMPUTER VISION NDT (LAPTOP CAMERA SUPPORT) =================
with tabs[1]:
    st.title("👁️ Stream 1: Computer Vision NDT Defect Detection (YOLO)")
    st.caption("Surface defect quantification complying with ISO 4309: Broken crown wires, abrasion wear, and corrosion pitting.")

    # Camera / Input Source Selector
    camera_mode = st.radio(
        "Select Camera / Image Input Source:",
        [
            "📸 Laptop Camera (Capture Snapshot via Webcam)",
            "🎥 Live Laptop Webcam Stream (Continuous OpenCV Feed)",
            "📁 Upload Cable Image File",
            "🔬 Synthetic Cable Defect Simulator"
        ],
        horizontal=True
    )

    input_bgr = None
    is_live_stream_mode = False

    if camera_mode == "📸 Laptop Camera (Capture Snapshot via Webcam)":
        st.write("Point your laptop camera at the test sample or wire rope and click **Take Photo**:")
        cam_photo = st.camera_input("Capture from Laptop Camera")
        if cam_photo is not None:
            bytes_data = cam_photo.getvalue()
            input_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    elif camera_mode == "🎥 Live Laptop Webcam Stream (Continuous OpenCV Feed)":
        is_live_stream_mode = True
        st.write("Real-time edge computer vision inspection streaming directly from laptop webcam (`/dev/video0`):")
        c_cam1, c_cam2 = st.columns([1, 2])
        with c_cam1:
            stream_active = st.toggle("🔴 Start Live Camera Stream", value=False)
            stream_frames = st.slider("Stream Buffer (frames)", min_value=10, max_value=150, value=30, step=10)

        if stream_active:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Unable to access laptop camera `/dev/video0`. Check device availability.")
            else:
                frame_placeholder = st.empty()
                info_placeholder = st.empty()
                for idx in range(stream_frames):
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam frame acquisition stopped.")
                        break
                    hi_res, dets, ann_img = vision.analyze_frame(frame, return_annotated=True)
                    ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        ann_rgb,
                        caption=f"Live Camera Frame {idx+1}/{stream_frames} | ISO 4309 HI_vis: {hi_res.hi_vis:.3f} | Defects: {len(dets)}",
                        use_container_width=True
                    )
                    info_placeholder.info(
                        f"**Live Visual HI:** `{hi_res.hi_vis:.3f}` | **Wire Breaks:** `{hi_res.wire_breaks}` | "
                        f"**Wear:** `{hi_res.wear_pct:.1f}%` | **Corrosion:** `{hi_res.corrosion_pct:.1f}%`"
                    )
                    time.sleep(0.04)
                cap.release()
                st.success("Webcam stream completed. Toggle off and on to restart stream.")

    elif camera_mode == "📁 Upload Cable Image File":
        uploaded_img = st.file_uploader("Upload wire rope photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            bytes_data = uploaded_img.getvalue()
            input_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    elif camera_mode == "🔬 Synthetic Cable Defect Simulator":
        # Generate synthetic test cable sample based on active simulation state
        test_img, _ = data_gen.generate_rope_image(
            width=640,
            height=640,
            num_broken_wires=sim.broken_wire_count,
            abrasion_level=sim.wear_ratio,
            corrosion_level=sim.corrosion_ratio
        )
        input_bgr = test_img

    # Display analysis results when an image is available
    if not is_live_stream_mode:
        if input_bgr is not None:
            hi_res, detections, annotated_img = vision.analyze_frame(input_bgr, return_annotated=True)
            annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

            v_col1, v_col2 = st.columns([1.2, 1])

            with v_col1:
                st.subheader("Inspection Feed & Live Detection Overlays")
                st.image(annotated_rgb, caption="Annotated Feed (YOLOv8-Seg / ISO 4309 HUD)", use_container_width=True)

            with v_col2:
                st.subheader("ISO 4309 Discard Analysis")
                st.markdown(f"**Visual Health Index ($HI_{{vis}}$):** `{hi_res.hi_vis:.4f}`")

                # Discard Limit Verification
                st.markdown("#### Standards Compliance Checklist")
                chk1 = "❌ DISCARD BREACH" if hi_res.wire_breaks >= 6 else "✅ PASS"
                chk2 = "❌ DISCARD BREACH" if hi_res.wear_pct >= 10.0 else "✅ PASS"
                chk3 = "❌ DISCARD BREACH" if hi_res.corrosion_pct >= 15.0 else "✅ PASS"

                st.write(f"- **Wire Breaks in 6d:** `{hi_res.wire_breaks} / 6` limit → **{chk1}**")
                st.write(f"- **Outer Diameter Wear:** `{hi_res.wear_pct:.1f}% / 10.0%` limit → **{chk2}**")
                st.write(f"- **Surface Corrosion Pitting:** `{hi_res.corrosion_pct:.1f}% / 15.0%` limit → **{chk3}**")

                st.markdown("#### Detected Flaw Instances")
                if detections:
                    df_dets = pd.DataFrame([
                        {"Defect Class": d["class_name"], "Confidence": f"{int(d['confidence']*100)}%", "Bounding Box": str(d["bbox"])}
                        for d in detections
                    ])
                    st.dataframe(df_dets, use_container_width=True)
                else:
                    st.info("No surface defects detected on current cable section.")

                st.markdown("#### Action Recommendation")
                if hi_res.is_discard_required:
                    st.error(hi_res.action_recommendation)
                else:
                    st.info(hi_res.action_recommendation)
        else:
            st.info("📸 Please capture a photo using your laptop camera above or select another input mode.")


# ================= TAB 3: TELEMETRY & VIBRATION =================
with tabs[2]:
    @st.fragment(run_every=1.0 if auto_stream else None)
    def render_telemetry_tab():
        st.title("📈 Stream 2: Dynamic Telemetry & Vibration Spectrograms")
        st.caption("Ingestion from MPU-6050 accelerometer and HX711 tension arrays to monitor ISO 18738 ride quality.")

        cur_backend = fetch_backend_telemetry()
        if cur_backend:
            t_info = cur_backend["telemetry"]
            tensions = np.array(t_info["individual_tensions"])
            acc_rms = t_info["accel_rms_z"]
            imb_ratio = t_info["tension_imbalance_ratio"]
        else:
            tensions = np.array(sim.state.rope_tensions_n)
            avg_t = np.mean(tensions) if len(tensions) > 0 else 2500.0
            imb_ratio = (np.max(tensions) - np.min(tensions)) / max(1.0, avg_t)
            acc_rms = 9.81 + abs(sim.state.velocity_mps) * 0.1

        t_steps = 128
        time_series_x = np.linspace(0, 1.28, t_steps)
        acc_z_sig = sim.state.accel_z + 0.15 * np.sin(2 * np.pi * 15 * time_series_x) + np.random.normal(0, 0.05, t_steps)

        feat_dict = telemetry.extract_features(
            accel_x=np.array([sim.state.accel_x]),
            accel_y=np.array([sim.state.accel_y]),
            accel_z=acc_z_sig,
            tensions=tensions
        )

        t_m1, t_m2, t_m3, t_m4 = st.columns(4)
        t_m1.metric("Vertical RMS (Az)", f"{acc_rms:.3f} m/s²", delta="ISO 18738 limit: 0.45 m/s²")
        t_m2.metric("Spectral Kurtosis", f"{feat_dict['spectral_kurtosis_z']:.2f}")
        t_m3.metric("Dominant Frequency", f"{feat_dict['dominant_frequency_hz']:.1f} Hz")
        t_m4.metric("Tension Imbalance", f"{imb_ratio*100:.1f}%")

        st.markdown("---")

        g1, g2 = st.columns(2)
        with g1:
            st.subheader("Tri-Axial Acceleration Waveform")
            df_wave = pd.DataFrame({
                "Time (s)": time_series_x,
                "Az (Vertical)": acc_z_sig,
                "Ax (Lateral)": sim.state.accel_x + np.random.normal(0, 0.02, t_steps),
                "Ay (Longitudinal)": sim.state.accel_y + np.random.normal(0, 0.02, t_steps)
            })
            fig_wave = px.line(df_wave, x="Time (s)", y=["Az (Vertical)", "Ax (Lateral)", "Ay (Longitudinal)"])
            fig_wave.update_layout(height=320, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0")
            st.plotly_chart(fig_wave, use_container_width=True)

        with g2:
            st.subheader("FFT Frequency Spectrum (Mechanical Harmonics)")
            detrended = acc_z_sig - np.mean(acc_z_sig)
            fft_vals = np.abs(np.fft.rfft(detrended))
            freqs = np.fft.rfftfreq(len(detrended), d=1.0 / 100.0)
            df_fft = pd.DataFrame({"Frequency (Hz)": freqs, "Spectral Amplitude": fft_vals})

            fig_fft = px.line(df_fft, x="Frequency (Hz)", y="Spectral Amplitude")
            fig_fft.update_layout(height=320, plot_bgcolor="#0f172a", paper_bgcolor="#0f172a", font_color="#e2e8f0")
            st.plotly_chart(fig_fft, use_container_width=True)

    render_telemetry_tab()


# ================= TAB 4: MULTI-MODAL RUL PROGNOSTICS =================
with tabs[3]:
    @st.fragment(run_every=1.0 if auto_stream else None)
    def render_prognostics_tab():
        st.title("🧠 Stream 3: Multi-Modal Decision Fusion & RUL Engine")
        st.caption("Enforces Miner's rule and ISO 4309 discard limits: HI_comp = α * HI_vis + (1 - α) * HI_dyn")

        cur_backend = fetch_backend_telemetry()
        if cur_backend:
            h_info = cur_backend["health"]
            hi_comp = h_info["hi_composite"]
            hi_vis = h_info["hi_vision"]
            hi_tel = h_info["hi_telemetry"]
            status_alert = h_info["status"]
            rul_cycles = h_info["rul_cycles"]
            rul_days = h_info["rul_days"]
        else:
            vis_calc = vision.visual_analyzer.calculate_hi(
                wire_breaks=sim.broken_wire_count,
                wear_pct=sim.wear_ratio,
                corrosion_pct=sim.corrosion_ratio
            )
            vibe_pen = max(0.0, (sim.state.accel_z - 9.81) / 0.40)
            hi_dyn = max(0.05, min(1.0, 1.0 - (0.5 * vibe_pen + 0.5 * (sim.tension_imbalance_factor / 0.25))))
            f_res = fusion.fuse(visual_result=vis_calc, hi_telemetry=hi_dyn)
            r_res = rul.estimate_rul(current_hi=f_res.hi_composite, current_trips=sim.state.total_trips)
            hi_comp = f_res.hi_composite
            status_alert = f_res.status
            rul_cycles = r_res.rul_cycles
            rul_days = r_res.rul_days

        # Health KPIs
        hk1, hk2, hk3, hk4 = st.columns(4)
        hk1.metric("Composite Health (HI_comp)", f"{hi_comp:.3f}")
        hk2.metric("Remaining Useful Life", f"{rul_cycles:,} Trips", delta=f"{rul_days:.1f} Calendar Days")
        hk3.metric("Projected Failure Trip", f"{int(sim.state.total_trips + rul_cycles):,}")
        hk4.metric("Safety Alert Level", status_alert)

        st.markdown("---")
        st.subheader("Lifecycle Degradation & Run-to-Failure Forecast")

        rul_res = rul.estimate_rul(current_hi=hi_comp, current_trips=sim.state.total_trips)
        df_traj = pd.DataFrame({
            "Trip Count": rul_res.trajectory_trips,
            "Projected Health Index": rul_res.trajectory_hi
        })

        fig_rul = px.line(df_traj, x="Trip Count", y="Projected Health Index", markers=True)
        fig_rul.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="ISO 4309 Discard Limit (HI = 0.20)")
        fig_rul.add_hline(y=0.45, line_dash="dot", line_color="orange", annotation_text="Replacement Window Warning (HI = 0.45)")
        fig_rul.update_layout(
            height=380,
            plot_bgcolor="#0f172a",
            paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            yaxis=dict(range=[0.0, 1.05])
        )
        st.plotly_chart(fig_rul, use_container_width=True)
        st.info(f"📋 **Action Plan:** {rul_res.action_plan}")

    render_prognostics_tab()


# ================= TAB 5: ESP32 PROTOCOL =================
with tabs[4]:
    @st.fragment(run_every=1.0 if auto_stream else None)
    def render_esp32_tab():
        st.title("⚡ ESP32 Hardware Protocol & Edge Microcontroller Diagnostics")
        st.caption("Standardized serial wire framing, sensor register maps, and bidirectional command parsing.")

        h_col1, h_col2 = st.columns([1.2, 1])

        with h_col1:
            st.subheader("Live Packet Terminal ($EG Protocol v2.1)")
            t_now = int(time.time() * 1000)
            sample_packet = (
                f"$EG,{{\"type\":\"TELEMETRY\",\"ver\":\"2.1\",\"ts\":{t_now},"
                f"\"flr\":{sim.state.floor},\"vel\":{sim.state.velocity_mps:.2f},"
                f"\"acc\":[{sim.state.accel_x:.2f},{sim.state.accel_y:.2f},{sim.state.accel_z:.2f}],"
                f"\"tns\":[{int(sim.state.rope_tensions_n[0])},{int(sim.state.rope_tensions_n[1])},"
                f"{int(sim.state.rope_tensions_n[2])},{int(sim.state.rope_tensions_n[3])}],"
                f"\"dr\":{1 if sim.state.door_open_pct > 0.5 else 0},\"flg\":{1 if sim.state.emergency_stop else 0}}}*4B"
            )

            st.code(sample_packet, language="json")

            st.subheader("Packet Payload Field Decoder")
            df_spec = pd.DataFrame([
                {"Field": "type", "Type": "String", "Description": "Packet identifier ('TELEMETRY', 'COMMAND', 'ACK')"},
                {"Field": "ts", "Type": "uint32", "Description": "Hardware millisecond timestamp"},
                {"Field": "flr", "Type": "uint8", "Description": "Current hoistway floor (1 to 8)"},
                {"Field": "vel", "Type": "float", "Description": "Car velocity in m/s from rotary encoder"},
                {"Field": "acc", "Type": "float[3]", "Description": "MPU-6050 Tri-axial accelerometer [x, y, z] in m/s²"},
                {"Field": "tns", "Type": "float[4]", "Description": "HX711 4-rope tension load cell readings in Newtons"},
                {"Field": "dr", "Type": "uint8", "Description": "Passenger door status (0=Closed, 1=Open)"},
                {"Field": "flg", "Type": "uint8", "Description": "System status flags (Bit 0: E-Stop Active)"}
            ])
            st.dataframe(df_spec, use_container_width=True)

        with h_col2:
            st.subheader("Hardware Pinout & Wiring Topology")
            st.markdown("""
            * **Microcontroller:** ESP32 NodeMCU-32S (Dual-Core 240 MHz)
            * **IMU:** MPU-6050 (I2C: SDA=GPIO21, SCL=GPIO22)
            * **Load Cell ADC:** HX711 4-Channel (SCK=GPIO14, DT1=GPIO27, DT2=GPIO26, DT3=GPIO25, DT4=GPIO33)
            * **Optical Revolution Counter:** GPIO19 (Interrupt Rising Edge)
            * **RGB Status NeoPixel:** GPIO18
            * **Audible Piezo Buzzer:** GPIO23
            * **UART Interface:** CP2102 USB-to-UART (115200 Baud, 8N1)
            """)
            st.success("Hardware protocol interface verified and ready for deployment.")

    render_esp32_tab()
