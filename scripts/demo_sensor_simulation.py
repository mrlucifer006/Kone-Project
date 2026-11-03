#!/usr/bin/env python3
"""
ElevatorGuard-AI: Live Multi-Sensor Simulation & Fault Injection Demonstration
Demonstrates end-to-end edge telemetry ingestion, sensor defect injection,
ISO 4309 discard limit tracking, and dynamic RUL prediction.
"""

import sys
import time
import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

# ANSI Terminal Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BLUE = "\033[34m"
C_DIM = "\033[2m"


def print_banner():
    print(f"\n{C_CYAN}{C_BOLD}" + "=" * 78 + f"{C_RESET}")
    print(f"{C_CYAN}{C_BOLD}   🛗  ElevatorGuard-AI: Live Sensor Simulation & Fault Injection Demo  🛗{C_RESET}")
    print(f"{C_CYAN}{C_BOLD}" + "=" * 78 + f"{C_RESET}")
    print(f" Connecting to Edge API at {C_BLUE}{BASE_URL}{C_RESET}...")


def get_status():
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=2.0)
        return r.json()
    except Exception as e:
        print(f"{C_RED}Error connecting to {BASE_URL}: {e}{C_RESET}")
        return None


def dispatch_elevator(floor: int):
    try:
        r = requests.post(f"{BASE_URL}/api/elevator/dispatch", json={"floor": floor}, timeout=2.0)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def inject_defect(defect_type: str, severity: float):
    try:
        r = requests.post(
            f"{BASE_URL}/api/elevator/inject_defect",
            json={"defect_type": defect_type, "severity": severity},
            timeout=2.0
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def reset_alarms():
    try:
        r = requests.post(f"{BASE_URL}/api/elevator/reset_alarm", timeout=2.0)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_telemetry_hud(step_name: str, status_data: dict, extra_msg: str = ""):
    kin = status_data.get("kinematics", {})
    telem = status_data.get("telemetry", {})
    hlth = status_data.get("health", {})
    alarms = telem.get("alarms", {})

    tensions = telem.get("individual_tensions", [0, 0, 0, 0])
    imb_ratio = telem.get("tension_imbalance_ratio", 0.0) * 100.0
    rms_z = telem.get("accel_rms_z", 0.0)

    hi_comp = hlth.get("hi_composite", 1.0)
    hi_vis = hlth.get("hi_vision", 1.0)
    hi_tel = hlth.get("hi_telemetry", 1.0)
    rul_days = hlth.get("rul_days", 0.0)
    rul_cycles = hlth.get("rul_cycles", 0)
    status_tag = hlth.get("status", "NORMAL")

    # Determine status color
    if alarms.get("emergency_stop") or hi_comp < 0.25:
        col_status = C_RED
    elif hi_comp < 0.50 or alarms.get("tension_imbalance"):
        col_status = C_YELLOW
    else:
        col_status = C_GREEN

    # Simulating raw ESP32 protocol frame string
    t_now = int(time.time() * 1000)
    raw_packet = (
        f"$EG,{{\"type\":\"TELEMETRY\",\"ts\":{t_now},"
        f"\"flr\":{kin.get('floor')},\"vel\":{kin.get('velocity_mps', 0.0):.2f},"
        f"\"acc\":[0.02,0.01,{rms_z:.2f}],"
        f"\"tns\":[{int(tensions[0])},{int(tensions[1])},{int(tensions[2])},{int(tensions[3])}],"
        f"\"imb\":{imb_ratio:.1f}%,\"flg\":{1 if alarms.get('emergency_stop') else 0}}}*CRC"
    )

    print(f"\n{C_MAGENTA}{C_BOLD}▶ [{step_name}]{C_RESET} {extra_msg}")
    print(f"{C_DIM}┌─ ESP32 Serial Protocol Frame (UART 115200 Baud) ─────────────────────┐{C_RESET}")
    print(f"{C_DIM}│{C_RESET} {C_CYAN}{raw_packet}{C_RESET}")
    print(f"{C_DIM}└──────────────────────────────────────────────────────────────────────┘{C_RESET}")
    print(
        f"  {C_BOLD}Lift Kinematics:{C_RESET} Floor {C_BOLD}{kin.get('floor')}/8{C_RESET} | "
        f"Target: F{kin.get('target_floor')} | "
        f"Pos: {kin.get('position_m', 0.0):4.1f}m | "
        f"Vel: {kin.get('velocity_mps', 0.0):+4.2f}m/s | "
        f"State: {C_BOLD}{'IN MOTION' if kin.get('is_moving') else 'LEVELLED'}{C_RESET}"
    )
    print(
        f"  {C_BOLD}Load Cells (HX711):{C_RESET} "
        f"R1: {int(tensions[0])}N  R2: {int(tensions[1])}N  R3: {int(tensions[2])}N  R4: {int(tensions[3])}N "
        f"| Imbalance: {C_YELLOW if imb_ratio > 15 else C_GREEN}{imb_ratio:4.1f}%{C_RESET} "
        f"({'⚠️ ALARM >15%' if alarms.get('tension_imbalance') else 'NOMINAL'})"
    )
    print(
        f"  {C_BOLD}Vibration (MPU-6050):{C_RESET} RMS_Z: {rms_z:.3f} m/s² "
        f"({'VIBRATION ELEVATED' if alarms.get('vibration_exceeded') else 'NORMAL'})"
    )
    print(
        f"  {C_BOLD}Health Indices:{C_RESET} "
        f"HI_vis: {hi_vis:.3f} | HI_dyn: {hi_tel:.3f} | "
        f"HI_composite: {col_status}{hi_comp:.3f}{C_RESET} ({col_status}{status_tag}{C_RESET})"
    )
    print(
        f"  {C_BOLD}Dynamic RUL Prognostics:{C_RESET} "
        f"{C_BOLD}{rul_cycles:,}{C_RESET} Cycles | {col_status}{rul_days:.1f} Calendar Days Remaining{C_RESET}"
    )


def run_scenario():
    print_banner()

    # Step 0: Initial Reset
    print(f"\n{C_BLUE}Initializing test environment and resetting state...{C_RESET}")
    reset_alarms()
    time.sleep(1.0)

    # -------------------------------------------------------------
    # DEMO STAGE 1: Baseline Nominal Sensor Telemetry
    # -------------------------------------------------------------
    s0 = get_status()
    if not s0:
        print(f"{C_RED}Could not reach API. Please ensure the backend is running.{C_RESET}")
        return
    render_telemetry_hud(
        "STAGE 1: Baseline Nominal Operation",
        s0,
        "Elevator at rest on Floor 1. All suspension ropes balanced, vibration normal."
    )
    time.sleep(2.5)

    # -------------------------------------------------------------
    # DEMO STAGE 2: Dispatch Elevator to Floor 5 (Motion Dynamics)
    # -------------------------------------------------------------
    print(f"\n{C_CYAN}Dispatching elevator car from Floor 1 to Floor 5...{C_RESET}")
    dispatch_elevator(5)

    for i in range(4):
        time.sleep(0.8)
        st_moving = get_status()
        render_telemetry_hud(
            f"STAGE 2: Car In Motion ({i+1}/4)",
            st_moving,
            "Traction motor accelerating, dynamic tension fluctuating during travel."
        )

    # -------------------------------------------------------------
    # DEMO STAGE 3: Fault Injection - Rope Tension Imbalance
    # -------------------------------------------------------------
    print(f"\n{C_YELLOW}{C_BOLD}>> INJECTING FAULT: Differential Cable Stretch / Tension Imbalance (+28%){C_RESET}")
    inject_defect("tension_imbalance", 0.28)
    time.sleep(1.2)

    s_imb = get_status()
    render_telemetry_hud(
        "STAGE 3: Sensor Fault Injected (HX711 Imbalance)",
        s_imb,
        "Unequal rope loading detected. Load cell differential exceeded 15% safety threshold!"
    )
    time.sleep(3.0)

    # -------------------------------------------------------------
    # DEMO STAGE 4: Fault Injection - Crown Wire Breakage
    # -------------------------------------------------------------
    print(f"\n{C_YELLOW}{C_BOLD}>> INJECTING FAULT: NDT Camera Detects 3 Crown Wire Breaks on Cable #2{C_RESET}")
    inject_defect("broken_wire", 3)
    time.sleep(1.2)

    s_break = get_status()
    render_telemetry_hud(
        "STAGE 4: Visual Defect Injected (YOLO Broken Wire Detection)",
        s_break,
        "Visual HI degraded. BiLSTM model calculates accelerated lifecycle fatigue."
    )
    time.sleep(3.0)

    # -------------------------------------------------------------
    # DEMO STAGE 5: Critical Threshold & Emergency Brake
    # -------------------------------------------------------------
    print(f"\n{C_RED}{C_BOLD}>> INJECTING FAULT: 6 Wire Breaks (ISO 4309 Discard Limit Reached!){C_RESET}")
    inject_defect("broken_wire", 6)
    time.sleep(1.2)

    s_crit = get_status()
    render_telemetry_hud(
        "STAGE 5: ISO 4309 Discard Criterion Triggered",
        s_crit,
        "MANDATORY DISCARD LIMIT REACHED. Emergency interlock engaged to prevent catastrophic failure!"
    )
    time.sleep(3.5)

    # -------------------------------------------------------------
    # DEMO STAGE 6: Technician Maintenance & Reset
    # -------------------------------------------------------------
    print(f"\n{C_GREEN}{C_BOLD}>> SIMULATING MAINTENANCE: Cable Re-Tensioned & Discarded Cable Replaced{C_RESET}")
    reset_alarms()
    time.sleep(1.2)

    s_clean = get_status()
    render_telemetry_hud(
        "STAGE 6: Maintenance Reset Completed",
        s_clean,
        "New suspension cable commissioned. HI restored to nominal, all alarms cleared."
    )

    print(f"\n{C_CYAN}{C_BOLD}" + "=" * 78 + f"{C_RESET}")
    print(f"{C_GREEN}{C_BOLD} ✅  Demonstration Completed Successfully!{C_RESET}")
    print(f" View live charts and graphical twins anytime at: {C_BLUE}http://localhost:8501{C_RESET}")
    print(f"{C_CYAN}{C_BOLD}" + "=" * 78 + f"{C_RESET}\n")


if __name__ == "__main__":
    run_scenario()
