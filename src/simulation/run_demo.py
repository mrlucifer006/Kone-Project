"""
ElevatorGuard-AI: Integrated System Demonstration Runner
Runs end-to-end multi-modal pipeline:
1. Verifies dataset & model checkpoints
2. Initializes ESP32 hardware simulator & protocol ingestor
3. Executes real-time hoistway physics & NDT vision defect analyzer
4. Launches visual lift simulation GUI or live telemetry demonstration
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.dataset_generator import SyntheticDataGenerator
from src.simulation.elevator_physics import ElevatorPhysicsSimulator
from src.simulation.esp32_simulator import VirtualESP32Device
from src.edge.serial_ingest import ESP32TelemetryIngestor
from src.telemetry.train_rul import train_bilstm_rul
from src.edge.export_onnx import export_bilstm_to_onnx


def bootstrap_environment():
    """Ensure baseline dataset and models exist."""
    print("=" * 70)
    print("        ElevatorGuard-AI: System Boot & Self-Test")
    print("=" * 70)

    # 1. Check datasets
    csv_path = Path("data/synthetic_degradation/lifecycle_telemetry.csv")
    if not csv_path.exists():
        print("[Bootstrap] Generating lifecycle telemetry and vision datasets...")
        gen = SyntheticDataGenerator()
        gen.generate_yolo_dataset(num_train=60, num_val=15)
        df = gen.generate_lifecycle_telemetry_dataset()
    else:
        print("[Bootstrap] Telemetry dataset found.")

    # 2. Check BiLSTM model checkpoint
    model_path = Path("models/bilstm_rul.pt")
    if not model_path.exists():
        print("[Bootstrap] Training baseline BiLSTM-Attention RUL model...")
        import pandas as pd
        df = pd.read_csv(csv_path)
        feature_cols = [
            "acc_rms_x", "acc_rms_y", "acc_rms_z", "acc_kurtosis_z",
            "acc_peak_z", "spectral_kurtosis_z", "tension_mean",
            "tension_std", "tension_imbalance_ratio"
        ]
        feats = df[feature_cols].values
        his = df["health_index"].values
        train_bilstm_rul(features=feats, health_indices=his, epochs=15)
        export_bilstm_to_onnx()
    else:
        print("[Bootstrap] BiLSTM model checkpoint verified.")

    print("[Bootstrap] All subsystems ready!\n")


def run_headless_simulation(duration_sec: int = 15):
    """Run simulation in terminal console mode with live ASCII dashboard."""
    print("Starting Headless ElevatorGuard-AI Simulation...\n")
    ingestor = ESP32TelemetryIngestor(virtual_mode=True)
    ingestor.start()

    esp32_sim = VirtualESP32Device(
        broadcast_rate_hz=20,
        packet_sink=ingestor.inject_virtual_packet
    )
    esp32_sim.start()

    print("Dispatching elevator from Floor 1 to Floor 8...")
    esp32_sim.physics.dispatch_to_floor(8)

    t_start = time.time()
    trip_phase = 0

    try:
        while (time.time() - t_start) < duration_sec:
            elapsed = time.time() - t_start
            state = esp32_sim.physics.state
            window = ingestor.get_latest_window(window_size=32)

            # Scenario events
            if elapsed > 4.0 and trip_phase == 0:
                print(">> [Fault Injection] Injecting Wire Break on Suspension Rope #2...")
                esp32_sim.physics.inject_defect("broken_wire", 2)
                trip_phase = 1
            elif elapsed > 8.0 and trip_phase == 1:
                print(">> [Fault Injection] Injecting Tension Imbalance across Ropes (+28%)...")
                esp32_sim.physics.inject_defect("tension_imbalance", 0.28)
                trip_phase = 2
            elif elapsed > 12.0 and trip_phase == 2:
                print(">> [Floor Call] Dispatching Elevator back to Floor 1...")
                esp32_sim.physics.dispatch_to_floor(1)
                trip_phase = 3

            # Display ASCII HUD
            tns = state.rope_tensions_n
            avg_t = sum(tns) / len(tns)
            imb = (max(tns) - min(tns)) / max(1.0, avg_t)
            status_tag = "CRITICAL STOP" if state.emergency_stop else ("IN MOTION" if state.is_moving else "LEVELLED")

            sys.stdout.write(
                f"\r[T={elapsed:04.1f}s] Floor: {state.floor}/8 | Pos: {state.position_m:04.1f}m | Vel: {state.velocity_mps:+04.2f}m/s | "
                f"Tensions: [{int(tns[0])}, {int(tns[1])}, {int(tns[2])}, {int(tns[3])}]N (Imb: {imb*100:04.1f}%) | "
                f"Wire Breaks: {state.wire_breaks} | HI: {state.hi_simulated:.3f} | {status_tag}   "
            )
            sys.stdout.flush()
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        esp32_sim.stop()
        ingestor.stop()
        print("\nHeadless simulation completed successfully.")


def run_gui_simulation():
    """Launch full Pygame graphical visual simulation."""
    from src.simulation.visual_lift_gui import VisualLiftSimulation
    sim = VisualLiftSimulation(headless=False)
    sim.run_standalone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElevatorGuard-AI Demonstration Runner")
    parser.add_argument("--mode", choices=["gui", "headless", "bootstrap"], default="gui")
    parser.add_argument("--duration", type=int, default=15, help="Duration for headless simulation in seconds")
    args = parser.parse_args()

    bootstrap_environment()

    if args.mode == "bootstrap":
        print("Bootstrap complete.")
    elif args.mode == "headless":
        run_headless_simulation(duration_sec=args.duration)
    else:
        # Check if display is available for GUI
        import os
        if "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ:
            run_gui_simulation()
        else:
            print("[Warning] No display server found. Falling back to headless simulation mode.")
            run_headless_simulation(duration_sec=args.duration)
