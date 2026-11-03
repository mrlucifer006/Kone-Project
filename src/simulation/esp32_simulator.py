"""
ElevatorGuard-AI: Virtual ESP32 Hardware Simulator
Simulates physical ESP32 NodeMCU microcontroller behavior, streaming
real-time sensor telemetry packets according to the ElevatorGuard-AI protocol.
"""

import time
import threading
from typing import Optional, Callable
from src.edge.esp32_protocol import ElevatorGuardESP32Protocol, PacketType
from src.simulation.elevator_physics import ElevatorPhysicsSimulator


class VirtualESP32Device:
    """
    Simulates physical ESP32 running edge firmware.
    Generates telemetry frames and processes incoming control commands.
    """

    def __init__(
        self,
        physics_sim: Optional[ElevatorPhysicsSimulator] = None,
        broadcast_rate_hz: int = 50,
        packet_sink: Optional[Callable[[str], None]] = None
    ):
        self.physics = physics_sim or ElevatorPhysicsSimulator()
        self.rate_hz = broadcast_rate_hz
        self.packet_sink = packet_sink
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.optical_pulse_counter = 0

    def start(self):
        """Start simulated ESP32 clock loop."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._thread.start()
        print(f"[Virtual ESP32] Running at {self.rate_hz} Hz.")

    def stop(self):
        """Stop simulated ESP32."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[Virtual ESP32] Stopped.")

    def handle_command_packet(self, raw_cmd: str):
        """Parse incoming command from host edge controller."""
        is_valid, data, _ = ElevatorGuardESP32Protocol.decode_packet(raw_cmd)
        if not is_valid or not data:
            return

        cmd = data.get("cmd")
        params = data.get("params", {})

        if cmd == "DISPATCH_FLOOR":
            flr = params.get("floor", 1)
            self.physics.dispatch_to_floor(flr)
            print(f"[Virtual ESP32] Received DISPATCH to Floor {flr}")
        elif cmd == "EMERGENCY_STOP":
            self.physics.trigger_emergency_stop()
            print("[Virtual ESP32] Emergency Brake Engaged!")
        elif cmd == "RESET_ALARM":
            self.physics.reset_emergency_stop()
            print("[Virtual ESP32] Emergency Brake Released.")
        elif cmd == "INJECT_DEFECT":
            dtype = params.get("defect_type", "broken_wire")
            sev = params.get("severity", 1.0)
            self.physics.inject_defect(dtype, sev)
            print(f"[Virtual ESP32] Defect Injected: {dtype} (severity={sev})")

    def _telemetry_loop(self):
        """Continuous sensor read and broadcast loop."""
        dt = 1.0 / self.rate_hz
        while self.running:
            t0 = time.time()
            
            # Step elevator physics
            self.physics.step(dt=dt)
            
            # Simulate optical tachometer pulses as car travels
            if self.physics.state.is_moving:
                self.optical_pulse_counter += int(abs(self.physics.state.velocity_mps) * 5)

            # Build telemetry packet according to ESP32 protocol
            state = self.physics.state
            packet_str = ElevatorGuardESP32Protocol.encode_telemetry_packet(
                timestamp_ms=int(time.time() * 1000),
                floor=state.floor,
                velocity_mps=state.velocity_mps,
                accel_x=state.accel_x,
                accel_y=state.accel_y,
                accel_z=state.accel_z,
                tensions=state.rope_tensions_n,
                door_open=(state.door_open_pct > 0.5),
                optical_rev_count=self.optical_pulse_counter,
                flags=1 if state.emergency_stop else 0
            )

            # Route to sink (ingestor queue or callback)
            if self.packet_sink:
                try:
                    self.packet_sink(packet_str)
                except Exception as e:
                    print(f"[Virtual ESP32] Sink write error: {e}")

            elapsed = time.time() - t0
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)


if __name__ == "__main__":
    def print_sink(pkt: str):
        print("OUT:", pkt.strip())

    sim = VirtualESP32Device(broadcast_rate_hz=10, packet_sink=print_sink)
    sim.start()
    sim.physics.dispatch_to_floor(5)
    time.sleep(2.0)
    sim.stop()
