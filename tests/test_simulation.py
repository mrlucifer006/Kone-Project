"""
Unit Tests for Elevator Physics Simulator & Virtual ESP32 Node
"""

import time
import pytest
from src.simulation.elevator_physics import ElevatorPhysicsSimulator
from src.simulation.esp32_simulator import VirtualESP32Device


def test_elevator_dispatch_and_kinematics():
    sim = ElevatorPhysicsSimulator(num_floors=8)
    assert sim.state.floor == 1
    assert sim.state.position_m == 0.0

    # Dispatch to floor 4
    dispatched = sim.dispatch_to_floor(4)
    assert dispatched is True
    assert sim.state.target_floor == 4
    assert sim.state.is_moving is True

    # Advance time steps
    for _ in range(100):
        sim.step(dt=0.05)

    # Car should have moved upwards
    assert sim.state.position_m > 0.0
    assert sim.state.floor >= 1


def test_emergency_stop():
    sim = ElevatorPhysicsSimulator(num_floors=8)
    sim.dispatch_to_floor(6)
    sim.step(dt=0.1)

    # Trigger emergency stop
    sim.trigger_emergency_stop()
    assert sim.state.emergency_stop is True
    assert sim.state.velocity_mps == 0.0

    # Cannot dispatch while E-stop active
    assert sim.dispatch_to_floor(2) is False

    # Reset
    sim.reset_emergency_stop()
    assert sim.state.emergency_stop is False


def test_defect_injection():
    sim = ElevatorPhysicsSimulator()
    initial_breaks = sim.state.wire_breaks
    sim.inject_defect("broken_wire", 2)
    sim.step(0.01)
    assert sim.state.wire_breaks == initial_breaks + 2


def test_virtual_esp32_device():
    packets_received = []

    def sink(pkt: str):
        packets_received.append(pkt)

    esp32 = VirtualESP32Device(broadcast_rate_hz=20, packet_sink=sink)
    esp32.start()
    time.sleep(0.2)
    esp32.stop()

    assert len(packets_received) > 0
    assert "$EG," in packets_received[0]
