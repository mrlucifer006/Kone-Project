"""
Unit Tests for ESP32 Protocol Framing and Checksum Verification
"""

import pytest
from src.edge.esp32_protocol import ElevatorGuardESP32Protocol, PacketType


def test_esp32_protocol_encoding_and_decoding():
    # Encode
    frame = ElevatorGuardESP32Protocol.encode_telemetry_packet(
        timestamp_ms=12345678,
        floor=4,
        velocity_mps=1.45,
        accel_x=0.05,
        accel_y=-0.02,
        accel_z=9.82,
        tensions=[2490.0, 2510.0, 2505.0, 2495.0],
        door_open=False,
        flags=0
    )

    assert frame.startswith("$EG,")
    assert "*" in frame

    # Decode
    is_valid, data, err = ElevatorGuardESP32Protocol.decode_packet(frame)
    assert is_valid is True
    assert err == ""
    assert data["type"] == PacketType.TELEMETRY.value
    assert data["flr"] == 4
    assert data["vel"] == 1.45
    assert len(data["acc"]) == 3
    assert len(data["tns"]) == 4


def test_esp32_checksum_failure_detection():
    # Corrupt packet
    frame = "$EG,{\"type\":\"TELEMETRY\",\"flr\":1}*FF\r\n"
    is_valid, data, err = ElevatorGuardESP32Protocol.decode_packet(frame)
    assert is_valid is False
    assert "Checksum mismatch" in err


def test_esp32_command_encoding():
    cmd_frame = ElevatorGuardESP32Protocol.encode_command_packet(
        cmd="DISPATCH_FLOOR",
        params={"floor": 7}
    )
    is_valid, data, err = ElevatorGuardESP32Protocol.decode_packet(cmd_frame)
    assert is_valid is True
    assert data["type"] == PacketType.COMMAND.value
    assert data["cmd"] == "DISPATCH_FLOOR"
    assert data["params"]["floor"] == 7
