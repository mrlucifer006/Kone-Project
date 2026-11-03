"""
ElevatorGuard-AI: ESP32 Hardware Communication Protocol
Defines message structures, framing, CRC verification, and command interfaces
for ESP32 edge microcontroller integration (UART Serial, WebSocket, MQTT).
"""

import json
import struct
import time
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class PacketType(str, Enum):
    TELEMETRY = "TELEMETRY"
    COMMAND = "COMMAND"
    ACK = "ACK"
    ALARM = "ALARM"
    HEARTBEAT = "HEARTBEAT"


class ElevatorGuardESP32Protocol:
    """
    Protocol parser and encoder for ESP32 elevator monitoring node.
    Supports JSON framing with checksum verification.
    """

    PROTOCOL_VERSION = "2.1"
    START_MARKER = "$EG"
    END_MARKER = "\r\n"

    @staticmethod
    def calculate_checksum(payload: str) -> str:
        """Calculate XOR checksum over string payload (NMEA standard)."""
        xor_sum = 0
        for char in payload:
            xor_sum ^= ord(char)
        return f"{xor_sum:02X}"

    @classmethod
    def encode_telemetry_packet(
        cls,
        timestamp_ms: int,
        floor: int,
        velocity_mps: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        tensions: list,
        door_open: bool = False,
        optical_rev_count: int = 0,
        flags: int = 0
    ) -> str:
        """
        Encode a sensor telemetry packet into JSON wire format.
        """
        data = {
            "type": PacketType.TELEMETRY.value,
            "ver": cls.PROTOCOL_VERSION,
            "ts": timestamp_ms,
            "flr": floor,
            "vel": round(velocity_mps, 3),
            "acc": [round(accel_x, 3), round(accel_y, 3), round(accel_z, 3)],
            "tns": [round(t, 1) for t in tensions],
            "dr": 1 if door_open else 0,
            "opt": optical_rev_count,
            "flg": flags
        }
        json_str = json.dumps(data, separators=(",", ":"))
        chk = cls.calculate_checksum(json_str)
        return f"{cls.START_MARKER},{json_str}*{chk}{cls.END_MARKER}"

    @classmethod
    def encode_command_packet(
        cls,
        cmd: str,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Encode a command packet sent from edge computer to ESP32.
        Commands: DISPATCH_FLOOR, EMERGENCY_STOP, TARE_SENSORS, SET_RGB_COLOR, BEEP_ALARM
        """
        data = {
            "type": PacketType.COMMAND.value,
            "cmd": cmd,
            "params": params or {},
            "ts": int(time.time() * 1000)
        }
        json_str = json.dumps(data, separators=(",", ":"))
        chk = cls.calculate_checksum(json_str)
        return f"{cls.START_MARKER},{json_str}*{chk}{cls.END_MARKER}"

    @classmethod
    def decode_packet(cls, raw_line: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Decode and validate an incoming frame from ESP32.
        
        Returns:
            (is_valid, decoded_data_dict, error_message)
        """
        raw = raw_line.strip()
        if not raw.startswith(cls.START_MARKER):
            # Try raw json fallback
            try:
                data = json.loads(raw)
                return True, data, ""
            except Exception:
                return False, None, "Invalid packet header"

        if "*" not in raw:
            return False, None, "Missing packet checksum delimiter"

        try:
            # Extract payload and checksum
            payload_part = raw[len(cls.START_MARKER) + 1: raw.rfind("*")]
            chk_part = raw[raw.rfind("*") + 1:]

            # Verify checksum
            expected_chk = cls.calculate_checksum(payload_part)
            if chk_part.upper() != expected_chk:
                return False, None, f"Checksum mismatch: received {chk_part}, expected {expected_chk}"

            data = json.loads(payload_part)
            return True, data, ""
        except Exception as e:
            return False, None, f"Malformed packet: {e}"
