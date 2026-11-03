"""
ElevatorGuard-AI: Vibration Sensor Agent (Stream 2)
Dedicated autonomous agent managing the IMU Accelerometer and Load Cell Telemetry stream:
Ingests vibration packets ($EG protocol v2.1 or raw sensor tuples), computes time-frequency features
(RMS, FFT spectral harmonics, Pearson kurtosis, rope tension imbalance), evaluates dynamic health index (HI_dyn),
and publishes vibration telemetry onto the message bus.
"""

import time
import queue
import threading
from typing import Optional, Dict, Any, List, Union
from collections import deque
import numpy as np

from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.message_bus import MessageBus
from src.edge.esp32_protocol import ElevatorGuardESP32Protocol, PacketType
from src.telemetry.feature_extraction import TelemetryFeatureExtractor


class VibrationSensorAgent(BaseAgent):
    """
    Stream 2 Ingestion & Processing Agent.
    Continuously monitors tri-axial vibration, tension load cells, and hoistway motion.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        name: str = "VibrationSensorAgent",
        feature_extractor: Optional[TelemetryFeatureExtractor] = None,
        window_size: int = 64,
        max_packet_queue: int = 500,
        sample_timeout_sec: float = 3.0
    ):
        super().__init__(name=name, message_bus=message_bus, tick_interval=0.01)
        self.feature_extractor = feature_extractor or TelemetryFeatureExtractor()
        self.window_size = window_size
        self.sample_timeout_sec = sample_timeout_sec
        self.packet_queue: queue.Queue = queue.Queue(maxsize=max_packet_queue)

        # Ring buffers for time-series feature extraction
        self.accel_x_buf = deque(maxlen=window_size)
        self.accel_y_buf = deque(maxlen=window_size)
        self.accel_z_buf = deque(maxlen=window_size)
        self.tensions_buf = deque(maxlen=window_size)

        # Operational metrics
        self.total_samples_ingested = 0
        self.total_samples_processed = 0
        self.dropped_samples = 0
        self.last_sample_timestamp: Optional[float] = None
        self.sample_rate_hz = 0.0
        self._hz_window = []

        # Latest kinematics & sensor states
        self.latest_floor: int = 1
        self.latest_velocity_mps: float = 0.0
        self.latest_accel: List[float] = [0.0, 0.0, 9.81]
        self.latest_tensions: List[float] = [2500.0, 2500.0, 2500.0, 2500.0]
        self.latest_door_open: bool = False
        self.latest_emergency_stop: bool = False

        # Latest computed telemetry features & HI_dyn
        self.latest_features: Dict[str, Any] = {}
        self.latest_hi_dyn: float = 1.0
        self.tension_imbalance_alarm: bool = False
        self.vibration_alarm: bool = False

        self._state_lock = threading.RLock()

    def ingest_packet(self, packet: Union[str, Dict[str, Any]], timestamp: Optional[float] = None) -> bool:
        """
        Ingest an ESP32 protocol string ($EG,...*XX) or parsed dictionary.
        """
        ts = timestamp or time.time()
        parsed_data = None

        if isinstance(packet, str):
            is_valid, data, err = ElevatorGuardESP32Protocol.decode_packet(packet)
            if not is_valid or not data:
                return False
            parsed_data = data
        elif isinstance(packet, dict):
            parsed_data = packet
        else:
            return False

        # Handle only TELEMETRY packets in this stream
        if parsed_data.get("type", PacketType.TELEMETRY.value) != PacketType.TELEMETRY.value:
            return False

        with self._state_lock:
            self.total_samples_ingested += 1
            self.last_sample_timestamp = ts

        try:
            self.packet_queue.put_nowait((parsed_data, ts))
            return True
        except queue.Full:
            try:
                _ = self.packet_queue.get_nowait()
                self.packet_queue.put_nowait((parsed_data, ts))
                with self._state_lock:
                    self.dropped_samples += 1
                return True
            except (queue.Empty, queue.Full):
                with self._state_lock:
                    self.dropped_samples += 1
                return False

    def ingest_sensor_sample(
        self,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        tensions: List[float],
        floor: int = 1,
        velocity_mps: float = 0.0,
        door_open: bool = False,
        emergency_stop: bool = False,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Direct numerical ingestion helper.
        """
        data = {
            "type": PacketType.TELEMETRY.value,
            "ts": int((timestamp or time.time()) * 1000),
            "flr": floor,
            "vel": float(velocity_mps),
            "acc": [float(accel_x), float(accel_y), float(accel_z)],
            "tns": [float(t) for t in tensions],
            "dr": 1 if door_open else 0,
            "flg": 1 if emergency_stop else 0
        }
        return self.ingest_packet(data, timestamp=timestamp)

    def on_tick(self):
        """Process buffered packets and compute dynamic features."""
        # Process up to 10 packets per tick to maintain real-time responsiveness
        processed_any = False
        for _ in range(10):
            try:
                pkt_data, ts = self.packet_queue.get_nowait()
            except queue.Empty:
                break

            processed_any = True
            acc = pkt_data.get("acc", [0.0, 0.0, 9.81])
            tns = pkt_data.get("tns", [2500.0, 2500.0, 2500.0, 2500.0])
            flr = pkt_data.get("flr", 1)
            vel = pkt_data.get("vel", 0.0)
            dr = bool(pkt_data.get("dr", 0))
            flg = int(pkt_data.get("flg", 0))
            e_stop = bool(flg & 0x01)

            with self._state_lock:
                self.total_samples_processed += 1
                self.accel_x_buf.append(acc[0])
                self.accel_y_buf.append(acc[1])
                self.accel_z_buf.append(acc[2])
                self.tensions_buf.append(tns)

                self.latest_floor = flr
                self.latest_velocity_mps = vel
                self.latest_accel = acc
                self.latest_tensions = tns
                self.latest_door_open = dr
                self.latest_emergency_stop = e_stop

                # Rate tracking (Hz)
                now = time.time()
                self._hz_window.append(now)
                cutoff = now - 2.0
                self._hz_window = [t for t in self._hz_window if t > cutoff]
                if len(self._hz_window) > 1:
                    dur = self._hz_window[-1] - self._hz_window[0]
                    self.sample_rate_hz = round(len(self._hz_window) / max(0.1, dur), 1)

        if not processed_any or len(self.accel_z_buf) < 4:
            return

        # Perform feature extraction on current buffer
        with self._state_lock:
            ax = np.array(self.accel_x_buf, dtype=np.float32)
            ay = np.array(self.accel_y_buf, dtype=np.float32)
            az = np.array(self.accel_z_buf, dtype=np.float32)
            latest_tns = np.array(self.latest_tensions, dtype=np.float32)

        feats = self.feature_extractor.extract_features(
            accel_x=ax,
            accel_y=ay,
            accel_z=az,
            tensions=latest_tns
        )

        # Dynamic health index HI_dyn calculation
        # Nominal vertical vibration RMS ~0.08 m/s^2 (ISO 18738 ride quality limit is 0.45 m/s^2)
        vibe_pen = max(0.0, (feats["acc_rms_z"] - 0.20) / 0.40)
        imbalance_pen = max(0.0, (feats["tension_imbalance_ratio"] / 0.25))
        hi_dyn = max(0.05, min(1.0, 1.0 - (0.50 * vibe_pen + 0.50 * imbalance_pen)))

        with self._state_lock:
            self.latest_features = feats
            self.latest_hi_dyn = round(float(hi_dyn), 4)
            self.tension_imbalance_alarm = feats["tension_imbalance_alarm"]
            self.vibration_alarm = feats["vibration_alarm"]

        # Publish telemetry.vibration message
        payload = {
            "stream": "vibration",
            "timestamp": time.time(),
            "hi_dyn": self.latest_hi_dyn,
            "sample_rate_hz": self.sample_rate_hz,
            "floor": self.latest_floor,
            "velocity_mps": self.latest_velocity_mps,
            "accel": self.latest_accel,
            "tensions": self.latest_tensions,
            "acc_rms_z": round(feats["acc_rms_z"], 3),
            "acc_kurtosis_z": round(feats["acc_kurtosis_z"], 2),
            "dominant_frequency_hz": round(feats["dominant_frequency_hz"], 1),
            "tension_imbalance_ratio": round(feats["tension_imbalance_ratio"], 3),
            "alarms": {
                "tension_imbalance": self.tension_imbalance_alarm,
                "vibration_exceeded": self.vibration_alarm,
                "emergency_stop": self.latest_emergency_stop
            }
        }
        self.publish("telemetry.vibration", payload)

    @property
    def is_stream_active(self) -> bool:
        """Return True if packets have been received within the timeout window."""
        with self._state_lock:
            if self.last_sample_timestamp is None:
                return False
            return (time.time() - self.last_sample_timestamp) < self.sample_timeout_sec

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Return the latest dynamic features and kinematics."""
        with self._state_lock:
            return {
                "hi_dyn": self.latest_hi_dyn,
                "sample_rate_hz": self.sample_rate_hz,
                "floor": self.latest_floor,
                "velocity_mps": self.latest_velocity_mps,
                "accel": self.latest_accel,
                "tensions": self.latest_tensions,
                "features": self.latest_features,
                "alarms": {
                    "tension_imbalance": self.tension_imbalance_alarm,
                    "vibration_exceeded": self.vibration_alarm,
                    "emergency_stop": self.latest_emergency_stop
                },
                "stream_active": self.is_stream_active
            }

    def get_status(self) -> Dict[str, Any]:
        """Extended status including Stream 2 telemetry statistics."""
        base_status = super().get_status()
        with self._state_lock:
            base_status.update({
                "stream_active": self.is_stream_active,
                "sample_rate_hz": self.sample_rate_hz,
                "total_samples_ingested": self.total_samples_ingested,
                "total_samples_processed": self.total_samples_processed,
                "dropped_samples": self.dropped_samples,
                "queue_backlog": self.packet_queue.qsize(),
                "latest_hi_dyn": self.latest_hi_dyn,
                "tension_imbalance_alarm": self.tension_imbalance_alarm,
                "vibration_alarm": self.vibration_alarm,
                "latest_floor": self.latest_floor
            })
        return base_status
