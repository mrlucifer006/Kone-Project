"""
ElevatorGuard-AI: Multi-Agent Pipeline Coordinator
Central orchestrator managing all autonomous agents in the multi-modal edge pipeline:
- CameraVisionAgent (Stream 1: Hoistway NDT Visual Cable Inspection)
- VibrationSensorAgent (Stream 2: MPU-6050 & HX711 Telemetry & Dynamics)
- FusionSupervisorAgent (Stream 3: Multi-Modal Decision Fusion & RUL Engine)
- SafetyInterlockAgent (Safety Enforcement, E-Stop Interlock & Incident Auditing)
- MessageBus (High-Performance Decoupled Pub/Sub Broker)
"""

import time
import threading
from typing import Optional, Dict, Any, List, Union, Callable
import numpy as np

from src.agents.message_bus import MessageBus
from src.agents.camera_agent import CameraVisionAgent
from src.agents.vibration_agent import VibrationSensorAgent
from src.agents.fusion_agent import FusionSupervisorAgent
from src.agents.safety_agent import SafetyInterlockAgent


class MultiAgentPipelineCoordinator:
    """
    Unified manager and ingress gateway for the ElevatorGuard-AI multi-agent pipeline.
    Ensures synchronous/asynchronous ingestion of both Camera and Vibration sensor streams.
    """

    def __init__(
        self,
        emergency_stop_callback: Optional[Callable[[], None]] = None,
        reset_callback: Optional[Callable[[], None]] = None
    ):
        self.message_bus = MessageBus()

        # Initialize Sub-Agents
        self.camera_agent = CameraVisionAgent(
            message_bus=self.message_bus,
            name="CameraVisionAgent"
        )
        self.vibration_agent = VibrationSensorAgent(
            message_bus=self.message_bus,
            name="VibrationSensorAgent"
        )
        self.fusion_agent = FusionSupervisorAgent(
            message_bus=self.message_bus,
            name="FusionSupervisorAgent"
        )
        self.safety_agent = SafetyInterlockAgent(
            message_bus=self.message_bus,
            name="SafetyInterlockAgent",
            emergency_stop_callback=emergency_stop_callback,
            reset_callback=reset_callback
        )

        self.agents = [
            self.camera_agent,
            self.vibration_agent,
            self.fusion_agent,
            self.safety_agent
        ]
        self._is_running = False
        self._lock = threading.RLock()

    def start(self):
        """Start all pipeline agents and message routing."""
        with self._lock:
            if self._is_running:
                return
            for agent in self.agents:
                agent.start()
            self._is_running = True
            print("[MultiAgentCoordinator] All 4 autonomous agents successfully started.")

    def stop(self, timeout: float = 2.0):
        """Cleanly terminate all pipeline agents."""
        with self._lock:
            if not self._is_running:
                return
            for agent in reversed(self.agents):
                agent.stop(timeout=timeout)
            self._is_running = False
            print("[MultiAgentCoordinator] Pipeline agents cleanly stopped.")

    # ---------------- DUAL-STREAM INGESTION GATEWAYS ----------------

    def ingest_camera_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Stream 1 Ingestion Gateway:
        Feed a raw camera frame (from RTSP, webcam, video, or synthetic hoistway feed)
        directly into the CameraVisionAgent.
        """
        return self.camera_agent.ingest_frame(
            frame=frame,
            timestamp=timestamp,
            metadata=metadata
        )

    def ingest_vibration_packet(
        self,
        packet: Union[str, Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Stream 2 Ingestion Gateway:
        Feed an ESP32 protocol string ($EG,...*XX) or sensor telemetry dictionary
        directly into the VibrationSensorAgent.
        """
        return self.vibration_agent.ingest_packet(
            packet=packet,
            timestamp=timestamp
        )

    def ingest_vibration_sample(
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
        Stream 2 Direct Sensor Tuple Gateway.
        """
        return self.vibration_agent.ingest_sensor_sample(
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            tensions=tensions,
            floor=floor,
            velocity_mps=velocity_mps,
            door_open=door_open,
            emergency_stop=emergency_stop,
            timestamp=timestamp
        )

    # ---------------- STREAM VERIFICATION & HEALTH ----------------

    def verify_streams_active(self) -> Dict[str, Any]:
        """
        Verify that the pipeline is receiving both the camera and vibration streams.
        Returns status and metrics for both streams.
        """
        camera_active = self.camera_agent.is_stream_active
        vibration_active = self.vibration_agent.is_stream_active

        if camera_active and vibration_active:
            status_code = "BOTH_STREAMS_ACTIVE"
            message = "Pipeline is concurrently receiving and processing both Camera and Vibration streams."
        elif camera_active and not vibration_active:
            status_code = "CAMERA_ONLY_ACTIVE"
            message = "Warning: Camera stream is active, but Vibration stream is currently idle or offline."
        elif not camera_active and vibration_active:
            status_code = "VIBRATION_ONLY_ACTIVE"
            message = "Warning: Vibration stream is active, but Camera stream is currently idle or offline."
        else:
            status_code = "NO_STREAMS_ACTIVE"
            message = "Notice: Both Camera and Vibration streams are currently idle."

        return {
            "status": status_code,
            "both_streams_active": (camera_active and vibration_active),
            "camera_stream": {
                "active": camera_active,
                "fps": self.camera_agent.fps,
                "total_frames_processed": self.camera_agent.total_frames_processed,
                "avg_inference_ms": self.camera_agent.avg_inference_ms,
                "last_frame_age_sec": round(time.time() - self.camera_agent.last_frame_timestamp, 2)
                if self.camera_agent.last_frame_timestamp else None
            },
            "vibration_stream": {
                "active": vibration_active,
                "sample_rate_hz": self.vibration_agent.sample_rate_hz,
                "total_samples_processed": self.vibration_agent.total_samples_processed,
                "last_sample_age_sec": round(time.time() - self.vibration_agent.last_sample_timestamp, 2)
                if self.vibration_agent.last_sample_timestamp else None
            },
            "message": message
        }

    # ---------------- CONSOLIDATED SYSTEM STATE ----------------

    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Return comprehensive real-time state of the entire multi-agent pipeline.
        """
        stream_verification = self.verify_streams_active()
        cam_insp = self.camera_agent.get_latest_inspection()
        vib_tel = self.vibration_agent.get_latest_telemetry()
        fusion_rep = self.fusion_agent.get_latest_fusion()
        safety_st = self.safety_agent.get_safety_state()

        return {
            "timestamp": round(time.time(), 4),
            "is_running": self._is_running,
            "streams": stream_verification,
            "stream_1_vision": {
                "hi_vis": cam_insp.get("hi_vis"),
                "wire_breaks": cam_insp.get("wire_breaks"),
                "wear_pct": cam_insp.get("wear_pct"),
                "corrosion_pct": cam_insp.get("corrosion_pct"),
                "is_discard_required": cam_insp.get("is_discard_required"),
                "status": cam_insp.get("status"),
                "detections_count": cam_insp.get("detections_count"),
                "fps": cam_insp.get("fps")
            },
            "stream_2_vibration": {
                "hi_dyn": vib_tel.get("hi_dyn"),
                "sample_rate_hz": vib_tel.get("sample_rate_hz"),
                "floor": vib_tel.get("floor"),
                "velocity_mps": vib_tel.get("velocity_mps"),
                "accel": vib_tel.get("accel"),
                "tensions": vib_tel.get("tensions"),
                "alarms": vib_tel.get("alarms")
            },
            "stream_3_fusion": {
                "hi_composite": fusion_rep.get("hi_composite"),
                "status": fusion_rep.get("status"),
                "summary": fusion_rep.get("summary"),
                "rul_cycles": fusion_rep.get("rul_cycles"),
                "rul_days": fusion_rep.get("rul_days"),
                "safety_override_active": fusion_rep.get("safety_override_active"),
                "action_plan": fusion_rep.get("action_plan")
            },
            "safety": {
                "emergency_stop_active": safety_st.get("emergency_stop_active"),
                "active_alarms": safety_st.get("active_alarms"),
                "total_critical_events": safety_st.get("total_critical_events")
            },
            "agents": {
                agent.name: agent.get_status() for agent in self.agents
            },
            "message_bus": self.message_bus.get_metrics()
        }

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Retrieve latest camera frame annotated with YOLO defect boxes."""
        return self.camera_agent.get_latest_annotated_frame()

    def trigger_emergency_stop(self, reason: str = "MANUAL_OVERRIDE", details: str = "Operator invoked stop."):
        """Trigger emergency stop via SafetyInterlockAgent."""
        self.safety_agent.trigger_emergency_stop(reason, details)

    def reset_alarms(self):
        """Reset emergency stop and safety interlocks."""
        self.safety_agent.reset_alarms()
