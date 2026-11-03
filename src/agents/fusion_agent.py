"""
ElevatorGuard-AI: Fusion Supervisor Agent (Stream 3)
Dedicated autonomous supervisor agent performing multi-modal decision fusion:
Subscribes to Camera Vision telemetry (Stream 1) and Vibration/Sensor telemetry (Stream 2),
aligns asynchronous streams, computes composite health index (HI_comp), enforces ISO 4309 safety
overrides, and projects Remaining Useful Life (RUL) with Miner's cumulative damage rule.
"""

import time
import threading
from typing import Optional, Dict, Any, List

from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.message_bus import MessageBus, AgentMessage
from src.vision.visual_hi import VisualInspectionResult
from src.fusion.multimodal_fusion import MultiModalFusionEngine, FusionResult
from src.fusion.rul_estimator import RULEstimator, RULPrediction


class FusionSupervisorAgent(BaseAgent):
    """
    Stream 3 Multi-Modal Decision & Prognostics Agent.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        name: str = "FusionSupervisorAgent",
        fusion_engine: Optional[MultiModalFusionEngine] = None,
        rul_estimator: Optional[RULEstimator] = None,
        sync_timeout_sec: float = 5.0
    ):
        super().__init__(name=name, message_bus=message_bus, tick_interval=0.05)
        self.fusion_engine = fusion_engine or MultiModalFusionEngine(alpha_vision_weight=0.55, beta_telemetry_weight=0.45)
        self.rul_estimator = rul_estimator or RULEstimator(hi_discard_limit=0.20)
        self.sync_timeout_sec = sync_timeout_sec

        # Stream alignment cache
        self.latest_vision_msg: Optional[AgentMessage] = None
        self.latest_vibration_msg: Optional[AgentMessage] = None
        self.latest_fusion_result: Optional[FusionResult] = None
        self.latest_rul_prediction: Optional[RULPrediction] = None

        self.last_fusion_time: Optional[float] = None
        self.total_fusions_performed = 0
        self.total_trips_tracked = 42500
        self.safety_overrides_triggered = 0

        self._state_lock = threading.RLock()

    def on_start(self):
        """Subscribe to vision and vibration telemetry streams."""
        self.subscribe("telemetry.vision")
        self.subscribe("telemetry.vibration")

    def process_message(self, msg: AgentMessage):
        """Cache incoming telemetry messages."""
        with self._state_lock:
            if msg.topic == "telemetry.vision":
                self.latest_vision_msg = msg
            elif msg.topic == "telemetry.vibration":
                self.latest_vibration_msg = msg

    def on_tick(self):
        """Execute decision fusion when fresh data is available."""
        with self._state_lock:
            vis_msg = self.latest_vision_msg
            vib_msg = self.latest_vibration_msg

        if vis_msg is None and vib_msg is None:
            return

        now = time.time()
        # Extract visual inspection data (fallback to nominal if vision stream delayed)
        if vis_msg is not None:
            vis_payload = vis_msg.payload
            vis_hi = vis_payload.get("hi_vis", 1.0)
            wire_breaks = vis_payload.get("wire_breaks", 0)
            wear_pct = vis_payload.get("wear_pct", 0.0)
            corr_pct = vis_payload.get("corrosion_pct", 0.0)
            is_discard = vis_payload.get("is_discard_required", False)
            vis_confidence = 1.0 if (now - vis_msg.timestamp) < self.sync_timeout_sec else 0.4
        else:
            vis_hi = 1.0
            wire_breaks = 0
            wear_pct = 0.0
            corr_pct = 0.0
            is_discard = False
            vis_confidence = 0.2

        # Extract dynamic vibration data (fallback to nominal if vibration stream delayed)
        if vib_msg is not None:
            vib_payload = vib_msg.payload
            hi_telemetry = vib_payload.get("hi_dyn", 1.0)
            t_confidence = 1.0 if (now - vib_msg.timestamp) < self.sync_timeout_sec else 0.4
            current_floor = vib_payload.get("floor", 1)
        else:
            hi_telemetry = 1.0
            t_confidence = 0.2
            current_floor = 1

        # Build VisualInspectionResult object
        vis_result = VisualInspectionResult(
            hi_vis=vis_hi,
            wire_breaks=wire_breaks,
            wear_pct=wear_pct,
            corrosion_pct=corr_pct,
            is_discard_required=is_discard,
            status="CRITICAL_DISCARD" if is_discard else ("MONITOR_FREQUENTLY" if vis_hi < 0.7 else "NOMINAL_HEALTHY"),
            action_recommendation="Safety shutdown mandatory." if is_discard else "Normal operating condition."
        )

        fusion_res = self.fusion_engine.fuse(
            visual_result=vis_result,
            hi_telemetry=hi_telemetry,
            vision_confidence=vis_confidence,
            telemetry_confidence=t_confidence
        )

        rul_res = self.rul_estimator.estimate_rul(
            current_hi=fusion_res.hi_composite,
            current_trips=self.total_trips_tracked
        )

        with self._state_lock:
            self.latest_fusion_result = fusion_res
            self.latest_rul_prediction = rul_res
            self.last_fusion_time = now
            self.total_fusions_performed += 1
            if fusion_res.safety_override_active:
                self.safety_overrides_triggered += 1

        # Publish composite health report
        payload = {
            "timestamp": now,
            "hi_composite": fusion_res.hi_composite,
            "hi_vision": fusion_res.hi_vision,
            "hi_telemetry": fusion_res.hi_telemetry,
            "alpha_vision": fusion_res.alpha_vision,
            "beta_telemetry": fusion_res.beta_telemetry,
            "safety_override_active": fusion_res.safety_override_active,
            "status": fusion_res.status,
            "summary_message": fusion_res.summary_message,
            "rul_cycles": rul_res.rul_cycles,
            "rul_days": rul_res.rul_days,
            "predicted_failure_trip": rul_res.predicted_failure_trip,
            "action_plan": rul_res.action_plan,
            "streams_aligned": {
                "vision_age_sec": round(now - vis_msg.timestamp, 2) if vis_msg else None,
                "vibration_age_sec": round(now - vib_msg.timestamp, 2) if vib_msg else None
            }
        }
        self.publish("telemetry.fusion", payload)

    def get_latest_fusion(self) -> Dict[str, Any]:
        """Return the latest fusion report and prognostics."""
        with self._state_lock:
            if self.latest_fusion_result is None or self.latest_rul_prediction is None:
                return {
                    "hi_composite": 1.0,
                    "hi_vision": 1.0,
                    "hi_telemetry": 1.0,
                    "status": "NOMINAL_HEALTHY",
                    "rul_cycles": 150000,
                    "rul_days": 750.0,
                    "safety_override_active": False
                }
            f = self.latest_fusion_result
            r = self.latest_rul_prediction
            return {
                "hi_composite": f.hi_composite,
                "hi_vision": f.hi_vision,
                "hi_telemetry": f.hi_telemetry,
                "alpha_vision": f.alpha_vision,
                "beta_telemetry": f.beta_telemetry,
                "safety_override_active": f.safety_override_active,
                "status": f.status,
                "summary": f.summary_message,
                "rul_cycles": r.rul_cycles,
                "rul_days": r.rul_days,
                "predicted_failure_trip": r.predicted_failure_trip,
                "action_plan": r.action_plan,
                "trajectory_trips": r.trajectory_trips,
                "trajectory_hi": r.trajectory_hi
            }

    def get_status(self) -> Dict[str, Any]:
        """Extended status including Stream 3 fusion statistics."""
        base_status = super().get_status()
        with self._state_lock:
            latest = self.get_latest_fusion()
            base_status.update({
                "total_fusions_performed": self.total_fusions_performed,
                "safety_overrides_triggered": self.safety_overrides_triggered,
                "hi_composite": latest.get("hi_composite"),
                "status": latest.get("status"),
                "rul_cycles": latest.get("rul_cycles"),
                "rul_days": latest.get("rul_days"),
                "safety_override_active": latest.get("safety_override_active")
            })
        return base_status
