"""
ElevatorGuard-AI: Multi-Modal Fusion Engine
Fuses Computer Vision NDT metrics (Stream 1) with Dynamic Telemetry Prognostics (Stream 2)
to calculate composite health index HI_comp and safety boundaries.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.vision.visual_hi import VisualInspectionResult


@dataclass
class FusionResult:
    hi_composite: float
    hi_vision: float
    hi_telemetry: float
    alpha_vision: float
    beta_telemetry: float
    safety_override_active: bool
    status: str
    summary_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hi_composite": round(float(self.hi_composite), 4),
            "hi_vision": round(float(self.hi_vision), 4),
            "hi_telemetry": round(float(self.hi_telemetry), 4),
            "alpha_vision": round(float(self.alpha_vision), 3),
            "beta_telemetry": round(float(self.beta_telemetry), 3),
            "safety_override_active": self.safety_override_active,
            "status": self.status,
            "summary_message": self.summary_message
        }


class MultiModalFusionEngine:
    """
    Ensemble decision fusion for elevator suspension health index estimation.
    """

    def __init__(
        self,
        alpha_vision_weight: float = 0.55,
        beta_telemetry_weight: float = 0.45,
        hi_discard_limit: float = 0.20
    ):
        total = alpha_vision_weight + beta_telemetry_weight
        self.default_alpha = alpha_vision_weight / total
        self.default_beta = beta_telemetry_weight / total
        self.hi_discard_limit = hi_discard_limit

    def fuse(
        self,
        visual_result: VisualInspectionResult,
        hi_telemetry: float,
        vision_confidence: float = 1.0,
        telemetry_confidence: float = 1.0
    ) -> FusionResult:
        """
        Compute weighted composite health index HI_comp.
        
        Args:
            visual_result: VisualInspectionResult from Stream 1
            hi_telemetry: Dynamic health index HI_dyn from Stream 2
            vision_confidence: Confidence score of optical stream [0.0, 1.0]
            telemetry_confidence: Confidence score of telemetry stream [0.0, 1.0]
        """
        hi_vis = float(visual_result.hi_vis)
        hi_dyn = float(max(0.0, min(1.0, hi_telemetry)))

        # Dynamic confidence weighting
        w_vis = self.default_alpha * max(0.1, vision_confidence)
        w_dyn = self.default_beta * max(0.1, telemetry_confidence)
        w_sum = w_vis + w_dyn
        alpha = w_vis / w_sum
        beta = w_dyn / w_sum

        hi_comp = (alpha * hi_vis) + (beta * hi_dyn)
        safety_override = False

        # Hard Safety Enforcement: ISO 4309 Discard Trumps Telemetry
        if visual_result.is_discard_required:
            hi_comp = min(hi_comp, self.hi_discard_limit - 0.02)
            safety_override = True
            status = "CRITICAL_DISCARD"
            msg = (
                f"SAFETY SHUTDOWN: Visual inspection breached ISO 4309 limits "
                f"({visual_result.wire_breaks} wire breaks). Immediate cable replacement mandatory."
            )
        elif hi_comp <= self.hi_discard_limit:
            status = "CRITICAL_DISCARD"
            msg = "CRITICAL: Composite health has reached safety discard threshold. Replace suspension ropes."
        elif hi_comp < 0.45:
            status = "WARNING_REPLACEMENT_NEAR"
            msg = "WARNING: Cable wear approaching end-of-life limit. Schedule replacement."
        elif hi_comp < 0.70:
            status = "MONITOR_FREQUENTLY"
            msg = "MONITOR: Moderate cumulative mechanical degradation detected."
        else:
            status = "NOMINAL_HEALTHY"
            msg = "NOMINAL: Suspension ropes operating within safe design tolerances."

        return FusionResult(
            hi_composite=hi_comp,
            hi_vision=hi_vis,
            hi_telemetry=hi_dyn,
            alpha_vision=alpha,
            beta_telemetry=beta,
            safety_override_active=safety_override,
            status=status,
            summary_message=msg
        )
