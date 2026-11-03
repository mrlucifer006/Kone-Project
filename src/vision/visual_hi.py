"""
ElevatorGuard-AI: ISO 4309 Visual Health Index (HI_vis) Calculator
Quantifies suspension cable surface degradation based on ISO 4309 standards:
- Broken wire count in 6d length (N_break)
- Loss of diameter / wear percentage (W_pct)
- Corrosion severity percentage (C_sev)
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np


@dataclass
class VisualInspectionResult:
    hi_vis: float
    status: str
    wire_breaks: int
    wear_pct: float
    corrosion_pct: float
    is_discard_required: bool
    discard_reasons: list
    action_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hi_vis": round(float(self.hi_vis), 4),
            "status": self.status,
            "wire_breaks": int(self.wire_breaks),
            "wear_pct": round(float(self.wear_pct) * 100, 2),
            "corrosion_pct": round(float(self.corrosion_pct) * 100, 2),
            "is_discard_required": self.is_discard_required,
            "discard_reasons": self.discard_reasons,
            "action_recommendation": self.action_recommendation
        }


class ISO4309VisualAnalyzer:
    """
    Implements ISO 4309: Wire Ropes - Care and Maintenance, Inspection and Discard.
    Calculates normalized HI_vis in [0.0, 1.0].
    """

    def __init__(
        self,
        n_max_wire_breaks: int = 6,
        max_wear_pct: float = 0.10,
        max_corrosion_pct: float = 0.15,
        w_breaks: float = 0.50,
        w_wear: float = 0.30,
        w_corrosion: float = 0.20
    ):
        self.n_max_breaks = n_max_wire_breaks
        self.max_wear_pct = max_wear_pct
        self.max_corrosion_pct = max_corrosion_pct
        
        # Normalize weights to sum to 1.0
        total_w = w_breaks + w_wear + w_corrosion
        self.w1 = w_breaks / total_w
        self.w2 = w_wear / total_w
        self.w3 = w_corrosion / total_w

    def calculate_hi(
        self,
        wire_breaks: int,
        wear_pct: float,
        corrosion_pct: float
    ) -> VisualInspectionResult:
        """
        Calculate Visual Health Index and evaluate ISO 4309 safety discard conditions.
        
        Formula:
            HI_vis = 1.0 - (w1 * min(N_break, N_max)/N_max + w2 * (wear/max_wear) + w3 * (corrosion/max_corrosion))
        """
        wire_breaks = max(0, int(wire_breaks))
        wear_pct = max(0.0, float(wear_pct))
        corrosion_pct = max(0.0, float(corrosion_pct))

        # Normalized damage components (each clamped to [0, 1])
        norm_breaks = min(1.0, wire_breaks / float(self.n_max_breaks))
        norm_wear = min(1.0, wear_pct / self.max_wear_pct)
        norm_corrosion = min(1.0, corrosion_pct / self.max_corrosion_pct)

        penalty = (self.w1 * norm_breaks) + (self.w2 * norm_wear) + (self.w3 * norm_corrosion)
        hi_vis = max(0.0, min(1.0, 1.0 - penalty))

        # Discard Criteria Verification
        discard_reasons = []
        if wire_breaks >= self.n_max_breaks:
            discard_reasons.append(
                f"Broken wire limit breached: {wire_breaks} breaks detected (ISO 4309 limit: {self.n_max_breaks} in 6d)"
            )
        if wear_pct >= self.max_wear_pct:
            discard_reasons.append(
                f"Severe cable wear: {wear_pct * 100:.1f}% reduction in diameter (Limit: {self.max_wear_pct * 100:.1f}%)"
            )
        if corrosion_pct >= self.max_corrosion_pct:
            discard_reasons.append(
                f"Severe corrosion pitting: {corrosion_pct * 100:.1f}% surface coverage (Limit: {self.max_corrosion_pct * 100:.1f}%)"
            )

        is_discard_required = len(discard_reasons) > 0

        # Health Status Categorization
        if is_discard_required:
            status = "CRITICAL_DISCARD"
            recommendation = (
                "IMMEDIATE SHUTDOWN REQUIRED: Suspension rope has met ISO 4309 discard criteria. "
                "Ground elevator immediately and initiate rope replacement."
            )
            # Hard safety clamp
            hi_vis = min(hi_vis, 0.19)
        elif hi_vis < 0.45:
            status = "WARNING_REPLACEMENT_NEAR"
            recommendation = (
                "SCHEDULE REPLACEMENT: Degradation approaching discard threshold. "
                "Schedule suspension cable replacement within 14 operational days."
            )
        elif hi_vis < 0.70:
            status = "MONITOR_FREQUENTLY"
            recommendation = (
                "INCREASE INSPECTION FREQUENCY: Visible strand wear detected. "
                "Inspect every 7 days and lubricate cable run."
            )
        else:
            status = "NOMINAL_HEALTHY"
            recommendation = "Cable in nominal condition. Maintain routine bi-monthly maintenance schedule."

        return VisualInspectionResult(
            hi_vis=hi_vis,
            status=status,
            wire_breaks=wire_breaks,
            wear_pct=wear_pct,
            corrosion_pct=corrosion_pct,
            is_discard_required=is_discard_required,
            discard_reasons=discard_reasons,
            action_recommendation=recommendation
        )


# ==============================================================================
# ISO 7465 & ISO 8100-34: Elevator Guide Rail Visual Health Analyzer
# ==============================================================================

@dataclass
class GuideRailInspectionResult:
    hi_rail_vis: float
    status: str
    joint_misalignment_mm: float
    surface_scoring_pct: float
    rust_pct: float
    dry_contact_pct: float
    is_service_required: bool
    fault_reasons: list
    action_recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hi_rail_vis": round(float(self.hi_rail_vis), 4),
            "status": self.status,
            "joint_misalignment_mm": round(float(self.joint_misalignment_mm), 2),
            "surface_scoring_pct": round(float(self.surface_scoring_pct) * 100, 2),
            "rust_pct": round(float(self.rust_pct) * 100, 2),
            "dry_contact_pct": round(float(self.dry_contact_pct) * 100, 2),
            "is_service_required": self.is_service_required,
            "fault_reasons": self.fault_reasons,
            "action_recommendation": self.action_recommendation
        }


class ISO7465GuideRailVisualAnalyzer:
    """
    Implements ISO 7465 / ISO 8100-34: Passenger and Goods Lifts Guide Rails.
    Evaluates machined guide rail blade surface, fishplate joint alignment,
    surface scratches/scoring from guide shoes, and rust/oxidation.
    """

    def __init__(
        self,
        max_joint_step_mm: float = 0.20,
        max_scoring_pct: float = 0.15,
        max_rust_pct: float = 0.10,
        max_dry_contact_pct: float = 0.20,
        w_joint: float = 0.45,
        w_scoring: float = 0.25,
        w_rust: float = 0.20,
        w_dry: float = 0.10
    ):
        self.max_joint_step_mm = max_joint_step_mm
        self.max_scoring_pct = max_scoring_pct
        self.max_rust_pct = max_rust_pct
        self.max_dry_contact_pct = max_dry_contact_pct

        total_w = w_joint + w_scoring + w_rust + w_dry
        self.w_joint = w_joint / total_w
        self.w_scoring = w_scoring / total_w
        self.w_rust = w_rust / total_w
        self.w_dry = w_dry / total_w

    def calculate_rail_hi(
        self,
        joint_misalignment_mm: float,
        surface_scoring_pct: float,
        rust_pct: float,
        dry_contact_pct: float = 0.0
    ) -> GuideRailInspectionResult:
        """
        Calculate Visual Health Index for elevator guide rail blade and joints.
        """
        joint_misalignment_mm = max(0.0, float(joint_misalignment_mm))
        surface_scoring_pct = max(0.0, float(surface_scoring_pct))
        rust_pct = max(0.0, float(rust_pct))
        dry_contact_pct = max(0.0, float(dry_contact_pct))

        norm_joint = min(1.0, joint_misalignment_mm / self.max_joint_step_mm)
        norm_scoring = min(1.0, surface_scoring_pct / self.max_scoring_pct)
        norm_rust = min(1.0, rust_pct / self.max_rust_pct)
        norm_dry = min(1.0, dry_contact_pct / self.max_dry_contact_pct)

        penalty = (
            (self.w_joint * norm_joint) +
            (self.w_scoring * norm_scoring) +
            (self.w_rust * norm_rust) +
            (self.w_dry * norm_dry)
        )
        hi_rail_vis = max(0.0, min(1.0, 1.0 - penalty))

        fault_reasons = []
        if joint_misalignment_mm > self.max_joint_step_mm:
            fault_reasons.append(
                f"Guide rail joint step ({joint_misalignment_mm:.2f} mm) exceeds ISO 7465 tolerance ({self.max_joint_step_mm:.2f} mm)"
            )
        if surface_scoring_pct > self.max_scoring_pct:
            fault_reasons.append(
                f"Severe rail scoring ({surface_scoring_pct * 100:.1f}%) exceeds allowable surface roughness threshold"
            )
        if rust_pct > self.max_rust_pct:
            fault_reasons.append(
                f"Rail blade corrosion ({rust_pct * 100:.1f}%) impairs smooth guide shoe contact"
            )
        if dry_contact_pct > self.max_dry_contact_pct:
            fault_reasons.append(
                f"Inadequate guide rail lubrication ({dry_contact_pct * 100:.1f}% dry contact)"
            )

        is_service_required = len(fault_reasons) > 0

        if is_service_required:
            status = "CRITICAL_REALIGNMENT_REQUIRED" if joint_misalignment_mm > self.max_joint_step_mm else "MAINTENANCE_REQUIRED"
            recommendation = (
                "IMMEDIATE ACTION REQUIRED: Guide rail joint step or severe surface flaw detected. "
                "Align fishplates, polish rail blade, and inspect guide shoe liners."
            )
            hi_rail_vis = min(hi_rail_vis, 0.25)
        elif hi_rail_vis < 0.50:
            status = "WARNING_MAINTENANCE_NEAR"
            recommendation = (
                "SCHEDULE RAIL SERVICING: Minor joint deviation or scratch accumulation detected. "
                "Schedule guide rail re-alignment and lubricator refill within 14 operational days."
            )
        elif hi_rail_vis < 0.75:
            status = "MONITOR_FREQUENTLY"
            recommendation = (
                "MONITOR GUIDE RAIL: Slight surface scoring. Check automatic rail lubricators and guide shoe play."
            )
        else:
            status = "NOMINAL_HEALTHY"
            recommendation = "Guide rail in pristine alignment and surface condition compliant with ISO 7465."

        return GuideRailInspectionResult(
            hi_rail_vis=hi_rail_vis,
            status=status,
            joint_misalignment_mm=joint_misalignment_mm,
            surface_scoring_pct=surface_scoring_pct,
            rust_pct=rust_pct,
            dry_contact_pct=dry_contact_pct,
            is_service_required=is_service_required,
            fault_reasons=fault_reasons,
            action_recommendation=recommendation
        )
