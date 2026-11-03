"""
ElevatorGuard-AI: Remaining Useful Life (RUL) Prognostics Estimator
Calculates remaining elevator operational trips and calendar days
based on continuous degradation rate and ISO 4309 discard limits.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class RULPredictionResult:
    rul_cycles: int
    rul_days: float
    current_trips: int
    predicted_failure_trip: int
    degradation_rate_per_trip: float
    health_status: str
    action_plan: str
    trajectory_trips: List[int]
    trajectory_hi: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rul_cycles": int(self.rul_cycles),
            "rul_days": round(float(self.rul_days), 1),
            "current_trips": int(self.current_trips),
            "predicted_failure_trip": int(self.predicted_failure_trip),
            "degradation_rate_per_trip": float(self.degradation_rate_per_trip),
            "health_status": self.health_status,
            "action_plan": self.action_plan,
            "trajectory_trips": self.trajectory_trips,
            "trajectory_hi": [round(v, 4) for v in self.trajectory_hi]
        }


class RULEstimator:
    """
    RUL Regression & Lifecycle Prognostics Engine.
    """

    def __init__(
        self,
        hi_discard_limit: float = 0.20,
        nominal_lifespan_trips: int = 150000,
        average_trips_per_day: int = 250
    ):
        self.hi_discard_limit = hi_discard_limit
        self.nominal_lifespan_trips = nominal_lifespan_trips
        self.trips_per_day = average_trips_per_day

    def estimate_rul(
        self,
        current_hi: float,
        current_trips: int,
        hi_history: Optional[List[float]] = None,
        trip_history: Optional[List[int]] = None
    ) -> RULPredictionResult:
        """
        Estimate RUL based on current composite health and historical degradation trajectory.
        
        Formula:
            RUL_cycles = (HI_current - HI_discard) / |d(HI)/dt|
        """
        current_hi = max(0.0, min(1.0, float(current_hi)))
        current_trips = max(0, int(current_trips))

        # Determine degradation rate d(HI)/dt
        if hi_history and trip_history and len(hi_history) >= 2 and len(trip_history) >= 2:
            h_arr = np.array(hi_history, dtype=np.float64)
            t_arr = np.array(trip_history, dtype=np.float64)
            delta_t = t_arr[-1] - t_arr[0]
            if delta_t > 0:
                # Linear regression slope
                slope, _ = np.polyfit(t_arr, h_arr, 1)
                deg_rate = abs(float(slope))
            else:
                deg_rate = (1.0 - self.hi_discard_limit) / float(self.nominal_lifespan_trips)
        else:
            # Baseline theoretical degradation rate from nominal design lifespan
            deg_rate = (1.0 - self.hi_discard_limit) / float(self.nominal_lifespan_trips)

        # Protect against division by zero
        deg_rate = max(1e-8, deg_rate)

        # Discard condition
        if current_hi <= self.hi_discard_limit:
            rul_cycles = 0
            rul_days = 0.0
            pred_fail_trip = current_trips
            status = "CRITICAL_DISCARD"
            action = "EMERGENCY: Ropes reached discard threshold. Ground elevator immediately and replace cables."
        else:
            usable_hi = current_hi - self.hi_discard_limit
            rul_cycles = int(usable_hi / deg_rate)
            rul_days = rul_cycles / float(self.trips_per_day)
            pred_fail_trip = current_trips + rul_cycles

            if current_hi < 0.45:
                status = "REPLACEMENT_SCHEDULED"
                action = f"Ropes within final 25% of life ({rul_days:.1f} days left). Procure replacement ropes."
            elif current_hi < 0.70:
                status = "INSPECTION_ADVISED"
                action = f"Normal progressive wear ({rul_days:.1f} days left). Perform lubrication and tension check."
            else:
                status = "NOMINAL_HEALTHY"
                action = f"System in optimal health ({rul_days:.1f} days / {rul_cycles:,} trips remaining)."

        # Generate future degradation trajectory curve (10 points)
        proj_trips = []
        proj_his = []
        step_trips = max(100, (pred_fail_trip - current_trips) // 9) if rul_cycles > 0 else 500

        for i in range(10):
            t_step = current_trips + (i * step_trips)
            h_step = max(0.0, current_hi - (deg_rate * (t_step - current_trips)))
            proj_trips.append(int(t_step))
            proj_his.append(float(h_step))

        return RULPredictionResult(
            rul_cycles=rul_cycles,
            rul_days=rul_days,
            current_trips=current_trips,
            predicted_failure_trip=pred_fail_trip,
            degradation_rate_per_trip=deg_rate,
            health_status=status,
            action_plan=action,
            trajectory_trips=proj_trips,
            trajectory_hi=proj_his
        )
