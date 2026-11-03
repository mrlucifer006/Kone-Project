"""
ElevatorGuard-AI: Hoistway Elevator Physics & Dynamics Simulation Engine
Models multi-floor elevator kinematics, S-curve velocity profiles,
dynamic suspension rope tensions, tri-axial car vibrations, and ISO 4309 fatigue degradation.
"""

import math
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ElevatorState:
    floor: int = 1
    target_floor: int = 1
    position_m: float = 0.0
    velocity_mps: float = 0.0
    acceleration_mps2: float = 0.0
    door_open_pct: float = 0.0 # 0.0 = closed, 1.0 = fully open
    is_moving: bool = False
    emergency_stop: bool = False
    passenger_count: int = 2
    car_mass_kg: float = 1200.0
    payload_mass_kg: float = 150.0
    
    # 4 Suspension Ropes
    rope_tensions_n: List[float] = field(default_factory=lambda: [2500.0, 2500.0, 2500.0, 2500.0])
    
    # Vibration telemetry [m/s^2]
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    
    # Degradation metrics
    total_trips: int = 42500
    wire_breaks: int = 0
    wear_pct: float = 0.02
    corrosion_pct: float = 0.01
    hi_simulated: float = 0.94


class ElevatorPhysicsSimulator:
    """
    Simulates a traction elevator hoistway with physical dynamics.
    """

    def __init__(
        self,
        num_floors: int = 8,
        floor_height_m: float = 3.5,
        max_speed_mps: float = 1.8,
        max_accel_mps2: float = 1.0,
        jerk_mps3: float = 1.5,
        num_ropes: int = 4
    ):
        self.num_floors = num_floors
        self.floor_height_m = floor_height_m
        self.max_speed = max_speed_mps
        self.max_accel = max_accel_mps2
        self.jerk = jerk_mps3
        self.num_ropes = num_ropes

        self.state = ElevatorState()
        self.state.position_m = 0.0 # Floor 1 is at 0.0 m
        self.last_update_time = time.time()

        # Tension calibration biases across ropes (simulates minor manufacturing variance)
        self.tension_bias = np.array([1.01, 0.99, 1.02, 0.98], dtype=np.float32)

        # Active defect injections
        self.broken_wire_count = 0
        self.wear_ratio = 0.02
        self.corrosion_ratio = 0.01
        self.tension_imbalance_factor = 0.0 # 0.0 = balanced, >0.15 = alert
        self.rail_joint_misalignment_mm = 0.05 # Nominal joint step < 0.20 mm
        self.rail_scoring_ratio = 0.02
        self.rail_corrosion_ratio = 0.01
        self.state.wire_breaks = self.broken_wire_count
        self.state.wear_pct = self.wear_ratio
        self.state.corrosion_pct = self.corrosion_ratio

    def dispatch_to_floor(self, target_floor: int) -> bool:
        """Command elevator to travel to target floor."""
        if self.state.emergency_stop:
            return False
        target_floor = max(1, min(self.num_floors, int(target_floor)))
        if target_floor != self.state.floor:
            self.state.target_floor = target_floor
            self.state.is_moving = True
            return True
        return False

    def trigger_emergency_stop(self):
        """Immediately engage safety gear / emergency brake."""
        self.state.emergency_stop = True
        self.state.velocity_mps = 0.0
        self.state.acceleration_mps2 = -3.5 # High deceleration spike
        self.state.is_moving = False

    def reset_emergency_stop(self):
        """Disengage emergency stop."""
        self.state.emergency_stop = False
        self.state.acceleration_mps2 = 0.0

    def inject_defect(self, defect_type: str, severity: float = 1.0):
        """
        Inject structural or sensor defect for testing.
        Types:
          Guide Rail: 'rail_joint_misalignment', 'rail_scoring', 'rail_corrosion', 'dry_rail'
          Legacy: 'broken_wire', 'abrasion', 'corrosion', 'tension_imbalance'
        """
        if defect_type in ("rail_joint_misalignment", "joint_misalignment"):
            self.rail_joint_misalignment_mm = max(0.25, severity * 0.35)
            self.broken_wire_count += int(max(1, severity)) # Correlated NDT flaw
        elif defect_type in ("rail_scoring", "abrasion"):
            self.rail_scoring_ratio = min(0.35, self.rail_scoring_ratio + (severity * 0.05))
            self.wear_ratio = self.rail_scoring_ratio
        elif defect_type in ("rail_corrosion", "corrosion"):
            self.rail_corrosion_ratio = min(0.30, self.rail_corrosion_ratio + (severity * 0.04))
            self.corrosion_ratio = self.rail_corrosion_ratio
        elif defect_type in ("dry_rail", "tension_imbalance", "guide_shoe_play"):
            self.tension_imbalance_factor = max(0.20, severity)
        elif defect_type == "broken_wire":
            self.broken_wire_count += int(max(1, severity))
            self.rail_joint_misalignment_mm = max(0.25, severity * 0.1)

    def step(self, dt: float = 0.02):
        """Advance physical simulation by dt seconds."""
        if self.state.emergency_stop:
            # Settle deceleration
            self.state.accel_z = 9.81 + np.random.normal(0, 0.02)
            self.state.accel_x = np.random.normal(0, 0.01)
            self.state.accel_y = np.random.normal(0, 0.01)
            return

        target_pos = (self.state.target_floor - 1) * self.floor_height_m
        dist = target_pos - self.state.position_m

        # Door handling
        if abs(dist) < 0.01:
            self.state.is_moving = False
            self.state.velocity_mps = 0.0
            self.state.acceleration_mps2 = 0.0
            self.state.position_m = target_pos
            self.state.floor = self.state.target_floor

            # Cycle doors if arrived
            if self.state.door_open_pct < 1.0:
                self.state.door_open_pct = min(1.0, self.state.door_open_pct + (dt * 1.5))
        else:
            self.state.is_moving = True
            self.state.door_open_pct = max(0.0, self.state.door_open_pct - (dt * 2.0))

            # Kinematic profile towards target
            direction = 1.0 if dist > 0 else -1.0
            abs_dist = abs(dist)
            
            # Braking distance: d = v^2 / (2 * a)
            stopping_dist = (self.state.velocity_mps ** 2) / (2.0 * self.max_accel + 1e-5)

            if abs_dist <= stopping_dist + 0.05:
                # Deceleration phase
                target_accel = -direction * self.max_accel
            else:
                # Acceleration or cruising phase
                if abs(self.state.velocity_mps) < self.max_speed:
                    target_accel = direction * self.max_accel
                else:
                    target_accel = 0.0

            # Smooth acceleration with jerk limit
            accel_diff = target_accel - self.state.acceleration_mps2
            max_jerk_step = self.jerk * dt
            self.state.acceleration_mps2 += max(-max_jerk_step, min(max_jerk_step, accel_diff))

            # Update velocity
            self.state.velocity_mps += self.state.acceleration_mps2 * dt
            
            # Clamp velocity
            if direction > 0:
                self.state.velocity_mps = min(self.max_speed, max(0.0, self.state.velocity_mps))
            else:
                self.state.velocity_mps = max(-self.max_speed, min(0.0, self.state.velocity_mps))

            # Update position
            self.state.position_m += self.state.velocity_mps * dt

            # Compute current floor from position
            self.state.floor = int(round(self.state.position_m / self.floor_height_m)) + 1
            self.state.floor = max(1, min(self.num_floors, self.state.floor))

        # ----------------- DYNAMIC TENSION SIMULATION -----------------
        total_mass = self.state.car_mass_kg + self.state.payload_mass_kg
        effective_g = 9.80665 + self.state.acceleration_mps2
        base_rope_tension = (total_mass * effective_g) / float(self.num_ropes)

        # Apply imbalance factor if injected
        tensions = []
        if self.tension_imbalance_factor > 0:
            # Overload rope 0, underload rope 3
            weights = np.array([
                1.0 + self.tension_imbalance_factor,
                1.0,
                1.0 - (self.tension_imbalance_factor * 0.4),
                1.0 - (self.tension_imbalance_factor * 0.6)
            ])
            weights = weights / np.mean(weights)
        else:
            weights = self.tension_bias

        for i in range(self.num_ropes):
            noise = np.random.normal(0, 8.0) # Load cell measurement ripple
            rope_t = (base_rope_tension * weights[i]) + noise
            tensions.append(float(max(100.0, rope_t)))

        self.state.rope_tensions_n = tensions

        # ----------------- VIBRATION SIMULATION (MPU-6050) -----------------
        # Z-axis (vertical): gravity + car acceleration + motor harmonics (15 Hz) + cable strand ripple
        motor_harmonics = 0.08 * math.sin(time.time() * 2.0 * math.pi * 14.5) if self.state.is_moving else 0.0
        strand_pulse = 0.25 * math.sin(time.time() * 2.0 * math.pi * 32.0) if (self.broken_wire_count > 2 and self.state.is_moving) else 0.0
        vibe_z_noise = np.random.normal(0, 0.04)

        self.state.accel_z = effective_g + motor_harmonics + strand_pulse + vibe_z_noise

        # ----------------- GUIDE RAIL VIBRATION SENSOR (MPU-6050) -----------------
        # Guide rail joint locations (spaced every 3.5 meters along hoistway)
        nearest_joint = round(self.state.position_m / 3.5) * 3.5
        dist_to_joint = abs(self.state.position_m - nearest_joint)
        joint_shock = 0.0
        if dist_to_joint < 0.25 and self.state.is_moving and nearest_joint > 0:
            # Impulsive lateral shock as guide shoes cross misaligned fishplate joint
            joint_shock = (self.rail_joint_misalignment_mm / 0.20) * 0.38 * math.sin(dist_to_joint * 20.0)

        # X-axis (lateral guide rail vibration): rail waviness + joint shock + guide shoe chatter
        sway_freq = 1.2
        rail_sway = 0.07 * math.sin(self.state.position_m * sway_freq) if self.state.is_moving else 0.0
        scoring_chatter = 0.15 * (self.rail_scoring_ratio / 0.15) * math.sin(time.time() * 2.0 * math.pi * 42.0) if self.state.is_moving else 0.0
        self.state.accel_x = rail_sway + joint_shock + scoring_chatter + np.random.normal(0, 0.02)

        # Y-axis (transverse rail face vibration): guide shoe face pressure
        self.state.accel_y = (0.05 * math.cos(self.state.position_m * 1.8) if self.state.is_moving else 0.0) + (joint_shock * 0.5) + np.random.normal(0, 0.02)

        # ----------------- FATIGUE & DEGRADATION UPDATE -----------------
        self.state.wire_breaks = self.broken_wire_count
        self.state.wear_pct = self.wear_ratio
        self.state.corrosion_pct = self.corrosion_ratio

        # Simulated combined health
        penalty = (0.5 * min(1.0, self.broken_wire_count / 6.0)) + (0.3 * min(1.0, self.wear_ratio / 0.10)) + (0.2 * min(1.0, self.corrosion_ratio / 0.15))
        self.state.hi_simulated = max(0.0, min(1.0, 1.0 - penalty))

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Format state as dictionary."""
        return {
            "floor": self.state.floor,
            "target_floor": self.state.target_floor,
            "position_m": round(self.state.position_m, 2),
            "velocity_mps": round(self.state.velocity_mps, 3),
            "acceleration_mps2": round(self.state.acceleration_mps2, 3),
            "door_open_pct": round(self.state.door_open_pct, 2),
            "is_moving": self.state.is_moving,
            "emergency_stop": self.state.emergency_stop,
            "accel": [
                round(self.state.accel_x, 3),
                round(self.state.accel_y, 3),
                round(self.state.accel_z, 3)
            ],
            "rope_tensions": [round(t, 1) for t in self.state.rope_tensions_n],
            "total_trips": self.state.total_trips,
            "wire_breaks": self.state.wire_breaks,
            "wear_pct": round(self.state.wear_pct * 100, 2),
            "corrosion_pct": round(self.state.corrosion_pct * 100, 2),
            "hi_simulated": round(self.state.hi_simulated, 3)
        }
