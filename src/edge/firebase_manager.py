"""
ElevatorGuard-AI: Firebase Realtime Database (RTDB) Fleet Sync & Aggregation Service
Manages cloud synchronization and local mock aggregation across multi-elevator deployments.
Supports direct REST API interactions with Firebase RTDB (no external heavy SDK needed),
offline/mock fallback, dynamic mock simulation for fleet nodes, and historical RUL caching.
"""

import os
import time
import math
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from copy import deepcopy
import threading
import yaml

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None

logger = logging.getLogger("ElevatorGuard.FirebaseManager")


class FirebaseManager:
    """
    Fleet Data Aggregator and Firebase Realtime Database Client.
    Communicates via Firebase REST API (/elevators/<elevator_id>.json).
    Provides in-memory fallback, multi-elevator mock data simulation,
    and historical telemetry trajectory caching for RUL prognostics.
    """

    DEFAULT_ELEVATOR_IDS = ["LIFT-CAR-01", "LIFT-CAR-02", "LIFT-EXPRESS-03"]

    def __init__(
        self,
        database_url: Optional[str] = None,
        auth_key: Optional[str] = None,
        offline_mode: Optional[bool] = None,
        timeout: float = 3.0,
        config_path: str = "config.yaml",
        fallback_to_mock: bool = True,
        history_maxlen: int = 500
    ):
        """
        Initialize the Firebase RTDB Manager.

        Args:
            database_url: Firebase RTDB base URL (e.g. https://<project>-default-rtdb.firebaseio.com)
            auth_key: Optional Firebase secret/auth key
            offline_mode: Explicitly enable/disable offline mode (uses mock data when True)
            timeout: HTTP timeout in seconds
            config_path: Path to config.yaml for fallback configuration
            fallback_to_mock: Whether network failures automatically fall back to mock data
            history_maxlen: Max historical points retained per elevator
        """
        self.config_path = config_path
        cfg = self._load_yaml_config(config_path)
        fb_cfg = cfg.get("firebase", {}) if isinstance(cfg, dict) else {}

        # Resolve Database URL
        self.database_url = (
            database_url
            or os.environ.get("FIREBASE_RTDB_URL")
            or fb_cfg.get("database_url")
            or "https://elevatorguard-kone-default-rtdb.firebaseio.com"
        ).rstrip("/")

        if not (self.database_url.startswith("http://") or self.database_url.startswith("https://")):
            self.database_url = f"https://{self.database_url}"

        # Resolve Auth Key
        self.auth_key = (
            auth_key
            if auth_key is not None
            else os.environ.get("FIREBASE_AUTH_KEY", fb_cfg.get("auth_key", ""))
        )

        # Resolve Offline Mode
        if offline_mode is not None:
            self.offline_mode = offline_mode
        elif "FIREBASE_OFFLINE_MODE" in os.environ:
            self.offline_mode = os.environ["FIREBASE_OFFLINE_MODE"].lower() in ("true", "1", "yes")
        elif "offline_mode" in fb_cfg:
            self.offline_mode = bool(fb_cfg["offline_mode"])
        else:
            self.offline_mode = True

        self.timeout = timeout
        self.fallback_to_mock = fallback_to_mock
        self.history_maxlen = history_maxlen

        # Thread synchronization
        self._lock = threading.RLock()

        # In-memory fleet state store and degradation history cache
        self._elevators: Dict[str, Dict[str, Any]] = {}
        self._history_cache: Dict[str, deque] = {}

        # HTTP Session setup
        self._session: Optional[Any] = None
        if requests is not None:
            self._session = requests.Session()
            retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retries)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

        # Pre-seed realistic multi-elevator fleet mock data and degradation trajectories
        self.seed_mock_data()

    @property
    def session(self) -> Any:
        if self._session is None and requests is not None:
            self._session = requests.Session()
        return self._session

    def _load_yaml_config(self, path: str) -> Dict[str, Any]:
        """Safely load YAML configuration file if present."""
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Could not load config file {path}: {e}")
        return {}

    def _build_url(self, path: str) -> str:
        """Construct full RTDB REST endpoint URL with optional auth param."""
        clean_path = path.strip("/")
        url = f"{self.database_url}/{clean_path}.json"
        if self.auth_key:
            url += f"?auth={self.auth_key}"
        return url

    def set_offline_mode(self, offline: bool) -> None:
        """Toggle offline simulation mode."""
        with self._lock:
            self.offline_mode = offline
            logger.info(f"FirebaseManager offline_mode set to {offline}")

    def is_offline(self) -> bool:
        """Return True if running in offline mock mode."""
        return self.offline_mode

    # =========================================================================
    # Mock Data Simulation & Seeding
    # =========================================================================

    def seed_mock_data(self) -> None:
        """
        Seed mock state and historical RUL degradation trajectories for:
        - LIFT-CAR-01: Standard Passenger Car, Pristine / Healthy condition.
        - LIFT-CAR-02: Freight / Service Car, Moderate Wear / Warning condition.
        - LIFT-EXPRESS-03: High-Speed Express Car, Critical / Discard alert condition.
        """
        now_ms = int(time.time() * 1000)

        with self._lock:
            self._elevators.clear()
            self._history_cache.clear()

            # -------------------------------------------------------------
            # LIFT-CAR-01: Healthy Normal Passenger Car
            # -------------------------------------------------------------
            car_01_id = "LIFT-CAR-01"
            self._elevators[car_01_id] = {
                "device_id": car_01_id,
                "name": "Passenger Lift A (Standard)",
                "status": "NORMAL",
                "floor": 4,
                "target_floor": 4,
                "velocity_mps": 0.0,
                "direction": "IDLE",
                "door_open": False,
                "load_kg": 350.0,
                "vibration": {
                    "frequency_hz": 5.15,
                    "intensity": 0.082,
                    "rms_z": 0.085,
                    "alarm": False
                },
                "prognostics": {
                    "baseline_freq_hz": 5.0,
                    "delta_freq_hz": 0.15,
                    "health_index": 0.942,
                    "remaining_trips": 282600,
                    "remaining_days": 1130.4,
                    "wear_stage": "Stage 1: Pristine",
                    "action_plan": "Optimal condition. Next standard maintenance in 180 days."
                },
                "vision": {
                    "rail_detected": True,
                    "surface_wear": 0.018,
                    "joint_step_mm": 0.08,
                    "visual_hi": 0.965
                },
                "current_trips": 17400,
                "nominal_lifespan_trips": 300000,
                "timestamp_ms": now_ms
            }
            self._seed_trajectory(
                elevator_id=car_01_id,
                total_trips=17400,
                start_hi=1.00,
                end_hi=0.942,
                start_rms=0.050,
                end_rms=0.085,
                start_hz=5.00,
                end_hz=5.15,
                num_points=30,
                wear_stage="Stage 1: Pristine"
            )

            # -------------------------------------------------------------
            # LIFT-CAR-02: Moderate Wear Freight / Service Car
            # -------------------------------------------------------------
            car_02_id = "LIFT-CAR-02"
            self._elevators[car_02_id] = {
                "device_id": car_02_id,
                "name": "Freight / Service Lift B",
                "status": "WARNING",
                "floor": 1,
                "target_floor": 1,
                "velocity_mps": 0.0,
                "direction": "IDLE",
                "door_open": True,
                "load_kg": 820.0,
                "vibration": {
                    "frequency_hz": 12.80,
                    "intensity": 0.284,
                    "rms_z": 0.292,
                    "alarm": False
                },
                "prognostics": {
                    "baseline_freq_hz": 5.0,
                    "delta_freq_hz": 7.80,
                    "health_index": 0.618,
                    "remaining_trips": 65100,
                    "remaining_days": 260.4,
                    "wear_stage": "Stage 2: Moderate Wear",
                    "action_plan": "Schedule guide shoe alignment and rail lubrication within 30 days."
                },
                "vision": {
                    "rail_detected": True,
                    "surface_wear": 0.225,
                    "joint_step_mm": 0.450,
                    "visual_hi": 0.635
                },
                "current_trips": 114600,
                "nominal_lifespan_trips": 300000,
                "timestamp_ms": now_ms
            }
            self._seed_trajectory(
                elevator_id=car_02_id,
                total_trips=114600,
                start_hi=1.00,
                end_hi=0.618,
                start_rms=0.055,
                end_rms=0.292,
                start_hz=5.00,
                end_hz=12.80,
                num_points=35,
                wear_stage="Stage 2: Moderate Wear"
            )

            # -------------------------------------------------------------
            # LIFT-EXPRESS-03: High Vibration Critical Express Car
            # -------------------------------------------------------------
            car_03_id = "LIFT-EXPRESS-03"
            self._elevators[car_03_id] = {
                "device_id": car_03_id,
                "name": "High-Speed Express Car C",
                "status": "CRITICAL",
                "floor": 14,
                "target_floor": 1,
                "velocity_mps": -2.25,
                "direction": "DOWN",
                "door_open": False,
                "load_kg": 460.0,
                "vibration": {
                    "frequency_hz": 24.50,
                    "intensity": 0.525,
                    "rms_z": 0.540,
                    "alarm": True
                },
                "prognostics": {
                    "baseline_freq_hz": 5.0,
                    "delta_freq_hz": 19.50,
                    "health_index": 0.348,
                    "remaining_trips": 12200,
                    "remaining_days": 48.8,
                    "wear_stage": "Stage 3: Severe Degradation",
                    "action_plan": "Urgent maintenance: Vibration exceeds ISO 18738 ride quality limit (0.45 m/s²). Inspect rail joint fishplates immediately."
                },
                "vision": {
                    "rail_detected": True,
                    "surface_wear": 0.485,
                    "joint_step_mm": 1.150,
                    "visual_hi": 0.315
                },
                "current_trips": 247800,
                "nominal_lifespan_trips": 300000,
                "timestamp_ms": now_ms
            }
            self._seed_trajectory(
                elevator_id=car_03_id,
                total_trips=247800,
                start_hi=1.00,
                end_hi=0.348,
                start_rms=0.060,
                end_rms=0.540,
                start_hz=5.00,
                end_hz=24.50,
                num_points=40,
                wear_stage="Stage 3: Severe Degradation"
            )

    def _seed_trajectory(
        self,
        elevator_id: str,
        total_trips: int,
        start_hi: float,
        end_hi: float,
        start_rms: float,
        end_rms: float,
        start_hz: float,
        end_hz: float,
        num_points: int,
        wear_stage: str
    ) -> None:
        """Generate smooth synthetic degradation history curves for RUL testing."""
        if elevator_id not in self._history_cache:
            self._history_cache[elevator_id] = deque(maxlen=self.history_maxlen)

        q = self._history_cache[elevator_id]
        base_time = time.time() - (num_points * 3600)

        for i in range(num_points):
            frac = i / max(1, num_points - 1)
            # Power law curve for realistic exponential degradation rate
            degrade_factor = frac ** 1.3
            hi = max(0.0, min(1.0, start_hi - (start_hi - end_hi) * degrade_factor))
            rms = start_rms + (end_rms - start_rms) * (frac ** 1.2)
            hz = start_hz + (end_hz - start_hz) * (frac ** 1.1)
            trips = int(total_trips * frac)
            rul_days = round(max(0.0, (hi - 0.20) * 1200.0), 1)

            stage = "Stage 1: Pristine"
            if hi < 0.40:
                stage = "Stage 3: Severe Degradation"
            elif hi < 0.70:
                stage = "Stage 2: Moderate Wear"

            point = {
                "ts": int((base_time + (i * 3600)) * 1000),
                "trips": trips,
                "health_index": round(hi, 4),
                "vibration_rms": round(rms, 4),
                "vibration_hz": round(hz, 2),
                "rul_days": rul_days,
                "wear_stage": stage
            }
            q.append(point)

    def simulate_tick(self, dt_sec: float = 1.0) -> Dict[str, Dict[str, Any]]:
        """
        Advance simulated physics & telemetry for mock elevators.
        Generates realistic minor fluctuations in vibration, floor movements, and timestamps.
        """
        with self._lock:
            now_ms = int(time.time() * 1000)

            # CAR-01: moving smoothly
            car01 = self._elevators.get("LIFT-CAR-01")
            if car01:
                car01["timestamp_ms"] = now_ms
                vib = car01.setdefault("vibration", {})
                vib["rms_z"] = round(0.080 + random.uniform(-0.008, 0.008), 3)
                vib["intensity"] = vib["rms_z"]
                vib["frequency_hz"] = round(5.15 + random.uniform(-0.1, 0.1), 2)
                # Cache a point occasionally
                self.cache_telemetry_point("LIFT-CAR-01", car01)

            # CAR-02: service elevator idle with minor drift
            car02 = self._elevators.get("LIFT-CAR-02")
            if car02:
                car02["timestamp_ms"] = now_ms
                vib = car02.setdefault("vibration", {})
                vib["rms_z"] = round(0.285 + random.uniform(-0.015, 0.015), 3)
                vib["intensity"] = vib["rms_z"]
                self.cache_telemetry_point("LIFT-CAR-02", car02)

            # EXPRESS-03: high vibration alarm condition
            car03 = self._elevators.get("LIFT-EXPRESS-03")
            if car03:
                car03["timestamp_ms"] = now_ms
                vib = car03.setdefault("vibration", {})
                vib["rms_z"] = round(0.535 + random.uniform(-0.025, 0.025), 3)
                vib["intensity"] = vib["rms_z"]
                vib["alarm"] = vib["rms_z"] > 0.450
                self.cache_telemetry_point("LIFT-EXPRESS-03", car03)

            return deepcopy(self._elevators)

    # =========================================================================
    # RTDB Fetch & Query Operations
    # =========================================================================

    def get_all_elevators(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch current live state for all elevator systems in the fleet.
        
        Returns:
            Dict mapping elevator_id -> state telemetry dictionary.
        """
        if not self.offline_mode and self.session is not None:
            try:
                url = self._build_url("elevators")
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        normalized = {}
                        for eid, val in data.items():
                            if isinstance(val, dict):
                                # Extract live sub-node if present
                                live = val.get("live", val)
                                normalized[eid] = live
                        with self._lock:
                            for eid, val in normalized.items():
                                self._update_local_cache(eid, val, record_history=False)
                        return normalized
            except Exception as e:
                logger.warning(f"Firebase RTDB get_all_elevators failed ({e}). Falling back to local cache.")
                if not self.fallback_to_mock:
                    return {}

        with self._lock:
            return deepcopy(self._elevators)

    def get_elevator_state(self, elevator_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch telemetry state for a single elevator node (/elevators/<elevator_id>).

        Args:
            elevator_id: Elevator identifier string (e.g. 'LIFT-CAR-01')

        Returns:
            Elevator telemetry dict or None if not found.
        """
        if not self.offline_mode and self.session is not None:
            try:
                # First attempt /elevators/<elevator_id>/live.json
                url = self._build_url(f"elevators/{elevator_id}/live")
                resp = self.session.get(url, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if data is not None and isinstance(data, dict):
                        self._update_local_cache(elevator_id, data, record_history=False)
                        return data

                # Fallback to /elevators/<elevator_id>.json
                url_root = self._build_url(f"elevators/{elevator_id}")
                resp_root = self.session.get(url_root, timeout=self.timeout)
                if resp_root.status_code == 200:
                    data = resp_root.json()
                    if data is not None and isinstance(data, dict):
                        live_data = data.get("live", data)
                        self._update_local_cache(elevator_id, live_data, record_history=False)
                        return live_data
            except Exception as e:
                logger.warning(f"Firebase RTDB get_elevator_state({elevator_id}) failed: {e}")
                if not self.fallback_to_mock:
                    return None

        with self._lock:
            state = self._elevators.get(elevator_id)
            return deepcopy(state) if state is not None else None

    # =========================================================================
    # RTDB State Update & History Push Operations
    # =========================================================================

    def update_elevator_state(self, elevator_id: str, payload: Dict[str, Any]) -> bool:
        """
        Push live telemetry update for an elevator node.
        Updates internal cache, history degradation records, and syncs to RTDB.

        Args:
            elevator_id: Target elevator ID (e.g. 'LIFT-CAR-01')
            payload: Telemetry state payload dict

        Returns:
            True if state was successfully updated/queued, False on unhandled failure.
        """
        if not isinstance(payload, dict):
            raise ValueError("Payload must be a dictionary")

        clean_payload = deepcopy(payload)
        if "device_id" not in clean_payload:
            clean_payload["device_id"] = elevator_id
        if "timestamp_ms" not in clean_payload:
            clean_payload["timestamp_ms"] = int(time.time() * 1000)

        # Update local in-memory store and historical RUL cache
        self._update_local_cache(elevator_id, clean_payload, record_history=True)

        if not self.offline_mode and self.session is not None:
            try:
                url = self._build_url(f"elevators/{elevator_id}/live")
                resp = self.session.put(url, json=clean_payload, timeout=self.timeout)
                if resp.status_code in (200, 204):
                    return True
                logger.warning(f"Firebase RTDB PUT returned status {resp.status_code}")
                return self.fallback_to_mock
            except Exception as e:
                logger.warning(f"Failed to push update to Firebase RTDB for {elevator_id}: {e}")
                return self.fallback_to_mock

        return True

    def push_history_point(self, elevator_id: str, point: Dict[str, Any]) -> bool:
        """
        Push a time-series history record to /elevators/<elevator_id>/history.json.
        Also caches locally for RUL degradation trajectory modeling.
        """
        if not isinstance(point, dict):
            raise ValueError("Point must be a dictionary")

        clean_point = deepcopy(point)
        if "ts" not in clean_point:
            clean_point["ts"] = int(time.time() * 1000)

        with self._lock:
            if elevator_id not in self._history_cache:
                self._history_cache[elevator_id] = deque(maxlen=self.history_maxlen)
            self._history_cache[elevator_id].append(clean_point)

        if not self.offline_mode and self.session is not None:
            try:
                url = self._build_url(f"elevators/{elevator_id}/history")
                resp = self.session.post(url, json=clean_point, timeout=self.timeout)
                return resp.status_code in (200, 204)
            except Exception as e:
                logger.warning(f"Failed to post history point to Firebase RTDB: {e}")
                return self.fallback_to_mock

        return True

    # =========================================================================
    # Historical Telemetry Caching & RUL Degradation Trajectories
    # =========================================================================

    def cache_telemetry_point(self, elevator_id: str, payload: Dict[str, Any]) -> None:
        """
        Extract key prognostics metrics from a telemetry payload and store into
        the historical trajectory cache.
        """
        ts = payload.get("timestamp_ms", int(time.time() * 1000))
        prog = payload.get("prognostics", {})
        vib = payload.get("vibration", {})

        # Extract Health Index
        hi = prog.get("health_index")
        if hi is None:
            hi = payload.get("health_index", 1.0)

        # Extract Vibration RMS
        rms = vib.get("rms_z")
        if rms is None:
            rms = payload.get("accel_rms_z", vib.get("intensity", 0.0))

        # Extract Vibration Freq
        hz = vib.get("frequency_hz", payload.get("pulse_rate_hz", 5.0))

        # Extract Trips & Days
        trips = payload.get("current_trips", prog.get("remaining_trips", 0))
        rul_days = prog.get("remaining_days", 0.0)
        stage = prog.get("wear_stage", "Normal")

        point = {
            "ts": ts,
            "trips": int(trips),
            "health_index": round(float(hi), 4),
            "vibration_rms": round(float(rms), 4),
            "vibration_hz": round(float(hz), 2),
            "rul_days": float(rul_days),
            "wear_stage": str(stage)
        }

        with self._lock:
            if elevator_id not in self._history_cache:
                self._history_cache[elevator_id] = deque(maxlen=self.history_maxlen)
            self._history_cache[elevator_id].append(point)

    def get_history(self, elevator_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve cached historical telemetry data points for an elevator.

        Args:
            elevator_id: Elevator ID
            limit: Maximum number of recent points to return

        Returns:
            List of telemetry history point dictionaries in chronological order.
        """
        with self._lock:
            q = self._history_cache.get(elevator_id)
            if not q:
                return []
            items = list(q)
            return items[-limit:] if limit > 0 else items

    def get_degradation_trajectory(self, elevator_id: str) -> Dict[str, Any]:
        """
        Extract degradation trajectory formatted for RULEstimator and lifecycle plotting.

        Returns:
            Dict containing:
                - elevator_id
                - trip_history: List[int]
                - hi_history: List[float]
                - rms_history: List[float]
                - timestamps: List[int]
                - current_hi: float
                - current_trips: int
                - wear_stage: str
                - action_plan: str
        """
        with self._lock:
            history = self.get_history(elevator_id, limit=self.history_maxlen)
            state = self._elevators.get(elevator_id, {})
            prog = state.get("prognostics", {})

            trip_history = [p.get("trips", 0) for p in history]
            hi_history = [p.get("health_index", 1.0) for p in history]
            rms_history = [p.get("vibration_rms", 0.0) for p in history]
            timestamps = [p.get("ts", 0) for p in history]

            current_hi = prog.get("health_index", hi_history[-1] if hi_history else 1.0)
            current_trips = state.get("current_trips", trip_history[-1] if trip_history else 0)

            return {
                "elevator_id": elevator_id,
                "trip_history": trip_history,
                "hi_history": hi_history,
                "rms_history": rms_history,
                "timestamps": timestamps,
                "current_hi": current_hi,
                "current_trips": current_trips,
                "wear_stage": prog.get("wear_stage", "Normal"),
                "action_plan": prog.get("action_plan", "No action needed.")
            }

    def clear_history(self, elevator_id: Optional[str] = None) -> None:
        """Clear cached history for a single elevator or all elevators."""
        with self._lock:
            if elevator_id:
                if elevator_id in self._history_cache:
                    self._history_cache[elevator_id].clear()
            else:
                self._history_cache.clear()

    # =========================================================================
    # Fleet Aggregation & Summary Service
    # =========================================================================

    def get_fleet_summary(self) -> Dict[str, Any]:
        """
        Aggregate fleet metrics across all managed elevator units.

        Returns:
            Fleet overview summary with status distribution, average HI,
            alarm states, and most critical elevator unit.
        """
        elevators = self.get_all_elevators()

        total = len(elevators)
        if total == 0:
            return {
                "total_elevators": 0,
                "online_count": 0,
                "status_breakdown": {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0},
                "average_health_index": 1.0,
                "average_vibration_rms": 0.0,
                "elevators_with_alarms": [],
                "lowest_rul_elevator": None,
                "mode": "offline" if self.offline_mode else "online",
                "timestamp_ms": int(time.time() * 1000)
            }

        status_counts = {"NORMAL": 0, "WARNING": 0, "CRITICAL": 0}
        hi_list: List[float] = []
        rms_list: List[float] = []
        alarms_active: List[str] = []
        lowest_rul: Optional[Tuple[str, float]] = None
        highest_vibration: Optional[Tuple[str, float]] = None

        for eid, state in elevators.items():
            st = str(state.get("status", "NORMAL")).upper()
            status_counts[st] = status_counts.get(st, 0) + 1

            prog = state.get("prognostics", {})
            vib = state.get("vibration", {})

            # Health index
            hi = prog.get("health_index")
            if hi is not None:
                hi_list.append(float(hi))

            # Vibration RMS
            rms = vib.get("rms_z", vib.get("intensity", 0.0))
            if rms is not None:
                rms_val = float(rms)
                rms_list.append(rms_val)
                if highest_vibration is None or rms_val > highest_vibration[1]:
                    highest_vibration = (eid, rms_val)

            # Alarms
            if vib.get("alarm") is True or state.get("alarm_active") is True:
                alarms_active.append(eid)

            # Remaining Days
            rem_days = prog.get("remaining_days")
            if rem_days is not None:
                days_val = float(rem_days)
                if lowest_rul is None or days_val < lowest_rul[1]:
                    lowest_rul = (eid, days_val)

        avg_hi = round(sum(hi_list) / len(hi_list), 3) if hi_list else 1.0
        avg_rms = round(sum(rms_list) / len(rms_list), 3) if rms_list else 0.0

        return {
            "total_elevators": total,
            "online_count": total,
            "status_breakdown": status_counts,
            "average_health_index": avg_hi,
            "average_vibration_rms": avg_rms,
            "elevators_with_alarms": alarms_active,
            "highest_vibration_elevator": {
                "elevator_id": highest_vibration[0],
                "rms_z": highest_vibration[1]
            } if highest_vibration else None,
            "lowest_rul_elevator": {
                "elevator_id": lowest_rul[0],
                "remaining_days": lowest_rul[1]
            } if lowest_rul else None,
            "mode": "offline" if self.offline_mode else "online",
            "timestamp_ms": int(time.time() * 1000)
        }

    # =========================================================================
    # Internal Utilities & Cache Management
    # =========================================================================

    def _update_local_cache(
        self,
        elevator_id: str,
        payload: Dict[str, Any],
        record_history: bool = True
    ) -> None:
        """Update internal store and optionally append to degradation history."""
        with self._lock:
            existing = self._elevators.get(elevator_id, {})
            # Shallow merge of dicts to preserve structure
            merged = deepcopy(existing)
            for k, v in payload.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k].update(v)
                else:
                    merged[k] = v

            self._elevators[elevator_id] = merged

            if record_history:
                self.cache_telemetry_point(elevator_id, merged)

    def list_elevator_ids(self) -> List[str]:
        """Return list of elevator IDs known to the manager."""
        with self._lock:
            return list(self._elevators.keys())

    def close(self) -> None:
        """Close underlying HTTP session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass

    def __enter__(self) -> "FirebaseManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
