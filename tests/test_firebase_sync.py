"""
Unit Tests for Multi-Elevator Firebase Realtime Database (RTDB) Sync & Fleet Aggregator
Tests offline mock simulation, RTDB REST interactions, degradation trajectories,
and fault condition handling.
"""

import pytest
from unittest.mock import patch, MagicMock
import requests

from src.edge.firebase_manager import FirebaseManager
from src.fusion.rul_estimator import RULEstimator


@pytest.fixture
def manager():
    """Provides a fresh offline FirebaseManager instance for tests."""
    mgr = FirebaseManager(
        database_url="https://elevatorguard-kone-default-rtdb.firebaseio.com",
        auth_key="test_secret_key",
        offline_mode=True,
        timeout=1.0,
        fallback_to_mock=True
    )
    yield mgr
    mgr.close()


def test_initialization_and_url_formatting():
    """Verify initialization and URL / parameter formatting."""
    mgr = FirebaseManager(
        database_url="elevatorguard-kone-default-rtdb.firebaseio.com/",
        auth_key="my_token",
        offline_mode=True
    )
    assert mgr.database_url == "https://elevatorguard-kone-default-rtdb.firebaseio.com"
    assert mgr.auth_key == "my_token"
    assert mgr.is_offline() is True

    url = mgr._build_url("elevators/LIFT-CAR-01/live")
    assert url == "https://elevatorguard-kone-default-rtdb.firebaseio.com/elevators/LIFT-CAR-01/live.json?auth=my_token"
    mgr.close()


def test_mock_get_all_elevators(manager):
    """Verify pre-seeded mock elevators (CAR-01, CAR-02, EXPRESS-03)."""
    elevators = manager.get_all_elevators()
    assert isinstance(elevators, dict)
    assert len(elevators) == 3

    assert "LIFT-CAR-01" in elevators
    assert "LIFT-CAR-02" in elevators
    assert "LIFT-EXPRESS-03" in elevators

    # Check distinct states
    car01 = elevators["LIFT-CAR-01"]
    assert car01["status"] == "NORMAL"
    assert car01["prognostics"]["health_index"] > 0.90
    assert car01["vibration"]["alarm"] is False

    car02 = elevators["LIFT-CAR-02"]
    assert car02["status"] == "WARNING"
    assert 0.50 <= car02["prognostics"]["health_index"] <= 0.70

    car03 = elevators["LIFT-EXPRESS-03"]
    assert car03["status"] == "CRITICAL"
    assert car03["vibration"]["alarm"] is True
    assert car03["vibration"]["rms_z"] > 0.45  # ISO 18738 alarm limit


def test_get_elevator_state(manager):
    """Test retrieving single elevator node state."""
    state = manager.get_elevator_state("LIFT-CAR-01")
    assert state is not None
    assert state["device_id"] == "LIFT-CAR-01"
    assert "vibration" in state
    assert "prognostics" in state
    assert "vision" in state

    # Test non-existent elevator
    missing = manager.get_elevator_state("LIFT-NONEXISTENT-99")
    assert missing is None


def test_update_elevator_state(manager):
    """Test pushing updates to an existing elevator and adding a new elevator."""
    update_payload = {
        "floor": 7,
        "velocity_mps": 1.75,
        "door_open": True,
        "vibration": {
            "rms_z": 0.095,
            "alarm": False
        }
    }

    success = manager.update_elevator_state("LIFT-CAR-01", update_payload)
    assert success is True

    updated_state = manager.get_elevator_state("LIFT-CAR-01")
    assert updated_state["floor"] == 7
    assert updated_state["velocity_mps"] == 1.75
    assert updated_state["door_open"] is True
    assert updated_state["vibration"]["rms_z"] == 0.095

    # Adding a new dynamic elevator
    new_car_payload = {
        "name": "Tower Freight D",
        "status": "NORMAL",
        "floor": 1,
        "velocity_mps": 0.0,
        "prognostics": {"health_index": 0.99, "remaining_days": 1400.0},
        "vibration": {"rms_z": 0.04, "alarm": False}
    }
    success_new = manager.update_elevator_state("LIFT-SERVICE-04", new_car_payload)
    assert success_new is True

    new_car = manager.get_elevator_state("LIFT-SERVICE-04")
    assert new_car is not None
    assert new_car["name"] == "Tower Freight D"
    assert "LIFT-SERVICE-04" in manager.list_elevator_ids()

    # Invalid payload raises ValueError
    with pytest.raises(ValueError):
        manager.update_elevator_state("LIFT-CAR-01", "invalid_string_payload")


def test_historical_telemetry_and_trajectories(manager):
    """Test degradation trajectory caching and shape."""
    history = manager.get_history("LIFT-CAR-01", limit=15)
    assert isinstance(history, list)
    assert len(history) == 15
    for p in history:
        assert "ts" in p
        assert "health_index" in p
        assert "vibration_rms" in p
        assert "trips" in p

    # Degradation trajectory inspection
    trajectory = manager.get_degradation_trajectory("LIFT-CAR-01")
    assert trajectory["elevator_id"] == "LIFT-CAR-01"
    assert len(trajectory["trip_history"]) > 0
    assert len(trajectory["hi_history"]) == len(trajectory["trip_history"])
    # Health Index should degrade or remain steady over lifetime
    assert trajectory["hi_history"][0] >= trajectory["hi_history"][-1]

    # Test pushing a custom history point
    custom_point = {
        "trips": 18000,
        "health_index": 0.938,
        "vibration_rms": 0.088,
        "wear_stage": "Stage 1: Pristine"
    }
    manager.push_history_point("LIFT-CAR-01", custom_point)
    latest_hist = manager.get_history("LIFT-CAR-01", limit=1)
    assert latest_hist[0]["health_index"] == 0.938

    # Test clearing history
    manager.clear_history("LIFT-CAR-01")
    assert len(manager.get_history("LIFT-CAR-01")) == 0
    # Other elevator history must remain intact
    assert len(manager.get_history("LIFT-CAR-02")) > 0


def test_fleet_summary_aggregation(manager):
    """Test fleet data aggregation calculation across all elevators."""
    summary = manager.get_fleet_summary()

    assert summary["total_elevators"] == 3
    assert summary["online_count"] == 3
    assert summary["status_breakdown"]["NORMAL"] == 1
    assert summary["status_breakdown"]["WARNING"] == 1
    assert summary["status_breakdown"]["CRITICAL"] == 1

    # CAR-03 has critical alarm active
    assert "LIFT-EXPRESS-03" in summary["elevators_with_alarms"]

    # Lowest RUL should be CAR-03 (~48.8 days)
    assert summary["lowest_rul_elevator"]["elevator_id"] == "LIFT-EXPRESS-03"
    assert summary["lowest_rul_elevator"]["remaining_days"] < 100.0

    # Highest vibration should be CAR-03 (> 0.50 m/s^2)
    assert summary["highest_vibration_elevator"]["elevator_id"] == "LIFT-EXPRESS-03"
    assert summary["highest_vibration_elevator"]["rms_z"] > 0.50

    # Average Health Index
    assert 0.0 < summary["average_health_index"] < 1.0


def test_simulation_tick(manager):
    """Test simulated telemetry advance and live fluctuations."""
    initial_state = manager.get_elevator_state("LIFT-CAR-01")
    initial_rms = initial_state["vibration"]["rms_z"]

    manager.simulate_tick(dt_sec=1.0)
    ticked_state = manager.get_elevator_state("LIFT-CAR-01")

    # Timestamp is refreshed
    assert ticked_state["timestamp_ms"] >= initial_state["timestamp_ms"]
    assert "rms_z" in ticked_state["vibration"]


def test_online_rest_api_calls():
    """Test that online mode correctly performs HTTP REST operations."""
    mgr = FirebaseManager(
        database_url="https://elevatorguard-kone-default-rtdb.firebaseio.com",
        auth_key="test_key",
        offline_mode=False
    )

    # Mock responses for GET all elevators
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "ELEVATOR-TEST-01": {
            "live": {
                "device_id": "ELEVATOR-TEST-01",
                "status": "NORMAL",
                "floor": 3
            }
        }
    }

    # Mock response for PUT
    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 200

    # Mock response for POST history
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200

    with patch.object(mgr.session, "get", return_value=mock_get_resp) as mock_get:
        elevators = mgr.get_all_elevators()
        assert "ELEVATOR-TEST-01" in elevators
        assert elevators["ELEVATOR-TEST-01"]["floor"] == 3
        mock_get.assert_called_once()
        expected_url = "https://elevatorguard-kone-default-rtdb.firebaseio.com/elevators.json?auth=test_key"
        assert mock_get.call_args[0][0] == expected_url

    with patch.object(mgr.session, "put", return_value=mock_put_resp) as mock_put:
        success = mgr.update_elevator_state("ELEVATOR-TEST-01", {"floor": 5})
        assert success is True
        mock_put.assert_called_once()
        expected_put_url = "https://elevatorguard-kone-default-rtdb.firebaseio.com/elevators/ELEVATOR-TEST-01/live.json?auth=test_key"
        assert mock_put.call_args[0][0] == expected_put_url

    with patch.object(mgr.session, "post", return_value=mock_post_resp) as mock_post:
        success_hist = mgr.push_history_point("ELEVATOR-TEST-01", {"health_index": 0.92})
        assert success_hist is True
        mock_post.assert_called_once()

    mgr.close()


def test_rest_api_network_failure_fallback():
    """Verify that network exceptions trigger fallback to mock store without crashing."""
    mgr = FirebaseManager(
        database_url="https://elevatorguard-kone-default-rtdb.firebaseio.com",
        offline_mode=False,
        fallback_to_mock=True
    )

    with patch.object(mgr.session, "get", side_effect=requests.ConnectionError("Network down")):
        # Should gracefully fall back to local mock elevators
        elevators = mgr.get_all_elevators()
        assert "LIFT-CAR-01" in elevators

        # Single elevator should also fall back to local mock
        single = mgr.get_elevator_state("LIFT-CAR-01")
        assert single is not None
        assert single["device_id"] == "LIFT-CAR-01"

    with patch.object(mgr.session, "put", side_effect=requests.Timeout("Timeout")):
        # Update should fall back to saving locally
        success = mgr.update_elevator_state("LIFT-CAR-01", {"floor": 6})
        assert success is True
        state = mgr.get_elevator_state("LIFT-CAR-01")
        assert state["floor"] == 6

    mgr.close()


def test_rul_estimator_integration(manager):
    """
    Verify integration between Firebase degradation trajectories
    and the RULEstimator prognostic engine.
    """
    estimator = RULEstimator(hi_discard_limit=0.20, average_trips_per_day=250)

    # Test CAR-02 trajectory
    trajectory = manager.get_degradation_trajectory("LIFT-CAR-02")
    trips = trajectory["trip_history"]
    his = trajectory["hi_history"]

    assert len(trips) >= 10
    assert len(his) >= 10

    rul_result = estimator.estimate_rul(
        current_hi=trajectory["current_hi"],
        current_trips=trajectory["current_trips"],
        hi_history=his,
        trip_history=trips
    )

    assert rul_result.rul_cycles > 0
    assert rul_result.rul_days > 0.0
    assert rul_result.health_status in ("NOMINAL_HEALTHY", "INSPECTION_ADVISED", "REPLACEMENT_SCHEDULED", "CRITICAL_DISCARD")
    assert len(rul_result.trajectory_trips) > 0
