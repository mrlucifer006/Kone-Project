"""
Unit Tests for Multi-Modal Decision Fusion & RUL Prognostics
"""

import pytest
from src.vision.visual_hi import ISO4309VisualAnalyzer
from src.fusion.multimodal_fusion import MultiModalFusionEngine
from src.fusion.rul_estimator import RULEstimator


def test_multimodal_fusion_healthy():
    analyzer = ISO4309VisualAnalyzer()
    engine = MultiModalFusionEngine(alpha_vision_weight=0.5, beta_telemetry_weight=0.5)

    vis_res = analyzer.calculate_hi(wire_breaks=0, wear_pct=0.01, corrosion_pct=0.01)
    hi_telemetry = 0.95

    fusion_res = engine.fuse(vis_res, hi_telemetry)
    assert fusion_res.hi_composite > 0.90
    assert not fusion_res.safety_override_active
    assert fusion_res.status == "NOMINAL_HEALTHY"


def test_multimodal_fusion_safety_override():
    analyzer = ISO4309VisualAnalyzer()
    engine = MultiModalFusionEngine()

    # Broken wire discard condition
    vis_res = analyzer.calculate_hi(wire_breaks=7, wear_pct=0.05, corrosion_pct=0.02)
    # Even if telemetry erroneously claims cable is 100% fine (1.0)
    hi_telemetry = 1.0

    fusion_res = engine.fuse(vis_res, hi_telemetry)
    # Safety override must clamp composite health below discard limit (0.20)
    assert fusion_res.safety_override_active is True
    assert fusion_res.hi_composite < 0.20
    assert fusion_res.status == "CRITICAL_DISCARD"


def test_rul_estimator():
    estimator = RULEstimator(hi_discard_limit=0.20, nominal_lifespan_trips=100000, average_trips_per_day=200)

    # Health = 0.60, current trips = 50,000
    res = estimator.estimate_rul(current_hi=0.60, current_trips=50000)
    assert res.rul_cycles > 0
    assert res.rul_days > 0.0
    assert res.predicted_failure_trip > 50000
    assert len(res.trajectory_trips) == 10
    assert len(res.trajectory_hi) == 10
