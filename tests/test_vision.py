"""
Unit Tests for ElevatorGuard-AI Computer Vision & ISO 4309 Engine
"""

import numpy as np
import pytest
import cv2
from src.vision.visual_hi import ISO4309VisualAnalyzer, ISO7465GuideRailVisualAnalyzer
from src.vision.defect_analyzer import CableDefectDetector, GuideRailDefectDetector


def test_iso4309_pristine_rope():
    analyzer = ISO4309VisualAnalyzer()
    res = analyzer.calculate_hi(wire_breaks=0, wear_pct=0.0, corrosion_pct=0.0)
    assert res.hi_vis == 1.0
    assert not res.is_discard_required
    assert res.status == "NOMINAL_HEALTHY"


def test_iso4309_critical_wire_breaks():
    analyzer = ISO4309VisualAnalyzer()
    # 6 breaks triggers immediate discard
    res = analyzer.calculate_hi(wire_breaks=6, wear_pct=0.02, corrosion_pct=0.01)
    assert res.is_discard_required
    assert res.status == "CRITICAL_DISCARD"
    assert res.hi_vis < 0.20


def test_iso4309_wear_and_corrosion_limits():
    analyzer = ISO4309VisualAnalyzer()
    # 10% diameter wear triggers discard
    res = analyzer.calculate_hi(wire_breaks=2, wear_pct=0.10, corrosion_pct=0.02)
    assert res.is_discard_required

    # 15% corrosion triggers discard
    res_corr = analyzer.calculate_hi(wire_breaks=1, wear_pct=0.03, corrosion_pct=0.15)
    assert res_corr.is_discard_required


def test_cable_defect_detector_frame_analysis():
    detector = CableDefectDetector()
    dummy_frame = np.ones((480, 480, 3), dtype=np.uint8) * 120
    hi_res, dets, annotated = detector.analyze_frame(dummy_frame)

    assert hi_res is not None
    assert isinstance(dets, list)
    assert annotated.shape == dummy_frame.shape


def test_guide_rail_defect_detector_analysis():
    detector = GuideRailDefectDetector()
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    hi_res, dets, annotated = detector.analyze_rail_frame(dummy_frame, return_annotated=True)

    assert hi_res is not None
    assert hasattr(hi_res, "hi_rail_vis")
    assert isinstance(dets, list)
    assert annotated.shape == dummy_frame.shape


def test_guide_rail_defect_classes():
    detector = GuideRailDefectDetector()
    expected_classes = {'rail_surface_wear', 'joint_misalignment', 'pitting_galling', 'rail_corrosion'}
    assert expected_classes.issubset(set(detector.CLASS_NAMES))

    # Create synthetic frame with synthetic defect triggers
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Rust patch (HSV 15, 200, 200) -> in BGR
    rust_bgr = cv2.cvtColor(np.uint8([[[15, 200, 200]]]), cv2.COLOR_HSV2BGR)[0][0]
    test_img[50:120, 50:120] = rust_bgr

    # Misalignment horizontal line
    cv2.line(test_img, (20, 200), (280, 200), (255, 255, 255), 3)

    hi_res, dets, annotated = detector.analyze_rail_frame(test_img)
    det_classes = [d["class_name"] for d in dets]
    assert "rail_corrosion" in det_classes
    assert annotated.shape == test_img.shape

