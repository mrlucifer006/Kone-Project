"""
Unit Tests for Elevator Guide Rail Edge Vision, Synthetic Dataset Generator, and ISO 7465 Scoring
"""

import os
from pathlib import Path
import numpy as np
import pytest
import yaml

from src.vision.rail_dataset_generator import (
    RailDatasetGenerator,
    RAIL_DEFECT_CLASSES,
    CLASS_TO_ID
)
from src.vision.defect_analyzer import GuideRailDefectDetector
from src.vision.visual_hi import ISO7465GuideRailVisualAnalyzer
from src.vision.train_detector import create_rail_dataset_yaml


@pytest.fixture
def generator():
    return RailDatasetGenerator(seed=42, mm_per_pixel=0.08)


@pytest.fixture
def detector():
    return GuideRailDefectDetector()


@pytest.fixture
def analyzer():
    return ISO7465GuideRailVisualAnalyzer()


# ==============================================================================
# 1. Synthetic Rail Image Generation Tests
# ==============================================================================

def test_pristine_rail_generation(generator):
    """Verify pristine machined guide rail image generation."""
    img = generator.generate_pristine_rail(width=640, height=640)
    
    assert img is not None
    assert img.shape == (640, 640, 3)
    assert img.dtype == np.uint8

    # Hoistway dark background check (corners)
    assert np.mean(img[:30, :30]) < 60

    # Machined guide rail blade center check (bright steel tone)
    blade_center_brightness = np.mean(img[200:440, 260:380])
    assert 120 < blade_center_brightness < 185

    # Pristine image with no defects requested
    img_clean, annotations = generator.generate_rail_image(width=640, height=640)
    assert len(annotations) == 0


def test_joint_gap_step_generation(generator):
    """Verify joint gap step generation and misalignment metadata."""
    img = generator.generate_pristine_rail(width=640, height=640)
    
    # 1. Step = 0.0 mm (no misalignment added)
    img_no_step, ann_none = generator.add_joint_gap_step(img, joint_step_mm=0.0)
    assert ann_none is None

    # 2. Step = 0.35 mm (exceeds ISO 7465 tolerance of 0.20 mm)
    img_step, ann = generator.add_joint_gap_step(img, joint_step_mm=0.35, y_pos=320)
    assert ann is not None
    assert ann["class_name"] == "joint_misalignment"
    assert ann["class_id"] == 1
    assert ann["joint_step_mm"] == pytest.approx(0.35, abs=0.01)

    # Check bounding box bounds
    bx1, by1, bx2, by2 = ann["bbox"]
    assert 0 <= bx1 < bx2 <= 640
    assert 0 <= by1 < by2 <= 640
    assert by1 <= 320 <= by2
    assert (bx2 - bx1) >= 200 # Spans across guide rail blade


def test_friction_wear_generation(generator):
    """Verify guide shoe friction wear marks / longitudinal scoring generation."""
    img = generator.generate_pristine_rail(width=640, height=640)
    img_wear, annotations = generator.add_friction_wear(img, wear_level=0.40)

    assert len(annotations) >= 1
    for ann in annotations:
        assert ann["class_name"] == "rail_surface_wear"
        assert ann["class_id"] == 0
        bx1, by1, bx2, by2 = ann["bbox"]
        assert 0 <= bx1 < bx2 <= 640
        assert 0 <= by1 < by2 <= 640
        # Friction wear marks are tall longitudinal striations
        assert (by2 - by1) > (bx2 - bx1)


def test_surface_pitting_and_galling_generation(generator):
    """Verify surface pitting cavities and galling adhesive wear generation."""
    img = generator.generate_pristine_rail(width=640, height=640)

    # Pitting
    img_pit, pit_anns = generator.add_surface_pitting(img, pitting_level=0.30)
    assert len(pit_anns) >= 1
    for ann in pit_anns:
        assert ann["class_name"] == "pitting_galling"
        assert ann["class_id"] == 2
        assert ann["metadata"]["defect_subtype"] == "surface_pitting"

    # Galling
    img_gall, gall_anns = generator.add_galling(img, galling_level=0.35)
    assert len(gall_anns) >= 1
    for ann in gall_anns:
        assert ann["class_name"] == "pitting_galling"
        assert ann["class_id"] == 2
        assert ann["metadata"]["defect_subtype"] == "galling_wear"


def test_composite_rail_generation(generator):
    """Verify multi-defect composite rail image synthesis."""
    img, annotations = generator.generate_rail_image(
        width=640,
        height=640,
        joint_step_mm=0.28,
        friction_wear_level=0.30,
        pitting_level=0.25,
        galling_level=0.20,
        corrosion_level=0.15
    )

    assert img.shape == (640, 640, 3)
    assert len(annotations) >= 4

    classes_present = {ann["class_name"] for ann in annotations}
    assert "joint_misalignment" in classes_present
    assert "rail_surface_wear" in classes_present
    assert "pitting_galling" in classes_present
    assert "rail_corrosion" in classes_present


# ==============================================================================
# 2. YOLO Dataset Export Tests
# ==============================================================================

def test_yolo_dataset_export(tmp_path, generator):
    """Verify YOLO format dataset generation and label serialization."""
    out_dir = tmp_path / "annotated_rail_yolo"
    res = generator.generate_yolo_dataset(
        output_dir=str(out_dir),
        num_train=6,
        num_val=2,
        create_yaml=True
    )

    assert res["train"] == 6
    assert res["val"] == 2
    assert res["annotations"] > 0

    train_imgs = list((out_dir / "train" / "images").glob("*.jpg"))
    train_lbls = list((out_dir / "train" / "labels").glob("*.txt"))
    val_imgs = list((out_dir / "val" / "images").glob("*.jpg"))
    val_lbls = list((out_dir / "val" / "labels").glob("*.txt"))

    assert len(train_imgs) == 6
    assert len(train_lbls) == 6
    assert len(val_imgs) == 2
    assert len(val_lbls) == 2

    # Verify YOLO label file content format: class_id xc yc w h
    for lbl_file in train_lbls + val_lbls:
        with open(lbl_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5
            cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            assert cid in [0, 1, 2, 3]
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < bw <= 1.0
            assert 0.0 < bh <= 1.0

    # Verify rail_defects.yaml creation
    yaml_file = tmp_path / "rail_defects.yaml"
    assert yaml_file.exists()
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    assert "names" in config
    assert config["names"][0] == "rail_surface_wear"
    assert config["names"][1] == "joint_misalignment"


# ==============================================================================
# 3. ISO 7465 Scoring Engine Tests
# ==============================================================================

def test_iso7465_pristine_condition(analyzer):
    """Pristine guide rail blade must score 1.0 with nominal status."""
    res = analyzer.calculate_rail_hi(
        joint_misalignment_mm=0.0,
        surface_scoring_pct=0.0,
        rust_pct=0.0,
        dry_contact_pct=0.0
    )
    assert res.hi_rail_vis == 1.0
    assert not res.is_service_required
    assert res.status == "NOMINAL_HEALTHY"
    assert len(res.fault_reasons) == 0


def test_iso7465_joint_step_tolerance_breach(analyzer):
    """Joint step exceeding ISO 7465 tolerance (0.20mm) must trigger critical realignment."""
    # 0.35mm exceeds 0.20mm tolerance
    res = analyzer.calculate_rail_hi(
        joint_misalignment_mm=0.35,
        surface_scoring_pct=0.02,
        rust_pct=0.01,
        dry_contact_pct=0.0
    )
    assert res.is_service_required
    assert res.status == "CRITICAL_REALIGNMENT_REQUIRED"
    assert res.hi_rail_vis <= 0.25
    assert any("exceeds ISO 7465 tolerance" in r for r in res.fault_reasons)


def test_iso7465_compliant_joint_step(analyzer):
    """Joint step within ISO 7465 tolerance (<= 0.20mm) must not trigger service required."""
    res = analyzer.calculate_rail_hi(
        joint_misalignment_mm=0.12,
        surface_scoring_pct=0.03,
        rust_pct=0.01,
        dry_contact_pct=0.0
    )
    assert not res.is_service_required
    assert res.hi_rail_vis > 0.65


def test_iso7465_surface_wear_and_corrosion_limits(analyzer):
    """Severe surface scoring (>15%) and rust (>10%) must trigger maintenance."""
    # Severe scoring
    res_scoring = analyzer.calculate_rail_hi(
        joint_misalignment_mm=0.05,
        surface_scoring_pct=0.22,
        rust_pct=0.02,
        dry_contact_pct=0.05
    )
    assert res_scoring.is_service_required
    assert any("Severe rail scoring" in r for r in res_scoring.fault_reasons)

    # Severe corrosion
    res_rust = analyzer.calculate_rail_hi(
        joint_misalignment_mm=0.05,
        surface_scoring_pct=0.05,
        rust_pct=0.18,
        dry_contact_pct=0.0
    )
    assert res_rust.is_service_required
    assert any("corrosion" in r for r in res_rust.fault_reasons)


# ==============================================================================
# 4. GuideRailDefectDetector Frame Analysis Tests
# ==============================================================================

def test_detector_analyze_pristine_frame(detector, generator):
    """Analyze a pristine guide rail frame."""
    pristine_img = generator.generate_pristine_rail(width=640, height=640)
    hi_res, detections, annotated = detector.analyze_rail_frame(pristine_img, return_annotated=True)

    assert hi_res is not None
    assert hi_res.hi_rail_vis >= 0.80
    assert not hi_res.is_service_required
    assert annotated.shape == pristine_img.shape
    assert isinstance(detections, list)


def test_detector_analyze_joint_misalignment_frame(detector, generator):
    """Analyze a guide rail frame with excessive joint step misalignment."""
    img, _ = generator.generate_rail_image(
        width=640,
        height=640,
        joint_step_mm=0.38
    )
    hi_res, detections, annotated = detector.analyze_rail_frame(img, return_annotated=True)

    assert hi_res is not None
    # Detection should register joint step
    assert hi_res.joint_misalignment_mm > 0.20
    assert hi_res.is_service_required
    assert hi_res.status == "CRITICAL_REALIGNMENT_REQUIRED"
    assert annotated.shape == img.shape


def test_detector_analyze_wear_and_pitting_frame(detector, generator):
    """Analyze a guide rail frame with friction wear and pitting."""
    img, _ = generator.generate_rail_image(
        width=640,
        height=640,
        friction_wear_level=0.45,
        pitting_level=0.35
    )
    hi_res, detections, annotated = detector.analyze_rail_frame(img, return_annotated=True)

    assert hi_res is not None
    assert len(detections) > 0
    detected_classes = {d["class_name"] for d in detections}
    assert ("rail_surface_wear" in detected_classes) or ("pitting_galling" in detected_classes)


def test_train_detector_rail_yaml_helper(tmp_path):
    """Verify rail dataset yaml config generation helper in train_detector."""
    yaml_out = tmp_path / "rail_defects.yaml"
    res_path = create_rail_dataset_yaml(data_dir=str(tmp_path), output_path=str(yaml_out))
    
    assert Path(res_path).exists()
    with open(res_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["names"][0] == "rail_surface_wear"
    assert cfg["names"][1] == "joint_misalignment"
    assert cfg["names"][2] == "pitting_galling"
    assert cfg["names"][3] == "rail_corrosion"
