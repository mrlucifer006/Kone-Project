"""
ElevatorGuard-AI: Computer Vision NDT Defect Analyzer
Performs real-time detection and quantification of elevator wire rope defects:
- Broken wires (point/local strand breaks)
- Surface abrasion / flattening
- Corrosion / oxidation pitting
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from src.vision.visual_hi import ISO4309VisualAnalyzer, VisualInspectionResult


class CableDefectDetector:
    """
    NDT Vision Processor for elevator hoistway cable inspection cameras.
    Combines YOLO deep learning inference with morphological verification.
    """

    CLASS_NAMES = ["broken_wire", "abrasion_wear", "corrosion_rust"]
    CLASS_COLORS = {
        "broken_wire": (0, 0, 255),      # Red (BGR)
        "abrasion_wear": (0, 165, 255),   # Orange (BGR)
        "corrosion_rust": (0, 215, 255)   # Gold/Yellow (BGR)
    }

    def __init__(
        self,
        model_weights: Optional[str] = "models/yolo_cable_detector.pt",
        confidence_thresh: float = 0.40,
        iou_thresh: float = 0.45,
        visual_analyzer: Optional[ISO4309VisualAnalyzer] = None
    ):
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.model_weights = model_weights
        self.visual_analyzer = visual_analyzer or ISO4309VisualAnalyzer()
        self.model = None

        if ULTRALYTICS_AVAILABLE and model_weights and os.path.exists(model_weights):
            try:
                self.model = YOLO(model_weights)
            except Exception as e:
                print(f"[CableDefectDetector] Warning: Could not load YOLO model from {model_weights}: {e}")
                self.model = None

    def analyze_frame(
        self,
        image: np.ndarray,
        return_annotated: bool = True
    ) -> Tuple[VisualInspectionResult, List[Dict[str, Any]], np.ndarray]:
        """
        Analyze a cable inspection frame, identify defects, and calculate ISO 4309 HI_vis.
        
        Args:
            image: BGR numpy image of the wire rope.
            return_annotated: If True, draws annotations on a copy of the frame.
            
        Returns:
            (VisualInspectionResult, list of detections, annotated_image)
        """
        if image is None or image.size == 0:
            empty_result = self.visual_analyzer.calculate_hi(0, 0.0, 0.0)
            return empty_result, [], image

        h, w = image.shape[:2]
        detections = []

        if self.model is not None:
            detections = self._run_yolo_inference(image)
        else:
            detections = self._run_heuristic_vision_inference(image)

        # Aggregate metrics from detections
        wire_break_count = 0
        total_abrasion_pixels = 0
        total_corrosion_pixels = 0
        cable_roi_area = max(1.0, float(w * h * 0.75)) # Estimated rope area in center

        for det in detections:
            cls = det["class_name"]
            box = det["bbox"] # [x1, y1, x2, y2]
            box_area = max(0, (box[2] - box[0]) * (box[3] - box[1]))

            if cls == "broken_wire":
                wire_break_count += 1
            elif cls == "abrasion_wear":
                total_abrasion_pixels += box_area
            elif cls == "corrosion_rust":
                total_corrosion_pixels += box_area

        # Compute percentage wear & corrosion relative to rope surface
        wear_pct = min(0.40, total_abrasion_pixels / cable_roi_area)
        corrosion_pct = min(0.50, total_corrosion_pixels / cable_roi_area)

        # Compute ISO 4309 Visual Health Index
        hi_result = self.visual_analyzer.calculate_hi(
            wire_breaks=wire_break_count,
            wear_pct=wear_pct,
            corrosion_pct=corrosion_pct
        )

        annotated_image = image.copy()
        if return_annotated:
            annotated_image = self.draw_inspection_hud(annotated_image, detections, hi_result)

        return hi_result, detections, annotated_image

    def _run_yolo_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO deep learning inference."""
        detections = []
        try:
            results = self.model.predict(
                image,
                conf=self.confidence_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"

                    detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 3),
                        "bbox": xyxy
                    })
        except Exception as e:
            print(f"[CableDefectDetector] Inference error: {e}")
            detections = self._run_heuristic_vision_inference(image)
        return detections

    def _run_heuristic_vision_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        High-precision edge vision analyzer for cable surfaces.
        Segments rust/corrosion (HSV color), broken crown wires (high frequency gradient spikes),
        and abrasion (brightness variance & edge thinning).
        """
        detections = []
        h, w = image.shape[:2]

        # 1. Corrosion Detection via Color Segmentation (Rust tones: H: 8-28, S: 90-255, V: 50-230)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rust_lower = np.array([8, 80, 50], dtype=np.uint8)
        rust_upper = np.array([28, 255, 230], dtype=np.uint8)
        rust_mask = cv2.inRange(hsv, rust_lower, rust_upper)
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in rust_contours:
            area = cv2.contourArea(cnt)
            if area > 120: # Minimum significant defect patch
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append({
                    "class_name": "corrosion_rust",
                    "class_id": 2,
                    "confidence": round(min(0.95, 0.60 + (area / 2000.0)), 2),
                    "bbox": [x, y, x + bw, y + bh]
                })

        # 2. Broken Wire Detection (High contrast localized strand fractures / bright metallic sharp ends)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(enhanced, 120, 240)
        
        # Look for horizontal/diagonal breaks across vertical rope strands
        sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        break_response = np.abs(sobely) - 0.5 * np.abs(sobelx)
        break_response = np.clip(break_response, 0, 255).astype(np.uint8)
        _, break_thresh = cv2.threshold(break_response, 130, 255, cv2.THRESH_BINARY)
        break_contours, _ = cv2.findContours(break_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in break_contours:
            area = cv2.contourArea(cnt)
            if 30 < area < 800:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                if aspect > 1.2: # Broken crown wire typically has horizontal aspect across vertical strand
                    detections.append({
                        "class_name": "broken_wire",
                        "class_id": 0,
                        "confidence": 0.88,
                        "bbox": [x, y, x + bw, y + bh]
                    })

        # 3. Surface Abrasion (Polished mirror wear on outer strand crowns)
        _, bright_sheen = cv2.threshold(enhanced, 230, 255, cv2.THRESH_BINARY)
        sheen_contours, _ = cv2.findContours(bright_sheen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in sheen_contours:
            area = cv2.contourArea(cnt)
            if area > 250:
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append({
                    "class_name": "abrasion_wear",
                    "class_id": 1,
                    "confidence": 0.78,
                    "bbox": [x, y, x + bw, y + bh]
                })

        return detections

    def draw_inspection_hud(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        hi_result: VisualInspectionResult
    ) -> np.ndarray:
        """Render defect boxes and ISO 4309 diagnostic HUD overlay."""
        out = image.copy()
        h, w = out.shape[:2]

        # Draw defect bounding boxes
        for det in detections:
            cls = det["class_name"]
            conf = det["confidence"]
            box = det["bbox"]
            color = self.CLASS_COLORS.get(cls, (0, 255, 0))

            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{cls} {int(conf * 100)}%"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(
                out,
                (box[0], box[1] - t_size[1] - 4),
                (box[0] + t_size[0] + 4, box[1]),
                color,
                -1
            )
            cv2.putText(
                out,
                label,
                (box[0] + 2, box[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Draw HUD Header Banner
        hud_h = 75
        hud_bg = np.zeros((hud_h, w, 3), dtype=np.uint8)
        
        # Color code based on status
        if hi_result.is_discard_required:
            hud_color = (0, 0, 180) # Red
            status_text = "ISO 4309 CRITICAL DISCARD: REPLACE ROPE"
        elif hi_result.hi_vis < 0.45:
            hud_color = (0, 140, 255) # Orange
            status_text = "WARNING: HIGH WEAR DETECTED"
        elif hi_result.hi_vis < 0.70:
            hud_color = (0, 215, 255) # Yellow
            status_text = "MONITOR: MODERATE STRAND WEAR"
        else:
            hud_color = (0, 150, 0) # Green
            status_text = "NOMINAL: ROPE HEALTHY"

        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.rectangle(hud_bg, (0, 0), (8, hud_h), hud_color, -1)

        cv2.putText(
            hud_bg,
            f"ElevatorGuard-AI NDT Cam | HI_vis: {hi_result.hi_vis:.3f} | {status_text}",
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        metrics_text = (
            f"Broken Wires: {hi_result.wire_breaks}/6 | "
            f"Wear: {hi_result.wear_pct * 100:.1f}% | "
            f"Corrosion: {hi_result.corrosion_pct * 100:.1f}% | "
            f"Detections: {len(detections)}"
        )
        cv2.putText(
            hud_bg,
            metrics_text,
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        # Alpha blend HUD onto top of image
        out[:hud_h, :] = cv2.addWeighted(out[:hud_h, :], 0.2, hud_bg, 0.8, 0)
        return out


# ==============================================================================
# ISO 7465: Elevator Guide Rail Vision NDT Defect Analyzer
# ==============================================================================

from src.vision.visual_hi import ISO7465GuideRailVisualAnalyzer, GuideRailInspectionResult

class GuideRailDefectDetector:
    """
    NDT Edge Vision Processor for Elevator Guide Rail Inspection Cameras.
    Evaluates:
    - Guide rail joint step / fishplate gap misalignment (ISO 7465 limit <= 0.20 mm)
    - Longitudinal surface scratches & abrasive scoring from guide shoes
    - Surface oxidation / rust pitting on machined guide blade
    - Dry contact wear / lack of guide rail lubrication
    """

    CLASS_NAMES = [
        "rail_surface_wear",
        "joint_misalignment",
        "pitting_galling",
        "rail_corrosion"
    ]

    CLASS_COLORS = {
        "joint_misalignment": (0, 0, 255),          # Red
        "rail_surface_wear": (0, 165, 255),         # Orange
        "rail_corrosion": (0, 215, 255),            # Gold/Yellow
        "pitting_galling": (200, 100, 255),         # Magenta
        # Backward compatibility aliases
        "rail_joint_misalignment": (0, 0, 255),
        "joint_gap_step": (0, 0, 255),
        "scratches_scoring_wear": (0, 165, 255),
        "friction_wear": (0, 165, 255),
        "corrosion_rust": (0, 215, 255),
        "pitting_galling": (200, 100, 255),
        "surface_pitting": (200, 100, 255),
        "galling_wear": (200, 100, 255),
        "dry_contact_wear": (200, 100, 255)
    }

    def __init__(
        self,
        model_weights: Optional[str] = "models/yolo_rail_detector.pt",
        confidence_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        visual_analyzer: Optional[ISO7465GuideRailVisualAnalyzer] = None
    ):
        self.confidence_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.model_weights = model_weights
        self.visual_analyzer = visual_analyzer or ISO7465GuideRailVisualAnalyzer()
        self.model = None

        if ULTRALYTICS_AVAILABLE and model_weights and os.path.exists(model_weights):
            try:
                self.model = YOLO(model_weights)
            except Exception as e:
                print(f"[GuideRailDefectDetector] Warning: Could not load YOLO model from {model_weights}: {e}")
                self.model = None

    def analyze_rail_frame(
        self,
        image: np.ndarray,
        return_annotated: bool = True
    ) -> Tuple[GuideRailInspectionResult, List[Dict[str, Any]], np.ndarray]:
        """
        Analyze a guide rail camera inspection frame and evaluate ISO 7465 compliance.
        Classifies defects into:
          * 'rail_surface_wear' (guide shoe friction wear / scoring)
          * 'joint_misalignment' (joint gap steps 0.1mm - 0.5mm misalignment)
          * 'pitting_galling' (surface pitting & galling wear)
          * 'rail_corrosion' (surface oxidation / rust)
        Returns:
          (GuideRailInspectionResult, detections, annotated_frame)
        """
        if image is None or image.size == 0:
            empty_res = self.visual_analyzer.calculate_rail_hi(0.0, 0.0, 0.0, 0.0)
            return empty_res, [], image

        h, w = image.shape[:2]
        joint_misalignment_mm = 0.0

        if self.model is not None:
            detections = self._run_yolo_inference(image)
        else:
            detections, joint_misalignment_mm = self._run_heuristic_vision_inference(image)

        # Aggregate metrics from detections
        total_scoring_pixels = 0
        total_rust_pixels = 0
        total_pitting_pixels = 0

        for det in detections:
            cls = det["class_name"]
            box = det["bbox"]
            box_area = max(0, (box[2] - box[0]) * (box[3] - box[1]))

            if cls in ["joint_misalignment", "rail_joint_misalignment", "joint_gap_step"]:
                step = det.get("joint_step_mm", 0.28)
                joint_misalignment_mm = max(joint_misalignment_mm, step)
            elif cls in ["rail_surface_wear", "friction_wear", "scratches_scoring_wear"]:
                total_scoring_pixels += box_area
            elif cls in ["pitting_galling", "surface_pitting", "galling_wear", "dry_contact_wear"]:
                total_pitting_pixels += box_area
            elif cls in ["rail_corrosion", "corrosion_rust"]:
                total_rust_pixels += box_area

        # Estimate surface area proportions relative to rail blade
        rail_blade_area = max(1.0, float(w * h * 0.50))
        surface_scoring_pct = min(0.35, total_scoring_pixels / (rail_blade_area * 1.5))
        rust_pct = min(0.40, total_rust_pixels / (rail_blade_area * 1.5))
        dry_contact_pct = min(0.30, total_pitting_pixels / (rail_blade_area * 1.5))

        # Calculate ISO 7465 Guide Rail Visual Health Index
        hi_result = self.visual_analyzer.calculate_rail_hi(
            joint_misalignment_mm=joint_misalignment_mm,
            surface_scoring_pct=surface_scoring_pct,
            rust_pct=rust_pct,
            dry_contact_pct=dry_contact_pct
        )

        annotated_image = image.copy()
        if return_annotated:
            annotated_image = self.draw_rail_hud(annotated_image, detections, hi_result)

        return hi_result, detections, annotated_image

    def _run_yolo_inference(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO deep learning inference for guide rail defects."""
        detections = []
        try:
            results = self.model.predict(
                image,
                conf=self.confidence_thresh,
                iou=self.iou_thresh,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"

                    detections.append({
                        "class_name": cls_name,
                        "class_id": cls_id,
                        "confidence": round(conf, 3),
                        "bbox": xyxy
                    })
        except Exception as e:
            print(f"[GuideRailDefectDetector] Inference error: {e}")
            detections, _ = self._run_heuristic_vision_inference(image)
        return detections

    def _run_heuristic_vision_inference(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        High-precision edge vision analyzer for guide rail surfaces.
        Segments corrosion, friction wear / scratches, pitting/galling, and joint gap steps.
        """
        h, w = image.shape[:2]
        detections = []
        joint_misalignment_mm = 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Automatically identify guide rail blade boundaries
        col_means = np.mean(gray, axis=0)
        rail_cols = np.where(col_means > 70)[0]
        if len(rail_cols) > 50:
            rx1 = int(rail_cols[0])
            rx2 = int(rail_cols[-1])
        else:
            rx1 = int(w * 0.25)
            rx2 = int(w * 0.75)

        rail_w = max(40, rx2 - rx1)
        # Active blade contact zone (excluding outer chamfer bevels and mounting shadows)
        blade_x1 = rx1 + int(rail_w * 0.08)
        blade_x2 = rx2 - int(rail_w * 0.08)

        blade_mask = np.zeros((h, w), dtype=np.uint8)
        blade_mask[:, blade_x1:blade_x2] = 255

        # 1. Detect Guide Rail Joint Misalignment / Transverse Seams (Sobel Y Row Energy & Step Measurement)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y_abs = np.uint8(np.absolute(sobel_y))

        # Check horizontal row gradient energy across blade center
        blade_margin = int(rail_w * 0.12)
        x_start = rx1 + blade_margin
        x_end = rx2 - blade_margin
        seam_y = None

        if x_end > x_start:
            row_seam_strength = np.mean(sobel_y_abs[:, x_start:x_end], axis=1)
            # Mask out image top/bottom borders
            row_seam_strength[:35] = 0
            row_seam_strength[-35:] = 0

            peak_y = int(np.argmax(row_seam_strength))
            peak_val = float(row_seam_strength[peak_y])

            # A joint seam produces a strong horizontal gradient spike (> 28) spanning across blade
            if peak_val > 28.0:
                span_pixels = np.sum(sobel_y_abs[peak_y, x_start:x_end] > 22)
                span_ratio = span_pixels / float(max(1, x_end - x_start))

                if span_ratio > 0.35:
                    seam_y = peak_y
                    # Measure lateral edge displacement above vs below seam
                    y_above = max(0, peak_y - 25)
                    y_below = min(h - 1, peak_y + 25)
                    edge_above = np.where(gray[y_above, :rx1 + rail_w // 2] > 65)[0]
                    edge_below = np.where(gray[y_below, :rx1 + rail_w // 2] > 65)[0]
                    shift_mm = 0.0
                    if len(edge_above) > 0 and len(edge_below) > 0:
                        shift_px = abs(int(edge_above[0]) - int(edge_below[0]))
                        shift_mm = round(shift_px * 0.08, 2)

                    # Intensity contrast across the seam
                    upper_roi = gray[max(0, peak_y - 12):max(1, peak_y - 2), x_start:x_end]
                    lower_roi = gray[min(h - 1, peak_y + 2):min(h, peak_y + 12), x_start:x_end]
                    contrast = abs(float(np.mean(upper_roi)) - float(np.mean(lower_roi)))

                    estimated_step_mm = max(
                        shift_mm,
                        round(0.12 + (peak_val / 125.0) * 0.35 + (contrast / 60.0) * 0.15, 2)
                    )
                    estimated_step_mm = min(0.65, max(0.10, estimated_step_mm))
                    joint_misalignment_mm = estimated_step_mm

                    detections.append({
                        "class_name": "joint_misalignment",
                        "class_id": 1,
                        "confidence": round(min(0.96, 0.75 + (peak_val / 180.0)), 2),
                        "joint_step_mm": estimated_step_mm,
                        "bbox": [rx1, max(0, peak_y - 14), rx2, min(h, peak_y + 14)]
                    })

        # Exclude joint seam band from surface wear analysis to avoid double counting
        if seam_y is not None:
            blade_mask[max(0, seam_y - 16):min(h, seam_y + 16), :] = 0

        # 2. Detect Rust / Corrosion on Machined Steel Rail Face (HSV) -> 'rail_corrosion'
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rust_lower = np.array([8, 65, 45], dtype=np.uint8)
        rust_upper = np.array([26, 255, 235], dtype=np.uint8)
        rust_mask = cv2.inRange(hsv, rust_lower, rust_upper)
        rust_mask = cv2.bitwise_and(rust_mask, blade_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        rust_mask = cv2.morphologyEx(rust_mask, cv2.MORPH_OPEN, kernel)
        rust_contours, _ = cv2.findContours(rust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in rust_contours:
            area = cv2.contourArea(cnt)
            if area > 120:
                x, y, bw, bh = cv2.boundingRect(cnt)
                detections.append({
                    "class_name": "rail_corrosion",
                    "class_id": 3,
                    "confidence": round(min(0.96, 0.65 + (area / 3000.0)), 2),
                    "bbox": [x, y, x + bw, y + bh]
                })

        # 3. Detect Longitudinal Scratches & Surface Scoring (from Guide Shoes) -> 'rail_surface_wear'
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_x_abs = np.uint8(np.absolute(sobel_x))
        # Deep scoring grooves have significantly higher gradient (>68) than normal grinding lines (<45)
        _, groove_thresh = cv2.threshold(sobel_x_abs, 68, 255, cv2.THRESH_BINARY)
        # Guide shoe burnishing sheen (bright polished track > 195)
        _, sheen_thresh = cv2.threshold(gray, 195, 255, cv2.THRESH_BINARY)
        wear_raw = cv2.bitwise_or(groove_thresh, sheen_thresh)
        wear_in_blade = cv2.bitwise_and(wear_raw, blade_mask)

        # Connect longitudinal striations into wear bands
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
        wear_clean = cv2.morphologyEx(wear_in_blade, cv2.MORPH_OPEN, v_kernel)
        wear_clean = cv2.morphologyEx(wear_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 25)))
        scratch_contours, _ = cv2.findContours(wear_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in scratch_contours:
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh > 40 and area > 180 and (bh / max(1, bw)) > 1.2:
                detections.append({
                    "class_name": "rail_surface_wear",
                    "class_id": 0,
                    "confidence": round(min(0.94, 0.62 + (bh / float(h))), 2),
                    "bbox": [x, y, x + bw, y + bh]
                })

        # 4. Detect Pitting / Galling (High local roughness, localized depressions & spalls) -> 'pitting_galling'
        dark_core_mask = (gray < 65) & (blade_mask > 0)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.uint8(np.clip(np.absolute(laplacian), 0, 255))
        _, rough_thresh = cv2.threshold(lap_abs, 95, 255, cv2.THRESH_BINARY)
        pitting_raw = cv2.bitwise_or((dark_core_mask.astype(np.uint8) * 255), rough_thresh)
        pitting_in_blade = cv2.bitwise_and(pitting_raw, blade_mask)

        pit_clean = cv2.morphologyEx(pitting_in_blade, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        pit_contours, _ = cv2.findContours(pit_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in pit_contours:
            area = cv2.contourArea(cnt)
            if 80 < area < 2500:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / max(1, bh)
                if 0.35 < aspect < 2.8:
                    detections.append({
                        "class_name": "pitting_galling",
                        "class_id": 2,
                        "confidence": round(min(0.92, 0.62 + (area / 1500.0)), 2),
                        "bbox": [x, y, x + bw, y + bh]
                    })

        return detections, joint_misalignment_mm

    def draw_rail_hud(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        hi_result: GuideRailInspectionResult
    ) -> np.ndarray:
        """Render guide rail defect bounding boxes and ISO 7465 HUD."""
        out = image.copy()
        h, w = out.shape[:2]

        # Draw bounding boxes
        for det in detections:
            cls = det["class_name"]
            conf = det["confidence"]
            box = det["bbox"]
            color = self.CLASS_COLORS.get(cls, (0, 255, 0))

            cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{cls} {int(conf * 100)}%"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(
                out,
                (box[0], box[1] - t_size[1] - 4),
                (box[0] + t_size[0] + 4, box[1]),
                color,
                -1
            )
            cv2.putText(
                out,
                label,
                (box[0] + 2, box[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Draw HUD Banner
        hud_h = 75
        hud_bg = np.zeros((hud_h, w, 3), dtype=np.uint8)

        if hi_result.is_service_required:
            hud_color = (0, 0, 180) # Red
            status_text = "ISO 7465 BREACH: REALIGNMENT / RESURFACING REQUIRED"
        elif hi_result.hi_rail_vis < 0.50:
            hud_color = (0, 140, 255) # Orange
            status_text = "WARNING: HIGH RAIL WEAR / JOINT STEP"
        elif hi_result.hi_rail_vis < 0.75:
            hud_color = (0, 215, 255) # Yellow
            status_text = "MONITOR: MODERATE RAIL SCORING"
        else:
            hud_color = (0, 150, 0) # Green
            status_text = "NOMINAL: GUIDE RAIL HEALTHY (ISO 7465)"

        cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.rectangle(hud_bg, (0, 0), (8, hud_h), hud_color, -1)

        cv2.putText(
            hud_bg,
            f"GUIDE RAIL NDT CAMERA | HI_rail: {hi_result.hi_rail_vis:.3f} | {status_text}",
            (16, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        metrics_text = (
            f"Joint Step: {hi_result.joint_misalignment_mm:.2f}mm (Max: 0.20mm) | "
            f"Scoring: {hi_result.surface_scoring_pct * 100:.1f}% | "
            f"Rust: {hi_result.rust_pct * 100:.1f}% | "
            f"Defects: {len(detections)}"
        )
        cv2.putText(
            hud_bg,
            metrics_text,
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        out[:hud_h, :] = cv2.addWeighted(out[:hud_h, :], 0.2, hud_bg, 0.8, 0)
        return out
