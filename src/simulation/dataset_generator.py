"""
ElevatorGuard-AI: Synthetic Dataset & Degradation Generator
Generates realistic multi-strand wire rope defect images (with YOLO annotations)
and run-to-failure vibration/tension telemetry logs based on Miner's cumulative fatigue rule.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2
import pandas as pd


class SyntheticDataGenerator:
    """
    Synthesizes high-fidelity training data for ElevatorGuard-AI:
    - Vision: Steel wire rope surfaces with broken wires, abrasion, and corrosion.
    - Telemetry: Tri-axial acceleration & tension signals over lifecycle degradation.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_rope_image(
        self,
        width: int = 640,
        height: int = 640,
        num_broken_wires: int = 0,
        abrasion_level: float = 0.0,
        corrosion_level: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate a synthetic steel wire rope inspection image with defect annotations.
        
        Returns:
            (image_bgr, list_of_defect_dicts)
        """
        # Create base hoistway dark background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (35, 35, 35) # Dark industrial hoistway

        # Rope geometry: centered vertical cylinder
        rope_width = int(width * 0.45)
        rope_x1 = (width - rope_width) // 2
        rope_x2 = rope_x1 + rope_width

        # Base metallic steel gradient
        rope_roi = np.zeros((height, rope_width, 3), dtype=np.float32)
        for x in range(rope_width):
            # Cylindrical lighting profile
            norm_x = (x / rope_width) * 2.0 - 1.0 # -1 to 1
            intensity = max(0.2, math_val := (1.0 - (norm_x ** 2)) ** 0.5)
            rope_roi[:, x] = [140 * intensity, 140 * intensity, 145 * intensity]

        # Draw helical cable strands (typical 6x19 or 8x19 regular lay)
        num_strands = 6
        lay_pitch = 120.0 # Vertical pixels per strand twist
        y_coords, x_coords = np.mgrid[0:height, 0:rope_width]

        for s in range(num_strands):
            phase = (s / num_strands) * 2.0 * np.pi
            strand_line = np.sin((y_coords / lay_pitch) * 2.0 * np.pi + phase)
            strand_dist = np.abs(norm_x_arr := ((x_coords / rope_width) * 2.0 - 1.0) - (0.6 * strand_line))
            groove_mask = np.exp(-12.0 * (strand_dist ** 2))
            
            # Darken valley grooves between strands
            for c in range(3):
                rope_roi[:, :, c] *= (1.0 - (0.45 * groove_mask))

        # Add micro wire texture noise
        noise = np.random.normal(0, 12, (height, rope_width, 3))
        rope_roi = np.clip(rope_roi + noise, 0, 255).astype(np.uint8)
        img[:, rope_x1:rope_x2] = rope_roi

        annotations = []

        # 1. Add Corrosion / Rust Patches
        if corrosion_level > 0.0:
            num_rust_spots = int(max(1, corrosion_level * 6))
            for _ in range(num_rust_spots):
                rw = int(np.random.uniform(25, 70))
                rh = int(np.random.uniform(35, 110))
                rx = int(np.random.uniform(rope_x1 + 10, rope_x2 - rw - 10))
                ry = int(np.random.uniform(20, height - rh - 20))

                # Rust color (orange-brown oxide: BGR ~ (20, 65, 160))
                rust_patch = np.zeros((rh, rw, 3), dtype=np.uint8)
                rust_patch[:, :, 0] = np.random.randint(15, 35, (rh, rw))
                rust_patch[:, :, 1] = np.random.randint(60, 100, (rh, rw))
                rust_patch[:, :, 2] = np.random.randint(150, 210, (rh, rw))

                # Irregular elliptical alpha mask
                mask = np.zeros((rh, rw), dtype=np.uint8)
                cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), np.random.randint(0, 180), 0, 360, 255, -1)
                mask = cv2.GaussianBlur(mask, (7, 7), 2)
                alpha = (mask / 255.0)[:, :, np.newaxis]

                img[ry:ry+rh, rx:rx+rw] = (img[ry:ry+rh, rx:rx+rw] * (1.0 - alpha) + rust_patch * alpha).astype(np.uint8)

                annotations.append({
                    "class_id": 2, # corrosion_rust
                    "class_name": "corrosion_rust",
                    "bbox": [rx, ry, rx + rw, ry + rh]
                })

        # 2. Add Abrasion / Flat Wear
        if abrasion_level > 0.0:
            num_abrasion_zones = int(max(1, abrasion_level * 5))
            for _ in range(num_abrasion_zones):
                aw = int(np.random.uniform(15, 45))
                ah = int(np.random.uniform(60, 160))
                ax = int(np.random.uniform(rope_x1 + 20, rope_x2 - aw - 20))
                ay = int(np.random.uniform(20, height - ah - 20))

                # Shiny flattened surface (high bright reflection)
                wear_patch = np.ones((ah, aw, 3), dtype=np.uint8) * 225
                wear_noise = np.random.normal(0, 8, (ah, aw, 3)).astype(np.uint8)
                wear_patch = cv2.add(wear_patch, wear_noise)

                mask = np.zeros((ah, aw), dtype=np.uint8)
                cv2.ellipse(mask, (aw // 2, ah // 2), (aw // 2, ah // 2), 0, 0, 360, 255, -1)
                alpha = (mask / 255.0)[:, :, np.newaxis] * 0.75

                img[ay:ay+ah, ax:ax+aw] = (img[ay:ay+ah, ax:ax+aw] * (1.0 - alpha) + wear_patch * alpha).astype(np.uint8)

                annotations.append({
                    "class_id": 1, # abrasion_wear
                    "class_name": "abrasion_wear",
                    "bbox": [ax, ay, ax + aw, ay + ah]
                })

        # 3. Add Broken Wires (Crown Fractures)
        for _ in range(num_broken_wires):
            bw = int(np.random.uniform(18, 38))
            bh = int(np.random.uniform(8, 16))
            bx = int(np.random.uniform(rope_x1 + 15, rope_x2 - bw - 15))
            by = int(np.random.uniform(25, height - bh - 25))

            # Render fractured gap (shadow gap + metallic barb tip)
            cv2.ellipse(img, (bx + bw // 2, by + bh // 2), (bw // 2, bh // 2), np.random.randint(-15, 15), 0, 360, (15, 15, 15), -1)
            # Sharp metallic wire tip
            cv2.line(img, (bx + 2, by + bh // 2), (bx + bw - 4, by + bh // 2), (250, 250, 255), 2)

            annotations.append({
                "class_id": 0, # broken_wire
                "class_name": "broken_wire",
                "bbox": [bx, by, bx + bw, by + bh]
            })

        return img, annotations

    def generate_yolo_dataset(
        self,
        output_dir: str = "data/annotated_yolo",
        num_train: int = 150,
        num_val: int = 40
    ):
        """
        Generate paired images and YOLO txt annotations for model training.
        """
        for split, count in [("train", num_train), ("val", num_val)]:
            img_dir = Path(output_dir) / split / "images"
            lbl_dir = Path(output_dir) / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Dataset Generator] Generating {count} images for {split} split...")

            for i in range(count):
                # Sample random degradation scenarios
                scenario = np.random.choice(["healthy", "minor_wear", "moderate", "severe_discard"])
                if scenario == "healthy":
                    n_breaks = 0
                    abrasion = np.random.uniform(0.0, 0.02)
                    corrosion = 0.0
                elif scenario == "minor_wear":
                    n_breaks = np.random.choice([0, 1])
                    abrasion = np.random.uniform(0.02, 0.08)
                    corrosion = np.random.uniform(0.01, 0.04)
                elif scenario == "moderate":
                    n_breaks = np.random.randint(1, 4)
                    abrasion = np.random.uniform(0.06, 0.14)
                    corrosion = np.random.uniform(0.03, 0.09)
                else: # severe_discard
                    n_breaks = np.random.randint(4, 9)
                    abrasion = np.random.uniform(0.12, 0.28)
                    corrosion = np.random.uniform(0.08, 0.20)

                img, annotations = self.generate_rope_image(
                    width=640,
                    height=640,
                    num_broken_wires=n_breaks,
                    abrasion_level=abrasion,
                    corrosion_level=corrosion
                )

                file_stem = f"rope_sample_{split}_{i:04d}"
                img_path = img_dir / f"{file_stem}.jpg"
                lbl_path = lbl_dir / f"{file_stem}.txt"

                cv2.imwrite(str(img_path), img)

                # Write YOLO format annotations: class_id x_center y_center width height (normalized)
                lines = []
                h, w = img.shape[:2]
                for ann in annotations:
                    cid = ann["class_id"]
                    x1, y1, x2, y2 = ann["bbox"]
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                with open(lbl_path, "w") as f:
                    f.writelines(lines)

        print("[Dataset Generator] Vision dataset generation complete.")

    def generate_lifecycle_telemetry_dataset(
        self,
        output_csv: str = "data/synthetic_degradation/lifecycle_telemetry.csv",
        total_trips: int = 150000,
        sample_steps: int = 3000
    ) -> pd.DataFrame:
        """
        Generate continuous run-to-failure telemetry dataset modeling Miner's cumulative fatigue rule.
        """
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        print(f"[Dataset Generator] Generating lifecycle telemetry ({sample_steps} steps up to {total_trips} trips)...")

        trip_steps = np.linspace(0, total_trips, sample_steps)
        rows = []

        for trip in trip_steps:
            # Lifecycle progress (0.0 = new, 1.0 = nominal failure)
            progress = trip / float(total_trips)
            
            # Non-linear exponential wear accumulation (Miner's fatigue curve)
            fatigue_damage = (progress ** 1.8)
            true_hi = max(0.05, 1.0 - (0.85 * fatigue_damage))

            # Dynamic vibration features increase with cable degradation
            rms_x = 0.08 + (0.35 * fatigue_damage) + np.random.normal(0, 0.015)
            rms_y = 0.07 + (0.30 * fatigue_damage) + np.random.normal(0, 0.015)
            rms_z = 0.12 + (0.55 * fatigue_damage) + np.random.normal(0, 0.02)
            kurtosis_z = 3.0 + (5.5 * (fatigue_damage ** 2)) + np.random.normal(0, 0.15)
            peak_z = 0.40 + (1.60 * fatigue_damage) + np.random.normal(0, 0.05)
            spec_kurtosis = 3.0 + (4.0 * fatigue_damage) + np.random.normal(0, 0.12)

            # Tension imbalance grows as individual strands stretch unevenly
            imbalance = 0.03 + (0.22 * fatigue_damage) + np.random.normal(0, 0.01)
            t_mean = 2500.0 + np.random.normal(0, 20.0)
            t_std = t_mean * imbalance * 0.5

            rows.append({
                "trip_count": int(trip),
                "acc_rms_x": max(0.01, rms_x),
                "acc_rms_y": max(0.01, rms_y),
                "acc_rms_z": max(0.01, rms_z),
                "acc_kurtosis_z": max(1.5, kurtosis_z),
                "acc_peak_z": max(0.1, peak_z),
                "spectral_kurtosis_z": max(1.5, spec_kurtosis),
                "tension_mean": t_mean,
                "tension_std": max(1.0, t_std),
                "tension_imbalance_ratio": max(0.01, imbalance),
                "health_index": true_hi
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"[Dataset Generator] Saved lifecycle telemetry to {output_csv}")
        return df


if __name__ == "__main__":
    gen = SyntheticDataGenerator()
    gen.generate_yolo_dataset(num_train=80, num_val=20)
    gen.generate_lifecycle_telemetry_dataset()


    def generate_guide_rail_image(
        self,
        width: int = 640,
        height: int = 640,
        joint_misalignment_mm: float = 0.05,
        scoring_level: float = 0.0,
        corrosion_level: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate a synthetic elevator T-profile guide rail blade inspection image.
        Simulates:
        - Machined cold-drawn steel rail blade (ISO 7465)
        - Transverse fishplate joint seam with step offset
        - Longitudinal guide shoe scratch score marks
        - Rust oxidation patches
        """
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (28, 30, 36) # Hoistway wall background

        # T-rail machined blade centered
        rail_w = int(width * 0.50)
        rx1 = (width - rail_w) // 2
        rx2 = rx1 + rail_w

        # Steel metallic blade with milling brush texture
        rail_blade = np.zeros((height, rail_w, 3), dtype=np.float32)
        for y in range(height):
            brush = np.random.normal(0, 4.0)
            rail_blade[y, :] = [160 + brush, 165 + brush, 172 + brush]

        # Blade chamfer bevel edges on left and right
        bevel_w = 12
        for bx in range(bevel_w):
            dim_factor = 0.45 + 0.55 * (bx / bevel_w)
            rail_blade[:, bx] *= dim_factor
            rail_blade[:, rail_w - 1 - bx] *= dim_factor

        img[:, rx1:rx2] = np.clip(rail_blade, 0, 255).astype(np.uint8)
        detections = []

        # 1. Guide Rail Fishplate Joint Seam
        joint_y = height // 2
        step_px = int(joint_misalignment_mm * 15.0)
        # Draw joint gap
        cv2.line(img, (rx1, joint_y), (rx2, joint_y), (15, 18, 22), max(2, step_px))
        cv2.line(img, (rx1, joint_y + 2), (rx2, joint_y + 2), (230, 235, 240), 1) # Highlight lip

        if joint_misalignment_mm > 0.15:
            detections.append({
                "class_name": "rail_joint_misalignment",
                "class_id": 0,
                "confidence": 0.91,
                "bbox": [rx1, joint_y - 6, rx2, joint_y + 8]
            })

        # 2. Longitudinal Scratches & Scoring (from Guide Shoes)
        if scoring_level > 0.03:
            num_scratches = int(scoring_level * 25)
            for _ in range(num_scratches):
                sx = np.random.randint(rx1 + bevel_w + 10, rx2 - bevel_w - 10)
                sy1 = np.random.randint(10, height - 100)
                slen = np.random.randint(40, 200)
                sy2 = min(height - 10, sy1 + slen)
                scratch_col = (110, 115, 120) if np.random.rand() > 0.5 else (220, 225, 230)
                cv2.line(img, (sx, sy1), (sx, sy2), scratch_col, 1)

            detections.append({
                "class_name": "scratches_scoring_wear",
                "class_id": 1,
                "confidence": round(min(0.95, 0.70 + scoring_level), 2),
                "bbox": [rx1 + 15, 50, rx2 - 15, height - 50]
            })

        # 3. Rust / Oxidation Patches on Rail Blade
        if corrosion_level > 0.02:
            num_rust = int(corrosion_level * 10)
            for _ in range(num_rust):
                cx = np.random.randint(rx1 + 15, rx2 - 15)
                cy = np.random.randint(20, height - 20)
                rw = np.random.randint(12, 35)
                rh = np.random.randint(10, 30)
                cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, (18, 75, 160), -1)
                detections.append({
                    "class_name": "corrosion_rust",
                    "class_id": 2,
                    "confidence": round(min(0.93, 0.65 + corrosion_level), 2),
                    "bbox": [max(0, cx - rw), max(0, cy - rh), min(width, cx + rw), min(height, cy + rh)]
                })

        return img, detections
