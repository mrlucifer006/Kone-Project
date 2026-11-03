"""
ElevatorGuard-AI: Synthetic Guide Rail Dataset Generator
Generates high-fidelity synthetic guide rail inspection images and YOLO annotations
compliant with ISO 7465 / ISO 8100-34 specifications:
- Pristine machined surface lines (grinding striations & milling feed marks)
- Guide shoe friction wear marks (longitudinal scoring & burnishing)
- Joint gap steps (0.1mm - 0.5mm misalignment across fishplate seams)
- Surface pitting (localized cavitation/oxidation craters) and galling (adhesive metal tearing)
- Rail surface corrosion (reddish-brown oxidation patches)
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import cv2
import yaml


# Standard defect class dictionary for guide rail inspection (ISO 7465)
RAIL_DEFECT_CLASSES = {
    0: "rail_surface_wear",  # Guide shoe friction wear marks / longitudinal scoring
    1: "joint_misalignment", # Joint gap steps (0.1mm - 0.5mm misalignment)
    2: "pitting_galling",    # Surface pitting / galling adhesive metal tearing
    3: "rail_corrosion"      # Surface oxidation / rust patches
}

# Reverse class lookup
CLASS_TO_ID = {name: cid for cid, name in RAIL_DEFECT_CLASSES.items()}
# Aliases for cross-compatibility
CLASS_TO_ID.update({
    "friction_wear": 0,
    "scratches_scoring_wear": 0,
    "rail_joint_misalignment": 1,
    "joint_gap_step": 1,
    "surface_pitting": 2,
    "galling_wear": 2,
    "dry_contact_wear": 2,
    "corrosion_rust": 3
})


class RailDatasetGenerator:
    """
    Synthesizes high-resolution elevator guide rail images with realistic
    machining textures, lighting profiles, and defects according to ISO 7465.
    """

    def __init__(self, seed: Optional[int] = 42, mm_per_pixel: float = 0.08):
        """
        Args:
            seed: Random seed for deterministic generation.
            mm_per_pixel: Spatial calibration scale (default: 0.08 mm per pixel).
        """
        if seed is not None:
            np.random.seed(seed)
        self.mm_per_pixel = mm_per_pixel

    def generate_pristine_rail(
        self,
        width: int = 640,
        height: int = 640,
        rail_width_ratio: float = 0.50
    ) -> np.ndarray:
        """
        Generate a pristine machined guide rail running surface.
        Features:
        - Dark hoistway background with subtle texture
        - Cold-drawn / machined steel blade with longitudinal grinding striations
        - Planar specular reflection across the blade face
        - Edge chamfer bevels with lighting contrast
        """
        # Hoistway shaft background (dark industrial matte surface)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = (32, 34, 38)
        bg_noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)

        # Rail blade geometry (centered column)
        rail_w = int(width * rail_width_ratio)
        rx1 = (width - rail_w) // 2
        rx2 = rx1 + rail_w

        # Base metallic steel profile (gray with slight cool tint)
        blade = np.zeros((height, rail_w, 3), dtype=np.float32)
        
        # Specular curvature across blade face (slight parabolic highlight)
        norm_x = (np.arange(rail_w) - rail_w / 2.0) / (rail_w / 2.0) # -1.0 to 1.0
        lighting = 1.0 - 0.12 * (norm_x ** 2)

        # Base steel tone
        base_steel = np.array([152.0, 156.0, 160.0], dtype=np.float32) # BGR
        blade[:, :] = base_steel * lighting[np.newaxis, :, np.newaxis]

        # 1. Longitudinal grinding striations (machined lines parallel to guide rail axis)
        striations_1d = np.random.normal(0, 1.0, rail_w)
        kernel_1d = np.array([0.15, 0.70, 0.15])
        striations_1d = np.convolve(striations_1d, kernel_1d, mode='same')
        striation_amplitude = 10.0 # ±10 intensity
        striation_map = np.tile(striations_1d * striation_amplitude, (height, 1))

        # Subtle vertical waviness / drift
        y_indices = np.arange(height)[:, np.newaxis]
        drift = np.sin((y_indices / 80.0) * np.pi) * 2.0
        striation_map += drift

        for c in range(3):
            blade[:, :, c] += striation_map

        # 2. Transverse milling feed marks (faint periodic horizontal tool chatter waves)
        feed_wave = np.sin((y_indices / 24.0) * 2.0 * np.pi) * 3.5
        for c in range(3):
            blade[:, :, c] += feed_wave

        # 3. Micro-roughness surface noise (simulating Ra 1.6 - 3.2 um surface finish)
        surface_roughness = np.random.normal(0, 5.0, (height, rail_w, 3))
        blade = np.clip(blade + surface_roughness, 0, 255).astype(np.uint8)

        # 4. Machined chamfers / side bevel borders (left and right edges of blade)
        bevel_w = max(4, int(rail_w * 0.04))
        for i in range(bevel_w):
            factor = 0.55 + 0.45 * (i / float(bevel_w))
            blade[:, i] = (blade[:, i].astype(np.float32) * factor).astype(np.uint8)
        for i in range(bevel_w):
            factor = 1.0 - 0.35 * (i / float(bevel_w))
            blade[:, -(i + 1)] = (blade[:, -(i + 1)].astype(np.float32) * factor).astype(np.uint8)

        img[:, rx1:rx2] = blade

        # Hoistway mounting shadows on borders of the rail
        cv2.line(img, (rx1 - 1, 0), (rx1 - 1, height), (15, 17, 20), 2)
        cv2.line(img, (rx2 + 1, 0), (rx2 + 1, height), (18, 20, 24), 2)

        return img

    def add_joint_gap_step(
        self,
        img: np.ndarray,
        joint_step_mm: float = 0.25,
        y_pos: Optional[int] = None,
        rail_width_ratio: float = 0.50
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Add a guide rail joint gap step (fishplate seam misalignment).
        
        Args:
            img: Guide rail image.
            joint_step_mm: Misalignment step in mm (0.1mm - 0.5mm).
            y_pos: Vertical position of the joint seam (default: center).
            rail_width_ratio: Width ratio of the rail blade.
            
        Returns:
            (modified_image, annotation_dict)
        """
        if joint_step_mm <= 0.0:
            return img, None

        out = img.copy()
        h, w = out.shape[:2]
        rail_w = int(w * rail_width_ratio)
        rx1 = (w - rail_w) // 2
        rx2 = rx1 + rail_w

        if y_pos is None:
            y_pos = h // 2 + int(np.random.uniform(-40, 40))

        y_pos = max(50, min(h - 50, y_pos))

        # Convert mm to pixel displacement
        step_px = max(1, int(round(joint_step_mm / self.mm_per_pixel)))
        step_dir = np.random.choice([-1, 1])
        shift_px = step_px * step_dir

        # Shift lower rail segment laterally to simulate joint step misalignment
        lower_segment = out[y_pos:h, :].copy()
        shifted_lower = np.zeros_like(lower_segment)
        shifted_lower[:, :] = (32, 34, 38)
        
        if shift_px > 0:
            shifted_lower[:, shift_px:] = lower_segment[:, :w - shift_px]
        elif shift_px < 0:
            shifted_lower[:, :w + shift_px] = lower_segment[:, -shift_px:]
        else:
            shifted_lower = lower_segment

        out[y_pos:h, :] = shifted_lower

        # Transverse Joint Gap Seam (Air gap between rail ends)
        gap_thickness = max(2, int(round(0.8 / self.mm_per_pixel)))
        y1_gap = max(0, y_pos - gap_thickness // 2)
        y2_gap = min(h, y_pos + gap_thickness // 2)

        # Draw dark seam gap
        cv2.rectangle(out, (min(rx1, rx1 + shift_px), y1_gap), (max(rx2, rx2 + shift_px), y2_gap), (12, 14, 16), -1)

        # Draw physical step shadow and light-catching edge
        if joint_step_mm > 0.18:
            highlight_color = (245, 245, 250)
            cv2.line(out, (rx1, y1_gap - 1), (rx2, y1_gap - 1), highlight_color, 1)
            shadow_color = (18, 18, 22)
            cv2.line(out, (rx1 + shift_px, y2_gap + 1), (rx2 + shift_px, y2_gap + 1), shadow_color, 2)
        else:
            cv2.line(out, (rx1, y1_gap - 1), (rx2, y1_gap - 1), (210, 215, 220), 1)

        cv2.circle(out, (rx1, y_pos), 3, (40, 42, 48), -1)
        cv2.circle(out, (rx2, y_pos), 3, (40, 42, 48), -1)

        bx1 = max(0, min(rx1, rx1 + shift_px) - 8)
        bx2 = min(w, max(rx2, rx2 + shift_px) + 8)
        by1 = max(0, y_pos - max(12, step_px + 8))
        by2 = min(h, y_pos + max(12, step_px + 8))

        annotation = {
            "class_id": 1,
            "class_name": "joint_misalignment",
            "bbox": [bx1, by1, bx2, by2],
            "joint_step_mm": round(joint_step_mm, 3),
            "metadata": {
                "joint_step_mm": round(joint_step_mm, 3),
                "step_px": step_px,
                "y_pos": y_pos
            }
        }

        return out, annotation

    def add_friction_wear(
        self,
        img: np.ndarray,
        wear_level: float = 0.35,
        rail_width_ratio: float = 0.50
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Add guide shoe friction wear marks.
        Produces:
        - Longitudinal burnishing sheen (bright polished track)
        - High-contrast abrasive scoring grooves along guide shoe sliding direction
        """
        if wear_level <= 0.0:
            return img, []

        out = img.copy()
        h, w = out.shape[:2]
        rail_w = int(w * rail_width_ratio)
        rx1 = (w - rail_w) // 2
        rx2 = rx1 + rail_w

        annotations = []
        num_wear_bands = int(max(1, round(wear_level * 5)))

        for _ in range(num_wear_bands):
            band_w = int(np.random.uniform(12, max(20, rail_w * 0.22)))
            band_h = int(np.random.uniform(h * 0.35, h * 0.90))
            bx = int(np.random.uniform(rx1 + 10, rx2 - band_w - 10))
            by = int(np.random.uniform(15, h - band_h - 15))

            # 1. Polished burnishing layer (bright specular wear patch)
            wear_overlay = out[by:by+band_h, bx:bx+band_w].astype(np.float32)
            sheen_boost = np.random.uniform(30.0, 55.0)
            wear_overlay += sheen_boost

            # 2. Longitudinal scoring grooves (vertical scratches within the wear band)
            num_grooves = np.random.randint(3, 8)
            for _ in range(num_grooves):
                gx = np.random.randint(0, band_w)
                groove_w = np.random.choice([1, 2])
                groove_intensity = np.random.uniform(-50.0, -25.0) if np.random.rand() > 0.3 else np.random.uniform(40.0, 70.0)
                gx_end = min(band_w, gx + groove_w)
                wear_overlay[:, gx:gx_end] += groove_intensity

            # Soft vertical mask so ends blend naturally
            mask = np.ones((band_h, band_w), dtype=np.float32)
            fade_h = min(20, band_h // 4)
            for i in range(fade_h):
                ramp = i / float(fade_h)
                mask[i, :] *= ramp
                mask[-(i + 1), :] *= ramp

            wear_clipped = np.clip(wear_overlay, 0, 255)
            alpha = mask[:, :, np.newaxis] * np.random.uniform(0.70, 0.95)
            blended = (out[by:by+band_h, bx:bx+band_w] * (1.0 - alpha) + wear_clipped * alpha).astype(np.uint8)
            out[by:by+band_h, bx:bx+band_w] = blended

            annotations.append({
                "class_id": 0,
                "class_name": "rail_surface_wear",
                "bbox": [bx, by, bx + band_w, by + band_h],
                "metadata": {
                    "wear_level": round(wear_level, 2),
                    "band_width_px": band_w,
                    "band_height_px": band_h
                }
            })

        return out, annotations

    def add_surface_pitting(
        self,
        img: np.ndarray,
        pitting_level: float = 0.25,
        rail_width_ratio: float = 0.50
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Add surface pitting corrosion and cavitation micro-craters.
        Produces:
        - Clustered dark pit cavities with reddish-brown rust oxide halos
        """
        if pitting_level <= 0.0:
            return img, []

        out = img.copy()
        h, w = out.shape[:2]
        rail_w = int(w * rail_width_ratio)
        rx1 = (w - rail_w) // 2
        rx2 = rx1 + rail_w

        annotations = []
        num_pit_clusters = int(max(1, round(pitting_level * 4)))

        for _ in range(num_pit_clusters):
            cw = int(np.random.uniform(25, 65))
            ch = int(np.random.uniform(25, 75))
            cx = int(np.random.uniform(rx1 + 15, rx2 - cw - 15))
            cy = int(np.random.uniform(30, h - ch - 30))

            num_pits = int(np.random.randint(5, 16))
            for _ in range(num_pits):
                px = np.random.randint(cx + 4, cx + cw - 4)
                py = np.random.randint(cy + 4, cy + ch - 4)
                pr = np.random.randint(2, 6)

                # Rust oxidation halo (orange-brown tone)
                rust_b = np.random.randint(15, 35)
                rust_g = np.random.randint(65, 105)
                rust_r = np.random.randint(155, 215)
                cv2.circle(out, (px, py), pr + 2, (rust_b, rust_g, rust_r), -1)

                # Dark pit crater cavity
                cavity_tone = np.random.randint(20, 45)
                cv2.circle(out, (px, py), pr, (cavity_tone, cavity_tone, cavity_tone), -1)
                cv2.circle(out, (px, py + 1), max(1, pr // 2), (cavity_tone + 20, cavity_tone + 20, cavity_tone + 20), -1)

            annotations.append({
                "class_id": 2,
                "class_name": "pitting_galling",
                "bbox": [cx, cy, cx + cw, cy + ch],
                "metadata": {
                    "defect_subtype": "surface_pitting",
                    "pitting_level": round(pitting_level, 2),
                    "cluster_pits": num_pits
                }
            })

        return out, annotations

    def add_galling(
        self,
        img: np.ndarray,
        galling_level: float = 0.30,
        rail_width_ratio: float = 0.50
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Add dry contact galling wear (severe adhesive metal tearing).
        Produces:
        - Ragged torn metal flakes with smeared high-contrast texture
        - Rough gouges representing unlubricated shoe-to-rail metal transfer
        """
        if galling_level <= 0.0:
            return img, []

        out = img.copy()
        h, w = out.shape[:2]
        rail_w = int(w * rail_width_ratio)
        rx1 = (w - rail_w) // 2
        rx2 = rx1 + rail_w

        annotations = []
        num_galling_zones = int(max(1, round(galling_level * 3)))

        for _ in range(num_galling_zones):
            gw = int(np.random.uniform(22, 55))
            gh = int(np.random.uniform(40, 110))
            gx = int(np.random.uniform(rx1 + 12, rx2 - gw - 12))
            gy = int(np.random.uniform(25, h - gh - 25))

            galling_roi = out[gy:gy+gh, gx:gx+gw].astype(np.float32)

            torn_noise = np.random.normal(0, 40.0, (gh, gw, 3))
            sobel_smear = np.zeros_like(torn_noise)
            for col in range(gw):
                tear_val = np.random.uniform(-45.0, 55.0)
                sobel_smear[:, col] = tear_val

            composite_tear = galling_roi + (torn_noise * 0.65) + (sobel_smear * 0.45)
            composite_tear = np.clip(composite_tear, 15, 245).astype(np.uint8)

            mask = np.zeros((gh, gw), dtype=np.uint8)
            cv2.ellipse(mask, (gw // 2, gh // 2), (gw // 2, gh // 2), 0, 0, 360, 255, -1)
            edge_noise = np.random.randint(0, 50, (gh, gw), dtype=np.uint8)
            mask = cv2.subtract(mask, edge_noise)
            mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
            alpha = (mask / 255.0)[:, :, np.newaxis]

            out[gy:gy+gh, gx:gx+gw] = (out[gy:gy+gh, gx:gx+gw] * (1.0 - alpha) + composite_tear * alpha).astype(np.uint8)

            annotations.append({
                "class_id": 2,
                "class_name": "pitting_galling",
                "bbox": [gx, gy, gx + gw, gy + gh],
                "metadata": {
                    "defect_subtype": "galling_wear",
                    "galling_level": round(galling_level, 2),
                    "zone_area": gw * gh
                }
            })

        return out, annotations

    def add_rail_corrosion(
        self,
        img: np.ndarray,
        corrosion_level: float = 0.25,
        rail_width_ratio: float = 0.50
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Add surface corrosion/oxidation patches on machined steel blade.
        """
        if corrosion_level <= 0.0:
            return img, []

        out = img.copy()
        h, w = out.shape[:2]
        rail_w = int(w * rail_width_ratio)
        rx1 = (w - rail_w) // 2
        rx2 = rx1 + rail_w

        annotations = []
        num_spots = int(max(1, round(corrosion_level * 4)))

        for _ in range(num_spots):
            rw = int(np.random.uniform(25, 65))
            rh = int(np.random.uniform(30, 90))
            rx = int(np.random.uniform(rx1 + 10, rx2 - rw - 10))
            ry = int(np.random.uniform(25, h - rh - 25))

            rust_patch = np.zeros((rh, rw, 3), dtype=np.uint8)
            rust_patch[:, :, 0] = np.random.randint(15, 35, (rh, rw))
            rust_patch[:, :, 1] = np.random.randint(65, 105, (rh, rw))
            rust_patch[:, :, 2] = np.random.randint(155, 215, (rh, rw))

            mask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), np.random.randint(0, 180), 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 2)
            alpha = (mask / 255.0)[:, :, np.newaxis]

            out[ry:ry+rh, rx:rx+rw] = (out[ry:ry+rh, rx:rx+rw] * (1.0 - alpha) + rust_patch * alpha).astype(np.uint8)

            annotations.append({
                "class_id": 3,
                "class_name": "rail_corrosion",
                "bbox": [rx, ry, rx + rw, ry + rh],
                "metadata": {
                    "corrosion_level": round(corrosion_level, 2)
                }
            })

        return out, annotations

    def generate_rail_image(
        self,
        width: int = 640,
        height: int = 640,
        joint_step_mm: float = 0.0,
        friction_wear_level: float = 0.0,
        pitting_level: float = 0.0,
        galling_level: float = 0.0,
        corrosion_level: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generate complete composite guide rail inspection image with specified defects.
        
        Returns:
            (image_bgr, list_of_defect_dicts)
        """
        img = self.generate_pristine_rail(width=width, height=height)
        all_annotations = []

        # 1. Friction wear marks
        if friction_wear_level > 0.0:
            img, wear_dets = self.add_friction_wear(img, wear_level=friction_wear_level)
            all_annotations.extend(wear_dets)

        # 2. Joint gap step misalignment
        if joint_step_mm > 0.0:
            img, joint_det = self.add_joint_gap_step(img, joint_step_mm=joint_step_mm)
            if joint_det:
                all_annotations.append(joint_det)

        # 3. Surface pitting
        if pitting_level > 0.0:
            img, pit_dets = self.add_surface_pitting(img, pitting_level=pitting_level)
            all_annotations.extend(pit_dets)

        # 4. Galling wear
        if galling_level > 0.0:
            img, gall_dets = self.add_galling(img, galling_level=galling_level)
            all_annotations.extend(gall_dets)

        # 5. Corrosion rust
        if corrosion_level > 0.0:
            img, corr_dets = self.add_rail_corrosion(img, corrosion_level=corrosion_level)
            all_annotations.extend(corr_dets)

        return img, all_annotations

    def generate_yolo_dataset(
        self,
        output_dir: str = "data/annotated_rail_yolo",
        num_train: int = 60,
        num_val: int = 15,
        create_yaml: bool = True
    ) -> Dict[str, int]:
        """
        Generate synthetic guide rail dataset with YOLO txt annotations.
        Writes images to:
            {output_dir}/train/images/
            {output_dir}/val/images/
        Writes labels to:
            {output_dir}/train/labels/
            {output_dir}/val/labels/
            
        Returns dictionary with generation summary counts.
        """
        out_path = Path(output_dir)
        total_generated = {"train": 0, "val": 0, "annotations": 0}

        splits = [("train", num_train), ("val", num_val)]

        for split, count in splits:
            img_dir = out_path / split / "images"
            lbl_dir = out_path / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Rail Dataset Generator] Generating {count} images for {split}...")

            for i in range(count):
                scenario = np.random.choice([
                    "pristine",
                    "minor_step_compliant",
                    "scoring_wear",
                    "joint_breach_step",
                    "pitting_and_galling",
                    "composite_severe"
                ])

                if scenario == "pristine":
                    joint_mm = 0.0
                    wear = 0.0
                    pitting = 0.0
                    galling = 0.0
                    corr = 0.0
                elif scenario == "minor_step_compliant":
                    # ISO 7465 compliant joint step (<= 0.20 mm)
                    joint_mm = float(np.random.uniform(0.10, 0.18))
                    wear = float(np.random.uniform(0.05, 0.15))
                    pitting = 0.0
                    galling = 0.0
                    corr = 0.0
                elif scenario == "scoring_wear":
                    joint_mm = 0.0
                    wear = float(np.random.uniform(0.20, 0.50))
                    pitting = float(np.random.uniform(0.0, 0.10))
                    galling = 0.0
                    corr = 0.0
                elif scenario == "joint_breach_step":
                    # ISO 7465 non-compliant step (> 0.20 mm, e.g. 0.25 - 0.50 mm)
                    joint_mm = float(np.random.uniform(0.25, 0.50))
                    wear = float(np.random.uniform(0.10, 0.30))
                    pitting = 0.0
                    galling = float(np.random.uniform(0.0, 0.15))
                    corr = 0.0
                elif scenario == "pitting_and_galling":
                    joint_mm = 0.0
                    wear = float(np.random.uniform(0.15, 0.35))
                    pitting = float(np.random.uniform(0.25, 0.60))
                    galling = float(np.random.uniform(0.20, 0.50))
                    corr = float(np.random.uniform(0.0, 0.15))
                else: # composite_severe
                    joint_mm = float(np.random.uniform(0.25, 0.45))
                    wear = float(np.random.uniform(0.30, 0.60))
                    pitting = float(np.random.uniform(0.20, 0.50))
                    galling = float(np.random.uniform(0.25, 0.55))
                    corr = float(np.random.uniform(0.10, 0.30))

                img, annotations = self.generate_rail_image(
                    width=640,
                    height=640,
                    joint_step_mm=joint_mm,
                    friction_wear_level=wear,
                    pitting_level=pitting,
                    galling_level=galling,
                    corrosion_level=corr
                )

                stem = f"rail_{split}_{i:04d}"
                img_file = img_dir / f"{stem}.jpg"
                lbl_file = lbl_dir / f"{stem}.txt"

                cv2.imwrite(str(img_file), img)

                # Format annotations in YOLO: class_id xc yc bw bh (normalized to [0, 1])
                lines = []
                h, w = img.shape[:2]
                for ann in annotations:
                    cid = ann["class_id"]
                    x1, y1, x2, y2 = ann["bbox"]
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))

                    xc = ((x1 + x2) / 2.0) / float(w)
                    yc = ((y1 + y2) / 2.0) / float(h)
                    bw = (x2 - x1) / float(w)
                    bh = (y2 - y1) / float(h)
                    lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    total_generated["annotations"] += 1

                with open(lbl_file, "w") as f:
                    f.writelines(lines)

                total_generated[split] += 1

        if create_yaml:
            yaml_path = out_path.parent / "rail_defects.yaml"
            data_config = {
                "path": str(out_path.resolve()),
                "train": "train/images",
                "val": "val/images",
                "names": {cid: name for cid, name in RAIL_DEFECT_CLASSES.items()}
            }
            with open(yaml_path, "w") as f:
                yaml.dump(data_config, f, default_flow_style=False)
            print(f"[Rail Dataset Generator] Generated YOLO config at {yaml_path}")

        print(
            f"[Rail Dataset Generator] Complete: {total_generated['train']} train, "
            f"{total_generated['val']} val, {total_generated['annotations']} total defect boxes."
        )
        return total_generated


if __name__ == "__main__":
    generator = RailDatasetGenerator(seed=42)
    generator.generate_yolo_dataset(
        output_dir="data/annotated_rail_yolo",
        num_train=30,
        num_val=10
    )
