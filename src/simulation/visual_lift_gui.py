"""
ElevatorGuard-AI: Interactive Visual Hoistway & Lift Simulator
Demonstrates full end-to-end integration:
- 8-Floor traction elevator physics & kinematics
- Moving suspension ropes over traction sheave
- Real-time YOLO Cable Defect Detection HUD
- ESP32 Hardware Telemetry Dashboard (vibration, 4x tension load cells, HI_comp, RUL)
- Interactive dispatch and fault injection
"""

import os
import sys
import time
import math
from typing import Optional, Tuple
import numpy as np
import cv2

# Set SDL to dummy if running headless without DISPLAY
if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from src.simulation.elevator_physics import ElevatorPhysicsSimulator
from src.vision.defect_analyzer import CableDefectDetector
from src.telemetry.feature_extraction import TelemetryFeatureExtractor
from src.fusion.multimodal_fusion import MultiModalFusionEngine
from src.fusion.rul_estimator import RULEstimator


class VisualLiftSimulation:
    """
    High-fidelity graphical 2D simulation of elevator hoistway,
    cable inspection camera, and ESP32 edge telemetry panel.
    """

    # Color Palette
    COLOR_BG = (22, 24, 28)
    COLOR_HOISTWAY_BG = (12, 14, 18)
    COLOR_WALL = (45, 52, 60)
    COLOR_CAR = (70, 130, 180) # Steel blue
    COLOR_CAR_INNER = (220, 225, 230)
    COLOR_DOOR = (140, 150, 160)
    COLOR_COUNTERWEIGHT = (90, 95, 105)
    COLOR_SHEAVE = (160, 160, 170)
    COLOR_ROPE = (185, 190, 200)
    COLOR_PANEL_BG = (30, 34, 42)
    COLOR_PANEL_BORDER = (55, 62, 75)
    COLOR_TEXT = (235, 240, 245)
    COLOR_GREEN = (46, 204, 113)
    COLOR_YELLOW = (241, 196, 15)
    COLOR_RED = (231, 76, 60)
    COLOR_ORANGE = (230, 126, 34)

    def __init__(
        self,
        width: int = 960,
        height: int = 700,
        headless: bool = False
    ):
        self.width = width
        self.height = height
        self.headless = headless

        # Core Engines
        self.physics = ElevatorPhysicsSimulator(num_floors=8)
        self.defect_detector = CableDefectDetector()
        self.feature_extractor = TelemetryFeatureExtractor()
        self.fusion_engine = MultiModalFusionEngine()
        self.rul_estimator = RULEstimator()

        # Telemetry history for live plotting
        self.vibe_history_z = [9.81] * 80
        self.hi_history = [0.94] * 40
        self.cable_texture_offset_y = 0.0

        if not PYGAME_AVAILABLE:
            print("[VisualLiftSimulation] Warning: Pygame not installed. GUI will run in headless buffer mode.")
            return

        pygame.init()
        pygame.font.init()
        if self.headless:
            self.screen = pygame.Surface((self.width, self.height))
        else:
            try:
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("ElevatorGuard-AI: Digital Twin Lift Simulation & NDT Inspection")
            except Exception as e:
                print(f"[VisualLiftSimulation] Display init fallback to surface: {e}")
                self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("DejaVu Sans,Arial", 18, bold=True)
        self.font_mid = pygame.font.SysFont("DejaVu Sans,Arial", 13)
        self.font_small = pygame.font.SysFont("DejaVu Sans,Arial", 11)

        # Pre-render a synthetic cable patch for the live NDT camera viewport
        self._update_synthetic_camera_feed()

    def _update_synthetic_camera_feed(self):
        """Generate current frame from hoistway inspection camera."""
        # Rope moves with car velocity
        self.cable_texture_offset_y += self.physics.state.velocity_mps * 8.0

        # Create camera frame showing cable
        frame = np.zeros((300, 420, 3), dtype=np.uint8)
        frame[:] = (30, 32, 36) # Hoistway shadow

        # Cable bounds
        cw = 180
        cx1 = (420 - cw) // 2
        cx2 = cx1 + cw

        # Draw metallic strands with dynamic vertical motion offset
        y_grid, x_grid = np.mgrid[0:300, 0:cw]
        lay_pitch = 60.0
        norm_x = ((x_grid / cw) * 2.0 - 1.0)
        cyl_light = np.maximum(0.25, (1.0 - (norm_x ** 2)) ** 0.5)

        rope_surf = np.zeros((300, cw, 3), dtype=np.float32)
        for c in range(3):
            rope_surf[:, :, c] = 135.0 * cyl_light

        # Helical strands moving with car
        for s in range(6):
            phase = (s / 6.0) * 2.0 * np.pi
            strand_y = np.sin(((y_grid + self.cable_texture_offset_y) / lay_pitch) * 2.0 * np.pi + phase)
            groove = np.exp(-14.0 * ((norm_x - 0.55 * strand_y) ** 2))
            for c in range(3):
                rope_surf[:, :, c] *= (1.0 - (0.42 * groove))

        frame[:, cx1:cx2] = np.clip(rope_surf, 0, 255).astype(np.uint8)

        # Inject visual defects according to physics degradation
        if self.physics.broken_wire_count > 0:
            for b in range(self.physics.broken_wire_count):
                by = int((70 + b * 45 + self.cable_texture_offset_y) % 260) + 20
                bx = cx1 + 35 + (b * 22) % (cw - 70)
                cv2.ellipse(frame, (bx + 12, by + 4), (14, 5), 0, 0, 360, (15, 15, 15), -1)
                cv2.line(frame, (bx + 2, by + 4), (bx + 22, by + 4), (250, 250, 255), 2)

        if self.physics.wear_ratio > 0.05:
            # Abrasion shiny strip
            wy = int((40 + self.cable_texture_offset_y) % 180) + 30
            cv2.ellipse(frame, (cx1 + cw // 2, wy + 40), (25, 45), 0, 0, 360, (215, 215, 220), -1)

        if self.physics.corrosion_ratio > 0.03:
            # Rust patch
            ry = int((130 + self.cable_texture_offset_y) % 200) + 30
            cv2.ellipse(frame, (cx1 + 45, ry + 25), (22, 35), 25, 0, 360, (25, 75, 185), -1)

        # Run CableDefectDetector NDT Pipeline
        hi_vis_res, dets, annotated_bgr = self.defect_detector.analyze_frame(frame, return_annotated=True)
        self.last_hi_vis = hi_vis_res
        self.last_annotated_frame = annotated_bgr

    def handle_events(self) -> bool:
        """Process keyboard & mouse inputs. Returns False if quit requested."""
        if not PYGAME_AVAILABLE:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False
                # Floor dispatch keys 1-8
                elif pygame.K_1 <= event.key <= pygame.K_8:
                    target_floor = event.key - pygame.K_1 + 1
                    self.physics.dispatch_to_floor(target_floor)
                # Fault Injections
                elif event.key == pygame.K_b:
                    self.physics.inject_defect("broken_wire", 1)
                elif event.key == pygame.K_r:
                    self.physics.inject_defect("corrosion", 1.0)
                elif event.key == pygame.K_w:
                    self.physics.inject_defect("abrasion", 1.0)
                elif event.key == pygame.K_t:
                    self.physics.inject_defect("tension_imbalance", 0.28)
                elif event.key == pygame.K_e:
                    self.physics.trigger_emergency_stop()
                elif event.key == pygame.K_c:
                    self.physics.reset_emergency_stop()
                    self.physics.tension_imbalance_factor = 0.0

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                self._handle_mouse_click(mx, my)

        return True

    def _handle_mouse_click(self, mx: int, my: int):
        """Handle clicks on floor call buttons and control panels."""
        # Floor buttons in Hoistway panel (x: 20-55, y: floor levels)
        for flr in range(1, 9):
            btn_y = self.height - 70 - (flr - 1) * 72
            if 15 <= mx <= 50 and btn_y <= my <= btn_y + 26:
                self.physics.dispatch_to_floor(flr)

        # Interactive Control Buttons (x: 480-940, y: 645-685)
        actions = [
            (480, 560, "DISPATCH_F8", lambda: self.physics.dispatch_to_floor(8)),
            (570, 650, "DISPATCH_F1", lambda: self.physics.dispatch_to_floor(1)),
            (660, 740, "INJECT_BREAK", lambda: self.physics.inject_defect("broken_wire", 1)),
            (750, 830, "IMBALANCE", lambda: self.physics.inject_defect("tension_imbalance", 0.30)),
            (840, 930, "ESTOP", lambda: self.physics.trigger_emergency_stop()),
        ]
        for bx1, bx2, name, act in actions:
            if bx1 <= mx <= bx2 and 650 <= my <= 685:
                act()

    def update(self, dt: float = 0.03):
        """Step simulation physics and compute multi-modal health & RUL."""
        self.physics.step(dt=dt)
        self._update_synthetic_camera_feed()

        # Update telemetry history
        self.vibe_history_z.append(self.physics.state.accel_z)
        if len(self.vibe_history_z) > 80:
            self.vibe_history_z.pop(0)

        # Feature Extraction
        t_arr = np.array(self.physics.state.rope_tensions_n, dtype=np.float32)
        features = self.feature_extractor.extract_features(
            accel_x=np.array([self.physics.state.accel_x]),
            accel_y=np.array([self.physics.state.accel_y]),
            accel_z=np.array(self.vibe_history_z[-16:]),
            tensions=t_arr
        )

        # Telemetry Dynamic Health (HI_dyn) calculation
        vibe_penalty = max(0.0, (features["acc_rms_z"] - 0.20) / 0.40)
        imbalance_penalty = max(0.0, features["tension_imbalance_ratio"] / 0.30)
        hi_dyn = max(0.05, min(1.0, 1.0 - (0.6 * vibe_penalty + 0.4 * imbalance_penalty)))

        # Stream 3: Multi-Modal Fusion
        fusion_res = self.fusion_engine.fuse(
            visual_result=self.last_hi_vis,
            hi_telemetry=hi_dyn
        )
        self.last_fusion = fusion_res

        # RUL Prognostics
        self.hi_history.append(fusion_res.hi_composite)
        if len(self.hi_history) > 40:
            self.hi_history.pop(0)

        rul_res = self.rul_estimator.estimate_rul(
            current_hi=fusion_res.hi_composite,
            current_trips=self.physics.state.total_trips,
            hi_history=self.hi_history[-10:],
            trip_history=list(range(self.physics.state.total_trips - 90, self.physics.state.total_trips + 1, 10))
        )
        self.last_rul = rul_res

    def render(self):
        """Draw complete hoistway, camera HUD, and telemetry panels."""
        if not PYGAME_AVAILABLE:
            return

        self.screen.fill(self.COLOR_BG)

        # 1. Render Left Panel: Hoistway & Lift Shaft (x: 0 to 450)
        self._render_hoistway(x=10, y=10, w=440, h=self.height - 20)

        # 2. Render Top-Right Panel: Real-Time Cable NDT Camera (x: 460, y: 10, w: 490, h: 320)
        self._render_camera_viewport(x=460, y=10, w=490, h=315)

        # 3. Render Bottom-Right Panel: ESP32 Telemetry & RUL Engine (x: 460, y: 335, w: 490, h: 345)
        self._render_telemetry_panel(x=460, y=335, w=490, h=345)

        if not self.headless:
            pygame.display.flip()

    def _render_hoistway(self, x: int, y: int, w: int, h: int):
        """Draw 8-floor elevator shaft with car, counterweight, and rotating sheave."""
        # Background shaft
        pygame.draw.rect(self.screen, self.COLOR_HOISTWAY_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        # Header Title
        title_surf = self.font_large.render("ELEVATORGUARD HOISTWAY TWIN", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (x + 15, y + 12))

        # Hoistway shaft area
        shaft_x = x + 70
        shaft_w = w - 90
        shaft_y = y + 45
        shaft_h = h - 60
        pygame.draw.rect(self.screen, (8, 10, 14), (shaft_x, shaft_y, shaft_w, shaft_h))

        # Draw 8 Floors & Floor Call Buttons
        floor_spacing = shaft_h / 8.0
        for flr in range(1, 9):
            fy = shaft_y + shaft_h - (flr * floor_spacing)
            # Floor separator beam
            pygame.draw.line(self.screen, (35, 40, 50), (shaft_x, fy), (shaft_x + shaft_w, fy), 1)

            # Floor call button on the left
            btn_active = (self.physics.state.target_floor == flr)
            btn_cur = (self.physics.state.floor == flr)
            btn_col = self.COLOR_GREEN if btn_cur else (self.COLOR_ORANGE if btn_active else (40, 45, 55))
            pygame.draw.rect(self.screen, btn_col, (x + 15, fy + 15, 38, 26), border_radius=4)
            lbl = self.font_mid.render(f"F{flr}", True, (255, 255, 255) if (btn_cur or btn_active) else (160, 165, 175))
            self.screen.blit(lbl, (x + 23, fy + 20))

        # Overhead Traction Sheave (Top center)
        sheave_cx = shaft_x + (shaft_w // 2) - 30
        sheave_cy = shaft_y + 25
        sheave_r = 22
        pygame.draw.circle(self.screen, self.COLOR_SHEAVE, (sheave_cx, sheave_cy), sheave_r)
        pygame.draw.circle(self.screen, (60, 65, 75), (sheave_cx, sheave_cy), 7)
        # Sheave rotation spokes
        rot_angle = (self.physics.state.position_m * 4.0)
        spoke_x = sheave_cx + int(sheave_r * math.cos(rot_angle))
        spoke_y = sheave_cy + int(sheave_r * math.sin(rot_angle))
        pygame.draw.line(self.screen, (40, 45, 55), (sheave_cx, sheave_cy), (spoke_x, spoke_y), 3)

        # Elevator Car Geometry
        car_w = 95
        car_h = 55
        max_travel_m = 7 * self.physics.floor_height_m
        norm_pos = min(1.0, max(0.0, self.physics.state.position_m / max_travel_m))
        car_bottom_y = shaft_y + shaft_h - 10
        car_top_limit_y = shaft_y + 60
        car_y = car_bottom_y - int(norm_pos * (car_bottom_y - car_top_limit_y)) - car_h
        car_x = shaft_x + 35

        # Counterweight Geometry (moves inversely)
        cw_w = 28
        cw_h = 70
        cw_x = shaft_x + shaft_w - 42
        cw_y = shaft_y + 60 + int(norm_pos * (car_bottom_y - car_top_limit_y - 20))

        # Draw 4 Suspension Wire Ropes (Traction Sheave to Car & Counterweight)
        for r_idx in range(4):
            rope_offset = (r_idx - 1.5) * 5
            # From sheave to car
            rx_car = car_x + (car_w // 2) + rope_offset
            pygame.draw.line(self.screen, self.COLOR_ROPE, (sheave_cx + rope_offset, sheave_cy), (rx_car, car_y), 2)
            # From sheave to counterweight
            rx_cw = cw_x + (cw_w // 2) + rope_offset
            pygame.draw.line(self.screen, self.COLOR_ROPE, (sheave_cx + 15 + rope_offset, sheave_cy), (rx_cw, cw_y), 2)

        # Draw Counterweight
        pygame.draw.rect(self.screen, self.COLOR_COUNTERWEIGHT, (cw_x, cw_y, cw_w, cw_h), border_radius=3)
        pygame.draw.rect(self.screen, (40, 45, 50), (cw_x, cw_y, cw_w, cw_h), 2)

        # Draw Elevator Car
        car_color = self.COLOR_RED if self.physics.state.emergency_stop else self.COLOR_CAR
        pygame.draw.rect(self.screen, car_color, (car_x, car_y, car_w, car_h), border_radius=4)
        pygame.draw.rect(self.screen, (255, 255, 255), (car_x, car_y, car_w, car_h), 2, border_radius=4)
        # Car interior
        pygame.draw.rect(self.screen, self.COLOR_CAR_INNER, (car_x + 6, car_y + 6, car_w - 12, car_h - 12))

        # Sliding Doors
        door_w = int(((car_w - 12) / 2.0) * (1.0 - self.physics.state.door_open_pct))
        pygame.draw.rect(self.screen, self.COLOR_DOOR, (car_x + 6, car_y + 6, door_w, car_h - 12))
        pygame.draw.rect(self.screen, self.COLOR_DOOR, (car_x + car_w - 6 - door_w, car_y + 6, door_w, car_h - 12))

        # Passenger indicator
        p_txt = self.font_small.render(f"LOAD: {self.physics.state.payload_mass_kg}kg", True, (20, 20, 20))
        self.screen.blit(p_txt, (car_x + 10, car_y + car_h // 2 - 6))

        # Hoistway Status HUD (Bottom left)
        state_color = self.COLOR_RED if self.physics.state.emergency_stop else (self.COLOR_GREEN if not self.physics.state.is_moving else self.COLOR_YELLOW)
        state_str = "EMERGENCY STOP" if self.physics.state.emergency_stop else ("STATIONARY" if not self.physics.state.is_moving else "IN MOTION")
        stat_lbl = self.font_mid.render(f"STATUS: {state_str} | VEL: {self.physics.state.velocity_mps:.2f} m/s", True, state_color)
        self.screen.blit(stat_lbl, (x + 15, h - 18))

    def _render_camera_viewport(self, x: int, y: int, w: int, h: int):
        """Render OpenCV annotated cable inspection frame inside Pygame."""
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        # Title
        hdr = self.font_large.render("NDT VISION DEFECT DETECTOR (YOLO & ISO 4309)", True, self.COLOR_TEXT)
        self.screen.blit(hdr, (x + 15, y + 10))

        if hasattr(self, "last_annotated_frame") and self.last_annotated_frame is not None:
            # Resize frame to viewport
            view_w = w - 24
            view_h = h - 45
            resized = cv2.resize(self.last_annotated_frame, (view_w, view_h))
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Create pygame surface
            frame_surf = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
            self.screen.blit(frame_surf, (x + 12, y + 36))

    def _render_telemetry_panel(self, x: int, y: int, w: int, h: int):
        """Render ESP32 telemetry waveforms, rope tensions, HI gauge, and control buttons."""
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BG, (x, y, w, h), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_PANEL_BORDER, (x, y, w, h), 2, border_radius=8)

        hdr = self.font_large.render("ESP32 PROTOCOL TELEMETRY & MULTI-MODAL RUL", True, self.COLOR_TEXT)
        self.screen.blit(hdr, (x + 15, y + 10))

        # 1. ESP32 Hardware Status Badge
        badge_col = self.COLOR_RED if self.physics.state.emergency_stop else self.COLOR_GREEN
        pygame.draw.circle(self.screen, badge_col, (x + w - 25, y + 20), 8)
        lbl_esp = self.font_small.render("ESP32: ONLINE", True, (180, 190, 200))
        self.screen.blit(lbl_esp, (x + w - 120, y + 14))

        # 2. 4-Rope Tension Bar Gauges
        tns_title = self.font_mid.render("SUSPENSION ROPE TENSIONS (HX711 LOAD CELLS):", True, self.COLOR_TEXT)
        self.screen.blit(tns_title, (x + 15, y + 42))

        tensions = self.physics.state.rope_tensions_n
        avg_t = np.mean(tensions)
        imbalance = (np.max(tensions) - np.min(tensions)) / max(1.0, avg_t)

        for i, t in enumerate(tensions):
            bar_x = x + 15 + (i * 115)
            bar_y = y + 65
            bar_w = 95
            bar_h = 16
            
            # Fill ratio (2000 N to 3200 N range)
            fill_pct = min(1.0, max(0.0, (t - 1500) / 2000.0))
            col = self.COLOR_RED if abs(t - avg_t) / avg_t > 0.15 else self.COLOR_GREEN

            pygame.draw.rect(self.screen, (40, 45, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            pygame.draw.rect(self.screen, col, (bar_x, bar_y, int(bar_w * fill_pct), bar_h), border_radius=3)
            pygame.draw.rect(self.screen, (70, 75, 85), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=3)

            t_txt = self.font_small.render(f"Rope {i+1}: {int(t)} N", True, self.COLOR_TEXT)
            self.screen.blit(t_txt, (bar_x, bar_y + 19))

        # Imbalance alert badge
        imb_color = self.COLOR_RED if imbalance > 0.15 else self.COLOR_GREEN
        imb_txt = self.font_small.render(f"Tension Imbalance: {imbalance * 100:.1f}% (Limit: 15.0%)", True, imb_color)
        self.screen.blit(imb_txt, (x + 15, y + 105))

        # 3. Vibration Waveform (MPU-6050 Az)
        vibe_box_x = x + 15
        vibe_box_y = y + 130
        vibe_box_w = 220
        vibe_box_h = 75
        pygame.draw.rect(self.screen, (15, 17, 22), (vibe_box_x, vibe_box_y, vibe_box_w, vibe_box_h), border_radius=4)
        pygame.draw.rect(self.screen, (50, 55, 65), (vibe_box_x, vibe_box_y, vibe_box_w, vibe_box_h), 1, border_radius=4)
        v_title = self.font_small.render("VIBRATION WAVEFORM (a_z m/s²)", True, (160, 170, 180))
        self.screen.blit(v_title, (vibe_box_x + 8, vibe_box_y + 6))

        # Draw waveform line
        points = []
        for i, val in enumerate(self.vibe_history_z[-45:]):
            px = vibe_box_x + 8 + int((i / 45.0) * (vibe_box_w - 16))
            norm_v = (val - 9.0) / 2.0 # Center around 9.81
            py = vibe_box_y + (vibe_box_h // 2) - int(norm_v * 20.0)
            py = max(vibe_box_y + 20, min(vibe_box_y + vibe_box_h - 4, py))
            points.append((px, py))
        if len(points) >= 2:
            pygame.draw.lines(self.screen, self.COLOR_YELLOW, False, points, 2)

        # 4. Multi-Modal Composite Health Index & RUL Counter
        hi_box_x = x + 250
        hi_box_y = y + 130
        hi_box_w = w - 265
        hi_box_h = 75
        pygame.draw.rect(self.screen, (15, 17, 22), (hi_box_x, hi_box_y, hi_box_w, hi_box_h), border_radius=4)
        pygame.draw.rect(self.screen, (50, 55, 65), (hi_box_x, hi_box_y, hi_box_w, hi_box_h), 1, border_radius=4)

        cur_hi = self.last_fusion.hi_composite if hasattr(self, "last_fusion") else 0.94
        hi_col = self.COLOR_RED if cur_hi < 0.25 else (self.COLOR_YELLOW if cur_hi < 0.65 else self.COLOR_GREEN)

        hi_val_surf = self.font_large.render(f"HI_comp: {cur_hi:.3f}", True, hi_col)
        self.screen.blit(hi_val_surf, (hi_box_x + 12, hi_box_y + 10))

        rul_cycles = self.last_rul.rul_cycles if hasattr(self, "last_rul") else 125000
        rul_days = self.last_rul.rul_days if hasattr(self, "last_rul") else 500.0
        rul_surf = self.font_mid.render(f"RUL: {rul_cycles:,} Trips | {rul_days:.1f} Days", True, self.COLOR_TEXT)
        self.screen.blit(rul_surf, (hi_box_x + 12, hi_box_y + 36))

        rec_surf = self.font_small.render(
            f"ISO 4309: {self.last_fusion.status if hasattr(self, 'last_fusion') else 'NOMINAL'}",
            True,
            hi_col
        )
        self.screen.blit(rec_surf, (hi_box_x + 12, hi_box_y + 55))

        # 5. Interactive Action Buttons
        btn_y = y + 225
        btn_configs = [
            ("Call Floor 8", x + 15, 105, (50, 100, 180)),
            ("Call Floor 1", x + 130, 105, (50, 100, 180)),
            ("+ Wire Break", x + 245, 110, (180, 80, 40)),
            ("Tension Skew", x + 365, 105, (160, 120, 30)),
            ("EMERGENCY STOP (E)", x + 15, 220, (180, 40, 40)),
            ("RESET ALARM (C)", x + 250, 220, (40, 140, 60)),
        ]

        for label, bx, bw, col in btn_configs[:4]:
            pygame.draw.rect(self.screen, col, (bx, btn_y, bw, 28), border_radius=4)
            b_lbl = self.font_small.render(label, True, (255, 255, 255))
            self.screen.blit(b_lbl, (bx + 10, btn_y + 7))

        for label, bx, bw, col in btn_configs[4:]:
            pygame.draw.rect(self.screen, col, (bx, btn_y + 38, bw, 32), border_radius=4)
            b_lbl = self.font_mid.render(label, True, (255, 255, 255))
            self.screen.blit(b_lbl, (bx + 18, btn_y + 44))

    def get_screen_frame_bgr(self) -> np.ndarray:
        """Capture the current GUI render as a BGR numpy image for web dashboard streaming."""
        if not PYGAME_AVAILABLE:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb_arr = pygame.surfarray.array3d(self.screen)
        # Transpose from (w, h, c) to (h, w, c) and convert RGB to BGR
        transposed = np.transpose(rgb_arr, (1, 0, 2))
        return cv2.cvtColor(transposed, cv2.COLOR_RGB2BGR)

    def run_standalone(self, max_seconds: Optional[float] = None):
        """Run interactive simulation loop."""
        print("[VisualLiftSimulation] Simulation started. Press 1-8 to dispatch, B for wire breaks, E for emergency brake, Q to quit.")
        running = True
        t_start = time.time()

        while running:
            dt = self.clock.tick(30) / 1000.0
            running = self.handle_events()
            self.update(dt=dt)
            self.render()

            if max_seconds and (time.time() - t_start) >= max_seconds:
                break

        if PYGAME_AVAILABLE and not self.headless:
            pygame.quit()
        print("[VisualLiftSimulation] Simulation stopped.")


if __name__ == "__main__":
    sim = VisualLiftSimulation(headless=False)
    sim.run_standalone()
