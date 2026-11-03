"""
ElevatorGuard-AI: FastAPI High-Performance REST & WebSocket Service
Provides live telemetry streaming, elevator dispatch, fault injection,
NDT computer vision inspection, and RUL prognostics endpoints.
"""

import time
import io
import asyncio
import cv2
import numpy as np
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.simulation.elevator_physics import ElevatorPhysicsSimulator
from src.simulation.esp32_simulator import VirtualESP32Device
from src.edge.serial_ingest import ESP32TelemetryIngestor
from src.vision.defect_analyzer import CableDefectDetector, GuideRailDefectDetector
from src.telemetry.feature_extraction import TelemetryFeatureExtractor
from src.fusion.multimodal_fusion import MultiModalFusionEngine
from src.fusion.rul_estimator import RULEstimator

app = FastAPI(
    title="ElevatorGuard-AI Edge API",
    version="1.0.0",
    description="Multi-Modal Edge Pipeline for Elevator Guide Rail Defect Quantification & RUL Prediction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Shared Singletons
physics_sim = ElevatorPhysicsSimulator(num_floors=8)
ingestor = ESP32TelemetryIngestor(virtual_mode=True)
esp32_device = VirtualESP32Device(
    physics_sim=physics_sim,
    broadcast_rate_hz=30,
    packet_sink=ingestor.inject_virtual_packet
)
defect_detector = CableDefectDetector()
rail_detector = GuideRailDefectDetector()
feature_extractor = TelemetryFeatureExtractor()
fusion_engine = MultiModalFusionEngine()
rul_estimator = RULEstimator()


# Request / Response Schemas
class DispatchRequest(BaseModel):
    floor: int


class DefectInjectionRequest(BaseModel):
    defect_type: str # 'broken_wire', 'abrasion', 'corrosion', 'tension_imbalance'
    severity: float = 1.0


@app.on_event("startup")
def startup_event():
    """Start ingestion and simulated ESP32 node."""
    ingestor.start()
    esp32_device.start()
    print("[FastAPI] ElevatorGuard-AI background services initialized.")


@app.on_event("shutdown")
def shutdown_event():
    """Cleanly terminate background threads."""
    esp32_device.stop()
    ingestor.stop()
    print("[FastAPI] Background services stopped.")


@app.get("/")
def read_root():
    return {
        "system": "ElevatorGuard-AI",
        "version": "1.0.0",
        "status": "ONLINE",
        "standards": ["ISO 4309", "EN 81-20/50", "ISO 18738"],
        "esp32_protocol": "v2.1"
    }


@app.get("/api/status")
def get_status():
    """Retrieve current elevator kinematics and multi-modal health state."""
    state = physics_sim.state
    window = ingestor.get_latest_window(window_size=32)
    features = feature_extractor.extract_features(
        accel_x=window["accel_x"],
        accel_y=window["accel_y"],
        accel_z=window["accel_z"],
        tensions=window["tensions"]
    )

    # Dynamic HI calculation
    vibe_penalty = max(0.0, (features["acc_rms_z"] - 0.20) / 0.40)
    hi_dyn = max(0.05, min(1.0, 1.0 - (0.5 * vibe_penalty + 0.5 * (features["tension_imbalance_ratio"] / 0.25))))

    # Multi-Modal Fusion with simulated/visual wear
    vis_res = defect_detector.visual_analyzer.calculate_hi(
        wire_breaks=state.wire_breaks,
        wear_pct=state.wear_pct,
        corrosion_pct=state.corrosion_pct
    )
    fusion_res = fusion_engine.fuse(visual_result=vis_res, hi_telemetry=hi_dyn)
    rul_res = rul_estimator.estimate_rul(current_hi=fusion_res.hi_composite, current_trips=state.total_trips)

    return {
        "kinematics": {
            "floor": state.floor,
            "target_floor": state.target_floor,
            "position_m": round(state.position_m, 2),
            "velocity_mps": round(state.velocity_mps, 3),
            "is_moving": state.is_moving,
            "door_open_pct": round(state.door_open_pct, 2),
            "emergency_stop": state.emergency_stop
        },
        "telemetry": {
            "accel_rms_z": round(features["acc_rms_z"], 3),
            "tension_imbalance_ratio": round(features["tension_imbalance_ratio"], 3),
            "individual_tensions": features["individual_tensions"],
            "alarms": {
                "tension_imbalance": features["tension_imbalance_alarm"],
                "vibration_exceeded": features["vibration_alarm"],
                "emergency_stop": state.emergency_stop
            }
        },
        "guide_rail": {
            "vibration_sensor_location": "Guide Rail & Car Guide Shoe Interface",
            "camera_target": "T-Profile Guide Rail Blade & Fishplate Joints",
            "joint_misalignment_mm": round(physics_sim.rail_joint_misalignment_mm, 2),
            "scoring_wear_pct": round(physics_sim.rail_scoring_ratio * 100, 1),
            "rust_pct": round(physics_sim.rail_corrosion_ratio * 100, 1),
            "iso_7465_compliant": physics_sim.rail_joint_misalignment_mm <= 0.20
        },
        "health": {
            "hi_composite": round(fusion_res.hi_composite, 4),
            "hi_vision": round(fusion_res.hi_vision, 4),
            "hi_telemetry": round(fusion_res.hi_telemetry, 4),
            "status": fusion_res.status,
            "summary": fusion_res.summary_message,
            "rul_cycles": rul_res.rul_cycles,
            "rul_days": rul_res.rul_days
        }
    }


@app.post("/api/elevator/dispatch")
def dispatch_elevator(req: DispatchRequest):
    """Command the elevator to travel to a designated floor."""
    success = physics_sim.dispatch_to_floor(req.floor)
    return {
        "success": success,
        "target_floor": req.floor,
        "message": f"Elevator dispatched to Floor {req.floor}" if success else "Dispatch rejected (E-stop active or floor matches current)."
    }


@app.post("/api/elevator/emergency_stop")
def trigger_e_stop():
    """Trigger emergency stop brake."""
    physics_sim.trigger_emergency_stop()
    ingestor.send_command("EMERGENCY_STOP")
    return {"status": "EMERGENCY_STOP_TRIGGERED", "success": True}


@app.post("/api/elevator/reset_alarm")
def reset_alarm():
    """Reset emergency stop and cleared faults."""
    physics_sim.reset_emergency_stop()
    physics_sim.tension_imbalance_factor = 0.0
    physics_sim.broken_wire_count = 0
    physics_sim.wear_ratio = 0.02
    physics_sim.corrosion_ratio = 0.01
    physics_sim.state.wire_breaks = 0
    physics_sim.state.wear_pct = 0.02
    physics_sim.state.corrosion_pct = 0.01
    ingestor.send_command("RESET_ALARM")
    return {"status": "ALARMS_CLEARED", "success": True}


@app.post("/api/elevator/inject_defect")
def inject_defect(req: DefectInjectionRequest):
    """Inject cable defects or sensor imbalance."""
    physics_sim.inject_defect(req.defect_type, req.severity)
    return {
        "status": "DEFECT_INJECTED",
        "defect_type": req.defect_type,
        "severity": req.severity
    }


@app.post("/api/vision/inspect")
async def inspect_guide_rail_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded guide rail inspection image and returns
    Guide Rail defect detections, joint step, and ISO 7465 Visual Health Index.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format."})

    hi_res, detections, annotated_img = rail_detector.analyze_rail_frame(img, return_annotated=True)

    return {
        "target": "ELEVATOR_GUIDE_RAIL",
        "standard": "ISO 7465 / ISO 8100-34",
        "iso_7465_result": hi_res.to_dict(),
        "detections_count": len(detections),
        "detections": detections
    }


@app.get("/api/simulation/frame")
def get_simulation_snapshot():
    """Return a JPEG snapshot of the virtual hoistway."""
    # Generate live hoistway visual representation
    h, w = 480, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (20, 22, 26)

    # Hoistway shaft
    cv2.rectangle(frame, (100, 30), (320, 450), (12, 14, 18), -1)
    cv2.rectangle(frame, (100, 30), (320, 450), (50, 55, 65), 2)

    # Floors
    for flr in range(1, 9):
        fy = int(450 - ((flr - 1) / 7.0) * 380)
        cv2.line(frame, (100, fy), (320, fy), (35, 40, 50), 1)
        cv2.putText(frame, f"F{flr}", (70, fy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 180), 1)

    # Car
    norm_pos = min(1.0, max(0.0, physics_sim.state.position_m / (7.0 * physics_sim.floor_height_m)))
    car_y = int(450 - (norm_pos * 380)) - 40
    car_col = (0, 0, 220) if physics_sim.state.emergency_stop else (180, 120, 50)
    cv2.rectangle(frame, (120, car_y), (220, car_y + 40), car_col, -1)
    cv2.rectangle(frame, (120, car_y), (220, car_y + 40), (240, 240, 240), 2)

    # Telemetry HUD on right
    cv2.putText(frame, "ELEVATORGUARD-AI TWIN", (350, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Floor: {physics_sim.state.floor} / 8", (350, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Velocity: {physics_sim.state.velocity_mps:.2f} m/s", (350, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Wire Breaks: {physics_sim.state.wire_breaks}", (350, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
    cv2.putText(frame, f"HI_sim: {physics_sim.state.hi_simulated:.3f}", (350, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 100), 1)

    _, jpeg_bytes = cv2.imencode(".jpg", frame)
    return Response(content=jpeg_bytes.tobytes(), media_type="image/jpeg")


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """Stream live telemetry frames to WebSocket clients."""
    await websocket.accept()
    try:
        while True:
            state = physics_sim.state
            data = {
                "ts": int(time.time() * 1000),
                "floor": state.floor,
                "velocity": round(state.velocity_mps, 3),
                "accel": [round(state.accel_x, 3), round(state.accel_y, 3), round(state.accel_z, 3)],
                "tensions": [round(t, 1) for t in state.rope_tensions_n],
                "hi": round(state.hi_simulated, 3),
                "wire_breaks": state.wire_breaks,
                "e_stop": state.emergency_stop
            }
            await websocket.send_json(data)
            await asyncio.sleep(0.05) # 20 Hz
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Error: {e}")
