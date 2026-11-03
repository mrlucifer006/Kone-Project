"""
ElevatorGuard-AI: Camera Vision Agent (Stream 1)
Dedicated autonomous agent managing the Hoistway NDT Camera stream:
Ingests video frames, runs deep learning / heuristic wire rope defect quantification,
calculates ISO 4309 Visual Health Index (HI_vis), and publishes vision telemetry onto the message bus.
"""

import time
import queue
import threading
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from src.agents.base_agent import BaseAgent, AgentStatus
from src.agents.message_bus import MessageBus
from src.vision.defect_analyzer import CableDefectDetector
from src.vision.visual_hi import VisualInspectionResult


class CameraVisionAgent(BaseAgent):
    """
    Stream 1 Ingestion & Processing Agent.
    Continuously receives and inspects elevator suspension cable video frames.
    """

    def __init__(
        self,
        message_bus: MessageBus,
        name: str = "CameraVisionAgent",
        defect_detector: Optional[CableDefectDetector] = None,
        max_frame_queue: int = 10,
        frame_timeout_sec: float = 3.0
    ):
        super().__init__(name=name, message_bus=message_bus, tick_interval=0.02)
        self.defect_detector = defect_detector or CableDefectDetector()
        self.frame_timeout_sec = frame_timeout_sec
        self.frame_queue: queue.Queue = queue.Queue(maxsize=max_frame_queue)

        # Stream 1 state & metrics
        self.total_frames_ingested = 0
        self.total_frames_processed = 0
        self.dropped_frames = 0
        self.last_frame_timestamp: Optional[float] = None
        self.last_inference_ms = 0.0
        self.avg_inference_ms = 0.0
        self.fps = 0.0
        self._fps_window = []

        # Latest inspection results
        self.latest_hi_vis: float = 1.0
        self.latest_wire_breaks: int = 0
        self.latest_wear_pct: float = 0.0
        self.latest_corrosion_pct: float = 0.0
        self.latest_discard_required: bool = False
        self.latest_status: str = "NOMINAL_HEALTHY"
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_annotated_frame: Optional[np.ndarray] = None

        self._state_lock = threading.RLock()

    def ingest_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Thread-safe entry point to push a raw camera frame into the pipeline.
        
        Args:
            frame: BGR numpy image array.
            timestamp: Optional capture timestamp (defaults to current time).
            metadata: Optional camera telemetry (e.g. exposure, camera_id).
            
        Returns:
            True if queued successfully, False if dropped due to queue congestion.
        """
        if frame is None or frame.size == 0:
            return False

        ts = timestamp or time.time()
        item = (frame, ts, metadata or {})

        with self._state_lock:
            self.total_frames_ingested += 1
            self.last_frame_timestamp = ts

        try:
            self.frame_queue.put_nowait(item)
            return True
        except queue.Full:
            # Overwrite oldest frame to keep pipeline real-time
            try:
                _ = self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(item)
                with self._state_lock:
                    self.dropped_frames += 1
                return True
            except (queue.Empty, queue.Full):
                with self._state_lock:
                    self.dropped_frames += 1
                return False

    def on_tick(self):
        """Process frames from frame_queue in agent worker thread."""
        try:
            frame, ts, meta = self.frame_queue.get_nowait()
        except queue.Empty:
            return

        t_start = time.perf_counter()
        hi_res, detections, annotated = self.defect_detector.analyze_frame(frame, return_annotated=True)
        infer_ms = (time.perf_counter() - t_start) * 1000.0

        with self._state_lock:
            self.total_frames_processed += 1
            self.last_inference_ms = round(infer_ms, 2)
            # Update running average inference time
            alpha = 0.1
            self.avg_inference_ms = round(
                infer_ms if self.avg_inference_ms == 0.0 else (1 - alpha) * self.avg_inference_ms + alpha * infer_ms,
                2
            )

            # Update FPS tracking
            now = time.time()
            self._fps_window.append(now)
            # Retain only timestamps from the last 2 seconds
            cutoff = now - 2.0
            self._fps_window = [t for t in self._fps_window if t > cutoff]
            if len(self._fps_window) > 1:
                duration = self._fps_window[-1] - self._fps_window[0]
                self.fps = round(len(self._fps_window) / max(0.1, duration), 1)

            # Update cached inspection states
            self.latest_hi_vis = round(float(hi_res.hi_vis), 4)
            self.latest_wire_breaks = hi_res.wire_breaks
            self.latest_wear_pct = round(float(hi_res.wear_pct), 4)
            self.latest_corrosion_pct = round(float(hi_res.corrosion_pct), 4)
            self.latest_discard_required = hi_res.is_discard_required
            self.latest_status = hi_res.status
            self.latest_detections = detections
            self.latest_annotated_frame = annotated

        # Publish vision telemetry message onto bus
        payload = {
            "stream": "camera",
            "timestamp": ts,
            "hi_vis": self.latest_hi_vis,
            "wire_breaks": self.latest_wire_breaks,
            "wear_pct": self.latest_wear_pct,
            "corrosion_pct": self.latest_corrosion_pct,
            "is_discard_required": self.latest_discard_required,
            "status": self.latest_status,
            "detections_count": len(detections),
            "detections": detections,
            "inference_ms": self.last_inference_ms,
            "fps": self.fps,
            "frame_shape": list(frame.shape)
        }
        self.publish("telemetry.vision", payload, metadata=meta)

    @property
    def is_stream_active(self) -> bool:
        """Return True if frames have been received within the timeout window."""
        with self._state_lock:
            if self.last_frame_timestamp is None:
                return False
            return (time.time() - self.last_frame_timestamp) < self.frame_timeout_sec

    def get_latest_inspection(self) -> Dict[str, Any]:
        """Return latest inspection metrics."""
        with self._state_lock:
            return {
                "hi_vis": self.latest_hi_vis,
                "wire_breaks": self.latest_wire_breaks,
                "wear_pct": self.latest_wear_pct,
                "corrosion_pct": self.latest_corrosion_pct,
                "is_discard_required": self.latest_discard_required,
                "status": self.latest_status,
                "detections_count": len(self.latest_detections),
                "detections": self.latest_detections,
                "inference_ms": self.last_inference_ms,
                "fps": self.fps,
                "stream_active": self.is_stream_active
            }

    def get_latest_annotated_frame(self) -> Optional[np.ndarray]:
        """Return the latest annotated frame with defect bounding boxes."""
        with self._state_lock:
            if self.latest_annotated_frame is not None:
                return self.latest_annotated_frame.copy()
            return None

    def get_status(self) -> Dict[str, Any]:
        """Extended status including Stream 1 camera statistics."""
        base_status = super().get_status()
        with self._state_lock:
            base_status.update({
                "stream_active": self.is_stream_active,
                "total_frames_ingested": self.total_frames_ingested,
                "total_frames_processed": self.total_frames_processed,
                "dropped_frames": self.dropped_frames,
                "queue_backlog": self.frame_queue.qsize(),
                "fps": self.fps,
                "avg_inference_ms": self.avg_inference_ms,
                "latest_hi_vis": self.latest_hi_vis,
                "latest_wire_breaks": self.latest_wire_breaks,
                "is_discard_required": self.latest_discard_required
            })
        return base_status
