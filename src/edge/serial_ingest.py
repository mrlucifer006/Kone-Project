"""
ElevatorGuard-AI: Serial & Telemetry Ingestion Service
Manages real-time data streaming from physical ESP32 or virtual simulation bridge.
"""

import threading
import time
import queue
from typing import Optional, Callable, Dict, Any, List
from collections import deque
import numpy as np

try:
    import serial
except ImportError:
    serial = None

from src.edge.esp32_protocol import ElevatorGuardESP32Protocol, PacketType


class ESP32TelemetryIngestor:
    """
    Ingestion service for ESP32 telemetry frames.
    Supports physical serial hardware and virtual socket/queue mock for testing.
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        buffer_size: int = 512,
        virtual_mode: bool = True
    ):
        self.port = port
        self.baudrate = baudrate
        self.virtual_mode = virtual_mode
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._serial_conn = None

        # Ring buffer for recent sensor readings
        self.history_len = buffer_size
        self.timestamps = deque(maxlen=buffer_size)
        self.accel_x = deque(maxlen=buffer_size)
        self.accel_y = deque(maxlen=buffer_size)
        self.accel_z = deque(maxlen=buffer_size)
        self.tensions = deque(maxlen=buffer_size) # List of 4 ropes
        self.floors = deque(maxlen=buffer_size)
        self.velocities = deque(maxlen=buffer_size)

        # Virtual injection queue for simulator
        self.virtual_queue: queue.Queue = queue.Queue(maxsize=1024)
        
        # Subscribed callbacks for live frames
        self.callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.lock = threading.Lock()

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]):
        """Register a callback invoked on each valid telemetry packet."""
        with self.lock:
            self.callbacks.append(cb)

    def start(self):
        """Start background ingestion thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[ESP32 Ingest] Ingestor started (Virtual Mode: {self.virtual_mode})")

    def stop(self):
        """Stop background ingestion."""
        self.running = False
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[ESP32 Ingest] Ingestor stopped.")

    def inject_virtual_packet(self, packet_str: str):
        """Inject an ESP32 protocol string directly into the virtual stream."""
        if not self.virtual_queue.full():
            self.virtual_queue.put_nowait(packet_str)

    def send_command(self, cmd: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Send a command packet to the ESP32 hardware."""
        packet_str = ElevatorGuardESP32Protocol.encode_command_packet(cmd, params)
        if self.virtual_mode:
            print(f"[ESP32 Ingest] [VIRTUAL CMD SENT]: {cmd} {params or {}}")
            return True
        elif self._serial_conn and self._serial_conn.is_open:
            try:
                self._serial_conn.write(packet_str.encode("utf-8"))
                return True
            except Exception as e:
                print(f"[ESP32 Ingest] Serial write error: {e}")
                return False
        return False

    def _run_loop(self):
        """Main ingestion loop."""
        while self.running:
            if self.virtual_mode:
                try:
                    packet_str = self.virtual_queue.get(timeout=0.1)
                    self._process_raw_line(packet_str)
                except queue.Empty:
                    time.sleep(0.01)
            else:
                self._handle_serial_hardware()

    def _handle_serial_hardware(self):
        """Connect to physical ESP32 via serial."""
        if serial is None:
            print("[ESP32 Ingest] pyserial not available. Switching to virtual mode.")
            self.virtual_mode = True
            return

        if self._serial_conn is None or not self._serial_conn.is_open:
            try:
                self._serial_conn = serial.Serial(self.port, self.baudrate, timeout=1.0)
                print(f"[ESP32 Ingest] Connected to hardware on {self.port}")
            except Exception as e:
                # Retry delay
                time.sleep(1.0)
                return

        try:
            line = self._serial_conn.readline().decode("utf-8", errors="ignore")
            if line:
                self._process_raw_line(line)
        except Exception as e:
            print(f"[ESP32 Ingest] Hardware read error: {e}")
            if self._serial_conn:
                self._serial_conn.close()
                self._serial_conn = None
            time.sleep(1.0)

    def _process_raw_line(self, line: str):
        """Parse packet and update ring buffers."""
        is_valid, data, err = ElevatorGuardESP32Protocol.decode_packet(line)
        if not is_valid or not data:
            return

        if data.get("type") == PacketType.TELEMETRY.value:
            ts = data.get("ts", int(time.time() * 1000))
            flr = data.get("flr", 1)
            vel = data.get("vel", 0.0)
            acc = data.get("acc", [0.0, 0.0, 9.81])
            tns = data.get("tns", [2500.0, 2500.0, 2500.0, 2500.0])

            with self.lock:
                self.timestamps.append(ts)
                self.floors.append(flr)
                self.velocities.append(vel)
                self.accel_x.append(acc[0])
                self.accel_y.append(acc[1])
                self.accel_z.append(acc[2])
                self.tensions.append(tns)

            # Fire callbacks
            for cb in list(self.callbacks):
                try:
                    cb(data)
                except Exception as e:
                    print(f"[ESP32 Ingest] Callback error: {e}")

    def get_latest_window(self, window_size: int = 128) -> Dict[str, np.ndarray]:
        """Retrieve the most recent window of sensor arrays for feature extraction."""
        with self.lock:
            n = min(window_size, len(self.timestamps))
            if n == 0:
                return {
                    "timestamps": np.array([]),
                    "accel_x": np.zeros(window_size, dtype=np.float32),
                    "accel_y": np.zeros(window_size, dtype=np.float32),
                    "accel_z": np.zeros(window_size, dtype=np.float32) + 9.81,
                    "tensions": np.ones((window_size, 4), dtype=np.float32) * 2500.0,
                    "floor": 1,
                    "velocity": 0.0
                }

            ax = list(self.accel_x)[-n:]
            ay = list(self.accel_y)[-n:]
            az = list(self.accel_z)[-n:]
            tns = list(self.tensions)[-n:]
            cur_flr = self.floors[-1] if self.floors else 1
            cur_vel = self.velocities[-1] if self.velocities else 0.0

            return {
                "timestamps": np.array(list(self.timestamps)[-n:]),
                "accel_x": np.array(ax, dtype=np.float32),
                "accel_y": np.array(ay, dtype=np.float32),
                "accel_z": np.array(az, dtype=np.float32),
                "tensions": np.array(tns, dtype=np.float32),
                "floor": cur_flr,
                "velocity": cur_vel
            }
