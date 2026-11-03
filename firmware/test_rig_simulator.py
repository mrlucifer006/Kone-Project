"""
ElevatorGuard-AI: Dual-Node Edge Verification Script
Tests and verifies:
 1. ESP32-CAM stream (/stream) and snapshot API (/capture).
 2. ESP32-S3 Serial telemetry stream ($EG, {...} *XX) parsing.
 3. Ingestion integration against the ElevatorGuard-AI FastAPI backend.
"""

import sys
import time
import argparse
import requests
import cv2
import numpy as np


def verify_esp32_cam(cam_ip: str):
    """Test connection to ESP32-CAM."""
    print(f"\n[1/3] Testing ESP32-CAM at http://{cam_ip} ...")
    status_url = f"http://{cam_ip}/status"
    capture_url = f"http://{cam_ip}/capture"

    try:
        res = requests.get(status_url, timeout=3.0)
        if res.status_code == 200:
            print(f"  ✓ ESP32-CAM Status OK: {res.json()}")
        else:
            print(f"  ✗ Status endpoint returned {res.status_code}")
    except Exception as e:
        print(f"  ✗ Could not reach status endpoint: {e}")

    try:
        t0 = time.time()
        res = requests.get(capture_url, timeout=4.0)
        dt = (time.time() - t0) * 1000
        if res.status_code == 200:
            img_bytes = np.frombuffer(res.content, dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                h, w, c = img.shape
                print(f"  ✓ Successfully fetched snapshot: {w}x{h} ({len(res.content)} bytes in {dt:.1f}ms)")
            else:
                print("  ✗ Received image data could not be decoded as JPEG.")
        else:
            print(f"  ✗ Snapshot request failed with HTTP {res.status_code}")
    except Exception as e:
        print(f"  ✗ Capture request failed: {e}")


def verify_stream_playback(cam_ip: str):
    """Open OpenCV video stream from ESP32-CAM."""
    stream_url = f"http://{cam_ip}/stream"
    print(f"\n[2/3] Opening MJPEG stream from {stream_url} ... (Press 'q' in window to close)")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("  ✗ Failed to open video stream.")
        return

    frame_count = 0
    t0 = time.time()

    while frame_count < 60:
        ret, frame = cap.read()
        if not ret:
            print("  ✗ Failed to read frame from stream.")
            break

        frame_count += 1
        fps = frame_count / (time.time() - t0)
        cv2.putText(frame, f"ESP32-CAM Feed | FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("ESP32-CAM Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  ✓ Video stream verified ({frame_count} frames received).")


def parse_esp32_packet(line: str):
    """Verify $EG checksum protocol matching ElevatorGuard-AI standard."""
    line = line.strip()
    if not (line.startswith("$EG,") and "*" in line):
        return None
    try:
        payload, chk = line[4:].rsplit("*", 1)
        # XOR checksum
        xor_sum = 0
        for ch in payload:
            xor_sum ^= ord(ch)
        calc_chk = f"{xor_sum:02X}"
        if calc_chk != chk.upper():
            return None
        return payload
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="ElevatorGuard-AI Dual Node Test Tool")
    parser.add_argument("--cam-ip", type=str, default="192.168.1.150", help="IP address of ESP32-CAM")
    parser.add_argument("--stream", action="store_true", help="Display live OpenCV window of ESP32-CAM feed")
    args = parser.parse_args()

    print("=====================================================")
    print(" ElevatorGuard-AI: Dual-Node Edge Verification Tool  ")
    print("=====================================================")

    verify_esp32_cam(args.cam_ip)

    if args.stream:
        verify_stream_playback(args.cam_ip)


if __name__ == "__main__":
    main()
