# ElevatorGuard-AI: Dual-Node Edge Firmware (ESP32-CAM & ESP32-S3 Hub)

This repository sub-project contains the complete edge firmware implementation for:
1. **Node A (`esp32_cam/`)**: AI-Thinker ESP32-CAM module serving video streams and snapshots over Web API.
2. **Node B (`esp32_s3_hub/`)**: ESP32-S3 Edge Hub collecting vibration data from an MPU-6050 accelerometer, ingesting the camera feed from Node A, and forwarding telemetry to the ElevatorGuard backend.

---

## 1. System Topology & Wiring

### Node A: ESP32-CAM
- **Hardware**: AI-Thinker ESP32-CAM with OV2640 sensor.
- **Flashing Wiring (via FTDI / USB-to-UART Adapter)**:
  - `GND` ➔ FTDI `GND`
  - `5V` (or `3V3`) ➔ FTDI `VCC` (5V recommended for stability with camera)
  - `U0R` (GPIO 3) ➔ FTDI `TX`
  - `U0T` (GPIO 1) ➔ FTDI `RX`
  - **Flash Mode**: Connect `GPIO 0` to `GND` before powering on or pressing RESET. Disconnect `GPIO 0` after uploading to run.

### Node B: ESP32-S3 DevKit
- **Hardware**: ESP32-S3-DevKitC-1 with HW-072 (SW-420) digital vibration sensor module.
- **Vibration Sensor Wiring (HW-072)**:
  - `VCC` ➔ `3V3` (or `5V`)
  - `GND` ➔ `GND`
  - `DO` (Digital Out) ➔ `GPIO 8`
- **Sensitivity Adjustment**:
  - The HW-072 module features an onboard blue potentiometer. Use a small screwdriver to turn the potentiometer clockwise or counter-clockwise until the indicator LED flashes reliably upon physical vibration or tapping.

---

## 2. Configuration

### Configuring ESP32-CAM (`esp32_cam/esp32_cam_config.h`)
Edit your WiFi credentials:
```c
#define WIFI_SSID       "Your_WiFi_Network"
#define WIFI_PASSWORD   "Your_WiFi_Password"
```

### Configuring ESP32-S3 (`esp32_s3_hub/s3_config.h`)
Set the WiFi credentials and point to the ESP32-CAM's IP address:
```c
#define WIFI_SSID             "Your_WiFi_Network"
#define WIFI_PASSWORD         "Your_WiFi_Password"

#define ESP32_CAM_CAPTURE_URL "http://192.168.1.150/capture"
#define BACKEND_API_URL       "http://192.168.1.100:8000/api/vision/inspect"
```

---

## 3. How to Build & Flash

### Option A: Using PlatformIO (Recommended)
From this directory:
```bash
# Upload to ESP32-CAM
cd esp32_cam
pio run -t upload

# Upload to ESP32-S3
cd ../esp32_s3_hub
pio run -t upload
```

### Option B: Using Arduino IDE
1. Open Arduino IDE:
   - For ESP32-CAM: Open `esp32_cam/esp32_cam.ino`. Select Board: `AI Thinker ESP32-CAM`. Set `PSRAM: Enabled`, `Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS)`.
   - For ESP32-S3: Open `esp32_s3_hub/esp32_s3_hub.ino`. Select Board: `ESP32S3 Dev Module`. Set `PSRAM: OPI PSRAM`.
2. Hit **Upload**.

---

## 4. Web API Endpoints

### ESP32-CAM Endpoints
- `GET /` : Interactive web preview dashboard.
- `GET /stream` : Live HTTP MJPEG continuous video stream (`multipart/x-mixed-replace`).
- `GET /capture` : Instant JPEG snapshot capture.
- `GET /status` : JSON telemetry (uptime, free heap, PSRAM, WiFi RSSI).
- `GET /flash?on=1` : Toggle onboard high-power flash LED.

### ESP32-S3 Serial Output
The ESP32-S3 continuously streams ElevatorGuard Protocol v2.1 packets at 115200 baud:
```text
$EG,{"type":"TELEMETRY","ver":"2.1","ts":14250,"acc":[-0.04,0.02,9.81],"rms_z":0.015,"alarm":0}*4F
```

---

## 5. Verification Tool

Run the provided Python test utility to test the camera feed and capture functionality:
```bash
python test_rig_simulator.py --cam-ip <ESP32_CAM_IP> --stream
```
