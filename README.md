# ElevatorGuard-AI: Multi-Modal Edge Pipeline for Elevator Wire Rope Defect Quantification & Dynamic RUL Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-teal)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![ESP32](https://img.shields.io/badge/Hardware-ESP32%20Protocol%20v2.1-red)
![Status](https://img.shields.io/badge/Status-Completed%20%26%20Verified-brightgreen)

---

## 🛗 Project Summary

**ElevatorGuard-AI** is an industrial-grade, multi-modal predictive maintenance pipeline and digital twin simulation developed for elevator suspension systems (wire ropes, sheaves, and tension equalizers).

Current elevator safety regulations (EN 81-20/50, ASME A17.1) rely on periodic calendar inspections by field technicians, risking either undetected subsurface fatigue failures or premature scrapping of safe cables. **ElevatorGuard-AI** bridges this gap by merging continuous multi-sensor telemetry with computer vision non-destructive testing (NDT) on edge microcontrollers.

---

## 🌟 Key Architecture & Streams

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION LAYER                          │
├───────────────────────────────────┬────────────────────────────────────┤
│ Hoistway NDT Camera (YOLO)        │ ESP32 Multi-Sensor Telemetry Node  │
│ (Broken Wires, Wear, Corrosion)   │ (MPU-6050 Accel, HX711 Load Cells) │
└─────────────────┬─────────────────┴──────────────────┬─────────────────┘
                  │                                    │
                  ▼                                    ▼
┌───────────────────────────────────┐ ┌──────────────────────────────────┐
│   STREAM 1: COMPUTER VISION NDT   │ │ STREAM 2: DYNAMIC RUL TELEMETRY  │
├───────────────────────────────────┤ ├──────────────────────────────────┤
│ • YOLOv8-Seg / Feature Extractor  │ │ • Time-Frequency Feature Eng.    │
│ • Detects: Broken Wires, Corrosion│ │ • BiLSTM + Bahdanau Attention    │
│ • Visual Health Index (HI_vis)    │ │ • Dynamic Health Index (HI_dyn)  │
│ • ISO 4309 Discard Criterion      │ │ • ISO 18738 Ride Quality Alerts  │
└─────────────────┬─────────────────┘ └────────────────┬─────────────────┘
                  │                                    │
                  └─────────────────┬──────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STREAM 3: MULTI-MODAL FUSION & RUL ENGINE                  │
├────────────────────────────────────────────────────────────────────────┤
│ • Composite Health: HI_comp = α * HI_vis + (1 - α) * HI_dyn            │
│ • Safety Override: ISO 4309 Discard Limit Clamping (N_break >= 6 in 6d)│
│ • Prognostics: Cycles and Days Remaining Useful Life (Miner's Rule)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT & SIMULATION                          │
├───────────────────────────────────┬────────────────────────────────────┤
│ 🎮 2D Pygame Visual Lift Twin     │ 📊 Streamlit Real-Time Dashboard   │
│ 🚀 FastAPI REST + WebSocket API   │ ⚡ ESP32 Protocol v2.1 Serial Edge │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── config.yaml                     # System, elevator, and sensor threshold configs
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive guide and execution instructions
├── solution.md                     # Top 10 paper review and project selection matrix
├── LPR.md                          # Detailed Laboratory Project Report (specifications & math)
├── data/
│   ├── annotated_yolo/             # Paired YOLO images & bounding box labels
│   ├── synthetic_degradation/      # Run-to-failure fatigue telemetry datasets
│   └── cable_defects.yaml          # YOLO dataset configuration
├── models/
│   ├── bilstm_rul.pt               # Trained BiLSTM PyTorch weights
│   ├── bilstm_rul.torchscript.pt   # Traced TorchScript for edge deployment
│   └── bilstm_rul.onnx             # Quantized ONNX model
├── src/
│   ├── vision/
│   │   ├── visual_hi.py            # ISO 4309 Visual Health Index calculator
│   │   ├── defect_analyzer.py      # YOLO NDT cable inspection & HUD overlay
│   │   └── train_detector.py       # YOLO fine-tuning pipeline
│   ├── telemetry/
│   │   ├── feature_extraction.py   # FFT, RMS, Kurtosis, Tension Imbalance
│   │   ├── model_bilstm.py         # BiLSTM with Attention PyTorch model
│   │   ├── dataset_loader.py       # Sliding window time-series data loader
│   │   └── train_rul.py            # RUL regression training loop
│   ├── fusion/
│   │   ├── multimodal_fusion.py    # Multi-modal decision fusion engine
│   │   └── rul_estimator.py        # Lifecycle prognostics solver
│   ├── edge/
│   │   ├── esp32_protocol.py       # Standardized ESP32 frame parser ($EG,...*XX)
│   │   ├── serial_ingest.py        # Ingestion service (Hardware UART & Virtual)
│   │   ├── export_onnx.py          # Model exporter for edge runtime
│   │   └── esp32_firmware/
│   │       ├── esp32_firmware.ino  # Production ESP32 Arduino C++ firmware
│   │       └── main.cpp            # PlatformIO entry point
│   └── simulation/
│       ├── elevator_physics.py     # 8-floor elevator kinematics & tension physics
│       ├── esp32_simulator.py      # Virtual ESP32 hardware daemon
│       ├── dataset_generator.py    # Synthetic image & fatigue data generator
│       ├── visual_lift_gui.py      # Interactive 2D graphical hoistway simulation
│       └── run_demo.py             # Single-command demonstration runner
├── dashboard/
│   ├── app.py                      # Multi-tab Streamlit dashboard
│   └── api.py                      # FastAPI REST & WebSocket backend
└── tests/
    ├── test_vision.py              # Unit tests for ISO 4309 and vision
    ├── test_telemetry.py           # Unit tests for signal features & BiLSTM
    ├── test_fusion.py              # Unit tests for decision fusion & RUL
    ├── test_esp32_protocol.py      # Unit tests for packet framing & CRC
    └── test_simulation.py          # Unit tests for physics kinematics
```

---

## ⚡ Quickstart & How to Run

### 1. Set Up Environment
```bash
# Clone and enter directory
cd "KONE project"

# Activate Python virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Interactive Lift Simulation Demonstration
You can run the end-to-end demonstration directly:

* **Graphical 2D Visual Mode (Pygame window with live Hoistway, NDT Camera HUD, and ESP32 Telemetry):**
  ```bash
  python3 src/simulation/run_demo.py --mode gui
  ```
  *Keybindings:*
  * `[1]` to `[8]`: Dispatch elevator to Floor 1 through 8
  * `[B]`: Inject crown wire breaks
  * `[T]`: Inject rope tension imbalance (>15%)
  * `[E]`: Engage emergency safety brake
  * `[C]`: Clear alarms and reset
  * `[Q]` / `[ESC]`: Exit

* **Headless Console Mode (Terminal live HUD):**
  ```bash
  python3 src/simulation/run_demo.py --mode headless --duration 20
  ```

### 3. Launch the Streamlit Multi-Modal Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501` to explore:
* 🏢 **Hoistway Digital Twin:** Animated elevator car, floor tracking, rope tension meters.
* 👁️ **Computer Vision NDT Hub:** Upload cable images or inspect live camera feeds with YOLO detections and ISO 4309 discard verification.
* 📈 **Telemetry & Vibration:** Live 3-axis vibration waveforms ($a_x, a_y, a_z$), FFT spectra, and tension imbalance alerts.
* 🧠 **Prognostics & RUL Forecast:** Projected degradation curve and maintenance schedule.
* ⚡ **ESP32 Protocol Explorer:** Live packet terminal, payload inspector, and hardware wiring pinout.

### 4. Launch the FastAPI Edge Microservice
```bash
uvicorn dashboard.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### 5. Run the Automated Unit Test Suite
```bash
pytest tests -v
```
All 17 automated tests verify vision algorithms, ISO 4309 discard thresholds, BiLSTM neural network forward passes, ESP32 checksum verification, and hoistway physics kinematics.

---

## 🔌 Hardware Integration: ESP32 Protocol v2.1

The ESP32 microcontroller firmware is located in `src/edge/esp32_firmware/esp32_firmware.ino`.

### Sensor Pinout:
* **MPU-6050 (IMU):** I2C `SDA=GPIO21`, `SCL=GPIO22`
* **HX711 (Rope Tension Load Cells):** `SCK=GPIO14`, `DT1=GPIO27`, `DT2=GPIO26`, `DT3=GPIO25`, `DT4=GPIO33`
* **Optical Cable Revolution Sensor:** `GPIO19` (Interrupt Rising Edge)
* **RGB Status NeoPixel:** `GPIO18`
* **Audible Piezo Buzzer:** `GPIO23`
* **Serial UART:** 115200 Baud (8N1)

### Packet Specification:
Telemetry frames use NMEA-standard XOR checksum framing:
```
$EG,{"type":"TELEMETRY","ver":"2.1","ts":1729701234,"flr":3,"vel":1.50,"acc":[0.02,-0.01,9.81],"tns":[2480.0,2510.0,2495.0,2515.0],"dr":0,"opt":1450,"flg":0}*4B\r\n
```

---

## 📐 Mathematical Standards Implemented
1. **ISO 4309 Wire Rope Discard Standard:**
   $$HI_{vis} = 1.0 - \left( w_1 \frac{\min(N_{break}, N_{max})}{N_{max}} + w_2 W_{pct} + w_3 C_{sev} \right)$$
   * $N_{max} = 6$ wire breaks in 6-diameter length
   * Wear limit = 10% diameter reduction
   * Corrosion limit = 15% surface area coverage
2. **Dynamic BiLSTM-Attention Prognostics:**
   Ingests tri-axial vibration and rope tensions to predict $HI_{dyn} \in [0, 1]$.
3. **Multi-Modal Decision Fusion:**
   $$HI_{comp} = \alpha \cdot HI_{vis} + (1 - \alpha) \cdot HI_{dyn}$$
4. **Remaining Useful Life (RUL):**
   $$RUL_{cycles} = \frac{HI_{comp} - HI_{discard}}{\left| \frac{d(HI)}{dt} \right|}$$
