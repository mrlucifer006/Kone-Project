# Laboratory Project Report (LPR) & Specification Document

# ElevatorGuard-AI: Multi-Modal Edge Pipeline for Elevator Wire Rope Defect Quantification and Dynamic Tension-Vibration Remaining Useful Life (RUL) Prediction

---

## Document Control & Metadata
- **Project Name:** ElevatorGuard-AI (KONE Capstone Research Track)
- **Directory:** `/run/media/mrlucifer006/346E71E66E71A0F2/Personal Projects/KONE project`
- **Target Audience:** Undergraduate Computer Science & Engineering (UG CSE) Final-Year Project / Academic Conference Submission
- **Domain:** Structural Health Monitoring (SHM), Prognostics and Health Management (PHM), Edge AI, Computer Vision, Deep Learning for Time-Series
- **Author:** UG Research Team & Antigravity Systems
- **Date:** October 2026

---

## 1. Project Overview & Motivation

### 1.1 Problem Statement
Traction elevator suspension ropes (typically 8–16 mm multi-strand steel wire ropes or polyurethane-coated steel belts) bear the deadweight of the elevator car, counterweight, and passenger payload. Over millions of round trips, suspension components undergo:
1. **Bending Fatigue:** Continuous reverse bending over the traction sheave and deflection pulleys initiates micro-cracking across outer and crown wires.
2. **Abrasion & Loss of Metallic Area (LMA):** Mechanical rubbing between adjacent strands and traction grooves thins the rope diameter.
3. **Fretting Corrosion & Moisture Pitting:** High humidity in elevator hoistways accelerates oxidation, weakening load-carrying strands.
4. **Tension Imbalances:** Unequal stretch across parallel suspension ropes concentrates excessive load onto fewer cables, triggering premature degradation.

Conventional maintenance relies on manual visual inspections by technicians using calipers at calendar intervals (e.g., every 6 or 12 months). This methodology is prone to human error, cannot track gradual micro-wear, and leads either to sudden component failure between inspection cycles or premature scrapping of safe cables.

### 1.2 Proposed AI Solution
**ElevatorGuard-AI** is a multi-modal, dual-stream predictive maintenance framework designed for edge execution:
* **Stream 1 (Vision NDT):** A lightweight convolutional object detection and segmentation network (YOLOv8-Seg / YOLOv11-Nano) running on edge camera feeds pointed at the hoistway cable run. It quantifies surface wire breaks, abrasion ratio, and corrosion percentage.
* **Stream 2 (Vibration & Tension Telemetry):** A sequence model (Bidirectional LSTM with Self-Attention) that ingests continuous tri-axial accelerometer data from the car frame and rope tension measurements, generating a continuous **Health Index (HI)** curve.
* **Stream 3 (Multi-Modal Decision Fusion):** An ensemble fusion layer merging visual structural damage metrics with dynamic vibrational strain to estimate **Remaining Useful Life (RUL)** in terms of operational cycles and days until the safety discard threshold (ISO 4309 standards).

---

## 2. System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DATA ACQUISITION LAYER                                 │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ Optical Video Stream                     │ Multi-Sensor Telemetry Time-Series          │
│ (Hoistway Rope Inspection Camera / NDT)  │ (Tri-Axial Accelerometer, Rope Tension Load)│
└────────────────────┬─────────────────────┴──────────────────────┬──────────────────────┘
                     │                                            │
                     ▼                                            ▼
┌──────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│     STREAM 1: COMPUTER VISION NDT        │ │      STREAM 2: TIME-SERIES PROGNOSTICS    │
├──────────────────────────────────────────┤ ├───────────────────────────────────────────┤
│ • Frame Extraction & Adaptive CLAHE      │ │ • Sliding Window Segmentation (256 steps) │
│ • YOLOv8-Seg Inference                  │ │ • Feature Extraction: RMS, Kurtosis, FFT  │
│ • Detect: Broken Wires, Wear, Corrosion  │ │ • BiLSTM with Bahdanau Attention          │
│ • Metric: Visual Health Index ($HI_{vis}$)│ │ • Metric: Dynamic Health Index ($HI_{dyn}$)│
└────────────────────┬─────────────────────┘ └────────────────────┬──────────────────────┘
                     │                                            │
                     └─────────────────────┬──────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STREAM 3: MULTI-MODAL FUSION & RUL ENGINE                       │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • Weighted Decision Fusion: $HI_{comp} = \alpha \cdot HI_{vis} + (1-\alpha) \cdot HI_{dyn}$    │
│ • Failure Discard Threshold Enforcement (ISO 4309: Wire break count $\ge 6$ in $6d$)    │
│ • RUL Regression: $RUL = \text{Time}(\text{Critical Threshold} - HI_{comp}) / \frac{d(HI)}{dt}$│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              EDGE DEPLOYMENT & DASHBOARD                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ • ONNX Runtime / TensorRT Edge Engine                                                  │
│ • FastAPI Backend + Streamlit Real-Time Monitoring UI                                  │
│ • Email / SMS Webhook Alerts for High-Risk Rope Tension / Imminent Discard             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Foundations

### 3.1 Visual Degradation Index ($HI_{vis}$)
In compliance with international standard **ISO 4309: Cranes and Elevators — Wire Ropes — Care and Maintenance, Inspection and Discard**, degradation is quantified by broken wires ($N_{break}$), local cross-section wear ($W_{pct}$), and corrosion severity ($C_{sev}$):

$$HI_{vis} = 1.0 - \left( w_1 \cdot \frac{\min(N_{break}, N_{max})}{N_{max}} + w_2 \cdot W_{pct} + w_3 \cdot C_{sev} \right)$$

Where:
* $N_{max} = 6$ (Standard discard limit for wire breaks in a 6-diameter length for regular lay ropes).
* $w_1 = 0.5$, $w_2 = 0.3$, $w_3 = 0.2$ are normalized severity weights.
* $HI_{vis} \in [0.0, 1.0]$, where $1.0$ represents a pristine new cable and $0.0$ triggers immediate mandatory replacement.

### 3.2 Dynamic Vibration & Tension Degradation
From tri-axial vibration signals $a_x(t), a_y(t), a_z(t)$, Root Mean Square (RMS) acceleration and Spectral Kurtosis ($SK$) are computed over sliding time windows:

$$RMS = \sqrt{\frac{1}{N} \sum_{i=1}^{N} a(t_i)^2}$$

$$Kurtosis = \frac{\frac{1}{N} \sum_{i=1}^{N} (a(t_i) - \mu)^4}{\left( \frac{1}{N} \sum_{i=1}^{N} (a(t_i) - \mu)^2 \right)^2}$$

The BiLSTM-Attention network predicts the monotonic degradation state:

$$h_t = \text{BiLSTM}(X_t, h_{t-1})$$
$$e_t = \tanh(W_a h_t + b_a), \quad \alpha_t = \frac{\exp(e_t^T u_a)}{\sum_k \exp(e_k^T u_a)}$$
$$c = \sum_{t} \alpha_t h_t$$
$$HI_{dyn} = \sigma(W_c c + b_c)$$

### 3.3 Multi-Modal Fusion & RUL Calculation
The composite health index $HI_{comp}$ merges both streams:

$$HI_{comp} = \alpha \cdot HI_{vis} + (1 - \alpha) \cdot HI_{dyn}$$

Given current operational trip count $T_{current}$ and health degradation slope $\frac{d(HI)}{dt}$, the Remaining Useful Life ($RUL$) in remaining elevator runs is estimated by:

$$RUL_{cycles} = \frac{HI_{comp} - HI_{discard}}{\left| \frac{d(HI)}{dt} \right|}$$

Where $HI_{discard} = 0.20$ (the 80% cumulative damage limit).

---

## 4. Hardware and Software Specifications

### 4.1 Software Stack
* **Programming Language:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch 2.x, Ultralytics YOLOv8 / YOLOv11
* **Signal Processing:** NumPy, SciPy (Signal & FFT), PyWavelets, Pandas
* **Model Export & Edge Acceleration:** ONNX Runtime, OpenVINO
* **API & Frontend:** FastAPI, Streamlit, Plotly
* **Development Environment:** Linux / Ubuntu (compatible with CUDA 12.x or CPU execution)

### 4.2 Hardware Requirements (Low-Cost / UG Accessible)
* **Training & Development Workstation:**
  * Minimum: 8 GB RAM, Intel i5/AMD Ryzen 5, NVIDIA GPU (GTX 1650 or higher) or Google Colab Free Tier.
* **Optional Benchtop Hardware Prototype (<$50 total):**
  * Microcontroller: ESP32 NodeMCU ($5)
  * Accelerometer: MPU-6050 6-DOF IMU ($2)
  * Load Cell & ADC: 5 kg Micro Load Cell + HX711 Amplifier Module ($4)
  * Optical Camera: USB Webcam or ESP32-CAM ($6)
  * Mechanical Rig: 3D-printed or acrylic miniature pulley and nylon/braided steel wire cable counterweight model.

---

## 5. Dataset Strategy

To ensure zero reliance on hazardous industrial equipment, the project leverages three verified open datasets:

1. **Roboflow & Mendeley Steel Wire Rope Defect Dataset:**
   * ~4,500 high-resolution images annotated for wire breaks, strand peeling, pitting corrosion, and outer wear.
   * Format: YOLO TXT annotations with bounding boxes and polygon masks.
2. **Figshare Elevator Car Vibration & Trip Telemetry:**
   * Tri-axial accelerometer time-series captured from operational traction elevators over normal operation, passenger loading, and degraded guide rail conditions.
3. **Synthetic Cumulative Fatigue Degradation Simulator:**
   * A Python script simulating multi-cycle elevator usage under varying load distributions and generating continuous run-to-failure vibration/tension curves based on **Miner’s rule** ($\sum \frac{n_i}{N_i} = 1$).

---

## 6. Project Directory Structure

```
KONE project/
├── README.md                      # Quickstart and overview
├── solution.md                    # Literature survey & top 10 papers review
├── LPR.md                         # Detailed project report & technical specs
├── requirements.txt               # Python dependencies
├── config.yaml                    # Hyperparameters and threshold configs
├── data/
│   ├── raw_images/                # Raw cable inspection images
│   ├── annotated_yolo/            # YOLO formatted dataset (train/val/test)
│   ├── vibration_telemetry/       # Vibration CSV logs
│   └── synthetic_degradation/     # Simulated run-to-failure fatigue cycles
├── src/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── train_detector.py      # YOLO fine-tuning pipeline
│   │   ├── defect_analyzer.py     # Broken wire counting & area estimation
│   │   └── visual_hi.py           # ISO 4309 Visual Health Index calculator
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py  # FFT, RMS, Kurtosis, Wavelet transform
│   │   ├── dataset_loader.py      # Sliding window time-series loader
│   │   ├── model_bilstm.py        # BiLSTM with Attention PyTorch model
│   │   └── train_rul.py           # Training loop for continuous RUL regression
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── multimodal_fusion.py   # Decision fusion layer
│   │   └── rul_estimator.py       # Cycle & days remaining useful life solver
│   └── edge/
│       ├── __init__.py
│       ├── export_onnx.py         # Model quantization and ONNX export
│       └── serial_ingest.py       # ESP32 / Arduino sensor stream receiver
├── dashboard/
│   ├── app.py                     # Streamlit real-time monitoring UI
│   └── api.py                     # FastAPI REST API serving live inferences
└── tests/
    ├── test_vision.py             # Unit tests for defect segmentation
    ├── test_telemetry.py          # Unit tests for feature extraction
    └── test_fusion.py             # Unit tests for RUL math formulas
```

---

## 7. Step-by-Step Implementation Roadmap (10-Week Plan)

| Week | Phase | Key Milestone & Deliverables |
|:---:|---|---|
| **Week 1** | Problem Formulation & Environment Setup | Set up Git repository, install PyTorch/CUDA/Ultralytics, create directory scaffold. |
| **Week 2** | Vision Data Acquisition & Labeling | Download Roboflow & Mendeley wire rope datasets; verify YOLO bounding boxes & masks. |
| **Week 3** | Vision Model Training (YOLOv8/11) | Train YOLOv8-Nano on cable defects; achieve mAP@50 $\ge 85\%$; validate inference speed. |
| **Week 4** | Visual Health Index Implementation | Implement ISO 4309 discard logic; convert broken wire count and wear ratio to $HI_{vis}$. |
| **Week 5** | Vibration & Telemetry Preprocessing | Ingest Figshare elevator vibration logs; extract time/frequency features (RMS, Kurtosis, FFT). |
| **Week 6** | Synthetic RUL Generator & Fatigue Curve | Build Python simulation for cyclic tension fatigue based on Miner’s cumulative damage rule. |
| **Week 7** | BiLSTM-Attention Model Training | Train temporal sequence model for RUL regression; evaluate RMSE and Mean Absolute Error. |
| **Week 8** | Multi-Modal Fusion Engine | Implement confidence-weighted fusion equation combining $HI_{vis}$ and $HI_{dyn}$. |
| **Week 9** | Edge Optimization & Hardware Integration | Export models to ONNX; test optional ESP32/webcam live stream or replay telemetry data. |
| **Week 10** | Dashboard & Final Documentation | Build Streamlit dashboard; run end-to-end integration tests; finalize project report. |

---

## 8. Expected Results & Evaluation Benchmarks

1. **Vision Inspection Metrics:**
   * Mean Average Precision ($mAP_{50}$): $\ge 0.88$ for broken wires and surface corrosion.
   * Frame Rate: $\ge 45\text{ FPS}$ on standard laptop GPU / $\ge 20\text{ FPS}$ on modern quad-core CPU.
2. **Telemetry RUL Regression Metrics:**
   * Root Mean Square Error ($RMSE$): $\le 8.5\%$ on normalized RUL lifecycle test curves.
   * Score Function (PHM benchmark metric penalizing late predictions): Score $\le 25.0$.
3. **Safety Discard Reliability:**
   * $100\%$ detection rate for cables reaching ISO 4309 discard criteria ($N_{break} \ge 6$).
   * Early warning trigger provided at least $15\%$ of lifetime prior to critical failure.

---

## 9. Conclusion
The **ElevatorGuard-AI** project bridges the gap between theoretical predictive maintenance and actionable engineering. By combining computer vision defect detection with dynamic telemetry prognostics, it offers an undergraduate CSE student a practical, publishable, and highly relevant project in smart urban mobility and industrial AI.
