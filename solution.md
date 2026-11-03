# Comprehensive Literature Review & Project Selection: AI-Assisted Reliability Prediction of Elevator Suspension Components

**Project Folder:** `KONE project`  
**Domain:** Elevator Predictive Maintenance (PdM), Prognostics & Health Management (PHM), Structural Health Monitoring (SHM)  
**Target Level:** Undergraduate Computer Science & Engineering (UG CSE) Final Year / Capstone Project  
**Date:** October 2026 

---

## 1. Executive Summary & Objective

Elevator suspension systems (steel wire ropes, coated steel belts, traction sheaves, and tension equalizers) represent the most safety-critical sub-assemblies in vertical transportation. Current building codes (EN 81-20/50, ASME A17.1) mandate scheduled calendar inspections, frequently leading to either premature cable replacements or undetected subsurface fatigue failures.

This document compiles the **Top 10 academic papers, open benchmarks, and engineering projects** at the intersection of Artificial Intelligence, Non-Destructive Testing (NDT), and elevator suspension reliability. It details a multi-criteria scoring matrix tailored for an undergraduate CSE student, selects the optimal project, and outlines a comprehensive technical architecture that guarantees feasibility, academic rigor, and practical deployment.

---

## 2. Review of the Top 10 Papers & Projects

### 1. WireGC-Former: Point Cloud Graph-Transformer for Steel Wire Rope Defect Segmentation
* **Source/Venue:** *PLOS ONE / IEEE Transactions on Instrumentation and Measurement* (2024–2025)
* **Code/Data:** Open-source benchmark repository (`junyili1020-cmd/wiregc-former-review`)
* **Modality & Tech:** 3D Point Clouds, Graph Attention Networks (GAT), Spatial Transformers.
* **Core Concept:** Captures high-precision 3D geometry of twisted cable strands to segment broken wires, strand protrusions, and local diameter thinning.
* **Pros:** State-of-the-art structural flaw identification; robust against lighting variations.
* **Cons for UG CSE:** Requires 3D LiDAR/RGB-D camera capture; high GPU VRAM requirement; real-time inference on edge devices is difficult.

### 2. Deep Vision Defect Detection on Steel Wire Ropes (YOLO & Segmentation)
* **Source/Venue:** *Roboflow Universe & Mendeley Data Benchmark* (2023–2025)
* **Code/Data:** Publicly annotated datasets (~4,000+ labeled images: broken wires, corrosion, fraying).
* **Modality & Tech:** 2D Computer Vision (YOLOv8 / YOLOv11 / Mask R-CNN).
* **Core Concept:** Real-time object detection and segmentation of surface defects on moving cables captured via high-frame-rate hoistway inspection cameras.
* **Pros:** Highly accessible dataset; plug-and-play with PyTorch/Ultralytics; fast inference (real-time 60+ FPS on edge NPU).
* **Cons for UG CSE:** Limited to outer surface flaws; cannot capture internal core fatigue or continuous Remaining Useful Life (RUL) degradation curves without sensor telemetry.

### 3. Figshare Traction Elevator Car & Hoistway Vibration Anomaly Telemetry
* **Source/Venue:** *IoT Open Data / Scientific Data* (2023–2024)
* **Code/Data:** Figshare Public Dataset (Tri-axial accelerometer logs over thousands of trips).
* **Modality & Tech:** Time-series analysis, FFT/Wavelet transforms, Variational Autoencoders (VAEs), Isolation Forest.
* **Core Concept:** Monitors ride quality and mechanical vibration harmonics on the elevator car frame and sheaves to flag mechanical wear and rope slip.
* **Pros:** Real operational data; reflects actual lift dynamics and passenger loading transitions.
* **Cons for UG CSE:** Lacks ground-truth run-to-failure labels (RUL regression); predominantly focuses on anomaly detection rather than component-specific cable life estimation.

### 4. Magnetic Flux Leakage (MFL) NDT with 1D-CNN and BiLSTM
* **Source/Venue:** *NDT & E International / IEEE Transactions on Industrial Informatics* (2022–2024)
* **Code/Data:** Academic test-rig datasets (magnetizer + Hall effect sensors).
* **Modality & Tech:** 1D-CNN, Bidirectional LSTM, Hall effect voltage pulses.
* **Core Concept:** Detects Loss of Metallic Cross-sectional Area (LMA) and Localized Flaws (LF) inside multi-strand wire ropes through leakage magnetic flux analysis.
* **Pros:** The industrial standard for certifying wire rope safety; detects internal broken wires invisible to cameras.
* **Cons for UG CSE:** **Severe hardware bottleneck.** Requires heavy permanent neodymium magnets, custom Hall probe collars, and hazardous high-tension test rigs impossible to reproduce in a college lab.

### 5. Multi-Sensor IoT Elevator Fault Telemetry
* **Source/Venue:** *GitHub: `datasetengineer/ElevatorFaultDetection`*
* **Code/Data:** Open-source telemetry streams (Motor current, rope tension load cells, vibration, brake status).
* **Modality & Tech:** Multi-sensor time-series classification, XGBoost, Random Forest, Temporal Convolutional Networks (TCN).
* **Core Concept:** Ingests live telemetry from elevator controllers to detect rope tension imbalances across multiple ropes and drive sheave wear.
* **Pros:** Mirrors practical IoT lift maintenance setups; includes rope tension columns; pure software/data engineering workflow.
* **Cons for UG CSE:** Real run-to-failure records are sparse; needs synthetic degradation modeling to predict continuous RUL.

### 6. Physics-Informed Neural Networks (PINN) for Wire Rope Fatigue Life
* **Source/Venue:** *Mechanical Systems and Signal Processing / AttnPINN* (2024)
* **Code/Data:** Fatigue journal benchmarks & open PINN implementations.
* **Modality & Tech:** PyTorch, DeepXDE, Miner’s Cumulative Damage Rule, Paris-Erdogan crack propagation law.
* **Core Concept:** Constrains neural network loss functions with mechanical fatigue equations so the model accurately predicts cable life even with sparse failure samples.
* **Pros:** Mathematically sound; overcomes the problem of rare real-world failure data.
* **Cons for UG CSE:** Requires extensive domain knowledge in contact mechanics, bending stress over sheaves, and fracture mechanics.

### 7. Acoustic Emission (AE) & Wavelet Spectrogram Analysis for Cable Snaps
* **Source/Venue:** *Structural Health Monitoring (SAGE)* (2023)
* **Code/Data:** High-frequency acoustic emission event logs.
* **Modality & Tech:** Piezoelectric sensors (>50 kHz), Continuous Wavelet Transform (CWT), 2D-CNN classification.
* **Core Concept:** Detects microscopic stress waves released as individual internal steel wires crack under tensile loading.
* **Pros:** Earliest possible warning sign before macroscopic failure.
* **Cons for UG CSE:** High hardware barrier (Mega-sample/sec DAQ cards, expensive piezoelectric sensors, acoustic noise dampening).

### 8. Edge-AI Predictive Maintenance on Microcontrollers (ESP32/Raspberry Pi)
* **Source/Venue:** *GitHub: `faizahkureshi232/Predictive_maintenance`*
* **Code/Data:** Lightweight Python/C++ codebases with Flask web dashboard.
* **Modality & Tech:** Edge ML (TensorFlow Lite / Edge Impulse), MPU6050 accelerometer, load cell (HX711).
* **Core Concept:** Edge inference device mounted on a pulley frame detecting abnormal vibration frequencies and tension variations.
* **Pros:** Highly feasible, cost-effective (<$30 setup), tangible hardware-software integration ideal for engineering presentations.
* **Cons for UG CSE:** Existing implementations use trivial synthetic features and basic classifiers; lacks deep temporal RUL modeling.

### 9. Sheave-Rope Slip & Tension Imbalance Estimator via Motor Current Signature Analysis (MCSA)
* **Source/Venue:** *IEEE Access / Sensors* (2023–2024)
* **Code/Data:** Variable Frequency Drive (VFD) current sensor logs paired with optical rotary encoder pulses.
* **Modality & Tech:** MCSA, Cross-correlation, Temporal Convolutional Networks.
* **Core Concept:** Uses current ripples from the hoist motor combined with rope slip detection to infer groove wear on the traction sheave and uneven rope stretch.
* **Pros:** Non-invasive monitoring directly from the machine room without placing sensors on the moving car.
* **Cons for UG CSE:** Paired motor current and slip ground-truth data is rarely published openly by elevator OEMs (Otis/Schindler/KONE).

### 10. Degradation Transfer Learning for Component Remaining Useful Life (RUL)
* **Source/Venue:** *IEEE Transactions on Reliability* (2024)
* **Code/Data:** NASA C-MAPSS Turbofan / XJTU-SY Bearing Run-to-Failure benchmarks.
* **Modality & Tech:** Domain Adaptation, LSTM-Autoencoder, Transformer RUL Regressors.
* **Core Concept:** Uses benchmark run-to-failure datasets to establish baseline degradation trajectories and applies transfer learning to mechanical elevator vibration time series.
* **Pros:** Full run-to-failure degradation curves; standard evaluation metrics (RMSE, Score, MAPE); highly publishable.
* **Cons for UG CSE:** Domain shift between continuous turbofan/bearing wear and reciprocating cyclic elevator motion requires careful calibration.

---

## 3. Evaluation Criteria & Comparative Scoring Matrix

To identify the optimal project for an **Undergraduate CSE Student**, each approach is scored from 1 to 5 across four criteria:

1. **Feasibility (25%):** Can a UG student build and run the system within 8–12 weeks on standard laptop/cloud GPUs without specialized heavy machinery?
2. **Dataset Accessibility (30%):** Are public, high-quality, pre-labeled datasets available under open licenses?
3. **Domain Relevance (25%):** Does the work directly assess elevator suspension components (wire ropes, tension, wear, RUL)?
4. **Hardware Simplicity (20%):** Can the model be demonstrated virtually or with commodity microcontrollers (ESP32, webcam) without hazardous high-tension cable test rigs?

| # | Candidate Project / Architecture | Feasibility (25%) | Data Access (30%) | Relevance (25%) | Hardware (20%) | **Weighted Score (/5.0)** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | WireGC-Former (3D Point Cloud GAT) | 3 | 4 | 4 | 4 | **3.70** |
| **2** | **Computer Vision Rope Defect Detection (YOLOv8/11)** | **5** | **5** | **4** | **5** | **4.75** |
| 3 | Figshare Elevator Vibration Anomaly Detection | 4 | 4 | 4 | 5 | **4.20** |
| 4 | Magnetic Flux Leakage (MFL) NDT with 1D-CNN | 2 | 2 | 5 | 1 | **2.55** |
| 5 | IoT Multi-Sensor Elevator Fault Telemetry | 4 | 4 | 5 | 4 | **4.25** |
| 6 | Physics-Informed Neural Network (PINN) Fatigue | 3 | 3 | 5 | 5 | **3.90** |
| 7 | Acoustic Emission Wavelet Spectrogram RUL | 2 | 2 | 4 | 1 | **2.25** |
| 8 | Arduino/RPi Edge Predictive Maintenance System | 5 | 3 | 4 | 4 | **4.00** |
| 9 | Sheave Slip & Tension Imbalance with TCN | 3 | 2 | 5 | 4 | **3.35** |
| 10| Transfer Learning RUL on Degradation Benchmarks | 4 | 5 | 3 | 5 | **4.25** |

---

## 4. The Selected Project

### **Title:**
> **"ElevatorGuard-AI: Multi-Modal Edge Pipeline for Elevator Wire Rope Defect Quantification and Dynamic Tension-Vibration Remaining Useful Life (RUL) Prediction"**

### Why This Project is Ideal for an Undergraduate CSE Student:
1. **Safety & Zero Hazardous Equipment:** No need to handle heavy cable tension rigs or destroy high-strength cables. The project uses open visual defect datasets and simulated/open vibration-tension time series.
2. **Dual-Stream Engineering Architecture:**
   * **Stream A (Spatial/Vision):** YOLOv8/v11 instance segmentation and bounding-box detection to spot wire breaks, corrosion pitting, and surface abrasion.
   * **Stream B (Temporal/Prognostics):** BiLSTM / Temporal Convolutional Network (TCN) with Attention that ingests vibration and rope tension data to estimate Remaining Useful Life (RUL) and health index (HI).
   * **Fusion Layer:** Visual surface defect severity acts as a calibration constraint on the temporal RUL model.
3. **High Capstone Impact & Demonstration:**
   * Can run fully on software benchmarks.
   * Optionally includes a low-cost benchtop prototype: an ESP32 or webcam running inference on cable samples with an interactive real-time dashboard (FastAPI/Streamlit).
4. **Strong CSE Skill Coverage:** Computer Vision (Ultralytics/OpenCV), Deep Learning & Time-Series (PyTorch), Edge Optimization (ONNX/TFLite), and Web Visualization.

---

## 5. Technical Blueprint of the Solution

```
                          ┌────────────────────────────────────────────────────────┐
                          │               ElevatorGuard-AI System                  │
                          └────────────────────────────────────────────────────────┘
                                                    │
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 ▼                                                                     ▼
    ┌─────────────────────────┐                                           ┌─────────────────────────┐
    │  Stream A: Vision NDT   │                                           │ Stream B: Telemetry RUL │
    │ (Hoistway / Sheave Cam) │                                           │ (Accelerometer/LoadCell)│
    └────────────┬────────────┘                                           └────────────┬────────────┘
                 │                                                                     │
                 ▼                                                                     ▼
    ┌─────────────────────────┐                                           ┌─────────────────────────┐
    │ YOLOv8/v11 Segmentation │                                           │ FFT / Wavelet Transform │
    │ - Broken wire count     │                                           │ - Feature Extraction    │
    │ - Corrosion area %      │                                           │ - Tension variance      │
    └────────────┬────────────┘                                           └────────────┬────────────┘
                 │                                                                     │
                 ▼                                                                     ▼
    ┌─────────────────────────┐                                           ┌─────────────────────────┐
    │ Visual Health Index     │                                           │ BiLSTM-Attention Model  │
    │ $HI_{vis} \in [0, 1]$   │                                           │ Continuous Health Curve │
    └────────────┬────────────┘                                           └────────────┬────────────┘
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    │
                                                    ▼
                                      ┌───────────────────────────┐
                                      │   Multi-Modal Decision    │
                                      │       Fusion Engine       │
                                      │ $HI_{final} = \alpha HI_v │
                                      │         + (1-\alpha) HI_t │
                                      └─────────────┬─────────────┘
                                                    │
                                                    ▼
                                      ┌───────────────────────────┐
                                      │    Remaining Useful Life  │
                                      │     (RUL in Days/Trips)   │
                                      │   + Web Alert Dashboard   │
                                      └───────────────────────────┘
```

Detailed implementation steps, codebase structure, and execution timelines are provided in [LPR.md](file:///run/media/mrlucifer006/346E71E66E71A0F2/Personal%20Projects/KONE%20project/LPR.md).
