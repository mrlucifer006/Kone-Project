#pragma once

// ==========================================
// Device Identity
// ==========================================
#define DEVICE_ID             "ELEVATOR-LIFT-01"

// ==========================================
// WiFi Network Configuration
// ==========================================
#define WIFI_SSID             "YOUR_WIFI_SSID"
#define WIFI_PASSWORD         "YOUR_WIFI_PASSWORD"

// ==========================================
// ESP32-CAM Node A Configuration
// ==========================================
// Points to the ESP32-CAM video streamer
#define ESP32_CAM_HOST        "192.168.1.150"
#define ESP32_CAM_PORT        80
#define ESP32_CAM_CAPTURE_URL "http://192.168.1.150/capture"
#define CAM_FETCH_INTERVAL_MS 400  // ~2.5 FPS continuous camera ingestion

// ==========================================
// Guide Rail RUL Vibration Physics Model
// ==========================================
// Natural guide rail baseline resonance frequency in pristine state (Hz)
#define RAIL_BASELINE_FREQ_HZ 5.0f
// Maximum acceptable vibration frequency before safety condemnation (Hz)
#define RAIL_MAX_ALARM_FREQ_HZ 35.0f
// Nominal lifecycle in elevator round trips (ISO 7465 / KONE guide rail lifespan)
#define NOMINAL_RAIL_LIFESPAN_TRIPS 300000
#define AVERAGE_TRIPS_PER_DAY 250

// ==========================================
// Firebase Realtime Database (RTDB) Fleet Sync
// ==========================================
#define ENABLE_FIREBASE_RTDB  true
#define FIREBASE_RTDB_URL     "https://elevatorguard-kone-default-rtdb.firebaseio.com"
#define FIREBASE_AUTH_KEY     ""    // Optional auth token or secret
#define FIREBASE_PUSH_INTERVAL_MS 1000 // Push telemetry once per second

// ==========================================
// Backend API / Telemetry Destination (Optional)
// ==========================================
#define BACKEND_API_URL       "http://192.168.1.100:8000/api/vision/inspect"
#define ENABLE_BACKEND_POST   false

// Serial Telemetry Streaming ($EG, {...} *XX)
#define ENABLE_SERIAL_STREAM  true
#define SERIAL_BAUD_RATE      115200

// ==========================================
// HW-072 Vibration Sensor (SW-420 Digital)
// ==========================================
#define PIN_HW072_DO          8
#define VIBRATION_SAMPLE_HZ   100
#define VIBRATION_WINDOW_SIZE 50
#define HW072_ALARM_PULSE_HZ  18.0f
