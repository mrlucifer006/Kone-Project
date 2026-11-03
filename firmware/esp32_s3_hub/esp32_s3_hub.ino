/*
 * ElevatorGuard-AI: ESP32-S3 Edge Hub Firmware & Embedded Web Dashboard
 *
 * Capabilities:
 *  1. Vibration Analysis: Core 0 processes digital pulses from the HW-072 / SW-420 sensor
 *     and tracks vibration pulse frequency (Hz), intensity, and baseline shifts.
 *  2. Embedded Edge Vision AI: Analyzes guide rail video frames captured from the ESP32-CAM (Node A),
 *     detecting surface wear, pitting, and joint misalignment steps (ISO 7465).
 *  3. Guide Rail RUL Prognostics: Calculates remaining life (Trips and Days) using dynamic
 *     vibration frequency drift (|f_current - f_baseline|) and visual wear.
 *  4. Onboard Web Dashboard: ESP32-S3 hosts an interactive HTML5/CSS3 web dashboard
 *     displaying live vibration, remaining life, and live ESP32-CAM video feed directly.
 *  5. Cloud Fleet RTDB: Synchronizes state and timeseries data to Firebase Realtime Database.
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include "s3_config.h"
#include "vibration_sensor.h"
#include "cam_receiver.h"
#include "edge_rail_detector.h"
#include "rail_rul_calculator.h"
#include "firebase_sync.h"
#include "web_dashboard.h"

// Hardware & Algorithmic Modules
VibrationSensor vibe_sensor;
CameraReceiver cam_receiver;
EdgeRailDetector rail_detector;
RailRULCalculator rul_calculator(
    RAIL_BASELINE_FREQ_HZ,
    RAIL_MAX_ALARM_FREQ_HZ,
    NOMINAL_RAIL_LIFESPAN_TRIPS,
    AVERAGE_TRIPS_PER_DAY
);
FirebaseSyncClient firebase_client;
WebDashboardHost web_dashboard;

// Task handles
TaskHandle_t vibeTaskHandle = NULL;
TaskHandle_t commsTaskHandle = NULL;

// Helper: Calculate XOR checksum for $EG protocol
String calculateChecksum(const String& payload) {
    uint8_t xor_sum = 0;
    for (size_t i = 0; i < payload.length(); i++) {
        xor_sum ^= (uint8_t)payload[i];
    }
    char buf[4];
    snprintf(buf, sizeof(buf), "%02X", xor_sum);
    return String(buf);
}

// ============================================================================
// Core 0: Vibration Acquisition Task
// ============================================================================
void vibrationTask(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1000 / VIBRATION_SAMPLE_HZ);

    Serial.printf("[CORE 0] Guide Rail HW-072 vibration monitoring active at %d Hz.\n", VIBRATION_SAMPLE_HZ);

    while (true) {
        vibe_sensor.update();
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
    }
}

// ============================================================================
// Core 1: Camera Ingestion, Edge AI, RUL Prognostics & Firebase Sync
// ============================================================================
void commsTask(void *pvParameters) {
    Serial.println("[CORE 1] Camera ingestion, Edge AI & Cloud Sync pipeline running.");
    unsigned long last_fetch_time = 0;
    unsigned long last_serial_telemetry = 0;
    unsigned long last_firebase_push = 0;

    while (true) {
        unsigned long now = millis();

        // 1. Ingest camera frame from ESP32-CAM and run Edge Rail Vision Model
        if (now - last_fetch_time >= CAM_FETCH_INTERVAL_MS) {
            last_fetch_time = now;
            bool got_frame = cam_receiver.fetchFrame(ESP32_CAM_CAPTURE_URL);

            if (got_frame) {
                ReceivedFrame frame = cam_receiver.getLatestFrame();
                if (frame.valid && frame.length > 0) {
                    // Run Edge Vision AI
                    rail_detector.runInference(frame.data, frame.length);
                }
            }

            // Update Guide Rail RUL based on vibration frequency and visual health
            VibrationMetrics v = vibe_sensor.getMetrics();
            EdgeVisionMetrics vis = rail_detector.getLatestMetrics();
            rul_calculator.update(v.pulse_rate_hz, vis.visual_health_index);
        }

        // 2. Transmit standard $EG serial packet every 100ms
        if (ENABLE_SERIAL_STREAM && (now - last_serial_telemetry >= 100)) {
            last_serial_telemetry = now;
            VibrationMetrics v = vibe_sensor.getMetrics();
            RailRULMetrics rul = rul_calculator.getLatestMetrics();
            EdgeVisionMetrics vis = rail_detector.getLatestMetrics();

            String json_payload = "{\"type\":\"RAIL_TELEMETRY\",\"ver\":\"2.2\",\"dev\":\"" + String(DEVICE_ID) + "\"" +
                                  ",\"ts\":" + String(millis()) +
                                  ",\"vibe_hz\":" + String(v.pulse_rate_hz, 1) +
                                  ",\"delta_f\":" + String(rul.delta_freq_hz, 1) +
                                  ",\"hi\":" + String(rul.rail_health_index, 3) +
                                  ",\"rul_trips\":" + String(rul.remaining_trips) +
                                  ",\"rul_days\":" + String(rul.remaining_life_days, 1) +
                                  ",\"joint_step\":" + String(vis.joint_step_mm, 2) +
                                  ",\"alarm\":" + (v.alarm_active ? "1" : "0") + "}";

            String chk = calculateChecksum(json_payload);
            String full_packet = "$EG," + json_payload + "*" + chk + "\r\n";
            Serial.print(full_packet);
        }

        // 3. Push real-time state to Firebase Realtime Database (RTDB)
        #if ENABLE_FIREBASE_RTDB
        if (now - last_firebase_push >= FIREBASE_PUSH_INTERVAL_MS) {
            last_firebase_push = now;
            if (WiFi.status() == WL_CONNECTED) {
                VibrationMetrics v = vibe_sensor.getMetrics();
                RailRULMetrics rul = rul_calculator.getLatestMetrics();
                EdgeVisionMetrics vis = rail_detector.getLatestMetrics();

                FirebasePayload payload;
                payload.device_id = DEVICE_ID;
                payload.pulse_rate_hz = v.pulse_rate_hz;
                payload.intensity = v.intensity;
                payload.rms_z = v.rms_z;
                payload.alarm_active = v.alarm_active;

                payload.baseline_freq_hz = rul.baseline_freq_hz;
                payload.delta_freq_hz = rul.delta_freq_hz;
                payload.rail_hi = rul.rail_health_index;
                payload.remaining_trips = rul.remaining_trips;
                payload.remaining_days = rul.remaining_life_days;
                payload.wear_stage = rul.wear_stage;
                payload.action_plan = rul.action_recommendation;

                payload.rail_detected = vis.rail_detected;
                payload.surface_wear_score = vis.surface_wear_score;
                payload.joint_step_mm = vis.joint_step_mm;
                payload.visual_hi = vis.visual_health_index;

                firebase_client.pushTelemetry(payload);
                firebase_client.pushHistoryPoint(payload);
            }
        }
        #endif

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ============================================================================
// Arduino Setup & Main Loop
// ============================================================================
void setup() {
    Serial.begin(SERIAL_BAUD_RATE);
    delay(1000);
    Serial.println("\n========================================================");
    Serial.println(" ElevatorGuard-AI: ESP32-S3 Guide Rail Hub & Dashboard ");
    Serial.println("========================================================");

    // 1. Initialize HW-072 Vibration Sensor
    if (!vibe_sensor.begin(PIN_HW072_DO, HW072_ALARM_PULSE_HZ)) {
        Serial.printf("[ERROR] HW-072 Vibration sensor init failed on GPIO %d\n", PIN_HW072_DO);
    }

    // 2. Initialize Camera Receiver Buffer
    if (!cam_receiver.begin(256 * 1024)) {
        Serial.println("[ERROR] Failed to allocate camera frame buffer in PSRAM/DRAM.");
    }

    // 3. Initialize Edge Vision Model
    rail_detector.begin();

    // 4. Connect to Local WiFi Network
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("[WIFI] Connecting to '%s' ...\n", WIFI_SSID);

    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 25) {
        delay(400);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WIFI] Connected Successfully!");
        Serial.print("[WIFI] ESP32-S3 IP Address: http://");
        Serial.println(WiFi.localIP());
        Serial.printf("[WEB] Local Web Dashboard available at: http://%s/\n", WiFi.localIP().toString().c_str());
    } else {
        Serial.println("\n[WIFI] Running in standalone sensor mode (AP or fallback).");
    }

    // 5. Initialize Firebase RTDB Client
    #if ENABLE_FIREBASE_RTDB
    firebase_client.begin(FIREBASE_RTDB_URL, FIREBASE_AUTH_KEY);
    #endif

    // 6. Launch Embedded Web Dashboard Server on Port 80
    web_dashboard.begin(80);

    // 7. Spawn FreeRTOS Multi-Core Tasks
    // Core 0: Dedicated high-rate vibration interrupt and window processing
    xTaskCreatePinnedToCore(
        vibrationTask,
        "VibeTask",
        4096,
        NULL,
        configMAX_PRIORITIES - 1,
        &vibeTaskHandle,
        0
    );

    // Core 1: Ingestion, Edge AI inference, Cloud Sync
    xTaskCreatePinnedToCore(
        commsTask,
        "CommsTask",
        8192,
        NULL,
        tskIDLE_PRIORITY + 3,
        &commsTaskHandle,
        1
    );

    Serial.println("[SYSTEM] Dual-core RTOS and Web Dashboard operational.");
    Serial.println("========================================================");
}

void loop() {
    // Process incoming HTTP requests for the onboard Web Dashboard
    web_dashboard.handleClient();
    vTaskDelay(pdMS_TO_TICKS(2));
}
