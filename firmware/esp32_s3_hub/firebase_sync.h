#pragma once

#include <Arduino.h>

/**
 * Firebase Realtime Database (RTDB) REST Client for ESP32
 * Uses direct HTTPS / HTTP REST API with zero external third-party library dependencies.
 * 
 * Target endpoints:
 *   https://<FIREBASE_PROJECT_ID>-default-rtdb.firebaseio.com/elevators/<DEVICE_ID>.json?auth=<FIREBASE_AUTH_KEY>
 */

struct FirebasePayload {
    const char *device_id;
    float pulse_rate_hz;
    float intensity;
    float rms_z;
    bool alarm_active;
    float baseline_freq_hz;
    float delta_freq_hz;
    float rail_hi;
    int remaining_trips;
    float remaining_days;
    const char *wear_stage;
    const char *action_plan;
    bool rail_detected;
    float surface_wear_score;
    float joint_step_mm;
    float visual_hi;
};

class FirebaseSyncClient {
public:
    FirebaseSyncClient();
    bool begin(const char *database_url, const char *auth_secret = "");

    // Push JSON state to /elevators/<device_id>/live.json
    bool pushTelemetry(const FirebasePayload &data);

    // Push historical timeseries point to /elevators/<device_id>/history.json
    bool pushHistoryPoint(const FirebasePayload &data);

    bool isConnected() const;

private:
    String _db_url;
    String _auth_key;
    bool _configured;
    unsigned long _last_push_ms;
    unsigned long _last_history_ms;
};
