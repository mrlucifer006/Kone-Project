#include "firebase_sync.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

FirebaseSyncClient::FirebaseSyncClient()
    : _configured(false),
      _last_push_ms(0),
      _last_history_ms(0)
{
}

bool FirebaseSyncClient::begin(const char *database_url, const char *auth_secret) {
    if (!database_url || strlen(database_url) == 0) {
        _configured = false;
        return false;
    }

    _db_url = String(database_url);
    if (_db_url.endsWith("/")) {
        _db_url = _db_url.substring(0, _db_url.length() - 1);
    }

    if (auth_secret && strlen(auth_secret) > 0) {
        _auth_key = String(auth_secret);
    } else {
        _auth_key = "";
    }

    _configured = true;
    Serial.printf("[FIREBASE] Configured RTDB target: %s\n", _db_url.c_str());
    return true;
}

bool FirebaseSyncClient::isConnected() const {
    return _configured && (WiFi.status() == WL_CONNECTED);
}

bool FirebaseSyncClient::pushTelemetry(const FirebasePayload &data) {
    if (!isConnected()) return false;

    // Build URL: <_db_url>/elevators/<device_id>/live.json(?auth=<key>)
    String url = _db_url + "/elevators/" + String(data.device_id) + "/live.json";
    if (_auth_key.length() > 0) {
        url += "?auth=" + _auth_key;
    }

    // Build JSON payload
    String json;
    json.reserve(512);
    json += "{";
    json += "\"device_id\":\"" + String(data.device_id) + "\",";
    json += "\"timestamp_ms\":" + String(millis()) + ",";
    json += "\"vibration\":{";
    json += "\"frequency_hz\":" + String(data.pulse_rate_hz, 2) + ",";
    json += "\"intensity\":" + String(data.intensity, 3) + ",";
    json += "\"rms_z\":" + String(data.rms_z, 3) + ",";
    json += "\"alarm\":" + String(data.alarm_active ? "true" : "false") + "},";
    json += "\"prognostics\":{";
    json += "\"baseline_freq_hz\":" + String(data.baseline_freq_hz, 2) + ",";
    json += "\"delta_freq_hz\":" + String(data.delta_freq_hz, 2) + ",";
    json += "\"health_index\":" + String(data.rail_hi, 3) + ",";
    json += "\"remaining_trips\":" + String(data.remaining_trips) + ",";
    json += "\"remaining_days\":" + String(data.remaining_days, 1) + ",";
    json += "\"wear_stage\":\"" + String(data.wear_stage) + "\",";
    json += "\"action_plan\":\"" + String(data.action_plan) + "\"},";
    json += "\"vision\":{";
    json += "\"rail_detected\":" + String(data.rail_detected ? "true" : "false") + ",";
    json += "\"surface_wear\":" + String(data.surface_wear_score, 3) + ",";
    json += "\"joint_step_mm\":" + String(data.joint_step_mm, 3) + ",";
    json += "\"visual_hi\":" + String(data.visual_hi, 3) + "}";
    json += "}";

    HTTPClient http;
    bool success = false;

    if (url.startsWith("https://")) {
        WiFiClientSecure client;
        client.setInsecure(); // Skip certificate pinning for embedded simplicity
        http.begin(client, url);
    } else {
        http.begin(url);
    }

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);

    // Using PUT on the live node replaces state atomically without unbounded growth
    int code = http.PUT(json);
    if (code == 200 || code == 204) {
        success = true;
    } else {
        // Serial.printf("[FIREBASE] PUT error code: %d\n", code);
    }
    http.end();

    return success;
}

bool FirebaseSyncClient::pushHistoryPoint(const FirebasePayload &data) {
    if (!isConnected()) return false;

    // Rate-limit history push to once every 10 seconds
    unsigned long now = millis();
    if (now - _last_history_ms < 10000) {
        return false;
    }
    _last_history_ms = now;

    String url = _db_url + "/elevators/" + String(data.device_id) + "/history.json";
    if (_auth_key.length() > 0) {
        url += "?auth=" + _auth_key;
    }

    String json = "{\"ts\":" + String(millis()) +
                  ",\"vibe_hz\":" + String(data.pulse_rate_hz, 2) +
                  ",\"hi\":" + String(data.rail_hi, 3) +
                  ",\"rul_days\":" + String(data.remaining_days, 1) +
                  ",\"wear_stage\":\"" + String(data.wear_stage) + "\"}";

    HTTPClient http;
    bool success = false;
    if (url.startsWith("https://")) {
        WiFiClientSecure client;
        client.setInsecure();
        http.begin(client, url);
    } else {
        http.begin(url);
    }

    http.addHeader("Content-Type", "application/json");
    http.setTimeout(3000);

    // POST appends a new child key in Firebase RTDB
    int code = http.POST(json);
    if (code == 200 || code == 204) {
        success = true;
    }
    http.end();

    return success;
}
