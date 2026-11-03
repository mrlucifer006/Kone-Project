#include "cam_receiver.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <esp_heap_caps.h>

CameraReceiver::CameraReceiver()
    : _frame_buffer(nullptr),
      _max_size(0),
      _current_len(0),
      _last_fetch_time(0),
      _is_valid(false),
      _frame_mutex(portMUX_INITIALIZER_UNLOCKED)
{
}

CameraReceiver::~CameraReceiver() {
    if (_frame_buffer) {
        free(_frame_buffer);
        _frame_buffer = nullptr;
    }
}

bool CameraReceiver::begin(size_t max_buffer_size) {
    _max_size = max_buffer_size;

    // Allocate in PSRAM (SPIRAM) if present on the ESP32-S3, else internal DRAM
    if (psramFound()) {
        _frame_buffer = (uint8_t *)ps_malloc(_max_size);
        if (_frame_buffer) {
            Serial.printf("[CAM_RX] Allocated %u KB frame buffer in ESP32-S3 PSRAM.\n", (uint32_t)(_max_size / 1024));
            return true;
        }
    }

    _frame_buffer = (uint8_t *)malloc(_max_size);
    if (_frame_buffer) {
        Serial.printf("[CAM_RX] Allocated %u KB frame buffer in internal RAM.\n", (uint32_t)(_max_size / 1024));
        return true;
    }

    Serial.println("[CAM_RX] Error: Failed to allocate frame buffer memory!");
    return false;
}

bool CameraReceiver::fetchFrame(const char *url) {
    if (!_frame_buffer) return false;

    HTTPClient http;
    http.setReuse(true);
    http.setTimeout(2500); // 2.5s timeout

    if (!http.begin(url)) {
        Serial.println("[CAM_RX] HTTP connection initiation failed.");
        return false;
    }

    int httpCode = http.GET();
    if (httpCode != HTTP_CODE_OK) {
        // Serial.printf("[CAM_RX] HTTP GET failed with code: %d\n", httpCode);
        http.end();
        return false;
    }

    int len = http.getSize();
    WiFiClient *stream = http.getStreamPtr();

    if (len > (int)_max_size) {
        Serial.printf("[CAM_RX] Image size (%d) exceeds max buffer (%u).\n", len, (uint32_t)_max_size);
        http.end();
        return false;
    }

    size_t total_read = 0;
    while (http.connected() && (len > 0 || len == -1)) {
        size_t size_avail = stream->available();
        if (size_avail) {
            size_t to_read = (size_avail > (_max_size - total_read)) ? (_max_size - total_read) : size_avail;
            int c = stream->readBytes(_frame_buffer + total_read, to_read);
            total_read += c;
            if (len > 0 && total_read >= (size_t)len) {
                break;
            }
        }
        delay(1);
    }
    http.end();

    // Verify JPEG Magic Bytes (0xFF 0xD8 at start)
    if (total_read > 4 && _frame_buffer[0] == 0xFF && _frame_buffer[1] == 0xD8) {
        portENTER_CRITICAL(&_frame_mutex);
        _current_len = total_read;
        _last_fetch_time = millis();
        _is_valid = true;
        portEXIT_CRITICAL(&_frame_mutex);
        return true;
    } else {
        portENTER_CRITICAL(&_frame_mutex);
        _is_valid = false;
        portEXIT_CRITICAL(&_frame_mutex);
        return false;
    }
}

ReceivedFrame CameraReceiver::getLatestFrame() {
    ReceivedFrame frame;
    portENTER_CRITICAL(&_frame_mutex);
    frame.data = _frame_buffer;
    frame.length = _current_len;
    frame.timestamp_ms = _last_fetch_time;
    frame.valid = _is_valid;
    portEXIT_CRITICAL(&_frame_mutex);
    return frame;
}
