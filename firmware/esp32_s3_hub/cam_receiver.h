#pragma once

#include <Arduino.h>

struct ReceivedFrame {
    uint8_t *data;
    size_t length;
    uint32_t timestamp_ms;
    bool valid;
};

class CameraReceiver {
public:
    CameraReceiver();
    ~CameraReceiver();

    bool begin(size_t max_buffer_size = 128 * 1024); // Allocate buffer in PSRAM if available
    bool fetchFrame(const char *url);
    ReceivedFrame getLatestFrame();

private:
    uint8_t *_frame_buffer;
    size_t _max_size;
    size_t _current_len;
    uint32_t _last_fetch_time;
    bool _is_valid;
    portMUX_TYPE _frame_mutex;
};
