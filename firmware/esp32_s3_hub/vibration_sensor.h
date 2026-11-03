#pragma once

#include <Arduino.h>

struct VibrationMetrics {
    bool vibration_detected;     // Instantaneous sensor state (true = vibrating)
    uint32_t total_pulses;       // Cumulative pulse count from sensor
    float pulse_rate_hz;         // Calculated pulse frequency (pulses/sec) in moving window
    float intensity;             // Normalized vibration intensity score (0.0 - 1.0)
    bool alarm_active;           // True if vibration rate exceeds alarm threshold
    uint32_t sample_count;       // Monotonic evaluation cycle count

    // Backward compatibility fields for $EG protocol / downstream consumers
    float accel_x;               // Synthetic/estimated shock vector
    float accel_y;
    float accel_z;
    float rms_z;                 // Mapped to vibration intensity index for ride quality
    float peak_to_peak_z;
};

class VibrationSensor {
public:
    VibrationSensor();
    bool begin(int pin_do, float alarm_threshold_hz = 15.0f);
    void update();
    VibrationMetrics getMetrics();

    // Interrupt service routine entry point
    void IRAM_ATTR handleInterrupt();

private:
    int _pin;
    bool _initialized;
    float _alarm_threshold_hz;

    // Interrupt pulse counters
    volatile uint32_t _isr_pulse_count;
    volatile unsigned long _last_interrupt_time_us;
    uint32_t _last_evaluated_pulse_count;
    unsigned long _last_eval_time_ms;

    // Moving window for pulse rate estimation
    static const int WINDOW_SIZE = 20;
    float _rate_history[WINDOW_SIZE];
    int _history_idx;
    int _history_count;

    VibrationMetrics _current_metrics;
    portMUX_TYPE _metrics_mutex;
};
