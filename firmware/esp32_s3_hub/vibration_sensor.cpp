#include "vibration_sensor.h"
#include <string.h>

// Global pointer for ISR routing
static VibrationSensor *g_vibe_instance = nullptr;

static void IRAM_ATTR hw072_isr_wrapper() {
    if (g_vibe_instance) {
        g_vibe_instance->handleInterrupt();
    }
}

VibrationSensor::VibrationSensor()
    : _pin(-1),
      _initialized(false),
      _alarm_threshold_hz(15.0f),
      _isr_pulse_count(0),
      _last_interrupt_time_us(0),
      _last_evaluated_pulse_count(0),
      _last_eval_time_ms(0),
      _history_idx(0),
      _history_count(0),
      _metrics_mutex(portMUX_INITIALIZER_UNLOCKED)
{
    memset(&_current_metrics, 0, sizeof(VibrationMetrics));
    memset(_rate_history, 0, sizeof(_rate_history));
}

void IRAM_ATTR VibrationSensor::handleInterrupt() {
    unsigned long now_us = micros();
    // 2ms software debounce dead-time to avoid contact chatter on spring switch
    if (now_us - _last_interrupt_time_us >= 2000) {
        _isr_pulse_count++;
        _last_interrupt_time_us = now_us;
    }
}

bool VibrationSensor::begin(int pin_do, float alarm_threshold_hz) {
    _pin = pin_do;
    _alarm_threshold_hz = alarm_threshold_hz;
    g_vibe_instance = this;

    pinMode(_pin, INPUT_PULLUP);

    // HW-072 DO goes HIGH or changes edge during vibration pulse
    attachInterrupt(digitalPinToInterrupt(_pin), hw072_isr_wrapper, CHANGE);

    _last_eval_time_ms = millis();
    _last_evaluated_pulse_count = 0;
    _isr_pulse_count = 0;
    _history_idx = 0;
    _history_count = 0;

    _initialized = true;
    Serial.printf("[VIBE] HW-072 digital vibration sensor initialized on GPIO %d (Alarm Threshold: %.1f Hz).\n",
                  _pin, _alarm_threshold_hz);
    return true;
}

void VibrationSensor::update() {
    if (!_initialized) {
        return;
    }

    unsigned long now_ms = millis();
    unsigned long dt_ms = now_ms - _last_eval_time_ms;

    // Evaluate window metrics approximately every 50ms
    if (dt_ms < 50) {
        return;
    }
    _last_eval_time_ms = now_ms;

    // Snapshot interrupt counts atomically
    uint32_t current_total_pulses = _isr_pulse_count;
    uint32_t delta_pulses = current_total_pulses - _last_evaluated_pulse_count;
    _last_evaluated_pulse_count = current_total_pulses;

    // Calculate instantaneous pulse rate (pulses per second)
    float instant_hz = ((float)delta_pulses * 1000.0f) / (float)dt_ms;

    // Update moving window
    _rate_history[_history_idx] = instant_hz;
    _history_idx = (_history_idx + 1) % WINDOW_SIZE;
    if (_history_count < WINDOW_SIZE) {
        _history_count++;
    }

    // Compute average rate over window
    float sum_rate = 0.0f;
    for (int i = 0; i < _history_count; i++) {
        sum_rate += _rate_history[i];
    }
    float avg_rate_hz = sum_rate / (float)_history_count;

    // Intensity score normalized: 0.0 (still) up to 1.0 (continuous heavy vibration)
    // Scale against double the alarm threshold
    float intensity = avg_rate_hz / (_alarm_threshold_hz * 2.0f);
    if (intensity > 1.0f) {
        intensity = 1.0f;
    }

    bool raw_pin_state = (digitalRead(_pin) == HIGH);
    bool alarm = (avg_rate_hz >= _alarm_threshold_hz);

    // Produce synthetic acceleration values for protocol and model backward compatibility
    // Baseline gravity (9.81 m/s^2) on Z with vibration ripple proportional to intensity
    float sim_jitter = intensity * 2.5f;
    float sim_acc_z = 9.80665f + ((raw_pin_state ? 1.0f : -0.5f) * sim_jitter);
    float sim_acc_x = (raw_pin_state ? 0.3f : 0.0f) * sim_jitter;
    float sim_acc_y = (raw_pin_state ? -0.2f : 0.0f) * sim_jitter;

    portENTER_CRITICAL(&_metrics_mutex);
    _current_metrics.vibration_detected = (delta_pulses > 0) || raw_pin_state;
    _current_metrics.total_pulses = current_total_pulses;
    _current_metrics.pulse_rate_hz = avg_rate_hz;
    _current_metrics.intensity = intensity;
    _current_metrics.alarm_active = alarm;
    _current_metrics.sample_count++;

    // Backward compatibility mappings
    _current_metrics.accel_x = sim_acc_x;
    _current_metrics.accel_y = sim_acc_y;
    _current_metrics.accel_z = sim_acc_z;
    _current_metrics.rms_z = intensity;  // 0.0 to 1.0 vibration intensity index
    _current_metrics.peak_to_peak_z = sim_jitter * 2.0f;
    portEXIT_CRITICAL(&_metrics_mutex);
}

VibrationMetrics VibrationSensor::getMetrics() {
    VibrationMetrics res;
    portENTER_CRITICAL(&_metrics_mutex);
    res = _current_metrics;
    portEXIT_CRITICAL(&_metrics_mutex);
    return res;
}
