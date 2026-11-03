#pragma once

#include <Arduino.h>

/**
 * Guide Rail Remaining Useful Life (RUL) & Frequency Shift Model
 * 
 * Mechanics & Physics:
 * As elevator guide rails wear, experience joint gaps, or guide shoes develop play:
 * 1. Base natural resonance frequency drops (f_res shifts lower due to structural degradation/loosening)
 * 2. Peak vibration pulse frequency f_vibe increases due to impacts and roughness.
 * 3. The ratio (delta_f = |f_vibe - f_baseline|) directly measures degradation velocity.
 * 4. Life expectancy (Trips and Remaining Days) is computed via progressive wear law.
 */

struct RailRULMetrics {
    float baseline_freq_hz;         // Pristine guide rail natural resonance / baseline vibe (e.g. 5.0 Hz)
    float current_freq_hz;          // Current vibration pulse frequency (from HW-072 / sensor)
    float delta_freq_hz;            // Frequency shift: |current - baseline|
    float rail_health_index;        // Composite HI (0.0 - 1.0)
    int remaining_trips;            // Remaining operational elevator round trips
    float remaining_life_days;      // Remaining service life in days
    float degradation_rate_pct;     // Rate of lifecycle consumption per 10k trips
    const char *wear_stage;         // "OPTIMAL", "NOMINAL_WEAR", "ELEVATED_FRICTION", "CRITICAL_ACTION_REQUIRED"
    const char *action_recommendation;
};

class RailRULCalculator {
public:
    RailRULCalculator(float baseline_freq = 5.0f,
                      float max_acceptable_freq = 35.0f,
                      int nominal_rail_life_trips = 300000,
                      int trips_per_day = 250);

    // Compute updated RUL metrics using current vibration frequency and visual health index
    RailRULMetrics update(float current_freq_hz, float hi_visual = 1.0f);

    RailRULMetrics getLatestMetrics() const;

private:
    float _baseline_freq_hz;
    float _max_acceptable_freq;
    int _nominal_rail_life_trips;
    int _trips_per_day;

    RailRULMetrics _latest;
    portMUX_TYPE _rul_mutex;
};
