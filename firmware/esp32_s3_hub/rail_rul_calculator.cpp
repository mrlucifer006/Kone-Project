#include "rail_rul_calculator.h"
#include <math.h>
#include <string.h>

RailRULCalculator::RailRULCalculator(float baseline_freq,
                                     float max_acceptable_freq,
                                     int nominal_rail_life_trips,
                                     int trips_per_day)
    : _baseline_freq_hz(baseline_freq),
      _max_acceptable_freq(max_acceptable_freq),
      _nominal_rail_life_trips(nominal_rail_life_trips),
      _trips_per_day(trips_per_day),
      _rul_mutex(portMUX_INITIALIZER_UNLOCKED)
{
    memset(&_latest, 0, sizeof(RailRULMetrics));
    _latest.baseline_freq_hz = _baseline_freq_hz;
    _latest.current_freq_hz = _baseline_freq_hz;
    _latest.delta_freq_hz = 0.0f;
    _latest.rail_health_index = 1.0f;
    _latest.remaining_trips = _nominal_rail_life_trips;
    _latest.remaining_life_days = (float)_nominal_rail_life_trips / (float)_trips_per_day;
    _latest.degradation_rate_pct = 0.33f;
    _latest.wear_stage = "OPTIMAL";
    _latest.action_recommendation = "Guide rail surface within ISO 7465 pristine limits.";
}

RailRULMetrics RailRULCalculator::update(float current_freq_hz, float hi_visual) {
    if (current_freq_hz < 0.0f) current_freq_hz = 0.0f;
    if (hi_visual < 0.0f) hi_visual = 0.0f;
    if (hi_visual > 1.0f) hi_visual = 1.0f;

    float delta_f = fabsf(current_freq_hz - _baseline_freq_hz);

    // Compute vibration frequency penalty:
    // When delta_f approaches (_max_acceptable_freq - _baseline_freq_hz), vibration HI reaches minimum
    float max_delta = _max_acceptable_freq - _baseline_freq_hz;
    if (max_delta <= 0.01f) max_delta = 30.0f;

    float freq_penalty = delta_f / max_delta;
    if (freq_penalty > 1.0f) freq_penalty = 1.0f;

    // Vibration-derived Health Index (HI_vibe)
    float hi_vibe = 1.0f - freq_penalty;
    if (hi_vibe < 0.0f) hi_vibe = 0.0f;

    // Multi-modal composite health index: 60% dynamic vibration frequency response + 40% camera edge inspection
    float hi_composite = (0.60f * hi_vibe) + (0.40f * hi_visual);
    if (hi_composite < 0.02f) hi_composite = 0.02f;
    if (hi_composite > 1.0f) hi_composite = 1.0f;

    // ISO 7465 discard / safety condemnation threshold = 0.20 HI
    const float hi_discard = 0.20f;
    float usable_hi = hi_composite - hi_discard;

    RailRULMetrics m;
    m.baseline_freq_hz = _baseline_freq_hz;
    m.current_freq_hz = current_freq_hz;
    m.delta_freq_hz = delta_f;
    m.rail_health_index = hi_composite;

    if (usable_hi <= 0.0f) {
        m.remaining_trips = 0;
        m.remaining_life_days = 0.0f;
        m.wear_stage = "CRITICAL_ACTION_REQUIRED";
        m.action_recommendation = "Guide rail discard threshold exceeded (ISO 7465). Condemn rail & replace guide shoes.";
    } else {
        // Remaining life calculation
        // Total usable fraction is (1.0 - hi_discard) = 0.80
        float remaining_fraction = usable_hi / (1.0f - hi_discard);
        m.remaining_trips = (int)(remaining_fraction * (float)_nominal_rail_life_trips);
        m.remaining_life_days = (float)m.remaining_trips / (float)_trips_per_day;

        // Categorize stage
        if (hi_composite >= 0.80f) {
            m.wear_stage = "OPTIMAL";
            m.action_recommendation = "Normal guide rail wear. Standard periodic lubrication schedule.";
        } else if (hi_composite >= 0.55f) {
            m.wear_stage = "NOMINAL_WEAR";
            m.action_recommendation = "Minor track wear detected. Check guide shoe roller alignment at next maintenance.";
        } else if (hi_composite >= 0.35f) {
            m.wear_stage = "ELEVATED_FRICTION";
            m.action_recommendation = "Vibration frequency elevated. Perform rail track re-alignment and guide shoe replacement.";
        } else {
            m.wear_stage = "IMMINENT_REPLACEMENT";
            m.action_recommendation = "High frequency rail chatter and wear. Schedule immediate guide rail reprofiling or replacement.";
        }
    }

    m.degradation_rate_pct = (1.0f - hi_composite) * 100.0f;

    portENTER_CRITICAL(&_rul_mutex);
    _latest = m;
    portEXIT_CRITICAL(&_rul_mutex);

    return m;
}

RailRULMetrics RailRULCalculator::getLatestMetrics() const {
    RailRULMetrics m;
    portENTER_CRITICAL((portMUX_TYPE*)&_rul_mutex);
    m = _latest;
    portEXIT_CRITICAL((portMUX_TYPE*)&_rul_mutex);
    return m;
}
