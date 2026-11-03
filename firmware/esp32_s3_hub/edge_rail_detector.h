#pragma once

#include <Arduino.h>

/**
 * Elevator Guide Rail Defect & Edge Inspection Result
 * Based on ISO 7465 / ISO 8100-34 specifications for guide rail geometry,
 * joint alignment steps, surface wear, and pitting.
 */
enum RailDefectClass {
    DEFECT_NONE = 0,
    DEFECT_SURFACE_WEAR = 1,
    DEFECT_JOINT_MISALIGNMENT = 2,
    DEFECT_SURFACE_PITTING = 3,
    DEFECT_CORROSION = 4
};

struct RailDetection {
    RailDefectClass defect_type;
    float confidence;           // 0.0 - 1.0
    int bbox[4];                // [x, y, w, h]
    float severity_score;       // 0.0 - 1.0
    const char *label;
};

struct EdgeVisionMetrics {
    bool rail_detected;
    float rail_confidence;
    float surface_wear_score;       // 0.0 (pristine mirror blade) to 1.0 (severe groove wear)
    float joint_step_mm;            // ISO 7465 max allowable step is typically <= 0.20 mm
    float pitting_density;          // 0.0 - 1.0
    float visual_health_index;      // HI_vis: 1.0 (new) to 0.0 (condemned)
    int detection_count;
    RailDetection detections[4];
    uint32_t inference_time_ms;
    uint32_t last_inference_ts;
};

class EdgeRailDetector {
public:
    EdgeRailDetector();
    bool begin();

    // Runs lightweight edge vision model / quantized CNN descriptor on the JPEG frame
    EdgeVisionMetrics runInference(const uint8_t *jpeg_data, size_t jpeg_len);

    EdgeVisionMetrics getLatestMetrics() const;

private:
    EdgeVisionMetrics _latest;
    portMUX_TYPE _model_mutex;
    bool _initialized;
};
