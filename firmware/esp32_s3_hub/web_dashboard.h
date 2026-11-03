#pragma once

#include <Arduino.h>

/**
 * Onboard High-Performance Async Web Server for ESP32-S3
 * Hosts:
 *  - GET /               : Responsive real-time Guide Rail Predictive Maintenance Dashboard
 *  - GET /api/telemetry  : Live JSON telemetry packet
 *  - GET /api/cam_feed   : Real-time proxied JPEG snapshot from ESP32-CAM with edge bounding boxes
 *  - GET /api/rul        : Guide Rail RUL prognostics & frequency shift breakdown
 *  - GET /api/config     : Read/Update WiFi, Cam IP, and Firebase parameters
 */

class WebDashboardHost {
public:
    WebDashboardHost();
    bool begin(uint16_t port = 80);
    void handleClient();

private:
    uint16_t _port;
    bool _running;
};
