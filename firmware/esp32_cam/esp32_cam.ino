/*
 * ElevatorGuard-AI: ESP32-CAM High-Performance Web API & Streamer
 *
 * Exposes:
 *  - GET /capture   : Single JPEG frame capture for API consumers (e.g., ESP32-S3 hub or Python client)
 *  - GET /stream    : Multipart/x-mixed-replace live MJPEG video stream
 *  - GET /status    : Device health, memory, and WiFi RSSI in JSON format
 *  - GET /flash?on=1: Controls the onboard high-power flash LED
 *
 * Board Target: AI-Thinker ESP32-CAM (OV2640 with external PSRAM)
 */

#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "camera_pins.h"
#include "esp32_cam_config.h"

// HTTP Server Handle
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

// Multipart boundary for MJPEG stream
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// State
static bool flash_state = false;

// ============================================================================
// Camera Initialization
// ============================================================================
esp_err_t initCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer   = LEDC_TIMER_0;
    config.pin_d0       = Y2_GPIO_NUM;
    config.pin_d1       = Y3_GPIO_NUM;
    config.pin_d2       = Y4_GPIO_NUM;
    config.pin_d3       = Y5_GPIO_NUM;
    config.pin_d4       = Y6_GPIO_NUM;
    config.pin_d5       = Y7_GPIO_NUM;
    config.pin_d6       = Y8_GPIO_NUM;
    config.pin_d7       = Y9_GPIO_NUM;
    config.pin_xclk     = XCLK_GPIO_NUM;
    config.pin_pclk     = PCLK_GPIO_NUM;
    config.pin_vsync    = VSYNC_GPIO_NUM;
    config.pin_href     = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn     = PWDN_GPIO_NUM;
    config.pin_reset    = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Check for PSRAM availability
    if (psramFound()) {
        Serial.println("[CAM] PSRAM detected! Enabling high resolution & double buffering.");
        config.frame_size = DEFAULT_FRAME_SIZE;
        config.jpeg_quality = DEFAULT_JPEG_QUALITY;
        config.fb_count = FB_COUNT_PSRAM;
        config.grab_mode = CAMERA_GRAB_LATEST;
    } else {
        Serial.println("[CAM] Warning: No PSRAM detected. Falling back to SVGA and single buffer.");
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = 14;
        config.fb_count = FB_COUNT_NO_PSRAM;
        config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Error: Camera initialization failed with error code 0x%x\n", err);
        return err;
    }

    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
        // Sensor fine tuning for industrial inspection
        s->set_brightness(s, 1);     // -2 to 2
        s->set_contrast(s, 1);       // -2 to 2
        s->set_saturation(s, 0);     // -2 to 2
        s->set_whitebal(s, 1);       // enable auto white balance
        s->set_awb_gain(s, 1);       // enable AWB gain
        s->set_wb_mode(s, 0);        // auto WB
        s->set_exposure_ctrl(s, 1);  // enable auto exposure
        s->set_aec2(s, 0);           // auto exposure algorithm
        s->set_ae_level(s, 0);       // -2 to 2
        s->set_gain_ctrl(s, 1);      // enable auto gain control
        s->set_agc_gain(s, 0);       // 0 to 30
        s->set_gainceiling(s, (gainceiling_t)2); // 0 to 6
        s->set_bpc(s, 1);            // black pixel correction
        s->set_wpc(s, 1);            // white pixel correction
        s->set_raw_gma(s, 1);        // enable gamma
        s->set_lenc(s, 1);           // lens correction
        s->set_hmirror(s, 0);        // horizontal mirror
        s->set_vflip(s, 0);          // vertical flip
        s->set_dcw(s, 1);            // downsize EN
    }

    Serial.println("[CAM] Camera sensor initialized successfully.");
    return ESP_OK;
}

// ============================================================================
// HTTP Handlers
// ============================================================================

// 1. GET /capture - Returns single JPEG image
static esp_err_t capture_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    int64_t fr_start = esp_timer_get_time();

    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("[HTTP] Camera capture failed.");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "image/jpeg");
    httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
    esp_camera_fb_return(fb);

    int64_t fr_end = esp_timer_get_time();
    // Serial.printf("[HTTP] Snapshot sent: %u bytes in %ums\n", (uint32_t)(fb->len), (uint32_t)((fr_end - fr_start) / 1000));
    return res;
}

// 2. GET /stream - Continuous MJPEG stream
static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t *fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];

    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) {
        return res;
    }
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("[HTTP] Stream: frame buffer get failed");
            res = ESP_FAIL;
            break;
        }

        _jpg_buf_len = fb->len;
        _jpg_buf = fb->buf;

        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        }
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
        }

        esp_camera_fb_return(fb);
        fb = NULL;

        if (res != ESP_OK) {
            break;
        }
    }

    return res;
}

// 3. GET /status - System telemetry & diagnostics JSON
static esp_err_t status_handler(httpd_req_t *req) {
    char json_response[256];
    snprintf(json_response, sizeof(json_response),
        "{\"status\":\"ONLINE\",\"device\":\"ESP32-CAM\",\"uptime_sec\":%lu,\"free_heap\":%u,\"min_free_heap\":%u,\"psram_free\":%u,\"wifi_rssi\":%d,\"flash_active\":%s}",
        millis() / 1000,
        ESP.getFreeHeap(),
        ESP.getMinFreeHeap(),
        ESP.getFreePsram(),
        WiFi.RSSI(),
        flash_state ? "true" : "false"
    );

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, json_response, strlen(json_response));
}

// 4. GET /flash?on=1 / flash?on=0 - Flashlight control
static esp_err_t flash_handler(httpd_req_t *req) {
    char buf[32];
    if (httpd_req_get_url_query_str(req, buf, sizeof(buf)) == ESP_OK) {
        char param[8];
        if (httpd_query_key_value(buf, "on", param, sizeof(param)) == ESP_OK) {
            if (strcmp(param, "1") == 0 || strcasecmp(param, "true") == 0) {
                flash_state = true;
                digitalWrite(FLASH_LED_PIN, HIGH);
            } else {
                flash_state = false;
                digitalWrite(FLASH_LED_PIN, LOW);
            }
        }
    }
    char resp[64];
    snprintf(resp, sizeof(resp), "{\"flash_on\":%s}", flash_state ? "true" : "false");
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    return httpd_resp_send(req, resp, strlen(resp));
}

// 5. GET / - Index page with preview
static const char PROGMEM INDEX_HTML[] = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ElevatorGuard-AI | ESP32-CAM Live Feed</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f141c; color: #e5e9f0; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
    h1 { color: #38bdf8; margin-bottom: 6px; font-size: 1.6rem; }
    .card { background: #1a2230; padding: 16px; border-radius: 12px; border: 1px solid #2d3748; max-width: 680px; width: 100%; box-shadow: 0 4px 16px rgba(0,0,0,0.4); text-align: center; }
    .stream-container { width: 100%; border-radius: 8px; overflow: hidden; background: #000; margin-top: 12px; }
    img { width: 100%; height: auto; display: block; }
    .btn-row { margin-top: 14px; display: flex; gap: 10px; justify-content: center; }
    button { background: #0284c7; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: 0.2s; }
    button:hover { background: #0369a1; }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; background: #059669; color: white; }
    .info { margin-top: 12px; font-size: 0.85rem; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
      <h1>ElevatorGuard-AI NDT Camera</h1>
      <span class="badge">LIVE STREAM</span>
    </div>
    <div class="stream-container">
      <img id="stream" src="/stream" alt="ESP32-CAM Stream">
    </div>
    <div class="btn-row">
      <button onclick="window.open('/capture', '_blank')">Single Snapshot</button>
      <button onclick="toggleFlash()">Toggle Flash LED</button>
      <button onclick="location.reload()">Refresh Stream</button>
    </div>
    <div class="info" id="status-info">Connecting to camera telemetry...</div>
  </div>
  <script>
    let flashOn = false;
    function toggleFlash() {
      flashOn = !flashOn;
      fetch('/flash?on=' + (flashOn ? '1' : '0'));
    }
    setInterval(async () => {
      try {
        const res = await fetch('/status');
        const data = await res.json();
        document.getElementById('status-info').innerText =
          `Uptime: ${data.uptime_sec}s | Heap: ${(data.free_heap / 1024).toFixed(1)} KB | PSRAM: ${(data.psram_free / 1024).toFixed(1)} KB | WiFi RSSI: ${data.wifi_rssi} dBm`;
      } catch (e) {}
    }, 2000);
  </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

// Start Web Server
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = HTTP_SERVER_PORT;
    config.ctrl_port = HTTP_SERVER_PORT + 1;
    config.max_uri_handlers = 8;
    config.task_priority = tskIDLE_PRIORITY + 5;
    config.stack_size = 8192;

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = capture_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t stream_uri = {
        .uri       = "/stream",
        .method    = HTTP_GET,
        .handler   = stream_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t status_uri = {
        .uri       = "/status",
        .method    = HTTP_GET,
        .handler   = status_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t flash_uri = {
        .uri       = "/flash",
        .method    = HTTP_GET,
        .handler   = flash_handler,
        .user_ctx  = NULL
    };

    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &capture_uri);
        httpd_register_uri_handler(camera_httpd, &stream_uri);
        httpd_register_uri_handler(camera_httpd, &status_uri);
        httpd_register_uri_handler(camera_httpd, &flash_uri);
        Serial.printf("[HTTP] Web Server listening on port %d\n", config.server_port);
    }
}

// ============================================================================
// Arduino Setup & Loop
// ============================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n==============================================");
    Serial.println(" ElevatorGuard-AI: ESP32-CAM Starting...     ");
    Serial.println("==============================================");

    // Flash LED pin
    #if defined(FLASH_LED_PIN) && FLASH_LED_PIN >= 0
    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);
    #endif

    // Onboard red LED pin (inverted on AI-Thinker)
    #if defined(ONBOARD_LED_PIN) && ONBOARD_LED_PIN >= 0
    pinMode(ONBOARD_LED_PIN, OUTPUT);
    digitalWrite(ONBOARD_LED_PIN, HIGH); // Off
    #endif

    // Initialize Camera Sensor
    if (initCamera() != ESP_OK) {
        Serial.println("[ERROR] Halting execution due to camera initialization error.");
        while (true) {
            #if defined(ONBOARD_LED_PIN) && ONBOARD_LED_PIN >= 0
            digitalWrite(ONBOARD_LED_PIN, LOW); // Blink on failure
            delay(200);
            digitalWrite(ONBOARD_LED_PIN, HIGH);
            delay(200);
            #endif
        }
    }

    // Connect to WiFi
    #if USE_STATIC_IP
    IPAddress staticIP(STATIC_IP);
    IPAddress gateway(GATEWAY_IP);
    IPAddress subnet(SUBNET_MASK);
    IPAddress dns(DNS_IP);
    WiFi.config(staticIP, gateway, subnet, dns);
    #endif

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("[WIFI] Connecting to '%s' ...\n", WIFI_SSID);

    int timeout = 0;
    while (WiFi.status() != WL_CONNECTED && timeout < 40) {
        delay(500);
        Serial.print(".");
        timeout++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WIFI] Connected successfully!");
        Serial.print("[WIFI] IP Address: http://");
        Serial.println(WiFi.localIP());
        Serial.printf("[WIFI] Signal Strength (RSSI): %d dBm\n", WiFi.RSSI());

        // Start Web Server
        startCameraServer();

        Serial.println("\nEndpoints available:");
        Serial.printf("  - Live Stream : http://%s/stream\n", WiFi.localIP().toString().c_str());
        Serial.printf("  - Snapshot    : http://%s/capture\n", WiFi.localIP().toString().c_str());
        Serial.printf("  - Status JSON : http://%s/status\n", WiFi.localIP().toString().c_str());
        Serial.println("==============================================");
    } else {
        Serial.println("\n[WIFI] Failed to connect! Check SSID and Password in esp32_cam_config.h.");
    }
}

void loop() {
    // FreeRTOS tasks and esp_http_server handle connections in background.
    // Sleep briefly to yield CPU to background server tasks.
    delay(100);
}
