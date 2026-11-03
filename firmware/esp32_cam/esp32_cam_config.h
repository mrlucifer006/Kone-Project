#pragma once

// ==========================================
// WiFi Network Configuration
// ==========================================
#define WIFI_SSID       "YOUR_WIFI_SSID"
#define WIFI_PASSWORD   "YOUR_WIFI_PASSWORD"

// Optional: Static IP configuration (set USE_STATIC_IP to true if needed)
#define USE_STATIC_IP   false
#define STATIC_IP       192, 168, 1, 150
#define GATEWAY_IP      192, 168, 1, 1
#define SUBNET_MASK     255, 255, 255, 0
#define DNS_IP          8, 8, 8, 8

// ==========================================
// Camera Image Configuration
// ==========================================
// Frame Size Options:
// FRAMESIZE_QVGA  (320x240)  - High FPS, lowest latency
// FRAMESIZE_CIF   (400x296)
// FRAMESIZE_VGA   (640x480)  - Recommended balance
// FRAMESIZE_SVGA  (800x600)  - Higher detail
// FRAMESIZE_XGA   (1024x768)
// FRAMESIZE_SXGA  (1280x1024)
// FRAMESIZE_UXGA  (1600x1200)
#define DEFAULT_FRAME_SIZE   FRAMESIZE_VGA

// JPEG Quality: 10-63 (Lower number = higher quality & larger payload, 10-14 ideal with PSRAM)
#define DEFAULT_JPEG_QUALITY 12

// Framebuffer count in PSRAM (2 for smooth double-buffered streaming)
#define FB_COUNT_PSRAM       2
#define FB_COUNT_NO_PSRAM    1

// ==========================================
// HTTP Server Configuration
// ==========================================
#define HTTP_SERVER_PORT     80
