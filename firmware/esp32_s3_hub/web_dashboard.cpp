#include "web_dashboard.h"
#include <WiFi.h>
#include <WebServer.h>
#include "s3_config.h"
#include "vibration_sensor.h"
#include "cam_receiver.h"
#include "edge_rail_detector.h"
#include "rail_rul_calculator.h"

// External instances from main sketch
extern VibrationSensor vibe_sensor;
extern CameraReceiver cam_receiver;
extern EdgeRailDetector rail_detector;
extern RailRULCalculator rul_calculator;

static WebServer server(80);

// Embedded HTML5 / CSS3 / JavaScript Real-Time Dashboard
static const char DASHBOARD_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ElevatorGuard-AI | Guide Rail Edge Dashboard</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #141c2e;
      --card-border: #23314d;
      --accent-blue: #0084ff;
      --accent-teal: #00d2be;
      --accent-green: #00e676;
      --accent-amber: #ffb300;
      --accent-red: #ff3d00;
      --text-main: #f0f4fc;
      --text-dim: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text-main); padding: 18px; min-height: 100vh; }
    .header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--card-border); margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .logo-badge { background: linear-gradient(135deg, #0066cc, #00d2be); color: white; padding: 6px 12px; border-radius: 8px; font-weight: 800; font-size: 1.1rem; letter-spacing: 0.5px; }
    .title-group h1 { font-size: 1.35rem; font-weight: 700; color: #fff; }
    .title-group p { font-size: 0.8rem; color: var(--text-dim); }
    .status-badge { display: flex; align-items: center; gap: 8px; background: rgba(0,230,118,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,118,0.3); padding: 6px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 600; }
    .pulse-dot { width: 8px; height: 8px; background-color: var(--accent-green); border-radius: 50%; box-shadow: 0 0 8px var(--accent-green); animation: pulse 1.8s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }

    .grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 18px; }
    .col-4 { grid-column: span 4; }
    .col-6 { grid-column: span 6; }
    .col-8 { grid-column: span 8; }
    .col-12 { grid-column: span 12; }
    @media (max-width: 960px) { .col-4, .col-6, .col-8 { grid-column: span 12; } }

    .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
    .card-title { font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-dim); margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; }

    /* Key Metric Big Number */
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1.1; margin-bottom: 6px; }
    .metric-unit { font-size: 1rem; font-weight: 500; color: var(--text-dim); }
    .metric-sub { font-size: 0.82rem; color: var(--text-dim); }

    /* Progress bar */
    .progress-track { width: 100%; height: 10px; background: #0b0f19; border-radius: 6px; overflow: hidden; margin: 12px 0 8px 0; border: 1px solid rgba(255,255,255,0.06); }
    .progress-fill { height: 100%; background: linear-gradient(90deg, var(--accent-green), var(--accent-teal)); border-radius: 6px; transition: width 0.4s ease; }

    /* Camera Feed Container */
    .cam-frame-box { position: relative; width: 100%; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid var(--card-border); min-height: 280px; display: flex; justify-content: center; align-items: center; }
    .cam-frame-box img { width: 100%; height: auto; max-height: 380px; object-fit: contain; display: block; }
    .cam-overlay-hud { position: absolute; top: 10px; left: 10px; background: rgba(11,15,25,0.8); backdrop-filter: blur(4px); padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.15); }
    .cam-overlay-hud span { color: var(--accent-teal); font-weight: 600; }

    /* Table & Stats */
    .stats-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .stats-table td { padding: 9px 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }
    .stats-table td:last-child { text-align: right; font-weight: 600; color: #fff; }

    /* Action Callout */
    .action-box { background: rgba(0,132,255,0.08); border-left: 4px solid var(--accent-blue); padding: 12px; border-radius: 0 8px 8px 0; font-size: 0.85rem; margin-top: 14px; }
    .action-box b { color: var(--accent-blue); }

    .tag { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 700; }
    .tag-optimal { background: rgba(0,230,118,0.2); color: var(--accent-green); }
    .tag-warn { background: rgba(255,179,0,0.2); color: var(--accent-amber); }
    .tag-danger { background: rgba(255,61,0,0.2); color: var(--accent-red); }
  </style>
</head>
<body>

  <div class="header">
    <div class="brand">
      <div class="logo-badge">KONE / EG-AI</div>
      <div class="title-group">
        <h1>Elevator Guide Rail Edge Monitor</h1>
        <p>ESP32-S3 Edge Hub Host &bull; Dual-Core RTOS &bull; Real-Time Vibration &amp; Visual RUL</p>
      </div>
    </div>
    <div class="status-badge" id="system-badge">
      <div class="pulse-dot"></div>
      <span id="system-status-text">SYSTEM NORMAL</span>
    </div>
  </div>

  <div class="grid">

    <!-- Card 1: Remaining Life to Live (RUL) -->
    <div class="card col-4">
      <div class="card-title">
        <span>Guide Rail Remaining Useful Life</span>
        <span class="tag tag-optimal" id="wear-tag">OPTIMAL</span>
      </div>
      <div class="metric-value">
        <span id="rul-days">--</span> <span class="metric-unit">Days</span>
      </div>
      <div class="metric-sub" id="rul-trips-sub">Estimated Remaining: -- Trips</div>

      <div class="progress-track">
        <div class="progress-fill" id="hi-progress" style="width: 100%;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:var(--text-dim); margin-bottom:12px;">
        <span>Health Index (HI)</span>
        <span id="hi-val" style="color:#fff; font-weight:600;">1.00</span>
      </div>

      <div class="action-box">
        <b>Prognostic Action:</b>
        <div id="rul-action-text" style="margin-top:4px; color:var(--text-main);">Evaluating edge degradation trend...</div>
      </div>
    </div>

    <!-- Card 2: Vibration Frequency & Dynamics -->
    <div class="card col-4">
      <div class="card-title">
        <span>Vibration Frequency &amp; Shift</span>
        <span id="alarm-tag" class="tag tag-optimal">NORMAL</span>
      </div>
      <div class="metric-value">
        <span id="vibe-hz">--</span> <span class="metric-unit">Hz</span>
      </div>
      <div class="metric-sub">Frequency Shift (&Delta;f): <b id="delta-freq" style="color:var(--accent-teal);">-- Hz</b></div>

      <table class="stats-table" style="margin-top: 14px;">
        <tr><td>Baseline Resonance Frequency</td><td id="base-freq">5.00 Hz</td></tr>
        <tr><td>Vibration Intensity Index</td><td id="vibe-intensity">0.00</td></tr>
        <tr><td>RMS Z-Axis Acceleration</td><td id="rms-z">0.00 g</td></tr>
        <tr><td>Cumulative Pulse Count</td><td id="total-pulses">0</td></tr>
      </table>
    </div>

    <!-- Card 3: Live Camera Feed from ESP32-CAM -->
    <div class="card col-4">
      <div class="card-title">
        <span>Live Guide Rail Camera Feed</span>
        <span style="font-size:0.75rem; color:var(--text-dim);">ESP32-CAM (Node A)</span>
      </div>
      <div class="cam-frame-box">
        <div class="cam-overlay-hud">
          Target: <span>T-Profile Rail Surface</span> | FPS: <span id="cam-fps">2</span>
        </div>
        <img id="live-cam-img" src="/api/cam_feed" alt="Live Guide Rail Camera Feed" onerror="handleCamError(this)">
      </div>
      <div style="margin-top:10px; display:flex; justify-content:space-between; font-size:0.78rem; color:var(--text-dim);">
        <span>ISO 7465 Rail Defect Status: <b id="rail-status" style="color:var(--accent-green);">Aligned</b></span>
        <span>Joint Step: <b id="joint-step" style="color:#fff;">0.05 mm</b></span>
      </div>
    </div>

    <!-- Card 4: Multi-Elevator Fleet & Cloud Link -->
    <div class="card col-12">
      <div class="card-title">
        <span>Distributed Edge Connectivity &amp; Multi-Elevator Cloud Sync</span>
        <span class="tag" style="background:#0084ff26; color:var(--accent-blue);">FIREBASE RTDB ACTIVE</span>
      </div>
      <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px;">
        <div style="background:#0b0f19; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:0.75rem; color:var(--text-dim);">DEVICE ID</div>
          <div style="font-size:1.1rem; font-weight:700; color:#fff;" id="device-id-label">ESP32-S3-HUB-01</div>
        </div>
        <div style="background:#0b0f19; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:0.75rem; color:var(--text-dim);">CLOUD REALTIME SYNC</div>
          <div style="font-size:1.1rem; font-weight:700; color:var(--accent-teal);" id="firebase-state">Connected</div>
        </div>
        <div style="background:#0b0f19; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:0.75rem; color:var(--text-dim);">EDGE INFERENCE LATENCY</div>
          <div style="font-size:1.1rem; font-weight:700; color:#fff;" id="inference-time">18 ms</div>
        </div>
        <div style="background:#0b0f19; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:0.75rem; color:var(--text-dim);">FREE HEAP / PSRAM</div>
          <div style="font-size:1.1rem; font-weight:700; color:#fff;" id="memory-state">-- KB</div>
        </div>
      </div>
    </div>

  </div>

  <script>
    function handleCamError(img) {
      setTimeout(() => { img.src = "/api/cam_feed?t=" + new Date().getTime(); }, 1500);
    }

    // Refresh Camera snapshot periodically
    setInterval(() => {
      const img = document.getElementById("live-cam-img");
      img.src = "/api/cam_feed?t=" + new Date().getTime();
    }, 500);

    // Poll live JSON telemetry every 250ms
    async function updateTelemetry() {
      try {
        const res = await fetch("/api/telemetry");
        if (!res.ok) return;
        const d = await res.json();

        // Vibration
        document.getElementById("vibe-hz").innerText = d.vibration.frequency_hz.toFixed(1);
        document.getElementById("delta-freq").innerText = d.prognostics.delta_freq_hz.toFixed(1) + " Hz";
        document.getElementById("base-freq").innerText = d.prognostics.baseline_freq_hz.toFixed(2) + " Hz";
        document.getElementById("vibe-intensity").innerText = d.vibration.intensity.toFixed(3);
        document.getElementById("rms-z").innerText = d.vibration.rms_z.toFixed(3) + " g";
        document.getElementById("total-pulses").innerText = d.vibration.pulses.toLocaleString();

        // RUL Prognostics
        document.getElementById("rul-days").innerText = d.prognostics.remaining_days.toFixed(1);
        document.getElementById("rul-trips-sub").innerText = "Estimated Remaining: " + d.prognostics.remaining_trips.toLocaleString() + " Trips";
        document.getElementById("hi-val").innerText = (d.prognostics.health_index * 100).toFixed(1) + "%";
        document.getElementById("hi-progress").style.width = (d.prognostics.health_index * 100) + "%";
        document.getElementById("rul-action-text").innerText = d.prognostics.action_plan;

        const wearTag = document.getElementById("wear-tag");
        wearTag.innerText = d.prognostics.wear_stage;
        if (d.prognostics.health_index > 0.75) {
          wearTag.className = "tag tag-optimal";
        } else if (d.prognostics.health_index > 0.40) {
          wearTag.className = "tag tag-warn";
        } else {
          wearTag.className = "tag tag-danger";
        }

        // Alarm status
        const alarmTag = document.getElementById("alarm-tag");
        const sysBadge = document.getElementById("system-badge");
        const sysText = document.getElementById("system-status-text");

        if (d.vibration.alarm) {
          alarmTag.className = "tag tag-danger";
          alarmTag.innerText = "HIGH VIBRATION";
          sysBadge.style.color = "var(--accent-red)";
          sysBadge.style.borderColor = "var(--accent-red)";
          sysText.innerText = "VIBRATION ALERT";
        } else {
          alarmTag.className = "tag tag-optimal";
          alarmTag.innerText = "NORMAL";
          sysBadge.style.color = "var(--accent-green)";
          sysBadge.style.borderColor = "rgba(0,230,118,0.3)";
          sysText.innerText = "SYSTEM NORMAL";
        }

        // Vision
        document.getElementById("joint-step").innerText = d.vision.joint_step_mm.toFixed(2) + " mm";
        document.getElementById("rail-status").innerText = d.vision.joint_step_mm > 0.20 ? "Misaligned (Step > 0.2mm)" : "Aligned & In-Tolerance";
        document.getElementById("rail-status").style.color = d.vision.joint_step_mm > 0.20 ? "var(--accent-red)" : "var(--accent-green)";

        // Edge diagnostics
        document.getElementById("device-id-label").innerText = d.device_id;
        document.getElementById("memory-state").innerText = (d.system.free_heap / 1024).toFixed(0) + " KB Heap";
        document.getElementById("inference-time").innerText = d.vision.inference_ms + " ms";

      } catch (e) {
        console.error("Telemetry fetch error:", e);
      }
    }

    setInterval(updateTelemetry, 250);
    updateTelemetry();
  </script>
</body>
</html>
)rawliteral";

WebDashboardHost::WebDashboardHost() : _port(80), _running(false) {}

bool WebDashboardHost::begin(uint16_t port) {
    _port = port;

    // 1. Root: Serves rich HTML5 Web Dashboard
    server.on("/", HTTP_GET, []() {
        server.send_P(200, "text/html", DASHBOARD_HTML);
    });

    // 2. /api/telemetry: Comprehensive JSON payload including vibration, RUL frequency shift, and vision metrics
    server.on("/api/telemetry", HTTP_GET, []() {
        VibrationMetrics vibe = vibe_sensor.getMetrics();
        EdgeVisionMetrics vis = rail_detector.getLatestMetrics();
        RailRULMetrics rul = rul_calculator.getLatestMetrics();

        String json;
        json.reserve(768);
        json += "{";
        json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
        json += "\"timestamp_ms\":" + String(millis()) + ",";
        json += "\"system\":{";
        json += "\"free_heap\":" + String(ESP.getFreeHeap()) + ",";
        json += "\"uptime_sec\":" + String(millis() / 1000) + "},";
        json += "\"vibration\":{";
        json += "\"frequency_hz\":" + String(vibe.pulse_rate_hz, 2) + ",";
        json += "\"intensity\":" + String(vibe.intensity, 3) + ",";
        json += "\"rms_z\":" + String(vibe.rms_z, 3) + ",";
        json += "\"pulses\":" + String(vibe.total_pulses) + ",";
        json += "\"alarm\":" + String(vibe.alarm_active ? "true" : "false") + "},";
        json += "\"prognostics\":{";
        json += "\"baseline_freq_hz\":" + String(rul.baseline_freq_hz, 2) + ",";
        json += "\"current_freq_hz\":" + String(rul.current_freq_hz, 2) + ",";
        json += "\"delta_freq_hz\":" + String(rul.delta_freq_hz, 2) + ",";
        json += "\"health_index\":" + String(rul.rail_health_index, 3) + ",";
        json += "\"remaining_trips\":" + String(rul.remaining_trips) + ",";
        json += "\"remaining_days\":" + String(rul.remaining_life_days, 1) + ",";
        json += "\"wear_stage\":\"" + String(rul.wear_stage) + "\",";
        json += "\"action_plan\":\"" + String(rul.action_recommendation) + "\"},";
        json += "\"vision\":{";
        json += "\"rail_detected\":" + String(vis.rail_detected ? "true" : "false") + ",";
        json += "\"confidence\":" + String(vis.rail_confidence, 2) + ",";
        json += "\"surface_wear\":" + String(vis.surface_wear_score, 3) + ",";
        json += "\"joint_step_mm\":" + String(vis.joint_step_mm, 3) + ",";
        json += "\"visual_hi\":" + String(vis.visual_health_index, 3) + ",";
        json += "\"inference_ms\":" + String(vis.inference_time_ms) + "}";
        json += "}";

        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.send(200, "application/json", json);
    });

    // 3. /api/cam_feed: Real-time proxied JPEG snapshot from ESP32-CAM
    server.on("/api/cam_feed", HTTP_GET, []() {
        ReceivedFrame frame = cam_receiver.getLatestFrame();
        server.sendHeader("Access-Control-Allow-Origin", "*");
        server.sendHeader("Cache-Control", "no-cache, no-store, must-revalidate");

        if (frame.valid && frame.length > 0 && frame.data != nullptr) {
            WiFiClient client = server.client();
            client.println("HTTP/1.1 200 OK");
            client.println("Content-Type: image/jpeg");
            client.println("Access-Control-Allow-Origin: *");
            client.println("Connection: close");
            client.printf("Content-Length: %u\r\n\r\n", (uint32_t)frame.length);
            client.write(frame.data, frame.length);
        } else {
            // Placeholder SVG when no frame is ready
            String svg = "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='240' viewBox='0 0 320 240'>"
                         "<rect width='320' height='240' fill='#141c2e'/>"
                         "<text x='160' y='120' fill='#94a3b8' font-family='sans-serif' font-size='14' text-anchor='middle'>Awaiting ESP32-CAM Frame...</text>"
                         "</svg>";
            server.send(200, "image/svg+xml", svg);
        }
    });

    server.begin();
    _running = true;
    Serial.printf("[WEB_SERVER] HTTP Web Dashboard listening on port %u\n", _port);
    return true;
}

void WebDashboardHost::handleClient() {
    if (_running) {
        server.handleClient();
    }
}
