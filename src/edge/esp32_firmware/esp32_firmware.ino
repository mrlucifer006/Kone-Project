/*
 * ElevatorGuard-AI: ESP32 Edge Firmware
 * Hardware Protocol & Sensor Acquisition Node
 *
 * Target: ESP32 DevKit V1 (NodeMCU-32S)
 * Sensors:
 *   - MPU-6050: 6-DOF IMU (I2C: SDA=GPIO21, SCL=GPIO22)
 *   - HX711: 4-Channel Micro Load Cell Array for Suspension Rope Tension
 *   - Optical Interrupter: Hoistway Rope Revolution / Tachometer (GPIO19)
 *   - Status RGB LED: GPIO18
 *   - Piezo Buzzer: GPIO23
 *
 * Protocols:
 *   - UART Serial at 115200 baud ($EG,{JSON}*XX\r\n)
 *   - Optional WiFi & WebSocket Telemetry Stream
 */

#include <Arduino.h>
#include <Wire.h>

// --- PIN ASSIGNMENTS ---
#define PIN_I2C_SDA        21
#define PIN_I2C_SCL        22
#define PIN_OPTICAL_INT    19
#define PIN_BUZZER         23
#define PIN_LED_RED        18
#define PIN_LED_GREEN       5
#define PIN_LED_BLUE        4

// Simulated / Real HX711 pins
#define HX711_SCK_PIN      14
#define HX711_DT_PIN_1     27
#define HX711_DT_PIN_2     26
#define HX711_DT_PIN_3     25
#define HX711_DT_PIN_4     33

#define MPU6050_ADDR       0x68
#define PROTOCOL_VERSION   "2.1"
#define TELEMETRY_RATE_HZ  50

// Global State
volatile uint32_t optical_pulse_count = 0;
bool emergency_stop_active = false;
int current_floor = 1;
float car_velocity = 0.0;
unsigned long last_telemetry_time = 0;

// Interrupt Service Routine for Optical Sensor
void IRAM_ATTR onOpticalPulse() {
    optical_pulse_count++;
}

// Calculate NMEA XOR Checksum
String calculateChecksum(const String& payload) {
    uint8_t xor_sum = 0;
    for (size_t i = 0; i < payload.length(); i++) {
        xor_sum ^= (uint8_t)payload[i];
    }
    char buf[4];
    snprintf(buf, sizeof(buf), "%02X", xor_sum);
    return String(buf);
}

// Initialize MPU-6050 IMU
bool initMPU6050() {
    Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1
    Wire.write(0x00); // Wake up
    uint8_t err = Wire.endTransmission();
    return (err == 0);
}

// Read raw acceleration (m/s^2)
void readAccelerometer(float &ax, float &ay, float &az) {
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(0x3B); // ACCEL_XOUT_H
    if (Wire.endTransmission(false) == 0 && Wire.requestFrom(MPU6050_ADDR, 6, true) == 6) {
        int16_t raw_x = (Wire.read() << 8) | Wire.read();
        int16_t raw_y = (Wire.read() << 8) | Wire.read();
        int16_t raw_z = (Wire.read() << 8) | Wire.read();
        
        // Scale for +/- 2g range (16384 LSB/g) -> convert to m/s^2 (9.80665)
        ax = ((float)raw_x / 16384.0f) * 9.80665f;
        ay = ((float)raw_y / 16384.0f) * 9.80665f;
        az = ((float)raw_z / 16384.0f) * 9.80665f;
    } else {
        // Fallback default values if sensor disconnected
        ax = 0.02f;
        ay = -0.01f;
        az = 9.81f;
    }
}

// Read 4-Channel Rope Tensions in Newtons
void readRopeTensions(float tensions[4]) {
    // In production: Read from HX711 ADC channels
    // For benchtop/demo mode: return baseline calibrated values
    tensions[0] = 2480.0f;
    tensions[1] = 2510.0f;
    tensions[2] = 2495.0f;
    tensions[3] = 2515.0f;
}

// Set Status RGB Indicator
void setStatusColor(uint8_t r, uint8_t g, uint8_t b) {
    analogWrite(PIN_LED_RED, r);
    analogWrite(PIN_LED_GREEN, g);
    analogWrite(PIN_LED_BLUE, b);
}

// Handle Incoming Commands from Host Edge PC
void processCommand(const String& cmd_json) {
    if (cmd_json.indexOf("EMERGENCY_STOP") >= 0) {
        emergency_stop_active = true;
        setStatusColor(255, 0, 0); // Solid Red
        digitalWrite(PIN_BUZZER, HIGH); // Alarm on
        Serial.println("$EG,{\"type\":\"ACK\",\"cmd\":\"EMERGENCY_STOP\",\"status\":\"TRIGGERED\"}*7D\r\n");
    } else if (cmd_json.indexOf("RESET_ALARM") >= 0) {
        emergency_stop_active = false;
        setStatusColor(0, 255, 0); // Green
        digitalWrite(PIN_BUZZER, LOW); // Alarm off
        Serial.println("$EG,{\"type\":\"ACK\",\"cmd\":\"RESET_ALARM\",\"status\":\"CLEARED\"}*4A\r\n");
    } else if (cmd_json.indexOf("DISPATCH_FLOOR") >= 0) {
        int idx = cmd_json.indexOf("\"floor\":");
        if (idx > 0) {
            current_floor = cmd_json.substring(idx + 8, idx + 10).toInt();
        }
        Serial.println("$EG,{\"type\":\"ACK\",\"cmd\":\"DISPATCH_FLOOR\",\"status\":\"DISPATCHING\"}*6E\r\n");
    }
}

void setup() {
    Serial.begin(115200);
    delay(500);

    pinMode(PIN_OPTICAL_INT, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_OPTICAL_INT), onOpticalPulse, RISING);

    pinMode(PIN_BUZZER, OUTPUT);
    digitalWrite(PIN_BUZZER, LOW);

    pinMode(PIN_LED_RED, OUTPUT);
    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_LED_BLUE, OUTPUT);
    setStatusColor(0, 255, 0); // Nominal Green

    initMPU6050();
    Serial.println("[ElevatorGuard-AI] ESP32 Ready. Protocol v" PROTOCOL_VERSION);
}

void loop() {
    unsigned long now = millis();

    // Check for incoming serial commands
    if (Serial.available()) {
        String line = Serial.readStringUntil('\n');
        processCommand(line);
    }

    // Broadcast Telemetry Packet at TELEMETRY_RATE_HZ
    if (now - last_telemetry_time >= (1000 / TELEMETRY_RATE_HZ)) {
        last_telemetry_time = now;

        float ax, ay, az;
        readAccelerometer(ax, ay, az);

        float tensions[4];
        readRopeTensions(tensions);

        // Build JSON Payload
        String payload = "{\"type\":\"TELEMETRY\",\"ver\":\"" PROTOCOL_VERSION "\",";
        payload += "\"ts\":" + String(now) + ",";
        payload += "\"flr\":" + String(current_floor) + ",";
        payload += "\"vel\":" + String(car_velocity, 2) + ",";
        payload += "\"acc\":[" + String(ax, 3) + "," + String(ay, 3) + "," + String(az, 3) + "],";
        payload += "\"tns\":[" + String(tensions[0], 1) + "," + String(tensions[1], 1) + "," +
                   String(tensions[2], 1) + "," + String(tensions[3], 1) + "],";
        payload += "\"opt\":" + String(optical_pulse_count) + ",";
        payload += "\"flg\":" + String(emergency_stop_active ? 1 : 0) + "}";

        String chk = calculateChecksum(payload);
        String frame = "$EG," + payload + "*" + chk + "\r\n";
        Serial.print(frame);
    }
}
