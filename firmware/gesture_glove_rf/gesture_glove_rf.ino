/*
 * Gesture Glove — Random Forest (RF) Firmware
 * Optimized for ESP32-S3 + MPU6050
 *
 * Pure C m2cgen model: fast, no ML framework dependency.
 * Realtime: raw IMU → 36 features → normalize → rf_predict → buzzer output
 *
 * Features:
 *   - On-device inference measurement (latency, heap, FPS)
 *   - BLE HID keyboard output (F5, arrows, escape)
 *   - Serial debug + research data logging
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "rf_model_data.h"
#include "measurements.h"

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ─── Config ──────────────────────────────────────────────────
#define I2C_SDA             5
#define I2C_SCL             6
#define MPU_ADDR            0x69
#define BUZZER_PIN          8     // GPIO8 / D9 / A9
#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)
#define WINDOW_SIZE         250
#define NUM_AXES            6
#define STEP_SIZE           125
#define DEBOUNCE_MS         400
#define CONF_THRESH         0.80

// ─── BLE HID Key Codes (USB HID Usage IDs) ────────────────────
#define KEY_F5     0x83   // F5 = start presentation
#define KEY_LEFT   0x50   // Arrow Left = previous slide
#define KEY_RIGHT  0x4F   // Arrow Right = next slide
#define KEY_ESC    0x29   // Escape = exit presentation
#define KEY_NONE   0x00   // No key (idle)

// ─── BLE Server & Characteristics ─────────────────────────────
static BLEServer* pServer = nullptr;
static BLECharacteristic* pReportChar = nullptr;
static bool ble_connected = false;

// ─── Buzzer helpers ──────────────────────────────────────────
#define BEEP_DUR_MS   60
#define BEEP_GAP_MS   100

static inline void beep_n(int n) {
    for (int i = 0; i < n; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(BEEP_DUR_MS);
        digitalWrite(BUZZER_PIN, LOW);
        if (i < n - 1) delay(BEEP_GAP_MS);
    }
}

// ─── BLE HID Setup ─────────────────────────────────────────────
class MyBLEServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        ble_connected = true;
        Serial.println("[BLE] Client connected");
    }
    void onDisconnect(BLEServer* pServer) {
        ble_connected = false;
        Serial.println("[BLE] Client disconnected");
    }
};

static void init_ble_hid() {
    BLEDevice::init("GestureGlove");
    BLEDevice::setPower(ESP_PWR_LVL_P3);  // max TX power

    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyBLEServerCallbacks());

    // HID Service
    BLEService* pHID = pServer->createService(BLEUUID((uint16_t)0x1812));

    // Report characteristic (keyboard input)
    pReportChar = pHID->createCharacteristic(
        BLEUUID((uint16_t)0x2A4D),
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pReportChar->addDescriptor(new BLE2902());

    pHID->start();

    // Start advertising
    pServer->getAdvertising()->setAppearance(0x03C0);  // keyboard appearance
    pServer->getAdvertising()->start();

    Serial.println("[BLE] Advertising as 'GestureGlove'");
}

static void ble_send_key(uint8_t key) {
    if (!ble_connected || key == KEY_NONE) return;

    uint8_t report[5] = {0, 0, key, 0, 0};  // modifier + keycode
    pReportChar->setValue(report, 5);
    pReportChar->notify();

    delay(15);  // hold briefly

    // Release key
    memset(report, 0, 5);
    pReportChar->setValue(report, 5);
    pReportChar->notify();
}

static inline uint8_t gesture_to_key(uint8_t idx) {
    switch (idx) {
        case 0: return KEY_NONE;   // idle
        case 1: return KEY_F5;     // flick_up → F5
        case 2: return KEY_LEFT;     // wave_left → Arrow Left
        case 3: return KEY_RIGHT;   // wave_right → Arrow Right
        case 4: return KEY_ESC;      // flick_down → Escape
        default: return KEY_NONE;
    }
}

// ─── Hardware ─────────────────────────────────────────────────
static Adafruit_MPU6050 mpu;

// ─── IMU buffers ──────────────────────────────────────────────
static float imu_buf[WINDOW_SIZE][NUM_AXES];
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;
static float feat_buf[MODEL_NUM_FEATURES];  // 36
static unsigned long last_fired_ms[5] = {0};
static unsigned long last_sample_us = 0;

// ─── Measurement Stats ────────────────────────────────────────
static InferenceStats inf_stats;

// ─── Feature extraction (36 features, in-place normalize) ─────
static inline void extract_features(int start) {
    for (int a = 0; a < NUM_AXES; a++) {
        float sum = 0, sumSq = 0;
        float mn = 1e6f, mx = -1e6f;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            float v = imu_buf[(start + i) % WINDOW_SIZE][a];
            sum   += v;
            sumSq += v * v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        float mean = sum / WINDOW_SIZE;
        float std  = sqrtf(fmaxf(0, (sumSq / WINDOW_SIZE) - (mean * mean)));
        float rms  = sqrtf(sumSq / WINDOW_SIZE);

        int crossings = 0;
        float last_v = imu_buf[start % WINDOW_SIZE][a] - mean;
        for (int i = 1; i < WINDOW_SIZE; i++) {
            float v = imu_buf[(start + i) % WINDOW_SIZE][a] - mean;
            if ((v >= 0) != (last_v >= 0)) crossings++;
            last_v = v;
        }

        int b = a * 6;
        feat_buf[b]     = (mean - scaler_mean[b])     / scaler_scale[b];
        feat_buf[b + 1] = (std  - scaler_mean[b + 1]) / scaler_scale[b + 1];
        feat_buf[b + 2] = (mx   - scaler_mean[b + 2]) / scaler_scale[b + 2];
        feat_buf[b + 3] = (mn   - scaler_mean[b + 3]) / scaler_scale[b + 3];
        feat_buf[b + 4] = (rms  - scaler_mean[b + 4]) / scaler_scale[b + 4];
        feat_buf[b + 5] = (((float)crossings / (WINDOW_SIZE - 1)) - scaler_mean[b + 5]) / scaler_scale[b + 5];
    }
}

// ─── Setup ─────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    Wire.begin(I2C_SDA, I2C_SCL);

    if (!mpu.begin(MPU_ADDR, &Wire)) {
        Serial.println("[ERR] MPU6050 not found!");
        while (1) delay(100);
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // Init measurement stats
    stats_init(&inf_stats);

    // Print startup info
    Serial.printf("[INFO] Model: RF | Trees: %d | Features: %d
", MODEL_NUM_TREES, MODEL_NUM_FEATURES);
    Serial.printf("[INFO] Heap: %d / %d bytes
", ESP.getFreeHeap(), ESP.getHeapSize());

    // Init BLE HID keyboard
    init_ble_hid();

    Serial.println("[OK] RF Ready");
    beep_n(1);  // startup confirmation: 1 beep
    last_sample_us = micros();
}

// ─── Loop ───────────────────────────────────────────────────────
void loop() {
    unsigned long now_us = micros();
    if (now_us - last_sample_us < SAMPLE_INTERVAL_US) return;
    last_sample_us = now_us;

    sensors_event_t accel, gyro, temp;
    if (!mpu.getEvent(&accel, &gyro, &temp)) return;

    imu_buf[buf_idx][0] = accel.acceleration.x;
    imu_buf[buf_idx][1] = accel.acceleration.y;
    imu_buf[buf_idx][2] = accel.acceleration.z;
    imu_buf[buf_idx][3] = gyro.gyro.x;
    imu_buf[buf_idx][4] = gyro.gyro.y;
    imu_buf[buf_idx][5] = gyro.gyro.z;

    buf_idx = (buf_idx + 1) % WINDOW_SIZE;
    if (buf_idx == 0) buf_full = true;
    samples_since_infer++;

    if (!buf_full || samples_since_infer < STEP_SIZE) return;
    samples_since_infer = 0;

    // ── Feature extraction ──
    extract_features(buf_idx);

    // ── Inference with timing ──
    int64_t t0 = esp_timer_get_time();
    float confidence = 0.0f;
    int predicted = rf_predict(feat_buf, &confidence);
    int64_t t1 = esp_timer_get_time();
    int64_t latency_us = t1 - t0;
    bool ok = true;  // RF never fails
    stats_record(&inf_stats, latency_us, ok);

    // ── Serial output: prediction with latency + heap ──
    Serial.printf("[PRED] %s %.2f %lld %ld
",
        MODEL_GESTURE_NAMES[predicted], confidence,
        latency_us, ESP.getFreeHeap());

    // ── Periodic stats (every 100 inferences) ──
    if (inf_stats.total_count > 0 && inf_stats.total_count % 100 == 0) {
        float avg_lat = stats_avg_latency(&inf_stats);
        float fps = stats_avg_fps(&inf_stats);
        Serial.printf(
            "[STATS] count=%u ok=%u fail=%u min=%lldus max=%lldus avg=%.0fus fps=%.0f heap=%ld
",
            inf_stats.total_count, inf_stats.ok_count, inf_stats.fail_count,
            inf_stats.latency_min, inf_stats.latency_max,
            avg_lat, fps, ESP.getFreeHeap()
        );
    }

    // ── Stability report (every 30,000 inferences ≈ 5 min @ 100Hz) ──
    if (inf_stats.total_count > 0 && inf_stats.total_count % 30000 == 0) {
        int64_t dur = (esp_timer_get_time() - inf_stats.test_start_us) / 1000000;
        float sr = 100.0f * (float)inf_stats.ok_count / inf_stats.total_count;
        Serial.printf(
            "[STABILITY] dur=%ds total=%u ok=%u fail=%u rate=%.2f%% avg=%.0fus fps=%.0f heap_min=%lld
",
            (int)dur, inf_stats.total_count, inf_stats.ok_count, inf_stats.fail_count,
            sr, stats_avg_latency(&inf_stats), stats_avg_fps(&inf_stats), inf_stats.heap_min
        );
    }

    // ── Buzzer + BLE output ──
    if (confidence >= CONF_THRESH && predicted != 0) {
        unsigned long now = millis();
        if (now - last_fired_ms[predicted] > DEBOUNCE_MS) {
            beep_n(predicted);                        // buzzer beep
            ble_send_key(gesture_to_key(predicted));   // BLE HID key
            last_fired_ms[predicted] = now;
        }
    }
}