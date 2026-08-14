/*
 * Gesture Glove — MLP (TFLite Micro int8) + BLE untuk SMARTBOARD
 * XIAO Seeed ESP32-S3 + MPU6050
 *
 * Inferensi MLP → kirim prediksi via BLE (Nordic UART Service) ke smartboard.py.
 * Format baris: "[PRED] <gesture> <conf> <latency_us>"
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mlp_model_data.h"

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ─── Config ──────────────────────────────────────────────────
#define I2C_SDA             5
#define I2C_SCL             6
#define MPU_ADDR            0x69
#define BUZZER_PIN          8
#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)
#define WINDOW_SIZE         250
#define NUM_AXES            6
#define STEP_SIZE           125
#define DEBOUNCE_MS         400
#define TENSOR_ARENA_SIZE   (100 * 1024)
#define CONF_THRESH         0.90
#define OUTPUT_COOLDOWN_MS  3000
// Per-class minimum peak gyro magnitude (rad/s, MPU6050 native units) required
// within a window before a non-idle prediction is accepted. Index matches
// MODEL_GESTURE_NAMES (0=idle, 1=flick_up, 2=wave_left, 3=wave_right,
// 4=flick_down). wave_left requires more motion (10.0) than the others (9.5).
// NOTE: in this dataset real gesture windows themselves mostly measure well
// below these values (e.g. flick_up 2.9-8.9, wave_left 5.2-9.9 peak rad/s) —
// thresholds this high are expected to reject most real gestures too, not
// just false positives from casual movement.
static const float GYRO_MOTION_THRESHOLD_BY_CLASS[5] = {
    0.0f,   // idle — unused, idle is never gated
    9.5f,   // flick_up
    9.7f,  // wave_left
    9.5f,   // wave_right
    9.5f,   // flick_down
};

#define NUS_SERVICE_UUID  "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_UUID       "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

// ─── Buzzer ──────────────────────────────────────────────────
#define BEEP_DUR_MS   60
#define BEEP_GAP_MS   100
static inline void beep_n(int n) {
    for (int i = 0; i < n; i++) {
        digitalWrite(BUZZER_PIN, HIGH); delay(BEEP_DUR_MS);
        digitalWrite(BUZZER_PIN, LOW);
        if (i < n - 1) delay(BEEP_GAP_MS);
    }
}

// ─── BLE ─────────────────────────────────────────────────────
static BLEServer*         pServer = nullptr;
static BLECharacteristic* pTxChar = nullptr;
static volatile bool      ble_connected = false;

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override    { ble_connected = true;  Serial.println("[BLE] connected"); }
    void onDisconnect(BLEServer* s) override { ble_connected = false; Serial.println("[BLE] disconnected"); s->getAdvertising()->start(); }
};

static void init_ble() {
    BLEDevice::init("GestureGlove");
    BLEDevice::setMTU(247);
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());
    BLEService* svc = pServer->createService(NUS_SERVICE_UUID);
    pTxChar = svc->createCharacteristic(NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    pTxChar->addDescriptor(new BLE2902());
    svc->start();
    BLEAdvertising* adv = pServer->getAdvertising();
    adv->addServiceUUID(NUS_SERVICE_UUID);
    adv->setScanResponse(true);
    adv->start();
    Serial.println("[BLE] Advertising as 'GestureGlove'");
}

static void send_pred(const char* gesture, float conf, int64_t latency_us) {
    char line[64];
    int n = snprintf(line, sizeof(line), "[PRED] %s %.2f %lld\n", gesture, conf, latency_us);
    Serial.print(line);
    if (ble_connected && pTxChar && n > 0) {
        pTxChar->setValue((uint8_t*)line, n);
        pTxChar->notify();
    }
}

// ─── TFLite ──────────────────────────────────────────────────
static tflite::MicroMutableOpResolver<3> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ─── Hardware & buffers ──────────────────────────────────────
static Adafruit_MPU6050 mpu;
static float imu_buf[WINDOW_SIZE][NUM_AXES];
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;
static float feat_buf[MODEL_NUM_FEATURES];
static int last_shown = -1;
static unsigned long last_gesture_ms = 0;
static unsigned long last_sample_us = 0;
static bool cooldown_active = false;

static inline void extract_features(int start) {
    for (int a = 0; a < NUM_AXES; a++) {
        float sum = 0, sumSq = 0, mn = 1e6, mx = -1e6;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            float v = imu_buf[(start + i) % WINDOW_SIZE][a];
            sum += v; sumSq += v * v;
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

// ─── Motion gate: peak gyro magnitude over the window ─────────
// Used to reject low-motion windows (slight/unintentional movement) before
// accepting a non-idle prediction, independent of model confidence.
static inline float motion_gyro_peak(int start) {
    float peak = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (start + i) % WINDOW_SIZE;
        float gx = imu_buf[idx][3], gy = imu_buf[idx][4], gz = imu_buf[idx][5];
        float gmag = sqrtf(gx * gx + gy * gy + gz * gz);
        if (gmag > peak) peak = gmag;
    }
    return peak;
}

// ─── Setup ───────────────────────────────────────────────────
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

    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[ERR] Model version mismatch!");
        while (1);
    }
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[ERR] AllocateTensors failed!");
        while (1);
    }

    init_ble();
    Serial.println("[OK] MLP + BLE Smartboard Ready");
    beep_n(1);
    last_sample_us = micros();
}

// ─── Loop ────────────────────────────────────────────────────
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

    extract_features(buf_idx);
    float motion_peak = motion_gyro_peak(buf_idx);

    // Quantize input
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(feat_buf[i] / scale + zp), -128, 127);
    }

    int64_t t0 = esp_timer_get_time();
    bool ok = (interpreter->Invoke() == kTfLiteOk);
    int64_t latency_us = esp_timer_get_time() - t0;
    if (!ok) { Serial.println("[ERR] Invoke failed"); return; }

    TfLiteTensor* out = interpreter->output(0);
    float max_prob = -1.0f, second_prob = -1.0f;
    int best = 0;
    for (int i = 0; i < 5; i++) {
        float p = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
        if (p > max_prob) { second_prob = max_prob; max_prob = p; best = i; }
        else if (p > second_prob) { second_prob = p; }
    }

    // ── DEBUG: print every window's raw numbers (motion, top prob, margin to
    // runner-up, predicted label) BEFORE any gating is applied. Use this to see
    // what's actually happening when a "slight gesture" triggers a false
    // detection — compare against real-gesture numbers to pick correct
    // thresholds, or spot that confidence stays high (~1.00) regardless of
    // motion (a sign the gate isn't the real fix and the model needs
    // retraining with more "slight movement" negative examples).
    Serial.printf("[DBG] motion=%.3f pred=%-10s conf=%.2f margin=%.2f\n",
        motion_peak, MODEL_GESTURE_NAMES[best], max_prob, max_prob - second_prob);

    // Cek cooldown selesai → beep 1x "gesture ready"
    unsigned long now = millis();
    if (cooldown_active && (now - last_gesture_ms >= OUTPUT_COOLDOWN_MS)) {
        cooldown_active = false;
        beep_n(1);  // sinyal "gesture ready"
        Serial.println("[INFO] Cooldown selesai — gesture ready");
    }

    // Windows with too little physical motion (peak gyro magnitude below the
    // per-class GYRO_MOTION_THRESHOLD_BY_CLASS entry) are forced to idle
    // regardless of confidence — rejects slight/unintentional hand movement.
    int shown = (max_prob >= CONF_THRESH && motion_peak >= GYRO_MOTION_THRESHOLD_BY_CLASS[best]) ? best : 0;
    if (shown != last_shown) {
        last_shown = shown;
        if (shown != 0 && !cooldown_active) {
            send_pred(MODEL_GESTURE_NAMES[shown], max_prob, latency_us);
            beep_n(2);  // sinyal "gesture terbaca"
            last_gesture_ms = now;
            cooldown_active = true;
        }
    }
}
