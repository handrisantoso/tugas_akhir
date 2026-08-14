/*
 * Gesture Glove — 1D-CNN (TFLite Micro int8) + BLE untuk SMARTBOARD
 * XIAO Seeed ESP32-S3 + MPU6050
 *
 * Inferensi CNN1D (arena 220KB internal, buffer raw di PSRAM) → kirim prediksi
 * via BLE (Nordic UART Service) ke smartboard.py.
 * Format baris: "[PRED] <gesture> <conf> <latency_us>"
 *
 * Catatan: CNN1D + BLE = RAM cukup ketat. Bila gagal AllocateTensors / crash,
 * pertimbangkan pakai model RF/MLP untuk smartboard (jauh lebih ringan).
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cnn1d_model_data.h"

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
#define NUM_RAW_FEATURES    1500
#define STEP_SIZE           125
#define DEBOUNCE_MS         400
#define TENSOR_ARENA_SIZE   (220 * 1024)
#define CONF_THRESH         0.80
#define OUTPUT_COOLDOWN_MS  5000
// Minimum peak gyro magnitude (rad/s, MPU6050 native units) required within
// a window before a non-idle prediction is accepted. Idle/small hand jitter
// measures ~0.01-0.02 rad/s in this dataset; real flick/wave gestures are
// ~3.6-9.3 rad/s. Raise this if slight movements still trigger false gestures,
// lower it if intentional-but-gentle gestures get suppressed.
#define GYRO_MOTION_THRESHOLD 1.0f

#define NUS_SERVICE_UUID  "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_UUID       "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

#if defined(ESP32) && defined(SOC_PSRAM_SIZE)
  #define USE_PSRAM
#endif
#ifdef USE_PSRAM
  #include <esp_psram.h>
  #define PSRAM_ALLOC(type, nelem) (type*)ps_malloc((nelem) * sizeof(type))
#else
  #define PSRAM_ALLOC(type, nelem) (type*)heap_caps_malloc((nelem) * sizeof(type), MALLOC_CAP_8BIT)
#endif

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

// ─── TFLite + buffers ────────────────────────────────────────
static tflite::MicroMutableOpResolver<15> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

static Adafruit_MPU6050 mpu;
static float* imu_buf;       // PSRAM: WINDOW_SIZE * NUM_AXES
static float* model_input;   // PSRAM: NUM_RAW_FEATURES
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;
static int last_shown = -1;
static unsigned long last_gesture_ms = 0;
static unsigned long last_sample_us = 0;
static bool cooldown_active = false;

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

    imu_buf     = PSRAM_ALLOC(float, WINDOW_SIZE * NUM_AXES);
    model_input = PSRAM_ALLOC(float, NUM_RAW_FEATURES);
    if (!imu_buf || !model_input) {
        Serial.println("[ERR] PSRAM allocation failed!");
        while (1) delay(100);
    }

    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddExpandDims();
    resolver.AddConcatenation();
    resolver.AddDepthwiseConv2D();
    resolver.AddLeakyRelu();
    resolver.AddAveragePool2D();
    resolver.AddMaxPool2D();
    resolver.AddMul();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddDequantize();
    resolver.AddTranspose();
    resolver.AddFullyConnected();
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
    Serial.println("[OK] CNN1D + BLE Smartboard Ready");
    beep_n(2);
    last_sample_us = micros();
}

// ─── Loop ────────────────────────────────────────────────────
void loop() {
    unsigned long now_us = micros();
    if (now_us - last_sample_us < SAMPLE_INTERVAL_US) return;
    last_sample_us = now_us;

    sensors_event_t accel, gyro, temp;
    if (!mpu.getEvent(&accel, &gyro, &temp)) return;

    imu_buf[buf_idx * NUM_AXES + 0] = accel.acceleration.x;
    imu_buf[buf_idx * NUM_AXES + 1] = accel.acceleration.y;
    imu_buf[buf_idx * NUM_AXES + 2] = accel.acceleration.z;
    imu_buf[buf_idx * NUM_AXES + 3] = gyro.gyro.x;
    imu_buf[buf_idx * NUM_AXES + 4] = gyro.gyro.y;
    imu_buf[buf_idx * NUM_AXES + 5] = gyro.gyro.z;

    buf_idx = (buf_idx + 1) % WINDOW_SIZE;
    if (buf_idx == 0) buf_full = true;
    samples_since_infer++;
    if (!buf_full || samples_since_infer < STEP_SIZE) return;
    samples_since_infer = 0;

    // Flatten 1500 raw values + normalize. Also track peak gyro magnitude in
    // the same pass, used below to gate out low-motion windows.
    int start = buf_idx;
    float motion_peak = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (start + i) % WINDOW_SIZE;
        for (int a = 0; a < NUM_AXES; a++) {
            int fi = i * NUM_AXES + a;
            model_input[fi] = (imu_buf[idx * NUM_AXES + a] - scaler_mean[fi]) / scaler_scale[fi];
        }
        float gx = imu_buf[idx * NUM_AXES + 3];
        float gy = imu_buf[idx * NUM_AXES + 4];
        float gz = imu_buf[idx * NUM_AXES + 5];
        float gmag = sqrtf(gx * gx + gy * gy + gz * gz);
        if (gmag > motion_peak) motion_peak = gmag;
    }

    // Quantize input
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < NUM_RAW_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(model_input[i] / scale + zp), -128, 127);
    }

    int64_t t0 = esp_timer_get_time();
    bool ok = (interpreter->Invoke() == kTfLiteOk);
    int64_t latency_us = esp_timer_get_time() - t0;
    if (!ok) { Serial.println("[ERR] Invoke failed!"); return; }

    TfLiteTensor* out = interpreter->output(0);
    float max_prob = -1.0f;
    int best = 0;
    for (int i = 0; i < 5; i++) {
        float p = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
        if (p > max_prob) { max_prob = p; best = i; }
    }

    // Cek cooldown selesai → beep 1x "gesture ready"
    unsigned long now = millis();
    if (cooldown_active && (now - last_gesture_ms >= OUTPUT_COOLDOWN_MS)) {
        cooldown_active = false;
        beep_n(1);  // sinyal "gesture ready"
        Serial.println("[INFO] Cooldown selesai — gesture ready");
    }

    // Windows with too little physical motion (peak gyro magnitude below
    // GYRO_MOTION_THRESHOLD) are forced to idle regardless of confidence —
    // rejects slight/unintentional hand movement.
    int shown = (max_prob >= CONF_THRESH && motion_peak >= GYRO_MOTION_THRESHOLD) ? best : 0;
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
