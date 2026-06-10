/*
 * Gesture Glove — 1D-CNN (TFLite Micro) Firmware
 * Optimized for ESP32-S3 + MPU6050
 *
 * Inference: raw IMU → scale 1500 values → TFLite int8 → buzzer output
 *
 * Model ops: CONV_2D, RESHAPE, ADD, MEAN, FULLY_CONNECTED, SOFTMAX, etc.
 *            (GlobalAveragePooling1D → MEAN in TFLite)
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "cnn1d_model_data.h"
#include "measurements.h"

// ─── Config ──────────────────────────────────────────────────
#define I2C_SDA             5
#define I2C_SCL             6
#define MPU_ADDR            0x69
#define BUZZER_PIN          8     // GPIO8 / D9 / A9
#define SAMPLE_RATE_HZ      100
#define SAMPLE_INTERVAL_US  (1000000 / SAMPLE_RATE_HZ)
#define WINDOW_SIZE         250    // timesteps
#define NUM_AXES            6
#define NUM_RAW_FEATURES   1500  // WINDOW_SIZE * NUM_AXES
#define STEP_SIZE           125
#define DEBOUNCE_MS         400
#define TENSOR_ARENA_SIZE   (220 * 1024)  // CNN1D: 205KB model + activations
#define CONF_THRESH         0.80
// Min gap between two GESTURE triggers. Must exceed the inference interval
// (STEP_SIZE / SAMPLE_RATE = 1.25 s) to actually gate, else it is a no-op.
#define OUTPUT_COOLDOWN_MS  5000

// Use PSRAM on ESP32-S3 to hold large buffers (saves internal DRAM)
#if defined(ESP32) && defined(SOC_PSRAM_SIZE)
  #define USE_PSRAM
#endif

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

// ─── TFLite — CNN1D ops (from tflm_esp32 library) ────────────
static tflite::MicroMutableOpResolver<15> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ─── Hardware ─────────────────────────────────────────────────
static Adafruit_MPU6050 mpu;

#ifdef USE_PSRAM
  #include <esp_psram.h>
  #define PSRAM_ALLOC(type, nelem) (type*)ps_malloc((nelem) * sizeof(type))
#else
  #define PSRAM_ALLOC(type, nelem) (type*)heap_caps_malloc((nelem) * sizeof(type), MALLOC_CAP_8BIT)
#endif

// ─── IMU circular buffer (in PSRAM to save internal DRAM) ───
static float* imu_buf;
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;

// ─── Model input (1500 raw values, in PSRAM) ──────────────────
static float* model_input;
static int last_shown = -1;                  // last printed class (-1 = none yet)
static unsigned long last_gesture_ms = 0;    // time of last non-idle trigger (cooldown anchor)
static unsigned long last_sample_us = 0;

// ─── Measurement Stats ────────────────────────────────────────
static InferenceStats inf_stats;

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

    // Allocate PSRAM buffers first (fail fast if PSRAM unavailable)
    imu_buf = PSRAM_ALLOC(float, WINDOW_SIZE * NUM_AXES);
    model_input = PSRAM_ALLOC(float, NUM_RAW_FEATURES);
    if (!imu_buf || !model_input) {
        Serial.println("[ERR] PSRAM allocation failed!");
        while (1) delay(100);
    }
    Serial.printf("[PSRAM] imu_buf=%p model_input=%p free=%d\n",
        (void*)imu_buf, (void*)model_input, ESP.getFreePsram());

    // Register CNN1D ops (15 total — includes MEAN for GlobalAveragePooling1D)
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
    resolver.AddMean();              // ← GlobalAveragePooling1D compiles to MEAN
    resolver.AddDequantize();
    resolver.AddTranspose();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    // Init TFLite
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

    // Init measurement stats
    stats_init(&inf_stats);
    Serial.printf("[INFO] Model: CNN1D | Features: %d | Arena: %d KB\n",
                  MODEL_NUM_FEATURES, TENSOR_ARENA_SIZE / 1024);
    Serial.printf("[INFO] Heap: %d / %d bytes\n", ESP.getFreeHeap(), ESP.getHeapSize());

    Serial.printf("[OK] CNN1D Ready (features=%d)\n", MODEL_NUM_FEATURES);
    beep_n(2);  // startup: 2 beeps = CNN1D model
    last_sample_us = micros();
}


// ─── Loop ───────────────────────────────────────────────────────
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

    // ── Prepare model input: flatten 1500 raw values + normalize ──
    int start = buf_idx;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (start + i) % WINDOW_SIZE;
        for (int a = 0; a < NUM_AXES; a++) {
            int fi = i * NUM_AXES + a;
            model_input[fi] = (imu_buf[idx * NUM_AXES + a] - scaler_mean[fi]) / scaler_scale[fi];
        }
    }

    // ── Quantize input ──
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(model_input[i] / scale + zp), -128, 127);
    }

    // ── Inference with timing ──
    int64_t t0 = esp_timer_get_time();
    bool ok = (interpreter->Invoke() == kTfLiteOk);
    int64_t t1 = esp_timer_get_time();
    int64_t latency_us = t1 - t0;

    if (!ok) {
        stats_record(&inf_stats, latency_us, false);
        Serial.println("[ERR] Invoke failed!");
        return;
    }

    stats_record(&inf_stats, latency_us, true);

    TfLiteTensor* out = interpreter->output(0);
    float max_prob = -1.0f;
    int best = 0;
    for (int i = 0; i < 5; i++) {
        float p = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
        if (p > max_prob) { max_prob = p; best = i; }
    }

    // ── Periodic stats (every 100 inferences) ──
    if (inf_stats.total_count > 0 && inf_stats.total_count % 100 == 0) {
        float avg_lat = stats_avg_latency(&inf_stats);
        float fps = stats_avg_fps(&inf_stats);
        Serial.printf(
            "[STATS] count=%u ok=%u fail=%u min=%lldus max=%lldus avg=%.0fus fps=%.0f heap=%ld\n",
            inf_stats.total_count, inf_stats.ok_count, inf_stats.fail_count,
            inf_stats.latency_min, inf_stats.latency_max,
            avg_lat, fps, ESP.getFreeHeap()
        );
    }

    // ── Stability report (every 30,000 inferences) ──
    if (inf_stats.total_count > 0 && inf_stats.total_count % 30000 == 0) {
        int64_t dur = (esp_timer_get_time() - inf_stats.test_start_us) / 1000000;
        float sr = 100.0f * (float)inf_stats.ok_count / inf_stats.total_count;
        Serial.printf(
            "[STABILITY] dur=%ds total=%u ok=%u fail=%u rate=%.2f%% avg=%.0fus fps=%.0f heap_min=%lld\n",
            (int)dur, inf_stats.total_count, inf_stats.ok_count, inf_stats.fail_count,
            sr, stats_avg_latency(&inf_stats), stats_avg_fps(&inf_stats), inf_stats.heap_min
        );
    }

    // ── Prediction output (idle included) with gesture cooldown ──
    // Print only on a class CHANGE. Idle is always shown (no beep). A non-idle
    // gesture only fires if OUTPUT_COOLDOWN_MS has passed since the last gesture,
    // so one physical motion triggers once and quick repeats are suppressed.
    // Low-confidence readings fall back to idle (class 0).
    int shown = (max_prob >= CONF_THRESH) ? best : 0;
    unsigned long now = millis();
    if (shown != last_shown) {
        last_shown = shown;
        if (shown == 0) {
            Serial.printf("[PRED] %s %.2f %lld %ld\n",
                MODEL_GESTURE_NAMES[shown], max_prob, latency_us, ESP.getFreeHeap());
        } else if (now - last_gesture_ms >= OUTPUT_COOLDOWN_MS) {
            Serial.printf("[PRED] %s %.2f %lld %ld\n",
                MODEL_GESTURE_NAMES[shown], max_prob, latency_us, ESP.getFreeHeap());
            beep_n(shown);
            last_gesture_ms = now;
        }
        // gesture within cooldown: suppressed (no print, no beep)
    }
}