/*
 * Gesture Glove — MLP (TFLite Micro) Firmware
 * Optimized for ESP32-S3 + MPU6050
 *
 * Realtime inference: raw IMU → 36 statistical features → scale → TFLite int8 → buzzer output
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "mlp_model_data.h"
#include "measurements.h"

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
#define TENSOR_ARENA_SIZE   (100 * 1024)
#define CONF_THRESH         0.80
// Min gap between two GESTURE triggers. Must exceed the inference interval
// (STEP_SIZE / SAMPLE_RATE = 1.25 s) to actually gate, else it is a no-op.
#define OUTPUT_COOLDOWN_MS  5000
// Minimum peak gyro magnitude (rad/s, MPU6050 native units) required within
// a window before a non-idle prediction is accepted. Idle/small hand jitter
// measures ~0.01-0.02 rad/s in this dataset; real flick/wave gestures are
// ~3.6-9.3 rad/s. Raise this if slight movements still trigger false gestures,
// lower it if intentional-but-gentle gestures get suppressed.
#define GYRO_MOTION_THRESHOLD 1.0f

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

// ─── TFLite ───────────────────────────────────────────────────
static tflite::MicroMutableOpResolver<3> resolver;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// ─── Hardware ─────────────────────────────────────────────────
static Adafruit_MPU6050 mpu;

// ─── IMU circular buffer ──────────────────────────────────────
static float imu_buf[WINDOW_SIZE][NUM_AXES];
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;

// ─── Model I/O ───────────────────────────────────────────────
static float feat_buf[MODEL_NUM_FEATURES];   // 36
static int last_shown = -1;                  // last printed class (-1 = none yet)
static unsigned long last_gesture_ms = 0;    // time of last non-idle trigger (cooldown anchor)
static unsigned long last_sample_us = 0;

// ─── Measurement Stats ────────────────────────────────────────
static InferenceStats inf_stats;

// ─── Feature extraction (36 features, in-place normalize via scaler) ────
static inline void extract_features(int start) {
    for (int a = 0; a < NUM_AXES; a++) {
        float sum = 0, sumSq = 0;
        float mn = 1e6, mx = -1e6;

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

    // Init TFLite
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

    // Init measurement stats
    stats_init(&inf_stats);

    // Print startup info
    Serial.printf("[INFO] Model: MLP | Features: %d | Arena: %d KB\n",
                  MODEL_NUM_FEATURES, TENSOR_ARENA_SIZE / 1024);
    Serial.printf("[INFO] Heap: %d / %d bytes\n", ESP.getFreeHeap(), ESP.getHeapSize());

    Serial.println("[OK] MLP Ready");
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

    // ── Feature extraction & normalization ──
    extract_features(buf_idx);
    float motion_peak = motion_gyro_peak(buf_idx);

    // ── Quantize input ──
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(feat_buf[i] / scale + zp), -128, 127);
    }

    // --- Inference timing (Model Only - comparable to PC benchmark) ---
    int64_t t0 = esp_timer_get_time();
    bool ok = (interpreter->Invoke() == kTfLiteOk);
    int64_t t1 = esp_timer_get_time();
    int64_t latency_us = t1 - t0;

    if (!ok) {
        stats_record(&inf_stats, latency_us, false);
        Serial.println("[ERR] Invoke failed");
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

    // --- Periodic stats (every 100 inferences) ---
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

    // --- Stability report (every 30,000 inferences) ---
    if (inf_stats.total_count > 0 && inf_stats.total_count % 30000 == 0) {
        int64_t dur = (esp_timer_get_time() - inf_stats.test_start_us) / 1000000;
        float sr = 100.0f * (float)inf_stats.ok_count / inf_stats.total_count;
        Serial.printf(
            "[STABILITY] dur=%ds total=%u ok=%u fail=%u rate=%.2f%% avg=%.0fus fps=%.0f heap_min=%lld\n",
            (int)dur, inf_stats.total_count, inf_stats.ok_count, inf_stats.fail_count,
            sr, stats_avg_latency(&inf_stats), stats_avg_fps(&inf_stats), inf_stats.heap_min
        );
    }

    // --- Prediction output (idle included) with gesture cooldown ---
    // Print only on a class CHANGE. Idle is always shown (no beep). A non-idle
    // gesture only fires if OUTPUT_COOLDOWN_MS has passed since the last gesture,
    // so one physical motion triggers once and quick repeats are suppressed.
    // Low-confidence readings fall back to idle (class 0). Windows with too
    // little physical motion (peak gyro magnitude below GYRO_MOTION_THRESHOLD)
    // are also forced to idle, regardless of model confidence — this rejects
    // slight/unintentional hand movement that the model might misclassify.
    int shown = (max_prob >= CONF_THRESH && motion_peak >= GYRO_MOTION_THRESHOLD) ? best : 0;
    unsigned long now = millis();
    if (shown != last_shown) {
        last_shown = shown;
        if (shown == 0) {
            Serial.printf("[PRED] %s %.2f %lld %ld\n",
                MODEL_GESTURE_NAMES[shown], max_prob, latency_us, ESP.getFreeHeap());
        } else if (now - last_gesture_ms >= OUTPUT_COOLDOWN_MS) {
            Serial.printf("[PRED] %s %.2f %lld %ld\n",
                MODEL_GESTURE_NAMES[shown], max_prob, latency_us, ESP.getFreeHeap());
            beep_n(2);  // sinyal "gesture terbaca"
            last_gesture_ms = now;
        }
        // gesture within cooldown: suppressed (no print, no beep)
    }
}
