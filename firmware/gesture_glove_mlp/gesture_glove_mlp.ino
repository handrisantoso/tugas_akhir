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
static unsigned long last_fired_ms[5] = {0};
static unsigned long last_sample_us = 0;

// ─── Feature extraction (36 features, in-place normalize) ────
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

    // ── Quantize input ──
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(feat_buf[i] / scale + zp), -128, 127);
    }

    // ── Inference ──
    if (interpreter->Invoke() != kTfLiteOk) return;

    TfLiteTensor* out = interpreter->output(0);
    float max_prob = -1.0f;
    int best = 0;
    for (int i = 0; i < 5; i++) {
        float p = (out->data.int8[i] - out->params.zero_point) * out->params.scale;
        if (p > max_prob) { max_prob = p; best = i; }
    }

    Serial.printf("[%s] %.2f\n", MODEL_GESTURE_NAMES[best], max_prob);

    // ── Buzzer output ──
    if (max_prob >= CONF_THRESH && best != 0) {
        unsigned long now = millis();
        if (now - last_fired_ms[best] > DEBOUNCE_MS) {
            beep_n(best);       // 1=flick_up, 2=wave_left, 3=wave_right, 4=flick_down
            last_fired_ms[best] = now;
        }
    }
}