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

// ─── IMU circular buffer ──────────────────────────────────────
static float imu_buf[WINDOW_SIZE][NUM_AXES];
static int buf_idx = 0;
static bool buf_full = false;
static int samples_since_infer = 0;

// ─── Model input (1500 raw values, normalized in-place) ──────
static float model_input[NUM_RAW_FEATURES];
static unsigned long last_fired_ms[5] = {0};
static unsigned long last_sample_us = 0;

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

    // ── Prepare model input: flatten 1500 raw values + normalize ──
    int start = buf_idx;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int idx = (start + i) % WINDOW_SIZE;
        for (int a = 0; a < NUM_AXES; a++) {
            int fi = i * NUM_AXES + a;
            model_input[fi] = (imu_buf[idx][a] - scaler_mean[fi]) / scaler_scale[fi];
        }
    }

    // ── Quantize input ──
    TfLiteTensor* in = interpreter->input(0);
    float scale = in->params.scale;
    int zp = in->params.zero_point;
    for (int i = 0; i < MODEL_NUM_FEATURES; i++) {
        in->data.int8[i] = (int8_t) constrain((int)(model_input[i] / scale + zp), -128, 127);
    }

    // ── Inference ──
    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("[ERR] Invoke failed!");
        return;
    }

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