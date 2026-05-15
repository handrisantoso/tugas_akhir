/*
 * Gesture Glove — ESP32-S3 Firmware
 * XIAO Seeed ESP32-S3 + MPU6050 → USB HID Keyboard
 *
 * Serial Commands:
 *   S  — Stream CSV only (recording mode)
 *   I  — Infer + HID only (normal use)
 *   B  — Both stream + infer (debug)
 *   T0.85 — Set confidence threshold live
 *
 * Serial Output (stream mode):
 *   ax,ay,az,gx,gy,gz  (CSV at 100Hz)
 *
 * Serial Output (infer mode):
 *   [PRED] gesture_name confidence latency_ms
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "USB.h"
#include "USBHIDKeyboard.h"

// ─── Include the trained model header ─────────────────────────
// ─── Include the trained model header ─────────────────────────
// The pipeline exports these headers directly to this folder.
// Uncomment the one you want to use:
#include "rf_model_data.h"
// #include "mlp_model_data.h"
// #include "cnn1d_model_data.h"

// ─── Configuration ────────────────────────────────────────────
#define I2C_SDA           4
#define I2C_SCL           5
#define MPU_ADDR          0x69    // AD0 pulled HIGH

#define SAMPLE_RATE_HZ    100
#define SAMPLE_INTERVAL   (1000000 / SAMPLE_RATE_HZ)  // microseconds
#define WINDOW_SIZE       250
#define NUM_AXES          6
#define NUM_FEATURES      1500    // WINDOW_SIZE * NUM_AXES
#define NUM_CLASSES       5
#define STEP_SIZE         125     // 50% overlap for inference
#define DEBOUNCE_MS       400

// ─── Gesture Names & Keys ─────────────────────────────────────
const char* GESTURE_NAMES[NUM_CLASSES] = {
    "idle", "swipe_up", "swipe_left", "swipe_right", "swipe_down"
};

const uint8_t GESTURE_KEYS[NUM_CLASSES] = {
    0,                  // idle — no key
    KEY_UP_ARROW,       // swipe_up
    KEY_LEFT_ARROW,     // swipe_left
    KEY_RIGHT_ARROW,    // swipe_right
    KEY_DOWN_ARROW      // swipe_down
};

// ─── Operating Mode ───────────────────────────────────────────
enum Mode { MODE_STREAM, MODE_INFER, MODE_BOTH };
Mode currentMode = MODE_STREAM;

// ─── Global Objects ───────────────────────────────────────────
Adafruit_MPU6050 mpu;
USBHIDKeyboard Keyboard;

// ─── Buffers ──────────────────────────────────────────────────
float imu_buffer[WINDOW_SIZE][NUM_AXES];
int buffer_idx = 0;
int samples_since_infer = 0;
bool buffer_full = false;

// ─── Inference State ──────────────────────────────────────────
float confidence_threshold = 0.85;
unsigned long last_fired_ms[NUM_CLASSES] = {0};
float normalized[NUM_FEATURES];

// ─── Timing ───────────────────────────────────────────────────
unsigned long last_sample_us = 0;
unsigned long infer_start_ms = 0;

// ─── Forward Declarations ─────────────────────────────────────
void readIMU(float* ax, float* ay, float* az, float* gx, float* gy, float* gz);
int runInference(float* input, float* conf);
void handleSerial();

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);

    // Initialize USB HID
    Keyboard.begin();
    USB.begin();

    // Initialize I2C and MPU6050
    Wire.begin(I2C_SDA, I2C_SCL);

    if (!mpu.begin(MPU_ADDR, &Wire)) {
        Serial.println("[ERROR] MPU6050 not found at 0x69!");
        while (1) { delay(100); }
    }

    // Configure MPU6050 ranges
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("[INIT] Gesture Glove ready — XIAO ESP32-S3 + MPU6050");
    Serial.println("[INIT] Send 'S' for stream, 'I' for infer, 'B' for both");

    #ifdef RF_MODEL_DATA_H
    Serial.println("[INIT] Model Loaded: Random Forest");
    #elif defined(MLP_MODEL_DATA_H) || defined(CNN1D_MODEL_DATA_H)
    Serial.println("[INIT] Model Loaded: TFLite (Ensure TFLite lib is installed)");
    #else
    Serial.println("[INIT] WARNING: No model header included! Using placeholder.");
    #endif

    delay(500);
    last_sample_us = micros();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
    // Handle serial commands
    handleSerial();

    // Sample at fixed rate
    unsigned long now_us = micros();
    if (now_us - last_sample_us < SAMPLE_INTERVAL) {
        return;
    }
    last_sample_us = now_us;

    // Read IMU
    float ax, ay, az, gx, gy, gz;
    readIMU(&ax, &ay, &az, &gx, &gy, &gz);

    // Store in circular buffer
    imu_buffer[buffer_idx][0] = ax;
    imu_buffer[buffer_idx][1] = ay;
    imu_buffer[buffer_idx][2] = az;
    imu_buffer[buffer_idx][3] = gx;
    imu_buffer[buffer_idx][4] = gy;
    imu_buffer[buffer_idx][5] = gz;

    buffer_idx = (buffer_idx + 1) % WINDOW_SIZE;
    if (buffer_idx == 0) buffer_full = true;
    samples_since_infer++;

    // Stream CSV if in stream or both mode
    if (currentMode == MODE_STREAM || currentMode == MODE_BOTH) {
        Serial.printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", ax, ay, az, gx, gy, gz);
    }

    // Run inference every STEP_SIZE samples
    if ((currentMode == MODE_INFER || currentMode == MODE_BOTH) &&
        buffer_full && samples_since_infer >= STEP_SIZE) {

        samples_since_infer = 0;
        infer_start_ms = millis();

        // Build flattened + normalized feature vector from circular buffer
        int start = buffer_idx;  // oldest sample is at current write position
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int idx = (start + i) % WINDOW_SIZE;
            for (int a = 0; a < NUM_AXES; a++) {
                int feat_idx = i * NUM_AXES + a;
                normalized[feat_idx] = imu_buffer[idx][a];
            }
        }

        // Apply StandardScaler normalization (if model header exists)
        #if defined(RF_MODEL_DATA_H) || defined(MLP_MODEL_DATA_H) || defined(CNN1D_MODEL_DATA_H)
        for (int i = 0; i < NUM_FEATURES; i++) {
            normalized[i] = (normalized[i] - scaler_mean[i]) / scaler_scale[i];
        }
        #endif

        // Run model inference
        float confidence = 0.0;
        int predicted = runInference(normalized, &confidence);

        unsigned long latency = millis() - infer_start_ms;

        // Output prediction
        Serial.printf("[PRED] %s %.2f %lums\n",
                      GESTURE_NAMES[predicted], confidence, latency);

        // HID keypress logic
        if (confidence >= confidence_threshold && predicted != 0) {  // 0 = idle
            unsigned long now = millis();
            if (now - last_fired_ms[predicted] > DEBOUNCE_MS) {
                Keyboard.press(GESTURE_KEYS[predicted]);
                delay(30);
                Keyboard.releaseAll();
                last_fired_ms[predicted] = now;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════
//  IMU READING
// ═══════════════════════════════════════════════════════════════
void readIMU(float* ax, float* ay, float* az, float* gx, float* gy, float* gz) {
    sensors_event_t accel, gyro, temp;
    if (mpu.getEvent(&accel, &gyro, &temp)) {
        *ax = accel.acceleration.x;
        *ay = accel.acceleration.y;
        *az = accel.acceleration.z;
        *gx = gyro.gyro.x;
        *gy = gyro.gyro.y;
        *gz = gyro.gyro.z;
    }
}

// ═══════════════════════════════════════════════════════════════
//  INFERENCE ENGINE
// ═══════════════════════════════════════════════════════════════
int runInference(float* input, float* conf) {
    #ifdef RF_MODEL_DATA_H
    // Pure C Random Forest (m2cgen)
    return rf_predict(input, conf);

    #elif defined(MLP_MODEL_DATA_H) || defined(CNN1D_MODEL_DATA_H)
    // TFLite Micro Placeholder
    // Note: Requires installation of "TensorFlowLite_ESP32" library
    // and initialization of interpreter (see bottom of file).
    *conf = 0.0;
    return 0;

    #else
    // Fallback: Simple motion-detection placeholder
    float sum_accel = 0;
    for (int i = 0; i < NUM_FEATURES; i += NUM_AXES) {
        sum_accel += fabs(input[i+0]) + fabs(input[i+1]);
    }
    float avg_motion = sum_accel / WINDOW_SIZE;
    if (avg_motion < 0.5) { *conf = 0.95; return 0; }
    *conf = min(avg_motion / 5.0f, 0.99f);
    return (input[0] > 0) ? 3 : 2; // dummy left/right
    #endif
}

// ═══════════════════════════════════════════════════════════════
//  SERIAL COMMAND HANDLER
// ═══════════════════════════════════════════════════════════════
void handleSerial() {
    if (!Serial.available()) return;

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "S") {
        currentMode = MODE_STREAM;
        Serial.println("[MODE] Stream CSV only");
    }
    else if (cmd == "I") {
        currentMode = MODE_INFER;
        Serial.println("[MODE] Infer + HID only");
    }
    else if (cmd == "B") {
        currentMode = MODE_BOTH;
        Serial.println("[MODE] Both stream + infer");
    }
    else if (cmd.startsWith("T")) {
        float newThresh = cmd.substring(1).toFloat();
        if (newThresh >= 0.1 && newThresh <= 1.0) {
            confidence_threshold = newThresh;
            Serial.printf("[THRESH] Set to %.2f\n", confidence_threshold);
        }
    }
}

/*
 * ═══════════════════════════════════════════════════════════════
 *  TFLITE MICRO INTEGRATION (Optional)
 * ═══════════════════════════════════════════════════════════════
 *
 * If you use MLP or CNN1D models:
 * 1. Install "TensorFlowLite_ESP32" in Arduino Library Manager.
 * 2. Uncomment the TFLITE includes and implementation below.
 *
 * #include <TensorFlowLite_ESP32.h>
 * #include "tensorflow/lite/micro/all_ops_resolver.h"
 * #include "tensorflow/lite/micro/micro_interpreter.h"
 * #include "tensorflow/lite/schema/schema_generated.h"
 *
 * // Note: Requires model_data and its length from the exported header
 * // Example for MLP:
 *
 * static tflite::MicroInterpreter* interpreter = nullptr;
 * static uint8_t* tensor_arena = nullptr;
 *
 * void setupTFLite() {
 *     static tflite::AllOpsResolver resolver;
 *     const tflite::Model* model = tflite::GetModel(model_data);
 *     tensor_arena = (uint8_t*)malloc(100 * 1024); // 100KB
 *     static tflite::MicroInterpreter static_interpreter(
 *         model, resolver, tensor_arena, 100 * 1024);
 *     interpreter = &static_interpreter;
 *     interpreter->AllocateTensors();
 * }
 *
 * int predictTFLite(float* input, float* confidence) {
 *     TfLiteTensor* input_tensor = interpreter->input(0);
 *     // Quantize input
 *     float scale = input_tensor->params.scale;
 *     int zp = input_tensor->params.zero_point;
 *     for (int i = 0; i < NUM_FEATURES; i++) {
 *         input_tensor->data.int8[i] = (int8_t)(input[i] / scale + zp);
 *     }
 *
 *     interpreter->Invoke();
 *
 *     TfLiteTensor* output_tensor = interpreter->output(0);
 *     int best = 0;
 *     float max_prob = -1.0;
 *     for (int i = 0; i < NUM_CLASSES; i++) {
 *         float prob = (output_tensor->data.int8[i] - output_tensor->params.zero_point) * output_tensor->params.scale;
 *         if (prob > max_prob) { max_prob = prob; best = i; }
 *     }
 *     *confidence = max_prob;
 *     return best;
 * }
 */
