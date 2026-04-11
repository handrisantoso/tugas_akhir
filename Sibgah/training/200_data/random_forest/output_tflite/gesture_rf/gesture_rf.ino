/*
 * gesture_rf.ino — Gesture detection using Random Forest (no TFLite needed)
 *
 * Required files (copy all into this sketch folder):
 *   model.h          — from random_forest/output_tflite/
 *   scaler_params.h  — from random_forest/output_tflite/
 *   labels.h         — from random_forest/output_tflite/
 *
 * Library dependencies (install via Arduino Library Manager):
 *   — I2Cdevlib-MPU6050  (jrowberg)  https://github.com/jrowberg/i2cdevlib
 */

#include "I2Cdev.h"
#include "MPU6050.h"
#include "model.h"
#include "scaler_params.h"
#include "labels.h"

// ── Pins / config ──────────────────────────────────────────────────────────────
#define BUZZER_PIN          20
#define SAMPLE_INTERVAL_MS  100    // 10 Hz — matches recording rate
#define N_CHANNELS          12
#define WINDOW_SIZE_SAMPLES 25
#define N_FEATURES          72     // 12 channels × 6 stats

// ── Sensors ────────────────────────────────────────────────────────────────────
MPU6050 mpu1(0x68);
MPU6050 mpu2(0x69);

// ── Globals ───────────────────────────────────────────────────────────────────
Eloquent::ML::Port::RandomForest clf;

float           window_buf[WINDOW_SIZE_SAMPLES][N_CHANNELS];
int             sample_count  = 0;
uint32_t        last_ms       = 0;

// ── Buzzer helpers ─────────────────────────────────────────────────────────────
void beep(int freq, int dur) { tone(BUZZER_PIN, freq, dur); delay(dur + 30); }

void startChime()       { beep(800, 80); beep(800, 80); noTone(BUZZER_PIN); }
void calibrationChime() {
  beep(523, 120); beep(659, 120); beep(784, 120); beep(1047, 300);
  noTone(BUZZER_PIN);
}
void gestureChime()     { beep(1200, 60); noTone(BUZZER_PIN); }

// ── Feature extraction ────────────────────────────────────────────────────────
// Must match Python extract_features():
//   for each channel c: [mean, std, min, max, median, rms]

float _sort_buf[WINDOW_SIZE_SAMPLES];

float compute_median(float *col) {
  for (int i = 0; i < WINDOW_SIZE_SAMPLES; i++) _sort_buf[i] = col[i];
  for (int i = 1; i < WINDOW_SIZE_SAMPLES; i++) {
    float key = _sort_buf[i]; int j = i - 1;
    while (j >= 0 && _sort_buf[j] > key) { _sort_buf[j + 1] = _sort_buf[j]; j--; }
    _sort_buf[j + 1] = key;
  }
  return _sort_buf[WINDOW_SIZE_SAMPLES / 2];
}

void extract_features(float scaled[WINDOW_SIZE_SAMPLES][N_CHANNELS],
                      float features[N_FEATURES]) {
  float col[WINDOW_SIZE_SAMPLES];
  int fi = 0;
  for (int c = 0; c < N_CHANNELS; c++) {
    for (int t = 0; t < WINDOW_SIZE_SAMPLES; t++) col[t] = scaled[t][c];

    float sum = 0;
    for (int t = 0; t < WINDOW_SIZE_SAMPLES; t++) sum += col[t];
    float mean = sum / WINDOW_SIZE_SAMPLES;

    float var = 0;
    for (int t = 0; t < WINDOW_SIZE_SAMPLES; t++) var += (col[t]-mean)*(col[t]-mean);

    float mn = col[0], mx = col[0], sq = 0;
    for (int t = 1; t < WINDOW_SIZE_SAMPLES; t++) {
      if (col[t] < mn) mn = col[t];
      if (col[t] > mx) mx = col[t];
    }
    for (int t = 0; t < WINDOW_SIZE_SAMPLES; t++) sq += col[t] * col[t];

    features[fi++] = mean;
    features[fi++] = sqrt(var / WINDOW_SIZE_SAMPLES);
    features[fi++] = mn;
    features[fi++] = mx;
    features[fi++] = compute_median(col);
    features[fi++] = sqrt(sq / WINDOW_SIZE_SAMPLES);
  }
}

// ── Inference ─────────────────────────────────────────────────────────────────
void run_inference() {
  // Scale raw window per channel
  float scaled[WINDOW_SIZE_SAMPLES][N_CHANNELS];
  for (int t = 0; t < WINDOW_SIZE_SAMPLES; t++)
    for (int c = 0; c < N_CHANNELS; c++)
      scaled[t][c] = (window_buf[t][c] - SCALER_MEAN[c]) / SCALER_STD[c];

  float features[N_FEATURES];
  extract_features(scaled, features);

  int class_idx = clf.predict(features);
  const char *label = (class_idx >= 0 && class_idx < N_CLASSES)
                      ? GESTURE_CLASSES[class_idx] : "unknown";

  Serial.printf("[RF] Gesture: %s\n", label);

  if (strcmp(label, "idle") != 0) {
    gestureChime();
    digitalWrite(LED_BUILTIN, HIGH); delay(100); digitalWrite(LED_BUILTIN, LOW);
  }
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
  Wire.begin();
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  Serial.println("Initializing MPU1 (0x68)...");
  mpu1.initialize();
  if (!mpu1.testConnection()) { Serial.println("MPU1 FAILED"); while (1); }
  Serial.println("MPU1 OK");

  Serial.println("Initializing MPU2 (0x69)...");
  mpu2.initialize();
  if (!mpu2.testConnection()) { Serial.println("MPU2 FAILED"); while (1); }
  Serial.println("MPU2 OK");

  Serial.println("Calibrating – keep sensors still...");
  startChime();
  digitalWrite(LED_BUILTIN, HIGH);
  delay(5000);
  digitalWrite(LED_BUILTIN, LOW);

  // TODO: replace with your values from MPU6050_Zero sketch
  mpu1.setXAccelOffset(0); mpu1.setYAccelOffset(0); mpu1.setZAccelOffset(0);
  mpu1.setXGyroOffset(0);  mpu1.setYGyroOffset(0);  mpu1.setZGyroOffset(0);
  mpu2.setXAccelOffset(0); mpu2.setYAccelOffset(0); mpu2.setZAccelOffset(0);
  mpu2.setXGyroOffset(0);  mpu2.setYGyroOffset(0);  mpu2.setZGyroOffset(0);

  calibrationChime();
  digitalWrite(LED_BUILTIN, HIGH); delay(200); digitalWrite(LED_BUILTIN, LOW);
  Serial.println("Ready [Random Forest]");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
  uint32_t now = millis();
  if (now - last_ms < SAMPLE_INTERVAL_MS) return;
  last_ms = now;

  int16_t ax1, ay1, az1, gx1, gy1, gz1;
  int16_t ax2, ay2, az2, gx2, gy2, gz2;
  mpu1.getMotion6(&ax1, &ay1, &az1, &gx1, &gy1, &gz1);
  mpu2.getMotion6(&ax2, &ay2, &az2, &gx2, &gy2, &gz2);

  window_buf[sample_count][0]  = (float)ax1;
  window_buf[sample_count][1]  = (float)ay1;
  window_buf[sample_count][2]  = (float)az1;
  window_buf[sample_count][3]  = (float)gx1;
  window_buf[sample_count][4]  = (float)gy1;
  window_buf[sample_count][5]  = (float)gz1;
  window_buf[sample_count][6]  = (float)ax2;
  window_buf[sample_count][7]  = (float)ay2;
  window_buf[sample_count][8]  = (float)az2;
  window_buf[sample_count][9]  = (float)gx2;
  window_buf[sample_count][10] = (float)gy2;
  window_buf[sample_count][11] = (float)gz2;

  sample_count++;
  if (sample_count >= WINDOW_SIZE_SAMPLES) {
    run_inference();
    sample_count = 0;
  }
}
