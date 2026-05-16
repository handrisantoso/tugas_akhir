/*
 * Gesture Glove — Recording Firmware (ESP32-S3)
 * XIAO Seeed ESP32-S3 + MPU6050 → Stream raw IMU data over USB Serial
 *
 * Purpose: Record dataset — no model inference, no HID, just raw data.
 * Data:    ax,ay,az,gx,gy,gz at 100Hz via Serial
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// ─── Configuration ────────────────────────────────────────────
#define SAMPLE_RATE_HZ    100
#define SAMPLE_INTERVAL   (1000000 / SAMPLE_RATE_HZ)  // microseconds

// ─── Global Objects ────────────────────────────────────────────
Adafruit_MPU6050 mpu;
unsigned long last_sample_us = 0;

// ═══════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════
void setup() {
    Serial.begin(115200);
    delay(100);  // Small settle time

    Wire.begin(5, 6);  // SDA=GPIO5 (D4), SCL=GPIO6 (D5)

    // Try 0x68 first (default), then fall back to 0x69 (AD0 HIGH)
    if (!mpu.begin(0x68, &Wire)) {
        if (!mpu.begin(0x69, &Wire)) {
            Serial.println("[ERROR] MPU6050 not found!");
            while (1) { delay(100); }
        } else {
            Serial.println("[INIT] MPU6050 found at 0x69");
        }
    } else {
        Serial.println("[INIT] MPU6050 found at 0x68");
    }

    // Configure MPU6050 ranges
    mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    Serial.println("[READY] Gesture Glove recording firmware");
    Serial.println("[READY] Streaming ax,ay,az,gx,gy,gz at 100Hz");

    delay(500);
    last_sample_us = micros();
}

// ═══════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════
void loop() {
    // Drain any stale commands left in the serial buffer
    while (Serial.available()) Serial.read();

    // Sample at fixed 100Hz rate
    unsigned long now_us = micros();
    if (now_us - last_sample_us < SAMPLE_INTERVAL) return;
    last_sample_us = now_us;

    sensors_event_t accel, gyro, temp;
    if (mpu.getEvent(&accel, &gyro, &temp)) {
        Serial.printf("%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            accel.acceleration.x,
            accel.acceleration.y,
            accel.acceleration.z,
            gyro.gyro.x,
            gyro.gyro.y,
            gyro.gyro.z
        );
    }
}