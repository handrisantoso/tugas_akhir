/*
 * Gesture Glove — Recording Firmware (ESP32-S3)
 * XIAO Seeed ESP32-S3 + MPU6050 → Stream raw IMU data
 *
 * Purpose: Record dataset — no model inference, no HID, just raw data.
 * Data:    ax,ay,az,gx,gy,gz at 100Hz
 * Output:  USB Serial  AND  BLE (Nordic UART Service) — pilih salah satu di GUI.
 *          BLE memungkinkan perekaman NIRKABEL (tanpa kabel USB).
 */

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ─── Configuration ────────────────────────────────────────────
#define SAMPLE_RATE_HZ    100
#define SAMPLE_INTERVAL   (1000000 / SAMPLE_RATE_HZ)  // microseconds

// Nordic UART Service (NUS) — standar "serial over BLE"
#define NUS_SERVICE_UUID  "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
#define NUS_TX_UUID       "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  // device → host (notify)

// ─── Global Objects ────────────────────────────────────────────
Adafruit_MPU6050 mpu;
unsigned long last_sample_us = 0;

static BLEServer*         pServer  = nullptr;
static BLECharacteristic* pTxChar  = nullptr;
static volatile bool      ble_connected = false;

// ─── BLE callbacks ─────────────────────────────────────────────
class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override {
        ble_connected = true;
        Serial.println("[BLE] Client connected");
    }
    void onDisconnect(BLEServer* s) override {
        ble_connected = false;
        Serial.println("[BLE] Client disconnected");
        s->getAdvertising()->start();   // izinkan reconnect
    }
};

static void init_ble() {
    BLEDevice::init("GestureGlove");
    BLEDevice::setMTU(247);             // notifikasi besar (satu baris CSV utuh)

    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService* svc = pServer->createService(NUS_SERVICE_UUID);
    pTxChar = svc->createCharacteristic(
        NUS_TX_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    pTxChar->addDescriptor(new BLE2902());
    svc->start();

    BLEAdvertising* adv = pServer->getAdvertising();
    adv->addServiceUUID(NUS_SERVICE_UUID);
    adv->setScanResponse(true);
    adv->start();
    Serial.println("[BLE] Advertising as 'GestureGlove'");
}

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

    init_ble();

    Serial.println("[READY] Gesture Glove recording firmware");
    Serial.println("[READY] Streaming ax,ay,az,gx,gy,gz at 100Hz (Serial + BLE)");

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
        // Build one CSV line once, send via both transports
        char line[64];
        int n = snprintf(line, sizeof(line),
            "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
            accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
            gyro.gyro.x, gyro.gyro.y, gyro.gyro.z);

        Serial.print(line);                       // USB Serial

        if (ble_connected && pTxChar && n > 0) {  // BLE (NUS notify)
            pTxChar->setValue((uint8_t*)line, n);
            pTxChar->notify();
        }
    }
}
