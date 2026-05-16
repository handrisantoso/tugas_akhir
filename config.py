"""
Gesture Glove — Central Configuration
ESP32-S3 (XIAO Seeed) + MPU6050 → USB HID Keyboard
"""

import os

# ─── IMU & Sampling ─────────────────────────────────────────────
SAMPLE_RATE_HZ = 100
WINDOW_SIZE    = 250          # samples per gesture window (2.5 sec at 100 Hz)
NUM_AXES       = 6            # ax, ay, az, gx, gy, gz
NUM_FEATURES   = WINDOW_SIZE * NUM_AXES  # 1500
NUM_CLASSES    = 5
OVERLAP        = 0.5          # inference-only sliding window overlap (50%)
STEP_SIZE      = int(WINDOW_SIZE * (1 - OVERLAP))  # 125 samples = 1.25 sec

# ─── Serial ──────────────────────────────────────────────────────
BAUD_RATE      = 115200

# ─── Hardware (XIAO Seeed ESP32-S3) ─────────────────────────────
I2C_SDA_PIN       = 4   # D4 (default ESP32-S3 I2C SDA)
I2C_SCL_PIN       = 5   # D5 (default ESP32-S3 I2C SCL)
MPU6050_ADDR      = 0x69      # AD0 pulled HIGH
MPU_ACCEL_RANGE   = 8         # ±8g
MPU_GYRO_RANGE    = 500       # ±500°/s

# ─── Gesture Definitions ────────────────────────────────────────
GESTURE_NAMES  = ["idle", "flick_up", "wave_left", "wave_right", "flick_down"]
GESTURE_LABELS = {name: i for i, name in enumerate(GESTURE_NAMES)}

GESTURE_COLORS = {
    "idle":        "#6B7280",   # gray
    "flick_up":    "#EF4444",   # red
    "wave_left":   "#3B82F6",   # blue
    "wave_right":  "#22C55E",   # green
    "flick_down":  "#F59E0B",   # amber
}

# HID key mapping (USB HID usage IDs for arrow keys)
HID_KEY_MAP = {
    0: None,        # idle — no key
    1: 0x52,        # flick_up    → Arrow Up
    2: 0x50,        # wave_left  → Arrow Left
    3: 0x4F,        # wave_right → Arrow Right
    4: 0x51,        # flick_down  → Arrow Down
}

# ─── Inference Thresholds ────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEBOUNCE_MS = 400

# ─── Model Defaults ─────────────────────────────────────────────
DEFAULT_EPOCHS     = 80
DEFAULT_BATCH_SIZE = 32
DEFAULT_RF_TREES   = 150
DEFAULT_RF_DEPTH   = 15
VALIDATION_SPLIT   = 0.2
TFLITE_ARENA_KB    = 100      # TFLite arena size for ESP32

# ─── Paths (relative to project root) ───────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
HEADERS_DIR   = os.path.join(PROJECT_ROOT, "output_headers")
FIRMWARE_RF_DIR   = os.path.join(PROJECT_ROOT, "firmware", "gesture_glove_rf")
FIRMWARE_MLP_DIR  = os.path.join(PROJECT_ROOT, "firmware", "gesture_glove_mlp")
DEFAULT_CSV   = os.path.join(DATASET_DIR, "gesture_dataset.csv")

# Create directories on import
for d in [DATASET_DIR, MODELS_DIR, HEADERS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── CSV Column Names ───────────────────────────────────────────
def get_csv_columns():
    """Generate column names for the 'Long' format dataset CSV."""
    return ["label", "sample_id", "timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"]

CSV_COLUMNS = get_csv_columns()

# ─── Axis Names ──────────────────────────────────────────────────
AXIS_NAMES  = ["Accel X", "Accel Y", "Accel Z", "Gyro X", "Gyro Y", "Gyro Z"]
AXIS_COLORS = ["#EF4444", "#22C55E", "#3B82F6", "#F59E0B", "#A855F7", "#EC4899"]
