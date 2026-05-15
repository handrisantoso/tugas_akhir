"""
Pipeline Integration Test
Tests the full flow: synthetic data → CSV → train → export → verify header
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    NUM_FEATURES, NUM_CLASSES, WINDOW_SIZE, NUM_AXES,
    GESTURE_NAMES, CSV_COLUMNS, DEFAULT_CSV, MODELS_DIR, HEADERS_DIR, FIRMWARE_DIR
)

def test_config():
    """Verify config consistency."""
    print("=== Config Consistency ===")
    assert NUM_FEATURES == WINDOW_SIZE * NUM_AXES == 1500, f"NUM_FEATURES mismatch: {NUM_FEATURES}"
    assert NUM_CLASSES == 5
    assert len(GESTURE_NAMES) == NUM_CLASSES
    assert len(CSV_COLUMNS) == NUM_FEATURES + 1, f"CSV columns: {len(CSV_COLUMNS)} expected {NUM_FEATURES + 1}"
    assert CSV_COLUMNS[-1] == "label"
    assert CSV_COLUMNS[0] == "ax_0"
    assert CSV_COLUMNS[5] == "gz_0"
    assert CSV_COLUMNS[6] == "ax_1"
    print("  [OK] All config values consistent")

def generate_synthetic_csv():
    """Generate a small synthetic dataset for testing."""
    print("\n=== Generating Synthetic Dataset ===")
    import pandas as pd

    np.random.seed(42)
    rows = []
    samples_per_class = 30  # Small but enough to test

    for label in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            # Create slightly different patterns per class
            base = np.random.randn(WINDOW_SIZE, NUM_AXES).astype(np.float32) * 0.5
            # Add class-specific signal
            if label == 1:   # swipe_up: positive Y accel
                base[:, 1] += np.linspace(0, 3, WINDOW_SIZE)
            elif label == 2:  # swipe_left: negative X accel
                base[:, 0] -= np.linspace(0, 3, WINDOW_SIZE)
            elif label == 3:  # swipe_right: positive X accel
                base[:, 0] += np.linspace(0, 3, WINDOW_SIZE)
            elif label == 4:  # swipe_down: negative Y accel
                base[:, 1] -= np.linspace(0, 3, WINDOW_SIZE)

            flat = base.flatten().tolist()
            flat.append(label)
            rows.append(flat)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(DEFAULT_CSV, index=False)
    print(f"  [OK] Saved {len(df)} samples to {DEFAULT_CSV}")
    print(f"  [OK] Shape: {df.shape} (expected ({samples_per_class * NUM_CLASSES}, {NUM_FEATURES + 1}))")
    return df

def test_recorder_csv_load():
    """Test CSV loading from recorder."""
    print("\n=== Recorder CSV Load ===")
    from recorder import GestureRecorder

    df, counts = GestureRecorder.load_csv(DEFAULT_CSV)
    assert df is not None, "Failed to load CSV"
    assert sum(counts.values()) == 150, f"Expected 150 total, got {sum(counts.values())}"
    for name in GESTURE_NAMES:
        assert counts[name] == 30, f"{name}: expected 30, got {counts[name]}"
    print(f"  [OK] Loaded {sum(counts.values())} samples")

    X, y = GestureRecorder.load_dataset(DEFAULT_CSV)
    assert X.shape == (150, NUM_FEATURES), f"X shape: {X.shape}"
    assert y.shape == (150,), f"y shape: {y.shape}"
    assert set(y.tolist()) == {0, 1, 2, 3, 4}
    print(f"  [OK] Dataset shapes: X={X.shape}, y={y.shape}")

def test_train_rf():
    """Test Random Forest training + m2cgen export."""
    print("\n=== Random Forest Training ===")
    from recorder import GestureRecorder
    from models import train_model

    X, y = GestureRecorder.load_dataset(DEFAULT_CSV)
    result = train_model("rf", X, y, n_estimators=10, max_depth=5)

    assert "model" in result
    assert "scaler" in result
    assert "accuracy" in result
    assert "confusion" in result
    assert "report" in result
    assert result["accuracy"] > 0.5, f"RF accuracy too low: {result['accuracy']}"
    print(f"  [OK] RF accuracy: {result['accuracy']:.1%}")
    print(f"  [OK] Confusion matrix shape: {result['confusion'].shape}")

    # Check header export
    assert "header_path" in result, f"Header export failed: {result.get('header_error', 'unknown')}"
    header_path = result["header_path"]
    assert os.path.isfile(header_path), f"Header file not found: {header_path}"

    with open(header_path, "r", encoding="utf-8") as f:
        header = f.read()
    assert "scaler_mean" in header
    assert "scaler_scale" in header
    assert "score" in header
    assert "rf_predict" in header
    assert "RF_MODEL_DATA_H" in header
    print(f"  [OK] C header exported: {header_path} ({len(header)} bytes)")

def test_train_mlp():
    """Test MLP training + TFLite export."""
    print("\n=== MLP Training ===")
    from recorder import GestureRecorder
    from models import train_model

    X, y = GestureRecorder.load_dataset(DEFAULT_CSV)
    result = train_model("mlp", X, y, epochs=3, batch_size=16, quantize_int8=True)

    assert "model" in result
    assert "history" in result
    assert "accuracy" in result
    assert "loss" in result["history"]
    assert "val_loss" in result["history"]
    assert "accuracy" in result["history"]
    print(f"  [OK] MLP accuracy: {result['accuracy']:.1%}")
    print(f"  [OK] Training epochs: {len(result['history']['loss'])}")

    # Check TFLite export
    if "tflite_path" in result:
        assert os.path.isfile(result["tflite_path"])
        print(f"  [OK] TFLite: {result.get('float_model_size', 0)/1024:.1f}KB -> {result.get('quantized_model_size', 0)/1024:.1f}KB")
    else:
        print(f"  [!] TFLite export failed: {result.get('tflite_error', 'unknown')}")

    # Check C header
    if "header_path" in result:
        assert os.path.isfile(result["header_path"])
        with open(result["header_path"], "r", encoding="utf-8") as f:
            header = f.read()
        assert "model_data" in header
        assert "scaler_mean" in header
        print(f"  [OK] C header: {result['header_path']}")
    else:
        print(f"  [!] Header export failed: {result.get('header_error', 'unknown')}")

def test_serial_protocol_parsing():
    """Test that the Python serial parser matches the firmware output format."""
    print("\n=== Serial Protocol Consistency ===")
    from recorder import SerialManager

    sm = SerialManager()
    predictions = []
    imu_samples = []

    sm.on_prediction = lambda g, c, l: predictions.append((g, c, l))
    sm.on_imu_data = lambda *args: imu_samples.append(args)

    # Test IMU CSV parsing (firmware format: %.4f,%.4f,%.4f,%.4f,%.4f,%.4f)
    sm._process_line("1.2345,0.5678,-9.8100,0.0123,-0.0456,0.0789")
    assert len(imu_samples) == 1
    assert len(imu_samples[0]) == 6
    assert abs(imu_samples[0][0] - 1.2345) < 0.001
    print("  [OK] IMU CSV parsing matches firmware format")

    # Test prediction parsing (firmware format: [PRED] %s %.2f %lums)
    sm._process_line("[PRED] swipe_right 0.94 25ms")
    assert len(predictions) == 1
    assert predictions[0] == ("swipe_right", 0.94, 25)
    print("  [OK] Prediction parsing matches firmware format")

    # Test with larger latency
    sm._process_line("[PRED] idle 0.87 1250ms")
    assert predictions[1] == ("idle", 0.87, 1250)
    print("  [OK] Large latency parsing OK")

    # Test mode commands (verify they don't crash)
    sm._process_line("[MODE] Stream CSV only")
    sm._process_line("[INIT] Gesture Glove ready")
    sm._process_line("[THRESH] Set to 0.75")
    print("  [OK] Non-data lines handled gracefully")

def test_firmware_constants_match():
    """Verify firmware constants match Python config."""
    print("\n=== Firmware/Python Constant Consistency ===")
    firmware_path = os.path.join(FIRMWARE_DIR, "gesture_glove_esp32.ino")
    with open(firmware_path, "r", encoding="utf-8") as f:
        firmware = f.read()

    # Check all critical constants
    checks = [
        ("#define I2C_SDA           4", "I2C SDA pin"),
        ("#define I2C_SCL           5", "I2C SCL pin"),
        ("#define MPU_ADDR          0x69", "MPU address"),
        ("#define SAMPLE_RATE_HZ    100", "Sample rate"),
        ("#define WINDOW_SIZE       250", "Window size"),
        ("#define NUM_AXES          6", "Num axes"),
        ("#define NUM_FEATURES      1500", "Num features"),
        ("#define NUM_CLASSES       5", "Num classes"),
        ("#define STEP_SIZE         125", "Step size"),
        ("#define DEBOUNCE_MS       400", "Debounce"),
    ]

    for pattern, label in checks:
        assert pattern in firmware, f"Firmware missing: {pattern}"
        print(f"  [OK] {label}: matches")

    # Check gesture names order
    assert '"idle", "swipe_up", "swipe_left", "swipe_right", "swipe_down"' in firmware
    print("  [OK] Gesture names order: matches")

    # Check serial baud
    assert "Serial.begin(115200)" in firmware
    print("  [OK] Baud rate: 115200 matches")

    # Check CSV output format matches parser expectation (6 values, comma-separated)
    assert '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' in firmware
    print("  [OK] CSV output format: 6 floats comma-separated")

    # Check prediction output format
    assert '[PRED] %s %.2f %lums' in firmware
    print("  [OK] Prediction output format: [PRED] name conf latencyms")

    # Check scaler normalization formula
    assert 'normalized[i] = (normalized[i] - scaler_mean[i]) / scaler_scale[i]' in firmware
    print("  [OK] Scaler formula: (x - mean) / scale")

def test_gui_imports():
    """Test that GUI modules import without errors."""
    print("\n=== GUI Import Test ===")
    from gui_recorder import RecorderTab
    print("  [OK] gui_recorder imports OK")
    from gui_train import TrainTab
    print("  [OK] gui_train imports OK")
    from gui_debug import DebugTab
    print("  [OK] gui_debug imports OK")
    from gui import GestureGloveApp
    print("  [OK] gui (main) imports OK")


if __name__ == "__main__":
    print("=" * 60)
    print("  GESTURE GLOVE — Pipeline Integration Test")
    print("=" * 60)

    test_config()
    generate_synthetic_csv()
    test_recorder_csv_load()
    test_serial_protocol_parsing()
    test_firmware_constants_match()
    test_gui_imports()
    test_train_rf()
    test_train_mlp()

    print("\n" + "=" * 60)
    print("  [OK] ALL TESTS PASSED")
    print("=" * 60)
