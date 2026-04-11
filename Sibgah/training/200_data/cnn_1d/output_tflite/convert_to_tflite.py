"""
convert_to_tflite.py — ONNX to TFLite conversion for cnn_1d.

Requirements: Python <= 3.12, tensorflow >= 2.13, onnx2tf
    pip install tensorflow onnx2tf

Google Colab quick-start:
    1. Upload cnn_1d.onnx to Colab.
    2. Run:  !pip install onnx2tf tensorflow
    3. Run:  !python convert_to_tflite.py
    4. Download model_data.h
    5. Copy model_data.h + scaler_params.h into your ESP32 project.
"""

import subprocess, sys, os

ONNX_FILE  = "cnn_1d.onnx"
OUTPUT_DIR = "tflite_out"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ONNX -> TF SavedModel -> TFLite
subprocess.run(
    [sys.executable, '-m', 'onnx2tf',
     '-i', ONNX_FILE, '-o', OUTPUT_DIR, '-oiqt'],
    check=True,
)

tflite_path = os.path.join(OUTPUT_DIR, "cnn_1d_float32.tflite")
print("TFLite model saved to:", tflite_path)

# Generate C array for ESP32 firmware
with open(tflite_path, "rb") as f:
    raw = f.read()

hex_vals = ", ".join("0x{:02x}".format(b) for b in raw)
c_code = (
    "// model_data.h -- cnn_1d TFLite model\n"
    "#pragma once\n"
    "const unsigned int MODEL_DATA_LEN = " + str(len(raw)) + ";\n"
    "const unsigned char MODEL_DATA[] = {\n  " + hex_vals + "\n};\n"
)
c_path = "model_data.h"
with open(c_path, "w") as f:
    f.write(c_code)
print("C array saved to:", c_path)
