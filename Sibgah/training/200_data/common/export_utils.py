"""
Shared export helpers for CNN model export.
"""

import os
import numpy as np


def export_scaler_header(scaler, path: str, window_size: int, n_features: int) -> None:
    """
    Write a C header with the StandardScaler parameters.
    The scaler was fitted on flattened (N*T, F) data so mean_/scale_
    have shape (n_features,) = (12,).

    On the ESP32, apply per timestep per channel before inference:
        scaled_val = (raw_val - SCALER_MEAN[channel]) / SCALER_STD[channel]
    """
    mean  = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    feature_names = [
        "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
        "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
    ]
    n = len(mean)

    lines = [
        "// scaler_params.h — StandardScaler parameters",
        "// Apply per channel before CNN inference:",
        "//   scaled = (raw - SCALER_MEAN[ch]) / SCALER_STD[ch]",
        "#pragma once",
        "",
        f"const int WINDOW_SIZE  = {window_size};",
        f"const int N_IMU_CHANNELS = {n_features};",
        "",
        f"const float SCALER_MEAN[{n}] = {{",
    ]
    for i, v in enumerate(mean):
        comma = "," if i < n - 1 else ""
        lines.append(f"    {v:.8f}f{comma}  // {feature_names[i]}")
    lines += ["};", ""]

    lines.append(f"const float SCALER_STD[{n}] = {{")
    for i, v in enumerate(scale):
        comma = "," if i < n - 1 else ""
        lines.append(f"    {v:.8f}f{comma}  // {feature_names[i]}")
    lines += ["};", ""]

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def export_labels_header(class_names: list[str], path: str) -> None:
    entries = ", ".join(f'"{c}"' for c in class_names)
    lines = [
        "// labels.h — gesture class names",
        "#pragma once",
        "",
        f"const int N_CLASSES = {len(class_names)};",
        f'const char* GESTURE_CLASSES[{len(class_names)}] = {{{entries}}};',
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def write_convert_script(path: str, onnx_filename: str, model_name: str) -> None:
    """Write a Colab/Python-3.12 script to convert ONNX → TFLite."""
    # Use explicit concatenation to avoid f-string collisions with the
    # inner Python code that also uses curly braces.
    lines = [
        '"""',
        "convert_to_tflite.py — ONNX to TFLite conversion for " + model_name + ".",
        "",
        "Requirements: Python <= 3.12, tensorflow >= 2.13, onnx2tf",
        "    pip install tensorflow onnx2tf",
        "",
        "Google Colab quick-start:",
        "    1. Upload " + onnx_filename + " to Colab.",
        "    2. Run:  !pip install onnx2tf tensorflow",
        "    3. Run:  !python convert_to_tflite.py",
        "    4. Download model_data.h",
        "    5. Copy model_data.h + scaler_params.h into your ESP32 project.",
        '"""',
        "",
        "import subprocess, sys, os",
        "",
        'ONNX_FILE  = "' + onnx_filename + '"',
        'OUTPUT_DIR = "tflite_out"',
        "",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "# ONNX -> TF SavedModel -> TFLite",
        "subprocess.run(",
        "    [sys.executable, '-m', 'onnx2tf',",
        "     '-i', ONNX_FILE, '-o', OUTPUT_DIR, '-oiqt'],",
        "    check=True,",
        ")",
        "",
        'tflite_path = os.path.join(OUTPUT_DIR, "' + model_name + '_float32.tflite")',
        'print("TFLite model saved to:", tflite_path)',
        "",
        "# Generate C array for ESP32 firmware",
        'with open(tflite_path, "rb") as f:',
        "    raw = f.read()",
        "",
        'hex_vals = ", ".join("0x{:02x}".format(b) for b in raw)',
        'c_code = (',
        '    "// model_data.h -- ' + model_name + ' TFLite model\\n"',
        '    "#pragma once\\n"',
        '    "const unsigned int MODEL_DATA_LEN = " + str(len(raw)) + ";\\n"',
        '    "const unsigned char MODEL_DATA[] = {\\n  " + hex_vals + "\\n};\\n"',
        ")",
        'c_path = "model_data.h"',
        'with open(c_path, "w") as f:',
        "    f.write(c_code)",
        'print("C array saved to:", c_path)',
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
