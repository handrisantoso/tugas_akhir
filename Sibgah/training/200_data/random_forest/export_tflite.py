"""
Random Forest export for ESP32.

Random Forest does NOT need TFLite — micromlgen generates a self-contained
C++ class that runs directly in Arduino / ESP32 firmware.

Outputs in output_tflite/:
  model.h          — C++ RandomForest classifier (include in your ESP32 sketch)
  scaler_params.h  — StandardScaler mean/std as C arrays (use before inference)
  labels.h         — gesture class name array

Usage on ESP32 (pseudocode):
  #include "model.h"
  #include "scaler_params.h"
  #include "labels.h"

  Eloquent::ML::Port::RandomForest clf;

  // 1. collect 25 samples x 12 channels, extract 72 statistical features
  float features[72] = { ... };

  // 2. scale each feature
  for (int i = 0; i < 72; i++)
      features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];

  // 3. predict
  int class_idx = clf.predict(features);
  Serial.println(GESTURE_CLASSES[class_idx]);
"""

import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.preprocessing import (
    load_file_list, split_file_list, get_class_names,
    extract_windows, fit_scaler, apply_scaler, compute_class_weights,
    FEATURES,
)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "cleaned_data")
MODEL_PKL   = os.path.join(os.path.dirname(__file__), "output", "model.pkl")
SCALER_PKL  = os.path.join(os.path.dirname(__file__), "output", "scaler.pkl")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output_tflite")
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATS = ["mean", "std", "min", "max", "median", "rms"]


def make_feature_names() -> list[str]:
    return [f"{stat}_{feat}" for feat in FEATURES for stat in STATS]


def export_scaler_header(scaler, feature_names: list[str], path: str) -> None:
    """
    The scaler operates on the 72 statistical features (not the raw window).
    mean_ and scale_ each have 72 values in feature order.
    """
    mean  = scaler.mean_
    scale = scaler.scale_
    n = len(mean)

    lines = [
        "// scaler_params.h — StandardScaler for Random Forest features",
        "// Apply: scaled = (raw_feature - SCALER_MEAN[i]) / SCALER_STD[i]",
        f"// {n} features: " + ", ".join(feature_names[:4]) + " ...",
        "#pragma once",
        "",
        f"const int RF_N_FEATURES = {n};",
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


def main() -> None:
    print("=" * 50)
    print("  RF Export for ESP32")
    print("=" * 50)

    if not os.path.exists(MODEL_PKL) or not os.path.exists(SCALER_PKL):
        print("ERROR: Run random_forest/train.py first to generate model.pkl and scaler.pkl")
        sys.exit(1)

    clf    = joblib.load(MODEL_PKL)
    scaler = joblib.load(SCALER_PKL)

    file_list   = load_file_list(DATA_DIR)
    class_names = get_class_names(file_list)
    print(f"Classes: {class_names}")

    feature_names = make_feature_names()

    # micromlgen export
    try:
        from micromlgen import port
        c_code = port(clf, classmap={i: name for i, name in enumerate(class_names)})
        model_h_path = os.path.join(OUTPUT_DIR, "model.h")
        with open(model_h_path, "w") as f:
            f.write(c_code)
        print(f"  Saved: {model_h_path}")
    except Exception as e:
        print(f"  WARNING: micromlgen export failed: {e}")
        print("  Install micromlgen: pip install micromlgen")

    export_scaler_header(
        scaler, feature_names,
        os.path.join(OUTPUT_DIR, "scaler_params.h"),
    )
    export_labels_header(
        class_names,
        os.path.join(OUTPUT_DIR, "labels.h"),
    )

    print(f"\nDone. Files in: {OUTPUT_DIR}")
    print("\n--- ESP32 usage ---")
    print("1. Copy output_tflite/*.h into your Arduino/PlatformIO project.")
    print("2. #include \"model.h\", \"scaler_params.h\", \"labels.h\"")
    print("3. Extract 72 statistical features per window (same order as SCALER_MEAN).")
    print("4. Scale each feature: scaled[i] = (raw[i] - SCALER_MEAN[i]) / SCALER_STD[i]")
    print("5. Call clf.predict(scaled) to get the class index.")
    print("6. Look up GESTURE_CLASSES[index] for the label string.")


if __name__ == "__main__":
    main()
