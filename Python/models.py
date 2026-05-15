"""
Gesture Glove — ML Models
MLP, 1D-CNN, Random Forest — train, evaluate, TFLite export, C header generation.
"""

import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

from config import (
    NUM_FEATURES, NUM_CLASSES, WINDOW_SIZE, NUM_AXES,
    GESTURE_NAMES, MODELS_DIR, HEADERS_DIR, FIRMWARE_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH,
    VALIDATION_SPLIT, TFLITE_ARENA_KB
)


def _import_tensorflow():
    """Lazy-import TensorFlow to avoid slow startup when only RF is needed."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    return tf


# ═══════════════════════════════════════════════════════════════════
#  SCALER UTILITIES
# ═══════════════════════════════════════════════════════════════════

def fit_scaler(X_train):
    """Fit a StandardScaler on training data.

    Args:
        X_train: array of shape (N, NUM_FEATURES)

    Returns:
        Fitted StandardScaler instance
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def save_scaler(scaler, model_name):
    """Save scaler to disk."""
    path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
    joblib.dump(scaler, path)
    return path


def load_scaler(model_name):
    """Load scaler from disk."""
    path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════
#  MLP MODEL
# ═══════════════════════════════════════════════════════════════════

def build_mlp():
    """Build the MLP Keras model.

    Architecture:
        Input(1500) → Dense(512) → BN → ReLU → Drop(0.3)
                    → Dense(256) → BN → ReLU → Drop(0.2)
                    → Dense(128) → BN → ReLU
                    → Dense(5, softmax)
    """
    tf = _import_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(NUM_FEATURES,)),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.3),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="gesture_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════
#  1D-CNN MODEL
# ═══════════════════════════════════════════════════════════════════

def build_cnn1d():
    """Build the 1D-CNN Keras model.

    Architecture:
        Input(250, 6) → Conv1D(64, k=7) → BN → MaxPool(2)
                      → Conv1D(128, k=5) → BN → MaxPool(2)
                      → Conv1D(256, k=3) → GlobalAvgPool
                      → Dense(128) → Drop(0.25)
                      → Dense(5, softmax)
    """
    tf = _import_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(WINDOW_SIZE, NUM_AXES)),
        layers.Conv1D(64, kernel_size=7, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(256, kernel_size=3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="gesture_cnn1d")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def train_keras_model(model_type, X, y, epochs=DEFAULT_EPOCHS,
                      batch_size=DEFAULT_BATCH_SIZE, progress_callback=None):
    """Train a Keras model (MLP or CNN1D).

    Args:
        model_type: "mlp" or "cnn1d"
        X: features array (N, NUM_FEATURES)
        y: labels array (N,)
        epochs: number of epochs
        batch_size: batch size
        progress_callback: optional fn(epoch, logs) called each epoch

    Returns:
        dict with keys: model, history, scaler, X_test, y_test, report, confusion
    """
    tf = _import_tensorflow()
    from tensorflow import keras

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    # Scale
    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Reshape for CNN1D
    if model_type == "cnn1d":
        X_train_s = X_train_s.reshape(-1, WINDOW_SIZE, NUM_AXES)
        X_test_s = X_test_s.reshape(-1, WINDOW_SIZE, NUM_AXES)
        model = build_cnn1d()
    else:
        model = build_mlp()

    # Callbacks
    callbacks = []
    if progress_callback:
        class ProgressCB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_callback(epoch, logs)
        callbacks.append(ProgressCB())

    # Early stopping
    callbacks.append(keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    ))

    # Learning rate reduction
    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
    ))

    # Train
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_test_s, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # Evaluate
    y_pred = np.argmax(model.predict(X_test_s, verbose=0), axis=1)
    report = classification_report(y_test, y_pred, target_names=GESTURE_NAMES, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Save model and scaler
    model_path = os.path.join(MODELS_DIR, f"{model_type}_model.keras")
    model.save(model_path)
    scaler_path = save_scaler(scaler, model_type)

    return {
        "model": model,
        "history": history.history,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "confusion": conf_mat,
        "accuracy": acc,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


def train_random_forest(X, y, n_estimators=DEFAULT_RF_TREES,
                        max_depth=DEFAULT_RF_DEPTH, progress_callback=None):
    """Train a Random Forest classifier.

    Args:
        X: features array (N, NUM_FEATURES)
        y: labels array (N,)
        n_estimators: number of trees
        max_depth: max tree depth
        progress_callback: optional fn(tree_idx, total_trees) — called periodically

    Returns:
        dict with keys: model, scaler, X_test, y_test, report, confusion
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=42, stratify=y
    )

    # Scale
    scaler = fit_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train with warm_start for progress
    if progress_callback:
        rf = RandomForestClassifier(
            n_estimators=1, max_depth=max_depth, random_state=42,
            n_jobs=-1, warm_start=True
        )
        for i in range(1, n_estimators + 1):
            rf.n_estimators = i
            rf.fit(X_train_s, y_train)
            if i % 10 == 0 or i == n_estimators:
                progress_callback(i, n_estimators)
    else:
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train_s, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_s)
    report = classification_report(y_test, y_pred, target_names=GESTURE_NAMES, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # Save
    model_path = os.path.join(MODELS_DIR, f"rf_model.joblib")
    joblib.dump(rf, model_path)
    scaler_path = save_scaler(scaler, "rf")

    return {
        "model": rf,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "confusion": conf_mat,
        "accuracy": acc,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


# ═══════════════════════════════════════════════════════════════════
#  TFLITE EXPORT
# ═══════════════════════════════════════════════════════════════════

def export_tflite(model, model_type, X_calibration, scaler, quantize_int8=True):
    """Export a Keras model to TFLite format.

    Args:
        model: trained Keras model
        model_type: "mlp" or "cnn1d"
        X_calibration: calibration data for int8 quantization (already scaled)
        scaler: fitted StandardScaler
        quantize_int8: whether to apply int8 quantization

    Returns:
        tuple: (tflite_path, model_size_bytes, quantized_size_bytes)
    """
    tf = _import_tensorflow()

    # Convert directly from Keras model (avoids temp file format issues)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Float32 model first (for size comparison)
    float_model = converter.convert()
    float_size = len(float_model)

    if quantize_int8:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Reshape calibration data for CNN1D
        if model_type == "cnn1d":
            cal_data = X_calibration.reshape(-1, WINDOW_SIZE, NUM_AXES)
        else:
            cal_data = X_calibration

        def representative_dataset():
            for i in range(min(200, len(cal_data))):
                yield [cal_data[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
    else:
        tflite_model = float_model

    quantized_size = len(tflite_model)

    # Save .tflite file
    tflite_path = os.path.join(MODELS_DIR, f"{model_type}_model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    return tflite_path, float_size, quantized_size


# ═══════════════════════════════════════════════════════════════════
#  C HEADER GENERATION
# ═══════════════════════════════════════════════════════════════════

def _format_c_array(data, var_name, line_width=12):
    """Format a byte array as a C constant array string."""
    lines = []
    lines.append(f"alignas(16) const unsigned char {var_name}[] = {{")

    for i in range(0, len(data), line_width):
        chunk = data[i:i+line_width]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        comma = "," if i + line_width < len(data) else ""
        lines.append(f"    {hex_vals}{comma}")

    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};")
    return "\n".join(lines)


def _format_c_float_array(arr, var_name, per_line=8):
    """Format a float array as a C constant array string."""
    lines = []
    lines.append(f"const float {var_name}[{len(arr)}] = {{")

    for i in range(0, len(arr), per_line):
        chunk = arr[i:i+per_line]
        vals = ", ".join(f"{v:.8f}f" for v in chunk)
        comma = "," if i + per_line < len(arr) else ""
        lines.append(f"    {vals}{comma}")

    lines.append("};")
    return "\n".join(lines)


def export_tflite_c_header(model_type, scaler):
    """Generate C header file from TFLite model + scaler.

    Args:
        model_type: "mlp" or "cnn1d"
        scaler: fitted StandardScaler

    Returns:
        str: path to generated header file
    """
    tflite_path = os.path.join(MODELS_DIR, f"{model_type}_model.tflite")
    if not os.path.isfile(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    with open(tflite_path, "rb") as f:
        tflite_data = f.read()

    guard = f"{model_type.upper()}_MODEL_DATA_H"
    header_lines = [
        f"// Auto-generated by Gesture Glove ML Pipeline",
        f"// Model: {model_type.upper()} — int8 quantized TFLite",
        f"// Features: {NUM_FEATURES} ({WINDOW_SIZE} timesteps × {NUM_AXES} axes)",
        f"// Classes:  {NUM_CLASSES} ({', '.join(GESTURE_NAMES)})",
        f"//",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <cstdint>",
        f"#include <cstddef>",
        f"",
        f"// ─── Model Configuration ───────────────────────────────",
        f"const int MODEL_NUM_FEATURES = {NUM_FEATURES};",
        f"const int MODEL_WINDOW_SIZE  = {WINDOW_SIZE};",
        f"const int MODEL_NUM_AXES     = {NUM_AXES};",
        f"const int MODEL_NUM_CLASSES  = {NUM_CLASSES};",
        f"",
        f"// ─── Gesture Names ─────────────────────────────────────",
        f'const char* MODEL_GESTURE_NAMES[{NUM_CLASSES}] = {{',
    ]

    for i, name in enumerate(GESTURE_NAMES):
        comma = "," if i < NUM_CLASSES - 1 else ""
        header_lines.append(f'    "{name}"{comma}')
    header_lines.append("};")
    header_lines.append("")

    # Scaler parameters
    header_lines.append("// ─── StandardScaler Parameters ────────────────────────")
    header_lines.append(_format_c_float_array(scaler.mean_, "scaler_mean"))
    header_lines.append("")
    header_lines.append(_format_c_float_array(scaler.scale_, "scaler_scale"))
    header_lines.append("")

    # TFLite model data
    header_lines.append("// ─── TFLite Model Data ────────────────────────────────")
    header_lines.append(_format_c_array(tflite_data, "model_data"))
    header_lines.append("")

    header_lines.append(f"#endif // {guard}")
    header_lines.append("")

    header_path = os.path.join(HEADERS_DIR, f"{model_type}_model_data.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))

    # Also save to firmware directory for easy compilation
    firmware_header = os.path.join(FIRMWARE_DIR, f"{model_type}_model_data.h")
    try:
        with open(firmware_header, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
    except Exception:
        pass # Firmware dir might not exist in some test environments

    return header_path


def export_rf_c_header(rf_model, scaler):
    """Generate C header file from Random Forest model via m2cgen.

    Args:
        rf_model: trained RandomForestClassifier
        scaler: fitted StandardScaler

    Returns:
        str: path to generated header file
    """
    import m2cgen as m2c

    # Generate C code from model
    c_code = m2c.export_to_c(rf_model)

    guard = "RF_MODEL_DATA_H"
    header_lines = [
        f"// Auto-generated by Gesture Glove ML Pipeline",
        f"// Model: Random Forest — {rf_model.n_estimators} trees, max_depth={rf_model.max_depth}",
        f"// Features: {NUM_FEATURES} ({WINDOW_SIZE} timesteps × {NUM_AXES} axes)",
        f"// Classes:  {NUM_CLASSES} ({', '.join(GESTURE_NAMES)})",
        f"// Export method: m2cgen (pure C, no TFLite dependency)",
        f"//",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <math.h>",
        f"",
        f"// ─── Model Configuration ───────────────────────────────",
        f"const int MODEL_NUM_FEATURES = {NUM_FEATURES};",
        f"const int MODEL_WINDOW_SIZE  = {WINDOW_SIZE};",
        f"const int MODEL_NUM_AXES     = {NUM_AXES};",
        f"const int MODEL_NUM_CLASSES  = {NUM_CLASSES};",
        f"",
        f"// ─── Gesture Names ─────────────────────────────────────",
        f'const char* MODEL_GESTURE_NAMES[{NUM_CLASSES}] = {{',
    ]

    for i, name in enumerate(GESTURE_NAMES):
        comma = "," if i < NUM_CLASSES - 1 else ""
        header_lines.append(f'    "{name}"{comma}')
    header_lines.append("};")
    header_lines.append("")

    # Scaler parameters
    header_lines.append("// ─── StandardScaler Parameters ────────────────────────")
    header_lines.append(_format_c_float_array(scaler.mean_, "scaler_mean"))
    header_lines.append("")
    header_lines.append(_format_c_float_array(scaler.scale_, "scaler_scale"))
    header_lines.append("")

    # m2cgen model function
    header_lines.append("// ─── Random Forest Model (m2cgen) ────────────────────")
    header_lines.append("// Call: score(input, output) where input[1500], output[5]")
    header_lines.append("// output[i] = vote count for class i")
    header_lines.append(c_code)
    header_lines.append("")

    # Helper: predict function that returns class index and confidence
    header_lines.append("// ─── Prediction Helper ────────────────────────────────")
    header_lines.append("int rf_predict(float* input, float* confidence) {")
    header_lines.append(f"    double output[{NUM_CLASSES}];")
    header_lines.append(f"    score(input, output);")
    header_lines.append(f"    ")
    header_lines.append(f"    // Convert votes to probabilities")
    header_lines.append(f"    double total = 0.0;")
    header_lines.append(f"    for (int i = 0; i < {NUM_CLASSES}; i++) total += output[i];")
    header_lines.append(f"    ")
    header_lines.append(f"    int best = 0;")
    header_lines.append(f"    double best_prob = 0.0;")
    header_lines.append(f"    for (int i = 0; i < {NUM_CLASSES}; i++) {{")
    header_lines.append(f"        double prob = (total > 0) ? output[i] / total : 0.0;")
    header_lines.append(f"        if (prob > best_prob) {{")
    header_lines.append(f"            best_prob = prob;")
    header_lines.append(f"            best = i;")
    header_lines.append(f"        }}")
    header_lines.append(f"    }}")
    header_lines.append(f"    *confidence = (float)best_prob;")
    header_lines.append(f"    return best;")
    header_lines.append("}")
    header_lines.append("")

    header_lines.append(f"#endif // {guard}")
    header_lines.append("")

    header_path = os.path.join(HEADERS_DIR, "rf_model_data.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))

    # Also save to firmware directory for easy compilation
    firmware_header = os.path.join(FIRMWARE_DIR, "rf_model_data.h")
    try:
        with open(firmware_header, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines))
    except Exception:
        pass

    return header_path


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED TRAINING API
# ═══════════════════════════════════════════════════════════════════

def train_model(model_type, X, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                n_estimators=DEFAULT_RF_TREES, max_depth=DEFAULT_RF_DEPTH,
                quantize_int8=True, progress_callback=None):
    """Unified training entry point.

    Args:
        model_type: "mlp", "cnn1d", or "rf"
        X, y: dataset arrays
        epochs, batch_size: for Keras models
        n_estimators, max_depth: for Random Forest
        quantize_int8: for TFLite export
        progress_callback: for UI updates

    Returns:
        dict with training results, model info, and export paths
    """
    if model_type in ("mlp", "cnn1d"):
        result = train_keras_model(model_type, X, y, epochs, batch_size, progress_callback)

        # Export TFLite
        try:
            tflite_path, float_size, quant_size = export_tflite(
                result["model"], model_type, result["X_test"],
                result["scaler"], quantize_int8
            )
            result["tflite_path"] = tflite_path
            result["float_model_size"] = float_size
            result["quantized_model_size"] = quant_size
        except Exception as e:
            result["tflite_error"] = str(e)

        # Export C header
        try:
            header_path = export_tflite_c_header(model_type, result["scaler"])
            result["header_path"] = header_path
        except Exception as e:
            result["header_error"] = str(e)

    elif model_type == "rf":
        result = train_random_forest(X, y, n_estimators, max_depth, progress_callback)

        # Export C header via m2cgen
        try:
            header_path = export_rf_c_header(result["model"], result["scaler"])
            result["header_path"] = header_path
        except Exception as e:
            result["header_error"] = str(e)

        # No TFLite for RF
        result["float_model_size"] = os.path.getsize(result["model_path"])
        result["quantized_model_size"] = result["float_model_size"]

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return result


def get_model_summary(model_type):
    """Get a text summary of a trained model."""
    if model_type in ("mlp", "cnn1d"):
        tf = _import_tensorflow()
        model_path = os.path.join(MODELS_DIR, f"{model_type}_model.keras")
        if os.path.isfile(model_path):
            model = tf.keras.models.load_model(model_path)
            lines = []
            model.summary(print_fn=lambda x: lines.append(x))
            return "\n".join(lines)
    elif model_type == "rf":
        model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
        if os.path.isfile(model_path):
            rf = joblib.load(model_path)
            return (f"Random Forest\n"
                    f"  Trees: {rf.n_estimators}\n"
                    f"  Max Depth: {rf.max_depth}\n"
                    f"  Features: {rf.n_features_in_}\n"
                    f"  Classes: {rf.n_classes_}")
    return "No trained model found."
