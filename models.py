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
    NUM_CLASSES, WINDOW_SIZE, NUM_AXES,
    GESTURE_NAMES, MODELS_DIR, HEADERS_DIR, FIRMWARE_RF_DIR, FIRMWARE_MLP_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH
)


def _import_tensorflow():
    """Lazy-import TensorFlow to avoid slow startup when only RF is needed."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    return tf


# ═══════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def extract_features(X_raw):
    """Transform raw window data into statistical features.
    
    Input shape: (N, 1500) or (N, 250, 6)
    Output shape: (N, 36) -> 6 stats per axis
    Stats: Mean, Std, Max, Min, RMS, Zero-Crossing Rate
    """
    # Reshape to (N, 250, 6) if flat
    if len(X_raw.shape) == 2:
        X = X_raw.reshape(-1, WINDOW_SIZE, NUM_AXES)
    else:
        X = X_raw

    N = X.shape[0]
    # 6 axes * 6 stats = 36 features
    features = np.zeros((N, NUM_AXES * 6), dtype=np.float32)

    for i in range(NUM_AXES):
        axis_data = X[:, :, i]
        
        # 1. Mean
        features[:, i*6 + 0] = np.mean(axis_data, axis=1)
        # 2. Std Dev
        features[:, i*6 + 1] = np.std(axis_data, axis=1)
        # 3. Max
        features[:, i*6 + 2] = np.max(axis_data, axis=1)
        # 4. Min
        features[:, i*6 + 3] = np.min(axis_data, axis=1)
        # 5. RMS
        features[:, i*6 + 4] = np.sqrt(np.mean(axis_data**2, axis=1))
        # 6. Zero Crossing Rate (approx via mean crossing)
        mean_centered = axis_data - np.mean(axis_data, axis=1, keepdims=True)
        features[:, i*6 + 5] = np.sum(np.diff(np.sign(mean_centered), axis=1) != 0, axis=1) / (WINDOW_SIZE - 1)

    return features


# ═══════════════════════════════════════════════════════════════════
#  SCALER UTILITIES
# ═══════════════════════════════════════════════════════════════════

def fit_scaler(X_train):
    """Fit a StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ═══════════════════════════════════════════════════════════════════
#  MLP MODEL
# ═══════════════════════════════════════════════════════════════════

def build_mlp(input_dim=36):
    """Build a tiny MLP for ESP32.
    Architecture: Input(36) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(5, Softmax)
    """
    tf = _import_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="gesture_mlp_tiny")

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

# ═══════════════════════════════════════════════════════════════════
#  DATASET SPLITTING
# ═══════════════════════════════════════════════════════════════════

def split_dataset(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, stratify=True):
    """Split dataset into train/val/test with proper stratification.

    Splits in two steps to avoid issues with small class counts:
      Step 1: split 90% (train+val) / 10% (test)
      Step 2: split train+val into train / val
      Result: ~70% train, ~20% val, ~10% test

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if stratify:
        # Step 1: (train+val) / test
        test_size_adj = test_ratio
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=test_size_adj, random_state=42, stratify=y
        )
        # Step 2: train / val from train+val
        val_ratio_of_trval = val_ratio / (train_ratio + val_ratio)  # 0.2/0.9 ≈ 0.2222
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval, test_size=val_ratio_of_trval, random_state=42, stratify=y_trval
        )
    else:
        # Fallback: no stratification
        total = train_ratio + val_ratio + test_ratio
        t1 = train_ratio / total
        t2 = (train_ratio + val_ratio) / total
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
            X, y, train_size=t1, test_size=1 - t1, random_state=42
        )
        _, _, X_test, _, _, y_test = train_test_split(
            X_val, y_val, train_size=t1, test_size=1 - t1, random_state=42
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═══════════════════════════════════════════════════════════════════
#  TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def train_keras_model(model_type, X, y, epochs=DEFAULT_EPOCHS,
                      batch_size=DEFAULT_BATCH_SIZE, full_train=False,
                      progress_callback=None):
    """Train a Keras model (MLP or CNN1D).

    Args:
        model_type: "mlp" or "cnn1d"
        X: features array (N, NUM_FEATURES)
        y: labels array (N,)
        epochs: number of epochs
        batch_size: batch size
        full_train: if True, train on ALL data after evaluating on 15% holdout.
                     The deployed model uses 100% of data; eval accuracy is reported
                     from the 85%/15% split. Best for single-user personal models.
        progress_callback: optional fn(epoch, logs) called each epoch

    Returns:
        dict with keys: model, history, scaler, X_test, y_test, y_pred,
                        report, confusion, accuracy, val_accuracy,
                        deployed_on_all, train_samples, ...
    """
    tf = _import_tensorflow()
    from tensorflow import keras

    if full_train:
        # ── Step 1: Evaluate on 15% holdout for honest reporting ──
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        if model_type == "cnn1d":
            scaler = fit_scaler(X_tr)
            X_tr_in = scaler.transform(X_tr).reshape(-1, WINDOW_SIZE, NUM_AXES)
            X_ev_in = scaler.transform(X_ev).reshape(-1, WINDOW_SIZE, NUM_AXES)
        else:
            X_tr_f = extract_features(X_tr)
            X_ev_f = extract_features(X_ev)
            scaler = fit_scaler(X_tr_f)
            X_tr_in = scaler.transform(X_tr_f)
            X_ev_in = scaler.transform(X_ev_f)

        eval_model = build_cnn1d() if model_type == "cnn1d" else build_mlp(input_dim=X_tr_in.shape[1])
        eval_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        eval_model.fit(X_tr_in, y_tr, validation_data=(X_ev_in, y_ev),
                       epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = np.argmax(eval_model.predict(X_ev_in, verbose=0), axis=1)
        eval_acc = accuracy_score(y_ev, y_pred)
        report = classification_report(y_ev, y_pred, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat = confusion_matrix(y_ev, y_pred)

        # ── Step 2: Retrain on ALL data for the deployed model ──
        if model_type == "cnn1d":
            scaler_full = fit_scaler(X)
            X_all_in = scaler_full.transform(X).reshape(-1, WINDOW_SIZE, NUM_AXES)
            X_ev_full_in = scaler_full.transform(X_ev).reshape(-1, WINDOW_SIZE, NUM_AXES)
        else:
            X_all_f = extract_features(X)
            scaler_full = fit_scaler(X_all_f)
            X_all_in = scaler_full.transform(X_all_f)
            X_ev_full_in = scaler_full.transform(X_ev_f) if 'X_ev_f' in dir() else scaler_full.transform(extract_features(X_ev))

        model_full = build_cnn1d() if model_type == "cnn1d" else build_mlp(input_dim=X_all_in.shape[1])
        model_full.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        history_full = model_full.fit(
            X_all_in, y,
            epochs=epochs, batch_size=batch_size, verbose=0,
        )

        model = model_full
        scaler = scaler_full
        history = history_full
        val_acc = eval_acc
        eval_samples = len(y_ev)
        train_samples = len(y)
        deployed_on_all = True
        X_test_in = X_ev_full_in
        y_test = y_ev
        y_pred = np.argmax(model_full.predict(X_ev_full_in, verbose=0), axis=1)
        report = classification_report(y_ev, y_pred, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat = confusion_matrix(y_ev, y_pred)

    else:
        # ── Standard 70/20/10 split ──
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        if model_type == "cnn1d":
            scaler = fit_scaler(X_train)
            X_train_in = scaler.transform(X_train).reshape(-1, WINDOW_SIZE, NUM_AXES)
            X_val_in   = scaler.transform(X_val).reshape(-1, WINDOW_SIZE, NUM_AXES)
            X_test_in  = scaler.transform(X_test).reshape(-1, WINDOW_SIZE, NUM_AXES)
            model = build_cnn1d()
        else:
            X_train_f = extract_features(X_train)
            X_val_f   = extract_features(X_val)
            X_test_f  = extract_features(X_test)
            scaler = fit_scaler(X_train_f)
            X_train_in = scaler.transform(X_train_f)
            X_val_in   = scaler.transform(X_val_f)
            X_test_in  = scaler.transform(X_test_f)
            model = build_mlp(input_dim=X_train_in.shape[1])

        callbacks = []
        if progress_callback:
            class ProgressCB(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_callback(epoch, logs)
            callbacks.append(ProgressCB())

        history = model.fit(
            X_train_in, y_train,
            validation_data=(X_val_in, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=0,
        )

        y_pred = np.argmax(model.predict(X_test_in, verbose=0), axis=1)
        eval_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred)
        val_acc = float(history.history.get("val_accuracy", [0])[-1])
        train_samples = len(y_train)
        eval_samples = len(y_test)
        deployed_on_all = False

    # ── Save deployed model ──
    model_path = os.path.join(MODELS_DIR, f"{model_type}_model.keras")
    model.save(model_path)
    scaler_path = os.path.join(MODELS_DIR, f"{model_type}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    return {
        "model": model,
        "history": history.history if hasattr(history, "history") else {},
        "scaler": scaler,
        "X_test": X_test_in,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "confusion": conf_mat,
        "accuracy": eval_acc,
        "val_accuracy": val_acc,
        "train_samples": train_samples,
        "val_samples": len(y_val) if not full_train else 0,
        "test_samples": eval_samples,
        "deployed_on_all": deployed_on_all,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }


def train_random_forest(X, y, n_estimators=DEFAULT_RF_TREES,
                        max_depth=DEFAULT_RF_DEPTH, full_train=False,
                        progress_callback=None):
    """Train a Random Forest classifier.

    Args:
        X: features array (N, NUM_FEATURES)
        y: labels array (N,)
        n_estimators: number of trees
        max_depth: max tree depth
        full_train: if True, train on ALL data after evaluating on 15% holdout.
                    The deployed model uses 100% of data; eval accuracy is from 85%/15% split.
        progress_callback: optional fn(tree_idx, total_trees) — called periodically

    Returns:
        dict with keys: model, scaler, X_test, y_test, y_pred,
                        report, confusion, accuracy, deployed_on_all, ...
    """
    if full_train:
        # ── Step 1: Evaluate on 15% holdout for honest reporting ──
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        X_tr_f = extract_features(X_tr)
        X_ev_f = extract_features(X_ev)
        scaler = fit_scaler(X_tr_f)
        X_tr_s = scaler.transform(X_tr_f)
        X_ev_s = scaler.transform(X_ev_f)

        rf_eval = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1
        )
        if progress_callback:
            for i in range(1, n_estimators + 1):
                rf_eval.n_estimators = i
                rf_eval.fit(X_tr_s, y_tr)
                if i % 10 == 0 or i == n_estimators:
                    progress_callback(i, n_estimators)
        else:
            rf_eval.fit(X_tr_s, y_tr)
        y_pred = rf_eval.predict(X_ev_s)
        eval_acc = accuracy_score(y_ev, y_pred)
        report = classification_report(y_ev, y_pred, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat = confusion_matrix(y_ev, y_pred)

        # ── Step 2: Retrain on ALL data for the deployed model ──
        X_all_f = extract_features(X)
        scaler_full = fit_scaler(X_all_f)
        X_all_s = scaler_full.transform(X_all_f)
        X_ev_full_s = scaler_full.transform(X_ev_f)

        rf_full = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, n_jobs=-1
        )
        rf_full.fit(X_all_s, y)
        # Predict on holdout using full model's scaler
        y_pred_full = rf_full.predict(X_ev_full_s)
        report_full = classification_report(y_ev, y_pred_full, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat_full = confusion_matrix(y_ev, y_pred_full)

        rf = rf_full
        scaler = scaler_full
        y_pred = y_pred_full
        report = report_full
        conf_mat = conf_mat_full
        eval_acc = accuracy_score(y_ev, y_pred_full)
        val_acc = eval_acc
        train_samples = len(y)
        eval_samples = len(y_ev)
        deployed_on_all = True
        X_test_s = X_ev_full_s
        y_test = y_ev

    else:
        # ── Standard 70/20/10 split ──
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

        X_train_f = extract_features(X_train)
        X_val_f   = extract_features(X_val)
        X_test_f  = extract_features(X_test)
        scaler = fit_scaler(X_train_f)
        X_train_s = scaler.transform(X_train_f)
        X_val_s   = scaler.transform(X_val_f)
        X_test_s  = scaler.transform(X_test_f)

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

        y_val_pred = rf.predict(X_val_s)
        val_acc = accuracy_score(y_val, y_val_pred)
        y_pred = rf.predict(X_test_s)
        eval_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=GESTURE_NAMES, output_dict=True)
        conf_mat = confusion_matrix(y_test, y_pred)
        train_samples = len(y_train)
        eval_samples = len(y_test)
        deployed_on_all = False

    # Save deployed model
    model_path = os.path.join(MODELS_DIR, f"rf_model.joblib")
    joblib.dump(rf, model_path)
    scaler_path = os.path.join(MODELS_DIR, "rf_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    return {
        "model": rf,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
        "confusion": conf_mat,
        "accuracy": eval_acc,
        "val_accuracy": val_acc,
        "train_samples": train_samples,
        "val_samples": len(y_val) if not full_train else 0,
        "test_samples": eval_samples,
        "deployed_on_all": deployed_on_all,
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
    # Scaler dimension = model input dimension
    # MLP:  36 statistical features (6 stats × 6 axes)
    # CNN1D: 1500 raw values (250 samples × 6 axes)
    n_features = len(scaler.mean_)
    header_lines = [
        f"// Auto-generated by Gesture Glove ML Pipeline",
        f"// Model: {model_type.upper()} — int8 quantized TFLite",
        f"// Scaler features: {n_features} ({'statistical' if model_type == 'mlp' else 'raw'})",
        f"// Window: {WINDOW_SIZE} timesteps × {NUM_AXES} axes",
        f"// Classes:  {NUM_CLASSES} ({', '.join(GESTURE_NAMES)})",
        f"//",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <cstdint>",
        f"#include <cstddef>",
        f"",
        f"// ─── Model Configuration ───────────────────────────────",
        f"const int MODEL_NUM_FEATURES = {n_features};",
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

    # Also save to firmware directories for easy compilation
    firmware_dirs = [FIRMWARE_MLP_DIR]
    # Also copy to the main ESP32 firmware directory
    esp32_dir = os.path.join(os.path.dirname(FIRMWARE_MLP_DIR), "gesture_glove_esp32")
    firmware_dirs.append(esp32_dir)
    for fw_dir in firmware_dirs:
        try:
            fw_path = os.path.join(fw_dir, f"{model_type}_model_data.h")
            with open(fw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines))
        except Exception:
            pass

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

    # Generate C code from model — m2cgen defaults to double; post-process to float
    c_code = m2c.export_to_c(rf_model)
    # Replace all double with float for ESP32 efficiency (float is native on Xtensa FPU)
    c_code = c_code.replace("double ", "float ")
    c_code = c_code.replace("double*", "float*")

    guard = "RF_MODEL_DATA_H"
    # RF uses 36 statistical features (6 stats × 6 axes)
    n_stat_features = NUM_AXES * 6
    header_lines = [
        f"// Auto-generated by Gesture Glove ML Pipeline",
        f"// Model: Random Forest — {rf_model.n_estimators} trees, max_depth={rf_model.max_depth}",
        f"// Features: {n_stat_features} statistical (mean/std/max/min/rms/mcr × 6 axes)",
        f"// Classes:  {NUM_CLASSES} ({', '.join(GESTURE_NAMES)})",
        f"// Export method: m2cgen (pure C, no TFLite dependency)",
        f"//",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <math.h>",
        f"#include <stdint.h>",
        f"#ifdef __cplusplus",
        f"extern \"C\" {{",
        f"#endif",
        f"",
        f"// ─── Model Configuration ───────────────────────────────",
        f"const int MODEL_NUM_FEATURES = {n_stat_features};  // statistical features",
        f"const int MODEL_WINDOW_SIZE  = {WINDOW_SIZE};",
        f"const int MODEL_NUM_AXES     = {NUM_AXES};",
        f"const int MODEL_NUM_CLASSES  = {NUM_CLASSES};",
        f"",
        f"// ─── Gesture Names ─────────────────────────────────────",
        f"const char* MODEL_GESTURE_NAMES[{NUM_CLASSES}] = {{",
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

    # m2cgen model function — double replaced with float above
    header_lines.append("// ─── Random Forest Model (m2cgen, float-optimized) ────")
    header_lines.append(f"// Call: score(input, output) where input[{len(scaler.mean_)}], output[{NUM_CLASSES}]")
    header_lines.append("// output[i] = vote count for class i")
    header_lines.append(c_code)
    header_lines.append("")

    # Helper: predict function that returns class index and confidence
    header_lines.append("// ─── Prediction Helper ────────────────────────────────")
    header_lines.append("int rf_predict(float* input, float* confidence) {")
    header_lines.append(f"    float output[{NUM_CLASSES}];")
    header_lines.append(f"    score(input, output);")
    header_lines.append(f"")
    header_lines.append(f"    // Convert votes to probabilities")
    header_lines.append(f"    float total = 0.0f;")
    header_lines.append(f"    for (int i = 0; i < {NUM_CLASSES}; i++) total += output[i];")
    header_lines.append(f"")
    header_lines.append(f"    int best = 0;")
    header_lines.append(f"    float best_prob = 0.0f;")
    header_lines.append(f"    for (int i = 0; i < {NUM_CLASSES}; i++) {{")
    header_lines.append(f"        float prob = (total > 0.0f) ? output[i] / total : 0.0f;")
    header_lines.append(f"        if (prob > best_prob) {{")
    header_lines.append(f"            best_prob = prob;")
    header_lines.append(f"            best = i;")
    header_lines.append(f"        }}")
    header_lines.append(f"    }}")
    header_lines.append(f"    *confidence = best_prob;")
    header_lines.append(f"    return best;")
    header_lines.append("}")
    header_lines.append("")
    header_lines.append("#ifdef __cplusplus")
    header_lines.append("}")
    header_lines.append("#endif")
    header_lines.append("")

    header_lines.append(f"#endif // {guard}")
    header_lines.append("")

    header_path = os.path.join(HEADERS_DIR, "rf_model_data.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))

    # Also save to firmware directories for easy compilation
    firmware_dirs = [FIRMWARE_RF_DIR]
    esp32_dir = os.path.join(os.path.dirname(FIRMWARE_RF_DIR), "gesture_glove_esp32")
    firmware_dirs.append(esp32_dir)
    for fw_dir in firmware_dirs:
        try:
            fw_path = os.path.join(fw_dir, "rf_model_data.h")
            with open(fw_path, "w", encoding="utf-8") as f:
                f.write("\n".join(header_lines))
        except Exception:
            pass

    return header_path


# ═══════════════════════════════════════════════════════════════════
#  UNIFIED TRAINING API
# ═══════════════════════════════════════════════════════════════════

def train_model(model_type, X, y, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                n_estimators=DEFAULT_RF_TREES, max_depth=DEFAULT_RF_DEPTH,
                quantize_int8=True, full_train=False, progress_callback=None):
    """Unified training entry point.

    Args:
        model_type: "mlp", "cnn1d", or "rf"
        X, y: dataset arrays
        epochs, batch_size: for Keras models
        n_estimators, max_depth: for Random Forest
        quantize_int8: for TFLite export
        full_train: if True, train on ALL data after evaluating on 15% holdout.
                    Best for single-user personal models — deployed model uses 100% of data.
        progress_callback: for UI updates

    Returns:
        dict with training results, model info, and export paths
    """
    if model_type in ("mlp", "cnn1d"):
        result = train_keras_model(
            model_type, X, y, epochs, batch_size,
            full_train=full_train, progress_callback=progress_callback,
        )

        # Export TFLite from the deployed (full) model
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

        # Export C header from the deployed (full) model
        try:
            header_path = export_tflite_c_header(model_type, result["scaler"])
            result["header_path"] = header_path
        except Exception as e:
            result["header_error"] = str(e)

    elif model_type == "rf":
        result = train_random_forest(
            X, y, n_estimators, max_depth,
            full_train=full_train, progress_callback=progress_callback,
        )

        # Export C header via m2cgen from the deployed (full) model
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
