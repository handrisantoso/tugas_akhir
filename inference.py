"""
Gesture Glove — PC-side Inference Engine
Loads trained models and runs inference on raw IMU data streamed from ESP32.
This gives higher accuracy than on-device inference (float32, no quantization).
"""

import os
import numpy as np
import joblib
import collections
from config import (
    WINDOW_SIZE, NUM_AXES, GESTURE_NAMES,
    MODELS_DIR, STEP_SIZE
)
from models import extract_features


class PCInferenceEngine:
    """Runs trained models on PC for higher-accuracy inference.

    Flow:
        ESP32 streams raw IMU CSV → Python collects window → model predicts

    Compared to embedded inference:
        - Float32 precision (no int8 quantization loss)
        - Full Python ML stack (sklearn, TensorFlow)
        - Can use larger/heavier models if desired
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_type = None
        self._tf = None

        # Rolling buffer for collecting IMU samples
        self.imu_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self._samples_since_infer = 0

    def load_model(self, model_type):
        """Load a trained model and its scaler from disk.

        Args:
            model_type: "mlp", "cnn1d", or "rf"

        Returns:
            True if loaded successfully, raises on failure
        """
        self.model_type = model_type

        if model_type in ("mlp", "cnn1d"):
            self._tf = self._import_tf()
            model_path = os.path.join(MODELS_DIR, f"{model_type}_model.keras")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = self._tf.keras.models.load_model(model_path)
        elif model_type == "rf":
            model_path = os.path.join(MODELS_DIR, "rf_model.joblib")
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = joblib.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        scaler_path = os.path.join(MODELS_DIR, f"{model_type}_scaler.joblib")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Reset buffer
        self.imu_buffer.clear()
        self._samples_since_infer = 0

        return True

    def feed_sample(self, ax, ay, az, gx, gy, gz):
        """Feed one IMU sample into the rolling buffer.

        Args:
            ax, ay, az, gx, gy, gz: float IMU values

        Returns:
            prediction dict if inference was triggered, None otherwise
        """
        self.imu_buffer.append([ax, ay, az, gx, gy, gz])
        self._samples_since_infer += 1

        # Run inference every STEP_SIZE samples once buffer is full
        if len(self.imu_buffer) >= WINDOW_SIZE and self._samples_since_infer >= STEP_SIZE:
            self._samples_since_infer = 0
            window = np.array(list(self.imu_buffer), dtype=np.float32)
            return self.predict_window(window)

        return None

    def predict_window(self, window):
        """Run inference on a complete (WINDOW_SIZE, NUM_AXES) window.

        Args:
            window: numpy array of shape (WINDOW_SIZE, NUM_AXES)

        Returns:
            dict with keys: gesture, confidence, all_probs, gesture_idx
        """
        if self.model is None:
            return None

        flat = window.flatten().reshape(1, -1)  # (1, 1500)

        if self.model_type == "mlp":
            # MLP: extract 36 features → scale → predict
            features = extract_features(flat)       # (1, 36)
            scaled = self.scaler.transform(features)
            probs = self.model.predict(scaled, verbose=0)[0]

        elif self.model_type == "cnn1d":
            # CNN1D: scale raw 1500 → reshape to (1, 250, 6) → predict
            scaled = self.scaler.transform(flat)
            reshaped = scaled.reshape(1, WINDOW_SIZE, NUM_AXES)
            probs = self.model.predict(reshaped, verbose=0)[0]

        elif self.model_type == "rf":
            # RF: extract 36 features → scale → predict probabilities
            features = extract_features(flat)
            scaled = self.scaler.transform(features)
            probs = self.model.predict_proba(scaled)[0]

        else:
            return None

        best_idx = int(np.argmax(probs))
        return {
            "gesture": GESTURE_NAMES[best_idx],
            "gesture_idx": best_idx,
            "confidence": float(probs[best_idx]),
            "all_probs": {name: float(probs[i]) for i, name in enumerate(GESTURE_NAMES)},
        }

    @property
    def is_loaded(self):
        return self.model is not None

    @staticmethod
    def get_available_models():
        """Check which trained models are available on disk."""
        available = []
        for mt in ["mlp", "cnn1d", "rf"]:
            if mt in ("mlp", "cnn1d"):
                path = os.path.join(MODELS_DIR, f"{mt}_model.keras")
            else:
                path = os.path.join(MODELS_DIR, f"{mt}_model.joblib")
            scaler_path = os.path.join(MODELS_DIR, f"{mt}_scaler.joblib")
            if os.path.isfile(path) and os.path.isfile(scaler_path):
                available.append(mt)
        return available

    def reset_buffer(self):
        """Clear the IMU sample buffer."""
        self.imu_buffer.clear()
        self._samples_since_infer = 0

    @staticmethod
    def _import_tf():
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import tensorflow as tf
        return tf
