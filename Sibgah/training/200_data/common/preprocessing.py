"""
Shared preprocessing utilities for all gesture classification models.

Workflow:
    1. load_file_list()       -> list of (path, label)
    2. split_file_list()      -> train / val / test at file level (no window leakage)
    3. get_class_names()      -> sorted class list (shared across models)
    4. extract_windows()      -> (X, y) arrays from a file list
    5. fit_scaler() / apply_scaler() -> per-feature standardisation
    6. compute_class_weights() -> handles class imbalance
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

FEATURES = [
    "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
    "ax2", "ay2", "az2", "gx2", "gy2", "gz2",
]

# Default window configuration.
# window_size = 25 covers most gesture recordings (12-30 rows) in one window.
# stride = 25 (non-overlapping) limits the number of idle windows.
WINDOW_SIZE = 25
STRIDE = 25
SEED = 42

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "cleaned_data")


def load_file_list(data_dir: str | None = None) -> list[tuple[str, str]]:
    """Return sorted list of (filepath, label) for every CSV under data_dir."""
    if data_dir is None:
        data_dir = _DATA_DIR
    file_list: list[tuple[str, str]] = []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.endswith(".csv"):
                file_list.append((os.path.join(label_dir, fname), label))
    return file_list


def get_class_names(file_list: list[tuple[str, str]]) -> list[str]:
    """Return sorted unique class names (consistent ordering for all models)."""
    return sorted(set(label for _, label in file_list))


def split_file_list(
    file_list: list[tuple[str, str]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = SEED,
) -> tuple[list, list, list]:
    """
    Stratified split at the recording-file level.
    Windows are generated after splitting so no CSV leaks across splits.
    Returns (train_list, val_list, test_list).
    """
    paths, labels = zip(*file_list)
    paths, labels = list(paths), list(labels)

    # Split off test first
    val_fraction_of_remainder = val_ratio / (1.0 - test_ratio)

    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        paths, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=val_fraction_of_remainder,
        stratify=labels_tv,
        random_state=seed,
    )

    return (
        list(zip(paths_train, labels_train)),
        list(zip(paths_val, labels_val)),
        list(zip(paths_test, labels_test)),
    )


def extract_windows(
    file_list: list[tuple[str, str]],
    class_names: list[str],
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a fixed-size window over each recording.
    Recordings shorter than window_size are zero-padded to produce one window.

    Returns:
        X: (N, window_size, n_features) float32
        y: (N,) int32 class indices
    """
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for path, label in file_list:
        df = pd.read_csv(path)
        data = df[FEATURES].values.astype(np.float32)
        T = len(data)

        if T < window_size:
            pad = np.zeros((window_size - T, len(FEATURES)), dtype=np.float32)
            data = np.vstack([data, pad])
            windows = [data]
        else:
            windows = []
            start = 0
            while start + window_size <= T:
                windows.append(data[start : start + window_size])
                start += stride

        for w in windows:
            X_list.append(w)
            y_list.append(label_to_idx[label])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on training windows. Normalises per feature channel."""
    N, T, F = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, F))
    return scaler


def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply a fitted scaler to X of shape (N, T, F)."""
    N, T, F = X.shape
    return scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)


def compute_class_weights(y: np.ndarray, num_classes: int) -> dict[int, float]:
    """Return balanced class weights as {class_idx: weight}."""
    weights = compute_class_weight(
        "balanced", classes=np.arange(num_classes), y=y
    )
    return {i: float(w) for i, w in enumerate(weights)}
