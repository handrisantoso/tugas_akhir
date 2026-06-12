"""
Gesture Glove — Serial Recorder
Handles serial communication with ESP32-S3 and manual gesture recording.
"""

import serial
import serial.tools.list_ports
import threading
import time
import queue
import csv
import os
import numpy as np
import pandas as pd
from config import (
    BAUD_RATE, WINDOW_SIZE, NUM_AXES, NUM_FEATURES,
    GESTURE_NAMES, CSV_COLUMNS, DEFAULT_CSV, DATASET_DIR, SAMPLE_RATE_HZ
)


class SerialManager:
    """Thread-safe serial port manager for ESP32-S3 communication."""

    def __init__(self):
        self._serial = None
        self._lock = threading.Lock()
        self._reader_thread = None
        self._running = False

        # Callbacks
        self.on_imu_data = None       # callback(ax, ay, az, gx, gy, gz)
        self.on_prediction = None     # callback(gesture_name, confidence, latency_ms)
        self.on_raw_line = None       # callback(line_str)
        self.on_status_change = None  # callback(status_str)

        # Data queue for recorder
        self._data_queue = queue.Queue()

    @staticmethod
    def list_ports():
        """Return list of available COM port names."""
        ports = serial.tools.list_ports.comports()
        return [(p.device, f"{p.device} - {p.description}") for p in ports]

    @property
    def is_connected(self):
        with self._lock:
            return self._serial is not None and self._serial.is_open

    def connect(self, port, baud=BAUD_RATE):
        """Open serial connection to ESP32."""
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.close()

            try:
                self._serial = serial.Serial(port, baud, timeout=0.1)
                time.sleep(0.5)  # Wait for ESP32 to reset
                self._serial.reset_input_buffer()
            except serial.SerialException as e:
                self._serial = None
                raise ConnectionError(f"Failed to connect to {port}: {e}")

        # Start reader thread
        self._running = True
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()

        if self.on_status_change:
            self.on_status_change("connected")

    def disconnect(self):
        """Close serial connection."""
        self._running = False
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None

        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.close()
            self._serial = None

        if self.on_status_change:
            self.on_status_change("disconnected")

    def send_command(self, cmd):
        """Send a command string to ESP32 (e.g., 'S', 'I', 'B', 'T0.85')."""
        with self._lock:
            if self._serial and self._serial.is_open:
                self._serial.write(f"{cmd}\n".encode("utf-8"))
                self._serial.flush()

    def get_imu_sample(self, timeout=0.05):
        """Get one IMU sample from the queue. Returns (ax,ay,az,gx,gy,gz) or None."""
        try:
            return self._data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Flush all pending IMU samples from the queue."""
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

    def _read_loop(self):
        """Background thread: read and parse serial lines."""
        while self._running:
            try:
                # Grab serial ref briefly — don't hold lock during IO
                with self._lock:
                    ser = self._serial
                if not ser or not ser.is_open:
                    break

                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        self._process_line(line)
                        continue

                time.sleep(0.002)  # Small sleep to avoid busy-waiting
            except (serial.SerialException, OSError):
                self._running = False
                if self.on_status_change:
                    self.on_status_change("error")
                break
            except Exception:
                continue

    def _process_line(self, line):
        """Parse a serial line — either IMU CSV data or prediction."""
        # Fire raw line callback
        if self.on_raw_line:
            self.on_raw_line(line)

        # Check for prediction line: [PRED] gesture_name confidence latency_ms
        if line.startswith("[PRED]"):
            parts = line.split()
            if len(parts) >= 4:
                gesture = parts[1]
                try:
                    confidence = float(parts[2])
                    latency = int(parts[3].replace("ms", ""))
                    if self.on_prediction:
                        self.on_prediction(gesture, confidence, latency)
                except (ValueError, IndexError):
                    pass
            return

        # Try parsing as CSV IMU data: ax,ay,az,gx,gy,gz
        parts = line.split(",")
        if len(parts) == NUM_AXES:
            try:
                values = tuple(float(v) for v in parts)
                self._data_queue.put(values)
                if self.on_imu_data:
                    self.on_imu_data(*values)
            except ValueError:
                pass


class GestureRecorder:
    """Manages manual gesture recording sessions and CSV persistence."""

    def __init__(self, serial_manager: SerialManager):
        self.serial = serial_manager
        self._recording = False
        self._buffer = []
        self._current_label = 0
        self._lock = threading.Lock()
        self._record_thread = None

        # Recorded samples for current session
        self.session_samples = []   # list of (flat_array, label) tuples
        self.sample_counts = {name: 0 for name in GESTURE_NAMES}

    @property
    def is_recording(self):
        with self._lock:
            return self._recording

    def start_recording(self, gesture_label: int):
        """Begin recording a gesture window."""
        with self._lock:
            if self._recording:
                return False
            self._recording = True
            self._current_label = gesture_label
            self._buffer = []

        # Clear any stale data
        self.serial.clear_queue()

        # Start recording thread
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        return True

    def stop_recording(self):
        """Stop recording and return the captured data.

        Returns:
            tuple: (flat_array of 1500 floats, label, sample_count) or None if failed
        """
        with self._lock:
            self._recording = False

        if self._record_thread:
            self._record_thread.join(timeout=5.0)
            self._record_thread = None

        with self._lock:
            raw_buffer = list(self._buffer)
            label = self._current_label

        num_samples = len(raw_buffer)

        if num_samples == 0:
            return None

        # Pad or truncate to exactly WINDOW_SIZE
        if num_samples < WINDOW_SIZE:
            # Pad with the last sample (hold position)
            last_sample = raw_buffer[-1] if raw_buffer else (0.0,) * NUM_AXES
            while len(raw_buffer) < WINDOW_SIZE:
                raw_buffer.append(last_sample)
        elif num_samples > WINDOW_SIZE:
            # Truncate to WINDOW_SIZE
            raw_buffer = raw_buffer[:WINDOW_SIZE]

        # Flatten: [(ax,ay,az,gx,gy,gz), ...] → [ax_0, ay_0, az_0, gx_0, gy_0, gz_0, ax_1, ...]
        flat = []
        for sample in raw_buffer:
            flat.extend(sample)

        flat_array = np.array(flat, dtype=np.float32)

        # Store in session
        self.session_samples.append((flat_array, label))
        gesture_name = GESTURE_NAMES[label]
        self.sample_counts[gesture_name] = self.sample_counts.get(gesture_name, 0) + 1

        return flat_array, label, num_samples

    def _record_loop(self):
        """Background thread: accumulate IMU samples while recording."""
        while True:
            with self._lock:
                if not self._recording:
                    break

            sample = self.serial.get_imu_sample(timeout=0.02)
            if sample is not None:
                with self._lock:
                    if self._recording:
                        self._buffer.append(sample)

    def get_last_window(self):
        """Return the last recorded window as a (WINDOW_SIZE, NUM_AXES) array for visualization."""
        if not self.session_samples:
            return None
        flat, _ = self.session_samples[-1]
        return flat.reshape(WINDOW_SIZE, NUM_AXES)

    def save_to_csv(self, filepath=None, subject_id=None):
        """Save all session samples to CSV file in 'Long' format.

        Args:
            filepath: Path to CSV file. If None, uses default path.
            subject_id: Explicit subject identifier string. If None, defaults
                        to session_id (Unix timestamp). Pass an explicit value
                        when recording data for a known subject to enable
                        multi-subject / LOSO support.
        """
        if filepath is None:
            filepath = DEFAULT_CSV

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file_exists = os.path.isfile(filepath) and os.path.getsize(filepath) > 0

        session_id = int(time.time())
        # Use explicit subject_id if provided; otherwise fall back to session_id
        subj = subject_id if subject_id is not None else str(session_id)

        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow(CSV_COLUMNS)

            window_count = 0
            for window_idx, (flat_array, label_idx) in enumerate(self.session_samples):
                gesture_name = GESTURE_NAMES[label_idx]
                sample_id = f"{gesture_name}_{session_id}_{window_idx}"

                # Reshape back to (WINDOW_SIZE, NUM_AXES)
                data = flat_array.reshape(WINDOW_SIZE, NUM_AXES)

                for t in range(WINDOW_SIZE):
                    timestamp_ms = t * (1000 // SAMPLE_RATE_HZ)
                    # Row: label, sample_id, timestamp_ms, subject_id, ax, ay, az, gx, gy, gz
                    row = [gesture_name, sample_id, timestamp_ms, subj] + data[t].tolist()
                    writer.writerow(row)

                window_count += 1

        return filepath, window_count

    def clear_session(self):
        """Clear all recorded samples from the current session."""
        self.session_samples = []
        self.sample_counts = {name: 0 for name in GESTURE_NAMES}

    @staticmethod
    def _read_csv_robust(filepath):
        """Read a CSV whose rows may have 9 or 10 columns due to a format mismatch.

        The recorder writes 10-column rows (with subject_id at index 3) but
        early sessions wrote a 9-column header (no subject_id). This normalises
        everything to 10 columns, inferring subject_id from the session timestamp
        embedded in sample_id for old 9-column rows.
        """
        import re as _re
        cols_10 = ['label', 'sample_id', 'timestamp_ms', 'subject_id',
                   'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        numeric_cols = ['timestamp_ms', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        rows = []
        with open(filepath, 'r', newline='', encoding='utf-8') as f:
            _reader = csv.reader(f)
            next(_reader)  # skip file header (may be 9-col, missing subject_id)
            for fields in _reader:
                if len(fields) == 10:
                    # New format: label, sample_id, ts, subject_id, ax..gz
                    rows.append(dict(zip(cols_10, fields)))
                elif len(fields) == 9:
                    # Old format: label, sample_id, ts, ax..gz — infer subject_id
                    m = _re.match(r'^[a-z_]+_(\d+)_\d+$', fields[1])
                    subj = m.group(1) if m else 'unknown'
                    d = {'label': fields[0], 'sample_id': fields[1],
                         'timestamp_ms': fields[2], 'subject_id': subj,
                         'ax': fields[3], 'ay': fields[4], 'az': fields[5],
                         'gx': fields[6], 'gy': fields[7], 'gz': fields[8]}
                    rows.append(d)
        df = pd.DataFrame(rows, columns=cols_10)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def load_csv(filepath=None):
        """Load a dataset CSV and return sample counts per gesture."""
        if filepath is None:
            filepath = DEFAULT_CSV

        if not os.path.isfile(filepath):
            return None, {name: 0 for name in GESTURE_NAMES}

        try:
            df = pd.read_csv(filepath)
        except pd.errors.ParserError:
            df = GestureRecorder._read_csv_robust(filepath)

        # Detection: is it "long" format (per-sample) or "wide" format (per-window)?
        if "sample_id" in df.columns and "label" in df.columns:
            # Long format
            counts = {}
            for name in GESTURE_NAMES:
                # Handle both string and int labels
                subset = df[df["label"].astype(str) == str(name)]
                if subset.empty and name in GESTURE_NAMES:
                    try:
                        idx = GESTURE_NAMES.index(name)
                        subset = df[df["label"].astype(str) == str(idx)]
                    except: pass
                counts[name] = len(subset["sample_id"].unique())
            return df, counts
        else:
            # Wide format
            counts = {}
            for name in GESTURE_NAMES:
                try:
                    label_idx = GESTURE_NAMES.index(name)
                    counts[name] = int((df["label"] == label_idx).sum())
                except:
                    counts[name] = 0
            return df, counts

    @staticmethod
    def load_dataset_from_folders(dataset_dir=None):
        """Load dataset from per-gesture subdirectories (one CSV file per window).

        File naming: {gesture}_{subject_id}_{index}.csv
        Each CSV has columns: timestamp_ms, ax, ay, az, gx, gy, gz (WINDOW_SIZE rows)

        Returns:
            X           : np.ndarray, shape (N, NUM_FEATURES), dtype float32
            y           : np.ndarray, shape (N,), dtype int32
            subject_ids : np.ndarray, shape (N,), dtype object
        """
        import re as _re
        if dataset_dir is None:
            dataset_dir = DATASET_DIR

        X, y, subject_ids = [], [], []
        label_map = {name: i for i, name in enumerate(GESTURE_NAMES)}

        for gesture_name, label_idx in label_map.items():
            folder = os.path.join(dataset_dir, gesture_name)
            if not os.path.isdir(folder):
                continue
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith('.csv'):
                    continue
                fpath = os.path.join(folder, fname)
                try:
                    df = pd.read_csv(fpath)
                    if len(df) < WINDOW_SIZE:
                        continue
                    data = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].iloc[:WINDOW_SIZE].values.flatten()
                    m = _re.match(rf'^{_re.escape(gesture_name)}_(\d+)_\d+\.csv$', fname)
                    subj = m.group(1) if m else 'unknown'
                    X.append(data)
                    y.append(label_idx)
                    subject_ids.append(subj)
                except Exception:
                    continue

        return (
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(subject_ids, dtype=object),
        )

    @staticmethod
    def count_folder_samples(dataset_dir):
        """Return total number of gesture window CSV files across all gesture subfolders."""
        total = 0
        for name in GESTURE_NAMES:
            subdir = os.path.join(dataset_dir, name)
            if os.path.isdir(subdir):
                total += sum(1 for f in os.listdir(subdir) if f.endswith('.csv'))
        return total

    @staticmethod
    def load_dataset(filepath=None):
        """Load dataset CSV and return X (features), y (labels), and subject_ids.

        Returns:
            X          : np.ndarray, shape (N, NUM_FEATURES), dtype float32
            y          : np.ndarray, shape (N,), dtype int32
            subject_ids: np.ndarray, shape (N,), dtype object
                         Unique values identify subjects.

        Backward compatibility:
            Legacy CSVs without a subject_id column produce
            subject_ids = np.array(["single_subject"] * N),
            which triggers the single-subject fallback in train_model().
        """
        if filepath is None:
            filepath = DEFAULT_CSV

        try:
            df = pd.read_csv(filepath)
        except pd.errors.ParserError:
            df = GestureRecorder._read_csv_robust(filepath)

        # Handle "Long" Format (User's format)
        if "sample_id" in df.columns and "ax" in df.columns:
            X = []
            y = []
            subject_ids = []
            label_map = {name: i for i, name in enumerate(GESTURE_NAMES)}
            has_subject_col = "subject_id" in df.columns

            for sid, group in df.groupby("sample_id"):
                if len(group) < WINDOW_SIZE:
                    continue # Skip too short

                # Take first WINDOW_SIZE samples
                window = group.iloc[:WINDOW_SIZE]
                data = window[["ax", "ay", "az", "gx", "gy", "gz"]].values.flatten()

                label_val = window["label"].iloc[0]
                if isinstance(label_val, str):
                    label_idx = label_map.get(label_val, -1)
                else:
                    label_idx = int(label_val)

                # Subject ID: explicit column → infer from sample_id → placeholder
                if has_subject_col:
                    subj = str(window["subject_id"].iloc[0])
                else:
                    import re as _re
                    m = _re.match(r'^[a-z_]+_(\d+)_\d+$', str(sid))
                    subj = m.group(1) if m else "single_subject"

                if label_idx != -1:
                    X.append(data)
                    y.append(label_idx)
                    subject_ids.append(subj)

            return (
                np.array(X, dtype=np.float32),
                np.array(y, dtype=np.int32),
                np.array(subject_ids, dtype=object),
            )

        # Handle "Wide" Format (Default)
        else:
            X = df.iloc[:, :NUM_FEATURES].values.astype(np.float32)
            y = df["label"].values.astype(np.int32)
            return X, y, np.array(["single_subject"] * len(y), dtype=object)
