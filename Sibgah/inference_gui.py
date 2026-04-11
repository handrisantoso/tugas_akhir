"""
Gesture Inference GUI — BLE MPU6050 → CNN-1D & CNN-LSTM

Streams raw IMU data from the ESP32 over BLE, buffers 25-sample windows,
normalises them with the training scaler, and runs both models in real time.
"""

import os
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque

import numpy as np
from bleak import BleakClient, BleakScanner
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Constants ──────────────────────────────────────────────────────────────────
NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAX_POINTS  = 300
WINDOW_SIZE = 25
N_FEATURES  = 12
CLASSES     = ["down", "idle", "left", "right", "up"]

# Sliding-window step: run inference every STEP new samples (< WINDOW_SIZE
# means windows overlap, giving more responsive real-time detections).
INFER_STEP = 5

# StandardScaler params from scaler_params.h (identical for both models)
SCALER_MEAN = np.array([
    -2432.98876953, -17429.23437500,  -9946.20410156,
     -175.46310425,     36.79816818,    -52.00880051,
     3629.64111328,   5866.31250000, -11915.72167969,
     -129.84693909,     50.68962097,    -72.22756195,
], dtype=np.float32)

SCALER_STD = np.array([
    5394.58447266, 8113.98144531, 5292.03466797,
    5173.42138672, 8431.63769531, 5183.42333984,
    6975.25683594, 8895.53808594, 7948.84375000,
    9034.15722656, 6229.82470703, 8394.33105469,
], dtype=np.float32)

# Default model file locations (relative to this script)
_HERE      = os.path.dirname(os.path.abspath(__file__))
_TRAIN_DIR = os.path.join(_HERE, "training", "200_data")

DEFAULT_CNN1D_PT     = os.path.join(_TRAIN_DIR, "cnn_1d",   "output", "model.pt")
DEFAULT_CNN1D_ONNX   = os.path.join(_TRAIN_DIR, "cnn_1d",   "output_tflite", "cnn_1d.onnx")
DEFAULT_CNNLSTM_PT   = os.path.join(_TRAIN_DIR, "cnn_lstm", "output", "model.pt")
DEFAULT_CNNLSTM_ONNX = os.path.join(_TRAIN_DIR, "cnn_lstm", "output_tflite", "cnn_lstm.onnx")

CLASS_COLORS = ["#e74c3c", "#27ae60", "#2980b9", "#e67e22", "#8e44ad"]


# ── Model wrappers ─────────────────────────────────────────────────────────────

def _normalize(window: np.ndarray) -> np.ndarray:
    """Z-score per channel. window: (25, 12) → (25, 12) float32"""
    return ((window - SCALER_MEAN) / SCALER_STD).astype(np.float32)


class _CNN1D_PT:
    """Load CNN-1D from a .pt state-dict."""

    def __init__(self, path: str):
        import torch
        import torch.nn as nn

        n_classes = len(CLASSES)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(N_FEATURES, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                )
                self.head = nn.Sequential(
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(64, n_classes),
                )

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = x.mean(dim=2)
                return self.head(x)

        self._model = _Model()
        self._model.load_state_dict(torch.load(path, map_location="cpu"))
        self._model.eval()
        self._torch = torch

    def predict(self, window: np.ndarray):
        x = self._torch.from_numpy(window[np.newaxis])
        with self._torch.no_grad():
            logits = self._model(x)
            probs  = self._torch.softmax(logits, dim=1).numpy()[0]
        return int(np.argmax(probs)), probs


class _CNNLSTM_PT:
    """Load CNN-LSTM from a .pt state-dict."""

    def __init__(self, path: str):
        import torch
        import torch.nn as nn

        n_classes = len(CLASSES)

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(N_FEATURES, 64, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
                )
                self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Dropout(0.3), nn.Linear(64, n_classes),
                )

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.conv(x)
                x = x.permute(0, 2, 1)
                _, (h_n, _) = self.lstm(x)
                return self.head(h_n.squeeze(0))

        self._model = _Model()
        self._model.load_state_dict(torch.load(path, map_location="cpu"))
        self._model.eval()
        self._torch = torch

    def predict(self, window: np.ndarray):
        x = self._torch.from_numpy(window[np.newaxis])
        with self._torch.no_grad():
            logits = self._model(x)
            probs  = self._torch.softmax(logits, dim=1).numpy()[0]
        return int(np.argmax(probs)), probs


class _ONNXModel:
    """Generic ONNX Runtime inference wrapper."""

    def __init__(self, path: str):
        import onnxruntime as ort
        self._sess       = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def predict(self, window: np.ndarray):
        x      = window[np.newaxis].astype(np.float32)
        logits = self._sess.run(None, {self._input_name: x})[0][0]
        e      = np.exp(logits - logits.max())
        probs  = e / e.sum()
        return int(np.argmax(probs)), probs


def _load_model(pt_path: str, onnx_path: str, kind: str):
    """Try .pt first, then .onnx. Returns model or None."""
    if os.path.exists(pt_path):
        try:
            return _CNN1D_PT(pt_path) if kind == "cnn1d" else _CNNLSTM_PT(pt_path)
        except Exception as e:
            print(f"[WARN] .pt load failed ({e}), trying ONNX…")
    if os.path.exists(onnx_path):
        try:
            return _ONNXModel(onnx_path)
        except Exception as e:
            print(f"[WARN] ONNX load failed: {e}")
    return None


def _load_from_path(path: str, kind: str):
    """Load a model from an explicit path (auto-detect format)."""
    if path.lower().endswith(".onnx"):
        return _ONNXModel(path)
    return _CNN1D_PT(path) if kind == "cnn1d" else _CNNLSTM_PT(path)


# ── Main GUI ───────────────────────────────────────────────────────────────────

class InferenceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gesture Inference — CNN-1D & CNN-LSTM")
        self.root.geometry("1200x860")
        self.root.minsize(1000, 760)

        self.client      = None
        self.running     = False
        self.ble_devices: list = []

        # Rolling plot buffers (one per channel)
        self.buffers = [deque(maxlen=MAX_POINTS) for _ in range(N_FEATURES)]

        # Sliding window accumulator
        self._slide_buf: deque = deque(maxlen=WINDOW_SIZE)
        self._since_last_infer = 0   # samples since last inference run

        self.model_cnn1d   = None
        self.model_cnnlstm = None
        self._infer_count  = 0

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=lambda: (
            asyncio.set_event_loop(self.loop), self.loop.run_forever()
        ), daemon=True).start()

        self._build_ui()
        self._autoload_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Auto-load ──────────────────────────────────────────────────────────────

    def _autoload_models(self):
        m = _load_model(DEFAULT_CNN1D_PT, DEFAULT_CNN1D_ONNX, "cnn1d")
        if m:
            self.model_cnn1d = m
            found = DEFAULT_CNN1D_PT if os.path.exists(DEFAULT_CNN1D_PT) else DEFAULT_CNN1D_ONNX
            self.cnn1d_path_var.set(found)
            self.cnn1d_status.config(text="Loaded ✓", foreground="green")

        m = _load_model(DEFAULT_CNNLSTM_PT, DEFAULT_CNNLSTM_ONNX, "cnnlstm")
        if m:
            self.model_cnnlstm = m
            found = DEFAULT_CNNLSTM_PT if os.path.exists(DEFAULT_CNNLSTM_PT) else DEFAULT_CNNLSTM_ONNX
            self.lstm_path_var.set(found)
            self.lstm_status.config(text="Loaded ✓", foreground="green")

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_controls()
        self._build_plots()
        self._build_predictions()

    def _build_controls(self):
        ctrl = ttk.LabelFrame(self.root, text="Controls", padding=8)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 4))

        # BLE row
        ble_row = ttk.Frame(ctrl)
        ble_row.pack(fill=tk.X, pady=2)

        ttk.Label(ble_row, text="BLE Device:").pack(side=tk.LEFT)
        self.device_var   = tk.StringVar()
        self.device_combo = ttk.Combobox(ble_row, textvariable=self.device_var,
                                         width=34, state="readonly")
        self.device_combo.pack(side=tk.LEFT, padx=(5, 8))
        self.scan_btn    = ttk.Button(ble_row, text="Scan",      command=self._scan_devices)
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.connect_btn = ttk.Button(ble_row, text="Connect",   command=self._toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        self.conn_status = ttk.Label(ble_row, text="Disconnected", foreground="red")
        self.conn_status.pack(side=tk.LEFT, padx=10)

        # CNN-1D model row
        self.cnn1d_path_var = tk.StringVar()
        self.cnn1d_status   = ttk.Label(text="Not loaded", foreground="gray")
        self._model_row(ctrl, "CNN-1D model:", self.cnn1d_path_var,
                        "cnn1d", self.cnn1d_status)

        # CNN-LSTM model row
        self.lstm_path_var = tk.StringVar()
        self.lstm_status   = ttk.Label(text="Not loaded", foreground="gray")
        self._model_row(ctrl, "CNN-LSTM model:", self.lstm_path_var,
                        "cnnlstm", self.lstm_status)

    def _model_row(self, parent, label_text, path_var, kind, status_lbl):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label_text, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=path_var, width=52).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(row, text="Browse",
                   command=lambda: self._browse_model(kind, path_var)).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row, text="Load",
                   command=lambda: self._load_model_ui(kind, path_var, status_lbl)).pack(side=tk.LEFT, padx=(0, 8))
        status_lbl.pack_forget()          # re-parent into this row
        status_lbl = ttk.Label(row, text="Not loaded", foreground="gray")
        status_lbl.pack(side=tk.LEFT)
        # keep reference
        if kind == "cnn1d":
            self.cnn1d_status = status_lbl
        else:
            self.lstm_status = status_lbl

    def _build_plots(self):
        frame = ttk.LabelFrame(self.root, text="Live IMU Data", padding=4)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self.fig    = Figure(figsize=(11, 3.8), dpi=90)
        axes        = self.fig.subplots(2, 2)
        titles      = ["Sensor 1 — Accel", "Sensor 1 — Gyro",
                        "Sensor 2 — Accel", "Sensor 2 — Gyro"]
        accel_lbl   = ["ax", "ay", "az"]
        gyro_lbl    = ["gx", "gy", "gz"]

        self.ax_list = list(axes.flat)
        self.lines   = []

        for idx, ax in enumerate(self.ax_list):
            ax.set_title(titles[idx], fontsize=8)
            ax.set_xlim(0, MAX_POINTS)
            is_accel = (idx % 2 == 0)
            ax.set_ylim(-20000, 20000) if is_accel else ax.set_ylim(-35000, 35000)
            lbls  = accel_lbl if is_accel else gyro_lbl
            group = []
            for lbl in lbls:
                line, = ax.plot([], [], label=lbl)
                group.append(line)
            ax.legend(loc="upper left", fontsize=7)
            self.lines.append(group)

        self.fig.subplots_adjust(hspace=0.5, wspace=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_predictions(self):
        frame = ttk.LabelFrame(self.root, text="Predictions", padding=8)
        frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Window buffer progress
        prog_row = ttk.Frame(frame)
        prog_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(prog_row, text="Window buffer:").pack(side=tk.LEFT)
        self.win_progress = ttk.Progressbar(prog_row, maximum=WINDOW_SIZE,
                                            length=180, mode="determinate")
        self.win_progress.pack(side=tk.LEFT, padx=(6, 6))
        self.win_label = ttk.Label(prog_row, text=f"0 / {WINDOW_SIZE}",
                                   foreground="gray")
        self.win_label.pack(side=tk.LEFT)
        self.infer_label = ttk.Label(prog_row, text="", foreground="steelblue")
        self.infer_label.pack(side=tk.LEFT, padx=(24, 0))

        # Two side-by-side prediction columns
        cols = ttk.Frame(frame)
        cols.pack(fill=tk.X)
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        self._pred_vars  = {}
        self._prob_bars  = {}

        for col_idx, (model_name, title) in enumerate([
            ("cnn1d",   "CNN-1D"),
            ("cnnlstm", "CNN-LSTM"),
        ]):
            col = ttk.LabelFrame(cols, text=title, padding=8)
            col.grid(row=0, column=col_idx, padx=6, pady=2, sticky="nsew")

            pred_var = tk.StringVar(value="—")
            self._pred_vars[model_name] = pred_var
            ttk.Label(col, textvariable=pred_var,
                      font=("Helvetica", 20, "bold"),
                      foreground="#2c3e50").pack(pady=(0, 8))

            bars = []
            for i, cls in enumerate(CLASSES):
                row = ttk.Frame(col)
                row.pack(fill=tk.X, pady=1)
                ttk.Label(row, text=f"{cls:<18}", width=18,
                          font=("Consolas", 9)).pack(side=tk.LEFT)
                bar = ttk.Progressbar(row, maximum=100, length=200,
                                      mode="determinate")
                bar.pack(side=tk.LEFT, padx=(4, 4))
                pct = ttk.Label(row, text="  0.0%", width=7,
                                font=("Consolas", 9))
                pct.pack(side=tk.LEFT)
                bars.append((bar, pct))
            self._prob_bars[model_name] = bars

    # ── Model loading ──────────────────────────────────────────────────────────

    def _browse_model(self, kind: str, path_var: tk.StringVar):
        path = filedialog.askopenfilename(
            title=f"Select model file ({kind})",
            filetypes=[
                ("Model files", "*.pt *.onnx"),
                ("PyTorch state-dict", "*.pt"),
                ("ONNX", "*.onnx"),
                ("All", "*.*"),
            ],
        )
        if path:
            path_var.set(path)

    def _load_model_ui(self, kind: str, path_var: tk.StringVar,
                       status_lbl: ttk.Label):
        path = path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("File not found", f"Path does not exist:\n{path}")
            return

        status_lbl.config(text="Loading…", foreground="orange")
        self.root.update_idletasks()

        try:
            model = _load_from_path(path, kind)
            if kind == "cnn1d":
                self.model_cnn1d = model
            else:
                self.model_cnnlstm = model
            status_lbl.config(text="Loaded ✓", foreground="green")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            status_lbl.config(text=f"Error", foreground="red")

    # ── BLE ────────────────────────────────────────────────────────────────────

    def _scan_devices(self):
        self.scan_btn.config(state="disabled")
        self.conn_status.config(text="Scanning…", foreground="orange")
        future = asyncio.run_coroutine_threadsafe(
            BleakScanner.discover(timeout=5.0), self.loop
        )

        def _check():
            if not future.done():
                self.root.after(200, _check)
                return
            try:
                devices          = future.result()
                self.ble_devices = [d for d in devices if d.name]
                names            = [f"{d.name}  ({d.address})"
                                    for d in self.ble_devices]
                self.device_combo["values"] = names
                if names:
                    self.device_combo.current(0)
                self.conn_status.config(
                    text=f"Found {len(self.ble_devices)} device(s)",
                    foreground="blue",
                )
            except Exception as e:
                messagebox.showerror("Scan Error", str(e))
                self.conn_status.config(text="Scan failed", foreground="red")
            finally:
                self.scan_btn.config(state="normal")

        self.root.after(200, _check)

    def _toggle_connection(self):
        if self.client and self.client.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        idx = self.device_combo.current()
        if idx < 0 or not self.ble_devices:
            messagebox.showwarning("No Device", "Scan and select a BLE device first.")
            return

        device = self.ble_devices[idx]
        self.conn_status.config(text="Connecting…", foreground="orange")
        self.connect_btn.config(state="disabled")

        async def _do():
            c = BleakClient(device.address)
            await c.connect()
            await c.start_notify(NOTIFY_UUID, self._handle_data)
            return c

        future = asyncio.run_coroutine_threadsafe(_do(), self.loop)

        def _check():
            if not future.done():
                self.root.after(200, _check)
                return
            try:
                self.client  = future.result()
                self.running = True
                self.conn_status.config(
                    text=f"Connected: {device.name}", foreground="green"
                )
                self.connect_btn.config(text="Disconnect", state="normal")
                self._update_plot()
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))
                self.conn_status.config(text="Connection failed", foreground="red")
                self.connect_btn.config(state="normal")

        self.root.after(200, _check)

    def _disconnect(self):
        self.running = False
        self.connect_btn.config(state="disabled")

        async def _do():
            if self.client:
                try:
                    await self.client.stop_notify(NOTIFY_UUID)
                except Exception:
                    pass
                await self.client.disconnect()

        future = asyncio.run_coroutine_threadsafe(_do(), self.loop)

        def _check():
            if not future.done():
                self.root.after(200, _check)
                return
            self.client = None
            self.conn_status.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect", state="normal")

        self.root.after(200, _check)

    # ── Data handling ──────────────────────────────────────────────────────────

    def _handle_data(self, _sender, data: bytearray):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return

        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != N_FEATURES:
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                continue

            # Update rolling plot buffers
            for buf, val in zip(self.buffers, values):
                buf.append(val)

            # Accumulate into sliding window
            self._slide_buf.append(values)
            self._since_last_infer += 1

            # Run inference every INFER_STEP samples once we have a full window
            if (len(self._slide_buf) == WINDOW_SIZE
                    and self._since_last_infer >= INFER_STEP):
                self._since_last_infer = 0
                window = np.array(list(self._slide_buf), dtype=np.float32)
                self._run_inference(window)

    def _run_inference(self, window: np.ndarray):
        """Normalise window and query both models (called from BLE thread)."""
        norm   = _normalize(window)
        result = {}
        for name, model in [("cnn1d", self.model_cnn1d),
                             ("cnnlstm", self.model_cnnlstm)]:
            if model is None:
                continue
            try:
                cls_idx, probs = model.predict(norm)
                result[name]   = (cls_idx, probs)
            except Exception as e:
                print(f"[WARN] Inference error ({name}): {e}")

        self._infer_count += 1
        self.root.after(0, lambda r=result: self._update_predictions(r))

    def _update_predictions(self, result: dict):
        for name in ("cnn1d", "cnnlstm"):
            if name not in result:
                continue
            cls_idx, probs = result[name]
            self._pred_vars[name].set(CLASSES[cls_idx].upper())
            for i, (bar, lbl) in enumerate(self._prob_bars[name]):
                pct = float(probs[i]) * 100.0
                bar["value"] = pct
                lbl.config(text=f"{pct:5.1f}%")

        self.infer_label.config(
            text=f"Inferences: {self._infer_count}"
        )

    # ── Plot refresh ───────────────────────────────────────────────────────────

    def _update_plot(self):
        if not self.running:
            return

        buf_len = len(self._slide_buf)
        self.win_progress["value"] = buf_len
        self.win_label.config(text=f"{buf_len} / {WINDOW_SIZE}")

        if self.buffers[0]:
            x          = range(len(self.buffers[0]))
            idx_groups = [
                [0, 1, 2], [3, 4, 5],     # sensor 1  accel / gyro
                [6, 7, 8], [9, 10, 11],    # sensor 2  accel / gyro
            ]
            for g_idx, indices in enumerate(idx_groups):
                for l_idx, b_idx in enumerate(indices):
                    self.lines[g_idx][l_idx].set_data(x, list(self.buffers[b_idx]))
                self.ax_list[g_idx].set_xlim(
                    0, max(len(self.buffers[0]), MAX_POINTS)
                )
            self.canvas.draw_idle()

        self.root.after(80, self._update_plot)

    # ── Close ──────────────────────────────────────────────────────────────────

    def _on_close(self):
        self.running = False
        if self.client and self.client.is_connected:
            future = asyncio.run_coroutine_threadsafe(
                self.client.disconnect(), self.loop
            )
            try:
                future.result(timeout=3)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
