import os
import csv
import time
import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from bleak import BleakClient, BleakScanner
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque

NOTIFY_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
MAX_POINTS = 300
CLASSES = ["right", "left", "up", "down", "clockwise", "counter_clockwise", "idle"]
DATASET_DIR = "dataset"
VALUES_PER_SENSOR = 6

# Hold a key to record that class; release to save and resume no_movement.
# no_movement is recorded automatically whenever no key is held.
KEY_MAP = {
    "1": "right",
    "2": "left",
    "3": "up",
    "4": "down",
    "5": "clockwise",
    "6": "counter_clockwise",
}


class DatasetRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MPU6050 Dataset Recorder (BLE)")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)

        self.client = None
        self.recording = False
        self.record_data = []
        self.record_start_time = None
        self.running = False
        self.num_sensors = 2
        self.ble_devices = []
        self._held_key = None       # tracks which key is currently held
        self._idle_save_job = None  # scheduled auto-save job for idle

        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._run_loop, daemon=True).start()

        self._ensure_folders()
        self._build_ui()
        self._apply_sensor_config()
        self._update_sample_counts()
        self._bind_shortcuts()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _ensure_folders(self):
        for cls in CLASSES:
            os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

    def _build_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill=tk.X, pady=2)

        ttk.Label(row1, text="Sensors:").pack(side=tk.LEFT)
        self.sensor_var = tk.IntVar(value=2)
        sensor_combo = ttk.Combobox(
            row1, textvariable=self.sensor_var, values=[1, 2], width=3, state="readonly"
        )
        sensor_combo.pack(side=tk.LEFT, padx=(5, 15))
        sensor_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_sensor_config())

        ttk.Label(row1, text="BLE Device:").pack(side=tk.LEFT)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            row1, textvariable=self.device_var, width=30, state="readonly"
        )
        self.device_combo.pack(side=tk.LEFT, padx=(5, 10))
        self.scan_btn = ttk.Button(row1, text="Scan", command=self._scan_devices)
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.connect_btn = ttk.Button(row1, text="Connect", command=self._toggle_connection)
        self.connect_btn.pack(side=tk.LEFT)
        self.conn_status = ttk.Label(row1, text="Disconnected", foreground="red")
        self.conn_status.pack(side=tk.LEFT, padx=10)

        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="Class:").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value=CLASSES[0])
        self.class_combo = ttk.Combobox(
            row2, textvariable=self.class_var, values=CLASSES, width=20, state="readonly"
        )
        self.class_combo.pack(side=tk.LEFT, padx=(5, 10))

        self.record_btn = ttk.Button(row2, text="⏺ Record", command=self._toggle_record)
        self.record_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.rec_label = ttk.Label(row2, text="", foreground="red")
        self.rec_label.pack(side=tk.LEFT, padx=5)

        # Shortcut hint row
        row3 = ttk.Frame(ctrl_frame)
        row3.pack(fill=tk.X, pady=(4, 0))
        hint = "  Hold to record, release to save  |  " + \
               "  ".join(f"[{k}] {v}" for k, v in KEY_MAP.items()) + \
               "  |  no key held → idle (auto)"
        ttk.Label(row3, text=hint, foreground="gray", font=("Consolas", 8)).pack(side=tk.LEFT)

        # Sample counts — two rows of 4 + 3
        count_frame = ttk.LabelFrame(self.root, text="Sample Counts", padding=10)
        count_frame.pack(fill=tk.X, padx=10, pady=5)

        self.count_labels = {}
        cols_per_row = 4
        for i, cls in enumerate(CLASSES):
            row_idx = i // cols_per_row
            col_idx = (i % cols_per_row) * 2
            ttk.Label(count_frame, text=f"{cls}:").grid(row=row_idx, column=col_idx, padx=(10, 2), sticky=tk.W)
            lbl = ttk.Label(count_frame, text="0", font=("Consolas", 11, "bold"))
            lbl.grid(row=row_idx, column=col_idx + 1, padx=(0, 10), sticky=tk.W)
            self.count_labels[cls] = lbl

        self.fig = Figure(figsize=(10, 5), dpi=90)

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _bind_shortcuts(self):
        for key in KEY_MAP:
            self.root.bind(f"<KeyPress-{key}>", self._on_key_press)
            self.root.bind(f"<KeyRelease-{key}>", self._on_key_release)

    def _auto_start_idle(self):
        # Cancel any pending auto-save before starting fresh
        if self._idle_save_job is not None:
            self.root.after_cancel(self._idle_save_job)
            self._idle_save_job = None

        self.class_var.set("idle")
        self.recording = True
        self.record_data = []
        self.record_start_time = time.time()
        self.record_btn.config(text="⏹ Stop")
        self.rec_label.config(text="REC  idle  0.0s  |  0 samples")

        # Auto-save idle every 30 seconds and restart
        self._idle_save_job = self.root.after(30_000, self._auto_save_idle)

    def _auto_save_idle(self):
        self._idle_save_job = None
        if self.recording and self.class_var.get() == "idle":
            self._stop_record()
            self._auto_start_idle()

    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key not in KEY_MAP:
            return
        # Guard against Windows auto-repeat firing multiple press events
        if self._held_key == key:
            return
        self._held_key = key
        # Cancel the 30s auto-save since we're manually saving idle now
        if self._idle_save_job is not None:
            self.root.after_cancel(self._idle_save_job)
            self._idle_save_job = None
        # Save idle chunk accumulated so far, then start the gesture
        if self.recording:
            self._stop_record()
        self.class_var.set(KEY_MAP[key])
        self._start_record()

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key != self._held_key:
            return
        self._held_key = None
        # Save the gesture chunk, then resume idle automatically
        if self.recording:
            self._stop_record()
        self._auto_start_idle()

    def _apply_sensor_config(self):
        self.num_sensors = self.sensor_var.get()
        total_values = VALUES_PER_SENSOR * self.num_sensors
        self.buffers = [deque(maxlen=MAX_POINTS) for _ in range(total_values)]

        self.fig.clear()
        accel_labels = ["ax", "ay", "az"]
        gyro_labels = ["gx", "gy", "gz"]
        self.lines = []

        if self.num_sensors == 1:
            axes = self.fig.subplots(1, 2)
            titles = ["Accelerometer", "Gyroscope"]
        else:
            axes = self.fig.subplots(2, 2)
            titles = [
                "Sensor 1 - Accel", "Sensor 1 - Gyro",
                "Sensor 2 - Accel", "Sensor 2 - Gyro",
            ]

        self.ax_list = list(axes.flat)
        for idx, ax in enumerate(self.ax_list):
            ax.set_title(titles[idx], fontsize=9)
            ax.set_xlim(0, MAX_POINTS)
            is_accel = idx % 2 == 0
            ax.set_ylim(-20000, 20000) if is_accel else ax.set_ylim(-35000, 35000)
            lbls = accel_labels if is_accel else gyro_labels
            group = []
            for lbl in lbls:
                line, = ax.plot([], [], label=lbl)
                group.append(line)
            ax.legend(loc="upper left", fontsize=7)
            self.lines.append(group)

        self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
        self.canvas.draw_idle()

    def _scan_devices(self):
        self.scan_btn.config(state="disabled")
        self.conn_status.config(text="Scanning...", foreground="orange")

        future = asyncio.run_coroutine_threadsafe(
            BleakScanner.discover(timeout=5.0), self.loop
        )

        def on_done():
            if not future.done():
                self.root.after(200, on_done)
                return
            try:
                devices = future.result()
                self.ble_devices = [d for d in devices if d.name]
                names = [f"{d.name} ({d.address})" for d in self.ble_devices]
                self.device_combo["values"] = names
                if names:
                    self.device_combo.current(0)
                self.conn_status.config(
                    text=f"Found {len(self.ble_devices)} device(s)", foreground="blue"
                )
            except Exception as e:
                messagebox.showerror("Scan Error", str(e))
                self.conn_status.config(text="Scan failed", foreground="red")
            finally:
                self.scan_btn.config(state="normal")

        self.root.after(200, on_done)

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
        self.conn_status.config(text="Connecting...", foreground="orange")
        self.connect_btn.config(state="disabled")

        async def do_connect():
            client = BleakClient(device.address)
            await client.connect()
            await client.start_notify(NOTIFY_UUID, self._handle_data)
            return client

        future = asyncio.run_coroutine_threadsafe(do_connect(), self.loop)

        def on_done():
            if not future.done():
                self.root.after(200, on_done)
                return
            try:
                self.client = future.result()
                self.running = True
                self.conn_status.config(
                    text=f"Connected: {device.name}", foreground="green"
                )
                self.connect_btn.config(text="Disconnect", state="normal")
                self._update_plot()
                self._auto_start_idle()
            except Exception as e:
                messagebox.showerror("Connection Error", str(e))
                self.conn_status.config(text="Connection failed", foreground="red")
                self.connect_btn.config(state="normal")

        self.root.after(200, on_done)

    def _disconnect(self):
        if self.recording:
            self._stop_record()
        self.running = False
        self.connect_btn.config(state="disabled")

        async def do_disconnect():
            if self.client:
                try:
                    await self.client.stop_notify(NOTIFY_UUID)
                except Exception:
                    pass
                await self.client.disconnect()

        future = asyncio.run_coroutine_threadsafe(do_disconnect(), self.loop)

        def on_done():
            if not future.done():
                self.root.after(200, on_done)
                return
            self.client = None
            self.conn_status.config(text="Disconnected", foreground="red")
            self.connect_btn.config(text="Connect", state="normal")

        self.root.after(200, on_done)

    def _handle_data(self, _sender, data: bytearray):
        expected = VALUES_PER_SENSOR * self.num_sensors
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != expected:
                continue
            try:
                values = [float(v) for v in parts]
            except ValueError:
                continue

            for buf, val in zip(self.buffers, values):
                buf.append(val)

            if self.recording:
                elapsed = time.time() - self.record_start_time
                self.record_data.append([elapsed] + values)

    def _update_plot(self):
        if not self.running:
            return

        if self.buffers[0]:
            x = range(len(self.buffers[0]))
            buf_groups = []
            for s in range(self.num_sensors):
                offset = s * VALUES_PER_SENSOR
                buf_groups.append([offset, offset + 1, offset + 2])
                buf_groups.append([offset + 3, offset + 4, offset + 5])

            for group_idx, indices in enumerate(buf_groups):
                for line_idx, buf_idx in enumerate(indices):
                    self.lines[group_idx][line_idx].set_data(x, list(self.buffers[buf_idx]))
                self.ax_list[group_idx].set_xlim(0, max(len(self.buffers[0]), MAX_POINTS))

            self.canvas.draw_idle()

        if self.recording:
            elapsed = time.time() - self.record_start_time
            count = len(self.record_data)
            cls = self.class_var.get()
            self.rec_label.config(text=f"REC [{cls}]  {elapsed:.1f}s  |  {count} samples")

        self.root.after(80, self._update_plot)

    def _toggle_record(self):
        if self.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        if not self.client or not self.client.is_connected:
            messagebox.showwarning("Not Connected", "Connect to a BLE device first.")
            return

        self.recording = True
        self.record_data = []
        self.record_start_time = time.time()
        self.record_btn.config(text="⏹ Stop")
        self.rec_label.config(text="REC  0.0s  |  0 samples")

    def _stop_record(self):
        self.recording = False
        self.record_btn.config(text="⏺ Record")

        if not self.record_data:
            self.rec_label.config(text="No data recorded.")
            return

        cls = self.class_var.get()
        self._save_csv(cls, self.record_data)
        count = len(self.record_data)
        self.rec_label.config(text=f"Saved {count} samples to '{cls}'")
        self.record_data = []
        self._update_sample_counts()

    def _save_csv(self, cls, data):
        folder = os.path.join(DATASET_DIR, cls)
        existing = [f for f in os.listdir(folder) if f.endswith(".csv")]
        idx = len(existing) + 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{idx:04d}_{timestamp}.csv"
        filepath = os.path.join(folder, filename)

        header = ["timestamp"]
        for s in range(1, self.num_sensors + 1):
            suffix = str(s) if self.num_sensors > 1 else ""
            header += [f"ax{suffix}", f"ay{suffix}", f"az{suffix}",
                       f"gx{suffix}", f"gy{suffix}", f"gz{suffix}"]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)

    def _update_sample_counts(self):
        for cls in CLASSES:
            folder = os.path.join(DATASET_DIR, cls)
            count = len([f for f in os.listdir(folder) if f.endswith(".csv")]) if os.path.isdir(folder) else 0
            self.count_labels[cls].config(text=str(count))

    def _on_close(self):
        self.running = False
        if self.client and self.client.is_connected:
            future = asyncio.run_coroutine_threadsafe(self.client.disconnect(), self.loop)
            try:
                future.result(timeout=3)
            except Exception:
                pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.root.destroy()


def main():
    root = tk.Tk()
    DatasetRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
