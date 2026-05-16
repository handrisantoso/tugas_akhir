"""Tab 1 — Dataset Recorder GUI"""
import os
import customtkinter as ctk
import time
import threading
import numpy as np
import collections
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from config import (
    GESTURE_NAMES, GESTURE_COLORS, WINDOW_SIZE, NUM_AXES,
    AXIS_NAMES, AXIS_COLORS, DEFAULT_CSV, SAMPLE_RATE_HZ
)
from recorder import SerialManager, GestureRecorder


class RecorderTab(ctk.CTkFrame):
    def __init__(self, parent, serial_mgr: SerialManager):
        super().__init__(parent, fg_color="transparent")
        self.serial = serial_mgr
        self.recorder = GestureRecorder(serial_mgr)

        # Keyboard recording: track which key is held
        self._held_key = None  # "1".."5" or None
        self._key_to_gesture = {str(i+1): i for i in range(len(GESTURE_NAMES))}

        # Live waveform — thread-safe rolling buffer
        self._live_buf = collections.deque(maxlen=WINDOW_SIZE)
        self._buf_lock = threading.Lock()
        self._live_running = False
        self._update_pending = False  # prevent overlapping plot updates

        self._build_ui()

    def _build_ui(self):
        left = ctk.CTkFrame(self, corner_radius=12)
        left.pack(side="left", fill="both", padx=(0,6), pady=0, expand=False)
        right = ctk.CTkFrame(self, corner_radius=12)
        right.pack(side="left", fill="both", padx=(6,0), pady=0, expand=True)

        # ── Connection ──
        conn_frame = ctk.CTkFrame(left, corner_radius=8)
        conn_frame.pack(fill="x", padx=10, pady=(10,5))
        ctk.CTkLabel(conn_frame, text="⚡ Connection", font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,2))

        port_row = ctk.CTkFrame(conn_frame, fg_color="transparent")
        port_row.pack(fill="x", padx=10, pady=4)
        self.port_var = ctk.StringVar()
        self.port_menu = ctk.CTkComboBox(port_row, variable=self.port_var, width=180, state="readonly")
        self.port_menu.pack(side="left", padx=(0,4))
        ctk.CTkButton(port_row, text="↻", width=32, command=self.refresh_ports).pack(side="left", padx=2)

        btn_row = ctk.CTkFrame(conn_frame, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(2,8))
        self.connect_btn = ctk.CTkButton(btn_row, text="Connect", width=100, command=self.toggle_connection,
                                          fg_color="#22C55E", hover_color="#16A34A")
        self.connect_btn.pack(side="left", padx=(0,6))
        self.status_label = ctk.CTkLabel(btn_row, text="● Disconnected", text_color="#EF4444", font=("Inter", 12))
        self.status_label.pack(side="left")

        # ── Keyboard Gesture Selector ──
        gest_frame = ctk.CTkFrame(left, corner_radius=8)
        gest_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(gest_frame, text="🎯 Hold [1-5] to record",
                     font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,4))

        self.gesture_buttons = {}
        for i, name in enumerate(GESTURE_NAMES):
            key = str(i + 1)
            color = GESTURE_COLORS[name]
            row = ctk.CTkFrame(gest_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)

            key_lbl = ctk.CTkLabel(row, text=key, font=("Inter", 14, "bold"),
                                   width=28, height=28, corner_radius=6,
                                   fg_color="#374151", text_color="#D1D5DB")
            key_lbl.pack(side="left", padx=(0, 6))

            name_lbl = ctk.CTkLabel(row, text=f"  {name}", font=("Inter", 13), text_color="#6B7280")
            name_lbl.pack(side="left")

            self.gesture_buttons[i] = (key_lbl, name_lbl)

        self._update_gesture_ui(0)

        # ── Recording Status ──
        rec_frame = ctk.CTkFrame(left, corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(rec_frame, text="🔴 Recording", font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,4))

        self.rec_indicator = ctk.CTkLabel(rec_frame, text="", font=("Inter", 48, "bold"),
                                          text_color="#374151")
        self.rec_indicator.pack(pady=4)

        self.rec_gesture_lbl = ctk.CTkLabel(rec_frame, text="", font=("Inter", 11),
                                            text_color="#9CA3AF")
        self.rec_gesture_lbl.pack(padx=10, pady=(0,8))

        self.capture_info = ctk.CTkLabel(rec_frame, text="", font=("Inter", 11), text_color="#9CA3AF")
        self.capture_info.pack(padx=10, pady=(0,8))

        # ── Sample Counters ──
        count_frame = ctk.CTkFrame(left, corner_radius=8)
        count_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(count_frame, text="📊 Samples", font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,4))
        self.progress_bars = {}
        self.count_labels = {}
        for name in GESTURE_NAMES:
            row = ctk.CTkFrame(count_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=1)
            ctk.CTkLabel(row, text=name, font=("Inter", 11), width=90, anchor="w").pack(side="left")
            pb = ctk.CTkProgressBar(row, width=120, height=12, progress_color=GESTURE_COLORS[name])
            pb.set(0)
            pb.pack(side="left", padx=4, expand=True, fill="x")
            cl = ctk.CTkLabel(row, text="0/150", font=("Inter", 11, "bold"), width=50)
            cl.pack(side="left")
            self.progress_bars[name] = pb
            self.count_labels[name] = cl

        # ── Save Section ──
        save_frame = ctk.CTkFrame(left, corner_radius=8)
        save_frame.pack(fill="x", padx=10, pady=(5,10))
        save_row = ctk.CTkFrame(save_frame, fg_color="transparent")
        save_row.pack(fill="x", padx=10, pady=8)
        self.csv_path_var = ctk.StringVar(value=DEFAULT_CSV)
        ctk.CTkEntry(save_row, textvariable=self.csv_path_var, width=160, font=("Inter", 10)).pack(
            side="left", expand=True, fill="x", padx=(0,4))
        ctk.CTkButton(save_row, text="📁", width=32, command=self.browse_csv).pack(side="left", padx=2)
        ctk.CTkButton(save_row, text="💾 Save CSV", width=90, command=self.save_csv,
                      fg_color="#3B82F6", hover_color="#2563EB").pack(side="left", padx=(4,0))

        # ── RIGHT SIDE: Live Waveform + Log ──
        ctk.CTkLabel(right, text="📈 Live IMU", font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(10,4))

        self.live_fig = Figure(figsize=(7, 3.2), dpi=100, facecolor="#1a1a2e")
        self.live_ax = self.live_fig.add_subplot(111)
        self.live_ax.set_facecolor("#16213e")
        self.live_ax.set_xlabel("Time (s)", color="#9CA3AF", fontsize=9)
        self.live_ax.set_ylabel("Value", color="#9CA3AF", fontsize=9)
        self.live_ax.tick_params(colors="#9CA3AF", labelsize=7)
        for spine in self.live_ax.spines.values():
            spine.set_color("#374151")
        self.live_fig.tight_layout(pad=1.2)
        self.live_canvas = FigureCanvasTkAgg(self.live_fig, right)
        self.live_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,5))

        # Session log
        ctk.CTkLabel(right, text="📋 Session Log", font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(5,2))
        self.log_text = ctk.CTkTextbox(right, height=100, font=("Consolas", 10), fg_color="#0f0f23",
                                        text_color="#22C55E", corner_radius=8)
        self.log_text.pack(fill="both", padx=10, pady=(0,5))

        # Browse Recordings button
        ctk.CTkButton(right, text="🔍 Browse Recordings", width=180, command=self._browse_recordings,
                       fg_color="#4B5563", hover_color="#374151").pack(padx=10, pady=(0,10))

        self.refresh_ports()
        self.log("Session started. Hold 1-5 to record gestures.")
        self._update_live()

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    def refresh_ports(self):
        ports = SerialManager.list_ports()
        port_names = [p[0] for p in ports]
        self.port_menu.configure(values=port_names if port_names else ["No ports found"])
        if port_names:
            self.port_var.set(port_names[0])

    def _update_gesture_ui(self, active_idx):
        for i, name in enumerate(GESTURE_NAMES):
            key_lbl, name_lbl = self.gesture_buttons[i]
            if i == active_idx:
                color = GESTURE_COLORS[name]
                key_lbl.configure(fg_color=color, text_color="#ffffff")
                name_lbl.configure(text_color=color)
            else:
                key_lbl.configure(fg_color="#374151", text_color="#D1D5DB")
                name_lbl.configure(text_color="#6B7280")

    # ── Keyboard recording ──────────────────────────────────────

    def on_key_press(self, key):
        """Called by parent GUI when a key is pressed."""
        if key not in self._key_to_gesture:
            return
        if self._held_key is not None:
            return  # already recording
        if not self.serial.is_connected:
            self.log("ERROR: Not connected to ESP32!")
            return

        self._held_key = key
        gesture_id = self._key_to_gesture[key]
        self._update_gesture_ui(gesture_id)
        self._live_buf.clear()
        self.recorder.start_recording(gesture_id)
        name = GESTURE_NAMES[gesture_id]
        color = GESTURE_COLORS[name]
        self.rec_indicator.configure(text="●REC", text_color=color)
        self.rec_gesture_lbl.configure(text=f"{name}", text_color=color)
        self.log(f"▶ Recording {name}...")

    def on_key_release(self, key):
        """Called by parent GUI when a key is released."""
        if key != self._held_key:
            return  # not the active recording key
        if self._held_key is None:
            return

        self._held_key = None
        result = self.recorder.stop_recording()
        self.rec_indicator.configure(text="", text_color="#374151")
        self.rec_gesture_lbl.configure(text="", text_color="#9CA3AF")
        self._update_gesture_ui(0)

        if result is None:
            self.log("Recording failed — no samples captured!")
            return

        flat_arr, label, raw_count = result
        name = GESTURE_NAMES[label]
        self.log(f"■ Saved {raw_count} samples for '{name}'.")
        self.capture_info.configure(text=f"Last: {raw_count} samples → {name}")
        self._update_counters()
        self._plot_saved(flat_arr)

    # ── Serial / UI ─────────────────────────────────────────────

    def toggle_connection(self):
        if self.serial.is_connected:
            self.serial.on_imu_data = None
            self.serial.disconnect()
            self.connect_btn.configure(text="Connect", fg_color="#22C55E", hover_color="#16A34A")
            self.status_label.configure(text="● Disconnected", text_color="#EF4444")
            self._live_running = False
            self.log("Disconnected.")
        else:
            try:
                port = self.port_var.get()
                self.serial.connect(port)
                self.serial.on_imu_data = self._on_live_sample
                self.connect_btn.configure(text="Disconnect", fg_color="#EF4444", hover_color="#DC2626")
                self.status_label.configure(text="● Connected", text_color="#22C55E")
                self._live_running = True
                self.log(f"Connected to {port}. Hold 1-5 to record.")
            except Exception as e:
                self.log(f"ERROR: {e}")

    def _on_live_sample(self, ax, ay, az, gx, gy, gz):
        with self._buf_lock:
            self._live_buf.append([ax, ay, az, gx, gy, gz])

    def _update_live(self):
        # Skip if previous plot is still rendering
        if self._update_pending:
            self.after(60, self._update_live)
            return
        self._update_pending = True

        try:
            with self._buf_lock:
                n = len(self._live_buf)
                if n > 5:
                    data = np.array(list(self._live_buf), dtype=np.float64)
                else:
                    data = None

            if data is not None:
                t = np.arange(n) / SAMPLE_RATE_HZ
                self.live_ax.clear()
                self.live_ax.set_facecolor("#16213e")
                for i in range(NUM_AXES):
                    self.live_ax.plot(t, data[:, i], color=AXIS_COLORS[i],
                                      linewidth=0.7, alpha=0.8, label=AXIS_NAMES[i])
                self.live_ax.legend(loc="upper right", fontsize=7,
                                    facecolor="#1a1a2e", edgecolor="#374151",
                                    labelcolor="#D1D5DB")
                self.live_ax.set_xlabel("Time (s)", color="#9CA3AF", fontsize=8)
                self.live_ax.set_ylabel("Value", color="#9CA3AF", fontsize=8)
                self.live_ax.tick_params(colors="#9CA3AF", labelsize=7)
                for spine in self.live_ax.spines.values():
                    spine.set_color("#374151")
                self.live_canvas.draw_idle()
        except Exception:
            pass
        finally:
            self._update_pending = False
            self.after(60, self._update_live)

    def _update_counters(self):
        for name in GESTURE_NAMES:
            count = self.recorder.sample_counts.get(name, 0)
            self.count_labels[name].configure(text=f"{count}/150")
            self.progress_bars[name].set(min(count / 150.0, 1.0))

    def _plot_saved(self, flat_arr):
        n = len(flat_arr) // NUM_AXES
        t = np.arange(WINDOW_SIZE) / SAMPLE_RATE_HZ
        data = flat_arr.reshape(WINDOW_SIZE, NUM_AXES)

        self.live_ax.clear()
        self.live_ax.set_facecolor("#16213e")

        n_time = n / SAMPLE_RATE_HZ
        y_min, y_max = self.live_ax.get_ylim()
        if y_min == y_max:
            y_min, y_max = -20, 20
        rect = Rectangle((0, y_min), n_time, y_max - y_min,
                          linewidth=0, facecolor="#22C55E", alpha=0.12, edgecolor="#22C55E", zorder=0)
        self.live_ax.add_patch(rect)

        for i in range(NUM_AXES):
            self.live_ax.plot(t, data[:, i], color=AXIS_COLORS[i],
                              linewidth=0.8, alpha=0.85, label=AXIS_NAMES[i])

        self.live_ax.axvline(x=0, color="#22C55E", linewidth=1.5, linestyle="--", alpha=0.7, label="REC start")
        if n < WINDOW_SIZE:
            self.live_ax.axvline(x=n_time, color="#EF4444", linewidth=1.5, linestyle="--", alpha=0.7, label="REC end")

        self.live_ax.legend(loc="upper right", fontsize=7,
                            facecolor="#1a1a2e", edgecolor="#374151",
                            labelcolor="#D1D5DB")
        self.live_ax.set_xlabel("Time (s)", color="#9CA3AF", fontsize=8)
        self.live_ax.set_ylabel("Value", color="#9CA3AF", fontsize=8)
        self.live_ax.tick_params(colors="#9CA3AF", labelsize=7)
        for spine in self.live_ax.spines.values():
            spine.set_color("#374151")
        self.live_fig.tight_layout(pad=1.2)
        self.live_canvas.draw_idle()

    def browse_csv(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")],
                                             initialfile="gesture_data.csv")
        if path:
            self.csv_path_var.set(path)

    def save_csv(self):
        if not self.recorder.session_samples:
            self.log("No samples to save!")
            return
        path, count = self.recorder.save_to_csv(self.csv_path_var.get())
        self.log(f"Saved {count} windows to {path}")
        self.recorder.clear_session()
        self._update_counters()

    def _browse_recordings(self):
        """Open a window to browse all saved recordings by sample_id."""
        import pandas as pd
        csv_path = self.csv_path_var.get()
        if not os.path.isfile(csv_path):
            self.log("No CSV file found!")
            return

        df, _ = GestureRecorder.load_csv(csv_path)
        if df is None:
            self.log("CSV file is empty!")
            return

        sample_ids = df["sample_id"].unique()
        # Group by label
        by_label = {}
        for sid in sample_ids:
            label = sid.rsplit("_", 2)[0]  # e.g. "flick_up" from "flick_up_1778848424_0"
            by_label.setdefault(label, []).append(sid)

        # Build a simple selection window
        win = ctk.CTkToplevel(self)
        win.title("Browse Recordings")
        win.geometry("700x500")
        win.configure(fg_color="#1a1a2e")

        ctk.CTkLabel(win, text="Select a recording to preview:",
                     font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(10,4))

        # Left: list of recordings
        list_frame = ctk.CTkScrollableFrame(win, width=220, fg_color="transparent")
        list_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        # Right: preview plot
        preview_frame = ctk.CTkFrame(win, corner_radius=8)
        preview_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        fig = Figure(figsize=(5.5, 3.5), dpi=90, facecolor="#1a1a2e")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#16213e")
        canvas = FigureCanvasTkAgg(fig, preview_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        self._preview_ax = ax
        self._preview_fig = fig
        self._preview_canvas = canvas
        self._preview_win = win

        def plot_sample(sample_id):
            subset = df[df["sample_id"] == sample_id].copy()
            n = len(subset)
            t = np.arange(n) / SAMPLE_RATE_HZ
            ax.clear()
            ax.set_facecolor("#16213e")
            labels = ["ax", "ay", "az", "gx", "gy", "gz"]
            for i in range(NUM_AXES):
                ax.plot(t, subset[labels[i]].values, color=AXIS_COLORS[i],
                        linewidth=0.9, alpha=0.85, label=labels[i])
            ax.legend(loc="upper right", fontsize=7,
                     facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#D1D5DB")
            ax.set_xlabel("Time (s)", color="#9CA3AF", fontsize=8)
            ax.tick_params(colors="#9CA3AF", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#374151")
            fig.tight_layout()
            canvas.draw_idle()

        def on_select(sample_id):
            label = sample_id.rsplit("_", 2)[0]
            n_samples = len(df[df["sample_id"] == sample_id])
            info_lbl.configure(text=f"{sample_id}\n{n_samples} samples · {label}")
            plot_sample(sample_id)

        info_lbl = ctk.CTkLabel(preview_frame, text="", font=("Inter", 10),
                                text_color="#9CA3AF")
        info_lbl.pack(pady=(4, 0))

        selection = {}

        def select_row(label, sid, row_widget):
            # Deselect others
            for k, v in selection.items():
                v[1].configure(fg_color="#16213e")
            selection.clear()
            selection[sid] = (label, row_widget)
            row_widget.configure(fg_color="#4B5563")
            on_select(sid)

        for label in GESTURE_NAMES:
            if label not in by_label:
                continue
            ctk.CTkLabel(list_frame, text=f"  {label.upper()}", font=("Inter", 11, "bold"),
                         text_color=GESTURE_COLORS.get(label, "#D1D5DB")).pack(anchor="w", padx=4, pady=(8, 2))
            for sid in sorted(by_label[label]):
                row = ctk.CTkButton(list_frame, text=f"  {sid}", font=("Consolas", 9),
                                    fg_color="#16213e", hover_color="#374151",
                                    anchor="w", height=22,
                                    command=lambda s=sid, r=None: None)
                row.pack(fill="x", padx=4, pady=1)
                row.configure(command=lambda s=sid, r=row, lbl=label: select_row(lbl, s, r))