"""Tab 3 — Live Debug GUI"""
import customtkinter as ctk
import time
import collections
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import (
    GESTURE_NAMES, GESTURE_COLORS, WINDOW_SIZE, NUM_AXES,
    AXIS_NAMES, AXIS_COLORS, DEFAULT_CONFIDENCE_THRESHOLD
)
from recorder import SerialManager


class DebugTab(ctk.CTkFrame):
    def __init__(self, parent, serial_mgr: SerialManager):
        super().__init__(parent, fg_color="transparent")
        self.serial = serial_mgr
        self.imu_buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.confidences = {name: 0.0 for name in GESTURE_NAMES}
        self.current_pred = "idle"
        self.current_conf = 0.0
        self.current_latency = 0
        self.hid_log_entries = []
        self.block_hid = ctk.BooleanVar(value=False)
        self.threshold_var = ctk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self._running = False
        self._build_ui()

    def _build_ui(self):
        # Main layout: left (waveform + bars) / right (controls + log)
        left = ctk.CTkFrame(self, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=0)
        right = ctk.CTkFrame(self, corner_radius=12, width=320)
        right.pack(side="right", fill="both", padx=(6, 0), pady=0)
        right.pack_propagate(False)

        # ── Prediction Display (top of left) ──
        pred_frame = ctk.CTkFrame(left, corner_radius=8, fg_color="#16213e")
        pred_frame.pack(fill="x", padx=10, pady=(10, 5))
        pred_inner = ctk.CTkFrame(pred_frame, fg_color="transparent")
        pred_inner.pack(fill="x", padx=15, pady=10)
        self.pred_label = ctk.CTkLabel(pred_inner, text="IDLE", font=("Inter", 36, "bold"), text_color="#22C55E")
        self.pred_label.pack(side="left")
        self.conf_label = ctk.CTkLabel(pred_inner, text="0.0%", font=("Inter", 24), text_color="#9CA3AF")
        self.conf_label.pack(side="left", padx=20)
        self.lat_label = ctk.CTkLabel(pred_inner, text="⏱ 0ms", font=("Inter", 14), text_color="#6B7280")
        self.lat_label.pack(side="right")

        # ── Live IMU Waveform ──
        ctk.CTkLabel(left, text="📈 Live IMU Stream", font=("Inter", 13, "bold")).pack(anchor="w", padx=12, pady=(5, 2))
        self.wave_fig = Figure(figsize=(7, 2.5), dpi=100, facecolor="#1a1a2e")
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_ax.set_facecolor("#16213e")
        self.wave_ax.tick_params(colors="#9CA3AF", labelsize=7)
        for s in self.wave_ax.spines.values(): s.set_color("#374151")
        self.wave_fig.tight_layout(pad=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, left)
        self.wave_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # ── Confidence Bars ──
        ctk.CTkLabel(left, text="📊 Confidence", font=("Inter", 13, "bold")).pack(anchor="w", padx=12, pady=(5, 2))
        bars_frame = ctk.CTkFrame(left, corner_radius=8)
        bars_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.conf_bars = {}
        self.conf_labels = {}
        for name in GESTURE_NAMES:
            row = ctk.CTkFrame(bars_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row, text=name, font=("Inter", 10), width=85, anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(row, height=16, progress_color=GESTURE_COLORS[name], corner_radius=4)
            bar.set(0)
            bar.pack(side="left", expand=True, fill="x", padx=4)
            lbl = ctk.CTkLabel(row, text="0%", font=("Inter", 10, "bold"), width=40)
            lbl.pack(side="left")
            self.conf_bars[name] = bar
            self.conf_labels[name] = lbl

        # ── Right Side: Controls ──
        ctk.CTkLabel(right, text="⚙️ Controls", font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 5))

        # Start/Stop streaming
        self.stream_btn = ctk.CTkButton(right, text="▶ Start Live", fg_color="#22C55E",
                                          hover_color="#16A34A", command=self.toggle_stream)
        self.stream_btn.pack(fill="x", padx=12, pady=4)

        # Block HID toggle
        ctk.CTkSwitch(right, text="🚫 Block HID Output", variable=self.block_hid,
                       font=("Inter", 12), command=self._on_block_hid).pack(anchor="w", padx=12, pady=6)

        # Threshold slider
        ctk.CTkLabel(right, text="Confidence Threshold", font=("Inter", 12)).pack(anchor="w", padx=12, pady=(10, 0))
        self.thresh_slider = ctk.CTkSlider(right, from_=0.50, to=0.99, variable=self.threshold_var,
                                            command=self._on_threshold_change, number_of_steps=49)
        self.thresh_slider.pack(fill="x", padx=12, pady=2)
        self.thresh_label = ctk.CTkLabel(right, text=f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}",
                                          font=("Inter", 16, "bold"), text_color="#F59E0B")
        self.thresh_label.pack(padx=12)

        # Mode selector
        ctk.CTkLabel(right, text="Firmware Mode", font=("Inter", 12)).pack(anchor="w", padx=12, pady=(10, 2))
        mode_frame = ctk.CTkFrame(right, fg_color="transparent")
        mode_frame.pack(fill="x", padx=12)
        for cmd, label in [("S", "Stream"), ("I", "Infer"), ("B", "Both")]:
            ctk.CTkButton(mode_frame, text=label, width=70, fg_color="#374151", hover_color="#4B5563",
                           command=lambda c=cmd: self._send_mode(c)).pack(side="left", padx=2, expand=True, fill="x")

        # ── HID Fire Log ──
        ctk.CTkLabel(right, text="🔥 HID Fire Log", font=("Inter", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        self.hid_log = ctk.CTkTextbox(right, font=("Consolas", 9), fg_color="#0f0f23",
                                        text_color="#F59E0B", corner_radius=8)
        self.hid_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def toggle_stream(self):
        if self._running:
            self._running = False
            self.stream_btn.configure(text="▶ Start Live", fg_color="#22C55E", hover_color="#16A34A")
        else:
            if not self.serial.is_connected:
                return
            self._running = True
            self.stream_btn.configure(text="⏹ Stop", fg_color="#EF4444", hover_color="#DC2626")
            self.serial.on_imu_data = self._on_imu
            self.serial.on_prediction = self._on_prediction
            self.serial.send_command("B")  # Both stream + infer
            self._update_loop()

    def _on_imu(self, ax, ay, az, gx, gy, gz):
        self.imu_buffer.append([ax, ay, az, gx, gy, gz])

    def _on_prediction(self, gesture, confidence, latency):
        self.current_pred = gesture
        self.current_conf = confidence
        self.current_latency = latency

        # Update confidence dict
        for name in GESTURE_NAMES:
            self.confidences[name] = confidence if name == gesture else max(0, self.confidences[name] * 0.3)

        # Log HID fires
        threshold = self.threshold_var.get()
        if confidence >= threshold and gesture != "idle":
            ts = time.strftime("%H:%M:%S")
            block = " [BLOCKED]" if self.block_hid.get() else ""
            entry = f"[{ts}] {gesture} ({confidence:.0%}){block}\n"
            self.hid_log.insert("end", entry)
            self.hid_log.see("end")

    def _update_loop(self):
        if not self._running:
            return

        # Update prediction display
        color = GESTURE_COLORS.get(self.current_pred, "#9CA3AF")
        self.pred_label.configure(text=self.current_pred.upper(), text_color=color)
        self.conf_label.configure(text=f"{self.current_conf:.1%}")
        self.lat_label.configure(text=f"⏱ {self.current_latency}ms")

        # Update confidence bars
        threshold = self.threshold_var.get()
        for name in GESTURE_NAMES:
            val = self.confidences.get(name, 0)
            self.conf_bars[name].set(min(val, 1.0))
            self.conf_labels[name].configure(text=f"{val:.0%}")
            c = GESTURE_COLORS[name] if val >= threshold else "#4B5563"
            self.conf_bars[name].configure(progress_color=c)

        # Update waveform
        if len(self.imu_buffer) > 10:
            data = np.array(list(self.imu_buffer))
            self.wave_ax.clear()
            self.wave_ax.set_facecolor("#16213e")
            for i in range(min(NUM_AXES, data.shape[1])):
                self.wave_ax.plot(data[:, i], color=AXIS_COLORS[i], linewidth=0.7, alpha=0.8)
            self.wave_ax.tick_params(colors="#9CA3AF", labelsize=7)
            for s in self.wave_ax.spines.values(): s.set_color("#374151")
            self.wave_fig.tight_layout(pad=1)
            self.wave_canvas.draw_idle()

        self.after(80, self._update_loop)

    def _on_threshold_change(self, val):
        self.thresh_label.configure(text=f"{val:.2f}")
        if self.serial.is_connected:
            self.serial.send_command(f"T{val:.2f}")

    def _on_block_hid(self):
        pass  # Visual only — ESP32 doesn't know about blocking

    def _send_mode(self, cmd):
        if self.serial.is_connected:
            self.serial.send_command(cmd)
