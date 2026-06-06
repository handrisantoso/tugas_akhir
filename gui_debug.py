"""Tab 3 — Live Debug GUI (Dual Mode: PC + Embedded Inference)"""
import customtkinter as ctk
import time
import collections
import threading
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
        self._update_pending = False

        # PC inference engine (lazy loaded)
        self._pc_engine = None
        self._pc_inference_enabled = ctk.BooleanVar(value=False)
        self._pc_model_var = ctk.StringVar(value="rf")
        self._pc_pred = "idle"
        self._pc_conf = 0.0

        self._build_ui()

    def _build_ui(self):
        # Main layout: left (waveform + bars) / right (controls + log)
        left = ctk.CTkFrame(self, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=0)
        right = ctk.CTkFrame(self, corner_radius=12, width=340)
        right.pack(side="right", fill="both", padx=(6, 0), pady=0)
        right.pack_propagate(False)

        # ── Prediction Display (top of left) ──
        pred_frame = ctk.CTkFrame(left, corner_radius=8, fg_color="#16213e")
        pred_frame.pack(fill="x", padx=10, pady=(10, 5))

        # ESP32 prediction row
        esp_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
        esp_row.pack(fill="x", padx=15, pady=(10, 2))
        ctk.CTkLabel(esp_row, text="ESP32:", font=("Inter", 11), text_color="#6B7280",
                     width=55).pack(side="left")
        self.pred_label = ctk.CTkLabel(esp_row, text="IDLE", font=("Inter", 28, "bold"),
                                        text_color="#22C55E")
        self.pred_label.pack(side="left")
        self.conf_label = ctk.CTkLabel(esp_row, text="0.0%", font=("Inter", 20),
                                        text_color="#9CA3AF")
        self.conf_label.pack(side="left", padx=16)
        self.lat_label = ctk.CTkLabel(esp_row, text="⏱ 0ms", font=("Inter", 12),
                                       text_color="#6B7280")
        self.lat_label.pack(side="right")

        # PC prediction row
        pc_row = ctk.CTkFrame(pred_frame, fg_color="transparent")
        pc_row.pack(fill="x", padx=15, pady=(2, 10))
        ctk.CTkLabel(pc_row, text="PC:", font=("Inter", 11), text_color="#A78BFA",
                     width=55).pack(side="left")
        self.pc_pred_label = ctk.CTkLabel(pc_row, text="—", font=("Inter", 28, "bold"),
                                            text_color="#A78BFA")
        self.pc_pred_label.pack(side="left")
        self.pc_conf_label = ctk.CTkLabel(pc_row, text="", font=("Inter", 20),
                                            text_color="#9CA3AF")
        self.pc_conf_label.pack(side="left", padx=16)
        self.pc_status_label = ctk.CTkLabel(pc_row, text="Not loaded", font=("Inter", 11),
                                              text_color="#6B7280")
        self.pc_status_label.pack(side="right")

        # ── Live IMU Waveform ──
        ctk.CTkLabel(left, text="📈 Live IMU Stream", font=("Inter", 13, "bold")).pack(
            anchor="w", padx=12, pady=(5, 2))
        self.wave_fig = Figure(figsize=(7, 2.5), dpi=100, facecolor="#1a1a2e")
        self.wave_ax = self.wave_fig.add_subplot(111)
        self.wave_ax.set_facecolor("#16213e")
        self.wave_ax.tick_params(colors="#9CA3AF", labelsize=7)
        for s in self.wave_ax.spines.values(): s.set_color("#374151")
        self.wave_fig.tight_layout(pad=1)
        self.wave_canvas = FigureCanvasTkAgg(self.wave_fig, left)
        self.wave_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 5))

        # ── Confidence Bars ──
        ctk.CTkLabel(left, text="📊 Confidence", font=("Inter", 13, "bold")).pack(
            anchor="w", padx=12, pady=(5, 2))
        bars_frame = ctk.CTkFrame(left, corner_radius=8)
        bars_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.conf_bars = {}
        self.conf_labels = {}
        for name in GESTURE_NAMES:
            row = ctk.CTkFrame(bars_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            ctk.CTkLabel(row, text=name, font=("Inter", 10), width=85, anchor="w").pack(side="left")
            bar = ctk.CTkProgressBar(row, height=16, progress_color=GESTURE_COLORS[name],
                                      corner_radius=4)
            bar.set(0)
            bar.pack(side="left", expand=True, fill="x", padx=4)
            lbl = ctk.CTkLabel(row, text="0%", font=("Inter", 10, "bold"), width=40)
            lbl.pack(side="left")
            self.conf_bars[name] = bar
            self.conf_labels[name] = lbl

        # ── Right Side: Controls ──
        ctk.CTkLabel(right, text="⚙️ Controls", font=("Inter", 14, "bold")).pack(
            anchor="w", padx=12, pady=(10, 5))

        # Start/Stop streaming
        self.stream_btn = ctk.CTkButton(right, text="▶ Start Live", fg_color="#22C55E",
                                          hover_color="#16A34A", command=self.toggle_stream)
        self.stream_btn.pack(fill="x", padx=12, pady=4)

        # Block HID toggle
        ctk.CTkSwitch(right, text="🚫 Block HID Output", variable=self.block_hid,
                       font=("Inter", 12), command=self._on_block_hid).pack(
            anchor="w", padx=12, pady=6)

        # Threshold slider
        ctk.CTkLabel(right, text="Confidence Threshold", font=("Inter", 12)).pack(
            anchor="w", padx=12, pady=(10, 0))
        self.thresh_slider = ctk.CTkSlider(right, from_=0.50, to=0.99,
                                            variable=self.threshold_var,
                                            command=self._on_threshold_change,
                                            number_of_steps=49)
        self.thresh_slider.pack(fill="x", padx=12, pady=2)
        self.thresh_label = ctk.CTkLabel(right, text=f"{DEFAULT_CONFIDENCE_THRESHOLD:.2f}",
                                          font=("Inter", 16, "bold"), text_color="#F59E0B")
        self.thresh_label.pack(padx=12)

        # Mode selector
        ctk.CTkLabel(right, text="Firmware Mode", font=("Inter", 12)).pack(
            anchor="w", padx=12, pady=(10, 2))
        mode_frame = ctk.CTkFrame(right, fg_color="transparent")
        mode_frame.pack(fill="x", padx=12)
        for cmd, label in [("S", "Stream"), ("I", "Infer"), ("B", "Both")]:
            ctk.CTkButton(mode_frame, text=label, width=70, fg_color="#374151",
                           hover_color="#4B5563",
                           command=lambda c=cmd: self._send_mode(c)).pack(
                side="left", padx=2, expand=True, fill="x")

        # ── PC Inference Section ──
        pc_frame = ctk.CTkFrame(right, corner_radius=8, fg_color="#1e1b4b")
        pc_frame.pack(fill="x", padx=10, pady=(12, 4))
        ctk.CTkLabel(pc_frame, text="💻 PC Inference", font=("Inter", 14, "bold"),
                     text_color="#A78BFA").pack(anchor="w", padx=12, pady=(8, 4))

        # Model type row
        pc_model_row = ctk.CTkFrame(pc_frame, fg_color="transparent")
        pc_model_row.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(pc_model_row, text="Model:", font=("Inter", 11), width=46).pack(side="left")
        self.pc_model_seg = ctk.CTkSegmentedButton(pc_model_row,
                                                     values=["mlp", "cnn1d", "rf"],
                                                     variable=self._pc_model_var,
                                                     command=self._on_model_type_change)
        self.pc_model_seg.pack(side="left", padx=8)

        # Source row: Standard vs LOSO fold
        src_row = ctk.CTkFrame(pc_frame, fg_color="transparent")
        src_row.pack(fill="x", padx=12, pady=2)
        ctk.CTkLabel(src_row, text="Source:", font=("Inter", 11), width=46).pack(side="left")
        self._pc_source_var = ctk.StringVar(value="standard")
        self.pc_src_seg = ctk.CTkSegmentedButton(
            src_row, values=["standard", "LOSO fold"],
            variable=self._pc_source_var, command=self._on_source_change, width=180)
        self.pc_src_seg.pack(side="left", padx=8)

        # LOSO fold selector (hidden until LOSO source chosen)
        self._fold_row = ctk.CTkFrame(pc_frame, fg_color="transparent")
        ctk.CTkLabel(self._fold_row, text="Fold:", font=("Inter", 11), width=46).pack(side="left")
        self._pc_fold_var = ctk.StringVar(value="0")
        self.pc_fold_menu = ctk.CTkOptionMenu(
            self._fold_row, variable=self._pc_fold_var,
            values=["0"], width=80, command=lambda _: self._refresh_path_label())
        self.pc_fold_menu.pack(side="left", padx=8)
        ctk.CTkLabel(self._fold_row, text="(trained on all except this fold's subject)",
                     font=("Inter", 9), text_color="#6B7280").pack(side="left", padx=4)

        # Load + Enable row
        pc_btn_row = ctk.CTkFrame(pc_frame, fg_color="transparent")
        pc_btn_row.pack(fill="x", padx=12, pady=(4, 4))
        self.pc_load_btn = ctk.CTkButton(pc_btn_row, text="Load Model", width=100,
                                           fg_color="#8B5CF6", hover_color="#7C3AED",
                                           command=self._load_pc_model)
        self.pc_load_btn.pack(side="left", padx=(0, 4))
        ctk.CTkSwitch(pc_btn_row, text="Enable", variable=self._pc_inference_enabled,
                       font=("Inter", 11)).pack(side="left", padx=4)

        # Model path info label
        self.pc_path_label = ctk.CTkLabel(
            pc_frame, text=self._model_path_preview(),
            font=("Consolas", 9), text_color="#6B7280",
            anchor="w", wraplength=290, justify="left",
        )
        self.pc_path_label.pack(fill="x", padx=12, pady=(0, 8))

        # ── HID Fire Log ──
        ctk.CTkLabel(right, text="🔥 HID Fire Log", font=("Inter", 13, "bold")).pack(
            anchor="w", padx=12, pady=(12, 2))
        self.hid_log = ctk.CTkTextbox(right, font=("Consolas", 9), fg_color="#0f0f23",
                                        text_color="#F59E0B", corner_radius=8)
        self.hid_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def toggle_stream(self):
        if self._running:
            self._running = False
            self.stream_btn.configure(text="▶ Start Live", fg_color="#22C55E",
                                       hover_color="#16A34A")
        else:
            if not self.serial.is_connected:
                self._log_hid("[ERROR] Not connected! Use Recorder tab to connect first.")
                return
            self._running = True
            self.stream_btn.configure(text="⏹ Stop", fg_color="#EF4444",
                                       hover_color="#DC2626")
            self.serial.on_imu_data = self._on_imu
            self.serial.on_prediction = self._on_prediction
            self.serial.send_command("B")  # Both stream + infer
            self._update_loop()

    def _on_imu(self, ax, ay, az, gx, gy, gz):
        self.imu_buffer.append([ax, ay, az, gx, gy, gz])

        # PC inference: feed sample to engine
        if self._pc_inference_enabled.get() and self._pc_engine and self._pc_engine.is_loaded:
            result = self._pc_engine.feed_sample(ax, ay, az, gx, gy, gz)
            if result:
                self._pc_pred = result["gesture"]
                self._pc_conf = result["confidence"]
                # Update confidence bars from PC model
                for name in GESTURE_NAMES:
                    self.confidences[name] = result["all_probs"].get(name, 0.0)

    def _on_prediction(self, gesture, confidence, latency):
        self.current_pred = gesture
        self.current_conf = confidence
        self.current_latency = latency

        # Update confidence bars from ESP32 (only if PC inference is disabled)
        if not self._pc_inference_enabled.get():
            for name in GESTURE_NAMES:
                self.confidences[name] = confidence if name == gesture else max(
                    0, self.confidences[name] * 0.3)

        # Log HID fires (thread-safe via after)
        threshold = self.threshold_var.get()
        if confidence >= threshold and gesture != "idle":
            ts = time.strftime("%H:%M:%S")
            block = " [BLOCKED]" if self.block_hid.get() else ""
            entry = f"[{ts}] {gesture} ({confidence:.0%}){block}\n"
            # Schedule on main thread for thread safety
            self.after(0, lambda e=entry: self._append_hid_log(e))

    def _append_hid_log(self, entry):
        """Thread-safe HID log append (runs on main thread)."""
        self.hid_log.insert("end", entry)
        self.hid_log.see("end")

    def _log_hid(self, msg):
        """Log a message to the HID log."""
        ts = time.strftime("%H:%M:%S")
        self.hid_log.insert("end", f"[{ts}] {msg}\n")
        self.hid_log.see("end")

    def _update_loop(self):
        if not self._running:
            return

        # Skip if previous update is still rendering
        if self._update_pending:
            self.after(80, self._update_loop)
            return
        self._update_pending = True

        try:
            # Update ESP32 prediction display
            color = GESTURE_COLORS.get(self.current_pred, "#9CA3AF")
            self.pred_label.configure(text=self.current_pred.upper(), text_color=color)
            self.conf_label.configure(text=f"{self.current_conf:.1%}")
            self.lat_label.configure(text=f"⏱ {self.current_latency}ms")

            # Update PC prediction display
            if self._pc_inference_enabled.get() and self._pc_engine and self._pc_engine.is_loaded:
                pc_color = GESTURE_COLORS.get(self._pc_pred, "#A78BFA")
                self.pc_pred_label.configure(text=self._pc_pred.upper(), text_color=pc_color)
                self.pc_conf_label.configure(text=f"{self._pc_conf:.1%}")
            else:
                self.pc_pred_label.configure(text="—", text_color="#A78BFA")
                self.pc_conf_label.configure(text="")

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
        except Exception:
            pass
        finally:
            self._update_pending = False

        self.after(80, self._update_loop)

    def _model_path_preview(self, model_type=None):
        """Return path/size/date string for the model that would be / is loaded."""
        import os, time as _time
        from config import MODELS_DIR
        mt = model_type or self._pc_model_var.get()
        ext = ".keras" if mt in ("mlp", "cnn1d") else ".joblib"
        if self._pc_source_var.get() == "LOSO fold":
            fold = self._pc_fold_var.get()
            if fold == "—":
                return f"No LOSO {mt.upper()} models found — run LOSO training first"
            model_dir = os.path.join(MODELS_DIR, f"loso_{mt}_fold{fold}")
            tag = f"LOSO fold {fold}"
        else:
            model_dir = MODELS_DIR
            tag = "Standard"
        path = os.path.join(model_dir, f"{mt}_model{ext}")
        if not os.path.isfile(path):
            return f"[{tag}] {path}\n(not trained yet)"
        mtime = os.path.getmtime(path)
        size  = os.path.getsize(path) / 1024
        date  = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(mtime))
        return f"[{tag}] {path}\nSize: {size:.1f} KB  |  Saved: {date}"

    def _load_pc_model(self):
        """Load a trained model for PC-side inference."""
        model_type = self._pc_model_var.get()
        source = self._pc_source_var.get()
        fold_str = self._pc_fold_var.get()
        loso_fold = int(fold_str) if source == "LOSO fold" and fold_str.isdigit() else None
        label = f"LOSO fold {loso_fold}" if loso_fold is not None else "Standard"
        self.pc_status_label.configure(text=f"Loading {model_type.upper()} ({label})...",
                                        text_color="#F59E0B")
        self.pc_path_label.configure(text=self._model_path_preview(), text_color="#F59E0B")
        self.pc_load_btn.configure(state="disabled")

        def _load():
            try:
                from inference import PCInferenceEngine
                self._pc_engine = PCInferenceEngine()
                self._pc_engine.load_model(model_type, loso_fold=loso_fold)
                self.after(0, lambda: self._on_pc_model_loaded(model_type, label))
            except Exception as e:
                self.after(0, lambda: self._on_pc_model_error(str(e)))

        threading.Thread(target=_load, daemon=True).start()

    def _on_model_type_change(self, val):
        self._refresh_fold_options(val)
        self._refresh_path_label()

    def _on_source_change(self, val):
        if val == "LOSO fold":
            self._fold_row.pack(fill="x", padx=12, pady=(0, 2))
            self._refresh_fold_options(self._pc_model_var.get())
        else:
            self._fold_row.pack_forget()
        self._refresh_path_label()

    def _refresh_fold_options(self, model_type=None):
        from inference import PCInferenceEngine
        mt = model_type or self._pc_model_var.get()
        folds = PCInferenceEngine.get_available_loso_folds(mt)
        if folds:
            self.pc_fold_menu.configure(values=[str(f) for f in folds])
            if self._pc_fold_var.get() not in [str(f) for f in folds]:
                self._pc_fold_var.set(str(folds[0]))
        else:
            self.pc_fold_menu.configure(values=["—"])
            self._pc_fold_var.set("—")

    def _refresh_path_label(self):
        self.pc_path_label.configure(
            text=self._model_path_preview(), text_color="#6B7280")

    def _on_pc_model_loaded(self, model_type, label="Standard"):
        self.pc_status_label.configure(
            text=f"✅ {model_type.upper()} ({label})", text_color="#22C55E")
        self.pc_load_btn.configure(state="normal")
        self._pc_inference_enabled.set(True)
        # Refresh path label with confirmed loaded file info
        if self._pc_engine:
            info = self._pc_engine.get_model_info()
            if info:
                self.pc_path_label.configure(
                    text=f"Path: {info['path']}\nSize: {info['size']}  |  Saved: {info['date']}",
                    text_color="#22C55E",
                )
        self._log_hid(f"PC model loaded: {model_type.upper()} — {self._model_path_preview(model_type)}")

    def _on_pc_model_error(self, error):
        self.pc_status_label.configure(text=f"❌ {error[:40]}", text_color="#EF4444")
        self.pc_load_btn.configure(state="normal")

    def _on_threshold_change(self, val):
        self.thresh_label.configure(text=f"{val:.2f}")
        if self.serial.is_connected:
            self.serial.send_command(f"T{val:.2f}")

    def _on_block_hid(self):
        pass  # Visual only — ESP32 doesn't know about blocking

    def _send_mode(self, cmd):
        if self.serial.is_connected:
            self.serial.send_command(cmd)
