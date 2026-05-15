"""Tab 1 — Dataset Recorder GUI"""
import customtkinter as ctk
import threading
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from config import (
    GESTURE_NAMES, GESTURE_COLORS, WINDOW_SIZE, NUM_AXES,
    AXIS_NAMES, AXIS_COLORS, DEFAULT_CSV
)
from recorder import SerialManager, GestureRecorder


class RecorderTab(ctk.CTkFrame):
    def __init__(self, parent, serial_mgr: SerialManager):
        super().__init__(parent, fg_color="transparent")
        self.serial = serial_mgr
        self.recorder = GestureRecorder(serial_mgr)
        self.selected_gesture = ctk.IntVar(value=0)
        self.countdown_active = False
        self._build_ui()

    def _build_ui(self):
        # Main layout: left panel + right panel
        left = ctk.CTkFrame(self, corner_radius=12)
        left.pack(side="left", fill="both", padx=(0,6), pady=0, expand=False)
        right = ctk.CTkFrame(self, corner_radius=12)
        right.pack(side="left", fill="both", padx=(6,0), pady=0, expand=True)

        # ── Connection Section ──
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

        # ── Gesture Selector ──
        gest_frame = ctk.CTkFrame(left, corner_radius=8)
        gest_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(gest_frame, text="🎯 Gesture", font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,4))
        for i, name in enumerate(GESTURE_NAMES):
            color = GESTURE_COLORS[name]
            rb = ctk.CTkRadioButton(gest_frame, text=f"  {name}", variable=self.selected_gesture,
                                     value=i, font=("Inter", 13), fg_color=color, hover_color=color)
            rb.pack(anchor="w", padx=20, pady=2)

        # ── Record Controls ──
        rec_frame = ctk.CTkFrame(left, corner_radius=8)
        rec_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(rec_frame, text="🔴 Record", font=("Inter", 14, "bold")).pack(anchor="w", padx=10, pady=(8,4))

        self.countdown_label = ctk.CTkLabel(rec_frame, text="", font=("Inter", 48, "bold"), text_color="#F59E0B")
        self.countdown_label.pack(pady=2)

        rec_btns = ctk.CTkFrame(rec_frame, fg_color="transparent")
        rec_btns.pack(fill="x", padx=10, pady=4)
        self.start_btn = ctk.CTkButton(rec_btns, text="▶ Start Record", fg_color="#EF4444",
                                        hover_color="#DC2626", command=self.start_recording)
        self.start_btn.pack(side="left", padx=(0,4), expand=True, fill="x")
        self.stop_btn = ctk.CTkButton(rec_btns, text="⏹ Stop", fg_color="#6B7280",
                                       hover_color="#4B5563", command=self.stop_recording, state="disabled")
        self.stop_btn.pack(side="left", padx=(4,0), expand=True, fill="x")

        self.capture_info = ctk.CTkLabel(rec_frame, text="", font=("Inter", 11), text_color="#9CA3AF")
        self.capture_info.pack(padx=10, pady=(2,8))

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
        ctk.CTkEntry(save_row, textvariable=self.csv_path_var, width=160, font=("Inter", 10)).pack(side="left", expand=True, fill="x", padx=(0,4))
        ctk.CTkButton(save_row, text="📁", width=32, command=self.browse_csv).pack(side="left", padx=2)
        ctk.CTkButton(save_row, text="💾 Save CSV", width=90, command=self.save_csv,
                       fg_color="#3B82F6", hover_color="#2563EB").pack(side="left", padx=(4,0))

        # ── RIGHT SIDE: Waveform + Log ──
        # Waveform preview
        wave_label = ctk.CTkLabel(right, text="📈 Last Recorded Window", font=("Inter", 14, "bold"))
        wave_label.pack(anchor="w", padx=12, pady=(10,4))

        self.fig = Figure(figsize=(7, 3.5), dpi=100, facecolor="#1a1a2e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#16213e")
        self.ax.tick_params(colors="#9CA3AF", labelsize=8)
        self.ax.set_xlabel("Sample", color="#9CA3AF", fontsize=9)
        self.ax.set_ylabel("Value", color="#9CA3AF", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#374151")
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0,5))

        # Session log
        ctk.CTkLabel(right, text="📋 Session Log", font=("Inter", 14, "bold")).pack(anchor="w", padx=12, pady=(5,2))
        self.log_text = ctk.CTkTextbox(right, height=120, font=("Consolas", 10), fg_color="#0f0f23",
                                        text_color="#22C55E", corner_radius=8)
        self.log_text.pack(fill="both", padx=10, pady=(0,10), expand=False)

        self.refresh_ports()
        self.log("Session started. Connect ESP32 to begin recording.")

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

    def toggle_connection(self):
        if self.serial.is_connected:
            self.serial.disconnect()
            self.connect_btn.configure(text="Connect", fg_color="#22C55E", hover_color="#16A34A")
            self.status_label.configure(text="● Disconnected", text_color="#EF4444")
            self.log("Disconnected.")
        else:
            try:
                port = self.port_var.get()
                self.serial.connect(port)
                self.serial.send_command("S")  # Start streaming mode
                self.connect_btn.configure(text="Disconnect", fg_color="#EF4444", hover_color="#DC2626")
                self.status_label.configure(text="● Connected", text_color="#22C55E")
                self.log(f"Connected to {port}. Streaming mode active.")
            except Exception as e:
                self.log(f"ERROR: {e}")

    def start_recording(self):
        if not self.serial.is_connected:
            self.log("ERROR: Not connected to ESP32!")
            return
        self.start_btn.configure(state="disabled")
        self.countdown_active = True
        threading.Thread(target=self._countdown_then_record, daemon=True).start()

    def _countdown_then_record(self):
        for i in [3, 2, 1]:
            if not self.countdown_active:
                return
            self.countdown_label.configure(text=str(i))
            time.sleep(0.7)
        self.countdown_label.configure(text="●REC", text_color="#EF4444")

        gesture_id = self.selected_gesture.get()
        self.recorder.start_recording(gesture_id)
        self.stop_btn.configure(state="normal")
        self.log(f"Recording {GESTURE_NAMES[gesture_id]}...")

    def stop_recording(self):
        self.countdown_active = False
        self.stop_btn.configure(state="disabled")
        result = self.recorder.stop_recording()
        self.countdown_label.configure(text="", text_color="#F59E0B")
        self.start_btn.configure(state="normal")

        if result is None:
            self.log("Recording failed — no samples captured!")
            return

        flat_arr, label, raw_count = result
        name = GESTURE_NAMES[label]
        self.log(f"Captured {raw_count} samples for '{name}'. Padded/trimmed to {WINDOW_SIZE}.")
        self.capture_info.configure(text=f"Last: {raw_count} samples → {name}")
        self._update_counters()
        self._plot_waveform(flat_arr)

    def _update_counters(self):
        for name in GESTURE_NAMES:
            count = self.recorder.sample_counts.get(name, 0)
            self.count_labels[name].configure(text=f"{count}/150")
            self.progress_bars[name].set(min(count / 150.0, 1.0))

    def _plot_waveform(self, flat_arr):
        data = flat_arr.reshape(WINDOW_SIZE, NUM_AXES)
        self.ax.clear()
        self.ax.set_facecolor("#16213e")
        for i in range(NUM_AXES):
            self.ax.plot(data[:, i], color=AXIS_COLORS[i], linewidth=0.8, alpha=0.85, label=AXIS_NAMES[i])
        self.ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e", edgecolor="#374151",
                       labelcolor="#D1D5DB")
        self.ax.set_xlabel("Sample", color="#9CA3AF", fontsize=9)
        self.ax.tick_params(colors="#9CA3AF", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#374151")
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

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
        self.log(f"Saved {count} samples to {path}")
        self.recorder.clear_session()
        self._update_counters()
