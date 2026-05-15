"""Tab 2 — Train & Export GUI"""
import customtkinter as ctk
import threading
import os
import queue
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from config import (
    GESTURE_NAMES, DEFAULT_CSV, HEADERS_DIR, DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH, NUM_FEATURES
)
from recorder import GestureRecorder
from models import train_model


class TrainTab(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.train_thread = None
        self.progress_queue = queue.Queue()
        self._build_ui()

    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=12)
        top.pack(fill="x", padx=10, pady=(10,5))

        # ── Dataset Row ──
        ds_row = ctk.CTkFrame(top, fg_color="transparent")
        ds_row.pack(fill="x", padx=12, pady=(10,4))
        ctk.CTkLabel(ds_row, text="📂 Dataset:", font=("Inter", 13, "bold")).pack(side="left")
        self.csv_var = ctk.StringVar(value=DEFAULT_CSV)
        ctk.CTkEntry(ds_row, textvariable=self.csv_var, width=300, font=("Inter", 10)).pack(side="left", padx=6, expand=True, fill="x")
        ctk.CTkButton(ds_row, text="Browse", width=70, command=self._browse_csv).pack(side="left")
        self.ds_info = ctk.CTkLabel(ds_row, text="", font=("Inter", 11), text_color="#9CA3AF")
        self.ds_info.pack(side="left", padx=8)

        # ── Model + Hyperparams ──
        cfg_row = ctk.CTkFrame(top, fg_color="transparent")
        cfg_row.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(cfg_row, text="Model:", font=("Inter", 12, "bold")).pack(side="left")
        self.model_var = ctk.StringVar(value="mlp")
        self.model_seg = ctk.CTkSegmentedButton(cfg_row, values=["mlp", "cnn1d", "rf"],
                                                  variable=self.model_var, command=self._on_model_change)
        self.model_seg.pack(side="left", padx=8)

        ctk.CTkLabel(cfg_row, text="Epochs:", font=("Inter", 11)).pack(side="left", padx=(16,2))
        self.epochs_var = ctk.StringVar(value=str(DEFAULT_EPOCHS))
        self.epochs_entry = ctk.CTkEntry(cfg_row, textvariable=self.epochs_var, width=50, font=("Inter", 11))
        self.epochs_entry.pack(side="left")

        ctk.CTkLabel(cfg_row, text="Batch:", font=("Inter", 11)).pack(side="left", padx=(8,2))
        self.batch_var = ctk.StringVar(value=str(DEFAULT_BATCH_SIZE))
        self.batch_entry = ctk.CTkEntry(cfg_row, textvariable=self.batch_var, width=50, font=("Inter", 11))
        self.batch_entry.pack(side="left")

        ctk.CTkLabel(cfg_row, text="Trees:", font=("Inter", 11)).pack(side="left", padx=(8,2))
        self.trees_var = ctk.StringVar(value=str(DEFAULT_RF_TREES))
        self.trees_entry = ctk.CTkEntry(cfg_row, textvariable=self.trees_var, width=50, font=("Inter", 11))
        self.trees_entry.pack(side="left")

        self.quant_var = ctk.BooleanVar(value=True)
        self.quant_cb = ctk.CTkCheckBox(cfg_row, text="Int8 Quantize", variable=self.quant_var, font=("Inter", 11))
        self.quant_cb.pack(side="left", padx=12)

        # ── Train + Export Buttons ──
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(4,10))
        self.train_btn = ctk.CTkButton(btn_row, text="🚀 Train Model", fg_color="#8B5CF6",
                                        hover_color="#7C3AED", command=self.start_training, width=140)
        self.train_btn.pack(side="left", padx=(0,8))
        self.export_btn = ctk.CTkButton(btn_row, text="📦 Export C Header", fg_color="#3B82F6",
                                         hover_color="#2563EB", command=self._export_header, width=140, state="disabled")
        self.export_btn.pack(side="left", padx=(0,8))
        ctk.CTkButton(btn_row, text="📂 Open Output", width=120, fg_color="#374151",
                       command=lambda: os.startfile(HEADERS_DIR) if os.path.isdir(HEADERS_DIR) else None).pack(side="left")
        self.status_lbl = ctk.CTkLabel(btn_row, text="Ready", font=("Inter", 11), text_color="#9CA3AF")
        self.status_lbl.pack(side="left", padx=12)
        self.size_lbl = ctk.CTkLabel(btn_row, text="", font=("Inter", 11), text_color="#F59E0B")
        self.size_lbl.pack(side="right")

        # ── Bottom: Plots ──
        bottom = ctk.CTkFrame(self, corner_radius=12)
        bottom.pack(fill="both", expand=True, padx=10, pady=(5,10))

        self.fig = Figure(figsize=(12, 4.5), dpi=100, facecolor="#1a1a2e")
        gs = self.fig.add_gridspec(1, 3, wspace=0.35)
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_acc = self.fig.add_subplot(gs[0, 1])
        self.ax_cm = self.fig.add_subplot(gs[0, 2])
        for ax in [self.ax_loss, self.ax_acc, self.ax_cm]:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#9CA3AF", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#374151")
        self.ax_loss.set_title("Loss", color="#D1D5DB", fontsize=10)
        self.ax_acc.set_title("Accuracy", color="#D1D5DB", fontsize=10)
        self.ax_cm.set_title("Confusion Matrix", color="#D1D5DB", fontsize=10)
        self.fig.tight_layout(pad=2)
        self.canvas = FigureCanvasTkAgg(self.fig, bottom)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        # ── Classification Report ──
        self.report_text = ctk.CTkTextbox(bottom, height=100, font=("Consolas", 10),
                                           fg_color="#0f0f23", text_color="#D1D5DB", corner_radius=8)
        self.report_text.pack(fill="x", padx=8, pady=(0,8))

        self._last_result = None

    def _browse_csv(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            self.csv_var.set(path)
            _, counts = GestureRecorder.load_csv(path)
            total = sum(counts.values())
            self.ds_info.configure(text=f"{total} samples loaded")

    def _on_model_change(self, val):
        is_keras = val in ("mlp", "cnn1d")
        state = "normal" if is_keras else "disabled"
        self.epochs_entry.configure(state=state)
        self.batch_entry.configure(state=state)
        self.quant_cb.configure(state=state)
        self.trees_entry.configure(state="normal" if val == "rf" else "disabled")

    def start_training(self):
        csv_path = self.csv_var.get()
        if not os.path.isfile(csv_path):
            self.status_lbl.configure(text="Dataset not found!", text_color="#EF4444")
            return

        self.train_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status_lbl.configure(text="Training...", text_color="#F59E0B")

        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.train_thread.start()
        self._poll_progress()

    def _train_worker(self):
        try:
            X, y = GestureRecorder.load_dataset(self.csv_var.get())
            model_type = self.model_var.get()
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            trees = int(self.trees_var.get())
            quant = self.quant_var.get()

            if model_type == "rf":
                def on_progress(tree_idx, total_trees):
                    self.progress_queue.put(("rf_progress", tree_idx, total_trees))
            else:
                def on_progress(epoch, logs):
                    self.progress_queue.put(("epoch", epoch, logs))

            result = train_model(
                model_type, X, y,
                epochs=epochs, batch_size=batch,
                n_estimators=trees, quantize_int8=quant,
                progress_callback=on_progress
            )
            self.progress_queue.put(("done", result))
        except Exception as e:
            import traceback
            self.progress_queue.put(("error", f"{e}\n{traceback.format_exc()}"))

    def _poll_progress(self):
        try:
            while not self.progress_queue.empty():
                msg = self.progress_queue.get_nowait()
                if msg[0] == "epoch":
                    epoch, logs = msg[1], msg[2]
                    if logs:
                        acc = logs.get("accuracy", 0)
                        val_acc = logs.get("val_accuracy", 0)
                        self.status_lbl.configure(text=f"Epoch {epoch+1} — acc: {acc:.3f} val: {val_acc:.3f}")
                elif msg[0] == "rf_progress":
                    tree_idx, total = msg[1], msg[2]
                    self.status_lbl.configure(text=f"Training RF... {tree_idx}/{total} trees")
                elif msg[0] == "done":
                    self._on_training_done(msg[1])
                    return
                elif msg[0] == "error":
                    self.status_lbl.configure(text=f"Error: {msg[1][:80]}", text_color="#EF4444")
                    self.train_btn.configure(state="normal")
                    return
        except Exception:
            pass
        self.after(200, self._poll_progress)

    def _on_training_done(self, result):
        self._last_result = result
        self.train_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        acc = result.get("accuracy", 0)
        self.status_lbl.configure(text=f"✅ Done — Accuracy: {acc:.1%}", text_color="#22C55E")

        # Model size
        fs = result.get("float_model_size", 0)
        qs = result.get("quantized_model_size", 0)
        self.size_lbl.configure(text=f"Size: {fs/1024:.1f}KB → {qs/1024:.1f}KB")

        # Plot training curves
        history = result.get("history")
        if history:
            self.ax_loss.clear()
            self.ax_loss.set_facecolor("#16213e")
            self.ax_loss.plot(history.get("loss", []), color="#EF4444", label="Train", linewidth=1.5)
            self.ax_loss.plot(history.get("val_loss", []), color="#F59E0B", label="Val", linewidth=1.5)
            self.ax_loss.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#D1D5DB")
            self.ax_loss.set_title("Loss", color="#D1D5DB", fontsize=10)
            self.ax_loss.tick_params(colors="#9CA3AF", labelsize=8)
            for s in self.ax_loss.spines.values(): s.set_color("#374151")

            self.ax_acc.clear()
            self.ax_acc.set_facecolor("#16213e")
            self.ax_acc.plot(history.get("accuracy", []), color="#22C55E", label="Train", linewidth=1.5)
            self.ax_acc.plot(history.get("val_accuracy", []), color="#3B82F6", label="Val", linewidth=1.5)
            self.ax_acc.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#D1D5DB")
            self.ax_acc.set_title("Accuracy", color="#D1D5DB", fontsize=10)
            self.ax_acc.tick_params(colors="#9CA3AF", labelsize=8)
            for s in self.ax_acc.spines.values(): s.set_color("#374151")

        # Confusion matrix
        cm = result.get("confusion")
        if cm is not None:
            self.ax_cm.clear()
            self.ax_cm.set_facecolor("#16213e")
            sns.heatmap(cm, annot=True, fmt="d", cmap="magma", ax=self.ax_cm,
                        xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                        cbar=False, linewidths=0.5, linecolor="#374151")
            self.ax_cm.set_title("Confusion Matrix", color="#D1D5DB", fontsize=10)
            self.ax_cm.tick_params(colors="#9CA3AF", labelsize=7)

        self.fig.tight_layout(pad=2)
        self.canvas.draw_idle()

        # Classification report
        report = result.get("report", {})
        self.report_text.delete("1.0", "end")
        header = f"{'Class':<15} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}\n"
        self.report_text.insert("end", header)
        self.report_text.insert("end", "─" * 45 + "\n")
        for name in GESTURE_NAMES:
            if name in report:
                r = report[name]
                line = f"{name:<15} {r['precision']:>6.2f} {r['recall']:>7.2f} {r['f1-score']:>6.2f} {int(r['support']):>8}\n"
                self.report_text.insert("end", line)
        if "accuracy" in report:
            self.report_text.insert("end", f"\n{'Accuracy':<15} {report['accuracy']:>20.2f}\n")

    def _export_header(self):
        if self._last_result and "header_path" in self._last_result:
            self.status_lbl.configure(text=f"Exported: {self._last_result['header_path']}", text_color="#22C55E")
        else:
            self.status_lbl.configure(text="Train a model first!", text_color="#EF4444")
