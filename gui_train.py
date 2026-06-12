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
    GESTURE_NAMES, DEFAULT_CSV, HEADERS_DIR, MODELS_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES
)
import os
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
        self.ds_label = ctk.CTkLabel(ds_row, text="📂 Dataset CSV:", font=("Inter", 13, "bold"))
        self.ds_label.pack(side="left")
        self.csv_var = ctk.StringVar(value=DEFAULT_CSV)
        ctk.CTkEntry(ds_row, textvariable=self.csv_var, width=300, font=("Inter", 10)).pack(side="left", padx=6, expand=True, fill="x")
        ctk.CTkButton(ds_row, text="Browse", width=70, command=self._browse_csv).pack(side="left")
        self.ds_info = ctk.CTkLabel(ds_row, text="", font=("Inter", 11), text_color="#9CA3AF")
        self.ds_info.pack(side="left", padx=8)

        # ── Model + Hyperparams ──
        cfg_row = ctk.CTkFrame(top, fg_color="transparent")
        cfg_row.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(cfg_row, text="Models:", font=("Inter", 12, "bold")).pack(side="left")
        ctk.CTkLabel(cfg_row, text="MLP + CNN1D + RF (trained together)",
                     font=("Inter", 11), text_color="#22C55E").pack(side="left", padx=8)
        # Retained for plot file naming / back-compat; training always covers all 3.
        self.model_var = ctk.StringVar(value="all")

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

        # ── LOSO / CV Mode Row ──────────────────────────────────────
        cv_row = ctk.CTkFrame(top, fg_color="transparent")
        cv_row.pack(fill="x", padx=12, pady=(4, 4))

        self.loso_var = ctk.BooleanVar(value=False)
        self.loso_cb = ctk.CTkCheckBox(
            cv_row,
            text="LOSO eval only (skip deployment export)",
            variable=self.loso_var,
            font=("Inter", 11, "bold"),
            command=self._on_loso_toggle,
        )
        self.loso_cb.pack(side="left", padx=(0, 8))

        self.loso_info_lbl = ctk.CTkLabel(
            cv_row, text="  Train 100% + export 3 models, then LOSO eval", font=("Inter", 10), text_color="#22C55E",
        )
        self.loso_info_lbl.pack(side="left", padx=4)

        # ── Train + Export Buttons ──
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(4,10))
        self.train_btn = ctk.CTkButton(btn_row, text="🚀 Train All (MLP+CNN+RF)", fg_color="#8B5CF6",
                                        hover_color="#7C3AED", command=self.start_training, width=190)
        self.train_btn.pack(side="left", padx=(0,8))
        self.thesis_btn = ctk.CTkButton(btn_row, text="📊 Thesis Metrics", fg_color="#0EA5E9",
                                         hover_color="#0284C7", command=self.start_thesis_metrics, width=140)
        self.thesis_btn.pack(side="left", padx=(0,8))
        self.export_btn = ctk.CTkButton(btn_row, text="📦 Export C Header", fg_color="#3B82F6",
                                         hover_color="#2563EB", command=self._export_header, width=140, state="disabled")
        self.export_btn.pack(side="left", padx=(0,8))
        self.save_plot_btn = ctk.CTkButton(btn_row, text="💾 Save Plot", fg_color="#374151",
                                            hover_color="#4B5563", command=self._save_plot, width=100, state="disabled")
        self.save_plot_btn.pack(side="left", padx=(0,8))
        ctk.CTkButton(btn_row, text="📂 Open Output", width=120, fg_color="#374151",
                       command=lambda: os.startfile(HEADERS_DIR) if os.path.isdir(HEADERS_DIR) else None).pack(side="left")
        self.status_lbl = ctk.CTkLabel(btn_row, text="Ready", font=("Inter", 11), text_color="#9CA3AF")
        self.status_lbl.pack(side="left", padx=12)
        self.size_lbl = ctk.CTkLabel(btn_row, text="", font=("Inter", 11), text_color="#F59E0B")
        self.size_lbl.pack(side="right")

        # ── Bottom: Plots ──
        bottom = ctk.CTkFrame(self, corner_radius=12)
        bottom.pack(fill="both", expand=True, padx=10, pady=(5,10))

        self.fig = Figure(figsize=(12, 8.5), dpi=100, facecolor="#1a1a2e")
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
        self._refresh_loso_info(DEFAULT_CSV)

    def _browse_csv(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv")], initialdir=os.path.dirname(self.csv_var.get()))
        if path:
            self.csv_var.set(path)
            self._refresh_loso_info(path)

    def _refresh_loso_info(self, csv_path):
        try:
            _, counts = GestureRecorder.load_csv(csv_path)
            total = sum(counts.values())
            self.ds_info.configure(text=f"{total} samples loaded")
            _, _, subj_ids = GestureRecorder.load_dataset(csv_path)
            unique_subjects = np.unique(subj_ids)
            if len(unique_subjects) > 1:
                self.loso_info_lbl.configure(
                    text=f"  {len(unique_subjects)} subjects detected — LOSO ready",
                    text_color="#22C55E"
                )
            else:
                self.loso_info_lbl.configure(
                    text="  Single subject — standard split will be used",
                    text_color="#6B7280"
                )
                if self.loso_var.get():
                    self.loso_var.set(False)
                    self._on_loso_toggle()
        except Exception:
            self.loso_info_lbl.configure(text="", text_color="#6B7280")

    def _on_model_change(self, val):
        is_keras = val in ("mlp", "cnn1d")
        state = "normal" if is_keras else "disabled"
        self.epochs_entry.configure(state=state)
        self.batch_entry.configure(state=state)
        self.quant_cb.configure(state=state)
        self.trees_entry.configure(state="normal" if val == "rf" else "disabled")

    def _on_loso_toggle(self):
        # Data source is always the CSV; the checkbox only toggles whether the
        # 3 deployment models are trained+exported before the LOSO evaluation.
        if self.loso_var.get():
            self.loso_info_lbl.configure(
                text="  LOSO evaluation only — no deployment export (faster)",
                text_color="#F59E0B"
            )
        else:
            self.loso_info_lbl.configure(
                text="  Train 100% + export 3 models, then LOSO eval",
                text_color="#22C55E"
            )

    def start_training(self):
        csv_path = self.csv_var.get()
        if not os.path.isfile(csv_path):
            self.status_lbl.configure(text="Dataset CSV not found!", text_color="#EF4444")
            return

        self.train_btn.configure(state="disabled")
        self.thesis_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status_lbl.configure(text="Training...", text_color="#F59E0B")

        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.train_thread.start()
        self._poll_progress()

    def _train_worker(self):
        """Train ALL 3 models (MLP+CNN1D+RF) in one run, then LOSO-eval all 3.

        Non-LOSO mode: train each model on 100% data + export C header, THEN run
        one LOSO pass (all 3 models per fold via run_loso) for the honest report.
        LOSO mode: skip deployment, just run the LOSO evaluation.
        """
        try:
            import thesis_all_metrics as tam
            path = self.csv_var.get()
            loso_mode = self.loso_var.get()   # True = skip deployment, LOSO eval only
            X, y, subject_ids = GestureRecorder.load_dataset(path)
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            trees = int(self.trees_var.get())
            quant = self.quant_var.get()
            model_types = ["mlp", "cnn1d", "rf"]

            # ── 1. Deployment: train + export all 3 on 100% data ──────────
            train_results = {}
            if not loso_mode:
                for i, mt in enumerate(model_types):
                    self.progress_queue.put((
                        "status_update",
                        f"[deploy {i+1}/3] Training {mt.upper()} on 100% data + export..."))
                    cb = None
                    if mt == "rf":
                        def cb(t, tot):
                            self.progress_queue.put((
                                "status_update", f"[deploy 3/3] RF {t}/{tot} trees"))
                    train_results[mt] = train_model(
                        mt, X, y, epochs=epochs, batch_size=batch,
                        n_estimators=trees, quantize_int8=quant,
                        full_train=True, progress_callback=cb,
                    )
                    # Surface any export errors immediately
                    res = train_results[mt]
                    if res.get("header_path"):
                        self.progress_queue.put((
                            "status_update",
                            f"[deploy {i+1}/3] {mt.upper()} header exported OK"))
                    else:
                        err = res.get("header_error") or res.get("tflite_error") or "unknown"
                        self.progress_queue.put((
                            "status_update",
                            f"[deploy {i+1}/3] {mt.upper()} export FAILED: {err[:60]}"))

            # ── 2. LOSO evaluation: all 3 models in a single pass ─────────
            def loso_progress(text):
                self.progress_queue.put(("status_update", text))
            raw = tam.run_loso(X, y, subject_ids, progress_cb=loso_progress)
            agg = tam.aggregate_loso(raw)
            esp = tam.esp32_profile()
            protocol = f"LOSO {len(np.unique(subject_ids))}-fold, cross-subject"
            report_text = "\n".join(tam.build_report_lines(agg, None, esp, protocol=protocol))

            self.progress_queue.put(
                ("train_all_done", agg, esp, report_text, protocol, train_results))
        except Exception as e:
            import traceback
            self.progress_queue.put(("error", f"{e}\n{traceback.format_exc()}"))

    # ── Thesis Metrics (all 3 models, full report) ──────────────────────
    def start_thesis_metrics(self):
        path = self.csv_var.get()
        if not os.path.isfile(path):
            self.status_lbl.configure(text="Dataset CSV not found!", text_color="#EF4444")
            return

        self.train_btn.configure(state="disabled")
        self.thesis_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.status_lbl.configure(
            text="Thesis metrics: training RF/MLP/CNN1D (this takes a few minutes)...",
            text_color="#F59E0B")

        self.train_thread = threading.Thread(target=self._thesis_worker, daemon=True)
        self.train_thread.start()
        self._poll_progress()

    def _thesis_worker(self):
        try:
            import thesis_all_metrics as tam
            path = self.csv_var.get()
            X, y, subj = GestureRecorder.load_dataset(path)

            def progress_cb(text):
                self.progress_queue.put(("thesis_progress", text))

            # Thesis metrics are always reported on the honest LOSO protocol.
            agg, pc, esp, protocol = tam.compute_thesis_metrics(
                X, y, subj, loso_mode=True, progress_cb=progress_cb,
                include_pc=True, include_esp=True,
            )
            report_lines = tam.build_report_lines(agg, pc, esp, protocol=protocol)
            report_text = "\n".join(report_lines)
            self.progress_queue.put(
                ("thesis_done", agg, esp, report_text, protocol))
        except Exception as e:
            import traceback
            self.progress_queue.put(("thesis_error", f"{e}\n{traceback.format_exc()}"))

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
                elif msg[0] == "loso_epoch":
                    fold_idx, n_folds, held_out, epoch, logs = msg[1], msg[2], msg[3], msg[4], msg[5]
                    acc = logs.get("accuracy", 0)
                    self.status_lbl.configure(
                        text=f"LOSO Fold {fold_idx+1}/{n_folds} — epoch {epoch+1} — acc: {acc:.3f}",
                        text_color="#A78BFA",
                    )
                elif msg[0] == "loso_fold":
                    fold_idx, n_folds, held_out, fold_result = msg[1], msg[2], msg[3], msg[4]
                    acc = fold_result.get("accuracy", 0)
                    self.status_lbl.configure(
                        text=f"LOSO Fold {fold_idx+1}/{n_folds} — "
                             f"held out: {held_out} — acc: {acc:.1%}",
                        text_color="#F59E0B"
                    )
                elif msg[0] == "status_update":
                    self.status_lbl.configure(text=msg[1][:90], text_color="#F59E0B")
                elif msg[0] == "train_all_done":
                    self._on_train_all_done(msg[1], msg[2], msg[3], msg[4], msg[5])
                    return
                elif msg[0] == "thesis_progress":
                    self.status_lbl.configure(text=msg[1][:90], text_color="#38BDF8")
                elif msg[0] == "thesis_done":
                    self._on_thesis_done(msg[1], msg[2], msg[3], msg[4])
                    return
                elif msg[0] == "thesis_error":
                    self.status_lbl.configure(text=f"Thesis error: {msg[1][:70]}", text_color="#EF4444")
                    self.report_text.delete("1.0", "end")
                    self.report_text.insert("end", msg[1])
                    self.train_btn.configure(state="normal")
                    self.thesis_btn.configure(state="normal")
                    return
                elif msg[0] == "done":
                    self._on_training_done(msg[1])
                    return
                elif msg[0] == "error":
                    self.status_lbl.configure(text=f"Error: {msg[1][:80]}", text_color="#EF4444")
                    self.train_btn.configure(state="normal")
                    self.thesis_btn.configure(state="normal")
                    return
        except Exception:
            pass
        self.after(200, self._poll_progress)

    def _on_training_done(self, result):
        self._last_result = result
        self.train_btn.configure(state="normal")
        self.thesis_btn.configure(state="normal")

        # ── LOSO result ────────────────────────────────────────────────
        if "per_fold" in result and "aggregated" in result:
            self._render_loso_results(result)
            return

        # ── Standard single-split result ────────────────────────────────
        self.export_btn.configure(state="normal")
        self.save_plot_btn.configure(state="normal")
        acc        = result.get("accuracy", 0)        # eval holdout accuracy
        val_acc    = result.get("val_accuracy", 0)
        train_n    = result.get("train_samples", 0)
        val_n      = result.get("val_samples", 0)
        test_n     = result.get("test_samples", 0)
        model_type = self.model_var.get()
        deployed_all = result.get("deployed_on_all", False)

        note = " [all data]" if deployed_all else ""
        self.status_lbl.configure(
            text=f"Done{note} — Eval: {acc:.1%}  "
                 f"(train={train_n} eval={test_n})",
            text_color="#22C55E"
        )

        # Model size
        fs = result.get("float_model_size", 0)
        qs = result.get("quantized_model_size", 0)
        self.size_lbl.configure(text=f"Size: {fs/1024:.1f}KB → {qs/1024:.1f}KB")

        # Auto-export header immediately after training
        header_path = result.get("header_path", "")
        if header_path:
            # Copy to all firmware directories
            targets = {
                "PC model": os.path.join(MODELS_DIR, f"{model_type}_model.keras" if model_type != "rf" else f"{model_type}_model.joblib"),
                "ESP32 header": header_path,
            }
            for label, path in targets.items():
                if os.path.isfile(path):
                    short = os.path.relpath(path)
                    self.status_lbl.configure(
                        text=f"✅ Exported → {short}", text_color="#22C55E")
                    break
            self.export_btn.configure(text="📦 Re-export C Header")
        else:
            err = result.get("header_error", "unknown error")
            self.status_lbl.configure(text=f"⚠️ Export skipped: {err[:60]}", text_color="#F59E0B")

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

        # Classification report (EVAL holdout set)
        report = result.get("report", {})
        self.report_text.delete("1.0", "end")
        mode_note = " [Deployed: ALL data]" if deployed_all else " [70/20/10 split]"
        self.report_text.insert("end", "=" * 52 + "\n")
        self.report_text.insert("end", f"  EVAL HOLD-OUT REPORT{mode_note}\n")
        self.report_text.insert("end", f"  Train (deployed): {train_n}  Eval holdout: {test_n}\n")
        self.report_text.insert("end", "=" * 52 + "\n")
        self.report_text.insert("end",
            f"{'Class':<14} {'Prec':>6} {'Recall':>7} {'F1':>6} {'Support':>8}\n")
        self.report_text.insert("end", "─" * 45 + "\n")
        for name in GESTURE_NAMES:
            if name in report:
                r = report[name]
                self.report_text.insert("end",
                    f"{name:<14} {r['precision']:>6.2f} {r['recall']:>7.2f} "
                    f"{r['f1-score']:>6.2f} {int(r['support']):>8}\n")
        if "accuracy" in report:
            self.report_text.insert("end", "─" * 45 + "\n")
            self.report_text.insert("end",
                f"{'Test Accuracy':<14} {report['accuracy']:>29.2f}\n")
            self.report_text.insert("end",
                f"{'Val Accuracy':<14} {val_acc:>29.2f}\n")
            self.report_text.insert("end", "─" * 45 + "\n")

    def _on_train_all_done(self, agg, esp, report_text, protocol, train_results):
        """All 3 models trained (+exported if non-LOSO) and LOSO-evaluated."""
        self._last_result = {"thesis": True, "train_results": train_results,
                             "agg": agg, "esp": esp}
        self.train_btn.configure(state="normal")
        self.thesis_btn.configure(state="normal")
        self.save_plot_btn.configure(state="normal")

        models = ["RF", "MLP", "CNN1D"]
        if train_results:
            self.export_btn.configure(state="normal", text="📦 Show Export Paths")
        else:
            self.export_btn.configure(state="disabled")

        best = max(models, key=lambda m: agg[m]["mean_acc"])
        if train_results:
            n_exported = sum(1 for r in train_results.values() if r.get("header_path"))
            prefix = f"✅ {n_exported}/3 headers exported"
        else:
            prefix = "✅ LOSO eval done (3 models)"
        self.status_lbl.configure(
            text=(f"{prefix} — best LOSO {best}: "
                  f"{agg[best]['mean_acc']:.1%} ± {agg[best]['std_acc']:.1%}"),
            text_color="#22C55E" if not train_results or n_exported == 3 else "#F59E0B",
        )
        self.size_lbl.configure(text="")
        self._render_thesis_results(agg, esp, report_text, protocol)

    def _render_loso_results(self, result):
        """Render full LOSO cross-validation results in the GUI."""
        agg = result["aggregated"]
        per_fold = result["per_fold"]
        n_subjects = result["n_subjects"]
        subjects = result["subjects"]
        confusion_agg = result["confusion_agg"]

        # ── Status bar ────────────────────────────────────────────────
        acc_m, acc_s = agg["accuracy_mean"], agg["accuracy_std"]
        f1_m,  f1_s  = agg["f1_mean"],       agg["f1_std"]
        self.status_lbl.configure(
            text=(f"LOSO Done — {n_subjects} subjects — "
                  f"Acc: {acc_m:.1%} ± {acc_s:.1%}  "
                  f"F1:  {f1_m:.1%} ± {f1_s:.1%}"),
            text_color="#22C55E"
        )
        self.export_btn.configure(state="disabled")
        self.save_plot_btn.configure(state="normal")
        self.size_lbl.configure(text="")

        # ── Reconfigure figure: same 3-panel layout as standard training ──
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 3, wspace=0.38)

        # ── Panel 1: Per-fold accuracy (replaces Loss curve) ───────────
        ax_fold = self.fig.add_subplot(gs[0, 0])
        ax_fold.set_facecolor("#16213e")
        fold_labels = [f"Fold {i+1}" for i in range(len(per_fold))]
        fold_accs   = [f["accuracy"] for f in per_fold]
        fold_f1s    = [f["f1"]       for f in per_fold]
        x = np.arange(len(fold_labels))
        w = 0.35
        ax_fold.bar(x - w/2, fold_accs, w, color="#22C55E",
                    edgecolor="#374151", linewidth=0.5, label="Accuracy")
        ax_fold.bar(x + w/2, fold_f1s,  w, color="#8B5CF6",
                    edgecolor="#374151", linewidth=0.5, label="F1")
        ax_fold.axhline(acc_m, color="#22C55E", linestyle="--", linewidth=1, alpha=0.6)
        ax_fold.axhline(f1_m,  color="#8B5CF6", linestyle="--", linewidth=1, alpha=0.6)
        ax_fold.set_xticks(x)
        ax_fold.set_xticklabels(fold_labels, color="#9CA3AF", fontsize=7)
        ax_fold.set_ylabel("Score", color="#D1D5DB", fontsize=9)
        ax_fold.set_ylim(0, 1.05)
        ax_fold.set_title(
            f"Per-Fold  (mean acc {acc_m:.1%})", color="#D1D5DB", fontsize=10)
        ax_fold.tick_params(colors="#9CA3AF", labelsize=7)
        for sp in ax_fold.spines.values(): sp.set_color("#374151")
        ax_fold.legend(fontsize=7, facecolor="#1a1a2e",
                        edgecolor="#374151", labelcolor="#D1D5DB")

        # ── Panel 2: Aggregated confusion matrix (same as standard) ────
        ax_cm = self.fig.add_subplot(gs[0, 1])
        ax_cm.set_facecolor("#16213e")
        sns.heatmap(
            confusion_agg, annot=True, fmt="d", cmap="magma", ax=ax_cm,
            xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
            cbar=False, linewidths=0.5, linecolor="#374151",
        )
        ax_cm.set_title("Confusion Matrix\n(aggregated across folds)",
                         color="#D1D5DB", fontsize=10)
        ax_cm.set_xlabel("Predicted", color="#9CA3AF", fontsize=8)
        ax_cm.set_ylabel("True",      color="#9CA3AF", fontsize=8)
        ax_cm.tick_params(colors="#9CA3AF", labelsize=7)

        # ── Panel 3: Per-gesture F1 breakdown ──────────────────────────
        ax_f1 = self.fig.add_subplot(gs[0, 2])
        ax_f1.set_facecolor("#16213e")
        # Average F1 per gesture class across all folds
        gesture_f1 = {g: [] for g in GESTURE_NAMES}
        for f in per_fold:
            rep = f.get("report", {})
            for g in GESTURE_NAMES:
                gesture_f1[g].append(rep.get(g, {}).get("f1-score", 0.0))
        g_means = [np.mean(gesture_f1[g]) for g in GESTURE_NAMES]
        g_stds  = [np.std(gesture_f1[g])  for g in GESTURE_NAMES]
        bar_colors = ["#8B5CF6", "#3B82F6", "#22C55E", "#F59E0B", "#EF4444"]
        xg = np.arange(len(GESTURE_NAMES))
        ax_f1.bar(xg, g_means, yerr=g_stds, capsize=4,
                  color=bar_colors, edgecolor="#374151", linewidth=0.5, alpha=0.85)
        ax_f1.set_xticks(xg)
        ax_f1.set_xticklabels(GESTURE_NAMES, color="#D1D5DB", fontsize=7,
                               rotation=15, ha="right")
        ax_f1.set_ylabel("F1 Score", color="#D1D5DB", fontsize=9)
        ax_f1.set_ylim(0, 1.15)
        ax_f1.set_title("Per-Gesture F1\n(mean ± std across folds)",
                         color="#D1D5DB", fontsize=10)
        ax_f1.tick_params(colors="#9CA3AF", labelsize=7)
        for sp in ax_f1.spines.values(): sp.set_color("#374151")
        for i, (m, s) in enumerate(zip(g_means, g_stds)):
            ax_f1.text(i, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom",
                       color="#D1D5DB", fontsize=7)

        self.fig.tight_layout(pad=2)
        self.canvas.draw_idle()

        # ── LOSO Classification Report Textbox ─────────────────────────
        self.report_text.delete("1.0", "end")
        sep = "─" * 62
        self.report_text.insert("end", "LOSO Cross-Validation Report\n")
        self.report_text.insert("end", f"{sep}\n")
        self.report_text.insert("end",
            f"  Subjects (K={n_subjects}): {', '.join(subjects)}\n")
        self.report_text.insert("end", f"{sep}\n")
        self.report_text.insert("end",
            f"{'Subject':<22} {'Train':>7} {'Test':>6} "
            f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}\n")
        self.report_text.insert("end", f"{sep}\n")
        for f in per_fold:
            self.report_text.insert("end",
                f"{str(f['subject_held_out']):<22} "
                f"{f['train_n']:>7} {f['test_n']:>6} "
                f"{f['accuracy']:>7.2f} {f['precision']:>7.2f} "
                f"{f['recall']:>7.2f} {f['f1']:>7.2f}\n")
        self.report_text.insert("end", f"{sep}\n")
        self.report_text.insert("end",
            f"{'AGGREGATED (mean ± std)':<22} {'':>14} "
            f"{acc_m:>7.2f} {agg['precision_mean']:>7.2f} "
            f"{agg['recall_mean']:>7.2f} {agg['f1_mean']:>7.2f}\n")
        self.report_text.insert("end",
            f"{'':>22}  ±{acc_s:>6.2f} ±{agg['precision_std']:>6.2f} "
            f"±{agg['recall_std']:>6.2f} ±{agg['f1_std']:>6.2f}\n")

    # ── Thesis Metrics rendering ─────────────────────────────────────────
    def _on_thesis_done(self, agg, esp, report_text, protocol):
        self._last_result = {"thesis": True}
        self.train_btn.configure(state="normal")
        self.thesis_btn.configure(state="normal")
        self.export_btn.configure(state="disabled")   # no single deployable model in this mode
        self.save_plot_btn.configure(state="normal")
        self.size_lbl.configure(text="")

        models = ["RF", "MLP", "CNN1D"]
        best = max(models, key=lambda m: agg[m]["mean_acc"])
        self.status_lbl.configure(
            text=(f"Thesis [{protocol}] — best {best}: "
                  f"{agg[best]['mean_acc']:.1%} ± {agg[best]['std_acc']:.1%}"),
            text_color="#22C55E",
        )
        self._render_thesis_results(agg, esp, report_text, protocol)

    def _render_thesis_results(self, agg, esp, report_text, protocol):
        """Clean 2×3 panel comparison across RF / MLP / CNN1D + per-model CM + text report."""
        models = ["RF", "MLP", "CNN1D"]
        model_colors = {"RF": "#22C55E", "MLP": "#3B82F6", "CNN1D": "#F59E0B"}
        x = np.arange(len(models))

        self.fig.clear()
        # 2 rows: top = summary metrics, bottom = per-model confusion matrices
        gs = self.fig.add_gridspec(2, 3, wspace=0.38, hspace=0.45)

        def _style(ax):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#9CA3AF", labelsize=7)
            for sp in ax.spines.values():
                sp.set_color("#374151")

        # ── Row 0, Panel 0: Accuracy & F1 (macro) per model ──────────
        ax_acc = self.fig.add_subplot(gs[0, 0]); _style(ax_acc)
        accs    = [agg[m]["mean_acc"] for m in models]
        acc_std = [agg[m]["std_acc"] for m in models]
        f1s     = [agg[m]["mean_f1_macro"] for m in models]
        f1_std  = [agg[m]["std_f1_macro"] for m in models]
        w = 0.36
        ax_acc.bar(x - w/2, accs, w, yerr=acc_std, capsize=3, color="#22C55E",
                   edgecolor="#374151", linewidth=0.5, label="Accuracy")
        ax_acc.bar(x + w/2, f1s, w, yerr=f1_std, capsize=3, color="#8B5CF6",
                   edgecolor="#374151", linewidth=0.5, label="F1 (macro)")
        ax_acc.set_xticks(x); ax_acc.set_xticklabels(models, color="#D1D5DB", fontsize=8)
        ax_acc.set_ylim(0, 1.08)
        ax_acc.set_ylabel("Score", color="#D1D5DB", fontsize=9)
        ax_acc.set_title("Accuracy & F1", color="#D1D5DB", fontsize=10)
        ax_acc.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#D1D5DB")
        for i, a in enumerate(accs):
            ax_acc.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", va="bottom",
                        color="#D1D5DB", fontsize=7)

        # ── Row 0, Panel 1: Per-gesture F1 (grouped bars per model) ──
        ax_f1 = self.fig.add_subplot(gs[0, 1]); _style(ax_f1)
        xg = np.arange(len(GESTURE_NAMES))
        bw = 0.26
        for j, m in enumerate(models):
            vals = [agg[m]["per_class"][g]["f1"] for g in GESTURE_NAMES]
            ax_f1.bar(xg + (j - 1) * bw, vals, bw, color=model_colors[m],
                      edgecolor="#374151", linewidth=0.4, label=m)
        ax_f1.set_xticks(xg)
        ax_f1.set_xticklabels(GESTURE_NAMES, color="#D1D5DB", fontsize=6.5,
                              rotation=20, ha="right")
        ax_f1.set_ylim(0, 1.12)
        ax_f1.set_ylabel("F1 Score", color="#D1D5DB", fontsize=9)
        ax_f1.set_title("Per-Gesture F1", color="#D1D5DB", fontsize=10)
        ax_f1.legend(fontsize=6.5, facecolor="#1a1a2e", edgecolor="#374151", labelcolor="#D1D5DB")

        # ── Row 0, Panel 2: ESP32 on-device latency ──────────────────
        ax_lat = self.fig.add_subplot(gs[0, 2]); _style(ax_lat)
        if esp is not None:
            lat = [esp[m]["latency_mean_ms"] for m in models]
            bars = ax_lat.bar(x, lat, 0.55, color=[model_colors[m] for m in models],
                              edgecolor="#374151", linewidth=0.5)
            ax_lat.set_yscale("log")
            ax_lat.set_xticks(x); ax_lat.set_xticklabels(models, color="#D1D5DB", fontsize=8)
            ax_lat.set_ylabel("Latency (ms, log)", color="#D1D5DB", fontsize=9)
            ax_lat.set_title("ESP32 Latency & FPS", color="#D1D5DB", fontsize=10)
            for i, m in enumerate(models):
                ax_lat.text(i, lat[i] * 1.1, f"{lat[i]:.2f}ms\n{esp[m]['fps']:.0f} FPS",
                            ha="center", va="bottom", color="#D1D5DB", fontsize=6.5)
        else:
            ax_lat.set_title("ESP32 (n/a)", color="#D1D5DB", fontsize=10)

        # ── Row 1: Per-model Confusion Matrix ────────────────────────
        cm_colormaps = {"RF": "Greens", "MLP": "Blues", "CNN1D": "Oranges"}
        for col, m in enumerate(models):
            ax_cm = self.fig.add_subplot(gs[1, col])
            ax_cm.set_facecolor("#16213e")
            cm = agg[m].get("confusion")
            if cm is not None:
                sns.heatmap(cm, annot=True, fmt="d", cmap=cm_colormaps[m], ax=ax_cm,
                            xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                            cbar=False, linewidths=0.5, linecolor="#374151",
                            annot_kws={"fontsize": 7})
                # Compute accuracy from CM for subtitle
                cm_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0
                ax_cm.set_title(f"CM — {m}  (acc {cm_acc:.1%})", color="#D1D5DB", fontsize=9)
            else:
                ax_cm.set_title(f"CM — {m}  (n/a)", color="#D1D5DB", fontsize=9)
                ax_cm.text(0.5, 0.5, "No data", transform=ax_cm.transAxes,
                           ha="center", va="center", color="#6B7280", fontsize=10)
            ax_cm.set_xlabel("Predicted", color="#9CA3AF", fontsize=7)
            ax_cm.set_ylabel("True", color="#9CA3AF", fontsize=7)
            ax_cm.tick_params(colors="#9CA3AF", labelsize=6)

        self.fig.suptitle(f"Thesis Metrics — {protocol}", color="#E5E7EB", fontsize=11)
        self.fig.tight_layout(pad=2, rect=(0, 0, 1, 0.95))
        self.canvas.draw_idle()

        # ── Full text report into the report box ──────────────────────
        self.report_text.delete("1.0", "end")
        self.report_text.insert("end", report_text)
        self.report_text.see("1.0")


    def _save_plot(self):
        from tkinter import filedialog
        is_loso = self._last_result and "per_fold" in self._last_result
        is_thesis = self._last_result and self._last_result.get("thesis")
        if is_thesis:
            default_name = "thesis_metrics.png"
        elif is_loso:
            default_name = "loso_results.png"
        else:
            default_name = f"{self.model_var.get()}_training.png"
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile=default_name,
        )
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        self.status_lbl.configure(text=f"✅ Plot saved → {os.path.basename(path)}",
                                   text_color="#22C55E")

    def _export_header(self):
        # Multi-model export (Train All) — show C-header path for every model
        train_results = (self._last_result or {}).get("train_results")
        if train_results:
            self.report_text.delete("1.0", "end")
            self.report_text.insert("end", "Exported deployment models (trained on 100% data):\n\n")
            n_ok = 0
            for mt, res in train_results.items():
                hp = res.get("header_path", "")
                if hp:
                    n_ok += 1
                    self.report_text.insert("end",
                        f"  • {mt.upper():<6} → {os.path.relpath(hp)}\n")
                err = res.get("header_error")
                if err:
                    self.report_text.insert("end", f"  • {mt.upper():<6} ✗ {err[:60]}\n")
            self.report_text.insert("end",
                "\nAll headers also copied to firmware/gesture_glove_esp32/\n")
            self.status_lbl.configure(
                text=f"✅ {n_ok}/3 model headers exported", text_color="#22C55E")
            return
        self.status_lbl.configure(text="Train models first!", text_color="#EF4444")
