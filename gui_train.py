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
    GESTURE_NAMES, DEFAULT_CSV, HEADERS_DIR, MODELS_DIR, FIRMWARE_RF_DIR,
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

        self.full_train_var = ctk.BooleanVar(value=True)
        self.full_train_cb = ctk.CTkCheckBox(
            cfg_row, text="Train on all data",
            variable=self.full_train_var, font=("Inter", 11),
        )
        self.full_train_cb.pack(side="left", padx=8)

        # ── LOSO / CV Mode Row ──────────────────────────────────────
        cv_row = ctk.CTkFrame(top, fg_color="transparent")
        cv_row.pack(fill="x", padx=12, pady=(4, 4))

        self.loso_var = ctk.BooleanVar(value=False)
        self.loso_cb = ctk.CTkCheckBox(
            cv_row,
            text="LOSO Cross-Validation (multi-subject)",
            variable=self.loso_var,
            font=("Inter", 11, "bold"),
            command=self._on_loso_toggle,
        )
        self.loso_cb.pack(side="left", padx=(0, 8))

        self.loso_info_lbl = ctk.CTkLabel(
            cv_row, text="", font=("Inter", 10), text_color="#6B7280",
        )
        self.loso_info_lbl.pack(side="left", padx=4)

        # ── Train + Export Buttons ──
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(4,10))
        self.train_btn = ctk.CTkButton(btn_row, text="🚀 Train Model", fg_color="#8B5CF6",
                                        hover_color="#7C3AED", command=self.start_training, width=140)
        self.train_btn.pack(side="left", padx=(0,8))
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

            # Check subject count for LOSO readiness
            try:
                _, _, subj_ids = GestureRecorder.load_dataset(path)
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
        """Disable 'Train on all data' when LOSO mode is active."""
        if self.loso_var.get():
            self.full_train_cb.configure(state="disabled")
            self.loso_info_lbl.configure(
                text="  'Train on all data' disabled in LOSO mode",
                text_color="#F59E0B"
            )
        else:
            self.full_train_cb.configure(state="normal")
            self.loso_info_lbl.configure(text="", text_color="#6B7280")

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
            X, y, subject_ids = GestureRecorder.load_dataset(self.csv_var.get())
            model_type = self.model_var.get()
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            trees = int(self.trees_var.get())
            quant = self.quant_var.get()
            full_train = self.full_train_var.get()
            loso_mode = self.loso_var.get()

            # Unified progress callback — LOSO folds dispatch differently
            if loso_mode:
                def on_progress(fold_idx, n_folds, held_out, fold_result):
                    self.progress_queue.put(("loso_fold", fold_idx, n_folds, held_out, fold_result))
                def on_epoch(fold_idx, n_folds, held_out, epoch, logs):
                    self.progress_queue.put(("loso_epoch", fold_idx, n_folds, held_out, epoch, logs))
            elif model_type == "rf":
                def on_progress(tree_idx, total_trees):
                    self.progress_queue.put(("rf_progress", tree_idx, total_trees))
                on_epoch = None
            else:
                def on_progress(epoch, logs):
                    self.progress_queue.put(("epoch", epoch, logs))
                on_epoch = None

            result = train_model(
                model_type, X, y,
                epochs=epochs, batch_size=batch,
                n_estimators=trees, quantize_int8=quant,
                full_train=full_train,
                progress_callback=on_progress,
                loso_mode=loso_mode,
                subject_ids=subject_ids,
                epoch_callback=on_epoch,
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

    def _save_plot(self):
        from tkinter import filedialog
        is_loso = self._last_result and "per_fold" in self._last_result
        default_name = "loso_results.png" if is_loso else f"{self.model_var.get()}_training.png"
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
        if self._last_result and "header_path" in self._last_result:
            header = self._last_result["header_path"]
            model_type = self.model_var.get()
            # Show firmware output paths
            esp32_path = os.path.join(
                os.path.dirname(FIRMWARE_RF_DIR), "gesture_glove_esp32",
                f"{model_type}_model_data.h" if model_type != "rf" else "rf_model_data.h"
            )
            self.status_lbl.configure(
                text=f"✅ Exported → {os.path.relpath(header)}", text_color="#22C55E")
            self.report_text.delete("1.0", "end")
            self.report_text.insert("end", "Export targets:\n")
            self.report_text.insert("end", f"  • output_headers/   ← {os.path.relpath(header)}\n")
            if os.path.isfile(esp32_path):
                self.report_text.insert("end", f"  • firmware/esp32/   ← {os.path.relpath(esp32_path)}\n")
            self.report_text.insert("end", "\nTo use on ESP32: flash firmware/gesture_glove_esp32/\n")
        else:
            self.status_lbl.configure(text="Train a model first!", text_color="#EF4444")
