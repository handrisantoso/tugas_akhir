"""
YOLO Accuracy Test - simple Tkinter UI

Pick one of the three trained models, run validation on the chosen split,
and read off mAP / precision / recall (overall and per class).

Run from a location where the relative paths below resolve, e.g. the same
folder your Training.ipynb lived in. If a path doesn't resolve, use the
Browse buttons or edit the entry fields in the UI.

Requires: ultralytics  (pip install ultralytics)
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Friendly model name -> trained weights path (as provided).
MODEL_PATHS = {
    "YOLO11n": "../Trained_Models/ImageFixYOLO11n/weights/best.pt",
    "YOLOv8n": "../Trained_Models/ImageFixYOLOv8n/weights/best.pt",
    "YOLO26n": "../Trained_Models/ImageFixYOLO26n/weights/best.pt",
}

DEFAULT_DATA_YAML = "../YOLO26n/dataset.yaml"


class App:
    def __init__(self, root):
        self.root = root
        root.title("YOLO Accuracy Test")

        pad = {"padx": 6, "pady": 4}
        row = 0

        # --- Model selection ---
        tk.Label(root, text="Model:").grid(row=row, column=0, sticky="w", **pad)
        self.model_var = tk.StringVar(value="YOLO26n")
        self.model_menu = tk.OptionMenu(
            root, self.model_var, *MODEL_PATHS.keys(), command=lambda _=None: self._refresh_weights_label()
        )
        self.model_menu.grid(row=row, column=1, sticky="w", **pad)
        row += 1

        # Resolved weights path (read-only feedback)
        tk.Label(root, text="Weights:").grid(row=row, column=0, sticky="w", **pad)
        self.weights_label = tk.Label(root, text="", anchor="w")
        self.weights_label.grid(row=row, column=1, columnspan=2, sticky="w", **pad)
        row += 1

        # --- dataset.yaml ---
        tk.Label(root, text="dataset.yaml:").grid(row=row, column=0, sticky="w", **pad)
        self.data_var = tk.StringVar(value=DEFAULT_DATA_YAML)
        tk.Entry(root, textvariable=self.data_var, width=42).grid(row=row, column=1, sticky="w", **pad)
        tk.Button(root, text="Browse...", command=self._browse_yaml).grid(row=row, column=2, sticky="w", **pad)
        row += 1

        # --- Split ---
        tk.Label(root, text="Split:").grid(row=row, column=0, sticky="w", **pad)
        self.split_var = tk.StringVar(value="test")
        split_frame = tk.Frame(root)
        split_frame.grid(row=row, column=1, sticky="w", **pad)
        tk.Radiobutton(split_frame, text="test", variable=self.split_var, value="test").pack(side="left")
        tk.Radiobutton(split_frame, text="val", variable=self.split_var, value="val").pack(side="left")
        row += 1

        # --- Numeric params ---
        tk.Label(root, text="Image size:").grid(row=row, column=0, sticky="w", **pad)
        self.imgsz_var = tk.StringVar(value="640")
        tk.Entry(root, textvariable=self.imgsz_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        tk.Label(root, text="Confidence:").grid(row=row, column=0, sticky="w", **pad)
        self.conf_var = tk.StringVar(value="0.001")
        tk.Entry(root, textvariable=self.conf_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        tk.Label(root, text="(low value = standard for mAP)").grid(row=row, column=2, sticky="w", **pad)
        row += 1

        tk.Label(root, text="IoU (NMS):").grid(row=row, column=0, sticky="w", **pad)
        self.iou_var = tk.StringVar(value="0.6")
        tk.Entry(root, textvariable=self.iou_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        row += 1

        tk.Label(root, text="Device:").grid(row=row, column=0, sticky="w", **pad)
        self.device_var = tk.StringVar(value="")
        tk.Entry(root, textvariable=self.device_var, width=10).grid(row=row, column=1, sticky="w", **pad)
        tk.Label(root, text="(blank = auto, e.g. 0 for GPU, cpu)").grid(row=row, column=2, sticky="w", **pad)
        row += 1

        # --- Run button + status ---
        self.run_btn = tk.Button(root, text="Run Accuracy Test", command=self.run_eval)
        self.run_btn.grid(row=row, column=0, columnspan=3, sticky="we", **pad)
        row += 1

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(root, textvariable=self.status_var, anchor="w").grid(
            row=row, column=0, columnspan=3, sticky="we", **pad
        )
        row += 1

        # --- Results output ---
        self.output = scrolledtext.ScrolledText(root, width=72, height=24, font=("Courier", 10))
        self.output.grid(row=row, column=0, columnspan=3, sticky="nsew", **pad)
        root.grid_rowconfigure(row, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self._refresh_weights_label()

    # ---------- helpers ----------
    def _refresh_weights_label(self):
        path = MODEL_PATHS[self.model_var.get()]
        exists = "  [found]" if os.path.isfile(path) else "  [NOT FOUND]"
        self.weights_label.config(text=path + exists)

    def _browse_yaml(self):
        path = filedialog.askopenfilename(
            title="Select dataset.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if path:
            self.data_var.set(path)

    def _set_status(self, text):
        self.status_var.set(text)

    def _write_output(self, text):
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, text)

    # ---------- run ----------
    def run_eval(self):
        model_name = self.model_var.get()
        weights = MODEL_PATHS[model_name]
        data_yaml = self.data_var.get().strip()
        split = self.split_var.get()

        if not os.path.isfile(weights):
            messagebox.showerror("Missing weights", f"Could not find model weights:\n{weights}")
            return
        if not os.path.isfile(data_yaml):
            messagebox.showerror("Missing dataset.yaml", f"Could not find dataset config:\n{data_yaml}")
            return

        try:
            imgsz = int(self.imgsz_var.get())
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
        except ValueError:
            messagebox.showerror("Bad input", "Image size must be an integer; confidence and IoU must be numbers.")
            return

        device = self.device_var.get().strip()

        self.run_btn.config(state="disabled")
        self._set_status("Running... this can take a while. The window may look idle; please wait.")
        self._write_output("")

        threading.Thread(
            target=self._worker,
            args=(weights, data_yaml, split, imgsz, conf, iou, device, model_name),
            daemon=True,
        ).start()

    def _worker(self, weights, data_yaml, split, imgsz, conf, iou, device, model_name):
        try:
            from ultralytics import YOLO

            model = YOLO(weights)
            kwargs = dict(
                data=data_yaml,
                split=split,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                plots=True,
                verbose=False,
                project=os.path.abspath("../EvalResults"),
                name=f"{model_name}_{split}",
                exist_ok=True,
            )
            if device:
                kwargs["device"] = device

            metrics = model.val(**kwargs)
            text = self._format_results(model, metrics, model_name, split)

            # Write the same overall + per-class summary to a .txt in the run folder.
            txt_path = os.path.join(str(metrics.save_dir), "accuracy_summary.txt")
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
                text = text + f"\n\nSummary saved to:\n  {txt_path}"
            except Exception as e:
                text = text + f"\n\n(Could not save summary .txt: {e})"

            self.root.after(0, self._on_success, text)
        except Exception:
            tb = traceback.format_exc()
            self.root.after(0, self._on_error, tb)

    def _on_success(self, text):
        self._write_output(text)
        self._set_status("Done.")
        self.run_btn.config(state="normal")

    def _on_error(self, tb):
        self._write_output("An error occurred:\n\n" + tb)
        self._set_status("Failed. See output below.")
        self.run_btn.config(state="normal")

    # ---------- formatting ----------
    def _format_results(self, model, m, model_name, split):
        names = model.names  # dict {id: name}
        b = m.box

        def g(attr):
            return getattr(b, attr, float("nan"))

        lines = []
        lines.append(f"Model:  {model_name}")
        lines.append(f"Split:  {split}")
        lines.append(f"Saved:  {m.save_dir}")
        lines.append("")
        lines.append("Overall")
        lines.append(f"  Precision (mean):  {g('mp'):.4f}")
        lines.append(f"  Recall (mean):     {g('mr'):.4f}")
        lines.append(f"  mAP@50:            {g('map50'):.4f}")
        lines.append(f"  mAP@75:            {g('map75'):.4f}")
        lines.append(f"  mAP@50-95:         {g('map'):.4f}")
        lines.append("")

        header = f"{'Class':<14}{'P':>9}{'R':>9}{'mAP50':>10}{'mAP50-95':>11}"
        lines.append("Per class")
        lines.append(header)
        lines.append("-" * len(header))

        try:
            for i, c in enumerate(b.ap_class_index):
                p, r, ap50, ap = b.class_result(i)
                c = int(c)
                nm = names[c] if isinstance(names, dict) else names[c]
                lines.append(f"{str(nm):<14}{p:>9.4f}{r:>9.4f}{ap50:>10.4f}{ap:>11.4f}")
        except Exception:
            lines.append("(per-class breakdown unavailable)")

        lines.append("")
        lines.append("Notes")
        lines.append("  mAP@50    = mean AP at IoU >= 0.50")
        lines.append("  mAP@50-95 = mean AP averaged over IoU 0.50 -> 0.95")
        lines.append("  The 'IoU (NMS)' field is the overlap threshold used to merge")
        lines.append("  overlapping boxes during detection, not a reported accuracy score.")
        lines.append("  Confusion matrix and PR curves are saved in the folder shown above.")
        return "\n".join(lines)


def main():
    root = tk.Tk()
    App(root)
    root.minsize(620, 560)
    root.mainloop()


if __name__ == "__main__":
    main()
