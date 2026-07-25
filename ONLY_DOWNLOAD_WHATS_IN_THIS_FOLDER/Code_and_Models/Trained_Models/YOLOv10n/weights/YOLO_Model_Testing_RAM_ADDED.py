"""
YOLO Accuracy Test - Raspberry Pi 5 edition (Hailo .hef + Ultralytics .pt)
==========================================================================
Evaluates a trained detector on a Raspberry Pi 5 and reports the standard
metrics (Precision, Recall, mAP@50, mAP@75, mAP@50-95, plus per-class numbers)
together with timing so you can compare the Pi 5 WITH the Hailo accelerator
against the Pi 5 running the plain PyTorch model on its CPU.

Two backends, selected with hailo=true|false:

    hailo=false  ->  runs best.pt through Ultralytics model.val() on the CPU
    hailo=true   ->  runs the compiled .hef through HailoRT, then scores the
                     detections against the dataset labels in this script

Run from a terminal, e.g.:
    python YOLO_Accuracy_CMD.py model=YOLOv10n hailo=false           # plain CPU
    python YOLO_Accuracy_CMD.py model=YOLOv10n hailo=true            # Hailo NPU
    python YOLO_Accuracy_CMD.py model=YOLOv10n hailo=true rgb=false  # if .hef wants BGR
    python YOLO_Accuracy_CMD.py model=YOLOv10n hailo=false split=val

Expected folder layout (siblings of the folder this script lives in). Each
model folder holds BOTH files, e.g. directly in the folder or in a weights/
subfolder:
    Trained_Models/<ModelName>/weights/best.pt          <- PyTorch weights
    Trained_Models/<ModelName>/weights/yolov11n.hef     <- Hailo-compiled model
    Dataset/dataset.yaml                                <- data config
    Output_Folder/<ModelName>_Hailo/                    <- Hailo results land here
    Output_Folder/<ModelName>_CPU/                      <- CPU results land here

Prerequisites
-------------
* hailo=false path:  pip install ultralytics   (pulls in a CPU torch build)
* hailo=true  path:  HailoRT + its Python package (the `hailo_platform` module
                     that ships with the Pi 5 AI Kit / hailo-all), plus
                     python3-opencv and pyyaml. It does NOT need torch.

IMPORTANT assumptions for the Hailo path (read these if your mAP looks wrong)
----------------------------------------------------------------------------
* The .hef is assumed to be compiled WITH on-chip NMS (the usual Hailo Model
  Zoo / hailomz output for YOLO). The output is then a per-class list of
  [y_min, x_min, y_max, x_max, score] boxes normalised to 0..1. If your .hef
  has NO built-in NMS, this script will tell you and you'll need a decode step.
* Images are stretch-resized to the network input (no aspect-ratio padding).
  Normalised detections map straight back onto the original image.
* Inputs are fed as RGB uint8. Pass rgb=false if your .hef expects BGR.
The Hailo and CPU numbers are computed by different code paths, so treat tiny
differences (<1 mAP point) as noise; the point of this tool is the big picture.
"""

import os
import sys
import glob
import time
import threading
import traceback
from contextlib import nullcontext

# Helps avoid the "OMP: Error #15 ... libiomp5md.dll" type clashes.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Folder this script lives in. All relative paths below are anchored to it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Friendly model name -> folder that holds the trained weights.
MODEL_DIRS = {
    "YOLO11n": "../Trained_Models/YOLO11n",
    "YOLOv8n": "../Trained_Models/YOLOv8n",
    "YOLOv10n": "../Trained_Models/YOLOv10n",
}

OUTPUT_ROOT = "../Output_Folder"   # results land in ../Output_Folder/<ModelName>_<Backend>/

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ==================================================================================
# Path helpers
# ==================================================================================
def anchor(path):
    """Absolute paths pass through; relative paths resolve against the script dir."""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


def _find_weights(model_name, exts):
    """Find weight files with the given extensions for a friendly model name."""
    folder = MODEL_DIRS.get(model_name)
    if not folder:
        return []
    folder = anchor(folder)
    if not os.path.isdir(folder):
        return []
    hits = []
    for ext in exts:
        hits += glob.glob(os.path.join(folder, "*" + ext))          # in the folder
        hits += glob.glob(os.path.join(folder, "*", "*" + ext))     # one level deep
        hits += glob.glob(os.path.join(folder, "*", "*", "*" + ext))  # weights/ etc.
    return sorted(set(hits))


def resolve_pt(model_name):
    """Return the .pt path for a friendly model name, or None if not found."""
    pts = _find_weights(model_name, (".pt",))
    if not pts:
        return None
    key = model_name.lower().replace("yolo", "")
    for p in pts:                       # prefer a name that matches the model key
        if key in os.path.basename(p).lower():
            return p
    for target in ("best.pt", "last.pt"):
        for p in pts:
            if os.path.basename(p).lower() == target:
                return p
    return pts[0]


def resolve_hef(model_name):
    """Return the .hef path for a friendly model name, or None if not found."""
    hefs = _find_weights(model_name, (".hef",))
    if not hefs:
        return None
    key = model_name.lower().replace("yolo", "")
    for h in hefs:                      # prefer a name that matches the model key
        if key in os.path.basename(h).lower():
            return h
    return hefs[0]


# ==================================================================================
# dataset.yaml parsing  (used by the Hailo path to load images + ground truth)
# ==================================================================================
def resolve_split(data_yaml, split):
    """Return a split that actually exists in the yaml, falling back if needed."""
    try:
        import yaml
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return split
    if cfg.get(split) is not None:
        return split
    fallback = "val" if split != "val" else "test"
    if cfg.get(fallback) is not None:
        print(f"[WARN] Split '{split}' not found in dataset.yaml; using '{fallback}'.")
        return fallback
    print(f"[WARN] dataset.yaml has no '{split}' entry and no obvious fallback. "
          f"Trying '{split}' anyway.")
    return split


def load_names(cfg):
    """Normalise the yaml 'names' field into {index: name}."""
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def gather_split_images(data_yaml, split):
    """Return (list_of_image_paths, {idx:name}) for a split in a YOLO dataset.yaml."""
    import yaml
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = load_names(cfg)
    yaml_dir = os.path.dirname(os.path.abspath(data_yaml))
    base = cfg.get("path") or ""
    base = base if os.path.isabs(base) else os.path.normpath(os.path.join(yaml_dir, base or "."))

    entry = cfg.get(split)
    if entry is None:
        raise ValueError(f"dataset.yaml has no '{split}' entry.")
    entries = entry if isinstance(entry, (list, tuple)) else [entry]

    images = []
    for e in entries:
        p = e if os.path.isabs(e) else os.path.normpath(os.path.join(base, e))
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(IMG_EXTS):
                        images.append(os.path.join(root, fn))
        elif os.path.isfile(p) and p.lower().endswith(".txt"):
            with open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    ip = line if os.path.isabs(line) else os.path.normpath(os.path.join(base, line))
                    images.append(ip)
        elif os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
            images.append(p)
        else:
            print(f"[WARN] Could not interpret split entry: {p}")

    images = sorted(set(images))
    return images, names


def label_path_for(image_path):
    """Map .../images/.../foo.jpg -> .../labels/.../foo.txt (YOLO convention)."""
    parts = image_path.replace("\\", "/").split("/")
    # replace the LAST 'images' segment with 'labels'
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    stem = os.path.splitext(parts[-1])[0] + ".txt"
    parts[-1] = stem
    return os.path.normpath("/".join(parts))


def load_gt(image_path, w0, h0):
    """Load YOLO-format ground truth as (boxes_xyxy [N,4] px, classes [N])."""
    lp = label_path_for(image_path)
    import numpy as np
    if not os.path.isfile(lp):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    boxes, classes = [], []
    with open(lp, "r", encoding="utf-8") as f:
        for line in f:
            v = line.split()
            if len(v) < 5:
                continue
            c, cx, cy, bw, bh = int(float(v[0])), *map(float, v[1:5])
            x1 = (cx - bw / 2.0) * w0
            y1 = (cy - bh / 2.0) * h0
            x2 = (cx + bw / 2.0) * w0
            y2 = (cy + bh / 2.0) * h0
            boxes.append([x1, y1, x2, y2])
            classes.append(c)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(boxes, dtype=np.float32), np.asarray(classes, dtype=np.int64)


# ==================================================================================
# Performance monitoring (peak system-wide CPU%)
# ==================================================================================
class CPUPeakMonitor:
    """Background sampler recording peak system-wide CPU% and RAM during a run.

    One daemon thread samples both metrics every `interval` seconds:
      * peak           -> highest system-wide CPU% seen (100% = all 4 cores maxed)
      * ram_peak_used  -> highest system-wide RAM in use, in bytes
      * ram_total      -> total physical RAM as psutil sees it, in bytes
    RAM "in use" is computed as (total - available), which matches the figure
    `free -h` / htop show as used and is independent of buffers/cache. If psutil
    is unavailable the monitor is a no-op and all peaks stay None.
    """
    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = None
        self.ram_peak_used = None
        self.ram_total = None
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._psutil = psutil
            # Total RAM doesn't change during a run, so grab it once up front.
            self.ram_total = int(psutil.virtual_memory().total)
        except Exception:
            self._psutil = None

    def _run(self):
        self._psutil.cpu_percent(interval=None)
        self.peak = 0.0
        self.ram_peak_used = 0
        while not self._stop.is_set():
            c = self._psutil.cpu_percent(interval=None)
            if c > self.peak:
                self.peak = c
            vm = self._psutil.virtual_memory()
            used = int(vm.total - vm.available)   # practical "used", matches free -h
            if used > self.ram_peak_used:
                self.ram_peak_used = used
            time.sleep(self.interval)

    def start(self):
        if self._psutil is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak   # CPU peak (RAM peak is read from .ram_peak_used)


# ==================================================================================
# Pre-processing for the Hailo path (stretch resize) + coordinate inversion
# ==================================================================================
def preprocess(img_bgr, net_w, net_h, to_rgb):
    """Stretch-resize img to the network size (no aspect-ratio preservation).
    Returns (uint8 HWC, meta) where meta maps boxes back to original pixels."""
    import cv2
    import numpy as np
    h0, w0 = img_bgr.shape[:2]
    if to_rgb:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    out = cv2.resize(img_bgr, (net_w, net_h), interpolation=cv2.INTER_LINEAR)
    meta = (w0, h0)
    return np.ascontiguousarray(out, dtype=np.uint8), meta


def boxes_to_original(nboxes_xyxy, meta):
    """Map normalised [0..1] xyxy boxes straight onto the original image.
    With a stretch resize the normalised coords map directly to original pixels."""
    import numpy as np
    w0, h0 = meta
    b = nboxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] *= w0
    b[:, [1, 3]] *= h0
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w0)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h0)
    return b


# ==================================================================================
# Metrics  (self-contained, mirrors the Ultralytics matching/AP logic)
# ==================================================================================
def box_iou_np(a, b):
    import numpy as np
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def match_one_image(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr):
    """Return a [num_pred, num_iou_thr] bool 'correct' matrix for one image."""
    import numpy as np
    correct = np.zeros((len(pred_boxes), len(iou_thr)), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return correct
    iou = box_iou_np(pred_boxes, gt_boxes)
    same_cls = gt_cls[None, :] == pred_cls[:, None]
    iou = np.where(same_cls, iou, 0.0)
    for ti, thr in enumerate(iou_thr):
        yx = np.nonzero(iou >= thr)
        if yx[0].size == 0:
            continue
        m = np.stack(yx, axis=1)                      # [K,2] (pred, gt)
        if m.shape[0] > 1:
            ious = iou[m[:, 0], m[:, 1]]
            m = m[np.argsort(-ious)]                   # highest IoU first
            _, keep = np.unique(m[:, 1], return_index=True)  # one pred per gt
            m = m[keep]
            _, keep = np.unique(m[:, 0], return_index=True)  # one gt per pred
            m = m[keep]
        correct[m[:, 0], ti] = True
    return correct


def _ap(recall, precision):
    """COCO-style 101-point interpolated AP."""
    import numpy as np
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def compute_metrics(correct, conf, pred_cls, target_cls, names, iou_thr):
    """Build the same numbers the .pt path reports, from accumulated arrays."""
    import numpy as np
    order = np.argsort(-conf)
    correct, conf, pred_cls = correct[order], conf[order], pred_cls[order]
    unique_classes, n_gt = np.unique(target_cls, return_counts=True)
    niou = correct.shape[1]
    i50 = int(np.argmin(np.abs(iou_thr - 0.5)))
    i75 = int(np.argmin(np.abs(iou_thr - 0.75)))

    per_class = []
    ap_all = []
    p_all, r_all = [], []
    for ci, c in enumerate(unique_classes):
        sel = pred_cls == c
        ngt = n_gt[ci]
        if sel.sum() == 0 or ngt == 0:
            ap_row = np.zeros(niou)
            per_class.append((int(c), 0.0, 0.0, 0.0, float(ap_row.mean())))
            ap_all.append(ap_row)
            p_all.append(0.0)
            r_all.append(0.0)
            continue
        tp = correct[sel]
        tpc = tp.cumsum(0)
        fpc = (~tp).cumsum(0)
        recall = tpc / (ngt + 1e-9)
        precision = tpc / (tpc + fpc + 1e-9)
        ap_row = np.array([_ap(recall[:, j], precision[:, j]) for j in range(niou)])
        # P / R scalar taken at the max-F1 point of the IoU@0.5 curve
        pc, rc = precision[:, i50], recall[:, i50]
        f1 = 2 * pc * rc / (pc + rc + 1e-9)
        k = int(f1.argmax())
        per_class.append((int(c), float(pc[k]), float(rc[k]),
                          float(ap_row[i50]), float(ap_row.mean())))
        ap_all.append(ap_row)
        p_all.append(float(pc[k]))
        r_all.append(float(rc[k]))

    ap_all = np.array(ap_all) if ap_all else np.zeros((0, niou))
    result = {
        "precision": float(np.mean(p_all)) if p_all else 0.0,
        "recall": float(np.mean(r_all)) if r_all else 0.0,
        "map50": float(ap_all[:, i50].mean()) if len(ap_all) else 0.0,
        "map75": float(ap_all[:, i75].mean()) if len(ap_all) else 0.0,
        "map": float(ap_all.mean()) if len(ap_all) else 0.0,
        "per_class": [{
            "name": str(names.get(c, c)), "p": p, "r": r, "ap50": a50, "ap": a
        } for (c, p, r, a50, a) in per_class],
    }
    return result


# ==================================================================================
# Confusion matrix  (manual, mirrors Ultralytics' ConfusionMatrix for the Hailo path)
# ==================================================================================
class ConfusionMatrix:
    """Class-agnostic IoU-matched confusion matrix, same scheme Ultralytics uses.

    matrix[i, j] counts detections of PREDICTED class i against TRUE class j, so
    rows are predictions and columns are ground truth. The final row/column
    (index == nc) is the 'background' bin: column nc holds false positives (a
    detection with no matching label) and row nc holds false negatives (a label
    with no matching detection).

    NOTE: unlike compute_metrics()/match_one_image() in this file, matching here
    is CLASS-AGNOSTIC - a prediction can match a label of a different class. That
    is exactly what makes the off-diagonal cells show real class confusion, and
    it's what Ultralytics' own confusion matrix does. Predictions are assumed
    already confidence-filtered by the caller (run_hailo drops score < conf), so
    the matrix reuses the script's conf; the IoU threshold is passed in.
    """

    def __init__(self, nc, iou_thres=0.45):
        import numpy as np
        self.nc = int(nc)
        self.iou_thres = float(iou_thres)
        self.matrix = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)  # +1 background

    def process_batch(self, pred_boxes, pred_cls, gt_boxes, gt_cls):
        """Update the matrix with one image. Boxes are xyxy in the SAME pixel
        space (here: original-image pixels); classes are integer arrays."""
        import numpy as np
        nc = self.nc
        gt_cls = np.asarray(gt_cls, dtype=np.int64).reshape(-1)
        pred_cls = np.asarray(pred_cls, dtype=np.int64).reshape(-1)

        # No ground truth -> every detection is a false positive (background col).
        if gt_cls.shape[0] == 0:
            for dc in pred_cls:
                if 0 <= dc < nc:
                    self.matrix[dc, nc] += 1
            return

        # No detections -> every label is a false negative (background row).
        if pred_cls.shape[0] == 0:
            for gc in gt_cls:
                if 0 <= gc < nc:
                    self.matrix[nc, gc] += 1
            return

        iou = box_iou_np(gt_boxes, pred_boxes)                 # [n_gt, n_pred]
        x = np.nonzero(iou > self.iou_thres)
        if x[0].shape[0]:
            ious = iou[x[0], x[1]]
            matches = np.concatenate((np.stack(x, axis=1),
                                      ious[:, None]), axis=1)   # [K, 3] (gt, pred, iou)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]  # 1 gt/pred
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]  # 1 pred/gt
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m_gt = matches[:, 0].astype(int)
        m_pred = matches[:, 1].astype(int)

        # Each label: matched -> [pred_class, true_class]; else background row (FN).
        for i, gc in enumerate(gt_cls):
            if not (0 <= gc < nc):
                continue
            sel = m_gt == i
            if n and sel.sum() == 1:
                dc = int(pred_cls[m_pred[sel][0]])
                self.matrix[dc if 0 <= dc < nc else nc, gc] += 1
            else:
                self.matrix[nc, gc] += 1

        # Each detection with no matched label -> false positive (background col).
        # (Guarded by `if n` to mirror Ultralytics exactly.)
        if n:
            for i, dc in enumerate(pred_cls):
                if not (m_pred == i).any() and 0 <= dc < nc:
                    self.matrix[dc, nc] += 1

    def plot(self, names, save_dir, split="", normalize=True):
        """Save a matplotlib PNG (normalised by default) plus a CSV of raw counts.
        Returns the list of files written. Never raises - a plotting failure on a
        headless Pi must not bin a finished inference run."""
        import os
        import numpy as np
        nc = self.nc
        labels = [str(names.get(i, i)) for i in range(nc)] + ["background"]
        written = []
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_{split}" if split else ""

        # Always dump raw integer counts - cheap, and survives a missing matplotlib.
        try:
            csv_path = os.path.join(save_dir, f"confusion_matrix{suffix}_hailo.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("predicted\\true," + ",".join(labels) + "\n")
                for i, rl in enumerate(labels):
                    f.write(rl + "," + ",".join(str(int(v)) for v in self.matrix[i]) + "\n")
            written.append(csv_path)
        except Exception as e:
            print(f"[WARN] Could not write confusion-matrix CSV: {e}")

        try:
            import matplotlib
            matplotlib.use("Agg")          # headless-safe: no display on a Pi
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] matplotlib unavailable; skipped the plot (CSV still "
                  f"written). pip install matplotlib to get the PNG. ({e})")
            return written

        array = self.matrix.astype(np.float64)
        if normalize:
            col = array.sum(axis=0, keepdims=True)   # per true-class totals
            array = array / (col + 1e-9)             # each true column sums to 1
        array[array < 0.005] = np.nan                # blank near-empty cells

        n = nc + 1
        vmax = 1.0 if normalize else float(np.nan_to_num(np.nanmax(array)) or 1.0)
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad(color="white")

        side = max(6.0, 0.55 * n + 3.0)
        fig, ax = plt.subplots(figsize=(side, side * 0.85), dpi=200)
        im = ax.imshow(array, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title("Hailo Confusion Matrix" +
                     (" (normalised)" if normalize else " (counts)"))

        fmt = "{:.2f}" if normalize else "{:.0f}"
        cut = 0.5 * vmax
        for i in range(n):
            for j in range(n):
                v = array[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=7, color="white" if v > cut else "black")

        fig.tight_layout()
        try:
            png = os.path.join(
                save_dir,
                f"confusion_matrix_normalized{suffix}_hailo.png" if normalize
                else f"confusion_matrix{suffix}_hailo.png")
            fig.savefig(png)
            written.append(png)
        except Exception as e:
            print(f"[WARN] Could not save confusion-matrix PNG: {e}")
        finally:
            plt.close(fig)
        return written


# ==================================================================================
# Hailo backend
# ==================================================================================
def _describe(x, depth=0, max_depth=4):
    """Tiny recursive describer for debugging an unknown Hailo output structure."""
    import numpy as np
    pad = "  " * depth
    if isinstance(x, dict):
        return "\n".join(f"{pad}dict[{k!r}]:\n{_describe(v, depth + 1, max_depth)}"
                         for k, v in x.items())
    if isinstance(x, np.ndarray):
        head = f"{pad}ndarray shape={x.shape} dtype={x.dtype}"
        if x.dtype == object and x.size and depth < max_depth:
            return head + "\n" + _describe(x.reshape(-1)[0], depth + 1, max_depth)
        return head
    if isinstance(x, (list, tuple)):
        head = f"{pad}{type(x).__name__} len={len(x)}"
        if x and depth < max_depth:
            return head + "\n" + _describe(x[0], depth + 1, max_depth)
        return head
    return f"{pad}{type(x).__name__}={x!r}"


def _det_rows(elem):
    """Coerce one class's detections into an (n, 5+) float array; () if empty."""
    import numpy as np
    a = np.asarray(elem, dtype=np.float32)
    if a.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return a.reshape(-1, a.shape[-1])


def _find_per_class(x, num_classes, depth=0):
    """Descend through batch/wrapper dims and return the per-class list of
    detection arrays (length == num_classes), or None if not found."""
    import numpy as np
    if isinstance(x, np.ndarray) and x.dtype == object:
        x = list(x)              # treat object arrays as plain lists at this level
    elif isinstance(x, np.ndarray):
        # A dense numeric array is handled separately, not here.
        return None
    if isinstance(x, (list, tuple)):
        x = list(x)
        # Exact match: this level is the per-class list.
        if len(x) == num_classes:
            return x
        # Single wrapper (e.g. batch dim of 1): descend.
        if len(x) == 1 and depth < 6:
            return _find_per_class(x[0], num_classes, depth + 1)
        # Heuristic: every element looks like a detection array of width 5/6.
        if x and depth < 6:
            try:
                widths = {(_det_rows(e).shape[1]) for e in x}
                if widths and widths.issubset({5, 6}):
                    return x
            except Exception:
                pass
    return None


def parse_hailo_output(raw, num_classes, debug=False):
    """Normalise a HailoRT inference result into a list of
    (class_id, score, ymin, xmin, ymax, xmax) with coords in 0..1.

    Handles the common 'on-chip NMS' layouts, including a leading batch dim:
      * dict {out_name: value}
      * per-class list / object array (optionally wrapped in a batch dim),
        each entry an (n, 5) array of [ymin, xmin, ymax, xmax, score]
      * dense (num_classes, max_det, 5) array
      * flat (N, 6) array of [ymin, xmin, ymax, xmax, score, class]
    """
    import numpy as np
    if isinstance(raw, dict):
        if not raw:
            return []
        raw = next(iter(raw.values()))

    if debug:
        print("[DEBUG] Hailo output structure:")
        print(_describe(raw))

    dets = []

    # --- Per-class layout (the usual on-chip-NMS output), batch dim tolerated ---
    per_class = _find_per_class(raw, num_classes)
    if per_class is not None:
        for cls_id, elem in enumerate(per_class):
            arr = _det_rows(elem)
            for row in arr:
                ymin, xmin, ymax, xmax, score = row[:5]
                dets.append((cls_id, float(score), float(ymin), float(xmin),
                             float(ymax), float(xmax)))
        return dets

    # --- Dense numeric layouts ---
    arr = np.asarray(raw)
    if arr.dtype != object:
        a = arr[0] if (arr.ndim and arr.shape[0] == 1) else arr  # drop batch dim

        # (num_classes, max_det, 5)
        if a.ndim == 3 and a.shape[-1] == 5:
            for cls_id in range(a.shape[0]):
                for row in a[cls_id]:
                    ymin, xmin, ymax, xmax, score = row[:5]
                    if score <= 0:
                        continue
                    dets.append((cls_id, float(score), float(ymin), float(xmin),
                                 float(ymax), float(xmax)))
            return dets

        # (N, 6) -> [ymin, xmin, ymax, xmax, score, class]
        if a.ndim == 2 and a.shape[-1] >= 6:
            for row in a:
                ymin, xmin, ymax, xmax, score, cls_id = row[:6]
                dets.append((int(cls_id), float(score), float(ymin), float(xmin),
                             float(ymax), float(xmax)))
            return dets

    raise RuntimeError(
        "Could not interpret the Hailo output. Re-run with debug=true and send me "
        "the printed structure. The .hef may have no on-chip NMS (raw feature maps "
        "need a YOLO decode step) or an unfamiliar output layout."
    )


def run_hailo(hef_path, images, names, conf, iou, to_rgb, debug, max_images,
              save_dir=None, split=""):
    """Run the .hef over the split and return (result_dict, speed_dict, n_images).
    If save_dir is given, a confusion matrix (PNG + CSV) is written there too."""
    import numpy as np
    import cv2
    try:
        from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams,
                                     ConfigureParams, InputVStreamParams,
                                     OutputVStreamParams, FormatType)
    except Exception as e:
        raise RuntimeError(
            "The Hailo Python package (hailo_platform) is not importable. On a Pi 5 "
            "with the AI Kit this comes from the 'hailo-all' / HailoRT install. "
            f"Underlying error: {e}"
        )

    if max_images and max_images > 0:
        images = images[:max_images]
    num_classes = (max(names.keys()) + 1) if names else 80

    iou_thr = np.linspace(0.5, 0.95, 10)
    all_correct, all_conf, all_pcls, all_tcls = [], [], [], []
    cm = ConfusionMatrix(num_classes, iou_thres=iou)   # class-agnostic matrix; reuses script iou
    t_pre = t_inf = t_post = 0.0
    n = 0

    hef = HEF(hef_path)
    in_info = hef.get_input_vstream_infos()[0]
    net_h, net_w, _ = in_info.shape

    with VDevice() as target:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, cfg)[0]
        ng_params = network_group.create_params()
        in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

        # network_group.activate() is needed on some HailoRT versions, a no-op on
        # others. Fall back gracefully if it isn't supported.
        try:
            act_ctx = network_group.activate(ng_params)
        except Exception:
            act_ctx = nullcontext()

        with act_ctx:
            with InferVStreams(network_group, in_params, out_params) as pipeline:
                for idx, ipath in enumerate(images):
                    img = cv2.imread(ipath)
                    if img is None:
                        print(f"[WARN] Could not read image: {ipath}")
                        continue
                    h0, w0 = img.shape[:2]

                    t0 = time.perf_counter()
                    inp, meta = preprocess(img, net_w, net_h, to_rgb)
                    batch = inp[None, ...]  # (1, H, W, 3)
                    t1 = time.perf_counter()

                    raw = pipeline.infer({in_info.name: batch})
                    t2 = time.perf_counter()

                    dets = parse_hailo_output(raw, num_classes, debug=(debug and idx == 0))
                    # Filter by score, build arrays.
                    boxes_n, scores, classes = [], [], []
                    for cls_id, score, ymin, xmin, ymax, xmax in dets:
                        if score < conf:
                            continue
                        boxes_n.append([xmin, ymin, xmax, ymax])  # to xyxy order
                        scores.append(score)
                        classes.append(cls_id)

                    if boxes_n:
                        boxes_n = np.asarray(boxes_n, dtype=np.float32)
                        pred_boxes = boxes_to_original(boxes_n, meta)
                        pred_cls = np.asarray(classes, dtype=np.int64)
                        pred_conf = np.asarray(scores, dtype=np.float32)
                    else:
                        pred_boxes = np.zeros((0, 4), dtype=np.float32)
                        pred_cls = np.zeros((0,), dtype=np.int64)
                        pred_conf = np.zeros((0,), dtype=np.float32)

                    gt_boxes, gt_cls = load_gt(ipath, w0, h0)
                    correct = match_one_image(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr)
                    cm.process_batch(pred_boxes, pred_cls, gt_boxes, gt_cls)
                    t3 = time.perf_counter()

                    all_correct.append(correct)
                    all_conf.append(pred_conf)
                    all_pcls.append(pred_cls)
                    all_tcls.append(gt_cls)

                    t_pre += (t1 - t0)
                    t_inf += (t2 - t1)
                    t_post += (t3 - t2)
                    n += 1
                    if n % 50 == 0:
                        print(f"  ...processed {n}/{len(images)} images")

    if n == 0:
        raise RuntimeError("No images were processed - check the split path in dataset.yaml.")

    correct = np.concatenate(all_correct) if all_correct else np.zeros((0, 10), dtype=bool)
    conf_arr = np.concatenate(all_conf) if all_conf else np.zeros((0,))
    pcls = np.concatenate(all_pcls) if all_pcls else np.zeros((0,), dtype=np.int64)
    tcls = np.concatenate(all_tcls) if all_tcls else np.zeros((0,), dtype=np.int64)

    result = compute_metrics(correct, conf_arr, pcls, tcls, names, iou_thr)

    if save_dir:
        cm_files = cm.plot(names, save_dir, split=split, normalize=True)
        if cm_files:
            print("\nConfusion matrix written:")
            for p in cm_files:
                print(f"  {p}")

    speed = {
        "preprocess": 1000.0 * t_pre / n,
        "inference": 1000.0 * t_inf / n,
        "postprocess": 1000.0 * t_post / n,
    }
    return result, speed, n


# ==================================================================================
# PyTorch (.pt) backend on CPU
# ==================================================================================
def run_pt(model_path, data_yaml, split, imgsz, conf, iou, device, project, run_name):
    """Run Ultralytics validation and return (result_dict, speed_dict, save_dir)."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml, split=split, imgsz=imgsz, conf=conf, iou=iou,
        device=device, project=project, name=run_name,
        exist_ok=True, plots=True, verbose=True,
    )
    box = metrics.box
    names = metrics.names
    idxs = list(box.ap_class_index) if len(box.ap_class_index) else []
    per_class = []
    for i, ci in enumerate(idxs):
        p, r, ap50, ap = box.class_result(i)
        per_class.append({"name": str(names.get(ci, ci)),
                          "p": float(p), "r": float(r),
                          "ap50": float(ap50), "ap": float(ap)})
    result = {
        "precision": float(box.mp), "recall": float(box.mr),
        "map50": float(box.map50), "map75": float(box.map75),
        "map": float(box.map), "per_class": per_class,
    }
    speed = getattr(metrics, "speed", {}) or {}
    return result, speed, str(metrics.save_dir)


# ==================================================================================
# Reporting
# ==================================================================================
def fmt_fps(speed):
    inf = float(speed.get("inference", 0) or 0)
    e2e = sum(float(speed.get(k, 0) or 0) for k in ("preprocess", "inference", "postprocess"))
    inf_fps = (1000.0 / inf) if inf > 0 else None
    e2e_fps = (1000.0 / e2e) if e2e > 0 else None
    return inf, inf_fps, e2e, e2e_fps


def _fmt_gb(num_bytes):
    """Bytes -> '2.42 GB' style string using GiB (1024^3), matching free -h."""
    return f"{num_bytes / (1024 ** 3):.2f} GB"


def write_summary(result, speed, model_name, weights, split, backend,
                  imgsz, conf, iou, device, cpu_peak, save_dir, n_images,
                  ram_peak_used=None, ram_total=None):
    lines = []
    lines.append(f"Model:    {model_name}")
    lines.append(f"Backend:  {backend}")
    lines.append(f"Weights:  {weights}")
    lines.append(f"Split:    {split}   (images: {n_images})")
    lines.append(f"Config:   imgsz={imgsz}  conf={conf}  iou={iou}  device={device}")
    lines.append(f"Saved:    {save_dir}")
    lines.append("")
    lines.append("Overall")
    lines.append(f"  Precision (mean):  {result['precision']:.4f}")
    lines.append(f"  Recall (mean):     {result['recall']:.4f}")
    lines.append(f"  mAP@50:            {result['map50']:.4f}")
    lines.append(f"  mAP@75:            {result['map75']:.4f}")
    lines.append(f"  mAP@50-95:         {result['map']:.4f}")
    lines.append("")
    inf, inf_fps, e2e, e2e_fps = fmt_fps(speed)
    lines.append("Performance")
    if inf_fps:
        lines.append(f"  Inference only:   {inf_fps:7.2f} FPS  ({inf:.2f} ms/img)")
    if e2e_fps:
        lines.append(f"  End-to-end:       {e2e_fps:7.2f} FPS  ({e2e:.2f} ms/img incl. pre/post)")
    cpu_s = f"{cpu_peak:.0f}%" if cpu_peak is not None else "n/a"
    lines.append(f"  CPU peak:         {cpu_s}   (system-wide; 100% = all 4 cores maxed)")
    if ram_peak_used is not None and ram_total:
        ram_pct = ram_peak_used / ram_total * 100.0
        lines.append(f"  RAM peak:         {_fmt_gb(ram_peak_used)} / {_fmt_gb(ram_total)}   "
                     f"({ram_pct:.1f}% system-wide)")
    else:
        lines.append("  RAM peak:         n/a   (install psutil to enable RAM monitoring)")
    lines.append("")

    header = f"{'Class':<20}{'P':>9}{'R':>9}{'mAP50':>10}{'mAP50-95':>11}"
    lines.append("Per class")
    lines.append(header)
    lines.append("-" * len(header))
    for pc in result["per_class"]:
        lines.append(f"{pc['name']:<20}{pc['p']:>9.4f}{pc['r']:>9.4f}"
                     f"{pc['ap50']:>10.4f}{pc['ap']:>11.4f}")

    text = "\n".join(lines)

    # Save .txt + per-class .csv into save_dir.
    saved = []
    os.makedirs(save_dir, exist_ok=True)
    try:
        txt_path = os.path.join(save_dir, f"accuracy_summary_{split}_{backend_tag(backend)}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        saved.append(txt_path)
    except Exception as e:
        text += f"\n\n(Could not save summary .txt: {e})"
    try:
        csv_path = os.path.join(save_dir, f"per_class_metrics_{split}_{backend_tag(backend)}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("class,precision,recall,mAP50,mAP50-95\n")
            for pc in result["per_class"]:
                f.write(f"{pc['name']},{pc['p']:.4f},{pc['r']:.4f},"
                        f"{pc['ap50']:.4f},{pc['ap']:.4f}\n")
        saved.append(csv_path)
    except Exception:
        pass

    if saved:
        text += "\n\nFiles written by this script:\n" + "\n".join(f"  {s}" for s in saved)
    return text


def backend_tag(backend):
    return "hailo" if "Hailo" in backend else "cpu"


# ==================================================================================
# CLI Execution Entrypoint
# ==================================================================================
def main():
    params = {
        "model": "YOLOv10n",
        "data": "../Dataset/dataset.yaml",
        "split": "test",
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.7,
        "device": "cpu",       # Pi 5 has no CUDA GPU; cpu is the right default
        "hailo": False,        # hailo=true uses the .hef, hailo=false uses best.pt
        "rgb": True,           # Hailo only: feed RGB (set rgb=false if .hef wants BGR)
        "debug": False,        # Hailo only: print the raw output structure for image 0
        "max_images": 0,       # 0 = all; set a small number for a quick smoke test
    }

    def parse_bool(v):
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)
        key = key.strip()
        if key == "testfolder":
            key = "data"
        if key not in params:
            params[key] = value
            continue
        default = params[key]
        if isinstance(default, bool):       # must come before int (bool is a subclass)
            params[key] = parse_bool(value)
        elif isinstance(default, int):
            params[key] = int(value)
        elif isinstance(default, float):
            params[key] = float(value)
        else:
            params[key] = value

    model_name = params["model"]
    data_yaml = anchor(params["data"])
    split = params["split"]
    imgsz = params["imgsz"]
    conf = params["conf"]
    iou = params["iou"]
    device = str(params["device"])
    use_hailo = bool(params["hailo"])
    to_rgb = bool(params["rgb"])
    debug = bool(params["debug"])
    max_images = int(params["max_images"])

    if not os.path.exists(data_yaml):
        print(f"[ERROR] Could not find dataset config: {data_yaml}")
        sys.exit(1)
    split = resolve_split(data_yaml, split)

    project = anchor(OUTPUT_ROOT)
    # Dynamically separate outputs based on the backend used
    run_name = f"{model_name}_Hailo" if use_hailo else f"{model_name}_CPU"
    save_dir = os.path.join(project, run_name)
    
    monitor = CPUPeakMonitor().start()
    t_start = time.perf_counter()

    try:
        if use_hailo:
            hef_path = resolve_hef(model_name)
            if not hef_path or not os.path.isfile(hef_path):
                print(f"[ERROR] Could not find a .hef for '{model_name}' in: "
                      f"{anchor(MODEL_DIRS.get(model_name, 'unknown'))}")
                monitor.stop()
                sys.exit(1)

            backend = "Hailo (.hef)"
            print(f"Starting Hailo eval for {model_name}...")
            print(f"Weights:  {hef_path}")
            print(f"Data:     {data_yaml} (split: {split})")
            print(f"Resize:   stretch   rgb={to_rgb}   conf>={conf}")
            print("Loading .hef and running inference... please wait.\n")

            images, names = gather_split_images(data_yaml, split)
            if not images:
                print("[ERROR] No images found for that split.")
                monitor.stop()
                sys.exit(1)

            result, speed, n_images = run_hailo(
                hef_path, images, names, conf, iou, to_rgb, debug, max_images,
                save_dir=save_dir, split=split)
            weights = hef_path

        else:
            pt_path = resolve_pt(model_name)
            if not pt_path or not os.path.isfile(pt_path):
                print(f"[ERROR] Could not find a .pt for '{model_name}' in: "
                      f"{anchor(MODEL_DIRS.get(model_name, 'unknown'))}")
                monitor.stop()
                sys.exit(1)

            # Soft device handling: fall back to CPU instead of dying.
            if device.lower() != "cpu":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        print(f"[WARN] device='{device}' requested but no CUDA GPU is "
                              f"available (expected on a Pi 5). Falling back to CPU.")
                        device = "cpu"
                except ImportError:
                    print("[ERROR] PyTorch is not installed. Install ultralytics (which "
                          "pulls in a CPU torch build) for the hailo=false path.")
                    monitor.stop()
                    sys.exit(1)

            backend = "PyTorch (CPU)" if device.lower() == "cpu" else f"PyTorch ({device})"
            print(f"Starting CPU eval for {model_name}...")
            print(f"Weights:  {pt_path}")
            print(f"Data:     {data_yaml} (split: {split})")
            print(f"Config:   imgsz={imgsz}  conf={conf}  iou={iou}  device={device}")
            print("Loading model and running validation... please wait.\n")

            result, speed, save_dir = run_pt(
                pt_path, data_yaml, split, imgsz, conf, iou, device, project, run_name)
            weights = pt_path
            # n_images isn't directly exposed; count from the split for the report.
            try:
                imgs, _ = gather_split_images(data_yaml, split)
                n_images = len(imgs)
            except Exception:
                n_images = -1

        cpu_peak = monitor.stop()
        ram_peak_used = monitor.ram_peak_used
        ram_total = monitor.ram_total
        total_s = time.perf_counter() - t_start

        text = write_summary(result, speed, model_name, weights, split, backend,
                             imgsz, conf, iou, device, cpu_peak, save_dir, n_images,
                             ram_peak_used=ram_peak_used, ram_total=ram_total)

        print("\n" + "=" * 80)
        print(text)
        print(f"\nWall-clock total: {total_s:.1f}s")
        print("=" * 80 + "\nDone.")

    except Exception:
        monitor.stop()
        print("\n[ERROR] An exception occurred during execution:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()