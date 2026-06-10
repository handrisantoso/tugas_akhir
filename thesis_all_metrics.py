"""
thesis_all_metrics.py
=====================
SATU program untuk SEMUA metrik thesis (BAB 4-5). Cukup dijalankan 1x:

    python thesis_all_metrics.py

Menggabungkan tiga sumber:
  1. LOSO 5-fold (cross-subject)  -> Accuracy & F1 UTAMA, per-fold, mean/std/range,
                                     per-class Precision/Recall/F1 agregat
  2. Training final (PC)          -> Training Time, PC-side latency (mean/std/min/max/P95),
                                     PC FPS
  3. ESP32 on-device (terukur)    -> Flash, RAM/heap, Arena, latency on-device, FPS,
                                     Stabilitas (success rate). NILAI INI TIDAK BISA
                                     diukur oleh PC — diambil dari Serial Monitor board
                                     (lihat blok ESP32_DATA di bawah, sumber sama dengan
                                     thesis_comparison.py). Update di sini bila ukur ulang.

Catatan ilmiah:
  - Accuracy headline = LOSO float32 (jujur, menunjukkan generalisasi ke pengguna baru).
  - Drop akurasi float32 -> int8 dibahas terpisah di compare_models_loso.py.
  - "Skalabilitas" di sini = margin real-time = FPS / laju inferensi yang dibutuhkan
    (1 inferensi tiap STEP_SIZE sampel = 100/125 = 0.8 Hz). Sesuaikan definisi bila perlu.

Output: cetak tabel lengkap ke terminal + thesis_figures/thesis_all_metrics.json
"""

import os, sys, json, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier

from config import (
    GESTURE_NAMES, WINDOW_SIZE, NUM_AXES, SAMPLE_RATE_HZ, STEP_SIZE,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH,
)
from recorder import GestureRecorder
from models import (
    extract_features, fit_scaler, build_mlp, build_cnn1d, augment_windows,
    train_keras_model, train_random_forest,
)

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)
REAL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "gesture_dataset.csv")
SEED     = 42
SEP      = "=" * 78

# Laju inferensi yang dibutuhkan sistem (1 inferensi tiap jendela geser STEP_SIZE)
REQUIRED_FPS = SAMPLE_RATE_HZ / STEP_SIZE   # 100 / 125 = 0.8 Hz

# ══════════════════════════════════════════════════════════════════
#  DATA ESP32 — TERUKUR DI SERIAL MONITOR (tidak bisa diukur dari PC)
#  Sumber: firmware/*/measurements.h -> [STATS] & [STABILITY]
#  total/fail dipakai untuk Stabilitas (success rate).
#
#  flash_kb = ukuran MODEL yang tertanam di flash (bukan ukuran teks file .h):
#    MLP   = model_data_len 5856 B   = 5.7 KB  (TFLite int8 binary)
#    CNN1D = model_data_len 205520 B = 200.7 KB (TFLite int8 binary)
#    RF    = kode C m2cgen (bukan array byte) -> ukuran flash HANYA dari
#            output build Arduino IDE ("Sketch uses N bytes"). Isi manual.
# ══════════════════════════════════════════════════════════════════
# Hasil compile Arduino IDE (XIAO ESP32-S3). Flash & RAM statis TOTAL FIRMWARE
# (model + core + driver MPU6050 + BLE/USB HID). RAM max = 327680 B (320 KB).
#   RF   : Sketch 453415 B | global 28564 B  (8%)
#   MLP  : Sketch 376943 B | global 131420 B (40%)
#   CNN1D: Sketch 685667 B | global 248644 B (75%)
ESP32_RAM_MAX_B = 327680

# Dua metrik flash terpisah (jangan dicampur):
#   model_flash_kb    = ukuran MODEL saja. RF=None (ter-inline sbg kode C, tak terpisah).
#   firmware_flash_kb = FLASH TOTAL FIRMWARE dari "Sketch uses N bytes" Arduino IDE
#                       (model + core + driver + HID). Isi setelah compile tiap .ino.
# heap_bytes   = free heap runtime (ESP.getFreeHeap()), diukur stabil tiap invoke
#                (arena TFLite Micro statis -> tak ada malloc saat Invoke()).
# heap_size_b  = total heap region (ESP.getHeapSize()); None bila belum diukur ulang.
# arena_kb     = tensor arena statis (di internal DRAM), konsumen RAM utama model.
# CNN1D: diperbarui dari run baru COM7 (XIAO ESP32-S3) -> fail=0 (100% stabil),
#        free heap 116 KB (build dgn layout memori teroptimasi, bukan 10 KB lama).
ESP32_DATA = {
    "RF": {
        # latency/heap dari run baru COM7 (firmware fixed float + TANPA BLE).
        # firmware_flash_b/static_ram_b MASIH era-BLE -> perlu "Sketch uses" baru.
        "latency_mean_us": 2200, "latency_min_us": 1749, "latency_max_us": 3541,
        "fps": 454, "heap_bytes": 348512, "heap_size_b": 384060,
        "model_flash_kb": None, "firmware_flash_b": 453415, "static_ram_b": 28564,
        "arena_kb": 0, "total_count": 100, "fail_count": 0, "model_format": "pure C (m2cgen)",
    },
    "MLP": {
        "latency_mean_us": 138, "latency_min_us": 112, "latency_max_us": 258,
        "fps": 7262, "heap_bytes": 245924, "heap_size_b": 281260,
        "model_flash_kb": 5.7, "firmware_flash_b": 376943, "static_ram_b": 131420,
        "arena_kb": 100, "total_count": 100, "fail_count": 0, "model_format": "TFLite Micro int8",
    },
    "CNN1D": {
        "latency_mean_us": 106228, "latency_min_us": 106194, "latency_max_us": 106294,
        "fps": 9, "heap_bytes": 116472, "heap_size_b": 164036,
        "model_flash_kb": 200.7, "firmware_flash_b": 685667, "static_ram_b": 248644,
        "arena_kb": 220, "total_count": 100, "fail_count": 0, "model_format": "TFLite Micro int8",
    },
}

MODELS = ["RF", "MLP", "CNN1D"]


# ══════════════════════════════════════════════════════════════════
#  1. LOSO  — accuracy/F1 utama, per-fold, per-class agregat
# ══════════════════════════════════════════════════════════════════
def run_loso(X, y, subj):
    """Leave-One-Subject-Out untuk RF, MLP, CNN1D (float32).

    Pipeline = deployment: augment train -> fitur/scaler -> model.
    Mengumpulkan metrik per-fold + prediksi tergabung untuk per-class agregat.
    """
    import random, tensorflow as tf
    from tensorflow import keras as K
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
    aug_rng = np.random.default_rng(SEED)

    subjects = list(np.unique(subj))
    print(f"\n[1] LOSO {len(subjects)}-fold  | subjects: {subjects}")

    out = {m: {"fold_acc": [], "fold_f1": [], "fold_prec": [], "fold_rec": [],
               "y_true": [], "y_pred": [], "subjects": []} for m in MODELS}

    splitter = LeaveOneGroupOut()
    for k, (tr, te) in enumerate(splitter.split(X, y, groups=subj)):
        held = str(np.unique(subj[te])[0])
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        print(f"    FOLD {k}  held-out={held}  (train={len(y_tr)} test={len(y_te)})")

        X_tr_aug, y_tr_aug = augment_windows(X_tr, y_tr, rng=aug_rng, n_copies=5)
        early = K.callbacks.EarlyStopping(monitor="loss", patience=10,
                                          restore_best_weights=True, verbose=0)

        # Fitur 36-dim untuk RF & MLP
        Xtr_f = extract_features(X_tr_aug)
        Xte_f = extract_features(X_te)
        sc_f = fit_scaler(Xtr_f)
        Xtr_s, Xte_s = sc_f.transform(Xtr_f), sc_f.transform(Xte_f)

        # ── RF ──
        rf = RandomForestClassifier(n_estimators=DEFAULT_RF_TREES, max_depth=DEFAULT_RF_DEPTH,
                                    random_state=SEED, n_jobs=-1)
        rf.fit(Xtr_s, y_tr_aug)
        _record_fold(out["RF"], y_te, rf.predict(Xte_s), held)

        # ── MLP ──
        mlp = build_mlp(input_dim=Xtr_s.shape[1])
        mlp.fit(Xtr_s, y_tr_aug, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                verbose=0, callbacks=[early])
        _record_fold(out["MLP"], y_te, np.argmax(mlp.predict(Xte_s, verbose=0), axis=1), held)

        # ── CNN1D (raw 1500 -> scaler -> (250,6)) ──
        sc_c = fit_scaler(X_tr_aug)
        Xtr_c = sc_c.transform(X_tr_aug).reshape(-1, WINDOW_SIZE, NUM_AXES)
        Xte_c = sc_c.transform(X_te).reshape(-1, WINDOW_SIZE, NUM_AXES)
        cnn = build_cnn1d()
        cnn.fit(Xtr_c, y_tr_aug, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
                verbose=0, callbacks=[early])
        _record_fold(out["CNN1D"], y_te, np.argmax(cnn.predict(Xte_c, verbose=0), axis=1), held)

        K.backend.clear_session()
        print(f"        acc -> RF={out['RF']['fold_acc'][-1]:.4f}  "
              f"MLP={out['MLP']['fold_acc'][-1]:.4f}  CNN1D={out['CNN1D']['fold_acc'][-1]:.4f}")

    return out


def _record_fold(d, y_true, y_pred, subject):
    d["fold_acc"].append(accuracy_score(y_true, y_pred))
    d["fold_f1"].append(f1_score(y_true, y_pred, average="macro", zero_division=0))
    d["fold_prec"].append(precision_score(y_true, y_pred, average="macro", zero_division=0))
    d["fold_rec"].append(recall_score(y_true, y_pred, average="macro", zero_division=0))
    d["y_true"].extend(list(y_true)); d["y_pred"].extend(list(y_pred))
    d["subjects"].append(subject)


def aggregate_loso(loso):
    """Ringkas LOSO: mean/std/range + per-class agregat (dari prediksi tergabung)."""
    agg = {}
    labels = list(range(len(GESTURE_NAMES)))
    for m in MODELS:
        d = loso[m]
        accs = np.array(d["fold_acc"]); f1s = np.array(d["fold_f1"])
        yt, yp = np.array(d["y_true"]), np.array(d["y_pred"])
        rep = classification_report(yt, yp, labels=labels, target_names=GESTURE_NAMES,
                                    output_dict=True, zero_division=0)
        agg[m] = {
            "mean_acc": float(accs.mean()), "std_acc": float(accs.std()),
            "min_acc": float(accs.min()),   "max_acc": float(accs.max()),
            "mean_f1_macro": float(f1s.mean()), "std_f1_macro": float(f1s.std()),
            "mean_prec_macro": float(np.mean(d["fold_prec"])),
            "mean_rec_macro":  float(np.mean(d["fold_rec"])),
            "fold_acc": [round(a, 4) for a in d["fold_acc"]],
            "fold_subjects": d["subjects"],
            "per_class": {g: {"precision": round(rep[g]["precision"], 4),
                              "recall":    round(rep[g]["recall"], 4),
                              "f1":        round(rep[g]["f1-score"], 4),
                              "support":   int(rep[g]["support"])} for g in GESTURE_NAMES},
        }
    return agg


# ══════════════════════════════════════════════════════════════════
#  2. TRAINING FINAL + LATENCY PC
# ══════════════════════════════════════════════════════════════════

# DLL C m2cgen (dari pc_bench/rf_score_wrapper.c, dikompilasi dengan MSVC)
_RF_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "pc_bench", "rf_score.dll")


def _bench_rf_c(Xt_scaled, n_warmup=50, n_runs=100):
    """Ukur latency RF via kode C m2cgen (identik firmware ESP32).
    Kembalikan dict atau None bila DLL tidak ditemukan.
    """
    import ctypes
    if not os.path.isfile(_RF_DLL_PATH):
        return None
    try:
        lib = ctypes.CDLL(_RF_DLL_PATH)
        lib.rf_score.argtypes = [ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float)]
        lib.rf_score.restype = None
    except Exception as e:
        print(f"    [WARN] Tidak bisa load rf_score.dll: {e}")
        return None

    n = min(n_runs, len(Xt_scaled))
    arr = np.ascontiguousarray(Xt_scaled[:n], dtype=np.float32)
    out = np.zeros(5, dtype=np.float32)

    def _call(i):
        lib.rf_score(arr[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    for i in range(n_warmup):
        _call(i % n)

    lat, t_all = [], time.perf_counter()
    for i in range(n):
        t = time.perf_counter(); _call(i)
        lat.append((time.perf_counter() - t) * 1000.0)
    elapsed = time.perf_counter() - t_all
    lat = np.array(lat)
    return {
        "mean": float(lat.mean()), "std": float(lat.std()),
        "min":  float(lat.min()),  "max": float(lat.max()),
        "p95":  float(np.percentile(lat, 95)),
        "fps":  n / elapsed if elapsed > 0 else 0,
    }


def measure_pc(X, y):
    """Latih model deploy (split standar) -> training time, lalu ukur latency PC-side.
    RF: implementasi C m2cgen via ctypes (setara ESP32). MLP/CNN1D: tf.function.
    """
    print("\n[2] Training final (PC) + ukur Training Time...")

    t0 = time.time(); rf = train_random_forest(X, y); rf["train_time"] = time.time() - t0
    print(f"    RF    trained in {rf['train_time']:.1f}s")
    t0 = time.time(); mlp = train_keras_model("mlp", X, y); mlp["train_time"] = time.time() - t0
    print(f"    MLP   trained in {mlp['train_time']:.1f}s")
    t0 = time.time(); cnn = train_keras_model("cnn1d", X, y); cnn["train_time"] = time.time() - t0
    print(f"    CNN1D trained in {cnn['train_time']:.1f}s")
    res = {"RF": rf, "MLP": mlp, "CNN1D": cnn}

    print("\n[3] Ukur PC-side latency (batch=1, warmup)...")
    import tensorflow as tf
    n_warmup = 50

    # ── RF: benchmark C m2cgen (identik firmware) ──────────────────
    r = res["RF"]
    c_res = _bench_rf_c(r["X_test"], n_warmup=n_warmup)
    if c_res:
        r["pc_lat_mean"] = c_res["mean"]; r["pc_lat_std"] = c_res["std"]
        r["pc_lat_min"]  = c_res["min"];  r["pc_lat_max"] = c_res["max"]
        r["pc_lat_p95"]  = c_res["p95"];  r["pc_fps"]     = c_res["fps"]
        r["pc_lat_method"] = "C m2cgen (ctypes)"
        print(f"    RF  [C m2cgen]: mean={r['pc_lat_mean']:.4f}ms "
              f"p95={r['pc_lat_p95']:.4f}ms fps={r['pc_fps']:.1f}")
    else:
        print("    RF  [WARN] rf_score.dll tidak ditemukan — fallback ke sklearn.predict")
        Xt_rf = r["X_test"]; n = min(100, len(Xt_rf))
        for i in range(n_warmup): r["model"].predict(Xt_rf[i % n:i % n + 1])
        lat, t_all = [], time.perf_counter()
        for i in range(n):
            t = time.perf_counter(); r["model"].predict(Xt_rf[i:i+1])
            lat.append((time.perf_counter() - t) * 1000.0)
        elapsed = time.perf_counter() - t_all; lat = np.array(lat)
        r["pc_lat_mean"] = float(lat.mean()); r["pc_lat_std"] = float(lat.std())
        r["pc_lat_min"]  = float(lat.min());  r["pc_lat_max"] = float(lat.max())
        r["pc_lat_p95"]  = float(np.percentile(lat, 95))
        r["pc_fps"]      = n / elapsed if elapsed > 0 else 0
        r["pc_lat_method"] = "sklearn.predict (fallback)"
        print(f"    RF  [sklearn]: mean={r['pc_lat_mean']:.2f}ms p95={r['pc_lat_p95']:.2f}ms "
              f"fps={r['pc_fps']:.1f}")

    # ── MLP + CNN1D: tf.function ────────────────────────────────────
    for name in ("MLP", "CNN1D"):
        r = res[name]
        Xt = r["X_test"]; n = min(100, len(Xt))
        infer = tf.function(r["model"], reduce_retracing=True)
        for i in range(n_warmup): infer(Xt[i % n:i % n + 1], training=False)
        lat, t_all = [], time.perf_counter()
        for i in range(n):
            t = time.perf_counter(); infer(Xt[i:i+1], training=False)
            lat.append((time.perf_counter() - t) * 1000.0)
        elapsed = time.perf_counter() - t_all
        r["pc_lat_mean"] = float(np.mean(lat)); r["pc_lat_std"] = float(np.std(lat))
        r["pc_lat_min"]  = float(np.min(lat));  r["pc_lat_max"] = float(np.max(lat))
        r["pc_lat_p95"]  = float(np.percentile(lat, 95))
        r["pc_fps"]      = n / elapsed if elapsed > 0 else 0
        r["pc_lat_method"] = "tf.function"
        print(f"    {name}: mean={r['pc_lat_mean']:.2f}ms p95={r['pc_lat_p95']:.2f}ms "
              f"fps={r['pc_fps']:.1f}")
    return res


# ══════════════════════════════════════════════════════════════════
#  3. PROFIL ESP32 (terukur)
# ══════════════════════════════════════════════════════════════════
def esp32_profile():
    prof = {}
    for m in MODELS:
        d = ESP32_DATA[m]
        ok = d["total_count"] - d["fail_count"]
        prof[m] = {
            "latency_mean_ms": d["latency_mean_us"] / 1000.0,
            "latency_min_ms":  d["latency_min_us"] / 1000.0,
            "latency_max_ms":  d["latency_max_us"] / 1000.0,
            "fps": d["fps"],
            "model_flash_kb": d["model_flash_kb"],
            "firmware_flash_kb": d["firmware_flash_b"] / 1024.0,
            "static_ram_kb": d["static_ram_b"] / 1024.0,
            "static_ram_pct": 100.0 * d["static_ram_b"] / ESP32_RAM_MAX_B,
            "free_heap_kb": d["heap_bytes"] / 1024.0, "arena_kb": d["arena_kb"],
            "heap_size_kb": (d["heap_size_b"] / 1024.0) if d["heap_size_b"] else None,
            "heap_used_kb": ((d["heap_size_b"] - d["heap_bytes"]) / 1024.0)
                            if d["heap_size_b"] else None,
            "stability_pct": 100.0 * ok / d["total_count"] if d["total_count"] else 0,
            "total_count": d["total_count"], "fail_count": d["fail_count"],
            "scalability_x": d["fps"] / REQUIRED_FPS,   # margin real-time
            "model_format": d["model_format"],
        }
    return prof


# ══════════════════════════════════════════════════════════════════
#  CETAK TABEL
# ══════════════════════════════════════════════════════════════════
def print_report(agg, pc, esp):
    def row(label, vals, fmt):
        print(f"  {label:<34}" + "".join(f"{fmt.format(v):>14}" for v in vals))

    print(f"\n{SEP}\n  TABEL UTAMA — RINGKASAN (copy ke thesis)\n{SEP}")
    print(f"  {'Metrik':<34}{'RF':>14}{'MLP':>14}{'CNN1D':>14}")
    print("  " + "-" * 74)
    print("  -- Akurasi & F1 (LOSO 5-fold, cross-subject) --")
    row("Mean Accuracy",          [agg[m]["mean_acc"]*100 for m in MODELS], "{:.2f}%")
    row("Std Akurasi antar Fold", [agg[m]["std_acc"]*100 for m in MODELS], "{:.2f}%")
    row("Range Fold Acc (min-max)",
        [f"{agg[m]['min_acc']*100:.1f}-{agg[m]['max_acc']*100:.1f}" for m in MODELS], "{}")
    row("Mean F1-Score (macro)",  [agg[m]["mean_f1_macro"]*100 for m in MODELS], "{:.2f}%")
    row("Mean Precision (macro)", [agg[m]["mean_prec_macro"]*100 for m in MODELS], "{:.2f}%")
    row("Mean Recall (macro)",    [agg[m]["mean_rec_macro"]*100 for m in MODELS], "{:.2f}%")
    print("  -- Kinerja PC-Side --")
    row("PC Latency mean (ms)",   [pc[m]["pc_lat_mean"] for m in MODELS], "{:.2f}")
    row("PC Latency P95 (ms)",    [pc[m]["pc_lat_p95"] for m in MODELS], "{:.2f}")
    row("PC FPS",                 [pc[m]["pc_fps"] for m in MODELS], "{:.1f}")
    row("Training Time (s)",      [pc[m]["train_time"] for m in MODELS], "{:.1f}")
    print("  -- ESP32 On-Device (terukur) --")
    row("ESP32 Latency mean (ms)",[esp[m]["latency_mean_ms"] for m in MODELS], "{:.2f}")
    row("ESP32 FPS",              [esp[m]["fps"] for m in MODELS], "{:.0f}")
    # Dua baris flash terpisah; None tampil 'inline'/'build*'
    def flash_row(label, key, none_text):
        cells = []
        for m in MODELS:
            v = esp[m][key]
            cells.append(f"{(none_text if v is None else f'{v:.1f}'):>14}")
        print(f"  {label:<34}" + "".join(cells))
    flash_row("ESP32 Model size (KB)", "model_flash_kb", "inline")
    row("ESP32 Firmware flash (KB)",   [esp[m]["firmware_flash_kb"] for m in MODELS], "{:.1f}")
    print("  -- ESP32 Memori (RAM) --")
    row("ESP32 Tensor arena (KB)",     [esp[m]["arena_kb"] for m in MODELS], "{:.0f}")
    row("ESP32 RAM statis total (KB)", [esp[m]["static_ram_kb"] for m in MODELS], "{:.1f}")
    row("ESP32 RAM statis (%)",        [esp[m]["static_ram_pct"] for m in MODELS], "{:.1f}%")
    # heap terpakai (heap_size - free); None bila heap_size belum diukur ulang
    cells = []
    for m in MODELS:
        v = esp[m]["heap_used_kb"]
        cells.append(f"{('n/a*' if v is None else f'{v:.1f}'):>14}")
    print(f"  {'ESP32 Heap terpakai (KB)':<34}" + "".join(cells))
    row("ESP32 Free heap runtime (KB)",[esp[m]["free_heap_kb"] for m in MODELS], "{:.0f}")
    print("  -- ESP32 Kinerja --")
    row("Stabilitas ESP32 (%)",        [esp[m]["stability_pct"] for m in MODELS], "{:.1f}%")
    row("Skalabilitas (x real-time)",  [esp[m]["scalability_x"] for m in MODELS], "{:.0f}x")
    print(SEP)

    # Akurasi per-fold
    n_folds = len(agg["RF"]["fold_acc"])
    subj = agg["RF"]["fold_subjects"]
    print(f"\n{SEP}\n  AKURASI PER-FOLD (LOSO)\n{SEP}")
    hdr = f"  {'Model':<8}" + "".join(f"{'F'+str(i)+'('+subj[i][:5]+')':>14}" for i in range(n_folds))
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for m in MODELS:
        print(f"  {m:<8}" + "".join(f"{a*100:>13.2f}%" for a in agg[m]["fold_acc"]))

    # Per-class agregat
    print(f"\n{SEP}\n  METRIK PER KELAS — Precision/Recall/F1 AGREGAT LOSO\n{SEP}")
    for m in MODELS:
        print(f"\n  [{m}]")
        print(f"    {'Gesture':<14}{'Precision':>11}{'Recall':>11}{'F1':>11}{'Support':>10}")
        for g in GESTURE_NAMES:
            c = agg[m]["per_class"][g]
            print(f"    {g:<14}{c['precision']:>11.3f}{c['recall']:>11.3f}"
                  f"{c['f1']:>11.3f}{c['support']:>10d}")


def save_json(agg, pc, esp):
    summary = {}
    for m in MODELS:
        summary[m] = {
            "loso": agg[m],
            "pc": {
                **{k: round(pc[m][k], 4) for k in
                   ("pc_lat_mean", "pc_lat_std", "pc_lat_min", "pc_lat_max",
                    "pc_lat_p95", "pc_fps", "train_time")},
                "pc_lat_method": pc[m].get("pc_lat_method", ""),
            },
            "esp32": {k: (round(v, 2) if isinstance(v, float) else v)
                      for k, v in esp[m].items()},
        }
    path = os.path.join(OUT_DIR, "thesis_all_metrics.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] JSON tersimpan: {path}")


# ══════════════════════════════════════════════════════════════════
def main():
    print(SEP); print("  GESTURE GLOVE — SEMUA METRIK THESIS (LOSO + PC + ESP32)"); print(SEP)
    csv = REAL_CSV if os.path.exists(REAL_CSV) else os.path.join(
        os.path.dirname(__file__), "gesture_dataset.csv")
    print(f"  Dataset: {csv}")
    X, y, subj = GestureRecorder.load_dataset(csv)
    print(f"  {len(y)} windows, {len(np.unique(subj))} subjek")

    loso = run_loso(X, y, subj)
    agg  = aggregate_loso(loso)
    pc   = measure_pc(X, y)
    esp  = esp32_profile()

    print_report(agg, pc, esp)
    save_json(agg, pc, esp)


if __name__ == "__main__":
    main()
