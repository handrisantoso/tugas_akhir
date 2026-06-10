"""
bench_rf_c.py — ukur latency RF di PC memakai implementasi C m2cgen yang IDENTIK
dengan firmware ESP32 (via ctypes), bukan sklearn.predict().

Ini menjadikan perbandingan RF PC vs ESP32 benar-benar setara (C-vs-C):
  - Input  : 36 fitur statistik yang SUDAH di-scale (sama seperti firmware).
  - Diukur : hanya panggilan score() (kernel inferensi), preprocessing di luar timer.

Juga memverifikasi argmax(C) == sklearn.predict agar terbukti modelnya sama.

Jalankan:
    python pc_bench/bench_rf_c.py
"""
import os, sys, time, ctypes
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR
from recorder import GestureRecorder
from models import extract_features, split_dataset
from config import DEFAULT_CSV

DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rf_score.dll")


def load_c_lib():
    lib = ctypes.CDLL(DLL_PATH)
    lib.rf_score.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.rf_score.restype = None
    return lib


def c_score(lib, x36):
    """Panggil score() C untuk satu sampel (36 float ter-scale) -> 5 vote."""
    out = np.zeros(5, dtype=np.float32)
    inp = np.ascontiguousarray(x36, dtype=np.float32)
    lib.rf_score(inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return out


def main(n_runs=100, n_warmup=50):
    print("=" * 62)
    print("  RF PC LATENCY — implementasi C m2cgen (setara ESP32)")
    print("=" * 62)

    # ── data uji: split yg sama dgn training, fitur ter-scale ──
    csv = DEFAULT_CSV if os.path.exists(DEFAULT_CSV) else "dataset/gesture_dataset.csv"
    X, y, _ = GestureRecorder.load_dataset(csv)
    _, _, X_test_raw, _, _, y_test = split_dataset(X, y)
    scaler = joblib.load(os.path.join(MODELS_DIR, "rf_scaler.joblib"))
    rf     = joblib.load(os.path.join(MODELS_DIR, "rf_model.joblib"))

    feats  = extract_features(X_test_raw)
    scaled = scaler.transform(feats).astype(np.float32)
    n = min(n_runs, len(scaled))
    print(f"  Sampel uji: {n}  (fitur 36-dim, sudah di-scale)")

    lib = load_c_lib()

    # ── verifikasi: argmax(C) == sklearn.predict ──
    sk_pred = rf.predict(scaled[:n])
    c_pred  = np.array([int(np.argmax(c_score(lib, scaled[i]))) for i in range(n)])
    agree = int(np.sum(sk_pred == c_pred))
    print(f"  Verifikasi C vs sklearn: {agree}/{n} cocok "
          f"({100.0*agree/n:.1f}%)  {'OK' if agree == n else 'PERIKSA!'}")

    # ── warmup ──
    for i in range(n_warmup):
        c_score(lib, scaled[i % n])

    # ── benchmark: hanya score() per sampel (batch=1, sama spt ESP32) ──
    lat = []
    t_all = time.perf_counter()
    for i in range(n):
        t = time.perf_counter()
        c_score(lib, scaled[i])
        lat.append((time.perf_counter() - t) * 1000.0)  # ms
    elapsed = time.perf_counter() - t_all
    lat = np.array(lat)

    print("-" * 62)
    print(f"  RF (C m2cgen) PC latency  [batch=1, kernel saja]")
    print(f"    mean = {lat.mean():.4f} ms")
    print(f"    std  = {lat.std():.4f} ms")
    print(f"    min  = {lat.min():.4f} ms")
    print(f"    max  = {lat.max():.4f} ms")
    print(f"    p95  = {np.percentile(lat, 95):.4f} ms")
    print(f"    fps  = {n / elapsed:.1f}")
    print("=" * 62)
    print("  Bandingkan dgn sklearn.predict (~20 ms) -> overhead Python/numpy.")
    print("  ESP32 (C, sama) = 1.17 ms. Kini PC pakai kode C yang sama.")


if __name__ == "__main__":
    main()
