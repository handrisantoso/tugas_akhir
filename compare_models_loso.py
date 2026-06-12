"""
compare_models_loso.py
======================
Perbandingan float32 vs int8 untuk SEMUA model (MLP, CNN1D, RF) lewat LOSO.
Metrik: accuracy, precision, recall, F1 (weighted) + ukuran model + acc drop.

Pipeline tiap model SAMA PERSIS dengan firmware deploy:
  MLP  : RAW -> extract_features(36) -> StandardScaler(36) -> TFLite (f32/int8)
  CNN1D: RAW 1500 -> StandardScaler(1500) -> TFLite (f32/int8)
  RF   : RAW -> extract_features(36) -> StandardScaler(36) -> float32 (no int8)

Default dataset: real subjects only (cross-subject jujur). Ganti DATASET_CSV
ke gesture_dataset_augmented.csv jika ingin yang augmented (lihat catatan leakage).

Usage:
    python compare_models_loso.py
    python compare_models_loso.py --augmented      # pakai dataset augmented
"""

import os, json, time, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from recorder import GestureRecorder
from models import (
    extract_features, fit_scaler, build_mlp, build_cnn1d,
)
from config import (
    GESTURE_NAMES, WINDOW_SIZE, NUM_AXES,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_RF_TREES, DEFAULT_RF_DEPTH,
)

REAL_CSV = "dataset/gesture_dataset.csv"
AUG_CSV  = "dataset/gesture_dataset_augmented.csv"
N_BENCH  = 100
SEP = "=" * 78


# ─── TFLite helpers ──────────────────────────────────────────────────────
def to_tflite_f32(model):
    import tensorflow as tf
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()

def to_tflite_int8(model, rep):
    import tensorflow as tf
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    def gen():
        for i in range(min(200, len(rep))):
            yield [rep[i:i+1].astype(np.float32)]
    c.representative_dataset = gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type  = tf.int8
    c.inference_output_type = tf.int8
    return c.convert()

def interp_predict(buf, X):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=buf)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds = []
    for x in X:
        xi = x[np.newaxis]
        if inp["dtype"] == np.int8:
            s, z = inp["quantization"]
            xi = np.clip(np.round(xi / s + z), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], xi)
        it.invoke()
        raw = it.get_tensor(out["index"])
        if out["dtype"] == np.int8:
            s, z = out["quantization"]
            raw = (raw.astype(np.float32) - z) * s
        preds.append(int(np.argmax(raw)))
    return np.array(preds)


def metrics(y_true, y_pred):
    return dict(
        acc=float(accuracy_score(y_true, y_pred)),
        prec=float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        rec=float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    )

def mean_std(lst, key):
    v = np.array([d[key] for d in lst], float)
    return float(v.mean()), float(v.std())


# ─── main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augmented", action="store_true",
                    help="pakai dataset augmented (6-fold, ada leakage)")
    ap.add_argument("--csv", type=str, default=None,
                    help="path CSV custom, misal dataset/gesture_dataset_6subj.csv")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for reproducible training (default: 42)")
    args = ap.parse_args()

    import random, tensorflow as tf
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = AUG_CSV if args.augmented else REAL_CSV
    X, y, subj = GestureRecorder.load_dataset(csv_path)
    subjects = np.unique(subj)

    print(SEP)
    print("  PERBANDINGAN MODEL — Float32 vs INT8 (LOSO)")
    print(SEP)
    print(f"  Dataset : {csv_path}")
    print(f"  Windows : {len(y)}  |  Subjects: {list(subjects)}")
    if args.augmented:
        print("  !! CATATAN: dataset augmented punya leakage subjek sintetis -> akurasi ~100%")

    if args.augmented:
        splitter = StratifiedGroupKFold(n_splits=len(subjects), shuffle=True, random_state=42)
        split_iter = splitter.split(X, y, groups=subj)
    else:
        split_iter = LeaveOneGroupOut().split(X, y, groups=subj)

    from tensorflow import keras as K

    # results[model][format] = list of per-fold metric dicts
    results = {m: {"f32": [], "int8": []} for m in ("mlp", "cnn1d")}
    results["rf"] = {"f32": []}
    sizes = {}        # model -> {"f32": bytes, "int8": bytes}
    fold_subjects = []  # held-out subject label per fold

    for k, (tr, te) in enumerate(split_iter):
        held = str(np.unique(subj[te])[0])   # actual held-out subject from test indices
        fold_subjects.append(held)
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        print(f"\n  FOLD {k}  held-out={held}  (train={len(y_tr)} test={len(y_te)})")

        # NO data augmentation — train on raw windows, identical to deployment
        # pipeline so the float32->int8 drop is measured on the real model.
        X_tr_aug, y_tr_aug = X_tr, y_tr
        early = K.callbacks.EarlyStopping(monitor="loss", patience=10,
                                          restore_best_weights=True, verbose=0)

        # ── MLP ──────────────────────────────────────────────────────────
        # Deployment pipeline: RAW -> 36 features -> scaler
        X_tr_f = extract_features(X_tr_aug)
        X_te_f = extract_features(X_te)
        sc = fit_scaler(X_tr_f)
        X_tr_s, X_te_s = sc.transform(X_tr_f), sc.transform(X_te_f)
        mlp = build_mlp(input_dim=X_tr_s.shape[1])
        mlp.compile(optimizer=K.optimizers.Adam(1e-3),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        mlp.fit(X_tr_s, y_tr_aug, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, verbose=0, callbacks=[early])
        b_f32 = to_tflite_f32(mlp)
        b_i8  = to_tflite_int8(mlp, X_tr_s)
        results["mlp"]["f32"].append(metrics(y_te, interp_predict(b_f32, X_te_s)))
        results["mlp"]["int8"].append(metrics(y_te, interp_predict(b_i8,  X_te_s)))
        if k == 0:   # model size is fixed by architecture; record once
            sizes["mlp"] = {"f32": len(b_f32), "int8": len(b_i8)}
        print(f"    MLP   f32 acc={results['mlp']['f32'][-1]['acc']:.4f}  "
              f"int8 acc={results['mlp']['int8'][-1]['acc']:.4f}")

        # ── CNN1D ────────────────────────────────────────────────────────
        # Deployment pipeline: RAW -> StandardScaler(1500)
        sc_c = fit_scaler(X_tr_aug)
        X_tr_c = sc_c.transform(X_tr_aug).reshape(-1, WINDOW_SIZE, NUM_AXES)
        X_te_c = sc_c.transform(X_te).reshape(-1, WINDOW_SIZE, NUM_AXES)
        cnn = build_cnn1d()
        cnn.compile(optimizer=K.optimizers.Adam(1e-3),
                    loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        cnn.fit(X_tr_c, y_tr_aug, epochs=DEFAULT_EPOCHS,
                batch_size=DEFAULT_BATCH_SIZE, verbose=0, callbacks=[early])
        b_f32 = to_tflite_f32(cnn)
        b_i8  = to_tflite_int8(cnn, X_tr_c)
        results["cnn1d"]["f32"].append(metrics(y_te, interp_predict(b_f32, X_te_c)))
        results["cnn1d"]["int8"].append(metrics(y_te, interp_predict(b_i8,  X_te_c)))
        if k == 0:   # model size is fixed by architecture; record once
            sizes["cnn1d"] = {"f32": len(b_f32), "int8": len(b_i8)}
        print(f"    CNN1D f32 acc={results['cnn1d']['f32'][-1]['acc']:.4f}  "
              f"int8 acc={results['cnn1d']['int8'][-1]['acc']:.4f}")

        # ── RF (float only) ──────────────────────────────────────────────
        rf = RandomForestClassifier(n_estimators=DEFAULT_RF_TREES,
                                    max_depth=DEFAULT_RF_DEPTH,
                                    random_state=args.seed, n_jobs=-1)
        rf.fit(X_tr_s, y_tr_aug)
        results["rf"]["f32"].append(metrics(y_te, rf.predict(X_te_s)))
        print(f"    RF    f32 acc={results['rf']['f32'][-1]['acc']:.4f}  (no int8)")

    # ── tabel ringkasan ──────────────────────────────────────────────────
    def fmt_pair(lst, key):
        m, s = mean_std(lst, key)
        return f"{m*100:.2f}% +/-{s*100:.2f}%"

    print(f"\n{SEP}")
    print("  RINGKASAN LOSO (mean +/- std semua fold)")
    print(SEP)
    hdr = f"  {'Model':<7} {'Format':<8} {'Accuracy':<18} {'Precision':<18} {'Recall':<18} {'F1':<18}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    table_rows = []
    for model, fmts in [("mlp", ["f32", "int8"]),
                        ("cnn1d", ["f32", "int8"]),
                        ("rf", ["f32"])]:
        for fmt in fmts:
            lst = results[model][fmt]
            fmt_label = "Float32" if fmt == "f32" else "INT8"
            row = {
                "model": model.upper(), "format": fmt_label,
                "acc": mean_std(lst, "acc"), "prec": mean_std(lst, "prec"),
                "rec": mean_std(lst, "rec"), "f1": mean_std(lst, "f1"),
            }
            table_rows.append(row)
            print(f"  {model.upper():<7} {fmt_label:<8} "
                  f"{fmt_pair(lst,'acc'):<18} {fmt_pair(lst,'prec'):<18} "
                  f"{fmt_pair(lst,'rec'):<18} {fmt_pair(lst,'f1'):<18}")
        # acc drop f32->int8
        if "int8" in results[model]:
            drops = [results[model]["f32"][i]["acc"] - results[model]["int8"][i]["acc"]
                     for i in range(len(results[model]["f32"]))]
            dm, ds = float(np.mean(drops)), float(np.std(drops))
            sz = sizes.get(model, {})
            comp = (sz["f32"] / sz["int8"]) if sz.get("int8") else 0
            print(f"  {'':7} {'-> drop':<8} acc {dm*100:+.2f}% +/-{ds*100:.2f}%   "
                  f"size f32={sz.get('f32',0)/1024:.1f}KB int8={sz.get('int8',0)/1024:.1f}KB "
                  f"(x{comp:.1f} lebih kecil)")
        print()

    # ── tabel per-fold ───────────────────────────────────────────────────
    n_folds = len(fold_subjects)
    col_w = 12
    print(f"\n{SEP}")
    print("  AKURASI PER-FOLD")
    print(SEP)

    # header row — last 5 digits of subject ID (unique per fold; first digits
    # collide between some subjects). Consistent with thesis_all_metrics.py.
    fold_hdr = f"  {'Model':<7} {'Format':<8} " + " ".join(
        f"{'Fold'+str(i)+' ('+fold_subjects[i][-5:]+')':>{col_w}}" for i in range(n_folds)
    )
    print(fold_hdr)
    print("  " + "-" * (len(fold_hdr) - 2))

    for model, fmts in [("mlp", ["f32", "int8"]),
                        ("cnn1d", ["f32", "int8"]),
                        ("rf", ["f32"])]:
        for fmt in fmts:
            lst = results[model][fmt]
            fmt_label = "Float32" if fmt == "f32" else "INT8"
            fold_accs = " ".join(f"{lst[i]['acc']*100:>{col_w}.2f}%" for i in range(n_folds))
            print(f"  {model.upper():<7} {fmt_label:<8} {fold_accs}")
        print()

    # ── simpan json + markdown ───────────────────────────────────────────
    out_dir = "replay_all"
    os.makedirs(out_dir, exist_ok=True)
    tag = "augmented" if args.augmented else "real"

    payload = {
        "dataset": csv_path, "variant": tag,
        "n_folds": len(results["mlp"]["f32"]),
        "per_fold": results, "model_sizes_bytes": sizes,
    }
    json.dump(payload, open(os.path.join(out_dir, f"compare_{tag}.json"), "w"), indent=2)

    md = [f"# Perbandingan Model Float32 vs INT8 (LOSO — {tag})\n",
          f"Dataset: `{csv_path}` | {len(results['mlp']['f32'])}-fold LOSO\n",
          "| Model | Format | Accuracy | Precision | Recall | F1-score |",
          "|---|---|---|---|---|---|"]
    for r in table_rows:
        md.append(f"| {r['model']} | {r['format']} "
                  f"| {r['acc'][0]*100:.2f}% ± {r['acc'][1]*100:.2f}% "
                  f"| {r['prec'][0]*100:.2f}% ± {r['prec'][1]*100:.2f}% "
                  f"| {r['rec'][0]*100:.2f}% ± {r['rec'][1]*100:.2f}% "
                  f"| {r['f1'][0]:.4f} ± {r['f1'][1]:.4f} |")
    md_path = os.path.join(out_dir, f"compare_{tag}.md")
    open(md_path, "w", encoding="utf-8").write("\n".join(md))

    print(f"  Saved: {out_dir}/compare_{tag}.json  +  compare_{tag}.md\n")


if __name__ == "__main__":
    main()
