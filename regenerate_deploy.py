"""
regenerate_deploy.py
====================
Retrain & re-export the DEPLOYED models (MLP, CNN1D, RF) so the baked scaler /
TFLite model match the current training pipeline in models.py and the firmware.

WAJIB dijalankan setiap kali pipeline training/firmware berubah — kalau scaler
yang di-bake tidak sesuai dengan preprocessing firmware → mismatch → akurasi rusak.

Trains on ALL data (full_train) for the strongest deployed model, then copies
each header into its firmware directory.

Usage:
    python regenerate_deploy.py                 # default: augmented dataset
    python regenerate_deploy.py --real          # real subjects only
"""

import os, shutil, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from recorder import GestureRecorder
from models import train_model
from config import HEADERS_DIR, PROJECT_ROOT

REAL_CSV = "dataset/gesture_dataset.csv"
AUG_CSV  = "dataset/gesture_dataset_augmented.csv"

# header file name -> firmware dir it belongs to
FIRMWARE_DIRS = {
    "mlp":   os.path.join(PROJECT_ROOT, "firmware", "gesture_glove_mlp"),
    "cnn1d": os.path.join(PROJECT_ROOT, "firmware", "gesture_glove_cnn1d"),
    "rf":    os.path.join(PROJECT_ROOT, "firmware", "gesture_glove_rf"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="train on real subjects only (default: augmented)")
    args = ap.parse_args()
    csv_path = REAL_CSV if args.real else AUG_CSV

    X, y, subj = GestureRecorder.load_dataset(csv_path)
    print(f"  Dataset: {csv_path}  ({len(y)} windows)")
    print(f"  Pipeline: RAW -> features/scaler -> int8 (matches firmware)\n")

    for model_type in ("mlp", "cnn1d", "rf"):
        print(f"  [{model_type.upper()}] training (full_train) + export ...", end=" ", flush=True)
        result = train_model(
            model_type, X, y,
            quantize_int8=True, full_train=True, loso_mode=False,
        )
        acc = result.get("accuracy", 0)
        hdr = result.get("header_path", "")
        print(f"holdout acc={acc:.3f}")

        # Copy the generated header into the matching firmware dir.
        src = os.path.join(HEADERS_DIR, f"{model_type}_model_data.h")
        dst_dir = FIRMWARE_DIRS[model_type]
        if os.path.isfile(src):
            dst = os.path.join(dst_dir, f"{model_type}_model_data.h")
            shutil.copy(src, dst)
            print(f"       header -> {dst}")
        else:
            print(f"       !! header not found at {src} (export_error={result.get('header_error')})")

    print("\n  Done. Flash the 3 updated firmware sketches.")
    print("  NOTE: deployed models trained on", csv_path)


if __name__ == "__main__":
    main()
