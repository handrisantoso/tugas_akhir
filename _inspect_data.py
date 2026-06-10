"""Show exactly how 70/20/10 split distributes data across subjects."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recorder import GestureRecorder
from config import GESTURE_NAMES
from models import split_dataset

# Load dataset
X, y, subject_ids = GestureRecorder.load_dataset("dataset/gesture_dataset.csv")
print("=" * 65)
print("  HOW THE 70/20/10 SPLIT WORKS")
print("=" * 65)

print(f"\n[FULL DATASET] {len(X)} windows")
for s in np.unique(subject_ids):
    mask = subject_ids == s
    print(f"  Subject '{s}': {mask.sum()} windows")
    for lbl in range(5):
        cnt = ((y[mask]) == lbl).sum()
        print(f"    {GESTURE_NAMES[lbl]:15s}: {cnt}")

# --- Standard 70/20/10 split ---
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

# To track subjects, we need to also split subject_ids the same way
from sklearn.model_selection import train_test_split
# Reproduce exact same split
X_trval, X_test2, y_trval, y_test2, s_trval, s_test = train_test_split(
    X, y, subject_ids, test_size=0.1, random_state=42, stratify=y
)
X_train2, X_val2, y_train2, y_val2, s_train, s_val = train_test_split(
    X_trval, y_trval, s_trval, test_size=0.2/0.9, random_state=42, stratify=y_trval
)

print(f"\n[70/20/10 SPLIT] stratify=y (by LABEL, NOT by subject)")
print(f"  Stratification: preserves gesture class ratios")
print(f"  Subject awareness: NONE — subjects are mixed across all splits")

for name, sy, ss in [("TRAIN (70%)", y_train2, s_train),
                       ("VAL   (20%)", y_val2,   s_val),
                       ("TEST  (10%)", y_test2,  s_test)]:
    print(f"\n  [{name}] {len(sy)} windows")
    for s in np.unique(subject_ids):
        mask = ss == s
        cnt = mask.sum()
        pct = 100 * cnt / len(sy)
        print(f"    Subject '{s}': {cnt} windows ({pct:.0f}%)")

print("\n" + "=" * 65)
print("  CONCLUSION")
print("=" * 65)
print("""
  The 70/20/10 split is stratified by LABEL (gesture class), NOT
  by subject. Both subjects appear in train, val, AND test sets.

  This means:
    ✅ Class balance is preserved in each split
    ❌ Subject leakage: the model sees data from both subjects
       during training, so test accuracy is OPTIMISTIC

  For a thesis with multi-subject data, you should report BOTH:
    1. 70/20/10 (random stratified) — shows best-case accuracy
    2. LOSO cross-validation — shows cross-subject generalization

  LOSO holds out 1 entire subject for testing, trains on the other.
  This is the more honest metric for real-world deployment.
""")
