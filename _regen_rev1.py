"""Quick script to regenerate only rev_1_rf_feature_importance.png"""
import os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_CSV
from recorder import GestureRecorder
from journal_revision_analysis import revision_1_feature_importance

csv = DEFAULT_CSV
if not os.path.exists(csv):
    csv = os.path.join(os.path.dirname(__file__), "gesture_dataset.csv")

X, y, subj = GestureRecorder.load_dataset(csv)
print(f"Loaded {len(y)} samples")

r1 = revision_1_feature_importance(X, y, subj)
print("Done!")
