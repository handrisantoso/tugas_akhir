"""
Test script to train and evaluate models on the ACTUAL gesture_dataset.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GESTURE_NAMES, DEFAULT_CSV, WINDOW_SIZE
from recorder import GestureRecorder
from models import train_model

def main():
    print("=" * 60)
    print("  GESTURE GLOVE — Actual Data Model Test")
    print("=" * 60)
    
    csv_path = DEFAULT_CSV
    if not os.path.exists(csv_path):
        # Check parent dir just in case
        csv_path = os.path.join("..", DEFAULT_CSV)
        if not os.path.exists(csv_path):
            print(f"[ERROR] Dataset not found at {DEFAULT_CSV}")
            return

    print(f"\n[1] Loading Dataset: {csv_path}")
    X, y = GestureRecorder.load_dataset(csv_path)
    
    if len(X) == 0:
        print("[ERROR] No valid samples found! (Check WINDOW_SIZE vs sample lengths)")
        return
        
    print(f"    Loaded {len(X)} samples with {X.shape[1]} features each.")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\n[2] Class Distribution:")
    for idx, count in zip(unique, counts):
        print(f"    {GESTURE_NAMES[idx]:<15}: {count} samples")

    # Train Random Forest
    print("\n[3] Training Random Forest...")
    rf_results = train_model("rf", X, y, n_estimators=100)
    print(f"    RF Accuracy: {rf_results['accuracy']:.2%}")
    # print(rf_results['report']) # It's a dict now

    # Train MLP
    print("\n[4] Training MLP (Deep Learning)...")
    mlp_results = train_model("mlp", X, y, epochs=50, batch_size=32)
    print(f"    MLP Accuracy: {mlp_results['accuracy']:.2%}")
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
