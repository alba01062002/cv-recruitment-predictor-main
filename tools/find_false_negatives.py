#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to identify False Negatives in LightGBM All Editions Combined model.
Loads the model and the MASI25 test data, ALIGNS features to training set,
predicts probabilities, and lists the Admitted candidates (Class 1) 
with the lowest predicted probabilities.
"""

import os
import joblib
import numpy as np
import scipy.sparse as sp

# Paths
ROOT_DIR = "/Volumes/ALBA_RE/TFG/cv-recruitment-predictor-main"
MODEL_PATH = os.path.join(ROOT_DIR, "output/models/custom_lightgbm/all_editions/combined/recruitment_model_lgbmclassifier.pkl")
FEATURES_ROOT = os.path.join(ROOT_DIR, "output/features")
TEST_EDITION = "MASI25"
MODE = "combined"

# Editions used for training (Order is critical!)
TRAIN_EDITIONS = [
    "MASI09", "MASI10", "MASI13", "MASI14", "MASI15", "MASI16", 
    "MASI17", "MASI18", "MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"
]

def load_feature_names(edition, mode):
    path = os.path.join(FEATURES_ROOT, edition, mode, "feature_names.txt")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def get_feature_union(train_editions, mode):
    print("Building feature union from training editions...")
    union = []
    seen = set()
    for ed in train_editions:
        names = load_feature_names(ed, mode)
        for n in names:
            if n not in seen:
                seen.add(n)
                union.append(n)
    print(f"Total features in union: {len(union)}")
    return union

def load_test_data(edition, mode):
    print(f"Loading test data for {edition}...")
    ed_dir = os.path.join(FEATURES_ROOT, edition, mode)
    x_path = os.path.join(ed_dir, "X.npz")
    y_path = os.path.join(ed_dir, "y.npy")
    fn_path = os.path.join(ed_dir, "filenames.txt")
    feat_path = os.path.join(ed_dir, "feature_names.txt")
    
    if not (os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(fn_path)):
        # Try to find filenames in other places if missing? No, assume consistency.
        pass

    X = sp.load_npz(x_path).tocsr()
    y = np.load(y_path)
    
    with open(fn_path, "r", encoding="utf-8") as f:
        filenames = [ln.strip() for ln in f if ln.strip()]
    
    with open(feat_path, "r", encoding="utf-8") as f:
        feat_names = [ln.strip() for ln in f if ln.strip()]
        
    return X, y, filenames, feat_names

def align_features(X_orig, orig_names, union_names):
    print("Aligning test features to training union...")
    name_to_pos = {n: i for i, n in enumerate(union_names)}
    col_map = {i: name_to_pos[n] for i, n in enumerate(orig_names) if n in name_to_pos}
    
    rows, cols = X_orig.nonzero()
    new_cols = np.array([col_map.get(c, -1) for c in cols], dtype=int)
    
    mask = new_cols >= 0
    new_rows = rows[mask]
    new_cols = new_cols[mask]
    data = X_orig.data[mask]
    
    X_aligned = sp.csr_matrix((data, (new_rows, new_cols)), shape=(X_orig.shape[0], len(union_names)))
    return X_aligned

def main():
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)

    # 2. Build Feature Union
    union_features = get_feature_union(TRAIN_EDITIONS, MODE)
    
    # 3. Load Test Data
    X_test_raw, y_test, filenames, test_feat_names = load_test_data(TEST_EDITION, MODE)
    
    # 4. Align Test Data
    X_test = align_features(X_test_raw, test_feat_names, union_features)
    
    print(f"Aligned X_test shape: {X_test.shape}")

    # 5. Predict
    print("Predicting probabilities...")
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # 6. Identify False Negatives (Actual=1, Low Probability)
    print("\nAnalyzing Admitted Candidates (Class 1)...")
    
    admitted_indices = np.where(y_test == 1)[0]
    results = []
    
    for idx in admitted_indices:
        results.append({
            "index": idx,
            "filename": filenames[idx],
            "probability": y_prob[idx]
        })
        
    # Sort by probability (lowest first)
    results.sort(key=lambda x: x["probability"])
    
    print(f"\nTotal Admitted Candidates: {len(results)}")
    print("Candidates with lowest predicted probabilities (potential False Negatives):")
    print(f"{'Prob':<10} | {'Filename'} | {'Predicted Status (if thr=0.5)'}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        status = "Rejected" if res['probability'] < 0.5 else "Admitted"
        print(f"{res['probability']:.4f}     | {res['filename']:<30} | {status}")
        if i == 14: # Show top 15
            print("...")
            break

if __name__ == "__main__":
    main()
