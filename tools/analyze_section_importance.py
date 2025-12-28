#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section-level importance analysis.

NOTE: This script is primarily designed for 'SEPARATE' mode, where features are split 
by section (Education, Work, etc.).
In 'COMBINED' mode, all text is merged, so we can only compare "All Text" vs "Dense Features".

Usage:
  python -m tools.analyze_section_importance --edition MASI25 --mode separate
"""

import os
import glob
import argparse
import joblib
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import load_npz, csr_matrix, hstack, diags
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings about feature names when passing numpy/matrix to model trained with DF
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

SECTIONS_CANON = ["education","work","international","languages","skills","other","volunteer"]

# -------------------- helpers --------------------
def _is_lgbm(model) -> bool:
    name = model.__class__.__name__.lower()
    return "lgbm" in name or "lightgbm" in name

def _unwrap_model(model):
    if hasattr(model, "steps"): 
        return model.steps[-1][1]
    return model

def _prep_X_for_model(model, X: csr_matrix, ncols: int):
    # Pad if needed
    if X.shape[1] < ncols:
        zeros = csr_matrix((X.shape[0], ncols - X.shape[1]), dtype=X.dtype)
        X = hstack([X, zeros], format="csr")
    
    clf = _unwrap_model(model)
    if _is_lgbm(clf) and X.dtype != np.float32:
        X = X.astype(np.float32)
    return X

def _load_xy(features_dir: str):
    X = load_npz(os.path.join(features_dir, "X.npz"))
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    y_path_npy = os.path.join(features_dir, "y.npy")
    if os.path.isfile(y_path_npy):
        y = np.load(y_path_npy, allow_pickle=False)
    else:
        # fallback npz
        tmp = np.load(os.path.join(features_dir, "y.npz"))
        y = tmp["arr_0"]
    return X, y

def _load_section_dims(models_dir: str, mode: str):
    """
    Returns list of (section_name, dim_size)
    """
    dims = []
    
    if mode == "combined":
        # In combined, we only have one text block "combined"
        path = os.path.join(models_dir, "tfidf_combined.joblib")
        if os.path.isfile(path):
            vec = joblib.load(path)
            # sklearn 1.0+
            if hasattr(vec, "vocabulary_"):
                dims.append(("Text (Combined)", len(vec.vocabulary_)))
            else:
                dims.append(("Text (Combined)", 0))
        else:
            dims.append(("Text (Combined)", 0))
            
    else:
        # Separate
        for sec in SECTIONS_CANON:
            path = os.path.join(models_dir, f"tfidf_{sec}.joblib")
            if os.path.isfile(path):
                vec = joblib.load(path)
                d = len(getattr(vec, "vocabulary_", {}) or {})
                dims.append((sec, d))
            else:
                dims.append((sec, 0))
                
    return dims

def _mask_columns(X: csr_matrix, start: int, end: int):
    """Zero out [start:end] columns."""
    n, d = X.shape
    if start >= end or start >= d:
        return X
    
    # Mask vector
    mask_diag = np.ones(d, dtype=np.float32)
    limit_end = min(end, d)
    if start < limit_end:
        mask_diag[start:limit_end] = 0.0
        
    D = diags(mask_diag, offsets=0, shape=(d, d), dtype=np.float32)
    return X @ D

def _predict_labels(model, X, thr: float):
    # Handle Pipeline
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return (s >= thr).astype(int), s
    else:
        return model.predict(X), None
        
    return (p >= thr).astype(int), p

def compute_metrics(y_true, y_pred):
    ba = balanced_accuracy_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rc = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return ba, pr, rc, f1

def _auto_find_train_tag(models_root: str, edition_test: str):
    patt = os.path.join(models_root, f"TRAIN_*__TEST_{edition_test}")
    cands = glob.glob(patt)
    if not cands: return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.basename(cands[0])

def _pick_model_path(tag_dir: str, mode: str, model_hint: str | None):
    # Search logic: 1. tag_dir/mode, 2. tag_dir
    search_dirs = [os.path.join(tag_dir, mode), tag_dir]
    files = []
    
    for d in search_dirs:
        if os.path.isdir(d):
            files.extend(glob.glob(os.path.join(d, "recruitment_model_*.pkl")))
            
    if not files:
        files = glob.glob(os.path.join(tag_dir, "**", "recruitment_model_*.pkl"), recursive=True)
    
    if not files:
        if mode == "combined":
             generic = os.path.join(os.path.dirname(tag_dir), "combined") 
             if os.path.isdir(generic):
                 files.extend(glob.glob(os.path.join(generic, "recruitment_model_*.pkl")))

    if not files:
        raise FileNotFoundError(f"No models found in {tag_dir}")

    if model_hint:
        filtered = [f for f in files if model_hint.lower() in os.path.basename(f).lower()]
        if filtered:
            files = filtered
            
    files = sorted(list(set(files)))
    f0 = files[0]
    name = os.path.basename(f0).replace("recruitment_model_", "").replace(".pkl","").lower()
    return f0, name

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True, help="MASIxx")
    ap.add_argument("--mode", required=True, choices=["combined","separate"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--train-tag", default=None)
    ap.add_argument("--vec-dir", default=None, help="Directory where tfidf_*.joblib used for X are stored")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # Alert user if they use combined mode for section importance
    if args.mode == "combined":
        print("NOTE: In 'combined' mode, all text is merged. "
              "Section Importance will only distinguish 'Text (All)' vs 'Dense Features'.")

    ed = args.edition.strip().upper()
    mode = args.mode
    
    out_root = args.output_root
    features_dir = os.path.join(out_root, "features", ed, mode)
    models_root = os.path.join(out_root, "models")
    
    if not os.path.isdir(features_dir):
        raise FileNotFoundError(f"Features dir not found: {features_dir}")

    # Find model
    train_tag = args.train_tag or _auto_find_train_tag(models_root, ed)
    if not train_tag:
        tag_dir = os.path.join(models_root)
    else:
        tag_dir = os.path.join(models_root, train_tag)

    model_path, model_name = _pick_model_path(tag_dir, mode, args.model)
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    
    print(f"Loading features from: {features_dir}")
    X, y = _load_xy(features_dir)
    
    clf = _unwrap_model(model)
    nfeat_model = getattr(clf, "n_features_in_", None)
    if nfeat_model is None: nfeat_model = X.shape[1]
    
    X = _prep_X_for_model(model, X, nfeat_model)
    
    thr = 0.5
    art_files = glob.glob(os.path.join(os.path.dirname(model_path), f"artifacts_*.pkl"))
    if not art_files and train_tag:
        art_files = glob.glob(os.path.join(models_root, train_tag, mode, f"artifacts_*.pkl"))
    
    if art_files:
        try:
            meta = joblib.load(art_files[0])
            thr = float(meta.get("decision_threshold", 0.5))
            print(f"Using loaded threshold: {thr:.3f}")
        except:
            print("Using default threshold 0.5")

    yhat_base, _ = _predict_labels(model, X, thr)
    ba0, pr0, rc0, f10 = compute_metrics(y, yhat_base)
    print(f"Baseline BA: {ba0:.4f} (Recall: {rc0:.4f})")
    
    possible_vec_dirs = []
    if args.vec_dir: possible_vec_dirs.append(args.vec_dir)
    possible_vec_dirs.append(os.path.join(models_root, ed, mode))
    possible_vec_dirs.append(os.path.dirname(model_path))
    
    vec_dir = None
    for d in possible_vec_dirs:
        if glob.glob(os.path.join(d, "tfidf_*.joblib")):
            vec_dir = d
            break
            
    if not vec_dir:
        print(f"Warning: No vectorizers found in {possible_vec_dirs}. Section boundaries impossible to determine.")
        dim_list = []
    else:
        print(f"Using vectorizers from: {vec_dir}")
        dim_list = _load_section_dims(vec_dir, mode)
    
    sum_text = sum(d for _, d in dim_list)
    total_cols = X.shape[1]
    dense_dim = total_cols - sum_text
    
    if dense_dim > 0:
        dim_list.append(("Dense Features", dense_dim))
    elif dense_dim < 0:
        print(f"Warning: Text dims ({sum_text}) > X columns ({total_cols}).")
        
    sections_offsets = []
    curr = 0
    for name, size in dim_list:
        if size <= 0: continue
        sections_offsets.append((name, curr, curr+size))
        curr += size
        
    results = []
    for sec, start, end in sections_offsets:
        print(f"Analyzing {sec} (cols {start}-{end})...")
        X_masked = _mask_columns(X, start, end)
        yhat_m, _ = _predict_labels(model, X_masked, thr)
        ba, pr, rc, f1 = compute_metrics(y, yhat_m)
        
        drop = ba0 - ba
        results.append({
            "section": sec,
            "BA_drop": drop,
            "BA_masked": ba,
            "Recall_drop": rc0 - rc,
            "F1_drop": f10 - f1
        })
        
    results.sort(key=lambda r: r["BA_drop"], reverse=True)
    
    outdir = args.outdir or os.path.join(out_root, "reports", "section_importance", f"{ed}_{mode}")
    os.makedirs(outdir, exist_ok=True)
    
    import csv
    csv_path = os.path.join(outdir, f"section_imp_{model_name}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "BA_drop", "BA_masked", "Recall_drop", "F1_drop"])
        w.writeheader()
        for r in results:
            w.writerow(r)
            
    topn = max(1, min(args.topn, len(results)))
    show = results[:topn]
    labels = [r["section"] for r in show]
    values = [r["BA_drop"] for r in show]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], values[::-1], color="salmon")
    ax.set_xlabel("Drop in Balanced Accuracy")
    ax.set_title(f"Section Importance ({mode.upper()}) - {model_name}")
    fig.tight_layout()
    png_path = os.path.join(outdir, f"section_imp_{model_name}.png")
    fig.savefig(png_path, dpi=300)
    plt.close()
    
    print(f"Results saved to {outdir}")

if __name__ == "__main__":
    main()