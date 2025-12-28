#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Feature Importance and plot Top-N terms.

Improvements:
- Supports sklearn Pipelines (unwraps step).
- Differentiates Positive vs Negative coefficients (Red/Blue) for linear models.
- Highlights "Dense" (numeric) features vs "Text" features.
- Works for both Combined and Separate modes.
- Saves reports to `output/reports/feature_importance/<TAG>_<MODE>/` by default.

Usage examples:

# 1) Using a trained edition (recommended if you have the model .pkl)
python -m tools.analyze_feature_importance \
  --models-dir output/models/MASI24/combined \
  --mode combined \
  --topn 30

# 2) If you DON'T have the .pkl, but you do have features (X.npz,y.npy)
python -m tools.analyze_feature_importance \
  --features-dir output/features/MASI24/combined \
  --models-dir output/models/MASI24/combined \
  --mode combined \
  --topn 30
"""

import os
import re
import glob
import argparse
import logging
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import load_npz, csr_matrix

# ------------------ logging ------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
logger = logging.getLogger(__name__)

SECTIONS_ORDER = ["education","work","international","languages","skills","other","volunteer"]

# Dense features known names (fallback/highlighting)
DENSE_PREFIXES = ["dense_"]

# ------------------ utils --------------------
def _load_vectorizers(models_dir: str, mode: str):
    """
    Returns (feature_names_list, section_info)
    """
    if mode == "combined":
        path = os.path.join(models_dir, "tfidf_combined.joblib")
        # If not found, maybe we still have X.npz but no vectorizer? 
        # But we need names. Return placeholder if fails, but typically required.
        if not os.path.isfile(path):
            logger.warning(f"Missing tfidf_combined.joblib in {models_dir}. Names might be missing.")
            return [], []
            
        vec = joblib.load(path)
        if hasattr(vec, "get_feature_names_out"):
            names = vec.get_feature_names_out()
        else:
            names = vec.get_feature_names()
        return list(names), []

    # separate
    vec_paths = []
    # Try canonical order first
    for sec in SECTIONS_ORDER:
        p = os.path.join(models_dir, f"tfidf_{sec}.joblib")
        if os.path.isfile(p):
            vec_paths.append((sec, p))

    # Any extras
    for p in sorted(glob.glob(os.path.join(models_dir, "tfidf_*.joblib"))):
        sec = os.path.splitext(os.path.basename(p))[0].replace("tfidf_", "")
        if sec not in [s for s,_ in vec_paths] and sec != "combined":
            vec_paths.append((sec, p))

    if not vec_paths:
        logger.warning(f"No tfidf_*.joblib files found in {models_dir}.")
        return [], []

    all_names = []
    sections = []
    offset = 0
    for sec, p in vec_paths:
        vec = joblib.load(p)
        if hasattr(vec, "get_feature_names_out"):
            names = list(vec.get_feature_names_out())
        else:
            names = list(vec.get_feature_names())
        # prefix
        names = [f"{sec}::{t}" for t in names]
        all_names.extend(names)
        sections.append((sec, offset, offset + len(names)))
        offset += len(names)

    return all_names, sections

def _load_dense_names(models_dir: str):
    # Try to verify if dense scaler exists to confirm dense features presence
    p = os.path.join(models_dir, "dense_scaler.joblib")
    return bool(os.path.isfile(p))

def _find_model_pkl(models_dir: str):
    if not os.path.isdir(models_dir):
        return None
    # Prioritize 'svd' or standard
    cands = sorted(glob.glob(os.path.join(models_dir, "*.pkl")))
    # Pick the most relevant one? Usually just one.
    return cands[0] if cands else None

def _load_features(features_dir: str):
    X_path = os.path.join(features_dir, "X.npz")
    y_npy  = os.path.join(features_dir, "y.npy")
    fn_path = os.path.join(features_dir, "feature_names.txt")
    
    if not os.path.isfile(X_path) or not os.path.isfile(y_npy):
        raise FileNotFoundError(f"Missing X.npz or y.npy in {features_dir}")
        
    X = load_npz(X_path)
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    y = np.load(y_npy, allow_pickle=False)
    
    names_from_txt = []
    if os.path.isfile(fn_path):
        with open(fn_path, "r", encoding="utf-8") as f:
            names_from_txt = [line.strip() for line in f if line.strip()]
            
    return X, y, names_from_txt

def _unwrap_model(model):
    """Unwrap Pipeline to find the final estimator."""
    if hasattr(model, "steps"):
        # It's a pipeline
        return model.steps[-1][1]
    return model

def _compute_importance_from_model(model, feature_names):
    """
    Returns (importance_values, coefficient_values_or_None)
    - importance_values: array of floats (absolute importance usually)
    - coefficient_values_or_None: signed array if linear, else None
    """
    clf = _unwrap_model(model)
    
    # LightGBM / Tree
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
        if imp.ndim > 1: imp = imp.ravel()
        return imp, None # No sign for trees

    # Linear
    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        # Binary case: (1, n_features)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        # Multiclass: (n_classes, n_features) -> take mean(abs) for importance, but sign is ambiguous
        if coef.ndim == 2:
            # For importance ranking, use mean(abs)
            imp = np.mean(np.abs(coef), axis=0)
            # Sign is undefined for multiclass aggregate
            return imp, None 
        else:
            # Binary (1D array)
            return np.abs(coef), coef

    raise ValueError(f"Model {type(clf)} does not expose feature_importances_ or coef_.")

def _fit_surrogate(X, y, kind="logreg", class_weight="balanced"):
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            clf.fit(X, y)
            return clf
        except ImportError:
            pass # fallback to logreg

    from sklearn.linear_model import LogisticRegression
    cw = "balanced" if class_weight == "balanced" else None
    clf = LogisticRegression(solver="liblinear", C=1.0, class_weight=cw, max_iter=2000, random_state=42)
    clf.fit(X, y)
    return clf

def _plot_barh_png(out_png, feats, imps, signed_vals, title, xlabel="Importance"):
    """
    feats: top feature names
    imps: absolute importance (already sorted)
    signed_vals: if not None, contains the original signed coef for these feats (to determine color)
    """
    # Determine colors
    colors = "skyblue"
    has_sign = False
    
    if signed_vals is not None:
        colors = []
        for v in signed_vals:
            if v > 0: colors.append("forestgreen") # Positive class (Hired?)
            else: colors.append("indianred")       # Negative class
        has_sign = True
    
    plt.figure(figsize=(10, max(6, int(len(feats)*0.4))))
    y_pos = np.arange(len(feats))
    
    plt.barh(y_pos, imps, color=colors)
    plt.yticks(y_pos, feats)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    
    # Add legend if signed
    if has_sign:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='forestgreen', label='Positive (Hired)'),
            Patch(facecolor='indianred', label='Negative (Rejected)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ------------------ main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=str, default="", help="Directory with .joblib files (vectorizers)")
    ap.add_argument("--model-file", type=str, default="", help="Explicit path to model .pkl")
    ap.add_argument("--features-dir", type=str, default="", help="Directory with X.npz if no model")
    ap.add_argument("--mode", type=str, choices=["combined","separate"], required=True)
    ap.add_argument("--topn", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--surrogate", choices=["logreg","lgbm"], default="logreg")
    ap.add_argument("--class-weight", default="balanced")
    args = ap.parse_args()

    if not args.models_dir and not args.features_dir and not args.model_file:
        ap.error("Must provide --models-dir (for vectorizers) or --features-dir.")

    # Determine Base and Tag for Report
    # Logic: Try to find 'output' dir in path to anchor reports.
    # Also try to find edition tag (MASIxx).
    
    input_path = args.models_dir or args.features_dir or args.model_file
    input_abs = os.path.abspath(input_path)
    
    # Try to find 'output' in the path
    output_root = "output" # default fallback
    if "output" in input_abs.split(os.sep):
        # Assumption: structure is .../output/...
        # We want to save to .../output/reports/feature_importance
        parts = input_abs.split(os.sep)
        idx = parts.index("output")
        output_root = os.sep.join(parts[:idx+1])
    elif os.path.isdir(os.path.join(os.getcwd(), "output")):
         output_root = os.path.join(os.getcwd(), "output")

    # Try to detect Edition
    match = re.search(r"(MASI\d+)", input_path)
    tag = match.group(1) if match else "unknown"

    # Default Outdir
    if args.outdir:
        outdir = args.outdir
    else:
        # Structure: output/reports/feature_importance/MASI24_combined
        folder_name = f"{tag}_{args.mode}"
        outdir = os.path.join(output_root, "reports", "feature_importance", folder_name)
    
    os.makedirs(outdir, exist_ok=True)

    # 1. Load Feature Names
    # Try loading from vectorizers first
    models_dir = args.models_dir or (os.path.dirname(args.model_file) if args.model_file else "") or args.features_dir
    feature_names, _ = _load_vectorizers(models_dir, args.mode)
    
    # If using features dir, we can also check feature_names.txt as a fallback or source of truth
    features_dir = args.features_dir or args.models_dir
    X, y, txt_names = None, None, []
    
    if args.features_dir:
        X, y, txt_names = _load_features(args.features_dir)
        # If we didn't get names from joblibs, use txt_names
        if not feature_names and txt_names:
            feature_names = txt_names
            logger.info("Loaded feature names from feature_names.txt")
            
        # If we have both, usually text_names from numpy save is safer as it includes dense features
        if txt_names and len(txt_names) > len(feature_names):
            logger.info(f"feature_names.txt has {len(txt_names)} vs joblib {len(feature_names)}. Using txt.")
            feature_names = txt_names

    if not feature_names and X is not None:
        # Fallback
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]

    # 2. Load Model
    model = None
    if args.model_file and os.path.isfile(args.model_file):
        model_path = args.model_file
    else:
        model_path = _find_model_pkl(models_dir)
        # Fallback search
        if not model_path:
             alt = os.path.join("output", "models", args.mode)
             if os.path.isdir(alt):
                 model_path = _find_model_pkl(alt)

    imp_vals, signed_vals = None, None
    
    if model_path:
        logger.info(f"Loading model: {model_path}")
        try:
            model = joblib.load(model_path)
            imp_vals, signed_vals = _compute_importance_from_model(model, feature_names)
        except Exception as e:
            logger.warning(f"Could not extract importance from model: {e}")
            model = None 

    # 3. Surrogate if needed
    if model is None:
        if X is None:
            # We need X to fit surrogate
            if not args.features_dir:
                raise RuntimeError("No trained model found and no --features-dir provided for surrogate.")
            X, y, _ = _load_features(args.features_dir) # reload if needed
            
        logger.info("Fitting surrogate model...")
        # Trim X if mismatch
        if len(feature_names) != X.shape[1]:
            logger.warning(f"Dimension mismatch: X {X.shape[1]}, names {len(feature_names)}. Truncating/Adjusting.")
            n = min(len(feature_names), X.shape[1])
            X = X[:, :n]
            feature_names = feature_names[:n]
            
        model = _fit_surrogate(X, y, kind=args.surrogate, class_weight=args.class_weight)
        imp_vals, signed_vals = _compute_importance_from_model(model, feature_names)

    # 4. Process and Plot
    # Align lengths
    n_feat = min(len(feature_names), len(imp_vals))
    feature_names = feature_names[:n_feat]
    imp_vals = imp_vals[:n_feat]
    if signed_vals is not None:
        signed_vals = signed_vals[:n_feat]

    # Sort
    idx = np.argsort(-imp_vals)[:args.topn]
    
    top_feats = [feature_names[i] for i in idx]
    top_imps  = [imp_vals[i] for i in idx]
    top_signed = [signed_vals[i] for i in idx] if signed_vals is not None else None
    
    max_imp = max(top_imps) if top_imps else 1.0
    top_imps_norm = [v / max_imp for v in top_imps]

    # Save CSV
    import csv
    csv_path = os.path.join(outdir, f"feature_importance_top{len(top_feats)}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance_abs", "coef_signed"])
        for i in range(len(top_feats)):
            s_val = f"{top_signed[i]:.6f}" if top_signed is not None else "N/A"
            w.writerow([i+1, top_feats[i], f"{top_imps[i]:.6f}", s_val])
    
    # Save PNG
    clf = _unwrap_model(model)
    mtype = clf.__class__.__name__
    title = f"Top {len(top_feats)} Features - {mtype} ({args.mode})"
    
    png_path = os.path.join(outdir, f"feature_importance_top{len(top_feats)}.png")
    _plot_barh_png(png_path, top_feats, top_imps_norm, top_signed, title)
    
    logger.info(f"Done. Results saved to: {outdir}")

if __name__ == "__main__":
    main()