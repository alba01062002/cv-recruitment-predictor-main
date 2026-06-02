#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Section-level importance analysis via Aggregated Feature Importance.

This script calculates the importance of each CV section by aggregating 
the feature importances (coefficients for linear models, gain/split for tree models) 
of all features belonging to that section.

Usage:
  python -m tools.analyze_section_importance --edition MASI25 --mode separate --config all_editions
"""

import os
import glob
import argparse
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

# Suppress sklearn warnings
import warnings
from sklearn.exceptions import InconsistentVersionWarning
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

def _load_section_dims(models_dir: str, mode: str):
    """
    Returns list of (section_name, dim_size) from vectorizer files.
    """
    dims = []
    if mode == "combined":
        path = os.path.join(models_dir, "tfidf_combined.joblib")
        if os.path.isfile(path):
            vec = joblib.load(path)
            if hasattr(vec, "vocabulary_"):
                dims.append(("Text (Combined)", len(vec.vocabulary_)))
            else:
                dims.append(("Text (Combined)", 0))
        else:
            dims.append(("Text (Combined)", 0))
    else:
        for sec in SECTIONS_CANON:
            # 1. Try new Unified Processor
            proc_path = os.path.join(models_dir, f"processor_{sec}.joblib")
            if os.path.isfile(proc_path):
                data = joblib.load(proc_path)
                # data = {'section':..., 'tfidf':..., 'scaler':...}
                
                d = 0
                # TF-IDF
                vec = data.get("tfidf")
                if vec and hasattr(vec, "vocabulary_"):
                    d += len(vec.vocabulary_)
                
                # Scaler
                scaler = data.get("scaler")
                if scaler and hasattr(scaler, "scale_"):
                    d += len(scaler.scale_)
                    
                dims.append((sec, d))
                continue

            # 2. Fallback to old split files
            path = os.path.join(models_dir, f"tfidf_{sec}.joblib")
            if os.path.isfile(path):
                vec = joblib.load(path)
                d = len(getattr(vec, "vocabulary_", {}) or {})
                dims.append((sec, d))
            else:
                dims.append((sec, 0))
    return dims

def _derive_section_dims_from_features(feature_names_path: str, mode: str):
    """
    Derive section boundaries by parsing feature_names.txt.
    Returns list of (section_name, start_idx, end_idx) sorted by first appearance.
    """
    if not os.path.isfile(feature_names_path):
        return []
    
    with open(feature_names_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    
    if mode == "combined":
        text_count = 0
        dense_start = -1
        for i, name in enumerate(names):
            if "::" not in name and not name.startswith("tfidf_"):
                if dense_start < 0:
                    dense_start = i
            else:
                text_count += 1
        
        result = []
        if text_count > 0:
            result.append(("Text (Combined)", 0, text_count))
        if dense_start >= 0:
            result.append(("Dense Features", dense_start, len(names)))
        return result
    
    section_ranges = {}
    dense_indices = []
    
    for i, name in enumerate(names):
        if "::" in name:
            sec = name.split("::")[0].lower()
            if sec not in section_ranges:
                section_ranges[sec] = [i, i]
            else:
                section_ranges[sec][1] = i
        else:
            dense_indices.append(i)
    
    result = []
    for sec in SECTIONS_CANON:
        if sec in section_ranges:
            start, end = section_ranges[sec]
            result.append((sec, start, end + 1))
    
    for sec, (start, end) in section_ranges.items():
        if sec not in SECTIONS_CANON:
            result.append((sec, start, end + 1))
    
    if dense_indices:
        result.append(("Dense Features", min(dense_indices), max(dense_indices) + 1))
    
    return result

def _get_feature_importances(model, n_features_expected: int) -> np.ndarray:
    """
    Extract global feature importance array.
    Returns array of positive float values.
    """
    clf = _unwrap_model(model)
    
    imps = None
    
    # 1. Linear Models (LogisticRegression, SVC, etc.)
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if issparse(coef):
            coef = coef.toarray()
        # Sum absolute coefficients across classes (usually axis 0)
        imps = np.sum(np.abs(coef), axis=0)
        
    # 2. Tree Models (LightGBM, XGBoost, etc.)
    elif hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
        
    if imps is None:
        print(f"Warning: Could not extract feature importance from {type(clf).__name__}")
        return np.zeros(n_features_expected)
        
    # Safety resizing if mismatch (though shouldn't happen if loaded correctly)
    if len(imps) != n_features_expected:
        print(f"Warning: Feature importance length ({len(imps)}) != expected features ({n_features_expected}). Truncating/Padding.")
        if len(imps) > n_features_expected:
            imps = imps[:n_features_expected]
        else:
            # pad
            imps = np.pad(imps, (0, n_features_expected - len(imps)))
            
    return np.abs(imps) # Iterate importance is always positive

def _find_all_models(models_root: str, edition: str, mode: str, config: str):
    """
    Finds all 'agile' models matching the config and mode.
    Returns list of (path, model_name).
    """
    found_models = []
    
    # Iterate over all custom_* directories
    # Pattern: models_root/custom_*/config/mode/recruitment_model_*.pkl
    search_pattern = os.path.join(models_root, "custom_*", config, mode, "recruitment_model_*.pkl")
    candidates = glob.glob(search_pattern)
    
    for path in candidates:
         # simple name extraction: recruitment_model_svc.pkl -> svc
         fname = os.path.basename(path)
         name = fname.replace("recruitment_model_", "").replace(".pkl", "").lower()
         found_models.append((path, name))
         
    # Sort by name for consistency
    found_models.sort(key=lambda x: x[1])
    return found_models

def process_model(model_path, model_name, features_dir, outdir, ed, mode, topn, vec_dir_arg=None):
    print(f"\n--- Analyzing Model: {model_name} ---")
    print(f"Loading model: {model_path}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return

    # We need to determine the total number of features the model expects
    clf = _unwrap_model(model)
    nfeat_model = getattr(clf, "n_features_in_", None)
    
    # If using feature_names.txt, we can check its length too
    feature_names_path = os.path.join(features_dir, "feature_names.txt")
    sections_offsets = _derive_section_dims_from_features(feature_names_path, mode)
    
    # Estimate total cols based on sections
    if sections_offsets:
        max_idx = max(end for _, _, end in sections_offsets)
    else:
        max_idx = 0
        
    print(f"Feature names derived max index: {max_idx}")
    
    if nfeat_model is None:
        nfeat_model = max_idx
        
    print(f"Analyzing importance for {nfeat_model} features...")
    
    feat_imps = _get_feature_importances(model, nfeat_model)
    print(f"Extracted importance array (sum={np.sum(feat_imps):.4f})")
    
    # If no section offsets from feature_names, try vectorizers
    models_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(model_path)))) # HACK-ish to get back to output/models
    
    if not sections_offsets:
        print("Warning: Could not derive sections from feature_names.txt. Trying vectorizers...")
        possible_vec_dirs = []
        if vec_dir_arg: possible_vec_dirs.append(vec_dir_arg)
        possible_vec_dirs.append(os.path.join(models_root, ed, mode)) # old fallback
        possible_vec_dirs.append(os.path.dirname(model_path))
        
        vec_dir = None
        for d in possible_vec_dirs:
            if glob.glob(os.path.join(d, "tfidf_*.joblib")):
                vec_dir = d
                break
                
        if vec_dir:
            dim_list = _load_section_dims(vec_dir, mode)
            sum_text = sum(d for _, d in dim_list)
            dense_dim = nfeat_model - sum_text
            if dense_dim > 0:
                dim_list.append(("Dense Features", dense_dim))
                
            curr = 0
            for name, size in dim_list:
                if size <= 0: continue
                sections_offsets.append((name, curr, curr+size))
                curr += size
        else:
            print("Error: Could not determine section boundaries.")
            return

    # Check for SVD/Mismatch
    # If sections imply we need e.g. 1170 features, but model only has 256, it's a mismatch.
    if sections_offsets:
        max_idx_needed = max(end for _, _, end in sections_offsets)
        if nfeat_model < max_idx_needed:
             print(f"Error: Model has {nfeat_model} features but sections require up to {max_idx_needed}.")
             print("       This suggests the model was trained with SVD or a different vocabulary.")
             print("       Section importance analysis requires the original feature space.")
             return

    # SVD check
    if nfeat_model and nfeat_model < 200 and "dense" not in mode:
        print(f"Warning: Model has very few features ({nfeat_model}). SVD might be active.")
        print("Section analysis requires original text features (no SVD). skipping...")
        return # Skip this model if it looks like SVD
        
    results = []
    total_agg = 0.0
    
    for sec, start, end in sections_offsets:
        if start >= len(feat_imps): 
            continue
            
        # Clip end if needed
        real_end = min(end, len(feat_imps))
        
        # If the range is invalid or empty
        if real_end <= start:
            continue
            
        imp_block = feat_imps[start:real_end]
        agg_val = np.sum(imp_block)
        
        results.append({
            "section": sec,
            "importance": agg_val,
            "n_features": real_end - start
        })
        total_agg += agg_val

    if not results:
        print("Error: No sections could be analyzed. Check feature alignment or SVD.")
        return
        
    # Add percentage
    for r in results:
        r["percentage"] = (r["importance"] / total_agg * 100) if total_agg > 0 else 0.0
        
    results.sort(key=lambda r: r["importance"], reverse=True)
    
    # Save Outputs
    # output/reports/section_importance/<config>/<mode>/
    os.makedirs(outdir, exist_ok=True)
    
    import csv
    csv_path = os.path.join(outdir, f"section_imp_{model_name}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "importance", "percentage", "n_features"])
        w.writeheader()
        for r in results:
            w.writerow(r)
            
    if not results:
        print("No results to plot.")
        return

    topn_val = min(topn, len(results))
    show_res = results[:topn_val]
    
    labels = [r["section"] for r in show_res]
    values = [r["percentage"] for r in show_res]
    
    # Invert for barh
    labels = labels[::-1]
    values = values[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Skyblue bars with steelblue edges
    bars = ax.barh(labels, values, color="skyblue", edgecolor="steelblue")
    
    ax.set_xlabel("Aggregated Feature Importance (%)")
    ax.set_title(f"Section Importance ({mode.upper()}) - {model_name}\n(Sum of Feature Importances)")
    
    # Add values to ends of bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', fontsize=9)
                
    # Add some padding to x-axis
    ax.set_xlim(0, max(values)*1.15)
    
    fig.tight_layout()
    png_path = os.path.join(outdir, f"section_imp_{model_name}.png")
    fig.savefig(png_path, dpi=300)
    plt.close()
    
    print(f"Results saved to {outdir}")
    print("Summary:")
    for r in results:
        print(f"  {r['section']:<20} | {r['importance']:>8.4f} | {r['percentage']:>5.1f}%")

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", required=True, help="MASIxx for fetching features")
    ap.add_argument("--mode", required=True, choices=["combined","separate"])
    ap.add_argument("--config", default="recent", choices=["recent", "all_editions"], help="Config to search (recent or all_editions)")
    ap.add_argument("--model", default=None, help="Specific model path (optional)")
    ap.add_argument("--vec-dir", default=None, help="Directory where tfidf_*.joblib used for X are stored")
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    ed = args.edition.strip().upper()
    mode = args.mode
    config = args.config
    out_root = args.output_root
    
    # Features always come from the edition
    features_dir = os.path.join(out_root, "features", ed, mode)
    if not os.path.isdir(features_dir):
        print(f"Warning: Features dir not found: {features_dir}")
        # Not raising error immediately, giving chance to run if vectorizers exist elsewhere

    models_root = os.path.join(out_root, "models")
    
    # If explicit model provided
    target_models = []
    if args.model and os.path.isfile(args.model):
        model_name = os.path.basename(args.model).replace("recruitment_model_", "").replace(".pkl", "").lower()
        target_models.append((args.model, model_name))
    else:
        # Find ALL models for this config
        target_models = _find_all_models(models_root, ed, mode, config)
        
    if not target_models:
        print(f"No models found for edition={ed}, mode={mode}, config={config} in {models_root}")
        return

    # Determine Output Directory
    # output/reports/section_importance/<config>/<mode>
    if args.outdir:
        final_outdir = args.outdir
    else:
        final_outdir = os.path.join(out_root, "reports", "section_importance", config, mode)
        
    print(f"Found {len(target_models)} models to analyze.")
    print(f"Output directory: {final_outdir}")
    
    for mpath, mname in target_models:
        process_model(mpath, mname, features_dir, final_outdir, ed, mode, args.topn, args.vec_dir)

if __name__ == "__main__":
    main()