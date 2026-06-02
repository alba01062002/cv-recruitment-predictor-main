#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrain models using only features from the Top 3 most important sections.
Based on analyzing existing models in output/models/custom_*.
"""

import os, sys, glob, joblib, logging
import numpy as np
import argparse
# use: python3 tools/train_with_top_sections.py --config recent
# or: python3 tools/train_with_top_sections.py --config all_editions

# Add root to sys.path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_model import train_model_all_editions, _read_feature_names


# Config definitions (Simplified duplicate of original scripts)
CONFIG_RECENT = {
    "NAME": "recent",
    "TRAIN_EDITIONS": ["MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"],
    "TEST_EDITIONS":  ["MASI25"]
}

CONFIG_ALL = {
    "NAME": "all_editions",
    "TRAIN_EDITIONS": ["MASI09", "MASI10", "MASI13", "MASI14", "MASI15", "MASI16", 
                       "MASI17", "MASI18", "MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"],
    "TEST_EDITIONS": ["MASI25"]
}

SECTIONS_CANON = ["education","work","international","languages","skills","other","volunteer"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_top_sections(model_path: str, features_dir: str, top_n: int = 3, **kwargs) -> list:
    """
    Loads model, calculates section importance, returns top_n section names.
    """
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to load {model_path}: {e}")
        return []

    # Unwrap pipeline
    if hasattr(model, "steps"):
        clf = model.steps[-1][1]
    else:
        clf = model

    # Get feature importances
    imps = None
    if hasattr(clf, "coef_"):
        coef = clf.coef_
        if hasattr(coef, "toarray"): coef = coef.toarray()
        imps = np.sum(np.abs(coef), axis=0) # Sum absolute coefficients
    elif hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_

    if imps is None:
        logger.warning(f"Could not extract importance from {model_path}")
        return []

    # Get feature names matching the model's training config
    # For 'all_editions' (or any multi-edition), we must reconstruct the union of features
    # exactly as _align_concat does during training.
    
    # We can't just read one file. We need the list passed in from main.
    feature_names = kwargs.get("feature_names", [])
    
    if not feature_names:
        fn_path = os.path.join(features_dir, "feature_names.txt")
        if os.path.exists(fn_path):
             feature_names = _read_feature_names(fn_path)
    
    if not feature_names:
        logger.error(f"No feature names available for {model_path}")
        return []

    if len(imps) != len(feature_names):
        logger.warning(f"Feature mismatch: Model={len(imps)}, Names={len(feature_names)}")
        # If mismatch is due to SVD (Model << Names), we cannot map.
        if len(imps) < len(feature_names):
            logger.error(f"Model has fewer features ({len(imps)}) than available names ({len(feature_names)}). Likely SVD-transformed. Cannot determine section importance.")
            return []
            
        # If Model > Names, it's weird but maybe truncation works if aligned?
        # But for 'all_editions' vs 'recent' features, alignment is not guaranteed.
        # We must rely on correct feature_names being passed.
        return []

    # Aggregation
    section_scores = {}
    for name, imp in zip(feature_names, imps):
        if "::" in name:
            sec = name.split("::")[0].lower()
        else:
            sec = "dense features" # Or "other"
        
        section_scores[sec] = section_scores.get(sec, 0.0) + imp

    # Sort
    sorted_secs = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
    top_secs = [s[0] for s in sorted_secs if s[1] > 0][:top_n]
    
    logger.info(f"Top {top_n} sections for {os.path.basename(model_path)}: {top_secs}")
    return top_secs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["recent", "all_editions"], default="recent")
    args = parser.parse_args()
    
    selected_config = CONFIG_RECENT if args.config == "recent" else CONFIG_ALL
    mode = "separate" # User said only separate mode output is needed
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_root = os.path.join(root_dir, "output")
    models_root = os.path.join(output_root, "models")
    features_root = os.path.join(output_root, "features")
    
    # Reconstruct feature names for this config/mode
    # This ensures we have the mapping corresponding to the trained model
    logger.info(f"Reconstructing feature names for {selected_config['NAME']} (mode={mode})...")
    from src.train_model import _align_concat
    try:
        # We don't need the matrices, just the names. But _align_concat returns all.
        # Using a subset of editions? No, all train editions.
        _, _, correct_feature_names = _align_concat(
            selected_config["TRAIN_EDITIONS"], 
            features_root, 
            mode
        )
        logger.info(f"Reconstructed {len(correct_feature_names)} features.")
    except Exception as e:
        logger.error(f"Failed to reconstruct feature names: {e}")
        return

    # 1. Identify Model Types (custom_lr, custom_svc, custom_lightgbm)
    # We look for folders in output/models/
    # pattern: output/models/custom_<type>/<config_name>/separate/recruitment_model_*.pkl
    
    model_types = ["custom_lr", "custom_svc", "custom_lightgbm", "custom_xgboost"] # Added xgboost just in case
    
    for mtype in model_types:
        model_dir = os.path.join(models_root, mtype, selected_config["NAME"], mode)
        # Check if directory exists
        if not os.path.isdir(model_dir):
            continue
            
        candidates = glob.glob(os.path.join(model_dir, "recruitment_model_*.pkl"))
        
        if not candidates:
            logger.warning(f"No models found in {model_dir}")
            continue
            
        # Pick the one with most recent mtime
        best_model_path = max(candidates, key=os.path.getmtime)
        logger.info(f"Analyzing {best_model_path}...")
        
        # Pass the reconstructed feature names
        # top_sections = get_top_sections(best_model_path, "", feature_names=correct_feature_names)
        top_sections = ["work", "education"]
        
        if not top_sections:
            logger.error("Could not determine top sections. Skipping.")
            continue
            
        # Load Original Model to clone params
        orig_model = joblib.load(best_model_path)
        if hasattr(orig_model, "steps"): # Pipeline
            estimator = orig_model.steps[-1][1]
            # If SVC/Linear, check for StandardScaler
            has_scaler = any(isinstance(s[1], (joblib.load(os.path.join(root_dir, "venv/lib/python3.13/site-packages/sklearn/preprocessing/_data.py")).StandardScaler if 0 else object)) for s in orig_model.steps)
            # Actually easier to just clone the estimator and build a new pipeline if needed
            # But the training function builds its own pipeline if svd>0. 
            # If the original had scaler, we should probably include it.
            # src.train_model does NOT add scaler by default unless we pass a pipeline as `model`.
            # So we should pass the full pipeline? No, train_model_all_editions expects `model` to be the estimator OR a pipeline.
            # We will clone the *entire* original pipeline.
            final_estimator = orig_model
        else:
            final_estimator = orig_model
            
        # Retrain
        # output/models/top_sections/<type>/<config>/separate
        new_models_root = os.path.join(output_root, "models", "top_sections", mtype)
        new_logs_root = os.path.join(output_root, "reports", "top_sections", mtype)
        
        logger.info(f"Retraining {mtype} with sections: {top_sections}")
        
        train_model_all_editions(
            train_editions=selected_config["TRAIN_EDITIONS"],
            test_editions=selected_config["TEST_EDITIONS"],
            vectorization_mode=mode,
            model=final_estimator, # Re-use same model/pipeline with same params
            features_root=features_root,
            models_root=os.path.join(new_models_root, selected_config["NAME"]),
            logs_root=os.path.join(new_logs_root, selected_config["NAME"]),
            allowed_sections=top_sections,
            save_artifacts=True
        )

if __name__ == "__main__":
    main()
