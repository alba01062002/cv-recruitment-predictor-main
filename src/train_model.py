#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena y evalúa (CV + TEST) con selección de umbral por barrido.
Soporte para SVD (Reducción de dimensionalidad) incluido.
"""

from __future__ import annotations
import os, json, logging, joblib
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        force=True)

_DATASET_CACHE: Dict[Tuple[str, Tuple[str, ...], Tuple[str, ...]], Dict[str, Any]] = {}

def _read_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _load_edition(edition_dir: str):
    x_path = os.path.join(edition_dir, "X.npz")
    y_path = os.path.join(edition_dir, "y.npy")
    fn_path = os.path.join(edition_dir, "feature_names.txt")
    if not (os.path.isfile(x_path) and os.path.isfile(y_path) and os.path.isfile(fn_path)):
        raise FileNotFoundError(f"Faltan artefactos en {edition_dir}")
    X = sp.load_npz(x_path).tocsr()
    y = np.load(y_path)
    names = _read_feature_names(fn_path)
    return X, y, names

def _align_concat(editions: List[str], features_root: str, mode: str, base_union: Optional[List[str]] = None):
    mats, labels, name_lists = [], [], []
    for ed in editions:
        ed_dir = os.path.join(features_root, ed, mode)
        X, y, names = _load_edition(ed_dir)
        mats.append(X); labels.append(y); name_lists.append(names)

    if base_union is None:
        union, seen = [], set()
        for names in name_lists:
            for n in names:
                if n not in seen:
                    seen.add(n); union.append(n)
    else:
        union = list(base_union)

    name_to_pos = {n: i for i, n in enumerate(union)}
    X_blocks = []
    for X, names in zip(mats, name_lists):
        col_map = {j: name_to_pos[n] for j, n in enumerate(names) if n in name_to_pos}
        if not col_map:
            X_blocks.append(sp.csr_matrix((X.shape[0], len(union)), dtype=X.dtype))
            continue
        rows, cols = X.nonzero()
        mapped = np.array([col_map.get(c, -1) for c in cols], dtype=int)
        keep = mapped >= 0
        X_aligned = sp.csr_matrix((X.data[keep], (rows[keep], mapped[keep])),
                                  shape=(X.shape[0], len(union)), dtype=X.dtype)
        X_blocks.append(X_aligned)

    X_all = sp.vstack(X_blocks, format="csr")
    y_all = np.concatenate(labels)
    return X_all, y_all, union

def _prep_X(X):
    return X 

def _metrics_at_threshold(y_true, y_prob, thr: float):
    y_hat = (y_prob >= thr).astype(int)
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    ba  = balanced_accuracy_score(y_true, y_hat)
    f1p = f1_score(y_true, y_hat, pos_label=1, zero_division=0)
    return ba, tpr, fpr, f1p, cm.tolist()

def _sweep_best_threshold(y_true, y_prob, thr_min: float, thr_max: float, thr_step: float):
    thr_grid = np.arange(thr_min, thr_max + 1e-9, thr_step)
    best = None 
    for thr in thr_grid:
        ba, tpr, fpr, f1p, cm = _metrics_at_threshold(y_true, y_prob, thr)
        cand = (ba, tpr, -fpr, -thr, thr, f1p, cm)
        if (best is None) or (cand > best):
            best = cand
    ba, tpr, _nfpr, _nthr, thr_star, f1p, cm = best
    return float(thr_star), float(ba), float(tpr), float(f1p), cm

def _draw_cm(cm, title: str, hint_box: str, out_png: Optional[str]):
    tn, fp = cm[0]; fn, tp = cm[1]
    mat = np.array([[tn, fp],[fn, tp]])
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)
    ax.imshow(mat, cmap="viridis")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i,j]}", ha="center", va="center", color="w", fontsize=12, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Not Hired","Hired"])
    ax.set_yticklabels(["Not Hired","Hired"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=14, fontweight="bold") # Increased font size
    ax.text(1.02, 0.10, hint_box, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", fc="#ecf0f1", ec="#bdc3c7"), fontsize=10)
    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

def train_model_all_editions(
    train_editions: List[str],
    test_editions: Optional[List[str]],
    vectorization_mode: str,
    model,
    features_root: str,
    models_root: str,
    logs_root: str,
    n_folds: int = 5,
    decision_threshold: Optional[float] = None,
    use_oversampling: bool = False,
    save_artifacts: bool = True,
    thr_min: float = 0.33,
    thr_max: float = 0.50,
    thr_step: float = 0.01,
    svd_components: int = 0
) -> Dict[str, Any]:

    key = (vectorization_mode, tuple(sorted(train_editions or [])), tuple(sorted(test_editions or [])))
    if key in _DATASET_CACHE:
        cache = _DATASET_CACHE[key]
        X_train = cache["X_train"]; y_train = cache["y_train"]
        X_test  = cache.get("X_test");  y_test = cache.get("y_test")
        logger.info(f"[{vectorization_mode}] Cached dataset ({len(train_editions)} train, {len(test_editions or [])} test)")
    else:
        X_train, y_train, names_union = _align_concat(train_editions, features_root, vectorization_mode)
        if test_editions:
            X_test, y_test, _ = _align_concat(test_editions, features_root, vectorization_mode, base_union=names_union)
        else:
            X_test, y_test = None, None
        _DATASET_CACHE[key] = dict(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, names_union=names_union)

    if svd_components > 0 and X_train.shape[1] > svd_components:
        pipeline = Pipeline([
            ("svd", TruncatedSVD(n_components=svd_components, random_state=42)),
            ("clf", model)
        ])
        final_estimator = pipeline
        logger.info(f"[{vectorization_mode}] SVD activado: {X_train.shape[1]} -> {svd_components} features.")
    else:
        final_estimator = model

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_prob = np.zeros(y_train.shape[0], dtype=float)
    covered = np.zeros(y_train.shape[0], dtype=bool)

    for tr_idx, va_idx in skf.split(X_train, y_train):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        
        m_fold = clone(final_estimator)
        m_fold.fit(X_tr, y_tr)
        
        if hasattr(m_fold, "predict_proba"):
            oof_prob[va_idx] = m_fold.predict_proba(X_va)[:, 1]
        elif hasattr(m_fold, "destination") and hasattr(m_fold.steps[-1][1], "predict_proba"): 
             oof_prob[va_idx] = m_fold.predict_proba(X_va)[:, 1]
        else:
             oof_prob[va_idx] = m_fold.predict_proba(X_va)[:, 1]
        
        covered[va_idx] = True

    if (decision_threshold is None) or (isinstance(decision_threshold, str) and decision_threshold.lower() == "auto"):
        thr_star, ba_cv, tpr_cv, f1_cv, cm_cv = _sweep_best_threshold(y_train, oof_prob, thr_min, thr_max, thr_step)
    else:
        thr_star = float(decision_threshold)
        ba_cv, tpr_cv, _fpr_cv, f1_cv, cm_cv = _metrics_at_threshold(y_train, oof_prob, thr_star)

    final_model = clone(final_estimator)
    final_model.fit(X_train, y_train)

    result: Dict[str, Any] = {
        "balanced_accuracy": float(ba_cv),
        "recall_pos": float(tpr_cv),
        "f1_pos": float(f1_cv),
        "agg_cm": cm_cv,
        "opt_threshold": float(thr_star),
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
    }

    if test_editions and X_test is not None:
        prob_test = final_model.predict_proba(X_test)[:, 1]
        ba_t, tpr_t, fpr_t, f1_t, cm_t = _metrics_at_threshold(y_test, prob_test, thr_star)
        result.update({
            "test_balanced_accuracy": float(ba_t),
            "test_recall_pos": float(tpr_t),
            "test_fpr": float(fpr_t),
            "test_f1_pos": float(f1_t),
            "test_cm": cm_t,
            "test_threshold": float(thr_star),
        })

    if save_artifacts:
        model_dir = os.path.join(models_root, f"{vectorization_mode}")
        os.makedirs(model_dir, exist_ok=True)
        
        NAME_MAP = {
            "lgbmclassifier": "LightGBM",
            "logisticregression": "Logistic Regression",
            "randomforestclassifier": "Random Forest", 
            "svc": "SVM"
        }

        if isinstance(final_model, Pipeline):
            raw_name = final_model.steps[-1][1].__class__.__name__.lower()
            joblib.dump(final_model, os.path.join(model_dir, f"recruitment_model_{raw_name}_svd.pkl"))
        else:
            raw_name = final_model.__class__.__name__.lower()
            joblib.dump(final_model, os.path.join(model_dir, f"recruitment_model_{raw_name}.pkl"))

        clf_title = NAME_MAP.get(raw_name, raw_name)
        clf_file  = raw_name

        figs_dir = os.path.join(logs_root, f"{vectorization_mode}")
        os.makedirs(figs_dir, exist_ok=True)
        ba, tpr, f1p = result["balanced_accuracy"]*100, result["recall_pos"]*100, result["f1_pos"]*100
        hint = (f"CV BA: {ba:.1f}%\nREC (1): {tpr:.1f}%\nF1 (1): {f1p:.1f}%\nthr*: {result['opt_threshold']:.3f}")
        
        # TITLE WITH CLF_TITLE
        _draw_cm(result["agg_cm"], f"Confusion Matrix (CV, {vectorization_mode}, {clf_title})", hint, os.path.join(figs_dir, f"cm_cv_{clf_file}.png"))

        if "test_cm" in result:
             ba, tpr, f1p = result["test_balanced_accuracy"]*100, result["test_recall_pos"]*100, result["test_f1_pos"]*100
             hint = (f"TEST BA: {ba:.1f}%\nREC (1): {tpr:.1f}%\nF1 (1): {f1p:.1f}%\nthr*: {result['test_threshold']:.3f}")
             _draw_cm(result["test_cm"], f"Confusion Matrix (TEST, {vectorization_mode}, {clf_title})", hint, os.path.join(figs_dir, f"cm_test_{clf_file}.png"))

    return result