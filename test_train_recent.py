#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid por modo ('combined'|'separate') con ALTO RECALL (FAST MODE) + SVD.
Filtro: MASI19+ (Standard Recent)
Objetivo: MAXIMIZAR Recall (~83%) manteniendo BA aceptable (~70%).
"""

import os, re, itertools, logging, warnings
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import DataConversionWarning

# Silenciar warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=DataConversionWarning)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from src.train_model import train_model_all_editions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

FEATURES_ROOT = os.path.join("output", "features")
MODELS_ROOT   = os.path.join("output", "models")
LOGS_ROOT     = os.path.join("output", "logs")

MODES = ["combined", "separate"] # Generar gráficos para ambos
EDITION_RE   = re.compile(r"^MASI(\d+)$", re.IGNORECASE)
TEST_HOLDOUT = {"MASI25"}

# --- POLÍTICA AGRESIVA DE HIGH RECALL (Restaurada) ---
MODEL_POLICY = {
    "logistic_regression": {
        "thr_min": 0.20, "thr_max": 0.60, "thr_step": 0.01,
        "W_BA": 0.30, 
        "W_TPR": 0.70, # Maximizamos Recall
        "W_FPR": 0.05,
        "MAX_FPR": 0.90, # Relaxed to ensure plot generation
    },
    "lightgbm": {
        "thr_min": 0.20, "thr_max": 0.55, "thr_step": 0.02,
        "W_BA": 0.30,  # Prioridad baja a balance puro
        "W_TPR": 0.70, # Prioridad ALTA a no perder candidatos (Recall)
        "W_FPR": 0.05,
        "MAX_FPR": 0.70,
    }
}

GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "penalty": ["l2", "l1"], 
        "C": [0.05, 0.1, 0.5, 1.0],
        "class_weight": [{0:1.0, 1:4.0}, {0:1.0, 1:5.0}], # Probamos peso 5.0 para forzar Recall
    }
}
if HAS_LGBM:
    GRIDS["lightgbm"] = {
        "n_estimators": [300],
        "learning_rate": [0.05],
        "num_leaves": [31],
        "min_child_samples": [20],
        "subsample": [0.8],
        "colsample_bytree": [0.7],
        "reg_lambda": [1.0, 5.0], # Bajamos un poco reg mínima para permitir más ajuste
        "scale_pos_weight": [4.0, 5.0], # Pesos altos
    }

def _has_required_artifacts(edition_dir: str, mode: str) -> bool:
    d = os.path.join(edition_dir, mode)
    return (os.path.isfile(os.path.join(d, "X.npz"))
            and os.path.isfile(os.path.join(d, "y.npy"))
            and os.path.isfile(os.path.join(d, "feature_names.txt")))

def _discover_editions(root: str, modes: List[str]) -> List[str]:
    if not os.path.isdir(root):
        logger.error(f"No existe {root}")
        return []
    found = []
    for name in sorted(os.listdir(root)):
        m = EDITION_RE.match(name)
        if not m:
            continue
        ed_dir = os.path.join(root, name)
        if not os.path.isdir(ed_dir):
            continue
        if any(_has_required_artifacts(ed_dir, md) for md in modes):
            found.append((name.upper(), int(m.group(1))))
    found.sort(key=lambda t: t[1])
    return [t[0] for t in found]

def product_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys()); vals = [grid[k] for k in keys]
    return [{k: v for k, v in zip(keys, comb)} for comb in itertools.product(*vals)]

def _tpr_fpr_from_cm(cm_list) -> Tuple[float, float]:
    tn, fp = cm_list[0]; fn, tp = cm_list[1]
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return float(tpr), float(fpr)

def composite_score(res: Dict[str, Any], policy: Dict[str, float]) -> Tuple[float, float, float, float]:
    W_BA, W_TPR, W_FPR = policy["W_BA"], policy["W_TPR"], policy["W_FPR"]
    ba = float(res.get("balanced_accuracy", 0.0))
    cm = res.get("agg_cm", [[0,0],[0,0]])
    tpr, fpr = _tpr_fpr_from_cm(cm)
    score = W_BA * ba + W_TPR * tpr - W_FPR * fpr
    return float(score), ba, tpr, fpr

def build_model(name: str, params: Dict[str, Any]):
    if name == "logistic_regression":
        return LogisticRegression(
            solver="liblinear",
            penalty=params.get("penalty", "l2"),
            C=float(params.get("C", 1.0)),
            class_weight=params.get("class_weight", None),
            max_iter=4000,
            random_state=42,
        )
    if name == "lightgbm":
        if not HAS_LGBM:
            raise RuntimeError("LightGBM no disponible.")
        return LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            num_leaves=int(params.get("num_leaves", 31)),
            max_depth=int(params.get("max_depth", -1)),
            min_child_samples=int(params.get("min_child_samples", 10)),
            subsample=float(params.get("subsample", 1.0)),
            subsample_freq=0,
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            reg_lambda=float(params.get("reg_lambda", 0.0)),
            scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
            random_state=42,
            n_jobs=-1,
            verbosity=-1, 
        )
    raise ValueError(f"Modelo no soportado: {name}")

def run_grid(editions: List[str]) -> None:
    ed_set    = set(editions)
    test_set  = sorted(list(ed_set & TEST_HOLDOUT))
    all_train = sorted(list(ed_set - TEST_HOLDOUT))
    
    # --- FILTRO MASI19+ ---
    train_set = [ed for ed in all_train if int(re.search(r"\d+", ed).group()) >= 19]

    if not train_set:
        return

    any_ok = False

    for mode in MODES:
        valid_train = [ed for ed in train_set if _has_required_artifacts(os.path.join(FEATURES_ROOT, ed), mode)]
        valid_test  = [ed for ed in test_set  if _has_required_artifacts(os.path.join(FEATURES_ROOT, ed), mode)]
        if not valid_train or not valid_test:
            continue

        logger.info(f"==== MODO {mode} (MASI19+) ====")
        
        # SVD Solo para Separate
        use_svd = 100 if mode == "separate" else 0

        for model_name, grid in GRIDS.items():
            policy = MODEL_POLICY[model_name]
            combos = product_grid(grid)
            logger.info(f"[GRID] {model_name} — {len(combos)} config")

            best = None

            for params in combos:
                clf = build_model(model_name, params)
                res = train_model_all_editions(
                    train_editions=valid_train,
                    test_editions=None,
                    vectorization_mode=mode,
                    model=clf,
                    features_root=FEATURES_ROOT,
                    models_root=MODELS_ROOT,
                    logs_root=LOGS_ROOT,
                    n_folds=5,
                    decision_threshold=None,
                    save_artifacts=False,
                    thr_min=policy["thr_min"],
                    thr_max=policy["thr_max"],
                    thr_step=policy["thr_step"],
                    svd_components=use_svd
                )
                if not res or res is True: continue

                score, ba, tpr, fpr = composite_score(res, policy)
                if fpr > policy["MAX_FPR"]: continue

                if (best is None) or (score > best[0]):
                    best = (score, params, res, ba, tpr, fpr)

            if best is None:
                continue

            best_params = dict(best[1])
            clf = build_model(model_name, best_params)
            
            # Save artifacts
            final = train_model_all_editions(
                train_editions=valid_train,
                test_editions=valid_test,
                vectorization_mode=mode,
                model=clf,
                features_root=FEATURES_ROOT,
                models_root=MODELS_ROOT,
                logs_root=LOGS_ROOT,
                n_folds=5,
                decision_threshold=None,
                save_artifacts=True,
                thr_min=policy["thr_min"],
                thr_max=policy["thr_max"],
                thr_step=policy["thr_step"],
                svd_components=use_svd
            )

            if final and final is not True:
                fscore, fba, ftpr, ffpr = composite_score(final, policy)
                logger.info(f"--> [FINAL {mode} {model_name}] TEST: BA={final.get('test_balanced_accuracy', 0):.3f} | REC={final.get('test_recall_pos', 0):.3f}")
                any_ok = True

    if any_ok:
        logger.info("✅ Grid completado.")

def main():
    editions = _discover_editions(FEATURES_ROOT, MODES)
    if not editions: return
    run_grid(editions)

if __name__ == "__main__":
    main()
