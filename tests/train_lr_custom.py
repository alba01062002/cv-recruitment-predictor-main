#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script específico para entrenar modelos Logistic Regression.
Soporta dos configuraciones independientes:
1. All Editions (todas las ediciones históricas)
2. Recent Editions (ediciones más recientes)

Cada configuración tiene sus propios parámetros, ediciones y carpetas de salida.
"""

import os
import sys

# Define Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to sys.path to allow 'from src.X import Y'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re, itertools, logging, warnings
from typing import Dict, List, Any, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ... (Previous imports)
from sklearn.exceptions import DataConversionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=DataConversionWarning)

from src.train_model import train_model_all_editions

FEATURES_ROOT = os.path.join(PROJECT_ROOT, "output", "features")
BASE_MODELS_ROOT = os.path.join(PROJECT_ROOT, "output", "models", "custom_lr")
BASE_LOGS_ROOT   = os.path.join(PROJECT_ROOT, "output", "logs", "custom_lr")
MODES = ["combined", "separate"] 

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURACIÓN 1: RECENT EDITIONS (MASI19-MASI24)
# =============================================================================
CONFIG_RECENT = {
    "NAME": "recent",
    "TRAIN_EDITIONS": ["MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"],
    "TEST_EDITIONS":  ["MASI25"],
    "POLICY": {
        "thr_min": 0.20, "thr_max": 0.80, "thr_step": 0.01,
        "W_BA": 0.50, 
        "W_TPR": 0.40,
        "W_FPR": 0.20,
        "MAX_FPR": 0.50,
    },
    "GRIDS": {
        "separate": [
            # Grid 1: Robust L2 (LibLinear) - Fast and reliable
            {
                "penalty": ["l2"], 
                "solver": ["liblinear"],
                "C": [0.01, 0.1, 1.0, 5.0, 10.0, 100.0], 
                "class_weight": ["balanced", None],
                "oversampling": [False], 
                "svd": [0]
            },
            # Grid 2: Feature Selection (L1/Lasso)
            {
                "penalty": ["l1"], 
                "solver": ["liblinear"],
                "C": [0.1, 1.0, 5.0, 10.0],
                "class_weight": ["balanced"],
                "oversampling": [False],
                "svd": [0]
            }
        ],
        "combined": [
            # Grid 1: Robust L2 (LibLinear)
            {
                "penalty": ["l2"], 
                "solver": ["liblinear"],
                "C": [0.01, 0.1, 1.0, 5.0, 10.0, 100.0], 
                "class_weight": ["balanced", None],
                "oversampling": [False], 
                "svd": [0]
            },
            # Grid 2: Feature Selection (L1/Lasso)
            {
                "penalty": ["l1"], 
                "solver": ["liblinear"],
                "C": [0.1, 1.0, 5.0, 10.0],
                "class_weight": ["balanced"],
                "oversampling": [False],
                "svd": [0]
            }
        ]
    }
}

# =============================================================================
#  CONFIGURACIÓN 2: ALL EDITIONS (MASI09-MASI24)
# =============================================================================
CONFIG_ALL = {
    "NAME": "all_editions",
    "TRAIN_EDITIONS": [
        "MASI09", "MASI10", "MASI13", "MASI14", "MASI15", "MASI16", 
        "MASI17", "MASI18", "MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"
    ],
    "TEST_EDITIONS": ["MASI25"],
    "POLICY": {
        "thr_min": 0.20, "thr_max": 0.80, "thr_step": 0.01,
        "W_BA": 1.0, 
        "W_TPR": 0.0,
        "W_FPR": 0.0,
        "MAX_FPR": 0.60,
    },
    "GRIDS": {
        "separate": [
            # Grid 1: L2 (LibLinear) - Wider search for balance
            {
                "penalty": ["l2"],
                "solver": ["liblinear"],
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 
                "class_weight": ["balanced", None, {0:1, 1:2}, {0:1, 1:1.5}],
                "oversampling": [False], 
                "svd": [0]
            },
            # Grid 2: L1 (Lasso) - Feature selection
            {
                "penalty": ["l1"],
                "solver": ["liblinear"],
                "C": [0.01, 0.05, 0.1, 0.5, 1.0],
                "class_weight": ["balanced", {0:1, 1:2}],
                "oversampling": [False],
                "svd": [0]
            }
        ],
        "combined": [
            # Grid 1: L2 (LibLinear) - Wider search for balance
            {
                "penalty": ["l2"],
                "solver": ["liblinear"],
                "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0], 
                "class_weight": ["balanced", None, {0:1, 1:2}, {0:1, 1:1.5}],
                "oversampling": [False], 
                "svd": [0]
            },
            # Grid 2: L1 (Lasso) - Feature selection
            {
                "penalty": ["l1"],
                "solver": ["liblinear"],
                "C": [0.01, 0.05, 0.1, 0.5, 1.0],
                "class_weight": ["balanced", {0:1, 1:2}],
                "oversampling": [False],
                "svd": [0]
            }
        ]
    }
}

CONFIGS_TO_RUN = [CONFIG_RECENT, CONFIG_ALL]

def _has_required_artifacts(edition_dir: str, mode: str) -> bool:
    d = os.path.join(edition_dir, mode)
    return (os.path.isfile(os.path.join(d, "X.npz"))
            and os.path.isfile(os.path.join(d, "y.npy"))
            and os.path.isfile(os.path.join(d, "feature_names.txt")))

def product_grid(grid: Any) -> List[Dict[str, Any]]:
    if isinstance(grid, list):
        return [c for g in grid for c in product_grid(g)]
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

def build_model(params: Dict[str, Any]):
    # Handle l1_ratio only if penalty is elasticnet
    penalty = params.get("penalty", "l2")
    l1_ratio = params.get("l1_ratio", None)
    if penalty != "elasticnet":
        l1_ratio = None
        
    lr = LogisticRegression(
        solver=params.get("solver", "saga"),
        penalty=penalty,
        C=float(params.get("C", 1.0)),
        l1_ratio=l1_ratio,
        class_weight=params.get("class_weight", None),
        max_iter=2000, 
        random_state=42,
    )
    
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", lr)
    ])

def run_training():
    logger.info("=== INICIANDO ENTRENAMIENTO CUSTOM LOGISTIC REGRESSION (DUAL CONFIG) ===")

    for config in CONFIGS_TO_RUN:
        config_name = config["NAME"]
        train_editions = config["TRAIN_EDITIONS"]
        test_editions = config["TEST_EDITIONS"]
        policy = config["POLICY"]
        grids = config["GRIDS"]
        
        models_root = os.path.join(BASE_MODELS_ROOT, config_name)
        logs_root = os.path.join(BASE_LOGS_ROOT, config_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"▶️  EJECUTANDO CONFIGURACIÓN: {config_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"   Train Editions: {len(train_editions)}")
        logger.info(f"   Output Models:  {models_root}")

        for mode in MODES:
            logger.info(f"\n   >>>> MODO: {mode} <<<<")
            
            # Select Grid
            grid = grids.get(mode)
            if not grid:
                logger.warning(f"   ⚠️ No hay grid definido para modo {mode}. Saltando...")
                continue

            valid_train = [ed for ed in train_editions if _has_required_artifacts(os.path.join(FEATURES_ROOT, ed), mode)]
            valid_test  = [ed for ed in test_editions  if _has_required_artifacts(os.path.join(FEATURES_ROOT, ed), mode)]
            
            if not valid_train:
                logger.warning(f"   ⚠️ No hay ediciones válidas para TRAIN en modo {mode}. Saltando...")
                continue
                
            combos = product_grid(grid)
            logger.info(f"   Grid Search con {len(combos)} combinaciones...")
            
            best = None

            for i, params in enumerate(combos):
                if i % 10 == 0:
                    print(f"      [Config {config_name}] Evaluando {i+1}/{len(combos)}...", end="\r", flush=True)
                    
                clf = build_model(params)
                
                res = train_model_all_editions(
                    train_editions=valid_train,
                    test_editions=None,
                    vectorization_mode=mode,
                    model=clf,
                    features_root=FEATURES_ROOT,
                    models_root=models_root,
                    logs_root=logs_root,
                    n_folds=5,
                    decision_threshold=None,
                    save_artifacts=False,
                    thr_min=policy["thr_min"],
                    thr_max=policy["thr_max"],
                    thr_step=policy["thr_step"],
                    svd_components=params.get("svd", 0),
                    oversampling=params.get("oversampling", False)
                )
                
                if not res or res is True: continue

                score, ba, tpr, fpr = composite_score(res, policy)
                
                if tpr < 0.10: continue # Relaxed constraint to ensure valid models found
                if fpr > policy["MAX_FPR"]: continue

                if (best is None) or (score > best[0]):
                    best = (score, params, res, ba, tpr, fpr)

            print(" " * 80, end="\r", flush=True) 

            if best is None:
                logger.error(f"   ❌ No se encontró configuración válida para {mode}.")
                continue

            best_score, best_params, _, best_ba, best_tpr, best_fpr = best
            logger.info(f"   ✅ MEJOR CONFIGURACIÓN ({mode}):")
            logger.info(f"      Score: {best_score:.4f} | BA: {best_ba:.3f} | Recall: {best_tpr:.3f} | FPR: {best_fpr:.3f}")
            logger.info(f"      Params: {best_params}")
            
            # Entrenamiento FINAL
            logger.info(f"   🔄 Entrenando modelo final y guardando en {models_root}...")
            final_clf = build_model(best_params)
            
            final_res = train_model_all_editions(
                train_editions=valid_train,
                test_editions=valid_test,
                vectorization_mode=mode,
                model=final_clf,
                features_root=FEATURES_ROOT,
                models_root=models_root,
                logs_root=logs_root,
                n_folds=5,
                decision_threshold=None,
                save_artifacts=True,
                thr_min=policy["thr_min"],
                thr_max=policy["thr_max"],
                thr_step=policy["thr_step"],
                svd_components=best_params.get("svd", 0),
                oversampling=best_params.get("oversampling", False)
            )
            
            if final_res and final_res is not True:
                 logger.info(f"      [RESULTADOS FINALES {mode}]")
                 logger.info(f"      CV BA: {final_res['balanced_accuracy']:.4f} | Rec: {final_res['recall_pos']:.4f}")
                 if "test_balanced_accuracy" in final_res:
                     logger.info(f"      TEST BA: {final_res['test_balanced_accuracy']:.4f} | Rec: {final_res['test_recall_pos']:.4f}")

    logger.info("=== FIN ENTRENAMIENTO LOGISTIC REGRESSION ===")

if __name__ == "__main__":
    run_training()
