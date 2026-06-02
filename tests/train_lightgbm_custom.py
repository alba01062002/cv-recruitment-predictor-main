#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script específico para entrenar modelos LightGBM.
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

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("❌ ERROR: LightGBM no está instalado. Instálalo con 'pip install lightgbm'")

from src.train_model import train_model_all_editions

FEATURES_ROOT = os.path.join(PROJECT_ROOT, "output", "features")
BASE_MODELS_ROOT = os.path.join(PROJECT_ROOT, "output", "models", "custom_lightgbm")
BASE_LOGS_ROOT   = os.path.join(PROJECT_ROOT, "output", "logs", "custom_lightgbm")
MODES = ["combined", "separate"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

# =============================================================================
#  CONFIGURACIÓN 1: RECENT EDITIONS (MASI19-MASI24)
# =============================================================================
CONFIG_RECENT = {
    "NAME": "recent",
    
    # Ediciones
    "TRAIN_EDITIONS": ["MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"],
    "TEST_EDITIONS":  ["MASI25"],
    
    # Política de Evaluación
    "POLICY": {
        "thr_min": 0.20, "thr_max": 0.80, "thr_step": 0.01,
        "W_BA": 0.30, 
        "W_TPR": 0.60,
        "W_FPR": 0.10,
        "MAX_FPR": 0.60,
    },
    
    # Grid Search Params (Per Mode)
    "GRIDS": {
        "separate": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05],
            "num_leaves": [31],
            "min_child_samples": [20],
            "subsample": [0.8],
            "colsample_bytree": [0.7],
            "reg_lambda": [15.0], 
            "scale_pos_weight": [4.0, 8.0], 
            "oversampling": [True], 
            "svd": [0]
        },
        "combined": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05],
            "num_leaves": [31],
            "min_child_samples": [20],
            "subsample": [0.8],
            "colsample_bytree": [0.7],
            "reg_lambda": [10.0], 
            "scale_pos_weight": [4.0, 8.0], 
            "oversampling": [True], 
            "svd": [0]
        }
    }
}

# =============================================================================
#  CONFIGURACIÓN 2: ALL EDITIONS (MASI09-MASI24)
# =============================================================================
CONFIG_ALL = {
    "NAME": "all_editions",
    
    # Ediciones (Excluyendo MASI11 y MASI12 por anomalías conocidas si se desea)
    "TRAIN_EDITIONS": [
        "MASI09", "MASI10", "MASI13", "MASI14", "MASI15", "MASI16", 
        "MASI17", "MASI18", "MASI19", "MASI20", "MASI21", "MASI22", "MASI23", "MASI24"
    ],
    "TEST_EDITIONS": ["MASI25"],
    
    # Política de Evaluación (Puede ser diferente si se desea)
    "POLICY": {
        "thr_min": 0.20, "thr_max": 0.80, "thr_step": 0.01,
        "W_BA": 0.20, 
        "W_TPR": 0.80,
        "W_FPR": 0.10,
        "MAX_FPR": 0.60,
    },
    
    # Grid Search Params (Per Mode)
    "GRIDS": {
        "separate": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05],
            "num_leaves": [31],
            "min_child_samples": [20],
            "subsample": [0.8],
            "colsample_bytree": [0.7],
            "reg_lambda": [15.0], 
            "scale_pos_weight": [4.0, 8.0], 
            "oversampling": [True], 
            "svd": [256] #256 ÓPTIMO
        },
        "combined": {
            "n_estimators": [200, 300],
            "learning_rate": [0.05],
            "num_leaves": [31],
            "min_child_samples": [20],
            "subsample": [0.8],
            "colsample_bytree": [0.7],
            "reg_lambda": [10.0], 
            "scale_pos_weight": [4.0, 8.0], 
            "oversampling": [True], 
            "svd": [625] ######## ÓPTIMO, NO CAMBIAR 625
        }
    }
}

CONFIGS_TO_RUN = [CONFIG_RECENT, CONFIG_ALL]

def _has_required_artifacts(edition_dir: str, mode: str) -> bool:
    d = os.path.join(edition_dir, mode)
    return (os.path.isfile(os.path.join(d, "X.npz"))
            and os.path.isfile(os.path.join(d, "y.npy"))
            and os.path.isfile(os.path.join(d, "feature_names.txt")))

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

def build_model(params: Dict[str, Any]):
    if not HAS_LGBM:
        raise RuntimeError("LightGBM no disponible.")
    
    return LGBMClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        num_leaves=int(params.get("num_leaves", 31)),
        min_child_samples=int(params.get("min_child_samples", 20)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        reg_lambda=float(params.get("reg_lambda", 0.0)),
        scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

def run_training():
    if not HAS_LGBM:
        return

    logger.info("=== INICIANDO ENTRENAMIENTO CUSTOM LIGHTGBM (DUAL CONFIG) ===")

    for config in CONFIGS_TO_RUN:
        config_name = config["NAME"]
        train_editions = config["TRAIN_EDITIONS"]
        test_editions = config["TEST_EDITIONS"]
        policy = config["POLICY"]
        grids = config["GRIDS"] # Updated to read GRIDS
        
        # Define output folders for this specific config
        models_root = os.path.join(BASE_MODELS_ROOT, config_name)
        logs_root = os.path.join(BASE_LOGS_ROOT, config_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"▶️  EJECUTANDO CONFIGURACIÓN: {config_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"   Train Editions: {len(train_editions)} ({train_editions[0]}...{train_editions[-1]})")
        logger.info(f"   Test Editions:  {test_editions}")
        logger.info(f"   Output Models:  {models_root}")

        for mode in MODES:
            logger.info(f"\n   >>>> MODO: {mode} <<<<")
            
            # Select Grid for this mode
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
                    models_root=models_root, # Use contextual root
                    logs_root=logs_root,     # Use contextual root
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

    logger.info("=== FIN ENTRENAMIENTO LIGHTGBM ===")

if __name__ == "__main__":
    run_training()
