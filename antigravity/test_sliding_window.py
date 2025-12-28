#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de Ventana Deslizante (Sliding Window Validation).
Valida la robustez del modelo LightGBM (Combined, Best Config) en periodos anteriores.
"""

import os, re, logging, warnings
import pandas as pd
from typing import List, Dict, Any, Tuple

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from src.train_model import train_model_all_editions

# Configuración
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

FEATURES_ROOT = os.path.join("output", "features")
MODELS_ROOT   = os.path.join("output", "models")
LOGS_ROOT     = os.path.join("output", "logs")

# Escenarios de Validación (Train Window -> Test Year)
SCENARIOS = [
    {"train": [f"MASI{i}" for i in range(17, 23)], "test": ["MASI23"]}, # T:17-22 -> Test:23
    {"train": [f"MASI{i}" for i in range(18, 24)], "test": ["MASI24"]}, # T:18-23 -> Test:24
    {"train": [f"MASI{i}" for i in range(19, 25)], "test": ["MASI25"]}, # T:19-24 -> Test:25 (Baseline)
]

# Mejor Configuración Encontrada (LightGBM Combined)
BEST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20, # Regularización
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,       # Regularización
    "scale_pos_weight": 4.0, # High Recall Weight
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1 
}

# Política de Decisión (High Recall)
POLICY_THR_MIN = 0.20
POLICY_THR_MAX = 0.60
POLICY_THR_STEP = 0.01

def build_model():
    if not HAS_LGBM:
        raise RuntimeError("LightGBM required")
    return LGBMClassifier(**BEST_PARAMS)

def run_scenarios():
    results = []
    
    logger.info(f"Running {len(SCENARIOS)} Sliding Window Scenarios with Best LightGBM Config...")
    
    for sc in SCENARIOS:
        train_eds = sc["train"]
        test_eds  = sc["test"]
        name = f"Train:{train_eds[0]}-{train_eds[-1]} -> Test:{test_eds[0]}"
        
        logger.info(f"\n--- SCENARIO: {name} ---")
        
        clf = build_model()
        
        # Entrenar y Evaluar
        # Usamos Combined porque fue el ganador
        res = train_model_all_editions(
            train_editions=train_eds,
            test_editions=test_eds,
            vectorization_mode="combined",
            model=clf,
            features_root=FEATURES_ROOT,
            models_root=MODELS_ROOT,
            logs_root=LOGS_ROOT,
            n_folds=5,
            decision_threshold=None, # Auto-sweep en TRAIN
            save_artifacts=False, # No sobrescribir el modelo final oficial
            thr_min=POLICY_THR_MIN,
            thr_max=POLICY_THR_MAX,
            thr_step=POLICY_THR_STEP
        )
        
        if not res or res is True:
            logger.error(f"Failed to run scenario {name}")
            continue
            
        # Extraer métricas del TEST real
        row = {
            "Scenario": name,
            "Target_Year": test_eds[0],
            "CV_BA": res.get("balanced_accuracy", 0),
            "CV_Recall": res.get("recall_pos", 0),
            "Test_BA": res.get("test_balanced_accuracy", 0),
            "Test_Recall": res.get("test_recall_pos", 0),
            "Test_F1": res.get("test_f1_pos", 0),
            "Thr": res.get("test_threshold", 0)
        }
        results.append(row)
        logger.info(f"RESULT: Test_BA={row['Test_BA']:.3f} | Test_Rec={row['Test_Recall']:.3f}")

    # Reporte Final
    print("\n\n=== SLIDING WINDOW EXPERIMENT RESULTS ===")
    df = pd.DataFrame(results)
    print(df[["Scenario", "Test_BA", "Test_Recall", "Test_F1", "Thr"]].to_markdown(index=False, floatfmt=".3f"))
    
    # Análisis Simple
    avg_ba = df["Test_BA"].mean()
    avg_rec = df["Test_Recall"].mean()
    print(f"\nAverage Stability across {len(results)} years:")
    print(f"Mean Balanced Accuracy: {avg_ba:.3f}")
    print(f"Mean Recall (Class 1):  {avg_rec:.3f}")

if __name__ == "__main__":
    run_scenarios()
