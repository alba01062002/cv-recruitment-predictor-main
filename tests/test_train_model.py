#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrena un único modelo GLOBAL por modo ('separate' y 'combined') usando TODAS
las ediciones MASI presentes en output/features/MASI** que tengan artefactos:

Se requiere por edición y modo:
  output/features/<EDICION>/<MODO>/X.npz
  output/features/<EDICION>/<MODO>/y.npy (o y.npz)
  output/features/<EDICION>/<MODO>/filenames.txt
"""

import os
import re
import logging
from typing import Dict, List

# Evita problemas de display en macOS / servidores sin GUI
import matplotlib
matplotlib.use("Agg")

# Modelos
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Entrenador global
from src.train_model import train_model_all_editions

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------- RUTAS ------------------
FEATURES_ROOT = os.path.join("output", "features")
MODELS_ROOT   = os.path.join("output", "models")
LOGS_ROOT     = os.path.join("output", "logs")

# Modos que quieres probar
MODES = ["separate", "combined"]

# Modelos a evaluar
MODELS: Dict[str, object] = {
    "random_forest":  RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "lightgbm":       lgb.LGBMClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    "xgboost":        XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
}

# === Configura aquí tus filtros ===
INCLUDE_EDITIONS: List[str] = []  # p.ej. ["MASI09","MASI10"]
EXCLUDE_EDITIONS: List[str] = []  # puedes poner varias: ["MASI12","MASI13","MASI14"]

# MASI09, MASI2018, MASI2024, etc. (respeta MAYÚSCULAS)
EDITION_RE = re.compile(r"^MASI(\d+)$")

def _has_required_artifacts(edition_dir: str, mode: str) -> bool:
    """Comprueba que existan X.npz + y.(npy|npz) + filenames.txt en <ed>/<mode>.
       Acepta 'X.npz' o 'x.npz' por compatibilidad.
    """
    d = os.path.join(edition_dir, mode)
    x_upper = os.path.isfile(os.path.join(d, "X.npz"))
    x_lower = os.path.isfile(os.path.join(d, "x.npz"))
    x_ok = x_upper or x_lower
    y_ok = os.path.isfile(os.path.join(d, "y.npy")) or os.path.isfile(os.path.join(d, "y.npz"))
    fn_ok = os.path.isfile(os.path.join(d, "filenames.txt"))
    if not (x_ok and y_ok and fn_ok):
        logger.warning(f"Artefactos incompletos en {d} "
                       f"(X.npz|x.npz={x_ok}, y.npy|y.npz={y_ok}, filenames.txt={fn_ok})")
    return x_ok and y_ok and fn_ok

def _discover_editions(root: str, modes: List[str]) -> List[str]:
    """
    Devuelve lista de ediciones MASI** que tengan artefactos válidos
    para al menos uno de los modos indicados.
    Ordena por el número de la edición (09, 10, 11, 2018, 2024, ...).
    """
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

        if any(_has_required_artifacts(ed_dir, mode) for mode in modes):
            found.append((name, int(m.group(1))))

    if not found:
        logger.error(f"No se encontraron ediciones con artefactos en {root}")
        return []

    found.sort(key=lambda t: t[1])
    editions = [t[0] for t in found]
    logger.info(f"Ediciones detectadas (con artefactos): {editions}")
    return editions

def _apply_filters(editions: List[str]) -> List[str]:
    eds = [e for e in editions]
    if INCLUDE_EDITIONS:
        inc = {e.strip().upper() for e in INCLUDE_EDITIONS if e.strip()}
        eds = [e for e in eds if e.upper() in inc]
    if EXCLUDE_EDITIONS:
        exc = {e.strip().upper() for e in EXCLUDE_EDITIONS if e.strip()}
        eds = [e for e in eds if e.upper() not in exc]
    return eds

def main() -> bool:
    editions = _discover_editions(FEATURES_ROOT, MODES)
    if not editions:
        return False

    # Aplica include/exclude globales
    filtered = _apply_filters(editions)
    logger.info(f"Ediciones tras filtros include/exclude: {filtered}")
    if not filtered:
        logger.error("No quedan ediciones tras aplicar filtros.")
        return False

    any_ok = False
    for mode in MODES:
        # Filtra solo las ediciones que tienen artefactos para este modo
        valid_for_mode = []
        for ed in filtered:
            if _has_required_artifacts(os.path.join(FEATURES_ROOT, ed), mode):
                valid_for_mode.append(ed)

        if not valid_for_mode:
            logger.error(f"No hay ediciones válidas para el modo '{mode}' (tras filtros).")
            continue

        logger.info(f"==== Entrenando GLOBAL (ALL ediciones) en modo: {mode} ====")
        for model_name, model in MODELS.items():
            logger.info(f"\n=== [ALL] Training {model_name} ({mode}) con ediciones: {valid_for_mode} ===")
            ok = train_model_all_editions(
                editions=valid_for_mode,
                vectorization_mode=mode,
                model=model,
                features_root=FEATURES_ROOT,
                models_root=MODELS_ROOT,
                logs_root=LOGS_ROOT,
                n_folds=5,
                include_editions=INCLUDE_EDITIONS or None,
                exclude_editions=EXCLUDE_EDITIONS or None
            )
            any_ok = any_ok or ok

    if not any_ok:
        logger.error("❌ Todos los entrenamientos globales fallaron.")
    else:
        logger.info("✅ Entrenamientos globales completados (al menos uno OK).")
    return any_ok

if __name__ == "__main__":
    main()