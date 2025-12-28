#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label script sencillo para CVs normalizados.

Hace exactamente esto:

- Lee todos los *_normalized.json de data/normalized/
- Normaliza el nombre de cada CV con normalize_name()
- Recorre data/selected/ y todas sus subcarpetas buscando .csv
- Para cada CSV, genera un fichero de etiquetas paralelo en data/labels/:

    data/selected/MASI09/MASI09_Becarios.csv
      -> data/labels/MASI09/MASI09_Becarios_labels.csv

- Cada fichero *_labels.csv tiene formato:
      name,label
      nombre normalizado,0|1

- Además genera un labels.csv global en data/labels/ combinando todo.
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple

import pandas as pd

from src.preprocess import normalize_name  # SOLO usamos esto

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- RUTAS ------------------
NORMALIZED_DIR = os.path.join("data", "normalized")
SELECTED_DIR   = os.path.join("data", "selected")
LABELS_DIR     = os.path.join("data", "labels")
GLOBAL_LABELS  = os.path.join(LABELS_DIR, "labels.csv")

os.makedirs(NORMALIZED_DIR, exist_ok=True)
os.makedirs(SELECTED_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

EDITION_RE = re.compile(r"(MASI\d{2})", re.IGNORECASE)


def _beautify_name(name: str) -> str:
    """
    Convierte 'Apellidos, Nombre' -> 'Nombre Apellidos' y colapsa espacios.
    """
    name = (name or "").strip().strip('"').strip("'")
    if not name:
        return ""
    if "," in name:
        left, right = [p.strip() for p in name.split(",", 1)]
        if left and right:
            name = f"{right} {left}"
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _infer_edition_from_cv_filename(base: str) -> str:
    """
    Intenta sacar MASIxx del nombre del fichero del CV.
    Si no lo encuentra, devuelve 'UNKNOWN'.
    """
    m = EDITION_RE.search(base)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def load_all_cvs() -> List[Dict]:
    """
    Carga TODOS los CVs normalizados de data/normalized y devuelve
    una lista de dicts con:
        - base_name        (sin _normalized.json)
        - display_name     (para escribir en labels)
        - norm_name        (clave para comparar)
        - edition          (MASIxx o UNKNOWN)
    """
    cvs: List[Dict] = []

    files = [f for f in os.listdir(NORMALIZED_DIR)
             if f.endswith("_normalized.json")]

    if not files:
        logger.warning("No se encontraron CVs en %s", NORMALIZED_DIR)
        return cvs

    logger.info("Encontrados %d CVs normalizados.", len(files))

    for fname in sorted(files):
        path = os.path.join(NORMALIZED_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                cv = json.load(f)
        except Exception as e:
            logger.error("Error leyendo %s: %s", fname, e)
            continue

        base = fname.replace("_normalized.json", "")

        full_name = (
            cv.get("personal_information", {}).get("full_name", "")
            or base
        )

        display_name = _beautify_name(full_name)
        norm_name = normalize_name(full_name)

        edition = _infer_edition_from_cv_filename(base)

        cvs.append({
            "base_name": base,
            "display_name": display_name,
            "norm_name": norm_name,
            "edition": edition,
        })

    logger.info("CVs cargados y normalizados por nombre.")
    return cvs


def build_selected_name_set(csv_path: str) -> Tuple[set, str]:
    """
    Carga un CSV de seleccionados y devuelve:

      - conjunto de nombres normalizados (set[str])
      - edición inferida desde el nombre del CSV (MASIxx o 'UNKNOWN')

    Asume columnas: 'First Name', 'Last Name'
    y opcionalmente 'Middle Name'.
    """
    fname = os.path.basename(csv_path)
    logger.info("Leyendo CSV de seleccionados: %s", csv_path)

    df = pd.read_csv(csv_path)

    if "First Name" not in df.columns or "Last Name" not in df.columns:
        logger.warning("CSV %s sin columnas 'First Name'/'Last Name'. Ignorado.", fname)
        return set(), "UNKNOWN"

    if "Middle Name" not in df.columns:
        df["Middle Name"] = ""

    full = (
        df["First Name"].fillna("").astype(str) + " " +
        df["Middle Name"].fillna("").astype(str) + " " +
        df["Last Name"].fillna("").astype(str)
    )

    norm = full.apply(normalize_name)
    name_set = set(norm.tolist())

    m = EDITION_RE.search(fname)
    edition = m.group(1).upper() if m else "UNKNOWN"

    logger.info("CSV %s: %d nombres normalizados, edición %s",
                fname, len(name_set), edition)

    return name_set, edition


def label_for_csv(all_cvs: List[Dict], csv_path: str) -> Tuple[List[Tuple[str, int]], str]:
    """
    Para un CSV de seleccionados (csv_path), genera la lista de
    (display_name, label) SOLO para los CVs cuya edición coincide.

    Si no se puede inferir edición del CSV -> se etiquetan todos los CVs.
    """
    selected_set, csv_edition = build_selected_name_set(csv_path)
    if not selected_set:
        return [], csv_edition

    rows: List[Tuple[str, int]] = []

    if csv_edition != "UNKNOWN":
        cvs_iter = [cv for cv in all_cvs if cv["edition"] == csv_edition]
    else:
        cvs_iter = all_cvs

    logger.info("Etiquetando %d CVs contra %s (edición %s)",
                len(cvs_iter), csv_path, csv_edition)

    for cv in cvs_iter:
        label = 1 if cv["norm_name"] in selected_set else 0
        rows.append((cv["display_name"], label))

    return rows, csv_edition


def main() -> None:
    # 1) Cargar todos los CVs una sola vez
    all_cvs = load_all_cvs()
    if not all_cvs:
        logger.error("No hay CVs normalizados. Nada que etiquetar.")
        return

    # Para el labels.csv global:
    global_labels: Dict[str, int] = {}    # clave: norm_name, valor: 0/1
    global_display: Dict[str, str] = {}   # norm_name -> display_name

    # 2) Recorrer data/selected/ y subcarpetas
    logger.info("Recorriendo %s para encontrar CSV de referencia...", SELECTED_DIR)
    for root, dirs, files in os.walk(SELECTED_DIR):
        rel_root = os.path.relpath(root, SELECTED_DIR)
        if rel_root == ".":
            rel_root = ""

        csv_files = [f for f in files if f.lower().endswith(".csv")]
        if not csv_files:
            continue

        logger.info("Directorio %s: CSV encontrados: %s",
                    root, ", ".join(csv_files))

        for f in sorted(csv_files):
            csv_path = os.path.join(root, f)

            # Etiquetas para ESTE CSV
            rows, csv_edition = label_for_csv(all_cvs, csv_path)
            if not rows:
                logger.warning("No se generaron etiquetas para %s", csv_path)
                continue

            # Ruta de salida paralela en data/labels/
            base_name = os.path.splitext(f)[0]      # MASI09_Becarios
            out_subdir = rel_root                   # ej: 'MASI09'
            out_dir = os.path.join(LABELS_DIR, out_subdir)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, base_name + "_labels.csv")

            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("name,label\n")
                for name, label in rows:
                    out_f.write(f"{name},{label}\n")

            logger.info("Escrito fichero de etiquetas: %s (%d filas)",
                        out_path, len(rows))

            # Actualizar labels globales (1 si aparece positivo en cualquier CSV)
            for cv, (name, label) in zip(
                [c for c in all_cvs if csv_edition == "UNKNOWN" or c["edition"] == csv_edition],
                rows
            ):
                norm = cv["norm_name"]
                if norm not in global_labels:
                    global_labels[norm] = label
                    global_display[norm] = name
                else:
                    # si ya tenía etiqueta, nos quedamos con el máximo (0->1)
                    if label > global_labels[norm]:
                        global_labels[norm] = label
                        global_display[norm] = name

    # 3) Escribir labels.csv global
    if global_labels:
        with open(GLOBAL_LABELS, "w", encoding="utf-8") as f:
            f.write("name,label\n")
            for norm, label in global_labels.items():
                name = global_display.get(norm, norm)
                f.write(f"{name},{label}\n")
        logger.info("Escrito labels global: %s (%d filas)",
                    GLOBAL_LABELS, len(global_labels))
    else:
        logger.warning("No se generó ninguna etiqueta global. ¿Hay CSVs en data/selected/?")

if __name__ == "__main__":
    logger.info("Running test_label_all_cvs.py ...")
    main()
    logger.info("Finished.")