#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label script sencillo para CVs normalizados.

- Lee todos los *_normalized.json de data/normalized/** (todas las subcarpetas MASIxx)
- Normaliza el nombre de cada CV con normalize_name()
- Recorre data/selected/ y todas sus subcarpetas buscando .csv
- Para cada CSV, genera un fichero de etiquetas paralelo en data/labels/:

    data/selected/MASI09/MASI09_Becarios.csv
      -> data/labels/MASI09/MASI09_Becarios_labels.csv

  En esos CSV por edición solo aparecen los CVs de ESA edición.

- Cada fichero *_labels.csv tiene formato:
      name,label
      nombre normalizado,0|1

- Además genera un labels.csv global en data/labels/ combinando todo
  y muestra por pantalla nº de 0/1 por edición y total.

- NUEVO: para cada CSV de seleccionados, muestra por pantalla
  los candidatos del CSV que NO tienen CV normalizado en esa edición,
  y al final un resumen global de todos los candidatos "sin CV".
"""

import os
import sys

# Define Project Root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add to sys.path to allow 'from src.X import Y'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re
import json
import logging
from typing import List, Dict, Tuple

import pandas as pd

from src.preprocess import normalize_name  # usamos la misma normalización que en el resto del proyecto

import unicodedata

def _deaccent(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

def _norm_col(c: str) -> str:
    """
    Normaliza nombres de columnas para poder detectarlas:
    - minúsculas
    - sin tildes
    - reemplaza 'º' por 'o'
    - quita espacios extra y puntuación suave
    """
    c = (c or "").strip().lower()
    c = _deaccent(c)
    c = c.replace("º", "o")
    c = re.sub(r"[_\-\.\s]+", " ", c)
    return c.strip()

def _read_csv_any(path: str) -> pd.DataFrame:
    """Lee CSV robusto (coma, punto y coma, tab; utf-8 / utf-8-sig / latin-1)."""
    encs = ["utf-8-sig", "utf-8", "latin-1"]
    seps = [None, ";", ",", "\t", "|"]
    last = None
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, sep=sep, engine="python", encoding=enc)
                if not df.empty:
                    return df
            except Exception as e:
                last = e
                continue
    raise RuntimeError(f"No pude leer {path}: {last}")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- RUTAS ------------------
NORMALIZED_DIR = os.path.join(PROJECT_ROOT, "data", "normalized")
SELECTED_DIR   = os.path.join(PROJECT_ROOT, "data", "selected")
LABELS_DIR     = os.path.join(PROJECT_ROOT, "data", "labels")
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


def _infer_edition_from_root_and_base(root: str, base: str) -> str:
    """
    Intenta inferir la edición MASIxx a partir de:
    - la subcarpeta dentro de data/normalized (MASI09, MASI10, ...)
    - si no, del propio nombre de fichero.
    """
    rel_root = os.path.relpath(root, NORMALIZED_DIR)
    if rel_root != ".":
        first = rel_root.split(os.sep)[0]
        m = EDITION_RE.search(first)
        if m:
            return m.group(1).upper()

    m2 = EDITION_RE.search(base)
    if m2:
        return m2.group(1).upper()

    return "UNKNOWN"


def load_all_cvs() -> List[Dict]:
    """
    Carga TODOS los CVs normalizados de data/normalized **incluyendo subcarpetas**
    (MASI09, MASI10, ...).

    Devuelve lista de dicts con:
        - norm_name    -> nombre normalizado (minúsculas, sin tildes, clave de matching)
        - display_name -> mismo texto pero “bonito” (para ver en CSV)
        - edition      -> MASIxx o 'UNKNOWN'
    """
    cvs: List[Dict] = []
    any_file = False

    for root, dirs, files in os.walk(NORMALIZED_DIR):
        for fname in files:
            if not fname.endswith("_normalized.json"):
                continue

            any_file = True
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cv = json.load(f)
            except Exception as e:
                logger.error("Error leyendo %s: %s", path, e)
                continue

            base = fname.replace("_normalized.json", "")

            pi_norm = cv.get("personal_information_normalized") or {}
            pi_raw  = cv.get("personal_information") or {}
            full_name = (
                pi_norm.get("full_name")
                or pi_raw.get("full_name")
                or base
            )

            display_name = _beautify_name(full_name)
            norm_name = normalize_name(display_name)
            edition = _infer_edition_from_root_and_base(root, base)

            cvs.append({
                "display_name": display_name,
                "norm_name": norm_name,
                "edition": edition,
            })

    if not any_file:
        logger.warning("No se encontraron CVs en %s", NORMALIZED_DIR)
    else:
        logger.info("Encontrados %d CVs normalizados.", len(cvs))

    return cvs


def build_selected_name_set(csv_path: str) -> Tuple[set, Dict[str, str]]:
    """
    Carga un CSV de seleccionados/admitidos y devuelve:
      - set de nombres normalizados (admitidos = 1)
      - dict norm_name -> nombre 'bonito' tal y como aparece

    Soporta formatos:
      - Inglés: 'First Name', 'Middle Name', 'Last Name'
      - Español MASI25: '1º APELLIDO'; '2º APELLIDO'; 'NOMBRE' (separador ';')
      - Variantes: 'PRIMER APELLIDO', 'SEGUNDO APELLIDO', 'APELLIDOS', 'NOMBRE Y APELLIDOS', etc.
    """
    fname = os.path.basename(csv_path)
    logger.info("Leyendo CSV de seleccionados: %s", csv_path)

    df = _read_csv_any(csv_path)
    if df.empty:
        logger.warning("CSV vacío: %s", csv_path)
        return set(), {}

    # mapa normalizado de nombre de columna -> original
    colmap = { _norm_col(c): c for c in df.columns }

    # --- Detectar columnas de nombre ---
    # Caso MASI25 típico
    cand_nombre  = next((colmap[x] for x in ["nombre","name","first name","nombre alumno","alumno","estudiante"] if x in colmap), None)
    cand_ap1     = next((colmap[x] for x in ["1o apellido","1 apellido","primer apellido","apellido1","apellido 1","apellido","apellidos","last name"] if x in colmap), None)
    cand_ap2     = next((colmap[x] for x in ["2o apellido","2 apellido","segundo apellido","apellido2","apellido 2"] if x in colmap), None)

    # O columna combinada
    combinadas = ["apellidos y nombre", "nombre y apellidos", "full name", "fullname"]
    cand_full  = next((colmap[x] for x in combinadas if x in colmap), None)

    if cand_full:
        series_full = df[cand_full].astype(str)
    else:
        # Si tenemos 'APELLIDOS' única, la usamos tal cual como "apellidos"
        if cand_ap1 and _norm_col(cand_ap1) == "apellidos" and not cand_ap2:
            apellidos = df[cand_ap1].astype(str).fillna("").str.strip()
            nombre = df[cand_nombre].astype(str).fillna("").str.strip() if cand_nombre else ""
            series_full = (nombre + " " + apellidos).str.replace(r"\s+", " ", regex=True).str.strip()
        else:
            nombre  = df[cand_nombre].astype(str).fillna("").str.strip() if cand_nombre else ""
            ap1     = df[cand_ap1].astype(str).fillna("").str.strip() if cand_ap1 else ""
            ap2     = df[cand_ap2].astype(str).fillna("").str.strip() if cand_ap2 else ""
            series_full = (nombre + " " + ap1 + " " + ap2).str.replace(r"\s+", " ", regex=True).str.strip()

    # Filtra filas vacías
    series_full = series_full[series_full.astype(bool)]

    # Normaliza con tu normalize_name()
    norm = series_full.apply(normalize_name)
    name_set = set(norm.tolist())
    norm_to_pretty = {n: p for n, p in zip(norm.tolist(), series_full.tolist())}

    logger.info("CSV %s: %d admitidos detectados (columnas=%s)", fname, len(name_set), list(df.columns)[:6])
    return name_set, norm_to_pretty


def _infer_edition_from_selected_root(rel_root: str, csv_filename: str) -> str:
    """
    Intenta inferir MASIxx de:
    - la subcarpeta dentro de data/selected (MASI09, MASI10, ...)
    - si no, del nombre del CSV.
    """
    if rel_root:
        first = rel_root.split(os.sep)[0]
        m = EDITION_RE.search(first)
        if m:
            return m.group(1).upper()

    m2 = EDITION_RE.search(csv_filename)
    if m2:
        return m2.group(1).upper()

    return "UNKNOWN"


# --------- SOFT MATCHING (ya lo tenías) ---------
def _soft_match_name(cv_norm: str, selected_names: List[str]) -> bool:
    """
    Soft matching entre un nombre normalizado del CV y la lista de nombres
    normalizados del CSV de seleccionados.

    Regla:
      - mismo primer nombre
      - y al menos UN apellido en común
    """
    cv_norm = (cv_norm or "").strip()
    if not cv_norm:
        return False

    cv_tokens = cv_norm.split()
    if not cv_tokens:
        return False

    cv_first = cv_tokens[0]
    cv_surnames = cv_tokens[1:]
    if not cv_surnames:
        return False

    for sel in selected_names:
        sel = (sel or "").strip()
        if not sel:
            continue
        sel_tokens = sel.split()
        if not sel_tokens:
            continue

        sel_first = sel_tokens[0]
        sel_surnames = sel_tokens[1:]
        if not sel_surnames:
            continue

        if sel_first != cv_first:
            continue

        common = set(cv_surnames) & set(sel_surnames)
        if common:
            logger.debug("Soft match entre '%s' y '%s' (apellidos comunes: %s)",
                         cv_norm, sel, ",".join(common))
            return True

    return False


# --------- NUEVO: comprobar si un seleccionado tiene algún CV ---------
def _has_cv_match_for_selected(selected_norm: str, cvs_iter: List[Dict]) -> bool:
    """
    Devuelve True si el nombre normalizado selected_norm
    tiene algún CV asociado en la lista cvs_iter
    (por match exacto o soft).
    """
    # 1) match exacto
    for cv in cvs_iter:
        if cv["norm_name"] == selected_norm:
            return True

    # 2) soft match: cv_norm vs [selected_norm]
    for cv in cvs_iter:
        if _soft_match_name(cv["norm_name"], [selected_norm]):
            return True

    return False


def main() -> None:
    # 1) Cargar todos los CVs una sola vez
    all_cvs = load_all_cvs()
    if not all_cvs:
        logger.error("No hay CVs normalizados. Nada que etiquetar.")
        return

    # Mapa rápido norm_name -> edition para los conteos finales
    norm_to_edition: Dict[str, str] = {
        cv["norm_name"]: cv["edition"] for cv in all_cvs
    }

    # Para el labels.csv global:
    global_labels: Dict[str, int]   = {}   # clave: norm_name, valor: 0/1
    global_display: Dict[str, str] = {}   # norm_name -> display_name

    # NUEVO: candidatos sin CV (global)
    global_missing: Dict[str, str] = {}   # norm_name -> pretty_name

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

            # Conjunto de nombres seleccionados para ESTE CSV + mapa a nombre "bonito"
            selected_set, norm_to_pretty = build_selected_name_set(csv_path)
            if not selected_set:
                logger.warning("CSV %s no tiene nombres válidos. Se omite.", csv_path)
                continue

            selected_list = list(selected_set)

            # ¿De qué edición es este CSV? (para filtrar CVs)
            csv_edition = _infer_edition_from_selected_root(rel_root, f)
            if csv_edition != "UNKNOWN":
                cvs_iter = [cv for cv in all_cvs if cv["edition"] == csv_edition]
            else:
                cvs_iter = all_cvs

            logger.info("Etiquetando %d CVs contra %s (edición inferida: %s)",
                        len(cvs_iter), csv_path, csv_edition)

            # --------- NUEVO: detectar candidatos del CSV sin CV en esa edición ----------
            missing_for_csv: List[str] = []
            for sel_norm in selected_list:
                if not _has_cv_match_for_selected(sel_norm, cvs_iter):
                    pretty = norm_to_pretty.get(sel_norm, sel_norm)
                    missing_for_csv.append(pretty)
                    # también guardar en global
                    if sel_norm not in global_missing:
                        global_missing[sel_norm] = pretty

            if missing_for_csv:
                logger.warning(
                    "[%s] En %s hay %d candidatos SIN CV normalizado en esta edición.",
                    csv_edition, f, len(missing_for_csv)
                )
                # si quieres verlos todos, descomenta esta línea:
                logger.warning("    Candidatos sin CV: %s",
                               "; ".join(sorted(missing_for_csv)))
            else:
                logger.info("[%s] En %s todos los candidatos tienen CV normalizado.",
                            csv_edition, f)
            # --------------------------------------------------------------------------

            # Ruta de salida paralela en data/labels/
            base_name = os.path.splitext(f)[0]      # p.ej. MASI09_Becarios
            out_subdir = rel_root                   # ej: 'MASI09'
            out_dir = os.path.join(LABELS_DIR, out_subdir)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, base_name + "_labels.csv")

            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write("name,label\\n")
                for cv in cvs_iter:
                    # 1) match exacto
                    if cv["norm_name"] in selected_set:
                        label = 1
                    else:
                        # 2) soft match (nombre + algún apellido en común)
                        label = 1 if _soft_match_name(cv["norm_name"], selected_list) else 0

                    out_f.write(f"{cv['norm_name']},{label}\\n")

                    # actualizar mapa global (máximo: 0 -> 1)
                    norm = cv["norm_name"]
                    if norm not in global_labels:
                        global_labels[norm] = label
                        global_display[norm] = cv["norm_name"]
                    else:
                        if label > global_labels[norm]:
                            global_labels[norm] = label
                            global_display[norm] = cv["norm_name"]

            logger.info("Escrito fichero de etiquetas: %s", out_path)

    # 3) Escribir labels.csv global
    if global_labels:
        with open(GLOBAL_LABELS, "w", encoding="utf-8") as f:
            f.write("name,label\\n")
            for norm, label in global_labels.items():
                name = global_display.get(norm, norm)
                f.write(f"{name},{label}\\n")
        logger.info("Escrito labels global: %s (%d filas)",
                    GLOBAL_LABELS, len(global_labels))
    else:
        logger.warning("No se generó ninguna etiqueta global. ¿Hay CSVs en data/selected/?")

    # 4) Conteos por edición y total (usando labels globales finales)
    edition_counts: Dict[str, Dict[str, int]] = {}
    total_zero = 0
    total_one = 0

    for norm, label in global_labels.items():
        ed = norm_to_edition.get(norm, "UNKNOWN")
        if ed not in edition_counts:
            edition_counts[ed] = {"zeros": 0, "ones": 0}
        if label == 1:
            edition_counts[ed]["ones"] += 1
            total_one += 1
        else:
            edition_counts[ed]["zeros"] += 1
            total_zero += 1

    for ed in sorted(edition_counts.keys()):
        c0 = edition_counts[ed]["zeros"]
        c1 = edition_counts[ed]["ones"]
        logger.info("[%-7s] total=%d  ->  label 0: %d | label 1: %d",
                    ed, c0 + c1, c0, c1)

    logger.info("=== TOTAL MASTER ===  label 0: %d | label 1: %d  (total=%d)",
                total_zero, total_one, total_zero + total_one)

    # 5) Resumen global de candidatos sin CV
    if global_missing:
        logger.warning("=== CANDIDATOS SIN CV EN TODO EL MÁSTER ===")
        logger.warning("Total candidatos sin CV: %d", len(global_missing))
        # si no quieres que se llene la pantalla, puedes limitar:
        # aquí los saco todos; si quieres límite, se puede recortar.
        for pretty in sorted(global_missing.values()):
            logger.warning("  - %s", pretty)
    else:
        logger.info("Todos los candidatos de los CSV tienen algún CV normalizado asociado.")

if __name__ == "__main__":
    logger.info("Running test_label_all_cvs.py ...")
    main()
    logger.info("Finished.")