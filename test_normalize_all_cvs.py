#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizes refined CV JSONs from data/refined/<edition>/ and saves to data/normalized/<edition>/.
Validates: CEFR languages only, no Erasmus entries left in education, and logs key summaries.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional

from src.normalization import normalize_llm_cv_output

# ---------------- Logging ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- Paths ---------------- #
REFINED_ROOT = os.path.join("data", "refined")
NORMALIZED_ROOT = os.path.join("data", "normalized")
os.makedirs(NORMALIZED_ROOT, exist_ok=True)

# ---------------- Helpers ---------------- #
_CEFR = {"A1", "A2", "B1", "B2", "C1", "C2"}
# acepta "MASI10", "masi10", "MASI10_Becarios", o rutas anidadas ".../masi11/..."
_EDITION_FLEX_RE = re.compile(r'(MASI(?P<yy>\d{2}))', re.IGNORECASE)

def _edition_from_relpath(rel_path: str) -> str:
    """
    Devuelve la etiqueta de edición en minúsculas 'masi09', 'masi10', etc.
    Si no se detecta, devuelve 'unknown'.
    """
    rel_norm = rel_path.replace(os.sep, '/')
    m = _EDITION_FLEX_RE.search(rel_norm)
    if m:
        return m.group(1).lower()  # 'masi09'
    # fallback: intenta tomar la carpeta inmediata bajo data/refined
    parts = rel_norm.split('/')
    if parts:
        # busca un componente que parezca masiXX
        for p in parts:
            if re.match(r'^MASI\d{2}$', p, flags=re.IGNORECASE):
                return p.lower()
    return "unknown"

def _save(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def _validate(normalized: Dict, filename: str) -> List[str]:
    errors: List[str] = []

    # 1) Idiomas en CEFR (languages_normalized debe ser [[lang, CEFR], ...])
    for pair in normalized.get("languages_normalized", []):
        if not (isinstance(pair, list) and len(pair) == 2 and str(pair[1]).upper() in _CEFR):
            errors.append(f"[{filename}] language not CEFR-normalized: {pair}")

    # 2) Erasmus no debe quedar en education_normalized
    for e in normalized.get("education_normalized", []):
        blob = " ".join(str(e.get(k, "")) for k in ("degree", "field", "year")).lower()
        if "erasmus" in blob:
            errors.append(f"[{filename}] Erasmus entry remained in education_normalized.")

    # 3) degree_years en rango razonable
    deg = normalized.get("degree_years", 0.0)
    if not isinstance(deg, (int, float)) or deg < 0 or deg > 10:
        errors.append(f"[{filename}] degree_years abnormal: {deg}")

    return errors

# ---------------- Main ---------------- #
def main():
    if not os.path.isdir(REFINED_ROOT):
        logger.error(f"Refined root not found: {REFINED_ROOT}")
        return

    processed = 0
    issues = 0
    seen_editions = set()

    logger.info(f"Scanning refined CVs under: {REFINED_ROOT}")

    for root, _, files in os.walk(REFINED_ROOT):
        for fn in files:
            if not fn.endswith("_refined.json"):
                continue

            in_path = os.path.join(root, fn)
            rel_path = os.path.relpath(in_path, REFINED_ROOT)  # e.g., "masi10/John_refined.json"
            edition = _edition_from_relpath(rel_path)          # e.g., "masi10"
            seen_editions.add(edition)

            out_dir = os.path.join(NORMALIZED_ROOT, edition)
            os.makedirs(out_dir, exist_ok=True)

            out_path_ok = os.path.join(out_dir, fn.replace("_refined.json", "_normalized.json"))
            out_path_err = os.path.join(out_dir, fn.replace("_refined.json", "_normalized_error.json"))

            # Carga el refinado
            try:
                with open(in_path, "r", encoding="utf-8") as f:
                    refined = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read JSON: {in_path} | {e}")
                continue

            # Normaliza
            try:
                normalized = normalize_llm_cv_output(refined)
            except Exception as e:
                logger.error(f"Normalization failed for {rel_path}: {e}")
                _save(out_path_err, {"error": str(e), "filename": rel_path})
                issues += 1
                continue

            # Si el refinado no llevaba master_edition, la inferimos de la ruta
            normalized.setdefault("master_edition", edition.upper())

            # Valida
            errs = _validate(normalized, rel_path)

            # Guarda segun validación
            if errs:
                _save(out_path_err, normalized)
                logger.warning(f"{rel_path} → validation issues: {errs}")
                issues += 1
            else:
                _save(out_path_ok, normalized)
                processed += 1
                logger.info(
                        f"{rel_path}: degree_years={normalized.get('degree_years')}, "
                        f"work={normalized.get('total_work_years')}, "
                        f"intl_has={normalized.get('has_international_experience')}, "
                        f"age_grad={normalized.get('age_at_graduation')}, "
                        f"langs={normalized.get('languages_normalized')}"
                )

    if not seen_editions:
        logger.warning("No refined files found. Ensure you have data/refined/<MASIXX>/*_refined.json")
    else:
        logger.info(f"Editions found: {', '.join(sorted(seen_editions))}")

    logger.info(f"Done. OK={processed}, Issues={issues}, Output root={NORMALIZED_ROOT}")

if __name__ == "__main__":
    main()