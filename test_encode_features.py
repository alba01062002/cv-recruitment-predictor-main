#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejecuta la codificación TF-IDF por edición (combined y separate) con dedupe global
y genera un INFORME DE ALINEAMIENTO de columnas entre ediciones.

Salida clave por edición:
- output/features/MASIxx/{combined|separate}/X.npz, y.npy
- .../feature_names.txt  (ORDEN COMPLETO DE COLUMNAS)
- .../sections_order.txt (solo en separate)
- .../meta.json
- output/models/MASIxx/{combined|separate}/tfidf_*.joblib

Informe global:
- output/reports/tfidf/alignment_combined.csv
- output/reports/tfidf/alignment_separate.csv
- output/reports/tfidf/union_combined.txt
- output/reports/tfidf/union_separate.txt
- duplicates/kept/dropped (como antes)
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Dict, Set
import pandas as pd

from src.encode_features import (
    encode_features,
    _iter_normalized_jsons,
    _get_display_name,
    _norm_name_key,
)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG -----------------
NORMALIZED_ROOT = os.path.join("data", "normalized")
FEATURES_ROOT   = os.path.join("output", "features")
MODELS_ROOT     = os.path.join("output", "models")
LABELS_DIR      = os.path.join("data", "labels")
REPORTS_ROOT    = os.path.join("output", "reports", "tfidf")
os.makedirs(REPORTS_ROOT, exist_ok=True)

EDITION_RE = re.compile(r"^MASI(?P<yy>\d{2})$", re.IGNORECASE)

DEDUP_STRATEGY = (os.environ.get("DEDUP_STRATEGY", "latest") or "latest").strip().lower()
if DEDUP_STRATEGY not in {"latest", "first"}:
    DEDUP_STRATEGY = "latest"

# ---------------- HELPERS ----------------
def _find_editions(root: str) -> List[Tuple[str, str, int]]:
    """Devuelve lista (EDITION_TAG, abs_path, edition_num) para MASIxx con *_normalized.json"""
    editions = []
    if not os.path.isdir(root):
        logger.error(f"Normalized root not found: {root}")
        return editions

    for entry in sorted(os.listdir(root)):
        m = EDITION_RE.match(entry)
        if not m:
            continue
        yy = int(m.group("yy"))
        abs_dir = os.path.join(root, entry)
        if not os.path.isdir(abs_dir):
            continue
        has_json = any(fn.endswith("_normalized.json") for fn in os.listdir(abs_dir))
        if has_json:
            editions.append((entry.upper(), abs_dir, yy))
        else:
            logger.warning(f"[{entry}] no *_normalized.json found under {abs_dir} — skipping.")
    return editions

def _pick_labels_file(edition: str) -> Optional[str]:
    """Prioriza labels por edición y cae al consolidado."""
    candidates = [
        os.path.join(LABELS_DIR, f"{edition}_labels.csv"),
        os.path.join(LABELS_DIR, f"labels_{edition}.csv"),
        os.path.join(LABELS_DIR, "labels.csv"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            logger.info(f"[{edition}] Using labels file: {p}")
            return p
    logger.error(f"[{edition}] No labels found. Tried: {', '.join(candidates)}")
    return None

def _scan_name_keys_for_edition(edition: str, norm_dir: str, edition_num: int):
    """Escanea todos los *_normalized.json y produce registros (name_key, edition, edition_num, filename)."""
    rows = []
    for fn, cv in _iter_normalized_jsons(norm_dir):
        display = _get_display_name(cv, fn.replace("_normalized.json", ""))
        key = _norm_name_key(display)
        rows.append({
            "name_key": key,
            "edition": edition,
            "edition_num": edition_num,
            "filename": fn,
        })
    return rows

def _build_allowed_sets(editions: List[Tuple[str, str, int]]) -> Dict[str, Set[str]]:
    """Deduplicación global por name_key siguiendo DEDUP_STRATEGY (first|latest)."""
    all_rows = []
    for ed, norm_dir, ednum in editions:
        all_rows.extend(_scan_name_keys_for_edition(ed, norm_dir, ednum))

    df = pd.DataFrame(all_rows)
    if df.empty:
        logger.error("No normalized CVs found to build dedupe plan.")
        return {}

    dup_counts = df.groupby("name_key").size().reset_index(name="count")
    dup_only = dup_counts[dup_counts["count"] > 1]
    if not dup_only.empty:
        df_dup = df.merge(dup_only[["name_key"]], on="name_key", how="inner") \
                   .sort_values(["name_key", "edition_num"])
        df_dup.to_csv(os.path.join(REPORTS_ROOT, "duplicates.csv"), index=False, encoding="utf-8")
    else:
        df.iloc[0:0].to_csv(os.path.join(REPORTS_ROOT, "duplicates.csv"), index=False, encoding="utf-8")

    kept_rows, dropped_rows = [], []
    allowed: Dict[str, Set[str]] = {ed: set() for ed, _, _ in editions}

    for key, grp in df.groupby("name_key"):
        grp = grp.sort_values("edition_num")
        keep = grp.iloc[0] if DEDUP_STRATEGY == "first" else grp.iloc[-1]
        kept_rows.append(keep.to_dict())
        allowed[keep["edition"]].add(key)
        for _, row in grp.iterrows():
            if row["edition"] != keep["edition"] or row["filename"] != keep["filename"]:
                dropped_rows.append(row.to_dict())

    pd.DataFrame(kept_rows).sort_values(["name_key"]).to_csv(
        os.path.join(REPORTS_ROOT, "kept_samples.csv"), index=False, encoding="utf-8"
    )
    pd.DataFrame(dropped_rows).sort_values(["name_key", "edition_num"]).to_csv(
        os.path.join(REPORTS_ROOT, "dropped_samples.csv"), index=False, encoding="utf-8"
    )
    logger.info(
        "Dedup plan: total keys=%d | duplicated=%d | kept=%d | dropped=%d | strategy=%s",
        df["name_key"].nunique(),
        int(dup_only.shape[0]),
        len(kept_rows),
        len(dropped_rows),
        DEDUP_STRATEGY
    )
    return allowed

def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)

def _run_for_edition(edition: str, norm_dir: str, edition_num: int, vectorization_mode: str, allowed_set: Optional[Set[str]]) -> bool:
    labels_file = _pick_labels_file(edition)
    if labels_file is None:
        logger.error(f"[{edition}] Missing labels file. Skipping {vectorization_mode}.")
        return False

    features_dir = os.path.join(FEATURES_ROOT, edition, vectorization_mode)
    models_dir   = os.path.join(MODELS_ROOT, edition, vectorization_mode)
    _ensure_dirs(features_dir, models_dir)

    logger.info(f"[{edition}] Encoding ({vectorization_mode}) | allowed_keys={len(allowed_set) if allowed_set else 'ALL'}")
    logger.info(f"  normalized_dir = {norm_dir}")
    logger.info(f"  features_dir   = {features_dir}")
    logger.info(f"  models_dir     = {models_dir}")

    try:
        ok = encode_features(
            vectorization_mode=vectorization_mode,
            normalized_dir=norm_dir,
            features_dir=features_dir,
            models_dir=models_dir,
            labels_file=labels_file,
            allowed_name_keys=allowed_set
        )
        logger.info(f"[{edition}] {'OK' if ok else 'FAILED'} ({vectorization_mode})")
        return bool(ok)
    except Exception as e:
        logger.exception(f"[{edition}] Exception in encode_features ({vectorization_mode}): {e}")
        return False

# ---------- Informe de alineamiento ----------
def _read_feature_names(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def _alignment_report(editions: List[str], mode: str) -> None:
    rows = []
    unionset: Set[str] = set()
    for ed in editions:
        fdir = os.path.join(FEATURES_ROOT, ed, mode)
        fn_path = os.path.join(fdir, "feature_names.txt")
        meta    = os.path.join(fdir, "meta.json")
        if not os.path.isfile(fn_path):
            logger.warning(f"[{ed}/{mode}] feature_names.txt not found — skipping in alignment.")
            continue
        names = _read_feature_names(fn_path)
        unionset.update(names)
        nfeat = len(names)
        # meta (opcional)
        n_docs = None
        try:
            import json
            with open(meta, "r", encoding="utf-8") as f:
                n_docs = json.load(f).get("n_docs")
        except Exception:
            pass
        rows.append({"edition": ed, "mode": mode, "n_features": nfeat, "n_docs": n_docs})

    df = pd.DataFrame(rows).sort_values(["edition"])
    out_csv = os.path.join(REPORTS_ROOT, f"alignment_{mode}.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8")
    with open(os.path.join(REPORTS_ROOT, f"union_{mode}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(unionset)))
    logger.info(f"[alignment/{mode}] editions={len(df)} | union_features={len(unionset)}")
    logger.info(f"  -> {out_csv}")

# ---------------- EXECUTION ----------------
def test_encode_features_all_editions() -> bool:
    eds = _find_editions(NORMALIZED_ROOT)
    if not eds:
        logger.error(f"No MASI** editions found under {NORMALIZED_ROOT}")
        return False

    allowed_map = _build_allowed_sets(eds)
    if not allowed_map:
        logger.error("Failed to build dedupe plan.")
        return False

    any_ok = False
    ordered_eds: List[str] = []
    for edition, norm_dir, _ednum in eds:
        ordered_eds.append(edition)
        allowed_set = allowed_map.get(edition, None)
        ok_sep = _run_for_edition(edition, norm_dir, _ednum, "separate", allowed_set)
        ok_com = _run_for_edition(edition, norm_dir, _ednum, "combined", allowed_set)
        any_ok = any_ok or ok_sep or ok_com

    # Informe de alineamiento por modo
    _alignment_report(ordered_eds, "combined")
    _alignment_report(ordered_eds, "separate")

    if any_ok:
        logger.info("✅ Feature encoding (TF-IDF) completed with global dedupe applied.")
    else:
        logger.error("❌ Feature encoding failed for all editions/modes.")
    return any_ok

if __name__ == "__main__":
    test_encode_features_all_editions()