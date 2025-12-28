#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refina todos los CVs con Ollama y guarda la salida refinada en data/refined/<EDICION>/.
Aprovecha la caché de extracción en data/extracted_output/<EDICION>/... si existe.
"""

import sys
import os
import re
import json
import nltk
from src.extract import extract_cv_information
from src.structure import normalize_llm_cv_output, save_to_json
from src.refine_cv_ollama import refine_cv_with_llm, QuotaExceededError 

# Ajuste de paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# NLTK
try:
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"), quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK 'punkt' due to {e}. Extraction may fail.")

# Directorios
raw_root = os.path.join("data", "raw")
extracted_root = os.path.join("data", "extracted_output")
refined_root = os.path.join("data", "refined")

# Regex edición flexible (MASI## en cualquier parte del path)
_EDITION_FLEX_RE = re.compile(r'(MASI(?P<yy>\d{2}))', re.IGNORECASE)

def _detect_edition_from_relpath(rel_path: str) -> str:
    m = _EDITION_FLEX_RE.search(rel_path.replace(os.sep, '/'))
    if m:
        return f"MASI{m.group('yy')}"
    return "UNKNOWN"

print("Running test_refine_all_cvs.py (Ollama mode, recursive by edition)...")

if not os.path.exists(raw_root):
    print(f"Warning: Raw directory not found at {raw_root}. Please create it and add CVs.")
else:
    os.makedirs(extracted_root, exist_ok=True)
    os.makedirs(refined_root, exist_ok=True)

    processed_count = 0

    # Recorre recursivamente /data/raw
    for root, _, files in os.walk(raw_root):
        for filename in files:
            if filename.startswith('.'):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ('.pdf', '.docx', '.doc', '.txt', '.xml'):
                continue

            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, raw_root)
            edition = _detect_edition_from_relpath(rel_path)
            base_name = os.path.splitext(os.path.basename(filename))[0]

            # Rutas por edición
            extracted_cache_dir = os.path.join(extracted_root, edition)
            os.makedirs(extracted_cache_dir, exist_ok=True)
            cached_extraction_json_path = os.path.join(extracted_cache_dir, base_name + "_extracted_raw.json")

            refined_dir_for_edition = os.path.join(refined_root, edition)
            os.makedirs(refined_dir_for_edition, exist_ok=True)
            final_refined_json_path = os.path.join(refined_dir_for_edition, base_name + "_refined.json")
            final_error_json_path = os.path.join(refined_dir_for_edition, base_name + "_error.json")

            print(f"\n--- Processing {rel_path} (edition: {edition}) ---")

            try:
                # Skip si ya existe resultado final
                if os.path.exists(final_refined_json_path) or os.path.exists(final_error_json_path):
                    print(f"  Final result already exists for {rel_path}. Skipping.")
                    continue

                # Cargar extracción previa o hacer nueva
                if os.path.exists(cached_extraction_json_path):
                    print(f"  Loading raw extraction from cache: {cached_extraction_json_path}")
                    with open(cached_extraction_json_path, 'r', encoding='utf-8') as f:
                        cv_data_raw_text = json.load(f)
                else:
                    print(f"  No cached extraction found. Running extract_cv_information for {rel_path}...")
                    cv_data_raw_text = extract_cv_information(abs_path)
                    if not isinstance(cv_data_raw_text, dict) or 'raw_text' not in cv_data_raw_text:
                        raise ValueError(f"Expected a dictionary with 'raw_text' from extract_cv_information, got {type(cv_data_raw_text)}")
                    # fuerza edición desde la ruta
                    if "master_edition" not in cv_data_raw_text:
                        cv_data_raw_text["master_edition"] = edition
                    save_to_json(cv_data_raw_text, cached_extraction_json_path)
                    print(f"  Raw extraction result saved to cache: {cached_extraction_json_path}")

                # Asegura que lleve master_edition
                if "master_edition" not in cv_data_raw_text:
                    cv_data_raw_text["master_edition"] = edition

                # Validar texto antes de refinar
                if not cv_data_raw_text.get('raw_text'):
                    print(f"  Skipping LLM and normalization for {rel_path} due to empty raw_text.")
                    continue

                # Refinar con Ollama
                try:
                    llm_raw_output_cv = refine_cv_with_llm(cv_data_raw_text)
                except QuotaExceededError as qe:
                    print(f"  CRITICAL ERROR (local quota): {qe}")
                    save_to_json(
                        {"filename": rel_path, "status": "failed_quota_exceeded", "error_message": str(qe), "master_edition": edition},
                        final_error_json_path
                    )
                    break

                # Añadir idioma detectado si existe
                if "detected_language" in cv_data_raw_text:
                    llm_raw_output_cv['detected_language'] = cv_data_raw_text['detected_language']

                # Asegura master_edition en la salida del LLM
                if "master_edition" not in llm_raw_output_cv and "master_edition" in cv_data_raw_text:
                    llm_raw_output_cv["master_edition"] = cv_data_raw_text["master_edition"]

                # Normalizar y guardar resultado final por edición
                normalized_cv = normalize_llm_cv_output(llm_raw_output_cv)
                save_to_json(normalized_cv, final_refined_json_path)
                print(f"  Final normalized/refined CV saved at: {final_refined_json_path}")

                processed_count += 1

                # --- Verificación ---
                print("\n  --- Key fields verification (for debugging) ---")
                pi = normalized_cv.get('personal_information') or {}
                name = pi.get('full_name', 'N/A')
                email = pi.get('email', 'N/A')

                education = normalized_cv.get('education')
                if not isinstance(education, list):
                    education = []
                first_education_title = (education[0].get('degree') if education and isinstance(education[0], dict) else 'N/A')

                work = normalized_cv.get('work_experience')
                if not isinstance(work, list):
                    work = []
                first_experience_company = (work[0].get('company') if work and isinstance(work[0], dict) else 'N/A')

                is_refined_by_llm = bool(normalized_cv.get('refined'))
                edition_out = normalized_cv.get('master_edition', edition)

                print(f"    Full Name (final): {name}")
                print(f"    Email (final): {email}")
                print(f"    First Education (degree): {first_education_title}")
                print(f"    First Experience (company): {first_experience_company}")
                print(f"    Edition: {edition_out}")
                print(f"    LLM Refined Successfully: {is_refined_by_llm}")
                print("  --- End of verification ---\n")

            except Exception as e:
                print(f"  CRITICAL ERROR processing {rel_path}: {e}")
                save_to_json(
                    {"filename": rel_path, "status": "failed_pipeline", "error_message": str(e), "master_edition": edition},
                    final_error_json_path
                )

    print(f"\nDebug run completed. Processed {processed_count} CVs.")