#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:02:22 2025

@author: fran
"""

import sys
import os
import json
import nltk
from src.extract import extract_cv_information
from src.structure import normalize_llm_cv_output, save_to_json
from src.refine_cv_ollama import refine_cv_with_llm, QuotaExceededError 

# Adjust sys.path to include the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Download NLTK data if not present
try:
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"), quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK 'punkt' due to {e}. Extraction may fail.")

# Define directories
raw_dir = os.path.join("data", "raw")                  # Where original PDFs/DOCX/TXT/XML are stored
extracted_cache_dir = os.path.join("data", "extracted_output")  # Cache for extract.py output
final_refined_dir = os.path.join("data", "refined")     # For normalized and standardized CVs

print("Running test_refine_all_cvs.py (Ollama mode)...")

if not os.path.exists(raw_dir):
    print(f"Warning: Raw directory not found at {raw_dir}. Please create it and add CVs.")
else:
    # Create directories if they don't exist
    os.makedirs(extracted_cache_dir, exist_ok=True)
    os.makedirs(final_refined_dir, exist_ok=True)

    processed_count = 0

    # Process all CVs
    for filename in os.listdir(raw_dir):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in ('.pdf', '.docx', '.doc', '.txt', '.xml'):
            base_name = os.path.splitext(filename)[0]  # Name without extension
            file_path = os.path.join(raw_dir, filename)

            print(f"\n--- Processing {filename} ---")

            # Define cache file paths
            cached_extraction_json_path = os.path.join(extracted_cache_dir, base_name + "_extracted_raw.json")
            final_refined_json_path = os.path.join(final_refined_dir, base_name + "_refined.json")
            final_error_json_path = os.path.join(final_refined_dir, base_name + "_error.json")

            try:
                # Skip si ya existe resultado final
                if os.path.exists(final_refined_json_path) or os.path.exists(final_error_json_path):
                    print(f"  Final result already exists for {filename}. Skipping.")
                    continue

                # Cargar extracción previa o hacer nueva
                if os.path.exists(cached_extraction_json_path):
                    print(f"  Loading raw extraction from cache: {cached_extraction_json_path}")
                    with open(cached_extraction_json_path, 'r', encoding='utf-8') as f:
                        cv_data_raw_text = json.load(f)
                else:
                    print(f"  No cached extraction found. Running extract_cv_information for {filename}...")
                    cv_data_raw_text = extract_cv_information(file_path)
                    if not isinstance(cv_data_raw_text, dict) or 'raw_text' not in cv_data_raw_text:
                        raise ValueError(f"Expected a dictionary with 'raw_text' from extract_cv_information, got {type(cv_data_raw_text)}")
                    save_to_json(cv_data_raw_text, cached_extraction_json_path)
                    print(f"  Raw extraction result saved to cache: {cached_extraction_json_path}")

                # Validar texto antes de refinar
                if not cv_data_raw_text.get('raw_text'):
                    print(f"  Skipping LLM and normalization for {filename} due to empty raw_text.")
                    continue

                # Refinar con Ollama
                try:
                    llm_raw_output_cv = refine_cv_with_llm(cv_data_raw_text)
                except QuotaExceededError as qe:
                    print(f"  CRITICAL ERROR (local quota): {qe}")
                    save_to_json(
                        {"filename": filename, "status": "failed_quota_exceeded", "error_message": str(qe)},
                        final_error_json_path
                    )
                    break

                # Añadir idioma detectado si existe
                if "detected_language" in cv_data_raw_text:
                    llm_raw_output_cv['detected_language'] = cv_data_raw_text['detected_language']

                # Guardar directamente la salida refinada (sin pasar por processed)
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

                print(f"    Full Name (final): {name}")
                print(f"    Email (final): {email}")
                print(f"    First Education (degree): {first_education_title}")
                print(f"    First Experience (company): {first_experience_company}")
                print(f"    LLM Refined Successfully: {is_refined_by_llm}")
                print("  --- End of verification ---\n")

            except Exception as e:
                print(f"  CRITICAL ERROR processing {filename}: {e}")
                save_to_json(
                    {"filename": filename, "status": "failed_pipeline", "error_message": str(e)},
                    final_error_json_path
                )

    print(f"\nDebug run completed. Processed {processed_count} CVs.")