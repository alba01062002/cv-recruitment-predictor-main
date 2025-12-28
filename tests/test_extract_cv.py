#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:02:22 2025

@author: fran

Test script to extract raw text from CVs in supported formats (PDF, DOCX, DOC, TXT, XML).
Saves extracted data as JSONs in data/extracted_output/ for inspection.
Supports multilingual CVs (Spanish/English) with debug prints for Spyder.
"""
import sys
import os
import json
import nltk
import tempfile
import xml.etree.ElementTree as ET

# Ensure sys.path includes the src folder to import extract.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the extraction function
from extract import extract_cv_information  # type: ignore

# Download NLTK data if not present (common failure point)
try:
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"), quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK 'punkt' due to {e}. Some text processing might be affected.")

# --- Directory configuration ---
raw_cv_dir = os.path.join("data", "raw")
extracted_output_dir = os.path.join("data", "extracted_output")

# Create directories if they don't exist
os.makedirs(raw_cv_dir, exist_ok=True)
os.makedirs(extracted_output_dir, exist_ok=True)

print(f"--- Running Extraction Test for CVs in '{raw_cv_dir}' ---")

def _xml_to_plain_text(path: str) -> str:
    """Extrae texto plano de un XML (sin etiquetas)."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        parts = []

        def walk(node):
            if node.text and node.text.strip():
                parts.append(node.text.strip())
            for child in node:
                walk(child)
            if node.tail and node.tail.strip():
                parts.append(node.tail.strip())

        walk(root)
        return " ".join(parts).replace("  ", " ").strip()
    except Exception:
        # fallback: abrir como texto por si es XML simple
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return " ".join(f.read().split())

# --- Process each CV in raw_cv_dir ---
processed_count = 0
for filename in os.listdir(raw_cv_dir):
    file_path = os.path.join(raw_cv_dir, filename)

    # Skip directories or system/temporary files
    if not os.path.isfile(file_path) or filename.startswith('.'):
        continue

    # Filter by supported extensions (add .xml)
    supported_extensions = ('.pdf', '.docx', '.doc', '.txt', '.xml')
    if not filename.lower().endswith(supported_extensions):
        if filename.endswith('.json'):
            print(f"Skipping output file: {filename}")
            continue
        print(f"Skipping unsupported file type: {filename}")
        continue

    # --- check if already processed ---
    output_json_filename = os.path.splitext(filename)[0] + "_extracted_raw.json"
    output_json_path = os.path.join(extracted_output_dir, output_json_filename)
    if os.path.exists(output_json_path):
        print(f"\nSkipping already processed file: {filename}")
        continue

    print(f"\nProcessing file: {filename}")

    temp_txt_path = None
    try:
        # Si es XML, conviértelo a texto y pásalo a extract_cv_information como .txt temporal
        path_for_extractor = file_path
        if filename.lower().endswith(".xml"):
            plain = _xml_to_plain_text(file_path)
            # crear txt temporal
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(plain)
                temp_txt_path = tf.name
            path_for_extractor = temp_txt_path

        # Call the extraction function (igual que siempre)
        extracted_data = extract_cv_information(path_for_extractor)  # type: ignore
        processed_count += 1

        # Print key results for debugging
        print(f"  RAW TEXT (first 500 chars): {extracted_data.get('raw_text', '')[:500]}...")
        if extracted_data.get('full_name_from_header'):
            print(f"  DETECTED NAME FROM HEADER: {extracted_data['full_name_from_header']}")
        else:
            print(f"  NO NAME DETECTED FROM HEADER.")

        # Print detected language for multilingual debugging
        print(f"  DETECTED LANGUAGE: {extracted_data.get('detected_language', 'unknown')}")

        # Print any extraction errors
        if "error" in extracted_data.get("personal_information", {}) or "error" in extracted_data:
            print(f"  EXTRACTION WARNING/ERROR: {extracted_data.get('raw_text', 'No specific error message.')}")

        # Save the complete extracted output for inspection
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)

        print(f"  Extracted raw data saved to: {output_json_path}")

    except Exception as e:
        print(f"  CRITICAL ERROR processing {filename}: {e}")
        # Save error log for this file
        error_log_path = os.path.join(extracted_output_dir, os.path.splitext(filename)[0] + "_extraction_error.json")
        with open(error_log_path, 'w', encoding='utf-8') as f:
            json.dump({"filename": filename, "status": "failed_extraction", "error_message": str(e)}, f, indent=4, ensure_ascii=False)
    finally:
        # limpia el archivo temporal si se creó
        if temp_txt_path and os.path.exists(temp_txt_path):
            try:
                os.remove(temp_txt_path)
            except Exception:
                pass

if processed_count == 0:
    print(f"\nNo supported CV files found in '{raw_cv_dir}' to process. Please ensure files are in this directory.")
else:
    print(f"\n--- Extraction Test Completed. Processed {processed_count} files. ---")
    print(f"Check '{extracted_output_dir}' for detailed raw extraction outputs.")