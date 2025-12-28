#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import nltk
import tempfile
import re
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from extract import extract_cv_information  # type: ignore

try:
    nltk.data.path.append(os.path.expanduser("~/nltk_data"))
    nltk.download('punkt', download_dir=os.path.expanduser("~/nltk_data"), quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK 'punkt': {e}")

RAW_ROOT = os.path.join("data", "raw")
OUT_ROOT = os.path.join("data", "extracted_output")
os.makedirs(RAW_ROOT, exist_ok=True)
os.makedirs(OUT_ROOT, exist_ok=True)

# acepta "MASI10", "MASI10_Becarios", "algo/MASI11_..." → devuelve "MASI10"/"MASI11"
_EDITION_FLEX_RE = re.compile(r'(MASI(?P<yy>\d{2}))', re.IGNORECASE)

def _xml_to_plain_text(path: str) -> str:
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
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return " ".join(f.read().split())

def _detect_edition_from_relpath(rel_path: str) -> str:
    m = _EDITION_FLEX_RE.search(rel_path.replace(os.sep, '/'))
    if m:
        return f"MASI{m.group('yy')}"
    return "UNKNOWN"

print(f"--- Running Extraction Test (recursive) in '{RAW_ROOT}' ---")

processed_count = 0
for root, _, files in os.walk(RAW_ROOT):
    for filename in files:
        if filename.startswith('.'):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.pdf', '.docx', '.doc', '.txt', '.xml'):
            if filename.endswith('.json'):
                print(f"Skipping output file: {filename}")
                continue
            print(f"Skipping unsupported file type: {filename}")
            continue

        abs_path = os.path.join(root, filename)
        rel_path = os.path.relpath(abs_path, RAW_ROOT)
        edition = _detect_edition_from_relpath(rel_path)
        base_name = os.path.splitext(os.path.basename(filename))[0]

        out_dir = os.path.join(OUT_ROOT, edition)
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, base_name + "_extracted_raw.json")

        if os.path.exists(out_json):
            print(f"\nSkipping already processed file: {rel_path}")
            continue

        print(f"\nProcessing file: {rel_path}  (edition: {edition})")

        temp_txt_path = None
        try:
            path_for_extractor = abs_path
            if ext == ".xml":
                plain = _xml_to_plain_text(abs_path)
                with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                    tf.write(plain)
                    temp_txt_path = tf.name
                path_for_extractor = temp_txt_path

            extracted = extract_cv_information(path_for_extractor)

            if not isinstance(extracted, dict):
                raise ValueError("Extractor returned non-dict (None or invalid).")

            # fuerza edición desde la ruta original (evita UNKNOWN por temporales)
            extracted["master_edition"] = edition

            processed_count += 1

            print(f"  RAW TEXT (first 500 chars): {extracted.get('raw_text', '')[:500]}...")
            if extracted.get('full_name_from_header'):
                print(f"  DETECTED NAME FROM HEADER: {extracted['full_name_from_header']}")
            else:
                print(f"  NO NAME DETECTED FROM HEADER.")
            print(f"  DETECTED LANGUAGE: {extracted.get('detected_language', 'unknown')}")

            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(extracted, f, indent=4, ensure_ascii=False)
            print(f"  Extracted raw data saved to: {out_json}")

        except Exception as e:
            print(f"  CRITICAL ERROR processing {rel_path}: {e}")
            err_dir = os.path.join(OUT_ROOT, edition)
            os.makedirs(err_dir, exist_ok=True)
            err_path = os.path.join(err_dir, base_name + "_extraction_error.json")
            with open(err_path, 'w', encoding='utf-8') as f:
                json.dump({"filename": rel_path, "status": "failed_extraction", "error_message": str(e)}, f, indent=4, ensure_ascii=False)
        finally:
            if temp_txt_path and os.path.exists(temp_txt_path):
                try:
                    os.remove(temp_txt_path)
                except Exception:
                    pass

if processed_count == 0:
    print(f"\nNo supported CV files found in '{RAW_ROOT}'. Put files under MASIxx subfolders (e.g., MASI10_Becarios/...).")
else:
    print(f"\n--- Extraction Test Completed. Processed {processed_count} files. ---")
    print(f"Check '{OUT_ROOT}/<EDITION>/' for detailed raw extraction outputs.")