#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalizes refined CV JSONs from data/refined and saves normalized JSONs to data/normalized.
Calculates years of experience, cleans errors, and structures languages as (language, level) tuples.

Created on Thu Sep 18 16:25:20 2025
@author: fran
"""
import os
import json
import logging
from typing import Dict

from src.structure import normalize_llm_cv_output, save_to_json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define directory constants
REFINED_DIR = os.path.join("data", "refined")
NORMALIZED_DIR = os.path.join("data", "normalized")

# Create directories if they don't exist
os.makedirs(REFINED_DIR, exist_ok=True)
os.makedirs(NORMALIZED_DIR, exist_ok=True)

def verify_normalized_cv(normalized_cv: Dict, filename: str) -> None:
    """
    Logs key fields from normalized CV for debugging.

    Args:
        normalized_cv (Dict): Normalized CV data.
        filename (str): Name of the processed file.
    """
    name = normalized_cv.get("personal_information", {}).get("full_name", "N/A")
    email = normalized_cv.get("personal_information", {}).get("email", "N/A")
    education = normalized_cv.get("education", [{}])[0].get("degree", "N/A")
    experience = normalized_cv.get("work_experience", [{}])[0].get("company", "N/A")
    total_work_years = normalized_cv.get("total_work_years", 0.0)
    languages = normalized_cv.get("languages", [])
    is_refined = normalized_cv.get("refined", False)

    logger.info(f"Verification for {filename}:")
    logger.info(f"  Full Name: {name}")
    logger.info(f"  Email: {email}")
    logger.info(f"  First Education (degree): {education}")
    logger.info(f"  First Experience (company): {experience}")
    logger.info(f"  Total Work Years: {total_work_years}")
    logger.info(f"  Languages: {languages}")
    logger.info(f"  Refined by LLM: {is_refined}")

def normalize_all_cvs() -> None:
    """
    Processes all JSONs in data/refined, normalizes them, and saves to data/normalized with _normalized.json suffix.
    """
    # Verify refined directory exists
    if not os.path.exists(REFINED_DIR):
        logger.error(f"Refined directory not found at {REFINED_DIR}")
        return

    processed_count = 0
    error_count = 0

    logger.info(f"Starting normalization of CVs from '{REFINED_DIR}'...")

    # Process all JSONs in refined_dir
    for filename in os.listdir(REFINED_DIR):
        if not filename.endswith("_refined.json"):
            logger.info(f"Skipping non-refined JSON file: {filename}")
            continue

        input_path = os.path.join(REFINED_DIR, filename)
        base_name = os.path.splitext(filename)[0].replace("_refined", "")
        output_path = os.path.join(NORMALIZED_DIR, f"{base_name}_normalized.json")

        # Skip if output already exists
        if os.path.exists(output_path):
            logger.info(f"Skipping '{filename}': Normalized output exists at '{output_path}'")
            processed_count += 1
            continue

        logger.info(f"Processing file: {filename}")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                llm_output = json.load(f)

            # Normalize CV data
            normalized_cv = normalize_llm_cv_output(llm_output)

            # Save normalized JSON
            save_to_json(normalized_cv, output_path)
            logger.info(f"Normalized and saved to: {output_path}")

            # Verify key fields
            verify_normalized_cv(normalized_cv, filename)
            processed_count += 1

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {filename}: {e}")
            error_count += 1
        except FileNotFoundError as e:
            logger.error(f"File not found for {filename}: {e}")
            error_count += 1
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            error_count += 1

    logger.info(f"Normalization completed. Processed {processed_count} CVs, {error_count} errors.")

# Execute the normalization process
normalize_all_cvs()