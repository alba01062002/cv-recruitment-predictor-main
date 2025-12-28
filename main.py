import os
import json
import pandas as pd
from src.extract import extract_cv_information
from src.preprocess import generate_labels
from src.structure import normalize_cv_data, save_to_json
from refine_cv_ollama import refine_cv_with_llm  # For Grok/OpenAI compatibility


# Adjust sys.path to include the project root (assumed in execution context)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define directories and parameters
cv_directory = os.path.join("data", "raw")
output_dir = "data"
processed_dir = os.path.join(output_dir, "processed")
refined_dir = os.path.join(output_dir, "refined")
csv_path = os.path.join(cv_directory, "MASI09_Becarios.csv")
api_key_grok = os.getenv("OPENAI_API_KEY")  # For Grok (xAI uses OpenAI-compatible SDK)
api_key_gemini = os.getenv("GOOGLE_API_KEY")  # For Gemini (university account)
api_key_gemini = "AIzaSyB62qCmpwU2rKH0U5mPBBjPKa5DY7sEnr4"

print("Running CV processing pipeline in debug mode...")

# Check input files
if not os.path.exists(cv_directory):
    print(f"Warning: Directory {cv_directory} not found. Please create it and add CV files.")
if not os.path.exists(csv_path):
    print(f"Warning: CSV not found at {csv_path}. Please place MASI09_Becarios.csv in data/raw/")

# Get list of CV files
cv_files = [f for f in os.listdir(cv_directory) if f.lower().endswith(('.pdf', '.docx', '.txt'))]
if not cv_files:
    print(f"No CV files found in {cv_directory}. Please add PDF, DOCX, or TXT files.")
else:
    print(f"Processing {len(cv_files)} CV files...")

# Process each CV
all_labels = []
for filename in cv_files:
    file_path = os.path.join(cv_directory, filename)
    
    # Extract data
    cv_data = extract_cv_information(file_path)
    print(f"Raw extracted data for {filename}: {cv_data.__dict__}")

    # Normalize data
    cv_data = normalize_cv_data(cv_data)
    print(f"Normalized data for {filename}: {cv_data.__dict__}")

    # Save initial JSON
    json_filename = os.path.splitext(filename)[0] + ".json"
    json_path = os.path.join(processed_dir, json_filename)
    os.makedirs(processed_dir, exist_ok=True)
    save_to_json(cv_data.__dict__, json_path)
    print(f"Initial JSON saved at {json_path}")

    # Refine with LLM (Grok or Gemini, optional)
    refined_cv = cv_data.__dict__
    if api_key_grok:
        print(f"Refining {filename} with Grok (xAI) API...")
        refined_cv = refine_cv_with_llm(cv_data.__dict__, api_key_grok)
    elif api_key_gemini:
        print(f"Refining {filename} with Gemini API...")
        genai.configure(api_key=api_key_gemini)
        model = genai.GenerativeModel('gemini-1.5-flash')
        raw_text = cv_data.__dict__.get('raw_text', '')
        prompt = f"Refina este CV: {raw_text[:4000]}. Output JSON con name, education, experience, skills, languages, fit_score (0-10 para internship ingeniería)."
        response = model.generate_content(prompt)
        refined_json = json.loads(response.text)
        refined_cv = {**cv_data.__dict__, **refined_json, 'refined': True}
        print(f"Gemini refined data for {filename}: {refined_cv}")
    else:
        print(f"Skipping LLM refinement for {filename} (no API key provided).")

    # Save refined JSON
    refined_filename = json_filename.replace('.json', '_refined.json')
    refined_path = os.path.join(refined_dir, refined_filename)
    os.makedirs(refined_dir, exist_ok=True)
    save_to_json(refined_cv, refined_path)
    print(f"Refined JSON saved at {refined_path}")

    # Generate labels
    becarios_df = pd.read_csv(csv_path)
    labels_df = generate_labels(refined_cv, becarios_df)
    all_labels.append(labels_df)
    print(f"Generated labels for {filename}: {labels_df.to_dict()}")

# Combine and save all labels
if all_labels:
    labels_df = pd.concat(all_labels, ignore_index=True)
    labels_path = os.path.join(output_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    print(f"Labels saved to {labels_path}")
else:
    print("No labels generated due to missing CV files or CSV.")

print("Pipeline execution completed.")