
import os
import joblib
import json
import numpy as np

# Config
base_dir = "output/features"
train_editions = [f"MASI{i:02d}" for i in range(9, 25)] # 09 to 24
test_edition = "MASI25"
mode = "combined"

def load_names(ed):
    path = os.path.join(base_dir, ed, mode, "feature_names.txt")
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

# 1. Vocab Analysis
print("--- Vocabulary Analysis ---")
train_union = set()
for ed in train_editions:
    s = load_names(ed)
    train_union.update(s)

test_names = load_names(test_edition)

print(f"Train Union Size: {len(train_union)}")
print(f"Test (MASI25) Size: {len(test_names)}")

missing_in_train = test_names - train_union
percent_missing = len(missing_in_train) / len(test_names) if len(test_names) else 0
print(f"Features in MASI25 NOT in Train: {len(missing_in_train)} ({percent_missing:.2%})")

# 2. Model Analysis
print("\n--- Model Analysis (Logistic Regression) ---")
model_path = "output/models/combined/recruitment_model_logisticregression.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    coefs = model.coef_[0]
    
    # Feature mapping: train_model.py sorts the union of keys
    feature_map = sorted(list(train_union))
    
    if len(feature_map) != len(coefs):
        print(f"WARNING: Model coefs ({len(coefs)}) != Train Union ({len(feature_map)}). Alignment mismatch!")
        # Try to read from meta if handy, but for now we assume sorted union.
    else:
        indices = np.argsort(coefs)
        
        print("Top 20 Negative Features (Predict REJECT):")
        for i in indices[:20]:
            print(f"  {feature_map[i]}: {coefs[i]:.4f}")
            
        print("\nTop 20 Positive Features (Predict ADMIT):")
        for i in indices[-20:][::-1]:
            print(f"  {feature_map[i]}: {coefs[i]:.4f}")
            
        # Check specific keywords
        keywords = ["2023", "2024", "2025", "python", "master", "english"]
        print("\nSpecific Keywords Check:")
        for k in keywords:
            if k in dict(zip(feature_map, coefs)):
                idx = feature_map.index(k)
                print(f"  {k}: {coefs[idx]:.4f}")
            else:
                print(f"  {k}: <NOT IN TRAIN VOCAB>")
else:
    print("Model file not found.")
