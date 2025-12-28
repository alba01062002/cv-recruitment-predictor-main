
import os
import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score, confusion_matrix
import sys
# Add current dir to path to import src
sys.path.append(os.getcwd())
try:
    from src.train_model import _align_concat
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from train_model import _align_concat

MODEL_PATH = "output/models/combined/recruitment_model_lgbmclassifier.pkl" # LGBM saved name
FEATURES_ROOT = "output/features"
# UPDATED TRAIN EDITIONS for this run
TRAIN_EDITIONS = [f"MASI{i}" for i in range(18, 25)] # MASI18-MASI24
TEST_EDITIONS = ["MASI25"]
MODE = "combined"

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print("Reconstructing feature space from Training Editions:", TRAIN_EDITIONS)
    _, _, names_union = _align_concat(TRAIN_EDITIONS, FEATURES_ROOT, MODE)
    print(f"Training Feature Count: {len(names_union)}")
    
    print("Aligning Test Data (MASI25)...")
    X_test, y_test, _ = _align_concat(TEST_EDITIONS, FEATURES_ROOT, MODE, base_union=names_union)
    
    # LGBM handles sparse directly usually
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation on MASI25 (LGBM - MASI18+) ---")
    best_ba = 0
    best_res = None
    
    for thr in np.arange(0.20, 0.65, 0.05):
        preds = (probs >= thr).astype(int)
        ba = balanced_accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        print(f"Thr={thr:.2f} | BA={ba:.3f} | Recall={rec:.3f} | F1={f1:.3f} | CM={cm.ravel()}")
        if ba > best_ba:
            best_ba = ba
            best_res = (thr, ba, rec, f1, cm)

    if best_res:
        print(f"\n[BEST BA CONFIG] Thr={best_res[0]:.2f}: BA={best_res[1]:.3f} | Rec={best_res[2]:.3f} | F1={best_res[3]:.3f}")

if __name__ == "__main__":
    evaluate()
