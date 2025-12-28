
import os
import numpy as np
import scipy.sparse as sp
import joblib
import pandas as pd

# Load features
FEATURES_ROOT = "output/features"
DENSE_FEATURE_NAMES = [
    "dense_degree_years",
    "dense_age_at_graduation",
    "dense_total_work_years",
    "dense_has_international_experience",
    "dense_has_master",
    "dense_n_hard_skills",
    "dense_n_languages",
    "dense_max_english_level"
]

def load_dense_data(editions):
    rows = []
    for ed in editions:
        path = os.path.join(FEATURES_ROOT, ed, "combined")
        x_path = os.path.join(path, "X.npz")
        y_path = os.path.join(path, "y.npy")
        
        if not os.path.exists(x_path): continue
        
        X = sp.load_npz(x_path)
        y = np.load(y_path)
        
        # Dense features are at the END. We know the count.
        n_dense = len(DENSE_FEATURE_NAMES)
        # Assuming the last n_dense cols are the dense ones
        X_dense = X[:, -n_dense:].toarray()
        
        # Unscale? tricky without the original scaler per edition, 
        # but let's just look at the stats as stored (which are scaled locally if we used StandardScaler inside encode? 
        # Wait, encode_features applies scaler globally per run? No, per execution of encode_features.
        # Actually in encode_features we calculate raw dense features then scale them.
        # If we want to compare distributions, we ideally want RAW values.
        # But we don't save raw X_dense separately. 
        # However, encode_features logs "Dense sample", let's trust the scaler was fit on that edition alone?
        # In encode_features: scaler.fit_transform(X_dense). Yes, it's per edition.
        # So comparing SCALED values is meaningless if they are all N(0,1).
        # PROBLEM: We typically normalize per SET (Train vs Test). 
        # If encode_features normalizes PER EDITION, then a value of "5 years" might be 0.0 in 2010 and 0.5 in 2025 depending on mean.
        # This is good for stationarity IF the semantic meaning of "mean" is constant.
        # BUT if 2025 has higher standards, normalizing per edition hides that strictness.
        
        # Let's try to infer if we can look at raw. 
        # We can't easily reverse without the scaler object. 
        # Scalers are saved in output/models/EDITION/combined/dense_scaler.joblib
        
        scaler_path = os.path.join("output", "models", ed, "combined", "dense_scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_raw = scaler.inverse_transform(X_dense)
            
            for i, feat_vals in enumerate(X_raw):
                row = {"edition": ed, "label": y[i]}
                for j, name in enumerate(DENSE_FEATURE_NAMES):
                    row[name] = feat_vals[j]
                rows.append(row)
                
    return pd.DataFrame(rows)

train_eds = [f"MASI{i:02d}" for i in range(9, 25)]
test_eds = ["MASI25"]

print("Loading Train Data...")
df_train = load_dense_data(train_eds)
print("Loading Test Data...")
df_test = load_dense_data(test_eds)

print(f"\nTRAIN samples: {len(df_train)}")
print(f"TEST samples: {len(df_test)}")

print("\n--- Feature Means (Admitted=1 vs Rejected=0) ---")
for col in DENSE_FEATURE_NAMES:
    print(f"\nFeature: {col}")
    
    # Train stats
    mu_tr_0 = df_train[df_train.label==0][col].mean()
    mu_tr_1 = df_train[df_train.label==1][col].mean()
    
    # Test stats
    mu_te_0 = df_test[df_test.label==0][col].mean()
    mu_te_1 = df_test[df_test.label==1][col].mean()
    
    print(f"  TRAIN -> Rej: {mu_tr_0:.2f} | Adm: {mu_tr_1:.2f} (Diff: {mu_tr_1-mu_tr_0:.2f})")
    print(f"  TEST  -> Rej: {mu_te_0:.2f} | Adm: {mu_te_1:.2f} (Diff: {mu_te_1-mu_te_0:.2f})")
    
    # Check if the "Admission Standard" shifted
    print(f"  Shift in Admitted Profile (Test - Train): {mu_te_1 - mu_tr_1:.2f}")

print("\n--- Model Coefficients (if available) ---")
model_path = "output/models/combined/recruitment_model_logisticregression.pkl"
try:
    model = joblib.load(model_path)
    # The model was trained on ALL features. We need to find the dense ones at result['n_features'] - 8
    # But names are in feature_names.txt used during training.
    # The last training run used TRAIN editions. We need the names from one of them.
    # Assuming standard sorting of features in train_model.py... 
    # Actually train_model aligns features by name.
    # So we can just print coefficients for DENSE_FEATURE_NAMES found in the model's feature names?
    # Wait, the model object itself doesn't store feature names usually.
    # We rely on the fact that DENSE features are appended at the end of feature_names.txt in encode_features.
    # If train_model respects that...
    pass
except:
    pass
