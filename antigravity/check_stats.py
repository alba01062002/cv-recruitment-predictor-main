
import os
import numpy as np
import scipy.sparse as sp

base_dir = "output/features"
editions = [d for d in os.listdir(base_dir) if d.startswith("MASI")]
editions.sort()

print(f"{'Edition':<10} {'Mode':<10} {'Docs':<6} {'Features':<8} {'Positives':<10} {'Pos_Rate':<8}")
print("-" * 60)

for ed in editions:
    for mode in ["combined", "separate"]:
        path = os.path.join(base_dir, ed, mode)
        y_path = os.path.join(path, "y.npy")
        x_path = os.path.join(path, "X.npz")
        
        if os.path.exists(y_path) and os.path.exists(x_path):
            y = np.load(y_path)
            try:
                X = sp.load_npz(x_path)
                n_docs, n_feats = X.shape
            except:
                n_docs, n_feats = (0,0)
                
            n_pos = y.sum()
            pos_rate = n_pos / len(y) if len(y) > 0 else 0
            
            print(f"{ed:<10} {mode:<10} {n_docs:<6} {n_feats:<8} {n_pos:<10} {pos_rate:.2%}")
