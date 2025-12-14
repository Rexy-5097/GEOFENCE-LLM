import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

# --- CONFIG ---
BASELINE_FEATURES = "data/phase5_5_windowed_baseline_feat.jsonl"
OBFUSCATED_FEATURES = "data/phase5_5_windowed_obf_feat.jsonl"
BASELINE_PROMPTS = "data/phase5_2_prompts.jsonl"

MODELS_DIR = "models/phase6_2_probes"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_LIST = ["directional_mean", "directional_var", "velocity_variance", "energy_drift", "tortuosity"]
SEED = 42

def load_features(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Recover original_id if missing
                if 'original_id' not in rec:
                    rec['original_id'] = rec['id'].rsplit('_w', 1)[0]
                data.append(rec)
    return data

def extract_arrays(records):
    X = np.array([[r[f] for f in FEATURES_LIST] for r in records])
    y = np.array([r['label'] for r in records])
    groups = np.array([r['original_id'] for r in records])
    return X, y, groups

def evaluate_agg(model, X, records, name):
    # Predict Window Probs
    probs_win = model.predict_proba(X)[:, 1]
    
    # Aggregate by Prompt (Max)
    res_map = {}
    for i, r in enumerate(records):
        oid = r['original_id']
        label = r['label']
        if oid not in res_map:
            res_map[oid] = {'probs': [], 'label': label}
        res_map[oid]['probs'].append(probs_win[i])
        
    y_true = []
    y_scores = []
    y_pred = []
    ids = []
    
    for oid, d in res_map.items():
        max_p = max(d['probs'])
        ids.append(oid)
        y_true.append(d['label'])
        y_scores.append(max_p)
        y_pred.append(1 if max_p >= 0.5 else 0)
        
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5
    rec = recall_score(y_true, y_pred, pos_label=1)
    
    print(f"  [{name}] Acc: {acc:.4f}, AUC: {auc:.4f}, Recall: {rec:.4f} (N={len(y_true)})")
    return {"acc": acc, "auc": auc, "rec": rec, "y_true": y_true, "y_scores": y_scores, "ids": ids}

def main():
    print("=== GEOFENCE-LLM Phase 6.2: Adversarial Retraining ===")
    
    # 1. Load Data
    print("Loading Features...")
    clean_recs = load_features(BASELINE_FEATURES)
    obf_recs = load_features(OBFUSCATED_FEATURES)
    
    all_recs = clean_recs + obf_recs
    
    # 2. Split (Group Aware on prompt ID)
    # We want to perform the split on the Unified Set of prompt IDs available in CLEAN (since Obf is derived)
    # Actually, simpler: Just GroupShuffleSplit the combined array. 
    # Just need to make sure 'groups' array is correct.
    
    X_all, y_all, groups_all = extract_arrays(all_recs)
    
    print(f"Total Windows: {len(X_all)}")
    print("Splitting Data (Group-Aware)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups_all))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    # We won't use X_test directly for mixed eval, we want specific subsets
    
    # Identify Test IDs
    test_ids_set = set(groups_all[test_idx])
    
    # 3. Create Test Subsets (Clean vs Obf)
    print(f"Train Windows: {len(X_train)} (Prompts: {len(set(groups_all[train_idx]))})")
    
    test_recs_clean = [r for r in clean_recs if r['original_id'] in test_ids_set]
    test_recs_obf = [r for r in obf_recs if r['original_id'] in test_ids_set]
    
    print(f"Test Clean Windows: {len(test_recs_clean)}")
    print(f"Test Obf Windows:   {len(test_recs_obf)}")
    
    X_test_clean, _, _ = extract_arrays(test_recs_clean)
    X_test_obf, _, _ = extract_arrays(test_recs_obf)
    
    # 4. Scale
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_clean_scaled = scaler.transform(X_test_clean)
    X_test_obf_scaled = scaler.transform(X_test_obf)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # 5. Train Models
    models = {
        "LogisticRegression": LogisticRegression(random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED),
        "MLP": MLPClassifier(hidden_layer_sizes=(16,8), max_iter=200, early_stopping=True, random_state=SEED)
    }
    
    # Results storage for reporting
    results_lr = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        
        # Eval Clean
        res_c = evaluate_agg(model, X_test_clean_scaled, test_recs_clean, f"{name} - Test Clean")
        # Eval Obf
        res_o = evaluate_agg(model, X_test_obf_scaled, test_recs_obf, f"{name} - Test Obf")
        
        if name == "LogisticRegression":
            results_lr["clean"] = res_c
            print("  Feature Coefficients:")
            for f, c in zip(FEATURES_LIST, model.coef_[0]):
                print(f"    {f}: {c:.4f}")

    # 6. Evaluate Length Shift (Test Clean)
    print("\n--- Evaluating Length Shift (Test Clean, LR) ---")
    
    # Load lengths
    lens = {}
    with open(BASELINE_PROMPTS, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lens[r['id']] = r['length']
                
    lr_res = results_lr["clean"]
    ids = lr_res['ids']
    lengths = np.array([lens[oid] for oid in ids])
    median = np.median(lengths)
    print(f"Median Length (Test Set): {median}")
    
    y_true = np.array(lr_res['y_true'])
    y_scores = np.array(lr_res['y_scores'])
    
    short_mask = lengths <= median
    long_mask = lengths > median
    
    auc_short = roc_auc_score(y_true[short_mask], y_scores[short_mask])
    auc_long = roc_auc_score(y_true[long_mask], y_scores[long_mask])
    
    print(f"  Short AUC: {auc_short:.4f}")
    print(f"  Long AUC:  {auc_long:.4f}")
    print(f"  Delta:     {abs(auc_short - auc_long):.4f}")

if __name__ == "__main__":
    main()
