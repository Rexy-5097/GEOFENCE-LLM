import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# --- CONFIG ---
BASELINE_FEATURES = "data/phase5_5_windowed_baseline_feat.jsonl"
OBFUSCATED_FEATURES = "data/phase5_5_windowed_obf_feat.jsonl" # For eval only
BASELINE_PROMPTS = "data/phase5_2_prompts.jsonl" # For length info

MODELS_DIR = "models/phase6_1_probes"
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES_LIST = ["directional_mean", "directional_var", "velocity_variance", "energy_drift", "tortuosity"]
SEED = 42

def load_features(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # Recover original_id if missing (legacy fix)
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
    
    for oid, d in res_map.items():
        max_p = max(d['probs'])
        y_true.append(d['label'])
        y_scores.append(max_p)
        y_pred.append(1 if max_p >= 0.5 else 0)
        
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    rec = recall_score(y_true, y_pred, pos_label=1)
    
    print(f"  [{name}] Acc: {acc:.4f}, AUC: {auc:.4f}, Recall: {rec:.4f} (N={len(y_true)})")
    return {"acc": acc, "auc": auc, "rec": rec, "y_true": y_true, "y_scores": y_scores, "ids": list(res_map.keys())}

def main():
    print("=== GEOFENCE-LLM Phase 6.1: Window-Aware Retraining ===")
    
    # 1. Load Data
    print("Loading Baseline Window Features...")
    base_recs = load_features(BASELINE_FEATURES)
    X_all, y_all, groups_all = extract_arrays(base_recs)
    
    # 2. Split (Group Aware)
    print("Splitting Data (Group-Aware)...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=SEED)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups_all))
    
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    
    train_recs = [base_recs[i] for i in train_idx]
    test_recs = [base_recs[i] for i in test_idx]
    
    # Identify Test Prompt IDs for filtering other datasets
    test_prompt_ids = set([r['original_id'] for r in test_recs])
    print(f"Train Windows: {len(X_train)} (Prompts: {len(set(groups_all[train_idx]))})")
    print(f"Test Windows:  {len(X_test)}  (Prompts: {len(test_prompt_ids)})")
    
    # 3. Scale
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    # 4. Train Models
    models = {
        "LogisticRegression": LogisticRegression(random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=SEED), # Slightly deeper for windows?
        "MLP": MLPClassifier(hidden_layer_sizes=(16,8), max_iter=200, early_stopping=True, random_state=SEED)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        
        # Eval on Baseline Test
        res = evaluate_agg(model, X_test_scaled, test_recs, f"{name} - Test Baseline")
        results[name] = res
        
        if name == "LogisticRegression":
            # Feature Importance
            print("  Feature Coefficients:")
            for f, c in zip(FEATURES_LIST, model.coef_[0]):
                print(f"    {f}: {c:.4f}")
    
    # 5. Evaluate on Obfuscated (Test Prompts Only)
    print("\n--- Evaluating on Obfuscated Data (Test Prompts Only) ---")
    obf_recs_all = load_features(OBFUSCATED_FEATURES)
    # Filter
    obf_recs_test = [r for r in obf_recs_all if r['original_id'] in test_prompt_ids]
    
    if not obf_recs_test:
        print("Warning: No obfuscated records matched test IDs. Check ID format.")
    
    X_obf, _, _ = extract_arrays(obf_recs_test)
    X_obf_scaled = scaler.transform(X_obf)
    
    for name, model in models.items():
        evaluate_agg(model, X_obf_scaled, obf_recs_test, f"{name} - Obfuscated")

    # 6. Evaluate Length Shift (Test Baseline)
    print("\n--- Evaluating Length Shift (Test Baseline) ---")
    # Using Logistic Regression results
    lr_res = results["LogisticRegression"]
    
    # Load lengths
    lens = {}
    with open(BASELINE_PROMPTS, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                lens[r['id']] = r['length']
                
    test_ids = lr_res['ids']
    lengths = np.array([lens[oid] for oid in test_ids])
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
