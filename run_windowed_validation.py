import os
import sys
import json
import numpy as np
import joblib
import subprocess
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
import time

# --- CONFIG ---
BASELINE_PROMPTS = "data/phase5_2_prompts.jsonl"
OBFUSCATED_PROMPTS = "data/phase5_3_obfuscated_prompts.jsonl"

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
WINDOW_LEN = 32
STRIDE = 16

PROBE_PATH = "models/phase5_2_probes/logistic_regression.joblib"
SCALER_PATH = "models/phase5_2_probes/scaler.joblib"
# Features expected by the scaler/model in order
FEATURES_LIST = ["directional_mean", "directional_var", "velocity_variance", "energy_drift", "tortuosity"]

def generate_windows(input_file, output_file, tokenizer):
    print(f"Generating windows for {input_file} -> {output_file}")
    t0 = time.time()
    
    with open(input_file, 'r') as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    
    window_records = []
    
    for r in prompts:
        text = r['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # If short, just 1 window
        if len(tokens) <= WINDOW_LEN:
            window_records.append({
                "id": f"{r['id']}_w0",
                "text": text, # Use original text
                "original_id": r['id'],
                "label": r['label'],
                "window_idx": 0
            })
            continue
            
        # Sliding Window
        # range(start, stop, step)
        # stop needs to go far enough to include the last partial window if needed?
        # Actually, let's just slide until end.
        
        idx = 0
        for i in range(0, len(tokens), STRIDE):
            chunk = tokens[i : i + WINDOW_LEN]
            if not chunk:
                break
            
            # Decode back to text for extraction
            window_text = tokenizer.decode(chunk)
            
            window_records.append({
                "id": f"{r['id']}_w{idx}",
                "text": window_text,
                "original_id": r['id'],
                "label": r['label'],
                "window_idx": idx
            })
            idx += 1
            
            if i + WINDOW_LEN >= len(tokens):
                break
                
    t1 = time.time()
    total_time = t1 - t0
    n_prompts = len(prompts)
    n_windows = len(window_records)
    
    print(f"Generated {n_windows} windows from {n_prompts} prompts. Time: {total_time:.2f}s")
    print(f"Mean Windows/Prompt: {n_windows/n_prompts if n_prompts else 0:.2f}")
    
    with open(output_file, 'w') as f:
        for w in window_records:
            f.write(json.dumps(w) + "\n")
            
    return n_windows, total_time, n_prompts

def run_extraction_geometry(prompts_file, traj_file, feat_file):
    # Extract
    if os.path.exists(traj_file):
        os.remove(traj_file) # Ensure fresh
        
    print(f"Extracting trajectories: {prompts_file} -> {traj_file}")
    t0_ext = time.time()
    subprocess.check_call([
        ".venv/bin/python", "extract_trajectories.py",
        "--input", prompts_file,
        "--output", traj_file
    ])
    t1_ext = time.time()
    
    # Geometry
    if os.path.exists(feat_file):
        os.remove(feat_file)
        
    print(f"Computing geometry: {traj_file} -> {feat_file}")
    t0_geom = time.time()
    subprocess.check_call([
        ".venv/bin/python", "compute_geometry.py",
        "--input", traj_file,
        "--output", feat_file
    ])
    t1_geom = time.time()
    
    return (t1_ext - t0_ext) + (t1_geom - t0_geom)

def evaluate_windows(features_file, original_prompts_file, name):
    print(f"\n--- Evaluating: {name} ---")
    
    # Load Model
    lr = joblib.load(PROBE_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load Features
    X_wins = []
    y_wins_dummy = [] # Labels exist but we care about aggregating
    win_meta = []
    
    with open(features_file, 'r') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                row = [rec[f] for f in FEATURES_LIST]
                X_wins.append(row)
                
                # Derive original_id from "id_w0" format
                if 'original_id' in rec:
                    win_meta.append(rec)
                else:
                    # Fallback
                    rec['original_id'] = rec['id'].rsplit('_w', 1)[0]
                    win_meta.append(rec)
    
    X_wins = np.array(X_wins)
    X_wins_scaled = scaler.transform(X_wins)
    
    # Predict Probabilities for all windows
    # Class 1 = Malicious
    probs = lr.predict_proba(X_wins_scaled)[:, 1]
    
    # Aggregate by original_id
    # Map original_id -> list of probs
    res_map = {} # id -> {'probs': [], 'label': label}
    
    for i, meta in enumerate(win_meta):
        oid = meta['original_id']
        label = meta['label']
        if oid not in res_map:
            res_map[oid] = {'probs': [], 'label': label}
        res_map[oid]['probs'].append(probs[i])
        
    # Stats
    prompt_ids = []
    y_true = []
    y_scores = []
    y_pred = []
    
    latencies = [] # Not tracked here directly, assuming acceptable logic
    
    for oid, data in res_map.items():
        # MAX Aggregation
        max_p = max(data['probs'])
        prompt_ids.append(oid)
        y_true.append(data['label'])
        y_scores.append(max_p)
        y_pred.append(1 if max_p >= 0.5 else 0)
        
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    rec = recall_score(y_true, y_pred, pos_label=1)
    
    print(f"Results for {name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Recall  : {rec:.4f}")
    
    return {"acc": acc, "auc": auc, "rec": rec, "y_true": y_true, "y_scores": y_scores, "ids": prompt_ids}

def evaluate_length_split(res_baseline, prompts_file):
    print("\n--- Evaluating Length Shift (Windowed) ---")
    
    # Load original lengths
    lens = {}
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                lens[rec['id']] = rec['length']
            
    # Map results
    y_true = np.array(res_baseline['y_true'])
    y_scores = np.array(res_baseline['y_scores'])
    ids = res_baseline['ids']
    
    # Assign lengths to result indices
    lengths_aligned = []
    
    for i, oid in enumerate(ids):
        lengths_aligned.append(lens.get(oid, 0))
        
    lengths_aligned = np.array(lengths_aligned)
    median_len = np.median(lengths_aligned)
    print(f"Median Length: {median_len}")
    
    short_mask = lengths_aligned <= median_len
    long_mask = lengths_aligned > median_len
    
    # Short
    y_short = y_true[short_mask]
    s_short = y_scores[short_mask]
    auc_short = roc_auc_score(y_short, s_short)
    rec_short = recall_score(y_short, (s_short>=0.5).astype(int), pos_label=1)
    
    # Long
    y_long = y_true[long_mask]
    s_long = y_scores[long_mask]
    auc_long = roc_auc_score(y_long, s_long)
    rec_long = recall_score(y_long, (s_long>=0.5).astype(int), pos_label=1)
    
    print(f"SHORT (N={len(y_short)}): AUC={auc_short:.4f}, Recall={rec_short:.4f}")
    print(f"LONG  (N={len(y_long)}) : AUC={auc_long:.4f}, Recall={rec_long:.4f}")
    print(f"Delta AUC: {abs(auc_short - auc_long):.4f}")
    
    return auc_short, auc_long

def main():
    print("=== GEOFENCE-LLM Phase 5.5: Windowed Validation ===")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Baseline Test
    base_win_prompts = "data/phase5_5_windowed_baseline_prompts.jsonl"
    base_traj = "data/phase5_5_windowed_baseline_traj.npz"
    base_feat = "data/phase5_5_windowed_baseline_feat.jsonl"
    
    n_wins, t_gen, n_p = generate_windows(BASELINE_PROMPTS, base_win_prompts, tokenizer)
    t_process = run_extraction_geometry(base_win_prompts, base_traj, base_feat)
    
    total_latency_ms = (t_gen + t_process) * 1000
    print(f"\n[Latency Report]")
    print(f"Total Processing Time: {total_latency_ms:.2f} ms for {n_p} prompts")
    print(f"Mean Latency/Prompt: {total_latency_ms/n_p:.2f} ms")
    
    res_base = evaluate_windows(base_feat, BASELINE_PROMPTS, "Baseline (Windowed)")
    evaluate_length_split(res_base, BASELINE_PROMPTS)
    
    # 2. Obfuscation Test
    obf_win_prompts = "data/phase5_5_windowed_obf_prompts.jsonl"
    obf_traj = "data/phase5_5_windowed_obf_traj.npz"
    obf_feat = "data/phase5_5_windowed_obf_feat.jsonl"
    
    generate_windows(OBFUSCATED_PROMPTS, obf_win_prompts, tokenizer)
    run_extraction_geometry(obf_win_prompts, obf_traj, obf_feat)
    evaluate_windows(obf_feat, OBFUSCATED_PROMPTS, "Obfuscated (Windowed)")

    # 3. Noise Test (on Baseline Window Features)
    # Load features, add noise, eval
    print("\n--- Evaluating: Noise Injection (Windowed) ---")
    data = []
    with open(base_feat, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]
        
    X = np.array([[r[k] for k in FEATURES_LIST] for r in data])
    meta = data
    
    scaler = joblib.load(SCALER_PATH)
    mean_std = np.std(scaler.transform(X), axis=0).mean()
    # Noise lvl 0.1
    noise = np.random.normal(0, 0.1, X.shape) 
    # Need to be careful: scaler expects raw input? No, scaler transforms raw. 
    # Noise should be added TO SCALED input or raw? 
    # Phase 5.3 added to SCALED input. 
    X_scaled = scaler.transform(X)
    X_noisy = X_scaled + np.random.normal(0, 0.1, X_scaled.shape)
    
    lr = joblib.load(PROBE_PATH)
    probs = lr.predict_proba(X_noisy)[:, 1]
    
    # Aggregate
    res_map = {}
    for i, r in enumerate(meta):
        oid = r.get('original_id', r['id'].rsplit('_w', 1)[0])
        label = r['label']
        if oid not in res_map: res_map[oid] = {'probs': [], 'label': label}
        res_map[oid]['probs'].append(probs[i])
        
    y_true, y_scores, y_pred = [], [], []
    for oid, d in res_map.items():
        max_p = max(d['probs'])
        y_true.append(d['label'])
        y_scores.append(max_p)
        y_pred.append(1 if max_p>=0.5 else 0)
        
    print(f"Noise Results: Acc={accuracy_score(y_true, y_pred):.4f}, AUC={roc_auc_score(y_true, y_scores):.4f}, Rec={recall_score(y_true, y_pred, pos_label=1):.4f}")

    print("\nPhase 5.5 Validation Complete.")

if __name__ == "__main__":
    main()
