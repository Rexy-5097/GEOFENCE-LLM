import json
import time
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from geofence_pipeline import GeofenceDefensor

OBF_DATA = "data/phase5_3_obfuscated_prompts.jsonl"
SAFE_DATA = "data/phase5_2_prompts.jsonl" # Use label=0 subset

def main():
    print("=== GEOFENCE PIPELINE VERIFICATION ===")
    
    # 1. Initialize Pipeline
    defensor = GeofenceDefensor()
    
    # 2. Load Data
    # Load 50 Malicious Obfuscated
    mal_obf = []
    with open(OBF_DATA, 'r') as f:
        for line in f:
            if len(mal_obf) >= 50: break
            r = json.loads(line)
            if r['label'] == 1:
                mal_obf.append(r)
                
    # Load 50 Safe (Clean)
    safe_clean = []
    with open(SAFE_DATA, 'r') as f:
        for line in f:
            if len(safe_clean) >= 50: break
            r = json.loads(line)
            if r['label'] == 0:
                safe_clean.append(r)
                
    print(f"Loaded {len(mal_obf)} Malicious Obfuscated, {len(safe_clean)} Safe Clean.")
    
    # 3. Run Inference
    y_true = []
    y_pred = [] # 1 if blocked, 0 if allowed
    latencies = []
    
    # Run Malicious
    print("Scanning Malicious Obfuscated...")
    for r in mal_obf:
        res = defensor.scan(r['text'])
        y_true.append(1)
        y_pred.append(1 if res['blocked'] else 0)
        latencies.append(res['latency_ms'])
        
    # Run Safe
    print("Scanning Safe...")
    for r in safe_clean:
        res = defensor.scan(r['text'])
        y_true.append(0)
        y_pred.append(1 if res['blocked'] else 0)
        latencies.append(res['latency_ms'])
        
    # 4. Metrics
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=1) # Detection Rate
    fpr = 1 - recall_score(y_true, y_pred, pos_label=0) # Safe blocked rate
    
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    
    print("\n--- RESULTS ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (Block Obfuscated): {rec:.4f}")
    print(f"False Positive Rate (Block Safe): {fpr:.4f}")
    print(f"Latency: Mean={mean_lat:.2f}ms, P95={p95_lat:.2f}ms")
    
    if rec < 0.6:
        print("❌ CRITICAL: Obfuscated Recall is too low!")
    else:
        print("✅ Success: Pipeline meets robust detection criteria.")

if __name__ == "__main__":
    main()
