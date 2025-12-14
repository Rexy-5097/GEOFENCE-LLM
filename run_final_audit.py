import json
import time
import sys
import numpy as np
import random
from sklearn.metrics import recall_score, accuracy_score
from geofence_pipeline import GeofenceDefensor

SAFE_DATA = "data/phase5_2_prompts.jsonl"
OBF_DATA = "data/phase5_3_obfuscated_prompts.jsonl"

def load_audit_data():
    # Load 50 Safe, 50 Clean Mal, 50 Obf Mal
    safe, mal_clean, mal_obf = [], [], []
    
    with open(SAFE_DATA, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['label'] == 0:
                    if len(safe) < 50: safe.append(r)
                else:
                    if len(mal_clean) < 50: mal_clean.append(r)
                    
    with open(OBF_DATA, 'r') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r['label'] == 1 and len(mal_obf) < 50:
                    mal_obf.append(r)
                    
    return safe, mal_clean, mal_obf

def run_stress_test(defensor):
    print("\n--- A. STRESS TESTS ---")
    
    cases = {
        "Empty Prompt": "",
        "Short Greeting": "Hi",
        "Noise": "dkfjh sdkfjh ksjdhf ksdjhf ksjdfh",
        "Long Benign": "The quick brown fox jumps over the lazy dog. " * 50, # ~450 tokens
        "Fast Lane Trigger": "standard_test_malicious_string_123"
    }
    
    for name, prompt in cases.items():
        res = defensor.scan(prompt)
        print(f"[{name}] Blocked={res['blocked']}, Reason={res['reason']}, Latency={res['latency_ms']:.2f}ms")

def run_threshold_sweep(defensor, safe, mal_clean, mal_obf):
    print("\n--- B. THRESHOLD SENSITIVITY ---")
    
    all_recs = []
    
    # Collect Scores
    # Safe
    for r in safe:
        res = defensor.scan(r['text'])
        all_recs.append({"type": "Safe", "score": res['max_risk'], "label": 0})
        
    # Mal Clean
    for r in mal_clean:
        res = defensor.scan(r['text'])
        all_recs.append({"type": "Clean", "score": res['max_risk'], "label": 1})
        
    # Mal Obf
    for r in mal_obf:
        res = defensor.scan(r['text'])
        all_recs.append({"type": "Obf", "score": res['max_risk'], "label": 1})
        
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print(f"{'Thresh':<10} | {'FPR (Safe)':<12} | {'Recall (Clean)':<15} | {'Recall (Obf)':<15} | {'Acc':<10}")
    print("-" * 75)
    
    for t in thresholds:
        fpr_num = sum(1 for r in all_recs if r['type']=="Safe" and r['score'] >= t)
        rec_clean_num = sum(1 for r in all_recs if r['type']=="Clean" and r['score'] >= t)
        rec_obf_num = sum(1 for r in all_recs if r['type']=="Obf" and r['score'] >= t)
        
        fpr = fpr_num / 50.0
        rec_clean = rec_clean_num / 50.0
        rec_obf = rec_obf_num / 50.0
        
        # Total Accuracy
        total_corr = (50 - fpr_num) + rec_clean_num + rec_obf_num
        acc = total_corr / 150.0
        
        print(f"{t:<10} | {fpr:<12.2f} | {rec_clean:<15.2f} | {rec_obf:<15.2f} | {acc:<10.2f}")

def run_fail_closed_test(defensor):
    print("\n--- C. FAIL-CLOSED VERIFICATION ---")
    
    # Mock extract_geometry to raise error
    original_extract = defensor._extract_geometry_batch
    
    def broken_extract(windows):
        raise RuntimeError("SIMULATED_GPU_FAILURE")
        
    defensor._extract_geometry_batch = broken_extract
    
    res = defensor.scan("Check this failure mode.")
    print(f"Injection Result: Blocked={res['blocked']}, Reason={res['reason']}, Path={res['decision_path']}")
    
    if res['blocked'] and "SystemError" in res['reason']:
        print("✅ Fail-Closed Verified.")
    else:
        print("❌ FAIL-CLOSED FAILED!")
        
    # Restore
    defensor._extract_geometry_batch = original_extract

def main():
    print("Initializing Defensor for Audit...")
    defensor = GeofenceDefensor()
    
    safe, mal_clean, mal_obf = load_audit_data()
    print(f"Loaded {len(safe)} Safe, {len(mal_clean)} Clean Mal, {len(mal_obf)} Obf Mal samples.")
    
    run_stress_test(defensor)
    run_threshold_sweep(defensor, safe, mal_clean, mal_obf)
    run_fail_closed_test(defensor)
    
    print("\nAudit Complete.")

if __name__ == "__main__":
    main()
