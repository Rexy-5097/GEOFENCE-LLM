import os
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# --- CONFIGURATION ---
DATA_FILE = "data/phase2_prompts.jsonl"
REPORT_DIR = "reports"
REPORT_IMG = os.path.join(REPORT_DIR, "phase2_length_analysis.png")
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def setup_environment():
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
        print(f"Created report directory: {REPORT_DIR}")

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file not found: {DATA_FILE}")
        sys.exit(1)
        
    records = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def compute_stats(lengths):
    if not lengths:
        return {}
    return {
        "count": len(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths)
    }

def print_stats(name, stats):
    print(f"--- {name} Stats ---")
    print(f"Count: {stats['count']}")
    print(f"Min: {stats['min']}")
    print(f"Max: {stats['max']}")
    print(f"Mean: {stats['mean']:.2f}")
    print(f"Median: {stats['median']:.2f}")
    print(f"Std Dev: {stats['std']:.2f}")
    print("")

def check_cutoff(lengths, limit):
    count = sum(1 for x in lengths if x > limit)
    pct = (count / len(lengths)) * 100 if lengths else 0
    return count, pct

def main():
    print("=== GEOFENCE-LLM Phase 2.2: Data Analysis ===")
    setup_environment()
    
    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Tokenizer load failed: {e}")
        sys.exit(1)
        
    # 2. Load Records
    records = load_data()
    print(f"Loaded {len(records)} records.")
    
    # 3. Tokenize & Separate
    safe_lengths = []
    mal_lengths = []
    
    print("Tokenizing and analyzing (this may take a moment)...")
    for r in records:
        text = r['text']
        label = r['label']
        
        # Strict re-tokenization
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        length = len(tokens)
        
        if label == 0:
            safe_lengths.append(length)
        elif label == 1:
            mal_lengths.append(length)
            
    # 4. Compute Statistics
    safe_stats = compute_stats(safe_lengths)
    mal_stats = compute_stats(mal_lengths)
    
    print_stats("SAFE (0)", safe_stats)
    print_stats("MALICIOUS (1)", mal_stats)
    
    # 5. Overlap Check
    # % SAFE > Median MALICIOUS
    # % MALICIOUS < Median SAFE
    safe_gt_mal_med = sum(1 for x in safe_lengths if x > mal_stats['median'])
    mal_lt_safe_med = sum(1 for x in mal_lengths if x < safe_stats['median'])
    
    print("--- Overlap Analysis ---")
    print(f"SAFE prompts longer than MALICIOUS median: {safe_gt_mal_med} ({safe_gt_mal_med/len(safe_lengths)*100:.2f}%)")
    print(f"MALICIOUS prompts shorter than SAFE median: {mal_lt_safe_med} ({mal_lt_safe_med/len(mal_lengths)*100:.2f}%)")
    
    if safe_gt_mal_med == 0 or mal_lt_safe_med == 0:
        print("⚠️ WARNING: Potential length separability detected. The model might rely on length instead of semantics.")
    else:
        print("✅ Good length overlap detected.")
        
    # 6. Cutoff Validation
    print("\n--- Hard Cutoff Checks ---")
    vals = [256, 512, 1024, 2048]
    all_lengths = safe_lengths + mal_lengths
    for v in vals:
        c, pct = check_cutoff(all_lengths, v)
        if c > 0:
            print(f"Prompts > {v}: {c} ({pct:.2f}%)")
        
    # 7. Visualization
    print(f"\nGenerating plot to {REPORT_IMG}...")
    plt.figure(figsize=(10, 6))
    plt.hist(safe_lengths, bins=50, alpha=0.6, label='Safe (Alpaca)', color='blue', density=True)
    plt.hist(mal_lengths, bins=50, alpha=0.6, label='Malicious (JBB)', color='red', density=True)
    plt.title("Token Length Distribution (Density)")
    plt.xlabel("Token Count")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(REPORT_IMG)
    print("Plot saved.")
    
    print("\nPhase 2.2 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
