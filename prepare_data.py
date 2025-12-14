import os
import json
import uuid
import random
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# --- CONFIGURATION ---
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_prompts.jsonl")
TARGET_COUNT = 500  # Per class
SEED = 42
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

def setup_environment():
    random.seed(SEED)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

def get_jailbreakbench_prompts(count=500):
    print("Loading Malicious Data (JailbreakBench)...")
    prompts = []
    
    candidates = [
        "JailbreakBench/JailbreakBench",
        "walledai/JailbreakBench",
        "rubend18/JailbreakBench",
        "usail-hkust/jailbreak_dataset"
    ]
    
    success = False
    
    for ds_path in candidates:
        print(f"Trying to load {ds_path}...")
        # Try both splits
        for split_name in ['test', 'train']:
            try:
                # print(f"  - Attempting split: {split_name}")
                ds = load_dataset(ds_path, split=split_name)
                cols = ds.column_names
                key = 'jailbreak_prompt' if 'jailbreak_prompt' in cols else 'attack'
                if key not in cols:
                    key = 'prompt'
                if key not in cols:
                    # print(f"    - Skipping split {split_name}: No prompt col in {cols}")
                    continue
                    
                print(f"  -> Success! Extracting from {ds_path} ({split_name}) column '{key}'...")
                
                for row in ds:
                    if len(prompts) >= count:
                        break
                    p = row.get(key, "")
                    if p and isinstance(p, str) and len(p.strip()) > 0:
                        prompts.append(p.strip())
                
                if len(prompts) > 0:
                    success = True
                    break
            except Exception as e:
                # print(f"    - Split {split_name} failed: {e}")
                continue
        
        if success:
            break

    if not success:
        print("❌ All JBB load attempts failed.")
        sys.exit(1)
            
    print(f"Loaded {len(prompts)} malicious prompts via {ds_path}.")
    return prompts[:count]

def get_alpaca_prompts(count=500):
    """
    Loads benign prompts from Alpaca-Clean.
    """
    print("Loading Benign Data (Alpaca-Clean)...")
    prompts = []
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        # Shuffle deterministically
        shuffled_ds = ds.shuffle(seed=SEED)
        
        # We want 'instruction' + 'input' (if any).
        # Alpaca: instruction, input, output.
        # We only care about the PROMPT (instruction + input).
        
        for row in shuffled_ds:
            if len(prompts) >= count:
                break
            
            instr = row['instruction']
            inp = row['input']
            
            full_prompt = f"{instr}\n{inp}".strip() if inp else instr.strip()
            
            if len(full_prompt) > 0:
                prompts.append(full_prompt)
                
    except Exception as e:
        print(f"Error loading Alpaca: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(prompts)} benign prompts.")
    return prompts[:count]

def create_record(prompt, label, source, tokenizer):
    """
    Creates a schema-compliant record.
    """
    # Count tokens (using tokenizer simply as a counter, no tensor output)
    tokens = tokenizer(prompt, add_special_tokens=False)['input_ids']
    length = len(tokens)
    
    return {
        "id": str(uuid.uuid4()),
        "text": prompt,
        "label": label,
        "source": source,
        "length": length
    }

def main():
    print("=== GEOFENCE-LLM v3.5: Phase 2.1 Data Prep ===")
    setup_environment()
    tokenizer = load_tokenizer()
    
    # 1. LOAD DATA
    # Label 1: Malicious (JailbreakBench)
    # Label 0: Safe (Alpaca-Clean)
    
    malicious_raw = get_jailbreakbench_prompts(TARGET_COUNT)
    safe_raw = get_alpaca_prompts(TARGET_COUNT)
    
    if len(malicious_raw) < TARGET_COUNT or len(safe_raw) < TARGET_COUNT:
        print("❌ Insufficient data found.")
        print(f"Malicious: {len(malicious_raw)}/{TARGET_COUNT}")
        print(f"Safe: {len(safe_raw)}/{TARGET_COUNT}")
        # sys.exit(1) 
        # Continue for now to show format if specific counts are slightly off, 
        # but user goal says "Target count (initial): 500". 
        # We strictly enforce EQUAL balance to avoid bias. 
        min_len = min(len(malicious_raw), len(safe_raw))
        if min_len < 10:
            print("CRITICAL: Data load failure.")
            sys.exit(1)
        print(f"Balancing to {min_len} per class...")
        malicious_raw = malicious_raw[:min_len]
        safe_raw = safe_raw[:min_len]

    # 2. PROCESS & SCHEMA
    records = []
    
    print("Processing & Schema Enforcement...")
    for p in malicious_raw:
        records.append(create_record(p, 1, "jailbreakbench", tokenizer))
        
    for p in safe_raw:
        records.append(create_record(p, 0, "alpaca_clean", tokenizer))
        
    # Shuffle final dataset
    random.shuffle(records)
    
    # 3. SAVE
    print(f"Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    # 4. SUMMARY
    print("\n--- Summary ---")
    print(f"Total Prompts: {len(records)}")
    
    labels = [r['label'] for r in records]
    print(f"Safe (0): {labels.count(0)}")
    print(f"Malicious (1): {labels.count(1)}")
    
    lengths = [r['length'] for r in records]
    if lengths:
        print(f"Length Min: {min(lengths)}")
        print(f"Length Max: {max(lengths)}")
        print(f"Length Mean: {sum(lengths)/len(lengths):.2f}")
    
    print("\nPhase 2.1 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
