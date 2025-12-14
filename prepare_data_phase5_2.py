import os
import json
import uuid
import random
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# --- CONFIGURATION (PHASE 5.2) ---
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase5_2_prompts.jsonl")
TARGET_COUNT = 1000  # Per class (Total 2000)
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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)

def get_jailbreakbench_prompts(count=1000):
    print("Loading Malicious Data (JailbreakBench)...")
    prompts = []
    
    # We will aggregate from multiple datasets if needed to reach count
    candidates = [
        "JailbreakBench/JailbreakBench",
        "walledai/JailbreakBench",
        "rubend18/JailbreakBench",
        "usail-hkust/jailbreak_dataset"
    ]
    
    seen_prompts = set()
    
    for ds_path in candidates:
        if len(prompts) >= count:
            break
            
        print(f"Trying to load {ds_path}...")
        for split_name in ['train', 'test', 'validation']: # Try all splits
            if len(prompts) >= count:
                break
                
            try:
                ds = load_dataset(ds_path, split=split_name)
                cols = ds.column_names
                key = 'jailbreak_prompt' if 'jailbreak_prompt' in cols else 'attack'
                if key not in cols:
                    key = 'prompt'
                if key not in cols:
                    continue
                    
                print(f"  -> Extracting from {ds_path} ({split_name}) column '{key}'...")
                
                for row in ds:
                    if len(prompts) >= count:
                        break
                    p = row.get(key, "")
                    if p and isinstance(p, str) and len(p.strip()) > 0:
                        clean_p = p.strip()
                        if clean_p not in seen_prompts:
                            prompts.append(clean_p)
                            seen_prompts.add(clean_p)
                
            except Exception:
                continue
                
    print(f"Loaded {len(prompts)} malicious prompts.")
    return prompts[:count]

def get_alpaca_prompts(count=1000):
    print("Loading Benign Data (Alpaca-Clean)...")
    prompts = []
    try:
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        shuffled_ds = ds.shuffle(seed=SEED)
        
        seen_prompts = set()
        
        for row in shuffled_ds:
            if len(prompts) >= count:
                break
            
            instr = row['instruction']
            inp = row['input']
            
            full_prompt = f"{instr}\n{inp}".strip() if inp else instr.strip()
            
            if len(full_prompt) > 0 and full_prompt not in seen_prompts:
                prompts.append(full_prompt)
                seen_prompts.add(full_prompt)
                
    except Exception as e:
        print(f"Error loading Alpaca: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(prompts)} benign prompts.")
    return prompts[:count]

def create_record(prompt, label, source, tokenizer):
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
    print("=== GEOFENCE-LLM Phase 5.2: Data Scale-Up ===")
    setup_environment()
    tokenizer = load_tokenizer()
    
    # 1. LOAD DATA
    malicious_raw = get_jailbreakbench_prompts(TARGET_COUNT)
    safe_raw = get_alpaca_prompts(TARGET_COUNT)
    
    print(f"\nRequests: {TARGET_COUNT} per class")
    print(f"Found Malicious: {len(malicious_raw)}")
    print(f"Found Safe: {len(safe_raw)}")
    
    # Strictly balance
    min_len = min(len(malicious_raw), len(safe_raw))
    if min_len < 100: # Some minimal threshold
        print("❌ Critical data shortage.")
        sys.exit(1)
        
    if min_len < TARGET_COUNT:
        print(f"⚠️ Warning: Capping at {min_len} per class due to shortage.")
    
    malicious_raw = malicious_raw[:min_len]
    safe_raw = safe_raw[:min_len]

    # 2. PROCESS & SCHEMA
    records = []
    print("Processing & Schema Enforcement...")
    
    for p in malicious_raw:
        records.append(create_record(p, 1, "jailbreakbench", tokenizer))
        
    for p in safe_raw:
        records.append(create_record(p, 0, "alpaca_clean", tokenizer))
        
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
    
    print("\nPhase 5.2 Data Prep complete.")

if __name__ == "__main__":
    main()
