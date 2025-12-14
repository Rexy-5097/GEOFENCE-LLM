import os
import sys
import json
# import psutil
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- CONFIG ---
import argparse

# --- CONFIG ---
# Default values
DEFAULT_DATA_FILE = "data/phase5_2_prompts.jsonl"
DEFAULT_OUTPUT_FILE = "data/phase5_2_trajectories.npz"
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
TARGET_LAYERS = [5, 10, 15, 20, 24]
MAX_PROMPTS = 2000 
DEVICE = "mps"

def get_memory_usage_mb():
    # process = psutil.Process(os.getpid())
    # return process.memory_info().rss / 1024 / 1024
    return 0.0

def load_data(limit=None):
    if not os.path.exists(DATA_FILE):
        print(f"❌ Data file missing: {DATA_FILE}")
        sys.exit(1)
    
    records = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_DATA_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    print("=== GEOFENCE-LLM Phase 3.1+: Trajectory Extraction ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # 1. Setup
    print(f"Target Layers: {TARGET_LAYERS}")
    if not torch.backends.mps.is_available():
        print("❌ MPS not available. Aborting.")
        sys.exit(1)
        
    # 2. Load Model
    print(f"Loading {MODEL_ID} on {DEVICE}...")
    mem_start = get_memory_usage_mb()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map=DEVICE,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"Load failed: {e}")
        sys.exit(1)
        
    print(f"Model loaded. Memory: {get_memory_usage_mb():.2f} MB")
    
    # 3. Load Data
    global DATA_FILE # Hack to use the global load_data without refactoring
    DATA_FILE = args.input
    records = load_data(MAX_PROMPTS)
    print(f"Loaded {len(records)} prompts for processing.")
    
    # 4. Processing Loop
    collected_ids = []
    collected_labels = []
    
    # Storage for each layer: List of numpy arrays
    layer_dbs = {l: [] for l in TARGET_LAYERS}
    
    print("Starting extraction (Forward Pass)...")
    latencies = []
    
    for idx, r in enumerate(records):
        prompt = r['text']
        label = r['label']
        uid = r['id']
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Forward Pass
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        
        # Extract & Pool
        # hidden_states is tuple len 29. Index 0 is embeddings. Index 1 is layer 1.
        # So Layer L is at index L (if we consider embedding as 0). 
        # Wait, HuggingFace convention:
        # hidden_states[0] = embeddings
        # hidden_states[1] = output of layer 1
        # ...
        # hidden_states[target] = output of layer target
        # Verify: Model config num_hidden_layers = 28. Tuple len = 29.
        # Yes, index matches layer number exactly.
        
        all_hidden = outputs.hidden_states
        
        for l in TARGET_LAYERS:
            # Shape: (1, seq_len, hidden_dim)
            h_state = all_hidden[l] 
            
            # Mean Pooling: dim=1 (seq_len)
            # vector shape: (1, hidden_dim)
            pooled = torch.mean(h_state, dim=1)
            
            # Move to CPU numpy
            vec_np = pooled.cpu().float().numpy().squeeze(0) # (hidden_dim,)
            layer_dbs[l].append(vec_np)
            
        collected_ids.append(uid)
        collected_labels.append(label)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(records)}...")

    # 5. Save
    print(f"Saving to {args.output}...")
    
    # Construct dict for savez
    save_dict = {
        "ids": np.array(collected_ids),
        "labels": np.array(collected_labels)
    }
    for l in TARGET_LAYERS:
        # Stack to (N, D)
        save_dict[f"layer_{l}"] = np.vstack(layer_dbs[l])
        
    np.savez(args.output, **save_dict)
    
    # 6. Validation Prints
    print("\n--- Validation Report ---")
    print(f"Prompts Processed: {len(records)}")
    if len(records) > 0:
        print(f"Avg Latency: {sum(latencies)/len(latencies):.2f} ms")
    
    # Check dimensions
    # Load back to verify
    # Load back to verify
    data = np.load(args.output)
    print("Saved Archives:")
    for k in data.files:
        print(f"  {k}: {data[k].shape}")
        
    print(f"Peak Memory: {get_memory_usage_mb():.2f} MB")
    
    print("\nPhase 3.1 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
