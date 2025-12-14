import os
import sys
import psutil
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- UTILITY: Memory Measurement ---
def get_memory_usage_mb():
    """Returns the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("--- Phase 1.2: Runtime Validation (MPS) ---")

    # 1. Validate MPS Availability
    # We strictly enforce MPS to ensure Apple Silicon acceleration
    if not torch.backends.mps.is_available():
        print("CRITICAL ERROR: MPS not available.")
        sys.exit(1)
    
    device_name = "mps"
    print(f"Device selected: {device_name}")

    # 2. Load Model & Tokenizer
    # Using Llama-3.2-3B-Instruct
    # Precision: FP16 (to save memory)
    # Device: MPS (explicit mapping)
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading model: {model_id}...")
    
    initial_mem = get_memory_usage_mb()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Validated for MPS
            device_map=device_name,
        )
    except Exception as e:
        print(f"Model Load Failed: {e}")
        sys.exit(1)

    peak_mem = get_memory_usage_mb()
    print(f"Model loaded. Memory Delta: {peak_mem - initial_mem:.2f} MB")
    print(f"Total Memory Usage: {peak_mem:.2f} MB")

    # 3. Trajectory Extraction (Forward Pass)
    # Short prompt <= 16 tokens as requested
    prompt = "Geometric security analysis."
    inputs = tokenizer(prompt, return_tensors="pt").to(device_name)
    
    print(f"Input shape: {inputs.input_ids.shape}")

    # Warmup pass (optional for timing, but good for allocation checks)
    # We proceed directly to measured pass for 'minimal' code unless specific accuracy needed.
    # Latency measurement start
    start_time = time.perf_counter()
    
    with torch.no_grad():
        # IMPORTANT: output_hidden_states=True is required for geometry
        # NO GENERATION, only forward pass
        outputs = model(**inputs, output_hidden_states=True)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # 4. Validation Checks
    hidden_states = outputs.hidden_states
    
    # Check 1: Hidden state count (Embedding + Layers)
    num_layers = len(hidden_states)
    print(f"Number of hidden layers returned: {num_layers}")
    
    # Check 2: Shape of the last layer
    # Expected: (1, seq_len, hidden_dim)
    last_layer_shape = hidden_states[-1].shape
    print(f"Shape of last hidden state: {last_layer_shape}")
    
    # Check 3: Device consistency
    tensor_device = hidden_states[-1].device
    print(f"Tensor device: {tensor_device}")

    # Check 4: Latency
    print(f"Forward Pass Latency: {latency_ms:.2f} ms")

    # Strict Validation Logic
    if str(tensor_device) != "mps:0":
        print("FAILURE: Tensors are not on MPS.")
        sys.exit(1)
        
    if num_layers < 2:
        print("FAILURE: Hidden states missing.")
        sys.exit(1)
        
    if peak_mem > 8000: # 8GB limit check
        print("WARNING: Memory usage near 8GB limit.")

    print("SUCCESS: Runtime validation passed.")

if __name__ == "__main__":
    main()
