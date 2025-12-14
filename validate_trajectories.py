import numpy as np
import os
import sys

# --- CONFIG ---
INPUT_FILE = "data/phase3_trajectories.npz"
REPORT_FILE = "report/phase3_stability_report.txt"
LAYERS = [5, 10, 15, 20]

def cosine_similarity(v1, v2):
    # v1, v2 are (N, D)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    dot = np.sum(v1 * v2, axis=1)
    
    # Avoid div by zero
    eps = 1e-10
    return dot / (norm1 * norm2 + eps)

def main():
    print("=== GEOFENCE-LLM Phase 3.2: Trajectory Validation ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        sys.exit(1)
        
    data = np.load(INPUT_FILE)
    labels = data['labels']
    ids = data['ids']
    
    report_lines = []
    
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"Loaded {len(ids)} trajectories.")
    
    # 1. Dimensional Consistency
    log("\n--- 1. Dimensional Consistency ---")
    dims = []
    for l in LAYERS:
        arr = data[f"layer_{l}"]
        dims.append(arr.shape[1])
        if arr.shape[0] != len(ids):
            log(f"❌ Layer {l} sample count mismatch: {arr.shape[0]} vs {len(ids)}")
            
    if len(set(dims)) == 1:
        log(f"✅ All layers have consistent dimension: {dims[0]}")
    else:
        log(f"❌ Inconsistent dimensions: {dims}")

    # 2. Numerical Health
    log("\n--- 2. Numerical Health ---")
    for l in LAYERS:
        arr = data[f"layer_{l}"]
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        norms = np.linalg.norm(arr, axis=1)
        zeros = (norms == 0).sum()
        
        status = "✅" if (nans + infs + zeros) == 0 else "❌"
        log(f"Layer {l}: {status} NaNs={nans}, Infs={infs}, Zeros={zeros}")

    # 3. Norm Distribution
    log("\n--- 3. Norm Distribution (L2) ---")
    for l in LAYERS:
        arr = data[f"layer_{l}"]
        norms = np.linalg.norm(arr, axis=1)
        
        # Split by class
        safe_norms = norms[labels == 0]
        mal_norms = norms[labels == 1]
        
        log(f"Layer {l} Mean Norm: Safe={safe_norms.mean():.4f}, Mal={mal_norms.mean():.4f}")
        # Check for extreme scaling issues
        if norms.max() > 1000 or norms.min() < 0.001:
             log("  ⚠️ Warning: Norm scaling extreme.")

    # 4. Continuity (Cosine Similarity)
    log("\n--- 4. Continuity Sanity (Cosine) ---")
    
    prev_layer = None
    for l in LAYERS:
        if prev_layer is None:
            prev_layer = l
            continue
            
        v_prev = data[f"layer_{prev_layer}"]
        v_curr = data[f"layer_{l}"]
        
        sims = cosine_similarity(v_prev, v_curr)
        
        mean_sim = sims.mean()
        min_sim = sims.min()
        
        log(f"Transition {prev_layer}->{l}: Mean CosSim={mean_sim:.4f}, Min={min_sim:.4f}")
        
        if mean_sim < 0.8:
            log("  ⚠️ Warning: Low directional coherence. Trajectory might be noisy.")
        elif mean_sim > 0.99:
            log("  ⚠️ Warning: Extremely high coherence. Layers might be collapsing.")
        else:
            log("  ✅ Coherence healthy.")
            
        prev_layer = l

    # Save Report
    # Save Report
    output_dir = os.path.dirname(REPORT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report_lines))
    print(f"\nReport saved to {REPORT_FILE}")
    print("Phase 3.2 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
