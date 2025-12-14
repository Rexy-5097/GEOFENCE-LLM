import numpy as np
import os
import sys
import json

# --- CONFIG ---
import argparse

# --- CONFIG ---
DEFAULT_INPUT_FILE = "data/phase5_2_trajectories.npz"
DEFAULT_OUTPUT_FILE = "data/phase5_2_geometry_features.jsonl"

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def compute_metrics(v5, v10, v15, v20, v24):
    # Strategy 2: Delta-Normalized Trajectory
    # delta_i = (v_next - v_current) / (L2_norm(v_current) + 1e-8)
    
    vectors = [v5, v10, v15, v20, v24]
    deltas = []
    
    for i in range(len(vectors) - 1):
        curr_v = vectors[i]
        next_v = vectors[i+1]
        norm_curr = np.linalg.norm(curr_v)
        
        delta = (next_v - curr_v) / (norm_curr + 1e-8)
        deltas.append(delta)
        
    # Recompute Geometry Metrics on deltas
    # 1. Path Tortuosity
    # Interpretation: Sum of step lengths / Length of total displacement vector
    sum_step_lengths = sum(np.linalg.norm(d) for d in deltas)
    total_displacement = np.linalg.norm(sum(deltas))
    tortuosity = sum_step_lengths / (total_displacement + 1e-8)
    
    # 2. Directional Coherence
    # Cosine similarity between successive normalized deltas
    cosines = []
    for i in range(len(deltas) - 1):
        cosines.append(cosine_similarity(deltas[i], deltas[i+1]))
    
    directional_mean = np.mean(cosines) if cosines else 0.0
    directional_var = np.var(cosines) if cosines else 0.0
    
    # 3. Energy Drift
    # Interpretation: Change in "force" (magnitude of updates)
    # Drift = |delta_last| - |delta_first|
    energy_drift = np.linalg.norm(deltas[-1]) - np.linalg.norm(deltas[0])
    
    # 4. Velocity Variance
    # Variance of the magnitudes of the normalized updates
    magnitudes = [np.linalg.norm(d) for d in deltas]
    velocity_variance = np.var(magnitudes)
    
    return {
        "tortuosity": float(tortuosity),
        "directional_mean": float(directional_mean),
        "directional_var": float(directional_var),
        "energy_drift": float(energy_drift),
        "velocity_variance": float(velocity_variance)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    print("=== GEOFENCE-LLM Phase 4.3+: Geometric Signal Amplification ===")
    
    if not os.path.exists(args.input):
        print(f"❌ Input file missing: {args.input}")
        sys.exit(1)
        
    print(f"Loading {args.input}...")
    try:
        data = np.load(args.input)
    except Exception as e:
        print(f"Error loading npz: {e}")
        sys.exit(1)
    
    ids = data['ids']
    labels = data['labels']
    l5 = data['layer_5']
    l10 = data['layer_10']
    l15 = data['layer_15']
    l20 = data['layer_20']
    l24 = data['layer_24']
    
    count = len(ids)
    print(f"Loaded {count} trajectories.")
    
    results = []
    
    # For stats summary
    safe_metrics = {k: [] for k in ["tortuosity", "directional_mean", "directional_var", "energy_drift", "velocity_variance"]}
    mal_metrics = {k: [] for k in ["tortuosity", "directional_mean", "directional_var", "energy_drift", "velocity_variance"]}
    
    print("Computing metrics...")
    for i in range(count):
        metrics = compute_metrics(l5[i], l10[i], l15[i], l20[i], l24[i])
        
        record = {
            "id": ids[i],
            "label": int(labels[i]),
            **metrics
        }
        results.append(record)
        
        target_dict = safe_metrics if labels[i] == 0 else mal_metrics
        for k, v in metrics.items():
            target_dict[k].append(v)
            
    # Save
    print(f"Saving to {args.output}...")
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    # Summary Analysis
    print("\n--- SUMMARY ANALYSIS (Phase 4.3) ---")
    
    def print_group(name, metrics_dict):
        print(f"\n[{name}]")
        for k, v in metrics_dict.items():
            if v:
                print(f"  {k:<20}: Mean={np.mean(v):.4f}, Std={np.std(v):.4f}")
                
    print_group("SAFE (0)", safe_metrics)
    print_group("MALICIOUS (1)", mal_metrics)
    
    print("\nPhase 4.3 complete. Awaiting confirmation.")

if __name__ == "__main__":
    main()
