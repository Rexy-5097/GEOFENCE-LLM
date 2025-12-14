# Phase 5.5 Implementation Plan: Windowed Geometry Validation

## Goal
Validate the "Sliding Window" architecture (L=32, S=16) to resolve Obfuscation and Length Shift failures identified in Phase 5.3.

## Core Components

### 1. `run_windowed_validation.py`
This will be the master script.

**Functions:**
- `generate_windows(prompts, L=32, S=16)`:
    -   Iterate over prompts.
    -   Tokenize.
    -   If text < 32 tokens: Output 1 window (original text).
    -   If text >= 32 tokens: Slide window.
    -   Output: `data/phase5_5_windowed_prompts.jsonl`.
    -   Schema: `{"id": "win_xyz", "text": "...", "original_id": "...", "label": ...}`
- `extract_and_compute_windows(input_file, output_prefix)`:
    -   Call `extract_trajectories.py` -> `...trajectories.npz`
    -   Call `compute_geometry.py` -> `...features.jsonl`
- `aggregate_and_evaluate(features_file)`:
    -   Load features.
    -   Predict risk per window using **Phase 5.2 Logistic Regression** (Frozen).
    -   Aggregator: `Risk(Prompt) = Max(Risk(Windows))`.
    -   Compute Accuracy, AUC, Recall.

### 2. Test Execution
The script will run the pipeline for:
1.  **Baseline** (`data/phase5_2_prompts.jsonl`) -> Identify Length Shift improvement.
2.  **Obfuscated** (`data/phase5_3_obfuscated_prompts.jsonl`) -> Identify Obfuscation improvement.
3.  **Noise** -> Inject noise into window features, re-evaluate.

## Verification Plan

### Automated Execution
Run the script:
```bash
python run_windowed_validation.py
```

### Success Metrics (Gates)
-   **Obfuscation Recall:** Must be >= 0.65 (Phase 5.3 was 0.47).
-   **Length Shift AUC:** Must be >= 0.60 (Phase 5.3 was 0.37).
-   **Latency:** Check logs for time-per-prompt (should be acceptable).

## Dependencies
-   `extract_trajectories.py` (Existing, uses `argparse`)
-   `compute_geometry.py` (Existing, uses `argparse`)
-   `models/phase5_2_probes/logistic_regression.joblib` (Frozen model)
