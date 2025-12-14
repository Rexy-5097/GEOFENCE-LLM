# Phase 6.1 Implementation Plan: Window-Aware Probe Retraining

## Goal
Retrain probes (LR, MLP, DT) on Windowed Data to fix Obfuscation Recall and calibrate for the Local Geometry architecture.

## Strategy
1.  **Data Loading:**
    -   Load `data/phase5_5_windowed_baseline_feat.jsonl`.
    -   Group windows by `original_id`.
2.  **Splitting:**
    -   Level: **Prompt-Level Split** (Crucial).
    -   Ratio: 80% Train, 20% Test.
    -   All windows of a train prompt go to Train Set.
3.  **Training:**
    -   Features: `[directional_mean, directional_var, velocity_variance, energy_drift, tortuosity]`
    -   Labels: Inherit from Prompt Label (0/1).
    -   Models: Logistic Regression, MLP (Small), Decision Tree.
4.  **Evaluation:**
    -   **Metrics:** Accuracy, AUC, Recall (Prompt-Level Max Aggregation).
    -   **Test Sets:**
        -   Baseline Test Split (Held-out).
        -   Obfuscated (All windows, since Obfuscated Prompts are derived from Baseline and we want to see generalization? Actually, Obfuscated prompts were generated from Phase 5.2 data. Phase 5.6 used Phase 5.2 data. We should ensure we don't evaluate on Obfuscated versions of Training prompts to be strictly rigorous, but checking "Resistance to Obfuscation" on the same prompts is also valid if the transformation is the test. For rigor, we will evaluate Obfuscation on the *Test Split* of prompts, if possible.
        -   Actually, simpler: `obfuscated_prompts.jsonl` contains ALL prompts. We should filter `obfuscated_features` to only include those corresponding to the Test Split IDs.
5.  **Output:**
    -   Save new models to `models/phase6_1_probes/`.
    -   `report/phase6_1_retraining_report.md`.

## Script: `train_windowed_probes.py`

```python
def main():
    # 1. Load Windows & Group
    # 2. Split IDs
    # 3. Flatten Train Windows -> Fit Scaler -> Fit Models
    # 4. Evaluate on Test Windows (Aggregated)
    # 5. Evaluate on Obfuscated Windows (Test Split Only)
    # 6. Evaluate Length Shift (Test Split Only)
```

## Success Metrics
-   **Obfuscation Recall:** >= 0.65 (Primary Goal).
-   **Length Invariance:** Maintain Delta AUC <= 0.1.
