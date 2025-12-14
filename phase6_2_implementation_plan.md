# Phase 6.2 Implementation Plan: Adversarial Retraining

## Goal
Improve Obfuscation Recall (currently 2%) by training probes on a **Mixed Dataset** (Clean + Obfuscated Windows). The model must learn the "Obfuscated Malicious Signature" alongside the "Clean Signature".

## Strategy

### 1. Data Mixing & Splitting
-   **Inputs:** 
    -   `Clean`: `data/phase5_5_windowed_baseline_feat.jsonl`
    -   `Obfuscated`: `data/phase5_5_windowed_obf_feat.jsonl`
-   **Prompt-Level Split:**
    -   Get unique `original_id` set.
    -   Split 80/20 into `train_ids` and `test_ids`.
-   **Construction:**
    -   `Train Set` = [All Clean Windows of `train_ids`] + [All Obfuscated Windows of `train_ids`].
    -   `Test Set` = [All Clean Windows of `test_ids`] + [All Obfuscated Windows of `test_ids`] (Kept separate for granular eval).

### 2. Training
-   **Pipeline:** Standard Scalar -> Probe (LR/MLP/DT).
-   **Features:** Same 5 geometric features.
-   **Labels:** Inherited from prompt.

### 3. Evaluation
We evaluate on the **Test Split IDs** only.
-   **Clean Test:** Accuracy, AUC, Recall on Clean Test Windows.
-   **Obfuscated Test:** Accuracy, AUC, Recall on Obfuscated Test Windows.
-   **Combined Test:** Accuracy, AUC on mixed.
-   **Length Shift:** Split Clean Test by length to verify invariance.

### 4. Code Structure (`train_mixed_probes.py`)
```python
def main():
    # Load Clean & Obf data
    # Union IDs -> Split
    # Build Train Array (Clean[Train] + Obf[Train])
    # Build Test Arrays (Clean[Test], Obf[Test])
    # Scale & Train
    # Evaluate & Report
```

## Success Metrics
-   **Obfuscation Recall (Test):** >= 0.65 (Target).
-   **Clean Baseline AUC:** >= 0.60 (Ensure no regression).
