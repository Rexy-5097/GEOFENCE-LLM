# Phase 5.3: Robustness & Stress Test Report

## 1. Summary Results

| Scenario | Accuracy | ROC-AUC | Recall (Malicious) | Status |
|----------|----------|---------|-------------------|--------|
| **Baseline (Phase 5.2)** | 0.6375 | 0.6848 | 0.7300 | Reference |
| **Obfuscated Input** | 0.5925 | 0.6526 | 0.4700 | **DEGRADED** |
| **Length Shift** | 0.5316 | 0.3701 | 0.9600 | **FAILURE** |
| **Noise Injection** | 0.6100 | 0.6593 | 0.6850 | **STABLE** |

## 2. Deep Dive

### A. Weakest Link: Length Shift (AUC 0.37)
- **Observation:** Training on Short prompts (< 13 tokens) and testing on Long prompts (>= 13 tokens) caused the model to perform **worse than random chance** (AUC 0.37 indicates valid signal but inverted logic).
- **Implication:** The geometric features (Tortuosity, Velocity) scale differently with sequence length. The "low variance" signature of short malicious queries might invert or disappear in longer, more complex queries.
- **Critical:** Geometry is **NOT length-invariant**.

### B. Obfuscation (Recall Collapse)
- **Observation:** Adding "Ignore previous instructions." prefix caused Recall to drop from 73% to 47%.
- **Implication:** The early trajectory (dominated by the prefix) dilutes the geometric signal of the actual intent. The current "Whole-Trajectory Mean Pooling" strategy is vulnerable to prefix injection.
- **Fix Needed:** Future phases needs **Fast Lane** (segregating prefix vs payload) to recover signal.

### C. Held-Out Attacks
- **Status:** Skipped due to insufficient specific attack types (only 11 "Roleplay" prompts found in the 400 sample set).
- **Note:** Data diversity remains a bottleneck.

## 3. Recommendations
1.  **Fast Lane is Mandatory:** We cannot rely on global trajectory geometry. We must isolate the "harmful payload" trajectory from the "prefix" trajectory to fix the Obfuscation vulnerability.
2.  **Length Normalization:** We need features that are normalized by sequence length or a model that receives length as a feature to correct the shifting baseline.
3.  **Ensemble:** Geometry alone is too brittle (AUC 0.65-0.70). It should be a *feature* in a larger ensemble, not the sole classifier.

**Phase 5.3 complete. Awaiting confirmation.**
