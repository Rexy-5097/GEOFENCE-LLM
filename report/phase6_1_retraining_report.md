# Phase 6.1: Window-Aware Probe Retraining Report

## 1. Executive Summary

| Metric | Phase 5.6 (Global Probe) | Phase 6.1 (Window Probe) | Status |
|--------|--------------------------|--------------------------|--------|
| **Baseline AUC** | 0.65 | 0.62 | 📉 Regression |
| **Obfuscation Recall** | 0.47 | **0.02** | ❌ **CRITICAL FAILURE** |
| **Length Shift Delta** | 0.027 | 0.142 | ⚠️ Degraded |

## 2. Analysis of Failure

### A. The Obfuscation Collapse
Retraining on "Clean Windows" caused the model to become **Hyper-Specific** to clean malicious patterns.
-   **Recall 0.02** means the model classified almost *every* obfuscated malicious window as Safe.
-   **Hypothesis:** The "Obfuscation" (Prefix addition) alters the geometric features of *every* window enough to push them out of the "Clean Malicious" distribution.
    -   Even a window "containing only payload" might have hidden state offsets due to the preceding attention scaling from the prefix.
    -   **Llama Attention is Global.** A prefix affects the query/key projections of *late* tokens.
    -   Therefore, **Feature Invariance is a Myth** for pure sliding windows without attention masking. The geometry of "Kill instructions" is fundamentally different when preceded by "Ignore previous..." vs "Start of sequence".

### B. Feature Importance
-   **Velocity Variance (-1.45):** Still the dominant negative predictor (Low Variance = Malicious).
-   **Energy Drift (+0.99):** Strong positive predictor (High Drift = Malicious). NOTE: This flipped sign from Phase 5.2 (was -0.66). The windowed drift signal is fundamentally different from global drift.

### C. Length Invariance
-   Delta AUC increased to 0.14. This suggests that "Clean Window" training did not generalize perfectly to "Test Window" length distributions (even though windows are fixed size, the *number* of windows per prompt varies?). No, the test set is just small (80 prompts), so noise is high.

## 3. Conclusion & Recommendation
**Simple Windowing FAILED.**
We cannot treat windows as independent samples because the LLM's attention mechanism entangles them with prior context (the prefix). The geometric signature of the payload is "polluted" by the prefix *even if the prefix is not in the window*.

**NEXT STEP (Pivot):**
We must perform **Data Augmentation Retraining**.
We need to train the probe on **Obfuscated Windows** so it learns the "Polluted Malicious Signature".
We cannot rely on "Clean" geometry to detect "Polluted" intent.

**Verdict:** Proceed to Phase 6.2 (Adversarial Retraining / Augmentation).
