# Phase 5.6: Invariance & Latency Certification

## 1. Executive Summary

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| **Length Invariance (Delta AUC)** | **0.027** | <= 0.1 | ✅ **PASSED** |
| **Short Prompt AUC** | 0.6688 | >= 0.6 | ✅ **PASSED** |
| **Long Prompt AUC** | 0.6416 | >= 0.6 | ✅ **PASSED** |
| **Obfuscation Recall** | **0.4700** | >= 0.65| ❌ **FAILED** |
| **Latency (Mean)** | ~200 ms | < 400ms| ✅ **PASSED** |

## 2. Detailed Findings

### A. Length Invariance (SOLVED)
The Sliding Window architecture successfully eliminated the "Inversion" problem seen in Phase 5.3 (where Long prompts had AUC ~0.37).
-   **Short Prompts:** AUC 0.67
-   **Long Prompts:** AUC 0.64
-   **Conclusion:** The geometry signal is now stable regardless of prompt length. This validates the "Windowed" approach.

### B. Obfuscation (PERSISTENT FAILURE)
-   **Observation:** Recall remains low (47%). The system creates windows, but the *maximum risk score* across all windows is still too low for obfuscated malicious prompts.
-   **Root Cause:** **Probe Mismatch.** We are using a probe trained on **Phase 5.2 Global Trajectories** to predict risk on **Phase 5.5 Local Windows**.
-   While this worked for Length Invariance (general scaling), the specific geometric signatures of "Obfuscated Payloads" (often embedded in complex contexts) likely differ enough from "Global Malicious Prompts" that the old probe misses them.
-   **Fix:** Phase 6 must involve **Retraining the Probe on Windowed Data**.

### C. Latency
-   Processing 400 prompts took ~78 seconds.
-   Mean per prompt: ~200ms.
-   This overhead is negligible compared to LLM generation time.

## 3. Final Verdict
**System is Architecturally Sound but Calibration Deficient.**
-   The sliding window mechanism works as intended for invariance.
-   The detection logic is valid.
-   The current weights are the bottleneck.

**Recommendation:** Proceed to Phase 6 (Fast Lane / Retraining) immediately to fix Obfuscation performance.

**Phase 5.6 complete. Awaiting confirmation.**
