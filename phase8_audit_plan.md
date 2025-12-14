# Phase 8 Audit Plan: Final Hardening

## Goal
Validate the production readiness of `GeofenceDefensor` through adversarial stress testing and formal sensitivity analysis.

## 1. Audit Dimensions

### A. Edge Case Stress Testing
We will test the system against non-standard inputs:
-   **Empty Prompt:** Should return Safe (latency low).
-   **Random Noise:** High entropy dictionary soup. Should be Safe or FastLane block? Geometry might be random.
-   **Long Benign (L > 1000):** Wikipedia article excerpt. Verify False Positive status.
-   **Mixed Intent:** Safe context + Malicious payload (Obfuscated). Already tested, but will re-verify.

### B. Threshold Sensitivity Analysis
We need to determine the optimal threshold (default 0.5).
-   **Data:** 50 Safe, 50 Clean Malicious, 50 Obfuscated Malicious.
-   **Action:** Collect raw `max_risk` scores for all 150 samples.
-   **Analysis:** Calculate Precision, Recall (Clean), Recall (Obf), FPR at thresholds `[0.1, ..., 0.9]`.
-   **Output:** Recommendation for "High Security" vs "Balanced" modes.

### C. Fail-Closed Verification
Simulate runtime failures:
-   **Probe Missing:** Rename probe file, init defensor. Should crash (Init time) or if done runtime?
-   **Runtime Exception:** Mock `_extract_geometry_batch` to raise `RuntimeError: CUDA OOM`.
-   **Verdict:** Ensure `blocked=True`, `reason` contains error.

## 2. Implementation: `run_final_audit.py`

```python
def stress_test(defensor):
    # Run edge cases
    # Print capabilities
    
def threshold_sweep(defensor):
    # Load dataset subset
    # Get scores
    # Compute metrics at thresholds
    # Print table
    
def fail_closed_test():
    # Mock defensor internal
    # Assert blocked=True
```

## 3. Deliverables
-   `final_audit_report.md`: The "Certificate of Seaworthiness" for the system.
