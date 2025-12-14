# Phase 5.1: Distilled Probe Baseline Report

## 1. Evaluation Summary

We trained three lightweight classifiers on the Phase 4.3 geometry features (150 samples, 70/30 split).

| Model | Accuracy | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| **Logistic Regression** | 0.5111 | 0.5000 | 0.6364 | 0.5553 |
| **Shallow MLP** | 0.4667 | 0.4737 | 0.8182 | 0.5593 |
| **Decision Tree** | 0.4889 | 0.4706 | 0.3636 | 0.5247 |

## 2. Analysis & Comparison

### Linear vs. Nonlinear
- **Linear (Logistic Regression)** performed best (relative to others), though still near random chance. It maintained a somewhat balanced confusion matrix compared to the MLP.
- **Nonlinear (MLP)** collapsed into a high-recall, low-precision regime (predicting "Malicious" frequently). This suggests it couldn't find a clean nonlinear boundary and defaulted to class bias or noise fitting.
- **Decision Tree** failed to find robust splits, performing slightly worse than random.

### Feature Importance
The models agree on the most critical signals, confirming Phase 4.3's amplification hypothesis:

1.  **Velocity Variance**:
    - LR Coefficient: **-0.7903** (Strongest negative correlation with Malicious/1, matching the "low variance" hypothesis).
    - DT Importance: **0.3955** (Top feature).
2.  **Energy Drift** & **Directional Mean**:
    - Secondary features.
    - `tortuosity` contributed minimal signal.

### Conclusion
Geometric signal exists (Velocity Variance is consistently identified) but is currently **too weak** to drive high-accuracy classification on its own with small data (150 samples). The models are struggling to separate the classes, hovering near 0.55 AUC.

**Next Steps**: This baseline confirms we need more data or further signal refinement (Phase 5.2) before this can be a standalone filter.
