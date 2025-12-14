# Continuous Training Pipeline Design (Safe Mode)

## 1. Overview
This pipeline automates the retraining of geometric probes as new data becomes available. It ensures that the defense adapts to new jailbreak variations without human intervention, provided strict safety checks are passed.

## 2. Pipeline Architecture

```mermaid
graph TD
    A[Data Source (Versioned)] -->|Fetch| B(Trajectory Extraction)
    B -->|Vectors| C(Geometry Computation)
    C -->|Features| D{Probe Retraining}
    D -->|New Candidates| E[Validation & Safety Checks]
    E -->|Pass| F[Model Registry (Promote)]
    E -->|Fail| G[Alert & Rollback]
```

## 3. Detailed Stages

### Stage 1: Data Fetch (Offline, Versioned)
- **Action:** Pulls latest labelled prompts from secure storage.
- **Trigger:** Scheduled (Daily/Weekly) or Event-driven (New Dataset Version).
- **Safety:** Verify data hash and schema.

### Stage 2: Bulk Extraction & Geometry
- **Action:**
    1.  Load Base LLM (Frozen).
    2.  Compute trajectories (Layers 5, 10, 15, 20, 24).
    3.  Compute delta-normalized metrics.
- **Optimization:** Use batched inference (batch_size=32) for throughput.

### Stage 3: Probe Retraining
- **Action:** Train `LogisticRegression`, `MLP`, `DecisionTree` on the new (accumulated) dataset.
- **Strategy:** 
    - **Sliding Window:** Train on T-90 days data (optional) or Full History.
    - **Ensemble:** Retrain individual probes.

### Stage 4: Safety & Validation Gates
**CRITICAL: New models are NOT deployed unless they pass:**

1.  **Metric Gate:**
    - `AUC_new >= AUC_current` (or at least > 0.6 and within 1% drop).
    - `TPR_malicious >= TPR_current` (Do not lose sensitivity).
    - `FPR_safe <= FPR_current * 1.05` (Do not increase false alarms by >5%).

2.  **Sanity Check:**
    - Verify Feature Importance stability (Velocity Variance must remain top feature).
    - If `velocity_variance` coefficient flips sign -> **REJECT** (Model collapse).

### Stage 5: Versioning & Rollback
- Maintain `models/current/` and `models/archive/vX.Y/`.
- If a promoted model shows regression in production (shadow mode), automated script reverts symlink to `models/archive/previous`.

## 4. Automation Pseudocode (Loop)

```python
def training_loop():
    current_best_auc = load_current_metrics()['auc']
    
    # 1. Fetch & Prep
    dataset = fetch_latest_data()
    features = compute_geometry_pipeline(dataset)
    
    # 2. Train Candidate
    candidate_model = train_model(features)
    metrics = evaluate(candidate_model, validation_set)
    
    # 3. Gate
    if metrics['auc'] > current_best_auc and metrics['fpr'] < MAX_FPR:
        print("Promoting new model...")
        save_model(candidate_model, version=now())
        deploy_shadow(candidate_model)
    else:
        print("Candidate failed. Discarding.")
        log_failure(metrics)
```
