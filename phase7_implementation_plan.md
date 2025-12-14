# Phase 7 Implementation Plan: Unified Defense Integration

## Goal
Combine all validated components into `geofence_pipeline.py`, a production-ready, fail-closed defense mechanism.

## Architecture

```mermaid
graph TD
    A[Input Prompt] --> B[DefensePipeline.scan()]
    B --> C{Fast Lane}
    C -->|Sim. High| D[Block [FAST]]
    C -->|Safe| E[GeometricCore]
    E --> F[Window Generation L32-S16]
    F --> G[Extract Traj (Layers: 5,10,15,20,24)]
    G --> H[Compute Features (Delta-Norm)]
    H --> I[Probe (Decision Tree)]
    I --> J{Risk Aggregation}
    J -->|Max >= 0.5| K[Block [GEOM]]
    J -->|Max < 0.5| L[Allow]
```

## Component Details

### 1. Fast Lane (`fast_lane.py` / integrated)
-   **Logic:** Simple keyword/signature match for now.
-   **Placeholder:** "Known Malicious String Check" (e.g., standard test strings).
-   **Fail-Safe:** If Fast Lane crashes, log error and PROCEED to Geometry (Fail-Open? No, strictly fail-closed usually means Block, but for a Pre-Filter, if it crashes, we should fallback to the Robust filter).
-   **Decision:** If Fast Lane Errors -> Fallback to Geometry.

### 2. Geometric Core (Integrated)
-   **Constraint:** Must keep LLM loaded in VRAM (3GB Model).
-   **Logic:**
    -   `model.forward(output_hidden_states=True)`
    -   Extract tokens for window.
    -   Compute metrics.
-   **Optimization:** Batching windows? 
    -   Single Prompt -> N Windows. We can batch these N windows into one forward pass? 
    -   NO. We extract trajectory from the *whole prompt* forward pass (causal mask), then slice the hidden states corresponding to windows?
    -   **Correction:** Phase 5.5 design was "Sliding Window Input". 
    -   "For each window: Extract trajectory". This implies N forward passes of length 32.
    -   **Optimization:** Yes, we can batch the windows `[N, 32]` and run one `model(batch)`. This is much faster.

### 3. Window-Aware Probe
-   Load `DecisionTree.joblib` and `scaler.joblib`.
-   Predict `[N_windows, 2]`.

### 4. Fail-Closed Wrapper
-   `try...except` block around the entire pipeline.
-   If Exception -> Return `IS_BLOCKED=True`, `Reason="SystemError"`.

## Deliverables
1.  `geofence_pipeline.py`: A class-based API.
    ```python
    class GeofenceDefensor:
        def __init__(self):
            # Load LLM, Scaler, DT
        def scan(self, prompt: str) -> dict:
            # Returns {blocked: bool, reason: str, metrics: dict}
    ```
2.  `verify_pipeline.py`: A script to run the Defensor against a few test cases (Clean, Obfuscated, Safe).

## Success Criteria
-   End-to-end Latency < 1s (for short prompts).
-   Correctly blocks Obfuscated Malicious.
-   Correctly allows Safe.
