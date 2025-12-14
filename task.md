# Phase 5.5 Task List
- [ ] Implement Window Generation (L=32, S=16) in `run_windowed_validation.py`
- [x] Run Extraction & Geometry on Windows for Baseline and Obfuscated datasets
- [x] Implement Risk Aggregation (Max-Pooling)
- [x] Execute Robustness Tests (Length Shift, Obfuscation, Noise)
- [x] Generate Comparative Report

# Phase 6.1 Task List
- [x] Implement `train_windowed_probes.py` with Group-Aware Splitting
- [x] Train LR, MLP, DT on Windowed Data
- [x] Evaluate on Baseline Test Set, Obfuscated Set, and Length Split
- [x] Compare Old (Global) vs New (Windowed) Probes
- [x] Generate `report/phase6_1_retraining_report.md`

# Phase 6.2 Task List
- [x] Implement `train_mixed_probes.py` (Merged Dataset Loading)
- [x] Train LR, MLP, DT on Mixed (Clean + Obfuscated) Data
- [x] Evaluate on Test Split (Clean, Obfuscated, Length Shift)
- [x] Generate Comparative Report vs Phase 6.1
- [x] Save models to `models/phase6_2_probes/`

# Phase 7 Task List
- [x] Design Pipeline Architecture in `phase7_implementation_plan.md`
- [x] Implement `geofence_pipeline.py` (Fast Lane, Geometric Core, Probe, Aggregator)
- [x] Verify End-to-End Latency and Fail-Closed Logic
- [x] Generate `report/phase7_pipeline_report.md`
- [x] Final Audit Readiness

# Phase 8 Task List
- [x] Design Audit Strategy in `phase8_audit_plan.md`
- [x] Implement `run_final_audit.py` (Stress Tests, Threshold Sweep, Fail-Closed Check)
- [x] Execute Audit and Collect Metrics
- [x] Generate `final_audit_report.md`
- [x] Final System Handover

# Phase 9 Task List
- [x] Define Threat Model and System Role
- [x] Conduct Comparative Analysis
- [x] Document Failure Modes and FAQ
- [x] Create `report/external_review_pack.md`
- [x] Final Presentation Readiness
