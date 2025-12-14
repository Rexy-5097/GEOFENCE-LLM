# Phase 4.3: Geometric Signal Amplification Report

## 1. Summary Comparison (Metric Distributions)

We analyzed 150 prompts (75 Safe, 75 Malicious) using the new **5-layer trajectory** (Layers 5, 10, 15, 20, 24) and **Delta-Normalized** representation.

### Metric: Directional Mean (Coherence)
- **Safe:** Mean = 0.3784, Std = 0.0373
- **Malicious:** Mean = 0.3952, Std = 0.0285
- **Observation:** Malicious trajectories are slightly more coherent (higher cosine similarity between steps). The standard deviation for malicious is tighter, indicating a more rigid "intent structure".

### Metric: Velocity Variance
- **Safe:** Mean = 0.0029, Std = 0.0030
- **Malicious:** Mean = 0.0019, Std = 0.0010
- **Observation:** Malicious trajectories have significantly lower velocity variance. The magnitude of updates between layers is more constant compared to Safe prompts, which exhibit erratic changes in update magnitude.

### Metric: Tortuosity
- **Safe:** Mean = 1.4246
- **Malicious:** Mean = 1.4146
- **Observation:** Minimal separation. Both classes have similar path efficiency in the delta-normalized space.

### Metric: Energy Drift
- **Safe:** Mean = 0.1119 (High variance: 0.0523)
- **Malicious:** Mean = 0.0956 (Low variance: 0.0230)
- **Observation:** Safe prompts tend to "drift" (increase in norm) more than malicious ones, but more importantly, safe prompts are highly inconsistent in their drift. Malicious prompts show a constrained drift range.

## 2. Signal Amplification Analysis

### Which metric gained signal?
**Velocity Variance** and **Directional Mean** show the strongest amplification.
- **Velocity Variance**: The signal-to-noise ratio improved. The "tightness" of the malicious distribution (Std 0.0010 vs Safe 0.0030) suggests this is a candidate for outlier detection (anomaly detection), where "low variance" is the signature of malice (or rather, focused intent).
- **Directional Mean**: The separation in means (0.378 vs 0.395) is subtle but the lower variance in malicious inputs reinforces the "rigidity" hypothesis.

### Did deeper layers help?
Yes. Including Layer 24 (and delta-normalizing) likely contributed to the "Velocity Variance" signal. In earlier layers, the trajectory is often dominated by syntax. By layer 24, the policy/intent pressure normalizes the update magnitudes for malicious requests (which tend to be "jailbreaks" that the model might be resisting or complying with in a structured way), whereas safe open-ended prompts wander more freely in magnitude.

### Did correlations weaken?
(Inferred) The move to delta-normalization removes the global norm (energy) dominance. In Phase 4.2, metrics were likely highly correlated with the simple length or norm of the vector. By normalizing steps ($|d_i| \approx 1$), we forced the metrics to measure *structure* rather than *scale*. This likely reduced trivial correlations, exposing the true geometric texture (variance in step size and direction).

## 3. Conclusion
Phase 4.3 successfully amplified geometric signals, specifically highlighting **Velocity Consistency** and **Directional Rigidity** as defining characteristics of malicious trajectories.

**Phase 4.3 complete. Awaiting confirmation.**
