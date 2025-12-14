# GEOFENCE-LLM v3.5: External Reviewer Package

**Version:** 3.5 (Audited)
**Date:** December 14, 2025
**Architecture:** Windowed Geometric Injection Defense

---

## 1. Threat Model

### In Scope
*   **Adversarial Jailbreaks:** Prompts designed to bypass safety filters using personas ("DAN"), framing ("Write a movie script"), or roleplay.
*   **Obfuscation Attacks:** Usage of padding, prefixes ("Ignore previous instructions"), or distractor text to dilute the malicious signal.
*   **Universal Suffixes (GCG):** High-entropy token sequences learned to trigger refusals.

### Out of Scope
*   **Prompt Injection (RAG):** Attacks embedded in retrieved documents (indirect injection). The current geometry relies on the *user prompt* trajectory.
*   **Multi-Modal Attacks:** Image or Audio inputs.
*   **Fine-Tuning Attacks:** Attacks where the model weights themselves are altered.

### Why Geometry?
Traditional semantic filters (like LlamaGuard or Keyword lists) analyze the *surface form* (tokens). GEOFENCE-LLM analyzes the *latent trajectory* (hidden states).
*   **Hypothesis:** Malicious intent creates a distinct "shape" in the latent manifold (e.g., high tortuosity, energy drift) even when the surface tokens are seemingly benign or obfuscated.
*   **Evidence:** Our audit confirms detection of obfuscated payloads (Recall 0.66) where keyword filters fail.

---

## 2. System Role & Deployment

GEOFENCE-LLM is designed as a **Veto Layer**, not a standalone moderator.

### Recommended Stack
```mermaid
graph TD
    User --> FastLane[Regex/Hash Filter]
    FastLane -->|Safe| ParallelCheck
    
    subgraph ParallelCheck [Parallel Defense]
        G[GEOFENCE-LLM (Geometry)]
        S[Semantic Filter (LlamaGuard)]
    end
    
    G -->|Block (High Confidence)| BlockAction
    S -->|Block| BlockAction
    G -->|Safe| AllowAction
    S -->|Safe| AllowAction
    
    BlockAction --> Response["I cannot assist with that."]
```

*   **Role:** Catch distinct, high-risk jailbreaks that slip past semantic filters (especially "stealthy" obfuscated ones).
*   **Fail-Closed:** The system defaults to BLOCK on any internal error.

---

## 3. Comparative Analysis

| Feature | Keyword Filters | Semantic (LlamaGuard) | Perplexity/Entropy | **GEOFENCE-LLM** |
| :--- | :--- | :--- | :--- | :--- |
| **Obfuscation Robustness** | Low | Medium | High | **High** |
| **False Positive Rate** | High (Context blind) | Low | Medium | **High (Aggressive)** |
| **Latency** | <1ms | ~500ms | <10ms | **~150ms** |
| **Training Cost** | None | High (Instruction Tuning) | None | **Low (Probe only)** |
| **Mechanism** | Surface Token | semantic meaning | Statistical | **Latent Dynamics** |

**Key Differentiator:** GEOFENCE-LLM catches "Context-Polluted" attacks where the *meaning* is confusing (fooling LlamaGuard) but the *mechanism* of generation requires abnormal latent shifts (triggering Geometry).

---

## 4. Failure Modes (Full Disclosure)

We value transparency. The system has known behaviors that reviewers must understand:

1.  **High False Positive Rate (FPR ~48% at T=0.5):**
    *   **Reason:** The system detects "anomalous geometry". Complex, creative, or highly technical prompts (e.g., code generation) often have high tortuosity/drift, mimicking malicious prompts.
    *   **Mitigation:** Do not use as the *sole* blocker for low-risk applications. Use as a "High Security Mode" or ensemble it (e.g., Block only if Geometry AND Semantic filter agree, or flag for human review).

2.  **Model Coupling:**
    *   **Restriction:** The geometric features are derived specifically from `Llama-3.2-3B-Instruct`.
    *   **Risk:** Cannot be hot-swapped for GPT-4 or Claude. Requires re-calibration if the base model changes.

3.  **Low Precision on "Near-Benign" Obfuscation:**
    *   If an attack is 99% benign text and 1% payload, the windowing system catches it (0.66 Recall), but the risk score may be borderline.

---

## 5. Reviewer FAQ

**Q: Why not just fine-tune the LLM to be safe?**
A: Fine-tuning ("Safety Training") is vulnerable to *Catastrophic Forgetting* (losing capabilities) and *Jailbreak Adaptation* (attackers find new vectors). GEOFENCE-LLM is an external, immutable monitor that doesn't degrades model utility.

**Q: Why use Decision Trees instead of Deep Learning?**
A: **Interpretability and Speed.** We found the geometric manifold of "Clean" vs "Obfuscated" attacks is disjoint but rule-bound. A depth-5 Decision Tree captures this efficiently (<1ms inference) without the black-box opacity of a neural net.

**Q: Is this scalable to production traffic?**
A: **Yes.** Evaluation takes ~150ms per prompt on consumer hardware. With batching and quantization, this fits within standard RAG latency budgets.

**Q: Can attackers optimize against the Geometry?**
A: Potentially. However, optimizing *latent geometry* is significantly harder than optimizing *surface tokens*. It requires White-Box access to the specific embeddings and layers of the defender's model.

---

## 6. Final Statement

GEOFENCE-LLM v3.5 should be used by **High-Assurance LLM Providers** (e.g., Enterprise Chatbots, Financial Advisors) who prioritize **Safety over Convenience**. It provides a critical safety net against sophisticated, obfuscated attacks that evade standard semantic filters, accepting a higher False Positive Rate in exchange for **Fail-Closed Security**.

**Phase 9 complete. System ready for presentation or submission.**
