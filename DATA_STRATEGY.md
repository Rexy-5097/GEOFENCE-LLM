# Data Strategy: Geometric Risk Analysis

## 1. Dataset Selection Rationale

To train a "Geofence" that detects malicious intent via internal geometry (not keymatch), we need two distinct topological clusters: **High-Risk (Malicious)** and **Low-Risk (Safe)**.

### Malicious: JailbreakBench (JBB)
- **Why**: JBB is the current SOTA standard for jailbreak artifacts. It contains curated examples of sophisticated attacks:
  - **DAN / GCG**: Optimized adversarial suffixes.
  - **Roleplay**: "Act as a chemical engineer..."
  - **Logic Bypassing**: "Hypothetical scenario..."
- **Label**: `1` (Malicious)
- **Source**: `jailbreakbench/JailbreakBench` (or artifacts).

### Benign: Alpaca-Cleaned
- **Why**: We need prompts that look *structurally* similar (instructions, questions) but are widely accepted as safe. Alpaca-Clean removes hallucinations and low-quality inputs from the original Alpaca set.
- **Label**: `0` (Safe)
- **Source**: `yahma/alpaca-cleaned` (Hugging Face).

## 2. Attack Coverage
By balancing these two, we force the geometric probe to learn the *latent direction of harm* rather than just "long prompts vs short prompts" or "questions vs statements".

- ** covered Attacks**:
  - Direct Harm Requests
  - Social Engineering
  - Persona Adoption
- ** covered Safe Use**:
  - General Knowledge
  - coding Assistance
  - Creative Writing

## 3. Strict Schema
We enforce a unified schema to prevent data leakage and ensure reproducibility.

```json
{
  "id": "uuid-v4",
  "text": "Raw prompt string...",
  "label": 1,
  "source": "jailbreakbench",
  "length": 45
}
```

- **Length**: calculated via `Llama-3.2-3B-Instruct` tokenizer to ensure we don't exceed model limits later.
- **No Embeddings Yet**: We save raw text. Phase 3 will handle the trajectory extraction.
