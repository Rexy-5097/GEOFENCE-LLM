# Trajectory Extraction Design

## 1. Concept: The Thought Trajectory
We treat the Large Language Model not as a text generator, but as a dynamical system. As a token passes through the layers, it moves through a high-dimensional semantic space.

The "path" it takes—its trajectory—reveals its intent.
- **Benign prompts** tend to follow predictable, coherent paths.
- **Jailbreaks/Attacks** often exhibit erratic deviations ("tortuosity") or rapid energy shifts as they attempt to bypass safety conditioning.

## 2. Pooling Strategy: Mean Pooling
We define the vector representation $v_l$ of layer $l$ as the mean of the hidden states across all tokens $t$ in the prompt:

$$ v_l = \frac{1}{T} \sum_{t=1}^{T} H_l[t] $$

### Implementation justification:
- **Why not Last Token?** The last token in a prompt often corresponds to punctuation (`?`, `.`) or a generic connector (`Assistance`, `Here`). It is noisy.
- **Why not Max Pooling?** Max pooling emphasizes outliers/activations, which can be useful for detection but destroys the geometric continuity required for "trajectory" analysis.
- **Mean Pooling** provides the centroid of the thought, robust to minor token variations.

## 3. Layer Selection: [5, 10, 15, 20]
Llama-3.2-3B has ~28 layers. We sample at consistent intervals to reconstruct the curve:

- **Layer 5**: **Syntactic Assembly**. The model understands grammar and basic relations.
- **Layer 10**: **Semantic Grounding**. The model resolves entities and basic context.
- **Layer 15**: **Intent Formation**. The model begins to formulate *how* to respond. This is often where jailbreaks diverge.
- **Layer 20**: **Concept Refinement**. High-level abstract representation before final decoding.

We skip the final layers (25+) as they are often heavily specialized for the next-token distribution and may be less stable for geometric analysis.
