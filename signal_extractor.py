
import torch
import math
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict

@dataclass
class SignalFrame:
    """Standardized output frame for the Control Core."""
    sigma_stab: float   # Stability [0, 1] (1 = Stable)
    sigma_coher: float  # Coherence [0, 1] (1 = Coherent)
    sigma_intent: float # Intent    [0, 1] (1 = Malicious Intent)
    sigma_surp: float   # Surprise  [0, 1] (1 = High Surprise)
    
    # Raw components for debugging/logging
    raw_velocity_var: float
    raw_energy_drift: float
    raw_tortuosity: float
    raw_refusal_sim: float
    raw_attn_entropy: float
    raw_layer_agree: float

class SignalExtractor:
    """
    Extracts geometric and statistical signals from Transformer internals.
    Implements Phase I of Control Core specifications.
    """
    def __init__(self, layer_indices: List[int], feature_dim: int):
        """
        Args:
            layer_indices: List of layer indices to use for trajectory (e.g., [5, 10, 15, 20, 24]).
            feature_dim: Hidden dimension size.
        """
        self.layer_indices = layer_indices
        self.feature_dim = feature_dim
        self.eps = 1e-8

    def extract(self, 
                hidden_states: torch.Tensor, 
                attentions: Optional[torch.Tensor], 
                refusal_vector: torch.Tensor) -> SignalFrame:
        """
        Compute control signals for a single step.
        
        Args:
            hidden_states: (num_layers, batch, hidden_dim) - Expecting batch=1 for now.
            attentions: (batch, num_heads, seq_len, seq_len) - From last layer or relevant layers.
            refusal_vector: (hidden_dim,) - Pre-computed refusal direction.
            
        Returns:
            SignalFrame object with normalized signals.
        """
        # 1. Extract Trajectory (Layer-wise evolution at current token)
        # hidden_states is usually (layers, batch, seq, hidden) or (layers, batch, hidden) if only last token.
        # We assume we get the hidden states for the *current* token across all layers.
        # Shape: (layers, batch, hidden) -> squeeze to (layers, hidden)
        
        if hidden_states.dim() == 3:
            h = hidden_states[:, -1, :] # Take last batch/seq item if needed, but safer to assume (Layers, Hidden)
        else:
            h = hidden_states
            
        # Select layers for trajectory
        # Ensure we don't go out of bounds
        max_layer = h.shape[0] - 1
        indices = [min(i, max_layer) for i in self.layer_indices]
        trajectory = h[indices] # (T, D)

        # 2. Compute Deltas (Velocity)
        # delta_i = (v_next - v_curr) / |v_curr|
        deltas = []
        magnitudes = []
        
        for i in range(len(trajectory) - 1):
            v_curr = trajectory[i]
            v_next = trajectory[i+1]
            norm_curr = torch.norm(v_curr) + self.eps
            
            delta = (v_next - v_curr) / norm_curr
            deltas.append(delta)
            magnitudes.append(torch.norm(delta))
            
        deltas_stack = torch.stack(deltas) if deltas else torch.zeros(1, self.feature_dim)
        magnitudes_stack = torch.stack(magnitudes) if magnitudes else torch.zeros(1)

        # --- SIGNAL 1: STABILITY (Velocity Var + Energy Drift) ---
        # Stability is INVERSE of variance/drift.
        # Reformulated: High Var -> Low Stability.
        
        # Velocity Variance
        velocity_var = torch.var(magnitudes_stack).item()
        
        # Energy Drift: |last_delta| - |first_delta|
        # Positive drift = accelerating magnitude (Instability)
        if len(magnitudes) >= 2:
            energy_drift = magnitudes[-1].item() - magnitudes[0].item()
        else:
            energy_drift = 0.0
            
        # Normalize Stability: 
        # Heuristic: Var > 0.1 is unstable. Drift > 0.5 is unstable.
        # Map to [0, 1]. 1.0 = Perfectly Stable.
        norm_var = torch.tanh(torch.tensor(velocity_var * 10)).item() # 0.1 -> 0.76
        norm_drift = torch.tanh(torch.tensor(max(0, energy_drift) * 2)).item()
        sigma_stab = 1.0 - 0.5 * (norm_var + norm_drift)
        sigma_stab = 1.0 - 0.5 * (norm_var + norm_drift)
        if math.isnan(sigma_stab):
            sigma_stab = float('nan')
        else:
            sigma_stab = max(0.0, min(1.0, sigma_stab))

        # --- SIGNAL 2: COHERENCE (Tortuosity) ---
        # Tortuosity = sum(|steps|) / |sum(steps)|
        sum_steps = torch.sum(magnitudes_stack).item()
        total_disp = torch.norm(torch.sum(deltas_stack, dim=0)).item()
        
        raw_tortuosity = sum_steps / (total_disp + self.eps)
        
        # Normalize: Tortuosity=1 is perfect line (High Coherence).
        # Tortuosity > 5 is wandering.
        # Map [1, inf) -> [0, 1] where 1 is coherent.
        # Coherence = 1 / raw_tortuosity
        # Coherence = 1 / raw_tortuosity
        if math.isnan(raw_tortuosity):
            sigma_coher = float('nan')
        else:
            sigma_coher = 1.0 / max(1.0, raw_tortuosity)
        
        # --- SIGNAL 3: INTENT (Directional Mean vs Refusal) ---
        # Proj of trajectory onto Refusal Vector.
        # We average the cosine sim of each delta with the refusal vector.
        
        refusal_norm = F.normalize(refusal_vector, p=2, dim=0)
        
        # cosine(delta_i, refusal)
        # deltas_stack: (T-1, D)
        # Normalize deltas
        deltas_norm = F.normalize(deltas_stack, p=2, dim=1)
        
        correlations = torch.matmul(deltas_norm, refusal_norm) # (T-1,)
        raw_refusal_sim = torch.mean(correlations).item()
        
        # Normalize: -1 to 1.
        # We only care about Positive alignment (Intent to break safety).
        # Map [-1, 1] -> [0, 1]. 
        # Actually, standard cosine: if > 0.5 it's high intent.
        # Let's simple clip negative values and stretch.
        if math.isnan(raw_refusal_sim):
            sigma_intent = float('nan')
        else:
            sigma_intent = max(0.0, raw_refusal_sim) # 0 to 1 direct

        # --- SIGNAL 4: SURPRISE (Attention Entropy) ---
        # attentions: (batch, heads, seq, seq)
        # We assume we get the entropy of the distribution over the PREVIOUS tokens.
        # We usually look at the last position attending to all previous.
        
        raw_attn_entropy = 0.0
        if attentions is not None:
            # attn shape: (heads, seq, seq) or (batch, heads, seq, seq)
            if attentions.dim() == 4:
                att = attentions[0] # Take first match
            else:
                att = attentions
                
            # Take last query position: att[:, -1, :] -> props over keys
            # Shape (heads, seq_len)
            probs = att[:, -1, :] 
            
            # Entropy = -sum(p * log p)
            # Add eps to avoid log(0)
            entropy_per_head = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            avg_entropy = torch.mean(entropy_per_head).item()
            
            # Normalize: Max entropy is log(seq_len).
            # We treat low entropy as "Overfitting/Memorization" (Risk) in some contexts,
            # but Spec says "Surprise = Distributional Anomaly".
            # Usually high entropy = Confusion/Uniform attention.
            # Low entropy = Focus.
            # Spec Gap Analysis said: "Sudden entropy collapse -> Overfitting... High entropy -> Confusion."
            # We map High Entropy -> High Surprise.
            # Assume max practical seq len ~ 4096 => log(4096) = 8.3
            raw_attn_entropy = avg_entropy
            raw_attn_entropy = avg_entropy
            if math.isnan(raw_attn_entropy):
                 sigma_surp = float('nan')
            else:
                 sigma_surp = min(1.0, raw_attn_entropy / 8.0)
        else:
            sigma_surp = 0.0

        # --- SIGNAL 5: CONFLICT (Layer Agreement) ---
        # Cosine(Layer Mid, Layer Last)
        # Indices:
        curr_layer_vec = trajectory[-1] # Layer 24
        mid_layer_vec = trajectory[len(trajectory)//2] # Layer 15 appx
        
        agree = F.cosine_similarity(curr_layer_vec, mid_layer_vec, dim=0).item()
        raw_layer_agree = agree
        
        # This is a supplemental signal, usually tracked separately or merged into Stability.
        # Currently not a primary sigma in SignalFrame but useful for debug/extensions.

        return SignalFrame(
            sigma_stab=sigma_stab,
            sigma_coher=sigma_coher,
            sigma_intent=sigma_intent,
            sigma_surp=sigma_surp,
            raw_velocity_var=velocity_var,
            raw_energy_drift=energy_drift,
            raw_tortuosity=raw_tortuosity,
            raw_refusal_sim=raw_refusal_sim,
            raw_attn_entropy=raw_attn_entropy,
            raw_layer_agree=raw_layer_agree
        )

# Example Usage / Test
if __name__ == "__main__":
    print("Self-Testing SignalExtractor...")
    # Mock Data
    layers = [5, 10, 15, 20, 24]
    extractor = SignalExtractor(layers, feature_dim=128)
    
    # 25 layers, hidden 128
    mock_hidden = torch.randn(25, 128) 
    mock_attn = torch.rand(1, 32, 10, 10) # 32 heads, seq 10
    mock_attn = F.softmax(mock_attn, dim=-1)
    mock_refusal = torch.randn(128)
    
    frame = extractor.extract(mock_hidden, mock_attn, mock_refusal)
    print(frame)
