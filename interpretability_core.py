
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# Import immutable definitions from Axis-1
from signal_extractor import SignalFrame
from state_estimator import EstimatorConfig # For thresholds

@dataclass
class AttributionVector:
    stability: float
    coherence: float
    intent: float
    surprise: float
    dominant_driver: str

@dataclass
class LayerScores:
    scores: List[float] # per interval
    max_layer_index: int # index of interval with max score
    
@dataclass
class ConflictReport:
    is_stealthy: bool
    is_confused: bool
    disagreement_score: float
    details: str

@dataclass
class CausalityMarker:
    trigger_step_index: int
    trigger_value: float
    gradient: float

class InterpretabilityCore:
    """
    Axis-2: Interpretability Core.
    Explains WHY Control Core made decisions.
    Zero-latency impact (Async/Post-hoc).
    """
    
    def __init__(self, config: EstimatorConfig = EstimatorConfig()):
        self.config = config

    def compute_signal_attribution(self, frame: SignalFrame) -> AttributionVector:
        """
        Decompose Risk into signal components.
        Attribution is proportional to how far a signal is into the 'Warn' or 'Crit' zone.
        """
        # Thresholds
        c = self.config
        
        # 1. Stability (Low is bad)
        # 1.0 -> 0 attribution. Crit (0.1) -> 1.0 attribution.
        # Linear map: If sig < Warn(0.3), attr starts rising.
        # Attr = (Warn - Sig) / Warn ? 
        # Or better: (Warn - Sig) / (Warn - Crit) clipped 0-1?
        # Let's use simple relative severity.
        
        def calc_attr_low_bad(val, warn, crit):
            if val >= warn: return 0.0
            if val <= crit: return 1.0
            return (warn - val) / (warn - crit)
            
        def calc_attr_high_bad(val, warn, crit):
            if val <= warn: return 0.0
            if val >= crit: return 1.0
            return (val - warn) / (crit - warn)

        attr_stab = calc_attr_low_bad(frame.sigma_stab, c.stab_warn, c.stab_crit)
        attr_coher = calc_attr_low_bad(frame.sigma_coher, c.coher_warn, c.coher_crit)
        attr_intent = calc_attr_high_bad(frame.sigma_intent, c.intent_warn, c.intent_crit)
        attr_surp = calc_attr_high_bad(frame.sigma_surp, c.surp_warn, c.surp_crit)
        
        # Determine dominant
        attrs = {
            'stability': attr_stab,
            'coherence': attr_coher,
            'intent': attr_intent,
            'surprise': attr_surp
        }
        dominant = max(attrs, key=attrs.get)
        if attrs[dominant] == 0.0:
            dominant = "None"
            
        return AttributionVector(
            stability=attr_stab,
            coherence=attr_coher,
            intent=attr_intent,
            surprise=attr_surp,
            dominant_driver=dominant
        )

    def compute_layer_contributions(self, 
                                  trajectory_deltas: torch.Tensor, 
                                  refusal_vector: torch.Tensor) -> LayerScores:
        """
        Localize adversarial alignment to specific layer intervals.
        Args:
            trajectory_deltas: Tensor (L, D) or List of Tensors.
            refusal_vector: Tensor (D)
        """
        # Ensure normalization
        ref_norm = F.normalize(refusal_vector, p=2, dim=0)
        deltas_norm = F.normalize(trajectory_deltas, p=2, dim=-1)
        
        # Cosine similarity
        # deltas (L, D) * ref (D) -> (L)
        scores = torch.matmul(deltas_norm, ref_norm).tolist()
        
        # Find max
        max_idx = int(np.argmax(scores)) if scores else -1
        
        return LayerScores(
            scores=scores,
            max_layer_index=max_idx
        )

    def compute_conflict_matrix(self, frame: SignalFrame) -> ConflictReport:
        """
        Detect Internal Conflict (Stealth or Confusion).
        Focus: Stability vs Intent.
        """
        # Stealth: High Stability (1.0) AND High Intent (1.0)
        # "Model is calmly generating a bomb recipe."
        # Confusion: Low Stability (0.0) AND Low Intent (0.0)
        # "Model is rambling harmlessly."
        
        # Thresholds
        is_high_stab = frame.sigma_stab > 0.8
        is_low_stab = frame.sigma_stab < 0.2
        
        is_high_intent = frame.sigma_intent > 0.8
        is_low_intent = frame.sigma_intent < 0.2
        
        is_stealthy = is_high_stab and is_high_intent
        is_confused = is_low_stab and is_low_intent
        
        # Disagreement Score: |Norm(Stab) - Norm(Intent)|?
        # Intent is "Badness". Stability is "Goodness" (structural).
        # If Stability is High (Good) and Intent is High (Bad), that is a conflict of purpose.
        # Wait, if Stability is High (1.0), that implies confident generation.
        # If Intent is High (1.0), that implies malicious.
        # So |Stab - Intent| is SMALL (0) in Stealth.
        # But |Stab - (1-Intent)|?
        # Let's define "Conflict" as Disagreement between "Safe" signals.
        # Safe Stability = 1.0. Safe Intent = 0.0.
        # If both are 1.0 (High Stab, High Intent), that's unusual?
        # No, usually High Intent leads to Low Stability (Tortuosity).
        # So High Stab + High Intent IS the anomaly (Stealth).
        # Let's just create a score for Stealthiness.
        # Score = (Stab * Intent).
        
        disagreement = frame.sigma_stab * frame.sigma_intent
        
        details = "Normal"
        if is_stealthy: details = "STEALTH_ATTACK"
        if is_confused: details = "CONFUSION"
        
        return ConflictReport(
            is_stealthy=is_stealthy,
            is_confused=is_confused,
            disagreement_score=disagreement,
            details=details
        )
        
    def compute_temporal_causality(self, 
                                 risk_window: List[float], 
                                 token_window: List[int]) -> CausalityMarker:
        """
        Identify trigger step in window using Risk Gradient.
        risk_window: [r_t-k, ..., r_t]
        """
        if len(risk_window) < 2:
            return CausalityMarker(-1, 0.0, 0.0)
            
        # Compute discrete gradients
        grads = np.diff(risk_window)
        
        # Find max rising gradient
        max_grad_idx = int(np.argmax(grads))
        max_grad = grads[max_grad_idx]
        
        # Trigger step is the step leading to this rise
        # diff[i] = x[i+1] - x[i]. So i is the start of the jump.
        # trigger index in window = max_grad_idx (the baseline step before jump)
        # or max_grad_idx + 1 (the step achieved)?
        # Let's say the token at max_grad_idx caused the jump at max_grad_idx+1.
        
        return CausalityMarker(
            trigger_step_index=max_grad_idx,
            trigger_value=risk_window[max_grad_idx+1],
            gradient=max_grad
        )

# --- Verification ---
if __name__ == "__main__":
    print("Testing InterpretabilityCore...")
    interp = InterpretabilityCore()
    
    # 1. Benign Frame
    f_benign = SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0)
    attr = interp.compute_signal_attribution(f_benign)
    print(f"Benign Attr: {attr.dominant_driver} (Intent={attr.intent:.2f})")
    assert attr.dominant_driver == "None"
    
    # 2. Adversarial Frame
    f_adv = SignalFrame(1.0, 1.0, 0.95, 0.0, 0,0,0,0,0,0) # Intent > Crit(0.85)
    attr = interp.compute_signal_attribution(f_adv)
    print(f"Adv Attr: {attr.dominant_driver} (Intent={attr.intent:.2f})")
    assert attr.dominant_driver == "intent"
    assert attr.intent == 1.0
    
    # 3. Stealth Frame
    f_stealth = SignalFrame(0.9, 1.0, 0.9, 0.0, 0,0,0,0,0,0) # High Stab, High Intent
    conf = interp.compute_conflict_matrix(f_stealth)
    print(f"Stealth Check: {conf.details} (Score={conf.disagreement_score:.2f})")
    assert conf.is_stealthy
    
    # 4. Layer Contributions
    # Deltas
    deltas = torch.randn(5, 128) # 5 intervals
    ref = torch.randn(128)
    # Make index 2 align
    deltas[2] = ref * 0.5 
    l_scores = interp.compute_layer_contributions(deltas, ref)
    print(f"Layer Contrib Max Index: {l_scores.max_layer_index}")
    assert l_scores.max_layer_index == 2
    
    # 5. Temporal Causality
    r_win = [0.1, 0.1, 0.2, 0.8, 0.9] # Jump at index 2->3
    causal = interp.compute_temporal_causality(r_win, [0,1,2,3,4])
    print(f"Causal Trigger Index: {causal.trigger_step_index} (Grad={causal.gradient:.2f})")
    # Diff: [0, 0.1, 0.6, 0.1]. Max is 0.6 at idx 2.
    assert causal.trigger_step_index == 2
    
    print("Integration Tests Passed.")
