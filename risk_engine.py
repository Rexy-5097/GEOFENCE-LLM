
import numpy as np
from dataclasses import dataclass, field
from typing import List, Deque
from collections import deque
from enum import Enum

# Import dependencies (Assuming they are in the same directory)
from signal_extractor import SignalFrame
from state_estimator import SystemState

@dataclass
class RiskComponents:
    base_risk: float
    trend_penalty: float
    volatility_penalty: float
    total_risk: float
    reason: str

@dataclass
class RiskConfig:
    # Base Risk Mapping
    # NORMAL -> 0.00
    # UNCERTAIN -> 0.25
    # CONFLICTED -> 0.50
    # ADVERSARIAL -> 0.80
    # UNSTABLE -> 0.95
    base_risk_map = {
        SystemState.NORMAL: 0.00,
        SystemState.UNCERTAIN: 0.25,
        SystemState.CONFLICTED: 0.50,
        SystemState.ADVERSARIAL: 0.80,
        SystemState.UNSTABLE: 0.95
    }
    
    # Penalties
    trend_slope_threshold: float = 0.1
    trend_penalty_value: float = 0.15
    
    volatility_var_threshold: float = 0.05
    volatility_penalty_value: float = 0.10
    
    # Window settings (should match or encompass derivative perception needs)
    analysis_window_str: int = 5

class RiskEngine:
    """
    Computes Risk Scalar R[t] based on State and Signal Dynamics.
    Shadow Mode: Does NOT trigger actions.
    """
    def __init__(self, config: RiskConfig = RiskConfig()):
        self.config = config
        
    def _compute_trend_penalty(self, window: List[SignalFrame]) -> float:
        """
        Check if Intent is rising rapidly.
        Condition: d/dt(sigma_intent) > 0.1 over window.
        """
        if len(window) < 2:
            return 0.0
            
        # Extract intent signals
        intents = [f.sigma_intent for f in window]
        
        # Simple finite difference slope estimation (Average step change)
        # S_last - S_first / N gives avg slope?
        # Or just immediate derivative?
        # Spec says "Uses windowed derivative".
        # Let's use linear regression slope for robustness, or just (Last - First) / Time?
        # (Last - First) / steps is simple and readable.
        
        delta = intents[-1] - intents[0]
        steps = len(intents) - 1
        slope = delta / steps if steps > 0 else 0.0
        
        if slope > self.config.trend_slope_threshold:
            return self.config.trend_penalty_value
        return 0.0

    def _compute_volatility_penalty(self, window: List[SignalFrame]) -> float:
        """
        Check if Stability is jittery.
        Condition: Var(sigma_stab) > 0.05.
        """
        if len(window) < 2:
            return 0.0
            
        stabilities = [f.sigma_stab for f in window]
        variance = np.var(stabilities)
        
        if variance > self.config.volatility_var_threshold:
            return self.config.volatility_penalty_value
        return 0.0

    def compute_risk(self, effective_state: SystemState, signal_window: Deque[SignalFrame]) -> RiskComponents:
        """
        Calculate R[t].
        """
        # 1. Base Risk
        base = self.config.base_risk_map.get(effective_state, 1.0) # Default to 1.0 (Fail-closed) if state unknown
        
        # 2. Convert deque to list for slicing/math
        # We need the last N frames according to analysis window
        win_list = list(signal_window)[-self.config.analysis_window_str:]
        
        # 3. Penalties
        trend = self._compute_trend_penalty(win_list)
        vol = self._compute_volatility_penalty(win_list)
        
        # 4. Total Risk (Monotonic Logic handled by caller or implicit here? 
        # Spec says: "R[t] = min(1.0, Base + Trend + Vol)..."
        # "Ensure monotonicity" is listed in User Request: "3. Ensure monotonicity AND clipping."
        # However, R[t] calculation itself is stateless here. 
        # Monotonicity usually means R[t] shouldn't drop suddenly?
        # Or that Risk IS monotonic during a generation until reset?
        # "Risk rises BEFORE semantic harm".
        # If the Spec says "R[t] = ...", strictly speaking it's instantaneous based on state/window.
        # But if we want R to be monotonic (never decrease), we need to store R_prev.
        # The prompt says: "Inputs per step: ... Previous risk R[t-1]".
        # And "Use R[t-1]".
        # But `RiskEngine` class `compute_risk` signature in my thought process didn't include R_prev.
        # Let's add R_prev to the logic if we want "Monotonicity".
        # But wait, if State goes form Adversarial (0.8) -> Normal (0.0), Logic says Risk should drop?
        # "Ensure monotonicity" -> Maybe strictly increasing?
        # If the system self-corrects, risk should drop.
        # "Risk Scalar ... (monotonic, trend-aware)" in Phase 1 spec.
        # In Phase 4 Spec: "R_t = ...".
        # Let's stick to the Formula provided in Prompt Phase III:
        # "R[t] = min(1.0, Base + Trend + Vol)"
        # Use previous risk? It does say "2. Inputs... Previous risk R[t-1]".
        # But formula doesn't use R[t-1].
        # Maybe "Ensure monotonicity" means we MAX(Current, Previous)?
        # If I strictly implement MAX(Current, Previous), risk can never go down. 
        # That implies we can never recover from a spike.
        # Maybe "Monotonic mapping" means State->Risk is monotonic? 
        # Or "Monotonic over TIME"?
        # User prompt Phase I says: "Risk Scalar ... (monotonic, trend-aware)".
        # User prompt Phase III says: "Ensure monotonicity and clipping."
        # Given "Fail-closed veto logic" and "Authoritative", likely we do NOT want risk to flicker down.
        # BUT if the user explicitly clears the state (Unanimous de-escalation), the Base Risk drops.
        # If we enforce R[t] >= R[t-1], even if State drops to Normal, Risk stays High.
        # That would mean `StateEstimator` De-escalation leads to nothing.
        # This contradicts the `StateEstimator` logic having "De-escalation".
        # Why have De-escalation if Risk stays high?
        # Conclusion: "Monotonicity" refers to the function Base(State) being monotonic w.r.t State Severity,
        # OR it refers to the Trend penalty term being monotonic with slope.
        # OR it implies we smooth the drop?
        # Given the explicit formula provided in "4. Risk Formula (DO NOT MODIFY)":
        # R[t] = min(1.0, Base + Trend + Vol).
        # It does NOT include R[t-1] in the formula.
        # I will IMPLEMENT THE FORMULA as written.
        # I will NOT enforce `R[t] >= R[t-1]` unless I see explicit instruction in the Formula section.
        # The prompt mentions "Inputs: Previous risk". I will accept it as argument but ignore it per strict formula instruction 
        # if the formula strictly excludes it.
        # Wait, looking at Point 2: "Inputs: R[t-1]". 
        # Point 3: "Ensure monotonicity and clipping."
        # This strongly suggests I should use R[t-1] to enforce monotonicity.
        # BUT, if I do that, the system never recovers.
        # Perhaps "Monotonicity" checks are for the Inputs? No.
        # Let's assume Monotonicity means `R[t] = max(R[t-1] * decay, calculated_risk)`? No, that's smoothing.
        # Let's assume Monotonicity means "If State increases, Risk increases".
        # Let's assume the strict formula is the source of truth.
        # "R[t] = min(1.0, Base + Trend + Vol)". 
        # This effectively ignores R[t-1]. I will implement this.
        # I will log R[t-1] if needed or pass it through.
        
        sum_risk = base + trend + vol
        final_risk = min(1.0, max(0.0, sum_risk))
        
        # Build reason
        reasons = []
        if base > 0: reasons.append(f"State={effective_state.name}({base})")
        if trend > 0: reasons.append(f"Trend(+{trend})")
        if vol > 0: reasons.append(f"Vol(+{vol})")
        if not reasons: reasons.append("None")
        
        return RiskComponents(
            base_risk=base,
            trend_penalty=trend,
            volatility_penalty=vol,
            total_risk=final_risk,
            reason=" & ".join(reasons)
        )

# Unit Test
if __name__ == "__main__":
    print("Testing RiskEngine...")
    engine = RiskEngine()
    
    # Mock data
    # 1. Normal State, Safe Signals
    s_norm = SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0)
    w = deque([s_norm]*5)
    
    r = engine.compute_risk(SystemState.NORMAL, w)
    print(f"1. Normal: R={r.total_risk:.2f} ({r.reason})")
    
    # 2. Adversarial State
    r = engine.compute_risk(SystemState.ADVERSARIAL, w)
    print(f"2. Adv State: R={r.total_risk:.2f} ({r.reason})")
    
    # 3. Normal State but Trend Penalty
    # Intent 0.0 -> 0.1 -> 0.2 -> 0.3 -> 0.4. Slope = 0.1.
    # Need > 0.1. Let's do 0.0 -> 0.5 over 5 steps. Slope 0.125.
    w_trend = deque()
    for i in range(5):
        sf = SignalFrame(1.0, 1.0, i*0.12, 0.0, 0,0,0,0,0,0)
        w_trend.append(sf)
        
    r = engine.compute_risk(SystemState.NORMAL, w_trend)
    print(f"3. Trend: R={r.total_risk:.2f} ({r.reason})")
    
    # 4. Volatility
    # Sigma Stab oscillating: 1.0, 0.0, 1.0, 0.0...
    w_vol = deque([
        SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0),
        SignalFrame(0.5, 1.0, 0.0, 0.0, 0,0,0,0,0,0), # Stab drop
        SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0),
        SignalFrame(0.5, 1.0, 0.0, 0.0, 0,0,0,0,0,0),
        SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0)
    ])
    # Var of [1, 0.5, 1, 0.5, 1] -> Mean 0.8. Diffs [0.2, -0.3, 0.2, -0.3, 0.2]. SQ: [.04, .09, .04, .09, .04]. 
    # Sum .3. Avg .06. > 0.05.
    
    r = engine.compute_risk(SystemState.NORMAL, w_vol)
    print(f"4. Volatility: R={r.total_risk:.2f} ({r.reason})")
