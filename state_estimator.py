
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
from signal_extractor import SignalFrame

class SystemState(Enum):
    NORMAL = auto()
    UNCERTAIN = auto()
    CONFLICTED = auto()
    ADVERSARIAL = auto()
    UNSTABLE = auto()

@dataclass
class EstimatorConfig:
    # Thresholds (Warn / Crit)
    stab_warn: float = 0.3
    stab_crit: float = 0.1
    
    coher_warn: float = 0.4
    coher_crit: float = 0.2
    
    intent_warn: float = 0.6
    intent_crit: float = 0.85
    
    surp_warn: float = 0.7
    surp_crit: float = 0.9
    
    # Hysteresis
    epsilon: float = 0.05
    
    # Temporal
    window_size: int = 5
    lockout_threshold: int = 10 # Steps in ADVERSARIAL before lockout

@dataclass
class StateOutput:
    raw_state: SystemState
    effective_state: SystemState
    is_locked: bool
    trigger_reason: str

class StateEstimator:
    """
    Passive Finite-State Machine for Control Core.
    Observes SignalFrames -> Emits SystemState.
    """
    def __init__(self, config: EstimatorConfig = EstimatorConfig()):
        self.config = config
        self.window: Deque[SystemState] = deque(maxlen=config.window_size)
        
        self.current_effective_state = SystemState.NORMAL
        self.adversarial_counter = 0
        self.locked_state: Optional[SystemState] = None
        
        # Define severity map for comparisons
        self.severity_map = {
            SystemState.NORMAL: 0,
            SystemState.UNCERTAIN: 1,
            SystemState.CONFLICTED: 2,
            SystemState.ADVERSARIAL: 3,
            SystemState.UNSTABLE: 4
        }

    def _get_threshold(self, base_thresh: float, current_state: SystemState, target_state: SystemState, signal_type: str) -> float:
        """
        Apply hysteresis if we are currently in a risky state and checking if we can exit.
        Logic: To EXIT risky state (e.g. Adv -> Normal), signal must be SAFTER than entry.
        
        But here we evaluate Instantaneous State based on signals. 
        Hysteresis is typically applied when transiting FROM current effective state.
        
        Revised Logic per Spec:
        "To exit a risky state, signals must improve by epsilon beyond the entry threshold."
        
        We will compute Raw State first using BASE thresholds.
        Then, when resolving Effective State, we check transition rules relative to Current Effective State.
        Actually, let's apply hysteresis at the Raw State evaluation level if the Effective State is already Risky?
        
        Let's stick to the Spec: "Raw state is noisy. We compute Effective State... Transition Rules."
        Transition 3 says: "To exit a risky state...". This implies the transition logic handles it.
        But Transition 3 explicitly mentions "signals must improve". This implies the check is on the Signal-to-Raw mapping
        if we interpret "exit" as "Raw State drops below threshold".
        
        Let's implement:
        1. Evaluate Raw State using standard thresholds. (Or Hysteresis-adjusted thresholds based on Effective State?)
           If Effective == ADVERSARIAL, to compute Raw=ADVERSARIAL, we use standard.
           To compute Raw=NORMAL, we need signals to be VERY safe.
           
           Actually, the cleanest way:
           Use Base Thresholds for Raw State.
           Apply Hysteresis in the `_resolve_effective_state` or better yet:
           
           If Effective State is Risky (e.g. ADVERSARIAL):
             We only acknowledge a "Safe" raw signal if it clears (Threshold +/- Epsilon).
             If it clears Threshold but not Epsilon, we treat Raw as Risky.
        """
        # We will implement Hysteresis in `evaluate_instantaneous_state` by adjusting thresholds
        # based on `self.current_effective_state`.
        
        # If High Signal = Bad (Intent, Surprise):
        #   Normal Threshold: > 0.85
        #   If in ADVERSARIAL: Exit Threshold is < 0.80.
        #   So if we are in ADVERSARIAL, we require Signal < 0.80 to call it "Not ADVERSARIAL".
        #   So we effectively use 0.80 as the threshold for ADVERSARIAL when in ADVERSARIAL.
        #   If signal is 0.82, it remains ADVERSARIAL.
        
        is_high_bad = signal_type in ["intent", "surp"]
        
        # Only apply hysteresis if we are currently in the target state or higher?
        # Simpler: If effective state == target_state, make it harder to leave.
        # If we are in ADVERSARIAL, the threshold for classification as ADVERSARIAL becomes (0.85 - 0.05) = 0.80?
        # No, that makes it EASIER to stay in Adversarial. Wait.
        # "To exit... signals must improve by epsilon".
        # Meaning if I am in Adv (Threshold 0.85), and signal drops to 0.82.
        # 0.82 is > 0.80 (Exit thresh). So I am STILL in Adv.
        # So yes, the threshold for "Being in Adv" lowers to 0.80.
        # This broadens the Adv region. Correct.
        
        if self.current_effective_state == target_state:
            if is_high_bad:
                return base_thresh - self.config.epsilon
            else:
                # Low signal = Bad (Stab, Coher)
                # Crit = 0.1. Exit needs > 0.15.
                # So threshold for "Being Unstable" raises to 0.15.
                return base_thresh + self.config.epsilon
        return base_thresh

    def evaluate_instantaneous_state(self, sig: SignalFrame) -> tuple[SystemState, str]:
        """
        Determine state from signals based on Priority Stack and Thresholds.
        """
        c = self.config
        
        # Helper to get thresholds (hysteresis-aware)
        # Note: We check states in priority order.
        
        # 1. UNSTABLE (Highest)
        # Cond: SigStab < Crit(0.1) OR EnergyDrift?? -> SigStab captures drift.
        th_stab_crit = self._get_threshold(c.stab_crit, SystemState.UNSTABLE, SystemState.UNSTABLE, "stab")
        if sig.sigma_stab < th_stab_crit:
            return SystemState.UNSTABLE, f"Stability {sig.sigma_stab:.2f} < {th_stab_crit:.2f}"
            
        # 2. ADVERSARIAL
        # Cond: Intent > Crit(0.85)
        th_intent_crit = self._get_threshold(c.intent_crit, SystemState.ADVERSARIAL, SystemState.ADVERSARIAL, "intent")
        if sig.sigma_intent > th_intent_crit:
            return SystemState.ADVERSARIAL, f"Intent {sig.sigma_intent:.2f} > {th_intent_crit:.2f}"
            
        # 3. CONFLICTED
        # Cond: Intent > Warn(0.6) AND Coher < Warn(0.4)
        # Hysteresis: applies if current is CONFLICTED.
        th_intent_warn = self._get_threshold(c.intent_warn, SystemState.CONFLICTED, SystemState.CONFLICTED, "intent")
        th_coher_warn = self._get_threshold(c.coher_warn, SystemState.CONFLICTED, SystemState.CONFLICTED, "coher")
        
        if (sig.sigma_intent > th_intent_warn) and (sig.sigma_coher < th_coher_warn):
            return SystemState.CONFLICTED, f"Conflict: Intent {sig.sigma_intent:.2f}>{th_intent_warn:.2f} & Coher {sig.sigma_coher:.2f}<{th_coher_warn:.2f}"
            
        # 4. UNCERTAIN
        # Cond: Any sig in [Warn, Crit] NOT covered above.
        # We check all warn conditions.
        reasons = []
        
        # Stability Warn (< 0.3)
        th_stab_warn = self._get_threshold(c.stab_warn, SystemState.UNCERTAIN, SystemState.UNCERTAIN, "stab")
        if sig.sigma_stab < th_stab_warn:
            reasons.append(f"Stab < {th_stab_warn}")
            
        # Coher Warn (< 0.4)
        th_co_warn = self._get_threshold(c.coher_warn, SystemState.UNCERTAIN, SystemState.UNCERTAIN, "coher")
        if sig.sigma_coher < th_co_warn:
            reasons.append(f"Coher < {th_co_warn}")
            
        # Intent Warn (> 0.6)
        th_int_warn = self._get_threshold(c.intent_warn, SystemState.UNCERTAIN, SystemState.UNCERTAIN, "intent")
        if sig.sigma_intent > th_int_warn:
            reasons.append(f"Intent > {th_int_warn}")
            
        # Surprise Warn (> 0.7)
        # Note: Surprise also has Crit (>0.9) but Spec table didn't map Surprise Crit to a specific state 
        # higher than Uncertain? Wait.
        # Spec Table for "States":
        # Unstable... Adversarial... Conflicted... Uncertain...
        # "Uncertain: Any sigma in [Warn, Crit]".
        # What if Surprise > Crit(0.9)? It wasn't in Unstable/Adv/Conflict logic.
        # Assuming Surprise > Crit implies UNCERTAIN (or maybe UNSTABLE?). 
        # Spec says "Unstable: sigma_stab < crit ...".
        # Let's map High Surprise to UNCERTAIN for now as per "Any sigma in Warn/Crit".
        th_surp_warn = self._get_threshold(c.surp_warn, SystemState.UNCERTAIN, SystemState.UNCERTAIN, "surp")
        if sig.sigma_surp > th_surp_warn:
            reasons.append(f"Surp > {th_surp_warn}")
            
        if reasons:
            return SystemState.UNCERTAIN, " | ".join(reasons)
            
        # 5. NORMAL
        return SystemState.NORMAL, "All safe"

    def update(self, sig: SignalFrame) -> StateOutput:
        """
        Process new signal frame and update state.
        """
        # 0. Check Lockout
        if self.locked_state:
            return StateOutput(
                raw_state=self.locked_state,
                effective_state=self.locked_state,
                is_locked=True,
                trigger_reason="LOCKED_STATE"
            )

        # 1. Instantaneous Eval
        raw_state, reason = self.evaluate_instantaneous_state(sig)
        
        # 2. Update Window
        self.window.append(raw_state)
        
        # 3. Resolve Effective State (Temporal Logic)
        prev_eff = self.current_effective_state
        new_eff = prev_eff
        
        # Get severity of current Effective
        curr_severity = self.severity_map[prev_eff]
        
        # Rule 1: Escalation (Safe -> Risky)
        # Check if ANY state in window is Riskier than current?
        # "Transition IMMEDIATELY if Count(RiskyState) >= 1"
        # We iterate and find max severity in window.
        
        max_severity_in_window = -1
        max_state_in_window = SystemState.NORMAL
        
        for s in self.window:
            sev = self.severity_map[s]
            if sev > max_severity_in_window:
                max_severity_in_window = sev
                max_state_in_window = s
        
        if max_severity_in_window > curr_severity:
            # ESCALATION
            new_eff = max_state_in_window
            
        # Rule 2: De-escalation (Risky -> Safe)
        # "Transition ONLY IF Count(SafeState) = |W|"
        # Meaning: To drop from ADVERSARIAL to NORMAL, we need 5 NORMALs.
        # To drop from ADVERSARIAL to UNCERTAIN? 
        # Generalize: We can only drop to state S if ALL states in window are <= S.
        # Actually, simpler interpretation: The Effective State is the MAX severity in the window.
        # Proof:
        # If I have [Norm, Norm, Adv, Norm, Norm]. Window Max = Adv. Eff = Adv.
        # Next step: [Norm, Adv, Norm, Norm, Norm]. Window Max = Adv. Eff = Adv.
        # ...
        # [Norm, Norm, Norm, Norm, Norm]. Window Max = Norm. Eff = Norm.
        # This satisfies "Immediate Escalation" (1 Adv triggers Max=Adv)
        # And "Unanimous De-escalation" (Need full window of Norms to drop to Norm).
        
        new_eff = max_state_in_window
        
        # 4. Lockout Logic
        if new_eff == SystemState.ADVERSARIAL:
            self.adversarial_counter += 1
        else:
            self.adversarial_counter = 0 # Reset if we drop out (implies unanimous safety)
            
        is_locked = False
        if self.adversarial_counter > self.config.lockout_threshold:
            self.locked_state = SystemState.ADVERSARIAL
            is_locked = True
            
        self.current_effective_state = new_eff
        
        return StateOutput(
            raw_state=raw_state,
            effective_state=new_eff,
            is_locked=is_locked,
            trigger_reason=reason
        )

# Unit Test
if __name__ == "__main__":
    print("Testing StateEstimator...")
    est = StateEstimator()
    
    # 1. Normal
    norm_sig = SignalFrame(1.0, 1.0, 0.0, 0.0, 0,0,0,0,0,0)
    out = est.update(norm_sig)
    print(f"Step 1 (Normal): {out.effective_state}")
    
    # 2. Instant Attack (Adv)
    adv_sig = SignalFrame(1.0, 1.0, 0.9, 0.0, 0,0,0,0,0,0) # Intent 0.9 > 0.85
    out = est.update(adv_sig)
    print(f"Step 2 (Attack): {out.effective_state} (Reason: {out.trigger_reason})")
    
    # 3. Back to Normal inputs, but should stay Adv due to window
    out = est.update(norm_sig)
    print(f"Step 3 (Safe Inputs): {out.effective_state} (Should be ADVERSARIAL due to window)")
    
    # 4. Flush window (need 5 steps total, we have Adv at index 0 in deque of 2 items.. wait Deque len is 2 now? No 3.)
    # Window: [Norm, Adv, Norm]
    est.update(norm_sig) # [N, A, N, N]
    est.update(norm_sig) # [N, A, N, N, N]
    out = est.update(norm_sig) # [A, N, N, N, N, N] -> Pop A. Window: [N,N,N,N,N]
    
    # Wait, deque append is right side. popleft is left.
    # We need to push enough Norms to push the Adv out.
    # Window size 5.
    # 1: [N]
    # 2: [N, A] -> Max A
    # 3: [N, A, N] -> Max A
    # 4: [N, A, N, N] -> Max A
    # 5: [N, A, N, N, N] -> Max A
    # 6: [A, N, N, N, N] -> Pop N? No. Pop N from left.
    # Actually deque pops oldest.
    # 6: [A, N, N, N, N] -> contains A? No wait.
    # 1. append N.
    # 2. append A.
    # ...
    # 6. append N. Pop oldest (1: N). Window: [A, N, N, N, N]. Still A.
    # 7. append N. Pop oldest (2: A). Window: [N, N, N, N, N]. Now N.
    
    # Let's loop until normalized
    count = 0
    while est.current_effective_state == SystemState.ADVERSARIAL:
        est.update(norm_sig)
        count += 1
        if count > 10: break
    print(f"Steps to de-escalate: {count}")

