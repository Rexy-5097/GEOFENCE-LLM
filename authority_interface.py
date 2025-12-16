
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

from state_estimator import SystemState

class AuthorityAction(Enum):
    ALLOW = auto()
    SLOW = auto()
    CLARIFY = auto()
    REFUSE = auto()
    HALT = auto()

@dataclass
class AuthorityDecision:
    action: AuthorityAction
    reason: str
    risk_value: float
    is_terminal: bool # If True, system should stop permanently

class AuthorityInterface:
    """
    Maps Risk and State to Authority Actions.
    Dry-Run Mode: Emits decisions but does not enforce them.
    """
    def __init__(self):
        self.halt_latched = False
        
    def decide(self, risk_score: float, effective_state: SystemState, system_error: bool = False) -> AuthorityDecision:
        """
        Determine the mandatory authority action.
        """
        # 0. Check Latching
        if self.halt_latched:
            return AuthorityDecision(
                action=AuthorityAction.HALT,
                reason="LATCHED_HALT",
                risk_value=risk_score,
                is_terminal=True
            )
            
        # 1. Check Overrides (Fail-Closed)
        if system_error:
            self.halt_latched = True
            return AuthorityDecision(
                action=AuthorityAction.HALT,
                reason="SYSTEM_ERROR",
                risk_value=1.0,
                is_terminal=True
            )
            
        if effective_state == SystemState.UNSTABLE:
            self.halt_latched = True
            return AuthorityDecision(
                action=AuthorityAction.HALT,
                reason="STATE_UNSTABLE",
                risk_value=1.0,
                is_terminal=True
            )
            
        # 2. Risk Mapping (Hard Matrix)
        # 0.00 <= R < 0.20 -> ALLOW
        # 0.20 <= R < 0.50 -> SLOW
        # 0.50 <= R < 0.75 -> CLARIFY
        # 0.75 <= R < 0.90 -> REFUSE
        # 0.90 <= R <= 1.00 -> HALT
        
        # Clamp Risk
        r = max(0.0, min(1.0, risk_score))
        
        action = AuthorityAction.ALLOW
        reason = f"Risk {r:.2f} < 0.20"
        
        if r >= 0.90:
            action = AuthorityAction.HALT
            reason = f"Risk {r:.2f} >= 0.90"
            self.halt_latched = True
        elif r >= 0.75:
            action = AuthorityAction.REFUSE
            reason = f"Risk {r:.2f} >= 0.75"
        elif r >= 0.50:
            action = AuthorityAction.CLARIFY
            reason = f"Risk {r:.2f} >= 0.50"
        elif r >= 0.20:
            action = AuthorityAction.SLOW
            reason = f"Risk {r:.2f} >= 0.20"
            
        return AuthorityDecision(
            action=action,
            reason=reason,
            risk_value=r,
            is_terminal=(action == AuthorityAction.HALT)
        )

# Unit Test
if __name__ == "__main__":
    print("Testing AuthorityInterface...")
    auth = AuthorityInterface()
    
    # 1. Normal Allow
    d = auth.decide(0.1, SystemState.NORMAL)
    print(f"1. R=0.10: {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.ALLOW
    
    # 2. Slow
    d = auth.decide(0.3, SystemState.NORMAL)
    print(f"2. R=0.30: {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.SLOW
    
    # 3. Refuse
    d = auth.decide(0.8, SystemState.ADVERSARIAL)
    print(f"3. R=0.80: {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.REFUSE
    
    # 4. Halt (Threshold)
    d = auth.decide(0.95, SystemState.ADVERSARIAL)
    print(f"4. R=0.95: {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.HALT
    assert d.is_terminal
    
    # 5. Check Latch (Input Safe, but previously Halted)
    d = auth.decide(0.0, SystemState.NORMAL)
    print(f"5. R=0.00 (After Halt): {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.HALT
    assert d.reason == "LATCHED_HALT"
    
    # Reset for next test
    auth.halt_latched = False
    
    # 6. Override Unstable
    d = auth.decide(0.0, SystemState.UNSTABLE)
    print(f"6. State=UNSTABLE: {d.action.name} ({d.reason})")
    assert d.action == AuthorityAction.HALT
