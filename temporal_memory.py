
import time
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Deque, Optional
from collections import deque

# Import Enums
from state_estimator import SystemState
from authority_interface import AuthorityAction

@dataclass(frozen=True) 
class TemporalRecord:
    """Immutable record of a single interaction turn."""
    timestamp: float
    step_index: int
    state: SystemState
    risk_score: float
    action: AuthorityAction
    
    # Validation helper
    def is_valid(self) -> bool:
        if self.timestamp < 0: return False
        if not (0.0 <= self.risk_score <= 1.0): return False
        return True

class MemoryCorruptionError(Exception):
    pass

class TemporalMemory:
    """
    Axis-3: Temporal Memory Module.
    Tracks session history and computes Risk Bias.
    Bounded, Decay-Aware, Fail-Closed.
    """
    MAX_BIAS = 0.15
    BUFFER_SIZE = 10
    DECAY_FACTOR = 0.8 # Decay of Risk Mass per turn
    RESET_THRESHOLD_TURNS = 5 # Consecutive normal turns to reset
    RESET_TIME_SEC = 3600 # 1 hour
    
    def __init__(self):
        self._buffer: Deque[TemporalRecord] = deque(maxlen=self.BUFFER_SIZE)
        self._risk_mass: float = 0.0
        self._consecutive_normal_count: int = 0
        self._last_update_time: float = time.time()
        
    def add_record(self, record: TemporalRecord):
        """
        Append new record, apply decay, update aggregates.
        """
        # 1. Integrity / Clock Check (Fail-Closed)
        if record.timestamp < self._last_update_time - 1.0: # Allow slight drift 
             # Clock Anomaly (Time went backwards significant amount)
             # In a real system, we'd log this.
             # We treat it as potential tampering -> Max Bias?
             # Handled in compute, but we flag here?
             # Let's just store it, and compute will check consistency.
             pass

        self._last_update_time = max(self._last_update_time, record.timestamp)
        
        # 2. Update Risk Mass
        # Decay existing mass
        self._risk_mass *= self.DECAY_FACTOR
        # Add new risk
        self._risk_mass += record.risk_score
        
        # 3. Update Buffer
        self._buffer.append(record)
        
        # 4. Check Stability for Reset
        if record.state == SystemState.NORMAL and record.risk_score < 0.1:
            self._consecutive_normal_count += 1
        else:
            self._consecutive_normal_count = 0
            
        self.reset_if_stable()

    def reset_if_stable(self):
        """
        Reset memory if stability conditions met.
        """
        # Time-based reset check handled in compute or add?
        # If time diff > RESET_TIME, we should claim cold start?
        # Let's check consecutive normal count.
        if self._consecutive_normal_count >= self.RESET_THRESHOLD_TURNS:
            self._clear_memory("Stability Reset")
            
    def _clear_memory(self, reason: str):
        self._buffer.clear()
        self._risk_mass = 0.0
        self._consecutive_normal_count = 0
        # print(f"[TemporalMemory] Cleared: {reason}") # No print in prod

    def compute_temporal_bias(self) -> float:
        """
        Compute bias B_t based on history.
        Range [0.0, 0.15].
        """
        # 0. Fail-Closed Checks
        try:
            if not self._buffer:
                return 0.0 # Cold Start
            
            # Check Consistency
            if len(self._buffer) > 0:
                last_ts = self._buffer[0].timestamp
                for i in range(1, len(self._buffer)):
                    curr_ts = self._buffer[i].timestamp
                    if curr_ts < last_ts: 
                         # Time travel detected -> Corruption/Anomaly
                         return self.MAX_BIAS
                    last_ts = curr_ts
                    
        except Exception:
             # Corruption/Error
             return self.MAX_BIAS

        # 1. Probe Intensity
        # Count UNCERTAIN/CONFLICTED states that were NOT REFUSED.
        # This implies "Getting away with it".
        probe_count = 0
        for r in self._buffer:
            if r.state in [SystemState.UNCERTAIN, SystemState.CONFLICTED]:
                if r.action not in [AuthorityAction.REFUSE, AuthorityAction.HALT]:
                    probe_count += 1
        
        # Normalize Probe Intensity: 3 probes = max effect?
        # Weight w1. 
        # Design says: mix w1, w2, w3.
        # Let's act conservatively. One probe is OK. 3 is bad.
        probe_bias = min(1.0, probe_count / 3.0) * 0.05 # Max 0.05 contribution
        
        # 2. Escalation Rate (Trend)
        # Linear regression on Risk scores
        escalation_bias = 0.0
        if len(self._buffer) >= 3:
            y = np.array([r.risk_score for r in self._buffer])
            x = np.arange(len(y))
            # Slope
            slope = np.polyfit(x, y, 1)[0]
            if slope > 0.05: # Rising
                escalation_bias = min(1.0, slope * 5) * 0.05 # Max 0.05 contribution
        
        # 3. Risk Mass (Persistence)
        # Bounded contribution from accumulated mass
        # If mass > 2.0 (e.g. 2 high risk turns or many low ones)
        mass_bias = min(1.0, self._risk_mass / 3.0) * 0.05 # Max 0.05
        
        # Total
        total_bias = probe_bias + escalation_bias + mass_bias
        
        return min(self.MAX_BIAS, max(0.0, total_bias))

# --- Unit Tests ---
if __name__ == "__main__":
    print("Testing TemporalMemory...")
    tm = TemporalMemory()
    
    # 1. Cold Start
    assert tm.compute_temporal_bias() == 0.0
    print("1. Cold Start: 0.0")
    
    # 2. Escalating Probes
    # Add 3 Uncertain/Allow events with rising risk
    now = time.time()
    for i in range(3):
        rec = TemporalRecord(
            timestamp=now + i,
            step_index=i,
            state=SystemState.UNCERTAIN,
            risk_score=0.2 + (i*0.1), # 0.2, 0.3, 0.4
            action=AuthorityAction.ALLOW
        )
        tm.add_record(rec)
    
    bias = tm.compute_temporal_bias()
    print(f"2. Escalating Bias: {bias:.3f} (Expected > 0)")
    # Probes: 3 -> 1.0 * 0.05 = 0.05
    # Slope: 0.1 -> 0.5 * 0.05 = 0.025
    # Mass: Sum(.2, .3, .4 decays..) approx 0.8? -> 0.26 * 0.05 ~ 0.01
    # Total ~ 0.08?
    assert bias > 0.05
    assert bias <= 0.15
    
    # 3. Benign Streak Reset
    # Add 5 Normal/Low events
    for i in range(5):
        rec = TemporalRecord(
            timestamp=now + 10 + i,
            step_index=10+i,
            state=SystemState.NORMAL,
            risk_score=0.0,
            action=AuthorityAction.ALLOW
        )
        tm.add_record(rec)
        
    bias_reset = tm.compute_temporal_bias()
    print(f"3. After Reset: {bias_reset:.3f}")
    assert bias_reset == 0.0
    assert len(tm._buffer) == 0
    
    # 4. Clock Failure (Anomaly)
    # Add record in past
    rec_bad = TemporalRecord(now - 1000, 999, SystemState.NORMAL, 0, AuthorityAction.ALLOW)
    # We manually inject into buffer to simulate corruption state if needed, 
    # but `add_record` doesn't enforce order on input, `compute` checks order.
    # Let's inject a time-travel sequence.
    tm.add_record(TemporalRecord(now+100, 1, SystemState.NORMAL, 0, AuthorityAction.ALLOW))
    tm.add_record(TemporalRecord(now+90, 2, SystemState.NORMAL, 0, AuthorityAction.ALLOW)) # Backwards
    
    bias_fail = tm.compute_temporal_bias()
    print(f"4. Clock Failure Bias: {bias_fail:.3f}")
    assert bias_fail == TemporalMemory.MAX_BIAS
