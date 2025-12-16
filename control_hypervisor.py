
import time
import math
import torch
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

# Import Control Core Modules
# Assuming they are in the same directory or python path
from signal_extractor import SignalExtractor, SignalFrame
from state_estimator import StateEstimator, StateOutput, SystemState
from risk_engine import RiskEngine, RiskComponents
from authority_interface import AuthorityInterface, AuthorityAction, AuthorityDecision

class GenerationHalt(Exception):
    """Raised when the Control Core mandates an immediate HALT."""
    pass

@dataclass
class HypervisorOutput:
    step_id: int
    action: AuthorityAction
    sampling_params: Dict[str, Any] # e.g. {'temp': 0.7, 'top_k': 10}
    injection_text: Optional[str] # For Clarify/Refuse
    metadata: Dict[str, Any] # For logging

class ControlHypervisor:
    """
    The Single Source of Truth for Safety.
    Orchestrates Signal -> State -> Risk -> Authority.
    Enforces Hard Interrupts.
    """
    def __init__(self, layer_indices: List[int], feature_dim: int):
        # Initialize Sub-modules
        self.signal_extractor = SignalExtractor(layer_indices, feature_dim)
        self.state_estimator = StateEstimator()
        self.risk_engine = RiskEngine()
        self.authority = AuthorityInterface()
        
        # State tracking
        self.step_counter = 0
        self.last_step_time = time.time()
        self.refusal_vector: Optional[torch.Tensor] = None # Should be loaded
        
    def set_refusal_vector(self, vector: torch.Tensor):
        self.refusal_vector = vector

    def _validate_signals(self, frame: SignalFrame):
        """Fail-closed check for NaNs or invalid ranges."""
        vals = [frame.sigma_stab, frame.sigma_coher, frame.sigma_intent, frame.sigma_surp]
        for v in vals:
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Signal NaN/Inf detected: {frame}")
                
    def process_step(self, 
                     step_id: int, 
                     hidden_states: torch.Tensor, 
                     attentions: torch.Tensor) -> HypervisorOutput:
        """
        Run the Control Loop for one generation step.
        MUST run synchronously.
        """
        start_time = time.time()
        
        # 1. Fail-Closed: Lag Check
        # We expect step_id to be sequential.
        if step_id != self.step_counter:
            # If step_id > counter + 1, we missed steps?
            # If step_id < counter, duplicate?
            # Strict lockstep required.
            if step_id > self.step_counter:
                 # We missed a step -> LAG -> HALT
                 # "Control loop lag > 1 step -> HALT"
                 # Actually if step_id is provided by caller, we check against our internal counter.
                 raise GenerationHalt(f"Control Lag Detected: Step {step_id} received, expected {self.step_counter}")
        
        self.step_counter += 1
        
        # 2. Fail-Closed: Error Boundary
        try:
             # Check inputs
            if self.refusal_vector is None:
                raise ValueError("Refusal Vector not initialized")

            # --- A. SIGNAL ---
            sig_frame = self.signal_extractor.extract(hidden_states, attentions, self.refusal_vector)
            self._validate_signals(sig_frame)
            
            # --- B. STATE ---
            state_out = self.state_estimator.update(sig_frame)
            
            # --- C. RISK ---
            # Risk Engine needs Effective State and Window
            # StateEstimator holds the window internally! 
            # RiskEngine.compute_risk expects inputs: (effective_state, window).
            # We need access to `state_estimator.window` but that deque holds `SystemState`s.
            # Wait, RiskEngine needs signal window (for Trend/Vol).
            # `state_estimator` holds `SystemState` window.
            # We need to maintain a `SignalFrame` window here in Hypervisor?
            # Or `RiskEngine` maintains it?
            # RiskEngine specification Phase III: "Inputs: ... Sliding window of Past SignalFrames".
            # `RiskEngine` class I wrote is stateless regarding window storage (it accepted `signal_window` as arg).
            # So Hypervisor must maintain the Signal History.
            
        except Exception as e:
            # "Exception in any control module -> HALT"
            # We treat this as critical system failure.
            # We delegate to Authority with system_error=True
            decision = self.authority.decide(1.0, SystemState.UNSTABLE, system_error=True)
            if decision.action == AuthorityAction.HALT:
                raise GenerationHalt(f"System Error: {str(e)}")
            else:
                # Should not happen given spec
                raise GenerationHalt(f"System Error (Escalated): {str(e)}")

        # Maintain Signal Window for Risk Engine
        if not hasattr(self, 'signal_window'):
            from collections import deque
            self.signal_window = deque(maxlen=5)
        self.signal_window.append(sig_frame)
        
        # --- C. RISK (Continued) ---
        try:
            risk_out = self.risk_engine.compute_risk(state_out.effective_state, self.signal_window)
        except Exception as e:
            raise GenerationHalt(f"Risk Computation Error: {str(e)}")

        # --- D. AUTHORITY ---
        decision = self.authority.decide(risk_out.total_risk, state_out.effective_state)
        
        # 3. Enforce Action
        if decision.action == AuthorityAction.HALT:
            raise GenerationHalt(f"HALT Triggered: {decision.reason} (R={decision.risk_value:.2f})")
            
        # Prepare Output Modifiers
        sampling_updates = {}
        injection = None
        
        if decision.action == AuthorityAction.SLOW:
            # "adjust sampling parameters (temperature / top-k)"
            # E.g. Lower temp to 0.7, Top-K to 10
            sampling_updates = {'temperature': 0.7, 'top_k': 10}
            
        elif decision.action == AuthorityAction.CLARIFY:
            # "interrupt and inject clarification system message"
            injection = "[SYSTEM: Please clarify your intent regarding safety directives.]"
            # Note: The Loop must handle this injection logic.
            
        elif decision.action == AuthorityAction.REFUSE:
             # "interrupt and emit refusal output"
             injection = "I cannot fulfill this request due to safety controls."
             # Refusal implies stopping the current gen and replacing it? 
             # Or just emitting this token sequence?
             # Spec Phase IV: "Replace output buffer with Refusal Token".
             # Hypervisor returns injection_text. Loop should use it.
             
        # Log (Mock)
        # In real sys, structured logging.
        # print(f"[HYPERVISOR] Step {step_id} | R={risk_out.total_risk:.2f} | Act={decision.action.name}")
        
        self.last_step_time = time.time()
        
        return HypervisorOutput(
            step_id=step_id,
            action=decision.action,
            sampling_params=sampling_updates,
            injection_text=injection,
            metadata={
                'signals': asdict(sig_frame),
                'state': state_out.effective_state.name,
                'risk': asdict(risk_out)
            }
        )

# Integration Test
if __name__ == "__main__":
    print("Testing ControlHypervisor Integration...")
    
    # Mock Tensors
    feature_dim = 128
    layer_indices = [5, 10, 15, 20, 24]
    
    hypervisor = ControlHypervisor(layer_indices, feature_dim)
    
    # Set refusal vector
    refusal_vec = torch.randn(feature_dim)
    hypervisor.set_refusal_vector(refusal_vec)
    
    # Simulation Loop
    try:
        for i in range(10):
            # Mock Data
            hidden = torch.randn(25, 128) 
            attn = torch.rand(1, 32, 10, 10)
            
            # Scenario: Step 5 introduces NaN -> Should Halt
            if i == 5:
                # Inject NaN in hidden states to cause NaN signal
                hidden[5, 0] = float('nan')
                print(f"\n--- Injecting NaN at Step {i} ---")
            
            # Scenario: Step 8 introduces High Intent -> Should Refuse/Halt
            # But loop will probably die at 5.
            
            print(f"gen_step({i})...")
            out = hypervisor.process_step(i, hidden, attn)
            print(f" -> Action: {out.action.name}")
            
    except GenerationHalt as halt:
        print(f"\n!!! GENERATION HALTED: {halt} !!!")
      
    # Reset for Scenario 2: High Risk
    print("\n--- Scenario 2: Adversarial Spike ---")
    hypervisor = ControlHypervisor(layer_indices, feature_dim)
    hypervisor.set_refusal_vector(refusal_vec)
    
    try:
        for i in range(10):
            hidden = torch.randn(25, 128)
            attn = torch.rand(1, 32, 10, 10)
            
            # Inject Adversarial intent (align with refusal vector)
            if i >= 3:
                # Make trajectory align with refusal
                # delta ~ refusal
                # v_next = v_curr + refusal * 0.5
                # We need to construct hidden states such that `extract` sees alignment.
                # Complex to mock perfectly without `extract` logic knowledge, 
                # but we can assume `extract` works and just mock the `signal_extractor.extract` method?
                # No, "Do not modify upstream".
                # Let's brute force valid tensors? NO.
                # Let's just trust unit test above for NaN detection.
                # To test Act enforcement, we can modify the internal state of signal extractor? 
                # Or just Mock the signal extractor instance method for testing?
                pass
                
                # Mocking logic for integration test purposes ONLY:
                # We manually inject a "Malicious Frame" into the Hypervisor's flow?
                # No, process_step calls extract.
                pass
    except GenerationHalt as halt:
         print(f"!!! HALTED: {halt} !!!")
         
    # To properly test "Adversarial Spike" without massive tensor math:
    # I will override `signal_extractor.extract` on the instance to return a bad frame.
    
    def mock_extract_bad(*args):
        # High Intent (0.95), Normal everything else
        return SignalFrame(1.0, 1.0, 0.95, 0.0, 0,0,0,0,0,0)
    
    hypervisor.signal_extractor.extract = mock_extract_bad
    
    try:
        print("gen_step(0) with Malicious Mock...")
        out = hypervisor.process_step(0, torch.randn(25,128), torch.randn(1,32,10,10))
        print(f" -> Action: {out.action.name}")
        
    except GenerationHalt as halt:
        print(f"!!! HALTED: {halt} !!!")
    except Exception as e:
        print(f"Error: {e}")

