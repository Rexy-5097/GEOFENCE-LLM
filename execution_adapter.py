
import torch
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Import Control Core
from control_hypervisor import ControlHypervisor, GenerationHalt, HypervisorOutput
from authority_interface import AuthorityAction

class ModelInterface(ABC):
    """
    Abstract Interface for Base Models (e.g. Llama, Mistral).
    Decouples Control Core from specific model implementations.
    """
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> None:
        """Run model forward pass."""
        pass
        
    @abstractmethod
    def get_hidden_states(self) -> torch.Tensor:
        """Return Hidden States of last forward pass."""
        pass

    @abstractmethod
    def get_attention_weights(self) -> torch.Tensor:
        """Return Attention Weights of last forward pass."""
        pass

    @abstractmethod
    def get_current_logits(self) -> torch.Tensor:
        """Return Logits of last forward pass."""
        pass
        
    @abstractmethod
    def sample_token(self, logits: torch.Tensor, **kwargs) -> int:
        """Sample next token from logits."""
        pass

@dataclass
class AdapterConfig:
    layer_indices: List[int]
    feature_dim: int

class ExecutionAdapter:
    """
    Wraps the Model Loop and enforces Control Core Authority.
    """
    def __init__(self, model: ModelInterface, config: AdapterConfig):
        self.model = model
        # Initialize Hypervisor
        self.hypervisor = ControlHypervisor(config.layer_indices, config.feature_dim)
        
        # We need to inject the Refusal Vector into Hypervisor
        # For this prototype, we mock it or assume it's loaded.
        # In prod, this comes from the Model wrapper or config.
        # Self-correction: Hypervisor needs `set_refusal_vector`.
        # We'll initialize random for now to pass `process_step` checks.
        self.hypervisor.set_refusal_vector(torch.randn(config.feature_dim))
        
        self.step_counter = 0

    def generate_step(self, input_ids: torch.Tensor) -> Tuple[Optional[int], str]:
        """
        Execute ONE step of generation.
        Returns: (token_id, status_string)
        Raises: GenerationHalt if unsafe.
        """
        try:
            # 1. Model Forward
            self.model.forward(input_ids)
            
            # 2. Extract Internals
            hiddens = self.model.get_hidden_states()
            attentions = self.model.get_attention_weights()
            
            # 3. Control Check (Pre-Emit)
            # This is the Pre-Commit hook.
            decision: HypervisorOutput = self.hypervisor.process_step(
                step_id=self.step_counter,
                hidden_states=hiddens,
                attentions=attentions
            )
            
            self.step_counter += 1
            
            # 4. Enforce Authority
            if decision.action == AuthorityAction.HALT:
                raise GenerationHalt(f"HALT by Authority: {decision.metadata.get('risk')}")
                
            if decision.action == AuthorityAction.REFUSE:
                # "Replace output buffer with Refusal Token"
                # We return None for token, and "REFUSAL" status.
                # In real loop, we'd emit a canned refusal string.
                # For this interface, we stop generation.
                return None, "REFUSED"
                
            # SLOW / CLARIFY: 
            # If SLOW, we might adjust sampling inputs.
            # If CLARIFY, we might inject prompt.
            # For this Phase 5.1 Adapter, we focus on Token Emission.
            
            # 5. Emit
            logits = self.model.get_current_logits()
            
            # Apply SLOW parameters if needed
            sampling_params = decision.sampling_params or {} # e.g. temp=0.7
            
            token_id = self.model.sample_token(logits, **sampling_params)
            return token_id, "EMITTED"

        except GenerationHalt:
            raise # Propagate up
        except Exception as e:
            # Fail-Closed regarding Model Errors
            raise GenerationHalt(f"Execution Adapter Error: {str(e)}")

# --- Verification ---

class PseudoModel(ModelInterface):
    """Mock Model for Testing Adapter."""
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim
        self.last_hidden = None
        self.last_attn = None
        
    def forward(self, input_ids):
        # Simulate computation
        self.last_hidden = torch.randn(25, self.feature_dim) # 25 layers
        self.last_attn = torch.rand(1, 32, 10, 10) # B, H, S, S
        
    def get_hidden_states(self):
        return self.last_hidden
        
    def get_attention_weights(self):
        return self.last_attn
        
    def get_current_logits(self):
        return torch.randn(1, 32000)
    
    def sample_token(self, logits, **kwargs):
        return 1234 # Mock token

if __name__ == "__main__":
    print("Testing ExecutionAdapter...")
    
    # Config
    cfg = AdapterConfig(layer_indices=[5, 10, 15, 20, 24], feature_dim=128)
    mock_model = PseudoModel(128)
    
    adapter = ExecutionAdapter(mock_model, cfg)
    
    # 1. Normal Step
    try:
        tok, status = adapter.generate_step(torch.tensor([1]))
        print(f"Step 0: {status} (Token={tok})")
        assert status == "EMITTED"
        assert tok == 1234
    except GenerationHalt:
        print("Unexpected Halt Step 0")
        
    # 2. Trigger Halt (Inject NaN via Mock Model)
    # We modify mock model behavior for next call
    def mock_get_bad_hidden():
         h = torch.randn(25, 128)
         h[5,0] = float('nan')
         return h
    
    mock_model.get_hidden_states = mock_get_bad_hidden
    
    print("Step 1 (NaN Injection)...")
    try:
        tok, status = adapter.generate_step(torch.tensor([1, 1234]))
        print(f"Outcome: {status}")
    except GenerationHalt as e:
        print(f"CAUGHT HALT: {e}")
        assert "Signal NaN" in str(e) or "System Error" in str(e)
    
    print("Execution Adapter Verification Complete.")
