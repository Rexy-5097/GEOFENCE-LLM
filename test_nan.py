
import torch
import math
from signal_extractor import SignalExtractor

def test_nan():
    layers = [5, 10, 15, 20, 24]
    extractor = SignalExtractor(layers, 128)
    
    hidden = torch.randn(25, 128)
    hidden[5, 0] = float('nan') # Inject NaN
    
    attn = torch.rand(1, 32, 10, 10)
    ref = torch.randn(128)
    
    frame = extractor.extract(hidden, attn, ref)
    print("Frame:", frame)
    
    vals = [frame.sigma_stab, frame.sigma_coher, frame.sigma_intent, frame.sigma_surp]
    has_nan = any(math.isnan(v) for v in vals)
    print("Has NaN:", has_nan)

if __name__ == "__main__":
    test_nan()
