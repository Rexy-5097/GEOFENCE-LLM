# GEOFENCE-LLM v3.5 - Project Setup & Verification Guide

## 1. Environment Explanation
We prioritize **stability** and **Apple Silicon (MPS) support**. Python 3.10 is chosen for broad compatibility with PyTorch ecosystem tools while maintaining stability.

### Conda Environment
Construct the environment to isolate dependencies and ensure reproducibility.

```bash
conda create -n geofence-llm python=3.10 -y
conda activate geofence-llm
```

### Package Versions
- **PyTorch (2.4+)**: Essential for robust MPS backend support (Hardware Acceleration on Mac).
- **Transformers**: For model loading.
- **Accelerate**: Optimizes large model loading on consumer hardware.
- **Numpy**: Vector operations.
- **Psutil**: Memory monitoring.

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate numpy psutil
```
*(PyTorch standard install on Mac now includes MPS support by default)*

## 2. MPS Verification
We must confirm the code is running on the Neural Engine / GPU, not CPU.
Code check: `torch.backends.mps.is_available()` must return `True`.

## 3. Loading Llama-3.2-3B-Instruct Safely
- **Precision**: `torch.float16` to halve memory usage (vital for <8GB target).
- **Device Map**: `device_map="mps"` automatically places layers on the GPU.
- **Auth**: You must be logged in via `huggingface-cli login` to access this gated model.

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="mps"
)
```

## 4. Extracting Hidden States (No Generation)
We do NOT use `.generate()`. We use the model as a pure function $F(x) \to h$.
We rely on `output_hidden_states=True` in the forward pass.

```python
outputs = model(input_ids, output_hidden_states=True)
# outputs.hidden_states is a tuple of (num_layers + 1) tensors
# Shape: (batch, seq_len, hidden_dim)
```

## 5. Expected Memory Footprint
- **Model Weights (FP16)**: ~3 billion parameters × 2 bytes ≈ **6.0 GB**
- **KV Cache / Activations**: Minimized by `torch.no_grad()` and short context.
- **System Overhead**: ~0.5 - 1.0 GB.
- **Total**: ~7.0 - 7.5 GB. (Fits within 8GB limit, but tight. Close other apps).
