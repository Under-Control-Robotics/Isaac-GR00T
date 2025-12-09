"""Test script to check if value head weights are loaded correctly."""

import torch
from gr00t.model.gr00t_n1 import GR00T_N1_5

# Load from checkpoint-30
print("Loading model from checkpoint-30...")
model = GR00T_N1_5.from_pretrained(
    pretrained_model_name_or_path="/data/anthony/Isaac-GR00T/checkpoints/1206_value_head/checkpoint-30",
    tune_llm=False,
    tune_visual=False,
    tune_projector=False,
    tune_diffusion_model=False,
    tune_value_head=False,
    enable_rl=True,
)

print("\n=== Value Head Weights ===")
for name, param in model.value_head.named_parameters():
    print(f"{name}:")
    print(f"  Shape: {param.shape}")
    print(f"  Mean: {param.mean().item():.6f}")
    print(f"  Std: {param.std().item():.6f}")
    print(f"  Min: {param.min().item():.6f}")
    print(f"  Max: {param.max().item():.6f}")
    if name == "value_net.0.weight":
        # Print first few values
        print(f"  First 5 values: {param.flatten()[:5].tolist()}")
    print()

print("\n=== Test Forward Pass ===")
# Create dummy input
batch_size = 1
seq_len = 549
hidden_size = 2048
dummy_features = (
    torch.randn(batch_size, seq_len, hidden_size).to(model.device).to(model.action_head.dtype)
)

with torch.no_grad():
    value_pred = model.value_head(dummy_features)
    print(f"Value prediction: {value_pred}")
    print(f"Value pred shape: {value_pred.shape}")
    print(f"Value pred dtype: {value_pred.dtype}")
