"""Test if value head can be trained."""

import torch
from gr00t.model.gr00t_n1 import GR00T_N1_5

print("Loading model...")
model = GR00T_N1_5.from_pretrained(
    pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
    tune_llm=False,
    tune_visual=False,
    tune_projector=False,
    tune_diffusion_model=False,
    tune_value_head=True,
    enable_rl=True,
    value_head_cfg={"hidden_dim": 1024, "dropout": 0.1},
)

print("\n=== Checking trainable parameters ===")
trainable_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        trainable_params.append(name)
        print(f"  {name}: requires_grad={param.requires_grad}")

print(f"\nTotal trainable params: {len(trainable_params)}")

if not any("value_head" in name for name in trainable_params):
    print("ERROR: No value_head parameters are trainable!")
else:
    print("OK: Value head parameters are trainable")

print("\n=== Testing forward pass with loss ===")
model.cuda()
model.train()  # Ensure model is in training mode

# Create dummy input
batch_size = 2
dummy_input = {
    "image_inputs": torch.randn(batch_size, 1, 3, 224, 224).cuda(),
    "text_inputs": ["test"] * batch_size,
    "action": torch.randn(batch_size, 16, 32).cuda(),
    "value": torch.tensor([[-0.5], [-0.3]], dtype=torch.float32).cuda(),  # Dummy value labels
}

# Forward pass
print("Running forward pass...")
output = model(dummy_input)

print(f"\nOutput keys: {output.keys()}")
if "value_loss" in output:
    print(f"Value loss: {output['value_loss']}")
    print(f"Value loss requires_grad: {output['value_loss'].requires_grad}")
else:
    print("ERROR: No value_loss in output!")

if "loss" in output:
    print(f"Total loss: {output['loss']}")
    print(f"Total loss requires_grad: {output['loss'].requires_grad}")

    # Test backward pass
    print("\n=== Testing backward pass ===")
    initial_weight = model.value_head.value_net[0].weight.clone()
    initial_mean = initial_weight.mean().item()

    output["loss"].backward()

    # Check if gradients exist
    value_head_grad = model.value_head.value_net[0].weight.grad
    if value_head_grad is not None:
        print(f"Value head gradient mean: {value_head_grad.mean().item():.6f}")
        print(f"Value head gradient norm: {value_head_grad.norm().item():.6f}")

        # Simulate optimizer step
        with torch.no_grad():
            model.value_head.value_net[0].weight -= 0.01 * value_head_grad

        final_weight = model.value_head.value_net[0].weight
        final_mean = final_weight.mean().item()

        print(f"\nWeight change: {initial_mean:.6f} -> {final_mean:.6f}")
        print(f"Difference: {abs(final_mean - initial_mean):.6f}")

        if abs(final_mean - initial_mean) > 1e-6:
            print("SUCCESS: Value head weights can be updated!")
        else:
            print("ERROR: Value head weights did not change!")
    else:
        print("ERROR: No gradient for value_head!")
else:
    print("ERROR: No loss in output!")
