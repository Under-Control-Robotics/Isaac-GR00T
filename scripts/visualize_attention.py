#!/usr/bin/env python3
"""
Script to extract and visualize cross-attention maps from GR00T model.
This shows which VLM image/text regions the action tokens attend to.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import seaborn as sns


class AttentionHook:
    """Hook to capture attention weights from the model."""

    def __init__(self):
        self.attention_maps = []
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks to capture attention weights."""

        def attention_hook(module, input, output):
            """Hook function to capture attention weights."""
            # The Attention module from diffusers returns the output
            # We need to modify it to also return attention weights
            # For now, we'll capture them by patching the processor
            if hasattr(module, "attn_weights"):
                self.attention_maps.append(module.attn_weights.detach().cpu())

        # Register hooks on all attention layers in DiT
        for name, module in model.named_modules():
            if "attn1" in name and hasattr(module, "forward"):
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)

        print(f"Registered {len(self.hooks)} attention hooks")

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_attention_maps(self):
        """Get captured attention maps."""
        return self.attention_maps

    def clear(self):
        """Clear captured attention maps."""
        self.attention_maps = []


def patch_attention_to_return_weights(model):
    """
    Patch the attention modules to save attention weights.
    This modifies the diffusers Attention processor.
    """
    from diffusers.models.attention_processor import Attention

    original_forward = Attention.forward

    def forward_with_attn_weights(
        self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
    ):
        """Modified forward that saves attention weights."""
        batch_size, sequence_length, _ = hidden_states.shape

        # Get query, key, value
        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        # Calculate attention weights
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        # Save attention weights
        self.attn_weights = attention_probs.detach()

        # Calculate output
        hidden_states = torch.matmul(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states

    # Patch all attention modules
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.forward = forward_with_attn_weights.__get__(module, Attention)

    print("Patched attention modules to return weights")


def visualize_attention_heatmap(
    attention_weights: torch.Tensor,
    layer_idx: int,
    head_idx: int = None,
    save_path: str = None,
    figsize=(12, 8),
):
    """
    Visualize attention heatmap.

    Args:
        attention_weights: Shape (num_heads, query_len, key_len) or (query_len, key_len)
        layer_idx: Which DiT layer this is from
        head_idx: Which attention head (if None, average over all heads)
        save_path: Path to save the figure
    """
    if attention_weights.dim() == 3:
        # Average over heads if not specified
        if head_idx is not None:
            attn = attention_weights[head_idx].numpy()
            title = f"Layer {layer_idx}, Head {head_idx}"
        else:
            attn = attention_weights.mean(dim=0).numpy()
            title = f"Layer {layer_idx}, All Heads (avg)"
    else:
        attn = attention_weights.numpy()
        title = f"Layer {layer_idx}"

    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        attn,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Attention Weight"},
    )

    plt.xlabel("VLM Token Index (Keys)", fontsize=12)
    plt.ylabel("Action Token Index (Queries)", fontsize=12)
    plt.title(f"Cross-Attention Heatmap\n{title}", fontsize=14, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved attention heatmap to {save_path}")

    plt.show()


def visualize_action_to_image_attention(
    attention_weights: torch.Tensor,
    action_idx: int,
    image_shape: tuple = (24, 24),  # Assuming 576 = 24x24 image tokens
    save_path: str = None,
):
    """
    Visualize which image regions a specific action token attends to.

    Args:
        attention_weights: Shape (num_heads, num_actions, num_vl_tokens)
        action_idx: Which action timestep to visualize
        image_shape: Shape to reshape VLM tokens into (H, W)
        save_path: Path to save figure
    """
    # Average over attention heads
    attn = attention_weights.mean(dim=0)[action_idx].numpy()  # (num_vl_tokens,)

    # Assume VLM tokens are [text_tokens, image_tokens]
    # You may need to adjust this based on your backbone
    num_image_tokens = image_shape[0] * image_shape[1]

    if len(attn) >= num_image_tokens:
        # Take last num_image_tokens (assuming image tokens are at the end)
        image_attn = attn[-num_image_tokens:]

        # Reshape to 2D
        attn_map = image_attn.reshape(image_shape)

        plt.figure(figsize=(8, 8))
        plt.imshow(attn_map, cmap="hot", interpolation="bilinear")
        plt.colorbar(label="Attention Weight")
        plt.title(f"Action Token {action_idx} → Image Regions", fontsize=14, fontweight="bold")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved image attention map to {save_path}")

        plt.show()
    else:
        print(f"Warning: Not enough tokens ({len(attn)}) for image shape {image_shape}")


def analyze_attention_statistics(attention_maps: List[torch.Tensor]):
    """Analyze attention statistics across layers."""
    print("\n=== Attention Statistics ===")

    for i, attn in enumerate(attention_maps):
        if attn.dim() == 3:
            # (num_heads, query_len, key_len)
            num_heads, query_len, key_len = attn.shape

            # Entropy (how focused is the attention)
            entropy = -(attn * torch.log(attn + 1e-10)).sum(dim=-1)
            avg_entropy = entropy.mean().item()

            # Max attention weight
            max_attn = attn.max().item()

            print(f"Layer {i}:")
            print(f"  Shape: {attn.shape}")
            print(f"  Avg Entropy: {avg_entropy:.4f} (lower = more focused)")
            print(f"  Max Attention: {max_attn:.4f}")
            print(f"  Mean Attention: {attn.mean().item():.4f}")
            print()


def extract_attention_from_model(model, inputs):
    """
    Extract attention maps from a forward pass.

    Returns:
        attention_maps: List of attention weight tensors
    """
    # Create hook
    hook = AttentionHook()

    # Patch attention modules
    patch_attention_to_return_weights(model.action_head.model)

    # Register hooks
    hook.register_hooks(model.action_head.model)

    # Forward pass
    with torch.no_grad():
        output = model.get_action(inputs)

    # Get attention maps
    attention_maps = hook.get_attention_maps()

    # Clean up
    hook.remove_hooks()

    return attention_maps, output


# Example usage
if __name__ == "__main__":
    """
    Example of how to use this script with your model.
    """

    print("Attention Visualization Script for GR00T")
    print("=" * 50)
    print()
    print("To use this script:")
    print("1. Load your GR00T model")
    print("2. Prepare your input data")
    print("3. Call extract_attention_from_model(model, inputs)")
    print("4. Visualize with the provided functions")
    print()
    print("Example:")
    print(
        """
    from gr00t.model.gr00t_n1 import GR00T_N1_5

    # Load model
    model = GR00T_N1_5.from_pretrained("your-checkpoint")
    model.eval()

    # Prepare inputs
    inputs = {
        'video': ...,  # Your video input
        'state': ...,  # Robot state
        'text': ...,   # Task description
        # ... other inputs
    }

    # Extract attention
    attention_maps, output = extract_attention_from_model(model, inputs)

    # Analyze
    analyze_attention_statistics(attention_maps)

    # Visualize
    for i, attn_map in enumerate(attention_maps):
        visualize_attention_heatmap(attn_map, layer_idx=i, save_path=f'attn_layer_{i}.png')

    # Visualize specific action token attention to image
    visualize_action_to_image_attention(
        attention_maps[0],  # First layer
        action_idx=0,       # First action timestep
        save_path='action_to_image_attn.png'
    )
    """
    )
