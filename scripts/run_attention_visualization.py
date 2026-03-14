#!/usr/bin/env python3
"""
Complete example script to extract and visualize attention maps from GR00T model.
This shows which regions of the VLM embeddings the action tokens focus on.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.model.gr00t_n1 import GR00T_N1_5
from visualize_attention import (
    extract_attention_from_model,
    visualize_attention_heatmap,
    visualize_action_to_image_attention,
    analyze_attention_statistics,
)


def main():
    """Main function to run attention visualization."""

    # Configuration
    checkpoint_path = "/data/anthony/Isaac-GR00T/checkpoints/0305_pretrain_up_sample10/checkpoint-18000"  # Adjust this
    output_dir = Path("/data/anthony/Isaac-GR00T/attention_visualizations")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GR00T Cross-Attention Visualization")
    print("=" * 60)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}\n")

    # Load model
    print("Loading model...")
    model = GR00T_N1_5.from_pretrained(
        checkpoint_path,
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
    )
    model.eval()
    model.cuda()
    print("Model loaded successfully!\n")

    # Prepare dummy input (you should replace this with real data)
    print("Preparing inputs...")
    batch_size = 1
    action_horizon = 16
    action_dim = 7  # Adjust based on your config

    # Dummy video input (replace with real video)
    # Shape: (batch, num_obs, time, height, width, channels)
    video = np.random.randint(0, 255, (batch_size, 1, 1, 224, 224, 3), dtype=np.uint8)

    inputs = {
        "video": video,
        "text": ["Pick up the plastic box then place it in the tray"],  # Your task
        "state": torch.randn(batch_size, 1, action_dim - 1).cuda(),  # Current robot state
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long).cuda(),
    }

    print(f"Input shapes:")
    print(f"  Video: {video.shape}")
    print(f"  State: {inputs['state'].shape}")
    print()

    # Extract attention maps
    print("Extracting attention maps (this may take a moment)...")
    attention_maps, output = extract_attention_from_model(model, inputs)

    print(f"Extracted {len(attention_maps)} attention map layers")
    print(f"Predicted actions shape: {output['action_pred'].shape}\n")

    # Analyze statistics
    analyze_attention_statistics(attention_maps)

    # Visualize attention heatmaps for each layer
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60 + "\n")

    for layer_idx, attn_map in enumerate(attention_maps):
        if attn_map.dim() == 3:  # (num_heads, query_len, key_len)
            num_heads, query_len, key_len = attn_map.shape

            print(f"Layer {layer_idx}: {attn_map.shape}")

            # 1. Average over all heads
            save_path = output_dir / f"layer_{layer_idx}_avg_heads.png"
            visualize_attention_heatmap(
                attn_map, layer_idx=layer_idx, head_idx=None, save_path=str(save_path)
            )

            # 2. Visualize individual heads (first 4 heads)
            for head_idx in range(min(4, num_heads)):
                save_path = output_dir / f"layer_{layer_idx}_head_{head_idx}.png"
                visualize_attention_heatmap(
                    attn_map, layer_idx=layer_idx, head_idx=head_idx, save_path=str(save_path)
                )

            # 3. Visualize action token attention to image regions
            # This shows which image regions each action timestep attends to
            for action_idx in [0, query_len // 2, query_len - 1]:  # First, middle, last
                if action_idx < query_len:
                    save_path = output_dir / f"layer_{layer_idx}_action_{action_idx}_to_image.png"
                    visualize_action_to_image_attention(
                        attn_map,
                        action_idx=action_idx,
                        image_shape=(24, 24),  # Adjust based on your vision encoder
                        save_path=str(save_path),
                    )

    # Create summary visualization
    print("\nCreating summary visualization...")
    create_summary_plot(attention_maps, output_dir)

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print("\nKey findings:")
    print("  - Check which image regions each action timestep focuses on")
    print("  - Earlier layers tend to be more diffuse, later layers more focused")
    print("  - Action tokens should attend to task-relevant regions")


def create_summary_plot(attention_maps, output_dir):
    """Create a summary plot showing attention evolution across layers."""
    num_layers = len(attention_maps)

    fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()

    for i, attn_map in enumerate(attention_maps):
        if attn_map.dim() == 3:
            # Average over heads
            attn = attn_map.mean(dim=0).numpy()

            ax = axes[i]
            im = ax.imshow(attn, cmap="viridis", aspect="auto")
            ax.set_title(f"Layer {i}", fontsize=10)
            ax.set_xlabel("VLM Tokens", fontsize=8)
            ax.set_ylabel("Action Tokens", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Remove extra subplots
    for i in range(len(attention_maps), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Cross-Attention Evolution Across Layers", fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = output_dir / "summary_all_layers.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot to {save_path}")


if __name__ == "__main__":
    main()
