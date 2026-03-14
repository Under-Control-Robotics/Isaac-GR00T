#!/usr/bin/env python3
"""
Extract attention maps from GR00T model and overlay them on actual video frames.
This shows which image regions the action tokens focus on for each timestep.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2
from typing import List, Tuple
import json
import json
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.model.gr00t_n1 import GR00T_N1_5
from transformers.feature_extraction_utils import BatchFeature


class AttentionExtractor:
    """Extract cross-attention weights from GR00T model."""

    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self.hooks = []

    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights."""
        if hasattr(module, "_attention_weights"):
            self.attention_weights.append(module._attention_weights.detach().cpu())

    def patch_attention_modules(self):
        """Patch attention modules to save weights."""
        from diffusers.models.attention import Attention

        def forward_with_weights(
            self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs
        ):
            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            # Compute attention scores
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

            # Save attention weights
            self._attention_weights = attention_probs.detach()

            # Compute output
            hidden_states = torch.matmul(attention_probs, value)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, self.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            # Linear projection
            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states

        # Patch all attention modules in the action head
        for name, module in self.model.action_head.model.named_modules():
            if isinstance(module, Attention):
                module.forward = forward_with_weights.__get__(module, Attention)
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)

        print(f"Patched {len(self.hooks)} attention modules")

    def extract(self, inputs):
        """Run forward pass and extract attention."""
        self.attention_weights = []

        with torch.no_grad():
            output = self.model.get_action(inputs)

        return self.attention_weights, output

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_lerobot_episode(dataset_path: str, episode_idx: int = 1):
    """Load episode data from LeRobot dataset."""
    dataset_path = Path(dataset_path)

    print(f"Loading LeRobot dataset from {dataset_path}")

    # Load episodes metadata
    with open(dataset_path / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]

    # Find the episode
    target_episode = None
    for ep in episodes:
        if ep["episode_index"] == episode_idx:
            target_episode = ep
            break

    if target_episode is None:
        raise ValueError(f"Episode {episode_idx} not found")

    print(f"Episode {episode_idx}: {target_episode}")

    # Load parquet data for this episode
    parquet_path = dataset_path / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    table = pq.read_table(str(parquet_path))
    episode_data = table.to_pydict()

    print(f"Episode {episode_idx}: {len(episode_data['action'])} frames")

    return episode_data, episodes


def overlay_attention_on_frame(
    frame: np.ndarray, attention_map: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    """
    Overlay attention heatmap on video frame.

    Args:
        frame: RGB frame (H, W, 3)
        attention_map: Attention weights (H_attn, W_attn)
        alpha: Transparency of heatmap overlay

    Returns:
        Overlayed frame
    """
    h, w = frame.shape[:2]

    # Resize attention map to frame size
    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # Normalize to 0-255
    attention_normalized = (attention_resized - attention_resized.min()) / (
        attention_resized.max() - attention_resized.min() + 1e-8
    )
    attention_normalized = (attention_normalized * 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlayed = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)

    return overlayed


def visualize_attention_on_video(
    video_path: str,
    dataset_path: str,
    checkpoint_path: str,
    output_dir: str,
    episode_idx: int = 1,
    num_frames_to_visualize: int = 16,
):
    """
    Main function to visualize attention on video.

    Args:
        video_path: Path to video file
        dataset_path: Path to LeRobot dataset
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save outputs
        episode_idx: Which episode to visualize
        num_frames_to_visualize: Number of frames to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("GR00T Attention Visualization on Video")
    print("=" * 80)
    print(f"Video: {video_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    model = GR00T_N1_5.from_pretrained(
        checkpoint_path,
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
        device_map="cuda",
    )
    model.eval()
    print("Model loaded!\n")

    # Load dataset
    episode_data, episodes = load_lerobot_episode(dataset_path, episode_idx)

    # Get metadata
    with open(Path(dataset_path) / "meta" / "info.json") as f:
        info = json.load(f)

    task_description = info.get("tasks", ["Pick up the plastic box then place it in the tray"])[0]
    print(f"Task: {task_description}\n")

    # Load video
    print(f"Loading video from {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames\n")

    # Read frames
    frames = []
    while len(frames) < num_frames_to_visualize and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames\n")

    # Prepare model input
    print("Preparing model input...")

    # Stack frames for model input
    video_array = np.stack(frames)  # (T, H, W, C)
    video_array = video_array[np.newaxis, np.newaxis, ...]  # (1, 1, T, H, W, C)

    # Get states and actions from dataset
    # episode_data is a dict with keys like 'observation.state', 'action', etc.
    # Each value is a list of arrays
    num_frames = min(num_frames_to_visualize, len(episode_data["action"]))

    states = episode_data["observation.state"][:num_frames]
    actions = episode_data["action"][:num_frames]

    # Convert to tensors
    state_array = np.array(states[0:1])
    action_array = np.array(actions[:64])  # Use up to 64 frames for action_horizon

    # Pad or repeat actions if needed
    if action_array.shape[0] < 64:
        # Repeat last action to fill 64 timesteps
        repeats = 64 - action_array.shape[0]
        last_action = action_array[-1:]
        padding = np.repeat(last_action, repeats, axis=0)
        action_array = np.concatenate([action_array, padding], axis=0)

    # Pad action dimensions from 30 to 32 if needed
    if action_array.shape[1] < 32:
        pad_width = 32 - action_array.shape[1]
        action_array = np.pad(
            action_array, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )

    # Pad state dimensions if needed
    if state_array.shape[1] < 32:
        pad_width = 32 - state_array.shape[1]
        state_array = np.pad(
            state_array, ((0, 0), (0, pad_width)), mode="constant", constant_values=0
        )

    state_tensor = torch.tensor(state_array, dtype=torch.float32).cuda()
    action_tensor = torch.tensor(action_array, dtype=torch.float32).unsqueeze(0).cuda()

    print(f"Adjusted shapes:")
    print(f"  State: {state_tensor.shape}")
    print(f"  Action: {action_tensor.shape}")

    inputs = {
        "video": video_array,
        "text": [task_description],
        "state": state_tensor,
        "action": action_tensor,
        "embodiment_id": torch.zeros(1, dtype=torch.long).cuda(),
    }

    print(f"Input shapes:")
    print(f"  Video: {video_array.shape}")
    print(f"  State: {state_tensor.shape}")
    print(f"  Action: {action_tensor.shape}")
    print()

    # Extract attention
    print("Extracting attention maps...")
    extractor = AttentionExtractor(model)
    extractor.patch_attention_modules()

    attention_maps, output = extractor.extract(inputs)
    extractor.cleanup()

    print(f"Extracted {len(attention_maps)} attention layers")
    print(f"Predicted actions: {output['action_pred'].shape}\n")

    # Process attention maps
    print("Processing attention maps...")

    # Use last layer attention (most refined)
    if len(attention_maps) > 0:
        # Shape: (num_heads, num_action_tokens, num_vl_tokens)
        last_attn = attention_maps[-1]

        print(f"Last layer attention shape: {last_attn.shape}")

        # Average over heads
        last_attn_avg = last_attn.mean(dim=0)  # (num_action_tokens, num_vl_tokens)

        # Assume VLM tokens are ordered as [text_tokens, image_tokens]
        # We need to extract only image token attention
        # This depends on your backbone - adjust as needed
        num_image_tokens = 576  # 24x24 for typical vision encoder
        image_attn = last_attn_avg[:, -num_image_tokens:]  # (num_action_tokens, num_image_tokens)

        # Reshape to spatial dimensions
        h_attn = w_attn = int(np.sqrt(num_image_tokens))

        print(f"Image attention shape: {image_attn.shape}")
        print(f"Reshaping to: ({image_attn.shape[0]}, {h_attn}, {w_attn})\n")

        # Create visualizations
        print("Creating visualizations...")

        # 1. Save individual frames with attention overlay
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        for action_idx in range(min(len(frames), image_attn.shape[0])):
            # Get attention for this action token
            attn_map = image_attn[action_idx].numpy().reshape(h_attn, w_attn)

            # Overlay on frame
            overlayed = overlay_attention_on_frame(frames[action_idx], attn_map, alpha=0.5)

            # Save
            save_path = frames_dir / f"frame_{action_idx:03d}_with_attention.png"
            plt.figure(figsize=(10, 8))
            plt.imshow(overlayed)
            plt.title(f"Action Token {action_idx} Attention\n{task_description}", fontsize=12)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            if action_idx % 4 == 0:
                print(f"  Saved frame {action_idx}")

        # 2. Create summary grid
        print("\nCreating summary grid...")
        num_samples = min(8, len(frames), image_attn.shape[0])
        indices = np.linspace(0, min(len(frames), image_attn.shape[0]) - 1, num_samples, dtype=int)

        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))

        for i, idx in enumerate(indices):
            # Original frame
            axes[0, i].imshow(frames[idx])
            axes[0, i].set_title(f"Frame {idx}", fontsize=10)
            axes[0, i].axis("off")

            # Attention overlay
            attn_map = image_attn[idx].numpy().reshape(h_attn, w_attn)
            overlayed = overlay_attention_on_frame(frames[idx], attn_map, alpha=0.6)
            axes[1, i].imshow(overlayed)
            axes[1, i].set_title(f"Attention {idx}", fontsize=10)
            axes[1, i].axis("off")

        plt.suptitle(f"Attention Evolution\n{task_description}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        summary_path = output_dir / "summary_grid.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved summary to {summary_path}")

        # 3. Create video with attention overlay
        print("\nCreating output video...")
        output_video_path = output_dir / "attention_overlay_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

        for action_idx in range(min(len(frames), image_attn.shape[0])):
            attn_map = image_attn[action_idx].numpy().reshape(h_attn, w_attn)
            overlayed = overlay_attention_on_frame(frames[action_idx], attn_map, alpha=0.5)
            overlayed_bgr = cv2.cvtColor(overlayed.astype(np.uint8), cv2.COLOR_RGB2BGR)
            out.write(overlayed_bgr)

        out.release()
        print(f"Saved video to {output_video_path}")

        # 4. Save attention heatmaps only
        print("\nSaving attention heatmaps...")
        heatmap_dir = output_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(min(8, image_attn.shape[0])):
            attn_map = image_attn[i].numpy().reshape(h_attn, w_attn)

            im = axes[i].imshow(attn_map, cmap="hot", interpolation="bilinear")
            axes[i].set_title(f"Action {i}", fontsize=10)
            axes[i].axis("off")
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle("Attention Heatmaps Across Action Tokens", fontsize=14, fontweight="bold")
        plt.tight_layout()

        heatmap_path = heatmap_dir / "attention_heatmaps.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved heatmaps to {heatmap_path}")

    print("\n" + "=" * 80)
    print("✓ Visualization complete!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    video_path = "/data/anthony/ucr_ros/data_files/vla_dataset_output/grippy/joint_single_cam/2026-03-03_23:39:31.287473/videos/chunk-000/observation.images.torso_camera/episode_000001.mp4"
    dataset_path = "/data/anthony/ucr_ros/data_files/vla_dataset_output/grippy/joint_single_cam/2026-03-03_23:39:31.287473"
    checkpoint_path = (
        "/data/anthony/Isaac-GR00T/checkpoints/0305_pretrain_no_dagger/checkpoint-60000"
    )
    output_dir = "/data/anthony/Isaac-GR00T/attention_visualizations/0305_no_dagger_episode1"

    visualize_attention_on_video(
        video_path=video_path,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        episode_idx=1,
        num_frames_to_visualize=16,
    )
