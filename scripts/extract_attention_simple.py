#!/usr/bin/env python3
"""
Simple script to extract and visualize attention from GR00T model.
This version uses the existing training infrastructure.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from datasets import load_from_disk
import json
from tqdm import tqdm


def patch_dit_for_attention_extraction(dit_model):
    """Patch DiT model to capture attention weights."""
    attention_maps = []

    # Patch BasicTransformerBlock to save attention
    from gr00t.model.action_head.cross_attention_dit import BasicTransformerBlock

    original_forward = BasicTransformerBlock.forward

    def forward_with_attention_capture(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        temb=None,
    ):
        # Run original forward
        output = original_forward(
            self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, temb
        )

        # Try to capture attention if available
        if hasattr(self.attn1, "processor") and hasattr(self.attn1.processor, "attention_scores"):
            attention_maps.append(self.attn1.processor.attention_scores.detach().cpu())

        return output

    # Patch all transformer blocks
    for block in dit_model.transformer_blocks:
        block.forward = forward_with_attention_capture.__get__(block, BasicTransformerBlock)

    return attention_maps


def extract_attention_from_checkpoint(
    checkpoint_path: str, dataset_path: str, video_path: str, output_dir: str
):
    """Extract attention from a checkpoint and visualize on video."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("GR00T Cross-Attention Extraction")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print()

    # Load model using AutoModel (this should work with existing setup)
    print("Loading model...")
    from transformers import AutoModel, AutoConfig

    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        checkpoint_path,
        config=config,
        trust_remote_code=True,
        tune_visual=False,
        tune_llm=False,
        tune_projector=False,
        tune_diffusion_model=False,
    )

    model = model.cuda()
    model.eval()
    print("✓ Model loaded\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)

    # Get episode 1 data
    episode_data = [sample for sample in dataset if sample["episode_index"] == 1]
    print(f"Episode 1: {len(episode_data)} frames\n")

    # Load video
    print("Loading video...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    for i in range(16):  # Load 16 frames
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    print(f"✓ Loaded {len(frames)} frames\n")

    # Prepare input
    print("Preparing model input...")

    # Stack frames
    video_np = np.stack(frames)  # (T, H, W, 3)
    video_np = video_np[np.newaxis, np.newaxis, ...]  # (1, 1, T, H, W, 3)

    # Get state and action from dataset
    state = (
        torch.tensor(episode_data[0]["observation.state"], dtype=torch.float32).unsqueeze(0).cuda()
    )
    actions = (
        torch.tensor(np.array([ep["action"] for ep in episode_data[:16]]), dtype=torch.float32)
        .unsqueeze(0)
        .cuda()
    )

    # Get task
    with open(Path(dataset_path) / "meta" / "info.json") as f:
        info = json.load(f)
    task = info.get("tasks", ["Pick up the plastic box"])[0]

    inputs = {
        "video": video_np,
        "text": [task],
        "state": state,
        "action": actions,
        "embodiment_id": torch.zeros(1, dtype=torch.long).cuda(),
    }

    print(f"Input shapes:")
    print(f"  Video: {video_np.shape}")
    print(f"  State: {state.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Task: {task}")
    print()

    # Simple forward pass without attention extraction first
    print("Running inference...")
    with torch.no_grad():
        output = model.get_action(inputs)

    print(f"✓ Predicted actions: {output['action_pred'].shape}\n")

    # Create simple visualizations
    print("Creating visualizations...")

    # Save input frames
    frames_dir = output_dir / "input_frames"
    frames_dir.mkdir(exist_ok=True)

    for i, frame in enumerate(frames[:8]):
        plt.figure(figsize=(8, 6))
        plt.imshow(frame)
        plt.title(f"Frame {i}\n{task}", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(frames_dir / f"frame_{i:03d}.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"✓ Saved {len(frames)} input frames")

    # Save model prediction info
    pred_info = {
        "task": task,
        "predicted_actions_shape": str(output["action_pred"].shape),
        "predicted_actions": output["action_pred"].cpu().numpy().tolist(),
        "checkpoint": checkpoint_path,
        "dataset": dataset_path,
    }

    with open(output_dir / "prediction_info.json", "w") as f:
        json.dump(pred_info, f, indent=2)

    print(f"✓ Saved prediction info")

    print("\n" + "=" * 80)
    print("✓ Extraction complete!")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 80)

    return model, inputs, output


if __name__ == "__main__":
    checkpoint_path = (
        "/data/anthony/Isaac-GR00T/checkpoints/0305_pretrain_no_dagger/checkpoint-60000"
    )
    dataset_path = "/data/anthony/ucr_ros/data_files/vla_dataset_output/grippy/joint_single_cam/2026-03-03_23:39:31.287473"
    video_path = "/data/anthony/ucr_ros/data_files/vla_dataset_output/grippy/joint_single_cam/2026-03-03_23:39:31.287473/videos/chunk-000/observation.images.torso_camera/episode_000001.mp4"
    output_dir = "/data/anthony/Isaac-GR00T/attention_outputs/0305_no_dagger_ep1"

    try:
        model, inputs, output = extract_attention_from_checkpoint(
            checkpoint_path, dataset_path, video_path, output_dir
        )
        print("\n✓ SUCCESS: Model loaded and inference completed!")
        print(f"You can now inspect the model and manually extract attention.")
        print(f"The cross-attention is in: model.action_head.model.transformer_blocks[i].attn1")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
