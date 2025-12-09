# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualization script for RL-finetuned GR00T model.

Loads a model with value head and creates videos showing:
- Top: Robot observation video
- Bottom: Value function curve over episode timesteps

Usage:
    python scripts/visualize_rl_rollout.py \
        --model-path /path/to/checkpoint \
        --dataset-path /path/to/dataset \
        --output-dir ./visualizations \
        --num-episodes 5
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import torch
import tyro
from tqdm import tqdm

matplotlib.use("Agg")  # Non-interactive backend

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator


@dataclass
class VisualizationConfig:
    """Configuration for RL rollout visualization."""

    model_path: str
    """Path to the RL-finetuned model checkpoint."""

    dataset_path: str
    """Path to the dataset directory."""

    output_dir: str = "./visualizations"
    """Directory to save visualization videos."""

    data_config: str = "fourier_gr1_arms_only"
    """Data configuration used during training."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"
    """Video backend to use."""

    num_episodes: int = 5
    """Number of episodes to visualize."""

    start_episode: int = 0
    """Starting episode index."""

    fps: int = 10
    """Frames per second for output video."""

    video_width: int = 640
    """Width of the output video."""

    video_height: int = 960
    """Height of the output video (includes plot area)."""

    camera_index: int = 0
    """Which camera view to use for visualization."""


def create_value_plot(values, current_step, episode_length, fig_width=6.4, fig_height=3.2):
    """
    Create a matplotlib figure showing the value function curve.

    Args:
        values: List of value predictions so far
        current_step: Current timestep
        episode_length: Total episode length
        fig_width: Figure width in inches
        fig_height: Figure height in inches

    Returns:
        Image array (H, W, 3) in uint8 format
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    steps = list(range(len(values)))
    ax.plot(steps, values, "b-", linewidth=2, label="Value Function")
    ax.scatter([current_step], [values[-1]], c="r", s=100, zorder=5, label="Current")

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Predicted Value Function", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # Set x-axis limits to show full episode
    ax.set_xlim(0, episode_length - 1)

    # Set y-axis limits based on expected value range
    if len(values) > 1:
        y_min = min(-1.0, min(values) - 0.1)
        y_max = max(0.0, max(values) + 0.1)
    else:
        y_min, y_max = -1.0, 0.0
    ax.set_ylim(y_min, y_max)

    # Add zero line
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Convert figure to image
    fig.canvas.draw()
    # Use buffer_rgba() for newer matplotlib versions
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    # Convert RGBA to RGB
    img = img[:, :, :3]

    plt.close(fig)
    return img


def resize_frame(frame, target_width):
    """Resize frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    target_height = int(h * target_width / w)
    return cv2.resize(frame, (target_width, target_height))


def combine_frames(video_frame, plot_frame, target_width):
    """
    Combine video frame (top) and plot frame (bottom) into single frame.

    Args:
        video_frame: RGB frame from video
        plot_frame: RGB frame from matplotlib plot
        target_width: Target width for output

    Returns:
        Combined frame (H, W, 3)
    """
    # Resize both to same width
    video_frame = resize_frame(video_frame, target_width)
    plot_frame = resize_frame(plot_frame, target_width)

    # Stack vertically
    combined = np.vstack([video_frame, plot_frame])
    return combined


def visualize_episode(model, dataset, episode_idx, config, data_collator):
    """
    Generate visualization for a single episode.

    Args:
        model: GR00T model with value head
        dataset: LeRobot dataset
        episode_idx: Index of episode to visualize
        config: Visualization configuration
        data_collator: Data collator for batching (same as training)

    Returns:
        Path to saved video file
    """
    # Get episode info
    trajectory_id = dataset.trajectory_ids[episode_idx]
    episode_length = dataset.trajectory_lengths[episode_idx]

    print(
        f"\nProcessing episode {episode_idx} (trajectory {trajectory_id}): {episode_length} steps"
    )

    # Prepare output
    output_path = Path(config.output_dir) / f"episode_{episode_idx:04d}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Store frames and values
    frames = []
    values = []

    # Calculate starting index for this episode
    episode_start_idx = sum(dataset.trajectory_lengths[:episode_idx])

    # Process each timestep
    for step in tqdm(range(episode_length), desc=f"Episode {episode_idx}"):
        # Get data for this step (already transformed) for model inference
        step_data = dataset[episode_start_idx + step]

        # Get the raw video data before transforms for visualization
        trajectory_id = dataset.trajectory_ids[episode_idx]
        raw_step_data = dataset.get_step_data(trajectory_id, step)

        # Get the first video key and extract the first frame
        video_keys = [k for k in raw_step_data.keys() if k.startswith("video.")]
        if video_keys:
            # Take the first camera's first frame
            # Shape after get_step_data: (n_frames, H, W, C)
            video_frame = raw_step_data[video_keys[config.camera_index]][0]  # First frame
        else:
            raise ValueError(f"No video keys found in raw data: {raw_step_data.keys()}")

        # Prepare input for model inference
        # IMPORTANT: Use the same collator as training to ensure data alignment
        # step_data contains: eagle_content (with image_inputs, NOT video_inputs),
        # state, action, reward, value (from enable_rl=True)
        # The collator processes eagle_content into eagle_input_ids, eagle_pixel_values, etc.

        # DEBUG: Check value data on first step
        if step == 0 and episode_idx == 0:
            print(f"\n[DEBUG] step_data keys: {list(step_data.keys())}")
            if "value" in step_data:
                print(
                    f"[DEBUG] Value in step_data: shape={step_data['value'].shape}, "
                    f"range=[{step_data['value'].min():.3f}, {step_data['value'].max():.3f}], "
                    f"first 5={step_data['value'][:5]}"
                )
            else:
                print("[DEBUG] WARNING: No 'value' in step_data!")

        batched_data = data_collator([step_data])

        # DEBUG: Check batched value data on first step
        if step == 0 and episode_idx == 0:
            print(f"[DEBUG] batched_data keys: {list(batched_data.keys())}")
            if "value" in batched_data:
                print(
                    f"[DEBUG] Value in batched_data: shape={batched_data['value'].shape}, "
                    f"dtype={batched_data['value'].dtype}, "
                    f"values={batched_data['value'][0]}"
                )
            else:
                print("[DEBUG] WARNING: No 'value' in batched_data!")

        # Get value prediction only (skip action generation)
        # Follow the same flow as training: prepare_input -> backbone -> process_backbone_output -> value_head
        with torch.no_grad():
            backbone_inputs, _ = model.prepare_input(batched_data)
            backbone_outputs = model.backbone(backbone_inputs)

            # Debug: print dtypes on first step
            if step == 0:
                print(
                    f"  Raw backbone features dtype: {backbone_outputs['backbone_features'].dtype}"
                )
                print(
                    f"  Raw backbone features shape: {backbone_outputs['backbone_features'].shape}"
                )
                print(f"  Action head dtype: {model.action_head.dtype}")

            # Convert to action_head dtype BEFORE process_backbone_output
            # This is needed because vlln expects bfloat16
            backbone_outputs["backbone_features"] = backbone_outputs["backbone_features"].to(
                dtype=model.action_head.dtype
            )

            # IMPORTANT: Process backbone output through vlln + vl_self_attention
            # This matches the training flow where action_head.forward() modifies backbone_outputs
            backbone_outputs = model.action_head.process_backbone_output(backbone_outputs)
            backbone_features = backbone_outputs["backbone_features"]

            if step == 0:
                print(f"  Processed backbone features shape: {backbone_features.shape}")

            # Predict single state value: (batch_size, 1, 1)
            value_pred = model.value_head(backbone_features)

            if step == 0:
                print(f"  Value pred shape: {value_pred.shape}")
                print(f"  Value pred: {value_pred}")

            # Extract scalar value
            # Shape: (1, 1, 1) -> scalar
            value_scalar = value_pred[0, 0, 0].item()
            values.append(value_scalar)

        # Create value plot
        plot_frame = create_value_plot(values, step, episode_length)

        # Combine frames
        combined_frame = combine_frames(video_frame, plot_frame, config.video_width)
        frames.append(combined_frame)

    # Save video using FFMpegWriter
    print(f"Saving video with {len(frames)} frames...")
    h, w = frames[0].shape[:2]

    # Create a figure and axis for the animation
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    im = ax.imshow(frames[0])

    writer = FFMpegWriter(
        fps=config.fps,
        bitrate=2000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    with writer.saving(fig, str(output_path), dpi=100):
        for frame in frames:
            im.set_data(frame)
            writer.grab_frame()

    plt.close(fig)

    print(f"Saved video to {output_path}")
    print(f"Value range: [{min(values):.3f}, {max(values):.3f}]")
    print(f"Mean value: {np.mean(values):.3f}")

    return output_path


def main(config: VisualizationConfig):
    """Main visualization function."""

    print("=" * 80)
    print("GR00T RL Rollout Visualization")
    print("=" * 80)
    print(f"Model: {config.model_path}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print("=" * 80)

    # Load data config
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Load dataset (with RL mode to match training setup)
    print("\nLoading dataset...")
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        enable_rl=True,  # Enable RL mode to match training configuration
    )

    num_episodes = len(dataset.trajectory_ids)
    print(f"Dataset contains {num_episodes} episodes")

    # Load model with value head
    print("\nLoading model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_value_head=False,  # Freeze for inference
        enable_rl=True,  # Enable to use value head
    )
    model.eval()
    model.cuda()

    # Ensure value_head is in the same dtype as the backbone (bfloat16)
    if model.value_head is not None:
        model.value_head = model.value_head.to(dtype=model.action_head.dtype)

    print(f"Model loaded with value head")
    print(f"Device: {model.device}")
    print(f"Backbone dtype: {next(model.backbone.parameters()).dtype}")
    print(f"Action head dtype: {model.action_head.dtype}")
    if model.value_head is not None:
        print(f"Value head dtype: {next(model.value_head.parameters()).dtype}")

    # Determine which episodes to visualize
    end_episode = min(config.start_episode + config.num_episodes, num_episodes)
    episodes_to_viz = range(config.start_episode, end_episode)

    print(f"\nVisualizing episodes {config.start_episode} to {end_episode-1}")

    # Initialize data collator (same as training)
    print("\nInitializing data collator...")
    data_collator = DefaultDataCollator()

    # Visualize each episode
    output_paths = []
    for ep_idx in episodes_to_viz:
        try:
            output_path = visualize_episode(model, dataset, ep_idx, config, data_collator)
            output_paths.append(output_path)
        except Exception as e:
            print(f"Error processing episode {ep_idx}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print(f"Visualization complete!")
    print(f"Generated {len(output_paths)} videos in {config.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    config = tyro.cli(VisualizationConfig)
    main(config)
