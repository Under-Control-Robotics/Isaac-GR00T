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

    # Initialize video writer (will set after first frame)
    video_writer = None

    # Store values for plotting
    values = []

    # Calculate starting index for this episode
    episode_start_idx = sum(dataset.trajectory_lengths[:episode_idx])

    # Process each timestep
    for step in tqdm(range(episode_length), desc=f"Episode {episode_idx}"):
        # Get data for this step (already transformed)
        step_data = dataset[episode_start_idx + step]

        # Extract raw video frames for visualization
        # After transforms, video has shape (n_frames, n_cameras, H, W, C)
        # We need the first frame from the specified camera
        video_tensor = step_data["eagle_content"]["video_inputs"]
        # video_inputs is a list of videos, take first video (first timestep)
        # The shape after eagle processing is complex, so we use a simpler approach

        # Get the raw video before VLM processing
        # We need to get the original frame from the step data
        # The dataset stores video in various formats depending on transforms
        # Let's check if there's a "video" key (before VLM processing)

        # Actually, let's access the raw data before transforms
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
        # Use the same collator as training to batch a single sample
        batched_data = data_collator([step_data])

        # Get value prediction only (skip action generation)
        # Follow the same flow as training: prepare_input -> backbone -> value_head
        with torch.no_grad():
            backbone_inputs, _ = model.prepare_input(batched_data)
            backbone_outputs = model.backbone(backbone_inputs)
            backbone_features = backbone_outputs["backbone_features"]

            # Debug: print dtypes on first step
            if step == 0:
                print(f"  Backbone features dtype: {backbone_features.dtype}")
                print(f"  Backbone features shape: {backbone_features.shape}")
                print(f"  Value head weight dtype: {model.value_head.mlp[0].weight.dtype}")

            # Convert backbone_features to match value_head dtype
            value_head_dtype = model.value_head.mlp[0].weight.dtype
            backbone_features = backbone_features.to(dtype=value_head_dtype)

            # Predict values: (batch_size, seq_len, 1)
            value_pred = model.value_head(backbone_features)

            # Take the last timestep value (current state's future return)
            # Shape: (1, seq_len, 1) -> scalar
            value_scalar = value_pred[0, -1, 0].item()
            values.append(value_scalar)

        # Create value plot
        plot_frame = create_value_plot(values, step, episode_length)

        # Combine frames
        combined_frame = combine_frames(video_frame, plot_frame, config.video_width)

        # Initialize video writer on first frame
        if video_writer is None:
            h, w = combined_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(output_path), fourcc, config.fps, (w, h))

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        video_writer.write(combined_frame_bgr)

    # Release video writer
    if video_writer is not None:
        video_writer.release()

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

    # Load dataset (without RL mode since we're just visualizing)
    print("\nLoading dataset...")
    dataset = LeRobotSingleDataset(
        dataset_path=config.dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=config.video_backend,
        enable_rl=False,  # Don't need RL labels for visualization
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
