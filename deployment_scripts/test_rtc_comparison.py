#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to compare RTC vs non-RTC inference and visualize the differences.

Usage:
    python deployment_scripts/test_rtc_comparison.py \\
        --model_path 0301-pipe-0416/checkpoint-12000/ \\
        --embodiment-tag new_embodiment \\
        --data-config ucr_wblm_moby_history \\
        --trt-engine-path 0301-pipe-0416/gr00t_engine/

This will:
1. Measure actual inference latency
2. Test both RTC and non-RTC modes
3. Generate action sequences and compute smoothness metrics
4. Visualize action trajectories (if matplotlib available)
"""

import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro
import torch

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class TestConfig:
    """Configuration for RTC comparison test."""

    model_path: str
    """Path to the model checkpoint."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "ucr_wblm_moby_history"
    """Data config name."""

    trt_engine_path: str = "gr00t_engine"
    """Path to TensorRT engines."""

    num_chunks: int = 16
    """Number of chunks to generate for comparison."""

    denoising_steps: int = 4
    """Number of denoising steps."""

    control_dt_ms: float = 20.0
    """Control period in ms."""

    execution_horizon: int = 8
    """Execution horizon for RTC."""

    plot_results: bool = True
    """Whether to plot results (requires matplotlib)."""


def create_dummy_observation(modality_config):
    """Create a dummy observation for testing."""
    obs = {}

    # Video modality
    if "video" in modality_config:
        video_config = modality_config["video"]
        T = len(video_config.delta_indices)
        obs["video.ego_view"] = np.random.randint(0, 256, (T, 256, 256, 3), dtype=np.uint8)

    # State modality
    if "state" in modality_config:
        state_config = modality_config["state"]
        T = len(state_config.delta_indices)
        # Assume state dimension from modality keys
        obs["state.state"] = np.random.randn(T, 29).astype(np.float32)

    # Language modality
    obs["annotation.human.action.task_description"] = ["test task"]

    return obs


def measure_latency(policy, obs, num_trials=10):
    """Measure inference latency."""
    print(f"\n{'='*80}")
    print("Measuring Inference Latency")
    print(f"{'='*80}")

    latencies = []
    for i in range(num_trials):
        start = time.time()
        _ = policy.get_action(obs)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
        print(f"Trial {i+1}/{num_trials}: {latency:.2f} ms")

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)

    print(f"\n{'─'*80}")
    print(f"Mean Latency:    {mean_latency:.2f} ± {std_latency:.2f} ms")
    print(f"Min Latency:     {min_latency:.2f} ms")
    print(f"Max Latency:     {max_latency:.2f} ms")
    print(f"{'='*80}\n")

    return mean_latency


def compute_smoothness_metrics(actions):
    """
    Compute smoothness metrics for action trajectory.

    Args:
        actions: Array of shape (total_timesteps, action_dim)

    Returns:
        Dictionary of metrics
    """
    # Velocity (first derivative)
    velocity = np.diff(actions, axis=0)
    velocity_norm = np.linalg.norm(velocity, axis=1)

    # Acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)

    # Jerk (third derivative)
    jerk = np.diff(acceleration, axis=0)
    jerk_norm = np.linalg.norm(jerk, axis=1)

    return {
        "mean_velocity": np.mean(velocity_norm),
        "max_velocity": np.max(velocity_norm),
        "mean_acceleration": np.mean(acceleration_norm),
        "max_acceleration": np.max(acceleration_norm),
        "mean_jerk": np.mean(jerk_norm),
        "max_jerk": np.max(jerk_norm),
    }


def test_standard_mode(policy, obs, num_chunks):
    """Test standard (non-RTC) mode."""
    print(f"\n{'='*80}")
    print("Testing Standard Mode (No RTC)")
    print(f"{'='*80}\n")

    all_actions = []
    chunk_boundaries = []

    for i in range(num_chunks):
        action = policy.get_action(obs)
        action_array = action["action.action"]  # Shape: (H, action_dim)

        all_actions.append(action_array)
        chunk_boundaries.append(len(all_actions) * action_array.shape[0])

        print(f"Chunk {i+1}/{num_chunks}: Generated {action_array.shape[0]} actions")

    # Concatenate all actions
    actions = np.concatenate(all_actions, axis=0)
    print(f"\nTotal actions generated: {actions.shape[0]}")

    # Compute metrics
    metrics = compute_smoothness_metrics(actions)
    print(f"\n{'─'*40}")
    print("Smoothness Metrics:")
    print(f"{'─'*40}")
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")
    print(f"{'='*80}\n")

    return actions, chunk_boundaries, metrics


def test_rtc_mode(policy, obs, num_chunks, control_dt_ms, fixed_delay_ms, execution_horizon):
    """Test RTC mode."""
    print(f"\n{'='*80}")
    print("Testing RTC Mode")
    print(f"{'='*80}\n")

    from trt_rtc_forward import setup_tensorrt_engines_with_rtc
    from trt_rtc_policy import TensorRTRealTimeChunkingPolicy

    # Setup RTC - extract engine directory from DiT engine file path
    # Engine.file contains the full path to the engine file (e.g., "path/to/DiT.engine")
    engine_dir = os.path.dirname(policy.model.action_head.DiT_engine.file)
    setup_tensorrt_engines_with_rtc(policy, engine_dir)

    rtc_policy = TensorRTRealTimeChunkingPolicy(
        policy=policy,
        control_dt_ms=control_dt_ms,
        fixed_delay_ms=fixed_delay_ms,
        s_min=execution_horizon,
        return_full_chunk=True,
        use_async=False,
    )

    all_actions = []
    chunk_boundaries = []

    for i in range(num_chunks):
        action = rtc_policy.get_action(obs)
        action_array = action["action.action"]  # Shape: (H, action_dim)

        all_actions.append(action_array)
        chunk_boundaries.append(len(all_actions) * action_array.shape[0])

        print(f"Chunk {i+1}/{num_chunks}: Generated {action_array.shape[0]} actions")

    # Concatenate all actions
    actions = np.concatenate(all_actions, axis=0)
    print(f"\nTotal actions generated: {actions.shape[0]}")

    # Compute metrics
    metrics = compute_smoothness_metrics(actions)
    print(f"\n{'─'*40}")
    print("Smoothness Metrics:")
    print(f"{'─'*40}")
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.6f}")
    print(f"{'='*80}\n")

    rtc_policy.close()

    return actions, chunk_boundaries, metrics


def plot_comparison(
    actions_standard,
    boundaries_standard,
    actions_rtc,
    boundaries_rtc,
    metrics_standard,
    metrics_rtc,
):
    """Plot comparison between standard and RTC modes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Select first action dimension for visualization
    dim = 0

    # Plot 1: Action trajectories
    ax = axes[0]
    ax.plot(actions_standard[:, dim], label="Standard", alpha=0.7, linewidth=2)
    ax.plot(actions_rtc[:, dim], label="RTC", alpha=0.7, linewidth=2)
    for b in boundaries_standard[:-1]:
        ax.axvline(b, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(f"Action Dimension {dim}")
    ax.set_title("Action Trajectories (Red lines = chunk boundaries)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Acceleration
    ax = axes[1]
    accel_std = np.diff(np.diff(actions_standard[:, dim]))
    accel_rtc = np.diff(np.diff(actions_rtc[:, dim]))
    ax.plot(np.abs(accel_std), label="Standard", alpha=0.7)
    ax.plot(np.abs(accel_rtc), label="RTC", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Absolute Acceleration")
    ax.set_title(
        f"Acceleration Magnitude (Standard: {metrics_standard['mean_acceleration']:.4f}, "
        f"RTC: {metrics_rtc['mean_acceleration']:.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Jerk
    ax = axes[2]
    jerk_std = np.diff(accel_std)
    jerk_rtc = np.diff(accel_rtc)
    ax.plot(np.abs(jerk_std), label="Standard", alpha=0.7)
    ax.plot(np.abs(jerk_rtc), label="RTC", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Absolute Jerk")
    ax.set_title(
        f"Jerk Magnitude (Standard: {metrics_standard['mean_jerk']:.4f}, "
        f"RTC: {metrics_rtc['mean_jerk']:.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rtc_comparison.png", dpi=150)
    print("✅ Saved comparison plot to: rtc_comparison.png")
    plt.show()


def main(config: TestConfig):
    print("\n" + "=" * 80)
    print("RTC vs Standard Mode Comparison Test")
    print("=" * 80)

    # Load policy
    print("\n📦 Loading model...")
    data_config = DATA_CONFIG_MAP[config.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=config.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
    )

    # Setup TensorRT
    print("🚀 Setting up TensorRT engines...")
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy, config.trt_engine_path)
    print("✅ Model loaded successfully!\n")

    # Create dummy observation
    obs = create_dummy_observation(modality_config)
    print("📸 Created test observation")

    # Measure latency
    mean_latency = measure_latency(policy, obs, num_trials=10)

    # Compute RTC parameters
    H = len(modality_config["action"].delta_indices)
    d = int(mean_latency / config.control_dt_ms)
    s = config.execution_horizon

    print(f"RTC Parameters:")
    print(f"  H (action horizon):     {H}")
    print(f"  d (inference delay):    {d} timesteps ({mean_latency:.1f} ms)")
    print(f"  s (execution horizon):  {s}")
    print(f"  Constraint check:       {d} ≤ {s} ≤ {H-d} → ", end="")
    if d <= s <= H - d:
        print("✅ Valid")
    else:
        print(f"❌ Invalid! Adjust execution_horizon to be in [{d}, {H-d}]")
        return

    # Test standard mode
    actions_std, boundaries_std, metrics_std = test_standard_mode(
        policy, obs, config.num_chunks
    )

    # Test RTC mode
    actions_rtc, boundaries_rtc, metrics_rtc = test_rtc_mode(
        policy, obs, config.num_chunks, config.control_dt_ms, mean_latency, config.execution_horizon
    )

    # Comparison
    print(f"\n{'='*80}")
    print("Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Standard':>15} {'RTC':>15} {'Improvement':>15}")
    print(f"{'─'*80}")

    for key in metrics_std.keys():
        std_val = metrics_std[key]
        rtc_val = metrics_rtc[key]
        improvement = ((std_val - rtc_val) / std_val) * 100 if std_val > 0 else 0
        print(f"{key:<25} {std_val:>15.6f} {rtc_val:>15.6f} {improvement:>14.1f}%")

    print(f"{'='*80}\n")

    # Plot if requested
    if config.plot_results:
        plot_comparison(
            actions_std, boundaries_std, actions_rtc, boundaries_rtc, metrics_std, metrics_rtc
        )


if __name__ == "__main__":
    config = tyro.cli(TestConfig)
    main(config)
