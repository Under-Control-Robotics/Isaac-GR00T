#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluation script to compare GR00T policy performance with and without RTC.

This script measures:
1. Inference latency
2. Action continuity (L2 distance between consecutive chunks)
3. Total execution time with realistic timing simulation

Usage:
    python deployment_scripts/evaluate_rtc.py \
        --model_path 0301-pipe-0416/checkpoint-12000/ \
        --embodiment-tag new_embodiment \
        --data-config ucr_wblm_moby_history \
        --trt-engine-path 0301-pipe-0416/gr00t_engine/ \
        --num-rollouts 10
"""

import time
import numpy as np
import tyro
from dataclasses import dataclass
from typing import Literal
import torch
import matplotlib.pyplot as plt

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from trt_model_forward import setup_tensorrt_engines
from trt_rtc_forward import setup_tensorrt_engines_with_rtc


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    trt_engine_path: str = "gr00t_engine"
    """Path to the TensorRT engine."""

    num_rollouts: int = 10
    """Number of rollouts to evaluate."""

    action_chunk_size: int = 16
    """Action chunk size (H)."""

    rtc_delay: int = 4
    """RTC inference delay in timesteps (d)."""

    rtc_execution_horizon: int = 8
    """RTC execution horizon in timesteps (s)."""

    control_frequency: float = 50.0
    """Control frequency in Hz."""

    visualize: bool = True
    """Generate visualization plots."""


class PolicyEvaluator:
    """Evaluates policy performance with realistic timing simulation."""

    def __init__(self, policy, config: EvalConfig):
        self.policy = policy
        self.config = config
        self.dt = 1.0 / config.control_frequency  # Control period in seconds

    def create_dummy_observation(self, modality_config):
        """Create a dummy observation for testing."""
        obs = {}

        # Video modality
        if "video" in modality_config:
            video_config = modality_config["video"]
            T = len(video_config.delta_indices)
            obs["video.ego_view"] = np.random.randint(0, 256, (T, 424, 480, 3), dtype=np.uint8)

        # State modality
        if "state" in modality_config:
            state_config = modality_config["state"]
            T = len(state_config.delta_indices)
            # Assume state dimension from modality keys
            obs["state.state"] = np.random.randn(T, 31).astype(np.float32)

        # Language modality
        obs["annotation.human.action.task_description"] = ["test task"]

        return obs

    def measure_inference_time(self, num_samples=20):
        """Measure average inference time."""
        print(f"\n{'='*60}")
        print("Measuring inference time...")
        print(f"{'='*60}")

        # Warmup
        obs = self.create_dummy_observation(self.policy.modality_config)
        for _ in range(3):
            _ = self.policy.get_action(obs)

        # Measure
        times = []
        for i in range(num_samples):
            obs = self.create_dummy_observation(self.policy.modality_config)
            start = time.time()
            _ = self.policy.get_action(obs)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Sample {i+1}/{num_samples}: {elapsed*1000:.2f}ms")

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        print(f"\nInference time statistics:")
        print(f"  Mean: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
        print(f"  Min:  {min_time*1000:.2f}ms")
        print(f"  Max:  {max_time*1000:.2f}ms")
        print(f"  Control timesteps @ {self.config.control_frequency}Hz: {avg_time/self.dt:.1f} steps")

        return {
            'mean': avg_time,
            'std': std_time,
            'min': min_time,
            'max': max_time,
            'times': times,
        }

    def measure_action_continuity(self, num_rollouts=10):
        """Measure action continuity between consecutive chunks."""
        print(f"\n{'='*60}")
        print(f"Measuring action continuity over {num_rollouts} rollouts...")
        print(f"{'='*60}")

        discontinuities = []
        all_actions = []

        for rollout in range(num_rollouts):
            prev_action_chunk = None
            rollout_actions = []

            # Simulate a rollout
            num_queries = 5  # Query 5 times per rollout
            for query_idx in range(num_queries):
                obs = self.create_dummy_observation(self.policy.modality_config)
                action = self.policy.get_action(obs)

                # Extract action array (assuming single modality for simplicity)
                action_array = action['action.action']  # Shape: (H, action_dim)
                rollout_actions.append(action_array)

                # Measure discontinuity with previous chunk
                if prev_action_chunk is not None:
                    # Compare the first action of current chunk with last action of previous chunk
                    discontinuity = np.linalg.norm(action_array[0] - prev_action_chunk[-1])
                    discontinuities.append(discontinuity)

                prev_action_chunk = action_array

            all_actions.append(rollout_actions)

        avg_discontinuity = np.mean(discontinuities)
        std_discontinuity = np.std(discontinuities)

        print(f"\nAction discontinuity statistics:")
        print(f"  Mean L2 norm: {avg_discontinuity:.4f} ± {std_discontinuity:.4f}")
        print(f"  Min:  {np.min(discontinuities):.4f}")
        print(f"  Max:  {np.max(discontinuities):.4f}")

        return {
            'mean': avg_discontinuity,
            'std': std_discontinuity,
            'discontinuities': discontinuities,
            'all_actions': all_actions,
        }

    def simulate_execution_timeline(self, mode='standard'):
        """
        Simulate realistic execution timeline.

        Args:
            mode: 'standard' for non-RTC, 'rtc' for RTC mode
        """
        print(f"\n{'='*60}")
        print(f"Simulating execution timeline ({mode} mode)...")
        print(f"{'='*60}")

        H = self.config.action_chunk_size
        d = self.config.rtc_delay
        s = self.config.rtc_execution_horizon

        # Measure actual inference time
        obs = self.create_dummy_observation(self.policy.modality_config)
        start = time.time()
        _ = self.policy.get_action(obs)
        inference_time = time.time() - start

        print(f"  Inference time: {inference_time*1000:.2f}ms")
        print(f"  Control period: {self.dt*1000:.2f}ms ({self.config.control_frequency}Hz)")

        if mode == 'standard':
            # Standard mode: Query → Wait for inference → Execute all H actions → Repeat
            query_period = H * self.dt  # Query after executing all actions
            total_time_per_cycle = inference_time + H * self.dt
            effective_delay = inference_time  # Robot waits before starting execution

            print(f"\n  Standard (non-RTC) mode:")
            print(f"    Query period: {query_period*1000:.1f}ms (every {H} steps)")
            print(f"    Time per cycle: {total_time_per_cycle*1000:.1f}ms")
            print(f"    Effective delay: {effective_delay*1000:.1f}ms (waits for inference)")

        elif mode == 'rtc':
            # RTC mode: Query every s steps, inference overlaps with execution
            query_period = s * self.dt  # Query after executing s actions
            execution_time_until_next_query = s * self.dt

            # Check if inference finishes before we run out of actions
            if inference_time <= execution_time_until_next_query:
                effective_delay = 0  # No waiting! Next chunk ready in time
                print(f"\n  RTC mode:")
                print(f"    Query period: {query_period*1000:.1f}ms (every {s} steps)")
                print(f"    Execution time per query: {execution_time_until_next_query*1000:.1f}ms")
                print(f"    Effective delay: 0ms (inference finishes during execution!)")
                print(f"    Safety margin: {(execution_time_until_next_query - inference_time)*1000:.1f}ms")
            else:
                effective_delay = inference_time - execution_time_until_next_query
                print(f"\n  RTC mode:")
                print(f"    Query period: {query_period*1000:.1f}ms (every {s} steps)")
                print(f"    WARNING: Inference too slow! Still waiting {effective_delay*1000:.1f}ms")

        return {
            'query_period': query_period,
            'inference_time': inference_time,
            'effective_delay': effective_delay,
        }


def evaluate_policy(policy, config: EvalConfig, mode_name: str):
    """Run full evaluation on a policy."""
    print(f"\n{'#'*60}")
    print(f"# Evaluating: {mode_name}")
    print(f"{'#'*60}")

    evaluator = PolicyEvaluator(policy, config)

    # Measure inference time
    inference_stats = evaluator.measure_inference_time(num_samples=20)

    # Measure action continuity
    continuity_stats = evaluator.measure_action_continuity(num_rollouts=config.num_rollouts)

    # Simulate timeline
    timeline_mode = 'rtc' if 'RTC' in mode_name else 'standard'
    timeline_stats = evaluator.simulate_execution_timeline(mode=timeline_mode)

    return {
        'inference': inference_stats,
        'continuity': continuity_stats,
        'timeline': timeline_stats,
    }


def create_comparison_plots(results_no_rtc, results_rtc, config: EvalConfig):
    """Create visualization comparing RTC vs non-RTC."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RTC vs Non-RTC Performance Comparison', fontsize=16, fontweight='bold')

    # 1. Inference time comparison
    ax = axes[0, 0]
    labels = ['Non-RTC', 'RTC']
    means = [
        results_no_rtc['inference']['mean'] * 1000,
        results_rtc['inference']['mean'] * 1000,
    ]
    stds = [
        results_no_rtc['inference']['std'] * 1000,
        results_rtc['inference']['std'] * 1000,
    ]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Latency')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)

    # 2. Action discontinuity comparison
    ax = axes[0, 1]
    means = [
        results_no_rtc['continuity']['mean'],
        results_rtc['continuity']['mean'],
    ]
    stds = [
        results_no_rtc['continuity']['std'],
        results_rtc['continuity']['std'],
    ]
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4'])
    ax.set_ylabel('L2 Distance')
    ax.set_title('Action Discontinuity (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)

    # 3. Inference time distribution
    ax = axes[1, 0]
    ax.hist(
        [t*1000 for t in results_no_rtc['inference']['times']],
        bins=15, alpha=0.6, label='Non-RTC', color='#FF6B6B'
    )
    ax.hist(
        [t*1000 for t in results_rtc['inference']['times']],
        bins=15, alpha=0.6, label='RTC', color='#4ECDC4'
    )
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Timeline comparison
    ax = axes[1, 1]
    H = config.action_chunk_size
    d = config.rtc_delay
    s = config.rtc_execution_horizon

    # Text summary
    summary = f"""
Timeline Comparison:

Non-RTC:
  • Query every {H} steps
  • Effective delay: {results_no_rtc['timeline']['effective_delay']*1000:.1f}ms
  • Total cycle: {results_no_rtc['timeline']['query_period']*1000 + results_no_rtc['timeline']['inference_time']*1000:.1f}ms

RTC (H={H}, d={d}, s={s}):
  • Query every {s} steps
  • Effective delay: {results_rtc['timeline']['effective_delay']*1000:.1f}ms
  • Speedup: {results_no_rtc['timeline']['effective_delay'] / max(results_rtc['timeline']['effective_delay'], 0.001):.2f}x

Improvement:
  • Latency reduction: {(results_no_rtc['timeline']['effective_delay'] - results_rtc['timeline']['effective_delay'])*1000:.1f}ms
  • Action smoothness: {(results_no_rtc['continuity']['mean'] - results_rtc['continuity']['mean']) / results_no_rtc['continuity']['mean'] * 100:.1f}% better
"""
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax.axis('off')

    plt.tight_layout()
    output_path = '/tmp/rtc_evaluation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*60}")
    print(f"Visualization saved to: {output_path}")
    print(f"{'='*60}")

    return output_path


def main(config: EvalConfig):
    """Main evaluation function."""
    print(f"\n{'='*60}")
    print("RTC Evaluation Script")
    print(f"{'='*60}")
    print(f"Model: {config.model_path}")
    print(f"TRT Engine: {config.trt_engine_path}")
    print(f"Embodiment: {config.embodiment_tag}")
    print(f"Denoising steps: {config.denoising_steps}")
    print(f"Action chunk size (H): {config.action_chunk_size}")
    print(f"RTC delay (d): {config.rtc_delay}")
    print(f"RTC execution horizon (s): {config.rtc_execution_horizon}")
    print(f"{'='*60}")

    # Setup data config
    data_config = DATA_CONFIG_MAP[config.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # ========== Evaluate without RTC ==========
    print("\n\n" + "="*60)
    print("PART 1: Evaluating WITHOUT RTC")
    print("="*60)

    policy_no_rtc = Gr00tPolicy(
        model_path=config.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
    )
    setup_tensorrt_engines(policy_no_rtc, config.trt_engine_path)

    results_no_rtc = evaluate_policy(policy_no_rtc, config, "Non-RTC (Standard)")

    # Free memory
    del policy_no_rtc
    torch.cuda.empty_cache()

    # ========== Evaluate with RTC ==========
    print("\n\n" + "="*60)
    print("PART 2: Evaluating WITH RTC")
    print("="*60)

    policy_rtc = Gr00tPolicy(
        model_path=config.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=config.embodiment_tag,
        denoising_steps=config.denoising_steps,
    )
    setup_tensorrt_engines_with_rtc(
        policy_rtc,
        config.trt_engine_path,
        d=config.rtc_delay,
        s=config.rtc_execution_horizon,
        enable_rtc=True,
    )

    results_rtc = evaluate_policy(policy_rtc, config, "RTC Enabled")

    # ========== Summary ==========
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n📊 Inference Time:")
    print(f"  Non-RTC: {results_no_rtc['inference']['mean']*1000:.2f}ms ± {results_no_rtc['inference']['std']*1000:.2f}ms")
    print(f"  RTC:     {results_rtc['inference']['mean']*1000:.2f}ms ± {results_rtc['inference']['std']*1000:.2f}ms")

    print("\n🎯 Action Continuity (L2 distance):")
    print(f"  Non-RTC: {results_no_rtc['continuity']['mean']:.4f} ± {results_no_rtc['continuity']['std']:.4f}")
    print(f"  RTC:     {results_rtc['continuity']['mean']:.4f} ± {results_rtc['continuity']['std']:.4f}")
    improvement = (results_no_rtc['continuity']['mean'] - results_rtc['continuity']['mean']) / results_no_rtc['continuity']['mean'] * 100
    print(f"  → {improvement:.1f}% improvement with RTC")

    print("\n⏱️  Execution Timeline:")
    print(f"  Non-RTC effective delay: {results_no_rtc['timeline']['effective_delay']*1000:.1f}ms")
    print(f"  RTC effective delay:     {results_rtc['timeline']['effective_delay']*1000:.1f}ms")
    latency_reduction = (results_no_rtc['timeline']['effective_delay'] - results_rtc['timeline']['effective_delay']) * 1000
    print(f"  → {latency_reduction:.1f}ms latency reduction")

    # Generate plots
    if config.visualize:
        plot_path = create_comparison_plots(results_no_rtc, results_rtc, config)
        print(f"\n📈 Visualization: {plot_path}")

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    config = tyro.cli(EvalConfig)
    main(config)
