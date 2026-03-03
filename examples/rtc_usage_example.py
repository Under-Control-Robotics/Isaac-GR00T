#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example usage of Real-Time Chunking (RTC) with Gr00tPolicy.

This script demonstrates how to wrap a Gr00tPolicy with RealTimeChunkingPolicy
to enable real-time execution with inference delays.
"""

import argparse
import time

import torch

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.RTC_gr00t import RealTimeChunkingPolicy


def main():
    parser = argparse.ArgumentParser(description="RTC Gr00t Policy Example")
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/GR00T-N1.5-3B",
        help="Path to the GR00T model",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="ucr_wblm_moby_history",
        help="Data configuration name",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        default=4,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--control_dt_ms",
        type=float,
        default=20.0,
        help="Control loop period in milliseconds",
    )
    parser.add_argument(
        "--fixed_delay_ms",
        type=float,
        default=80.0,
        help="Expected inference delay in milliseconds",
    )
    parser.add_argument(
        "--s_min",
        type=int,
        default=8,
        help="Minimum execution horizon",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Maximum guidance weight for ΠGDM",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data config
    print(f"Loading data config: {args.data_config}")
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    # Create base policy
    print(f"Loading model: {args.model_path}")
    base_policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        modality_config=modality_config,
        modality_transform=modality_transform,
        denoising_steps=args.denoising_steps,
        device=device,
    )

    # Get action horizon
    H = len(base_policy.modality_config["action"].delta_indices)
    d = int(args.fixed_delay_ms / args.control_dt_ms)

    print(f"\nRTC Configuration:")
    print(f"  Prediction horizon (H): {H}")
    print(f"  Control timestep (Δt): {args.control_dt_ms}ms")
    print(f"  Inference delay (δ): {args.fixed_delay_ms}ms")
    print(f"  Delay in timesteps (d): {d}")
    print(f"  Minimum execution horizon (s_min): {args.s_min}")
    print(f"  Maximum guidance weight (β): {args.beta}")
    print(f"  Constraint check: d ≤ s_min ≤ H - d => {d} ≤ {args.s_min} ≤ {H - d}")

    # Wrap with RTC
    print(f"\nCreating RealTimeChunkingPolicy...")
    rtc_policy = RealTimeChunkingPolicy(
        policy=base_policy,
        control_dt_ms=args.control_dt_ms,
        fixed_delay_ms=args.fixed_delay_ms,
        s_min=args.s_min,
        beta=args.beta,
    )

    print("\nRealTimeChunkingPolicy created successfully!")
    print(f"Background inference thread started.")

    # Example: Simulate a control loop
    print("\n" + "=" * 60)
    print("Simulating control loop (press Ctrl+C to stop)")
    print("=" * 60)

    try:
        # Create dummy observation (you'd get this from your robot/environment)
        # For demonstration, we just create random data matching the expected format
        dummy_obs = create_dummy_observation(base_policy)

        control_hz = 1000.0 / args.control_dt_ms
        control_period = args.control_dt_ms / 1000.0  # Convert to seconds

        step = 0
        while True:
            start_time = time.time()

            # Get action from RTC policy
            action = rtc_policy.get_action(dummy_obs)

            # Print action info (every 10 steps to avoid spam)
            if step % 10 == 0:
                action_value = action["action"]
                print(
                    f"Step {step:4d}: Action shape={action_value.shape}, "
                    f"norm={float(torch.tensor(action_value).norm()):.4f}"
                )

            # Simulate control loop timing
            elapsed = time.time() - start_time
            sleep_time = max(0, control_period - elapsed)
            time.sleep(sleep_time)

            step += 1

    except KeyboardInterrupt:
        print("\n\nShutting down...")

    finally:
        # Clean up
        rtc_policy.close()
        print("RTC policy closed successfully.")


def create_dummy_observation(policy):
    """
    Create a dummy observation matching the policy's expected format.

    In a real application, this would come from your robot/environment.
    """
    modality_config = policy.modality_config
    obs = {}

    # Create dummy video observation
    if "video" in modality_config:
        video_config = modality_config["video"]
        T = len(video_config.delta_indices)
        H, W = 224, 224  # Standard image size
        C = 3

        # Create random video frames
        import numpy as np
        obs["video"] = np.random.randint(0, 255, size=(T, H, W, C), dtype=np.uint8)

    # Create dummy state observation
    if "state" in modality_config:
        state_config = modality_config["state"]
        T = len(state_config.delta_indices)
        D = state_config.shape[-1] if hasattr(state_config, "shape") else 14  # Default dim

        # Create random state
        import numpy as np
        obs["state"] = np.random.randn(T, D).astype(np.float32)

    return obs


if __name__ == "__main__":
    main()
