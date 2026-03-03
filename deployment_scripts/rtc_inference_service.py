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
GR00T Real-Time Chunking (RTC) Inference Service

This extends the standard inference service with Real-Time Chunking support for TensorRT models.
RTC provides smooth action transitions between chunks by using weighted initialization instead of
pure noise, reducing jerky behavior and maintaining temporal consistency.

Usage:
    # RTC TensorRT Server (recommended for deployment)
    python deployment_scripts/rtc_inference_service.py \\
        --server \\
        --model_path 0301-pipe-0416/checkpoint-12000/ \\
        --embodiment-tag new_embodiment \\
        --data-config ucr_wblm_moby_history \\
        --denoising-steps 4 \\
        --inference_mode tensorrt \\
        --trt-engine-path 0301-pipe-0416/gr00t_engine/ \\
        --use_rtc \\
        --rtc_control_dt_ms 20.0 \\
        --rtc_fixed_delay_ms 80.0 \\
        --rtc_execution_horizon 8

    # Standard PyTorch Server (no RTC, for comparison)
    python deployment_scripts/rtc_inference_service.py \\
        --server \\
        --model_path 0301-pipe-0416/checkpoint-12000/ \\
        --embodiment-tag new_embodiment \\
        --data-config ucr_wblm_moby_history \\
        --denoising-steps 4 \\
        --inference_mode pytorch

Key Parameters for RTC:
    - H (action_horizon): Set by data_config (e.g., 16 for ucr_wblm_moby_history)
    - d (inference_delay): ⌊rtc_fixed_delay_ms / rtc_control_dt_ms⌋ (e.g., ⌊80/20⌋ = 4)
    - s (execution_horizon): rtc_execution_horizon (e.g., 8)
    - Constraint: d ≤ s ≤ H - d

Scheduling:
    For 50Hz control (Δt=20ms), 80ms inference delay (d=4), H=16, s=8:
    - Start inference s-d=4 timesteps after chunk execution begins
    - New chunk ready exactly when needed at timestep s=8
    - Smooth transitions via weighted initialization: W*prev_chunk + (1-W)*noise
"""

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy


@dataclass
class RTCArgsConfig:
    """Command line arguments for the RTC inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    http_server: bool = False
    """Whether to run it as HTTP server. Default is ZMQ server."""

    inference_mode: Literal["pytorch", "tensorrt"] = "pytorch"
    """Inference mode: 'pytorch' for PyTorch inference, 'tensorrt' for TensorRT inference."""

    trt_engine_path: str = "gr00t_engine"
    """Path to the TensorRT engine. Used only in 'tensorrt' inference mode."""

    # ============================================================================
    # Real-Time Chunking (RTC) Parameters
    # ============================================================================

    use_rtc: bool = False
    """Enable Real-Time Chunking for smoother action transitions. Only works with tensorrt mode."""

    rtc_control_dt_ms: float = 20.0
    """Control loop period in milliseconds. For 50Hz control, use 20ms. For 20Hz, use 50ms."""

    rtc_fixed_delay_ms: float = 80.0
    """Expected inference delay in milliseconds. Measure your actual inference latency and set this.
    For example: 80ms is typical for TensorRT on RTX 4090 with network overhead."""

    rtc_execution_horizon: int = 8
    """Execution horizon (s parameter) - number of actions executed before starting next inference.
    Must satisfy: d ≤ s ≤ H - d, where d = ⌊rtc_fixed_delay_ms / rtc_control_dt_ms⌋

    Example for H=16, d=4: valid range is [4, 12], recommended s=8 for balance.
    - Larger s: smoother but less reactive
    - Smaller s: more reactive but may have slight discontinuities
    """

    rtc_use_async: bool = False
    """Use asynchronous inference in background thread.
    Server mode should typically use False (synchronous).
    Client mode can use True for better parallelism."""

    rtc_return_full_chunk: bool = True
    """Return full action chunks instead of single actions.
    Server should use True, client handles chunk execution."""


#####################################################################################


def _example_zmq_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example ZMQ client call to the server.
    """
    # Create a policy wrapper
    policy_client = RobotInferenceClient(host=host, port=port, api_token=api_token)

    print("Available modality config:")
    modality_configs = policy_client.get_modality_config()
    print(modality_configs.keys())

    time_start = time.time()
    action = policy_client.get_action(obs)
    print(f"Total time taken to get action from server: {time.time() - time_start:.3f} seconds")
    return action


def _example_http_client_call(obs: dict, host: str, port: int, api_token: str):
    """
    Example HTTP client call to the server.
    """
    import json_numpy

    json_numpy.patch()
    import requests

    # Send request to HTTP server
    print("Testing HTTP server...")

    time_start = time.time()
    response = requests.post(f"http://{host}:{port}/act", json={"observation": obs})
    print(f"Total time taken to get action from HTTP server: {time.time() - time_start:.3f} seconds")

    if response.status_code == 200:
        action = response.json()
        return action
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


def _validate_rtc_params(args: RTCArgsConfig, H: int):
    """
    Validate RTC parameters against constraints.

    Args:
        args: Configuration arguments
        H: Action horizon from data config

    Raises:
        ValueError: If parameters violate RTC constraints
    """
    d = int(args.rtc_fixed_delay_ms / args.rtc_control_dt_ms)
    s = args.rtc_execution_horizon

    print("\n" + "=" * 80)
    print("Real-Time Chunking (RTC) Configuration")
    print("=" * 80)
    print(f"Action Horizon (H):          {H} timesteps")
    print(f"Control Period (Δt):         {args.rtc_control_dt_ms:.1f} ms ({1000/args.rtc_control_dt_ms:.0f} Hz)")
    print(f"Inference Delay (δ):         {args.rtc_fixed_delay_ms:.1f} ms")
    print(f"Inference Delay (d):         {d} timesteps")
    print(f"Execution Horizon (s):       {s} timesteps")
    print(f"Constraint:                  d ≤ s ≤ H - d")
    print(f"Valid Range:                 [{d}, {H - d}]")
    print(f"Inference Start Offset:      s - d = {s - d} timesteps")
    print("-" * 80)

    # Validate constraint: d ≤ s ≤ H - d
    if not (d <= s <= H - d):
        print(f"❌ ERROR: Constraint violated!")
        print(f"   Current: {d} ≤ {s} ≤ {H - d}")
        print(f"   Required: d ≤ s ≤ H - d")
        print(f"\nSuggestions:")
        print(f"   - Decrease rtc_fixed_delay_ms (optimize inference)")
        print(f"   - Adjust rtc_execution_horizon to be in range [{d}, {H - d}]")
        print(f"   - Use a data config with larger action horizon")
        print("=" * 80 + "\n")
        raise ValueError(
            f"RTC constraint violated: d={d} ≤ s={s} ≤ H-d={H-d} must hold. "
            f"Valid range for s is [{d}, {H - d}]"
        )

    # Warnings for suboptimal configurations
    if s < 2 * d:
        print(f"⚠️  WARNING: Small execution horizon (s={s} < 2*d={2*d})")
        print(f"   This may reduce smoothness benefits. Consider increasing s.")

    if s > H - 2 * d:
        print(f"⚠️  WARNING: Large execution horizon (s={s} > H-2*d={H - 2*d})")
        print(f"   This may reduce reactivity. Consider decreasing s.")

    # Optimal suggestion
    optimal_s = (d + (H - d)) // 2
    if abs(s - optimal_s) > 2:
        print(f"💡 TIP: For balanced smoothness/reactivity, try s={optimal_s}")

    print(f"✅ RTC parameters validated successfully!")
    print("=" * 80 + "\n")


def main(args: RTCArgsConfig):
    if args.server:
        # Validate RTC mode compatibility
        if args.use_rtc and args.inference_mode != "tensorrt":
            raise ValueError(
                "RTC mode (--use_rtc) requires TensorRT inference mode (--inference_mode tensorrt). "
                "PyTorch RTC should use the standard gr00t RealTimeChunkingPolicy wrapper."
            )

        # Create data config and get modality config
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # Get action horizon for RTC validation
        action_modality_config = modality_config.get("action")
        if action_modality_config is None:
            raise ValueError(
                f"Data config '{args.data_config}' does not have 'action' modality config"
            )
        H = len(action_modality_config.delta_indices)

        # Validate RTC parameters if enabled
        if args.use_rtc:
            _validate_rtc_params(args, H)

        # Create base policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # Setup inference backend
        if args.inference_mode == "tensorrt":
            if args.use_rtc:
                print("\n🚀 Setting up TensorRT with Real-Time Chunking (RTC)...")
                from trt_rtc_forward import setup_tensorrt_engines_with_rtc
                from trt_rtc_policy import TensorRTRealTimeChunkingPolicy

                # Setup TensorRT engines with RTC-enabled forward functions
                setup_tensorrt_engines_with_rtc(policy, args.trt_engine_path)

                # Wrap policy with RTC wrapper
                policy = TensorRTRealTimeChunkingPolicy(
                    policy=policy,
                    control_dt_ms=args.rtc_control_dt_ms,
                    fixed_delay_ms=args.rtc_fixed_delay_ms,
                    s_min=args.rtc_execution_horizon,
                    return_full_chunk=args.rtc_return_full_chunk,
                    use_async=args.rtc_use_async,
                )
                print("✅ TensorRT RTC setup complete!\n")
            else:
                print("\n🚀 Setting up standard TensorRT (no RTC)...")
                from trt_model_forward import setup_tensorrt_engines

                setup_tensorrt_engines(policy, args.trt_engine_path)
                print("✅ TensorRT setup complete!\n")
        else:
            print("\n🚀 Using PyTorch inference mode (no TensorRT acceleration)\n")

        # Start the server
        print(f"Starting {'HTTP' if args.http_server else 'ZMQ'} server on {args.host}:{args.port}...")
        if args.http_server:
            from gr00t.eval.http_server import HTTPInferenceServer

            server = HTTPInferenceServer(
                policy, port=args.port, host=args.host, api_token=args.api_token
            )
            server.run()
        else:
            server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
            server.run()

    elif args.client:
        # Test client mode
        print("\n🧪 Running in test client mode...")
        print("Sending random observation to server for testing...\n")

        # Create random observation matching expected format
        # This should match your actual observation structure from vla_inference_client
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.state": np.random.rand(1, 29).astype(np.float32),
            "annotation.human.action.task_description": ["pick up the object"],
        }

        if args.http_server:
            action = _example_http_client_call(obs, args.host, args.port, args.api_token)
        else:
            action = _example_zmq_client_call(obs, args.host, args.port, args.api_token)

        print("\nReceived action:")
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(RTCArgsConfig)
    main(config)
