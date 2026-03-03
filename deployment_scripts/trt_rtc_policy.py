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
Real-Time Chunking Policy Wrapper for TensorRT GR00T models.

This provides a wrapper around TensorRT GR00T policy that implements Real-Time Chunking.
Unlike the PyTorch RTC implementation which uses ΠGDM guidance, this version uses weighted
initialization to blend the previous chunk with noise, since TensorRT engines don't support autograd.
"""

import threading
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


class TensorRTRealTimeChunkingPolicy:
    """
    Wrapper around TensorRT GR00T policy that implements Real-Time Chunking.

    Unlike the PyTorch RTC implementation which uses ΠGDM guidance, this version
    uses weighted initialization to blend the previous chunk with noise, since
    TensorRT engines don't support autograd.

    Args:
        policy: GR00T policy with TensorRT engines loaded
        control_dt_ms: Control loop period in milliseconds (e.g., 20ms for 50Hz)
        fixed_delay_ms: Expected inference delay in milliseconds (e.g., 80ms)
        s_min: Minimum execution horizon in timesteps. Must satisfy: d ≤ s_min ≤ H - d
        initial_chunk: Optional initial action chunk
        return_full_chunk: If True, get_action() returns full chunks instead of single actions
        use_async: If True, run inference in background thread (default: True)
    """

    def __init__(
        self,
        policy,
        control_dt_ms: float = 20.0,
        fixed_delay_ms: float = 80.0,
        s_min: int = 8,
        initial_chunk: Optional[Dict[str, np.ndarray]] = None,
        return_full_chunk: bool = False,
        use_async: bool = True,
    ):
        self.policy = policy
        self.control_dt_ms = control_dt_ms
        self.fixed_delay_ms = fixed_delay_ms
        self.return_full_chunk = return_full_chunk
        self.use_async = use_async

        # Compute delay in timesteps: d = ⌊δ/∆t⌋
        self.d = int(fixed_delay_ms / control_dt_ms)

        # Get prediction horizon from the model
        self.H = len(policy.modality_config["action"].delta_indices)

        # Validate s_min
        self.s_min = s_min
        if not (self.d <= self.s_min <= self.H - self.d):
            raise ValueError(
                f"s_min={s_min} must satisfy d <= s_min <= H - d, "
                f"where d={self.d}, H={self.H}, so {self.d} <= s_min <= {self.H - self.d}"
            )

        print(f"TensorRT RTC initialized: H={self.H}, d={self.d}, s_min={self.s_min}")

        # Shared state protected by mutex
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.t = 0  # Index into current chunk
        self.current_chunk = initial_chunk
        self.latest_obs = None
        self.running = True
        self.chunk_ready = False

        # Start background inference thread if async is enabled
        if self.use_async:
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()
        else:
            self.inference_thread = None

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the next action for the current timestep.

        Args:
            observation: Current observation dict

        Returns:
            Action dict with either a single action or full chunk
        """
        if self.return_full_chunk:
            return self._get_full_chunk(observation)
        else:
            return self._get_single_action(observation)

    def _get_full_chunk(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Get a full action chunk (for environments that execute chunks)."""
        with self.lock:
            self.latest_obs = observation

            # First call: generate initial chunk
            if self.current_chunk is None:
                self.lock.release()
                try:
                    print("TensorRT RTC: Generating initial chunk...")
                    self.current_chunk = self.policy.get_action(observation)
                finally:
                    self.lock.acquire()

                self.t = 0
                return self.current_chunk

            # Wait if background thread is still generating
            while self.chunk_ready:
                print("TensorRT RTC: Waiting for background inference...")
                self.condition.wait(timeout=5.0)
                if not self.chunk_ready:
                    break

            result = self.current_chunk
            self.t += self.s_min
            self.condition.notify()

            return result

    def _get_single_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Get a single action at the current timestep."""
        with self.lock:
            self.latest_obs = observation

            # First call: generate initial chunk synchronously
            if self.current_chunk is None:
                print("TensorRT RTC: Generating initial chunk...")
                self.current_chunk = self.policy.get_action(observation)
                self.t = 0

            # Extract action at current timestep
            action = {}
            for key, value in self.current_chunk.items():
                if len(value.shape) >= 2:  # Has time dimension
                    action[key] = value[self.t]
                else:
                    action[key] = value

            self.t += 1
            self.condition.notify()

            return action

    def _inference_loop(self):
        """Background inference loop for async generation."""
        while self.running:
            with self.lock:
                # Wait until s_min actions have been executed
                while self.t < self.s_min and self.running:
                    self.condition.wait()

                if not self.running:
                    break

                self.chunk_ready = True
                s = self.t  # Actual execution horizon

                # Extract overlapping portion of previous chunk
                prev_chunk_overlap = None
                if self.current_chunk is not None:
                    # Get actions from index s onwards (normalized action space)
                    if "action" in self.current_chunk:
                        prev_actions = self.current_chunk["action"]
                        if len(prev_actions.shape) >= 2:  # (H, action_dim)
                            prev_chunk_overlap = prev_actions[s:]  # (H-s, action_dim)
                        elif len(prev_actions.shape) == 3:  # (1, H, action_dim)
                            prev_chunk_overlap = prev_actions[0, s:]

                obs_copy = self.latest_obs.copy() if self.latest_obs is not None else None
                d = self.d

            # Release lock during inference
            if obs_copy is None:
                with self.lock:
                    self.chunk_ready = False
                continue

            try:
                print(f"TensorRT RTC: Generating new chunk (d={d}, s={s})...")
                new_chunk = self._guided_inference(obs_copy, prev_chunk_overlap, d, s)
                print("TensorRT RTC: New chunk ready!")
            except Exception as e:
                print(f"TensorRT RTC: Error in inference: {e}")
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.chunk_ready = False
                continue

            with self.lock:
                self.current_chunk = new_chunk
                self.t = self.t - s  # Reset index into new chunk
                self.chunk_ready = False
                self.condition.notify_all()

    def _guided_inference(
        self,
        observation: Dict[str, Any],
        prev_chunk_overlap: Optional[np.ndarray],
        d: int,
        s: int,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a new action chunk using RTC weighted initialization.

        Args:
            observation: Current observation
            prev_chunk_overlap: Overlapping actions from previous chunk (H-s, action_dim) in normalized space
            d: Inference delay in timesteps
            s: Execution horizon

        Returns:
            New action chunk dict
        """
        # Prepare the previous chunk as a tensor for the TensorRT forward
        # The previous chunk overlap needs to be right-padded to length H
        prev_chunk_tensor = None
        if prev_chunk_overlap is not None:
            # prev_chunk_overlap is (H-s, action_dim) in normalized action space
            # We need to pad it to (H, action_dim) with zeros
            H = self.H
            action_dim = prev_chunk_overlap.shape[-1]

            prev_chunk_padded = np.zeros((H, action_dim), dtype=np.float32)
            prev_chunk_padded[:len(prev_chunk_overlap)] = prev_chunk_overlap

            # Convert to torch tensor
            prev_chunk_tensor = torch.from_numpy(prev_chunk_padded)

        # Call the policy's get_action with RTC parameters
        # This requires modifying how we call the action head
        model = self.policy.model
        device = model.device

        # Prepare observation
        obs_copy = observation.copy()
        is_batch = self.policy._check_state_is_batched(obs_copy)
        if not is_batch:
            from gr00t.model.policy import unsqueeze_dict_values
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        # Apply transforms (normalization)
        normalized_input = self.policy.apply_transforms(obs_copy)

        # Convert to torch tensors
        for k, v in normalized_input.items():
            if isinstance(v, np.ndarray):
                normalized_input[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                normalized_input[k] = v.to(device)

        # Run backbone
        with torch.inference_mode():
            backbone_output = model.backbone(normalized_input)

            # Prepare action input
            action_input_data = {
                "state": normalized_input.get(
                    "state",
                    torch.zeros((1, 1, model.action_head.config.max_state_dim), device=device)
                ),
                "embodiment_id": torch.tensor([0], device=device),
            }
            action_input = BatchFeature(data=action_input_data)

            # Call action head with RTC parameters
            # The action head's get_action method now accepts prev_chunk, d, s
            action_output = model.action_head.get_action(
                backbone_output,
                action_input,
                prev_chunk=prev_chunk_tensor,
                d=d,
                s=s,
            )

        # Unnormalize actions
        action_pred = action_output["action_pred"].float()
        unnormalized_action = self.policy.unapply_transforms({"action": action_pred.cpu()})

        # Remove batch dimension if input wasn't batched
        if not is_batch:
            from gr00t.model.policy import squeeze_dict_values
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        return unnormalized_action

    def close(self):
        """Stop background inference thread."""
        with self.lock:
            self.running = False
            self.condition.notify_all()

        if self.inference_thread is not None and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass

    @property
    def modality_config(self):
        """Forward modality_config from wrapped policy."""
        return self.policy.modality_config

    @property
    def modality_transform(self):
        """Forward modality_transform from wrapped policy."""
        return self.policy.modality_transform

    def get_modality_config(self):
        """Forward get_modality_config from wrapped policy."""
        return self.policy.get_modality_config()
