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
Real-Time Chunking (RTC) for Action Chunking Policies with Flow Matching.

This implements the RTC algorithm from the paper, which uses inpainting-based
guidance to maintain continuity between action chunks during real-time execution
with inference delays.

Key concepts:
- d: inference delay in timesteps (e.g., 80ms / 20ms = 4 timesteps)
- H: prediction horizon (action chunk size)
- s: execution horizon (number of actions executed before starting next inference)
- Soft masking: Gradually decay guidance weight from frozen actions to new actions
- ΠGDM: Pseudoinverse Guided Diffusion/Flow Matching for inpainting
"""

import threading
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor

from gr00t.model.policy import Gr00tPolicy

from transformers.feature_extraction_utils import BatchFeature

COMPUTE_DTYPE = torch.bfloat16

class RealTimeChunkingPolicy:
    """
    Wrapper around Gr00tPolicy that implements Real-Time Chunking (RTC).

    This policy maintains continuity between action chunks by using inpainting-based
    guidance during flow matching denoising. Actions are generated in a background
    thread to handle inference delays gracefully.

    Args:
        policy: The underlying Gr00tPolicy to wrap
        control_dt_ms: Control loop period in milliseconds (e.g., 20ms for 50Hz)
        fixed_delay_ms: Expected inference delay in milliseconds (e.g., 80ms)
        s_min: Minimum execution horizon in timesteps. Must satisfy: d ≤ s_min ≤ H - d
        beta: Maximum guidance weight for ΠGDM (default: 1.0)
        initial_chunk: Optional initial action chunk. If None, will generate on first call.
        return_full_chunk: If True, get_action() returns full action chunks instead of single timesteps.
                          Background inference is disabled in this mode. (default: False)
    """

    def __init__(
        self,
        policy: Gr00tPolicy,
        control_dt_ms: float = 20.0,
        fixed_delay_ms: float = 80.0,
        s_min: int = 8,
        beta: float = 1.0,
        initial_chunk: Optional[Dict[str, np.ndarray]] = None,
        return_full_chunk: bool = False,
    ):
        self.policy = policy
        self.control_dt_ms = control_dt_ms
        self.fixed_delay_ms = fixed_delay_ms
        self.beta = beta
        self.return_full_chunk = return_full_chunk

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

        # Shared state protected by mutex
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.t = 0  # Index into current chunk (or virtual time for full chunk mode)
        self.current_chunk = initial_chunk  # Dict[str, np.ndarray] with shape (H, action_dim)
        self.latest_obs = None  # Latest observation from controller
        self.running = True  # Flag to stop inference loop
        self.chunk_ready = False  # Flag indicating if next chunk is being computed

        # Always start background inference thread for async generation
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def get_action(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the next action for the current timestep.

        This method is called by the controller at each control timestep (every ∆t).
        It returns the next action from the current chunk and updates the shared
        observation for the background inference thread.

        Args:
            observation: Current observation dict (e.g., {"video.<>": np.ndarray, "state.<>": np.ndarray})

        Returns:
            Action dict (e.g., {"action": np.ndarray with shape (action_dim,)} or full chunk if return_full_chunk=True)
        """
        # If return_full_chunk is True, return full chunks with async background generation
        if self.return_full_chunk:
            with self.lock:
                # Update latest observation for inference thread
                self.latest_obs = observation

                # First call: generate initial chunk synchronously
                if self.current_chunk is None:
                    # Release lock during inference
                    self.lock.release()
                    try:
                        print("RTC: Generating initial chunk synchronously...")
                        initial_chunk = self.policy.get_action(observation)
                    finally:
                        self.lock.acquire()

                    self.current_chunk = initial_chunk
                    self.t = 0
                    print(f"RTC: Initial chunk ready. Background thread will start generating next chunk after t >= {self.s_min}")
                    return initial_chunk

                # Wait if background thread is still generating the chunk
                # This should rarely happen if inference is faster than query rate
                while self.chunk_ready:
                    print("RTC: Waiting for background inference to complete...")
                    self.condition.wait(timeout=5.0)  # Timeout to prevent deadlock
                    if not self.chunk_ready:
                        break

                # Return the current chunk (already pre-computed by background thread)
                result = self.current_chunk

                # Simulate "executing" the chunk by advancing virtual time
                # This represents that the executor will execute this chunk locally
                self.t += self.s_min  # Advance by execution horizon

                # Notify background thread to start generating next chunk
                self.condition.notify()

                return result

        # Original per-timestep behavior
        with self.lock:
            # Update latest observation for inference thread
            self.latest_obs = observation

            # If no chunk available yet, generate one synchronously (first call only)
            if self.current_chunk is None:
                print("RTC: Generating initial chunk synchronously...")
                self.current_chunk = self.policy.get_action(observation)
                self.t = 0

            # Get action at current timestep
            action = {}
            for key, value in self.current_chunk.items():
                if len(value.shape) >= 2:  # Has time dimension
                    action[key] = value[self.t]
                else:
                    action[key] = value

            # Increment timestep counter
            self.t += 1

            # Notify inference thread that we've consumed an action
            self.condition.notify()

            return action

    def _inference_loop(self):
        """
        Background inference loop that generates action chunks.

        This runs continuously in a separate thread, waiting until enough actions
        have been executed (t >= s_min), then generating a new chunk using guided
        inference with the previous chunk as inpainting guidance.
        """
        while self.running:
            with self.lock:
                # Wait until at least s_min actions have been executed
                while self.t < self.s_min and self.running:
                    self.condition.wait()

                if not self.running:
                    break

                # Mark that we're generating a new chunk
                self.chunk_ready = True

                # Save state for inference
                s = self.t  # Execution horizon for this iteration

                # Extract the overlapping part of the previous chunk
                # Remove the first s actions that have already been executed
                if self.current_chunk is not None:
                    prev_chunk_overlap = {}
                    for key, value in self.current_chunk.items():
                        if len(value.shape) >= 2:  # Has time dimension
                            # Keep actions from index s onwards (these overlap with new chunk)
                            prev_chunk_overlap[key] = value[s:]
                        else:
                            prev_chunk_overlap[key] = value
                else:
                    prev_chunk_overlap = None

                obs_copy = self.latest_obs.copy() if self.latest_obs is not None else None
                d = self.d  # Use fixed delay

            # Release lock during inference (this can take a while)
            if obs_copy is None:
                with self.lock:
                    self.chunk_ready = False
                continue

            # Run guided inference with inpainting
            try:
                print(f"RTC: Background thread generating new chunk (d={d}, s={s})...")
                new_chunk = self._guided_inference(obs_copy, prev_chunk_overlap, d, s)
                print("RTC: New chunk ready!")
            except Exception as e:
                print(f"RTC: Error in guided inference: {e}")
                import traceback
                traceback.print_exc()
                with self.lock:
                    self.chunk_ready = False
                continue

            # Acquire lock to update shared state
            with self.lock:
                self.current_chunk = new_chunk
                self.t = self.t - s  # Reset t to index into new chunk
                self.chunk_ready = False  # Mark chunk as ready for use
                self.condition.notify_all()  # Wake up any waiting get_action calls

                # Note: we observe the actual delay by checking self.t here
                # but since we use fixed delay, we don't update the delay estimate

    def _guided_inference(
        self,
        observation: Dict[str, Any],
        prev_chunk_overlap: Optional[Dict[str, np.ndarray]],
        d: int,
        s: int,
    ) -> Dict[str, np.ndarray]:
        """
        Generate a new action chunk using ΠGDM-guided flow matching with soft masking.

        This implements Algorithm 1's GUIDEDINFERENCE function, which uses inpainting
        to ensure the new chunk is consistent with the overlapping actions from the
        previous chunk.

        Args:
            observation: Current observation
            prev_chunk_overlap: Overlapping actions from previous chunk (H-s actions)
            d: Inference delay in timesteps
            s: Execution horizon (number of actions executed since last inference)

        Returns:
            New action chunk dict
        """
        # Compute soft mask weights (Equation 5)
        W = self._compute_soft_mask(d, s)  # Shape: (H,)

        # Get model components
        model = self.policy.model
        action_head = model.action_head
        device = model.device

        # Prepare observation (apply transforms, move to device)
        obs_copy = observation.copy()
        is_batch = self.policy._check_state_is_batched(obs_copy)
        if not is_batch:
            from gr00t.model.policy import unsqueeze_dict_values
            obs_copy = unsqueeze_dict_values(obs_copy)

        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                obs_copy[k] = np.array(v)

        normalized_input = self.policy.apply_transforms(obs_copy)

        # Convert normalized input to torch tensors and move to device
        for k, v in normalized_input.items():
            if isinstance(v, np.ndarray):
                normalized_input[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, torch.Tensor):
                normalized_input[k] = v.to(device)

        # Process backbone output (vision + language)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            backbone_output = model.backbone(normalized_input)
            backbone_output = action_head.process_backbone_output(backbone_output)

        # Prepare action input
        action_input_data = {
            "state": normalized_input.get("state", torch.zeros((1, 1, action_head.config.max_state_dim), device=device)),
            "embodiment_id": torch.tensor([0], device=device),  # Assuming single embodiment
        }

        action_input = BatchFeature(data=action_input_data)

        # Prepare previous chunk for guidance (Y in Equation 2)
        # Right-pad to length H with zeros
        Y_prev = None
        if prev_chunk_overlap is not None and "action" in prev_chunk_overlap:
            prev_actions = prev_chunk_overlap["action"]  # Shape: (H-s, action_dim)
            action_dim = prev_actions.shape[-1]

            # Right-pad to length H
            Y_prev = np.zeros((self.H, action_dim), dtype=prev_actions.dtype)
            Y_prev[:len(prev_actions)] = prev_actions

            # Convert to torch tensor (keep in action space, not normalized)
            Y_prev = torch.from_numpy(Y_prev).unsqueeze(0).to(device)  # Shape: (1, H, action_dim)

        # Convert W to torch tensor
        W_tensor = torch.from_numpy(W).to(device).float()  # Shape: (H,)
        W_diag = torch.diag(W_tensor.repeat(action_head.config.action_dim))  # Repeat for each action dim

        # Run flow matching with ΠGDM guidance
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = action_head.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]

        # Initialize A^0 ~ N(0, I)
        A_tau = torch.randn(
            size=(batch_size, self.H, action_head.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = action_head.num_inference_timesteps
        dt = 1.0 / num_steps

        # Denoising loop with ΠGDM guidance
        for step_idx in range(num_steps):
            tau = step_idx / float(num_steps)  # τ ∈ [0, 1)
            tau_next = (step_idx + 1) / float(num_steps)

            # Compute r_tau^2 (Equation 4)
            r_tau_sq = ((1 - tau) ** 2) / (tau ** 2 + (1 - tau) ** 2)

            # Enable gradients for vector-Jacobian product computation
            A_tau_input = A_tau.detach().requires_grad_(True)

            # Compute velocity field v_π(A^τ, o, τ)
            tau_discretized = int(tau * action_head.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=tau_discretized, device=device
            )

            # Encode action with timestep
            action_features = action_head.action_encoder(A_tau_input, timesteps_tensor, embodiment_id)

            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join embeddings
            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run DiT model
            model_output = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )

            v_pred = action_head.action_decoder(model_output, embodiment_id)
            v_pi = v_pred[:, -self.H:]  # Extract velocity for action chunk

            # Compute A^{c1} (Equation 3): estimate of final denoised action
            A_c1 = A_tau_input + (1 - tau) * v_pi

            # Apply ΠGDM guidance if we have previous chunk
            guidance = torch.zeros_like(v_pi)
            if Y_prev is not None:
                # Compute weighted error: (Y - A^{c1})^T diag(W)
                error = Y_prev - A_c1  # Shape: (1, H, action_dim)
                error_flat = error.reshape(batch_size, -1)  # Shape: (1, H * action_dim)
                weighted_error = torch.matmul(error_flat, W_diag)  # Shape: (1, H * action_dim)

                # Compute vector-Jacobian product: weighted_error · ∂A^{c1}/∂A^τ
                # This is equivalent to computing the gradient
                A_c1_flat = A_c1.reshape(batch_size, -1)
                vjp = torch.autograd.grad(
                    outputs=A_c1_flat,
                    inputs=A_tau_input,
                    grad_outputs=weighted_error,
                    create_graph=False,
                )[0]

                # Compute guidance weight with clipping
                guidance_weight = min(self.beta, (1 - tau) / (tau * r_tau_sq + 1e-8))
                guidance = guidance_weight * vjp

            # Update A^τ using Euler integration (Equation 1) with guidance
            with torch.no_grad():
                A_tau = A_tau + dt * (v_pi + guidance)

        # A_tau is now A^1, the final denoised action chunk
        # Unnormalize and convert back to numpy
        action_pred = A_tau.float()
        unnormalized_action = self.policy.unapply_transforms({"action": action_pred.cpu()})

        # Remove batch dimension if input wasn't batched
        if not is_batch:
            from gr00t.model.policy import squeeze_dict_values
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        return unnormalized_action

    def _compute_soft_mask(self, d: int, s: int) -> np.ndarray:
        """
        Compute soft mask weights W according to Equation 5.

        The mask has three regions:
        1. [0, d): Weight = 1 (frozen actions, will be executed before new chunk is ready)
        2. [d, H-s): Weight = exponentially decaying from 1 to 0 (intermediate region)
        3. [H-s, H): Weight = 0 (beyond previous chunk, must be freshly generated)

        Args:
            d: Inference delay in timesteps
            s: Execution horizon

        Returns:
            Weight array W of shape (H,) with values in [0, 1]
        """
        W = np.zeros(self.H, dtype=np.float32)

        for i in range(self.H):
            if i < d:
                # Frozen region
                W[i] = 1.0
            elif i < self.H - s:
                # Intermediate region with exponential decay
                c_i = (self.H - s - i) / (self.H - s - d + 1)
                W[i] = c_i * np.exp(c_i - 1) / (np.e - 1)

        return W

    def close(self):
        """Stop the background inference thread and clean up resources."""
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


# Default constants
CTRL_LOOP_DT_MS = 20  # 50Hz control loop
AVERAGE_INFERENCE_DELAY_MS = 80  # Typical VLA inference delay
