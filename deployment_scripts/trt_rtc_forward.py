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
Real-Time Chunking (RTC) for TensorRT GR00T models.

Since TensorRT engines don't support autograd, we use a simplified RTC approach:
- Weighted initialization: Blend previous chunk with noise based on soft mask
- No gradient-based guidance during denoising loop

This is simpler than full ΠGDM but still provides significant continuity benefits.
"""

import os
from functools import partial
import numpy as np

import torch
import trt_torch as trt
from transformers.feature_extraction_utils import BatchFeature


def compute_soft_mask_weights(H, d, s):
    """
    Compute soft mask weights W according to Equation 5 from the RTC paper.

    The mask has three regions:
    1. [0, d): Weight = 1 (frozen actions, will be executed before new chunk is ready)
    2. [d, H-s): Weight = exponentially decaying from 1 to 0 (intermediate region)
    3. [H-s, H): Weight = 0 (beyond previous chunk, must be freshly generated)

    Args:
        H: Prediction horizon (action chunk size)
        d: Inference delay in timesteps
        s: Execution horizon (actions executed before next inference starts)

    Returns:
        Weight array W of shape (H,) with values in [0, 1]
    """
    W = np.zeros(H, dtype=np.float32)

    for i in range(H):
        if i < d:
            # Frozen region - these actions will execute before new chunk is ready
            W[i] = 1.0
        elif i < H - s:
            # Intermediate region with exponential decay
            c_i = (H - s - i) / (H - s - d + 1)
            W[i] = c_i * np.exp(c_i - 1) / (np.e - 1)
        # else: W[i] = 0 (already initialized to 0)

    return W


def eagle_tensorrt_forward(self, vl_input):
    """Original eagle forward function - no changes needed for RTC."""
    eagle_prefix = "eagle_"
    eagle_input = {
        k.removeprefix(eagle_prefix): v for k, v in vl_input.items() if k.startswith(eagle_prefix)
    }
    del eagle_input["image_sizes"]
    vl_input = eagle_input

    self.set_frozen_modules_to_eval_mode()
    batch_size = vl_input["pixel_values"].shape[0]
    position_ids = torch.arange(self.num_patches, device="cuda").expand((batch_size, -1))
    if vl_input["pixel_values"].dtype != torch.float16:
        vl_input["pixel_values"] = vl_input["pixel_values"].to(torch.float16)

    assert (
        vl_input["pixel_values"].shape[0] <= 8
    ), "Batch size must be <= 8 because TensorRT engine was built with max_batch_size=8, you can try to adjust the max_batch_size in the build_engine.sh script and rebuild the engine."

    self.vit_engine.set_runtime_tensor_shape("pixel_values", vl_input["pixel_values"].shape)
    self.vit_engine.set_runtime_tensor_shape("position_ids", position_ids.shape)
    vit_embeds = self.vit_engine(vl_input["pixel_values"], position_ids)["vit_embeds"]
    vit_embeds = vit_embeds.view(1, -1, vit_embeds.shape[-1])
    self.llm_engine.set_runtime_tensor_shape("input_ids", vl_input["input_ids"].shape)
    self.llm_engine.set_runtime_tensor_shape("vit_embeds", vit_embeds.shape)
    self.llm_engine.set_runtime_tensor_shape("attention_mask", vl_input["attention_mask"].shape)
    embeddings = self.llm_engine(vl_input["input_ids"], vit_embeds, vl_input["attention_mask"])[
        "embeddings"
    ]

    return BatchFeature(
        data={
            "backbone_features": embeddings,
            "backbone_attention_mask": vl_input["attention_mask"],
        }
    )


def action_head_tensorrt_forward_rtc(
    self,
    backbone_output,
    action_input,
    prev_chunk=None,
    d=None,
    s=None,
):
    """
    TensorRT action head forward with RTC support via weighted initialization.

    This is a modified version of action_head_tensorrt_forward that supports
    Real-Time Chunking by initializing actions with a weighted blend of the
    previous chunk and new noise.

    Args:
        self: Action head instance
        backbone_output: Output from backbone (vision + language features)
        action_input: Action input containing state and embodiment_id
        prev_chunk: Previous action chunk tensor of shape (batch_size, H, action_dim) or None
        d: Inference delay in timesteps (e.g., 4 for 80ms delay with 20ms control)
        s: Execution horizon (actions executed before next inference starts)

    Returns:
        BatchFeature with "action_pred" key
    """
    # Process backbone output
    if backbone_output.backbone_features.dtype != torch.float16:
        backbone_output.backbone_features = backbone_output.backbone_features.to(torch.float16)
    self.vlln_vl_self_attention_engine.set_runtime_tensor_shape(
        "backbone_features", backbone_output.backbone_features.shape
    )
    backbone_output.backbone_features = self.vlln_vl_self_attention_engine(
        backbone_output.backbone_features
    )["output"]
    vl_embs = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id
    batch_size = vl_embs.shape[0]

    if action_input.state.dtype != torch.float16:
        action_input.state = action_input.state.to(torch.float16)

    if embodiment_id.dtype != torch.int64:
        embodiment_id = embodiment_id.to(torch.int64)

    if vl_embs.dtype != torch.float16:
        vl_embs = vl_embs.to(torch.float16)

    # Embed state with batch processing
    self.state_encoder_engine.set_runtime_tensor_shape("state", action_input.state.shape)
    self.state_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
    state_features = self.state_encoder_engine(action_input.state, embodiment_id)["output"]

    # Initialize actions with RTC weighted blending
    device = vl_embs.device

    # Sample fresh noise
    noise = torch.randn(
        size=(batch_size, self.config.action_horizon, self.config.action_dim),
        dtype=vl_embs.dtype,
        device=device,
    )

    # Apply RTC weighted initialization if previous chunk is provided
    if prev_chunk is not None and d is not None and s is not None:
        # Ensure prev_chunk is on the correct device and dtype
        if not isinstance(prev_chunk, torch.Tensor):
            prev_chunk = torch.from_numpy(prev_chunk)
        prev_chunk = prev_chunk.to(device=device, dtype=vl_embs.dtype)

        # Expand batch dimension if needed
        if prev_chunk.ndim == 2:  # (H, action_dim)
            prev_chunk = prev_chunk.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute soft mask weights
        H = self.config.action_horizon
        W = compute_soft_mask_weights(H, d, s)
        W_tensor = torch.from_numpy(W).to(device=device, dtype=vl_embs.dtype)

        # Weighted blending: actions = W * prev_chunk + (1 - W) * noise
        # Reshape W for broadcasting: (H,) -> (1, H, 1)
        W_expanded = W_tensor.view(1, H, 1)

        # Blend previous chunk with noise
        actions = W_expanded * prev_chunk + (1 - W_expanded) * noise

        # Store RTC info for debugging (optional)
        if not hasattr(self, '_rtc_debug'):
            self._rtc_debug = {}
        self._rtc_debug['last_d'] = d
        self._rtc_debug['last_s'] = s
        self._rtc_debug['last_weights'] = W
    else:
        # No RTC - use pure noise (standard behavior)
        actions = noise

    # This attribute is used to ensure the same actions is used for both PyTorch and TensorRT inference
    if hasattr(self, "init_actions"):
        actions = self.init_actions.expand((batch_size, -1, -1))

    num_steps = self.num_inference_timesteps
    dt = 1.0 / num_steps

    # Standard flow matching denoising loop
    for t in range(num_steps):
        t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory with batch processing
        timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)

        self.action_encoder_engine.set_runtime_tensor_shape("actions", actions.shape)
        self.action_encoder_engine.set_runtime_tensor_shape(
            "timesteps_tensor", timesteps_tensor.shape
        )
        self.action_encoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        action_features = self.action_encoder_engine(actions, timesteps_tensor, embodiment_id)[
            "output"
        ]

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0).to(torch.float16)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1).to(
            torch.float16
        )
        # Run model forward with batch processing
        if vl_embs.dtype != torch.float16:
            vl_embs = vl_embs.to(torch.float16)

        self.DiT_engine.set_runtime_tensor_shape("vl_embs", vl_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("sa_embs", sa_embs.shape)
        self.DiT_engine.set_runtime_tensor_shape("timesteps_tensor", timesteps_tensor.shape)
        model_output = self.DiT_engine(sa_embs, vl_embs, timesteps_tensor)["output"]

        self.action_decoder_engine.set_runtime_tensor_shape("model_output", model_output.shape)
        self.action_decoder_engine.set_runtime_tensor_shape("embodiment_id", embodiment_id.shape)
        pred = self.action_decoder_engine(model_output, embodiment_id)["output"]
        pred_velocity = pred[:, -self.action_horizon :]

        # Update actions using euler integration.
        actions = actions + dt * pred_velocity

    return BatchFeature(data={"action_pred": actions})


def setup_tensorrt_engines_with_rtc(policy, trt_engine_path):
    """
    Setup TensorRT engines for GR00T model inference with RTC support.

    This is identical to setup_tensorrt_engines but uses the RTC-enabled forward function.

    Args:
        policy: GR00T policy model instance
        trt_engine_path: Path to TensorRT engine files
    """
    policy.model.backbone.num_patches = (
        policy.model.backbone.eagle_model.vision_model.vision_model.embeddings.num_patches
    )
    if hasattr(policy.model.backbone.eagle_model, "vision_model"):
        del policy.model.backbone.eagle_model.vision_model
    if hasattr(policy.model.backbone.eagle_model, "language_model"):
        del policy.model.backbone.eagle_model.language_model
    if hasattr(policy.model.action_head, "vlln"):
        del policy.model.action_head.vlln
    if hasattr(policy.model.action_head, "vl_self_attention"):
        del policy.model.action_head.vl_self_attention
    if hasattr(policy.model.action_head, "model"):
        del policy.model.action_head.model
    if hasattr(policy.model.action_head, "state_encoder"):
        del policy.model.action_head.state_encoder
    if hasattr(policy.model.action_head, "action_encoder"):
        del policy.model.action_head.action_encoder
    if hasattr(policy.model.action_head, "action_decoder"):
        del policy.model.action_head.action_decoder
    torch.cuda.empty_cache()

    # Setup backbone engines
    policy.model.backbone.vit_engine = trt.Engine(os.path.join(trt_engine_path, "vit.engine"))
    policy.model.backbone.llm_engine = trt.Engine(os.path.join(trt_engine_path, "llm.engine"))

    # Setup action head engines
    policy.model.action_head.vlln_vl_self_attention_engine = trt.Engine(
        os.path.join(trt_engine_path, "vlln_vl_self_attention.engine")
    )
    policy.model.action_head.action_encoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "action_encoder.engine")
    )
    policy.model.action_head.action_decoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "action_decoder.engine")
    )
    policy.model.action_head.DiT_engine = trt.Engine(os.path.join(trt_engine_path, "DiT.engine"))
    policy.model.action_head.state_encoder_engine = trt.Engine(
        os.path.join(trt_engine_path, "state_encoder.engine")
    )

    # Set TensorRT forward functions with RTC support
    policy.model.backbone.forward = partial(eagle_tensorrt_forward, policy.model.backbone)
    policy.model.action_head.get_action = partial(
        action_head_tensorrt_forward_rtc, policy.model.action_head
    )
