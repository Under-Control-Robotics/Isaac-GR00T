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

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import torch
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
from .value_head import ValueHead, ValueHeadConfig

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    value_head_cfg: dict = field(
        default_factory=dict, metadata={"help": "Value head configuration."}
    )

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})
    enable_rl: bool = field(
        default=False, metadata={"help": "Enable RL finetuning with value head."}
    )
    enable_advantage_conditioning: bool = field(
        default=False, metadata={"help": "Enable advantage-conditioned policy training."}
    )
    indicator_embedding_dim: int = field(
        default=4096,
        metadata={"help": "Dimension for indicator embedding (should match backbone hidden_size)."},
    )
    enable_advantage_weighted_loss: bool = field(
        default=False,
        metadata={"help": "Enable advantage-weighted loss: wt = sigmoid(k * (At - threshold))."},
    )
    advantage_loss_weight_k: float = field(
        default=75.0,
        metadata={"help": "Constant k for advantage-weighted loss (typically 50-100)."},
    )
    global_threshold: float | None = field(
        default=None, metadata={"help": "Global advantage threshold for weighted loss computation."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize value_head_cfg with default if not provided
        if "value_head_cfg" not in kwargs:
            self.value_head_cfg = {}
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        # Add value head if RL is enabled
        self.enable_rl = getattr(config, "enable_rl", False)
        if self.enable_rl:
            # Use default config if not provided
            value_head_cfg_dict = getattr(config, "value_head_cfg", {}) or {}
            # Set default hidden_size to backbone output dimension (1536 from EagleBackbone)
            if "hidden_size" not in value_head_cfg_dict:
                value_head_cfg_dict["hidden_size"] = config.backbone_cfg.get("project_to_dim", 1536)
            value_head_cfg = ValueHeadConfig(**value_head_cfg_dict)
            self.value_head = ValueHead(value_head_cfg)
            print(f"Initialized value head for RL finetuning")
        else:
            self.value_head = None

        # Add indicator embedding if advantage conditioning is enabled
        self.enable_advantage_conditioning = getattr(config, "enable_advantage_conditioning", False)
        if self.enable_advantage_conditioning:
            from .indicator_embedding import IndicatorEmbedding

            # Get backbone hidden size from action_head_cfg (same as used for value head)
            # This is the dimension of features after backbone processing
            backbone_hidden_size = None
            if hasattr(config, "action_head_cfg") and isinstance(config.action_head_cfg, dict):
                backbone_hidden_size = config.action_head_cfg.get("backbone_embedding_dim", None)

            # Fallback to indicator_embedding_dim from config
            if backbone_hidden_size is None:
                indicator_embedding_dim = getattr(config, "indicator_embedding_dim", None)
                if indicator_embedding_dim is not None:
                    backbone_hidden_size = indicator_embedding_dim

            # Final fallback to default
            if backbone_hidden_size is None:
                backbone_hidden_size = 4096
                print(
                    f"Warning: Could not determine backbone hidden size from config, using default {backbone_hidden_size}"
                )

            self.indicator_embedding = IndicatorEmbedding(
                hidden_size=backbone_hidden_size,
                num_indicators=2,  # Binary: 0 or 1
            )
            print(f"Initialized indicator embedding for advantage-conditioned policy training")
            print(f"  Indicator embedding dim: {backbone_hidden_size}")
        else:
            self.indicator_embedding = None

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        # Prepend indicator token if advantage conditioning is enabled
        if self.enable_advantage_conditioning and self.indicator_embedding is not None:
            if "indicator" in inputs:
                indicators = inputs["indicator"]  # (batch_size,) or (batch_size, action_horizon)

                # Convert to tensor if numpy
                if not isinstance(indicators, torch.Tensor):
                    indicators = torch.from_numpy(indicators).to(self.device)

                # Take first timestep indicator if multiple timesteps provided
                if indicators.dim() > 1:
                    indicators = indicators[:, 0]  # (batch_size,)

                # Embed indicator: (batch_size,) -> (batch_size, 1, hidden_size)
                indicator_tokens = self.indicator_embedding(indicators)

                # Prepend indicator token to backbone features
                # Before: backbone_features shape (batch_size, seq_len, hidden_size)
                # After:  backbone_features shape (batch_size, seq_len+1, hidden_size)
                backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
                backbone_outputs[BACKBONE_FEATURE_KEY] = torch.cat(
                    [indicator_tokens, backbone_features], dim=1
                )

                # Debug logging on first batch
                if not hasattr(self, "_logged_indicator_conditioning"):
                    self._logged_indicator_conditioning = True
                    print("\n" + "=" * 80)
                    print("[MODEL] Advantage conditioning verification (first forward pass):")
                    print("=" * 80)
                    print(
                        f"  Input indicators (first 10): {indicators[:min(10, len(indicators))].tolist()}"
                    )
                    print(f"  Unique indicator values: {torch.unique(indicators).tolist()}")
                    print(
                        f"  Distribution: {(indicators == 1).sum().item()}/{len(indicators)} are 1 (good)"
                    )
                    print(f"  Indicator tokens shape: {indicator_tokens.shape}")
                    print(f"  Backbone features shape (before): {backbone_features.shape}")
                    print(
                        f"  Backbone features shape (after prepending): {backbone_outputs[BACKBONE_FEATURE_KEY].shape}"
                    )
                    print(f"  ✓ Indicator token successfully prepended as FIRST token")
                    print("=" * 80 + "\n")

        # If training ONLY value head (RL mode), skip expensive action generation
        # Only need to process backbone features through vlln/vl_self_attention
        train_value_only = (
            self.enable_rl
            and self.value_head is not None
            and "value" in inputs
            and not any(
                p.requires_grad for p in self.action_head.model.parameters()
            )  # diffusion frozen
        )

        if train_value_only:
            # Skip action loss computation, only process backbone for value head
            if not hasattr(self, "_logged_value_only_mode"):
                self._logged_value_only_mode = True
                print("\n[INFO] Value-only training mode: Skipping action loss computation")
            action_head_outputs = BatchFeature()
            # Still need to process backbone through vlln + vl_self_attention for value head
            backbone_outputs = self.action_head.process_backbone_output(backbone_outputs)
        else:
            # Normal forward pass with action loss
            # The action head will now receive indicator token + VLM tokens
            action_head_outputs = self.action_head(backbone_outputs, action_inputs)
            self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

            # Apply advantage-weighted loss if enabled
            if (
                self.config.enable_advantage_weighted_loss
                and self.enable_advantage_conditioning
                and "advantage" in inputs
                and LOSS_KEY in action_head_outputs
            ):

                advantages = inputs["advantage"]  # (batch_size,) or (batch_size, action_horizon)

                # Convert to tensor if numpy
                if not isinstance(advantages, torch.Tensor):
                    advantages = torch.from_numpy(advantages).to(self.device)

                # Take first timestep advantage if multiple timesteps provided
                if advantages.dim() > 1:
                    advantages = advantages[:, 0]  # (batch_size,)

                # Get global threshold
                threshold = self.config.global_threshold
                if threshold is None:
                    raise ValueError(
                        "Advantage-weighted loss is enabled but global_threshold is None. "
                        "Please set global_threshold in the model config."
                    )

                # Compute loss weight: wt = sigmoid(k * (At - threshold))
                k = self.config.advantage_loss_weight_k
                loss_weights = torch.sigmoid(k * (advantages - threshold))  # (batch_size,)

                # Apply weight to loss (loss is typically a scalar, so we compute weighted mean)
                original_loss = action_head_outputs[LOSS_KEY]
                weighted_loss = original_loss * loss_weights.mean()
                action_head_outputs[LOSS_KEY] = weighted_loss

                # Debug logging on first batch
                if not hasattr(self, "_logged_advantage_weighted_loss"):
                    self._logged_advantage_weighted_loss = True
                    print("\n" + "=" * 80)
                    print("[MODEL] Advantage-weighted loss verification (first forward pass):")
                    print("=" * 80)
                    print(
                        f"  Advantages (first 10): {advantages[:min(10, len(advantages))].tolist()}"
                    )
                    print(f"  Global threshold: {threshold:.6f}")
                    print(f"  k constant: {k}")
                    print(
                        f"  Loss weights (first 10): {loss_weights[:min(10, len(loss_weights))].tolist()}"
                    )
                    print(f"  Mean loss weight: {loss_weights.mean().item():.4f}")
                    print(f"  Original loss: {original_loss.item():.6f}")
                    print(f"  Weighted loss: {weighted_loss.item():.6f}")
                    print(f"  ✓ Advantage-weighted loss successfully applied")
                    print("=" * 80 + "\n")

        # Add value prediction and loss if RL is enabled
        if self.enable_rl and self.value_head is not None:
            # Get backbone features: (batch_size, seq_len, hidden_size)
            backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]

            # Predict single state value: (batch_size, 1, 1)
            value_pred = self.value_head(backbone_features)

            # Compute value loss if target values are provided
            if "value" in inputs:
                value_target = inputs["value"]  # (batch_size, action_horizon)
                # Convert to tensor if needed
                if not isinstance(value_target, torch.Tensor):
                    value_target = torch.tensor(
                        value_target, dtype=value_pred.dtype, device=value_pred.device
                    )

                # Compute value loss (loss function handles shape internally)
                value_loss = self.value_head.compute_value_loss(value_pred, value_target)

                # DEBUG: Log value statistics every 100 steps and accumulate for end stats
                if not hasattr(self, "_value_log_counter"):
                    self._value_log_counter = 0
                    self._value_losses = []
                    self._value_pred_ranges = []
                self._value_log_counter += 1

                # Store stats for final summary
                actual_targets = value_target[:, 0] if value_target.dim() > 1 else value_target
                self._value_losses.append(value_loss.item())
                self._value_pred_ranges.append(
                    (value_pred.min().item(), value_pred.max().item(), value_pred.mean().item())
                )

                # Print every 100 steps
                if self._value_log_counter % 100 == 1:
                    print(f"\n[Value Training Stats - Step {self._value_log_counter}]")
                    print(
                        f"  Target values: min={actual_targets.min().item():.3f}, max={actual_targets.max().item():.3f}, mean={actual_targets.mean().item():.3f}"
                    )
                    print(
                        f"  Predictions: min={value_pred.min().item():.3f}, max={value_pred.max().item():.3f}, mean={value_pred.mean().item():.3f}"
                    )
                    print(f"  Value loss: {value_loss.item():.6f}")

                # Print final summary at last 10 steps (to ensure we catch the end)
                if self._value_log_counter % 10 == 0 and self._value_log_counter >= (
                    getattr(self, "_total_steps", 1000) - 10
                ):
                    print(f"\n[Value Training Stats - Step {self._value_log_counter}] (Near End)")
                    print(
                        f"  Target values: min={actual_targets.min().item():.3f}, max={actual_targets.max().item():.3f}, mean={actual_targets.mean().item():.3f}"
                    )
                    print(
                        f"  Predictions: min={value_pred.min().item():.3f}, max={value_pred.max().item():.3f}, mean={value_pred.mean().item():.3f}"
                    )
                    print(f"  Value loss: {value_loss.item():.6f}")

                    # Print summary of training progress
                    if len(self._value_losses) > 100:
                        recent_losses = self._value_losses[-100:]
                        recent_pred_ranges = self._value_pred_ranges[-100:]
                        all_mins = [r[0] for r in recent_pred_ranges]
                        all_maxs = [r[1] for r in recent_pred_ranges]
                        print(f"\n  Last 100 steps summary:")
                        print(
                            f"    Loss: {min(recent_losses):.6f} -> {recent_losses[-1]:.6f} (improvement: {recent_losses[0] - recent_losses[-1]:.6f})"
                        )
                        print(f"    Pred range: [{min(all_mins):.3f}, {max(all_maxs):.3f}]")

                # Add to outputs
                action_head_outputs["value_pred"] = value_pred
                action_head_outputs["value_loss"] = value_loss

                # For value-only training, use only value loss
                # Otherwise combine with action loss
                if train_value_only:
                    action_head_outputs[LOSS_KEY] = value_loss
                else:
                    if LOSS_KEY in action_head_outputs:
                        action_head_outputs[LOSS_KEY] = action_head_outputs[LOSS_KEY] + value_loss
                    else:
                        action_head_outputs[LOSS_KEY] = value_loss

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)

        # Add value prediction if RL is enabled
        if self.enable_rl and self.value_head is not None:
            backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
            value_pred = self.value_head(backbone_features)
            action_head_outputs["value_pred"] = value_pred

        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)
        tune_value_head = kwargs.pop("tune_value_head", True)

        # Check if we need to enable RL mode (add value head)
        enable_rl = kwargs.pop("enable_rl", False)
        value_head_cfg = kwargs.pop("value_head_cfg", None)

        # Check if we need to enable advantage conditioning (add indicator embedding)
        enable_advantage_conditioning = kwargs.pop("enable_advantage_conditioning", False)
        indicator_embedding_dim = kwargs.pop("indicator_embedding_dim", 4096)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")
        if enable_advantage_conditioning:
            print(f"Enable advantage conditioning: True")
            print(f"Indicator embedding dim: {indicator_embedding_dim}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        # Load with strict=False to allow missing value_head weights
        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )

        # Handle value head initialization if RL mode is enabled but value head doesn't exist
        if enable_rl and pretrained_model.value_head is None:
            print("Initializing new value head for RL finetuning...")
            pretrained_model.enable_rl = True

            # Create value head config
            value_head_cfg_dict = value_head_cfg if value_head_cfg else {}
            if "hidden_size" not in value_head_cfg_dict:
                value_head_cfg_dict["hidden_size"] = pretrained_model.config.action_head_cfg.get(
                    "backbone_embedding_dim", 1536
                )
            if "hidden_dim" not in value_head_cfg_dict:
                value_head_cfg_dict["hidden_dim"] = 1024
            if "dropout" not in value_head_cfg_dict:
                value_head_cfg_dict["dropout"] = 0.1

            value_head_config = ValueHeadConfig(**value_head_cfg_dict)
            pretrained_model.value_head = ValueHead(value_head_config)

            # Move to device
            pretrained_model.value_head.to(
                device=pretrained_model.device, dtype=pretrained_model.action_head.dtype
            )

            # Update config
            pretrained_model.config.enable_rl = True
            pretrained_model.config.value_head_cfg = value_head_cfg_dict

            print(f"Created value head with hidden_size={value_head_cfg_dict['hidden_size']}")

        # Set value head trainability if it exists
        if pretrained_model.value_head is not None:
            pretrained_model.value_head.set_trainable_parameters(tune_value_head=tune_value_head)
            print(f"Tune value head: {tune_value_head}")

        # Handle indicator embedding initialization if advantage conditioning is enabled but it doesn't exist
        if enable_advantage_conditioning and pretrained_model.indicator_embedding is None:
            print("Initializing indicator embedding for advantage-conditioned policy training...")
            from .indicator_embedding import IndicatorEmbedding

            # Get backbone hidden size from action_head_cfg
            backbone_hidden_size = pretrained_model.config.action_head_cfg.get(
                "backbone_embedding_dim", indicator_embedding_dim
            )

            pretrained_model.indicator_embedding = IndicatorEmbedding(
                hidden_size=backbone_hidden_size,
                num_indicators=2,  # Binary: 0 or 1
            )

            # Move to device
            pretrained_model.indicator_embedding.to(
                device=pretrained_model.device, dtype=pretrained_model.action_head.dtype
            )

            # Update config
            pretrained_model.enable_advantage_conditioning = True
            pretrained_model.config.enable_advantage_conditioning = True
            pretrained_model.config.indicator_embedding_dim = backbone_hidden_size

            # Set indicator embedding as trainable
            pretrained_model.indicator_embedding.set_trainable_parameters(
                tune_indicator_embedding=True
            )

            print(f"Created indicator embedding with hidden_size={backbone_hidden_size}")
            print(f"Indicator embedding is trainable")

        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
