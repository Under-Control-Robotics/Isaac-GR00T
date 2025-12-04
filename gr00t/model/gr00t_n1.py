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
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)

        # Add value prediction and loss if RL is enabled
        if self.enable_rl and self.value_head is not None:
            # Get backbone features: (batch_size, seq_len, hidden_size)
            backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]

            # Predict values: (batch_size, seq_len, 1)
            value_pred = self.value_head(backbone_features)

            # Compute value loss if target values are provided
            if "value" in inputs:
                value_target = inputs["value"]  # (batch_size, action_horizon)
                # Convert to tensor if needed
                if not isinstance(value_target, torch.Tensor):
                    value_target = torch.tensor(
                        value_target, dtype=value_pred.dtype, device=value_pred.device
                    )

                # Compute value loss
                value_loss = self.value_head.compute_value_loss(
                    value_pred.squeeze(-1), value_target
                )

                # Add to outputs
                action_head_outputs["value_pred"] = value_pred
                action_head_outputs["value_loss"] = value_loss

                # Combine with action loss
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

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

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

        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
