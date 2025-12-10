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
Advantage-Conditioned GR00T Policy

This policy extends the base Gr00tPolicy to support advantage-conditioned inference.
The model was trained with advantage indicators (binary 0/1 tokens) that guide the policy
to produce high-quality actions.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.policy import COMPUTE_DTYPE, Gr00tPolicy


class AdvantageConditionedGr00tPolicy(Gr00tPolicy):
    """
    Advantage-Conditioned Policy wrapper for GR00T models trained with indicator tokens.

    This policy requires an "indicator" field in the observation dictionary that specifies
    whether to condition on high-advantage (1) or low-advantage (0) behavior.

    Usage:
        policy = AdvantageConditionedGr00tPolicy(
            model_path="/path/to/advantage_conditioned_checkpoint",
            embodiment_tag="new_embodiment",
            modality_config=modality_config,
            modality_transform=modality_transform,
        )

        # REQUIRED: Always include "indicator" in observation
        obs = {
            "video.ego_view": ...,
            "state.left_arm": ...,
            "indicator": 1.0,  # <-- REQUIRED: 1 = high-quality, 0 = low-quality
            ...
        }
        action = policy.get_action(obs)

        # This will raise ValueError (indicator missing):
        obs_bad = {
            "video.ego_view": ...,
            "state.left_arm": ...,
            # Missing indicator!
        }
        action = policy.get_action(obs_bad)  # ❌ Error!
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the AdvantageConditionedGr00tPolicy.

        Args:
            model_path (str): Path to the advantage-conditioned model checkpoint.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        # Initialize base policy
        super().__init__(
            model_path=model_path,
            embodiment_tag=embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=denoising_steps,
            device=device,
        )

        # Verify that the model has advantage conditioning enabled
        if not self.model.enable_advantage_conditioning:
            raise ValueError(
                f"Model at {model_path} does not have advantage conditioning enabled. "
                "Please ensure you are loading a checkpoint trained with advantage conditioning."
            )

        print("\n" + "=" * 80)
        print("ADVANTAGE-CONDITIONED POLICY LOADED")
        print("=" * 80)
        print(f"Model path: {model_path}")
        print(f"Advantage conditioning: ENABLED")
        print(f"Indicator embedding dim: {self.model.config.indicator_embedding_dim}")
        print("\nREQUIRED: You must pass 'indicator' field in every observation:")
        print("  obs['indicator'] = 1.0  # High-quality actions")
        print("  obs['indicator'] = 0.0  # Low-quality actions")
        print("\nThe policy will raise an error if 'indicator' is missing.")
        print("=" * 80 + "\n")

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the advantage-conditioned model.

        Args:
            observations (Dict[str, Any]): The observation dictionary. MUST contain
                                          an 'indicator' field (0.0 or 1.0).

        Example observations:
            obs = {
                "video.ego_view": np.ndarray,  # (T, H, W, C) or (B, T, H, W, C)
                "state.left_arm": np.ndarray,  # (T, D) or (B, T, D)
                "indicator": 1.0,  # <-- REQUIRED: 1=high-quality, 0=low-quality
                ...
            }

        Returns:
            Dict[str, Any]: The predicted action dictionary.

        Raises:
            ValueError: If 'indicator' field is missing from observations.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()

        # Check if indicator is present
        if "indicator" not in obs_copy:
            raise ValueError(
                "Advantage-conditioned policy requires 'indicator' field in observations.\n"
                "Please add: obs['indicator'] = 1.0 (high-quality) or 0.0 (low-quality)\n"
                "Example:\n"
                "  obs = {\n"
                "      'video.ego_view': ...,\n"
                "      'state.left_arm': ...,\n"
                "      'indicator': 1.0,  # <-- Add this\n"
                "  }"
            )

        # Ensure indicator is a numpy array with correct shape
        indicator = obs_copy["indicator"]
        if isinstance(indicator, (int, float)):
            # Convert scalar to array
            indicator = np.array([indicator], dtype=np.float32)
        elif isinstance(indicator, np.ndarray):
            # Ensure it's float32
            indicator = indicator.astype(np.float32)
            if indicator.ndim == 0:  # scalar array
                indicator = indicator.reshape(1)
        else:
            raise ValueError(f"Invalid indicator type: {type(indicator)}")

        obs_copy["indicator"] = indicator

        # Call parent's get_action which will pass the indicator through to the model
        return super().get_action(obs_copy)

    def _load_model(self, model_path):
        """Load the advantage-conditioned model."""
        model = GR00T_N1_5.from_pretrained(
            model_path,
            torch_dtype=COMPUTE_DTYPE,
            # These flags should be automatically loaded from the checkpoint config
            # but we can also explicitly set them here for clarity
        )
        model.eval()  # Set model to eval mode

        # Verify advantage conditioning is enabled
        if not model.enable_advantage_conditioning:
            raise ValueError(
                f"Model at {model_path} does not have advantage conditioning enabled. "
                "The checkpoint config should have enable_advantage_conditioning=True."
            )

        # Update action_horizon to match modality config (same as base class)
        expected_action_horizon = len(self._modality_config["action"].delta_indices)

        if expected_action_horizon != model.action_head.config.action_horizon:
            print(
                f"Policy: Recreating action head with action_horizon {expected_action_horizon} "
                f"(was {model.action_head.config.action_horizon})"
            )

            # Update the action head config
            new_action_head_config = model.action_head.config
            new_action_head_config.action_horizon = expected_action_horizon

            # Import the FlowmatchingActionHead class
            from gr00t.model.action_head.flow_matching_action_head import (
                FlowmatchingActionHead,
            )

            # Create new action head with updated config
            new_action_head = FlowmatchingActionHead(new_action_head_config)

            # Copy the weights from the old action head to the new one
            new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

            # Replace the action head
            model.action_head = new_action_head

            # Update model config
            model.config.action_horizon = expected_action_horizon
            model.action_horizon = expected_action_horizon
            model.config.action_head_cfg["action_horizon"] = expected_action_horizon

        model.to(device=self.device)  # type: ignore

        self.model = model
