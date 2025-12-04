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

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import PretrainedConfig


@dataclass
class ValueHeadConfig(PretrainedConfig):
    """Configuration for the value head."""

    hidden_size: int = 4096  # Input size from backbone features
    hidden_dim: int = 1024  # Hidden layer dimension
    dropout: float = 0.1  # Dropout rate


class ValueHead(nn.Module):
    """
    2-layer MLP value function head.

    Takes backbone features and predicts a value in range (-1, 0].
    Values are clipped to this range during training to match the data distribution.
    """

    def __init__(self, config: ValueHeadConfig):
        super().__init__()
        self.config = config

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the MLP."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict value.

        Args:
            backbone_features: Tensor of shape (batch_size, seq_len, hidden_size)
                from the backbone (VLM embeddings).

        Returns:
            value_pred: Tensor of shape (batch_size, seq_len, 1) with value predictions.
                Values are in range (-1, 0] after clipping.
        """
        # Apply MLP
        value_pred = self.mlp(backbone_features)

        # Clip to (-1, 0] range to match data distribution
        # Use a small epsilon to avoid exactly 0
        value_pred = torch.clamp(value_pred, min=-1.0, max=0.0)

        return value_pred

    def compute_value_loss(
        self, value_pred: torch.Tensor, value_target: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between predicted and target values.

        Args:
            value_pred: Predicted values, shape (batch_size, seq_len, 1)
            value_target: Target values from data, shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            mask: Optional mask for valid timesteps, shape (batch_size, seq_len)

        Returns:
            loss: Scalar loss value
        """
        # Ensure target has same shape as prediction
        if value_target.dim() == 2:
            value_target = value_target.unsqueeze(-1)

        # Compute MSE loss
        loss = nn.functional.mse_loss(value_pred, value_target, reduction="none")

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def set_trainable_parameters(self, tune_value_head: bool = True):
        """Set which parameters should be trainable."""
        for param in self.parameters():
            param.requires_grad = tune_value_head
