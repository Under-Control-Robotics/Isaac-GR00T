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
    num_heads: int = 8  # Number of attention heads for temporal modeling


class ValueHead(nn.Module):
    """
    Value function head with cross-attention pooling.

    Takes backbone features (after VLN self-attention) and predicts a single state value V(s_t).
    Uses a learnable query token with cross-attention to aggregate information from all VLM tokens,
    followed by a 3-layer MLP. Output is clipped to (-1, 0] to match training data distribution.

    Note: Expects backbone features AFTER action_head.process_backbone_output() applies
    vlln + vl_self_attention processing.
    """

    def __init__(self, config: ValueHeadConfig):
        super().__init__()
        self.config = config

        # Learnable query token for cross-attention (1 token → 1 value prediction)
        self.value_query = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        # Cross-attention to aggregate information from all VLM tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(config.hidden_size)

        # Deeper 3-layer MLP for value prediction
        self.value_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for the MLP, attention, and query token."""
        # Initialize learnable query token
        nn.init.normal_(self.value_query, mean=0.0, std=0.02)

        # Initialize MLP layers
        for i, module in enumerate(self.value_net.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    # For the final linear layer (index 8), initialize bias to -0.5
                    # This puts initial predictions in the middle of [-1, 0] range
                    # helping the model learn the full range faster
                    is_final_layer = i == len(list(self.value_net.modules())) - 1
                    if is_final_layer:
                        nn.init.constant_(module.bias, -0.5)
                    else:
                        nn.init.zeros_(module.bias)

        # Initialize cross-attention layers
        nn.init.normal_(self.cross_attention.in_proj_weight, mean=0.0, std=0.02)
        if self.cross_attention.in_proj_bias is not None:
            nn.init.zeros_(self.cross_attention.in_proj_bias)
        nn.init.normal_(self.cross_attention.out_proj.weight, mean=0.0, std=0.02)
        if self.cross_attention.out_proj.bias is not None:
            nn.init.zeros_(self.cross_attention.out_proj.bias)

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict single state value with cross-attention.

        Args:
            backbone_features: Tensor of shape (batch_size, seq_len, hidden_size)
                VLM embeddings AFTER action_head.process_backbone_output() has applied
                vlln + vl_self_attention. NOT raw backbone output!

        Returns:
            value_pred: Tensor of shape (batch_size, 1, 1) with single value prediction.
                Values are clipped to (-1, 0] to match training data distribution.
        """
        batch_size = backbone_features.size(0)

        # Expand learnable query token for batch
        query = self.value_query.expand(batch_size, -1, -1)  # (B, 1, hidden_size)

        # Cross-attention: query attends to all VLM tokens to aggregate information
        attn_out, _ = self.cross_attention(
            query,  # Query: (B, 1, hidden_size)
            backbone_features,  # Key: (B, seq_len, hidden_size)
            backbone_features,  # Value: (B, seq_len, hidden_size)
        )  # Output: (B, 1, hidden_size)

        # Residual connection + layer norm
        features = self.attn_norm(query + attn_out)  # (B, 1, hidden_size)

        # Apply 3-layer MLP for value prediction
        value_pred = self.value_net(features)  # (B, 1, 1)

        # Only clip during inference to avoid blocking gradients during training
        # During training, let the loss function handle out-of-range predictions
        if not self.training:
            value_pred = torch.clamp(value_pred, min=-1.0, max=0.0)

        return value_pred

    def compute_value_loss(
        self, value_pred: torch.Tensor, value_target: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute Huber loss between predicted and target single value.

        Huber loss is more robust to outliers than MSE, using L2 loss for small errors
        and L1 loss for large errors. This helps with noisy reward signals.

        Args:
            value_pred: Predicted value, shape (batch_size, 1, 1) or (batch_size, 1)
            value_target: Target value from data, shape (batch_size,) or (batch_size, 1)
                Expected to be the first timestep value V(s_t) from the episode
            mask: Optional mask for valid samples, shape (batch_size,)

        Returns:
            loss: Scalar loss value
        """
        # Squeeze to (batch_size,)
        value_pred = value_pred.squeeze(-1).squeeze(-1)  # (B, 1, 1) -> (B,)

        # Ensure target is also (batch_size,)
        if value_target.dim() > 1:
            # If target has multiple timesteps, take the first one (current state value)
            value_target = (
                value_target[:, 0] if value_target.size(1) > 1 else value_target.squeeze(-1)
            )

        # Compute Huber loss (smooth L1 loss with delta=1.0)
        # For |error| <= delta: 0.5 * error^2
        # For |error| > delta: delta * (|error| - 0.5 * delta)
        loss = nn.functional.huber_loss(value_pred, value_target, reduction="none", delta=1.0)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss

    def set_trainable_parameters(self, tune_value_head: bool = True):
        """Set which parameters should be trainable."""
        for param in self.parameters():
            param.requires_grad = tune_value_head
