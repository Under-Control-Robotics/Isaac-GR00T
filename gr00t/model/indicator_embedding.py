# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Indicator Embedding Module for Advantage-Conditioned Policy Training.

Embeds binary advantage indicators (0 or 1) and prepends them to VLM tokens.
"""

import torch
import torch.nn as nn


class IndicatorEmbedding(nn.Module):
    """
    Embeds binary advantage indicator (0 or 1) into hidden dimension.

    The embedded indicator token is prepended to VLM tokens before action generation,
    allowing the policy to be conditioned on advantage signals.

    Args:
        hidden_size: Dimension to match backbone features (e.g., 4096 for GR00T)
        num_indicators: Number of indicator values (2 for binary 0/1)
    """

    def __init__(self, hidden_size: int = 4096, num_indicators: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_indicators = num_indicators

        # Embedding table: maps indicator values [0, 1] to hidden_size vectors
        self.embedding = nn.Embedding(num_indicators, hidden_size)

        # Learned position embedding for the indicator token
        # This helps the model distinguish the indicator token from VLM tokens
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, indicators: torch.Tensor) -> torch.Tensor:
        """
        Embed indicator values.

        Args:
            indicators: (batch_size,) tensor with values in {0, 1}

        Returns:
            indicator_tokens: (batch_size, 1, hidden_size) embedded indicator tokens
        """
        # Ensure indicators are long type for embedding lookup
        if indicators.dtype != torch.long:
            indicators = indicators.long()

        # Embed indicators: (B,) -> (B, hidden_size)
        indicator_emb = self.embedding(indicators)

        # Add sequence dimension: (B, hidden_size) -> (B, 1, hidden_size)
        indicator_emb = indicator_emb.unsqueeze(1)

        # Add position embedding to distinguish from VLM tokens
        indicator_tokens = indicator_emb + self.position_embedding

        return indicator_tokens

    def set_trainable_parameters(self, tune_indicator_embedding: bool = True):
        """Set which parameters should be trainable."""
        for param in self.parameters():
            param.requires_grad = tune_indicator_embedding
