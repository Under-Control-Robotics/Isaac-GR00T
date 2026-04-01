import math

import torch
import torch.nn as nn
from torch import Tensor


class RLTokenModule(nn.Module):
    """
    Encoder-decoder transformer that compresses VLA backbone features
    into a single RL token z_rl (Eq. 1-2 from RLT paper).

    Encoder: [z_1:M, e_rl] -> z_rl  (output at appended token position)
    Decoder: non-autoregressive.
             Queries = proj(z_rl) + fixed sinusoidal PE.
             This forces z_rl through every decoder query so the decoder
             cannot memorise per-position means without using z_rl.
    """

    def __init__(
        self,
        vla_dim: int = 1536,
        rl_token_dim: int = 2048,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.vla_dim = vla_dim
        self.rl_token_dim = rl_token_dim
        self.max_seq_len = max_seq_len

        # Learned readout token e_rl (appended to encoder input)
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, vla_dim))

        # Small encoder transformer g_ϕ
        enc_layer = nn.TransformerEncoderLayer(
            d_model=vla_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(vla_dim)
        )

        # Project encoder output -> RL token dim
        self.rl_proj = nn.Linear(vla_dim, rl_token_dim)

        # Project z_rl back to vla_dim — used as BOTH decoder query base AND memory.
        # Every query = rl_to_vla(z_rl) + sinusoidal_pe[i], so z_rl is always present.
        self.rl_to_vla = nn.Linear(rl_token_dim, vla_dim)

        # Register fixed sinusoidal PE (not learned — prevents per-position memorisation)
        self.register_buffer("_sin_pe", self._build_sinusoidal_pe(max_seq_len, vla_dim))

        # Small decoder transformer d_ϕ
        dec_layer = nn.TransformerDecoderLayer(
            d_model=vla_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(vla_dim)
        )

        # Output projection h_ϕ
        self.output_proj = nn.Linear(vla_dim, vla_dim)

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        return pe  # (max_len, d_model)

    def encode(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            backbone_features: (B, M, vla_dim)
            attn_mask:         (B, M) int/bool — 1=real, 0=padding
        Returns:
            z_rl: (B, rl_token_dim)
        """
        B = backbone_features.size(0)
        e_rl = self.rl_token_embedding.expand(B, -1, -1)
        seq = torch.cat([backbone_features, e_rl], dim=1)   # (B, M+1, vla_dim)

        if attn_mask is not None:
            rl_mask = torch.ones(B, 1, dtype=torch.bool, device=attn_mask.device)
            src_key_padding_mask = ~torch.cat([attn_mask.bool(), rl_mask], dim=1)
        else:
            src_key_padding_mask = None

        encoded = self.encoder(seq, src_key_padding_mask=src_key_padding_mask)
        return self.rl_proj(encoded[:, -1, :])              # (B, rl_token_dim)

    def decode(self, z_rl: Tensor, seq_len: int) -> Tensor:
        """
        Non-autoregressive decode.

        Query construction:  q_i = proj(z_rl) + sin_pe[i]
          - proj(z_rl) is sample-specific → decoder cannot ignore z_rl
          - sin_pe[i] is fixed → position info without learnable memorisation

        Args:
            z_rl:    (B, rl_token_dim)
            seq_len: M
        Returns:
            pred_embeddings: (B, M, vla_dim)
        """
        B = z_rl.size(0)
        z_base = self.rl_to_vla(z_rl).unsqueeze(1)                 # (B, 1, vla_dim)
        pe = self._sin_pe[:seq_len].unsqueeze(0)                    # (1, M, vla_dim)
        queries = z_base.expand(B, seq_len, -1) + pe               # (B, M, vla_dim)
        memory = z_base                                             # (B, 1, vla_dim)

        decoded = self.decoder(tgt=queries, memory=memory)         # (B, M, vla_dim)
        return self.output_proj(decoded)

    @staticmethod
    def _normalize_tokens(x: Tensor) -> Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def variance_loss(self, z_rl: Tensor, eps: float = 1e-4) -> Tensor:
        """VICReg variance term: penalises z_rl collapse across the batch."""
        std = z_rl.std(dim=0)
        return torch.mean(torch.relu(1.0 - std + eps))

    def reconstruction_loss(
        self,
        backbone_features: Tensor,
        attn_mask: Tensor | None = None,
        var_weight: float = 1.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (total_loss, recon_loss, var_loss)."""
        z_targets = self._normalize_tokens(backbone_features.detach())
        z_rl = self.encode(z_targets, attn_mask)
        preds = self.decode(z_rl, seq_len=z_targets.size(1))

        if attn_mask is not None:
            mask = attn_mask.bool()
            recon = nn.functional.mse_loss(preds[mask], z_targets[mask])
        else:
            recon = nn.functional.mse_loss(preds, z_targets)

        var = self.variance_loss(z_rl)
        return recon + var_weight * var, recon, var

    def forward(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Encode only — used at rollout time."""
        return self.encode(backbone_features, attn_mask)
