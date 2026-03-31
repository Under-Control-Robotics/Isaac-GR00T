import torch
import torch.nn as nn
from torch import Tensor

class RLTokenModule(nn.Module):
    """
    Encoder-decoder transformer that compresses VLA backbone features
    into a single RL token z_rl (Eq. 1-2 from RLT paper).

    Encoder: [z_1:M, e_rl] -> z_rl  (output at appended token position)
    Decoder: z_rl -> reconstructs each z_i autoregressively
    """

    def __init__(
        self,
        vla_dim: int = 1536,       # backbone_features dim
        rl_token_dim: int = 2048,  # matches Fig. 2 in paper
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

        # Learned readout token e_rl
        self.rl_token_embedding = nn.Parameter(torch.randn(1, 1, vla_dim))

        # Small encoder transformer g_ϕ
        enc_layer = nn.TransformerEncoderLayer(
            d_model=vla_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers,
                                            norm=nn.LayerNorm(vla_dim))

        # Project encoder output -> RL token dim
        self.rl_proj = nn.Linear(vla_dim, rl_token_dim)

        # Small decoder transformer d_ϕ  (z_rl as memory, z_{1:i-1} as target)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=vla_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers,
                                            norm=nn.LayerNorm(vla_dim))

        # Project z_rl back to vla_dim for decoder cross-attention
        self.rl_to_vla = nn.Linear(rl_token_dim, vla_dim)

        # Output projection h_ϕ (predicts each z_i)
        self.output_proj = nn.Linear(vla_dim, vla_dim)

        # Causal mask for decoder (registered as buffer)
        self.register_buffer(
            "_causal_mask",
            nn.Transformer.generate_square_subsequent_mask(max_seq_len),
        )

    def encode(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            backbone_features: (B, M, 1536)  — stop-gradient in RL phase
            attn_mask:         (B, M) bool mask from backbone
        Returns:
            z_rl: (B, rl_token_dim)
        """
        B = backbone_features.size(0)
        e_rl = self.rl_token_embedding.expand(B, -1, -1)          # (B, 1, 1536)
        seq = torch.cat([backbone_features, e_rl], dim=1)          # (B, M+1, 1536)

        # Extend attention mask to include the RL token (always attended)
        if attn_mask is not None:
            rl_mask = torch.ones(B, 1, dtype=torch.bool, device=attn_mask.device)
            src_key_padding_mask = ~torch.cat([attn_mask, rl_mask], dim=1)  # (B, M+1) inverted
        else:
            src_key_padding_mask = None

        encoded = self.encoder(seq, src_key_padding_mask=src_key_padding_mask)  # (B, M+1, 1536)
        z_rl = self.rl_proj(encoded[:, -1, :])                     # (B, rl_token_dim)
        return z_rl

    def decode(self, z_rl: Tensor, target_embeddings: Tensor) -> Tensor:
        """
        Autoregressively reconstruct backbone embeddings from z_rl (Eq. 2).
        Args:
            z_rl:               (B, rl_token_dim)
            target_embeddings:  (B, M, 1536)  — stop-gradient targets
        Returns:
            pred_embeddings: (B, M, 1536)  — predictions for each z_i
        """
        B, M, _ = target_embeddings.shape

        # Decoder input: [z_rl_projected, z_1, ..., z_{M-1}]  (teacher forcing)
        z_rl_proj = self.rl_to_vla(z_rl).unsqueeze(1)             # (B, 1, 1536)
        decoder_input = torch.cat([z_rl_proj, target_embeddings[:, :-1, :]], dim=1)  # (B, M, 1536)

        causal_mask = self._causal_mask[:M, :M]
        memory = z_rl_proj                                          # (B, 1, 1536) as cross-attn memory

        decoded = self.decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=causal_mask,
        )                                                           # (B, M, 1536)
        return self.output_proj(decoded)

    def reconstruction_loss(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """
        Eq. 2: L_ro = E_D [ sum_i || h_ϕ(d_ϕ([z_rl, sg(z_{1:i-1})])) - sg(z_i) ||^2 ]
        VLA backbone must be called with torch.no_grad() before this.
        """
        z_targets = backbone_features.detach()                     # stop-gradient sg(z_i)
        z_rl = self.encode(backbone_features.detach(), attn_mask)  # encoder also sees sg features
        preds = self.decode(z_rl, z_targets)
        return nn.functional.mse_loss(preds, z_targets)

    def forward(self, backbone_features: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        """Encode only — used at rollout time after training."""
        return self.encode(backbone_features, attn_mask)