import torch
import torch.nn as nn
from torch import Tensor

def _mlp(in_dim, hidden, out_dim, layers=3):
      dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
      return nn.Sequential(*[
          nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU())
          if i < len(dims) - 2 else nn.Linear(a, b)
          for i, (a, b) in enumerate(zip(dims, dims[1:]))
      ])


class RLActor(nn.Module):
    """
    π_θ(a_{1:C} | x, ã_{1:C}) = N(µ_θ(x, ã_{1:C}), σ²I)
    x = (z_rl, s_p),  ã = VLA reference chunk
    """

    def __init__(
        self,
        rl_token_dim: int = 2048,
        proprio_dim: int = 14,
        action_dim: int = 14,
        action_chunk_len: int = 20,   # C in the paper
        hidden_dim: int = 512,
        log_std_init: float = -2.0,
        ref_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.action_chunk_len = action_chunk_len
        self.action_dim = action_dim
        self.ref_dropout_prob = ref_dropout_prob

        ref_flat = action_dim * action_chunk_len
        in_dim = rl_token_dim + proprio_dim + ref_flat
        out_dim = action_dim * action_chunk_len

        self.net = _mlp(in_dim, hidden_dim, out_dim)
        self.log_std = nn.Parameter(torch.full((out_dim,), log_std_init))

    def forward(
        self,
        z_rl: Tensor,                  # (B, rl_token_dim)
        proprio: Tensor,               # (B, proprio_dim)
        ref_actions: Tensor,           # (B, C, action_dim)  — VLA reference chunk
        apply_dropout: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Returns (mean, log_std) each (B, C, action_dim)."""
        B = z_rl.size(0)

        # Reference action dropout (Sec. IV-B)
        if apply_dropout:
            mask = (torch.rand(B, 1, 1, device=z_rl.device) > self.ref_dropout_prob).float()
            ref_actions = ref_actions * mask

        ref_flat = ref_actions.reshape(B, -1)                     # (B, C*action_dim)
        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mu = self.net(x).reshape(B, self.action_chunk_len, self.action_dim)
        log_std = self.log_std.reshape(1, self.action_chunk_len, self.action_dim).expand_as(mu)
        return mu, log_std

    def sample(self, z_rl, proprio, ref_actions, apply_dropout=False):
        mu, log_std = self.forward(z_rl, proprio, ref_actions, apply_dropout)
        return mu + torch.randn_like(mu) * log_std.exp(), mu, log_std


class RLCritic(nn.Module):
    """Twin critic Q_ψ(x, a_{1:C}) — TD3 style."""

    def __init__(
        self,
        rl_token_dim: int = 2048,
        proprio_dim: int = 14,
        action_dim: int = 14,
        action_chunk_len: int = 20,
        hidden_dim: int = 512,
    ):
        super().__init__()
        in_dim = rl_token_dim + proprio_dim + action_dim * action_chunk_len
        self.q1 = _mlp(in_dim, hidden_dim, 1)
        self.q2 = _mlp(in_dim, hidden_dim, 1)

    def forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (q1, q2) each (B, 1)."""
        x = torch.cat([z_rl, proprio, actions.reshape(actions.size(0), -1)], dim=-1)
        return self.q1(x), self.q2(x)
