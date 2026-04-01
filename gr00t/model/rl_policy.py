import copy

import torch
import torch.nn as nn
from torch import Tensor

from gr00t.model.policy import Gr00tPolicy
from gr00t.model.rl_token import RLTokenModule
from gr00t.model.rl_actor_critic import RLActor, RLCritic


class Gr00tRLTokenPolicy(Gr00tPolicy):
    """
    Wraps Gr00tPolicy with RLT (RL Token) training and inference.

    Phase 1: Train RLTokenModule with frozen VLA (call train_rl_token())
    Phase 2: Freeze VLA + RL token, train RLActor + RLCritic online
    """

    def __init__(self, *args, rl_chunk_len=20, proprio_dim=14, action_dim=14, **kwargs):
        super().__init__(*args, **kwargs)

        # Freeze the VLA completely
        for p in self.model.parameters():
            p.requires_grad_(False)

        linear = self.model.backbone.eagle_linear
        feat_dim = linear.out_features if hasattr(linear, "out_features") else 2048

        self.rl_token_module = RLTokenModule(vla_dim=feat_dim)
        self.actor = RLActor(
            rl_token_dim=2048, proprio_dim=proprio_dim,
            action_dim=action_dim, action_chunk_len=rl_chunk_len,
        )
        self.critic = RLCritic(
            rl_token_dim=2048, proprio_dim=proprio_dim,
            action_dim=action_dim, action_chunk_len=rl_chunk_len,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def extract_rl_token(self, normalized_input: dict) -> Tensor:
        """Get z_rl from a (already normalized) observation dict."""
        backbone_out = self.model.get_backbone_features(normalized_input)
        feats = backbone_out["backbone_features"]
        mask  = backbone_out.get("backbone_attention_mask")
        return self.rl_token_module(feats, mask)              # (B, 2048)

    def get_rl_action(self, observations: dict, proprio: Tensor) -> dict:
        """Full inference: VLA reference + RL actor refinement."""
        obs_copy = self._preprocess_obs(observations)
        normalized = self.apply_transforms(obs_copy)

        # VLA reference chunk (frozen)
        with torch.no_grad():
            vla_out = self.model.get_action(normalized)
            ref_chunk = vla_out["action_pred"][:, :self.actor.action_chunk_len]

        # RL token (frozen after Phase 1)
        z_rl = self.extract_rl_token(normalized)

        # Actor refinement
        action, _, _ = self.actor.sample(z_rl, proprio, ref_chunk)
        return {"action": action, "ref_action": ref_chunk, "z_rl": z_rl}

    # --- Phase 1 training loss ---
    def rl_token_loss(self, normalized_input: dict) -> tuple[Tensor, Tensor, Tensor]:
        """Returns (total_loss, recon_loss, var_loss)."""
        with torch.no_grad():
            backbone_out = self.model.get_backbone_features(normalized_input)
        feats = backbone_out["backbone_features"].float()
        mask  = backbone_out.get("backbone_attention_mask")
        return self.rl_token_module.reconstruction_loss(feats, mask)

    # --- Phase 2 critic update (Eq. 3) ---
    def critic_loss(self, batch: dict, gamma: float = 0.99) -> Tensor:
        z_rl, proprio, actions, rewards, z_rl_next, proprio_next, ref_next = (
            batch["z_rl"], batch["proprio"], batch["actions"],
            batch["rewards"], batch["z_rl_next"], batch["proprio_next"], batch["ref_next"],
        )
        with torch.no_grad():
            a_next, _, _ = self.actor.sample(z_rl_next, proprio_next, ref_next)
            q1_t, q2_t = self.target_critic(z_rl_next, proprio_next, a_next)
            q_target = rewards + gamma ** self.actor.action_chunk_len * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(z_rl, proprio, actions)
        return nn.functional.mse_loss(q1, q_target) + nn.functional.mse_loss(q2, q_target)

    # --- Phase 2 actor update (Eq. 5) ---
    def actor_loss(self, batch: dict, beta: float = 0.1) -> Tensor:
        z_rl, proprio, ref_actions = batch["z_rl"], batch["proprio"], batch["ref_actions"]
        actions, _, _ = self.actor.sample(z_rl, proprio, ref_actions, apply_dropout=True)
        q1, q2 = self.critic(z_rl, proprio, actions)
        q_val = torch.min(q1, q2)
        bc_reg = beta * ((actions - ref_actions) ** 2).sum(-1).sum(-1).mean()
        return (-q_val.mean()) + bc_reg

    def soft_update_target(self, tau: float = 0.005):
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.data.lerp_(p.data, tau)