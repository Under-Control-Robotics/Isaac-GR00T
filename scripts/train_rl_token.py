"""
Phase 1 training: RL Token encoder-decoder.

Trains only RLTokenModule parameters while the VLA backbone is completely frozen.
The reconstruction objective forces the RL token to be a compressed but
information-preserving summary of the VLA's final-layer token embeddings.

Usage:
    python scripts/train_rl_token.py \
        --dataset_path /data/my_task \
        --model_path nvidia/GR00T-N1.5-3B \
        --embodiment_tag new_embodiment \
        --data_config fourier_gr1_arms_only \
        --output_dir /tmp/rl_token_ckpt
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.rl_policy import Gr00tRLTokenPolicy
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class Config:
    # --- data ---
    dataset_path: List[str]
    """Paths to LeRobot dataset directories."""

    data_config: str = "fourier_gr1_arms_only"
    """Data config name — same value you used for fine-tuning."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"

    # --- model ---
    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to fine-tuned GR00T checkpoint (after SFT, before RL)."""

    # --- RL token architecture ---
    rl_token_dim: int = 2048
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8

    # --- training ---
    output_dir: str = "/tmp/rl_token_ckpt"
    batch_size: int = 16
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    log_every: int = 50
    save_every: int = 1000

    # Optional: also fine-tune the VLA jointly (alpha > 0 in paper Eq. 3)
    # Keep 0 unless you want to combine SFT + RL token in one pass.
    vla_lr: float = 0.0


def collate_fn(batch):
    """Stack list-of-dicts into dict-of-tensors, skipping non-tensor values."""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Dataset
    # ------------------------------------------------------------------ #
    data_config_cls = load_data_config(cfg.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()
    embodiment_tag = EmbodimentTag(cfg.embodiment_tag)

    datasets = [
        LeRobotSingleDataset(
            dataset_path=p,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=cfg.video_backend,
            transforms=transforms,
        )
        for p in cfg.dataset_path
    ]
    # Use first dataset; wrap multiple with ConcatDataset if needed
    from torch.utils.data import ConcatDataset
    train_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Policy (VLA frozen inside Gr00tRLTokenPolicy.__init__)
    # ------------------------------------------------------------------ #
    policy = Gr00tRLTokenPolicy(
        model_path=cfg.model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=device,
    )
    policy.rl_token_module.to(device)
    policy.rl_token_module.train()

    # ------------------------------------------------------------------ #
    # 3. Optimizer — only RL token parameters (+ optionally VLA)
    # ------------------------------------------------------------------ #
    param_groups = [
        {"params": list(policy.rl_token_module.parameters()), "lr": cfg.learning_rate},
    ]
    if cfg.vla_lr > 0:
        # Unfreeze VLA for joint training (paper's α > 0 variant)
        for p in policy.model.parameters():
            p.requires_grad_(True)
        param_groups.append(
            {"params": list(policy.model.parameters()), "lr": cfg.vla_lr}
        )

    optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps, eta_min=1e-6)

    # ------------------------------------------------------------------ #
    # 4. Training loop
    # ------------------------------------------------------------------ #
    data_iter = iter(dataloader)
    step = 0
    running_loss = 0.0

    print(f"Starting RL token training for {cfg.max_steps} steps...")

    while step < cfg.max_steps:
        # Refill iterator when exhausted
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move tensors to device
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        # rl_token_loss: runs backbone with no_grad, then trains encoder-decoder
        loss = policy.rl_token_loss(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.rl_token_module.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1

        if step % cfg.log_every == 0:
            avg = running_loss / cfg.log_every
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:>6d} | loss {avg:.6f} | lr {lr:.2e}")
            running_loss = 0.0

        if step % cfg.save_every == 0:
            ckpt_path = output_dir / f"rl_token_step{step}.pt"
            torch.save(
                {
                    "step": step,
                    "rl_token_module": policy.rl_token_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

    # Final save
    final_path = output_dir / "rl_token_final.pt"
    torch.save(
        {
            "step": step,
            "rl_token_module": policy.rl_token_module.state_dict(),
            "config": cfg.__dict__,
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
