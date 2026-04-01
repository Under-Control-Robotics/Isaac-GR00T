"""
Phase 1 training: RL Token encoder-decoder.

Trains only RLTokenModule parameters while the VLA backbone is completely frozen.
The reconstruction objective forces the RL token to be a compressed but
information-preserving summary of the VLA's final-layer token embeddings.

Usage:
    python scripts/train_rl_token.py \
        --dataset_path $(cat datasets.txt) \
        --model_path nvidia/GR00T-N1.5-3B \
        --embodiment_tag new_embodiment \
        --data_config ucr_wblm_moby_history \
        --output_dir /tmp/rl_token_ckpt
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.rl_policy import Gr00tRLTokenPolicy
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator


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
    max_steps: int = 10000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    num_workers: int = 4
    log_every: int = 50
    save_every: int = 1000

    # Optional: also fine-tune the VLA jointly (alpha > 0 in paper Eq. 3)
    # Keep 0 unless you want to combine SFT + RL token in one pass.
    vla_lr: float = 0.0

    # --- logging ---
    wandb_project: str = "gr00t-rl-token"
    wandb_run_name: str = ""


collate_fn = DefaultDataCollator()


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
    # 2. Write metadata so Gr00tPolicy._load_metadata can find it.
    #    (mirrors what TrainRunner does before training)
    # ------------------------------------------------------------------ #
    # Resolve HuggingFace model id → local cache path (or keep local path)
    try:
        from huggingface_hub import snapshot_download
        resolved_model_path = Path(snapshot_download(cfg.model_path, repo_type="model"))
    except Exception:
        resolved_model_path = Path(cfg.model_path)

    exp_cfg_dir = resolved_model_path / "experiment_cfg"
    exp_cfg_dir.mkdir(exist_ok=True)
    metadata_path = exp_cfg_dir / "metadata.json"
    metadata_json = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if embodiment_tag.value not in metadata_json:
        metadata_json[embodiment_tag.value] = datasets[0].metadata.model_dump(mode="json")
        metadata_path.write_text(json.dumps(metadata_json, indent=4))
        print(f"Wrote {embodiment_tag.value} metadata to {metadata_path}")

    # ------------------------------------------------------------------ #
    # 3. Policy (VLA frozen inside Gr00tRLTokenPolicy.__init__)
    # ------------------------------------------------------------------ #
    policy = Gr00tRLTokenPolicy(
        model_path=cfg.model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=device,
    )
    policy.rl_token_module.to(device)  # keep float32 — backbone features cast to float32 at boundary
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
    running_loss = running_recon = running_var = 0.0

    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name or None,
        config=cfg.__dict__,
    )

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

        loss, recon_loss, var_loss = policy.rl_token_loss(batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(policy.rl_token_module.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_recon += recon_loss.item()
        running_var += var_loss.item()
        step += 1

        if step % cfg.log_every == 0:
            avg = running_loss / cfg.log_every
            avg_recon = running_recon / cfg.log_every
            avg_var = running_var / cfg.log_every
            lr = scheduler.get_last_lr()[0]
            print(f"step {step:>6d} | loss {avg:.6f} | recon {avg_recon:.6f} | var {avg_var:.6f} | lr {lr:.2e}")
            wandb.log({
                "train/loss": avg,
                "train/recon_loss": avg_recon,
                "train/var_loss": avg_var,
                "train/lr": lr,
            }, step=step)
            running_loss = running_recon = running_var = 0.0

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
    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
