"""
Evaluate a trained RLTokenModule checkpoint.

Runs three diagnostics to determine whether z_rl actually encodes
task-relevant information or whether the decoder ignores it:

  1. Normal recon loss   — encoder → z_rl → decoder → MSE
  2. Null-z_rl loss      — z_rl = 0 → decoder → MSE
                           If ≈ normal, decoder ignores z_rl entirely.
  3. Random-z_rl loss    — z_rl ~ N(0,1) → decoder → MSE
                           Sanity check; should be ≥ null loss if decoder uses z_rl.

  Interpretation:
    null_loss / recon_loss >> 1  →  decoder relies on z_rl  ✓
    null_loss / recon_loss ≈  1  →  decoder ignores z_rl   ✗  (positional memorisation)

Usage:
    python scripts/eval_rl_token.py \
        --ckpt_path /tmp/rl_token_ckpt/rl_token_step5000.pt \
        --dataset_path $(cat datasets.txt) \
        --model_path nvidia/GR00T-N1.5-3B \
        --embodiment_tag new_embodiment \
        --data_config ucr_wblm_moby_history
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.rl_policy import Gr00tRLTokenPolicy
from gr00t.model.rl_token import RLTokenModule
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator
from torch.utils.data import DataLoader


@dataclass
class EvalConfig:
    ckpt_path: str
    """Path to a .pt checkpoint saved by train_rl_token.py"""

    dataset_path: List[str]
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    data_config: str = "fourier_gr1_arms_only"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"

    num_batches: int = 20
    """How many batches to average the metrics over."""
    batch_size: int = 8


def move_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main(cfg: EvalConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # 1. Load dataset
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
    from torch.utils.data import ConcatDataset
    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=DefaultDataCollator(),
        drop_last=True,
    )

    # ------------------------------------------------------------------ #
    # 2. Load policy (backbone) + RLTokenModule from checkpoint
    # ------------------------------------------------------------------ #
    # Resolve metadata the same way train_rl_token.py does
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
        ds0 = datasets[0]
        metadata_json[embodiment_tag.value] = ds0.metadata.model_dump(mode="json")
        metadata_path.write_text(json.dumps(metadata_json, indent=4))

    policy = Gr00tRLTokenPolicy(
        model_path=cfg.model_path,
        embodiment_tag=embodiment_tag,
        modality_config=modality_configs,
        modality_transform=transforms,
        device=device,
    )
    policy.model.eval()

    # Load trained RLTokenModule weights
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    policy.rl_token_module.load_state_dict(ckpt["rl_token_module"])
    policy.rl_token_module.to(device).eval()

    rl_mod: RLTokenModule = policy.rl_token_module
    rl_token_dim = rl_mod.rl_token_dim

    # ------------------------------------------------------------------ #
    # 3. Evaluate
    # ------------------------------------------------------------------ #
    normal_losses, null_losses, rand_losses = [], [], []
    zrl_stds = []

    data_iter = iter(loader)
    for i in range(cfg.num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = move_to_device(batch, device)

        with torch.no_grad():
            # Backbone features (float32)
            backbone_out = policy.model.get_backbone_features(batch)
            feats = backbone_out["backbone_features"].float()
            mask  = backbone_out.get("backbone_attention_mask")

            z_targets = rl_mod._normalize_tokens(feats)
            M = z_targets.size(1)
            B = z_targets.size(0)

            # --- normal: encode → decode ---
            z_rl = rl_mod.encode(z_targets, mask)
            preds_normal = rl_mod.decode(z_rl, seq_len=M)

            # --- null: zero z_rl → decode ---
            z_rl_null = torch.zeros(B, rl_token_dim, device=device)
            preds_null = rl_mod.decode(z_rl_null, seq_len=M)

            # --- random: random z_rl → decode ---
            z_rl_rand = torch.randn(B, rl_token_dim, device=device)
            preds_rand = rl_mod.decode(z_rl_rand, seq_len=M)

            if mask is not None:
                m = mask.bool()
                normal_loss = torch.nn.functional.mse_loss(preds_normal[m], z_targets[m])
                null_loss   = torch.nn.functional.mse_loss(preds_null[m],   z_targets[m])
                rand_loss   = torch.nn.functional.mse_loss(preds_rand[m],   z_targets[m])
            else:
                normal_loss = torch.nn.functional.mse_loss(preds_normal, z_targets)
                null_loss   = torch.nn.functional.mse_loss(preds_null,   z_targets)
                rand_loss   = torch.nn.functional.mse_loss(preds_rand,   z_targets)

            normal_losses.append(normal_loss.item())
            null_losses.append(null_loss.item())
            rand_losses.append(rand_loss.item())
            zrl_stds.append(z_rl.std(dim=0).mean().item())

        print(f"batch {i+1:>3d}/{cfg.num_batches} | "
              f"normal {normal_loss.item():.6f} | "
              f"null {null_loss.item():.6f} | "
              f"rand {rand_loss.item():.6f} | "
              f"null/normal {null_loss.item()/normal_loss.item():.2f}x | "
              f"z_rl_std {z_rl.std(dim=0).mean().item():.4f}")

    # ------------------------------------------------------------------ #
    # 4. Summary
    # ------------------------------------------------------------------ #
    avg_normal = sum(normal_losses) / len(normal_losses)
    avg_null   = sum(null_losses)   / len(null_losses)
    avg_rand   = sum(rand_losses)   / len(rand_losses)
    avg_std    = sum(zrl_stds)      / len(zrl_stds)
    ratio      = avg_null / avg_normal

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"  avg normal recon loss : {avg_normal:.6f}")
    print(f"  avg null-z_rl loss    : {avg_null:.6f}")
    print(f"  avg rand-z_rl loss    : {avg_rand:.6f}")
    print(f"  zero-predictor baseline : {1/1536:.6f}  (unit vectors in R^1536)")
    print(f"  null / normal ratio   : {ratio:.2f}x")
    print(f"  avg z_rl std (across dims) : {avg_std:.4f}")
    print()
    if ratio > 2.0:
        print("✓  PASS — decoder relies on z_rl (null loss is significantly worse).")
    elif ratio > 1.2:
        print("~  MARGINAL — decoder uses z_rl weakly. Consider stronger bottleneck.")
    else:
        print("✗  FAIL — decoder ignores z_rl (null loss ≈ normal loss).")
        print("   The RL token does not carry task-relevant information.")
    print("=" * 60)


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
