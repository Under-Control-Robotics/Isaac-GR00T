# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute advantages and generate binary indicators for RECAP/Pi-style training.

This script:
1. Loads a trained value function
2. Predicts values for all timesteps in the dataset
3. Computes advantages: A_t = G_norm[t] - V_pred[t]
4. Computes task-specific threshold (e.g., 70th percentile)
5. Generates binary indicators: I_t = 1 if A_t > threshold else 0
6. Saves the indicators to advantage_labels.json
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import torch
import tyro
from PIL import Image
from tqdm import tqdm
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotMixtureDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class AdvantageComputeConfig:
    """Configuration for computing advantages and generating indicators."""

    # Model parameters
    model_path: str
    """Path to the trained model with value function."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories with reward labels."""

    dataset_language_prompts: List[str] | None = None
    """Optional language prompt override for each dataset."""

    data_config: str = "fourier_gr1_arms_only"
    """Data configuration to use."""

    # Output parameters
    output_dir: str = "/tmp/gr00t_advantages"
    """Directory to save advantage labels."""

    # Advantage parameters
    advantage_quantile: float = 0.5
    """Quantile threshold for binary indicators. 0.5 = top 50%, 0.7 = top 30%."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"
    """Video backend to use."""

    batch_size: int = 32
    """Batch size for inference."""

    num_workers: int = 4
    """Number of workers for data loading."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """If True, balance dataset weights."""

    balance_trajectory_weights: bool = True
    """If True, sample trajectories weighted by length."""


def collate_fn_with_pil(batch):
    """Custom collate function that handles PIL Images by converting them to tensors."""
    import torchvision.transforms.functional as F

    # Convert PIL Images to tensors in the batch
    def process_item(item):
        if item is None:
            raise ValueError("Found None value in batch - check dataset integrity")
        elif isinstance(item, Image.Image):
            # Convert PIL Image to tensor (C, H, W) and normalize to [0, 1]
            return F.to_tensor(item)
        elif isinstance(item, dict):
            return {k: process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [process_item(v) for v in item]
        elif isinstance(item, tuple):
            return tuple(process_item(v) for v in item)
        else:
            return item

    # Filter out any None samples in the batch
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in batch are None")

    processed_batch = [process_item(sample) for sample in batch]

    # Now use the default collate
    return torch.utils.data.dataloader.default_collate(processed_batch)


def main(config: AdvantageComputeConfig):
    """Main function to compute advantages and generate indicators."""

    print("\n" + "=" * 80)
    print("COMPUTING ADVANTAGES AND GENERATING INDICATORS")
    print("=" * 80 + "\n")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------ Step 1: Load model with trained value function ------------
    print("Loading model with trained value function...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.model_path,
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        tune_value_head=False,  # Freeze for inference
        enable_rl=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    # ------------ Step 2: Load dataset with RL mode enabled ------------
    print("\nLoading dataset...")
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    if len(config.dataset_path) == 1:
        language_prompt = (
            config.dataset_language_prompts[0] if config.dataset_language_prompts else None
        )
        dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            language_override=language_prompt,
            enable_rl=True,
        )
    else:
        single_datasets = []
        for idx, p in enumerate(config.dataset_path):
            language_prompt = (
                config.dataset_language_prompts[idx] if config.dataset_language_prompts else None
            )
            dataset_single = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                language_override=language_prompt,
                enable_rl=True,
            )
            single_datasets.append(dataset_single)

        dataset = LeRobotMixtureDataset(
            data_mixture=[(ds, 1.0) for ds in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )

    print(f"Dataset loaded: {len(dataset)} steps")

    # Validate that reward labels exist for all datasets
    if isinstance(dataset, LeRobotMixtureDataset):
        datasets_to_validate = dataset.datasets
    else:
        datasets_to_validate = [dataset]

    for ds in datasets_to_validate:
        reward_labels_path = ds.dataset_path / "reward_labels.json"
        if not reward_labels_path.exists():
            raise FileNotFoundError(
                f"reward_labels.json not found for dataset {ds.dataset_name} at {reward_labels_path}. "
                f"You must first generate reward labels before computing advantages. "
                f"Please run the reward labeling script first."
            )

        # Validate reward labels file integrity
        with open(reward_labels_path, "r") as f:
            reward_data = json.load(f)

        if "episodes" not in reward_data:
            raise ValueError(
                f"reward_labels.json for {ds.dataset_name} is malformed: missing 'episodes' key"
            )

        # Check that all episodes in the dataset have reward labels
        labeled_episodes = {ep["episode_index"] for ep in reward_data["episodes"]}
        dataset_episodes = set(ds.trajectory_ids)
        missing_episodes = dataset_episodes - labeled_episodes

        if missing_episodes:
            raise ValueError(
                f"reward_labels.json for {ds.dataset_name} is incomplete. "
                f"Missing labels for {len(missing_episodes)} episodes: {list(missing_episodes)[:10]}... "
                f"Total episodes in dataset: {len(dataset_episodes)}, Labeled: {len(labeled_episodes)}"
            )

        print(f"✓ Verified reward labels for {ds.dataset_name} ({len(labeled_episodes)} episodes)")

    # ------------ Step 3: Predict values for all timesteps ------------
    print("\nPredicting values for all timesteps...")

    # Store results per dataset
    if isinstance(dataset, LeRobotMixtureDataset):
        datasets_to_process = dataset.datasets
    else:
        datasets_to_process = [dataset]

    all_advantages_per_dataset = []

    for ds_idx, single_dataset in enumerate(datasets_to_process):
        print(
            f"\nProcessing dataset {ds_idx + 1}/{len(datasets_to_process)}: {single_dataset.dataset_name}"
        )

        # Create dataloader for this dataset
        dataloader = torch.utils.data.DataLoader(
            single_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_fn_with_pil,
        )

        # Collect predictions
        all_values_pred = []
        all_values_target = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Predicting values")):
                try:
                    # Prepare inputs
                    inputs = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            inputs[key] = value.to(device)
                        else:
                            inputs[key] = value

                    # Verify required keys exist
                    if "value" not in batch:
                        raise KeyError(
                            f"Batch {batch_idx} missing 'value' key. "
                            f"Available keys: {list(batch.keys())}. "
                            f"Make sure reward_labels.json exists and contains all episodes."
                        )

                    # Get backbone features
                    backbone_inputs, _ = model.prepare_input(inputs)
                    backbone_outputs = model.backbone(backbone_inputs)

                    # Convert to action_head dtype BEFORE process_backbone_output
                    # This is needed because vlln expects bfloat16
                    backbone_outputs["backbone_features"] = backbone_outputs[
                        "backbone_features"
                    ].to(dtype=model.action_head.dtype)

                    # IMPORTANT: Process backbone output through vlln + vl_self_attention
                    # This matches the training flow where action_head.forward() modifies backbone_outputs
                    backbone_outputs = model.action_head.process_backbone_output(backbone_outputs)
                    backbone_features = backbone_outputs["backbone_features"]

                    # Predict single state value
                    value_pred = model.value_head(backbone_features)  # (B, 1, 1)

                    # Get target values (G_norm) from the batch - take first timestep only
                    value_target = batch["value"]  # (B, action_horizon)

                    # Take only the first timestep value (current state V(s_t))
                    if value_target.dim() > 1 and value_target.size(1) > 1:
                        value_target = value_target[:, 0]  # (B,)

                    # Convert to tensor if needed
                    if not isinstance(value_target, torch.Tensor):
                        value_target = torch.tensor(
                            value_target, dtype=value_pred.dtype, device=device
                        )

                    # Store predictions - squeeze to (B,)
                    all_values_pred.append(value_pred.squeeze(-1).squeeze(-1).cpu().numpy())  # (B,)
                    all_values_target.append(value_target.cpu().numpy())  # (B,)

                except Exception as e:
                    print(f"\nError processing batch {batch_idx}:")
                    print(f"  Error: {e}")
                    print(
                        f"  Batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'not a dict'}"
                    )
                    print(f"  Dataset: {single_dataset.dataset_name}")
                    raise

        # Concatenate all batches
        all_values_pred = np.concatenate(all_values_pred, axis=0)  # (N,) - one value per step
        all_values_target = np.concatenate(all_values_target, axis=0)  # (N,) - one value per step

        print(f"Predicted values for {len(all_values_pred)} steps")
        print(f"Value pred range: [{all_values_pred.min():.3f}, {all_values_pred.max():.3f}]")
        print(f"Value target range: [{all_values_target.min():.3f}, {all_values_target.max():.3f}]")

        # ------------ Step 4: Compute advantages ------------
        print("\nComputing advantages...")
        # A_t = G_norm[t] - V_pred[t]
        advantages = all_values_target - all_values_pred  # (N,) - one advantage per step

        print(f"Advantage range: [{advantages.min():.3f}, {advantages.max():.3f}]")
        print(f"Advantage mean: {advantages.mean():.3f}, std: {advantages.std():.3f}")

        # Use advantages directly for threshold computation (already flat)
        advantages_flat = advantages

        # ------------ Step 5: Compute threshold ------------
        print(f"\nComputing {config.advantage_quantile:.0%} quantile threshold...")
        threshold = np.quantile(advantages_flat, config.advantage_quantile)
        print(f"Threshold: {threshold:.3f}")
        print(f"This means actions with advantage > {threshold:.3f} are labeled as 'good' (I_t=1)")

        # ------------ Step 6: Generate binary indicators ------------
        print("\nGenerating binary indicators...")
        indicators = (advantages > threshold).astype(np.float32)  # (N,) - one indicator per step

        good_ratio = indicators.mean()
        print(f"Good actions (I_t=1): {good_ratio:.1%}")
        print(f"Bad actions (I_t=0): {(1-good_ratio):.1%}")

        # ------------ Step 7: Save indicators to file ------------
        print("\nSaving indicators...")

        # Reshape back to per-episode format (one value per step in episode)
        episode_indicators = []
        step_idx = 0
        for ep_idx, traj_len in enumerate(single_dataset.trajectory_lengths):
            episode_data = {
                "episode_index": int(single_dataset.trajectory_ids[ep_idx]),
                "length": int(traj_len),
                "indicators": [float(indicators[step_idx + i]) for i in range(traj_len)],
                "advantages": [float(advantages[step_idx + i]) for i in range(traj_len)],
            }
            episode_indicators.append(episode_data)
            step_idx += traj_len

        # Save to JSON
        output_data = {
            "metadata": {
                "model_path": config.model_path,
                "dataset_path": single_dataset.dataset_path.as_posix(),
                "advantage_quantile": config.advantage_quantile,
                "threshold": float(threshold),
                "total_steps": int(len(indicators)),
                "good_ratio": float(good_ratio),
            },
            "episodes": episode_indicators,
        }

        output_file = (
            Path(config.output_dir) / f"advantage_labels_{single_dataset.dataset_name}.json"
        )
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Saved advantage labels to: {output_file}")

        # Also save to the dataset directory
        dataset_output_file = single_dataset.dataset_path / "advantage_labels.json"
        with open(dataset_output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Also saved to dataset directory: {dataset_output_file}")

        all_advantages_per_dataset.append(advantages_flat)

    # ------------ Step 8: Print summary statistics ------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(datasets_to_process)} dataset(s)")
    print(f"Advantage quantile: {config.advantage_quantile:.0%}")

    all_advantages = np.concatenate(all_advantages_per_dataset)
    print(f"\nOverall advantage statistics:")
    print(f"  Mean: {all_advantages.mean():.3f}")
    print(f"  Std: {all_advantages.std():.3f}")
    print(f"  Min: {all_advantages.min():.3f}")
    print(f"  Max: {all_advantages.max():.3f}")
    print(f"  Threshold: {np.quantile(all_advantages, config.advantage_quantile):.3f}")

    print("\n" + "=" * 80)
    print("DONE! Now you can train an advantage-conditioned policy using:")
    print(f"  python scripts/gr00t_recap_train.py --dataset_path <path> ...")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    config = tyro.cli(AdvantageComputeConfig)

    print("\n" + "=" * 50)
    print("ADVANTAGE COMPUTATION CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    main(config)
