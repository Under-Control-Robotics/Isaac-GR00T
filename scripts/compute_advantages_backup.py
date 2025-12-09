# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute advantages and generate binary indicators for RECAP/Pi-style training.

This script:
1. Loads a trained value function
2. Runs rollouts on all episodes to predict values
3. Computes TD-style advantages: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
   - Special handling for final step:
     * Success (reward=0): inherit advantage from previous step
     * Failure (reward<0): use negative final reward as advantage
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
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, DefaultDataCollator


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
    gamma: float = 0.01
    """Discount factor for TD advantage computation (very small as recommended)."""

    advantage_quantile: float = 0.7
    """Quantile threshold for binary indicators. 0.7 = top 30% marked as good."""

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


def compute_episode_advantages(
    model,
    dataset,
    data_collator,
    episode_idx: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute advantages for a single episode using trained value head.

    Computes TD-style advantages: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    Special handling for final timestep:
    - If success (reward ~ 0): inherit advantage from previous timestep
    - If failure (large negative reward): use negative final reward

    Args:
        model: GR00T model with trained value head
        dataset: Dataset with RL mode enabled
        data_collator: Data collator for batching
        episode_idx: Episode index
        gamma: Discount factor

    Returns:
        advantages: Array of shape (episode_length,) with advantages
        values: Array of shape (episode_length,) with predicted values
        rewards: Array of shape (episode_length,) with rewards
    """
    trajectory_id = dataset.trajectory_ids[episode_idx]
    episode_length = dataset.trajectory_lengths[episode_idx]
    episode_start_idx = sum(dataset.trajectory_lengths[:episode_idx])

    # Collect all values and rewards for the episode
    values_list = []
    rewards_list = []

    # Run value prediction on all timesteps (similar to visualize_rl_rollout.py)
    for step in range(episode_length):
        # Get data for this step (already transformed)
        step_data = dataset[episode_start_idx + step]

        # Batch the data using the same collator as training
        batched_data = data_collator([step_data])

        # Get value prediction
        with torch.no_grad():
            backbone_inputs, _ = model.prepare_input(batched_data)
            backbone_outputs = model.backbone(backbone_inputs)

            # Convert to action_head dtype and process through vlln
            backbone_outputs["backbone_features"] = backbone_outputs["backbone_features"].to(
                dtype=model.action_head.dtype
            )
            backbone_outputs = model.action_head.process_backbone_output(backbone_outputs)

            # Predict value
            value_pred = model.value_head(backbone_outputs["backbone_features"])
            value_scalar = value_pred[0, 0, 0].item()

        values_list.append(value_scalar)

        # Get reward for this timestep (first element of reward array)
        reward = (
            step_data["reward"][0] if len(step_data["reward"].shape) > 0 else step_data["reward"]
        )
        rewards_list.append(float(reward))

    values = np.array(values_list, dtype=np.float32)
    rewards = np.array(rewards_list, dtype=np.float32)

    # Compute TD-style advantages: A_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    advantages = np.zeros(episode_length, dtype=np.float32)

    for t in range(episode_length - 1):
        # Standard TD advantage computation
        advantages[t] = rewards[t] + gamma * values[t + 1] - values[t]

    # Special handling for final timestep:
    # - If success (reward close to 0): inherit from previous timestep
    # - If failure (large negative reward): use negative value
    final_reward = rewards[-1]
    if abs(final_reward) < 0.1:  # Success case (reward ~ 0)
        # Inherit advantage from previous timestep
        advantages[-1] = advantages[-2] if episode_length > 1 else 0.0
    else:  # Failure case (large negative reward)
        # Use the negative final reward as advantage
        advantages[-1] = final_reward

    return advantages, values, rewards


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

    # ------------ Step 3: Compute advantages using episode rollouts ------------
    print("\nComputing advantages using episode-by-episode rollouts...")
    print(f"Gamma: {config.gamma}")

    # Initialize data collator (same as training/visualization)
    data_collator = DefaultDataCollator()

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

        num_episodes = len(single_dataset.trajectory_ids)
        print(f"Number of episodes: {num_episodes}")

        # Store per-episode results
        episode_advantages_dict = {}
        episode_values_dict = {}
        episode_rewards_dict = {}

        # Process each episode
        for ep_idx in tqdm(range(num_episodes), desc="Computing advantages"):
            try:
                advantages, values, rewards = compute_episode_advantages(
                    model, single_dataset, data_collator, ep_idx, config.gamma
                )

                trajectory_id = single_dataset.trajectory_ids[ep_idx]
                episode_advantages_dict[trajectory_id] = advantages
                episode_values_dict[trajectory_id] = values
                episode_rewards_dict[trajectory_id] = rewards

            except Exception as e:
                print(f"\nError processing episode {ep_idx}:")
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Flatten all advantages for statistics
        advantages_flat = np.concatenate([adv for adv in episode_advantages_dict.values()])

        print(f"\nAdvantage statistics:")
        print(f"  Range: [{advantages_flat.min():.3f}, {advantages_flat.max():.3f}]")
        print(f"  Mean: {advantages_flat.mean():.3f}")
        print(f"  Std: {advantages_flat.std():.3f}")

        # ------------ Step 5: Compute threshold ------------
        print(f"\nComputing {config.advantage_quantile:.0%} quantile threshold...")
        threshold = np.quantile(advantages_flat, config.advantage_quantile)
        print(f"Threshold: {threshold:.3f}")
        print(f"This means actions with advantage > {threshold:.3f} are labeled as 'good' (I_t=1)")

        # ------------ Step 4: Generate binary indicators ------------
        print("\nGenerating binary indicators...")

        # Compute indicators per episode
        episode_indicators_dict = {}
        for traj_id, advantages in episode_advantages_dict.items():
            indicators = (advantages >= threshold).astype(np.float32)
            episode_indicators_dict[traj_id] = indicators

        # Compute overall good ratio
        all_indicators = np.concatenate([ind for ind in episode_indicators_dict.values()])
        good_ratio = all_indicators.mean()
        print(f"Good actions (I_t=1): {good_ratio:.1%}")
        print(f"Bad actions (I_t=0): {(1-good_ratio):.1%}")

        # ------------ Step 5: Save indicators to file ------------
        print("\nSaving indicators...")

        # Prepare episode data
        episode_data_list = []
        for traj_id in sorted(episode_advantages_dict.keys()):
            episode_data = {
                "episode_index": int(traj_id),
                "length": int(len(episode_advantages_dict[traj_id])),
                "indicators": episode_indicators_dict[traj_id].tolist(),
                "advantages": episode_advantages_dict[traj_id].tolist(),
                "values": episode_values_dict[traj_id].tolist(),
                "rewards": episode_rewards_dict[traj_id].tolist(),
            }
            episode_data_list.append(episode_data)

        # Save to JSON
        output_data = {
            "metadata": {
                "model_path": config.model_path,
                "dataset_path": single_dataset.dataset_path.as_posix(),
                "gamma": config.gamma,
                "advantage_quantile": config.advantage_quantile,
                "threshold": float(threshold),
                "total_steps": int(len(advantages_flat)),
                "good_ratio": float(good_ratio),
            },
            "episodes": episode_data_list,
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

    # ------------ Step 6: Print summary statistics ------------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(datasets_to_process)} dataset(s)")
    print(f"Gamma (discount factor): {config.gamma}")
    print(f"Advantage quantile: {config.advantage_quantile:.0%}")

    if len(all_advantages_per_dataset) > 0:
        all_advantages = np.concatenate(all_advantages_per_dataset)
        print(f"\nOverall advantage statistics:")
        print(f"  Mean: {all_advantages.mean():.3f}")
        print(f"  Std: {all_advantages.std():.3f}")
        print(f"  Min: {all_advantages.min():.3f}")
        print(f"  Max: {all_advantages.max():.3f}")
        print(f"  Threshold: {np.quantile(all_advantages, config.advantage_quantile):.3f}")

    print("\n" + "=" * 80)
    print("DONE! Advantage labels saved.")
    print(
        "Now you can train an advantage-conditioned policy using enable_advantage_conditioning=True"
    )
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
