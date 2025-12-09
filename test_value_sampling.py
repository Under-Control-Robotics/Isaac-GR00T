"""Test script to check value distribution in training data."""

import sys
import numpy as np
import torch
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config

# Load dataset
data_config_cls = load_data_config("ucr_wblm_moby_history")
modality_configs = data_config_cls.modality_config()
transforms = data_config_cls.transform()

dataset = LeRobotSingleDataset(
    dataset_path="/data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_0",
    modality_configs=modality_configs,
    transforms=transforms,
    embodiment_tag=EmbodimentTag("new_embodiment"),
    video_backend="decord",
    enable_rl=True,
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of trajectories: {len(dataset.trajectory_ids)}")
print(f"Trajectory lengths: {dataset.trajectory_lengths}")
print()

# Sample 100 random steps and check value distribution
np.random.seed(42)
sampled_values = []

print("Sampling 100 random steps...")
for _ in range(100):
    idx = np.random.randint(0, len(dataset))
    data = dataset[idx]

    # Get the first value (current state value)
    if "value" in data:
        value = data["value"]
        # Take first timestep value
        if isinstance(value, torch.Tensor):
            first_value = value[0].item() if value.dim() > 0 else value.item()
        else:
            first_value = value[0] if hasattr(value, "__len__") else value
        sampled_values.append(first_value)

sampled_values = np.array(sampled_values)

print(f"\nSampled value statistics (n={len(sampled_values)}):")
print(f"  Min: {sampled_values.min():.4f}")
print(f"  Max: {sampled_values.max():.4f}")
print(f"  Mean: {sampled_values.mean():.4f}")
print(f"  Std: {sampled_values.std():.4f}")
print(f"  Median: {np.median(sampled_values):.4f}")
print(f"  25th percentile: {np.percentile(sampled_values, 25):.4f}")
print(f"  75th percentile: {np.percentile(sampled_values, 75):.4f}")

# Check distribution in bins
bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]
hist, _ = np.histogram(sampled_values, bins=bins)
print(f"\nValue distribution:")
for i in range(len(bins) - 1):
    print(
        f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} samples ({100*hist[i]/len(sampled_values):.1f}%)"
    )

print("\n" + "=" * 60)
if sampled_values.min() > -0.5:
    print("⚠️  WARNING: Sampling appears biased toward late timesteps (values near 0)")
    print("   This explains why the model learns a compressed value range.")
elif sampled_values.max() < -0.5:
    print("⚠️  WARNING: Sampling appears biased toward early timesteps (values near -1)")
else:
    print("✓  Sampling appears unbiased across full value range [-1, 0]")
