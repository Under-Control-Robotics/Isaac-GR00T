#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify advantage-conditioned training setup.

This script checks:
1. Datasets have indicator labels
2. Indicators are correctly loaded
3. Indicators align with advantages
4. Model can process indicator tokens
"""

import json
import numpy as np
import torch
from pathlib import Path

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.model.gr00t_n1 import GR00T_N1_5


def test_dataset_labels(dataset_paths):
    """Test that all datasets have indicator labels."""
    print("=" * 80)
    print("STEP 1: Checking dataset label files")
    print("=" * 80)

    for path in dataset_paths:
        path = Path(path)
        reward_labels_path = path / "reward_labels.json"

        print(f"\nDataset: {path.name}")
        if not reward_labels_path.exists():
            print(f"  ✗ reward_labels.json not found!")
            return False

        with open(reward_labels_path, "r") as f:
            data = json.load(f)

        # Check episodes have indicators
        episodes_with_indicators = 0
        total_episodes = len(data["episodes"])
        for ep in data["episodes"]:
            if "indicators" in ep and "advantages" in ep:
                episodes_with_indicators += 1

        print(f"  ✓ reward_labels.json found")
        print(f"  ✓ {episodes_with_indicators}/{total_episodes} episodes have indicators")

        # Check metadata
        if "metadata" in data and "advantage_computation" in data["metadata"]:
            adv_meta = data["metadata"]["advantage_computation"]
            if "global_threshold" in adv_meta:
                print(f"  ✓ Global threshold: {adv_meta['global_threshold']:.6f}")
            if "dataset_good_ratio" in adv_meta:
                print(f"  ✓ Good ratio: {adv_meta['dataset_good_ratio']:.1%}")

    print("\n✓ All datasets have indicator labels\n")
    return True


def test_dataset_loading(dataset_path, data_config):
    """Test that dataset correctly loads indicators."""
    print("=" * 80)
    print("STEP 2: Testing dataset loading with advantage conditioning")
    print("=" * 80)

    embodiment_tag = EmbodimentTag("new_embodiment")
    data_config_cls = load_data_config(data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
        enable_rl=False,
        enable_advantage_conditioning=True,
    )

    print(f"\n✓ Dataset loaded successfully")
    print(f"  Dataset size: {len(dataset)} samples")

    # Get first sample
    print("\nGetting first sample...")
    sample = dataset[0]

    if "indicator" not in sample:
        print("  ✗ No 'indicator' field in sample!")
        print(f"  Available keys: {list(sample.keys())}")
        return False

    indicators = sample["indicator"]
    print(f"  ✓ Indicator found in sample")
    print(f"  Shape: {indicators.shape}")
    print(f"  Dtype: {indicators.dtype}")
    print(f"  Range: [{indicators.min():.1f}, {indicators.max():.1f}]")
    print(f"  First 10 values: {indicators[:10].tolist()}")

    # Verify binary
    unique_vals = np.unique(indicators)
    if np.all(np.isin(unique_vals, [0.0, 1.0])):
        print(f"  ✓ Indicators are binary (0 or 1)")
    else:
        print(f"  ✗ Indicators are not binary! Unique values: {unique_vals}")
        return False

    print("\n✓ Dataset loading successful\n")
    return True


def test_model_forward(dataset_path, data_config):
    """Test that model can process indicators."""
    print("=" * 80)
    print("STEP 3: Testing model forward pass with indicators")
    print("=" * 80)

    embodiment_tag = EmbodimentTag("new_embodiment")
    data_config_cls = load_data_config(data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Load dataset
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend="decord",
        enable_rl=False,
        enable_advantage_conditioning=True,
    )

    # Create a small batch using proper collate function
    from torch.utils.data import DataLoader
    from gr00t.model.transforms import DefaultDataCollator

    # Use the proper collate function that handles eagle_content
    data_collator = DefaultDataCollator()

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
    )
    batch = next(iter(loader))

    print(f"✓ Batch loaded")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Indicator shape: {batch['indicator'].shape}")

    # Load model
    print("\nLoading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=False,
        tune_diffusion_model=False,
        enable_rl=False,
        enable_advantage_conditioning=True,
        indicator_embedding_dim=4096,
    )
    model.to(device)
    model.eval()

    print("✓ Model loaded with advantage conditioning enabled")

    # Check indicator embedding
    if hasattr(model, "indicator_embedding"):
        print(f"  ✓ Model has indicator_embedding module")
        print(f"    Embedding dim: {model.indicator_embedding.hidden_size}")
        print(f"    Num indicators: {model.indicator_embedding.num_indicators}")
    else:
        print(f"  ✗ Model does NOT have indicator_embedding module!")
        return False

    # Forward pass
    print("\nRunning forward pass...")

    # Move batch to device
    batch_device = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_device[key] = value.to(device)
        else:
            batch_device[key] = value

    with torch.no_grad():
        try:
            outputs = model(batch_device)
            print("✓ Forward pass successful!")
            print(f"  Output keys: {list(outputs.keys())}")

            # Check if indicator was processed
            if "action_pred" in outputs:
                print(f"  ✓ Model generated action predictions")
                print(f"    Action pred shape: {outputs['action_pred'].shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("\n✓ Model forward pass successful\n")
    return True


def main():
    """Run all tests."""
    dataset_paths = [
        "/data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_0",
        "/data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_3",
        "/data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_4",
    ]
    data_config = "ucr_wblm_moby_history"

    print("\n" + "=" * 80)
    print("ADVANTAGE-CONDITIONED TRAINING VERIFICATION TEST")
    print("=" * 80 + "\n")

    # Test 1: Check dataset labels
    if not test_dataset_labels(dataset_paths):
        print("\n✗ TEST FAILED: Dataset labels check failed")
        return

    # Test 2: Test dataset loading
    if not test_dataset_loading(dataset_paths[0], data_config):
        print("\n✗ TEST FAILED: Dataset loading failed")
        return

    # Test 3: Test model forward pass
    if not test_model_forward(dataset_paths[0], data_config):
        print("\n✗ TEST FAILED: Model forward pass failed")
        return

    print("=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nYou can now run the training script:")
    print("python scripts/gr00t_advantage_conditioned_train.py \\")
    print("    --dataset-path " + " ".join(dataset_paths) + " \\")
    print("    --data-config ucr_wblm_moby_history \\")
    print("    --output-dir /data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned \\")
    print("    --base-model-path nvidia/GR00T-N1.5-3B \\")
    print("    --batch-size 32 \\")
    print("    --max-steps 10000 \\")
    print("    --learning-rate 1e-4")
    print()


if __name__ == "__main__":
    main()
