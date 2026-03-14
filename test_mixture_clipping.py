#!/usr/bin/env python3
"""Test script to verify error recovery dataset clipping works with mixture datasets."""

import sys
from pathlib import Path
import numpy as np

# Add the gr00t module to path
sys.path.insert(0, str(Path(__file__).parent))

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotMixtureDataset
from gr00t.experiment.data_config import UCRWBLMMobyHistoryDataConfig


def test_mixture_dataset_clipping():
    """Test that error recovery clipping works with mixture datasets and upsampling."""

    # Get data config
    data_config = UCRWBLMMobyHistoryDataConfig()
    modality_configs = data_config.modality_config()

    # Load datasets
    normal_path = "/data/anthony/ucr_ros/data_files/dataset/normal_data/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-19_20:53:47.501581"
    error_recovery_path = "/data/anthony/ucr_ros/data_files/dataset/error_recovery/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-27_08:14:08.248872"

    print(f"\n{'='*80}")
    print("Loading datasets...")
    print(f"{'='*80}")

    normal_dataset = LeRobotSingleDataset(
        dataset_path=normal_path,
        modality_configs=modality_configs,
        embodiment_tag="new_embodiment",
    )

    error_dataset = LeRobotSingleDataset(
        dataset_path=error_recovery_path,
        modality_configs=modality_configs,
        embodiment_tag="new_embodiment",
    )

    print(f"\nNormal dataset: {len(normal_dataset)} steps")
    print(f"Error recovery dataset: {len(error_dataset)} steps (clipped)")

    # Test 1: Verify get_valid_indices_for_trajectory works correctly
    print(f"\n{'='*80}")
    print("Test 1: Verify get_valid_indices_for_trajectory()")
    print(f"{'='*80}")

    # For normal dataset, should return full range
    normal_traj_id = normal_dataset.trajectory_ids[0]
    normal_valid_indices = normal_dataset.get_valid_indices_for_trajectory(normal_traj_id)
    normal_traj_len = normal_dataset.trajectory_lengths[0]

    print(f"\nNormal dataset trajectory {normal_traj_id}:")
    print(f"  Length: {normal_traj_len}")
    print(
        f"  Valid indices: [{normal_valid_indices[0]}...{normal_valid_indices[-1]}] ({len(normal_valid_indices)} total)"
    )
    assert (
        len(normal_valid_indices) == normal_traj_len
    ), "Normal dataset should have all indices valid"
    assert normal_valid_indices[0] == 0, "Normal dataset should start at 0"
    assert normal_valid_indices[-1] == normal_traj_len - 1, "Normal dataset should end at len-1"

    # For error recovery dataset, should return clipped range
    error_traj_id = error_dataset.trajectory_ids[0]
    error_valid_indices = error_dataset.get_valid_indices_for_trajectory(error_traj_id)
    error_traj_len = error_dataset.trajectory_lengths[0]

    print(f"\nError recovery dataset trajectory {error_traj_id}:")
    print(f"  Length: {error_traj_len}")
    print(
        f"  Valid indices: [{error_valid_indices[0]}...{error_valid_indices[-1]}] ({len(error_valid_indices)} total)"
    )
    assert (
        error_valid_indices[0] == 30
    ), f"Error recovery should start at 30, got {error_valid_indices[0]}"
    assert (
        error_valid_indices[-1] == error_traj_len - 61
    ), f"Error recovery should end at {error_traj_len - 61}, got {error_valid_indices[-1]}"

    print("\n✓ get_valid_indices_for_trajectory() works correctly!")

    # Test 2: Create mixture dataset and verify sampling
    print(f"\n{'='*80}")
    print("Test 2: Verify mixture dataset sampling respects clipping")
    print(f"{'='*80}")

    # Create mixture with 10x upsampling for error recovery
    mixture_dataset = LeRobotMixtureDataset(
        data_mixture=[
            (normal_dataset, 1.0),
            (error_dataset, 10.0),  # 10x upsampling
        ],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
    )

    print(f"\nMixture dataset created:")
    print(f"  Total length: {len(mixture_dataset)}")
    print(f"  Dataset weights: {mixture_dataset.dataset_sampling_weights}")

    # Sample multiple steps and verify they're all within valid ranges
    print(f"\nSampling 1000 steps to verify clipping...")
    mixture_dataset.set_epoch(0)

    normal_samples = []
    error_samples = []

    for i in range(1000):
        dataset, traj_id, base_index = mixture_dataset.sample_step(i)

        if dataset == normal_dataset:
            normal_samples.append(base_index)
        else:
            error_samples.append(base_index)
            # Verify error recovery samples are in clipped range
            assert base_index >= 30, f"Error recovery sample {base_index} is before frame 30!"
            traj_index = dataset.get_trajectory_index(traj_id)
            traj_len = dataset.trajectory_lengths[traj_index]
            assert (
                base_index < traj_len - 60
            ), f"Error recovery sample {base_index} is after frame {traj_len - 60}!"

    print(f"\n✓ All 1000 samples verified!")
    print(f"  Normal dataset samples: {len(normal_samples)}")
    print(f"  Error recovery samples: {len(error_samples)}")

    if len(error_samples) > 0:
        print(f"  Error recovery sample range: [{min(error_samples)}, {max(error_samples)}]")
        print(f"    (expected: all >= 30 and < trajectory_length - 60)")

    # Test 3: Verify actual data can be fetched from clipped samples
    print(f"\n{'='*80}")
    print("Test 3: Verify data fetching from clipped samples")
    print(f"{'='*80}")

    # Get a few error recovery samples and verify we can fetch data
    test_samples = min(5, len(error_samples))
    if test_samples > 0:
        for i in range(test_samples):
            dataset, traj_id, base_index = mixture_dataset.sample_step(i)
            if dataset == error_dataset:
                print(
                    f"\nFetching data for error recovery trajectory {traj_id}, base_index {base_index}"
                )

                # Get step data (without video to avoid torchcodec dependency)
                step_data = {}
                traj_data = dataset.get_trajectory_data(traj_id)
                dataset.curr_traj_data = traj_data

                # Fetch state and action data
                for key in ["state.state", "action.action"]:
                    modality = key.split(".")[0]
                    data = dataset.get_state_or_action(traj_id, modality, key, base_index)
                    step_data[key] = data
                    print(f"  {key}: shape={data.shape}")

                # Verify state history and action chunk shapes
                assert step_data["state.state"].shape[0] == len(
                    data_config.state_observation_indices
                ), f"Expected {len(data_config.state_observation_indices)} state history frames"
                assert step_data["action.action"].shape[0] == len(
                    data_config.action_indices
                ), f"Expected {len(data_config.action_indices)} action frames"

                print(f"  ✓ Data fetched successfully!")
                break

    print(f"\n{'='*80}")
    print("✓ ALL TESTS PASSED!")
    print(f"{'='*80}")
    print("\nSummary:")
    print("  1. ✓ get_valid_indices_for_trajectory() returns correct ranges")
    print("  2. ✓ Mixture dataset sampling respects error recovery clipping")
    print("  3. ✓ Data can be fetched from clipped samples with full history/action access")
    print("\nThe implementation is ready for training!")


if __name__ == "__main__":
    test_mixture_dataset_clipping()
