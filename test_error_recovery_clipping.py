#!/usr/bin/env python3
"""Test script to verify error recovery dataset clipping."""

import sys
from pathlib import Path

# Add the gr00t module to path
sys.path.insert(0, str(Path(__file__).parent))

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import UCRWBLMMobyHistoryDataConfig


def test_dataset_clipping():
    """Test that error recovery datasets are properly clipped."""

    # Get data config
    data_config = UCRWBLMMobyHistoryDataConfig()

    # Test 1: Load a normal dataset
    normal_dataset_path = "/data/anthony/ucr_ros/data_files/dataset/normal_data/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-19_20:53:47.501581"
    print(f"\n{'='*80}")
    print(f"Testing NORMAL dataset:")
    print(f"Path: {normal_dataset_path}")
    print(f"{'='*80}")

    normal_dataset = LeRobotSingleDataset(
        dataset_path=normal_dataset_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag="new_embodiment",
    )

    print(f"\nTrajectory IDs: {normal_dataset.trajectory_ids}")
    print(f"Trajectory lengths: {normal_dataset.trajectory_lengths}")
    print(f"Total steps in dataset: {len(normal_dataset.all_steps)}")
    print(
        f"Expected steps (sum of all trajectory lengths): {sum(normal_dataset.trajectory_lengths)}"
    )

    # Show first few and last few steps
    print(f"\nFirst 5 steps: {normal_dataset.all_steps[:5]}")
    print(f"Last 5 steps: {normal_dataset.all_steps[-5:]}")

    # Test 2: Load an error recovery dataset
    error_recovery_path = "/data/anthony/ucr_ros/data_files/dataset/error_recovery/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-27_08:14:08.248872"
    print(f"\n{'='*80}")
    print(f"Testing ERROR RECOVERY dataset:")
    print(f"Path: {error_recovery_path}")
    print(f"{'='*80}")

    error_dataset = LeRobotSingleDataset(
        dataset_path=error_recovery_path,
        modality_configs=data_config.modality_config(),
        embodiment_tag="new_embodiment",
    )

    print(f"\nTrajectory IDs: {error_dataset.trajectory_ids}")
    print(f"Trajectory lengths: {error_dataset.trajectory_lengths}")
    print(f"Total steps in dataset: {len(error_dataset.all_steps)}")

    # Calculate expected clipped steps
    expected_clipped_steps = 0
    for traj_len in error_dataset.trajectory_lengths:
        start_frame = 30
        end_frame = traj_len - 60
        if end_frame > start_frame:
            expected_clipped_steps += end_frame - start_frame
        else:
            print(f"  Warning: Trajectory too short ({traj_len} frames), would be skipped")

    print(f"Expected clipped steps (30 to len-60 for each trajectory): {expected_clipped_steps}")

    # Show first few and last few steps
    print(f"\nFirst 5 steps: {error_dataset.all_steps[:5]}")
    print(f"Last 5 steps: {error_dataset.all_steps[-5:]}")

    # Verify clipping
    print(f"\n{'='*80}")
    print("Verification:")
    print(f"{'='*80}")

    for i, (traj_id, traj_len) in enumerate(
        zip(error_dataset.trajectory_ids, error_dataset.trajectory_lengths)
    ):
        # Get all steps for this trajectory
        traj_steps = [step for step in error_dataset.all_steps if step[0] == traj_id]

        if len(traj_steps) > 0:
            min_index = min(step[1] for step in traj_steps)
            max_index = max(step[1] for step in traj_steps)

            print(
                f"Trajectory {traj_id}: length={traj_len}, steps={len(traj_steps)}, "
                f"index range=[{min_index}, {max_index}]"
            )

            # Verify that min_index is 30 and max_index is traj_len-61
            assert min_index == 30, f"Expected min_index=30, got {min_index}"
            assert max_index == traj_len - 61, f"Expected max_index={traj_len-61}, got {max_index}"
        else:
            print(f"Trajectory {traj_id}: length={traj_len}, SKIPPED (too short)")

    print(f"\n{'='*80}")
    print("✓ All tests passed! Error recovery clipping is working correctly.")
    print(f"{'='*80}")

    # Test 3: Verify index access without actually loading video
    print(f"\n{'='*80}")
    print("Testing index access for state history and action chunks:")
    print(f"{'='*80}")

    # Get first valid step (should be at frame 30)
    first_step_idx = 0
    traj_id, base_index = error_dataset.all_steps[first_step_idx]

    print(f"\nFor trajectory {traj_id}, base_index {base_index}:")
    print(f"State history indices: {data_config.state_observation_indices}")
    state_frames = [base_index + idx for idx in data_config.state_observation_indices]
    print(f"  -> Will access frames: {state_frames}")
    print(f"  -> Min frame: {min(state_frames)}, Max frame: {max(state_frames)}")

    print(f"\nAction indices: [0...{len(data_config.action_indices)-1}]")
    action_frames = [base_index + idx for idx in range(len(data_config.action_indices))]
    print(f"  -> Will access frames: [{action_frames[0]}...{action_frames[-1]}]")
    print(f"  -> Min frame: {min(action_frames)}, Max frame: {max(action_frames)}")

    # Verify the frames are within valid range
    traj_index = error_dataset.get_trajectory_index(traj_id)
    traj_length = error_dataset.trajectory_lengths[traj_index]

    print(f"\nTrajectory length: {traj_length}")
    print(f"State history goes back to frame {min(state_frames)} (valid: >= 0)")
    print(f"Action chunk goes forward to frame {max(action_frames)} (valid: < {traj_length})")

    assert min(state_frames) >= 0, f"State history goes before start of episode!"
    assert max(action_frames) < traj_length, f"Action chunk goes beyond end of episode!"

    print(f"\n✓ All frame indices are within valid range!")
    print(f"\n{'='*80}")
    print("✓ Index access test passed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_dataset_clipping()
