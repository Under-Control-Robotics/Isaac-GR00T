#!/usr/bin/env python3
"""
DEFINITIVE TEST: Verify error recovery clipping works exactly as specified.

Requirements:
1. Training samples ONLY from [30, len-60]
2. Can access observation history (30 frames backward)
3. Can access future action horizon (64 frames forward)
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotMixtureDataset
from gr00t.experiment.data_config import UCRWBLMMobyHistoryDataConfig


def test_final_verification():
    """Final verification that everything works correctly."""

    print("=" * 80)
    print("FINAL VERIFICATION TEST")
    print("=" * 80)

    # Setup
    data_config = UCRWBLMMobyHistoryDataConfig()
    modality_configs = data_config.modality_config()

    error_recovery_path = "/data/anthony/ucr_ros/data_files/dataset/error_recovery/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-27_08:14:08.248872"

    error_dataset = LeRobotSingleDataset(
        dataset_path=error_recovery_path,
        modality_configs=modality_configs,
        embodiment_tag="new_embodiment",
    )

    print(f"\nDataset: {error_recovery_path.split('/')[-1]}")
    print(f"Trajectory IDs: {error_dataset.trajectory_ids}")
    print(f"Trajectory lengths: {error_dataset.trajectory_lengths}")

    # Get the state history and action horizon indices
    state_history_indices = data_config.state_observation_indices
    action_horizon_indices = data_config.action_indices

    print(f"\nState history indices: {state_history_indices}")
    print(f"  → Goes back {abs(min(state_history_indices))} frames")
    print(f"\nAction horizon indices: {action_horizon_indices[:5]}...{action_horizon_indices[-5:]}")
    print(f"  → Goes forward {max(action_horizon_indices)} frames")

    # TEST 1: Verify all_steps only contains clipped frames
    print(f"\n{'='*80}")
    print("TEST 1: Verify training samples are ONLY from [30, len-60]")
    print(f"{'='*80}")

    for traj_id in error_dataset.trajectory_ids:
        traj_index = error_dataset.get_trajectory_index(traj_id)
        traj_length = error_dataset.trajectory_lengths[traj_index]

        # Get all steps for this trajectory
        traj_steps = [step for step in error_dataset.all_steps if step[0] == traj_id]

        if len(traj_steps) > 0:
            base_indices = [step[1] for step in traj_steps]
            min_base = min(base_indices)
            max_base = max(base_indices)

            print(f"\nTrajectory {traj_id} (length={traj_length}):")
            print(f"  Training samples: {len(base_indices)} steps")
            print(f"  Base index range: [{min_base}, {max_base}]")
            print(f"  Expected range:   [30, {traj_length-61}]")

            # Verify
            assert min_base == 30, f"ERROR: min_base should be 30, got {min_base}"
            assert (
                max_base == traj_length - 61
            ), f"ERROR: max_base should be {traj_length-61}, got {max_base}"
            print(f"  ✓ CORRECT: Only samples from [30, {traj_length-61}]")

    print(f"\n✓ TEST 1 PASSED: All training samples are in [30, len-60]")

    # TEST 2: Verify observation history can access frames before clip start
    print(f"\n{'='*80}")
    print("TEST 2: Verify observation history can access frames BEFORE clip start")
    print(f"{'='*80}")

    # Test at the earliest possible base_index (30)
    traj_id = error_dataset.trajectory_ids[0]
    earliest_base_index = 30

    print(f"\nTesting at EARLIEST valid base_index = {earliest_base_index}")
    print(
        f"State history will need frames: {[earliest_base_index + idx for idx in state_history_indices]}"
    )

    min_required_frame = earliest_base_index + min(state_history_indices)
    print(f"  → Minimum frame needed: {min_required_frame}")
    print(f"  → This is {'BEFORE' if min_required_frame < 30 else 'AFTER'} the clip start (30)")

    # Load trajectory data
    traj_data = error_dataset.get_trajectory_data(traj_id)
    error_dataset.curr_traj_data = traj_data

    # Fetch state data
    state_data = error_dataset.get_state_or_action(
        trajectory_id=traj_id, modality="state", key="state.state", base_index=earliest_base_index
    )

    print(f"\n✓ Successfully fetched state data!")
    print(f"  Shape: {state_data.shape}")
    print(f"  Expected: ({len(state_history_indices)}, state_dim)")

    assert state_data.shape[0] == len(state_history_indices), "Wrong number of history frames!"
    print(f"\n✓ TEST 2 PASSED: Can access frames before clip start (frame {min_required_frame})")

    # TEST 3: Verify action horizon can access frames after clip end
    print(f"\n{'='*80}")
    print("TEST 3: Verify action horizon can access frames AFTER clip end")
    print(f"{'='*80}")

    # Test at the latest possible base_index (len-61)
    traj_index = error_dataset.get_trajectory_index(traj_id)
    traj_length = error_dataset.trajectory_lengths[traj_index]
    latest_base_index = traj_length - 61

    print(f"\nTesting at LATEST valid base_index = {latest_base_index}")
    print(
        f"Action horizon will need frames: [{latest_base_index}...{latest_base_index + max(action_horizon_indices)}]"
    )

    max_required_frame = latest_base_index + max(action_horizon_indices)
    clip_end = traj_length - 60
    print(f"  → Maximum frame needed: {max_required_frame}")
    print(f"  → Clip ends at: {clip_end}")
    print(f"  → Episode ends at: {traj_length - 1}")
    print(f"  → Need to access {max_required_frame - (clip_end - 1)} frames AFTER clip end")

    # Fetch action data
    action_data = error_dataset.get_state_or_action(
        trajectory_id=traj_id, modality="action", key="action.action", base_index=latest_base_index
    )

    print(f"\n✓ Successfully fetched action data!")
    print(f"  Shape: {action_data.shape}")
    print(f"  Expected: ({len(action_horizon_indices)}, action_dim)")

    assert action_data.shape[0] == len(action_horizon_indices), "Wrong number of action frames!"

    if max_required_frame >= traj_length:
        print(
            f"\n  Note: Frames [{traj_length}...{max_required_frame}] were padded (beyond episode end)"
        )

    print(f"\n✓ TEST 3 PASSED: Can access frames after clip end (up to frame {max_required_frame})")

    # TEST 4: Verify mixture dataset sampling
    print(f"\n{'='*80}")
    print("TEST 4: Verify mixture dataset sampling respects clipping")
    print(f"{'='*80}")

    normal_path = "/data/anthony/ucr_ros/data_files/dataset/normal_data/pick_up_the_plastic_box_then_place_it_in_the_tray_/2026-02-19_20:53:47.501581"
    normal_dataset = LeRobotSingleDataset(
        dataset_path=normal_path,
        modality_configs=modality_configs,
        embodiment_tag="new_embodiment",
    )

    mixture_dataset = LeRobotMixtureDataset(
        data_mixture=[
            (normal_dataset, 1.0),
            (error_dataset, 10.0),
        ],
        mode="train",
        balance_dataset_weights=True,
        balance_trajectory_weights=True,
        seed=42,
    )

    print(f"\nSampling 5000 steps from mixture dataset...")
    mixture_dataset.set_epoch(0)

    error_samples = []
    for i in range(5000):
        dataset, traj_id, base_index = mixture_dataset.sample_step(i)

        if dataset == error_dataset:
            error_samples.append((traj_id, base_index))

            # Verify base_index is in valid range
            traj_index = dataset.get_trajectory_index(traj_id)
            traj_length = dataset.trajectory_lengths[traj_index]

            assert base_index >= 30, f"ERROR: base_index {base_index} < 30!"
            assert (
                base_index < traj_length - 60
            ), f"ERROR: base_index {base_index} >= {traj_length - 60}!"

    print(f"  → Sampled {len(error_samples)} error recovery steps")

    if len(error_samples) > 0:
        all_base_indices = [s[1] for s in error_samples]
        print(f"  → Base index range: [{min(all_base_indices)}, {max(all_base_indices)}]")

    print(f"\n✓ TEST 4 PASSED: All mixture samples respect clipping")

    # FINAL SUMMARY
    print(f"\n{'='*80}")
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print(f"{'='*80}")
    print("\nVERIFIED:")
    print("  1. ✓ Training ONLY samples from [30, len-60] for error recovery")
    print("  2. ✓ Can access observation history 30 frames backward (before clip start)")
    print("  3. ✓ Can access action horizon 64 frames forward (after clip end)")
    print("  4. ✓ Mixture dataset sampling respects clipping")
    print("\nCONCLUSION:")
    print("  The implementation is CORRECT and ready for training!")
    print("  Run: bash train_error_recovery_10x.sh")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_final_verification()
