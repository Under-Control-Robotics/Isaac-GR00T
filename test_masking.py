"""Test script for StateActionRandomMask logic."""

import torch
import sys

sys.path.insert(0, "/data/anthony/Isaac-GR00T")

from gr00t.data.transform.state_action import StateActionRandomMask


def test_full_masking():
    """Test that masking works correctly."""

    # Create test data: (timesteps, joint_dims)
    # Simulating 11 timesteps, 7 joint dimensions
    test_state = torch.randn(11, 7)
    original_state = test_state.clone()

    print("Original state shape:", test_state.shape)
    print("Original state sample:\n", test_state[:3, :3])

    # Test 1: mask_prob = 1.0 (should always mask)
    print("\n" + "=" * 60)
    print("Test 1: mask_prob = 1.0 (should always mask)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=1.0)
    transform.training = True

    masked_count = 0
    for i in range(10):
        data = {"state": test_state.clone()}
        result = transform.apply(data)
        if torch.all(result["state"] == 0):
            masked_count += 1

    print(f"Masked {masked_count}/10 times (expected: 10/10)")
    assert masked_count == 10, "With mask_prob=1.0, should mask every time"

    # Test 2: mask_prob = 0.0 (should never mask)
    print("\n" + "=" * 60)
    print("Test 2: mask_prob = 0.0 (should never mask)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=0.0)
    transform.training = True

    unmasked_count = 0
    for i in range(10):
        data = {"state": test_state.clone()}
        result = transform.apply(data)
        if torch.allclose(result["state"], test_state):
            unmasked_count += 1

    print(f"Unmasked {unmasked_count}/10 times (expected: 10/10)")
    assert unmasked_count == 10, "With mask_prob=0.0, should never mask"

    # Test 3: mask_prob = 0.3 (should mask ~30% of the time)
    print("\n" + "=" * 60)
    print("Test 3: mask_prob = 0.3 (should mask ~30% of the time)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=0.3)
    transform.training = True

    num_trials = 1000
    masked_count = 0
    for i in range(num_trials):
        data = {"state": test_state.clone()}
        result = transform.apply(data)
        if torch.all(result["state"] == 0):
            masked_count += 1

    mask_rate = masked_count / num_trials
    print(f"Masked {masked_count}/{num_trials} times = {mask_rate:.1%}")
    print(f"Expected: ~30%, Got: {mask_rate:.1%}")

    # Allow 5% tolerance (between 25% and 35%)
    assert (
        0.25 <= mask_rate <= 0.35
    ), f"Mask rate {mask_rate:.1%} is outside expected range (25%-35%)"

    # Test 4: Evaluation mode (should never mask)
    print("\n" + "=" * 60)
    print("Test 4: Evaluation mode with mask_prob=1.0 (should never mask)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=1.0)
    transform.training = False  # Evaluation mode

    unmasked_count = 0
    for i in range(10):
        data = {"state": test_state.clone()}
        result = transform.apply(data)
        if torch.allclose(result["state"], test_state):
            unmasked_count += 1

    print(f"Unmasked {unmasked_count}/10 times (expected: 10/10)")
    assert unmasked_count == 10, "In eval mode, should never mask"

    # Test 5: Verify entire trajectory is masked (not partial)
    print("\n" + "=" * 60)
    print("Test 5: Verify entire trajectory is masked (not just some dims)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=1.0)
    transform.training = True

    data = {"state": test_state.clone()}
    result = transform.apply(data)

    print("Result after masking:")
    print(result["state"])

    # Check that ALL values are 0, not just some
    assert torch.all(result["state"] == 0), "Should mask ALL timesteps and ALL dimensions"
    print("✓ Confirmed: All timesteps and all dimensions are masked")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_full_masking()
