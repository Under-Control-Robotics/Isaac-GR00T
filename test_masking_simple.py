"""Simplified test script for StateActionRandomMask logic."""

import torch
from typing import Any


# Simplified version of the masking logic to test
class StateActionRandomMask:
    def __init__(self, apply_to, mask_prob):
        self.apply_to = apply_to
        self.mask_prob = mask_prob
        self.training = True

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't mask the data in eval mode
            return data
        if self.mask_prob <= 0:
            # If mask probability is 0 or negative, don't mask
            return data

        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)

            # Clone to avoid in-place modification
            state = state.clone()

            # With probability mask_prob, mask entire trajectory to 0
            if torch.rand(1).item() < self.mask_prob:
                state[...] = 0

            data[key] = state
        return data


def test_full_masking():
    """Test that masking works correctly."""

    # Create test data: (timesteps, joint_dims)
    # Simulating 11 timesteps, 7 joint dimensions
    test_state = torch.randn(11, 7)

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
    print("✓ PASS")

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
    print("✓ PASS")

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
    print("✓ PASS")

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
    print("✓ PASS")

    # Test 5: Verify entire trajectory is masked (not partial)
    print("\n" + "=" * 60)
    print("Test 5: Verify entire trajectory is masked (not just some dims)")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=1.0)
    transform.training = True

    data = {"state": test_state.clone()}
    result = transform.apply(data)

    print("Result after masking (first 3x3):")
    print(result["state"][:3, :3])

    # Check that ALL values are 0, not just some
    assert torch.all(result["state"] == 0), "Should mask ALL timesteps and ALL dimensions"
    print("✓ PASS - Confirmed: All timesteps and all dimensions are masked")

    # Test 6: Verify when not masked, data is unchanged
    print("\n" + "=" * 60)
    print("Test 6: When not masked (30% prob), data should be unchanged")
    print("=" * 60)

    transform = StateActionRandomMask(apply_to=["state"], mask_prob=0.3)
    transform.training = True

    # Run multiple times and check unmasked samples
    unmasked_unchanged_count = 0
    total_unmasked = 0
    for i in range(100):
        data = {"state": test_state.clone()}
        result = transform.apply(data)
        if not torch.all(result["state"] == 0):  # If not masked
            total_unmasked += 1
            if torch.allclose(result["state"], test_state):
                unmasked_unchanged_count += 1

    print(f"Total unmasked samples: {total_unmasked}/100")
    print(f"Unmasked samples that are unchanged: {unmasked_unchanged_count}/{total_unmasked}")
    assert (
        unmasked_unchanged_count == total_unmasked
    ), "Unmasked samples should be completely unchanged"
    print("✓ PASS")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓✓✓")
    print("=" * 60)
    print("\nSummary:")
    print("- Full trajectory masking works correctly")
    print("- Masking probability is respected")
    print("- Evaluation mode doesn't mask")
    print("- When masked: ALL dimensions and timesteps are 0")
    print("- When not masked: data is completely unchanged")


if __name__ == "__main__":
    test_full_masking()
