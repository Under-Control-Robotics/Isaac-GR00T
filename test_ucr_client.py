#!/usr/bin/env python3
"""
Example client script for UCR GR00T server (UCR branch legacy style).
Shows how to format observations and send requests to the inference server.

This is the DEFAULT client that works with the DEFAULT server mode.
Uses HTTP with FLAT observation format (video.ego_view, state.waist_joint, etc.)

Dependencies: pip install requests
Optional: pip install json-numpy (for automatic numpy serialization)
"""

import numpy as np
import requests

# Try to use json_numpy for automatic numpy serialization (optional)
try:
    import json_numpy
    json_numpy.patch()
    HAS_JSON_NUMPY = True
except ImportError:
    HAS_JSON_NUMPY = False

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/act"

def create_dummy_observation(batch_size=1):
    """
    Create a dummy observation with FLAT format (UCR branch legacy style).

    Action horizon: 16 steps
    Video: 2 frames (t-30, t) from ego_view camera
    State: 6 joint groups at current timestep (t)
    Language: task instruction

    Returns:
        Dictionary with FLAT keys (video.ego_view, state.waist_joint, etc.)
    """
    # Example image dimensions (adjust to your actual camera resolution)
    H, W = 480, 640

    observation = {
        # Video modality - flat format
        "video.ego_view": np.random.randint(0, 255, size=(batch_size, 2, H, W, 3), dtype=np.uint8),
        # State modalities - flat format
        "state.waist_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
        "state.right_arm_joint": np.random.randn(batch_size, 1, 7).astype(np.float32),
        "state.left_arm_joint": np.random.randn(batch_size, 1, 7).astype(np.float32),
        "state.right_leg_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
        "state.left_leg_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
        "state.orientation_joint": np.random.randn(batch_size, 1, 3).astype(np.float32),
        # Language - simple list format (UCR branch style)
        "task": ["pick up the box"],
    }

    return observation

def _numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def send_observation(observation):
    """
    Send observation to the HTTP server and get action prediction.

    Args:
        observation: Dictionary with flat keys (video.ego_view, state.waist_joint, etc.)

    Returns:
        Dictionary with predicted actions in flat format (action.left_ee_position, etc.)
    """
    # Serialize observation
    if HAS_JSON_NUMPY:
        # json_numpy.patch() handles numpy serialization automatically
        payload = {"observation": observation}
    else:
        # Manual conversion to lists
        payload = {"observation": _numpy_to_list(observation)}

    # Send POST request
    response = requests.post(SERVER_URL, json=payload)

    if response.status_code == 200:
        actions = response.json()
        # Convert lists back to numpy arrays
        actions = {k: np.array(v, dtype=np.float32) for k, v in actions.items()}
        return actions
    else:
        raise RuntimeError(f"Server error: {response.status_code} - {response.text}")

def main():
    print("=" * 60)
    print("UCR GR00T Client (UCR Branch Legacy Style)")
    print("=" * 60)
    print()
    print(f"Server URL: {SERVER_URL}")
    print("Format: Flat observations (video.ego_view, state.waist_joint, etc.)")
    print()

    # Create dummy observation
    print("Creating dummy observation (FLAT format)...")
    obs = create_dummy_observation(batch_size=1)
    print(f"  video.ego_view: shape={obs['video.ego_view'].shape}")
    print(f"  state.waist_joint: shape={obs['state.waist_joint'].shape}")
    print(f"  state.right_arm_joint: shape={obs['state.right_arm_joint'].shape}")
    print(f"  state.left_arm_joint: shape={obs['state.left_arm_joint'].shape}")
    print(f"  state.right_leg_joint: shape={obs['state.right_leg_joint'].shape}")
    print(f"  state.left_leg_joint: shape={obs['state.left_leg_joint'].shape}")
    print(f"  state.orientation_joint: shape={obs['state.orientation_joint'].shape}")
    print(f"  task: {obs['task']}")
    print()

    # Send to server
    print("Sending observation to HTTP server...")
    try:
        actions = send_observation(obs)
        print("Received actions (FLAT format):")
        for action_key, action_value in sorted(actions.items()):
            print(f"  {action_key}: shape={action_value.shape}, dtype={action_value.dtype}")

        print()
        print("=" * 60)
        print("✓ UCR Legacy Server Test SUCCESSFUL!")
        print("=" * 60)
        print(f"  Action horizon: {next(iter(actions.values())).shape[1]} steps")

    except Exception as e:
        print()
        print("=" * 60)
        print("✗ ERROR")
        print("=" * 60)
        print(f"{e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
