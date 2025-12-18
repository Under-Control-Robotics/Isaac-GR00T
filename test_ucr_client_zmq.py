#!/usr/bin/env python3
"""
Example ZMQ client script for UCR GR00T server (current default mode).
Shows how to format observations and send requests to the ZMQ inference server.
"""

import numpy as np
from gr00t.policy.server_client import PolicyClient

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5555


def create_dummy_observation(batch_size=1):
    """
    Create a dummy observation with the correct format for UCR model.

    Action horizon: 16 steps
    Video: 2 frames (t-30, t) from ego_view camera
    State: 6 joint groups at current timestep (t)
    Language: task instruction

    Returns:
        Dictionary with video, state, and language observations
    """
    # Example image dimensions (adjust to your actual camera resolution)
    H, W = 480, 640

    observation = {
        "video": {
            "ego_view": np.random.randint(0, 255, size=(batch_size, 2, H, W, 3), dtype=np.uint8)
        },
        "state": {
            "waist_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
            "right_arm_joint": np.random.randn(batch_size, 1, 7).astype(np.float32),
            "left_arm_joint": np.random.randn(batch_size, 1, 7).astype(np.float32),
            "right_leg_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
            "left_leg_joint": np.random.randn(batch_size, 1, 6).astype(np.float32),
            "orientation_joint": np.random.randn(batch_size, 1, 3).astype(np.float32),
        },
        "language": {
            "task": [["pick up the box"]]  # List of list of strings
        },
    }

    return observation


def main():
    print(f"Testing UCR GR00T ZMQ server at {SERVER_HOST}:{SERVER_PORT}")
    print()

    # Create ZMQ client
    print("Connecting to ZMQ server...")
    client = PolicyClient(host=SERVER_HOST, port=SERVER_PORT)

    # Get modality config
    print("Getting modality config from server...")
    modality_configs = client.get_modality_config()
    print(f"  Available modalities: {list(modality_configs.keys())}")
    print()

    # Create dummy observation
    print("Creating dummy observation...")
    obs = create_dummy_observation(batch_size=1)
    print(f"  Video shape: {obs['video']['ego_view'].shape}")
    print(f"  State keys: {list(obs['state'].keys())}")
    print(f"  Language: {obs['language']['task']}")
    print()

    # Send to server
    print("Sending observation to ZMQ server...")
    try:
        actions, info = client.get_action(obs)
        print("Received actions:")
        for action_key, action_value in actions.items():
            print(f"  {action_key}: shape={action_value.shape}, dtype={action_value.dtype}")

        print()
        print("✓ ZMQ server test successful!")
        print(f"  Action horizon: {next(iter(actions.values())).shape[1]} steps")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
