"""
Robot inference server/client using pickle-based ZMQ (UCR branch compatible).

This is for UCR branch compatibility. For new code, use PolicyServer/PolicyClient from
gr00t.policy.server_client (msgpack-based).
"""

from typing import Any, Dict

from gr00t.data.types import ModalityConfig
from gr00t.eval.pickle_service import PickleInferenceServer, PickleInferenceClient
from gr00t.policy.policy import BasePolicy


class RobotInferenceServer(PickleInferenceServer):
    """
    Server with endpoints for robot policies (UCR branch compatible).
    Uses pickle-based ZMQ serialization.

    For new code, use PolicyServer from gr00t.policy.server_client.
    """

    def __init__(self, model, host: str = "*", port: int = 5555, api_token: str = None):
        super().__init__(host, port)
        self.model = model

        # Wrap get_action to handle both old and new API
        def get_action_wrapper(data):
            # Old UCR branch client sends observations directly in data
            # not wrapped in {"observation": ...}
            # AND uses flat format: video.ego_view, state.waist_joint
            # Need to convert to nested format: {"video": {"ego_view": ...}}

            obs = self._convert_flat_to_nested(data)
            result = model.get_action(obs)

            # Return just the action (UCR branch style), not (action, info) tuple
            if isinstance(result, tuple):
                action, _ = result  # New API: extract action from (action, info)
                # Convert nested action back to flat for old UCR client
                return self._convert_nested_to_flat(action)
            return self._convert_nested_to_flat(result)  # Old API: already just action

        self.register_endpoint("get_action", get_action_wrapper)
        self.register_endpoint(
            "get_modality_config", model.get_modality_config, requires_input=False
        )

    def _convert_flat_to_nested(self, flat_obs: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat observation format to nested format.

        Old UCR branch uses: video.ego_view, state.waist_joint, annotation.human.action.task_description
        Current branch expects: {"video": {"ego_view": ...}, "state": {...}, "language": {...}}

        Also adds batch dimension if missing:
        - Video: (T, H, W, C) → (1, T, H, W, C)
        - State: (T, D) → (1, T, D)
        """
        import numpy as np

        nested = {"video": {}, "state": {}, "language": {}}

        for key, value in flat_obs.items():
            if key.startswith("video."):
                video_key = key.replace("video.", "")
                # Old UCR: (T, H, W, C) → New: (B=1, T, H, W, C)
                if isinstance(value, np.ndarray) and value.ndim == 4:
                    value = value[np.newaxis, ...]  # Add batch dimension
                nested["video"][video_key] = value

            elif key.startswith("state."):
                state_key = key.replace("state.", "")
                # Old UCR: (T, D) → New: (B=1, T, D)
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    value = value[np.newaxis, ...]  # Add batch dimension
                nested["state"][state_key] = value

            else:
                # Everything else is language (task, annotation.human.action.task_description, etc.)
                # Old UCR uses simple list: ["instruction"] or string "instruction"
                # New expects: [["instruction"]]

                # Handle string input
                if isinstance(value, str):
                    nested["language"][key] = [[value]]
                # Handle list of strings
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], str):
                        nested["language"][key] = [[v] for v in value]
                    else:
                        nested["language"][key] = value
                else:
                    nested["language"][key] = value

        return nested

    def _convert_nested_to_flat(self, nested_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert nested action format to flat format.

        Current branch returns: {"left_ee_position": ..., "right_ee_position": ...}
        Old UCR expects: {"action.left_ee_position": ..., "action.right_ee_position": ...}

        Also removes batch dimension:
        - Action: (B=1, T, D) → (T, D)
        """
        import numpy as np

        flat = {}
        for key, value in nested_action.items():
            # Remove batch dimension if present
            if isinstance(value, np.ndarray) and value.shape[0] == 1:
                value = value[0]  # (1, T, D) → (T, D)
            flat[f"action.{key}"] = value
        return flat

    @staticmethod
    def start_server(policy: BasePolicy, port: int, api_token: str = None):
        server = RobotInferenceServer(policy, port=port, api_token=api_token)
        server.run()


class RobotInferenceClient(PickleInferenceClient, BasePolicy):
    """
    Client for communicating with the RobotInferenceServer (UCR branch compatible).
    Uses pickle-based ZMQ serialization.

    For new code, use PolicyClient from gr00t.policy.server_client.
    """

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: str = None, strict: bool = False):
        PickleInferenceClient.__init__(self, host=host, port=port)
        BasePolicy.__init__(self, strict=strict)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        # Old UCR branch sends observations directly, not wrapped
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        return self.call_endpoint("get_modality_config", requires_input=False)

    def _get_action(self, observation: Dict[str, Any], options: Dict[str, Any] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Wrapper for BasePolicy compatibility - returns (action, info) tuple"""
        action = self.get_action(observation)
        return action, {}

    def reset(self, options: Dict[str, Any] = None) -> Dict[str, Any]:
        return {}

    def check_observation(self, observation: Dict[str, Any]) -> None:
        pass

    def check_action(self, action: Dict[str, Any]) -> None:
        pass
