from dataclasses import dataclass
import json
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the Groot N1.5 inference server."""

    # Gr00t policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag"""

    device: str = "cuda"
    """Device to run the model on"""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "127.0.0.1"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""

    server_type: str = "zmq"
    """Server type: 'zmq' (default, current) or 'http' (legacy from ucr branch)"""

    zmq_serialization: str = "msgpack"
    """ZMQ serialization: 'msgpack' (default, current) or 'pickle' (old UCR branch)"""


def main(config: ServerConfig):
    print("Starting GR00T inference server...")
    print(f"  Server type: {config.server_type}")
    if config.server_type == "zmq":
        print(f"  ZMQ serialization: {config.zmq_serialization}")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")

    # check if the model path exists
    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # Create and start the server
    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    # Apply sim policy wrapper if needed
    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    # Create server based on type
    if config.server_type == "http":
        print("\nUsing HTTP server (legacy mode from ucr branch)")
        print("Endpoint: POST /act with JSON payload {'observation': {...}}")
        print("Install dependencies: pip install uvicorn fastapi")
        from gr00t.eval.http_server import HTTPInferenceServer

        server = HTTPInferenceServer(
            policy=policy,
            host=config.host,
            port=config.port,
        )
    elif config.server_type == "zmq":
        if config.zmq_serialization == "pickle":
            print("\nUsing ZMQ server with PICKLE serialization (old UCR branch compatible)")
            print("This mode is compatible with old UCR branch clients")
            from gr00t.eval.robot import RobotInferenceServer

            server = RobotInferenceServer(
                model=policy,
                host=config.host,
                port=config.port,
            )
        elif config.zmq_serialization == "msgpack":
            print("\nUsing ZMQ server with MSGPACK serialization (current default)")
            from gr00t.policy.server_client import PolicyServer

            server = PolicyServer(
                policy=policy,
                host=config.host,
                port=config.port,
            )
        else:
            raise ValueError(
                f"Invalid zmq_serialization: {config.zmq_serialization}. Must be 'msgpack' or 'pickle'"
            )
    else:
        raise ValueError(f"Invalid server_type: {config.server_type}. Must be 'zmq' or 'http'")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
