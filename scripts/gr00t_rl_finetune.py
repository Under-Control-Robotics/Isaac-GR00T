# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GR00T RL Finetuning Script

This script is for RL finetuning that ONLY trains the value function head.
The policy (backbone + action head) is frozen.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
import json

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class RLArgsConfig:
    """Configuration for GR00T RL fine-tuning (value function only)."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories with reward labels."""

    dataset_language_prompts: List[str] | None = None
    """Optional language prompt override for each dataset."""

    output_dir: str = "/data/anthony/Isaac-GR00T/checkpoints/1202_value_head"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """
    Data configuration to use for training.
    Options:
    - Built-in configs: Use predefined config names like 'so100', 'fourier_gr1_arms_only', 'unitree_g1'.
    - External configs: Use 'module:ClassName' format to load custom configs from external files.
    See gr00t/experiment/data_config.py for more details.
    """

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model (with pretrained policy)."""

    value_hidden_dim: int = 1024
    """Hidden dimension for value head MLP."""

    value_dropout: float = 0.1
    """Dropout rate for value head."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    dataloader_num_workers: int = 12
    """Number of workers for data loading per GPU."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps for training."""

    dataloader_prefetch_factor: int = 4
    """Prefetch factor for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training."""

    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "decord"
    """Video backend to use for training."""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, balance dataset weights."""

    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories weighted by length."""


#####################################################################################
# main RL training function
#####################################################################################


def main(config: RLArgsConfig):
    """Main RL training function - trains ONLY the value head."""
    # Validate language prompts if provided
    if config.dataset_language_prompts is not None:
        assert len(config.dataset_language_prompts) == len(config.dataset_path), (
            f"Number of language prompts ({len(config.dataset_language_prompts)}) "
            f"must match number of dataset paths ({len(config.dataset_path)})"
        )

    # ------------ step 1: load dataset with RL mode enabled ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    # IMPORTANT: enable_rl=True to load reward/value labels
    if len(config.dataset_path) == 1:
        language_prompt = (
            config.dataset_language_prompts[0] if config.dataset_language_prompts else None
        )
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=config.video_backend,
            language_override=language_prompt,
            enable_rl=True,  # Enable RL mode to load reward/value
        )
    else:
        single_datasets = []
        for idx, p in enumerate(config.dataset_path):
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            language_prompt = (
                config.dataset_language_prompts[idx] if config.dataset_language_prompts else None
            )
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                language_override=language_prompt,
                enable_rl=True,  # Enable RL mode to load reward/value
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0) for dataset in single_datasets  # Equal weights for all datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets with RL labels")

    # ------------ step 2: load model with RL mode enabled ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(getattr(data_config_cls, "action_indices", list(range(16))))

    # Load model with RL enabled - this will add a value head
    # IMPORTANT: Freeze ALL policy parameters (backbone + action head)
    # Only train the value head
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=False,  # Freeze LLM
        tune_visual=False,  # Freeze vision tower
        tune_projector=False,  # Freeze action head projector
        tune_diffusion_model=False,  # Freeze action head DiT
        tune_value_head=True,  # ONLY train value head
        enable_rl=True,  # Enable RL mode to add value head
        value_head_cfg={
            "hidden_dim": config.value_hidden_dim,
            "dropout": config.value_dropout,
        },
    )

    print("=" * 80)
    print("RL FINETUNING MODE: Training ONLY the value function head")
    print("Policy (backbone + action head) is FROZEN")
    print("=" * 80)

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # List trainable parameter groups
    print("\nTrainable parameter groups:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.shape}")

    # Update action_horizon to match data config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} "
            f"(was {model.action_head.config.action_horizon})"
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        # Make sure action head stays frozen
        model.action_head.set_trainable_parameters(tune_projector=False, tune_diffusion_model=False)

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()

    # 2.4 save indices configuration
    indices_config = {
        "video_observation_indices": getattr(data_config_cls, "video_observation_indices", [0]),
        "state_observation_indices": getattr(data_config_cls, "state_observation_indices", [0]),
        "action_indices": getattr(data_config_cls, "action_indices", list(range(16))),
        "data_config": config.data_config,
        "rl_mode": True,  # Mark this as RL training
    }

    indices_path = Path(config.output_dir) / "indices_config.json"
    with open(indices_path, "w") as f:
        json.dump(indices_config, f, indent=2)
    print(f"\nSaved indices configuration to {indices_path}")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(RLArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T RL FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            script_path = Path(__file__).absolute()

            # Use subprocess.run instead of os.system
            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
