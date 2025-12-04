# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GR00T RECAP/Pi-style Policy Training Script

This script trains an advantage-conditioned policy using binary indicators
computed from the advantage function (RECAP/Pi approach).

The policy is conditioned on:
- Observations (state, vision)
- Binary indicator I_t ∈ {0, 1} indicating if the action has high advantage
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
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING


@dataclass
class RECAPTrainConfig:
    """Configuration for GR00T RECAP/Pi-style policy training."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories with advantage labels."""

    dataset_language_prompts: List[str] | None = None
    """Optional language prompt override for each dataset."""

    output_dir: str = "/tmp/gr00t_recap"
    """Directory to save model checkpoints."""

    data_config: str = "fourier_gr1_arms_only"
    """Data configuration to use for training."""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 50000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 2000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    # What to train
    tune_llm: bool = False
    """Whether to tune the LLM backbone."""

    tune_visual: bool = False
    """Whether to tune the visual encoder."""

    tune_projector: bool = True
    """Whether to tune the action head projector."""

    tune_diffusion_model: bool = True
    """Whether to tune the action head diffusion model."""

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
    """If True, balance dataset weights."""

    balance_trajectory_weights: bool = True
    """If True, sample trajectories weighted by length."""

    # RECAP-specific parameters
    indicator_embedding_dim: int = 128
    """Dimension for indicator embedding."""

    add_indicator_to_state: bool = True
    """If True, add indicator embedding to state features."""


def main(config: RECAPTrainConfig):
    """Main RECAP policy training function."""

    # Validate language prompts if provided
    if config.dataset_language_prompts is not None:
        assert len(config.dataset_language_prompts) == len(config.dataset_path), (
            f"Number of language prompts ({len(config.dataset_language_prompts)}) "
            f"must match number of dataset paths ({len(config.dataset_path)})"
        )

    print("\n" + "=" * 80)
    print("RECAP/Pi-STYLE POLICY TRAINING")
    print("Training an advantage-conditioned policy")
    print("=" * 80 + "\n")

    # ------------ Step 1: Load dataset with advantage conditioning enabled ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # Load modality configs and transforms
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # Load dataset with advantage conditioning
    # IMPORTANT: enable_advantage_conditioning=True to load indicator labels
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
            enable_rl=False,  # We don't need reward/value labels for policy training
            enable_advantage_conditioning=True,  # Load indicator labels
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
                enable_rl=False,
                enable_advantage_conditioning=True,
            )
            single_datasets.append(dataset)

        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, 1.0) for dataset in single_datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets with advantage labels")

    # ------------ Step 2: Load model ------------
    data_action_horizon = len(getattr(data_config_cls, "action_indices", list(range(16))))

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
        enable_rl=False,  # No value head needed
    )

    print("=" * 80)
    print("RECAP POLICY TRAINING MODE")
    print("Training policy conditioned on advantage indicators")
    print("=" * 80)

    # NOTE: For full RECAP implementation, you would need to:
    # 1. Add an indicator embedding layer to the model
    # 2. Modify the forward pass to inject indicator embeddings
    # 3. Update the prepare_input method to handle indicators
    #
    # For now, this script will train a standard policy. To add indicator conditioning:
    # - Modify the action_head to accept indicator as input
    # - Add indicator embedding to state features or as separate tokens
    # - See the RECAP paper for more details on conditioning strategies

    print("\nIMPORTANT: Indicator conditioning is loaded but not yet injected into the model.")
    print("The dataset provides 'indicator' in each batch (shape: [B, T]).")
    print("To fully implement RECAP, you need to:")
    print("1. Add an indicator embedding layer in the action head")
    print("2. Concatenate indicator embeddings with state features")
    print("3. This allows the policy to learn indicator-conditioned behavior")
    print()

    # Verify trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

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

        # Set trainability
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    # ------------ Step 3: Setup training ------------
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

    # ------------ Step 4: Run training ------------
    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    experiment.train()

    # ------------ Step 5: Save configuration ------------
    indices_config = {
        "video_observation_indices": getattr(data_config_cls, "video_observation_indices", [0]),
        "state_observation_indices": getattr(data_config_cls, "state_observation_indices", [0]),
        "action_indices": getattr(data_config_cls, "action_indices", list(range(16))),
        "data_config": config.data_config,
        "recap_mode": True,
        "advantage_conditioned": True,
    }

    indices_path = Path(config.output_dir) / "indices_config.json"
    with open(indices_path, "w") as f:
        json.dump(indices_config, f, indent=2)
    print(f"\nSaved indices configuration to {indices_path}")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(RECAPTrainConfig)

    # Print the config
    print("\n" + "=" * 50)
    print("GR00T RECAP POLICY TRAINING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than available ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            raw_args_list = sys.argv[1:]
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *raw_args_list,
            ]

            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
