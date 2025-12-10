# Advantage-Conditioned GR00T Inference Guide

This guide explains how to use the advantage-conditioned GR00T model for inference. The model was trained with advantage indicators that allow you to control the quality of generated actions.

## Overview

The advantage-conditioned model was trained using the advantage-weighted loss approach where:
- **High-advantage demonstrations** (indicator = 1) → High-quality actions
- **Low-advantage demonstrations** (indicator = 0) → Low-quality actions

During inference, you control the action quality by passing an `indicator` field in the observation dictionary.

## Model Architecture Differences

### Standard GR00T Model
```
Input → Backbone → Action Head → Output
```

### Advantage-Conditioned GR00T Model
```
Input + Indicator Token → Indicator Embedding → [Prepend to Backbone Features] → Action Head → Output
```

**Key differences:**
1. Model has an **indicator embedding layer** that embeds the binary indicator (0 or 1)
2. The indicator token is **prepended as the first token** to the backbone features
3. The model learns to generate different quality actions based on the indicator

## Checkpoint Location

The advantage-conditioned checkpoint is saved at:
```
/data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned
```

This checkpoint was created by running `scripts/gr00t_advantage_conditioned_train.py` on datasets that were processed with advantage labels.

## Usage

### 1. Server Mode (ZMQ)

Start the inference server:
```bash
python scripts/inference_service_advantage_conditioned.py \
  --server \
  --model-path /data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned \
  --data-config ucr_wblm_moby_history \
  --embodiment-tag new_embodiment \
  --port 5556
```

### 2. Server Mode (HTTP)

Start the HTTP inference server:
```bash
python scripts/inference_service_advantage_conditioned.py \
  --server \
  --http-server \
  --model-path /data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned \
  --data-config ucr_wblm_moby_history \
  --embodiment-tag new_embodiment \
  --port 8000 \
  --host 0.0.0.0
```

### 3. Client Mode (Testing)

Test the server with a client:
```bash
python scripts/inference_service_advantage_conditioned.py \
  --client \
  --host localhost \
  --port 5556
```

## Python API Usage

### Using the Policy Directly

```python
from gr00t.model.advantage_conditioned_policy import AdvantageConditionedGr00tPolicy
from gr00t.experiment.data_config import load_data_config
import numpy as np

# Load data config
data_config = load_data_config("ucr_wblm_moby_history")
modality_config = data_config.modality_config()
modality_transform = data_config.transform()

# Create policy
policy = AdvantageConditionedGr00tPolicy(
    model_path="/data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned",
    embodiment_tag="new_embodiment",
    modality_config=modality_config,
    modality_transform=modality_transform,
    denoising_steps=4,
)

# Create observation - REQUIRED: Must include "indicator" field
obs = {
    "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
    "state.left_arm": np.random.rand(1, 7),
    "state.right_arm": np.random.rand(1, 7),
    "state.left_hand": np.random.rand(1, 6),
    "state.right_hand": np.random.rand(1, 6),
    "state.waist": np.random.rand(1, 3),
    "annotation.human.action.task_description": ["Pick up the box"],
    # ==================== REQUIRED: INDICATOR TOKEN ====================
    "indicator": 1.0,  # REQUIRED: 1.0 = high-quality, 0.0 = low-quality
    # ===================================================================
}

# Get action
action = policy.get_action(obs)
print(action.keys())  # Dict with action modalities
```

### Using the ZMQ Client

```python
from gr00t.eval.robot import RobotInferenceClient
import numpy as np

# Connect to server
client = RobotInferenceClient(host="localhost", port=5556)

# Create observation with indicator
obs = {
    "video.ego_view": ...,
    "state.left_arm": ...,
    "indicator": 1.0,  # High-quality actions
    ...
}

# Get action
action = client.get_action(obs)
```

### Using the HTTP Client

```python
import requests
import json_numpy
json_numpy.patch()
import numpy as np

# Create observation with indicator
obs = {
    "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
    "state.left_arm": np.random.rand(1, 7),
    "indicator": 1.0,  # High-quality actions
    ...
}

# Send request
response = requests.post(
    "http://localhost:8000/act",
    json={"observation": obs}
)

if response.status_code == 200:
    action = response.json()
    print(action)
```

## Indicator Token Specification

### Where to Add the Indicator

The indicator token must be added to the **observation dictionary** with the key `"indicator"`:

```python
obs = {
    # Standard observation fields
    "video.ego_view": ...,
    "state.left_arm": ...,

    # ==================== ADD THIS ====================
    "indicator": 1.0,  # or 0.0
    # ==================================================

    "annotation.human.action.task_description": ["task"],
}
```

### Indicator Values

- **`indicator = 1.0`**: Generate **HIGH-QUALITY** actions
  - Uses demonstrations with advantage >= global_threshold
  - Produces more successful, efficient behaviors

- **`indicator = 0.0`**: Generate **LOW-QUALITY** actions
  - Uses demonstrations with advantage < global_threshold
  - May produce less successful behaviors (useful for debugging/analysis)

### Indicator is REQUIRED

The `indicator` field **must** be present in every observation. If you forget to include it, the policy will raise a clear error:

```python
# This will FAIL with ValueError
obs = {"video.ego_view": ..., "state.left_arm": ...}  # Missing indicator!
action = policy.get_action(obs)  # ❌ Raises: "indicator field required"

# This is correct
obs = {
    "video.ego_view": ...,
    "state.left_arm": ...,
    "indicator": 1.0,  # ✓ Required field
}
action = policy.get_action(obs)  # ✓ Works
```

## Model Flow with Indicator Token

Here's what happens when you pass an observation with an indicator:

1. **Observation Input**
   ```python
   obs = {
       "video.ego_view": (1, 256, 256, 3),
       "state.left_arm": (1, 7),
       "indicator": 1.0,  # <-- Binary indicator
       ...
   }
   ```

2. **Indicator Processing** (in `gr00t_n1.py:forward()`)
   ```python
   # Line 232-252 in gr00t_n1.py
   indicators = inputs["indicator"]  # (batch_size,)

   # Embed indicator: (batch_size,) -> (batch_size, 1, hidden_size)
   indicator_tokens = self.indicator_embedding(indicators)

   # Prepend indicator token to backbone features
   backbone_features = backbone_outputs[BACKBONE_FEATURE_KEY]
   backbone_outputs[BACKBONE_FEATURE_KEY] = torch.cat([
       indicator_tokens,  # <-- First token
       backbone_features
   ], dim=1)
   ```

3. **Action Generation**
   - The action head receives: `[indicator_token, vlm_token_1, vlm_token_2, ...]`
   - The indicator token influences the entire action generation process
   - Output actions match the quality specified by the indicator

## Comparison with Standard Inference

### Standard GR00T Inference Service

```python
# File: scripts/inference_service.py
from gr00t.model.policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.5-3B",
    ...
)

obs = {
    "video.ego_view": ...,
    "state.left_arm": ...,
    # No indicator field
}
action = policy.get_action(obs)
```

### Advantage-Conditioned Inference Service

```python
# File: scripts/inference_service_advantage_conditioned.py
from gr00t.model.advantage_conditioned_policy import AdvantageConditionedGr00tPolicy

policy = AdvantageConditionedGr00tPolicy(
    model_path="/data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned",
    default_indicator=1.0,
    ...
)

obs = {
    "video.ego_view": ...,
    "state.left_arm": ...,
    "indicator": 1.0,  # <-- REQUIRED: Indicator token
}
action = policy.get_action(obs)
```

## Files Created

1. **Policy Class**: `gr00t/model/advantage_conditioned_policy.py`
   - Extends `Gr00tPolicy` with indicator token support
   - Validates that model has advantage conditioning enabled
   - Handles default indicator values

2. **Inference Service**: `scripts/inference_service_advantage_conditioned.py`
   - ZMQ and HTTP server implementations
   - Client examples with indicator tokens
   - Default port 5556 (to avoid conflict with standard service on 5555)

3. **Documentation**: `ADVANTAGE_CONDITIONED_INFERENCE.md`
   - This file - comprehensive usage guide

## Training Pipeline Reference

The advantage-conditioned checkpoint was created using this pipeline:

1. **Value Function Training**: `scripts/gr00t_rl_finetune.py`
   - Trains value head to predict future returns

2. **Advantage Computation**: `scripts/compute_advantages.py`
   - Computes advantages A_t for all demonstrations
   - Generates binary indicators based on global threshold
   - Saves to `reward_labels.json`

3. **Advantage-Conditioned Training**: `scripts/gr00t_advantage_conditioned_train.py`
   - Trains action head with indicator embedding
   - Uses advantage-weighted loss: wt = sigmoid(k * (A_t - threshold))
   - Saves checkpoint with `enable_advantage_conditioning=True`

4. **Inference**: `scripts/inference_service_advantage_conditioned.py` ← **YOU ARE HERE**
   - Deploy the trained model for inference
   - Control action quality via indicator token

## Troubleshooting

### Error: "indicator field required"

This means you forgot to add the `indicator` field to your observation dictionary. Fix:
```python
obs["indicator"] = 1.0  # Add this line
```

### Error: "Model does not have advantage conditioning enabled"

This means you're trying to use a standard checkpoint with the advantage-conditioned policy. Make sure you're loading a checkpoint trained with `gr00t_advantage_conditioned_train.py`.

### Indicator not having an effect

Make sure:
1. The indicator field is named exactly `"indicator"` (lowercase)
2. The value is 0.0 or 1.0 (not 0/1 integers)
3. The checkpoint was trained with advantage conditioning

### Missing indicator_embedding in checkpoint

If you get an error about missing `indicator_embedding` weights, your checkpoint wasn't trained with advantage conditioning. You need to use a checkpoint from `scripts/gr00t_advantage_conditioned_train.py`.

## Example: Full Deployment Workflow

```bash
# Terminal 1: Start server
python scripts/inference_service_advantage_conditioned.py \
  --server \
  --model-path /data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned \
  --data-config ucr_wblm_moby_history \
  --embodiment-tag new_embodiment \
  --port 5556

# Terminal 2: Test with client
python scripts/inference_service_advantage_conditioned.py \
  --client \
  --port 5556
```

## Summary

**Key Points:**
1. **Indicator is REQUIRED**: The policy will raise an error if `indicator` is missing
2. Indicator position: Added as `obs["indicator"]` in the observation dictionary
3. Network difference: Has `indicator_embedding` layer that prepends a token to backbone features
4. Use `indicator=1.0` for high-quality actions in production
5. Use `indicator=0.0` for debugging or analyzing failure modes
6. No default value - you must explicitly specify the indicator for every observation
