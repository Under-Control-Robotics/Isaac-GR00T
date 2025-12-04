# RECAP/Pi-Style Training Guide

This guide explains how to train an advantage-conditioned policy using the RECAP/Pi approach in Isaac GR00T.

## Overview

The RECAP/Pi training pipeline consists of three main steps:

1. **Train a value function** (already done with `gr00t_rl_finetune.py`)
2. **Compute advantages and generate indicators** (use `compute_advantages.py`)
3. **Train an advantage-conditioned policy** (use `gr00t_recap_train.py`)

## Step-by-Step Guide

### Step 1: Train Value Function

First, train a value function using your dataset with reward labels:

```bash
python scripts/gr00t_rl_finetune.py \
    --dataset_path /path/to/dataset \
    --output_dir /tmp/gr00t_value_head \
    --base_model_path nvidia/GR00T-N1.5-3B \
    --max_steps 10000 \
    --batch_size 32
```

**Requirements:**
- Your dataset must have `reward_labels.json` file with reward and value targets
- This trains ONLY the value head while freezing the policy

**Output:**
- Trained model with value function at `/tmp/gr00t_value_head`

---

### Step 2: Compute Advantages and Generate Indicators

Use the trained value function to compute advantages and generate binary indicators:

```bash
python scripts/compute_advantages.py \
    --model_path /tmp/gr00t_value_head \
    --dataset_path /path/to/dataset \
    --output_dir /tmp/gr00t_advantages \
    --advantage_quantile 0.5 \
    --batch_size 32
```

**Parameters:**
- `model_path`: Path to the trained model with value function
- `dataset_path`: Path to your dataset(s)
- `advantage_quantile`: Threshold quantile (0.5 = top 50%, 0.7 = top 30%)
  - 0.5 means actions with advantage > median are labeled as "good" (I_t=1)
  - 0.7 means only top 30% are labeled as "good"

**What it does:**
1. Loads your trained value function
2. Predicts values V(o_t) for all timesteps
3. Computes advantages: **A_t = G_norm[t] - V(o_t)**
4. Computes threshold: **ε = quantile(A_t, q)**
5. Generates indicators: **I_t = 1 if A_t > ε else 0**
6. Saves to `advantage_labels.json` in your dataset directory

**Output:**
- `advantage_labels.json` in each dataset directory
- Contains binary indicators and advantages for each timestep

**Example output:**
```json
{
  "metadata": {
    "advantage_quantile": 0.5,
    "threshold": 0.023,
    "good_ratio": 0.50
  },
  "episodes": [
    {
      "episode_index": 0,
      "length": 100,
      "indicators": [1, 1, 0, 1, ...],  // I_t values
      "advantages": [0.05, 0.03, -0.01, ...]  // A_t values
    }
  ]
}
```

---

### Step 3: Train Advantage-Conditioned Policy

Train a policy conditioned on the binary indicators:

```bash
python scripts/gr00t_recap_train.py \
    --dataset_path /path/to/dataset \
    --output_dir /tmp/gr00t_recap_policy \
    --base_model_path nvidia/GR00T-N1.5-3B \
    --max_steps 50000 \
    --batch_size 32 \
    --tune_projector True \
    --tune_diffusion_model True
```

**What it does:**
- Loads the dataset with `enable_advantage_conditioning=True`
- Each batch includes indicators: `data["indicator"]` with shape `[B, T]`
- Trains the policy on (observation, action, indicator) tuples

**Note:** The current implementation loads indicators but does not yet inject them into the model. To fully implement RECAP, you need to modify the action head to use indicators as conditioning.

---

## Full Implementation (To-Do)

To complete the RECAP implementation, you need to modify the action head:

### 1. Add Indicator Embedding Layer

In `gr00t/model/action_head/flow_matching_action_head.py`:

```python
class FlowmatchingActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ... existing code ...

        # Add indicator embedding
        self.indicator_embedding = nn.Linear(1, self.input_embedding_dim)
```

### 2. Modify Forward Pass

Inject indicator embeddings into the conditioning:

```python
def forward(self, backbone_output, action_input):
    # ... existing code to get state_features ...

    # Get indicator from input
    if "indicator" in action_input:
        indicator = action_input.indicator  # shape: (B, T)
        indicator_emb = self.indicator_embedding(indicator.unsqueeze(-1))  # (B, T, D)

        # Add to state features
        state_features = state_features + indicator_emb.mean(dim=1, keepdim=True)

    # ... rest of forward pass ...
```

### 3. At Inference Time

You can control the policy behavior by setting the indicator:

```python
# For good actions (high advantage)
action_input["indicator"] = torch.ones(batch_size, action_horizon)

# For average actions
action_input["indicator"] = torch.zeros(batch_size, action_horizon)
```

---

## RECAP Theory

### Advantage Computation

For each timestep t:

```
A_t = G_norm[t] - V(o_t)
```

Where:
- `G_norm[t]` = normalized return-to-go (from reward labels)
- `V(o_t)` = predicted value from trained value function
- `A_t` = advantage (how much better than expected)

### Binary Indicator

```
I_t = 1  if A_t > ε
I_t = 0  otherwise
```

Where ε is the quantile threshold (e.g., 50th percentile).

**Interpretation:**
- `I_t = 1`: Action is better than typical → learn from this
- `I_t = 0`: Action is average or worse → ignore or learn to avoid

### Policy Training

The policy is trained to predict actions conditioned on:
- Observation o_t
- Indicator I_t

At inference, you can sample actions with `I_t=1` to get high-advantage behaviors.

---

## Dataset Requirements

Your dataset must have:

1. **For value training:** `reward_labels.json`
   - Contains rewards and normalized returns (G_norm)

2. **For policy training:** `advantage_labels.json`
   - Generated by `compute_advantages.py`
   - Contains binary indicators and advantages

---

## Troubleshooting

### Error: "Advantage labels not found"
→ Run `compute_advantages.py` first to generate indicators

### Error: "Reward labels not found"
→ Your dataset needs reward labels for value training

### Low good_ratio after computing advantages
→ Adjust `advantage_quantile` parameter (higher = fewer good actions)

### Model not using indicators
→ You need to implement indicator conditioning in the action head (see "Full Implementation" above)

---

## References

- **Pi (Preference-Informed RL)**: Uses advantage indicators for policy improvement
- **RECAP**: Regularized Critic Advantage-based Policy optimization
- Both methods use binary indicators based on advantage thresholding

---

## Example Complete Workflow

```bash
# 1. Train value function
python scripts/gr00t_rl_finetune.py \
    --dataset_path /data/my_robot_dataset \
    --output_dir /models/value_head \
    --max_steps 10000

# 2. Compute advantages (top 50%)
python scripts/compute_advantages.py \
    --model_path /models/value_head \
    --dataset_path /data/my_robot_dataset \
    --advantage_quantile 0.5

# 3. Train advantage-conditioned policy
python scripts/gr00t_recap_train.py \
    --dataset_path /data/my_robot_dataset \
    --output_dir /models/recap_policy \
    --max_steps 50000
```

Your final model at `/models/recap_policy` will be conditioned on advantage indicators!
