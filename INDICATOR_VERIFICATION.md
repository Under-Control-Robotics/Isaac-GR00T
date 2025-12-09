# Indicator Data Verification

This document describes the verification checks added to ensure indicators are correctly loaded and used for advantage-conditioned training.

## Verification Points

### 1. Dataset Loading Verification (`gr00t/data/dataset.py:978-1010`)

When loading advantage labels, the dataset automatically verifies:

**✓ Binary Indicators**: Ensures all indicators are exactly 0.0 or 1.0
```python
assert np.all(np.isin(unique_vals, [0.0, 1.0]))
```

**✓ Alignment with Advantages**: Checks that indicators match the threshold formula
```python
expected_indicators = (advantages >= global_threshold).astype(np.float32)
accuracy = correct_indicators / total_steps
```

**✓ Distribution Check**: Reports the ratio of good (1) vs bad (0) actions
```
Good action ratio: 69.5%
```

**Output Example:**
```
Loaded advantage labels for 120 episodes
  Global advantage threshold: -0.004991
  Dataset good action ratio (I_t=1): 69.5%
  Advantage computation method: smooth

  Verifying indicator-advantage alignment...
  ✓ Indicator alignment accuracy (first 5 episodes): 100.0%
  ✓ Indicator distribution (first 5 episodes): 4523 ones, 1982 zeros
    Good action ratio: 69.5%
```

### 2. Batch Loading Verification (`scripts/gr00t_advantage_conditioned_train.py:320-361`)

Before training starts, the script loads a test batch and verifies:

**✓ Indicator Field Exists**: Checks `"indicator"` is in the batch
**✓ Shape and Dtype**: Verifies tensor shape and data type
**✓ Binary Values**: Confirms only 0 and 1 are present
**✓ Distribution**: Shows the ratio in the batch

**Output Example:**
```
================================================================================
VERIFYING INDICATOR DATA IN FIRST BATCH
================================================================================
✓ Indicator found in batch
  Shape: torch.Size([4, 16])
  Dtype: torch.float32
  Range: [0.0, 1.0]
  Unique values: [0.0, 1.0]
  First sample indicators: [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
✓ Indicators are binary (0 or 1)
  Distribution: 44/64 (68.8%) are 1 (good), 20/64 (31.2%) are 0 (bad)
================================================================================
```

### 3. Model Forward Pass Verification (`gr00t/model/gr00t_n1.py:240-252`)

On the first forward pass, the model logs:

**✓ Input Indicators**: Shows the raw indicator values
**✓ Distribution**: Counts how many are 1 vs 0
**✓ Token Shapes**: Verifies tensor shapes before and after prepending
**✓ Successful Prepending**: Confirms indicator token is the FIRST token

**Output Example:**
```
================================================================================
[MODEL] Advantage conditioning verification (first forward pass):
================================================================================
  Input indicators (first 10): [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
  Unique indicator values: [0.0, 1.0]
  Distribution: 22/32 are 1 (good)
  Indicator tokens shape: torch.Size([32, 1, 4096])
  Backbone features shape (before): torch.Size([32, 128, 4096])
  Backbone features shape (after prepending): torch.Size([32, 129, 4096])
  ✓ Indicator token successfully prepended as FIRST token
================================================================================
```

## Pre-Training Test Script

Run the comprehensive test before training:

```bash
python scripts/test_advantage_conditioning.py
```

This script runs three tests:

1. **Dataset Labels Check**: Verifies all datasets have indicator data in reward_labels.json
2. **Dataset Loading Test**: Loads a sample and verifies indicator format
3. **Model Forward Pass Test**: Runs a forward pass to verify end-to-end pipeline

**Expected Output:**
```
================================================================================
ADVANTAGE-CONDITIONED TRAINING VERIFICATION TEST
================================================================================

STEP 1: Checking dataset label files
✓ All datasets have indicator labels

STEP 2: Testing dataset loading with advantage conditioning
✓ Dataset loading successful

STEP 3: Testing model forward pass with indicators
✓ Model forward pass successful

================================================================================
✓ ALL TESTS PASSED!
================================================================================
```

## What Gets Verified

| Check | Location | What It Verifies |
|-------|----------|------------------|
| Indicators are binary | Dataset loading | All values are exactly 0.0 or 1.0 |
| Indicator-advantage alignment | Dataset loading | Indicators = (advantage >= threshold) |
| Good action ratio | Dataset loading | ~70% of actions should be "good" |
| Indicator in batch | Training script | Dataloader correctly passes indicators |
| Batch distribution | Training script | Batch has expected good/bad ratio |
| Token shape | Model forward | Indicator embedding produces correct shape |
| Token prepending | Model forward | Indicator token is first in sequence |
| Sequence length | Model forward | Seq length increases by 1 after prepending |

## Data Flow Verification

```
reward_labels.json
  ├─ "indicators": [1, 0, 1, ...]  ✓ Binary values
  ├─ "advantages": [0.005, -0.036, ...]  ✓ Align with threshold
  └─ metadata["advantage_computation"]["global_threshold"]  ✓ Loaded

      ↓

Dataset.__getitem__()
  └─ data["indicator"]: np.array([1, 0, 1, ...])  ✓ Shape (action_horizon,)

      ↓

DataLoader collate
  └─ batch["indicator"]: torch.Tensor([[1,0,1,...], ...])  ✓ Shape (B, action_horizon)

      ↓

Model.forward()
  ├─ indicators[:, 0]  ✓ Take first timestep (B,)
  ├─ indicator_embedding(indicators)  ✓ Embed to (B, 1, hidden_size)
  └─ cat([indicator_tokens, backbone_features])  ✓ Prepend as first token
```

## Troubleshooting

### Indicators not binary
**Symptom**: Unique values are not [0.0, 1.0]
**Fix**: Re-run `compute_advantages.py` to regenerate indicators

### Alignment accuracy < 99%
**Symptom**: Indicators don't match advantages with threshold
**Fix**: Check that `compute_advantages.py` used the same threshold formula

### "indicator" not in batch
**Symptom**: Batch doesn't contain indicator field
**Fix**: Ensure `enable_advantage_conditioning=True` in dataset loading

### Shape mismatch
**Symptom**: Indicator tensor has wrong shape
**Fix**: Check that action_horizon matches between data config and model

## Expected Training Behavior

Once verified, during training:
- Model receives indicator tokens for each batch
- Indicator token is prepended as the FIRST token before VLM tokens
- Action head attends to [indicator_token, vlm_token_1, vlm_token_2, ...]
- Policy learns:
  - When indicator=1 (good): generate similar actions
  - When indicator=0 (bad): avoid similar actions
