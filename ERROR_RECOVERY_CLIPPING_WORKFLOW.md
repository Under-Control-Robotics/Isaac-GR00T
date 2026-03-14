# Error Recovery Dataset Clipping - Complete Workflow

## Overview
This document explains how error recovery dataset clipping works in the training pipeline.

## Key Concept
- **Training clips**: Only sample from frames [30, len-60] for error recovery datasets
- **Data access**: Full episode [0, len-1] still accessible for state history and action chunks
- **Upsampling**: 10x weight multiplier applied to error recovery datasets

## Complete Workflow

### 1. Dataset Initialization (gr00t_finetune.py)

```python
# For each dataset path:
dataset = LeRobotSingleDataset(
    dataset_path=path,  # e.g., .../error_recovery/.../2026-02-27_08:14:08.248872
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",
)

# During initialization, _get_all_steps() is called:
# - Normal datasets: all_steps = [(0,0), (0,1), ..., (0,len-1), (1,0), ...]
# - Error recovery: all_steps = [(0,30), (0,31), ..., (0,len-61), (1,30), ...]
```

**Example:**
- Normal dataset with trajectory length=5471
  - all_steps has 5471 entries: frames [0, 5470]

- Error recovery dataset with trajectory length=130
  - all_steps has 40 entries: frames [30, 69]
  - Skips first 30 frames (1s @ 30fps)
  - Skips last 60 frames (2s @ 30fps)

### 2. Mixture Dataset Creation

```python
# Apply 10x upsampling to error recovery datasets
weight = 10.0 if "error_recovery" in path else 1.0

# Create mixture
mixture_dataset = LeRobotMixtureDataset(
    data_mixture=[(dataset, weight) for dataset, weight in zip(datasets, weights)],
    balance_dataset_weights=True,  # Multiply weight by total trajectory length
    balance_trajectory_weights=True,  # Weight trajectories by their length
)
```

**Effective weights:**
- Normal dataset (112,290 steps): weight = 1.0 × 112,290 = 112,290
- Error recovery (467 steps): weight = 10.0 × 467 = 4,670
- Normalized: Normal=96.0%, Error=4.0%

### 3. Training Loop - Sampling

```python
# For each training iteration:
dataset, trajectory_id, base_index = mixture_dataset.sample_step(index)

# sample_step() flow:
# 1. Sample a dataset (96% normal, 4% error recovery)
# 2. Sample a trajectory from that dataset
# 3. Sample a base_index from VALID INDICES for that trajectory
#    - Uses get_valid_indices_for_trajectory()
#    - For error recovery: returns [30, len-60)
#    - For normal: returns [0, len)
```

**Example error recovery sample:**
- Sampled trajectory_id=0, base_index=45
- This is WITHIN the clipped range [30, 69] ✓

### 4. Data Fetching

```python
# Fetch data for the sampled step:
data = dataset.get_step_data(trajectory_id, base_index)

# For error recovery at base_index=45:
# State history indices: [-30, -27, -24, -21, -18, -15, -12, -9, -6, -3, 0]
#   -> Accesses frames: [15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]
#   -> Goes OUTSIDE clipped range to frame 15 ✓ (allowed!)
#
# Action chunk indices: [0, 1, 2, ..., 63]
#   -> Accesses frames: [45, 46, 47, ..., 108]
#   -> Goes OUTSIDE clipped range to frame 108 ✓ (allowed!)
#
# Video observation indices: [0]
#   -> Accesses frame: [45]
```

**Key point:** Even though training only samples from [30, 69], the data loading functions (`get_state_or_action()`, `get_video()`) can access the FULL episode [0, 129] to get history and future frames.

### 5. Data Padding

```python
# retrieve_data_and_pad() handles out-of-bounds indices:
# - State history going before frame 0: pad with first frame
# - Action chunks going beyond last frame: pad with last frame
```

**Example at base_index=30 (earliest valid sample):**
- State history needs frame 0 (base_index -30): ✓ Available
- State history needs frame 3 (base_index -27): ✓ Available
- All frames [0, 30] are accessible

**Example at base_index=69 (latest valid sample for traj_len=130):**
- Action chunk needs frame 69+0: ✓ Available
- Action chunk needs frame 69+63=132: Padded with frame 129

## Visual Diagram

```
Error Recovery Episode (length=130):

Frame:  0    30              69            129
        |====|===============|=============|
        ^    ^               ^             ^
        |    |               |             |
        |    Start train     End train     End episode
        |    (skip 1s)       (2s before)
        |
        Start episode

Training samples: [30, 31, 32, ..., 69]  (40 samples)
State history can access: [0, 3, 6, ..., 69]
Action chunks can access: [30, 31, ..., 129]

Example sample at frame 45:
  State history: [15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45]  ✓
  Video obs:     [45]                                           ✓
  Action chunk:  [45, 46, 47, ..., 108]                        ✓
```

## Verification Results

### Test 1: Single Dataset
- ✅ Normal dataset: 112,290 steps (all frames)
- ✅ Error recovery: 467 steps (clipped to [30, len-60])

### Test 2: Mixture Dataset Sampling
- ✅ 1000 samples: 959 normal, 41 error recovery
- ✅ All error recovery samples in range [32, 133]
- ✅ All samples >= 30 and < trajectory_length - 60

### Test 3: Data Fetching
- ✅ State history accesses frames before clipping start
- ✅ Action chunks access frames after clipping end
- ✅ No index out of bounds errors

## Files Modified

1. **`/data/anthony/Isaac-GR00T/gr00t/data/dataset.py`**
   - `_get_all_steps()`: Clips error recovery datasets to [30, len-60]
   - `get_valid_indices_for_trajectory()`: Returns valid indices for trajectory
   - `LeRobotMixtureDataset.sample_step()`: Uses valid indices for sampling

2. **`/data/anthony/Isaac-GR00T/scripts/gr00t_finetune.py`**
   - Already has 10x upsampling: `error_recovery_upsample_factor = 10.0`
   - No changes needed!

## Usage

Simply run your existing training script:

```bash
bash train_error_recovery_10x.sh
```

The clipping will automatically apply to all datasets with "error_recovery" in their path.

## Expected Output

When the script runs, you'll see:
```
Error recovery dataset clipping: Generated 467 training steps (skipping first 30 and last 60 frames of each episode)
Initialized dataset 2026-02-27_08:14:08.248872 with new_embodiment
...
Loaded 95 datasets:
  - 74 regular datasets (weight=1.0)
  - 21 error_recovery datasets (weight=10.0)
```

This confirms:
- Error recovery clipping is active
- 10x upsampling is applied
- All datasets loaded successfully
