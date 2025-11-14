# Multi-Camera Training Setup for Isaac GR00T

This guide explains how to train Isaac GR00T models with multi-camera observations.

## Overview

The Isaac GR00T codebase already supports multi-camera training. The key is properly configuring:
1. **Dataset**: LeRobot format with multiple video modalities
2. **Data Config**: Specifying multiple `video_keys`
3. **Model**: Automatically handles multi-view input

---

## 1. Dataset Preparation (UCR VLA)

### 1.1 Generate Multi-Camera Dataset

Use the UCR VLA data processing script with multi-camera enabled:

```bash
cd /data/anthony/ucr_vla

python src/ucr_vla/data/generate_dataset_v2.py \
    --bags /path/to/bag1 /path/to/bag2 \
    --output dataset_name \
    --fps 30 \
    --root /output/directory
```

**Note**: Default is multi-camera. For single camera only, add `--single-camera` flag.

### 1.2 Update modality.json

In your dataset's `meta/modality.json`, add the second camera:

```json
{
    "state": {
        "waist_joint": {"start": 0, "end": 3},
        "right_arm_joint": {"start": 3, "end": 8},
        ...
    },
    "action": {
        "behavior_mode": {"start": 0, "end": 1},
        "left_ee_position": {"start": 3, "end": 6},
        ...
    },
    "video": {
        "ego_view": {
            "original_key": "observation.images.torso_camera"
        },
        "ego_view_up": {
            "original_key": "observation.images.torso_camera_up"
        }
    },
    "annotation": {
        "human.action.task_description": {
            "original_key": "task_index"
        }
    }
}
```

**Key points**:
- `ego_view` → `torso_camera` (main camera)
- `ego_view_up` → `torso_camera_up` (second camera)
- Both must exist in your dataset's video files

---

## 2. Isaac GR00T Training Configuration

### 2.1 Available Data Configs

We've added two new configs in `gr00t/experiment/data_config.py`:

| Config Name | Cameras | Temporal History | Action Horizon |
|-------------|---------|------------------|----------------|
| `ucr_wblm_moby` | 1 (ego_view) | No (single frame) | 16 |
| `ucr_wblm_moby_dualcam` | 2 (ego_view + ego_view_up) | No (single frame) | 16 |
| `ucr_wblm_moby_history` | 1 (ego_view) | Yes (2 frames: -30, 0) | 64 |
| `ucr_wblm_moby_dualcam_history` | 2 (ego_view + ego_view_up) | Yes (2 frames: -30, 0) | 64 |

### 2.2 Training Commands

#### Single Camera (Baseline)
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --data-config ucr_wblm_moby \
    --output-dir /output/single_cam \
    --batch-size 32 \
    --max-steps 10000
```

#### Dual Camera (Multi-View)
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --data-config ucr_wblm_moby_dualcam \
    --output-dir /output/dual_cam \
    --batch-size 32 \
    --max-steps 10000
```

#### Dual Camera + Temporal History
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --data-config ucr_wblm_moby_dualcam_history \
    --output-dir /output/dual_cam_history \
    --batch-size 32 \
    --max-steps 10000
```

#### Dynamic Video History Override (Advanced)
```bash
python scripts/gr00t_finetune.py \
    --dataset-path /path/to/dataset \
    --data-config ucr_wblm_moby_dualcam \
    --output-dir /output/dual_cam_custom \
    --video-history-enabled \
    --video-observation-indices "-30,-15,0" \
    --batch-size 32 \
    --max-steps 10000
```

---

## 3. How Multi-Camera Works Internally

### 3.1 Data Flow

```
Dataset Loading:
├─ video.ego_view     → [T, H, W, C]  (e.g., [2, 224, 224, 3] with history)
└─ video.ego_view_up  → [T, H, W, C]  (e.g., [2, 224, 224, 3] with history)

ConcatTransform (concat.py:100-112):
├─ Expand dims: [..., H, W, C] → [..., 1, H, W, C]
├─ Concatenate along V axis
└─ Output: [T, V, H, W, C]  (e.g., [2, 2, 224, 224, 3])
   where T=temporal frames, V=number of cameras

GR00TTransform (transforms.py:216-223):
├─ Rearrange: [T, V, H, W, C] → [V, T, C, H, W]
└─ Process through vision encoder (handles multi-view)

Model Input:
└─ Vision encoder processes all views together
```

### 3.2 Key Code Components

**ConcatTransform** (`gr00t/data/transform/concat.py`):
```python
# Lines 100-112
for video_key in self.video_concat_order:
    video_data = data.pop(video_key)
    unsqueezed_video = np.expand_dims(video_data, axis=-4)  # Add V dimension
    unsqueezed_videos.append(unsqueezed_video)

# Concatenate along V axis
unsqueezed_video = np.concatenate(unsqueezed_videos, axis=-4)  # [T, V, H, W, C]
data["video"] = unsqueezed_video
```

**GR00TTransform** (`gr00t/model/transforms.py`):
```python
# Line 219-222
def _prepare_video(self, data: dict):
    images = rearrange(
        data["video"],
        "t v h w c -> v t c h w",  # Rearrange for model input
    )
    return images
```

---

## 4. Creating Custom Multi-Camera Configs

If you need a custom configuration:

```python
# In gr00t/experiment/data_config.py

class MyCustomDualCamConfig(BaseDataConfig):
    # Specify both cameras
    video_keys = ["video.ego_view", "video.ego_view_up"]

    state_keys = [
        "state.waist_joint",
        # ... your state keys
    ]

    action_keys = [
        "action.left_ee_position",
        # ... your action keys
    ]

    language_keys = ["annotation.human.action.task_description"]

    # Configure temporal sampling
    video_observation_indices = [-30, 0]  # 2 temporal frames
    state_observation_indices = [0]       # Current state only
    action_indices = list(range(64))      # 64-step action horizon

    def transform(self) -> ModalityTransform:
        transforms = [
            # Video transforms (applied to BOTH cameras)
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224),
            VideoToNumpy(apply_to=self.video_keys),

            # State/action transforms
            StateActionToTensor(apply_to=self.state_keys + self.action_keys),
            StateActionSinCosTransform(apply_to=self.state_keys),

            # CRITICAL: ConcatTransform merges multiple cameras
            ConcatTransform(
                video_concat_order=self.video_keys,  # Order matters!
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),

            # Model transform
            GR00TTransform(
                state_horizon=len(self.state_observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
```

Then register it:
```python
DATA_CONFIG_MAP = {
    # ...
    "my_custom_dualcam": MyCustomDualCamConfig(),
}
```

---

## 5. Verification

### 5.1 Check Dataset

```python
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import load_data_config

data_config = load_data_config("ucr_wblm_moby_dualcam")
modality_configs = data_config.modality_config()

dataset = LeRobotSingleDataset(
    dataset_path="/path/to/dataset",
    modality_configs=modality_configs,
    transforms=data_config.transform(),
    embodiment_tag="new_embodiment",
)

# Sample a batch
sample = dataset[0]
print(f"Video shape: {sample['video'].shape}")  # Should be [T, V, H, W, C]
print(f"State shape: {sample['state'].shape}")  # [T_state, D_state]
print(f"Action shape: {sample['action'].shape}")  # [T_action, D_action]
```

Expected output:
```
Video shape: torch.Size([2, 2, 224, 224, 3])  # [T=2, V=2, H=224, W=224, C=3]
State shape: torch.Size([1, 58])              # [T=1, D=58]
Action shape: torch.Size([16, 30])            # [T=16, D=30]
```

### 5.2 Training Logs

During training, you should see:
```
==================================================
VIDEO HISTORY ENABLED (if using --video-history-enabled)
==================================================
Original video observation indices: [0]
New video observation indices: [-30, 0]
State observation indices (unchanged): [0]
Number of video frames per observation: 2
Number of cameras per frame: 2
==================================================
```

---

## 6. Troubleshooting

### Issue: "Key 'video.ego_view_up' not found in modality metadata"

**Solution**: Update your dataset's `meta/modality.json` to include the second camera.

### Issue: Video shape mismatch

**Solution**: Ensure both cameras have the same resolution in `meta/info.json`.

### Issue: Model trains but performance is poor

**Possible causes**:
1. Cameras are misaligned (different timestamps)
2. One camera has poor quality images
3. Need more training steps (multi-view requires more data)

**Solution**:
- Verify timestamp alignment in your ROS2 bags
- Check camera calibration
- Increase `--max-steps` to 20000+

---

## 7. Summary

| Feature | Single Camera | Dual Camera | Dual Camera + History |
|---------|---------------|-------------|----------------------|
| Config | `ucr_wblm_moby` | `ucr_wblm_moby_dualcam` | `ucr_wblm_moby_dualcam_history` |
| Video Keys | 1 | 2 | 2 |
| Temporal Frames | 1 | 1 | 2 |
| Video Shape | `[1, 1, 224, 224, 3]` | `[1, 2, 224, 224, 3]` | `[2, 2, 224, 224, 3]` |
| Action Horizon | 16 | 16 | 64 |

The infrastructure fully supports multi-camera training - you just need to:
1. ✅ Prepare dataset with multiple video modalities
2. ✅ Use appropriate data config
3. ✅ Train normally (model handles multi-view automatically)
