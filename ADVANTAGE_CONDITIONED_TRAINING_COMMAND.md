# Advantage-Conditioned Policy Training

## Overview
This training pipeline trains an action head policy conditioned on binary advantage indicators (0 or 1) computed from a trained value function.

## Architecture
```
Video/State -> Backbone -> VLM tokens
Indicator (0/1) -> Indicator Embedding -> indicator_token

[indicator_token] + [VLM tokens] -> Action Head -> Actions
```

The indicator token is prepended as the FIRST token before VLM tokens, allowing the policy to be conditioned on advantage signals.

## Training Command

```bash
python scripts/gr00t_advantage_conditioned_train.py \
    --dataset-path /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_0 \
                   /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_3 \
                   /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_4 \
    --data-config ucr_wblm_moby_history \
    --output-dir /data/anthony/Isaac-GR00T/checkpoints/advantage_conditioned \
    --base-model-path nvidia/GR00T-N1.5-3B \
    --batch-size 32 \
    --max-steps 10000 \
    --learning-rate 1e-4 \
    --tune-action-head \
    --indicator-embedding-dim 4096
```

## What Gets Trained
- ✅ **Indicator Embedding**: Maps {0, 1} -> hidden dimension
- ✅ **Action Head Projector**: Cross-attention and projection layers
- ✅ **Action Head DiT**: Flow-matching diffusion transformer
- ❌ **Backbone**: Frozen (vision + LLM)
- ❌ **Value Head**: Not used in this training

## Dataset Requirements
Each dataset must have a `reward_labels.json` file with the following structure:
```json
{
  "episodes": [
    {
      "episode_index": 0,
      "indicators": [1, 1, 0, 1, ...],  // Binary indicators per timestep
      "advantages": [0.5, 0.3, -0.2, ...],  // Advantage values (for reference)
      "threshold": -0.15  // Global threshold used for indicators
    }
  ]
}
```

Generate these files using:
```bash
python scripts/compute_advantages.py \
    --checkpoint /path/to/value_head_checkpoint \
    --dataset-path /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_0 \
                   /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_3 \
                   /data/anthony/ucr_vla/output/1205_discount/pick_up_box_1205_4 \
    --good-ratio 0.7
```

## Implementation Files
1. `/data/anthony/Isaac-GR00T/gr00t/model/indicator_embedding.py` - Indicator embedding module
2. `/data/anthony/Isaac-GR00T/gr00t/model/gr00t_n1.py` - Model with advantage conditioning support
3. `/data/anthony/Isaac-GR00T/scripts/gr00t_advantage_conditioned_train.py` - Training script
4. `/data/anthony/Isaac-GR00T/gr00t/data/dataset.py` - Dataset loading with indicator support

## Expected Behavior
- Policy learns to generate better actions when `indicator=1` (good advantage)
- Policy learns to avoid actions when `indicator=0` (bad advantage)
- Creates a conditional policy that can be guided by advantage signals

## Next Steps After Training
Use the trained model for inference by providing indicator values based on desired behavior quality.
