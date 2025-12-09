# Advantage-Conditioned Policy Implementation Plan

## Overview
Train a policy conditioned on advantage indicators by embedding the indicator (0 or 1) and prepending it to VLM tokens before the action head processing.

## Architecture Changes

### 1. Data Flow
```
Current:
  Video/State -> Backbone -> VLM tokens -> vlln -> vl_self_attention -> Action Head

New (Advantage-Conditioned):
  Video/State -> Backbone -> VLM tokens
  Indicator (0/1) -> Indicator Embedding
  [Indicator Token] + [VLM tokens] -> vlln -> vl_self_attention -> Action Head
```

### 2. Model Components to Add

#### A. Indicator Embedding Module (`gr00t/model/indicator_embedding.py`)
```python
class IndicatorEmbedding(nn.Module):
    """
    Embeds binary advantage indicator (0 or 1) into hidden dimension.

    Args:
        hidden_size: Dimension to match backbone features (e.g., 4096)
        num_indicators: Number of indicator values (2 for binary 0/1)
    """
    def __init__(self, hidden_size: int = 4096, num_indicators: int = 2):
        super().__init__()
        # Embedding table: [0, 1] -> hidden_size
        self.embedding = nn.Embedding(num_indicators, hidden_size)
        # Optional: learned position embedding for indicator token
        self.position_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, indicators: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indicators: (batch_size,) tensor of 0s and 1s
        Returns:
            indicator_tokens: (batch_size, 1, hidden_size)
        """
        # Embed indicators: (B,) -> (B, 1, hidden_size)
        indicator_emb = self.embedding(indicators).unsqueeze(1)
        # Add position embedding
        indicator_tokens = indicator_emb + self.position_embedding
        return indicator_tokens
```

#### B. Model Config Updates (`GR00T_N1_5_Config`)
Add fields:
```python
enable_advantage_conditioning: bool = False
indicator_embedding_dim: int = 4096  # Match backbone hidden_size
```

#### C. Model Initialization (`GR00T_N1_5.__init__`)
```python
# Add indicator embedding if advantage conditioning enabled
self.enable_advantage_conditioning = getattr(config, "enable_advantage_conditioning", False)
if self.enable_advantage_conditioning:
    from .indicator_embedding import IndicatorEmbedding
    self.indicator_embedding = IndicatorEmbedding(
        hidden_size=config.indicator_embedding_dim
    )
```

### 3. Forward Pass Modifications (`GR00T_N1_5.forward`)

**Current flow:**
```python
backbone_outputs = self.backbone(backbone_inputs)  # (B, seq_len, hidden_size)
action_outputs = self.action_head(backbone_outputs, action_inputs)
```

**New flow (with advantage conditioning):**
```python
backbone_outputs = self.backbone(backbone_inputs)  # (B, seq_len, hidden_size)

# If advantage conditioning enabled, prepend indicator token
if self.enable_advantage_conditioning and "indicator" in inputs:
    indicators = inputs["indicator"]  # (B,) or (B, action_horizon)
    # Take first timestep indicator
    if indicators.dim() > 1:
        indicators = indicators[:, 0]  # (B,)

    # Embed indicator
    indicator_tokens = self.indicator_embedding(indicators)  # (B, 1, hidden_size)

    # Prepend to backbone features
    backbone_features = backbone_outputs["backbone_features"]  # (B, seq_len, hidden_size)
    backbone_outputs["backbone_features"] = torch.cat([
        indicator_tokens, backbone_features
    ], dim=1)  # (B, seq_len+1, hidden_size)

action_outputs = self.action_head(backbone_outputs, action_inputs)
```

### 4. Dataset Loading Changes

**In training script:**
```python
# Load dataset with advantage conditioning enabled
dataset = LeRobotSingleDataset(
    dataset_path=path,
    modality_configs=modality_configs,
    transforms=transforms,
    embodiment_tag=embodiment_tag,
    video_backend=video_backend,
    enable_rl=False,  # Not training value head
    enable_advantage_conditioning=True,  # NEW: Load indicators
)
```

**Dataset will load from reward_labels.json:**
```json
{
  "episodes": [
    {
      "episode_index": 0,
      "indicators": [1, 1, 0, 1, ...],  // Per-step indicators
      "advantages": [...],
      ...
    }
  ]
}
```

### 5. Cross-Attention Considerations

The action head's cross-attention will automatically handle the increased sequence length:
- **Input:** `(B, seq_len+1, hidden_size)` where +1 is the indicator token
- **Cross-attention** in action head already handles variable sequence lengths
- **No changes needed** - the attention mechanism will attend to all tokens including the indicator

### 6. Training Script (`scripts/gr00t_advantage_conditioned_train.py`)

**Key differences from RL finetuning:**
1. Enable `enable_advantage_conditioning=True` instead of `enable_rl=True`
2. Train action head instead of value head:
   ```python
   model = GR00T_N1_5.from_pretrained(
       pretrained_model_name_or_path=config.base_model_path,
       tune_llm=False,
       tune_visual=False,
       tune_projector=True,  # Train action head projector
       tune_diffusion_model=True,  # Train action head DiT
       enable_advantage_conditioning=True,  # Enable indicator conditioning
   )
   ```

## Implementation Steps

1. ✅ Create `gr00t/model/indicator_embedding.py`
2. ✅ Modify `GR00T_N1_5_Config` to add advantage conditioning flags
3. ✅ Modify `GR00T_N1_5.__init__` to initialize indicator embedding
4. ✅ Modify `GR00T_N1_5.forward` to prepend indicator tokens
5. ✅ Update dataset to load indicators from reward_labels.json
6. ✅ Create training script `gr00t_advantage_conditioned_train.py`

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
    --no-tune-backbone
```

## Expected Behavior

During training:
- Model receives indicator (0 or 1) for each timestep
- Indicator is embedded to match VLM token dimension
- Indicator token is prepended as the FIRST token in the sequence
- Action head attends to [indicator_token, vlm_tokens...]
- Policy learns to generate better actions when indicator=1 (good advantage)
- Policy learns to avoid actions when indicator=0 (bad advantage)

This creates a conditional policy that can be guided by the advantage signal.
