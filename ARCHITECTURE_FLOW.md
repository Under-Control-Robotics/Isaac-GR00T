# Advantage-Conditioned Policy Architecture Flow

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                │
├─────────────────────────────────────────────────────────────────────┤
│ - Video frames: (B, T, C, H, W)                                     │
│ - State: (B, state_dim)                                              │
│ - Indicator: (B,) or (B, action_horizon)  ← NEW for advantage cond. │
│ - Action (for training): (B, action_horizon, action_dim)             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ BACKBONE (Eagle VLM)                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Vision encoder + LLM → VLM features                                  │
│ Output: backbone_features (B, seq_len, 4096)                         │
│         seq_len ≈ 128 (typical for 2 history frames)                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ INDICATOR EMBEDDING (if advantage conditioning enabled)              │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Take first timestep: indicator[:, 0] → (B,)                       │
│ 2. Embed: nn.Embedding(2, 4096) → (B, 4096)                          │
│ 3. Add position embedding: + learned_pos_emb → (B, 1, 4096)          │
│ 4. Prepend to backbone:                                               │
│    backbone_features = cat([indicator_tokens, backbone_features])     │
│    Output: (B, seq_len+1, 4096)  ← seq_len increases by 1            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ACTION HEAD: vlln (LayerNorm)                                        │
├─────────────────────────────────────────────────────────────────────┤
│ LayerNorm over features                                               │
│ Input:  (B, seq_len+1, 4096)                                          │
│ Output: (B, seq_len+1, 4096)                                          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ACTION HEAD: vl_self_attention                                        │
├─────────────────────────────────────────────────────────────────────┤
│ Self-attention with sinusoidal position embeddings                    │
│                                                                        │
│ For each transformer block:                                           │
│   1. Add sinusoidal pos emb based on position:                        │
│      - Position 0: indicator token                                    │
│      - Position 1, 2, ..., seq_len: VLM tokens                        │
│   2. Self-attention: all tokens attend to all tokens                  │
│   3. Feed-forward network                                             │
│                                                                        │
│ Input:  (B, seq_len+1, 4096)                                          │
│ Output: (B, seq_len+1, 4096)                                          │
│                                                                        │
│ Config: max_num_positional_embeddings = 512                           │
│         seq_len+1 ≈ 129 << 512  ✓                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ ACTION HEAD: DiT (Diffusion Transformer)                             │
├─────────────────────────────────────────────────────────────────────┤
│ Cross-attention between noisy actions and conditioned VLM features    │
│                                                                        │
│ Query: noisy actions (B, action_horizon, hidden_dim)                  │
│ Key/Value: VLM features (B, seq_len+1, 4096)                          │
│             ↑                                                          │
│             └── Includes indicator at position 0!                     │
│                                                                        │
│ The action head attends to:                                           │
│   [indicator_token, vlm_token_1, vlm_token_2, ..., vlm_token_seq_len]│
│    ↑                                                                   │
│    └── Policy learns from this conditioning signal                    │
│                                                                        │
│ Output: predicted actions (B, action_horizon, action_dim)             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                                │
├─────────────────────────────────────────────────────────────────────┤
│ - Actions: (B, action_horizon, action_dim)                            │
│ - Loss: flow matching loss between predicted and target actions       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Points

### 1. Indicator Position
```
Position:  [    0    ] [  1  ] [  2  ] ... [ seq_len ]
Token:     [indicator] [VLM_1] [VLM_2] ... [VLM_128 ]
           └─ Special ─┘ └──────── VLM features ──────┘
```

### 2. Position Embedding Layers

**Layer 1: Indicator's Learned Position Embedding**
- Applied: In `IndicatorEmbedding.__init__` (line 36)
- Type: Learned parameter
- Purpose: Mark "I am an indicator token" (token type)
- Shape: (1, 1, 4096)
- Added before prepending to VLM tokens

**Layer 2: Sinusoidal Position Embedding**
- Applied: In `BasicTransformerBlock.forward` (line 165-166)
- Type: Sinusoidal (dynamically computed)
- Purpose: Encode "I am at position i in the sequence"
- Max positions: 512
- Applied to entire sequence including indicator

### 3. Attention Pattern

```
Self-Attention in vl_self_attention:
┌──────────┬──────┬──────┬─────┬────────┐
│ From\To  │ IND  │ VLM1 │ ... │ VLMn   │
├──────────┼──────┼──────┼─────┼────────┤
│ IND      │  ✓   │  ✓   │  ✓  │   ✓    │ ← Indicator attends to all
├──────────┼──────┼──────┼─────┼────────┤
│ VLM1     │  ✓   │  ✓   │  ✓  │   ✓    │ ← VLM tokens attend to indicator
├──────────┼──────┼──────┼─────┼────────┤
│ ...      │  ✓   │  ✓   │  ✓  │   ✓    │
├──────────┼──────┼──────┼─────┼────────┤
│ VLMn     │  ✓   │  ✓   │  ✓  │   ✓    │
└──────────┴──────┴──────┴─────┴────────┘

Cross-Attention in DiT:
┌──────────┬──────┬──────┬─────┬────────┐
│ From\To  │ IND  │ VLM1 │ ... │ VLMn   │
├──────────┼──────┼──────┼─────┼────────┤
│ Action0  │  ✓   │  ✓   │  ✓  │   ✓    │ ← Actions attend to [IND + VLM]
├──────────┼──────┼──────┼─────┼────────┤
│ Action1  │  ✓   │  ✓   │  ✓  │   ✓    │
├──────────┼──────┼──────┼─────┼────────┤
│ ...      │  ✓   │  ✓   │  ✓  │   ✓    │
└──────────┴──────┴──────┴─────┴────────┘
                 ↑
                 └── Policy learns from indicator signal
```

## Training Behavior

### When indicator = 1 (Good advantage)
- Model learns: "This state-action pair is good"
- Policy: Generate similar actions in similar states

### When indicator = 0 (Bad advantage)
- Model learns: "This state-action pair is bad"
- Policy: Avoid similar actions in similar states

### How It Works
1. **Self-attention**: VLM tokens exchange information with indicator
2. **Cross-attention**: Action generation is conditioned on [indicator + VLM]
3. **Gradient flow**: Loss backprops through indicator embedding
4. **Learning**: Indicator embedding learns to modulate action generation

## Verification Points

During training, you'll see these logs:

```
[Dataset Loading]
Loaded advantage labels for 120 episodes
  Global advantage threshold: -0.004991
  Verifying indicator-advantage alignment...
  ✓ Indicator alignment accuracy: 100.0%

[Training Script - Batch Check]
VERIFYING INDICATOR DATA IN FIRST BATCH
✓ Indicator found in batch
  Shape: torch.Size([32, 16])
  Unique values: [0.0, 1.0]
  Distribution: 22/512 (68.8%) are 1 (good)

[Model - First Forward Pass]
[MODEL] Advantage conditioning verification:
  Input indicators: [1.0, 0.0, 1.0, ...]
  Indicator tokens shape: torch.Size([32, 1, 4096])
  Backbone features shape (before): torch.Size([32, 128, 4096])
  Backbone features shape (after): torch.Size([32, 129, 4096])
  ✓ Indicator token successfully prepended as FIRST token
```

## Comparison: RL Training vs Advantage-Conditioned Training

| Aspect | RL Training | Advantage-Conditioned |
|--------|-------------|----------------------|
| What's trained | Value head | Action head + indicator embedding |
| Input | Video + state | Video + state + indicator |
| Backbone output | (B, seq_len, 4096) | (B, seq_len+1, 4096) |
| Conditioning | None | Indicator at position 0 |
| Goal | Predict state values | Generate advantage-conditioned actions |
| Use case | Compute advantages | Deploy conditional policy |

## Files Modified for Advantage Conditioning

1. `gr00t/model/indicator_embedding.py` - NEW: Indicator embedding module
2. `gr00t/model/gr00t_n1.py` - Added indicator prepending logic
3. `gr00t/data/dataset.py` - Load indicators from reward_labels.json
4. `scripts/gr00t_advantage_conditioned_train.py` - NEW: Training script

**No changes needed to:**
- `vlln` (LayerNorm) - handles variable sequence length
- `vl_self_attention` (SelfAttentionTransformer) - sinusoidal pos emb is dynamic
- DiT cross-attention - already handles variable key/value lengths
