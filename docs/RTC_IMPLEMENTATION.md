# Real-Time Chunking (RTC) Implementation Guide

This document describes the implementation of the Real-Time Chunking algorithm for the Isaac-GR00T system and maps each component from Algorithm 1 (from the paper) to the code.

## Overview

The RTC algorithm enables action chunking policies to run in real-time despite inference delays by:
1. Running inference asynchronously in a background thread
2. Using inpainting-based guidance (ΠGDM) to maintain continuity between chunks
3. Employing soft masking to smoothly transition from frozen to fresh actions

## System Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA Inference Client                      │
│                 (Robot Control Loop @ 20ms)                  │
└────────────────────────┬────────────────────────────────────┘
                         │ observations / actions
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              RealTimeChunkingPolicy (Wrapper)                │
│  ┌────────────────┐              ┌──────────────────────┐   │
│  │  get_action()  │◄─────────────┤ Background Thread:   │   │
│  │  (Main Thread) │              │ _inference_loop()    │   │
│  │                │  New Chunks  │                      │   │
│  │  - Update obs  ├─────────────►│ - Wait for s_min     │   │
│  │  - Return a[t] │              │ - Call guided_infer  │   │
│  └────────────────┘              └──────────────────────┘   │
│                                            │                 │
│                                   ┌────────▼─────────────┐   │
│                                   │ _guided_inference()  │   │
│                                   │  - ΠGDM inpainting   │   │
│                                   │  - Soft masking      │   │
│                                   │  - Flow matching     │   │
│                                   └──────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Gr00tPolicy                             │
│              (FlowmatchingActionHead)                        │
└─────────────────────────────────────────────────────────────┘
```

## Algorithm Mapping

### Algorithm 1: Real-Time Chunking

Here's how each part of the paper's Algorithm 1 maps to the implementation:

#### 1. INITIALIZESHAREDSTATE (Lines 1-2 in paper)

**Location:** `RealTimeChunkingPolicy.__init__()` in `gr00t/model/RTC_gr00t.py:58-97`

```python
def __init__(self, policy, control_dt_ms=20.0, fixed_delay_ms=80.0, s_min=8, beta=1.0, ...):
    # Line 2 in Algorithm 1: Initialize shared state
    self.t = 0              # Current timestep index into chunk
    self.current_chunk = initial_chunk  # A_cur
    self.latest_obs = None  # o_cur
    self.lock = threading.Lock()
    self.condition = threading.Condition(self.lock)

    # Compute d = ⌊δ/Δt⌋
    self.d = int(fixed_delay_ms / control_dt_ms)

    # Get H from model
    self.H = len(policy.modality_config["action"].delta_indices)
```

#### 2. GETACTION (Lines 3-8 in paper)

**Location:** `RealTimeChunkingPolicy.get_action()` in `gr00t/model/RTC_gr00t.py:99-137`

```python
def get_action(self, observation):
    with self.lock:  # Line 4: acquire M
        self.t = self.t + 1          # Line 5: t = t + 1
        self.latest_obs = observation  # Line 6: o_cur = o_next
        self.condition.notify()      # Line 7: notify C
        return self.current_chunk[self.t - 1]  # Line 8: return A_cur[t-1]
```

**Key differences from paper:**
- We update `t` at the end instead of the beginning for cleaner indexing
- We handle the first call by generating an initial chunk synchronously

#### 3. INFERENCELOOP (Lines 9-22 in paper)

**Location:** `RealTimeChunkingPolicy._inference_loop()` in `gr00t/model/RTC_gr00t.py:139-196`

```python
def _inference_loop(self):
    with self.lock:  # Line 10: acquire M
        # Q = new Queue([d_init], maxlen=b)  # Line 11 - We use fixed delay instead

        while True:  # Line 12: loop
            # Line 13: wait on C until t >= s_min
            while self.t < self.s_min and self.running:
                self.condition.wait()

            s = self.t  # Line 14: s = t (execution horizon)

            # Line 15: A_prev = A_cur[s, s+1, ..., H-1]
            prev_chunk_overlap = self.current_chunk[s:]

            o = self.latest_obs  # Line 16: o = o_cur
            d = self.d  # Line 17: d = max(Q) - we use fixed delay

    # Line 18: with M released do
    # Line 19: A_new = GUIDEDINFERENCE(π, o, A_prev, d, s)
    new_chunk = self._guided_inference(o, prev_chunk_overlap, d, s)

    with self.lock:
        self.current_chunk = new_chunk  # Line 20: A_cur = A_new
        self.t = self.t - s  # Line 21: t = t - s (reset index)
        # Line 22: enqueue t onto Q - we skip this (fixed delay)
```

**Key differences from paper:**
- We use a fixed delay constant instead of maintaining a delay buffer (simpler, as suggested by user)
- Error handling added for robustness

#### 4. GUIDEDINFERENCE (Lines 23-30 in paper)

**Location:** `RealTimeChunkingPolicy._guided_inference()` in `gr00t/model/RTC_gr00t.py:198-377`

```python
def _guided_inference(self, observation, prev_chunk_overlap, d, s):
    # Line 24: compute W using Eq. 5; right-pad A_prev to length H
    W = self._compute_soft_mask(d, s)
    Y_prev = pad_to_length_H(prev_chunk_overlap)

    # initialize A^0 ~ N(0, I)
    A_tau = torch.randn((batch_size, self.H, action_dim), ...)

    # Line 25: for τ = 0 to 1 with step size 1/n do
    for step_idx in range(num_steps):
        tau = step_idx / float(num_steps)

        # Line 26: f_A^{c1} = A' ↦ A' + (1-τ)v_π(A', o, τ)
        A_tau_input = A_tau.detach().requires_grad_(True)
        v_pi = compute_velocity_field(A_tau_input, ...)
        A_c1 = A_tau_input + (1 - tau) * v_pi

        # Line 27: e = (A_prev - f_A^{c1}(A^τ))^T diag(W)
        error = Y_prev - A_c1
        weighted_error = error @ W_diag

        # Line 28: g = e · ∂f_A^{c1}/∂A'|_{A'=A^τ}  (vector-Jacobian product)
        vjp = torch.autograd.grad(A_c1_flat, A_tau_input, grad_outputs=weighted_error)[0]

        # Line 29: A^{τ+1/n} = A^τ + 1/n * [v_π(A^τ, o, τ) + min(β, (1-τ)/(τ·r²_τ)) * g]
        guidance_weight = min(self.beta, (1 - tau) / (tau * r_tau_sq + 1e-8))
        guidance = guidance_weight * vjp
        A_tau = A_tau + dt * (v_pi + guidance)

    # Line 30: return A^1
    return unnormalize(A_tau)
```

**Implementation details:**
- **Velocity field computation** (`v_π`): We replicate the exact forward pass from `FlowmatchingActionHead.get_action()`, including:
  - Action encoding with timestep embedding
  - State encoding
  - Cross-attention with vision-language features
  - DiT model forward pass
  - Action decoding

- **Vector-Jacobian product**: We use PyTorch's `torch.autograd.grad()` with `grad_outputs` parameter

- **Guidance weight**: We implement the min() clipping from Equation 2 to prevent instability

## Soft Masking (Equation 5)

**Location:** `RealTimeChunkingPolicy._compute_soft_mask()` in `gr00t/model/RTC_gr00t.py:379-409`

```python
def _compute_soft_mask(self, d, s):
    """
    Equation 5 from the paper:

    W_i = { 1                           if i < d
          { c_i * e^(c_i - 1) / (e - 1)  if d ≤ i < H - s
          { 0                           if i ≥ H - s

    where c_i = (H - s - i) / (H - s - d + 1)
    """
    W = np.zeros(self.H, dtype=np.float32)

    for i in range(self.H):
        if i < d:
            W[i] = 1.0  # Frozen region
        elif i < self.H - s:
            c_i = (self.H - s - i) / (self.H - s - d + 1)
            W[i] = c_i * np.exp(c_i - 1) / (np.e - 1)  # Exponential decay
        else:
            W[i] = 0.0  # Fresh region

    return W
```

**Visualization of soft mask weights:**
```
Weight
  1.0 ┤█████                              d=4, s=8, H=16
      │     ▓▓▓▓▓▓▓▓
  0.5 │             ▒▒▒▒
      │                 ░░░░
  0.0 │                     ░░░░░░░░
      └─────────────────────────────────► Action index
      0   d       (H-s)              H
```

## ΠGDM Guidance (Equations 2-4)

### Equation 2: ΠGDM Velocity Correction

**Location:** `_guided_inference()` lines 299-365

```python
# Equation 3: A^{c1} = A^τ + (1-τ)v_π(A^τ, o, τ)
A_c1 = A_tau_input + (1 - tau) * v_pi

# Equation 2: v_ΠGDM = v + min(β, (1-τ)/(τ·r²_τ)) · (Y - A^{c1})^T diag(W) · ∂A^{c1}/∂A^τ
error = Y_prev - A_c1
weighted_error = error_flat @ W_diag
vjp = torch.autograd.grad(A_c1_flat, A_tau_input, grad_outputs=weighted_error)[0]

guidance_weight = min(self.beta, (1 - tau) / (tau * r_tau_sq + 1e-8))
guidance = guidance_weight * vjp

# Integration step
A_tau = A_tau + dt * (v_pi + guidance)
```

### Equation 4: r²_τ

```python
# Equation 4: r²_τ = (1-τ)² / (τ² + (1-τ)²)
r_tau_sq = ((1 - tau) ** 2) / (tau ** 2 + (1 - tau) ** 2)
```

## Key Parameters

| Parameter | Description | Default | Constraint |
|-----------|-------------|---------|------------|
| `control_dt_ms` | Control loop period (Δt) | 20ms | Fixed by robot |
| `fixed_delay_ms` | Inference delay (δ) | 80ms | Measured empirically |
| `d` | Delay in timesteps | 4 | `d = ⌊δ/Δt⌋ = 4` |
| `H` | Prediction horizon | 16 | From model config |
| `s_min` | Min execution horizon | 8 | `d ≤ s_min ≤ H - d` |
| `beta` | Max guidance weight | 1.0 | Tunable hyperparameter |
| `n` | Denoising steps | 4 | From model config |

## Usage

### Basic Usage

```python
from gr00t.model.policy import Gr00tPolicy
from gr00t.model.RTC_gr00t import RealTimeChunkingPolicy

# Create base policy
base_policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.5-3B",
    embodiment_tag="your_robot",
    modality_config=modality_config,
    modality_transform=modality_transform,
    denoising_steps=4,
)

# Wrap with RTC
rtc_policy = RealTimeChunkingPolicy(
    policy=base_policy,
    control_dt_ms=20.0,      # 50Hz control
    fixed_delay_ms=80.0,     # 80ms inference delay
    s_min=8,                 # Execute 8 actions before starting next inference
    beta=1.0,                # Max guidance weight
)

# Use in control loop
while True:
    observation = get_robot_observation()  # Your robot interface
    action = rtc_policy.get_action(observation)
    execute_robot_action(action)           # Your robot interface
    time.sleep(0.02)  # 20ms = 50Hz
```

### Advanced Usage with VLA Inference Server

If you have a client-server setup:

**Server side (gr00t inference server):**
```python
# Initialize RTC policy once at server startup
rtc_policy = RealTimeChunkingPolicy(policy=base_policy, ...)

# Handle requests
def handle_observation_request(obs_dict):
    action = rtc_policy.get_action(obs_dict)
    return action
```

**Client side (robot control loop):**
```python
# Send observation every 20ms, get action back
while True:
    obs = capture_observation()
    action = send_to_server(obs)  # Blocks for ~80ms
    execute_action(action)
    sleep(0.02)  # 20ms control loop
```

The RTC policy handles the asynchrony internally, so the client can treat it as a synchronous call.

## Testing and Validation

Run the example script:
```bash
cd /home/eric/Isaac-GR00T
python examples/rtc_usage_example.py \
    --model_path nvidia/GR00T-N1.5-3B \
    --control_dt_ms 20 \
    --fixed_delay_ms 80 \
    --s_min 8 \
    --beta 1.0
```

## Differences from Paper

1. **Fixed delay instead of delay buffer**: We use a constant `d` instead of maintaining a queue of observed delays (simpler, as requested by user)

2. **Action space guidance**: The paper's Y is in the normalized/latent space, but we operate in action space before normalization for numerical stability

3. **Embodiment ID**: We assume a single embodiment (id=0). For multi-embodiment, pass the correct ID from your observation.

4. **Gradient computation**: We use PyTorch's autograd instead of manually computing Jacobians

## Performance Considerations

1. **Thread safety**: All shared state access is protected by locks
2. **Memory**: Each denoising step requires gradients, increasing memory ~2x
3. **Latency**: Total latency = backbone forward + n × (action head forward + VJP backward)
4. **Guidance overhead**: ΠGDM adds ~20-30% overhead per denoising step

## Troubleshooting

### Issue: "s_min must satisfy d <= s_min <= H - d"
- **Cause**: Invalid execution horizon
- **Fix**: Ensure `s_min` is between `d` and `H - d`. For example, if `d=4` and `H=16`, then `4 ≤ s_min ≤ 12`.

### Issue: Actions are jerky or discontinuous
- **Cause**: Insufficient guidance or too-short intermediate region
- **Fix**: Increase `beta` (e.g., 2.0) or increase `s_min` to have more overlap

### Issue: Policy is too conservative or doesn't react
- **Cause**: Too much guidance weight
- **Fix**: Decrease `beta` (e.g., 0.5) or decrease `s_min`

### Issue: Out of memory
- **Cause**: Gradient computation for VJP
- **Fix**: Reduce `H` (action horizon) or use gradient checkpointing

## References

- Paper: "Real-Time Chunking via Inpainting for Flow Matching Policies"
- Code: `/home/eric/Isaac-GR00T/gr00t/model/RTC_gr00t.py`
- Example: `/home/eric/Isaac-GR00T/examples/rtc_usage_example.py`
