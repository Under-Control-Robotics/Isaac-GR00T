# UCR GR00T Inference Server Guide

This guide explains how to launch the UCR finetuned GR00T model server with support for both **ZMQ** (current) and **HTTP** (legacy from ucr branch) modes.

## Quick Start

### Option 1: Default (ZMQ mode)
```bash
./launch_ucr_server.sh
```

### Option 2: Choose mode explicitly
```bash
./launch_ucr_server.sh zmq   # ZMQ mode (current default)
./launch_ucr_server.sh http  # HTTP mode (legacy from ucr branch)
```

### Option 3: Dedicated launchers
```bash
./launch_ucr_server_zmq.sh   # ZMQ mode
./launch_ucr_server_http.sh  # HTTP mode
```

---

## Server Modes Comparison

| Feature | ZMQ (Current) | HTTP (Legacy) |
|---------|---------------|---------------|
| **Protocol** | ZeroMQ message queue | REST API (FastAPI) |
| **From branch** | ucr_16 (current) | ucr (old) |
| **Port** | 5555 | 5555 |
| **Client library** | `PolicyClient` from `gr00t.policy.server_client` | HTTP requests (curl, requests, etc.) |
| **Dependencies** | Built-in (zmq, msgpack) | `pip install uvicorn fastapi json-numpy` |
| **Endpoint** | ZMQ socket | `POST /act` |
| **Serialization** | msgpack with custom numpy encoding | JSON with json-numpy |

---

## Model Configuration

- **Checkpoint**: `/data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000`
- **Embodiment**: `NEW_EMBODIMENT` (ucr_wblm_moby_history)
- **Action horizon**: 16 steps
- **Host**: `0.0.0.0` (accepts connections from any machine on local network)
- **Port**: `5555`

---

## Input Format

All modes expect the same observation format:

```python
{
    "video": {
        "ego_view": np.ndarray[np.uint8, (B, 2, H, W, 3)]  # 2 frames at t-30, t
    },
    "state": {
        "waist_joint": np.ndarray[np.float32, (B, 1, D)],
        "right_arm_joint": np.ndarray[np.float32, (B, 1, D)],
        "left_arm_joint": np.ndarray[np.float32, (B, 1, D)],
        "right_leg_joint": np.ndarray[np.float32, (B, 1, D)],
        "left_leg_joint": np.ndarray[np.float32, (B, 1, D)],
        "orientation_joint": np.ndarray[np.float32, (B, 1, D)],
    },
    "language": {
        "task": [["your task instruction"]]  # (B, 1) list[list[str]]
    }
}
```

---

## Output Format

All modes return the same action format:

```python
{
    "behavior_mode": np.ndarray[np.float32, (B, 16, D)],
    "left_ee_position": np.ndarray[np.float32, (B, 16, 3)],
    "right_ee_position": np.ndarray[np.float32, (B, 16, 3)],
    "left_ee_orientation": np.ndarray[np.float32, (B, 16, D)],
    "right_ee_orientation": np.ndarray[np.float32, (B, 16, D)],
    "base_height": np.ndarray[np.float32, (B, 16, D)],
    "base_orientation": np.ndarray[np.float32, (B, 16, D)],
    "base_vel": np.ndarray[np.float32, (B, 16, D)],
}
```

**Action horizon**: 16 steps into the future

---

## Client Examples

### ZMQ Client (Current)

```bash
python test_ucr_client_zmq.py
```

**Code example:**
```python
from gr00t.policy.server_client import PolicyClient

client = PolicyClient(host="127.0.0.1", port=5555)

# Get modality config
modality_configs = client.get_modality_config()

# Get action prediction
actions, info = client.get_action(observation)
```

### HTTP Client (Legacy)

```bash
python test_ucr_client_http.py
```

**Code example:**
```python
import requests
import json_numpy

json_numpy.patch()  # Enables numpy array serialization in JSON

response = requests.post(
    "http://127.0.0.1:5555/act",
    json={"observation": observation}
)
actions = response.json()
```

**curl example:**
```bash
curl -X POST http://127.0.0.1:5555/health
curl -X POST http://127.0.0.1:5555/act -H "Content-Type: application/json" -d '{"observation": {...}}'
```

---

## Installation Dependencies

### ZMQ mode (built-in)
No extra dependencies needed. Uses standard `zmq` and `msgpack` libraries.

### HTTP mode
**Required:**
```bash
pip install uvicorn fastapi
```

**Optional (for faster numpy serialization):**
```bash
pip install json-numpy
```

The server works without `json-numpy` by using fallback serialization via `tolist()`.

---

## Files Created

- `gr00t/eval/http_server.py` - HTTP server implementation (ported from ucr branch)
- `gr00t/eval/run_gr00t_server.py` - Modified to support both ZMQ and HTTP modes
- `launch_ucr_server.sh` - Main launcher with mode selection
- `launch_ucr_server_zmq.sh` - Dedicated ZMQ launcher
- `launch_ucr_server_http.sh` - Dedicated HTTP launcher
- `test_ucr_client_zmq.py` - ZMQ client test script
- `test_ucr_client_http.py` - HTTP client test script

---

## Advanced Usage

### Manual server launch

**ZMQ mode:**
```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./ucr_modality_config.py \
  --host 0.0.0.0 \
  --port 5555 \
  --server-type zmq
```

**HTTP mode:**
```bash
python gr00t/eval/run_gr00t_server.py \
  --model-path /data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000 \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path ./ucr_modality_config.py \
  --host 0.0.0.0 \
  --port 5555 \
  --server-type http
```

---

## Troubleshooting

### HTTP mode: ModuleNotFoundError for uvicorn/fastapi
```
ModuleNotFoundError: No module named 'uvicorn'
```
**Solution:** Install required HTTP dependencies:
```bash
pip install uvicorn fastapi
```

### json-numpy not found warning
```
Warning: json-numpy not installed. Using manual numpy serialization.
```
**This is OK!** The server works fine without json-numpy. It will use fallback serialization.

**To remove the warning** (optional):
```bash
pip install json-numpy
```

### ZMQ mode: Connection refused
**Solution:** Make sure the server is running and the host/port are correct. Check firewall settings if connecting from a different machine.

### Wrong action format
**Solution:** Verify your observation format matches the expected format above. Use the test client scripts to verify connectivity first.

### Server returns list instead of numpy arrays
**This is expected!** When json-numpy is not installed, the server returns actions as nested Python lists instead of numpy arrays. The client automatically converts them back to numpy arrays.
