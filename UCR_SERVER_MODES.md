# UCR Server Modes - Complete Guide

## Summary

The UCR server now supports **4 different modes** to ensure compatibility with both old and new code:

| Mode | Protocol | Serialization | Compatible With | Default? |
|------|----------|---------------|-----------------|----------|
| **HTTP Legacy** | HTTP (FastAPI) | JSON/json-numpy | Old UCR branch flat format clients | ✅ YES |
| **HTTP Current** | HTTP (FastAPI) | JSON/json-numpy | Current branch nested format clients | No |
| **ZMQ Msgpack** | ZMQ | msgpack | Current branch clients | No |
| **ZMQ Pickle** | ZMQ | pickle | Old UCR branch clients | No |

---

## Mode 1: HTTP Legacy (DEFAULT)

**Use this for OLD UCR branch compatibility with flat observations.**

### Launch Server
```bash
./launch_ucr_server.sh
# or
./launch_ucr_server.sh legacy
```

### Observation Format
```python
{
    "video.ego_view": np.ndarray[...],
    "state.waist_joint": np.ndarray[...],
    "state.right_arm_joint": np.ndarray[...],
    ...
    "task": ["instruction"]
}
```

### Test Client
```bash
python test_ucr_client.py
```

### When to Use
- Working with old UCR branch code
- Need HTTP REST API
- Want flat observation format (video.ego_view, state.waist_joint, etc.)

---

## Mode 2: HTTP Current

**Use this for current branch with HTTP and nested observations.**

### Launch Server
```bash
./launch_ucr_server.sh http
```

### Observation Format
```python
{
    "video": {
        "ego_view": np.ndarray[...]
    },
    "state": {
        "waist_joint": np.ndarray[...],
        "right_arm_joint": np.ndarray[...]
    },
    "language": {
        "task": [["instruction"]]
    }
}
```

### Test Client
```bash
python test_ucr_client_http.py
```

### When to Use
- Working with current branch code
- Need HTTP REST API
- Want nested observation format

---

## Mode 3: ZMQ Msgpack (Current Branch)

**Use this for current branch with ZMQ and msgpack serialization.**

### Launch Server
```bash
./launch_ucr_server.sh zmq
# or
./launch_ucr_server_zmq.sh
```

### Observation Format
```python
# Same as HTTP Current - nested format
{
    "video": {"ego_view": np.ndarray[...]},
    "state": {"waist_joint": np.ndarray[...], ...},
    "language": {"task": [["instruction"]]}
}
```

### Test Client
```bash
python test_ucr_client_zmq.py
```

### When to Use
- Working with current branch code
- Need faster serialization than HTTP
- Want nested observation format
- **This is the NEW current branch ZMQ implementation**

---

## Mode 4: ZMQ Pickle (Old UCR Branch)

**Use this for OLD UCR branch compatibility with ZMQ.**

### Launch Server
```bash
./launch_ucr_server.sh zmq-pickle
# or
./launch_ucr_server_zmq_pickle.sh
```

### Observation Format
```python
# Same as HTTP Current - nested format
{
    "video": {"ego_view": np.ndarray[...]},
    "state": {"waist_joint": np.ndarray[...], ...},
    "language": {"task": [["instruction"]]}
}
```

### Test Client
```bash
python test_ucr_client_zmq_pickle.py
```

### When to Use
- Working with old UCR branch code
- Need ZMQ protocol (not HTTP)
- Old UCR branch client expects pickle serialization
- **This matches the OLD UCR branch ZMQ implementation**

---

## Quick Decision Guide

### I have old UCR branch client code that uses:

**HTTP requests?** → Use **Mode 1: HTTP Legacy** (default)
```bash
./launch_ucr_server.sh
```

**ZMQ with pickle?** → Use **Mode 4: ZMQ Pickle**
```bash
./launch_ucr_server.sh zmq-pickle
```

### I'm writing new code and want:

**HTTP REST API?** → Use **Mode 2: HTTP Current**
```bash
./launch_ucr_server.sh http
```

**Faster ZMQ?** → Use **Mode 3: ZMQ Msgpack**
```bash
./launch_ucr_server.sh zmq
```

---

## Compatibility Matrix

| Server Mode | Client | Compatible? |
|-------------|--------|-------------|
| HTTP Legacy | test_ucr_client.py | ✅ Yes |
| HTTP Legacy | test_ucr_client_legacy.py | ✅ Yes |
| HTTP Legacy | Old UCR branch HTTP client | ✅ Yes |
| HTTP Current | test_ucr_client_http.py | ✅ Yes |
| ZMQ Msgpack | test_ucr_client_zmq.py | ✅ Yes |
| ZMQ Msgpack | PolicyClient (current branch) | ✅ Yes |
| ZMQ Pickle | test_ucr_client_zmq_pickle.py | ✅ Yes |
| ZMQ Pickle | RobotInferenceClient (old UCR) | ✅ Yes |
| ZMQ Pickle | Old UCR branch ZMQ client | ✅ Yes |
| **Cross-mode** | | ❌ No |

**Important:** You cannot mix server and client from different ZMQ modes:
- ZMQ Msgpack server ❌ ZMQ Pickle client
- ZMQ Pickle server ❌ ZMQ Msgpack client

---

## Implementation Files

### Server Files
- `gr00t/eval/run_gr00t_server.py` - Main server launcher (all modes)
- `gr00t/eval/http_server.py` - HTTP server implementation
- `gr00t/policy/server_client.py` - ZMQ msgpack implementation (current)
- `gr00t/eval/pickle_service.py` - ZMQ pickle implementation (old UCR)
- `gr00t/eval/robot.py` - ZMQ pickle wrapper (old UCR)

### Client Files
- `test_ucr_client.py` - HTTP Legacy client (default)
- `test_ucr_client_legacy.py` - HTTP Legacy client (explicit)
- `test_ucr_client_http.py` - HTTP Current client
- `test_ucr_client_zmq.py` - ZMQ Msgpack client
- `test_ucr_client_zmq_pickle.py` - ZMQ Pickle client (old UCR)

### Launcher Scripts
- `launch_ucr_server.sh` - Main launcher (choose mode with argument)
- `launch_ucr_server_legacy.sh` - HTTP Legacy (flat format)
- `launch_ucr_server_http.sh` - HTTP Current (nested format)
- `launch_ucr_server_zmq.sh` - ZMQ Msgpack (current branch)
- `launch_ucr_server_zmq_pickle.sh` - ZMQ Pickle (old UCR branch)

---

## Technical Details

### Why Multiple Modes?

1. **HTTP Legacy** - Old UCR branch used flat observation format with HTTP
2. **HTTP Current** - New code uses nested observation format with HTTP
3. **ZMQ Msgpack** - New current branch switched to msgpack serialization for better numpy handling
4. **ZMQ Pickle** - Old UCR branch used pickle serialization over ZMQ

### Serialization Differences

**Msgpack (current):**
- Custom numpy serialization via `np.save()`
- More efficient for large arrays
- Type-safe (no arbitrary code execution)

**Pickle (old UCR):**
- Python's built-in pickle
- Simpler but can execute arbitrary code
- Was used in old UCR branch

### Format Differences

**Flat format (HTTP Legacy):**
```python
"video.ego_view", "state.waist_joint", "task"
```

**Nested format (all others):**
```python
{"video": {"ego_view": ...}, "state": {"waist_joint": ...}}
```

---

## Migration Guide

### From Old UCR Branch → Current

**If you were using HTTP:**
```bash
# Old UCR branch
./launch_server.sh  # or similar

# Current branch
./launch_ucr_server.sh legacy  # Same flat format!
```

**If you were using ZMQ:**
```bash
# Old UCR branch
# (used pickle serialization)

# Current branch - Use pickle mode
./launch_ucr_server.sh zmq-pickle  # Old UCR compatible!
```

### From Current → New Code

**Switch from msgpack ZMQ:**
```bash
# Already using msgpack? Good to go!
./launch_ucr_server.sh zmq
```

**Switch from HTTP:**
```bash
# Use nested format
./launch_ucr_server.sh http
```

---

## Troubleshooting

### "unpack(b) received extra data" error

**Problem:** Client and server are using different ZMQ serializations

**Solution:** Match serializations:
- If server uses msgpack → use msgpack client
- If server uses pickle → use pickle client

### "404 Not Found" or wrong endpoint

**Problem:** Using HTTP client with ZMQ server (or vice versa)

**Solution:** Match protocols:
- HTTP server → HTTP client
- ZMQ server → ZMQ client

### Wrong observation format

**Problem:** Server expects nested but client sends flat (or vice versa)

**Solution:**
- HTTP Legacy expects flat format
- All others expect nested format

Use `--use-sim-policy-wrapper` for flat→nested conversion.
