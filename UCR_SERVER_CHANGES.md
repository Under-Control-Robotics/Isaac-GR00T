# UCR Server Changes - UCR Branch Compatibility

## Summary

Updated the HTTP server and client scripts to be **fully compatible with the old UCR branch** while **making json-numpy optional**.

## Key Changes

### 1. HTTP Server (`gr00t/eval/http_server.py`)

✅ **json-numpy is now optional** - no longer required!

The server now:
- Tries to import `json_numpy` for automatic numpy serialization
- Falls back to manual `tolist()` conversion if not available
- Handles both old API (`get_action()` returns action) and new API (`get_action()` returns `(action, info)`)
- Works exactly like the UCR branch server

**Dependencies:**
```bash
# Required
pip install uvicorn fastapi

# Optional (for faster serialization)
pip install json-numpy
```

### 2. Client Scripts

All HTTP client scripts updated with the same fallback:
- `test_ucr_client.py` - Default UCR legacy client
- `test_ucr_client_legacy.py` - Explicit legacy client
- `test_ucr_client_http.py` - HTTP client for nested format

**Dependencies:**
```bash
# Required
pip install requests

# Optional (for faster serialization)
pip install json-numpy
```

### 3. Server Behavior

| Feature | With json-numpy | Without json-numpy |
|---------|----------------|-------------------|
| **Numpy serialization** | Automatic via json_numpy.patch() | Manual via tolist() |
| **Performance** | Faster | Slightly slower |
| **Compatibility** | Same | Same |
| **Works?** | ✅ Yes | ✅ Yes |

## Usage

### Default Setup (UCR Branch Legacy Style)

**Start server:**
```bash
./launch_ucr_server.sh
# or
./launch_ucr_server.sh legacy
```

**Test client:**
```bash
python test_ucr_client.py
```

**No json-numpy needed!** The server and client will automatically use fallback serialization.

### With json-numpy (Optional Performance Boost)

If you want faster numpy serialization:

```bash
pip install json-numpy
```

Then run the same commands. The server/client will automatically detect and use json-numpy.

## Technical Details

### Manual Serialization Function

When json-numpy is not available, the server/client use this fallback:

```python
def _numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
```

This recursively converts all numpy arrays to Python lists, which are JSON-serializable.

### Old vs New API Compatibility

The server handles both policy APIs:

```python
# Old UCR branch API
action = policy.get_action(obs)  # Returns just action

# New current branch API
action, info = policy.get_action(obs)  # Returns (action, info) tuple
```

Detection is automatic:
```python
result = self.policy.get_action(obs)
if isinstance(result, tuple):
    action, _ = result  # New API
else:
    action = result  # Old API
```

## Verification

To verify the server works without json-numpy:

1. Make sure json-numpy is NOT installed:
   ```bash
   pip uninstall json-numpy
   ```

2. Start the server:
   ```bash
   ./launch_ucr_server.sh
   ```

   You should see:
   ```
   Warning: json-numpy not installed. Using manual numpy serialization.
   Install with: pip install json-numpy
   ```

3. Test the client:
   ```bash
   python test_ucr_client.py
   ```

   Should work perfectly with manual serialization!

## Files Modified

- ✅ `gr00t/eval/http_server.py` - Made json-numpy optional
- ✅ `test_ucr_client.py` - Made json-numpy optional
- ✅ `test_ucr_client_http.py` - Made json-numpy optional
- ✅ `test_ucr_client_legacy.py` - Made json-numpy optional
- ✅ `launch_ucr_server_legacy.sh` - Updated dependency docs
- ✅ `launch_ucr_server_http.sh` - Updated dependency docs

## Backward Compatibility

✅ **100% compatible with old UCR branch**
- Same `/act` endpoint
- Same flat observation format
- Same flat action format
- Same behavior with or without json-numpy

✅ **Works with new current branch**
- Handles new `(action, info)` return format
- Uses new `BasePolicy` class
- Compatible with both ZMQ and HTTP modes
