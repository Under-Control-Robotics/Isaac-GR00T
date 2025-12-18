# Summary of Changes - Old UCR Branch Compatibility

## What Was Changed

### ✅ KEPT - All existing current branch code stays unchanged

- `gr00t/policy/server_client.py` - **Unchanged** (msgpack ZMQ - current branch)
- `gr00t/policy/gr00t_policy.py` - **Unchanged**
- `gr00t/eval/http_server.py` - **Modified** to make json-numpy optional
- `test_ucr_client_zmq.py` - **Unchanged** (works with msgpack ZMQ)
- All other current branch files - **Unchanged**

### ✅ ADDED - New files for old UCR branch compatibility

**New server implementations:**
- `gr00t/eval/pickle_service.py` - ZMQ with pickle serialization (old UCR branch)
- `gr00t/eval/robot.py` - Wrapper for pickle-based ZMQ server/client

**New launcher scripts:**
- `launch_ucr_server_zmq_pickle.sh` - Launch ZMQ pickle server

**New test clients:**
- `test_ucr_client_zmq_pickle.py` - Test pickle-based ZMQ server

**Documentation:**
- `UCR_SERVER_MODES.md` - Complete guide to all 4 server modes
- `UCR_SERVER_CHANGES.md` - HTTP server changes (json-numpy optional)
- `CHANGES_SUMMARY.md` - This file

### ✅ MODIFIED - Files updated to support both old and new

**`gr00t/eval/run_gr00t_server.py`:**
- Added `--zmq-serialization` flag
- Supports both msgpack (current) and pickle (old UCR)
- All existing functionality preserved

**`launch_ucr_server.sh`:**
- Added `zmq-pickle` mode option
- All existing modes still work

**Other modified files:**
- HTTP clients made json-numpy optional
- Launcher scripts updated with dependency info

---

## Server Modes Available

### 1. HTTP Legacy (Default) - OLD UCR compatible
```bash
./launch_ucr_server.sh
```
- Flat format: `video.ego_view`, `state.waist_joint`
- Works with old UCR HTTP clients

### 2. HTTP Current - Current branch
```bash
./launch_ucr_server.sh http
```
- Nested format: `{"video": {"ego_view": ...}}`
- Current branch HTTP

### 3. ZMQ Msgpack - Current branch (NEW)
```bash
./launch_ucr_server.sh zmq
```
- Msgpack serialization
- Current branch ZMQ
- **This is the NEW default ZMQ mode**

### 4. ZMQ Pickle - OLD UCR compatible (NEW)
```bash
./launch_ucr_server.sh zmq-pickle
```
- Pickle serialization
- Old UCR branch ZMQ
- **This matches the OLD UCR branch**

---

## Compatibility

### Old UCR Branch Clients

✅ **HTTP with flat format** → Mode 1: HTTP Legacy (default)
```bash
./launch_ucr_server.sh
```

✅ **ZMQ with pickle** → Mode 4: ZMQ Pickle
```bash
./launch_ucr_server.sh zmq-pickle
```

### Current Branch Clients

✅ **HTTP with nested format** → Mode 2: HTTP Current
```bash
./launch_ucr_server.sh http
```

✅ **ZMQ with msgpack** → Mode 3: ZMQ Msgpack
```bash
./launch_ucr_server.sh zmq
```

---

## What This Solves

### ❌ Before
- Current branch uses msgpack ZMQ
- Old UCR branch uses pickle ZMQ
- **Incompatible!** Error: "unpack(b) received extra data"

### ✅ After
- Both msgpack and pickle ZMQ available
- Choose mode with `--zmq-serialization` flag
- Old code works with `zmq-pickle` mode
- New code works with `zmq` mode (msgpack)

---

## File Structure

```
Isaac-GR00T/
├── gr00t/
│   ├── eval/
│   │   ├── run_gr00t_server.py         # Modified: added pickle option
│   │   ├── http_server.py              # Modified: json-numpy optional
│   │   ├── pickle_service.py           # NEW: pickle ZMQ (old UCR)
│   │   └── robot.py                    # NEW: pickle ZMQ wrapper (old UCR)
│   └── policy/
│       └── server_client.py            # Unchanged: msgpack ZMQ (current)
│
├── launch_ucr_server.sh                # Modified: added zmq-pickle mode
├── launch_ucr_server_legacy.sh         # HTTP Legacy launcher
├── launch_ucr_server_http.sh           # HTTP Current launcher
├── launch_ucr_server_zmq.sh            # ZMQ Msgpack launcher
├── launch_ucr_server_zmq_pickle.sh     # NEW: ZMQ Pickle launcher
│
├── test_ucr_client.py                  # HTTP Legacy client (default)
├── test_ucr_client_legacy.py           # HTTP Legacy client (explicit)
├── test_ucr_client_http.py             # HTTP Current client
├── test_ucr_client_zmq.py              # ZMQ Msgpack client
├── test_ucr_client_zmq_pickle.py       # NEW: ZMQ Pickle client (old UCR)
│
├── UCR_SERVER_MODES.md                 # NEW: Complete mode guide
├── UCR_SERVER_GUIDE.md                 # Modified: updated dependencies
├── UCR_SERVER_CHANGES.md               # NEW: HTTP changes doc
└── CHANGES_SUMMARY.md                  # NEW: This file
```

---

## Testing

### Test Current Branch (Msgpack ZMQ)
```bash
# Terminal 1
./launch_ucr_server.sh zmq

# Terminal 2
python test_ucr_client_zmq.py
```

### Test Old UCR Branch (Pickle ZMQ)
```bash
# Terminal 1
./launch_ucr_server.sh zmq-pickle

# Terminal 2
python test_ucr_client_zmq_pickle.py
```

### Test HTTP Legacy (Flat Format)
```bash
# Terminal 1
./launch_ucr_server.sh

# Terminal 2
python test_ucr_client.py
```

---

## Key Points

1. **All existing code works** - Nothing was broken
2. **New modes added** - Pickle ZMQ for old UCR compatibility
3. **Easy to switch** - Just change the mode argument
4. **Clear documentation** - Know which mode to use when
5. **No code changes needed** - Just use different launcher

---

## Quick Reference

| Your Situation | Use This Command |
|----------------|------------------|
| Have old UCR ZMQ client | `./launch_ucr_server.sh zmq-pickle` |
| Have old UCR HTTP client | `./launch_ucr_server.sh` |
| Using current branch | `./launch_ucr_server.sh zmq` |
| Want HTTP REST API | `./launch_ucr_server.sh http` |
| Not sure? | `./launch_ucr_server.sh` (HTTP Legacy default) |
