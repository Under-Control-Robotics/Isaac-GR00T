#!/bin/bash

# Launch GR00T inference server for UCR - ZMQ with PICKLE serialization (old UCR branch)
# This is for compatibility with old UCR branch clients
# Action horizon: 16 steps
# Embodiment: NEW_EMBODIMENT (ucr_wblm_moby_history)

CHECKPOINT_PATH="/data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000"
MODALITY_CONFIG="./ucr_modality_config.py"
HOST="192.168.0.137"
PORT=5555

echo "=========================================="
echo "UCR GR00T Server (ZMQ Pickle - Old UCR Branch)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Action horizon: 16 steps"
echo "  Server: $HOST:$PORT"
echo "  Mode: ZMQ with PICKLE serialization"
echo "  Compatible with: Old UCR branch clients"
echo ""
echo "This mode uses pickle serialization over ZMQ,"
echo "matching the old UCR branch implementation."
echo ""
echo "=========================================="
echo ""

python gr00t/eval/run_gr00t_server.py \
  --model-path "$CHECKPOINT_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --server-type zmq \
  --zmq-serialization pickle
