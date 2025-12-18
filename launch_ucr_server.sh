#!/bin/bash

# Launch GR00T inference server for UCR finetuned model
# Action horizon: 16 steps
# Embodiment: NEW_EMBODIMENT (ucr_wblm_moby_history)
#
# Usage:
#   ./launch_ucr_server.sh           # Default: HTTP mode with flat format (UCR branch style)
#   ./launch_ucr_server.sh legacy    # Same as default
#   ./launch_ucr_server.sh zmq       # ZMQ mode with nested format (current branch)
#   ./launch_ucr_server.sh http      # HTTP mode with nested format

CHECKPOINT_PATH="/data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000"
MODALITY_CONFIG="./ucr_modality_config.py"
HOST="192.168.0.137"
PORT=5555

# Parse mode argument
MODE="${1:-legacy}"

case "$MODE" in
  legacy)
    echo "=========================================="
    echo "UCR GR00T Server (UCR Branch Legacy Mode)"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Action horizon: 16 steps"
    echo "  Server: $HOST:$PORT"
    echo "  Mode: HTTP (legacy ucr branch style)"
    echo "  Format: Flat observations (video.ego_view, state.waist_joint, etc.)"
    echo ""
    echo "Endpoints:"
    echo "  POST http://$HOST:$PORT/act"
    echo "  GET  http://$HOST:$PORT/health"
    echo ""

    python gr00t/eval/run_gr00t_server.py \
      --model-path "$CHECKPOINT_PATH" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY_CONFIG" \
      --host "$HOST" \
      --port "$PORT" \
      --server-type http \
      --use-sim-policy-wrapper
    ;;

  zmq)
    echo "Starting GR00T server (ZMQ mode - current branch with msgpack)..."
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Action horizon: 16"
    echo "  Server: $HOST:$PORT"
    echo "  Mode: ZMQ with msgpack serialization"
    echo "  Format: Nested (video: {ego_view: ...}, state: {waist_joint: ...})"
    echo ""

    python gr00t/eval/run_gr00t_server.py \
      --model-path "$CHECKPOINT_PATH" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY_CONFIG" \
      --host "$HOST" \
      --port "$PORT" \
      --server-type zmq \
      --zmq-serialization msgpack
    ;;

  zmq-pickle)
    echo "Starting GR00T server (ZMQ mode - old UCR branch with pickle)..."
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Action horizon: 16"
    echo "  Server: $HOST:$PORT"
    echo "  Mode: ZMQ with pickle serialization (OLD UCR branch compatible)"
    echo "  Format: Nested (video: {ego_view: ...}, state: {waist_joint: ...})"
    echo ""

    python gr00t/eval/run_gr00t_server.py \
      --model-path "$CHECKPOINT_PATH" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY_CONFIG" \
      --host "$HOST" \
      --port "$PORT" \
      --server-type zmq \
      --zmq-serialization pickle
    ;;

  http)
    echo "Starting GR00T server (HTTP mode - current branch)..."
    echo "  Checkpoint: $CHECKPOINT_PATH"
    echo "  Action horizon: 16"
    echo "  Server: $HOST:$PORT"
    echo "  Mode: HTTP"
    echo "  Format: Nested (video: {ego_view: ...}, state: {waist_joint: ...})"
    echo ""

    python gr00t/eval/run_gr00t_server.py \
      --model-path "$CHECKPOINT_PATH" \
      --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$MODALITY_CONFIG" \
      --host "$HOST" \
      --port "$PORT" \
      --server-type http
    ;;

  *)
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [legacy|zmq|zmq-pickle|http]"
    echo "  legacy     - HTTP with flat format (UCR branch style) [DEFAULT]"
    echo "  zmq        - ZMQ with msgpack (current branch)"
    echo "  zmq-pickle - ZMQ with pickle (old UCR branch compatible)"
    echo "  http       - HTTP with nested format (current branch)"
    exit 1
    ;;
esac
