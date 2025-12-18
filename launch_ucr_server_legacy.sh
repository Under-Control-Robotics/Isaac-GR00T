#!/bin/bash

# Launch GR00T inference server for UCR - UCR branch legacy style
# This mimics the old ucr branch approach:
# - Uses HTTP server with /act endpoint
# - Uses flat observation format (video.ego_view, state.waist_joint, etc.)
# - Compatible with old UCR client code
#
# Equivalent to ucr branch:
#   python scripts/inference_service.py --server --http-server \
#     --model-path <path> --data-config ucr_wblm_moby_history --embodiment-tag gr1

CHECKPOINT_PATH="/data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000"
MODALITY_CONFIG="./ucr_modality_config.py"
HOST="0.0.0.0"
PORT=5555

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
echo "Dependencies:"
echo "  Required: pip install uvicorn fastapi"
echo "  Optional: pip install json-numpy (for faster numpy serialization)"
echo ""
echo "=========================================="
echo ""

python gr00t/eval/run_gr00t_server.py \
  --model-path "$CHECKPOINT_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --server-type http \
  --use-sim-policy-wrapper
