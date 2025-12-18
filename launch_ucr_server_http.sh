#!/bin/bash

# Launch GR00T inference server for UCR finetuned model (HTTP mode - legacy from ucr branch)
# Action horizon: 16 steps
# Embodiment: NEW_EMBODIMENT (ucr_wblm_moby_history)
# Dependencies: pip install uvicorn fastapi
# Optional: pip install json-numpy (for faster numpy serialization)

CHECKPOINT_PATH="/data/anthony/Isaac-GR00T/checkpoints/1217_ucr_17_finetune/checkpoint-25000"
MODALITY_CONFIG="./ucr_modality_config.py"
HOST="0.0.0.0"
PORT=5555

echo "Starting GR00T server for UCR model (HTTP mode)..."
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Action horizon: 16"
echo "  Server: $HOST:$PORT"
echo "  Mode: HTTP (legacy from ucr branch)"
echo "  Endpoint: POST http://$HOST:$PORT/act"
echo ""

python gr00t/eval/run_gr00t_server.py \
  --model-path "$CHECKPOINT_PATH" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG" \
  --host "$HOST" \
  --port "$PORT" \
  --server-type http
