#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0

echo "---------------- GR00T TensorRT Engine Builder ----------------"
echo "Notes:"
echo "1: Default engine batch size = 8"
echo "2: Default MIN_LEN/OPT_LEN/MAX_LEN for sequence inputs = 80/283/300"
echo "3: Adjustable via joint_history / image_history / action_horizon"

export PATH=/usr/src/tensorrt/bin:$PATH

# ----------- User Parameters -------------
JOINT_HISTORY=11         # number of historical states
IMAGE_HISTORY=1        # number of image frames
ACTION_HORIZON=64      # number of predicted actions
BATCH=8                       # max batch size

MIN_LEN=80
OPT_LEN=283
MAX_LEN=600

# Derived sequence lengths for models
DIT_SEQ=$((49 + JOINT_HISTORY - 1 + ACTION_HORIZON - 16))
STATE_SEQ=$JOINT_HISTORY
ACTION_SEQ=$ACTION_HORIZON
DECODER_SEQ=$DIT_SEQ
VIT_SEQ=256
LLM_VIT_SEQ=$((VIT_SEQ * IMAGE_HISTORY))

echo "-------------------------------------------------------------"
echo " Joint History    : $JOINT_HISTORY"
echo " Image History    : $IMAGE_HISTORY"
echo " Action Horizon   : $ACTION_HORIZON"
echo " DIT Seq Length   : $DIT_SEQ"
echo " State Seq Length : $STATE_SEQ"
echo " Action Seq Len   : $ACTION_SEQ"
echo " Decoder Seq Len  : $DECODER_SEQ"
echo " VLM-LLM vit_embeds seq: $LLM_VIT_SEQ"
echo "-------------------------------------------------------------"

if [ ! -e /usr/src/tensorrt/bin/trtexec ]; then
    echo "❌ TensorRT 'trtexec' not found! Please install TensorRT."
    exit 1
fi

mkdir -p gr00t_engine

# VLLN-VLSelfAttention
echo "------------Building vlln_vl_self_attention Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/action_head/vlln_vl_self_attention.onnx \
  --saveEngine=gr00t_engine/vlln_vl_self_attention.engine \
  --minShapes=backbone_features:1x${MIN_LEN}x2048 \
  --optShapes=backbone_features:1x${OPT_LEN}x2048 \
  --maxShapes=backbone_features:${BATCH}x${MAX_LEN}x2048 \
  > gr00t_engine/vlln_vl_self_attention.log 2>&1

# DiT Model
echo "------------Building DiT Model--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/action_head/DiT.onnx \
  --saveEngine=gr00t_engine/DiT.engine \
  --minShapes=sa_embs:1x${DIT_SEQ}x1536,vl_embs:1x${MIN_LEN}x2048,timesteps_tensor:1  \
  --optShapes=sa_embs:1x${DIT_SEQ}x1536,vl_embs:1x${OPT_LEN}x2048,timesteps_tensor:1  \
  --maxShapes=sa_embs:${BATCH}x${DIT_SEQ}x1536,vl_embs:${BATCH}x${MAX_LEN}x2048,timesteps_tensor:${BATCH} \
  > gr00t_engine/build_DiT.log 2>&1

# State Encoder
echo "------------Building State Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/action_head/state_encoder.onnx \
  --saveEngine=gr00t_engine/state_encoder.engine \
  --minShapes=state:1x${STATE_SEQ}x64,embodiment_id:1  \
  --optShapes=state:1x${STATE_SEQ}x64,embodiment_id:1 \
  --maxShapes=state:${BATCH}x${STATE_SEQ}x64,embodiment_id:${BATCH} \
  > gr00t_engine/build_state_encoder.log 2>&1

# Action Encoder
echo "------------Building Action Encoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/action_head/action_encoder.onnx \
  --saveEngine=gr00t_engine/action_encoder.engine \
  --minShapes=actions:1x${ACTION_SEQ}x32,timesteps_tensor:1,embodiment_id:1  \
  --optShapes=actions:1x${ACTION_SEQ}x32,timesteps_tensor:1,embodiment_id:1  \
  --maxShapes=actions:${BATCH}x${ACTION_SEQ}x32,timesteps_tensor:${BATCH},embodiment_id:${BATCH} \
  > gr00t_engine/build_action_encoder.log 2>&1

# Action Decoder
echo "------------Building Action Decoder--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/action_head/action_decoder.onnx \
  --saveEngine=gr00t_engine/action_decoder.engine \
  --minShapes=model_output:1x${DECODER_SEQ}x1024,embodiment_id:1  \
  --optShapes=model_output:1x${DECODER_SEQ}x1024,embodiment_id:1  \
  --maxShapes=model_output:${BATCH}x${DECODER_SEQ}x1024,embodiment_id:${BATCH} \
  > gr00t_engine/build_action_decoder.log 2>&1

# VLM-ViT
echo "------------Building VLM-ViT--------------------"
trtexec --useCudaGraph --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/eagle2/vit.onnx  \
  --saveEngine=gr00t_engine/vit.engine \
  --minShapes=pixel_values:1x3x224x224,position_ids:1x${VIT_SEQ} \
  --optShapes=pixel_values:1x3x224x224,position_ids:1x${VIT_SEQ} \
  --maxShapes=pixel_values:${BATCH}x3x224x224,position_ids:${BATCH}x${VIT_SEQ}  \
  > gr00t_engine/vit.log 2>&1

# VLM-LLM
echo "------------Building VLM-LLM--------------------"
trtexec --verbose --stronglyTyped --separateProfileRun --noDataTransfers \
  --onnx=gr00t_onnx/eagle2/llm.onnx  \
  --saveEngine=gr00t_engine/llm.engine \
  --minShapes=input_ids:1x${MIN_LEN},vit_embeds:1x${LLM_VIT_SEQ}x1152,attention_mask:1x${MIN_LEN} \
  --optShapes=input_ids:1x${OPT_LEN},vit_embeds:1x${LLM_VIT_SEQ}x1152,attention_mask:1x${OPT_LEN} \
  --maxShapes=input_ids:${BATCH}x${MAX_LEN},vit_embeds:${BATCH}x${LLM_VIT_SEQ}x1152,attention_mask:${BATCH}x${MAX_LEN} \
  > gr00t_engine/llm.log 2>&1

echo "✅ All engines built successfully."
