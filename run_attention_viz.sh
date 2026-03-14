#!/bin/bash

cd /data/anthony/Isaac-GR00T

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || eval "$(conda shell.bash hook)"

# Activate the isolated attention_viz environment
conda activate attention_viz

# Verify environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Check if torch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "ERROR: PyTorch not found!"

# Run visualization
python scripts/visualize_attention_on_video.py
