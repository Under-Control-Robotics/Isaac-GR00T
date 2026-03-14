#!/bin/bash

# Create new isolated environment for attention visualization
echo "Creating new conda environment: attention_viz"

eval "$(conda shell.bash hook)"

# Create new environment with Python 3.10
conda create -n attention_viz python=3.10 -y

# Activate the new environment
conda activate attention_viz

# Install required packages
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers diffusers accelerate
pip install datasets opencv-python matplotlib seaborn
pip install huggingface_hub

echo "✓ Environment setup complete!"
echo "To use: conda activate attention_viz"
