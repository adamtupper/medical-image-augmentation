#!/bin/bash

# Switch to project root
cd ..

# Setup virtual environment
virtualenv .env -p python3.11
source .env/bin/activate
pip install --upgrade pip

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
