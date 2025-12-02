#!/bin/bash
# Render build script

# Install system dependencies
apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0

# Install Python dependencies
pip install -r requirements.txt
