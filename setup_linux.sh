#!/bin/bash
# -------------------------------------------------------------
# Paper Review System - Native Linux Setup Script
# -------------------------------------------------------------

echo "Starting Paper Review System Linux Setup..."

# Install basic dependencies (requires sudo)
if [ -x "$(command -v apt-get)" ]; then
    echo "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y python3-venv python3-pip
elif [ -x "$(command -v dnf)" ]; then
    sudo dnf install -y python3-virtualenv python3-pip
fi

# Create Virtual Environment
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

# Activate venv and install requirements
echo "Installing Python requirements..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Ensure models directory exists
mkdir -p local_models

echo ""
echo "Setup Complete!"
echo "To start the application, run: ./run_linux.sh"
