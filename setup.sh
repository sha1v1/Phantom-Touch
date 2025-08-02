#!/bin/bash

# Setup script for Phantom Touch Project

echo "Setting up Python virtual environment (Python 3.12 required)..."

# Check Python version
PYVER=$(python3 --version 2>&1)
if [[ $PYVER != *"Python 3.12"* ]]; then
  echo "Python 3.12 is required. Detected: $PYVER"
  echo "Please install Python 3.12 manually before running this script."
  exit 1
fi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip

pip install -r requirements.txt

echo "Setup complete."
echo "To activate your environment later, run: source .venv/bin/activate"
