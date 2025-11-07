#!/bin/bash
# Script to restart training with fixed code

cd "$(dirname "$0")"

echo "?? Stopping any running training processes..."
pkill -f "python.*03_train" || echo "No training processes found"

echo "? Waiting 2 seconds..."
sleep 2

echo "? Activating virtual environment..."
source venv/bin/activate

echo "?? Starting training..."
python 03_train.py

