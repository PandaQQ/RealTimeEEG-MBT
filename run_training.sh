#!/bin/bash

# Exit on error
set -e

# Print each command before execution
set -x

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root directory
cd "$SCRIPT_DIR"

# Run the spa_export.py script
echo "Running spa_export.py..."
python ./training/spa_export.py

# Run the training_cwt.py script
echo "Running training_cwt.py..."
python ./training/traning_cwt.py

echo "EEG training pipeline completed successfully!"