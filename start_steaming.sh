#!/bin/bash

# Exit on error
set -e

# Print each command before execution
set -x

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root directory
cd "$SCRIPT_DIR"

# Run the streaming.py script in the background
echo "Running streaming.py..."
python ./real-time-streaming/eeglab_streaming.py &

# Wait a moment to ensure the stream is initialized
sleep 2

# Run the eeglab_receiving.py script
echo "Running eeglab_receving.py..."
python ./real-time-streaming/eeglab_receving.py

# Run the spa_export.py script
#echo "Running spa_export.py..."
#python ./training/spa_export.py
#
## Run the training_cwt.py script
#echo "Running training_cwt.py..."
#python ./training/traning_cwt.py
#
#echo "EEG training pipeline completed successfully!"