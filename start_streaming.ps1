# Exit on error
$ErrorActionPreference = "Stop"

# Change to project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate

# # Run the streaming.py script in the background
# Write-Host "Running streaming.py..."
# Start-Process python -ArgumentList "./real-time-streaming/eeglab_streaming.py" -NoNewWindow

# Wait a moment to ensure the stream is initialized
Write-Host "Waiting for stream initialization..."
Start-Sleep -Seconds 2

# Run the eeglab_receiving.py script
Write-Host "Running eeglab_receving_2.py..."
python ./real-time-streaming/eeglab_receving.py
