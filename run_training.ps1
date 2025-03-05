# Exit on error
$ErrorActionPreference = "Stop"

# Change to project root directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

.\.venv\Scripts\Activate

# Run the spa_export.py script
Write-Host "Running spa_export.py..."
python .\training\spa_export.py

# Run the training_cwt.py script
Write-Host "Running training_cwt.py..."
python .\training\traning_cwt.py

Write-Host "EEG training pipeline completed successfully!"