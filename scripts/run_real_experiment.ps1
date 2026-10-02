<#
.SYNOPSIS
    One-command automated execution pipeline for the real Celeb-DF v2 experiment.

.DESCRIPTION
    Runs the full deepfake detection pipeline in order:
    1. validate_dataset (raw)
    2. preprocess
    3. validate_dataset on processed data
    4. train
    5. evaluate
    6. predict

    Stops immediately on the first non-zero exit code.
    Tees each stage's output to outputs/logs/<stage>.log.
    Refuses to start if any of the four required Celeb-DF dataset items are missing.

.PARAMETER DatasetPath
    Root directory containing Celeb-DF v2 dataset. Defaults to 'data/raw'.

.PARAMETER Workers
    Number of parallel worker processes for preprocessing (default: 1).

.PARAMETER Force
    Force re-extraction of all clips even if existing readable clips are found.

.PARAMETER SampleVideo
    Optional path to sample video for prediction stage.
#>

[CmdletBinding()]
param (
    [string]$DatasetPath = "data/raw",
    [int]$Workers = 1,
    [switch]$Force,
    [string]$SampleVideo = ""
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " CELEB-DF v2 EXPERIMENT RUNNER" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "Dataset Path: $DatasetPath" -ForegroundColor Yellow
Write-Host "Workers:      $Workers" -ForegroundColor Yellow
Write-Host "Force:        $Force" -ForegroundColor Yellow
Write-Host ""

# -----------------------------------------------------------------------------
# Pre-Flight: Check the 4 required Celeb-DF v2 dataset items
# -----------------------------------------------------------------------------
$missingItems = @()

$hasCelebReal = (Test-Path (Join-Path $DatasetPath "Celeb-real")) -or (Test-Path (Join-Path $DatasetPath "real/Celeb-real"))
if (-not $hasCelebReal) {
    $missingItems += "Celeb-real directory (checked '$DatasetPath/Celeb-real' and '$DatasetPath/real/Celeb-real')"
}

$hasYoutubeReal = (Test-Path (Join-Path $DatasetPath "YouTube-real")) -or (Test-Path (Join-Path $DatasetPath "real/YouTube-real")) -or (Test-Path (Join-Path $DatasetPath "Youtube-real")) -or (Test-Path (Join-Path $DatasetPath "real/Youtube-real"))
if (-not $hasYoutubeReal) {
    $missingItems += "YouTube-real directory (checked '$DatasetPath/YouTube-real' and '$DatasetPath/real/YouTube-real')"
}

$hasCelebSynthesis = (Test-Path (Join-Path $DatasetPath "Celeb-synthesis")) -or (Test-Path (Join-Path $DatasetPath "fake/Celeb-synthesis"))
if (-not $hasCelebSynthesis) {
    $missingItems += "Celeb-synthesis directory (checked '$DatasetPath/Celeb-synthesis' and '$DatasetPath/fake/Celeb-synthesis')"
}

$hasTestList = (Test-Path (Join-Path $DatasetPath "List_of_testing_videos.txt"))
if (-not $hasTestList) {
    $missingItems += "List_of_testing_videos.txt file (checked '$DatasetPath/List_of_testing_videos.txt')"
}

if ($missingItems.Count -gt 0) {
    Write-Host "=====================================================================" -ForegroundColor Red
    Write-Host "EXPERIMENT REFUSED TO START: Missing Required Celeb-DF v2 Dataset Items" -ForegroundColor Red
    Write-Host "=====================================================================" -ForegroundColor Red
    Write-Host "The following required dataset items were not found in '$DatasetPath':" -ForegroundColor Red
    foreach ($item in $missingItems) {
        Write-Host "  - [MISSING] $item" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Expected dataset folder structure:" -ForegroundColor Cyan
    Write-Host "  $DatasetPath/" -ForegroundColor Cyan
    Write-Host "    |-- Celeb-real/           (or real/Celeb-real/)" -ForegroundColor Cyan
    Write-Host "    |-- YouTube-real/         (or real/YouTube-real/)" -ForegroundColor Cyan
    Write-Host "    |-- Celeb-synthesis/      (or fake/Celeb-synthesis/)" -ForegroundColor Cyan
    Write-Host "    \-- List_of_testing_videos.txt" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Please place the official dataset at '$DatasetPath' or run:" -ForegroundColor Yellow
    Write-Host '  .\scripts\run_real_experiment.ps1 -DatasetPath "PATH_TO_CELEBDF"' -ForegroundColor Yellow
    exit 1
}

Write-Host "All 4 required Celeb-DF dataset items verified on disk." -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------------------------
# Setup Logs Directory
# -----------------------------------------------------------------------------
$logDir = "outputs/logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

function Invoke-Stage {
    param (
        [string]$StageName,
        [string[]]$ArgumentList
    )
    $logFile = Join-Path $logDir "$StageName.log"
    Write-Host ""
    Write-Host "=====================================================================" -ForegroundColor Cyan
    Write-Host " STAGE: $StageName" -ForegroundColor Cyan
    Write-Host " RUNNING: python $($ArgumentList -join ' ')" -ForegroundColor Gray
    Write-Host " LOG: $logFile" -ForegroundColor Gray
    Write-Host "=====================================================================" -ForegroundColor Cyan

    & python $ArgumentList 2>&1 | Tee-Object -FilePath $logFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "=====================================================================" -ForegroundColor Red
        Write-Host " STAGE FAILED: $StageName (Exit Code: $LASTEXITCODE)" -ForegroundColor Red
        Write-Host " Review log file for details: $logFile" -ForegroundColor Red
        Write-Host " Pipeline stopped immediately." -ForegroundColor Red
        Write-Host "=====================================================================" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "--> Stage '$StageName' completed successfully." -ForegroundColor Green
}

# -----------------------------------------------------------------------------
# Stage 1: Validate Raw Dataset
# -----------------------------------------------------------------------------
Invoke-Stage -StageName "validate_dataset_raw" -ArgumentList @("validate_dataset.py", "--raw-dir", $DatasetPath, "--check-raw")

# -----------------------------------------------------------------------------
# Stage 2: Preprocess Dataset
# -----------------------------------------------------------------------------
$preprocessArgs = @("preprocess.py", "--raw-dir", $DatasetPath, "--workers", "$Workers")
if ($Force) {
    $preprocessArgs += "--force"
}
Invoke-Stage -StageName "preprocess" -ArgumentList $preprocessArgs

# -----------------------------------------------------------------------------
# Stage 3: Validate Processed Dataset
# -----------------------------------------------------------------------------
Invoke-Stage -StageName "validate_dataset_processed" -ArgumentList @("validate_dataset.py", "--processed-dir", "data/processed")

# -----------------------------------------------------------------------------
# Stage 4: Train CNN-LSTM Model
# -----------------------------------------------------------------------------
Invoke-Stage -StageName "train" -ArgumentList @("train.py", "--processed-dir", "data/processed", "--output-dir", "outputs")

# -----------------------------------------------------------------------------
# Stage 5: Evaluate on Held-Out Test Split
# -----------------------------------------------------------------------------
Invoke-Stage -StageName "evaluate" -ArgumentList @("evaluate.py", "--model-path", "outputs/best_model.keras", "--processed-dir", "data/processed", "--output-dir", "outputs", "--val-threshold")

# -----------------------------------------------------------------------------
# Stage 6: Predict on Sample Video
# -----------------------------------------------------------------------------
if (-not $SampleVideo -or -not (Test-Path $SampleVideo)) {
    $foundVid = Get-ChildItem -Path $DatasetPath -Recurse -Include *.mp4,*.avi,*.mov -File | Select-Object -First 1
    if ($foundVid) {
        $SampleVideo = $foundVid.FullName
    }
}

if ($SampleVideo -and (Test-Path $SampleVideo)) {
    Invoke-Stage -StageName "predict" -ArgumentList @("predict.py", $SampleVideo, "--model-path", "outputs/best_model.keras")
} else {
    Write-Host "[WARNING] No sample video found to execute stage 'predict'." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host " REAL EXPERIMENT PIPELINE COMPLETED SUCCESSFULLY" -ForegroundColor Green
Write-Host "=====================================================================" -ForegroundColor Green
Write-Host "All stages finished with exit code 0." -ForegroundColor Green
Write-Host "Logs saved in: $logDir" -ForegroundColor Cyan
