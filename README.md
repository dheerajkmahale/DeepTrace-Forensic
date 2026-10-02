# Deepfake Detector

A modular, reproducible deep learning system for detecting facial deepfake videos using a hybrid CNN-LSTM architecture. The pipeline extracts spatial features from individual frames and models temporal inconsistencies across frame sequences.

---

## Current Status & Verification

- **Pipeline & Prototype Status**: FULLY FUNCTIONAL and verified end-to-end using procedural synthetic video data.
- **Interactive UI**: Streamlit web demo (`app.py`) is verified and runs locally on `http://localhost:8501` using the standalone prototype model (`outputs/demo/best_model.keras`).
- **Real Experiment Status**: The real Celeb-DF v2 experiment is pending. All evaluation metrics below remain empty ("not yet run"). No performance numbers (accuracy, precision, recall, F1, or AUC) are claimed.

---

## Problem Statement

The rapid advancement of deep generative modeling (e.g., autoencoders, GANs, and diffusion models) has made high-fidelity facial manipulation accessible. Deepfake videos pose serious threats to information integrity, personal privacy, and security.

Detecting manipulated videos requires examining both **spatial artifacts** (e.g., blending boundaries, color discrepancies, warping distortions) and **temporal incoherence** (e.g., unnatural blinking patterns, lighting flicker across successive frames, erratic landmark motion).

---

## Approach & Architecture

This repository implements a two-stage hybrid spatio-temporal deep learning network:

```
Video Clip: (seq_len, H, W, 3) [uint8]
   │
   ▼
Rescaling(scale=1/127.5, offset=-1.0) -> [-1.0, 1.0]
   │
   ▼
TimeDistributed(CNN Feature Extractor)
   │  ├── light: 4 Conv2D blocks (32, 64, 128, 128 filters)
   │  │          BatchNormalization (momentum=0.9) + ReLU + MaxPool2D
   │  │          GlobalAveragePooling2D
   │  └── mobilenetv2: MobileNetV2 (optional ImageNet weights) + GlobalAvgPool
   │
   ▼
Dropout(0.4)
   │
   ▼
LSTM(units=128) -> Captures temporal dynamics and cross-frame anomalies
   │
   ▼
Dropout(0.4)
   │
   ▼
Dense(64, activation='relu')
   │
   ▼
Dense(1, activation='sigmoid') -> P(fake) ∈ [0, 1]
```

### Key Design Principles:
1. **BatchNorm Momentum Calibration**: The lightweight CNN backbone uses `momentum=0.9` (rather than default 0.99) to ensure batch statistics update rapidly and prevent inference calibration lag on small datasets.
2. **Strict Video-Level Splitting**: Splitting into train, validation, and test sets occurs strictly at the **video level**. All clips from a given video belong to exactly one split, preventing temporal and visual data leakage.
3. **Clip & Video Level Evaluation**: Video-level prediction aggregates clip probabilities via mean pooling, providing robust sequence-level verdicts.
4. **Reproducible Pipelines**: All splitting, training, and evaluation scripts accept fixed seeds and export machine-readable configuration and metric logs.

---

## Pipeline Overview

| Stage | Script | Input | Output | Description |
|---|---|---|---|---|
| **1. Preprocess** | `preprocess.py` | `data/raw/` (or custom) | `data/processed/clips/*.npy`, `meta.csv` | Resumable video extraction, Haar face detection, atomic `meta.csv` writes, skip log (`outputs/preprocess_skipped.csv`). |
| **2. Dataset** | `dataset.py` | `data/processed/meta.csv` | `tf.data.Dataset` | Generator-based data pipeline with clip-consistent horizontal flip and brightness augmentations. |
| **3. Train** | `train.py` | `data/processed/` | `outputs/best_model.keras`, `outputs/run_info.json` | Adam optimizer, binary crossentropy, dynamic class weights, early stopping, and LR scheduling on `val_loss`. |
| **4. Evaluate** | `evaluate.py` | `outputs/best_model.keras`, `meta.csv` | `outputs/metrics.json`, `predictions.csv`, plots | Evaluates held-out TEST split only. Records test set real/fake counts and threshold (0.5 or tuned on val split). |
| **5. Predict** | `predict.py` | Any target `.mp4` video | Console verdict | Runs identical preprocessing, predicts all clips, and outputs aggregate $P(\text{fake})$ and verdict. |
| **6. Demo** | `demo.py` | Self-contained | `data/demo/`, `outputs/demo/` | End-to-end synthetic demo generating videos, training, evaluating, and running prediction. |
| **Runner** | `scripts/run_real_experiment.ps1` | `data/raw/` (or `-DatasetPath`) | `outputs/logs/*.log`, all artifacts | One-command orchestrator executing pre-flight check, validation, preprocess, training, evaluation, and prediction. |

---

## Dataset Acquisition & Setup (Celeb-DF v2)

This project is designed to benchmark on the **Celeb-DF (v2)** dataset (consisting of approximately 590 `Celeb-real`, 300 `YouTube-real`, and 5,639 `Celeb-synthesis` videos, with 518 entries in `List_of_testing_videos.txt`).

> **DATASET POLICY**: In compliance with dataset terms of service, Celeb-DF is **NOT** bundled with this repository and is **NOT** downloaded automatically. Users must request access through official academic channels.

### How to Obtain Celeb-DF v2
1. Visit the official Celeb-DF project page or GitHub repository: [Celeb-DF on GitHub](https://github.com/yuezunli/celeb-deepfakeforensics).
2. Complete and submit the official Celeb-DF Access Request Form agreeing to research use terms.
3. Once download links are granted by the authors, download and unpack the dataset archives.

### Expected Directory Organization
Organize downloaded Celeb-DF v2 files inside `data/raw/` (or any custom directory, e.g. `D:\datasets\Celeb-DF-v2`):

```text
data/raw/
  |-- Celeb-real/               (or real/Celeb-real/)
  |     |-- id0_0000.mp4
  |     \-- ...
  |-- YouTube-real/             (or real/YouTube-real/)
  |     |-- 00000.mp4
  |     \-- ...
  |-- Celeb-synthesis/          (or fake/Celeb-synthesis/)
  |     |-- id0_id1_0000.mp4
  |     \-- ...
  \-- List_of_testing_videos.txt
```

*Note: Preprocessing discovers videos recursively. Nested subdirectories (e.g. `real/Celeb-real/videos/*.mp4`) and case-insensitive folder names (`youtube-real`, `celeb-real`, `celeb-synthesis`) are automatically detected.*

### Official Test-List Label Convention & Strict Check
The official Celeb-DF v2 `List_of_testing_videos.txt` lists videos with numeric labels:
- Official list convention assumption: `1 = real`, `0 = fake`
- This project's internal convention: `0 = real`, `1 = fake`

> **Unverified Assumption Notice**: The assumption that the official Celeb-DF v2 testing list uses `1 = real` and `0 = fake` is an **unverified assumption until checked against the real file**.

**Why a strict check exists:**
Because this mapping is an unverified assumption until the genuine dataset is inspected on disk, the parser is built to fail loudly if this assumption is incorrect. `preprocess.py` maps official test-list labels via `convert_official_list_label_to_project_label` and strictly cross-references each entry against the folder-derived class (`Celeb-real` / `YouTube-real` for real vs `Celeb-synthesis` for fake). If any mismatch, unexpected numeric value, or missing video occurs, parsing halts immediately with a clear `ValueError` detailing the exact line number. Labels are never inferred from numeric columns alone without folder verification.

---

## One-Command Real Experiment Runner

To run the complete real experiment pipeline with a single command:

```powershell
# Using default data/raw path:
.\scripts\run_real_experiment.ps1

# Using custom dataset path outside OneDrive (e.g., on SSD) with multiprocessing:
.\scripts\run_real_experiment.ps1 -DatasetPath "D:\datasets\Celeb-DF-v2" -Workers 4
```

### Runner Workflow:
1. **Pre-Flight Validation**: Refuses to start if any of the four required items (`Celeb-real`, `YouTube-real`, `Celeb-synthesis`, `List_of_testing_videos.txt`) are missing, identifying exactly which ones are absent.
2. **Stage 1 (validate_dataset_raw)**: Checks raw video file readability, formats, and test list integrity.
3. **Stage 2 (preprocess)**: Extracts faces, writes `.npy` clips, saves `meta.csv` atomically, logs skipped videos.
4. **Stage 3 (validate_dataset_processed)**: Asserts zero cross-split video/clip leakage, validates tensor shapes and uint8 datatypes.
5. **Stage 4 (train)**: Trains CNN-LSTM model with class weights and callbacks, records `outputs/run_info.json`.
6. **Stage 5 (evaluate)**: Evaluates held-out test split, exports `metrics.json`, `predictions.csv`, and plots.
7. **Stage 6 (predict)**: Runs single-video inference on a sample video to verify end-to-end deployment readiness.

*Each stage's terminal output is teed to `outputs/logs/<stage>.log`. If any stage returns a non-zero exit code, the runner stops immediately.*

---

## Interactive Web Prototype (Streamlit)

Launch the interactive prototype UI locally at `http://localhost:8501`:
```bash
streamlit run app.py
```
The UI allows uploading videos, inspecting frame extraction and face detection, visualizing prediction confidence, and switching between prototype mode (`outputs/demo/best_model.keras`) and trained mode (`outputs/best_model.keras`).

---

## Results & Performance

> **Metric Honesty Commitment**: Performance metrics are populated only after a complete real training and evaluation run on Celeb-DF v2. Synthetic prototype metrics are never presented as model capabilities.

### Celeb-DF Evaluation Results

| Metric | Clip-Level Result | Video-Level Result |
|---|---|---|
| **Accuracy** | not yet run | not yet run |
| **Precision** | not yet run | not yet run |
| **Recall** | not yet run | not yet run |
| **F1-Score** | not yet run | not yet run |
| **ROC-AUC** | not yet run | not yet run |

---

## Limitations

1. **Haar Cascade Face Detection**: OpenCV Haar cascade detectors are fast and CPU-friendly, but can fail on non-frontal faces, rapid head rotations, partial occlusions, low lighting, or compression artifacts compared to deep neural detectors (e.g., RetinaFace, MTCNN).
2. **Severe Class Imbalance**: Celeb-DF v2 has ~890 real videos and ~5,639 fake videos (~6.3:1 fake-to-real ratio). Although dynamic class weights are computed during training, unweighted evaluation metrics must be interpreted cautiously; ROC-AUC, Precision, and Recall are critical.
3. **Celebrity Identity Overlap Between Train and Test**: The official Celeb-DF v2 benchmark split protocol (`List_of_testing_videos.txt`) allows certain celebrity identities to appear in both training and test sets. While split disjointness guarantees that no video or frame leaks, models may inadvertently learn identity-specific facial features rather than pure manipulation artifacts.
4. **Single-Dataset Evaluation & Lack of Cross-Dataset Testing**: The model is evaluated solely on Celeb-DF v2. Deepfake detectors frequently suffer performance degradation when tested across different datasets (e.g., FaceForensics++, DFDC) or unseen manipulation techniques (diffusion-based inpainting, neural face reenactment).

---

## Project Structure

```text
deepfake-detector/
├── config.py                       # Configuration dataclass defaults
├── preprocess.py                   # Resumable extraction, atomic writes, skip log, workers
├── validate_dataset.py             # Pre-training data audit (leakage, shapes, readability)
├── dataset.py                      # tf.data pipeline with consistent clip augmentations
├── model.py                        # CNN-LSTM architecture (light and MobileNetV2)
├── train.py                        # Training pipeline with run_info.json export
├── evaluate.py                     # Held-out test evaluation, val threshold tuning, metrics
├── predict.py                      # CLI inference and verdict generation
├── app.py                          # Streamlit interactive web interface
├── demo.py                         # End-to-end synthetic demonstration pipeline
├── requirements.txt                # Project dependencies
├── .gitignore                      # Clean git tracking preserving .gitkeep
├── README.md                       # Documentation and honest results
│
├── scripts/
│   └── run_real_experiment.ps1     # One-command real experiment runner
│
├── tests/
│   └── test_pipeline.py            # Unit tests for pipeline, splitting, and resume
│
└── data/
    └── raw/
        ├── real/
        │   └── .gitkeep            # Destination for real videos (Celeb-real, YouTube-real)
        └── fake/
            └── .gitkeep            # Destination for fake videos (Celeb-synthesis)
```
