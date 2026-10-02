# Deepfake Detector

A modular, reproducible deep learning system for detecting facial deepfake videos using a hybrid CNN-LSTM architecture. The pipeline extracts spatial features from individual frames and models temporal inconsistencies across frame sequences.

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
1. **BatchNorm Momentum Calibration**: The lightweight CNN backbone uses `momentum=0.9` (rather than the default 0.99) to ensure batch statistics update rapidly and prevent inference calibration lag on small datasets.
2. **Strict Video-Level Splitting**: Splitting into train, validation, and test sets occurs strictly at the **video level**. All clips from a given video belong to exactly one split, preventing temporal and visual data leakage.
3. **Clip & Video Level Evaluation**: Video-level prediction aggregates clip probabilities via mean pooling, providing robust sequence-level verdicts.
4. **Reproducible Pipelines**: All splitting, training, and evaluation scripts accept fixed seeds and export machine-readable configuration and metric logs.

---

## Pipeline Overview

| Stage | Script | Input | Output | Description |
|---|---|---|---|---|
| **1. Preprocess** | `preprocess.py` | `data/raw/real/`, `data/raw/fake/` | `data/processed/clips/*.npy`, `meta.csv` | Video-level splitting, Haar face detection, temporal segment extraction, RGB normalization. |
| **2. Dataset** | `dataset.py` | `data/processed/meta.csv` | `tf.data.Dataset` | Generator-based data pipeline with clip-consistent horizontal flip and brightness augmentations. |
| **3. Train** | `train.py` | `data/processed/` | `outputs/best_model.keras`, `history.json` | Adam optimizer, binary crossentropy, class weights, early stopping, and LR scheduling on `val_loss`. |
| **4. Evaluate** | `evaluate.py` | `outputs/best_model.keras`, `meta.csv` | `metrics.json`, `predictions.csv`, plots | Evaluates held-out TEST split only. Computes clip and video level Accuracy, Precision, Recall, F1, ROC-AUC. |
| **5. Predict** | `predict.py` | Any target `.mp4` video | Console verdict | Runs identical preprocessing, predicts all clips, and outputs aggregate $P(\text{fake})$ and verdict. |
| **6. Demo** | `demo.py` | Self-contained | `data/demo/`, `outputs/demo/` | End-to-end synthetic demo generating videos, training, evaluating, and running prediction. |

---

## Dataset Information & Celeb-DF Setup

This project is designed to benchmark on the **Celeb-DF (v2)** dataset.

> **IMPORTANT**: In compliance with dataset terms of service, Celeb-DF is **NOT** bundled with this repository and must **NOT** be downloaded automatically by scripts. Users must request access through official academic channels.

### Obtaining Celeb-DF
1. Visit the official Celeb-DF project page / GitHub repository.
2. Submit the access request form agreeing to the Celeb-DF research license terms.
3. Once approved, download the dataset archives.

### Expected Directory Organization
Organize downloaded Celeb-DF v2 videos inside `data/raw/` as follows:

```text
data/
└── raw/
    ├── List_of_testing_videos.txt  # Optional: official Celeb-DF test video list
    ├── real/
    │   ├── .gitkeep
    │   ├── Celeb-real/
    │   │   ├── id0_0000.mp4
    │   │   └── ...
    │   └── YouTube-real/
    │       ├── 00000.mp4
    │       └── ...
    │
    └── fake/
        ├── .gitkeep
        └── Celeb-synthesis/
            ├── id0_id1_0000.mp4
            └── ...
```

*Note: Preprocessing discovers videos recursively within `data/raw/real/` and `data/raw/fake/`. Flat structures (`data/raw/real/*.mp4`) and nested subdirectories (`data/raw/real/Celeb-real/`) are both automatically supported.*

### Official Celeb-DF Test Protocol
Celeb-DF provides an official test video partition (`List_of_testing_videos.txt`).
- If this file is placed at `data/raw/List_of_testing_videos.txt` (or specified via `--test-list`), `preprocess.py` automatically assigns those listed videos to the `test` split, splitting the remainder into `train` and `val`.
- If the official list is not present locally, `preprocess.py` clearly reports:
  ```text
  [INFO] Official Celeb-DF testing-video list not provided locally. Using random video-level splitting.
  ```
  and performs stratified random video-level splitting.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- TensorFlow / Keras >= 2.16

### Install Dependencies
```bash
git clone https://github.com/dheerajkmahale/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
```

---

## Execution Modes: Prototype vs. Real Experiment

This project maintains a strict distinction between the synthetic prototype and real experiments:

| Aspect | Synthetic Demo Prototype | Real Experiment (Celeb-DF v2) |
|---|---|---|
| **Data Source** | Procedural moving-blob videos (`data/demo/raw/`) | Celeb-DF v2 (`data/raw/`) |
| **Model Checkpoint** | `outputs/demo/best_model.keras` | `outputs/best_model.keras` |
| **Face Detection** | Center crop (`--no-face-detect`) | OpenCV Haar cascade face extraction |
| **Purpose** | Pipeline verification & UI prototyping | Academic evaluation & benchmark |
| **Claimed Metrics** | Demonstrative only; NOT real-world metrics | Populated only after real training run |

---

## Real-Data CLI Workflow

Once Celeb-DF v2 videos are placed in `data/raw/`, run the full pipeline from the terminal:

### Step 1: Preprocess Dataset
```bash
python preprocess.py
```
*Customizable options:*
- `--raw-dir data/raw`: Path to raw data folder.
- `--processed-dir data/processed`: Destination for `.npy` clips and `meta.csv`.
- `--seq-len 10`: Frames per clip.
- `--img-size 128`: Frame resolution (128x128).
- `--clips-per-video 3`: Temporal segments extracted per video.
- `--face-margin 0.25`: Face bounding box margin expansion.
- `--test-list data/raw/List_of_testing_videos.txt`: Official test list path (if available).

### Step 2: Validate Dataset Integrity Before Training
```bash
python validate_dataset.py
```
Verifies clip counts, class distributions, tensor shapes `(10, 128, 128, 3)`, `uint8` datatypes, and guarantees **zero cross-split leakage**:
```text
train_video_ids ∩ val_video_ids = ∅
train_video_ids ∩ test_video_ids = ∅
val_video_ids ∩ test_video_ids = ∅
```

### Step 3: Train CNN-LSTM Model
```bash
python train.py
```
*Customizable options:*
- `--backbone light`: 4-block CNN (or `mobilenetv2`).
- `--epochs 15`: Maximum training epochs (with early stopping).
- `--batch-size 8`: Batch size.
- `--lr 0.001`: Initial learning rate.
- Automatically validates dataset integrity before starting training.

### Step 4: Evaluate on Held-Out Test Split
```bash
python evaluate.py
```
*Outputs generated in `outputs/`:*
- `metrics.json`: Clip-level and video-level Accuracy, Precision, Recall, F1, and ROC-AUC.
- `predictions.csv`: Predictions for all test clips and videos.
- `confusion_matrix.png`: Confusion matrices.
- `roc_curve.png`: Receiver operating characteristic curves.

### Step 5: Predict on a Single Video
```bash
python predict.py path/to/sample_video.mp4 --model-path outputs/best_model.keras
```

---

## Interactive Web Prototype (Streamlit)

Launch the interactive prototype UI locally at `http://localhost:8501`:
```bash
streamlit run app.py
```
The UI allows uploading videos, inspecting frame extraction and face detection, visualizing prediction confidence, and switching between the prototype model (`outputs/demo/best_model.keras`) and the real trained model (`outputs/best_model.keras`).

---

## Synthetic Demonstration & Testing

To run the self-contained synthetic pipeline (without Celeb-DF):
```bash
python demo.py
```

To run the automated test suite:
```bash
python -m unittest discover tests
```

---

## Results & Performance

> **Policy on Metric Integrity**: Every performance metric reported in this repository must come from actual execution of `evaluate.py` on real data. Synthetic demonstration metrics are **never** presented as model performance claims.

### Celeb-DF Evaluation Results

| Metric | Clip-Level Result | Video-Level Result |
|---|---|---|
| **Accuracy** | Not yet run on real data | Not yet run on real data |
| **Precision** | Not yet run on real data | Not yet run on real data |
| **Recall** | Not yet run on real data | Not yet run on real data |
| **F1-Score** | Not yet run on real data | Not yet run on real data |
| **ROC-AUC** | Not yet run on real data | Not yet run on real data |

*Note: Results will be populated once the model is trained and evaluated on the official Celeb-DF v2 dataset test split.*

---

## Known Limitations

1. **Haar Cascade Face Detection**: OpenCV Haar cascade detectors are fast and lightweight, but can fail on non-frontal faces, rapid head rotations, partial occlusions, or low-light frames.
2. **Identity Overlap Considerations**: In Celeb-DF, multiple real and fake videos may share the same celebrity identity. While video-level splitting prevents frame/clip leakage, strict identity-disjoint splitting (or the official Celeb-DF test protocol) is required for rigorous identity-independent evaluation.
3. **Official Test Protocol**: For academic benchmark comparisons, researchers should evaluate against the official Celeb-DF test list (`List_of_testing_videos.txt`) rather than random splits.
4. **Class Imbalance Sensitivity**: Real-world deepfake datasets often have significant class imbalances. Metrics like ROC-AUC, Precision, and Recall must always be evaluated alongside raw Accuracy.
5. **Cross-Generator Generalization**: Deepfake detectors trained on specific synthesis methods (e.g. DeepFake face-swap) often suffer performance degradation when applied to unseen generation architectures (e.g. diffusion models or modern neural reenactment).

---

## Future Improvements

- [ ] Incorporate MTCNN or RetinaFace for improved face detection robustness under extreme poses.
- [ ] Implement explicit celebrity identity grouping during dataset splitting.
- [x] Add support for the official Celeb-DF `List_of_testing_videos.txt` split file parser.
- [x] Add pre-training dataset validation CLI (`validate_dataset.py`).
- [ ] Explore Video Vision Transformers (ViViT / Timesformer) and 3D CNNs (I3D).
- [ ] Integrate Grad-CAM spatial/temporal saliency visualization to inspect manipulated facial regions.

---

## Project Structure

```text
deepfake-detector/
├── config.py             # Central dataclass configuration defaults
├── preprocess.py         # Video discovery, face extraction, video-level splitting
├── validate_dataset.py   # Pre-training dataset audit (leakage, shape, missing checks)
├── dataset.py            # tf.data pipeline with consistent clip augmentations
├── model.py              # CNN-LSTM architecture (light and MobileNetV2 backbones)
├── train.py              # Model training with pre-flight check and val_loss callbacks
├── evaluate.py           # Evaluation on held-out test split (metrics, plots, CSV)
├── predict.py            # CLI single-video inference and verdict generation
├── app.py                # Streamlit interactive web interface (prototype & real model)
├── demo.py               # End-to-end synthetic demo pipeline
├── requirements.txt      # Minimal project dependencies
├── .gitignore            # Clean git tracking preserving .gitkeep
├── README.md             # Documentation, setup guide, and honest results
│
├── tests/
│   └── test_pipeline.py  # Unit tests for pipeline, splitting, and validation
│
└── data/
    └── raw/
        ├── real/
        │   └── .gitkeep  # Destination for real videos (Celeb-real, YouTube-real)
        └── fake/
            └── .gitkeep  # Destination for fake videos (Celeb-synthesis)
```
