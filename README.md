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
Organize downloaded videos inside `data/raw/` as follows:

```
data/
└── raw/
    ├── real/
    │   ├── .gitkeep
    │   ├── Celeb-real_0001.mp4
    │   ├── YouTube-real_0001.mp4
    │   └── ...
    └── fake/
        ├── .gitkeep
        ├── Celeb-synthesis_0001.mp4
        └── ...
```

*Note: Both `Celeb-real` and `YouTube-real` videos represent real footage and must be placed in `data/raw/real/`. `Celeb-synthesis` videos represent deepfakes and belong in `data/raw/fake/`.*

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

## CLI Usage Guide

All stages are fully controllable via command-line arguments without modifying source code.

### 1. Preprocess Raw Videos
```bash
python preprocess.py --raw-dir data/raw --processed-dir data/processed --seq-len 10 --img-size 128 --clips-per-video 3 --val-fraction 0.15 --test-fraction 0.15 --seed 42
```
*Options:*
- `--no-face-detect`: Performs center square cropping. **Allowed ONLY for synthetic demonstration data.** Do not use for real Celeb-DF videos.

### 2. Train Model
```bash
python train.py --processed-dir data/processed --output-dir outputs --backbone light --epochs 15 --batch-size 8 --lr 0.001
```
*Backbones:*
- `--backbone light`: Custom 4-block CNN (fast, calibrated BatchNorm).
- `--backbone mobilenetv2 --pretrained`: MobileNetV2 with ImageNet pretrained weights.

### 3. Evaluate on Held-Out Test Set
```bash
python evaluate.py --model-path outputs/best_model.keras --processed-dir data/processed --output-dir outputs --threshold 0.5
```
Generates:
- `outputs/metrics.json` (distinct clip-level and video-level metrics)
- `outputs/predictions.csv` (detailed predictions per clip and video)
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

### 4. Run Single-Video Prediction
```bash
python predict.py path/to/sample_video.mp4 --model-path outputs/best_model.keras --threshold 0.5
```
Example Output:
```text
Clips analysed: 3
P(fake): 0.8924
Verdict: FAKE
```

### 5. Run Self-Contained Synthetic Demo
To verify the entire pipeline end-to-end on synthetic data without downloading external datasets:
```bash
python demo.py
```
This generates 60 synthetic videos, runs preprocessing, trains the network, evaluates test metrics, and executes single-sample inference.

### 6. Run Unit Tests
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
- [ ] Add support for the official Celeb-DF `List_of_testing_videos.txt` split file parser.
- [ ] Explore Video Vision Transformers (ViViT / Timesformer) and 3D CNNs (I3D).
- [ ] Integrate Grad-CAM spatial/temporal saliency visualization to inspect manipulated facial regions.

---

## Project Structure

```
deepfake-detector/
├── config.py             # Central dataclass configuration defaults
├── preprocess.py         # Video parsing, face extraction, video-level splitting
├── dataset.py            # tf.data pipeline with consistent clip augmentations
├── model.py              # CNN-LSTM architecture (light and MobileNetV2 backbones)
├── train.py              # Model training with val_loss callbacks and class weights
├── evaluate.py           # Evaluation on held-out test split (metrics, plots, CSV)
├── predict.py            # CLI single-video inference and verdict generation
├── demo.py               # End-to-end synthetic demo pipeline
├── requirements.txt      # Minimal project dependencies
├── .gitignore            # Clean git tracking preserving .gitkeep
├── README.md             # Documentation, setup guide, and honest results
│
├── tests/
│   └── test_pipeline.py  # Unit tests for model architecture & split integrity
│
└── data/
    └── raw/
        ├── real/
        │   └── .gitkeep  # Destination for real videos (Celeb-real, YouTube-real)
        └── fake/
            └── .gitkeep  # Destination for fake videos (Celeb-synthesis)
```
