# DeepTrace — AI-Powered Deepfake Forensic Analysis

> **Detect. Analyze. Verify.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://deeptrace-forensics.streamlit.app/)
[![Tests Passing](https://img.shields.io/badge/pytest-57%20passed-brightgreen.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DeepTrace** is an end-to-end, production-verified deep learning forensic analysis application for facial deepfake video detection. Built with a **TimeDistributed 4-Block CNN + Bidirectional LSTM + Temporal Attention** hybrid neural architecture, DeepTrace exposes both spatial blending artifacts and subtle inter-frame temporal inconsistencies. 

Evaluated against the official 518-video held-out test split of the authentic **Celeb-DF v2** benchmark, DeepTrace achieves **81.27% Video Accuracy**, **79.71% Balanced Accuracy**, **86.49% Precision**, **84.71% Fake Recall**, **85.59% F1-Score**, and **88.22% ROC-AUC** at a strictly locked validation threshold of $\tau^* = 0.39$.

🌐 **Live Production Application**: [https://deeptrace-forensics.streamlit.app/](https://deeptrace-forensics.streamlit.app/)  
📂 **GitHub Repository**: [https://github.com/dheerajkmahale/DeepTrace-Forensic](https://github.com/dheerajkmahale/DeepTrace-Forensic)

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Key Features](#2-key-features)
3. [Forensic Workspace & Live Interface](#3-forensic-workspace--live-interface)
4. [System Architecture](#4-system-architecture)
5. [Machine Learning Architecture](#5-machine-learning-architecture)
6. [Face Detection & Temporal Tracking Pipeline](#6-face-detection--temporal-tracking-pipeline)
7. [Temporal Attention Explainability](#7-temporal-attention-explainability)
8. [Calibration & Decision Threshold](#8-calibration--decision-threshold)
9. [Benchmark Evaluation: Celeb-DF v2](#9-benchmark-evaluation-celeb-df-v2)
10. [Official Performance Metrics & Visualizations](#10-official-performance-metrics--visualizations)
11. [Verified Reference Samples](#11-verified-reference-samples)
12. [Technology Stack](#12-technology-stack)
13. [Project Structure](#13-project-structure)
14. [How to Run Locally](#14-how-to-run-locally)
15. [Safe Production Model Delivery](#15-safe-production-model-delivery)
16. [Limitations & Technical Disclosures](#16-limitations--technical-disclosures)
17. [Responsible Use Notice](#17-responsible-use-notice)
18. [License & Citation](#18-license--citation)

---

## 1. Project Overview & Problem Statement

Modern facial synthesis engines—such as deep generative autoencoders, GAN face-swappers, and diffusion pipelines—generate photorealistic manipulations that easily deceive the human eye. While individual synthesized frames can appear plausible, they consistently leave micro-level spatial boundaries (e.g., hair-skin blending seams, jawline color discrepancies) and temporal incoherencies across successive frames (e.g., temporal jitter, flickering illumination, landmark trajectory jitter).

Standard single-frame image classifiers fail on video deepfakes because they cannot observe inter-frame motion or temporal dynamics. Conversely, uncalibrated video models suffer from extreme majority-class collapse when trained on heavily skewed datasets.

**DeepTrace solves this through a leak-free, spatio-temporal forensic pipeline**:
- Ingests raw video containers, standardizes temporal sampling across uniform segments, and isolates facial regions with bounding-box margins.
- Employs a 4-block convolutional network to extract spatial boundary features, followed by a Bidirectional LSTM to model temporal forward-backward sequence continuity.
- Pools recurrent representations using a custom **Temporal Attention** layer, exposing exact frame-level attention weights ($\alpha_t$) for transparent forensic auditability.
- Employs validation-isolated threshold calibration ($\tau^* = 0.3900$) and class-balanced dynamic batching to guarantee unbiased detection.
- Delivers a production-grade, browser-safe Streamlit web application with automated WebM preview transcoding and cryptographically verified model artifact delivery.

---

## 2. Key Features

- **Spatio-Temporal Hybrid Analysis**: Joint spatial feature extraction (CNN) and bidirectional temporal sequence modeling (BiLSTM) over 10-frame sequences.
- **Explainable Temporal Attention**: Computes normalized attention weights ($\alpha_t$) identifying the exact peak frames contributing most heavily to the classification verdict.
- **Calibrated Decision Boundary**: Locked threshold $\tau^* = 0.39$ calibrated on validation data to maximize balanced accuracy and eliminate arbitrary 0.5 classification bias.
- **Browser-Safe Media Transcoding Engine**: Converts incompatible raw container streams (e.g., `mp4v`, `FMP4`) to browser-safe WebM for instant client playback while preserving original bitstreams for tensor extraction.
- **Dynamic Class-Balanced Sampling**: Enforces exact 50% Real / 50% Fake gradient contributions per batch, preventing majority-class collapse on Celeb-DF v2's 8.5:1 imbalance.
- **Zero-Leakage Benchmark Protocol**: Guarantees zero subject or video leakage by strictly isolating Celeb-DF v2's official 518-video test split (`List_of_testing_videos.txt`).
- **Cryptographic Model Delivery**: Automates SHA-256 verified runtime delivery of production weights from GitHub Release assets without polluting Git history with binary blobs.
- **Forensic Audit & Export**: Session audit trail logging, confidence distribution gauges, and downloadable forensic reports in both JSON and HTML formats.
- **Robust Automated Test Suite**: 57 automated tests covering data processing, model layers, attention arithmetic, serialization integrity, and UI safety.

---

## 3. Forensic Workspace & Live Interface

The production application is deployed live on Streamlit Cloud at **[deeptrace-forensics.streamlit.app](https://deeptrace-forensics.streamlit.app/)**. The interface provides a modular forensic investigation suite:

| Workspace Module | Capabilities & Workflow |
| :--- | :--- |
| **🔍 Forensic Analysis** | Ingests video containers (MP4, MOV, AVI, MKV), initiates real-time OpenCV Haar face tracking, extracts 3 equidistant temporal clips (10 frames each), and renders the calibrated binary verdict with dual confidence gauges. |
| **⏱️ Temporal Attention** | Generates dynamic attention curves ($\alpha_t$) across all frames, highlighting exact high-discrepancy temporal moments where the network identified facial synthesis anomalies. |
| **🧩 Keyframe Evidence** | Interactive frame inspector displaying cropped, margin-expanded facial crops alongside per-frame attention contributions and detection bounding boxes. |
| **📊 Confidence Calibration** | Real-time gauge visualizing the calibrated decision boundary ($\tau^* = 0.39$), distance-to-boundary metrics, and inconclusive margin alerts ($0.34 \le P \le 0.44$). |
| **📜 Forensic Audit Report** | Produces timestamped session audit trails and one-click downloadable forensic intelligence reports in structured JSON and styled HTML formats. |
| **🔬 Architecture Inspector** | Full neural topology breakdown detailing parameter counts, receptive fields, and layer activations from TimeDistributed Conv4 through BiLSTM to Temporal Attention. |

> **Live Application**: [Launch DeepTrace Forensics](https://deeptrace-forensics.streamlit.app/)  
> *(Live demo loads verified production weights with sub-second inference latency on standard cloud CPU compute)*

---

## 4. System Architecture

```mermaid
graph TD
    A[Input Video Container: MP4 / MOV / AVI / MKV] --> B[Temporal Clip Segmentation: 3 Uniform Segments]
    A --> C[Browser-Safe WebM Transcoder]
    C --> D[HTML5 Forensic Media Player & Playback]
    
    B --> E[OpenCV Haar Cascade Face Detector + 25% Margin Crop]
    E --> F["Normalized Tensor: (10, 128, 128, 3) Scaled to [-1, 1]"]
    
    F --> G[TimeDistributed 4-Block CNN Feature Extractor]
    G --> H[Global Average Pooling: (10, 128)]
    
    H --> I[Bidirectional LSTM: 128 Units Each Direction]
    I --> J["Recurrent Forward-Backward Sequence: (10, 256)"]
    
    J --> K[Temporal Attention Layer: 64 Units]
    K --> L["Attention Context Vector (256) + Frame Attention Weights alpha_t"]
    
    L --> M[Dense Layer 128 Units + ReLU + Dropout 0.3]
    M --> N[Sigmoid Output Head: Clip Score P_fake]
    
    N --> O[Mean Pooling Aggregation across Video Clips]
    O --> P{Calibrated Decision Threshold: tau = 0.39}
    
    P -->|P >= 0.39| Q[Verdict: MANIPULATED / FAKE]
    P -->|P < 0.39| R[Verdict: AUTHENTIC / REAL]
```

---

## 5. Machine Learning Architecture

DeepTrace uses a custom neural architecture combining spatial convolution, bidirectional recurrent memory, and parametric attention:

| Layer Stage | Specification | Output Shape | Parameters | Functional Role |
| :--- | :--- | :--- | :---: | :--- |
| **Input** | `Input(shape=(10, 128, 128, 3))` | `(None, 10, 128, 128, 3)` | 0 | 10 uniformly sampled RGB facial frames |
| **Rescaling** | `Rescaling(scale=1/127.5, offset=-1.0)` | `(None, 10, 128, 128, 3)` | 0 | Fast numerical scaling to $[-1.0, 1.0]$ |
| **Conv Block 1** | `Conv2D(16, 3x3)` + `BN(0.9)` + `ReLU` + `MaxPool(2x2)` | `(None, 10, 64, 64, 16)` | 496 | Low-level edge and boundary filters |
| **Conv Block 2** | `Conv2D(32, 3x3)` + `BN(0.9)` + `ReLU` + `MaxPool(2x2)` | `(None, 10, 32, 32, 32)` | 4,768 | Intermediate texture and color transitions |
| **Conv Block 3** | `Conv2D(64, 3x3)` + `BN(0.9)` + `ReLU` + `MaxPool(2x2)` | `(None, 10, 16, 16, 64)` | 18,752 | Facial component alignment representations |
| **Conv Block 4** | `Conv2D(128, 3x3)` + `BN(0.9)` + `ReLU` + `MaxPool(2x2)` | `(None, 10, 8, 8, 128)` | 74,368 | High-level synthesis artifact abstractions |
| **Spatial GAP** | `GlobalAveragePooling2D()` | `(None, 10, 128)` | 0 | Spatially condensed frame embedding |
| **Dropout 1** | `Dropout(rate=0.4)` | `(None, 10, 128)` | 0 | Feature regularization |
| **BiLSTM** | `Bidirectional(LSTM(128, return_sequences=True))` | `(None, 10, 256)` | 263,168 | Models forward and reverse temporal sequence dependencies |
| **Temporal Attention** | `TemporalAttention(units=64)` | `(None, 256)` | 16,513 | Learns frame importance coefficients $\alpha_t$ and pools context |
| **Dropout 2** | `Dropout(rate=0.4)` | `(None, 256)` | 0 | Regularization before classification head |
| **Dense Head** | `Dense(128, activation="relu")` | `(None, 128)` | 32,896 | Non-linear forensic classification mapping |
| **Dropout 3** | `Dropout(rate=0.3)` | `(None, 128)` | 0 | Classifier regularization |
| **Classifier** | `Dense(1, activation="sigmoid")` | `(None, 1)` | 129 | Single scalar manipulation probability $P(\text{Fake})$ |

- **Total Parameters**: 411,105 (410,625 trainable)
- **Model Checkpoint Size**: 4.07 MB (`outputs/best_model.keras`)
- **CPU Inference Latency**: ~0.18s per clip on standard compute

---

## 6. Face Detection & Temporal Tracking Pipeline

1. **Uniform Temporal Windowing**: The video duration is segmented into 3 equidistant temporal blocks. Within each block, 10 frames are uniformly indexed:
   $$t_i = \left\lfloor i \cdot \frac{N_{\text{frames}}}{T} \right\rfloor, \quad i \in \{0, 1, \dots, T-1\}$$
2. **Haar Frontal Cascade**: OpenCV's `haarcascade_frontalface_default.xml` scans each sampled frame for facial landmarks.
3. **25% Boundary Expansion**: When a bounding box $(x, y, w, h)$ is located, a 25% proportional margin is appended:
   $$x' = \max\left(0, x - 0.25w\right), \quad y' = \max\left(0, y - 0.25h\right)$$
   $$w' = w + 0.50w, \quad h' = h + 0.50h$$
   This ensures jawlines, hairlines, and blending borders where face-swapping seam artifacts cluster are preserved.
4. **Temporal Box Smoothing & Interpolation**: If a face is momentarily obscured or undetected on an intermediate frame, the pipeline reuses the nearest verified bounding coordinates, falling back to a center crop only when no face is found across the clip.

---

## 7. Temporal Attention Explainability

Standard temporal pooling (e.g., Global Average Pooling over time) dilutes sudden, localized manipulation artifacts. DeepTrace implements a trainable **Temporal Attention** mechanism:

1. **Relevance Scoring**: For each temporal recurrent state $h_t \in \mathbb{R}^{256}$ ($t = 1, \dots, 10$):
   $$u_t = \tanh(W_a h_t + b_a), \quad W_a \in \mathbb{R}^{64 \times 256}, \ b_a \in \mathbb{R}^{64}$$
2. **Normalized Frame Attention**:
   $$\alpha_t = \frac{\exp(v_a^\top u_t)}{\sum_{j=1}^{10} \exp(v_a^\top u_j)}, \quad v_a \in \mathbb{R}^{64}, \quad \sum_{t=1}^{10} \alpha_t = 1.0$$
3. **Context Pooling**:
   $$c = \sum_{t=1}^{10} \alpha_t h_t, \quad c \in \mathbb{R}^{256}$$

The resulting coefficients $\alpha_t$ represent each frame's contribution to the classification decision. In the DeepTrace user interface, the top attention frames are ranked and rendered as visual evidence cards with exact timestamps and percentage distributions.

---

## 8. Calibration & Decision Threshold

In real-world deepfake forensic analysis, arbitrary default thresholds (such as 0.5) produce sub-optimal tradeoffs, particularly on class-imbalanced data. 

DeepTrace performs **Validation-Isolated Threshold Calibration**:
- Model training and threshold selection are completely decoupled from the test set.
- Threshold candidates $\tau \in [0.10, 0.90]$ were scanned on the isolated validation set.
- Threshold **$\tau^* = 0.3900$** was chosen to balance sensitivity and specificity, yielding:
  - Optimal balance between Real Recall and Fake Detection Rate.
  - A calibrated **Inconclusive Band** between $0.34$ and $0.44$, alerting analysts when model predictions fall near the decision boundary.

```text
Decision Rules:
- P(Fake) < 0.34       --> Confident AUTHENTIC (REAL)
- 0.34 <= P <= 0.44  --> INCONCLUSIVE (Low-confidence zone near boundary)
- P(Fake) > 0.44       --> Confident MANIPULATED (FAKE)
- Strict binary cutoff: tau* = 0.3900
```

---

## 9. Benchmark Evaluation: Celeb-DF v2

DeepTrace was trained and evaluated on **Celeb-DF v2**, one of the most challenging, high-quality public deepfake benchmarks:

| Dataset Split | Real Videos | Fake Videos | Total Videos | Real Clips | Fake Clips | Total Clips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Training** | 604 | 4,504 | 5,108 | 1,812 | 13,511 | 15,323 |
| **Validation** | 107 | 795 | 902 | 321 | 2,385 | 2,706 |
| **Official Held-Out Test** | 178 | 340 | **518** | 534 | 1,020 | **1,554** |
| **Total Benchmark** | **889** | **5,639** | **6,528** | **2,667** | **16,916** | **19,583** |

*Protocol Isolation Guarantee*: The 518 test videos defined in `List_of_testing_videos.txt` were completely held out and never used during training, hyperparameter optimization, or threshold calibration.

---

## 10. Official Performance Metrics & Visualizations

Measured on 100% of the official held-out Celeb-DF v2 test split (518 distinct videos, 1,554 clips) at locked threshold $\tau^* = 0.39$:

| Metric | Measured Test Result | Interpretation |
| :--- | :---: | :--- |
| **Video Accuracy** | **81.27%** | 421 / 518 videos correctly classified |
| **Balanced Accuracy** | **79.71%** | Unbiased mean of real recall and fake recall |
| **Precision** | **86.49%** | Accuracy of positive (fake) predictions |
| **Fake Recall (Sensitivity)** | **84.71%** | 288 / 340 manipulated videos caught |
| **Real Recall (Specificity)** | **74.72%** | 133 / 178 authentic videos correctly verified |
| **F1-Score** | **85.59%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **88.22%** | Area under the Receiver Operating Characteristic curve |

### Confusion Matrix (518 Test Videos)
$$\begin{pmatrix} \text{TN: Authentic Verified} & \text{FP: False Alarm} \\ \text{FN: Missed Manipulation} & \text{TP: Deepfake Detected} \end{pmatrix} = \begin{pmatrix} 133 & 45 \\ 52 & 288 \end{pmatrix}$$

### Official Benchmark Curves
<p align="center">
  <img src="docs/assets/roc_curve.png" width="48%" alt="DeepTrace Test ROC Curve (AUC = 88.22%)" />
  <img src="docs/assets/confusion_matrix.png" width="48%" alt="DeepTrace Held-Out Test Confusion Matrix" />
</p>

---

## 11. Verified Reference Samples

Verified through both local CLI inference and the live public Streamlit Cloud deployment:

### 1. Authentic Human Video (`Celeb-real/id0_0000.mp4`)
```text
Video Source: Celeb-DF v2 Real Benchmark
Target Resolution: 942x500 @ 30.0 FPS (15.6s)
Face Detection: 30 / 30 frames (100.0% Haar face tracking)
Clips Analyzed: 3 clips (10 frames each)
Calculated Verdict: AUTHENTIC (REAL)
Manipulation Probability: 7.33% (0.0733)
Authentic Probability: 92.67% (0.9267)
Peak Attention Frames: Frame #155 (14.73%), Frame #0 (12.93%), Frame #137 (11.61%)
Decision: P(Fake) << 0.39 Cutoff --> Confident Authentic
```

### 2. Synthesized Deepfake Video (`Celeb-synthesis/id0_id16_0000.mp4`)
```text
Video Source: Celeb-DF v2 Synthesis Benchmark
Target Resolution: 944x500 @ 30.0 FPS (15.6s)
Face Detection: 30 / 30 frames (100.0% Haar face tracking)
Clips Analyzed: 3 clips (10 frames each)
Calculated Verdict: MANIPULATED (FAKE)
Manipulation Probability: 79.25% (0.7925)
Authentic Probability: 20.75% (0.2075)
Per-Clip Probabilities: Clip 1: 78.6%, Clip 2: 73.7%, Clip 3: 85.5%
Peak Attention Frames: Frame #188 (16.19%), Frame #204 (12.21%), Frame #117 (10.27%)
Decision: P(Fake) >> 0.39 Cutoff --> Confident Manipulation
```

---

## 12. Technology Stack

- **Core Deep Learning**: TensorFlow 2.16+, Keras 3 (TimeDistributed, BiLSTM, Custom Layers)
- **Computer Vision**: OpenCV (Headless) 4.8+ (video container reading, Haar cascade face tracking, frame normalization)
- **Web Interface**: Streamlit 1.30+ (Reactive UI, media streaming, session audit tracking)
- **Data Engineering**: NumPy 1.24+, Pandas 2.0+, Scikit-Learn 1.3+
- **Data Visualization**: Plotly 5.0+, Matplotlib 3.7+
- **Quality Assurance**: Pytest 7.0+ (57 unit, pipeline, and UI AppTest integration tests)
- **Runtime Environment**: Python 3.11 Linux container (Streamlit Community Cloud)

---

## 13. Project Structure

```text
DeepTrace-Forensic/
├── app.py                     # Streamlit forensic workspace (6 interactive tabs)
├── config.py                  # Pipeline configuration, threshold constants, and model fetcher
├── dataset.py                 # Balanced batch generator with online augmentations
├── demo.py                    # Procedural synthetic prototype generator
├── evaluate.py                # Official held-out test split evaluation suite
├── model.py                   # TimeDistributed CNN + BiLSTM + TemporalAttention layers
├── predict.py                 # Standalone single-video CLI inference entrypoint
├── preprocess.py              # Haar face extraction, segmenting, and metadata generator
├── theme.py                   # Dark SaaS forensic color tokens and styling constants
├── train.py                   # Training loop with validation callbacks
├── train_v2.py                # V2 production model training script
├── ui_helpers.py              # Media transcoding, chart builders, and report generators
├── ui_styles.py               # Custom CSS design tokens and container styles
├── validate_dataset.py        # Dataset partition audits and leakage verification
├── requirements.txt           # Verified production dependencies
├── pytest.ini                 # Pytest test discovery configuration
├── docs/                      # Documentation figures and benchmark comparisons
├── tests/
│   ├── test_pipeline.py       # Pipeline, model shape, serialization, and leakage tests
│   └── test_ui.py             # UI rendering, report generation, and AppTest tests
└── outputs/                   # (Git-ignored) Model artifacts and training logs
    └── best_model.keras       # Verified production model (4.07 MB)
```

---

## 14. How to Run Locally

### Prerequisites
- Python 3.10 to 3.12
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/dheerajkmahale/DeepTrace-Forensic.git
cd DeepTrace-Forensic

# 2. Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Launch the Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. On startup, DeepTrace verifies or downloads the production model checkpoint and initializes the forensic workspace.

### Run Single-Video CLI Inference
```bash
python predict.py path/to/video.mp4 --threshold 0.39
```

### Run Automated Test Suite
```bash
pytest -q
```
*Expected Result*: `57 passed`

---

## 15. Safe Production Model Delivery

To maintain a clean Git repository history, binary model checkpoints (`*.keras`) and large video files are excluded from Git tracking via `.gitignore`. 

On application startup, [`config.py`](config.py) executes automated artifact delivery:
1. **Local Integrity Check**: Checks for `outputs/best_model.keras`. If present, computes its SHA-256 hash.
2. **Automated Download**: If missing or corrupted, downloads the production model from the official GitHub Release asset (`v2.0.0-model`).
3. **Cryptographic Validation**: Compares the downloaded file against the locked SHA-256 digest:
   ```text
   20b2d5dcf6300f4b001aebc7f73142e29ef6e06bbe55610062dea23425da8689
   ```
4. **Enforced Security**: Rejects corrupted files and halts with a clear `MODEL UNAVAILABLE` notification if the checksum fails. Never silently substitutes unverified weights.

---

## 16. Limitations & Technical Disclosures

- **High Social Media Compression**: Severe lossy re-compression (e.g., heavily forwarded WhatsApp/X clips) removes subtle high-frequency edge gradients, which can increase inconclusive classifications.
- **Occlusions & Extreme Poses**: Severe profile angles (>60°), dark lighting, or heavy facial occlusions (masks, large sunglasses) degrade Haar cascade face tracking confidence.
- **Unseen Generative Heads**: The model was trained on GAN and autoencoder swap architectures; zero-shot generalization to newer diffusion-based generative models (e.g., Sora, LivePortrait) may yield lower confidence.
- **Resolution Floor**: Input crops are scaled to $128\times128$ pixels; spatial manipulation signals smaller than this resolution cannot be resolved.

---

## 17. Responsible Use Notice

DeepTrace is engineered as an **AI-assisted forensic decision-support tool** for researchers, journalism integrity analysts, and cybersecurity teams.
- **Probabilistic Estimations**: All outputs represent statistical likelihoods derived from learned spatio-temporal representations.
- **Not Judicial Evidence**: DeepTrace verdicts should not be used as sole, definitive proof of authenticity or manipulation in legal or judicial proceedings without corroborating forensic, metadata, and cryptographic evidence.
- **Human-in-the-Loop**: Automated forensic predictions should always be reviewed alongside contextual provenance by a qualified analyst.

---

## 18. License & Citation

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Citation
```bibtex
@misc{mahale2026deeptrace,
  author = {Dheeraj K Mahale},
  title = {DeepTrace: AI-Powered Deepfake Forensic Analysis},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/dheerajkmahale/DeepTrace-Forensic}}
}
```

*Celeb-DF v2 Benchmark Reference*:
> Yuezun Li, Peng Sun, Qi Shen, and Siwei Lyu. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
