# DeepTrace — Final Project Status Report

### PROJECT
**DeepTrace — AI-Powered Deepfake Forensic Analysis**

---

### MODEL ARCHITECTURE
- **Architecture**: TimeDistributed 4-Block Conv2D $\rightarrow$ Bidirectional LSTM (128 units each direction) $\rightarrow$ Temporal Attention (64 units) $\rightarrow$ Dense Representation (128 units) $\rightarrow$ Sigmoid Classifier
- **Input Dimensions**: 10 RGB frames $\times 128 \times 128$
- **Total Parameters**: 411,105 (410,625 trainable)
- **Production Checkpoint**: `outputs/best_model.keras`
- **Prototype Checkpoint**: `outputs/demo/best_model.keras`

---

### DATASET & PROTOCOL
- **Dataset**: Authentic Celeb-DF v2 (889 Real videos, 5,639 Fake videos)
- **Training Set**: 15,323 clips across 5,108 videos (balanced dynamic batch sampling: 8 Real / 8 Fake)
- **Validation Set**: 2,706 clips across 902 videos (unmodified distribution)
- **Official Held-Out Test Set**: 1,554 clips across 518 videos (`List_of_testing_videos.txt`)
- **Leakage Isolation**: Strict zero-leakage ($\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). Test set evaluated only once after validation threshold locking.

---

### CALIBRATED DECISION THRESHOLD
- **Locked Threshold**: $\tau^* = \mathbf{0.3900}$ (calibrated strictly on validation split to maximize Balanced Accuracy and F1)
- **Inconclusive Band**: $[0.40, 0.60]$ (user configurable)

---

### OFFICIAL HELD-OUT BENCHMARK RESULTS (518 VIDEOS / 1,554 CLIPS)
- **Video Accuracy**: **81.27%** (421 / 518 videos correctly classified)
- **Video ROC-AUC**: **88.22%**
- **Video F1-Score**: **85.59%**
- **Fake Video Recall**: **84.71%** (288 / 340 manipulated videos detected)
- **Real Video Recall**: **74.72%** (133 / 178 authentic videos verified)
- **Video Balanced Accuracy**: **79.71%**
- **Video Precision**: **86.49%**

---

### TEST SUITE VERIFICATION
- **Test Results**: **56 passed, 0 failed** via `pytest -q`
- **Suites**:
  - `TestPipeline`: Input/output shapes, tensor rescaling, model compilation, serialization, data leakage prevention, evaluate logic
  - `TestTemporalAttention`: Layer weights, context vector computation, softmax normalization, zero-padding stability
  - `TestUI`: Streamlit rendering, tab structure, report serialization (JSON/HTML), provenance badges, honesty controls
  - `TestForbiddenColors`: Zero pure black, blue, or green hex/named colors across all UI stylesheets

---

### USER INTERFACE & VERIFIED INFERENCE
- **Interface**: Production-ready Streamlit interface with dark cinematic visual aesthetic.
- **Local URL**: `http://localhost:8501`
- **Live Video Verifications**:
  - **Authentic Fake Sample** (`id0_id16_0000.mp4`):
    - Predicted Verdict: **FAKE**
    - Fake Probability: **79.25%** (Threshold: 0.39)
    - Face Detection: Detected (OpenCV Haar Cascade)
    - Top Attention Frame: Frame #0 ($16.19\%$ attention weight)
  - **Authentic Real Sample** (`id0_0000.mp4`):
    - Predicted Verdict: **REAL**
    - Fake Probability: **7.33%** (Threshold: 0.39)
    - Face Detection: Detected (OpenCV Haar Cascade)
    - Top Attention Frame: Frame #8 ($16.19\%$ attention weight)

---

### DEPLOYMENT READINESS
- **Primary Target**: Streamlit Community Cloud ([share.streamlit.io](https://share.streamlit.io))
- **Entry Point**: `app.py`
- **Dependencies**: `requirements.txt` (Python packages) + `packages.txt` (Debian system libraries: `libgl1`, `libglib2.0-0`, `ffmpeg`)
- **Zero Authentication**: No login barriers, no external paid API dependencies, immediate out-of-the-box functionality upon deployment.
- **Git Safety**: 0 dataset videos, `.zip` archives, or `.npy` arrays tracked in Git.

---

### DOCUMENTED LIMITATIONS
1. **Resolution Bounds**: Faces are normalized to $128 \times 128$ RGB. Minute microscopic artifacts beneath this scale cannot be detected.
2. **Haar Face Detection Bounds**: Extreme profile yaw, heavy occlusions (e.g. masks, dark sunglasses), or poor illumination can degrade face tracking, triggering center-crop fallback.
3. **Cross-Generator Domain Shift**: While performing strongly on Celeb-DF v2 facial autoencoder manipulations, detection confidence may vary on unseen modern diffusion-based generation architectures.
4. **Probabilistic Outputs**: Intended as an AI-assisted forensic decision-support instrument, not definitive legal proof.
