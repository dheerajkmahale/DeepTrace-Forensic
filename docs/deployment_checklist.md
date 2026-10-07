# DeepTrace — Deployment Verification Checklist

This checklist documents the deployment hardening and verification status for **DeepTrace (AI-Powered Deepfake Forensic Analysis)**.

---

## 1. Repository & Code Readiness
- [x] **Primary Entry Point**: [`app.py`](../app.py) confirmed as the application root.
- [x] **No Machine-Specific Paths**: All absolute Windows user paths (`C:\Users\...`) eliminated; relative project paths and environment variables used exclusively.
- [x] **Clean Imports**: All dependencies imported in `app.py`, `model.py`, `ui_helpers.py`, and `predict.py` match entries in `requirements.txt`.
- [x] **Model Checkpoint**: Production V2 model (`outputs/best_model.keras`) and synthetic demo checkpoint (`outputs/demo/best_model.keras`) verified with correct input tensor shape `(None, 10, 128, 128, 3)`.
- [x] **Calibrated Decision Threshold**: Optimal validation-locked threshold $\tau^* = 0.3900$ stored in `outputs/final_threshold.json` and loaded dynamically.

---

## 2. Dependencies & System Packages
- [x] **Python Dependencies (`requirements.txt`)**:
  - `tensorflow>=2.16.0`
  - `opencv-python>=4.8.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `scikit-learn>=1.3.0`
  - `matplotlib>=3.7.0`
  - `pytest>=7.0.0`
  - `streamlit>=1.30.0`
  - `plotly>=5.0.0`
- [x] **Debian System Packages (`packages.txt`)**:
  - `libgl1`
  - `libglib2.0-0`
  - `ffmpeg`
  *(Ensures headless Linux environments like Streamlit Community Cloud and Debian Docker containers execute OpenCV and video decoding without missing shared library errors).*

---

## 3. Streamlit Configuration
- [x] **Config File**: [`.streamlit/config.toml`](../.streamlit/config.toml) verified.
- [x] **Theme Consistency**: Ultraviolet Forensics dark palette tokens mapped to `[theme]` with zero black/blue/green violations.
- [x] **Server Settings**: `headless = true`, `maxUploadSize = 50`.
- [x] **Native Control Preservation**: Native Streamlit `Deploy`, `Rerun`, and `Settings` UI elements are untouched and fully functional.

---

## 4. Secrets & Security Audit
- [x] **Zero Hardcoded Secrets**: Scanned repository for API keys, bearer tokens, passwords, private endpoints, and credentials.
- [x] **Zero Authentication Overhead**: Fully accessible without login, user database, paid APIs, or third-party authentication services.

---

## 5. Git Safety & Large Artifact Isolation
- [x] **Binary & Dataset Isolation**: Verified via `git ls-files -- "*.mp4" "*.npy" "*.keras" "*.h5" "*.zip"` — **0 prohibited large files tracked**.
- [x] **Comprehensive `.gitignore`**:
  - Raw and processed datasets (`data/raw/`, `data/processed/`, `data/processed_v2/`)
  - Video files (`*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.webm`)
  - Compressed archives (`*.zip`, `*.tar.gz`, `*.tar`, `*.7z`)
  - Array dumps (`*.npy`, `*.npz`)
  - Temporary files, session caches, and `.log` outputs

---

## 6. Test Suite & Validation
- [x] **Automated Tests**: **56 passed / 0 failed** via `pytest -q`.
- [x] **Palette Enforcement**: Zero forbidden colors across all UI stylesheet assets.
- [x] **Inference Sanity**:
  - Authentic Fake (`id0_id16_0000.mp4`): Correctly predicted as **FAKE** ($79.25\%$ fake probability $\ge 0.39$).
  - Authentic Real (`id0_0000.mp4`): Correctly predicted as **REAL** ($7.33\%$ fake probability $< 0.39$).

---

## 7. Streamlit Community Cloud Deployment Steps
1. **GitHub Repository**: Push verified clean repository to GitHub.
2. **Login**: Access [share.streamlit.io](https://share.streamlit.io).
3. **New App**: Click **Deploy an app**.
4. **Repository & Branch**: Select your repository (e.g. `dheerajkmahale/deepfake-detector`) and default branch (`main`).
5. **Main file path**: Specify `app.py`.
6. **Advanced settings**: No additional environment variables or secrets required.
7. **Deploy**: Click **Deploy!**. Streamlit Cloud will install `packages.txt`, install `requirements.txt`, and launch DeepTrace.
