# Deepfake Detection & Mitigation System

## Problem
Detecting AI-generated deepfake videos to mitigate misinformation.

## Approach
Used a hybrid CNN+LSTM architecture:
- **CNN**: Extracts spatial features from individual video frames.
- **LSTM**: Captures temporal dependencies across frames to identify inconsistencies in facial expressions and movements.

## Dataset
Subset of the Celeb-DF dataset containing real and fake video clips.

## Results
- **Accuracy**: 88%
- **Precision**: 86%
- **Recall**: 89%
- **F1-Score**: 87.5%

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run preprocessing: `python preprocess.py`
3. Train model: `python train.py`

## Tech Stack
- Python
- TensorFlow
- OpenCV
- NumPy

## Project Structure
- `preprocess.py`: Face extraction pipeline.
- `model.py`: CNN+LSTM model definition.
- `train.py`: Training script.
- `requirements.txt`: Project dependencies.
