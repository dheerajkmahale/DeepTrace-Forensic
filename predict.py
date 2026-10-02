"""Single-video inference script for deepfake detection.

Preprocesses a supplied video using the identical pipeline as training/evaluation,
runs inference across all extracted clips with best_model.keras, and produces
an aggregated video-level verdict (REAL or FAKE) using a fixed threshold.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from preprocess import extract_clips_from_video, get_face_cascade


class PredictionResult:
    """Structured container for single-video deepfake prediction results."""

    def __init__(
        self,
        verdict: str,
        fake_probability: float,
        real_probability: float,
        clips_analyzed: int,
        clip_probabilities: List[float],
    ):
        self.verdict = verdict
        self.fake_probability = fake_probability
        self.real_probability = real_probability
        self.clips_analyzed = clips_analyzed
        self.clip_probabilities = clip_probabilities

    def __iter__(self):
        # Enables backward-compatible unpacking: (num_clips, p_fake, verdict)
        return iter((self.clips_analyzed, self.fake_probability, self.verdict))

    def __getitem__(self, idx):
        return (self.clips_analyzed, self.fake_probability, self.verdict)[idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "fake_probability": self.fake_probability,
            "real_probability": self.real_probability,
            "clips_analyzed": self.clips_analyzed,
            "clip_probabilities": self.clip_probabilities,
        }

    def __repr__(self) -> str:
        return (
            f"PredictionResult(verdict='{self.verdict}', "
            f"fake_prob={self.fake_probability:.4f}, "
            f"real_prob={self.real_probability:.4f}, "
            f"clips_analyzed={self.clips_analyzed})"
        )


def predict_video(
    video_path: str,
    model_path: str = "outputs/best_model.keras",
    threshold: float = 0.5,
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    no_face_detect: bool = False,
) -> PredictionResult:
    """Run deepfake inference on a single video file.

    Args:
        video_path: Path to target video (.mp4, etc.).
        model_path: Path to trained Keras model file.
        threshold: Classification decision threshold (default: 0.5).
        seq_len: Frames per clip.
        img_size: Frame resolution.
        clips_per_video: Number of clips to extract.
        face_margin: Margin expansion ratio for face bounding boxes.
        no_face_detect: If True, uses center crop (for synthetic demo data).

    Returns:
        PredictionResult containing verdict, fake_probability, real_probability,
        clips_analyzed, and clip_probabilities (also unpacks as (num_clips, p_fake, verdict)).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please ensure the model has been trained or run 'python demo.py' to generate a prototype model."
        )

    # Extract clips using identical preprocessing pipeline
    face_cascade = None if no_face_detect else get_face_cascade()
    clips = extract_clips_from_video(
        video_path=video_path,
        seq_len=seq_len,
        img_size=img_size,
        clips_per_video=clips_per_video,
        face_margin=face_margin,
        no_face_detect=no_face_detect,
        face_cascade=face_cascade,
    )

    if len(clips) == 0:
        return PredictionResult(
            verdict="UNKNOWN",
            fake_probability=0.0,
            real_probability=0.0,
            clips_analyzed=0,
            clip_probabilities=[],
        )

    clips_array = np.array(clips, dtype=np.uint8)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(clips_array, verbose=0).flatten()

    p_fake = float(np.mean(predictions))
    p_real = float(1.0 - p_fake)
    verdict = "FAKE" if p_fake >= threshold else "REAL"

    return PredictionResult(
        verdict=verdict,
        fake_probability=p_fake,
        real_probability=p_real,
        clips_analyzed=len(clips),
        clip_probabilities=[float(p) for p in predictions],
    )


def main():
    parser = argparse.ArgumentParser(description="Predict deepfake probability on a single video.")
    parser.add_argument("video", type=str, help="Path to video file for inference.")
    parser.add_argument("--model-path", type=str, default="outputs/best_model.keras", help="Path to model file.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification decision threshold (default: 0.5).")
    parser.add_argument("--seq-len", type=int, default=10, help="Frames per clip.")
    parser.add_argument("--img-size", type=int, default=128, help="Frame resolution (H=W).")
    parser.add_argument("--clips-per-video", type=int, default=3, help="Number of clips to sample.")
    parser.add_argument("--face-margin", type=float, default=0.25, help="Margin around detected faces.")
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Use center cropping instead of face detection. ONLY for synthetic demo data.",
    )

    args = parser.parse_args()

    result = predict_video(
        video_path=args.video,
        model_path=args.model_path,
        threshold=args.threshold,
        seq_len=args.seq_len,
        img_size=args.img_size,
        clips_per_video=args.clips_per_video,
        face_margin=args.face_margin,
        no_face_detect=args.no_face_detect,
    )

    print(f"Clips analysed: {result.clips_analyzed}")
    print(f"P(fake): {result.fake_probability:.4f}")
    print(f"Verdict: {result.verdict}")


if __name__ == "__main__":
    main()
