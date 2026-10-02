"""Single-video inference script for deepfake detection.

Preprocesses a supplied video using the identical pipeline as training/evaluation,
runs inference across all extracted clips with best_model.keras, and produces
an aggregated video-level verdict (REAL or FAKE) using a fixed threshold.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from preprocess import extract_clips_from_video, get_face_cascade


def predict_video(
    video_path: str,
    model_path: str = "outputs/best_model.keras",
    threshold: float = 0.5,
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    no_face_detect: bool = False,
) -> Tuple[int, float, str]:
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
        Tuple of (num_clips_analyzed, p_fake, verdict).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

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
        print(f"[WARNING] No valid clips could be extracted from {video_path}.")
        return 0, 0.0, "UNKNOWN"

    clips_array = np.array(clips, dtype=np.uint8)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(clips_array, verbose=0).flatten()

    p_fake = float(np.mean(predictions))
    verdict = "FAKE" if p_fake >= threshold else "REAL"

    return len(clips), p_fake, verdict


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

    num_clips, p_fake, verdict = predict_video(
        video_path=args.video,
        model_path=args.model_path,
        threshold=args.threshold,
        seq_len=args.seq_len,
        img_size=args.img_size,
        clips_per_video=args.clips_per_video,
        face_margin=args.face_margin,
        no_face_detect=args.no_face_detect,
    )

    print(f"Clips analysed: {num_clips}")
    print(f"P(fake): {p_fake:.4f}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
