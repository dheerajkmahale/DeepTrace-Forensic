"""End-to-end synthetic demonstration pipeline for deepfake detector.

This script:
1. Generates ~60 tiny synthetic videos (30 real, 30 fake).
   - REAL: Smooth textured moving blob across frames.
   - FAKE: Same blob with temporal flicker, jitter, and noise.
2. Runs video-level stratified train/val/test preprocessing.
3. Trains the CNN-LSTM model on synthetic clips.
4. Evaluates the model strictly on the held-out test split.
5. Runs predict.py inference on a sample synthetic video.
6. Saves evaluation artifacts to outputs/demo/:
   - metrics.json
   - predictions.csv
   - confusion_matrix.png
   - roc_curve.png
7. Prints a prominent warning that demo metrics are NOT real deepfake results.
"""

import argparse
import os
import shutil
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

from evaluate import evaluate
from predict import predict_video
from preprocess import run_preprocessing
from train import train_model


def generate_single_video(
    output_path: str,
    is_fake: bool,
    num_frames: int = 30,
    width: int = 128,
    height: int = 128,
    seed: int = 42,
) -> None:
    """Generate a single tiny synthetic video file.

    Real videos:
    - Smooth textured moving blob with constant illumination and trajectory.

    Fake videos:
    - Same basic blob, but subject to per-frame brightness flickering,
      spatial jitter, and high-frequency noise.
    """
    rng = np.random.RandomState(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 15.0, (width, height))

    # Base trajectory endpoints
    start_x = rng.randint(35, 50)
    end_x = rng.randint(75, 95)
    start_y = rng.randint(35, 50)
    end_y = rng.randint(75, 95)
    radius = rng.randint(20, 26)

    # Base blob color (BGR)
    base_color = (
        rng.randint(60, 220),
        rng.randint(60, 220),
        rng.randint(60, 220),
    )

    for i in range(num_frames):
        alpha = i / max(1, num_frames - 1)
        cx = int(start_x + alpha * (end_x - start_x))
        cy = int(start_y + alpha * (end_y - start_y))

        # Create textured background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        # Add subtle background gradient
        gradient = np.linspace(20, 60, height, dtype=np.uint8)[:, None, None]
        frame = np.clip(frame.astype(np.int16) + gradient, 0, 255).astype(np.uint8)

        blob_color = list(base_color)

        if is_fake:
            # 1. Spatial jitter (simulate poor face alignment / warping)
            jitter_x = rng.randint(-3, 4)
            jitter_y = rng.randint(-3, 4)
            cx = max(radius + 2, min(width - radius - 2, cx + jitter_x))
            cy = max(radius + 2, min(height - radius - 2, cy + jitter_y))

            # 2. Per-frame brightness flicker (simulate temporal luminance inconsistency)
            flicker = rng.randint(-45, 46)
            blob_color = [max(0, min(255, c + flicker)) for c in blob_color]

        # Draw main blob
        cv2.circle(frame, (cx, cy), radius, tuple(blob_color), -1)

        # Draw inner concentric circle for texture
        inner_color = [max(0, min(255, c + 35)) for c in blob_color]
        cv2.circle(frame, (cx, cy), max(5, radius // 2), tuple(inner_color), -1)

        if is_fake:
            # 3. High-frequency noise overlay in blob region
            noise = rng.normal(0, 15, frame.shape).astype(np.int16)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius + 2, 255, -1)
            noisy_frame = frame.astype(np.int16) + noise
            frame[mask == 255] = np.clip(noisy_frame[mask == 255], 0, 255).astype(np.uint8)

        out.write(frame)

    out.release()


def generate_synthetic_dataset(
    data_dir: str = "data/demo/raw",
    num_real: int = 30,
    num_fake: int = 30,
    num_frames: int = 30,
    seed: int = 42,
) -> Tuple[int, int]:
    """Generate a balanced synthetic dataset of real and fake videos."""
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print(f"Generating {num_real} real synthetic videos in {real_dir}...")
    for i in range(num_real):
        vpath = os.path.join(real_dir, f"synth_real_{i:02d}.mp4")
        generate_single_video(
            output_path=vpath,
            is_fake=False,
            num_frames=num_frames,
            seed=seed + i,
        )

    print(f"Generating {num_fake} fake synthetic videos in {fake_dir}...")
    for i in range(num_fake):
        vpath = os.path.join(fake_dir, f"synth_fake_{i:02d}.mp4")
        generate_single_video(
            output_path=vpath,
            is_fake=True,
            num_frames=num_frames,
            seed=seed + 1000 + i,
        )

    return num_real, num_fake


def run_demo(
    num_videos: int = 60,
    epochs: int = 6,
    batch_size: int = 8,
    demo_dir: str = "data/demo",
    output_dir: str = "outputs/demo",
    seed: int = 42,
) -> None:
    """Run full demonstration pipeline end-to-end."""
    print("=" * 70)
    print("DEEPFAKE DETECTOR - SYNTHETIC PIPELINE DEMO")
    print("=" * 70)

    raw_dir = os.path.join(demo_dir, "raw")
    processed_dir = os.path.join(demo_dir, "processed")

    num_real = num_videos // 2
    num_fake = num_videos - num_real

    # Step 1: Generate synthetic videos
    print("\n[Step 1/5] Generating synthetic video dataset...")
    generate_synthetic_dataset(
        data_dir=raw_dir,
        num_real=num_real,
        num_fake=num_fake,
        num_frames=30,
        seed=seed,
    )

    # Step 2: Preprocess with video-level splitting and center cropping
    print("\n[Step 2/5] Running preprocessing pipeline with video-level splitting...")
    meta_df = run_preprocessing(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        seq_len=10,
        img_size=128,
        clips_per_video=3,
        face_margin=0.25,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=seed,
        no_face_detect=True,  # Documented exception for synthetic demo blobs
    )

    # Step 3: Train CNN-LSTM model
    print(f"\n[Step 3/5] Training CNN-LSTM model for {epochs} epochs...")
    train_results = train_model(
        processed_dir=processed_dir,
        output_dir=output_dir,
        seq_len=10,
        img_size=128,
        backbone="light",
        pretrained=False,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-3,
        seed=seed,
        patience=4,
        verbose=2,
    )

    # Step 4: Evaluate held-out test split
    print("\n[Step 4/5] Evaluating strictly on held-out test split...")
    eval_results = evaluate(
        model_path=train_results["best_model_path"],
        processed_dir=processed_dir,
        output_dir=output_dir,
        threshold=0.5,
        batch_size=batch_size,
    )

    # Step 5: Test single-video predict logic on both fake and real samples
    print("\n[Step 5/5] Running predict.py inference on sample synthetic videos...")

    # 5A: Predict on fake sample
    test_fake = meta_df[(meta_df["split"] == "test") & (meta_df["label"] == 1)]
    if len(test_fake) > 0:
        fake_sample_name = test_fake["video"].iloc[0]
        fake_sample_path = os.path.join(raw_dir, "fake", fake_sample_name)
    else:
        fake_sample_path = os.path.join(raw_dir, "fake", "synth_fake_00.mp4")

    print(f"\nTarget Fake Sample: {os.path.basename(fake_sample_path)}")
    fake_clips, fake_p, fake_verdict = predict_video(
        video_path=fake_sample_path,
        model_path=train_results["best_model_path"],
        threshold=0.5,
        seq_len=10,
        img_size=128,
        clips_per_video=3,
        no_face_detect=True,
    )
    print(f"  Clips analysed: {fake_clips}")
    print(f"  P(fake): {fake_p:.4f}")
    print(f"  Verdict: {fake_verdict}")

    # 5B: Predict on real sample
    test_real = meta_df[(meta_df["split"] == "test") & (meta_df["label"] == 0)]
    if len(test_real) > 0:
        real_sample_name = test_real["video"].iloc[0]
        real_sample_path = os.path.join(raw_dir, "real", real_sample_name)
    else:
        real_sample_path = os.path.join(raw_dir, "real", "synth_real_00.mp4")

    print(f"\nTarget Real Sample: {os.path.basename(real_sample_path)}")
    real_clips, real_p, real_verdict = predict_video(
        video_path=real_sample_path,
        model_path=train_results["best_model_path"],
        threshold=0.5,
        seq_len=10,
        img_size=128,
        clips_per_video=3,
        no_face_detect=True,
    )
    print(f"  Clips analysed: {real_clips}")
    print(f"  P(fake): {real_p:.4f}")
    print(f"  Verdict: {real_verdict}")

    # MANDATORY WARNING REQUIREMENT
    print("\n" + "=" * 70)
    print("WARNING: These metrics are from synthetic demonstration data only")
    print("         and are NOT deepfake-detection results.")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run synthetic deepfake detector demo.")
    parser.add_argument("--num-videos", type=int, default=60, help="Total synthetic videos to create (default: 60).")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs for demo (default: 6).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--demo-dir", type=str, default="data/demo", help="Demo data directory.")
    parser.add_argument("--output-dir", type=str, default="outputs/demo", help="Demo output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    run_demo(
        num_videos=args.num_videos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        demo_dir=args.demo_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
