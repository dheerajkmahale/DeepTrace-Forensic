"""Video preprocessing and face extraction pipeline for deepfake detection.

Pipeline:
1. Discovers video files in raw directories (real/ and fake/).
2. Splits videos at the VIDEO level into train/val/test splits using stratification.
3. For each video:
   - Divides into clips_per_video equal temporal segments.
   - Samples seq_len evenly spaced frames per segment.
   - Detects the largest face in each frame using OpenCV Haar cascade.
   - Reuses previous bounding box if detection fails on a frame.
   - Skips clips where no face could be found.
   - Crops face with face_margin, resizes to img_size x img_size, converts to RGB.
   - Stores clip as uint8 NumPy array of shape (seq_len, img_size, img_size, 3).
4. Saves processed clips and generates meta.csv (path, label, video, clip, split).

Note:
--no-face-detect performs center cropping and is strictly intended for synthetic
demo data where human faces are not present. It is NOT for real Celeb-DF data.
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from config import Config


def get_face_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV default frontal face Haar cascade classifier."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
    return cascade


def split_videos_stratified(
    video_records: List[Tuple[str, int]],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Perform per-video stratified split into train, val, and test.

    Ensures:
    - Every video occurs in exactly one split.
    - No video appears in multiple splits.
    - Every class appears in every split (when sample counts allow).

    Args:
        video_records: List of tuples (video_path_or_id, label).
        val_fraction: Fraction of videos for validation split.
        test_fraction: Fraction of videos for test split.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict mapping video identifier to split name ("train", "val", "test").
    """
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[str]] = {}
    for vid, lbl in video_records:
        by_class.setdefault(lbl, []).append(vid)

    video_to_split: Dict[str, str] = {}

    for lbl, vids in sorted(by_class.items()):
        vids_shuffled = list(vids)
        rng.shuffle(vids_shuffled)
        n = len(vids_shuffled)

        if n < 3:
            # Too few videos to split across all 3 sets; assign to train
            for v in vids_shuffled:
                video_to_split[v] = "train"
            continue

        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))

        # Ensure at least 1 video in val and test if possible, leaving at least 1 for train
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        while n_test + n_val >= n:
            if n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            else:
                break

        test_vids = vids_shuffled[:n_test]
        val_vids = vids_shuffled[n_test : n_test + n_val]
        train_vids = vids_shuffled[n_test + n_val :]

        for v in train_vids:
            video_to_split[v] = "train"
        for v in val_vids:
            video_to_split[v] = "val"
        for v in test_vids:
            video_to_split[v] = "test"

    return video_to_split


def center_crop_and_resize(frame: np.ndarray, img_size: int) -> np.ndarray:
    """Center square crop a frame and resize to img_size x img_size."""
    h, w = frame.shape[:2]
    crop_size = min(h, w)
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    crop = frame[start_y : start_y + crop_size, start_x : start_x + crop_size]
    resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized


def crop_face_with_margin(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float,
    img_size: int,
) -> np.ndarray:
    """Crop face bounding box with margin and resize to img_size x img_size."""
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    dx = int(w * margin)
    dy = int(h * margin)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(frame_w, x + w + dx)
    y2 = min(frame_h, y + h + dy)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return center_crop_and_resize(frame, img_size)

    resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return resized


def extract_clips_from_video(
    video_path: str,
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    no_face_detect: bool = False,
    face_cascade: Optional[cv2.CascadeClassifier] = None,
) -> List[np.ndarray]:
    """Extract processed clips from a video file.

    Args:
        video_path: Path to input video file.
        seq_len: Number of frames per clip.
        img_size: Height and width of extracted frames.
        clips_per_video: Number of temporal segments to extract.
        face_margin: Expansion margin ratio around face bounding box.
        no_face_detect: If True, uses center cropping instead of face detector.
        face_cascade: Preloaded cv2 CascadeClassifier instance.

    Returns:
        List of uint8 NumPy arrays of shape (seq_len, img_size, img_size, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len:
        cap.release()
        return []

    if not no_face_detect and face_cascade is None:
        face_cascade = get_face_cascade()

    valid_clips: List[np.ndarray] = []

    for c in range(clips_per_video):
        seg_start = int(c * total_frames / clips_per_video)
        seg_end = int((c + 1) * total_frames / clips_per_video)
        if seg_end <= seg_start:
            continue

        # Sample seq_len evenly spaced frame indices in this temporal segment
        frame_indices = np.linspace(seg_start, seg_end - 1, seq_len, dtype=int)

        raw_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            raw_frames.append(frame)

        if len(raw_frames) != seq_len:
            continue

        if no_face_detect:
            processed_frames = []
            for frame in raw_frames:
                crop = center_crop_and_resize(frame, img_size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb)
            clip_array = np.array(processed_frames, dtype=np.uint8)
            valid_clips.append(clip_array)
        else:
            # Detect faces across frames
            bboxes: List[Optional[Tuple[int, int, int, int]]] = []
            for frame in raw_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
                if len(detected) > 0:
                    # Select largest face by area
                    largest = max(detected, key=lambda b: b[2] * b[3])
                    bboxes.append(tuple(largest))
                else:
                    bboxes.append(None)

            # Check if any face was detected in this clip
            valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
            if not valid_indices:
                # Skip clip if no face can be found anywhere in the clip
                continue

            # Fill missing bounding boxes using forward / backward propagation
            filled_bboxes: List[Tuple[int, int, int, int]] = []
            last_valid = bboxes[valid_indices[0]]
            for b in bboxes:
                if b is not None:
                    last_valid = b
                filled_bboxes.append(last_valid)

            processed_frames = []
            for frame, bbox in zip(raw_frames, filled_bboxes):
                crop = crop_face_with_margin(frame, bbox, face_margin, img_size)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb)

            clip_array = np.array(processed_frames, dtype=np.uint8)
            valid_clips.append(clip_array)

    cap.release()
    return valid_clips


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    no_face_detect: bool = False,
) -> pd.DataFrame:
    """Run full preprocessing pipeline.

    1. Discovers videos in raw_dir/real and raw_dir/fake.
    2. Performs video-level stratified splitting.
    3. Extracts clips and writes .npy files.
    4. Writes meta.csv.
    5. Prints summary statistics.
    """
    if no_face_detect:
        print("[WARNING] --no-face-detect enabled: using center crop instead of face detection.")
        print("          This option is ONLY intended for synthetic demonstration data.")
        print("          Do NOT use --no-face-detect for real Celeb-DF preprocessing.")

    real_dir = os.path.join(raw_dir, "real")
    fake_dir = os.path.join(raw_dir, "fake")

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    real_videos: List[str] = []
    fake_videos: List[str] = []

    for ext in video_extensions:
        real_videos.extend(glob.glob(os.path.join(real_dir, ext)))
        real_videos.extend(glob.glob(os.path.join(real_dir, ext.upper())))
        fake_videos.extend(glob.glob(os.path.join(fake_dir, ext)))
        fake_videos.extend(glob.glob(os.path.join(fake_dir, ext.upper())))

    real_videos = sorted(list(set(real_videos)))
    fake_videos = sorted(list(set(fake_videos)))

    total_video_count = len(real_videos) + len(fake_videos)
    print(f"Found {len(real_videos)} real videos and {len(fake_videos)} fake videos (Total: {total_video_count}).")

    if total_video_count == 0:
        print("[WARNING] No raw videos found in real/ or fake/ directories.")
        # Create empty meta.csv with required columns
        os.makedirs(processed_dir, exist_ok=True)
        meta_df = pd.DataFrame(columns=["path", "label", "video", "clip", "split"])
        meta_df.to_csv(os.path.join(processed_dir, "meta.csv"), index=False)
        return meta_df

    # Prepare records for video-level splitting
    video_records: List[Tuple[str, int]] = []
    for p in real_videos:
        video_records.append((os.path.basename(p), 0))
    for p in fake_videos:
        video_records.append((os.path.basename(p), 1))

    # Split videos strictly at the video level BEFORE assigning clips
    video_to_split = split_videos_stratified(
        video_records=video_records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    clips_output_dir = os.path.join(processed_dir, "clips")
    os.makedirs(clips_output_dir, exist_ok=True)

    face_cascade = None if no_face_detect else get_face_cascade()
    all_videos = [(p, 0) for p in real_videos] + [(p, 1) for p in fake_videos]

    meta_rows: List[Dict] = []
    for video_path, label in all_videos:
        video_name = os.path.basename(video_path)
        video_split = video_to_split.get(video_name, "train")

        clips = extract_clips_from_video(
            video_path=video_path,
            seq_len=seq_len,
            img_size=img_size,
            clips_per_video=clips_per_video,
            face_margin=face_margin,
            no_face_detect=no_face_detect,
            face_cascade=face_cascade,
        )

        stem = Path(video_name).stem
        for clip_idx, clip_data in enumerate(clips):
            clip_filename = f"{stem}_clip{clip_idx}.npy"
            clip_path = os.path.join(clips_output_dir, clip_filename)
            np.save(clip_path, clip_data)

            # Store relative path from processed_dir
            rel_path = os.path.join("clips", clip_filename).replace("\\", "/")
            meta_rows.append({
                "path": rel_path,
                "label": int(label),
                "video": video_name,
                "clip": clip_idx,
                "split": video_split,
            })

    meta_df = pd.DataFrame(meta_rows, columns=["path", "label", "video", "clip", "split"])
    meta_path = os.path.join(processed_dir, "meta.csv")
    meta_df.to_csv(meta_path, index=False)

    # Print useful split statistics
    print("\n" + "=" * 50)
    print("PREPROCESSING & SPLIT STATISTICS")
    print("=" * 50)
    print(f"Total videos processed: {total_video_count}")
    print(f"Total clips saved:      {len(meta_df)}")
    if len(meta_df) > 0:
        real_clips = (meta_df['label'] == 0).sum()
        fake_clips = (meta_df['label'] == 1).sum()
        print(f"Clips by label:         Real (0): {real_clips} | Fake (1): {fake_clips}")
        print("Clips by split:")
        for s in ["train", "val", "test"]:
            cnt = (meta_df['split'] == s).sum()
            vid_cnt = meta_df[meta_df['split'] == s]['video'].nunique()
            print(f"  - {s:<5}: {cnt} clips from {vid_cnt} videos")
    print(f"Metadata saved to:      {meta_path}")
    print("=" * 50 + "\n")

    return meta_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess video dataset for deepfake detection.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw video directory.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to save processed data.")
    parser.add_argument("--seq-len", type=int, default=10, help="Frames per sequence clip.")
    parser.add_argument("--img-size", type=int, default=128, help="Frame spatial resolution (H=W).")
    parser.add_argument("--clips-per-video", type=int, default=3, help="Number of clips to extract per video.")
    parser.add_argument("--face-margin", type=float, default=0.25, help="Expansion margin around face bbox.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of videos for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of videos for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Use center cropping instead of face detection. ONLY for synthetic demo data.",
    )

    args = parser.parse_args()

    run_preprocessing(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        clips_per_video=args.clips_per_video,
        face_margin=args.face_margin,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        no_face_detect=args.no_face_detect,
    )


if __name__ == "__main__":
    main()
