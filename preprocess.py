"""Video preprocessing and face extraction pipeline for deepfake detection.

Supports both Celeb-DF v2 dataset structures and synthetic demo data.

Pipeline:
1. Discovers video files in real and fake directories (supporting both flat
   directories and nested subdirectories such as Celeb-real, YouTube-real,
   and Celeb-synthesis).
2. Performs video-level splitting into train/val/test splits:
   - If an official Celeb-DF testing list (List_of_testing_videos.txt) is provided,
     videos matching the list are assigned to 'test', and remaining videos are split
     into 'train' and 'val'.
   - If no official test list is provided locally, reports that clearly and performs
     stratified random video-level splitting.
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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def discover_videos(search_dir: str) -> List[str]:
    """Recursively discover video files within a directory.

    Supports nested directory structures (e.g. real/Celeb-real, real/YouTube-real).
    """
    if not os.path.exists(search_dir):
        return []

    valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_paths: List[str] = []

    for root, _, files in os.walk(search_dir):
        for f in files:
            if Path(f).suffix.lower() in valid_extensions:
                video_paths.append(os.path.join(root, f))

    return sorted(video_paths)


def load_celebdf_test_list(file_path: str) -> Set[str]:
    """Parse official Celeb-DF test video list (e.g. List_of_testing_videos.txt).

    Supports formats:
    - '<label> <rel_path>' (e.g. '1 Celeb-synthesis/id0_id1_0000.mp4')
    - '<rel_path>' (e.g. 'Celeb-synthesis/id0_id1_0000.mp4')
    Returns normalized lowercase paths and basenames for flexible matching.
    """
    if not os.path.exists(file_path):
        return set()

    test_ids = set()
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            vpath = parts[-1].replace("\\", "/").lower()
            test_ids.add(vpath)
            test_ids.add(os.path.basename(vpath))

    return test_ids


def split_videos_with_test_list(
    video_records: List[Tuple[str, int]],
    test_identifiers: Set[str],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, str], int]:
    """Assign videos to splits using the official Celeb-DF test video list.

    Videos matching the official test list are assigned to 'test'.
    Remaining videos are stratified into 'train' and 'val'.
    """
    rng = np.random.RandomState(seed)
    video_to_split: Dict[str, str] = {}
    non_test_records: List[Tuple[str, int]] = []
    test_count = 0

    for vid_id, lbl in video_records:
        norm_id = vid_id.lower().replace("\\", "/")
        base_name = os.path.basename(norm_id)
        if norm_id in test_identifiers or base_name in test_identifiers:
            video_to_split[vid_id] = "test"
            test_count += 1
        else:
            non_test_records.append((vid_id, lbl))

    # Stratify remaining non-test records into train and val
    by_class: Dict[int, List[str]] = {}
    for vid, lbl in non_test_records:
        by_class.setdefault(lbl, []).append(vid)

    for lbl, vids in sorted(by_class.items()):
        vids_shuffled = list(vids)
        rng.shuffle(vids_shuffled)
        n = len(vids_shuffled)
        if n == 0:
            continue

        n_val = int(round(n * val_fraction))
        if n_val == 0 and n > 1:
            n_val = 1
        if n_val >= n and n > 1:
            n_val = n - 1

        val_vids = vids_shuffled[:n_val]
        train_vids = vids_shuffled[n_val:]
        for v in train_vids:
            video_to_split[v] = "train"
        for v in val_vids:
            video_to_split[v] = "val"

    return video_to_split, test_count


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
            for v in vids_shuffled:
                video_to_split[v] = "train"
            continue

        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))

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
    """Extract processed clips from a video file."""
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
                    largest = max(detected, key=lambda b: b[2] * b[3])
                    bboxes.append(tuple(largest))
                else:
                    bboxes.append(None)

            valid_indices = [i for i, b in enumerate(bboxes) if b is not None]
            if not valid_indices:
                continue

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
    real_dir: Optional[str] = None,
    fake_dir: Optional[str] = None,
    test_list_path: Optional[str] = None,
    processed_dir: str = "data/processed",
    seq_len: int = 10,
    img_size: int = 128,
    clips_per_video: int = 3,
    face_margin: float = 0.25,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    no_face_detect: bool = False,
    max_videos: Optional[int] = None,
) -> pd.DataFrame:
    """Run full preprocessing pipeline.

    Supports nested directories and optional official Celeb-DF test list.
    """
    if no_face_detect:
        print("[WARNING] --no-face-detect enabled: using center crop instead of face detection.")
        print("          This option is ONLY intended for synthetic demonstration data.")
        print("          Do NOT use --no-face-detect for real Celeb-DF preprocessing.")

    if real_dir is None:
        real_dir = os.path.join(raw_dir, "real")
    if fake_dir is None:
        fake_dir = os.path.join(raw_dir, "fake")

    # Discover videos recursively to support subfolders (Celeb-real, YouTube-real, Celeb-synthesis)
    real_videos = discover_videos(real_dir)
    fake_videos = discover_videos(fake_dir)

    if max_videos:
        half_max = max(1, max_videos // 2)
        real_videos = real_videos[:half_max]
        fake_videos = fake_videos[:half_max]

    total_video_count = len(real_videos) + len(fake_videos)
    print(f"Discovered {len(real_videos)} real videos and {len(fake_videos)} fake videos (Total: {total_video_count}).")

    if total_video_count == 0:
        print(f"[WARNING] No raw videos found in '{real_dir}' or '{fake_dir}'.")
        os.makedirs(processed_dir, exist_ok=True)
        meta_df = pd.DataFrame(columns=["path", "label", "video", "clip", "split"])
        meta_df.to_csv(os.path.join(processed_dir, "meta.csv"), index=False)
        return meta_df

    # Prepare unique video identifiers relative to real_dir / fake_dir
    video_records: List[Tuple[str, int, str]] = []
    for p in real_videos:
        rel_id = os.path.relpath(p, real_dir).replace("\\", "/")
        video_records.append((f"real/{rel_id}", 0, p))
    for p in fake_videos:
        rel_id = os.path.relpath(p, fake_dir).replace("\\", "/")
        video_records.append((f"fake/{rel_id}", 1, p))

    # Check for official Celeb-DF test list
    test_list_candidate = test_list_path or os.path.join(raw_dir, "List_of_testing_videos.txt")
    official_test_set = load_celebdf_test_list(test_list_candidate) if os.path.exists(test_list_candidate) else set()

    split_records = [(vid_id, lbl) for vid_id, lbl, _ in video_records]

    if official_test_set:
        print(f"\n[INFO] Found official Celeb-DF test list at: {test_list_candidate}")
        video_to_split, test_matched = split_videos_with_test_list(
            video_records=split_records,
            test_identifiers=official_test_set,
            val_fraction=val_fraction,
            seed=seed,
        )
        print(f"       Assigned {test_matched} videos to held-out test split via official list.")
    else:
        print("\n[INFO] Official Celeb-DF testing-video list not provided locally.")
        print("       Using per-video stratified random splitting.")
        print("       Note: Random video-level split is not equivalent to the official Celeb-DF test benchmark protocol.")
        video_to_split = split_videos_stratified(
            video_records=split_records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    clips_output_dir = os.path.join(processed_dir, "clips")
    os.makedirs(clips_output_dir, exist_ok=True)

    face_cascade = None if no_face_detect else get_face_cascade()
    meta_rows: List[Dict] = []

    for video_id, label, video_path in video_records:
        video_split = video_to_split.get(video_id, "train")

        clips = extract_clips_from_video(
            video_path=video_path,
            seq_len=seq_len,
            img_size=img_size,
            clips_per_video=clips_per_video,
            face_margin=face_margin,
            no_face_detect=no_face_detect,
            face_cascade=face_cascade,
        )

        # Create collision-free safe stem from relative video_id
        safe_stem = video_id.replace("/", "_").replace("\\", "_")
        safe_stem = Path(safe_stem).stem

        for clip_idx, clip_data in enumerate(clips):
            clip_filename = f"{safe_stem}_clip{clip_idx}.npy"
            clip_path = os.path.join(clips_output_dir, clip_filename)
            np.save(clip_path, clip_data)

            rel_path = os.path.join("clips", clip_filename).replace("\\", "/")
            meta_rows.append({
                "path": rel_path,
                "label": int(label),
                "video": video_id,
                "clip": clip_idx,
                "split": video_split,
            })

    meta_df = pd.DataFrame(meta_rows, columns=["path", "label", "video", "clip", "split"])
    meta_path = os.path.join(processed_dir, "meta.csv")
    meta_df.to_csv(meta_path, index=False)

    print("\n" + "=" * 55)
    print("PREPROCESSING & SPLIT STATISTICS")
    print("=" * 55)
    print(f"Total videos processed: {total_video_count}")
    print(f"Total clips saved:      {len(meta_df)}")
    if len(meta_df) > 0:
        real_clips = (meta_df["label"] == 0).sum()
        fake_clips = (meta_df["label"] == 1).sum()
        print(f"Clips by label:         Real (0): {real_clips} | Fake (1): {fake_clips}")
        print("Clips by split:")
        for s in ["train", "val", "test"]:
            cnt = (meta_df["split"] == s).sum()
            vid_cnt = meta_df[meta_df["split"] == s]["video"].nunique()
            print(f"  - {s:<5}: {cnt} clips from {vid_cnt} videos")
    print(f"Metadata saved to:      {meta_path}")
    print("=" * 55 + "\n")

    return meta_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess video dataset for deepfake detection.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw video directory.")
    parser.add_argument("--real-dir", type=str, default=None, help="Custom path to real videos directory.")
    parser.add_argument("--fake-dir", type=str, default=None, help="Custom path to fake videos directory.")
    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="Path to official Celeb-DF List_of_testing_videos.txt split file.",
    )
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to save processed data.")
    parser.add_argument("--seq-len", type=int, default=10, help="Frames per sequence clip.")
    parser.add_argument("--img-size", type=int, default=128, help="Frame spatial resolution (H=W).")
    parser.add_argument("--clips-per-video", type=int, default=3, help="Number of clips to extract per video.")
    parser.add_argument("--face-margin", type=float, default=0.25, help="Expansion margin around face bbox.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of videos for validation.")
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Fraction of videos for test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional limit on total videos to preprocess (useful for quick checks).",
    )
    parser.add_argument(
        "--no-face-detect",
        action="store_true",
        help="Use center cropping instead of face detection. ONLY for synthetic demo data.",
    )

    args = parser.parse_args()

    run_preprocessing(
        raw_dir=args.raw_dir,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        test_list_path=args.test_list,
        processed_dir=args.processed_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        clips_per_video=args.clips_per_video,
        face_margin=args.face_margin,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        no_face_detect=args.no_face_detect,
        max_videos=args.max_videos,
    )


if __name__ == "__main__":
    main()
