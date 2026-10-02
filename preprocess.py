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
5. Provides resumable preprocessing, atomic metadata writes, skip logging,
   and multiprocessing support (--workers N).

Note:
--no-face-detect performs center cropping and is strictly intended for synthetic
demo data where human faces are not present. It is NOT for real Celeb-DF data.
"""

import argparse
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

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


def is_clip_readable(clip_path: str, seq_len: int, img_size: int) -> bool:
    """Check if an existing clip .npy file is readable and matches expected dimensions."""
    if not os.path.exists(clip_path):
        return False
    try:
        arr = np.load(clip_path)
        if arr.shape != (seq_len, img_size, img_size, 3):
            return False
        if arr.dtype != np.uint8:
            return False
        return True
    except Exception:
        return False


def save_meta_csv_atomically(meta_rows: List[Dict], meta_path: str) -> pd.DataFrame:
    """Save metadata rows to meta.csv atomically and deterministically.

    Sorts by video and clip to ensure identical output regardless of worker execution order.
    Writes to a temporary file before atomic renaming.
    """
    os.makedirs(os.path.dirname(os.path.abspath(meta_path)), exist_ok=True)
    meta_df = pd.DataFrame(meta_rows, columns=["path", "label", "video", "clip", "split"])
    if len(meta_df) > 0:
        meta_df = meta_df.sort_values(by=["video", "clip"]).reset_index(drop=True)
    temp_path = f"{meta_path}.tmp"
    meta_df.to_csv(temp_path, index=False)
    os.replace(temp_path, meta_path)
    return meta_df


def save_skipped_csv_atomically(skipped_rows: List[Dict], skipped_path: str) -> pd.DataFrame:
    """Save skipped video records to CSV atomically.

    Writes to a temporary file before atomic renaming.
    """
    os.makedirs(os.path.dirname(os.path.abspath(skipped_path)), exist_ok=True)
    skipped_df = pd.DataFrame(skipped_rows, columns=["video_path", "label", "reason"])
    if len(skipped_df) > 0:
        skipped_df = skipped_df.sort_values(by=["video_path"]).reset_index(drop=True)
    temp_path = f"{skipped_path}.tmp"
    skipped_df.to_csv(temp_path, index=False)
    os.replace(temp_path, skipped_path)
    return skipped_df


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


def classify_video_folder(rel_path: str) -> Optional[Tuple[str, int]]:
    """Derive canonical class folder and binary label (0=real, 1=fake) from folder name.

    Matches class folders case-insensitively:
    - Celeb-real -> real (0)
    - YouTube-real / Youtube-real -> real (0)
    - Celeb-synthesis -> fake (1)
    - Tolerates an extra nested 'videos/' folder inside each.
    - Also supports generic 'real' -> real (0) and 'fake' -> fake (1).

    Derives the label strictly from the FOLDER name, never from a numeric column.
    """
    norm = rel_path.replace("\\", "/").strip("/")
    parts = [p.lower() for p in norm.split("/") if p]
    if not parts:
        return None

    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if any(parts[-1].endswith(ext) for ext in video_exts):
        dir_parts = parts[:-1]
    else:
        dir_parts = parts

    if any(p == "celeb-synthesis" for p in dir_parts):
        return ("Celeb-synthesis", 1)
    if any(p in ("youtube-real", "youtubereal") for p in dir_parts):
        return ("YouTube-real", 0)
    if any(p in ("celeb-real", "celebreal") for p in dir_parts):
        return ("Celeb-real", 0)
    if any(p == "fake" for p in dir_parts):
        return ("fake", 1)
    if any(p == "real" for p in dir_parts):
        return ("real", 0)

    return None


# ==============================================================================
# Official Celeb-DF v2 Test List Label Convention (ASSUMPTION TO CONFIRM)
# ==============================================================================
# ASSUMPTION TO CONFIRM AGAINST REAL FILE:
# The official Celeb-DF v2 List_of_testing_videos.txt is documented/believed to use
# 1 = real and 0 = fake/synthetic, which is inverted relative to this project's
# internal convention (0 = real, 1 = fake).
# This assumption remains UNVERIFIED until the real dataset files are placed on disk.
# If this assumption is incorrect, parsing will fail loudly with a ValueError.
OFFICIAL_LIST_LABELS: Dict[str, int] = {
    "real": 1,
    "fake": 0,
}


def convert_official_list_label_to_project_label(list_label: int) -> int:
    """Convert an official test-list numeric label to this project's binary label convention.

    Project internal convention:
    - 0 = real
    - 1 = fake

    Official list assumption (OFFICIAL_LIST_LABELS):
    - 1 = real
    - 0 = fake

    Args:
        list_label: Integer label parsed from the official list file.

    Returns:
        Converted project integer label (0 for real, 1 for fake).

    Raises:
        ValueError: If list_label is not recognized (i.e. neither 0 nor 1).
    """
    if list_label == OFFICIAL_LIST_LABELS["real"]:
        return 0
    elif list_label == OFFICIAL_LIST_LABELS["fake"]:
        return 1
    else:
        raise ValueError(
            f"Unrecognized official test-list numeric label: {list_label}. "
            f"Expected {OFFICIAL_LIST_LABELS['real']} (real) or {OFFICIAL_LIST_LABELS['fake']} (fake)."
        )


def parse_celebdf_test_list(
    file_path: str,
) -> List[Tuple[int, str, str, str]]:
    """Parse official Celeb-DF test video list (e.g. List_of_testing_videos.txt).

    Each valid line format: '<numeric_label> <rel_path>'
    (e.g. '1 YouTube-real/00170.mp4' or '0 Celeb-synthesis/id0_id1_0000.mp4').

    The numeric column is converted from official list convention (1=real, 0=fake)
    to project convention (0=real, 1=fake) via convert_official_list_label_to_project_label,
    and then cross-checked strictly against the folder-derived label.

    Returns:
        List of tuples: (numeric_label, rel_path, folder_category, filename)

    Raises:
        FileNotFoundError: If the test list file does not exist.
        ValueError: If line format is invalid, numeric label is unrecognized, or if
                    converted label disagrees with folder-derived label.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test list file not found: '{file_path}'")

    parsed_entries: List[Tuple[int, str, str, str]] = []
    mismatches: List[str] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_num, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue

            parts = line_str.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Invalid format in test list '{file_path}' at line {line_num}: "
                    f"expected '<numeric_label> <rel_path>', got: '{line_str}'"
                )

            try:
                num_label = int(parts[0])
            except ValueError:
                raise ValueError(
                    f"Non-integer label in test list '{file_path}' at line {line_num}: '{parts[0]}'"
                )

            rel_path = parts[1].replace("\\", "/")
            filename = os.path.basename(rel_path)

            cls = classify_video_folder(rel_path)
            if cls is None:
                raise ValueError(
                    f"Unrecognized class folder in test list '{file_path}' at line {line_num}: '{rel_path}'"
                )

            folder_category, folder_label = cls

            try:
                converted_project_label = convert_official_list_label_to_project_label(num_label)
            except ValueError as e:
                mismatches.append(f"Line {line_num}: '{line_str}' -> {e}")
                continue

            if converted_project_label != folder_label:
                mismatches.append(
                    f"Line {line_num}: '{line_str}' -> official list label is {num_label} "
                    f"(converted to project label {converted_project_label}), "
                    f"but folder '{folder_category}' indicates project label {folder_label}"
                )

            parsed_entries.append((num_label, rel_path, folder_category, filename))

    if mismatches:
        raise ValueError(
            f"Label mismatch detected in official test list '{file_path}' ({len(mismatches)} mismatches):\n"
            + "\n".join(mismatches[:10])
        )

    return parsed_entries


def match_test_list_to_disk(
    test_entries: List[Tuple[int, str, str, str]],
    video_records: List[Tuple[str, str, int, str]],
    raw_dir: str,
) -> Dict[str, str]:
    """Match test list entries to discovered videos on disk by folder + filename.

    Returns:
        Dict mapping video_id to 'test' for all matched test videos.

    Raises:
        FileNotFoundError: If any test list entry has no matching video file on disk.
    """
    disk_lookup: Dict[Tuple[str, str], str] = {}
    for vid_id, folder_cat, lbl, abs_path in video_records:
        key = (folder_cat.lower(), os.path.basename(abs_path).lower())
        disk_lookup[key] = vid_id

    matched_video_ids: Set[str] = set()
    missing_entries: List[str] = []

    for num_lbl, rel_path, folder_cat, filename in test_entries:
        key = (folder_cat.lower(), filename.lower())
        if key in disk_lookup:
            matched_video_ids.add(disk_lookup[key])
        else:
            missing_entries.append(f"{rel_path} (folder: {folder_cat}, filename: {filename})")

    if missing_entries:
        missing_preview = "\n  - ".join(missing_entries[:10])
        raise FileNotFoundError(
            f"Official test list contains {len(missing_entries)} entries that do not exist on disk in '{raw_dir}':\n"
            f"  - {missing_preview}\n"
            f"Please verify that the dataset files exist on disk."
        )

    return {vid_id: "test" for vid_id in matched_video_ids}


def split_remaining_train_val(
    remaining_records: List[Tuple[str, int]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    """Stratified per-video split for remaining non-test videos into train and val."""
    rng = np.random.RandomState(seed)
    by_class: Dict[int, List[str]] = {}
    for vid_id, lbl in remaining_records:
        by_class.setdefault(lbl, []).append(vid_id)

    train_val_split: Dict[str, str] = {}
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
            train_val_split[v] = "train"
        for v in val_vids:
            train_val_split[v] = "val"

    return train_val_split


def load_celebdf_test_list(file_path: str) -> Set[str]:
    """Parse official Celeb-DF test video list (legacy compatibility wrapper)."""
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
    """Assign videos to splits using test identifiers (legacy compatibility wrapper)."""
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
    return_reason: bool = False,
) -> Any:
    """Extract processed clips from a video file.

    Returns:
        If return_reason is False: List[np.ndarray] of valid clips.
        If return_reason is True: Tuple[List[np.ndarray], Optional[str]] where
        reason is one of "unreadable", "too short", "no face detected", or None.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ([], "unreadable") if return_reason else []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len:
        cap.release()
        return ([], "too short") if return_reason else []

    if not no_face_detect and face_cascade is None:
        face_cascade = get_face_cascade()

    valid_clips: List[np.ndarray] = []
    any_frame_read = False
    any_face_detected = False

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
            any_frame_read = True
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
                    any_face_detected = True
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

    if return_reason:
        if len(valid_clips) > 0:
            return valid_clips, None
        if not any_frame_read:
            return [], "unreadable"
        if not no_face_detect and not any_face_detected:
            return [], "no face detected"
        return [], "too short"

    return valid_clips


def process_single_video(task: Tuple) -> Dict[str, Any]:
    """Top-level worker function to process a single video.

    Compatible with multiprocessing spawn on Windows.
    Handles resume check, clip extraction, atomic clip saving, and skip tracking.
    """
    (
        video_id,
        folder_cat,
        label,
        video_path,
        video_split,
        clips_output_dir,
        seq_len,
        img_size,
        clips_per_video,
        face_margin,
        no_face_detect,
        force,
    ) = task

    safe_stem = video_id.replace("/", "_").replace("\\", "_")
    safe_stem = Path(safe_stem).stem

    # Check if clips already exist and are readable (resumable)
    if not force:
        existing_clips = []
        for clip_idx in range(clips_per_video):
            clip_filename = f"{safe_stem}_clip{clip_idx}.npy"
            clip_path = os.path.join(clips_output_dir, clip_filename)
            if is_clip_readable(clip_path, seq_len, img_size):
                existing_clips.append((clip_idx, clip_filename))

        if len(existing_clips) > 0:
            meta_rows = []
            for clip_idx, clip_filename in existing_clips:
                rel_path = os.path.join("clips", clip_filename).replace("\\", "/")
                meta_rows.append({
                    "path": rel_path,
                    "label": int(label),
                    "video": video_id,
                    "clip": clip_idx,
                    "split": video_split,
                })
            return {
                "video_id": video_id,
                "video_path": video_path,
                "label": int(label),
                "split": video_split,
                "meta_rows": meta_rows,
                "skipped": False,
                "reason": None,
                "resumed": True,
            }

    # Extract clips from scratch
    face_cascade = None if no_face_detect else get_face_cascade()
    clips, reason = extract_clips_from_video(
        video_path=video_path,
        seq_len=seq_len,
        img_size=img_size,
        clips_per_video=clips_per_video,
        face_margin=face_margin,
        no_face_detect=no_face_detect,
        face_cascade=face_cascade,
        return_reason=True,
    )

    if len(clips) == 0:
        return {
            "video_id": video_id,
            "video_path": video_path,
            "label": int(label),
            "split": video_split,
            "meta_rows": [],
            "skipped": True,
            "reason": reason or "unreadable",
            "resumed": False,
        }

    meta_rows = []
    for clip_idx, clip_data in enumerate(clips):
        clip_filename = f"{safe_stem}_clip{clip_idx}.npy"
        clip_path = os.path.join(clips_output_dir, clip_filename)

        # Atomic clip write: save to .tmp.npy then rename to .npy
        tmp_clip_path = os.path.join(clips_output_dir, f"{safe_stem}_clip{clip_idx}.tmp.npy")
        np.save(tmp_clip_path, clip_data)
        os.replace(tmp_clip_path, clip_path)

        rel_path = os.path.join("clips", clip_filename).replace("\\", "/")
        meta_rows.append({
            "path": rel_path,
            "label": int(label),
            "video": video_id,
            "clip": clip_idx,
            "split": video_split,
        })

    return {
        "video_id": video_id,
        "video_path": video_path,
        "label": int(label),
        "split": video_split,
        "meta_rows": meta_rows,
        "skipped": False,
        "reason": None,
        "resumed": False,
    }


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
    force: bool = False,
    workers: int = 1,
    skipped_log_path: str = "outputs/preprocess_skipped.csv",
) -> pd.DataFrame:
    """Run full preprocessing pipeline with resume capability, atomic writes, and multiprocessing."""
    if no_face_detect:
        print("[WARNING] --no-face-detect enabled: using center crop instead of face detection.")
        print("          This option is ONLY intended for synthetic demonstration data.")
        print("          Do NOT use --no-face-detect for real Celeb-DF preprocessing.")

    video_records: List[Tuple[str, str, int, str]] = []

    custom_dirs_provided = (real_dir is not None and fake_dir is not None)

    if custom_dirs_provided:
        for p in discover_videos(real_dir):
            cls = classify_video_folder(p)
            cat = cls[0] if cls else "real"
            rel_id = f"real/{os.path.relpath(p, real_dir).replace('\\', '/')}"
            video_records.append((rel_id, cat, 0, p))
        for p in discover_videos(fake_dir):
            cls = classify_video_folder(p)
            cat = cls[0] if cls else "fake"
            rel_id = f"fake/{os.path.relpath(p, fake_dir).replace('\\', '/')}"
            video_records.append((rel_id, cat, 1, p))
    else:
        all_raw_videos = discover_videos(raw_dir)
        for p in all_raw_videos:
            rel_to_raw = os.path.relpath(p, raw_dir).replace("\\", "/")
            cls = classify_video_folder(rel_to_raw)
            if cls is not None:
                cat, lbl = cls
                video_records.append((rel_to_raw, cat, lbl, p))

    # Sort video records to ensure deterministic processing and splitting
    video_records.sort(key=lambda x: x[0])

    if max_videos and len(video_records) > max_videos:
        video_records = video_records[:max_videos]

    total_video_count = len(video_records)
    real_count = sum(1 for _, _, lbl, _ in video_records if lbl == 0)
    fake_count = sum(1 for _, _, lbl, _ in video_records if lbl == 1)
    print(f"Discovered {real_count} real videos and {fake_count} fake videos (Total: {total_video_count}).")

    meta_path = os.path.join(processed_dir, "meta.csv")

    if total_video_count == 0:
        print(f"[WARNING] No raw videos found in '{raw_dir}'.")
        os.makedirs(processed_dir, exist_ok=True)
        meta_df = save_meta_csv_atomically([], meta_path)
        save_skipped_csv_atomically([], skipped_log_path)
        return meta_df

    # Check for official Celeb-DF test list
    test_list_candidate = test_list_path or os.path.join(raw_dir, "List_of_testing_videos.txt")

    if os.path.exists(test_list_candidate):
        print(f"\n[INFO] Found official Celeb-DF test list at: {test_list_candidate}")
        parsed_test_entries = parse_celebdf_test_list(test_list_candidate)
        print(f"       Parsed {len(parsed_test_entries)} entries from official test list.")

        video_to_split = match_test_list_to_disk(
            test_entries=parsed_test_entries,
            video_records=video_records,
            raw_dir=raw_dir,
        )
        print(f"       Assigned {len(video_to_split)} videos to held-out test split via official list.")

        remaining_records = [
            (vid_id, lbl) for vid_id, _, lbl, _ in video_records if vid_id not in video_to_split
        ]
        remaining_records.sort(key=lambda x: x[0])
        train_val_split = split_remaining_train_val(
            remaining_records=remaining_records,
            val_fraction=val_fraction,
            seed=seed,
        )
        video_to_split.update(train_val_split)
    else:
        print("\n[INFO] Official Celeb-DF testing-video list not provided locally.")
        print("       Using per-video stratified random splitting.")
        split_records = [(vid_id, lbl) for vid_id, _, lbl, _ in video_records]
        split_records.sort(key=lambda x: x[0])
        video_to_split = split_videos_stratified(
            video_records=split_records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    clips_output_dir = os.path.join(processed_dir, "clips")
    os.makedirs(clips_output_dir, exist_ok=True)

    # Build tasks for processing
    tasks = []
    for video_id, folder_cat, label, video_path in video_records:
        video_split = video_to_split.get(video_id, "train")
        task = (
            video_id,
            folder_cat,
            label,
            video_path,
            video_split,
            clips_output_dir,
            seq_len,
            img_size,
            clips_per_video,
            face_margin,
            no_face_detect,
            force,
        )
        tasks.append(task)

    all_meta_rows: List[Dict] = []
    skipped_records: List[Dict] = []
    resumed_count = 0

    print(f"\nStarting video preprocessing (workers={workers}, force={force})...")
    start_time = time.time()

    def handle_result(idx: int, res: Dict[str, Any]):
        nonlocal resumed_count
        if res["skipped"]:
            skipped_records.append({
                "video_path": res["video_path"],
                "label": res["label"],
                "reason": res["reason"],
            })
        else:
            all_meta_rows.extend(res["meta_rows"])
            if res.get("resumed"):
                resumed_count += 1

        if idx % 25 == 0 or idx == total_video_count:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = total_video_count - idx
            eta = remaining / rate if rate > 0 else 0.0
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
            print(
                f"[{idx:>{len(str(total_video_count))}}/{total_video_count}] "
                f"Elapsed: {elapsed_str} | ETA: {eta_str} ({rate:.1f} vids/s) | "
                f"Clips: {len(all_meta_rows)} | Skipped: {len(skipped_records)}"
            )
            save_meta_csv_atomically(all_meta_rows, meta_path)
            save_skipped_csv_atomically(skipped_records, skipped_log_path)

    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            for idx, res in enumerate(pool.imap(process_single_video, tasks), start=1):
                handle_result(idx, res)
    else:
        for idx, task in enumerate(tasks, start=1):
            res = process_single_video(task)
            handle_result(idx, res)

    # Final safe writes
    meta_df = save_meta_csv_atomically(all_meta_rows, meta_path)
    save_skipped_csv_atomically(skipped_records, skipped_log_path)

    print("\n" + "=" * 55)
    print("SKIPPED / FAILED VIDEOS SUMMARY")
    print("=" * 55)
    print(f"Total skipped/failed videos: {len(skipped_records)}")
    reasons_count: Dict[str, int] = {}
    for r in skipped_records:
        reasons_count[r["reason"]] = reasons_count.get(r["reason"], 0) + 1
    for reason, count in sorted(reasons_count.items()):
        print(f"  - {reason}: {count}")
    if len(reasons_count) == 0:
        print("  (None - all videos processed successfully)")
    print(f"Skipped log written to:      {skipped_log_path}")
    print("=" * 55)

    print("\n" + "=" * 55)
    print("PREPROCESSING & SPLIT STATISTICS")
    print("=" * 55)
    print(f"Total videos processed: {total_video_count}")
    print(f"Videos resumed:         {resumed_count}")
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction of all clips even if valid clips already exist.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel clip extraction (default: 1).",
    )
    parser.add_argument(
        "--skipped-log",
        type=str,
        default="outputs/preprocess_skipped.csv",
        help="Path to save CSV log of skipped or failed videos.",
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
        force=args.force,
        workers=args.workers,
        skipped_log_path=args.skipped_log,
    )


if __name__ == "__main__":
    main()
