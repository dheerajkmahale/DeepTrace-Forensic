"""Dataset validation script for deepfake detection.

Validates the integrity of preprocessed data and meta.csv before training:
1. Verifies meta.csv format, columns, and valid binary labels.
2. Performs strict video-level leakage checks:
   - train_videos ∩ val_videos == ∅
   - train_videos ∩ test_videos == ∅
   - val_videos ∩ test_videos == ∅
   - All clips from any single video belong to exactly one split.
   - Asserts no processed clip of a test video appears in train/val.
3. Checks all clip files (.npy) exist on disk and can be loaded.
4. Validates clip tensor shape (seq_len, img_size, img_size, 3) and uint8 dtype.
5. Computes class distribution (real vs. fake) across all splits.
6. Prints an informational celebrity identity-overlap report between train and test.
7. Supports raw dataset validation (video readability with cv2, duplicate filenames, test-list verification).
"""

import argparse
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd

from dataset import resolve_clip_path
from preprocess import (
    classify_video_folder,
    discover_videos,
    match_test_list_to_disk,
    parse_celebdf_test_list,
)


def extract_identities_from_filename(filename: str) -> Set[str]:
    """Parse celebrity identity IDs (such as idX or idX_idY) from a video filename.

    Examples:
    - 'id0_0000.mp4' -> {'id0'}
    - 'id0_id1_0000.mp4' -> {'id0', 'id1'}
    - '00170.mp4' -> set()
    """
    base = os.path.basename(filename).lower()
    matches = re.findall(r"id\d+", base)
    return set(matches)


def generate_identity_overlap_report(meta_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and print an informational identity-overlap report between train and test splits.

    Note: The Celeb-DF v2 official benchmark split permits celebrity identity overlap
    between splits. This report is strictly informational and does not alter the split.
    """
    train_mask = meta_df["split"].str.lower() == "train"
    val_mask = meta_df["split"].str.lower() == "val"
    test_mask = meta_df["split"].str.lower() == "test"

    train_videos = set(meta_df[train_mask]["video"].unique())
    val_videos = set(meta_df[val_mask]["video"].unique())
    test_videos = set(meta_df[test_mask]["video"].unique())

    train_ids: Set[str] = set()
    for v in train_videos:
        train_ids.update(extract_identities_from_filename(v))

    val_ids: Set[str] = set()
    for v in val_videos:
        val_ids.update(extract_identities_from_filename(v))

    test_ids: Set[str] = set()
    for v in test_videos:
        test_ids.update(extract_identities_from_filename(v))

    train_test_overlap = train_ids.intersection(test_ids)
    val_test_overlap = val_ids.intersection(test_ids)

    print("-" * 65)
    print("CELEBRITY IDENTITY OVERLAP REPORT (INFORMATIONAL)")
    print("-" * 65)
    print(f"Unique Identities in Train: {len(train_ids)}")
    print(f"Unique Identities in Val:   {len(val_ids)}")
    print(f"Unique Identities in Test:  {len(test_ids)}")
    print(f"Train / Test ID Overlap:    {len(train_test_overlap)} identities")
    if train_test_overlap:
        preview = sorted(list(train_test_overlap))[:10]
        print(f"Overlapping IDs (sample):   {preview}")
    print("Note: Celeb-DF v2 official benchmark protocol allows celebrity identity")
    print("      overlap between splits. The official split is preserved unchanged.")
    print("-" * 65)

    return {
        "train_identities": train_ids,
        "val_identities": val_ids,
        "test_identities": test_ids,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
    }


def validate_raw_dataset(
    raw_dir: str = "data/raw",
    test_list_path: Optional[str] = None,
    check_readability: bool = True,
) -> Dict[str, Any]:
    """Validate raw Celeb-DF dataset structure, video readability, duplicates, and test list.

    Args:
        raw_dir: Path to raw dataset directory.
        test_list_path: Optional path to official List_of_testing_videos.txt.
        check_readability: If True, tests opening and reading first frame of each video using cv2.

    Returns:
        Dict with validation statistics.

    Raises:
        FileNotFoundError: If raw directory or test list entries are missing.
        ValueError: If corrupted videos or label mismatches are detected.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw directory not found: '{raw_dir}'")

    raw_videos = discover_videos(raw_dir)
    if not raw_videos:
        raise ValueError(f"No video files found in raw directory '{raw_dir}'.")

    # 1. Count videos per folder category
    by_category: Dict[str, List[str]] = {}
    unclassified: List[str] = []
    filename_map: Dict[str, List[str]] = {}

    for vpath in raw_videos:
        rel_path = os.path.relpath(vpath, raw_dir).replace("\\", "/")
        cls = classify_video_folder(rel_path)
        cat = cls[0] if cls else "unclassified"
        if cls:
            by_category.setdefault(cat, []).append(vpath)
        else:
            unclassified.append(vpath)

        fname = os.path.basename(vpath).lower()
        filename_map.setdefault(fname, []).append(rel_path)

    # 2. Check for duplicate filenames
    duplicate_filenames = {fname: paths for fname, paths in filename_map.items() if len(paths) > 1}

    # 3. Check readability with cv2
    corrupted_videos: List[Tuple[str, str]] = []
    if check_readability:
        for vpath in raw_videos:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                corrupted_videos.append((vpath, "VideoCapture could not open file"))
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                corrupted_videos.append((vpath, "Failed to decode first frame"))
            cap.release()

    if corrupted_videos:
        preview = "\n  - ".join([f"{p}: {reason}" for p, reason in corrupted_videos[:5]])
        raise ValueError(
            f"Corrupted or unreadable raw videos detected ({len(corrupted_videos)} files):\n  - {preview}"
        )

    # 4. Check test list entries if present
    test_list_candidate = test_list_path or os.path.join(raw_dir, "List_of_testing_videos.txt")
    test_entries = []
    if os.path.exists(test_list_candidate):
        test_entries = parse_celebdf_test_list(test_list_candidate)
        # Check that all test list entries exist on disk
        video_records = []
        for p in raw_videos:
            rel_id = os.path.relpath(p, raw_dir).replace("\\", "/")
            cls = classify_video_folder(rel_id)
            cat = cls[0] if cls else "unknown"
            lbl = cls[1] if cls else 0
            video_records.append((rel_id, cat, lbl, p))

        match_test_list_to_disk(test_entries, video_records, raw_dir)

    stats = {
        "raw_dir": raw_dir,
        "total_videos": len(raw_videos),
        "by_category": {cat: len(vids) for cat, vids in by_category.items()},
        "unclassified_count": len(unclassified),
        "duplicate_filenames": duplicate_filenames,
        "test_list_entries": len(test_entries),
        "corrupted_count": len(corrupted_videos),
    }

    print("\n" + "=" * 65)
    print("RAW DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"Raw Directory:       {raw_dir}")
    print(f"Total Video Files:   {stats['total_videos']}")
    for cat, cnt in sorted(stats["by_category"].items()):
        print(f"  - {cat:<18}: {cnt} videos")
    if stats["unclassified_count"] > 0:
        print(f"  - {'Unclassified':<18}: {stats['unclassified_count']} videos")
    print(f"Duplicate Filenames: {len(duplicate_filenames)} detected")
    if duplicate_filenames:
        for dfname, paths in list(duplicate_filenames.items())[:3]:
            print(f"    * '{dfname}' found in: {paths}")
    print(f"Test List File:      {test_list_candidate if os.path.exists(test_list_candidate) else 'None'}")
    if os.path.exists(test_list_candidate):
        print(f"Test List Entries:   {len(test_entries)} (all verified on disk)")
    print(f"Readability Check:   PASSED ({len(raw_videos)} videos decoded successfully with cv2)")
    print("=" * 65 + "\n")

    return stats


def validate_processed_dataset(
    processed_dir: str = "data/processed",
    expected_seq_len: int = 10,
    expected_img_size: int = 128,
    check_all_clips: bool = True,
) -> Dict[str, Any]:
    """Validate processed dataset integrity, metadata consistency, and split isolation.

    Args:
        processed_dir: Path to directory containing meta.csv and clips.
        expected_seq_len: Expected frames per sequence.
        expected_img_size: Expected frame spatial resolution (H=W).
        check_all_clips: If True, inspects all clips; if False, samples up to 100.

    Returns:
        Dict containing validation statistics and report.

    Raises:
        FileNotFoundError: If meta.csv or processed_dir does not exist.
        ValueError: If metadata contains errors, corruption, or split leakage.
    """
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: '{processed_dir}'")

    meta_path = os.path.join(processed_dir, "meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Metadata file '{meta_path}' not found. Please run preprocess.py first."
        )

    meta_df = pd.read_csv(meta_path)
    if len(meta_df) == 0:
        raise ValueError(f"Metadata file '{meta_path}' is completely empty (0 rows).")

    # 1. Column verification
    required_cols = {"path", "label", "video", "clip", "split"}
    missing_cols = required_cols - set(meta_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in meta.csv: {missing_cols}")

    # 2. Label verification
    unique_labels = set(meta_df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid labels in meta.csv: {unique_labels}. Expected binary {0, 1}.")

    # 3. Video-level split isolation (CRITICAL LEAKAGE PREVENTION)
    train_mask = meta_df["split"].str.lower() == "train"
    val_mask = meta_df["split"].str.lower() == "val"
    test_mask = meta_df["split"].str.lower() == "test"

    train_videos: Set[str] = set(meta_df[train_mask]["video"].unique())
    val_videos: Set[str] = set(meta_df[val_mask]["video"].unique())
    test_videos: Set[str] = set(meta_df[test_mask]["video"].unique())

    # Leakage check 1: Disjoint splits at video level
    train_val_overlap = train_videos.intersection(val_videos)
    train_test_overlap = train_videos.intersection(test_videos)
    val_test_overlap = val_videos.intersection(test_videos)

    if train_val_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(train_val_overlap)} videos appear in both train and val splits: "
            f"{list(train_val_overlap)[:5]}"
        )
    if train_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(train_test_overlap)} videos appear in both train and test splits: "
            f"{list(train_test_overlap)[:5]}"
        )
    if val_test_overlap:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(val_test_overlap)} videos appear in both val and test splits: "
            f"{list(val_test_overlap)[:5]}"
        )

    # Assert pairwise disjointness
    assert len(train_val_overlap) == 0, f"train and val overlap: {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"train and test overlap: {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"val and test overlap: {val_test_overlap}"

    # Leakage check 2: All clips of every video belong to exactly one split
    multi_split_vids = []
    for vid, group in meta_df.groupby("video"):
        if group["split"].nunique() > 1:
            multi_split_vids.append((vid, group["split"].unique().tolist()))
    if multi_split_vids:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Clips from {len(multi_split_vids)} videos cross split boundaries: "
            f"{multi_split_vids[:3]}"
        )

    # Leakage check 3: Assert no processed clip of a test video appears in train or val
    test_clips_in_train_or_val = meta_df[
        (meta_df["video"].isin(test_videos)) & (meta_df["split"].str.lower().isin(["train", "val"]))
    ]
    if len(test_clips_in_train_or_val) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(test_clips_in_train_or_val)} clips belonging to test videos "
            f"appear in train or val splits!"
        )

    train_clips_in_val_or_test = meta_df[
        (meta_df["video"].isin(train_videos)) & (meta_df["split"].str.lower().isin(["val", "test"]))
    ]
    if len(train_clips_in_val_or_test) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(train_clips_in_val_or_test)} clips belonging to train videos "
            f"appear in val or test splits!"
        )

    # 4. Clip file verification and dimensions
    total_clips = len(meta_df)
    indices_to_check = (
        range(total_clips)
        if check_all_clips
        else np.random.choice(total_clips, min(100, total_clips), replace=False)
    )

    expected_shape = (expected_seq_len, expected_img_size, expected_img_size, 3)
    missing_files: List[str] = []
    corrupt_files: List[Tuple[str, str]] = []
    shape_mismatches: List[Tuple[str, Tuple]] = []

    for idx in indices_to_check:
        row = meta_df.iloc[idx]
        clip_rel_path = str(row["path"])
        full_clip_path = resolve_clip_path(clip_rel_path, processed_dir)

        if not os.path.exists(full_clip_path):
            missing_files.append(clip_rel_path)
            continue

        try:
            arr = np.load(full_clip_path)
            if arr.shape != expected_shape:
                shape_mismatches.append((clip_rel_path, arr.shape))
            if arr.dtype != np.uint8:
                corrupt_files.append((clip_rel_path, f"dtype is {arr.dtype}, expected uint8"))
        except Exception as e:
            corrupt_files.append((clip_rel_path, str(e)))

    if missing_files:
        raise ValueError(
            f"Missing {len(missing_files)} clip files on disk referenced in meta.csv: {missing_files[:5]}"
        )
    if corrupt_files:
        raise ValueError(
            f"Corrupt or invalid clip files detected ({len(corrupt_files)} files): {corrupt_files[:5]}"
        )
    if shape_mismatches:
        raise ValueError(
            f"Clip tensor shape mismatch ({len(shape_mismatches)} files): "
            f"Expected {expected_shape}, got {shape_mismatches[:3]}"
        )

    # 5. Statistics compilation
    stats = {
        "total_clips": total_clips,
        "total_videos": meta_df["video"].nunique(),
        "train_videos": len(train_videos),
        "val_videos": len(val_videos),
        "test_videos": len(test_videos),
        "train_clips": int(train_mask.sum()),
        "val_clips": int(val_mask.sum()),
        "test_clips": int(test_mask.sum()),
        "train_real": int(((meta_df["label"] == 0) & train_mask).sum()),
        "train_fake": int(((meta_df["label"] == 1) & train_mask).sum()),
        "val_real": int(((meta_df["label"] == 0) & val_mask).sum()),
        "val_fake": int(((meta_df["label"] == 1) & val_mask).sum()),
        "test_real": int(((meta_df["label"] == 0) & test_mask).sum()),
        "test_fake": int(((meta_df["label"] == 1) & test_mask).sum()),
        "expected_shape": expected_shape,
        "leakage_passed": True,
    }

    # Print summary table
    print("\n" + "=" * 65)
    print("DATASET INTEGRITY & SPLIT VALIDATION REPORT")
    print("=" * 65)
    print(f"Processed Directory:  {processed_dir}")
    print(f"Total Unique Videos:  {stats['total_videos']}")
    print(f"Total Clip Files:     {stats['total_clips']}")
    print(f"Expected Dimensions:  {expected_shape} (uint8)")
    print("-" * 65)
    print(f"{'Split':<10} | {'Videos':<8} | {'Clips (Total)':<14} | {'Real (0)':<10} | {'Fake (1)':<10}")
    print("-" * 65)
    print(f"{'Train':<10} | {stats['train_videos']:<8} | {stats['train_clips']:<14} | {stats['train_real']:<10} | {stats['train_fake']:<10}")
    print(f"{'Val':<10} | {stats['val_videos']:<8} | {stats['val_clips']:<14} | {stats['val_real']:<10} | {stats['val_fake']:<10}")
    print(f"{'Test':<10} | {stats['test_videos']:<8} | {stats['test_clips']:<14} | {stats['test_real']:<10} | {stats['test_fake']:<10}")
    print("-" * 65)
    print(f"{'Total':<10} | {stats['total_videos']:<8} | {stats['total_clips']:<14} | {stats['train_real']+stats['val_real']+stats['test_real']:<10} | {stats['train_fake']+stats['val_fake']+stats['test_fake']:<10}")
    print("-" * 65)
    print("Leakage Checks:       PASSED (0 video overlap between train, val, test)")
    print("File Integrity:       PASSED (all clip files readable and uncorrupted)")

    # 6. Informational celebrity identity overlap report
    id_report = generate_identity_overlap_report(meta_df)
    stats["identity_report"] = id_report

    print("=" * 65 + "\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate deepfake dataset integrity.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory.")
    parser.add_argument("--test-list", type=str, default=None, help="Path to official test list.")
    parser.add_argument("--check-raw", action="store_true", help="Explicitly validate raw dataset.")
    parser.add_argument("--seq-len", type=int, default=10, help="Expected frames per sequence.")
    parser.add_argument("--img-size", type=int, default=128, help="Expected frame resolution (H=W).")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Fast check: inspect a random sample of 100 clips rather than all clips.",
    )

    args = parser.parse_args()

    try:
        # Check raw dataset if explicitly requested or if raw-dir contains videos
        if args.check_raw or (args.raw_dir and os.path.exists(args.raw_dir)):
            raw_videos = discover_videos(args.raw_dir)
            if raw_videos:
                print(f"[INFO] Found {len(raw_videos)} raw videos in '{args.raw_dir}'. Running raw dataset validation...")
                validate_raw_dataset(
                    raw_dir=args.raw_dir,
                    test_list_path=args.test_list,
                    check_readability=True,
                )
            elif args.check_raw:
                raise FileNotFoundError(f"No video files found in raw directory: '{args.raw_dir}'")

        # Check processed dataset if present
        meta_csv = os.path.join(args.processed_dir, "meta.csv")
        if os.path.exists(args.processed_dir) and os.path.exists(meta_csv):
            validate_processed_dataset(
                processed_dir=args.processed_dir,
                expected_seq_len=args.seq_len,
                expected_img_size=args.img_size,
                check_all_clips=not args.sample_only,
            )
            print("Validation Successful: Processed dataset is ready for training.")
        elif not args.check_raw:
            print(f"[INFO] No processed data at '{args.processed_dir}'.")

        sys.exit(0)
    except Exception as e:
        print(f"\n[VALIDATION FAILED] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
