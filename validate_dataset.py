"""Dataset validation script for deepfake detection.

Validates the integrity of preprocessed data and meta.csv before training:
1. Verifies meta.csv format, columns, and valid binary labels.
2. Performs strict video-level leakage checks:
   - train_videos ∩ val_videos == ∅
   - train_videos ∩ test_videos == ∅
   - val_videos ∩ test_videos == ∅
   - All clips from any single video belong to exactly one split.
3. Checks all clip files (.npy) exist on disk and can be loaded.
4. Validates clip tensor shape (seq_len, img_size, img_size, 3) and uint8 dtype.
5. Computes class distribution (real vs. fake) across all splits.
6. Prints a detailed dataset summary report.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from dataset import resolve_clip_path


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

    # Leakage check 1: Disjoint splits
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

    # 4. Clip file verification and dimensions
    total_clips = len(meta_df)
    indices_to_check = range(total_clips) if check_all_clips else np.random.choice(total_clips, min(100, total_clips), replace=False)

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
    print("=" * 65 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed deepfake dataset before training.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--seq-len", type=int, default=10, help="Expected frames per sequence.")
    parser.add_argument("--img-size", type=int, default=128, help="Expected frame resolution (H=W).")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Fast check: inspect a random sample of 100 clips rather than all clips.",
    )

    args = parser.parse_args()

    try:
        validate_processed_dataset(
            processed_dir=args.processed_dir,
            expected_seq_len=args.seq_len,
            expected_img_size=args.img_size,
            check_all_clips=not args.sample_only,
        )
        print("Validation Successful: Dataset is ready for training.")
    except Exception as e:
        print(f"\n[VALIDATION FAILED] {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
