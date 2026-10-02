"""Unit tests for deepfake detector pipeline components.

Tests cover:
1. Model output shape (batch_size, 1) and output range [0, 1].
2. Per-video stratified splitting (leakage prevention, single-split membership,
   class balance).
3. Balanced class weight calculation.
4. Temporal clip augmentation shape & dtype preservation.
5. Dataset cardinality validation via tf.data.experimental.cardinality.
6. Model serialization (saving to and loading from .keras format).
"""

import math
import os
import shutil
import sys
import tempfile
import unittest

# Ensure project root is in sys.path for direct pytest execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import augment_clip, compute_class_weights, get_dataset
from model import build_model
from preprocess import split_videos_stratified


class TestDeepfakePipeline(unittest.TestCase):
    """Test suite verifying pipeline integrity, model architecture, and leakage prevention."""

    def test_model_output_shape_and_range(self):
        """Test 1: Model output shape and range.

        Verifies that:
        - Output shape is precisely (batch_size, 1).
        - Sigmoid activation outputs are strictly bounded within [0, 1].
        """
        batch_size = 4
        seq_len = 5
        img_size = 64

        model = build_model(
            seq_len=seq_len,
            img_size=img_size,
            backbone="light",
            lstm_units=32,
            dropout=0.1,
        )

        dummy_input = np.random.randint(
            0,
            256,
            size=(batch_size, seq_len, img_size, img_size, 3),
            dtype=np.uint8,
        )

        outputs = model.predict(dummy_input, verbose=0)

        # Check shape
        self.assertEqual(
            outputs.shape,
            (batch_size, 1),
            f"Expected output shape ({batch_size}, 1), got {outputs.shape}",
        )

        # Check range [0, 1]
        self.assertTrue(
            np.all(outputs >= 0.0),
            f"Found outputs less than 0.0: min is {outputs.min()}",
        )
        self.assertTrue(
            np.all(outputs <= 1.0),
            f"Found outputs greater than 1.0: max is {outputs.max()}",
        )

    def test_per_video_stratified_splitting(self):
        """Test 2: Per-video stratified splitting.

        Verifies:
        - Every video occurs in exactly one split.
        - No video appears in multiple splits.
        - Every class appears in every split.
        """
        # Create a synthetic dataset of 20 videos (10 real, 10 fake)
        records = [(f"real_vid_{i:02d}.mp4", 0) for i in range(10)] + [
            (f"fake_vid_{i:02d}.mp4", 1) for i in range(10)
        ]

        video_names = [v for v, _ in records]
        val_fraction = 0.20
        test_fraction = 0.20
        seed = 42

        video_to_split = split_videos_stratified(
            video_records=records,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

        # 1. Total count matches
        self.assertEqual(
            len(video_to_split),
            len(records),
            "Split mapping does not cover all input videos.",
        )

        # 2. Every video in exactly one split
        all_assigned_videos = list(video_to_split.keys())
        self.assertEqual(
            len(all_assigned_videos),
            len(set(all_assigned_videos)),
            "Duplicate video assignments found.",
        )
        self.assertEqual(
            set(all_assigned_videos),
            set(video_names),
            "Assigned videos do not match original video set.",
        )

        # Group by split and verify disjointness
        split_to_videos = {"train": [], "val": [], "test": []}
        for vid, s in video_to_split.items():
            self.assertIn(s, split_to_videos, f"Invalid split name: {s}")
            split_to_videos[s].append(vid)

        train_set = set(split_to_videos["train"])
        val_set = set(split_to_videos["val"])
        test_set = set(split_to_videos["test"])

        self.assertEqual(
            len(train_set.intersection(val_set)),
            0,
            "Leakage detected: videos overlap between train and val splits.",
        )
        self.assertEqual(
            len(train_set.intersection(test_set)),
            0,
            "Leakage detected: videos overlap between train and test splits.",
        )
        self.assertEqual(
            len(val_set.intersection(test_set)),
            0,
            "Leakage detected: videos overlap between val and test splits.",
        )

        # 3. Every class appears in every split
        record_dict = dict(records)
        for s_name, v_list in split_to_videos.items():
            classes_in_split = {record_dict[v] for v in v_list}
            self.assertIn(0, classes_in_split, f"Class 0 (real) missing in split '{s_name}'")
            self.assertIn(1, classes_in_split, f"Class 1 (fake) missing in split '{s_name}'")

    def test_class_weights_calculation(self):
        """Verify dynamic balanced class weight computation."""
        labels = np.array([0, 0, 0, 0, 1])  # 4 real, 1 fake
        weights = compute_class_weights(labels)

        self.assertIn(0, weights)
        self.assertIn(1, weights)
        self.assertGreater(
            weights[1],
            weights[0],
            "Minority class weight should exceed majority class weight.",
        )

    def test_augmentation_shape_and_consistency(self):
        """Verify clip augmentation maintains array shape and uint8 dtype."""
        rng = np.random.RandomState(42)
        clip = np.random.randint(0, 256, (6, 32, 32, 3), dtype=np.uint8)
        augmented = augment_clip(clip, rng)

        self.assertEqual(augmented.shape, clip.shape)
        self.assertEqual(augmented.dtype, np.uint8)

    def test_dataset_cardinality(self):
        """Verify dataset pipeline sets exact expected cardinality for Keras."""
        temp_dir = tempfile.mkdtemp()
        try:
            seq_len = 4
            img_size = 32
            num_samples = 7
            batch_size = 3
            expected_batches = math.ceil(num_samples / batch_size)

            clips_dir = os.path.join(temp_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)

            meta_rows = []
            for i in range(num_samples):
                clip_data = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)
                clip_name = f"dummy_{i}.npy"
                clip_path = os.path.join(clips_dir, clip_name)
                np.save(clip_path, clip_data)
                meta_rows.append({
                    "path": os.path.join("clips", clip_name),
                    "label": i % 2,
                    "video": f"dummy_vid_{i}",
                    "clip": 0,
                    "split": "train",
                })

            meta_df = pd.DataFrame(meta_rows)
            meta_path = os.path.join(temp_dir, "meta.csv")
            meta_df.to_csv(meta_path, index=False)

            ds, n_samples = get_dataset(
                meta_path=meta_path,
                split="train",
                processed_dir=temp_dir,
                batch_size=batch_size,
                seq_len=seq_len,
                img_size=img_size,
                is_training=False,
            )

            card = tf.data.experimental.cardinality(ds).numpy()
            self.assertEqual(
                card,
                expected_batches,
                f"Expected cardinality {expected_batches}, got {card}",
            )
            self.assertEqual(n_samples, num_samples)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_model_save_and_load(self):
        """Verify model serialization and deserialization (.keras format)."""
        temp_file = os.path.join(tempfile.gettempdir(), "test_save_load_model.keras")
        try:
            model = build_model(
                seq_len=3,
                img_size=32,
                backbone="light",
                lstm_units=16,
                dropout=0.1,
            )
            model.compile(optimizer="adam", loss="binary_crossentropy")

            dummy = np.random.randint(0, 256, (2, 3, 32, 32, 3), dtype=np.uint8)
            preds_orig = model.predict(dummy, verbose=0)

            model.save(temp_file)
            self.assertTrue(os.path.exists(temp_file), "Failed to save .keras model file.")

            loaded_model = tf.keras.models.load_model(temp_file)
            preds_loaded = loaded_model.predict(dummy, verbose=0)

            np.testing.assert_allclose(
                preds_orig,
                preds_loaded,
                rtol=1e-5,
                atol=1e-5,
                err_msg="Predictions differed after model load.",
            )
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_celebdf_test_list_splitting(self):
        """Verify official Celeb-DF test list split mapping."""
        from preprocess import split_videos_with_test_list

        records = [(f"real_vid_{i}.mp4", 0) for i in range(6)] + [
            (f"fake_vid_{i}.mp4", 1) for i in range(6)
        ]
        test_ids = {"real_vid_0.mp4", "fake_vid_1.mp4"}

        mapping, test_count = split_videos_with_test_list(
            video_records=records,
            test_identifiers=test_ids,
            val_fraction=0.25,
            seed=42,
        )

        self.assertEqual(test_count, 2)
        self.assertEqual(mapping["real_vid_0.mp4"], "test")
        self.assertEqual(mapping["fake_vid_1.mp4"], "test")

        # Disjointness check
        train_vids = {v for v, s in mapping.items() if s == "train"}
        val_vids = {v for v, s in mapping.items() if s == "val"}
        test_vids = {v for v, s in mapping.items() if s == "test"}

        self.assertEqual(len(train_vids.intersection(val_vids)), 0)
        self.assertEqual(len(train_vids.intersection(test_vids)), 0)
        self.assertEqual(len(val_vids.intersection(test_vids)), 0)

    def test_validate_dataset_leakage_detection(self):
        """Verify validate_processed_dataset catches injected data leakage."""
        from validate_dataset import validate_processed_dataset

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a meta.csv with intentional leakage: same video in train and test
            meta_rows = [
                {"path": "clips/c1.npy", "label": 0, "video": "leaked_video", "clip": 0, "split": "train"},
                {"path": "clips/c2.npy", "label": 0, "video": "leaked_video", "clip": 1, "split": "test"},
                {"path": "clips/c3.npy", "label": 1, "video": "safe_video", "clip": 0, "split": "val"},
            ]
            meta_df = pd.DataFrame(meta_rows)
            meta_df.to_csv(os.path.join(temp_dir, "meta.csv"), index=False)

            with self.assertRaises(ValueError) as ctx:
                validate_processed_dataset(processed_dir=temp_dir, check_all_clips=False)
            self.assertIn("DATA LEAKAGE DETECTED", str(ctx.exception))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_discover_videos_nested(self):
        """Verify discover_videos finds videos across arbitrary nested subdirectories."""
        from preprocess import discover_videos

        temp_dir = tempfile.mkdtemp()
        try:
            sub1 = os.path.join(temp_dir, "Celeb-real")
            sub2 = os.path.join(temp_dir, "YouTube-real", "nested")
            os.makedirs(sub1, exist_ok=True)
            os.makedirs(sub2, exist_ok=True)

            open(os.path.join(sub1, "vid1.mp4"), "w").close()
            open(os.path.join(sub2, "vid2.avi"), "w").close()
            open(os.path.join(temp_dir, "ignored.txt"), "w").close()

            found = discover_videos(temp_dir)
            self.assertEqual(len(found), 2)
            basenames = [os.path.basename(p) for p in found]
            self.assertIn("vid1.mp4", basenames)
            self.assertIn("vid2.avi", basenames)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ==============================================================================
# Pytest Fixture-Based Tests for Celeb-DF v2 Preprocessing & Validation
# ==============================================================================

import cv2
import pytest
from preprocess import (
    classify_video_folder,
    discover_videos,
    match_test_list_to_disk,
    parse_celebdf_test_list,
    run_preprocessing,
)
from validate_dataset import (
    extract_identities_from_filename,
    generate_identity_overlap_report,
    validate_processed_dataset,
    validate_raw_dataset,
)


def _make_dummy_video(path: str, num_frames: int = 12, width: int = 32, height: int = 32) -> str:
    """Helper to generate a tiny valid video file with cv2.VideoWriter."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(path)


def test_recursive_discovery_and_classification(tmp_path):
    """Test 1: Recursive discovery finds videos in subdirectories and nested 'videos/' folders,
    deriving labels strictly from folder names.
    """
    raw_dir = tmp_path / "raw"
    v1 = _make_dummy_video(str(raw_dir / "Celeb-real" / "videos" / "id0_0000.mp4"))
    v2 = _make_dummy_video(str(raw_dir / "YouTube-real" / "00000.mp4"))
    v3 = _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "videos" / "id0_id1_0000.mp4"))

    # Recursive video discovery
    discovered = discover_videos(str(raw_dir))
    assert len(discovered) == 3

    # Derive labels strictly from folder name
    cls1 = classify_video_folder("Celeb-real/videos/id0_0000.mp4")
    assert cls1 == ("Celeb-real", 0)  # Real

    cls2 = classify_video_folder("YouTube-real/00000.mp4")
    assert cls2 == ("YouTube-real", 0)  # Real

    cls3 = classify_video_folder("Celeb-synthesis/videos/id0_id1_0000.mp4")
    assert cls3 == ("Celeb-synthesis", 1)  # Fake

    # Validate raw dataset finds and verifies all 3
    stats = validate_raw_dataset(raw_dir=str(raw_dir), check_readability=True)
    assert stats["total_videos"] == 3
    assert stats["by_category"]["Celeb-real"] == 1
    assert stats["by_category"]["YouTube-real"] == 1
    assert stats["by_category"]["Celeb-synthesis"] == 1
    assert stats["corrupted_count"] == 0


def test_case_insensitive_folders(tmp_path):
    """Test 2: Case-insensitive folder matching tolerates arbitrary casing."""
    raw_dir = tmp_path / "raw_mixed"
    _make_dummy_video(str(raw_dir / "celeb-REAL" / "id1_0000.mp4"))
    _make_dummy_video(str(raw_dir / "Youtube-Real" / "videos" / "00001.mp4"))
    _make_dummy_video(str(raw_dir / "Celeb-SYNTHESIS" / "videos" / "id1_id2_0000.mp4"))

    cls1 = classify_video_folder("celeb-REAL/id1_0000.mp4")
    assert cls1 == ("Celeb-real", 0)

    cls2 = classify_video_folder("Youtube-Real/videos/00001.mp4")
    assert cls2 == ("YouTube-real", 0)

    cls3 = classify_video_folder("Celeb-SYNTHESIS/videos/id1_id2_0000.mp4")
    assert cls3 == ("Celeb-synthesis", 1)


def test_official_test_list_parsing_and_matching(tmp_path):
    """Test 3: Official test list parsing by folder + filename matching."""
    raw_dir = tmp_path / "raw"
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"))
    _make_dummy_video(str(raw_dir / "YouTube-real" / "00000.mp4"))
    _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4"))

    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "0 Celeb-real/id0_0000.mp4\n"
        "1 Celeb-synthesis/id0_id1_0000.mp4\n"
    )

    parsed = parse_celebdf_test_list(str(test_list_file))
    assert len(parsed) == 2
    assert parsed[0] == (0, "Celeb-real/id0_0000.mp4", "Celeb-real", "id0_0000.mp4")
    assert parsed[1] == (1, "Celeb-synthesis/id0_id1_0000.mp4", "Celeb-synthesis", "id0_id1_0000.mp4")

    # Match against disk
    video_records = [
        ("Celeb-real/id0_0000.mp4", "Celeb-real", 0, str(raw_dir / "Celeb-real" / "id0_0000.mp4")),
        ("YouTube-real/00000.mp4", "YouTube-real", 0, str(raw_dir / "YouTube-real" / "00000.mp4")),
        ("Celeb-synthesis/id0_id1_0000.mp4", "Celeb-synthesis", 1, str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4")),
    ]
    mapping = match_test_list_to_disk(parsed, video_records, str(raw_dir))
    assert len(mapping) == 2
    assert mapping["Celeb-real/id0_0000.mp4"] == "test"
    assert mapping["Celeb-synthesis/id0_id1_0000.mp4"] == "test"
    assert "YouTube-real/00000.mp4" not in mapping


def test_label_mismatch_detection(tmp_path):
    """Test 4: Cross-check each list entry's numeric label against folder label and fail on mismatch."""
    mismatched_list = tmp_path / "mismatch_test_list.txt"
    # Line 1: numeric label says 1 (fake), but folder Celeb-real indicates 0 (real)
    mismatched_list.write_text("1 Celeb-real/id0_0000.mp4\n")

    with pytest.raises(ValueError) as excinfo:
        parse_celebdf_test_list(str(mismatched_list))
    assert "Label mismatch detected" in str(excinfo.value)
    assert "numeric label is 1, but folder 'Celeb-real' indicates label 0" in str(excinfo.value)

    # Line 2: numeric label says 0 (real), but folder Celeb-synthesis indicates 1 (fake)
    mismatched_list.write_text("0 Celeb-synthesis/id0_id1_0000.mp4\n")
    with pytest.raises(ValueError) as excinfo2:
        parse_celebdf_test_list(str(mismatched_list))
    assert "Label mismatch detected" in str(excinfo2.value)
    assert "numeric label is 0, but folder 'Celeb-synthesis' indicates label 1" in str(excinfo2.value)


def test_missing_file_failure(tmp_path):
    """Test 5: Fail loudly with clear message if any list entry has no file on disk."""
    raw_dir = tmp_path / "raw_missing"
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"))

    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "0 Celeb-real/id0_0000.mp4\n"
        "1 Celeb-synthesis/id99_id99_9999.mp4\n"  # Missing on disk!
    )

    parsed = parse_celebdf_test_list(str(test_list_file))
    video_records = [
        ("Celeb-real/id0_0000.mp4", "Celeb-real", 0, str(raw_dir / "Celeb-real" / "id0_0000.mp4")),
    ]

    with pytest.raises(FileNotFoundError) as excinfo:
        match_test_list_to_disk(parsed, video_records, str(raw_dir))
    assert "do not exist on disk" in str(excinfo.value)
    assert "id99_id99_9999.mp4" in str(excinfo.value)


def test_split_disjointness_and_zero_clip_leakage(tmp_path):
    """Test 6: Video-level split disjointness with official test list, zero clip leakage,
    and no test_fraction usage for test set creation.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    # Create 6 videos (3 real, 3 fake)
    v_real1 = _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"))
    v_real2 = _make_dummy_video(str(raw_dir / "Celeb-real" / "id1_0000.mp4"))
    v_real3 = _make_dummy_video(str(raw_dir / "YouTube-real" / "00000.mp4"))

    v_fake1 = _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4"))
    v_fake2 = _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id2_0000.mp4"))
    v_fake3 = _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id1_id2_0000.mp4"))

    # Test list contains exactly 2 videos: 1 real, 1 fake
    test_list_file = raw_dir / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "0 Celeb-real/id0_0000.mp4\n"
        "1 Celeb-synthesis/id0_id1_0000.mp4\n"
    )

    meta_df = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        test_list_path=str(test_list_file),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        val_fraction=0.33,
        seed=42,
        no_face_detect=True,
    )

    # 1. Total clips = 6 videos * 2 clips = 12 clips
    assert len(meta_df) == 12

    # 2. Test split contains exactly the 2 test list videos
    test_videos = set(meta_df[meta_df["split"] == "test"]["video"].unique())
    train_videos = set(meta_df[meta_df["split"] == "train"]["video"].unique())
    val_videos = set(meta_df[meta_df["split"] == "val"]["video"].unique())

    assert len(test_videos) == 2
    assert "Celeb-real/id0_0000.mp4" in test_videos
    assert "Celeb-synthesis/id0_id1_0000.mp4" in test_videos

    # 3. Splits are strictly pairwise disjoint at the video level
    assert len(train_videos.intersection(val_videos)) == 0
    assert len(train_videos.intersection(test_videos)) == 0
    assert len(val_videos.intersection(test_videos)) == 0

    # 4. Zero clip leakage into train/val
    test_clips_in_train_or_val = meta_df[
        (meta_df["video"].isin(test_videos)) & (meta_df["split"].isin(["train", "val"]))
    ]
    assert len(test_clips_in_train_or_val) == 0

    # 5. Run validate_processed_dataset and confirm it passes
    stats = validate_processed_dataset(
        processed_dir=str(processed_dir),
        expected_seq_len=6,
        expected_img_size=32,
        check_all_clips=True,
    )
    assert stats["leakage_passed"] is True
    assert stats["test_videos"] == 2


def test_celebrity_identity_overlap_report():
    """Test 7: Informational celebrity identity extraction and overlap reporting."""
    # Filename parsing
    assert extract_identities_from_filename("id0_0000.mp4") == {"id0"}
    assert extract_identities_from_filename("id0_id1_0000.mp4") == {"id0", "id1"}
    assert extract_identities_from_filename("00170.mp4") == set()

    # Synthetic meta dataframe with known identity overlap
    meta_df = pd.DataFrame([
        {"path": "c1.npy", "label": 0, "video": "Celeb-real/id0_0000.mp4", "clip": 0, "split": "train"},
        {"path": "c2.npy", "label": 0, "video": "Celeb-real/id1_0000.mp4", "clip": 0, "split": "val"},
        {"path": "c3.npy", "label": 1, "video": "Celeb-synthesis/id0_id2_0000.mp4", "clip": 0, "split": "test"},
    ])

    report = generate_identity_overlap_report(meta_df)
    assert "id0" in report["train_identities"]
    assert "id1" in report["val_identities"]
    assert "id0" in report["test_identities"]
    assert "id2" in report["test_identities"]
    assert "id0" in report["train_test_overlap"]
    assert len(report["val_test_overlap"]) == 0


if __name__ == "__main__":
    unittest.main()


