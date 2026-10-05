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
    OFFICIAL_LIST_LABELS,
    classify_video_folder,
    convert_official_list_label_to_project_label,
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


def test_official_label_conversion_logic():
    """Verify conversion from official convention (1=real, 0=fake) to project convention (0=real, 1=fake)."""
    assert OFFICIAL_LIST_LABELS["real"] == 1
    assert OFFICIAL_LIST_LABELS["fake"] == 0

    assert convert_official_list_label_to_project_label(1) == 0  # real
    assert convert_official_list_label_to_project_label(0) == 1  # fake

    with pytest.raises(ValueError) as excinfo:
        convert_official_list_label_to_project_label(2)
    assert "Unrecognized official test-list numeric label: 2" in str(excinfo.value)


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
    """Test 3: Official test list parsing by folder + filename matching using official convention (1=real, 0=fake)."""
    raw_dir = tmp_path / "raw"
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"))
    _make_dummy_video(str(raw_dir / "YouTube-real" / "00000.mp4"))
    _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4"))

    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "1 Celeb-real/id0_0000.mp4\n"
        "0 Celeb-synthesis/id0_id1_0000.mp4\n"
    )

    parsed = parse_celebdf_test_list(str(test_list_file))
    assert len(parsed) == 2
    assert parsed[0] == (1, "Celeb-real/id0_0000.mp4", "Celeb-real", "id0_0000.mp4")
    assert parsed[1] == (0, "Celeb-synthesis/id0_id1_0000.mp4", "Celeb-synthesis", "id0_id1_0000.mp4")

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


def test_official_convention_test_list_parses(tmp_path):
    """(a) Test that a correct official-convention list (1=real, 0=fake) parses successfully."""
    list_file = tmp_path / "official_test_list.txt"
    list_file.write_text(
        "1 YouTube-real/00170.mp4\n"
        "1 Celeb-real/id0_0000.mp4\n"
        "0 Celeb-synthesis/id0_id1_0000.mp4\n"
    )
    parsed = parse_celebdf_test_list(str(list_file))
    assert len(parsed) == 3
    assert parsed[0][0] == 1  # 1 in list -> real
    assert parsed[0][2] == "YouTube-real"
    assert parsed[1][0] == 1  # 1 in list -> real
    assert parsed[1][2] == "Celeb-real"
    assert parsed[2][0] == 0  # 0 in list -> fake
    assert parsed[2][2] == "Celeb-synthesis"


def test_project_convention_test_list_rejected(tmp_path):
    """(b) Test that a list using the project's internal convention (0=real, 1=fake) is rejected."""
    list_file = tmp_path / "project_convention_list.txt"
    # Project convention uses 0=real and 1=fake. Under official list assumption (1=real, 0=fake),
    # this must be caught and rejected with clear line numbers.
    list_file.write_text(
        "0 YouTube-real/00170.mp4\n"
        "1 Celeb-synthesis/id0_id1_0000.mp4\n"
    )
    with pytest.raises(ValueError) as excinfo:
        parse_celebdf_test_list(str(list_file))
    err_msg = str(excinfo.value)
    assert "Label mismatch detected in official test list" in err_msg
    assert "Line 1:" in err_msg
    assert "official list label is 0" in err_msg
    assert "Line 2:" in err_msg
    assert "official list label is 1" in err_msg


def test_single_wrong_entry_rejected(tmp_path):
    """(c) Test that a list with even a single wrong entry is rejected with line number."""
    list_file = tmp_path / "single_wrong_entry_list.txt"
    list_file.write_text(
        "1 YouTube-real/00170.mp4\n"
        "0 Celeb-real/id0_0000.mp4\n"  # Wrong: Celeb-real is real, so should be 1 under official list convention
        "0 Celeb-synthesis/id0_id1_0000.mp4\n"
    )
    with pytest.raises(ValueError) as excinfo:
        parse_celebdf_test_list(str(list_file))
    err_msg = str(excinfo.value)
    assert "Label mismatch detected in official test list" in err_msg
    assert "Line 2:" in err_msg
    assert "Celeb-real" in err_msg


def test_missing_file_failure(tmp_path):
    """Test 5: Fail loudly with clear message if any list entry has no file on disk."""
    raw_dir = tmp_path / "raw_missing"
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"))

    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "1 Celeb-real/id0_0000.mp4\n"
        "0 Celeb-synthesis/id99_id99_9999.mp4\n"  # Missing on disk!
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

    # Test list contains exactly 2 videos: 1 real (label 1), 1 fake (label 0)
    test_list_file = raw_dir / "List_of_testing_videos.txt"
    test_list_file.write_text(
        "1 Celeb-real/id0_0000.mp4\n"
        "0 Celeb-synthesis/id0_id1_0000.mp4\n"
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


def test_resume_skips_finished_videos(tmp_path):
    """Test resume functionality: existing readable clips are skipped and not reprocessed."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=12)
    _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4"), num_frames=12)

    meta_df1 = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
        force=False,
    )
    assert len(meta_df1) == 4

    clips_dir = processed_dir / "clips"
    clip_files = list(clips_dir.glob("*.npy"))
    assert len(clip_files) == 4
    mtimes_before = {p: p.stat().st_mtime_ns for p in clip_files}

    # Run again without force (should resume and skip extraction)
    meta_df2 = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
        force=False,
    )
    assert len(meta_df2) == 4
    pd.testing.assert_frame_equal(meta_df1, meta_df2)

    mtimes_after = {p: p.stat().st_mtime_ns for p in clip_files}
    assert mtimes_before == mtimes_after, "Clips should not be rewritten when resuming"


def test_force_reprocesses_all_videos(tmp_path):
    """Test --force flag forces reprocessing and rewriting of all clips."""
    import time
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=12)

    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
    )

    clips_dir = processed_dir / "clips"
    clip_files = list(clips_dir.glob("*.npy"))
    assert len(clip_files) == 2
    mtimes_before = {p: p.stat().st_mtime for p in clip_files}

    time.sleep(1.1)

    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
        force=True,
    )

    mtimes_after = {p: p.stat().st_mtime for p in clip_files}
    for p in clip_files:
        assert mtimes_after[p] > mtimes_before[p], "Clip file should be updated with force=True"


def test_atomic_meta_csv_write(tmp_path):
    """Test that meta.csv is written atomically and deterministically with no lingering tmp files."""
    from preprocess import save_meta_csv_atomically
    meta_path = str(tmp_path / "meta.csv")
    rows = [
        {"path": "clips/vid_b_clip0.npy", "label": 1, "video": "vid_b", "clip": 0, "split": "train"},
        {"path": "clips/vid_a_clip1.npy", "label": 0, "video": "vid_a", "clip": 1, "split": "val"},
        {"path": "clips/vid_a_clip0.npy", "label": 0, "video": "vid_a", "clip": 0, "split": "train"},
    ]

    df = save_meta_csv_atomically(rows, meta_path)
    assert os.path.exists(meta_path)
    assert not os.path.exists(f"{meta_path}.tmp")

    # Verify sorting by video, clip
    loaded = pd.read_csv(meta_path)
    assert list(loaded["video"]) == ["vid_a", "vid_a", "vid_b"]
    assert list(loaded["clip"]) == [0, 1, 0]


def test_skipped_video_log_contents(tmp_path):
    """Test that unreadable and too short videos are logged to skipped CSV."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    skipped_csv = tmp_path / "outputs" / "preprocess_skipped.csv"

    # 1. Valid video
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=12)

    # 2. Too short video (3 frames when seq_len=6)
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id1_short.mp4"), num_frames=3)

    # 3. Unreadable corrupt video
    corrupt_path = raw_dir / "Celeb-synthesis" / "corrupt_vid.mp4"
    os.makedirs(corrupt_path.parent, exist_ok=True)
    corrupt_path.write_bytes(b"NOT_A_VALID_VIDEO_FILE_CONTENT")

    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        skipped_log_path=str(skipped_csv),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
    )

    assert skipped_csv.exists()
    skipped_df = pd.read_csv(str(skipped_csv))
    assert len(skipped_df) == 2
    reasons = set(skipped_df["reason"].unique())
    assert "too short" in reasons
    assert "unreadable" in reasons


def test_identical_output_for_workers(tmp_path):
    """Test that workers=1 and workers=2 produce identical splits, clips, and meta.csv ordering."""
    raw_dir = tmp_path / "raw"
    out_w1 = tmp_path / "processed_w1"
    out_w2 = tmp_path / "processed_w2"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=12)
    _make_dummy_video(str(raw_dir / "Celeb-real" / "id1_0000.mp4"), num_frames=12)
    _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id0_id1_0000.mp4"), num_frames=12)
    _make_dummy_video(str(raw_dir / "Celeb-synthesis" / "id1_id2_0000.mp4"), num_frames=12)

    meta_w1 = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(out_w1),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        workers=1,
        no_face_detect=True,
        seed=42,
    )

    meta_w2 = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(out_w2),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        workers=2,
        no_face_detect=True,
        seed=42,
    )

    # 1. Exact DataFrame equality (same rows, columns, split assignments, order)
    pd.testing.assert_frame_equal(meta_w1, meta_w2)

    # 2. Exact clip array equality for all clips
    for _, row in meta_w1.iterrows():
        p1 = out_w1 / row["path"]
        p2 = out_w2 / row["path"]
        arr1 = np.load(str(p1))
        arr2 = np.load(str(p2))
        assert np.array_equal(arr1, arr2)


def test_run_info_fields(tmp_path):
    """Test that save_run_info generates run_info.json containing all required experiment metadata."""
    import json
    from train import save_run_info
    from config import Config

    output_dir = tmp_path / "outputs"
    meta_df = pd.DataFrame([
        {"path": "clips/c1.npy", "label": 0, "video": "v1", "clip": 0, "split": "train"},
        {"path": "clips/c2.npy", "label": 1, "video": "v2", "clip": 0, "split": "train"},
        {"path": "clips/c3.npy", "label": 0, "video": "v3", "clip": 0, "split": "val"},
        {"path": "clips/c4.npy", "label": 1, "video": "v4", "clip": 0, "split": "test"},
    ])

    cfg = Config()
    run_info_path = save_run_info(
        output_dir=str(output_dir),
        meta_df=meta_df,
        config=cfg,
        seed=42,
        class_weights={0: 1.0, 1: 1.5},
        backbone="light",
        dataset_path=str(tmp_path / "data" / "processed"),
    )

    assert os.path.exists(run_info_path)
    with open(run_info_path, "r") as f:
        data = json.load(f)

    # Check all required fields
    required_keys = [
        "git_commit",
        "git_dirty",
        "timestamp",
        "versions",
        "config",
        "seed",
        "split_counts",
        "class_weights",
        "backbone",
        "dataset_path",
    ]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in run_info.json"

    # Versions
    assert "python" in data["versions"]
    assert "tensorflow" in data["versions"]
    assert "opencv" in data["versions"]

    # Split counts
    assert "train" in data["split_counts"]
    assert "val" in data["split_counts"]
    assert "test" in data["split_counts"]
    assert data["split_counts"]["train"]["clips"]["real"] == 1
    assert data["split_counts"]["train"]["clips"]["fake"] == 1
    assert data["split_counts"]["test"]["videos"]["fake"] == 1


def test_evaluate_test_counts_and_val_threshold(tmp_path):
    """Test that evaluate records test_counts and threshold in metrics.json, and threshold is never chosen on test split."""
    import json
    from evaluate import evaluate
    from model import build_model

    processed_dir = tmp_path / "processed"
    clips_dir = processed_dir / "clips"
    output_dir = tmp_path / "outputs"
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    seq_len = 4
    img_size = 32

    # Create dummy clips
    c_test_real = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)
    c_test_fake = np.full((seq_len, img_size, img_size, 3), 200, dtype=np.uint8)
    c_val = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)

    np.save(str(clips_dir / "test_real.npy"), c_test_real)
    np.save(str(clips_dir / "test_fake.npy"), c_test_fake)
    np.save(str(clips_dir / "val_vid.npy"), c_val)

    meta_df = pd.DataFrame([
        {"path": "clips/test_real.npy", "label": 0, "video": "v_real", "clip": 0, "split": "test"},
        {"path": "clips/test_fake.npy", "label": 1, "video": "v_fake", "clip": 0, "split": "test"},
        {"path": "clips/val_vid.npy", "label": 0, "video": "v_val", "clip": 0, "split": "val"},
    ])
    meta_df.to_csv(str(processed_dir / "meta.csv"), index=False)

    model = build_model(seq_len=seq_len, img_size=img_size, backbone="light", lstm_units=16, dropout=0.1)
    model_path = str(output_dir / "test_eval_model.keras")
    model.save(model_path)

    # 1. Run evaluate with default threshold 0.5
    res1 = evaluate(
        model_path=model_path,
        processed_dir=str(processed_dir),
        output_dir=str(output_dir),
        threshold=0.5,
    )

    metrics_file = output_dir / "metrics.json"
    assert metrics_file.exists()
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    assert "test_counts" in metrics_data
    assert metrics_data["test_counts"]["clips"]["real"] == 1
    assert metrics_data["test_counts"]["clips"]["fake"] == 1
    assert metrics_data["test_counts"]["videos"]["real"] == 1
    assert metrics_data["test_counts"]["videos"]["fake"] == 1
    assert metrics_data["threshold"] == 0.5

    # 2. Run evaluate with val_threshold (chosen on validation split, never test)
    res2 = evaluate(
        model_path=model_path,
        processed_dir=str(processed_dir),
        output_dir=str(output_dir),
        val_threshold=True,
    )
    with open(metrics_file, "r") as f:
        metrics_data_val = json.load(f)

    assert metrics_data_val["threshold_source"] == "validation_split"
    assert "test_counts" in metrics_data_val


def test_preprocessing_fingerprint_same_settings_resumes(tmp_path):
    """Test that same settings generate and verify fingerprint, resuming without error."""
    import json
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=12)

    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        face_margin=0.25,
        seed=42,
        no_face_detect=True,
    )

    config_file = processed_dir / "preprocess_config.json"
    assert config_file.exists()
    with open(config_file, "r") as f:
        cfg1 = json.load(f)

    assert "fingerprint" in cfg1
    assert cfg1["seq_len"] == 6
    assert cfg1["img_size"] == 32
    assert cfg1["clips_per_video"] == 2
    assert cfg1["face_margin"] == 0.25
    assert cfg1["seed"] == 42
    assert cfg1["face_detector"]["name"] == "center_crop"

    # Resume with identical settings
    meta_df = run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        face_margin=0.25,
        seed=42,
        no_face_detect=True,
        force=False,
    )
    assert len(meta_df) == 2


def test_preprocessing_fingerprint_changed_setting_refused(tmp_path):
    """Test that resuming with a changed setting is refused with a clear error listing differences."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=16)

    # Initial run with seq_len=6, img_size=32
    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        face_margin=0.25,
        seed=42,
        no_face_detect=True,
    )

    # Attempt resume with different seq_len (8 instead of 6) and different img_size (48 instead of 32)
    with pytest.raises(ValueError) as excinfo:
        run_preprocessing(
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            seq_len=8,
            img_size=48,
            clips_per_video=2,
            face_margin=0.25,
            seed=42,
            no_face_detect=True,
            force=False,
        )

    err_msg = str(excinfo.value)
    assert "Preprocessing configuration mismatch detected" in err_msg
    assert "seq_len: existing=6, current=8" in err_msg
    assert "img_size: existing=32, current=48" in err_msg
    assert "--force" in err_msg


def test_preprocessing_fingerprint_force_overwrites(tmp_path):
    """Test that --force overwrites preprocess_config.json with new settings and fingerprint."""
    import json
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    _make_dummy_video(str(raw_dir / "Celeb-real" / "id0_0000.mp4"), num_frames=16)

    # Initial run with seq_len=6
    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=6,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
    )

    config_file = processed_dir / "preprocess_config.json"
    with open(config_file, "r") as f:
        fp1 = json.load(f)["fingerprint"]

    # Reprocess with --force and seq_len=8
    run_preprocessing(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        seq_len=8,
        img_size=32,
        clips_per_video=2,
        no_face_detect=True,
        force=True,
    )

    with open(config_file, "r") as f:
        cfg2 = json.load(f)

    assert cfg2["seq_len"] == 8
    assert cfg2["fingerprint"] != fp1


def test_check_dataset_readiness(tmp_path):
    """Test check_dataset_readiness on empty and populated layouts."""
    from validate_dataset import check_dataset_readiness

    # 1. On empty temp directory, should be blocked
    res_empty = check_dataset_readiness(raw_dir=str(tmp_path))
    assert res_empty["is_ready"] is False
    assert len(res_empty["missing_items"]) > 0

    # 2. Populate required directories
    (tmp_path / "real" / "Celeb-real").mkdir(parents=True)
    (tmp_path / "real" / "YouTube-real").mkdir(parents=True)
    (tmp_path / "fake" / "Celeb-synthesis").mkdir(parents=True)

    res_partial = check_dataset_readiness(raw_dir=str(tmp_path))
    assert res_partial["is_ready"] is False
    assert res_partial["test_list_exists"] is False

    # 3. Create dummy videos and test list
    _make_dummy_video(str(tmp_path / "real" / "Celeb-real" / "id0_0000.mp4"))
    _make_dummy_video(str(tmp_path / "real" / "YouTube-real" / "00000.mp4"))
    _make_dummy_video(str(tmp_path / "fake" / "Celeb-synthesis" / "id0_id1_0000.mp4"))

    test_list_file = tmp_path / "List_of_testing_videos.txt"
    test_list_file.write_text("1 Celeb-real/id0_0000.mp4\n")

    res_ready = check_dataset_readiness(raw_dir=str(tmp_path))
    assert res_ready["is_ready"] is True
    assert res_ready["total_videos"] == 3
    assert res_ready["readability_passed"] is True
    assert res_ready["test_list_resolved"] is True
    assert res_ready["leakage_passed"] is True


if __name__ == "__main__":
    unittest.main()



