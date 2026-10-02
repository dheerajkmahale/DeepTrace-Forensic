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


if __name__ == "__main__":
    unittest.main()

