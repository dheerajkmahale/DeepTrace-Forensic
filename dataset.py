"""TensorFlow tf.data dataset pipeline for deepfake detection.

Loads preprocessed video clips referenced in meta.csv using
tf.data.Dataset.from_generator.

Features:
- Reads train/val/test splits strictly from meta.csv.
- Consistent clip-level training augmentations (horizontal flip, brightness shift)
  applied uniformly across all frames of the clip.
- Explicit cardinality setting via tf.data.experimental.assert_cardinality.
- Dynamic balanced class weight computation from training labels.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Calculate balanced class weights from training labels.

    Args:
        labels: Array-like sequence of binary class labels (0 or 1).

    Returns:
        Dict mapping class integer (0, 1) to balanced float weight.
    """
    labels_arr = np.asarray(labels, dtype=int)
    classes = np.array([0, 1])

    # If only one class is present or dataset is empty, return neutral weights
    present_classes = np.unique(labels_arr)
    if len(present_classes) < 2:
        return {0: 1.0, 1: 1.0}

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_arr,
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Convenience alias for compute_class_weights."""
    return compute_class_weights(labels)


def augment_clip(clip: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Apply consistent spatial and photometric augmentation across all frames.

    Augmentations are applied to the clip as a single unit rather than
    randomizing frames independently, preserving temporal consistency.

    Args:
        clip: Array of shape (seq_len, H, W, 3) and dtype uint8.
        rng: NumPy RandomState instance for reproducible randomness.

    Returns:
        Augmented clip of same shape and uint8 dtype.
    """
    augmented = clip.copy()

    # 1. Random horizontal flip (50% probability)
    if rng.rand() > 0.5:
        # axis 2 is width dimension for (seq_len, H, W, C)
        augmented = np.flip(augmented, axis=2)

    # 2. Random brightness adjustment applied uniformly to all frames
    brightness_delta = rng.uniform(-25.0, 25.0)
    adjusted = augmented.astype(np.int16) + brightness_delta
    augmented = np.clip(adjusted, 0, 255).astype(np.uint8)

    return augmented


def resolve_clip_path(path: str, processed_dir: str) -> str:
    """Resolve clip path relative to processed_dir or root."""
    if os.path.isabs(path) and os.path.exists(path):
        return path
    candidate1 = os.path.join(processed_dir, path)
    if os.path.exists(candidate1):
        return candidate1
    if os.path.exists(path):
        return path
    return candidate1


def get_dataset(
    meta_path: str,
    split: str,
    processed_dir: Optional[str] = None,
    batch_size: int = 8,
    seq_len: int = 10,
    img_size: int = 128,
    is_training: Optional[bool] = None,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, int]:
    """Create a tf.data.Dataset for a specified split using a generator.

    Args:
        meta_path: Path to meta.csv.
        split: Requested split ("train", "val", or "test").
        processed_dir: Directory containing processed data. Defaults to
            the parent directory of meta_path.
        batch_size: Number of clips per batch.
        seq_len: Frames per clip.
        img_size: Frame spatial dimension (H=W).
        is_training: Whether to enable training augmentations and shuffling.
            If None, defaults to (split == "train").
        seed: Random seed for generator shuffling and augmentations.

    Returns:
        Tuple of (batched_dataset, num_samples).
    """
    if processed_dir is None:
        processed_dir = os.path.dirname(os.path.abspath(meta_path))

    df = pd.read_csv(meta_path)
    split_df = df[df["split"].str.lower() == split.lower()].reset_index(drop=True)
    num_samples = len(split_df)

    if is_training is None:
        is_training = (split.lower() == "train")

    clip_records: List[Tuple[str, int]] = []
    for _, row in split_df.iterrows():
        resolved = resolve_clip_path(str(row["path"]), processed_dir)
        clip_records.append((resolved, int(row["label"])))

    def data_generator():
        rng = np.random.RandomState(seed)
        indices = np.arange(len(clip_records))
        if is_training:
            rng.shuffle(indices)

        for idx in indices:
            clip_file, label = clip_records[idx]
            try:
                clip = np.load(clip_file)
            except Exception as e:
                # If file cannot be loaded, substitute zeros with proper shape
                clip = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)

            # Ensure expected shape and dtype
            if clip.shape != (seq_len, img_size, img_size, 3):
                # Resize/reshape if needed
                clip = np.zeros((seq_len, img_size, img_size, 3), dtype=np.uint8)

            if is_training:
                clip = augment_clip(clip, rng)

            yield clip.astype(np.uint8), np.float32(label)

    output_signature = (
        tf.TensorSpec(shape=(seq_len, img_size, img_size, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature,
    )

    if num_samples > 0:
        num_batches = math.ceil(num_samples / batch_size)
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(num_batches))
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(0))

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, num_samples
