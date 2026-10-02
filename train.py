"""Training script for deepfake detection model.

Trains CNN-LSTM network using tf.data pipeline with:
- Adam optimizer and binary crossentropy loss.
- Accuracy and AUC metrics.
- Dynamic balanced class weights from training labels.
- Strict val_loss monitoring across ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau.
- Saves best_model.keras, history.json, and train_config.json.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import random
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from dataset import compute_class_weights, get_dataset
from model import build_model


def get_git_info() -> Tuple[str, bool]:
    """Retrieve current git commit hash and dirty status."""
    commit_hash = "unknown"
    is_dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hash = res.stdout.strip()
    except Exception:
        pass

    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = len(res.stdout.strip()) > 0
    except Exception:
        pass

    return commit_hash, is_dirty


def compute_split_counts(meta_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Compute per-split clip and video counts by class."""
    counts = {}
    for split_name in ["train", "val", "test"]:
        sub = meta_df[meta_df["split"].str.lower() == split_name]
        if len(sub) == 0:
            counts[split_name] = {
                "clips": {"real": 0, "fake": 0, "total": 0},
                "videos": {"real": 0, "fake": 0, "total": 0},
            }
            continue

        real_clips = int((sub["label"] == 0).sum())
        fake_clips = int((sub["label"] == 1).sum())
        total_clips = int(len(sub))

        real_vids = int(sub[sub["label"] == 0]["video"].nunique())
        fake_vids = int(sub[sub["label"] == 1]["video"].nunique())
        total_vids = int(sub["video"].nunique())

        counts[split_name] = {
            "clips": {"real": real_clips, "fake": fake_clips, "total": total_clips},
            "videos": {"real": real_vids, "fake": fake_vids, "total": total_vids},
        }
    return counts


def save_run_info(
    output_dir: str,
    meta_df: pd.DataFrame,
    config: Config,
    seed: int,
    class_weights: Dict[int, float],
    backbone: str,
    dataset_path: str,
) -> str:
    """Save comprehensive experiment run information to run_info.json."""
    os.makedirs(output_dir, exist_ok=True)
    commit_hash, is_dirty = get_git_info()
    timestamp = datetime.now(timezone.utc).isoformat()

    run_info = {
        "git_commit": commit_hash,
        "git_dirty": is_dirty,
        "timestamp": timestamp,
        "versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "opencv": cv2.__version__,
        },
        "config": config.to_dict(),
        "seed": seed,
        "split_counts": compute_split_counts(meta_df),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "backbone": backbone,
        "dataset_path": dataset_path,
    }

    run_info_path = os.path.join(output_dir, "run_info.json")
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    return run_info_path


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



def train_model(
    processed_dir: str = "data/processed",
    output_dir: str = "outputs",
    seq_len: int = 10,
    img_size: int = 128,
    backbone: str = "light",
    pretrained: bool = False,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    lstm_units: int = 128,
    dropout: float = 0.4,
    seed: int = 42,
    patience: int = 5,
    verbose: int = 2,
) -> Dict[str, Any]:
    """Train the deepfake detection model.

    Args:
        processed_dir: Directory containing meta.csv and preprocessed clips.
        output_dir: Directory to save best_model.keras and history.
        seq_len: Frames per clip.
        img_size: Frame spatial resolution.
        backbone: Backbone feature extractor ("light" or "mobilenetv2").
        pretrained: Whether to use ImageNet weights for MobileNetV2.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        lr: Initial learning rate for Adam optimizer.
        lstm_units: Hidden units for LSTM layer.
        dropout: Dropout rate.
        seed: Random seed.
        patience: Early stopping patience epochs.
        verbose: Verbosity level for model.fit.

    Returns:
        Dict with paths to saved artifacts and training history.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Pre-training validation
    from validate_dataset import validate_processed_dataset
    print("Validating processed dataset integrity before training...")
    validate_processed_dataset(
        processed_dir=processed_dir,
        expected_seq_len=seq_len,
        expected_img_size=img_size,
        check_all_clips=False,
    )

    meta_path = os.path.join(processed_dir, "meta.csv")
    meta_df = pd.read_csv(meta_path)
    train_df = meta_df[meta_df["split"].str.lower() == "train"]
    val_df = meta_df[meta_df["split"].str.lower() == "val"]

    print(f"Loaded {len(train_df)} training clips and {len(val_df)} validation clips from {meta_path}.")

    if len(train_df) == 0:
        raise ValueError("No training samples found in meta.csv for split='train'.")

    # Compute dynamic balanced class weights from training labels
    class_weight_dict = compute_class_weights(train_df["label"].values)
    print(f"Computed training class weights: {class_weight_dict}")

    # Save outputs/run_info.json
    cfg = Config(
        seq_len=seq_len,
        img_size=img_size,
        backbone=backbone,
        pretrained=pretrained,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        lstm_units=lstm_units,
        dropout=dropout,
        seed=seed,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )
    run_info_path = save_run_info(
        output_dir=output_dir,
        meta_df=meta_df,
        config=cfg,
        seed=seed,
        class_weights=class_weight_dict,
        backbone=backbone,
        dataset_path=processed_dir,
    )
    print(f"Saved run metadata to: {run_info_path}")

    # Build datasets
    train_ds, n_train = get_dataset(
        meta_path=meta_path,
        split="train",
        processed_dir=processed_dir,
        batch_size=batch_size,
        seq_len=seq_len,
        img_size=img_size,
        is_training=True,
        seed=seed,
    )

    has_val = len(val_df) > 0
    val_ds = None
    if has_val:
        val_ds, n_val = get_dataset(
            meta_path=meta_path,
            split="val",
            processed_dir=processed_dir,
            batch_size=batch_size,
            seq_len=seq_len,
            img_size=img_size,
            is_training=False,
            seed=seed,
        )

    # Build and compile model
    model = build_model(
        seq_len=seq_len,
        img_size=img_size,
        backbone=backbone,
        pretrained=pretrained,
        lstm_units=lstm_units,
        dropout=dropout,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    best_model_path = os.path.join(output_dir, "best_model.keras")

    # Callbacks: ALL THREE MUST MONITOR val_loss (mode='min') when validation data is available
    callbacks = []
    monitor_metric = "val_loss" if has_val else "loss"

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor=monitor_metric,
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    callbacks.append(checkpoint_cb)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        mode="min",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    callbacks.append(early_stopping_cb)

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        mode="min",
        factor=0.5,
        patience=max(2, patience // 2),
        min_lr=1e-6,
        verbose=1,
    )
    callbacks.append(reduce_lr_cb)

    print(f"\nStarting training for {epochs} epochs (monitoring '{monitor_metric}', mode='min')...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Ensure best_model.keras exists even if checkpoint callback did not trigger save
    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    # Save history.json
    history_path = os.path.join(output_dir, "history.json")
    clean_history = {}
    for k, v in history.history.items():
        clean_history[k] = [float(x) for x in v]
    with open(history_path, "w") as f:
        json.dump(clean_history, f, indent=2)

    # Save train_config.json
    config_dict = {
        "processed_dir": processed_dir,
        "output_dir": output_dir,
        "seq_len": seq_len,
        "img_size": img_size,
        "backbone": backbone,
        "pretrained": pretrained,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lstm_units": lstm_units,
        "dropout": dropout,
        "seed": seed,
        "class_weights": class_weight_dict,
        "best_model_path": best_model_path,
    }
    train_config_path = os.path.join(output_dir, "train_config.json")
    with open(train_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nTraining completed.")
    print(f"  - Model saved to:  {best_model_path}")
    print(f"  - History saved:   {history_path}")
    print(f"  - Config saved:    {train_config_path}")

    return {
        "best_model_path": best_model_path,
        "history_path": history_path,
        "train_config_path": train_config_path,
        "run_info_path": run_info_path,
        "history": clean_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train deepfake detection model.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save model outputs.")
    parser.add_argument("--seq-len", type=int, default=10, help="Frames per sequence clip.")
    parser.add_argument("--img-size", type=int, default=128, help="Frame resolution (H=W).")
    parser.add_argument("--backbone", type=str, default="light", choices=["light", "mobilenetv2"], help="CNN backbone.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights for MobileNetV2.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    train_model(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        img_size=args.img_size,
        backbone=args.backbone,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
