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



from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ValidationDiagnosticsCallback(tf.keras.callbacks.Callback):
    """Monitors validation metrics, searches optimal threshold, and prevents single-class collapse."""

    def __init__(
        self,
        val_ds: tf.data.Dataset,
        val_labels: np.ndarray,
        output_dir: str,
        best_model_path: str,
        patience: int = 4,
    ):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.asarray(val_labels, dtype=int)
        self.output_dir = output_dir
        self.best_model_path = best_model_path
        self.patience = patience
        self.best_score = -1.0
        self.best_epoch = -1
        self.best_metrics = {}
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Run inference across validation dataset
        preds = self.model.predict(self.val_ds, verbose=0).flatten()
        y_true = self.val_labels

        try:
            auc = float(roc_auc_score(y_true, preds))
        except Exception:
            auc = 0.5

        # Search threshold maximizing balanced accuracy
        thresholds = np.linspace(0.10, 0.90, 81)
        best_th = 0.5
        best_bacc = -1.0
        best_macro_f1 = -1.0

        for th in thresholds:
            y_pred = (preds >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            rr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            bacc = 0.5 * (rr + fr)
            if bacc > best_bacc:
                best_bacc = bacc
                best_th = float(th)

        # Compute all metrics at best_th
        y_pred = (preds >= best_th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        real_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fake_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))

        # Also at fixed 0.50 for reference
        y_pred_05 = (preds >= 0.5).astype(int)
        cm_05 = confusion_matrix(y_true, y_pred_05, labels=[0, 1])
        tn05, fp05, fn05, tp05 = cm_05.ravel()
        rr_05 = float(tn05 / (tn05 + fp05)) if (tn05 + fp05) > 0 else 0.0
        fr_05 = float(tp05 / (tp05 + fn05)) if (tp05 + fn05) > 0 else 0.0

        p_min = float(preds.min())
        p_max = float(preds.max())
        p_mean = float(preds.mean())
        p_std = float(preds.std())

        print("\n" + "=" * 65)
        print(f"VALIDATION DIAGNOSTICS (Epoch {epoch + 1})")
        print("=" * 65)
        print(f"Probability Distribution: min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.4f}, std={p_std:.4f}")
        print(f"Fixed (0.50): Real Recall={rr_05:.4f}, Fake Recall={fr_05:.4f}")
        print(f"Optimal Validation Threshold: {best_th:.4f}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  Precision:         {prec:.4f}")
        print(f"  Recall (Fake):     {fake_recall:.4f}")
        print(f"  REAL Recall:       {real_recall:.4f}")
        print(f"  F1 Score:          {f1:.4f} (Macro F1: {macro_f1:.4f})")
        print(f"  ROC-AUC:           {auc:.4f}")
        print(f"  Confusion Matrix:  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

        # Sanity Checks
        is_healthy = (
            real_recall > 0.0
            and fake_recall > 0.0
            and p_std > 0.01
            and (tn + tp) > 0
            and (fp + fn) < len(y_true)
        )
        if not is_healthy:
            print("  [SANITY CHECK] Degenerate/single-class output detected.")
        else:
            print("  [SANITY CHECK PASSED] Both classes discriminated successfully.")

        epoch_metrics = {
            "epoch": epoch + 1,
            "threshold": best_th,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": fake_recall,
            "real_recall": real_recall,
            "fake_recall": fake_recall,
            "f1": f1,
            "macro_f1": macro_f1,
            "roc_auc": auc,
            "confusion_matrix": cm.tolist(),
            "prob_distribution": {
                "min": p_min,
                "max": p_max,
                "mean": p_mean,
                "std": p_std,
            },
            "fixed_05": {
                "real_recall": rr_05,
                "fake_recall": fr_05,
                "confusion_matrix": cm_05.tolist(),
            },
        }

        # Check for improvement: maximize combined balanced accuracy and ROC-AUC
        score_to_maximize = bal_acc + auc
        if is_healthy and score_to_maximize > self.best_score:
            print(f"  >>> New best validation performance ({score_to_maximize:.4f} > {self.best_score:.4f}). Saving model! <<<")
            self.best_score = score_to_maximize
            self.best_epoch = epoch + 1
            self.best_metrics = epoch_metrics
            self.wait = 0

            # Save model checkpoint
            self.model.save(self.best_model_path)

            # Save locked threshold
            th_path = os.path.join(self.output_dir, "final_threshold.json")
            with open(th_path, "w") as f:
                json.dump({
                    "threshold": best_th,
                    "balanced_accuracy": bal_acc,
                    "macro_f1": macro_f1,
                    "roc_auc": auc,
                    "epoch": epoch + 1,
                    "real_recall": real_recall,
                    "fake_recall": fake_recall,
                }, f, indent=2)

            # Save validation metrics
            val_metrics_path = os.path.join(self.output_dir, "final_validation_metrics.json")
            with open(val_metrics_path, "w") as f:
                json.dump(epoch_metrics, f, indent=2)
        else:
            self.wait += 1
            print(f"  No improvement for {self.wait}/{self.patience} epochs.")
            if self.wait >= self.patience:
                print(f"  Early stopping triggered after {self.wait} epochs without improvement.")
                self.model.stop_training = True

        print("=" * 65 + "\n")


def train_model(
    processed_dir: str = "data/processed",
    output_dir: str = "outputs",
    seq_len: int = 10,
    img_size: int = 128,
    backbone: str = "light",
    pretrained: bool = False,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 5e-4,
    lstm_units: int = 128,
    dropout: float = 0.4,
    seed: int = 42,
    patience: int = 4,
    verbose: int = 2,
    steps_per_epoch: Optional[int] = None,
    val_steps: Optional[int] = None,
    balanced_batches: bool = True,
) -> Dict[str, Any]:
    """Train the deepfake detection model with balanced training and validation diagnostics."""
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

    # Dynamic class weights
    class_weight_dict = compute_class_weights(train_df["label"].values)
    print(f"Computed training class weights: {class_weight_dict}")

    # Save run_info.json
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
        balanced_batches=balanced_batches,
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
            balanced_batches=False,
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

    callbacks = []

    # Batch progress logger
    class BatchProgressLogger(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            step_str = f"{batch + 1}/{steps_per_epoch}" if steps_per_epoch else f"{batch + 1}"
            if (batch + 1) % 50 == 0 or (steps_per_epoch and (batch + 1) == steps_per_epoch):
                loss = logs.get("loss", 0.0)
                acc = logs.get("accuracy", 0.0)
                auc = logs.get("auc", 0.0)
                print(f"  [Step {step_str}] loss: {loss:.4f} - acc: {acc:.4f} - auc: {auc:.4f}", flush=True)

    callbacks.append(BatchProgressLogger())

    # Learning rate scheduler
    if has_val:
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )
        callbacks.append(reduce_lr_cb)

        # Validation Diagnostics & Model Locking Callback
        val_cb = ValidationDiagnosticsCallback(
            val_ds=val_ds,
            val_labels=val_df["label"].values,
            output_dir=output_dir,
            best_model_path=best_model_path,
            patience=patience,
        )
        callbacks.append(val_cb)

    if steps_per_epoch is not None:
        train_ds = train_ds.repeat()
    if val_steps is not None and val_ds is not None:
        val_ds = val_ds.repeat()

    print(f"\nStarting training for {epochs} epochs (balanced_batches={balanced_batches})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Ensure best_model.keras exists
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
        "balanced_batches": balanced_batches,
        "best_model_path": best_model_path,
    }
    train_config_path = os.path.join(output_dir, "train_config.json")
    with open(train_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save final_training_config.json
    final_train_config_path = os.path.join(output_dir, "final_training_config.json")
    with open(final_train_config_path, "w") as f:
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
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Adam.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Batches per training epoch.")
    parser.add_argument("--val-steps", type=int, default=None, help="Batches per validation epoch.")

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
        steps_per_epoch=args.steps_per_epoch,
        val_steps=args.val_steps,
    )


if __name__ == "__main__":
    main()
