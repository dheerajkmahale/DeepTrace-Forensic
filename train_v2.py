"""V2 Training script: CNN -> BiLSTM -> Temporal Attention on Celeb-DF v2.

Implements a 2-stage training strategy:
- Stage 1: Transfers spatial CNN weights from baseline V1, freezes CNN backbone,
  and trains BiLSTM + Temporal Attention pooling + Dense head.
- Stage 2: Fine-tunes with unfreezing at a reduced learning rate.
- Uses ValidationDiagnosticsCallback to strictly monitor validation data.
- Saves:
  - outputs/v2/best_model_v2.keras
  - outputs/v2_threshold.json
  - outputs/v2_validation_metrics.json
"""

import argparse
from datetime import datetime, timezone
import json
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

from config import Config
from dataset import compute_class_weights, get_dataset
from model import (
    TemporalAttention,
    build_model_v2,
    extract_temporal_attention,
    transfer_cnn_weights,
)
from train import set_seed


class V2ValidationCallback(tf.keras.callbacks.Callback):
    """Monitors validation metrics, searches optimal threshold, and tracks best model."""

    def __init__(
        self,
        val_ds: tf.data.Dataset,
        val_labels: np.ndarray,
        output_dir: str,
        best_model_path: str,
        patience: int = 3,
    ):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.asarray(val_labels, dtype=int)
        self.output_dir = output_dir
        self.best_model_path = best_model_path
        self.patience = patience
        self.best_score = -1.0
        self.best_epoch = -1
        self.best_metrics: Dict[str, Any] = {}
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        preds = self.model.predict(self.val_ds, verbose=0).flatten()
        y_true = self.val_labels

        try:
            auc = float(roc_auc_score(y_true, preds))
        except Exception:
            auc = 0.5

        thresholds = np.linspace(0.10, 0.90, 81)
        best_th = 0.5
        best_bacc = -1.0

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

        p_min = float(preds.min())
        p_max = float(preds.max())
        p_mean = float(preds.mean())
        p_std = float(preds.std())

        print("\n" + "=" * 65)
        print(f"V2 VALIDATION DIAGNOSTICS (Epoch {epoch + 1})")
        print("=" * 65)
        print(f"Prob Distribution: min={p_min:.4f}, max={p_max:.4f}, mean={p_mean:.4f}, std={p_std:.4f}")
        print(f"Optimal Val Threshold: {best_th:.4f}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {bal_acc:.4f}")
        print(f"  Precision:         {prec:.4f}")
        print(f"  Fake Recall:       {fake_recall:.4f}")
        print(f"  Real Recall:       {real_recall:.4f}")
        print(f"  F1 Score:          {f1:.4f} (Macro: {macro_f1:.4f})")
        print(f"  ROC-AUC:           {auc:.4f}")
        print(f"  Confusion Matrix:  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")

        is_healthy = (
            real_recall > 0.0
            and fake_recall > 0.0
            and p_std > 0.01
            and (tn + tp) > 0
            and (fp + fn) < len(y_true)
        )

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
        }

        score_to_maximize = bal_acc + auc
        if is_healthy and score_to_maximize > self.best_score:
            print(f"  >>> New best V2 validation performance ({score_to_maximize:.4f} > {self.best_score:.4f}). Saving! <<<")
            self.best_score = score_to_maximize
            self.best_epoch = epoch + 1
            self.best_metrics = epoch_metrics
            self.wait = 0

            # Save candidate V2 model
            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            self.model.save(self.best_model_path)

            # Save V2 threshold to outputs/v2_threshold.json and local dir
            th_payload = {
                "threshold": best_th,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "roc_auc": auc,
                "epoch": epoch + 1,
                "real_recall": real_recall,
                "fake_recall": fake_recall,
            }
            with open("outputs/v2_threshold.json", "w") as f:
                json.dump(th_payload, f, indent=2)
            with open(os.path.join(self.output_dir, "v2_threshold.json"), "w") as f:
                json.dump(th_payload, f, indent=2)

            # Save V2 validation metrics to outputs/v2_validation_metrics.json and local dir
            with open("outputs/v2_validation_metrics.json", "w") as f:
                json.dump(epoch_metrics, f, indent=2)
            with open(os.path.join(self.output_dir, "v2_validation_metrics.json"), "w") as f:
                json.dump(epoch_metrics, f, indent=2)
        else:
            self.wait += 1
            print(f"  No improvement for {self.wait}/{self.patience} epochs.")
            if self.wait >= self.patience:
                print(f"  Early stopping triggered after {self.wait} epochs.")
                self.model.stop_training = True

        print("=" * 65 + "\n")


def train_v2_model(
    processed_dir: str = "data/processed",
    output_dir: str = "outputs/v2",
    baseline_model_path: str = "outputs/baseline_v1/best_model.keras",
    seq_len: int = 10,
    img_size: int = 128,
    epochs_stage1: int = 3,
    epochs_stage2: int = 1,
    batch_size: int = 16,
    lr_stage1: float = 3e-4,
    lr_stage2: float = 3e-5,
    lstm_units: int = 128,
    attention_units: int = 64,
    dropout: float = 0.4,
    seed: int = 42,
    patience: int = 3,
) -> Dict[str, Any]:
    """Train V2 model: CNN -> BiLSTM -> Temporal Attention."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(processed_dir, "meta.csv")
    meta_df = pd.read_csv(meta_path)
    train_df = meta_df[meta_df["split"].str.lower() == "train"]
    val_df = meta_df[meta_df["split"].str.lower() == "val"]

    print(f"\n==================================================")
    print(f"STARTING V2 CNN-BiLSTM-ATTENTION TRAINING PIPELINE")
    print(f"Train clips: {len(train_df)} | Val clips: {len(val_df)}")
    print(f"==================================================\n")

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
        balanced_batches=True,
    )

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

    # Build V2 model
    model = build_model_v2(
        seq_len=seq_len,
        img_size=img_size,
        backbone="light",
        lstm_units=lstm_units,
        dropout=dropout,
        attention_units=attention_units,
    )

    # Transfer pretrained CNN weights from V1
    if os.path.exists(baseline_model_path):
        print(f"Loading baseline V1 model from {baseline_model_path} for spatial weight transfer...")
        v1_model = tf.keras.models.load_model(baseline_model_path)
        transfer_success = transfer_cnn_weights(v1_model, model)
        print(f"CNN backbone weight transfer success: {transfer_success}")
    else:
        print(f"[WARNING] Baseline model {baseline_model_path} not found. Training CNN from scratch.")

    best_model_path = os.path.join(output_dir, "best_model_v2.keras")
    val_callback = V2ValidationCallback(
        val_ds=val_ds,
        val_labels=val_df["label"].values,
        output_dir=output_dir,
        best_model_path=best_model_path,
        patience=patience,
    )

    # Step progress logger
    class BatchLogger(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            logs = logs or {}
            if (batch + 1) % 50 == 0:
                loss = logs.get("loss", 0.0)
                acc = logs.get("accuracy", 0.0)
                auc = logs.get("auc", 0.0)
                print(f"  [Batch {batch + 1}] loss: {loss:.4f} - acc: {acc:.4f} - auc: {auc:.4f}", flush=True)

    # ==========================================
    # STAGE 1: Freeze CNN, Train Temporal & Head
    # ==========================================
    print(f"\n--- STAGE 1: Training BiLSTM + Temporal Attention (CNN Frozen, lr={lr_stage1}) ---")
    model.get_layer("time_distributed_cnn").trainable = False

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=lr_stage1)
    model.compile(
        optimizer=optimizer1,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=[BatchLogger(), val_callback],
        verbose=1,
    )

    # ==========================================
    # STAGE 2: Fine-tune Upper Layers (if epochs_stage2 > 0)
    # ==========================================
    if epochs_stage2 > 0 and not model.stop_training:
        print(f"\n--- STAGE 2: Fine-tuning Full Model (lr={lr_stage2}) ---")
        model.get_layer("time_distributed_cnn").trainable = True

        optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr_stage2)
        model.compile(
            optimizer=optimizer2,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

        stage2_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage2,
            callbacks=[BatchLogger(), val_callback],
            verbose=1,
        )

    # Ensure best model exists
    if not os.path.exists(best_model_path):
        model.save(best_model_path)

    # Also save to outputs/best_model_v2.keras
    top_best_v2 = "outputs/best_model_v2.keras"
    if os.path.exists(best_model_path):
        import shutil
        shutil.copyfile(best_model_path, top_best_v2)

    print(f"\nV2 training complete. Best model saved to {best_model_path}")
    print(f"Best metrics: {val_callback.best_metrics}")

    return {
        "best_model_path": best_model_path,
        "best_metrics": val_callback.best_metrics,
        "best_epoch": val_callback.best_epoch,
    }


def main():
    parser = argparse.ArgumentParser(description="Train V2 deepfake detector.")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--output-dir", type=str, default="outputs/v2")
    parser.add_argument("--baseline-model", type=str, default="outputs/baseline_v1/best_model.keras")
    parser.add_argument("--epochs-stage1", type=int, default=3)
    parser.add_argument("--epochs-stage2", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-stage1", type=float, default=3e-4)
    parser.add_argument("--lr-stage2", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_v2_model(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        baseline_model_path=args.baseline_model,
        epochs_stage1=args.epochs_stage1,
        epochs_stage2=args.epochs_stage2,
        batch_size=args.batch_size,
        lr_stage1=args.lr_stage1,
        lr_stage2=args.lr_stage2,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
