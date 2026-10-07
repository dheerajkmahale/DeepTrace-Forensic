"""Evaluation pipeline for deepfake detection model on held-out test split.

Strict rules:
- Evaluates ONLY the held-out TEST split (no train or val data leakage).
- Computes metrics at both clip-level and video-level.
- Video score is computed as the mean of clip probabilities for that video.
- Outputs:
  1. metrics.json (with distinct clip_level and video_level sections)
  2. predictions.csv (video, clip, true_label, predicted_probability, predicted_label)
  3. confusion_matrix.png
  4. roc_curve.png
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_curve,
)
import tensorflow as tf

from dataset import resolve_clip_path
from model import TemporalAttention


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, f1, roc_auc, balanced_accuracy, real_recall, fake_recall, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)
    y_pred = (y_scores >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_scores))
        else:
            auc = float("nan")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    real_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fake_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "balanced_accuracy": bal_acc,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "confusion_matrix": cm.tolist(),
        "total_samples": int(len(y_true)),
    }


def plot_confusion_matrices(
    clip_true: np.ndarray,
    clip_pred: np.ndarray,
    video_true: np.ndarray,
    video_pred: np.ndarray,
    save_path: str,
) -> None:
    """Plot and save side-by-side confusion matrices for clips and videos."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    classes = ["Real (0)", "Fake (1)"]

    for ax, y_t, y_p, title in [
        (axes[0], clip_true, clip_pred, "Clip-Level Confusion Matrix"),
        (axes[1], video_true, video_pred, "Video-Level Confusion Matrix"),
    ]:
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    clip_true: np.ndarray,
    clip_scores: np.ndarray,
    video_true: np.ndarray,
    video_scores: np.ndarray,
    clip_auc: float,
    video_auc: float,
    save_path: str,
) -> None:
    """Plot and save ROC curves for clip-level and video-level evaluation."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Chance baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance (AUC = 0.50)")

    if len(np.unique(clip_true)) > 1:
        fpr_c, tpr_c, _ = roc_curve(clip_true, clip_scores)
        auc_str_c = f"{clip_auc:.3f}" if not np.isnan(clip_auc) else "N/A"
        ax.plot(fpr_c, tpr_c, color="#1f77b4", lw=2, label=f"Clip-Level (AUC = {auc_str_c})")

    if len(np.unique(video_true)) > 1:
        fpr_v, tpr_v, _ = roc_curve(video_true, video_scores)
        auc_str_v = f"{video_auc:.3f}" if not np.isnan(video_auc) else "N/A"
        ax.plot(fpr_v, tpr_v, color="#ff7f0e", lw=2, label=f"Video-Level (AUC = {auc_str_v})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Test Set ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def find_optimal_validation_threshold(
    model: tf.keras.Model,
    meta_df: pd.DataFrame,
    processed_dir: str,
    batch_size: int = 8,
) -> float:
    """Find decision threshold that maximizes F1 score on validation split.

    Crucially, this searches ONLY on the validation split, never on the test split.
    """
    val_df = meta_df[meta_df["split"].str.lower() == "val"].copy().reset_index(drop=True)
    if len(val_df) == 0:
        print("[WARNING] No validation samples found in meta.csv. Using default threshold 0.5.")
        return 0.5

    clip_probs: List[float] = []
    loaded_clips: List[np.ndarray] = []

    for _, row in val_df.iterrows():
        clip_file = resolve_clip_path(str(row["path"]), processed_dir)
        try:
            clip = np.load(clip_file)
        except Exception as e:
            raise RuntimeError(f"Could not load val clip from {clip_file}: {e}")
        loaded_clips.append(clip)

        if len(loaded_clips) == batch_size:
            batch_arr = np.array(loaded_clips, dtype=np.uint8)
            preds = model.predict(batch_arr, verbose=0).flatten()
            clip_probs.extend(preds.tolist())
            loaded_clips = []

    if loaded_clips:
        batch_arr = np.array(loaded_clips, dtype=np.uint8)
        preds = model.predict(batch_arr, verbose=0).flatten()
        clip_probs.extend(preds.tolist())

    val_df["predicted_probability"] = clip_probs
    val_vids = []
    for vid_name, grp in val_df.groupby("video"):
        val_vids.append({
            "video": vid_name,
            "true_label": int(grp["label"].iloc[0]),
            "predicted_probability": float(grp["predicted_probability"].mean()),
        })
    val_vid_df = pd.DataFrame(val_vids)
    y_true = val_vid_df["true_label"].values
    y_scores = val_vid_df["predicted_probability"].values

    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        preds = (y_scores >= th).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)

    print(f"[INFO] Tuned threshold on validation split: {best_th:.4f} (Val Video F1: {best_f1:.4f})")
    return best_th


def evaluate(
    model_path: str = "outputs/best_model.keras",
    processed_dir: str = "data/processed",
    output_dir: str = "outputs",
    threshold: Optional[float] = None,
    threshold_file: Optional[str] = None,
    val_threshold: bool = False,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Evaluate trained model on the held-out test split.

    Args:
        model_path: Path to saved Keras model file.
        processed_dir: Directory containing meta.csv and preprocessed clips.
        output_dir: Directory where evaluation artifacts will be written.
        threshold: Classification decision threshold (default: 0.5 if not specified).
        threshold_file: Path to JSON file containing locked validation threshold.
        val_threshold: If True, tunes decision threshold on validation split.
        batch_size: Batch size for model inference.

    Returns:
        Dict containing full clip and video metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    meta_path = os.path.join(processed_dir, "meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.csv not found at {meta_path}")

    meta_df = pd.read_csv(meta_path)
    test_df = meta_df[meta_df["split"].str.lower() == "test"].copy().reset_index(drop=True)

    print(f"\nEvaluating ONLY held-out TEST split from {meta_path}...")
    print(f"Total test clips: {len(test_df)} across {test_df['video'].nunique()} distinct videos.")

    if len(test_df) == 0:
        raise ValueError("No samples found for split='test' in meta.csv. Evaluation requires held-out test split.")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, custom_objects={"TemporalAttention": TemporalAttention})

    # Determine decision threshold
    auto_th_file = threshold_file or os.path.join(output_dir, "final_threshold.json")
    if threshold is not None:
        chosen_threshold = float(threshold)
        threshold_source = "fixed_0.5" if chosen_threshold == 0.5 else "fixed_specified"
    elif val_threshold:
        chosen_threshold = find_optimal_validation_threshold(
            model=model,
            meta_df=meta_df,
            processed_dir=processed_dir,
            batch_size=batch_size,
        )
        threshold_source = "validation_split"
    elif os.path.exists(auto_th_file):
        with open(auto_th_file, "r") as f:
            th_data = json.load(f)
            chosen_threshold = float(th_data.get("threshold", 0.5))
            threshold_source = f"locked_validation_{os.path.basename(auto_th_file)}"
        print(f"Loaded locked validation threshold {chosen_threshold:.4f} from {auto_th_file}")
    else:
        chosen_threshold = 0.5
        threshold_source = "fixed_0.5"

    # Load clips and run inference on test set
    clip_probs: List[float] = []
    loaded_clips: List[np.ndarray] = []

    for _, row in test_df.iterrows():
        clip_file = resolve_clip_path(str(row["path"]), processed_dir)
        try:
            clip = np.load(clip_file)
        except Exception as e:
            raise RuntimeError(f"Could not load test clip from {clip_file}: {e}")
        loaded_clips.append(clip)

        if len(loaded_clips) == batch_size:
            batch_arr = np.array(loaded_clips, dtype=np.uint8)
            preds = model.predict(batch_arr, verbose=0).flatten()
            clip_probs.extend(preds.tolist())
            loaded_clips = []

    if loaded_clips:
        batch_arr = np.array(loaded_clips, dtype=np.uint8)
        preds = model.predict(batch_arr, verbose=0).flatten()
        clip_probs.extend(preds.tolist())

    clip_probs = np.array(clip_probs, dtype=float)
    clip_preds = (clip_probs >= chosen_threshold).astype(int)
    clip_trues = test_df["label"].values.astype(int)

    test_df["predicted_probability"] = clip_probs
    test_df["predicted_label"] = clip_preds

    # Video-level aggregation: mean of clip probabilities belonging to that video
    video_records = []
    for video_name, group in test_df.groupby("video"):
        mean_prob = float(group["predicted_probability"].mean())
        true_label = int(group["label"].iloc[0])
        pred_label = int(mean_prob >= chosen_threshold)
        video_records.append({
            "video": video_name,
            "true_label": true_label,
            "predicted_probability": mean_prob,
            "predicted_label": pred_label,
            "num_clips": len(group),
        })

    video_df = pd.DataFrame(video_records)

    # Compute distinct metrics
    clip_metrics = compute_metrics(clip_trues, clip_probs, threshold=chosen_threshold)
    video_metrics = compute_metrics(
        video_df["true_label"].values,
        video_df["predicted_probability"].values,
        threshold=chosen_threshold,
    )

    real_test_clips = int((test_df["label"] == 0).sum())
    fake_test_clips = int((test_df["label"] == 1).sum())
    real_test_vids = int(video_df[video_df["true_label"] == 0]["video"].nunique())
    fake_test_vids = int(video_df[video_df["true_label"] == 1]["video"].nunique())

    results = {
        "clip_level": clip_metrics,
        "video_level": video_metrics,
        "threshold": float(chosen_threshold),
        "threshold_source": threshold_source,
        "model_path": model_path,
        "test_counts": {
            "clips": {
                "real": real_test_clips,
                "fake": fake_test_clips,
                "total": int(len(test_df)),
            },
            "videos": {
                "real": real_test_vids,
                "fake": fake_test_vids,
                "total": int(len(video_df)),
            },
        },
        "test_clips_count": int(len(test_df)),
        "test_videos_count": int(len(video_df)),
    }

    # Save metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions.csv (video, clip, true label, predicted probability, predicted label)
    # Merge video mean probability for completeness
    video_mean_map = dict(zip(video_df["video"], video_df["predicted_probability"]))
    test_df["video_mean_probability"] = test_df["video"].map(video_mean_map)
    test_df["video_predicted_label"] = (test_df["video_mean_probability"] >= chosen_threshold).astype(int)

    export_df = test_df[[
        "video",
        "clip",
        "label",
        "predicted_probability",
        "predicted_label",
        "video_mean_probability",
        "video_predicted_label",
    ]].rename(columns={"label": "true_label"})

    predictions_path = os.path.join(output_dir, "predictions.csv")
    export_df.to_csv(predictions_path, index=False)

    # Generate plots
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrices(
        clip_true=clip_trues,
        clip_pred=clip_preds,
        video_true=video_df["true_label"].values,
        video_pred=video_df["predicted_label"].values,
        save_path=cm_path,
    )

    roc_path = os.path.join(output_dir, "roc_curve.png")
    plot_roc_curves(
        clip_true=clip_trues,
        clip_scores=clip_probs,
        video_true=video_df["true_label"].values,
        video_scores=video_df["predicted_probability"].values,
        clip_auc=clip_metrics["roc_auc"],
        video_auc=video_metrics["roc_auc"],
        save_path=roc_path,
    )

    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET EVALUATION RESULTS")
    print("=" * 55)
    print(f"Classification threshold: {chosen_threshold:.4f} (Source: {threshold_source})")
    print(f"Test counts - Real clips: {real_test_clips} | Fake clips: {fake_test_clips}")
    print(f"Test counts - Real videos: {real_test_vids} | Fake videos: {fake_test_vids}")
    print(f"\nClip-Level Metrics ({clip_metrics['total_samples']} clips):")
    print(f"  Accuracy:  {clip_metrics['accuracy']:.4f}")
    print(f"  Precision: {clip_metrics['precision']:.4f}")
    print(f"  Recall:    {clip_metrics['recall']:.4f}")
    print(f"  F1-Score:  {clip_metrics['f1']:.4f}")
    auc_str = f"{clip_metrics['roc_auc']:.4f}" if not np.isnan(clip_metrics['roc_auc']) else "N/A"
    print(f"  ROC-AUC:   {auc_str}")

    print(f"\nVideo-Level Metrics ({video_metrics['total_samples']} videos):")
    print(f"  Accuracy:  {video_metrics['accuracy']:.4f}")
    print(f"  Precision: {video_metrics['precision']:.4f}")
    print(f"  Recall:    {video_metrics['recall']:.4f}")
    print(f"  F1-Score:  {video_metrics['f1']:.4f}")
    vauc_str = f"{video_metrics['roc_auc']:.4f}" if not np.isnan(video_metrics['roc_auc']) else "N/A"
    print(f"  ROC-AUC:   {vauc_str}")

    print(f"\nArtifacts generated:")
    print(f"  - Metrics:          {metrics_path}")
    print(f"  - Predictions:      {predictions_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - ROC Curve:        {roc_path}")
    print("=" * 55 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector on held-out test split.")
    parser.add_argument("--model-path", type=str, default="outputs/best_model.keras", help="Path to model.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output files.")
    parser.add_argument("--threshold", type=float, default=None, help="Classification decision threshold.")
    parser.add_argument("--threshold-file", type=str, default=None, help="Path to JSON file with locked threshold.")
    parser.add_argument(
        "--val-threshold",
        action="store_true",
        help="Tune decision threshold on validation split instead of fixed threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        threshold_file=args.threshold_file,
        val_threshold=args.val_threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
