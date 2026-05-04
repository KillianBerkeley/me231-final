"""K-means model script with clustering and comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)

from data_cleaning import load_clean_split


MODEL_NAME = "kmeans"
DEFAULT_MODEL_PATH = "results/kmeans_model.pkl"
DEFAULT_METRICS_PATH = "results/kmeans_metrics.json"
DEFAULT_PLOT_PATH = "results/kmeans_confusion_matrix.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate k-means model.")
    parser.add_argument("--data", default=None, help="Path to input CSV data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--therapy-threshold",
        type=float,
        default=0.8,
        help="Drop seeks_professional_help if abs correlation with has_therapy is above this value.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Save/load model path.")
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH, help="Metrics JSON path.")
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH, help="Confusion matrix image path.")
    parser.add_argument(
        "--predictions-path",
        default="results/kmeans_predictions.csv",
        help="Path to save y_true/y_pred for ensemble comparison.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Skip training and only evaluate with an existing saved model.",
    )
    parser.add_argument(
        "--preprocessed-bundle",
        default=None,
        help="Joblib file with x_train,x_test,y_train,y_test from run_all_models (skips cleaning here).",
    )
    return parser.parse_args()


def load_xy(args, target_column: str):
    if args.preprocessed_bundle:
        import joblib

        print("Using preprocessed bundle (no separate clean in this process):", args.preprocessed_bundle)
        bundle = joblib.load(args.preprocessed_bundle)
        return bundle["x_train"], bundle["x_test"], bundle["y_train"], bundle["y_test"]
    return load_clean_split(
        data_path=args.data,
        target_column=target_column,
        test_size=args.test_size,
        random_state=args.seed,
        therapy_threshold=args.therapy_threshold,
    )


def save_metrics(path: str, metrics: dict):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def map_clusters_to_labels(y_true, clusters):
    """Map each cluster id to the most common true class in that cluster."""
    mapping = {}
    crosstab = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(y_true, name="label"))
    for cluster_id, row in crosstab.iterrows():
        mapping[cluster_id] = row.idxmax()
    mapped = np.array([mapping[c] for c in clusters])
    return mapped


LEVEL_LABELS = {0: "Low", 1: "Moderate", 2: "High"}


def save_confusion_plot(y_true, y_pred, plot_path: str):
    labels = sorted(pd.unique(pd.Series(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = [LEVEL_LABELS.get(l, str(l)) for l in labels]
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples")
    plt.title("K-Means — Burnout Level (Cluster→Label Mapped)")
    plt.ylabel("True Burnout Level")
    plt.xlabel("Predicted Burnout Level")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    args = parse_args()
    target_column = "burnout_level"
    print("=== KMeans Model ===")
    print(f"Using fixed target column: {target_column}")

    x_train, x_test, y_train, y_test = load_xy(args, target_column)

    n_clusters = int(y_train.nunique())
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.test:
        print("Training model...")
        model = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=20)
        model.fit(x_train)
        import joblib

        joblib.dump({"model": model}, model_path)
    else:
        print("Test mode enabled: skipping training.")
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"--test requested but model file not found at '{model_path}'")
        bundle = joblib.load(model_path)
        model = bundle["model"]

    clusters = model.predict(x_test)
    mapped_preds = map_clusters_to_labels(y_test, clusters)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, mapped_preds, average="weighted", zero_division=0
    )
    precision_m, recall_m, _, _ = precision_recall_fscore_support(
        y_test, mapped_preds, average="macro", zero_division=0
    )

    sil = float(silhouette_score(x_test, clusters)) if len(np.unique(clusters)) > 1 else float("nan")
    metrics = {
        "model": MODEL_NAME,
        "adjusted_rand_index": float(adjusted_rand_score(y_test, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y_test, clusters)),
        "precision_macro": float(precision_m),
        "precision_weighted": float(precision_w),
        "recall_macro": float(recall_m),
        "recall_weighted": float(recall_w),
        "f1_macro": float(f1_score(y_test, mapped_preds, average="macro")),
        "f1_weighted": float(f1_w),
        "silhouette_score": sil,
        "plot_path": args.plot_path,
    }

    save_metrics(args.metrics_path, metrics)
    save_confusion_plot(y_test, mapped_preds, args.plot_path)
    pred_out = Path(args.predictions_path)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true_class": y_test, "y_pred_class": mapped_preds}).to_csv(pred_out, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
