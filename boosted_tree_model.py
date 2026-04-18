"""Histogram-based Gradient Boosting classifier (sklearn) - train/test and comparison outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from data_cleaning import load_clean_split


MODEL_NAME = "boosted_tree"
DEFAULT_MODEL_PATH = "results/boosted_tree_model.pkl"
DEFAULT_METRICS_PATH = "results/boosted_tree_metrics.json"
DEFAULT_PLOT_PATH = "results/boosted_tree_confusion_matrix.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate HistGradientBoostingClassifier.")
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
        default="results/boosted_tree_predictions.csv",
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


def save_confusion_plot(y_true, y_pred, plot_path: str):
    labels = sorted(pd.unique(pd.Series(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
    plt.title("Boosted Tree (HistGradientBoosting) Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    args = parse_args()
    target_column = "burnout_level"

    print("=== Boosted Tree (HistGradientBoosting) Model ===")
    print(f"Using fixed target column: {target_column}")
    x_train, x_test, y_train, y_test = load_xy(args, target_column)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.test:
        print("Training model...")
        model = HistGradientBoostingClassifier(
            max_iter=200,
            random_state=args.seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        model.fit(x_train, y_train)
        import joblib

        joblib.dump({"model": model}, model_path)
    else:
        print("Test mode enabled: skipping training.")
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"--test requested but model file not found at '{model_path}'")
        bundle = joblib.load(model_path)
        model = bundle["model"]

    preds = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    metrics = {
        "model": MODEL_NAME,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "plot_path": args.plot_path,
    }

    save_metrics(args.metrics_path, metrics)
    save_confusion_plot(y_test, preds, args.plot_path)
    pred_out = Path(args.predictions_path)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true_class": y_test, "y_pred_class": preds}).to_csv(pred_out, index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
