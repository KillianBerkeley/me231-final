"""Regression model script with train/test metrics and plots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_cleaning import load_clean_split, score_to_level


MODEL_NAME = "regression"
DEFAULT_MODEL_PATH = "results/regression_model.pkl"
DEFAULT_METRICS_PATH = "results/regression_metrics.json"
DEFAULT_PLOT_PATH = "results/regression_pred_vs_true.png"


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate regression model.")
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
    parser.add_argument("--plot-path", default=DEFAULT_PLOT_PATH, help="Prediction plot image path.")
    parser.add_argument(
        "--predictions-path",
        default="results/regression_predictions.csv",
        help="Path to save y_true/y_pred class labels for ensemble comparison.",
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


def save_plot(y_true, y_pred, plot_path: str):
    from sklearn.metrics import r2_score as _r2
    r2 = _r2(y_true, y_pred)
    out = Path(plot_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=8, color="#4f8ef7")
    y_min, y_max = min(float(y_true.min()), float(y_pred.min())), max(float(y_true.max()), float(y_pred.max()))
    plt.plot([y_min, y_max], [y_min, y_max], linestyle="--", linewidth=1.5, color="#f97316", label="Perfect fit")
    plt.title(f"Linear Regression: Predicted vs True Burnout Score\nR² = {r2:.3f}")
    plt.xlabel("True Burnout Score")
    plt.ylabel("Predicted Burnout Score")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def main():
    args = parse_args()
    target_column = "burnout_score"
    print("=== Regression Model ===")
    print(f"Using fixed target column: {target_column}")

    x_train, x_test, y_train, y_test = load_xy(args, target_column)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.test:
        print("Training model...")
        model = LinearRegression()
        model.fit(x_train, y_train)
        import joblib

        joblib.dump(model, model_path)
    else:
        print("Test mode enabled: skipping training.")
        import joblib

        if not model_path.exists():
            raise FileNotFoundError(f"--test requested but model file not found at '{model_path}'")
        model = joblib.load(model_path)

    preds = model.predict(x_test)
    low_cut = float(y_train.quantile(0.33))
    high_cut = float(y_train.quantile(0.66))
    y_true_class = score_to_level(y_test, low_cut, high_cut)
    y_pred_class = score_to_level(preds, low_cut, high_cut)
    metrics = {
        "model": MODEL_NAME,
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, preds))),
        "class_low_cut": low_cut,
        "class_high_cut": high_cut,
        "plot_path": args.plot_path,
    }

    save_metrics(args.metrics_path, metrics)
    save_plot(y_test, preds, args.plot_path)
    pred_out = Path(args.predictions_path)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true_class": y_true_class, "y_pred_class": y_pred_class}).to_csv(
        pred_out, index=False
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
