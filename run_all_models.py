"""Run all model scripts with a shared argparse contract and summary output."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from data_cleaning import load_clean_split


def parse_args():
    parser = argparse.ArgumentParser(description="Run all model files with shared args.")
    parser.add_argument("--data", default=None, help="Path to input CSV data.")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Fraction for test split."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Skip training and only run evaluation with existing saved models.",
    )
    parser.add_argument(
        "--therapy-threshold",
        type=float,
        default=0.8,
        help="Drop seeks_professional_help if abs correlation with has_therapy is above this value.",
    )
    parser.add_argument(
        "--comparison-path",
        default="results/comparison.csv",
        help="Path to save combined metrics table.",
    )
    parser.add_argument(
        "--ensemble-metrics-path",
        default="results/ensemble_metrics.json",
        help="Path to save vote + weighted ensemble metrics.",
    )
    parser.add_argument(
        "--ensemble-plot-path",
        default="results/ensemble_vote_confusion_matrix.png",
        help="Path to save majority-vote ensemble confusion matrix.",
    )
    parser.add_argument(
        "--ensemble-weighted-plot-path",
        default="results/ensemble_weighted_confusion_matrix.png",
        help="Path to save weighted ensemble confusion matrix.",
    )
    parser.add_argument(
        "--dashboard-plot-path",
        default="results/model_comparison_dashboard.png",
        help="Path to save side-by-side confusion matrices with ranking.",
    )
    parser.add_argument(
        "--stats-table-plot-path",
        default="results/model_metrics_table.png",
        help="Path to save a full metrics table as PNG.",
    )
    parser.add_argument(
        "--ensemble-weights",
        default="1,1,1,1",
        help="Comma weights for nn,svm,boosted_tree,regression (KMeans excluded from ensemble).",
    )
    parser.add_argument(
        "--ensemble-auto-weights",
        action="store_true",
        help="Set ensemble weights proportional to each model's macro F1 on the test split.",
    )
    parser.add_argument(
        "--no-ensemble-auto-weights",
        action="store_true",
        help="Disable auto macro-F1 weighting and use --ensemble-weights values directly.",
    )
    return parser.parse_args()


def run_model(script_name: str, args, preprocessed_bundle: Path):
    metrics_name = script_name.replace(".py", "_metrics.json")
    metrics_path = str(Path("results") / metrics_name)
    plot_name = script_name.replace(".py", "_plot.png")
    plot_path = str(Path("results") / plot_name)
    preds_name = script_name.replace(".py", "_predictions.csv")
    predictions_path = str(Path("results") / preds_name)
    cmd = [
        sys.executable,
        script_name,
        "--preprocessed-bundle",
        str(preprocessed_bundle),
        "--seed",
        str(args.seed),
        "--metrics-path",
        metrics_path,
        "--plot-path",
        plot_path,
        "--predictions-path",
        predictions_path,
    ]
    if args.test:
        cmd.append("--test")

    print(f"\nRunning {script_name}")
    subprocess.run(cmd, check=True)
    return metrics_path, predictions_path


def parse_weight_string(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(x) for x in parts]


def majority_vote_row(row: pd.Series, pred_cols: list[str]) -> int:
    votes = [int(row[c]) for c in pred_cols]
    counts = Counter(votes)
    best = max(counts.values())
    candidates = [c for c, v in counts.items() if v == best]
    return min(candidates)


def weighted_vote_row(row: pd.Series, pred_cols: list[str], weights: np.ndarray) -> int:
    """Sum weights per class label; break ties by smaller class index."""
    n_class = 3
    scores = np.zeros(n_class, dtype=float)
    for col, w in zip(pred_cols, weights):
        cls = int(row[col])
        scores[cls] += w
    return int(np.argmax(scores))


def save_confusion_plot(y_true, y_pred, output_path: str, title: str):
    labels = sorted(pd.unique(pd.Series(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def save_comparison_dashboard(
    model_entries: list[tuple[str, pd.Series, pd.Series, str, str | None]],
    output_path: str,
    weighted_details: str | None = None,
):
    """model_entries: (display_name, y_true, y_pred, cmap). Ranking panel: bottom-right."""
    y0 = model_entries[0][1]
    labels = sorted(pd.unique(pd.Series(y0)))

    n = len(model_entries)
    n_cols = 2
    n_heatmap_rows = (n + n_cols - 1) // n_cols
    last_r, last_c = divmod(n - 1, n_cols)
    if last_c == n_cols - 1:
        rank_r = n_heatmap_rows
        rank_c = 1
        n_total_rows = n_heatmap_rows + 1
    else:
        rank_r = last_r
        rank_c = last_c + 1
        n_total_rows = n_heatmap_rows

    ranking_rows = []
    fig = plt.figure(figsize=(15, 3.9 * n_total_rows))
    gs = gridspec.GridSpec(n_total_rows, n_cols, figure=fig, hspace=0.45, wspace=0.3)

    for idx, (name, y_true, y_pred, cmap, extra_note) in enumerate(model_entries):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[r, c])
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap, ax=ax, cbar=False)
        mf1 = float(f1_score(y_true, y_pred, average="macro"))
        wf1 = float(f1_score(y_true, y_pred, average="weighted"))
        acc = float(accuracy_score(y_true, y_pred))
        title = (
            f"{name}\n"
            f"F1 macro={mf1:.4f}  weighted={wf1:.4f}  acc={acc:.4f}"
        )
        if extra_note:
            title = f"{title}\n{extra_note}"
        ax.set_title(
            title,
            fontsize=10,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ranking_rows.append(
            {
                "model": name,
                "macro_f1": mf1,
                "weighted_f1": wf1,
                "accuracy": acc,
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    )
    ranking_lines = [
        "Model ranking",
        "(macro F1, weighted F1, then accuracy)",
        "",
    ]
    if weighted_details:
        ranking_lines.extend([weighted_details, ""])
    for rank, row in ranking_df.reset_index(drop=True).iterrows():
        ranking_lines.append(
            f"{rank + 1}. {row['model']}\n"
            f"   macro={row['macro_f1']:.4f}  weighted={row['weighted_f1']:.4f}  acc={row['accuracy']:.4f}"
        )

    ax_rank = fig.add_subplot(gs[rank_r, rank_c])
    ax_rank.axis("off")
    ax_rank.text(
        0.02,
        0.98,
        "\n".join(ranking_lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        transform=ax_rank.transAxes,
    )
    fig.suptitle("Model Comparison Dashboard", fontsize=15, y=1.02)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    level_bundle_path = results_dir / "run_all_burnout_level_split.joblib"
    score_bundle_path = results_dir / "run_all_burnout_score_split.joblib"

    print("=======================")
    print("RUN_ALL: CLEANING DATA (burnout_level pipeline)")
    print("=======================")
    x_lt, x_lte, y_lt, y_lte = load_clean_split(
        args.data,
        "burnout_level",
        args.test_size,
        args.seed,
        args.therapy_threshold,
    )
    joblib.dump(
        {"x_train": x_lt, "x_test": x_lte, "y_train": y_lt, "y_test": y_lte},
        level_bundle_path,
    )
    print(f"Saved classification split bundle: {level_bundle_path}")

    print("=======================")
    print("RUN_ALL: CLEANING DATA (burnout_score pipeline)")
    print("=======================")
    x_st, x_ste, y_st, y_ste = load_clean_split(
        args.data,
        "burnout_score",
        args.test_size,
        args.seed,
        args.therapy_threshold,
    )
    joblib.dump(
        {"x_train": x_st, "x_test": x_ste, "y_train": y_st, "y_test": y_ste},
        score_bundle_path,
    )
    print(f"Saved regression split bundle: {score_bundle_path}")

    scripts = [
        "nn_model.py",
        "kmeans_model.py",
        "svm_model.py",
        "boosted_tree_model.py",
        "regression_model.py",
    ]
    # KMeans excluded from ensemble; indices for nn,svm,boosted,reg in pred_frames order
    ensemble_pred_indices = [0, 2, 3, 4]
    pred_cols = ["nn_pred", "svm_pred", "boosted_pred", "regression_pred"]

    metrics_files = []
    prediction_files = []
    for script in scripts:
        bundle = score_bundle_path if script == "regression_model.py" else level_bundle_path
        metrics_path, pred_path = run_model(script, args, bundle)
        metrics_files.append(metrics_path)
        prediction_files.append(pred_path)

    rows = []
    for metrics_file in metrics_files:
        metrics = json.loads(Path(metrics_file).read_text(encoding="utf-8"))
        rows.append(metrics)

    comparison_df = pd.DataFrame(rows)
    out = Path(args.comparison_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out, index=False)
    print(f"\nSaved comparison table: {out}")

    # Build a full metrics table (all available numeric columns) for a separate PNG.
    full_metrics_df = comparison_df.copy()
    metrics_by_model = {
        str(row["model"]): row for _, row in comparison_df.iterrows() if "model" in row and pd.notna(row["model"])
    }

    pred_frames = [pd.read_csv(path) for path in prediction_files]
    n0 = len(pred_frames[0])
    if not all(len(p) == n0 for p in pred_frames):
        raise ValueError("Prediction files have mismatched lengths; cannot ensemble.")

    ensemble_df = pd.DataFrame({"y_true_class": pred_frames[0]["y_true_class"]})
    for col_name, idx in zip(pred_cols, ensemble_pred_indices):
        ensemble_df[col_name] = pred_frames[idx]["y_pred_class"]

    ensemble_df["ensemble_vote"] = ensemble_df.apply(
        lambda r: majority_vote_row(r, pred_cols), axis=1
    )

    weights = parse_weight_string(args.ensemble_weights)
    if len(weights) != 4:
        raise ValueError(
            "--ensemble-weights must have exactly 4 values (nn,svm,boosted_tree,regression)."
        )
    w = np.array(weights, dtype=float)
    use_auto_weights = args.ensemble_auto_weights or not args.no_ensemble_auto_weights
    if use_auto_weights:
        f1s = []
        for i in ensemble_pred_indices:
            yt = pred_frames[i]["y_true_class"].astype(int)
            yp = pred_frames[i]["y_pred_class"].astype(int)
            f1s.append(f1_score(yt, yp, average="macro"))
        w = np.array(f1s, dtype=float)
        print("Ensemble auto-weights (macro F1 proportional, no KMeans):", w / w.sum())
    if w.sum() <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    w = w / w.sum()

    ensemble_df["ensemble_weighted"] = ensemble_df.apply(
        lambda r: weighted_vote_row(r, pred_cols, w), axis=1
    )

    y_true = ensemble_df["y_true_class"].astype(int)
    y_vote = ensemble_df["ensemble_vote"].astype(int)
    y_wtd = ensemble_df["ensemble_weighted"].astype(int)
    ensemble_model_order = ["nn", "svm", "boosted_tree", "regression"]
    weights_by_model = {name: float(weight) for name, weight in zip(ensemble_model_order, w)}
    weight_source = "auto_macro_f1" if use_auto_weights else "manual"
    print(f"Weighted ensemble source: {weight_source}")
    print(f"Weighted ensemble by model: {weights_by_model}")

    ensemble_bundle = {
        "vote": {
            "model": "ensemble_majority_vote",
            "accuracy": float(accuracy_score(y_true, y_vote)),
            "precision_macro": float(
                precision_recall_fscore_support(y_true, y_vote, average="macro", zero_division=0)[0]
            ),
            "precision_weighted": float(
                precision_recall_fscore_support(y_true, y_vote, average="weighted", zero_division=0)[0]
            ),
            "recall_macro": float(
                precision_recall_fscore_support(y_true, y_vote, average="macro", zero_division=0)[1]
            ),
            "recall_weighted": float(
                precision_recall_fscore_support(y_true, y_vote, average="weighted", zero_division=0)[1]
            ),
            "macro_f1": float(f1_score(y_true, y_vote, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_vote, average="weighted")),
            "weights": None,
            "plot_path": args.ensemble_plot_path,
        },
        "weighted": {
            "model": "ensemble_weighted_vote",
            "accuracy": float(accuracy_score(y_true, y_wtd)),
            "precision_macro": float(
                precision_recall_fscore_support(y_true, y_wtd, average="macro", zero_division=0)[0]
            ),
            "precision_weighted": float(
                precision_recall_fscore_support(y_true, y_wtd, average="weighted", zero_division=0)[0]
            ),
            "recall_macro": float(
                precision_recall_fscore_support(y_true, y_wtd, average="macro", zero_division=0)[1]
            ),
            "recall_weighted": float(
                precision_recall_fscore_support(y_true, y_wtd, average="weighted", zero_division=0)[1]
            ),
            "macro_f1": float(f1_score(y_true, y_wtd, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_wtd, average="weighted")),
            "weights": w.tolist(),
            "weights_by_model": weights_by_model,
            "weight_source": weight_source,
            "plot_path": args.ensemble_weighted_plot_path,
        },
    }
    Path(args.ensemble_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.ensemble_metrics_path).write_text(
        json.dumps(ensemble_bundle, indent=2), encoding="utf-8"
    )

    # Append ensemble rows into the full metrics table for the separate PNG.
    ensemble_rows = pd.DataFrame(
        [
            {
                "model": ensemble_bundle["vote"]["model"],
                "accuracy": ensemble_bundle["vote"]["accuracy"],
                "precision_macro": ensemble_bundle["vote"]["precision_macro"],
                "precision_weighted": ensemble_bundle["vote"]["precision_weighted"],
                "recall_macro": ensemble_bundle["vote"]["recall_macro"],
                "recall_weighted": ensemble_bundle["vote"]["recall_weighted"],
                "f1_macro": ensemble_bundle["vote"]["macro_f1"],
                "f1_weighted": ensemble_bundle["vote"]["weighted_f1"],
            },
            {
                "model": ensemble_bundle["weighted"]["model"],
                "accuracy": ensemble_bundle["weighted"]["accuracy"],
                "precision_macro": ensemble_bundle["weighted"]["precision_macro"],
                "precision_weighted": ensemble_bundle["weighted"]["precision_weighted"],
                "recall_macro": ensemble_bundle["weighted"]["recall_macro"],
                "recall_weighted": ensemble_bundle["weighted"]["recall_weighted"],
                "f1_macro": ensemble_bundle["weighted"]["macro_f1"],
                "f1_weighted": ensemble_bundle["weighted"]["weighted_f1"],
                "weights_by_model": json.dumps(
                    ensemble_bundle["weighted"]["weights_by_model"], sort_keys=True
                ),
                "weight_source": ensemble_bundle["weighted"]["weight_source"],
            },
        ]
    )
    full_metrics_df = pd.concat([full_metrics_df, ensemble_rows], ignore_index=True)

    # Save all-stats table PNG (readable: no long paths, compact weights).
    def save_stats_table_png(df: pd.DataFrame, output_path: str):
        weight_key_short = {
            "nn": "nn",
            "svm": "sv",
            "boosted_tree": "bt",
            "regression": "rg",
        }

        def _format_weights(val) -> str:
            """Very compact: nn=.24,sv=.24,bt=.24,rg=.28 (split across two lines if still long)."""
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return ""

            def _from_dict(d: dict) -> str:
                parts = [
                    f"{weight_key_short.get(k, k[:2])}={float(v):.2f}"
                    for k, v in sorted(d.items(), key=lambda kv: kv[0])
                ]
                return ",".join(parts)

            if isinstance(val, dict):
                s = _from_dict(val)
            elif isinstance(val, str) and val.strip().startswith("{"):
                try:
                    s = _from_dict(json.loads(val))
                except (json.JSONDecodeError, TypeError, ValueError):
                    s = val[:40] + ("…" if len(val) > 40 else "")
            else:
                s = str(val).replace(" ", "")
                s = s[:40] + ("…" if len(s) > 40 else "")

            if len(s) > 22 and "," in s:
                chunks = s.split(",")
                mid = (len(chunks) + 1) // 2
                return ",\n".join([",".join(chunks[:mid]), ",".join(chunks[mid:])])
            return s

        def _cell_str(val, max_len: int = 14) -> str:
            if val is None or pd.isna(val):
                return "—"
            if isinstance(val, (np.floating, float)):
                if pd.isna(val):
                    return "—"
                return f"{float(val):.4f}"
            if isinstance(val, (np.integer, int)):
                return str(int(val))
            s = str(val).replace("\n", " ")
            return s[:max_len] + ("…" if len(s) > max_len else "")

        table_df = df.copy()
        drop_cols = {
            "plot_path",
        }
        for c in list(table_df.columns):
            if c in drop_cols or c.endswith("_path"):
                table_df = table_df.drop(columns=[c], errors="ignore")

        if "weights_by_model" in table_df.columns:
            table_df["weights"] = table_df["weights_by_model"].map(_format_weights)
            table_df = table_df.drop(columns=["weights_by_model"])

        preferred = [
            "model",
            "accuracy",
            "precision_macro",
            "precision_weighted",
            "recall_macro",
            "recall_weighted",
            "f1_macro",
            "f1_weighted",
            "r2",
            "rmse",
            "mae",
            "weight_source",
            "weights",
            "adjusted_rand_index",
            "normalized_mutual_info",
            "silhouette_score",
        ]
        ordered_cols = [c for c in preferred if c in table_df.columns] + [
            c for c in table_df.columns if c not in preferred
        ]
        table_df = table_df[ordered_cols]

        max_lens = {c: min(max(len(str(c)), 10), 22) for c in table_df.columns}
        for c in table_df.columns:
            if c == "weights":
                max_lens[c] = 26
            if c == "model":
                max_lens[c] = min(34, max(max_lens[c], 22))

        display_df = table_df.copy()
        for c in table_df.columns:
            ml = max_lens[c]
            if c == "weights":
                display_df[c] = table_df[c].astype(str)
                continue
            if pd.api.types.is_numeric_dtype(table_df[c]):
                display_df[c] = table_df[c].map(lambda x, _ml=ml: _cell_str(x, max_len=_ml))
            else:
                display_df[c] = table_df[c].map(lambda x, _ml=ml: _cell_str(x, max_len=_ml))

        n_rows, n_cols = display_df.shape
        fig_h = max(4.0, 0.42 * (n_rows + 1) + 1.1)
        fig_w = max(14.0, min(34.0, 1.35 * n_cols + 6.0))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        tbl = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.15, 1.35)

        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(rotation=0, ha="center", va="center", fontsize=8)
            else:
                cell.PAD = 0.02

        plt.title("All model metrics", fontsize=12, pad=12)
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outp, dpi=220, bbox_inches="tight")
        plt.close()

    save_stats_table_png(full_metrics_df, args.stats_table_plot_path)
    print(f"Saved metrics table PNG: {args.stats_table_plot_path}")
    save_confusion_plot(y_true, y_vote, args.ensemble_plot_path, "Ensemble: Majority Vote")
    save_confusion_plot(
        y_true, y_wtd, args.ensemble_weighted_plot_path, "Ensemble: Weighted Vote"
    )
    print(f"Saved ensemble metrics: {args.ensemble_metrics_path}")

    model_entries = [
        ("Neural Net", pred_frames[0]["y_true_class"], pred_frames[0]["y_pred_class"], "Blues", None),
        ("KMeans", pred_frames[1]["y_true_class"], pred_frames[1]["y_pred_class"], "Purples", None),
        ("Linear SVM", pred_frames[2]["y_true_class"], pred_frames[2]["y_pred_class"], "YlOrBr", None),
        ("Boosted Tree", pred_frames[3]["y_true_class"], pred_frames[3]["y_pred_class"], "Greens", None),
        (
            "Regression→Class",
            pred_frames[4]["y_true_class"],
            pred_frames[4]["y_pred_class"],
            "Oranges",
            (
                f"R2={float(metrics_by_model['regression']['r2']):.4f} "
                f"RMSE={float(metrics_by_model['regression']['rmse']):.4f}"
                if "regression" in metrics_by_model
                else None
            ),
        ),
        (
            "Ensemble (equal-weight / majority)",
            ensemble_df["y_true_class"],
            ensemble_df["ensemble_vote"],
            "BuGn",
            None,
        ),
        (
            "Ensemble (weighted vote)",
            ensemble_df["y_true_class"],
            ensemble_df["ensemble_weighted"],
            "PuRd",
            None,
        ),
    ]
    details = (
        "Weighted ensemble weights "
        f"(source={weight_source}): "
        + ", ".join(f"{k}={v:.3f}" for k, v in weights_by_model.items())
    )
    save_comparison_dashboard(model_entries, args.dashboard_plot_path, weighted_details=details)
    print(f"Saved comparison dashboard: {args.dashboard_plot_path}")


if __name__ == "__main__":
    main()
