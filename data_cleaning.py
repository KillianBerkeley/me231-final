"""Shared data loading, cleaning, encoding, and splitting for all model scripts."""

from __future__ import annotations
from pathlib import Path

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_LOCAL_DATA = "tech_mental_health_burnout.csv"
KAGGLE_DATASET = "suhanigupta04/employee-mental-health-and-burnout-dataset"


def resolve_data_path(data_path: str | None) -> str:
    """Resolve dataset path: explicit path -> local CSV -> Kaggle download."""
    if data_path:
        explicit = Path(data_path)
        if explicit.exists():
            return str(explicit)
        raise FileNotFoundError(f"Provided data path not found: {data_path}")

    local = Path(DEFAULT_LOCAL_DATA)
    if local.exists():
        print(f"Using local dataset: {local}")
        return str(local)

    print("Local CSV not found. Downloading dataset from Kaggle...")
    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    kaggle_csv = dataset_root / DEFAULT_LOCAL_DATA
    if not kaggle_csv.exists():
        raise FileNotFoundError(f"Dataset downloaded but CSV missing: {kaggle_csv}")
    print(f"Using Kaggle dataset file: {kaggle_csv}")
    return str(kaggle_csv)


def score_to_level(values, low_cut: float, high_cut: float):
    """Convert burnout_score values to class labels 0,1,2."""
    return (
        pd.Series(values)
        .apply(lambda v: 0 if v <= low_cut else (1 if v <= high_cut else 2))
        .to_numpy()
    )


def load_clean_split(
    data_path: str | None,
    target_column: str,
    test_size: float,
    random_state: int,
    therapy_threshold: float = 0.8,
):
    """Load CSV, apply cleaning/encoding/scaling, and return train/test split."""
    print("=======================")
    print("LOADING DATA")
    print("=======================")
    df = pd.read_csv(resolve_data_path(data_path))
    df = df.dropna()

    if target_column not in df.columns:
        raise ValueError(f"target column '{target_column}' not found in data")

    print("=======================")
    print("CLEANING DATA")
    print("=======================")
    # Drop impossible work hours that can distort model behavior.
    if "work_hours_per_week" in df.columns:
        bad_hours = int((df["work_hours_per_week"] > 168).sum())
        if bad_hours > 0:
            print(f"Dropping {bad_hours} rows with work_hours_per_week > 168")
            df = df[df["work_hours_per_week"] <= 168]

    # Prevent leakage between burnout targets.
    if target_column == "burnout_level" and "burnout_score" in df.columns:
        df = df.drop(columns=["burnout_score"])
    if target_column == "burnout_score" and "burnout_level" in df.columns:
        df = df.drop(columns=["burnout_level"])

    # Optional redundancy check for therapy features.
    therapy_cols = {"has_therapy", "seeks_professional_help"}
    if therapy_cols.issubset(df.columns):
        corr = df["has_therapy"].corr(df["seeks_professional_help"])
        corr_abs = abs(float(corr)) if pd.notna(corr) else 0.0
        if corr_abs >= therapy_threshold:
            print(
                "Dropping seeks_professional_help due to high correlation with "
                f"has_therapy (|corr|={corr_abs:.3f} >= {therapy_threshold:.3f})"
            )
            df = df.drop(columns=["seeks_professional_help"])
        else:
            print(
                "Keeping both has_therapy and seeks_professional_help "
                f"(|corr|={corr_abs:.3f} < {therapy_threshold:.3f})"
            )

    # Ordinal encode burnout_level if needed as target.
    if target_column == "burnout_level":
        level_map = {"Low": 0, "Moderate": 1, "High": 2}
        df[target_column] = df[target_column].map(level_map)
        if df[target_column].isna().any():
            raise ValueError("burnout_level contains values outside Low/Moderate/High")

    x = df.drop(columns=[target_column])
    y = df[target_column]

    print("=======================")
    print("ENCODING DATA")
    print("=======================")
    # One-hot encode nominal categorical columns.
    categorical_cols = [c for c in ["gender", "job_role", "company_size", "work_mode"] if c in x.columns]
    if categorical_cols:
        x = pd.get_dummies(x, columns=categorical_cols, drop_first=False)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    print("=======================")
    print("SCALING DATA")
    print("=======================")
    # Standardize numeric columns (fit on train only).
    numeric_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    if numeric_cols:
        # Build float DataFrames first to avoid pandas dtype assignment errors.
        x_train = x_train.astype(float)
        x_test = x_test.astype(float)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train[numeric_cols])
        x_test_scaled = scaler.transform(x_test[numeric_cols])
        x_train = pd.DataFrame(x_train_scaled, columns=numeric_cols, index=x_train.index)
        x_test = pd.DataFrame(x_test_scaled, columns=numeric_cols, index=x_test.index)

    return x_train, x_test, y_train, y_test
