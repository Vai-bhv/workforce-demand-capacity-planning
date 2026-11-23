from pathlib import Path

import numpy as np
import pandas as pd


def load_processed_data(project_root: Path) -> pd.DataFrame:
    """
    Load the processed daily regional dataset.
    """

    path = (
        project_root
        / "data"
        / "processed"
        / "daily_regional_demand.csv"
    )

    df = pd.read_csv(path, parse_dates=["date"])

    return df


def create_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numerical trend feature.
    """

    df = df.copy()

    df = df.sort_values(["region", "date"])

    df["time_index"] = (
        df.groupby("region")
        .cumcount()
    )

    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclic calendar features.
    """

    df = df.copy()

    # Day of week

    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Month

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    return df


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags=(1, 2, 7, 14, 30),
) -> pd.DataFrame:

    df = df.copy()

    for lag in lags:

        df[f"{target_column}_lag_{lag}"] = (
            df.groupby("region")[target_column]
            .shift(lag)
        )

    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:

    df = df.copy()

    grouped = df.groupby("region")[target_column]

    df["rolling_mean_7"] = grouped.transform(
        lambda demand: demand.shift(1).rolling(7).mean()
    )
    df["rolling_mean_30"] = grouped.transform(
        lambda demand: demand.shift(1).rolling(30).mean()
    )
    df["rolling_std_7"] = grouped.transform(
        lambda demand: demand.shift(1).rolling(7).std()
    )

    return df


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    """

    df = create_time_index(df)

    df = create_calendar_features(df)

    df = create_lag_features(
        df,
        target_column="incoming_requests",
    )

    df = create_rolling_features(
        df,
        target_column="incoming_requests",
    )

    return df
