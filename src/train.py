from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "incoming_requests"

CATEGORICAL_FEATURES = [
    "region",
]

NUMERICAL_FEATURES = [
    "available_staff",
    "staff_absence_rate",
    "is_weekend",
    "is_month_end",
    "is_holiday_period",
    "day_of_week",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "time_index",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "incoming_requests_lag_1",
    "incoming_requests_lag_2",
    "incoming_requests_lag_7",
    "incoming_requests_lag_14",
    "incoming_requests_lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_feature_data() -> pd.DataFrame:
    """Load the engineered feature dataset."""
    data_path = (
        get_project_root()
        / "data"
        / "processed"
        / "feature_dataset.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {data_path}. "
            "Create it before training."
        )

    df = pd.read_csv(data_path, parse_dates=["date"])

    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN, "date"])
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df.sort_values(["date", "region"]).reset_index(drop=True)


def create_time_split(
    df: pd.DataFrame,
    test_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically.

    The most recent 90 calendar days are reserved for testing.
    """
    maximum_date = df["date"].max()
    split_date = maximum_date - pd.Timedelta(days=test_days)

    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty dataset.")

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numerical and categorical columns."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Create the initial regression models."""
    preprocessor = build_preprocessor()

    return {
        "linear_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    non_zero_mask = y_true != 0

    if non_zero_mask.any():
        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero_mask]
                    - y_pred[non_zero_mask]
                )
                / y_true[non_zero_mask]
            )
        ) * 100
    else:
        mape = np.nan

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 4),
    }


def evaluate_naive_baseline(
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Use demand from seven days earlier as the baseline forecast.

    A seven-day lag is suitable because workforce demand has
    strong weekly seasonality.
    """
    predictions = test_df[
        "incoming_requests_lag_7"
    ].to_numpy()

    metrics = calculate_metrics(
        test_df[TARGET_COLUMN],
        predictions,
    )

    return predictions, metrics


def save_model(
    model: Pipeline,
    model_name: str,
) -> Path:
    """Save a trained model as a pickle file."""
    model_directory = get_project_root() / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    model_path = model_directory / f"{model_name}.pkl"

    with model_path.open("wb") as file:
        pickle.dump(model, file)

    return model_path


def save_results(
    results: dict[str, dict[str, float]],
) -> Path:
    """Save model metrics to JSON."""
    reports_directory = get_project_root() / "reports"
    reports_directory.mkdir(parents=True, exist_ok=True)

    output_path = reports_directory / "model_metrics.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    return output_path


def save_test_predictions(
    test_df: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> Path:
    """Save actual values and model predictions."""
    output = test_df[
        ["date", "region", TARGET_COLUMN]
    ].copy()

    for model_name, model_predictions in predictions.items():
        output[f"{model_name}_prediction"] = np.maximum(
            model_predictions,
            0,
        )

    reports_directory = get_project_root() / "reports"
    reports_directory.mkdir(parents=True, exist_ok=True)

    output_path = reports_directory / "test_predictions.csv"
    output.to_csv(output_path, index=False)

    return output_path


def train_models() -> None:
    """Train, evaluate, and save all initial models."""
    df = load_feature_data()
    train_df, test_df = create_time_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    print("Training configuration")
    print("----------------------")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(
        f"Training period: {train_df['date'].min().date()} "
        f"to {train_df['date'].max().date()}"
    )
    print(
        f"Testing period: {test_df['date'].min().date()} "
        f"to {test_df['date'].max().date()}"
    )
    print()

    results = {}
    predictions = {}

    naive_predictions, naive_metrics = evaluate_naive_baseline(
        test_df
    )

    results["naive_lag_7"] = naive_metrics
    predictions["naive_lag_7"] = naive_predictions

    print(f"Naive baseline: {naive_metrics}")

    models = build_models()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        model_predictions = model.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            model_predictions,
        )

        results[model_name] = metrics
        predictions[model_name] = model_predictions

        model_path = save_model(model, model_name)

        print(f"Metrics: {metrics}")
        print(f"Saved model: {model_path}")

    metrics_path = save_results(results)
    predictions_path = save_test_predictions(
        test_df,
        predictions,
    )

    print("\nTraining completed.")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["mae"],
    )

    print("\nModel ranking by MAE")
    print("--------------------")

    for position, (model_name, metrics) in enumerate(
        ranking,
        start=1,
    ):
        print(
            f"{position}. {model_name}: "
            f"MAE={metrics['mae']}, "
            f"RMSE={metrics['rmse']}"
        )


if __name__ == "__main__":
    train_models()