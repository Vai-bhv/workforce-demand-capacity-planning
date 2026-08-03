from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


TARGET_COLUMN = "incoming_requests"
BEST_MODEL_NAME = "random_forest"

FEATURE_COLUMNS = [
    "region",
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


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_predictions() -> pd.DataFrame:
    """Load test-set predictions created during training."""
    path = (
        get_project_root()
        / "reports"
        / "test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found at {path}. "
            "Run src/train.py first."
        )

    return pd.read_csv(path, parse_dates=["date"])


def load_feature_data() -> pd.DataFrame:
    """Load the full engineered feature dataset."""
    path = (
        get_project_root()
        / "data"
        / "processed"
        / "feature_dataset.csv"
    )

    return pd.read_csv(path, parse_dates=["date"])


def load_model(model_name: str = BEST_MODEL_NAME):
    """Load a saved machine-learning pipeline."""
    path = (
        get_project_root()
        / "models"
        / f"{model_name}.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run training first."
        )

    with path.open("rb") as file:
        return pickle.load(file)


def calculate_wape(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Calculate weighted absolute percentage error.

    WAPE is more stable than MAPE when actual values contain zeros
    or very small numbers.
    """
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return float("nan")

    return (
        np.abs(actual - predicted).sum()
        / denominator
        * 100
    )


def calculate_bias(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    """
    Calculate average forecast bias.

    Positive bias means overprediction.
    Negative bias means underprediction.
    """
    return float((predicted - actual).mean())


def prepare_prediction_data(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create residual and absolute-error columns."""
    prediction_column = f"{BEST_MODEL_NAME}_prediction"

    predictions = predictions.copy()

    predictions["prediction"] = predictions[prediction_column]
    predictions["residual"] = (
        predictions[TARGET_COLUMN]
        - predictions["prediction"]
    )
    predictions["absolute_error"] = predictions["residual"].abs()
    predictions["squared_error"] = predictions["residual"] ** 2

    return predictions


def create_overall_summary(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Calculate overall performance metrics."""
    actual = predictions[TARGET_COLUMN]
    predicted = predictions["prediction"]

    return {
        "mae": round(
            float(mean_absolute_error(actual, predicted)),
            4,
        ),
        "rmse": round(
            float(
                np.sqrt(
                    mean_squared_error(actual, predicted)
                )
            ),
            4,
        ),
        "wape": round(
            float(calculate_wape(actual, predicted)),
            4,
        ),
        "bias": round(
            calculate_bias(actual, predicted),
            4,
        ),
    }


def create_region_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate model performance by region."""
    rows = []

    for region, group in predictions.groupby("region"):
        actual = group[TARGET_COLUMN]
        predicted = group["prediction"]

        rows.append(
            {
                "region": region,
                "observations": len(group),
                "mae": mean_absolute_error(
                    actual,
                    predicted,
                ),
                "rmse": np.sqrt(
                    mean_squared_error(
                        actual,
                        predicted,
                    )
                ),
                "wape": calculate_wape(
                    actual,
                    predicted,
                ),
                "bias": calculate_bias(
                    actual,
                    predicted,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("mae")
        .round(4)
    )


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot actual and predicted demand by region."""
    for region, group in predictions.groupby("region"):
        group = group.sort_values("date")

        figure, axis = plt.subplots(figsize=(14, 6))

        axis.plot(
            group["date"],
            group[TARGET_COLUMN],
            label="Actual",
        )

        axis.plot(
            group["date"],
            group["prediction"],
            label="Random Forest Prediction",
        )

        axis.set_title(
            f"Actual vs Predicted Demand — {region}"
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("Incoming Requests")
        axis.legend()
        axis.grid(alpha=0.3)

        figure.tight_layout()

        filename = (
            region.lower()
            .replace(" ", "_")
            + "_actual_vs_predicted.png"
        )

        figure.savefig(
            output_directory / filename,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)


def plot_residuals(
    predictions: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot residuals over time."""
    figure, axis = plt.subplots(figsize=(14, 6))

    axis.scatter(
        predictions["date"],
        predictions["residual"],
        alpha=0.65,
    )

    axis.axhline(
        y=0,
        linewidth=1.5,
    )

    axis.set_title("Random Forest Residuals Over Time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Actual − Predicted")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory / "residuals_over_time.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_error_distribution(
    predictions: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the distribution of residual values."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.hist(
        predictions["residual"],
        bins=30,
        edgecolor="black",
    )

    axis.axvline(
        x=0,
        linewidth=1.5,
    )

    axis.set_title("Distribution of Forecast Residuals")
    axis.set_xlabel("Residual: Actual − Predicted")
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory / "residual_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_error_by_region(
    region_summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Compare MAE across regions."""
    figure, axis = plt.subplots(figsize=(10, 6))

    axis.bar(
        region_summary["region"],
        region_summary["mae"],
    )

    axis.set_title("Mean Absolute Error by Region")
    axis.set_xlabel("Region")
    axis.set_ylabel("MAE")
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory / "mae_by_region.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def calculate_permutation_importance(
    model,
    feature_data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance on the test set.

    A feature is important when shuffling it causes model
    performance to deteriorate.
    """
    test_keys = predictions[["date", "region"]].copy()

    test_data = feature_data.merge(
        test_keys,
        on=["date", "region"],
        how="inner",
    )

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    importance_result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": (
                importance_result.importances_mean
            ),
            "importance_std": (
                importance_result.importances_std
            ),
        }
    )

    return importance_df.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Plot the ten most important features."""
    top_features = (
        importance_df
        .head(10)
        .sort_values("importance_mean")
    )

    figure, axis = plt.subplots(figsize=(10, 7))

    axis.barh(
        top_features["feature"],
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
    )

    axis.set_title(
        "Top 10 Features by Permutation Importance"
    )
    axis.set_xlabel(
        "Increase in MAE after feature shuffling"
    )
    axis.set_ylabel("Feature")
    axis.grid(axis="x", alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        output_directory / "feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)


def evaluate_model() -> None:
    """Run the complete evaluation workflow."""
    project_root = get_project_root()
    output_directory = project_root / "reports" / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions()
    predictions = prepare_prediction_data(predictions)

    feature_data = load_feature_data()
    model = load_model()

    overall_summary = create_overall_summary(predictions)
    region_summary = create_region_summary(predictions)

    print("Overall model evaluation")
    print("------------------------")

    for metric, value in overall_summary.items():
        print(f"{metric.upper()}: {value}")

    print("\nPerformance by region")
    print("---------------------")
    print(region_summary.to_string(index=False))

    region_summary.to_csv(
        project_root
        / "reports"
        / "region_metrics.csv",
        index=False,
    )

    with (
        project_root
        / "reports"
        / "evaluation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            overall_summary,
            file,
            indent=4,
        )

    predictions.to_csv(
        project_root
        / "reports"
        / "evaluated_predictions.csv",
        index=False,
    )

    plot_actual_vs_predicted(
        predictions,
        output_directory,
    )

    plot_residuals(
        predictions,
        output_directory,
    )

    plot_error_distribution(
        predictions,
        output_directory,
    )

    plot_error_by_region(
        region_summary,
        output_directory,
    )

    print("\nCalculating permutation importance...")

    importance_df = calculate_permutation_importance(
        model,
        feature_data,
        predictions,
    )

    importance_df.to_csv(
        project_root
        / "reports"
        / "feature_importance.csv",
        index=False,
    )

    plot_feature_importance(
        importance_df,
        output_directory,
    )

    print("\nTop 10 features")
    print("---------------")
    print(
        importance_df.head(10).to_string(index=False)
    )

    print(
        f"\nEvaluation completed. Figures saved to: "
        f"{output_directory}"
    )


if __name__ == "__main__":
    evaluate_model()