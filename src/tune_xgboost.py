from pathlib import Path
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


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
    """Load and chronologically sort the engineered dataset."""
    path = (
        get_project_root()
        / "data"
        / "processed"
        / "feature_dataset.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {path}."
        )

    dataframe = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    return dataframe.sort_values(
        ["date", "region"]
    ).reset_index(drop=True)


def create_final_holdout(
    dataframe: pd.DataFrame,
    test_days: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reserve the latest calendar period as an untouched test set.
    """
    maximum_date = dataframe["date"].max()
    split_date = maximum_date - pd.Timedelta(days=test_days)

    training_data = dataframe[
        dataframe["date"] <= split_date
    ].copy()

    test_data = dataframe[
        dataframe["date"] > split_date
    ].copy()

    return training_data, test_data


def create_date_based_cv(
    training_data: pd.DataFrame,
    number_of_splits: int = 4,
    validation_days: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create expanding-window cross-validation folds by calendar date.

    Every region belonging to the same date remains in the same fold.
    """
    unique_dates = np.array(
        sorted(training_data["date"].unique())
    )

    required_validation_dates = (
        number_of_splits * validation_days
    )

    if len(unique_dates) <= required_validation_dates:
        raise ValueError(
            "Not enough dates for the requested CV configuration."
        )

    first_validation_start = (
        len(unique_dates) - required_validation_dates
    )

    folds = []

    for fold_number in range(number_of_splits):
        validation_start_position = (
            first_validation_start
            + fold_number * validation_days
        )

        validation_end_position = (
            validation_start_position
            + validation_days
        )

        train_dates = unique_dates[
            :validation_start_position
        ]

        validation_dates = unique_dates[
            validation_start_position:validation_end_position
        ]

        train_indices = training_data.index[
            training_data["date"].isin(train_dates)
        ].to_numpy()

        validation_indices = training_data.index[
            training_data["date"].isin(validation_dates)
        ].to_numpy()

        folds.append(
            (train_indices, validation_indices)
        )

    return folds


def build_pipeline() -> Pipeline:
    """Create preprocessing and XGBoost model pipeline."""
    preprocessor = ColumnTransformer(
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

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_parameter_distributions() -> dict:
    """Return the hyperparameter search space."""
    return {
        "model__n_estimators": [
            200,
            300,
            500,
            700,
        ],
        "model__max_depth": [
            3,
            4,
            5,
            6,
            8,
        ],
        "model__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.08,
            0.1,
        ],
        "model__subsample": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "model__colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0,
        ],
        "model__min_child_weight": [
            1,
            3,
            5,
            8,
        ],
        "model__reg_alpha": [
            0,
            0.01,
            0.1,
            0.5,
        ],
        "model__reg_lambda": [
            0.5,
            1.0,
            2.0,
            5.0,
        ],
    }


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate final holdout metrics."""
    predictions = np.maximum(predicted, 0)

    mae = mean_absolute_error(
        actual,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions,
        )
    )

    denominator = np.abs(actual).sum()

    wape = (
        np.abs(actual - predictions).sum()
        / denominator
        * 100
        if denominator != 0
        else np.nan
    )

    bias = float(
        np.mean(predictions - actual)
    )

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "wape": round(float(wape), 4),
        "bias": round(bias, 4),
    }


def display_cv_folds(
    training_data: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Print the date range represented by every fold."""
    print("Cross-validation configuration")
    print("------------------------------")

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(folds, start=1):

        train_dates = training_data.loc[
            train_indices,
            "date",
        ]

        validation_dates = training_data.loc[
            validation_indices,
            "date",
        ]

        print(
            f"Fold {fold_number}: "
            f"train {train_dates.min().date()} "
            f"to {train_dates.max().date()} | "
            f"validate {validation_dates.min().date()} "
            f"to {validation_dates.max().date()}"
        )


def save_outputs(
    search: RandomizedSearchCV,
    metrics: dict[str, float],
    test_data: pd.DataFrame,
    predictions: np.ndarray,
) -> None:
    """Save the tuned model, parameters, results, and predictions."""
    project_root = get_project_root()

    models_directory = project_root / "models"
    reports_directory = project_root / "reports"

    models_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        models_directory
        / "xgboost_tuned.pkl"
    )

    with model_path.open("wb") as file:
        pickle.dump(
            search.best_estimator_,
            file,
        )

    summary = {
        "best_parameters": search.best_params_,
        "best_cv_mae": round(
            float(-search.best_score_),
            4,
        ),
        "final_test_metrics": metrics,
    }

    with (
        reports_directory
        / "xgboost_tuning_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    cv_results = pd.DataFrame(
        search.cv_results_
    ).sort_values("rank_test_score")

    cv_results.to_csv(
        reports_directory
        / "xgboost_cv_results.csv",
        index=False,
    )

    prediction_output = test_data[
        [
            "date",
            "region",
            TARGET_COLUMN,
        ]
    ].copy()

    prediction_output[
        "xgboost_prediction"
    ] = np.maximum(predictions, 0)

    prediction_output.to_csv(
        reports_directory
        / "xgboost_test_predictions.csv",
        index=False,
    )

    print(f"\nSaved model: {model_path}")


def main() -> None:
    """Tune XGBoost and evaluate it on the untouched holdout."""
    dataframe = load_feature_data()

    training_data, test_data = create_final_holdout(
        dataframe
    )

    # Reset indices so CV indices refer directly to X_train rows.
    training_data = training_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    folds = create_date_based_cv(
        training_data,
        number_of_splits=4,
        validation_days=60,
    )

    display_cv_folds(
        training_data,
        folds,
    )

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data[TARGET_COLUMN]

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    search = RandomizedSearchCV(
        estimator=build_pipeline(),
        param_distributions=get_parameter_distributions(),
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=folds,
        random_state=42,
        n_jobs=-1,
        verbose=2,
        refit=True,
        return_train_score=True,
    )

    print("\nStarting XGBoost hyperparameter search...")
    start_time = time.time()

    search.fit(
        X_train,
        y_train,
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60

    predictions = search.predict(X_test)

    metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("\nXGBoost tuning completed")
    print("------------------------")
    print(
        f"Runtime: {elapsed_minutes:.2f} minutes"
    )
    print(
        f"Best cross-validation MAE: "
        f"{-search.best_score_:.4f}"
    )
    print(f"Best parameters: {search.best_params_}")

    print("\nFinal untouched test metrics")
    print("----------------------------")

    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value}")

    save_outputs(
        search,
        metrics,
        test_data,
        predictions,
    )


if __name__ == "__main__":
    main()