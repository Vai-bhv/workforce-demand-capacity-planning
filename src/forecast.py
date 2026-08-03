from pathlib import Path
import pickle

import numpy as np
import pandas as pd


MODEL_NAME = "xgboost_tuned.pkl"

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


def load_model():
    """Load the tuned XGBoost pipeline."""
    model_path = (
        get_project_root()
        / "models"
        / MODEL_NAME
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run src/tune_xgboost.py first."
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


def load_historical_data() -> pd.DataFrame:
    """Load daily regional demand history."""
    data_path = (
        get_project_root()
        / "data"
        / "processed"
        / "daily_regional_demand.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {data_path}."
        )

    data = pd.read_csv(
        data_path,
        parse_dates=["date"],
    )

    return data.sort_values(
        ["region", "date"]
    ).reset_index(drop=True)


def is_holiday_period(date: pd.Timestamp) -> int:
    """
    Reproduce the holiday-period rule used during data generation.
    """
    return int(
        (date.month == 12 and date.day >= 20)
        or (date.month == 1 and date.day <= 3)
    )


def create_future_feature_row(
    date: pd.Timestamp,
    region: str,
    demand_history: list[float],
    time_index: int,
    available_staff: int,
    staff_absence_rate: float,
) -> pd.DataFrame:
    """Create one model-ready future observation."""

    if len(demand_history) < 30:
        raise ValueError(
            "At least 30 historical observations are required."
        )

    day_of_week = date.dayofweek
    month = date.month

    row = {
        "region": region,
        "available_staff": available_staff,
        "staff_absence_rate": staff_absence_rate,
        "is_weekend": int(day_of_week in [5, 6]),
        "is_month_end": int(date.is_month_end),
        "is_holiday_period": is_holiday_period(date),
        "day_of_week": day_of_week,
        "week_of_year": int(date.isocalendar().week),
        "month": month,
        "quarter": date.quarter,
        "year": date.year,
        "time_index": time_index,
        "day_sin": np.sin(
            2 * np.pi * day_of_week / 7
        ),
        "day_cos": np.cos(
            2 * np.pi * day_of_week / 7
        ),
        "month_sin": np.sin(
            2 * np.pi * month / 12
        ),
        "month_cos": np.cos(
            2 * np.pi * month / 12
        ),
        "incoming_requests_lag_1":
            demand_history[-1],
        "incoming_requests_lag_2":
            demand_history[-2],
        "incoming_requests_lag_7":
            demand_history[-7],
        "incoming_requests_lag_14":
            demand_history[-14],
        "incoming_requests_lag_30":
            demand_history[-30],
        "rolling_mean_7":
            float(np.mean(demand_history[-7:])),
        "rolling_mean_30":
            float(np.mean(demand_history[-30:])),
        "rolling_std_7":
            float(np.std(demand_history[-7:], ddof=1)),
    }

    return pd.DataFrame([row])[FEATURE_COLUMNS]


def forecast_region(
    region: str,
    forecast_days: int = 14,
    available_staff: int | None = None,
    staff_absence_rate: float | None = None,
) -> pd.DataFrame:
    """
    Generate a recursive demand forecast for one region.

    Each predicted day becomes part of the history used to predict
    subsequent days.
    """
    if forecast_days < 1 or forecast_days > 90:
        raise ValueError(
            "forecast_days must be between 1 and 90."
        )

    model = load_model()
    historical_data = load_historical_data()

    region_data = historical_data[
        historical_data["region"] == region
    ].sort_values("date")

    if region_data.empty:
        available_regions = sorted(
            historical_data["region"].unique()
        )

        raise ValueError(
            f"Unknown region '{region}'. "
            f"Available regions: {available_regions}"
        )

    if available_staff is None:
        available_staff = int(
            round(
                region_data
                .tail(30)["available_staff"]
                .median()
            )
        )

    if staff_absence_rate is None:
        staff_absence_rate = float(
            region_data
            .tail(30)["staff_absence_rate"]
            .mean()
        )

    demand_history = (
        region_data["incoming_requests"]
        .astype(float)
        .tolist()
    )

    final_historical_date = region_data["date"].max()
    initial_time_index = len(region_data)

    forecast_records = []

    for day_number in range(1, forecast_days + 1):
        future_date = (
            final_historical_date
            + pd.Timedelta(days=day_number)
        )

        feature_row = create_future_feature_row(
            date=future_date,
            region=region,
            demand_history=demand_history,
            time_index=(
                initial_time_index
                + day_number
                - 1
            ),
            available_staff=available_staff,
            staff_absence_rate=staff_absence_rate,
        )

        prediction = float(
            model.predict(feature_row)[0]
        )

        prediction = max(0.0, prediction)

        demand_history.append(prediction)

        forecast_records.append(
            {
                "date": future_date,
                "region": region,
                "predicted_requests": round(
                    prediction,
                    2,
                ),
                "available_staff_assumption":
                    available_staff,
                "absence_rate_assumption": round(
                    staff_absence_rate,
                    4,
                ),
            }
        )

    return pd.DataFrame(forecast_records)


def main() -> None:
    """Generate example forecasts for every region."""
    historical_data = load_historical_data()

    regions = sorted(
        historical_data["region"].unique()
    )

    all_forecasts = []

    for region in regions:
        region_forecast = forecast_region(
            region=region,
            forecast_days=14,
        )

        all_forecasts.append(region_forecast)

    forecast_output = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    output_path = (
        get_project_root()
        / "reports"
        / "future_forecast.csv"
    )

    forecast_output.to_csv(
        output_path,
        index=False,
    )

    print("Future forecast generated successfully.")
    print(f"Saved to: {output_path}")
    print()
    print(forecast_output.to_string(index=False))


if __name__ == "__main__":
    main()