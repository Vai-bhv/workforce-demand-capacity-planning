from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

REGIONS = {
    "Europe": {
        "base_demand": 32,
        "staff": 9,
        "growth_rate": 0.00035,
    },
    "North America": {
        "base_demand": 38,
        "staff": 11,
        "growth_rate": 0.00045,
    },
    "Asia Pacific": {
        "base_demand": 27,
        "staff": 8,
        "growth_rate": 0.00055,
    },
}

REQUEST_TYPES = {
    "New Worker Request": 0.38,
    "Extension": 0.29,
    "Onboarding Support": 0.21,
    "Data Correction": 0.12,
}


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[1]


def create_date_dimension() -> pd.DataFrame:
    """Create a daily calendar containing useful forecasting features."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    calendar = pd.DataFrame({"date": dates})
    calendar["day_of_week"] = calendar["date"].dt.dayofweek
    calendar["day_name"] = calendar["date"].dt.day_name()
    calendar["week_of_year"] = calendar["date"].dt.isocalendar().week.astype(int)
    calendar["month"] = calendar["date"].dt.month
    calendar["quarter"] = calendar["date"].dt.quarter
    calendar["year"] = calendar["date"].dt.year
    calendar["is_weekend"] = calendar["day_of_week"].isin([5, 6]).astype(int)
    calendar["is_month_end"] = calendar["date"].dt.is_month_end.astype(int)
    calendar["is_month_start"] = calendar["date"].dt.is_month_start.astype(int)

    return calendar


def calculate_seasonality(date: pd.Timestamp) -> float:
    """Return a seasonal multiplier for workforce demand."""
    month = date.month
    day_of_week = date.dayofweek

    monthly_multiplier = {
        1: 1.18,
        2: 1.05,
        3: 1.10,
        4: 1.00,
        5: 1.04,
        6: 0.96,
        7: 0.78,
        8: 0.82,
        9: 1.12,
        10: 1.16,
        11: 1.10,
        12: 0.72,
    }[month]

    weekday_multiplier = {
        0: 1.15,
        1: 1.08,
        2: 1.03,
        3: 1.00,
        4: 0.90,
        5: 0.28,
        6: 0.18,
    }[day_of_week]

    return monthly_multiplier * weekday_multiplier


def generate_workforce_data() -> pd.DataFrame:
    """Generate synthetic workforce-demand and capacity data."""
    rng = np.random.default_rng(RANDOM_SEED)
    calendar = create_date_dimension()

    records = []

    for region, region_config in REGIONS.items():
        backlog = int(region_config["base_demand"] * 1.5)

        for row_number, calendar_row in calendar.iterrows():
            date = calendar_row["date"]

            seasonal_multiplier = calculate_seasonality(date)
            growth_multiplier = 1 + (
                region_config["growth_rate"] * row_number
            )

            month_end_multiplier = 1.18 if calendar_row["is_month_end"] else 1.0

            demand_spike = rng.random() < 0.025
            spike_multiplier = rng.uniform(1.35, 1.85) if demand_spike else 1.0

            holiday_period = (
                (date.month == 12 and date.day >= 20)
                or (date.month == 1 and date.day <= 3)
            )
            holiday_multiplier = 0.55 if holiday_period else 1.0

            expected_total_demand = (
                region_config["base_demand"]
                * seasonal_multiplier
                * growth_multiplier
                * month_end_multiplier
                * spike_multiplier
                * holiday_multiplier
            )

            total_requests = rng.poisson(max(expected_total_demand, 0.5))

            request_counts = rng.multinomial(
                total_requests,
                list(REQUEST_TYPES.values()),
            )

            staff_absence_rate = np.clip(
                rng.normal(loc=0.07, scale=0.035),
                0,
                0.25,
            )

            available_staff = max(
                1,
                round(
                    region_config["staff"]
                    * (1 - staff_absence_rate)
                ),
            )

            productivity_per_staff = rng.normal(loc=4.0, scale=0.35)

            completion_capacity = max(
                0,
                int(available_staff * productivity_per_staff),
            )

            completed_requests = min(
                backlog + total_requests,
                rng.poisson(max(completion_capacity, 0.5)),
            )

            backlog = max(
                0,
                backlog + total_requests - completed_requests,
            )

            for request_type, request_count in zip(
                REQUEST_TYPES.keys(),
                request_counts,
            ):
                records.append(
                    {
                        "date": date,
                        "region": region,
                        "request_type": request_type,
                        "incoming_requests": int(request_count),
                        "completed_requests": int(
                            round(
                                completed_requests
                                * (
                                    request_count / total_requests
                                    if total_requests > 0
                                    else 0
                                )
                            )
                        ),
                        "total_daily_requests": int(total_requests),
                        "daily_backlog": int(backlog),
                        "available_staff": int(available_staff),
                        "staff_absence_rate": round(
                            float(staff_absence_rate),
                            4,
                        ),
                        "is_weekend": int(calendar_row["is_weekend"]),
                        "is_month_end": int(calendar_row["is_month_end"]),
                        "is_holiday_period": int(holiday_period),
                        "demand_spike": int(demand_spike),
                        "day_of_week": int(calendar_row["day_of_week"]),
                        "week_of_year": int(calendar_row["week_of_year"]),
                        "month": int(calendar_row["month"]),
                        "quarter": int(calendar_row["quarter"]),
                        "year": int(calendar_row["year"]),
                    }
                )

    dataset = pd.DataFrame(records)

    return dataset.sort_values(
        ["date", "region", "request_type"]
    ).reset_index(drop=True)


def save_dataset(dataset: pd.DataFrame) -> Path:
    """Save the generated dataset to the raw-data directory."""
    output_path = (
        get_project_root()
        / "data"
        / "raw"
        / "workforce_demand.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    dataset = generate_workforce_data()
    output_path = save_dataset(dataset)

    print("Synthetic workforce dataset created successfully.")
    print(f"Saved to: {output_path}")
    print(f"Rows: {len(dataset):,}")
    print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    print()
    print(dataset.head())


if __name__ == "__main__":
    main()