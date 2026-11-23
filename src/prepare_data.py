"""Validate and prepare the tracked synthetic data for the demo and experiments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_set


KEY_COLUMNS = ["date", "region", "request_type"]
REQUEST_COUNTS = ["incoming_requests", "completed_requests"]
REGIONAL_COUNTS = ["total_daily_requests", "daily_backlog", "available_staff"]
CALENDAR_COLUMNS = [
    "is_weekend", "is_month_end", "is_holiday_period", "demand_spike",
    "day_of_week", "week_of_year", "month", "quarter", "year",
]
DAILY_ATTRIBUTES = ["daily_backlog", "available_staff", "staff_absence_rate", *CALENDAR_COLUMNS]
REQUIRED_COLUMNS = KEY_COLUMNS + REQUEST_COUNTS + REGIONAL_COUNTS + ["staff_absence_rate"] + CALENDAR_COLUMNS


def aggregate_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a complete region-day panel from request-type rows."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if raw.empty:
        raise ValueError("Raw data must contain observations.")

    raw = raw[REQUIRED_COLUMNS].copy()
    dates = raw["date"].astype("string")
    if not dates.str.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", na=False).all():
        raise ValueError("Dates must use YYYY-MM-DD.")
    raw["date"] = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    for column in ["region", "request_type"]:
        if raw[column].astype("string").str.strip().fillna("").eq("").any():
            raise ValueError(f"{column} must be present and nonempty.")
    if raw.duplicated(KEY_COLUMNS).any():
        raise ValueError("Duplicate date/region/request-type key.")

    expected_dates = (raw["date"].max() - raw["date"].min()).days + 1
    expected_rows = expected_dates * raw["region"].nunique() * raw["request_type"].nunique()
    if len(raw) != expected_rows:
        raise ValueError("Expected every region and request type on every calendar date.")

    for column in REQUEST_COUNTS + REGIONAL_COUNTS + CALENDAR_COLUMNS:
        values = pd.to_numeric(raw[column], errors="coerce")
        valid_counts = np.isfinite(values) & values.ge(0) & values.mod(1).eq(0)
        if not valid_counts.fillna(False).all() or values.ge(2**63).any():
            raise ValueError(f"{column} must contain finite nonnegative integers.")
        raw[column] = values.astype("int64")
    absence = pd.to_numeric(raw["staff_absence_rate"], errors="coerce")
    if not absence.between(0, 1).fillna(False).all():
        raise ValueError("staff_absence_rate must be between 0 and 1.")
    raw["staff_absence_rate"] = absence

    groups = raw.groupby(["date", "region"], sort=True)
    if groups[DAILY_ATTRIBUTES + ["total_daily_requests"]].nunique().gt(1).any().any():
        raise ValueError("Repeated region-day attributes disagree across request types.")
    daily = groups.agg({
        **dict.fromkeys(REQUEST_COUNTS, "sum"),
        **dict.fromkeys(DAILY_ATTRIBUTES, "first"),
    })
    if not daily["incoming_requests"].eq(groups["total_daily_requests"].first()).all():
        raise ValueError("Summed arrivals disagree with total_daily_requests.")
    return daily.reset_index()


def count_backlog_discrepancies(daily: pd.DataFrame) -> int:
    """Compare recorded completions with changes in backlog between observed dates."""
    ordered = daily.sort_values(["region", "date"])
    opening = ordered.groupby("region")["daily_backlog"].shift(1)
    implied_completions = opening + ordered["incoming_requests"] - ordered["daily_backlog"]
    return int((opening.notna() & implied_completions.ne(ordered["completed_requests"])).sum())


def prepare_data(input_path: Path, output_dir: Path) -> tuple[int, int, int]:
    raw = pd.read_csv(input_path, dtype=dict.fromkeys(KEY_COLUMNS, "string"))
    daily = aggregate_daily(raw)
    features = build_feature_set(daily)
    features = features.loc[features["incoming_requests_lag_30"].notna()]
    discrepancies = count_backlog_discrepancies(daily)

    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "daily_regional_demand.csv", index=False)
    features.to_csv(output_dir / "feature_dataset.csv", index=False)
    return len(daily), len(features), discrepancies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        daily, features, discrepancies = prepare_data(args.input, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(f"Wrote {daily:,} regional rows and {features:,} feature rows to {args.output_dir}.")
    if discrepancies:
        print(
            f"Warning: recorded completions do not reconcile with backlog on "
            f"{discrepancies:,} region-days with an observed opening backlog. "
            "Recorded completions and backlog were preserved."
        )


if __name__ == "__main__":
    main()
