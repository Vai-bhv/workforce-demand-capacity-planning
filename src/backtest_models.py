"""Candidate forecasts that receive only the daily panel observed at issuance."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from .baseline import forecast_seasonal_naive


HORIZONS = (7, 14, 30)
MAX_HORIZON = max(HORIZONS)
WEEKDAY_WEEKS = 4
LOOKBACK_DAYS = 7 * WEEKDAY_WEEKS
MODEL_NAMES = ("seasonal_naive", "weekday_average", "calendar_ridge", "xgboost_direct")
RIDGE_ALPHA = 10.0
XGBOOST_PARAMETERS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cpu",
    "n_estimators": 160,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "random_state": 42,
    "n_jobs": 1,
}


def calendar_features(dates: pd.DatetimeIndex, regions: pd.Index, region_indices: np.ndarray) -> pd.DataFrame:
    """Build features from target dates and region alone."""
    trend = (dates - pd.Timestamp("2023-01-01")).days.to_numpy() / 365.25
    features = {
        "trend_years": trend,
        "month_end": dates.is_month_end.astype(int),
        "year_end_holiday": (
            ((dates.month == 12) & (dates.day >= 20)) | ((dates.month == 1) & (dates.day <= 3))
        ).astype(int),
    }
    features.update({f"weekday_{day}": (dates.dayofweek == day).astype(int) for day in range(7)})
    features.update({f"month_{month}": (dates.month == month).astype(int) for month in range(1, 13)})
    for index, region in enumerate(regions):
        member = (region_indices == index).astype(int)
        features[f"region_{region}"] = member
        features[f"trend_{region}"] = member * trend
    return pd.DataFrame(features)


def weekday_average(history: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Use the four most recent observations of each target weekday."""
    if len(history) < LOOKBACK_DAYS:
        raise ValueError(f"Weekday average requires {LOOKBACK_DAYS} observed days.")
    recent = history.tail(LOOKBACK_DAYS)
    weekdays = recent.groupby(recent.index.dayofweek).mean()
    return weekdays.loc[dates.dayofweek].to_numpy()


def direct_rows(history: pd.DataFrame, issue_positions: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build direct-model features in issue, lead, region order and return label indices."""
    if len(issue_positions) == 0 or np.any(issue_positions < LOOKBACK_DAYS - 1):
        raise ValueError("Every direct issue needs four complete weeks of observed arrivals.")
    if np.any(issue_positions >= len(history)):
        raise ValueError("Direct issue must be inside observed history.")
    issues, leads, region_indices = (
        values.ravel()
        for values in np.meshgrid(
            issue_positions, np.arange(1, MAX_HORIZON + 1), np.arange(len(history.columns)), indexing="ij"
        )
    )
    target_indices = issues + leads
    dates = history.index[0] + pd.to_timedelta(target_indices, unit="D")
    features = calendar_features(dates, history.columns, region_indices)
    features["lead"] = leads

    # The matching weekday at or before each issue. The next three taps move back by whole weeks.
    weekday_positions = issues + (leads - 1) % 7 - 6
    demand = history.to_numpy()
    same_weekday = demand[weekday_positions[:, None] - 7 * np.arange(WEEKDAY_WEEKS), region_indices[:, None]]
    features["last_same_weekday"] = same_weekday[:, 0]
    features["mean_same_weekday_4"] = same_weekday.mean(axis=1)
    for window in (7, LOOKBACK_DAYS):
        preceding = demand[issues[:, None] - np.arange(window), region_indices[:, None]]
        features[f"mean_arrivals_{window}"] = preceding.mean(axis=1)
    return features, target_indices, region_indices


def direct_training_data(history: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Use complete historical issues on the same weekday as the evaluation origin."""
    last_observed = len(history) - 1
    issues = np.arange(last_observed, LOOKBACK_DAYS - 2, -7)[::-1]
    issues = issues[issues + MAX_HORIZON <= last_observed]
    features, targets, regions = direct_rows(history, issues)
    return features, history.to_numpy()[targets, regions]


def predict_candidates(history: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fit on observed arrivals and forecast the next 30 days for all regions."""
    origin = history.index[-1]
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=MAX_HORIZON)
    region_indices = np.arange(len(history.columns))
    history_long = history.reset_index().melt(id_vars="date", var_name="region", value_name="incoming_requests")
    naive = np.column_stack([
        forecast_seasonal_naive(history_long, region, MAX_HORIZON, origin)["predicted_requests"]
        for region in history.columns
    ])

    training_dates = history.index.repeat(len(region_indices))
    train_calendar = calendar_features(training_dates, history.columns, np.tile(region_indices, len(history)))
    future_calendar = calendar_features(
        dates.repeat(len(region_indices)), history.columns, np.tile(region_indices, MAX_HORIZON)
    )
    ridge = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(train_calendar, history.to_numpy().ravel())

    train_direct, labels = direct_training_data(history)
    future_direct, _, _ = direct_rows(history, np.array([len(history) - 1]))
    challenger = XGBRegressor(**XGBOOST_PARAMETERS).fit(train_direct, labels)

    predictions = {
        "seasonal_naive": naive,
        "weekday_average": weekday_average(history, dates),
        "calendar_ridge": ridge.predict(future_calendar).reshape(naive.shape),
        "xgboost_direct": challenger.predict(future_direct).reshape(naive.shape),
    }
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Candidates must return finite predictions.")
    return {name: np.maximum(np.asarray(values, dtype=float), 0) for name, values in predictions.items()}
