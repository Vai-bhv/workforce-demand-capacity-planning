"""Origin-safe weekly seasonal-naïve forecasts for the interactive demo."""

from numbers import Integral

import numpy as np
import pandas as pd


def forecast_seasonal_naive(
    history: pd.DataFrame,
    region: str,
    horizon: int,
    origin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Repeat the last seven observed daily arrivals up to and including the origin."""
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("Forecast horizon must be a positive whole number of days.")
    regional = history.loc[history["region"].eq(region), ["date", "incoming_requests"]].copy()
    if regional.empty:
        raise ValueError(f"No history for region {region!r}.")

    regional["date"] = pd.to_datetime(regional["date"], errors="raise")
    origin = regional["date"].max() if origin is None else pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.normalize():
        raise ValueError("Forecast origin must be a calendar date.")
    last_week = pd.date_range(end=origin, periods=7, freq="D")
    observed = regional.loc[regional["date"].between(last_week[0], origin)].sort_values("date")
    if not pd.DatetimeIndex(observed["date"]).equals(last_week):
        raise ValueError("Forecast requires exactly one observation on each of the last seven days.")

    arrivals = pd.to_numeric(observed["incoming_requests"], errors="coerce")
    if not (np.isfinite(arrivals) & arrivals.ge(0)).fillna(False).all():
        raise ValueError("Observed arrivals must be finite and nonnegative.")
    return pd.DataFrame({
        "date": pd.date_range(start=origin + pd.Timedelta(days=1), periods=horizon, freq="D"),
        "region": region,
        "forecast_origin": origin,
        "predicted_requests": np.resize(arrivals.to_numpy(), horizon),
    })
