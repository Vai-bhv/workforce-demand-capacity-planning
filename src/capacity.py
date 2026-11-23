"""Assumptions-based daily capacity for a single pool of scheduled staff."""

import math
from fractions import Fraction
from numbers import Integral, Real

import numpy as np
import pandas as pd


def create_capacity_plan(
    forecast: pd.DataFrame,
    scheduled_staff: int,
    absence_rate: float,
    requests_per_available_person_day: float,
) -> pd.DataFrame:
    """Estimate headcount and capacity for forecast arrivals under a constant scenario."""
    if isinstance(scheduled_staff, bool) or not isinstance(scheduled_staff, Integral) or scheduled_staff < 0:
        raise ValueError("Scheduled staff must be a nonnegative integer.")
    if (
        isinstance(absence_rate, bool)
        or not isinstance(absence_rate, Real)
        or not math.isfinite(absence_rate)
        or not 0 <= absence_rate <= 1
    ):
        raise ValueError("Absence rate must be between 0 and 1.")
    productivity = requests_per_available_person_day
    if (
        isinstance(productivity, bool)
        or not isinstance(productivity, Real)
        or not math.isfinite(productivity)
        or productivity < 0
    ):
        raise ValueError("Productivity must be finite and nonnegative.")

    if forecast.empty or "predicted_requests" not in forecast:
        raise ValueError("Forecast must contain predicted requests.")
    arrivals = pd.to_numeric(forecast["predicted_requests"], errors="coerce")
    if not (np.isfinite(arrivals) & arrivals.ge(0)).fillna(False).all():
        raise ValueError("Predicted arrivals must be finite and nonnegative.")

    # Use the supplied decimal values so exact multiples stay exact when rounding up.
    availability = 1 - Fraction(str(absence_rate))
    capacity_per_person = availability * Fraction(str(productivity))
    expected_available_staff = scheduled_staff * availability
    daily_capacity = scheduled_staff * capacity_per_person

    required = [
        math.ceil(Fraction(str(demand)) / capacity_per_person)
        if capacity_per_person else (0 if demand == 0 else None)
        for demand in arrivals
    ]
    if (
        daily_capacity > np.finfo(float).max
        or scheduled_staff > np.iinfo(np.int64).max
        or any(staff is not None and staff > np.iinfo(np.int64).max for staff in required)
    ):
        raise ValueError("Scenario exceeds the supported capacity or headcount range.")
    incremental = [None if staff is None else max(0, staff - scheduled_staff) for staff in required]

    plan = forecast.copy()
    plan["predicted_requests"] = arrivals
    plan["scheduled_staff"] = scheduled_staff
    plan["absence_rate"] = absence_rate
    plan["requests_per_available_person_day"] = productivity
    plan["expected_available_staff"] = float(expected_available_staff)
    plan["daily_capacity"] = float(daily_capacity)
    plan["capacity_gap"] = plan["daily_capacity"] - arrivals
    plan["required_scheduled_staff"] = pd.array(required, dtype="Int64")
    plan["additional_scheduled_staff"] = pd.array(incremental, dtype="Int64")
    return plan
