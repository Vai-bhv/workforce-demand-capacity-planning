"""Budget-matched 14-day staffing episodes on the overlapping Phase 2 final windows."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from numbers import Integral
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
import xgboost

from .backtest import PEAK_QUANTILE, benchmark_origins, daily_panel, evaluate_origins
from .generate_data import REGIONS
from .prepare_data import KEY_COLUMNS, aggregate_daily


HORIZON = 14
FORECAST_MODELS = ("seasonal_naive", "xgboost_direct")
ABSENCE_PERCENT = 7
PRODUCTIVITY = 4
# Integer hundredths of an expected request keep the backlog recurrence exact.
CAPACITY_HUNDREDTHS_PER_STAFF = (100 - ABSENCE_PERCENT) * PRODUCTIVITY
OPENING_BACKLOG = 0
BACKLOG_THRESHOLD = 20
STAFF_FLEX = 2
EPISODE_KEY = ["forecast_origin", "region"]


def allocate_staff(budget: int, minimum: int, maximum: int, prediction=None) -> np.ndarray:
    """Allocate integer staff-days over 14 dates, with earliest-date priority on ties."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (budget, minimum, maximum)):
        raise ValueError("Budget and bounds must be integers.")
    if minimum < 0 or maximum < minimum:
        raise ValueError("Expected nonnegative ordered daily staffing bounds.")
    if budget % HORIZON or not minimum * HORIZON <= budget <= maximum * HORIZON:
        raise ValueError("Budget must admit a constant integer schedule within the daily bounds.")
    if prediction is None:
        return np.full(HORIZON, budget // HORIZON, dtype=int)

    forecast = np.asarray(prediction, dtype=float)
    if forecast.shape != (HORIZON,) or not np.isfinite(forecast).all() or (forecast < 0).any():
        raise ValueError("Expected 14 finite nonnegative forecast arrivals.")
    staff = np.full(HORIZON, minimum, dtype=int)
    for _ in range(budget - minimum * HORIZON):
        # Match the precision of the Phase 2 prediction export for ties.
        shortfall = (forecast - staff * CAPACITY_HUNDREDTHS_PER_STAFF / 100).round(10)
        shortfall[staff == maximum] = -np.inf
        staff[shortfall.argmax()] += 1
    return staff


def replay_episode(arrivals: np.ndarray, schedule: np.ndarray) -> pd.DataFrame:
    """Replay a frozen schedule with continuous expected throughput, resetting at each issue."""
    arrivals, schedule = np.asarray(arrivals), np.asarray(schedule)
    for values in (arrivals, schedule):
        if values.shape != (HORIZON,) or not np.issubdtype(values.dtype, np.integer) or (values < 0).any():
            raise ValueError("Expected a complete 14-day vector of nonnegative integers.")

    backlog = 100 * OPENING_BACKLOG
    days = []
    for demand, staff in zip(arrivals.tolist(), schedule.tolist()):
        capacity = staff * CAPACITY_HUNDREDTHS_PER_STAFF
        work = backlog + 100 * demand
        processed = min(work, capacity)
        ending = work - processed
        days.append({
            "actual": demand, "scheduled_staff": staff, "capacity": capacity / 100,
            "opening_backlog": backlog / 100, "processed": processed / 100,
            "ending_backlog": ending / 100, "unused_capacity": (capacity - processed) / 100,
            "threshold_breach": ending > 100 * BACKLOG_THRESHOLD,
            "capacity_below_arrivals": capacity < 100 * demand,
        })
        backlog = ending
    return pd.DataFrame(days)


def validated_forecasts(predictions: pd.DataFrame, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Verify completeness, alignment, and actuals, then return only forecasts keyed by issue and date."""
    required = {"split", "forecast_origin", "region", "model_name", "horizon", "lead", "target_date", "actual", "prediction"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Missing Phase 2 columns: {sorted(required - set(predictions.columns))}")
    rows = predictions.loc[
        predictions["split"].eq("final") & predictions["horizon"].eq(HORIZON)
        & predictions["model_name"].isin(FORECAST_MODELS)
    ].copy()
    keys = ["forecast_origin", "region", "lead", "model_name"]
    expected = pd.MultiIndex.from_product([origins, panel.columns, range(1, HORIZON + 1), FORECAST_MODELS], names=keys)
    rows = rows.set_index(keys)
    if rows.index.has_duplicates or len(rows) != len(expected) or not expected.isin(rows.index).all():
        raise ValueError("Incomplete or duplicate final forecast keys.")
    rows = rows.loc[expected].reset_index()
    dates = rows["forecast_origin"] + pd.to_timedelta(rows["lead"], unit="D")
    if not rows["target_date"].eq(dates).all():
        raise ValueError("Target dates disagree with origin and lead.")
    target_positions = panel.index.get_indexer(dates)
    if (target_positions < 0).any():
        raise ValueError("Target dates are missing from tracked arrivals.")
    actual = panel.to_numpy()[target_positions, panel.columns.get_indexer(rows["region"])]
    if not rows["actual"].eq(actual).all():
        raise ValueError("Exported actuals disagree with tracked arrivals.")
    if not np.isfinite(rows["prediction"]).all() or rows["prediction"].lt(0).any():
        raise ValueError("Invalid forecast values.")
    return rows.pivot(index=[*EPISODE_KEY, "target_date"], columns="model_name", values="prediction").sort_index()


def simulate(panel: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (origin, region), forecast in forecasts.groupby(level=EPISODE_KEY, sort=True):
        roster = REGIONS[region]["staff"]
        budget = HORIZON * roster
        minimum, maximum = roster - STAFF_FLEX, roster + STAFF_FLEX
        forecast = forecast.droplevel(EPISODE_KEY).sort_index()
        schedules = {"fixed": allocate_staff(budget, minimum, maximum)}
        for model in FORECAST_MODELS:
            schedules[model] = allocate_staff(budget, minimum, maximum, forecast[model].to_numpy())
        # No realized arrivals are supplied to the scheduling function.
        arrivals = panel.loc[forecast.index, region].to_numpy()
        cutoff = panel.loc[:origin, region].quantile(PEAK_QUANTILE, interpolation="higher")
        for policy, schedule in schedules.items():
            days = replay_episode(arrivals, schedule)
            frames.append(days.assign(
                forecast_origin=origin, region=region, target_date=forecast.index, lead=np.arange(1, HORIZON + 1),
                policy=policy, forecast_arrivals=np.nan if policy == "fixed" else forecast[policy].to_numpy(),
                high_arrival_threshold=cutoff, high_arrival=arrivals >= cutoff,
                at_staff_max=schedule == maximum,
                above_max_capacity=100 * arrivals > maximum * CAPACITY_HUNDREDTHS_PER_STAFF,
            ))
    return pd.concat(frames, ignore_index=True)


def episode_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    metrics = daily.groupby([*EPISODE_KEY, "policy"], sort=True).agg(
        backlog_days=("ending_backlog", "sum"), ending_backlog=("ending_backlog", "last"),
        threshold_days=("threshold_breach", "sum"), peak_backlog=("ending_backlog", "max"),
        staff_days=("scheduled_staff", "sum"), processed=("processed", "sum"), unused_capacity=("unused_capacity", "sum"),
    ).reset_index()
    columns = metrics.columns.difference([*EPISODE_KEY, "policy"])
    fixed = metrics.loc[metrics.policy.eq("fixed")].set_index(EPISODE_KEY)
    paired = metrics.set_index(EPISODE_KEY)
    differences = paired[columns] - fixed[columns].reindex(paired.index)
    return pd.concat([paired, differences.add_prefix("delta_")], axis=1).reset_index()


def run_replay(raw_path: Path, output_dir: Path) -> pd.DataFrame:
    content = raw_path.read_bytes()
    raw = pd.read_csv(BytesIO(content), dtype=dict.fromkeys(KEY_COLUMNS, "string"))
    panel = daily_panel(aggregate_daily(raw))
    if set(panel.columns) != set(REGIONS):
        raise ValueError("Replay requires the three regions with prespecified roster budgets.")
    origins = benchmark_origins(panel)["final"]
    predictions = evaluate_origins(panel, origins, "final")
    forecasts = validated_forecasts(predictions, panel, origins)
    daily = simulate(panel, forecasts)
    episodes = episode_metrics(daily)
    aggregate = episodes.drop(columns=EPISODE_KEY).groupby("policy").mean()
    by_region = episodes.drop(columns="forecast_origin").groupby(["region", "policy"]).mean()
    by_origin = episodes.drop(columns="region").groupby(["forecast_origin", "policy"]).mean()
    high = daily.loc[daily.high_arrival].copy()
    high["forecast_error"] = high.forecast_arrivals - high.actual
    high["capacity_gap"] = high.capacity - high.actual
    diagnostics = high.groupby("policy").agg(
        episode_days=("actual", "size"), mean_forecast_error=("forecast_error", "mean"),
        mean_staff=("scheduled_staff", "mean"), mean_capacity_gap=("capacity_gap", "mean"),
        below_arrivals_fraction=("capacity_below_arrivals", "mean"),
        at_staff_max_fraction=("at_staff_max", "mean"), above_max_capacity_fraction=("above_max_capacity", "mean"),
        threshold_breach_fraction=("threshold_breach", "mean"), mean_ending_backlog=("ending_backlog", "mean"),
        peak_backlog=("ending_backlog", "max"),
    )
    manifest = {
        "protocol": "retrospective synthetic overlapping region-origin episodes with separate opening balances",
        "horizon_days": HORIZON, "origins": origins.strftime("%Y-%m-%d").tolist(),
        "opening_backlog_requests": OPENING_BACKLOG, "absence_percent": ABSENCE_PERCENT,
        "productivity_requests_per_available_person_day": PRODUCTIVITY,
        "breach_if_ending_backlog_above": BACKLOG_THRESHOLD,
        "high_arrival_quantile": PEAK_QUANTILE, "high_arrival_rule": "actual >= regional origin-history higher quantile",
        "policies": ["fixed", *FORECAST_MODELS],
        "allocation": "Start at minimum. Allocate one staff-day to the greatest forecast minus assigned capacity among uncapped days. Earliest date wins ties at 10 decimals.",
        "units": "Integer staff-days, continuous expected throughput in hundredths of a request",
        "summary_weighting": "Arithmetic means of region-origin episode metrics and within-episode differences versus fixed",
        "budgets": {
            region: {"fixed_daily": spec["staff"], "staff_days": spec["staff"] * HORIZON,
                     "min_daily": spec["staff"] - STAFF_FLEX, "max_daily": spec["staff"] + STAFF_FLEX}
            for region, spec in REGIONS.items()
        },
        "raw_sha256": sha256(content).hexdigest(),
        "forecast_source": "src.backtest.evaluate_origins with frozen Phase 2 candidate configurations",
        "forecast_sha256": sha256(forecasts.to_csv(float_format="%.10f").encode()).hexdigest(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("daily_outcomes", daily), ("episode_metrics", episodes), ("aggregate_summary", aggregate.reset_index()),
        ("region_summary", by_region.reset_index()), ("origin_summary", by_origin.reset_index()),
        ("high_arrival_summary", diagnostics.reset_index()),
    ]:
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10f")
    (output_dir / "replay.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = run_replay(args.input, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Replay failed: {error}\n")
    print(summary[["backlog_days", "delta_backlog_days", "staff_days"]].round(2).to_string())


if __name__ == "__main__":
    main()
