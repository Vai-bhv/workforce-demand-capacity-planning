"""Run a retrospective, origin-safe rolling forecast benchmark from the tracked raw data."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
import xgboost

from .backtest_models import (
    HORIZONS, MAX_HORIZON, MODEL_NAMES, RIDGE_ALPHA, WEEKDAY_WEEKS, XGBOOST_PARAMETERS, predict_candidates,
)
from .prepare_data import KEY_COLUMNS, aggregate_daily


ORIGIN_RANGES = {
    "development": ("2025-01-06", "2025-09-01"),
    "final": ("2025-10-06", "2025-12-01"),
}
ORIGIN_CADENCE = "W-MON"
MIN_HISTORY_DAYS = 365
SELECTION_HORIZON = 14
MAE_TOLERANCE = 0.25
RELATIVE_TOLERANCE = 0.02
PEAK_QUANTILE = 0.90
TARGET_KEYS = ["forecast_origin", "region", "horizon", "lead", "target_date"]
PREDICTION_KEYS = TARGET_KEYS + ["model_name"]


def daily_panel(daily: pd.DataFrame) -> pd.DataFrame:
    panel = daily.pivot(index="date", columns="region", values="incoming_requests").sort_index().sort_index(axis=1)
    if panel.empty or not panel.index.equals(pd.date_range(panel.index[0], panel.index[-1])):
        raise ValueError("A complete daily date index is required.")
    if not np.isfinite(panel.to_numpy()).all() or (panel < 0).any().any():
        raise ValueError("Every region must have finite nonnegative arrivals on every date.")
    return panel


def benchmark_origins(panel: pd.DataFrame, smoke: bool = False) -> dict[str, pd.DatetimeIndex]:
    splits = {name: pd.date_range(*dates, freq=ORIGIN_CADENCE) for name, dates in ORIGIN_RANGES.items()}
    for origins in splits.values():
        if (origins[0] - panel.index[0]).days + 1 < MIN_HISTORY_DAYS:
            raise ValueError("Benchmark requires at least 365 observed days at the first origin.")
        if origins[-1] + pd.Timedelta(days=MAX_HORIZON) > panel.index[-1]:
            raise ValueError("Benchmark requires complete 30-day windows for all scheduled origins.")
    if splits["development"][-1] + pd.Timedelta(days=MAX_HORIZON) >= splits["final"][0]:
        raise ValueError("Development results must be available before final evaluation begins.")
    return {name: origins[:1] for name, origins in splits.items()} if smoke else splits


def forecast_at_origin(panel: pd.DataFrame, origin: pd.Timestamp) -> dict[str, np.ndarray]:
    """Keep the observation boundary outside the candidate implementations."""
    if origin not in panel.index:
        raise ValueError("Forecast origin must be an observed date.")
    return predict_candidates(panel.loc[:origin])


def evaluate_origins(panel: pd.DataFrame, origins: pd.DatetimeIndex, split: str) -> pd.DataFrame:
    frames = []
    for origin in origins:
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=MAX_HORIZON)
        if not dates.isin(panel.index).all():
            raise ValueError(f"Incomplete 30-day target window at {origin:%Y-%m-%d}.")
        predictions = forecast_at_origin(panel, origin)
        targets = panel.loc[dates]
        peak_cutoff = panel.loc[:origin].quantile(PEAK_QUANTILE, interpolation="higher")
        keys = pd.DataFrame({
            "forecast_origin": origin,
            "region": np.tile(panel.columns, MAX_HORIZON),
            "lead": np.repeat(np.arange(1, MAX_HORIZON + 1), len(panel.columns)),
            "target_date": dates.repeat(len(panel.columns)),
            "actual": targets.to_numpy().ravel(),
            "high_demand": targets.ge(peak_cutoff).to_numpy().ravel(),
        })
        for model, values in predictions.items():
            forecast = keys.assign(model_name=model, prediction=values.ravel(), split=split)
            for horizon in HORIZONS:
                frames.append(forecast.loc[forecast["lead"] <= horizon].assign(horizon=horizon))
    return pd.concat(frames, ignore_index=True)


def metric_table(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    error = predictions["prediction"] - predictions["actual"]
    metrics = predictions.assign(error=error, absolute_error=error.abs(), squared_error=error.pow(2))
    table = metrics.groupby(groups, sort=True).agg(
        observations=("actual", "size"),
        mae=("absolute_error", "mean"),
        mse=("squared_error", "mean"),
        bias=("error", "mean"),
        total_absolute_error=("absolute_error", "sum"),
        total_actual=("actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    actual = table.pop("total_actual")
    table["wape_pct"] = 100 * table.pop("total_absolute_error") / actual.where(actual > 0)
    return table.reset_index()


def select_model(development: pd.DataFrame) -> dict:
    if not development["split"].eq("development").all():
        raise ValueError("Model selection requires development predictions only.")
    summary = metric_table(development, ["model_name", "horizon"])
    scores = summary.loc[summary["horizon"].eq(SELECTION_HORIZON)].set_index("model_name")["mae"]
    best_mae = float(scores.min())
    tolerance = max(MAE_TOLERANCE, RELATIVE_TOLERANCE * best_mae)
    selected = next(name for name in MODEL_NAMES if scores[name] <= best_mae + tolerance)
    return {
        "selected_model": selected,
        "horizon": SELECTION_HORIZON,
        "development_mae": scores.to_dict(),
        "best_mae": best_mae,
        "tolerance_requests": tolerance,
        "simplicity_order": list(MODEL_NAMES),
    }


def write_reports(predictions: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions[["split", *TARGET_KEYS, "actual", "prediction", "model_name"]].to_csv(
        output_dir / "predictions.csv", index=False, float_format="%.10f"
    )
    common = ["split", "model_name", "horizon"]
    reports = {
        "metrics": metric_table(predictions, common),
        "metrics_by_region": metric_table(predictions, [*common, "region"]),
        "metrics_by_lead": metric_table(
            predictions.loc[predictions["horizon"].eq(MAX_HORIZON)], ["split", "model_name", "lead"]
        ),
        "metrics_by_origin": metric_table(predictions, [*common, "forecast_origin"]),
        "metrics_by_demand": metric_table(predictions, [*common, "high_demand"]),
    }
    for name, table in reports.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, float_format="%.10f")
    (output_dir / "benchmark.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def run_backtest(raw_path: Path, output_dir: Path, smoke: bool = False) -> dict:
    content = raw_path.read_bytes()
    raw = pd.read_csv(BytesIO(content), dtype=dict.fromkeys(KEY_COLUMNS, "string"))
    panel = daily_panel(aggregate_daily(raw))
    splits = benchmark_origins(panel, smoke)
    development = evaluate_origins(panel, splits["development"], "development")
    selection = select_model(development)
    final = evaluate_origins(panel, splits["final"], "final")
    metadata = {
        "protocol": "retrospective_synthetic",
        "smoke": smoke,
        "raw_sha256": sha256(content).hexdigest(),
        "data_start": panel.index[0].date().isoformat(),
        "data_end": panel.index[-1].date().isoformat(),
        "regions": panel.columns.tolist(),
        "origins": {name: dates.strftime("%Y-%m-%d").tolist() for name, dates in splits.items()},
        "origin_cadence": ORIGIN_CADENCE,
        "minimum_history_days": MIN_HISTORY_DAYS,
        "horizons": HORIZONS,
        "horizon_aggregation": "cumulative leads 1 through H",
        "peak_quantile": PEAK_QUANTILE,
        "weekday_average_weeks": WEEKDAY_WEEKS,
        "ridge_alpha": RIDGE_ALPHA,
        "xgboost_parameters": XGBOOST_PARAMETERS,
        "selection": selection,
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        },
    }
    write_reports(pd.concat([development, final], ignore_index=True), metadata, output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw request-type CSV.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Run the first origin of each split only.")
    args = parser.parse_args()
    try:
        metadata = run_backtest(args.input, args.output_dir, args.smoke)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Backtest failed: {error}\n")
    runs = ", ".join(f"{name}: {len(dates)} origins" for name, dates in metadata["origins"].items())
    print(f"{runs}. Development selection: {metadata['selection']['selected_model']}.")
    print(f"Wrote benchmark outputs to {args.output_dir}.")


if __name__ == "__main__":
    main()
