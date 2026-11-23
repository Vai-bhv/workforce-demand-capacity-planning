import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.backtest import (
    PREDICTION_KEYS, TARGET_KEYS, benchmark_origins, daily_panel, evaluate_origins,
    forecast_at_origin, metric_table, select_model,
)
from src.backtest_models import HORIZONS, MAX_HORIZON, MODEL_NAMES, direct_rows, direct_training_data, weekday_average
from src.prepare_data import aggregate_daily


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/workforce_demand.csv"


class OriginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.daily = aggregate_daily(pd.read_csv(RAW))
        cls.panel = daily_panel(cls.daily)

    def test_splits_are_separated_and_complete(self):
        splits = benchmark_origins(self.panel)
        development, final = splits["development"], splits["final"]
        self.assertEqual((len(development), len(final)), (35, 9))
        self.assertTrue(development.intersection(final).empty)
        self.assertLess(development[-1] + pd.Timedelta(days=30), final[0])
        for origin in development.union(final):
            self.assertEqual(origin.dayofweek, 0)
            targets = pd.date_range(origin + pd.Timedelta(days=1), periods=MAX_HORIZON)
            self.assertTrue(targets.isin(self.panel.index).all())
        with self.assertRaisesRegex(ValueError, "complete 30-day"):
            benchmark_origins(self.panel.iloc[:-1])
        with self.assertRaisesRegex(ValueError, "Incomplete 30-day"):
            evaluate_origins(self.panel, pd.DatetimeIndex([self.panel.index[-1]]), "final")
        with self.assertRaisesRegex(ValueError, "complete daily"):
            daily_panel(self.daily[self.daily["date"].ne(pd.Timestamp("2025-04-01"))])

    def test_future_arrivals_and_realized_staffing_cannot_change_any_forecast(self):
        origin = pd.Timestamp("2025-01-06")
        changed = self.daily.copy()
        changed.loc[changed["date"].gt(origin), "incoming_requests"] += 1000
        changed["available_staff"] = 900
        changed["staff_absence_rate"] = 0.99
        before = evaluate_origins(self.panel, pd.DatetimeIndex([origin]), "development")
        after = evaluate_origins(daily_panel(changed), pd.DatetimeIndex([origin]), "development")
        pd.testing.assert_frame_equal(before[PREDICTION_KEYS + ["prediction"]], after[PREDICTION_KEYS + ["prediction"]])
        np.testing.assert_array_equal(after["actual"] - before["actual"], 1000)

    def test_identical_model_keys_and_exact_cumulative_horizons(self):
        origin = pd.Timestamp("2025-10-06")
        forecasts = evaluate_origins(self.panel, pd.DatetimeIndex([origin]), "final")
        self.assertFalse(forecasts.duplicated(PREDICTION_KEYS).any())
        self.assertEqual(set(forecasts.groupby("model_name").groups), set(MODEL_NAMES))
        self.assertEqual(tuple(sorted(forecasts["horizon"].unique())), (7, 14, 30))
        self.assertTrue(forecasts["target_date"].gt(forecasts["forecast_origin"]).all())
        self.assertTrue(
            (forecasts["target_date"] - forecasts["forecast_origin"]).dt.days.eq(forecasts["lead"]).all()
        )
        expected = forecasts.loc[forecasts["model_name"].eq(MODEL_NAMES[0]), TARGET_KEYS + ["actual"]]
        for _, result in forecasts.groupby("model_name"):
            pd.testing.assert_frame_equal(result[expected.columns].reset_index(drop=True), expected.reset_index(drop=True))
            for horizon, rows in result.groupby("horizon"):
                self.assertEqual(set(rows["lead"]), set(range(1, horizon + 1)))
                self.assertEqual(len(rows), horizon * len(self.panel.columns))
                full = result.loc[result["horizon"].eq(30)].set_index(["region", "lead"])
                np.testing.assert_array_equal(rows["prediction"], full.loc[list(zip(rows["region"], rows["lead"])), "prediction"])
        naive = forecasts[forecasts["model_name"].eq("seasonal_naive") & forecasts["horizon"].eq(30)]
        dates = origin + pd.to_timedelta((naive["lead"] - 1) % 7 - 6, unit="D")
        expected_naive = [self.panel.at[date, region] for date, region in zip(dates, naive["region"])]
        np.testing.assert_array_equal(naive["prediction"], expected_naive)

    def test_same_weekday_average_uses_the_last_four_observed_weekdays(self):
        origin = pd.Timestamp("2025-01-06")
        observed = self.panel.loc[:origin]
        forecast_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=30)
        expected = [
            observed.loc[observed.index.dayofweek == date.dayofweek].tail(4).mean().tolist()
            for date in forecast_dates
        ]
        np.testing.assert_array_equal(weekday_average(observed, forecast_dates), expected)
        with self.assertRaisesRegex(ValueError, "28 observed days"):
            weekday_average(observed.tail(27), forecast_dates)

    def test_ridge_fit_receives_only_the_observed_target(self):
        origin = pd.Timestamp("2025-01-06")
        observed = self.panel.loc[:origin]
        estimator = Ridge(alpha=10.0, solver="cholesky")
        with patch("src.backtest_models.Ridge", return_value=estimator), patch.object(
            estimator, "fit", wraps=estimator.fit
        ) as fit:
            forecast_at_origin(self.panel, origin)
        features, labels = fit.call_args.args
        np.testing.assert_array_equal(labels, observed.to_numpy().ravel())
        self.assertEqual(len(features), observed.size)

    def test_direct_training_and_inference_features_match_at_same_issue(self):
        through = pd.Timestamp("2025-01-06")
        history = self.panel.loc[:through]
        training_features, training_labels = direct_training_data(history)
        historical_issue = len(history) - 1 - 35
        known_then = history.iloc[:historical_issue + 1]
        inference, targets, region_indices = direct_rows(known_then, np.array([historical_issue]))
        rows_per_issue = MAX_HORIZON * len(history.columns)
        pd.testing.assert_frame_equal(training_features.tail(rows_per_issue).reset_index(drop=True), inference)
        np.testing.assert_array_equal(training_labels[-rows_per_issue:], history.to_numpy()[targets, region_indices])
        self.assertLessEqual(targets.max(), len(history) - 1)
        self.assertTrue(targets.min() > historical_issue)
        self.assertFalse({"available_staff", "staff_absence_rate"} & set(inference.columns))

    def test_direct_training_targets_stop_at_fitting_origin(self):
        fitting_origin = pd.Timestamp("2025-01-06")
        history = self.panel.loc[:fitting_origin].tail(84)
        last_eligible = history.index.get_loc(pd.Timestamp("2024-12-02"))
        first_excluded = history.index.get_loc(pd.Timestamp("2024-12-09"))

        _, excluded_targets, _ = direct_rows(history, np.array([first_excluded]))
        self.assertLess(excluded_targets.min(), len(history))
        self.assertGreaterEqual(excluded_targets.max(), len(history))

        def check_training_rows(observed, issue_positions):
            features, targets, regions = direct_rows(observed, issue_positions)
            self.assertFalse((targets >= len(observed)).any(), "Selected a post-origin training label.")
            self.assertEqual(issue_positions.max(), last_eligible)
            self.assertNotIn(first_excluded, issue_positions)
            target_dates = observed.index[targets]
            self.assertTrue((target_dates <= fitting_origin).all())
            return features, targets, regions

        with patch("src.backtest_models.direct_rows", side_effect=check_training_rows) as build_rows:
            direct_training_data(history)
        build_rows.assert_called_once()

    def test_clean_tracked_source_preparation_and_repeatable_cli(self):
        with TemporaryDirectory() as directory:
            scratch = Path(directory)
            (scratch / "src").mkdir()
            names = ("baseline", "features", "prepare_data", "backtest", "backtest_models")
            for name in names:
                shutil.copyfile(ROOT / "src" / f"{name}.py", scratch / "src" / f"{name}.py")
            shutil.copyfile(RAW, scratch / "input.csv")
            subprocess.run(
                [sys.executable, "-B", "-m", "src.prepare_data", "--input", "input.csv", "--output-dir", "derived"],
                cwd=scratch, check=True, capture_output=True, text=True,
            )
            for output in ("first", "repeat"):
                subprocess.run(
                    [sys.executable, "-B", "-m", "src.backtest", "--input", "input.csv",
                     "--output-dir", output, "--smoke"],
                    cwd=scratch, check=True, capture_output=True, text=True,
                )
            first, repeat = scratch / "first", scratch / "repeat"
            self.assertEqual(len(list(first.iterdir())), 7)
            for report in first.iterdir():
                self.assertEqual(report.read_bytes(), (repeat / report.name).read_bytes())
            manifest = json.loads((first / "benchmark.json").read_text())
            self.assertEqual(manifest["horizons"], list(HORIZONS))
            saved = pd.read_csv(first / "predictions.csv")
            self.assertEqual(len(saved), 2 * sum(HORIZONS) * 3 * len(MODEL_NAMES))


class MetricTests(unittest.TestCase):
    def test_metrics_and_zero_denominator(self):
        rows = pd.DataFrame({
            "group": ["observed", "observed", "zero"],
            "actual": [2, 6, 0],
            "prediction": [4, 3, 2],
        })
        metrics = metric_table(rows, ["group"]).set_index("group")
        self.assertAlmostEqual(metrics.loc["observed", "mae"], 2.5)
        self.assertAlmostEqual(metrics.loc["observed", "rmse"], np.sqrt(6.5))
        self.assertAlmostEqual(metrics.loc["observed", "bias"], -0.5)
        self.assertAlmostEqual(metrics.loc["observed", "wape_pct"], 62.5)
        self.assertEqual(metrics.loc["zero", "mae"], 2.0)
        self.assertTrue(pd.isna(metrics.loc["zero", "wape_pct"]))

    def test_selection_prefers_simplicity_and_rejects_final_data(self):
        rows = pd.DataFrame({
            "split": "development", "horizon": 14, "model_name": MODEL_NAMES,
            "actual": 0, "prediction": [7.0, 5.2, 5.1, 5.0],
        })
        self.assertEqual(select_model(rows)["selected_model"], "weekday_average")
        with self.assertRaisesRegex(ValueError, "development predictions only"):
            select_model(rows.assign(split="final"))
