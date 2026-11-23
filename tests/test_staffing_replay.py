import json
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.capacity import create_capacity_plan
from src.staffing_replay import (
    EPISODE_KEY, FORECAST_MODELS, allocate_staff, episode_metrics, replay_episode, simulate, validated_forecasts,
)


ROOT = Path(__file__).resolve().parents[1]


class ScheduleTests(unittest.TestCase):
    def test_budget_bounds_integers_and_earliest_tie(self):
        # Seven high days would receive everything without a daily cap.
        forecast = np.array([100] * 7 + [0] * 7)
        expected = np.array([11] * 7 + [7] * 7)
        np.testing.assert_array_equal(allocate_staff(126, 7, 11, forecast), expected)
        np.testing.assert_array_equal(allocate_staff(126, 7, 11), np.full(14, 9))
        forecast = np.array([100] * 3 + [0] * 11)
        schedule = allocate_staff(14, 0, 10, forecast)
        np.testing.assert_array_equal(schedule, [5, 5, 4] + [0] * 11)
        self.assertTrue(np.issubdtype(schedule.dtype, np.integer))

    def test_impossible_budget_or_invalid_inputs(self):
        for budget, lower, upper in [(27, 2, 4), (70, 2, 4), (35, 2, 4), (28, -1, 4),
                                     (28, 4, 2), (28.0, 2, 4), (True, 0, 1)]:
            with self.subTest(budget=budget, lower=lower, upper=upper), self.assertRaises(ValueError):
                allocate_staff(budget, lower, upper, np.ones(14))
        for bad in ([1] * 13, [np.nan] * 14, [-1] * 14):
            with self.assertRaises(ValueError):
                allocate_staff(28, 1, 3, bad)

    def test_recurrence_clears_and_accumulates_work(self):
        arrivals = np.array([10, 2, 0, 8] + [0] * 10)
        schedule = np.array([1, 2, 3, 0] + [2] * 10)
        daily = replay_episode(arrivals, schedule)
        np.testing.assert_allclose(daily.ending_backlog.iloc[:5], [6.28, 0.84, 0, 8, 0.56])
        np.testing.assert_allclose(daily.opening_backlog.iloc[1:], daily.ending_backlog.iloc[:-1])
        np.testing.assert_allclose(daily.processed, np.minimum(daily.opening_backlog + arrivals, daily.capacity))
        self.assertTrue(daily.ending_backlog.ge(0).all())
        for staff in np.unique(schedule):
            plan = create_capacity_plan(pd.DataFrame({"predicted_requests": [10]}), int(staff), 0.07, 4)
            self.assertTrue(daily.loc[daily.scheduled_staff.eq(staff), "capacity"].eq(plan.daily_capacity.iloc[0]).all())


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2025-09-01", "2025-10-27", name="date")
        self.panel = pd.DataFrame({"Europe": 60}, index=dates)
        self.origins = pd.DatetimeIndex(["2025-10-06", "2025-10-13"])
        self.predictions = pd.DataFrame([
            dict(forecast_origin=origin, region="Europe", model_name=model, horizon=14, lead=lead,
                 target_date=origin + pd.Timedelta(days=lead), actual=60,
                 prediction=factor * lead, split="final")
            for origin in self.origins for model, factor in zip(FORECAST_MODELS, (2, 5))
            for lead in range(1, 15)
        ])
        self.forecasts = validated_forecasts(self.predictions, self.panel, self.origins)

    def test_reject_misaligned_forecasts(self):
        for bad in (
            self.predictions.iloc[:-1], pd.concat([self.predictions, self.predictions.iloc[:1]]),
            self.predictions.drop(columns="prediction"), self.predictions.assign(actual=61),
            self.predictions.assign(target_date=self.predictions.forecast_origin),
            self.predictions.assign(prediction=np.inf),
        ):
            with self.assertRaises(ValueError):
                validated_forecasts(bad, self.panel, self.origins)

    def test_overlapping_episodes_reset_and_pair_by_issue(self):
        days = simulate(self.panel, self.forecasts)
        self.assertEqual(len(days), 2 * 3 * 14)
        self.assertEqual(days.target_date.nunique(), 21)
        for _, episode in days.groupby(EPISODE_KEY):
            self.assertTrue(episode.groupby("lead").actual.nunique().eq(1).all())
            self.assertTrue(episode.loc[episode.lead.eq(1), "opening_backlog"].eq(0).all())
            totals = episode.groupby("policy").scheduled_staff.sum()
            self.assertEqual(totals.tolist(), [126] * 3)
            self.assertTrue(episode.scheduled_staff.between(7, 11).all())
            self.assertTrue(np.issubdtype(episode.scheduled_staff.dtype, np.integer))
        metrics = episode_metrics(days)
        self.assertEqual(len(metrics), 6)
        self.assertTrue(metrics.delta_staff_days.eq(0).all())
        self.assertTrue(metrics.loc[metrics.policy.eq("fixed"), "delta_backlog_days"].eq(0).all())

    def test_future_perturbation_keeps_all_schedule_inputs_and_schedules_identical(self):
        changed = self.panel.copy()
        changed.loc[changed.index > self.origins[0]] += 100
        with patch("src.staffing_replay.allocate_staff", wraps=allocate_staff) as allocator:
            original_days = simulate(self.panel, self.forecasts)
            original_calls = allocator.call_args_list.copy()
            allocator.reset_mock()
            changed_days = simulate(changed, self.forecasts)
            self.assertEqual(len(original_calls), 6)
            self.assertEqual(len(allocator.call_args_list), 6)
            for first, second in zip(original_calls, allocator.call_args_list):
                self.assertEqual(first.args[:3], second.args[:3])
                self.assertEqual(first.kwargs, {})
                self.assertEqual(second.kwargs, {})
                np.testing.assert_array_equal(first.args[3:], second.args[3:])
        pd.testing.assert_series_equal(original_days.scheduled_staff, changed_days.scheduled_staff)
        self.assertTrue(changed_days.ending_backlog.gt(original_days.ending_backlog).all())


class SourceOnlyTests(unittest.TestCase):
    def test_complete_cli_twice_from_source_and_tracked_input(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src").mkdir()
            for name in ("staffing_replay", "backtest", "backtest_models", "baseline", "generate_data", "prepare_data", "features"):
                shutil.copyfile(ROOT / "src" / f"{name}.py", root / "src" / f"{name}.py")
            shutil.copyfile(ROOT / "data/raw/workforce_demand.csv", root / "raw.csv")
            for output in ("first", "repeat"):
                subprocess.run(
                    [sys.executable, "-B", "-m", "src.staffing_replay", "--input", "raw.csv", "--output-dir", output],
                    cwd=root, capture_output=True, check=True, text=True,
                )
            first = root / "first"
            self.assertEqual(len(list(first.iterdir())), 7)
            for report in first.iterdir():
                self.assertEqual(report.read_bytes(), (root / "repeat" / report.name).read_bytes())
            manifest = json.loads((first / "replay.json").read_text())
            self.assertEqual(len(manifest["origins"]), 9)
            accounting_columns = ["opening_backlog", "actual", "capacity", "processed", "ending_backlog", "unused_capacity"]
            daily = pd.read_csv(first / "daily_outcomes.csv", converters=dict.fromkeys(accounting_columns, Decimal))
            self.assertEqual(len(daily), 9 * 3 * 3 * 14)
            for (origin, region), rows in daily.groupby(EPISODE_KEY):
                budget = manifest["budgets"][region]
                self.assertTrue(rows.groupby("policy").scheduled_staff.sum().eq(budget["staff_days"]).all())
                self.assertTrue(rows.scheduled_staff.between(budget["min_daily"], budget["max_daily"]).all())
                self.assertTrue(rows.groupby("lead").actual.nunique().eq(1).all())
                self.assertTrue(rows.loc[rows.lead.eq(1), "opening_backlog"].eq(0).all())

            high = daily.loc[daily.high_arrival]
            high_summary = pd.read_csv(first / "high_arrival_summary.csv").set_index("policy")
            pd.testing.assert_series_equal(
                high_summary["episode_days"], high.groupby("policy").size(), check_names=False,
            )
            self.assertEqual(len(high[["region", "target_date"]].drop_duplicates()), 61)

            # The CSV preserves the exact hundredths used for daily and episode accounting.
            schedule_key = [*EPISODE_KEY, "policy"]
            episodes = pd.read_csv(
                first / "episode_metrics.csv",
                converters={"backlog_days": Decimal, "delta_backlog_days": Decimal},
            ).set_index(schedule_key)
            schedules = daily.sort_values("lead").groupby(schedule_key)
            self.assertTrue(episodes.index.is_unique)
            self.assertEqual(set(episodes.index), set(schedules.groups))
            for key, days in schedules:
                with self.subTest(schedule=key):
                    for opening, arrivals, capacity, processed, ending, unused in days[accounting_columns].itertuples(
                        index=False, name=None,
                    ):
                        self.assertGreaterEqual(min(opening, arrivals, capacity, processed, ending, unused), 0)
                        work = opening + arrivals
                        self.assertEqual(processed, min(work, capacity))
                        self.assertEqual(ending, work - processed)
                        self.assertEqual(unused, capacity - processed)
                    self.assertEqual(days.opening_backlog.tolist()[1:], days.ending_backlog.tolist()[:-1])
                    backlog_days = episodes.at[key, "backlog_days"]
                    self.assertEqual(backlog_days, sum(days.ending_backlog))
                    fixed = episodes.at[(*key[:2], "fixed"), "backlog_days"]
                    self.assertEqual(episodes.at[key, "delta_backlog_days"], backlog_days - fixed)
