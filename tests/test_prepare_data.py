import csv
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from collections import defaultdict

import pandas as pd

from src.features import build_feature_set
from src.prepare_data import aggregate_daily, prepare_data


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/workforce_demand.csv"


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(RAW)

    def test_aggregation_matches_every_raw_group(self):
        expected = defaultdict(list)
        with RAW.open(newline="") as source:
            for row in csv.DictReader(source):
                expected[row["date"], row["region"]].append(row)
        daily = aggregate_daily(self.raw)
        self.assertEqual(len(daily), 3288)
        for date, region, arrivals, completions, backlog in daily[
            ["date", "region", "incoming_requests", "completed_requests", "daily_backlog"]
        ].itertuples(index=False, name=None):
            rows = expected.pop((date.strftime("%Y-%m-%d"), region))
            self.assertEqual(len(rows), 4)
            self.assertEqual(arrivals, sum(int(row["incoming_requests"]) for row in rows))
            self.assertEqual(completions, sum(int(row["completed_requests"]) for row in rows))
            self.assertEqual(backlog, int(rows[0]["daily_backlog"]))
        self.assertFalse(expected)

    def test_rejects_invalid_schema_keys_dates_and_counts(self):
        for reason, raw in [
            ("Missing required", self.raw.drop(columns="request_type")),
            ("Duplicate", pd.concat([self.raw, self.raw.iloc[[0]]])),
            ("every region", self.raw.iloc[1:]),
            ("every region", self.raw.loc[self.raw["date"].ne("2024-02-29")]),
            ("YYYY-MM-DD", self.raw.assign(date="01/02/2023")),
            ("day is out of range", self.raw.assign(date="2023-02-30")),
            ("nonnegative integers", self.raw.assign(incoming_requests=-1)),
            ("nonnegative integers", self.raw.assign(completed_requests=float("nan"))),
            ("Repeated", self.raw.assign(available_staff=self.raw.index)),
            ("Summed arrivals", self.raw.assign(total_daily_requests=0)),
            ("between 0 and 1", self.raw.assign(staff_absence_rate=1.2)),
        ]:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                aggregate_daily(raw)

    def test_regional_lag_and_rolling_boundaries(self):
        daily = aggregate_daily(self.raw)
        features = build_feature_set(daily.sample(frac=1, random_state=3))
        for _, region in features.groupby("region"):
            series = region["incoming_requests"]
            for lag in (1, 2, 7, 14, 30):
                pd.testing.assert_series_equal(
                    region[f"incoming_requests_lag_{lag}"], series.shift(lag), check_names=False
                )
            for window, operation, column in [
                (7, "mean", "rolling_mean_7"),
                (30, "mean", "rolling_mean_30"),
                (7, "std", "rolling_std_7"),
            ]:
                expected = getattr(series.shift(1).rolling(window), operation)()
                pd.testing.assert_series_equal(region[column], expected, check_names=False)

    def test_clean_source_run_and_deterministic_output(self):
        original_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
        with TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "src").mkdir()
            for name in ("features.py", "prepare_data.py"):
                shutil.copyfile(ROOT / "src" / name, checkout / "src" / name)
            shutil.copyfile(RAW, checkout / "input.csv")
            result = subprocess.run(
                [sys.executable, "-B", "-m", "src.prepare_data", "--input", "input.csv",
                 "--output-dir", "derived"],
                cwd=checkout, capture_output=True, text=True, check=True,
            )
            self.assertIn("1,058 region-days", result.stdout)
            self.assertEqual(prepare_data(RAW, checkout / "repeat"), (3288, 3198, 1058))
            for name in ("daily_regional_demand.csv", "feature_dataset.csv"):
                self.assertEqual((checkout / "derived" / name).read_bytes(), (checkout / "repeat" / name).read_bytes())
            features = pd.read_csv(checkout / "derived/feature_dataset.csv")
            self.assertEqual(features.groupby("region").size().tolist(), [1066] * 3)
            self.assertEqual(features.groupby("region")["date"].min().tolist(), ["2023-01-31"] * 3)
            self.assertFalse(features.isna().any().any())
        self.assertEqual(hashlib.sha256(RAW.read_bytes()).hexdigest(), original_hash)
