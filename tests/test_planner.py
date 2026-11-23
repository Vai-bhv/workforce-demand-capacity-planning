from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.baseline import forecast_seasonal_naive
from src.capacity import create_capacity_plan
from src.prepare_data import prepare_data


ROOT = Path(__file__).resolve().parents[1]


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.history = pd.DataFrame({
            "date": pd.date_range("2025-12-25", periods=7),
            "region": "Europe",
            "incoming_requests": list(range(1, 8)),
        })

    def test_repeats_observed_week_for_each_horizon(self):
        for horizon in (7, 14, 30):
            with self.subTest(horizon=horizon):
                result = forecast_seasonal_naive(self.history, "Europe", horizon)
                self.assertEqual(result["predicted_requests"].tolist(), [(day % 7) + 1 for day in range(horizon)])
                self.assertEqual(result["forecast_origin"].unique().tolist(), [pd.Timestamp("2025-12-31")])
                self.assertEqual(result["date"].tolist(), pd.date_range("2026-01-01", periods=horizon).tolist())

    def test_explicit_origin_prevents_post_origin_dependence(self):
        future = self.history.assign(date=lambda frame: frame["date"] + pd.Timedelta(days=7), incoming_requests=999)
        extended = pd.concat([self.history, future]).sample(frac=1, random_state=3)
        expected = forecast_seasonal_naive(self.history, "Europe", 30)
        actual = forecast_seasonal_naive(extended, "Europe", 30, pd.Timestamp("2025-12-31"))
        pd.testing.assert_frame_equal(actual, expected)

    def test_requires_one_observation_on_each_of_seven_days(self):
        for history in (self.history.iloc[1:], pd.concat([self.history, self.history.iloc[[1]]])):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                forecast_seasonal_naive(history, "Europe", 7)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.forecast = pd.DataFrame({"predicted_requests": [0, 16, 16.01, 34]})

    def test_absence_and_exact_ceiling(self):
        plan = create_capacity_plan(self.forecast, 5, 0.2, 4)
        self.assertEqual(plan["expected_available_staff"].tolist(), [4] * 4)
        self.assertEqual(plan["daily_capacity"].tolist(), [16] * 4)
        self.assertEqual(plan["required_scheduled_staff"].tolist(), [0, 5, 6, 11])
        self.assertEqual(plan["additional_scheduled_staff"].tolist(), [0, 0, 1, 6])
        other = create_capacity_plan(self.forecast, 3, 0.1, 2)
        pd.testing.assert_series_equal(plan["predicted_requests"], other["predicted_requests"])
        self.assertEqual(list(self.forecast), ["predicted_requests"])

    def test_zero_capacity(self):
        for inputs in ((5, 1.0, 4), (5, 0.2, 0)):
            with self.subTest(inputs=inputs):
                plan = create_capacity_plan(self.forecast, *inputs)
                self.assertTrue(plan["daily_capacity"].eq(0).all())
                self.assertEqual(plan.loc[0, "required_scheduled_staff"], 0)
                self.assertTrue(plan["required_scheduled_staff"].iloc[1:].isna().all())
                self.assertTrue(plan["additional_scheduled_staff"].iloc[1:].isna().all())
        no_staff = create_capacity_plan(self.forecast, 0, 0.2, 4)
        self.assertEqual(no_staff["additional_scheduled_staff"].tolist(), [0, 5, 6, 11])

    def test_invalid_numeric_inputs(self):
        for inputs in (
            (-1, 0, 4), (1.5, 0, 4), (float("nan"), 0, 4), (float("inf"), 0, 4),
            (5, -0.1, 4), (5, 1.1, 4), (5, float("nan"), 4), (5, float("inf"), 4),
            (5, 0, -1), (5, 0, float("nan")), (5, 0, float("inf")),
        ):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                create_capacity_plan(self.forecast, *inputs)
        for arrival in (-1, float("nan"), float("inf"), pd.NA):
            with self.subTest(arrival=arrival), self.assertRaises(ValueError):
                create_capacity_plan(pd.DataFrame({"predicted_requests": [arrival]}), 5, 0, 4)


class AppSmokeTests(unittest.TestCase):
    def test_prepared_history_forecast_and_capacity_scenario(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            prepare_data(ROOT / "data/raw/workforce_demand.csv", output)
            with patch.dict("os.environ", {"WORKFORCE_HISTORY_PATH": str(output / "daily_regional_demand.csv")}):
                app = AppTest.from_file(str(ROOT / "app/app.py")).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertIn("Forecast origin **2025-12-31**", app.caption[1].value)
                self.assertIn("Planning horizon **14 days**", app.caption[1].value)
                self.assertIn("2026-01-01 to 2026-01-14", app.caption[1].value)
                first_plan = app.dataframe[0].value
                self.assertEqual(first_plan["Expected processing capacity"].tolist(), [33.48] * 14)
                self.assertEqual(first_plan["Forecast arrivals"].iloc[:7].tolist(),
                                 first_plan["Forecast arrivals"].iloc[7:].tolist())

                app.number_input[0].set_value(10)
                app.slider[1].set_value(25)
                app.number_input[1].set_value(6.0)
                app.run()
                self.assertFalse(app.exception)
                self.assertIn("**10 scheduled staff**", app.caption[2].value)
                self.assertIn("**25% absence**", app.caption[2].value)
                self.assertIn("**6 requests per available person-day**", app.caption[2].value)
                changed_plan = app.dataframe[0].value
                pd.testing.assert_series_equal(first_plan["Forecast arrivals"], changed_plan["Forecast arrivals"])
                self.assertEqual(changed_plan["Expected processing capacity"].tolist(), [45.0] * 14)
