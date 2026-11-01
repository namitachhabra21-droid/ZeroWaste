"""Causality and data-integrity checks. Run with python -m unittest test_real_pipeline."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from real_data import FEATURES, TARGET, build_forecast_rows, chronological_split, load_clean_data
from predict_real_waste import forecast


def example():
    dates = pd.date_range("2018-01-01", periods=20)
    return pd.DataFrame([
        {"kitchen": kitchen, "date": date, TARGET: float(i + offset)}
        for kitchen, offset in [("A", 1), ("B", 100)] for i, date in enumerate(dates)
    ])


class RealPipelineTests(unittest.TestCase):
    def test_current_and_future_targets_cannot_change_current_or_past_features(self):
        data = example()
        cutoff = pd.Timestamp("2018-01-12")
        before = build_forecast_rows(data)
        changed = data.copy()
        changed.loc[changed.date >= cutoff, TARGET] += 1000
        after = build_forecast_rows(changed)
        assert_frame_equal(before.loc[before.date <= cutoff, ["date", *FEATURES]],
                           after.loc[after.date <= cutoff, ["date", *FEATURES]])

    def test_other_kitchens_do_not_change_history_features(self):
        data = example()
        before = build_forecast_rows(data)
        data.loc[data.kitchen == "B", TARGET] = 50000
        after = build_forecast_rows(data)
        assert_frame_equal(before.loc[before.kitchen == "A"], after.loc[after.kitchen == "A"])

    def test_missing_previous_day_is_not_zero_or_previous_observation(self):
        data = example()
        data.loc[(data.kitchen == "A") & (data.date == "2018-01-10"), TARGET] = np.nan
        rows = build_forecast_rows(data)
        self.assertFalse(((rows.kitchen == "A") & (rows.date == "2018-01-11")).any())
        day = rows[(rows.kitchen == "A") & (rows.date == "2018-01-12")].iloc[0]
        self.assertEqual(day.lag_1, 11)
        self.assertEqual(day.lag_2, 9)

    def test_zero_is_a_real_measurement(self):
        data = example()
        data.loc[(data.kitchen == "A") & (data.date == "2018-01-10"), TARGET] = 0
        rows = build_forecast_rows(data)
        row = rows[(rows.kitchen == "A") & (rows.date == pd.Timestamp("2018-01-11"))].iloc[0]
        self.assertEqual(row.lag_1, 0)

    def test_splits_share_no_dates_and_preserve_time_order(self):
        parts = chronological_split(build_forecast_rows(example()))
        self.assertLess(parts[0].date.max(), parts[1].date.min())
        self.assertLess(parts[1].date.max(), parts[2].date.min())

    def test_duplicate_kitchen_dates_rejected(self):
        data = example()
        with self.assertRaises(ValueError):
            build_forecast_rows(pd.concat([data, data.iloc[:1]]))

    def test_real_source_preserves_missing_measurements_and_zeros(self):
        clean = load_clean_data()
        self.assertEqual(len(clean), 3321)
        self.assertEqual(clean[TARGET].isna().sum(), 161)
        self.assertEqual(clean[TARGET].eq(0).sum(), 3)

    def test_inference_matches_training_features_and_ignores_future(self):
        data = example()
        payload = {"selected_name": "Recent 7 observations mean", "known_kitchens": ["A", "B"],
                   "training_last_date": "2018-01-10", "unit": "unconfirmed"}
        result = forecast(payload, data, "A", "2018-01-12")
        rows = build_forecast_rows(data)
        row = rows[(rows.kitchen == "A") & (rows.date == pd.Timestamp("2018-01-12"))].iloc[0]
        self.assertEqual(result["predicted_waste"], row.mean_7)
        changed = data.copy()
        changed.loc[changed.date >= "2018-01-12", TARGET] = 9999
        self.assertEqual(result, forecast(payload, changed, "A", "2018-01-12"))

    def test_inference_rejects_fit_period_unknown_kitchen_and_stale_history(self):
        payload = {"known_kitchens": ["A"], "training_last_date": "2018-01-10"}
        for kitchen, date in [("A", "2018-01-10"), ("C", "2018-01-12"), ("A", "2018-01-22")]:
            with self.subTest(kitchen=kitchen, date=date), self.assertRaises(ValueError):
                forecast(payload, example(), kitchen, date)


if __name__ == "__main__":
    unittest.main()
