"""Exercise real-data Streamlit flows with isolated saved-artifact fixtures."""

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

import real_app
from predict_real_waste import forecast
from real_data import ROOT, TARGET, load_clean_data


class RealAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / "model").mkdir()
        self.artifact = root / "model/real_notebook_model.pkl"
        self.payload = {
            "model": None, "selected_name": "Recent 7 observations mean",
            "known_kitchens": ["Copenhagen / Axel"], "training_last_date": "2018-09-30",
            "unit": "source weight units (exact CSV unit unconfirmed)",
            "metrics": json.loads((ROOT / "docs/real_model/metrics.json").read_text()),
        }
        self.save(self.payload)
        self.patch = patch.multiple(real_app, ROOT=root, MODEL_PATH=root / "model/cli.pkl")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.select_view("Forecast waste")

    def save(self, payload):
        self.artifact.write_bytes(pickle.dumps(payload))

    def select_view(self, name):
        next(r for r in self.app.radio if r.label == "Real-data section").set_value(name).run()
        self.assertFalse(self.app.exception)

    def test_default_notebook_forecast_matches_shared_predictor(self):
        self.assertFalse(self.app.exception)
        self.assertEqual(next(s.value for s in self.app.selectbox if s.label == "Training run"), "Jupyter notebook")
        self.assertFalse(any("Cost per kg" in n.label for n in self.app.number_input))
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        date = self.app.date_input[0].value
        expected = forecast(self.payload, load_clean_data(), "Copenhagen / Axel", date)
        self.assertAlmostEqual(float(self.app.metric[0].value), expected["predicted_waste"])

    def test_scores_and_dataset_views(self):
        self.select_view("Training results")
        self.assertEqual(self.app.metric[0].value, "0.7634")
        self.assertEqual(len(self.app.dataframe), 2)
        self.select_view("Dataset explorer")
        self.assertEqual([m.value for m in self.app.metric], ["3321", "34", "161"])
        self.select_view("How it works")

    def test_resaved_notebook_invalidates_model_cache(self):
        changed = copy.deepcopy(self.payload)
        changed["selected_name"] = "Previous day"
        self.save(changed)
        self.app.run()
        self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        expected = forecast(changed, load_clean_data(), "Copenhagen / Axel", self.app.date_input[0].value)
        self.assertAlmostEqual(float(self.app.metric[0].value), expected["predicted_waste"])

    def test_missing_artifact_has_setup_guidance(self):
        self.artifact.unlink()
        self.app.run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("No saved real-data forecast" in x.value for x in self.app.info))

    def test_missing_previous_day_is_explained(self):
        clean = load_clean_data()
        # Create a controlled missing measurement rather than assuming source gaps.
        clean.loc[(clean.kitchen == "Copenhagen / Axel") &
                  (clean.date == pd.Timestamp("2018-11-19")), TARGET] = float("nan")
        with patch.object(real_app, "load_observations", return_value=clean):
            self.app.date_input[0].set_value(pd.Timestamp("2018-11-20").date())
            self.app.button[0].click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("previous calendar day" in x.value for x in self.app.warning))


if __name__ == "__main__":
    unittest.main()
