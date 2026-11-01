"""Post-hoc test scores for every candidate, for comparison only.

The selected method was already frozen by validation MAE in train_real_model.py.
This script scores the other candidates on the same final holdout so the report
can show them. Their test scores are NOT used to select anything.
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from real_data import FEATURES, TARGET, ROOT, build_forecast_rows, chronological_split, load_clean_data
from train_real_model import candidates, scores

OUTPUT = ROOT / "docs/real_model/posthoc_test_results.json"


def run():
    clean = load_clean_data()
    rows = build_forecast_rows(clean).dropna(subset=[TARGET])
    training, validation_all, test_all = chronological_split(rows)
    fit_rows = pd.concat([training, validation_all], ignore_index=True)
    test = test_all[test_all.kitchen.isin(fit_rows.kitchen.unique())].copy()
    results = {}
    for name, model in candidates().items():
        fitted = clone(model).fit(fit_rows[FEATURES], fit_rows[TARGET])
        results[name] = scores(test[TARGET], np.maximum(0, fitted.predict(test[FEATURES])))
    payload = {
        "label": "Post-hoc comparison. Not used for selection; selection used validation MAE only.",
        "test_rows": len(test),
        "test_results": results,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


if __name__ == "__main__":
    out = run()
    for name, s in out["test_results"].items():
        print(f"{name:18s} MAE {s['MAE']:.3f}  RMSE {s['RMSE']:.3f}  R2 {s['R2']:.3f}")
    print("saved", OUTPUT)
