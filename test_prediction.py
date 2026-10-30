"""
Sanity checks for the trained model.

Run with:  python test_prediction.py   (or: pytest test_prediction.py)
Trains the model first if model/food_waste_model.pkl does not exist yet.
"""

import os
import pickle

import pandas as pd

from common import DATASET_PATH, MODEL_PATH, build_features


def load_payload():
    if not os.path.exists(DATASET_PATH):
        from generate_dataset import save_dataset
        save_dataset()
    if not os.path.exists(MODEL_PATH):
        from train_model import train
        train(verbose=False)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(payload, **overrides):
    row = {
        "Number_of_Customers": 150,
        "Food_Prepared_kg": 100.0,
        "Food_Category": "Buffet",
        "Day_of_Week": "Monday",
        "Holiday_Event": 0,
        "Previous_Food_Waste_kg": 15.0,
        "Temperature_C": 28.0,
    }
    row.update(overrides)
    X = build_features(pd.DataFrame([row]))[payload["feature_names"]]
    return float(payload["model"].predict(X)[0])


def test_features_do_not_use_food_consumed():
    payload = load_payload()
    assert "Food_Consumed_kg" not in payload["feature_names"]


def test_selected_model_is_best_scoring():
    payload = load_payload()
    best = max(payload["results"], key=lambda n: payload["results"][n]["R2_Score"])
    assert payload["model_name"] == best


def test_default_prediction_is_reasonable():
    payload = load_payload()
    waste = predict(payload)
    assert 0 <= waste <= 100, waste


def test_more_food_prepared_means_more_waste():
    payload = load_payload()
    assert predict(payload, Food_Prepared_kg=120) > predict(payload, Food_Prepared_kg=90)


def test_no_negative_predictions_across_training_range():
    # Check every service type across its own normal "kg per customer" range
    payload = load_payload()
    lo_c, hi_c = payload["input_ranges"]["Number_of_Customers"]
    for category, (lo_r, hi_r) in payload["per_customer_ranges"].items():
        for customers in (lo_c, (lo_c + hi_c) / 2, hi_c):
            for ratio in (lo_r, (lo_r + hi_r) / 2, hi_r):
                waste = predict(payload, Food_Category=category, Number_of_Customers=customers,
                                Food_Prepared_kg=customers * ratio)
                assert waste > -5, (category, customers, ratio, waste)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} checks passed.")
