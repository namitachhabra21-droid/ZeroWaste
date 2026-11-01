"""Predict one next day for a known kitchen using strictly earlier measured history."""

import argparse
import json
import pickle

import pandas as pd

from real_data import FEATURES, MODEL_PATH, RAW_PATH, TARGET, features_from_history, load_clean_data
from train_real_model import predict_selected


def forecast(payload, clean, kitchen, date):
    date = pd.Timestamp(date).normalize()
    if date <= pd.Timestamp(payload["training_last_date"]):
        raise ValueError("Forecast date must be later than the saved model's training period")
    if kitchen not in payload["known_kitchens"]:
        raise ValueError("Kitchen was not present during model fitting; no validated cold-start forecast")
    past = clean.loc[(clean.kitchen == kitchen) & (clean.date < date)].dropna(subset=[TARGET]).sort_values("date")
    history = list(past[["date", TARGET]].itertuples(index=False, name=None))
    features = features_from_history(kitchen, date, history)
    predicted = predict_selected(payload, pd.DataFrame([features])[FEATURES])[0]
    return {"kitchen": kitchen, "forecast_date": str(date.date()),
            "predicted_waste": round(float(predicted), 3), "unit": payload["unit"],
            "selected_model": payload["selected_name"], "history_last_date": str(history[-1][0].date()),
            "notice": "Historical-data prototype; no kg, financial-loss or preparation claim until unit confirmation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitchen", required=True)
    parser.add_argument("--date", required=True, help="Forecast date YYYY-MM-DD")
    parser.add_argument("--dataset", default=str(RAW_PATH), help="Original-format CSV including prior measurements")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        parser.error("Run train_real_model.py first")
    with MODEL_PATH.open("rb") as f:
        payload = pickle.load(f)
    try:
        result = forecast(payload, load_clean_data(args.dataset), args.kitchen, args.date)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
