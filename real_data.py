"""Real URBAN-WASTE data preparation; no synthetic inputs or targets."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "dataset/raw/urban_waste_food_tracking.csv"
CLEAN_PATH = ROOT / "dataset/processed/urban_waste_clean.csv"
MODEL_PATH = ROOT / "model/real_food_waste_model.pkl"
TARGET = "waste_weight"
UNIT = "source weight units (exact CSV unit unconfirmed)"
MIN_HISTORY = 3
NUMERIC_FEATURES = [
    "lag_1", "lag_2", "lag_3", "mean_3", "mean_7", "std_7",
    "historical_mean", "history_count", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]
FEATURES = ["kitchen", *NUMERIC_FEATURES]


def load_clean_data(path=RAW_PATH):
    raw = pd.read_csv(path, sep=";", encoding="cp1252", dtype=str)
    df = raw[["City / kitchen", "Timestamp", "Waste weight sum"]].rename(columns={
        "City / kitchen": "kitchen", "Timestamp": "date", "Waste weight sum": TARGET,
    })
    df["kitchen"] = df["kitchen"].str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="raise")
    df[TARGET] = pd.to_numeric(df[TARGET].str.replace(",", ".", regex=False), errors="raise")
    if df["kitchen"].isna().any() or df["kitchen"].eq("").any():
        raise ValueError("Missing kitchen identifier")
    if df.duplicated(["kitchen", "date"]).any():
        raise ValueError("Repeated kitchen/date: investigate before aggregating")
    observed = df[TARGET].dropna()
    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError("Waste observations must be finite and nonnegative")
    # Preserve missing targets and outliers; missing measurement != zero waste.
    return df.sort_values(["kitchen", "date"]).reset_index(drop=True)


def features_from_history(kitchen, date, history):
    """history consists only of strictly earlier (date, measured weight) pairs."""
    date = pd.Timestamp(date).normalize()
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} prior measured services")
    if any(d >= date for d, _ in history):
        raise ValueError("History must precede the forecast date")
    if (date - history[-1][0]).days != 1:
        raise ValueError("This forecast requires a measured waste value from the previous calendar day")
    values = np.array([w for _, w in history], dtype=float)
    return {
        "kitchen": kitchen,
        "lag_1": values[-1], "lag_2": values[-2], "lag_3": values[-3],
        "mean_3": values[-3:].mean(), "mean_7": values[-7:].mean(),
        "std_7": values[-7:].std(), "historical_mean": values.mean(),
        "history_count": len(values),
        "weekday_sin": np.sin(2 * np.pi * date.day_of_week / 7),
        "weekday_cos": np.cos(2 * np.pi * date.day_of_week / 7),
        "month_sin": np.sin(2 * np.pi * date.month / 12),
        "month_cos": np.cos(2 * np.pi * date.month / 12),
    }


def build_forecast_rows(df):
    """Build rolling next-day examples; update history AFTER each prediction row.

    Later evaluation rows can use earlier evaluation observations, as they would
    become available in daily operations. This is not a fixed-origin multi-day forecast.
    """
    if df.duplicated(["kitchen", "date"]).any():
        raise ValueError("Repeated kitchen/date")
    records = []
    for kitchen, group in df.sort_values("date").groupby("kitchen", sort=True):
        history = []
        for row in group.itertuples(index=False):
            date, weight = row.date, getattr(row, TARGET)
            if len(history) >= MIN_HISTORY and (date - history[-1][0]).days == 1:
                features = features_from_history(kitchen, date, history)
                records.append({"date": date, **features, TARGET: weight})
            if pd.notna(weight):
                history.append((date, float(weight)))
    return pd.DataFrame(records, columns=["date", *FEATURES, TARGET]).sort_values(
        ["date", "kitchen"]
    ).reset_index(drop=True)


def chronological_split(rows):
    """Split unique dates 60/20/20; no day is shared between partitions."""
    dates = sorted(rows["date"].unique())
    if len(dates) < 10:
        raise ValueError("At least 10 distinct eligible dates required")
    val_start, test_start = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    return (
        rows.loc[rows.date < val_start].copy(),
        rows.loc[(rows.date >= val_start) & (rows.date < test_start)].copy(),
        rows.loc[rows.date >= test_start].copy(),
    )
