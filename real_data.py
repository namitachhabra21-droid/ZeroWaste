"""Clean observed kitchen records and build inputs for next-day waste forecasting.

Data flow: original CSV -> validated observations -> historical features -> date splits.
The target is measured waste for the forecast date. Inputs use only earlier waste
measurements plus the kitchen identity and the forecast date's calendar features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "dataset/raw/urban_waste_food_tracking.csv"
CLEAN_PATH = ROOT / "dataset/processed/urban_waste_clean.csv"
MODEL_PATH = ROOT / "model/real_food_waste_model.pkl"
# Supervised learning target (y): the recorded waste amount on the forecast date.
TARGET = "waste_weight"
UNIT = "source weight units (exact CSV unit unconfirmed)"
# Three prior measurements are needed to construct all three lag features.
MIN_HISTORY = 3
# Predictor columns (X), calculated automatically rather than entered individually.
NUMERIC_FEATURES = [
    "lag_1", "lag_2", "lag_3", "mean_3", "mean_7", "std_7",
    "historical_mean", "history_count", "weekday_sin", "weekday_cos",
    "month_sin", "month_cos",
]
FEATURES = ["kitchen", *NUMERIC_FEATURES]


def load_clean_data(path=RAW_PATH):
    """Parse and validate records while preserving genuine zeros and missing values."""
    # Read as text first: the source uses semicolons, decimal commas and cp1252 text.
    raw = pd.read_csv(path, sep=";", encoding="cp1252", dtype=str)
    # Retain the fields needed for forecasting; same-day served counts are excluded
    # because they are not assumed to be available when predicting the next day.
    df = raw[["City / kitchen", "Timestamp", "Waste weight sum"]].rename(columns={
        "City / kitchen": "kitchen", "Timestamp": "date", "Waste weight sum": TARGET,
    })
    # Normalize whitespace and types; malformed dates or numbers raise an error.
    df["kitchen"] = df["kitchen"].str.strip()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="raise")
    df[TARGET] = pd.to_numeric(df[TARGET].str.replace(",", ".", regex=False), errors="raise")
    if df["kitchen"].isna().any() or df["kitchen"].eq("").any():
        raise ValueError("Missing kitchen identifier")
    # A kitchen/date pair must identify one record; duplicates are rejected, not
    # silently removed or added together, because the correct treatment is unknown.
    if df.duplicated(["kitchen", "date"]).any():
        raise ValueError("Repeated kitchen/date: investigate before aggregating")
    # Validate measured amounts without imputing unrecorded waste.
    observed = df[TARGET].dropna()
    if not np.isfinite(observed).all() or observed.lt(0).any():
        raise ValueError("Waste observations must be finite and nonnegative")
    # Preserve missing targets and outliers; missing measurement != zero waste.
    return df.sort_values(["kitchen", "date"]).reset_index(drop=True)


def features_from_history(kitchen, date, history):
    """Convert a kitchen's earlier measurements into one model input row.

    History is supplied in date order as (date, measured waste) pairs. The latest
    measurement must be from yesterday; other earlier dates may contain gaps.
    """
    date = pd.Timestamp(date).normalize()
    if len(history) < MIN_HISTORY:
        raise ValueError(f"Need at least {MIN_HISTORY} prior measured services")
    if any(d >= date for d, _ in history):
        raise ValueError("History must precede the forecast date")
    if (date - history[-1][0]).days != 1:
        raise ValueError("This forecast requires a measured waste value from the previous calendar day")
    # Lag features refer to measured observations, not necessarily consecutive days.
    # mean_7/std_7 use up to seven values when fewer than seven are available.
    # Standard deviation measures how much recent waste varies around its mean.
    values = np.array([w for _, w in history], dtype=float)
    return {
        "kitchen": kitchen,
        "lag_1": values[-1], "lag_2": values[-2], "lag_3": values[-3],
        "mean_3": values[-3:].mean(), "mean_7": values[-7:].mean(),
        "std_7": values[-7:].std(), "historical_mean": values.mean(),
        "history_count": len(values),
        # Cyclic encoding keeps Sunday close to Monday and December close to January.
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
            # Build X before adding today's actual waste to history. This prevents
            # the answer (y) from becoming an input to its own prediction.
            if len(history) >= MIN_HISTORY and (date - history[-1][0]).days == 1:
                features = features_from_history(kitchen, date, history)
                records.append({"date": date, **features, TARGET: weight})
            # After creating this example, the measured value can inform later days.
            # Missing measurements never become artificial zero-waste observations.
            if pd.notna(weight):
                history.append((date, float(weight)))
    return pd.DataFrame(records, columns=["date", *FEATURES, TARGET]).sort_values(
        ["date", "kitchen"]
    ).reset_index(drop=True)


def chronological_split(rows):
    """Use the earliest 60% of dates for training, next 20% for validation, then test.

    Percentages apply to distinct eligible dates, not row counts. Keeping every
    kitchen record for a given date in one partition preserves chronological order.
    Training learns parameters; validation selects a model; test measures performance.
    """
    dates = sorted(rows["date"].unique())
    if len(dates) < 10:
        raise ValueError("At least 10 distinct eligible dates required")
    val_start, test_start = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.8)]
    return (
        rows.loc[rows.date < val_start].copy(),
        rows.loc[(rows.date >= val_start) & (rows.date < test_start)].copy(),
        rows.loc[rows.date >= test_start].copy(),
    )
