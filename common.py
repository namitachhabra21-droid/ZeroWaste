"""
Shared settings and feature preparation for the Food Waste Prediction System.

Training, the Streamlit app and the tests all import from here, so the model
always sees the same columns in the same order.
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "food_waste.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "food_waste_model.pkl")

CATEGORIES = ["Buffet", "Fast Food", "Fine Dining", "Bakery", "Catering"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

TARGET = "Food_Waste_kg"

# Only information a kitchen has BEFORE service starts.
# Food_Consumed_kg is in the dataset, but it is only measured after service,
# and waste is roughly "prepared minus consumed". Using it as an input would
# hand the model the answer, so it is deliberately left out.
NUMERIC_INPUTS = [
    "Number_of_Customers",
    "Food_Prepared_kg",
    "Holiday_Event",
    "Previous_Food_Waste_kg",
    "Temperature_C",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw rows (dataset rows or app inputs) into model-ready features."""
    X = df[NUMERIC_INPUTS].astype(float).copy()
    X["Prepared_per_Customer"] = df["Food_Prepared_kg"] / df["Number_of_Customers"]
    for cat in CATEGORIES:
        is_cat = (df["Food_Category"] == cat).astype(int)
        X[f"Food_Category_{cat}"] = is_cat
        # Each service type eats a different amount per customer, so let the
        # model learn a separate "kg per customer" for each one
        X[f"Customers_x_{cat}"] = df["Number_of_Customers"] * is_cat
    for day in DAYS:
        X[f"Day_of_Week_{day}"] = (df["Day_of_Week"] == day).astype(int)
    return X
