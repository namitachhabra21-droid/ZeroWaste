"""
Synthetic Dataset Generator for Food Waste Prediction System
============================================================
Note: This dataset is synthetic (artificially generated for educational & demonstration purposes).

How a row is simulated:
1. A kitchen expects some number of customers and knows roughly how much each
   customer eats for its service type (buffet guests eat more than bakery customers).
2. It prepares that amount plus a safety margin. The margin differs per row,
   so some kitchens over-prepare a lot and some only a little.
3. Actual consumption then varies with the day of the week, holidays, hot weather
   and random chance.
4. Waste = prepared - consumed, plus spoilage and handling losses.

Food_Consumed_kg is saved for analysis, but it is only known after service,
so the model does not use it as an input (see common.py).
"""

import os

import numpy as np
import pandas as pd

from common import CATEGORIES, DATASET_PATH, DAYS

# Typical kg eaten per customer for each service type
PORTION_KG = {"Buffet": 0.60, "Fast Food": 0.45, "Fine Dining": 0.50, "Bakery": 0.35, "Catering": 0.55}

# Appetite / turnout relative to a normal Friday (start of week is quieter)
DAY_FACTOR = {"Monday": 0.94, "Tuesday": 0.95, "Wednesday": 0.97, "Thursday": 0.98,
              "Friday": 1.00, "Saturday": 1.02, "Sunday": 0.99}

# Extra spoilage / unserved display food for some service types (kg)
CATEGORY_SPOILAGE_KG = {"Buffet": 2.5, "Catering": 3.0, "Bakery": 1.0}


def generate_food_waste_data(num_samples=1000, seed=42):
    rng = np.random.default_rng(seed)

    # 1. Categorical features
    food_category = rng.choice(CATEGORIES, size=num_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    day_of_week = rng.choice(DAYS, size=num_samples)
    holiday_event = rng.choice([0, 1], size=num_samples, p=[0.8, 0.2])

    # 2. Numerical features known before service
    number_of_customers = rng.integers(40, 501, size=num_samples)
    temperature_c = np.round(rng.uniform(15.0, 38.0, size=num_samples), 1)

    portion = pd.Series(food_category).map(PORTION_KG).to_numpy()
    expected_demand = number_of_customers * portion

    # Each row gets its own over-preparation margin (5% to 40% extra)
    overprep_ratio = rng.uniform(1.05, 1.40, size=num_samples)
    food_prepared_kg = np.round(expected_demand * overprep_ratio, 2)

    # Kitchens that habitually over-prepare also had high waste last time
    previous_food_waste_kg = expected_demand * (overprep_ratio - 1) * rng.uniform(0.7, 1.3, size=num_samples)
    previous_food_waste_kg = np.round(np.maximum(0.5, previous_food_waste_kg + rng.normal(0, 2, size=num_samples)), 2)

    # 3. What actually gets eaten (only known after service)
    day_factor = pd.Series(day_of_week).map(DAY_FACTOR).to_numpy()
    holiday_factor = np.where(holiday_event == 1, 0.93, 1.0)       # events are harder to forecast
    heat_factor = 1 - np.maximum(0, temperature_c - 28.0) * 0.01   # people eat less when it's hot
    random_factor = rng.normal(1.0, 0.05, size=num_samples)

    food_consumed_kg = expected_demand * day_factor * holiday_factor * heat_factor * random_factor
    food_consumed_kg = np.round(np.minimum(food_prepared_kg, food_consumed_kg), 2)

    # 4. Target: Food_Waste_kg
    spoilage = pd.Series(food_category).map(CATEGORY_SPOILAGE_KG).fillna(0).to_numpy()
    heat_spoilage = np.maximum(0, temperature_c - 28.0) * 0.15
    noise = rng.normal(0, 0.8, size=num_samples)

    food_waste_kg = food_prepared_kg - food_consumed_kg + spoilage + heat_spoilage + noise
    food_waste_kg = np.round(np.maximum(0.5, food_waste_kg), 2)

    return pd.DataFrame({
        "Number_of_Customers": number_of_customers,
        "Food_Prepared_kg": food_prepared_kg,
        "Food_Consumed_kg": food_consumed_kg,
        "Food_Category": food_category,
        "Day_of_Week": day_of_week,
        "Holiday_Event": holiday_event,
        "Previous_Food_Waste_kg": previous_food_waste_kg,
        "Temperature_C": temperature_c,
        "Food_Waste_kg": food_waste_kg,
    })


def save_dataset(file_path=DATASET_PATH, num_samples=1000):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = generate_food_waste_data(num_samples=num_samples)
    df.to_csv(file_path, index=False)
    return df


if __name__ == "__main__":
    df = save_dataset()
    print(f"Synthetic dataset saved to: {DATASET_PATH}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
