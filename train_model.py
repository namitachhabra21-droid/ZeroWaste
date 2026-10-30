"""
Food Waste Prediction System - Model Training Script
=====================================================
Trains Linear Regression, Decision Tree and Random Forest regressors,
scores each one on held-out test data, keeps the one with the best test R²,
and saves it (plus all the scores) to model/food_waste_model.pkl.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from common import CATEGORIES, DATASET_PATH, DAYS, MODEL_PATH, TARGET, build_features

# Inputs whose training range the app uses as limits (the models are only
# reliable inside the range they were trained on)
RANGE_COLUMNS = ["Number_of_Customers", "Food_Prepared_kg", "Previous_Food_Waste_kg",
                 "Temperature_C", "Prepared_per_Customer"]


def train(dataset_path=DATASET_PATH, model_path=MODEL_PATH, verbose=True):
    log = print if verbose else (lambda *a, **k: None)

    # STEP 1: Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_dataset.py first!")
    df = pd.read_csv(dataset_path)
    log(f"Step 1: Loaded {df.shape[0]} rows x {df.shape[1]} columns from '{dataset_path}'")

    # STEP 2: Clean
    df = df.dropna().drop_duplicates()
    log(f"Step 2: {df.shape[0]} rows after removing missing values and duplicates")

    # STEP 3: Features (Food_Consumed_kg is excluded on purpose, see common.py)
    X = build_features(df)
    y = df[TARGET]
    feature_names = list(X.columns)
    log(f"Step 3: {len(feature_names)} features built")

    # STEP 4: Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    log(f"Step 4: {len(X_train)} training rows, {len(X_test)} test rows\n")

    # STEP 5: Train and score each model on the test rows
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "R2_Score": float(r2_score(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
        }

    log("=" * 56)
    log(f"{'Model':<20}{'Test R2':>10}{'RMSE (kg)':>13}{'MAE (kg)':>12}")
    for name, m in results.items():
        log(f"{name:<20}{m['R2_Score']:>10.4f}{m['RMSE']:>13.2f}{m['MAE']:>12.2f}")
    log("=" * 56)

    # STEP 6: Keep the model with the best test R²
    best_name = max(results, key=lambda n: results[n]["R2_Score"])
    best_model = models[best_name]
    log(f"\nSelected model: {best_name} (test R2 = {results[best_name]['R2_Score']:.4f})")

    # Which inputs matter most (works for any model type): how many extra kg of
    # average error the model makes when one input is randomly shuffled
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42,
                                  scoring="neg_mean_absolute_error")
    feature_importance = dict(zip(feature_names, perm.importances_mean.round(4).tolist()))

    input_ranges = {col: (float(X[col].min()), float(X[col].max())) for col in RANGE_COLUMNS}
    per_customer = X["Prepared_per_Customer"].groupby(df["Food_Category"])
    per_customer_ranges = {cat: (float(lo), float(hi))
                           for cat, lo, hi in zip(per_customer.min().index, per_customer.min(), per_customer.max())}

    # STEP 7: Save model and everything the app needs to display
    payload = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "r2_score": results[best_name]["R2_Score"],
        "results": results,
        "test_actual": y_test.tolist(),
        "test_predicted": best_model.predict(X_test).tolist(),
        "feature_importance": feature_importance,
        "input_ranges": input_ranges,
        "per_customer_ranges": per_customer_ranges,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "categories": CATEGORIES,
        "days": DAYS,
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    log(f"Model saved to '{model_path}'")
    return payload


if __name__ == "__main__":
    train()
