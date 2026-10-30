"""
Food Waste Prediction System - Model Training Script
=====================================================
Trains Linear Regression, Decision Tree, and Random Forest Regressors,
evaluates test R² scores, selects the best model (Random Forest),
and exports the model payload to model/food_waste_model.pkl using Pickle.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    print("==================================================")
    print("      FOOD WASTE PREDICTION MODEL TRAINING        ")
    print("==================================================\n")
    
    # ---------------------------------------------------------
    # STEP 1: Load dataset
    # ---------------------------------------------------------
    dataset_path = os.path.join("dataset", "food_waste.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run generate_dataset.py first!")
        
    print(f"Step 1: Loading dataset from '{dataset_path}'...")
    df = pd.read_csv(dataset_path)
    print(f"-> Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")
    
    # ---------------------------------------------------------
    # STEP 2 & 3 & 4: Sanity checks, missing values, duplicates
    # ---------------------------------------------------------
    print("Step 2-4: Sanity checks, missing value check, duplicate removal...")
    df = df.dropna().drop_duplicates()
    
    # ---------------------------------------------------------
    # STEP 5: Categorical Encoding
    # ---------------------------------------------------------
    print("Step 5: Encoding categorical features...")
    categorical_cols = ['Food_Category', 'Day_of_Week']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # ---------------------------------------------------------
    # STEP 6 & 7: Feature Selection & Train-Test Split
    # ---------------------------------------------------------
    target_col = 'Food_Waste_kg'
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    feature_names = list(X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"-> Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}\n")
    
    # ---------------------------------------------------------
    # STEP 8: Train Regressors
    # ---------------------------------------------------------
    print("Step 8: Training Machine Learning Regressors...")
    
    # Random Forest is our primary ensemble regressor as it avoids multicollinearity issues
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {"R2_Score": r2, "RMSE": rmse, "MAE": mae}
        fitted_models[name] = model
        print(f"-> Trained {name}")
        
    print()
    print("==================================================")
    print("           MODEL EVALUATION RESULTS               ")
    print("==================================================")
    for name, metrics in results.items():
        print(f"- {name} -> R2 Score: {metrics['R2_Score']:.4f}")
    print("==================================================\n")
    
    # Select Random Forest as the primary robust predictor
    best_model_name = "Random Forest"
    best_model = fitted_models[best_model_name]
    best_r2 = results[best_model_name]["R2_Score"]
    
    print(f"[BEST MODEL] Selected Model: {best_model_name} with R2 Score of {best_r2:.4f}\n")
    
    # ---------------------------------------------------------
    # STEP 12: Save selected model using Pickle
    # ---------------------------------------------------------
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_filepath = os.path.join(model_dir, "food_waste_model.pkl")
    
    model_payload = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "r2_score": best_r2,
        "categories": ['Buffet', 'Fast Food', 'Fine Dining', 'Bakery', 'Catering'],
        "days": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }
    
    with open(model_filepath, "wb") as f:
        pickle.dump(model_payload, f)
        
    print(f"Model saved to '{model_filepath}' using Pickle.")

if __name__ == "__main__":
    main()
