import pickle
import pandas as pd
import numpy as np

with open("model/food_waste_model.pkl", "rb") as f:
    payload = pickle.load(f)

model = payload["model"]
feature_names = payload["feature_names"]

print("Feature names in model:", feature_names)
print("Model coefficients:", getattr(model, "coef_", "No coef"))
print("Model intercept:", getattr(model, "intercept_", "No intercept"))

# Test sample input matching the user screenshot:
# Food Prepared = 125.0, Food Consumed = 74.5 (Unconsumed = 50.5), Customers = 150, Temp = 28, Prev Waste = 12.5
sample_input = {
    'Number_of_Customers': 150,
    'Food_Prepared_kg': 125.0,
    'Food_Consumed_kg': 74.5,
    'Holiday_Event': 0,
    'Previous_Food_Waste_kg': 12.5,
    'Temperature_C': 28.0,
    'Food_Category_Buffet': 1,
    'Day_of_Week_Monday': 1
}

df_input = pd.DataFrame([sample_input])
for col in feature_names:
    if col not in df_input.columns:
        df_input[col] = 0

df_input = df_input[feature_names]

raw_pred = model.predict(df_input)[0]
print("\nRaw Model Prediction:", raw_pred)
