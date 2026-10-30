"""
Synthetic Dataset Generator for Food Waste Prediction System
============================================================
Note: This dataset is synthetic (artificially generated for educational & demonstration purposes).

It simulates restaurant/catering food preparation, consumption, environmental conditions,
and resulting food waste in kilograms.
"""

import os
import numpy as np
import pandas as pd

def generate_food_waste_data(num_samples=1000, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # 1. Categories and Days
    categories = ['Buffet', 'Fast Food', 'Fine Dining', 'Bakery', 'Catering']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Random selection of categorical features
    food_category = np.random.choice(categories, size=num_samples, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    day_of_week = np.random.choice(days, size=num_samples)
    holiday_event = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    
    # 2. Numerical features
    number_of_customers = np.random.randint(40, 500, size=num_samples)
    
    # Food prepared based on customer count (with variation)
    food_prepared_kg = np.round(number_of_customers * np.random.uniform(0.45, 0.75) + np.random.uniform(5, 20), 2)
    
    # Food consumed is a portion of food prepared (typically 70% to 95%)
    consumption_ratio = np.random.uniform(0.72, 0.94, size=num_samples)
    food_consumed_kg = np.round(food_prepared_kg * consumption_ratio, 2)
    
    # Ensure food consumed never exceeds food prepared
    food_consumed_kg = np.minimum(food_consumed_kg, food_prepared_kg - 0.5)
    
    # Previous food waste (historical baseline)
    previous_food_waste_kg = np.round(np.random.uniform(2.0, 35.0, size=num_samples), 2)
    
    # Temperature (in Celsius, affecting spoilage/consumption)
    temperature_c = np.round(np.random.uniform(15.0, 38.0, size=num_samples), 1)
    
    # 3. Calculate target variable: Food_Waste_kg
    # Primary factor: Prepared - Consumed
    base_waste = food_prepared_kg - food_consumed_kg
    
    # Category adjustments (Buffet & Catering tend to have more extra unserved waste)
    category_extra = np.zeros(num_samples)
    category_extra[food_category == 'Buffet'] += 2.5
    category_extra[food_category == 'Catering'] += 3.0
    category_extra[food_category == 'Bakery'] += 1.0
    
    # Event adjustment
    event_extra = holiday_event * 1.8
    
    # Temperature effect (higher heat leads to slightly faster spoilage/waste)
    temp_extra = np.maximum(0, (temperature_c - 28.0) * 0.15)
    
    # Previous waste correlation
    prev_extra = previous_food_waste_kg * 0.04
    
    # Random Gaussian noise
    noise = np.random.normal(0, 0.8, size=num_samples)
    
    # Total waste calculation
    food_waste_kg = np.round(base_waste + category_extra + event_extra + temp_extra + prev_extra + noise, 2)
    
    # Ensure target is positive and non-zero
    food_waste_kg = np.maximum(0.5, food_waste_kg)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        'Number_of_Customers': number_of_customers,
        'Food_Prepared_kg': food_prepared_kg,
        'Food_Consumed_kg': food_consumed_kg,
        'Food_Category': food_category,
        'Day_of_Week': day_of_week,
        'Holiday_Event': holiday_event,
        'Previous_Food_Waste_kg': previous_food_waste_kg,
        'Temperature_C': temperature_c,
        'Food_Waste_kg': food_waste_kg
    })
    
    return df

if __name__ == "__main__":
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "food_waste.csv")
    
    df = generate_food_waste_data(num_samples=1000)
    df.to_csv(file_path, index=False)
    
    print(f"Synthetic dataset successfully generated and saved to: {file_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
