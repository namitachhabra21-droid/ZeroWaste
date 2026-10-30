"""
Script to generate notebooks/food_waste_analysis.ipynb with complete EDA & ML pipeline (clean, text-only).
"""

import os
import json

def create_notebook():
    notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Food Waste Prediction System - Exploratory Data Analysis & Modeling\n",
    "\n",
    "**Predictive Analysis — Course Project Submission**  \n",
    "Submitted by: **Namita Chhabra** (Student ID: 2427030677)\n",
    "\n",
    "Welcome to the beginner-friendly Jupyter Notebook for the **Food Waste Prediction System**.\n",
    "\n",
    "## Notebook Objectives:\n",
    "1. Load and inspect the synthetic food waste dataset.\n",
    "2. Perform data cleaning (handling missing values, duplicates).\n",
    "3. Perform Exploratory Data Analysis (EDA) using **Matplotlib only**.\n",
    "4. Build model features from information known **before** service (Food Consumed is left out).\n",
    "5. Train and compare 3 Regressors: **Linear Regression**, **Decision Tree**, and **Random Forest**.\n",
    "6. Evaluate models on unseen test data using R2 Score and pick the best performing model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 1: Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import essential libraries\n",
    "import os\n",
    "import pickle\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
    "from sklearn.inspection import permutation_importance\n",
    "\n",
    "# Let the notebook use common.py from the project folder (works from notebooks/ or the project root)\n",
    "import sys\n",
    "sys.path.extend(['..', '.'])\n",
    "from common import build_features, TARGET\n",
    "\n",
    "# Set Matplotlib style parameters\n",
    "plt.rcParams['font.size'] = 10\n",
    "plt.rcParams['figure.autolayout'] = True"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Load Dataset & Data Sanity Checks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "dataset_path = os.path.join('..', 'dataset', 'food_waste.csv')\n",
    "if not os.path.exists(dataset_path):\n",
    "    dataset_path = os.path.join('dataset', 'food_waste.csv')\n",
    "\n",
    "df = pd.read_csv(dataset_path)\n",
    "print(\"Dataset Shape:\", df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset Info and Missing Values Check\n",
    "print(\"=== Dataset Info ===\")\n",
    "df.info()\n",
    "\n",
    "print(\"\\n=== Missing Values ===\")\n",
    "print(df.isnull().sum())\n",
    "\n",
    "print(\"\\n=== Duplicate Records ===\")\n",
    "print(\"Duplicates count:\", df.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 3: Exploratory Data Analysis (EDA) using Matplotlib Only\n",
    "\n",
    "We will now create the 6 required Matplotlib visualizations."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 1: Food Prepared vs Food Consumed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "ax.scatter(df['Food_Prepared_kg'], df['Food_Consumed_kg'], alpha=0.6, color='#2563EB', edgecolors='none')\n",
    "\n",
    "# Add diagonal 1:1 reference line (where Prepared == Consumed)\n",
    "max_val = max(df['Food_Prepared_kg'].max(), df['Food_Consumed_kg'].max())\n",
    "ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 Equal Line (Zero Waste)')\n",
    "\n",
    "ax.set_title('Food Prepared vs Food Consumed (kg)', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Food Prepared (kg)')\n",
    "ax.set_ylabel('Food Consumed (kg)')\n",
    "ax.grid(True, linestyle='--', alpha=0.5)\n",
    "ax.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 2: Food Waste Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "n, bins, patches = ax.hist(df['Food_Waste_kg'], bins=25, color='#10B981', edgecolor='black', alpha=0.75)\n",
    "\n",
    "# Highlight mean and median waste\n",
    "mean_waste = df['Food_Waste_kg'].mean()\n",
    "median_waste = df['Food_Waste_kg'].median()\n",
    "ax.axvline(mean_waste, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_waste:.2f} kg)')\n",
    "ax.axvline(median_waste, color='purple', linestyle='dashdot', linewidth=2, label=f'Median ({median_waste:.2f} kg)')\n",
    "\n",
    "ax.set_title('Food Waste Distribution (kg)', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Food Waste (kg)')\n",
    "ax.set_ylabel('Frequency')\n",
    "ax.grid(True, linestyle='--', alpha=0.5)\n",
    "ax.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 3: Food Waste by Food Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "category_avg = df.groupby('Food_Category')['Food_Waste_kg'].mean().sort_values(ascending=False)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "bars = ax.bar(category_avg.index, category_avg.values, color='#8B5CF6', edgecolor='black', alpha=0.85)\n",
    "\n",
    "# Annotate bar tops\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f} kg', ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "ax.set_title('Average Food Waste by Food Category', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Food Category')\n",
    "ax.set_ylabel('Average Waste (kg)')\n",
    "ax.grid(axis='y', linestyle='--', alpha=0.5)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 4: Customer Count vs Food Waste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "scatter = ax.scatter(df['Number_of_Customers'], df['Food_Waste_kg'], c=df['Temperature_C'], cmap='viridis', alpha=0.7)\n",
    "cbar = fig.colorbar(scatter, ax=ax)\n",
    "cbar.set_label('Temperature (C)')\n",
    "\n",
    "ax.set_title('Customer Count vs Food Waste (kg)', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Number of Customers')\n",
    "ax.set_ylabel('Food Waste (kg)')\n",
    "ax.grid(True, linestyle='--', alpha=0.5)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 4: Feature Building & Data Splitting\n",
    "\n",
    "`build_features()` (in `common.py`) one-hot encodes Food Category and Day of Week, and adds\n",
    "*Prepared per Customer* and *Customers x Category* features.\n",
    "\n",
    "**Food_Consumed_kg is deliberately NOT used.** It is only measured after service, and waste is roughly\n",
    "prepared minus consumed, so using it would leak the answer and inflate the scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate Features (X) and Target (y)\n",
    "X = build_features(df)\n",
    "y = df[TARGET]\n",
    "print('Features used:', list(X.columns))\n",
    "\n",
    "# Train-Test Split (80% train, 20% test)\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "print(\"Training Set Shape:\", X_train.shape)\n",
    "print(\"Testing Set Shape :\", X_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 5: Model Training & Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define 3 regressors\n",
    "models = {\n",
    "    'Linear Regression': LinearRegression(),\n",
    "    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),\n",
    "    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "}\n",
    "\n",
    "results = {}\n",
    "predictions = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    results[name] = r2\n",
    "    predictions[name] = y_pred\n",
    "\n",
    "print(\"=== R2 SCORE EVALUATION RESULTS (test data) ===\")\n",
    "for name, score in results.items():\n",
    "    print(f\"- {name} -> R2 Score: {score:.4f}\")\n",
    "\n",
    "best_name = max(results, key=results.get)\n",
    "best_model = models[best_name]\n",
    "print(f\"\\nBest model: {best_name}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 5: Actual vs Predicted Food Waste"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_pred = predictions[best_name]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(8, 5))\n",
    "ax.scatter(y_test, best_pred, alpha=0.7, color='#059669', label='Predicted vs Actual')\n",
    "\n",
    "# 45-degree reference line\n",
    "min_val = min(y_test.min(), best_pred.min())\n",
    "max_val = max(y_test.max(), best_pred.max())\n",
    "ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit Line (1:1)')\n",
    "\n",
    "ax.set_title(f'Actual vs Predicted Food Waste - Test Data ({best_name})', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Actual Food Waste (kg)')\n",
    "ax.set_ylabel('Predicted Food Waste (kg)')\n",
    "ax.grid(True, linestyle='--', alpha=0.5)\n",
    "ax.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Visualization 6: Feature Importance (Permutation Importance)\n",
    "\n",
    "Shuffle one input at a time and measure how many extra kg of average error (MAE) the model makes. Works for any model type."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42,\n",
    "                              scoring='neg_mean_absolute_error')\n",
    "importances = perm.importances_mean\n",
    "feature_names = X.columns\n",
    "\n",
    "# Sort feature importances\n",
    "indices = np.argsort(importances)[::-1]\n",
    "top_n = 10  # Top 10 features\n",
    "\n",
    "sorted_features = [feature_names[i] for i in indices[:top_n]][::-1]\n",
    "sorted_importances = importances[indices[:top_n]][::-1]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(9, 5))\n",
    "ax.barh(sorted_features, sorted_importances, color='#EC4899', edgecolor='black', alpha=0.85)\n",
    "ax.set_title(f'Top 10 Feature Importances ({best_name})', fontsize=12, fontweight='bold')\n",
    "ax.set_xlabel('Extra error when shuffled (kg)')\n",
    "ax.grid(axis='x', linestyle='--', alpha=0.5)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "food_waste_analysis.ipynb")
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)
        
    print(f"Jupyter Notebook generated at: {file_path}")

if __name__ == "__main__":
    create_notebook()
