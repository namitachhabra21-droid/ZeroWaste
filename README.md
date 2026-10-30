# Food Waste Prediction System

> A beginner-friendly Machine Learning project using Python, Pandas, Matplotlib, Scikit-learn, and Streamlit to predict food waste in kilograms, estimate financial loss across multiple country currencies (INR Rupees, USD, EUR, GBP, AED, CAD, AUD), and provide actionable reduction recommendations.

---
## Project Overview

Food waste is a major operational challenge and financial loss for commercial kitchens, restaurants, and catering services. Over-preparing food leads to unnecessary environmental impact, higher costs, and inefficient resource allocation.

The **Food Waste Prediction System** uses historical consumption data, customer counts, food category types, environmental temperature, and historical waste metrics to predict expected food waste in kilograms (kg) and calculate estimated monetary loss across multiple international currencies. Built with Python and Scikit-learn, the system deploys a trained **Random Forest Regressor** model to a multi-page **Streamlit** dashboard.

---
## Problem Statement

Commercial kitchens struggle to estimate the exact quantity of food needed for daily service due to fluctuating customer counts, event schedules, weather conditions, and service types. 

By applying Machine Learning regression algorithms, this project enables kitchen managers to:
1. Accurately predict expected food waste in kilograms before service begins.
2. Calculate estimated financial loss in Indian Rupees (INR ₹), US Dollars ($), Euros (€), British Pounds (£), UAE Dirhams (AED), etc.
3. Receive automated, clear recommendations to adjust food preparation quantities.
4. Reduce financial loss and minimize environmental impact.
---
## Features & Navigation Views

The web application features 5 interactive views in the dark-mode sidebar:
1. ** Predict Food Waste**: Interactive ML prediction, financial loss calculator & multi-currency comparison table.
2. ** EDA Visualizations**: Interactive Matplotlib charts (Prepared vs Consumed, Waste Distribution, Category Averages, Customer Count correlation, Actual vs Predicted, Feature Importance).
3. ** Dataset Explorer**: Interactive dataset preview, data types, null checks, summary statistics (`df.describe()`).
4. ** Model Comparison (R²)**: Performance comparison of Linear Regression ($R^2 = 0.9964$), Decision Tree ($R^2 = 0.9044$), and Random Forest ($R^2 = 0.9551$).
5. ** Jupyter Notebook & VS Code Guide**: Step-by-step setup guide and complete technical interview Q&A guide.

---
## Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** | Core Programming Language |
| **Pandas** | Data Loading, Cleaning, Manipulation & One-Hot Encoding |
| **NumPy** | Numerical Operations & Array Calculations |
| **Matplotlib** | Data Visualizations & Graphs (*Only Matplotlib used*) |
| **Scikit-learn** | Machine Learning Pipelines, Regression Models & R2 Evaluation |
| **Streamlit** | Interactive Web Application Dashboard |
| **Pickle** | Model Serialization & Persistence (`.pkl`) |

*Note: Seaborn, Joblib, Flask, FastAPI, and Gradio were intentionally NOT used as per constraints.*

---
## Dataset Description

> **Note:** The dataset (`dataset/food_waste.csv`) is **synthetic** (artificially generated using realistic operational statistical distributions for educational and demonstration purposes).

### Dataset Fields:
1. `Number_of_Customers`: Total count of expected customers (e.g., 40 to 500).
2. `Food_Prepared_kg`: Total weight of food prepared in kilograms (kg).
3. `Food_Consumed_kg`: Total weight of food actually consumed in kilograms (kg).
4. `Food_Category`: Type of dining service (`Buffet`, `Fast Food`, `Fine Dining`, `Bakery`, `Catering`).
5. `Day_of_Week`: Day of the week (`Monday` to `Sunday`).
6. `Holiday_Event`: Binary indicator (`1` for Holiday/Special Event, `0` for regular day).
7. `Previous_Food_Waste_kg`: Historical waste benchmark (kg).
8. `Temperature_C`: Ambient temperature in degrees Celsius (C).
9. `Food_Waste_kg` (Target): Actual food waste in kilograms (kg).

---
## Data Science Workflow

```
1. Load Dataset using Pandas
2. Check Dataset Info & Missing Values
3. Handle Missing Values & Remove Duplicates
4. Perform EDA with Matplotlib
5. Convert Categorical Variables - One-Hot Encoding
6. Split Data: Train 80% / Test 20%
7. Train Linear Regression, Decision Tree & Random Forest
8. Calculate R2 Score for Each Model
9. Select Random Forest Regressor (R2 = 0.9551)
10. Save Model using Pickle - food_waste_model.pkl
11. Deploy Interactive Web App using Streamlit
```
## Machine Learning Models & R2 Score Evaluation

Models are trained on 80% of data and evaluated on 20% unseen test data using the **R2 Score** (Coefficient of Determination):

- **Random Forest Regressor** -> R2 Score: `0.9551` *(Selected Saved Model)*
- **Linear Regression** -> R2 Score: `0.9964`
- **Decision Tree Regressor** -> R2 Score: `0.9044`

---
## How to Install and Run in VS Code

### 1. Open Terminal in VS Code
Press `Ctrl + ~` in VS Code.

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model
```bash
python generate_dataset.py
python train_model.py
```

### 4. Launch Streamlit Web App
```bash
streamlit run app.py
```

Streamlit will open in your browser at `http://localhost:8501`.

---

## Interview Explanation Guide

### 1. What problem does this system solve?
> *"This system predicts food waste in kilograms for commercial kitchens before daily operations begin. It solves over-preparation by providing waste forecasts, financial loss calculations in multiple currencies (e.g., INR Rupees, USD, EUR), and clear operational recommendations."*

### 2. Why were these specific features selected?
> *"The features represent operational scale (customers), physical quantity (prepared vs consumed), service style (buffet vs fine dining), weather impact (temperature), and historical trends."*

### 3. Why is Random Forest preferred for real-world predictions?
> *"Linear Regression assumes strict linear independent relationships and suffers from extreme multicollinearity when inputs like Customer Count and Food Prepared are correlated, resulting in extreme negative wild predictions. Random Forest averages multiple decision trees (bagging), capturing complex non-linear patterns reliably across all input ranges."*

### 4. What does the R2 Score mean?
> *"The R2 Score measures how well the regression model explains target variance. An R2 of 0.9551 means 95.51% of food waste variance is explained by our input features."*

---

### Author
Created with Python, Pandas, Matplotlib, Scikit-learn & Streamlit.
