"""
Food Waste Prediction System - Multi-Page Web Application
=========================================================
Features:
- Centered header title with food prediction symbol (🍲 Food Waste Prediction System).
- 5 Interactive Sidebar Navigation Views.
- Builds the dataset and model automatically on first run if they are missing.
"""

import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from common import DATASET_PATH, MODEL_PATH, build_features
from generate_dataset import save_dataset
from train_model import train

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Food Waste Prediction System",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Sidebar & High Contrast Content Area
st.markdown("""
<style>
    /* Main Content Area */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F4F7F5 !important;
        color: #0F172A !important;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.5rem !important;
        max-width: 95% !important;
    }

    /* Input Labels: Dark, Bold, High Contrast */
    [data-testid="stAppViewContainer"] label,
    [data-testid="stAppViewContainer"] .stNumberInput label,
    [data-testid="stAppViewContainer"] .stSelectbox label,
    [data-testid="stAppViewContainer"] .stRadio label,
    .stWidgetLabel p {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 0.8rem !important;
    }

    .stNumberInput, .stSelectbox {
        margin-bottom: 0.4rem !important;
    }

    /* Dark Mode Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
        border-right: 1px solid #334155 !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stWidgetLabel p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] .stAlert {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border: 1px solid #334155 !important;
    }

    /* Centered Header Title with Food Symbol - No Background Box */
    .header-title-clean {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        text-align: center !important;
        margin-top: 0;
        margin-bottom: 1.2rem;
        padding: 0;
        background: transparent !important;
    }

    /* Result Cards */
    .metric-waste-card {
        background-color: #FFFFFF;
        border: 2px solid #0284C7;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }

    .metric-loss-card {
        background-color: #FFFFFF;
        border: 2px solid #DC2626;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
        color: #FFFFFF !important;
        font-size: 1.05rem;
        font-weight: 700;
        padding: 0.55rem 1rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 3px 6px rgba(22, 163, 74, 0.2);
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #15803D 0%, #166534 100%);
    }

    /* Alerts Styling */
    [data-testid="stNotification"], div.stAlert {
        border-radius: 8px !important;
        padding: 0.6rem 0.9rem !important;
        font-weight: 600 !important;
    }

    div.stAlert[data-baseweb="notification"] > div:first-child,
    .stAlert p, .stAlert span {
        color: #78350F !important;
        font-weight: 700 !important;
    }

    /* Table Styling */
    table, [data-testid="stTable"] table {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
        border-collapse: collapse !important;
        width: 100% !important;
        border: 1px solid #CBD5E1 !important;
        margin-top: 0.4rem !important;
    }

    th {
        background-color: #E2E8F0 !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 0.8rem !important;
        border: 1px solid #CBD5E1 !important;
    }

    td {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.45rem 0.8rem !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
    }

    tr:nth-child(even) td {
        background-color: #F8FAFC !important;
    }

    h3, h4 {
        color: #0F172A !important;
        font-weight: 700 !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cache Data & Model Functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Preparing dataset and training the model (first run only)...")
def load_trained_model():
    payload = None
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
    # Missing, or saved by an older version of train_model.py: rebuild it
    if payload is None or "results" not in payload:
        if not os.path.exists(DATASET_PATH):
            save_dataset()
        payload = train(verbose=False)
    return payload


@st.cache_data
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return save_dataset()
    return pd.read_csv(DATASET_PATH)


def clamp(value, lo, hi):
    return min(max(value, lo), hi)


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
def main():
    # Centered Clean Title Header with Food Symbol
    st.markdown('<h1 class="header-title-clean">🍲 Food Waste Prediction System</h1>', unsafe_allow_html=True)

    payload = load_trained_model()
    df = load_dataset()

    model = payload["model"]
    model_name = payload["model_name"]
    feature_names = payload["feature_names"]
    r2_score_val = payload["r2_score"]
    results = payload["results"]
    categories = payload["categories"]
    days = payload["days"]
    ranges = payload["input_ranges"]
    per_customer_ranges = payload["per_customer_ranges"]
    train_size, test_size = payload["train_size"], payload["test_size"]

    currency_config = {
        "India (Rupees - INR)": {"symbol": "INR ", "default_cost_per_kg": 140.0},
        "United States (USD - $)": {"symbol": "$", "default_cost_per_kg": 3.50},
        "Europe (Euro - EUR)": {"symbol": "EUR ", "default_cost_per_kg": 3.20},
        "United Kingdom (GBP)": {"symbol": "GBP ", "default_cost_per_kg": 2.80},
        "United Arab Emirates (AED)": {"symbol": "AED ", "default_cost_per_kg": 13.00},
        "Canada (CAD - $)": {"symbol": "CAD $", "default_cost_per_kg": 4.80},
        "Australia (AUD - $)": {"symbol": "AUD $", "default_cost_per_kg": 5.20}
    }

    # ---------------------------------------------------------
    # DARK MODE SIDEBAR: Navigation, Currency Settings & Model Summary
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("🧭 Navigation")
        selected_view = st.radio(
            "Select View / Section:",
            options=[
                "🔮 Predict Food Waste",
                "📊 EDA Visualizations",
                "📋 Dataset Explorer",
                "🎯 Model Comparison (R²)",
                "📓 Jupyter Notebook & VS Code Guide"
            ]
        )
        st.markdown("---")

        st.header("Currency Settings")
        selected_country = st.selectbox("Select Country / Currency", options=list(currency_config.keys()))
        curr_info = currency_config[selected_country]

        cost_per_kg = st.number_input(
            f"Cost per kg ({curr_info['symbol']})",
            min_value=0.1,
            max_value=10000.0,
            value=curr_info["default_cost_per_kg"],
            step=1.0 if curr_info["symbol"] == "INR " else 0.5
        )

        st.markdown("---")
        st.header("Model Summary")
        st.info(f"Model: {model_name}\n\nTest R² score: {r2_score_val:.4f}")

        st.markdown("---")
        st.caption("Predictive Analysis course submission\n\n"
                   "**Namita Chhabra** · Student ID 2427030677")

    # =========================================================
    # VIEW 1: PREDICT FOOD WASTE
    # =========================================================
    if selected_view == "🔮 Predict Food Waste":
        st.subheader("Predict Expected Food Waste & Financial Loss")
        st.caption("Enter what you know before service starts. Input limits match the range the model was trained on.")

        cust_lo, cust_hi = int(ranges["Number_of_Customers"][0]), int(ranges["Number_of_Customers"][1])
        prep_lo, prep_hi = math.floor(ranges["Food_Prepared_kg"][0]), math.ceil(ranges["Food_Prepared_kg"][1])
        prev_hi = math.ceil(ranges["Previous_Food_Waste_kg"][1])
        temp_lo, temp_hi = math.floor(ranges["Temperature_C"][0]), math.ceil(ranges["Temperature_C"][1])

        col1, col2, col3 = st.columns(3)
        with col1:
            customers = st.number_input("Number of Customers", min_value=cust_lo, max_value=cust_hi,
                                        value=clamp(150, cust_lo, cust_hi), step=5)
        with col2:
            food_prepared = st.number_input("Food Prepared (kg)", min_value=float(prep_lo), max_value=float(prep_hi),
                                            value=float(clamp(100, prep_lo, prep_hi)), step=2.5)
        with col3:
            category = st.selectbox("Food Category", options=categories)

        col4, col5, col6 = st.columns(3)
        with col4:
            day_of_week = st.selectbox("Day of Week", options=days)
        with col5:
            holiday_event_str = st.selectbox("Holiday / Special Event?", options=["No", "Yes"])
            holiday_event = 1 if holiday_event_str == "Yes" else 0
        with col6:
            prev_waste = st.number_input("Previous Food Waste (kg)", min_value=0.0, max_value=float(prev_hi),
                                         value=float(clamp(15, 0, prev_hi)), step=1.0)

        col7, _ = st.columns([1, 2])
        with col7:
            temp_c = st.number_input("Temperature (C)", min_value=float(temp_lo), max_value=float(temp_hi),
                                     value=float(clamp(28, temp_lo, temp_hi)), step=0.5)

        predict_clicked = st.button("Predict Expected Food Waste & Calculate Financial Loss")

        if predict_clicked:
            per_customer = food_prepared / customers
            lo, hi = per_customer_ranges[category]
            if per_customer < lo:
                st.warning(f"{per_customer:.2f} kg per customer is less than any {category} kitchen in the training data "
                           f"({lo:.2f}–{hi:.2f} kg). You may run short of food, and this prediction is less reliable.")
            elif per_customer > hi:
                st.warning(f"{per_customer:.2f} kg per customer is more than any {category} kitchen in the training data "
                           f"({lo:.2f}–{hi:.2f} kg). This prediction is less reliable.")

            input_df = pd.DataFrame([{
                'Number_of_Customers': customers,
                'Food_Prepared_kg': food_prepared,
                'Food_Category': category,
                'Day_of_Week': day_of_week,
                'Holiday_Event': holiday_event,
                'Previous_Food_Waste_kg': prev_waste,
                'Temperature_C': temp_c,
            }])
            input_X = build_features(input_df)[feature_names]

            predicted_waste = float(model.predict(input_X)[0])
            predicted_waste = max(0.0, round(predicted_waste, 2))
            estimated_loss = round(predicted_waste * cost_per_kg, 2)
            waste_pct = predicted_waste / food_prepared * 100
            expected_eaten = max(0.0, food_prepared - predicted_waste)

            st.markdown("### Prediction Results & Financial Impact")

            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.markdown(f"""
                <div class="metric-waste-card">
                    <div style="color:#4B5563; font-size:0.95rem; font-weight:700;">Predicted Food Waste</div>
                    <div style="color:#0284C7; font-size:2.3rem; font-weight:800; margin-top:0.3rem;">
                        {predicted_waste:.2f} <span style="font-size:1.1rem; font-weight:600;">kg</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="metric-loss-card">
                    <div style="color:#4B5563; font-size:0.95rem; font-weight:700;">Estimated Financial Loss</div>
                    <div style="color:#DC2626; font-size:2.1rem; font-weight:800; margin-top:0.3rem;">
                        {curr_info['symbol']} {estimated_loss:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown("#### Recommendation")
                suggested_prep = expected_eaten * 1.05  # expected demand plus a 5% safety buffer
                if waste_pct < 10:
                    st.success("Low Waste Expected: Food preparation is appropriate.")
                elif waste_pct <= 20:
                    st.warning(f"Medium Waste Expected: Consider preparing about {suggested_prep:.0f} kg instead.")
                else:
                    st.error(f"High Waste Expected: Consider preparing about {suggested_prep:.0f} kg instead "
                             "and reviewing portion sizes.")

                st.write(f"- Waste as share of food prepared: {waste_pct:.1f}%")
                st.write(f"- Food prepared per customer: {per_customer:.2f} kg")

            # Multi-Country Loss Table
            st.markdown("#### Global Currency Financial Loss Comparison")
            st.caption("Each country uses a typical local food cost per kg (editable for your own country in the sidebar). "
                       "These are not exchange-rate conversions.")
            country_loss_data = []
            for country_name, info in currency_config.items():
                if country_name == selected_country:
                    cntry_loss = estimated_loss
                else:
                    cntry_loss = round(predicted_waste * info["default_cost_per_kg"], 2)

                country_loss_data.append({
                    "Country / Region": country_name.split("(")[0].strip(),
                    "Estimated Financial Loss": f"{info['symbol']}{cntry_loss:,.2f}"
                })

            loss_df = pd.DataFrame(country_loss_data)
            st.table(loss_df)

            # Quantity Chart
            st.markdown("#### Quantity Breakdown Chart")
            fig, ax = plt.subplots(figsize=(8, 2.2), facecolor='white')
            ax.set_facecolor('white')

            bars = ax.barh(['Food Prepared', 'Expected Eaten', 'Predicted Waste'],
                           [food_prepared, expected_eaten, predicted_waste],
                           color=['#0284C7', '#16A34A', '#DC2626'])
            ax.set_xlabel('Weight (kg)', fontsize=9, fontweight='bold', color='#0F172A')
            ax.set_title('Food Weight Comparison (kg)', fontsize=10, fontweight='bold', color='#0F172A')
            ax.tick_params(colors='#0F172A', labelsize=9)
            for bar in bars:
                width = bar.get_width()
                ax.text(width + (max(food_prepared, 10)*0.01), bar.get_y() + bar.get_height()/2,
                        f'{width:.1f} kg', ha='left', va='center', fontweight='bold', fontsize=8.5, color='#0F172A')
            plt.tight_layout()
            st.pyplot(fig)

    # =========================================================
    # VIEW 2: EDA VISUALIZATIONS
    # =========================================================
    elif selected_view == "📊 EDA Visualizations":
        st.subheader("Exploratory Data Analysis (EDA) - Matplotlib Graphs")

        chart_choice = st.selectbox(
            "Select Visualization:",
            options=[
                "1. Food Prepared vs Food Consumed",
                "2. Food Waste Distribution",
                "3. Food Waste by Food Category",
                "4. Customer Count vs Food Waste",
                "5. Actual vs Predicted Food Waste",
                "6. Feature Importance"
            ]
        )

        if chart_choice == "1. Food Prepared vs Food Consumed":
            st.markdown("#### 1. Food Prepared vs Food Consumed (kg)")
            st.caption("Food consumed is only known after service, so it is shown here for analysis but is not a model input.")
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            ax.scatter(df['Food_Prepared_kg'], df['Food_Consumed_kg'], alpha=0.6, color='#2563EB', edgecolors='none')
            max_val = max(df['Food_Prepared_kg'].max(), df['Food_Consumed_kg'].max())
            ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 Zero-Waste Line')
            ax.set_xlabel('Food Prepared (kg)', fontweight='bold')
            ax.set_ylabel('Food Consumed (kg)', fontweight='bold')
            ax.set_title('Food Prepared vs Food Consumed', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        elif chart_choice == "2. Food Waste Distribution":
            st.markdown("#### 2. Food Waste Distribution (kg)")
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            ax.hist(df['Food_Waste_kg'], bins=25, color='#10B981', edgecolor='black', alpha=0.75)
            mean_waste = df['Food_Waste_kg'].mean()
            median_waste = df['Food_Waste_kg'].median()
            ax.axvline(mean_waste, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_waste:.2f} kg)')
            ax.axvline(median_waste, color='purple', linestyle='dashdot', linewidth=2, label=f'Median ({median_waste:.2f} kg)')
            ax.set_xlabel('Food Waste (kg)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Food Waste Frequency Distribution', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        elif chart_choice == "3. Food Waste by Food Category":
            st.markdown("#### 3. Average Food Waste by Food Category")
            category_avg = df.groupby('Food_Category')['Food_Waste_kg'].mean().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            bars = ax.bar(category_avg.index, category_avg.values, color='#8B5CF6', edgecolor='black', alpha=0.85)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f'{yval:.2f} kg', ha='center', va='bottom', fontweight='bold')
            ax.set_xlabel('Food Category', fontweight='bold')
            ax.set_ylabel('Average Waste (kg)', fontweight='bold')
            ax.set_title('Average Food Waste by Service Category', fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)

        elif chart_choice == "4. Customer Count vs Food Waste":
            st.markdown("#### 4. Customer Count vs Food Waste (Color = Temp C)")
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            scatter = ax.scatter(df['Number_of_Customers'], df['Food_Waste_kg'], c=df['Temperature_C'], cmap='viridis', alpha=0.7)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Temperature (C)', fontweight='bold')
            ax.set_xlabel('Number of Customers', fontweight='bold')
            ax.set_ylabel('Food Waste (kg)', fontweight='bold')
            ax.set_title('Customer Count vs Food Waste', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        elif chart_choice == "5. Actual vs Predicted Food Waste":
            st.markdown(f"#### 5. Actual vs Predicted Food Waste ({model_name})")
            st.caption(f"Uses only the {test_size} test rows the model never saw during training.")
            y = np.array(payload["test_actual"])
            y_pred = np.array(payload["test_predicted"])

            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            ax.scatter(y, y_pred, alpha=0.6, color='#059669', label='Predicted vs Actual')
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (1:1)')
            ax.set_xlabel('Actual Food Waste (kg)', fontweight='bold')
            ax.set_ylabel('Predicted Food Waste (kg)', fontweight='bold')
            ax.set_title('Actual vs Predicted Food Waste (Test Data)', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        elif chart_choice == "6. Feature Importance":
            st.markdown(f"#### 6. Top Feature Importances ({model_name})")
            st.caption("How many extra kg of average error the model makes when each input is shuffled "
                       "(permutation importance on test data). Bigger = more important.")
            importance = pd.Series(payload["feature_importance"]).sort_values(ascending=False).head(10)[::-1]

            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            ax.barh(importance.index, importance.values, color='#EC4899', edgecolor='black', alpha=0.85)
            ax.set_xlabel('Extra error when shuffled (kg)', fontweight='bold')
            ax.set_title('Top 10 Feature Importances', fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            st.pyplot(fig)

    # =========================================================
    # VIEW 3: DATASET EXPLORER
    # =========================================================
    elif selected_view == "📋 Dataset Explorer":
        st.subheader("Dataset Explorer")
        st.caption("Note: This dataset is synthetic (artificially generated for educational and demonstration purposes).")

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("Total Records", f"{df.shape[0]}")
        col_d2.metric("Total Features", f"{df.shape[1]}")
        col_d3.metric("Missing Values", f"{df.isnull().sum().sum()}")

        st.markdown("#### Dataset Preview (First 100 Rows)")
        st.dataframe(df.head(100), use_container_width=True)

        st.markdown("#### Summary Statistics (`df.describe()`)")
        st.dataframe(df.describe().T, use_container_width=True)

        st.markdown("#### Dataset Structure & Info")
        info_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": [str(dtype) for dtype in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values
        })
        st.table(info_df)

    # =========================================================
    # VIEW 4: MODEL COMPARISON (R²)
    # =========================================================
    elif selected_view == "🎯 Model Comparison (R²)":
        st.subheader("Machine Learning Regressors Comparison")

        total = train_size + test_size
        st.markdown(f"""
        All three models were trained on {train_size / total:.0%} of the synthetic food waste data ({train_size} samples)
        and evaluated on the remaining {test_size / total:.0%} ({test_size} unseen samples) using the **R² Score**
        (Coefficient of Determination), RMSE and MAE. These numbers come straight from the last run of `train_model.py`.
        """)

        comp_df = pd.DataFrame([
            {
                "Model": name,
                "Test R² Score": round(m["R2_Score"], 4),
                "RMSE (kg)": round(m["RMSE"], 2),
                "MAE (kg)": round(m["MAE"], 2),
                "Status": "Saved & Deployed Model" if name == model_name else "Evaluated",
            }
            for name, m in results.items()
        ])
        st.table(comp_df)

        st.markdown("#### R² Score Comparison Bar Chart")
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor='white')
        ax.set_facecolor('white')
        colors = ['#16A34A' if name == model_name else '#94A3B8' for name in comp_df['Model']]
        bars = ax.bar(comp_df['Model'], comp_df['Test R² Score'], color=colors, edgecolor='black')
        ax.set_ylim(max(0.0, comp_df['Test R² Score'].min() - 0.1), 1.02)
        ax.set_ylabel('Test R² Score', fontweight='bold')
        ax.set_title('Regressors Test R² Score Comparison', fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

        st.info(f"**Why {model_name} is saved**: it had the highest R² score on the test data, so it is picked "
                "automatically by `train_model.py`. Food Consumed is not used as an input because it is only known "
                "after service; including it would make the scores look better than any real forecast could be.")

    # =========================================================
    # VIEW 5: JUPYTER NOTEBOOK & VS CODE GUIDE
    # =========================================================
    elif selected_view == "📓 Jupyter Notebook & VS Code Guide":
        st.subheader("Jupyter Notebook & VS Code Setup Guide")

        st.markdown("""
        ### VS Code Execution Steps:
        1. **Open VS Code** in the project folder.
        2. **Open Terminal** (`Ctrl + ~`).
        3. **Install Dependencies**:
           ```bash
           pip install -r requirements.txt
           ```
        4. **Generate Dataset**:
           ```bash
           python generate_dataset.py
           ```
        5. **Train Models & Export Pickle Payload**:
           ```bash
           python train_model.py
           ```
        6. **Run Sanity Checks** (optional):
           ```bash
           python test_prediction.py
           ```
        7. **Run Streamlit Web App**:
           ```bash
           streamlit run app.py
           ```
        Steps 4 and 5 are optional: the app builds the dataset and model itself on first run if they are missing.

        **Notebook:** `python build_notebook.py` writes `notebooks/food_waste_analysis.ipynb`.
        """)

        st.markdown("---")
        st.markdown("### Technical Interview Q&A Guide")

        st.markdown(f"""
        **1. What problem does this system solve?**
        > *This system predicts food waste in kilograms for commercial kitchens before daily operations begin, helping kitchen managers optimize food preparation and reduce monetary losses.*

        **2. Why were these specific features selected?**
        > *They are all known before service starts: operational scale (customers), quantity prepared, dining style (buffet vs fine dining), day and holiday, weather (temperature), and last service's waste. A "customers × category" feature lets the model learn a different portion size for each service type.*

        **3. Why isn't Food Consumed an input?**
        > *It is only measured after service, and waste is roughly prepared minus consumed. Using it would leak the answer into the inputs and give an unrealistically high score.*

        **4. Which model is used and why?**
        > *{model_name}, because it scored the highest R² on unseen test data ({r2_score_val:.4f}). `train_model.py` compares Linear Regression, Decision Tree and Random Forest and keeps the best automatically.*

        **5. What does Exploratory Data Analysis (EDA) mean?**
        > *EDA is the process of examining and visualizing data before modeling to understand distributions, detect anomalies, check for missing values or duplicates, and discover key relationships.*

        **6. What does R² Score mean?**
        > *The R² Score measures how well the regression model explains target variance. An R² of {r2_score_val:.4f} means about {r2_score_val:.0%} of the variation in food waste is explained by the input features.*

        **7. How is Streamlit used in this project?**
        > *Streamlit converts Python scripts into an interactive user interface, rendering input controls, model predictions, tables, and Matplotlib charts directly in the web browser.*

        **8. How is Pickle used in this project?**
        > *Pickle saves the trained model and its test scores into `model/food_waste_model.pkl`. Streamlit loads this file to make real-time predictions without retraining.*
        """)

if __name__ == "__main__":
    main()
