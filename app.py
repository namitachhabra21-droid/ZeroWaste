"""
Food Waste Prediction System - Multi-Page Web Application
=========================================================
Features:
- Centered header title with food prediction symbol (🍲 Food Waste Prediction System).
- 5 Interactive Sidebar Navigation Views.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
@st.cache_resource
def load_trained_model():
    model_path = os.path.join("model", "food_waste_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload

@st.cache_data
def load_dataset():
    dataset_path = os.path.join("dataset", "food_waste.csv")
    if not os.path.exists(dataset_path):
        return None
    return pd.read_csv(dataset_path)

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
def main():
    # Centered Clean Title Header with Food Symbol
    st.markdown('<h1 class="header-title-clean">🍲 Food Waste Prediction System</h1>', unsafe_allow_html=True)
    
    payload = load_trained_model()
    df = load_dataset()
    
    if payload is None:
        st.error("Model file `model/food_waste_model.pkl` not found! Please run `python train_model.py` first.")
        st.stop()
        
    model = payload["model"]
    model_name = payload["model_name"]
    feature_names = payload["feature_names"]
    r2_score_val = payload["r2_score"]
    categories = payload["categories"]
    days = payload["days"]
    
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
        st.info(f"Model: {model_name}\n\nAccuracy (R2): {r2_score_val:.4f}")

    # =========================================================
    # VIEW 1: PREDICT FOOD WASTE
    # =========================================================
    if selected_view == "🔮 Predict Food Waste":
        st.subheader("Predict Expected Food Waste & Financial Loss")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            customers = st.number_input("Number of Customers", min_value=1, max_value=2000, value=150, step=5)
        with col2:
            food_prepared = st.number_input("Food Prepared (kg)", min_value=1.0, max_value=2000.0, value=95.0, step=2.5)
        with col3:
            food_consumed = st.number_input("Food Consumed (kg)", min_value=0.0, max_value=2000.0, value=82.0, step=2.5)

        col4, col5, col6 = st.columns(3)
        with col4:
            category = st.selectbox("Food Category", options=categories)
        with col5:
            day_of_week = st.selectbox("Day of Week", options=days)
        with col6:
            holiday_event_str = st.selectbox("Holiday / Special Event?", options=["No", "Yes"])
            holiday_event = 1 if holiday_event_str == "Yes" else 0

        col7, col8 = st.columns(2)
        with col7:
            prev_waste = st.number_input("Previous Food Waste (kg)", min_value=0.0, max_value=500.0, value=12.5, step=1.0)
        with col8:
            temp_c = st.number_input("Temperature (C)", min_value=-10.0, max_value=55.0, value=28.0, step=0.5)

        predict_clicked = st.button("Predict Expected Food Waste & Calculate Financial Loss")

        if predict_clicked:
            if food_consumed > food_prepared:
                st.warning("Warning: Food Consumed exceeds Food Prepared.")
                
            input_data = {
                'Number_of_Customers': customers,
                'Food_Prepared_kg': food_prepared,
                'Food_Consumed_kg': food_consumed,
                'Holiday_Event': holiday_event,
                'Previous_Food_Waste_kg': prev_waste,
                'Temperature_C': temp_c
            }
            
            for cat in categories:
                input_data[f'Food_Category_{cat}'] = 1 if category == cat else 0
                
            for d in days:
                input_data[f'Day_of_Week_{d}'] = 1 if day_of_week == d else 0
                
            input_df = pd.DataFrame([input_data])
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0
                    
            input_df = input_df[feature_names]
            
            predicted_waste = float(model.predict(input_df)[0])
            predicted_waste = max(0.0, round(predicted_waste, 2))
            estimated_loss = round(predicted_waste * cost_per_kg, 2)
            
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
                if predicted_waste < 10.0:
                    st.success("Low Waste Expected: Food preparation is appropriate.")
                elif 10.0 <= predicted_waste <= 25.0:
                    st.warning("Medium Waste Expected: Consider slightly reducing preparation.")
                else:
                    st.error("High Waste Expected: Consider reducing preparation and reviewing consumption patterns.")
                    
                st.write(f"- Unconsumed Food: {max(0.0, food_prepared - food_consumed):.2f} kg")
                st.write(f"- Consumption Efficiency: {((food_consumed / food_prepared) * 100):.1f}%")

            # Multi-Country Loss Table
            st.markdown("#### Global Currency Financial Loss Comparison")
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
            
            bars = ax.barh(['Food Prepared', 'Food Consumed', 'Predicted Waste'], 
                           [food_prepared, food_consumed, predicted_waste], 
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
        
        if df is None:
            st.warning("Dataset not found at `dataset/food_waste.csv`!")
            st.stop()

        chart_choice = st.selectbox(
            "Select Visualization:",
            options=[
                "1. Food Prepared vs Food Consumed",
                "2. Food Waste Distribution",
                "3. Food Waste by Food Category",
                "4. Customer Count vs Food Waste",
                "5. Actual vs Predicted Food Waste",
                "6. Feature Importance (Random Forest)"
            ]
        )
        
        if chart_choice == "1. Food Prepared vs Food Consumed":
            st.markdown("#### 1. Food Prepared vs Food Consumed (kg)")
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
            st.markdown("#### 5. Actual vs Predicted Food Waste (Random Forest)")
            categorical_cols = ['Food_Category', 'Day_of_Week']
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
            X = df_encoded.drop(columns=['Food_Waste_kg'])
            y = df_encoded['Food_Waste_kg']
            y_pred = model.predict(X[feature_names])
            
            fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
            ax.set_facecolor('white')
            ax.scatter(y, y_pred, alpha=0.6, color='#059669', label='Predicted vs Actual')
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit (1:1)')
            ax.set_xlabel('Actual Food Waste (kg)', fontweight='bold')
            ax.set_ylabel('Predicted Food Waste (kg)', fontweight='bold')
            ax.set_title('Actual vs Predicted Food Waste Comparison', fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

        elif chart_choice == "6. Feature Importance (Random Forest)":
            st.markdown("#### 6. Top Feature Importances (Random Forest)")
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                sorted_features = [feature_names[i] for i in indices][::-1]
                sorted_importances = importances[indices][::-1]
                
                fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='white')
                ax.set_facecolor('white')
                ax.barh(sorted_features, sorted_importances, color='#EC4899', edgecolor='black', alpha=0.85)
                ax.set_xlabel('Relative Importance', fontweight='bold')
                ax.set_title('Top 10 Feature Importances', fontweight='bold')
                ax.grid(axis='x', linestyle='--', alpha=0.5)
                st.pyplot(fig)
            else:
                st.info("Feature importance is available for Random Forest Regressor.")

    # =========================================================
    # VIEW 3: DATASET EXPLORER
    # =========================================================
    elif selected_view == "📋 Dataset Explorer":
        st.subheader("Dataset Explorer")
        st.caption("Note: This dataset is synthetic (artificially generated for educational and demonstration purposes).")
        
        if df is None:
            st.warning("Dataset file `dataset/food_waste.csv` not found!")
            st.stop()

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
        
        st.markdown("""
        All three models were trained on 80% of the synthetic food waste data (800 samples) 
        and evaluated on 20% unseen testing data (200 samples) using the **R² Score** (Coefficient of Determination):
        """)
        
        comparison_data = [
            {"Model": "Linear Regression", "Test R2 Score": 0.9964, "Status": "Evaluated"},
            {"Model": "Decision Tree Regressor", "Test R2 Score": 0.9044, "Status": "Evaluated"},
            {"Model": "Random Forest Regressor", "Test R2 Score": 0.9551, "Status": "Saved & Deployed Model"}
        ]
        comp_df = pd.DataFrame(comparison_data)
        st.table(comp_df)
        
        st.markdown("#### R² Score Comparison Bar Chart")
        fig, ax = plt.subplots(figsize=(8, 3.2), facecolor='white')
        ax.set_facecolor('white')
        bars = ax.bar(comp_df['Model'], comp_df['Test R2 Score'], color=['#3B82F6', '#8B5CF6', '#16A34A'], edgecolor='black')
        ax.set_ylim(0.8, 1.02)
        ax.set_ylabel('Test R2 Score', fontweight='bold')
        ax.set_title('Regressors Test R2 Score Comparison', fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.008, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
        st.info("**Why Random Forest is saved**: Linear Regression suffers from extreme multicollinearity when inputs like Customer Count and Food Prepared are correlated. Random Forest handles non-linear decision trees robustly without generating wild negative predictions.")

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
        6. **Run Streamlit Web App**:
           ```bash
           streamlit run app.py
           ```
        """)
        
        st.markdown("---")
        st.markdown("### Technical Interview Q&A Guide")
        
        st.markdown("""
        **1. What problem does this system solve?**
        > *This system predicts food waste in kilograms for commercial kitchens before daily operations begin, helping kitchen managers optimize food preparation and reduce monetary losses.*

        **2. Why were these specific features selected?**
        > *Features represent operational scale (customers), physical quantity (prepared vs consumed), dining style (buffet vs fine dining), weather impact (temperature), and historical trends.*

        **3. Why is Random Forest preferred for real-world predictions?**
        > *Linear Regression suffers from extreme multicollinearity when inputs like Customer Count and Food Prepared are correlated, resulting in negative predictions. Random Forest averages multiple decision trees (bagging), capturing complex non-linear patterns reliably across all input ranges.*

        **4. What does Exploratory Data Analysis (EDA) mean?**
        > *EDA is the process of examining and visualizing data before modeling to understand distributions, detect anomalies, check for missing values or duplicates, and discover key relationships.*

        **5. What does R² Score mean?**
        > *The R² Score measures how well the regression model explains target variance. An R² of 0.9551 means 95.51% of food waste variance is explained by our input features.*

        **6. How is Streamlit used in this project?**
        > *Streamlit converts Python scripts into an interactive user interface, rendering input controls, model predictions, tables, and Matplotlib charts directly in the web browser.*

        **7. How is Pickle used in this project?**
        > *Pickle serializes the trained model into `model/food_waste_model.pkl`. Streamlit loads this file into memory instantly to make real-time predictions without retraining.*
        """)

if __name__ == "__main__":
    main()
