<div align="center">

# 🍲 ZeroWaste — Food Waste Forecasting

**Forecast how much food a real kitchen will waste tomorrow, using only what it measured recently.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## 🎓 Course Submission

| | |
| :--- | :--- |
| **Course** | Predictive Analysis |
| **Project** | ZeroWaste — Food Waste Forecasting |
| **Submitted by** | Namita Chhabra |
| **Student ID** | 2427030677 |
| **Submitted** | November 2026 |

---

## 🌐 Live Demo

**[Open the dashboard](https://zerowaste-4shrcvvdzgxdd3dl5544xs.streamlit.app/)**

The app is hosted on Streamlit Community Cloud and deploys from the `main` branch.

---

## ⚡ At a Glance

| | |
| :--- | :--- |
| **Task** | Next-day forecast of recorded food waste for each kitchen |
| **Data** | [URBAN-WASTE](https://zenodo.org/records/3355300) hotel and restaurant records · 3,321 raw rows · 34 kitchens · 2018 |
| **Method used** | **Ridge Regression**: a trained linear model with L2 regularization |
| **Why this method** | Lowest **validation** MAE among trained ML candidates; baselines retained for comparison |
| **Test result** | MAE **10.81**, RMSE 17.38, R² **0.769** on 669 unseen rows |
| **vs previous-day baseline** | **18.6%** lower average error (MAE 13.27 → 10.81) |
| **Deliverable** | Streamlit dashboard with an overview, forecasts with model inputs, training results and a dataset explorer |

---

## 🎯 Problem

Kitchens prepare food before they know how many people will come. Preparing too much means waste; too little means shortages. This project asks a simple question:

> Given a kitchen's recent measured waste, how much will it waste tomorrow?

The answer is a number in source weight units. The export's exact unit is not confirmed, so the results are shown in those units rather than in kilograms.

---

## Restaurant inputs and prediction output

The demo already contains dated kitchen waste measurements. Select a known kitchen and forecast date. At least three earlier measurements are required, including one from the previous calendar day. Ridge Regression uses kitchen identity, the last three waste measurements, recent means and variation, historical mean and measurement count, and weekday/month features calculated automatically from earlier records.

The output is one predicted next-day waste value in source weight units. The seven-observation moving average is retained as a forecasting baseline. The dashboard has no new-restaurant upload form; present-day use requires updated history and evaluation.

---

## 📊 Data

- **Source:** URBAN-WASTE, collected from real hotel and restaurant kitchens.
- **Target:** recorded `Waste weight sum` per kitchen per day.
- **Cleaning:** missing waste is never filled with zero. Outliers are kept. Served portions and advance orders are excluded.
- **Eligibility:** a row is used only when the kitchen has at least three earlier measurements and one from the previous calendar day.

The cleaned data is in [`dataset/processed/`](dataset/processed/) and the original export is in [`dataset/raw/`](dataset/raw/).

---

## 🧪 Method

### 1. Split by time, not randomly

| Split | Dates | Eligible rows |
| :--- | :--- | ---: |
| Train | 7 Mar – 9 Aug 2018 | 1,249 |
| Validation | 10 Aug – 30 Sep 2018 | 729 |
| Test (final holdout) | 1 Oct – 21 Nov 2018 | 700 (669 scored) |

Future records never leak into training. Test rows from kitchens unseen during fitting are excluded (31 rows), and the report says so.

### 2. Compare candidates on validation data

| Candidate | Validation MAE | Validation R² |
| :--- | ---: | ---: |
| Recent 7 observations mean (baseline) | **12.58** | 0.775 |
| **Ridge Regression (selected ML)** | **12.65** | 0.783 |
| Random Forest | 13.14 | 0.768 |
| Historical mean | 14.77 | 0.719 |
| Previous day | 15.46 | 0.642 |
| Decision Tree | 15.46 | 0.676 |

For the course requirement, selection uses lowest validation MAE among trained ML candidates, selecting Ridge. The moving-average baseline has slightly lower validation MAE (12.58 vs 12.65). The holdout was inspected in the earlier experiment and is not a fresh untouched test in this revision.

### 3. Evaluate once on the holdout

The selected ML model is evaluated on the existing holdout; selection uses validation MAE. Evaluation is rolling next-day: earlier test-day actuals become available for later test-day forecasts, as they would in daily operation.

| Test (669 rows) | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| **Ridge Regression (selected)** | **10.81** | **17.38** | **0.769** |
| Recent 7 observations mean (baseline) | 10.99 | 17.60 | 0.763 |
| Previous day | 13.27 | 21.43 | 0.649 |
| Historical mean | 15.56 | 22.59 | 0.610 |

### 4. Other models, for comparison only

The additional test comparison includes Ridge (the selected reference), Decision Tree and Random Forest. They are reported in [`docs/real_model/posthoc_test_results.json`](docs/real_model/posthoc_test_results.json) and labelled as not used for selection.

| Model (test comparison) | Test MAE | Test R² |
| :--- | ---: | ---: |
| Ridge Regression | 10.81 | 0.769 |
| Random Forest | 11.66 | 0.741 |
| Decision Tree | 12.35 | 0.696 |

Ridge is selected by validation MAE among trained ML models, rather than by test scores.

### What "Recent 7 observations mean" means

This comparison baseline averages a kitchen's last seven measured values before the forecast date. It is a simple time-series forecasting method. The main predictor is Ridge Regression.

---

## 🖥️ Dashboard

```bash
streamlit run app.py
```

| View | What it shows |
| :--- | :--- |
| **Overview** | Verdict, key numbers, method comparison, predicted vs recorded timeline, error histogram, and per-kitchen improvement |
| **Forecast waste** | Pick a kitchen and date. See the required records, Ridge forecast and model inputs |
| **Training results** | Validation and test tables, the selection step, predicted vs recorded scatter, and post-hoc models |
| **Dataset explorer** | Records by date range and kitchen, with the raw table and a CSV download |
| **How it works** | Plain-language description of the method and its limits |

---

## 🚀 Getting Started

```bash
git clone https://github.com/namitachhabra21-droid/ZeroWaste.git
cd ZeroWaste
pip install -r requirements.txt

# Retrain (optional; the saved model is already in model/)
python train_real_model.py
python posthoc_comparison.py

# Dashboard
streamlit run app.py

# Tests
python -m unittest test_real_pipeline test_real_app
```

The notebook [`notebooks/real_food_waste_training.ipynb`](notebooks/real_food_waste_training.ipynb) walks through the full pipeline step by step.

---

## 📁 Project Structure

```
ZeroWaste/
├── app.py                        # Streamlit entry point
├── real_app.py                   # Dashboard views
├── real_data.py                  # Loading, cleaning, rolling features
├── train_real_model.py           # Candidates, selection, final holdout
├── predict_real_waste.py         # Next-day forecast helper
├── posthoc_comparison.py         # Post-selection scores for comparison
├── audit_real_dataset.py         # Dataset audit report
├── build_real_notebook.py        # Generates the training notebook
├── test_real_pipeline.py         # Pipeline tests
├── test_real_app.py              # Dashboard tests
├── dataset/
│   ├── raw/                      # Original URBAN-WASTE export
│   └── processed/                # Cleaned observations
├── docs/
│   ├── real_data_workflow.md     # Workflow notes
│   ├── real_dataset_audit.md     # Dataset audit
│   └── real_model/               # Metrics, predictions, post-hoc results
├── notebooks/
│   └── real_food_waste_training.ipynb
├── model/                        # Saved forecasts (see note below)
├── requirements.txt
└── requirements-notebook.txt
```

The saved forecasts in `model/` are committed so the dashboard runs without retraining.

---

## 🔭 Limitations

| Limitation | Next step |
| :--- | :--- |
| The waste unit is not confirmed | Confirm the unit with the data source before reporting kilograms or money |
| 2018 data from one source | Validate on more recent kitchen records |
| Results cover kitchens seen during fitting | Test how the method behaves for new kitchens |
| The method uses only past waste, not orders, weather or menus | Add those inputs if they become available |
| One 80/20-style holdout | Add rolling-origin evaluation over more periods |

---

<div align="center">

**Namita Chhabra · Student ID 2427030677**
Predictive Analysis — Course Project

*Built with Python · pandas · scikit-learn · Altair · Streamlit*

</div>
