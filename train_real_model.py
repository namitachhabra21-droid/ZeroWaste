"""Train and compare regressors, then save the selected next-day waste predictor.

Ridge Regression is selected by validation MAE among three trained ML candidates.
Simple forecasting baselines remain benchmarks, including the seven-observation
mean, which has slightly lower validation MAE. The existing holdout was inspected
in an earlier experiment; rerunning this script does not create a fresh test set.
"""

import json
import pickle
import platform
from hashlib import sha256

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from real_data import (
    CLEAN_PATH, FEATURES, MODEL_PATH, NUMERIC_FEATURES, RAW_PATH, ROOT, TARGET, UNIT,
    build_forecast_rows, chronological_split, load_clean_data,
)

# Baselines are rules without fitted regression coefficients. They provide a
# reference for judging whether the trained models offer a useful improvement.
BASELINES = {"Previous day": "lag_1", "Recent 7 observations mean": "mean_7",
             "Historical mean": "historical_mean"}


def candidates():
    """Return three fixed scikit-learn pipelines; no hyperparameter search is run."""
    def pipeline(model):
        # One-hot encoding represents each kitchen as binary indicator columns.
        # StandardScaler centers numeric features and scales them to unit variance
        # where variance is nonzero. Scaling is particularly relevant to Ridge.
        prep = ColumnTransformer([
            ("kitchen", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["kitchen"]),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ])
        # Fitting the pipeline learns preprocessing statistics and model weights
        # together. During selection, both are fitted only on the training split.
        return make_pipeline(prep, model)
    return {
        # Ridge is linear regression with an L2 penalty on coefficient magnitude:
        # squared prediction error + alpha * sum(squared coefficients).
        # alpha=10 is a fixed penalty strength, not a value optimized in this study.
        "Ridge Regression": pipeline(Ridge(alpha=10.0)),
        # Tree depth and leaf sizes limit complexity; the forest averages 200 trees.
        # random_state fixes randomness to make repeated runs reproducible.
        "Decision Tree": pipeline(DecisionTreeRegressor(max_depth=6, min_samples_leaf=10, random_state=42)),
        "Random Forest": pipeline(RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    }


def scores(actual, predicted):
    """Measure regression performance against the recorded waste amounts.

    MAE: mean absolute error, in the target's units; lower is better.
    RMSE: root mean squared error, which penalizes large misses more strongly.
    R2: improvement relative to predicting the evaluated targets' mean; it can be
    negative and must not be reported as a percentage of correct predictions.
    """
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R2": float(r2_score(actual, predicted))}


def predict_selected(payload, X):
    """Apply a baseline rule or the saved fitted pipeline to the same feature schema."""
    if payload["selected_name"] in BASELINES:
        return X[BASELINES[payload["selected_name"]]].to_numpy(dtype=float)
    # Waste cannot be negative, so clip any negative regression estimates to zero.
    return np.maximum(0, payload["model"].predict(X[FEATURES]))


def train():
    """Prepare data, select on validation, refit, evaluate and save model artifacts."""
    clean = load_clean_data()
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False, date_format="%Y-%m-%d")
    # Rows without a measured target cannot teach or score supervised regression.
    rows = build_forecast_rows(clean).dropna(subset=[TARGET])
    training, validation_all, test_all = chronological_split(rows)
    # Scope: kitchens already represented when the model is fitted.
    validation = validation_all[validation_all.kitchen.isin(training.kitchen.unique())].copy()
    if len(validation) < 2:
        raise ValueError("Insufficient known-kitchen validation rows")
    models = candidates()
    validation_results = {
        name: scores(validation[TARGET], validation[col]) for name, col in BASELINES.items()
    }
    for name, model in models.items():
        # X = historical/calendar inputs; y = actual waste on the forecast date.
        # fit() learns the preprocessing statistics and regression parameters.
        model.fit(training[FEATURES], training[TARGET])
        validation_results[name] = scores(validation[TARGET], np.maximum(0, model.predict(validation[FEATURES])))
    # Select only among the three trained ML candidates, not the baseline rules.
    # Current data select Ridge; the slightly better validation baseline is retained
    # transparently in the comparison rather than described as a trained ML model.
    selected = min(models, key=lambda name: validation_results[name]["MAE"])
    # Selection precedes test-score calculation in this run. The same holdout has
    # already been inspected in earlier project experiments; it is not a fresh test.
    fit_rows = pd.concat([training, validation_all], ignore_index=True)
    test = test_all[test_all.kitchen.isin(fit_rows.kitchen.unique())].copy()
    if len(test) < 2:
        raise ValueError("Insufficient known-kitchen test rows")
    final_model = None
    if selected in models:
        # Refit a fresh pipeline on train + validation after selecting the model.
        # The test targets are never passed to this fit() call.
        final_model = clone(models[selected]).fit(fit_rows[FEATURES], fit_rows[TARGET])
    payload = {"model": final_model, "selected_name": selected, "features": FEATURES,
               "unit": UNIT, "known_kitchens": sorted(fit_rows.kitchen.unique().tolist()),
               "training_last_date": str(fit_rows.date.max().date()),
               "evaluation_protocol": "rolling next-day, previous calendar-day measurement required"}
    # Rolling next-day evaluation uses earlier measured test-day waste as history
    # for later dates, but does not retrain model weights on those test targets.
    predicted = predict_selected(payload, test[FEATURES])
    test_results = {name: scores(test[TARGET], test[col]) for name, col in BASELINES.items()}
    test_results[selected] = scores(test[TARGET], predicted)
    windows = {}
    for name, part in [("train", training), ("validation", validation_all), ("test", test_all)]:
        windows[name] = {"start": str(part.date.min().date()), "end": str(part.date.max().date()),
                         "rows": len(part), "kitchens": int(part.kitchen.nunique())}
    metrics = {
        "source": "https://zenodo.org/records/3355300", "source_sha256": sha256(RAW_PATH.read_bytes()).hexdigest(),
        "unit": UNIT, "raw_rows": len(clean), "measured_targets": int(clean[TARGET].notna().sum()),
        "eligible_next_day_rows": len(rows), "split": windows,
        "validation_scored_rows": len(validation), "test_scored_rows": len(test),
        "validation_unseen_kitchen_rows_excluded": len(validation_all) - len(validation),
        "test_unseen_kitchen_rows_excluded": len(test_all) - len(test),
        "selection": "lowest validation MAE among three fixed ML candidates; baselines retained for comparison",
        "selected_name": selected, "validation_results": validation_results, "test_results": test_results,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    payload["metrics"] = metrics
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save the full fitted pipeline with its schema, coverage and evaluation data.
    # The dashboard reuses this artifact; clicking Predict does not retrain it.
    with MODEL_PATH.open("wb") as f:
        pickle.dump(payload, f)
    output = ROOT / "docs/real_model"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    comparison = test[["date", "kitchen", TARGET]].copy()
    comparison["selected_prediction"] = predicted
    for name, col in BASELINES.items():
        comparison[name] = test[col].to_numpy()
    comparison.to_csv(output / "test_predictions.csv", index=False, date_format="%Y-%m-%d")
    # Overall scores can hide differences between kitchens; report local errors too.
    per_kitchen = []
    for kitchen, group in comparison.groupby("kitchen"):
        entry = {"kitchen": kitchen, "rows": len(group),
                 "MAE": float(mean_absolute_error(group[TARGET], group.selected_prediction)),
                 "previous_day_MAE": float(mean_absolute_error(group[TARGET], group["Previous day"]))}
        per_kitchen.append(entry)
    pd.DataFrame(per_kitchen).to_csv(output / "per_kitchen.csv", index=False)
    write_report(metrics, output)
    print(json.dumps({"selected": selected, "test": test_results[selected],
                      "test_rows": len(test), "unit": UNIT}, indent=2))
    return payload


def write_report(m, output):
    lines = ["# Real-data training results", "",
             "Source: [URBAN-WASTE](https://zenodo.org/records/3355300).",
             f"Target: recorded `Waste weight sum`; unit: **{UNIT}**.", "",
             "## Protocol", "",
             "Global chronological split by unique eligible dates (60/20/20), not random rows.",
             "Feature preprocessing is fitted only on training data during selection. Selection uses lowest validation MAE among trained ML candidates; baselines are comparison benchmarks.",
             "The selected candidate is refitted on train + validation, then evaluated on the final holdout.",
             "Evaluation is rolling next-day: earlier test-day actuals become available for later test-day features.",
             "It is not a forecast of the whole test period made at one fixed origin.",
             "Only kitchens present at fit time are scored. At least three previous measurements and a previous-calendar-day measurement are required.",
             "Missing waste is never filled with zero; outliers are retained. Served portions and advance orders are excluded.", "",
             "| Split | Start | End | Eligible rows |", "| --- | --- | --- | ---: |"]
    for name, window in m["split"].items():
        lines.append(f"| {name} | {window['start']} | {window['end']} | {window['rows']} |")
    lines += ["", f"Raw records: {m['raw_rows']}; measured targets: {m['measured_targets']}; eligible next-day rows: {m['eligible_next_day_rows']}.",
              f"Validation scored: {m['validation_scored_rows']} (unseen-kitchen exclusions: {m['validation_unseen_kitchen_rows_excluded']}).",
              f"Test scored: {m['test_scored_rows']} (unseen-kitchen exclusions: {m['test_unseen_kitchen_rows_excluded']}).", "",
              "## Validation comparison", "", "| Candidate | MAE | RMSE | R² |", "| --- | ---: | ---: | ---: |"]
    for name, result in sorted(m["validation_results"].items(), key=lambda item: item[1]["MAE"]):
        lines.append(f"| {name} | {result['MAE']:.3f} | {result['RMSE']:.3f} | {result['R2']:.3f} |")
    lines += ["", f"Selected **{m['selected_name']}** by lowest validation MAE among trained ML candidates. The seven-observation average has slightly lower validation MAE and remains a benchmark.", "",
              "## Final holdout (choice already frozen)", "", "| Candidate | MAE | RMSE | R² |", "| --- | ---: | ---: | ---: |"]
    for name, result in m["test_results"].items():
        lines.append(f"| {name} | {result['MAE']:.3f} | {result['RMSE']:.3f} | {result['R2']:.3f} |")
    selected_mae = m["test_results"][m["selected_name"]]["MAE"]
    previous_mae = m["test_results"]["Previous day"]["MAE"]
    lines += ["", f"Selected-model holdout MAE improvement over previous day: {100 * (1 - selected_mae / previous_mae):.1f}%.", "",
              "## Interpretation and limits", "",
              "Selection is restricted to trained ML candidates for the course requirement. The holdout was inspected in the earlier experiment and is not a fresh untouched test in this revision.",
              "These scores concern observed 2018 services at the covered kitchens, with measured history supplied each day.",
              "Aggregate R² can be inflated by differences between kitchens; inspect `per_kitchen.csv` alongside overall metrics.",
              "This does not validate present-day Indian kitchens, unseen kitchens, missing/unlogged services or preparation optimization.",
              "The project guidance discusses kg, but also permits bin/bag proxies; the exported CSV itself does not explicitly define its unit.",
              "Therefore weight values are not relabelled kg, and financial-loss/preparation advice is not generated by this pipeline.",
              "Unit confirmation source checked: [URBAN-WASTE M20 guidance](https://www.gobiernodecanarias.org/medioambiente/descargas/Cofinanciacion-europea/URBAN_WASTE/Documentos_Generados_Comunidad_Practicas/M20_dispositivo_seguimiento_alimentos.pdf).",
              "The Streamlit app opens real-data views by default and prefers a saved Jupyter artifact when present.", "",
              "See `metrics.json` for versions, checksum and exact scores; `test_predictions.csv` for all scored examples.",
              "The saved model retains train+validation fit and is not refitted on the test set."]
    (output / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    train()
