"""Streamlit views for the real-data forecast, using saved notebook/CLI artifacts."""

import hashlib
import json
import pickle

import altair as alt
import pandas as pd
import streamlit as st

from predict_real_waste import forecast
from real_data import FEATURES, features_from_history, MIN_HISTORY, MODEL_PATH, RAW_PATH, ROOT, TARGET, load_clean_data

SELECTED_COLOR = "#0D6E65"
BASELINE_COLOR = "#8A9A96"
LEARNED_COLOR = "#D08C1E"
ROLE_SCALE = alt.Scale(
    domain=["Selected method", "Baseline (not selected)", "Learned model (not selected)"],
    range=[SELECTED_COLOR, BASELINE_COLOR, LEARNED_COLOR],
)
BASELINE_NAMES = {"Previous day", "Recent 7 observations mean", "Historical mean"}


@st.cache_resource
def load_forecaster(path, revision):
    """Load a fitted pipeline and its metrics; this function does not train models."""
    # revision changes when a notebook run replaces its saved artifact.
    with open(path, "rb") as f:
        payload = pickle.load(f)
    required = {"selected_name", "known_kitchens", "training_last_date", "unit", "metrics"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Saved forecast is incomplete. Run the notebook's save step again.")
    selected = payload["selected_name"]
    metrics = payload["metrics"]
    if selected not in metrics.get("test_results", {}) or selected not in metrics.get("validation_results", {}):
        raise ValueError("Saved forecast has no matching evaluation results. Run the notebook again.")
    return payload


@st.cache_data
def load_observations(path, revision):
    return load_clean_data(path)


def file_revision(path):
    # Content hash ensures retraining is reflected even if the file size is unchanged.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_docs_csv(name):
    path = ROOT / "docs/real_model" / name
    return pd.read_csv(path) if path.exists() else None


def read_posthoc():
    path = ROOT / "docs/real_model/posthoc_test_results.json"
    return json.loads(path.read_text()) if path.exists() else None


CARD_CSS = """
<style>
div[data-testid="stMetric"] {
  background: rgba(13, 110, 101, 0.06);
  border: 1px solid rgba(13, 110, 101, 0.18);
  border-radius: 12px;
  padding: 14px 16px;
}
.zw-card {
  border: 1px solid rgba(128, 128, 128, 0.25);
  border-radius: 12px;
  padding: 14px 16px;
  height: 100%;
}
.zw-card h4 { margin: 0 0 6px 0; font-size: 0.95rem; color: #0D6E65; }
.zw-card p { margin: 0; font-size: 0.92rem; line-height: 1.45; }
.zw-step-num {
  display: inline-block; width: 26px; height: 26px; line-height: 26px; text-align: center;
  border-radius: 50%; background: #0D6E65; color: #fff; font-weight: 600; margin-right: 6px;
}
</style>
"""


def card(title, body):
    return f'<div class="zw-card"><h4>{title}</h4><p>{body}</p></div>'


def render_real_app():
    """Load a saved training run and route the sidebar selection to its view."""
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.caption("Real hotel and restaurant records · URBAN-WASTE · historical next-day forecasting")
    # The notebook and training script save separate artifacts. Prefer the notebook
    # when both exist; the sidebar can switch between the saved training runs.
    artifacts = {
        label: path for label, path in [
            ("Jupyter notebook", ROOT / "model/real_notebook_model.pkl"),
            ("Training script", MODEL_PATH),
        ] if path.exists()
    }
    if not artifacts:
        st.info("No saved real-data forecast found. Run Steps 1–8 in real_food_waste_training.ipynb, then refresh this page.")
        return
    if not RAW_PATH.exists():
        st.error("The real kitchen dataset is missing. Restore dataset/raw/urban_waste_food_tracking.csv.")
        return
    with st.sidebar:
        source = st.selectbox("Training run", options=list(artifacts))
        view = st.radio("Real-data section", [
            "Overview", "Forecast waste", "Training results", "Dataset explorer", "How it works",
        ])
    try:
        artifact = artifacts[source]
        payload = load_forecaster(str(artifact), file_revision(artifact))
        source_revision = file_revision(RAW_PATH)
        clean = load_observations(str(RAW_PATH), source_revision)
    except (ValueError, KeyError, OSError, EOFError, pickle.UnpicklingError, ImportError) as exc:
        st.error(f"Unable to load the saved forecast: {exc}")
        return
    metrics = payload["metrics"]
    selected = payload["selected_name"]
    test_scores = metrics["test_results"][selected]
    with st.sidebar:
        st.markdown("---")
        st.write(f"**Selected method:** {selected}")
        st.write(f"**Test MAE:** {test_scores['MAE']:.2f} source units")
        st.caption(f"Training and validation end: {payload['training_last_date']}")
    if metrics.get("source_sha256") != source_revision:
        st.warning("The available data differs from the training dataset. Scores below describe the saved training run; they do not evaluate these updated observations.")
    if view == "Overview":
        render_overview(payload, clean)
    elif view == "Forecast waste":
        render_forecast(payload, clean)
    elif view == "Training results":
        render_results(payload)
    elif view == "Dataset explorer":
        render_dataset(clean)
    else:
        render_how_it_works()


def render_overview(payload, clean):
    metrics = payload["metrics"]
    selected = payload["selected_name"]
    test = metrics["test_results"][selected]
    previous = metrics["test_results"]["Previous day"]
    improvement = 100 * (1 - test["MAE"] / previous["MAE"])
    predictions = read_docs_csv("test_predictions.csv")

    st.markdown(f"## Next-day food waste forecast for real kitchens")
    st.success(
        f"**Method used: {selected}.** On {metrics.get('test_scored_rows', metrics.get('test_rows'))} "
        f"unseen test records it had {improvement:.1f}% lower average error than the previous-day baseline. "
        "It has the lowest validation MAE among trained ML candidates. The moving-average baseline had slightly lower validation MAE and remains a benchmark."
    )

    mae_text = f"Typical error about {test['MAE']:.1f} units. That is {improvement:.0f}% better than repeating yesterday's value."
    a, b, c = st.columns(3)
    a.markdown(card("What it predicts", "Tomorrow's recorded food waste for one kitchen, using only that kitchen's recent measurements."), unsafe_allow_html=True)
    b.markdown(card("How well it does", mae_text), unsafe_allow_html=True)
    c.markdown(card("What to trust it for", "Comparing methods on historical data. It is not validated for new kitchens or live daily decisions."), unsafe_allow_html=True)
    st.write("")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average error (MAE)", f"{test['MAE']:.2f}", help="Typical gap between predicted and recorded waste, in source weight units.")
    k2.metric("Better than previous day", f"{improvement:.1f}%", help="Relative MAE reduction versus repeating yesterday's value.")
    k3.metric("R² (explained variation)", f"{test['R2']:.3f}")
    k4.metric("Kitchens in test", int(predictions.kitchen.nunique()) if predictions is not None else "—")

    st.markdown("### How the methods compare on unseen test data")
    include_learned = st.toggle("Include learned models tested after selection", value=True)
    st.caption("Lower average error is better. Teal is the method used. Grey are simple baselines. Amber models were trained and tested for comparison only. Click a legend entry to highlight that group.")
    st.altair_chart(method_chart(metrics, selected, include_learned), width="stretch")

    if predictions is not None:
        st.markdown("### Predicted vs recorded waste over time")
        kitchens = sorted(predictions.kitchen.unique())
        kitchen = st.selectbox("Kitchen", kitchens, index=0, key="overview_kitchen")
        st.altair_chart(timeline_chart(predictions[predictions.kitchen == kitchen], selected), width="stretch")
        st.caption("Dots are recorded waste. The teal line is the forecast for each day.")

    if predictions is not None:
        st.markdown("### How big are the typical misses?")
        errors = (predictions["waste_weight"] - predictions["selected_prediction"]).abs()
        median_error = errors.median()
        st.altair_chart(error_histogram(errors, median_error), width="stretch")
        st.markdown(card(
            "In plain words",
            f"Half of all forecasts were within <b>{median_error:.1f}</b> units of the recorded waste. "
            "Bars further right are the bigger misses, and each one is a single test day.",
        ), unsafe_allow_html=True)

    per_kitchen = read_docs_csv("per_kitchen.csv")
    if per_kitchen is not None:
        st.markdown("### Does it beat yesterday's value in every kitchen?")
        st.caption("How much lower the method's average error is than simply repeating the previous day, per kitchen. Longer bars are better.")
        st.altair_chart(per_kitchen_chart(per_kitchen), width="stretch")

    with st.expander("How to read these numbers"):
        st.markdown("""
        - **MAE (average error):** the typical gap between predicted and recorded waste. It's easy to explain and uses the same unit as the data.
        - **R²:** share of the variation in waste that the forecast explains. 1.0 is perfect; 0 means no better than the average.
        - **Test data:** records from after the training period, never used to choose the method.
        - **Units:** the source export's exact weight unit is unconfirmed, so values are shown in source units.
        """)


def method_chart(metrics, selected, include_learned):
    posthoc = read_posthoc() if include_learned else None
    rows = []
    for name, s in metrics["test_results"].items():
        rows.append({"Method": name, "MAE": s["MAE"], "Role": "Selected method" if name == selected else "Baseline (not selected)"})
    if posthoc:
        for name, s in posthoc["test_results"].items():
            if name in metrics["test_results"]:
                continue
            rows.append({"Method": name, "MAE": s["MAE"], "Role": "Learned model (not selected)"})
    df = pd.DataFrame(rows)
    # Click a legend entry to highlight that group; click again to clear.
    pick = alt.selection_point(fields=["Role"], bind="legend")
    return alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
        y=alt.Y("Method:N", sort=alt.EncodingSortField(field="MAE", order="ascending"), title=None),
        x=alt.X("MAE:Q", title="Average error on test data (source units, lower is better)"),
        color=alt.Color("Role:N", scale=ROLE_SCALE, legend=alt.Legend(orient="bottom", title="Click to highlight")),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.25)),
        tooltip=["Method", alt.Tooltip("MAE:Q", format=".2f"), "Role"],
    ).add_params(pick).properties(height=220)


def timeline_chart(frame, selected):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    base = alt.Chart(frame).encode(x=alt.X("date:T", title=None))
    actual = base.mark_point(filled=True, size=55, color="#2B3A38", opacity=0.8).encode(
        y=alt.Y("waste_weight:Q", title="Waste (source units)"),
        tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip("waste_weight:Q", format=".1f", title="Recorded")],
    )
    forecast_line = base.mark_line(strokeWidth=2.5, color=SELECTED_COLOR).encode(
        y="selected_prediction:Q",
        tooltip=[alt.Tooltip("selected_prediction:Q", format=".1f", title=selected)],
    )
    return (actual + forecast_line).properties(height=300)


def per_kitchen_chart(per_kitchen):
    frame = per_kitchen.assign(
        improvement=lambda d: 100 * (1 - d["MAE"] / d["previous_day_MAE"]),
    ).sort_values("improvement")
    bars = alt.Chart(frame).mark_bar(color=SELECTED_COLOR, cornerRadiusEnd=3).encode(
        y=alt.Y("kitchen:N", sort=frame.kitchen.tolist(), title=None),
        x=alt.X("improvement:Q", title="Error reduction vs repeating yesterday (%)"),
        tooltip=[
            "kitchen",
            alt.Tooltip("improvement:Q", format=".0f", title="Reduction %"),
            alt.Tooltip("MAE:Q", format=".2f", title="Method MAE"),
            alt.Tooltip("previous_day_MAE:Q", format=".2f", title="Previous-day MAE"),
        ],
    )
    labels = bars.mark_text(align="left", dx=4, color="#2B3A38").encode(text=alt.Text("improvement:Q", format=".0f"))
    return (bars + labels).properties(height=max(240, 20 * len(frame)))


def error_histogram(errors, median_error):
    frame = pd.DataFrame({"Absolute error": errors})
    hist = alt.Chart(frame).mark_bar(color=SELECTED_COLOR, opacity=0.85).encode(
        x=alt.X("Absolute error:Q", bin=alt.Bin(maxbins=20), title="Size of the miss (source units)"),
        y=alt.Y("count()", title="Number of test days"),
        tooltip=[alt.Tooltip("count()", title="Days")],
    )
    median = alt.Chart(pd.DataFrame({"x": [median_error]})).mark_rule(color=LEARNED_COLOR, strokeWidth=2).encode(x="x:Q")
    return (hist + median).properties(height=240)


def forecast_window(clean, kitchen, date, size=7):
    """The same measured history the forecast uses: the latest `size` values before the date."""
    date = pd.Timestamp(date).normalize()
    past = clean.loc[(clean.kitchen == kitchen) & (clean.date < date)].dropna(subset=[TARGET]).sort_values("date")
    return past.tail(size)[["date", TARGET]].reset_index(drop=True)


def kitchen_typical_error(payload, kitchen):
    per_kitchen = read_docs_csv("per_kitchen.csv")
    if per_kitchen is not None and kitchen in set(per_kitchen.kitchen):
        return float(per_kitchen.loc[per_kitchen.kitchen == kitchen, "MAE"].iloc[0]), "this kitchen's own test error"
    return payload["metrics"]["test_results"][payload["selected_name"]]["MAE"], "the average test error"


def render_explanation(payload, clean, kitchen, date, result):
    st.markdown("#### How this number was made")
    window = forecast_window(clean, kitchen, date)
    if result["selected_model"] != "Recent 7 observations mean" or window.empty:
        st.write("The fitted model combines past waste, kitchen identity and calendar features using weights learned during training.")
        past = clean.loc[(clean.kitchen == kitchen) & (clean.date < pd.Timestamp(date))].dropna(subset=[TARGET]).sort_values("date")
        history = list(past[["date", TARGET]].itertuples(index=False, name=None))
        features = features_from_history(kitchen, date, history)
        st.dataframe(pd.DataFrame([features])[FEATURES], hide_index=True, width="stretch")
        st.caption("lag_1/2/3: last three measurements; mean_3/7 and std_7: recent averages and variation; historical_mean/count: earlier history; calendar sin/cos: weekday and month.")
        return
    values = window[TARGET].to_list()
    total = sum(values)
    n = len(values)
    mean = total / n
    typical, source = kitchen_typical_error(payload, kitchen)
    low, high = max(0.0, result["predicted_waste"] - typical), result["predicted_waste"] + typical

    s1, s2, s3 = st.columns(3)
    s1.markdown(card(
        f'<span class="zw-step-num">1</span>Last {n} measured days',
        f"{window.date.min().date()} to {window.date.max().date()}. Missing days are skipped, not treated as zero.",
    ), unsafe_allow_html=True)
    s2.markdown(card(
        f'<span class="zw-step-num">2</span>Add them up',
        f"Sum of the {n} values = <b>{total:.1f}</b> units.",
    ), unsafe_allow_html=True)
    s3.markdown(card(
        f'<span class="zw-step-num">3</span>Average them',
        f"{total:.1f} ÷ {n} = <b>{mean:.2f}</b>. That is the forecast.",
    ), unsafe_allow_html=True)

    bars = alt.Chart(window.assign(label=window.date.dt.strftime("%d %b"))).mark_bar(color=SELECTED_COLOR, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("label:O", sort=None, title=None),
        y=alt.Y(f"{TARGET}:Q", title="Recorded waste (source units)"),
        tooltip=[alt.Tooltip("label:O", title="Day"), alt.Tooltip(f"{TARGET}:Q", format=".1f", title="Recorded")],
    )
    mean_line = alt.Chart(pd.DataFrame({"y": [mean]})).mark_rule(color=LEARNED_COLOR, strokeWidth=2, strokeDash=[6, 4]).encode(
        y="y:Q",
        tooltip=[alt.Tooltip("y:Q", format=".2f", title="Forecast (mean)")],
    )
    st.altair_chart((bars + mean_line).properties(height=260), width="stretch")
    st.caption("Bars are the recorded values the forecast used. The dashed amber line is their average, which is the forecast.")

    st.markdown(card(
        "How sure to be",
        f"Expect about <b>{result['predicted_waste']:.1f}</b> units, most likely between <b>{low:.1f}</b> and <b>{high:.1f}</b>. "
        f"That range uses {source} (±{typical:.1f}). It is a guide, not a guarantee.",
    ), unsafe_allow_html=True)


@st.fragment
def what_if(payload, clean, kitchen, date):
    """Offer an arithmetic what-if editor only for the moving-average baseline.

    The editor is hidden for Ridge, whose prediction combines multiple features.
    Streamlit reruns only this fragment when its editable table changes.
    """
    if payload["selected_name"] != "Recent 7 observations mean":
        return
    window = forecast_window(clean, kitchen, date)
    if window.empty:
        return
    st.markdown("#### Try a what-if")
    st.caption("Change any recorded value below. The forecast is their average, so it updates as you type.")
    editable = pd.DataFrame({
        "Day": window.date.dt.strftime("%d %b %Y"),
        "Recorded waste": window[TARGET].astype(float),
    })
    edited = st.data_editor(
        editable, disabled=["Day"], hide_index=True, width="stretch",
        column_config={"Recorded waste": st.column_config.NumberColumn(min_value=0.0, format="%.1f")},
        key=f"whatif_{kitchen}_{pd.Timestamp(date).date()}",
    )
    original = window[TARGET].mean()
    changed = pd.to_numeric(edited["Recorded waste"], errors="coerce").dropna().mean()
    delta = changed - original
    a, b = st.columns(2)
    a.markdown(card("Original forecast", f"<b>{original:.2f}</b> units, from the recorded values."), unsafe_allow_html=True)
    b.markdown(card(
        "What-if forecast",
        f"<b>{changed:.2f}</b> units ({delta:+.2f} vs original).",
    ), unsafe_allow_html=True)


def render_forecast(payload, clean):
    """Collect a known kitchen and date, derive inputs, and display one forecast.

    Historical waste is already loaded from the supplied dataset. This view is not
    a data-upload workflow for new restaurants, nor a live present-day data feed.
    """
    st.subheader("Forecast next-day food waste")
    st.markdown(f"**Past waste measurements → {payload['selected_name']} → predicted next-day waste**")
    st.write("A restaurant supplies dated records of its measured food waste and its kitchen ID. In this demo those records are already loaded: choose a kitchen and the date you want to predict.")
    st.info("Required history: at least 3 earlier waste measurements, including the day immediately before the forecast. The model automatically calculates previous waste values, recent averages, variation and calendar features. Output: one predicted waste value for the selected kitchen and date, in source weight units.")
    st.caption("This dashboard uses the supplied historical dataset; it does not currently collect or upload a new restaurant's daily records. Present-day use needs updated records and evaluation.")
    known = sorted(set(payload["known_kitchens"]) & set(clean.kitchen))
    if not known:
        st.warning("No kitchens from the saved training run are present in this dataset.")
        return
    default = known.index("Copenhagen / Axel") if "Copenhagen / Axel" in known else 0
    kitchen = st.selectbox("Kitchen", known, index=default)
    kitchen_data = clean[clean.kitchen == kitchen].sort_values("date")
    observed = kitchen_data.dropna(subset=[TARGET])
    if len(observed) < MIN_HISTORY:
        st.warning("This kitchen needs at least three measured services before forecasting.")
        return
    first_forecast = pd.Timestamp(payload["training_last_date"]) + pd.Timedelta(days=1)
    last_forecast = observed.date.max() + pd.Timedelta(days=1)
    if last_forecast < first_forecast:
        st.warning("This kitchen has no recent enough history after the saved training period.")
        return
    st.caption(f"Available measurements: {observed.date.min().date()} to {observed.date.max().date()}. This is a historical example; a forecast for today needs updated prior-day measurements.")
    with st.form("real_forecast"):
        date = st.date_input("Forecast date", value=last_forecast.date(),
                             min_value=first_forecast.date(), max_value=last_forecast.date(),
                             key=f"real_date_{kitchen}")
        submitted = st.form_submit_button("Predict next-day waste")
    what_if(payload, clean, kitchen, date)
    past = kitchen_data[kitchen_data.date < pd.Timestamp(date)]
    st.markdown("#### Recorded history before the forecast date")
    st.line_chart(past.set_index("date")[[TARGET]].tail(30).rename(columns={TARGET: "Recorded waste (source units)"}))
    st.caption("Shows the last 30 recorded rows. Missing measurements are not treated as zero.")
    if submitted:
        try:
            # Call the shared inference function used by the command-line tool.
            # It rebuilds historical features and applies the saved fitted pipeline.
            result = forecast(payload, clean, kitchen, date)
        except ValueError as exc:
            st.warning(str(exc))
            return
        st.metric("Predicted waste · source weight units", f"{result['predicted_waste']:.3f}")
        st.write(f"Forecast for **{result['forecast_date']}** using measurements through **{result['history_last_date']}**.")
        st.write(f"Selected method: **{result['selected_model']}**")
        if result["selected_model"] == "Recent 7 observations mean":
            st.caption("Average of up to seven previous measured observations for this kitchen. Missing dates mean these are not necessarily seven consecutive days.")
        render_explanation(payload, clean, kitchen, date, result)


def render_results(payload):
    """Display validation comparisons and test scores belonging to the saved run."""
    st.subheader("Training and test results")
    metrics = payload["metrics"]
    selected = payload["selected_name"]
    scores = metrics["test_results"][selected]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R²", f"{scores['R2']:.4f}")
    c2.metric("Test MAE · source units", f"{scores['MAE']:.3f}")
    c3.metric("Test RMSE · source units", f"{scores['RMSE']:.3f}")
    st.caption("R² describes explained variation, not the percentage of correct predictions. MAE is the average absolute prediction error.")

    st.markdown("#### Step 1 · Selection on validation data")
    st.success(f"Selected by lowest validation MAE among trained ML candidates: **{selected}**. Validation data was used to choose; test data was not.")
    st.caption("We select a trained ML model for the course requirement. The seven-observation average has slightly lower validation MAE. This holdout was inspected in the earlier experiment and is not a fresh untouched test.")
    comparison = pd.DataFrame(metrics["validation_results"]).T.sort_values("MAE")
    comparison.index.name = "Method"
    st.dataframe(comparison, width="stretch")

    st.markdown("#### Step 2 · Final test on unseen records")
    st.dataframe(pd.DataFrame(metrics["test_results"]).T, width="stretch")
    rows = metrics.get("test_scored_rows", metrics.get("test_rows"))
    st.write(f"Scored final test records: **{rows}**")
    excluded = metrics.get("test_unseen_kitchen_rows_excluded")
    if excluded is not None:
        st.write(f"Unseen-kitchen test records excluded: **{excluded}**")
    else:
        st.caption("Testing is limited to kitchens present when the method was fitted.")

    predictions = read_docs_csv("test_predictions.csv")
    if predictions is not None:
        st.markdown("#### Predicted vs recorded, every test day")
        st.caption("Each dot is one test day. Dots on the dashed line were forecast exactly.")
        st.altair_chart(scatter_chart(predictions), width="stretch")

    posthoc = read_posthoc()
    st.markdown("#### Other models, tested after selection")
    if posthoc:
        st.warning(posthoc["label"])
        rows_table = pd.DataFrame(posthoc["test_results"]).T[["MAE", "RMSE", "R2"]]
        rows_table.index.name = "Model"
        st.table(rows_table.round(3))
        st.caption("Additional test comparisons do not determine selection. Ridge Regression is selected by lowest validation MAE among trained ML candidates.")
    else:
        st.info("Post-hoc comparison not generated yet. Run `python posthoc_comparison.py`.")
    st.link_button("Dataset source", metrics.get("source", "https://zenodo.org/records/3355300"))


def scatter_chart(predictions):
    points = alt.Chart(predictions).mark_circle(size=50, color=SELECTED_COLOR, opacity=0.5).encode(
        x=alt.X("selected_prediction:Q", title="Predicted waste (source units)"),
        y=alt.Y("waste_weight:Q", title="Recorded waste (source units)"),
        tooltip=[
            "kitchen", alt.Tooltip("date:T", format="%d %b %Y"),
            alt.Tooltip("selected_prediction:Q", format=".1f", title="Predicted"),
            alt.Tooltip("waste_weight:Q", format=".1f", title="Recorded"),
        ],
    )
    top = max(predictions["waste_weight"].max(), predictions["selected_prediction"].max())
    diagonal = alt.Chart(pd.DataFrame({"v": [0, top]})).mark_line(strokeDash=[6, 4], color=BASELINE_COLOR).encode(x="v:Q", y="v:Q")
    return (diagonal + points).properties(height=320)


def render_how_it_works():
    st.subheader("How the forecast works")
    st.markdown("""
    1. Choose a kitchen already represented in training and a date after the training period.
    2. The forecast uses only that kitchen's measurements from earlier dates.
    3. At least three earlier measurements, including one from the previous calendar day, are required.
    4. The saved method estimates the next day's recorded waste.

    Ridge Regression, Decision Tree and Random Forest were compared with simple baselines.
    Ridge Regression has the lowest validation MAE among trained ML candidates and is the main predictor. The moving-average baseline has slightly lower validation MAE. The existing holdout was inspected in the earlier experiment.
    The comparison seven-observation moving average is a simple time-series forecasting method. It does not fit regression coefficients and is different from the MA component in ARIMA.

    Testing simulates daily operation: each day's actual measurement becomes available for the following days.
    These results do not establish reliability for unseen kitchens or current operations.
    """)
    st.write("The dataset does not supply the original app's prepared-food and weather inputs. Financial loss and preparation advice require confirmed units and additional data.")
    st.link_button("View original dataset", "https://zenodo.org/records/3355300")


def render_dataset(clean):
    st.subheader("Real kitchen dataset")
    a, b, c = st.columns(3)
    a.metric("Records", len(clean))
    b.metric("Kitchens", clean.kitchen.nunique())
    c.metric("Missing waste measurements", int(clean[TARGET].isna().sum()))
    selected = st.selectbox("Filter kitchen", ["All kitchens", *sorted(clean.kitchen.unique())])
    shown = clean if selected == "All kitchens" else clean[clean.kitchen == selected]

    dated = shown.dropna(subset=[TARGET])
    if not dated.empty:
        low, high = dated.date.min().date(), dated.date.max().date()
        window = st.slider("Date range", min_value=low, max_value=high, value=(low, high), format="DD MMM YYYY")
        dated = dated[(dated.date >= pd.Timestamp(window[0])) & (dated.date <= pd.Timestamp(window[1]))]
        st.markdown("#### Recorded waste over time")
        plotted = st.multiselect(
            "Kitchens to plot (up to 3)", sorted(dated.kitchen.unique()),
            default=sorted(dated.kitchen.unique())[:1], max_selections=3,
        )
        if plotted:
            trend = alt.Chart(dated[dated.kitchen.isin(plotted)]).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y(f"{TARGET}:Q", title="Recorded waste (source units)"),
                color=alt.Color("kitchen:N", legend=alt.Legend(orient="top", title=None)),
                tooltip=["kitchen", alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip(f"{TARGET}:Q", format=".1f")],
            ).properties(height=300)
            st.altair_chart(trend, width="stretch")
        st.caption(f"{len(dated)} measured records in this date range.")

    st.dataframe(shown.rename(columns={TARGET: "Recorded waste (source units)"}), width="stretch")
    st.download_button("Download filtered observations", shown.to_csv(index=False),
                       file_name="real_kitchen_observations.csv", mime="text/csv")
