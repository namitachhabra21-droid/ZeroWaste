"""
Builds the charts used in README.md (saved to docs/images/).

Run from the project folder after training:
    python train_model.py
    python docs/make_figures.py
"""

import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import BASE_DIR, DATASET_PATH, DAYS, MODEL_PATH  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "docs", "images")

# Colors (single-series charts use one hue; the selected model is highlighted)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
MUTED = "#b9b8b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
})


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(path, BASE_DIR))


def label_bars(ax, bars, fmt, horizontal=False):
    for b in bars:
        if horizontal:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(b.get_width()),
                    va="center", ha="left", color=INK, fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                    va="bottom", ha="center", color=INK, fontsize=9)


def eda_figure(df):
    df = df.assign(Waste_pct=df.Food_Waste_kg / df.Food_Prepared_kg * 100,
                   Per_customer=df.Food_Prepared_kg / df.Number_of_Customers)
    # Over-preparation relative to what is normal for that service type
    relative = df.Per_customer / df.groupby("Food_Category").Per_customer.transform("median")
    df["Overprep_quartile"] = pd.qcut(relative, 4, labels=["Lowest\n25%", "25–50%", "50–75%", "Highest\n25%"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.28)

    ax = axes[0, 0]
    ax.hist(df.Food_Waste_kg, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1)
    mean = df.Food_Waste_kg.mean()
    ax.axvline(mean, color=INK, linestyle="--", linewidth=1.2)
    ax.text(mean, ax.get_ylim()[1] * 0.95, f"  mean {mean:.1f} kg", color=INK, fontsize=9, va="top")
    ax.set_title("Food waste per service")
    ax.set_xlabel("Food waste (kg)")
    ax.set_ylabel("Number of services")
    ax.grid(axis="x", visible=False)

    ax = axes[0, 1]
    by_quart = df.groupby("Overprep_quartile", observed=True).Waste_pct.mean()
    bars = ax.bar(by_quart.index.astype(str), by_quart.values, color=BLUE, width=0.6)
    label_bars(ax, bars, "{:.1f}%")
    ax.set_title("Over-preparing drives waste")
    ax.set_ylim(0, by_quart.max() * 1.18)
    ax.set_xlabel("Food prepared per customer (vs. service-type norm)")
    ax.set_ylabel("Avg. waste (% of food prepared)")
    ax.grid(axis="x", visible=False)

    ax = axes[1, 0]
    by_day = df.groupby("Day_of_Week").Waste_pct.mean().reindex(DAYS)
    bars = ax.bar([d[:3] for d in DAYS], by_day.values, color=BLUE, width=0.6)
    label_bars(ax, bars, "{:.1f}%")
    ax.set_ylim(0, by_day.max() * 1.18)
    ax.set_title("Mondays waste the most, Saturdays the least")
    ax.set_ylabel("Avg. waste (% of food prepared)")
    ax.grid(axis="x", visible=False)

    ax = axes[1, 1]
    groups = {
        "Regular\nday": df[df.Holiday_Event == 0].Waste_pct.mean(),
        "Holiday /\nevent": df[df.Holiday_Event == 1].Waste_pct.mean(),
        "Up to\n28 °C": df[df.Temperature_C <= 28].Waste_pct.mean(),
        "Above\n28 °C": df[df.Temperature_C > 28].Waste_pct.mean(),
    }
    bars = ax.bar(list(groups), list(groups.values()), color=[MUTED, BLUE, MUTED, BLUE], width=0.6)
    label_bars(ax, bars, "{:.1f}%")
    ax.set_ylim(0, max(groups.values()) * 1.18)
    ax.set_title("Holidays and hot days waste more")
    ax.set_ylabel("Avg. waste (% of food prepared)")
    ax.grid(axis="x", visible=False)

    save(fig, "eda_insights.png")


def model_figure(payload):
    results = pd.DataFrame(payload["results"]).T.sort_values("R2_Score", ascending=False)
    colors = [BLUE if n == payload["model_name"] else MUTED for n in results.index]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.subplots_adjust(wspace=0.45)
    for ax, col, title, fmt in [
        (axes[0], "R2_Score", "Test R² (higher is better)", "{:.3f}"),
        (axes[1], "MAE", "Mean absolute error, kg (lower is better)", "{:.2f}"),
    ]:
        bars = ax.barh(results.index[::-1], results[col][::-1], color=colors[::-1], height=0.55)
        label_bars(ax, bars, fmt, horizontal=True)
        ax.set_title(title)
        ax.set_xlim(0, results[col].max() * 1.2)
        ax.grid(axis="y", visible=False)
    save(fig, "model_comparison.png")


def prediction_figure(payload):
    y = np.array(payload["test_actual"])
    p = np.array(payload["test_predicted"])
    resid = y - p

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.subplots_adjust(wspace=0.3)

    ax = axes[0]
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(hi, hi, "perfect prediction  ", color=INK_2, fontsize=9, ha="right", va="bottom")
    ax.scatter(y, p, s=22, color=BLUE, alpha=0.75, edgecolors=SURFACE, linewidths=0.6)
    ax.set_title(f"Actual vs predicted (200 unseen test services)")
    ax.set_xlabel("Actual waste (kg)")
    ax.set_ylabel("Predicted waste (kg)")

    ax = axes[1]
    ax.hist(resid, bins=25, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
    within = np.mean(np.abs(resid) <= 10) * 100
    ax.set_title(f"Prediction errors ({within:.0f}% within ±10 kg)")
    ax.set_xlabel("Actual − predicted (kg)")
    ax.set_ylabel("Number of services")
    ax.grid(axis="x", visible=False)
    save(fig, "predictions.png")


def importance_figure(payload):
    imp = pd.Series(payload["feature_importance"]).sort_values(ascending=False).head(8)[::-1]
    labels = [n.replace("Customers_x_", "Customers × ").replace("_", " ") for n in imp.index]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.barh(labels, imp.values, color=BLUE, height=0.6)
    label_bars(ax, bars, "{:.1f} kg", horizontal=True)
    ax.set_xlim(0, imp.max() * 1.15)
    ax.set_title(f"What drives the prediction ({payload['model_name']}, permutation importance)")
    ax.set_xlabel("Extra average error (kg) when the input is shuffled")
    ax.grid(axis="y", visible=False)
    save(fig, "feature_importance.png")


if __name__ == "__main__":
    data = pd.read_csv(DATASET_PATH)
    with open(MODEL_PATH, "rb") as f:
        model_payload = pickle.load(f)
    eda_figure(data)
    model_figure(model_payload)
    prediction_figure(model_payload)
    importance_figure(model_payload)
