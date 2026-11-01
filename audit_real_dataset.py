"""Audit the unmodified URBAN-WASTE CSV without third-party dependencies."""

import csv
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset/raw/urban_waste_food_tracking.csv"
REPORT = ROOT / "docs/real_dataset_audit.md"
EXPECTED_MD5 = "a88a1d7b09abd17064087e62d1704797"


def main():
    raw = SOURCE.read_bytes()
    checksum = hashlib.md5(raw).hexdigest()
    if checksum != EXPECTED_MD5:
        raise ValueError("Source checksum differs from the published Zenodo file.")
    # The source uses semicolons, decimal commas and Western European text.
    rows = list(csv.DictReader(raw.decode("cp1252").splitlines(), delimiter=";"))
    dates = [datetime.strptime(r["Timestamp"], "%d/%m/%Y").date() for r in rows]
    kitchens = Counter(r["City / kitchen"] for r in rows)
    missing = {col: sum(not r[col].strip() for r in rows) for col in rows[0]}
    target = "Waste weight sum"
    valid = [r for r in rows if r[target].strip()]
    weights = [float(r[target].replace(",", ".")) for r in valid]
    duplicates = len(rows) - len({tuple(r.values()) for r in rows})
    keys = Counter((r["City / kitchen"], r["Timestamp"]) for r in rows)
    report = [
        "# Real dataset feasibility audit",
        "",
        "Source: [URBAN-WASTE Food Tracking Device data](https://zenodo.org/records/3355300)",
        "by Mattias Eriksson and Samuel Lindgren (2019).",
        "The source describes food-waste records collected at pilot hotels and restaurants.",
        f"Raw file: `dataset/raw/urban_waste_food_tracking.csv`; MD5: `{checksum}` (matches publisher).",
        "",
        "Run `python3 audit_real_dataset.py` to reproduce this report. No pandas required.",
        "",
        "## Observed contents",
        "",
        f"- Rows: {len(rows):,}; columns: {len(rows[0])}; kitchen identifiers: {len(kitchens)}.",
        f"- Dates: {min(dates)} through {max(dates)}.",
        f"- Rows with a waste target: {len(valid):,}; missing waste targets: {missing[target]}.",
        f"- Exact duplicate rows: {duplicates}; repeated kitchen/date keys: {sum(v > 1 for v in keys.values())}.",
        f"- Recorded target range: {min(weights)} to {max(weights)}; negative values: {sum(w < 0 for w in weights)}.",
        "",
        "| Column | Missing rows | Missing % |",
        "| --- | ---: | ---: |",
    ]
    report += [f"| {col} | {n:,} | {n / len(rows):.1%} |" for col, n in missing.items()]
    report += [
        "",
        "## Decision and implementation plan",
        "",
        "This is a candidate for waste forecasting within the observed kitchens, subject to unit verification and evaluation.",
        "The source has no prepared weight, temperature or service category columns, so it cannot replace the earlier simulated data one-to-one.",
        "",
        "1. Verify the exact unit and interpretation of `Waste weight sum`. The CSV header and Zenodo description do not state the unit explicitly; do not relabel it kg without confirmation.",
        "2. Keep raw data intact. Parse semicolon separators, decimal commas and day-first dates; exclude missing targets from supervised scoring and investigate extremes rather than deleting them automatically.",
        "3. Forecast using kitchen, calendar features and strictly prior waste history. Create lags within each kitchen after date sorting. Distinguish previous recorded service from previous calendar day; retain missing dates as missing, not zero waste.",
        "4. Do not rely on advance orders: these fields are almost entirely missing. Same-day served portions are unavailable before service and should not be pre-service predictors.",
        "5. Use a chronological holdout and training-only preprocessing. Compare against previous-observation and historical-average baselines. Report MAE/RMSE and per-kitchen coverage; reserve the final holdout from model selection.",
        "6. Adapt app inputs and documentation only after evaluation. Existing preparation recommendations and money estimates need revised assumptions and confirmed weight units.",
        "",
        "This audit does not train a model. Training and evaluation are available separately via `train_real_model.py`; see `docs/real_model/results.md` after running it. Real observations alone do not establish forecasting accuracy or transfer to new kitchens.",
        "The Streamlit app shows real-data views only.",
        "",
        "## Records per kitchen",
        "",
        "| Kitchen | Rows |",
        "| --- | ---: |",
    ]
    report += [f"| {name} | {n} |" for name, n in kitchens.most_common()]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Audited {len(rows):,} rows; {len(valid):,} non-missing targets across {len(kitchens)} kitchen identifiers.")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
