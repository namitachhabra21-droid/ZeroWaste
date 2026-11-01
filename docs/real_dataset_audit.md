# Real dataset feasibility audit

Source: [URBAN-WASTE Food Tracking Device data](https://zenodo.org/records/3355300)
by Mattias Eriksson and Samuel Lindgren (2019).
The source describes food-waste records collected at pilot hotels and restaurants.
Raw file: `dataset/raw/urban_waste_food_tracking.csv`; MD5: `a88a1d7b09abd17064087e62d1704797` (matches publisher).

Run `python3 audit_real_dataset.py` to reproduce this report. No pandas required.

## Observed contents

- Rows: 3,321; columns: 13; kitchen identifiers: 34.
- Dates: 2018-03-02 through 2018-11-21.
- Rows with a waste target: 3,160; missing waste targets: 161.
- Exact duplicate rows: 0; repeated kitchen/date keys: 0.
- Recorded target range: 0.0 to 425.5; negative values: 0.

| Column | Missing rows | Missing % |
| --- | ---: | ---: |
| City / kitchen | 0 | 0.0% |
| Timestamp | 0 | 0.0% |
| Waste weight sum | 161 | 4.8% |
| portions, Whole day | 2,784 | 83.8% |
| ordered portions, Whole day | 3,313 | 99.8% |
| portions, Midday | 3,072 | 92.5% |
| ordered portions, Midday | 3,320 | 100.0% |
| portions, Breakfast | 2,925 | 88.1% |
| ordered portions, Breakfast | 3,320 | 100.0% |
| portions, Lunch | 3,190 | 96.1% |
| ordered portions, Lunch | 3,319 | 99.9% |
| portions, Brunch | 3,316 | 99.8% |
| ordered portions, Brunch | 3,321 | 100.0% |

## Decision and implementation plan

This is a candidate for waste forecasting within the observed kitchens, subject to unit verification and evaluation.
It is not a direct replacement for the current synthetic CSV: prepared weight, temperature and the current service categories are absent.

1. Verify the exact unit and interpretation of `Waste weight sum`. The CSV header and Zenodo description do not state the unit explicitly; do not relabel it kg without confirmation.
2. Keep raw data intact. Parse semicolon separators, decimal commas and day-first dates; exclude missing targets from supervised scoring and investigate extremes rather than deleting them automatically.
3. Forecast using kitchen, calendar features and strictly prior waste history. Create lags within each kitchen after date sorting. Distinguish previous recorded service from previous calendar day; retain missing dates as missing, not zero waste.
4. Do not rely on advance orders: these fields are almost entirely missing. Same-day served portions are unavailable before service and should not be pre-service predictors.
5. Use a chronological holdout and training-only preprocessing. Compare against previous-observation and historical-average baselines. Report MAE/RMSE and per-kitchen coverage; reserve the final holdout from model selection.
6. Adapt app inputs and documentation only after evaluation. Existing preparation recommendations and money estimates need revised assumptions and confirmed weight units.

This audit does not train a model. Training and evaluation are available separately via `train_real_model.py`; see `docs/real_model/results.md` after running it. Real observations alone do not establish forecasting accuracy or transfer to new kitchens.
The original synthetic dataset is preserved. The Streamlit app provides real-data views alongside the explicitly labelled synthetic demonstration.

## Records per kitchen

| Kitchen | Rows |
| --- | ---: |
| Tenerife / El Tope | 232 |
| Tenerife / Tiagaiga | 228 |
| Tenerife / Marte | 206 |
| Santander / El Italiano Trattoria Pizzeria | 197 |
| Portugal / Tivoli Oriente | 183 |
| Tenerife / Nielsen restaurant | 171 |
| Cyprus / Artisan Burger Bar | 164 |
| Cyprus / CU Ledras | 162 |
| Copenhagen / Babette | 156 |
| Cyprus / Taco Bell | 154 |
| Copenhagen / Manon Les Suites | 152 |
| Copenhagen / Axel | 147 |
| Portugal / Hotel 3k | 143 |
| Portugal / Onj São Lázaro | 128 |
| Santander / Dias Desur | 124 |
| Santander / Restaurante Machichaco | 120 |
| Cyprus / Bamboo Restaurant | 116 |
| Cyprus / Ocean Basket | 103 |
| Tenerife / Hotel Eden | 68 |
| Cyprus / Hilton Cyprus | 65 |
| Santander / La Caseta de Bombas | 65 |
| Copenhagen / Steelhouse | 53 |
| kavala / Naftis | 35 |
| kavala / Paraliakon | 27 |
| Portugal / Escolade Hotelaria | 24 |
| kavala / Ta Varelia | 23 |
| Santander / Restaurante Deluz | 19 |
| Copenhagen / 66 | 16 |
| Cyprus / Burger King Cyprus | 14 |
| Tenerife / Botanico | 11 |
| kavala / campingalexandros | 11 |
| kavala / psaraki | 2 |
| kavala / Kynigos | 1 |
| kavala / KyraVangelio | 1 |
