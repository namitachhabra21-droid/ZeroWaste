# Real-data forecasting workflow

This experiment uses the original [URBAN-WASTE Food Tracking Device export](https://zenodo.org/records/3355300)
by Mattias Eriksson and Samuel Lindgren (2019). The source describes records from hotel/restaurant pilot kitchens.
The unmodified CSV is in `dataset/raw/urban_waste_food_tracking.csv`.

## Run locally

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python audit_real_dataset.py
python -m unittest test_real_pipeline -v
python train_real_model.py
```

The supplied raw file is checked against the publisher's MD5 by the audit.
To download it again, use the CSV download on the linked Zenodo page and retain its original bytes.

## Train interactively in Jupyter / VS Code

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
```

Open `notebooks/real_food_waste_training.ipynb` in VS Code and select the project's `.venv` Python environment as the kernel.
Run the cells from top to bottom with Shift+Enter. Step 5 contains the actual `.fit()` calls for all three ML models.
The notebook includes saved execution outputs so tables, charts and the example prediction are visible when it opens. You can rerun the cells to reproduce them. `python build_real_notebook.py` regenerates an unexecuted copy and clears saved outputs; rerun and save the notebook afterwards to restore them.
For the standalone Jupyter web interface, install `jupyterlab` in the environment and run `python -m jupyter lab`.

The notebook saves its own artifacts to `model/real_notebook_model.pkl` and `model/real_notebook_metrics.json`.
It does not overwrite the CLI-trained artifact. The Streamlit app prefers this saved notebook artifact when available.

## Use the real-data dashboard

```bash
.venv/bin/python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The default **Real kitchen data** mode loads your saved Jupyter artifact.
Choose **Forecast waste**, pick a kitchen and date, then press **Predict next-day waste**.
For an example, use **Copenhagen / Axel** and **2018-11-22**. This uses recorded history through the previous day.
The **Training results** section shows metrics from the selected saved artifact, not hard-coded scores.
If both notebook and CLI artifacts exist, **Training run** lets you choose. Saving the notebook artifact again updates the app on its next rerun/refresh.
Missing artifacts prompt you to complete the notebook's save step; the app does not train a model silently.

## Outputs

- `dataset/processed/urban_waste_clean.csv`: sorted kitchen/date/weight records; missing target values remain missing.
- `model/real_food_waste_model.pkl`: selected forecasting method, fitted model when applicable, schema and metrics. Rebuild locally; ignored by Git.
- `docs/real_model/results.md`: readable protocol, comparisons and limitations.
- `docs/real_model/metrics.json`: exact metrics, date boundaries, coverage, source checksum and software versions.
- `docs/real_model/test_predictions.csv`: predictions and actual values on scored holdout rows.
- `docs/real_model/per_kitchen.csv`: selected-method and previous-day errors by kitchen.

Models: Ridge Regression, Decision Tree, Random Forest. Baselines: previous calendar day, mean of the last seven measured observations, historical kitchen mean.
Selection uses lowest validation MAE among trained ML candidates, selecting Ridge Regression for the course requirement. The seven-observation mean has slightly lower validation MAE and remains a benchmark. The holdout was inspected previously and is not a fresh untouched test in this revision.
The final holdout does not determine model selection. No hyperparameter search is performed.

## Forecast one day

```bash
python predict_real_waste.py --kitchen "Copenhagen / Axel" --date 2018-11-22
```

This is a historical example, not a forecast for today. A forecast requires:

- a kitchen represented in the saved fit;
- a date after the model's train/validation period;
- at least three strictly earlier measurements for that kitchen;
- a measured waste value on the previous calendar day.

Use `--dataset PATH` for a CSV in the original export format containing updated history.
Historical gaps are not treated as zero. The seven-observation average is not necessarily a seven-calendar-day average.
Same-day and future measurements are ignored during inference.

## Unit investigation and scope

The exported header says `Waste weight sum` without a unit. The Zenodo description does not define the unit.
[URBAN-WASTE M20 guidance](https://www.gobiernodecanarias.org/medioambiente/descargas/Cofinanciacion-europea/URBAN_WASTE/Documentos_Generados_Comunidad_Practicas/M20_dispositivo_seguimiento_alimentos.pdf)
mentions kilograms but also permits bin/bag counts and estimated equivalent weights. This contextual guidance does not explicitly establish the exported column's unit.
The pipeline therefore reports source weight units and records the unit as unconfirmed. It does not calculate money or preparation recommendations.

The evaluation simulates daily predictions with newly observed prior-day waste becoming available over time.
It does not simulate forecasting many days ahead without new measurements. Kitchens unseen at fit time are excluded from scoring and rejected by inference.
Results concern covered 2018 services, not current operations or unseen locations. Missing target rows are excluded from scoring, not imputed.

The app shows the real-data views only. The earlier synthetic demo was removed from this project.
