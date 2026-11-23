# Workforce Demand Forecasting & Capacity Planning

This synthetic demo forecasts daily incoming workforce operations requests in three regions and converts the forecast into an assumptions-based staffing scenario. Requests include onboarding support, worker extensions, data corrections, and new worker requests. The dashboard shows the historical forecast origin, planning dates, daily expected capacity, and required and additional scheduled staff. It exports the daily plan as CSV.

## Quickstart

Use Python 3.12 and run the commands from the repository root.

```bash
git clone https://github.com/Vai-bhv/workforce-demand-capacity-planning.git
cd workforce-demand-capacity-planning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m src.prepare_data \
  --input data/raw/workforce_demand.csv \
  --output-dir data/processed
streamlit run app/app.py
```

Open http://localhost:8501. The tracked source produces 3,288 regional rows and 3,198 feature rows. Preparation writes the two processed CSVs to the specified directory. Choose a fresh directory to avoid replacing earlier outputs. Set `WORKFORCE_HISTORY_PATH` to the resulting `daily_regional_demand.csv` to launch from a different preparation directory.

### Docker

```bash
docker build -t workforce-capacity-planner .
docker run --rm -p 127.0.0.1:8501:8501 workforce-capacity-planner
```

Docker prepares the data at build time from the tracked raw CSV. The image uses a non-root account and serves the same baseline dashboard. The notebook, locally generated files, and saved model are unnecessary for this build. The health check probes the Streamlit server. The planner smoke test below checks the calculations through the UI.

## Forecast and capacity assumptions

The forecast starts the day after the last observation for the selected region. It repeats the last seven complete, observed daily values in the same weekday order for the selected horizon. With the tracked dataset, the origin is 2025-12-31. The default horizon is 14 days. The forecast only uses observations up to the origin, and the scenario inputs affect capacity independently of arrivals.

| Quantity | Calculation |
|---|---|
| Expected available staff | Scheduled staff × (1 − expected absence rate) |
| Expected processing capacity per day | Expected available staff × requests per available person-day |
| Minimum required scheduled staff | Daily forecast arrivals divided by expected capacity per scheduled person, rounded up |
| Additional scheduled staff | Required scheduled staff minus current scheduled staff, floored at zero |

The scenario assumes one interchangeable staff pool, a constant absence rate and productivity, and a single daily count of arriving requests. Expected availability may be fractional. Zero scheduled staff gives zero current capacity; a finite headcount requirement still exists when each additional scheduled person contributes positive capacity. With 100% absence or zero productivity, positive demand has no finite headcount requirement. The dashboard covers incoming work under a point forecast. Opening backlog, skill constraints, service times, and SLAs are outside the calculation.

## Synthetic data and known discrepancy

[The tracked raw CSV](data/raw/workforce_demand.csv) covers 2023–2025 with one row per date, region, and request type. [The generator](src/generate_data.py) defines the calendar effects, including strong weekdays and weekends, a shared holiday period, regional growth, and occasional random spikes. These patterns are features of the simulator. The data contains no observed enterprise staffing or service outcomes.

[Preparation](src/prepare_data.py) validates the required schema, unique keys, a complete daily panel across the observed regions and request types, nonnegative integer counts, and agreement between the repeated region-day attributes. It sums type-level arrivals and completions and carries forward the recorded regional attributes. [Feature engineering](src/features.py) uses strictly earlier, explicitly grouped observations and drops the first 30 days in each region from the feature export.

On **1,058 region-days with a previous observed backlog**, recorded completions disagree with previous backlog plus arrivals minus ending backlog. This check excludes the first observed day in each region because its initial backlog is absent from the CSV. The generator updates backlog from the simulated regional completion count, then allocates and rounds completions using the mix of new arrivals. This can lose completions when arrivals are zero or change their sum through rounding. Preparation reports the discrepancy and preserves both fields. The demo capacity calculation uses neither field.

## Historical XGBoost experiment

The original [training](src/train.py), [tuning](src/tune_xgboost.py), [evaluation](src/evaluate.py), [recursive forecast](src/forecast.py), and [saved artifact](models/xgboost_tuned.pkl) are retained. Use the original [experiment dependencies](requirements.txt) in a separate Python 3.12 environment to inspect or rerun those workflows. The legacy scripts write to repository-relative output locations, including the tracked tuned artifact during tuning. Run experiments in a scratch checkout and prepare the inputs there first. The legacy prediction and evaluation paths deserialize pickle files, so use only trusted artifacts.

The historical evaluation uses chronological date splits and precomputed rows containing observed demand lags and realized same-day staffing. It describes one-step-style prediction with refreshed inputs, including staffing that may be unavailable at a planning origin. The recursive forecast substitutes its own future lag values and uses assumed staffing. Its multi-step forecast accuracy remains unvalidated. The default demo uses the weekly baseline because its information at the forecast origin is explicit. Comparing it with XGBoost requires a matched-horizon rolling-origin backtest.

The [EDA notebook](notebooks/01_exploratory_data_analysis.ipynb) is an optional historical exploration. Use the preparation command for the supported data workflow.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

The tests cover deterministic preparation from a source-only copy, regional aggregation and validation, forecast origin and weekly repetition, capacity calculations, and a Streamlit scenario that changes staffing inputs while holding arrivals fixed.
