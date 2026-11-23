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

The historical evaluation uses chronological splits and precomputed rows with observed demand lags and realized same-day staffing. It measures one-step-style prediction with refreshed inputs. The historical recursive forecast substitutes predictions for future lag values and uses assumed staffing. Its multi-step accuracy remains unvalidated. The benchmark below fits a separate, direct XGBoost candidate from source.

The [EDA notebook](notebooks/01_exploratory_data_analysis.ipynb) is an optional historical exploration. Use the preparation command for the supported data workflow.

## Rolling-origin forecast benchmark

This is a **retrospective synthetic benchmark**. The October–December 2025 period was examined in the historical experiment. The target is daily incoming requests per region. Each forecast is issued on the last fully observed date and uses arrivals through that date plus known calendar features. Staffing, realized absence, and realized future arrivals are excluded. All four methods forecast 30 days from the same origin and clip negative predictions to zero without rounding.

| Candidate | Method |
|---|---|
| Weekly seasonal naïve | Phase 1 baseline. Repeat the last observed seven days with weekday alignment. |
| Same-weekday average | Average the four latest observations of each weekday. Requires 28 complete daily observations. |
| Calendar ridge | Ridge with alpha 10. Uses region, weekday, month, month-end, the simulator's fixed year-end holiday dates, and overall and regional elapsed-time trends. Refit on observed arrivals at each origin. |
| Direct XGBoost | A single pooled model predicts all 30 leads using the same calendar inputs, lead, most recent matching weekday, mean of the last four matching weekdays, and trailing 7- and 28-day arrival means. Training constructs weekly historical forecast issues with all labels observed by the current origin. Training and inference share a feature builder, and later leads use the same issue-time information. Uses a fixed 160-tree CPU configuration with depth 3, learning rate 0.05, and seed 42. |

Development uses 35 Monday origins from 2025-01-06 through 2025-09-01. The final retrospective evaluation uses nine Monday origins from 2025-10-06 through 2025-12-01. Each origin has at least 365 observed days and all 30 target dates. Development targets end on 2025-10-01 and final targets start on 2025-10-07. Models may refit on arrivals observed since earlier origins within the fixed protocol. Horizon metrics pool leads 1 through H over all regions and origins. Exact-lead metrics are reported separately. Overlapping windows reuse some target dates, so paired origin scores are descriptive and are not independent confidence samples.

The selection rule uses only 14-day development MAE, the default demo horizon. It picks the simplest candidate within the larger of 0.25 requests or 2% of the best MAE, in the order listed above. Direct XGBoost was selected with development MAE **5.92**, ahead of calendar ridge at **6.29** by more than the **0.25** tolerance. The dashboard continues to use weekly seasonal naïve. The benchmark measures retrospective accuracy on generator-defined patterns and does not establish that a model should be deployed.

**Final retrospective MAE** in requests per region-day

| Candidate | 7 days | 14 days | 30 days |
|---|---:|---:|---:|
| Weekly seasonal naïve | 9.11 | 9.50 | 11.38 |
| Same-weekday average | 7.95 | 8.35 | 10.48 |
| Calendar ridge | 7.46 | 7.20 | 7.00 |
| Direct XGBoost | **6.45** | **6.26** | **6.29** |

At 14 days, direct XGBoost beat ridge on eight of nine final origins. Their MAE differed by just 0.02 requests in Asia Pacific. On 116 high-demand forecast rows, direct XGBoost MAE was 7.08 with mean signed bias of −4.79 requests. High demand means actual arrivals met or exceeded the region's historical 90th percentile at the forecast origin. These are forecast rows and may share target dates.

Run from the repository root with Python 3.12. Reports stay under the ignored `reports/` directory by default in this example.

```bash
python -m pip install -r requirements-backtest.txt
python -B -m src.backtest --input data/raw/workforce_demand.csv --output-dir reports/backtest
```

Use `--smoke` for one origin per split. The output includes keyed predictions, a manifest with the input hash and model-selection decision, and CSV summaries by horizon, region, exact lead, origin, and demand level. MAE, RMSE, and signed bias are in requests. WAPE is left blank for groups with zero total actual demand.

The unique key in `predictions.csv` is `forecast_origin, region, model_name, horizon, lead, target_date`. Horizons are cumulative, so leads 1–7 appear in the 7-, 14-, and 30-day views. Filter to `horizon=30` for one row per origin, region, model, lead, and target date.

## Budget-matched staffing replay

For each of the nine final Monday origins, the replay issues a 14-day schedule for each region. Each episode starts with **zero synthetic backlog** under all three policies. Budgets use the generator's nominal roster and permit two fewer or two more scheduled staff per day. The budget, bounds, zero opening balance, and threshold of 20 requests were fixed before evaluating policy outcomes.

| Region | Fixed staff per day | Daily bounds | Staff-days per episode |
|---|---:|---:|---:|
| Asia Pacific | 8 | 6–10 | 112 |
| Europe | 9 | 7–11 | 126 |
| North America | 11 | 9–13 | 154 |

Fixed staffing schedules the nominal roster every day. The seasonal-naïve and direct XGBoost policies both start each day at minimum staffing. Each remaining staff-day goes to the uncapped date with the largest forecast arrivals minus already scheduled capacity, breaking ties by earliest date. This spends the exact regional budget even when all remaining gaps are negative. Schedules are frozen at issuance.

Daily expected capacity is scheduled staff × 0.93 availability × 4 requests per available person-day. Processing is continuous expected throughput, calculated in hundredths of a request. Available work equals opening backlog plus realized arrivals. Processed work is the smaller of available work and capacity. The remainder carries into the next day within that episode. Historical completions and backlog do not enter this recurrence.

| Mean per 14-day region-origin episode | Backlog-days | Paired change vs fixed | Ending backlog | Peak backlog | Days above 20 |
|---|---:|---:|---:|---:|---:|
| Fixed | 993.01 | 0.00 | 100.82 | 122.60 | 11.63 |
| Seasonal naïve | 945.99 | −47.03 | 100.96 | 116.08 | 11.04 |
| Direct XGBoost | 932.11 | −60.90 | 100.21 | 113.90 | 11.15 |

Backlog-days sum the daily ending backlog within an episode. The table averages 27 paired region-origin episodes equally. The windows overlap by seven days and cover 70 distinct dates. These comparisons describe alternative schedules on repeated synthetic arrivals. They cannot establish statistical significance or real operational impact.

XGBoost had both lower retrospective forecast MAE and lower mean backlog-days than seasonal naïve under the tested allocation rule, with slightly more threshold-breach days. Its backlog-days were lower in 14 episodes, higher in 12, and equal in one. The comparison uses synthetic arrivals, zero opening backlog, fixed productivity and absence, overlapping episodes, and one staffing-budget scenario.

The `episode_days` column in `high_arrival_summary.csv` counts 116 high-arrival episode-days per policy, representing 61 distinct region-dates. Demand exceeded the permitted maximum daily capacity on all of them. XGBoost underforecast those episode-days by 4.79 requests on average.

Run the replay from the repository root with the benchmark dependencies installed.

```bash
python -B -m src.staffing_replay --input data/raw/workforce_demand.csv --output-dir reports/staffing_replay
```

The CLI uses the Phase 2 forecast code and fixed configuration at the same final origins. It checks the forecast keys, target dates, actuals, and predicted values. Outputs include daily outcomes, episode metrics and paired differences, region and origin means, overall means, high-arrival diagnostics, and a configuration manifest with data and forecast hashes. Staffing bounds apply to a single interchangeable pool. Shift assignments, skill constraints, and random daily service rates are outside this scenario.

## Tests

```bash
python -m pip install -r requirements-app.txt -r requirements-backtest.txt
python -B -m unittest discover -s tests -v
```

The tests cover preparation, leakage and feature parity, repeatable benchmark output, capacity calculations, and a Streamlit scenario that changes staffing inputs while holding arrivals fixed.
