# BTC Price Prediction — Databricks Medallion Architecture

> End-to-end Data Engineering + ML pipeline on Databricks Community Edition  
> Predicts BTC/USD hourly prices using Linear Regression with **99.8% R² accuracy**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Delta Tables](#delta-tables)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Setup Instructions](#setup-instructions)
- [Running Each Notebook](#running-each-notebook)
- [Databricks Job Schedules](#databricks-job-schedules)
- [Dashboard](#dashboard)
- [Known Constraints](#known-constraints)

---

## Project Overview

A complete, production-style data pipeline built entirely on **Databricks Community Edition (Free Tier)**. It ingests synthetic BTC/USD OHLCV data generated via Geometric Brownian Motion, processes it through the Medallion Architecture (Bronze → Silver → Gold), trains a Linear Regression model tracked with MLflow, and serves predictions on a Databricks Native Dashboard.

### Why Synthetic Data?
Real-time BTC sources are unavailable on Community Edition:
- `yfinance` — Yahoo Finance rate-limits shared Databricks IPs
- Binance API — geo-blocked in India
- CoinGecko free tier — restricts historical data range

**Solution:** Geometric Brownian Motion (GBM) — the industry-standard stochastic process used in quantitative finance for price simulation.

Parameters used: `mu=0.60, sigma=0.80, S0=45000, seed=42`

---

## Architecture

```
+----------------------------------------------------------+
|                    DATA SOURCE                           |
|        GBM Synthetic BTC/USD (2 years hourly)            |
+-------------------------+--------------------------------+
                          |
                          v
+----------------------------------------------------------+
|                  BRONZE LAYER                            |
|            workspace.bronze.btc_raw                      |
|       Raw OHLCV Delta Table  |  17,520 rows              |
|   Columns: open_time, open, high, low, close, volume     |
+-------------------------+--------------------------------+
                          |
              EDA + Cleaning + Feature Engineering
                          |
                          v
+----------------------------------------------------------+
|                  SILVER LAYER                            |
|           workspace.silver.btc_features                  |
|       Enriched Delta Table  |  17,497 rows               |
|    19 feature columns (time, price, MA, RSI, lags)       |
+-------------------------+--------------------------------+
                          |
               ML Training + Predictions
                          |
                          v
+----------------------------------------------------------+
|                   GOLD LAYER                             |
|      workspace.gold.btc_predictions  (3,499 rows)        |
|      workspace.gold.btc_model_metrics                    |
|      MLflow Experiment: /btc_linear_regression           |
+-------------------------+--------------------------------+
                          |
                          v
+----------------------------------------------------------+
|               SERVING / DASHBOARD                        |
|      Databricks Native Dashboard: BTC Price Dashboard    |
|      4 widgets: Actual vs Predicted, RSI, Volume, Metrics|
+----------------------------------------------------------+
```

---

## Tech Stack

| Component        | Technology                            |
|------------------|---------------------------------------|
| Platform         | Databricks Community Edition          |
| Runtime          | DBR 15.x, Python 3.12                 |
| Catalog          | Unity Catalog (`workspace`)           |
| Storage Format   | Delta Lake                            |
| ML Tracking      | MLflow (built-in Databricks)          |
| ML Model         | scikit-learn LinearRegression         |
| Orchestration    | Databricks Jobs (Quartz cron)         |
| Visualization    | matplotlib + Databricks Native Dashboard |
| Data Simulation  | Geometric Brownian Motion (NumPy)     |

---

## Project Structure

```
btc_price_prediction/
├── 00_setup.py                   # Schema creation + environment setup
├── 01_bronze_silver_pipeline.py  # Data ingestion + feature engineering
├── 02_ml_gold_pipeline.py        # ML training + MLflow logging + predictions
├── 03_dashboard.py               # Dashboard chart generation
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Delta Tables

### `workspace.bronze.btc_raw`
Raw OHLCV data — append-only source of truth.

| Column    | Type      | Description                  |
|-----------|-----------|------------------------------|
| open_time | timestamp | Candle open time (hourly)    |
| open      | double    | Opening price (USD)          |
| high      | double    | Highest price in period      |
| low       | double    | Lowest price in period       |
| close     | double    | Closing price (USD)          |
| volume    | double    | Traded volume                |

---

### `workspace.silver.btc_features`
Cleaned and feature-enriched table. 17,497 rows (23 rows dropped during cleaning).

| Column       | Type      | Description                            |
|--------------|-----------|----------------------------------------|
| open_time    | timestamp | Candle timestamp                       |
| open/high/low/close/volume | double | OHLCV raw values          |
| returns      | double    | Log return: ln(close/prev_close)       |
| ma_7         | double    | 7-period moving average                |
| ma_24        | double    | 24-period moving average               |
| ma_168       | double    | 168-period (7-day) moving average      |
| volatility_24| double    | 24-period rolling std of returns       |
| rsi_14       | double    | 14-period RSI via exponential smoothing|
| lag_1        | double    | Close price lagged 1 period            |
| lag_3        | double    | Close price lagged 3 periods           |
| lag_6        | double    | Close price lagged 6 periods           |
| lag_24       | double    | Close price lagged 24 periods          |
| hour         | int       | Hour of day (0-23)                     |
| day_of_week  | int       | Day of week (0=Mon, 6=Sun)             |
| month        | int       | Month (1-12)                           |
| price_range  | double    | high - low for the candle              |

---

### `workspace.gold.btc_predictions`
Model output. 3,499 rows (test set predictions).

| Column    | Type      | Description                    |
|-----------|-----------|--------------------------------|
| open_time | timestamp | Prediction timestamp           |
| actual    | double    | Actual BTC close price         |
| predicted | double    | Model predicted price          |
| error     | double    | actual - predicted             |
| pct_error | double    | Percentage error               |
| run_id    | string    | MLflow run ID                  |
| model     | string    | Model name                     |
| created_at| timestamp | Prediction creation time       |

---

### `workspace.gold.btc_model_metrics`
One row per training run.

| Column     | Type   | Description                      |
|------------|--------|----------------------------------|
| run_id     | string | MLflow run ID                    |
| model      | string | Model name                       |
| rmse       | double | Root Mean Squared Error          |
| mae        | double | Mean Absolute Error              |
| r2         | double | R2 score (test set)              |
| mape       | double | Mean Absolute Percentage Error   |
| r2_train   | double | R2 score (train set)             |
| train_rows | int    | Training set size                |
| test_rows  | int    | Test set size                    |
| features   | string | Comma-separated feature list     |
| created_at | timestamp | Run timestamp                 |

---

## Feature Engineering

19 features engineered in the Silver layer:

| Category        | Features                                      |
|-----------------|-----------------------------------------------|
| Time features   | `hour`, `day_of_week`, `month`                |
| Price features  | `price_range`, `returns` (log return)         |
| Moving averages | `ma_7`, `ma_24`, `ma_168`                     |
| Volatility      | `volatility_24`                               |
| Momentum        | `rsi_14` (14-period, computed via pandas ewm) |
| Lag features    | `lag_1`, `lag_3`, `lag_6`, `lag_24`           |

---

## Model Performance

| Metric        | Value        |
|---------------|--------------|
| R2 (Test)     | **0.998371** |
| R2 (Train)    | 0.999419     |
| RMSE          | $848.63      |
| MAE           | $661.98      |
| MAPE          | 0.6858%      |
| Train rows    | 13,996       |
| Test rows     | 3,499        |
| Features used | 19           |
| Train/Test split | 80/20 (time-ordered, no shuffle) |
| Scaler        | StandardScaler |

> The high R2 is expected: lag features (lag_1, lag_3, lag_6, lag_24) encode recent price history, which is highly predictive for short-horizon financial time series.

---

## Setup Instructions

### Prerequisites
- Databricks Community Edition account ([sign up free](https://community.cloud.databricks.com))
- Python 3.12 (DBR 15.x)
- Unity Catalog enabled with catalog name: `workspace`

### Step 1 — Run `00_setup.py`
This notebook:
- Installs `yfinance` (included even though it's blocked, for documentation purposes)
- Creates catalog schemas: `workspace.bronze`, `workspace.silver`, `workspace.gold`
- Verifies Unity Catalog connection

### Step 2 — Run `01_bronze_silver_pipeline.py`
This notebook:
- Generates 2 years of hourly synthetic BTC data (17,520 rows) using GBM
- Saves raw data to `workspace.bronze.btc_raw`
- Cleans, enriches, and saves 17,497 rows to `workspace.silver.btc_features`

### Step 3 — Run `02_ml_gold_pipeline.py`
This notebook:
- Loads silver table and engineers target variable (next-hour close price)
- Trains LinearRegression with StandardScaler
- Logs everything to MLflow experiment `/btc_linear_regression`
- Saves predictions to `workspace.gold.btc_predictions`
- Saves metrics to `workspace.gold.btc_model_metrics`

### Step 4 — Run `03_dashboard.py`
This notebook:
- Reads from silver and gold tables
- Generates matplotlib dashboard saved to `/tmp/btc_dashboard.png`

---

## Running Each Notebook

| Notebook | Purpose | Run Order | Estimated Time |
|----------|---------|-----------|----------------|
| `00_setup.py` | Schema setup | 1st | ~1 min |
| `01_bronze_silver_pipeline.py` | Data ingestion + features | 2nd | ~3 min |
| `02_ml_gold_pipeline.py` | ML training + predictions | 3rd | ~2 min |
| `03_dashboard.py` | Chart generation | 4th | ~1 min |

Run notebooks top-to-bottom (Cmd+Shift+Enter) or use **Run All**.

---

## Databricks Job Schedules

| Job Name            | Notebook                      | Schedule (Quartz)  | Description              |
|---------------------|-------------------------------|--------------------|--------------------------|
| `btc_append_hourly` | `01_bronze_silver_pipeline.py`| `0 0 * * * ?`      | Appends 1 new hourly candle |
| `btc_ml_gold_job`   | `02_ml_gold_pipeline.py`      | `0 6 * * * ?`      | Daily retraining at 6am IST |
| `btc_dashboard_job` | `03_dashboard.py`             | `0 0 * * * ?`      | Refreshes dashboard hourly |

> Quartz cron format uses 6 fields: `seconds minutes hours day-of-month month day-of-week`

---

## Dashboard

**Databricks Native Dashboard — "BTC Price Dashboard"**

Page: *BTC Price Analysis* | Status: Published

| Widget | Type | Dataset |
|--------|------|---------|
| Actual vs Predicted BTC Price | Line chart | `gold.btc_predictions` |
| RSI Indicator 14 Period | Line chart | `silver.btc_features` |
| BTC Trading Volume | Bar chart | `silver.btc_features` |
| Model Performance Metrics | Table | `gold.btc_model_metrics` |

---

## Known Constraints (Community Edition)

| Constraint | Reason | Solution |
|------------|--------|----------|
| Save plots to `/tmp/` not `/dbfs/tmp/` | I/O error on Community Edition | Use `/tmp/` path |
| No `registered_model_name` in `mlflow.sklearn.log_model` | Unity Catalog hangs on Community Edition | Omit the parameter |
| `infer_signature` import inside function | Import fails in job context if at module level | Import locally |
| Quartz cron is 6 fields | Databricks uses Quartz, not Unix cron | e.g. `0 0 * * * ?` |
| `matplotlib.use('Agg')` required in job entry point | No display server in job context | Set backend explicitly |
| No unicode arrow characters in code | Causes SyntaxError in Databricks | Use ASCII alternatives |
| `yfinance` blocked | Yahoo Finance rate-limits shared IPs | GBM synthetic data |
| Binance geo-blocked | India network restriction | GBM synthetic data |
| CoinGecko free tier limited | Historical data range restriction | GBM synthetic data |

---

## License
 
MIT License — free to use, modify, and distribute.

---

*Built with Databricks Community Edition | MLflow | Delta Lake | scikit-learn*
