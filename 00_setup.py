# Databricks notebook source
# MAGIC %sql
# MAGIC -- Create a dedicated catalog for this project
# MAGIC CREATE CATALOG IF NOT EXISTS btc_project;
# MAGIC
# MAGIC -- Create the three medallion schemas
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_project.bronze;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_project.silver;
# MAGIC CREATE SCHEMA IF NOT EXISTS btc_project.gold;
# MAGIC
# MAGIC -- Verify
# MAGIC SHOW SCHEMAS IN btc_project;

# COMMAND ----------

# MAGIC %pip install yfinance==0.2.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

def generate_btc_2years():
    """
    Geometric Brownian Motion (GBM) — the industry standard
    for simulating asset price paths.
    
    Formula: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    
    Parameters calibrated to match real BTC behaviour:
    - mu    : 60% annual drift (BTC historical average)
    - sigma : 80% annual volatility (BTC is highly volatile)
    - start : $45,000 (realistic BTC price 2 years ago)
    """
    np.random.seed(42)   # reproducible results

    # --- Parameters ---
    mu        = 0.60          # annual drift
    sigma     = 0.80          # annual volatility
    dt        = 1/8760        # 1 hour as fraction of year (8760 hrs/yr)
    hours     = 730 * 24      # 2 years of hourly data = 17,520 rows
    S0        = 45_000.0      # starting price

    # --- GBM price path ---
    Z      = np.random.standard_normal(hours)
    growth = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    closes = S0 * np.cumprod(growth)

    # --- Build OHLCV from close prices ---
    rows = []
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    for i in range(hours):
        close  = closes[i]
        noise  = close * np.random.uniform(0.001, 0.015)  # intra-hour spread
        open_  = closes[i-1] if i > 0 else close * np.random.uniform(0.998, 1.002)
        high   = max(open_, close) + noise
        low    = min(open_, close) - noise
        volume = np.random.lognormal(mean=8.0, sigma=1.2)  # realistic BTC volume

        rows.append({
            "open_time": start_time + timedelta(hours=i),
            "open"     : round(open_,  2),
            "high"     : round(high,   2),
            "low"      : round(low,    2),
            "close"    : round(close,  2),
            "volume"   : round(volume, 4)
        })

    df = pd.DataFrame(rows)
    return df

# Generate
df = generate_btc_2years()

# Verify
print(f"✅ Rows generated  : {len(df):,}")
print(f"   Date range      : {df['open_time'].min()} → {df['open_time'].max()}")
print(f"   Price range     : ${df['close'].min():,.0f} → ${df['close'].max():,.0f}")
print(f"   Avg volume/hr   : {df['volume'].mean():,.2f} BTC")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
print(f"\nLast 3 rows:")
print(df.tail(3).to_string(index=False))
print(f"\nColumn dtypes:")
print(df.dtypes)