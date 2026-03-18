# Databricks notebook source
# MAGIC %pip install yfinance==0.2.40
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()

# Add this line to Cell 2, after spark = SparkSession...
spark.sql("USE CATALOG workspace")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.gold")

print(f"✅ Current catalog : {spark.sql('SELECT current_catalog()').collect()[0][0]}")
print(f"✅ Schemas ready")

# ── Catalog config ──────────────────────────────────────────
CATALOG = "workspace"   # Unity Catalog name (you confirmed this earlier)
BRONZE  = f"{CATALOG}.bronze"
SILVER  = f"{CATALOG}.silver"

print(f"✅ Config ready")
print(f"   Bronze : {BRONZE}")
print(f"   Silver : {SILVER}")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check what catalog we're actually in
# MAGIC SELECT current_catalog(), current_schema();

# COMMAND ----------

#get_initial_data()
def generate_btc_synthetic():
    """
    Geometric Brownian Motion — industry standard for
    simulating asset price paths.
    Calibrated to real BTC parameters.
    """
    np.random.seed(42)
    mu, sigma = 0.60, 0.80
    dt        = 1 / 8760
    hours     = 730 * 24       # 17,520 rows = 2 years hourly
    S0        = 45_000.0

    Z      = np.random.standard_normal(hours)
    growth = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    closes = S0 * np.cumprod(growth)

    rows       = []
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    for i in range(hours):
        close  = closes[i]
        noise  = close * np.random.uniform(0.001, 0.015)
        open_  = closes[i-1] if i > 0 else close * np.random.uniform(0.998, 1.002)
        high   = max(open_, close) + noise
        low    = min(open_, close) - noise
        volume = np.random.lognormal(mean=8.0, sigma=1.2)

        rows.append({
            "open_time" : start_time + timedelta(hours=i),
            "open"      : round(open_,  2),
            "high"      : round(high,   2),
            "low"       : round(low,    2),
            "close"     : round(close,  2),
            "volume"    : round(volume, 4)
        })

    return pd.DataFrame(rows)


def get_initial_data():
    """
    Fetch raw BTC/USD hourly data and save to Bronze schema.
    This is the RAW layer — no transformations, just as-is.
    """
    print("── Bronze: generating BTC data ──────────────────────")
    pdf = generate_btc_synthetic()
    print(f"   Pandas rows      : {len(pdf):,}")
    print(f"   Date range       : {pdf['open_time'].min()} → {pdf['open_time'].max()}")
    print(f"   Price range      : ${pdf['close'].min():,.0f} → ${pdf['close'].max():,.0f}")

    # Convert to Spark DataFrame
    schema = StructType([
        StructField("open_time", TimestampType(), False),
        StructField("open",      DoubleType(),    False),
        StructField("high",      DoubleType(),    False),
        StructField("low",       DoubleType(),    False),
        StructField("close",     DoubleType(),    False),
        StructField("volume",    DoubleType(),    False),
    ])

    sdf = spark.createDataFrame(pdf, schema=schema)

    # Save to Bronze — overwrite on initial load
    (sdf.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{BRONZE}.btc_raw"))

    count = spark.table(f"{BRONZE}.btc_raw").count()
    print(f"\n✅ Bronze table saved : {BRONZE}.btc_raw")
    print(f"   Row count          : {count:,}")


# ── Run it ──
get_initial_data()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     MIN(open_time) AS earliest,
# MAGIC     MAX(open_time) AS latest,
# MAGIC     COUNT(*)       AS total_rows,
# MAGIC     ROUND(AVG(close), 2) AS avg_price,
# MAGIC     ROUND(MIN(close), 2) AS min_price,
# MAGIC     ROUND(MAX(close), 2) AS max_price
# MAGIC FROM workspace.bronze.btc_raw;

# COMMAND ----------

def process_to_silver():
    """
    Read from Bronze, do EDA + cleaning + feature engineering,
    save to Silver schema.
    
    Features added:
    - returns       : hourly log return
    - ma_7          : 7-hour moving average
    - ma_24         : 24-hour moving average  
    - ma_168        : 7-day moving average
    - volatility_24 : 24-hour rolling volatility
    - rsi_14        : 14-period Relative Strength Index
    - lag_1/lag_3/lag_6/lag_24 : lagged close prices
    - hour/day_of_week/month   : time-based features
    - price_range   : high - low per candle
    """
    from pyspark.sql.window import Window

    print("── Silver: reading from Bronze ───────────────────────")
    sdf = spark.table(f"{BRONZE}.btc_raw")
    total = sdf.count()
    print(f"   Rows from Bronze : {total:,}")

    # ── 1. EDA checks ──────────────────────────────────────
    print("\n── EDA ───────────────────────────────────────────────")
    nulls = {c: sdf.filter(F.col(c).isNull()).count() for c in sdf.columns}
    print(f"   Null counts      : {nulls}")

    dupes = total - sdf.dropDuplicates(["open_time"]).count()
    print(f"   Duplicate rows   : {dupes}")

    price_stats = sdf.select(
        F.round(F.mean("close"),   2).alias("mean"),
        F.round(F.stddev("close"), 2).alias("stddev"),
        F.round(F.min("close"),    2).alias("min"),
        F.round(F.max("close"),    2).alias("max")
    ).collect()[0]
    print(f"   Price stats      : mean=${price_stats['mean']:,} "
          f"| std=${price_stats['stddev']:,} "
          f"| min=${price_stats['min']:,} "
          f"| max=${price_stats['max']:,}")

    # ── 2. Clean ───────────────────────────────────────────
    print("\n── Cleaning ──────────────────────────────────────────")
    sdf = (sdf
           .dropDuplicates(["open_time"])
           .filter(F.col("close") > 0)
           .filter(F.col("volume") >= 0)
           .orderBy("open_time"))
    print(f"   Rows after clean : {sdf.count():,}")

    # ── 3. Feature Engineering ─────────────────────────────
    print("\n── Feature Engineering ───────────────────────────────")

    # Window specs
    w_24  = Window.orderBy("open_time").rowsBetween(-23,  0)
    w_7   = Window.orderBy("open_time").rowsBetween(-6,   0)
    w_168 = Window.orderBy("open_time").rowsBetween(-167, 0)
    w_lag = Window.orderBy("open_time")

    sdf = (sdf
        # Time features
        .withColumn("hour",        F.hour("open_time"))
        .withColumn("day_of_week", F.dayofweek("open_time"))
        .withColumn("month",       F.month("open_time"))

        # Price range
        .withColumn("price_range", F.round(F.col("high") - F.col("low"), 2))

        # Log return
        .withColumn("prev_close",  F.lag("close", 1).over(w_lag))
        .withColumn("returns",     F.round(F.log(F.col("close") / F.col("prev_close")), 6))

        # Moving averages
        .withColumn("ma_7",        F.round(F.avg("close").over(w_7),   2))
        .withColumn("ma_24",       F.round(F.avg("close").over(w_24),  2))
        .withColumn("ma_168",      F.round(F.avg("close").over(w_168), 2))

        # Volatility
        .withColumn("volatility_24", F.round(F.stddev("close").over(w_24), 2))

        # Lag features
        .withColumn("lag_1",  F.lag("close",  1).over(w_lag))
        .withColumn("lag_3",  F.lag("close",  3).over(w_lag))
        .withColumn("lag_6",  F.lag("close",  6).over(w_lag))
        .withColumn("lag_24", F.lag("close", 24).over(w_lag))

        # Drop helper column
        .drop("prev_close")

        # Drop rows with nulls from lags/rolling windows
        .filter(F.col("lag_24").isNotNull())
        .filter(F.col("returns").isNotNull())
        .filter(F.col("volatility_24").isNotNull())
    )

    # ── RSI (14 period) ────────────────────────────────────
    # Compute in pandas (easier for RSI), then convert back
    pdf = sdf.toPandas()
    pdf = pdf.sort_values("open_time").reset_index(drop=True)

    delta     = pdf["close"].diff()
    gain      = delta.clip(lower=0)
    loss      = (-delta).clip(lower=0)
    avg_gain  = gain.ewm(com=13, adjust=False).mean()
    avg_loss  = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, 1e-10)
    pdf["rsi_14"] = (100 - (100 / (1 + rs))).round(2)

    sdf = spark.createDataFrame(pdf)

    # ── 4. Save to Silver ──────────────────────────────────
    (sdf.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(f"{SILVER}.btc_features"))

    count = spark.table(f"{SILVER}.btc_features").count()
    cols  = spark.table(f"{SILVER}.btc_features").columns

    print(f"\n✅ Silver table saved : {SILVER}.btc_features")
    print(f"   Row count          : {count:,}")
    print(f"   Columns ({len(cols)})       : {cols}")


# ── Run it ──
process_to_silver()

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC     MIN(open_time)        AS earliest,
# MAGIC     MAX(open_time)        AS latest,
# MAGIC     COUNT(*)              AS total_rows,
# MAGIC     ROUND(AVG(close), 2)  AS avg_close,
# MAGIC     ROUND(AVG(rsi_14), 2) AS avg_rsi,
# MAGIC     ROUND(AVG(ma_24),  2) AS avg_ma24,
# MAGIC     ROUND(AVG(returns), 6) AS avg_return
# MAGIC FROM workspace.silver.btc_features;

# COMMAND ----------

#append_new_hour()
def append_new_hour():
    """
    Simulates fetching the latest 1-hour BTC candle
    and appends it to both Bronze and Silver tables.
    
    In production this would call a live API.
    Here we generate one new GBM candle from the last known price.
    """
    from pyspark.sql.window import Window

    print("── Append New Hour ───────────────────────────────────")

    # ── Step 1: Get last known price from Bronze ───────────
    last_row = (spark.table(f"{BRONZE}.btc_raw")
                     .orderBy(F.col("open_time").desc())
                     .limit(1)
                     .collect()[0])

    last_price = last_row["close"]
    last_time  = last_row["open_time"]
    new_time   = last_time + timedelta(hours=1)

    print(f"   Last candle time  : {last_time}")
    print(f"   Last close price  : ${last_price:,.2f}")
    print(f"   New candle time   : {new_time}")

    # ── Step 2: Generate 1 new GBM candle ─────────────────
    np.random.seed(None)   # fresh randomness each run
    mu, sigma = 0.60, 0.80
    dt        = 1 / 8760

    Z         = np.random.standard_normal()
    new_close = last_price * np.exp((mu - 0.5 * sigma**2) * dt
                                     + sigma * np.sqrt(dt) * Z)
    noise     = new_close * np.random.uniform(0.001, 0.015)
    new_row   = {
        "open_time" : new_time,
        "open"      : round(last_price, 2),
        "high"      : round(max(last_price, new_close) + noise, 2),
        "low"       : round(min(last_price, new_close) - noise, 2),
        "close"     : round(new_close, 2),
        "volume"    : round(np.random.lognormal(mean=8.0, sigma=1.2), 4)
    }
    print(f"   New close price   : ${new_close:,.2f}")

    # ── Step 3: Append to Bronze ───────────────────────────
    schema = StructType([
        StructField("open_time", TimestampType(), False),
        StructField("open",      DoubleType(),    False),
        StructField("high",      DoubleType(),    False),
        StructField("low",       DoubleType(),    False),
        StructField("close",     DoubleType(),    False),
        StructField("volume",    DoubleType(),    False),
    ])

    new_pdf = pd.DataFrame([new_row])
    new_sdf = spark.createDataFrame(new_pdf, schema=schema)

    (new_sdf.write
            .format("delta")
            .mode("append")
            .saveAsTable(f"{BRONZE}.btc_raw"))

    bronze_count = spark.table(f"{BRONZE}.btc_raw").count()
    print(f"\n✅ Bronze updated    : {bronze_count:,} rows")

    # ── Step 4: Compute Silver features for new row ────────
    # Pull last 168 rows from bronze for window calculations
    recent_pdf = (spark.table(f"{BRONZE}.btc_raw")
                       .orderBy(F.col("open_time").desc())
                       .limit(168)
                       .toPandas()
                       .sort_values("open_time")
                       .reset_index(drop=True))

    # Compute features on last row using recent history
    delta    = recent_pdf["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    recent_pdf["rsi_14"] = (100 - (100 / (1 + rs))).round(2)

    # Only take the newest row
    new_silver = recent_pdf.iloc[[-1]].copy()
    new_silver["hour"]          = new_silver["open_time"].dt.hour
    new_silver["day_of_week"]   = new_silver["open_time"].dt.dayofweek
    new_silver["month"]         = new_silver["open_time"].dt.month
    new_silver["price_range"]   = (new_silver["high"] - new_silver["low"]).round(2)
    new_silver["returns"]       = (np.log(recent_pdf["close"].iloc[-1] /
                                          recent_pdf["close"].iloc[-2])).round(6)
    new_silver["ma_7"]          = recent_pdf["close"].tail(7).mean().round(2)
    new_silver["ma_24"]         = recent_pdf["close"].tail(24).mean().round(2)
    new_silver["ma_168"]        = recent_pdf["close"].tail(168).mean().round(2)
    new_silver["volatility_24"] = recent_pdf["close"].tail(24).std().round(2)
    new_silver["lag_1"]         = recent_pdf["close"].iloc[-2]
    new_silver["lag_3"]         = recent_pdf["close"].iloc[-4]
    new_silver["lag_6"]         = recent_pdf["close"].iloc[-7]
    new_silver["lag_24"]        = recent_pdf["close"].iloc[-25] \
                                  if len(recent_pdf) >= 25 \
                                  else recent_pdf["close"].iloc[0]

    # Append to Silver
    new_silver_sdf = spark.createDataFrame(new_silver)
    (new_silver_sdf.write
                   .format("delta")
                   .mode("append")
                   .saveAsTable(f"{SILVER}.btc_features"))

    silver_count = spark.table(f"{SILVER}.btc_features").count()
    print(f"✅ Silver updated    : {silver_count:,} rows")
    print(f"\n── New candle summary ────────────────────────────────")
    print(f"   open_time : {new_row['open_time']}")
    print(f"   open      : ${new_row['open']:>12,.2f}")
    print(f"   high      : ${new_row['high']:>12,.2f}")
    print(f"   low       : ${new_row['low']:>12,.2f}")
    print(f"   close     : ${new_row['close']:>12,.2f}")
    print(f"   volume    : {new_row['volume']:>12,.4f}")
    print(f"   rsi_14    : {new_silver['rsi_14'].values[0]:>12.2f}")


# ── Run it once to test ──
append_new_hour()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Check last 5 rows to confirm new candle appended
# MAGIC SELECT open_time, open, high, low, close, volume, rsi_14, ma_24
# MAGIC FROM   workspace.silver.btc_features
# MAGIC ORDER  BY open_time DESC
# MAGIC LIMIT  5;