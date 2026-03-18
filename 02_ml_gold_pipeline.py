# Databricks notebook source
# MAGIC %pip install mlflow scikit-learn matplotlib
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error, r2_score)
from sklearn.preprocessing import StandardScaler

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()

# ── Config ──────────────────────────────────────────────
CATALOG = "workspace"
SILVER  = f"{CATALOG}.silver"
GOLD    = f"{CATALOG}.gold"

# MLflow experiment — saves all runs here
mlflow.set_experiment("/btc_linear_regression")

print("✅ Imports done")
print(f"   Silver : {SILVER}")
print(f"   Gold   : {GOLD}")
print(f"   MLflow experiment : /btc_linear_regression")

# COMMAND ----------

def prepare_ml_data():
    print("-- Loading Silver data --")
    pdf = spark.table(f"{SILVER}.btc_features").toPandas()
    pdf = pdf.sort_values("open_time").reset_index(drop=True)
    print(f"   Rows loaded      : {len(pdf):,}")

    # Target: next hour close price
    pdf["target"] = pdf["close"].shift(-1)

    # Feature columns
    feature_cols = [
        "close", "open", "high", "low", "volume",
        "returns", "ma_7", "ma_24", "ma_168",
        "volatility_24", "rsi_14",
        "lag_1", "lag_3", "lag_6", "lag_24",
        "hour", "day_of_week", "month", "price_range"
    ]

    # Drop ALL rows with any NaN in features or target
    cols_to_check = feature_cols + ["target"]
    before = len(pdf)
    pdf = pdf.dropna(subset=cols_to_check).reset_index(drop=True)
    after = len(pdf)
    print(f"   Rows after dropna: {after:,} (dropped {before - after} rows)")

    # Double check - fill any remaining NaN with column median
    for col in feature_cols:
        if pdf[col].isna().any():
            median_val = pdf[col].median()
            pdf[col]   = pdf[col].fillna(median_val)
            print(f"   Filled NaN in {col} with median {median_val:.2f}")

    # Confirm no NaNs remain
    nan_count = pdf[feature_cols].isna().sum().sum()
    print(f"   NaN remaining    : {nan_count} (must be 0)")

    X     = pdf[feature_cols].values
    y     = pdf["target"].values
    dates = pdf["open_time"].values

    # Train/Test split 80/20 - no shuffle, time series!
    split      = int(len(X) * 0.80)
    X_train    = X[:split]
    X_test     = X[split:]
    y_train    = y[:split]
    y_test     = y[split:]
    dates_test = dates[split:]

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"   Features         : {len(feature_cols)}")
    print(f"   Train rows       : {len(X_train):,} (80%)")
    print(f"   Test rows        : {len(X_test):,}  (20%)")
    print(f"✅ Data ready for ML")

    return X_train, X_test, y_train, y_test, dates_test, scaler, feature_cols, pdf

X_train, X_test, y_train, y_test, dates_test, scaler, feature_cols, pdf = prepare_ml_data()

# COMMAND ----------

def train_and_log():
    print("-- Training Linear Regression --")

    with mlflow.start_run(run_name="btc_linear_regression") as run:
        run_id = run.info.run_id

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("   Model trained")

        # Predictions
        y_pred_test  = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Metrics
        rmse  = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae   = mean_absolute_error(y_test, y_pred_test)
        r2    = r2_score(y_test, y_pred_test)
        mape  = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        r2_tr = r2_score(y_train, y_pred_train)

        print(f"   RMSE     : ${rmse:,.2f}")
        print(f"   MAE      : ${mae:,.2f}")
        print(f"   R2       : {r2:.6f}")
        print(f"   MAPE     : {mape:.4f}%")
        print(f"   R2 train : {r2_tr:.6f}")

        # Log params
        mlflow.log_params({
            "model"      : "LinearRegression",
            "features"   : len(feature_cols),
            "train_rows" : len(X_train),
            "test_rows"  : len(X_test),
            "train_pct"  : 0.80,
            "scaler"     : "StandardScaler",
            "target"     : "next_hour_close"
        })

        # Log metrics
        mlflow.log_metrics({
            "rmse"     : round(rmse,  4),
            "mae"      : round(mae,   4),
            "r2"       : round(r2,    6),
            "mape"     : round(mape,  4),
            "r2_train" : round(r2_tr, 6)
        })

        # Log model WITHOUT Unity Catalog registration
        # (avoids the hanging issue on Community Edition)
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_pred_train)
        mlflow.sklearn.log_model(
            sk_model      = model,
            artifact_path = "linear_regression_btc",
            signature     = signature
            # removed registered_model_name to avoid UC hanging
        )
        print("   Model logged to MLflow")

        # Save predictions to Gold
        print("\n-- Saving predictions to Gold --")
        pred_pdf = pd.DataFrame({
            "open_time"  : dates_test,
            "actual"     : y_test,
            "predicted"  : y_pred_test.round(2),
            "error"      : (y_test - y_pred_test).round(2),
            "pct_error"  : ((np.abs(y_test - y_pred_test) / y_test) * 100).round(4),
            "run_id"     : run_id,
            "model"      : "LinearRegression",
            "created_at" : datetime.now(timezone.utc)
        })

        pred_sdf = spark.createDataFrame(pred_pdf)
        (pred_sdf.write
                 .format("delta")
                 .mode("overwrite")
                 .saveAsTable(f"{GOLD}.btc_predictions"))

        gold_count = spark.table(f"{GOLD}.btc_predictions").count()

        # Save metrics to Gold
        metrics_pdf = pd.DataFrame([{
            "run_id"     : run_id,
            "model"      : "LinearRegression",
            "rmse"       : round(rmse,  4),
            "mae"        : round(mae,   4),
            "r2"         : round(r2,    6),
            "mape"       : round(mape,  4),
            "r2_train"   : round(r2_tr, 6),
            "train_rows" : len(X_train),
            "test_rows"  : len(X_test),
            "features"   : len(feature_cols),
            "created_at" : datetime.now(timezone.utc)
        }])

        metrics_sdf = spark.createDataFrame(metrics_pdf)
        (metrics_sdf.write
                    .format("delta")
                    .mode("overwrite")
                    .saveAsTable(f"{GOLD}.btc_model_metrics"))

        print(f"✅ Gold predictions  : {gold_count:,} rows")
        print(f"✅ Gold metrics      : workspace.gold.btc_model_metrics")
        print(f"✅ MLflow run ID     : {run_id}")

        return model, y_pred_test, run_id

model, y_pred_test, run_id = train_and_log()

# COMMAND ----------

def generate_graphs():
    """
    Generate 4 graphs:
    1. Actual vs Predicted price (full test set)
    2. Actual vs Predicted price (last 7 days zoomed)
    3. Prediction error over time
    4. Model metrics bar chart
    """
    # Load from Gold
    pred_pdf = spark.table(f"{GOLD}.btc_predictions").toPandas()
    pred_pdf = pred_pdf.sort_values("open_time").reset_index(drop=True)
    pred_pdf["open_time"] = pd.to_datetime(pred_pdf["open_time"])

    metrics_pdf = spark.table(f"{GOLD}.btc_model_metrics").toPandas()

    fig, axes = plt.subplots(4, 1, figsize=(16, 24))
    fig.suptitle("BTC/USD Price Prediction — Linear Regression\nMedallion Architecture Project",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Graph 1: Full actual vs predicted ─────────────────
    ax1 = axes[0]
    ax1.plot(pred_pdf["open_time"], pred_pdf["actual"],
             color="#2196F3", linewidth=0.8, label="Actual Price", alpha=0.9)
    ax1.plot(pred_pdf["open_time"], pred_pdf["predicted"],
             color="#FF5722", linewidth=0.8, label="Predicted Price",
             alpha=0.9, linestyle="--")
    ax1.set_title("Actual vs Predicted BTC Price (Full Test Set — 20%)", fontsize=13)
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(pred_pdf["open_time"],
                     pred_pdf["actual"], pred_pdf["predicted"],
                     alpha=0.1, color="red", label="Error gap")

    # ── Graph 2: Last 7 days zoomed in ────────────────────
    ax2 = axes[1]
    last_7d = pred_pdf.tail(7 * 24)
    ax2.plot(last_7d["open_time"], last_7d["actual"],
             color="#2196F3", linewidth=1.5, label="Actual", marker="o",
             markersize=2)
    ax2.plot(last_7d["open_time"], last_7d["predicted"],
             color="#FF5722", linewidth=1.5, label="Predicted",
             linestyle="--", marker="s", markersize=2)
    ax2.set_title("Actual vs Predicted — Last 7 Days (Zoomed)", fontsize=13)
    ax2.set_ylabel("Price (USD)")
    ax2.legend(loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.grid(True, alpha=0.3)

    # ── Graph 3: Prediction error over time ───────────────
    ax3 = axes[2]
    ax3.plot(pred_pdf["open_time"], pred_pdf["error"],
             color="#9C27B0", linewidth=0.6, alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=1, linestyle="-")
    ax3.fill_between(pred_pdf["open_time"], pred_pdf["error"], 0,
                     where=(pred_pdf["error"] > 0),
                     color="#4CAF50", alpha=0.3, label="Overestimate")
    ax3.fill_between(pred_pdf["open_time"], pred_pdf["error"], 0,
                     where=(pred_pdf["error"] < 0),
                     color="#F44336", alpha=0.3, label="Underestimate")
    ax3.set_title("Prediction Error Over Time (Actual - Predicted)", fontsize=13)
    ax3.set_ylabel("Error (USD)")
    ax3.legend(loc="upper left")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.tick_params(axis="x", rotation=45)
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.grid(True, alpha=0.3)

    # ── Graph 4: Model metrics bar chart ──────────────────
    ax4 = axes[3]
    metrics = {
        "R2 Score\n(test)"  : float(metrics_pdf["r2"].iloc[0]),
        "R2 Score\n(train)" : float(metrics_pdf["r2_train"].iloc[0]),
        "1 - MAPE %\n(accuracy)" : 1 - float(metrics_pdf["mape"].iloc[0]) / 100
    }
    bars = ax4.bar(metrics.keys(), metrics.values(),
                   color=["#2196F3", "#4CAF50", "#FF9800"],
                   width=0.4, edgecolor="white")
    ax4.set_title("Model Performance Metrics", fontsize=13)
    ax4.set_ylabel("Score")
    ax4.set_ylim(0, 1.1)
    ax4.axhline(y=1.0, color="red", linestyle="--",
                linewidth=1, alpha=0.5, label="Perfect score")
    ax4.legend()
    for bar, val in zip(bars, metrics.values()):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 fontweight="bold", fontsize=11)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save to DBFS
    plot_path = "/tmp/btc_prediction_graph.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.show()
    print(f"✅ Graph saved to : {plot_path}")
    print(f"   Total predictions plotted : {len(pred_pdf):,}")

generate_graphs()

# COMMAND ----------

# ── Job Entry Point ─────────────────────────────────────
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature   
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
CATALOG = "workspace"
SILVER  = f"{CATALOG}.silver"
GOLD    = f"{CATALOG}.gold"
mlflow.set_experiment("/btc_linear_regression")

# Run full pipeline
X_train, X_test, y_train, y_test, dates_test, scaler, feature_cols, pdf = prepare_ml_data()
model, y_pred_test, run_id = train_and_log()
generate_graphs()
print("✅ ML Gold job complete")