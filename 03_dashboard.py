# Databricks notebook source
# IMPORTANT: Use non-interactive backend (required for Databricks jobs)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql.functions import col
from datetime import datetime, timedelta
import time

print("Setup complete")

# COMMAND ----------

# Load Silver table (features)
silver_df = spark.table("workspace.silver.btc_features").toPandas()

# Load Gold predictions
gold_pred_df = spark.table("workspace.gold.btc_predictions").toPandas()

# Load Gold metrics
gold_metrics_df = spark.table("workspace.gold.btc_model_metrics").toPandas()

# Convert time columns to datetime
silver_df['open_time'] = pd.to_datetime(silver_df['open_time'])
gold_pred_df['open_time'] = pd.to_datetime(gold_pred_df['open_time'])

# Sort data (important for time series)
silver_df = silver_df.sort_values('open_time')
gold_pred_df = gold_pred_df.sort_values('open_time')

print("Data Loaded")
print("Silver shape:", silver_df.shape)
print("Predictions shape:", gold_pred_df.shape)
print("Metrics shape:", gold_metrics_df.shape)

# COMMAND ----------

# Get latest timestamp from silver data
latest_time = silver_df['open_time'].max()

# Calculate 30 days back
start_time = latest_time - timedelta(days=30)

# Filter last 30 days
silver_30d = silver_df[silver_df['open_time'] >= start_time]
gold_30d = gold_pred_df[gold_pred_df['open_time'] >= start_time]

print("Filtered Data (Last 30 Days)")
print("Silver rows:", silver_30d.shape)
print("Pred rows:", gold_30d.shape)

# COMMAND ----------

# Create figure
plt.figure(figsize=(14, 6))

# Plot Actual prices
plt.plot(
    silver_30d['open_time'],
    silver_30d['close'],
    label='Actual Price',
    color='blue'
)

# Plot Predicted prices
plt.plot(
    gold_30d['open_time'],
    gold_30d['predicted'],
    label='Predicted Price',
    color='orange'
)

# Styling
plt.title("BTC/USD Actual vs Predicted (Last 30 Days)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

# Save image
plt.savefig("/tmp/btc_actual_vs_predicted.png")

# Close to avoid memory issues in jobs
plt.close()

print("Chart saved: /tmp/btc_actual_vs_predicted.png")

# COMMAND ----------

# Create RSI plot
plt.figure(figsize=(14, 4))

# Plot RSI line
plt.plot(
    silver_30d['open_time'],
    silver_30d['rsi_14'],
    label='RSI (14)',
    color='purple'
)

# Overbought / Oversold lines
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')

# Fill zones
plt.fill_between(
    silver_30d['open_time'],
    silver_30d['rsi_14'],
    70,
    where=(silver_30d['rsi_14'] >= 70),
    color='red',
    alpha=0.2
)

plt.fill_between(
    silver_30d['open_time'],
    silver_30d['rsi_14'],
    30,
    where=(silver_30d['rsi_14'] <= 30),
    color='green',
    alpha=0.2
)

# Styling
plt.title("RSI Indicator (Last 30 Days)")
plt.xlabel("Time")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)

# Save
plt.savefig("/tmp/btc_rsi.png")
plt.close()

print("RSI chart saved: /tmp/btc_rsi.png")

# COMMAND ----------

# Create color column (green if price up, red if down)
colors = np.where(
    silver_30d['close'] >= silver_30d['open'],
    'green',
    'red'
)

# Plot volume
plt.figure(figsize=(14, 4))

plt.bar(
    silver_30d['open_time'],
    silver_30d['volume'],
    color=colors
)

# Styling
plt.title("Volume (Last 30 Days)")
plt.xlabel("Time")
plt.ylabel("Volume")
plt.grid(True)

# Save
plt.savefig("/tmp/btc_volume.png")
plt.close()

print("Volume chart saved: /tmp/btc_volume.png")

# COMMAND ----------

# Get latest metrics (last run)
latest_metrics = gold_metrics_df.sort_values('created_at').iloc[-1]

# Create text content
metrics_text = f"""
Model: {latest_metrics['model']}

RMSE: {latest_metrics['rmse']:.2f}
MAE: {latest_metrics['mae']:.2f}
R2 Score: {latest_metrics['r2']:.4f}
MAPE: {latest_metrics['mape']:.4f}%

Train Rows: {latest_metrics['train_rows']}
Test Rows: {latest_metrics['test_rows']}
"""

# Create figure
plt.figure(figsize=(8, 4))

# Display text
plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')

# Remove axes
plt.axis('off')

# Save
plt.savefig("/tmp/btc_metrics.png")
plt.close()

print("Metrics panel saved: /tmp/btc_metrics.png")

# COMMAND ----------

from PIL import Image

# Load all saved images
img1 = Image.open("/tmp/btc_actual_vs_predicted.png")
img2 = Image.open("/tmp/btc_rsi.png")
img3 = Image.open("/tmp/btc_volume.png")
img4 = Image.open("/tmp/btc_metrics.png")

# Resize images (to align properly)
img1 = img1.resize((1200, 400))
img2 = img2.resize((1200, 300))
img3 = img3.resize((1200, 300))
img4 = img4.resize((1200, 200))

# Create blank canvas
total_height = img1.height + img2.height + img3.height + img4.height
dashboard = Image.new('RGB', (1200, total_height), color=(255, 255, 255))

# Paste images one below another
y_offset = 0
for img in [img1, img2, img3, img4]:
    dashboard.paste(img, (0, y_offset))
    y_offset += img.height

# Save final dashboard
dashboard.save("/tmp/btc_dashboard.png")

print("FINAL DASHBOARD saved at /tmp/btc_dashboard.png")

# COMMAND ----------

def run_dashboard():
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from PIL import Image
    from datetime import timedelta

    # Load data
    silver_df = spark.table("workspace.silver.btc_features").toPandas()
    gold_pred_df = spark.table("workspace.gold.btc_predictions").toPandas()
    gold_metrics_df = spark.table("workspace.gold.btc_model_metrics").toPandas()

    # Convert datetime
    silver_df['open_time'] = pd.to_datetime(silver_df['open_time'])
    gold_pred_df['open_time'] = pd.to_datetime(gold_pred_df['open_time'])

    # Sort
    silver_df = silver_df.sort_values('open_time')
    gold_pred_df = gold_pred_df.sort_values('open_time')

    # Filter last 30 days
    latest_time = silver_df['open_time'].max()
    start_time = latest_time - timedelta(days=30)

    silver_30d = silver_df[silver_df['open_time'] >= start_time]
    gold_30d = gold_pred_df[gold_pred_df['open_time'] >= start_time]

    # -------- Chart 1: Actual vs Predicted --------
    plt.figure(figsize=(14, 6))
    plt.plot(silver_30d['open_time'], silver_30d['close'], label='Actual', color='blue')
    plt.plot(gold_30d['open_time'], gold_30d['predicted'], label='Predicted', color='orange')
    plt.legend()
    plt.grid(True)
    plt.title("Actual vs Predicted")
    plt.savefig("/tmp/c1.png")
    plt.close()

    # -------- Chart 2: RSI --------
    plt.figure(figsize=(14, 4))
    plt.plot(silver_30d['open_time'], silver_30d['rsi_14'], color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.grid(True)
    plt.title("RSI")
    plt.savefig("/tmp/c2.png")
    plt.close()

    # -------- Chart 3: Volume --------
    colors = np.where(silver_30d['close'] >= silver_30d['open'], 'green', 'red')
    plt.figure(figsize=(14, 4))
    plt.bar(silver_30d['open_time'], silver_30d['volume'], color=colors)
    plt.grid(True)
    plt.title("Volume")
    plt.savefig("/tmp/c3.png")
    plt.close()

    # -------- Chart 4: Metrics --------
    latest_metrics = gold_metrics_df.sort_values('created_at').iloc[-1]

    text = f"""
RMSE: {latest_metrics['rmse']:.2f}
MAE: {latest_metrics['mae']:.2f}
R2: {latest_metrics['r2']:.4f}
MAPE: {latest_metrics['mape']:.4f}%
"""

    plt.figure(figsize=(8, 3))
    plt.text(0.1, 0.5, text, fontsize=12)
    plt.axis('off')
    plt.savefig("/tmp/c4.png")
    plt.close()

    # -------- Combine --------
    imgs = [Image.open(f"/tmp/c{i}.png") for i in range(1, 5)]
    widths = [img.width for img in imgs]
    heights = [img.height for img in imgs]

    dashboard = Image.new('RGB', (max(widths), sum(heights)), (255, 255, 255))

    y = 0
    for img in imgs:
        dashboard.paste(img, (0, y))
        y += img.height

    dashboard.save("/tmp/btc_dashboard.png")

    print("Dashboard updated successfully")

# Run once manually
run_dashboard()

# COMMAND ----------

from IPython.display import Image, display

display(Image("/tmp/btc_dashboard.png"))

# COMMAND ----------

from IPython.display import Image, display

# Run dashboard first (important)
run_dashboard()

# Display image (THIS makes it visible in dashboard view)
display(Image("/tmp/btc_dashboard.png"))