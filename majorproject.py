"""
================================================================================
 RETAIL ANALYTICS & AI-POWERED SALES FORECASTING SYSTEM
 Dataset: Retail_Sales_Data_Unlox.csv (~73,000 records, Jan 2023 - Dec 2024)
================================================================================
 RUN:
   1) pip install pandas numpy matplotlib seaborn scikit-learn
   2) python retail_analytics.py
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

OUT = "outputs"
os.makedirs(f"{OUT}/figures", exist_ok=True)
os.makedirs(f"{OUT}/reports", exist_ok=True)

DATA_FILE = "Retail_Sales_Data_Unlox.csv"


# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
def load_data(path=DATA_FILE):
    print("=" * 70)
    print(" STEP 1: LOADING DATA")
    print("=" * 70)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ '{path}' not found. Place the CSV in this folder.")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    print(f"[✔] Shape: {df.shape}")
    print(f"[✔] Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"[✔] Columns: {list(df.columns)}")
    print(df.head(), "\n")
    return df


# ==============================================================================
# 2. CLEAN + FEATURE ENGINEERING
# ==============================================================================
def prepare(df):
    print("=" * 70)
    print(" STEP 2: CLEANING & FEATURE ENGINEERING")
    print("=" * 70)

    # Drop rows with no revenue
    df = df.dropna(subset=["Revenue"])
    df = df[df["Revenue"] >= 0].drop_duplicates().reset_index(drop=True)

    # Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)

    print(f"[✔] After cleaning: {df.shape}\n")
    return df


# ==============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================================================================
def run_eda(df):
    print("=" * 70)
    print(" STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # 3.1 Monthly revenue trend
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()
    monthly.index = monthly.index.astype(str)

    plt.figure(figsize=(12, 5))
    monthly.plot(marker="o", color="#1f77b4")
    plt.title("Monthly Revenue Trend (2023–2024)")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/01_monthly_revenue.png", dpi=120)
    plt.close()

    # 3.2 Revenue by Product Category
    cat = df.groupby("Product_Category")["Revenue"].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=cat.values, y=cat.index, palette="viridis")
    plt.title("Total Revenue by Product Category")
    plt.xlabel("Revenue")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/02_category_revenue.png", dpi=120)
    plt.close()

    # 3.3 Top 10 stores
    top_stores = df.groupby("Store_ID")["Revenue"].sum().nlargest(10)
    plt.figure(figsize=(11, 5))
    sns.barplot(x=top_stores.index, y=top_stores.values, palette="magma")
    plt.title("Top 10 Stores by Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/03_top_stores.png", dpi=120)
    plt.close()

    # 3.4 Seasonality heatmap (DayOfWeek × Month)
    pivot = df.pivot_table(values="Revenue", index="DayOfWeek",
                           columns="Month", aggfunc="mean")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Avg Revenue"})
    plt.title("Seasonality: Day of Week × Month")
    plt.yticks(ticks=range(7),
               labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
               rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/04_seasonality.png", dpi=120)
    plt.close()

    # 3.5 Revenue by Region
    region = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=region.index, y=region.values, palette="coolwarm")
    plt.title("Revenue by Region")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/05_region_revenue.png", dpi=120)
    plt.close()

    # 3.6 Customer type split
    cust = df.groupby("Customer_Type")["Revenue"].sum()
    plt.figure(figsize=(6, 6))
    plt.pie(cust.values, labels=cust.index, autopct="%1.1f%%",
            colors=["#4C72B0", "#DD8452"], startangle=90)
    plt.title("Revenue Share by Customer Type")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/06_customer_type.png", dpi=120)
    plt.close()

    # 3.7 Payment mode
    pay = df.groupby("Payment_Mode")["Revenue"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=pay.index, y=pay.values, palette="Set2")
    plt.title("Revenue by Payment Mode")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/07_payment_mode.png", dpi=120)
    plt.close()

    # YoY growth
    yoy = df.groupby("Year")["Revenue"].sum()
    growth = yoy.pct_change().mul(100).round(2)

    # Executive summary
    lines = [
        "=" * 70,
        " RETAIL ANALYTICS – EXECUTIVE SUMMARY",
        "=" * 70,
        f"Total Revenue        : ${df['Revenue'].sum():,.2f}",
        f"Total Transactions   : {len(df):,}",
        f"Unique Stores        : {df['Store_ID'].nunique()}",
        f"Product Categories   : {df['Product_Category'].nunique()}",
        f"Date Range           : {df['Date'].min().date()} → {df['Date'].max().date()}",
        "",
        f"Top Category         : {cat.index[0]}  (${cat.values[0]:,.0f})",
        f"Top Store            : {top_stores.index[0]}  (${top_stores.values[0]:,.0f})",
        f"Best Revenue Month   : {monthly.idxmax()}  (${monthly.max():,.0f})",
        f"YoY Growth           : {growth.iloc[-1] if len(growth) > 1 else 0:.2f}%",
        "=" * 70,
    ]
    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(f"{OUT}/reports/executive_summary.txt", "w") as f:
        f.write(text)

    return {"monthly": monthly, "category": cat, "store": top_stores, "yoy": yoy}


# ==============================================================================
# 4. SALES FORECASTING (Random Forest + Lag Features)
# ==============================================================================
def forecast_sales(df, horizon=90):
    print("=" * 70)
    print(" STEP 4: 90-DAY SALES FORECAST")
    print("=" * 70)

    # Daily total revenue
    ts = df.groupby("Date")["Revenue"].sum().asfreq("D").fillna(0)
    ts_df = pd.DataFrame({"Date": ts.index, "Revenue": ts.values})

    # Lag + rolling features
    for lag in [1, 7, 14, 30]:
        ts_df[f"lag_{lag}"] = ts_df["Revenue"].shift(lag)
    ts_df["roll_7"] = ts_df["Revenue"].shift(1).rolling(7).mean()
    ts_df["roll_30"] = ts_df["Revenue"].shift(1).rolling(30).mean()
    ts_df["dow"] = ts_df["Date"].dt.dayofweek
    ts_df["month"] = ts_df["Date"].dt.month
    ts_df["dayofyear"] = ts_df["Date"].dt.dayofyear

    ts_df = ts_df.dropna().reset_index(drop=True)
    features = [c for c in ts_df.columns if c not in ("Date", "Revenue")]

    X, y = ts_df[features], ts_df["Revenue"]
    split = int(len(ts_df) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                  random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    mape = np.mean(np.abs((y_te - pred) / np.maximum(y_te, 1))) * 100
    print(f"[✔] MAE={mae:,.2f} | RMSE={rmse:,.2f} | R²={r2:.4f} | MAPE={mape:.2f}%\n")

    # Retrain on full data
    final = RandomForestRegressor(n_estimators=200, max_depth=12,
                                  random_state=42, n_jobs=-1)
    final.fit(X, y)

    # Recursive future forecast
    recent = list(ts_df["Revenue"].values)
    last_date = ts_df["Date"].iloc[-1]
    future_rows = []

    for i in range(1, horizon + 1):
        d = last_date + pd.Timedelta(days=i)
        row = {
            "lag_1": recent[-1],
            "lag_7": recent[-7],
            "lag_14": recent[-14],
            "lag_30": recent[-30],
            "roll_7": np.mean(recent[-7:]),
            "roll_30": np.mean(recent[-30:]),
            "dow": d.dayofweek,
            "month": d.month,
            "dayofyear": d.dayofyear,
        }
        yhat = final.predict(pd.DataFrame([row])[features])[0]
        recent.append(yhat)
        future_rows.append({"Date": d, "Forecast_Revenue": yhat})

    fc = pd.DataFrame(future_rows)
    fc.to_csv(f"{OUT}/reports/forecast_90days.csv", index=False)

    # Plot
    plt.figure(figsize=(13, 5))
    plt.plot(ts_df["Date"], ts_df["Revenue"], label="Historical", color="#1f77b4")
    plt.plot(fc["Date"], fc["Forecast_Revenue"], label="Forecast (90d)",
             color="#d62728", linewidth=2)
    plt.axvline(ts_df["Date"].iloc[-1], color="gray", ls="--", alpha=0.7)
    plt.title("90-Day Revenue Forecast")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/08_forecast.png", dpi=120)
    plt.close()

    # Feature importance
    imp = pd.Series(final.feature_importances_, index=features
                    ).sort_values(ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=imp.values, y=imp.index, palette="crest")
    plt.title("Top Forecast Drivers")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/09_feature_importance.png", dpi=120)
    plt.close()

    print(f"[✔] Next 90 days projected revenue: ${fc['Forecast_Revenue'].sum():,.2f}")
    print(f"[✔] Forecast saved → {OUT}/reports/forecast_90days.csv\n")
    return fc


# ==============================================================================
# 5. STORE SEGMENTATION (K-MEANS)
# ==============================================================================
def segment_stores(df, k=4):
    print("=" * 70)
    print(" STEP 5: STORE SEGMENTATION (K-Means)")
    print("=" * 70)

    stats = df.groupby("Store_ID").agg(
        Total_Revenue=("Revenue", "sum"),
        Avg_Revenue=("Revenue", "mean"),
        Transactions=("Revenue", "count"),
        Avg_Rating=("Store_Rating", "mean"),
    ).fillna(0)

    X = StandardScaler().fit_transform(stats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    stats["Segment"] = km.fit_predict(X)

    order = stats.groupby("Segment")["Total_Revenue"].mean().sort_values().index
    labels = {seg: lab for seg, lab in zip(
        order, ["Low Performer", "Stable", "Growing", "Star"][:k])}
    stats["Segment_Label"] = stats["Segment"].map(labels)

    stats.to_csv(f"{OUT}/reports/store_segments.csv")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stats, x="Total_Revenue", y="Avg_Revenue",
                    hue="Segment_Label", s=200, palette="Set2")
    plt.title("Store Segments")
    plt.tight_layout()
    plt.savefig(f"{OUT}/figures/10_store_segments.png", dpi=120)
    plt.close()

    print(stats.groupby("Segment_Label")[["Total_Revenue", "Avg_Revenue"]]
          .mean().round(2))
    print(f"\n[✔] Segments saved → {OUT}/reports/store_segments.csv\n")
    return stats


# ==============================================================================
# 6. MAIN
# ==============================================================================
def main():
    print("\n" + "🛒" * 35)
    print(" RETAIL ANALYTICS & AI-POWERED SALES FORECASTING SYSTEM")
    print("🛒" * 35 + "\n")

    df = load_data()
    df = prepare(df)

    run_eda(df)
    forecast_sales(df, horizon=90)
    segment_stores(df, k=4)

    print("=" * 70)
    print(" ✅ PROJECT COMPLETE")
    print("=" * 70)
    print(f" 📁 Outputs → ./{OUT}/")
    print(f"    ├── figures/ (10 charts)")
    print(f"    └── reports/ (executive_summary.txt, forecast_90days.csv,")
    print(f"                  store_segments.csv)\n")


if __name__ == "__main__":
    main()