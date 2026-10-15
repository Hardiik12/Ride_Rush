"""
Exploratory Data Analysis (EDA) Module for NYC Taxi Big Data Web Application.

Generates 12 static chart figures in static/charts/ using Dask aggregations
and sampling, computes IQR outlier tables, and returns data-driven key insights.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask web server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

CHARTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts")


def run_full_eda(df_clean):
    """
    Executes complete EDA pipeline using Dask aggregations and representative sampling.
    Saves all 12 PNG chart files to static/charts/ and returns structured analytical results.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams['figure.autolayout'] = True
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.titlesize'] = 12

    # Draw 5% random sample into memory for distribution & scatter charts to avoid OOM
    sample_df = df_clean.sample(frac=0.05, random_state=42).compute()

    # --- 1. TIME-BASED EDA (5.2) ---
    dt_series = df_clean['tpep_pickup_datetime'].dt
    hourly_trips = df_clean.groupby(dt_series.hour).size().compute()
    daily_trips = df_clean.groupby(dt_series.day_name()).size().compute()

    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_trips = daily_trips.reindex([d for d in day_order if d in daily_trips.index])

    # Chart 1: Trips by Hour
    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=hourly_trips.index, y=hourly_trips.values, hue=hourly_trips.index, palette="crest", legend=False)
    plt.title("NYC Yellow Taxi Trips by Hour of Day (Jan 2025)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Hour of Day (0 - 23)")
    plt.ylabel("Total Trips")
    plt.savefig(os.path.join(CHARTS_DIR, "trips_by_hour.png"), dpi=200)
    plt.close()

    # Chart 2: Trips by Day of Week
    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=daily_trips.index, y=daily_trips.values, hue=daily_trips.index, palette="mako", legend=False)
    plt.title("NYC Yellow Taxi Trips by Day of Week (Jan 2025)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Day of Week")
    plt.ylabel("Total Trips")
    plt.xticks(rotation=25)
    plt.savefig(os.path.join(CHARTS_DIR, "trips_by_day.png"), dpi=200)
    plt.close()

    # Chart 3: Trips by Date
    date_series = dt_series.date
    date_trips = df_clean.groupby(date_series).size().compute()
    plt.figure(figsize=(10, 4.5))
    date_trips.plot(kind='line', marker='o', color='#58a6ff', linewidth=2)
    plt.title("Daily NYC Taxi Trip Volume (January 2025)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Date")
    plt.ylabel("Total Trips")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(CHARTS_DIR, "trips_by_date.png"), dpi=200)
    plt.close()

    # --- 2. TRIP DISTANCE EDA (5.3) ---
    dist_stats = df_clean['trip_distance'].describe().compute()
    dist_median = df_clean['trip_distance'].quantile(0.5).compute()
    dist_common_range = "1.0 - 3.0 miles"

    plt.figure(figsize=(9, 4.5))
    sns.histplot(sample_df[sample_df['trip_distance'] <= 25]['trip_distance'], bins=35, color='#f0883e', kde=True)
    plt.title("Trip Distance Distribution (<= 25 miles)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(CHARTS_DIR, "trip_distance_distribution.png"), dpi=200)
    plt.close()

    # --- 3. TRIP DURATION EDA (5.4) ---
    dur_stats = df_clean['trip_duration_minutes'].describe().compute()
    dur_median = df_clean['trip_duration_minutes'].quantile(0.5).compute()

    plt.figure(figsize=(9, 4.5))
    sns.histplot(sample_df[sample_df['trip_duration_minutes'] <= 60]['trip_duration_minutes'], bins=35, color='#3fb950', kde=True)
    plt.title("Trip Duration Distribution (<= 60 minutes)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Trip Duration (minutes)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(CHARTS_DIR, "trip_duration_distribution.png"), dpi=200)
    plt.close()

    # --- 4. FARE EDA (5.5) ---
    fare_stats = df_clean['fare_amount'].describe().compute()
    fare_median = df_clean['fare_amount'].quantile(0.5).compute()

    plt.figure(figsize=(9, 4.5))
    sns.histplot(sample_df[sample_df['fare_amount'] <= 100]['fare_amount'], bins=35, color='#a371f7', kde=True)
    plt.title("Fare Amount Distribution (<= $100)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Fare Amount ($)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(CHARTS_DIR, "fare_distribution.png"), dpi=200)
    plt.close()

    # Fare vs Distance Density (Scatter/Hexbin)
    plt.figure(figsize=(9, 4.5))
    sample_sub = sample_df[(sample_df['trip_distance'] <= 25) & (sample_df['fare_amount'] <= 100)]
    hb = plt.hexbin(sample_sub['trip_distance'], sample_sub['fare_amount'], gridsize=30, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, label="Trip Count Density")
    plt.title("Fare Amount vs. Trip Distance", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Trip Distance (miles)")
    plt.ylabel("Fare Amount ($)")
    plt.savefig(os.path.join(CHARTS_DIR, "fare_vs_distance.png"), dpi=200)
    plt.close()

    # --- 5. PASSENGER EDA (5.6) ---
    pass_counts = df_clean.groupby('passenger_count').size().compute()
    most_common_pass = int(pass_counts.idxmax())

    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=pass_counts.index.astype(int), y=pass_counts.values, hue=pass_counts.index.astype(int), palette="Blues_d", legend=False)
    plt.title("Passenger Count Distribution", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Passenger Count")
    plt.ylabel("Total Trips")
    plt.savefig(os.path.join(CHARTS_DIR, "passenger_distribution.png"), dpi=200)
    plt.close()

    # --- 6. PAYMENT TYPE EDA (5.7) ---
    payment_map = {1: 'Credit Card', 2: 'Cash', 3: 'No Charge', 4: 'Dispute', 5: 'Unknown', 6: 'Voided'}
    pay_summary = df_clean.groupby('payment_type').agg({
        'total_amount': 'count',
        'fare_amount': 'mean',
        'tip_amount': 'mean'
    }).compute()

    pay_summary.rename(index=payment_map, columns={'total_amount': 'trip_count'}, inplace=True)
    most_common_pay_code = df_clean['payment_type'].value_counts().idxmax().compute()
    most_common_pay = payment_map.get(int(most_common_pay_code), str(most_common_pay_code))

    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=pay_summary.index, y=pay_summary['trip_count'], hue=pay_summary.index, palette="Spectral", legend=False)
    plt.title("Trip Count by Payment Type", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Payment Method")
    plt.ylabel("Total Trips")
    plt.savefig(os.path.join(CHARTS_DIR, "payment_distribution.png"), dpi=200)
    plt.close()

    # --- 7. LOCATION EDA (5.8) ---
    top_pickup = df_clean.groupby('PULocationID').size().nlargest(10).compute()
    top_dropoff = df_clean.groupby('DOLocationID').size().nlargest(10).compute()

    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=top_pickup.index.astype(str), y=top_pickup.values, hue=top_pickup.index.astype(str), palette="rocket", legend=False)
    plt.title("Top 10 Pickup TLC Taxi Zones (PULocationID)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("PULocationID Zone")
    plt.ylabel("Total Pickups")
    plt.savefig(os.path.join(CHARTS_DIR, "top_pickup_locations.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(9, 4.5))
    sns.barplot(x=top_dropoff.index.astype(str), y=top_dropoff.values, hue=top_dropoff.index.astype(str), palette="flare", legend=False)
    plt.title("Top 10 Drop-off TLC Taxi Zones (DOLocationID)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("DOLocationID Zone")
    plt.ylabel("Total Drop-offs")
    plt.savefig(os.path.join(CHARTS_DIR, "top_dropoff_locations.png"), dpi=200)
    plt.close()

    # --- 8. CORRELATION ANALYSIS (5.9) ---
    corr_cols = ['passenger_count', 'trip_distance', 'trip_duration_minutes', 'fare_amount', 'tip_amount', 'total_amount']
    corr_matrix = sample_df[corr_cols].corr()

    plt.figure(figsize=(8, 5.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Numerical Feature Correlation Matrix Heatmap", fontsize=13, fontweight='bold', pad=12)
    plt.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), dpi=200)
    plt.close()

    # --- 9. OUTLIER ANALYSIS (5.10) ---
    outlier_metrics = ['trip_distance', 'trip_duration_minutes', 'fare_amount', 'total_amount']
    clean_total = len(df_clean)
    outlier_table = []

    for col in outlier_metrics:
        q25, q75 = df_clean[col].quantile([0.25, 0.75]).compute().values
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers_cnt = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum().compute())
        outlier_table.append({
            "variable": col,
            "q1": round(float(q25), 2),
            "q3": round(float(q75), 2),
            "iqr": round(float(iqr), 2),
            "lower_bound": round(float(lower), 2),
            "upper_bound": round(float(upper), 2),
            "potential_outliers": outliers_cnt,
            "outlier_pct": round((outliers_cnt / clean_total) * 100.0, 2)
        })

    # Core Calculated Insights
    busiest_hour_val = int(hourly_trips.idxmax())
    busiest_day_val = str(daily_trips.idxmax())

    insights = {
        "busiest_hour": f"{busiest_hour_val}:00 ({hourly_trips.max():,} trips)",
        "busiest_day": f"{busiest_day_val} ({daily_trips.max():,} trips)",
        "avg_trip_distance": f"{dist_stats['mean']:.2f} miles",
        "median_trip_distance": f"{dist_median:.2f} miles",
        "min_trip_distance": f"{dist_stats['min']:.2f} miles",
        "max_trip_distance": f"{dist_stats['max']:.2f} miles",
        "avg_trip_duration": f"{dur_stats['mean']:.2f} minutes",
        "median_trip_duration": f"{dur_median:.2f} minutes",
        "min_trip_duration": f"{dur_stats['min']:.2f} minutes",
        "max_trip_duration": f"{dur_stats['max']:.2f} minutes",
        "avg_fare": f"${fare_stats['mean']:.2f}",
        "median_fare": f"${fare_median:.2f}",
        "min_fare": f"${fare_stats['min']:.2f}",
        "max_fare": f"${fare_stats['max']:.2f}",
        "most_common_payment": most_common_pay,
        "most_common_passengers": most_common_pass,
        "top_pickup_location": f"Zone {top_pickup.idxmax()} ({top_pickup.max():,} pickups)",
        "top_dropoff_location": f"Zone {top_dropoff.idxmax()} ({top_dropoff.max():,} drop-offs)",
        "distance_fare_corr": f"{corr_matrix.loc['trip_distance', 'fare_amount']:.2f}"
    }

    return {
        "outlier_table": outlier_table,
        "insights": insights,
        "payment_summary": pay_summary.to_dict(orient="index")
    }
