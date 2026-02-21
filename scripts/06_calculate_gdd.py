#!/usr/bin/env python3
"""
Calculate Growing Degree Days (GDD) and related thermal features
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# GDD parameters
BASE_TEMP = 5.0  # Base temperature for cherry blossoms (°C)

def calculate_daily_gdd(T_mean, base_temp=5.0):
    """
    Calculate daily Growing Degree Days

    GDD = max(0, T_mean - base_temp)
    """
    return np.maximum(T_mean - base_temp, 0)

def calculate_accumulated_gdd(daily_df):
    """
    Calculate accumulated GDD from January 1st each year

    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily weather data with T_mean

    Returns:
    --------
    pd.DataFrame with accumulated GDD
    """

    print("Calculating accumulated GDD...")

    df = daily_df.copy()

    # Ensure date is parsed
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Calculate daily GDD
    df['GDD_daily'] = calculate_daily_gdd(df['T_mean'], BASE_TEMP)

    # Sort by location, year, day_of_year
    df = df.sort_values(['location', 'year', 'day_of_year'])

    # Accumulated GDD from Jan 1
    df['GDD_accum'] = df.groupby(['location', 'year'])['GDD_daily'].cumsum()

    print(f"  ✓ Calculated GDD for {len(df):,} daily records")

    return df

def calculate_monthly_gdd(monthly_df):
    """
    Calculate monthly GDD sums

    Parameters:
    -----------
    monthly_df : pd.DataFrame
        Monthly weather data

    Returns:
    --------
    pd.DataFrame with monthly GDD
    """

    print("Calculating monthly GDD...")

    df = monthly_df.copy()

    # Approximate GDD per month (using monthly mean temp)
    # Days per month (approximate)
    days_in_month = {
        1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
    }

    df['days'] = df['month'].map(days_in_month)

    # Monthly GDD = GDD_daily * days_in_month
    df['GDD_daily_avg'] = calculate_daily_gdd(df['T_mean_monthly'], BASE_TEMP)
    df['GDD_monthly'] = df['GDD_daily_avg'] * df['days']

    print(f"  ✓ Calculated monthly GDD for {len(df):,} records")

    return df

def calculate_gdd_features(monthly_gdd):
    """
    Calculate GDD features for modeling

    Key features:
    - GDD_jan: January GDD
    - GDD_feb: February GDD
    - GDD_jan_feb: Cumulative Jan-Feb GDD (no data leakage)
    - GDD_winter: Dec-Jan-Feb GDD
    """

    print("Creating GDD features...")

    features = []

    for location in monthly_gdd['location'].unique():
        loc_data = monthly_gdd[monthly_gdd['location'] == location].sort_values(['year', 'month'])

        for year in loc_data['year'].unique():
            record = {'location': location, 'year': year}

            # January GDD
            jan = loc_data[(loc_data['year'] == year) & (loc_data['month'] == 1)]
            if len(jan) > 0:
                record['GDD_jan'] = jan['GDD_monthly'].values[0]

            # February GDD
            feb = loc_data[(loc_data['year'] == year) & (loc_data['month'] == 2)]
            if len(feb) > 0:
                record['GDD_feb'] = feb['GDD_monthly'].values[0]

            # Jan-Feb cumulative (NO DATA LEAKAGE - safe for prediction)
            jan_feb = loc_data[(loc_data['year'] == year) & (loc_data['month'].isin([1, 2]))]
            if len(jan_feb) > 0:
                record['GDD_jan_feb'] = jan_feb['GDD_monthly'].sum()

            # Winter (Dec-Jan-Feb) - includes previous December
            winter = loc_data[
                ((loc_data['year'] == year - 1) & (loc_data['month'] == 12)) |
                ((loc_data['year'] == year) & (loc_data['month'].isin([1, 2])))
            ]
            if len(winter) > 0:
                record['GDD_winter'] = winter['GDD_monthly'].sum()

            # November-December (for interaction with chilling)
            nov_dec = loc_data[
                ((loc_data['year'] == year - 1) & (loc_data['month'].isin([11, 12])))
            ]
            if len(nov_dec) > 0:
                record['GDD_nov_dec'] = nov_dec['GDD_monthly'].sum()

            features.append(record)

    gdd_features = pd.DataFrame(features)

    print(f"  ✓ Created GDD features for {len(gdd_features):,} location-years")

    return gdd_features

def main():
    """Calculate GDD from weather data"""

    print("="*70)
    print("GROWING DEGREE DAYS CALCULATION")
    print("="*70)
    print(f"Base temperature: {BASE_TEMP}°C")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load processed weather data
    daily_file = PROCESSED_DIR / 'weather_daily_cleaned.csv'
    monthly_file = PROCESSED_DIR / 'weather_monthly.csv'

    if not daily_file.exists():
        print(f"ERROR: {daily_file} not found. Run 05_process_weather.py first.")
        return None

    print(f"Loading daily weather data from {daily_file}...")
    daily = pd.read_csv(daily_file)

    # Calculate accumulated GDD on daily data
    daily_with_gdd = calculate_accumulated_gdd(daily)

    # Save daily GDD
    daily_gdd_file = PROCESSED_DIR / 'weather_daily_with_gdd.csv'
    daily_with_gdd.to_csv(daily_gdd_file, index=False)
    print(f"Saved daily GDD to: {daily_gdd_file}")

    # Load monthly data
    if monthly_file.exists():
        print(f"\nLoading monthly weather data from {monthly_file}...")
        monthly = pd.read_csv(monthly_file)

        # Calculate monthly GDD
        monthly_with_gdd = calculate_monthly_gdd(monthly)

        # Save monthly GDD
        monthly_gdd_file = PROCESSED_DIR / 'weather_monthly_with_gdd.csv'
        monthly_with_gdd.to_csv(monthly_gdd_file, index=False)
        print(f"Saved monthly GDD to: {monthly_gdd_file}")

        # Create GDD features
        gdd_features = calculate_gdd_features(monthly_with_gdd)

        # Save GDD features
        gdd_features_file = PROCESSED_DIR / 'gdd_features.csv'
        gdd_features.to_csv(gdd_features_file, index=False)
        print(f"Saved GDD features to: {gdd_features_file}")

        # Summary statistics
        print("\n" + "="*70)
        print("GDD CALCULATION SUMMARY")
        print("="*70)
        print(f"Daily records with GDD: {len(daily_with_gdd):,}")
        print(f"Monthly records with GDD: {len(monthly_with_gdd):,}")
        print(f"GDD feature records: {len(gdd_features):,}")

        print(f"\nGDD Jan-Feb statistics (for prediction):")
        print(gdd_features['GDD_jan_feb'].describe())

        print(f"\nSample GDD features (recent years):")
        print(gdd_features[gdd_features['year'] >= 2020].head(10))

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        return gdd_features
    else:
        print(f"WARNING: {monthly_file} not found. Skipping monthly GDD.")
        return None

if __name__ == '__main__':
    data = main()
