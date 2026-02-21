#!/usr/bin/env python3
"""
Calculate chilling hours for winter dormancy requirements
Cherry trees need cold exposure to break dormancy
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Chilling parameters
CHILL_TEMP_MIN = 0.0   # Minimum temperature for chilling (°C)
CHILL_TEMP_MAX = 7.2   # Maximum temperature for chilling (°C)

def calculate_daily_chill_hours(T_min, T_max, T_mean):
    """
    Estimate daily chilling hours

    Simplified model: if daily mean temp is in chilling range, count full day (24 hours)
    More sophisticated: use triangular approximation

    Parameters:
    -----------
    T_min, T_max, T_mean : float
        Daily temperatures

    Returns:
    --------
    float : estimated chilling hours for the day
    """

    # Simple approach: if mean temp in range, assume ~12 hours of chilling
    if pd.isna(T_mean):
        return 0

    if CHILL_TEMP_MIN <= T_mean <= CHILL_TEMP_MAX:
        # Temperature mostly in chilling range
        return 12.0
    elif T_mean < CHILL_TEMP_MIN:
        # Too cold - partial chilling from daytime warming
        if not pd.isna(T_max) and T_max > CHILL_TEMP_MIN:
            return 6.0  # Partial day
        return 0
    elif T_mean > CHILL_TEMP_MAX:
        # Too warm - partial chilling from nighttime cooling
        if not pd.isna(T_min) and T_min < CHILL_TEMP_MAX:
            return 6.0  # Partial night
        return 0
    else:
        return 0

def calculate_accumulated_chill(daily_df):
    """
    Calculate accumulated chilling hours for Nov-Dec-Jan period

    Parameters:
    -----------
    daily_df : pd.DataFrame
        Daily weather data

    Returns:
    --------
    pd.DataFrame with accumulated chilling hours
    """

    print("Calculating accumulated chilling hours...")

    df = daily_df.copy()

    # Ensure date is parsed
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    # Calculate daily chilling hours
    df['chill_hours_daily'] = df.apply(
        lambda row: calculate_daily_chill_hours(row['T_min'], row['T_max'], row['T_mean']),
        axis=1
    )

    print(f"  ✓ Calculated daily chilling hours for {len(df):,} records")

    return df

def calculate_chill_features(daily_with_chill):
    """
    Calculate chilling hour features for modeling

    Key features:
    - chill_hours_nov_dec: November-December accumulation (previous year)
    - chill_hours_winter: Nov-Dec-Jan accumulation
    - chill_hours_nov_jan: Extended winter period
    """

    print("Creating chilling hour features...")

    features = []

    for location in daily_with_chill['location'].unique():
        loc_data = daily_with_chill[daily_with_chill['location'] == location].sort_values('date')

        for year in range(loc_data['year'].min() + 1, loc_data['year'].max() + 1):
            record = {'location': location, 'year': year}

            # November-December of previous year
            nov_dec = loc_data[
                (loc_data['year'] == year - 1) &
                (loc_data['month'].isin([11, 12]))
            ]

            if len(nov_dec) > 0:
                record['chill_hours_nov_dec'] = nov_dec['chill_hours_daily'].sum()

            # Nov-Dec-Jan (previous Nov-Dec, current Jan)
            winter = loc_data[
                ((loc_data['year'] == year - 1) & (loc_data['month'].isin([11, 12]))) |
                ((loc_data['year'] == year) & (loc_data['month'] == 1))
            ]

            if len(winter) > 0:
                record['chill_hours_winter'] = winter['chill_hours_daily'].sum()

            # Extended: Nov-Dec-Jan-Feb
            extended = loc_data[
                ((loc_data['year'] == year - 1) & (loc_data['month'].isin([11, 12]))) |
                ((loc_data['year'] == year) & (loc_data['month'].isin([1, 2])))
            ]

            if len(extended) > 0:
                record['chill_hours_nov_feb'] = extended['chill_hours_daily'].sum()

            # December only (for granularity)
            dec = loc_data[(loc_data['year'] == year - 1) & (loc_data['month'] == 12)]
            if len(dec) > 0:
                record['chill_hours_dec'] = dec['chill_hours_daily'].sum()

            # January only
            jan = loc_data[(loc_data['year'] == year) & (loc_data['month'] == 1)]
            if len(jan) > 0:
                record['chill_hours_jan'] = jan['chill_hours_daily'].sum()

            features.append(record)

    chill_features = pd.DataFrame(features)

    print(f"  ✓ Created chilling features for {len(chill_features):,} location-years")

    return chill_features

def main():
    """Calculate chilling hours from weather data"""

    print("="*70)
    print("CHILLING HOURS CALCULATION")
    print("="*70)
    print(f"Chilling temperature range: {CHILL_TEMP_MIN}°C - {CHILL_TEMP_MAX}°C")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load daily weather data
    daily_file = PROCESSED_DIR / 'weather_daily_cleaned.csv'

    if not daily_file.exists():
        print(f"ERROR: {daily_file} not found. Run 05_process_weather.py first.")
        return None

    print(f"Loading daily weather data from {daily_file}...")
    daily = pd.read_csv(daily_file)

    # Calculate chilling hours
    daily_with_chill = calculate_accumulated_chill(daily)

    # Save daily chilling data
    daily_chill_file = PROCESSED_DIR / 'weather_daily_with_chill.csv'
    daily_with_chill.to_csv(daily_chill_file, index=False)
    print(f"Saved daily chilling data to: {daily_chill_file}")

    # Create chilling features
    chill_features = calculate_chill_features(daily_with_chill)

    # Save chilling features
    chill_features_file = PROCESSED_DIR / 'chill_features.csv'
    chill_features.to_csv(chill_features_file, index=False)
    print(f"Saved chilling features to: {chill_features_file}")

    # Summary statistics
    print("\n" + "="*70)
    print("CHILLING HOURS SUMMARY")
    print("="*70)
    print(f"Daily records with chilling: {len(daily_with_chill):,}")
    print(f"Chilling feature records: {len(chill_features):,}")

    print(f"\nChilling hours Nov-Dec statistics:")
    print(chill_features['chill_hours_nov_dec'].describe())

    print(f"\nChilling hours winter statistics:")
    print(chill_features['chill_hours_winter'].describe())

    print(f"\nSample chilling features (recent years):")
    print(chill_features[chill_features['year'] >= 2020].head(10))

    print(f"\nChilling hours by location (mean):")
    print(chill_features.groupby('location')['chill_hours_winter'].mean().sort_values())

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return chill_features

if __name__ == '__main__':
    data = main()
