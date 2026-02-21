#!/usr/bin/env python3
"""
Process and aggregate weather data
Creates daily and monthly aggregates from raw downloads
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
RAW_DIR = Path('data/raw/weather')
PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_weather_data():
    """Load and combine weather data from Open-Meteo and Meteostat"""

    print("Loading weather data...")

    # First try combined file (has all sites)
    combined_file = RAW_DIR / 'all_sites_combined.csv'
    openmeteo_file = RAW_DIR / 'openmeteo_all_sites.csv'
    meteostat_file = RAW_DIR / 'meteostat_all_sites.csv'

    df_list = []

    if combined_file.exists():
        print(f"  Loading combined data: {combined_file}")
        df_combined = pd.read_csv(combined_file)
        df_list.append(df_combined)
        print(f"    ✓ Loaded {len(df_combined):,} records")
    else:
        # Fall back to individual files
        if openmeteo_file.exists():
            print(f"  Loading Open-Meteo data: {openmeteo_file}")
            df_openmeteo = pd.read_csv(openmeteo_file)
            df_openmeteo['source'] = 'openmeteo'
            df_list.append(df_openmeteo)
            print(f"    ✓ Loaded {len(df_openmeteo):,} records")

        if meteostat_file.exists():
            print(f"  Loading Meteostat data: {meteostat_file}")
            df_meteostat = pd.read_csv(meteostat_file)
            df_meteostat['source'] = 'meteostat'
            df_list.append(df_meteostat)
            print(f"    ✓ Loaded {len(df_meteostat):,} records")

    if not df_list:
        print("ERROR: No weather data files found!")
        return None

    # Combine
    df = pd.concat(df_list, ignore_index=True)

    # Parse date (handle mixed formats)
    df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')

    # Ensure we have year, month, day_of_year
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    if 'month' not in df.columns:
        df['month'] = df['date'].dt.month
    if 'day_of_year' not in df.columns:
        df['day_of_year'] = df['date'].dt.dayofyear

    print(f"\nTotal combined records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Locations: {df['location'].nunique()}")

    return df

def create_monthly_aggregates(df):
    """
    Create monthly temperature and precipitation aggregates

    Parameters:
    -----------
    df : pd.DataFrame
        Daily weather data

    Returns:
    --------
    pd.DataFrame with monthly aggregates
    """

    print("\nCreating monthly aggregates...")

    # Group by location, year, month
    monthly = df.groupby(['location', 'year', 'month']).agg({
        'T_mean': 'mean',
        'T_max': 'mean',
        'T_min': 'mean',
        'precip': 'sum',
        'snowfall': 'sum'
    }).reset_index()

    # Rename for clarity
    monthly = monthly.rename(columns={
        'T_mean': 'T_mean_monthly',
        'T_max': 'T_max_monthly',
        'T_min': 'T_min_monthly',
        'precip': 'precip_monthly',
        'snowfall': 'snowfall_monthly'
    })

    print(f"  ✓ Created {len(monthly):,} monthly records")

    return monthly

def pivot_monthly_to_annual(monthly):
    """
    Pivot monthly data to wide format with one row per location-year

    Creates columns like: T_mean_jan, T_mean_feb, ..., precip_jan, etc.
    """

    print("\nPivoting monthly data to annual format...")

    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # Create separate pivots for each variable
    pivots = []

    for var in ['T_mean_monthly', 'T_max_monthly', 'T_min_monthly',
                'precip_monthly', 'snowfall_monthly']:

        pivot = monthly.pivot_table(
            index=['location', 'year'],
            columns='month',
            values=var,
            aggfunc='first'
        ).reset_index()

        # Rename columns
        base_name = var.replace('_monthly', '')
        new_cols = {i+1: f'{base_name}_{month_names[i]}' for i in range(12)}
        pivot = pivot.rename(columns=new_cols)

        pivots.append(pivot)

    # Merge all pivots
    annual = pivots[0]
    for pivot in pivots[1:]:
        annual = annual.merge(pivot, on=['location', 'year'], how='outer')

    print(f"  ✓ Created {len(annual):,} annual records with {len(annual.columns)} columns")

    return annual

def calculate_seasonal_aggregates(monthly):
    """
    Calculate seasonal temperature and precipitation means

    Winter: Dec-Jan-Feb (using previous Dec)
    Spring: Mar-Apr-May
    """

    print("\nCalculating seasonal aggregates...")

    seasonal_records = []

    for location in monthly['location'].unique():
        loc_data = monthly[monthly['location'] == location].sort_values(['year', 'month'])

        for year in loc_data['year'].unique():
            record = {'location': location, 'year': year}

            # Winter (DJF) - previous Dec, current Jan-Feb
            winter_months = loc_data[
                ((loc_data['year'] == year - 1) & (loc_data['month'] == 12)) |
                ((loc_data['year'] == year) & (loc_data['month'].isin([1, 2])))
            ]

            if len(winter_months) > 0:
                record['T_mean_winter'] = winter_months['T_mean_monthly'].mean()
                record['T_max_winter'] = winter_months['T_max_monthly'].mean()
                record['T_min_winter'] = winter_months['T_min_monthly'].mean()
                record['precip_winter'] = winter_months['precip_monthly'].sum()

            # Spring (MAM)
            spring_months = loc_data[
                (loc_data['year'] == year) & (loc_data['month'].isin([3, 4, 5]))
            ]

            if len(spring_months) > 0:
                record['T_mean_spring'] = spring_months['T_mean_monthly'].mean()
                record['T_max_spring'] = spring_months['T_max_monthly'].mean()
                record['T_min_spring'] = spring_months['T_min_monthly'].mean()
                record['precip_spring'] = spring_months['precip_monthly'].sum()

            # Jan-Feb only (to avoid data leakage)
            jan_feb = loc_data[
                (loc_data['year'] == year) & (loc_data['month'].isin([1, 2]))
            ]

            if len(jan_feb) > 0:
                record['T_mean_jan_feb'] = jan_feb['T_mean_monthly'].mean()
                record['T_max_jan_feb'] = jan_feb['T_max_monthly'].mean()
                record['T_min_jan_feb'] = jan_feb['T_min_monthly'].mean()
                record['precip_jan_feb'] = jan_feb['precip_monthly'].sum()

            seasonal_records.append(record)

    seasonal = pd.DataFrame(seasonal_records)

    print(f"  ✓ Created {len(seasonal):,} seasonal records")

    return seasonal

def main():
    """Process all weather data"""

    print("="*70)
    print("WEATHER DATA PROCESSING")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load raw data
    daily = load_weather_data()

    if daily is None:
        return None

    # Save cleaned daily data
    daily_file = PROCESSED_DIR / 'weather_daily_cleaned.csv'
    daily.to_csv(daily_file, index=False)
    print(f"\nSaved cleaned daily data to: {daily_file}")

    # Create monthly aggregates
    monthly = create_monthly_aggregates(daily)

    # Save monthly data
    monthly_file = PROCESSED_DIR / 'weather_monthly.csv'
    monthly.to_csv(monthly_file, index=False)
    print(f"Saved monthly data to: {monthly_file}")

    # Create annual wide format
    annual_wide = pivot_monthly_to_annual(monthly)

    # Save annual wide
    annual_file = PROCESSED_DIR / 'weather_annual_wide.csv'
    annual_wide.to_csv(annual_file, index=False)
    print(f"Saved annual wide data to: {annual_file}")

    # Create seasonal aggregates
    seasonal = calculate_seasonal_aggregates(monthly)

    # Save seasonal
    seasonal_file = PROCESSED_DIR / 'weather_seasonal.csv'
    seasonal.to_csv(seasonal_file, index=False)
    print(f"Saved seasonal data to: {seasonal_file}")

    # Summary
    print("\n" + "="*70)
    print("PROCESSING SUMMARY")
    print("="*70)
    print(f"Daily records: {len(daily):,}")
    print(f"Monthly records: {len(monthly):,}")
    print(f"Annual records: {len(annual_wide):,}")
    print(f"Seasonal records: {len(seasonal):,}")
    print(f"\nFiles created in: {PROCESSED_DIR}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    return {
        'daily': daily,
        'monthly': monthly,
        'annual': annual_wide,
        'seasonal': seasonal
    }

if __name__ == '__main__':
    data = main()
