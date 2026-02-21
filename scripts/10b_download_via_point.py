#!/usr/bin/env python3
"""
Download weather for missing sites using Meteostat Point API (simplified)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from meteostat import Point, daily
from meteostat import config

# Allow large requests (>30 years)
config.block_large_requests = False

RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sites to download
SITES = {
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'alt': 370,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'alt': 4,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'alt': 8.5,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    }
}

print("="*80)
print("DOWNLOADING WEATHER FOR MISSING SITES (METEOSTAT POINT API)")
print("="*80)

all_data = []

# Load existing data
existing_file = RAW_DIR / 'openmeteo_all_sites.csv'
if existing_file.exists():
    existing = pd.read_csv(existing_file)
    all_data.append(existing)
    print(f"✓ Loaded existing data: {len(existing):,} records\n")

for location_name, config in SITES.items():
    print(f"\nProcessing {location_name}...")

    # Create Point
    point = Point(config['lat'], config['lon'], config['alt'])

    # Get daily data using function API
    df = daily(point, config['start'], config['end'])

    if len(df) == 0:
        print(f"  ⚠️  No data available")
        continue

    # Reset index
    df = df.reset_index()

    # Rename columns
    df = df.rename(columns={
        'time': 'date',
        'tavg': 'T_mean',
        'tmax': 'T_max',
        'tmin': 'T_min',
        'prcp': 'precip',
        'snow': 'snowfall'
    })

    # Add location info
    df['location'] = location_name
    df['lat'] = config['lat']
    df['lon'] = config['lon']
    df['source'] = 'meteostat'

    # Parse date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear

    print(f"  ✓ Downloaded {len(df):,} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Check coverage
    coverage = (1 - df['T_mean'].isna().sum() / len(df)) * 100
    print(f"  T_mean coverage: {coverage:.1f}%")

    # Save individual file
    output_file = RAW_DIR / f'meteostat_{location_name}.csv'
    df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")

    all_data.append(df)

# Combine all
if len(all_data) > 1:
    combined = pd.concat(all_data, ignore_index=True)

    # Save combined
    combined_file = RAW_DIR / 'all_sites_combined.csv'
    combined.to_csv(combined_file, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total sites: {combined['location'].nunique()}")
    print(f"Total records: {len(combined):,}")
    print(f"\nRecords per site:")
    print(combined['location'].value_counts().sort_index())
    print(f"\nSaved to: {combined_file}")
    print(f"\n✅ SUCCESS!")
    print(f"{'='*80}")
else:
    print(f"\n❌ No new data downloaded")
