#!/usr/bin/env python3
"""
Download Vancouver weather data only
"""

import requests
import pandas as pd
from pathlib import Path
import time

RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

VANCOUVER_CONFIG = {
    'lat': 49.2827,
    'lon': -123.1207,
    'start': '1940-01-01',
    'end': '2025-12-31'
}

print("="*80)
print("DOWNLOADING VANCOUVER WEATHER DATA")
print("="*80)
print("Waiting 60 seconds to ensure rate limit has reset...")
time.sleep(60)

print("\nProcessing vancouver...")

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": VANCOUVER_CONFIG['lat'],
    "longitude": VANCOUVER_CONFIG['lon'],
    "start_date": VANCOUVER_CONFIG['start'],
    "end_date": VANCOUVER_CONFIG['end'],
    "daily": [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "snowfall_sum"
    ],
    "timezone": "UTC"
}

try:
    print(f"  Requesting data from Open-Meteo...")
    response = requests.get(url, params=params, timeout=60)

    if response.status_code == 429:
        print(f"  ⚠️  Rate limit still exceeded")
        print(f"  Please wait and run this script again later")
        exit(1)

    response.raise_for_status()
    data = response.json()

    if 'daily' not in data:
        print(f"  ERROR: No daily data in response")
        exit(1)

    daily = data['daily']

    # Create DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(daily['time']),
        'T_mean': daily['temperature_2m_mean'],
        'T_max': daily['temperature_2m_max'],
        'T_min': daily['temperature_2m_min'],
        'precip': daily['precipitation_sum'],
        'snowfall': daily['snowfall_sum']
    })

    # Add location info
    df['location'] = 'vancouver'
    df['lat'] = VANCOUVER_CONFIG['lat']
    df['lon'] = VANCOUVER_CONFIG['lon']
    df['source'] = 'openmeteo_archive'

    # Parse date components
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
    output_file = RAW_DIR / 'openmeteo_vancouver.csv'
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}")

    # Combine with existing data
    all_data = []

    existing_file = RAW_DIR / 'all_sites_combined.csv'
    if existing_file.exists():
        existing = pd.read_csv(existing_file)
        all_data.append(existing)
        print(f"\n✓ Loaded existing data: {len(existing):,} records")

    all_data.append(df)
    combined = pd.concat(all_data, ignore_index=True)

    # Save combined
    combined.to_csv(existing_file, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total sites in dataset: {combined['location'].nunique()}")
    print(f"Total records: {len(combined):,}")
    print(f"\nRecords per site:")
    print(combined['location'].value_counts().sort_index())
    print(f"\nSaved to: {existing_file}")
    print(f"\n✅ SUCCESS! Vancouver data downloaded")
    print(f"{'='*80}")

except requests.exceptions.RequestException as e:
    print(f"  ERROR: {e}")
    exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    exit(1)
