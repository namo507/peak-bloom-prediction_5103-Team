#!/usr/bin/env python3
"""
Retry Open-Meteo API for missing sites with conservative rate limiting
"""

import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sites to download
SITES = {
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'start': '1940-01-01',
        'end': '2025-12-31'
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'start': '1940-01-01',
        'end': '2025-12-31'
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'start': '1940-01-01',
        'end': '2025-12-31'
    }
}

def download_openmeteo(location_name, lat, lon, start_date, end_date):
    """Download weather data from Open-Meteo Archive API"""

    print(f"\nProcessing {location_name}...")
    print(f"  Coordinates: ({lat}, {lon})")

    # Use archive API which has better historical coverage
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
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
            print(f"  ⚠️  Rate limit exceeded - please wait and try again later")
            return None

        response.raise_for_status()
        data = response.json()

        # Extract daily data
        if 'daily' not in data:
            print(f"  ERROR: No daily data in response")
            return None

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
        df['location'] = location_name
        df['lat'] = lat
        df['lon'] = lon
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

        return df

    except requests.exceptions.RequestException as e:
        print(f"  ERROR: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

print("="*80)
print("DOWNLOADING WEATHER FOR MISSING SITES (OPEN-METEO ARCHIVE API)")
print("="*80)
print("Using archive-api.open-meteo.com with 30-second delays between requests")
print()

all_data = []

# Load existing data
existing_file = RAW_DIR / 'openmeteo_all_sites.csv'
if existing_file.exists():
    existing = pd.read_csv(existing_file)
    all_data.append(existing)
    print(f"✓ Loaded existing data: {len(existing):,} records")
    print(f"  Existing locations: {sorted(existing['location'].unique())}")

success_count = 0

for location_name, config in SITES.items():
    # Download data
    df = download_openmeteo(
        location_name,
        config['lat'],
        config['lon'],
        config['start'],
        config['end']
    )

    if df is not None:
        # Save individual file
        output_file = RAW_DIR / f'openmeteo_{location_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")

        all_data.append(df)
        success_count += 1

        # Wait 30 seconds before next request to avoid rate limiting
        if location_name != list(SITES.keys())[-1]:  # Don't wait after last request
            print(f"\n  Waiting 30 seconds before next request...")
            time.sleep(30)
    else:
        print(f"  ⚠️  Failed to download {location_name}")
        # Still wait before next attempt
        if location_name != list(SITES.keys())[-1]:
            print(f"\n  Waiting 30 seconds before next request...")
            time.sleep(30)

# Combine all
if len(all_data) > 1:
    combined = pd.concat(all_data, ignore_index=True)

    # Save combined
    combined_file = RAW_DIR / 'all_sites_combined.csv'
    combined.to_csv(combined_file, index=False)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully downloaded: {success_count}/{len(SITES)} sites")
    print(f"Total sites in dataset: {combined['location'].nunique()}")
    print(f"Total records: {len(combined):,}")
    print(f"\nRecords per site:")
    print(combined['location'].value_counts().sort_index())
    print(f"\nSaved to: {combined_file}")

    if success_count == len(SITES):
        print(f"\n✅ SUCCESS! All sites downloaded")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS - {len(SITES) - success_count} sites failed")
    print(f"{'='*80}")
else:
    print(f"\n❌ No new data downloaded")
