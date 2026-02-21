#!/usr/bin/env python3
"""
Download weather for missing sites using Meteostat Stations API
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from meteostat import Point, stations, daily, config

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

def find_best_station(lat, lon, location_name):
    """Find the best weather station near a location"""

    print(f"  Finding stations near {location_name}...")

    try:
        # Create point and find nearby stations
        point = Point(lat, lon)
        nearby = stations.nearby(point, radius=50000, limit=20)  # 50km radius, top 20

        if len(nearby) == 0:
            print(f"    No stations found within 50km")
            return None

        print(f"    Found {len(nearby)} stations within 50km")

        # Show top 3 stations
        for i in range(min(3, len(nearby))):
            station = nearby.iloc[i]
            print(f"    {i+1}. {station.get('name', 'Unknown')} (ID: {station.name})")
            print(f"       Distance: {station.get('distance', 0)/1000:.1f} km")

        # Return the closest station ID
        best_station_id = nearby.index[0]
        print(f"    Using station: {best_station_id}")
        return best_station_id

    except Exception as e:
        print(f"    ERROR finding stations: {e}")
        return None

def download_station_data(station_id, start, end, location_name):
    """Download daily data from a station"""

    print(f"  Downloading data from station {station_id}...")

    try:
        # Use daily function with station ID
        df = daily(station_id, start, end)

        if len(df) == 0:
            print(f"    No data retrieved")
            return None

        # Reset index to get date as column
        df = df.reset_index()

        # Rename columns to match our schema
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
        df['station_id'] = station_id
        df['source'] = 'meteostat'

        # Parse date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear

        print(f"    ✓ Downloaded {len(df):,} daily records")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

        # Check data quality
        coverage = (1 - df['T_mean'].isna().sum() / len(df)) * 100
        print(f"    T_mean coverage: {coverage:.1f}%")

        return df

    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        return None

print("="*80)
print("DOWNLOADING WEATHER FOR MISSING SITES (METEOSTAT STATIONS API)")
print("="*80)

all_data = []

# Load existing data
existing_file = RAW_DIR / 'openmeteo_all_sites.csv'
if existing_file.exists():
    existing = pd.read_csv(existing_file)
    all_data.append(existing)
    print(f"✓ Loaded existing data: {len(existing):,} records\n")

for location_name, config_data in SITES.items():
    print(f"\nProcessing {location_name}...")

    # Find best station
    station_id = find_best_station(
        config_data['lat'],
        config_data['lon'],
        location_name
    )

    if station_id is None:
        print(f"  ⚠️  Could not find station for {location_name}")
        continue

    # Download data
    df = download_station_data(
        station_id,
        config_data['start'],
        config_data['end'],
        location_name
    )

    if df is not None:
        # Save individual file
        output_file = RAW_DIR / f'meteostat_{location_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to: {output_file}")

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
