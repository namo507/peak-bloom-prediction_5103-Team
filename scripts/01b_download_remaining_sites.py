#!/usr/bin/env python3
"""
Download remaining weather sites with longer delays to avoid rate limits
"""

import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Only the sites we haven't downloaded yet
SITES = {
    'washingtondc': {
        'lat': 38.8853,
        'lon': -77.0386,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    }
}

def download_openmeteo_weather(lat, lon, start_date, end_date, location_name):
    """Download daily weather data from Open-Meteo"""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "snowfall_sum"
        ],
        "timezone": "UTC"
    }

    print(f"Downloading {location_name}...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        if 'daily' not in data:
            print(f"  ERROR: No data for {location_name}")
            return None

        df = pd.DataFrame(data['daily'])
        df = df.rename(columns={
            'time': 'date',
            'temperature_2m_max': 'T_max',
            'temperature_2m_min': 'T_min',
            'temperature_2m_mean': 'T_mean',
            'precipitation_sum': 'precip',
            'snowfall_sum': 'snowfall'
        })

        df['location'] = location_name
        df['lat'] = lat
        df['lon'] = lon
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear

        print(f"  ✓ Downloaded {len(df)} records")

        return df

    except requests.exceptions.HTTPError as e:
        if '429' in str(e):
            print(f"  ⚠️  Rate limit hit for {location_name} - need to wait")
        else:
            print(f"  ERROR: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    print("="*70)
    print("DOWNLOADING REMAINING SITES (with 10-second delays)")
    print("="*70)
    print()

    all_data = []

    # Load Kyoto data if it exists
    kyoto_file = RAW_DIR / 'openmeteo_kyoto.csv'
    if kyoto_file.exists():
        kyoto = pd.read_csv(kyoto_file)
        all_data.append(kyoto)
        print("✓ Loaded existing Kyoto data")
        print()

    for location_name, config in SITES.items():
        df = download_openmeteo_weather(
            lat=config['lat'],
            lon=config['lon'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            location_name=location_name
        )

        if df is not None:
            output_file = RAW_DIR / f'openmeteo_{location_name}.csv'
            df.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")
            all_data.append(df)

        # Long delay to avoid rate limits
        print(f"  Waiting 10 seconds before next download...")
        time.sleep(10)
        print()

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_file = RAW_DIR / 'openmeteo_all_sites.csv'
        combined.to_csv(combined_file, index=False)

        print("="*70)
        print(f"✓ Downloaded {len(all_data)} sites total")
        print(f"✓ Total records: {len(combined):,}")
        print(f"✓ Saved to: {combined_file}")
        print("="*70)

        return combined
    else:
        print("ERROR: No new data downloaded")
        return None

if __name__ == '__main__':
    main()
