#!/usr/bin/env python3
"""
Download historical weather data from Open-Meteo API
Retrieves daily temperature and precipitation for all competition sites
"""

import requests
import pandas as pd
from pathlib import Path
import time
from datetime import datetime

# Configuration
RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Competition sites with coordinates
SITES = {
    'kyoto': {
        'lat': 35.0119831,
        'lon': 135.6761135,
        'alt': 44,
        'start_date': '1940-01-01',  # ERA5 starts 1940
        'end_date': '2025-12-31'
    },
    'washingtondc': {
        'lat': 38.8853,
        'lon': -77.0386,
        'alt': 2.7,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'alt': 370,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'alt': 4,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'alt': 8.5,
        'start_date': '1940-01-01',
        'end_date': '2025-12-31'
    }
}

def download_openmeteo_weather(lat, lon, start_date, end_date, location_name):
    """
    Download daily weather data from Open-Meteo Historical Weather API

    Parameters:
    -----------
    lat, lon : float
        Coordinates
    start_date, end_date : str
        Date range in YYYY-MM-DD format
    location_name : str
        Name of location for logging

    Returns:
    --------
    pandas.DataFrame with daily weather data
    """

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

    print(f"Downloading {location_name} ({start_date} to {end_date})...")

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()

        # Check if we got data
        if 'daily' not in data:
            print(f"  ERROR: No daily data in response for {location_name}")
            print(f"  Response: {data}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data['daily'])

        # Rename columns for clarity
        df = df.rename(columns={
            'time': 'date',
            'temperature_2m_max': 'T_max',
            'temperature_2m_min': 'T_min',
            'temperature_2m_mean': 'T_mean',
            'precipitation_sum': 'precip',
            'snowfall_sum': 'snowfall'
        })

        # Add location info
        df['location'] = location_name
        df['lat'] = lat
        df['lon'] = lon

        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear

        print(f"  ✓ Downloaded {len(df)} records for {location_name}")

        return df

    except requests.exceptions.RequestException as e:
        print(f"  ERROR downloading {location_name}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR processing {location_name}: {e}")
        return None

def main():
    """Download weather data for all sites"""

    print("="*70)
    print("OPEN-METEO HISTORICAL WEATHER DOWNLOAD")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sites to download: {len(SITES)}")
    print()

    all_data = []

    for location_name, config in SITES.items():
        # Download
        df = download_openmeteo_weather(
            lat=config['lat'],
            lon=config['lon'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            location_name=location_name
        )

        if df is not None:
            # Save individual file
            output_file = RAW_DIR / f'openmeteo_{location_name}.csv'
            df.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")

            all_data.append(df)

        # Be nice to the API - longer delay to avoid rate limits
        time.sleep(5)  # Increased from 1 to 5 seconds
        print()

    # Combine all sites
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = RAW_DIR / 'openmeteo_all_sites.csv'
        combined.to_csv(combined_file, index=False)

        print("="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Total records downloaded: {len(combined):,}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Locations: {combined['location'].nunique()}")
        print(f"\nRecords per location:")
        print(combined['location'].value_counts().sort_index())
        print(f"\nCombined file saved to: {combined_file}")
        print(f"\nFile size: {combined_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Check for missing values
        print(f"\nMissing values:")
        missing = combined[['T_max', 'T_min', 'T_mean', 'precip']].isnull().sum()
        print(missing)

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        return combined
    else:
        print("ERROR: No data downloaded successfully")
        return None

if __name__ == '__main__':
    data = main()
