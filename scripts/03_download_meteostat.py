#!/usr/bin/env python3
"""
Download station-level weather data from Meteostat
Higher quality than reanalysis for urban sites
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Check if meteostat is installed
try:
    from meteostat import Point, Daily, Stations
    METEOSTAT_AVAILABLE = True
except ImportError:
    print("WARNING: meteostat package not installed")
    print("Install with: pip install meteostat")
    METEOSTAT_AVAILABLE = False

# Configuration
RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Competition sites with approximate station coordinates
SITES = {
    'kyoto': {
        'lat': 35.0119831,
        'lon': 135.6761135,
        'alt': 44,
        'search_radius': 50,  # km
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'washingtondc': {
        'lat': 38.8853,
        'lon': -77.0386,
        'alt': 2.7,
        'search_radius': 30,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'alt': 370,
        'search_radius': 30,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'alt': 4,
        'search_radius': 30,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'alt': 8.5,
        'search_radius': 30,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31)
    }
}

def find_nearby_stations(lat, lon, radius_km=50):
    """
    Find weather stations near a location

    Parameters:
    -----------
    lat, lon : float
        Coordinates
    radius_km : int
        Search radius in kilometers

    Returns:
    --------
    pandas.DataFrame of nearby stations
    """

    if not METEOSTAT_AVAILABLE:
        return None

    try:
        # Find stations nearby
        stations = Stations()
        stations = stations.nearby(lat, lon)
        stations_df = stations.fetch(limit=10)

        if len(stations_df) > 0:
            return stations_df
        else:
            print(f"  No stations found near ({lat}, {lon})")
            return None

    except Exception as e:
        print(f"  ERROR finding stations: {e}")
        return None

def download_station_data(station_id, start, end, location_name):
    """
    Download daily data from a specific station

    Parameters:
    -----------
    station_id : str
        Meteostat station ID
    start, end : datetime
        Date range
    location_name : str
        Location name for logging

    Returns:
    --------
    pandas.DataFrame with daily data
    """

    if not METEOSTAT_AVAILABLE:
        return None

    try:
        # Get daily data
        data = Daily(station_id, start, end)
        df = data.fetch()

        if len(df) > 0:
            # Reset index to get date as column
            df = df.reset_index()

            # Rename columns
            df = df.rename(columns={
                'time': 'date',
                'tavg': 'T_mean',
                'tmax': 'T_max',
                'tmin': 'T_min',
                'prcp': 'precip',
                'snow': 'snowfall',
                'wspd': 'wind_speed'
            })

            # Add location info
            df['location'] = location_name
            df['station_id'] = station_id

            # Parse date
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear

            return df
        else:
            return None

    except Exception as e:
        print(f"  ERROR downloading station {station_id}: {e}")
        return None

def download_meteostat_for_location(location_name, config):
    """
    Download Meteostat data for a location

    Finds nearby stations and downloads the one with best coverage
    """

    if not METEOSTAT_AVAILABLE:
        return None

    print(f"Processing {location_name}...")

    # Find nearby stations
    stations = find_nearby_stations(
        config['lat'],
        config['lon'],
        config.get('search_radius', 50)
    )

    if stations is None or len(stations) == 0:
        print(f"  No stations found for {location_name}")
        return None

    print(f"  Found {len(stations)} nearby stations")
    print(f"  Top station: {stations.index[0]} - {stations.iloc[0].get('name', 'Unknown')}")

    # Try to download from the best station
    best_station_id = stations.index[0]

    df = download_station_data(
        best_station_id,
        config['start'],
        config['end'],
        location_name
    )

    if df is not None:
        print(f"  ✓ Downloaded {len(df)} records from station {best_station_id}")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"    Coverage: {(1 - df['T_mean'].isna().sum() / len(df)) * 100:.1f}%")

        return df
    else:
        print(f"  ERROR: No data retrieved for {location_name}")
        return None

def main():
    """Download Meteostat data for all sites"""

    if not METEOSTAT_AVAILABLE:
        print("ERROR: meteostat package not installed")
        print("Install with: pip install meteostat")
        return None

    print("="*70)
    print("METEOSTAT STATION DATA DOWNLOAD")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sites to download: {len(SITES)}")
    print()

    all_data = []

    for location_name, config in SITES.items():
        df = download_meteostat_for_location(location_name, config)

        if df is not None:
            # Save individual file
            output_file = RAW_DIR / f'meteostat_{location_name}.csv'
            df.to_csv(output_file, index=False)
            print(f"  Saved to: {output_file}")

            all_data.append(df)

        print()

    # Combine all sites
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = RAW_DIR / 'meteostat_all_sites.csv'
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

        # Check coverage
        print(f"\nData coverage:")
        for col in ['T_mean', 'T_max', 'T_min', 'precip']:
            coverage = (1 - combined[col].isna().sum() / len(combined)) * 100
            print(f"  {col}: {coverage:.1f}%")

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        return combined
    else:
        print("ERROR: No data downloaded successfully")
        return None

if __name__ == '__main__':
    data = main()
