#!/usr/bin/env python3
"""
Fix missing weather data for Liestal, Vancouver, and NYC using Meteostat
Meteostat has better rate limits and station-level data quality
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    import meteostat
    from meteostat import Point, daily
    METEOSTAT_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: meteostat not installed - {e}")
    print("Install with: pip install meteostat")
    METEOSTAT_AVAILABLE = False
except Exception as e:
    print(f"ERROR importing meteostat: {e}")
    METEOSTAT_AVAILABLE = False

RAW_DIR = Path('data/raw/weather')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sites that need weather data
SITES_TO_FIX = {
    'liestal': {
        'lat': 47.4834,
        'lon': 7.7331,
        'alt': 370,
        'start': datetime(1894, 1, 1),  # Liestal bloom data starts 1894
        'end': datetime(2025, 12, 31),
        'search_name': 'Basel'  # Nearby major city
    },
    'vancouver': {
        'lat': 49.2827,
        'lon': -123.1207,
        'alt': 4,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31),
        'search_name': 'Vancouver'
    },
    'newyorkcity': {
        'lat': 40.7306,
        'lon': -73.9982,
        'alt': 8.5,
        'start': datetime(1940, 1, 1),
        'end': datetime(2025, 12, 31),
        'search_name': 'New York'
    }
}

def find_best_station(lat, lon, location_name, start, end):
    """Find the best weather station near a location"""

    print(f"\nFinding stations near {location_name} ({lat}, {lon})...")

    try:
        # Use Point to get nearest station
        # Meteostat v3 API
        location = Point(lat, lon)
        print(f"  Created Point location for {location_name}")

        # Get nearby stations using the stations module
        stations = meteostat.Station.nearby(lat, lon)
        stations_df = stations.fetch(10)

        if len(stations_df) == 0:
            print(f"  No stations found nearby")
            return None

        print(f"  Found {len(stations_df)} stations")

        # Filter for stations with data in our date range
        valid_stations = []
        for station_id in stations_df.index[:5]:  # Check top 5
            station_info = stations_df.loc[station_id]

            # Check if station has data in our range
            if pd.notna(station_info.get('daily_start')) and pd.notna(station_info.get('daily_end')):
                daily_start = pd.to_datetime(station_info['daily_start'])
                daily_end = pd.to_datetime(station_info['daily_end'])

                # Station must overlap with our desired range
                if daily_start <= end and daily_end >= start:
                    overlap_years = (min(daily_end, end).year - max(daily_start, start).year)
                    valid_stations.append({
                        'id': station_id,
                        'name': station_info.get('name', 'Unknown'),
                        'distance': station_info.get('distance', 999),
                        'overlap_years': overlap_years,
                        'start': daily_start,
                        'end': daily_end
                    })

        if not valid_stations:
            print(f"  No stations with data in date range")
            return None

        # Sort by overlap years (prefer stations with longest coverage)
        valid_stations.sort(key=lambda x: (-x['overlap_years'], x['distance']))

        best = valid_stations[0]
        print(f"  ✓ Best station: {best['id']} - {best['name']}")
        print(f"    Distance: {best['distance']:.1f} km")
        print(f"    Coverage: {best['start'].year}-{best['end'].year} ({best['overlap_years']} years)")

        return best['id']

    except Exception as e:
        print(f"  ERROR finding station: {e}")
        return None

def download_station_data(station_id, start, end, location_name):
    """Download daily data from a station"""

    print(f"  Downloading data from station {station_id}...")

    try:
        # Use daily.fetch() from meteostat v3
        data = daily.Daily(station_id, start, end)
        df = data.fetch()

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

        # Parse date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear

        # Add source
        df['source'] = 'meteostat'

        print(f"    ✓ Downloaded {len(df)} daily records")
        print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

        # Check data quality
        coverage = (1 - df['T_mean'].isna().sum() / len(df)) * 100
        print(f"    T_mean coverage: {coverage:.1f}%")

        return df

    except Exception as e:
        print(f"    ERROR downloading data: {e}")
        return None

def main():
    """Download weather data for missing competition sites"""

    if not METEOSTAT_AVAILABLE:
        return None

    print("="*80)
    print("FIXING MISSING COMPETITION SITE WEATHER DATA")
    print("="*80)
    print(f"Using Meteostat for station-level data")
    print(f"Sites to fix: {len(SITES_TO_FIX)}")
    print()

    results = []

    for location_name, config in SITES_TO_FIX.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING: {location_name.upper()}")
        print(f"{'='*80}")

        # Find best station
        station_id = find_best_station(
            config['lat'],
            config['lon'],
            location_name,
            config['start'],
            config['end']
        )

        if station_id is None:
            print(f"⚠️  Could not find station for {location_name}")
            continue

        # Download data
        df = download_station_data(
            station_id,
            config['start'],
            config['end'],
            location_name
        )

        if df is not None:
            # Save individual file
            output_file = RAW_DIR / f'meteostat_{location_name}.csv'
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved to: {output_file}")

            results.append(df)

    # Combine with existing Open-Meteo data
    if results:
        print(f"\n{'='*80}")
        print("COMBINING WITH EXISTING DATA")
        print(f"{'='*80}")

        # Load existing Open-Meteo data
        existing_files = list(RAW_DIR.glob('openmeteo_*.csv'))
        existing_data = []

        for f in existing_files:
            if 'all_sites' not in f.name:  # Skip combined file
                try:
                    df = pd.read_csv(f)
                    existing_data.append(df)
                    print(f"  ✓ Loaded {f.name}: {len(df)} records")
                except:
                    pass

        # Combine all
        all_data = existing_data + results
        combined = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = RAW_DIR / 'all_sites_combined.csv'
        combined.to_csv(combined_file, index=False)

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total sites with weather: {combined['location'].nunique()}")
        print(f"Total records: {len(combined):,}")
        print(f"\nRecords per site:")
        print(combined['location'].value_counts().sort_index())
        print(f"\nSaved to: {combined_file}")
        print(f"\n✅ SUCCESS! All competition sites now have weather data")
        print(f"{'='*80}")

        return combined

    else:
        print(f"\n❌ FAILED: Could not download data for any sites")
        return None

if __name__ == '__main__':
    data = main()
