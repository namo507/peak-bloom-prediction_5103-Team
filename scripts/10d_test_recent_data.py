#!/usr/bin/env python3
"""
Test downloading recent data to see if Meteostat works at all
"""

import pandas as pd
from datetime import datetime
from meteostat import stations, daily, Point, config

# Allow large requests
config.block_large_requests = False

# Test with recent date range (last 10 years)
start = datetime(2015, 1, 1)
end = datetime(2024, 12, 31)

# Test Vancouver
lat, lon = 49.2827, -123.1207
point = Point(lat, lon)

print("Finding stations near Vancouver...")
nearby = stations.nearby(point, radius=50000, limit=5)

if len(nearby) > 0:
    print(f"Found {len(nearby)} stations")

    # Try the top 3 stations
    for i in range(min(3, len(nearby))):
        station = nearby.iloc[i]
        station_id = nearby.index[i]

        print(f"\n{i+1}. Testing station: {station.get('name', 'Unknown')} ({station_id})")
        print(f"   Distance: {station.get('distance', 0)/1000:.1f} km")

        try:
            df = daily(station_id, start, end)
            print(f"   Retrieved {len(df)} records")

            if len(df) > 0:
                df = df.reset_index()
                print(f"   Date range: {df['time'].min()} to {df['time'].max()}")

                # Check coverage
                if 'tavg' in df.columns:
                    coverage = (1 - df['tavg'].isna().sum() / len(df)) * 100
                    print(f"   Tavg coverage: {coverage:.1f}%")

                    # Show sample
                    print("\n   Sample data:")
                    print(df[['time', 'tavg', 'tmax', 'tmin', 'prcp']].head())

                    print(f"\n   ✅ Station {station_id} has good data!")
                    break
        except Exception as e:
            print(f"   ERROR: {e}")
else:
    print("No stations found")
