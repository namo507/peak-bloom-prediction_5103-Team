#!/usr/bin/env python3
"""
Fix ONI (ENSO) climate index download
Previous version had parsing error - this fixes it
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

RAW_DIR = Path('data/raw/climate_indices')
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_oni_fixed():
    """
    Download and parse Oceanic Niño Index (ENSO) - FIXED VERSION

    The previous version failed because the file format has a header line
    with 'DJF' as a column label, not a data value.
    """

    url = 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt'
    print(f"Downloading ONI from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Read as text and parse manually
        lines = response.text.strip().split('\n')

        # Skip the header line (contains 'DJF', 'JFM', etc.)
        # Data starts from line 2
        data_lines = [line for line in lines[1:] if line.strip()]

        records = []

        for line in data_lines:
            # Split by whitespace
            parts = line.split()

            if len(parts) < 4:
                continue

            try:
                # Format is: SEASON YEAR TOTAL ANOM
                season = parts[0]
                year = int(parts[1])

                # Get the anomaly value (4th column)
                anom = float(parts[3])

                # Map 3-month season to representative month
                season_to_month = {
                    'DJF': 1,   # Dec-Jan-Feb -> January
                    'JFM': 2,   # Jan-Feb-Mar -> February
                    'FMA': 3,   # Feb-Mar-Apr -> March
                    'MAM': 4,   # Mar-Apr-May -> April
                    'AMJ': 5,   # Apr-May-Jun -> May
                    'MJJ': 6,   # May-Jun-Jul -> June
                    'JJA': 7,   # Jun-Jul-Aug -> July
                    'JAS': 8,   # Jul-Aug-Sep -> August
                    'ASO': 9,   # Aug-Sep-Oct -> September
                    'SON': 10,  # Sep-Oct-Nov -> October
                    'OND': 11,  # Oct-Nov-Dec -> November
                    'NDJ': 12   # Nov-Dec-Jan -> December
                }

                month = season_to_month.get(season)

                if month is not None:
                    records.append({
                        'year': year,
                        'month': month,
                        'season': season,
                        'ONI': anom
                    })

            except (ValueError, IndexError) as e:
                # Skip lines that don't parse correctly
                continue

        df = pd.DataFrame(records)

        if len(df) == 0:
            print(f"  ERROR: No data parsed")
            return None

        print(f"  ✓ Downloaded {len(df)} ONI records ({df['year'].min()}-{df['year'].max()})")

        return df

    except Exception as e:
        print(f"  ERROR downloading ONI: {e}")
        return None

def calculate_seasonal_oni(oni_df):
    """Calculate seasonal ONI averages"""

    seasonal_data = []

    for year in range(oni_df['year'].min(), oni_df['year'].max() + 1):
        record = {'year': year}

        # Winter (DJF) - previous Dec, current Jan-Feb
        winter_months = oni_df[
            ((oni_df['year'] == year - 1) & (oni_df['month'] == 12)) |
            ((oni_df['year'] == year) & (oni_df['month'].isin([1, 2])))
        ]

        if len(winter_months) > 0:
            record['ONI_winter'] = winter_months['ONI'].mean()

        # Spring (MAM)
        spring_months = oni_df[
            (oni_df['year'] == year) & (oni_df['month'].isin([3, 4, 5]))
        ]

        if len(spring_months) > 0:
            record['ONI_spring'] = spring_months['ONI'].mean()

        # Annual average
        annual = oni_df[oni_df['year'] == year]
        if len(annual) > 0:
            record['ONI_annual'] = annual['ONI'].mean()

        seasonal_data.append(record)

    return pd.DataFrame(seasonal_data)

def merge_with_existing_indices():
    """Merge ONI with existing climate indices"""

    existing_file = RAW_DIR / 'climate_indices_seasonal.csv'

    if not existing_file.exists():
        print(f"WARNING: {existing_file} not found")
        return None

    # Load existing indices
    existing = pd.read_csv(existing_file)
    print(f"\nLoaded existing climate indices: {len(existing)} records")
    print(f"Existing columns: {existing.columns.tolist()}")

    # Download ONI
    oni_monthly = download_oni_fixed()

    if oni_monthly is None:
        return None

    # Save monthly ONI
    oni_monthly_file = RAW_DIR / 'oni_monthly.csv'
    oni_monthly.to_csv(oni_monthly_file, index=False)
    print(f"Saved monthly ONI to: {oni_monthly_file}")

    # Calculate seasonal ONI
    oni_seasonal = calculate_seasonal_oni(oni_monthly)
    print(f"Calculated seasonal ONI: {len(oni_seasonal)} years")

    # Merge with existing indices
    merged = existing.merge(oni_seasonal, on='year', how='outer')

    # Sort by year
    merged = merged.sort_values('year')

    print(f"\nMerged dataset:")
    print(f"  Years: {merged['year'].min()}-{merged['year'].max()}")
    print(f"  Records: {len(merged)}")
    print(f"  Columns: {merged.columns.tolist()}")

    # Save updated file
    merged.to_csv(existing_file, index=False)
    print(f"\n✓ Updated climate indices saved to: {existing_file}")

    # Show sample
    print(f"\nSample data (recent years):")
    print(merged.tail(5))

    return merged

def main():
    """Main execution"""

    print("="*80)
    print("FIXING ONI (ENSO) CLIMATE INDEX DOWNLOAD")
    print("="*80)
    print()

    result = merge_with_existing_indices()

    if result is not None:
        print(f"\n{'='*80}")
        print("✅ SUCCESS! ONI climate index added")
        print(f"{'='*80}")

        # Check ONI coverage
        oni_coverage = (1 - result['ONI_winter'].isna().sum() / len(result)) * 100
        print(f"\nONI_winter coverage: {oni_coverage:.1f}%")
        print(f"ONI data from: {result[result['ONI_winter'].notna()]['year'].min()}")
        print(f"ONI data to: {result[result['ONI_winter'].notna()]['year'].max()}")

        return result
    else:
        print(f"\n{'='*80}")
        print("❌ FAILED to fix ONI download")
        print(f"{'='*80}")
        return None

if __name__ == '__main__':
    data = main()
