#!/usr/bin/env python3
"""
Download climate indices from NOAA Physical Sciences Laboratory
Retrieves ENSO, NAO, PDO, AO, and AMO indices
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import io

# Configuration
RAW_DIR = Path('data/raw/climate_indices')
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Climate index sources
INDICES = {
    'ONI': {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'description': 'Oceanic Niño Index (ENSO)',
        'parser': 'oni'
    },
    'NAO': {
        'url': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table',
        'description': 'North Atlantic Oscillation',
        'parser': 'nao'
    },
    'PDO': {
        'url': 'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat',
        'description': 'Pacific Decadal Oscillation',
        'parser': 'pdo'
    },
    'AO': {
        'url': 'https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/monthly.ao.index.b50.current.ascii.table',
        'description': 'Arctic Oscillation',
        'parser': 'ao'
    }
}

def download_oni():
    """Download and parse Oceanic Niño Index (ENSO)"""

    url = INDICES['ONI']['url']
    print(f"Downloading ONI from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the fixed-width format
        lines = response.text.strip().split('\n')

        # Skip header (first line)
        data_lines = [line for line in lines[1:] if line.strip()]

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                year = int(parts[0])
                season = parts[1]

                # Extract month from season (DJF -> 12, JFM -> 1, etc.)
                season_to_month = {
                    'DJF': 12, 'JFM': 1, 'FMA': 2, 'MAM': 3,
                    'AMJ': 4, 'MJJ': 5, 'JJA': 6, 'JAS': 7,
                    'ASO': 8, 'SON': 9, 'OND': 10, 'NDJ': 11
                }
                month = season_to_month.get(season, 1)

                # Get anomaly value
                if len(parts) >= 3:
                    try:
                        value = float(parts[2])
                        records.append({
                            'year': year,
                            'month': month,
                            'season': season,
                            'ONI': value
                        })
                    except ValueError:
                        continue

        df = pd.DataFrame(records)
        print(f"  ✓ Downloaded {len(df)} ONI records ({df['year'].min()}-{df['year'].max()})")

        return df

    except Exception as e:
        print(f"  ERROR downloading ONI: {e}")
        return None

def download_nao():
    """Download and parse North Atlantic Oscillation"""

    url = INDICES['NAO']['url']
    print(f"Downloading NAO from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the table format
        lines = response.text.strip().split('\n')

        # Skip header
        data_lines = [line for line in lines[1:] if line.strip() and not line.startswith('year')]

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                try:
                    year = int(parts[0])
                    for month_idx in range(12):
                        value = float(parts[month_idx + 1])
                        if value != -99.90:  # Missing value flag
                            records.append({
                                'year': year,
                                'month': month_idx + 1,
                                'NAO': value
                            })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(records)
        print(f"  ✓ Downloaded {len(df)} NAO records ({df['year'].min()}-{df['year'].max()})")

        return df

    except Exception as e:
        print(f"  ERROR downloading NAO: {e}")
        return None

def download_pdo():
    """Download and parse Pacific Decadal Oscillation"""

    url = INDICES['PDO']['url']
    print(f"Downloading PDO from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the data format
        lines = response.text.strip().split('\n')

        # Skip header
        data_lines = [line for line in lines[1:] if line.strip() and not line.startswith('YEAR')]

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                try:
                    year = int(parts[0])
                    for month_idx in range(12):
                        value_str = parts[month_idx + 1]
                        if value_str != '99.99':  # Missing value flag
                            value = float(value_str)
                            records.append({
                                'year': year,
                                'month': month_idx + 1,
                                'PDO': value
                            })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(records)
        print(f"  ✓ Downloaded {len(df)} PDO records ({df['year'].min()}-{df['year'].max()})")

        return df

    except Exception as e:
        print(f"  ERROR downloading PDO: {e}")
        return None

def download_ao():
    """Download and parse Arctic Oscillation"""

    url = INDICES['AO']['url']
    print(f"Downloading AO from {url}...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the table format
        lines = response.text.strip().split('\n')

        # Skip header
        data_lines = [line for line in lines[1:] if line.strip() and not line.startswith('year')]

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                try:
                    year = int(parts[0])
                    for month_idx in range(12):
                        value = float(parts[month_idx + 1])
                        if value != -99.90:  # Missing value flag
                            records.append({
                                'year': year,
                                'month': month_idx + 1,
                                'AO': value
                            })
                except (ValueError, IndexError):
                    continue

        df = pd.DataFrame(records)
        print(f"  ✓ Downloaded {len(df)} AO records ({df['year'].min()}-{df['year'].max()})")

        return df

    except Exception as e:
        print(f"  ERROR downloading AO: {e}")
        return None

def calculate_seasonal_indices(df_dict):
    """
    Calculate seasonal averages for climate indices

    Winter: Dec-Jan-Feb (for bloom prediction)
    Spring: Mar-Apr-May
    """

    seasonal_data = []

    for year in range(1948, 2026):
        record = {'year': year}

        # For each index
        for index_name, df in df_dict.items():
            if df is None:
                continue

            # Winter (DJF) - use previous Dec, current Jan-Feb
            winter_months = df[
                ((df['year'] == year - 1) & (df['month'] == 12)) |
                ((df['year'] == year) & (df['month'].isin([1, 2])))
            ]

            if len(winter_months) > 0:
                record[f'{index_name}_winter'] = winter_months[index_name].mean()

            # Spring (MAM)
            spring_months = df[
                (df['year'] == year) & (df['month'].isin([3, 4, 5]))
            ]

            if len(spring_months) > 0:
                record[f'{index_name}_spring'] = spring_months[index_name].mean()

            # Annual average
            annual = df[df['year'] == year]
            if len(annual) > 0:
                record[f'{index_name}_annual'] = annual[index_name].mean()

        seasonal_data.append(record)

    return pd.DataFrame(seasonal_data)

def main():
    """Download all climate indices"""

    print("="*70)
    print("NOAA CLIMATE INDICES DOWNLOAD")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Download each index
    downloaded = {}

    # ONI (ENSO)
    oni_df = download_oni()
    if oni_df is not None:
        oni_file = RAW_DIR / 'oni_monthly.csv'
        oni_df.to_csv(oni_file, index=False)
        print(f"  Saved to: {oni_file}\n")
        downloaded['ONI'] = oni_df

    # NAO
    nao_df = download_nao()
    if nao_df is not None:
        nao_file = RAW_DIR / 'nao_monthly.csv'
        nao_df.to_csv(nao_file, index=False)
        print(f"  Saved to: {nao_file}\n")
        downloaded['NAO'] = nao_df

    # PDO
    pdo_df = download_pdo()
    if pdo_df is not None:
        pdo_file = RAW_DIR / 'pdo_monthly.csv'
        pdo_df.to_csv(pdo_file, index=False)
        print(f"  Saved to: {pdo_file}\n")
        downloaded['PDO'] = pdo_df

    # AO
    ao_df = download_ao()
    if ao_df is not None:
        ao_file = RAW_DIR / 'ao_monthly.csv'
        ao_df.to_csv(ao_file, index=False)
        print(f"  Saved to: {ao_file}\n")
        downloaded['AO'] = ao_df

    # Calculate seasonal averages
    if downloaded:
        print("Calculating seasonal averages...")
        seasonal = calculate_seasonal_indices(downloaded)

        seasonal_file = RAW_DIR / 'climate_indices_seasonal.csv'
        seasonal.to_csv(seasonal_file, index=False)

        print("="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Indices downloaded: {len(downloaded)}")
        print(f"Seasonal data: {len(seasonal)} years ({seasonal['year'].min()}-{seasonal['year'].max()})")
        print(f"Seasonal file: {seasonal_file}")
        print(f"\nSample data (recent years):")
        print(seasonal.tail(5))
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        return seasonal
    else:
        print("ERROR: No indices downloaded successfully")
        return None

if __name__ == '__main__':
    data = main()
