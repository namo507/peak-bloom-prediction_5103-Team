#!/usr/bin/env python3
"""
Create final master dataset by merging all data sources
Combines bloom data, weather, GDD, chilling hours, and climate indices
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration
DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
FINAL_DIR = DATA_DIR / 'final'
FINAL_DIR.mkdir(parents=True, exist_ok=True)

def load_bloom_data():
    """Load existing bloom observation data"""

    print("Loading bloom observation data...")

    files = [
        DATA_DIR / 'kyoto.csv',
        DATA_DIR / 'washingtondc.csv',
        DATA_DIR / 'liestal.csv',
        DATA_DIR / 'vancouver.csv',
        DATA_DIR / 'nyc.csv',
        DATA_DIR / 'japan.csv',
        DATA_DIR / 'meteoswiss.csv',
        DATA_DIR / 'south_korea.csv'
    ]

    dfs = []
    for file in files:
        if file.exists():
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  ✓ Loaded {file.name}: {len(df)} records")

    bloom = pd.concat(dfs, ignore_index=True)

    # Standardize columns
    if 'bloom_doy' not in bloom.columns and 'bloom_date' in bloom.columns:
        bloom['date'] = pd.to_datetime(bloom['bloom_date'])
        bloom['bloom_doy'] = bloom['date'].dt.dayofyear

    # Ensure year is integer
    bloom['year'] = bloom['year'].astype(int)

    print(f"\nTotal bloom records: {len(bloom):,}")
    print(f"Year range: {bloom['year'].min()} - {bloom['year'].max()}")
    print(f"Unique locations: {bloom['location'].nunique()}")

    return bloom

def load_weather_features():
    """Load processed weather features"""

    print("\nLoading weather features...")

    # Seasonal weather
    seasonal_file = PROCESSED_DIR / 'weather_seasonal.csv'
    if seasonal_file.exists():
        seasonal = pd.read_csv(seasonal_file)
        print(f"  ✓ Loaded seasonal weather: {len(seasonal)} records")
    else:
        print(f"  WARNING: {seasonal_file} not found")
        seasonal = None

    # GDD features
    gdd_file = PROCESSED_DIR / 'gdd_features.csv'
    if gdd_file.exists():
        gdd = pd.read_csv(gdd_file)
        print(f"  ✓ Loaded GDD features: {len(gdd)} records")
    else:
        print(f"  WARNING: {gdd_file} not found")
        gdd = None

    # Chilling hours
    chill_file = PROCESSED_DIR / 'chill_features.csv'
    if chill_file.exists():
        chill = pd.read_csv(chill_file)
        print(f"  ✓ Loaded chilling hours: {len(chill)} records")
    else:
        print(f"  WARNING: {chill_file} not found")
        chill = None

    return seasonal, gdd, chill

def load_climate_indices():
    """Load climate indices"""

    print("\nLoading climate indices...")

    indices_file = DATA_DIR / 'raw/climate_indices/climate_indices_seasonal.csv'
    if indices_file.exists():
        indices = pd.read_csv(indices_file)
        print(f"  ✓ Loaded climate indices: {len(indices)} records")
        return indices
    else:
        print(f"  WARNING: {indices_file} not found")
        return None

def add_engineered_features(df):
    """
    Add engineered features based on Gemini recommendations

    Features to add:
    - Piecewise year splines
    - Hopkins' Bioclimatic Index
    - Coastal proximity flag
    - Photoperiod
    - Interaction terms
    """

    print("\nAdding engineered features...")

    df = df.copy()

    # 1. Piecewise year splines (replace year²)
    df['year_c'] = df['year'] - 1950
    df['year_seg_1950'] = np.maximum(0, df['year'] - 1950)
    df['year_seg_1980'] = np.maximum(0, df['year'] - 1980)
    df['year_seg_2000'] = np.maximum(0, df['year'] - 2000)

    # 2. Hopkins' Bioclimatic Index
    # Hopkins Index = 4×lat - 1.25×long + 4×(alt/122)
    df['Hopkins_Index'] = (
        4 * df['lat'] -
        1.25 * df['long'] +
        4 * (df['alt'] / 122)
    )

    # 3. Coastal proximity flag
    coastal_locations = {
        'washingtondc': 1,
        'vancouver': 1,
        'newyorkcity': 1,
        'kyoto': 0,
        'liestal': 0
    }

    df['is_coastal'] = df['location'].map(coastal_locations)
    df['is_coastal'] = df['is_coastal'].fillna(0)  # Auxiliary sites default to 0

    # 4. Photoperiod at Spring Equinox (March 20, DOY=79)
    # Photoperiod formula: 24/π × arccos(-tan(lat) × tan(declination))
    # Declination at spring equinox ≈ 0
    # Simplified: photoperiod ≈ 12 hours at equinox for all latitudes
    # More accurate: use latitude
    lat_rad = np.radians(df['lat'])
    declination = 0  # Spring equinox
    # At equinox, day length is approximately 12 hours everywhere
    # But we can use a latitude-based approximation for variation
    df['photoperiod_equinox'] = 12.0  # Baseline
    # Add latitude effect (simplified)
    df['photoperiod_mar20'] = 12.0 + 0.1 * np.abs(df['lat'])  # Very rough approximation

    # 5. Altitude transformations
    df['alt_log1p'] = np.log1p(np.maximum(df['alt'], 0))

    # 6. Interaction terms
    if 'GDD_jan_feb' in df.columns:
        df['lat_x_GDD'] = df['lat'] * df['GDD_jan_feb']
        df['year_c_x_GDD'] = df['year_c'] * df['GDD_jan_feb']

    if 'chill_hours_winter' in df.columns:
        df['lat_x_chill'] = df['lat'] * df['chill_hours_winter']

    # Year × latitude (Arctic amplification)
    df['lat_x_year_c'] = df['lat'] * df['year_c']

    # Coastal × year (differential warming)
    df['coastal_x_year_c'] = df['is_coastal'] * df['year_c']

    print(f"  ✓ Added engineered features")

    return df

def create_master_dataset():
    """Combine all data sources into master dataset"""

    print("="*70)
    print("CREATING MASTER DATASET")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load bloom data (base dataset)
    bloom = load_bloom_data()

    # Load weather features
    seasonal, gdd, chill = load_weather_features()

    # Load climate indices
    indices = load_climate_indices()

    # Merge step-by-step
    master = bloom.copy()

    # Merge seasonal weather
    if seasonal is not None:
        master = master.merge(seasonal, on=['location', 'year'], how='left')
        print(f"  ✓ Merged seasonal weather")

    # Merge GDD
    if gdd is not None:
        master = master.merge(gdd, on=['location', 'year'], how='left')
        print(f"  ✓ Merged GDD features")

    # Merge chilling hours
    if chill is not None:
        master = master.merge(chill, on=['location', 'year'], how='left')
        print(f"  ✓ Merged chilling hours")

    # Merge climate indices (by year only, applies to all locations)
    if indices is not None:
        master = master.merge(indices, on='year', how='left')
        print(f"  ✓ Merged climate indices")

    # Add engineered features
    master = add_engineered_features(master)

    # Create site_id
    if 'source' not in master.columns:
        master['source'] = 'competition'
    master['site_id'] = master['source'] + '::' + master['location']

    # Add observation count per site
    site_obs = master.groupby('site_id').size().reset_index(name='site_obs')
    master = master.merge(site_obs, on='site_id', how='left')

    print(f"\n  ✓ Created master dataset with {len(master):,} records")
    print(f"  ✓ Total columns: {len(master.columns)}")

    return master

def save_dataset(master):
    """Save final dataset in multiple formats"""

    print("\nSaving final dataset...")

    # CSV
    csv_file = FINAL_DIR / 'master_dataset.csv'
    master.to_csv(csv_file, index=False)
    print(f"  ✓ Saved CSV: {csv_file}")
    print(f"    Size: {csv_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Parquet (compressed)
    try:
        parquet_file = FINAL_DIR / 'master_dataset.parquet'
        master.to_parquet(parquet_file, index=False, compression='gzip')
        print(f"  ✓ Saved Parquet: {parquet_file}")
        print(f"    Size: {parquet_file.stat().st_size / 1024 / 1024:.2f} MB")
    except ImportError:
        print(f"  WARNING: pyarrow not installed, skipping parquet")

    # Data dictionary
    create_data_dictionary(master)

def create_data_dictionary(master):
    """Create data dictionary documenting all columns"""

    print("\nCreating data dictionary...")

    columns_info = []

    for col in master.columns:
        info = {
            'column': col,
            'dtype': str(master[col].dtype),
            'non_null': master[col].notna().sum(),
            'null_count': master[col].isna().sum(),
            'null_pct': f"{master[col].isna().sum() / len(master) * 100:.1f}%"
        }

        # Add sample values
        if master[col].dtype in ['int64', 'float64']:
            info['min'] = master[col].min()
            info['max'] = master[col].max()
            info['mean'] = master[col].mean()
        else:
            unique_vals = master[col].unique()
            info['unique_values'] = len(unique_vals)
            if len(unique_vals) <= 10:
                info['sample_values'] = ', '.join(map(str, unique_vals[:5]))

        columns_info.append(info)

    dict_df = pd.DataFrame(columns_info)

    dict_file = FINAL_DIR / 'data_dictionary.csv'
    dict_df.to_csv(dict_file, index=False)
    print(f"  ✓ Saved data dictionary: {dict_file}")

def print_summary(master):
    """Print summary statistics"""

    print("\n" + "="*70)
    print("MASTER DATASET SUMMARY")
    print("="*70)

    print(f"\nDataset dimensions: {master.shape[0]:,} rows × {master.shape[1]} columns")

    print(f"\nYear range: {master['year'].min()} - {master['year'].max()}")

    print(f"\nLocations: {master['location'].nunique()}")
    print(master['location'].value_counts().head(10))

    print(f"\nData completeness:")
    critical_cols = ['bloom_doy', 'lat', 'long', 'alt', 'GDD_jan_feb',
                     'chill_hours_winter', 'T_mean_winter']

    for col in critical_cols:
        if col in master.columns:
            coverage = (1 - master[col].isna().sum() / len(master)) * 100
            print(f"  {col}: {coverage:.1f}%")

    print(f"\nFeature groups:")
    print(f"  Core identifiers: location, year, lat, long, alt, source, site_id")
    print(f"  Target: bloom_doy")
    print(f"  Weather: T_mean_winter, T_mean_spring, precip_*, etc.")
    print(f"  GDD: GDD_jan_feb, GDD_winter, etc.")
    print(f"  Chilling: chill_hours_winter, chill_hours_nov_dec, etc.")
    print(f"  Climate indices: ONI_winter, NAO_winter, PDO_annual, etc.")
    print(f"  Engineered: Hopkins_Index, is_coastal, year_seg_*, interactions")

    print(f"\nSample of recent data:")
    print(master[master['year'] >= 2020][
        ['location', 'year', 'bloom_doy', 'GDD_jan_feb', 'T_mean_winter']
    ].head(10))

def main():
    """Main execution"""

    master = create_master_dataset()
    save_dataset(master)
    print_summary(master)

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print("\n✅ MASTER DATASET CREATED SUCCESSFULLY!")
    print(f"   Location: {FINAL_DIR / 'master_dataset.csv'}")
    print("\n   Ready for modeling!")
    print("="*70)

    return master

if __name__ == '__main__':
    data = main()
