#!/usr/bin/env python3
"""
Fix photoperiod calculation to use proper astronomical formula
Current version uses rough approximation - this replaces with accurate calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

FINAL_DIR = Path('data/final')

def calculate_photoperiod_accurate(lat, doy=79):
    """
    Calculate photoperiod (day length) using proper astronomical formula

    Parameters:
    -----------
    lat : float or array
        Latitude in degrees
    doy : int
        Day of year (default 79 = March 20, spring equinox)

    Returns:
    --------
    photoperiod : float or array
        Day length in hours

    Formula:
    --------
    photoperiod = 24/π × arccos(-tan(lat) × tan(declination))

    Where declination (solar declination angle) is:
    declination = 23.44° × sin(2π × (DOY - 81) / 365)
    """

    # Convert latitude to radians
    lat_rad = np.radians(lat)

    # Calculate solar declination for the day of year
    # Declination is the angle between the sun's rays and the equatorial plane
    # Maximum at summer solstice (23.44°), minimum at winter solstice (-23.44°)
    # Zero at equinoxes

    declination_degrees = 23.44 * np.sin(np.radians(360 * (doy - 81) / 365))
    declination_rad = np.radians(declination_degrees)

    # Hour angle at sunrise/sunset
    # This is the angle the Earth must rotate for the sun to appear at horizon
    cos_hour_angle = -np.tan(lat_rad) * np.tan(declination_rad)

    # Clamp to [-1, 1] to handle extreme latitudes
    # (polar regions have 24-hour day/night during solstices)
    cos_hour_angle = np.clip(cos_hour_angle, -1, 1)

    # Calculate photoperiod
    # Hour angle in radians, converted to hours
    photoperiod_hours = (2 / np.pi) * np.arccos(cos_hour_angle) * 12

    return photoperiod_hours

def fix_photoperiod_in_dataset():
    """Load dataset, recalculate photoperiod, save"""

    dataset_file = FINAL_DIR / 'master_dataset.csv'

    if not dataset_file.exists():
        print(f"ERROR: {dataset_file} not found")
        print("Run 09_create_final_dataset.py first")
        return None

    print("="*80)
    print("FIXING PHOTOPERIOD CALCULATION")
    print("="*80)
    print()

    # Load dataset
    print(f"Loading dataset from {dataset_file}...")
    df = pd.read_csv(dataset_file)
    print(f"  ✓ Loaded {len(df):,} records")

    # Check current photoperiod
    if 'photoperiod_mar20' in df.columns:
        old_values = df['photoperiod_mar20'].copy()
        print(f"\nOLD photoperiod (approximation):")
        print(f"  Range: {old_values.min():.2f} - {old_values.max():.2f} hours")
        print(f"  Mean: {old_values.mean():.2f} hours")
        print(f"  Formula used: 12.0 + 0.1 × |lat|")

    # Calculate accurate photoperiod
    print(f"\nCalculating ACCURATE photoperiod using astronomical formula...")
    df['photoperiod_mar20'] = calculate_photoperiod_accurate(df['lat'], doy=79)

    print(f"  ✓ Recalculated for {len(df):,} records")

    print(f"\nNEW photoperiod (astronomical formula):")
    print(f"  Range: {df['photoperiod_mar20'].min():.2f} - {df['photoperiod_mar20'].max():.2f} hours")
    print(f"  Mean: {df['photoperiod_mar20'].mean():.2f} hours")
    print(f"  Formula: 24/π × arccos(-tan(lat) × tan(declination))")

    # Show comparison for sample latitudes
    print(f"\nSample comparisons (March 20 daylength):")
    print(f"{'Latitude':>10s} | {'OLD (approx)':>13s} | {'NEW (accurate)':>15s} | {'Difference':>11s}")
    print(f"{'-'*10}-+-{'-'*13}-+-{'-'*15}-+-{'-'*11}")

    sample_lats = [0, 20, 35, 40, 45, 50, 60]
    for lat in sample_lats:
        old = 12.0 + 0.1 * abs(lat)
        new = calculate_photoperiod_accurate(lat, 79)
        diff = new - old
        print(f"{lat:>10.1f}° | {old:>13.2f} hrs | {new:>15.2f} hrs | {diff:>+11.2f} hrs")

    # Save updated dataset
    print(f"\nSaving updated dataset...")

    # Save CSV
    df.to_csv(dataset_file, index=False)
    print(f"  ✓ Saved CSV: {dataset_file}")

    # Save parquet if possible
    try:
        parquet_file = FINAL_DIR / 'master_dataset.parquet'
        df.to_parquet(parquet_file, index=False, compression='gzip')
        print(f"  ✓ Saved Parquet: {parquet_file}")
    except:
        pass

    print(f"\n{'='*80}")
    print("✅ SUCCESS! Photoperiod calculation fixed")
    print(f"{'='*80}")
    print(f"\nNote: Photoperiod now uses scientifically accurate astronomical formula")
    print(f"This accounts for:")
    print(f"  - Solar declination angle (varies by day of year)")
    print(f"  - Latitude effects (polar regions have extreme day lengths)")
    print(f"  - Earth's axial tilt (23.44 degrees)")

    return df

def main():
    """Main execution"""

    result = fix_photoperiod_in_dataset()

    if result is not None:
        print(f"\n✓ Dataset updated with accurate photoperiod values")
        return result
    else:
        print(f"\n✗ Failed to fix photoperiod")
        return None

if __name__ == '__main__':
    data = main()
