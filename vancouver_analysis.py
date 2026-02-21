#!/usr/bin/env python3
"""
Vancouver-Specific Analysis and Improvement Strategy
Problem: Vancouver only has 4 years of data (2022-2025) - worst MAE at 7.62 days
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent

print("="*80)
print("VANCOUVER DEEP DIVE ANALYSIS")
print("="*80)

# Load Vancouver data
vancouver = pd.read_csv(ROOT / 'data/vancouver.csv')
print("\n1. VANCOUVER HISTORICAL DATA:")
print("-" * 60)
print(vancouver)
print(f"\nTotal samples: {len(vancouver)} (⚠️ VERY LIMITED!)")
print(f"Years: {vancouver['year'].min()} - {vancouver['year'].max()}")
print(f"Mean bloom DOY: {vancouver['bloom_doy'].mean():.1f}")
print(f"Std dev: {vancouver['bloom_doy'].std():.1f}")
print(f"Range: {vancouver['bloom_doy'].min()} - {vancouver['bloom_doy'].max()}")

# Vancouver characteristics
print("\n2. VANCOUVER CHARACTERISTICS:")
print("-" * 60)
print(f"Latitude: {vancouver['lat'].iloc[0]:.2f}°N (far north!)")
print(f"Longitude: {vancouver['long'].iloc[0]:.2f}°W (Pacific coast)")
print(f"Altitude: {vancouver['alt'].iloc[0]}m (sea level)")
print(f"Climate: Coastal Pacific, temperate oceanic (Cfb)")
print(f"Key feature: Mild, wet winters + cool springs")

# Load all auxiliary data to find similar locations
print("\n3. FINDING CLIMATE ANALOGUES:")
print("-" * 60)

japan = pd.read_csv(ROOT / 'data/japan.csv')
south_korea = pd.read_csv(ROOT / 'data/south_korea.csv')
meteoswiss = pd.read_csv(ROOT / 'data/meteoswiss.csv')

# Vancouver: lat=49.22, long=-123.16, coastal, temperate
# Look for similar characteristics:
# - Latitude 40-55°N (similar seasonal patterns)
# - Coastal locations (maritime climate)
# - Similar bloom timing (85-95 DOY)

print("\nSearching for locations with similar characteristics...")
print("Criteria:")
print("  - Latitude: 40-55°N (similar to Vancouver's 49.22°N)")
print("  - Coastal/low altitude: <200m")
print("  - Similar bloom DOY: 80-100")

all_aux = pd.concat([
    japan.assign(source='japan'),
    south_korea.assign(source='south_korea'),
    meteoswiss.assign(source='meteoswiss')
], ignore_index=True)

# Filter for similar locations
similar = all_aux[
    (all_aux['lat'] >= 40) & (all_aux['lat'] <= 55) &  # Similar latitude
    (all_aux['alt'] < 200) &  # Coastal/low altitude
    (all_aux['bloom_doy'] >= 80) & (all_aux['bloom_doy'] <= 100)  # Similar bloom timing
].copy()

print(f"\nFound {len(similar)} observations from similar locations")

# Group by location to see which locations are most similar
location_stats = similar.groupby('location').agg({
    'bloom_doy': ['mean', 'count'],
    'lat': 'first',
    'alt': 'first'
}).round(2)
location_stats.columns = ['mean_bloom_doy', 'samples', 'lat', 'alt']
location_stats = location_stats[location_stats['samples'] >= 10].sort_values('mean_bloom_doy')

print("\nTop similar locations (with 10+ samples):")
print(location_stats.head(15))

# Calculate climate similarity score
# Based on latitude difference and bloom timing similarity
vancouver_lat = 49.22
vancouver_mean_bloom = vancouver['bloom_doy'].mean()

location_stats['lat_diff'] = abs(location_stats['lat'] - vancouver_lat)
location_stats['bloom_diff'] = abs(location_stats['mean_bloom_doy'] - vancouver_mean_bloom)
location_stats['similarity_score'] = (
    1 / (1 + location_stats['lat_diff']) *
    1 / (1 + location_stats['bloom_diff']) *
    np.log(location_stats['samples'])  # Favor locations with more data
)
location_stats = location_stats.sort_values('similarity_score', ascending=False)

print("\n4. BEST CLIMATE ANALOGUES (by similarity score):")
print("-" * 60)
print(location_stats[['lat', 'mean_bloom_doy', 'samples', 'similarity_score']].head(10))

# Get the top analogues
top_analogues = location_stats.head(5).index.tolist()
print(f"\nTop 5 analogues: {top_analogues}")

# Load master dataset to check if we have enhanced features for Vancouver
master = pd.read_csv(ROOT / 'data/final/master_dataset.csv')
vancouver_master = master[master['location'] == 'vancouver']

print("\n5. VANCOUVER IN MASTER DATASET:")
print("-" * 60)
if len(vancouver_master) > 0:
    print(f"Years available: {vancouver_master['year'].unique()}")
    print(f"\nAvailable features:")
    for col in vancouver_master.columns:
        if col not in ['location', 'year', 'source']:
            non_null = vancouver_master[col].notna().sum()
            print(f"  {col:30s}: {non_null}/{len(vancouver_master)} non-null")
else:
    print("⚠️ Vancouver not found in master dataset!")

# Year-by-year analysis
print("\n6. VANCOUVER YEAR-BY-YEAR PATTERNS:")
print("-" * 60)
vancouver_sorted = vancouver.sort_values('year')
print(vancouver_sorted[['year', 'bloom_doy', 'bloom_date']].to_string(index=False))

print("\nYear-over-year changes:")
for i in range(1, len(vancouver_sorted)):
    curr = vancouver_sorted.iloc[i]
    prev = vancouver_sorted.iloc[i-1]
    diff = curr['bloom_doy'] - prev['bloom_doy']
    print(f"  {curr['year']}: {curr['bloom_doy']} (DOY) - {diff:+d} days from {prev['year']}")

# Check if there's a trend
years = vancouver['year'].values
doys = vancouver['bloom_doy'].values
if len(years) > 2:
    trend = np.polyfit(years - years.min(), doys, 1)[0]
    print(f"\nOverall trend: {trend:+.2f} days/year")
    if abs(trend) > 0.5:
        print(f"  ⚠️ Strong trend detected! Blooms getting {'earlier' if trend < 0 else 'later'}")
    else:
        print(f"  ✅ Relatively stable")

print("\n7. RECOMMENDATIONS FOR VANCOUVER:")
print("-" * 60)
print("""
STRATEGY A: CLIMATE ANALOGUE TRANSFER LEARNING
  1. Use top 5 similar locations as training data
  2. Weight by similarity score
  3. Fine-tune on Vancouver's 4 observations
  Expected gain: 1-2 days MAE improvement

STRATEGY B: ENSEMBLE WITH SIMPLE MODEL
  1. Complex model (current): Uses all features
  2. Simple model: Just lat + year + site_mean
  3. Ensemble: 50% complex + 50% simple
  Expected gain: 0.5-1 day MAE improvement (reduces overfitting)

STRATEGY C: LAG-ONLY MODEL
  1. Vancouver has perfect lag data (2022-2025 continuous)
  2. Use previous year bloom as primary feature
  3. Add small trend adjustment
  Expected gain: 1-3 days MAE improvement

STRATEGY D: NEAREST NEIGHBOR
  1. Find 5 most similar years from analogues
  2. Average their bloom DOY
  3. Adjust for latitude difference
  Expected gain: 0.5-1.5 days MAE improvement

RECOMMENDED: Combine B + C
  - Ensemble of lag-based model + climate analogue model
  - Should reduce MAE from 7.62 → 4-5 days
""")

# Export analogue locations for use in improved model
analogue_data = similar[similar['location'].isin(top_analogues)]
analogue_data.to_csv(ROOT / 'vancouver_analogues.csv', index=False)
print(f"\n✅ Saved {len(analogue_data)} analogue observations to vancouver_analogues.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
