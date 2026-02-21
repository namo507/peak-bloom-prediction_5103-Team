#!/usr/bin/env python3
"""
Validate the master dataset, especially for competition sites
"""

import pandas as pd
from pathlib import Path

FINAL_DIR = Path('data/final')
master_file = FINAL_DIR / 'master_dataset.csv'

print("="*80)
print("MASTER DATASET VALIDATION")
print("="*80)

# Load dataset
df = pd.read_csv(master_file)

print(f"\nDataset dimensions: {len(df)} rows × {len(df.columns)} columns")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# Competition sites
COMPETITION_SITES = ['kyoto', 'washingtondc', 'liestal', 'vancouver', 'newyorkcity']

print(f"\n{'='*80}")
print("COMPETITION SITES VALIDATION")
print(f"{'='*80}")

for site in COMPETITION_SITES:
    site_data = df[df['location'] == site]

    if len(site_data) == 0:
        print(f"\n❌ {site.upper()}: NO DATA FOUND")
        continue

    print(f"\n{site.upper()}:")
    print(f"  Records: {len(site_data)}")
    print(f"  Year range: {site_data['year'].min()} - {site_data['year'].max()}")

    # Check recent years (2020-2025)
    recent = site_data[site_data['year'] >= 2020]
    print(f"  Recent records (2020+): {len(recent)}")

    # Check feature coverage
    features_to_check = [
        'bloom_doy',
        'T_mean_winter',
        'T_mean_spring',
        'GDD_jan_feb',
        'chill_hours_winter',
        'ONI_winter',
        'NAO_winter',
        'photoperiod'
    ]

    print(f"\n  Feature coverage (overall):")
    for feature in features_to_check:
        if feature in site_data.columns:
            coverage = (1 - site_data[feature].isna().sum() / len(site_data)) * 100
            status = "✓" if coverage > 90 else "⚠️" if coverage > 50 else "❌"
            print(f"    {status} {feature:25s}: {coverage:6.1f}%")
        else:
            print(f"    ❌ {feature:25s}: NOT FOUND")

    # Check recent years specifically
    if len(recent) > 0:
        print(f"\n  Feature coverage (2020+):")
        for feature in features_to_check:
            if feature in recent.columns:
                coverage = (1 - recent[feature].isna().sum() / len(recent)) * 100
                status = "✓" if coverage > 90 else "⚠️" if coverage > 50 else "❌"
                print(f"    {status} {feature:25s}: {coverage:6.1f}%")

    # Show sample of recent data
    if len(recent) > 0:
        print(f"\n  Sample data (recent years):")
        sample_cols = ['year', 'bloom_doy', 'T_mean_winter', 'GDD_jan_feb',
                       'chill_hours_winter', 'ONI_winter']
        available_cols = [col for col in sample_cols if col in recent.columns]
        print(recent[available_cols].tail(3).to_string(index=False))

print(f"\n{'='*80}")
print("OVERALL DATASET STATISTICS")
print(f"{'='*80}")

# Count records by site
print(f"\nRecords per competition site:")
for site in COMPETITION_SITES:
    count = len(df[df['location'] == site])
    print(f"  {site:15s}: {count:5d}")

# Feature coverage overall
print(f"\nFeature coverage (all locations):")
important_features = [
    'bloom_doy', 'lat', 'long', 'alt',
    'T_mean_winter', 'T_mean_spring',
    'GDD_jan_feb', 'GDD_winter',
    'chill_hours_winter', 'chill_hours_nov_dec',
    'ONI_winter', 'NAO_winter', 'PDO_annual',
    'photoperiod', 'Hopkins_Index'
]

for feature in important_features:
    if feature in df.columns:
        coverage = (1 - df[feature].isna().sum() / len(df)) * 100
        status = "✓" if coverage > 90 else "⚠️" if coverage > 50 else "❌"
        print(f"  {status} {feature:25s}: {coverage:6.1f}%")

# Check for competition sites with complete features
print(f"\n{'='*80}")
print("COMPETITION SITES READINESS")
print(f"{'='*80}")

comp_sites_df = df[df['location'].isin(COMPETITION_SITES)]

if len(comp_sites_df) > 0:
    print(f"\nTotal competition site records: {len(comp_sites_df)}")

    # Check completeness for key features
    key_features = ['bloom_doy', 'T_mean_winter', 'GDD_jan_feb',
                    'chill_hours_winter', 'ONI_winter']

    complete_records = comp_sites_df.copy()
    for feature in key_features:
        if feature in complete_records.columns:
            complete_records = complete_records[complete_records[feature].notna()]

    print(f"Records with ALL key features: {len(complete_records)}")
    print(f"Completeness: {len(complete_records) / len(comp_sites_df) * 100:.1f}%")

    if len(complete_records) > 0:
        print(f"\nComplete records by site:")
        print(complete_records['location'].value_counts().sort_index())

print(f"\n{'='*80}")
if len(comp_sites_df) > 0:
    comp_coverage = (1 - comp_sites_df['GDD_jan_feb'].isna().sum() / len(comp_sites_df)) * 100
    if comp_coverage > 90:
        print("✅ VALIDATION PASSED - Competition sites have good feature coverage")
    elif comp_coverage > 50:
        print("⚠️  VALIDATION WARNING - Some competition sites missing features")
    else:
        print("❌ VALIDATION FAILED - Most competition sites missing features")
else:
    print("❌ VALIDATION FAILED - No competition site data found")
print(f"{'='*80}")
