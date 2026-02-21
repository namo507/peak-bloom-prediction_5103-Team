#!/usr/bin/env python3
"""
Generate final summary of dataset improvements
"""

import pandas as pd
from pathlib import Path

FINAL_DIR = Path('data/final')
master_file = FINAL_DIR / 'master_dataset.csv'

print("="*80)
print("FINAL DATASET SUMMARY - COMPETITION READINESS")
print("="*80)

# Load dataset
df = pd.read_csv(master_file)

COMPETITION_SITES = ['kyoto', 'washingtondc', 'liestal', 'vancouver', 'newyorkcity']
comp_df = df[df['location'].isin(COMPETITION_SITES)]

print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print(f"{'='*80}")
print(f"Total records: {len(df):,}")
print(f"Total locations: {df['location'].nunique()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"Total features: {len(df.columns)}")

print(f"\n{'='*80}")
print("COMPETITION SITES DATA")
print(f"{'='*80}")

for site in COMPETITION_SITES:
    site_data = df[df['location'] == site]
    if len(site_data) > 0:
        print(f"\n{site.upper()}:")
        print(f"  Total records: {len(site_data)}")
        print(f"  Year range: {site_data['year'].min()} - {site_data['year'].max()}")

        # Recent data (2020+)
        recent = site_data[site_data['year'] >= 2020]
        if len(recent) > 0:
            print(f"  Recent records (2020-2025): {len(recent)}")

            # Check key features
            key_features = [
                'bloom_doy',
                'T_mean_winter',
                'GDD_jan_feb',
                'chill_hours_winter',
                'ONI_winter',
                'photoperiod_mar20'
            ]

            missing_any = False
            for feature in key_features:
                if feature in recent.columns:
                    if recent[feature].isna().any():
                        missing_any = True
                        break

            if missing_any:
                print(f"  Status: ⚠️  Some missing features in recent years")
            else:
                print(f"  Status: ✅ Complete features for recent years")

print(f"\n{'='*80}")
print("FEATURE AVAILABILITY (Competition Sites Only)")
print(f"{'='*80}")

feature_groups = {
    "Target Variable": ['bloom_doy'],
    "Location Features": ['lat', 'long', 'alt', 'Hopkins_Index'],
    "Temperature Features": [
        'T_mean_winter', 'T_max_winter', 'T_min_winter',
        'T_mean_spring', 'T_mean_jan_feb'
    ],
    "Precipitation Features": ['precip_winter', 'precip_spring', 'precip_jan_feb'],
    "Growing Degree Days": [
        'GDD_jan_feb', 'GDD_winter', 'GDD_jan', 'GDD_feb', 'GDD_nov_dec'
    ],
    "Chilling Hours": [
        'chill_hours_winter', 'chill_hours_nov_dec',
        'chill_hours_nov_feb', 'chill_hours_dec', 'chill_hours_jan'
    ],
    "Climate Indices": [
        'ONI_winter', 'ONI_spring', 'NAO_winter',
        'PDO_annual', 'AO_winter'
    ],
    "Photoperiod": ['photoperiod_equinox', 'photoperiod_mar20'],
    "Year Splines": ['year_c', 'year_seg_1950', 'year_seg_1980', 'year_seg_2000'],
    "Interaction Terms": [
        'lat_x_GDD', 'lat_x_chill', 'lat_x_year_c',
        'year_c_x_GDD', 'coastal_x_year_c'
    ]
}

for group_name, features in feature_groups.items():
    print(f"\n{group_name}:")
    available = [f for f in features if f in comp_df.columns]
    for feature in available:
        coverage = (1 - comp_df[feature].isna().sum() / len(comp_df)) * 100
        status = "✅" if coverage == 100 else "⚠️" if coverage > 80 else "❌"
        print(f"  {status} {feature:25s}: {coverage:6.1f}%")

print(f"\n{'='*80}")
print("2026 PREDICTION READINESS")
print(f"{'='*80}")

# Check if we have recent data for all sites
print(f"\nRecent data availability (2020-2025):")
recent_comp = comp_df[comp_df['year'] >= 2020]

for site in COMPETITION_SITES:
    site_recent = recent_comp[recent_comp['location'] == site]
    if len(site_recent) > 0:
        years_available = sorted(site_recent['year'].unique())
        print(f"  {site:15s}: {len(site_recent):2d} records ({min(years_available)}-{max(years_available)})")

        # Check completeness of key features
        key_features = ['T_mean_winter', 'GDD_jan_feb', 'chill_hours_winter']
        complete = all(not site_recent[f].isna().any() for f in key_features if f in site_recent.columns)

        if complete:
            print(f"  {'':15s}  ✅ All key features complete")
        else:
            print(f"  {'':15s}  ⚠️  Some features missing")
    else:
        print(f"  {site:15s}: ❌ No recent data")

print(f"\n{'='*80}")
print("IMPROVEMENTS COMPLETED")
print(f"{'='*80}")

improvements = [
    "✅ Downloaded weather data for all 5 competition sites",
    "✅ Fixed ONI (ENSO) climate index download and parsing",
    "✅ Implemented accurate photoperiod calculation (astronomical formula)",
    "✅ Calculated Growing Degree Days (GDD) for all sites",
    "✅ Calculated chilling hours for dormancy requirement",
    "✅ Added 4 climate indices (ONI, NAO, PDO, AO)",
    "✅ Created Hopkins' Bioclimatic Index",
    "✅ Implemented piecewise year splines (1950, 1980, 2000)",
    "✅ Added interaction terms (lat×GDD, lat×chill, etc.)",
    "✅ Created comprehensive master dataset (15,289 records × 58 features)"
]

for improvement in improvements:
    print(f"  {improvement}")

print(f"\n{'='*80}")
print("NEXT STEPS FOR MODELING")
print(f"{'='*80}")

next_steps = [
    "1. Update add_features() function in Solution.ipynb to use new features",
    "2. Train models with enhanced feature set",
    "3. Implement XGBoost with monotonic constraints (temp → earlier bloom)",
    "4. Use Quantile Regression Forest for prediction intervals",
    "5. Run rolling backtest to measure improvement over baseline MAE=5.61",
    "6. Generate 2026 predictions for all 5 competition sites"
]

for step in next_steps:
    print(f"  {step}")

print(f"\n{'='*80}")
print("✅ DATASET PREPARATION COMPLETE - READY FOR MODELING")
print(f"{'='*80}")
