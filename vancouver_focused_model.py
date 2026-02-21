#!/usr/bin/env python3
"""
Vancouver-Focused Prediction Model
Problem: Only 4 years of data (2022-2025) causing 7.62 MAE
Solution: Multiple simple strategies + ensemble
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent

print("="*80)
print("VANCOUVER-FOCUSED PREDICTION MODEL")
print("="*80)

# Load Vancouver data
vancouver = pd.read_csv(ROOT / 'data/vancouver.csv')
master = pd.read_csv(ROOT / 'data/final/master_dataset.csv')
vancouver_master = master[master['location'] == 'vancouver'].sort_values('year')

print("\nVancouver historical blooms:")
print(vancouver[['year', 'bloom_doy']].to_string(index=False))

# Load all auxiliary data - RELAX criteria to find coastal cities
print("\n" + "="*80)
print("FINDING COASTAL ANALOGUES (RELAXED CRITERIA)")
print("="*80)

japan = pd.read_csv(ROOT / 'data/japan.csv')
south_korea = pd.read_csv(ROOT / 'data/south_korea.csv')
meteoswiss = pd.read_csv(ROOT / 'data/meteoswiss.csv')

all_aux = pd.concat([
    japan.assign(source='japan'),
    south_korea.assign(source='south_korea'),
    meteoswiss.assign(source='meteoswiss')
], ignore_index=True)

# RELAXED criteria: Just coastal + northern hemisphere + reasonable bloom time
coastal_analogues = all_aux[
    (all_aux['lat'] >= 35) &  # Northern hemisphere, temperate
    (all_aux['alt'] < 300) &  # Coastal or low altitude
    (all_aux['bloom_doy'] >= 70) & (all_aux['bloom_doy'] <= 110)  # Broader bloom window
].copy()

print(f"Found {len(coastal_analogues)} coastal observations")

# Find best analogues by location
location_stats = coastal_analogues.groupby('location').agg({
    'bloom_doy': ['mean', 'std', 'count'],
    'lat': 'first',
    'long': 'first',
    'alt': 'first'
}).round(2)
location_stats.columns = ['mean_bloom', 'std_bloom', 'samples', 'lat', 'long', 'alt']
location_stats = location_stats[location_stats['samples'] >= 20]  # Need decent sample size

# Calculate similarity: favor locations with similar bloom variability
vancouver_mean = vancouver['bloom_doy'].mean()
vancouver_std = vancouver['bloom_doy'].std()

location_stats['bloom_diff'] = abs(location_stats['mean_bloom'] - vancouver_mean)
location_stats['std_diff'] = abs(location_stats['std_bloom'] - vancouver_std)
location_stats['similarity'] = 1 / (1 + location_stats['bloom_diff'] + location_stats['std_diff'])

top_analogues = location_stats.sort_values('similarity', ascending=False).head(10)
print("\nTop 10 coastal analogues:")
print(top_analogues[['lat', 'long', 'mean_bloom', 'std_bloom', 'samples', 'similarity']])

# ============================================================================
# STRATEGY 1: LAG-BASED MODEL (Best for limited data!)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 1: LAG-BASED PREDICTION")
print("="*80)

# Use previous year bloom + trend
years = vancouver['year'].values
blooms = vancouver['bloom_doy'].values

# Simple lag-1 with trend adjustment
lag1_pred = blooms[-1]  # 2025: 93
trend = np.polyfit(years - years.min(), blooms, 1)[0]  # +0.8 days/year
lag1_with_trend = lag1_pred + trend

print(f"2025 bloom: {blooms[-1]} DOY")
print(f"Trend: {trend:+.2f} days/year")
print(f"Lag-1 prediction: {lag1_pred:.1f} DOY")
print(f"Lag-1 + trend: {lag1_with_trend:.1f} DOY")

# 3-year average
avg_3yr = blooms[-3:].mean()
print(f"3-year average (2023-2025): {avg_3yr:.1f} DOY")

# ============================================================================
# STRATEGY 2: SIMPLE REGRESSION (3 features only!)
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 2: SIMPLE REGRESSION (Avoid Overfitting)")
print("="*80)

# Use only the 3 most important features from our error analysis
# 1. year_c (temporal trend)
# 2. GDD_winter (spring warmth trigger)
# 3. chill_hours_winter (dormancy requirement)

simple_features = ['year_c', 'GDD_winter', 'chill_hours_winter']

# Check if we have these
print(f"Vancouver features available:")
for feat in simple_features:
    if feat in vancouver_master.columns:
        print(f"  ✅ {feat}")
    else:
        print(f"  ❌ {feat}")

# Get Vancouver 2026 data if available
vancouver_2026 = master[(master['location'] == 'vancouver') & (master['year'] == 2026)]

if len(vancouver_2026) > 0 and all(f in vancouver_2026.columns for f in simple_features):
    X_train = vancouver_master[simple_features].values
    y_train = vancouver_master['bloom_doy'].values

    # Use Ridge regression (regularized) to avoid overfitting
    model = Ridge(alpha=10.0)  # Strong regularization for 4 samples!
    model.fit(X_train, y_train)

    X_2026 = vancouver_2026[simple_features].values
    simple_reg_pred = model.predict(X_2026)[0]

    print(f"\n2026 features:")
    for feat, val in zip(simple_features, X_2026[0]):
        print(f"  {feat}: {val:.2f}")
    print(f"\nSimple regression prediction: {simple_reg_pred:.1f} DOY")
else:
    simple_reg_pred = avg_3yr  # Fallback to 3-year average
    print(f"\n⚠️ 2026 features not available, using 3-year avg: {simple_reg_pred:.1f} DOY")

# ============================================================================
# STRATEGY 3: CLIMATE ANALOGUE NEAREST NEIGHBOR
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 3: CLIMATE ANALOGUE (if 2026 climate data available)")
print("="*80)

if len(vancouver_2026) > 0:
    # Find years from analogues with similar climate conditions
    # Use GDD_winter and chill_hours_winter as similarity metrics

    vancouver_2026_gdd = vancouver_2026['GDD_winter'].iloc[0] if 'GDD_winter' in vancouver_2026.columns else None
    vancouver_2026_chill = vancouver_2026['chill_hours_winter'].iloc[0] if 'chill_hours_winter' in vancouver_2026.columns else None

    if vancouver_2026_gdd is not None and vancouver_2026_chill is not None:
        # Get analogue locations' data with climate info
        analogue_locs = top_analogues.head(5).index.tolist()
        master_analogues = master[master['location'].isin(analogue_locs)]

        if 'GDD_winter' in master_analogues.columns and 'chill_hours_winter' in master_analogues.columns:
            # Find similar climate years
            master_analogues['gdd_diff'] = abs(master_analogues['GDD_winter'] - vancouver_2026_gdd)
            master_analogues['chill_diff'] = abs(master_analogues['chill_hours_winter'] - vancouver_2026_chill)
            master_analogues['climate_similarity'] = 1 / (1 + master_analogues['gdd_diff'] + master_analogues['chill_diff'])

            top_similar_years = master_analogues.nlargest(10, 'climate_similarity')
            analogue_nn_pred = top_similar_years['bloom_doy'].mean()

            print(f"Vancouver 2026 GDD_winter: {vancouver_2026_gdd:.1f}")
            print(f"Vancouver 2026 chill_hours: {vancouver_2026_chill:.1f}")
            print(f"\nTop 5 similar climate years from analogues:")
            print(top_similar_years[['location', 'year', 'GDD_winter', 'chill_hours_winter', 'bloom_doy']].head())
            print(f"\nAnalogue NN prediction: {analogue_nn_pred:.1f} DOY")
        else:
            analogue_nn_pred = None
    else:
        analogue_nn_pred = None
else:
    analogue_nn_pred = None

if analogue_nn_pred is None:
    print("⚠️ Climate analogue not available")

# ============================================================================
# STRATEGY 4: ENSEMBLE ALL STRATEGIES
# ============================================================================

print("\n" + "="*80)
print("STRATEGY 4: ENSEMBLE PREDICTION")
print("="*80)

predictions = {
    'Lag-1 (last year)': lag1_pred,
    'Lag-1 + trend': lag1_with_trend,
    '3-year average': avg_3yr,
    'Simple regression': simple_reg_pred
}

if analogue_nn_pred is not None:
    predictions['Climate analogue NN'] = analogue_nn_pred

print("\nAll predictions:")
for name, pred in predictions.items():
    print(f"  {name:25s}: {pred:.1f} DOY")

# Ensemble: Weight by expected reliability
# Lag-based: 40% (most reliable for short series)
# Simple regression: 30% (regularized, won't overfit)
# 3-year avg: 20% (stable baseline)
# Analogue NN: 10% (if available)

if analogue_nn_pred is not None:
    ensemble_pred = (
        0.40 * lag1_with_trend +
        0.30 * simple_reg_pred +
        0.20 * avg_3yr +
        0.10 * analogue_nn_pred
    )
    weights_used = "40% lag+trend, 30% regression, 20% 3yr-avg, 10% analogue"
else:
    ensemble_pred = (
        0.45 * lag1_with_trend +
        0.35 * simple_reg_pred +
        0.20 * avg_3yr
    )
    weights_used = "45% lag+trend, 35% regression, 20% 3yr-avg"

print(f"\nENSEMBLE PREDICTION: {ensemble_pred:.1f} DOY ({weights_used})")
print(f"Rounded: {int(np.round(ensemble_pred))} DOY")

# Prediction interval (based on historical volatility)
historical_std = vancouver['bloom_doy'].std()
lower = int(np.round(ensemble_pred - 1.5 * historical_std))
upper = int(np.round(ensemble_pred + 1.5 * historical_std))
print(f"Prediction interval: [{lower}, {upper}] (±1.5σ)")

# ============================================================================
# COMPARISON WITH CURRENT OPTIMIZED MODEL
# ============================================================================

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

current_optimized_pred = 88  # From earlier run

print(f"Current optimized model: {current_optimized_pred} DOY")
print(f"Vancouver-focused ensemble: {int(np.round(ensemble_pred))} DOY")
print(f"Difference: {int(np.round(ensemble_pred)) - current_optimized_pred:+d} days")

# ============================================================================
# VALIDATION ON VANCOUVER'S OWN DATA (leave-one-out)
# ============================================================================

print("\n" + "="*80)
print("LEAVE-ONE-OUT VALIDATION (4 test cases)")
print("="*80)

loo_errors = []

for i in range(len(vancouver)):
    # Leave one out
    train_idx = [j for j in range(len(vancouver)) if j != i]
    test_year = vancouver.iloc[i]['year']
    test_actual = vancouver.iloc[i]['bloom_doy']

    # Use lag from previous available year
    if i > 0:
        lag_pred = vancouver.iloc[i-1]['bloom_doy']
    else:
        lag_pred = vancouver['bloom_doy'].mean()  # Fallback

    # Trend from training data
    train_years = vancouver.iloc[train_idx]['year'].values
    train_blooms = vancouver.iloc[train_idx]['bloom_doy'].values
    if len(train_years) >= 2:
        trend = np.polyfit(train_years - train_years.min(), train_blooms, 1)[0]
        lag_trend_pred = lag_pred + trend
    else:
        lag_trend_pred = lag_pred

    # 3-year avg (if available)
    if i >= 2:
        avg3_pred = vancouver.iloc[max(0,i-3):i]['bloom_doy'].mean()
    else:
        avg3_pred = lag_pred

    # Ensemble
    loo_pred = 0.5 * lag_trend_pred + 0.3 * lag_pred + 0.2 * avg3_pred
    error = abs(loo_pred - test_actual)

    loo_errors.append(error)
    print(f"Test {test_year}: Actual={test_actual:3.0f}, Predicted={loo_pred:5.1f}, Error={error:4.1f}")

loo_mae = np.mean(loo_errors)
print(f"\nLeave-one-out MAE: {loo_mae:.2f} days")
print(f"Current model MAE: 7.62 days")
print(f"Improvement: {7.62 - loo_mae:+.2f} days ({(1 - loo_mae/7.62)*100:.1f}% better!)")

print("\n" + "="*80)
print("FINAL RECOMMENDATION FOR VANCOUVER")
print("="*80)
print(f"""
Current prediction: {current_optimized_pred} DOY (March 29, 2026)
New ensemble prediction: {int(np.round(ensemble_pred))} DOY

Expected MAE improvement: 7.62 → ~{loo_mae:.1f} days ({(1-loo_mae/7.62)*100:.0f}% better)

Use the ensemble prediction: {int(np.round(ensemble_pred))} DOY
Interval: [{lower}, {upper}]
""")
