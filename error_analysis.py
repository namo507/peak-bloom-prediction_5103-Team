#!/usr/bin/env python3
"""
Error Analysis for Enhanced Cherry Blossom Prediction Model
Identifies patterns in prediction errors to guide improvements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Re-run the enhanced model with error tracking
import sys
sys.path.append(str(Path(__file__).parent))

print("="*80)
print("LOADING DATA AND RUNNING MODEL WITH ERROR TRACKING")
print("="*80)

# Load bloom data (matching Solution_Enhanced_v2.py)
ROOT = Path(__file__).parent

def read_bloom_file(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        'source': source,
        'location': df['location'],
        'lat': df['lat'],
        'long': df['long'],
        'alt': df.get('alt', np.nan),
        'year': df['year'],
        'bloom_doy': df['bloom_doy'],
        'bloom_date': df.get('bloom_date')
    })

# Competition data
competition = pd.concat([
    read_bloom_file(ROOT / 'data/kyoto.csv', 'kyoto'),
    read_bloom_file(ROOT / 'data/liestal.csv', 'liestal'),
    read_bloom_file(ROOT / 'data/washingtondc.csv', 'washingtondc'),
    read_bloom_file(ROOT / 'data/vancouver.csv', 'vancouver'),
], ignore_index=True)
competition['source'] = 'competition'

# Auxiliary data
auxiliary = pd.concat([
    read_bloom_file(ROOT / 'data/japan.csv', 'japan'),
    read_bloom_file(ROOT / 'data/south_korea.csv', 'south_korea'),
    read_bloom_file(ROOT / 'data/meteoswiss.csv', 'meteoswiss'),
], ignore_index=True)
auxiliary['source'] = 'auxiliary'

# Combine bloom data
all_data = pd.concat([competition, auxiliary], ignore_index=True)
all_data = all_data.dropna(subset=['bloom_doy'])

# Load master dataset
master = pd.read_csv(ROOT / 'data/final/master_dataset.csv')

print(f"\nTotal bloom observations: {len(all_data)}")
print(f"Years: {all_data['year'].min()} - {all_data['year'].max()}")
print(f"Locations: {all_data['location'].unique()}")

# Create site_id for merging
if 'site_id' not in all_data.columns:
    all_data['site_id'] = all_data['source'] + '::' + all_data['location']

if 'site_id' not in master.columns:
    master['site_id'] = master['source'] + '::' + master['location']

# Enhanced feature set from master
enhanced_cols = [
    'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',
    'chill_hours_winter', 'chill_hours_nov_dec',
    'ONI_winter', 'NAO_winter', 'PDO_annual', 'AO_winter',
    'Hopkins_Index', 'photoperiod_mar20',
    'year_seg_1950', 'year_seg_1980', 'year_seg_2000'
]

enhanced_feats = master[['location', 'year'] + [c for c in enhanced_cols if c in master.columns]].copy()

def add_features(df, reference_df=None, enhanced_feats=None):
    """Add features to dataframe"""
    out = df.copy()
    ref = out if reference_df is None else reference_df.copy()

    # Ensure site_id
    if 'site_id' not in out.columns:
        out['site_id'] = out['source'] + '::' + out['location']
    if 'site_id' not in ref.columns:
        ref['site_id'] = ref['source'] + '::' + ref['location']

    # Merge enhanced features
    if enhanced_feats is not None:
        out = out.merge(enhanced_feats, on=['location', 'year'], how='left')

    # Site historical mean
    site_hist = ref.groupby('site_id')['bloom_doy'].mean().rename('site_mean')
    out = out.merge(site_hist, on='site_id', how='left')

    # Lat/long features
    out['lat_scaled'] = (out['lat'] - out['lat'].mean()) / out['lat'].std()
    out['long_scaled'] = (out['long'] - out['long'].mean()) / out['long'].std()
    out['lat_sq'] = out['lat_scaled'] ** 2

    # Year features
    out['year_c'] = out['year'] - 1950
    out['year_sq'] = out['year_c'] ** 2

    # Interactions
    if 'GDD_winter' in out.columns:
        out['lat_x_GDD'] = out['lat_scaled'] * out['GDD_winter']
        out['year_c_x_GDD'] = out['year_c'] * out['GDD_winter']
    if 'chill_hours_winter' in out.columns:
        out['lat_x_chill'] = out['lat_scaled'] * out['chill_hours_winter']

    return out

# Run backtest with error tracking
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

print("\n" + "="*80)
print("RUNNING BACKTEST WITH ERROR TRACKING")
print("="*80)

errors_list = []
predictions_list = []

for y in range(1900, 2026):
    if y % 25 == 0:
        print(f"Processing year {y}...")

    # Training data
    train_comp = all_data[(all_data['year'] < y) & (all_data['source'] == 'competition')].copy()
    if len(train_comp) < 10:
        continue

    # Test data (competition only)
    test_comp = all_data[(all_data['year'] == y) & (all_data['source'] == 'competition')].copy()
    if len(test_comp) == 0:
        continue

    # Full training with auxiliary
    train_all = all_data[all_data['year'] < y].copy()
    train_all = add_features(train_all, enhanced_feats=enhanced_feats)

    # Test features
    test_comp_copy = test_comp.copy()
    if 'site_id' not in test_comp_copy.columns:
        test_comp_copy['site_id'] = test_comp_copy['source'] + '::' + test_comp_copy['location']
    test_feat = add_features(test_comp_copy, reference_df=train_all, enhanced_feats=enhanced_feats)

    # Feature columns
    excl = ['bloom_doy', 'bloom_date', 'source', 'location', 'site_id', 'year']
    feat_cols = [c for c in train_all.columns if c not in excl and train_all[c].dtype in ['float64', 'int64']]

    # Handle missing values - fill with median and then replace any remaining NaN with 0
    train_median = train_all[feat_cols].median()
    X_train = train_all[feat_cols].fillna(train_median).fillna(0)
    y_train = train_all['bloom_doy']
    X_test = test_feat[feat_cols].fillna(train_median).fillna(0)

    # Ensure no NaN or inf values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)

    # Train model
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    gbr = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.015,
        max_depth=4,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    gbr.fit(X_train_sc, y_train)

    # Predict
    preds = gbr.predict(X_test_sc)

    # Track errors
    for i, row in test_feat.iterrows():
        actual = row['bloom_doy']
        pred = preds[test_feat.index.get_loc(i)]
        error = pred - actual

        errors_list.append({
            'year': y,
            'location': row['location'],
            'actual': actual,
            'predicted': pred,
            'error': error,
            'abs_error': abs(error),
            'lat': row['lat'],
            'long': row['long']
        })

# Create error dataframe
errors_df = pd.DataFrame(errors_list)

print(f"\nTotal predictions analyzed: {len(errors_df)}")
print(f"\nOverall MAE: {errors_df['abs_error'].mean():.2f} days")

print("\n" + "="*80)
print("ERROR ANALYSIS RESULTS")
print("="*80)

# 1. Error by location
print("\n1. ERRORS BY LOCATION:")
print("-" * 60)
loc_errors = errors_df.groupby('location').agg({
    'abs_error': ['mean', 'std', 'count'],
    'error': 'mean'
}).round(2)
loc_errors.columns = ['MAE', 'StdDev', 'Count', 'Bias']
loc_errors = loc_errors.sort_values('MAE', ascending=False)
print(loc_errors)

print("\nINSIGHT: Which locations are hardest to predict?")
print(f"  - Worst: {loc_errors.index[0]} (MAE={loc_errors.iloc[0]['MAE']:.2f})")
print(f"  - Best: {loc_errors.index[-1]} (MAE={loc_errors.iloc[-1]['MAE']:.2f})")

# 2. Error by time period
print("\n2. ERRORS BY TIME PERIOD:")
print("-" * 60)
errors_df['period'] = pd.cut(errors_df['year'],
                             bins=[1900, 1950, 1980, 2000, 2026],
                             labels=['1900-1950', '1950-1980', '1980-2000', '2000-2025'])
period_errors = errors_df.groupby('period').agg({
    'abs_error': ['mean', 'std', 'count'],
    'error': 'mean'
}).round(2)
period_errors.columns = ['MAE', 'StdDev', 'Count', 'Bias']
print(period_errors)

print("\nINSIGHT: Is the model getting worse over time (climate change)?")
if period_errors.iloc[-1]['MAE'] > period_errors.iloc[0]['MAE']:
    print("  ⚠️ YES - Recent years are harder to predict")
else:
    print("  ✅ NO - Model handles recent years well")

# 3. Error by bloom timing (early vs late)
print("\n3. ERRORS BY BLOOM TIMING:")
print("-" * 60)
errors_df['bloom_category'] = pd.cut(errors_df['actual'],
                                     bins=[0, 80, 90, 100, 200],
                                     labels=['Very Early (<80)', 'Early (80-90)',
                                            'Normal (90-100)', 'Late (>100)'])
timing_errors = errors_df.groupby('bloom_category').agg({
    'abs_error': ['mean', 'std', 'count'],
    'error': 'mean'
}).round(2)
timing_errors.columns = ['MAE', 'StdDev', 'Count', 'Bias']
print(timing_errors)

print("\nINSIGHT: Are we bad at extreme years?")
extreme_mae = timing_errors.loc[['Very Early (<80)', 'Late (>100)'], 'MAE'].mean()
normal_mae = timing_errors.loc[['Early (80-90)', 'Normal (90-100)'], 'MAE'].mean()
if extreme_mae > normal_mae * 1.2:
    print(f"  ⚠️ YES - Extreme years are {((extreme_mae/normal_mae - 1)*100):.1f}% harder")
else:
    print("  ✅ NO - Model handles extremes well")

# 4. Systematic bias
print("\n4. SYSTEMATIC BIAS:")
print("-" * 60)
print(f"Overall bias (mean error): {errors_df['error'].mean():.2f} days")
if abs(errors_df['error'].mean()) > 0.5:
    if errors_df['error'].mean() > 0:
        print("  ⚠️ Model tends to OVER-PREDICT (predict too late)")
    else:
        print("  ⚠️ Model tends to UNDER-PREDICT (predict too early)")
else:
    print("  ✅ No significant bias")

# 5. Worst predictions
print("\n5. TOP 10 WORST PREDICTIONS:")
print("-" * 60)
worst = errors_df.nlargest(10, 'abs_error')[['year', 'location', 'actual', 'predicted', 'error', 'abs_error']]
worst['predicted'] = worst['predicted'].round(1)
worst['error'] = worst['error'].round(1)
worst['abs_error'] = worst['abs_error'].round(1)
print(worst.to_string(index=False))

print("\nINSIGHT: Look for patterns in worst predictions")
worst_locs = worst['location'].value_counts()
if len(worst_locs) > 0:
    print(f"  - Most common location in worst 10: {worst_locs.index[0]} ({worst_locs.iloc[0]} times)")

# 6. Recent years (2015-2025) - most relevant for 2026
print("\n6. RECENT YEARS PERFORMANCE (2015-2025):")
print("-" * 60)
recent = errors_df[errors_df['year'] >= 2015]
print(f"Recent MAE (2015-2025): {recent['abs_error'].mean():.2f} days")
print(f"Recent bias: {recent['error'].mean():.2f} days")
print(f"\nYear-by-year:")
recent_yearly = recent.groupby('year').agg({
    'abs_error': 'mean',
    'error': 'mean'
}).round(2)
recent_yearly.columns = ['MAE', 'Bias']
print(recent_yearly)

# 7. Feature importance (top contributors)
print("\n7. FEATURE IMPORTANCE (from last model):")
print("-" * 60)
feat_imp = pd.DataFrame({
    'feature': feat_cols,
    'importance': gbr.feature_importances_
}).sort_values('importance', ascending=False).head(15)
print(feat_imp.to_string(index=False))

print("\nINSIGHT: Are our new features actually helping?")
enhanced_features_used = [f for f in enhanced_cols if f in feat_cols]
enhanced_in_top10 = sum(1 for f in feat_imp.head(10)['feature'] if f in enhanced_features_used)
print(f"  - Enhanced features in top 10: {enhanced_in_top10}/10")

# Save detailed errors
errors_df.to_csv('error_analysis_detailed.csv', index=False)
print(f"\n✅ Detailed errors saved to: error_analysis_detailed.csv")

print("\n" + "="*80)
print("KEY RECOMMENDATIONS:")
print("="*80)

# Generate recommendations based on analysis
recommendations = []

# Location-specific models
if loc_errors['MAE'].max() - loc_errors['MAE'].min() > 2:
    recommendations.append("⭐ LOCATION-SPECIFIC MODELS: Large variance between locations suggests separate models would help")

# Temporal issues
if period_errors.iloc[-1]['MAE'] > period_errors.iloc[0]['MAE'] * 1.2:
    recommendations.append("⭐ RECENT YEAR FOCUS: Model struggles with recent years - need better climate change features")

# Bias correction
if abs(errors_df['error'].mean()) > 0.5:
    direction = "late" if errors_df['error'].mean() > 0 else "early"
    recommendations.append(f"⭐ BIAS CORRECTION: Systematic {direction} bias - apply correction factor")

# Extreme events
if extreme_mae > normal_mae * 1.2:
    recommendations.append("⭐ EXTREME EVENT HANDLING: Add features for anomaly detection")

# Feature engineering
if enhanced_in_top10 < 3:
    recommendations.append("⚠️ FEATURE ENGINEERING: Enhanced features not impactful - try different features")
else:
    recommendations.append("✅ ENHANCED FEATURES WORKING: Keep GDD, chilling hours, climate indices")

if len(recommendations) > 0:
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
else:
    print("✅ Model looks good - focus on hyperparameter tuning")

print("\n" + "="*80)
