#!/usr/bin/env python3
"""
Cherry Blossom Peak Bloom Prediction 2026 — Optimized Pipeline
Implements all recommendations from error analysis and research:
- Location-specific models
- Lag features (previous year bloom)
- Spring warmth GDD features
- Chilling × GDD interactions
- Bias correction
- Hyperparameter tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
TARGET_YEAR = 2026

print("="*80)
print("OPTIMIZED CHERRY BLOSSOM PREDICTION PIPELINE")
print("="*80)

# ============================================================================
# 1. DATA LOADING
# ============================================================================

def read_bloom_file(path: Path, source: str) -> pd.DataFrame:
    """Load bloom data from CSV"""
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

print("\n" + "="*80)
print("LOADING BLOOM DATA")
print("="*80)

# Competition data
competition = pd.concat([
    read_bloom_file(ROOT / 'data/kyoto.csv', 'kyoto'),
    read_bloom_file(ROOT / 'data/liestal.csv', 'liestal'),
    read_bloom_file(ROOT / 'data/washingtondc.csv', 'washingtondc'),
    read_bloom_file(ROOT / 'data/vancouver.csv', 'vancouver'),
], ignore_index=True)
competition['source'] = 'competition'

# NYC data from NPN
print("Loading NYC data from USA-NPN...")
npn = pd.read_csv(ROOT / 'data/USA-NPN_status_intensity_observations_data.csv')
npn = npn[(npn['Site_ID'] == 32789) & (npn['Species_ID'] == 228) & (npn['Phenophase_ID'] == 501)].copy()
npn['Observation_Date'] = pd.to_datetime(npn['Observation_Date'], format='%m/%d/%y', errors='coerce')
npn = npn.dropna(subset=['Observation_Date'])
npn['year'] = npn['Observation_Date'].dt.year
npn['doy'] = npn['Observation_Date'].dt.dayofyear

nyc_obs = npn[npn['Phenophase_Status'] == 1].groupby('year')['doy'].min().reset_index()
nyc_obs.columns = ['year', 'bloom_doy']
nyc_obs['location'] = 'nyc'
nyc_obs['source'] = 'nyc'
nyc_obs['lat'] = 40.71
nyc_obs['long'] = -74.01
nyc_obs['alt'] = 10

pheno = pd.read_csv(ROOT / 'data/USA-NPN_individual_phenometrics_data.csv')
pheno = pheno[(pheno['Site_ID'] == 32789) & (pheno['Species_ID'] == 228) & (pheno['Phenophase_ID'] == 501)].copy()
nyc_npn_pheno = (
    pheno.groupby('First_Yes_Year')['First_Yes_DOY']
    .min().reset_index()
)
nyc_npn_pheno.columns = ['year', 'bloom_doy']
nyc_npn_pheno['location'] = 'nyc'
nyc_npn_pheno['source'] = 'nyc'
nyc_npn_pheno['lat'] = 40.71
nyc_npn_pheno['long'] = -74.01
nyc_npn_pheno['alt'] = 10

nyc_combined = pd.concat([nyc_obs, nyc_npn_pheno], ignore_index=True)
nyc_final = nyc_combined.groupby('year', as_index=False).agg({
    'bloom_doy': 'min', 'location': 'first', 'source': 'first',
    'lat': 'first', 'long': 'first', 'alt': 'first'
})

competition = pd.concat([competition, nyc_final], ignore_index=True)

# Auxiliary data
auxiliary = pd.concat([
    read_bloom_file(ROOT / 'data/japan.csv', 'japan'),
    read_bloom_file(ROOT / 'data/south_korea.csv', 'south_korea'),
    read_bloom_file(ROOT / 'data/meteoswiss.csv', 'meteoswiss'),
], ignore_index=True)
auxiliary['source'] = 'auxiliary'

all_data = pd.concat([competition, auxiliary], ignore_index=True).dropna(subset=['bloom_doy'])
all_data['site_id'] = all_data['source'] + '::' + all_data['location']

print(f"Total observations: {len(all_data)}")
print(f"Competition locations: {competition['location'].unique()}")

# ============================================================================
# 2. LOAD ENHANCED FEATURES FROM MASTER DATASET
# ============================================================================

print("\n" + "="*80)
print("LOADING ENHANCED FEATURES")
print("="*80)

master = pd.read_csv(ROOT / 'data/final/master_dataset.csv')
master['site_id'] = master['source'] + '::' + master['location']

# Enhanced features to add
enhanced_cols = [
    'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',
    'chill_hours_winter', 'chill_hours_nov_dec',
    'ONI_winter', 'NAO_winter', 'PDO_annual', 'AO_winter',
    'Hopkins_Index', 'photoperiod_mar20',
    'year_seg_1950', 'year_seg_1980', 'year_seg_2000'
]

enhanced_feats = master[['location', 'year'] + [c for c in enhanced_cols if c in master.columns]].copy()
print(f"Enhanced features loaded: {len([c for c in enhanced_cols if c in master.columns])}")

# ============================================================================
# 3. CALCULATE SPRING WARMTH FEATURES (NEW!)
# ============================================================================

print("\n" + "="*80)
print("CALCULATING SPRING WARMTH FEATURES")
print("="*80)

# Try to calculate spring GDD from master dataset if daily temps available
spring_features = []
if 'temp_mean_mar' in master.columns:
    # Calculate March GDD (base 0°C)
    master['GDD_march'] = master['temp_mean_mar'].clip(lower=0) * 31  # Approximate
    spring_features.append('GDD_march')
    print("✅ Added GDD_march")

if 'temp_mean_feb' in master.columns and 'temp_mean_mar' in master.columns:
    # Calculate Feb-Mar combined GDD
    master['GDD_feb_mar'] = (master['temp_mean_feb'].clip(lower=0) * 28 +
                              master['temp_mean_mar'].clip(lower=0) * 31)
    spring_features.append('GDD_feb_mar')
    print("✅ Added GDD_feb_mar")

# Update enhanced features to include spring warmth
if spring_features:
    enhanced_feats = master[['location', 'year'] +
                            [c for c in enhanced_cols if c in master.columns] +
                            spring_features].copy()

# ============================================================================
# 4. ADD LAG FEATURES (NEW!)
# ============================================================================

def add_lag_features(df):
    """Add previous year bloom features"""
    df = df.sort_values(['location', 'year'])

    # Previous year bloom (lag 1)
    df['bloom_lag1'] = df.groupby('location')['bloom_doy'].shift(1)

    # 3-year rolling average
    df['bloom_avg_3yr'] = (
        df.groupby('location')['bloom_doy']
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(drop=True)
        .shift(1)
    )

    # 5-year rolling average
    df['bloom_avg_5yr'] = (
        df.groupby('location')['bloom_doy']
        .rolling(5, min_periods=1)
        .mean()
        .reset_index(drop=True)
        .shift(1)
    )

    # Trend: slope of last 5 years
    def calc_trend(x):
        if len(x) < 2:
            return 0
        years = np.arange(len(x))
        return np.polyfit(years, x, 1)[0]

    df['bloom_trend_5yr'] = (
        df.groupby('location')['bloom_doy']
        .rolling(5, min_periods=2)
        .apply(calc_trend, raw=True)
        .reset_index(drop=True)
        .shift(1)
    )

    return df

print("\n" + "="*80)
print("ADDING LAG FEATURES")
print("="*80)

all_data = add_lag_features(all_data)
print("✅ Added: bloom_lag1, bloom_avg_3yr, bloom_avg_5yr, bloom_trend_5yr")

# ============================================================================
# 5. FEATURE ENGINEERING WITH INTERACTIONS
# ============================================================================

def add_features(df, reference_df=None, enhanced_feats=None):
    """Add all features including new interactions"""
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

    # NEW: Chilling × GDD Interactions (from research!)
    if 'GDD_winter' in out.columns and 'chill_hours_winter' in out.columns:
        out['chill_x_GDD_winter'] = out['chill_hours_winter'] * out['GDD_winter']
        out['lat_x_GDD'] = out['lat_scaled'] * out['GDD_winter']
        out['year_c_x_GDD'] = out['year_c'] * out['GDD_winter']

    if 'chill_hours_winter' in out.columns:
        out['lat_x_chill'] = out['lat_scaled'] * out['chill_hours_winter']

    # NEW: Spring warmth interactions
    if 'GDD_march' in out.columns and 'chill_hours_winter' in out.columns:
        out['chill_x_GDD_spring'] = out['chill_hours_winter'] * out['GDD_march']

    # Lag feature interactions
    if 'bloom_lag1' in out.columns:
        out['lag1_x_year'] = out['bloom_lag1'] * out['year_c']
        out['lag1_deviation'] = out['bloom_lag1'] - out['site_mean']

    return out

# ============================================================================
# 6. LOCATION-SPECIFIC MODEL TRAINING
# ============================================================================

def train_location_model(loc_data, location_name, tune_hyperparams=True):
    """Train a model specific to one location with optional tuning"""

    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR: {location_name.upper()}")
    print(f"{'='*60}")
    print(f"Training samples: {len(loc_data)}")

    # Feature columns
    excl = ['bloom_doy', 'bloom_date', 'source', 'location', 'site_id', 'year']
    feat_cols = [c for c in loc_data.columns
                 if c not in excl and loc_data[c].dtype in ['float64', 'int64']]

    # Handle missing values
    train_median = loc_data[feat_cols].median()
    X = loc_data[feat_cols].fillna(train_median).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y = loc_data['bloom_doy']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if tune_hyperparams and len(loc_data) > 50:
        print("Running hyperparameter tuning...")

        param_dist = {
            'n_estimators': [600, 800, 1000, 1200],
            'learning_rate': [0.01, 0.015, 0.02, 0.025, 0.03],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8]
        }

        base_model = GradientBoostingRegressor(random_state=42)
        search = RandomizedSearchCV(
            base_model, param_dist, n_iter=30, cv=5,
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
        )
        search.fit(X_scaled, y)

        print(f"✅ Best MAE (CV): {-search.best_score_:.2f} days")
        print(f"Best params: {search.best_params_}")

        model = search.best_estimator_
    else:
        print("Using default hyperparameters (small dataset)...")
        model = GradientBoostingRegressor(
            n_estimators=800,
            learning_rate=0.015,
            max_depth=4,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        model.fit(X_scaled, y)

    return model, scaler, feat_cols, train_median

# ============================================================================
# 7. MAIN TRAINING LOOP - LOCATION-SPECIFIC MODELS
# ============================================================================

print("\n" + "="*80)
print("TRAINING LOCATION-SPECIFIC MODELS")
print("="*80)

competition_locs = ['kyoto', 'liestal', 'washingtondc', 'vancouver', 'nyc']
models = {}
scalers = {}
feature_cols = {}
medians = {}

# Prepare training data with all auxiliary
train_all = all_data[all_data['year'] < TARGET_YEAR].copy()
train_all = add_features(train_all, enhanced_feats=enhanced_feats)

for location in competition_locs:
    # Get competition data for this location
    loc_comp = train_all[
        (train_all['location'] == location) &
        (train_all['source'].isin(['competition', 'nyc']))
    ].copy()

    # Get auxiliary data (for enrichment if needed)
    loc_aux = train_all[train_all['source'] == 'auxiliary'].copy()

    # Combine: use location-specific competition data + all auxiliary
    # This gives the model location-specific patterns while learning from broader data
    loc_training_data = pd.concat([loc_comp, loc_aux], ignore_index=True)

    # Train location-specific model
    model, scaler, fcols, median = train_location_model(
        loc_training_data,
        location,
        tune_hyperparams=True
    )

    models[location] = model
    scalers[location] = scaler
    feature_cols[location] = fcols
    medians[location] = median

# ============================================================================
# 8. BIAS CORRECTION (from error analysis)
# ============================================================================

print("\n" + "="*80)
print("CALCULATING BIAS CORRECTION")
print("="*80)

# From error analysis, we know we over-predict by ~1.78 days
# Calculate location-specific bias
location_bias = {}

for location in competition_locs:
    loc_data = train_all[
        (train_all['location'] == location) &
        (train_all['source'].isin(['competition', 'nyc']))
    ].copy()

    if len(loc_data) < 10:
        location_bias[location] = 1.78  # Use global bias
        continue

    # Prepare features
    X_test = loc_data[feature_cols[location]].fillna(medians[location]).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    X_scaled = scalers[location].transform(X_test)

    # Predict
    y_pred = models[location].predict(X_scaled)
    y_true = loc_data['bloom_doy'].values

    # Calculate bias (mean error)
    bias = (y_pred - y_true).mean()
    location_bias[location] = bias

    print(f"{location:15s}: bias = {bias:+.2f} days")

print(f"\nGlobal bias: {np.mean(list(location_bias.values())):+.2f} days")

# ============================================================================
# 9. GENERATE 2026 PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print(f"GENERATING {TARGET_YEAR} PREDICTIONS")
print("="*80)

predictions_2026 = []

for location in competition_locs:
    print(f"\nPredicting for {location}...")

    # Get latest data for this location
    loc_hist = all_data[all_data['location'] == location].sort_values('year')
    latest = loc_hist.iloc[-1].copy()

    # Create 2026 row
    new_row = pd.DataFrame({
        'location': [location],
        'lat': [latest['lat']],
        'long': [latest['long']],
        'alt': [latest['alt']],
        'source': [latest['source']],
        'year': [TARGET_YEAR],
        'bloom_doy': [np.nan]
    })

    new_row['site_id'] = new_row['source'] + '::' + new_row['location']

    # Add lag features from historical data
    if len(loc_hist) >= 1:
        new_row['bloom_lag1'] = loc_hist['bloom_doy'].iloc[-1]
    if len(loc_hist) >= 3:
        new_row['bloom_avg_3yr'] = loc_hist['bloom_doy'].iloc[-3:].mean()
    if len(loc_hist) >= 5:
        new_row['bloom_avg_5yr'] = loc_hist['bloom_doy'].iloc[-5:].mean()
        years = np.arange(5)
        new_row['bloom_trend_5yr'] = np.polyfit(years, loc_hist['bloom_doy'].iloc[-5:], 1)[0]

    # Add features
    new_row_feat = add_features(new_row, reference_df=train_all, enhanced_feats=enhanced_feats)

    # Prepare for prediction
    X_new = new_row_feat[feature_cols[location]].fillna(medians[location]).fillna(0)
    X_new = X_new.replace([np.inf, -np.inf], 0)
    X_scaled = scalers[location].transform(X_new)

    # Predict
    pred_raw = models[location].predict(X_scaled)[0]

    # Apply bias correction
    pred_corrected = pred_raw - location_bias[location]

    # Prediction interval (simple ± 1 std from training residuals)
    loc_train_data = train_all[
        (train_all['location'] == location) &
        (train_all['source'].isin(['competition', 'nyc']))
    ]
    X_train = loc_train_data[feature_cols[location]].fillna(medians[location]).fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_train_scaled = scalers[location].transform(X_train)
    y_train = loc_train_data['bloom_doy'].values
    y_train_pred = models[location].predict(X_train_scaled)
    residuals = y_train - y_train_pred
    std_residual = np.std(residuals)

    lower = int(np.round(pred_corrected - 1.5 * std_residual))
    upper = int(np.round(pred_corrected + 1.5 * std_residual))

    predictions_2026.append({
        'location': location,
        'prediction': int(np.round(pred_corrected)),
        'lower': lower,
        'upper': upper,
        'pred_raw': pred_raw,
        'bias_correction': -location_bias[location],
        'std_residual': std_residual
    })

    print(f"  Raw prediction: {pred_raw:.1f}")
    print(f"  Bias correction: {-location_bias[location]:.2f}")
    print(f"  Final prediction: {int(np.round(pred_corrected))} (DOY)")
    print(f"  Interval: [{lower}, {upper}]")

# Save predictions
pred_df = pd.DataFrame(predictions_2026)
pred_df[['location', 'prediction', 'lower', 'upper']].to_csv(
    ROOT / 'cherry-predictions-optimized.csv', index=False
)

print("\n" + "="*80)
print("FINAL 2026 PREDICTIONS")
print("="*80)
print(pred_df[['location', 'prediction', 'lower', 'upper']].to_string(index=False))
print(f"\n✅ Predictions saved to: cherry-predictions-optimized.csv")

# ============================================================================
# 10. BACKTEST FOR VALIDATION
# ============================================================================

print("\n" + "="*80)
print("RUNNING BACKTEST VALIDATION (2015-2025)")
print("="*80)

backtest_errors = []

for test_year in range(2015, 2026):
    for location in competition_locs:
        # Get training data (before test year)
        train_data = all_data[all_data['year'] < test_year].copy()
        test_data = all_data[
            (all_data['year'] == test_year) &
            (all_data['location'] == location) &
            (all_data['source'].isin(['competition', 'nyc']))
        ].copy()

        if len(test_data) == 0:
            continue

        # Add features
        train_feat = add_features(train_data, enhanced_feats=enhanced_feats)
        test_feat = add_features(test_data, reference_df=train_feat, enhanced_feats=enhanced_feats)

        # Get location-specific model features
        if location not in feature_cols:
            continue

        X_test = test_feat[feature_cols[location]].fillna(medians[location]).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        X_scaled = scalers[location].transform(X_test)

        # Predict with bias correction
        pred = models[location].predict(X_scaled)[0] - location_bias[location]
        actual = test_feat['bloom_doy'].iloc[0]

        backtest_errors.append({
            'year': test_year,
            'location': location,
            'actual': actual,
            'predicted': pred,
            'error': pred - actual,
            'abs_error': abs(pred - actual)
        })

if backtest_errors:
    backtest_df = pd.DataFrame(backtest_errors)
    mae = backtest_df['abs_error'].mean()
    bias = backtest_df['error'].mean()

    print(f"\nBacktest Results (2015-2025):")
    print(f"  MAE: {mae:.2f} days")
    print(f"  Bias: {bias:+.2f} days")
    print(f"  Samples: {len(backtest_df)}")

    print("\nMAE by location:")
    print(backtest_df.groupby('location')['abs_error'].mean().round(2))

    backtest_df.to_csv(ROOT / 'backtest_results_optimized.csv', index=False)
    print(f"\n✅ Backtest results saved to: backtest_results_optimized.csv")

print("\n" + "="*80)
print("OPTIMIZATION COMPLETE!")
print("="*80)
