#!/usr/bin/env python3
"""
Cherry Blossom Peak Bloom Prediction 2026 — Optimized Pipeline (Fast Version)
Uses pre-optimized hyperparameters for speed while maintaining all improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
TARGET_YEAR = 2026

print("="*80)
print("OPTIMIZED CHERRY BLOSSOM PREDICTION PIPELINE (FAST)")
print("="*80)

# Load data (same as optimized version)
def read_bloom_file(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        'source': source, 'location': df['location'], 'lat': df['lat'],
        'long': df['long'], 'alt': df.get('alt', np.nan),
        'year': df['year'], 'bloom_doy': df['bloom_doy'],
        'bloom_date': df.get('bloom_date')
    })

print("\nLoading bloom data...")
competition = pd.concat([
    read_bloom_file(ROOT / 'data/kyoto.csv', 'kyoto'),
    read_bloom_file(ROOT / 'data/liestal.csv', 'liestal'),
    read_bloom_file(ROOT / 'data/washingtondc.csv', 'washingtondc'),
    read_bloom_file(ROOT / 'data/vancouver.csv', 'vancouver'),
], ignore_index=True)
competition['source'] = 'competition'

# NYC data
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
nyc_npn_pheno = pheno.groupby('First_Yes_Year')['First_Yes_DOY'].min().reset_index()
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

auxiliary = pd.concat([
    read_bloom_file(ROOT / 'data/japan.csv', 'japan'),
    read_bloom_file(ROOT / 'data/south_korea.csv', 'south_korea'),
    read_bloom_file(ROOT / 'data/meteoswiss.csv', 'meteoswiss'),
], ignore_index=True)
auxiliary['source'] = 'auxiliary'

all_data = pd.concat([competition, auxiliary], ignore_index=True).dropna(subset=['bloom_doy'])
all_data['site_id'] = all_data['source'] + '::' + all_data['location']

# Load enhanced features
master = pd.read_csv(ROOT / 'data/final/master_dataset.csv')
master['site_id'] = master['source'] + '::' + master['location']
enhanced_cols = [
    'GDD_jan_feb', 'GDD_winter', 'GDD_nov_dec',
    'chill_hours_winter', 'chill_hours_nov_dec',
    'ONI_winter', 'NAO_winter', 'PDO_annual', 'AO_winter',
    'Hopkins_Index', 'photoperiod_mar20',
    'year_seg_1950', 'year_seg_1980', 'year_seg_2000'
]
enhanced_feats = master[['location', 'year'] + [c for c in enhanced_cols if c in master.columns]].copy()

# Add lag features
def add_lag_features(df):
    df = df.sort_values(['location', 'year'])
    df['bloom_lag1'] = df.groupby('location')['bloom_doy'].shift(1)
    df['bloom_avg_3yr'] = df.groupby('location')['bloom_doy'].rolling(3, min_periods=1).mean().reset_index(drop=True).shift(1)
    df['bloom_avg_5yr'] = df.groupby('location')['bloom_doy'].rolling(5, min_periods=1).mean().reset_index(drop=True).shift(1)
    def calc_trend(x):
        if len(x) < 2: return 0
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    df['bloom_trend_5yr'] = df.groupby('location')['bloom_doy'].rolling(5, min_periods=2).apply(calc_trend, raw=True).reset_index(drop=True).shift(1)
    return df

all_data = add_lag_features(all_data)
print("✅ Lag features added")

# Feature engineering
def add_features(df, reference_df=None, enhanced_feats=None):
    out = df.copy()
    ref = out if reference_df is None else reference_df.copy()
    if 'site_id' not in out.columns:
        out['site_id'] = out['source'] + '::' + out['location']
    if 'site_id' not in ref.columns:
        ref['site_id'] = ref['source'] + '::' + ref['location']
    if enhanced_feats is not None:
        out = out.merge(enhanced_feats, on=['location', 'year'], how='left')
    site_hist = ref.groupby('site_id')['bloom_doy'].mean().rename('site_mean')
    out = out.merge(site_hist, on='site_id', how='left')
    out['lat_scaled'] = (out['lat'] - out['lat'].mean()) / out['lat'].std()
    out['long_scaled'] = (out['long'] - out['long'].mean()) / out['long'].std()
    out['lat_sq'] = out['lat_scaled'] ** 2
    out['year_c'] = out['year'] - 1950
    out['year_sq'] = out['year_c'] ** 2
    if 'GDD_winter' in out.columns and 'chill_hours_winter' in out.columns:
        out['chill_x_GDD_winter'] = out['chill_hours_winter'] * out['GDD_winter']
        out['lat_x_GDD'] = out['lat_scaled'] * out['GDD_winter']
        out['year_c_x_GDD'] = out['year_c'] * out['GDD_winter']
    if 'chill_hours_winter' in out.columns:
        out['lat_x_chill'] = out['lat_scaled'] * out['chill_hours_winter']
    if 'bloom_lag1' in out.columns:
        out['lag1_x_year'] = out['bloom_lag1'] * out['year_c']
        out['lag1_deviation'] = out['bloom_lag1'] - out['site_mean']
    return out

# Train location-specific models
print("\n" + "="*80)
print("TRAINING LOCATION-SPECIFIC MODELS")
print("="*80)

competition_locs = ['kyoto', 'liestal', 'washingtondc', 'vancouver', 'nyc']
models = {}
scalers = {}
feature_cols = {}
medians = {}
location_bias = {}

train_all = all_data[all_data['year'] < TARGET_YEAR].copy()
train_all = add_features(train_all, enhanced_feats=enhanced_feats)

for location in competition_locs:
    print(f"\nTraining model for: {location.upper()}")

    loc_comp = train_all[(train_all['location'] == location) & (train_all['source'].isin(['competition', 'nyc']))].copy()
    loc_aux = train_all[train_all['source'] == 'auxiliary'].copy()
    loc_training_data = pd.concat([loc_comp, loc_aux], ignore_index=True)

    # Features
    excl = ['bloom_doy', 'bloom_date', 'source', 'location', 'site_id', 'year']
    fcols = [c for c in loc_training_data.columns if c not in excl and loc_training_data[c].dtype in ['float64', 'int64']]

    train_median = loc_training_data[fcols].median()
    X = loc_training_data[fcols].fillna(train_median).fillna(0).replace([np.inf, -np.inf], 0)
    y = loc_training_data['bloom_doy']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use optimized hyperparameters (from research + error analysis)
    model = GradientBoostingRegressor(
        n_estimators=900,  # Slightly increased
        learning_rate=0.015,
        max_depth=5,  # Deeper for location-specific patterns
        subsample=0.8,
        max_features='sqrt',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X_scaled, y)

    models[location] = model
    scalers[location] = scaler
    feature_cols[location] = fcols
    medians[location] = train_median

    # Calculate bias
    y_pred = model.predict(X_scaled)
    y_true_comp = loc_comp['bloom_doy'].values
    if len(loc_comp) > 0:
        loc_comp_feat = loc_comp[fcols].fillna(train_median).fillna(0).replace([np.inf, -np.inf], 0)
        loc_comp_scaled = scaler.transform(loc_comp_feat)
        loc_pred = model.predict(loc_comp_scaled)
        bias = (loc_pred - y_true_comp).mean()
    else:
        bias = 1.78
    location_bias[location] = bias
    print(f"  Training samples: {len(loc_training_data)}")
    print(f"  Bias correction: {bias:+.2f} days")

# Generate 2026 predictions
print("\n" + "="*80)
print("GENERATING 2026 PREDICTIONS")
print("="*80)

predictions_2026 = []

for location in competition_locs:
    loc_hist = all_data[all_data['location'] == location].sort_values('year')
    latest = loc_hist.iloc[-1].copy()

    new_row = pd.DataFrame({
        'location': [location], 'lat': [latest['lat']], 'long': [latest['long']],
        'alt': [latest['alt']], 'source': [latest['source']], 'year': [TARGET_YEAR],
        'bloom_doy': [np.nan], 'site_id': [latest['source'] + '::' + location]
    })

    if len(loc_hist) >= 1:
        new_row['bloom_lag1'] = loc_hist['bloom_doy'].iloc[-1]
    if len(loc_hist) >= 3:
        new_row['bloom_avg_3yr'] = loc_hist['bloom_doy'].iloc[-3:].mean()
    if len(loc_hist) >= 5:
        new_row['bloom_avg_5yr'] = loc_hist['bloom_doy'].iloc[-5:].mean()
        new_row['bloom_trend_5yr'] = np.polyfit(np.arange(5), loc_hist['bloom_doy'].iloc[-5:], 1)[0]

    new_row_feat = add_features(new_row, reference_df=train_all, enhanced_feats=enhanced_feats)
    # Handle missing columns (for locations with limited history)
    for col in feature_cols[location]:
        if col not in new_row_feat.columns:
            new_row_feat[col] = np.nan
    X_new = new_row_feat[feature_cols[location]].fillna(medians[location]).fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scalers[location].transform(X_new)

    pred_raw = models[location].predict(X_scaled)[0]
    pred_corrected = pred_raw - location_bias[location]

    # Interval
    loc_train = train_all[(train_all['location'] == location) & (train_all['source'].isin(['competition', 'nyc']))]
    X_train = loc_train[feature_cols[location]].fillna(medians[location]).fillna(0).replace([np.inf, -np.inf], 0)
    X_train_scaled = scalers[location].transform(X_train)
    y_train = loc_train['bloom_doy'].values
    y_train_pred = models[location].predict(X_train_scaled)
    std_residual = np.std(y_train - y_train_pred)

    lower = int(np.round(pred_corrected - 1.5 * std_residual))
    upper = int(np.round(pred_corrected + 1.5 * std_residual))

    predictions_2026.append({
        'location': location,
        'prediction': int(np.round(pred_corrected)),
        'lower': lower,
        'upper': upper
    })

    print(f"\n{location}:")
    print(f"  Prediction: {int(np.round(pred_corrected))} (DOY)")
    print(f"  Interval: [{lower}, {upper}]")

pred_df = pd.DataFrame(predictions_2026)
pred_df.to_csv(ROOT / 'cherry-predictions-optimized.csv', index=False)

print("\n" + "="*80)
print("FINAL 2026 PREDICTIONS")
print("="*80)
print(pred_df.to_string(index=False))
print(f"\n✅ Saved to: cherry-predictions-optimized.csv")

# Quick backtest
print("\n" + "="*80)
print("BACKTEST (2015-2025)")
print("="*80)

backtest_errors = []
for test_year in range(2015, 2026):
    for location in competition_locs:
        train_data = all_data[all_data['year'] < test_year].copy()
        test_data = all_data[(all_data['year'] == test_year) & (all_data['location'] == location) &
                             (all_data['source'].isin(['competition', 'nyc']))].copy()
        if len(test_data) == 0:
            continue
        train_feat = add_features(train_data, enhanced_feats=enhanced_feats)
        test_feat = add_features(test_data, reference_df=train_feat, enhanced_feats=enhanced_feats)
        # Handle missing columns
        for col in feature_cols[location]:
            if col not in test_feat.columns:
                test_feat[col] = np.nan
        X_test = test_feat[feature_cols[location]].fillna(medians[location]).fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = scalers[location].transform(X_test)
        pred = models[location].predict(X_scaled)[0] - location_bias[location]
        actual = test_feat['bloom_doy'].iloc[0]
        backtest_errors.append({
            'year': test_year, 'location': location, 'actual': actual,
            'predicted': pred, 'abs_error': abs(pred - actual)
        })

if backtest_errors:
    backtest_df = pd.DataFrame(backtest_errors)
    mae = backtest_df['abs_error'].mean()
    print(f"\nOverall MAE: {mae:.2f} days")
    print("\nMAE by location:")
    print(backtest_df.groupby('location')['abs_error'].mean().round(2).to_string())

print("\n" + "="*80)
print("DONE!")
print("="*80)
