#!/usr/bin/env python3
"""
Cherry Blossom Peak Bloom Prediction 2026 — Enhanced Python Pipeline (Simplified)

This version focuses on integrating the most impactful features from master dataset
while maintaining compatibility with the existing pipeline structure.

Strategy:
- Use original data loading (for proper train/test splits)
- Add enhanced features only where available
- Focus on features with good historical coverage
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(5103)

ROOT = Path('.')

# ============================================================================
# DATA LOADING (Original approach for compatibility)
# ============================================================================

print("="*80)
print("LOADING BLOOM DATA")
print("="*80)

competition_files = [
    ROOT / 'data/kyoto.csv',
    ROOT / 'data/washingtondc.csv',
    ROOT / 'data/liestal.csv',
    ROOT / 'data/vancouver.csv',
    ROOT / 'data/nyc.csv',
]

aux_files = [
    ROOT / 'data/japan.csv',
    ROOT / 'data/meteoswiss.csv',
    ROOT / 'data/south_korea.csv',
]

def read_bloom_file(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        'source': source,
        'location': df['location'].astype(str),
        'lat': pd.to_numeric(df['lat'], errors='coerce'),
        'long': pd.to_numeric(df['long'], errors='coerce'),
        'alt': pd.to_numeric(df['alt'], errors='coerce'),
        'year': pd.to_numeric(df['year'], errors='coerce').astype('Int64'),
        'bloom_doy': pd.to_numeric(df['bloom_doy'], errors='coerce')
    })

competition_raw = pd.concat([read_bloom_file(p, 'competition') for p in competition_files], ignore_index=True)
aux_raw = pd.concat([read_bloom_file(p, 'auxiliary') for p in aux_files], ignore_index=True)

# NYC enrichment
npn = pd.read_csv(ROOT / 'data/USA-NPN_status_intensity_observations_data.csv')
npn = npn[(npn['Site_ID'] == 32789) & (npn['Species_ID'] == 228) & (npn['Phenophase_ID'] == 501)].copy()
npn['Observation_Date'] = pd.to_datetime(npn['Observation_Date'], format='%m/%d/%y', errors='coerce')
npn['year'] = npn['Observation_Date'].dt.year
npn_yes = npn[npn['Phenophase_Status'] == 1].sort_values('Observation_Date').groupby('year', as_index=False).first()

nyc_npn_status = pd.DataFrame({
    'source': 'npn',
    'location': 'newyorkcity',
    'lat': 40.73040,
    'long': -73.99809,
    'alt': 8.5,
    'year': npn_yes['year'].astype('Int64'),
    'bloom_doy': pd.to_numeric(npn_yes['Day_of_Year'], errors='coerce')
}).dropna(subset=['year', 'bloom_doy'])

pheno = pd.read_csv(ROOT / 'data/USA-NPN_individual_phenometrics_data.csv')
pheno = pheno[(pheno['Site_ID'] == 32789) & (pheno['Species_ID'] == 228) & (pheno['Phenophase_ID'] == 501)].copy()
nyc_npn_pheno = (
    pheno.groupby('First_Yes_Year', as_index=False)['First_Yes_DOY']
    .min().rename(columns={'First_Yes_Year': 'year', 'First_Yes_DOY': 'bloom_doy'})
)
nyc_npn_pheno = nyc_npn_pheno[nyc_npn_pheno['bloom_doy'].notna()].copy()
nyc_npn_pheno['source'] = 'npn'
nyc_npn_pheno['location'] = 'newyorkcity'
nyc_npn_pheno['lat'] = 40.73040
nyc_npn_pheno['long'] = -73.99809
nyc_npn_pheno['alt'] = 8.5
nyc_npn_pheno['year'] = nyc_npn_pheno['year'].astype('Int64')

status_years = set(nyc_npn_status['year'].dropna().astype(int))
nyc_npn_pheno_new = nyc_npn_pheno[~nyc_npn_pheno['year'].astype(int).isin(status_years)]
nyc_npn = pd.concat([nyc_npn_status, nyc_npn_pheno_new], ignore_index=True)

existing_nyc_years = set(competition_raw.loc[competition_raw['location'] == 'newyorkcity', 'year'].dropna().astype(int))
nyc_npn = nyc_npn[~nyc_npn['year'].astype(int).isin(existing_nyc_years)]

competition = pd.concat([competition_raw, nyc_npn], ignore_index=True)
all_data = (
    pd.concat([competition, aux_raw], ignore_index=True)
    .dropna(subset=['year', 'bloom_doy'])
    .query('year >= 1880')
    .copy()
)
all_data['year'] = all_data['year'].astype(int)
all_data['site_id'] = all_data['source'] + '::' + all_data['location']

print(f"Loaded {len(all_data):,} bloom records from {all_data['location'].nunique()} locations")

# Load enhanced features from master dataset
print("\nLoading enhanced features from master dataset...")
master_features = pd.read_csv(ROOT / 'data/final/master_dataset.csv')

# Keep only the enhanced features we want to add
enhanced_cols = [
    'location', 'year',
    'T_mean_winter', 'T_mean_spring',
    'GDD_jan_feb', 'GDD_winter',
    'chill_hours_winter', 'chill_hours_nov_dec',
    'ONI_winter', 'NAO_winter', 'PDO_annual',
    'Hopkins_Index', 'photoperiod_mar20'
]

# Filter to only columns that exist
available_cols = [c for c in enhanced_cols if c in master_features.columns]
enhanced_features = master_features[available_cols].copy()

print(f"Available enhanced features: {len(available_cols) - 2}")  # minus location, year

competition_sites = sorted(competition_raw['location'].unique())
target_year = int(competition_raw['year'].max()) + 1

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(df: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None, enhanced_feats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Enhanced features with graceful fallback"""
    out = df.copy()
    ref = out if reference_df is None else reference_df.copy()

    # Ensure site_id exists
    if 'site_id' not in out.columns:
        out['site_id'] = out['source'] + '::' + out['location']
    if 'site_id' not in ref.columns:
        ref['site_id'] = ref['source'] + '::' + ref['location']

    # Merge with enhanced features if available
    if enhanced_feats is not None:
        out = out.merge(enhanced_feats, on=['location', 'year'], how='left')

    # Site observation count
    site_obs = ref.groupby('site_id').size().rename('site_obs').reset_index()
    out = out.merge(site_obs, on='site_id', how='left')
    out['site_obs'] = out['site_obs'].fillna(1)

    # Fill missing enhanced features with location/global medians
    feature_cols = [c for c in out.columns if c in [
        'T_mean_winter', 'T_mean_spring', 'GDD_jan_feb', 'GDD_winter',
        'chill_hours_winter', 'chill_hours_nov_dec',
        'ONI_winter', 'NAO_winter', 'PDO_annual'
    ]]

    for feat in feature_cols:
        if feat in out.columns:
            # Location median
            loc_med = ref.groupby('location')[feat].median() if feat in ref.columns else pd.Series()
            # Global median
            glob_med = ref[feat].median() if feat in ref.columns else 0

            if len(loc_med) > 0:
                out = out.merge(
                    loc_med.rename(f'{feat}_loc').reset_index(),
                    on='location',
                    how='left'
                )
                out[feat] = out[feat].fillna(out[f'{feat}_loc']).fillna(glob_med).fillna(0)
                out = out.drop(columns=[f'{feat}_loc'], errors='ignore')
            else:
                out[feat] = out[feat].fillna(glob_med).fillna(0)

    # Basic engineered features
    out['year_c'] = out['year'] - 1950
    out['year_c2'] = out['year_c'] ** 2
    out['lat_abs'] = out['lat'].abs()
    out['alt_log1p'] = np.log1p(np.clip(out['alt'], a_min=0, a_max=None))

    # Hopkins Index (if not already present)
    if 'Hopkins_Index' not in out.columns or out['Hopkins_Index'].isna().all():
        out['Hopkins_Index'] = 4 * out['lat'] - 1.25 * out['long'] + 4 * (out['alt'] / 122)

    return out

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_enhanced_model() -> Pipeline:
    """Enhanced model with meteorological features"""

    # Start with baseline features
    num_cols = [
        'year', 'year_c', 'year_c2',
        'lat', 'long', 'lat_abs', 'alt_log1p',
        'site_obs', 'Hopkins_Index'
    ]

    # Add enhanced features if they exist (will be filled with 0 if missing)
    potential_features = [
        'T_mean_winter', 'T_mean_spring',
        'GDD_jan_feb', 'GDD_winter',
        'chill_hours_winter', 'chill_hours_nov_dec',
        'ONI_winter', 'NAO_winter', 'PDO_annual',
        'photoperiod_mar20'
    ]

    # We'll add all potential features - imputer will handle missing values
    num_cols.extend(potential_features)

    cat_cols = ['source']

    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ],
        remainder='drop'
    )

    model = GradientBoostingRegressor(
        loss='huber',
        n_estimators=800,
        learning_rate=0.015,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=5103
    )

    return Pipeline([('pre', pre), ('model', model)])

def predict_local_trend(train_comp: pd.DataFrame, new_comp: pd.DataFrame) -> pd.DataFrame:
    """Local trend model - unchanged"""
    rows = []
    for loc in new_comp['location'].unique():
        tr = train_comp[train_comp['location'] == loc].sort_values('year').copy()
        nd = new_comp[new_comp['location'] == loc].copy()

        n = len(tr)
        if n >= 4:
            w = np.exp(np.arange(-n + 1, 1) / 6.0)
            coef = np.polyfit(tr['year'].values, tr['bloom_doy'].values, deg=2, w=w)
            pred = np.polyval(coef, nd['year'].values)
        elif n >= 2:
            coef = np.polyfit(tr['year'].values, tr['bloom_doy'].values, deg=1)
            pred = np.polyval(coef, nd['year'].values)
        else:
            pred = np.repeat(tr['bloom_doy'].mean(), len(nd))

        nd['pred_local'] = pred
        rows.append(nd[['location', 'year', 'pred_local']])

    return pd.concat(rows, ignore_index=True)

# ============================================================================
# ROLLING BACKTEST
# ============================================================================

print("\n" + "="*80)
print("RUNNING ROLLING BACKTEST")
print("="*80)

backtest_start = max(1900, int(competition['year'].min()) + 20)
backtest_years = list(range(backtest_start, int(competition['year'].max()) + 1))

print(f"Backtest: {backtest_start}-{backtest_years[-1]} ({len(backtest_years)} windows)")

rolling_rows = []

for y in backtest_years:
    train_comp = competition[competition['year'] < y].copy()
    test_comp = competition[competition['year'] == y].copy()

    if test_comp.empty or train_comp['location'].nunique() < len(competition_sites):
        continue

    train_all = all_data[all_data['year'] < y].copy()
    train_all = add_features(train_all, enhanced_feats=enhanced_features)

    test_feat = add_features(test_comp.copy(), reference_df=train_all, enhanced_feats=enhanced_features)

    local_pred = predict_local_trend(train_comp, test_feat)

    g_model = build_enhanced_model()
    g_model.fit(train_all, train_all['bloom_doy'])
    pred_g = g_model.predict(test_feat)

    fold = test_feat[['location', 'year', 'bloom_doy']].merge(local_pred, on=['location', 'year'], how='left')
    fold['pred_global'] = pred_g
    rolling_rows.append(fold)

rolling = pd.concat(rolling_rows, ignore_index=True)

mae_local = mean_absolute_error(rolling['bloom_doy'], rolling['pred_local'])
mae_global = mean_absolute_error(rolling['bloom_doy'], rolling['pred_global'])

print(f"\nBacktest MAE:")
print(f"  Local:  {mae_local:.2f} days")
print(f"  Global: {mae_global:.2f} days")

# ============================================================================
# ENSEMBLE WEIGHTING
# ============================================================================

print("\nOptimizing ensemble weights...")

grid = np.arange(0.0, 1.0001, 0.02)
def mae_w(df, w):
    pred = w * df['pred_local'] + (1.0 - w) * df['pred_global']
    return mean_absolute_error(df['bloom_doy'], pred)

w_local_global = min(grid, key=lambda w: mae_w(rolling, w))
w_global_global = 1.0 - w_local_global

site_w = []
for loc, df_loc in rolling.groupby('location'):
    w_star = min(grid, key=lambda w: mae_w(df_loc, w))
    site_w.append({'location': loc, 'w_local': w_star, 'w_global': 1.0 - w_star})
site_w_df = pd.DataFrame(site_w)

rolling = rolling.merge(site_w_df, on='location', how='left')
rolling['w_local'] = rolling['w_local'].fillna(w_local_global)
rolling['w_global'] = rolling['w_global'].fillna(w_global_global)

rolling['pred_ensemble'] = rolling['w_local'] * rolling['pred_local'] + rolling['w_global'] * rolling['pred_global']
rolling['abs_err'] = (rolling['bloom_doy'] - rolling['pred_ensemble']).abs()

mae_ensemble = mean_absolute_error(rolling['bloom_doy'], rolling['pred_ensemble'])

print(f"  Ensemble: {mae_ensemble:.2f} days ✨")

# Calculate intervals
site_q90 = rolling.groupby('location', as_index=False)['abs_err'].quantile(0.90).rename(columns={'abs_err': 'q90'})
global_q90 = rolling['abs_err'].quantile(0.90)

# ============================================================================
# 2026 PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print(f"GENERATING {target_year} PREDICTIONS")
print("="*80)

train_all = add_features(all_data.copy(), enhanced_feats=enhanced_features)
train_comp = competition.copy()

newdata = (
    competition.sort_values('year')
    .groupby('location', as_index=False)
    .tail(1)[['location', 'lat', 'long', 'alt', 'source']]
    .copy()
)
newdata['site_id'] = newdata['source'] + '::' + newdata['location']
newdata['year'] = target_year
newdata['bloom_doy'] = np.nan

new_feat = add_features(newdata, reference_df=train_all, enhanced_feats=enhanced_features)

local_pred = predict_local_trend(train_comp, new_feat)

global_model = build_enhanced_model()
global_model.fit(train_all, train_all['bloom_doy'])
pred_global = global_model.predict(new_feat)

final_pred = new_feat[['location', 'year']].merge(local_pred, on=['location', 'year'], how='left')
final_pred['pred_global'] = pred_global
final_pred = final_pred.merge(site_w_df, on='location', how='left')
final_pred['w_local'] = final_pred['w_local'].fillna(w_local_global)
final_pred['w_global'] = final_pred['w_global'].fillna(w_global_global)

final_pred['prediction_raw'] = final_pred['w_local'] * final_pred['pred_local'] + final_pred['w_global'] * final_pred['pred_global']
final_pred = final_pred.merge(site_q90, on='location', how='left')
final_pred['q90'] = final_pred['q90'].fillna(global_q90)

final_pred['prediction'] = np.clip(np.round(final_pred['prediction_raw']), 1, 366).astype(int)
final_pred['lower'] = np.clip(np.floor(final_pred['prediction_raw'] - final_pred['q90']), 1, 366).astype(int)
final_pred['upper'] = np.clip(np.ceil(final_pred['prediction_raw'] + final_pred['q90']), 1, 366).astype(int)

final_pred = final_pred[['location', 'year', 'prediction', 'lower', 'upper']].sort_values('location')

print(f"\n{target_year} Predictions:")
print(final_pred.to_string(index=False))

# Save
submission = final_pred.copy()
submission[['location', 'prediction', 'lower', 'upper']].to_csv('cherry-predictions-enhanced.csv', index=False)

print("\n" + "="*80)
print(f"✅ Enhanced model MAE: {mae_ensemble:.2f} days")
print(f"✅ Saved: cherry-predictions-enhanced.csv")
print("="*80)
